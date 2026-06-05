"""
ISAC dataset loader for AI training.

Reads scenario .mat files produced by GenerateISACData_Main.
Each scenario yields one (image, presence_label, size_label) sample.

Input:  rdMap_dB  [64 x 128] single-channel RD map in dB
Labels: targetPresent    0/1 binary presence
        sceneLargestSize 0/1/2/3  absent / small / medium / large
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# RD map normalisation: dB values span roughly [-30, +60] dB
# shifted and scaled to roughly [-1, +1] for stable training.
RDMAP_OFFSET_DB = 15.0
RDMAP_SCALE_DB  = 30.0


def normalise_rdmap_db(rd_db: np.ndarray) -> np.ndarray:
    """Centre and scale the dB-domain RD map."""
    return (rd_db - RDMAP_OFFSET_DB) / RDMAP_SCALE_DB


def load_scenario_mat(mat_path: str) -> dict:
    """Read an HDF5/v7.3 scenario .mat file and return the fields needed for training."""
    with h5py.File(mat_path, "r") as f:
        rd_db    = np.array(f["rdMap_dB"]).astype(np.float32)
        present  = int(np.array(f["targetPresent"]).item())
        size_lbl = int(np.array(f["sceneLargestSize"]).item())

        # MATLAB stores arrays transposed in HDF5
        if rd_db.shape != (64, 128):
            rd_db = rd_db.T
        if rd_db.shape != (64, 128):
            raise ValueError(f"Unexpected rdMap_dB shape {rd_db.shape} in {mat_path}")

    return {
        "rdMap_dB":         rd_db,
        "targetPresent":    present,
        "sceneLargestSize": size_lbl,
        "path":             mat_path,
    }


class ISACDataset(Dataset):
    """
    PyTorch Dataset for ISAC RD-map scenarios.

    Parameters
    ----------
    manifest_csv : str
        Path to manifest.csv from GenerateISACData_Main.
    split : {'train', 'val', 'test'}
    augment : bool
        Doppler flip, gain perturbation, and additive noise.
        Applied only when split == 'train'.
    """

    def __init__(self, manifest_csv: str, split: str = "train", augment: bool = False):
        self.manifest_csv = Path(manifest_csv)
        self.split        = split
        self.augment      = augment and (split == "train")

        df = pd.read_csv(self.manifest_csv)
        if "split" not in df.columns:
            raise ValueError("manifest.csv missing 'split' column")
        df = df[df["split"] == split].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"No scenarios in split='{split}'")

        self.df           = df
        self.manifest_dir = self.manifest_csv.parent

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, raw: str) -> str:
        """Handle absolute and manifest-relative file paths."""
        p = Path(raw)
        if p.is_absolute() and p.exists():
            return str(p)
        for base in [self.manifest_dir, self.manifest_dir.parent]:
            cand = base / p.name
            if cand.exists():
                return str(cand)
        return raw

    def _augment(self, rd: np.ndarray) -> np.ndarray:
        """Training-time augmentation on a (64, 128) RD map."""
        if np.random.rand() < 0.5:
            rd = rd[::-1, :].copy()                              # Doppler flip
        rd = rd + np.random.uniform(-2.0, 2.0)                  # gain perturbation
        rd = rd + np.random.normal(0, 0.5, rd.shape).astype(np.float32)  # noise
        return rd

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row    = self.df.iloc[idx]
        path   = self._resolve_path(row["file"])
        sample = load_scenario_mat(path)

        rd = sample["rdMap_dB"].astype(np.float32)
        if self.augment:
            rd = self._augment(rd)
        rd = normalise_rdmap_db(rd)

        image    = torch.from_numpy(rd).unsqueeze(0).float()    # (1, 64, 128)
        present  = torch.tensor(sample["targetPresent"],    dtype=torch.float32)
        size_lbl = torch.tensor(sample["sceneLargestSize"], dtype=torch.long)
        return image, present, size_lbl


def make_loaders(manifest_csv: str, batch_size: int = 64, num_workers: int = 0) -> dict:
    """Build train/val/test DataLoaders from manifest.csv."""
    loaders = {}
    for split, augment, shuffle in [
        ("train", True,  True),
        ("val",   False, False),
        ("test",  False, False),
    ]:
        try:
            ds = ISACDataset(manifest_csv, split=split, augment=augment)
        except ValueError:
            print(f"  [warn] split='{split}' has no scenarios - skipped")
            continue
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,
        )
        print(f"  {split}: {len(ds)} scenarios")
    return loaders


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python dataset.py <path-to-manifest.csv>")
        sys.exit(1)
    loaders = make_loaders(sys.argv[1], batch_size=4)
    if "train" in loaders:
        x, y_pres, y_size = next(iter(loaders["train"]))
        print(f"image shape: {x.shape}  range [{x.min():.2f}, {x.max():.2f}]")
        print(f"presence:    {y_pres.tolist()}")
        print(f"size:        {y_size.tolist()}")
