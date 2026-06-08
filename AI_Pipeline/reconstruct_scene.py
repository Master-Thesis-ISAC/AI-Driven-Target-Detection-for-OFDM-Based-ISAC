"""
Hybrid scene reconstruction from RD maps.

Combines two sources:
  1. Trained CNN  ->  binary presence + scene-level size label
  2. CFAR peaks   ->  (range, velocity) of each detected target

All detected targets in a scene share the same CNN size label because
the model was trained to predict one label per RD map, not per target.

Usage:
    # Single scenario
    python reconstruct_scene.py --manifest <path>/manifest.csv \
                                --checkpoint checkpoints/best_model.pt \
                                --scenario_idx 42

    # Random selection from test split (default 6)
    python reconstruct_scene.py --manifest <path>/manifest.csv \
                                --checkpoint checkpoints/best_model.pt \
                                --num_examples 6
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dataset import normalise_rdmap_db
from model   import ISACDetectorCNN

SIZE_NAMES   = ["absent", "small", "medium", "large"]
SIZE_COLOURS = {0: "gray", 1: "g",  2: "b",  3: "r"}
SIZE_MARKERS = {0: ".",    1: "o",  2: "s",  3: "^"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",     required=True)
    p.add_argument("--checkpoint",   default="checkpoints/best_model.pt")
    p.add_argument("--out_dir",      default="reconstructed")
    p.add_argument("--scenario_idx", type=int, default=None)
    p.add_argument("--num_examples", type=int, default=6)
    p.add_argument("--seed",         type=int, default=0)
    return p.parse_args()


def load_full_mat(path: str) -> dict:
    """Load all fields needed for reconstruction from a scenario .mat file."""
    with h5py.File(path, "r") as f:
        rd_db = np.array(f["rdMap_dB"]).astype(np.float32)
        if rd_db.shape != (64, 128):
            rd_db = rd_db.T

        out = {
            "rdMap_dB":         rd_db,
            "targetPresent":    int(np.array(f["targetPresent"]).item()),
            "sceneLargestSize": int(np.array(f["sceneLargestSize"]).item()),
            "rangeAxis":        np.array(f["rangeAxis"]).astype(float).ravel(),
            "velocityAxis":     np.array(f["velocityAxis"]).astype(float).ravel(),
            "targetRange_m":    np.array(f["targetRange_m"]).ravel()   if "targetRange_m" in f else np.array([]),
            "targetVel_mps":    np.array(f["targetVel_mps"]).ravel()   if "targetVel_mps" in f else np.array([]),
            "targetSizeIdx":    np.array(f["targetSizeIdx"]).ravel().astype(int) if "targetSizeIdx" in f else np.array([], dtype=int),
        }

        # CFAR peaks: [N x 2] (rangeBin, dopplerBin), 1-based in MATLAB
        if "classicalCFAR_peaks" in f:
            peaks = np.atleast_2d(np.array(f["classicalCFAR_peaks"])).astype(int)
            if peaks.size == 0:
                peaks = np.zeros((0, 2), dtype=int)
            else:
                if peaks.shape[1] != 2 and peaks.shape[0] == 2:
                    peaks = peaks.T
                if peaks.shape[1] == 2:
                    peaks = np.clip(peaks - 1, 0, [127, 63])  # 1-based -> 0-based
                else:
                    peaks = np.zeros((0, 2), dtype=int)
            out["cfarPeaks_rdBin"] = peaks
        else:
            out["cfarPeaks_rdBin"] = np.zeros((0, 2), dtype=int)

    return out


def cluster_cfar_peaks(peaks_rd: np.ndarray, min_distance: int = 3) -> np.ndarray:
    """
    Merge nearby CFAR bins into one centroid per target.
    Uses greedy Chebyshev-distance grouping.
    """
    if len(peaks_rd) == 0:
        return peaks_rd
    peaks_rd = np.atleast_2d(peaks_rd).astype(int)
    if peaks_rd.shape[1] != 2:
        peaks_rd = peaks_rd.T
    if peaks_rd.shape[1] != 2:
        return np.zeros((0, 2), dtype=int)

    used      = np.zeros(len(peaks_rd), dtype=bool)
    centroids = []
    for i in range(len(peaks_rd)):
        if used[i]:
            continue
        d_r     = np.abs(peaks_rd[:, 0] - peaks_rd[i, 0])
        d_d     = np.abs(peaks_rd[:, 1] - peaks_rd[i, 1])
        cluster = (d_r <= min_distance) & (d_d <= min_distance) & ~used
        used   |= cluster
        centroids.append([int(round(peaks_rd[cluster, 0].mean())),
                          int(round(peaks_rd[cluster, 1].mean()))])
    return np.array(centroids, dtype=int)


def predict_scene(model, full: dict, device) -> dict:
    """
    Run the CNN on the full RD map and combine with CFAR centroids.
    Returns one presence decision and one scene-level size label.
    All CFAR-detected targets share the scene-level size label.
    """
    x = torch.from_numpy(normalise_rdmap_db(full["rdMap_dB"])) \
            .unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        p_logit, s_logits = model(x)

    pres_prob  = float(torch.sigmoid(p_logit).item())
    pres_label = int(pres_prob >= 0.5)
    scene_size = int(s_logits.argmax(dim=-1).item())

    centroids = cluster_cfar_peaks(full["cfarPeaks_rdBin"])
    targets   = []
    if pres_label == 1 and len(centroids) > 0:
        assigned_size = scene_size if scene_size > 0 else 1
        for rb, db in centroids:
            rb = int(np.clip(rb, 0, len(full["rangeAxis"])    - 1))
            db = int(np.clip(db, 0, len(full["velocityAxis"]) - 1))
            targets.append({
                "range_m":   float(full["rangeAxis"][rb]),
                "vel_mps":   float(full["velocityAxis"][db]),
                "size_idx":  assigned_size,
                "size_name": SIZE_NAMES[assigned_size],
                "rangeBin":  rb,
                "doppBin":   db,
            })

    return {
        "presence_prob":   pres_prob,
        "presence_label":  pres_label,
        "scene_size":      scene_size,
        "scene_size_name": SIZE_NAMES[scene_size],
        "targets":         targets,
    }


def plot_reconstruction(full: dict, prediction: dict, fname: Path, scenario_idx: int) -> None:
    """4-panel reconstruction figure: layout, RD map, range stems, velocity stems."""
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    # Panel 1: scene layout
    ax = axs[0]
    ax.plot(0, 0,   "ks", markersize=12, markerfacecolor="k", label="gNB")
    ax.plot(30, 100, "b^", markersize=12, label="UE")
    ax.plot([0, 30], [0, 100], "b:", lw=1)

    added = set()
    for t in prediction["targets"]:
        c   = SIZE_COLOURS[t["size_idx"]]
        m   = SIZE_MARKERS[t["size_idx"]]
        lbl = f"AI scene size: {t['size_name']}"
        ax.plot(0, t["range_m"], m, color=c, markersize=12, mew=2,
                label=lbl if lbl not in added else None)
        added.add(lbl)

    for i, R in enumerate(full["targetRange_m"]):
        s = int(full["targetSizeIdx"][i]) if i < len(full["targetSizeIdx"]) else 0
        ax.plot(0, R, "x", color=SIZE_COLOURS.get(s,"gray"), alpha=0.4,
                markersize=14, mew=2, label="ground truth" if i == 0 else None)

    ax.set_xlim(-50, 50); ax.set_ylim(0, 200)
    ax.set_xlabel("cross-range (m)"); ax.set_ylabel("range (m)")
    ax.set_title(f"Scene #{scenario_idx}\n"
                 f"AI: presence={prediction['presence_label']} "
                 f"(p={prediction['presence_prob']:.2f}), "
                 f"size={prediction['scene_size_name']}")
    ax.grid(True); ax.legend(loc="upper right", fontsize=8)

    # Panel 2: RD map
    ax    = axs[1]
    rd_db = full["rdMap_dB"]
    im = ax.imshow(rd_db.T, aspect="auto", origin="lower",
                   extent=[full["velocityAxis"][0], full["velocityAxis"][-1],
                            full["rangeAxis"][0],    full["rangeAxis"][-1]],
                   cmap="jet", vmin=rd_db.max()-50, vmax=rd_db.max(),
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Magnitude (dB)")

    for t in prediction["targets"]:
        ax.plot(t["vel_mps"], t["range_m"], "wo",
                markersize=12, markerfacecolor="none", mew=2)
    for i, R in enumerate(full["targetRange_m"]):
        ax.plot(full["targetVel_mps"][i], R, "wx", markersize=10, mew=2, alpha=0.7,
                label="ground truth" if i == 0 else None)

    ax.set_xlabel("Radial velocity (m/s)"); ax.set_ylabel("Range (m)")
    ax.set_title("RD map\n(circles = reconstructed, x = ground truth)")
    ax.set_xlim(full["velocityAxis"][0], full["velocityAxis"][-1])
    ax.set_ylim(full["rangeAxis"][0],    full["rangeAxis"][-1])

    # Panel 3: range stems
    ax = axs[2]
    for t in prediction["targets"]:
        ax.stem([t["range_m"]], [t["size_idx"]],
                linefmt=SIZE_COLOURS[t["size_idx"]] + "-",
                markerfmt=SIZE_COLOURS[t["size_idx"]] + SIZE_MARKERS[t["size_idx"]],
                basefmt="k-")
    for r in full["targetRange_m"]:
        ax.axvline(r, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlim(0, 200); ax.set_ylim(-0.5, 3.5)
    ax.set_yticks([0,1,2,3]); ax.set_yticklabels(SIZE_NAMES)
    ax.set_xlabel("range (m)"); ax.set_ylabel("scene-level size (AI)")
    ax.set_title("Range stem"); ax.grid(True)

    # Panel 4: velocity stems
    ax = axs[3]
    for t in prediction["targets"]:
        ax.stem([t["vel_mps"]], [t["size_idx"]],
                linefmt=SIZE_COLOURS[t["size_idx"]] + "-",
                markerfmt=SIZE_COLOURS[t["size_idx"]] + SIZE_MARKERS[t["size_idx"]],
                basefmt="k-")
    for v in full["targetVel_mps"]:
        ax.axvline(v, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_yticks([0,1,2,3]); ax.set_yticklabels(SIZE_NAMES)
    ax.set_xlabel("velocity (m/s)"); ax.set_ylabel("scene-level size (AI)")
    ax.set_title("Velocity stem"); ax.grid(True)

    plt.tight_layout()
    plt.savefig(fname, dpi=140)
    plt.close()


def print_scene_description(prediction: dict, full: dict, idx: int) -> None:
    print(f"\n=== Scenario {idx} ===")
    print(f"  AI presence:   {prediction['presence_label']} (prob {prediction['presence_prob']:.2f})")
    print(f"  AI scene size: {prediction['scene_size_name']}")
    print(f"  Ground truth:  presence={full['targetPresent']}, "
          f"largest size={SIZE_NAMES[full['sceneLargestSize']]}")
    if prediction["targets"]:
        print(f"  {len(prediction['targets'])} detected target(s):")
        for j, t in enumerate(prediction["targets"], 1):
            print(f"    {j}. range={t['range_m']:6.1f} m  "
                  f"velocity={t['vel_mps']:+6.1f} m/s  size={t['size_name']}")
    else:
        print("  No targets detected")
    if len(full["targetRange_m"]) > 0:
        print(f"  Ground-truth targets ({len(full['targetRange_m'])}):")
        for j in range(len(full["targetRange_m"])):
            s = full["targetSizeIdx"][j] if j < len(full["targetSizeIdx"]) else 0
            print(f"    {j+1}. range={full['targetRange_m'][j]:6.1f} m  "
                  f"velocity={full['targetVel_mps'][j]:+6.1f} m/s  "
                  f"size={SIZE_NAMES[int(s)]}")


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {args.checkpoint}...")
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = ISACDetectorCNN().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    df = pd.read_csv(args.manifest)
    if args.scenario_idx is not None:
        rows = df[df["idx"] == args.scenario_idx]
        if len(rows) == 0:
            raise ValueError(f"idx {args.scenario_idx} not in manifest")
    else:
        df_test = df[df["split"] == "test"] if "split" in df.columns else df
        rows    = df_test.sample(min(args.num_examples, len(df_test)),
                                 random_state=args.seed)

    print(f"\nReconstructing {len(rows)} scenario(s)...")
    for _, row in rows.iterrows():
        idx  = int(row["idx"])
        path = row["file"]
        if not Path(path).exists():
            path = Path(args.manifest).parent / Path(path).name

        full       = load_full_mat(str(path))
        prediction = predict_scene(model, full, device)
        print_scene_description(prediction, full, idx)

        out_fname = out_dir / f"reconstructed_{idx:04d}.png"
        plot_reconstruction(full, prediction, out_fname, idx)
        print(f"  saved -> {out_fname}")


if __name__ == "__main__":
    main()
