"""
Training script for the ISAC two-head CNN.

Usage:
    python train.py --manifest <path_to_manifest.csv>
                    [--epochs 30] [--batch_size 64]
                    [--lr 1e-3] [--out_dir checkpoints]

Saves best_model.pt (by val loss) and training_log.csv to out_dir.
Early stops if val loss does not improve for `patience` epochs.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import make_loaders
from model   import ISACDetectorCNN, count_parameters


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",    required=True)
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--alpha_size",  type=float, default=0.5,
                   help="Weight on the size-classification loss term")
    p.add_argument("--patience",    type=int,   default=8)
    p.add_argument("--out_dir",     default="checkpoints")
    p.add_argument("--num_workers", type=int,   default=0)
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


def evaluate(model, loader, device, alpha_size):
    """Return average loss and accuracies on a DataLoader."""
    model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    ce  = nn.CrossEntropyLoss(reduction="sum")
    total_loss = pres_correct = size_correct = n = 0

    with torch.no_grad():
        for x, y_pres, y_size in loader:
            x, y_pres, y_size = x.to(device), y_pres.to(device), y_size.to(device)
            p_logit, s_logits = model(x)
            loss = bce(p_logit, y_pres) + alpha_size * ce(s_logits, y_size)
            total_loss   += loss.item()
            n            += x.size(0)
            pres_correct += ((torch.sigmoid(p_logit) >= 0.5).float() == y_pres).sum().item()
            size_correct += (s_logits.argmax(dim=-1) == y_size).sum().item()

    return {
        "loss":     total_loss / max(n, 1),
        "pres_acc": pres_correct / max(n, 1),
        "size_acc": size_correct / max(n, 1),
        "n":        n,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading data...")
    loaders = make_loaders(args.manifest, batch_size=args.batch_size,
                           num_workers=args.num_workers)
    if "train" not in loaders or "val" not in loaders:
        raise RuntimeError("Need both train and val splits in manifest.csv")

    model = ISACDetectorCNN().to(device)
    print(f"\nModel parameters: {count_parameters(model):,}")

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    bce = nn.BCEWithLogitsLoss()
    ce  = nn.CrossEntropyLoss()

    log_path = out_dir / "training_log.csv"
    log_file = open(log_path, "w", newline="")
    writer   = csv.writer(log_file)
    writer.writerow(["epoch","train_loss","val_loss",
                     "train_pres_acc","train_size_acc",
                     "val_pres_acc","val_size_acc","lr"])

    best_val_loss          = float("inf")
    epochs_without_improve = 0

    print(f"\nTraining up to {args.epochs} epochs  (patience={args.patience})\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        run_loss = pres_correct = size_correct = n = 0

        for x, y_pres, y_size in loaders["train"]:
            x, y_pres, y_size = x.to(device), y_pres.to(device), y_size.to(device)
            optimizer.zero_grad()
            p_logit, s_logits = model(x)
            loss = bce(p_logit, y_pres) + args.alpha_size * ce(s_logits, y_size)
            loss.backward()
            optimizer.step()
            run_loss     += loss.item() * x.size(0)
            n            += x.size(0)
            pres_correct += ((torch.sigmoid(p_logit) >= 0.5).float() == y_pres).sum().item()
            size_correct += (s_logits.argmax(dim=-1) == y_size).sum().item()

        tr = {"loss": run_loss/max(n,1), "pres_acc": pres_correct/max(n,1),
              "size_acc": size_correct/max(n,1)}
        vl = evaluate(model, loaders["val"], device, args.alpha_size)
        scheduler.step(vl["loss"])
        cur_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:3d}/{args.epochs} ({time.time()-t0:5.1f}s) | "
              f"train loss={tr['loss']:.4f} pres={tr['pres_acc']:.3f} size={tr['size_acc']:.3f} | "
              f"val loss={vl['loss']:.4f} pres={vl['pres_acc']:.3f} size={vl['size_acc']:.3f} | "
              f"lr={cur_lr:.2e}")

        writer.writerow([epoch, tr["loss"], vl["loss"],
                         tr["pres_acc"], tr["size_acc"],
                         vl["pres_acc"], vl["size_acc"], cur_lr])
        log_file.flush()

        if vl["loss"] < best_val_loss - 1e-4:
            best_val_loss          = vl["loss"]
            epochs_without_improve = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "val_loss": best_val_loss, "args": vars(args)},
                       out_dir / "best_model.pt")
            print(f"  -> saved best model (val_loss={best_val_loss:.4f})")
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    log_file.close()
    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {out_dir / 'best_model.pt'}")
    print(f"Log:        {log_path}")


if __name__ == "__main__":
    main()