"""
Evaluation script for the trained ISAC CNN.

Produces five outputs from the held-out test set:
    RQ1_AI_confusion_matrix.png      presence detection confusion matrix
    RQ2_ROC_AI_vs_CFAR.png           ROC overlay: AI vs CA-CFAR
    RQ2_Pd_vs_SNR.png                Pd vs sensing SNR
    RQ2_size_confusion_matrix.png    4-class size classification matrix
    RQ2_size_acc_vs_SNR.png          size accuracy vs sensing SNR
    per_scenario_scores.csv          per-scenario CNN scores
    summary.csv                      headline metrics

Usage:
    python evaluate.py --manifest <path_to_manifest.csv>
                       [--checkpoint checkpoints/best_model.pt]
                       [--out_dir results]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc

from dataset import ISACDataset
from model   import ISACDetectorCNN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",   required=True)
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--out_dir",    default="results")
    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()


def collect_predictions(model, loader, device):
    """Run model on loader; return scores and labels as numpy arrays."""
    model.eval()
    pres_scores, size_preds, pres_true, size_true = [], [], [], []
    with torch.no_grad():
        for x, y_pres, y_size in loader:
            p_logit, s_logits = model(x.to(device))
            pres_scores.append(torch.sigmoid(p_logit).cpu().numpy())
            size_preds.append(s_logits.argmax(dim=-1).cpu().numpy())
            pres_true.append(y_pres.numpy())
            size_true.append(y_size.numpy())
    return {
        "pres_score": np.concatenate(pres_scores),
        "size_pred":  np.concatenate(size_preds),
        "pres_true":  np.concatenate(pres_true).astype(int),
        "size_true":  np.concatenate(size_true),
    }


def plot_confusion(cm, labels, title, fname, out_dir):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=140)
    plt.close()
    print(f"  saved {fname}")


def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading test set...")
    test_ds     = ISACDataset(args.manifest, split="test", augment=False)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False)
    print(f"  {len(test_ds)} test scenarios")

    print(f"\nLoading checkpoint {args.checkpoint}...")
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = ISACDetectorCNN().to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"  trained for {ckpt['epoch']} epochs, val_loss={ckpt['val_loss']:.4f}")

    print("\nRunning predictions...")
    pred = collect_predictions(model, test_loader, device)

    df_manifest = pd.read_csv(args.manifest)
    df_test     = df_manifest[df_manifest["split"] == "test"].reset_index(drop=True)
    assert len(df_test) == len(pred["pres_true"]), \
        "Manifest test rows do not match prediction count"

    # Per-scenario scores
    df_scores = df_test.copy()
    df_scores["pres_score_cnn"] = pred["pres_score"]
    df_scores["pres_pred_cnn"]  = (pred["pres_score"] >= 0.5).astype(int)
    df_scores["pres_true"]      = pred["pres_true"]
    df_scores["size_pred_cnn"]  = pred["size_pred"]
    df_scores["size_true"]      = pred["size_true"]
    df_scores["correct_pres"]   = (df_scores["pres_pred_cnn"] == df_scores["pres_true"]).astype(int)
    df_scores.to_csv(out_dir / "per_scenario_scores.csv", index=False)
    print("  saved per_scenario_scores.csv")

    # RQ1: presence confusion matrix
    print("\nRQ1 - presence detection")
    pres_pred = (pred["pres_score"] >= 0.5).astype(int)
    pres_acc  = (pres_pred == pred["pres_true"]).mean()
    cm_pres   = confusion_matrix(pred["pres_true"], pres_pred, labels=[0, 1])
    print(f"  accuracy: {pres_acc:.4f}")
    plot_confusion(cm_pres, ["absent", "present"],
                   f"RQ1 AI presence detection (acc={pres_acc:.3f})",
                   "RQ1_AI_confusion_matrix.png", out_dir)

    # RQ2: ROC overlay
    print("\nRQ2 - ROC curve")
    fpr_ai, tpr_ai, _ = roc_curve(pred["pres_true"], pred["pres_score"])
    auc_ai = auc(fpr_ai, tpr_ai)

    cfar_pred = df_test["CFAR_present"].astype(int).values
    cfar_true = pred["pres_true"]
    if cfar_true.sum() > 0 and (1 - cfar_true).sum() > 0:
        cfar_tpr = ((cfar_pred == 1) & (cfar_true == 1)).sum() / cfar_true.sum()
        cfar_fpr = ((cfar_pred == 1) & (cfar_true == 0)).sum() / (1 - cfar_true).sum()
    else:
        cfar_tpr = cfar_fpr = float("nan")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_ai, tpr_ai, "-", lw=2, color="C0", label=f"AI (AUC={auc_ai:.3f})")
    if not np.isnan(cfar_tpr):
        ax.plot(cfar_fpr, cfar_tpr, "s", markersize=12, color="C3",
                label=f"CA-CFAR (Pd={cfar_tpr:.2f}, Pfa={cfar_fpr:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="random")
    ax.set_xlabel("False positive rate (P_fa)")
    ax.set_ylabel("True positive rate (P_d)")
    ax.set_title("RQ2 ROC: AI vs CA-CFAR")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.grid(True); ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "RQ2_ROC_AI_vs_CFAR.png", dpi=140)
    plt.close()
    print(f"  AI AUC={auc_ai:.3f}  saved RQ2_ROC_AI_vs_CFAR.png")

    # RQ2: Pd vs SNR
    print("\nRQ2 - Pd vs SNR")
    snr_levels = sorted(df_test["SNR_sense_dB"].unique())
    pd_ai, pd_cfar = [], []
    for snr in snr_levels:
        mask = (df_test["SNR_sense_dB"] == snr).values & (cfar_true == 1)
        if mask.sum() > 0:
            pd_ai.append(pres_pred[mask].mean())
            pd_cfar.append(cfar_pred[mask].mean())
        else:
            pd_ai.append(np.nan); pd_cfar.append(np.nan)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(snr_levels, pd_ai,   "o-", lw=2, color="C0", label="AI")
    ax.plot(snr_levels, pd_cfar, "s-", lw=2, color="C3", label="CA-CFAR")
    ax.set_xlabel("SNR_sense (dB)"); ax.set_ylabel("P_d")
    ax.set_title("RQ2 P_d vs SNR")
    ax.set_ylim(0, 1.05); ax.grid(True); ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "RQ2_Pd_vs_SNR.png", dpi=140)
    plt.close()
    print("  saved RQ2_Pd_vs_SNR.png")

    # RQ2-ext: size confusion matrix
    print("\nRQ2-ext - size classification")
    size_acc = (pred["size_pred"] == pred["size_true"]).mean()
    cm_size  = confusion_matrix(pred["size_true"], pred["size_pred"], labels=[0,1,2,3])
    print(f"  size accuracy: {size_acc:.4f}")
    plot_confusion(cm_size, ["absent", "small", "medium", "large"],
                   f"RQ2-ext size classification (acc={size_acc:.3f})",
                   "RQ2_size_confusion_matrix.png", out_dir)

    # RQ2-ext: size accuracy vs SNR
    size_acc_snr = []
    for snr in snr_levels:
        mask = (df_test["SNR_sense_dB"] == snr).values
        size_acc_snr.append(
            (pred["size_pred"][mask] == pred["size_true"][mask]).mean()
            if mask.sum() > 0 else np.nan)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(snr_levels, size_acc_snr, "o-", lw=2, color="C2", label="AI size accuracy")
    ax.set_xlabel("SNR_sense (dB)"); ax.set_ylabel("4-class size accuracy")
    ax.set_title("RQ2-ext size accuracy vs SNR")
    ax.set_ylim(0, 1.05); ax.grid(True); ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "RQ2_size_acc_vs_SNR.png", dpi=140)
    plt.close()
    print("  saved RQ2_size_acc_vs_SNR.png")

    # Summary
    summary = {
        "n_test":      len(pred["pres_true"]),
        "AI_pres_acc": pres_acc,
        "AI_AUC":      auc_ai,
        "CFAR_Pd":     cfar_tpr,
        "CFAR_Pfa":    cfar_fpr,
        "AI_size_acc": size_acc,
    }
    print("\n" + "=" * 50)
    print("Test-set summary")
    print("=" * 50)
    for k, v in summary.items():
        print(f"  {k:20s}  {v}")
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)
    print(f"\nResults written to {out_dir}/")


if __name__ == "__main__":
    main()