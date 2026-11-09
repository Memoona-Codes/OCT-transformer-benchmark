"""
Computes the full Table 3 metric set (Accuracy, AUC, Precision, Recall,
Specificity, F1 -- mean +/- std across the 3 outer folds) for any model
that has saved roc_data/{model}_fold*.npz files (y_true, y_proba per fold).

This does NOT retrain anything -- it reads the predictions DeiT/MAE already
saved during their normal run and computes the metrics your other four
models (Retformer/ViT/Swin/KAN) already have in Table 3, so DeiT/MAE can be
added to that same table on equal footing.

Usage:
    python3 compute_full_metrics.py deit
    python3 compute_full_metrics.py mae
"""

import sys
import glob
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)

ROC_DATA_DIR = "roc_data"


def compute_fold_metrics(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba > threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)  # == sensitivity
    f1 = f1_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": acc, "auc": auc, "precision": precision,
        "recall": recall, "specificity": specificity, "f1": f1,
    }


def main(model_key):
    pattern = f"{ROC_DATA_DIR}/{model_key}_fold*.npz"
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files found matching {pattern}")
        print("Check the model has completed a full run and roc_data/ exists.")
        sys.exit(1)

    print(f"Found {len(files)} fold file(s): {files}\n")

    per_fold_metrics = []
    for f in files:
        data = np.load(f)
        y_true, y_proba = data["y_true"], data["y_proba"]
        metrics = compute_fold_metrics(y_true, y_proba)
        per_fold_metrics.append(metrics)
        print(f"{f}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    print(f"\n=== {model_key.upper()} -- mean +/- std across {len(files)} folds ===")
    keys = per_fold_metrics[0].keys()
    for k in keys:
        vals = [m[k] for m in per_fold_metrics]
        mean, std = np.mean(vals), np.std(vals)
        print(f"{k}: {mean*100:.2f}% +/- {std*100:.2f}%")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("deit", "mae"):
        print("Usage: python3 compute_full_metrics.py [deit|mae]")
        sys.exit(1)
    main(sys.argv[1])
