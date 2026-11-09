"""
ROC curve overlay -- single figure, all models.

Run this AFTER model training scripts have produced their roc_data/*.npz
files (y_true + y_proba per outer fold). This script does NOT retrain
anything -- it only loads saved predictions and plots.

Expected files, one per (model, fold):
    roc_data/retformer_fold1.npz, retformer_fold2.npz, retformer_fold3.npz
    roc_data/vit_fold1.npz, ...
    roc_data/swin_fold1.npz, ...
    roc_data/deit_fold1.npz, ...
    roc_data/mae_fold1.npz, ...
each containing arrays 'y_true' and 'y_proba'.

For Retformer/ViT/Swin: these files probably do NOT exist yet from the
original niter15_v3 runs, which only logged scalar accuracy/AUC, not raw
probabilities. If load_model_folds() reports a model as MISSING below,
you have two options:
  (a) if you saved the trained Keras models (.h5/.keras) for each fold,
      just reload them and call .predict() on the held-out fold data --
      no BO search needed, so this is cheap (~seconds per fold, not hours).
  (b) if neither probabilities nor saved models exist, you do need a
      refit -- but ONLY at each fold's already-known best hyperparameters
      (from Hyperparameter_Tables_All_Models.docx), so it's 3 short
      training runs per model, not a new 15-iteration BO search.
Either way this is far cheaper than rerunning the original benchmark.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

ROC_DATA_DIR = "roc_data"

MODELS = ["retformer", "vit", "swin", "deit", "mae"]
MODEL_LABELS = {
    "retformer": "Retformer",
    "vit": "ViT",
    "swin": "Swin",
    "deit": "DeiT (no-teacher)",
    "mae": "MAE (no pretrain)",
}
MODEL_COLORS = {
    "retformer": "#1f77b4",
    "vit": "#ff7f0e",
    "swin": "#2ca02c",
    "deit": "#d62728",
    "mae": "#9467bd",
}


def load_model_folds(model_key):
    """Load and pool y_true/y_proba across all folds found for a model."""
    pattern = os.path.join(ROC_DATA_DIR, f"{model_key}_fold*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        return None

    y_true_all, y_proba_all = [], []
    for f in files:
        data = np.load(f)
        y_true_all.append(data["y_true"])
        y_proba_all.append(data["y_proba"])
    return np.concatenate(y_true_all), np.concatenate(y_proba_all)


def main():
    plt.figure(figsize=(7, 7))

    missing = []
    for model_key in MODELS:
        result = load_model_folds(model_key)
        if result is None:
            missing.append(model_key)
            continue

        y_true, y_proba = result
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr, tpr,
            label=f"{MODEL_LABELS[model_key]} (AUC = {roc_auc:.3f})",
            color=MODEL_COLORS[model_key], linewidth=2,
        )

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — AD vs CO Classification")
    plt.legend(loc="lower right")
    plt.tight_layout()

    out_path = "figure_roc_overlay.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")

    if missing:
        print(
            "\nWARNING -- no roc_data/*.npz found for: "
            + ", ".join(missing)
            + ". These models are NOT in the figure above. "
              "See the module docstring for how to backfill them "
              "(reload saved models or refit at known best params -- "
              "no new BO search needed)."
        )


if __name__ == "__main__":
    main()
