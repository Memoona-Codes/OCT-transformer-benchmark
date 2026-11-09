"""
Confusion matrix grid -- one subplot per model, using the same saved
roc_data/{model}_fold*.npz files as roc_curve_overlay.py (y_true, y_proba
pooled across all 3 outer folds, thresholded at 0.5). Does NOT retrain
anything.

For Retformer/ViT/Swin/KAN: same gap as the ROC overlay -- these need
roc_data/*.npz backfilled (reload saved models or refit at known best
params) before they'll appear here. DeiT and MAE already save this data
automatically during their normal run, so they'll appear as soon as their
full runs complete.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

ROC_DATA_DIR = "roc_data"

MODELS = ["retformer", "vit", "swin", "deit", "mae"]
MODEL_LABELS = {
    "retformer": "Retformer",
    "vit": "ViT",
    "swin": "Swin",
    "deit": "DeiT-style (no-teacher)",
    "mae": "MAE-style encoder (no pretrain)",
}


def load_model_folds(model_key):
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
    available = {}
    missing = []
    for model_key in MODELS:
        result = load_model_folds(model_key)
        if result is None:
            missing.append(model_key)
        else:
            available[model_key] = result

    if not available:
        print("No roc_data/*.npz files found for any model. Nothing to plot.")
        return

    n_models = len(available)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(-1) if n_models > 1 else [axes]

    for idx, (model_key, (y_true, y_proba)) in enumerate(available.items()):
        y_pred = (y_proba > 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)

        ax = axes[idx]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(MODEL_LABELS[model_key])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["CO", "AD"])
        ax.set_yticklabels(["CO", "AD"])

        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")

    # hide unused subplot axes
    for idx in range(n_models, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    out_path = "figure_confusion_matrices.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")

    if missing:
        print(
            "\nWARNING -- no roc_data/*.npz found for: "
            + ", ".join(missing)
            + ". These models are NOT in the figure above. Same backfill "
              "needed as roc_curve_overlay.py (reload saved models or "
              "refit at known best params -- no full rerun needed)."
        )


if __name__ == "__main__":
    main()
