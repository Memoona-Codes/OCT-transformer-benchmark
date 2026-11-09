"""
Figure 1 builder -- Sample OCT slices, AD vs. CO.
"""
import os
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SELECTION_MODE = "auto"
MANUAL_AD_FILES = []
MANUAL_CO_FILES = []


def list_images(folder):
    exts = (".png", ".jpg", ".jpeg")
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(exts))


def score_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return -1
    return float(np.std(img))


def pick_representative(folder, n, mode="auto"):
    files = list_images(folder)
    if not files:
        raise RuntimeError(f"No image files found in {folder}")
    scored = [(f, score_image(os.path.join(folder, f))) for f in files]
    scored = [s for s in scored if s[1] >= 0]
    scored.sort(key=lambda x: x[1], reverse=True)

    if mode == "random":
        rng = np.random.default_rng(1)
        chosen = list(rng.choice([f for f, _ in scored], size=min(n, len(scored)), replace=False))
        return chosen

    top_half = scored[: max(n, len(scored) // 2)]
    idxs = np.linspace(0, len(top_half) - 1, num=min(n, len(top_half))).astype(int)
    return [top_half[i][0] for i in idxs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", type=str, default="./oct_data_adco")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", type=str, default="figure1_ad_vs_co.png")
    args = ap.parse_args()

    ad_dir = os.path.join(args.address, "ad")
    co_dir = os.path.join(args.address, "hc")
    if not os.path.isdir(co_dir):
        alt = os.path.join(args.address, "co")
        if os.path.isdir(alt):
            co_dir = alt

    if SELECTION_MODE == "manual" and MANUAL_AD_FILES and MANUAL_CO_FILES:
        ad_files = MANUAL_AD_FILES[: args.n]
        co_files = MANUAL_CO_FILES[: args.n]
    else:
        ad_files = pick_representative(ad_dir, args.n, mode=SELECTION_MODE)
        co_files = pick_representative(co_dir, args.n, mode=SELECTION_MODE)

    n = min(len(ad_files), len(co_files))
    ad_files, co_files = ad_files[:n], co_files[:n]

    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 5.2))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i, f in enumerate(ad_files):
        img = cv2.imread(os.path.join(ad_dir, f), cv2.IMREAD_GRAYSCALE)
        axes[0, i].imshow(img, cmap="gray")
        axes[0, i].axis("off")
    for i, f in enumerate(co_files):
        img = cv2.imread(os.path.join(co_dir, f), cv2.IMREAD_GRAYSCALE)
        axes[1, i].imshow(img, cmap="gray")
        axes[1, i].axis("off")

    axes[0, 0].text(-0.25, 0.5, "AD", transform=axes[0, 0].transAxes,
                     fontsize=16, fontweight="bold", va="center", ha="right", rotation=90)
    axes[1, 0].text(-0.25, 0.5, "CO", transform=axes[1, 0].transAxes,
                     fontsize=16, fontweight="bold", va="center", ha="right", rotation=90)

    plt.tight_layout(rect=[0.03, 0.05, 1, 1])
    fig.text(0.5, 0.01,
              "Figure 1. Sample OCT slices from (top) AD patients and (bottom) "
              "age-matched cognitively normal controls.",
              ha="center", fontsize=10, style="italic")

    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved {args.out}")
    print("AD files used:", ad_files)
    print("CO files used:", co_files)


if __name__ == "__main__":
    main()
