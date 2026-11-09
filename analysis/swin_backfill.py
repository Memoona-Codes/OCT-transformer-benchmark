"""
Swin backfill -- refits at each outer fold's ALREADY-KNOWN best
hyperparameters (from the paper's Table II), then saves y_true/y_proba to
roc_data/swin_fold*.npz so Swin can join the combined ROC overlay and
confusion matrix figures alongside Retformer/ViT/DeiT/MAE.

Same approach as retformer_backfill.py / vit_backfill.py: 3 refits (one per
fold, 100 epochs each), NOT a new BO search.

Architecture reconstructed VERBATIM from vit_swin_kfold_nested_matched_v2.py
(confirmed against actual source): same Patches/PatchEmbedLearned as ViT,
but with ShiftedWindowAttention (real cyclic-shift + windowed self-attention,
alternating shift=0/window_size//2 per layer) instead of standard
MultiHeadAttention -- matching Liu et al. 2021 Swin Transformer's mechanism,
without the learned relative position bias table (disclosed simplification
per the paper's own module docstring).
"""

import os
import numpy as np
import keras
from keras import layers, ops
from sklearn.model_selection import KFold

from dataloader import read_data

ROC_DATA_DIR = "roc_data"
INPUT_SHAPE = (50, 50, 1)
RANDOM_STATE = 1
N_OUTER_FOLDS = 3
EPOCHS = 100
BATCH_SIZE = 250

# Table II best-found configs, one per outer fold (already known, no search needed)
# grid = 50 // patch_size = 10 for all folds (patch_size fixed at 5); grid must
# be divisible by window_size -- confirmed: 10%10=0, 10%5=0, all valid.
FOLD_CONFIGS = {
    1: dict(mlp_head_unit_1=54, mlp_head_unit_2=22, num_heads=2, patch_size=5, projection_dim=44, transformer_layers=1, window_size=10),
    2: dict(mlp_head_unit_1=35, mlp_head_unit_2=16, num_heads=5, patch_size=5, projection_dim=50, transformer_layers=1, window_size=5),
    3: dict(mlp_head_unit_1=34, mlp_head_unit_2=18, num_heads=5, patch_size=5, projection_dim=60, transformer_layers=3, window_size=5),
}


class Patches(layers.Layer):
    """Verbatim from vit_swin_kfold_nested_matched_v2.py (shared with ViT)."""
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = ops.shape(images)[0]
        patches = keras.ops.image.extract_patches(images, size=self.patch_size)
        patches = ops.reshape(patches, (batch_size, -1, self.patch_size * self.patch_size * images.shape[-1]))
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config


class PatchEmbedLearned(layers.Layer):
    """Verbatim from vit_swin_kfold_nested_matched_v2.py (shared with ViT)."""
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        cnn_patches = self.projection(patch)
        positions = ops.arange(0, self.num_patches, 1)
        return cnn_patches + self.position_embedding(positions)

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches, "projection_dim": self.projection_dim})
        return config


def mlp_block(x, hidden_dim, projection_dim):
    """Verbatim from vit_swin_kfold_nested_matched_v2.py."""
    x = layers.Dense(hidden_dim, activation=keras.activations.gelu)(x)
    x = layers.Dense(projection_dim)(x)
    return x


def window_partition(x, h, w, window_size, projection_dim, batch_size):
    """Verbatim from vit_swin_kfold_nested_matched_v2.py."""
    x = ops.reshape(x, (batch_size, h, w, projection_dim))
    x = ops.reshape(x, (batch_size, h // window_size, window_size,
                         w // window_size, window_size, projection_dim))
    x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
    windows = ops.reshape(x, (-1, window_size * window_size, projection_dim))
    return windows


def window_reverse(windows, h, w, window_size, projection_dim, batch_size):
    """Verbatim from vit_swin_kfold_nested_matched_v2.py."""
    x = ops.reshape(windows, (batch_size, h // window_size, w // window_size,
                               window_size, window_size, projection_dim))
    x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
    x = ops.reshape(x, (batch_size, h, w, projection_dim))
    return ops.reshape(x, (batch_size, h * w, projection_dim))


class ShiftedWindowAttention(layers.Layer):
    """Verbatim from vit_swin_kfold_nested_matched_v2.py."""
    def __init__(self, projection_dim, num_heads, window_size, shift_size, h, w):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.h = h
        self.w = w
        self.projection_dim = projection_dim
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim // num_heads)

    def call(self, x):
        batch_size = ops.shape(x)[0]
        x_2d = ops.reshape(x, (batch_size, self.h, self.w, self.projection_dim))
        if self.shift_size > 0:
            x_2d = ops.roll(x_2d, shift=(-self.shift_size, -self.shift_size), axis=(1, 2))
        x_flat = ops.reshape(x_2d, (batch_size, self.h * self.w, self.projection_dim))
        windows = window_partition(x_flat, self.h, self.w, self.window_size, self.projection_dim, batch_size)
        attended = self.mha(windows, windows)
        merged = window_reverse(attended, self.h, self.w, self.window_size, self.projection_dim, batch_size)
        merged_2d = ops.reshape(merged, (batch_size, self.h, self.w, self.projection_dim))
        if self.shift_size > 0:
            merged_2d = ops.roll(merged_2d, shift=(self.shift_size, self.shift_size), axis=(1, 2))
        return ops.reshape(merged_2d, (batch_size, self.h * self.w, self.projection_dim))

    def get_config(self):
        config = super().get_config()
        config.update({
            "window_size": self.window_size, "shift_size": self.shift_size,
            "h": self.h, "w": self.w, "projection_dim": self.projection_dim,
        })
        return config


def create_swin_flat(input_shape, projection_dim, patch_size, transformer_layers,
                      num_heads, window_size, mlp_head_unit_1, mlp_head_unit_2):
    """Verbatim from vit_swin_kfold_nested_matched_v2.py."""
    grid = input_shape[0] // patch_size
    num_patches = grid * grid
    assert grid % window_size == 0, (
        f"grid size {grid} (patch_size={patch_size}) must be divisible by "
        f"window_size {window_size}"
    )

    inputs = keras.Input(shape=input_shape)
    patches = Patches(patch_size)(inputs)
    encoded = PatchEmbedLearned(num_patches, projection_dim)(patches)

    for i in range(transformer_layers):
        shift = 0 if i % 2 == 0 else window_size // 2
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded)
        attn = ShiftedWindowAttention(projection_dim, num_heads, window_size, shift, grid, grid)(x1)
        x2 = layers.Add()([attn, encoded])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        mlp_out = mlp_block(x3, projection_dim * 2, projection_dim)
        encoded = layers.Add()([mlp_out, x2])

    rep = layers.LayerNormalization(epsilon=1e-6)(encoded)
    rep = layers.Flatten()(rep)
    features = rep
    for units in (mlp_head_unit_1, mlp_head_unit_2):
        features = layers.Dense(units, activation=keras.activations.gelu)(features)
    logits = layers.Dense(1, activation='sigmoid')(features)
    return keras.Model(inputs=inputs, outputs=logits)


def main():
    X, y = read_data(INPUT_SHAPE, debug=False)
    print(f"Input shape: {X.shape}")

    outer_cv = KFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(outer_cv.split(X, y))

    os.makedirs(ROC_DATA_DIR, exist_ok=True)

    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        config = FOLD_CONFIGS[fold_idx]
        print(f"\n[Swin backfill] Fold {fold_idx}: refitting at {config}")

        model = create_swin_flat(INPUT_SHAPE, **config)
        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001),
            loss=keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=['accuracy'],
        )
        model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0)

        y_proba = model.predict(X_test, verbose=0).ravel()
        y_pred = (y_proba > 0.5).astype(int)
        acc = np.mean(y_pred == y_test)
        print(f"[Swin backfill] Fold {fold_idx}: refit test accuracy = {acc:.4f} "
              f"(original logged: overall mean was 0.9062)")

        np.savez(
            os.path.join(ROC_DATA_DIR, f"swin_fold{fold_idx}.npz"),
            y_true=y_test, y_proba=y_proba,
        )

    print("\n[Swin backfill] Done -- roc_data/swin_fold*.npz saved for all 3 folds.")


if __name__ == "__main__":
    main()
