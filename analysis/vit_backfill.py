"""
ViT backfill -- refits at each outer fold's ALREADY-KNOWN best hyperparameters
(from the paper's Table II), then saves y_true/y_proba to
roc_data/vit_fold*.npz so ViT can join the combined ROC overlay and
confusion matrix figures alongside Retformer/DeiT/MAE.

Same approach as retformer_backfill.py: 3 refits (one per fold, 100 epochs
each), NOT a new BO search -- the hyperparameters are already known from the
original run's Table II.

Architecture reconstructed VERBATIM from vit_swin_kfold_nested_matched_v2.py
(confirmed against actual source): Patches (raw patch extraction, flattened)
-> PatchEmbedLearned (Dense projection + learned position embedding) ->
standard transformer blocks (LayerNorm -> MultiHeadAttention -> Add ->
LayerNorm -> mlp_block -> Add) -> LayerNorm -> Flatten -> mlp head -> sigmoid.
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
FOLD_CONFIGS = {
    1: dict(mlp_head_unit_1=32, mlp_head_unit_2=32, num_heads=1, patch_size=5, projection_dim=36, transformer_layers=1),
    2: dict(mlp_head_unit_1=35, mlp_head_unit_2=20, num_heads=1, patch_size=5, projection_dim=32, transformer_layers=1),
    3: dict(mlp_head_unit_1=54, mlp_head_unit_2=32, num_heads=5, patch_size=5, projection_dim=36, transformer_layers=2),
}


class Patches(layers.Layer):
    """Verbatim from vit_swin_kfold_nested_matched_v2.py."""
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
    """Verbatim from vit_swin_kfold_nested_matched_v2.py."""
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


def create_vanilla_vit(input_shape, projection_dim, patch_size, transformer_layers,
                        num_heads, mlp_head_unit_1, mlp_head_unit_2):
    """Verbatim from vit_swin_kfold_nested_matched_v2.py."""
    num_patches = (input_shape[0] // patch_size) ** 2

    inputs = keras.Input(shape=input_shape)
    patches = Patches(patch_size)(inputs)
    encoded = PatchEmbedLearned(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded)
        attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim // num_heads)(x1, x1)
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
        print(f"\n[ViT backfill] Fold {fold_idx}: refitting at {config}")

        model = create_vanilla_vit(INPUT_SHAPE, **config)
        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001),
            loss=keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=['accuracy'],
        )
        model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0)

        y_proba = model.predict(X_test, verbose=0).ravel()
        y_pred = (y_proba > 0.5).astype(int)
        acc = np.mean(y_pred == y_test)
        print(f"[ViT backfill] Fold {fold_idx}: refit test accuracy = {acc:.4f} "
              f"(original logged: overall mean was 0.9179)")

        np.savez(
            os.path.join(ROC_DATA_DIR, f"vit_fold{fold_idx}.npz"),
            y_true=y_test, y_proba=y_proba,
        )

    print("\n[ViT backfill] Done -- roc_data/vit_fold*.npz saved for all 3 folds.")


if __name__ == "__main__":
    main()
