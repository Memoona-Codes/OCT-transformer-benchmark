"""
Retformer backfill -- refits at each outer fold's ALREADY-KNOWN best
hyperparameters (from the paper's Table II, produced by the original BO
search), then saves y_true/y_proba to roc_data/retformer_fold*.npz so
Retformer can join the combined ROC overlay and confusion matrix figures
alongside DeiT and MAE.

This is NOT a full BO search -- no new hyperparameter search happens here,
each fold trains exactly once at its known-best config. But it IS a real
refit (100 epochs, same as the original run) since no model checkpoints
were saved from the original run, so there is no faster shortcut available.
Expect roughly the same per-fold wall-clock time as the original training
took. Predictions will be close to, but not bit-identical to, the
originally-reported 90.72%+/-2.54% accuracy (different random init), since
this is a genuine refit, not a reload of the exact original weights.

Architecture reconstructed VERBATIM from retformer.py (confirmed against
actual source, not guessed): Patches (raw patch extraction, patches stay in
grid shape, NOT flattened) -> PatchEncoder (Conv2D 3x3 projection + RoPE
positional embedding via keras_nlp.layers.RotaryEmbedding) -> transformer
blocks (LayerNorm -> GroupQueryAttention -> Add -> LayerNorm -> SwiGLU ->
Add) -> LayerNorm -> Flatten -> mlp head -> sigmoid.
"""

import os
import numpy as np
import keras
import keras_nlp
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
    1: dict(mlp_head_unit_1=52, mlp_head_unit_2=25, num_q_heads=3, num_kv_heads=3, patch_size=5, projection_dim=40, transformer_layers=2),
    2: dict(mlp_head_unit_1=53, mlp_head_unit_2=17, num_q_heads=1, num_kv_heads=1, patch_size=4, projection_dim=44, transformer_layers=1),
    3: dict(mlp_head_unit_1=40, mlp_head_unit_2=19, num_q_heads=3, num_kv_heads=2, patch_size=4, projection_dim=56, transformer_layers=1),
}


class SwiGLU(layers.Layer):
    """Verbatim from retformer.py."""
    def __init__(self, bias=True, dim=-1, **kwargs):
        super(SwiGLU, self).__init__(**kwargs)
        self.bias = bias
        self.dim = dim
        self.dense = layers.Dense(2, use_bias=bias)

    def __call__(self, x):
        gate = keras.activations.swish(x)
        x = ops.multiply(x, gate)
        return x


def mlp(x, hidden_units):
    """Verbatim from retformer.py."""
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
    return x


class Patches(layers.Layer):
    """Verbatim from retformer.py -- NOTE: patches stay in grid shape
    (batch, num_patches_h, num_patches_w, patch_dim), NOT flattened."""
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        input_shape = ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = keras.ops.image.extract_patches(images, size=self.patch_size)
        patches = ops.reshape(
            patches,
            (batch_size, num_patches_h, num_patches_w,
             self.patch_size * self.patch_size * channels),
        )
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config


class PatchEncoder(layers.Layer):
    """Verbatim from retformer.py -- Conv2D(3x3) projection + RoPE."""
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Conv2D(projection_dim, kernel_size=(3, 3), padding='same')
        self.position_embedding = keras_nlp.layers.RotaryEmbedding()

    def call(self, patch):
        cnn_patches = self.projection(patch)
        projected_patches = ops.reshape(cnn_patches, (-1, self.num_patches, self.projection_dim))
        encoded = projected_patches + self.position_embedding(projected_patches)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config


def create_vit_classifier(input_shape, projection_dim, patch_size, transformer_layers,
                           num_q_heads, num_kv_heads, mlp_head_unit_1, mlp_head_unit_2):
    """Verbatim from retformer.py (this IS Retformer despite the generic name)."""
    mlp_head_units = [mlp_head_unit_1, mlp_head_unit_2]
    num_patches = (input_shape[0] // patch_size) ** 2

    inputs = keras.Input(shape=input_shape)
    patches = Patches(patch_size)(inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.GroupQueryAttention(
            num_query_heads=num_q_heads * num_kv_heads, head_dim=projection_dim,
            num_key_value_heads=num_kv_heads,
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = SwiGLU()(x3)
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    features = mlp(representation, hidden_units=mlp_head_units)
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
        print(f"\n[Retformer backfill] Fold {fold_idx}: refitting at {config}")

        model = create_vit_classifier(INPUT_SHAPE, **config)
        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001),
            loss=keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=['accuracy'],
        )
        model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0)

        y_proba = model.predict(X_test, verbose=0).ravel()
        y_pred = (y_proba > 0.5).astype(int)
        acc = np.mean(y_pred == y_test)
        print(f"[Retformer backfill] Fold {fold_idx}: refit test accuracy = {acc:.4f} "
              f"(original logged: varies by fold, overall mean was 0.9072)")

        np.savez(
            os.path.join(ROC_DATA_DIR, f"retformer_fold{fold_idx}.npz"),
            y_true=y_test, y_proba=y_proba,
        )

    print("\n[Retformer backfill] Done -- roc_data/retformer_fold*.npz saved for all 3 folds.")


if __name__ == "__main__":
    main()
