"""
DeiT (no-teacher variant) -- CORRECTED to match vit_swin_kfold_nested_matched_v2.py's
actual proven architecture, not a from-scratch reconstruction.

Root cause of the original near-chance results (0.57/0.59/0.53): the working
ViT/Swin in this codebase does NOT use a CLS-token bottleneck. create_vanilla_vit
flattens EVERY patch token's final representation and feeds all of it into the
MLP head. My original DeiT extracted only a single CLS token's representation,
discarding ~99% of the transformer's computed output before classification --
a severe information bottleneck on a small (1,120-sample) dataset. Even the
CLS-only single-head diagnostic (no distillation token) still landed at 62.83%
because the CLS bottleneck itself was the problem, not the dual-head averaging.

Fix: use the REAL Patches (raw patch extraction, not Conv2D) and
PatchEmbedLearned (Dense projection + learned position embedding) classes
verbatim from vit_swin_kfold_nested_matched_v2.py. Add ONE distillation
token as an extra sequence element (not a separate CLS/dist pooled pair),
then Flatten the ENTIRE sequence (all patches + the distillation token)
before the MLP head, matching the proven working readout exactly. Single
output head (the dual-head-averaging design is dropped -- it measurably
hurt performance in diagnostics and isn't needed once Flatten preserves
full information anyway).

This keeps the "extra learnable token, trained on true labels, no teacher"
DeiT spirit while using the architecture pattern that's actually proven to
work on this dataset/pipeline.

Everything else unchanged: AdamW(lr=0.001, wd=0.0001), BinaryCrossentropy,
epochs=100, batch=250, KFold(3, shuffle=True, random_state=1) both levels,
BayesSearchCV(n_iter=15, no random_state), standalone Keras 3.
"""

import os
import numpy as np
import keras
from keras import layers, ops
from sklearn.model_selection import KFold
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical

ROC_DATA_DIR = "roc_data"
from dataloader import read_data

RANDOM_STATE = 1
N_OUTER_FOLDS = 3
N_INNER_FOLDS = 3
N_ITER = 15
EPOCHS = 100
BATCH_SIZE = 250
INPUT_SHAPE = (50, 50, 1)


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


class PatchEmbedWithDistillToken(layers.Layer):
    """Adapted from PatchEmbedLearned: same Dense projection + learned
    position embedding, but with one extra learnable distillation token
    prepended to the patch sequence (so it gets its own position embedding
    slot and participates in every transformer block like a normal token)."""

    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(projection_dim)
        # +1 slot for the distillation token
        self.position_embedding = layers.Embedding(input_dim=num_patches + 1, output_dim=projection_dim)

    def build(self, input_shape):
        self.dist_token = self.add_weight(
            name="dist_token", shape=(1, 1, self.projection_dim),
            initializer="random_normal", trainable=True,
        )
        super().build(input_shape)

    def call(self, patch):
        batch_size = ops.shape(patch)[0]
        cnn_patches = self.projection(patch)
        dist_token = ops.broadcast_to(self.dist_token, (batch_size, 1, self.projection_dim))
        x = ops.concatenate([dist_token, cnn_patches], axis=1)
        positions = ops.arange(0, self.num_patches + 1, 1)
        return x + self.position_embedding(positions)

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches, "projection_dim": self.projection_dim})
        return config


def mlp_block(x, hidden_dim, projection_dim):
    x = layers.Dense(hidden_dim, activation=keras.activations.gelu)(x)
    x = layers.Dense(projection_dim)(x)
    return x


def build_deit(input_shape, proj_dim, patch_size, layers_n, heads,
                mlp_head_1, mlp_head_2):
    num_patches = (input_shape[0] // patch_size) ** 2

    inputs = keras.Input(shape=input_shape)
    patches = Patches(patch_size)(inputs)
    encoded = PatchEmbedWithDistillToken(num_patches, proj_dim)(patches)

    for _ in range(layers_n):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded)
        attn = layers.MultiHeadAttention(num_heads=heads, key_dim=proj_dim // heads)(x1, x1)
        x2 = layers.Add()([attn, encoded])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        mlp_out = mlp_block(x3, proj_dim * 2, proj_dim)
        encoded = layers.Add()([mlp_out, x2])

    rep = layers.LayerNormalization(epsilon=1e-6)(encoded)
    rep = layers.Flatten()(rep)  # flattens ALL tokens (patches + dist token), matches proven ViT readout
    features = rep
    for units in (mlp_head_1, mlp_head_2):
        features = layers.Dense(units, activation=keras.activations.gelu)(features)
    logits = layers.Dense(1, activation='sigmoid')(features)
    return keras.Model(inputs=inputs, outputs=logits)


class DeiTClassifier:
    def __init__(self, proj_dim=48, patch_size=5, layers_n=2, heads=2,
                 mlp_head_1=48, mlp_head_2=24, input_shape=(50, 50, 1)):
        self.proj_dim = proj_dim
        self.patch_size = patch_size
        self.layers_n = layers_n
        self.heads = heads
        self.mlp_head_1 = mlp_head_1
        self.mlp_head_2 = mlp_head_2
        self.input_shape = input_shape
        self.model = None

    def get_params(self, deep=True):
        return {
            "proj_dim": self.proj_dim, "patch_size": self.patch_size,
            "layers_n": self.layers_n, "heads": self.heads,
            "mlp_head_1": self.mlp_head_1, "mlp_head_2": self.mlp_head_2,
            "input_shape": self.input_shape,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y):
        self.model = build_deit(
            self.input_shape, self.proj_dim, self.patch_size, self.layers_n,
            self.heads, self.mlp_head_1, self.mlp_head_2,
        )
        self.model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001),
            loss=keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=['accuracy'],
        )
        self.model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        return self

    def predict(self, X):
        preds = self.model.predict(X, verbose=0)
        return (preds > 0.5).astype(int).ravel()

    def predict_proba(self, X):
        return self.model.predict(X, verbose=0).ravel()

    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)


def deit_search_space():
    return {
        "proj_dim": Categorical(list(range(32, 64, 2))),
        "patch_size": Categorical([2, 5, 10]),
        "layers_n": Integer(1, 4),
        "heads": Categorical([1, 2, 4, 5]),
        "mlp_head_1": Categorical([32, 64]),
        "mlp_head_2": Categorical([16, 32]),
    }


def run_nested_cv():
    X, y = read_data(INPUT_SHAPE, debug=False)
    print(f"Input shape: {X.shape}")

    outer_cv = KFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    outer_scores = []
    best_params_per_fold = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        inner_cv = KFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        search = BayesSearchCV(
            estimator=DeiTClassifier(),
            search_spaces=deit_search_space(),
            n_iter=N_ITER,
            cv=list(inner_cv.split(X_train, y_train)),
            n_jobs=1,
            verbose=3,
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        test_score = best_model.score(X_test, y_test)
        y_proba = best_model.predict_proba(X_test)

        print(f"[DeiT] Fold {fold_idx}: test accuracy = {test_score:.4f}")
        print(f"[DeiT] Fold {fold_idx}: best params = {search.best_params_}")

        os.makedirs(ROC_DATA_DIR, exist_ok=True)
        np.savez(
            os.path.join(ROC_DATA_DIR, f"deit_fold{fold_idx}.npz"),
            y_true=y_test, y_proba=y_proba,
        )

        outer_scores.append(test_score)
        best_params_per_fold.append(search.best_params_)

    print(f"\n[DeiT] Mean accuracy: {np.mean(outer_scores):.4f} "
          f"+/- {np.std(outer_scores):.4f}")
    return outer_scores, best_params_per_fold


if __name__ == "__main__":
    run_nested_cv()
