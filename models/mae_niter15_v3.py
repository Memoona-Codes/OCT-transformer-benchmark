"""
MAE (encoder-only, no pretraining) -- CORRECTED readout to match
vit_swin_kfold_nested_matched_v2.py's proven pattern, following the same
fix that resolved DeiT's near-chance accuracy.

DeiT's bug was extracting a single CLS token's representation and discarding
the rest -- a severe information bottleneck on this small (1,120-sample)
dataset. The working ViT/Swin instead Flatten() every patch token's final
representation into the MLP head. MAE's ORIGINAL design here used
GlobalAveragePooling1D, which is the MAE paper's real documented
fine-tuning protocol (no CLS token, GAP + LayerNorm) -- but it carries the
same category of risk as CLS-token extraction: compressing many tokens down
to one summary vector before classification, on a dataset that's already
shown it needs the full flattened representation to learn anything useful.

Given DeiT's diagnostic evidence, this version replaces GAP with Flatten
(matching the proven working readout) as a precaution BEFORE spending
compute on a full run that might turn out equally broken. This departs
further from the literal MAE fine-tuning recipe (GAP was already a
concession; Flatten is a bigger one) -- if this test recovers good
accuracy, the Methods/Limitations section will need to say MAE's readout
was adapted for this dataset's scale rather than following the paper's
GAP protocol exactly. Still testing this: no augmentation, no
pretraining, no CLS token, fixed sin-cos positional embedding retained.

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


def sincos_position_embedding(num_patches, dim):
    """Fixed 2D sine-cosine positional embedding, as used in the MAE paper.
    FIXED (21 Aug 2026): the original version computed dim//2 for each axis
    and then ANOTHER //2 inside get_1d_sincos, which silently drops a column
    whenever that intermediate value is odd (e.g. dim=58 -> dim//2=29 (odd)
    -> internal d//2=14 -> only 28 columns produced instead of 29), causing
    a shape mismatch and crash partway through the BO search on ~half the
    search space's proj_dim values (34, 38, 42, 46, 50, 54, 58, 62). This
    version always produces exactly `dim` columns for any dim, even/odd."""
    grid_size = int(np.sqrt(num_patches))
    assert grid_size * grid_size == num_patches, \
        "sincos embedding assumes a square patch grid"

    def get_1d_sincos(pos, d):
        # Always returns exactly d columns, regardless of whether d is
        # even or odd -- compute enough frequency bins to cover d after
        # sin+cos concatenation, then truncate to exactly d.
        half = (d + 1) // 2  # ceil(d/2)
        omega = np.arange(half, dtype=np.float32)
        omega /= max(half, 1)
        omega = 1.0 / (10000 ** omega)
        out = np.einsum("m,d->md", pos, omega)
        emb = np.concatenate([np.sin(out), np.cos(out)], axis=1)  # 2*half columns
        return emb[:, :d]  # truncate to exactly d columns

    dim_h = dim // 2
    dim_w = dim - dim_h  # dim_h + dim_w == dim exactly, for any dim
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape(2, -1)

    emb_h = get_1d_sincos(grid[0], dim_h)
    emb_w = get_1d_sincos(grid[1], dim_w)
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)  # exactly dim columns
    return ops.convert_to_tensor(pos_embed, dtype="float32")


def transformer_encoder_block(x, proj_dim, heads, mlp_ratio=2):
    attn_input = layers.LayerNormalization(epsilon=1e-6)(x)
    attn_output = layers.MultiHeadAttention(num_heads=heads, key_dim=proj_dim // heads)(attn_input, attn_input)
    x = layers.Add()([x, attn_output])

    mlp_input = layers.LayerNormalization(epsilon=1e-6)(x)
    mlp = layers.Dense(proj_dim * mlp_ratio, activation="gelu")(mlp_input)
    mlp = layers.Dense(proj_dim)(mlp)
    x = layers.Add()([x, mlp])
    return x


def build_mae_encoder(input_shape, proj_dim, patch_size, layers_n, heads,
                       mlp_head_1, mlp_head_2):
    inputs = layers.Input(shape=input_shape)

    patches = layers.Conv2D(filters=proj_dim, kernel_size=patch_size, strides=patch_size, padding="valid")(inputs)
    num_patches_h = input_shape[0] // patch_size
    num_patches_w = input_shape[1] // patch_size
    num_patches = num_patches_h * num_patches_w
    x = layers.Reshape((num_patches, proj_dim))(patches)

    pos_embed = sincos_position_embedding(num_patches, proj_dim)
    x = x + pos_embed

    for _ in range(layers_n):
        x = transformer_encoder_block(x, proj_dim, heads)

    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # CHANGED: Flatten (not GAP) -- matches the proven working readout
    rep = layers.Flatten()(x)

    h = layers.Dense(mlp_head_1, activation="gelu")(rep)
    h = layers.Dense(mlp_head_2, activation="gelu")(h)
    out = layers.Dense(1, activation="sigmoid")(h)

    return keras.Model(inputs=inputs, outputs=out)


class MAEClassifier:
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
        self.model = build_mae_encoder(
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


def mae_search_space():
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
            estimator=MAEClassifier(),
            search_spaces=mae_search_space(),
            n_iter=N_ITER,
            cv=list(inner_cv.split(X_train, y_train)),
            n_jobs=1,
            verbose=3,
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        test_score = best_model.score(X_test, y_test)
        y_proba = best_model.predict_proba(X_test)

        print(f"[MAE] Fold {fold_idx}: test accuracy = {test_score:.4f}")
        print(f"[MAE] Fold {fold_idx}: best params = {search.best_params_}")

        os.makedirs(ROC_DATA_DIR, exist_ok=True)
        np.savez(
            os.path.join(ROC_DATA_DIR, f"mae_fold{fold_idx}.npz"),
            y_true=y_test, y_proba=y_proba,
        )

        outer_scores.append(test_score)
        best_params_per_fold.append(search.best_params_)

    print(f"\n[MAE] Mean accuracy: {np.mean(outer_scores):.4f} "
          f"+/- {np.std(outer_scores):.4f}")
    return outer_scores, best_params_per_fold


if __name__ == "__main__":
    run_nested_cv()
