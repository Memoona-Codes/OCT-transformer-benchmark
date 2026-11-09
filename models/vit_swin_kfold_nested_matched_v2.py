"""
Matched-protocol ViT/Swin nested-CV script -- rewritten to fix the
pretrained/augmentation/resolution mismatch found in the original
vit_kfold_nested_leaky.py (which used timm ImageNet-pretrained backbones at
224x224 with real data augmentation -- fundamentally incomparable to
KAN/Retformer/Diffusion, which all train from scratch at 50x50/32x32 with
zero augmentation).

Trains a small, from-scratch Keras model (vanilla ViT or a flat,
single-stage Swin-style shifted-window transformer) matched EXACTLY to
Retformer's bayes.py on every axis except the attention mechanism itself:

  - input_shape = (50, 50, 1)                      <- same as bayes.py
  - KFold(n_splits=k, shuffle=True, random_state=1) <- same as bayes.py
    for BOTH outer and inner CV
  - n_iter = 10 (BayesSearchCV)                     <- same as bayes.py
  - batch_size = 250, epochs = 100, NO early stopping <- same as
    retformer.py's ViTWrapper.fit()
  - NO data augmentation
  - NO pretrained weights -- trained fully from scratch, every run

This makes "ViT (nested)" and "Swin (nested)" genuine architectural
baselines against Retformer: same resolution, same training budget, same
CV structure, same seed -- differing ONLY in attention design (standard
multi-head self-attention for ViT, shifted-window self-attention for Swin,
vs. Retformer's GQA + SwiGLU + RoPE).

IMPORTANT, DISCLOSED SIMPLIFICATIONS (for the paper's methods/limitations
section -- do not treat these as accidental):

1. Swin here is a FLAT, SINGLE-STAGE shifted-window transformer (patch
   embed -> N shifted-window attention blocks -> classifier head) -- NOT
   the original 4-stage hierarchical Swin with patch merging between
   stages. A faithful 4-stage hierarchical Swin changes spatial resolution
   between stages, which would break structural comparability with
   Retformer/ViT's flat single-resolution design. This flat variant keeps
   the SAME skeleton as Retformer/ViT, changing only the attention
   mechanism -- which is the actual point of this comparison.

2. The shifted-window attention implemented here does NOT include Swin's
   usual learned relative position bias table. Real cyclic shift + window
   partitioning IS implemented (matching Liu et al. 2021's mechanism), but
   the relative position bias add-on was left out for scope. Flag this
   explicitly if writing up Swin's results -- it is a real, intentional
   simplification, not a hidden omission.

3. Position embeddings for both models are LEARNED (standard
   keras.layers.Embedding), not RoPE -- RoPE is one of Retformer's specific
   named design choices (see the Jamshidiha et al. paper's ablation study,
   which shows RoPE outperforms learned embeddings), not a generic
   ViT/Swin baseline feature, so using RoPE here would blur the very
   comparison this script exists to make.

Usage (run from the SAME directory as bayes.py / bayes_kan_leaky.py, so the
default --address './oct_data_adco' resolves correctly):

    conda activate retformer_repro
    cd ~/retformer_paper_repro

    python vit_swin_kfold_nested_matched.py --model_name vit \
        --n_outer_folds 3 --n_inner_folds 3 \
        --run_name vit_3fold_nested_matched \
        2>&1 | tee vit_3fold_nested_matched_output.txt

    python vit_swin_kfold_nested_matched.py --model_name swin \
        --n_outer_folds 3 --n_inner_folds 3 \
        --run_name swin_3fold_nested_matched \
        2>&1 | tee swin_3fold_nested_matched_output.txt

    (repeat with --n_outer_folds 5 --n_inner_folds 5 for the 5-fold cells)

Before running: confirm the data folder actually exists at that path --
    find ~/retformer_paper_repro -maxdepth 1 -iname "oct_data_adco"
If it's somewhere else, pass --address <path> explicitly.

Output printed to stdout in the same format as bayes.py (searchable via the
same `grep "Fold.*Accuracy:"` / `grep -a -A1 "Runtime:"` workflow already
used for Retformer/KAN in this study).
"""

import os
import time
import argparse
import warnings

import numpy as np
import keras
from keras import layers, ops
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from skopt import BayesSearchCV

from dataloader import read_data

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = ops.shape(images)[0]
        patches = keras.ops.image.extract_patches(images, size=self.patch_size)
        # h/w are statically known (input_shape and patch_size are fixed
        # Python ints at graph-build time) -- do NOT return them from this
        # layer's call(): Keras 3 resolves ops.shape() to a plain int here,
        # not a tensor, and a layer returning a bare int alongside a tensor
        # raises "non-tensor received". Reshape using -1 and let the caller
        # compute the grid size directly in Python instead.
        patches = ops.reshape(patches, (batch_size, -1, self.patch_size * self.patch_size * images.shape[-1]))
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config


class PatchEmbedLearned(layers.Layer):
    """Conv2D patch projection + a LEARNED position embedding -- see module
    docstring point 3 for why this is learned rather than RoPE."""

    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        # Patches now arrive pre-flattened as (batch, num_patches, patch_dim)
        # from the Patches layer, so a Dense projection (the standard ViT
        # linear-patch-embedding design) is used here instead of Conv2D,
        # which required unflattened spatial input.
        self.projection = layers.Dense(projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        # patch arrives already flattened to (batch, num_patches, patch_dim)
        # from the Patches layer -- project each patch vector directly,
        # no h/w extraction needed (see Patches.call for why that broke).
        batch_size = ops.shape(patch)[0]
        cnn_patches = self.projection(patch)
        positions = ops.arange(0, self.num_patches, 1)
        return cnn_patches + self.position_embedding(positions)

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches, "projection_dim": self.projection_dim})
        return config


def mlp_block(x, hidden_dim, projection_dim):
    x = layers.Dense(hidden_dim, activation=keras.activations.gelu)(x)
    x = layers.Dense(projection_dim)(x)
    return x


# ---------------------------------------------------------------------------
# Vanilla ViT (standard multi-head self-attention, learned position embed)
# ---------------------------------------------------------------------------

def create_vanilla_vit(input_shape, projection_dim, patch_size, transformer_layers,
                        num_heads, mlp_head_unit_1, mlp_head_unit_2):
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


# ---------------------------------------------------------------------------
# Flat, single-stage Swin-style transformer (real shifted-window attention;
# see module docstring points 1-2 for disclosed simplifications)
# ---------------------------------------------------------------------------

def window_partition(x, h, w, window_size, projection_dim, batch_size):
    x = ops.reshape(x, (batch_size, h, w, projection_dim))
    x = ops.reshape(x, (batch_size, h // window_size, window_size,
                         w // window_size, window_size, projection_dim))
    x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
    windows = ops.reshape(x, (-1, window_size * window_size, projection_dim))
    return windows


def window_reverse(windows, h, w, window_size, projection_dim, batch_size):
    x = ops.reshape(windows, (batch_size, h // window_size, w // window_size,
                               window_size, window_size, projection_dim))
    x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
    x = ops.reshape(x, (batch_size, h, w, projection_dim))
    return ops.reshape(x, (batch_size, h * w, projection_dim))


class ShiftedWindowAttention(layers.Layer):
    """Real cyclic-shift + windowed self-attention, matching the mechanism
    from Liu et al. 2021 (Swin Transformer) -- WITHOUT the learned relative
    position bias table. See module docstring point 2."""

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


# ---------------------------------------------------------------------------
# sklearn-compatible wrappers -- batch_size=250, epochs=100, no augmentation,
# no early stopping: matches retformer.py's ViTWrapper.fit() exactly
# ---------------------------------------------------------------------------

class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print("End epoch {} of training; {} : {}".format(epoch, logs.keys(), logs.values()), end='\r')


class VanillaViTWrapper(BaseEstimator):
    def __init__(self, projection_dim=64, patch_size=5, transformer_layers=3,
                 num_heads=4, mlp_head_unit_1=64, mlp_head_unit_2=32):
        self.projection_dim = projection_dim
        self.patch_size = patch_size
        self.transformer_layers = transformer_layers
        self.num_heads = num_heads
        self.mlp_head_unit_1 = mlp_head_unit_1
        self.mlp_head_unit_2 = mlp_head_unit_2

    def get_params(self, deep=True):
        return {
            'projection_dim': self.projection_dim,
            'patch_size': self.patch_size,
            'transformer_layers': self.transformer_layers,
            'num_heads': self.num_heads,
            'mlp_head_unit_1': self.mlp_head_unit_1,
            'mlp_head_unit_2': self.mlp_head_unit_2,
        }

    def fit(self, inputs, targets):
        input_shape = (50, 50, 1)
        self.model = create_vanilla_vit(
            input_shape, self.projection_dim, self.patch_size, self.transformer_layers,
            self.num_heads, self.mlp_head_unit_1, self.mlp_head_unit_2,
        )
        self.model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001),
            loss=keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=['accuracy'],
        )
        self.model.fit(inputs, targets, batch_size=250, epochs=100,
                        callbacks=[CustomCallback()], verbose=0)
        return self

    def score(self, inputs, targets):
        probs = self.model.predict(inputs, verbose=0)
        _, accuracy = self.model.evaluate(inputs, targets, verbose=0)
        try:
            auc = roc_auc_score(targets, probs)
        except ValueError:
            auc = 0.0
        self.last_auc_ = auc
        self.last_accuracy_ = accuracy
        return accuracy


class SwinFlatWrapper(BaseEstimator):
    def __init__(self, projection_dim=64, patch_size=5, transformer_layers=3,
                 num_heads=4, window_size=5, mlp_head_unit_1=64, mlp_head_unit_2=32):
        self.projection_dim = projection_dim
        self.patch_size = patch_size
        self.transformer_layers = transformer_layers
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_head_unit_1 = mlp_head_unit_1
        self.mlp_head_unit_2 = mlp_head_unit_2

    def get_params(self, deep=True):
        return {
            'projection_dim': self.projection_dim,
            'patch_size': self.patch_size,
            'transformer_layers': self.transformer_layers,
            'num_heads': self.num_heads,
            'window_size': self.window_size,
            'mlp_head_unit_1': self.mlp_head_unit_1,
            'mlp_head_unit_2': self.mlp_head_unit_2,
        }

    def fit(self, inputs, targets):
        input_shape = (50, 50, 1)
        self.model = create_swin_flat(
            input_shape, self.projection_dim, self.patch_size, self.transformer_layers,
            self.num_heads, self.window_size, self.mlp_head_unit_1, self.mlp_head_unit_2,
        )
        self.model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001),
            loss=keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=['accuracy'],
        )
        self.model.fit(inputs, targets, batch_size=250, epochs=100,
                        callbacks=[CustomCallback()], verbose=0)
        return self

    def score(self, inputs, targets):
        probs = self.model.predict(inputs, verbose=0)
        _, accuracy = self.model.evaluate(inputs, targets, verbose=0)
        try:
            auc = roc_auc_score(targets, probs)
        except ValueError:
            auc = 0.0
        self.last_auc_ = auc
        self.last_accuracy_ = accuracy
        return accuracy


# ---------------------------------------------------------------------------
# Main: outer/inner nested CV loop -- structurally identical to bayes.py
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, required=True, choices=["vit", "swin"])
    p.add_argument("--n_outer_folds", type=int, required=True, choices=[3, 5])
    p.add_argument("--n_inner_folds", type=int, required=True, choices=[3, 5])
    p.add_argument("--n_iter", type=int, default=10)
    p.add_argument("--address", type=str, default=os.environ.get("OCT_DATA_DIR", "./OCT_Dataset"),
                    help="Dataset path -- default: $OCT_DATA_DIR or ./OCT_Dataset.")
    p.add_argument("--run_name", type=str, required=True)
    return p.parse_args()


def main():
    args = get_args()
    t0 = time.time()

    input_shape = (50, 50, 1)
    _orig_cwd = os.getcwd()
    os.chdir(args.address)
    inputs, targets = read_data(input_shape, debug=False)
    os.chdir(_orig_cwd)
    targets = targets.reshape(-1, 1)
    print('Input shape: {}'.format(inputs.shape))
    print('Target shape: {}'.format(targets.shape))
    print('*** MATCHED-PROTOCOL VARIANT -- no pretraining, no augmentation, '
          '50x50 input, seed=1, batch_size=250, epochs=100 -- matches '
          "Retformer's bayes.py exactly. LEAKY: no patient/group fix "
          "(deliberate, for comparison against other leaky architectures). ***")

    if args.model_name == "vit":
        params = dict()
        params['projection_dim'] = np.arange(32, 64, 2)
        params['patch_size'] = [2, 5, 10]        # must divide 50 evenly
        params['transformer_layers'] = np.arange(1, 5)
        params['num_heads'] = [1, 2, 4, 5]
        params['mlp_head_unit_1'] = [32, 64]
        params['mlp_head_unit_2'] = [16, 32]
        estimator = VanillaViTWrapper()
        label = "MyViT"
    else:
        params = dict()
        params['projection_dim'] = np.arange(32, 64, 2)
        params['patch_size'] = [5]               # grid=10, divisible by window options below
        params['transformer_layers'] = np.arange(1, 5)
        params['num_heads'] = [1, 2, 4, 5]
        params['window_size'] = [2, 5, 10]        # must divide grid (10) evenly
        params['mlp_head_unit_1'] = [32, 64]
        params['mlp_head_unit_2'] = [16, 32]
        estimator = SwinFlatWrapper()
        label = "MySwin"

    kfold_inner = KFold(n_splits=args.n_inner_folds, shuffle=True, random_state=1)
    kfold_outer = KFold(n_splits=args.n_outer_folds, shuffle=True, random_state=1)

    from sklearn.metrics import (precision_score, recall_score, f1_score,
                                  confusion_matrix, roc_curve, roc_auc_score)

    search = BayesSearchCV(estimator=estimator, search_spaces=params,
                            n_jobs=1, n_iter=args.n_iter, cv=kfold_inner, verbose=3)

    accuracies = []
    aucs = []
    precisions = []
    recalls = []
    specificities = []
    f1s = []
    fold_num = 0
    for train_idx, test_idx in kfold_outer.split(inputs, targets):
        fold_num += 1
        print('\n=== {} Outer fold {}/{} ==='.format(label, fold_num, args.n_outer_folds))
        X_train, X_test = inputs[train_idx], inputs[test_idx]
        y_train, y_test = targets[train_idx], targets[test_idx]

        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        y_true = np.asarray(y_test).reshape(-1)
        probs = best_model.model.predict(X_test, verbose=0).reshape(-1)
        y_pred = (probs >= 0.5).astype(int)

        fold_acc = float(np.mean(y_pred == y_true))
        try:
            fold_auc = roc_auc_score(y_true, probs)
        except ValueError:
            fold_auc = 0.0
        fold_precision = precision_score(y_true, y_pred, zero_division=0)
        fold_recall = recall_score(y_true, y_pred, zero_division=0)
        fold_f1 = f1_score(y_true, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        fold_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        fpr, tpr, thresholds = roc_curve(y_true, probs)
        np.savez('{}_roc_fold{}.npz'.format(label, fold_num), fpr=fpr, tpr=tpr, thresholds=thresholds)

        print('Fold {} Accuracy: {:.4f} | AUC: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | '
              'Specificity: {:.4f} | F1: {:.4f}'.format(
                  fold_num, fold_acc, fold_auc, fold_precision, fold_recall, fold_specificity, fold_f1))
        print('Confusion matrix (fold {}): TN={} FP={} FN={} TP={}'.format(fold_num, tn, fp, fn, tp))
        print('Best params this fold: {}'.format(search.best_params_))

        accuracies.append(fold_acc)
        aucs.append(fold_auc)
        precisions.append(fold_precision)
        recalls.append(fold_recall)
        specificities.append(fold_specificity)
        f1s.append(fold_f1)

    accuracies = np.array(accuracies)
    aucs = np.array(aucs)
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    specificities = np.array(specificities)
    f1s = np.array(f1s)

    print('\n{} Accuracy: '.format(label), np.mean(accuracies), np.std(accuracies))
    print('{} AUC: '.format(label), np.mean(aucs), np.std(aucs))
    print('{} Precision: '.format(label), np.mean(precisions), np.std(precisions))
    print('{} Recall/Sensitivity: '.format(label), np.mean(recalls), np.std(recalls))
    print('{} Specificity: '.format(label), np.mean(specificities), np.std(specificities))
    print('{} F1: '.format(label), np.mean(f1s), np.std(f1s))
    print('Runtime: %d seconds' % (time.time() - t0))


if __name__ == "__main__":
    main()
