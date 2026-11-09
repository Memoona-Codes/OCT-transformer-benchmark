"""
v2 of bayes.py's ViT (Retformer)/SVM outer-fold evaluation -- adds
precision, recall/sensitivity, specificity, F1, confusion matrix, and
saved ROC curve arrays per outer fold for the Retformer model, plus
prints the winning BayesSearchCV configuration each fold (already done
in the original, kept here). n_iter explicitly set to 15 to match the
paper's own Table 7 methodology and the confirmed 91.88% Track 2 result --
NOT the n_iter=10 default that may be sitting in the current bayes.py.

Everything else (dataset, folds, seed, params, SVM baseline) is unchanged
from the original script.
"""
import numpy as np
import os
import time
import warnings

import keras
from sklearn.model_selection import KFold, cross_val_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.svm import SVC
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              confusion_matrix, roc_curve, roc_auc_score)

from retformer import ViTWrapper
from dataloader import read_data
warnings.filterwarnings("ignore")

t0 = time.time()

input_shape = (50, 50, 1)

num_inner_folds = 3
num_outer_folds = 3
n_iter = 15   # explicitly matches the paper's Table 7 methodology / confirmed 91.88% run
n_jobs = 1
_orig_cwd = os.getcwd()
os.chdir(os.environ.get('OCT_DATA_DIR', './OCT_Dataset'))
inputs, targets = read_data(input_shape, debug=False)
os.chdir(_orig_cwd)
targets = targets.reshape(-1, 1)
print('Input shape: {}'.format(inputs.shape))
print('Target shape: {}'.format(targets.shape))

vit_params = dict()
vit_params['projection_dim'] = np.arange(32, 64, 2)
vit_params['patch_size'] = np.arange(4, 9)
vit_params['transformer_layers'] = np.arange(1, 5)
vit_params['num_q_heads'] = np.arange(1, 5)
vit_params['num_kv_heads'] = np.arange(1, 5)
vit_params['mlp_head_unit_1'] = [32, 64]
vit_params['mlp_head_unit_2'] = [16, 32]

svc_params = dict()
svc_params['kernel'] = ['linear', 'rbf']
svc_params['C'] = Real(1e-6, 1e+6, prior='log-uniform')
svc_params['gamma'] = Real(1e-6, 1e+1, prior='log-uniform')

kfold_inner = KFold(n_splits=num_inner_folds, shuffle=True, random_state=1)
kfold_outer = KFold(n_splits=num_outer_folds, shuffle=True, random_state=1)

svc_search = BayesSearchCV(estimator=SVC(), search_spaces=svc_params, scoring='recall', n_jobs=-1, n_iter=n_iter, cv=kfold_inner, verbose=3)
vit_search = BayesSearchCV(estimator=ViTWrapper(), search_spaces=vit_params, n_jobs=n_jobs, n_iter=n_iter, cv=kfold_inner, verbose=3)

# --- Retformer (ViTWrapper) outer-fold loop: full metric set ---
vit_accuracies = []
vit_aucs = []
vit_precisions = []
vit_recalls = []
vit_specificities = []
vit_f1s = []
fold_num = 0
for train_idx, test_idx in kfold_outer.split(inputs, targets):
    fold_num += 1
    print('\n=== ViT Outer fold {}/{} ==='.format(fold_num, num_outer_folds))
    X_train, X_test = inputs[train_idx], inputs[test_idx]
    y_train, y_test = targets[train_idx], targets[test_idx]

    vit_search.fit(X_train, y_train)
    best_model = vit_search.best_estimator_

    y_true = np.asarray(y_test).reshape(-1)
    probs = best_model.model.predict(X_test, verbose=0).reshape(-1)  # single sigmoid output
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
    np.savez('retformer_roc_fold{}.npz'.format(fold_num), fpr=fpr, tpr=tpr, thresholds=thresholds)

    print('Fold {} Accuracy: {:.4f} | AUC: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | '
          'Specificity: {:.4f} | F1: {:.4f}'.format(
              fold_num, fold_acc, fold_auc, fold_precision, fold_recall, fold_specificity, fold_f1))
    print('Confusion matrix (fold {}): TN={} FP={} FN={} TP={}'.format(fold_num, tn, fp, fn, tp))
    print('Best params this fold: {}'.format(vit_search.best_params_))

    vit_accuracies.append(fold_acc)
    vit_aucs.append(fold_auc)
    vit_precisions.append(fold_precision)
    vit_recalls.append(fold_recall)
    vit_specificities.append(fold_specificity)
    vit_f1s.append(fold_f1)

vit_accuracies = np.array(vit_accuracies)
vit_aucs = np.array(vit_aucs)
vit_precisions = np.array(vit_precisions)
vit_recalls = np.array(vit_recalls)
vit_specificities = np.array(vit_specificities)
vit_f1s = np.array(vit_f1s)

print('\nMyViT Accuracy: ', np.mean(vit_accuracies), np.std(vit_accuracies))
print('MyViT AUC: ', np.mean(vit_aucs), np.std(vit_aucs))
print('MyViT Precision: ', np.mean(vit_precisions), np.std(vit_precisions))
print('MyViT Recall/Sensitivity: ', np.mean(vit_recalls), np.std(vit_recalls))
print('MyViT Specificity: ', np.mean(vit_specificities), np.std(vit_specificities))
print('MyViT F1: ', np.mean(vit_f1s), np.std(vit_f1s))

svc_scores = cross_val_score(svc_search, inputs.reshape(inputs.shape[0], -1), targets, n_jobs=-1, cv=kfold_outer, verbose=3)
print('SVM Recall: ', np.mean(svc_scores), np.std(svc_scores))

print('Runtime: %d seconds'%(time.time() - t0))
