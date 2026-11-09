# Benchmarking Transformer Architectures for Alzheimer's Disease Screening Using Retinal OCT

Code, logs and per-fold prediction files for the paper:

> M. Sami, F. Hajati, and R. Chiong, "Benchmarking Transformer Architectures for Alzheimer's Disease Screening Using Retinal OCT," submitted to IEEE SSCI 2027.

Five transformer architectures (Retformer, ViT, Swin, DeiT-style, MAE-style) are compared for Alzheimer's disease (AD) versus control classification from retinal OCT, under one identical training and evaluation protocol.

## Data
The Bissig OCT dataset is public on Dryad (CC0): https://doi.org/10.5061/dryad.msbcc2ftc
14 AD and 14 age-matched control participants. `models/dataloader.py` builds 1,120 grayscale slices (50 x 50), balanced between classes.
The data are not redistributed here. Download them and set:

    export OCT_DATA_DIR=/path/to/OCT_Dataset

Check that a run prints `Input shape: (1120, 50, 50, 1)`.

## Protocol (identical for all models)
* Trained from scratch: no pretraining, no augmentation
* Nested cross-validation: 3 outer and 3 inner folds, `random_state=1`
* Bayesian optimisation (BayesSearchCV): 15 iterations per model; search dimensions are architecture-specific
* AdamW, 100 epochs, batch size 250

DeiT-style = distillation token trained on true labels, no teacher.
MAE-style = MAE encoder architecture trained end-to-end for classification, no self-supervised pretraining; Flatten readout instead of global average pooling.

## Folders
| Folder | Contents |
|---|---|
| `models/` | Data loader and the training script for each model |
| `analysis/` | Metrics, efficiency, ROC and confusion-matrix figures, Figure 1, and backfill scripts that refit Retformer, ViT and Swin at their selected settings to save predictions |
| `results/logs/` | Full output logs of every final run |
| `results/roc_data/` | Out-of-fold predictions per model and fold (`<model>_fold<k>.npz`) |

## Run
    python models/bayes_niter15_v3.py                    # Retformer
    python models/vit_swin_kfold_nested_matched_v2.py    # ViT and Swin
    python models/deit_niter15_v3.py                     # DeiT-style
    python models/mae_niter15_v3.py                      # MAE-style

Figures and tables from the saved predictions (no GPU needed):

    cd analysis
    python compute_full_metrics.py
    python roc_curve_overlay.py
    python confusion_matrix_display.py

## Results (mean ± SD over 3 outer folds)
| Model | Accuracy (%) | AUC (%) | Parameters |
|---|---|---|---|
| ViT | 91.79 ± 1.76 | 96.80 ± 0.79 | 222,089 |
| Retformer | 90.72 ± 2.54 | 95.59 ± 1.80 | 382,647 |
| Swin | 90.62 ± 0.44 | 96.24 ± 0.82 | 300,743 |
| DeiT-style | 89.56 ± 2.45 | 95.65 ± 1.73 | 223,667 |
| MAE-style | 84.73 ± 2.65 | 91.15 ± 2.16 | 91,097 |

## Important limitations
* **Slice-level cross-validation.** Folds split slices, not participants, so slices from the same participant can appear in both training and test folds. The results compare architectures under one protocol; they do not measure performance on unseen patients. With participant-grouped folds, accuracy falls to about 45 to 65%. A leakage-free follow-up is LEAD: https://github.com/Memoona-Codes/LEAD-retinal-OCT
* **Run-to-run variance.** BayesSearchCV was not fully seeded. Swin was run twice (`swin_niter15_v3b_output.txt` is reported; `swin_niter15_v3_output.txt` is the second run).
* Experiments ran on CPU TensorFlow; a full search takes several hours per model.

## Environment
`environment.yml` (conda): `conda env create -f environment.yml`

## AI assistance
AI tools were used to help write and debug parts of the code. All experiments were designed, run and checked by the authors.

## License
MIT (see `LICENSE`).
