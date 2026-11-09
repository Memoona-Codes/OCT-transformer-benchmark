# LEAD: a leakage-free framework for adapting retinal foundation models to Alzheimer's disease detection from OCT

Code, patient-disjoint split definitions, prediction files and result tables for the LEAD manuscript.
Every result in the paper can be recomputed from the prediction files in `data/predictions/` without a GPU.

## Folders
| Folder | Contents |
|---|---|
| `1_synthetic_generation/` | VAE training, synthetic image generation, intensity rescaling, quality audit, and the leakage-free per-split generators |
| `2_eyeclip_adaptation/` | EyeCLIP features, CLAHE, the ten adaptation methods, threshold choice, real-anchored score standardisation, slice and patient level evaluation |
| `3_robustness_generalisation/` | CLAHE settings, all-patient cross-validation, second dataset (OCTDL), linear probe, second encoder (RETFound), label efficiency, set-up checks |
| `data/splits/` | `split_seed0.csv` to `split_seed4.csv`: split_seed, set (train/val), class (AD/CO), patient_id, file. `all_splits.csv` joins them |
| `data/predictions/` | Validation scores and thresholds for every method, input, split and seed |
| `results/` | Metric tables produced from the prediction files |

## How leakage is prevented
* Splits are made by patient identity. `final_unified_eval.py` stops with an error if any patient is in both training and validation.
* The decision threshold is chosen on out-of-fold scores from five inner folds, grouped by patient, on the training patients only. It is then applied unchanged to validation.
* Scores are standardised with statistics of the real training images of each fit (real-anchored standardisation), never with validation data.
* CLAHE is applied to each image on its own and has no fitted parameters. EyeCLIP and RETFound stay frozen.
* No hyperparameter was tuned on validation patients.

## Prediction files (`data/predictions/`)
| File | Contents | Paper item |
|---|---|---|
| `predictions_final.csv` | Real images only, 10 methods, raw and CLAHE | Table 2, Figure 3, Supplementary Table S2 |
| `predictions_synth_batch2.csv` | Real + 500 or 1,000 images from the original generator (trained on all 28 patients) | Table 5 (original pool), Supplementary Table S3 |
| `predictions_synth_leakfree_nearcopy.csv` | Real + 500 or 1,000 images from per-split generators, original sampling (near-copy pool) | Table 5 (near-copy pool) |
| `predictions_synth_leakfree_newimage.csv` | Real + 500 or 1,000 images from per-split generators, new-image sampling | Table 5 (new-image pool) |
| `predictions_dosage_batch2.csv` | Standardisation study, real-anchored rule | Table 6, Figure 5 |
| `predictions_dosage_batch2_pooled.csv` | Standardisation study, pooled rule | Table 6, Figure 5 |

Columns: run_time, method, preproc, dosage, split, seed, slice_id, patient_id, condition (DARK/LIGHT), y_true (1 = AD), score (standardised), threshold, patient_threshold.

## Paper items and scripts
| Paper item | Script |
|---|---|
| Figure 2 (CLAHE example) | `2_eyeclip_adaptation/make_clahe_figure.py` |
| Table 2, left (five splits), Figure 3, Supplementary Table S2 | `2_eyeclip_adaptation/final_unified_eval.py`, then `analyze_predictions.py` |
| Table 2, right (all 28 patients, 7 folds × 5 repeats) | `3_robustness_generalisation/exp_B_grouped_cv.py`, `exp_B_summary.py` |
| Linear-probe rows (Tables 2 and 3) | `3_robustness_generalisation/exp_D_linear_probe.py` (`--design main` and `--design grouped`) |
| Table 3, OCTDL rows; per-disease results (Section 3.3) | `3_robustness_generalisation/exp_C_octdl_clahe.py`, `exp_C2_per_disease.py` |
| Table 3, RETFound rows | `3_robustness_generalisation/exp_E_retfound_clahe.py` |
| Figure 4 (label efficiency) | `3_robustness_generalisation/exp_G_label_efficiency.py` |
| CLAHE settings (Section 3.2) | `3_robustness_generalisation/exp_A_clahe_sensitivity.py`, `exp_summary.py` |
| Crop-height check (Section 3.2) | result in `results/robustness/expF_height_check.csv` |
| Supplementary Table S1 (VAE) | `1_synthetic_generation/train_v8_vae.py` |
| Table 4 (quality audit) | `compute_batch2_fid.py`, `check_batch2_class_correlation.py`, `audit_existing_synthetic.py` |
| Supplementary Figure S1 | `1_synthetic_generation/fig2_real_vs_synthetic.py`, `batch2_pearsonr_scatter_pairs.py` |
| Table 5, original pool; Supplementary Table S3 | `2_eyeclip_adaptation/final_synthetic_all_methods.py`, then `analyze_predictions.py` |
| Table 5, leakage-free pools | `1_synthetic_generation/exp_H_vae_per_split.py` (near-copy), `exp_H2_regenerate.py` (new-image), `audit_existing_synthetic.py --leakfree`, then `final_synthetic_all_methods.py --pool ...` |
| Table 6, Figure 5 | `final_synthetic_dosage.py` (real-anchored), `final_synthetic_dosage_pooled_rule.py` (pooled), `plot_standardisation.py` |

## Reproduce the tables from the prediction files (no GPU needed)
    cd 2_eyeclip_adaptation
    python analyze_predictions.py --pred ../data/predictions/predictions_final.csv ../data/predictions/predictions_synth_batch2.csv --out ../results_check --best "TaskRes+SAM"
    python analyze_predictions.py --pred ../data/predictions/predictions_synth_leakfree_newimage.csv --out ../results_check_leakfree --best "TaskRes+SAM"
    python analyze_predictions.py --pred ../data/predictions/predictions_dosage_batch2.csv --out ../results_check_dosage --best "TaskRes+SAM"
    python plot_standardisation.py

## Rerun the experiments (GPU)
1. Download the Bissig OCT images from Dryad (https://doi.org/10.5061/dryad.msbcc2ftc, CC0) and arrange them by the files listed in `data/splits/`.
2. Download OCTDL from Mendeley Data (Kulyabin et al., Scientific Data 11:365, 2024; CC BY 4.0).
3. Obtain the EyeCLIP ViT-B/32 weights from the EyeCLIP release, and the RETFound OCT weights from Hugging Face (`YukunZhou/RETFound_mae_natureOCT`). Images and weights are not redistributed here.
4. Run `1_synthetic_generation/`, then `2_eyeclip_adaptation/`, then `3_robustness_generalisation/` (see the README in each folder).

Experiments ran on one NVIDIA A100 (40 GB). `requirements.txt` lists the main packages; `requirements_full.txt` is the full environment.

## Note on the synthetic images
* **Original pool.** The first VAE was trained on images of all 28 patients, so its synthetic images drew on every validation patient. Intensity rescaling for this pool used one fixed training set (197 B-scans, 20 patients) that contains 5 to 7 of the 8 validation patients of each split (`data/checks.md`). Results with this pool are therefore an optimistic test of augmentation.
* **Leakage-free pools.** For each split, a new VAE with the same architecture was trained on the 20 training patients only, and its images were rescaled with the statistics of those training images. Two sampling settings were used (near-copy and new-image). Validation patients were never used.
* Neither pool gave a consistent gain over real images only (Table 5).
* Real-image results (Tables 2 and 3) use no synthetic images and are not affected by either point.

## Citation
Sami, M., Hajati, F., & Chiong, R. LEAD: a leakage-free framework for adapting retinal foundation models to Alzheimer's disease detection from OCT. Manuscript under review.
