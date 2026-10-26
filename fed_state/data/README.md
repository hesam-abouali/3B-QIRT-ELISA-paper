# Fed-state data

Nine rats: Rat1–Rat3 LZ (`healthy`), Rat4–Rat6 OZ (`pre-diabetic`) and Rat7–Rat9 ZDF (`diabetic`). Times are in minutes from the start of the recording. Each folder is written by the script named in its heading.

## raw/

**`fed_state_raw_data.csv`**: 117 rows, 9 rats x 13 time points (0–30 min, every 2.5 min).

| Column | Content |
|---|---|
| `animal_id`, `rat_type` | rat and phenotype (`Healthy`, `Pre-diabetic`, `Diabetic`) |
| `timepoint` | time (min) |
| `insulin_au`, `c_peptide_au`, `glucagon_au` | normalized QIRT-ELISA signal Fi/F0 (F0: the rat's mean peak area over the recording) |
| `glucose_start_mM`, `glucose_end_mM` | glucometer glucose at 0 and 30 min, repeated on every row of the rat |
| `<hormone>_start_ELISA_pM`, `<hormone>_end_ELISA_pM` | ELISA at 0 and 30 min, repeated on every row of the rat |

## concentrations/ (`src/calibrate.py`)

**`concentration_timeseries.csv`**: 351 rows, 9 rats x 3 hormones x 13 time points: `animal_id`, `rat_type`, `timepoint`, `hormone` and the calibrated concentration `conc_per_rat_pM`.

**`calibration_parameters.csv`**: 27 rows, one per rat and hormone.

| Column | Content |
|---|---|
| `signal_t0`, `signal_t30` | Fi/F0 at 0 and 30 min |
| `elisa_t0_pM`, `elisa_t30_pM` | ELISA at 0 and 30 min |
| `relative_signal_change` | \|signal_t30 − signal_t0\| / max(\|signal_t0\|, \|signal_t30\|) |
| `slope_pM`, `intercept_pM` | the calibration line C = slope · Fi/F0 + intercept through the two anchors |
| `flag` | `ok` for every rat and hormone |

## features/ (`src/extract_features.py`)

| File | Content |
|---|---|
| `fedstate_features_all.csv` | 9 rats x 189 columns: `animal_id`, `rat_type` and the feature table of the SI |
| `fedstate_X.csv` | 9 rats x 161 features: the classification matrix. One column is constant; the classification drops it, which leaves 160 candidate features |
| `fedstate_y.csv` | `rat_type` of each rat, in the row order of `fedstate_X.csv` |
| `fedstate_meta.csv` | `animal_id`, `rat_type` and the summary columns left out of the matrix (`overall_function_score`, `function_classification`, `has_elisa`) |

## classification/ (`src/run_classification.py`, `src/predictive_confidence.py`)

Selectors and classifiers are named as in `src/nested_cv.py`. `fold` 1–9 is the outer fold in which Rat1–Rat9 is held out; `selected_k` is the number of features chosen in the fold's inner loop.

| File | Rows | Columns |
|---|---|---|
| `results_summary.csv` | 81 combinations | `selector`, `classifier`, `inner_cv_mode` (`loo`, or `stratified_3fold` for permutation importance), `macro_f1`, `micro_f1`, `weighted_f1`, `accuracy`, `mean_k` and `std_k` (over the 9 folds), and `prec_`, `rec_`, `f1_<class>` |
| `per_fold_details.csv` | 81 x 9 folds | `selector`, `classifier`, `fold`, `rat_id`, `true_label`, `pred_label`, `correct`, `selected_k` |
| `confusion_matrices.csv` | 81 x 3 x 3 | `selector`, `classifier`, `true_label`, `pred_label`, `count` |
| `selected_features.csv` | one per selected feature | `selector`, `classifier`, `fold`, `selected_k`, `feature` |
| `confidence_by_rat.csv` | 3 recipes x 9 rats | held-out class probabilities of mRMR with LogReg_L2, SVM_linear and KNN: `fold`, `rat_id`, `true_label`, `pred_label`, `prob_pred_label` (class of the highest probability), `probability_prediction_matches_label`, `correct`, `selected_k`, `selected_features` (separated by `; `), `p_healthy`, `p_pre_diabetic`, `p_diabetic`, `max_probability` (predictive confidence), `classification_uncertainty` (1 − `max_probability`) |

## permutation/ (`src/run_permutation_test.py`)

**`per_draw_LogReg_L2.csv`**, **`per_draw_SVM_linear.csv`**: 1000 rows each, the nested cross-validation of mRMR with the recipe's classifier on permuted labels.

| Column | Content |
|---|---|
| `draw` | 0–999 |
| `assignment` | permuted labels of Rat1–Rat9 (`H` healthy, `P` pre-diabetic, `D` diabetic) |
| `computed_at_draw` | first draw with this assignment: each of the 749 distinct assignments is computed once and keeps its multiplicity |
| `macro_f1`, `accuracy` | of the nine held-out predictions |
| `selected_k`, `predictions` | per fold, separated by spaces |

## shap/ (`src/shap_values.py`)

| File | Content |
|---|---|
| `shap_values.csv` | 105 rows: SHAP value (`shap_value`) of each feature selected in the rat's fold (`feature`) for each class (`class`), mRMR + LogReg_L2, with `animal_id`, `true_label`, `pred_label` and `fold` |
| `shap_base_values.csv` | 9 rows: base value E (`base_value`) and predicted probability F (`final_value`) of the rat's true class |
