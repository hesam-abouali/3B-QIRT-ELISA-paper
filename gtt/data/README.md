# GTT data

Nine rats: Rat1–Rat3 LZ (`healthy`), Rat4–Rat6 OZ (`pre-diabetic`) and Rat7–Rat9 ZDF (`diabetic`). Times are in minutes from the glucose injection (t = 0). Blood was sampled at 0, 16, 29.3, 45.3 and 61.3 min (Rat1 has no 45-min sample); the figures plot these times at 0, 15, 30, 45 and 60 min. Each folder is written by the script named in its heading.

## raw/

**`gtt_post_injection.csv`**: 216 rows, 9 rats x 24 samples (0–61.3 min).

| Column | Content |
|---|---|
| `animal_id`, `rat_type` | rat and phenotype |
| `timepoint` | time (min) |
| `insulin_au`, `c_peptide_au`, `glucagon_au` | normalized QIRT-ELISA signal F/Fmax |
| `glucose_mM` | glucometer glucose, at the blood-sampling times (empty otherwise) |
| `insulin_ELISA_pM`, `c_peptide_ELISA_pM`, `glucagon_ELISA_pM` | ELISA, at the blood-sampling times (empty otherwise) |

**`gtt_pre_injection.csv`**: 36 rows, 9 rats x 4 samples (−10 to −2.5 min): `animal_id`, `rat_type`, `timepoint` and the three normalized signals.

## concentrations/ (`src/calibrate.py`, `src/held_out_elisa.py`)

**`concentration_timeseries.csv`**: 252 rows, 9 rats x 28 samples (4 before and 24 after the injection), with five columns per hormone:

| Column | Content |
|---|---|
| `<hormone>_au` | normalized signal |
| `<hormone>_calibration_au` | signal the calibration line is applied to: Hamming-smoothed after the injection, unsmoothed before |
| `<hormone>_ELISA_pM` | ELISA, at the blood-sampling times |
| `<hormone>_predicted_pM` | calibrated concentration (values below 0 set to 0) |
| `<hormone>_raw_predicted_pM` | the unsmoothed signal through the same line |

**`calibration_parameters.csv`**: 27 rows, one per rat and hormone: `n_elisa` (ELISA values the line was fitted to: 5, or 4 for Rat1), `slope_pM` and `intercept_pM` (C = slope · smoothed signal + intercept).

**`held_out_elisa.csv`**: 87 rows, 29 held-out ELISA values per hormone (LZ and ZDF rats): `animal_id`, `rat_type`, `hormone`, `timepoint`, `elisa_pM`, the prediction from the line refitted without this value `predicted_pM` (values below 0 set to 0) and `unfloored_pM`.

**`held_out_elisa_summary.csv`**: 3 rows, one per hormone: `n`, Lin's concordance correlation coefficient `ccc`, mean absolute error `mae_pM`, mean bias (device − ELISA) `bias_pM`, 95% limits of agreement `loa_low_pM` and `loa_high_pM`, and `negative_predictions_set_to_0`.

## features/ (`src/extract_features.py`)

| File | Content |
|---|---|
| `gtt_features_all.csv` | 9 rats x 194 columns: `animal_id`, `rat_type` and the 192 features of the GTT feature table of the SI |
| `gtt_X.csv` | 9 rats x 181 features: the classification matrix, without the 7 clinical indices from the ELISA insulin (`*_elisa`) and 4 constant or empty columns; missing values are left empty |
| `gtt_y.csv` | `rat_type` of each rat, in the row order of `gtt_X.csv` |
| `gtt_meta.csv` | `animal_id` and `rat_type` |
| `feature_families_and_types.csv` | 341 rows, the 181 GTT and 160 fed-state candidate features: `state`, `feature`, `source_category` (GTT feature category A–O, or the fed-state feature type), hormone `family` and feature `type` (Fig. 6 h) |

## classification/ (`src/run_classification.py`)

The six LZ and ZDF rats are scored; the OZ rats are in every training set with the diabetic label. `assignment` 0–19 numbers the 20 ways of labelling three of the six scored rats as diabetic; assignment 19 is the true labelling (Rat7–Rat9 diabetic).

| File | Rows | Columns |
|---|---|---|
| `scores_by_assignment.csv` | 20 assignments x 81 combinations | `assignment`, `selector`, `classifier`, `diabetic_rats`, `true_labels` (assignment 19), `macro_f1`, `accuracy`, `brier_score`, `log_loss`, `mean_predictive_confidence`, `mean_selected_k` |
| `predictions_by_assignment.csv` | 1620 x 6 held-out rats | `assignment`, `selector`, `classifier`, `animal_id`, `label` (1: diabetic in this assignment), `prediction`, `probability_diabetic`, `selected_k` (3, 5 or 10, chosen in the inner loop), `inner_correct_k3`, `inner_correct_k5`, `inner_correct_k10` (inner held-out rats, of 5, predicted correctly with each k) |
| `selected_features_true_labels.csv` | 81 x 6 held-out rats | `selector`, `classifier`, held-out rat `animal_id`, `selected_k`, `selected_features` (separated by `\|`), true labels only |
