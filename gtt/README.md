# QIRT-ELISA glucose tolerance test analysis

Code and data of the glucose tolerance test (GTT) analysis of the paper: 9 rats (LZ, OZ and ZDF, 3 each), from the normalized QIRT-ELISA profiles of insulin, C-peptide and glucagon to the classification results and figures.

**[walkthrough.ipynb](walkthrough.ipynb)** goes through every step with its results. The files in `data/` and their columns are described in [data/README.md](data/README.md).

| Step | Script | Data used in the paper |
|---|---|---|
| Raw data | — | `data/raw/` |
| Calibration to concentration | `src/calibrate.py` | `data/concentrations/` |
| Held-out ELISA check (Fig. 5 d–e) | `src/held_out_elisa.py` | `data/concentrations/held_out_elisa*.csv` |
| Feature extraction | `src/extract_features.py` (features defined in `gtt_features.py`) | `data/features/` |
| Nested cross-validation of 81 selector–classifier combinations, for all 20 label assignments | `src/run_classification.py` (engine: `nested_cv.py`) | `data/classification/` |
| Perfect combinations, exact label test, feature recurrence | `src/summarize.py` | — |
| Figures (Fig. 5 b–e, Fig. 6 a–h and SI) | `figures/fig5.py`, `figures/fig6.py`, `figures/si.py` | `data/features/feature_families_and_types.csv` (Fig. 6 h) |

Labels in the files: `healthy` = LZ, `pre-diabetic` = OZ, `diabetic` = ZDF. The classification scores the three LZ and three ZDF rats; the OZ rats are in every training set with the diabetic label.

## How to run

With Python 3.13, from this folder:

```bash
pip install -r ../requirements.txt
python src/calibrate.py
python src/held_out_elisa.py
python src/extract_features.py
python src/summarize.py
```

Every script writes to `outputs/`, and the calibration and feature extraction report whether their results match the files in `data/`. These take seconds. `summarize.py` gives the reported classification results from `data/classification/`: 28 of 81 combinations classify all six rats correctly, mean macro-F1 0.823, exact label test p = 4/20 = 0.20. `run_classification.py` reruns the nested cross-validation: about 1.5 hours for all 20 label assignments, a few minutes with `--true-labels-only`, under a minute with `--selector NAME`. The figure scripts need the Arial font; `fig6.py` (panel h) also reads the fed-state selections in `../fed_state/`.

## Reproducibility

The calibration and the features were computed with Python 3.13.5, NumPy 2.1.3, pandas 2.2.3 and SciPy 1.15.3 on macOS arm64; with these versions both reproduce their files byte for byte. The classification results in `data/classification/` were computed on Linux x86-64 with Python 3.10, NumPy 2.0.2, SciPy 1.13.1, pandas 2.3.3, and scikit-learn 1.6.1; in other environments, a few combinations can select slightly different features.
