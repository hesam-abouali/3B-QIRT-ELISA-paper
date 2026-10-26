# QIRT-ELISA fed-state analysis

Code and data for the fed-state analysis in the paper: 9 rats (LZ, OZ, and ZDF; n=3 per phenotype)<br>

From the normalized QIRT-ELISA profiles of insulin, C-peptide, and glucagon to the classification results and figures.

**[walkthrough.ipynb](walkthrough.ipynb)** goes through every step with its results. The files in `data/` and their columns are described in [data/README.md](data/README.md).

| Step | Script | Data used in the paper |
|---|---|---|
| Raw data | — | `data/raw/fed_state_raw_data.csv` |
| Calibration to concentration | `src/calibrate.py` | `data/concentrations/` |
| Feature extraction | `src/extract_features.py` (features defined in `fed_state_features.py`) | `data/features/` |
| Nested cross-validation of 81 selector–classifier combinations | `src/run_classification.py` (engine: `nested_cv.py`) | `data/classification/` |
| Permutation test | `src/run_permutation_test.py` | `data/permutation/` |
| Class probabilities | `src/predictive_confidence.py` | `data/classification/confidence_by_rat.csv` |
| SHAP values | `src/shap_values.py` | `data/shap/` |
| Figures (Fig. 3 b–d, Fig. 4 d–j and SI) | `figures/fig3.py`, `figures/fig4.py`, `figures/si.py` | — |

Labels in the files: `healthy` = LZ, `pre-diabetic` = OZ, `diabetic` = ZDF.

## How to run

With Python 3.13, from this folder:

```bash
pip install -r ../requirements.txt
python src/calibrate.py
python src/extract_features.py
python src/run_classification.py
```

Every script writes to `outputs/` and, where it applies, reports whether its result matches the file in `data/`. Most take seconds; `run_classification.py` takes 30–50 min, and the full permutation test a few hours (`--draws 0,9` computes single draws). The figure scripts need the Arial font. If SciPy 1.15.3 does not install, a newer SciPy works; the feature extraction then reports rounding-level differences.

## Reproducibility

The results in `data/` were computed with Python 3.13.5, NumPy 2.1.3, pandas 2.2.3, SciPy 1.15.3, scikit-learn 1.6.1 and shap 0.50.0 on macOS arm64. With these versions:

- the calibration and the feature extraction reproduce their files byte for byte;
- the nested cross-validation reproduces all 72 deterministic combinations exactly; the nine `LogReg_L1` combinations (saga solver without a fixed random state) can change between runs and are not among the reported recipes;
- the class probabilities, the SHAP values and the checked permutation draws are identical;
- the figures are pixel-identical to the paper's with matplotlib 3.10.0, except Fig. 4 h–i, which matches with matplotlib 3.10.8 and pandas 3.0.0.
