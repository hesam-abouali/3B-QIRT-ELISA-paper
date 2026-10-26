# Dynamic Hormonal Responses in Diabetes Progression: QIRT-ELISA In Vivo Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
![Python 3.13](https://img.shields.io/badge/python-3.13-blue)



<div align="center">

Code and in vivo data for the paper Dynamic Hormonal Responses in Diabetes Progression Measured in Real-Time and Continuously by a Microfluidic Platform <br>

From the normalized QIRT-ELISA profiles of insulin, C-peptide, and glucagon to the classification results and the figures.

</div>


---

## Overview

QIRT-ELISA (quantum dot–integrated real-time ELISA) measures insulin, C-peptide, and glucagon continuously in the whole blood sample of conscious, freely moving rats.

Two different experiments were analyzed: (1) a fed-state experiment without a glucose challenge, and (2) a glucose tolerance test (GTT) after an overnight fast.

Each experiment was conducted with three strains of Zucker rats: three lean Zucker (LZ, healthy) rats, three obese normoglycemic (OZ, pre-diabetic) rats, and three Zucker diabetic fatty (ZDF, diabetic) rats.

| Experiment | Recording | Analysis | Folder |
|---|---|---|---|
| Fed state | 0–30 minutes, every 2.5 minutes, conventional ELISA at 0 and 30 minutes | two-point ELISA calibration, feature extraction (160 candidate features), classification of the three phenotypes | [`fed_state/`](fed_state/) |
| Glucose tolerance test (GTT) | -10–60 minutes, every 2.67 minutes, 4 samples before (-10 to 0) and 24 after a glucose injection (0 to 60 minutes); conventional ELISA at 0, 15, 30, 45, and 60 minutes | calibration to ELISA with a held-out ELISA check, feature extraction (181 candidate features), classification of LZ against ZDF with the OZ rats in training only | [`gtt/`](gtt/) |

In both experiments, 81 combinations of 9 feature selectors and 9 classifiers are evaluated by nested leave-one-rat-out cross-validation (nested LOO-CV), with the number of features chosen in the inner loop.

**Main results** (reproduced by the scripts from the files in `data/`):

- **Fed state:** The two reported recipes, mRMR with L2 logistic regression and mRMR with linear SVM, classify 8 of 9 rats correctly (macro-F1 0.886, the best of the 81 combinations). In the label-permutation test with 1000 draws, p = 0.010 and 0.009.
- **GTT:** The held-out ELISA check gives Lin's CCC of 0.92 for insulin, 0.91 for C-peptide, and 0.21 for glucagon. 28 of the 81 combinations classify all six LZ and ZDF rats correctly (mean macro-F1 0.823). In the exact label test over all 20 labelings, p = 0.20 (lowest possible one-tailed p-value = 0.05).

The peak areas behind the fluorescence signals and their normalized profiles are computed with the Streamlit app reported in this paper: [3B-QIRT-ELISA Peak Analyzer](https://github.com/hesam-abouali/3B-QIRT-ELISA).

---

## Repository layout

```
├── requirements.txt
├── fed_state/              fed-state experiment
│   ├── data/               raw profiles and every result used in the paper (columns in data/README.md)
│   ├── src/                calibration, feature extraction and classification
│   ├── figures/            Figs. 3 and 4 and the SI figures
│   └── walkthrough.ipynb   every step with its results
└── gtt/                    glucose tolerance test
    ├── data/               raw profiles and every result used in the paper (columns in data/README.md)
    ├── src/                calibration with the held-out ELISA check, feature extraction and classification
    ├── figures/            Figs. 5 and 6 and the SI figures
    └── walkthrough.ipynb   every step with its results
```

Each experiment's README lists its steps, scripts, and run times; its notebook runs every step and shows the results next to the files in `data/`.

## Quick start

With Python 3.13:

```bash
git clone https://github.com/hesam-abouali/3B-QIRT-ELISA-paper.git
cd 3B-QIRT-ELISA-paper
pip install -r requirements.txt
cd gtt
python src/calibrate.py
python src/extract_features.py
python src/summarize.py
python figures/fig6.py
```

Every script writes to `outputs/` in its experiment folder and reports whether its result matches the file in `data/`. The figure scripts need the Arial font; the notebooks need Jupyter.

---

## Citation

If you use this code or data, please cite:

> Abouali H, et al. Dynamic Hormonal Responses in Diabetes Progression Measured in Real-Time and Continuously by a Microfluidic Platform. Manuscript under preparation.

The QIRT-ELISA platform:

> Abouali H, Srikant S, Al Fattah MF, Barra NG, Chan D, Ban D, Schertzer JD, Poudineh M. A Bead-Based Quantum Dot Immunoassay Integrated with Multi-Module Microfluidics Enables Real-Time Multiplexed Detection of Blood Insulin and Glucagon. *Advanced Science*, 2412185 (2025).

## Contact

**Hesam Abouali**\
[[hesam.abouali@uwaterloo.ca](mailto:hesam.abouali@uwaterloo.ca)] Department of Electrical & Computer Engineering\
University of Waterloo, Waterloo, ON, Canada

**Mahla Poudineh**\
[[mahla.poudineh@uwaterloo.ca](mailto:mahla.poudineh@uwaterloo.ca)] Department of Electrical & Computer Engineering\
University of Waterloo, Waterloo, ON, Canada

## Acknowledgements

This research was supported by a Canadian Institutes of Health Research grant to Dr. Poudineh and Dr. Schertzer (202104PJT-461445-DOL-CENA-79527). <br>
Claude Code (Anthropic) was used to help optimize, organize, and document the code in this repository and to check the reproducibility of the paper's results.

## License

MIT; see [LICENSE](LICENSE).
