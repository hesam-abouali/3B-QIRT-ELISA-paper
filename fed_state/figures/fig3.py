#!/usr/bin/env python3
"""Fig. 3 b–d: glucose and the calibrated insulin, C-peptide and glucagon profiles of the fed state.

Rows: LZ (b), OZ (c) and ZDF (d) rats; columns: glucose, insulin, C-peptide and glucagon. Glucose:
glucometer mean +/- SD of the three rats at 0 and 30 min. Hormones: mean +/- SD of the three calibrated
profiles (line and band) and ELISA mean +/- SD at 0 and 30 min (open circles).

Input   data/raw/fed_state_raw_data.csv, data/concentrations/concentration_timeseries.csv
Output  outputs/figures/fig3_b_to_d.pdf and .png

    python figures/fig3.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "fed_state_raw_data.csv"
CONCENTRATIONS = ROOT / "data" / "concentrations" / "concentration_timeseries.csv"
OUT_DIR = ROOT / "outputs" / "figures"

PHENOTYPES = ["Healthy", "Pre-diabetic", "Diabetic"]
HORMONES = {"insulin": "Insulin", "c_peptide": "C-peptide", "glucagon": "Glucagon"}
ROW_LABELS = {"Healthy": "LZ", "Pre-diabetic": "OZ", "Diabetic": "ZDF"}
COLORS = {"glucose": "#2D5F8A", "insulin": "#800002", "c_peptide": "#408002", "glucagon": "#FD8008"}
Y_LABELS = {"glucose": "Glucose (mM)", "insulin": "[Insulin] (pM)", "c_peptide": "[C-peptide] (pM)",
            "glucagon": "[Glucagon] (pM)"}
Y_LIMITS = {"glucose": {"Healthy": (0, 12), "Pre-diabetic": (0, 16), "Diabetic": (0, 40)},
            "insulin": {"Healthy": (0, 600), "Pre-diabetic": (0, 12000), "Diabetic": (0, 1000)},
            "c_peptide": {"Healthy": (0, 3000), "Pre-diabetic": (0, 18000), "Diabetic": (0, 5000)},
            "glucagon": {"Healthy": (0, 20), "Pre-diabetic": (0, 50), "Diabetic": (0, 20)}}
RC = {"font.family": "Arial", "font.sans-serif": ["Arial"], "font.weight": "normal", "axes.labelweight": "normal",
      "axes.titleweight": "normal", "mathtext.fontset": "custom", "mathtext.rm": "Arial",
      "mathtext.it": "Arial:italic", "mathtext.bf": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42}


def load_profiles():
    """The raw table (normalized signal, glucose and ELISA values) and the calibrated profiles (long format)."""
    return pd.read_csv(RAW), pd.read_csv(CONCENTRATIONS)


def phenotype_matrix(table, column, phenotype, timepoints):
    """Values of the rats of a phenotype (rows, in rat order) at the given times, with their mean and SD."""
    rats = table[table.rat_type == phenotype]
    matrix = rats.pivot(index="animal_id", columns="timepoint", values=column).reindex(columns=timepoints)
    matrix = matrix.to_numpy(dtype=float)
    return np.asarray(timepoints, dtype=float), np.nanmean(matrix, axis=0), np.nanstd(matrix, axis=0, ddof=1), matrix


def anchor_stats(raw, phenotype, start_column, end_column):
    """Mean and SD of the rats of a phenotype at 0 and 30 min (glucose or ELISA)."""
    rats = raw[raw.rat_type == phenotype].sort_values("animal_id").drop_duplicates("animal_id")
    values = rats[[start_column, end_column]].to_numpy(dtype=float)
    return np.array([0.0, 30.0]), np.nanmean(values, axis=0), np.nanstd(values, axis=0, ddof=1)


def panels_b_to_d():
    raw, profiles = load_profiles()
    timepoints = sorted(profiles.timepoint.unique())
    fig, axes = plt.subplots(3, 4, figsize=(28, 18))
    for row, phenotype in enumerate(PHENOTYPES):
        for column, key in enumerate(["glucose", *HORMONES]):
            ax = axes[row, column]
            color = COLORS[key]
            if key == "glucose":
                t, mean, sd = anchor_stats(raw, phenotype, "glucose_start_mM", "glucose_end_mM")
            else:
                t, mean, sd, _ = phenotype_matrix(profiles[profiles.hormone == key], "conc_per_rat_pM", phenotype,
                                                  timepoints)
            ax.plot(t, mean, "-", color=color, lw=2.4, zorder=4)
            ax.fill_between(t, mean - sd, mean + sd, color=color, alpha=0.18, zorder=2)
            if key == "glucose":
                ax.errorbar(t, mean, yerr=sd, fmt="o", color=color, ms=8, mfc="white", mew=1.6, capsize=4, lw=1.4,
                            zorder=6)
            else:
                et, em, es = anchor_stats(raw, phenotype, f"{key}_start_ELISA_pM", f"{key}_end_ELISA_pM")
                ax.errorbar(et, em, yerr=es, fmt="o", color="#111111", ms=8, mfc="white", mew=1.6, capsize=4, lw=1.4,
                            zorder=6)
            ax.set_ylabel(Y_LABELS[key], fontsize=26.4)
            ax.yaxis.set_label_coords(-0.24 if key in ("insulin", "c_peptide") else -0.17, 0.5)
            ax.set_xlabel("Time (min)", fontsize=33)
            ax.set_xlim(-2, 32)
            ax.set_xticks([0, 10, 20, 30])
            ax.set_ylim(Y_LIMITS[key][phenotype])
            if key == "insulin" and phenotype == "Pre-diabetic":
                ax.set_yticks([0, 4000, 8000, 12000])
            ax.tick_params(axis="both", which="major", labelsize=30)
            for spine in ax.spines.values():
                spine.set_linewidth(4.0)
                spine.set_color("black")
            ax.text(0.10, 0.95, ROW_LABELS[phenotype], transform=ax.transAxes, ha="left", va="top", fontsize=30,
                    color="#111111", zorder=7)
    fig.subplots_adjust(left=0.065, right=0.998, bottom=0.065, top=0.99, wspace=0.34, hspace=0.30)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        path = OUT_DIR / f"fig3_b_to_d.{extension}"
        fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
        print(path)
    plt.close(fig)


def main():
    with plt.rc_context(RC):
        panels_b_to_d()


if __name__ == "__main__":
    main()
