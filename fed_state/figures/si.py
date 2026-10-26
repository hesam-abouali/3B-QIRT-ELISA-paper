#!/usr/bin/env python3
"""SI figures of the fed state.

normalized_profiles   normalized profile and ELISA values of each rat (outputs/figures/si_normalized_profiles)
individual_profiles   glucose and calibrated profiles of each rat (si_individual_profiles)
macro_f1_heatmap      macro-F1 of the 81 combinations (si_macro_f1_heatmap)
permutation_test      null distribution and its cumulative distribution (si_permutation_null, si_permutation_ecdf)
rat4_shap             SHAP attributions of Rat4, the misclassified OZ rat (si_rat4_shap)

Every figure is written as .pdf and .png.

Input   data/raw/fed_state_raw_data.csv, data/concentrations/concentration_timeseries.csv,
        data/classification/results_summary.csv, per_fold_details.csv, data/permutation/per_draw_<recipe>.csv,
        data/shap/shap_values.csv

    python figures/si.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MultipleLocator
from sklearn.metrics import f1_score

from fig3 import HORMONES, PHENOTYPES, load_profiles, phenotype_matrix

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "figures"
BORDER_LINEWIDTH = 1.8
PROFILE_COLORS = {"glucose": "#2D5F8A", "insulin": "#800002", "c_peptide": "#408002", "glucagon": "#FDA909"}
ELISA_GREY = "#555555"


def profile_axis(ax, y_major, y_minor, minor=True):
    """Black frame and ticks of the profile panels."""
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("black")
    if minor:
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(which="minor", length=3.0, width=0.8)
    ax.tick_params(which="major", labelsize=12, length=5.0, width=1.0)
    ax.tick_params(axis="y", which="major", length=y_major)
    ax.tick_params(axis="y", which="minor", length=y_minor)


# ---- normalized profiles and ELISA values of each rat ----

RAT_STYLES = [("#2a78d6", "o"), ("#d55181", "s"), ("#4a3aa7", "^")]   # first, second and third rat of a phenotype
COLUMN_TITLES = {"Healthy": "Healthy, LZ", "Pre-diabetic": "Pre-diabetic, OZ", "Diabetic": "Diabetic, ZDF"}
FONT_SCALE = 1.44
NORMALIZED_RC = {"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.weight": "normal",
                 "axes.labelweight": "normal", "axes.titleweight": "normal", "axes.formatter.use_mathtext": False,
                 "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                 "mathtext.bf": "Arial:bold", "pdf.fonttype": 42, "ps.fonttype": 42}


def label_positions(values, low, high, min_fraction=0.075):
    """Label heights at least min_fraction of the range apart, in the order of the values, within [low, high]."""
    gap = (high - low) * min_fraction
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = [0.0] * len(values)
    previous = low - gap
    for i in order:
        placed[i] = previous = max(values[i], previous + gap)
    overflow = placed[order[-1]] - high
    if overflow > 0:
        placed = [y - overflow for y in placed]
    return [min(max(y, low), high) for y in placed]


def rat_label(ax, rat, xy, xytext, ha, color):
    ax.annotate(rat, xy=xy, xytext=xytext, textcoords="data", fontsize=10.0 * FONT_SCALE, color="0.15", va="center",
                ha=ha, clip_on=False, zorder=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.6},
                arrowprops={"arrowstyle": "-", "color": color, "lw": 1.3, "shrinkA": 1.5, "shrinkB": 3.0})


def elisa_values(raw, key, phenotype):
    """(rat, ELISA at 0 min, ELISA at 30 min) of each rat of a phenotype."""
    rows = raw[raw.rat_type == phenotype].sort_values(["animal_id", "timepoint"]).drop_duplicates("animal_id")
    return [(row.animal_id, float(getattr(row, f"{key}_start_ELISA_pM")), float(getattr(row, f"{key}_end_ELISA_pM")))
            for row in rows.itertuples()]


def normalized_profiles():
    """Normalized device profile of each rat (left axis, Fi/F0) and its ELISA values at 0 and 30 min (right axis,
    pM), with the phenotype mean +/- SD of the profiles and the mean of the ELISA values; each rat has its own colour
    and marker."""
    plt.rcParams.update(NORMALIZED_RC)
    raw, _ = load_profiles()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig = normalized_figure(raw)
        path = OUT_DIR / f"si_normalized_profiles.{extension}"
        fig.savefig(path, dpi=600, bbox_inches="tight")
        print(path)
        plt.close(fig)


def normalized_figure(raw):
    timepoints = sorted(raw.timepoint.unique())
    ticks = np.arange(0, 31, 5)
    fig, axes = plt.subplots(3, 3, figsize=(16, 11.6), constrained_layout=True)
    for row, (key, label) in enumerate(HORMONES.items()):
        color = PROFILE_COLORS[key]
        for column, phenotype in enumerate(PHENOTYPES):
            ax = axes[row, column]
            profile_axis(ax, 3.0, 1.8)
            ax.tick_params(axis="both", which="both", colors="black")
            ax.tick_params(axis="both", which="major", labelsize=12 * FONT_SCALE)
            if row == 0:
                ax.set_title(COLUMN_TITLES[phenotype], fontsize=16 * FONT_SCALE, fontweight="normal", pad=6)
            if column == 0:
                ax.set_ylabel(f"[{label}] ($F_i/F_0$)", fontsize=13 * FONT_SCALE)
            if row == 2:
                ax.set_xlabel("Time (min)", fontsize=13 * FONT_SCALE)
            ax.set_ylim(0, 2.5)
            ax.set_xlim(min(timepoints) - 6.5, max(timepoints) + 6.5)
            ax.set_xticks(ticks)

            # left axis: phenotype mean +/- SD and the profile of each rat
            t, mean, sd, matrix = phenotype_matrix(raw, f"{key}_au", phenotype, timepoints)
            rats = sorted(raw.loc[raw.rat_type == phenotype, "animal_id"].unique())
            ax.fill_between(t, mean - sd, mean + sd, color=color, alpha=0.10, lw=0, zorder=1)
            ax.plot(t, mean, color=color, lw=4.2, alpha=0.9, zorder=2, solid_capstyle="round")
            ax.scatter(t, mean, facecolors="white", edgecolors=color, s=150.0, linewidths=1.6, zorder=2.1)
            for rat, values, (rat_color, marker) in zip(rats, matrix, RAT_STYLES):
                ax.plot(t, values, color=rat_color, lw=1.5, alpha=0.95, marker=marker, markersize=4.2,
                        markerfacecolor=rat_color, markeredgecolor="white", markeredgewidth=0.5, zorder=3,
                        solid_joinstyle="round")
            for rat, values, (rat_color, _), y_text in zip(rats, matrix, RAT_STYLES,
                                                           label_positions(list(matrix[:, -1]), 0.1, 2.4)):
                rat_label(ax, rat, (float(t[-1]), float(values[-1])), (max(timepoints) + 1.4, y_text), "left",
                          rat_color)

            # right axis: ELISA values of each rat and their mean
            ax2 = ax.twinx()
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(ticks)
            points = elisa_values(raw, key, phenotype)
            values = [v for _, v0, v30 in points for v in (v0, v30)]
            low, high = min(values), max(values)
            pad = max((high - low) * 0.20, high * 0.05)
            ax2.set_ylim(max(0, low - pad), high + pad)
            for (rat, v0, v30), (rat_color, marker) in zip(points, RAT_STYLES):
                ax2.plot([0, 30], [v0, v30], color=rat_color, lw=1.4, alpha=0.9, linestyle=(0, (4, 2.5)), zorder=5)
                ax2.scatter([0, 30], [v0, v30], color=rat_color, s=95, marker=marker, edgecolors="white",
                            linewidths=1.2, zorder=6.5)
            means = [float(np.mean([p[1] for p in points])), float(np.mean([p[2] for p in points]))]
            ax2.plot([0, 30], means, color=ELISA_GREY, lw=2.0, linestyle=(0, (5, 3)), alpha=0.85, zorder=5.5)
            ax2.scatter([0, 30], means, color=ELISA_GREY, s=80, marker="D", edgecolors="white", linewidths=1.0,
                        zorder=6)
            y_low, y_high = ax2.get_ylim()
            for (rat, v0, _), (rat_color, _), y_text in zip(points, RAT_STYLES,
                                                            label_positions([p[1] for p in points], y_low, y_high)):
                rat_label(ax2, rat, (0, v0), (min(timepoints) - 1.4, y_text), "right", rat_color)
            if column == 2:
                ax2.set_ylabel("ELISA (pM)", fontsize=11 * FONT_SCALE, color="black")
            ax2.tick_params(axis="x", which="both", colors="black")
            ax2.tick_params(axis="y", labelsize=9 * FONT_SCALE, colors="black")
            ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax2.tick_params(axis="y", which="major", length=3.0, width=1.0, colors="black")
            ax2.tick_params(axis="y", which="minor", length=1.8, width=0.8, colors="black")
            ax2.spines["right"].set_color("black")
            ax2.spines["top"].set_visible(False)
            if column != 2:
                ax2.tick_params(axis="y", labelright=False)
                ax2.set_yticklabels([])

    ink = "0.30"
    handles = [Line2D([], [], color=ink, lw=1.5, marker="o", markersize=4.2, markeredgecolor="white",
                      markeredgewidth=0.5, label=r"Device, individual rat ($F_i/F_0$, left axis)"),
               Line2D([], [], color=ink, lw=1.4, linestyle=(0, (4, 2.5)), marker="o", markersize=9.5,
                      markeredgecolor="white", markeredgewidth=1.2,
                      label="ELISA cross validation, individual rat (pM, right axis)"),
               Line2D([], [], color="0.45", lw=4.2, marker="o", markersize=11, markerfacecolor="white",
                      markeredgecolor="0.45", markeredgewidth=1.6, label="Device, group mean"),
               Patch(facecolor="0.45", alpha=0.18, label="± SD (device)"),
               Line2D([], [], color=ELISA_GREY, lw=2.0, linestyle=(0, (5, 3)), marker="D", markersize=8.5,
                      markeredgecolor="white", markeredgewidth=1.0, label="ELISA cross validation, group mean")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=10.5 * FONT_SCALE, frameon=False,
               handlelength=2.8, columnspacing=1.8)
    return fig


# ---- glucose and calibrated profiles of each rat ----

RAT_GROUPS = [("LZ", ("Rat1", "Rat2", "Rat3")), ("OZ", ("Rat4", "Rat5", "Rat6")), ("ZDF", ("Rat7", "Rat8", "Rat9"))]
ROW_AXES = [
    {"glucose": ((0, 12), 6, 3), "insulin": ((0, 600), 200, 100), "c_peptide": ((0, 4000), 2000, 1000),
     "glucagon": ((0, 20), 10, 5)},
    {"glucose": ((0, 16), 8, 4), "insulin": ((0, 12000), 4000, 2000), "c_peptide": ((4000, 18000), 4000, 2000),
     "glucagon": ((0, 50), 25, 5)},
    {"glucose": ((0, 40), 20, 10), "insulin": ((0, 1000), 500, 250), "c_peptide": ((0, 4000), 2000, 1000),
     "glucagon": ((0, 20), 10, 5)},
]
INSETS = {("Rat4", "insulin"): [0.49, 0.49, 0.48, 0.39], ("Rat4", "c_peptide"): [0.49, 0.49, 0.48, 0.39],
          ("Rat4", "glucagon"): [0.49, 0.49, 0.48, 0.39], ("Rat9", "c_peptide"): [0.49, 0.18, 0.48, 0.40]}

LABEL_CORNERS = {("Rat2", "glucose"): "top-right", ("Rat6", "insulin"): "top-right",
                 ("Rat6", "c_peptide"): "top-right", ("Rat8", "glucagon"): "top-right",
                 ("Rat4", "glucose"): "bottom-left", ("Rat7", "glucose"): "bottom-left",
                 ("Rat8", "glucose"): "bottom-left", ("Rat9", "c_peptide"): "bottom-left"}
CORNERS = {"top-left": (0.03, 0.975, "left", "top"), "top-right": (0.97, 0.975, "right", "top"),
           "bottom-left": (0.03, 0.04, "left", "bottom")}
INDIVIDUAL_HEIGHT = 11.0 * 4.0 / 3.0 + 0.25
INDIVIDUAL_RC = {"font.family": "Arial", "font.sans-serif": ["Arial"], "font.weight": "normal",
                 "axes.labelweight": "normal", "axes.titleweight": "normal", "pdf.fonttype": 42, "ps.fonttype": 42}


def bracketing_ticks(low, high, max_intervals=4):
    """Round tick step (1, 2, 2.5 or 5 x 10^n) whose outer ticks enclose [low, high] most tightly in at most
    max_intervals intervals (ties: the larger step)."""
    best = None
    exponent = int(np.floor(np.log10(max(high - low, 1e-9)))) - 1
    for power in range(exponent, exponent + 4):
        for mantissa in (1.0, 2.0, 2.5, 5.0):
            step = mantissa * 10.0 ** power
            lower, upper = np.floor(low / step) * step, np.ceil(high / step) * step
            if 1 <= round((upper - lower) / step) <= max_intervals:
                key = (upper - lower, -step)
                if best is None or key < best[0]:
                    best = (key, lower, upper, step)
    _, lower, upper, step = best
    return float(lower), float(upper), float(step)


def inset(ax, timepoints, values, bounds, anchor_times, elisa, color):
    """Close-range view of a low profile, its y axis from one round tick to another."""
    box = ax.inset_axes(bounds)
    box.set_facecolor("white")
    box.patch.set_alpha(0.96)
    box.set_zorder(20)
    profile_axis(box, 1.6, 0.9, minor=False)
    for spine in box.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color("0.25")
    box.plot(timepoints, values, color=color, lw=0.9, zorder=3)
    box.scatter(timepoints, values, facecolors="white", edgecolors=color, s=5.5, linewidths=0.4, zorder=4)
    box.plot(anchor_times, elisa, color=ELISA_GREY, lw=0.55, linestyle="--", zorder=5)
    box.scatter(anchor_times, elisa, color=ELISA_GREY, marker="D", s=8, edgecolors="white", linewidths=0.25, zorder=6)
    box.set_xlim(-2, 32)
    data = np.asarray([*values, *elisa], dtype=float)
    lower, upper, step = bracketing_ticks(float(data.min()), float(data.max()))
    margin = 0.03 * (upper - lower)
    box.set_ylim(lower - margin, upper + margin)
    box.set_yticks(np.arange(lower, upper + 0.5 * step, step))
    box.xaxis.set_major_locator(MultipleLocator(15))
    box.xaxis.set_minor_locator(MultipleLocator(5))
    box.yaxis.set_minor_locator(AutoMinorLocator(2))
    box.tick_params(which="major", labelsize=4.5, length=1.7, width=0.45, pad=0.8)
    box.tick_params(which="minor", length=0.9, width=0.35)


def individual_axis(ax):
    profile_axis(ax, 2.4, 1.4)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
    ax.tick_params(which="major", labelsize=10.08, length=2.8, width=0.75, pad=1.5)
    ax.tick_params(which="minor", length=1.5, width=0.6)
    ax.set_xlim(-2, 32)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.set_xlabel("Time (min)", fontsize=11.088, labelpad=1.0)


def individual_profiles():
    """Glucose and the calibrated insulin, C-peptide and glucagon profiles of each rat: glucometer glucose and ELISA
    values (dashed line, diamonds) at 0 and 30 min. Axis ranges are shared within each phenotype; the low profiles of
    Rat4 and Rat9 C-peptide have an enlarged inset."""
    plt.rcParams.update(INDIVIDUAL_RC)
    raw, profiles = load_profiles()
    fig = plt.figure(figsize=(7.2, INDIVIDUAL_HEIGHT), facecolor="white")
    groups = fig.add_gridspec(3, 1, left=0.115, right=0.995, bottom=(0.068 * 11.0 + 0.25) / INDIVIDUAL_HEIGHT,
                              top=1.0 - (1.0 - 0.992) * 11.0 / INDIVIDUAL_HEIGHT, hspace=0.18)
    for g, (group, rats) in enumerate(RAT_GROUPS):
        cells = groups[g, 0].subgridspec(4, 3, hspace=0.62, wspace=0.13)
        for row, key in enumerate(["glucose", *HORMONES]):
            color = PROFILE_COLORS[key]
            for column, rat in enumerate(rats):
                ax = fig.add_subplot(cells[row, column])
                individual_axis(ax)
                rat_raw = raw[raw.animal_id == rat].sort_values("timepoint")
                first = rat_raw.iloc[0]
                anchor_times = [float(rat_raw.timepoint.min()), float(rat_raw.timepoint.max())]
                if key == "glucose":
                    glucose = [float(first.glucose_start_mM), float(first.glucose_end_mM)]
                    ax.plot(anchor_times, glucose, color=color, lw=1.45, zorder=3)
                    ax.scatter(anchor_times, glucose, facecolors="white", edgecolors=color, s=18, linewidths=0.8,
                               zorder=4)
                    ylabel = "[Glucose]\n(mM)"
                else:
                    rows = profiles[(profiles.animal_id == rat) & (profiles.hormone == key)].sort_values("timepoint")
                    timepoints, values = rows.timepoint.to_numpy(float), rows.conc_per_rat_pM.to_numpy(float)
                    ax.plot(timepoints, values, color=color, lw=1.45, zorder=3)
                    ax.scatter(timepoints, values, facecolors="white", edgecolors=color, s=12, linewidths=0.65,
                               zorder=4)
                    elisa = [float(first[f"{key}_start_ELISA_pM"]), float(first[f"{key}_end_ELISA_pM"])]
                    ax.plot(anchor_times, elisa, color=ELISA_GREY, lw=0.9, linestyle="--", zorder=5)
                    ax.scatter(anchor_times, elisa, color=ELISA_GREY, marker="D", s=18, edgecolors="white",
                               linewidths=0.45, zorder=6)
                    if (rat, key) in INSETS:
                        inset(ax, timepoints, values, INSETS[rat, key], anchor_times, elisa, color)
                    ylabel = f"[{HORMONES[key]}]\n(pM)"
                limits, major, minor = ROW_AXES[g][key]
                ax.set_ylim(*limits)
                ax.yaxis.set_major_locator(MultipleLocator(major))
                ax.yaxis.set_minor_locator(MultipleLocator(minor))
                if column == 0:
                    ax.set_ylabel(ylabel, fontsize=11.088, labelpad=3.0)
                    frame = ax.get_position()
                    ax.yaxis.set_label_coords(0.055, frame.y0 + frame.height / 2.0, transform=fig.transFigure)
                else:
                    ax.tick_params(axis="y", which="both", labelleft=False)
                x, y, ha, va = CORNERS[LABEL_CORNERS.get((rat, key), "top-left")]
                ax.text(x, y, f"{group} {rat.replace('Rat', 'Rat ')}", transform=ax.transAxes, ha=ha, va=va,
                        fontsize=8.5, color="#111111", zorder=25)

    handles = [Line2D([], [], color=PROFILE_COLORS[key], lw=1.45, marker="o", markersize=3.6, markerfacecolor="white",
                      markeredgecolor=PROFILE_COLORS[key], markeredgewidth=0.65, label=f"{label}-QIRT-ELISA")
               for key, label in HORMONES.items()]
    handles.append(Line2D([], [], color=PROFILE_COLORS["glucose"], lw=1.45, marker="o", markersize=4.2,
                          markerfacecolor="white", markeredgecolor=PROFILE_COLORS["glucose"], markeredgewidth=0.8,
                          label="Glucose glucometer"))
    handles.append(Line2D([], [], color=ELISA_GREY, lw=0.9, linestyle="--", marker="D", markersize=3.3,
                          markerfacecolor=ELISA_GREY, markeredgecolor="white", markeredgewidth=0.45,
                          label="ELISA calibration point"))

    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.115, 0.004, 0.995 - 0.115, 0.02), mode="expand",
               ncol=5, frameon=True, framealpha=1.0, edgecolor="0.65", fontsize=7.0, handlelength=1.5,
               handletextpad=0.4, columnspacing=0.6, borderaxespad=0.0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        path = OUT_DIR / f"si_individual_profiles.{extension}"
        fig.savefig(path, dpi=600, facecolor="white")
        print(path)
    plt.close(fig)


# ---- macro-F1 of the 81 combinations ----

SUMMARY = ROOT / "data" / "classification" / "results_summary.csv"

SELECTORS = ["anova", "elastic_net", "kruskal", "mrmr", "mutual_info", "perm_importance", "relieff",
             "rf_importance", "rfe"]
CLASSIFIERS = ["DecisionTree", "KNN", "LDA", "LogReg_L1", "LogReg_L2", "NaiveBayes", "RandomForest",
               "SVM_linear", "SVM_rbf"]
REPORTED = [("mrmr", "LogReg_L2"), ("mrmr", "SVM_linear")]
SCALE = 1.5
CMAP = LinearSegmentedColormap.from_list("blue", ["#f7fbff", "#e5f0fa", "#c9dff2", "#94bfdc", "#5f98c3"])


def macro_f1_heatmap():
    """Macro-F1 of the 81 selector-classifier combinations of the fed-state classification.

    Rows are the feature selectors and columns the classifiers; the two reported recipes, mRMR + LogReg_L2
    and mRMR + SVM_linear, are framed and printed in bold."""
    df = pd.read_csv(SUMMARY)
    pivot = df.pivot_table(index="selector", columns="classifier", values="macro_f1",
                           aggfunc="max").reindex(index=SELECTORS, columns=CLASSIFIERS)
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8 * SCALE,
        "axes.labelsize": 9 * SCALE,
        "axes.titlesize": 10 * SCALE,
        "xtick.labelsize": 7 * SCALE,
        "ytick.labelsize": 7 * SCALE,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
    })

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    im = ax.imshow(pivot.values, cmap=CMAP, vmin=0, vmax=1, aspect="auto")
    for i, selector in enumerate(SELECTORS):
        for j, classifier in enumerate(CLASSIFIERS):
            value = pivot.loc[selector, classifier]
            if np.isnan(value):
                continue
            reported = (selector, classifier) in REPORTED
            ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                    fontsize=(7.0 if reported else 6.2) * SCALE, fontweight="bold" if reported else "normal",
                    color="white" if value > 0.72 else "#1f2933")
    for selector, classifier in REPORTED:
        ax.add_patch(patches.Rectangle((CLASSIFIERS.index(classifier) - 0.5, SELECTORS.index(selector) - 0.5),
                                       1, 1, fill=False, edgecolor="black", linewidth=BORDER_LINEWIDTH))
    ax.add_patch(patches.Rectangle((-0.5, -0.5), len(CLASSIFIERS), len(SELECTORS), fill=False,
                                   edgecolor="black", linewidth=BORDER_LINEWIDTH))

    ax.set_xticks(range(len(CLASSIFIERS)))
    ax.set_xticklabels(CLASSIFIERS, rotation=45, ha="right")
    ax.set_yticks(range(len(SELECTORS)))
    ax.set_yticklabels(SELECTORS)
    ax.set_xlabel("Classifier")
    ax.set_ylabel("Feature selector")
    ax.set_title("")
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label("Macro-F1", fontsize=9 * SCALE * 0.75)
    cbar.ax.tick_params(labelsize=7 * SCALE)
    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(BORDER_LINEWIDTH)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        path = OUT_DIR / f"si_macro_f1_heatmap.{extension}"
        fig.savefig(path, dpi=600, bbox_inches="tight")
        print(path)
    plt.close(fig)


# ---- label-permutation test ----

FOLDS = ROOT / "data" / "classification" / "per_fold_details.csv"
PERMUTATION = ROOT / "data" / "permutation"

RECIPES = [
    {"classifier": "LogReg_L2", "label": "mRMR + LogReg L2", "short_label": "LogReg L2", "color": "#7B5BC7"},
    {"classifier": "SVM_linear", "label": "mRMR + linear SVM", "short_label": "linear SVM", "color": "#D9669A"},
]
TEXT = "#111111"


def load_results():
    """Null macro-F1 of every draw, observed macro-F1, null mean, draws reaching the observed value and p."""
    folds = pd.read_csv(FOLDS)
    results = []
    for recipe in RECIPES:
        draws = pd.read_csv(PERMUTATION / f"per_draw_{recipe['classifier']}.csv").sort_values("draw")
        assert draws["draw"].tolist() == list(range(1000))
        null = draws["macro_f1"].to_numpy(dtype=float)
        fold = folds[(folds.selector == "mrmr") & (folds.classifier == recipe["classifier"])]
        observed = f1_score(fold.true_label.to_numpy(), fold.pred_label.to_numpy(), average="macro",
                            labels=["healthy", "pre-diabetic", "diabetic"], zero_division=0)
        reached = int(np.sum(null >= observed - 1e-12))
        results.append({**recipe, "null": null, "observed": observed, "mean": float(np.mean(null)),
                        "reached": reached, "p_value": (reached + 1) / (len(null) + 1)})
    return results


def style_axis(ax):
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(BORDER_LINEWIDTH)
        spine.set_color("black")
    ax.tick_params(color="black", labelcolor=TEXT, width=1.0)


def save(fig, stem):
    for extension in ("pdf", "png"):
        path = OUT_DIR / f"{stem}.{extension}"
        fig.savefig(path, bbox_inches="tight")
        print(path)
    plt.close(fig)


def plot_null_distribution(results):
    fig, ax = plt.subplots(figsize=(8.2, 7.6))
    for result in results:
        null = result["null"]
        ax.hist(null, bins=np.linspace(0, 1.0, 26), weights=np.ones_like(null) / len(null),
                color=to_rgba(result["color"], 0.42), edgecolor="black", linewidth=1.3, label=result["label"])
    observed = results[0]["observed"]
    ax.axvline(observed, color=TEXT, linewidth=2.2, label=f"Observed F1 = {observed:.3f}")
    for result in results:
        ax.axvline(result["mean"], color=result["color"], linestyle="--", linewidth=2.0,
                   label=f"{result['short_label']} null mean = {result['mean']:.3f}")
    ax.set_xlabel("Null macro-F1 under shuffled labels")
    ax.set_ylabel("Fraction of permutations")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 0.34)
    style_axis(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=1, frameon=False)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.96, bottom=0.34)
    save(fig, "si_permutation_null")


def plot_ecdf(results):
    fig, ax = plt.subplots(figsize=(8.6, 6.8))
    for result in results:
        null = np.sort(result["null"])
        p_label = f"p = {result['p_value']:.4f}, {result['reached']}/{len(null)} reached"
        ax.step(null, np.arange(1, len(null) + 1) / len(null), where="post", color=result["color"],
                linewidth=2.3, label=f"{result['label']} ({p_label})")
        ax.scatter([result["observed"]], [np.mean(null <= result["observed"])], color=result["color"],
                   edgecolor="black", linewidth=BORDER_LINEWIDTH, s=78, zorder=3)
    observed = results[0]["observed"]
    ax.axvline(observed, color=TEXT, linewidth=2.0, label=f"Observed F1 = {observed:.3f}")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Macro-F1")
    ax.set_ylabel("Empirical cumulative probability")
    style_axis(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.96, bottom=0.34)
    save(fig, "si_permutation_ecdf")


def permutation_test():
    """Label-permutation test of mRMR + LogReg_L2 and mRMR + SVM_linear (B = 1000).

    a  null distribution of macro-F1 under shuffled labels, with the observed macro-F1 and the null means
    b  empirical cumulative distribution of the null macro-F1, with p and the number of draws that reach
       the observed macro-F1

    The two panels are written as separate files."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 19,
        "axes.labelsize": 21,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 14,
        "axes.grid": False,
        "font.weight": "normal",
    })
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()
    plot_null_distribution(results)
    plot_ecdf(results)


# ---- SHAP attributions of Rat4 ----

SHAP_VALUES = ROOT / "data" / "shap" / "shap_values.csv"

FONT = "Arial"
PANEL_TITLE_SIZE = 21.0
LABEL_SIZE = 18.9
TICK_SIZE = 16.8
LEGEND_SIZE = 16.8
POSITIVE = "#FF145F"
NEGATIVE = "#1494FB"
FEATURE_LABELS = {
    "glucagon_peak_total_excursion": "Glucagon peak excursion",
    "glucose_avg_mM": "Glucose avg (mM)",
    "glucagon_auc_glucose_corrected": "Glucagon AUC/glucose",
}
TRUE_ORDER = ["glucagon_peak_total_excursion", "glucagon_auc_glucose_corrected", "glucose_avg_mM"]
PREDICTED_ORDER = ["glucagon_peak_total_excursion", "glucose_avg_mM", "glucagon_auc_glucose_corrected"]


def tick_label(value, _position):
    return "0.0" if abs(value) < 1e-12 else f"{value:.2f}".rstrip("0").rstrip(".")


def draw_panel(ax, values, feature_order, title, title_color):
    shap_values = np.array([values[feature] for feature in feature_order])
    y = np.arange(len(feature_order))
    ax.barh(y, shap_values, color=[POSITIVE if value > 0 else NEGATIVE for value in shap_values],
            edgecolor="#202020", linewidth=0.65, height=0.58, zorder=2)
    ax.axvline(0, color="black", linewidth=1.15, zorder=3)
    ax.set_xlim(-0.15, 0.15)
    ax.set_xticks([-0.10, -0.05, 0.0, 0.05, 0.10])
    ax.xaxis.set_major_formatter(FuncFormatter(tick_label))
    ax.set_yticks(y)
    ax.set_yticklabels([FEATURE_LABELS[feature] for feature in feature_order], fontsize=TICK_SIZE, fontname=FONT)
    ax.invert_yaxis()
    ax.set_xlabel("SHAP value", fontsize=LABEL_SIZE, fontname=FONT)
    ax.set_title(title, fontsize=PANEL_TITLE_SIZE, fontname=FONT, color=title_color, pad=10)
    ax.tick_params(axis="both", labelsize=TICK_SIZE, width=1.2, length=4.5)
    plt.setp(ax.get_xticklabels(), fontname=FONT)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.35)


def rat4_shap():
    """Fold-specific SHAP attributions of Rat4, the misclassified OZ rat (mRMR + LogReg_L2).

    The three features selected in its fold are shown with their attributions to the true class (OZ, left)
    and to the predicted class (LZ, right)."""
    df = pd.read_csv(SHAP_VALUES)
    rat = df[df["animal_id"] == "Rat4"]
    assert (rat["true_label"].iloc[0], rat["pred_label"].iloc[0]) == ("pre-diabetic", "healthy")
    values = {c: dict(zip(rat[rat["class"] == c]["feature"], rat[rat["class"] == c]["shap_value"]))
              for c in ("pre-diabetic", "healthy")}

    with plt.rc_context({"font.family": FONT, "font.sans-serif": [FONT], "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig = plt.figure(figsize=(15.0, 5.6), facecolor="white")
        draw_panel(fig.add_axes([0.24, 0.31, 0.24, 0.45]), values["pre-diabetic"], TRUE_ORDER,
                   "True class: OZ", "#B08955")
        draw_panel(fig.add_axes([0.735, 0.31, 0.24, 0.45]), values["healthy"], PREDICTED_ORDER,
                   "Predicted class: LZ", "#607D8B")
        fig.suptitle("Rat 4: true OZ, predicted LZ", x=0.52, y=0.955, fontsize=LABEL_SIZE, fontname=FONT,
                     color="black")
        fig.legend(handles=[Patch(facecolor=POSITIVE, edgecolor="white", label="Positive contribution"),
                            Patch(facecolor=NEGATIVE, edgecolor="white", label="Negative contribution")],
                   loc="lower center", bbox_to_anchor=(0.52, 0.055), ncol=2, frameon=False,
                   prop={"family": FONT, "size": LEGEND_SIZE}, handlelength=1.8, handletextpad=0.5,
                   columnspacing=1.4)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_DIR / "si_rat4_shap.pdf", format="pdf", bbox_inches="tight")
        fig.savefig(OUT_DIR / "si_rat4_shap.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(OUT_DIR / "si_rat4_shap.pdf")


def main():
    for draw in (normalized_profiles, individual_profiles, macro_f1_heatmap, permutation_test, rat4_shap):
        with plt.rc_context():
            draw()


if __name__ == "__main__":
    main()
