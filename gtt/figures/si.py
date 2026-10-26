#!/usr/bin/env python3
"""SI figures of the GTT analysis.

individual_profiles  glucose and calibrated profiles of each rat (outputs/figures/si_individual_profiles)
fasting_levels       fasting insulin, C-peptide and glucagon (si_fasting_levels)
associations         C-peptide–glucagon and insulin–C-peptide trajectories and correlations
                     (si_cpeptide_glucagon_trajectories, si_cpeptide_glucagon_r, si_insulin_cpeptide_trajectories,
                     si_insulin_cpeptide_r)
classification       macro-F1 of the 81 combinations and the exact label test (si_macro_f1_heatmap,
                     si_label_test_ecdf, si_label_test_null)
feature_recurrence   features selected by the perfect combinations, by held-out rat (si_feature_recurrence_by_rat)
dynamics             glucagon relative to the pre-injection level and the cumulative C-peptide/insulin exposure ratio
                     (si_glucagon_percent_of_baseline, si_cpeptide_insulin_exposure_ratio)
sampling             peak responses of the continuous and the sparse sampling (si_sampling_lz_r3, si_sampling_maxima)

Every figure is written as .pdf and .png.

Input   data/concentrations/concentration_timeseries.csv, data/raw/gtt_post_injection.csv,
        data/classification/scores_by_assignment.csv, selected_features_true_labels.csv

    python figures/si.py
"""
import sys

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb, to_rgba
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, FixedLocator, MultipleLocator
from scipy.integrate import cumulative_trapezoid

from figure_style import (BORDER, FONT_SIZE, LZ, NOMINAL_ANCHORS_MIN, RECORDED_ANCHORS_5DP, ROOT, ROW_HEIGHT, ZDF,
                          label_margin, nominal_time, save, use_arial)
from fig6 import (COLORBAR_GAP, COLORBAR_WIDTH, FEATURE_LABELS, INK, MARKERS, N_FITS, PROFILES, RATS, RAW, REFERENCE,
                  RIGHT_MARGIN, TICK_SIZE, colourbar, correlation_figure, correlations, count_cells, group_header,
                  index_figure, merged_recurrence, style_axis, trajectory_grid)

sys.path.insert(0, str(ROOT / "src"))
from summarize import assignment_maxima  # noqa: E402
from nested_cv import TRUE_ASSIGNMENT  # noqa: E402


# ---- glucose and calibrated profiles of each rat ----

HORMONES = ("insulin", "c_peptide", "glucagon")
ANALYTES = ("glucose", *HORMONES)
GROUPS = (("LZ", LZ), ("ZDF", ZDF))
Y_LABELS = {"insulin": "[Insulin]\n(pM)", "c_peptide": "[C-peptide]\n(pM)", "glucagon": "[Glucagon]\n(pM)",
            "glucose": "[Glucose]\n(mM)"}
COLORS = {"insulin": "#800002", "c_peptide": "#408002", "glucagon": "#FD8008", "glucose": "#2D5F8A"}
Y_SCALES = {   # (lower, upper, major step), shared within each group
    "LZ": {"insulin": (0.0, 1000.0, 200.0), "c_peptide": (0.0, 3000.0, 1000.0), "glucagon": (0.0, 16.0, 4.0),
           "glucose": (0.0, 40.0, 10.0)},
    "ZDF": {"insulin": (0.0, 1600.0, 400.0), "c_peptide": (0.0, 8000.0, 2000.0), "glucagon": (0.0, 8.0, 2.0),
            "glucose": (0.0, 40.0, 10.0)},
}
INSETS = {("Rat7", "insulin"), ("Rat7", "c_peptide"), ("Rat8", "insulin"), ("Rat8", "c_peptide")}
# panels whose label sits top-left on a white background (the others: top-right), clear of the data
LEFT_LABELS = {("Rat7", "glucose"), ("Rat8", "glucose"), ("Rat9", "insulin"), ("Rat9", "c_peptide")}
WIDTH, HEIGHT = 8.27, 15.25 - 0.17     # inches
AXIS_LABEL_SIZE, TICK_LABEL_SIZE, PANEL_LABEL_SIZE, LEGEND_SIZE = 10.5, 8.5, 10.5, 8.0
DEVICE_LINE_WIDTH, ELISA_LINE_WIDTH = 1.45, 1.15
LEGEND_GAP = 0.06                      # inches between the last row's x-axis labels and the legend


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


def device_and_elisa(ax, profile, hormone, small=False):
    """Device profile (line with open circles) and ELISA values (open diamonds, dashed line)."""
    color = COLORS[hormone]
    elisa = profile.dropna(subset=[f"{hormone}_ELISA_pM"])
    ax.plot(nominal_time(profile.timepoint, RECORDED_ANCHORS_5DP), profile[f"{hormone}_predicted_pM"],
            color=color, linewidth=1.0 if small else DEVICE_LINE_WIDTH, marker="o", markersize=1.9 if small else 2.9,
            markerfacecolor="white", markeredgecolor=color, markeredgewidth=0.55 if small else 0.75,
            zorder=3 if small else 4)
    ax.plot(nominal_time(elisa.timepoint, RECORDED_ANCHORS_5DP), elisa[f"{hormone}_ELISA_pM"], color="#777777",
            linestyle="--", linewidth=0.8 if small else ELISA_LINE_WIDTH, marker="D", markersize=2.5 if small else 4.2,
            markerfacecolor="white", markeredgecolor="#111111", markeredgewidth=0.55 if small else 0.9,
            zorder=4 if small else 6)
    ax.axvline(0.0, color="#888888", linestyle="--", linewidth=0.65 if small else 0.95, zorder=1)


def inset(ax, profile, hormone):
    """Close-range view of a low-amplitude profile, its y axis from one round tick to another."""
    values = np.concatenate([profile[f"{hormone}_predicted_pM"], profile[f"{hormone}_ELISA_pM"].dropna()])
    box = ax.inset_axes([0.41, 0.31, 0.56, 0.51])
    box.set_facecolor((1.0, 1.0, 1.0, 0.95))
    device_and_elisa(box, profile, hormone, small=True)
    box.set_xlim(-12.5, 61.5)
    lower, upper, step = bracketing_ticks(float(values.min()), float(values.max()))
    margin = 0.03 * (upper - lower)
    box.set_ylim(lower - margin, upper + margin)
    box.set_xticks([0.0, 30.0, 60.0])
    box.set_yticks(np.arange(lower, upper + 0.5 * step, step))
    box.tick_params(axis="both", labelsize=5.3, width=0.65, length=2.0, pad=1.0)
    box.grid(False)
    for spine in box.spines.values():
        spine.set_linewidth(0.75)
        spine.set_color("#777777")


def glucose_panel(ax, raw):
    """Glucometer glucose at the five blood-sampling times (gap where a sample is missing)."""
    anchors = raw.dropna(subset=["glucose_mM"])
    values = (pd.Series(anchors.glucose_mM.to_numpy(), index=nominal_time(anchors.timepoint, RECORDED_ANCHORS_5DP))
              .reindex(NOMINAL_ANCHORS_MIN))
    ax.plot(NOMINAL_ANCHORS_MIN, values, color=COLORS["glucose"], linewidth=DEVICE_LINE_WIDTH, marker="o",
            markersize=3.6, markerfacecolor="white", markeredgecolor=COLORS["glucose"], markeredgewidth=0.9, zorder=4)
    ax.axvline(0.0, color="#888888", linestyle="--", linewidth=0.95, zorder=1)


def profile_axis(ax, group, analyte, column):
    lower, upper, step = Y_SCALES[group][analyte]
    ax.set_xlim(-12.5, 61.5)
    ax.set_ylim(lower, upper)
    ax.xaxis.set_major_locator(FixedLocator([-10.0, 0.0, 15.0, 30.0, 45.0, 60.0]))
    ax.xaxis.set_minor_locator(MultipleLocator(5.0))
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_minor_locator(MultipleLocator(step / 2.0))
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE, width=1.0, length=4.0, pad=2.0)
    ax.tick_params(axis="both", which="minor", width=0.8, length=2.4)
    if column != 0:
        ax.tick_params(axis="y", labelleft=False)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.15)


def legend_handles():
    line = lambda color, label, size, edge, width, **kw: Line2D(   # noqa: E731
        [], [], color=color, marker=kw.pop("marker", "o"), markersize=size, markerfacecolor="white",
        markeredgecolor=edge, markeredgewidth=width, label=label, **kw)
    handles = [line(COLORS[h], f"{Y_LABELS[h].splitlines()[0][1:-1]}-QIRT-ELISA", 3.3, COLORS[h], 0.8,
                    linewidth=DEVICE_LINE_WIDTH) for h in HORMONES]
    handles.append(line(COLORS["glucose"], "Glucose glucometer", 3.8, COLORS["glucose"], 0.9,
                        linewidth=DEVICE_LINE_WIDTH))
    handles.append(line("#777777", "ELISA calibration point", 4.2, "#111111", 0.9, marker="D", linestyle="--",
                        linewidth=ELISA_LINE_WIDTH))
    return handles


def individual_profiles():
    """Glucose and the calibrated insulin, C-peptide and glucagon profiles of each LZ and ZDF rat.

    Columns: R1–R3 of each group; rows: glucose, insulin, C-peptide and glucagon of the LZ rats, then of the ZDF
    rats. Device profiles in the hormone colours, ELISA values as open diamonds joined by a dashed line, glucometer
    glucose at the blood-sampling times; the dashed vertical line marks the glucose injection. Axis ranges are
    shared within each group; the low insulin and C-peptide profiles of ZDF R1 and R2 have an enlarged inset.
    Times are on the nominal clock (figure_style.nominal_time). LZ R1 has no 45-min blood sample."""
    use_arial()
    profiles = pd.read_csv(PROFILES).sort_values(["animal_id", "timepoint"])
    raw = pd.read_csv(RAW)
    fig, axes = plt.subplots(8, 3, figsize=(WIDTH, HEIGHT), squeeze=False)
    for block, (group, rats) in enumerate(GROUPS):
        for a, analyte in enumerate(ANALYTES):
            for column, rat in enumerate(rats):
                ax = axes[4 * block + a, column]
                if analyte == "glucose":
                    glucose_panel(ax, raw[raw.animal_id == rat])
                else:
                    profile = profiles[profiles.animal_id == rat]
                    device_and_elisa(ax, profile, analyte)
                    if (rat, analyte) in INSETS:
                        inset(ax, profile, analyte)
                profile_axis(ax, group, analyte, column)
                ax.set_xlabel("Time (min)", fontsize=AXIS_LABEL_SIZE, labelpad=1.5)
                if column == 0:
                    ax.set_ylabel(Y_LABELS[analyte], fontsize=AXIS_LABEL_SIZE, labelpad=7.0)
    fig.subplots_adjust(left=0.145, right=0.985, bottom=(0.052 * 15.25 - 0.17) / HEIGHT,
                        top=1.0 - (1.0 - 0.988) * 15.25 / HEIGHT, wspace=0.13, hspace=0.62)
    fig.canvas.draw()
    for block, (group, rats) in enumerate(GROUPS):
        for a, analyte in enumerate(ANALYTES):
            for column, rat in enumerate(rats):
                style = dict(transform=axes[4 * block + a, column].transAxes, va="top", fontsize=PANEL_LABEL_SIZE,
                             color="#111111", zorder=8)
                label = f"{group} R{column + 1}"
                if (rat, analyte) in LEFT_LABELS:
                    axes[4 * block + a, column].text(0.025, 0.975, label, ha="left", **style, bbox=dict(
                        boxstyle="square,pad=0.12", facecolor="white", edgecolor="none"))
                else:
                    axes[4 * block + a, column].text(0.975, 0.975, label, ha="right", **style)

    # one-row legend as wide as the three columns, just below the last row's x-axis labels
    renderer = fig.canvas.get_renderer()
    labels_bottom = min(ax.get_tightbbox(renderer).y0 for ax in axes[-1]) / fig.dpi / HEIGHT
    left, right = axes[-1, 0].get_position().x0, axes[-1, -1].get_position().x1
    fig.legend(handles=legend_handles(), loc="upper left", mode="expand", borderaxespad=0.0, ncol=5,
               bbox_to_anchor=(left, labels_bottom - LEGEND_GAP / HEIGHT, right - left, 0.0), fontsize=LEGEND_SIZE,
               frameon=True, framealpha=0.95, edgecolor="#b0b0b0", fancybox=False, handlelength=1.5,
               handletextpad=0.4, columnspacing=0.6, borderpad=0.35)
    save(fig, "si_individual_profiles", facecolor="white")


# ---- fasting levels ----

PANELS = [("insulin", "[Insulin] (pM)", "Fasting insulin", True),
          ("c_peptide", "[C-peptide] (pM)", "Fasting C-peptide", True),
          ("glucagon", "[Glucagon] (pM)", "Fasting glucagon", True)]
# labels placed by hand: LZ insulin R2 left of its symbol (clear of the error bar), ZDF glucagon R2 beside it
LABEL_OVERRIDE = {("insulin", "Rat2"): ((-24, 0), "right", "center"),
                  ("glucagon", "Rat8"): ((24, 2), "left", "center")}


def fasting_levels():
    """Fasting insulin, C-peptide and glucagon of the LZ and ZDF rats.

    Fasting value: the mean of the four pre-injection samples (t = -10 to -2.5 min) of the calibrated profile. Bars:
    group mean +/- SD, with each rat (R1–R3 within each group). p: two-sided Student's t-test on log values."""
    use_arial()
    profiles = pd.read_csv(PROFILES)
    pre = profiles[profiles.timepoint < 0].groupby("animal_id")
    fasting = pd.DataFrame({h: pre[f"{h}_predicted_pM"].mean() for h, *_ in PANELS}).loc[RATS]
    index_figure(fasting, PANELS, "si_fasting_levels", label_override=LABEL_OVERRIDE)


# ---- C-peptide–glucagon and insulin–C-peptide associations ----

RIGHT_OF_SYMBOL = (0.075, 0.0, "left", "center")


def associations():
    """C-peptide–glucagon and insulin–C-peptide associations after the glucose injection.

    a, c  trajectory of each rat from t = 0 (open circle) to the last sample (black square), coloured by the time
          after the injection: C-peptide (x) against glucagon (y), and insulin (x) against C-peptide (y)
    b, d  Pearson r between the two profiles of each rat over the 24 samples from t = 0, with the group mean +/- SD;
          p: Student's t-test on Fisher z-transformed coefficients"""
    use_arial()
    profiles = pd.read_csv(PROFILES)
    trajectory_grid(profiles, "c_peptide", "glucagon", "[C-peptide] (pM)", "[Glucagon] (pM)",
                    "si_cpeptide_glucagon_trajectories", x_nbins=2)          # four-digit x ticks need room
    correlation_figure(correlations(profiles, "c_peptide", "glucagon"), "C-peptide–glucagon correlation",
                       "si_cpeptide_glucagon_r", label_at={"Rat8": (-0.10, -0.018, "right", "top")})
    trajectory_grid(profiles, "insulin", "c_peptide", "[Insulin] (pM)", "[C-peptide] (pM)",
                    "si_insulin_cpeptide_trajectories", ytick_space=1.16)    # four-digit y ticks
    # the two middle rats sit on their group's error bar: labels to the right of the symbol
    correlation_figure(correlations(profiles, "insulin", "c_peptide"), "Insulin–C-peptide correlation",
                       "si_insulin_cpeptide_r", label_at={"Rat2": RIGHT_OF_SYMBOL, "Rat8": RIGHT_OF_SYMBOL})


# ---- macro-F1 of the 81 combinations and the exact label test ----

SCORES = ROOT / "data" / "classification" / "scores_by_assignment.csv"
SELECTORS = ["anova", "elastic_net", "kruskal", "lasso", "mrmr", "mutual_info", "relieff", "rf_importance", "rfe"]
CLASSIFIERS = ["DecisionTree", "KNN", "LDA", "LogReg_L1", "LogReg_L2", "NaiveBayes", "RandomForest", "SVM_linear",
               "SVM_rbf"]
CMAP = LinearSegmentedColormap.from_list("blue", ["#f7fbff", "#e5f0fa", "#c9dff2", "#94bfdc", "#5f98c3"])
TEXT, EDGE, LINE_WIDTH = "#111111", "black", 1.8
SET_LABEL, SET_COLOR = "Expanded (181 features)", "#3B6FB6"


def heatmap(scores):
    true = scores[scores.true_labels]
    matrix = true.pivot(index="selector", columns="classifier", values="macro_f1").reindex(index=SELECTORS,
                                                                                         columns=CLASSIFIERS)
    perfect = [(s, c) for s in SELECTORS for c in CLASSIFIERS if matrix.loc[s, c] == 1.0]
    plt.rcParams.update({"font.family": "Arial", "font.size": 12, "axes.labelsize": 13.5, "axes.titlesize": 15,
                         "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "figure.dpi": 150,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.2, 5.8), facecolor="white")
    edges = np.arange(10) - 0.5
    image = ax.pcolormesh(edges, edges, matrix.to_numpy(), cmap=CMAP, vmin=0, vmax=1, shading="flat",
                          edgecolors="face", linewidth=0.1, antialiased=False, rasterized=False)
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(8.5, -0.5)
    ax.set_aspect("auto")
    for i, selector in enumerate(SELECTORS):
        for j, classifier in enumerate(CLASSIFIERS):
            value = float(matrix.loc[selector, classifier])
            bold = (selector, classifier) in perfect
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=10.5 if bold else 9.3,
                    fontweight="bold" if bold else "normal", color="white" if value > 0.72 else "#1f2933")
    for selector, classifier in perfect:
        ax.add_patch(patches.Rectangle((CLASSIFIERS.index(classifier) - 0.5, SELECTORS.index(selector) - 0.5), 1, 1,
                                       fill=False, edgecolor="black", linewidth=LINE_WIDTH, clip_on=False))
    ax.add_patch(patches.Rectangle((-0.5, -0.5), 9, 9, fill=False, edgecolor="black", linewidth=LINE_WIDTH,
                                   clip_on=False))
    ax.set_xticks(range(9), CLASSIFIERS, rotation=45, ha="right")
    ax.set_yticks(range(9), SELECTORS)
    ax.set_xlabel("Classifier")
    ax.set_ylabel("Feature selector")
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
    cbar.solids.set_rasterized(False)
    cbar.solids.set_edgecolor("face")
    cbar.set_label("Macro-F1", fontsize=10.125)
    cbar.set_ticks(np.linspace(0, 1, 6))
    cbar.ax.tick_params(labelsize=10.5)
    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(LINE_WIDTH)
    fig.tight_layout()
    print(f"  {len(perfect)} perfect combinations, mean macro-F1 {true.macro_f1.mean():.3f}")
    save(fig, "si_macro_f1_heatmap", pdf_dpi=600, bbox_inches="tight", facecolor="white")


def label_test_axis(ax):
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(LINE_WIDTH)
        spine.set_color(EDGE)
    ax.tick_params(color=EDGE, labelcolor=TEXT, width=1.0)


def label_test(scores):
    maxima = assignment_maxima(scores)
    observed = float(maxima.loc[maxima.assignment == TRUE_ASSIGNMENT, "max_macro_f1"].iloc[0])
    null = maxima.loc[maxima.assignment != TRUE_ASSIGNMENT, "max_macro_f1"].to_numpy(float)
    reached = int(np.sum(null >= observed - 1e-12))
    p = (reached + 1) / len(maxima)
    print(f"  {reached} of {len(null)} other assignments reach the observed {observed:.3f}; p = {p:.2f}; "
          f"null mean {null.mean():.3f}")
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 600, "font.family": "Arial", "font.sans-serif": ["Arial"],
                         "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 19, "axes.labelsize": 21,
                         "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 14, "axes.grid": False,
                         "font.weight": "normal", "axes.spines.top": True, "axes.spines.right": True})

    # b: empirical cumulative distribution
    fig, ax = plt.subplots(figsize=(8.6, 6.8))
    values = np.sort(null)
    ax.step(np.concatenate([[0.0], values]), np.concatenate([[0.0], np.arange(1, len(values) + 1) / len(values)]),
            where="post", color=SET_COLOR, linewidth=2.3,
            label=f"{SET_LABEL} (p = {p:.2f}, {reached}/{len(null)} reached)")
    ax.scatter([observed], [np.mean(null <= observed)], color=SET_COLOR, edgecolor=EDGE, linewidth=LINE_WIDTH, s=78,
               zorder=3)
    ax.axvline(observed, color=TEXT, linewidth=2.0, label=f"Observed F1 = {observed:.3f}")
    ax.set_xlim(0, 1.04)
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("Best macro-F1 of the 81 recipes")
    ax.set_ylabel("Empirical cumulative probability")
    label_test_axis(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.96, bottom=0.34)
    save(fig, "si_label_test_ecdf", png_dpi=None, pdf_dpi=None, bbox_inches="tight")

    # c: histogram
    fig, ax = plt.subplots(figsize=(8.2, 7.6))
    bins = np.linspace(0, 1.0, 26)
    counts, _ = np.histogram(null, bins=bins)
    keep = counts > 0
    ax.bar(bins[:-1][keep], (counts / len(null))[keep], width=bins[1] - bins[0], align="edge",
           color=to_rgba(SET_COLOR, 0.42), edgecolor=EDGE, linewidth=1.3, label=SET_LABEL)
    ax.axvline(observed, color=TEXT, linewidth=2.2, label=f"Observed F1 = {observed:.3f}")
    ax.axvline(null.mean(), color=SET_COLOR, linestyle="--", linewidth=2.0,
               label=f"Expanded null mean = {null.mean():.3f}")
    ax.set_xlabel("Best macro-F1 of the 81 recipes\nunder relabeled rats")
    ax.set_ylabel("Fraction of relabelings")
    ax.set_xlim(0, 1.04)
    ax.set_ylim(0, 0.55)
    label_test_axis(ax)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(SET_LABEL)] + [i for i, label in enumerate(labels) if label != SET_LABEL]   # bars first
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center",
              bbox_to_anchor=(0.5, -0.27), ncol=1, frameon=False)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.96, bottom=0.34)
    save(fig, "si_label_test_null", png_dpi=None, pdf_dpi=None, bbox_inches="tight")


def classification():
    """Macro-F1 of the 81 combinations and the exact label test of the GTT classification.

    a  macro-F1 of each selector (rows) and classifier (columns) with the true labels; the 28 combinations that
       classified all six rats correctly are outlined and printed in bold
    b  empirical cumulative distribution of the best macro-F1 of the 81 combinations under the 19 label assignments
       other than the true one; the circle marks the observed value
    c  histogram of the same 19 values, with their mean (dashed) and the observed value (solid)
    p counts the true assignment too: 4 of the 20 assignments reach the observed value (p = 0.20)."""
    scores = pd.read_csv(SCORES, float_precision="round_trip")
    heatmap(scores)
    label_test(scores)


# ---- features selected by held-out rat ----

N_COMBINATIONS, TOTAL_WIDTH = 28, 1.75     # inches: the total column


def feature_recurrence():
    """The features selected by the 28 perfect combinations, by held-out rat.

    Each cell: the number of the 28 perfect combinations that selected the feature in the fold where that rat was
    held out (blank when none); right, the total over the 168 held-out fits. Rows: the 12 features of main-text
    Fig. 6e, equivalent features merged (fig6.py)."""
    plt.rcParams.update({"font.family": "Arial", "font.sans-serif": ["Arial"], "pdf.fonttype": 42, "ps.fonttype": 42})
    top = merged_recurrence()
    labels = [FEATURE_LABELS[f] for f in top.feature]
    n = len(labels)

    # frame, inches: labels | 6 held-out rats | total | colour bar
    left = label_margin(labels)
    plot_w, plot_h, bottom = 1.2 * 6, ROW_HEIGHT * n, 1.0
    x_total = left + plot_w + 0.12
    cbar_left = x_total + TOTAL_WIDTH + COLORBAR_GAP
    width = cbar_left + COLORBAR_WIDTH + RIGHT_MARGIN + 0.35
    height = bottom + plot_h + 1.0
    fig = plt.figure(figsize=(width, height))
    axis = lambda x, w: fig.add_axes([x / width, bottom / height, w / width, plot_h / height])   # noqa: E731
    ax, tax, cax = axis(left, plot_w), axis(x_total, TOTAL_WIDTH), axis(cbar_left, COLORBAR_WIDTH)

    image = count_cells(ax, top[LZ + ZDF].to_numpy(int), N_COMBINATIONS, int(np.ceil(N_COMBINATIONS * 4 / 6)),
                        ["R1", "R2", "R3"] * 2)
    ax.set_yticks(range(n), labels=labels)
    ax.axvline(2.5, color="black", lw=BORDER, zorder=3)
    texts = []
    group_header(ax, texts)
    texts.append(ax.text(2.5, n - 0.5 + 1.15, "Held-out rat (LOO fold)", ha="center", va="top", fontsize=FONT_SIZE,
                         fontfamily="Arial", clip_on=False))
    ax.set_ylim(n - 0.5, -0.5)
    tax.set_xlim(0, 1)
    tax.set_ylim(n - 0.5, -0.5)
    tax.axis("off")
    for y, total in enumerate(top.fits):
        tax.text(0.08, y, f"{total}/{N_FITS}", ha="left", va="center", fontsize=FONT_SIZE, fontfamily="Arial")
    texts.append(tax.text(0.08, -0.95, "Total", ha="left", va="bottom", fontsize=FONT_SIZE, fontfamily="Arial",
                          clip_on=False))
    colourbar(fig, image, cax, [0, 7, 14, 21, 28], "Recipes selecting (of 28)")
    save(fig, "si_feature_recurrence_by_rat", png_dpi=300, pdf_dpi=600, bbox_inches="tight",
         bbox_extra_artists=fig.get_default_bbox_extra_artists() + texts)


# ---- glucagon and exposure-ratio dynamics ----

DYNAMICS_GROUPS = (("LZ", LZ, "#607D8B"), ("ZDF", ZDF, "#8A4F4A"))
NOMINAL_TICKS, NOMINAL_MINOR = [-10, 0, 15, 30, 45, 60], [-5, 7.5, 22.5, 37.5, 52.5]
END_LABEL_GAP = 0.95        # minimum vertical spacing of the end-of-trace labels (ratio units)


def glucagon_percent(profiles):
    fig, ax = plt.subplots(figsize=(8.93, 7.72))
    for group, rats, color in DYNAMICS_GROUPS:
        curves = []
        for rat in rats:
            rows = profiles[profiles.animal_id == rat]
            baseline = rows.loc[rows.timepoint < 0, "glucagon_predicted_pM"].mean()
            curves.append(100.0 * rows.glucagon_predicted_pM.to_numpy(float) / baseline)
        time = np.round(nominal_time(rows.timepoint, RECORDED_ANCHORS_5DP), 3)
        mean, sd = np.mean(curves, axis=0), np.std(curves, axis=0, ddof=1)
        ax.fill_between(time, mean - sd, mean + sd, color=color, alpha=0.18, linewidth=0, zorder=1)
        ax.plot(time, mean, color=color, lw=3.6, solid_capstyle="round", zorder=3)
    ax.axvline(0, color=REFERENCE, ls="--", lw=2.2, zorder=2)
    ax.axhline(100, color="#AAAAAA", ls=":", lw=2.0, zorder=2)
    ax.text(61.5, 103.0, "pre-injection", ha="right", va="bottom", fontsize=23.25, color=REFERENCE)
    ax.set_xlim(-12, 64)
    ax.set_ylim(-20, 150)
    ax.set_xticks(NOMINAL_TICKS)
    ax.xaxis.set_minor_locator(FixedLocator(NOMINAL_MINOR))
    ax.set_yticks([0, 50, 100, 150])
    ax.yaxis.set_minor_locator(MultipleLocator(25))
    ax.set_xlabel("Time (min)", labelpad=12)
    ax.set_ylabel("[Glucagon] (% of pre-injection)", labelpad=14)
    style_axis(ax)
    ax.yaxis.label.set_size(30.0)
    legend = ax.legend(handles=[Line2D([], [], color=color, lw=3.6, label=f"{group} (n = 3)")
                                for group, _, color in DYNAMICS_GROUPS],
                       loc="lower right", frameon=True, framealpha=0.50, edgecolor="#CCCCCC", fontsize=23.25,
                       handlelength=1.8, borderpad=0.55, labelspacing=0.35)
    legend.get_frame().set_linewidth(1.0)
    fig.subplots_adjust(left=0.235, right=0.97, bottom=0.205, top=0.97)
    save(fig, "si_glucagon_percent_of_baseline", bbox_inches="tight", facecolor="white")


def spread(values):
    """End-label positions at least END_LABEL_GAP apart, keeping their order and centre."""
    order = sorted(values, key=values.get)
    placed = [values[a] for a in order]
    for _ in range(50):
        moved = False
        for k in range(1, len(placed)):
            gap = placed[k] - placed[k - 1]
            if gap < END_LABEL_GAP:
                placed[k - 1] -= (END_LABEL_GAP - gap) / 2
                placed[k] += (END_LABEL_GAP - gap) / 2
                moved = True
        if not moved:
            break
    return dict(zip(order, placed))


def exposure_ratio(profiles):
    ratios = []
    for rat in LZ + ZDF:
        rows = profiles[profiles.animal_id == rat].sort_values("timepoint")
        t = rows.timepoint.to_numpy(float)
        auc_insulin = cumulative_trapezoid(rows.insulin_predicted_pM.to_numpy(float), t, initial=0.0)
        auc_cpeptide = cumulative_trapezoid(rows.c_peptide_predicted_pM.to_numpy(float), t, initial=0.0)
        ratio = np.divide(auc_cpeptide, auc_insulin, out=np.full_like(auc_cpeptide, np.nan), where=auc_insulin > 0)
        ratios.append(pd.DataFrame({"animal_id": rat, "time": nominal_time(t, RECORDED_ANCHORS_5DP), "ratio": ratio}))
    ratios = pd.concat(ratios).dropna()

    fig, ax = plt.subplots(figsize=(8.25, 7.72))
    ends = {}
    for group, rats, color in DYNAMICS_GROUPS:
        light = tuple(np.asarray(to_rgb(color)) * (1.0 - 0.48) + 0.48)        # 48% towards white
        for rat in rats:
            rows = ratios[ratios.animal_id == rat]
            ax.plot(rows.time, rows.ratio, color=light, linewidth=1.8, alpha=1.0, zorder=2)
            ax.scatter([rows.time.iloc[-1]], [rows.ratio.iloc[-1]], marker=MARKERS[rat], s=54, facecolor=light,
                       edgecolor="white", linewidth=0.75, zorder=5)
            ends[rat] = (float(rows.time.iloc[-1]), float(rows.ratio.iloc[-1]), color)
    for _, rats, _ in DYNAMICS_GROUPS:
        placed = spread({rat: ends[rat][1] for rat in rats})
        for k, rat in enumerate(rats):
            x, y, color = ends[rat]
            ax.annotate(f"R{k + 1}", xy=(x, y), xytext=(x - 1.6, placed[rat]), textcoords="data", fontsize=20,
                        color=color, va="center", ha="right")
    for group, rats, color in DYNAMICS_GROUPS:
        table = ratios[ratios.animal_id.isin(rats)].pivot(index="time", columns="animal_id", values="ratio")
        mean, sd = table.mean(axis=1), table.std(axis=1, ddof=1)
        ax.fill_between(mean.index.to_numpy(float), (mean - sd).to_numpy(float), (mean + sd).to_numpy(float),
                        color=color, alpha=0.18, linewidth=0, zorder=1)
        ax.plot(mean.index, mean.values, color=color, linewidth=3.6, solid_capstyle="round", zorder=4)
        print(f"  {group} exposure ratio at the end: {', '.join(f'{ends[r][1]:.1f}' for r in rats)}")
    # each mean line named in its own colour: ZDF inside its band, LZ below its band
    ax.text(22.5, 11.7, "ZDF (n = 3)", color=DYNAMICS_GROUPS[1][2], fontsize=23.25, ha="center", va="bottom", zorder=6)
    ax.text(22.5, 1.55, "LZ (n = 3)", color=DYNAMICS_GROUPS[0][2], fontsize=23.25, ha="center", va="top", zorder=6)
    ax.axvline(0.0, color=REFERENCE, linestyle="--", linewidth=2.2, zorder=3)
    ax.set_xlim(-10.5, 64.0)
    ax.set_ylim(0.0, 20.0)
    ax.set_xticks([float(t) for t in NOMINAL_TICKS])
    ax.xaxis.set_minor_locator(FixedLocator(NOMINAL_MINOR))
    ax.set_yticks([0.0, 5.0, 10.0, 15.0, 20.0])
    ax.yaxis.set_minor_locator(MultipleLocator(2.5))
    ax.set_xlabel("Time (min)", labelpad=12)
    ax.set_ylabel("Cumulative exposure ratio\n(C-peptide/insulin)", labelpad=18)
    style_axis(ax)
    ax.yaxis.label.set_size(30.0)
    fig.subplots_adjust(left=0.18, right=0.95, bottom=0.205, top=0.97)
    save(fig, "si_cpeptide_insulin_exposure_ratio", bbox_inches="tight", facecolor="white")


def dynamics():
    """Glucagon relative to the pre-injection level, and the cumulative C-peptide/insulin exposure ratio.

    a  glucagon as a percentage of each rat's pre-injection mean (the four samples before t = 0); group mean (line)
       +/- SD (band); dotted line, 100%; dashed line, the glucose injection
    b  ratio of the cumulative C-peptide to the cumulative insulin exposure (trapezoidal areas under the profiles
       from t = -10 min to each sample) of each rat (thin lines) with the group mean +/- SD
    Times are on the nominal clock (figure_style.nominal_time)."""
    use_arial()
    profiles = pd.read_csv(PROFILES).sort_values(["animal_id", "timepoint"])
    glucagon_percent(profiles)
    exposure_ratio(profiles)


# ---- continuous versus sparse sampling ----

SAMPLING_HORMONES = {"insulin": ("Insulin", "#800002", "o"), "c_peptide": ("C-peptide", "#408002", "s")}
ELISA_GREY = "#555555"
Y_LABEL_SIZE, NOTE_SIZE = 30.0, 23.25


def nominal(t):
    return np.round(nominal_time(t, RECORDED_ANCHORS_5DP), 3)


def series(device, elisa, rat, hormone):
    """Device samples and ELISA values of one rat and hormone, after the injection."""
    d = device[device.animal_id == rat].sort_values("timepoint")
    e = elisa[(elisa.animal_id == rat) & elisa[f"{hormone}_ELISA_pM"].notna()].sort_values("timepoint")
    return (d.timepoint.to_numpy(float), d[f"{hormone}_predicted_pM"].to_numpy(float),
            e.timepoint.to_numpy(float), e[f"{hormone}_ELISA_pM"].to_numpy(float))


def maxima(device, elisa):
    """Device and ELISA maxima of each rat and hormone (the earliest if tied)."""
    rows = []
    for rat in RATS:
        for hormone in SAMPLING_HORMONES:
            t, y, te, ye = series(device, elisa, rat, hormone)
            im, ie = int(np.argmax(y)), int(np.argmax(ye))
            rows.append({"animal_id": rat, "hormone": hormone,
                         "elisa_below_device_max_percent": 100 * (1 - ye[ie] / y[im]),
                         "device_max_time": float(nominal([t[im]])[0]),
                         "elisa_minus_device_time": float(nominal([te[ie]])[0] - nominal([t[im]])[0]),
                         "device_max_at_first_or_last_sample": im in (0, len(t) - 1)})
    return pd.DataFrame(rows)


def group_axis(ax):
    """Rats R1–R3 of each group on x, with LZ and ZDF below them."""
    ax.set_xticks(range(6))
    ax.set_xticklabels(["R1", "R2", "R3"] * 2)
    ax.set_xlim(-0.6, 5.6)
    ax.axvline(2.5, color="#BBBBBB", lw=2.0, ls=":", zorder=1)
    for x, name in ((1, "LZ"), (4, "ZDF")):
        ax.text(x, -0.155, name, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=TICK_SIZE,
                color=INK)


def lz_r3(device, elisa, table):
    """Panels a and b."""
    fig, axes = plt.subplots(1, 2, figsize=(16.8, 8.1))
    for ax, (hormone, (name, color, marker)) in zip(axes, SAMPLING_HORMONES.items()):
        t, y, te, ye = series(device, elisa, "Rat3", hormone)
        ax.plot(nominal(t), y, color=color, lw=3.6, marker=marker, ms=9, zorder=3, label="Device (24 samples)")
        ax.plot(nominal(te), ye, color=ELISA_GREY, ls="--", lw=2.4, marker="o", ms=16, mfc="white", mec=ELISA_GREY,
                mew=2.4, zorder=4, label="ELISA (5 draws)")
        m = table[(table.animal_id == "Rat3") & (table.hormone == hormone)].iloc[0]
        ax.axvline(m.device_max_time, color=color, lw=2.2, ls=":", zorder=2)
        ax.set_ylim(0, max(y.max(), ye.max()) * 1.32)
        ax.set_xlim(-2, 62)
        ax.set_xticks([0, 15, 30, 45, 60])
        ax.xaxis.set_minor_locator(FixedLocator([7.5, 22.5, 37.5, 52.5]))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.set_xlabel("Time (min)", labelpad=12)
        ax.set_ylabel(f"[{name}] (pM)", labelpad=14)
        style_axis(ax)
        ax.yaxis.label.set_size(Y_LABEL_SIZE)
        ax.set_title("LZ R3", loc="left", fontsize=Y_LABEL_SIZE, pad=10, color=INK)
        ax.text(0.97, 0.955, f"ELISA maximum\n{m.elisa_below_device_max_percent:.0f}% lower", transform=ax.transAxes,
                ha="right", va="top", fontsize=NOTE_SIZE, color=INK, linespacing=1.1)
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=NOTE_SIZE,
                        handlelength=2.2, columnspacing=2.0, bbox_to_anchor=(0.54, -0.035))
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.25, top=0.92, wspace=0.34)
    save(fig, "si_sampling_lz_r3", bbox_inches="tight", facecolor="white",
         bbox_extra_artists=fig.get_default_bbox_extra_artists() + [legend])


def maxima_panels(table):
    """Panels c and d."""
    fig, axes = plt.subplots(1, 2, figsize=(16.8, 8.1))
    specs = (("elisa_below_device_max_percent", "ELISA maximum below\ndevice maximum (%)", (-15, 55)),
             ("elisa_minus_device_time", "Time of ELISA maximum\nminus device (min)", (-45, 45)))
    for ax, (column, ylabel, ylim) in zip(axes, specs):
        for j, (hormone, (_, color, marker)) in enumerate(SAMPLING_HORMONES.items()):
            rows = table[table.hormone == hormone].set_index("animal_id").loc[RATS]
            xs = np.arange(6) + (j - 0.5) * 0.30
            ax.vlines(xs, 0, rows[column], colors=color, lw=3.0, zorder=2)
            edge = rows.device_max_at_first_or_last_sample.to_numpy()
            ax.scatter(xs[~edge], rows[column].to_numpy()[~edge], s=260, marker=marker, color=color, edgecolor="white",
                       linewidth=1.5, zorder=4)
            ax.scatter(xs[edge], rows[column].to_numpy()[edge], s=260, marker=marker, facecolor="white",
                       edgecolor=color, linewidth=3.0, zorder=4)
        ax.axhline(0, color="#888888", lw=2.0, zorder=1)
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_locator(MultipleLocator(15))
        ax.yaxis.set_minor_locator(AutoMinorLocator(3))
        ax.set_ylabel(ylabel, labelpad=14)
        style_axis(ax)
        ax.yaxis.label.set_size(Y_LABEL_SIZE)
        ax.tick_params(axis="x", which="minor", bottom=False)
        group_axis(ax)
    handles = [Line2D([], [], ls="none", marker="o", ms=15, color=SAMPLING_HORMONES["insulin"][1], label="Insulin"),
               Line2D([], [], ls="none", marker="s", ms=14, color=SAMPLING_HORMONES["c_peptide"][1], label="C-peptide"),
               Line2D([], [], ls="none", marker="o", ms=15, mfc="white", mec="#666666", mew=3.0,
                      label="Device maximum at\nfirst or last sample")]
    axes[1].legend(handles=handles, loc="upper right", frameon=False, fontsize=NOTE_SIZE, handletextpad=0.3,
                   labelspacing=0.45, borderaxespad=0.3)
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.24, top=0.97, wspace=0.40)
    save(fig, "si_sampling_maxima", bbox_inches="tight", facecolor="white",
         bbox_extra_artists=fig.get_default_bbox_extra_artists())


def sampling():
    """Peak responses captured by the continuous device record and by the sparse ELISA sampling.

    a, b  insulin and C-peptide of LZ R3: the device profile (24 samples after the injection) and the ELISA values
          (5 blood draws); the dotted line marks the device maximum
    c     how far the largest ELISA value lies below the device maximum, for each rat (insulin, circles; C-peptide,
          squares)
    d     time of the largest ELISA value minus the time of the device maximum (negative: earlier ELISA maximum);
          open symbols, device maximum at the first or last sample
    Times are on the nominal clock (figure_style.nominal_time). The device is calibrated to the same ELISA values."""
    use_arial()
    profiles = pd.read_csv(PROFILES)
    device = profiles[profiles.animal_id.isin(RATS) & (profiles.timepoint >= 0)]
    raw = pd.read_csv(RAW)
    elisa = raw[raw.animal_id.isin(RATS) & (raw.timepoint >= 0)]
    table = maxima(device, elisa)
    print(table.round(1).to_string(index=False))
    lz_r3(device, elisa, table)
    maxima_panels(table)


def main():
    for draw in (individual_profiles, fasting_levels, associations, classification, feature_recurrence, dynamics,
                 sampling):
        with plt.rc_context():          # every figure starts from the default settings
            draw()


if __name__ == "__main__":
    main()
