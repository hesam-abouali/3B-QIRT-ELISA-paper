#!/usr/bin/env python3
"""Fig. 4 d–j: classification results of the fed state.

d–g  macro-F1 of the 81 combinations, predictive confidence, permutation test and confusion matrix
     (outputs/figures/fig4_d_to_g.pdf and .png)
h–i  features of the two reported recipes: selection counts and z-scores (fig4_h_i.pdf and .png)
j    fold-specific SHAP waterfalls of the nine held-out rats (fig4_j.pdf and .png, fig4_j_legend.pdf and .png)

Input   data/classification/results_summary.csv, per_fold_details.csv, confusion_matrices.csv,
        selected_features.csv, confidence_by_rat.csv
        data/permutation/per_draw_LogReg_L2.csv, per_draw_SVM_linear.csv
        data/shap/shap_values.csv, shap_base_values.csv
        data/features/fedstate_X.csv, fedstate_meta.csv

    python figures/fig4.py
"""
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb, to_rgba
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch, Polygon, Rectangle
from matplotlib.textpath import TextPath
from matplotlib.transforms import Bbox
from scipy import stats
from sklearn.metrics import f1_score

from figure_style import BORDER_LINEWIDTH, FONT_SIZE, INTERNAL_LINEWIDTH, create_panels, darker, style_axis

ROOT = Path(__file__).resolve().parents[1]
CLASSIFICATION = ROOT / "data" / "classification"
PERMUTATION = ROOT / "data" / "permutation"
SHAP_DIR = ROOT / "data" / "shap"
FEATURES = ROOT / "data" / "features"
OUT_DIR = ROOT / "outputs" / "figures"

PHENOTYPES = ["healthy", "pre-diabetic", "diabetic"]
PHENOTYPE_LABELS = {"healthy": "LZ", "pre-diabetic": "OZ", "diabetic": "ZDF"}


# ---- d–g: classification results ----

RECIPES = ["LogReg_L2", "SVM_linear"]           # with the mRMR selector
DISPLAY = {"LogReg_L2": "LR", "SVM_linear": "SVM"}
FILL_COLORS = {"LogReg_L2": "#CDD9DE", "SVM_linear": "#DED3C5"}   # bars
POINT_COLORS = {"LogReg_L2": "#95AAB3", "SVM_linear": "#BAA58C"}  # points and diamonds
MARKERS = {"LogReg_L2": "o", "SVM_linear": "s"}
GREY = "#858585"
GREY_LIGHT = "#C6C6C6"
P_VALUE_COLOR = "#3F3F3F"
P_VALUE_SIZE = 15
CONFUSION_CMAP = LinearSegmentedColormap.from_list(
    "confusion_blue", ["#f7fbff", "#d8e9f5", "#a8cae2", "#5f98c3", "#1f4f7a"])


def observed_macro_f1(classifier):
    """Macro-F1 of mRMR + classifier from its per-fold predictions."""
    folds = pd.read_csv(CLASSIFICATION / "per_fold_details.csv")
    folds = folds[(folds.selector == "mrmr") & (folds.classifier == classifier)]
    return f1_score(folds.true_label.to_numpy(), folds.pred_label.to_numpy(), average="macro",
                    labels=PHENOTYPES, zero_division=0)


def plot_macro_f1(ax):
    """d: macro-F1 of the 81 combinations."""
    df = pd.read_csv(CLASSIFICATION / "results_summary.csv")
    values = df["macro_f1"].to_numpy()
    best_f1 = values.max()
    is_best = (df["selector"] == "mrmr") & df["classifier"].isin(RECIPES)
    best = df.loc[is_best].sort_values("classifier").reset_index(drop=True)
    assert len(best) == 2 and np.allclose(best["macro_f1"], best_f1)

    violin = ax.violinplot([values], positions=[0], widths=0.62, showmedians=False, showextrema=False)
    for body in violin["bodies"]:
        body.set_facecolor(to_rgba(GREY, 0.23))
        body.set_alpha(None)
        body.set_edgecolor("black")
        body.set_linewidth(INTERNAL_LINEWIDTH)

    others = df.loc[~is_best, "macro_f1"].to_numpy()  # mRMR + KNN (also 0.886) is drawn as an ordinary point
    rng = np.random.default_rng(42)
    ax.scatter(rng.uniform(-0.17, 0.17, size=len(others)), others, zorder=2, s=18 * 2.5,
               color=to_rgba(GREY_LIGHT, 0.62), edgecolors="black", linewidths=0.45)

    box = ax.boxplot(
        [values], positions=[0], widths=0.26, patch_artist=True, showfliers=False,
        medianprops={"color": "black", "linewidth": INTERNAL_LINEWIDTH},
        whiskerprops={"color": "black", "linewidth": INTERNAL_LINEWIDTH},
        capprops={"color": "black", "linewidth": INTERNAL_LINEWIDTH},
        boxprops={"color": "black", "linewidth": INTERNAL_LINEWIDTH},
    )
    box["boxes"][0].set_facecolor(to_rgba(GREY, 0.48))
    box["boxes"][0].set_alpha(None)

    ax.scatter(np.linspace(-0.055, 0.055, len(best)), np.repeat(best_f1, len(best)), zorder=5, s=68 * 2.5,
               color=[POINT_COLORS[c] for c in best["classifier"]],
               edgecolors=[darker(POINT_COLORS[c]) for c in best["classifier"]],
               linewidths=INTERNAL_LINEWIDTH, marker="D")

    ax.set_xticks([0])
    ax.set_xticklabels(["Calibrated profile"])
    ax.set_ylabel("Macro-F1")
    ax.set_xlim(-0.48, 0.48)
    ax.set_ylim(0, 1.0)
    ax.minorticks_off()
    ax.set_title("")
    style_axis(ax)


def plot_confidence(ax):
    """e: predictive confidence of each rat, with the Wilcoxon signed-rank test of LR vs SVM."""
    df = pd.read_csv(CLASSIFICATION / "confidence_by_rat.csv")
    df = df[(df["selector"] == "mrmr") & df["classifier"].isin(RECIPES)]
    wide = df.pivot(index="rat_id", columns="classifier", values="max_probability")[RECIPES]
    correct = df.pivot(index="rat_id", columns="classifier", values="correct")[RECIPES]
    wrong = {c: set(correct.index[~correct[c].astype(bool)]) for c in RECIPES}
    assert len(wide) == 9 and wrong["LogReg_L2"] == wrong["SVM_linear"] and len(wrong["LogReg_L2"]) == 1
    wide = wide.loc[sorted(wide.index, key=lambda rat: int(rat.replace("Rat", "")))]

    x = np.arange(len(RECIPES))
    ax.bar(x, wide.mean(axis=0).to_numpy(), color=[FILL_COLORS[c] for c in RECIPES], edgecolor="black",
           linewidth=INTERNAL_LINEWIDTH, width=0.69, zorder=1)
    for rat in wide.index:
        values = wide.loc[rat, RECIPES].to_numpy(float)
        ax.plot(x, values, color="#A0A0A0", linewidth=0.8, linestyle="-", alpha=0.85, zorder=2)
        for j, classifier in enumerate(RECIPES):
            ax.scatter(x[j], values[j], s=140 * 1.25, marker=MARKERS[classifier],
                       facecolor=POINT_COLORS[classifier], edgecolor=darker(POINT_COLORS[classifier]),
                       linewidth=0.65 * 1.25 * 2.0, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[c] for c in RECIPES])
    ax.set_ylabel("Predictive confidence")
    ax.set_xlim(-0.55, len(RECIPES) - 0.45)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.minorticks_off()
    style_axis(ax)

    lr, svm = (wide[c].to_numpy(float) for c in RECIPES)
    p = float(stats.wilcoxon(lr, svm, alternative="two-sided", method="exact").pvalue)
    ax.text(0.5, 1.01, f"p = {p:.4f}", ha="center", va="bottom", fontsize=P_VALUE_SIZE,
            fontfamily="Arial", fontweight="normal", color=P_VALUE_COLOR)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    print(f"e: exact Wilcoxon signed-rank test, two-sided p = {p:.6f}")


def plot_permutation(ax):
    """f: null distribution (mean +/- SD and every draw) and observed macro-F1 of both recipes."""
    width, spacing = 0.2274, 0.68
    x = np.arange(len(RECIPES)) * spacing
    null_x, observed_x = x - width / 1.8, x + width / 1.8
    nulls, observed, labels = [], [], []
    for classifier in RECIPES:
        draws = pd.read_csv(PERMUTATION / f"per_draw_{classifier}.csv").sort_values("draw")
        assert draws["draw"].tolist() == list(range(1000))
        null = draws["macro_f1"].to_numpy(dtype=float)
        obs = observed_macro_f1(classifier)
        reached = int(np.sum(null >= obs - 1e-12))
        nulls.append(null)
        observed.append(obs)
        labels.append(f"p = {(reached + 1) / (len(null) + 1):.4f}\n{reached}/{len(null)} reached")
    means = np.array([null.mean() for null in nulls])
    sds = np.array([null.std(ddof=0) for null in nulls])

    ax.bar(null_x, means, width=width, yerr=sds, color="#E6E6E6", edgecolor="black",
           linewidth=INTERNAL_LINEWIDTH, capsize=5,
           error_kw={"elinewidth": INTERNAL_LINEWIDTH, "capthick": INTERNAL_LINEWIDTH, "ecolor": "black"}, zorder=1)
    rng = np.random.default_rng(42)
    for i, classifier in enumerate(RECIPES):
        jitter = rng.normal(0, width * 0.16, size=len(nulls[i]))
        ax.scatter(np.full(len(nulls[i]), null_x[i], dtype=float) + jitter, nulls[i], s=16,
                   marker=MARKERS[classifier], facecolor=POINT_COLORS[classifier],
                   edgecolor=darker(POINT_COLORS[classifier]), linewidth=0.30, alpha=0.10, zorder=2)
    # redraw the null bar outlines and SD lines on top of the points
    ax.bar(null_x, means, width=width, facecolor="none", edgecolor="black", linewidth=INTERNAL_LINEWIDTH, zorder=2.5)
    ax.errorbar(null_x, means, yerr=sds, fmt="none", ecolor="black", elinewidth=INTERNAL_LINEWIDTH,
                capsize=5, capthick=INTERNAL_LINEWIDTH, zorder=2.6)
    ax.bar(observed_x, observed, width=width, color=[FILL_COLORS[c] for c in RECIPES], edgecolor="black",
           linewidth=INTERNAL_LINEWIDTH, zorder=3)
    for i, label in enumerate(labels):
        ax.text(x[i], 1.01, label, ha="center", va="bottom", fontsize=P_VALUE_SIZE, fontfamily="Arial",
                fontweight="normal", color=P_VALUE_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[c] for c in RECIPES])
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1.15)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_yticklabels([f"{value:.1f}" for value in np.arange(0, 1.1, 0.2)])
    ax.set_title("")
    style_axis(ax)


def plot_confusion(ax):
    """g: confusion matrix of mRMR + LogReg_L2."""
    table = pd.read_csv(CLASSIFICATION / "confusion_matrices.csv")
    table = table[(table["selector"] == "mrmr") & (table["classifier"] == "LogReg_L2")]
    index = {label: i for i, label in enumerate(PHENOTYPES)}
    matrix = np.zeros((3, 3), dtype=int)
    for row in table.itertuples(index=False):
        matrix[index[row.true_label], index[row.pred_label]] = int(row.count)

    ax.imshow(matrix, cmap=CONFUSION_CMAP, vmin=0, vmax=3, aspect="auto", interpolation="nearest")
    for row in range(3):
        for column in range(3):
            value = matrix[row, column]
            ax.text(column, row, str(value), ha="center", va="center", fontsize=FONT_SIZE, fontweight="normal",
                    color="white" if value >= 3 else "#1f2933")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    labels = [PHENOTYPE_LABELS[p] for p in PHENOTYPES]
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("")
    style_axis(ax)


def panels_d_to_g():
    """Fig. 4 d–g: classification results of the fed state.

    d  macro-F1 of the 81 selector-classifier combinations (violin, box and points); the diamonds mark
       mRMR + LogReg_L2 and mRMR + SVM_linear
    e  predictive confidence (largest class probability) of each rat for the same two recipes, with the
       two-sided exact Wilcoxon signed-rank test between them (paired by rat, n = 9)
    f  fixed-recipe label-permutation test (B = 1000): null mean +/- SD with the null values, and the
       observed macro-F1
    g  confusion matrix of mRMR + LogReg_L2 (mRMR + SVM_linear has the same matrix)"""
    fig, axes = create_panels(4)
    plot_macro_f1(axes[0])
    plot_confidence(axes[1])
    plot_permutation(axes[2])
    plot_confusion(axes[3])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        path = OUT_DIR / f"fig4_d_to_g.{extension}"
        fig.savefig(path, dpi=600)
        print(path)
    plt.close(fig)


# ---- h–i: features of the two reported recipes ----

PHENOTYPE_COLORS = {"healthy": "#607D8B", "pre-diabetic": "#A88962", "diabetic": "#8A4F4A"}
PAIRS = [("healthy", "pre-diabetic"), ("healthy", "diabetic"), ("pre-diabetic", "diabetic")]
TEXT_COLOR = "#1f2933"
HEDGES_COLOR = "#3F3F3F"

# h: selection counts
TOP_N = 12
HEATMAP_LABELS = {
    "glucose_avg_mM": "Glucose avg mM",
    "c_peptide_osc_signal_entropy": "C-peptide oscillation signal entropy",
    "beta_cell_insulin_cpeptide_correlation": "β-cell insulin/C-peptide coupling",
    "c_peptide_osc_frequency": "C-peptide oscillation frequency",
    "c_peptide_glucose_efficiency": "C-peptide–glucose efficiency",
    "glucagon_auc_glucose_corrected": "Glucose-corrected glucagon AUC",
    "c_peptide_auc_above_baseline": "C-peptide AUC above baseline",
    "c_peptide_osc_zero_crossings": "C-peptide osc. zero crossings",
    "c_peptide_rate_max_decrease": "C-peptide rate max decrease",
    "corr_c_peptide_glucagon_significant": "Corr. C-peptide glucagon significant",
    "corr_insulin_c_peptide": "Corr. insulin C-peptide",
    "glucagon_auc_glucose_efficiency": "Glucagon AUC glucose efficiency",
}
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "count_blue", ["#f7fbff", "#d8e9f5", "#a8cae2", "#5f98c3", "#1f4f7a"])
HEATMAP_WIDTH = 3.6       # inches
COLORBAR_GAP = 0.24
COLORBAR_WIDTH = 0.22
RIGHT_MARGIN = 1.25

# i: z-score panels, in panel order: feature, title, physiological axis
PANELS = [
    ("glucose_avg_mM", "Glucose avg mM", "Axis 1"),
    ("beta_cell_insulin_cpeptide_correlation", "β-cell insulin/C-peptide coupling", "Axis 2"),
    ("c_peptide_glucose_efficiency", "C-peptide–glucose efficiency", "Axis 2"),
    ("glucagon_auc_glucose_corrected", "Glucose-corrected glucagon AUC", "Axis 2"),
    ("c_peptide_osc_frequency", "C-peptide oscillation frequency", "Axis 3"),
    ("c_peptide_osc_signal_entropy", "C-peptide oscillation signal entropy", "Axis 3"),
]
TOP_ANNOTATION = {"glucose_avg_mM", "c_peptide_osc_signal_entropy"}  # Hedges' g at the top of these panels
Z_BORDER = 1.8 * 1.5
ANNOTATION_SIZE = 13.0 * 1.40 * 0.80 * 0.60 * 1.35 * 1.35 * 1.15
X_TICK_SIZE = 21.0 * 1.40 * 1.40 * 0.60
Y_TICK_SIZE = 17.85 * 1.40 * 1.40 * 0.60 * 1.20
TITLE_SIZE = 18.0 * 1.40 * 1.40 * 0.60
Y_LABEL_SIZE = 25.875 * 1.40 * 1.40 * 0.60
AXIS_COLORS = {"Axis 1": "#397EA8", "Axis 2": "#C48D35", "Axis 3": "#8661A5"}
HEADER_HEIGHT = 0.72      # inches
HEADER_GAP = 0.045
TAG_WIDTH = 0.90


def selection_table():
    """Folds in which mRMR selected each feature, for LR and SVM: the TOP_N most often selected features."""
    selected = pd.read_csv(CLASSIFICATION / "selected_features.csv")
    columns = {}
    for classifier in RECIPES:
        rows = selected[(selected["selector"] == "mrmr") & (selected["classifier"] == classifier)]
        columns[DISPLAY[classifier]] = rows["feature"].value_counts()
    table = pd.DataFrame(columns).fillna(0).astype(int).rename_axis("feature").reset_index()
    table["max_count"] = table[["LR", "SVM"]].max(axis=1)
    table["total_count"] = table[["LR", "SVM"]].sum(axis=1)
    table = table.sort_values(["max_count", "total_count", "feature"], ascending=[False, False, True])
    table = table.head(TOP_N).reset_index(drop=True)
    table["label"] = [HEATMAP_LABELS.get(feature, feature) for feature in table["feature"]]
    return table


def label_margin(table):
    """Width (inches) of the longest heatmap row label, plus a gap."""
    font = FontProperties(family="Arial", size=FONT_SIZE)
    return max(TextPath((0, 0), text, prop=font).get_extents().width / 72 for text in table["label"]) + 0.28


def plot_heatmap(ax, cax, table):
    plt.rcParams.update({"font.family": "Arial", "font.sans-serif": ["Arial"], "font.weight": "normal",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    matrix = table[["LR", "SVM"]].to_numpy()
    im = ax.imshow(matrix, cmap=HEATMAP_CMAP, vmin=1, vmax=7, aspect="auto")
    ax.set_yticks(range(len(table)), labels=table["label"])
    ax.set_xticks([0, 1], labels=["LR", "SVM"])
    style_axis(ax)
    ax.tick_params(axis="both", length=0, pad=8)
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            ax.text(x, y, f"{matrix[y, x]}/9", ha="center", va="center", fontfamily="Arial", fontsize=FONT_SIZE,
                    color="white" if matrix[y, x] >= 5 else "black")
    cbar = ax.figure.colorbar(im, cax=cax, ticks=range(1, 8))
    cbar.set_label("LOO folds selected", fontsize=FONT_SIZE, fontfamily="Arial", color="black", labelpad=14)
    cbar.ax.tick_params(labelsize=FONT_SIZE, length=0, pad=7, labelcolor="black")
    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(BORDER_LINEWIDTH)


def hedges_g(values_a, values_b):
    """Hedges' g of b minus a, with the pooled SD."""
    n_a, n_b = len(values_a), len(values_b)
    pooled_df = n_a + n_b - 2
    pooled_variance = ((n_a - 1) * float(np.std(values_a, ddof=1)) ** 2
                       + (n_b - 1) * float(np.std(values_b, ddof=1)) ** 2) / pooled_df
    cohen_d = float(np.mean(values_b) - np.mean(values_a)) / np.sqrt(pooled_variance)
    return float((1 - (3 / (4 * pooled_df - 1))) * cohen_d)


def zscore_figure():
    """The six z-score panels, as a figure of its own."""
    x_df = pd.read_csv(FEATURES / "fedstate_X.csv")
    meta = pd.read_csv(FEATURES / "fedstate_meta.csv")
    order = meta.assign(
        phenotype_order=meta["rat_type"].map({label: i for i, label in enumerate(PHENOTYPES)}),
        rat_number=meta["animal_id"].str.replace("Rat", "", regex=False).astype(int),
    ).sort_values(["phenotype_order", "rat_number"]).index
    phenotype = meta.loc[order, "rat_type"].to_numpy()
    values = x_df.loc[order, [feature for feature, _, _ in PANELS]].astype(float).reset_index(drop=True)
    z_values = ((values - values.mean(axis=0)) / values.std(axis=0, ddof=0)).fillna(0.0)

    extremes = []
    for feature, _, _ in PANELS:
        extremes.extend(np.abs(z_values[feature]).tolist())
        for group in PHENOTYPES:
            group_values = z_values[feature].to_numpy()[phenotype == group]
            mean, sd = float(np.mean(group_values)), float(np.std(group_values, ddof=1))
            extremes.extend([abs(mean - sd), abs(mean + sd)])
    y_lim = max(2.0, np.ceil(max(extremes) * 2) / 2 + 0.25)

    plt.rcParams.update({"font.family": "Arial", "font.sans-serif": ["Arial"], "pdf.fonttype": 42, "ps.fonttype": 42,
                         "font.weight": "normal", "axes.titleweight": "normal", "axes.labelweight": "normal",
                         "figure.dpi": 150, "savefig.dpi": 600})
    fig, axes = plt.subplots(2, 3, figsize=(15.28299756097561, 8.8), sharex=True, sharey=True)
    axes = axes.ravel()
    x = np.arange(len(PHENOTYPES))
    for index, (ax, (feature, title, _)) in enumerate(zip(axes, PANELS)):
        groups = [z_values[feature].to_numpy()[phenotype == group] for group in PHENOTYPES]
        error_width = 1.8 * 1.40 * 1.20 * 1.20
        ax.bar(x, [float(np.mean(g)) for g in groups], yerr=[float(np.std(g, ddof=1)) for g in groups],
               color=[PHENOTYPE_COLORS[group] for group in PHENOTYPES], edgecolor="black", linewidth=1.8,
               alpha=0.38, width=0.62, capsize=3.5,
               error_kw={"elinewidth": error_width, "capthick": error_width, "ecolor": "black", "zorder": 4},
               zorder=2)
        for i, (group, group_values) in enumerate(zip(PHENOTYPES, groups)):
            ax.scatter(i + np.array([-0.12, 0.0, 0.12])[:len(group_values)], group_values,
                       s=58 * 1.40 * 1.20 * 1.20, color=PHENOTYPE_COLORS[group], edgecolor="black", linewidth=1.0,
                       zorder=3)

        lines = []
        for a, b in PAIRS:
            g = hedges_g(groups[PHENOTYPES.index(a)], groups[PHENOTYPES.index(b)])
            lines.append(f"g({PHENOTYPE_LABELS[b]}-{PHENOTYPE_LABELS[a]})={g:+.2f}")
        top = feature in TOP_ANNOTATION
        ax.text(0.40 if top else 0.50, 0.93 if top else 0.07, "\n".join(lines), transform=ax.transAxes,
                ha="center", va="top" if top else "bottom", fontsize=ANNOTATION_SIZE, color=HEDGES_COLOR)
        ax.axhline(0, color="black", linewidth=1.8, zorder=1)
        ax.set_ylim(-y_lim, y_lim)
        ax.set_yticks([-2, -1, 0, 1, 2])
        ax.set_xticks(x)
        ax.set_title("\n".join(textwrap.wrap(title, width=24, break_long_words=False)), fontsize=TITLE_SIZE, pad=10)
        tick_width = max(1.0, Z_BORDER * 0.6)
        ax.tick_params(axis="x", length=7.0, width=tick_width, direction="out", bottom=True, top=False,
                       labelbottom=False, labelsize=X_TICK_SIZE, colors=TEXT_COLOR)
        ax.tick_params(axis="y", length=7.0, width=tick_width, direction="out", left=True, right=False,
                       labelsize=Y_TICK_SIZE, colors=TEXT_COLOR, labelleft=index % 3 == 0)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(Z_BORDER)
    for ax in axes[:3]:
        ax.set_xticks(x)
        ax.tick_params(axis="x", labelbottom=False, bottom=True)
    for ax in axes[-3:]:
        ax.set_xticks(x)
        ax.set_xticklabels([PHENOTYPE_LABELS[group] for group in PHENOTYPES], rotation=0, ha="center",
                           fontsize=X_TICK_SIZE)
        ax.tick_params(axis="x", labelbottom=True)
    fig.text(0.025, 0.54, "Feature z-score", rotation=90, ha="center", va="center", fontsize=Y_LABEL_SIZE,
             color=TEXT_COLOR)
    fig.subplots_adjust(left=0.0957272939528368, right=0.98488516411271, top=0.88, bottom=0.19, wspace=0.126,
                        hspace=0.62)
    return fig, list(axes)


def add_axis_headers(fig, axes):
    """Replace each panel title by a header in its axis colour, with an axis tag on the left."""
    figure_width, figure_height = fig.get_size_inches()
    for ax, (_, _, axis) in zip(axes, PANELS):
        title, title_size = ax.get_title(), ax.title.get_fontsize()
        ax.set_title("")
        frame = ax.get_position()
        bottom = frame.y1 + HEADER_GAP / figure_height
        height = HEADER_HEIGHT / figure_height
        tag_width = TAG_WIDTH / figure_width
        rgb = to_rgb(AXIS_COLORS[axis])
        # header: 12% of the axis colour blended with white; tag: 55%
        fig.add_artist(Rectangle((frame.x0, bottom), frame.width, height, transform=fig.transFigure,
                                 facecolor=tuple(0.12 * c + (1 - 0.12) for c in rgb), edgecolor="none",
                                 clip_on=False))
        fig.add_artist(Rectangle((frame.x0, bottom), tag_width, height, transform=fig.transFigure,
                                 facecolor=tuple(0.55 * c + (1 - 0.55) for c in rgb), edgecolor="none",
                                 clip_on=False))
        fig.text(frame.x0 + tag_width / 2, bottom + height / 2, axis, ha="center", va="center", fontfamily="Arial",
                 fontsize=16, fontweight="bold", color="#333333")
        fig.text(frame.x0 + tag_width + (frame.width - tag_width) / 2, bottom + height / 2, title, ha="center",
                 va="center", multialignment="center", fontfamily="Arial", fontsize=title_size,
                 fontweight="normal", color="black")


def panels_h_i():
    """Fig. 4 h–i: the features of the two reported fed-state recipes.

    h  number of outer folds (of 9) in which mRMR selected each feature, for LogReg_L2 and SVM_linear
       (the 12 most often selected features)
    i  phenotype mean +/- SD of the z-scored values of six of these features, with the value of each rat and
       Hedges' g of the three phenotype pairs; the coloured headers give the physiological axis of each feature"""
    fig, z_axes = zscore_figure()
    table = selection_table()

    # widen the figure to the left for the heatmap, keeping the size of every z-score panel
    old_width, height = fig.get_size_inches()
    left = label_margin(table)
    cbar_left = left + HEATMAP_WIDTH + COLORBAR_GAP
    offset = cbar_left + COLORBAR_WIDTH + RIGHT_MARGIN + 0.1
    width = old_width + offset
    frames = [ax.get_position().frozen() for ax in z_axes]
    fig.set_size_inches(width, height)
    for ax, frame in zip(z_axes, frames):
        ax.set_position([(offset + frame.x0 * old_width) / width, frame.y0, frame.width * old_width / width,
                         frame.height])
    for text in fig.texts:
        x, y = text.get_position()
        text.set_position(((offset + x * old_width) / width, y))
    add_axis_headers(fig, z_axes)

    # heatmap from the bottom of the z-score panels to the top of their headers
    bottom = min(frame.y0 for frame in frames)
    top = max(frame.y1 for frame in frames) + (HEADER_HEIGHT + HEADER_GAP) / height
    ax = fig.add_axes([left / width, bottom, HEATMAP_WIDTH / width, top - bottom])
    cax = fig.add_axes([cbar_left / width, bottom, COLORBAR_WIDTH / width, top - bottom])
    plot_heatmap(ax, cax, table)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig4_h_i.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig4_h_i.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(OUT_DIR / "fig4_h_i.pdf")


# ---- j: SHAP waterfalls ----

DPI = 600

POSITIVE = "#ff0051"
NEGATIVE = "#1597e5"
TITLE_COLORS = {"healthy": "#587b8d", "pre-diabetic": "#b48a55", "diabetic": "#a35b52"}
FEATURE_LABELS = {
    "glucose_avg_mM": "Glucose mean",
    "beta_cell_insulin_cpeptide_correlation": "Ins-Cpep corr.",
    "insulin_peak_total_excursion": "Insulin peak exc.",
    "c_peptide_glucose_efficiency": "C-pep glucose eff.",
    "c_peptide_osc_signal_entropy": "C-pep entropy",
    "insulin_glucose_corrected_5p6": "Ins/glucose corrected",
    "glucagon_auc_glucose_corrected": "Gluc. AUC/glucose",
    "insulin_glucose_correlation_pvalue": "Insulin-glucose corr. p",
    "c_peptide_rate_max_decrease": "C-pep max drop",
    "glucagon_glucose_efficiency": "Gluc. glucose eff.",
    "glucagon_auc_glucose_efficiency": "Gluc. AUC eff.",
    "glucagon_peak_total_excursion": "Gluc. peak excursion",
    "c_peptide_auc_above_baseline": "C-pep AUC/base",
    "glucagon_kinetics_peak_count": "Gluc. peak count",
    "corr_c_peptide_glucagon_significant": "Cpep-glucagon corr.",
    "c_peptide_osc_zero_crossings": "C-pep zero crossings",
    "c_peptide_osc_frequency": "C-pep freq.",
    "glucagon_glucose_corrected_5p6": "Gluc./glucose corr.",
    "corr_insulin_c_peptide": "Ins-Cpep corr.",
    "insulin_glucose_peak_efficiency": "Ins-glucose peak eff.",
}
# Contribution labels of short arrows are printed outside, on the given side
OUTSIDE_LABELS = {
    "Rat1": {"features": {"glucose_avg_mM", "beta_cell_insulin_cpeptide_correlation"}, "side": "right"},
    "Rat2": {"features": {"glucose_avg_mM", "c_peptide_glucose_efficiency"}, "side": "right"},
    "Rat3": {"all": True, "side": "left"},
    "Rat4": {"all": True, "side": "right"},
    "Rat5": {"all": True, "side": "right"},
    "Rat6": {"all": True, "side": "right"},
    "Rat7": {"all": True, "side": "right"},
    "Rat8": {"all": True, "side": "left"},
    "Rat9": {"all": True, "side": "left"},
}
SIZE = 1.25 * 1.15 * 1.15                  # scale of the arrows and their labels
CONTRIBUTION_SIZE = 13.5 * 1.25 * 1.15 * 1.15


def raw_value_text(value):
    magnitude = abs(value)
    if magnitude < 1:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    if magnitude < 100:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.0f}"


def contribution_text(value):
    if 0 < abs(value) < 0.005:
        return "<0.01" if value > 0 else ">-0.01"
    return f"{value:+.2f}"


def arrow(ax, start, end, y, height, value, force_outside=False, outside_side=None):
    """One waterfall arrow from start to end, labelled with its contribution."""
    width = abs(end - start)
    head = min(0.045 * 1.15 * 1.15, max(width * 0.42 * 1.15 * 1.15, 0.008 * 1.15 * 1.15))
    half_height = height / 2
    shoulder = max(start, end - head) if end >= start else min(start, end + head)
    color = POSITIVE if end >= start else NEGATIVE
    points = [(start, y - half_height), (shoulder, y - half_height), (end, y),
              (shoulder, y + half_height), (start, y + half_height)]
    ax.add_patch(Polygon(points, closed=True, facecolor=color, edgecolor="#202020", lw=0.8 * 1.25 * 1.15 * 1.15))

    label = contribution_text(value)
    if width >= 0.075 and not force_outside:
        ax.text((start + end) / 2, y, label, color="white", ha="center", va="center", fontsize=CONTRIBUTION_SIZE)
    elif outside_side in {"left", "right"}:
        label_x = max(start, end) + 0.018 if outside_side == "right" else min(start, end) - 0.018
        ax.text(label_x, y, label, color=color, ha="left" if outside_side == "right" else "right", va="center",
                fontsize=CONTRIBUTION_SIZE)
    else:
        offset = 0.014 if end >= start else -0.014
        ax.text(end + offset, y, label, color=color, ha="left" if end >= start else "right", va="center",
                fontsize=CONTRIBUTION_SIZE)


def draw_panel(ax, animal, rows, baseline):
    """Waterfall of one held-out rat."""
    n_rows = len(rows)
    y_positions = np.arange(n_rows - 1, -1, -1)
    cumulative = float(baseline["base_value"])
    endpoints = []
    settings = OUTSIDE_LABELS.get(animal, {})
    for y, row in zip(y_positions, rows.itertuples(index=False)):
        end = cumulative + float(row.shap_value)
        force_outside = bool(settings.get("all", False)) or row.feature in settings.get("features", set())
        side = settings.get("side")
        if animal == "Rat5" and row.feature in {"c_peptide_auc_above_baseline", "glucagon_kinetics_peak_count",
                                                "glucose_avg_mM"}:
            side = "left"
        if animal == "Rat6" and row.feature == "c_peptide_glucose_efficiency":
            force_outside = False
        arrow(ax, cumulative, end, float(y), 0.43 * 1.25 * 1.15 * 1.15, float(row.shap_value),
              force_outside=force_outside, outside_side=side if force_outside else None)
        endpoints.append(end)
        cumulative = end
    for i in range(len(endpoints) - 1):
        ax.plot([endpoints[i], endpoints[i]],
                [y_positions[i] - 0.22 * 1.25 * 1.15 * 1.15, y_positions[i + 1] + 0.22 * 1.25 * 1.15 * 1.15],
                color="#d6d6d6", lw=0.8, ls=(0, (1.2, 2.0)), zorder=0)

    base_value, final_value = float(baseline["base_value"]), float(baseline["final_value"])
    shift = (0.08 if animal in {"Rat3", "Rat5", "Rat6", "Rat8", "Rat9"} else 0.16) * abs(final_value - base_value)
    e_shift, f_shift, y_label = shift, shift, 0.89
    if animal == "Rat3":
        e_shift, y_label = 0.02 * abs(final_value - base_value), 0.94
    elif animal == "Rat6":
        f_shift = 0.02 * abs(final_value - base_value)
    ax.axvline(base_value, color="#9da5ab", lw=1.3, ls=(0, (1.2, 1.8)), zorder=0)
    ax.axvline(final_value, color="#56616a", lw=1.3, ls=(0, (1.2, 1.8)), zorder=0)
    transform = ax.get_xaxis_transform()
    ax.text(max(0.03, base_value - e_shift), y_label, "E", transform=transform, ha="center", va="center",
            fontsize=18.45 * 1.25)
    ax.text(min(0.97, final_value + f_shift), y_label, "F", transform=transform, ha="center", va="center",
            fontsize=18.45 * 1.25)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{raw_value_text(float(row.raw_value))} = {FEATURE_LABELS.get(row.feature, row.feature)}"
                        for row in rows.itertuples(index=False)], fontsize=18.9 * 1.15 * 1.15 * 1.05)
    true_label, pred_label = str(baseline["true_label"]), str(baseline["pred_label"])
    outcome = "correct" if true_label == pred_label else "misclassified"
    ax.set_title(f"{animal} | {PHENOTYPE_LABELS[true_label]} {outcome}: pred. {PHENOTYPE_LABELS[pred_label]}",
                 fontsize=17.25 * 1.25 * 1.17, color=TITLE_COLORS[true_label], pad=18.0)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.6, max(n_rows - 0.02, 0.6))
    ax.set_xticks([0.0, 0.5, 1.0])
    for spine in ax.spines.values():
        spine.set_linewidth(3.75)
        spine.set_color("#222222")
    ax.tick_params(axis="x", labelsize=23.7 * 1.25, length=3, width=0.8)
    ax.tick_params(axis="y", length=2, width=0.7, pad=2)


def panel_j():
    """Fig. 4 j: fold-specific SHAP waterfalls of mRMR + LogReg_L2 for all nine held-out rats.

    Each panel starts from the base value E of the rat's true class and adds the SHAP value of every feature
    selected in its fold, largest first, to reach the predicted true-class probability F. Red arrows increase
    and blue arrows decrease the true-class output; the rat's value of each feature is printed on the left.
    The colour legend is written as a separate file."""
    plt.rcParams.update({"font.family": "Arial", "font.sans-serif": ["Arial"], "pdf.fonttype": 42,
                         "ps.fonttype": 42, "font.weight": "normal"})
    shap_values = pd.read_csv(SHAP_DIR / "shap_values.csv")
    baselines = pd.read_csv(SHAP_DIR / "shap_base_values.csv")
    x = pd.read_csv(FEATURES / "fedstate_X.csv")
    meta = pd.read_csv(FEATURES / "fedstate_meta.csv")
    rat_to_row = {str(animal): i for i, animal in enumerate(meta["animal_id"])}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 3, figsize=(30.152325225629376, 12.10416), squeeze=False)
    for ax, animal in zip(axes.flat, baselines.sort_values("fold")["animal_id"].astype(str)):
        rows = shap_values[(shap_values["animal_id"] == animal) & (shap_values["class"] == shap_values["true_label"])]
        rows = rows.assign(abs_shap=rows["shap_value"].abs()).sort_values("abs_shap", ascending=False)
        rows = rows.reset_index(drop=True)
        rows["raw_value"] = [float(x.iloc[rat_to_row[animal]][feature]) for feature in rows["feature"]]
        draw_panel(ax, animal, rows, baselines.loc[baselines["animal_id"] == animal].iloc[0])
    fig.subplots_adjust(left=0.15, right=0.9868665850133708, top=0.959158, bottom=0.126608, wspace=0.97,
                        hspace=0.45696)
    fig.canvas.draw()
    tight = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.05)
    bbox = Bbox.from_extents(tight.x0 - 0.17 / 72, tight.y0, tight.x1 + 8.104 / 72, tight.y1 + 12.103 / 72)
    fig.savefig(OUT_DIR / "fig4_j.png", dpi=DPI, bbox_inches=bbox, pad_inches=0)
    fig.savefig(OUT_DIR / "fig4_j.pdf", bbox_inches=bbox, pad_inches=0)
    plt.close(fig)

    legend = plt.figure(figsize=(9.2, 0.7))
    legend.legend(handles=[Patch(facecolor=POSITIVE, edgecolor="#202020", label="Increases true-class output"),
                           Patch(facecolor=NEGATIVE, edgecolor="#202020", label="Decreases true-class output")],
                  loc="center", ncol=2, frameon=False, fontsize=17.25, handlelength=1.8)
    legend.savefig(OUT_DIR / "fig4_j_legend.png", dpi=DPI, bbox_inches="tight", pad_inches=0.04)
    legend.savefig(OUT_DIR / "fig4_j_legend.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(legend)
    print(OUT_DIR / "fig4_j.pdf")


def main():
    for draw in (panels_d_to_g, panels_h_i, panel_j):
        with plt.rc_context():          # every figure starts from the default settings
            draw()


if __name__ == "__main__":
    main()
