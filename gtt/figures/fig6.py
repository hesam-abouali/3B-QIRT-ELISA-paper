#!/usr/bin/env python3
"""Fig. 6 a–h: clinical indices, hormone coupling and the classification features of the GTT.

a    fasting HOMA-IR, HOMA-β and QUICKI (outputs/figures/fig6_a.pdf and .png)
b    insulin–glucagon trajectory of each rat (fig6_b)
c    insulin–glucagon correlation (fig6_c)
d    adapted Matsuda index (fig6_d)
e–f  features selected by the perfect combinations, and their z-scores (fig6_e_f)
g    principal component analysis of the 181 features (fig6_g)
h    hormone family and feature type of the selected features, GTT and fed state (fig6_h)

Input   data/concentrations/concentration_timeseries.csv, data/raw/gtt_post_injection.csv,
        data/classification/scores_by_assignment.csv, selected_features_true_labels.csv,
        data/features/gtt_X.csv, gtt_meta.csv, feature_families_and_types.csv,
        ../fed_state/data/classification/selected_features.csv

    python figures/fig6.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import AutoMinorLocator, FixedLocator
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from figure_style import (BORDER, FONT_SIZE, LZ, ROOT, ROW_HEIGHT, ZDF, label_margin, nominal_time, save,
                          style_table_axis, use_arial)

PROFILES = ROOT / "data" / "concentrations" / "concentration_timeseries.csv"
RAW = ROOT / "data" / "raw" / "gtt_post_injection.csv"
CLASSIFICATION = ROOT / "data" / "classification"
FEATURES = ROOT / "data" / "features"
RATS = LZ + ZDF
MARKERS = dict(zip(RATS, ["o", "s", "^", "D", "v", "X"]))


# ---- a, c, d: clinical indices and the insulin–glucagon correlation ----

RAT_LABELS = {rat: f"R{i % 3 + 1}" for i, rat in enumerate(RATS)}
BAR_COLORS = ((0, "#607D8B", "#40545E"), (1, "#8A4F4A", "#633834"))   # (x, fill, edge): LZ, ZDF
INK, REFERENCE = "#111111", "#888888"
LABEL_SIZE, TICK_SIZE, RAT_LABEL_SIZE, P_SIZE = 35.578125, 33.0, 28.25, 28.5
# (index, y label, x label, log values for the test)
FASTING = [("homa_ir", "HOMA-IR", "Insulin resistance (fasting)", True),
           ("homa_beta", "HOMA-β (%)", "β-cell function (fasting)", True),
           ("quicki", "QUICKI", "Insulin sensitivity (fasting)", False)]
MATSUDA = [("matsuda", "Matsuda index", "Insulin sensitivity (dynamic)", True)]


def clinical_indices(profiles, raw):
    """HOMA-IR, HOMA-β, QUICKI and the adapted Matsuda index of each rat."""
    rows = {}
    for rat in RATS:
        profile = profiles[profiles.animal_id == rat]
        anchors = raw[(raw.animal_id == rat) & raw.glucose_mM.notna()].sort_values("timepoint")
        fasting_insulin = profile.loc[profile.timepoint < 0, "insulin_predicted_pM"].mean()
        g0 = float(anchors.glucose_mM.iloc[0])
        insulin_uU = fasting_insulin / 6.0
        # device insulin at the sampling times (nearest sample), with the fasting mean at t = 0
        insulin = [profile.loc[(profile.timepoint - t).abs().idxmin(), "insulin_predicted_pM"]
                   for t in anchors.timepoint]
        insulin[0] = fasting_insulin
        product = (g0 * 18.0 * fasting_insulin * 0.167) * (anchors.glucose_mM.mean() * 18.0 * np.mean(insulin) * 0.167)
        rows[rat] = {"homa_ir": g0 * insulin_uU / 22.5,
                     "homa_beta": 20.0 * insulin_uU / (g0 - 3.5),
                     "quicki": 1.0 / (np.log10(insulin_uU) + np.log10(g0 * 18.0182)),
                     "matsuda": 10000.0 / np.sqrt(product)}
    return pd.DataFrame(rows).T


def correlations(profiles, first, second):
    """Pearson r between two calibrated profiles of each rat over the 24 samples from t = 0."""
    post = profiles[profiles.timepoint >= 0]
    return pd.Series({rat: np.corrcoef(post.loc[post.animal_id == rat, f"{first}_predicted_pM"],
                                       post.loc[post.animal_id == rat, f"{second}_predicted_pM"])[0, 1]
                      for rat in RATS})


def p_label(p):
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.2f}" if p >= 0.1 else f"p = {p:.3f}"


def label_offsets(ys, top):
    """Label placement of the three rats of a bar (offset in points, ha, va), in left-centre-right order:
    highest above, lowest below, middle to a side; all three above when they are close together."""
    ny = [y / top for y in ys]
    specs = [None, None, None]
    if max(ny) - min(ny) < 0.13:
        return [(off, "center", "bottom") for off in ((-22, 20), (0, 30), (22, 20))]
    low, mid, high = sorted(range(3), key=lambda i: ny[i])
    specs[high] = ((34, 12), "left", "bottom") if high == 1 else ((0, 20), "center", "bottom")
    if ny[low] < 0.11:
        specs[low] = ((-26, 3), "right", "center") if low == 0 else ((26, 3), "left", "center")
    elif low == 1:
        specs[low] = ((34, -12), "left", "top")
    else:
        specs[low] = ((0, -20), "center", "top")
    specs[mid] = ((-26, 0), "right", "center") if mid == 0 else ((26, 4 if mid == 1 else 0), "left", "center")
    return specs


def style_axis(ax):
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color(INK)
        spine.set_linewidth(2.4)
    ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE, width=1.6, length=7.0, color=INK, labelcolor=INK)
    ax.tick_params(axis="both", which="minor", width=1.2, length=4.0, color=INK)
    ax.xaxis.label.set_size(LABEL_SIZE)
    ax.yaxis.label.set_size(LABEL_SIZE)


def bars(ax, values):
    """Group mean +/- SD bars; returns the LZ and ZDF values."""
    groups = values.loc[LZ].to_numpy(float), values.loc[ZDF].to_numpy(float)
    for (x, color, edge), group in zip(BAR_COLORS, groups):
        ax.bar(x, group.mean(), width=0.58, color=color, alpha=0.34, edgecolor=edge, linewidth=3.0, zorder=2)
        ax.errorbar(x, group.mean(), yerr=group.std(ddof=1), fmt="none", ecolor=INK, elinewidth=2.2, capsize=8.0,
                    capthick=2.2, zorder=3)
    return groups


def bracket(ax, y, drop, text_y, p):
    ax.plot([0, 0, 1, 1], [y - drop, y, y, y - drop], color=INK, lw=2.0, clip_on=False, zorder=7)
    ax.text(0.5, text_y, p_label(p), ha="center", va="bottom", fontsize=P_SIZE, color=INK)


def index_figure(table, panels, stem, label_override=None):
    """One bar panel per (column, y label, x label, log values for the test) in panels; label_override places
    the label of a (column, rat) by hand."""
    fig, axes = plt.subplots(1, len(panels), figsize=(8.4 * len(panels), 8.1), squeeze=False)
    for ax, (column, ylabel, xlabel, use_log) in zip(axes.flat, panels):
        values = table[column].astype(float)
        top = float(values.max()) * 1.45
        ax.set_ylim(0, top)
        lz, zdf = bars(ax, values)
        for (x, color, _), rats in zip(BAR_COLORS, (LZ, ZDF)):
            ys = [float(values[rat]) for rat in rats]
            for rat, dx, y, (offset, ha, va) in zip(rats, (-0.13, 0.0, 0.13), ys, label_offsets(ys, top)):
                offset, ha, va = (label_override or {}).get((column, rat), (offset, ha, va))
                ax.scatter(x + dx, y, s=352.5, marker=MARKERS[rat], facecolor=color, edgecolor="white",
                           linewidth=1.7, zorder=5)
                ax.annotate(RAT_LABELS[rat], (x + dx, y), textcoords="offset points", xytext=offset, ha=ha, va=va,
                            fontsize=RAT_LABEL_SIZE, color=INK, zorder=6)
        ax.set_xlim(-0.48, 1.48)
        ax.set_xticks([0, 1], labels=["LZ", "ZDF"])
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.set_ylabel(ylabel, labelpad=14)
        ax.set_xlabel(xlabel, labelpad=16)
        style_axis(ax)
        a, b = (np.log(lz), np.log(zdf)) if use_log else (lz, zdf)
        bracket(ax, top * 0.90, top * 0.035, top * 0.905, float(ttest_ind(a, b).pvalue))
        print(f"  {ylabel}: LZ {', '.join(f'{v:.3g}' for v in lz)}; ZDF {', '.join(f'{v:.3g}' for v in zdf)}; "
              f"p = {ttest_ind(a, b).pvalue:.4f}")
    fig.subplots_adjust(left=min(0.052 * 4 / len(panels), 0.21), right=0.995 if len(panels) > 1 else 0.98,
                        bottom=0.235, top=0.965, wspace=0.30)
    save(fig, stem, bbox_inches="tight", facecolor="white")


def correlation_figure(r, xlabel, stem, label_at=None):
    """Bars of the per-rat r; label_at places the label of a rat by hand: (dx, dy) from its symbol, ha, va."""
    fig, ax = plt.subplots(figsize=(9.0, 8.1))
    lz, zdf = bars(ax, r)
    for (x, color, _), rats in zip(BAR_COLORS, (LZ, ZDF)):
        for dx, rat in zip((-0.13, 0.0, 0.13), rats):
            value = float(r[rat])
            ax.scatter(x + dx, value, s=352.5, marker=MARKERS[rat], facecolor=color, edgecolor="white",
                       linewidth=1.7, zorder=5)
            shift = 0.075 if value <= 0.82 else -0.10     # label above the symbol, below it near the top
            ldx, ldy, ha, va = (0.0, shift, "center", "bottom" if shift > 0 else "top")
            if rat == "Rat2":                             # below the low LZ R2 point
                ldx, ldy, va = -0.05, -0.105, "top"
            ldx, ldy, ha, va = (label_at or {}).get(rat, (ldx, ldy, ha, va))
            ax.text(x + dx + ldx, value + ldy, RAT_LABELS[rat], ha=ha, va=va, fontsize=23.25, color=INK, zorder=6)
    ax.axhline(0, color=REFERENCE, ls="--", lw=2.2, zorder=2)
    ax.set_xlim(-0.48, 1.48)
    ax.set_ylim(-1.12, 1.36)
    ax.set_xticks([0, 1], labels=["LZ", "ZDF"])
    ax.xaxis.set_minor_locator(FixedLocator([0.5]))
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlabel(xlabel, labelpad=14)
    ax.set_ylabel("Pearson r (t ≥ 0 min)", labelpad=18)
    style_axis(ax)
    p = float(ttest_ind(np.arctanh(lz), np.arctanh(zdf)).pvalue)
    bracket(ax, 1.06, 0.055, 1.08, p)
    print(f"  {xlabel}: LZ {', '.join(f'{v:.2f}' for v in lz)}; ZDF {', '.join(f'{v:.2f}' for v in zdf)}; "
          f"p = {p:.4f}")
    fig.subplots_adjust(left=0.205, right=0.93, bottom=0.24, top=0.975)
    extra = fig.get_default_bbox_extra_artists() + [ax.xaxis.label, ax.yaxis.label]
    save(fig, stem, bbox_inches="tight", bbox_extra_artists=extra, facecolor="white")


def panel_a():
    """Fasting HOMA-IR, HOMA-β and QUICKI, from the device insulin (mean of the four pre-injection samples) and the
    glucose at t = 0:  HOMA-IR = G I / 22.5,  HOMA-β = 20 I / (G - 3.5),  QUICKI = 1 / (log10 I + log10 G'), with I in
    µU/mL (pM / 6), G in mM and G' in mg/dL (mM x 18.0182). Bars: group mean +/- SD, with each rat (R1–R3 within
    each group). p: two-sided Student's t-test on log values (HOMA-IR, HOMA-β) or raw values (QUICKI)."""
    use_arial()
    indices = clinical_indices(pd.read_csv(PROFILES), pd.read_csv(RAW))
    index_figure(indices, FASTING, "fig6_a")


def panel_c():
    """Pearson r between the insulin and glucagon profiles of each rat over the 24 samples from t = 0, with the group
    mean +/- SD. p: two-sided Student's t-test on Fisher z-transformed r."""
    use_arial()
    profiles = pd.read_csv(PROFILES)
    correlation_figure(correlations(profiles, "insulin", "glucagon"), "Insulin–glucagon correlation", "fig6_c")


def panel_d():
    """Adapted Matsuda index, 10000 / sqrt(G0 I0 Gmean Imean), from the glucose and the device insulin at the
    blood-sampling times (G in mg/dL = mM x 18, I in µU/mL = pM x 0.167; I0 = the pre-injection mean). Bars as in a;
    p: two-sided Student's t-test on log values."""
    use_arial()
    indices = clinical_indices(pd.read_csv(PROFILES), pd.read_csv(RAW))
    # ZDF R2 labelled left of its symbol, clear of R3
    index_figure(indices, MATSUDA, "fig6_d", label_override={("matsuda", "Rat8"): ((-26, 0), "right", "center")})


# ---- b: insulin–glucagon trajectories ----

TITLES = dict(zip(LZ + ZDF, ["LZ R1", "LZ R2", "LZ R3", "ZDF R1", "ZDF R2", "ZDF R3"]))
PAGE_W, PAGE_H = 930.32 / 72.0, 566.075 / 72.0       # inches
TICK, LABEL, TITLE, CBAR_TICK, CBAR_LABEL, LEGEND = 22.0, 26.0, 26.0, 22.0, 24.0, 22.0
LINE_W, SPINE_W = 4.0, 1.9
# layout, inches
LEFT, CBAR_SPACE, YTICK_SPACE, CBAR_GAP = 0.50, 1.45, 0.98, 0.18
TOP, ROW_TITLE, XTICK_SPACE, BOTTOM = 0.06, 0.50, 0.52, 0.95


def set_round_axis(axis, set_lim, values, nbins):
    """Axis from one labelled tick to another, enclosing the data widened by 4% on each side (not below 0):
    the step is the smallest of 1, 2, 2.5, 5 x 10^k that covers the range in at most nbins intervals."""
    low, high = float(np.min(values)), float(np.max(values))
    pad = 0.04 * (high - low if high > low else max(abs(low), 1.0))
    lo, hi = max(low - pad, 0.0) if low >= 0 else low - pad, high + pad
    exponent = np.floor(np.log10((hi - lo) / nbins)) - 1
    for k in np.arange(exponent, exponent + 4):
        for mantissa in (1.0, 2.0, 2.5, 5.0):
            step = mantissa * 10.0 ** k
            first, last = np.floor(lo / step + 1e-9) * step, np.ceil(hi / step - 1e-9) * step
            if round((last - first) / step) <= nbins:
                set_lim(first, last)
                ticks = first + step * np.arange(round((last - first) / step) + 1)
                axis.set_major_locator(FixedLocator(np.round(ticks, 10)))
                return
    raise RuntimeError("no tick step found")


def trajectory_grid(profiles, x_hormone, y_hormone, x_title, y_title, stem, x_nbins=3, ytick_space=YTICK_SPACE):
    """One panel per rat: y_hormone against x_hormone from t = 0, coloured by time."""
    post = profiles[profiles.timepoint >= 0].sort_values(["animal_id", "timepoint"])
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    norm, cmap = Normalize(vmin=0.0, vmax=60.0), plt.get_cmap("viridis")
    cell_w = (PAGE_W - LEFT - CBAR_SPACE) / 3.0
    plot_w = cell_w - ytick_space
    plot_h = (PAGE_H - TOP - BOTTOM - 2 * (ROW_TITLE + XTICK_SPACE)) / 2.0

    axes = []
    for k, rat in enumerate(LZ + ZDF):
        row, col = divmod(k, 3)
        left = LEFT + col * cell_w + ytick_space
        top = PAGE_H - TOP - ROW_TITLE - row * (ROW_TITLE + plot_h + XTICK_SPACE)
        ax = fig.add_axes([left / PAGE_W, (top - plot_h) / PAGE_H, plot_w / PAGE_W, plot_h / PAGE_H])
        axes.append(ax)
        data = post[post.animal_id == rat]
        x, y = data[f"{x_hormone}_predicted_pM"].to_numpy(float), data[f"{y_hormone}_predicted_pM"].to_numpy(float)
        time = np.round(nominal_time(data.timepoint), 3)
        points = np.column_stack([x, y]).reshape(-1, 1, 2)
        lines = LineCollection(np.concatenate([points[:-1], points[1:]], axis=1), cmap=cmap, norm=norm,
                               linewidth=LINE_W, capstyle="round", joinstyle="round", zorder=2)
        lines.set_array((time[:-1] + time[1:]) / 2.0)
        ax.add_collection(lines)
        ax.scatter(x[0], y[0], s=190, marker="o", facecolor="white", edgecolor=INK, linewidth=2.4, zorder=5,
                   clip_on=False)
        ax.scatter(x[-1], y[-1], s=160, marker="s", facecolor=INK, edgecolor=INK, linewidth=1.0, zorder=5,
                   clip_on=False)
        set_round_axis(ax.xaxis, ax.set_xlim, x, x_nbins)
        set_round_axis(ax.yaxis, ax.set_ylim, y, 4)
        ax.set_title(TITLES[rat], fontsize=TITLE, pad=8, color=INK)
        for spine in ax.spines.values():
            spine.set_color(INK)
            spine.set_linewidth(SPINE_W)
        ax.tick_params(axis="both", which="major", labelsize=TICK, width=1.6, length=6.0, color=INK,
                       labelcolor=INK, pad=4)
        ax.grid(False)

    plots_left, plots_right = LEFT + ytick_space, LEFT + 3 * cell_w
    plots_bottom, plots_top = BOTTOM + XTICK_SPACE, PAGE_H - TOP - ROW_TITLE
    fig.text((plots_left + plots_right) / 2 / PAGE_W, (BOTTOM - 0.10) / PAGE_H, x_title, ha="center",
             va="top", fontsize=LABEL, color=INK)
    # y title just left of the left column's tick labels
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tick_left = min(ax.yaxis.get_tightbbox(renderer).x0 for ax in axes[::3])
    y_label = fig.text((tick_left - 0.10 * fig.dpi) / fig.bbox.width, (plots_bottom + plots_top) / 2 / PAGE_H,
                       y_title, rotation=90, ha="right", va="center", fontsize=LABEL, color=INK)
    # colour bar clear of the right column's last x tick label
    right_ticks = max(t.get_window_extent(renderer).x1 for ax in axes[2::3] for t in ax.get_xticklabels()
                      if t.get_text())
    cbar_left = max(PAGE_W - CBAR_SPACE + 0.30, (right_ticks + CBAR_GAP * fig.dpi) / fig.dpi)
    cbar_ax = fig.add_axes([cbar_left / PAGE_W, plots_bottom / PAGE_H, 0.24 / PAGE_W,
                            (plots_top - plots_bottom) / PAGE_H])
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar.set_ticks([0, 15, 30, 45, 60])
    cbar.ax.tick_params(labelsize=CBAR_TICK, width=1.6, length=6.0)
    cbar.outline.set_linewidth(SPINE_W)
    cbar.set_label("Time after glucose (min)", fontsize=CBAR_LABEL, labelpad=10)

    handles = [Line2D([], [], ls="none", marker="o", markersize=13, markerfacecolor="white", markeredgecolor=INK,
                      markeredgewidth=2.4, label="Start (0 min)"),
               Line2D([], [], ls="none", marker="s", markersize=12, markerfacecolor=INK, markeredgecolor=INK,
                      label="Last sample (60 min)")]
    legend = fig.legend(handles=handles, loc="lower left", ncol=1, frameon=False, fontsize=LEGEND,
                        bbox_to_anchor=(y_label.get_window_extent(renderer).x0 / fig.bbox.width, 0.0),
                        handletextpad=0.4, labelspacing=0.35, borderaxespad=0.1, borderpad=0.0, handlelength=1.0)
    extra = fig.get_default_bbox_extra_artists() + [y_label, legend]
    save(fig, stem, bbox_inches="tight", pad_inches=0.06, bbox_extra_artists=extra, facecolor="white")


def panel_b():
    """Fig. 6 b: insulin–glucagon trajectory of each LZ and ZDF rat after the glucose injection.

    Each panel follows one rat's calibrated insulin (x) and glucagon (y) over the 24 samples from t = 0 (open
    circle) to the last sample (black square); the line colour gives the time after the injection on the nominal
    clock (figure_style.nominal_time). Axis ranges are set per rat, from one labelled tick to another."""
    use_arial()
    trajectory_grid(pd.read_csv(PROFILES), "insulin", "glucagon", "[Insulin] (pM)", "[Glucagon] (pM)", "fig6_b")


# ---- e–f: features selected by the perfect combinations ----

TOP_N, N_FITS = 12, 168
EQUIVALENT = []
for _h in ("insulin", "c_peptide", "glucagon"):
    EQUIVALENT += [[f"{_h}_peak_magnitude_pM", f"{_h}_response_max_pM"],                    # identical
                   [f"{_h}_peak_time_min", f"{_h}_rise_time_min", f"{_h}_decay_time_min"],  # rise = peak; decay = T - peak
                   [f"{_h}_auc_total_pM_min", f"{_h}_time_weighted_mean_pM"],               # mean = AUC / T
                   [f"{_h}_trend_r_squared", f"{_h}_trend_p_value"]]                        # p = f(R², n = 24)
EQUIVALENT += [[f"{_h}_first_phase_dominance", f"{_h}_first_second_ratio"] for _h in ("insulin", "c_peptide")]
EQUIVALENT += [["baseline_glucagon_mean_pM", "glucagon_fasting_ref_pM"],
               ["glucagon_nadir_pM", "glucagon_response_min_pM"]]
FEATURE_LABELS = {
    "c_peptide_pulsatility_index": "C-peptide pulsatility index",
    "c_peptide_response_cv_pct": "C-peptide response CV",
    "c_peptide_first_phase_dominance": "C-peptide first-phase dominance / phase ratio",
    "insulin_trend_r_squared": "Insulin trend R² / significance",
    "c_peptide_peak_fold_change": "C-peptide fold change",
    "insulin_glucose_corr_r": "Insulin–glucose correlation",
    "insulin_peak_fold_change": "Insulin fold change",
    "disposition_relative_si": "Disposition index (relative)",
    "insulin_incremental_auc_pM_min": "Insulin incremental AUC",
    "c_peptide_glucose_corr_r": "C-peptide–glucose correlation",
    "glucose_at_0min_mM": "Glucose at 0 min",
    "baseline_insulin_mean_pM": "Insulin baseline mean",
}
COLORBAR_GAP, COLORBAR_WIDTH, RIGHT_MARGIN = 0.24, 0.22, 1.25
COUNT_CMAP = LinearSegmentedColormap.from_list("count_blue", ["#f7fbff", "#d8e9f5", "#a8cae2", "#5f98c3", "#1f4f7a"])
Z_CMAP = LinearSegmentedColormap.from_list("pastel_blue_red", ["#7FA8D6", "#F7F7F7", "#C96464"])
Z_LIMIT = 2.2


def merged_recurrence():
    """Fits of the perfect combinations that selected each feature (equivalent features merged), in total and by
    held-out rat, most first."""
    scores = pd.read_csv(CLASSIFICATION / "scores_by_assignment.csv")
    perfect = scores[scores.true_labels & (scores.macro_f1 == 1)]
    selected = pd.read_csv(CLASSIFICATION / "selected_features_true_labels.csv").merge(
        perfect[["selector", "classifier"]])
    fits = [set(row.split("|")) for row in selected.selected_features]
    if len(fits) != N_FITS:
        raise ValueError(f"{len(fits)} fits of perfect combinations, expected {N_FITS}")
    group_of = {member: tuple(group) for group in EQUIVALENT for member in group}
    rows = []
    for group in sorted({group_of.get(f, (f,)) for fit in fits for f in fit}):
        members = set(group)
        totals = {m: sum(m in fit for fit in fits) for m in group}
        hits = [members & fit for fit in fits]
        combos = {tuple(c) for c, h in zip(selected[["selector", "classifier"]].values, hits) if h}
        rows.append({"feature": max(group, key=lambda m: (totals[m], -group.index(m))),
                     "fits": sum(bool(h) for h in hits), "combinations": len(combos),
                     **{rat: sum(bool(h) for h, a in zip(hits, selected.animal_id) if a == rat) for rat in LZ + ZDF}})
    table = pd.DataFrame(rows).sort_values(["fits", "combinations", "feature"], ascending=[False, False, True])
    if table.fits.iloc[TOP_N - 1] == table.fits.iloc[TOP_N]:
        raise ValueError("tie at the top-12 cut")
    return table.head(TOP_N).reset_index(drop=True)


def zscores(features):
    X = pd.read_csv(FEATURES / "gtt_X.csv")
    X.index = pd.read_csv(FEATURES / "gtt_meta.csv").animal_id
    six = X.loc[LZ + ZDF, features].astype(float)
    return (six - six.mean()) / six.std(ddof=0)


def count_cells(ax, counts, denominator, white_from, x_labels):
    """Heatmap of counts out of denominator, each cell labelled 'count/denominator' (blank when 0)."""
    image = ax.imshow(counts, cmap=COUNT_CMAP, vmin=0, vmax=denominator, aspect="auto")
    style_table_axis(ax)
    ax.tick_params(axis="both", length=0, pad=8)
    ax.set_xticks(range(counts.shape[1]), labels=x_labels)
    for y in range(counts.shape[0]):
        for x in range(counts.shape[1]):
            if counts[y, x]:
                ax.text(x, y, f"{counts[y, x]}/{denominator}", ha="center", va="center", fontfamily="Arial",
                        fontsize=FONT_SIZE, color="white" if counts[y, x] >= white_from else "black")
    return image


def colourbar(fig, image, cax, ticks, label):
    cbar = fig.colorbar(image, cax=cax, ticks=ticks)
    cbar.set_label(label, fontsize=FONT_SIZE, fontfamily="Arial", color="black", labelpad=14)
    cbar.ax.tick_params(labelsize=FONT_SIZE, length=0, pad=7, labelcolor="black")
    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(BORDER)
    return cbar


def group_header(ax, texts):
    """LZ and ZDF above their three columns."""
    for start, end, name in ((0, 2, "LZ"), (3, 5, "ZDF")):
        ax.plot([start - 0.38, end + 0.38], [-0.78, -0.78], color="black", lw=2.0, clip_on=False, solid_capstyle="butt")
        texts.append(ax.text((start + end) / 2, -0.95, name, ha="center", va="bottom", fontsize=FONT_SIZE,
                             fontfamily="Arial", clip_on=False))


def panels_e_f():
    """Fig. 6 e–f: the features selected by the perfect combinations, and their values in each rat.

    e  number of the 168 held-out fits of the 28 perfect combinations (28 x 6 folds) that selected each feature,
       the 12 most frequently selected shown. Features that carry the same information (exact duplicates or exact
       transformations of each other, EQUIVALENT below) count as one: a fit counts once if it selected any of them.
       A merged row is labelled with both names and shows its more often selected member in f.
    f  the values of these features in each rat as z-scores across the six rats"""
    plt.rcParams.update({"font.family": "Arial", "font.sans-serif": ["Arial"], "pdf.fonttype": 42, "ps.fonttype": 42})
    top = merged_recurrence()
    labels = [FEATURE_LABELS[f] for f in top.feature]
    z = zscores(list(top.feature))
    for row in top.itertuples():
        print(f"  {FEATURE_LABELS[row.feature]}: {row.fits}/{N_FITS}")

    # frame, inches: labels | recurrence column | colour bar | gap | z heatmap | colour bar
    n = len(labels)
    left = label_margin(labels)
    col_w, z_w, plot_h, bottom = 1.8, 1.2 * 6, ROW_HEIGHT * n, 1.25
    cbar_left = left + col_w + COLORBAR_GAP
    z_left = cbar_left + COLORBAR_WIDTH + RIGHT_MARGIN + 0.45 + 0.55
    # summed in this order so that the axes positions, and so the heatmap edge pixels, match the paper figure
    width = z_left + (z_w + COLORBAR_GAP + COLORBAR_WIDTH + RIGHT_MARGIN)
    height = bottom + plot_h + 1.05
    fig = plt.figure(figsize=(width, height))
    axis = lambda x, w: fig.add_axes([x / width, bottom / height, w / width, plot_h / height])
    ax, cax = axis(left, col_w), axis(cbar_left, COLORBAR_WIDTH)

    # e
    image = count_cells(ax, top.fits.to_numpy(int)[:, None], N_FITS, int(N_FITS * 4 / 6), ["28 recipes"])
    ax.set_yticks(range(n), labels=labels)
    colourbar(fig, image, cax, [0, 42, 84, 126, 168], "Recipe-folds selected")

    # f
    zax, zcax = axis(z_left, z_w), axis(z_left + z_w + COLORBAR_GAP, COLORBAR_WIDTH)
    values = z.T.to_numpy()
    zimage = zax.imshow(values, cmap=Z_CMAP, norm=Normalize(-Z_LIMIT, Z_LIMIT), aspect="auto")
    style_table_axis(zax)
    zax.tick_params(axis="both", length=0, pad=8)
    zax.set_xticks(range(6), labels=["R1", "R2", "R3"] * 2)
    zax.set_yticks([])
    for y in range(n):
        for x in range(6):
            zax.text(x, y, f"{values[y, x]:.1f}".replace("-", "−"), ha="center", va="center", fontsize=FONT_SIZE,
                     fontfamily="Arial", color="black")
    zax.axvline(2.5, color="black", lw=BORDER, zorder=3)
    texts = []
    group_header(zax, texts)
    zax.set_ylim(n - 0.5, -0.5)
    cbar = fig.colorbar(zimage, cax=zcax, ticks=[-2, -1, 0, 1, 2])
    cbar.ax.set_yticklabels(["−2", "−1", "0", "1", "2"])
    cbar.set_label("Feature z-score", fontsize=FONT_SIZE, fontfamily="Arial", labelpad=14)
    cbar.ax.tick_params(labelsize=FONT_SIZE, length=0, pad=7)
    cbar.outline.set_linewidth(BORDER)

    extra = fig.get_default_bbox_extra_artists() + texts
    save(fig, "fig6_e_f", png_dpi=300, pdf_dpi=600, bbox_inches="tight", bbox_extra_artists=extra)


# ---- g: principal component analysis ----

GROUP_COLORS = {"LZ": "#607D8B", "ZDF": "#8A4F4A"}
# label, offset (points), ha, va: inside the frame and clear of the symbols
PCA_LABELS = {"Rat1": ("LZ R1", (-40, 12), "left", "bottom"), "Rat2": ("LZ R2", (0, 14), "center", "bottom"),
          "Rat3": ("LZ R3", (12, -6), "left", "top"), "Rat7": ("ZDF R1", (10, -12), "left", "top"),
          "Rat8": ("ZDF R2", (-6, 14), "center", "bottom"), "Rat9": ("ZDF R3", (-12, 12), "right", "bottom")}
WIDTH_PT, HEIGHT_PT, FRAME_PT = 430.0, 493.302490234375, 410.2200012207031


def principal_components():
    X = pd.read_csv(FEATURES / "gtt_X.csv")
    X.index = pd.read_csv(FEATURES / "gtt_meta.csv").animal_id
    values = X.loc[RATS].to_numpy(float)
    values = np.where(np.isnan(values), np.nanmedian(values, axis=0), values)
    z = (values - values.mean(axis=0)) / values.std(axis=0, ddof=0)
    pca = PCA(n_components=2, svd_solver="full").fit(z)
    return pca.transform(z), pca.explained_variance_ratio_


def clip_rectangle(rectangle, coefficients, intercept, direction):
    """The part of the plot rectangle on one side of the line coefficients . x + intercept = 0."""
    polygon = []
    for previous, current in zip(rectangle, np.roll(rectangle, -1, axis=0)):
        before = direction * (previous @ coefficients + intercept)
        after = direction * (current @ coefficients + intercept)
        if (before >= 0) != (after >= 0):
            polygon.append(previous + before / (before - after) * (current - previous))
        if after >= 0:
            polygon.append(current)
    return np.asarray(polygon)


def panel_g():
    """Fig. 6 g: principal component analysis of the 181 features of the six LZ and ZDF rats.

    The PCA uses no labels: missing values are set to the feature median of the six rats and every feature is
    standardized (mean 0, SD 1 across the six rats). The dashed line and the shaded regions show a shrinkage
    linear discriminant analysis fitted to the two principal components of these six rats with their labels, an
    in-sample illustration, not the boundary of the cross-validated classifiers."""
    coords, variance = principal_components()
    groups = ["LZ"] * 3 + ["ZDF"] * 3
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5]).fit(coords, [0, 0, 0, 1, 1, 1])
    coefficients, intercept = lda.coef_[0], float(lda.intercept_[0])
    print(f"  PC1 {variance[0]:.1%}, PC2 {variance[1]:.1%}; LDA boundary {coefficients[0]:.4f} PC1 "
          f"{coefficients[1]:+.4f} PC2 {intercept:+.2g} = 0")

    plt.rcParams.update({"font.family": "Arial", "font.sans-serif": ["Arial"], "font.size": 28, "axes.labelsize": 28,
                         "xtick.labelsize": 24, "ytick.labelsize": 24, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.grid": False})
    left_pt, right_pt, top_pt = 94.0, 10.0, 5.0
    bottom_pt = HEIGHT_PT - top_pt - FRAME_PT
    fig = plt.figure(figsize=(WIDTH_PT / 72, HEIGHT_PT / 72))
    ax = fig.add_axes([left_pt / WIDTH_PT, bottom_pt / HEIGHT_PT, (WIDTH_PT - left_pt - right_pt) / WIDTH_PT,
                       FRAME_PT / HEIGHT_PT])
    margin = np.maximum(0.24 * np.ptp(coords, axis=0), 0.1)
    lo, hi = coords.min(axis=0) - margin, coords.max(axis=0) + margin
    rectangle = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    line_points = []
    for group, direction in (("LZ", -1), ("ZDF", 1)):
        polygon = clip_rectangle(rectangle, coefficients, intercept, direction)
        ax.fill(polygon[:, 0], polygon[:, 1], facecolor=0.15 * np.asarray(to_rgb(GROUP_COLORS[group])) + 0.85,
                edgecolor="none", zorder=0)
        line_points.extend(p for p in polygon if abs(p @ coefficients + intercept) < 1e-10)
    line = np.unique(np.round(line_points, 12), axis=0)
    ax.plot(line[:, 0], line[:, 1], color="#333333", linewidth=2.4, linestyle=(0, (5, 3)), zorder=2)
    for rat, group, (x, y) in zip(RATS, groups, coords):
        ax.scatter([x], [y], s=230, marker=MARKERS[rat], facecolor=GROUP_COLORS[group], edgecolor="white",
                   linewidth=1.4, zorder=4)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_xticks([-10, 0, 10, 20])
    ax.set_yticks([-10, -5, 0, 5, 10])
    ax.set_xlabel(f"PC1 ({variance[0]:.1%})", labelpad=10)
    ax.set_ylabel(f"PC2 ({variance[1]:.1%})", labelpad=8)
    ax.tick_params(direction="out", length=8, width=BORDER, pad=6, colors="black", top=False, right=False)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(BORDER)
    for rat, point in zip(RATS, coords):
        text, offset, ha, va = PCA_LABELS[rat]
        ax.annotate(text, point, xytext=offset, textcoords="offset points", ha=ha, va=va, fontsize=22, color="black",
                    zorder=5)
    save(fig, "fig6_g", bbox_inches=None, facecolor="white")


# ---- h: hormone family and feature type of the selected features ----

MAP = ROOT / "data" / "features" / "feature_families_and_types.csv"
FED_SELECTED = ROOT.parent / "fed_state" / "data" / "classification" / "selected_features.csv"
FAMILIES = ["Insulin", "C-peptide", "Glucagon", "Glucose", "Hormone + glucose", "Cross-hormone"]
SHARED_TYPES = ["Glucose context", "AUC / exposure", "Cell-function / clinical indices", "Basic device metrics",
                "Oscillation patterns", "Inter-biomarker correlations", "Peak kinetics", "Peak dynamics",
                "Rate / trend dynamics", "Glucose-adjusted response"]          # Fig. 4c order
GTT_ONLY = ["Pre-injection baseline", "Biphasic response", "Glucagon suppression"]
STATES = ["GTT", "Fed"]
STATE_COLORS = {"GTT": "#2F6FA7", "Fed": "#D08A2E"}
COLUMN_WIDTH, HEADER_GREY = 1.65, "#E6E6E6"


def selections():
    """Selected features of every fitted model, with weight 1/K: GTT and fed state."""
    scores = pd.read_csv(CLASSIFICATION / "scores_by_assignment.csv")
    perfect = scores[scores.true_labels & (scores.macro_f1 == 1)][["selector", "classifier"]]
    gtt = pd.read_csv(CLASSIFICATION / "selected_features_true_labels.csv").merge(perfect)
    gtt = gtt.assign(model=gtt.selector + "+" + gtt.classifier + "|" + gtt.animal_id,
                     feature=gtt.selected_features.str.split("|")).explode("feature")
    fed = pd.read_csv(FED_SELECTED)
    fed = fed[(fed.selector == "mrmr") & (fed.classifier == "LogReg_L2")].assign(model=lambda d: d.fold.astype(str))
    slots = {}
    for state, table in (("GTT", gtt), ("Fed", fed)):
        slots[state] = table.assign(weight=1 / table.groupby("model").feature.transform("size"))[["model", "feature",
                                                                                                    "weight"]]
    return slots


def shares(candidates, slots, key):
    """Percentage of the selections (models weighted equally) in each family or type."""
    group = slots.feature.map(candidates.set_index("feature")[key])
    if group.isna().any():
        raise ValueError("a selected feature is not a candidate")
    return 100 * slots.assign(group=group).groupby("group").weight.sum() / slots.model.nunique()


def bubble_table(ax, table, available, title, texts, v_ref):
    """One table: a circle (area proportional to the share) and the share in each cell, '–' where the state has
    no candidates of that kind."""
    n = len(table)
    max_d = 0.50 * 72          # points: diameter of the largest share
    for k in range(1, n):
        ax.axhline(k - 0.5, color=HEADER_GREY, lw=1.2, zorder=0)
    ax.axvline(0.5, color="black", lw=BORDER, zorder=3)
    for y, row in enumerate(table.index):
        for x, state in enumerate(STATES):
            if not available[state](row):
                ax.text(x, y, "–", ha="center", va="center", fontsize=FONT_SIZE, fontfamily="Arial")
                continue
            v = table.at[row, state]
            if v <= 0:
                continue
            d = max_d * np.sqrt(v / v_ref)
            ax.scatter([x - 0.22], [y], s=d ** 2, zorder=4, linewidth=2.0, edgecolor=STATE_COLORS[state],
                       facecolor=STATE_COLORS[state])
            ax.text(x - 0.02, y, "<1%" if v < 0.5 else f"{v:.0f}%", ha="left", va="center", fontsize=FONT_SIZE,
                    fontfamily="Arial", zorder=5)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks(range(n), labels=table.index)
    ax.set_xticks([])
    style_table_axis(ax)
    ax.tick_params(axis="both", length=0, pad=8)
    texts.append(ax.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(-14, 6), textcoords="offset points",
                             ha="right", va="bottom", fontsize=FONT_SIZE, fontfamily="Arial", fontstyle="italic"))


def column_headers(ax, n_rows, texts):
    """'GTT' | 'Fed state' over the two columns, 'Selected features' above both."""
    for x, label in enumerate(["GTT", "Fed state"]):
        texts.append(ax.text(x, -0.62, label, ha="center", va="bottom", fontsize=FONT_SIZE, fontfamily="Arial",
                             clip_on=False))
    ax.plot([-0.42, 1.42], [-1.42, -1.42], color="black", lw=2.0, clip_on=False, solid_capstyle="butt")
    texts.append(ax.text(0.5, -1.55, "Selected features", ha="center", va="bottom", fontsize=FONT_SIZE,
                         fontfamily="Arial", clip_on=False))
    ax.set_ylim(n_rows - 0.5, -0.5)


def header_bands(fig, bands, pad=0.06):
    """Grey band behind each table's title and column labels, from the title to the table's right edge."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_fig = fig.transFigure.inverted()
    width, height = fig.get_size_inches()
    for title, labels, ax in bands:
        boxes = [a.get_window_extent(renderer).transformed(to_fig) for a in [title, *labels]]
        x0 = boxes[0].x0 - pad / width
        x1 = ax.get_position().x1 if labels else boxes[0].x1 + pad / width
        y0 = min(b.y0 for b in boxes) - pad / height
        if y0 <= ax.get_position().y1:                    # keep clear of the table frame
            y0 = ax.get_position().y1 + 0.02 / height
        y1 = max(b.y1 for b in boxes) + pad / height
        fig.add_artist(Rectangle((x0, y0), x1 - x0, y1 - y0, transform=fig.transFigure, facecolor=HEADER_GREY,
                                 edgecolor="none", zorder=-1))


def panel_h():
    """Fig. 6 h: hormone family and feature type of the features selected in the GTT and in the fed state.

    GTT: the 28 perfect combinations x 6 folds (168 fitted models). Fed state: mRMR with logistic regression x 9
    folds (main-text Fig. 4h). Each fitted model is weighted equally: each of its K selected features counts 1/K.
    Circle area is proportional to the share. Every candidate feature of both states (GTT 181, fed 160) is
    assigned a hormone family and a feature type in data/features/feature_families_and_types.csv; the feature
    types are those of the fed-state feature inventory (main-text Fig. 4c), with the GTT feature categories A–O
    mapped onto them, and three types that only the GTT has ('–' for the fed state)."""
    candidates = pd.read_csv(MAP)
    slots = selections()
    family = pd.DataFrame({s: shares(candidates[candidates.state == s], slots[s], "family") for s in STATES})
    kind = pd.DataFrame({s: shares(candidates[candidates.state == s], slots[s], "type") for s in STATES})
    family = family.reindex(FAMILIES).fillna(0)
    kind = kind.reindex(SHARED_TYPES + GTT_ONLY).fillna(0)
    fed_groups = set(candidates[candidates.state == "Fed"][["family", "type"]].to_numpy().ravel())
    available = {"GTT": lambda row: True, "Fed": lambda row: row in fed_groups}
    print(pd.concat([family, kind]).round(1).to_string())

    # frame, inches: family table above the GTT-only types | the shared types
    plot_w = COLUMN_WIDTH * 2
    left_a = label_margin(FAMILIES + GTT_ONLY)
    left_b = label_margin(SHARED_TYPES)
    h, bottom = ROW_HEIGHT * len(SHARED_TYPES), 0.3
    height = bottom + h + 1.25
    x_b = left_a + plot_w + 0.25 + left_b
    width = x_b + plot_w + 0.4
    fig = plt.figure(figsize=(width, height))
    ax_family = fig.add_axes([left_a / width, (bottom + h - ROW_HEIGHT * len(FAMILIES)) / height, plot_w / width,
                              ROW_HEIGHT * len(FAMILIES) / height])
    ax_only = fig.add_axes([left_a / width, bottom / height, plot_w / width, ROW_HEIGHT * len(GTT_ONLY) / height])
    ax_shared = fig.add_axes([x_b / width, bottom / height, plot_w / width, h / height])
    v_ref = max(family.to_numpy().max(), kind.to_numpy().max())     # one size scale for all tables
    texts, bands = [], []
    for ax, table, title in ((ax_family, family, "Hormone family"),
                             (ax_only, kind.loc[GTT_ONLY], "GTT-only feature types"),
                             (ax_shared, kind.loc[SHARED_TYPES], "Feature type")):
        bubble_table(ax, table, available, title, texts, v_ref)
        title_artist, start = texts[-1], len(texts)
        if ax is not ax_only:
            column_headers(ax, len(table), texts)
        bands.append((title_artist, texts[start:start + 2], ax))
    header_bands(fig, bands)
    save(fig, "fig6_h", png_dpi=300, pdf_dpi=600, bbox_inches="tight",
         bbox_extra_artists=fig.get_default_bbox_extra_artists() + texts)


def main():
    for draw in (panel_a, panel_b, panel_c, panel_d, panels_e_f, panel_g, panel_h):
        with plt.rc_context():          # every figure starts from the default settings
            draw()


if __name__ == "__main__":
    main()
