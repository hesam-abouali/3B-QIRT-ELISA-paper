#!/usr/bin/env python3
"""Fig. 5 b–e: calibrated GTT profiles and their agreement with ELISA.

b–c  glucose and the calibrated insulin, C-peptide and glucagon profiles of the LZ (b) and ZDF (c) rats
     (outputs/figures/fig5_b_c.pdf and .png)
d–e  held-out device concentration versus ELISA, and Bland–Altman plots (fig5_d_e.pdf and .png)

Input   data/concentrations/concentration_timeseries.csv, held_out_elisa.csv, held_out_elisa_summary.csv,
        data/raw/gtt_post_injection.csv

    python figures/fig5.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, FixedLocator, FormatStrFormatter, MaxNLocator, MultipleLocator

from figure_style import LZ, NOMINAL_ANCHORS_MIN, ROOT, ZDF, nominal_time, save, use_arial

HORMONES = ("insulin", "c_peptide", "glucagon")
TICK_LABEL_SIZE = 30.0


# ---- b–c: glucose and calibrated hormone profiles ----

PROFILES = ROOT / "data" / "concentrations" / "concentration_timeseries.csv"
RAW = ROOT / "data" / "raw" / "gtt_post_injection.csv"
LABELS = {"insulin": "Insulin", "c_peptide": "C-peptide", "glucagon": "Glucagon"}
COLORS = {"insulin": "#800002", "c_peptide": "#408002", "glucagon": "#FD8008"}
GLUCOSE_COLOR = "#2D5F8A"
AXIS_LABEL_SIZE = 33.0
LEGEND = dict(fontsize=14.58, framealpha=0.50, markerscale=0.81, handlelength=1.62, borderpad=0.4725,
              labelspacing=0.3375, handletextpad=0.6075, borderaxespad=0.6075)
Y_LABEL_PADS = {"glucose": 11.88, "insulin": 11.88, "c_peptide": 20.0, "glucagon": 11.88}
# y range and major tick step of each panel (None: from the data, see insulin_limits)
Y_RANGES = {"LZ": {"insulin": None, "c_peptide": (0.0, 3000.0), "glucagon": (0.0, 16.0)},
            "ZDF": {"insulin": None, "c_peptide": (0.0, 8000.0), "glucagon": (0.0, 8.0)}}
Y_STEPS = {"LZ": {"insulin": 200.0, "c_peptide": 1000.0, "glucagon": 4.0},
           "ZDF": {"insulin": 400.0, "c_peptide": 2000.0, "glucagon": 2.0}}


def profile_axis(ax, show_x_labels):
    ax.set_xlim(-12.5, 61.5)
    ax.xaxis.set_major_locator(FixedLocator([-10.0, 0.0, 15.0, 30.0, 45.0, 60.0]))
    ax.xaxis.set_minor_locator(MultipleLocator(5.0))
    if show_x_labels:
        ax.set_xlabel("Time (min)", fontsize=AXIS_LABEL_SIZE)
    else:
        ax.tick_params(axis="x", which="both", labelbottom=False)
    ax.grid(False)
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE, width=1.6, length=5.5)
    ax.tick_params(axis="x", which="minor", width=1.2, length=4.0, labelsize=TICK_LABEL_SIZE)
    for spine in ax.spines.values():
        spine.set_linewidth(2.4)
    ax.axvline(0, color="#888888", linestyle="--", linewidth=2.2, zorder=1)


def insulin_limits(low, high, group):
    """Axis range of an insulin panel: the data range rounded outward to 250 pM (from 0 for LZ)."""
    lower = 0.0 if group == "LZ" else min(0.0, float(np.floor(low * 1.05 / 250.0) * 250.0))
    upper = float(np.ceil(high * 1.08 / 250.0) * 250.0)
    return lower, (1600.0 if group == "ZDF" else upper)


def glucose_panel(ax, raw, rats, show_x_labels):
    anchors = raw[raw.animal_id.isin(rats)].dropna(subset=["glucose_mM"])
    anchors = anchors.assign(time=nominal_time(anchors.timepoint))
    matrix = anchors.pivot(index="animal_id", columns="time", values="glucose_mM").loc[rats, NOMINAL_ANCHORS_MIN]
    mean, sd = np.nanmean(matrix, axis=0), np.nanstd(matrix, axis=0, ddof=1)
    t = NOMINAL_ANCHORS_MIN
    ax.plot(t, mean, color=GLUCOSE_COLOR, linewidth=3.6, marker="o", markersize=8, markerfacecolor="white",
            markeredgecolor=GLUCOSE_COLOR, markeredgewidth=1.6, zorder=4, label="Glucometer mean")
    ax.fill_between(t, mean - sd, mean + sd, color=GLUCOSE_COLOR, alpha=0.18, linewidth=0, zorder=2,
                    label="Glucometer mean ± SD")
    ax.errorbar(t, mean, yerr=sd, fmt="none", color="#111111", elinewidth=2.1, capthick=2.1, capsize=4, zorder=5)
    ax.set_ylim(0.0, 40.0)
    ax.yaxis.set_major_locator(MultipleLocator(10.0))
    ax.yaxis.set_minor_locator(MultipleLocator(5.0))
    ax.set_ylabel("[Glucose] (mM)", fontsize=AXIS_LABEL_SIZE, labelpad=Y_LABEL_PADS["glucose"])
    ax.legend(loc="upper right", **LEGEND)
    profile_axis(ax, show_x_labels)


def hormone_panel(ax, profiles, rats, hormone, group, show_x_labels):
    data = profiles[profiles.animal_id.isin(rats)]
    grid = np.sort(data.timepoint.unique())
    matrix = np.array([data[data.animal_id == rat].set_index("timepoint")[f"{hormone}_predicted_pM"]
                       .reindex(grid).to_numpy(float) for rat in rats])
    if not np.isfinite(matrix).all():
        raise ValueError(f"{group} {hormone}: the profiles do not share the same sampling times")
    t, mean, sd = nominal_time(grid), matrix.mean(axis=0), matrix.std(axis=0, ddof=1)
    elisa = data.dropna(subset=[f"{hormone}_ELISA_pM"])
    elisa = elisa.assign(time=nominal_time(elisa.timepoint)).groupby("time")[f"{hormone}_ELISA_pM"]
    elisa_mean, elisa_sd = elisa.mean(), elisa.std(ddof=1).fillna(0.0)

    color = COLORS[hormone]
    ax.plot(t, mean, color=color, linewidth=3.6, zorder=4, label="QIRT-ELISA mean")
    ax.fill_between(t, mean - sd, mean + sd, color=color, alpha=0.18, linewidth=0, zorder=2, label="QIRT-ELISA ± SD")
    ax.errorbar(elisa_mean.index, elisa_mean, yerr=elisa_sd, fmt="o", color="#111111", markerfacecolor="white",
                markersize=8, markeredgewidth=1.6, elinewidth=2.1, capthick=2.1, capsize=4, zorder=6,
                label="ELISA mean ± SD")
    limits = Y_RANGES[group][hormone]
    if limits is None:
        low = min(np.min(mean - sd), np.min(elisa_mean - elisa_sd))
        high = max(np.max(mean + sd), np.max(elisa_mean + elisa_sd))
        limits = insulin_limits(low, high, group)
    ax.set_ylim(*limits)
    step = Y_STEPS[group][hormone]
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_minor_locator(MultipleLocator(step / 2.0))
    ax.set_ylabel(f"[{LABELS[hormone]}] (pM)", fontsize=AXIS_LABEL_SIZE, labelpad=Y_LABEL_PADS[hormone])
    if hormone == "glucagon":   # the two glucagon axis titles aligned across the rows
        ax.yaxis.set_label_coords(-0.22, 0.5)
    if group == "LZ":
        ax.legend(loc="upper right", **LEGEND)
    profile_axis(ax, show_x_labels)


def panels_b_c():
    """Fig. 5 b–c: glucose and the calibrated insulin, C-peptide and glucagon profiles of the LZ (b) and ZDF (c) rats.

    Glucose: glucometer mean +/- SD of the three rats at the blood-sampling times. Hormones: mean +/- SD of the
    three calibrated profiles (line and band) and ELISA mean +/- SD at the blood-sampling times (open circles).
    Times are on the nominal clock (figure_style.nominal_time)."""
    use_arial()
    profiles = pd.read_csv(PROFILES)
    profiles["timepoint"] = profiles.timepoint.round(3)
    raw = pd.read_csv(RAW)
    fig, axes = plt.subplots(2, 4, figsize=(26.2, 10.4))
    for row, (group, rats) in enumerate((("LZ", LZ), ("ZDF", ZDF))):
        glucose_panel(axes[row, 0], raw, rats, show_x_labels=row == 1)
        for ax, hormone in zip(axes[row, 1:], HORMONES):
            hormone_panel(ax, profiles, rats, hormone, group, show_x_labels=row == 1)
        if row == 1:
            axes[row, 0].get_legend().remove()
        for ax in axes[row]:
            ax.text(0.03, 0.95, group, transform=ax.transAxes, ha="left", va="top", fontsize=TICK_LABEL_SIZE,
                    color="#111111")
    fig.subplots_adjust(left=0.073, right=0.995, bottom=0.105, top=0.985, wspace=0.45, hspace=0.115)
    save(fig, "fig5_b_c", bbox_inches="tight", pad_inches=0.05, facecolor="white")


# ---- d–e: held-out device concentration versus ELISA ----

HELD_OUT = ROOT / "data" / "concentrations" / "held_out_elisa.csv"
SUMMARY = ROOT / "data" / "concentrations" / "held_out_elisa_summary.csv"
AXIS_NAMES = {"insulin": "insulin", "c_peptide": "C-peptide", "glucagon": "glucagon"}
IDENTITY_LIMITS = {"insulin": 1800.0, "c_peptide": 10000.0, "glucagon": 10.0}   # one C-peptide prediction is 9197 pM
IDENTITY_TICKS = {"insulin": [0.0, 600.0, 1200.0, 1800.0],
                  "c_peptide": [0.0, 2000.0, 4000.0, 6000.0, 8000.0],
                  "glucagon": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]}
RAT_COLORS = {**{rat: "#787878" for rat in LZ}, **{rat: "#8A4F4A" for rat in ZDF}}
MARKERS = dict(zip(LZ + ZDF, ["o", "s", "^", "D", "v", "X"]))
AGREEMENT_LABEL_SIZE = 29.7
ANNOTATION_SIZE = 24.75
Y_LABEL_PAD = 11.88


def agreement_axis(ax):
    ax.grid(False)
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE, width=1.6, length=6.0)
    ax.tick_params(axis="both", which="minor", width=1.2, length=4.0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    for spine in ax.spines.values():
        spine.set_linewidth(2.4)


def scatter_by_rat(ax, frame, x, y):
    for rat in LZ + ZDF:
        points = frame[frame.animal_id == rat]
        ax.scatter(points[x], points[y], s=216, marker=MARKERS[rat], color=RAT_COLORS[rat], edgecolor="white",
                   linewidth=1.2, alpha=0.92, zorder=4)


def annotate(ax, text, y, va, alpha, size=ANNOTATION_SIZE):
    ax.text(0.04, y, text, transform=ax.transAxes, ha="left", va=va, fontsize=size, color="#111111",
            linespacing=1.18, zorder=8, bbox={"facecolor": "white", "edgecolor": "none", "alpha": alpha, "pad": 1.5})


def identity_panel(ax, frame, hormone, stats):
    ticks, upper = IDENTITY_TICKS[hormone], IDENTITY_LIMITS[hormone]
    scatter_by_rat(ax, frame, "elisa_pM", "predicted_pM")
    ax.plot([0.0, upper], [0.0, upper], color="#888888", linestyle="--", linewidth=3.3, zorder=1)
    ax.set_xlim(0.0, upper)
    ax.set_ylim(0.0, upper)
    ax.set_xlabel(f"ELISA [{AXIS_NAMES[hormone]}] (pM)", fontsize=AGREEMENT_LABEL_SIZE)
    ax.set_ylabel(f"Held-out device\n[{AXIS_NAMES[hormone]}] (pM)", fontsize=AGREEMENT_LABEL_SIZE, labelpad=Y_LABEL_PAD)
    annotate(ax, f"MAE = {stats.mae_pM:.1f} pM\nCCC = {stats.ccc:.2f}", 0.96, "top", 0.5)
    agreement_axis(ax)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_minor_locator(AutoMinorLocator(2))
        axis.set_major_formatter(FormatStrFormatter("%d"))


def bland_altman_panel(ax, frame, hormone, stats):
    frame = frame.assign(pair_mean=(frame.elisa_pM + frame.predicted_pM) / 2.0,
                         difference=frame.predicted_pM - frame.elisa_pM)
    scatter_by_rat(ax, frame, "pair_mean", "difference")
    ax.axhline(0.0, color="#aaaaaa", linestyle=":", linewidth=2.0, zorder=1)
    ax.axhline(stats.bias_pM, color="#111111", linewidth=2.2, zorder=2)
    for limit in (stats.loa_low_pM, stats.loa_high_pM):
        ax.axhline(limit, color="#888888", linestyle="--", linewidth=2.0, zorder=1)
    y = np.concatenate([frame.difference.to_numpy(float), [stats.loa_low_pM, stats.loa_high_pM, 0.0]])
    span = max(float(np.ptp(y)), 1e-9)
    ax.set_ylim(float(y.min() - 0.08 * span), float(y.max() + 0.08 * span))
    ax.set_xlim(0.0, 1.06 * float(frame.pair_mean.max()))
    ax.set_xlabel(f"Mean [{AXIS_NAMES[hormone]}] (pM)", fontsize=AGREEMENT_LABEL_SIZE)
    ax.set_ylabel(f"Device - ELISA\n[{AXIS_NAMES[hormone]}] (pM)", fontsize=AGREEMENT_LABEL_SIZE, labelpad=Y_LABEL_PAD)
    text = f"Bias = {stats.bias_pM:.1f} pM\n95% limits = [{stats.loa_low_pM:.1f}, {stats.loa_high_pM:.1f}] pM"
    if hormone == "glucagon":   # bottom left, clear of the points
        annotate(ax, text, 0.04, "bottom", 0.5)
    else:
        annotate(ax, text, 0.96, "top", 1.0, size=22.75 if hormone == "c_peptide" else ANNOTATION_SIZE)
    agreement_axis(ax)


def panels_d_e():
    """Fig. 5 d–e: held-out device concentration versus ELISA (d) and Bland–Altman plots (e).

    Each ELISA value of the LZ and ZDF rats was left out of its rat's calibration and predicted from the line
    refitted to the rat's other ELISA values (src/held_out_elisa.py). d: prediction against ELISA with the
    line of identity, MAE and Lin's CCC. e: difference (device - ELISA) against the mean of the two, with the mean
    bias (solid), the 95% limits of agreement (dashed) and zero (dotted)."""
    use_arial()
    held_out = pd.read_csv(HELD_OUT)
    summary = pd.read_csv(SUMMARY).set_index("hormone")
    fig, axes = plt.subplots(2, 3, figsize=(22.0, 12.8))
    for column, hormone in enumerate(HORMONES):
        frame = held_out[held_out.hormone == hormone]
        identity_panel(axes[0, column], frame, hormone, summary.loc[hormone])
        bland_altman_panel(axes[1, column], frame, hormone, summary.loc[hormone])
    fig.subplots_adjust(left=0.060, right=0.995, bottom=0.105, top=0.985, wspace=0.54, hspace=0.364)
    save(fig, "fig5_d_e", bbox_inches="tight", pad_inches=0.05, facecolor="white")


def main():
    for draw in (panels_b_c, panels_d_e):
        with plt.rc_context():          # every figure starts from the default settings
            draw()


if __name__ == "__main__":
    main()
