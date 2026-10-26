"""Shared typography, nominal time axis and saving of the GTT figures."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "figures"
LZ = ["Rat1", "Rat2", "Rat3"]
ZDF = ["Rat7", "Rat8", "Rat9"]
RECORDED_ANCHORS_MIN = np.array([0.0, 16.0, 29.333, 45.333, 61.333])
NOMINAL_ANCHORS_MIN = np.array([0.0, 15.0, 30.0, 45.0, 60.0])
# the blood-sampling times to five decimals, as some of the paper's figures used them (the nominal-clock positions
# of the device samples differ by less than 0.001 min)
RECORDED_ANCHORS_5DP = np.array([0.0, 16.0, 29.33333, 45.33333, 61.33333])
# heatmap and table panels (Fig. 6 e, f, h), in the style of the fed-state Fig. 4h
FONT_SIZE = 28
BORDER = 2.8
ROW_HEIGHT = (8.509 - 1.672) / 12      # inches


def use_arial():
    plt.rcParams.update({
        "font.family": "Arial",
        "font.weight": "normal",
        "axes.labelweight": "normal",
        "axes.titleweight": "normal",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": True,
    })


def nominal_time(values, recorded=RECORDED_ANCHORS_MIN):
    """Times on the nominal clock: the blood-sampling times (0, 16, 29.3, 45.3 and 61.3 min after the
    injection) are plotted at 0, 15, 30, 45 and 60 min and the device samples between them by linear
    interpolation; pre-injection times are unchanged."""
    t = np.asarray(values, dtype=float)
    mapped = np.where(t < 0.0, t, np.interp(t, recorded, NOMINAL_ANCHORS_MIN))
    for anchor, nominal in zip(recorded, NOMINAL_ANCHORS_MIN):
        mapped = np.where(np.isclose(t, anchor, atol=5e-4), nominal, mapped)
    return np.round(mapped, 6)


def style_table_axis(ax):
    """Heatmap and table panels: black borders, Arial labels of FONT_SIZE, no grid or minor ticks."""
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(BORDER)
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE, length=4, width=BORDER, color="black",
                   labelcolor="black")
    for text in [ax.xaxis.label, ax.yaxis.label, ax.title, *ax.get_xticklabels(), *ax.get_yticklabels()]:
        text.set_fontfamily("Arial")
        text.set_fontsize(FONT_SIZE)
        text.set_fontweight("normal")
    ax.xaxis.labelpad = ax.yaxis.labelpad = 10
    ax.minorticks_off()
    ax.grid(False)


def label_margin(labels):
    """Width (inches) of the longest row label of a heatmap or table panel, plus a gap."""
    font = FontProperties(family="Arial", size=FONT_SIZE)
    return max(TextPath((0, 0), text, prop=font).get_extents().width / 72 for text in labels) + 0.28


def save(fig, stem, png_dpi=600, pdf_dpi="figure", **options):
    """Write outputs/figures/<stem>.pdf and .png (pdf_dpi sets the resolution of rasterized parts, such as
    colour bars, in the PDF)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path, dpi in ((OUT_DIR / f"{stem}.pdf", pdf_dpi), (OUT_DIR / f"{stem}.png", png_dpi)):
        fig.savefig(path, dpi=dpi, **options)
        print(f"  Saved: {path}")
    plt.close(fig)
