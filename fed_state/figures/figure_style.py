"""Shared frame sizes, typography and colour helper of the classification figures."""

import colorsys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

FONT_SIZE = 28
BORDER_LINEWIDTH = 2.8
INTERNAL_LINEWIDTH = 1.4
PLOT_WIDTH = 3.6
PLOT_HEIGHT = 4.0
LEFT_MARGIN = 1.4
RIGHT_MARGIN = 0.25
BOTTOM_MARGIN = 1.2
TOP_MARGIN = 0.3
PANEL_WIDTH = LEFT_MARGIN + PLOT_WIDTH + RIGHT_MARGIN
PANEL_HEIGHT = BOTTOM_MARGIN + PLOT_HEIGHT + TOP_MARGIN


def darker(color, factor=0.62):
    """The colour with its HLS lightness scaled by factor (used for marker outlines)."""
    hue, lightness, saturation = colorsys.rgb_to_hls(*to_rgb(color))
    return colorsys.hls_to_rgb(hue, lightness * factor, saturation)


def create_panels(count=1):
    """A figure with count side-by-side panels of identical frame size and margins."""
    plt.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "font.size": FONT_SIZE,
        "font.weight": "normal",
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelweight": "normal",
        "axes.titleweight": "normal",
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 150,
        "savefig.dpi": 600,
    })
    width = PANEL_WIDTH * count
    fig = plt.figure(figsize=(width, PANEL_HEIGHT))
    axes = [
        fig.add_axes([
            (index * PANEL_WIDTH + LEFT_MARGIN) / width,
            BOTTOM_MARGIN / PANEL_HEIGHT,
            PLOT_WIDTH / width,
            PLOT_HEIGHT / PANEL_HEIGHT,
        ])
        for index in range(count)
    ]
    return fig, axes


def style_axis(ax):
    """Black borders, Arial labels of FONT_SIZE and no grid or minor ticks."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(BORDER_LINEWIDTH)
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE,
                   length=4, width=BORDER_LINEWIDTH, color="black", labelcolor="black")
    for text in [ax.xaxis.label, ax.yaxis.label, ax.title,
                 *ax.get_xticklabels(), *ax.get_yticklabels()]:
        text.set_fontfamily("Arial")
        text.set_fontsize(FONT_SIZE)
        text.set_fontweight("normal")
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10
    ax.minorticks_off()
    ax.grid(False)
