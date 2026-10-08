"""Shared matplotlib style for the report figures (dh2/svi.py, dh2/vixcompare.py): the validated categorical palette of the dataviz
method (blue, orange, aqua in that fixed order; the first three validate for all-pairs use and for colour-vision deficiency, and the
aqua is under 3:1 contrast on the light surface, so every chart carries a legend and direct end labels as the method requires),
a one-hue blue sequential ramp for magnitudes, large type for slides, 300 dpi.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib.colors import LinearSegmentedColormap       # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d9d8d3"
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]           # blue, orange, aqua, yellow (fixed order)
# one colour per entity across every figure of the report (blue = the reference series: India VIX / SVI; orange = jump model; aqua = diffusion
# Double Heston; violet = the surface ATM 30-day IV). {orange, aqua, violet} and {blue, violet} pass the all-pairs colour-vision check.
ENT = {"vix": PAL[0], "svi": PAL[0], "jump": PAL[1], "diff": PAL[2], "atm": "#4a3aa7"}
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def setup() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "font.size": 16, "axes.titlesize": 19, "axes.labelsize": 17, "legend.fontsize": 15, "xtick.labelsize": 15, "ytick.labelsize": 15,
        "lines.linewidth": 2.2, "axes.titleweight": "regular", "axes.titlelocation": "left", "legend.frameon": False,
        "savefig.dpi": 300, "figure.dpi": 100,
    })


def seq_cmap() -> LinearSegmentedColormap:
    """Light to dark single-hue (blue) ramp for magnitudes."""
    return LinearSegmentedColormap.from_list("blue_seq", BLUES)


def cell_text_color(frac: float) -> str:
    """Ink on light cells, white on dark ones (frac = position on the ramp, 0 light .. 1 dark)."""
    return "white" if frac > 0.55 else INK


def end_label(ax, x, y, text: str, color_idx: int, dx_frac: float = 0.01) -> None:
    """A direct label at the right end of a line, in secondary ink, with a short colour bar beside it carrying the identity."""
    ax.annotate(text, xy=(x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=14, color=INK2)
