"""Shared publication-style plotting helpers for the textbook.

The palette follows the restrained, color-safe visual hierarchy of the Payne
Zero paper.  Quantitative plots remain precise and typeset; the hand-sketched
website aesthetic is reserved for conceptual schematics.
"""

from __future__ import annotations

from types import MappingProxyType

import matplotlib as mpl
import matplotlib.pyplot as plt


PAPER_COLORS = MappingProxyType(
    {
        "black": "#171717",
        "slate": "#526777",
        "blue": "#0072B2",
        "orange": "#D55E00",
        "gold": "#E69F00",
        "green": "#009E73",
        "magenta": "#B2477A",
        "grey": "#8A8A8A",
        "light_grey": "#D9DDE1",
    }
)


def apply_book_plot_style() -> None:
    """Apply the book-wide quantitative-figure style."""

    mpl.rcParams.update(
        {
            "figure.figsize": (7.2, 4.35),
            "figure.dpi": 130,
            "figure.facecolor": "white",
            "savefig.dpi": 220,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "font.family": "serif",
            "font.serif": ("STIX Two Text", "STIXGeneral", "DejaVu Serif"),
            "font.size": 11.5,
            "mathtext.fontset": "stix",
            "axes.linewidth": 1.0,
            "axes.labelsize": 12.5,
            "axes.titlesize": 13.0,
            "axes.titlepad": 10.0,
            "axes.grid": False,
            "axes.axisbelow": True,
            "axes.prop_cycle": mpl.cycler(
                color=(
                    PAPER_COLORS["blue"],
                    PAPER_COLORS["orange"],
                    PAPER_COLORS["green"],
                    PAPER_COLORS["magenta"],
                    PAPER_COLORS["gold"],
                    PAPER_COLORS["slate"],
                )
            ),
            "lines.linewidth": 1.8,
            "lines.markersize": 5.5,
            "legend.frameon": False,
            "legend.fontsize": 10.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 5.0,
            "ytick.major.size": 5.0,
            "xtick.minor.size": 3.0,
            "ytick.minor.size": 3.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.minor.width": 0.8,
            "ytick.minor.width": 0.8,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
        }
    )


def single_panel(*, width: float = 7.2, height: float = 4.35):
    """Return one consistently sized figure and axes with constrained layout."""

    return plt.subplots(figsize=(width, height), layout="constrained")


def add_quiet_grid(axes, *, axis: str = "both") -> None:
    """Add unobtrusive major guides only when comparison benefits from them."""

    axes.grid(
        visible=True,
        which="major",
        axis=axis,
        color=PAPER_COLORS["light_grey"],
        linewidth=0.65,
        alpha=0.65,
    )
