#!/usr/bin/env python
"""Generate deterministic Chapter 1 teaching schematics."""
from pathlib import Path
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "stellar_spectroscopy_mpl"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


BOOK = Path(__file__).resolve().parent.parent
FIG = BOOK / "resources" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

INK = "#20242E"
SLATE = "#EEF2F7"
SLATE_DARK = "#CBD4E1"
AMBER = "#E8A23D"
MUTED = "#6B7280"


def box(ax, x, y, w, h, text, *, face=SLATE, edge=INK, fontsize=16):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.02",
        linewidth=1.4,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=INK)


def arrow(ax, start, end, *, color=INK, lw=1.8, mutation_scale=18):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
        )
    )


def mini_plot(ax, x, y, w, h, label, curve_x, curve_y, *, color=INK):
    box(ax, x, y, w, h, "", fontsize=12)
    ax.plot(x + 0.18 * w + 0.68 * w * curve_x, y + 0.18 * h + 0.64 * h * curve_y, color=color, lw=2.5)
    arrow(ax, (x + 0.17 * w, y + 0.17 * h), (x + 0.86 * w, y + 0.17 * h), lw=1.2, mutation_scale=12)
    arrow(ax, (x + 0.17 * w, y + 0.17 * h), (x + 0.17 * w, y + 0.85 * h), lw=1.2, mutation_scale=12)
    ax.text(x + 0.5 * w, y + 0.88 * h, label, ha="center", va="center", fontsize=15, color=INK)
    ax.text(x + 0.52 * w, y + 0.06 * h, "inward", ha="center", va="center", fontsize=11, color=MUTED)


def atmosphere_structure():
    fig, ax = plt.subplots(figsize=(14.08, 7.68), dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.93, "A first model atmosphere is a depth grid with physics assigned to each layer",
            ha="center", va="center", fontsize=22, color=INK)

    slab_x, slab_y, slab_w, slab_h = 0.31, 0.20, 0.25, 0.62

    box(ax, 0.06, 0.60, 0.12, 0.10, r"$T_{\rm eff}$", fontsize=22)
    box(ax, 0.06, 0.43, 0.12, 0.10, r"$\log g$", fontsize=22)
    ax.text(0.12, 0.34, "stellar\nparameters", ha="center", va="center", fontsize=14, color=MUTED)
    arrow(ax, (0.20, 0.65), (slab_x - 0.005, 0.65), lw=2.0, mutation_scale=22)

    slab = FancyBboxPatch(
        (slab_x, slab_y),
        slab_w,
        slab_h,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=1.4,
        edgecolor=INK,
        facecolor=SLATE,
    )
    ax.add_patch(slab)
    for i in range(14):
        y = slab_y + slab_h * (i + 1) / 15
        ax.plot([slab_x, slab_x + slab_w], [y, y], color=SLATE_DARK, lw=1)
    ax.text(slab_x + slab_w / 2, slab_y + slab_h + 0.045, "plane-parallel layers",
            ha="center", va="center", fontsize=16, color=INK)

    axis_x = slab_x - 0.030
    arrow(ax, (axis_x, slab_y + slab_h), (axis_x, slab_y), lw=1.8, mutation_scale=18)
    ax.text(axis_x - 0.020, slab_y + slab_h, r"$\tau \ll 1$", ha="right", va="center", fontsize=15, color=INK)
    ax.text(axis_x - 0.020, slab_y, r"$\tau \gg 1$", ha="right", va="center", fontsize=15, color=INK)
    ax.text(axis_x - 0.003, slab_y + slab_h / 2, "inward",
            ha="center", va="center", rotation=90, fontsize=14, color=INK)

    yy = np.linspace(0, 1, 200)
    tcurve_x = 0.18 + 0.45 / (1 + np.exp(-7 * (yy - 0.48)))
    ax.plot(slab_x + slab_w * tcurve_x, slab_y + slab_h * (1 - yy), color=INK, lw=3)
    ax.text(slab_x + 0.62 * slab_w, slab_y + 0.76 * slab_h, r"$T(\tau)$",
            ha="center", va="center", fontsize=17, color=INK)

    arrow(ax, (slab_x + 0.50 * slab_w, slab_y + 0.82 * slab_h),
          (slab_x + 0.50 * slab_w, slab_y + 0.18 * slab_h),
          color=AMBER, lw=5, mutation_scale=24)
    ax.text(slab_x + 0.31 * slab_w, slab_y + 0.50 * slab_h,
            "pressure builds\nfrom overlying mass",
            ha="center", va="center", fontsize=14, color=INK)

    curve_x = np.linspace(0, 1, 100)
    mini_plot(ax, 0.70, 0.54, 0.12, 0.22, r"$T$ vs depth", curve_x, curve_x**0.55, color=INK)
    mini_plot(ax, 0.84, 0.54, 0.12, 0.22, r"$P_{\rm gas}$ vs depth", curve_x, curve_x**2.0, color=INK)
    mini_plot(ax, 0.70, 0.23, 0.12, 0.22, r"$m$ vs depth", curve_x, curve_x, color=INK)
    mini_plot(ax, 0.84, 0.23, 0.12, 0.22, r"$B_\lambda[T]$", curve_x, np.exp(2.3 * curve_x) / np.exp(2.3), color=AMBER)

    ax.text(0.5, 0.07,
            r"$T_{\rm eff}$ fixes the flux scale; $\log g$ fixes the weight scale; $\tau$ organizes where photons escape.",
            ha="center", va="center", fontsize=15, color=MUTED)

    out = FIG / "s1_atmosphere_structure.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    atmosphere_structure()
