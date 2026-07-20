"""Shared matplotlib style for PromptLite-Seg report figures."""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams


COLORS = {
    "center_color":             "#E15759",  # red
    "robust_no_color_seed":     "#F28E2B",  # orange
    "robust_no_spatial_prior":  "#4E79A7",  # blue
    "robust_single_box":        "#76B7B2",  # teal
    "robust_superpixel":        "#59A14F",  # green
    "adaptive_superpixel":      "#9C755F",  # brown
    "grabcut_point_box":        "#B07AA1",  # purple
    "sam_point_only":           "#FF9D76",  # peach
    "sam_box_only":             "#499894",  # dark teal
    "sam_point_box":            "#F4A460",  # sandy brown (remapped below)
}

# Final color list matching method order in charts
METHOD_COLORS = [
    "#E15759",  # center_color — red
    "#F28E2B",  # robust_no_color_seed — orange
    "#4E79A7",  # robust_no_spatial_prior — blue
    "#76B7B2",  # robust_single_box — teal
    "#59A14F",  # robust_superpixel — green
    "#B07AA1",  # grabcut_point_box — purple
]

SAM_COLORS = [
    "#FF9D76",  # point_only — peach
    "#499894",  # box_only — dark teal
    "#EDC948",  # point_box — gold
]

FULL_COLORS = METHOD_COLORS + SAM_COLORS


def apply_style() -> None:
    """Apply unified clean style for all report figures."""
    rcParams.update({
        # Figure
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "figure.edgecolor": "white",
        "figure.titlesize": 13,
        "figure.titleweight": "bold",
        # Font
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "font.size": 10,
        # Axes
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.grid.which": "major",
        "grid.alpha": 0.25,
        "grid.linewidth": 0.4,
        "grid.color": "#888888",
        # Legend
        "legend.fontsize": 8,
        "legend.frameon": True,
        "legend.framealpha": 0.85,
        "legend.edgecolor": "#CCCCCC",
        "legend.borderpad": 0.4,
        # Ticks
        "xtick.labelsize": 8,
        "ytick.labelsize": 9,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        # Bar
        "patch.linewidth": 0.3,
        "patch.edgecolor": "white",
        # Lines
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
    })


def save_figure(fig: plt.Figure, path: str, dpi: int = 200) -> None:
    """Save figure with tight bounding box and white background."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white",
                edgecolor="none", pad_inches=0.1)
    plt.close(fig)
