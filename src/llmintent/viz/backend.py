"""Matplotlib backend setup for headless rendering."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

DEFAULT_FIGSIZE = (12, 7)
REGIME_COLORS = {
    "sensory": "#4C72B0",
    "workspace": "#55A868",
    "motor": "#C44E52",
}
MODULE_COLORS = {
    "identity": "#8172B3",
    "reasoning": "#CCB974",
    "meta_reasoning": "#64B5CD",
    "ideation": "#E377C2",
}


def new_figure(nrows: int = 1, ncols: int = 1, figsize=DEFAULT_FIGSIZE):
    sns.set_theme(style="whitegrid", context="notebook")
    return plt.subplots(nrows, ncols, figsize=figsize)


def save_figure(fig, path: str, *, dpi: int = 150) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
