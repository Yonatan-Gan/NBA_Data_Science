"""Shared paths and publication styling for the candidate analyses."""

from __future__ import annotations

from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
NBA = DATA / "NBA_api"
OUTPUT = HERE / "output"
FIGURES = OUTPUT / "figures"
TABLES = OUTPUT / "tables"
PDF = OUTPUT / "pdf"

for directory in (FIGURES, TABLES, PDF):
    directory.mkdir(parents=True, exist_ok=True)

SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
SEASON_ORDER = {season: i for i, season in enumerate(SEASONS)}
RANDOM_SEED = 42

# Color-blind-friendly palette with stable semantic roles.
INK = "#182230"
MUTED = "#64748B"
GRID = "#DDE3EA"
BLUE = "#2563A6"
TEAL = "#138A7E"
ORANGE = "#D97706"
RED = "#B84A4A"
PALE_BLUE = "#DCEAF7"
PALE_TEAL = "#D8F0EC"
PALE_ORANGE = "#F8E8CC"


def set_style() -> None:
    """Apply a restrained report style with legible labels and light grids."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "axes.labelcolor": INK,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": INK,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.alpha": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def finish_figure(fig: plt.Figure, stem: str, *, dpi: int = 220) -> None:
    """Save both raster and vector versions for the report."""
    fig.savefig(FIGURES / f"{stem}.png", dpi=dpi, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def add_source(fig: plt.Figure, text: str) -> None:
    fig.text(0.01, -0.015, text, color=MUTED, fontsize=8, ha="left", va="top")


def cluster_bootstrap_difference(
    losses_a: np.ndarray,
    losses_b: np.ndarray,
    clusters: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = RANDOM_SEED,
) -> tuple[float, float, float]:
    """Bootstrap mean(loss_a - loss_b), resampling whole clusters."""
    rng = np.random.default_rng(seed)
    unique = np.unique(clusters)
    by_cluster = {c: np.flatnonzero(clusters == c) for c in unique}
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([by_cluster[c] for c in sampled])
        diffs[i] = np.mean(losses_a[idx] - losses_b[idx])
    point = float(np.mean(losses_a - losses_b))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, float(lo), float(hi)


def seed_everything(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
