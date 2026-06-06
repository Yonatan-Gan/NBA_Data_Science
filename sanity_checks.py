#!/usr/bin/env python3
"""
Two simple sanity-check visualizations.

  Fig 1 – Histogram of player ages
          Confirms ages are realistic (17–42 range, peak ~26)
          Shows the small number of bad / outlier rows

  Fig 2 – Distribution of points scored per game, across every season
          Confirms no negative scores and no impossible 100+ values
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# ── shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#ebebeb",
    "grid.linewidth":    0.8,
    "axes.titlesize":    16,
    "axes.titleweight":  "bold",
    "axes.labelsize":    13,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "figure.dpi":        150,
})


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Player Age Histogram
# ══════════════════════════════════════════════════════════════════════════════
print("Building Fig 1: Player Age …")

df_age = pd.read_csv("data/processed/q2_player_attributes.csv",
                     usecols=["age_computed"])
df_age["age_computed"] = pd.to_numeric(df_age["age_computed"], errors="coerce")
df_age = df_age.dropna()

VALID_MIN, VALID_MAX = 17, 45

valid   = df_age[df_age["age_computed"].between(VALID_MIN, VALID_MAX)]["age_computed"]
outlier = df_age[~df_age["age_computed"].between(VALID_MIN, VALID_MAX)]["age_computed"]
outlier_low  = outlier[outlier < VALID_MIN]
outlier_high = outlier[outlier > VALID_MAX]

all_min = max(df_age["age_computed"].min(), -5)
all_max = min(df_age["age_computed"].max(), 80)

fig, ax = plt.subplots(figsize=(13, 6))

# Valid ages — blue bars
bins_valid = np.arange(VALID_MIN, VALID_MAX + 2, 1)
ax.hist(valid, bins=bins_valid, color="#2979FF", edgecolor="white",
        linewidth=0.6, zorder=2)

# Outlier ages — 3-year buckets so each bar is tall enough to see, vivid
# red with a thin dark-red border (no white halo).
RED      = "#dc143c"
RED_EDGE = "#7d0a1f"
COARSE   = 3  # 3-year buckets
bins_low  = np.arange(int(all_min) - 1, VALID_MIN + 1, COARSE)
bins_high = np.arange(VALID_MAX, int(all_max) + COARSE + 1, COARSE)

if len(outlier_low) > 0:
    ax.hist(outlier_low,  bins=bins_low,  color=RED,
            edgecolor=RED_EDGE, linewidth=1.0, zorder=10)
if len(outlier_high) > 0:
    ax.hist(outlier_high, bins=bins_high, color=RED,
            edgecolor=RED_EDGE, linewidth=1.0, zorder=10)

# Shade the outlier zones
ax.axvspan(all_min - 1, VALID_MIN,   color="#fde8e8", alpha=0.55, zorder=0)
ax.axvspan(VALID_MAX,   all_max + 1, color="#fde8e8", alpha=0.55, zorder=0)

# Mean of valid ages
mean_age = valid.mean()
ax.axvline(mean_age, color="#333333", linewidth=2, linestyle="--", zorder=5)
ax.text(mean_age + 0.3, ax.get_ylim()[1] * 0.88,
        f"Mean = {mean_age:.1f}", color="#333333", fontsize=10, fontweight="bold")

ax.set_xlabel("Player Age  (years)")
# Main y-axis label — small parenthetical sub-line
ax.set_ylabel("How many records", labelpad=22)
ax.annotate(
    "(one row = one player in one season)",
    xy=(0, 0.5), xycoords="axes fraction",
    xytext=(-40, 0), textcoords="offset points",
    rotation=90, va="center", ha="center",
    fontsize=7.5, color="#777", style="italic",
)

ax.set_title("Sanity Check 1: Player Age Distribution")
ax.set_xlim(all_min - 1, all_max + 1)
ax.set_xticks(range(int(all_min), int(all_max) + 1, 5))

handles = [
    mpatches.Patch(facecolor="#2979FF", edgecolor="white",
                   label=f"Valid ages (17-45 years)  ·  {len(valid):,} rows"),
    mpatches.Patch(facecolor=RED, edgecolor=RED_EDGE, linewidth=1.0,
                   label=f"Outliers (bad data)  ·  {len(outlier):,} rows"),
    plt.Line2D([0], [0], color="#333333", lw=2, linestyle="--",
               label=f"Mean = {mean_age:.1f}"),
]
ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.95,
          title="Legend", title_fontsize=10, fontsize=10)

fig.tight_layout()
fig.savefig(FIG_DIR / "fig1_age_sanity.png", bbox_inches="tight")
plt.close(fig)
print("  ✓  figures/fig1_age_sanity.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Points Per Game by Season (box plot)
# ══════════════════════════════════════════════════════════════════════════════
print("Building Fig 2: Points per Season …")

df_pts = pd.read_csv("data/processed/q1_player_game_logs.csv",
                     usecols=["SEASON", "PTS"], low_memory=False)
df_pts["PTS"]    = pd.to_numeric(df_pts["PTS"],    errors="coerce")
df_pts["SEASON"] = pd.to_numeric(df_pts["SEASON"], errors="coerce")
df_pts = df_pts.dropna(subset=["PTS", "SEASON"])

seasons = sorted(df_pts["SEASON"].unique().astype(int))
data_by_season = [df_pts[df_pts["SEASON"] == s]["PTS"].values for s in seasons]

VIOLIN_FACE  = "#5DADE2"
VIOLIN_EDGE  = "#1B4F72"
ZONE_RED     = "#fde8e8"
ZONE_TEXT    = "#c0392b"

fig, ax = plt.subplots(figsize=(14, 7))

# Violin plot — width at each height = how many games had that score.
# No median line, no internal markers. Just the density shape itself.
parts = ax.violinplot(
    data_by_season,
    positions=range(len(seasons)),
    widths=0.85,
    showmeans=False,
    showmedians=False,
    showextrema=False,
)
for body in parts["bodies"]:
    body.set_facecolor(VIOLIN_FACE)
    body.set_edgecolor(VIOLIN_EDGE)
    body.set_alpha(1.0)        # no fade — keep the tail fully colored
    body.set_linewidth(1.2)

# Out-of-range zones — at <0 (impossible) and >100 (not observed in this data)
ax.axhspan(-8, 0,    color=ZONE_RED, alpha=0.75, zorder=0)
ax.axhspan(100, 115, color=ZONE_RED, alpha=0.75, zorder=0)
mid_x = (len(seasons) - 1) / 2
ax.text(mid_x, -4.5,  "Impossible: scores below 0",
        ha="center", fontsize=10, color=ZONE_TEXT, style="italic")
ax.text(mid_x, 107.5, "Has never happened in NBA history: scores above 100",
        ha="center", fontsize=10, color=ZONE_TEXT, style="italic")

# Annotate Kobe's 81 (max single-game score in the data)
max_row = df_pts.loc[df_pts["PTS"].idxmax()]
kobe_idx = seasons.index(int(max_row["SEASON"]))
ax.annotate(
    "Kobe Bryant: 81 pts\n(Jan 2006)",
    xy=(kobe_idx, 81), xytext=(kobe_idx + 3.5, 92),
    fontsize=9.5, color="#333",
    arrowprops=dict(arrowstyle="->", color="#888", lw=1.2),
    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#ccc"),
)

ax.set_xticks(range(len(seasons)))
ax.set_xticklabels([str(s) for s in seasons], rotation=45, ha="right", fontsize=10)
ax.set_xlabel("NBA Season", labelpad=8)
ax.set_ylabel("Points Scored by One Player in One Game", labelpad=8)
ax.set_title("Sanity Check 2: Points per Player per Game, by Season (2003-2022)")
ax.set_ylim(-8, 115)
ax.set_xlim(-0.7, len(seasons) - 0.3)

# Simple legend — one shape, with a clear explanation of how to read it
from matplotlib.lines import Line2D
legend_handles = [
    mpatches.Patch(facecolor=VIOLIN_FACE, edgecolor=VIOLIN_EDGE,
                   label="Each season's shape: wider = more games\nscored that value (narrow tail = rare scores)"),
]
ax.legend(
    handles=legend_handles, loc="upper left", frameon=True, framealpha=0.95,
    fontsize=10, title="Legend", title_fontsize=11,
    borderpad=0.8, handlelength=2.5, handleheight=2.0, handletextpad=0.8,
)

# Summary stats — top right
total_games = len(df_pts)
ax.text(
    0.99, 0.97,
    f"{total_games:,} total game records\n"
    f"Min: {df_pts['PTS'].min():.0f}    Max: {df_pts['PTS'].max():.0f}    "
    f"Mean: {df_pts['PTS'].mean():.1f}",
    transform=ax.transAxes, va="top", ha="right", fontsize=10.5,
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc", alpha=0.95),
)

fig.tight_layout()
fig.savefig(FIG_DIR / "fig2_points_sanity.png", bbox_inches="tight")
plt.close(fig)
print("  ✓  figures/fig2_points_sanity.png")

print("\nDone.")
