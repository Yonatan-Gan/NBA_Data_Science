#!/usr/bin/env python3
"""
Dumbbell plot of the top 10 playoff risers and top 10 playoff decliners.
Each row shows a player's regular-season scoring average -> playoff scoring
average, with an arrow showing the direction and magnitude of the change.

Output: figures/playoff_risers_decliners.png
"""

import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

FIG_DIR = Path("figures"); FIG_DIR.mkdir(exist_ok=True)

# Load & filter for meaningful sample sizes

df = pd.read_csv("data/processed/q3_player_split_v2.csv")
df = df.dropna(subset=["REG_PTS", "POF_PTS"])

# Only meaningful players: real regular season, real playoff sample, real scorer
strong = df[(df["POF_G"] >= 10) & (df["REG_G"] >= 50) & (df["REG_PTS"] >= 15)].copy()

risers    = strong.nlargest(10,  "DELTA_PTS").copy()
decliners = strong.nsmallest(10, "DELTA_PTS").copy()

# Build season-label "2022-23" style for clarity
def _season_label(yr):
    return f"{int(yr)}-{str(int(yr) + 1)[-2:]}"
for d in (risers, decliners):
    d["season_label"] = d["season_start"].apply(_season_label)
    d["row_label"]    = d["Player"] + "   " + d["season_label"]

# Style
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  False,
    "axes.grid":         False,
    "axes.titlesize":    17,
    "axes.titleweight":  "bold",
})

REG_COLOR    = "#7f8c8d"   # gray for regular season dot
RISER_COLOR  = "#2ecc71"   # green
FALL_COLOR   = "#e74c3c"   # red
DIVIDER      = "#bdc3c7"

fig, ax = plt.subplots(figsize=(14, 12))

# Stack: decliners first (bottom 10), then divider gap, then risers (top 10)
# Within each block, biggest swing is FARTHEST from center (most dramatic at the edges).
risers_sorted    = risers.sort_values("DELTA_PTS", ascending=True)   # biggest riser at the TOP
decliners_sorted = decliners.sort_values("DELTA_PTS", ascending=False)  # biggest decliner at the BOTTOM

y_decliners = list(range(0, len(decliners_sorted)))
y_risers    = list(range(len(decliners_sorted) + 3,
                          len(decliners_sorted) + 3 + len(risers_sorted)))

def _draw(rows, ys, color):
    for y, (_, r) in zip(ys, rows.iterrows()):
        reg, pof = r["REG_PTS"], r["POF_PTS"]
        # connector line (with arrowhead)
        ax.annotate(
            "", xy=(pof, y), xytext=(reg, y),
            arrowprops=dict(arrowstyle="->", color=color, lw=2.5,
                            shrinkA=6, shrinkB=6),
            zorder=3,
        )
        # regular-season dot
        ax.scatter(reg, y, s=140, color="white", edgecolor=REG_COLOR,
                   linewidth=2, zorder=4)
        ax.scatter(reg, y, s=40, color=REG_COLOR, zorder=5)
        # playoff dot
        ax.scatter(pof, y, s=180, color=color, zorder=6,
                   edgecolor="white", linewidth=1.5)
        # value labels
        ax.text(reg, y + 0.32, f"{reg:.1f}", ha="center", va="bottom",
                fontsize=8.5, color=REG_COLOR)
        ax.text(pof, y + 0.32, f"{pof:.1f}", ha="center", va="bottom",
                fontsize=9, color=color, fontweight="bold")
        # delta on the right
        sign = "+" if r["DELTA_PTS"] > 0 else "−"
        ax.text(48, y, f"{sign}{abs(r['DELTA_PTS']):.1f} pts",
                va="center", ha="right", fontsize=10.5, color=color,
                fontweight="bold")

_draw(decliners_sorted, y_decliners, FALL_COLOR)
_draw(risers_sorted,    y_risers,    RISER_COLOR)

# Divider line between the two groups
gap_y = (max(y_decliners) + min(y_risers)) / 2
ax.axhline(gap_y, color=DIVIDER, linewidth=1, linestyle="--", alpha=0.6)

# Section labels at the LEFT, above each group (in the y-tick-label column)
ax.text(-13, y_risers[-1] + 1.2, "▲  BIGGEST RISERS",
        fontsize=13, fontweight="bold", color=RISER_COLOR,
        ha="left", va="center", clip_on=False)
ax.text(-13, y_decliners[-1] + 1.2, "▼  BIGGEST DECLINERS",
        fontsize=13, fontweight="bold", color=FALL_COLOR,
        ha="left", va="center", clip_on=False)

# Y-axis: player + season labels
all_y     = y_decliners + y_risers
all_label = list(decliners_sorted["row_label"]) + list(risers_sorted["row_label"])
ax.set_yticks(all_y)
ax.set_yticklabels(all_label, fontsize=10.5)

# Color the y-tick labels by category
for lbl, y in zip(ax.get_yticklabels(), all_y):
    lbl.set_color(FALL_COLOR if y in y_decliners else RISER_COLOR)
    lbl.set_fontweight("bold")

# X axis
ax.set_xlim(0, 50)
ax.set_xticks(range(0, 41, 5))
ax.set_xlabel("Points Per Game", fontsize=12, labelpad=10)
ax.grid(axis="x", color="#ecf0f1", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
ax.set_ylim(-1, max(y_risers) + 1.5)

# Title and subtitle (positioned with figure-level text so they have room)
fig.suptitle("Playoff Performance Shift",
             fontsize=18, fontweight="bold", y=0.97)
fig.text(0.5, 0.935,
         "Top 10 risers and top 10 decliners - change in scoring from regular season to playoffs",
         ha="center", fontsize=11, color="#555")
fig.text(0.5, 0.915,
         "Filtered to ≥ 50 regular-season games, ≥ 10 playoff games, ≥ 15.0 regular-season PPG  ·  Seasons 1995-2023",
         ha="center", fontsize=9, color="#888", style="italic")

# Legend - just the grey dot for regular-season PPG
legend_handles = [
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=REG_COLOR,
               markeredgecolor=REG_COLOR, markersize=11, linestyle="None",
               label="Regular-season PPG"),
]
ax.legend(handles=legend_handles,
          loc="lower center", bbox_to_anchor=(0.5, -0.10),
          frameon=False, fontsize=11,
          handlelength=2.5, handletextpad=0.8)

plt.subplots_adjust(left=0.20, right=0.96, top=0.89, bottom=0.10)
fig.savefig(FIG_DIR / "playoff_risers_decliners.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved figures/playoff_risers_decliners.png")
