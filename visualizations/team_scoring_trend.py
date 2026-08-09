"""
Line chart: average NBA team score per game, by season (2003-2022).
Output: figures/team_scoring_trend.png
"""

import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

FIG_DIR = Path("figures"); FIG_DIR.mkdir(exist_ok=True)

# Build season averages
df = pd.read_csv("data/processed/q1_player_game_logs.csv",
                 usecols=["GAME_ID", "TEAM_ID", "SEASON", "PTS"],
                 low_memory=False)
df["PTS"]    = pd.to_numeric(df["PTS"],    errors="coerce")
df["SEASON"] = pd.to_numeric(df["SEASON"], errors="coerce")
df = df.dropna()

team_game = df.groupby(["SEASON", "GAME_ID", "TEAM_ID"])["PTS"].sum().reset_index()
avg = team_game.groupby("SEASON")["PTS"].mean().reset_index()
avg.columns = ["SEASON", "AVG_PTS"]
avg = avg.sort_values("SEASON")

seasons = avg["SEASON"].astype(int).tolist()
scores  = avg["AVG_PTS"].tolist()

# Style
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         False,
    "figure.dpi":        150,
})

LINE_COLOR = "#2563EB"   # bold blue
DOT_COLOR  = "#1D4ED8"
AREA_COLOR = "#DBEAFE"   # light blue fill
ANNOT_UP   = "#16a34a"   # green for "up" annotations
ANNOT_DN   = "#dc2626"   # red for "down" annotations

fig, ax = plt.subplots(figsize=(14, 7))

x = list(range(len(seasons)))

# Filled area under the line
ax.fill_between(x, scores, min(scores) - 1, color=AREA_COLOR, alpha=0.6, zorder=1)

# Main line
ax.plot(x, scores, color=LINE_COLOR, linewidth=2.8, zorder=3)

# Dots at each season
ax.scatter(x, scores, color="white", edgecolor=DOT_COLOR,
           s=80, linewidth=2.2, zorder=4)

# Annotate a few notable moments
notable = {
    # (label, color, side, x_offset for text)
    2011: ("Lockout-shortened\nseason", ANNOT_DN, "above", 0.0),
    2016: ("3-point revolution\ntakes hold", ANNOT_UP, "above", 0.0),
    2018: ("Pace era begins", ANNOT_UP, "above", 0.0),
}

for season, (label, color, side, x_offset) in notable.items():
    if season in seasons:
        xi = seasons.index(season)
        yi = scores[xi]
        dy = 5.5 if side == "above" else -3.5
        ax.annotate(
            label,
            xy=(xi, yi), xytext=(xi + x_offset, yi + dy),
            ha="center", va="bottom" if side == "above" else "top",
            fontsize=8.5, color=color, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=color,
                            lw=1.3, connectionstyle="arc3,rad=0"),
        )

# Season labels & score labels
ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in seasons], rotation=45, ha="right", fontsize=10.5)

# Score label above every dot
for xi, yi in zip(x, scores):
    ax.text(xi, yi + 0.55, f"{yi:.1f}", ha="center", va="bottom",
            fontsize=8, color=DOT_COLOR, fontweight="bold")

# Axes
ax.set_ylim(88, 118)
ax.set_xlim(-0.5, len(x) - 0.5)
ax.set_ylabel("Average Points per Game", fontsize=12, labelpad=10)

# Horizontal reference lines every 5 pts
for y_ref in range(90, 116, 5):
    ax.axhline(y_ref, color="#e2e8f0", linewidth=0.9, zorder=0)
    ax.text(-0.6, y_ref, str(y_ref), ha="right", va="center",
            fontsize=9, color="#94a3b8")

# Hide y-axis ticks/spine
ax.tick_params(axis="y", which="both", left=False, labelleft=False)
ax.spines["left"].set_visible(False)
ax.yaxis.set_visible(False)

# Y label placed far enough left to clear the manual number labels
fig.text(0.005, 0.5, "Average Points per Game",
         va="center", ha="left", rotation=90, fontsize=12, color="#333")

ax.set_xlabel("Season", fontsize=12, labelpad=10)

# % increase callout
pct = (scores[-1] - scores[0]) / scores[0] * 100
ax.text(0.985, 0.06,
        f"+{pct:.0f}% scoring increase\nsince 2003",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=12, fontweight="bold", color=ANNOT_UP,
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#bbf7d0", linewidth=1.5))

# Title / subtitle
fig.suptitle("NBA Teams Are Scoring More Than Ever",
             fontsize=19, fontweight="bold", y=0.98)
fig.text(0.5, 0.915,
         "Average points scored per team per game · Regular season · 2003-2022",
         ha="center", fontsize=11, color="#555")

plt.subplots_adjust(left=0.06, right=0.97, top=0.87, bottom=0.13)
fig.savefig(FIG_DIR / "team_scoring_trend.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved figures/team_scoring_trend.png")
