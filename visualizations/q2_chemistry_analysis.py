"""
Q2 - Team Chemistry and Success
===============================
Extracts team-level usage and age variance from the NBA API season stats,
calculates a weighted Composite Success Score (Net Rating + Playoff Wins),
and generates visualizations mapping roster construction to success.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import zscore
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

# ============================================================
# DIRECTORIES & STYLE SETUP
# ============================================================
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_PROC = os.path.join(BASE, "data", "processed", "NBA_api")
SEASON_STATS_DIR = os.path.join(API_PROC, "season_stats")
ROSTERS_DIR = os.path.join(API_PROC, "rosters")
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

HIGH_SUCCESS_COL = "#27ae60"  # Riser green
NEUTRAL_COL      = "#7f8c8d"  # Neutral gray
LOW_SUCCESS_COL  = "#c0392b"  # Decliner red

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.grid":         True,
    "grid.color":        "#ebebeb",
    "grid.linewidth":    0.8,
})

# ============================================================
# LOAD DATA AND BUILD COMPOSITE SCORE
# ============================================================
print("Loading NBA API data and building chemistry features...")

if not os.path.exists(SEASON_STATS_DIR):
    raise FileNotFoundError(f"Missing directory: {SEASON_STATS_DIR}")

# Load Team History for Playoff Wins
team_history_path = os.path.join(ROSTERS_DIR, "team_year_by_year.csv")
if os.path.exists(team_history_path):
    df_team_history = pd.read_csv(team_history_path, low_memory=False).drop_duplicates()
else:
    print("  WARNING: team_year_by_year.csv not found. Playoff wins will default to 0.")
    df_team_history = pd.DataFrame()

seasons = [d for d in os.listdir(SEASON_STATS_DIR) if os.path.isdir(os.path.join(SEASON_STATS_DIR, d))]
all_teams_features = []

for season in sorted(seasons):
    season_path = os.path.join(SEASON_STATS_DIR, season)
    team_adv_path = os.path.join(season_path, "team_stats_regular_season_advanced.csv")
    player_usg_path = os.path.join(season_path, "player_stats_regular_season_usage.csv")
    
    if not os.path.exists(team_adv_path) or not os.path.exists(player_usg_path):
        continue
        
    df_team_adv = pd.read_csv(team_adv_path, low_memory=False).drop_duplicates()
    df_player_usg = pd.read_csv(player_usg_path, low_memory=False).drop_duplicates()
    
    # FILTER: Only keep players with >= 20 games for the team to remove 10-day contracts/trades
    df_player_usg = df_player_usg[df_player_usg["GP"] >= 20]
    
    # Quantify Chemistry (Variance in Usage and Age)
    team_chemistry = df_player_usg.groupby("TEAM_ID").agg(
        USG_VARIANCE=("USG_PCT", "std"),
        USG_MAX=("USG_PCT", "max"),
        AGE_VARIANCE=("AGE", "std"),
        AGE_MEAN=("AGE", "mean")
    ).reset_index()
    
    # Merge with Net Rating
    season_merged = df_team_adv[["TEAM_ID", "TEAM_NAME", "W_PCT", "NET_RATING"]].merge(
        team_chemistry, on="TEAM_ID", how="inner"
    )
    
    # Map Playoff Wins
    if not df_team_history.empty:
        history_season = df_team_history[df_team_history["YEAR"] == season]
        season_merged = season_merged.merge(
            history_season[["TEAM_ID", "PO_WINS"]], on="TEAM_ID", how="left"
        )
    else:
        season_merged["PO_WINS"] = 0
        
    season_merged["SEASON"] = season
    all_teams_features.append(season_merged)

df = pd.concat(all_teams_features, ignore_index=True)

# Handle NaNs
df["PO_WINS"] = pd.to_numeric(df["PO_WINS"], errors='coerce').fillna(0)
df = df.dropna(subset=["NET_RATING", "W_PCT", "USG_VARIANCE", "AGE_VARIANCE"])

# Build Composite Success Score (45% Net Rating, 45% PO Wins, 10% Win PCT)
df["z_NET_RATING"] = zscore(df["NET_RATING"])
df["z_PO_WINS"] = zscore(df["PO_WINS"])
df["z_W_PCT"] = zscore(df["W_PCT"])

df["COMPOSITE"] = (0.45 * df["z_NET_RATING"]) + (0.45 * df["z_PO_WINS"]) + (0.10 * df["z_W_PCT"])

# Tiering for categorical visualizations
top_third = df["COMPOSITE"].quantile(0.66)
bottom_third = df["COMPOSITE"].quantile(0.33)

df["SUCCESS_TIER"] = "Average"
df.loc[df["COMPOSITE"] >= top_third, "SUCCESS_TIER"] = "High Success"
df.loc[df["COMPOSITE"] <= bottom_third, "SUCCESS_TIER"] = "Low Success"

n_total = len(df)
n_high = (df["SUCCESS_TIER"] == "High Success").sum()
n_low = (df["SUCCESS_TIER"] == "Low Success").sum()
print(f"  {n_total:,} team-seasons  |  High Success {n_high}  |  Low Success {n_low}")

# ============================================================
# FIG 1 - Usage Variance Distribution
# ============================================================
print("Fig 1 (Usage Variance)...")

fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle("Does Sharing the Ball Win Championships?", fontsize=13, fontweight="bold", y=0.98)

high_df = df[df["SUCCESS_TIER"] == "High Success"]["USG_VARIANCE"].dropna()
low_df = df[df["SUCCESS_TIER"] == "Low Success"]["USG_VARIANCE"].dropna()

parts = ax.violinplot([low_df, high_df], positions=[0, 1], widths=0.6, 
                      showmeans=True, showextrema=False)

colors = [LOW_SUCCESS_COL, HIGH_SUCCESS_COL]
for pc, color in zip(parts['bodies'], colors):
    pc.set_facecolor(color)
    pc.set_edgecolor(color)
    pc.set_alpha(0.7)
    
parts['cmeans'].set_color('#333333')
parts['cmeans'].set_linewidth(2)

ax.set_xticks([0, 1])
ax.set_xticklabels([f"Low Success Teams\n(n={len(low_df)})", f"High Success Teams\n(n={len(high_df)})"], 
                   fontsize=11, fontweight="bold")
ax.set_ylabel("Usage Variance (Standard Deviation of USG%)", fontsize=11)
ax.set_title("Distribution of Usage Variance Among Top Rotation Players", fontsize=11, color="#555555", style="italic")

handles = [
        mpatches.Patch(facecolor=HIGH_SUCCESS_COL, alpha=0.7, label="Top 33% of Teams"),
        mpatches.Patch(facecolor=LOW_SUCCESS_COL, alpha=0.7, label="Bottom 33% of Teams")
    ]
    
ax.legend(
    handles=handles, 
    loc="upper left",          # The anchor point of the legend box
    bbox_to_anchor=(1.02, 1),  # Pushes the box completely OUTSIDE the right edge of the graph
    frameon=True,              # Puts it inside a visible box
    title="Legend",            # Adds your requested headline
    title_fontsize=11,         # Makes the headline slightly larger
    fontsize=10,
    edgecolor="#cccccc"        # Adds a clean, subtle gray border to the box
)

fig.tight_layout()
p1 = os.path.join(FIG, "q2_fig1_usage_variance.png")
fig.savefig(p1, dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# FIG 2 - Age Barbell Scatter
# ============================================================
print("Fig 2 (Age Barbell Strategy)...")

fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle("The Barbell Strategy in Team Roster Construction", fontsize=13, fontweight="bold", y=0.98)

sns.regplot(
    x="AGE_VARIANCE", y="COMPOSITE", data=df, 
    scatter_kws={"color": NEUTRAL_COL, "alpha": 0.6, "s": 40, "edgecolors": "none"}, 
    line_kws={"color": "#333333", "linewidth": 2.5}, ax=ax
)

ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.3)

ax.set_xlabel("Roster Age Variance (Standard Deviation in Years)", fontsize=11)
ax.set_ylabel("Composite Team Success Score", fontsize=11)
ax.set_title("Does a mix of young rookies and older veterans yield higher success?", fontsize=11, color="#555555", style="italic")

fig.tight_layout()
p2 = os.path.join(FIG, "q2_fig2_age_barbell.png")
fig.savefig(p2, dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# FIG 3 - Feature Importance (Random Forest Regressor)
# ============================================================
print("Fig 3 (Feature Importance)...")

features = ["USG_VARIANCE", "USG_MAX", "AGE_VARIANCE", "AGE_MEAN"]
clean_names = {
    "USG_VARIANCE": "Usage Distribution (Ego Equality)",
    "USG_MAX": "Alpha Star Usage (Highest USG%)",
    "AGE_VARIANCE": "Age Diversity (Barbell Strategy)",
    "AGE_MEAN": "Average Roster Age"
}

X = df[features].fillna(0).values
y = df["COMPOSITE"].values

rf = RandomForestRegressor(n_estimators=300, max_depth=5, random_state=42, n_jobs=-1)
rf.fit(X, y)

imp = pd.Series(rf.feature_importances_ * 100, index=features)
imp.index = [clean_names.get(c, c) for c in imp.index]
imp = imp.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
fig.suptitle("Which Roster Dynamic Actually Drives Team Success?", fontsize=13, fontweight="bold", y=0.98)

# Color the most important feature green, the rest neutral
colors = [HIGH_SUCCESS_COL if v == imp.max() else NEUTRAL_COL for v in imp.values]
bars = ax.barh(imp.index, imp.values, color=colors, edgecolor="white", height=0.6)

ax.set_xlabel("Importance score for predicting Team Success (%)", fontsize=11)
ax.set_title("Random Forest Regressor Feature Importance (using engineered chemistry metrics)", fontsize=10, color="#555555", style="italic")

# Add text labels to the bars
for i, v in enumerate(imp.values):
    ax.text(v + 0.5, i, f"{v:.1f}%", va='center', fontweight='bold', color='#333333')
    
ax.set_xlim(0, imp.max() + 5)

fig.tight_layout()
p3 = os.path.join(FIG, "q2_fig3_feature_importance.png")
fig.savefig(p3, dpi=150, bbox_inches="tight")
plt.close()

print("\nPipeline Complete.")
print(f"  Fig 1: {os.path.basename(p1)}")
print(f"  Fig 2: {os.path.basename(p2)}")
print(f"  Fig 3: {os.path.basename(p3)}")