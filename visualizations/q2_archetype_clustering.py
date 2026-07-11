#!/usr/bin/env python3
"""
Q2 Deep Dive - Unsupervised Learning (Team Archetypes)
======================================================
Uses K-Means clustering to group the 180 team-seasons into
distinct roster archetypes based on Age and Usage Dynamics.
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore

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

def load_and_build_features():
    print("Extracting chemistry features for Unsupervised Learning...")
    
    team_history_path = os.path.join(ROSTERS_DIR, "team_year_by_year.csv")
    if os.path.exists(team_history_path):
        df_team_history = pd.read_csv(team_history_path, low_memory=False).drop_duplicates()
    else:
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
        
        # Drop mid-season anomalies
        df_player_usg = df_player_usg[df_player_usg["GP"] >= 20]
        
        team_chemistry = df_player_usg.groupby("TEAM_ID").agg(
            USG_VARIANCE=("USG_PCT", "std"),
            AGE_VARIANCE=("AGE", "std"),
            AGE_MEAN=("AGE", "mean")
        ).reset_index()
        
        season_merged = df_team_adv[["TEAM_ID", "TEAM_NAME", "W_PCT", "NET_RATING"]].merge(
            team_chemistry, on="TEAM_ID", how="inner"
        )
        
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
    df["PO_WINS"] = pd.to_numeric(df["PO_WINS"], errors='coerce').fillna(0)
    df = df.dropna(subset=["NET_RATING", "W_PCT", "USG_VARIANCE", "AGE_VARIANCE"])

    df["z_NET_RATING"] = zscore(df["NET_RATING"])
    df["z_PO_WINS"] = zscore(df["PO_WINS"])
    df["z_W_PCT"] = zscore(df["W_PCT"])
    df["COMPOSITE_SCORE"] = (0.45 * df["z_NET_RATING"]) + (0.45 * df["z_PO_WINS"]) + (0.10 * df["z_W_PCT"])

    return df

def main():
    df = load_and_build_features()

    # Prepare features for the K-Means algorithm
    features = ["USG_VARIANCE", "AGE_MEAN", "AGE_VARIANCE"]
    X = df[features]
    
    # Standardize the features so Age doesn't overpower Usage Variance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply K-Means
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["CLUSTER"] = kmeans.fit_predict(X_scaled)

    # Dynamically name the clusters based on their mathematical centroids
    cluster_means = df.groupby("CLUSTER")[features].mean()
    global_age_mean = df["AGE_MEAN"].mean()
    global_usg_mean = df["USG_VARIANCE"].mean()

    def name_cluster(row):
        age_label = "Veteran" if row["AGE_MEAN"] > global_age_mean else "Young"
        usg_label = "Star-Centric" if row["USG_VARIANCE"] > global_usg_mean else "Democratic"
        return f"{age_label} & {usg_label}"

    cluster_names = {c: name_cluster(cluster_means.loc[c]) for c in cluster_means.index}
    df["ARCHETYPE"] = df["CLUSTER"].map(cluster_names)

    print("Building Fig 4: Archetype Clusters ...")
    build_cluster_scatter(df)

    print("Building Fig 5: Success by Archetype ...")
    build_success_by_archetype(df)

    print("\nDeep Dive Complete.")
    print("  Fig 4: q2_fig4_archetype_clusters.png")
    print("  Fig 5: q2_fig5_success_by_archetype.png")

def build_cluster_scatter(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Unsupervised Learning: NBA Team Archetypes", fontsize=13, fontweight="bold", y=0.98)

    palette = sns.color_palette("Set2", n_colors=4)

    sns.scatterplot(
        data=df, x="AGE_MEAN", y="USG_VARIANCE", hue="ARCHETYPE",
        palette=palette, s=75, alpha=0.85, edgecolor="white", ax=ax
    )

    ax.set_xlabel("Average Roster Age (Years)", fontsize=11)
    ax.set_ylabel("Usage Variance (Star Concentration)", fontsize=11)
    ax.set_title("K-Means Clustering of 180 Team-Seasons based on Age and Usage", fontsize=11, color="#555555", style="italic")

    ax.legend(
        title="Machine-Defined Archetype", 
        bbox_to_anchor=(1.02, 1), 
        loc="upper left", 
        frameon=True,
        edgecolor="#cccccc"
    )
    
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "q2_fig4_archetype_clusters.png"), dpi=150, bbox_inches="tight")
    plt.close()

def build_success_by_archetype(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Which Team Archetype Actually Wins?", fontsize=13, fontweight="bold", y=0.98)

    avg_success = df.groupby("ARCHETYPE")["COMPOSITE_SCORE"].mean().sort_values(ascending=False)

    colors = ["#27ae60" if val > 0 else "#c0392b" for val in avg_success.values]

    bars = ax.bar(avg_success.index, avg_success.values, color=colors, edgecolor="white", alpha=0.85, width=0.6)

    ax.set_ylabel("Average Composite Success Score", fontsize=11)
    ax.set_title("Average Success Score by K-Means Cluster", fontsize=11, color="#555555", style="italic")
    ax.axhline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.4)

    for bar in bars:
        yval = bar.get_height()
        offset = 0.05 if yval > 0 else -0.1
        ax.text(bar.get_x() + bar.get_width()/2, yval + offset, f"{yval:+.2f}",
                ha='center', va='bottom' if yval > 0 else 'top', fontweight='bold', fontsize=10)

    plt.xticks(rotation=0, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "q2_fig5_success_by_archetype.png"), dpi=150, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()