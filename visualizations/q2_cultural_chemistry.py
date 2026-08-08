"""
Q2 Finale - Cultural Synergy & Chemistry
======================================================
Processes local player origin data and generates the final
plots mapping international diversity against ball movement and success.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_PROC = os.path.join(BASE, "data", "processed", "NBA_api")
SEASON_STATS_DIR = os.path.join(API_PROC, "season_stats")
ROSTERS_DIR = os.path.join(API_PROC, "rosters")
BIO_FILE = os.path.join(API_PROC, "player_profiles", "player_bio_info.csv")
FIG = os.path.join(BASE, "figures")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.color": "#ebebeb",
})

def load_local_data():
    if not os.path.exists(BIO_FILE):
        raise FileNotFoundError(f"Cannot find bio file at {BIO_FILE}")

    # Load from the existing bio file you pointed out
    bio_df = pd.read_csv(BIO_FILE)
    id_col = "PERSON_ID" if "PERSON_ID" in bio_df.columns else "PLAYER_ID"
    origins_dict = dict(zip(bio_df[id_col], bio_df["COUNTRY"]))
    
    team_history_path = os.path.join(ROSTERS_DIR, "team_year_by_year.csv")
    df_team_history = pd.read_csv(team_history_path) if os.path.exists(team_history_path) else pd.DataFrame()

    seasons = [d for d in os.listdir(SEASON_STATS_DIR) if os.path.isdir(os.path.join(SEASON_STATS_DIR, d))]
    all_teams_features = []

    for season in sorted(seasons):
        team_adv_path = os.path.join(SEASON_STATS_DIR, season, "team_stats_regular_season_advanced.csv")
        player_usg_path = os.path.join(SEASON_STATS_DIR, season, "player_stats_regular_season_usage.csv")
        
        if not os.path.exists(team_adv_path) or not os.path.exists(player_usg_path): continue
            
        df_team_adv = pd.read_csv(team_adv_path)
        df_player_usg = pd.read_csv(player_usg_path)
        df_player_usg = df_player_usg[df_player_usg["GP"] >= 15]
        
        # Map countries and default to USA if a player somehow isn't in the bio file
        df_player_usg["COUNTRY"] = df_player_usg["PLAYER_ID"].map(origins_dict).fillna("USA")
        df_player_usg["IS_INTL"] = np.where(df_player_usg["COUNTRY"] != "USA", 1, 0)
        
        team_culture = df_player_usg.groupby("TEAM_ID").agg(INTL_PCT=("IS_INTL", "mean")).reset_index()
        
        season_merged = df_team_adv[["TEAM_ID", "TEAM_NAME", "W_PCT", "NET_RATING", "AST_PCT"]].merge(
            team_culture, on="TEAM_ID", how="inner"
        )
        
        if not df_team_history.empty:
            history_season = df_team_history[df_team_history["YEAR"] == season]
            season_merged = season_merged.merge(history_season[["TEAM_ID", "PO_WINS"]], on="TEAM_ID", how="left")
        else:
            season_merged["PO_WINS"] = 0
            
        all_teams_features.append(season_merged)

    df = pd.concat(all_teams_features, ignore_index=True).dropna()
    df["PO_WINS"] = pd.to_numeric(df["PO_WINS"], errors='coerce').fillna(0)
    
    df["z_NET_RATING"] = zscore(df["NET_RATING"])
    df["z_PO_WINS"] = zscore(df["PO_WINS"])
    df["z_W_PCT"] = zscore(df["W_PCT"])
    df["COMPOSITE_SCORE"] = (0.45 * df["z_NET_RATING"]) + (0.45 * df["z_PO_WINS"]) + (0.10 * df["z_W_PCT"])
    return df

def main():
    print("Loading data directly from your pre-existing bio file...")
    df = load_local_data()

    # Fig 6
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.regplot(data=df, x="INTL_PCT", y="AST_PCT", 
                scatter_kws={"alpha": 0.6, "s": 60, "color": "#3498db", "edgecolor": "white"},
                line_kws={"color": "#e74c3c", "linewidth": 2}, ax=ax)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Does International Diversity Drive Democratic Ball Movement?", fontsize=12, fontweight="bold", pad=15)
    ax.set_xlabel("International Players in Rotation (%)")
    ax.set_ylabel("Assist Percentage (AST%)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "q2_fig6_diversity_vs_passing.png"), dpi=150)
    plt.close()

    # 1. UPDATE THE LABELS HERE:
    df["DIVERSITY_TIER"] = pd.qcut(df["INTL_PCT"], 3, labels=[
        "Low\n(<10% International)", 
        "Medium\n(10-25% International)", 
        "High\n(>25% International)"
    ])
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # 2. UPDATE THE PALETTE TO MATCH THE REPORT COLORS:
    sns.barplot(
        data=df, 
        x="DIVERSITY_TIER", 
        y="COMPOSITE_SCORE", 
        palette=["#43537a", "#2f8278", "#6cba6e"],
        edgecolor="white", 
        errorbar=None, 
        ax=ax
    )
    
    ax.set_title("Team Success by Cultural Diversity Tier", fontsize=12, fontweight="bold", pad=15)
    ax.set_xlabel("International Roster Concentration")
    ax.set_ylabel("Average Composite Success Score")
    ax.axhline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.4)
    fig.tight_layout()
    
    fig.savefig(os.path.join(FIG, "q2_fig7_diversity_vs_success.png"), dpi=150) 
    plt.close()
    
    print("Complete! Check figures 6 and 7.")

if __name__ == "__main__":
    main()