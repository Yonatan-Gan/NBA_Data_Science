"""
Q2 ML Pipeline - Predicting the "Chemistry Delta"
======================================================
Calculates a team's over/underperformance (Chemistry Delta) using 
Pythagorean Expectation, then trains a Random Forest model to determine 
which structural traits best predict that delta.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ============================================================
# DIRECTORIES & SETUP
# ============================================================
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_PROC = os.path.join(BASE, "data", "processed", "NBA_api")
SEASON_STATS_DIR = os.path.join(API_PROC, "season_stats")
BIO_FILE = os.path.join(API_PROC, "player_profiles", "player_bio_info.csv")
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.color": "#ebebeb",
})

def load_and_engineer_data():
    if not os.path.exists(BIO_FILE):
        raise FileNotFoundError(f"Cannot find bio file at {BIO_FILE}")

    bio_df = pd.read_csv(BIO_FILE)
    id_col = "PERSON_ID" if "PERSON_ID" in bio_df.columns else "PLAYER_ID"
    origins_dict = dict(zip(bio_df[id_col], bio_df["COUNTRY"]))

    seasons = [d for d in os.listdir(SEASON_STATS_DIR) if os.path.isdir(os.path.join(SEASON_STATS_DIR, d))]
    all_teams_features = []

    for season in sorted(seasons):
        team_adv_path = os.path.join(SEASON_STATS_DIR, season, "team_stats_regular_season_advanced.csv")
        player_usg_path = os.path.join(SEASON_STATS_DIR, season, "player_stats_regular_season_usage.csv")
        
        if not os.path.exists(team_adv_path) or not os.path.exists(player_usg_path): continue
            
        df_team_adv = pd.read_csv(team_adv_path)
        df_player_usg = pd.read_csv(player_usg_path)
        df_player_usg = df_player_usg[df_player_usg["GP"] >= 15] # Rotation players only
        
        # Demographic Features
        df_player_usg["COUNTRY"] = df_player_usg["PLAYER_ID"].map(origins_dict).fillna("USA")
        df_player_usg["IS_INTL"] = np.where(df_player_usg["COUNTRY"] != "USA", 1, 0)
        
        team_structural = df_player_usg.groupby("TEAM_ID").agg(
            INTL_PCT=("IS_INTL", "mean"),
            AGE_MEAN=("AGE", "mean"),
            AGE_VARIANCE=("AGE", "std"),
            USG_VARIANCE=("USG_PCT", "std")
        ).reset_index()
        
        # Merge with Advanced Team Stats
        season_merged = df_team_adv[[
            "TEAM_ID", "TEAM_NAME", "W_PCT", "OFF_RATING", "DEF_RATING", "AST_PCT", "PACE"
        ]].merge(team_structural, on="TEAM_ID", how="inner")
        
        all_teams_features.append(season_merged)

    df = pd.concat(all_teams_features, ignore_index=True).dropna()

    # Calculate Pythagorean Expected Win % (Exponent 13.91 is standard for NBA)
    exp = 13.91
    df["EXP_W_PCT"] = (df["OFF_RATING"]**exp) / ((df["OFF_RATING"]**exp) + (df["DEF_RATING"]**exp))
    
    # Target Variable: Over/Underperformance
    df["CHEMISTRY_DELTA"] = df["W_PCT"] - df["EXP_W_PCT"]
    
    return df

def train_and_plot_model(df):
    # Select structural traits, deliberately dropping direct performance metrics
    features = ["AGE_MEAN", "AGE_VARIANCE", "USG_VARIANCE", "INTL_PCT", "AST_PCT", "PACE"]
    
    X = df[features]
    y = df["CHEMISTRY_DELTA"]

    # Initialize a Random Forest model to capture non-linear relationships
    rf_model = RandomForestRegressor(n_estimators=500, max_depth=5, random_state=42)
    rf_model.fit(X, y)

    # Extract and format feature importances
    importances = rf_model.feature_importances_
    feature_names = [
        "Average Age (Maturity)",
        "Age Variance (Barbell Strategy)",
        "Usage Variance (Hierarchy)",
        "Intl % (Cultural Diversity)",
        "AST% (Democratic Ball Movement)",
        "Pace (Tempo)"
    ]

    importance_df = pd.DataFrame({
        "Trait": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importance_df, x="Importance", y="Trait", palette="mako", edgecolor="white", ax=ax)
    
    ax.set_title("Which Structural Traits Predict High Team Chemistry?", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Random Forest Feature Weight (Importance)", fontsize=11)
    ax.set_ylabel("")
    
    # Annotate exact weights
    for i, p in enumerate(ax.patches):
        ax.annotate(f"{p.get_width():.1%}", 
                    (p.get_width() + 0.005, p.get_y() + p.get_height() / 2.), 
                    va='center', fontsize=10, fontweight='bold', color='#333333')

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "q2_fig8_chemistry_trait_weights.png"), dpi=150)
    plt.close()

def main():
    print("Building Chemistry Target Variable using Pythagorean Expectation...")
    df = load_and_engineer_data()
    
    print("Training Random Forest to extract trait weights...")
    train_and_plot_model(df)
    
    print("Pipeline Complete! Feature weights plotted to figures/q2_fig8_chemistry_trait_weights.png")

if __name__ == "__main__":
    main()