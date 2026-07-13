"""
config.py
=========
Single source of truth for all paths, constants, and hyperparameters.
Import this everywhere — never hard-code paths or magic numbers elsewhere.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict

# ── Repo root (two levels up from Q1_v2/) ────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_ROOT     = REPO_ROOT / "data" / "processed"
NBA_API_DIR   = DATA_ROOT / "NBA_api"
KAGGLE_DIR    = DATA_ROOT / "Kaggle"
BREF_DIR      = DATA_ROOT / "BRef"

# ── Output paths ──────────────────────────────────────────────────────────────
Q1_DIR        = REPO_ROOT / "Q1_v2"
FIGURES_DIR   = REPO_ROOT / "figures" / "Q1"
RESULTS_DIR   = Q1_DIR / "results"

for _p in [FIGURES_DIR, RESULTS_DIR]:
    _p.mkdir(parents=True, exist_ok=True)

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Seasons covered ───────────────────────────────────────────────────────────
SEASONS = [
    "2019-20", "2020-21", "2021-22",
    "2022-23", "2023-24", "2024-25",
]

# ── Prediction targets ────────────────────────────────────────────────────────
PRIMARY_TARGET = "PTS"        # next-game points
ALL_TARGETS    = ["PTS", "REB", "AST"]

# ── Train / test split ────────────────────────────────────────────────────────
TRAIN_RATIO   = 0.80          # chronological split
MIN_GAMES     = 10            # minimum games played before a player enters the dataset

# ── Rolling window sizes ──────────────────────────────────────────────────────
WINDOWS = [3, 5, 10]

# ── Feature groups (used for ablation experiments) ───────────────────────────
FEATURE_GROUPS: Dict[str, List[str]] = {
    "player_form": [
        "PTS_roll3", "PTS_roll5", "PTS_roll10",
        "PTS_ewma5", "PTS_std5", "PTS_trend5",
        "MIN_roll3", "MIN_roll5",
        "REB_roll3", "AST_roll3", "TOV_roll3",
        "FG_PCT_roll3", "FG3_PCT_roll3", "FT_PCT_roll3",
        "eFG_roll3",
        "USG_roll3",
        "season_avg_pts", "season_avg_min",
        "pts_volatility", "consistency_score",
        "hot_streak", "cold_streak",
        "pts_vs_season_avg",
    ],
    "schedule": [
        "is_home",
        "rest_days",
        "is_back_to_back",
        "is_3_in_4",
        "away_streak",
        "home_streak",
        "games_in_last_7",
    ],
    "opponent": [
        "opp_def_rating_roll5",
        "opp_pace_roll5",
        "opp_eFG_allowed_roll5",
        "opp_reb_pct_roll5",
        "opp_win_pct_roll5",
        "opp_pts_allowed_roll5",   # rolling pts the opponent allows per game
    ],
    "team_context": [
        "team_off_rating_roll5",
        "team_def_rating_roll5",
        "team_net_rating_roll5",
        "team_pace_roll5",
        "team_ast_pct_roll5",
    ],
    "career": [
        "player_age",
        "experience_years",
        "career_avg_pts",
        "career_avg_min",
        "career_efg",
    ],
}

ALL_FEATURES: List[str] = [
    f for group in FEATURE_GROUPS.values() for f in group
]

# ── Model hyperparameters ─────────────────────────────────────────────────────
@dataclass
class XGBConfig:
    n_estimators:      int   = 500
    learning_rate:     float = 0.04
    max_depth:         int   = 5
    subsample:         float = 0.8
    colsample_bytree:  float = 0.8
    min_child_weight:  int   = 5
    gamma:             float = 0.1
    reg_alpha:         float = 0.1
    reg_lambda:        float = 1.0

@dataclass
class LGBMConfig:
    n_estimators:    int   = 500
    learning_rate:   float = 0.04
    num_leaves:      int   = 63
    min_child_samples: int = 20
    subsample:       float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha:       float = 0.1
    reg_lambda:      float = 1.0

@dataclass
class RidgeConfig:
    alpha: float = 10.0

@dataclass
class RFConfig:
    n_estimators: int   = 300
    max_depth:    int   = 10
    min_samples_leaf: int = 5
    n_jobs:       int   = -1

# ── Experiment definitions ────────────────────────────────────────────────────
# Each experiment adds a feature group on top of the previous one.
# This lets us measure the marginal value of each group.
EXPERIMENTS: Dict[str, List[str]] = {
    "EXP1_baseline":         ["player_form"],
    "EXP2_schedule":         ["player_form", "schedule"],
    "EXP3_opponent":         ["player_form", "schedule", "opponent"],
    "EXP4_team":             ["player_form", "schedule", "opponent", "team_context"],
    "EXP5_career":           ["player_form", "schedule", "opponent", "team_context", "career"],
    "EXP6_all":              list(FEATURE_GROUPS.keys()),
}

# ── Visualization style ───────────────────────────────────────────────────────
STYLE = {
    "blue":      "#1565C0",
    "light_blue":"#42A5F5",
    "red":       "#C62828",
    "green":     "#2E7D32",
    "orange":    "#E65100",
    "grey":      "#9E9E9E",
    "dark":      "#0d0d0d",
    "dpi":        200,
    "fig_ext":   ".png",
}
