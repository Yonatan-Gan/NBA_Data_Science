"""
main.py
=======
Orchestrates the entire Q1 pipeline:

  1. Load data        (data_loader.py)
  2. Preprocess       (preprocessing.py)
  3. Feature engineer (feature_engineering.py)
  4. Run experiments  (experiments.py)
  5. Statistical tests(statistics.py)
  6. Visualize        (visualization.py)

Run from the repo root:
    cd NBA_Data_Science
    python Q1_v2/main.py

Or from inside Q1_v2/:
    python main.py
"""

from __future__ import annotations
import logging
import sys
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Add Q1_v2 to sys.path so relative imports work ───────────────────────────
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(THIS_DIR / "results" / "run.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

import config as cfg
from data_loader       import NBADataLoader
from preprocessing     import GameLogPreprocessor
from feature_engineering import FeatureEngineer
from experiments       import ExperimentRunner
from statistics        import StatisticalAnalyzer
from visualization     import (
    fig01_ablation,
    fig02_model_comparison,
    fig03_actual_vs_predicted,
    fig04_shap_summary,
    fig05_context_effects,
    fig06_error_by_subgroup,
    fig07_statistical_tests,
    fig08_rolling_prediction_examples,
)


def main() -> None:
    t_start = time.perf_counter()
    logger.info("=" * 60)
    logger.info("  Q1 — Predicting NBA Player Next-Game Performance")
    logger.info("  Understanding and Predicting NBA Player Performance")
    logger.info("  Using Context-Aware Machine Learning")
    logger.info("=" * 60)

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 1: LOAD
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n[1/6] Loading data...")
    loader  = NBADataLoader()
    data    = loader.load_all()

    player_logs  = data["player_gamelogs"]    # required
    team_logs    = data["team_gamelogs"]       # optional
    player_bio   = data["player_bio"]          # optional
    career_stats = data["career_stats"]        # optional

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 2: PREPROCESS
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n[2/6] Preprocessing...")
    prep        = GameLogPreprocessor()
    clean_logs  = prep.clean(player_logs)

    team_clean  = prep.clean_team_logs(team_logs) if team_logs is not None else None

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 3: FEATURE ENGINEERING
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n[3/6] Feature engineering...")
    fe = FeatureEngineer(
        player_logs  = clean_logs,
        team_logs    = team_clean,
        player_bio   = player_bio,
        career_stats = career_stats,
    )
    df_features = fe.build()   # builds all groups

    # Drop rows still missing target or any required feature
    df_features = df_features.dropna(subset=["PTS"]).copy()
    logger.info(f"  Final modelling dataset: {len(df_features):,} rows, "
                f"{df_features['PLAYER_ID'].nunique()} players")

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 4: EXPERIMENTS
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n[4/6] Running experiments...")
    runner = ExperimentRunner(df_features)

    # 4a. Ablation: XGBoost with each feature group layer
    ablation_results = runner.run_ablation()

    # 4b. Model comparison: all models on full feature set
    model_results = runner.run_model_comparison()

    # Save summary tables
    abl_df = runner.summary_df(ablation_results)
    mod_df = runner.summary_df(model_results)
    abl_df.to_csv(cfg.RESULTS_DIR / "ablation_results.csv", index=False)
    mod_df.to_csv(cfg.RESULTS_DIR / "model_comparison.csv", index=False)
    logger.info(f"\n  Ablation results:\n{abl_df.to_string(index=False)}")
    logger.info(f"\n  Model comparison:\n{mod_df.to_string(index=False)}")

    # Best model and its predictions for downstream figures
    best = min(model_results, key=lambda r: r.mae)
    logger.info(f"\n  Best model: {best.model_name}  MAE={best.mae:.3f}  R²={best.r2:.3f}")

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 5: STATISTICAL TESTS
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n[5/6] Statistical analysis...")
    analyzer  = StatisticalAnalyzer(runner.df_test, best.predictions)
    stats_df  = analyzer.run_all()
    stats_df.to_csv(cfg.RESULTS_DIR / "statistical_tests.csv", index=False)
    logger.info(f"\n  Statistical tests:\n{stats_df[['Question','p-value','Significant']].to_string(index=False)}")

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 6: VISUALIZATIONS
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n[6/6] Generating figures...")

    fig01_ablation(ablation_results)
    fig02_model_comparison(model_results)
    fig03_actual_vs_predicted(
        best.actuals, best.predictions,
        best.model_name, best.mae, best.r2,
    )

    # SHAP — use the stored model_object from ExperimentResult (no re-fitting needed)
    SHAP_SUPPORTED = {"XGBoost", "LightGBM", "Random Forest", "Extra Trees", "CatBoost",
                      "Gradient Boosting"}
    if best.model_name in SHAP_SUPPORTED and best.model_object is not None:
        try:
            import shap
            features = best.feature_names
            X_sample = (
                runner.df_test[features]
                .fillna(0)
                .sample(min(2000, len(runner.df_test)), random_state=cfg.RANDOM_SEED)
            )
            fitted_model = best.model_object._model   # unwrap _BaseModel wrapper

            # TreeExplainer works for XGBoost, LightGBM, RF, ET, CatBoost, GB
            explainer = shap.TreeExplainer(fitted_model)
            shap_vals = explainer.shap_values(X_sample.values)

            fig04_shap_summary(shap_vals, X_sample, features)
            logger.info("  ✓  SHAP figure generated")
        except Exception as exc:
            logger.warning(f"  SHAP figure skipped: {exc}")
    else:
        logger.warning(
            f"  SHAP skipped: {best.model_name} not in supported set or model_object is None"
        )

    fig05_context_effects(runner.df_test, best.predictions)
    fig06_error_by_subgroup(runner.df_test, best.predictions)
    fig07_statistical_tests(stats_df)
    fig08_rolling_prediction_examples(runner.df_test, best.predictions)

    # ══════════════════════════════════════════════════════════════════════════
    #  DONE
    # ══════════════════════════════════════════════════════════════════════════
    elapsed = time.perf_counter() - t_start
    logger.info(f"\n{'='*60}")
    logger.info(f"  ✅  Pipeline complete in {elapsed:.1f}s")
    logger.info(f"  Figures  → {cfg.FIGURES_DIR}/")
    logger.info(f"  Results  → {cfg.RESULTS_DIR}/")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()
