"""
Runs two types of experiments:

  1. Feature ablation: fix model (XGBoost), vary feature groups
     -> quantifies marginal contribution of each context layer

  2. Model comparison: fix features (all), vary models
     -> identifies the best model for this task

Both use a strict chronological train/test split.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import EXPERIMENTS, FEATURE_GROUPS, PRIMARY_TARGET, TRAIN_RATIO, RANDOM_SEED
from models import XGBoostModel, ModelResult, get_model_registry, _BaseModel

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Full results for one experiment (one feature set + one model)."""
    experiment_name: str
    model_name:      str
    feature_groups:  list[str]
    n_features:      int
    n_train:         int
    n_test:          int
    mae:             float
    rmse:            float
    r2:              float
    train_time_s:    float
    predictions:     np.ndarray
    actuals:         np.ndarray
    feature_names:   list[str]
    feature_importance: dict[str, float] = field(default_factory=dict)
    model_object:    any = field(default=None, repr=False)  # fitted model for SHAP


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sort by GAME_DATE and split at train_ratio.
    Chronological split is critical for time-series data - a random split
    would leak future information into training.
    """
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    split = int(len(df) * train_ratio)
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def _get_features(df: pd.DataFrame, groups: list[str]) -> list[str]:
    """
    Resolve feature names for given groups, keeping only columns
    that actually exist in df.
    """
    requested = [f for g in groups for f in FEATURE_GROUPS.get(g, [])]
    available = [f for f in requested if f in df.columns]
    missing   = set(requested) - set(available)
    if missing:
        logger.debug(f"  Features requested but not in df: {missing}")
    return available


def _run_one(
    df_train:      pd.DataFrame,
    df_test:       pd.DataFrame,
    features:      list[str],
    model:         _BaseModel,
    exp_name:      str,
    feature_groups: list[str],
) -> ExperimentResult:
    """Fit one model on one feature set, return metrics and predictions."""
    target = PRIMARY_TARGET

    X_tr = df_train[features].fillna(0).values
    y_tr = df_train[target].values
    X_te = df_test[features].fillna(0).values
    y_te = df_test[target].values

    # Fit (pass validation set to XGBoost for early stopping if available)
    if isinstance(model, XGBoostModel):
        model.fit(X_tr, y_tr, X_te, y_te)
    elif hasattr(model, '_feature_names'):   # LightGBM
        model.fit(X_tr, y_tr, feature_names=features)
    else:
        model.fit(X_tr, y_tr)

    preds = model.predict(X_te)

    mae  = mean_absolute_error(y_te, preds)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    r2   = r2_score(y_te, preds)
    fimp = model.feature_importances(features)

    logger.info(f"  {exp_name:<35} | {model.name:<18} | "
                f"MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}  "
                f"[{model.train_time:.1f}s train]")

    return ExperimentResult(
        experiment_name=exp_name,
        model_name=model.name,
        feature_groups=feature_groups,
        n_features=len(features),
        n_train=len(df_train),
        n_test=len(df_test),
        mae=mae,
        rmse=rmse,
        r2=r2,
        train_time_s=model.train_time,
        predictions=preds,
        actuals=y_te,
        feature_names=features,
        feature_importance=fimp,
        model_object=model,   # store fitted model for SHAP / downstream use
    )


class ExperimentRunner:
    """Orchestrates ablation + model comparison experiments."""

    def __init__(self, df: pd.DataFrame):
        if "GAME_DATE" not in df.columns:
            raise ValueError("df must contain GAME_DATE for chronological split")
        self.df_train, self.df_test = chronological_split(df)
        logger.info(f"Split: {len(self.df_train):,} train, {len(self.df_test):,} test")
        self.ablation_results: list[ExperimentResult] = []
        self.model_results:    list[ExperimentResult] = []

    # Ablation study

    def run_ablation(self) -> list[ExperimentResult]:
        """
        Run 6 experiments (EXP1-EXP6) using XGBoost.
        Each experiment adds one feature group to the previous.
        Quantifies: what does each context layer contribute?
        """
        logger.info("\n" + "─"*60)
        logger.info("  ABLATION STUDY (XGBoost, varying feature groups)")
        logger.info("─"*60)

        results = []
        for exp_name, groups in EXPERIMENTS.items():
            features = _get_features(self.df_train, groups)
            if not features:
                logger.warning(f"  {exp_name}: no features found - skipping")
                continue
            model = XGBoostModel()  # fresh model for each experiment
            res   = _run_one(
                self.df_train, self.df_test, features,
                model, exp_name, groups,
            )
            results.append(res)

        self.ablation_results = results
        return results

    # Model comparison

    def run_model_comparison(
        self,
        feature_groups: Optional[list[str]] = None,
        model_names:    Optional[list[str]] = None,
    ) -> list[ExperimentResult]:
        """
        Run all models on a fixed feature set (default: all features).
        Quantifies: which model architecture is best for this data?
        """
        logger.info("\n" + "─"*60)
        logger.info("  MODEL COMPARISON (all features)")
        logger.info("─"*60)

        groups   = feature_groups or list(FEATURE_GROUPS.keys())
        features = _get_features(self.df_train, groups)
        registry = get_model_registry()

        if model_names:
            registry = {k: v for k, v in registry.items() if k in model_names}

        results = []
        for mkey, model in registry.items():
            try:
                res = _run_one(
                    self.df_train, self.df_test, features,
                    model, f"model_{mkey}", groups,
                )
                results.append(res)
            except Exception as exc:
                logger.warning(f"  {mkey} failed: {exc}")

        self.model_results = results
        return results

    # Summary

    def summary_df(self, results: list[ExperimentResult]) -> pd.DataFrame:
        """Convert results list to a tidy DataFrame for reporting."""
        rows = []
        for r in results:
            rows.append({
                "experiment": r.experiment_name,
                "model":      r.model_name,
                "n_features": r.n_features,
                "mae":        round(r.mae,  3),
                "rmse":       round(r.rmse, 3),
                "r2":         round(r.r2,   3),
                "train_time_s": round(r.train_time_s, 1),
            })
        return pd.DataFrame(rows)
