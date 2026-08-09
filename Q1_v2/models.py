"""
Every model wraps sklearn/xgb/lgbm/catboost in a consistent interface:
  fit(X_train, y_train) -> self
  predict(X_test)       -> np.ndarray
  name                  -> str label for plots

This makes experiments.py trivial: loop over MODEL_REGISTRY, call fit/predict.
"""

from __future__ import annotations
import time
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy         import DummyRegressor
from sklearn.linear_model  import Ridge, Lasso, ElasticNet
from sklearn.ensemble      import (
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    import catboost as cb
    CB_AVAILABLE = True
except ImportError:
    CB_AVAILABLE = False

from config import RANDOM_SEED, XGBConfig, LGBMConfig, RidgeConfig, RFConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Container for a single model's fit/predict results."""
    name:          str
    mae:           float
    rmse:          float
    r2:            float
    train_time_s:  float
    predict_time_s: float
    predictions:   np.ndarray
    feature_importance: dict[str, float] = field(default_factory=dict)


class _BaseModel:
    """Shared interface for all models."""
    name: str = "base"
    needs_scaling: bool = False

    def __init__(self):
        self._model    = None
        self._scaler   = StandardScaler() if self.needs_scaling else None
        self.train_time: float = 0.0
        self.pred_time:  float = 0.0

    def _scale(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if self._scaler is None:
            return X
        return self._scaler.fit_transform(X) if fit else self._scaler.transform(X)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_BaseModel":
        t0 = time.perf_counter()
        X_in = self._scale(X, fit=True)
        # LightGBM needs a DataFrame with column names to avoid the
        # "X does not have valid feature names" warning on predict
        self._model.fit(X_in, y)
        self.train_time = time.perf_counter() - t0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        X_in = self._scale(X)
        preds = self._model.predict(X_in)
        self.pred_time = time.perf_counter() - t0
        return preds.flatten()

    def feature_importances(self, feature_names: list[str]) -> dict[str, float]:
        """Return feature importance dict if available, else empty."""
        model = self._model
        if hasattr(model, "feature_importances_"):
            return dict(zip(feature_names, model.feature_importances_))
        if hasattr(model, "coef_"):
            return dict(zip(feature_names, np.abs(model.coef_)))
        return {}


#  Concrete models

class NaiveBaselineModel(_BaseModel):
    """Predict the player's season-to-date average. No ML, just a reference point."""
    name = "Naive Baseline"

    def __init__(self):
        super().__init__()
        self._model = DummyRegressor(strategy="mean")


class RidgeModel(_BaseModel):
    name = "Ridge"
    needs_scaling = True

    def __init__(self, cfg: RidgeConfig = RidgeConfig()):
        super().__init__()
        self._model = Ridge(alpha=cfg.alpha, random_state=RANDOM_SEED)


class LassoModel(_BaseModel):
    name = "Lasso"
    needs_scaling = True

    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self._model = Lasso(alpha=alpha, random_state=RANDOM_SEED, max_iter=5000)


class ElasticNetModel(_BaseModel):
    name = "ElasticNet"
    needs_scaling = True

    def __init__(self, alpha: float = 0.1, l1_ratio: float = 0.5):
        super().__init__()
        self._model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                                  random_state=RANDOM_SEED, max_iter=5000)


class RandomForestModel(_BaseModel):
    name = "Random Forest"

    def __init__(self, cfg: RFConfig = RFConfig()):
        super().__init__()
        self._model = RandomForestRegressor(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            min_samples_leaf=cfg.min_samples_leaf,
            n_jobs=cfg.n_jobs,
            random_state=RANDOM_SEED,
        )


class ExtraTreesModel(_BaseModel):
    name = "Extra Trees"

    def __init__(self):
        super().__init__()
        self._model = ExtraTreesRegressor(
            n_estimators=300, max_depth=10, min_samples_leaf=5,
            n_jobs=-1, random_state=RANDOM_SEED,
        )


class GradientBoostingModel(_BaseModel):
    name = "Gradient Boosting"

    def __init__(self):
        super().__init__()
        self._model = GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, random_state=RANDOM_SEED,
        )


class XGBoostModel(_BaseModel):
    name = "XGBoost"

    def __init__(self, cfg: XGBConfig = XGBConfig()):
        super().__init__()
        self._cfg = cfg
        self._model = xgb.XGBRegressor(
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            max_depth=cfg.max_depth,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            min_child_weight=cfg.min_child_weight,
            gamma=cfg.gamma,
            reg_alpha=cfg.reg_alpha,
            reg_lambda=cfg.reg_lambda,
            random_state=RANDOM_SEED,
            verbosity=0,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None) -> "XGBoostModel":
        t0 = time.perf_counter()
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self._model.fit(X, y, eval_set=eval_set, verbose=False)
        self.train_time = time.perf_counter() - t0
        return self


class LightGBMModel(_BaseModel):
    name = "LightGBM"

    def __init__(self, cfg: LGBMConfig = LGBMConfig()):
        super().__init__()
        if not LGBM_AVAILABLE:
            raise ImportError("lightgbm not installed. pip install lightgbm")
        self._model = lgb.LGBMRegressor(
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            num_leaves=cfg.num_leaves,
            min_child_samples=cfg.min_child_samples,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            reg_alpha=cfg.reg_alpha,
            reg_lambda=cfg.reg_lambda,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1,
        )
        self._feature_names: list[str] | None = None

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: list[str] | None = None) -> "LightGBMModel":
        import pandas as pd
        t0 = time.perf_counter()
        # Wrap in DataFrame so LightGBM stores feature names and won't warn on predict
        if feature_names is not None:
            self._feature_names = feature_names
        X_df = pd.DataFrame(X, columns=self._feature_names) if self._feature_names else X
        self._model.fit(X_df, y)
        self.train_time = time.perf_counter() - t0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import pandas as pd
        t0 = time.perf_counter()
        X_df = pd.DataFrame(X, columns=self._feature_names) if self._feature_names else X
        preds = self._model.predict(X_df)
        self.pred_time = time.perf_counter() - t0
        return preds.flatten()


class CatBoostModel(_BaseModel):
    name = "CatBoost"

    def __init__(self):
        super().__init__()
        if not CB_AVAILABLE:
            raise ImportError("catboost not installed. pip install catboost")
        self._model = cb.CatBoostRegressor(
            iterations=400, learning_rate=0.04, depth=6,
            random_seed=RANDOM_SEED, verbose=0,
            loss_function="MAE",
        )


# Registry: all models available for experiments
def get_model_registry() -> dict[str, _BaseModel]:
    """
    Return all available models.  Models are instantiated lazily here
    so import errors for optional packages surface at construction, not import.
    """
    registry: dict[str, _BaseModel] = {
        "naive_baseline":   NaiveBaselineModel(),
        "ridge":            RidgeModel(),
        "lasso":            LassoModel(),
        "elasticnet":       ElasticNetModel(),
        "random_forest":    RandomForestModel(),
        "extra_trees":      ExtraTreesModel(),
        "gradient_boosting":GradientBoostingModel(),
        "xgboost":          XGBoostModel(),
    }
    if LGBM_AVAILABLE:
        registry["lightgbm"] = LightGBMModel()
    if CB_AVAILABLE:
        registry["catboost"] = CatBoostModel()
    return registry
