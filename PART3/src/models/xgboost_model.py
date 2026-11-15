"""XGBoost regression and classification reference models with early stopping and hyperparameter tuning.
Tuned strictly on the validation set; native feature importance computed via Gain.
"""
from __future__ import annotations
import itertools
from pathlib import Path
import numpy as np
import xgboost as xgb

from src.models.base import BaseModel
from src.config import XGBOOST_GRID, XGBOOST_FIXED


class _XGBBase(BaseModel):
    framework: str = "xgboost"
    implementation: str = "native"

    def _make(self, **params):
        common = dict(
            XGBOOST_FIXED,
            random_state=self.seed,
            early_stopping_rounds=50,
            **params,
        )
        if self.task == "regression":
            return xgb.XGBRegressor(objective="reg:squarederror", eval_metric="rmse", **common)
        return xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)

    def _fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        if self.task == "classification":
            y_train, y_val = y_train.astype(int), y_val.astype(int)

        best = None
        for depth, lr in itertools.product(XGBOOST_GRID["max_depth"], XGBOOST_GRID["learning_rate"]):
            m = self._make(max_depth=depth, learning_rate=lr)
            m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            score = float(m.best_score)
            if best is None or score < best[0]:
                best = (
                    score,
                    m,
                    {
                        "max_depth": depth,
                        "learning_rate": lr,
                        "best_iteration": int(m.best_iteration),
                        "val_score": score,
                    },
                )

        _, self.model_, self.best_params_ = best
        self.epochs_trained_ = self.best_params_["best_iteration"] + 1
        self.n_params_ = len(self.model_.get_booster().get_dump())
        imp = self.model_.feature_importances_
        self.native_importance_ = dict(zip(self.feature_names, [float(v) for v in imp]))

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        bi = self.best_params_.get("best_iteration")
        kw = {"iteration_range": (0, int(bi) + 1)} if bi is not None else {}
        if self.task == "regression":
            return self.model_.predict(X, **kw)
        return self.model_.predict_proba(X, **kw)[:, 1]

    def _save(self, directory: Path):
        self.model_.save_model(str(directory / "model.json"))

    def _load(self, directory: Path):
        self.model_ = xgb.XGBRegressor() if self.task == "regression" else xgb.XGBClassifier()
        self.model_.load_model(str(directory / "model.json"))


class XGBoostRegressorModel(_XGBBase):
    task: str = "regression"


class XGBoostClassifierModel(_XGBBase):
    task: str = "classification"
