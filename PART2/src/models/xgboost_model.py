"""XGBoost regression and classification. Small hyper-parameter grid selected on the VALIDATION set;
early stopping also uses validation. Native importance = gain."""
from __future__ import annotations
import itertools
from pathlib import Path
import numpy as np
from src.models.base import BaseModel

GRID = {"max_depth": [2, 3, 4], "learning_rate": [0.02, 0.05]}
FIXED = dict(n_estimators=2000, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=5.0,
             tree_method="hist", n_jobs=1, importance_type="gain")


class _XGB(BaseModel):
    framework = "xgboost"

    def _make(self, **params):
        import xgboost as xgb
        common = dict(FIXED, random_state=self.seed, early_stopping_rounds=50, **params)
        if self.task == "regression":
            return xgb.XGBRegressor(objective="reg:squarederror", eval_metric="rmse", **common)
        return xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)

    def _fit(self, X_train, y_train, X_val, y_val):
        if self.task == "classification":                      # XGBClassifier expects integer class labels
            y_train, y_val = y_train.astype(int), y_val.astype(int)
        best = None
        for depth, lr in itertools.product(GRID["max_depth"], GRID["learning_rate"]):
            m = self._make(max_depth=depth, learning_rate=lr)
            m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            score = float(m.best_score)                       # validation RMSE / logloss
            if best is None or score < best[0]:
                best = (score, m, {"max_depth": depth, "learning_rate": lr, "best_iteration": int(m.best_iteration),
                                   "val_score": score})
        _, self.model_, self.best_params_ = best
        self.epochs_trained_ = self.best_params_["best_iteration"] + 1      # boosting rounds kept
        self.n_params_ = len(self.model_.get_booster().get_dump())          # number of trees
        imp = self.model_.feature_importances_
        self.native_importance_ = dict(zip(self.feature_names, [float(v) for v in imp]))

    def _predict_raw(self, X):
        # Use exactly the early-stopped rounds (best_iteration) even after save/load, instead of relying on
        # booster attributes surviving serialisation.
        bi = self.best_params_.get("best_iteration")
        kw = {"iteration_range": (0, int(bi) + 1)} if bi is not None else {}
        if self.task == "regression":
            return self.model_.predict(X, **kw)
        return self.model_.predict_proba(X, **kw)[:, 1]

    def _save(self, d: Path):
        self.model_.save_model(str(d / "model.json"))

    def _load(self, d: Path):
        import xgboost as xgb
        self.model_ = (xgb.XGBRegressor() if self.task == "regression" else xgb.XGBClassifier())
        self.model_.load_model(str(d / "model.json"))


class XGBoostRegressorModel(_XGB):
    task = "regression"


class XGBoostClassifierModel(_XGB):
    task = "classification"


if __name__ == "__main__":
    from src.training import cli_main
    cli_main("xgboost")
