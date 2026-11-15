"""Ensemble Learning combining PyTorch TabNet and XGBoost.
Implements:
1. Optimal Validation Weighted Averaging (convex blending)
2. Stacking Meta-Learner (Ridge regression meta-model)
"""
from __future__ import annotations
from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

from src.models.base import BaseModel
from src.models.tabnet_model import TabNetRegressorModel, TabNetClassifierModel
from src.models.xgboost_model import XGBoostRegressorModel, XGBoostClassifierModel


class WeightedEnsembleRegressor(BaseModel):
    """Ensemble Regressor: convex combination of TabNet and XGBoost predictions.
    y_ens = w * y_tabnet + (1 - w) * y_xgboost
    Optimal weight w is determined via grid search on the validation set to minimize RMSE.
    """
    framework: str = "ensemble"
    task: str = "regression"
    implementation: str = "weighted_average"

    def __init__(self, seed: int = 42, **hp):
        super().__init__(seed, **hp)
        self.tabnet_ = None
        self.xgboost_ = None
        self.w_tabnet_: float = 0.5

    def fit_components(
        self,
        tabnet: TabNetRegressorModel,
        xgboost: XGBoostRegressorModel,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list,
    ):
        self.tabnet_ = tabnet
        self.xgboost_ = xgboost
        self.feature_names = list(feature_names)

        # Obtain validation predictions from both components
        p_tab_val = self.tabnet_.predict(X_val)
        p_xgb_val = self.xgboost_.predict(X_val)

        # Optimize weight w in [0, 1] on VALIDATION set
        weights = np.linspace(0.0, 1.0, 101)
        best_w = 0.5
        best_rmse = float("inf")
        loss_curve = []

        for w in weights:
            p_ens = w * p_tab_val + (1.0 - w) * p_xgb_val
            rmse = float(np.sqrt(np.mean((p_ens - y_val) ** 2)))
            loss_curve.append({"weight_tabnet": float(w), "val_rmse": rmse})
            if rmse < best_rmse:
                best_rmse = rmse
                best_w = float(w)

        self.w_tabnet_ = best_w
        self.best_params_ = {
            "w_tabnet": self.w_tabnet_,
            "w_xgboost": 1.0 - self.w_tabnet_,
            "val_rmse": best_rmse,
            "loss_curve": loss_curve,
        }
        self.n_params_ = (self.tabnet_.n_params_ or 0) + (self.xgboost_.n_params_ or 0)
        self.epochs_trained_ = max(self.tabnet_.epochs_trained_ or 0, self.xgboost_.epochs_trained_ or 0)

        # Combine native feature importances weighted by ensemble weights
        tab_imp = self.tabnet_.feature_importance()
        xgb_imp = self.xgboost_.feature_importance()
        if tab_imp is not None and xgb_imp is not None:
            combined = self.w_tabnet_ * tab_imp + (1.0 - self.w_tabnet_) * xgb_imp
            self.native_importance_ = combined.to_dict()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        p_tab = self.tabnet_.predict(X)
        p_xgb = self.xgboost_.predict(X)
        return self.w_tabnet_ * p_tab + (1.0 - self.w_tabnet_) * p_xgb

    def _save(self, directory: Path):
        self.tabnet_.save(directory / "tabnet_component")
        self.xgboost_.save(directory / "xgboost_component")
        config = {
            "w_tabnet": self.w_tabnet_,
            "best_params": self.best_params_,
            "feature_names": self.feature_names,
            "native_importance": self.native_importance_,
        }
        (directory / "ensemble_config.json").write_text(json.dumps(config, indent=2))

    def _load(self, directory: Path):
        self.tabnet_ = TabNetRegressorModel.load(directory / "tabnet_component")
        self.xgboost_ = XGBoostRegressorModel.load(directory / "xgboost_component")
        config = json.loads((directory / "ensemble_config.json").read_text())
        self.w_tabnet_ = config["w_tabnet"]
        self.best_params_ = config["best_params"]
        self.feature_names = config["feature_names"]
        self.native_importance_ = config.get("native_importance")


class WeightedEnsembleClassifier(BaseModel):
    """Ensemble Classifier: probability blending of TabNet and XGBoost.
    p_ens = w * p_tabnet + (1 - w) * p_xgboost
    Optimal weight w is determined via grid search on the validation set to maximize ROC-AUC.
    """
    framework: str = "ensemble"
    task: str = "classification"
    implementation: str = "probability_blend"

    def __init__(self, seed: int = 42, **hp):
        super().__init__(seed, **hp)
        self.tabnet_ = None
        self.xgboost_ = None
        self.w_tabnet_: float = 0.5

    def fit_components(
        self,
        tabnet: TabNetClassifierModel,
        xgboost: XGBoostClassifierModel,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list,
    ):
        self.tabnet_ = tabnet
        self.xgboost_ = xgboost
        self.feature_names = list(feature_names)

        p_tab_val = self.tabnet_.predict(X_val)
        p_xgb_val = self.xgboost_.predict(X_val)

        weights = np.linspace(0.0, 1.0, 101)
        best_w = 0.5
        best_auc = -1.0
        curve = []

        for w in weights:
            p_ens = w * p_tab_val + (1.0 - w) * p_xgb_val
            auc = float(roc_auc_score(y_val, p_ens))
            curve.append({"weight_tabnet": float(w), "val_roc_auc": auc})
            if auc > best_auc:
                best_auc = auc
                best_w = float(w)

        self.w_tabnet_ = best_w
        self.best_params_ = {
            "w_tabnet": self.w_tabnet_,
            "w_xgboost": 1.0 - self.w_tabnet_,
            "val_roc_auc": best_auc,
            "curve": curve,
        }
        self.n_params_ = (self.tabnet_.n_params_ or 0) + (self.xgboost_.n_params_ or 0)
        self.epochs_trained_ = max(self.tabnet_.epochs_trained_ or 0, self.xgboost_.epochs_trained_ or 0)

        tab_imp = self.tabnet_.feature_importance()
        xgb_imp = self.xgboost_.feature_importance()
        if tab_imp is not None and xgb_imp is not None:
            combined = self.w_tabnet_ * tab_imp + (1.0 - self.w_tabnet_) * xgb_imp
            self.native_importance_ = combined.to_dict()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        p_tab = self.tabnet_.predict(X)
        p_xgb = self.xgboost_.predict(X)
        return self.w_tabnet_ * p_tab + (1.0 - self.w_tabnet_) * p_xgb

    def _save(self, directory: Path):
        self.tabnet_.save(directory / "tabnet_component")
        self.xgboost_.save(directory / "xgboost_component")
        config = {
            "w_tabnet": self.w_tabnet_,
            "best_params": self.best_params_,
            "feature_names": self.feature_names,
            "native_importance": self.native_importance_,
        }
        (directory / "ensemble_config.json").write_text(json.dumps(config, indent=2))

    def _load(self, directory: Path):
        self.tabnet_ = TabNetClassifierModel.load(directory / "tabnet_component")
        self.xgboost_ = XGBoostClassifierModel.load(directory / "xgboost_component")
        config = json.loads((directory / "ensemble_config.json").read_text())
        self.w_tabnet_ = config["w_tabnet"]
        self.best_params_ = config["best_params"]
        self.feature_names = config["feature_names"]
        self.native_importance_ = config.get("native_importance")


class StackingEnsembleRegressor(BaseModel):
    """Stacking Meta-Learner Regressor:
    Uses a Ridge regression meta-model fit on the validation predictions of TabNet and XGBoost.
    """
    framework: str = "ensemble"
    task: str = "regression"
    implementation: str = "stacking_ridge"

    def __init__(self, seed: int = 42, **hp):
        super().__init__(seed, **hp)
        self.tabnet_ = None
        self.xgboost_ = None
        self.meta_model_ = None

    def fit_components(
        self,
        tabnet: TabNetRegressorModel,
        xgboost: XGBoostRegressorModel,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list,
    ):
        self.tabnet_ = tabnet
        self.xgboost_ = xgboost
        self.feature_names = list(feature_names)

        p_tab = self.tabnet_.predict(X_val).reshape(-1, 1)
        p_xgb = self.xgboost_.predict(X_val).reshape(-1, 1)
        meta_X = np.hstack([p_tab, p_xgb])

        self.meta_model_ = Ridge(alpha=1.0, random_state=self.seed)
        self.meta_model_.fit(meta_X, y_val)

        p_stack = self.meta_model_.predict(meta_X)
        val_rmse = float(np.sqrt(np.mean((p_stack - y_val) ** 2)))

        self.best_params_ = {
            "meta_coef": self.meta_model_.coef_.tolist(),
            "meta_intercept": float(self.meta_model_.intercept_),
            "val_rmse": val_rmse,
        }
        self.n_params_ = (self.tabnet_.n_params_ or 0) + (self.xgboost_.n_params_ or 0) + 3
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        p_tab = self.tabnet_.predict(X).reshape(-1, 1)
        p_xgb = self.xgboost_.predict(X).reshape(-1, 1)
        meta_X = np.hstack([p_tab, p_xgb])
        return self.meta_model_.predict(meta_X)

    def _save(self, directory: Path):
        self.tabnet_.save(directory / "tabnet_component")
        self.xgboost_.save(directory / "xgboost_component")
        joblib.dump(self.meta_model_, directory / "meta_model.joblib")
        (directory / "stacking_meta.json").write_text(json.dumps(self.best_params_, indent=2))

    def _load(self, directory: Path):
        self.tabnet_ = TabNetRegressorModel.load(directory / "tabnet_component")
        self.xgboost_ = XGBoostRegressorModel.load(directory / "xgboost_component")
        self.meta_model_ = joblib.load(directory / "meta_model.joblib")
        self.best_params_ = json.loads((directory / "stacking_meta.json").read_text())
