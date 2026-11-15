"""Decision Tree baseline models for regression and classification.
Serves as the traditional tree baseline (single CART tree) to evaluate the value added
by gradient boosting (XGBoost) and deep attention (TabNet).
"""
from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from src.models.base import BaseModel


class _DecisionTreeBase(BaseModel):
    framework: str = "decision_tree"
    implementation: str = "sklearn"

    def _fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        best_depth = None
        best_score = float("inf")
        best_model = None

        # Tune tree depth on validation set to prevent catastrophic overfitting
        depth_candidates = [2, 3, 4, 5, 6, 8]
        for depth in depth_candidates:
            if self.task == "regression":
                dt = DecisionTreeRegressor(max_depth=depth, random_state=self.seed)
                dt.fit(X_train, y_train)
                preds = dt.predict(X_val)
                score = float(np.mean((preds - y_val) ** 2))  # MSE
            else:
                dt = DecisionTreeClassifier(max_depth=depth, random_state=self.seed)
                dt.fit(X_train, y_train.astype(int))
                probs = np.clip(dt.predict_proba(X_val)[:, 1], 1e-6, 1 - 1e-6)
                score = float(-np.mean(y_val * np.log(probs) + (1 - y_val) * np.log(1 - probs)))  # Log-loss

            if score < best_score:
                best_score = score
                best_depth = depth
                best_model = dt

        self.model_ = best_model
        self.best_params_ = {"max_depth": best_depth, "val_loss": best_score}
        self.n_params_ = self.model_.tree_.node_count
        self.epochs_trained_ = 1
        imp = self.model_.feature_importances_
        self.native_importance_ = dict(zip(self.feature_names, [float(v) for v in imp]))

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        if self.task == "regression":
            return self.model_.predict(X)
        return self.model_.predict_proba(X)[:, 1]

    def _save(self, directory: Path):
        joblib.dump(self.model_, directory / "decision_tree.joblib")

    def _load(self, directory: Path):
        self.model_ = joblib.load(directory / "decision_tree.joblib")


class DecisionTreeRegressorModel(_DecisionTreeBase):
    task: str = "regression"


class DecisionTreeClassifierModel(_DecisionTreeBase):
    task: str = "classification"
