"""PyTorch TabNet model implementation for regression and binary classification.
TabNet incorporates sequential attentive feature selection at each decision step,
providing intrinsic explainability through step-wise attention masks.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier

from src.models.base import BaseModel
from src.config import TABNET_HP


class _TabNetBase(BaseModel):
    framework: str = "tabnet"
    implementation: str = "native"

    def _fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        hp = {**TABNET_HP, **self.hp}
        torch.manual_seed(self.seed)

        model_kwargs = dict(
            n_d=hp["n_d"],
            n_a=hp["n_a"],
            n_steps=hp["n_steps"],
            gamma=hp["gamma"],
            lambda_sparse=hp["lambda_sparse"],
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=hp["lr"]),
            seed=self.seed,
            verbose=0,
            device_name="cpu",
        )

        fit_kwargs = dict(
            max_epochs=hp["max_epochs"],
            patience=hp["patience"],
            batch_size=hp["batch_size"],
            virtual_batch_size=hp["virtual_batch_size"],
            drop_last=False,
        )

        if self.task == "regression":
            self.model_ = TabNetRegressor(**model_kwargs)
            y_train_fit = y_train.reshape(-1, 1).astype(np.float32)
            y_val_fit = y_val.reshape(-1, 1).astype(np.float32)
            self.model_.fit(
                X_train,
                y_train_fit,
                eval_set=[(X_val, y_val_fit)],
                eval_name=["val"],
                eval_metric=["rmse"],
                **fit_kwargs,
            )
        else:
            self.model_ = TabNetClassifier(**model_kwargs)
            self.model_.fit(
                X_train,
                y_train.astype(int),
                eval_set=[(X_val, y_val.astype(int))],
                eval_name=["val"],
                eval_metric=["logloss"],
                **fit_kwargs,
            )

        self.epochs_trained_ = int(len(self.model_.history["loss"]))
        self.n_params_ = int(sum(p.numel() for p in self.model_.network.parameters()))
        self.native_importance_ = dict(
            zip(self.feature_names, [float(v) for v in self.model_.feature_importances_])
        )
        self.best_params_ = {
            "n_d": hp["n_d"],
            "n_a": hp["n_a"],
            "n_steps": hp["n_steps"],
            "gamma": hp["gamma"],
            "lambda_sparse": hp["lambda_sparse"],
            "lr": hp["lr"],
            "epochs_trained": self.epochs_trained_,
        }

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        if self.task == "regression":
            return self.model_.predict(X).reshape(-1)
        return self.model_.predict_proba(X)[:, 1]

    def explain(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """Extract both the aggregate attention matrix and the step-by-step attention masks M[step]."""
        explain_matrix, masks = self.model_.explain(X)
        return explain_matrix, masks

    def _save(self, directory: Path):
        self.model_.save_model(str(directory / "tabnet_model"))

    def _load(self, directory: Path):
        zip_path = directory / "tabnet_model.zip"
        if not zip_path.exists():
            raise FileNotFoundError(f"TabNet model weights not found at {zip_path}")
        self.model_ = (
            TabNetRegressor(device_name="cpu")
            if self.task == "regression"
            else TabNetClassifier(device_name="cpu")
        )
        self.model_.load_model(str(zip_path))


class TabNetRegressorModel(_TabNetBase):
    task: str = "regression"


class TabNetClassifierModel(_TabNetBase):
    task: str = "classification"
