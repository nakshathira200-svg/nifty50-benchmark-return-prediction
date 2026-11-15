"""Abstract Base Model defining a consistent interface across all frameworks and ensembles.
Handles target standardization for regression and probability formatting for classification.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


class BaseModel:
    framework: str = "base"
    task: str = "regression"  # 'regression' | 'classification'
    implementation: str = "native"

    def __init__(self, seed: int = 42, **hp):
        self.seed = seed
        self.hp = hp
        self.feature_names: List[str] = []
        self.y_mean_: float = 0.0
        self.y_std_: float = 1.0
        self.native_importance_: Optional[Dict[str, float]] = None
        self.best_params_: dict = {}
        self.n_params_: Optional[int] = None
        self.epochs_trained_: Optional[int] = None
        self.scaler = None

    def _fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        raise NotImplementedError

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _save(self, directory: Path):
        raise NotImplementedError

    def _load(self, directory: Path):
        raise NotImplementedError

    def attach_context(self, scaler=None):
        self.scaler = scaler

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, feature_names: List[str]):
        self.feature_names = list(feature_names)
        y_train = np.asarray(y_train, dtype=float)
        y_val = np.asarray(y_val, dtype=float)

        if self.task == "regression":
            # Normalizing tiny financial returns (~1e-3) prevents gradient collapse in neural networks
            self.y_mean_ = float(y_train.mean())
            self.y_std_ = float(y_train.std() if y_train.std() > 0 else 1.0)
            y_train = (y_train - self.y_mean_) / self.y_std_
            y_val = (y_val - self.y_mean_) / self.y_std_

        self._fit(np.asarray(X_train, dtype=np.float32), y_train, np.asarray(X_val, dtype=np.float32), y_val)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw = np.asarray(self._predict_raw(np.asarray(X, dtype=np.float32)), dtype=float).reshape(-1)
        if self.task == "regression":
            return raw * self.y_std_ + self.y_mean_
        return raw

    def feature_importance(self) -> Optional[pd.Series]:
        if not self.native_importance_:
            return None
        s = pd.Series(self.native_importance_).reindex(self.feature_names).fillna(0.0)
        total = s.sum()
        return s / total if total > 0 else s

    def meta(self) -> dict:
        return {
            "framework": self.framework,
            "task": self.task,
            "implementation": self.implementation,
            "seed": self.seed,
            "feature_names": self.feature_names,
            "y_mean": self.y_mean_,
            "y_std": self.y_std_,
            "native_importance": self.native_importance_,
            "best_params": self.best_params_,
            "n_parameters": self.n_params_,
            "epochs_trained": self.epochs_trained_,
            "hp": self._jsonable(self.hp),
        }

    def save(self, directory: Path | str) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self._save(d)
        (d / "meta.json").write_text(json.dumps(self.meta(), indent=2, default=str))
        return d

    @classmethod
    def load(cls, directory: Path | str):
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text())
        obj = cls(seed=meta["seed"], **meta.get("hp", {}))
        obj.feature_names = meta["feature_names"]
        obj.y_mean_ = meta["y_mean"]
        obj.y_std_ = meta["y_std"]
        obj.native_importance_ = meta.get("native_importance")
        obj.best_params_ = meta.get("best_params", {})
        obj.n_params_ = meta.get("n_parameters")
        obj.epochs_trained_ = meta.get("epochs_trained")
        obj.implementation = meta.get("implementation", cls.implementation)
        obj._load(d)
        return obj

    @staticmethod
    def _jsonable(d: dict) -> dict:
        out = {}
        for k, v in d.items():
            try:
                json.dumps(v)
                out[k] = v
            except (TypeError, OverflowError):
                out[k] = str(v)
        return out
