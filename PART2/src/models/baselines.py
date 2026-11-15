"""Baseline models (sklearn / rule based). These are cheap, transparent reference points: a complex model that
cannot beat them has not learned anything useful.

Regression : train_mean, zero (predict 0), persistence (tomorrow = today's return), ridge
Classification : majority_class, persistence (tomorrow's direction = today's), logistic_regression
"""
from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from src.models.base import BaseModel


class Baseline(BaseModel):
    framework = "baseline"
    kind = "train_mean"

    def __init__(self, seed: int = 42, kind: str = None, **hp):
        super().__init__(seed, **hp)
        self.kind = kind or self.kind
        self.est_ = None
        self.const_ = 0.0
        self._ret_mean = self._ret_scale = None

    # un-scaled last-day return (the `daily_return` feature) for persistence baselines
    def _last_return(self, X):
        j = self.feature_names.index("daily_return")
        return X[:, j] * self._ret_scale + self._ret_mean

    def _fit(self, X_train, y_train, X_val, y_val):
        if self.kind in ("persistence",) and self.scaler is not None:
            j = self.feature_names.index("daily_return")
            self._ret_mean, self._ret_scale = float(self.scaler.mean_[j]), float(self.scaler.scale_[j])
        if self.task == "regression":
            if self.kind == "train_mean":
                self.const_ = float(y_train.mean())
            elif self.kind == "zero":
                self.const_ = -self.y_mean_ / self.y_std_          # maps to 0 after un-standardising
            elif self.kind == "ridge":
                scores = {}
                for a in [0.1, 1, 10, 100, 1000, 10000]:       # alpha chosen on VALIDATION MSE
                    m = Ridge(alpha=a).fit(X_train, y_train)
                    scores[a] = float(np.mean((m.predict(X_val) - y_val) ** 2))
                alpha = min(scores, key=scores.get)
                self.best_params_ = {"alpha": alpha}
                self.est_ = Ridge(alpha=alpha).fit(X_train, y_train)
                self.native_importance_ = dict(zip(self.feature_names, np.abs(self.est_.coef_).tolist()))
        else:
            if self.kind == "majority_class":
                self.const_ = float(y_train.mean() >= 0.5)
            elif self.kind == "logistic_regression":
                best = None
                for c in [0.001, 0.01, 0.1, 1, 10]:             # C chosen on VALIDATION log-loss
                    m = LogisticRegression(C=c, max_iter=2000, random_state=self.seed).fit(X_train, y_train)
                    p = np.clip(m.predict_proba(X_val)[:, 1], 1e-6, 1 - 1e-6)
                    ll = float(-np.mean(y_val * np.log(p) + (1 - y_val) * np.log(1 - p)))
                    if best is None or ll < best[0]:
                        best = (ll, c)
                self.best_params_ = {"C": best[1]}
                self.est_ = LogisticRegression(C=best[1], max_iter=2000, random_state=self.seed).fit(X_train, y_train)
                self.native_importance_ = dict(zip(self.feature_names, np.abs(self.est_.coef_[0]).tolist()))
        self.n_params_ = (len(self.feature_names) + 1) if self.est_ is not None else 1

    def _predict_raw(self, X):
        n = len(X)
        if self.kind == "persistence":
            r = self._last_return(X)
            # regression output is returned in standardised space; BaseModel.predict() un-standardises it
            return (r - self.y_mean_) / self.y_std_ if self.task == "regression" else (r > 0).astype(float)
        if self.est_ is None:
            return np.full(n, self.const_)
        return self.est_.predict(X) if self.task == "regression" else self.est_.predict_proba(X)[:, 1]

    def _save(self, d: Path):
        joblib.dump({"kind": self.kind, "est": self.est_, "const": self.const_,
                     "rm": self._ret_mean, "rs": self._ret_scale}, d / "model.joblib")

    def _load(self, d: Path):
        s = joblib.load(d / "model.joblib")
        self.kind, self.est_, self.const_, self._ret_mean, self._ret_scale = s["kind"], s["est"], s["const"], s["rm"], s["rs"]


def make_baseline(kind: str, task: str):
    cls = type(f"Baseline_{kind}_{task}", (Baseline,), {"task": task, "kind": kind, "__module__": __name__})
    return cls


REG_BASELINES = ["train_mean", "zero", "persistence", "ridge"]
CLF_BASELINES = ["majority_class", "persistence", "logistic_regression"]
