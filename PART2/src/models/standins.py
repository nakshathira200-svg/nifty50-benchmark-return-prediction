"""sklearn STAND-INS used ONLY to smoke-test the train/save/load/evaluate/compare harness when the real frameworks
cannot be installed (e.g. an offline sandbox). They are NOT XGBoost/Keras/fastai/TabNet/Flash and their results must
never be reported as such. Everything they produce is written under the run tag `smoke_standin`, and the
implementation column always reads 'sklearn_standin_for_<framework>'."""
from __future__ import annotations
import copy
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from src.models.base import BaseModel

HIDDEN = {"keras": (64, 32), "fastai": (64, 32), "tabnet": (16,), "flash": (32, 16)}


class StandIn(BaseModel):
    task = "regression"
    framework = "xgboost"

    def __init__(self, seed: int = 42, framework: str = None, **hp):
        super().__init__(seed, **hp)
        if framework:
            self.framework = framework
        self.implementation = f"sklearn_standin_for_{self.framework}"

    def _fit(self, X_train, y_train, X_val, y_val):
        reg = self.task == "regression"
        if self.framework == "xgboost":
            Est = GradientBoostingRegressor if reg else GradientBoostingClassifier
            m = Est(n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8, random_state=self.seed)
            m.fit(X_train, y_train)
            if reg:
                losses = [np.mean((p - y_val) ** 2) for p in m.staged_predict(X_val)]
            else:
                losses = [-np.mean(y_val * np.log(np.clip(p[:, 1], 1e-6, 1 - 1e-6)) +
                                   (1 - y_val) * np.log(np.clip(1 - p[:, 1], 1e-6, 1 - 1e-6)))
                          for p in m.staged_predict_proba(X_val)]
            n = int(np.argmin(losses)) + 1
            self.model_ = Est(n_estimators=n, learning_rate=0.03, max_depth=3, subsample=0.8,
                              random_state=self.seed).fit(X_train, y_train)
            self.best_params_ = {"n_estimators": n, "max_depth": 3, "learning_rate": 0.03}
            self.epochs_trained_, self.n_params_ = n, n
            self.native_importance_ = dict(zip(self.feature_names, self.model_.feature_importances_.tolist()))
            return
        Est = MLPRegressor if reg else MLPClassifier
        m = Est(hidden_layer_sizes=HIDDEN[self.framework], alpha=1e-3, learning_rate_init=1e-3,
                random_state=self.seed + len(self.framework), batch_size=64)
        best, best_state, wait, ep = np.inf, None, 0, 0
        for ep in range(1, 201):                            # manual early stopping on the CHRONOLOGICAL val set
            if reg:
                m.partial_fit(X_train, y_train); loss = np.mean((m.predict(X_val) - y_val) ** 2)
            else:
                m.partial_fit(X_train, y_train, classes=[0, 1])
                p = np.clip(m.predict_proba(X_val)[:, 1], 1e-6, 1 - 1e-6)
                loss = -np.mean(y_val * np.log(p) + (1 - y_val) * np.log(1 - p))
            if loss < best - 1e-8:
                best, best_state, wait = loss, copy.deepcopy(m), 0
            else:
                wait += 1
                if wait >= 15:
                    break
        self.model_, self.epochs_trained_ = best_state, ep
        self.n_params_ = int(sum(w.size for w in self.model_.coefs_) + sum(b.size for b in self.model_.intercepts_))

    def _predict_raw(self, X):
        return self.model_.predict(X) if self.task == "regression" else self.model_.predict_proba(X)[:, 1]

    def _save(self, d: Path):
        joblib.dump(self.model_, d / "model.joblib")

    def _load(self, d: Path):
        self.model_ = joblib.load(d / "model.joblib")


def make_standin(framework: str, task: str):
    def init(self, seed=42, **hp):
        StandIn.__init__(self, seed, framework=framework, **hp)
    return type(f"StandIn_{framework}_{task}", (StandIn,), {"task": task, "framework": framework,
                                                             "__init__": init, "__module__": __name__})
