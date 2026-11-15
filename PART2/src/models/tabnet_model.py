"""PyTorch TabNet (pytorch-tabnet package). Attentive feature selection gives NATIVE feature importances.
Early stopping on the chronological validation set."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from src.models.base import BaseModel

HP = dict(n_d=8, n_a=8, n_steps=3, gamma=1.3, lambda_sparse=1e-3, lr=2e-2, max_epochs=200, patience=20,
          batch_size=256, virtual_batch_size=64)


class _TabNet(BaseModel):
    framework = "tabnet"

    def _fit(self, X_train, y_train, X_val, y_val):
        import torch
        from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier
        hp = {**HP, **self.hp}
        torch.manual_seed(self.seed)
        kw = dict(n_d=hp["n_d"], n_a=hp["n_a"], n_steps=hp["n_steps"], gamma=hp["gamma"],
                  lambda_sparse=hp["lambda_sparse"], optimizer_fn=torch.optim.Adam,
                  optimizer_params=dict(lr=hp["lr"]), seed=self.seed, verbose=0)
        fit_kw = dict(max_epochs=hp["max_epochs"], patience=hp["patience"], batch_size=hp["batch_size"],
                      virtual_batch_size=hp["virtual_batch_size"], drop_last=False)
        if self.task == "regression":
            self.model_ = TabNetRegressor(**kw)
            self.model_.fit(X_train, y_train.reshape(-1, 1).astype(np.float32),
                            eval_set=[(X_val, y_val.reshape(-1, 1).astype(np.float32))],
                            eval_name=["val"], eval_metric=["rmse"], **fit_kw)
        else:
            self.model_ = TabNetClassifier(**kw)
            self.model_.fit(X_train, y_train.astype(int), eval_set=[(X_val, y_val.astype(int))],
                            eval_name=["val"], eval_metric=["logloss"], **fit_kw)
        self.epochs_trained_ = int(len(self.model_.history["loss"]))
        self.n_params_ = int(sum(p.numel() for p in self.model_.network.parameters()))
        self.native_importance_ = dict(zip(self.feature_names, [float(v) for v in self.model_.feature_importances_]))

    def _predict_raw(self, X):
        if self.task == "regression":
            return self.model_.predict(X).reshape(-1)
        return self.model_.predict_proba(X)[:, 1]

    def _save(self, d: Path):
        self.model_.save_model(str(d / "tabnet_model"))          # writes tabnet_model.zip

    def _load(self, d: Path):
        from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier
        self.model_ = TabNetRegressor() if self.task == "regression" else TabNetClassifier()
        self.model_.load_model(str(d / "tabnet_model.zip"))


class TabNetRegressorModel(_TabNet):
    task = "regression"


class TabNetClassifierModel(_TabNet):
    task = "classification"


if __name__ == "__main__":
    from src.training import cli_main
    cli_main("tabnet")
