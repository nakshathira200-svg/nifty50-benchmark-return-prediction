"""PyTorch + Lightning Flash tabular models.

Flash is in maintenance mode and pins older Python / PyTorch / Lightning versions, so it may not install next to
the other frameworks (see requirements-flash.txt). Behaviour:

1. Try the real Lightning Flash tabular task (TabularRegressor / TabularClassifier).
2. If Flash cannot be imported, trained, or produce usable predictions, fall back to a plain PyTorch Lightning MLP
   and record the exact Flash error. The result is then labelled implementation='pytorch_lightning_fallback' in
   every output table -- it is never reported as Flash.
Set allow_fallback=False in hyper-parameters to make Flash failures hard errors instead.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from src.models.base import BaseModel
from src.models import _lightning_utils as LU

HP = dict(hidden=(64, 32), dropout=0.2, lr=1e-3, wd=1e-4, batch_size=64, max_epochs=200, patience=20,
          backbone="category_embedding", allow_fallback=True)


def _collect(o, acc):
    """Flatten Flash prediction output (nested lists / tensors / scalars) into rows of floats."""
    import torch
    if isinstance(o, (list, tuple)):
        if o and all(isinstance(x, (int, float, np.integer, np.floating)) for x in o):
            acc.append([float(x) for x in o])
        else:
            for x in o:
                _collect(x, acc)
    elif torch.is_tensor(o):
        a = o.detach().cpu().numpy()
        acc.extend(a.reshape(1, -1).tolist() if a.ndim == 0 else a.reshape(len(a), -1).tolist())
    elif isinstance(o, (int, float, np.integer, np.floating)):
        acc.append([float(o)])
    elif isinstance(o, dict):
        for v in o.values():
            _collect(v, acc)


class _Flash(BaseModel):
    framework = "flash"
    implementation = "lightning_flash"

    def __init__(self, seed: int = 42, **hp):
        super().__init__(seed, **hp)
        self.flash_error_ = None
        self.backbone_used_ = None

    # ---------------------------------------------------------------- Flash path
    def _classes(self):
        from flash.tabular import (TabularRegressionData, TabularRegressor, TabularClassificationData,
                                   TabularClassifier)
        return (TabularRegressionData, TabularRegressor) if self.task == "regression" else \
               (TabularClassificationData, TabularClassifier)

    def _frame(self, X, y=None):
        df = pd.DataFrame(np.asarray(X, dtype=np.float32), columns=self.feature_names)
        if y is not None:
            df["target"] = np.asarray(y, dtype=np.float32 if self.task == "regression" else np.int64)
        return df

    def _fit_flash(self, X_train, y_train, X_val, y_val):
        import flash
        hp = {**HP, **self.hp}
        LU.seed_everything(self.seed)
        Data, Model = self._classes()
        dm = Data.from_data_frame(numerical_fields=self.feature_names, categorical_fields=None, target_fields="target",
                                  train_data_frame=self._frame(X_train, y_train),
                                  val_data_frame=self._frame(X_val, y_val),
                                  batch_size=hp["batch_size"], num_workers=0)
        try:
            self.model_ = Model.from_data(dm, backbone=hp["backbone"]); self.backbone_used_ = hp["backbone"]
        except Exception:                                   # backbone name not available in this Flash version
            self.model_ = Model.from_data(dm); self.backbone_used_ = "flash_default"
        cb = LU.make_best_weight_early_stopping(hp["patience"])
        self.trainer_ = flash.Trainer(max_epochs=hp["max_epochs"], callbacks=[cb], accelerator="cpu", logger=False,
                                      enable_checkpointing=False, enable_progress_bar=False)
        self.trainer_.fit(self.model_, datamodule=dm)
        self.dm_params_ = dm.parameters
        self.epochs_trained_ = len(cb.val_history)
        self.n_params_ = int(sum(p.numel() for p in self.model_.parameters()))

    def _predict_flash(self, X):
        import flash
        Data, _ = self._classes()
        dm = Data.from_data_frame(predict_data_frame=self._frame(X), parameters=self.dm_params_,
                                  batch_size=1024, num_workers=0)
        trainer = getattr(self, "trainer_", None) or flash.Trainer(accelerator="cpu", logger=False,
                                                                   enable_progress_bar=False)
        try:
            out = trainer.predict(self.model_, datamodule=dm,
                                  output="probabilities" if self.task == "classification" else None)
        except TypeError:
            out = trainer.predict(self.model_, datamodule=dm)
        acc = []
        _collect(out, acc)
        arr = np.asarray(acc, dtype=float)
        if len(arr) != len(X):
            raise RuntimeError(f"Flash returned {len(arr)} predictions for {len(X)} rows")
        if self.task == "regression":
            return arr[:, 0]
        if arr.shape[1] >= 2:
            return arr[:, 1]
        if set(np.unique(arr)) <= {0.0, 1.0}:
            raise RuntimeError("Flash returned hard labels, not probabilities")
        return arr[:, 0]

    # ---------------------------------------------------------------- fallback path
    def _fit_fallback(self, X_train, y_train, X_val, y_val):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        try:
            import lightning.pytorch as pl
        except Exception:
            import pytorch_lightning as pl
        hp = {**HP, **self.hp}
        LU.seed_everything(self.seed)
        net = LU.make_lightning_mlp(X_train.shape[1], self.task, hp["hidden"], hp["dropout"], hp["lr"], hp["wd"])
        mk = lambda X, y, sh: DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32),
                                                       torch.tensor(y, dtype=torch.float32)),
                                         batch_size=hp["batch_size"], shuffle=sh)
        cb = LU.make_best_weight_early_stopping(hp["patience"])
        trainer = pl.Trainer(max_epochs=hp["max_epochs"], callbacks=[cb], accelerator="cpu", logger=False,
                             enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
        trainer.fit(net, mk(X_train, y_train, True), mk(X_val, y_val, False))
        self.model_, self.trainer_ = net, trainer
        self.epochs_trained_ = len(cb.val_history)
        self.n_params_ = int(sum(p.numel() for p in net.parameters()))

    def _predict_fallback(self, X):
        import torch
        self.model_.eval()
        with torch.no_grad():
            out = self.model_(torch.tensor(X, dtype=torch.float32))
            return (out if self.task == "regression" else torch.sigmoid(out)).numpy()

    # ---------------------------------------------------------------- BaseModel hooks
    def _fit(self, X_train, y_train, X_val, y_val):
        hp = {**HP, **self.hp}
        try:
            self.implementation = "lightning_flash"
            self._fit_flash(X_train, y_train, X_val, y_val)
            self._predict_flash(X_val[:32])                  # prove prediction works before trusting Flash
        except Exception as e:
            self.flash_error_ = f"{type(e).__name__}: {e}"
            if not hp["allow_fallback"]:
                raise
            try:
                self.implementation = "pytorch_lightning_fallback"
                self._fit_fallback(X_train, y_train, X_val, y_val)
            except Exception as e2:
                raise RuntimeError(f"Lightning Flash failed ({self.flash_error_}); the PyTorch Lightning fallback "
                                   f"also failed ({type(e2).__name__}: {e2})") from e2

    def _predict_raw(self, X):
        return self._predict_flash(X) if self.implementation == "lightning_flash" else self._predict_fallback(X)

    def meta(self):
        m = super().meta()
        m.update({"flash_error": self.flash_error_, "backbone": self.backbone_used_})
        return m

    def _save(self, d: Path):
        if self.implementation == "lightning_flash":
            import joblib
            self.trainer_.save_checkpoint(str(d / "flash.ckpt"))
            joblib.dump(self.dm_params_, d / "dm_parameters.joblib")
        else:
            import torch
            torch.save(self.model_.state_dict(), d / "fallback_mlp.pt")

    def _load(self, d: Path):
        if self.implementation == "lightning_flash":
            import joblib
            _, Model = self._classes()
            self.model_ = Model.load_from_checkpoint(str(d / "flash.ckpt"))
            self.dm_params_ = joblib.load(d / "dm_parameters.joblib")
        else:
            import torch
            hp = {**HP, **self.hp}
            self.model_ = LU.make_lightning_mlp(len(self.feature_names), self.task, hp["hidden"], hp["dropout"],
                                                hp["lr"], hp["wd"])
            self.model_.load_state_dict(torch.load(d / "fallback_mlp.pt", map_location="cpu"))


class FlashRegressorModel(_Flash):
    task = "regression"


class FlashClassifierModel(_Flash):
    task = "classification"


if __name__ == "__main__":
    from src.training import cli_main
    cli_main("flash")
