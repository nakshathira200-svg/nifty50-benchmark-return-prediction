"""PyTorch + fastai tabular learner. Inputs are the shared standardised features (so every framework sees identical
preprocessing); fastai handles the DataLoaders, embedding-free MLP, training loop and callbacks.
Early stopping / best-weight reload use validation loss. The validation split is the Part 1 chronological
validation set, passed explicitly (never fastai's random splitter)."""
from __future__ import annotations
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from src.models.base import BaseModel

HP = dict(layers=(64, 32), lr=1e-3, wd=1e-2, ps=0.2, bs=64, max_epochs=200, patience=15)


class _FastAI(BaseModel):
    framework = "fastai"

    def _frame(self, X, y=None):
        df = pd.DataFrame(np.asarray(X, dtype=np.float32), columns=self.feature_names)
        if y is not None:
            df["y"] = np.asarray(y, dtype=np.float32 if self.task == "regression" else np.int64)
        return df

    def _fit(self, X_train, y_train, X_val, y_val):
        from fastai.tabular.all import (TabularPandas, RegressionBlock, CategoryBlock, tabular_learner,
                                        EarlyStoppingCallback, SaveModelCallback, tabular_config)
        import torch
        hp = {**HP, **self.hp}
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        df = pd.concat([self._frame(X_train, y_train), self._frame(X_val, y_val)], ignore_index=True)
        splits = (list(range(len(X_train))), list(range(len(X_train), len(df))))
        y_block = RegressionBlock() if self.task == "regression" else CategoryBlock()
        to = TabularPandas(df, procs=[], cat_names=[], cont_names=self.feature_names, y_names="y",
                           y_block=y_block, splits=splits)
        dls = to.dataloaders(bs=hp["bs"], num_workers=0)
        self.learn_ = tabular_learner(dls, layers=list(hp["layers"]), config=tabular_config(ps=hp["ps"]), path=Path(tempfile.mkdtemp(prefix="fastai_tmp_")))
        cbs = [EarlyStoppingCallback(monitor="valid_loss", patience=hp["patience"]),
               SaveModelCallback(monitor="valid_loss", fname="best_tmp", with_opt=False)]
        self.learn_.fit(hp["max_epochs"], hp["lr"], wd=hp["wd"], cbs=cbs)
        self.epochs_trained_ = int(len(self.learn_.recorder.values))
        self.n_params_ = int(sum(p.numel() for p in self.learn_.model.parameters()))

    def _predict_raw(self, X):
        dl = self.learn_.dls.test_dl(self._frame(X), bs=1024)
        preds, _ = self.learn_.get_preds(dl=dl)
        preds = preds.numpy()
        if self.task == "classification":
            return preds[:, 1]                      # softmax probability of class 1 (up)
        return preds.reshape(-1)

    def _save(self, d: Path):
        self.learn_.export(d / "learner.pkl")

    def _load(self, d: Path):
        from fastai.tabular.all import load_learner
        self.learn_ = load_learner(d / "learner.pkl", cpu=True)


class FastAIRegressorModel(_FastAI):
    task = "regression"


class FastAIClassifierModel(_FastAI):
    task = "classification"


if __name__ == "__main__":
    from src.training import cli_main
    cli_main("fastai")
