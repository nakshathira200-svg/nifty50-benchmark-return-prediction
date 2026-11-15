"""Shared PyTorch-Lightning helpers: a framework-agnostic early-stopping callback that restores the best weights,
and a plain Lightning MLP used as the documented fallback when Lightning Flash cannot be installed/run.

The callback discovers the validation-loss key at runtime instead of hard-coding it, because the logged metric
name differs between plain Lightning ('val_loss') and Flash tasks."""
from __future__ import annotations
import copy
import numpy as np


def _callback_base():
    try:
        import lightning.pytorch as pl
        return pl.Callback
    except Exception:
        import pytorch_lightning as pl
        return pl.Callback


def make_best_weight_early_stopping(patience: int = 20):
    Base = _callback_base()

    class BestWeightEarlyStopping(Base):
        def __init__(self):
            super().__init__()
            self.best, self.wait, self.state, self.key, self.best_epoch, self.val_history = np.inf, 0, None, None, 0, []

        def _find_key(self, metrics):
            if "val_loss" in metrics:
                return "val_loss"
            cands = [k for k in metrics if k.startswith("val") and not any(t in k for t in ("acc", "auc", "f1", "r2"))]
            return cands[0] if cands else None

        def on_validation_end(self, trainer, pl_module):   # same hook Lightning's own EarlyStopping uses
            if getattr(trainer, "sanity_checking", False):
                return
            metrics = trainer.callback_metrics
            if self.key is None:
                self.key = self._find_key(metrics)
                if self.key is None:
                    raise RuntimeError(f"No validation loss found among logged metrics: {list(metrics)}")
            v = float(metrics[self.key]); self.val_history.append(v)
            if v < self.best - 1e-8:
                self.best, self.wait, self.best_epoch = v, 0, trainer.current_epoch
                self.state = copy.deepcopy(pl_module.state_dict())
            else:
                self.wait += 1
                if self.wait >= patience:
                    trainer.should_stop = True

        def on_fit_end(self, trainer, pl_module):
            if self.state is not None:
                pl_module.load_state_dict(self.state)

    return BestWeightEarlyStopping()


def make_lightning_mlp(n_in: int, task: str, hidden=(64, 32), dropout=0.2, lr=1e-3, wd=1e-4):
    try:
        import lightning.pytorch as pl
    except Exception:
        import pytorch_lightning as pl
    import torch
    from torch import nn

    class LitMLP(pl.LightningModule):
        def __init__(self):
            super().__init__()
            layers, d = [], n_in
            for h in hidden:
                layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]; d = h
            layers.append(nn.Linear(d, 1))
            self.net = nn.Sequential(*layers)
            self.loss_fn = nn.MSELoss() if task == "regression" else nn.BCEWithLogitsLoss()

        def forward(self, x):
            return self.net(x).squeeze(-1)

        def training_step(self, batch, _):
            x, y = batch
            loss = self.loss_fn(self(x), y); self.log("train_loss", loss); return loss

        def validation_step(self, batch, _):
            x, y = batch
            self.log("val_loss", self.loss_fn(self(x), y), prog_bar=False)

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

    return LitMLP()


def seed_everything(seed: int) -> None:
    try:
        import lightning.pytorch as pl
    except Exception:
        import pytorch_lightning as pl
    pl.seed_everything(seed, workers=True)
