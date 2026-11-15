"""Central configuration for Part 3: Deep Learning (TabNet), Ensembling & Explainability.
Defines paths, random seeds, target definitions, feature subsets, and hyperparameters.
"""
from __future__ import annotations
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "results"
FIGURES_DIR = OUTPUTS_DIR / "figures"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
DELIVERABLES_DIR = ROOT / "deliverables"

SEED = 42

REG_TARGET = "target_next_day_return"
CLF_TARGET = "target_next_day_up"

FORBIDDEN_INPUT_COLUMNS = [
    REG_TARGET, CLF_TARGET, "target_date", "price_date",
    "adj_close", "volume", "dividends", "stock_splits"
]

PRICE_LEVEL_FEATURES = ["open", "high", "low", "close"]

TABNET_HP = {
    "n_d": 8,
    "n_a": 8,
    "n_steps": 3,
    "gamma": 1.3,
    "lambda_sparse": 1e-3,
    "lr": 2e-2,
    "max_epochs": 200,
    "patience": 20,
    "batch_size": 256,
    "virtual_batch_size": 64,
}

XGBOOST_GRID = {
    "max_depth": [2, 3, 4],
    "learning_rate": [0.02, 0.05],
}

XGBOOST_FIXED = {
    "n_estimators": 2000,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 5.0,
    "tree_method": "hist",
    "n_jobs": 1,
    "importance_type": "gain",
}

def set_global_seed(seed: int = SEED) -> None:
    """Enforce strict reproducibility across Python, NumPy, PyTorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
