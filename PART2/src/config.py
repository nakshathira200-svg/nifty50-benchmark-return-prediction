"""Central configuration for Part 2 (modelling). Paths, seeds, targets and feature sets."""
from __future__ import annotations
import os, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "deliverables" / "datasets"
SEED = 42

REG_TARGET = "target_next_day_return"
CLF_TARGET = "target_next_day_up"
# Columns that are never allowed to be model inputs.
FORBIDDEN_INPUT_COLUMNS = [REG_TARGET, CLF_TARGET, "target_date", "price_date",
                           "adj_close", "volume", "dividends", "stock_splits"]

# Part 1 delivered 21 predictors (see deliverables/PART1_RUN_SUMMARY.json).
# 4 of them are raw price LEVELS. Train close max = 15,315 but 93% of validation rows are above it,
# so levels are out-of-range for trees (no extrapolation) and unstable for NNs.
PRICE_LEVEL_FEATURES = ["open", "high", "low", "close"]

# All 10 model slots: <framework>_<task>
FRAMEWORKS = ["xgboost", "keras", "fastai", "tabnet", "flash"]
TASKS = ["regression", "classification"]


def run_dirs(tag: str = "main") -> dict:
    """Output locations. `main` = real results; any other tag (e.g. smoke_surrogate) is kept separate."""
    suffix = "" if tag == "main" else f"_{tag}"
    d = {
        "models": ROOT / f"models{suffix}",
        "results": ROOT / f"outputs/results{suffix}",
        "predictions": ROOT / f"outputs/predictions{suffix}",
        "graphs": ROOT / f"outputs/model_graphs{suffix}",
    }
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def set_global_seed(seed: int = SEED) -> None:
    """Seed python/numpy and, if installed, torch / tensorflow. Frameworks are imported lazily and optionally."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass
