"""Model-ready data: load Part 1 splits, choose features, guard against leakage, scale on TRAIN only."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import (DATASET_DIR, REG_TARGET, CLF_TARGET, FORBIDDEN_INPUT_COLUMNS, PRICE_LEVEL_FEATURES)

SPLIT_FILES = {"train": "train.csv", "validation": "validation.csv", "test": "test.csv"}


def load_split(name: str, dataset_dir: Path = DATASET_DIR) -> pd.DataFrame:
    df = pd.read_csv(dataset_dir / SPLIT_FILES[name], parse_dates=["price_date", "target_date"])
    return df.sort_values("price_date").set_index("price_date")


def select_features(df: pd.DataFrame, feature_set: str = "stationary") -> List[str]:
    """Numeric predictors only. 'stationary' drops raw price levels; 'all' keeps Part 1's full list."""
    cols = [c for c in df.columns if c not in FORBIDDEN_INPUT_COLUMNS and pd.api.types.is_numeric_dtype(df[c])]
    if feature_set == "stationary":
        cols = [c for c in cols if c not in PRICE_LEVEL_FEATURES]
    elif feature_set != "all":
        raise ValueError("feature_set must be 'stationary' or 'all'")
    return cols


def assert_no_leakage(features: List[str], train: pd.DataFrame, val: Optional[pd.DataFrame] = None,
                      test: Optional[pd.DataFrame] = None) -> None:
    """Hard checks that fail loudly. Called on every data load."""
    bad = [c for c in features if c.startswith("target") or c in FORBIDDEN_INPUT_COLUMNS]
    assert not bad, f"Forbidden columns used as features: {bad}"
    # chronological order, no overlap, and no target window crossing a partition boundary
    parts = [p for p in (train, val, test) if p is not None]
    for p in parts:
        assert p.index.is_monotonic_increasing, "split not sorted chronologically"
    for a, b in zip(parts[:-1], parts[1:]):
        assert a.index.max() < b.index.min(), "splits overlap in time"
        assert a["target_date"].max() < b.index.min(), "a target date falls inside the next partition"
    # a feature that (almost) equals a target is a leak
    for c in features:
        for t in (REG_TARGET, CLF_TARGET):
            r = np.corrcoef(train[c], train[t])[0, 1]
            assert abs(r) < 0.5, f"feature {c} suspiciously correlated with {t} (r={r:.2f})"


@dataclass
class Bundle:
    """Scaled arrays + raw frames for one experiment."""
    features: List[str]
    scaler: StandardScaler
    X_train: np.ndarray
    X_val: np.ndarray
    y_train_reg: np.ndarray
    y_val_reg: np.ndarray
    y_train_clf: np.ndarray
    y_val_clf: np.ndarray
    X_test: Optional[np.ndarray] = None
    y_test_reg: Optional[np.ndarray] = None
    y_test_clf: Optional[np.ndarray] = None
    test_index: Optional[pd.DatetimeIndex] = None
    val_index: Optional[pd.DatetimeIndex] = None
    # last-day return of each row is available at prediction time (feature `daily_return`)


def build_bundle(feature_set: str = "stationary", include_test: bool = False,
                 scaler: Optional[StandardScaler] = None, dataset_dir: Path = DATASET_DIR) -> Bundle:
    """include_test=False for training/tuning (the test file is not even read). The scaler is FIT ON TRAIN ONLY;
    if a fitted scaler is passed (evaluation time) it is only applied."""
    tr, va = load_split("train", dataset_dir), load_split("validation", dataset_dir)
    te = load_split("test", dataset_dir) if include_test else None
    feats = select_features(tr, feature_set)
    assert_no_leakage(feats, tr, va, te)
    if scaler is None:
        scaler = StandardScaler().fit(tr[feats].values)
    b = Bundle(features=feats, scaler=scaler,
               X_train=scaler.transform(tr[feats].values), X_val=scaler.transform(va[feats].values),
               y_train_reg=tr[REG_TARGET].values, y_val_reg=va[REG_TARGET].values,
               y_train_clf=tr[CLF_TARGET].values.astype(int), y_val_clf=va[CLF_TARGET].values.astype(int),
               val_index=va.index)
    if te is not None:
        b.X_test, b.y_test_reg = scaler.transform(te[feats].values), te[REG_TARGET].values
        b.y_test_clf, b.test_index = te[CLF_TARGET].values.astype(int), te.index
    return b


def save_preprocessing(bundle: Bundle, model_root: Path, feature_set: str) -> None:
    d = Path(model_root) / "preprocessing"; d.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle.scaler, d / "scaler.joblib")
    (d / "features.json").write_text(json.dumps({"feature_set": feature_set, "features": bundle.features}, indent=2))


def load_preprocessing(model_root: Path):
    d = Path(model_root) / "preprocessing"
    meta = json.loads((d / "features.json").read_text())
    return joblib.load(d / "scaler.joblib"), meta["feature_set"], meta["features"]
