"""Data loading and preprocessing with anti-leakage verification.
Scales features using StandardScaler fit exclusively on the chronological training split.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import (
    DATA_DIR,
    REG_TARGET,
    CLF_TARGET,
    FORBIDDEN_INPUT_COLUMNS,
    PRICE_LEVEL_FEATURES,
)

SPLIT_FILES = {
    "train": "train.csv",
    "validation": "validation.csv",
    "test": "test.csv",
}


def load_split(name: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load a split and parse dates, sorted chronologically."""
    path = data_dir / SPLIT_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    df = pd.read_csv(path, parse_dates=["price_date", "target_date"])
    return df.sort_values("price_date").set_index("price_date")


def select_features(df: pd.DataFrame, feature_set: str = "stationary") -> List[str]:
    """Select numeric predictors. 'stationary' excludes raw price levels."""
    cols = [
        c for c in df.columns
        if c not in FORBIDDEN_INPUT_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
    ]
    if feature_set == "stationary":
        cols = [c for c in cols if c not in PRICE_LEVEL_FEATURES]
    elif feature_set != "all":
        raise ValueError(f"Unknown feature_set '{feature_set}'; choose 'stationary' or 'all'")
    return cols


def assert_no_leakage(
    features: List[str],
    train: pd.DataFrame,
    val: Optional[pd.DataFrame] = None,
    test: Optional[pd.DataFrame] = None,
) -> None:
    """Enforce strict anti-leakage invariants:
    1. No target column or forbidden column included as a predictor.
    2. Chronological ascending order in each split.
    3. No temporal overlap across splits.
    4. Boundary purge: no target date extends into the subsequent partition.
    5. No feature exhibits an unrealistic correlation (|r| >= 0.5) with target on train.
    """
    bad = [c for c in features if c.startswith("target") or c in FORBIDDEN_INPUT_COLUMNS]
    if bad:
        raise AssertionError(f"Leakage detected: forbidden columns in feature set: {bad}")

    splits = [s for s in (train, val, test) if s is not None]
    for s in splits:
        if not s.index.is_monotonic_increasing:
            raise AssertionError("Temporal ordering error: split index is not monotonically increasing.")

    for early, late in zip(splits[:-1], splits[1:]):
        if early.index.max() >= late.index.min():
            raise AssertionError(f"Temporal overlap detected between {early.index.max()} and {late.index.min()}.")
        if early["target_date"].max() >= late.index.min():
            raise AssertionError(f"Boundary leakage: target_date {early['target_date'].max()} crosses into {late.index.min()}.")

    for col in features:
        for t in (REG_TARGET, CLF_TARGET):
            r = np.corrcoef(train[col], train[t])[0, 1]
            if abs(r) >= 0.5:
                raise AssertionError(f"Suspicious feature correlation: '{col}' with '{t}' r={r:.3f} >= 0.5")


@dataclass
class Bundle:
    """Encapsulates scaled feature matrices, target vectors, date indices, and feature names."""
    features: List[str]
    scaler: StandardScaler
    X_train: np.ndarray
    X_val: np.ndarray
    y_train_reg: np.ndarray
    y_val_reg: np.ndarray
    y_train_clf: np.ndarray
    y_val_clf: np.ndarray
    val_index: pd.DatetimeIndex
    X_test: Optional[np.ndarray] = None
    y_test_reg: Optional[np.ndarray] = None
    y_test_clf: Optional[np.ndarray] = None
    test_index: Optional[pd.DatetimeIndex] = None


def build_bundle(
    feature_set: str = "stationary",
    include_test: bool = True,
    scaler: Optional[StandardScaler] = None,
    data_dir: Path = DATA_DIR,
) -> Bundle:
    """Build data bundle. StandardScaler is fit exclusively on the training set."""
    train_df = load_split("train", data_dir)
    val_df = load_split("validation", data_dir)
    test_df = load_split("test", data_dir) if include_test else None

    features = select_features(train_df, feature_set)
    assert_no_leakage(features, train_df, val_df, test_df)

    if scaler is None:
        scaler = StandardScaler().fit(train_df[features].values)

    bundle = Bundle(
        features=features,
        scaler=scaler,
        X_train=scaler.transform(train_df[features].values),
        X_val=scaler.transform(val_df[features].values),
        y_train_reg=train_df[REG_TARGET].values.astype(np.float64),
        y_val_reg=val_df[REG_TARGET].values.astype(np.float64),
        y_train_clf=train_df[CLF_TARGET].values.astype(np.int64),
        y_val_clf=val_df[CLF_TARGET].values.astype(np.int64),
        val_index=val_df.index,
    )

    if test_df is not None:
        bundle.X_test = scaler.transform(test_df[features].values)
        bundle.y_test_reg = test_df[REG_TARGET].values.astype(np.float64)
        bundle.y_test_clf = test_df[CLF_TARGET].values.astype(np.int64)
        bundle.test_index = test_df.index

    return bundle


def save_preprocessing(bundle: Bundle, model_root: Path, feature_set: str = "stationary") -> None:
    """Save the fitted scaler and feature metadata."""
    prep_dir = model_root / "preprocessing"
    prep_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle.scaler, prep_dir / "scaler.joblib")
    meta = {
        "feature_set": feature_set,
        "n_features": len(bundle.features),
        "features": bundle.features,
    }
    (prep_dir / "features.json").write_text(json.dumps(meta, indent=2))


def load_preprocessing(model_root: Path):
    """Load fitted scaler and feature list."""
    prep_dir = model_root / "preprocessing"
    meta = json.loads((prep_dir / "features.json").read_text())
    scaler = joblib.load(prep_dir / "scaler.joblib")
    return scaler, meta["feature_set"], meta["features"]
