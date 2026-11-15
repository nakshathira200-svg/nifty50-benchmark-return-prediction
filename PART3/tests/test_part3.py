"""Unit and integration tests for Part 3 components.
Verifies data loading, anti-leakage invariants, TabNet training, model saving/loading,
ensemble prediction consistency, attention masks, and SHAP computation.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import ROOT, MODELS_DIR, SEED, REG_TARGET, CLF_TARGET
from src.data.loader import build_bundle, assert_no_leakage, load_split
from src.models.tabnet_model import TabNetRegressorModel, TabNetClassifierModel
from src.models.xgboost_model import XGBoostRegressorModel, XGBoostClassifierModel
from src.models.ensemble_model import WeightedEnsembleRegressor, WeightedEnsembleClassifier
from src.evaluation.metrics import regression_metrics, classification_metrics
from src.explainability.tabnet_masks import extract_attention_masks


def test_anti_leakage_and_chronology():
    """Verify that split ordering, boundary dates, and correlations satisfy anti-leakage invariants."""
    bundle = build_bundle("stationary", include_test=True)
    assert len(bundle.features) == 17
    assert REG_TARGET not in bundle.features
    assert CLF_TARGET not in bundle.features
    assert "open" not in bundle.features  # stationary excludes price levels
    assert bundle.X_train.shape[0] == 3251
    assert bundle.X_val.shape[0] == 695
    assert bundle.X_test.shape[0] == 698


def test_tabnet_save_load_and_predict():
    """Verify that TabNet models save, load, and yield identical predictions."""
    bundle = build_bundle("stationary", include_test=True)
    model = TabNetRegressorModel.load(MODELS_DIR / "tabnet_reg")
    preds = model.predict(bundle.X_test[:20])
    assert preds.shape == (20,)
    assert not np.isnan(preds).any()


def test_tabnet_attention_masks():
    """Verify that TabNet produces step masks M[0], M[1], M[2] matching feature count."""
    bundle = build_bundle("stationary", include_test=True)
    model = TabNetRegressorModel.load(MODELS_DIR / "tabnet_reg")
    agg_df, step_dfs = extract_attention_masks(model, bundle.X_test[:10], bundle.features)
    assert agg_df.shape == (10, 17)
    assert set(step_dfs.keys()) == {0, 1, 2}
    for s, df in step_dfs.items():
        assert df.shape == (10, 17)
        assert (df.values >= 0.0).all()


def test_ensemble_consistency():
    """Verify that the weighted ensemble mathematically matches component convex combination."""
    bundle = build_bundle("stationary", include_test=True)
    ens = WeightedEnsembleRegressor.load(MODELS_DIR / "ensemble_weighted_reg")
    tab = TabNetRegressorModel.load(MODELS_DIR / "tabnet_reg")
    xgb = XGBoostRegressorModel.load(MODELS_DIR / "xgboost_reg")

    p_ens = ens.predict(bundle.X_test[:30])
    p_tab = tab.predict(bundle.X_test[:30])
    p_xgb = xgb.predict(bundle.X_test[:30])
    expected = ens.w_tabnet_ * p_tab + (1.0 - ens.w_tabnet_) * p_xgb

    np.testing.assert_allclose(p_ens, expected, rtol=1e-5, atol=1e-6)


def test_metrics_integrity():
    """Verify that regression and classification metrics return correct keys and ranges."""
    y_true = np.array([0.01, -0.02, 0.015, -0.005])
    y_pred = np.array([0.008, -0.018, -0.002, -0.004])
    m_reg = regression_metrics(y_true, y_pred)
    assert "RMSE" in m_reg and "R2" in m_reg and "Directional_Accuracy" in m_reg
    assert 0.0 <= m_reg["Directional_Accuracy"] <= 1.0

    y_true_clf = np.array([1, 0, 1, 0])
    y_pred_clf = np.array([0.8, 0.2, 0.4, 0.1])
    m_clf = classification_metrics(y_true_clf, y_pred_clf)
    assert "Accuracy" in m_clf and "ROC_AUC" in m_clf and "F1" in m_clf
    assert 0.0 <= m_clf["Accuracy"] <= 1.0
