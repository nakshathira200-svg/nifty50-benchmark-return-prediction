"""Tests for Part 2. Framework-specific tests are SKIPPED (not passed) when the framework is not installed."""
import importlib, tempfile, unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import REG_TARGET, CLF_TARGET, FORBIDDEN_INPUT_COLUMNS, FRAMEWORKS, ROOT
from src.data import prepare
from src.data.prepare import build_bundle, select_features, assert_no_leakage, load_split
from src.evaluation.metrics import directional_accuracy, regression_metrics, classification_metrics
from src.models.registry import (framework_model_names, baseline_model_names, get_class, parse)


def _tiny(n=400):
    b = build_bundle("stationary", include_test=False)
    b.X_train, b.y_train_reg, b.y_train_clf = b.X_train[:n], b.y_train_reg[:n], b.y_train_clf[:n]
    b.X_val, b.y_val_reg, b.y_val_clf = b.X_val[:150], b.y_val_reg[:150], b.y_val_clf[:150]
    return b


# ---------------------------------------------------------------- data / leakage
def test_targets_and_identifiers_never_in_features():
    tr = load_split("train")
    for fs in ("stationary", "all"):
        feats = select_features(tr, fs)
        assert not set(feats) & set(FORBIDDEN_INPUT_COLUMNS)
        assert not any(f.startswith("target") for f in feats)


def test_feature_sets():
    tr = load_split("train")
    assert len(select_features(tr, "all")) == 21           # Part 1 delivered 21 predictors
    st = select_features(tr, "stationary")
    assert len(st) == 17 and not {"open", "high", "low", "close"} & set(st)


def test_scaler_is_fit_on_train_only():
    b = build_bundle("stationary")
    tr, va = load_split("train"), load_split("validation")
    assert np.allclose(b.scaler.mean_, tr[b.features].mean().values)
    assert not np.allclose(b.scaler.mean_, va[b.features].mean().values)
    assert np.allclose(b.X_train.mean(0), 0, atol=1e-8)


def test_test_split_not_loaded_for_training():
    b = build_bundle("stationary", include_test=False)
    assert b.X_test is None and b.y_test_reg is None and b.test_index is None


def test_leakage_guard_catches_target_copy():
    tr, va = load_split("train"), load_split("validation")
    tr = tr.copy(); tr["leaky"] = tr[REG_TARGET] * 3
    try:
        assert_no_leakage(select_features(tr, "stationary") + ["leaky"], tr, va)
    except AssertionError:
        return
    raise AssertionError("leakage guard failed to flag a feature that copies the target")


def test_leakage_guard_catches_forbidden_column():
    tr, va = load_split("train"), load_split("validation")
    try:
        assert_no_leakage(["daily_return", CLF_TARGET], tr, va)
    except AssertionError:
        return
    raise AssertionError("forbidden column accepted as feature")


def test_chronological_order_and_no_target_crossing():
    tr, va, te = (load_split(s) for s in ("train", "validation", "test"))
    assert tr.index.max() < va.index.min() < te.index.min()
    assert tr.target_date.max() < va.index.min() and va.target_date.max() < te.index.min()


# ---------------------------------------------------------------- metrics
def test_directional_accuracy_known_values():
    assert directional_accuracy([0.01, -0.01, 0.02, -0.02], [0.5, -0.5, -0.1, 0.1]) == 0.5
    assert directional_accuracy([0.0], [-0.1]) == 1.0      # zero return counts as 'not up' like the label


def test_regression_metrics_known_values():
    m = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 4.0])
    assert abs(m["MAE"] - 1 / 3) < 1e-12 and abs(m["MSE"] - 1 / 3) < 1e-12 and abs(m["RMSE"] - np.sqrt(1 / 3)) < 1e-12


def test_classification_metrics_known_values():
    m = classification_metrics([1, 1, 0, 0], [0.9, 0.4, 0.6, 0.1])
    assert (m["TP"], m["FN"], m["FP"], m["TN"]) == (1, 1, 1, 1)
    assert m["Accuracy"] == 0.5 and abs(m["ROC_AUC"] - 0.75) < 1e-12


# ---------------------------------------------------------------- registry / structure
def test_registry_has_ten_framework_models_and_baselines():
    names = framework_model_names()
    assert len(names) == 10 and len(set(names)) == 10
    assert {parse(n)["framework"] for n in names} == set(FRAMEWORKS)
    assert {parse(n)["task"] for n in names} == {"regression", "classification"}
    assert len(baseline_model_names()) == 7


def test_every_framework_module_imports_without_the_framework():
    """Frameworks are imported lazily so the code base stays importable and statically checkable."""
    for n in framework_model_names():
        assert get_class(n).task == parse(n)["task"]


# ---------------------------------------------------------------- round trips that need no heavy framework
def _roundtrip(name, standin=False):
    b = _tiny(); info = parse(name); reg = info["task"] == "regression"
    cls = get_class(name, standin=standin); m = cls(seed=7); m.attach_context(scaler=b.scaler)
    yt, yv = (b.y_train_reg, b.y_val_reg) if reg else (b.y_train_clf, b.y_val_clf)
    m.fit(b.X_train, yt, b.X_val, yv, b.features)
    p1 = m.predict(b.X_val)
    with tempfile.TemporaryDirectory() as d:
        m.save(d); m2 = cls.load(d)
    p2 = m2.predict(b.X_val)
    assert p1.shape == (len(b.X_val),) and np.allclose(p1, p2, atol=1e-5), f"{name}: save/load changed predictions"
    if not reg:
        assert ((p2 >= 0) & (p2 <= 1)).all(), "classifier must output probabilities"


def test_baselines_train_save_load():
    for n in baseline_model_names():
        _roundtrip(n)


def test_persistence_baseline_uses_unscaled_last_return():
    b = _tiny(); cls = get_class("baseline_persistence_reg"); m = cls(seed=1); m.attach_context(scaler=b.scaler)
    m.fit(b.X_train, b.y_train_reg, b.X_val, b.y_val_reg, b.features)
    j = b.features.index("daily_return")
    raw = b.X_val[:, j] * b.scaler.scale_[j] + b.scaler.mean_[j]
    assert np.allclose(m.predict(b.X_val), raw, atol=1e-9)


def test_standin_harness_roundtrip_all_ten_slots():
    for n in framework_model_names():
        _roundtrip(n, standin=True)


def test_reproducible_with_same_seed():
    b = _tiny(); cls = get_class("xgboost_reg", standin=True)
    ps = []
    for _ in range(2):
        m = cls(seed=3); m.fit(b.X_train, b.y_train_reg, b.X_val, b.y_val_reg, b.features); ps.append(m.predict(b.X_val))
    assert np.allclose(ps[0], ps[1])


# ---------------------------------------------------------------- real frameworks (skipped when not installed)
_NEEDS = {"xgboost": ["xgboost"], "keras": ["tensorflow"], "fastai": ["fastai"], "tabnet": ["pytorch_tabnet", "torch"],
          "flash": ["torch", "flash"]}
_FAST_HP = {"keras": dict(max_epochs=2), "fastai": dict(max_epochs=2), "tabnet": dict(max_epochs=2),
            "flash": dict(max_epochs=2, allow_fallback=False), "xgboost": {}}


def _require(fw):
    for mod in _NEEDS[fw]:
        try:
            importlib.import_module(mod)
        except Exception as e:
            raise unittest.SkipTest(f"{mod} not installed ({type(e).__name__})")


def _framework_roundtrip(fw, task):
    _require(fw)
    b = _tiny(); reg = task == "regression"
    from src.models.registry import model_name
    cls = get_class(model_name(fw, task)); m = cls(seed=11, **_FAST_HP[fw])
    yt, yv = (b.y_train_reg, b.y_val_reg) if reg else (b.y_train_clf, b.y_val_clf)
    m.fit(b.X_train, yt, b.X_val, yv, b.features)
    p1 = m.predict(b.X_val)
    with tempfile.TemporaryDirectory() as d:
        m.save(d); m2 = cls.load(d)
    p2 = m2.predict(b.X_val)
    assert np.allclose(p1, p2, atol=1e-4), f"{fw}_{task}: predictions changed after save/load"
    assert np.isfinite(p2).all()


def test_xgboost_regression_real():     _framework_roundtrip("xgboost", "regression")
def test_xgboost_classification_real(): _framework_roundtrip("xgboost", "classification")
def test_keras_regression_real():       _framework_roundtrip("keras", "regression")
def test_keras_classification_real():   _framework_roundtrip("keras", "classification")
def test_fastai_regression_real():      _framework_roundtrip("fastai", "regression")
def test_fastai_classification_real():  _framework_roundtrip("fastai", "classification")
def test_tabnet_regression_real():      _framework_roundtrip("tabnet", "regression")
def test_tabnet_classification_real():  _framework_roundtrip("tabnet", "classification")
def test_flash_regression_real():       _framework_roundtrip("flash", "regression")
def test_flash_classification_real():   _framework_roundtrip("flash", "classification")


# Plain PyTorch-Lightning fallback used when Lightning Flash cannot be installed (labelled in all outputs)
def _flash_fallback(task):
    for mod in ("torch", "pytorch_lightning"):
        try:
            importlib.import_module(mod)
        except Exception as e:
            try:
                importlib.import_module("lightning")
            except Exception:
                raise unittest.SkipTest(f"{mod} not installed ({type(e).__name__})")
    b = _tiny(); reg = task == "regression"
    from src.models.flash_model import FlashRegressorModel, FlashClassifierModel
    from src.models import flash_model as fm
    cls = FlashRegressorModel if reg else FlashClassifierModel
    m = cls(seed=5, max_epochs=2)
    orig = cls._fit_flash
    cls._fit_flash = lambda *a, **k: (_ for _ in ()).throw(ImportError("flash disabled for fallback test"))
    try:
        yt, yv = (b.y_train_reg, b.y_val_reg) if reg else (b.y_train_clf, b.y_val_clf)
        m.fit(b.X_train, yt, b.X_val, yv, b.features)
    finally:
        cls._fit_flash = orig
    assert m.implementation == "pytorch_lightning_fallback" and np.isfinite(m.predict(b.X_val)).all()


def test_flash_fallback_regression_real(): _flash_fallback("regression")
def test_flash_fallback_classification_real(): _flash_fallback("classification")


def test_fastai_learner_config_uses_tabular_config():
    # regression test for a bug: `ps` was passed straight to tabular_learner (-> Learner TypeError)
    src = (ROOT / "src" / "models" / "fastai_model.py").read_text()
    assert "config=tabular_config(ps=" in src and "learner(dls, layers=list(hp[\"layers\"]), ps=" not in src


def test_comparison_has_all_five_regression_metrics():
    import pandas as pd
    df = pd.read_csv(ROOT / "outputs" / "results" / "model_comparison.csv")
    for c in ["test_RMSE", "test_MSE", "test_MAE", "test_R2", "test_Directional_Accuracy"]:
        assert c in df.columns, c
    ok = df[(df.task == "regression") & (df.status == "evaluated")]
    assert len(ok) >= 4 and ok[["test_RMSE", "test_MSE", "test_MAE", "test_R2", "test_Directional_Accuracy"]].notna().all().all()
    assert np.allclose(ok.test_MSE, ok.test_RMSE ** 2)
    failed = df[df.status != "evaluated"]
    assert (failed.error.fillna("") != "").all()      # failures are never silent
