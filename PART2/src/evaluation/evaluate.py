"""Load saved models from disk, evaluate on validation + test, build comparison tables.
The TEST split is read here (and only here) for final, one-shot evaluation."""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import ROOT, SEED, run_dirs
from src.data.prepare import build_bundle, load_preprocessing, load_split
from src.evaluation.metrics import regression_metrics, classification_metrics
from src.models.registry import get_class, parse, framework_model_names, baseline_model_names

# Analyst judgement (NOT measured): 1 = poor, 5 = excellent. Rationale is in docs/METHODOLOGY_PART2.md.
QUALITATIVE = {
    "xgboost": {"simplicity_rating": 4, "transparency_rating": 4,
                "note": "few hyper-parameters; gain importance + inspectable trees; needs no scaling"},
    "keras":   {"simplicity_rating": 3, "transparency_rating": 2,
                "note": "clear Sequential API; weights are opaque; needs scaling and early stopping"},
    "fastai":  {"simplicity_rating": 4, "transparency_rating": 2,
                "note": "very short high-level API but a lot is hidden inside the Learner"},
    "tabnet":  {"simplicity_rating": 3, "transparency_rating": 4,
                "note": "built-in attentive feature selection, but many interacting hyper-parameters"},
    "flash":   {"simplicity_rating": 2, "transparency_rating": 2,
                "note": "highest abstraction, strict/old version pins, version-dependent output formats"},
    "baseline": {"simplicity_rating": 5, "transparency_rating": 5, "note": "closed-form or rule based"},
}


def code_lines(framework: str):
    f = ROOT / "src" / "models" / f"{framework}_model.py"
    if not f.exists():
        return np.nan
    lines = [l.strip() for l in f.read_text().splitlines()]
    return sum(1 for l in lines if l and not l.startswith("#"))


def permutation_importance(model, X, y, task: str, n_repeats: int = 5, seed: int = SEED) -> pd.DataFrame:
    """Model-agnostic importance on VALIDATION data: increase in loss when one feature is shuffled."""
    rng = np.random.default_rng(seed)

    def loss(p):
        if task == "regression":
            return float(np.mean((p - y) ** 2))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    base = loss(model.predict(X))
    rows = []
    for j, f in enumerate(model.feature_names):
        inc = []
        for _ in range(n_repeats):
            Xp = X.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
            inc.append(loss(model.predict(Xp)) - base)
        rows.append({"feature": f, "permutation_mean": float(np.mean(inc)), "permutation_std": float(np.std(inc))})
    return pd.DataFrame(rows)


def evaluate_run(tag: str = "main", n_repeats: int = 5) -> dict:
    dirs = run_dirs(tag)
    status = pd.read_csv(dirs["results"] / "run_status.csv")
    scaler, feature_set, feats = load_preprocessing(dirs["models"])
    tr = load_split("train")
    assert np.allclose(scaler.mean_, tr[feats].mean().values), "scaler was not fitted on the training split"
    b = build_bundle(feature_set, include_test=True, scaler=scaler)

    reg_val, reg_test, clf_val, clf_test, eval_rows, importances, preds = [], [], [], [], [], {}, {}
    for _, r in status.iterrows():
        name, info = r["model"], parse(r["model"])
        rec = {"model": name, "framework": info["framework"], "task": info["task"], "status": r["status"],
               "implementation": r["implementation"], "train_time_sec": r["train_time_sec"],
               "n_parameters": r["n_parameters"], "epochs_trained": r["epochs_trained"], "error": r["error"]}
        if r["status"] != "trained":
            eval_rows.append(rec); continue
        try:
            cls = get_class(name, standin=str(r["implementation"]).startswith("sklearn_standin"))
            model = cls.load(dirs["models"] / name)                              # proves save -> load works
            reg = info["task"] == "regression"
            yv, yt = (b.y_val_reg, b.y_test_reg) if reg else (b.y_val_clf, b.y_test_clf)
            pv, pt = model.predict(b.X_val), model.predict(b.X_test)
            mv = regression_metrics(yv, pv) if reg else classification_metrics(yv, pv)
            mt = regression_metrics(yt, pt) if reg else classification_metrics(yt, pt)
            (reg_val if reg else clf_val).append({"model": name, **mv})
            (reg_test if reg else clf_test).append({"model": name, **mt})
            preds[name] = {"task": info["task"], "y_true": yt, "y_pred": pt}
            pd.DataFrame({"price_date": b.test_index, "y_true": yt, "y_pred": pt}).to_csv(
                dirs["predictions"] / f"{name}_test.csv", index=False)
            imp = permutation_importance(model, b.X_val, yv, info["task"], n_repeats)
            nat = model.feature_importance()
            imp["native"] = imp["feature"].map(nat) if nat is not None else np.nan
            imp.to_csv(dirs["results"] / f"feature_importance_{name}.csv", index=False)
            importances[name] = imp
            rec["status"] = "evaluated"
        except Exception as e:
            rec["status"], rec["error"] = "evaluation_failed", f"{type(e).__name__}: {str(e)[:300]}"
        eval_rows.append(rec)

    out = {}
    for fname, rows in [("regression_validation_metrics", reg_val), ("regression_test_metrics", reg_test),
                        ("classification_validation_metrics", clf_val), ("classification_test_metrics", clf_test)]:
        df = pd.DataFrame(rows); df.to_csv(dirs["results"] / f"{fname}.csv", index=False); out[fname] = df
    out["comparison"] = build_comparison(pd.DataFrame(eval_rows), out)
    out["comparison"].to_csv(dirs["results"] / "model_comparison.csv", index=False)
    out["preds"], out["importances"], out["bundle"], out["dirs"] = preds, importances, b, dirs
    return out


def build_comparison(ev: pd.DataFrame, res: dict) -> pd.DataFrame:
    """One row per model slot: status, key test metrics, cost, simplicity/transparency, efficacy rank and
    lift over the best baseline. Models that did not train keep NaN metrics (nothing is invented)."""
    rt, ct = res["regression_test_metrics"], res["classification_test_metrics"]
    df = ev.copy()
    for col in ["test_RMSE", "test_MSE", "test_R2", "test_MAE", "test_Directional_Accuracy", "test_Accuracy", "test_F1",
                "test_ROC_AUC", "test_Accuracy_CI95_halfwidth"]:
        df[col] = np.nan
    for _, r in rt.iterrows():
        i = df.index[df.model == r["model"]][0]
        df.loc[i, ["test_RMSE", "test_MSE", "test_R2", "test_MAE", "test_Directional_Accuracy"]] = [r["RMSE"], r["MSE"], r["R2"], r["MAE"], r["Directional_Accuracy"]]
    for _, r in ct.iterrows():
        i = df.index[df.model == r["model"]][0]
        df.loc[i, ["test_Accuracy", "test_F1", "test_ROC_AUC", "test_Accuracy_CI95_halfwidth"]] = [r["Accuracy"], r["F1"], r["ROC_AUC"], r["Accuracy_CI95_halfwidth"]]
    df["simplicity_loc"] = df["framework"].map(code_lines)
    df["simplicity_rating"] = df["framework"].map(lambda f: QUALITATIVE[f]["simplicity_rating"])
    df["transparency_rating"] = df["framework"].map(lambda f: QUALITATIVE[f]["transparency_rating"])
    df["rating_note"] = df["framework"].map(lambda f: QUALITATIVE[f]["note"])
    # efficacy: rank among evaluated models of the same task (lower RMSE / higher AUC = better)
    df["efficacy_rank"] = np.nan
    r_ok, c_ok = df[df.test_RMSE.notna()], df[df.test_ROC_AUC.notna()]
    df.loc[r_ok.index, "efficacy_rank"] = r_ok["test_RMSE"].rank(method="min")
    df.loc[c_ok.index, "efficacy_rank"] = c_ok["test_ROC_AUC"].rank(ascending=False, method="min")
    # lift over reference baselines
    bm = df[(df.model == "baseline_train_mean_reg")]["test_RMSE"]
    bmaj = df[(df.model == "baseline_majority_class_clf")]["test_Accuracy"]
    df["rmse_vs_train_mean_baseline_pct"] = (df["test_RMSE"] / bm.iloc[0] - 1) * 100 if len(bm) else np.nan
    df["accuracy_vs_majority_baseline_pp"] = (df["test_Accuracy"] - bmaj.iloc[0]) * 100 if len(bmaj) else np.nan
    keep = ["model", "framework", "task", "implementation", "status", "test_RMSE", "test_MSE", "test_MAE", "test_R2",
            "test_Directional_Accuracy", "rmse_vs_train_mean_baseline_pct", "test_Accuracy", "test_Accuracy_CI95_halfwidth",
            "test_F1", "test_ROC_AUC", "accuracy_vs_majority_baseline_pp", "efficacy_rank", "train_time_sec",
            "n_parameters", "epochs_trained", "simplicity_loc", "simplicity_rating", "transparency_rating",
            "rating_note", "error"]
    # completeness guard: an evaluated regression model must carry all five metrics; a failed one must carry its error
    reg_cols = ["test_RMSE", "test_MSE", "test_MAE", "test_R2", "test_Directional_Accuracy"]
    ok = df[(df.task == "regression") & df.test_RMSE.notna()]
    bad = ok[ok[reg_cols].isna().any(axis=1)]
    assert bad.empty, f"regression metrics incomplete for: {list(bad.model)}"
    nometric = df[(df.task == "regression") & df.test_RMSE.isna() & (df.error.fillna("") == "")]
    assert nometric.empty, f"models without metrics and without a recorded error: {list(nometric.model)}"
    return df[keep]
