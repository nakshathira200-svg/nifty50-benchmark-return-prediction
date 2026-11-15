"""Training orchestration shared by run_models.py and each model's own CLI.
Uses ONLY train + validation data: the test file is never read here."""
from __future__ import annotations
import argparse, json, time, traceback
from pathlib import Path
from typing import List
import pandas as pd

from src.config import ROOT, SEED, run_dirs, set_global_seed
from src.data.prepare import build_bundle, save_preprocessing
from src.evaluation.metrics import regression_metrics, classification_metrics
from src.models.registry import (get_class, parse, framework_model_names, baseline_model_names, model_name)

STATUS_COLS = ["model", "framework", "task", "implementation", "status", "train_time_sec", "n_parameters",
               "epochs_trained", "best_params", "val_primary_metric", "error"]


def train_one(name: str, bundle, dirs: dict, standin: bool = False, seed: int = SEED) -> dict:
    info = parse(name)
    set_global_seed(seed)
    cls = get_class(name, standin=standin)
    model = cls(seed=seed)
    model.attach_context(scaler=bundle.scaler)
    reg = info["task"] == "regression"
    y_tr, y_va = (bundle.y_train_reg, bundle.y_val_reg) if reg else (bundle.y_train_clf, bundle.y_val_clf)
    t0 = time.perf_counter()
    model.fit(bundle.X_train, y_tr, bundle.X_val, y_va, bundle.features)
    elapsed = time.perf_counter() - t0
    pred = model.predict(bundle.X_val)                                  # validation predictions only
    vm = regression_metrics(y_va, pred) if reg else classification_metrics(y_va, pred)
    pd.DataFrame({"price_date": bundle.val_index, "y_true": y_va, "y_pred": pred}).to_csv(
        dirs["predictions"] / f"{name}_validation.csv", index=False)
    model.save(dirs["models"] / name)
    return {"model": name, "framework": info["framework"], "task": info["task"],
            "implementation": model.implementation, "status": "trained", "train_time_sec": round(elapsed, 3),
            "n_parameters": model.n_params_, "epochs_trained": model.epochs_trained_,
            "best_params": json.dumps(model.best_params_), "val_primary_metric":
            vm["RMSE"] if reg else vm["ROC_AUC"], "error": ""}


def run_names(names: List[str], feature_set: str = "stationary", tag: str = "main", standin: bool = False,
              seed: int = SEED) -> pd.DataFrame:
    dirs = run_dirs(tag)
    bundle = build_bundle(feature_set, include_test=False)              # test split is NOT loaded
    save_preprocessing(bundle, dirs["models"], feature_set)
    status_path = dirs["results"] / "run_status.csv"
    old = pd.read_csv(status_path) if status_path.exists() else pd.DataFrame(columns=STATUS_COLS)
    rows = []
    for name in names:
        print(f"[train] {name} ...", flush=True)
        try:
            row = train_one(name, bundle, dirs, standin=standin, seed=seed)
            stale = ROOT / "logs" / f"{tag}_{name}_error.log"
            if stale.exists():
                stale.unlink()                                          # an old failure no longer applies
            print(f"        ok  {row['train_time_sec']}s  impl={row['implementation']}")
        except Exception as e:                                          # keep the REAL error; never fake a result
            info = parse(name)
            tb = traceback.format_exc()
            (ROOT / "logs").mkdir(exist_ok=True)
            (ROOT / "logs" / f"{tag}_{name}_error.log").write_text(tb)
            row = {"model": name, "framework": info["framework"], "task": info["task"], "implementation": "none",
                   "status": "failed", "train_time_sec": None, "n_parameters": None, "epochs_trained": None,
                   "best_params": "", "val_primary_metric": None, "error": f"{type(e).__name__}: {str(e)[:300]}"}
            print(f"        FAILED  {row['error']}")
        rows.append(row)
    new = pd.DataFrame(rows, columns=STATUS_COLS)
    merged = pd.concat([old[~old["model"].isin(new["model"])], new], ignore_index=True)
    order = {n: i for i, n in enumerate(baseline_model_names() + framework_model_names())}
    merged = merged.sort_values("model", key=lambda s: s.map(order)).reset_index(drop=True)
    merged.to_csv(status_path, index=False)
    return new


def cli_main(framework: str):
    """Entry point for running ONE framework on its own: python -m src.models.<fw>_model --task both"""
    p = argparse.ArgumentParser(description=f"Train {framework} model(s) on train/validation data")
    p.add_argument("--task", choices=["regression", "classification", "both"], default="both")
    p.add_argument("--feature-set", choices=["stationary", "all"], default="stationary")
    p.add_argument("--tag", default="main")
    p.add_argument("--seed", type=int, default=SEED)
    a = p.parse_args()
    tasks = ["regression", "classification"] if a.task == "both" else [a.task]
    run_names([model_name(framework, t) for t in tasks], a.feature_set, a.tag, False, a.seed)
