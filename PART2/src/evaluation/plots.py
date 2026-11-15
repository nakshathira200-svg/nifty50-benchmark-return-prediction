"""All comparison figures. Non-main run tags are watermarked so stand-in results can never be mistaken for real ones."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

PALETTE = {"baseline": "#9e9e9e", "xgboost": "#1f77b4", "keras": "#ff7f0e", "fastai": "#2ca02c",
           "tabnet": "#d62728", "flash": "#9467bd"}


def _mark(fig, tag):
    if tag != "main":
        fig.text(0.5, 0.995, "SMOKE TEST: sklearn stand-ins, NOT the real frameworks", ha="center", va="top",
                 color="red", fontsize=10, weight="bold")


def _colors(models):
    return [PALETTE.get(m.split("_")[0] if not m.startswith("baseline") else "baseline", "#555") for m in models]


def _short(m):
    return m.replace("baseline_", "b:").replace("_reg", "").replace("_clf", "")


def _save(fig, dirs, name):
    fig.tight_layout(rect=(0, 0, 1, 0.90)); fig.savefig(dirs["graphs"] / name, dpi=150); plt.close(fig)


def _zoom_barh(ax, labels, values, colors, lo, hi, xerr=None):
    """Horizontal bars on a zoomed axis; bars beyond the window are clipped and annotated with their true value."""
    v = np.asarray(values, float); c = np.clip(v, lo, hi)
    ax.barh(labels, c, color=colors, xerr=xerr, capsize=3 if xerr is not None else 0, zorder=2)
    ax.set_xlim(lo, hi); ax.invert_yaxis()
    for i, (raw, cl) in enumerate(zip(v, c)):
        if raw != cl:
            ax.text(cl, i, f"off-scale: {raw:.4f} ", va="center", ha="right" if raw > hi else "left", fontsize=7,
                    color="white" if raw > hi else "black", zorder=3)


def regression_comparison(res, tag):
    df = res["regression_test_metrics"]
    if df.empty:
        return
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    d = df.sort_values("RMSE")
    lo, hi = d.RMSE.min() * 0.985, np.median(d.RMSE) * 1.04
    _zoom_barh(ax[0], [_short(m) for m in d.model], d.RMSE, _colors(d.model), lo, hi)
    ax[0].set(title="Test RMSE (lower is better; zoomed)", xlabel="RMSE")
    if (d.model == "baseline_train_mean_reg").any():
        ax[0].axvline(d.loc[d.model == "baseline_train_mean_reg", "RMSE"].iloc[0], color="k", ls="--", lw=.8)
    d = df.sort_values("R2", ascending=False)
    lo, hi = max(d.R2.min(), -0.06), max(d.R2.max(), 0) + 0.005
    _zoom_barh(ax[1], [_short(m) for m in d.model], d.R2, _colors(d.model), lo, hi)
    ax[1].axvline(0, color="k", lw=.8); ax[1].set(title="Test R² (0 = no better than the mean; zoomed)", xlabel="R²")
    d = df.sort_values("Directional_Accuracy", ascending=False)
    n = len(res["bundle"].y_test_reg); hw = 1.96 * np.sqrt(.25 / n)
    ax[2].axvspan(.5 - hw, .5 + hw, color="k", alpha=.12, zorder=0, label="±95% noise band around 50%")
    _zoom_barh(ax[2], [_short(m) for m in d.model], d.Directional_Accuracy, _colors(d.model), .40, .65)
    ax[2].set(title="Test directional accuracy", xlabel="share of days"); ax[2].legend(fontsize=8, loc="lower right")
    fig.suptitle("Regression: target_next_day_return", y=0.96); _mark(fig, tag)
    _save(fig, res["dirs"], "regression_comparison.png")


def classification_comparison(res, tag):
    df = res["classification_test_metrics"]
    if df.empty:
        return
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    n = len(res["bundle"].y_test_clf)
    d = df.sort_values("Accuracy", ascending=False)
    _zoom_barh(ax[0], [_short(m) for m in d.model], d.Accuracy, _colors(d.model), .40, .65, xerr=d.Accuracy_CI95_halfwidth)
    ax[0].axvline(1 - res["bundle"].y_test_clf.mean(), color="k", ls=":", lw=.8)
    ax[0].axvline(res["bundle"].y_test_clf.mean(), color="k", ls="--", lw=.8)
    ax[0].set(title="Test accuracy ±95% CI (dashed: always-up)", xlabel="accuracy")
    d = df.sort_values("F1", ascending=False)
    ax[1].barh([_short(m) for m in d.model], d.F1, color=_colors(d.model)); ax[1].invert_yaxis(); ax[1].set(title="Test F1", xlabel="F1")
    d = df.sort_values("ROC_AUC", ascending=False)
    ax[2].barh([_short(m) for m in d.model], d.ROC_AUC, color=_colors(d.model)); ax[2].invert_yaxis()
    ax[2].axvline(.5, color="k", lw=.8); ax[2].set(title="Test ROC-AUC (0.5 = chance)", xlabel="AUC"); ax[2].set_xlim(.4, .65)
    fig.suptitle("Classification: target_next_day_up", y=0.96); _mark(fig, tag); _save(fig, res["dirs"], "classification_comparison.png")


def roc_curves(res, tag):
    P = {k: v for k, v in res["preds"].items() if v["task"] == "classification" and len(np.unique(v["y_pred"])) > 2}
    if not P:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    for k, v in P.items():
        fpr, tpr, _ = roc_curve(v["y_true"], v["y_pred"])
        ax.plot(fpr, tpr, label=_short(k), color=_colors([k])[0], lw=1.2)
    ax.plot([0, 1], [0, 1], "k--", lw=.8); ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves (test)")
    ax.legend(fontsize=8); _mark(fig, tag); _save(fig, res["dirs"], "roc_curves.png")


def confusion_matrices(res, tag):
    df = res["classification_test_metrics"]
    if df.empty:
        return
    n = len(df); cols = min(4, n); rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, r) in zip(axes.ravel(), df.iterrows()):
        ax.axis("on"); cm = np.array([[r.TN, r.FP], [r.FN, r.TP]])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["down", "up"], yticklabels=["down", "up"],
               xlabel="predicted", ylabel="actual", title=_short(r.model), )
    fig.suptitle("Confusion matrices (test)", y=0.96); _mark(fig, tag); _save(fig, res["dirs"], "confusion_matrices.png")


def pred_vs_actual(res, tag):
    P = {k: v for k, v in res["preds"].items() if v["task"] == "regression"}
    if not P:
        return
    n = len(P); cols = min(4, n); rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.8 * cols, 3.5 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (k, v) in zip(axes.ravel(), P.items()):
        ax.axis("on"); ax.scatter(v["y_true"], v["y_pred"], s=4, alpha=.5, color=_colors([k])[0])
        lim = np.abs(v["y_true"]).max(); ax.plot([-lim, lim], [-lim, lim], "k--", lw=.6)
        ax.axhline(0, color="gray", lw=.4); ax.axvline(0, color="gray", lw=.4)
        ax.set(title=_short(k), xlabel="actual return", ylabel="predicted return")
    fig.suptitle("Predicted vs actual next-day return (test). Dashed line = perfect prediction", y=0.96)
    _mark(fig, tag); _save(fig, res["dirs"], "regression_pred_vs_actual.png")


def training_time(res, tag):
    df = res["comparison"]; df = df[df.train_time_sec.notna()].copy()
    if df.empty:
        return
    df["train_time_sec"] = df["train_time_sec"].astype(float); df = df.sort_values("train_time_sec")
    fig, ax = plt.subplots(figsize=(9, max(4, .32 * len(df))))
    ax.barh([_short(m) for m in df.model], df.train_time_sec.clip(lower=1e-3), color=_colors(df.model))
    ax.set_xscale("log"); ax.set(xlabel="training time (s, log scale)", title="Training time (CPU, train+val early stopping)")
    _mark(fig, tag); _save(fig, res["dirs"], "training_time.png")


def feature_importance(res, tag):
    for name, imp in res["importances"].items():
        if name.split("_")[0] == "baseline" and ("train_mean" in name or "zero" in name or "majority" in name):
            continue
        has_native = imp["native"].notna().any()
        fig, axes = plt.subplots(1, 2 if has_native else 1, figsize=(12 if has_native else 6.5, 5.2), squeeze=False)
        d = imp.sort_values("permutation_mean")
        axes[0, 0].barh(d.feature, d.permutation_mean, xerr=d.permutation_std, color=_colors([name])[0])
        axes[0, 0].set(title="Permutation importance (validation)\nloss increase when shuffled", xlabel="Δ loss")
        if has_native:
            d = imp.sort_values("native")
            axes[0, 1].barh(d.feature, d.native, color=_colors([name])[0])
            axes[0, 1].set(title="Native importance (model's own)", xlabel="normalised")
        fig.suptitle(_short(name), y=0.96); _mark(fig, tag); _save(fig, res["dirs"], f"feature_importance_{name}.png")


def training_curves(res, tag, models_dir: Path):
    for p in sorted(Path(models_dir).glob("*/history.json")):
        h = json.loads(p.read_text())
        if not h:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        for k, v in h.items():
            ax.plot(v, label=k)
        ax.set(title=f"Training curves: {p.parent.name}", xlabel="epoch", ylabel="loss"); ax.legend()
        _mark(fig, tag); _save(fig, res["dirs"], f"training_curve_{p.parent.name}.png")


def make_all(res, tag):
    for fn in (regression_comparison, classification_comparison, roc_curves, confusion_matrices, pred_vs_actual,
               training_time, feature_importance):
        fn(res, tag)
    training_curves(res, tag, res["dirs"]["models"])
