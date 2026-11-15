"""Evaluation metrics for regression and classification tasks.
All formulas strictly adhere to standard quantitative finance benchmarks.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Computes MAE, MSE, RMSE, R², and Directional Accuracy.
    Directional Accuracy: proportion of trading days where sign(y_pred) == sign(y_true),
    with 'positive return' defined strictly as > 0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    # Directional accuracy: sign agreement
    dir_acc = float(np.mean((y_pred > 0) == (y_true > 0)))

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Directional_Accuracy": dir_acc,
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> dict:
    """Computes Accuracy, Precision, Recall, F1, ROC-AUC, 95% Confidence Interval half-width,
    and confusion matrix components (TN, FP, FN, TP).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_pred, dtype=float)
    y_class = (y_prob >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_class))
    prec = float(precision_score(y_true, y_class, zero_division=0))
    rec = float(recall_score(y_true, y_class, zero_division=0))
    f1 = float(f1_score(y_true, y_class, zero_division=0))

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = 0.5

    cm = confusion_matrix(y_true, y_class, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # Wilson / normal approximation 95% binomial confidence interval half-width: 1.96 * sqrt(p*(1-p)/n)
    n = len(y_true)
    half_width = float(1.96 * np.sqrt((acc * (1.0 - acc)) / max(n, 1)))

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC_AUC": auc,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Accuracy_CI95_halfwidth": half_width,
    }
