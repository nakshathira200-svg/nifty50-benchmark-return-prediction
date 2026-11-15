"""Justify the default feature set using TRAIN -> VALIDATION only (the test split is never loaded).
Compares Part 1's full 21 predictors (incl. raw price levels) with the 17 stationary predictors, using simple
sklearn models. Output: outputs/results/feature_set_ablation_validation.csv"""
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from src.config import SEED, run_dirs
from src.data.prepare import build_bundle
from src.evaluation.metrics import regression_metrics, classification_metrics

rows = []
for fs in ["all", "stationary"]:
    b = build_bundle(fs, include_test=False)
    for name, est in [("ridge(alpha=100)", Ridge(alpha=100)),
                      ("gradient_boosting", GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.03,
                                                                      subsample=.8, random_state=SEED))]:
        m = est.fit(b.X_train, b.y_train_reg); mt = regression_metrics(b.y_val_reg, m.predict(b.X_val))
        rows.append({"feature_set": fs, "n_features": len(b.features), "task": "regression", "model": name,
                     "val_RMSE": mt["RMSE"], "val_R2": mt["R2"], "val_directional_accuracy": mt["Directional_Accuracy"]})
    for name, est in [("logistic(C=0.1)", LogisticRegression(C=0.1, max_iter=2000, random_state=SEED)),
                      ("gradient_boosting", GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.03,
                                                                       subsample=.8, random_state=SEED))]:
        m = est.fit(b.X_train, b.y_train_clf); mt = classification_metrics(b.y_val_clf, m.predict_proba(b.X_val)[:, 1])
        rows.append({"feature_set": fs, "n_features": len(b.features), "task": "classification", "model": name,
                     "val_Accuracy": mt["Accuracy"], "val_ROC_AUC": mt["ROC_AUC"]})
df = pd.DataFrame(rows)
df.to_csv(run_dirs("main")["results"] / "feature_set_ablation_validation.csv", index=False)
print(df.round(4).to_string(index=False))
