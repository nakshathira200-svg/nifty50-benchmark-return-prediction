"""Model registry for Part 3."""
from src.models.base import BaseModel
from src.models.tabnet_model import TabNetRegressorModel, TabNetClassifierModel
from src.models.xgboost_model import XGBoostRegressorModel, XGBoostClassifierModel
from src.models.decision_tree_model import DecisionTreeRegressorModel, DecisionTreeClassifierModel
from src.models.ensemble_model import (
    WeightedEnsembleRegressor,
    WeightedEnsembleClassifier,
    StackingEnsembleRegressor,
)

__all__ = [
    "BaseModel",
    "TabNetRegressorModel",
    "TabNetClassifierModel",
    "XGBoostRegressorModel",
    "XGBoostClassifierModel",
    "DecisionTreeRegressorModel",
    "DecisionTreeClassifierModel",
    "WeightedEnsembleRegressor",
    "WeightedEnsembleClassifier",
    "StackingEnsembleRegressor",
]
