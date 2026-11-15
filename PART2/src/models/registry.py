"""Model registry: 10 framework models (5 frameworks x {regression, classification}) + baselines."""
from __future__ import annotations
import importlib
from typing import Dict, List
from src.config import FRAMEWORKS

_SHORT = {"regression": "reg", "classification": "clf"}
_TASK = {v: k for k, v in _SHORT.items()}
_MODULES = {"xgboost": ("xgboost_model", "XGBoost"), "keras": ("keras_model", "Keras"),
            "fastai": ("fastai_model", "FastAI"), "tabnet": ("tabnet_model", "TabNet"),
            "flash": ("flash_model", "Flash")}


def model_name(framework: str, task: str) -> str:
    return f"{framework}_{_SHORT[task]}"


def framework_model_names() -> List[str]:
    return [model_name(f, t) for f in FRAMEWORKS for t in ("regression", "classification")]


def baseline_model_names() -> List[str]:
    from src.models.baselines import REG_BASELINES, CLF_BASELINES
    return [f"baseline_{k}_reg" for k in REG_BASELINES] + [f"baseline_{k}_clf" for k in CLF_BASELINES]


def parse(name: str) -> Dict[str, str]:
    if name.startswith("baseline_"):
        _, rest = name.split("_", 1); kind, short = rest.rsplit("_", 1)
        return {"framework": "baseline", "task": _TASK[short], "kind": kind}
    fw, short = name.rsplit("_", 1)
    return {"framework": fw, "task": _TASK[short]}


def get_class(name: str, standin: bool = False):
    info = parse(name)
    if info["framework"] == "baseline":
        from src.models.baselines import make_baseline
        return make_baseline(info["kind"], info["task"])
    if standin:
        from src.models.standins import make_standin
        return make_standin(info["framework"], info["task"])
    module, prefix = _MODULES[info["framework"]]
    mod = importlib.import_module(f"src.models.{module}")
    return getattr(mod, f"{prefix}{'Regressor' if info['task'] == 'regression' else 'Classifier'}Model")
