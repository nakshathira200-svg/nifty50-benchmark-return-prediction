"""Train and save models. Uses train + validation data only (test is never loaded).

  python run_models.py                       # baselines + all 10 framework models
  python run_models.py --models baselines    # only baselines
  python run_models.py --models xgboost_reg keras_clf
  python run_models.py --models frameworks --standin --tag smoke_standin   # harness smoke test, NOT real results
"""
import argparse
from src.config import SEED
from src.models.registry import framework_model_names, baseline_model_names
from src.training import run_names


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["all"],
                   help="all | baselines | frameworks | explicit names such as xgboost_reg fastai_clf")
    p.add_argument("--feature-set", choices=["stationary", "all"], default="stationary")
    p.add_argument("--tag", default="main", help="output namespace; non-'main' runs are kept separate")
    p.add_argument("--standin", action="store_true",
                   help="use sklearn stand-ins instead of the real frameworks (harness test only)")
    p.add_argument("--seed", type=int, default=SEED)
    a = p.parse_args()
    if a.standin and a.tag == "main":
        p.error("--standin results must not be written to the 'main' tag; pass e.g. --tag smoke_standin")
    names = []
    for m in a.models:
        names += (baseline_model_names() + framework_model_names() if m == "all" else
                  baseline_model_names() if m == "baselines" else framework_model_names() if m == "frameworks" else [m])
    new = run_names(names, a.feature_set, a.tag, a.standin, a.seed)
    print("\n", new[["model", "implementation", "status", "train_time_sec", "error"]].to_string(index=False))
    failed = int((new.status == "failed").sum())
    print(f"\nSUMMARY: {len(new) - failed} trained, {failed} FAILED (details: outputs/results/run_status.csv, logs/)"
          if a.tag == "main" else f"\nSUMMARY ({a.tag}): {len(new) - failed} trained, {failed} failed")


if __name__ == "__main__":
    main()
