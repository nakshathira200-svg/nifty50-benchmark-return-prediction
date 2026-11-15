"""Load the saved models, evaluate on validation + test, write comparison CSVs and figures.

  python run_evaluation.py                      # real results (tag main)
  python run_evaluation.py --tag smoke_standin  # evaluate the stand-in smoke-test run
"""
import argparse
import pandas as pd
from src.evaluation.evaluate import evaluate_run
from src.evaluation.plots import make_all
from src.evaluation.report import write_results_doc, publish_deliverables


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="main")
    p.add_argument("--repeats", type=int, default=5, help="permutation-importance repeats")
    a = p.parse_args()
    res = evaluate_run(a.tag, a.repeats)
    make_all(res, a.tag)
    if a.tag == "main":
        write_results_doc(res, a.tag)          # docs/RESULTS.md (only for real results)
        publish_deliverables(res)              # copies to deliverables/model_results and deliverables/graphs/part2
    pd.set_option("display.width", 250, "display.max_columns", 30)
    cmp = res["comparison"]
    print("\n=== Regression (test) ==="); print(res["regression_test_metrics"].round(5).to_string(index=False))
    print("\n=== Classification (test) ==="); print(res["classification_test_metrics"].round(4).to_string(index=False))
    not_ok = cmp[~cmp.status.isin(["evaluated"])]
    if len(not_ok):
        print("\n=== Models with NO results (nothing fabricated) ===")
        print(not_ok[["model", "status", "error"]].to_string(index=False))
    print(f"\nTables: {res['dirs']['results']}\nFigures: {res['dirs']['graphs']}")


if __name__ == "__main__":
    main()
