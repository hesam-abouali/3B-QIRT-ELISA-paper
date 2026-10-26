#!/usr/bin/env python3
"""Run the nested cross-validation of the 81 combinations for the 20 label assignments (GTT classification).

Each job is one label assignment with one selector and its nine classifiers (180 jobs; about 1.5 h with
6 workers, most of it in the mutual-information and mRMR jobs). Every finished job is saved in
outputs/classification/jobs/, so an interrupted run continues where it stopped. The jobs are then combined into

    scores_by_assignment.csv            macro-F1, accuracy, Brier score, log loss, mean predictive confidence
                                        and mean k of every combination and assignment
    predictions_by_assignment.csv       held-out label, prediction, probability and selected k of every rat
    selected_features_true_labels.csv   features selected in each held-out fit with the true labels

and summarized (summarize.py: perfect combinations, exact label test, feature recurrence).

The results reported in the paper are saved in data/classification/ (run on Linux x86-64, Python 3.10, NumPy
2.0.2, SciPy 1.13.1, pandas 2.3.3, scikit-learn 1.6.1); summarize.py reproduces the reported numbers from them.
In other environments a few combinations can select slightly different features.

    python src/run_classification.py [--workers 6] [--true-labels-only | --selector NAME] [--output-dir DIR]
"""
import os
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                  "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"   # single-threaded linear algebra, as in the reported run
import argparse
import json
import multiprocessing as mp
import time
import warnings
from pathlib import Path

import pandas as pd

import nested_cv as cv
import summarize

ROOT = Path(__file__).resolve().parents[1]
X_CSV = ROOT / "data" / "features" / "gtt_X.csv"
META_CSV = ROOT / "data" / "features" / "gtt_meta.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "classification"
SLOW = ("mutual_info", "mrmr")


def compute(job):
    """Run one job (assignment, selector) and save it; skip it if it is already saved."""
    assignment, selector, jobs_dir = job
    path = Path(jobs_dir) / f"assignment{assignment:02d}_{selector}.json"
    if path.exists():
        return job, "saved earlier", 0.0
    warnings.filterwarnings("ignore")
    start = time.time()
    X9, names, scored_pos, oz_pos = cv.load_matrix(X_CSV, META_CSV)
    pipelines = cv.run_selector(X9, names, scored_pos, oz_pos, selector, cv.assignment_labels(assignment))
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"assignment": assignment, "selector": selector, "pipelines": pipelines}) + "\n")
    temporary.replace(path)
    return job, "done", time.time() - start


def combine(jobs_dir, assignments, selectors):
    """The three result tables from the saved jobs."""
    scores, predictions, features = [], [], []
    for assignment in assignments:
        labels = cv.assignment_labels(assignment)
        diabetic = " ".join(a for a, y in zip(cv.SCORED, labels) if y == 1)
        for selector in selectors:
            doc = json.loads((Path(jobs_dir) / f"assignment{assignment:02d}_{selector}.json").read_text())
            for p in doc["pipelines"]:
                key = {"assignment": assignment, "selector": selector, "classifier": p["classifier"]}
                scores.append({**key, "diabetic_rats": diabetic, "true_labels": assignment == cv.TRUE_ASSIGNMENT,
                               **{m: p[m] for m in ("macro_f1", "accuracy", "brier_score", "log_loss",
                                                    "mean_predictive_confidence", "mean_selected_k")}})
                for label, prediction, probability, fold in zip(labels, p["predictions"],
                                                                 p["probabilities_diabetic"], p["folds"]):
                    predictions.append({**key, "animal_id": fold["animal_id"], "label": int(label),
                                        "prediction": prediction, "probability_diabetic": probability,
                                        "selected_k": fold["selected_k"],
                                        **{f"inner_correct_k{k}": fold["inner_correct_by_k"][str(k)] for k in cv.K_GRID}})
                    if assignment == cv.TRUE_ASSIGNMENT:
                        features.append({"selector": selector, "classifier": p["classifier"],
                                         "animal_id": fold["animal_id"], "selected_k": fold["selected_k"],
                                         "selected_features": "|".join(fold["selected_features"])})
    return pd.DataFrame(scores), pd.DataFrame(predictions), pd.DataFrame(features)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=4, help="parallel processes (default 4)")
    parser.add_argument("--true-labels-only", action="store_true",
                        help="run only the true labels (9 jobs, about 5 min); no label test")
    parser.add_argument("--selector", choices=cv.SELECTORS,
                        help="run only this selector, with its nine classifiers, on the true labels (about 1 min)")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help="folder for the jobs and result tables")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    jobs_dir = output_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    true_only = args.true_labels_only or args.selector is not None
    assignments = [cv.TRUE_ASSIGNMENT] if true_only else list(range(len(cv.ASSIGNMENTS)))
    selectors = [args.selector] if args.selector else cv.SELECTORS
    jobs = [(a, s, str(jobs_dir)) for a in assignments for s in selectors]
    jobs.sort(key=lambda j: (j[0] != cv.TRUE_ASSIGNMENT, j[1] not in SLOW))   # true labels and slow jobs first
    print(f"{len(jobs)} jobs ({len(assignments)} label assignments x {len(selectors)} selectors), "
          f"{args.workers} workers")
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for n, (job, status, seconds) in enumerate(pool.imap_unordered(compute, jobs), 1):
            print(f"  {n}/{len(jobs)} assignment {job[0]:2d} {job[1]:<13} {status} ({seconds:.0f} s)", flush=True)

    scores, predictions, features = combine(jobs_dir, sorted(assignments), selectors)
    scores.to_csv(output_dir / "scores_by_assignment.csv", index=False)
    predictions.to_csv(output_dir / "predictions_by_assignment.csv", index=False)
    features.to_csv(output_dir / "selected_features_true_labels.csv", index=False)
    print(f"  Saved: {output_dir / 'scores_by_assignment.csv'}")
    print(f"  Saved: {output_dir / 'predictions_by_assignment.csv'}")
    print(f"  Saved: {output_dir / 'selected_features_true_labels.csv'}\n")
    summarize.summarize(output_dir, output_dir)


if __name__ == "__main__":
    main()
