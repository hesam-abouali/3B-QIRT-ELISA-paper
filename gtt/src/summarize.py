#!/usr/bin/env python3
"""Summarize the GTT classification results.

1. True labels: the combinations that classify all six rats correctly (macro-F1 = 1) and the mean macro-F1
   of the 81 combinations.
2. Exact label test: the best macro-F1 of the 81 combinations is found for each of the 20 ways of labelling
   three of the six rats as diabetic; p is the fraction of the 20 assignments (the true one included) whose
   best macro-F1 reaches that of the true labels.
3. Feature recurrence: the number of held-out fits of the perfect combinations (6 per combination) that
   selected each feature, in total and by held-out rat.

Input   scores_by_assignment.csv, selected_features_true_labels.csv   (default: data/classification/, the
                                                                       reported run)
Output  assignment_maxima.csv            best macro-F1 and number of perfect combinations of each assignment
        feature_recurrence_perfect.csv   feature recurrence among the held-out fits of the perfect combinations

    python src/summarize.py [--results DIR] [--output-dir DIR]
"""
import argparse
from pathlib import Path

import pandas as pd

from nested_cv import ASSIGNMENTS, CLASSIFIERS, SCORED, SELECTORS, TRUE_ASSIGNMENT

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "data" / "classification"
DEFAULT_OUTPUT = ROOT / "outputs" / "classification" / "reported"


def true_label_scores(scores):
    """The 81 combinations with the true labels, in selector and classifier order."""
    true = scores[scores.assignment == TRUE_ASSIGNMENT].copy()
    true["order"] = true.selector.map(SELECTORS.index) * len(CLASSIFIERS) + true.classifier.map(CLASSIFIERS.index)
    return true.sort_values("order").drop(columns="order").reset_index(drop=True)


def assignment_maxima(scores):
    """Best macro-F1 and number of perfect combinations for each label assignment."""
    return (scores.assign(perfect=scores.macro_f1 == 1)
            .groupby("assignment")
            .agg(diabetic_rats=("diabetic_rats", "first"), true_labels=("true_labels", "first"),
                 max_macro_f1=("macro_f1", "max"), perfect_combinations=("perfect", "sum"))
            .reset_index())


def label_test(maxima):
    """Exact p-value and the assignments whose best macro-F1 reaches that of the true labels."""
    observed = maxima.loc[maxima.assignment == TRUE_ASSIGNMENT, "max_macro_f1"].iloc[0]
    reaching = maxima[maxima.max_macro_f1 >= observed - 1e-9]
    return len(reaching) / len(maxima), reaching.assignment.tolist()


def feature_recurrence(features, perfect):
    """Fits of the perfect combinations that selected each feature, in total and by held-out rat."""
    keep = features.merge(perfect[["selector", "classifier"]], on=["selector", "classifier"])
    rows = keep.assign(feature=keep.selected_features.str.split("|")).explode("feature")
    table = rows.pivot_table(index="feature", columns="animal_id", values="selector", aggfunc="size", fill_value=0)
    table = table.reindex(columns=list(SCORED), fill_value=0)
    table.columns = [f"held_out_{a}" for a in table.columns]
    table.insert(0, "fits", table.sum(axis=1))
    return table.reset_index().sort_values(["fits", "feature"], ascending=[False, True]).reset_index(drop=True)


def summarize(results_dir, output_dir):
    results_dir, output_dir = Path(results_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(results_dir / "scores_by_assignment.csv", float_precision="round_trip")
    features = pd.read_csv(results_dir / "selected_features_true_labels.csv")

    true = true_label_scores(scores)
    perfect = true[true.macro_f1 == 1]
    print(f"True labels: {len(perfect)} of {len(true)} combinations classify all six rats correctly; "
          f"mean macro-F1 {true.macro_f1.mean():.3f}")
    recurrence = feature_recurrence(features, perfect)
    recurrence.to_csv(output_dir / "feature_recurrence_perfect.csv", index=False)
    print(f"  Most selected features ({6 * len(perfect)} held-out fits of the perfect combinations): "
          + ", ".join(f"{f} {n}" for f, n in recurrence[["feature", "fits"]].head(4).values))

    if scores.assignment.nunique() == len(ASSIGNMENTS):
        maxima = assignment_maxima(scores)
        maxima.to_csv(output_dir / "assignment_maxima.csv", index=False)
        p, reaching = label_test(maxima)
        print(f"Exact label test: {len(reaching)} of {len(maxima)} assignments reach the best macro-F1 of the true "
              f"labels (assignments {', '.join(map(str, reaching))}); p = {p:.2f}")
    else:
        print("Exact label test: needs all 20 label assignments")
    print(f"  Saved in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(DEFAULT_RESULTS), help="folder with the result tables")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help="folder for the two summary CSVs")
    args = parser.parse_args()
    summarize(args.results, args.output_dir)


if __name__ == "__main__":
    main()
