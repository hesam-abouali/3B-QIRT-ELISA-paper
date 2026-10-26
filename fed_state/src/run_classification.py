#!/usr/bin/env python3
"""Run the fed-state classification: nested leave-one-rat-out cross-validation of the 81 combinations
(9 feature selectors x 9 classifiers) on the classification matrix (Supplementary Information, Section S10).

Input   data/features/fedstate_X.csv, fedstate_y.csv, fedstate_meta.csv
Output  results_summary.csv, per_fold_details.csv, confusion_matrices.csv, selected_features.csv

The constant column of the matrix is dropped, which leaves 160 candidate features. The number of features
is chosen in the inner loop from {3, 5, 10, 20, 30}, restricted to values smaller than the number of
training rats (8), so k is 3 or 5.

LogReg_L1 is fitted with the saga solver without a random_state, so its nine combinations can change from
run to run. The comparison with data/classification/ therefore covers the other 72 combinations.

    python src/run_classification.py [--selector NAME] [--classifier NAME] [--output-dir DIR]
"""
import argparse
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

from nested_cv import Config, NestedCVPipeline, save_results

# scikit-learn repeats warnings about the small inner folds (for example, a class with only two
# training rats in the stratified 3-fold of permutation importance); they are not shown
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT / "data" / "classification"
COMPARED = ("results_summary", "per_fold_details", "confusion_matrices", "selected_features")
NON_DETERMINISTIC = "LogReg_L1"


def run(selectors, classifiers, output_dir):
    """Nested cross-validation of every selector-classifier combination."""
    combos = [(s, c) for s in selectors for c in classifiers]
    print("=" * 70)
    print("NESTED CV — QIRT-ELISA FED-STATE (n=9)")
    print("=" * 70)
    print(f"Selectors  : {selectors}")
    print(f"Classifiers: {classifiers}")
    print(f"Combinations: {len(combos)}")
    print(f"Started    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    X = pd.read_csv(Config.X_PATH)
    y = pd.read_csv(Config.Y_PATH)["rat_type"]
    meta_df = pd.read_csv(Config.META_PATH)
    rat_ids = meta_df["animal_id"].tolist()
    assert list(meta_df["rat_type"]) == list(y), "Row mismatch between meta_df['rat_type'] and y!"
    print(f"\nData loaded: X {X.shape}, y {y.shape}")
    print(f"   Classes: {y.value_counts().to_dict()}")
    print(f"   Rat IDs: {rat_ids}")

    # Drop zero-variance columns (constant features distort rank-based selectors)
    zero_var = X.columns[X.var() == 0].tolist()
    if zero_var:
        print(f"\nDropping {len(zero_var)} zero-variance column(s): {zero_var}")
        X = X.drop(columns=zero_var)

    feature_names = list(X.columns)
    results_list = []
    for idx, (sel_name, clf_name) in enumerate(combos, 1):
        print(f"\n[{idx}/{len(combos)}] {sel_name} + {clf_name}")
        t0 = datetime.now()
        pipeline = NestedCVPipeline(sel_name, clf_name, Config.K_GRID)
        res = pipeline.run(X, y, feature_names, rat_ids)
        elapsed = (datetime.now() - t0).total_seconds()
        print(f"  -> Macro-F1={res['macro_f1']:.4f}  Acc={res['accuracy']:.4f}  "
              f"mean_k={res['mean_k']:.1f}  ({elapsed:.0f}s)")
        results_list.append(res)

    save_results(results_list, output_dir)
    best = max(results_list, key=lambda x: x["macro_f1"])
    print("\n" + "=" * 70)
    print(f"BEST: {best['selector']} + {best['classifier']} — Macro-F1 = {best['macro_f1']:.4f}")
    print("=" * 70)


def compare_with_reference(output_dir):
    """Report whether the deterministic combinations match the paper's run in data/classification/."""
    print(f"\nComparison with {REFERENCE_DIR.relative_to(ROOT)}/, without {NON_DETERMINISTIC}:")
    for name in COMPARED:
        new = pd.read_csv(output_dir / f"{name}.csv")
        ref = pd.read_csv(REFERENCE_DIR / f"{name}.csv")
        combos = new[["selector", "classifier"]].drop_duplicates()
        ref = ref.merge(combos, on=["selector", "classifier"])          # the combinations of this run
        new, ref = (d[d["classifier"] != NON_DETERMINISTIC].sort_values(list(d.columns)).reset_index(drop=True)
                    for d in (new, ref))
        n = len(ref.drop_duplicates(["selector", "classifier"]))
        print(f"  {name}: {'identical' if new.equals(ref) else 'DIFFERENT'} ({n} combinations)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selector", default="all", choices=Config.ALL_SELECTORS + ["all"],
                        help="selector(s) to run (default: all 9)")
    parser.add_argument("--classifier", default="all", choices=Config.ALL_CLASSIFIERS + ["all"],
                        help="classifier(s) to run (default: all 9)")
    parser.add_argument("--output-dir", default=Config.OUTPUT_DIR, help="folder for the result files")
    args = parser.parse_args()

    selectors = Config.ALL_SELECTORS if args.selector == "all" else [args.selector]
    classifiers = Config.ALL_CLASSIFIERS if args.classifier == "all" else [args.classifier]
    output_dir = Path(args.output_dir)
    run(selectors, classifiers, output_dir)
    compare_with_reference(output_dir)


if __name__ == "__main__":
    main()
