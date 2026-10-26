#!/usr/bin/env python3
"""Fold-specific SHAP values of the fed-state recipe mRMR + LogReg_L2 for each held-out rat (main-text Fig. 4).

For each outer fold of the nested cross-validation, L2 logistic regression is refitted on the eight
training rats with the features that mRMR selected in that fold, read from the saved selected_features
file. Kernel SHAP then explains the class probabilities of the held-out rat, with the eight training rats
of the fold as the background. For every rat, the script checks that the base value E plus the sum of the
attributions equals the predicted probability F of its true class.

Input   data/features/fedstate_X.csv, fedstate_y.csv, fedstate_meta.csv
        data/classification/selected_features.csv
Output  shap_values.csv        SHAP value of every selected feature for each class, per rat
        shap_base_values.csv   E and F of the true class, per rat

The outputs are compared with data/shap/.

    python src/shap_values.py [--selected-features CSV] [--output-dir DIR]
"""
from __future__ import annotations

import argparse
import filecmp
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "features"
DEFAULT_SELECTED = ROOT / "data" / "classification" / "selected_features.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "shap"
REFERENCE_DIR = ROOT / "data" / "shap"
CLASS_ORDER = ["healthy", "pre-diabetic", "diabetic"]


def compute_shap(selected_features_csv: Path, output_dir: Path) -> None:
    x = pd.read_csv(FEATURES / "fedstate_X.csv")
    y = pd.read_csv(FEATURES / "fedstate_y.csv").iloc[:, 0].to_numpy()
    meta = pd.read_csv(FEATURES / "fedstate_meta.csv")
    selected = pd.read_csv(selected_features_csv)
    selected = selected[
        (selected["selector"] == "mrmr")
        & (selected["classifier"] == "LogReg_L2")
    ].copy()

    y_encoded = np.array([CLASS_ORDER.index(label) for label in y], dtype=int)
    shap_rows: list[dict] = []
    baseline_rows: list[dict] = []

    for fold in range(1, len(x) + 1):
        test_index = fold - 1
        train_index = [i for i in range(len(x)) if i != test_index]
        features = selected.loc[selected["fold"] == fold, "feature"].tolist()
        if not features:
            raise ValueError(f"Fold {fold} has no selected mRMR features")

        scaler = StandardScaler()
        x_train = scaler.fit_transform(
            x.iloc[train_index][features].to_numpy(dtype=float)
        )
        x_test = scaler.transform(
            x.iloc[[test_index]][features].to_numpy(dtype=float)
        )
        y_train = y_encoded[train_index]

        model = LogisticRegression(
            penalty="l2",
            C=1.0,
            max_iter=5000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=0,
        )
        model.fit(x_train, y_train)

        background = x_train  # all eight training rats of the fold
        explanation = shap.KernelExplainer(model.predict_proba, background)(x_test)
        values = np.asarray(explanation.values)[0]
        base_values = np.asarray(explanation.base_values)[0]

        animal = str(meta.iloc[test_index]["animal_id"])
        true_label = str(y[test_index])
        pred_label = CLASS_ORDER[int(model.predict(x_test)[0])]
        true_class_index = CLASS_ORDER.index(true_label)
        final_probability = float(model.predict_proba(x_test)[0, true_class_index])
        additive_final = float(
            base_values[true_class_index] + values[:, true_class_index].sum()
        )
        if not np.isclose(final_probability, additive_final, atol=1e-8):
            raise RuntimeError(
                f"SHAP additivity failed for {animal}: "
                f"predict_proba={final_probability}, SHAP={additive_final}"
            )

        baseline_rows.append(
            {
                "animal_id": animal,
                "true_label": true_label,
                "pred_label": pred_label,
                "fold": fold,
                "base_value": float(base_values[true_class_index]),
                "final_value": final_probability,
            }
        )
        for feature_index, feature in enumerate(features):
            for class_index, class_name in enumerate(CLASS_ORDER):
                shap_rows.append(
                    {
                        "animal_id": animal,
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "feature": feature,
                        "class": class_name,
                        "shap_value": float(values[feature_index, class_index]),
                        "fold": fold,
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    shap_path = output_dir / "shap_values.csv"
    baseline_path = output_dir / "shap_base_values.csv"
    recomputed = pd.DataFrame(shap_rows)
    recomputed.to_csv(shap_path, index=False)
    pd.DataFrame(baseline_rows).to_csv(baseline_path, index=False)

    print(shap_path)
    print(baseline_path)


def compare_with_reference(output_dir):
    """Report whether each output file matches its copy in data/shap/."""
    print(f"\nComparison with {REFERENCE_DIR.relative_to(ROOT)}/:")
    for name in ("shap_values.csv", "shap_base_values.csv"):
        new, ref = output_dir / name, REFERENCE_DIR / name
        if filecmp.cmp(new, ref, shallow=False):
            result = "identical"
        else:
            a, b = pd.read_csv(new), pd.read_csv(ref)
            num = a.select_dtypes("number").columns
            same = (a.shape == b.shape and a.columns.equals(b.columns)
                    and a.drop(columns=num).equals(b.drop(columns=num))
                    and np.allclose(a[num], b[num], rtol=1e-9, atol=1e-12, equal_nan=True))
            result = "equal within rounding" if same else "DIFFERENT"
        print(f"  {name}: {result}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selected-features", default=str(DEFAULT_SELECTED),
                        help="selected_features CSV of the nested cross-validation")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help="folder for the two output CSVs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    compute_shap(Path(args.selected_features), output_dir)
    compare_with_reference(output_dir)


if __name__ == "__main__":
    main()
