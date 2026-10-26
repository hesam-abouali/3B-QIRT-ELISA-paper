#!/usr/bin/env python3
"""Per-rat class probabilities of the three best fed-state recipes, mRMR with LogReg_L2, SVM_linear or KNN
(predictive confidence, main-text Fig. 4).

For each outer fold of the nested cross-validation, the recipe's classifier is refitted on the eight
training rats with the features that the fold selected, read from the saved selected_features file, and
the class probabilities of the held-out rat are recorded. The classifiers and their settings are those of
nested_cv.py; probability estimates are switched on for SVM_linear (Platt scaling, random_state 0). The
predictions, correctness and numbers of features are checked against the saved per_fold_details file,
fold by fold; if any of them differs, the script stops without writing its output.

Input   data/features/fedstate_X.csv, fedstate_y.csv, fedstate_meta.csv
        data/classification/selected_features.csv, per_fold_details.csv
Output  confidence_by_rat.csv   one row per recipe and rat: prediction, selected features, class probabilities

The output is compared with data/classification/confidence_by_rat.csv.

    python src/predictive_confidence.py [--classification-dir DIR] [--output-dir DIR]
"""
from __future__ import annotations

import argparse
import filecmp
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import nested_cv as cv

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "features"
DEFAULT_CLASSIFICATION = ROOT / "data" / "classification"
DEFAULT_OUTPUT = ROOT / "outputs" / "predictive_confidence"
REFERENCE = ROOT / "data" / "classification" / "confidence_by_rat.csv"

CLASS_LABELS = ["healthy", "pre-diabetic", "diabetic"]
RECIPES = [("mrmr", "LogReg_L2"), ("mrmr", "SVM_linear"), ("mrmr", "KNN")]


def load_inputs() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    X = pd.read_csv(FEATURES / "fedstate_X.csv")
    y = pd.read_csv(FEATURES / "fedstate_y.csv")["rat_type"]
    meta = pd.read_csv(FEATURES / "fedstate_meta.csv")

    if "rat_type" in meta.columns and list(meta["rat_type"]) != list(y):
        raise ValueError("fedstate_meta and fedstate_y row order differ")

    zero_var = X.columns[X.var() == 0].tolist()
    if zero_var:
        X = X.drop(columns=zero_var)
    return X, y, meta


def load_per_fold_details(results_dir: Path) -> pd.DataFrame:
    return pd.read_csv(results_dir / "per_fold_details.csv")


def load_selected_feature_sets(results_dir: Path, selector: str, classifier: str) -> dict[int, list[str]]:
    df = pd.read_csv(results_dir / "selected_features.csv")
    df = df[(df["selector"] == selector) & (df["classifier"] == classifier)]
    if df.empty:
        raise ValueError(f"No selected features for {selector} + {classifier}")
    return {
        int(fold): fold_df["feature"].tolist()
        for fold, fold_df in df.sort_values("fold").groupby("fold", sort=True)
    }


def make_probability_classifier(classifier_name: str):
    """Instantiate a classifier of nested_cv.py, enabling probabilities for SVM."""
    params = dict(cv.Config.CLF_PARAMS[classifier_name])

    if classifier_name.startswith("LogReg"):
        return LogisticRegression(**params)
    if classifier_name == "RandomForest":
        return RandomForestClassifier(**params)
    if classifier_name == "DecisionTree":
        return DecisionTreeClassifier(**params)
    if classifier_name in ("SVM_linear", "SVM_rbf"):
        params["probability"] = True
        params.setdefault("random_state", cv.Config.RANDOM_STATE)
        return SVC(**params)
    if classifier_name == "LDA":
        return LinearDiscriminantAnalysis(**params)
    if classifier_name == "KNN":
        return KNeighborsClassifier(**params)
    if classifier_name == "NaiveBayes":
        return GaussianNB(**params)
    raise ValueError(f"Unsupported classifier: {classifier_name}")


def probability_vector(clf, X_te_k) -> tuple[np.ndarray, str]:
    raw = clf.predict_proba(X_te_k)[0]
    by_class = dict(zip(clf.classes_, raw))
    probs = np.array([by_class.get(lbl, 0.0) for lbl in CLASS_LABELS], dtype=float)
    prob_pred_label = CLASS_LABELS[int(np.argmax(probs))]
    return probs, prob_pred_label


def run_recipe(selector: str, classifier: str, results_dir: Path) -> pd.DataFrame:
    """Refit the final outer-fold models of one recipe and collect held-out probabilities."""
    X_df, y_ser, meta = load_inputs()
    feature_names = list(X_df.columns)
    feature_to_index = {feature: idx for idx, feature in enumerate(feature_names)}
    X_arr = X_df.to_numpy()
    y_arr = y_ser.to_numpy()
    rat_ids = meta["animal_id"].tolist()
    saved_features = load_selected_feature_sets(results_dir, selector, classifier)

    rows = []
    outer_cv = LeaveOneOut()
    for fold_idx, (tr_idx, te_idx) in enumerate(outer_cv.split(X_arr), start=1):
        X_tr, X_te = X_arr[tr_idx], X_arr[te_idx]
        y_tr, y_te = y_arr[tr_idx], y_arr[te_idx]
        selected_features = saved_features[fold_idx]
        selected_k = len(selected_features)
        missing = [feature for feature in selected_features if feature not in feature_to_index]
        if missing:
            raise ValueError(f"fold {fold_idx}: missing features {missing}")
        feat_idx = np.array([feature_to_index[feature] for feature in selected_features], dtype=int)

        X_tr_k = X_tr[:, feat_idx]
        X_te_k = X_te[:, feat_idx]
        if classifier in cv.SCALE_SENSITIVE:
            scaler = StandardScaler()
            X_tr_k = scaler.fit_transform(X_tr_k)
            X_te_k = scaler.transform(X_te_k)

        clf = make_probability_classifier(classifier)
        clf.fit(X_tr_k, y_tr)
        pred_label = clf.predict(X_te_k)[0]
        probs, prob_pred_label = probability_vector(clf, X_te_k)
        max_probability = float(probs.max())

        rows.append(
            {
                "selector": selector,
                "classifier": classifier,
                "fold": fold_idx,
                "rat_id": rat_ids[te_idx[0]],
                "true_label": y_te[0],
                "pred_label": pred_label,
                "prob_pred_label": prob_pred_label,
                "probability_prediction_matches_label": pred_label == prob_pred_label,
                "correct": pred_label == y_te[0],
                "selected_k": int(selected_k),
                "selected_features": "; ".join(selected_features),
                "p_healthy": float(probs[0]),
                "p_pre_diabetic": float(probs[1]),
                "p_diabetic": float(probs[2]),
                "max_probability": max_probability,
                "classification_uncertainty": 1.0 - max_probability,
            }
        )

    return pd.DataFrame(rows)


REQUIRED_CHECKS = ["folds_match_saved", "prediction_matches_saved", "correct_matches_saved",
                   "selected_k_matches_saved"]


def verify_against_saved_results(df: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    """Compare the refitted predictions, correctness and k values with the saved per-fold results.

    The checks in REQUIRED_CHECKS must all hold. prob_argmax_matches_pred_label is informational: for
    SVM_linear, the class with the highest Platt-scaled probability can differ from the prediction."""
    saved_all = load_per_fold_details(results_dir)
    checks = []
    for (selector, classifier), group in df.groupby(["selector", "classifier"]):
        saved = saved_all[(saved_all["selector"] == selector) & (saved_all["classifier"] == classifier)]
        merged = group.merge(
            saved[["fold", "rat_id", "pred_label", "correct", "selected_k"]],
            on=["fold", "rat_id"],
            how="left",
            suffixes=("_refitted", "_saved"),
            validate="one_to_one",
        )
        checks.append(
            {
                "selector": selector,
                "classifier": classifier,
                "n_folds": len(group),
                "folds_match_saved": bool(len(saved) == len(group) and merged["pred_label_saved"].notna().all()),
                "prediction_matches_saved": bool((merged["pred_label_refitted"] == merged["pred_label_saved"]).all()),
                "correct_matches_saved": bool((merged["correct_refitted"] == merged["correct_saved"]).all()),
                "selected_k_matches_saved": bool((merged["selected_k_refitted"] == merged["selected_k_saved"]).all()),
                "prob_argmax_matches_pred_label": bool(group["probability_prediction_matches_label"].all()),
            }
        )
    return pd.DataFrame(checks)


def compare_with_reference(out_path: Path) -> None:
    """Report whether the output matches data/classification/confidence_by_rat.csv."""
    if filecmp.cmp(out_path, REFERENCE, shallow=False):
        result = "identical"
    else:
        a, b = pd.read_csv(out_path), pd.read_csv(REFERENCE)
        num = a.select_dtypes("number").columns
        same = (a.shape == b.shape and a.columns.equals(b.columns)
                and a.drop(columns=num).equals(b.drop(columns=num))
                and np.allclose(a[num], b[num], rtol=1e-9, atol=1e-12, equal_nan=True))
        result = "equal within rounding" if same else "DIFFERENT"
    print(f"\nComparison with {REFERENCE.relative_to(ROOT)}: {result}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classification-dir", default=str(DEFAULT_CLASSIFICATION),
                        help="folder with the nested-CV results")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help="folder for confidence_by_rat.csv")
    args = parser.parse_args()
    results_dir = Path(args.classification_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for selector, classifier in RECIPES:
        print(f"Refitting {selector} + {classifier}")
        frames.append(run_recipe(selector, classifier, results_dir))
    df = pd.concat(frames, ignore_index=True)
    checks = verify_against_saved_results(df, results_dir)
    print(checks.to_string(index=False))
    if not checks[REQUIRED_CHECKS].all().all():
        sys.exit("The refitted results differ from per_fold_details.csv; confidence_by_rat.csv not written")

    out_path = output_dir / "confidence_by_rat.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
    compare_with_reference(out_path)


if __name__ == "__main__":
    main()
