#!/usr/bin/env python3
"""
Nested cross-validation of feature selectors and classifiers, GTT classification (LZ vs ZDF)
===========================================================================================

81 combinations: 9 selectors x 9 classifiers
Selectors (each ranks the features of a training set):
    ANOVA F-test, Kruskal-Wallis test, mutual information, random-forest importance, lasso (L1 logistic
    regression), recursive feature elimination (L2 logistic regression), ReliefF, mRMR (minimum
    redundancy, maximum relevance), elastic net (L1 + L2 logistic regression)
Classifiers:
    LogReg_L2, LogReg_L1, SVM_linear, SVM_rbf, DecisionTree, RandomForest, LDA, KNN, NaiveBayes

Rats     : the three LZ (Rat1-3) and three ZDF (Rat7-9) rats are scored; the three OZ rats (Rat4-6) are in
           every training set with the diabetic label and are never held out or used for validation.
Outer CV : leave one scored rat out (6 folds); the model is trained on the other 5 scored rats and the 3 OZ rats
Inner CV : leave one of the 5 scored training rats out; chooses the number of features k from {3, 5, 10} by the
           number of correctly classified validation rats (a tie goes to the smaller k)
Per fit  : missing values are replaced by the training-set median and every feature is standardized with the
           training-set mean and SD; the selector then ranks the features (exact ties in its scores go to the
           feature that comes first in the matrix, and features without variance go last)
Metric   : macro-F1 of the six held-out predictions

The label test uses all 20 ways of choosing 3 of the 6 scored rats as diabetic (assignment 19 = true labels).

run_classification.py runs the combinations with this module.
"""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, f_classif, mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SELECTORS = ["anova", "kruskal", "mutual_info", "rf_importance", "lasso",
             "rfe", "relieff", "mrmr", "elastic_net"]
CLASSIFIERS = ["LogReg_L2", "LogReg_L1", "SVM_linear", "SVM_rbf",
               "DecisionTree", "RandomForest", "LDA", "KNN", "NaiveBayes"]
K_GRID = (3, 5, 10)
RNG = 0
ZERO_VAR_TOL = 1e-12
SCORED = ("Rat1", "Rat2", "Rat3", "Rat7", "Rat8", "Rat9")    # LZ, ZDF
OZ = ("Rat4", "Rat5", "Rat6")                                 # training only, labelled diabetic
ASSIGNMENTS = list(combinations(range(6), 3))                 # positions in SCORED labelled diabetic
TRUE_ASSIGNMENT = ASSIGNMENTS.index((3, 4, 5))                # 19: Rat7-9 diabetic


# ---------------------------------------------------------------------------
# Data and folds
# ---------------------------------------------------------------------------
def load_matrix(x_csv, meta_csv):
    """Feature matrix (9 x n), feature names and the row positions of the scored and the OZ rats."""
    X = pd.read_csv(x_csv)
    ids = pd.read_csv(meta_csv)["animal_id"].tolist()
    return X.values.astype(float), list(X.columns), [ids.index(a) for a in SCORED], [ids.index(a) for a in OZ]


def assignment_labels(index):
    """Labels of the six scored rats (0 = LZ, 1 = ZDF) under one of the 20 assignments."""
    y = np.zeros(6, dtype=int)
    y[list(ASSIGNMENTS[index])] = 1
    return y


def impute_scale(Xtr, Xte):
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)      # column all-NaN in this fold -> 0
    Xtr = np.where(np.isnan(Xtr), med, Xtr)
    Xte = np.where(np.isnan(Xte), med, Xte)
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1.0
    return (Xtr - mu) / sd, (Xte - mu) / sd


def build_folds(X9, scored_pos, oz_pos, y_scored):
    """Outer and inner splits, each imputed and standardized on its own training set.

    folds[t] = (t, Xtr, y_tr, Xte, inner), inner = [(Xi, y_i, Xj, y_j), ...] for the 5 inner splits.
    """
    X9 = np.asarray(X9, dtype=float)
    y_by_row = {row: int(y_scored[m]) for m, row in enumerate(scored_pos)}
    for row in oz_pos:
        y_by_row[row] = 1
    folds = []
    for t in range(6):
        te = scored_pos[t]
        tr_scored = [scored_pos[m] for m in range(6) if m != t]
        tr = tr_scored + list(oz_pos)
        Xtr, Xte = impute_scale(X9[tr], X9[[te]])
        y_tr = np.array([y_by_row[r] for r in tr], dtype=int)
        inner = []
        for j in range(5):
            itr = [tr_scored[m] for m in range(5) if m != j] + list(oz_pos)
            ite = tr_scored[j]
            Xi, Xj = impute_scale(X9[itr], X9[[ite]])
            inner.append((Xi, np.array([y_by_row[r] for r in itr], dtype=int), Xj, int(y_by_row[ite])))
        folds.append((t, Xtr, y_tr, Xte, inner))
    return folds


# ---------------------------------------------------------------------------
# Selectors: feature ranking of a training set
# ---------------------------------------------------------------------------
def deterministic_order(scores):
    """Descending score; exact ties go to the lowest feature index."""
    scores = np.asarray(scores, dtype=float)
    index = np.arange(scores.size)
    return np.lexsort((index, -scores))


def apply_zero_variance_demotion(order, X):
    """Move features without variance in the training set to the end, keeping the order of the others."""
    var = np.asarray(X, dtype=float).var(axis=0)
    bad = var[order] <= ZERO_VAR_TOL
    if not bad.any():
        return order
    return np.concatenate([order[~bad], order[bad]])


def relieff_weights(X, y, n_neighbors=1):
    """ReliefF weights; equidistant neighbours are resolved toward the lowest training-row index."""
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    fmin = X.min(0)
    frng = X.max(0) - fmin
    frng[frng == 0] = 1.0
    Xn = (X - fmin) / frng
    w = np.zeros(p)
    classes, counts = np.unique(y, return_counts=True)
    prior = dict(zip(classes, counts / n))
    rows = np.arange(n)
    for i in range(n):
        d = np.sqrt(((Xn - Xn[i]) ** 2).sum(1))
        d[i] = np.inf
        same = (y == y[i]).copy()
        same[i] = False
        if same.any():
            k = min(n_neighbors, int(same.sum()))
            cand = np.where(same, d, np.inf)
            pick = np.lexsort((rows, cand))[:k]          # distance, then lowest row index
            for h in pick:
                w -= np.abs(Xn[i] - Xn[h]) / (n * k)
        for c in classes:
            if c == y[i]:
                continue
            cm = (y == c)
            if not cm.any():
                continue
            k = min(n_neighbors, int(cm.sum()))
            wt = prior[c] / (1 - prior[y[i]])
            cand = np.where(cm, d, np.inf)
            pick = np.lexsort((rows, cand))[:k]
            for m in pick:
                w += wt * np.abs(Xn[i] - Xn[m]) / (n * k)
    return w


def mrmr_order(X, y, max_features=6):
    """mRMR: greedy selection of max_features features (relevance minus mean redundancy, both by mutual
    information; the lowest index wins an exact tie), then the remaining features by relevance."""
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    nn = min(3, n - 1)
    rel = np.nan_to_num(mutual_info_classif(X, y, n_neighbors=nn, random_state=RNG))
    sel = [int(np.argmax(rel))]                       # argmax already returns the lowest index
    rem = set(range(p)) - set(sel)
    cache = {}
    for _ in range(1, min(max_features, p)):
        best, best_s = None, -np.inf
        for f in sorted(rem):                          # deterministic traversal
            red = 0.0
            for s in sel:
                key = (min(f, s), max(f, s))
                if key not in cache:
                    cache[key] = max(mutual_info_regression(
                        X[:, f].reshape(-1, 1), X[:, s],
                        n_neighbors=min(nn, n - 1), random_state=RNG)[0], 0.0)
                red += cache[key]
            s_ = rel[f] - red / len(sel)
            if s_ > best_s:                            # strict >: first (lowest index) wins ties
                best_s, best = s_, f
        sel.append(best)
        rem.discard(best)
    tail = sorted(rem, key=lambda f: (-rel[f], f))     # relevance desc, then lowest index
    return np.array(sel + tail, dtype=int)


def selector_scores(selector, X, y):
    """Per-feature scores; kind is 'descending' (higher = better), 'rfe_rank' (lower = better) or 'greedy'."""
    X = np.asarray(X, dtype=float)
    if selector == "anova":
        return np.nan_to_num(f_classif(X, y)[0]), "descending"
    if selector == "kruskal":
        return np.array([
            -np.log10(stats.kruskal(*[X[y == c, j] for c in np.unique(y)])[1] + 1e-300)
            if np.ptp(X[:, j]) > 0 else 0.0 for j in range(X.shape[1])]), "descending"
    if selector == "mutual_info":
        return mutual_info_classif(X, y, n_neighbors=min(3, len(y) - 1), random_state=RNG), "descending"
    if selector == "rf_importance":
        return RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                      random_state=RNG).fit(X, y).feature_importances_, "descending"
    if selector == "lasso":
        return np.abs(LogisticRegression(penalty="l1", C=1.0, solver="saga", max_iter=10000,
                                         class_weight="balanced", random_state=RNG)
                      .fit(X, y).coef_).mean(0), "descending"
    if selector == "elastic_net":
        return np.abs(LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=1.0, solver="saga",
                                         max_iter=10000, class_weight="balanced", random_state=RNG)
                      .fit(X, y).coef_).mean(0), "descending"
    if selector == "relieff":
        return relieff_weights(X, y), "descending"
    if selector == "rfe":
        est = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=5000,
                                 class_weight="balanced", random_state=RNG)
        return RFE(est, n_features_to_select=1).fit(X, y).ranking_.astype(float), "rfe_rank"
    if selector == "mrmr":
        return None, "greedy"
    raise ValueError(f"unknown selector {selector!r}")


def rank_features(selector, X, y):
    """Feature ranking (indices, best first) of one selector on one training set."""
    X = np.asarray(X, dtype=float)
    scores, kind = selector_scores(selector, X, y)
    if kind == "greedy":
        order = mrmr_order(X, y)
    elif kind == "rfe_rank":
        if len(np.unique(scores)) != scores.size:
            order = np.lexsort((np.arange(scores.size), scores))
        else:
            order = np.argsort(scores, kind="stable")
    else:
        order = deterministic_order(scores)
    return apply_zero_variance_demotion(np.asarray(order, dtype=int), X)


def selector_rankings(selector, folds):
    """Rankings of every outer and inner training set: [(outer_order, [inner_order, ...]), ...]."""
    return [(rank_features(selector, Xtr, ytr), [rank_features(selector, Xi, yi) for (Xi, yi, _, _) in inner])
            for (_, Xtr, ytr, _, inner) in folds]


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------
def build_clf(name, proba=False):
    """Classifier with balanced class weights; LDA and naive Bayes use equal class priors."""
    return {
        "LogReg_L2":    LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=5000,
                                           class_weight="balanced", random_state=RNG),
        "LogReg_L1":    LogisticRegression(penalty="l1", C=1.0, solver="saga", max_iter=10000,
                                           class_weight="balanced", random_state=RNG),
        # probability=True triggers an internal cross-validated calibration at fit time,
        # so it is enabled only for the outer fit
        "SVM_linear":   SVC(kernel="linear", C=1.0, class_weight="balanced",
                            probability=proba, random_state=RNG),
        "SVM_rbf":      SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced",
                            probability=proba, random_state=RNG),
        "DecisionTree": DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=RNG),
        "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                               random_state=RNG),
        "LDA":          LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[.5, .5]),
        "KNN":          KNeighborsClassifier(n_neighbors=3, weights="distance"),
        "NaiveBayes":   GaussianNB(priors=[.5, .5]),
    }[name]


def fit_predict(clf_name, Xtr, ytr, Xte, need_proba=False):
    clf = build_clf(clf_name, proba=need_proba)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    proba = None
    if need_proba and hasattr(clf, "predict_proba"):
        try:
            proba = clf.predict_proba(Xte)[:, 1]
        except Exception:
            proba = None
    return pred, proba


# ---------------------------------------------------------------------------
# Nested cross-validation of one combination
# ---------------------------------------------------------------------------
def run_combination(classifier, folds, rankings, feature_names):
    """Held-out prediction of each scored rat, with the number of features chosen in the inner loop."""
    predictions, probabilities, fold_rows = [], [], []
    for fold_index, ((_, Xtr, ytr, Xte, inner), (outer_rank, inner_rank)) in enumerate(zip(folds, rankings), 1):
        inner_correct = {k: 0 for k in K_GRID}
        for (Xi, yi, Xj, yj), inner_order in zip(inner, inner_rank):
            for k in K_GRID:
                selected = inner_order[:k]
                prediction, _ = fit_predict(classifier, Xi[:, selected], yi, Xj[:, selected])
                inner_correct[k] += int(prediction[0] == yj)
        best_k = max(K_GRID, key=lambda k: inner_correct[k])   # first (smallest) k on a tie
        selected_outer = outer_rank[:best_k]
        prediction, probability = fit_predict(classifier, Xtr[:, selected_outer], ytr, Xte[:, selected_outer],
                                              need_proba=True)
        if probability is None or not np.isfinite(probability[0]):
            raise RuntimeError(f"{classifier} fold {fold_index}: missing probability")
        predictions.append(int(prediction[0]))
        probabilities.append(float(probability[0]))
        fold_rows.append({
            "fold": fold_index,
            "animal_id": SCORED[fold_index - 1],
            "selected_k": int(best_k),
            "inner_correct_by_k": {str(k): int(inner_correct[k]) for k in K_GRID},
            "selected_features": [feature_names[int(i)] for i in selected_outer],
        })
    return predictions, probabilities, fold_rows


def run_selector(X9, feature_names, scored_pos, oz_pos, selector, y_scored):
    """All nine classifiers with one selector, for one labelling of the scored rats."""
    folds = build_folds(X9, scored_pos, oz_pos, y_scored)
    rankings = selector_rankings(selector, folds)
    pipelines = []
    for classifier in CLASSIFIERS:
        predictions, probabilities, fold_rows = run_combination(classifier, folds, rankings, feature_names)
        pipelines.append({
            "selector": selector,
            "classifier": classifier,
            **metrics(y_scored, predictions, probabilities, fold_rows),
            "predictions": predictions,
            "probabilities_diabetic": probabilities,
            "folds": fold_rows,
        })
    return pipelines


def metrics(truth, predictions, probabilities, fold_rows):
    truth, predictions = np.asarray(truth, dtype=int), np.asarray(predictions, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "macro_f1": float(f1_score(truth, predictions, average="macro")),
        "accuracy": float(np.mean(truth == predictions)),
        "brier_score": float(np.mean((p - truth) ** 2)),
        "log_loss": float(-np.mean(truth * np.log(clipped) + (1 - truth) * np.log(1 - clipped))),
        "mean_predictive_confidence": float(np.mean(np.maximum(p, 1.0 - p))),
        "mean_selected_k": float(np.mean([f["selected_k"] for f in fold_rows])),
    }
