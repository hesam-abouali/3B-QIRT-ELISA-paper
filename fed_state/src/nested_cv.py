#!/usr/bin/env python3
"""
Nested cross-validation of feature selectors and classifiers, fed-state classification (n = 9)
=============================================================================================

81 combinations: 9 selectors x 9 classifiers
Selectors:
  Univariate (inner LOO):
    - ANOVA F-test
    - Kruskal-Wallis H-test
    - Mutual information
  Model-based (inner LOO):
    - RFE (recursive feature elimination, L2 logistic regression base)
    - RF importance (random forest mean decrease in impurity)
    - Elastic net (mean absolute coefficient of an L1 + L2 logistic regression)
  Multivariate (inner LOO):
    - ReliefF (instance-based feature weighting)
    - mRMR (minimum redundancy, maximum relevance)
  Permutation (inner stratified 3-fold, requires held-out samples):
    - Permutation importance (shrinkage LDA base estimator)

Classifiers (all combinations):
    - LogReg_L2
    - LogReg_L1
    - LDA
    - SVM_linear
    - SVM_rbf
    - DecisionTree
    - RandomForest
    - KNN
    - NaiveBayes

Outer CV : leave-one-out (9 folds)
Inner CV : leave-one-out, or stratified 3-fold for permutation importance; chooses k from the k-grid
k-grid   : [3, 5, 10, 20, 30], restricted to k smaller than the number of training rats
Metric   : macro-F1 (primary)

run_classification.py runs all 81 combinations with this module.
"""
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import kruskal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, f_classif, mutual_info_classif, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Classifiers that are sensitive to feature magnitudes are fitted on standardized features
# (NaiveBayes is not: it estimates per-feature Gaussians independently)
SCALE_SENSITIVE = {"LogReg_L2", "LogReg_L1", "SVM_linear", "SVM_rbf", "LDA", "KNN"}


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    # ---- Paths ----
    _ROOT = Path(__file__).resolve().parents[1]
    X_PATH    = str(_ROOT / "data" / "features" / "fedstate_X.csv")
    Y_PATH    = str(_ROOT / "data" / "features" / "fedstate_y.csv")
    META_PATH = str(_ROOT / "data" / "features" / "fedstate_meta.csv")
    OUTPUT_DIR = str(_ROOT / "outputs" / "classification")

    # ---- Dataset ----
    N_SAMPLES    = 9
    CLASS_LABELS = ["healthy", "pre-diabetic", "diabetic"]

    # ---- Feature selection ----
    K_GRID = [3, 5, 10, 20, 30]

    # ---- Selector groups ----
    # inner_cv_mode determines which inner CV structure to use
    SELECTOR_INNER_CV = {
        "anova":           "loo",
        "kruskal":         "loo",
        "mutual_info":     "loo",
        "rfe":             "loo",
        "rf_importance":   "loo",
        "perm_importance": "stratified_3fold",
        "relieff":         "loo",
        "mrmr":            "loo",
        "elastic_net":     "loo",
    }

    ALL_SELECTORS    = list(SELECTOR_INNER_CV.keys())
    ALL_CLASSIFIERS  = ["LogReg_L2", "LogReg_L1", "SVM_linear",
                        "SVM_rbf", "DecisionTree", "RandomForest",
                        "LDA", "KNN", "NaiveBayes"]

    # ---- Reproducibility ----
    RANDOM_STATE = 0

    # ---- Classifier hyperparameters ----
    CLF_PARAMS = {
        "LogReg_L2": dict(C=1.0, max_iter=5000,  penalty="l2",
                          solver="lbfgs", class_weight="balanced"),
        "LogReg_L1": dict(C=1.0, max_iter=10000, penalty="l1",
                          solver="saga",  class_weight="balanced"),
        "SVM_linear": dict(C=1.0,  kernel="linear", class_weight="balanced"),
        "SVM_rbf":    dict(C=10.0, kernel="rbf", gamma="scale",
                           class_weight="balanced"),
        "DecisionTree": dict(max_depth=3, class_weight="balanced",
                             random_state=RANDOM_STATE),
        "RandomForest": dict(n_estimators=300, class_weight="balanced_subsample",
                             random_state=RANDOM_STATE),
        "LDA":        dict(solver="lsqr", shrinkage="auto"),
        "KNN":        dict(n_neighbors=3, weights="distance"),
        "NaiveBayes": dict(),
    }


# =============================================================================
# SELECTOR BASE CLASS
# =============================================================================

class BaseSelector:
    """
    Common interface for all selectors.

    Subclasses must implement fit() and set self.ranking_ (array of feature
    indices ordered best → worst).

    Attributes
    ----------
    name : str
    inner_cv_mode : str  — 'loo' or 'stratified_3fold'
    scores_  : np.ndarray or None
    ranking_ : np.ndarray or None
    """

    inner_cv_mode: str = "loo"

    def __init__(self, name: str):
        self.name     = name
        self.scores_  = None
        self.ranking_ = None

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None) -> "BaseSelector":
        raise NotImplementedError

    def get_top_k_features(self, k: int) -> np.ndarray:
        if self.ranking_ is None:
            raise ValueError("Selector not fitted.")
        return self.ranking_[:k]


# =============================================================================
# UNIVARIATE SELECTORS  (inner_cv_mode = 'loo')
# =============================================================================

class ANOVASelector(BaseSelector):
    inner_cv_mode = "loo"

    def __init__(self):
        super().__init__("anova")

    def fit(self, X, y, X_val=None, y_val=None):
        scores, _ = f_classif(X, y)
        self.scores_  = np.nan_to_num(scores, nan=0.0)
        self.ranking_ = np.argsort(self.scores_)[::-1]
        return self


class KruskalWallisSelector(BaseSelector):
    inner_cv_mode = "loo"

    def __init__(self):
        super().__init__("kruskal")

    def fit(self, X, y, X_val=None, y_val=None):
        n_features = X.shape[1]
        scores = np.zeros(n_features)
        classes = np.unique(y)
        for i in range(n_features):
            groups = [X[y == c, i] for c in classes]
            try:
                h, _ = kruskal(*groups)
                scores[i] = h
            except Exception:
                scores[i] = 0.0
        self.scores_  = scores
        self.ranking_ = np.argsort(scores)[::-1]
        return self


class MutualInfoSelector(BaseSelector):
    inner_cv_mode = "loo"

    def __init__(self, n_neighbors: int = 3, random_state: int = 0):
        super().__init__("mutual_info")
        self.n_neighbors  = n_neighbors
        self.random_state = random_state

    def fit(self, X, y, X_val=None, y_val=None):
        nn = min(self.n_neighbors, X.shape[0] - 1)
        scores = mutual_info_classif(X, y, n_neighbors=nn,
                                     random_state=self.random_state)
        self.scores_  = np.nan_to_num(scores, nan=0.0)
        self.ranking_ = np.argsort(self.scores_)[::-1]
        return self


# =============================================================================
# MODEL-BASED SELECTORS  (inner_cv_mode = 'loo')
# =============================================================================

class RFESelector(BaseSelector):
    """
    Recursive Feature Elimination using LogReg L2 as the base estimator.
    sklearn RFE.ranking_ : 1 = best (last to be eliminated).
    """
    inner_cv_mode = "loo"

    def __init__(self, random_state: int = 0):
        super().__init__("rfe")
        self.random_state = random_state

    def fit(self, X, y, X_val=None, y_val=None):
        base = LogisticRegression(C=1.0, max_iter=5000, penalty="l2",
                                  solver="lbfgs", class_weight="balanced",
                                  random_state=self.random_state)
        # Scale before RFE (LogReg is scale-sensitive)
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)

        # step=10: ten features are eliminated per round. Features eliminated in the same round share a
        # rank, so the top feature is followed by a group of tied features (the last round removes the
        # remaining nine at once), and which of them enter the top 3 or top 5 depends on how argsort
        # breaks the ties.
        rfe = RFE(estimator=base, n_features_to_select=1, step=10)
        rfe.fit(X_sc, y)

        # ranking_: 1 = selected last (best). Convert to descending order.
        # Lower ranking_ value → kept longer → more important.
        self.scores_  = (rfe.ranking_.max() + 1 - rfe.ranking_).astype(float)
        self.ranking_ = np.argsort(self.scores_)[::-1]
        return self


class RFImportanceSelector(BaseSelector):
    """
    Random Forest mean decrease in impurity importance.
    """
    inner_cv_mode = "loo"

    def __init__(self, n_estimators: int = 200, max_depth: int = 4,
                 min_samples_leaf: int = 2, random_state: int = 0):
        super().__init__("rf_importance")
        self.n_estimators    = n_estimators
        self.max_depth       = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state    = random_state

    def fit(self, X, y, X_val=None, y_val=None):
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight="balanced_subsample",
            random_state=self.random_state,
        )
        rf.fit(X, y)
        self.scores_  = rf.feature_importances_
        self.ranking_ = np.argsort(self.scores_)[::-1]
        return self


# =============================================================================
# PERMUTATION IMPORTANCE SELECTOR  (inner_cv_mode = 'stratified_3fold')
# =============================================================================

class PermImportanceSelector(BaseSelector):
    """
    Permutation importance using Shrinkage LDA as base estimator.

    IMPORTANT: requires X_val / y_val (held-out samples) to be passed to
    fit().  The pipeline uses StratifiedKFold(3) as the inner CV so that
    each inner validation set has 2-3 samples — sufficient for meaningful
    permutation scores.

    In the inner CV the selector is refitted in each fold. For the final model of an outer fold it is
    fitted on the first stratified 3-fold split of the outer training set (NestedCVPipeline._train_predict).
    """
    inner_cv_mode = "stratified_3fold"

    def __init__(self, n_repeats: int = 10, random_state: int = 0):
        super().__init__("perm_importance")
        self.n_repeats    = n_repeats
        self.random_state = random_state

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X, y       : inner training data  (used to train the base estimator)
        X_val, y_val : inner validation data (used to compute permutation scores)
        """
        if X_val is None or y_val is None:
            raise ValueError("PermImportanceSelector requires X_val and y_val.")

        base = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        scaler = StandardScaler()
        X_sc     = scaler.fit_transform(X)
        X_val_sc = scaler.transform(X_val)

        base.fit(X_sc, y)

        result = permutation_importance(
            base, X_val_sc, y_val,
            scoring="f1_macro",
            n_repeats=self.n_repeats,
            random_state=self.random_state,
            n_jobs=1,
        )
        self.scores_  = np.maximum(result.importances_mean, 0.0)
        self.ranking_ = np.argsort(self.scores_)[::-1]
        return self


# =============================================================================
# MULTIVARIATE AND REGULARIZED SELECTORS  (inner_cv_mode = 'loo')
# =============================================================================

class ReliefFSelector(BaseSelector):
    """
    ReliefF: instance-based feature weighting (multiclass extension).

    For each sample, finds nearest same-class neighbor and nearest
    different-class neighbors (one miss per other class). Features that
    separate classes (large miss diff, small hit diff) get higher weights.

    Features are normalized to [0, 1] internally so distance computation
    is scale-invariant.
    """
    inner_cv_mode = "loo"

    def __init__(self, n_neighbors=1):
        super().__init__("relieff")
        self.n_neighbors = n_neighbors

    def fit(self, X, y, X_val=None, y_val=None):
        n_samples, n_features = X.shape

        # Normalize to [0, 1] for fair distance computation
        f_min = X.min(axis=0)
        f_range = X.max(axis=0) - f_min
        f_range[f_range == 0] = 1.0
        X_norm = (X - f_min) / f_range

        weights = np.zeros(n_features)
        classes, counts = np.unique(y, return_counts=True)
        prior = dict(zip(classes, counts / n_samples))

        for i in range(n_samples):
            xi = X_norm[i]
            yi = y[i]

            # Euclidean distances to all other samples
            dists = np.sqrt(np.sum((X_norm - xi) ** 2, axis=1))
            dists[i] = np.inf

            # Nearest hit(s) — same class
            same_mask = (y == yi)
            same_mask[i] = False
            if same_mask.any():
                hit_dists = np.where(same_mask, dists, np.inf)
                k_hits = min(self.n_neighbors, same_mask.sum())
                hit_idx = np.argpartition(hit_dists, k_hits)[:k_hits]
                for h in hit_idx:
                    weights -= np.abs(xi - X_norm[h]) / (n_samples * k_hits)

            # Nearest miss(es) — one per other class, weighted by prior
            for c in classes:
                if c == yi:
                    continue
                c_mask = (y == c)
                if not c_mask.any():
                    continue
                miss_dists = np.where(c_mask, dists, np.inf)
                k_misses = min(self.n_neighbors, c_mask.sum())
                miss_idx = np.argpartition(miss_dists, k_misses)[:k_misses]
                w = prior[c] / (1 - prior[yi])
                for m in miss_idx:
                    weights += w * np.abs(xi - X_norm[m]) / (n_samples * k_misses)

        self.scores_ = weights
        self.ranking_ = np.argsort(weights)[::-1]
        return self


class MRMRSelector(BaseSelector):
    """
    Minimum Redundancy Maximum Relevance (mRMR) — greedy forward selection.

    At each step selects the feature maximizing:
        score(f) = MI(f, target) - (1/|S|) * Σ MI(f, s)   for s ∈ S

    where MI is mutual information.

    Produces less redundant feature subsets than pure MI ranking, which is
    important when QIRT-ELISA features are highly correlated (e.g., all
    insulin AUC variants).

    max_features controls how many features receive the mRMR-aware ranking;
    remaining features are appended sorted by raw relevance. Since effective
    k is only 3–5 at n=9, max_features=10 provides sufficient buffer.
    """
    inner_cv_mode = "loo"

    def __init__(self, max_features=10, random_state=0):
        super().__init__("mrmr")
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y, X_val=None, y_val=None):
        n_samples, n_features = X.shape
        nn = min(3, n_samples - 1)

        # Relevance: MI(feature, target)
        relevance = mutual_info_classif(X, y, n_neighbors=nn,
                                        random_state=self.random_state)
        relevance = np.nan_to_num(relevance, nan=0.0)

        max_k = min(self.max_features, n_features)
        selected = []
        remaining = set(range(n_features))
        mi_cache = {}

        # First feature: highest relevance
        first = int(np.argmax(relevance))
        selected.append(first)
        remaining.discard(first)

        for _ in range(1, max_k):
            best_score = -np.inf
            best_feat = None

            for f in remaining:
                rel = relevance[f]
                # Redundancy: average MI with already-selected features
                red = 0.0
                for s in selected:
                    key = (min(f, s), max(f, s))
                    if key not in mi_cache:
                        mi_val = mutual_info_regression(
                            X[:, f].reshape(-1, 1), X[:, s],
                            n_neighbors=min(nn, n_samples - 1),
                            random_state=self.random_state,
                        )[0]
                        mi_cache[key] = max(mi_val, 0.0)
                    red += mi_cache[key]
                red /= len(selected)
                score = rel - red

                if score > best_score:
                    best_score = score
                    best_feat = f

            if best_feat is not None:
                selected.append(best_feat)
                remaining.discard(best_feat)

        # Full ranking: mRMR-selected first, then unselected by raw relevance
        unselected = sorted(remaining, key=lambda f: relevance[f], reverse=True)
        full_ranking = selected + unselected

        self.scores_ = np.zeros(n_features)
        for rank, idx in enumerate(full_ranking):
            self.scores_[idx] = n_features - rank
        self.ranking_ = np.array(full_ranking)
        return self


class ElasticNetSelector(BaseSelector):
    """
    Elastic Net feature ranking via multinomial LogisticRegression
    with elasticnet penalty (L1 + L2).

    Ranks features by mean absolute coefficient across classes.
    Bridges sparsity (L1) and stability (L2).
    """
    inner_cv_mode = "loo"

    def __init__(self, l1_ratio=0.5, C=1.0, random_state=0):
        super().__init__("elastic_net")
        self.l1_ratio = l1_ratio
        self.C = C
        self.random_state = random_state

    def fit(self, X, y, X_val=None, y_val=None):
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)

        model = LogisticRegression(
            penalty="elasticnet",
            l1_ratio=self.l1_ratio,
            C=self.C,
            solver="saga",
            max_iter=10000,
            class_weight="balanced",
            random_state=self.random_state,
        )
        model.fit(X_sc, y)

        self.scores_ = np.mean(np.abs(model.coef_), axis=0)
        self.ranking_ = np.argsort(self.scores_)[::-1]
        return self



# =============================================================================
# SELECTOR / CLASSIFIER FACTORIES
# =============================================================================

def get_selector(name: str) -> BaseSelector:
    registry = {
        "anova":           ANOVASelector(),
        "kruskal":         KruskalWallisSelector(),
        "mutual_info":     MutualInfoSelector(n_neighbors=3,
                                              random_state=Config.RANDOM_STATE),
        "rfe":             RFESelector(random_state=Config.RANDOM_STATE),
        "rf_importance":   RFImportanceSelector(random_state=Config.RANDOM_STATE),
        "perm_importance": PermImportanceSelector(n_repeats=10,
                                                  random_state=Config.RANDOM_STATE),
        "relieff":         ReliefFSelector(n_neighbors=1),
        "mrmr":            MRMRSelector(max_features=10,
                                        random_state=Config.RANDOM_STATE),
        "elastic_net":     ElasticNetSelector(l1_ratio=0.5, C=1.0,
                                              random_state=Config.RANDOM_STATE),
    }
    if name not in registry:
        raise ValueError(f"Unknown selector '{name}'. Choose from {list(registry)}")
    return registry[name]


def get_classifier(name: str):
    params = Config.CLF_PARAMS[name]
    if name.startswith("LogReg"):
        clf = LogisticRegression(**params)
    elif name.startswith("SVM"):
        clf = SVC(**params)
    elif name == "DecisionTree":
        clf = DecisionTreeClassifier(**params)
    elif name == "RandomForest":
        clf = RandomForestClassifier(**params)
    elif name == "LDA":
        clf = LinearDiscriminantAnalysis(**params)
    elif name == "KNN":
        clf = KNeighborsClassifier(**params)
    elif name == "NaiveBayes":
        clf = GaussianNB(**params)
    else:
        raise ValueError(f"Unknown classifier '{name}'")
    return clf


# =============================================================================
# INNER CV HELPERS
# =============================================================================

def _inner_loo(X_train, y_train, selector_name, classifier_name, k_grid):
    """
    Inner LOO: select best k using macro-F1 on (n-1) training samples.
    Returns best_k, k_metrics dict.
    """
    inner_cv = LeaveOneOut()
    k_preds  = {k: [] for k in k_grid}
    k_trues  = {k: [] for k in k_grid}

    for itr_idx, ival_idx in inner_cv.split(X_train):
        Xi_tr, Xi_val = X_train[itr_idx], X_train[ival_idx]
        yi_tr, yi_val = y_train[itr_idx], y_train[ival_idx]

        sel = get_selector(selector_name)
        sel.fit(Xi_tr, yi_tr)          # no val data needed for LOO selectors

        for k in k_grid:
            feat_idx   = sel.get_top_k_features(k)
            Xi_tr_k    = Xi_tr[:, feat_idx]
            Xi_val_k   = Xi_val[:, feat_idx]

            # Scale only for scale-sensitive classifiers
            if classifier_name in SCALE_SENSITIVE:
                scaler     = StandardScaler()
                Xi_tr_k    = scaler.fit_transform(Xi_tr_k)
                Xi_val_k   = scaler.transform(Xi_val_k)

            clf = get_classifier(classifier_name)
            clf.fit(Xi_tr_k, yi_tr)
            k_preds[k].append(clf.predict(Xi_val_k)[0])
            k_trues[k].append(yi_val[0])

    return _pick_best_k(k_preds, k_trues, k_grid)


def _inner_stratified3fold(X_train, y_train, selector_name, classifier_name, k_grid):
    """
    Inner Stratified 3-Fold: used for PermImportanceSelector so each
    inner validation set has 2-3 samples for meaningful permutation scores.
    The selector ranks the features on the inner validation samples, and k is
    then scored on the same samples, so the inner score that chooses k is
    optimistic for this selector. The held-out rat of the outer fold is not
    used.
    Returns best_k, k_metrics dict.
    """
    skf      = StratifiedKFold(n_splits=3, shuffle=True,
                               random_state=Config.RANDOM_STATE)
    k_preds  = {k: [] for k in k_grid}
    k_trues  = {k: [] for k in k_grid}

    for itr_idx, ival_idx in skf.split(X_train, y_train):
        Xi_tr, Xi_val = X_train[itr_idx], X_train[ival_idx]
        yi_tr, yi_val = y_train[itr_idx], y_train[ival_idx]

        # Fit selector with validation data (required for perm_importance)
        sel = get_selector(selector_name)
        sel.fit(Xi_tr, yi_tr, X_val=Xi_val, y_val=yi_val)

        for k in k_grid:
            feat_idx = sel.get_top_k_features(k)
            Xi_tr_k  = Xi_tr[:, feat_idx]
            Xi_val_k = Xi_val[:, feat_idx]

            # Scale only for scale-sensitive classifiers
            if classifier_name in SCALE_SENSITIVE:
                scaler   = StandardScaler()
                Xi_tr_k  = scaler.fit_transform(Xi_tr_k)
                Xi_val_k = scaler.transform(Xi_val_k)

            clf = get_classifier(classifier_name)
            clf.fit(Xi_tr_k, yi_tr)
            for pred, true in zip(clf.predict(Xi_val_k), yi_val):
                k_preds[k].append(pred)
                k_trues[k].append(true)

    return _pick_best_k(k_preds, k_trues, k_grid)


def _pick_best_k(k_preds, k_trues, k_grid):
    """
    Choose k with highest inner macro-F1; tie-break by smallest k.
    Returns (best_k, k_metrics).
    """
    k_metrics = {}
    for k in k_grid:
        mf1 = f1_score(k_trues[k], k_preds[k], average="macro",
                       labels=Config.CLASS_LABELS, zero_division=0)
        k_metrics[k] = {"macro_f1": mf1}

    best_f1 = max(v["macro_f1"] for v in k_metrics.values())
    best_k  = min(k for k, v in k_metrics.items() if v["macro_f1"] == best_f1)
    return best_k, k_metrics


# =============================================================================
# NESTED CV PIPELINE (OUTER LOO)
# =============================================================================

class NestedCVPipeline:
    """
    Outer LOO + inner CV (mode determined by selector).
    Supports all nine selectors uniformly.
    """

    def __init__(self, selector_name: str, classifier_name: str,
                 k_grid: List[int]):
        self.selector_name   = selector_name
        self.classifier_name = classifier_name
        self.k_grid          = k_grid
        self.inner_cv_mode   = Config.SELECTOR_INNER_CV[selector_name]

        # Results
        self.outer_predictions_       = []
        self.outer_true_labels_       = []
        self.outer_selected_features_ = []
        self.outer_selected_k_        = []
        self.outer_rat_ids_           = []  # Track which rat was tested

    def run(self, X: pd.DataFrame, y: pd.Series,
            feature_names: List[str], rat_ids: List[str] = None) -> Dict:

        print(f"\n  Outer LOO | inner={self.inner_cv_mode} | "
              f"{self.selector_name} + {self.classifier_name}")

        # Start from empty results, so that a second run() does not add to the first
        self.outer_predictions_       = []
        self.outer_true_labels_       = []
        self.outer_selected_features_ = []
        self.outer_selected_k_        = []
        self.outer_rat_ids_           = []

        X_arr = X.values
        y_arr = y.values
        outer_cv = LeaveOneOut()

        # Use provided rat_ids or generate default identifiers
        if rat_ids is None:
            rat_ids = [f"Sample_{i}" for i in range(len(y_arr))]

        for fold_idx, (tr_idx, te_idx) in enumerate(outer_cv.split(X_arr)):
            X_tr, X_te = X_arr[tr_idx], X_arr[te_idx]
            y_tr, y_te = y_arr[tr_idx], y_arr[te_idx]
            rat_id_te = rat_ids[te_idx[0]]  # Track which rat is being tested

            # Cap the k-grid at the number of outer-training samples: k must be
            # smaller than it. With n=9 outer LOO there are 8 outer-training
            # samples (the inner folds train on 7, or on 5-6 for the stratified
            # 3-fold), so this keeps k ∈ {3, 5} and drops k=10, 20, 30.
            # k also cannot exceed the number of features.
            n_inner_train = len(X_tr)  # samples of the outer training set
            effective_k_grid = [k for k in self.k_grid
                                if k < n_inner_train and k <= len(feature_names)]
            dropped_k = [k for k in self.k_grid if k not in effective_k_grid]
            if not effective_k_grid:
                raise ValueError(f"No k in {self.k_grid} is smaller than {n_inner_train} "
                                 f"training samples and at most {len(feature_names)} features")
            if dropped_k:
                print(f"    Warning: Fold {fold_idx+1}: dropped k={dropped_k} "
                      f"(>= {n_inner_train} training samples or > {len(feature_names)} features)")

            # ---- Inner CV: select best k ----
            if self.inner_cv_mode == "loo":
                best_k, _ = _inner_loo(X_tr, y_tr,
                                       self.selector_name,
                                       self.classifier_name,
                                       effective_k_grid)
            else:
                best_k, _ = _inner_stratified3fold(X_tr, y_tr,
                                                   self.selector_name,
                                                   self.classifier_name,
                                                   effective_k_grid)

            # ---- Train final model with best k on full outer training set ----
            y_pred, selected_feat = self._train_predict(
                X_tr, y_tr, X_te, best_k, feature_names
            )

            self.outer_predictions_.append(y_pred[0])
            self.outer_true_labels_.append(y_te[0])
            self.outer_selected_k_.append(best_k)
            self.outer_selected_features_.append(selected_feat)
            self.outer_rat_ids_.append(rat_id_te)

        return self._compute_outer_metrics()

    def _train_predict(self, X_tr, y_tr, X_te, k, feature_names):
        """Fit selector on full outer training set, train classifier, predict."""
        sel = get_selector(self.selector_name)

        if self.inner_cv_mode == "stratified_3fold":
            # Architectural limitation (perm_importance only):
            # Permutation importance requires held-out samples to score feature
            # shuffles. We take one stratified split of the outer training set,
            # using ~2/3 (imp_tr_idx) to train the base estimator and ~1/3
            # (imp_val_idx) to compute importances. The selector therefore ranks
            # features on ~5 of 8 outer-training samples (at n=9), not the full 8.
            # The final classifier is still trained on the full X_tr below, but
            # the feature ranking driving selection is derived from fewer samples.
            # This inflates variance for perm_importance relative to all other
            # selectors and is an inherent cost of requiring held-out importance
            # estimation at small n. Results for this selector should be
            # interpreted with that caveat.
            skf = StratifiedKFold(n_splits=3, shuffle=True,
                                  random_state=Config.RANDOM_STATE)
            # Use first split as the importance-estimation split
            imp_tr_idx, imp_val_idx = next(skf.split(X_tr, y_tr))
            sel.fit(X_tr[imp_tr_idx], y_tr[imp_tr_idx],
                    X_val=X_tr[imp_val_idx], y_val=y_tr[imp_val_idx])
        else:
            sel.fit(X_tr, y_tr)

        feat_idx      = sel.get_top_k_features(k)
        selected_feat = [feature_names[i] for i in feat_idx]

        X_tr_k = X_tr[:, feat_idx]
        X_te_k = X_te[:, feat_idx]

        # Scale only for scale-sensitive classifiers
        if self.classifier_name in SCALE_SENSITIVE:
            scaler = StandardScaler()
            X_tr_k = scaler.fit_transform(X_tr_k)
            X_te_k = scaler.transform(X_te_k)

        clf = get_classifier(self.classifier_name)
        clf.fit(X_tr_k, y_tr)
        return clf.predict(X_te_k), selected_feat

    def _compute_outer_metrics(self) -> Dict:
        y_true = np.array(self.outer_true_labels_)
        y_pred = np.array(self.outer_predictions_)
        labels = Config.CLASS_LABELS

        macro_f1    = f1_score(y_true, y_pred, average="macro",
                               labels=labels, zero_division=0)
        micro_f1    = f1_score(y_true, y_pred, average="micro",
                               zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted",
                               labels=labels, zero_division=0)
        accuracy    = accuracy_score(y_true, y_pred)
        cm          = confusion_matrix(y_true, y_pred, labels=labels)

        pc_prec = precision_score(y_true, y_pred, average=None,
                                  labels=labels, zero_division=0)
        pc_rec  = recall_score(y_true, y_pred, average=None,
                               labels=labels, zero_division=0)
        pc_f1   = f1_score(y_true, y_pred, average=None,
                           labels=labels, zero_division=0)

        return {
            "selector":            self.selector_name,
            "classifier":          self.classifier_name,
            "inner_cv_mode":       self.inner_cv_mode,
            "macro_f1":            macro_f1,
            "micro_f1":            micro_f1,
            "weighted_f1":         weighted_f1,
            "accuracy":            accuracy,
            "per_class_precision": dict(zip(labels, pc_prec)),
            "per_class_recall":    dict(zip(labels, pc_rec)),
            "per_class_f1":        dict(zip(labels, pc_f1)),
            "confusion_matrix":    cm,
            "selected_k_distribution":        list(self.outer_selected_k_),
            "mean_k":              float(np.mean(self.outer_selected_k_)),
            "std_k":               float(np.std(self.outer_selected_k_)),
            "predictions":         list(y_pred),
            "true_labels":         list(y_true),
            "rat_ids":             list(self.outer_rat_ids_),
            "selected_features_per_fold": [list(f) for f in self.outer_selected_features_],
        }


# =============================================================================
# OUTPUT
# =============================================================================

def save_results(results_list: List[Dict], output_dir: Path):
    """Write results_summary, per_fold_details, confusion_matrices and selected_features (CSV)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = Config.CLASS_LABELS

    # ---- 1. Summary table ----
    rows = []
    for r in results_list:
        row = {
            "selector":      r["selector"],
            "classifier":    r["classifier"],
            "inner_cv_mode": r["inner_cv_mode"],
            "macro_f1":      round(r["macro_f1"], 4),
            "micro_f1":      round(r["micro_f1"], 4),
            "weighted_f1":   round(r["weighted_f1"], 4),
            "accuracy":      round(r["accuracy"], 4),
            "mean_k":        round(r["mean_k"], 2),
            "std_k":         round(r["std_k"], 2),
        }
        for lbl in labels:
            row[f"prec_{lbl}"]  = round(r["per_class_precision"][lbl], 4)
            row[f"rec_{lbl}"]   = round(r["per_class_recall"][lbl], 4)
            row[f"f1_{lbl}"]    = round(r["per_class_f1"][lbl], 4)
        rows.append(row)

    summary_df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    summary_path = output_dir / "results_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path}")

    # ---- 2. Per-fold details (with rat IDs) ----
    fold_rows = []
    for r in results_list:
        for fold_i, (true, pred, k, rat_id) in enumerate(zip(
                r["true_labels"], r["predictions"],
                r["selected_k_distribution"], r["rat_ids"])):
            fold_rows.append({
                "selector":    r["selector"],
                "classifier":  r["classifier"],
                "fold":        fold_i + 1,
                "rat_id":      rat_id,
                "true_label":  true,
                "pred_label":  pred,
                "correct":     true == pred,
                "selected_k":  k,
            })
    pd.DataFrame(fold_rows).to_csv(output_dir / "per_fold_details.csv", index=False)

    # ---- 3. Confusion matrices ----
    cm_rows = []
    for r in results_list:
        cm = r["confusion_matrix"]
        for i, tl in enumerate(labels):
            for j, pl in enumerate(labels):
                cm_rows.append({
                    "selector":    r["selector"],
                    "classifier":  r["classifier"],
                    "true_label":  tl,
                    "pred_label":  pl,
                    "count":       int(cm[i, j]),
                })
    pd.DataFrame(cm_rows).to_csv(output_dir / "confusion_matrices.csv", index=False)

    # ---- 4. Selected features ----
    feat_rows = []
    for r in results_list:
        for fold_i, (feat_list, k) in enumerate(zip(
                r["selected_features_per_fold"],
                r["selected_k_distribution"])):
            for feat in feat_list:
                feat_rows.append({
                    "selector":   r["selector"],
                    "classifier": r["classifier"],
                    "fold":       fold_i + 1,
                    "selected_k": k,
                    "feature":    feat,
                })
    pd.DataFrame(feat_rows).to_csv(output_dir / "selected_features.csv", index=False)

    print(f"All outputs in: {output_dir}")
