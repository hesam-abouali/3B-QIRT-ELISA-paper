#!/usr/bin/env python3
"""Fixed-recipe label-permutation test of the two reported fed-state recipes, mRMR + LogReg_L2 and
mRMR + SVM_linear, with B = 1000 draws (Supplementary Information, Section S10).

For draw b = 0..B-1 the nine labels are permuted as y[RandomState(42 + b + 1).permutation(9)]; the
complete nested cross-validation of the recipe, including the choice of the number of features, is rerun
on the permuted labels and its macro-F1 recorded. p = (1 + r) / (1 + B), where r is the number of draws
whose macro-F1 reaches the observed value. Identical label assignments are computed once and keep their
multiplicity in the null distribution. The mRMR selector fits are memoized by their exact inputs
(mrmr_cache.py), so the two recipes share them; this does not change any result.

Input   data/features/fedstate_X.csv, fedstate_y.csv, fedstate_meta.csv
Output  checkpoints/ (one JSON per label assignment), fingerprint.txt, per_draw_<recipe>.csv, summary.csv,
        environment.json

    python src/run_permutation_test.py [--B 1000] [--workers 8] [--output-dir DIR]
    python src/run_permutation_test.py --draws 0,9,894,999     # spot check of single draws

The full test computes 749 distinct label assignments, about 1.5 min each per process. An interrupted run
resumes from its checkpoints when it is started again with the same --output-dir. A checkpoint is reused
only if its draw and labels are the expected ones and fingerprint.txt matches the current inputs, code
and package versions; checkpoints are written to a temporary file and then renamed, so an interrupted
write never leaves a partial one. The per-draw results are compared with data/permutation/; a spot check
compares the true labels with data/classification/ and each requested draw with its row in
data/permutation/.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse
import contextlib
import hashlib
import io
import json
import multiprocessing
import platform
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data" / "features"
DEFAULT_OUTPUT = ROOT / "outputs" / "permutation"
REFERENCE_DIR = ROOT / "data" / "permutation"
CLASSIFICATION_DIR = ROOT / "data" / "classification"
warnings.filterwarnings("ignore")
RECIPES = ("LogReg_L2", "SVM_linear")
SELECTOR = "mrmr"
FIELDS = ("macro_f1", "accuracy", "selected_k_distribution", "predictions", "selected_features_per_fold")
FINGERPRINT_FILES = (DATA / "fedstate_X.csv", DATA / "fedstate_y.csv", DATA / "fedstate_meta.csv",
                     HERE / "nested_cv.py", HERE / "mrmr_cache.py", Path(__file__).resolve())
_engine = None


def load_data():
    X = pd.read_csv(DATA / "fedstate_X.csv")
    y = pd.read_csv(DATA / "fedstate_y.csv")["rat_type"]
    meta = pd.read_csv(DATA / "fedstate_meta.csv")
    assert y.tolist() == meta.rat_type.tolist()
    X = X.drop(columns=X.columns[X.var() == 0].tolist())
    return X, y, meta.animal_id.tolist()


def labels_for(b):
    y = load_data()[1].to_numpy()
    return tuple(y if b is None else y[np.random.RandomState(42 + b + 1).permutation(len(y))])


def engine():
    global _engine
    if _engine is None:
        import nested_cv
        import mrmr_cache
        mrmr_cache.install(nested_cv)
        _engine = nested_cv
    return _engine


def clean(v):
    if isinstance(v, np.ndarray):
        return [clean(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    return v


def compute(b):
    """Both recipes for the labels of draw b (None = true labels)."""
    base = engine()
    X, _, rats = load_data()
    y = pd.Series(list(labels_for(b)), name="rat_type")
    out = {"draw": b, "labels": list(y)}
    for recipe in RECIPES:
        pipe = base.NestedCVPipeline(SELECTOR, recipe, base.Config.K_GRID)
        with contextlib.redirect_stdout(io.StringIO()):
            res = pipe.run(X, y, list(X.columns), rats)
        out[recipe] = {f: clean(res[f]) for f in FIELDS}
    return out


def fingerprint():
    """Hash of the inputs, the code and the package versions that the checkpoints depend on."""
    import scipy, sklearn
    h = hashlib.sha256()
    for p in FINGERPRINT_FILES:
        h.update(p.read_bytes())
    versions = [sys.version, platform.machine(), np.__version__, scipy.__version__, pd.__version__, sklearn.__version__]
    h.update(json.dumps(versions).encode())
    return h.hexdigest()


def check_fingerprint(out, ck):
    """Stop rather than mix checkpoints computed with other inputs, code or package versions."""
    fp_path = out / "fingerprint.txt"
    current = fingerprint()
    if fp_path.exists():
        if fp_path.read_text().strip() != current:
            sys.exit(f"{out} holds checkpoints of other inputs, code or package versions; use a new --output-dir")
    elif any(ck.glob("*.json")):
        sys.exit(f"{out} holds checkpoints without a fingerprint; use a new --output-dir")
    else:
        fp_path.write_text(current + "\n")


def valid_result(r, n):
    """True if a recipe's stored result is complete and its scores are finite and in [0, 1]."""
    if not isinstance(r, dict) or not all(f in r for f in FIELDS):
        return False
    scores_ok = all(isinstance(r[s], (int, float)) and 0.0 <= r[s] <= 1.0 for s in ("macro_f1", "accuracy"))
    lengths_ok = all(isinstance(r[f], list) and len(r[f]) == n
                     for f in ("selected_k_distribution", "predictions", "selected_features_per_fold"))
    return scores_ok and lengths_ok


def load_checkpoint(path, b):
    """The checkpoint of draw b, or None if it is missing, unreadable, incomplete or for other labels."""
    try:
        res = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    labels = labels_for(b)
    if (not isinstance(res, dict) or res.get("draw") != b or not isinstance(res.get("labels"), list)
            or tuple(res["labels"]) != labels or not all(valid_result(res.get(r), len(labels)) for r in RECIPES)):
        return None
    return res


def write_checkpoint(path, res):
    """Write to a temporary file and rename it, so that an interrupted write never leaves a partial checkpoint."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(res, indent=1) + "\n")
    os.replace(tmp, path)


def read_reference(path):
    """Read a reference CSV with exact float parsing, so that values compare equal to the computed ones."""
    return pd.read_csv(path, float_precision="round_trip")


def compare_true_labels(observed):
    """Compare the true-label run with the paper's classification run in data/classification/
    (results_summary gives macro-F1 to four decimals)."""
    summary = read_reference(CLASSIFICATION_DIR / "results_summary.csv")
    folds = read_reference(CLASSIFICATION_DIR / "per_fold_details.csv")
    for recipe in RECIPES:
        row = summary[(summary.selector == SELECTOR) & (summary.classifier == recipe)].iloc[0]
        fold = folds[(folds.selector == SELECTOR) & (folds.classifier == recipe)].sort_values("fold")
        same = (round(observed[recipe]["macro_f1"], 4) == row.macro_f1
                and observed[recipe]["predictions"] == fold.pred_label.tolist()
                and observed[recipe]["selected_k_distribution"] == fold.selected_k.tolist())
        print(f'  true labels, mRMR + {recipe}: {"identical" if same else "DIFFERENT"} '
              f'(macro-F1 {observed[recipe]["macro_f1"]:.3f})')


def compare_draws(results):
    """Compare spot-check draws with their rows in data/permutation/."""
    for recipe in RECIPES:
        ref = read_reference(REFERENCE_DIR / f"per_draw_{recipe}.csv").set_index("draw")
        for res in sorted(results, key=lambda r: r["draw"]):
            r, row = res[recipe], ref.loc[res["draw"]]
            same = (r["macro_f1"] == row.macro_f1 and " ".join(r["predictions"]) == row.predictions
                    and " ".join(map(str, r["selected_k_distribution"])) == row.selected_k)
            print(f'  draw {res["draw"]}, mRMR + {recipe}: {"identical" if same else "DIFFERENT"}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--draws", help="comma-separated draws to compute only (spot check); no summary")
    a = ap.parse_args()
    out = Path(a.output_dir)
    ck = out / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    check_fingerprint(out, ck)
    path = lambda b: ck / ("true_labels.json" if b is None else f"draw_{b:04d}.json")

    def checkpoint(b):
        res = load_checkpoint(path(b), b)
        assert res is not None, f"checkpoint of draw {b} is missing or invalid"
        return res

    groups = {}
    for b in range(a.B):
        groups.setdefault(labels_for(b), []).append(b)
    rep_of = {lab: draws[0] for lab, draws in groups.items()}
    todo = [None] + ([int(x) for x in a.draws.split(",")] if a.draws else sorted(rep_of.values()))
    todo = [b for b in todo if load_checkpoint(path(b), b) is None]
    print(f"{a.B} draws -> {len(groups)} distinct label assignments; {len(todo)} to compute with {a.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(compute, b): b for b in todo}
        for i, f in enumerate(as_completed(futures), 1):
            res = f.result()
            write_checkpoint(path(res["draw"]), res)
            if i % 10 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} done", flush=True)

    observed = checkpoint(None)
    print(f'\nComparison with {CLASSIFICATION_DIR.relative_to(ROOT)}/ and {REFERENCE_DIR.relative_to(ROOT)}/:')
    compare_true_labels(observed)
    if a.draws:
        compare_draws([checkpoint(int(x)) for x in a.draws.split(",")])
        return

    import scipy, sklearn
    rows = []
    for recipe in RECIPES:
        obs = observed[recipe]["macro_f1"]
        per = []
        for b in range(a.B):
            lab = labels_for(b)
            r = checkpoint(rep_of[lab])[recipe]
            per.append({"draw": b, "assignment": "".join(x[0].upper() for x in lab), "computed_at_draw": rep_of[lab],
                        "macro_f1": r["macro_f1"], "accuracy": r["accuracy"],
                        "selected_k": " ".join(map(str, r["selected_k_distribution"])),
                        "predictions": " ".join(r["predictions"])})
        d = pd.DataFrame(per)
        d.to_csv(out / f"per_draw_{recipe}.csv", index=False)
        if a.B == 1000:
            same = d.equals(read_reference(REFERENCE_DIR / f"per_draw_{recipe}.csv"))
            print(f'  per_draw_{recipe}.csv: {"identical" if same else "DIFFERENT"} (1000 draws)')
        v = d.macro_f1.to_numpy()
        r_ge = int((v >= obs - 1e-12).sum())
        rows.append({"recipe": f"mRMR + {recipe}", "B": a.B, "distinct_assignments": len(groups), "observed_macro_f1": obs,
                     "null_mean": v.mean(), "null_sd": v.std(), "null_median": float(np.median(v)),
                     "null_p95": float(np.percentile(v, 95)), "null_max": float(v.max()),
                     "draws_reaching_observed": r_ge, "p_value": (1 + r_ge) / (1 + a.B)})
    s = pd.DataFrame(rows)
    s.to_csv(out / "summary.csv", index=False)
    (out / "environment.json").write_text(json.dumps(
        {"python": sys.version, "machine": platform.machine(), "numpy": np.__version__, "scipy": scipy.__version__,
         "pandas": pd.__version__, "scikit-learn": sklearn.__version__}, indent=1) + "\n")
    print(s.T.to_string(), flush=True)


if __name__ == "__main__":
    main()
