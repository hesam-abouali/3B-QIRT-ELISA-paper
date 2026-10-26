#!/usr/bin/env python3
"""Extract features from the fed-state concentration profiles and build the classification matrix.

Input   data/concentrations/concentration_timeseries.csv   calibrated profiles (conc_per_rat_pM)
        data/raw/fed_state_raw_data.csv                    glucose and ELISA values of each rat

Output  fedstate_features_all.csv   9 rats x 189 columns, the feature table of SI Section S9
        fedstate_X.csv              9 rats x 161 features, the classification matrix
        fedstate_y.csv              phenotype label of each rat
        fedstate_meta.csv           rat ID, label and the summary columns left out of the classification matrix

1. The calibrated profiles are put in the input format of fed_state_features.py together with each rat's measured glucose and conventional ELISA values.
The profile columns keep the names *_au but hold pM.
Negative concentrations would be set to 0 if they exist; there are no negative concentrations in the current data.

2. fed_state_features.py computes the features. Columns named *_raw_au are renamed *_raw_pM.

3. The classification matrix is the feature table excluding: the rat ID and label, the ELISA flag, the
   overall function score and class, the ELISA-based features (Category 8) and the start and end glucose values.
   Infinite values would be replaced by the column maximum and missing values by the column median; there are none in the current data.
   One column of the matrix (corr_insulin_glucagon_significant) is constant, and the classification drops it,
   which leaves 160 candidate features.
"""
import argparse
import filecmp
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from fed_state_features import ComprehensiveFedStateAnalyzer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONCENTRATIONS = ROOT / "data" / "concentrations" / "concentration_timeseries.csv"
DEFAULT_RAW = ROOT / "data" / "raw" / "fed_state_raw_data.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "feature_extraction"
REFERENCE_DIR = ROOT / "data" / "features"

LABEL_COL = "rat_type"
ID_COL = "animal_id"

# Identifiers and summary columns kept out of the matrix (written to fedstate_meta.csv)
COMPROMISING_COLS = [
    ID_COL,
    LABEL_COL,
    "overall_function_score",
    "function_classification",
    "has_elisa",
]

# Start and end glucose; the mean and change of glucose stay in the matrix
GLUCOSE_REDUNDANT = [
    "glucose_start_mM",
    "glucose_end_mM",
]


def _pivot_calibrated(conc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot concentration_timeseries.csv (long format) to wide format.
    Uses conc_per_rat_pM as the primary calibration output.
    Output: one row per (animal_id, rat_type, timepoint) with
            insulin_au / c_peptide_au / glucagon_au columns
            (renamed so the analyzer accepts them; values are in pM).
    """
    slim = conc_df[["animal_id", "rat_type", "timepoint",
                     "hormone", "conc_per_rat_pM"]].copy()

    wide = slim.pivot_table(
        index=["animal_id", "rat_type", "timepoint"],
        columns="hormone",
        values="conc_per_rat_pM",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None

    # Clamp negatives (can arise from 2-point extrapolation at low signal; non existent in the current work)
    hormone_cols = [c for c in wide.columns if c in ("insulin", "c_peptide", "glucagon")]
    n_neg = 0
    for col in hormone_cols:
        mask = wide[col] < 0
        n_neg += int(mask.sum())
        if mask.any():
            print(f"    ⚠  {col}: {mask.sum()} negative pM values → clamped to 0")
            wide.loc[mask, col] = 0.0
    if n_neg == 0:
        print("    No negative pM values.")

    # Rename to analyzer expected column names
    rename = {"insulin": "insulin_au",
               "c_peptide": "c_peptide_au",
               "glucagon": "glucagon_au"}
    wide = wide.rename(columns=rename)

    return wide


def _merge_metadata(signal_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Merge glucose and ELISA anchor columns from the raw CSV."""
    meta_cols = ["animal_id", "glucose_start_mM", "glucose_end_mM",
                 "insulin_start_ELISA_pM", "insulin_end_ELISA_pM",
                 "c_peptide_start_ELISA_pM", "c_peptide_end_ELISA_pM",
                 "glucagon_start_ELISA_pM", "glucagon_end_ELISA_pM"]
    avail = [c for c in meta_cols if c in raw_df.columns]
    meta = raw_df[avail].drop_duplicates(subset=["animal_id"])
    merged = signal_df.merge(meta, on="animal_id", how="left")
    return merged


def _rename_raw_au_to_pM(csv_path: str):
    """Post-process: rename _raw_au → _raw_pM for calibrated extraction."""
    df = pd.read_csv(csv_path)
    renames = {c: c.replace("_raw_au", "_raw_pM")
               for c in df.columns if "_raw_au" in c}
    if renames:
        df = df.rename(columns=renames)
        df.to_csv(csv_path, index=False)
        print(f"    Renamed {len(renames)} columns: _raw_au → _raw_pM")
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Separate df into X (predictors), y (labels), meta (IDs + excluded cols).
    Handles booleans, inf, and NaN.
    """
    # Bool → int
    bool_cols = [c for c in df.columns if df[c].dtype == bool]
    for c in bool_cols:
        df[c] = df[c].astype(int)

    # Drop unnamed index artifacts
    df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]

    # y and meta
    y    = df[[LABEL_COL]].copy()
    drop = [c for c in COMPROMISING_COLS if c in df.columns]
    meta = df[[c for c in drop if c in df.columns]].copy()
    X    = df.drop(columns=drop, errors="ignore")

    # Non-numeric columns → warn and drop
    non_num = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_num:
        print(f"    ⚠  Dropping non-numeric columns: {non_num}")
        X = X.drop(columns=non_num)

    # Inf → column finite max
    n_inf = np.isinf(X).sum().sum()
    if n_inf > 0:
        inf_cols = [c for c in X.columns if np.isinf(X[c]).any()]
        print(f"    Fixing {n_inf} inf values in {len(inf_cols)} columns ...")
        for col in inf_cols:
            finite_max = X.loc[np.isfinite(X[col]), col].max()
            X.loc[np.isinf(X[col]), col] = finite_max

    # NaN → column median
    n_nan = X.isna().sum().sum()
    if n_nan > 0:
        print(f"    Filling {n_nan} NaN values with column median ...")
        X = X.fillna(X.median())

    return X, y, meta


def extract_features(concentrations_csv, raw_csv, out_csv):
    """Write the 9 x 189 feature table computed from the calibrated profiles."""
    conc_df = pd.read_csv(concentrations_csv)
    raw_df = pd.read_csv(raw_csv)
    calib_wide = _merge_metadata(_pivot_calibrated(conc_df), raw_df)

    with tempfile.TemporaryDirectory() as tmp:
        input_csv = os.path.join(tmp, "calib_input.csv")
        calib_wide.to_csv(input_csv, index=False)
        analyzer = ComprehensiveFedStateAnalyzer()
        analyzer.load_csv_data(input_csv, format_type="long")
        analyzer.analyze_all_animals()
        analyzer.export_metrics_to_csv(str(out_csv))
    return _rename_raw_au_to_pM(str(out_csv))


def build_matrix(features_csv, output_dir):
    """Write fedstate_X/y/meta.csv from the feature table."""
    X, y, meta = clean(pd.read_csv(features_csv))
    elisa_cols = [c for c in X.columns if "_elisa_" in c]
    X = X.drop(columns=elisa_cols + GLUCOSE_REDUNDANT)
    X.to_csv(output_dir / "fedstate_X.csv", index=False)
    y.to_csv(output_dir / "fedstate_y.csv", index=False)
    meta.to_csv(output_dir / "fedstate_meta.csv", index=False)
    print(f"  Classification matrix: {X.shape[0]} rats x {X.shape[1]} features "
          f"({len(elisa_cols)} ELISA-based and {len(GLUCOSE_REDUNDANT)} glucose columns left out)")
    return X


def compare_with_reference(output_dir):
    """Report whether each output file matches its copy in data/features/."""
    print(f"\nComparison with {REFERENCE_DIR.relative_to(ROOT)}/:")
    for name in ("fedstate_features_all.csv", "fedstate_X.csv", "fedstate_y.csv", "fedstate_meta.csv"):
        new, ref = output_dir / name, REFERENCE_DIR / name
        if filecmp.cmp(new, ref, shallow=False):
            result = "identical"
        else:
            a, b = pd.read_csv(new), pd.read_csv(ref)
            num = a.select_dtypes("number").columns
            same = (a.shape == b.shape and a.columns.equals(b.columns)
                    and a.drop(columns=num).equals(b.drop(columns=num))
                    and np.allclose(a[num], b[num], rtol=1e-9, atol=0, equal_nan=True))
            result = "equal within rounding (relative 1e-9)" if same else "DIFFERENT"
        print(f"  {name}: {result}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--concentrations", default=str(DEFAULT_CONCENTRATIONS), help="calibrated profiles CSV")
    parser.add_argument("--raw", default=str(DEFAULT_RAW), help="raw fed-state CSV (glucose and ELISA values)")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help="folder for the four output CSVs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features = extract_features(args.concentrations, args.raw, output_dir / "fedstate_features_all.csv")
    print(f"  Feature table: {features.shape[0]} rats x {features.shape[1]} columns")
    build_matrix(output_dir / "fedstate_features_all.csv", output_dir)
    compare_with_reference(output_dir)


if __name__ == "__main__":
    main()
