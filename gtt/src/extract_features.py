#!/usr/bin/env python3
"""Extract the GTT features from the calibrated profiles and build the classification matrix.

Input   data/concentrations/concentration_timeseries.csv   calibrated profiles (<hormone>_predicted_pM)
        data/raw/gtt_post_injection.csv                    glucose at the blood-sampling times
Output  gtt_features_all.csv   9 rats x 192 features, the GTT feature table
        gtt_X.csv              9 rats x 181 features, the classification matrix
        gtt_y.csv              phenotype label of each rat
        gtt_meta.csv           rat ID and label

1. gtt_features.py computes the features from the Hamming-smoothed profiles of all nine rats.
   glucagon_rebound_ratio is set to missing where the glucagon nadir is at the floor of 0 pM
   (nadir <= 1e-3 pM), where the ratio is undefined.
2. The classification matrix is the feature table without the 7 clinical indices computed from the ELISA
   insulin (columns ending in _elisa) and 4 columns that are constant or empty across the rats. Missing
   values stay missing; the classification imputes them inside each training set.

The outputs are compared with the files in data/features/.

    python src/extract_features.py [--output-dir DIR]
"""
import argparse
import filecmp
from pathlib import Path

import numpy as np
import pandas as pd

from gtt_features import GTTFeatureExtractor

ROOT = Path(__file__).resolve().parents[1]
CONCENTRATIONS = ROOT / "data" / "concentrations" / "concentration_timeseries.csv"
GLUCOSE = ROOT / "data" / "raw" / "gtt_post_injection.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "feature_extraction"
REFERENCE_DIR = ROOT / "data" / "features"
RATS = [f"Rat{i}" for i in range(1, 10)]
NADIR_FLOOR_PM = 1e-3
CONSTANT_OR_EMPTY = ["first_phase_end_min", "glucagon_peak_right_censored", "glucagon_return_to_baseline",
                     "glucagon_time_to_half_recovery_min"]


def extract_features(output_dir):
    """Write the 9 x 192 feature table; return it."""
    path = output_dir / "gtt_features_all.csv"
    GTTFeatureExtractor(fmax_csv=CONCENTRATIONS, glucose_csv=GLUCOSE, signal_kind="windowed", animals=RATS).run(path)
    table = pd.read_csv(path)
    table.loc[table["glucagon_nadir_pM"] <= NADIR_FLOOR_PM, "glucagon_rebound_ratio"] = np.nan
    table.to_csv(path, index=False)
    return table


def build_matrix(table, output_dir):
    """Write gtt_X/y/meta.csv from the feature table."""
    table = table.set_index("animal_id")
    left_out = [c for c in table.columns if c.endswith("_elisa")] + CONSTANT_OR_EMPTY
    X = table.loc[RATS].drop(columns=["rat_type"] + left_out).astype(float)
    meta = pd.DataFrame({"animal_id": RATS, "rat_type": table.loc[RATS, "rat_type"].tolist()})
    X.reset_index(drop=True).to_csv(output_dir / "gtt_X.csv", index=False)
    meta.to_csv(output_dir / "gtt_meta.csv", index=False)
    meta[["rat_type"]].to_csv(output_dir / "gtt_y.csv", index=False)
    print(f"  Classification matrix: {X.shape[0]} rats x {X.shape[1]} features "
          f"({len(left_out)} left out: {len(left_out) - len(CONSTANT_OR_EMPTY)} ELISA-based, "
          f"{len(CONSTANT_OR_EMPTY)} constant or empty); {int(X.isna().sum().sum())} missing values")


def compare_with_reference(output_dir):
    """Report whether each output file matches its copy in data/features/."""
    print(f"\nComparison with {REFERENCE_DIR.relative_to(ROOT)}/:")
    for name in ("gtt_features_all.csv", "gtt_X.csv", "gtt_y.csv", "gtt_meta.csv"):
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
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help="folder for the four output CSVs")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    build_matrix(extract_features(output_dir), output_dir)
    compare_with_reference(output_dir)


if __name__ == "__main__":
    main()
