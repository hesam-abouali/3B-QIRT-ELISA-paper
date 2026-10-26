#!/usr/bin/env python3
"""This script converts the normalized data (Fi/F0) of the QIRT-ELISA in the fed-state experiment
to concentrations (pM) by a per-rat, per-hormone two-point ELISA calibration.

Conventional ELISA measurements (e) were taken at the first and last time points of the experimental profile
(Supplementary Information, Section S8):

    slope = (e30 - e0) / (x30 - x0),    intercept = e0 - slope * x0,    C(t) = slope * x(t) + intercept

A rat-hormone pair with a missing or non-positive ELISA value is left uncalibrated, and a pair whose
signal changes by less than 1 % between the two anchors is set to the mean of its two ELISA values.
Neither case occurs in the current data of the paper.

Input   normalized data stored in data/raw/fed_state_raw_data.csv
Output  concentration_timeseries.csv   calibrated profile (conc_per_rat_pM) of every rat, hormone and time point
        calibration_parameters.csv     ELISA anchors, relative signal change, slope, intercept and flag of every
                                       rat and hormone
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "raw" / "fed_state_raw_data.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "calibration"

HORMONES = ["insulin", "c_peptide", "glucagon"]

SIGNAL_COLS = {
    "insulin": "insulin_au",
    "c_peptide": "c_peptide_au",
    "glucagon": "glucagon_au",
}

ELISA_START_COLS = {
    "insulin": "insulin_start_ELISA_pM",
    "c_peptide": "c_peptide_start_ELISA_pM",
    "glucagon": "glucagon_start_ELISA_pM",
}

ELISA_END_COLS = {
    "insulin": "insulin_end_ELISA_pM",
    "c_peptide": "c_peptide_end_ELISA_pM",
    "glucagon": "glucagon_end_ELISA_pM",
}

FLAT_SIGNAL_TOL = 0.01  # 1% relative tolerance for flat signal detection
ANCHOR_START_TIME = 0.0
ANCHOR_END_TIME = 30.0


def load_and_validate(
    csv_path,
    flat_signal_tol=FLAT_SIGNAL_TOL,
    anchor_start_time=ANCHOR_START_TIME,
    anchor_end_time=ANCHOR_END_TIME,
):
    """Load input CSV and validate data quality for each rat × hormone combination.

    Parameters
    ----------
    csv_path : str
        Path to the input CSV file.
    flat_signal_tol : float
        Relative tolerance for flat-signal detection. Anchor pairs whose
        |Δsignal| / max(|signal|) is below this value are flagged as
        'flat_signal' and will be assigned a concentration equal to the mean of their two ELISA values.
    anchor_start_time, anchor_end_time : float
        Explicit signal timepoints paired with the ELISA start/end anchors.
        Each rat must have exactly one row at each anchor time.

    Returns
    -------
    df : pd.DataFrame
        The loaded dataframe.
    anchors : dict
        {animal_id: {hormone: {'signal_start', 'signal_end',
                                'elisa_start', 'elisa_end', 'flag'}}}
    """
    df = pd.read_csv(csv_path)

    required_cols = (
        ["animal_id", "rat_type", "timepoint"]
        + list(SIGNAL_COLS.values())
        + list(ELISA_START_COLS.values())
        + list(ELISA_END_COLS.values())
    )
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    animals = df["animal_id"].unique()
    anchors = {}
    quality_rows = []
    anchor_start_time = float(anchor_start_time)
    anchor_end_time = float(anchor_end_time)
    if np.isclose(anchor_start_time, anchor_end_time):
        raise ValueError("Anchor start and end timepoints must be different.")

    for aid in sorted(animals):
        adf = df[df["animal_id"] == aid]
        rat_type = adf["rat_type"].iloc[0]
        anchors[aid] = {}

        start_mask = np.isclose(adf["timepoint"].astype(float), anchor_start_time)
        end_mask = np.isclose(adf["timepoint"].astype(float), anchor_end_time)
        t0_rows = adf[start_mask]
        t30_rows = adf[end_mask]

        anchor_errors = []
        if len(t0_rows) != 1:
            anchor_errors.append(
                f"{len(t0_rows)} rows at start anchor t={anchor_start_time:g}"
            )
        if len(t30_rows) != 1:
            anchor_errors.append(
                f"{len(t30_rows)} rows at end anchor t={anchor_end_time:g}"
            )
        if anchor_errors:
            details = "; ".join(anchor_errors)
            available = ", ".join(f"{t:g}" for t in sorted(adf["timepoint"].unique()))
            raise ValueError(
                f"{aid} anchor timepoint validation failed: {details}. "
                f"Available timepoints: {available}"
            )

        t0 = t0_rows.iloc[0]
        t30 = t30_rows.iloc[0]

        for h in HORMONES:
            sig_start = t0[SIGNAL_COLS[h]]
            sig_end = t30[SIGNAL_COLS[h]]
            elisa_start = t0[ELISA_START_COLS[h]]
            elisa_end = t30[ELISA_END_COLS[h]]

            flag = "ok"

            # Check ELISA validity
            if pd.isna(elisa_start) or pd.isna(elisa_end):
                flag = "elisa_missing"
            elif elisa_start <= 0 or elisa_end <= 0:
                flag = "elisa_missing"

            # Check flat signal
            if flag == "ok":
                denom = max(abs(sig_start), abs(sig_end), 1e-12)
                if abs(sig_end - sig_start) / denom < flat_signal_tol:
                    flag = "flat_signal"

            anchors[aid][h] = {
                "time_start": anchor_start_time,
                "time_end": anchor_end_time,
                "signal_start": sig_start,
                "signal_end": sig_end,
                "elisa_start": elisa_start,
                "elisa_end": elisa_end,
                "flag": flag,
            }

            quality_rows.append({
                "animal_id": aid, "rat_type": rat_type, "hormone": h,
                "time_start": anchor_start_time,
                "time_end": anchor_end_time,
                "signal_start": sig_start, "signal_end": sig_end,
                "elisa_start": elisa_start, "elisa_end": elisa_end,
                "flag": flag,
            })

    quality_df = pd.DataFrame(quality_rows)

    print("\n" + "=" * 80)
    print("DATA QUALITY SUMMARY")
    print("=" * 80)
    print(quality_df.to_string(index=False, float_format="%.4f"))
    print()

    flag_counts = quality_df["flag"].value_counts()
    for f, n in flag_counts.items():
        print(f"  {f}: {n} rat×hormone combinations")
    print("=" * 80 + "\n")

    return df, anchors


def per_rat_calibration(df, anchors):
    """Fit a 2-point linear calibration for each rat × hormone.

    Returns
    -------
    rat_conc : dict
        {animal_id: {hormone: np.array of concentrations (pM) for all timepoints}}
    rat_params : dict
        {animal_id: {hormone: {'slope', 'intercept', 'flag'}}}
    """
    rat_conc = {}
    rat_params = {}

    for aid in sorted(anchors.keys()):
        adf = df[df["animal_id"] == aid].sort_values("timepoint")
        rat_conc[aid] = {}
        rat_params[aid] = {}

        for h in HORMONES:
            anc = anchors[aid][h]
            signal = adf[SIGNAL_COLS[h]].values

            if anc["flag"] == "elisa_missing" or anc.get("flag") == "missing_timepoints":
                rat_conc[aid][h] = np.full(len(signal), np.nan)
                rat_params[aid][h] = {
                    "slope": np.nan, "intercept": np.nan, "flag": anc["flag"]
                }
                continue

            if anc["flag"] == "flat_signal":
                mean_elisa = (anc["elisa_start"] + anc["elisa_end"]) / 2.0
                rat_conc[aid][h] = np.full(len(signal), mean_elisa)
                rat_params[aid][h] = {
                    "slope": 0.0, "intercept": mean_elisa, "flag": "flat_signal"
                }
                continue

            # Normal 2-point calibration
            ds = anc["signal_end"] - anc["signal_start"]
            slope = (anc["elisa_end"] - anc["elisa_start"]) / ds
            intercept = anc["elisa_start"] - slope * anc["signal_start"]

            conc = slope * signal + intercept
            rat_conc[aid][h] = conc
            rat_params[aid][h] = {
                "slope": slope, "intercept": intercept, "flag": "ok"
            }

    return rat_conc, rat_params


def save_concentration_timeseries(df, rat_conc, output_dir):
    """Save the calibrated profiles in long format: one row per rat, hormone and time point."""
    rows = []
    for aid in sorted(rat_conc.keys()):
        adf = df[df["animal_id"] == aid].sort_values("timepoint")
        pheno = adf["rat_type"].iloc[0]
        for h in HORMONES:
            for t, conc in zip(adf["timepoint"].values, rat_conc[aid][h]):
                rows.append({
                    "animal_id": aid,
                    "rat_type": pheno,
                    "timepoint": t,
                    "hormone": h,
                    "conc_per_rat_pM": conc,
                })
    out_path = os.path.join(output_dir, "concentration_timeseries.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False, float_format="%.6f")
    print(f"  Saved: {out_path}")


def save_calibration_parameters(df, anchors, rat_params, output_dir):
    """Save the anchors and the fitted line of every rat and hormone."""
    rat_type = df.groupby("animal_id")["rat_type"].first()
    rows = []
    for aid in sorted(rat_params.keys()):
        for h in HORMONES:
            anc, par = anchors[aid][h], rat_params[aid][h]
            rows.append({
                "animal_id": aid,
                "rat_type": rat_type[aid],
                "hormone": h,
                "signal_t0": anc["signal_start"],
                "signal_t30": anc["signal_end"],
                "elisa_t0_pM": anc["elisa_start"],
                "elisa_t30_pM": anc["elisa_end"],
                "relative_signal_change": abs(anc["signal_end"] - anc["signal_start"])
                / max(abs(anc["signal_start"]), abs(anc["signal_end"]), 1e-12),
                "slope_pM": par["slope"],
                "intercept_pM": par["intercept"],
                "flag": par["flag"],
            })
    out_path = os.path.join(output_dir, "calibration_parameters.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False, float_format="%.6f")
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="raw fed-state CSV")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help="folder for the two output CSVs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df, anchors = load_and_validate(args.input)
    rat_conc, rat_params = per_rat_calibration(df, anchors)
    save_concentration_timeseries(df, rat_conc, args.output_dir)
    save_calibration_parameters(df, anchors, rat_params, args.output_dir)


if __name__ == "__main__":
    main()
