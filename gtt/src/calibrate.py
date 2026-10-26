#!/usr/bin/env python3
"""Calibrate the GTT profiles to concentration (pM) by a per-rat linear fit to the ELISA values.

For each rat and hormone, the 24 post-injection samples of the normalized device signal (F/Fmax) are
smoothed with a Hamming-weighted moving average (half-width 2 samples, 5 weights, even reflection at the
ends). A straight line is fitted by least squares to the rat's ELISA values on the smoothed signal at the
blood-sampling times (0, 15, 30, 45 and 60 min; Rat1 has no 45-min sample):

    C = slope * x_smoothed + intercept

The line converts the smoothed post-injection profile to pM; concentrations below 0 are set to 0. The 4
pre-injection samples (-10 to -2.5 min) are too few to filter and are calibrated unsmoothed with the same line.

Input   data/raw/gtt_post_injection.csv   24 samples per rat from the injection t = 0 to 60 min, with the
                                          glucose and ELISA values at the blood-sampling times
        data/raw/gtt_pre_injection.csv    4 samples per rat before the injection
Output  concentration_timeseries.csv      for every rat and sample: the signal, the signal the calibration was
                                          applied to, the ELISA value, the calibrated concentration
                                          (<hormone>_predicted_pM) and the unsmoothed signal through the same
                                          line (<hormone>_raw_predicted_pM)
        calibration_parameters.csv        slope, intercept and number of ELISA values of every rat and hormone

    python src/calibrate.py [--output-dir DIR]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DEFAULT_OUTPUT = ROOT / "outputs" / "calibration"
HORMONES = ["insulin", "c_peptide", "glucagon"]
HALF_WIDTH = 2


def hamming(x, half_width=HALF_WIDTH):
    """Hamming-weighted moving average with even reflection at both ends."""
    x = np.asarray(x, float)
    w = np.hamming(2 * half_width + 1)
    w /= w.sum()
    pad = (len(w) - 1) // 2
    return np.convolve(np.pad(x, pad, mode="reflect"), w, "valid")


def fit_line(signal, elisa):
    """Least-squares line of the ELISA values on the signal (slope, intercept)."""
    m = np.isfinite(signal) & np.isfinite(elisa)
    if m.sum() < 2:
        return np.nan, np.nan
    slope, intercept = np.polyfit(signal[m], elisa[m], 1)
    return float(slope), float(intercept)


def floor_at_zero(value):
    return max(float(value), 0.0) if np.isfinite(value) else np.nan


def calibrate(post, pre):
    """Calibrated profiles of all rats (rows sorted by rat and time) and the fitted lines."""
    rows, lines = [], []
    for rat in sorted(post["animal_id"].unique()):
        tp = post[post.animal_id == rat].sort_values("timepoint").reset_index(drop=True)
        bl = pre[pre.animal_id == rat].sort_values("timepoint").reset_index(drop=True)
        rat_type = tp["rat_type"].iloc[0]
        fit = {}
        for h in HORMONES:
            smoothed = hamming(tp[f"{h}_au"].values.astype(float))
            elisa = tp[f"{h}_ELISA_pM"].replace(0, np.nan).values.astype(float)
            anchors = np.isfinite(elisa)
            slope, intercept = fit_line(smoothed[anchors], elisa[anchors])
            fit[h] = (slope, intercept, smoothed)
            lines.append({"animal_id": rat, "rat_type": rat_type, "hormone": h, "n_elisa": int(anchors.sum()),
                          "slope_pM": slope, "intercept_pM": intercept})

        def emit(source, i, post_injection):
            row = {"animal_id": rat, "rat_type": rat_type, "timepoint": float(source["timepoint"].iloc[i])}
            for h in HORMONES:
                slope, intercept, smoothed = fit[h]
                signal = float(source[f"{h}_au"].iloc[i])
                calibrated_signal = float(smoothed[i]) if post_injection else signal
                elisa = np.nan
                if post_injection:
                    value = source[f"{h}_ELISA_pM"].iloc[i]
                    elisa = np.nan if (pd.isna(value) or value == 0) else float(value)
                row[f"{h}_au"] = signal
                row[f"{h}_calibration_au"] = calibrated_signal
                row[f"{h}_ELISA_pM"] = elisa
                row[f"{h}_predicted_pM"] = floor_at_zero(slope * calibrated_signal + intercept)
                row[f"{h}_raw_predicted_pM"] = floor_at_zero(slope * signal + intercept)
            rows.append(row)

        for i in range(len(bl)):
            emit(bl, i, post_injection=False)
        for i in range(len(tp)):
            emit(tp, i, post_injection=True)
    profiles = pd.DataFrame(rows).sort_values(["animal_id", "timepoint"]).reset_index(drop=True)
    return profiles, pd.DataFrame(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help="folder for the two output CSVs")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    profiles, lines = calibrate(pd.read_csv(RAW / "gtt_post_injection.csv"), pd.read_csv(RAW / "gtt_pre_injection.csv"))
    profiles.to_csv(output_dir / "concentration_timeseries.csv", index=False)
    lines.to_csv(output_dir / "calibration_parameters.csv", index=False, float_format="%.6f")
    print(f"{profiles.animal_id.nunique()} rats, {len(profiles)} samples; "
          f"{(lines.n_elisa < 5).sum()} of {len(lines)} lines fitted to 4 ELISA values (Rat1), the rest to 5")
    print(f"  Saved: {output_dir / 'concentration_timeseries.csv'}")
    print(f"  Saved: {output_dir / 'calibration_parameters.csv'}")


if __name__ == "__main__":
    main()
