#!/usr/bin/env python3
"""Check of the GTT calibration by withholding each ELISA value in turn (main-text Fig. 5 d–e).

For each LZ and ZDF rat and hormone, one ELISA value is left out, the calibration line is refitted to the
rat's other ELISA values (on the same Hamming-smoothed signal as in calibrate.py), and the left-out value
is predicted from the device signal at its sampling time (negative predictions are set to 0). This gives
29 held-out values per hormone (Rat1 has no 45-min sample). Agreement with ELISA is summarized by Lin's
concordance correlation coefficient (CCC), the mean absolute error (MAE), the mean bias (device minus
ELISA) and the 95% limits of agreement (bias +/- 1.96 SD of the differences).

Input   data/raw/gtt_post_injection.csv
Output  held_out_elisa.csv           one row per held-out ELISA value: rat, hormone, time, ELISA, prediction
        held_out_elisa_summary.csv   CCC, MAE, bias and limits of agreement of each hormone

    python src/held_out_elisa.py [--output-dir DIR]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from calibrate import HORMONES, fit_line, floor_at_zero, hamming

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DEFAULT_OUTPUT = ROOT / "outputs" / "calibration"
SCORED = ["healthy", "diabetic"]  # the OZ rats are not part of this check


def held_out_predictions(post):
    rows = []
    for rat in sorted(post["animal_id"].unique()):
        tp = post[post.animal_id == rat].sort_values("timepoint").reset_index(drop=True)
        if tp["rat_type"].iloc[0] not in SCORED:
            continue
        for h in HORMONES:
            smoothed = hamming(tp[f"{h}_au"].values.astype(float))
            elisa = tp[f"{h}_ELISA_pM"].replace(0, np.nan).values.astype(float)
            anchors = np.flatnonzero(np.isfinite(elisa))
            for left_out in anchors:
                keep = anchors[anchors != left_out]
                slope, intercept = fit_line(smoothed[keep], elisa[keep])
                rows.append({"animal_id": rat, "rat_type": tp["rat_type"].iloc[0], "hormone": h,
                             "timepoint": float(tp["timepoint"].iloc[left_out]), "elisa_pM": float(elisa[left_out]),
                             "predicted_pM": floor_at_zero(slope * smoothed[left_out] + intercept),
                             "unfloored_pM": float(slope * smoothed[left_out] + intercept)})
    return pd.DataFrame(rows)


def lin_ccc(x, y):
    """Lin's concordance correlation coefficient (population moments)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    covariance = np.mean((x - x.mean()) * (y - y.mean()))
    return 2 * covariance / (x.var() + y.var() + (x.mean() - y.mean()) ** 2)


def summary(predictions):
    rows = []
    for h in HORMONES:
        d = predictions[predictions.hormone == h]
        difference = d.predicted_pM - d.elisa_pM
        bias, sd = difference.mean(), difference.std(ddof=1)
        rows.append({"hormone": h, "n": len(d), "ccc": lin_ccc(d.predicted_pM, d.elisa_pM),
                     "mae_pM": difference.abs().mean(), "bias_pM": bias, "loa_low_pM": bias - 1.96 * sd,
                     "loa_high_pM": bias + 1.96 * sd, "negative_predictions_set_to_0": int((d.unfloored_pM < 0).sum())})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help="folder for the two output CSVs")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = held_out_predictions(pd.read_csv(RAW / "gtt_post_injection.csv"))
    table = summary(predictions)
    predictions.to_csv(output_dir / "held_out_elisa.csv", index=False)
    table.to_csv(output_dir / "held_out_elisa_summary.csv", index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
