#!/usr/bin/env python3
"""
GTT feature definitions: class GTTFeatureExtractor
=================================================

Computes, for each rat, the features of the GTT feature table (Supplementary Information) from its
calibrated insulin, C-peptide and glucagon profiles (pM) and its glucose and ELISA values at the
blood-sampling times. extract_features.py runs this module.

Profiles       : 28 samples per rat: 4 before the glucose injection (t = -10 to -2.5 min) and 24 after it
                 (t = 0 to 61.33 min). signal_kind selects the profile:
                   - windowed    -> <hormone>_predicted_pM      (Hamming-smoothed, used for classification)
                   - nonwindowed -> <hormone>_raw_predicted_pM  (unsmoothed)
Baseline ref   : the 4 pre-injection samples
Clinical inputs: ELISA values (pM) and glucose (mM) at the 5 blood-sampling times
                 (t = 0, 16, 29.33, 45.33, 61.33 min); Rat1 has 4 (no sample at t = 45.33).

Feature categories
------------------
  A. Baseline reference statistics  (baseline mean/CV used as F0 reference)
  B. Response overview              (mean, std, CV, min, max, range)
  C. Peak detection                 (time, magnitude, fold-change, prominence, width, sharpness)
  D. Rise kinetics                  (rise time, rate, max rate)
  E. Decay kinetics                 (decay time/rate, half-recovery, return-to-baseline)
  F. Biphasic response              (first/second phase AUC, ratio, dominance; insulin & C-peptide only)
  G. AUC metrics                    (total, incremental above baseline, net, time-weighted mean)
  H. Rate of change                 (max secretion/suppression rate, mean abs rate, variability, trend)
  I. Oscillation & pulsatility      (pulse frequency, amplitude, index; early/late phase osc freq)
  J. Clinical glucose tolerance     (HOMA-IR + adapted SI-unit surrogate indices)
  K. Glucagon suppression           (% suppression, nadir time, nadir concentration)
  L. Glucose response metrics       (glucose AUC, peak, at each ELISA timepoint)
  M. Inter-biomarker correlations   (Pearson r, p-value, cross-corr lag for all 3 pairs)
  N. Glucose-hormone relationships  (anchor-point Pearson r, glucose-corrected AUC)
  O. Timing relationships           (C-peptide peak lag, glucagon-nadir/insulin-peak lag)
"""

from __future__ import annotations   # lazy annotations: run on Python 3.9+ too

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid as trapz
from scipy import stats
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d

# Default rats: the six LZ and ZDF rats (extract_features.py passes all nine)
SCORED_ANIMALS = ["Rat1", "Rat2", "Rat3", "Rat7", "Rat8", "Rat9"]

# Signal source -> prediction column suffix in the calibrated profiles
SIGNAL_COLUMNS = {
    "windowed":    "predicted_pM",       # Hamming-smoothed profile (used for classification)
    "nonwindowed": "raw_predicted_pM",   # unsmoothed profile
}

BIOMARKERS = ["insulin", "c_peptide", "glucagon"]
EXPECTED_TIMEPOINTS_PER_ANIMAL = 28
EXPECTED_BASELINE_POINTS = 4
EXPECTED_RESPONSE_POINTS = 24
EXPECTED_FEATURE_COUNT = 192   # 185 + 7 ELISA-reference clinical indices

# HOMA-IR unit conversion. The 0.167 factor is the conventional human-insulin
# approximation (1 μU/mL ≈ 6 pmol/L); its use for the rat assay is explicitly
# treated as approximate and is not justified by MW alone.
# Standard HOMA-IR = (glucose_mg/dL × insulin_μU/mL) / 405
INSULIN_PM_TO_UU_ML = 0.167
GLUCOSE_MM_TO_MGDL  = 18.0    # 1 mmol/L → mg/dL

# Prespecified COMPUTATIONAL validity floor for the ratio indices (Matsuda/DI):
# positivity alone is insufficient — an arbitrarily tiny fasting insulin still
# makes the denominator explode. Below this floor the ratio index is set to NaN.
# This is a numerical safeguard, NOT a scientifically established sub-physiological
# boundary; an assay/calibration-range (LOD) validity criterion would be more
# principled and is left as future work.
FASTING_INSULIN_FLOOR_PM = 1.0


def _safe_pearsonr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Pearson r/p with finite/constant guards and narrowly scoped warnings."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < 3 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return np.nan, np.nan
    with warnings.catch_warnings():
        for warning_name in ("ConstantInputWarning", "NearConstantInputWarning"):
            warning_cls = getattr(stats, warning_name, None)
            if warning_cls is not None:
                warnings.simplefilter("ignore", warning_cls)
        r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def _circular_shift_pvalue(x: np.ndarray, y: np.ndarray, observed_r: float) -> float:
    """
    Deterministic circular-shift p-value for two serial time courses.

    Circular shifts preserve each trajectory's autocorrelation while breaking
    their time alignment. With 24 response points, the minimum attainable
    two-sided p-value is 1/24; this is an exploratory alignment test rather than
    an IID Pearson p-value.
    """
    if not np.isfinite(observed_r):
        return np.nan
    null_abs_r = []
    for shift in range(1, len(y)):
        shifted_r, _ = _safe_pearsonr(x, np.roll(y, shift))
        if np.isfinite(shifted_r):
            null_abs_r.append(abs(shifted_r))
    if not null_abs_r:
        return np.nan
    exceed = sum(value >= abs(observed_r) - 1e-12 for value in null_abs_r)
    return float((exceed + 1) / (len(null_abs_r) + 1))


# ===========================================================================
# GTTFeatureExtractor
# ===========================================================================

class GTTFeatureExtractor:
    """
    Loads QIRT_predicted timeseries + ELISA/glucose anchor data,
    then extracts all GTT features for every animal.
    """

    def __init__(self, fmax_csv: Path, glucose_csv: Path,
                 signal_kind: str = "windowed",
                 animals: list | None = None):
        if signal_kind not in SIGNAL_COLUMNS:
            raise ValueError(
                f"signal_kind must be one of {list(SIGNAL_COLUMNS)}, got '{signal_kind}'"
            )
        self.signal_kind = signal_kind
        self._keep_animals = animals
        self._load_data(fmax_csv, glucose_csv)

    # -----------------------------------------------------------------------
    # Data loading
    # -----------------------------------------------------------------------

    def _load_data(self, fmax_path: Path, glucose_path: Path) -> None:
        """
        Read the wide-format F/Fmax predictions, select the prediction column
        for the chosen signal_kind, and merge glucose from the raw timepoints CSV.
        """
        suffix = SIGNAL_COLUMNS[self.signal_kind]   # 'predicted_pM' | 'raw_predicted_pM'

        fm = pd.read_csv(fmax_path)
        required_fm = {
            "animal_id", "rat_type", "timepoint",
            "insulin_ELISA_pM", "c_peptide_ELISA_pM", "glucagon_ELISA_pM",
            *[f"{bm}_{SIGNAL_COLUMNS[self.signal_kind]}" for bm in BIOMARKERS],
        }
        missing_fm = sorted(required_fm - set(fm.columns))
        if missing_fm:
            raise ValueError(f"F/Fmax input is missing required columns: {missing_fm}")
        fm["timepoint"] = fm["timepoint"].round(4)

        # Restrict to the requested rats
        keep = list(self._keep_animals) if self._keep_animals is not None else list(SCORED_ANIMALS)
        if len(set(keep)) != len(keep):
            raise ValueError(f"Requested animal list contains duplicates: {keep}")
        fm = fm[fm["animal_id"].isin(keep)].copy()

        expected_animals = set(keep)
        observed_animals = set(fm["animal_id"].unique())
        if observed_animals != expected_animals:
            missing = sorted(expected_animals - observed_animals)
            unexpected = sorted(observed_animals - expected_animals)
            raise ValueError(
                "F/Fmax cohort mismatch: "
                f"missing={missing}, unexpected={unexpected}, expected={sorted(expected_animals)}"
            )
        if fm.duplicated(["animal_id", "timepoint"]).any():
            duplicates = fm.loc[
                fm.duplicated(["animal_id", "timepoint"], keep=False),
                ["animal_id", "timepoint"],
            ].to_dict("records")
            raise ValueError(f"Duplicate rounded F/Fmax keys: {duplicates}")

        counts = fm.groupby("animal_id").size().to_dict()
        bad_counts = {a: counts.get(a, 0) for a in keep if counts.get(a, 0) != EXPECTED_TIMEPOINTS_PER_ANIMAL}
        if bad_counts:
            raise ValueError(
                f"Expected {EXPECTED_TIMEPOINTS_PER_ANIMAL} timepoints per animal; got {bad_counts}"
            )
        baseline_counts = fm[fm["timepoint"] < 0].groupby("animal_id").size().to_dict()
        response_counts = fm[fm["timepoint"] >= 0].groupby("animal_id").size().to_dict()
        bad_phase_counts = {
            a: {"baseline": baseline_counts.get(a, 0), "response": response_counts.get(a, 0)}
            for a in keep
            if baseline_counts.get(a, 0) != EXPECTED_BASELINE_POINTS
            or response_counts.get(a, 0) != EXPECTED_RESPONSE_POINTS
        }
        if bad_phase_counts:
            raise ValueError(
                f"Expected {EXPECTED_BASELINE_POINTS} baseline and "
                f"{EXPECTED_RESPONSE_POINTS} response points per animal; got {bad_phase_counts}"
            )
        rat_type_counts = fm.groupby("animal_id")["rat_type"].nunique()
        if (rat_type_counts != 1).any():
            raise ValueError(
                "Each animal must have exactly one rat_type; bad animals: "
                f"{rat_type_counts[rat_type_counts != 1].to_dict()}"
            )

        # Select the prediction column per biomarker and rename to *_pM
        wide_cols = ["animal_id", "rat_type", "timepoint"]
        wide = fm[wide_cols].copy()
        for bm in BIOMARKERS:
            wide[f"{bm}_pM"] = fm[f"{bm}_{suffix}"].astype(float).values
        if not np.isfinite(wide[[f"{bm}_pM" for bm in BIOMARKERS]].to_numpy(float)).all():
            raise ValueError("Selected predicted-pM columns contain NaN or infinite values")

        # ELISA anchors: rows where ELISA is populated (from the fmax file itself,
        # consistent with the calibration that was actually fit)
        elisa_mask = fm["insulin_ELISA_pM"].notna()
        elisa_rows = fm.loc[elisa_mask, [
            "animal_id", "rat_type", "timepoint",
            "insulin_ELISA_pM", "c_peptide_ELISA_pM", "glucagon_ELISA_pM",
        ]].copy()
        elisa_value_cols = [
            "insulin_ELISA_pM", "c_peptide_ELISA_pM", "glucagon_ELISA_pM"
        ]
        if elisa_rows[elisa_value_cols].isna().any().any():
            raise ValueError("An ELISA anchor row is missing one or more hormone values")

        # Merge glucose_mM from the raw timepoints CSV (only source of glucose)
        glu = pd.read_csv(glucose_path)
        required_glucose = {"animal_id", "timepoint", "glucose_mM"}
        missing_glucose = sorted(required_glucose - set(glu.columns))
        if missing_glucose:
            raise ValueError(f"Glucose input is missing required columns: {missing_glucose}")
        glu["timepoint"] = glu["timepoint"].round(4)
        glu = glu[glu["animal_id"].isin(keep)][["animal_id", "timepoint", "glucose_mM"]]
        if glu.duplicated(["animal_id", "timepoint"]).any():
            duplicates = glu.loc[
                glu.duplicated(["animal_id", "timepoint"], keep=False),
                ["animal_id", "timepoint"],
            ].to_dict("records")
            raise ValueError(f"Duplicate rounded glucose keys: {duplicates}")
        elisa_rows = elisa_rows.merge(
            glu,
            on=["animal_id", "timepoint"],
            how="left",
            validate="one_to_one",
        )
        missing_anchor_glucose = elisa_rows.loc[
            elisa_rows["glucose_mM"].isna(), ["animal_id", "timepoint"]
        ].to_dict("records")
        if missing_anchor_glucose:
            raise ValueError(f"ELISA anchors missing matched glucose values: {missing_anchor_glucose}")

        # Rat1 has four anchors; the other rats
        # have five. Validate rather than silently dropping anchors.
        expected_anchor_counts = {a: (4 if a == "Rat1" else 5) for a in keep}
        observed_anchor_counts = elisa_rows.groupby("animal_id").size().to_dict()
        bad_anchor_counts = {
            a: observed_anchor_counts.get(a, 0)
            for a, expected in expected_anchor_counts.items()
            if observed_anchor_counts.get(a, 0) != expected
        }
        if bad_anchor_counts:
            raise ValueError(
                f"Unexpected ELISA/glucose anchor counts: {bad_anchor_counts}; "
                f"expected {expected_anchor_counts}"
            )

        # Validate the exact anchor TIMES, not just counts: every animal must have
        # the fasting t=0 anchor (assumed downstream as I0/G0), and Rat1's single
        # gap must specifically be t=45.33 (not some other anchor silently missing).
        expected_anchor_times = {0.0, 16.0, 29.33, 45.33, 61.33}
        anchor_times = {
            a: set(elisa_rows.loc[elisa_rows["animal_id"] == a, "timepoint"].round(2))
            for a in keep
        }
        for a, ts in anchor_times.items():
            if 0.0 not in ts:
                raise ValueError(f"{a}: fasting anchor t=0 missing; anchors={sorted(ts)}")
            extra = ts - expected_anchor_times
            if extra:
                raise ValueError(f"{a}: unexpected anchor timepoints {sorted(extra)}")
        if "Rat1" in keep and (expected_anchor_times - anchor_times["Rat1"]) != {45.33}:
            raise ValueError(
                f"Rat1 expected to be missing only t=45.33; missing "
                f"{sorted(expected_anchor_times - anchor_times['Rat1'])}"
            )

        self._wide      = wide
        self._elisa_src = elisa_rows
        self._expected_animals = list(keep)
        self.animals    = sorted(wide["animal_id"].unique())

        n_anchor = elisa_rows.groupby("animal_id").size().to_dict()
        print(f"[{self.signal_kind}] Loaded {len(self.animals)} animals: {self.animals}")
        print(f"  Signal column: *_{suffix}")
        print(f"  Timepoints per animal: {EXPECTED_TIMEPOINTS_PER_ANIMAL}")
        print(f"  Anchor (glucose) timepoints per animal: {n_anchor}")

    def _get_animal_data(self, animal_id: str) -> dict:
        df      = self._wide[self._wide["animal_id"] == animal_id].sort_values("timepoint")
        rat_type = df["rat_type"].iloc[0]
        baseline = df[df["timepoint"] < 0.0].copy()
        response = df[df["timepoint"] >= 0.0].copy()
        elisa    = self._elisa_src[self._elisa_src["animal_id"] == animal_id].sort_values("timepoint")
        return {
            "animal_id": animal_id,
            "rat_type":  rat_type,
            "baseline":  baseline,
            "response":  response,
            "elisa":     elisa,
        }

    # -----------------------------------------------------------------------
    # A. Baseline reference
    # -----------------------------------------------------------------------

    def _baseline_stats(self, values: np.ndarray) -> dict:
        mean = float(np.mean(values))
        std  = float(np.std(values))
        return {
            "mean_pM": mean,
            "std_pM":  std,
            "cv_pct":  float(std / mean * 100) if mean > 0 else 0.0,
            "min_pM":  float(np.min(values)),
            "max_pM":  float(np.max(values)),
        }

    # -----------------------------------------------------------------------
    # B. Response overview
    # -----------------------------------------------------------------------

    def _response_stats(self, values: np.ndarray) -> dict:
        mean = float(np.mean(values))
        std  = float(np.std(values))
        return {
            "mean_pM":  mean,
            "std_pM":   std,
            "cv_pct":   float(std / mean * 100) if mean > 0 else 0.0,
            "min_pM":   float(np.min(values)),
            "max_pM":   float(np.max(values)),
            "range_pM": float(np.max(values) - np.min(values)),
        }

    # -----------------------------------------------------------------------
    # C. Peak detection
    # -----------------------------------------------------------------------

    def _detect_peak(
        self, times: np.ndarray, values: np.ndarray, baseline_mean: float
    ) -> dict:
        """
        Peak = the global maximum of the response window (the physiologically
        meaningful "highest concentration reached"), NOT the most-prominent
        interior peak.  scipy.find_peaks ignores endpoints and can select a
        small interior bump over the true maximum, so it is not used for the
        magnitude/time.  We add explicit flags:

          found          : True iff the global max is a strict interior local
                           maximum (so rise AND decay limbs both exist).
          right_censored : True iff the max is the last sample (still rising at
                           the end of the GTT window — no decay observed).

        Prominence/width are only well defined for an interior maximum.
        """
        n = len(values)
        null = {
            "found": False, "time_min": np.nan, "_idx": -1,
            "magnitude_pM": np.nan, "prominence": 0.0, "width_min": 0.0,
            "fold_change": np.nan, "absolute_change_pM": np.nan, "sharpness": 0.0,
            "right_censored": False,
        }
        if n == 0:
            return null

        idx = int(np.argmax(values))          # global maximum
        mag = float(values[idx])
        dt  = float(np.mean(np.diff(times))) if n > 1 else 1.0
        interior = 0 < idx < n - 1
        strict_interior = bool(
            interior and mag > float(values[idx - 1]) and mag > float(values[idx + 1])
        )
        right_censored = bool(np.isclose(values[-1], mag, rtol=1e-12, atol=1e-12))

        if strict_interior:
            # Standard prominence: height above the higher of the two base minima
            left_min  = float(np.min(values[:idx]))
            right_min = float(np.min(values[idx + 1:]))
            prom = float(mag - max(left_min, right_min))
            try:
                widths, *_ = scipy_signal.peak_widths(values, [idx], rel_height=0.5)
                width_min = float(widths[0] * dt)
            except Exception:
                width_min = 0.0
        else:
            prom = 0.0
            width_min = 0.0

        return {
            "found": strict_interior,
            "time_min": float(times[idx]),
            "_idx": idx,
            "magnitude_pM": mag,
            "prominence": prom,
            "width_min": width_min,
            "fold_change": float(mag / (baseline_mean + 1e-10)),
            "absolute_change_pM": float(mag - baseline_mean),
            "sharpness": float(prom / (width_min + 1e-10)),
            "right_censored": right_censored,
        }

    # -----------------------------------------------------------------------
    # D. Rise kinetics  (t=0 → peak)
    # -----------------------------------------------------------------------

    def _rise_kinetics(
        self, times: np.ndarray, values: np.ndarray, peak_idx: int
    ) -> dict:
        if peak_idx <= 0:
            return {
                "rise_time_min": 0.0,
                "rise_rate_pM_per_min": 0.0,
                "max_rise_rate_pM_per_min": 0.0,
            }
        rt, rv = times[: peak_idx + 1], values[: peak_idx + 1]
        rise_time = float(rt[-1] - rt[0])
        rise_rate = float((rv[-1] - rv[0]) / (rise_time + 1e-10))
        inst = np.diff(rv) / (np.diff(rt) + 1e-10)
        return {
            "rise_time_min":            rise_time,
            "rise_rate_pM_per_min":     rise_rate,
            "max_rise_rate_pM_per_min": float(np.max(inst)) if len(inst) else 0.0,
        }

    # -----------------------------------------------------------------------
    # E. Decay kinetics  (peak → end)
    # -----------------------------------------------------------------------

    def _decay_kinetics(
        self,
        times: np.ndarray,
        values: np.ndarray,
        peak_idx: int,
        baseline_mean: float,
        right_censored: bool = False,
    ) -> dict:
        null = {
            "decay_time_min": np.nan,
            "decay_rate_pM_per_min": 0.0,
            "time_to_half_recovery_min": np.nan,
            "return_to_baseline": False,
        }
        # No decay limb if the peak is the last sample (right-censored) or absent.
        if peak_idx < 0 or peak_idx >= len(values) - 1 or right_censored:
            return null

        dt, dv     = times[peak_idx:], values[peak_idx:]
        peak_val   = float(values[peak_idx])
        decay_time = float(dt[-1] - dt[0])
        decay_rate = float(abs((dv[-1] - dv[0]) / (decay_time + 1e-10)))

        # Time-to-half-recovery: time from peak until the trace first falls to the
        # midpoint between peak and fasting baseline. Only defined when the peak
        # actually rises above baseline (otherwise there is nothing to recover
        # from — e.g. a suppressed glucagon channel). The crossing is linearly
        # interpolated rather than snapped to the 2.667-min grid.
        t_half = np.nan
        if peak_val > baseline_mean:
            half_level = (peak_val + baseline_mean) / 2.0
            for i in range(1, len(dv)):
                if dv[i] <= half_level:
                    y0, y1 = dv[i - 1], dv[i]
                    t0, t1 = dt[i - 1], dt[i]
                    t_cross = t1 if y0 == y1 else t0 + (half_level - y0) * (t1 - t0) / (y1 - y0)
                    t_half = float(t_cross - dt[0])
                    break

        return {
            "decay_time_min":            decay_time,
            "decay_rate_pM_per_min":     decay_rate,
            "time_to_half_recovery_min": t_half,
            "return_to_baseline":        bool(
                abs(float(dv[-1]) - baseline_mean) <= 0.1 * abs(baseline_mean)
            ),
        }

    # -----------------------------------------------------------------------
    # F. Biphasic response  (insulin & C-peptide only)
    # -----------------------------------------------------------------------

    def _biphasic(
        self, times: np.ndarray, values: np.ndarray, first_phase_end: float
    ) -> dict:
        """
        Split the response into first/second phase at `first_phase_end`.
        The boundary value is interpolated and inserted into BOTH phase
        integrals so the trapezoid straddling the split is not dropped
        (otherwise first-phase AUC is materially underestimated).
        """
        split = float(first_phase_end)
        # Interpolated signal value at the split boundary.
        v_split = float(np.interp(split, times, values))

        # Phase 1: points strictly before split, then the boundary point.
        t1 = np.append(times[times < split], split)
        v1 = np.append(values[times < split], v_split)
        # Phase 2: the boundary point, then points strictly after split.
        t2 = np.insert(times[times > split], 0, split)
        v2 = np.insert(values[times > split], 0, v_split)

        fp_auc = float(trapz(v1, t1)) if len(v1) > 1 else 0.0
        sp_auc = float(trapz(v2, t2)) if len(v2) > 1 else 0.0
        total  = fp_auc + sp_auc

        return {
            "first_phase_auc_pM_min":    fp_auc,
            "second_phase_auc_pM_min":   sp_auc,
            "first_second_ratio":        float(fp_auc / (sp_auc + 1e-10)),
            "first_phase_dominance":     float(fp_auc / (total + 1e-10)),
            "first_phase_peak_pM":       float(np.max(v1)) if len(v1) > 0 else 0.0,
            "second_phase_mean_pM":      float(np.mean(v2)) if len(v2) > 0 else 0.0,
        }

    # -----------------------------------------------------------------------
    # G. AUC metrics
    # -----------------------------------------------------------------------

    def _auc_metrics(
        self, times: np.ndarray, values: np.ndarray, baseline_mean: float
    ) -> dict:
        auc_total = float(trapz(values, times))
        duration  = float(times[-1] - times[0])
        incremental = values - baseline_mean
        auc_inc  = float(trapz(np.maximum(incremental, 0.0), times))
        auc_net  = float(trapz(incremental, times))
        return {
            "auc_total_pM_min":        auc_total,
            "incremental_auc_pM_min":  auc_inc,   # area above baseline only
            "net_auc_pM_min":          auc_net,    # signed (can be negative)
            "time_weighted_mean_pM":   float(auc_total / (duration + 1e-10)),
        }

    # -----------------------------------------------------------------------
    # H. Rate of change & trend
    # -----------------------------------------------------------------------

    def _rate_metrics(self, times: np.ndarray, values: np.ndarray) -> dict:
        rates = np.diff(values) / (np.diff(times) + 1e-10)
        slope, _, r_val, p_val, _ = stats.linregress(times, values)
        return {
            "max_secretion_rate_pM_per_min":    float(np.max(rates)),
            "max_suppression_rate_pM_per_min":  float(abs(np.min(rates))),
            "mean_absolute_rate_pM_per_min":    float(np.mean(np.abs(rates))),
            "rate_variability_sd":              float(np.std(rates)),
            "trend_slope_pM_per_min":           float(slope),
            "trend_r_squared":                  float(r_val ** 2),
            "trend_p_value":                    float(p_val),
        }

    # -----------------------------------------------------------------------
    # I. Oscillation & pulsatility
    # -----------------------------------------------------------------------

    def _oscillation_metrics(self, times: np.ndarray, values: np.ndarray) -> dict:
        duration_hr = float(times[-1] - times[0]) / 60.0
        detrended   = values - np.mean(values)
        sig_std     = float(np.std(values))
        min_prom    = 0.05 * sig_std  # 5% of SD as minimum pulse prominence

        peaks,  _ = scipy_signal.find_peaks( detrended, prominence=min_prom)
        troughs,_ = scipy_signal.find_peaks(-detrended, prominence=min_prom)

        pulse_freq = float(len(peaks) / (duration_hr + 1e-10))
        if len(peaks) > 0 and len(troughs) > 0:
            pulse_amp = float(np.mean(detrended[peaks]) - np.mean(detrended[troughs]))
        else:
            pulse_amp = 0.0

        mean_abs        = float(np.mean(np.abs(values)))
        pulsatility_idx = float(pulse_amp / (mean_abs + 1e-10))

        # Pulse regularity (CV of inter-pulse intervals)
        if len(peaks) > 1:
            intervals        = np.diff(peaks).astype(float)
            pulse_reg_cv_pct = float(np.std(intervals) / (np.mean(intervals) + 1e-10) * 100)
        else:
            pulse_reg_cv_pct = 0.0

        # Phase-specific oscillation frequency (peaks per hour)
        def _phase_osc_freq(t_sub: np.ndarray, v_sub: np.ndarray) -> float:
            if len(v_sub) < 4:
                return 0.0
            d    = v_sub - np.mean(v_sub)
            pks, _ = scipy_signal.find_peaks(d, height=0)
            dur  = float(t_sub[-1] - t_sub[0]) / 60.0
            return float(len(pks) / (dur + 1e-10))

        early_mask = times <= 15.0
        late_mask  = times >  15.0
        early_freq = _phase_osc_freq(times[early_mask], values[early_mask])
        late_freq  = _phase_osc_freq(times[late_mask],  values[late_mask])

        return {
            "pulse_frequency_per_hour":        pulse_freq,
            "pulse_amplitude_pM":              pulse_amp,
            "pulse_regularity_cv_pct":         pulse_reg_cv_pct,
            "pulsatility_index":               pulsatility_idx,
            "osc_freq_early_0_15min_per_hr":   early_freq,
            "osc_freq_late_15_61min_per_hr":   late_freq,
        }

    # -----------------------------------------------------------------------
    # J. Clinical glucose tolerance indices
    # -----------------------------------------------------------------------

    def _glucose_response(self, times: np.ndarray, g: dict, g_vals: list) -> dict:
        """L. Glucose response metrics (glucometer only — source-independent)."""
        out: dict = {}
        if len(times) > 1:
            out["glucose_auc_mM_min"] = float(trapz(g_vals, times))
            out["glucose_incremental_auc_mM_min"] = float(
                trapz([max(v - g_vals[0], 0.0) for v in g_vals], times)
            )
        out["glucose_peak_mM"] = float(max(g_vals))
        for t in times:
            out[f"glucose_at_{int(round(t))}min_mM"] = float(g[t])
        return out

    def _tolerance_indices(
        self, times: np.ndarray, g: dict, ins: dict, suffix: str
    ) -> dict:
        """
        HOMA-IR / Matsuda / insulinogenic / disposition from a given INSULIN
        source paired with glucometer glucose.

          suffix=""        device-derived (QIRT predicted insulin) — PRIMARY
          suffix="_elisa"  conventional-ELISA insulin — validation REFERENCE

        Glucose always comes from the glucometer (the device does not measure
        glucose). HOMA-IR is in conventional units; Matsuda/IGI/DI are in raw SI
        units (suffix '_si') and are relative — see class notes.
        """
        out: dict = {}
        t0   = times[0]
        g0   = g.get(t0, np.nan)
        ins0 = ins.get(t0, np.nan)
        g_vals   = [g.get(t, np.nan)   for t in times]
        ins_vals = [ins.get(t, np.nan) for t in times]

        # HOMA-IR (conventional units)
        if not (np.isnan(g0) or np.isnan(ins0)):
            ins0_uU = ins0 * INSULIN_PM_TO_UU_ML
            g0_mgdl = g0 * GLUCOSE_MM_TO_MGDL
            out[f"homa_ir{suffix}"] = float((g0_mgdl * ins0_uU) / 405.0)

        # Matsuda (raw SI; relative). Require positive fasting insulin/glucose so
        # the index stays defined (NaN, not a huge number) when fasting insulin is
        # ~0 — the device fasting insulin uses the fasting-window mean to avoid this.
        mean_g   = float(np.nanmean(g_vals))
        mean_ins = float(np.nanmean(ins_vals))
        if ins0 >= FASTING_INSULIN_FLOOR_PM and g0 > 0 and mean_g > 0 and mean_ins > 0:
            denom = np.sqrt((g0 * ins0) * (mean_g * mean_ins))
            out[f"matsuda_relative_si{suffix}"] = float(10000.0 / denom)

        # Insulinogenic index at each post-injection anchor (pM/mM)
        for t in times[1:]:
            delta_ins = ins.get(t, np.nan) - ins0
            delta_g   = g.get(t, np.nan) - g0
            if not (np.isnan(delta_ins) or np.isnan(delta_g)):
                out[f"insulinogenic_index_{int(round(t))}min_si{suffix}"] = (
                    float(delta_ins / delta_g) if abs(delta_g) > 0.001 else np.nan
                )

        # Disposition = IGI(~30 min) × Matsuda
        mk = f"matsuda_relative_si{suffix}"
        if mk in out:
            closest_30 = min(times[1:], key=lambda t: abs(t - 30.0))
            ik = f"insulinogenic_index_{int(round(closest_30))}min_si{suffix}"
            if ik in out and not np.isnan(out[ik]):
                out[f"disposition_relative_si{suffix}"] = float(out[ik] * out[mk])
        return out

    def _clinical_indices(self, elisa_df: pd.DataFrame, device_ins: dict) -> dict:
        """
        Glucose-tolerance indices at the 4-5 anchor timepoints, computed TWO ways:

          PRIMARY   — device-derived: QIRT predicted insulin (`device_ins`) +
                      glucometer glucose. Clean names (homa_ir, matsuda_relative_si,
                      insulinogenic_index_*min_si, disposition_relative_si).
          REFERENCE — ELISA-derived: conventional ELISA insulin + glucometer
                      glucose. Same names with a '_elisa' suffix (validation only).

        Glucose response metrics (L) are glucometer-only and reported once.
        The device does not measure glucose, so every glucose-dependent index
        always uses the external glucometer value.
        """
        idx: dict = {}
        if elisa_df.empty:
            return idx

        elisa = elisa_df.sort_values("timepoint").reset_index(drop=True)
        times = elisa["timepoint"].values.astype(float)

        g         = dict(zip(times, elisa["glucose_mM"].values.astype(float)))
        ins_elisa = dict(zip(times, elisa["insulin_ELISA_pM"].values.astype(float)))
        g_vals    = [g[t] for t in times]

        # L. glucose response (glucometer only, computed once)
        idx.update(self._glucose_response(times, g, g_vals))
        # J PRIMARY: device-derived
        idx.update(self._tolerance_indices(times, g, device_ins, ""))
        # J REFERENCE: ELISA-derived
        idx.update(self._tolerance_indices(times, g, ins_elisa, "_elisa"))
        return idx

    # -----------------------------------------------------------------------
    # K. Glucagon suppression (referenced to fasting baseline)
    # -----------------------------------------------------------------------

    def _glucagon_suppression(
        self,
        resp_times: np.ndarray,
        glucagon_resp: np.ndarray,
        baseline_mean: float,
    ) -> dict:
        """
        Glucagon suppression measured from the predicted response curve,
        referenced to the fasting baseline mean (not t=0): by the time of
        injection glucagon is already partly suppressed, so the pre-injection
        fasting mean is the physiological reference.

          suppression_pct = (baseline_mean - nadir) / baseline_mean * 100
          rebound_ratio   = end value / nadir   (alpha-cell escape in late GTT)
        """
        if len(glucagon_resp) == 0 or not np.isfinite(baseline_mean):
            return {}
        nadir_idx = int(np.argmin(glucagon_resp))
        nadir_val = float(glucagon_resp[nadir_idx])
        nadir_t   = float(resp_times[nadir_idx])
        ref       = float(baseline_mean)
        supp_pct  = (ref - nadir_val) / (ref + 1e-10) * 100.0
        return {
            "glucagon_suppression_pct":   float(supp_pct),
            "glucagon_nadir_pM":          nadir_val,
            "glucagon_nadir_time_min":    nadir_t,
            "glucagon_fasting_ref_pM":    ref,
            "glucagon_rebound_ratio":     float(glucagon_resp[-1] / (nadir_val + 1e-10)),
        }

    # -----------------------------------------------------------------------
    # M. Inter-biomarker correlations
    # -----------------------------------------------------------------------

    def _correlation_metrics(
        self, times: np.ndarray, bm_values: dict
    ) -> dict:
        out  = {}
        dt   = float(np.mean(np.diff(times)))
        pairs = [
            ("insulin",   "c_peptide"),
            ("insulin",   "glucagon"),
            ("c_peptide", "glucagon"),
        ]
        n        = len(times)
        max_lag  = max(1, n // 3)   # restrict lag search; edge lags use too few
                                    # overlapping points to be meaningful
        for bm1, bm2 in pairs:
            v1, v2 = bm_values[bm1], bm_values[bm2]
            r, _iid_p = _safe_pearsonr(v1, v2)
            # Timepoints are serially autocorrelated, so the ordinary IID Pearson
            # p-value is inappropriate. Use a deterministic circular-shift null
            # that preserves each trace's autocorrelation.
            p = _circular_shift_pvalue(v1, v2, r)

            # Compare like with like at every lag: Pearson-normalize each
            # overlapping segment separately, rather than maximizing an
            # unnormalized cross-product whose scale changes with overlap length.
            lag_samples = np.arange(-max_lag, max_lag + 1, dtype=int)
            lag_scores = []
            for lag in lag_samples:
                if lag < 0:
                    x_lag, y_lag = v1[: n + lag], v2[-lag:]
                elif lag > 0:
                    x_lag, y_lag = v1[lag:], v2[: n - lag]
                else:
                    x_lag, y_lag = v1, v2
                lag_r, _ = _safe_pearsonr(x_lag, y_lag)
                lag_scores.append(lag_r)
            lag_scores = np.asarray(lag_scores, dtype=float)
            finite_lags = np.isfinite(lag_scores)
            if finite_lags.any():
                best_abs = float(np.max(np.abs(lag_scores[finite_lags])))
                candidates = np.flatnonzero(
                    finite_lags & np.isclose(np.abs(lag_scores), best_abs, rtol=1e-12, atol=1e-12)
                )
                # Deterministic tie-break: prefer the smallest absolute lag.
                best_i = min(candidates, key=lambda i: (abs(lag_samples[i]), lag_samples[i]))
                opt_lag = float(lag_samples[best_i] * dt)
            else:
                opt_lag = np.nan
            key   = f"corr_{bm1}_{bm2}"
            out[f"{key}_pearson_r"] = float(r)
            out[f"{key}_pvalue"]    = float(p)
            out[f"{key}_lag_min"]   = opt_lag
        return out

    # -----------------------------------------------------------------------
    # N. Glucose-hormone relationships
    # -----------------------------------------------------------------------

    def _glucose_hormone_metrics(
        self,
        resp_times:  np.ndarray,
        bm_values:   dict,
        elisa_df:    pd.DataFrame,
    ) -> dict:
        if elisa_df.empty:
            return {}
        out = {}
        gdata  = elisa_df[["timepoint", "glucose_mM"]].sort_values("timepoint")
        gt, gv = gdata["timepoint"].values.astype(float), gdata["glucose_mM"].values.astype(float)
        if len(gt) < 2:
            return {}

        # Pearson r/p are computed at the REAL glucose anchor timepoints (n = 4-5),
        # pairing each glucose draw with the predicted hormone sample at the same
        # time. This avoids the invalid inflated-n that arises from correlating the
        # 24-pt hormone curve against glucose interpolated to 24 serially-correlated
        # points. r/p here are exploratory (small n); interpret with caution.
        anchor_idx = [int(np.argmin(np.abs(resp_times - t))) for t in gt]

        # Interpolated glucose is used ONLY for the exposure-ratio AUC (an integral,
        # where dense sampling is appropriate), not for the correlation.
        f_g      = interp1d(gt, gv, kind="linear", fill_value="extrapolate")
        g_interp = f_g(resp_times)

        for bm, vals in bm_values.items():
            h_anchor = np.array([vals[i] for i in anchor_idx], dtype=float)
            r, p = _safe_pearsonr(h_anchor, gv)
            gc_auc = float(trapz(vals / (g_interp + 1e-10), resp_times))
            out[f"{bm}_glucose_corr_r"]        = float(r)
            out[f"{bm}_glucose_corr_pvalue"]   = float(p)
            out[f"{bm}_glucose_corrected_auc"] = gc_auc
        return out

    # -----------------------------------------------------------------------
    # Main extraction
    # -----------------------------------------------------------------------

    def extract_features(self, animal_id: str) -> dict:
        """Extract all features for one animal. Returns a flat dict (one CSV row)."""
        d    = self._get_animal_data(animal_id)
        resp = d["response"]
        base = d["baseline"]
        elisa = d["elisa"]

        row: dict = {
            "animal_id": animal_id,
            "rat_type":  d["rat_type"],
        }

        resp_times = resp["timepoint"].values.astype(float)
        bm_values  = {
            "insulin":   resp["insulin_pM"].values.astype(float),
            "c_peptide": resp["c_peptide_pM"].values.astype(float),
            "glucagon":  resp["glucagon_pM"].values.astype(float),
        }

        # A. Baseline reference statistics + compute baseline mean for each biomarker
        baseline_means: dict = {}
        if not base.empty:
            for bm in BIOMARKERS:
                bvals = base[f"{bm}_pM"].values.astype(float)
                bstats = self._baseline_stats(bvals)
                for k, v in bstats.items():
                    row[f"baseline_{bm}_{k}"] = v
                baseline_means[bm] = bstats["mean_pM"]
        else:
            # Fallback: use mean of first 2 response points
            for bm in BIOMARKERS:
                baseline_means[bm] = float(np.mean(bm_values[bm][:2]))

        # Per-biomarker response features
        peak_times: dict = {}
        peak_idxs:  dict = {}

        for bm in BIOMARKERS:
            vals     = bm_values[bm]
            bm_mean  = baseline_means[bm]
            pfx      = f"{bm}_"

            # B. Response overview
            for k, v in self._response_stats(vals).items():
                row[f"{pfx}response_{k}"] = v

            # C. Peak detection
            peak = self._detect_peak(resp_times, vals, bm_mean)
            peak_times[bm] = peak["time_min"]
            peak_idxs[bm]  = peak["_idx"]
            for k, v in peak.items():
                if k == "_idx":
                    continue
                row[f"{pfx}peak_{k}"] = int(v) if isinstance(v, bool) else v

            # D. Rise kinetics
            for k, v in self._rise_kinetics(resp_times, vals, peak["_idx"]).items():
                row[f"{pfx}{k}"] = v

            # E. Decay kinetics
            for k, v in self._decay_kinetics(
                resp_times,
                vals,
                peak["_idx"],
                bm_mean,
                right_censored=peak["right_censored"],
            ).items():
                row[f"{pfx}{k}"] = v

            # G. AUC metrics
            for k, v in self._auc_metrics(resp_times, vals, bm_mean).items():
                row[f"{pfx}{k}"] = v

            # H. Rate of change & trend
            for k, v in self._rate_metrics(resp_times, vals).items():
                row[f"{pfx}{k}"] = v

            # I. Oscillation & pulsatility
            for k, v in self._oscillation_metrics(resp_times, vals).items():
                row[f"{pfx}{k}"] = v

        # F. Biphasic analysis (insulin & C-peptide only)
        # Fixed 10-min first-phase boundary (standard physiology, comparable
        # across animals — an adaptive boundary would shift the phase split
        # between animals once peak times differ).
        first_phase_end = 10.0
        row["first_phase_end_min"] = first_phase_end

        for bm in ["insulin", "c_peptide"]:
            bip = self._biphasic(resp_times, bm_values[bm], first_phase_end)
            for k, v in bip.items():
                row[f"{bm}_{k}"] = v

        # O. Timing relationships
        # C-peptide vs insulin: peak-to-peak (co-secretion → near-zero expected).
        if not np.isnan(peak_times.get("insulin",   np.nan)) and \
           not np.isnan(peak_times.get("c_peptide", np.nan)):
            row["cpeptide_insulin_peak_lag_min"] = float(
                peak_times["c_peptide"] - peak_times["insulin"]
            )
        # Glucagon suppression latency: glucagon NADIR time relative to the
        # insulin peak (glucagon is suppressed, so its nadir — not its peak — is
        # the physiologically meaningful event).
        glucagon_nadir_t = float(resp_times[int(np.argmin(bm_values["glucagon"]))])
        if not np.isnan(peak_times.get("insulin", np.nan)):
            row["glucagon_nadir_insulin_peak_lag_min"] = float(
                glucagon_nadir_t - peak_times["insulin"]
            )

        # J + L. Clinical indices (device-derived primary + ELISA reference) + glucose
        # Device-predicted insulin at the anchor timepoints (matched to the glucose
        # anchors) is the PRIMARY insulin source for the indices.
        resp_ins_by_t = dict(zip(resp["timepoint"].round(4).values,
                                 resp["insulin_pM"].astype(float).values))
        device_ins = {float(t): resp_ins_by_t.get(round(float(t), 4), np.nan)
                      for t in elisa["timepoint"].values}
        # Use the fasting-WINDOW mean (pre-injection baseline) as the device's
        # fasting insulin instead of the single t=0 sample. This is the device's
        # natural fasting estimate and is robust to single-sample dips (e.g. a
        # non-windowed value clamped to 0 at t=0) that would otherwise make the
        # Matsuda / disposition denominators blow up. ELISA keeps its t=0 draw.
        fasting_t0 = min(device_ins.keys())
        device_ins[fasting_t0] = baseline_means.get("insulin", device_ins[fasting_t0])
        for k, v in self._clinical_indices(elisa, device_ins).items():
            row[k] = v

        # K. Glucagon suppression (predicted response curve vs fasting baseline)
        for k, v in self._glucagon_suppression(
            resp_times, bm_values["glucagon"], baseline_means.get("glucagon", np.nan)
        ).items():
            row[k] = v

        # M. Inter-biomarker correlations
        for k, v in self._correlation_metrics(resp_times, bm_values).items():
            row[k] = v

        # N. Glucose-hormone relationships
        for k, v in self._glucose_hormone_metrics(resp_times, bm_values, elisa).items():
            row[k] = v

        return row

    def extract_all(self) -> pd.DataFrame:
        """
        Extract features for every animal. Per-animal failures are collected and
        re-raised AFTER the loop, so a partial matrix is never silently written.
        """
        rows, errors = [], []
        for animal_id in self.animals:
            print(f"  Processing {animal_id} ...")
            try:
                rows.append(self.extract_features(animal_id))
            except Exception as exc:
                errors.append((animal_id, repr(exc)))
                print(f"  ERROR for {animal_id}: {exc}")
        if errors:
            raise RuntimeError(
                "Feature extraction failed for "
                f"{len(errors)} animal(s): {errors}"
            )
        return pd.DataFrame(rows)

    def run(self, output_path: Path) -> pd.DataFrame:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = self.extract_all()

        # Schema guard: never write an incomplete matrix that looks successful.
        expected_animals = set(self._expected_animals)
        observed_animals = set(df["animal_id"].unique()) if "animal_id" in df else set()
        if observed_animals != expected_animals or df["animal_id"].duplicated().any():
            raise RuntimeError(
                "Extracted cohort mismatch before write: "
                f"expected={sorted(expected_animals)}, observed={sorted(observed_animals)}, "
                f"duplicate_ids={df.loc[df['animal_id'].duplicated(), 'animal_id'].tolist()}"
            )

        # Column ordering: identifiers first, then alphabetical
        basic = ["animal_id", "rat_type"]
        other = sorted(c for c in df.columns if c not in basic)
        df = df[basic + other]
        if len(other) != EXPECTED_FEATURE_COUNT:
            raise RuntimeError(
                f"Expected {EXPECTED_FEATURE_COUNT} feature columns, got {len(other)}"
            )

        df.to_csv(output_path, index=False)

        n_feat = df.shape[1] - 2   # minus animal_id, rat_type
        print(f"\nSaved → {output_path}")
        print(f"  {df.shape[0]} animals  ×  {n_feat} features ({df.shape[1]} columns incl. 2 ids)")

        # Category summary. J = 7 device + 7 ELISA reference.
        categories = {
            "Baseline ref (A)": 15,
            "Response stats (B)": 18,
            "Peak detection (C)": 27,
            "Rise kinetics (D)": 9,
            "Decay kinetics (E)": 12,
            "Biphasic (F)": 13,
            "AUC metrics (G)": 12,
            "Rate of change (H)": 21,
            "Oscillation/pulsatility (I)": 18,
            "Clinical indices (J)": 14,
            "Glucagon suppression (K)": 5,
            "Glucose response (L)": 8,
            "Correlations (M)": 9,
            "Glucose-hormone (N)": 9,
            "Timing relationships (O)": 2,
        }
        print("\nFeature counts by category:")
        for label, count in categories.items():
            print(f"  {label:<35} {count:>3}")

        return df
