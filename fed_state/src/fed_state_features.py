"""Fed-state feature definitions: class ComprehensiveFedStateAnalyzer.

This script computes the features of the fed-state feature table (Supplementary Information,
Section S9) from its insulin, C-peptide and glucagon profiles and its start and end glucose and
ELISA values for each rat.

In the paper the profiles are the calibrated concentrations (pM); the input columns for this script
keep the names insulin_au, c_peptide_au and glucagon_au. extract_features.py runs this module.
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks, peak_widths, peak_prominences
try:
    from numpy import trapezoid as trapz
except ImportError:
    from numpy import trapz
import warnings
from itertools import combinations
import os
warnings.filterwarnings("ignore")


class ComprehensiveFedStateAnalyzer:
    """
    Comprehensive analysis pipeline for the fed state experiments by the QIRT-ELISA device

    Analyzer that calculates all metrics identified in the
    fed state experiment analysis of animal studies, including the following metrics:
    - Basic Device Measurements: SD, CV, mean, max, min, range of the calibrated profiles of the hormone/peptides
    - All glucose-adjusted metrics: values divded by the average gluose, normalized to reference glucose level of 5.6 mM, and change in glucose
    - Area under curve metrics: total, above baseline, below baseline, net AUC, normalized AUC, glucose-adjusted AUC
    - Oscillation pattern metrics: baseline crossing, autocorrelation, FTT-based rhythmicity index, entropy, complexity index, temporal stability
    - Inter-biomarker correlations: Pearson correlation between insulin, C-peptide and glucagon, lagged cross-correlations, Hilbert transform-based phase analysis
    - ELISA-validated quantitative metrics: two-point ELISA validation of the device measurements, equals to the device-derived features
    - Peak kinetics analysis: peak detection, peak width, peak prominence, peak frequency, peak timing, peak-to-peak intervals, and peak dynamics

    The data is expected to be in a CSV file with the following columns:
    - animal_id (unique identifier for each animal)
    - rat_type (type of rat; e.g. 'Healthy', 'Pre-diabetic', 'Diabetic')
    - timepoint (timepoint of the measurement; e.g. 0, 2.5, 5, 7.5, 10, 12.5, 15 minutes)
    - insulin_au (profile to analyze; calibrated concentration in pM in the paper, but kept in the input format as *_au)
    - c_peptide_au (profile to analyze; calibrated concentration in pM in the paper, but kept in the input format as *_au)
    - glucagon_au (profile to analyze; calibrated concentration in pM in the paper, but kept in the input format as *_au)
    - glucose_start_mM (Glucometer measurement at the start of the experiment in [mM])
    - glucose_end_mM (Glucometer measurement at the end of the experiment in [mM])
    - insulin_start_ELISA_pM (Conventional ELISA measurement at the start of the experiment in [pM])
    - insulin_end_ELISA_pM (Conventional ELISA measurement at the end of the experiment in [pM])
    - c_peptide_start_ELISA_pM (Conventional ELISA measurement at the start of the experiment in [pM])
    - c_peptide_end_ELISA_pM (Conventional ELISA measurement at the end of the experiment in [pM])
    - glucagon_start_ELISA_pM (Conventional ELISA measurement at the start of the experiment in [pM])
    - glucagon_end_ELISA_pM (Conventional ELISA measurement at the end of the experiment in [pM])
    """

    def __init__(self):
        self.biomarkers = ["insulin", "c_peptide", "glucagon"]
        self.data = None
        self.processed_data = {}
        self.group_metrics = {}
        self.comprehensive_metrics = {}
        self.peak_kinetics_data = {}

        # Analysis parameters (auto-detected)
        self.time_points = None
        self.num_timepoints = None
        self.time_interval = None
        self.experiment_duration = None

        # Results storage
        self.all_metric_results = {}
        self.statistical_comparisons = {}

    def load_csv_data(self, csv_path, format_type="long"):
        """Load experimental data from a CSV file"""

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        print("=" * 80)
        print("Comprehensive Analysis Pipeline for the Fed State Experiments by the QIRT-ELISA Device")
        print("=" * 80)
        print(f"Successfully loaded {len(df)} rows with {len(df.columns)} columns")

        # Auto-detect timepoint structure
        self.time_points, self.time_interval, self.num_timepoints, self.experiment_duration = self._detect_timepoint_structure(df)

        # Process data
        if format_type == "long":
            self.data = self._process_long_format_data(df)
        else:
            raise ValueError("Only 'long' format supported in this version")

        if len(self.data) == 0:
            print("ERROR: No valid animals loaded.")
            return None

        print(f"Successfully loaded data for {len(self.data)} animals:")
        for rat_type in self.data["rat_type"].unique():
            count = sum(self.data["rat_type"] == rat_type)
            print(f" - {rat_type}: {count} animals")

        return self.data

    def _detect_timepoint_structure(self, df):
        """Auto-detect timepoint structure from the data"""
        timepoints = sorted(df["timepoint"].unique())

        if len(timepoints) < 2:
            raise ValueError("Need at least 2 timepoints to detect structure")

        intervals = np.diff(timepoints)
        if not np.allclose(intervals, intervals[0], rtol=0.1):
            print(f"Warning: Irregular time intervals detected: {intervals}")

        time_interval = np.mean(intervals)
        num_timepoints = len(timepoints)
        experiment_duration = timepoints[-1] - timepoints[0]

        print(f"Auto-detected timepoint structure:")
        print(f" - Number of timepoints: {num_timepoints}")
        print(f" - Time interval: {time_interval:.1f} minutes")
        print(f" - Experiment duration: {experiment_duration:.1f} minutes")
        print(f" - Timepoints: {timepoints}")

        return np.array(timepoints), time_interval, num_timepoints, experiment_duration

    def _process_long_format_data(self, df):
        """Process long format CSV data"""
        required_cols = ["animal_id", "rat_type", "timepoint", "insulin_au", "c_peptide_au", "glucagon_au",
                        "glucose_start_mM", "glucose_end_mM"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check for ELISA columns
        elisa_cols = ["insulin_start_ELISA_pM", "insulin_end_ELISA_pM", "c_peptide_start_ELISA_pM",
                     "c_peptide_end_ELISA_pM", "glucagon_start_ELISA_pM", "glucagon_end_ELISA_pM"]
        has_elisa = all(col in df.columns for col in elisa_cols)

        if has_elisa:
            print("ELISA validation data detected - QUANTITATIVE analysis enabled.")
        else:
            print("No ELISA data - relative analysis only")

        animals_data = []

        for animal_id in df["animal_id"].unique():
            if pd.isna(animal_id):
                continue

            animal_df = df[df["animal_id"] == animal_id].sort_values("timepoint")

            if len(animal_df) != self.num_timepoints:
                print(f"Warning: Animal {animal_id} has {len(animal_df)} timepoints instead of {self.num_timepoints}")
                continue

            rat_type = animal_df["rat_type"].iloc[0]
            if isinstance(rat_type, str):
                rat_type = rat_type.lower().strip()

            animal_data = {
                "animal_id": animal_id,
                "rat_type": rat_type,
                "glucose_start_mM": animal_df["glucose_start_mM"].iloc[0],
                "glucose_end_mM": animal_df["glucose_end_mM"].iloc[0],
                "insulin_au": animal_df["insulin_au"].values,
                "c_peptide_au": animal_df["c_peptide_au"].values,
                "glucagon_au": animal_df["glucagon_au"].values,
                "has_elisa": has_elisa,
                "timepoints": np.array(sorted(animal_df["timepoint"].values))
            }

            # Add ELISA validation data if available
            if has_elisa:
                for biomarker in self.biomarkers:
                    start_elisa = animal_df[f"{biomarker}_start_ELISA_pM"].iloc[0]
                    end_elisa = animal_df[f"{biomarker}_end_ELISA_pM"].iloc[0]

                    if not (pd.isna(start_elisa) or pd.isna(end_elisa)):
                        animal_data[f"{biomarker}_start_ELISA_pM"] = start_elisa
                        animal_data[f"{biomarker}_end_ELISA_pM"] = end_elisa
                    else:
                        animal_data["has_elisa"] = False

            animals_data.append(animal_data)

        return pd.DataFrame(animals_data)

    def calculate_comprehensive_metrics(self, animal_id):
        """Calculate all metrics for a single animal subject of the fed state experiment"""

        if self.data is None:
            raise ValueError("No data loaded. Call load_csv_data() first.")

        animal_data = self.data[self.data["animal_id"] == animal_id]
        if len(animal_data) == 0:
            raise ValueError(f"Animal {animal_id} not found in data")

        animal_row = animal_data.iloc[0]

        # Basic data extraction
        raw_data = {
            "subject_id": animal_row["animal_id"],
            "condition": animal_row["rat_type"],
            "glucose_start": animal_row["glucose_start_mM"],
            "glucose_end": animal_row["glucose_end_mM"]
        }

        for biomarker in self.biomarkers:
            raw_data[biomarker] = animal_row[f"{biomarker}_au"]

        # Glucose calculations
        glucose_avg = (raw_data["glucose_start"] + raw_data["glucose_end"]) / 2
        glucose_change = raw_data["glucose_end"] - raw_data["glucose_start"]

        # Calculate all metric categories
        all_metrics = {}

        # 1. Basic Device Measurements
        all_metrics["basic_device"] = self._calculate_basic_device_metrics(raw_data)

        # 2. Glucose-Adjusted Metrics
        all_metrics["glucose_adjusted"] = self._calculate_glucose_adjusted_metrics(raw_data, glucose_avg, glucose_change)

        # 3. Peak and Dynamics Analysis
        all_metrics["peak_dynamics"] = self._calculate_peak_dynamics_metrics(raw_data)

        # 4. Rate of Change Analysis
        all_metrics["rate_analysis"] = self._calculate_rate_change_metrics(raw_data)

        # 5. Area Under Curve Metrics
        all_metrics["auc_metrics"] = self._calculate_auc_metrics(raw_data, glucose_avg)

        # 6. Oscillation and Pattern Metrics
        all_metrics["oscillation_patterns"] = self._calculate_oscillation_pattern_metrics(raw_data)

        # 7. Inter-biomarker Correlations
        all_metrics["correlations"] = self._calculate_correlation_metrics(raw_data, glucose_avg, glucose_change)

        # 8. ELISA-Validated Metrics (if available)
        if animal_row.get("has_elisa", False):
            all_metrics["elisa_quantitative"] = self._calculate_elisa_quantitative_metrics(animal_id, animal_row, raw_data)
        else:
            all_metrics["elisa_quantitative"] = None

        # 9. Pancreatic Cell Functionality
        all_metrics["pancreatic_function"] = self._calculate_pancreatic_function_metrics(all_metrics)

        # 10. Peak Kinetics Analysis
        all_metrics["peak_kinetics"] = self._calculate_peak_kinetics_metrics(raw_data)

        # Store results
        self.comprehensive_metrics[animal_id] = {
            "raw_data": raw_data,
            "glucose_avg": glucose_avg,
            "glucose_change": glucose_change,
            "metrics": all_metrics,
            "has_elisa": animal_row.get("has_elisa", False)
        }

        return all_metrics

    def _calculate_basic_device_metrics(self, raw_data):
        """Calculate basic device measurement metrics"""
        metrics = {}

        for biomarker in self.biomarkers:
            if biomarker in raw_data:
                signal = raw_data[biomarker]
                mean_val = np.mean(signal)
                std_val = np.std(signal)

                # `signal` is the calibrated profile (pM); the *_raw_au names are kept from the input format
                # and extract_features.py renames them *_raw_pM in the exported table.
                metrics[biomarker] = {
                    # Calibrated profile measurements (pM)
                    "mean_raw_au": float(mean_val),
                    "std_raw_au": float(std_val),
                    "cv_raw_percent": float(std_val / mean_val * 100) if mean_val > 0 else 0.0,
                    "max_raw_au": float(np.max(signal)),
                    "min_raw_au": float(np.min(signal)),
                    "range_raw_au": float(np.max(signal) - np.min(signal)),

                    # Signal characteristics
                    "signal_values": signal
                }

        return metrics

    def _calculate_glucose_adjusted_metrics(self, raw_data, glucose_avg, glucose_change):
        """Calculate all glucose-adjusted metrics"""
        metrics = {}

        for biomarker in self.biomarkers:
            if biomarker in raw_data:
                signal = raw_data[biomarker]
                mean_signal = np.mean(signal)
                std_signal = np.std(signal)
                max_signal = np.max(signal)
                min_signal = np.min(signal)

                metrics[biomarker] = {
                    # Glucose efficiency metrics
                    "glucose_efficiency": float(mean_signal / glucose_avg),
                    "peak_glucose_efficiency": float(max_signal / glucose_avg),
                    "min_glucose_efficiency": float(min_signal / glucose_avg),
                    "glucose_efficiency_variability": float(std_signal / glucose_avg),

                    # Glucose-corrected values (normalized to 5.6 mM reference glucose)
                    "mean_glucose_corrected_5p6": float(mean_signal * (5.6 / glucose_avg)),
                    "max_glucose_corrected_5p6": float(max_signal * (5.6 / glucose_avg)),
                    "min_glucose_corrected_5p6": float(min_signal * (5.6 / glucose_avg)),
                    "std_glucose_corrected_5p6": float(std_signal * (5.6 / glucose_avg)),

                    # Glucose sensitivity metrics
                    "glucose_sensitivity": float(std_signal / abs(glucose_change)) if abs(glucose_change) > 0.001 else float(std_signal),
                    "glucose_responsiveness": float((max_signal - min_signal) / abs(glucose_change)) if abs(glucose_change) > 0.001 else float(max_signal - min_signal),

                    # Glucose-adjusted variability
                    "glucose_adjusted_cv": float((std_signal / glucose_avg) / (mean_signal / glucose_avg) * 100) if mean_signal > 0 else 0.0,

                    # Glucose trend correlation
                    "glucose_correlation": self._calculate_glucose_correlation(signal, glucose_avg, glucose_change)
                }

        return metrics

    def _calculate_glucose_correlation(self, signal, glucose_avg, glucose_change):
        """Calculate correlation with glucose trend"""
        if abs(glucose_change) < 0.001:
            # No glucose change - use flat trend
            glucose_trend = np.full(len(signal), glucose_avg)
        else:
            # Linear glucose trend from start to end
            glucose_start = glucose_avg - glucose_change/2
            glucose_end = glucose_avg + glucose_change/2
            glucose_trend = np.linspace(glucose_start, glucose_end, len(signal))

        correlation, p_value = stats.pearsonr(signal, glucose_trend)

        return {
            "correlation": float(correlation),
            "p_value": float(p_value),
            "significant": p_value < 0.05
        }

    def _calculate_peak_dynamics_metrics(self, raw_data):
        """Calculate peak detection and dynamics metrics"""

        metrics = {}

        for biomarker in self.biomarkers:
            if biomarker in raw_data:
                signal = raw_data[biomarker]
                baseline = np.mean(signal)
                signal_std = np.std(signal)

                # Detect ALL local maxima (no threshold)
                # This captures peaks even in declining signal phases
                peaks_all_local, _ = find_peaks(signal)

                # For backward compatibility, also detect peaks above mean
                peaks_above_mean, _ = find_peaks(signal, height=baseline)

                # Detect troughs (local minima)
                troughs, _ = find_peaks(-signal)

                # Peak characteristics using all local maxima
                if len(peaks_all_local) > 0:
                    try:
                        prominences = peak_prominences(signal, peaks_all_local)[0]
                        widths = peak_widths(signal, peaks_all_local, rel_height=0.5)[0] * self.time_interval
                    except:
                        prominences = np.abs(signal[peaks_all_local] - baseline)
                        widths = np.full(len(peaks_all_local), self.time_interval)
                else:
                    prominences = np.array([])
                    widths = np.array([])

                metrics[biomarker] = {
                    # Peak counts
                    "num_peaks_above_mean": len(peaks_above_mean),
                    "num_peaks_above_std": len(peaks_all_local),
                    "num_troughs": len(troughs),
                    "peak_frequency": len(peaks_all_local) / self.experiment_duration if self.experiment_duration > 0 else 0,

                    # Peak characteristics
                    "mean_peak_prominence": float(np.mean(prominences)) if len(prominences) > 0 else 0.0,
                    "max_peak_prominence": float(np.max(prominences)) if len(prominences) > 0 else 0.0,
                    "mean_peak_width": float(np.mean(widths)) if len(widths) > 0 else 0.0,

                    # Signal extremes
                    "max_value": float(np.max(signal)),
                    "min_value": float(np.min(signal)),
                    "max_time": float(self.time_points[np.argmax(signal)]),
                    "min_time": float(self.time_points[np.argmin(signal)]),

                    # Dynamic range
                    "dynamic_range_ratio": float(np.max(signal) / np.min(signal)) if np.min(signal) > 0 else np.inf,
                    "total_excursion": float(np.max(signal) - np.min(signal)),
                    "relative_excursion": float((np.max(signal) - np.min(signal)) / baseline) if baseline > 0 else 0.0,

                    # Variability metrics
                    "coefficient_of_variation": float(signal_std / baseline * 100) if baseline > 0 else 0.0,
                    "relative_std": float(signal_std / baseline) if baseline > 0 else 0.0,

                    # Timing metrics
                    "peak_times": self.time_points[peaks_all_local] if len(peaks_all_local) > 0 else np.array([]),
                    "trough_times": self.time_points[troughs] if len(troughs) > 0 else np.array([])
                }

        return metrics

    def _calculate_rate_change_metrics(self, raw_data):
        """Calculate rate of change metrics"""
        metrics = {}

        for biomarker in self.biomarkers:
            if biomarker in raw_data:
                signal = raw_data[biomarker]

                # Point-to-point rates
                rates = np.diff(signal) / self.time_interval

                # Smoothed rates (3-point average)
                if len(signal) >= 3:
                    smooth_rates = []
                    for i in range(1, len(signal) - 1):
                        rate = (signal[i+1] - signal[i-1]) / (2 * self.time_interval)
                        smooth_rates.append(rate)
                    smooth_rates = np.array(smooth_rates)
                else:
                    smooth_rates = rates

                # Acceleration (second derivative)
                if len(rates) > 1:
                    accelerations = np.diff(rates) / self.time_interval
                else:
                    accelerations = np.array([0])

                metrics[biomarker] = {
                    # Basic rates
                    "max_increase_rate": float(np.max(rates)) if len(rates) > 0 else 0.0,
                    "max_decrease_rate": float(abs(np.min(rates))) if len(rates) > 0 else 0.0,
                    "mean_absolute_rate": float(np.mean(np.abs(rates))) if len(rates) > 0 else 0.0,
                    "rate_variability": float(np.std(rates)) if len(rates) > 0 else 0.0,

                    # Smoothed rates
                    "max_smooth_increase_rate": float(np.max(smooth_rates)) if len(smooth_rates) > 0 else 0.0,
                    "max_smooth_decrease_rate": float(abs(np.min(smooth_rates))) if len(smooth_rates) > 0 else 0.0,
                    "mean_smooth_absolute_rate": float(np.mean(np.abs(smooth_rates))) if len(smooth_rates) > 0 else 0.0,

                    # Acceleration metrics
                    "max_acceleration": float(np.max(accelerations)) if len(accelerations) > 0 else 0.0,
                    "max_deceleration": float(abs(np.min(accelerations))) if len(accelerations) > 0 else 0.0,
                    "mean_acceleration": float(np.mean(accelerations)) if len(accelerations) > 0 else 0.0,

                    # Rate statistics
                    "rate_range": float(np.max(rates) - np.min(rates)) if len(rates) > 0 else 0.0,
                    "rate_cv": float(np.std(rates) / np.mean(np.abs(rates)) * 100) if len(rates) > 0 and np.mean(np.abs(rates)) > 0 else 0.0,

                    # Trend analysis
                    "trend_slope": self._calculate_trend_slope(signal),
                    "trend_r_squared": self._calculate_trend_r_squared(signal),

                    # Raw arrays for further analysis
                    "all_rates": rates,
                    "smooth_rates": smooth_rates
                }

        return metrics

    def _calculate_trend_slope(self, signal):
        """Calculate linear trend slope"""
        try:
            slope, _, r_value, p_value, std_err = stats.linregress(self.time_points, signal)
            return {
                "slope": float(slope),
                "r_value": float(r_value),
                "r_squared": float(r_value**2),
                "p_value": float(p_value),
                "std_err": float(std_err),
                "significant": p_value < 0.05
            }
        except:
            return {
                "slope": 0.0,
                "r_value": 0.0,
                "r_squared": 0.0,
                "p_value": 1.0,
                "std_err": 0.0,
                "significant": False
            }

    def _calculate_trend_r_squared(self, signal):
        """Calculate R-squared for linear trend"""
        try:
            _, _, r_value, _, _ = stats.linregress(self.time_points, signal)
            return float(r_value**2)
        except:
            return 0.0

    def _calculate_auc_metrics(self, raw_data, glucose_avg):
        """Calculate area under curve metrics"""
        metrics = {}

        for biomarker in self.biomarkers:
            if biomarker in raw_data:
                signal = raw_data[biomarker]
                baseline = np.mean(signal)

                # Various AUC calculations
                auc_total = trapz(signal, self.time_points)
                auc_above_baseline = trapz(np.maximum(signal - baseline, 0), self.time_points)
                auc_below_baseline = trapz(np.maximum(baseline - signal, 0), self.time_points)

                # Not exported: for a profile in pM, auc_normalized equals auc_total, and the 1.0 reference of
                # auc_above_mean / auc_below_mean (and so auc_normalized_net) has no meaning.
                normalized_signal = signal
                auc_normalized = trapz(normalized_signal, self.time_points)
                auc_above_mean = trapz(np.maximum(normalized_signal - 1.0, 0), self.time_points)
                auc_below_mean = trapz(np.maximum(1.0 - normalized_signal, 0), self.time_points)

                # Glucose-adjusted AUC
                auc_glucose_corrected = auc_total * (5.6 / glucose_avg)
                auc_efficiency = auc_total / glucose_avg

                metrics[biomarker] = {
                    # Basic AUC metrics
                    "auc_total": float(auc_total),
                    "auc_above_baseline": float(auc_above_baseline),
                    "auc_below_baseline": float(auc_below_baseline),
                    "auc_net": float(auc_above_baseline - auc_below_baseline),

                    # Normalized AUC
                    "auc_normalized": float(auc_normalized),
                    "auc_above_mean": float(auc_above_mean),
                    "auc_below_mean": float(auc_below_mean),
                    "auc_normalized_net": float(auc_above_mean - auc_below_mean),

                    # Glucose-adjusted AUC
                    "auc_glucose_corrected_5p6": float(auc_glucose_corrected),
                    "auc_glucose_efficiency": float(auc_efficiency),

                    # Relative AUC metrics
                    "auc_fraction_above_baseline": float(auc_above_baseline / auc_total) if auc_total > 0 else 0.0,
                    "auc_balance_ratio": float(auc_above_baseline / auc_below_baseline) if auc_below_baseline > 0 else np.inf,

                    # Time-weighted averages
                    "time_weighted_average": float(auc_total / self.experiment_duration) if self.experiment_duration > 0 else 0.0,
                    "time_weighted_glucose_efficiency": float(auc_efficiency / self.experiment_duration) if self.experiment_duration > 0 else 0.0
                }

        return metrics

    def _calculate_oscillation_pattern_metrics(self, raw_data):
        """Calculate oscillation and pattern metrics"""
        metrics = {}

        for biomarker in self.biomarkers:
            if biomarker in raw_data:
                signal = raw_data[biomarker]
                baseline = np.mean(signal)

                # Zero crossings (crossings of baseline)
                centered_signal = signal - baseline
                zero_crossings = len(np.where(np.diff(np.signbit(centered_signal)))[0])

                # Oscillation frequency
                oscillation_frequency = zero_crossings / self.experiment_duration if self.experiment_duration > 0 else 0

                # Pattern regularity (autocorrelation)
                try:
                    autocorr = np.correlate(centered_signal, centered_signal, mode="full")
                    autocorr = autocorr[autocorr.size // 2:]
                    autocorr = autocorr / autocorr[0] # Normalize
                    pattern_regularity = float(np.max(autocorr[1:min(3, len(autocorr)-1)])) if len(autocorr) > 1 else 0.0
                except:
                    pattern_regularity = 0.0

                # Rhythmicity index (FFT-based)
                try:
                    fft_signal = np.fft.fft(centered_signal)
                    power_spectrum = np.abs(fft_signal)**2
                    # Find dominant frequency component
                    dominant_freq_power = np.max(power_spectrum[1:len(power_spectrum)//2])
                    total_power = np.sum(power_spectrum[1:len(power_spectrum)//2])
                    rhythmicity_index = float(dominant_freq_power / total_power) if total_power > 0 else 0.0
                except:
                    rhythmicity_index = 0.0

                # Phase analysis (if multiple oscillations detected)
                phase_consistency = self._calculate_phase_consistency(signal, baseline)

                metrics[biomarker] = {
                    # Basic oscillation metrics
                    "zero_crossings": zero_crossings,
                    "oscillation_frequency": float(oscillation_frequency),
                    "pattern_regularity": pattern_regularity,
                    "rhythmicity_index": rhythmicity_index,

                    # Phase and timing metrics
                    "phase_consistency": phase_consistency,
                    "dominant_period": float(2 * self.experiment_duration / zero_crossings) if zero_crossings > 0 else np.inf,

                    # Signal complexity
                    "signal_entropy": self._calculate_signal_entropy(signal),
                    "complexity_index": self._calculate_complexity_index(signal),

                    # Stability metrics
                    "temporal_stability": self._calculate_temporal_stability(signal),
                    "pattern_consistency": self._calculate_pattern_consistency(signal)
                }

        return metrics

    def _calculate_phase_consistency(self, signal, baseline):
        """Calculate phase consistency of oscillations"""
        try:
            # Find peaks and calculate inter-peak intervals
            peaks, _ = find_peaks(signal, height=baseline)
            if len(peaks) > 2:
                intervals = np.diff(peaks) * self.time_interval
                consistency = 1.0 - (np.std(intervals) / np.mean(intervals)) if np.mean(intervals) > 0 else 0.0
                return float(max(0.0, consistency))
            else:
                return 0.0
        except:
            return 0.0

    def _calculate_signal_entropy(self, signal):
        """Calculate Shannon entropy of signal"""
        try:
            # Discretize signal into bins
            bins = min(10, len(signal) // 2)
            hist, _ = np.histogram(signal, bins=bins, density=True)
            hist = hist[hist > 0] # Remove zero entries
            entropy = -np.sum(hist * np.log2(hist))
            return float(entropy)
        except:
            return 0.0

    def _calculate_complexity_index(self, signal):
        """Calculate complexity index based on signal structure"""
        try:
            # Calculate first and second differences
            first_diff = np.diff(signal)
            second_diff = np.diff(first_diff)

            # Complexity as ratio of second to first derivative variance
            if np.var(first_diff) > 0:
                complexity = np.var(second_diff) / np.var(first_diff)
            else:
                complexity = 0.0

            return float(complexity)
        except:
            return 0.0

    def _calculate_temporal_stability(self, signal):
        """Calculate temporal stability of signal"""
        try:
            # Split signal into halves and compare
            mid = len(signal) // 2
            first_half = signal[:mid]
            second_half = signal[-mid:]

            if len(first_half) > 1 and len(second_half) > 1:
                correlation, _ = stats.pearsonr(first_half, second_half[:len(first_half)])
                return float(correlation) if not np.isnan(correlation) else 0.0
            else:
                return 0.0
        except:
            return 0.0

    def _calculate_pattern_consistency(self, signal):
        """Calculate pattern consistency across timepoints"""
        try:
            # Calculate moving coefficient of variation
            window = min(3, len(signal) // 2)
            if window < 2:
                return 0.0

            cvs = []
            for i in range(len(signal) - window + 1):
                window_data = signal[i:i+window]
                if np.mean(window_data) > 0:
                    cv = np.std(window_data) / np.mean(window_data)
                    cvs.append(cv)

            if cvs:
                # Consistency is inverse of CV variability
                cv_stability = 1.0 / (1.0 + np.std(cvs))
                return float(cv_stability)
            else:
                return 0.0
        except:
            return 0.0

    def _calculate_correlation_metrics(self, raw_data, glucose_avg, glucose_change):
        """Calculate inter-biomarker correlation metrics"""
        correlations = {}

        # Pairwise biomarker correlations
        biomarker_pairs = [("insulin", "c_peptide"), ("insulin", "glucagon"), ("c_peptide", "glucagon")]

        for bm1, bm2 in biomarker_pairs:
            if bm1 in raw_data and bm2 in raw_data:
                signal1 = raw_data[bm1]
                signal2 = raw_data[bm2]

                try:
                    # Basic correlation
                    corr, p_val = stats.pearsonr(signal1, signal2)

                    # Partial correlation (controlling for time trend)
                    time_trend = np.linspace(0, 1, len(signal1))
                    partial_corr = self._calculate_partial_correlation(signal1, signal2, time_trend)

                    # Cross-correlation (time-lagged)
                    cross_corr = self._calculate_cross_correlation(signal1, signal2)

                    # Phase relationship
                    phase_relationship = self._calculate_phase_relationship(signal1, signal2)

                    correlations[f"{bm1}_{bm2}"] = {
                        "correlation": float(corr),
                        "p_value": float(p_val),
                        "significant": p_val < 0.05,
                        "partial_correlation": partial_corr,
                        "max_cross_correlation": cross_corr["max_correlation"],
                        "optimal_lag": cross_corr["optimal_lag"],
                        "phase_lag": phase_relationship["phase_lag"],
                        "coherence": phase_relationship["coherence"]
                    }

                except Exception as e:
                    correlations[f"{bm1}_{bm2}"] = {
                        "correlation": 0.0,
                        "p_value": 1.0,
                        "significant": False,
                        "partial_correlation": 0.0,
                        "max_cross_correlation": 0.0,
                        "optimal_lag": 0.0,
                        "phase_lag": 0.0,
                        "coherence": 0.0
                    }

        # Individual biomarker vs glucose correlations
        glucose_correlations = {}
        for biomarker in self.biomarkers:
            if biomarker in raw_data:
                glucose_corr = self._calculate_glucose_correlation(raw_data[biomarker], glucose_avg, glucose_change)
                glucose_correlations[biomarker] = glucose_corr

        return {
            "biomarker_correlations": correlations,
            "glucose_correlations": glucose_correlations
        }

    def _calculate_partial_correlation(self, x, y, z):
        """Calculate partial correlation between x and y controlling for z"""
        try:
            # Simple partial correlation using residuals
            x_resid = x - np.polyval(np.polyfit(z, x, 1), z)
            y_resid = y - np.polyval(np.polyfit(z, y, 1), z)

            corr, _ = stats.pearsonr(x_resid, y_resid)
            return float(corr) if not np.isnan(corr) else 0.0
        except:
            return 0.0

    def _calculate_cross_correlation(self, signal1, signal2):
        """Calculate cross-correlation with time lags"""
        try:
            # Normalize signals
            s1 = (signal1 - np.mean(signal1)) / np.std(signal1)
            s2 = (signal2 - np.mean(signal2)) / np.std(signal2)

            # Calculate cross-correlation
            cross_corr = np.correlate(s1, s2, mode="full")

            # Find maximum correlation and corresponding lag
            max_corr_idx = np.argmax(np.abs(cross_corr))
            max_correlation = cross_corr[max_corr_idx]

            # Convert index to time lag
            lags = np.arange(-len(s2) + 1, len(s1))
            optimal_lag = lags[max_corr_idx] * self.time_interval

            return {
                "max_correlation": float(max_correlation),
                "optimal_lag": float(optimal_lag)
            }
        except:
            return {
                "max_correlation": 0.0,
                "optimal_lag": 0.0
            }

    def _calculate_phase_relationship(self, signal1, signal2):
        """Calculate phase relationship between two signals"""
        try:
            # Use Hilbert transform to get phase
            from scipy.signal import hilbert

            analytic1 = hilbert(signal1 - np.mean(signal1))
            analytic2 = hilbert(signal2 - np.mean(signal2))

            phase1 = np.angle(analytic1)
            phase2 = np.angle(analytic2)

            # Phase difference
            phase_diff = phase1 - phase2

            # Wrap to [-pi, pi]
            phase_diff = np.mod(phase_diff + np.pi, 2*np.pi) - np.pi

            # Mean phase lag and coherence
            mean_phase_lag = np.mean(phase_diff)
            phase_coherence = np.abs(np.mean(np.exp(1j * phase_diff)))

            return {
                "phase_lag": float(mean_phase_lag),
                "coherence": float(phase_coherence)
            }
        except:
            return {
                "phase_lag": 0.0,
                "coherence": 0.0
            }

    def _calculate_elisa_quantitative_metrics(self, animal_id, animal_row, raw_data):
        """Category 8: ELISA quantitative metrics; written to the feature table, not used for classification.

        The input profiles are already calibrated, so this second two-point calibration has slope 1 and
        calibration error 0, and its metrics repeat those of the calibrated profile (for example, the mean
        concentration equals the basic mean). Seven metrics per hormone are exported as <hormone>_elisa_*;
        extract_features.py leaves all 21 out of the classification matrix. The other values are not exported.
        """
        try:
            # Get ELISA calibration data
            elisa_metrics = {}

            for biomarker in self.biomarkers:
                if (f"{biomarker}_start_ELISA_pM" in animal_row.index and
                    f"{biomarker}_end_ELISA_pM" in animal_row.index):

                    device_signal = raw_data[biomarker]
                    elisa_start = animal_row[f"{biomarker}_start_ELISA_pM"]
                    elisa_end = animal_row[f"{biomarker}_end_ELISA_pM"]

                    # Two-point calibration
                    device_start = device_signal[0]
                    device_end = device_signal[-1]

                    if device_end != device_start:
                        slope = (elisa_end - elisa_start) / (device_end - device_start)
                        intercept = elisa_start - slope * device_start

                        # Convert all timepoints to pM
                        concentrations_pM = slope * device_signal + intercept
                    else:
                        # No device change - use average ELISA
                        concentrations_pM = np.full(len(device_signal), (elisa_start + elisa_end) / 2)
                        slope = 0
                        intercept = (elisa_start + elisa_end) / 2

                    # Calculate quantitative metrics
                    mean_conc = np.mean(concentrations_pM)
                    std_conc = np.std(concentrations_pM)

                    elisa_metrics[biomarker] = {
                        # Absolute concentrations
                        "mean_concentration_pM": float(mean_conc),
                        "std_concentration_pM": float(std_conc),
                        "cv_concentration_percent": float(std_conc / mean_conc * 100) if mean_conc > 0 else 0.0,
                        "max_concentration_pM": float(np.max(concentrations_pM)),
                        "min_concentration_pM": float(np.min(concentrations_pM)),
                        "concentration_range_pM": float(np.max(concentrations_pM) - np.min(concentrations_pM)),

                        # Dynamic metrics
                        "absolute_dynamic_range": float(np.max(concentrations_pM) / np.min(concentrations_pM)) if np.min(concentrations_pM) > 0 else np.inf,
                        "concentration_auc_pM_min": float(trapz(concentrations_pM, self.time_points)),

                        # Rate metrics
                        "max_secretion_rate_pM_per_min": float(np.max(np.diff(concentrations_pM)) / self.time_interval) if len(concentrations_pM) > 1 else 0.0,
                        "max_clearance_rate_pM_per_min": float(abs(np.min(np.diff(concentrations_pM))) / self.time_interval) if len(concentrations_pM) > 1 else 0.0,
                        "mean_abs_rate_pM_per_min": float(np.mean(np.abs(np.diff(concentrations_pM))) / self.time_interval) if len(concentrations_pM) > 1 else 0.0,

                        # Calibration quality
                        "calibration_slope": float(slope),
                        "calibration_intercept": float(intercept),
                        "elisa_start_pM": float(elisa_start),
                        "elisa_end_pM": float(elisa_end),
                        "calibration_error_percent": self._calculate_calibration_error(device_signal, concentrations_pM, elisa_start, elisa_end),

                        # Glucose relationships in pM units
                        "concentration_per_mM_glucose": float(mean_conc / raw_data["glucose_start"]),
                        "glucose_sensitivity_pM_per_mM": float(std_conc / abs(raw_data["glucose_end"] - raw_data["glucose_start"])) if abs(raw_data["glucose_end"] - raw_data["glucose_start"]) > 0.001 else float(std_conc),

                        # Full concentration array
                        "concentrations_pM": concentrations_pM
                    }

            return elisa_metrics

        except Exception as e:
            print(f"Error calculating ELISA metrics for {animal_id}: {e}")
            return None

    def _calculate_calibration_error(self, device_signal, concentrations_pM, elisa_start, elisa_end):
        """Calculate calibration error percentage"""
        try:
            predicted_start = concentrations_pM[0]
            predicted_end = concentrations_pM[-1]

            start_error = abs(predicted_start - elisa_start) / elisa_start * 100 if elisa_start > 0 else 0
            end_error = abs(predicted_end - elisa_end) / elisa_end * 100 if elisa_end > 0 else 0

            return float((start_error + end_error) / 2)
        except:
            return 0.0

    def _calculate_pancreatic_function_metrics(self, all_metrics):
        """Calculate comprehensive pancreatic cell functionality metrics"""
        function_metrics = {}

        # Beta-cell function metrics
        beta_cell_metrics = {}

        if ("insulin_c_peptide" in all_metrics.get("correlations", {}).get("biomarker_correlations", {}) and
            "insulin" in all_metrics.get("basic_device", {}) and
            "c_peptide" in all_metrics.get("basic_device", {})):

            # Insulin-C-peptide coordination
            insulin_cpep_corr = all_metrics["correlations"]["biomarker_correlations"].get("insulin_c_peptide", {})
            beta_cell_metrics["insulin_cpeptide_correlation"] = insulin_cpep_corr.get("correlation", 0.0)
            beta_cell_metrics["coordination_significance"] = insulin_cpep_corr.get("significant", False)
            beta_cell_metrics["coordination_quality"] = self._classify_correlation(beta_cell_metrics["insulin_cpeptide_correlation"])

            # Secretory stability
            insulin_cv = all_metrics["basic_device"]["insulin"].get("cv_raw_percent", 0)
            cpeptide_cv = all_metrics["basic_device"]["c_peptide"].get("cv_raw_percent", 0)
            avg_cv = (insulin_cv + cpeptide_cv) / 2
            beta_cell_metrics["secretory_stability_cv"] = avg_cv
            beta_cell_metrics["stability_status"] = self._classify_stability(avg_cv)

            # Glucose responsiveness
            insulin_glucose_resp = all_metrics["glucose_adjusted"]["insulin"].get("glucose_responsiveness", 0)
            cpeptide_glucose_resp = all_metrics["glucose_adjusted"]["c_peptide"].get("glucose_responsiveness", 0)
            avg_responsiveness = (insulin_glucose_resp + cpeptide_glucose_resp) / 2
            beta_cell_metrics["glucose_responsiveness"] = avg_responsiveness
            beta_cell_metrics["responsiveness_status"] = self._classify_responsiveness(avg_responsiveness)

            # Dynamic range capability
            insulin_range = all_metrics["peak_dynamics"]["insulin"].get("dynamic_range_ratio", 1.0)
            cpeptide_range = all_metrics["peak_dynamics"]["c_peptide"].get("dynamic_range_ratio", 1.0)
            avg_range = (insulin_range + cpeptide_range) / 2
            beta_cell_metrics["dynamic_range"] = avg_range
            beta_cell_metrics["range_status"] = self._classify_dynamic_range(avg_range)

            # Secretory coherence (pattern matching)
            coherence = insulin_cpep_corr.get("coherence", 0.0)
            beta_cell_metrics["secretory_coherence"] = coherence
            beta_cell_metrics["coherence_status"] = self._classify_coherence(coherence)

        # Alpha-cell function metrics
        alpha_cell_metrics = {}

        if "glucagon" in all_metrics["basic_device"]:
            # Glucose suppression ability
            glucagon_glucose_corr = all_metrics["correlations"]["glucose_correlations"].get("glucagon", {})
            glucose_suppression = glucagon_glucose_corr.get("correlation", 0.0)
            alpha_cell_metrics["glucose_suppression_correlation"] = glucose_suppression
            alpha_cell_metrics["suppression_quality"] = self._classify_suppression(glucose_suppression)

            # Regulatory precision
            glucagon_cv = all_metrics["basic_device"]["glucagon"].get("cv_raw_percent", 0)
            alpha_cell_metrics["regulatory_precision_cv"] = glucagon_cv
            alpha_cell_metrics["precision_status"] = self._classify_precision(glucagon_cv)

            # Counter-regulatory capacity
            glucagon_range = all_metrics["peak_dynamics"]["glucagon"].get("dynamic_range_ratio", 1.0)
            alpha_cell_metrics["counter_regulatory_range"] = glucagon_range
            alpha_cell_metrics["counter_reg_status"] = self._classify_counter_regulation(glucagon_range)

            # Insulin-glucagon reciprocity
            insulin_glucagon_corr = all_metrics["correlations"]["biomarker_correlations"].get("insulin_glucagon", {})
            reciprocity = insulin_glucagon_corr.get("correlation", 0.0)
            alpha_cell_metrics["insulin_glucagon_reciprocity"] = reciprocity
            alpha_cell_metrics["reciprocity_status"] = self._classify_reciprocity(reciprocity)

        # Overall pancreatic function score
        overall_score = self._calculate_overall_function_score(beta_cell_metrics, alpha_cell_metrics)

        function_metrics = {
            "beta_cell_function": beta_cell_metrics,
            "alpha_cell_function": alpha_cell_metrics,
            "overall_function_score": overall_score,
            "function_classification": self._classify_overall_function(overall_score)
        }

        return function_metrics

    def _calculate_peak_kinetics_metrics(self, raw_data):
        """Calculate detailed peak kinetics analysis"""
        kinetics_metrics = {}

        for biomarker in self.biomarkers:
            if biomarker in raw_data:
                signal = raw_data[biomarker]
                baseline = np.mean(signal)

                # Detect all local maxima (no threshold)
                peaks, _ = find_peaks(signal)

                if len(peaks) > 0:
                    # Calculate rise and decline characteristics
                    rise_times = []
                    decline_times = []
                    rise_rates = []
                    decline_rates = []

                    # CHECK IF SIGNAL STARTS WITH A FALL (e.g., glucagon)
                    # If first trough comes BEFORE first peak AND at a later timepoint
                    first_trough_idx = np.argmin(signal[:peaks[0]+1])

                    # INITIAL FALL PHASE: If signal drops from start before rising to first peak
                    if first_trough_idx > 0 and first_trough_idx < peaks[0]:
                        # There's a trough between start and first peak
                        decline_time = first_trough_idx * self.time_interval
                        decline_height = signal[0] - signal[first_trough_idx]
                        if decline_time > 0 and decline_height > 0:
                            decline_rate = decline_height / decline_time
                            decline_times.append(decline_time)
                            decline_rates.append(decline_rate)

                    # FIRST RISE PHASE: From trough to first peak
                    if peaks[0] > first_trough_idx:
                        rise_time = (peaks[0] - first_trough_idx) * self.time_interval
                        rise_height = signal[peaks[0]] - signal[first_trough_idx]
                        if rise_time > 0 and rise_height > 0:
                            rise_rate = rise_height / rise_time
                            rise_times.append(rise_time)
                            rise_rates.append(rise_rate)

                    # INTER-PEAK PHASES
                    for i in range(len(peaks) - 1):
                        # Find trough between peaks[i] and peaks[i+1]
                        trough_idx = np.argmin(signal[peaks[i]:peaks[i+1]]) + peaks[i]

                        # Fall phase: From current peak DOWN to trough
                        if trough_idx > peaks[i]:
                            decline_time = (trough_idx - peaks[i]) * self.time_interval
                            decline_height = signal[peaks[i]] - signal[trough_idx]
                            if decline_time > 0 and decline_height > 0:
                                decline_rate = decline_height / decline_time
                                decline_times.append(decline_time)
                                decline_rates.append(decline_rate)

                        # Rise phase: From trough UP to next peak
                        if peaks[i+1] > trough_idx:
                            rise_time = (peaks[i+1] - trough_idx) * self.time_interval
                            rise_height = signal[peaks[i+1]] - signal[trough_idx]
                            if rise_time > 0 and rise_height > 0:
                                rise_rate = rise_height / rise_time
                                rise_times.append(rise_time)
                                rise_rates.append(rise_rate)

                    # LAST FALL PHASE: From last peak to end
                    last_trough_idx = np.argmin(signal[peaks[-1]:]) + peaks[-1]
                    if last_trough_idx > peaks[-1]:
                        decline_time = (last_trough_idx - peaks[-1]) * self.time_interval
                        decline_height = signal[peaks[-1]] - signal[last_trough_idx]
                        if decline_time > 0 and decline_height > 0:
                            decline_rate = decline_height / decline_time
                            decline_times.append(decline_time)
                            decline_rates.append(decline_rate)

                    # Inter-peak analysis
                    if len(peaks) > 1:
                        inter_peak_intervals = np.diff(peaks) * self.time_interval
                        peak_frequency = len(peaks) / self.experiment_duration
                    else:
                        inter_peak_intervals = []
                        peak_frequency = 0

                    kinetics_metrics[biomarker] = {
                        # Peak characteristics
                        "peak_count": len(peaks),
                        "peak_frequency": float(peak_frequency),
                        "peak_times": self.time_points[peaks],
                        "peak_heights": signal[peaks],

                        # Rise kinetics (using validated values only)
                        "mean_rise_time": float(np.mean(rise_times)) if rise_times else 0.0,
                        "std_rise_time": float(np.std(rise_times)) if rise_times else 0.0,
                        "mean_rise_rate": float(np.mean(rise_rates)) if rise_rates else 0.0,
                        "max_rise_rate": float(np.max(rise_rates)) if rise_rates else 0.0,

                        # Decline kinetics (using validated values only)
                        "mean_decline_time": float(np.mean(decline_times)) if decline_times else 0.0,
                        "std_decline_time": float(np.std(decline_times)) if decline_times else 0.0,
                        "mean_decline_rate": float(np.mean(decline_rates)) if decline_rates else 0.0,
                        "max_decline_rate": float(np.max(decline_rates)) if decline_rates else 0.0,

                        # Inter-peak analysis
                        "mean_inter_peak_interval": float(np.mean(inter_peak_intervals)) if len(inter_peak_intervals) > 0 else 0.0,
                        "inter_peak_variability": float(np.std(inter_peak_intervals)) if len(inter_peak_intervals) > 0 else 0.0,

                        # Kinetic ratios (with safety checks)
                        "rise_decline_ratio": float(np.mean(rise_times) / np.mean(decline_times)) if rise_times and decline_times and np.mean(decline_times) > 0 else 0.0,
                        "secretion_clearance_ratio": float(np.mean(rise_rates) / np.mean(decline_rates)) if rise_rates and decline_rates and np.mean(decline_rates) > 0 else 0.0,

                        # Raw arrays for further analysis
                        "all_rise_times": np.array(rise_times),
                        "all_decline_times": np.array(decline_times),
                        "all_rise_rates": np.array(rise_rates),
                        "all_decline_rates": np.array(decline_rates)
                    }
                else:
                    # No peaks detected
                    kinetics_metrics[biomarker] = {
                        "peak_count": 0,
                        "peak_frequency": 0.0,
                        "peak_times": np.array([]),
                        "peak_heights": np.array([]),
                        "mean_rise_time": 0.0,
                        "std_rise_time": 0.0,
                        "mean_rise_rate": 0.0,
                        "max_rise_rate": 0.0,
                        "mean_decline_time": 0.0,
                        "std_decline_time": 0.0,
                        "mean_decline_rate": 0.0,
                        "max_decline_rate": 0.0,
                        "mean_inter_peak_interval": 0.0,
                        "inter_peak_variability": 0.0,
                        "rise_decline_ratio": 0.0,
                        "secretion_clearance_ratio": 0.0,
                        "all_rise_times": np.array([]),
                        "all_decline_times": np.array([]),
                        "all_rise_rates": np.array([]),
                        "all_decline_rates": np.array([])
                    }

        return kinetics_metrics

    # Classification helper methods
    def _classify_correlation(self, correlation):
        abs_corr = abs(correlation)
        if abs_corr > 0.8: return "Excellent coordination"
        elif abs_corr > 0.6: return "Good coordination"
        elif abs_corr > 0.4: return "Moderate coordination"
        else: return "Poor coordination"

    def _classify_stability(self, cv):
        if cv < 15: return "Highly stable"
        elif cv < 25: return "Moderately stable"
        elif cv < 35: return "Somewhat unstable"
        else: return "Highly unstable"

    def _classify_responsiveness(self, responsiveness):
        if responsiveness > 2.0: return "Highly responsive"
        elif responsiveness > 1.0: return "Moderately responsive"
        elif responsiveness > 0.5: return "Mildly responsive"
        else: return "Poorly responsive"

    def _classify_dynamic_range(self, range_val):
        if range_val > 3.0: return "Excellent range"
        elif range_val > 2.0: return "Good range"
        elif range_val > 1.5: return "Moderate range"
        else: return "Limited range"

    def _classify_coherence(self, coherence):
        if coherence > 0.85: return "Highly coherent"
        elif coherence > 0.70: return "Moderately coherent"
        elif coherence > 0.55: return "Somewhat coherent"
        else: return "Poorly coherent"

    def _classify_suppression(self, correlation):
        if correlation < -0.6: return "Strong suppression"
        elif correlation < -0.3: return "Moderate suppression"
        elif correlation < 0: return "Weak suppression"
        else: return "No/inappropriate suppression"

    def _classify_precision(self, cv):
        if cv < 20: return "High precision"
        elif cv < 30: return "Moderate precision"
        elif cv < 40: return "Low precision"
        else: return "Very low precision"

    def _classify_counter_regulation(self, range_val):
        if range_val > 2.5: return "Strong capacity"
        elif range_val > 1.8: return "Moderate capacity"
        elif range_val > 1.3: return "Weak capacity"
        else: return "Very weak capacity"

    def _classify_reciprocity(self, correlation):
        if correlation < -0.5: return "Strong reciprocity"
        elif correlation < -0.2: return "Moderate reciprocity"
        elif correlation < 0.1: return "Weak reciprocity"
        else: return "No/disrupted reciprocity"

    def _calculate_overall_function_score(self, beta_metrics, alpha_metrics):
        """Calculate comprehensive pancreatic function score (0-100)"""
        score = 0
        max_score = 0

        # Beta-cell contributions (60% of total)
        if "insulin_cpeptide_correlation" in beta_metrics:
            score += max(0, beta_metrics["insulin_cpeptide_correlation"] * 20)
            max_score += 20

        if "secretory_stability_cv" in beta_metrics:
            stability_score = max(0, 20 - beta_metrics["secretory_stability_cv"] * 0.5)
            score += stability_score
            max_score += 20

        if "glucose_responsiveness" in beta_metrics:
            resp_score = min(20, beta_metrics["glucose_responsiveness"] * 10)
            score += resp_score
            max_score += 20

        # Alpha-cell contributions (40% of total)
        if "glucose_suppression_correlation" in alpha_metrics:
            supp_score = max(0, abs(alpha_metrics["glucose_suppression_correlation"]) * 20)
            score += supp_score
            max_score += 20

        if "insulin_glucagon_reciprocity" in alpha_metrics:
            recip_score = max(0, abs(alpha_metrics["insulin_glucagon_reciprocity"]) * 20)
            score += recip_score
            max_score += 20

        return float((score / max_score * 100)) if max_score > 0 else 0.0

    def _classify_overall_function(self, score):
        """Classify overall pancreatic function"""
        if score > 80: return "Excellent function"
        elif score > 65: return "Good function"
        elif score > 50: return "Moderate function"
        elif score > 35: return "Impaired function"
        else: return "Severely impaired function"

    def analyze_all_animals(self):
        """Calculate comprehensive metrics for all animals"""
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv_data() first.")

        print("\n" + "="*80)
        print("CALCULATING COMPREHENSIVE METRICS FOR ALL ANIMALS")
        print("="*80)
        print("Calculating 120+ metrics per animal including:")
        print("- Basic device measurements (CV%, range, etc.)")
        print("- Glucose-adjusted metrics (efficiency, corrected values)")
        print("- Peak and dynamics analysis")
        print("- Rate of change analysis")
        print("- Area under curve metrics")
        print("- Oscillation and pattern metrics")
        print("- Inter-biomarker correlations")
        print("- ELISA-validated quantitative metrics")
        print("- Pancreatic cell functionality assessment")
        print("- Peak kinetics analysis")
        print()

        for animal_id in self.data["animal_id"]:
            try:
                metrics = self.calculate_comprehensive_metrics(animal_id)
                print(f"Calculated comprehensive metrics for {animal_id}")
            except Exception as e:
                print(f"Error analyzing {animal_id}: {e}")

        print(f"\nSuccessfully analyzed {len(self.comprehensive_metrics)} animals")
        print(f"Generated {self._count_total_metrics()} individual metrics per animal")

        # Calculate group statistics
        self._calculate_group_statistics()

        return self.comprehensive_metrics

    def _count_total_metrics(self):
        """Count total number of individual metrics"""
        if not self.comprehensive_metrics:
            return 0

        # Get one animal's metrics to count
        sample_animal = list(self.comprehensive_metrics.keys())[0]
        sample_metrics = self.comprehensive_metrics[sample_animal]["metrics"]

        total_count = 0
        for category in sample_metrics:
            if sample_metrics[category] is not None:
                if isinstance(sample_metrics[category], dict):
                    total_count += self._count_dict_metrics(sample_metrics[category])

        return total_count

    def _count_dict_metrics(self, metrics_dict):
        """Recursively count metrics in nested dictionary"""
        count = 0
        for key, value in metrics_dict.items():
            if isinstance(value, dict):
                count += self._count_dict_metrics(value)
            elif isinstance(value, (int, float, bool)):
                count += 1
            elif isinstance(value, np.ndarray) and value.size == 1:
                count += 1
        return count

    def _calculate_group_statistics(self):
        """Calculate group-level statistics"""
        print("\nCalculating group statistics...")

        # Group animals by rat type
        grouped_animals = {}
        for animal_id, data in self.comprehensive_metrics.items():
            rat_type = data["raw_data"]["condition"]
            if rat_type not in grouped_animals:
                grouped_animals[rat_type] = []
            grouped_animals[rat_type].append(animal_id)

        print(f"Grouped animals: {dict([(k, len(v)) for k, v in grouped_animals.items()])}")

        self.group_metrics = grouped_animals
        return grouped_animals

    def export_metrics_to_csv(self, csv_path):
        """Export all comprehensive metrics to a CSV file"""

        if not self.comprehensive_metrics:
            print("No metrics calculated. Run analyze_all_animals() first.")
            return None

        print(f"\nExporting comprehensive metrics to CSV...")

        # Collect all metrics for all animals
        export_data = []

        for animal_id, animal_data in self.comprehensive_metrics.items():
            row = {}

            # Basic animal information
            row["animal_id"] = animal_id
            row["rat_type"] = animal_data["raw_data"]["condition"]
            row["glucose_start_mM"] = animal_data["raw_data"]["glucose_start"]
            row["glucose_end_mM"] = animal_data["raw_data"]["glucose_end"]
            row["glucose_avg_mM"] = animal_data["glucose_avg"]
            row["glucose_change_mM"] = animal_data["glucose_change"]
            row["has_elisa"] = animal_data["has_elisa"]

            metrics = animal_data["metrics"]

            # 1. Basic Device Measurements
            for biomarker in self.biomarkers:
                if biomarker in metrics["basic_device"]:
                    bm_metrics = metrics["basic_device"][biomarker]
                    prefix = f"{biomarker}_basic_"

                    row[f"{prefix}mean_raw_au"] = bm_metrics["mean_raw_au"]
                    row[f"{prefix}cv_percent"] = bm_metrics["cv_raw_percent"]
                    row[f"{prefix}max_raw_au"] = bm_metrics["max_raw_au"]
                    row[f"{prefix}min_raw_au"] = bm_metrics["min_raw_au"]
                    row[f"{prefix}range_raw_au"] = bm_metrics["range_raw_au"]

            # 2. Glucose-Adjusted Metrics
            for biomarker in self.biomarkers:
                if biomarker in metrics["glucose_adjusted"]:
                    bm_metrics = metrics["glucose_adjusted"][biomarker]
                    prefix = f"{biomarker}_glucose_"

                    row[f"{prefix}efficiency"] = bm_metrics["glucose_efficiency"]
                    row[f"{prefix}peak_efficiency"] = bm_metrics["peak_glucose_efficiency"]
                    row[f"{prefix}corrected_5p6"] = bm_metrics["mean_glucose_corrected_5p6"]
                    row[f"{prefix}sensitivity"] = bm_metrics["glucose_sensitivity"]
                    row[f"{prefix}responsiveness"] = bm_metrics["glucose_responsiveness"]
                    row[f"{prefix}correlation"] = bm_metrics["glucose_correlation"]["correlation"]
                    row[f"{prefix}correlation_pvalue"] = bm_metrics["glucose_correlation"]["p_value"]

            # 3. Peak Dynamics
            for biomarker in self.biomarkers:
                if biomarker in metrics["peak_dynamics"]:
                    bm_metrics = metrics["peak_dynamics"][biomarker]
                    prefix = f"{biomarker}_peak_"

                    row[f"{prefix}count_above_std"] = bm_metrics["num_peaks_above_std"]
                    row[f"{prefix}frequency"] = bm_metrics["peak_frequency"]
                    row[f"{prefix}max_prominence"] = bm_metrics["max_peak_prominence"]
                    row[f"{prefix}mean_width"] = bm_metrics["mean_peak_width"]
                    row[f"{prefix}dynamic_range_ratio"] = bm_metrics["dynamic_range_ratio"]
                    row[f"{prefix}total_excursion"] = bm_metrics["total_excursion"]
                    row[f"{prefix}coefficient_variation"] = bm_metrics["coefficient_of_variation"]

            # 4. Rate Analysis
            for biomarker in self.biomarkers:
                if biomarker in metrics["rate_analysis"]:
                    bm_metrics = metrics["rate_analysis"][biomarker]
                    prefix = f"{biomarker}_rate_"

                    row[f"{prefix}max_increase"] = bm_metrics["max_increase_rate"]
                    row[f"{prefix}max_decrease"] = bm_metrics["max_decrease_rate"]
                    row[f"{prefix}mean_absolute"] = bm_metrics["mean_absolute_rate"]
                    row[f"{prefix}variability"] = bm_metrics["rate_variability"]
                    row[f"{prefix}max_acceleration"] = bm_metrics["max_acceleration"]
                    row[f"{prefix}trend_slope"] = bm_metrics["trend_slope"]["slope"]
                    row[f"{prefix}trend_r_squared"] = bm_metrics["trend_slope"]["r_squared"]

            # 5. AUC Metrics
            for biomarker in self.biomarkers:
                if biomarker in metrics["auc_metrics"]:
                    bm_metrics = metrics["auc_metrics"][biomarker]
                    prefix = f"{biomarker}_auc_"

                    row[f"{prefix}total"] = bm_metrics["auc_total"]
                    row[f"{prefix}above_baseline"] = bm_metrics["auc_above_baseline"]
                    row[f"{prefix}glucose_corrected"] = bm_metrics["auc_glucose_corrected_5p6"]
                    row[f"{prefix}glucose_efficiency"] = bm_metrics["auc_glucose_efficiency"]
                    row[f"{prefix}time_weighted_avg"] = bm_metrics["time_weighted_average"]

            # 6. Oscillation Patterns
            for biomarker in self.biomarkers:
                if biomarker in metrics["oscillation_patterns"]:
                    bm_metrics = metrics["oscillation_patterns"][biomarker]
                    prefix = f"{biomarker}_osc_"

                    row[f"{prefix}frequency"] = bm_metrics["oscillation_frequency"]
                    row[f"{prefix}zero_crossings"] = bm_metrics["zero_crossings"]
                    row[f"{prefix}pattern_regularity"] = bm_metrics["pattern_regularity"]
                    row[f"{prefix}rhythmicity_index"] = bm_metrics["rhythmicity_index"]
                    row[f"{prefix}signal_entropy"] = bm_metrics["signal_entropy"]
                    row[f"{prefix}temporal_stability"] = bm_metrics["temporal_stability"]

            # 7. Inter-biomarker Correlations
            correlations = metrics["correlations"]["biomarker_correlations"]
            for pair, corr_data in correlations.items():
                row[f"corr_{pair}"] = corr_data["correlation"]
                row[f"corr_{pair}_pvalue"] = corr_data["p_value"]
                row[f"corr_{pair}_significant"] = corr_data["significant"]
                row[f"corr_{pair}_partial"] = corr_data["partial_correlation"]
                row[f"corr_{pair}_max_cross"] = corr_data["max_cross_correlation"]
                row[f"corr_{pair}_optimal_lag"] = corr_data["optimal_lag"]

            # 8. ELISA Quantitative Metrics (if available)
            if metrics["elisa_quantitative"] is not None:
                for biomarker in self.biomarkers:
                    if biomarker in metrics["elisa_quantitative"]:
                        bm_metrics = metrics["elisa_quantitative"][biomarker]
                        prefix = f"{biomarker}_elisa_"

                        row[f"{prefix}mean_concentration_pM"] = bm_metrics["mean_concentration_pM"]
                        row[f"{prefix}cv_concentration_percent"] = bm_metrics["cv_concentration_percent"]
                        row[f"{prefix}max_concentration_pM"] = bm_metrics["max_concentration_pM"]
                        row[f"{prefix}concentration_range_pM"] = bm_metrics["concentration_range_pM"]
                        row[f"{prefix}max_secretion_rate"] = bm_metrics["max_secretion_rate_pM_per_min"]
                        row[f"{prefix}calibration_slope"] = bm_metrics["calibration_slope"]
                        row[f"{prefix}calibration_error_percent"] = bm_metrics["calibration_error_percent"]

            # 9. Pancreatic Function Metrics
            func_metrics = metrics["pancreatic_function"]

            # Beta-cell function
            if "beta_cell_function" in func_metrics:
                beta_metrics = func_metrics["beta_cell_function"]
                for key, value in beta_metrics.items():
                    if isinstance(value, (int, float, bool)):
                        row[f"beta_cell_{key}"] = value

            # Alpha-cell function
            if "alpha_cell_function" in func_metrics:
                alpha_metrics = func_metrics["alpha_cell_function"]
                for key, value in alpha_metrics.items():
                    if isinstance(value, (int, float, bool)):
                        row[f"alpha_cell_{key}"] = value

            # Overall function
            row["overall_function_score"] = func_metrics["overall_function_score"]
            row["function_classification"] = func_metrics["function_classification"]

            # 10. Peak Kinetics
            for biomarker in self.biomarkers:
                if biomarker in metrics["peak_kinetics"]:
                    bm_metrics = metrics["peak_kinetics"][biomarker]
                    prefix = f"{biomarker}_kinetics_"

                    row[f"{prefix}peak_count"] = bm_metrics["peak_count"]
                    row[f"{prefix}peak_frequency"] = bm_metrics["peak_frequency"]
                    row[f"{prefix}mean_rise_time"] = bm_metrics["mean_rise_time"]
                    row[f"{prefix}mean_decline_time"] = bm_metrics["mean_decline_time"]
                    row[f"{prefix}mean_rise_rate"] = bm_metrics["mean_rise_rate"]
                    row[f"{prefix}mean_decline_rate"] = bm_metrics["mean_decline_rate"]
                    row[f"{prefix}rise_decline_ratio"] = bm_metrics["rise_decline_ratio"]

            export_data.append(row)

        # Create DataFrame and save to CSV
        df = pd.DataFrame(export_data)

        # Reorder columns to put basic info first
        basic_cols = ["animal_id", "rat_type", "glucose_start_mM", "glucose_end_mM",
                      "glucose_avg_mM", "glucose_change_mM", "has_elisa", "overall_function_score",
                      "function_classification"]

        other_cols = [col for col in df.columns if col not in basic_cols]
        df = df[basic_cols + sorted(other_cols)]

        # Save to CSV
        df.to_csv(csv_path, index=False)

        print(f"Exported {len(df)} animals with {len(df.columns)} metrics to: {csv_path}")
        print(f"Columns include:")
        print(f" - Basic info: {len(basic_cols)} columns")
        print(f" - Device measurements: {len([c for c in df.columns if '_basic_' in c])} columns")
        print(f" - Glucose-adjusted: {len([c for c in df.columns if '_glucose_' in c])} columns")
        print(f" - Peak dynamics: {len([c for c in df.columns if '_peak_' in c])} columns")
        print(f" - Rate analysis: {len([c for c in df.columns if '_rate_' in c])} columns")
        print(f" - AUC metrics: {len([c for c in df.columns if '_auc_' in c])} columns")
        print(f" - Oscillations: {len([c for c in df.columns if '_osc_' in c])} columns")
        print(f" - Correlations: {len([c for c in df.columns if 'corr_' in c])} columns")
        print(f" - ELISA validation: {len([c for c in df.columns if '_elisa_' in c])} columns")
        print(f" - Function scores: {len([c for c in df.columns if any(x in c for x in ['beta_cell_', 'alpha_cell_', 'function'])])} columns")
        print(f" - Peak kinetics: {len([c for c in df.columns if '_kinetics_' in c])} columns")

        return df
