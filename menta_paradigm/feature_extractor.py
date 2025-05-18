import os
import pandas as pd
import numpy as np
import mne
import time
import argparse
from tqdm import tqdm
from scipy.stats import skew, kurtosis, entropy
from mne.time_frequency import psd_array_welch
import math
from pywt import wavedec


# ---------------------------
# EEG Feature Extractor Class
# ---------------------------
class EEGFeatureExtractor:
    def __init__(self, input_csv, sfreq=250.0, channel_column='Timestamp'):
        """
        Initializes the extractor.

        Args:
            input_csv (str): Path to the preprocessed CSV file.
            sfreq (float): Sampling frequency (Hz). Default is 250 Hz.
            channel_column (str): Name of the timestamp column.
        """
        self.input_csv = input_csv
        self.sfreq = sfreq
        self.channel_column = channel_column
        self.df = None  # Holds the CSV data as a DataFrame.
        self.raw = None  # MNE RawArray created from the DataFrame.
        self.feature_funcs = {}  # Dictionary mapping feature names to functions.

    def load_data(self):
        """
        Loads the CSV file and converts the EEG data into an MNE RawArray.
        Assumes the CSV has a "Timestamp" column and the remaining columns are EEG channels.
        """
        try:
            self.df = pd.read_csv(self.input_csv)

            # Verify the channel column exists
            if self.channel_column not in self.df.columns:
                print(f"Warning: '{self.channel_column}' column not found in {self.input_csv}. "
                      f"Available columns: {list(self.df.columns)}")
                return None

            eeg_data = self.df.drop(columns=[self.channel_column]).values.T  # shape: (n_channels, n_times)
            channel_names = list(self.df.columns)
            channel_names.remove(self.channel_column)

            # Verify we have channel data
            if len(channel_names) == 0:
                print(f"Error: No EEG channels found in {self.input_csv} after removing '{self.channel_column}'")
                return None

            ch_types = ["eeg"] * len(channel_names)
            info = mne.create_info(ch_names=channel_names, sfreq=self.sfreq, ch_types=ch_types)
            self.raw = mne.io.RawArray(eeg_data, info)
            return self.raw
        except Exception as e:
            print(f"Error loading data from {self.input_csv}: {str(e)}")
            return None

    def register_feature(self, name, func):
        """
        Registers a feature function.

        Args:
            name (str): Abbreviation/name of the feature.
            func (callable): A function with signature func(signal, sfreq) that returns a scalar.
        """
        self.feature_funcs[name] = func

    def compute_features(self):
        """
        Computes all registered features for each EEG channel.

        Returns:
            pd.DataFrame: A DataFrame with channel names as rows and computed features as columns.
        """
        if self.df is None or self.raw is None:
            if self.load_data() is None:
                return None

        ch_names = self.raw.info['ch_names']
        features = {}

        for ch in ch_names:
            signal = self.raw.get_data(picks=ch)[0]  # Get 1D array for channel ch.
            feats = {}
            for feat_name, func in self.feature_funcs.items():
                try:
                    feats[feat_name] = func(signal, self.sfreq)
                except Exception as e:
                    print(f"Error computing {feat_name} for channel {ch}: {e}")
                    feats[feat_name] = np.nan
            features[ch] = feats

        feature_df = pd.DataFrame.from_dict(features, orient='index')
        feature_df.index.name = 'Channel'
        return feature_df

    def save_features(self, output_csv):
        """
        Saves the computed features DataFrame to a CSV file.

        Args:
            output_csv (str): Path to the output CSV file.
        """
        feature_df = self.compute_features()
        if feature_df is None:
            return False

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

        feature_df.to_csv(output_csv)
        print(f"Features saved to {output_csv}")
        return True


# ---------------------------
# Feature Functions
# ---------------------------
# --- Basic Statistical Features ---
def feat_mean(signal, sfreq=None):
    return np.mean(signal)


def feat_median(signal, sfreq=None):
    return np.median(signal)


def feat_variance(signal, sfreq=None):
    return np.var(signal)


def feat_std(signal, sfreq=None):
    return np.std(signal)


def feat_skewness(signal, sfreq=None):
    return skew(signal)


def feat_kurtosis(signal, sfreq=None):
    return kurtosis(signal)


def feat_zero_crossing_rate(signal, sfreq=None):
    return np.sum(np.diff(np.sign(signal)) != 0) / len(signal)


def feat_energy(signal, sfreq=None):
    return np.sum(signal ** 2) / len(signal)


def feat_peak_to_peak(signal, sfreq=None):
    return np.max(signal) - np.min(signal)


# --- Frequency-Domain Features ---
def feat_psd_delta(signal, sfreq):
    psd, freqs = psd_array_welch(np.atleast_2d(signal), sfreq=sfreq, n_fft=256, verbose=False)
    idx = np.logical_and(freqs >= 1, freqs <= 4)
    return np.mean(psd[0, idx])


def feat_psd_theta(signal, sfreq):
    psd, freqs = psd_array_welch(np.atleast_2d(signal), sfreq=sfreq, n_fft=256, verbose=False)
    idx = np.logical_and(freqs >= 4, freqs <= 8)
    return np.mean(psd[0, idx])


def feat_psd_alpha(signal, sfreq):
    psd, freqs = psd_array_welch(np.atleast_2d(signal), sfreq=sfreq, n_fft=256, verbose=False)
    idx = np.logical_and(freqs >= 8, freqs <= 12)
    return np.mean(psd[0, idx])


def feat_psd_beta(signal, sfreq):
    psd, freqs = psd_array_welch(np.atleast_2d(signal), sfreq=sfreq, n_fft=256, verbose=False)
    idx = np.logical_and(freqs >= 12, freqs <= 30)
    return np.mean(psd[0, idx])


def feat_total_power(signal, sfreq):
    psd, freqs = psd_array_welch(np.atleast_2d(signal), sfreq=sfreq, n_fft=256, verbose=False)
    idx = np.logical_and(freqs >= 1, freqs <= 30)
    return np.sum(psd[0, idx])


def feat_relative_alpha(signal, sfreq):
    alpha = feat_psd_alpha(signal, sfreq)
    total = feat_total_power(signal, sfreq)
    return alpha / total if total != 0 else np.nan


def feat_ratio_theta_alpha(signal, sfreq):
    theta = feat_psd_theta(signal, sfreq)
    alpha = feat_psd_alpha(signal, sfreq)
    return theta / alpha if alpha != 0 else np.nan


def feat_ratio_beta_alpha(signal, sfreq):
    beta = feat_psd_beta(signal, sfreq)
    alpha = feat_psd_alpha(signal, sfreq)
    return beta / alpha if alpha != 0 else np.nan


def feat_spectral_edge(signal, sfreq, edge=0.95):
    psd, freqs = psd_array_welch(np.atleast_2d(signal), sfreq=sfreq, n_fft=256, verbose=False)
    psd = psd[0]
    total_power = np.sum(psd)
    cumulative_power = np.cumsum(psd)
    try:
        sef_idx = np.where(cumulative_power >= edge * total_power)[0][0]
        return freqs[sef_idx]
    except IndexError:
        return np.nan


# --- Nonlinear Features ---
# ---------- Genuine Sample Entropy ----------
def feat_sample_entropy(signal, sfreq=None, m=2, r_ratio=0.2):
    """
    Sample Entropy (Richman & Moorman, 2000).

    Parameters
    ----------
    signal : 1-D numpy array
    m      : embedding dimension (default 2)
    r_ratio: tolerance as a fraction of the signal's SD (default 0.20)

    Returns
    -------
    SampEn value or np.nan on error.
    """
    try:
        import nolds                      # lightweight dependency
        r = r_ratio * np.std(signal)
        return nolds.sampen(signal, emb_dim=m, tolerance=r)
    except Exception:
        # nolds missing or sampen failed (too short, all-zeros, etc.)
        raise ModuleNotFoundError



def feat_permutation_entropy(signal, sfreq, order=3):
    n = len(signal)
    if n < order:
        return np.nan
    patterns = []
    for i in range(n - order + 1):
        pattern = tuple(np.argsort(signal[i:i + order]))
        patterns.append(pattern)
    pattern_counts = {}
    for pattern in patterns:
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    probs = np.array(list(pattern_counts.values())) / float(len(patterns))
    pe = -np.sum(probs * np.log2(probs))
    norm_pe = pe / np.log2(math.factorial(order))
    return norm_pe


def feat_higuchi_fd(signal, sfreq, kmax=10):
    N = len(signal)
    Lk = []
    for k in range(1, kmax + 1):
        Lk_temp = []
        for m in range(k):
            Lm = 0
            n_max = int(np.floor((N - m) / k))
            if n_max < 2:
                continue
            for i in range(1, n_max):
                Lm += abs(signal[m + i * k] - signal[m + (i - 1) * k])
            Lm = (Lm * (N - 1)) / (k * n_max * k)
            Lk_temp.append(Lm)
        if Lk_temp:
            Lk.append(np.mean(Lk_temp))
    if not Lk:
        return np.nan
    lnLk = np.log(Lk)
    lnk = np.log(np.arange(1, len(Lk) + 1))
    slope, _ = np.polyfit(lnk, lnLk, 1)
    return -slope


# --- Additional Features for Visual Imagery Classification ---
# Hjorth Parameters
def feat_hjorth_activity(signal, sfreq=None):
    return np.var(signal)


def feat_hjorth_mobility(signal, sfreq=None):
    dx = np.diff(signal)
    var_signal = np.var(signal)
    var_dx = np.var(dx)
    return np.sqrt(var_dx / var_signal) if var_signal != 0 else np.nan


def feat_hjorth_complexity(signal, sfreq=None):
    dx = np.diff(signal)
    ddx = np.diff(dx)
    mob = feat_hjorth_mobility(signal)
    mob_dx = np.sqrt(np.var(ddx) / np.var(dx)) if np.var(dx) != 0 else np.nan
    return mob_dx / mob if mob != 0 else np.nan


# Spectral Entropy (of the normalized PSD)
def feat_spectral_entropy(signal, sfreq):
    psd, freqs = psd_array_welch(np.atleast_2d(signal), sfreq=sfreq, n_fft=256, verbose=False)
    psd = psd[0]
    psd_norm = psd / np.sum(psd) if np.sum(psd) != 0 else np.zeros_like(psd)
    return entropy(psd_norm, base=2)


# Lempel-Ziv Complexity
def feat_lempel_ziv(signal, sfreq=None):
    median_val = np.median(signal)
    binary_seq = ''.join('1' if x > median_val else '0' for x in signal)
    i, c, l = 0, 1, 1
    n = len(binary_seq)
    while i + l <= n:
        sub_str = binary_seq[i:i + l]
        if sub_str not in binary_seq[:i]:
            c += 1
            i += l
            l = 1
        else:
            l += 1
            if i + l > n:
                i += 1
                l = 1
    return c


# --- Additional Features for Visual Imagery Classification ---

# Phase-Based Features
def feat_self_phase_locking_value(signal, sfreq, band=(8, 13)):
    """
    Calculate the phase locking value in the alpha band between consecutive windows.

    Alpha band phase synchrony is often related to visual processing.
    """
    from scipy.signal import hilbert

    # Band-pass filter the signal
    filtered_signal = mne.filter.filter_data(
        signal.reshape(1, -1),
        sfreq=sfreq,
        l_freq=band[0],
        h_freq=band[1],
        verbose=False
    )[0]

    # Get the analytic signal (complex)
    analytic_signal = hilbert(filtered_signal)

    # Extract instantaneous phase
    inst_phase = np.angle(analytic_signal)

    # Split the signal into windows
    win_size = int(sfreq)  # 1-second windows
    n_windows = len(inst_phase) // win_size

    if n_windows < 2:
        return np.nan

    # Calculate phase difference between consecutive windows
    plv_values = []
    for i in range(n_windows - 1):
        phase1 = inst_phase[i * win_size:(i + 1) * win_size]
        phase2 = inst_phase[(i + 1) * win_size:(i + 2) * win_size]
        min_len = min(len(phase1), len(phase2))

        # Calculate PLV
        complex_phase_diff = np.exp(1j * (phase1[:min_len] - phase2[:min_len]))
        plv = np.abs(np.mean(complex_phase_diff))
        plv_values.append(plv)

    return np.mean(plv_values) if plv_values else np.nan


def feat_alpha_peak_frequency(signal, sfreq):
    """
    Find the peak frequency within the alpha band (8-13 Hz).

    Alpha peak frequency can vary during visual imagery tasks.
    """
    psd, freqs = psd_array_welch(np.atleast_2d(signal), sfreq=sfreq, n_fft=512, verbose=False)
    psd = psd[0]

    # Find alpha band indices
    alpha_idx = np.logical_and(freqs >= 8, freqs <= 13)
    alpha_freqs = freqs[alpha_idx]
    alpha_psd = psd[alpha_idx]

    if len(alpha_psd) == 0 or np.sum(alpha_psd) == 0:
        return np.nan

    # Find the frequency with maximum power in alpha band
    peak_idx = np.argmax(alpha_psd)
    return alpha_freqs[peak_idx] if peak_idx < len(alpha_freqs) else np.nan


def feat_individual_alpha_power(signal, sfreq, width=1.0):
    """
    Calculate power in a narrow band around the individual alpha peak frequency.

    Individual alpha frequency bands can be more informative than fixed bands.
    """
    alpha_peak = feat_alpha_peak_frequency(signal, sfreq)

    if np.isnan(alpha_peak):
        return np.nan

    psd, freqs = psd_array_welch(np.atleast_2d(signal), sfreq=sfreq, n_fft=512, verbose=False)
    psd = psd[0]

    # Calculate power in narrow band around alpha peak
    band_idx = np.logical_and(freqs >= alpha_peak - width, freqs <= alpha_peak + width)
    return np.mean(psd[band_idx]) if np.any(band_idx) else np.nan


# Anterior-Posterior Alpha Asymmetry (if multiple channels available)
def feat_alpha_asymmetry(signal_pairs, sfreq):
    """
    Calculate alpha asymmetry between channel pairs.
    This should be calculated at the dataset level, not within this framework.
    """
    pass  # This is just a placeholder to note this potential feature


# Time-Frequency Features
def feat_wavelet_complexity(signal, sfreq):
    """
    Calculate wavelet-based signal complexity.

    Complexity in time-frequency domain can help distinguish visual imagery patterns.
    """
    # Perform wavelet decomposition
    coeffs = wavedec(signal, 'db4', level=5)

    # Calculate entropy of wavelet coefficients
    entropy_values = []
    for coef in coeffs:
        if len(coef) > 1:
            # Normalize coefficients
            coef_norm = np.abs(coef) / np.sum(np.abs(coef)) if np.sum(np.abs(coef)) > 0 else np.zeros_like(coef)
            # Calculate entropy
            ent = entropy(coef_norm + 1e-10, base=2)
            entropy_values.append(ent)

    return np.mean(entropy_values) if entropy_values else np.nan


# Functional Connectivity Measures
def feat_self_signal_coherence(signal, sfreq, band=(8, 13)):
    """
    Calculate mean coherence across signal segments.

    Coherence measures can help identify connectivity patterns relevant for visual imagery.
    """
    from scipy.signal import coherence

    # Split signal into segments
    win_size = int(2 * sfreq)  # 2-second windows
    hop_size = int(sfreq)  # 1-second hop

    n_segments = (len(signal) - win_size) // hop_size + 1
    if n_segments < 2:
        return np.nan

    coherence_values = []
    for i in range(n_segments - 1):
        seg1 = signal[i * hop_size:i * hop_size + win_size]
        seg2 = signal[(i + 1) * hop_size:(i + 1) * hop_size + win_size]

        # Calculate coherence
        f, coh = coherence(seg1, seg2, fs=sfreq, nperseg=min(256, win_size // 2))

        # Find indices for the frequency band of interest
        band_idx = np.logical_and(f >= band[0], f <= band[1])

        # Average coherence in the band
        band_coh = np.mean(coh[band_idx]) if np.any(band_idx) else np.nan
        coherence_values.append(band_coh)

    return np.nanmean(coherence_values) if coherence_values else np.nan


# Dynamic Connectivity
def feat_self_coherence_variance(signal, sfreq, band=(8, 13)):
    """
    Calculate the variance of self coherence

    This measures how stable the coherence patterns are, which may differ during visual imagery.
    """
    from scipy.signal import coherence

    # Split signal into segments
    win_size = int(2 * sfreq)  # 2-second windows
    hop_size = int(sfreq // 2)  # 0.5-second hop for more granular analysis

    n_segments = (len(signal) - win_size) // hop_size + 1
    if n_segments < 3:  # Need at least 3 segments for meaningful variance
        return np.nan

    # Calculate coherence for each pair of adjacent segments
    coherence_values = []
    for i in range(n_segments - 1):
        seg1 = signal[i * hop_size:i * hop_size + win_size]
        seg2 = signal[(i + 1) * hop_size:(i + 1) * hop_size + win_size]

        f, coh = coherence(seg1, seg2, fs=sfreq, nperseg=min(256, win_size // 2))

        band_idx = np.logical_and(f >= band[0], f <= band[1])
        band_coh = np.mean(coh[band_idx]) if np.any(band_idx) else np.nan
        coherence_values.append(band_coh)

    return np.nanvar(coherence_values) if len(coherence_values) > 1 else np.nan


# Visualization-specific frequency bands
def feat_psd_gamma_low(signal, sfreq):
    """
    Calculate power in the low gamma band (30-45 Hz).

    Gamma activity is associated with visual processing and attention.
    """
    psd, freqs = psd_array_welch(np.atleast_2d(signal), sfreq=sfreq, n_fft=512, verbose=False)
    idx = np.logical_and(freqs >= 30, freqs <= 45)
    return np.mean(psd[0, idx]) if np.any(idx) else np.nan


def feat_psd_gamma_high(signal, sfreq):
    """
    Calculate power in the high gamma band (55-80 Hz).

    High gamma activity may reflect specific aspects of visual processing.
    """
    psd, freqs = psd_array_welch(np.atleast_2d(signal), sfreq=sfreq, n_fft=512, verbose=False)
    idx = np.logical_and(freqs >= 55, freqs <= 80)
    return np.mean(psd[0, idx]) if np.any(idx) else np.nan


# Detrended Fluctuation Analysis - Long-range correlations
def feat_dfa(signal, sfreq):
    """
    Perform Detrended Fluctuation Analysis.

    DFA measures long-range correlations in time series, which may differ during different imagery tasks.
    """
    # Simple implementation of DFA
    # For production, consider using nolds or another specialized library

    # Remove mean
    signal = signal - np.mean(signal)

    # Calculate cumulative sum
    y = np.cumsum(signal)

    # Define box sizes
    n_min = 10
    n_max = len(signal) // 4
    if n_max <= n_min:
        return np.nan

    # Use logarithmically spaced box sizes
    n_boxes = 10
    box_sizes = np.unique(np.logspace(np.log10(n_min), np.log10(n_max), n_boxes).astype(int))

    fluctuations = []
    for box_size in box_sizes:
        # Number of boxes
        n_boxes = len(signal) // box_size

        if n_boxes == 0:
            continue

        # Truncate the signal to fit into boxes
        y_trunc = y[:n_boxes * box_size]

        # Reshape signal into boxes
        y_reshaped = y_trunc.reshape((n_boxes, box_size))

        # Calculate local trends using polynomial fit
        x = np.arange(box_size)
        trends = np.array([np.polyval(np.polyfit(x, y_box, 1), x) for y_box in y_reshaped])

        # Calculate fluctuation as root mean square deviation
        f = np.sqrt(np.mean((y_reshaped - trends) ** 2, axis=1))
        fluctuations.append(np.mean(f))

    if len(box_sizes) < 4 or len(fluctuations) < 4:
        return np.nan

    # Fit line to log-log plot and get slope (alpha)
    poly = np.polyfit(np.log(box_sizes[:len(fluctuations)]), np.log(fluctuations), 1)
    return poly[0]  # Return alpha (the scaling exponent)


# Time-Frequency Energy Distribution
def feat_energy_ratio(signal, sfreq):
    """
    Calculate the ratio of energy in the latter half to the first half of the signal.

    This may help identify changes in mental imagery over time.
    """
    half_len = len(signal) // 2
    energy_first = np.sum(signal[:half_len] ** 2)
    energy_second = np.sum(signal[half_len:] ** 2)

    return energy_second / energy_first if energy_first > 0 else np.nan


# EEG microstate features (simplified version)
def feat_proxy_microstate_variance(signal, sfreq):
    """
    Calculate a simplified measure of EEG microstate variability.

    True microstate analysis requires spatial information across multiple channels.
    This is a simplified proxy using temporal segmentation.
    """
    # Filter signal to alpha band which is relevant for visual processing
    filtered_signal = mne.filter.filter_data(
        signal.reshape(1, -1),
        sfreq=sfreq,
        l_freq=8,
        h_freq=13,
        verbose=False
    )[0]

    # Create a simple proxy for microstates using signal amplitude
    # Real microstate analysis would use spatial patterns across channels
    median_val = np.median(filtered_signal)
    states = (filtered_signal > median_val).astype(int)

    # Calculate mean duration of each "state"
    state_changes = np.diff(states)
    change_indices = np.where(state_changes != 0)[0]

    if len(change_indices) < 2:
        return np.nan

    durations = np.diff(np.append(0, np.append(change_indices, len(states))))

    # Return the coefficient of variation (std/mean) of durations
    mean_duration = np.mean(durations)
    std_duration = np.std(durations)

    return std_duration / mean_duration if mean_duration > 0 else np.nan


# ---------------------------
# Function to register all features
# ---------------------------
def register_all_features(extractor):
    """Register all feature extraction functions to the extractor."""

    # Register Basic Statistical Features
    extractor.register_feature('mean', feat_mean)
    extractor.register_feature('median', feat_median)
    extractor.register_feature('variance', feat_variance)
    extractor.register_feature('std', feat_std)
    extractor.register_feature('skewness', feat_skewness)
    extractor.register_feature('kurtosis', feat_kurtosis)
    extractor.register_feature('zcr', feat_zero_crossing_rate)
    extractor.register_feature('energy', feat_energy)
    extractor.register_feature('ptp', feat_peak_to_peak)

    # Register Frequency Domain Features
    extractor.register_feature('psd_delta', feat_psd_delta)
    extractor.register_feature('psd_theta', feat_psd_theta)
    extractor.register_feature('psd_alpha', feat_psd_alpha)
    extractor.register_feature('psd_beta', feat_psd_beta)
    extractor.register_feature('total_power', feat_total_power)
    extractor.register_feature('rel_alpha', feat_relative_alpha)
    extractor.register_feature('ratio_theta_alpha', feat_ratio_theta_alpha)
    extractor.register_feature('ratio_beta_alpha', feat_ratio_beta_alpha)
    extractor.register_feature('spectral_edge', feat_spectral_edge)

    # Register Nonlinear Features
    extractor.register_feature('sample_entropy', feat_sample_entropy)
    extractor.register_feature('perm_entropy', feat_permutation_entropy)
    extractor.register_feature('higuchi_fd', feat_higuchi_fd)

    # Register Additional Features for Visual Imagery
    extractor.register_feature('hjorth_activity', feat_hjorth_activity)
    extractor.register_feature('hjorth_mobility', feat_hjorth_mobility)
    extractor.register_feature('hjorth_complexity', feat_hjorth_complexity)
    extractor.register_feature('spectral_entropy', feat_spectral_entropy)
    extractor.register_feature('lempel_ziv', feat_lempel_ziv)

    # Register New Visual Imagery Specific Features
    extractor.register_feature('self_phase_locking_value', feat_self_phase_locking_value)
    extractor.register_feature('alpha_peak_freq', feat_alpha_peak_frequency)
    extractor.register_feature('ind_alpha_power', feat_individual_alpha_power)
    extractor.register_feature('wavelet_complexity', feat_wavelet_complexity)
    extractor.register_feature('self_coherence', feat_self_signal_coherence)
    extractor.register_feature('self_coherence_variance', feat_self_coherence_variance)
    # extractor.register_feature('psd_gamma_low', feat_psd_gamma_low) >30Hz /canceled out in our freq. spectrum
    # extractor.register_feature('psd_gamma_high', feat_psd_gamma_high) >30Hz /canceled out in our freq. spectrum
    extractor.register_feature('dfa', feat_dfa)
    extractor.register_feature('energy_ratio', feat_energy_ratio)
    extractor.register_feature('proxy_microstate_var', feat_proxy_microstate_variance)

# ---------------------------
# Function to process a single file
# ---------------------------
def process_file(input_file, output_file, sfreq=250.0):
    """Process a single CSV file and extract features."""

    # Skip if the output file already exists
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Skipping.")
        return True

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Initialize the feature extractor
    extractor = EEGFeatureExtractor(input_csv=input_file, sfreq=sfreq)

    # Register all features
    register_all_features(extractor)

    # Compute and save features
    return extractor.save_features(output_file)


# ---------------------------
# Function to process directory structure
# ---------------------------
def process_directory_structure(input_base_dir, output_base_dir):
    """
    Process all CSV files in the directory structure, preserving the hierarchy.
    Extract features from each file and save to the corresponding location.

    Args:
        input_base_dir (str): Base directory containing processed data folders
        output_base_dir (str): Base directory where feature files will be saved
    """
    # Count total files for progress tracking
    total_files = 0
    for root, _, files in os.walk(input_base_dir):
        total_files += sum(1 for f in files if f.endswith('.csv'))

    print(f"Found {total_files} CSV files to process")

    # Create counter for success/failure tracking
    success_count = 0
    failure_count = 0

    # Process all CSV files
    with tqdm(total=total_files, desc="Extracting features") as pbar:
        for root, dirs, files in os.walk(input_base_dir):
            # Create corresponding output directory structure
            rel_path = os.path.relpath(root, input_base_dir)
            output_dir = os.path.join(output_base_dir, rel_path)
            os.makedirs(output_dir, exist_ok=True)

            # Process each CSV file in this directory
            for file in files:
                if file.endswith('.csv'):
                    input_file = os.path.join(root, file)
                    # Create output filename - replace or add _features suffix
                    if file.startswith('processed_'):
                        output_file = os.path.join(output_dir, file.replace('processed_', 'features_'))
                    else:
                        output_file = os.path.join(output_dir, f"features_{file}")

                    # Process the file
                    result = process_file(input_file, output_file)

                    if result:
                        success_count += 1
                    else:
                        failure_count += 1

                    pbar.update(1)

    print(f"\nFeature extraction complete!")
    print(f"Successfully processed: {success_count} files")
    print(f"Failed to process: {failure_count} files")

    return success_count, failure_count


# ---------------------------
# Main Function
# ---------------------------
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract EEG features from processed data files.')
    parser.add_argument('--input_dir', type=str, default='data/processed',
                        help='Base input directory containing processed data folders')
    parser.add_argument('--output_dir', type=str, default='data/extracted_features',
                        help='Base output directory for feature files')
    parser.add_argument('--sfreq', type=float, default=250.0,
                        help='Sampling frequency in Hz (default: 250.0)')

    args = parser.parse_args()

    # Process the directory structure
    start_time = time.time()
    process_directory_structure(args.input_dir, args.output_dir)
    end_time = time.time()

    print(f"Total processing time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()