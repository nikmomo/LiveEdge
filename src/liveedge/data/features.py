"""Feature extraction functions for accelerometer data.

This module provides functions for extracting time-domain, frequency-domain,
dynamics, and periodicity features from accelerometer windows.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import signal, stats


def compute_magnitude(data: NDArray[np.float32]) -> NDArray[np.float32]:
    """Compute acceleration magnitude from 3-axis data.

    Args:
        data: Accelerometer data of shape (..., 3) or (n_samples, 3).

    Returns:
        Magnitude array of shape (...,) or (n_samples,).
    """
    return np.sqrt(np.sum(data**2, axis=-1)).astype(np.float32)


def extract_time_domain_features(window: NDArray[np.float32]) -> dict[str, float]:
    """Extract time-domain features from accelerometer window.

    Args:
        window: Accelerometer window of shape (window_size, 3).

    Returns:
        Dictionary of feature name to value.

    Features:
        - acc_magnitude_{mean, std, max, min, range}: Magnitude statistics
        - acc_{x,y,z}_{mean, std, max, min}: Per-axis statistics
        - zero_crossing_rate: Rate of zero crossings in magnitude
        - signal_magnitude_area: Sum of absolute values across axes
        - mean_crossing_rate: Rate of mean crossings
        - rms: Root mean square of magnitude
        - energy: Signal energy (sum of squares)
        - interquartile_range: IQR of magnitude
    """
    features = {}

    # Compute magnitude
    magnitude = compute_magnitude(window)

    # Magnitude statistics
    features["acc_magnitude_mean"] = float(np.mean(magnitude))
    features["acc_magnitude_std"] = float(np.std(magnitude))
    features["acc_magnitude_max"] = float(np.max(magnitude))
    features["acc_magnitude_min"] = float(np.min(magnitude))
    features["acc_magnitude_range"] = float(np.ptp(magnitude))

    # Per-axis statistics
    axis_names = ["x", "y", "z"]
    for i, axis in enumerate(axis_names):
        axis_data = window[:, i]
        features[f"acc_{axis}_mean"] = float(np.mean(axis_data))
        features[f"acc_{axis}_std"] = float(np.std(axis_data))
        features[f"acc_{axis}_max"] = float(np.max(axis_data))
        features[f"acc_{axis}_min"] = float(np.min(axis_data))

    # Zero crossing rate (on de-meaned signal)
    centered_mag = magnitude - np.mean(magnitude)
    zero_crossings = np.sum(np.abs(np.diff(np.sign(centered_mag))) > 0)
    features["zero_crossing_rate"] = float(zero_crossings / len(magnitude))

    # Signal Magnitude Area (SMA)
    features["signal_magnitude_area"] = float(np.sum(np.abs(window)))

    # Mean crossing rate
    mean_val = np.mean(magnitude)
    mean_crossings = np.sum(np.abs(np.diff(np.sign(magnitude - mean_val))) > 0)
    features["mean_crossing_rate"] = float(mean_crossings / len(magnitude))

    # RMS
    features["rms"] = float(np.sqrt(np.mean(magnitude**2)))

    # Energy
    features["energy"] = float(np.sum(magnitude**2))

    # Interquartile range
    q75, q25 = np.percentile(magnitude, [75, 25])
    features["interquartile_range"] = float(q75 - q25)

    return features


def extract_frequency_domain_features(
    window: NDArray[np.float32],
    fs: int = 50,
) -> dict[str, float]:
    """Extract frequency-domain features.

    Args:
        window: Accelerometer window of shape (window_size, 3).
        fs: Sampling frequency in Hz.

    Returns:
        Dictionary of feature name to value.

    Features:
        - dominant_frequency: Frequency with highest power
        - spectral_entropy: Entropy of power spectrum
        - spectral_centroid: Center of mass of spectrum
        - high_freq_ratio: Ratio of energy above 5Hz
        - band_power_{0_2, 2_5, 5_10, 10_25}: Power in frequency bands
        - spectral_rolloff: Frequency below which 85% of energy is contained
        - spectral_flatness: Geometric mean / arithmetic mean of spectrum
    """
    features = {}

    # Compute magnitude
    magnitude = compute_magnitude(window)

    # Apply window function
    windowed = magnitude * np.hanning(len(magnitude))

    # FFT
    n = len(windowed)
    fft_vals = np.fft.rfft(windowed)
    fft_freqs = np.fft.rfftfreq(n, d=1 / fs)
    power_spectrum = np.abs(fft_vals) ** 2

    # Avoid division by zero
    total_power = np.sum(power_spectrum)
    if total_power == 0:
        total_power = 1e-10

    # Dominant frequency
    dominant_idx = np.argmax(power_spectrum)
    features["dominant_frequency"] = float(fft_freqs[dominant_idx])

    # Spectral entropy
    normalized_spectrum = power_spectrum / total_power
    normalized_spectrum = np.clip(normalized_spectrum, 1e-10, None)
    spectral_entropy = -np.sum(normalized_spectrum * np.log2(normalized_spectrum))
    max_entropy = np.log2(len(power_spectrum))
    features["spectral_entropy"] = float(spectral_entropy / max_entropy if max_entropy > 0 else 0)

    # Spectral centroid
    features["spectral_centroid"] = float(
        np.sum(fft_freqs * power_spectrum) / total_power
    )

    # High frequency ratio (above 5Hz)
    high_freq_mask = fft_freqs > 5
    high_freq_power = np.sum(power_spectrum[high_freq_mask])
    features["high_freq_ratio"] = float(high_freq_power / total_power)

    # Band power
    bands = [(0, 2), (2, 5), (5, 10), (10, 25)]
    for low, high in bands:
        mask = (fft_freqs >= low) & (fft_freqs < high)
        band_power = np.sum(power_spectrum[mask])
        features[f"band_power_{low}_{high}"] = float(band_power / total_power)

    # Spectral rolloff (85%)
    cumulative_power = np.cumsum(power_spectrum)
    rolloff_idx = np.searchsorted(cumulative_power, 0.85 * total_power)
    rolloff_idx = min(rolloff_idx, len(fft_freqs) - 1)
    features["spectral_rolloff"] = float(fft_freqs[rolloff_idx])

    # Spectral flatness
    geometric_mean = np.exp(np.mean(np.log(power_spectrum + 1e-10)))
    arithmetic_mean = np.mean(power_spectrum)
    features["spectral_flatness"] = float(
        geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0
    )

    return features


def extract_dynamics_features(
    window: NDArray[np.float32],
    fs: int = 50,
) -> dict[str, float]:
    """Extract motion dynamics features.

    Args:
        window: Accelerometer window of shape (window_size, 3).
        fs: Sampling frequency in Hz.

    Returns:
        Dictionary of feature name to value.

    Features:
        - jerk_{mean, std, max}: Statistics of jerk (derivative of acceleration)
        - acceleration_change_rate: Rate of acceleration change
        - jerk_magnitude_{mean, std}: Statistics of jerk magnitude
    """
    features = {}

    # Compute jerk (derivative of acceleration)
    dt = 1.0 / fs
    jerk = np.diff(window, axis=0) / dt

    # Jerk magnitude
    jerk_magnitude = compute_magnitude(jerk)

    features["jerk_mean"] = float(np.mean(jerk_magnitude))
    features["jerk_std"] = float(np.std(jerk_magnitude))
    features["jerk_max"] = float(np.max(jerk_magnitude))

    # Acceleration change rate
    magnitude = compute_magnitude(window)
    acc_changes = np.abs(np.diff(magnitude))
    features["acceleration_change_rate"] = float(np.mean(acc_changes) * fs)

    # Per-axis jerk statistics
    axis_names = ["x", "y", "z"]
    for i, axis in enumerate(axis_names):
        features[f"jerk_{axis}_mean"] = float(np.mean(np.abs(jerk[:, i])))
        features[f"jerk_{axis}_std"] = float(np.std(jerk[:, i]))

    return features


def extract_periodicity_features(
    window: NDArray[np.float32],
    fs: int = 50,
) -> dict[str, float]:
    """Extract periodicity features.

    Args:
        window: Accelerometer window of shape (window_size, 3).
        fs: Sampling frequency in Hz.

    Returns:
        Dictionary of feature name to value.

    Features:
        - autocorr_peak_value: Value of first autocorrelation peak
        - autocorr_peak_lag: Lag of first autocorrelation peak (in samples)
        - periodicity_strength: Ratio of peak to surrounding values
        - autocorr_peak_freq: Frequency corresponding to peak lag
    """
    features = {}

    # Compute magnitude
    magnitude = compute_magnitude(window)

    # Normalize
    centered = magnitude - np.mean(magnitude)
    std = np.std(centered)
    if std > 0:
        centered = centered / std
    else:
        centered = centered * 0

    # Autocorrelation
    n = len(centered)
    autocorr = np.correlate(centered, centered, mode="full")[n - 1 :]
    autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr

    # Find first peak after lag 0 (minimum lag of ~0.1 seconds)
    min_lag = int(0.1 * fs)
    max_lag = min(n // 2, int(2.0 * fs))  # Max 2 seconds

    if max_lag > min_lag:
        search_range = autocorr[min_lag:max_lag]
        peaks, _ = signal.find_peaks(search_range)

        if len(peaks) > 0:
            # First peak
            peak_idx = peaks[0] + min_lag
            features["autocorr_peak_value"] = float(autocorr[peak_idx])
            features["autocorr_peak_lag"] = float(peak_idx)
            features["autocorr_peak_freq"] = float(fs / peak_idx) if peak_idx > 0 else 0.0

            # Periodicity strength
            surrounding = autocorr[max(0, peak_idx - 5) : peak_idx + 5]
            features["periodicity_strength"] = float(
                autocorr[peak_idx] / np.mean(np.abs(surrounding))
                if np.mean(np.abs(surrounding)) > 0
                else 0
            )
        else:
            features["autocorr_peak_value"] = 0.0
            features["autocorr_peak_lag"] = 0.0
            features["autocorr_peak_freq"] = 0.0
            features["periodicity_strength"] = 0.0
    else:
        features["autocorr_peak_value"] = 0.0
        features["autocorr_peak_lag"] = 0.0
        features["autocorr_peak_freq"] = 0.0
        features["periodicity_strength"] = 0.0

    return features


def extract_statistical_features(window: NDArray[np.float32]) -> dict[str, float]:
    """Extract additional statistical features.

    Args:
        window: Accelerometer window of shape (window_size, 3).

    Returns:
        Dictionary of feature name to value.

    Features:
        - skewness_{magnitude, x, y, z}: Skewness
        - kurtosis_{magnitude, x, y, z}: Kurtosis
        - correlation_{xy, xz, yz}: Pairwise correlations
    """
    features = {}

    magnitude = compute_magnitude(window)

    # Skewness and kurtosis
    features["skewness_magnitude"] = float(stats.skew(magnitude))
    features["kurtosis_magnitude"] = float(stats.kurtosis(magnitude))

    axis_names = ["x", "y", "z"]
    for i, axis in enumerate(axis_names):
        features[f"skewness_{axis}"] = float(stats.skew(window[:, i]))
        features[f"kurtosis_{axis}"] = float(stats.kurtosis(window[:, i]))

    # Pairwise correlations
    axis_pairs = [("x", "y", 0, 1), ("x", "z", 0, 2), ("y", "z", 1, 2)]
    for a1, a2, i1, i2 in axis_pairs:
        corr = np.corrcoef(window[:, i1], window[:, i2])[0, 1]
        features[f"correlation_{a1}{a2}"] = float(corr if not np.isnan(corr) else 0)

    return features


def extract_all_features(
    window: NDArray[np.float32],
    fs: int = 50,
) -> dict[str, float]:
    """Extract all features from a single window.

    Args:
        window: Accelerometer window of shape (window_size, 3).
        fs: Sampling frequency in Hz.

    Returns:
        Dictionary of all feature names to values.
    """
    features = {}
    features.update(extract_time_domain_features(window))
    features.update(extract_frequency_domain_features(window, fs))
    features.update(extract_dynamics_features(window, fs))
    features.update(extract_periodicity_features(window, fs))
    features.update(extract_statistical_features(window))
    return features


def extract_features_batch(
    windows: NDArray[np.float32],
    fs: int = 50,
) -> NDArray[np.float32]:
    """Extract features from a batch of windows.

    Args:
        windows: Batch of windows of shape (n_windows, window_size, n_channels).
        fs: Sampling frequency in Hz.

    Returns:
        Feature matrix of shape (n_windows, n_features).
    """
    n_windows = windows.shape[0]

    # Get feature names from first window
    sample_features = extract_all_features(windows[0], fs)
    feature_names = sorted(sample_features.keys())
    n_features = len(feature_names)

    # Extract features for all windows
    feature_matrix = np.zeros((n_windows, n_features), dtype=np.float32)

    for i in range(n_windows):
        features = extract_all_features(windows[i], fs)
        feature_matrix[i] = [features[name] for name in feature_names]

    return feature_matrix


def get_feature_names(fs: int = 50) -> list[str]:
    """Get list of all feature names.

    Args:
        fs: Sampling frequency in Hz.

    Returns:
        Sorted list of feature names.
    """
    # Create dummy window
    dummy_window = np.random.randn(int(1.5 * fs), 3).astype(np.float32)
    features = extract_all_features(dummy_window, fs)
    return sorted(features.keys())
