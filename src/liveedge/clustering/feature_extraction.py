"""Feature extraction for behavior clustering.

This module provides functions for extracting motion dynamics features
used for grouping behaviors into adaptive sampling clusters.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def compute_behavior_dynamics(
    windows: NDArray[np.float32],
    labels: NDArray[np.int64],
    class_names: list[str],
    fs: int = 50,
) -> dict[str, dict[str, float]]:
    """Compute aggregated motion dynamics features per behavior.

    Args:
        windows: Windows of shape (n_windows, window_size, n_channels).
        labels: Labels of shape (n_windows,).
        class_names: List of behavior names.
        fs: Sampling frequency in Hz.

    Returns:
        Dictionary mapping behavior name to feature dictionary.
    """
    from liveedge.data.features import (
        compute_magnitude,
        extract_dynamics_features,
        extract_frequency_domain_features,
        extract_periodicity_features,
    )

    behavior_features: dict[str, dict[str, list[float]]] = {
        name: {} for name in class_names
    }

    for i, (window, label) in enumerate(zip(windows, labels)):
        behavior = class_names[label]

        # Extract dynamics-related features
        freq_feats = extract_frequency_domain_features(window, fs)
        dyn_feats = extract_dynamics_features(window, fs)
        period_feats = extract_periodicity_features(window, fs)

        # Compute additional dynamics features
        magnitude = compute_magnitude(window)
        variance = float(np.var(magnitude))

        # Collect features
        features = {
            "dominant_frequency": freq_feats["dominant_frequency"],
            "high_freq_ratio": freq_feats["high_freq_ratio"],
            "spectral_entropy": freq_feats["spectral_entropy"],
            "jerk_mean": dyn_feats["jerk_mean"],
            "jerk_std": dyn_feats["jerk_std"],
            "acceleration_change_rate": dyn_feats["acceleration_change_rate"],
            "periodicity_strength": period_feats["periodicity_strength"],
            "variance": variance,
        }

        for feat_name, feat_value in features.items():
            if feat_name not in behavior_features[behavior]:
                behavior_features[behavior][feat_name] = []
            behavior_features[behavior][feat_name].append(feat_value)

    # Aggregate features (mean across all windows for each behavior)
    aggregated: dict[str, dict[str, float]] = {}
    for behavior, feat_dict in behavior_features.items():
        aggregated[behavior] = {}
        for feat_name, feat_values in feat_dict.items():
            if feat_values:
                aggregated[behavior][feat_name] = float(np.mean(feat_values))
            else:
                aggregated[behavior][feat_name] = 0.0

    return aggregated


def create_clustering_feature_matrix(
    behavior_features: dict[str, dict[str, float]],
    feature_names: list[str] | None = None,
) -> tuple[NDArray[np.float32], list[str], list[str]]:
    """Create a feature matrix for clustering.

    Args:
        behavior_features: Dictionary from compute_behavior_dynamics.
        feature_names: Optional list of features to include.
            If None, uses all available features.

    Returns:
        Tuple of (feature_matrix, behavior_names, feature_names).
        feature_matrix: Shape (n_behaviors, n_features).
    """
    behavior_names = list(behavior_features.keys())

    if feature_names is None:
        # Get all feature names from first behavior
        if behavior_names:
            feature_names = list(behavior_features[behavior_names[0]].keys())
        else:
            feature_names = []

    n_behaviors = len(behavior_names)
    n_features = len(feature_names)

    feature_matrix = np.zeros((n_behaviors, n_features), dtype=np.float32)

    for i, behavior in enumerate(behavior_names):
        for j, feat_name in enumerate(feature_names):
            feature_matrix[i, j] = behavior_features[behavior].get(feat_name, 0.0)

    return feature_matrix, behavior_names, feature_names


def normalize_clustering_features(
    feature_matrix: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Normalize features for clustering.

    Uses z-score normalization per feature.

    Args:
        feature_matrix: Feature matrix of shape (n_samples, n_features).

    Returns:
        Normalized feature matrix.
    """
    mean = np.mean(feature_matrix, axis=0)
    std = np.std(feature_matrix, axis=0)
    std = np.where(std == 0, 1, std)  # Avoid division by zero

    return ((feature_matrix - mean) / std).astype(np.float32)


def estimate_minimum_sampling_rate(
    windows: NDArray[np.float32],
    labels: NDArray[np.int64],
    class_names: list[str],
    fs_original: int = 50,
    accuracy_threshold: float = 0.95,
) -> dict[str, float]:
    """Estimate minimum sampling rate per behavior to maintain classification accuracy.

    Uses frequency content analysis to determine the minimum sampling rate
    needed to capture the behavior's characteristics.

    Args:
        windows: Windows of shape (n_windows, window_size, n_channels).
        labels: Labels of shape (n_windows,).
        class_names: List of behavior names.
        fs_original: Original sampling frequency in Hz.
        accuracy_threshold: Threshold for cumulative energy to capture.

    Returns:
        Dictionary mapping behavior name to recommended minimum sampling rate (Hz).
    """
    from scipy import signal as sig

    from liveedge.data.features import compute_magnitude

    behavior_rates: dict[str, list[float]] = {name: [] for name in class_names}

    for window, label in zip(windows, labels):
        behavior = class_names[label]

        # Compute magnitude
        magnitude = compute_magnitude(window)

        # FFT
        freqs = np.fft.rfftfreq(len(magnitude), d=1 / fs_original)
        fft_vals = np.abs(np.fft.rfft(magnitude)) ** 2

        # Cumulative energy
        total_energy = np.sum(fft_vals)
        if total_energy == 0:
            # Default to low sampling rate for zero-energy signals
            behavior_rates[behavior].append(5.0)
            continue

        cumulative_energy = np.cumsum(fft_vals) / total_energy

        # Find frequency that captures accuracy_threshold of energy
        idx = np.searchsorted(cumulative_energy, accuracy_threshold)
        idx = min(idx, len(freqs) - 1)
        min_freq = freqs[idx]

        # Nyquist: sampling rate should be at least 2x the max frequency
        min_rate = max(5.0, min_freq * 2.5)  # Add some margin

        behavior_rates[behavior].append(min_rate)

    # Aggregate (take 75th percentile to be conservative)
    recommended_rates = {}
    for behavior, rates in behavior_rates.items():
        if rates:
            recommended_rates[behavior] = float(np.percentile(rates, 75))
        else:
            recommended_rates[behavior] = 50.0  # Default to max

    return recommended_rates
