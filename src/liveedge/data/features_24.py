"""24-feature extraction module for LiveEdge.

24-Feature Specification:
    Per-axis (X, Y, Z) - 7 features each = 21 total:
    - mean, std, min, max, Q25, Q75, RMS

    Magnitude - 3 features:
    - magnitude_mean, magnitude_std, magnitude_range
"""

from __future__ import annotations

from typing import List

import numpy as np
from numpy.typing import NDArray


FEATURE_NAMES_24: List[str] = [
    # X-axis features
    "acc_x_mean", "acc_x_std", "acc_x_min", "acc_x_max",
    "acc_x_q25", "acc_x_q75", "acc_x_rms",
    # Y-axis features
    "acc_y_mean", "acc_y_std", "acc_y_min", "acc_y_max",
    "acc_y_q25", "acc_y_q75", "acc_y_rms",
    # Z-axis features
    "acc_z_mean", "acc_z_std", "acc_z_min", "acc_z_max",
    "acc_z_q25", "acc_z_q75", "acc_z_rms",
    # Magnitude features
    "magnitude_mean", "magnitude_std", "magnitude_range",
]


def extract_24_features(window: NDArray[np.floating]) -> NDArray[np.floating]:
    """Extract 24 statistical features from a single window.

    Args:
        window: Shape (n_samples, 3) accelerometer data [x, y, z].

    Returns:
        Shape (24,) feature vector.
    """
    features = []

    # Per-axis features (7 each)
    for axis in range(3):
        data = window[:, axis]
        features.extend([
            np.mean(data),
            np.std(data),
            np.min(data),
            np.max(data),
            np.percentile(data, 25),
            np.percentile(data, 75),
            np.sqrt(np.mean(data ** 2)),
        ])

    # Magnitude features (3)
    magnitude = np.sqrt(np.sum(window ** 2, axis=1))
    features.extend([
        np.mean(magnitude),
        np.std(magnitude),
        np.max(magnitude) - np.min(magnitude),
    ])

    return np.array(features, dtype=np.float32)


def get_feature_names_24() -> List[str]:
    """Return the list of 24 feature names."""
    return list(FEATURE_NAMES_24)


def extract_24_features_batch(windows: NDArray[np.floating]) -> NDArray[np.floating]:
    """Extract 24 features from a batch of windows (vectorized).

    Args:
        windows: Shape (n_windows, n_samples, 3) accelerometer data.

    Returns:
        Shape (n_windows, 24) feature matrix.
    """
    n_windows = windows.shape[0]
    features = np.zeros((n_windows, 24), dtype=np.float32)

    # Per-axis features (vectorized over all windows)
    for axis in range(3):
        data = windows[:, :, axis]  # (n_windows, n_samples)
        col = axis * 7
        features[:, col + 0] = np.mean(data, axis=1)
        features[:, col + 1] = np.std(data, axis=1)
        features[:, col + 2] = np.min(data, axis=1)
        features[:, col + 3] = np.max(data, axis=1)
        features[:, col + 4] = np.percentile(data, 25, axis=1)
        features[:, col + 5] = np.percentile(data, 75, axis=1)
        features[:, col + 6] = np.sqrt(np.mean(data ** 2, axis=1))

    # Magnitude features (vectorized)
    magnitude = np.sqrt(np.sum(windows ** 2, axis=2))  # (n_windows, n_samples)
    features[:, 21] = np.mean(magnitude, axis=1)
    features[:, 22] = np.std(magnitude, axis=1)
    features[:, 23] = np.max(magnitude, axis=1) - np.min(magnitude, axis=1)

    return features
