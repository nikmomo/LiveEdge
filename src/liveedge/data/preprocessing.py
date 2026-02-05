"""Data preprocessing utilities for accelerometer data.

This module provides functions for loading, cleaning, normalizing,
and windowing accelerometer data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import signal


@dataclass
class WindowConfig:
    """Configuration for signal windowing.

    Attributes:
        window_size: Window duration in seconds.
        overlap: Overlap ratio between consecutive windows (0-1).
        sampling_rate: Original signal sampling rate in Hz.
    """

    window_size: float = 1.5
    overlap: float = 0.5
    sampling_rate: int = 50

    @property
    def window_samples(self) -> int:
        """Number of samples per window."""
        return int(self.window_size * self.sampling_rate)

    @property
    def step_samples(self) -> int:
        """Number of samples between window starts."""
        return int(self.window_samples * (1 - self.overlap))


@dataclass
class NormalizationParams:
    """Parameters for data normalization.

    Attributes:
        method: Normalization method used.
        mean: Mean values per channel (for standardization).
        std: Standard deviation per channel (for standardization).
        min_val: Minimum values per channel (for min-max scaling).
        max_val: Maximum values per channel (for min-max scaling).
    """

    method: str
    mean: NDArray[np.float32] | None = None
    std: NDArray[np.float32] | None = None
    min_val: NDArray[np.float32] | None = None
    max_val: NDArray[np.float32] | None = None


def load_raw_data(
    data_path: str | Path,
    file_pattern: str = "*.csv",
) -> pd.DataFrame:
    """Load raw accelerometer data from files.

    Args:
        data_path: Path to directory containing data files.
        file_pattern: Glob pattern for data files.

    Returns:
        DataFrame with columns: timestamp, acc_x, acc_y, acc_z, behavior, subject_id.

    Raises:
        FileNotFoundError: If no data files are found.
    """
    data_path = Path(data_path)
    files = list(data_path.glob(file_pattern))

    if not files:
        raise FileNotFoundError(f"No data files found matching {file_pattern} in {data_path}")

    dfs = []
    for file in files:
        df = pd.read_csv(file)
        # Ensure required columns exist
        required_cols = ["timestamp", "acc_x", "acc_y", "acc_z", "behavior"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {file}: {missing}")

        # Add subject_id from filename if not present
        if "subject_id" not in df.columns:
            df["subject_id"] = file.stem

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def remove_outliers(
    data: NDArray[np.float32],
    method: str = "zscore",
    threshold: float = 3.0,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Remove outlier samples using specified method.

    Args:
        data: Input data array of shape (n_samples, n_channels).
        method: Outlier detection method ("zscore" or "iqr").
        threshold: Threshold for outlier detection.
            For zscore: number of standard deviations.
            For iqr: IQR multiplier.

    Returns:
        Tuple of (cleaned_data, outlier_mask).
        outlier_mask is True for outlier samples.

    Raises:
        ValueError: If unknown method is specified.
    """
    if method == "zscore":
        # Z-score based outlier detection
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        z_scores = np.abs((data - mean) / std)
        outlier_mask = np.any(z_scores > threshold, axis=1)

    elif method == "iqr":
        # IQR based outlier detection
        q1 = np.percentile(data, 25, axis=0)
        q3 = np.percentile(data, 75, axis=0)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        outlier_mask = np.any((data < lower_bound) | (data > upper_bound), axis=1)

    else:
        raise ValueError(f"Unknown outlier removal method: {method}")

    cleaned_data = data[~outlier_mask]
    return cleaned_data, outlier_mask


def normalize(
    data: NDArray[np.float32],
    method: str = "standardize",
    params: NormalizationParams | None = None,
) -> tuple[NDArray[np.float32], NormalizationParams]:
    """Normalize data and return normalization parameters.

    Args:
        data: Input data array of shape (n_samples, n_channels).
        method: Normalization method ("standardize" or "minmax").
        params: Optional pre-computed normalization parameters.
            If provided, these are used instead of computing from data.

    Returns:
        Tuple of (normalized_data, normalization_params).

    Raises:
        ValueError: If unknown method is specified.
    """
    if method == "standardize":
        if params is not None:
            mean = params.mean
            std = params.std
        else:
            mean = np.mean(data, axis=0, dtype=np.float32)
            std = np.std(data, axis=0, dtype=np.float32)
            std = np.where(std == 0, 1, std)  # Avoid division by zero

        normalized = (data - mean) / std
        params = NormalizationParams(method=method, mean=mean, std=std)

    elif method == "minmax":
        if params is not None:
            min_val = params.min_val
            max_val = params.max_val
        else:
            min_val = np.min(data, axis=0).astype(np.float32)
            max_val = np.max(data, axis=0).astype(np.float32)

        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1, range_val)  # Avoid division by zero
        normalized = (data - min_val) / range_val
        params = NormalizationParams(method=method, min_val=min_val, max_val=max_val)

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized.astype(np.float32), params


def apply_lowpass_filter(
    data: NDArray[np.float32],
    cutoff_freq: float = 20.0,
    sampling_rate: int = 50,
    order: int = 4,
) -> NDArray[np.float32]:
    """Apply Butterworth lowpass filter to remove high-frequency noise.

    Args:
        data: Input data array of shape (n_samples, n_channels).
        cutoff_freq: Cutoff frequency in Hz.
        sampling_rate: Sampling rate in Hz.
        order: Filter order.

    Returns:
        Filtered data array.
    """
    nyquist = sampling_rate / 2
    normalized_cutoff = cutoff_freq / nyquist

    if normalized_cutoff >= 1.0:
        # Cutoff frequency is at or above Nyquist, no filtering needed
        return data

    b, a = signal.butter(order, normalized_cutoff, btype="low")

    filtered = np.zeros_like(data)
    for i in range(data.shape[1]):
        filtered[:, i] = signal.filtfilt(b, a, data[:, i])

    return filtered.astype(np.float32)


def resample_data(
    data: NDArray[np.float32],
    original_rate: int,
    target_rate: int,
) -> NDArray[np.float32]:
    """Resample data to a different sampling rate.

    Args:
        data: Input data array of shape (n_samples, n_channels).
        original_rate: Original sampling rate in Hz.
        target_rate: Target sampling rate in Hz.

    Returns:
        Resampled data array.
    """
    if original_rate == target_rate:
        return data

    # Calculate new number of samples
    n_samples = data.shape[0]
    new_n_samples = int(n_samples * target_rate / original_rate)

    resampled = signal.resample(data, new_n_samples, axis=0)
    return resampled.astype(np.float32)


def create_windows(
    data: NDArray[np.float32],
    labels: NDArray[np.int64],
    config: WindowConfig,
    timestamps: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.int64], NDArray[np.float64] | None]:
    """Create overlapping windows from continuous data.

    Uses majority voting for window labels.

    Args:
        data: Continuous data of shape (n_samples, n_channels).
        labels: Labels for each sample of shape (n_samples,).
        config: Windowing configuration.
        timestamps: Optional timestamps for each sample.

    Returns:
        Tuple of (windows, window_labels, window_timestamps).
        windows: Shape (n_windows, window_samples, n_channels).
        window_labels: Shape (n_windows,).
        window_timestamps: Shape (n_windows,) or None.
    """
    n_samples = data.shape[0]
    window_size = config.window_samples
    step_size = config.step_samples

    # Calculate number of complete windows
    n_windows = max(0, (n_samples - window_size) // step_size + 1)

    if n_windows == 0:
        return (
            np.array([], dtype=np.float32).reshape(0, window_size, data.shape[1]),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64) if timestamps is not None else None,
        )

    windows = np.zeros((n_windows, window_size, data.shape[1]), dtype=np.float32)
    window_labels = np.zeros(n_windows, dtype=np.int64)
    window_timestamps = np.zeros(n_windows, dtype=np.float64) if timestamps is not None else None

    for i in range(n_windows):
        start = i * step_size
        end = start + window_size

        windows[i] = data[start:end]

        # Majority voting for window label
        window_label_counts = np.bincount(labels[start:end])
        window_labels[i] = np.argmax(window_label_counts)

        if timestamps is not None and window_timestamps is not None:
            window_timestamps[i] = timestamps[start]

    return windows, window_labels, window_timestamps


def create_windows_by_subject(
    df: pd.DataFrame,
    config: WindowConfig,
    acc_columns: list[str] | None = None,
) -> dict[str, tuple[NDArray[np.float32], NDArray[np.int64], NDArray[np.float64]]]:
    """Create windows for each subject separately.

    This prevents windows from spanning different subjects.

    Args:
        df: DataFrame with accelerometer data.
        config: Windowing configuration.
        acc_columns: Names of accelerometer columns (default: ["acc_x", "acc_y", "acc_z"]).

    Returns:
        Dictionary mapping subject_id to (windows, labels, timestamps).
    """
    if acc_columns is None:
        acc_columns = ["acc_x", "acc_y", "acc_z"]

    # Encode labels to integers
    unique_behaviors = sorted(df["behavior"].unique())
    behavior_to_idx = {b: i for i, b in enumerate(unique_behaviors)}

    result = {}

    for subject_id, subject_df in df.groupby("subject_id"):
        data = subject_df[acc_columns].values.astype(np.float32)
        labels = subject_df["behavior"].map(behavior_to_idx).values.astype(np.int64)
        timestamps = subject_df["timestamp"].values.astype(np.float64)

        windows, window_labels, window_timestamps = create_windows(
            data, labels, config, timestamps
        )

        result[subject_id] = (windows, window_labels, window_timestamps)

    return result


def split_by_subject(
    windows_by_subject: dict[str, tuple[NDArray[np.float32], NDArray[np.int64], Any]],
    test_subjects: list[str],
    val_subjects: list[str] | None = None,
) -> dict[str, tuple[NDArray[np.float32], NDArray[np.int64]]]:
    """Split data by subject for LOSO cross-validation.

    Args:
        windows_by_subject: Dictionary from create_windows_by_subject.
        test_subjects: List of subject IDs for test set.
        val_subjects: Optional list of subject IDs for validation set.

    Returns:
        Dictionary with "train", "val" (if provided), and "test" keys,
        each containing (windows, labels) tuple.
    """
    train_windows = []
    train_labels = []
    val_windows = []
    val_labels = []
    test_windows = []
    test_labels = []

    for subject_id, (windows, labels, _) in windows_by_subject.items():
        if subject_id in test_subjects:
            test_windows.append(windows)
            test_labels.append(labels)
        elif val_subjects is not None and subject_id in val_subjects:
            val_windows.append(windows)
            val_labels.append(labels)
        else:
            train_windows.append(windows)
            train_labels.append(labels)

    result = {
        "train": (
            np.concatenate(train_windows) if train_windows else np.array([]),
            np.concatenate(train_labels) if train_labels else np.array([]),
        ),
        "test": (
            np.concatenate(test_windows) if test_windows else np.array([]),
            np.concatenate(test_labels) if test_labels else np.array([]),
        ),
    }

    if val_subjects is not None:
        result["val"] = (
            np.concatenate(val_windows) if val_windows else np.array([]),
            np.concatenate(val_labels) if val_labels else np.array([]),
        )

    return result


def compute_class_weights(
    labels: NDArray[np.int64],
) -> NDArray[np.float32]:
    """Compute class weights for imbalanced classification.

    Uses inverse frequency weighting.

    Args:
        labels: Array of class labels.

    Returns:
        Array of class weights.
    """
    unique, counts = np.unique(labels, return_counts=True)
    n_samples = len(labels)
    n_classes = len(unique)

    weights = n_samples / (n_classes * counts)
    return weights.astype(np.float32)
