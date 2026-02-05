"""Signal filtering methods for behavior classification enhancement.

This module implements various filtering techniques including wavelet denoising,
Hampel outlier detection, and behavior-specific filtering strategies.

Reference:
    Zhang et al. (2025). "Behavior-Specific Filtering for Enhanced Pig Behavior
    Classification in Precision Livestock Farming."
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pywt
from numpy.typing import NDArray
from scipy import signal


def wavelet_denoise(
    data: NDArray[np.float32],
    wavelet: str = "db4",
    level: int = 3,
    mode: str = "soft",
) -> NDArray[np.float32]:
    """Apply wavelet denoising to sensor data.

    Uses discrete wavelet transform (DWT) with soft thresholding to remove noise
    while preserving high-frequency signal components critical for active behaviors.

    Args:
        data: Input data array of shape (n_samples, n_channels).
        wavelet: Wavelet family to use (e.g., 'db4', 'sym4', 'coif3').
        level: Decomposition level (typically 3-5 for accelerometer data).
        mode: Thresholding mode ('soft' or 'hard').
            Soft thresholding is recommended as it produces smoother results.

    Returns:
        Denoised data array of same shape as input.

    Note:
        - Preserves high-frequency components better than lowpass filtering
        - Recommended for active behaviors (eating, walking, drinking)
        - From Zhang et al. 2025: achieved 91.58% accuracy vs 84.09% raw data

    Example:
        >>> window = np.random.randn(75, 3).astype(np.float32)
        >>> denoised = wavelet_denoise(window, wavelet='db4', level=3)
        >>> denoised.shape
        (75, 3)
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    denoised = np.zeros_like(data)

    for i in range(data.shape[1]):
        channel_data = data[:, i]

        # Perform wavelet decomposition
        coeffs = pywt.wavedec(channel_data, wavelet, level=level)

        # Calculate universal threshold (VisuShrink)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(channel_data)))

        # Apply thresholding to detail coefficients
        coeffs_thresh = [coeffs[0]]  # Keep approximation coefficients
        for detail_coeff in coeffs[1:]:
            if mode == "soft":
                thresh_coeff = pywt.threshold(detail_coeff, threshold, mode="soft")
            else:
                thresh_coeff = pywt.threshold(detail_coeff, threshold, mode="hard")
            coeffs_thresh.append(thresh_coeff)

        # Reconstruct signal
        denoised[:, i] = pywt.waverec(coeffs_thresh, wavelet)[:len(channel_data)]

    return denoised.astype(np.float32)


def hampel_filter(
    data: NDArray[np.float32],
    window_size: int = 5,
    n_sigma: float = 3.0,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Apply Hampel filter for outlier detection and removal.

    Identifies outliers using median absolute deviation (MAD) within a sliding
    window. More robust than z-score for accelerometer data with sudden movements.

    Args:
        data: Input data array of shape (n_samples, n_channels).
        window_size: Size of the sliding window (samples).
        n_sigma: Number of standard deviations for outlier threshold.

    Returns:
        Tuple of (filtered_data, outlier_mask).
        - filtered_data: Data with outliers replaced by local median.
        - outlier_mask: Boolean array indicating outlier locations.

    Note:
        - From Zhang et al. 2025: only 6.64% data loss vs 15.45% for IQR
        - More effective at preserving useful data while removing noise

    Example:
        >>> data = np.random.randn(100, 3).astype(np.float32)
        >>> data[50, :] = 100  # Add outlier
        >>> filtered, mask = hampel_filter(data, window_size=5)
        >>> mask[50]
        True
    """
    filtered = data.copy()
    outlier_mask = np.zeros(data.shape[0], dtype=bool)

    half_window = window_size // 2

    for i in range(data.shape[1]):  # For each channel
        channel_data = data[:, i]

        for j in range(half_window, len(channel_data) - half_window):
            # Extract window
            window = channel_data[j - half_window : j + half_window + 1]

            # Compute median and MAD
            median = np.median(window)
            mad = np.median(np.abs(window - median))

            # MAD-based threshold (1.4826 is for consistency with std)
            threshold = n_sigma * 1.4826 * mad

            # Check if current point is outlier
            if np.abs(channel_data[j] - median) > threshold:
                filtered[j, i] = median
                outlier_mask[j] = True

    return filtered, outlier_mask


def lowpass_filter(
    data: NDArray[np.float32],
    cutoff_freq: float = 5.0,
    sampling_rate: int = 50,
    order: int = 4,
) -> NDArray[np.float32]:
    """Apply Butterworth lowpass filter.

    Aggressively removes high-frequency noise. Best for inactive behaviors
    where motion is minimal and signal stability is prioritized.

    Args:
        data: Input data array of shape (n_samples, n_channels).
        cutoff_freq: Cutoff frequency in Hz.
        sampling_rate: Sampling rate in Hz.
        order: Filter order.

    Returns:
        Filtered data array.

    Note:
        - Recommended for inactive behaviors (lying, standing)
        - Not recommended for active behaviors as it removes important motion details
    """
    nyquist = sampling_rate / 2
    normalized_cutoff = cutoff_freq / nyquist

    if normalized_cutoff >= 1.0:
        return data

    b, a = signal.butter(order, normalized_cutoff, btype="low")

    filtered = np.zeros_like(data)
    for i in range(data.shape[1]):
        filtered[:, i] = signal.filtfilt(b, a, data[:, i])

    return filtered.astype(np.float32)


def median_filter(
    data: NDArray[np.float32],
    kernel_size: int = 3,
) -> NDArray[np.float32]:
    """Apply median filter for noise reduction.

    Non-linear filter that preserves edges while removing impulsive noise.

    Args:
        data: Input data array of shape (n_samples, n_channels).
        kernel_size: Size of the median filter kernel (odd number).

    Returns:
        Filtered data array.
    """
    filtered = np.zeros_like(data)
    for i in range(data.shape[1]):
        filtered[:, i] = signal.medfilt(data[:, i], kernel_size=kernel_size)

    return filtered.astype(np.float32)


def apply_filter(
    data: NDArray[np.float32],
    filter_type: Literal["none", "wavelet", "lowpass", "median", "hampel"],
    **kwargs,
) -> NDArray[np.float32]:
    """Apply specified filter to data.

    Unified interface for applying different filtering methods.

    Args:
        data: Input data array of shape (n_samples, n_channels).
        filter_type: Type of filter to apply.
        **kwargs: Additional arguments passed to the specific filter function.

    Returns:
        Filtered data array.

    Example:
        >>> data = np.random.randn(75, 3).astype(np.float32)
        >>> filtered = apply_filter(data, "wavelet", wavelet="db4", level=3)
    """
    if filter_type == "none":
        return data
    elif filter_type == "wavelet":
        return wavelet_denoise(data, **kwargs)
    elif filter_type == "lowpass":
        return lowpass_filter(data, **kwargs)
    elif filter_type == "median":
        return median_filter(data, **kwargs)
    elif filter_type == "hampel":
        filtered, _ = hampel_filter(data, **kwargs)
        return filtered
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


def behavior_specific_filter(
    data: NDArray[np.float32],
    behavior_label: str | int,
    active_behaviors: list[str | int] | None = None,
    active_filter: str = "wavelet",
    inactive_filter: str = "lowpass",
    **kwargs,
) -> NDArray[np.float32]:
    """Apply behavior-specific filtering strategy.

    Applies different filters based on whether the behavior is active or inactive.
    This is the method from Zhang et al. 2025 that achieved 94.73% accuracy.

    Args:
        data: Input data array of shape (n_samples, n_channels).
        behavior_label: Ground truth behavior label (for training only).
        active_behaviors: List of behaviors considered "active".
            If None, defaults to common active pig behaviors.
        active_filter: Filter to use for active behaviors (default: "wavelet").
        inactive_filter: Filter to use for inactive behaviors (default: "lowpass").
        **kwargs: Additional arguments for filters.

    Returns:
        Filtered data array.

    Warning:
        This function requires ground truth labels and should ONLY be used during
        training data preprocessing. It cannot be used during inference.

    Example:
        >>> # Training time only
        >>> window = np.random.randn(75, 3).astype(np.float32)
        >>> filtered = behavior_specific_filter(
        ...     window,
        ...     behavior_label="eating",
        ...     active_behaviors=["eating", "walking", "drinking"]
        ... )
    """
    if active_behaviors is None:
        # Default active behaviors for pigs
        active_behaviors = ["eating", "walking", "drinking", "interacting"]

    # Determine if behavior is active
    is_active = behavior_label in active_behaviors

    # Apply appropriate filter
    if is_active:
        return apply_filter(data, active_filter, **kwargs)
    else:
        return apply_filter(data, inactive_filter, **kwargs)
