"""Signal reconstruction and interpolation methods.

This module provides methods for reconstructing full-rate signals
from adaptively sampled data.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from numpy.typing import NDArray
from scipy import interpolate, signal


class InterpolationMethod(Enum):
    """Available interpolation methods."""

    LINEAR = "linear"
    CUBIC = "cubic"
    SPLINE = "spline"
    NEAREST = "nearest"
    ZERO_ORDER_HOLD = "zero_order_hold"
    SINC = "sinc"


def reconstruct_signal(
    samples: NDArray[np.float32],
    sample_times: NDArray[np.float64],
    target_times: NDArray[np.float64],
    method: InterpolationMethod | str = InterpolationMethod.LINEAR,
) -> NDArray[np.float32]:
    """Reconstruct signal at target times from sparse samples.

    Args:
        samples: Sampled values of shape (n_samples,) or (n_samples, n_channels).
        sample_times: Times of samples.
        target_times: Times at which to reconstruct signal.
        method: Interpolation method to use.

    Returns:
        Reconstructed signal at target times.

    Raises:
        ValueError: If method is not supported.
    """
    if isinstance(method, str):
        method = InterpolationMethod(method.lower())

    # Handle multi-channel data
    if samples.ndim == 1:
        samples = samples[:, np.newaxis]
        single_channel = True
    else:
        single_channel = False

    n_channels = samples.shape[1]
    n_target = len(target_times)
    reconstructed = np.zeros((n_target, n_channels), dtype=np.float32)

    for ch in range(n_channels):
        channel_samples = samples[:, ch]

        if method == InterpolationMethod.LINEAR:
            reconstructed[:, ch] = np.interp(target_times, sample_times, channel_samples)

        elif method == InterpolationMethod.NEAREST:
            # Find nearest sample for each target time
            indices = np.searchsorted(sample_times, target_times)
            indices = np.clip(indices, 0, len(sample_times) - 1)
            # Check if previous sample is closer
            for i, (idx, t) in enumerate(zip(indices, target_times)):
                if idx > 0:
                    if abs(sample_times[idx - 1] - t) < abs(sample_times[idx] - t):
                        indices[i] = idx - 1
            reconstructed[:, ch] = channel_samples[indices]

        elif method == InterpolationMethod.ZERO_ORDER_HOLD:
            # Hold previous value
            indices = np.searchsorted(sample_times, target_times, side="right") - 1
            indices = np.clip(indices, 0, len(sample_times) - 1)
            reconstructed[:, ch] = channel_samples[indices]

        elif method == InterpolationMethod.CUBIC:
            if len(sample_times) >= 4:
                f = interpolate.interp1d(
                    sample_times,
                    channel_samples,
                    kind="cubic",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                reconstructed[:, ch] = f(target_times)
            else:
                # Fall back to linear for insufficient samples
                reconstructed[:, ch] = np.interp(target_times, sample_times, channel_samples)

        elif method == InterpolationMethod.SPLINE:
            if len(sample_times) >= 4:
                # B-spline interpolation
                tck = interpolate.splrep(sample_times, channel_samples, k=3)
                reconstructed[:, ch] = interpolate.splev(target_times, tck)
            else:
                reconstructed[:, ch] = np.interp(target_times, sample_times, channel_samples)

        elif method == InterpolationMethod.SINC:
            reconstructed[:, ch] = sinc_interpolate(
                channel_samples, sample_times, target_times
            )

        else:
            raise ValueError(f"Unsupported interpolation method: {method}")

    if single_channel:
        return reconstructed[:, 0].astype(np.float32)
    return reconstructed.astype(np.float32)


def sinc_interpolate(
    samples: NDArray[np.float32],
    sample_times: NDArray[np.float64],
    target_times: NDArray[np.float64],
) -> NDArray[np.float32]:
    """Perform sinc interpolation (ideal bandlimited reconstruction).

    Args:
        samples: Sampled values.
        sample_times: Times of samples.
        target_times: Times at which to reconstruct.

    Returns:
        Reconstructed signal.
    """
    if len(samples) < 2:
        return np.full(len(target_times), samples[0] if len(samples) > 0 else 0, dtype=np.float32)

    # Estimate sampling period
    dt = np.median(np.diff(sample_times))

    # Sinc interpolation
    reconstructed = np.zeros(len(target_times), dtype=np.float64)

    for i, t in enumerate(target_times):
        # Sinc weights
        sinc_args = (t - sample_times) / dt
        weights = np.sinc(sinc_args)
        reconstructed[i] = np.sum(samples * weights)

    return reconstructed.astype(np.float32)


def adaptive_to_fixed_rate(
    samples: NDArray[np.float32],
    sample_times: NDArray[np.float64],
    target_rate: float,
    duration: float | None = None,
    method: InterpolationMethod | str = InterpolationMethod.LINEAR,
) -> tuple[NDArray[np.float32], NDArray[np.float64]]:
    """Convert adaptively sampled data to fixed-rate.

    Args:
        samples: Adaptively sampled values.
        sample_times: Times of samples.
        target_rate: Target sampling rate in Hz.
        duration: Total duration in seconds (None = infer from data).
        method: Interpolation method.

    Returns:
        Tuple of (reconstructed_samples, target_times).
    """
    if duration is None:
        duration = sample_times[-1] - sample_times[0]

    start_time = sample_times[0]
    n_samples = int(duration * target_rate) + 1
    target_times = start_time + np.arange(n_samples) / target_rate

    reconstructed = reconstruct_signal(samples, sample_times, target_times, method)

    return reconstructed, target_times


def downsample_signal(
    data: NDArray[np.float32],
    original_rate: float,
    target_rate: float,
    antialias: bool = True,
) -> NDArray[np.float32]:
    """Downsample signal to a lower rate.

    Args:
        data: Signal data of shape (n_samples,) or (n_samples, n_channels).
        original_rate: Original sampling rate in Hz.
        target_rate: Target sampling rate in Hz.
        antialias: Whether to apply anti-aliasing filter.

    Returns:
        Downsampled signal.
    """
    if target_rate >= original_rate:
        return data

    # Calculate decimation factor
    factor = int(original_rate / target_rate)

    if data.ndim == 1:
        data = data[:, np.newaxis]
        single_channel = True
    else:
        single_channel = False

    n_channels = data.shape[1]
    n_output = len(data) // factor

    if antialias:
        # Apply anti-aliasing lowpass filter
        cutoff = target_rate / 2 * 0.9  # 90% of Nyquist
        b, a = signal.butter(4, cutoff / (original_rate / 2), btype="low")
        filtered = np.zeros_like(data)
        for ch in range(n_channels):
            filtered[:, ch] = signal.filtfilt(b, a, data[:, ch])
        data = filtered

    # Decimate
    downsampled = data[::factor]

    if single_channel:
        return downsampled[:, 0].astype(np.float32)
    return downsampled.astype(np.float32)


def simulate_adaptive_sampling(
    full_rate_data: NDArray[np.float32],
    full_rate_times: NDArray[np.float64],
    sampling_rates: NDArray[np.float64],
    rate_change_times: NDArray[np.float64],
) -> tuple[NDArray[np.float32], NDArray[np.float64]]:
    """Simulate adaptive sampling from full-rate data.

    Args:
        full_rate_data: Full-rate signal data.
        full_rate_times: Full-rate timestamps.
        sampling_rates: Sequence of sampling rates.
        rate_change_times: Times when sampling rate changes.

    Returns:
        Tuple of (sampled_data, sample_times).
    """
    sampled_data = []
    sample_times = []

    for i, (rate, start_time) in enumerate(zip(sampling_rates, rate_change_times)):
        # Determine end time
        if i < len(rate_change_times) - 1:
            end_time = rate_change_times[i + 1]
        else:
            end_time = full_rate_times[-1]

        # Find samples in this interval
        mask = (full_rate_times >= start_time) & (full_rate_times < end_time)
        interval_data = full_rate_data[mask]
        interval_times = full_rate_times[mask]

        if len(interval_data) == 0:
            continue

        # Downsample to target rate
        full_rate = 1.0 / np.median(np.diff(full_rate_times))
        if rate < full_rate:
            factor = max(1, int(full_rate / rate))
            sampled_data.append(interval_data[::factor])
            sample_times.append(interval_times[::factor])
        else:
            sampled_data.append(interval_data)
            sample_times.append(interval_times)

    if sampled_data:
        return (
            np.concatenate(sampled_data).astype(np.float32),
            np.concatenate(sample_times),
        )
    return np.array([], dtype=np.float32), np.array([], dtype=np.float64)
