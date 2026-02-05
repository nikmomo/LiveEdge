"""Signal reconstruction evaluation metrics.

This module provides metrics for evaluating signal reconstruction quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class ReconstructionMetrics:
    """Reconstruction evaluation metrics.

    Attributes:
        rmse: Root Mean Square Error.
        mae: Mean Absolute Error.
        mape: Mean Absolute Percentage Error.
        correlation: Pearson correlation coefficient.
        snr_db: Signal-to-Noise Ratio in dB.
        max_error: Maximum reconstruction error.
    """

    rmse: float
    mae: float
    mape: float
    correlation: float
    snr_db: float
    max_error: float

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "mape": self.mape,
            "correlation": self.correlation,
            "snr_db": self.snr_db,
            "max_error": self.max_error,
        }

    def summary(self) -> str:
        """Get summary string."""
        return (
            f"RMSE: {self.rmse:.4f}\n"
            f"MAE: {self.mae:.4f}\n"
            f"Correlation: {self.correlation:.4f}\n"
            f"SNR: {self.snr_db:.2f} dB"
        )


def compute_reconstruction_metrics(
    original: NDArray[np.float32],
    reconstructed: NDArray[np.float32],
) -> ReconstructionMetrics:
    """Compute reconstruction quality metrics.

    Args:
        original: Original full-rate signal.
        reconstructed: Reconstructed signal at same rate.

    Returns:
        ReconstructionMetrics with computed values.
    """
    # Ensure same length
    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]

    # Flatten if multi-channel
    if original.ndim > 1:
        original = original.flatten()
        reconstructed = reconstructed.flatten()

    # Error
    error = original - reconstructed

    # RMSE
    rmse = float(np.sqrt(np.mean(error**2)))

    # MAE
    mae = float(np.mean(np.abs(error)))

    # MAPE (avoid division by zero)
    nonzero_mask = np.abs(original) > 1e-10
    if nonzero_mask.any():
        mape = float(np.mean(np.abs(error[nonzero_mask] / original[nonzero_mask])) * 100)
    else:
        mape = 0.0

    # Correlation
    if np.std(original) > 1e-10 and np.std(reconstructed) > 1e-10:
        correlation = float(np.corrcoef(original, reconstructed)[0, 1])
        if np.isnan(correlation):
            correlation = 0.0
    else:
        correlation = 1.0 if np.allclose(original, reconstructed) else 0.0

    # SNR
    signal_power = np.mean(original**2)
    noise_power = np.mean(error**2)
    if noise_power > 1e-10:
        snr_db = float(10 * np.log10(signal_power / noise_power))
    else:
        snr_db = 100.0  # Essentially perfect

    # Max error
    max_error = float(np.max(np.abs(error)))

    return ReconstructionMetrics(
        rmse=rmse,
        mae=mae,
        mape=mape,
        correlation=correlation,
        snr_db=snr_db,
        max_error=max_error,
    )


def compute_feature_preservation(
    original: NDArray[np.float32],
    reconstructed: NDArray[np.float32],
    fs: int = 50,
) -> dict[str, float]:
    """Compute how well features are preserved in reconstruction.

    Args:
        original: Original signal.
        reconstructed: Reconstructed signal.
        fs: Sampling frequency.

    Returns:
        Dictionary of feature preservation scores (0-1).
    """
    from liveedge.data.features import extract_all_features

    # Handle multi-channel
    if original.ndim == 1:
        original = original.reshape(-1, 1)
        reconstructed = reconstructed.reshape(-1, 1)

    # Ensure 3 channels for feature extraction
    if original.shape[1] < 3:
        n_pad = 3 - original.shape[1]
        original = np.pad(original, ((0, 0), (0, n_pad)), mode="constant")
        reconstructed = np.pad(reconstructed, ((0, 0), (0, n_pad)), mode="constant")

    # Extract features from both
    orig_features = extract_all_features(original.astype(np.float32), fs)
    recon_features = extract_all_features(reconstructed.astype(np.float32), fs)

    preservation = {}
    for name in orig_features:
        orig_val = orig_features[name]
        recon_val = recon_features[name]

        if abs(orig_val) > 1e-10:
            error_ratio = abs(orig_val - recon_val) / abs(orig_val)
            preservation[name] = max(0.0, 1.0 - error_ratio)
        else:
            preservation[name] = 1.0 if abs(recon_val) < 1e-10 else 0.0

    return preservation
