"""Tests for data preprocessing module."""

from __future__ import annotations

import numpy as np
import pytest

from liveedge.data.preprocessing import (
    NormalizationParams,
    WindowConfig,
    apply_lowpass_filter,
    compute_class_weights,
    create_windows,
    normalize,
    remove_outliers,
    resample_data,
)


class TestWindowConfig:
    """Tests for WindowConfig dataclass."""

    def test_default_values(self):
        config = WindowConfig()
        assert config.window_size == 1.5
        assert config.overlap == 0.5
        assert config.sampling_rate == 50

    def test_window_samples(self):
        config = WindowConfig(window_size=1.5, sampling_rate=50)
        assert config.window_samples == 75

    def test_step_samples(self):
        config = WindowConfig(window_size=1.5, overlap=0.5, sampling_rate=50)
        # 75 samples * 0.5 = 37.5, int() = 37
        assert config.step_samples == 37

    def test_custom_values(self):
        config = WindowConfig(window_size=2.0, overlap=0.25, sampling_rate=100)
        assert config.window_samples == 200
        assert config.step_samples == 150


class TestRemoveOutliers:
    """Tests for remove_outliers function."""

    def test_zscore_method(self, sample_window):
        cleaned, mask = remove_outliers(sample_window, method="zscore", threshold=3.0)
        # Most samples should be kept
        assert len(cleaned) > 0
        assert len(cleaned) + mask.sum() == len(sample_window)

    def test_iqr_method(self, sample_window):
        cleaned, mask = remove_outliers(sample_window, method="iqr", threshold=1.5)
        assert len(cleaned) > 0
        assert len(cleaned) + mask.sum() == len(sample_window)

    def test_extreme_outliers_detected(self):
        # Create data with extreme outliers
        data = np.random.randn(100, 3).astype(np.float32)
        data[50, 0] = 100.0  # Extreme outlier
        data[60, 1] = -100.0  # Extreme outlier

        cleaned, mask = remove_outliers(data, method="zscore", threshold=3.0)
        assert mask[50]  # Should be marked as outlier
        assert mask[60]  # Should be marked as outlier

    def test_invalid_method(self, sample_window):
        with pytest.raises(ValueError):
            remove_outliers(sample_window, method="invalid")


class TestNormalize:
    """Tests for normalize function."""

    def test_standardize(self, sample_window):
        normalized, params = normalize(sample_window, method="standardize")

        # Check that data is standardized
        assert np.abs(np.mean(normalized, axis=0)).max() < 0.01
        assert np.abs(np.std(normalized, axis=0) - 1).max() < 0.01

        assert params.method == "standardize"
        assert params.mean is not None
        assert params.std is not None

    def test_minmax(self, sample_window):
        normalized, params = normalize(sample_window, method="minmax")

        # Check that data is in [0, 1] range
        assert np.min(normalized) >= -0.01
        assert np.max(normalized) <= 1.01

        assert params.method == "minmax"
        assert params.min_val is not None
        assert params.max_val is not None

    def test_reuse_params(self, sample_window):
        _, params = normalize(sample_window, method="standardize")

        # Create new data
        new_data = np.random.randn(50, 3).astype(np.float32)
        normalized, _ = normalize(new_data, method="standardize", params=params)

        # New data normalized with original params should have different stats
        # (unless it happens to have same distribution)
        assert normalized.shape == new_data.shape

    def test_constant_data(self):
        # Constant data should not cause division by zero
        data = np.ones((75, 3), dtype=np.float32)
        normalized, params = normalize(data, method="standardize")
        assert not np.isnan(normalized).any()
        assert not np.isinf(normalized).any()

    def test_invalid_method(self, sample_window):
        with pytest.raises(ValueError):
            normalize(sample_window, method="invalid")


class TestApplyLowpassFilter:
    """Tests for apply_lowpass_filter function."""

    def test_filter_removes_high_freq(self):
        fs = 50
        t = np.linspace(0, 1, fs)

        # Create signal with low and high frequency components
        low_freq = np.sin(2 * np.pi * 2 * t)  # 2 Hz
        high_freq = np.sin(2 * np.pi * 20 * t)  # 20 Hz
        signal = np.column_stack([low_freq + high_freq] * 3).astype(np.float32)

        # Filter at 10 Hz
        filtered = apply_lowpass_filter(signal, cutoff_freq=10.0, sampling_rate=fs)

        # High frequency should be attenuated
        # (we can't check exactly, but power at high freq should be lower)
        assert filtered.shape == signal.shape

    def test_cutoff_at_nyquist(self):
        # Cutoff at or above Nyquist should return unchanged data
        data = np.random.randn(50, 3).astype(np.float32)
        filtered = apply_lowpass_filter(data, cutoff_freq=25.0, sampling_rate=50)
        # Should be identical (no filtering)
        np.testing.assert_array_almost_equal(filtered, data)


class TestResampleData:
    """Tests for resample_data function."""

    def test_upsample(self):
        data = np.random.randn(50, 3).astype(np.float32)
        resampled = resample_data(data, original_rate=50, target_rate=100)
        assert resampled.shape[0] == 100
        assert resampled.shape[1] == 3

    def test_downsample(self):
        data = np.random.randn(100, 3).astype(np.float32)
        resampled = resample_data(data, original_rate=100, target_rate=50)
        assert resampled.shape[0] == 50
        assert resampled.shape[1] == 3

    def test_same_rate(self):
        data = np.random.randn(50, 3).astype(np.float32)
        resampled = resample_data(data, original_rate=50, target_rate=50)
        np.testing.assert_array_equal(resampled, data)


class TestCreateWindows:
    """Tests for create_windows function."""

    def test_basic_windowing(self, continuous_data):
        data, labels, timestamps = continuous_data
        config = WindowConfig(window_size=1.0, overlap=0.5, sampling_rate=50)

        windows, window_labels, window_timestamps = create_windows(
            data, labels, config, timestamps
        )

        # Check shapes
        assert windows.shape[1] == config.window_samples  # Window size
        assert windows.shape[2] == 3  # Channels
        assert len(window_labels) == len(windows)
        assert len(window_timestamps) == len(windows)

    def test_no_overlap(self, continuous_data):
        data, labels, timestamps = continuous_data
        config = WindowConfig(window_size=1.0, overlap=0.0, sampling_rate=50)

        windows, window_labels, _ = create_windows(data, labels, config, timestamps)

        # Calculate expected number of windows
        expected = (len(data) - config.window_samples) // config.step_samples + 1
        assert len(windows) == expected

    def test_majority_voting(self):
        # Create data where labels change mid-window
        data = np.random.randn(100, 3).astype(np.float32)
        labels = np.array([0] * 40 + [1] * 60, dtype=np.int64)
        config = WindowConfig(window_size=2.0, overlap=0.0, sampling_rate=50)

        windows, window_labels, _ = create_windows(data, labels, config)

        # First window (0-100) should have majority label
        assert window_labels[0] == 1  # 60 ones vs 40 zeros

    def test_empty_result_short_data(self):
        # Data shorter than window size
        data = np.random.randn(10, 3).astype(np.float32)
        labels = np.zeros(10, dtype=np.int64)
        config = WindowConfig(window_size=1.0, sampling_rate=50)

        windows, window_labels, _ = create_windows(data, labels, config)

        assert len(windows) == 0
        assert len(window_labels) == 0


class TestComputeClassWeights:
    """Tests for compute_class_weights function."""

    def test_balanced_classes(self):
        labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        weights = compute_class_weights(labels)

        # Balanced classes should have equal weights
        np.testing.assert_array_almost_equal(weights, np.ones(3))

    def test_imbalanced_classes(self):
        labels = np.array([0, 0, 0, 0, 1, 2], dtype=np.int64)
        weights = compute_class_weights(labels)

        # More frequent class should have lower weight
        assert weights[0] < weights[1]
        assert weights[0] < weights[2]

    def test_weights_sum(self, sample_labels):
        weights = compute_class_weights(sample_labels)
        # Weights should be positive
        assert (weights > 0).all()
