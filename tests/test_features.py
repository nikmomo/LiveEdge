"""Tests for feature extraction module."""

from __future__ import annotations

import numpy as np
import pytest

from liveedge.data.features import (
    compute_magnitude,
    extract_all_features,
    extract_dynamics_features,
    extract_features_batch,
    extract_frequency_domain_features,
    extract_periodicity_features,
    extract_statistical_features,
    extract_time_domain_features,
    get_feature_names,
)


class TestComputeMagnitude:
    """Tests for compute_magnitude function."""

    def test_basic_magnitude(self):
        data = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        mag = compute_magnitude(data)
        np.testing.assert_array_almost_equal(mag, [1.0, 1.0, 1.0])

    def test_3d_vector(self):
        data = np.array([[3, 4, 0]], dtype=np.float32)
        mag = compute_magnitude(data)
        np.testing.assert_almost_equal(mag[0], 5.0)

    def test_shape(self, sample_window):
        mag = compute_magnitude(sample_window)
        assert mag.shape == (sample_window.shape[0],)


class TestTimeDomainFeatures:
    """Tests for extract_time_domain_features function."""

    def test_output_keys(self, sample_window):
        features = extract_time_domain_features(sample_window)

        # Check required keys
        assert "acc_magnitude_mean" in features
        assert "acc_magnitude_std" in features
        assert "acc_magnitude_max" in features
        assert "acc_magnitude_min" in features
        assert "acc_magnitude_range" in features
        assert "zero_crossing_rate" in features
        assert "signal_magnitude_area" in features
        assert "rms" in features
        assert "energy" in features

    def test_magnitude_positive(self, sample_window):
        features = extract_time_domain_features(sample_window)
        assert features["acc_magnitude_mean"] > 0
        assert features["acc_magnitude_std"] >= 0
        assert features["rms"] > 0
        assert features["energy"] > 0

    def test_constant_signal(self, constant_window):
        features = extract_time_domain_features(constant_window)

        # Std should be zero for constant signal
        assert features["acc_magnitude_std"] == pytest.approx(0, abs=1e-6)
        assert features["zero_crossing_rate"] == pytest.approx(0, abs=1e-6)

    def test_per_axis_features(self, sample_window):
        features = extract_time_domain_features(sample_window)

        for axis in ["x", "y", "z"]:
            assert f"acc_{axis}_mean" in features
            assert f"acc_{axis}_std" in features
            assert f"acc_{axis}_max" in features
            assert f"acc_{axis}_min" in features


class TestFrequencyDomainFeatures:
    """Tests for extract_frequency_domain_features function."""

    def test_output_keys(self, sample_window):
        features = extract_frequency_domain_features(sample_window, fs=50)

        assert "dominant_frequency" in features
        assert "spectral_entropy" in features
        assert "spectral_centroid" in features
        assert "high_freq_ratio" in features
        assert "spectral_rolloff" in features
        assert "spectral_flatness" in features

    def test_dominant_frequency_detection(self, periodic_window):
        features = extract_frequency_domain_features(periodic_window, fs=50)

        # Should detect ~2 Hz dominant frequency
        assert features["dominant_frequency"] == pytest.approx(2.0, abs=1.0)

    def test_spectral_entropy_bounds(self, sample_window):
        features = extract_frequency_domain_features(sample_window, fs=50)

        # Normalized entropy should be between 0 and 1
        assert 0 <= features["spectral_entropy"] <= 1

    def test_band_powers_sum(self, sample_window):
        features = extract_frequency_domain_features(sample_window, fs=50)

        # Band powers should be non-negative
        bands = ["band_power_0_2", "band_power_2_5", "band_power_5_10", "band_power_10_25"]
        for band in bands:
            assert features[band] >= 0

    def test_high_activity_high_freq_ratio(self, high_activity_window):
        features = extract_frequency_domain_features(high_activity_window, fs=50)

        # High activity should have higher high-frequency content
        assert features["high_freq_ratio"] > 0.1


class TestDynamicsFeatures:
    """Tests for extract_dynamics_features function."""

    def test_output_keys(self, sample_window):
        features = extract_dynamics_features(sample_window, fs=50)

        assert "jerk_mean" in features
        assert "jerk_std" in features
        assert "jerk_max" in features
        assert "acceleration_change_rate" in features

    def test_jerk_positive(self, sample_window):
        features = extract_dynamics_features(sample_window, fs=50)

        assert features["jerk_mean"] >= 0
        assert features["jerk_max"] >= 0

    def test_constant_signal_low_jerk(self, constant_window):
        features = extract_dynamics_features(constant_window, fs=50)

        # Constant signal should have zero jerk
        assert features["jerk_mean"] == pytest.approx(0, abs=1e-6)
        assert features["jerk_max"] == pytest.approx(0, abs=1e-6)

    def test_high_activity_high_jerk(self, high_activity_window):
        low_activity = np.sin(np.linspace(0, 2 * np.pi, 75))[:, None] * np.ones((1, 3))
        low_activity = low_activity.astype(np.float32)

        features_high = extract_dynamics_features(high_activity_window, fs=50)
        features_low = extract_dynamics_features(low_activity, fs=50)

        # High activity should have higher jerk
        assert features_high["jerk_mean"] > features_low["jerk_mean"]


class TestPeriodicityFeatures:
    """Tests for extract_periodicity_features function."""

    def test_output_keys(self, sample_window):
        features = extract_periodicity_features(sample_window, fs=50)

        assert "autocorr_peak_value" in features
        assert "autocorr_peak_lag" in features
        assert "periodicity_strength" in features
        assert "autocorr_peak_freq" in features

    def test_periodic_signal_detection(self, periodic_window):
        features = extract_periodicity_features(periodic_window, fs=50)

        # 2 Hz signal should have peak around 25 samples (0.5 second period)
        assert features["autocorr_peak_value"] > 0.5
        assert features["autocorr_peak_freq"] == pytest.approx(2.0, abs=0.5)

    def test_random_signal_low_periodicity(self, sample_window):
        features = extract_periodicity_features(sample_window, fs=50)

        # Random signal should have low periodicity
        assert features["periodicity_strength"] < 5.0


class TestStatisticalFeatures:
    """Tests for extract_statistical_features function."""

    def test_output_keys(self, sample_window):
        features = extract_statistical_features(sample_window)

        assert "skewness_magnitude" in features
        assert "kurtosis_magnitude" in features
        assert "correlation_xy" in features
        assert "correlation_xz" in features
        assert "correlation_yz" in features

    def test_correlation_bounds(self, sample_window):
        features = extract_statistical_features(sample_window)

        # Correlations should be between -1 and 1
        for key in ["correlation_xy", "correlation_xz", "correlation_yz"]:
            assert -1 <= features[key] <= 1


class TestExtractAllFeatures:
    """Tests for extract_all_features function."""

    def test_all_feature_types_included(self, sample_window):
        features = extract_all_features(sample_window, fs=50)

        # Check that all feature types are present
        # Time domain
        assert "acc_magnitude_mean" in features
        assert "rms" in features

        # Frequency domain
        assert "dominant_frequency" in features
        assert "spectral_entropy" in features

        # Dynamics
        assert "jerk_mean" in features

        # Periodicity
        assert "autocorr_peak_value" in features

        # Statistical
        assert "skewness_magnitude" in features

    def test_no_nan_or_inf(self, sample_window):
        features = extract_all_features(sample_window, fs=50)

        for name, value in features.items():
            assert not np.isnan(value), f"{name} is NaN"
            assert not np.isinf(value), f"{name} is inf"

    def test_consistent_feature_count(self, sample_window, periodic_window):
        features1 = extract_all_features(sample_window, fs=50)
        features2 = extract_all_features(periodic_window, fs=50)

        assert len(features1) == len(features2)
        assert set(features1.keys()) == set(features2.keys())


class TestExtractFeaturesBatch:
    """Tests for extract_features_batch function."""

    def test_batch_extraction(self, sample_windows):
        feature_matrix = extract_features_batch(sample_windows, fs=50)

        # Check shape
        assert feature_matrix.shape[0] == len(sample_windows)
        assert feature_matrix.shape[1] > 0

    def test_no_nan_in_batch(self, sample_windows):
        feature_matrix = extract_features_batch(sample_windows, fs=50)
        assert not np.isnan(feature_matrix).any()


class TestGetFeatureNames:
    """Tests for get_feature_names function."""

    def test_returns_list(self):
        names = get_feature_names(fs=50)
        assert isinstance(names, list)
        assert len(names) > 0

    def test_all_strings(self):
        names = get_feature_names(fs=50)
        assert all(isinstance(name, str) for name in names)

    def test_sorted(self):
        names = get_feature_names(fs=50)
        assert names == sorted(names)
