"""Tests for energy modeling and estimation."""

from __future__ import annotations

import numpy as np
import pytest

from liveedge.clustering import BehaviorCluster
from liveedge.energy import EnergyModel, HardwareSpec


class TestHardwareSpec:
    """Tests for HardwareSpec."""

    def test_default_spec(self):
        """Test default hardware specification."""
        spec = HardwareSpec()

        assert spec.sensor_power_mw > 0
        assert spec.mcu_active_power_mw > 0
        assert spec.mcu_sleep_power_mw > 0
        assert spec.radio_tx_power_mw > 0
        assert spec.battery_capacity_mah > 0

    def test_custom_spec(self):
        """Test custom hardware specification."""
        spec = HardwareSpec(
            sensor_power_mw=0.5,
            mcu_active_power_mw=5.0,
            mcu_sleep_power_mw=0.01,
            radio_tx_power_mw=20.0,
            battery_capacity_mah=500,
            battery_voltage=3.3,
        )

        assert spec.sensor_power_mw == 0.5
        assert spec.mcu_active_power_mw == 5.0
        assert spec.battery_capacity_mah == 500
        assert spec.battery_voltage == 3.3

    def test_from_dict(self):
        """Test creating spec from dictionary."""
        spec_dict = {
            "sensor_power_mw": 1.0,
            "mcu_active_power_mw": 10.0,
            "mcu_sleep_power_mw": 0.02,
            "radio_tx_power_mw": 30.0,
            "battery_capacity_mah": 1000,
            "battery_voltage": 3.7,
        }

        spec = HardwareSpec.from_dict(spec_dict)

        assert spec.sensor_power_mw == 1.0
        assert spec.mcu_active_power_mw == 10.0
        assert spec.battery_capacity_mah == 1000

    def test_battery_capacity_wh(self):
        """Test battery capacity in Wh."""
        spec = HardwareSpec(
            battery_capacity_mah=1000,
            battery_voltage=3.7,
        )

        expected_wh = 1000 * 3.7 / 1000  # mAh * V / 1000 = Wh
        assert abs(spec.battery_capacity_wh - expected_wh) < 0.001


class TestEnergyModel:
    """Tests for EnergyModel."""

    @pytest.fixture
    def energy_model(self):
        """Create an energy model with default specs."""
        spec = HardwareSpec()
        return EnergyModel(spec)

    @pytest.fixture
    def custom_energy_model(self):
        """Create energy model with custom specs."""
        spec = HardwareSpec(
            sensor_power_mw=1.0,
            mcu_active_power_mw=10.0,
            mcu_sleep_power_mw=0.01,
            radio_tx_power_mw=25.0,
            battery_capacity_mah=500,
            battery_voltage=3.7,
        )
        return EnergyModel(spec)

    def test_sensing_energy(self, energy_model):
        """Test sensing energy computation."""
        # 50 Hz for 1 second
        energy = energy_model.compute_sensing_energy(
            sampling_rate=50,
            duration_seconds=1.0,
        )

        assert energy > 0
        assert isinstance(energy, float)

    def test_sensing_energy_rate_dependency(self, energy_model):
        """Test sensing energy increases with rate."""
        energy_low = energy_model.compute_sensing_energy(
            sampling_rate=10,
            duration_seconds=1.0,
        )
        energy_high = energy_model.compute_sensing_energy(
            sampling_rate=50,
            duration_seconds=1.0,
        )

        assert energy_high > energy_low

    def test_computation_energy(self, energy_model):
        """Test computation energy for inference."""
        energy = energy_model.compute_computation_energy(
            model_name="random_forest",
            n_samples=100,
        )

        assert energy > 0

    def test_computation_energy_model_dependency(self, energy_model):
        """Test different models have different energy."""
        energy_rf = energy_model.compute_computation_energy(
            model_name="random_forest",
            n_samples=100,
        )
        energy_tcn = energy_model.compute_computation_energy(
            model_name="tcn",
            n_samples=100,
        )

        # TCN should use more energy than Random Forest
        assert energy_tcn > energy_rf

    def test_transmission_energy(self, energy_model):
        """Test transmission energy computation."""
        energy = energy_model.compute_transmission_energy(
            data_bytes=1000,
            tx_duration_seconds=0.1,
        )

        assert energy > 0

    def test_total_energy(self, energy_model):
        """Test total energy computation."""
        total = energy_model.compute_total_energy(
            sampling_rate=50,
            duration_seconds=60.0,
            model_name="random_forest",
            n_inferences=60,
            tx_bytes=1000,
        )

        assert total > 0

        # Total should be sum of components
        sensing = energy_model.compute_sensing_energy(50, 60.0)
        computation = energy_model.compute_computation_energy("random_forest", 60)
        transmission = energy_model.compute_transmission_energy(1000, 0.1)

        # Allow some tolerance for additional overhead
        assert total >= sensing + computation

    def test_battery_lifetime_estimation(self, custom_energy_model):
        """Test battery lifetime estimation."""
        # Average power consumption
        avg_power_mw = 5.0  # 5mW average

        lifetime_hours = custom_energy_model.estimate_battery_life_hours(
            avg_power_mw=avg_power_mw,
        )

        assert lifetime_hours > 0

        # Expected: 500mAh * 3.7V / 5mW = ~370 hours
        expected = 500 * 3.7 / 5.0
        assert abs(lifetime_hours - expected) < 10  # Within 10 hours

    def test_energy_reduction_ratio(self, energy_model):
        """Test energy reduction ratio computation."""
        baseline_rate = 50
        adaptive_rates = [5, 10, 15, 25, 50]  # Average ~21 Hz

        baseline_energy = energy_model.compute_sensing_energy(baseline_rate, 1.0)

        adaptive_energy = sum(
            energy_model.compute_sensing_energy(rate, 0.2)
            for rate in adaptive_rates
        )

        reduction = 1 - (adaptive_energy / baseline_energy)

        assert 0 < reduction < 1


class TestEnergyMetrics:
    """Tests for energy metrics computation."""

    def test_compute_energy_metrics(self):
        """Test energy metrics computation from sampling log."""
        from liveedge.evaluation import compute_energy_metrics

        spec = HardwareSpec()
        energy_model = EnergyModel(spec)

        # Create sample sampling log
        # Format: (timestamp, rate, state)
        sampling_log = [
            (0.0, 5.0, BehaviorCluster.INACTIVE),
            (1.0, 10.0, BehaviorCluster.RUMINATING),
            (2.0, 15.0, BehaviorCluster.FEEDING),
            (3.0, 25.0, BehaviorCluster.LOCOMOTION),
            (4.0, 50.0, BehaviorCluster.HIGH_ACTIVITY),
        ]

        metrics = compute_energy_metrics(
            sampling_log=sampling_log,
            energy_model=energy_model,
            model_name="random_forest",
            baseline_rate=50.0,
        )

        assert metrics.total_energy_mj > 0
        assert metrics.avg_sampling_rate > 0
        assert 0 <= metrics.energy_reduction_ratio <= 1
        assert metrics.estimated_battery_hours > 0

    def test_energy_metrics_with_fixed_rate(self):
        """Test that fixed rate gives zero reduction."""
        from liveedge.evaluation import compute_energy_metrics

        spec = HardwareSpec()
        energy_model = EnergyModel(spec)

        # All at 50 Hz
        sampling_log = [
            (float(i), 50.0, BehaviorCluster.INACTIVE)
            for i in range(10)
        ]

        metrics = compute_energy_metrics(
            sampling_log=sampling_log,
            energy_model=energy_model,
            model_name="random_forest",
            baseline_rate=50.0,
        )

        # No reduction when using baseline rate
        assert metrics.energy_reduction_ratio < 0.01  # ~0

    def test_energy_metrics_with_low_rate(self):
        """Test that low rate gives high reduction."""
        from liveedge.evaluation import compute_energy_metrics

        spec = HardwareSpec()
        energy_model = EnergyModel(spec)

        # All at 5 Hz
        sampling_log = [
            (float(i), 5.0, BehaviorCluster.INACTIVE)
            for i in range(10)
        ]

        metrics = compute_energy_metrics(
            sampling_log=sampling_log,
            energy_model=energy_model,
            model_name="random_forest",
            baseline_rate=50.0,
        )

        # Significant reduction when using 5 Hz vs 50 Hz
        assert metrics.energy_reduction_ratio > 0.5


class TestEnergyModelProfiles:
    """Tests for model-specific energy profiles."""

    @pytest.fixture
    def energy_model(self):
        """Create energy model."""
        spec = HardwareSpec()
        return EnergyModel(spec)

    def test_random_forest_profile(self, energy_model):
        """Test Random Forest has low energy."""
        energy = energy_model.compute_computation_energy("random_forest", 100)
        assert energy > 0

    def test_xgboost_profile(self, energy_model):
        """Test XGBoost energy."""
        energy = energy_model.compute_computation_energy("xgboost", 100)
        assert energy > 0

    def test_cnn_profile(self, energy_model):
        """Test CNN has higher energy."""
        energy_cnn = energy_model.compute_computation_energy("cnn1d", 100)
        energy_rf = energy_model.compute_computation_energy("random_forest", 100)

        assert energy_cnn > energy_rf

    def test_tcn_profile(self, energy_model):
        """Test TCN has highest energy."""
        energy_tcn = energy_model.compute_computation_energy("tcn", 100)
        energy_cnn = energy_model.compute_computation_energy("cnn1d", 100)

        assert energy_tcn >= energy_cnn

    def test_unknown_model_fallback(self, energy_model):
        """Test unknown model uses default profile."""
        energy = energy_model.compute_computation_energy("unknown_model", 100)
        # Should not raise, uses default
        assert energy > 0


class TestBatteryLifeEstimation:
    """Tests for battery life estimation."""

    def test_typical_deployment_scenario(self):
        """Test battery life for typical deployment."""
        spec = HardwareSpec(
            sensor_power_mw=0.5,
            mcu_active_power_mw=5.0,
            mcu_sleep_power_mw=0.01,
            radio_tx_power_mw=25.0,
            battery_capacity_mah=2000,
            battery_voltage=3.7,
        )
        model = EnergyModel(spec)

        # Assume active 10% of time
        duty_cycle = 0.1
        avg_power = (
            spec.mcu_active_power_mw * duty_cycle
            + spec.mcu_sleep_power_mw * (1 - duty_cycle)
            + spec.sensor_power_mw * duty_cycle
        )

        lifetime = model.estimate_battery_life_hours(avg_power)

        # Should be many hundreds of hours
        assert lifetime > 100

    def test_high_activity_scenario(self):
        """Test battery life with high activity."""
        spec = HardwareSpec(
            battery_capacity_mah=1000,
            battery_voltage=3.7,
        )
        model = EnergyModel(spec)

        # High power consumption
        avg_power_mw = 50.0

        lifetime = model.estimate_battery_life_hours(avg_power_mw)

        # ~74 hours
        expected = 1000 * 3.7 / 50.0
        assert abs(lifetime - expected) < 1
