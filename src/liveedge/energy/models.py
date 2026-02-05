"""Energy consumption models for adaptive sampling systems.

This module provides energy modeling for edge devices performing
behavior classification with adaptive sampling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from liveedge.clustering.clustering import BehaviorCluster
from liveedge.energy.hardware import HardwareSpec


@dataclass
class EnergyBreakdown:
    """Breakdown of energy consumption by component.

    Attributes:
        sensing_mj: Energy consumed by sensing (mJ).
        computation_mj: Energy consumed by computation (mJ).
        transmission_mj: Energy consumed by transmission (mJ).
        idle_mj: Energy consumed in idle/sleep mode (mJ).
        total_mj: Total energy consumption (mJ).
    """

    sensing_mj: float = 0.0
    computation_mj: float = 0.0
    transmission_mj: float = 0.0
    idle_mj: float = 0.0

    @property
    def total_mj(self) -> float:
        """Total energy in millijoules."""
        return self.sensing_mj + self.computation_mj + self.transmission_mj + self.idle_mj

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "sensing_mj": self.sensing_mj,
            "computation_mj": self.computation_mj,
            "transmission_mj": self.transmission_mj,
            "idle_mj": self.idle_mj,
            "total_mj": self.total_mj,
        }


class EnergyModel:
    """Energy consumption model for adaptive sampling system.

    Computes energy consumption for sensing, computation, and transmission
    based on hardware specifications and sampling patterns.

    Example:
        >>> hw = HardwareSpec()
        >>> model = EnergyModel(hw)
        >>> sensing_energy = model.compute_sensing_energy(sampling_rate=25, duration=3600)
        >>> print(f"Sensing energy: {sensing_energy:.2f} mJ")
    """

    def __init__(self, hardware: HardwareSpec):
        """Initialize the energy model.

        Args:
            hardware: Hardware specifications.
        """
        self.hw = hardware

    def compute_sensing_energy(
        self,
        sampling_rate: float,
        duration: float,
    ) -> float:
        """Compute sensing energy in millijoules.

        Args:
            sampling_rate: Sampling rate in Hz.
            duration: Duration in seconds.

        Returns:
            Energy consumption in millijoules.
        """
        n_samples = int(sampling_rate * duration)

        # Time spent active per sample (us to s)
        sample_time_s = self.hw.sensor.sample_time_us / 1e6
        active_time = n_samples * sample_time_s

        # Time spent sleeping
        sleep_time = duration - active_time

        # Energy calculation
        active_energy_mj = active_time * self.hw.sensor.active_power_mw
        sleep_energy_mj = sleep_time * (self.hw.sensor.sleep_power_uw / 1000)

        return active_energy_mj + sleep_energy_mj

    def compute_computation_energy(
        self,
        model_name: str,
        n_inferences: int,
    ) -> float:
        """Compute computation energy in millijoules.

        Args:
            model_name: Name of the ML model.
            n_inferences: Number of inference operations.

        Returns:
            Energy consumption in millijoules.
        """
        inference_time_ms = self.hw.inference_time_ms.get(model_name, 3.0)
        inference_time_s = inference_time_ms / 1000

        # Total active time for inference
        total_inference_time = n_inferences * inference_time_s

        # Energy = power * time
        return total_inference_time * self.hw.mcu.active_power_mw

    def compute_transmission_energy(
        self,
        data_bytes: int,
    ) -> float:
        """Compute BLE transmission energy in millijoules.

        Args:
            data_bytes: Number of bytes to transmit.

        Returns:
            Energy consumption in millijoules.
        """
        # Convert bytes to bits
        data_bits = data_bytes * 8

        # Transmission time in seconds
        tx_time_s = data_bits / (self.hw.radio.data_rate_kbps * 1000)

        # Energy = power * time
        return tx_time_s * self.hw.radio.tx_power_mw

    def compute_idle_energy(
        self,
        duration: float,
        active_fraction: float = 0.0,
    ) -> float:
        """Compute idle/sleep energy in millijoules.

        Args:
            duration: Total duration in seconds.
            active_fraction: Fraction of time spent active (0-1).

        Returns:
            Energy consumption in millijoules.
        """
        idle_time = duration * (1 - active_fraction)
        return idle_time * self.hw.mcu.idle_power_mw

    def compute_total_energy(
        self,
        sampling_log: list[tuple[float, float, BehaviorCluster | None]],
        model_name: str,
        window_duration: float = 1.5,
        bytes_per_window: int = 12,
    ) -> EnergyBreakdown:
        """Compute total energy breakdown.

        Args:
            sampling_log: List of (timestamp, sampling_rate, behavior_state) tuples.
            model_name: ML model name for inference timing.
            window_duration: Window duration in seconds.
            bytes_per_window: Bytes transmitted per classification window.

        Returns:
            EnergyBreakdown with component-wise energy.
        """
        if not sampling_log:
            return EnergyBreakdown()

        total_sensing = 0.0
        total_computation = 0.0
        total_transmission = 0.0
        total_idle = 0.0

        n_windows = 0

        for i, (time, rate, _) in enumerate(sampling_log):
            # Determine duration of this segment
            if i < len(sampling_log) - 1:
                segment_duration = sampling_log[i + 1][0] - time
            else:
                segment_duration = window_duration

            # Sensing energy for this segment
            total_sensing += self.compute_sensing_energy(rate, segment_duration)

            # Count windows in this segment
            n_windows += segment_duration / window_duration

        # Computation energy (one inference per window)
        total_computation = self.compute_computation_energy(model_name, int(n_windows))

        # Transmission energy (classification result per window)
        total_transmission = self.compute_transmission_energy(int(n_windows) * bytes_per_window)

        # Total duration
        if sampling_log:
            total_duration = sampling_log[-1][0] - sampling_log[0][0] + window_duration
            # Idle energy (rough estimate)
            active_fraction = (total_sensing / self.hw.sensor.active_power_mw) / total_duration
            total_idle = self.compute_idle_energy(total_duration, active_fraction)

        return EnergyBreakdown(
            sensing_mj=total_sensing,
            computation_mj=total_computation,
            transmission_mj=total_transmission,
            idle_mj=total_idle,
        )

    def estimate_battery_life_hours(
        self,
        avg_power_mw: float,
    ) -> float:
        """Estimate battery life given average power consumption.

        Args:
            avg_power_mw: Average power consumption in mW.

        Returns:
            Estimated battery life in hours.
        """
        if avg_power_mw <= 0:
            return float("inf")
        return self.hw.battery.capacity_mwh / avg_power_mw

    def compute_avg_power(
        self,
        energy_breakdown: EnergyBreakdown,
        duration: float,
    ) -> float:
        """Compute average power consumption.

        Args:
            energy_breakdown: Energy breakdown from compute_total_energy.
            duration: Total duration in seconds.

        Returns:
            Average power in mW.
        """
        if duration <= 0:
            return 0.0
        return energy_breakdown.total_mj / duration


def compute_energy_reduction(
    adaptive_energy: EnergyBreakdown,
    baseline_energy: EnergyBreakdown,
) -> float:
    """Compute energy reduction ratio.

    Args:
        adaptive_energy: Energy with adaptive sampling.
        baseline_energy: Energy with fixed-rate sampling.

    Returns:
        Reduction ratio (1 - adaptive/baseline).
    """
    if baseline_energy.total_mj <= 0:
        return 0.0
    return 1 - (adaptive_energy.total_mj / baseline_energy.total_mj)


def estimate_sampling_rate_from_energy_budget(
    energy_model: EnergyModel,
    energy_budget_mj: float,
    duration: float,
    model_name: str = "random_forest",
) -> float:
    """Estimate maximum sampling rate for a given energy budget.

    Args:
        energy_model: Energy model to use.
        energy_budget_mj: Energy budget in millijoules.
        duration: Duration in seconds.
        model_name: ML model name.

    Returns:
        Maximum sustainable sampling rate in Hz.
    """
    # Binary search for sustainable rate
    min_rate = 1.0
    max_rate = 100.0

    for _ in range(20):  # 20 iterations for precision
        mid_rate = (min_rate + max_rate) / 2

        # Create dummy sampling log
        n_windows = int(duration / 1.5)
        sampling_log = [(i * 1.5, mid_rate, None) for i in range(n_windows)]

        energy = energy_model.compute_total_energy(sampling_log, model_name)

        if energy.total_mj <= energy_budget_mj:
            min_rate = mid_rate
        else:
            max_rate = mid_rate

    return min_rate
