"""Energy efficiency evaluation metrics.

This module provides metrics for evaluating energy efficiency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from liveedge.clustering.clustering import BehaviorCluster
from liveedge.energy.models import EnergyBreakdown, EnergyModel


@dataclass
class EnergyMetrics:
    """Energy efficiency metrics.

    Attributes:
        avg_sampling_rate: Average sampling rate in Hz.
        energy_total_mj: Total energy consumption in mJ.
        energy_breakdown: Energy breakdown by component.
        energy_reduction_ratio: Reduction compared to baseline (0-1).
        estimated_battery_hours: Estimated battery life in hours.
        switching_rate_per_hour: Rate changes per hour.
        avg_power_mw: Average power consumption in mW.
    """

    avg_sampling_rate: float
    energy_total_mj: float
    energy_breakdown: dict[str, float]
    energy_reduction_ratio: float
    estimated_battery_hours: float
    switching_rate_per_hour: float
    avg_power_mw: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "avg_sampling_rate": self.avg_sampling_rate,
            "energy_total_mj": self.energy_total_mj,
            "energy_breakdown": self.energy_breakdown,
            "energy_reduction_ratio": self.energy_reduction_ratio,
            "estimated_battery_hours": self.estimated_battery_hours,
            "switching_rate_per_hour": self.switching_rate_per_hour,
            "avg_power_mw": self.avg_power_mw,
        }

    def summary(self) -> str:
        """Get summary string."""
        return (
            f"Avg Sampling Rate: {self.avg_sampling_rate:.1f} Hz\n"
            f"Energy Reduction: {self.energy_reduction_ratio*100:.1f}%\n"
            f"Battery Life: {self.estimated_battery_hours:.1f} hours\n"
            f"Avg Power: {self.avg_power_mw:.2f} mW"
        )


def compute_energy_metrics(
    sampling_log: list[tuple[float, float, BehaviorCluster | None]],
    energy_model: EnergyModel,
    model_name: str = "random_forest",
    baseline_rate: float = 50.0,
) -> EnergyMetrics:
    """Compute energy efficiency metrics.

    Args:
        sampling_log: List of (timestamp, sampling_rate, state) tuples.
        energy_model: Energy model for calculations.
        model_name: ML model name for inference timing.
        baseline_rate: Baseline fixed sampling rate for comparison.

    Returns:
        EnergyMetrics with computed values.
    """
    if not sampling_log:
        return EnergyMetrics(
            avg_sampling_rate=0.0,
            energy_total_mj=0.0,
            energy_breakdown={},
            energy_reduction_ratio=0.0,
            estimated_battery_hours=0.0,
            switching_rate_per_hour=0.0,
            avg_power_mw=0.0,
        )

    # Calculate total duration and average sampling rate
    start_time = sampling_log[0][0]
    end_time = sampling_log[-1][0]
    total_duration = end_time - start_time

    if total_duration <= 0:
        total_duration = 1.5  # Assume one window

    # Weighted average sampling rate
    total_weighted = 0.0
    for i, (time, rate, _) in enumerate(sampling_log):
        if i < len(sampling_log) - 1:
            segment_duration = sampling_log[i + 1][0] - time
        else:
            segment_duration = 1.5  # Window duration
        total_weighted += rate * segment_duration

    avg_sampling_rate = total_weighted / total_duration

    # Compute energy breakdown
    energy_breakdown = energy_model.compute_total_energy(
        sampling_log, model_name, window_duration=1.5
    )

    # Compute baseline energy
    baseline_log = [(sampling_log[0][0], baseline_rate, None)]
    baseline_breakdown = energy_model.compute_total_energy(
        [(t, baseline_rate, s) for t, _, s in sampling_log],
        model_name,
        window_duration=1.5,
    )

    # Energy reduction
    if baseline_breakdown.total_mj > 0:
        energy_reduction_ratio = 1 - (energy_breakdown.total_mj / baseline_breakdown.total_mj)
    else:
        energy_reduction_ratio = 0.0

    # Average power
    avg_power_mw = energy_breakdown.total_mj / total_duration

    # Battery life
    estimated_battery_hours = energy_model.estimate_battery_life_hours(avg_power_mw)

    # Switching rate
    n_switches = count_rate_switches(sampling_log)
    switching_rate_per_hour = n_switches * 3600 / total_duration if total_duration > 0 else 0

    return EnergyMetrics(
        avg_sampling_rate=avg_sampling_rate,
        energy_total_mj=energy_breakdown.total_mj,
        energy_breakdown=energy_breakdown.to_dict(),
        energy_reduction_ratio=energy_reduction_ratio,
        estimated_battery_hours=estimated_battery_hours,
        switching_rate_per_hour=switching_rate_per_hour,
        avg_power_mw=avg_power_mw,
    )


def count_rate_switches(
    sampling_log: list[tuple[float, float, BehaviorCluster | None]],
) -> int:
    """Count number of sampling rate changes.

    Args:
        sampling_log: List of (timestamp, sampling_rate, state) tuples.

    Returns:
        Number of rate changes.
    """
    if len(sampling_log) < 2:
        return 0

    switches = 0
    prev_rate = sampling_log[0][1]

    for _, rate, _ in sampling_log[1:]:
        if rate != prev_rate:
            switches += 1
            prev_rate = rate

    return switches


def compute_sampling_efficiency(
    sampling_log: list[tuple[float, float, BehaviorCluster | None]],
    optimal_rates: dict[BehaviorCluster, float],
) -> float:
    """Compute how close actual sampling rates are to optimal.

    Args:
        sampling_log: List of (timestamp, sampling_rate, state) tuples.
        optimal_rates: Optimal sampling rate per behavior.

    Returns:
        Efficiency score (0-1, higher is better).
    """
    if not sampling_log:
        return 1.0

    total_error = 0.0
    total_weight = 0.0

    for i, (time, rate, state) in enumerate(sampling_log):
        if state is None:
            continue

        # Get segment duration
        if i < len(sampling_log) - 1:
            segment_duration = sampling_log[i + 1][0] - time
        else:
            segment_duration = 1.5

        optimal_rate = optimal_rates.get(state, 50.0)

        # Compute relative error (penalize under-sampling more than over-sampling)
        if rate < optimal_rate:
            error = (optimal_rate - rate) / optimal_rate
        else:
            error = (rate - optimal_rate) / optimal_rate * 0.5  # Less penalty for over-sampling

        total_error += error * segment_duration
        total_weight += segment_duration

    if total_weight <= 0:
        return 1.0

    avg_error = total_error / total_weight
    return max(0.0, 1.0 - avg_error)
