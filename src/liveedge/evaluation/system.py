"""End-to-end system evaluation metrics.

This module provides metrics for evaluating the complete adaptive sampling system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from liveedge.evaluation.classification import ClassificationMetrics
from liveedge.evaluation.energy import EnergyMetrics
from liveedge.evaluation.reconstruction import ReconstructionMetrics


@dataclass
class SystemMetrics:
    """Combined system evaluation metrics.

    Attributes:
        classification: Classification performance metrics.
        energy: Energy efficiency metrics.
        reconstruction: Signal reconstruction metrics (if applicable).
        overall_score: Combined performance score.
    """

    classification: ClassificationMetrics
    energy: EnergyMetrics
    reconstruction: ReconstructionMetrics | None = None
    overall_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "classification": self.classification.to_dict(),
            "energy": self.energy.to_dict(),
            "overall_score": self.overall_score,
        }
        if self.reconstruction is not None:
            result["reconstruction"] = self.reconstruction.to_dict()
        return result

    def summary(self) -> str:
        """Get summary string."""
        lines = [
            "=== Classification ===",
            self.classification.summary(),
            "",
            "=== Energy ===",
            self.energy.summary(),
        ]
        if self.reconstruction is not None:
            lines.extend(["", "=== Reconstruction ===", self.reconstruction.summary()])
        lines.extend(["", f"Overall Score: {self.overall_score:.4f}"])
        return "\n".join(lines)


def compute_overall_score(
    classification: ClassificationMetrics,
    energy: EnergyMetrics,
    reconstruction: ReconstructionMetrics | None = None,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute overall system performance score.

    Args:
        classification: Classification metrics.
        energy: Energy metrics.
        reconstruction: Optional reconstruction metrics.
        weights: Weights for different components.

    Returns:
        Overall score (0-1, higher is better).
    """
    if weights is None:
        weights = {
            "accuracy": 0.3,
            "f1": 0.2,
            "energy_reduction": 0.3,
            "battery_life": 0.1,
            "reconstruction": 0.1,
        }

    score = 0.0

    # Classification component
    score += weights.get("accuracy", 0.3) * classification.accuracy
    score += weights.get("f1", 0.2) * classification.macro_f1

    # Energy component
    energy_score = min(1.0, energy.energy_reduction_ratio)  # Cap at 1.0
    score += weights.get("energy_reduction", 0.3) * energy_score

    # Battery life (normalize to 0-1 based on 24-hour target)
    battery_score = min(1.0, energy.estimated_battery_hours / 168)  # 1 week = 1.0
    score += weights.get("battery_life", 0.1) * battery_score

    # Reconstruction component
    if reconstruction is not None:
        recon_score = reconstruction.correlation
        score += weights.get("reconstruction", 0.1) * recon_score
    else:
        # Redistribute weight
        score += weights.get("reconstruction", 0.1) * classification.accuracy

    return score


def create_system_metrics(
    classification: ClassificationMetrics,
    energy: EnergyMetrics,
    reconstruction: ReconstructionMetrics | None = None,
    weights: dict[str, float] | None = None,
) -> SystemMetrics:
    """Create combined system metrics.

    Args:
        classification: Classification metrics.
        energy: Energy metrics.
        reconstruction: Optional reconstruction metrics.
        weights: Weights for overall score.

    Returns:
        SystemMetrics with all components and overall score.
    """
    overall = compute_overall_score(classification, energy, reconstruction, weights)

    return SystemMetrics(
        classification=classification,
        energy=energy,
        reconstruction=reconstruction,
        overall_score=overall,
    )


def compare_systems(
    systems: dict[str, SystemMetrics],
) -> dict[str, dict[str, Any]]:
    """Compare multiple system configurations.

    Args:
        systems: Dictionary mapping system name to metrics.

    Returns:
        Comparison dictionary with rankings and differences.
    """
    if not systems:
        return {}

    comparison = {}

    # Extract key metrics for comparison
    for name, metrics in systems.items():
        comparison[name] = {
            "accuracy": metrics.classification.accuracy,
            "macro_f1": metrics.classification.macro_f1,
            "energy_reduction": metrics.energy.energy_reduction_ratio,
            "battery_hours": metrics.energy.estimated_battery_hours,
            "overall_score": metrics.overall_score,
        }

    # Rank systems
    rankings = {}
    for metric in ["accuracy", "macro_f1", "energy_reduction", "battery_hours", "overall_score"]:
        sorted_systems = sorted(
            systems.keys(), key=lambda x: comparison[x][metric], reverse=True
        )
        rankings[metric] = sorted_systems

    # Find best system
    best_overall = max(systems.keys(), key=lambda x: systems[x].overall_score)

    return {
        "metrics": comparison,
        "rankings": rankings,
        "best_overall": best_overall,
    }
