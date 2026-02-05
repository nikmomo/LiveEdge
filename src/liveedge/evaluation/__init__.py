"""Evaluation metrics module for LiveEdge.

This module provides metrics for evaluating classification,
reconstruction, energy efficiency, and overall system performance.
"""

from liveedge.evaluation.classification import (
    ClassificationMetrics,
    compute_classification_metrics,
    compute_per_class_support,
    compute_transition_accuracy,
)
from liveedge.evaluation.energy import (
    EnergyMetrics,
    compute_energy_metrics,
    compute_sampling_efficiency,
    count_rate_switches,
)
from liveedge.evaluation.reconstruction import (
    ReconstructionMetrics,
    compute_feature_preservation,
    compute_reconstruction_metrics,
)
from liveedge.evaluation.system import (
    SystemMetrics,
    compare_systems,
    compute_overall_score,
    create_system_metrics,
)

__all__ = [
    # Classification
    "ClassificationMetrics",
    "compute_classification_metrics",
    "compute_transition_accuracy",
    "compute_per_class_support",
    # Reconstruction
    "ReconstructionMetrics",
    "compute_reconstruction_metrics",
    "compute_feature_preservation",
    # Energy
    "EnergyMetrics",
    "compute_energy_metrics",
    "count_rate_switches",
    "compute_sampling_efficiency",
    # System
    "SystemMetrics",
    "compute_overall_score",
    "create_system_metrics",
    "compare_systems",
]
