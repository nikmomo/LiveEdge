"""Experiment runner for simulation.

This module provides utilities for running and managing simulation experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from liveedge.evaluation import (
    ClassificationMetrics,
    EnergyMetrics,
    SystemMetrics,
    compute_classification_metrics,
    compute_energy_metrics,
    create_system_metrics,
)
from liveedge.models.base import BaseClassifier
from liveedge.simulation.environment import SimulationConfig, SimulationResult, StreamingSimulator


@dataclass
class ExperimentResult:
    """Results from a complete experiment.

    Attributes:
        name: Experiment name.
        simulation_result: Raw simulation results.
        classification_metrics: Classification performance.
        energy_metrics: Energy efficiency.
        system_metrics: Combined system metrics.
    """

    name: str
    simulation_result: SimulationResult
    classification_metrics: ClassificationMetrics
    energy_metrics: EnergyMetrics
    system_metrics: SystemMetrics

    def summary(self) -> str:
        """Get experiment summary."""
        return f"=== {self.name} ===\n\n{self.system_metrics.summary()}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "classification": self.classification_metrics.to_dict(),
            "energy": self.energy_metrics.to_dict(),
            "overall_score": self.system_metrics.overall_score,
        }


class ExperimentRunner:
    """Run and manage simulation experiments.

    Example:
        >>> runner = ExperimentRunner()
        >>> result = runner.run_experiment(
        ...     name="adaptive_rule",
        ...     classifier=my_classifier,
        ...     windows=test_windows,
        ...     labels=test_labels,
        ...     config=my_config,
        ... )
        >>> print(result.summary())
    """

    def __init__(self, output_dir: str | Path | None = None):
        """Initialize the runner.

        Args:
            output_dir: Directory for saving results.
        """
        self.output_dir = Path(output_dir) if output_dir else None
        self.results: dict[str, ExperimentResult] = {}

    def run_experiment(
        self,
        name: str,
        classifier: BaseClassifier,
        windows: NDArray[np.float32],
        labels: NDArray[np.int64],
        config: SimulationConfig | None = None,
        class_names: list[str] | None = None,
        timestamps: NDArray[np.float64] | None = None,
        progress: bool = True,
    ) -> ExperimentResult:
        """Run a single experiment.

        Args:
            name: Experiment name.
            classifier: Trained classifier.
            windows: Test windows.
            labels: Test labels.
            config: Simulation configuration.
            class_names: Optional class names.
            timestamps: Optional timestamps.
            progress: Show progress bar.

        Returns:
            ExperimentResult with all metrics.
        """
        from liveedge.energy.models import EnergyModel

        if config is None:
            config = SimulationConfig()

        # Create simulator
        simulator = StreamingSimulator(config)
        simulator.set_classifier(classifier)

        # Run simulation
        sim_result = simulator.run(windows, labels, timestamps, progress)

        # Compute metrics
        y_true = sim_result.get_y_true()
        y_pred = sim_result.get_y_pred()

        classification_metrics = compute_classification_metrics(
            y_true, y_pred, class_names, compute_transition=True
        )

        energy_model = EnergyModel(config.hardware_spec)
        energy_metrics = compute_energy_metrics(
            sim_result.sampling_log,
            energy_model,
            config.model_name,
            baseline_rate=50.0,
        )

        system_metrics = create_system_metrics(
            classification_metrics,
            energy_metrics,
        )

        result = ExperimentResult(
            name=name,
            simulation_result=sim_result,
            classification_metrics=classification_metrics,
            energy_metrics=energy_metrics,
            system_metrics=system_metrics,
        )

        self.results[name] = result

        # Save results if output directory is set
        if self.output_dir is not None:
            self._save_result(result)

        return result

    def run_comparison(
        self,
        classifier: BaseClassifier,
        windows: NDArray[np.float32],
        labels: NDArray[np.int64],
        configs: dict[str, SimulationConfig],
        class_names: list[str] | None = None,
        progress: bool = True,
    ) -> dict[str, ExperimentResult]:
        """Run multiple experiments for comparison.

        Args:
            classifier: Trained classifier.
            windows: Test windows.
            labels: Test labels.
            configs: Dictionary of config name to config.
            class_names: Optional class names.
            progress: Show progress bar.

        Returns:
            Dictionary of experiment name to result.
        """
        results = {}
        for name, config in configs.items():
            result = self.run_experiment(
                name=name,
                classifier=classifier,
                windows=windows,
                labels=labels,
                config=config,
                class_names=class_names,
                progress=progress,
            )
            results[name] = result
        return results

    def _save_result(self, result: ExperimentResult) -> None:
        """Save experiment result to file."""
        import json

        if self.output_dir is None:
            return

        output_path = self.output_dir / f"{result.name}_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    def get_comparison_table(self) -> str:
        """Get comparison table of all results.

        Returns:
            Formatted comparison table string.
        """
        if not self.results:
            return "No results available."

        lines = [
            "| Experiment | Accuracy | Macro F1 | Energy Red. | Battery (h) | Score |",
            "|------------|----------|----------|-------------|-------------|-------|",
        ]

        for name, result in self.results.items():
            lines.append(
                f"| {name:10} | "
                f"{result.classification_metrics.accuracy:.4f} | "
                f"{result.classification_metrics.macro_f1:.4f} | "
                f"{result.energy_metrics.energy_reduction_ratio:.3f} | "
                f"{result.energy_metrics.estimated_battery_hours:.1f} | "
                f"{result.system_metrics.overall_score:.4f} |"
            )

        return "\n".join(lines)
