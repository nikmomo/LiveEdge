"""Simulation environment for adaptive sampling.

This module provides the core simulation environment for testing
adaptive sampling policies in a streaming context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from liveedge.clustering.clustering import BehaviorCluster
from liveedge.data.preprocessing import WindowConfig
from liveedge.energy.hardware import HardwareSpec
from liveedge.sampling.fsm import BehaviorFSM, FSMConfig


@dataclass
class SimulationConfig:
    """Configuration for streaming simulation.

    Attributes:
        window_config: Window configuration.
        fsm_config: FSM configuration.
        hardware_spec: Hardware specifications.
        model_name: Name of classifier model.
        policy_type: Type of sampling policy.
        policy_config: Policy-specific configuration.
        seed: Random seed for reproducibility.
    """

    window_config: WindowConfig = field(default_factory=WindowConfig)
    fsm_config: FSMConfig = field(default_factory=FSMConfig)
    hardware_spec: HardwareSpec = field(default_factory=HardwareSpec)
    model_name: str = "random_forest"
    policy_type: str = "fixed"
    policy_config: dict[str, Any] = field(default_factory=dict)
    seed: int = 42


@dataclass
class StepResult:
    """Result from a single simulation step.

    Attributes:
        timestamp: Current timestamp.
        predicted_label: Classifier prediction.
        true_label: Ground truth label.
        confidence: Classifier confidence.
        current_state: Current FSM state.
        sampling_rate: Current sampling rate.
        state_changed: Whether state changed this step.
    """

    timestamp: float
    predicted_label: int
    true_label: int
    confidence: float
    current_state: BehaviorCluster
    sampling_rate: int
    state_changed: bool


@dataclass
class SimulationResult:
    """Complete simulation results.

    Attributes:
        predictions: List of (timestamp, predicted, true) tuples.
        sampling_log: List of (timestamp, rate, state) tuples.
        config: Simulation configuration.
        duration_seconds: Total simulation duration.
    """

    predictions: list[tuple[float, int, int]]
    sampling_log: list[tuple[float, float, BehaviorCluster | None]]
    config: SimulationConfig
    duration_seconds: float

    def get_y_true(self) -> NDArray[np.int64]:
        """Get true labels array."""
        return np.array([p[2] for p in self.predictions], dtype=np.int64)

    def get_y_pred(self) -> NDArray[np.int64]:
        """Get predicted labels array."""
        return np.array([p[1] for p in self.predictions], dtype=np.int64)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "predictions": self.predictions,
            "sampling_log": [
                (t, r, s.name if s else None) for t, r, s in self.sampling_log
            ],
            "duration_seconds": self.duration_seconds,
        }

    def save(self, path: str | Path) -> None:
        """Save results to file."""
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class StreamingSimulator:
    """Simulate streaming sensor data with adaptive sampling.

    Provides a step-by-step simulation of the adaptive sampling system,
    including classification, FSM state management, and sampling rate control.

    Example:
        >>> config = SimulationConfig()
        >>> simulator = StreamingSimulator(config)
        >>> simulator.set_classifier(my_classifier)
        >>> result = simulator.run(windows, labels, timestamps)
    """

    def __init__(self, config: SimulationConfig):
        """Initialize the simulator.

        Args:
            config: Simulation configuration.
        """
        self.config = config
        self.fsm = BehaviorFSM(config.fsm_config)
        self.classifier = None
        self.policy = None
        self._predictions: list[tuple[float, int, int]] = []
        self._sampling_log: list[tuple[float, float, BehaviorCluster | None]] = []

    def set_classifier(self, classifier: Any) -> None:
        """Set the behavior classifier.

        Args:
            classifier: Classifier with predict and predict_proba methods.
        """
        self.classifier = classifier

    def set_policy(self, policy: Any) -> None:
        """Set the sampling policy.

        Args:
            policy: Sampling policy with select_rate method.
        """
        self.policy = policy

    def reset(self) -> None:
        """Reset the simulator to initial state."""
        self.fsm.reset()
        self._predictions.clear()
        self._sampling_log.clear()

    def step(
        self,
        timestamp: float,
        window: NDArray[np.float32],
        true_label: int,
    ) -> StepResult:
        """Perform a single simulation step.

        Args:
            timestamp: Current timestamp.
            window: Sensor window for classification.
            true_label: Ground truth behavior label.

        Returns:
            StepResult with step information.
        """
        if self.classifier is None:
            raise ValueError("Classifier not set. Call set_classifier() first.")

        # Classify the window
        pred_label = int(self.classifier.predict(window[np.newaxis, :])[0])
        proba = self.classifier.predict_proba(window[np.newaxis, :])[0]
        confidence = float(np.max(proba))

        # Map prediction to behavior cluster (simplified)
        predicted_cluster = self._label_to_cluster(pred_label)

        # Update FSM
        current_state, sampling_rate, state_changed = self.fsm.update(
            predicted_cluster, timestamp
        )

        # Record results
        self._predictions.append((timestamp, pred_label, true_label))
        self._sampling_log.append((timestamp, float(sampling_rate), current_state))

        return StepResult(
            timestamp=timestamp,
            predicted_label=pred_label,
            true_label=true_label,
            confidence=confidence,
            current_state=current_state,
            sampling_rate=sampling_rate,
            state_changed=state_changed,
        )

    def _label_to_cluster(self, label: int) -> BehaviorCluster:
        """Map class label to behavior cluster.

        Args:
            label: Class label index.

        Returns:
            Corresponding BehaviorCluster.
        """
        # Simple mapping: assume 5 clusters match 5 classes
        clusters = list(BehaviorCluster)
        return clusters[label % len(clusters)]

    def run(
        self,
        windows: NDArray[np.float32],
        labels: NDArray[np.int64],
        timestamps: NDArray[np.float64] | None = None,
        progress: bool = True,
    ) -> SimulationResult:
        """Run full simulation.

        Args:
            windows: Windows of shape (n_windows, window_size, n_channels).
            labels: Labels of shape (n_windows,).
            timestamps: Optional timestamps of shape (n_windows,).
            progress: Whether to show progress bar.

        Returns:
            SimulationResult with all results.
        """
        self.reset()

        n_windows = len(windows)

        if timestamps is None:
            window_duration = self.config.window_config.window_size
            step_duration = window_duration * (1 - self.config.window_config.overlap)
            timestamps = np.arange(n_windows) * step_duration

        # Run simulation
        if progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(range(n_windows), desc="Simulating")
            except ImportError:
                iterator = range(n_windows)
        else:
            iterator = range(n_windows)

        for i in iterator:
            self.step(timestamps[i], windows[i], labels[i])

        duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0

        return SimulationResult(
            predictions=self._predictions.copy(),
            sampling_log=self._sampling_log.copy(),
            config=self.config,
            duration_seconds=duration,
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get simulation statistics.

        Returns:
            Dictionary with simulation statistics.
        """
        fsm_stats = self.fsm.get_statistics()

        return {
            "n_predictions": len(self._predictions),
            "n_rate_changes": sum(
                1
                for i in range(1, len(self._sampling_log))
                if self._sampling_log[i][1] != self._sampling_log[i - 1][1]
            ),
            "fsm_stats": fsm_stats,
        }
