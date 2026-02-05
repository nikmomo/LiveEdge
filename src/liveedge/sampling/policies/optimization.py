"""Policy optimization utilities.

This module provides utilities for hyperparameter optimization
and policy comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from liveedge.clustering.clustering import BehaviorCluster
from liveedge.sampling.policies.base import BasePolicy, SamplingDecision


@dataclass
class OptimizationConfig:
    """Configuration for policy optimization.

    Attributes:
        n_trials: Number of optimization trials.
        n_jobs: Number of parallel jobs.
        timeout: Timeout in seconds.
        seed: Random seed.
        metric: Optimization metric (minimize or maximize).
        study_name: Name for the optimization study.
    """

    n_trials: int = 100
    n_jobs: int = 1
    timeout: int | None = None
    seed: int = 42
    metric: str = "maximize"
    study_name: str = "policy_optimization"


@dataclass
class OptimizationResult:
    """Result of policy optimization.

    Attributes:
        best_params: Best hyperparameters found.
        best_value: Best objective value.
        n_trials: Number of completed trials.
        optimization_history: History of objective values.
        trial_params: Parameters for each trial.
    """

    best_params: dict[str, Any]
    best_value: float
    n_trials: int
    optimization_history: list[float] = field(default_factory=list)
    trial_params: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        """Get optimization summary.

        Returns:
            Formatted summary string.
        """
        lines = [
            "=== Optimization Results ===",
            f"Best Value: {self.best_value:.4f}",
            f"Trials Completed: {self.n_trials}",
            "",
            "Best Parameters:",
        ]

        for key, value in self.best_params.items():
            lines.append(f"  {key}: {value}")

        return "\n".join(lines)


class PolicyOptimizer:
    """Hyperparameter optimizer for sampling policies.

    Uses Optuna for Bayesian optimization of policy hyperparameters.
    """

    def __init__(self, config: OptimizationConfig | None = None):
        """Initialize the optimizer.

        Args:
            config: Optimization configuration.
        """
        self.config = config or OptimizationConfig()
        self.study = None
        self.results: OptimizationResult | None = None

    def optimize(
        self,
        policy_factory: Callable[[dict[str, Any]], BasePolicy],
        param_space: dict[str, Any],
        objective_fn: Callable[[BasePolicy], float],
    ) -> OptimizationResult:
        """Run hyperparameter optimization.

        Args:
            policy_factory: Function to create policy from params.
            param_space: Parameter search space.
            objective_fn: Objective function to optimize.

        Returns:
            OptimizationResult with best parameters.
        """
        try:
            import optuna
        except ImportError:
            raise ImportError(
                "optuna is required for optimization. "
                "Install with: pip install optuna"
            )

        def objective(trial: optuna.Trial) -> float:
            params = self._sample_params(trial, param_space)
            policy = policy_factory(params)
            return objective_fn(policy)

        direction = "maximize" if self.config.metric == "maximize" else "minimize"

        self.study = optuna.create_study(
            study_name=self.config.study_name,
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=self.config.seed),
        )

        self.study.optimize(
            objective,
            n_trials=self.config.n_trials,
            n_jobs=self.config.n_jobs,
            timeout=self.config.timeout,
            show_progress_bar=True,
        )

        self.results = OptimizationResult(
            best_params=self.study.best_params,
            best_value=self.study.best_value,
            n_trials=len(self.study.trials),
            optimization_history=[t.value for t in self.study.trials if t.value is not None],
            trial_params=[t.params for t in self.study.trials],
        )

        return self.results

    def _sample_params(
        self,
        trial: Any,
        param_space: dict[str, Any],
    ) -> dict[str, Any]:
        """Sample parameters from search space.

        Args:
            trial: Optuna trial object.
            param_space: Parameter search space definition.

        Returns:
            Dictionary of sampled parameters.
        """
        params = {}

        for name, spec in param_space.items():
            param_type = spec.get("type", "float")

            if param_type == "float":
                params[name] = trial.suggest_float(
                    name,
                    spec["low"],
                    spec["high"],
                    log=spec.get("log", False),
                )
            elif param_type == "int":
                params[name] = trial.suggest_int(
                    name,
                    spec["low"],
                    spec["high"],
                    log=spec.get("log", False),
                )
            elif param_type == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")

        return params


class PolicyComparator:
    """Compare multiple sampling policies.

    Provides utilities for systematic policy comparison.
    """

    def __init__(self):
        """Initialize the comparator."""
        self.results: dict[str, dict[str, Any]] = {}

    def evaluate_policy(
        self,
        name: str,
        policy: BasePolicy,
        windows: NDArray[np.float32],
        labels: NDArray[np.int64],
        energy_model: Any | None = None,
    ) -> dict[str, Any]:
        """Evaluate a policy on test data.

        Args:
            name: Policy name.
            policy: Policy to evaluate.
            windows: Test windows.
            labels: Test labels.
            energy_model: Optional energy model for energy computation.

        Returns:
            Dictionary with evaluation metrics.
        """
        decisions: list[SamplingDecision] = []
        rates: list[int] = []
        states: list[BehaviorCluster] = []

        # Simulate policy decisions
        for i, (window, label) in enumerate(zip(windows, labels)):
            # Compute variance from window
            variance = float(np.var(window))

            # Map label to state
            state = list(BehaviorCluster)[label % len(BehaviorCluster)]

            decision = policy.select_rate(
                current_state=state,
                variance=variance,
                confidence=0.9,  # Assume high confidence for evaluation
            )

            decisions.append(decision)
            rates.append(decision.rate)
            states.append(decision.state)

        # Compute metrics
        avg_rate = np.mean(rates)
        rate_std = np.std(rates)

        # Count rate changes
        rate_changes = sum(
            1 for i in range(1, len(rates)) if rates[i] != rates[i - 1]
        )

        # Energy estimation (simplified)
        if energy_model is not None:
            total_energy = sum(
                energy_model.compute_sensing_energy(rate, 1.0)
                for rate in rates
            )
        else:
            # Normalize by max rate
            max_rate = max(policy.config.available_rates)
            total_energy = sum(rate / max_rate for rate in rates)

        metrics = {
            "name": name,
            "avg_rate": avg_rate,
            "rate_std": rate_std,
            "rate_changes": rate_changes,
            "total_energy": total_energy,
            "n_samples": len(windows),
        }

        self.results[name] = metrics
        return metrics

    def compare(self) -> str:
        """Generate comparison report.

        Returns:
            Formatted comparison table.
        """
        if not self.results:
            return "No results available."

        lines = [
            "| Policy | Avg Rate | Rate Std | Changes | Energy |",
            "|--------|----------|----------|---------|--------|",
        ]

        for name, metrics in self.results.items():
            lines.append(
                f"| {name:6} | "
                f"{metrics['avg_rate']:.1f} | "
                f"{metrics['rate_std']:.1f} | "
                f"{metrics['rate_changes']:7d} | "
                f"{metrics['total_energy']:.2f} |"
            )

        return "\n".join(lines)

    def get_best_policy(self, metric: str = "total_energy") -> str:
        """Get the best policy by metric.

        Args:
            metric: Metric to use for comparison.

        Returns:
            Name of best policy.
        """
        if not self.results:
            raise ValueError("No results available.")

        # Lower is better for energy, rate_std, rate_changes
        minimize_metrics = {"total_energy", "rate_std", "rate_changes"}

        if metric in minimize_metrics:
            best = min(self.results.items(), key=lambda x: x[1][metric])
        else:
            best = max(self.results.items(), key=lambda x: x[1][metric])

        return best[0]


def grid_search_policy(
    policy_class: type[BasePolicy],
    param_grid: dict[str, list[Any]],
    objective_fn: Callable[[BasePolicy], float],
    maximize: bool = True,
) -> tuple[dict[str, Any], float]:
    """Perform grid search over policy hyperparameters.

    Args:
        policy_class: Policy class to instantiate.
        param_grid: Grid of parameters to search.
        objective_fn: Objective function.
        maximize: Whether to maximize objective.

    Returns:
        Tuple of (best_params, best_score).
    """
    from itertools import product

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    best_params: dict[str, Any] = {}
    best_score = float("-inf") if maximize else float("inf")

    for values in product(*param_values):
        params = dict(zip(param_names, values))

        # Create config from params
        from liveedge.sampling.policies.base import PolicyConfig

        config = PolicyConfig(**params)
        policy = policy_class(config)

        score = objective_fn(policy)

        if maximize and score > best_score:
            best_score = score
            best_params = params.copy()
        elif not maximize and score < best_score:
            best_score = score
            best_params = params.copy()

    return best_params, best_score


def compute_policy_score(
    policy: BasePolicy,
    accuracy: float,
    energy_reduction: float,
    accuracy_weight: float = 0.7,
    energy_weight: float = 0.3,
) -> float:
    """Compute overall policy score.

    Args:
        policy: Policy instance (unused, for interface compatibility).
        accuracy: Classification accuracy.
        energy_reduction: Energy reduction ratio.
        accuracy_weight: Weight for accuracy.
        energy_weight: Weight for energy.

    Returns:
        Weighted score.
    """
    return accuracy_weight * accuracy + energy_weight * energy_reduction
