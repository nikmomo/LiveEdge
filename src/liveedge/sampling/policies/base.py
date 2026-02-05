"""Abstract base class for adaptive sampling policies.

This module defines the interface that all sampling policies must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from liveedge.clustering.clustering import BehaviorCluster


@dataclass
class PolicyConfig:
    """Configuration for sampling policies.

    Attributes:
        available_rates: List of available sampling rates.
        state_frequencies: Mapping from state to default frequency.
        min_rate: Minimum sampling rate.
        max_rate: Maximum sampling rate.
    """

    available_rates: list[int] = field(
        default_factory=lambda: [5, 10, 15, 25, 50]
    )
    state_frequencies: dict[BehaviorCluster, int] = field(default_factory=dict)
    min_rate: int = 5
    max_rate: int = 50

    def __post_init__(self) -> None:
        """Initialize default state frequencies if not provided."""
        if not self.state_frequencies:
            self.state_frequencies = {
                BehaviorCluster.INACTIVE: 5,
                BehaviorCluster.RUMINATING: 10,
                BehaviorCluster.FEEDING: 15,
                BehaviorCluster.LOCOMOTION: 25,
                BehaviorCluster.HIGH_ACTIVITY: 50,
            }


@dataclass
class SamplingDecision:
    """Decision output from a sampling policy.

    Attributes:
        rate: Selected sampling rate.
        state: Current behavioral state.
        confidence: Decision confidence.
        metadata: Additional decision metadata.
    """

    rate: int
    state: BehaviorCluster
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class BasePolicy(ABC):
    """Abstract base class for configurable sampling policies.

    Provides a unified interface for all policy types including
    rule-based and DRL-based policies.
    """

    def __init__(self, config: PolicyConfig | None = None):
        """Initialize the policy.

        Args:
            config: Policy configuration.
        """
        self.config = config or PolicyConfig()
        self._decision_count = 0
        self._rate_history: list[int] = []

    @abstractmethod
    def select_rate(
        self,
        current_state: BehaviorCluster,
        variance: float | None = None,
        confidence: float | None = None,
        features: NDArray[np.float32] | None = None,
        **kwargs: Any,
    ) -> SamplingDecision:
        """Select sampling rate based on current context.

        Args:
            current_state: Current behavioral state.
            variance: Signal variance.
            confidence: Classification confidence.
            features: Optional feature vector.
            **kwargs: Additional policy-specific arguments.

        Returns:
            SamplingDecision with selected rate.
        """
        pass

    def reset(self) -> None:
        """Reset policy state."""
        self._decision_count = 0
        self._rate_history.clear()

    def record_decision(self, decision: SamplingDecision) -> None:
        """Record a decision for tracking.

        Args:
            decision: The sampling decision made.
        """
        self._decision_count += 1
        self._rate_history.append(decision.rate)

    def get_statistics(self) -> dict[str, Any]:
        """Get policy statistics.

        Returns:
            Dictionary of statistics.
        """
        if not self._rate_history:
            return {
                "decision_count": 0,
                "avg_rate": 0.0,
                "rate_std": 0.0,
            }

        return {
            "decision_count": self._decision_count,
            "avg_rate": float(np.mean(self._rate_history)),
            "rate_std": float(np.std(self._rate_history)),
        }


@dataclass
class PolicyState:
    """State information for a sampling policy.

    Attributes:
        current_rate: Current sampling rate in Hz.
        current_behavior: Current detected behavior cluster.
        confidence: Classifier confidence for current prediction.
        signal_variance: Recent signal variance.
        time_in_state: Time spent in current state.
    """

    current_rate: float
    current_behavior: BehaviorCluster | None = None
    confidence: float = 0.0
    signal_variance: float = 0.0
    time_in_state: float = 0.0


@dataclass
class PolicyAction:
    """Action output from a sampling policy.

    Attributes:
        sampling_rate: New sampling rate in Hz.
        reason: Reason for the rate selection.
    """

    sampling_rate: float
    reason: str = ""


class BaseSamplingPolicy(ABC):
    """Abstract base class for adaptive sampling policies.

    All policies must implement the select_rate method which determines
    the sampling rate based on current state and context.

    Attributes:
        min_rate: Minimum allowed sampling rate in Hz.
        max_rate: Maximum allowed sampling rate in Hz.
        available_rates: List of available sampling rates.
    """

    def __init__(
        self,
        min_rate: float = 5.0,
        max_rate: float = 50.0,
        available_rates: list[float] | None = None,
    ):
        """Initialize the policy.

        Args:
            min_rate: Minimum sampling rate in Hz.
            max_rate: Maximum sampling rate in Hz.
            available_rates: List of available rates (None = continuous).
        """
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.available_rates = available_rates or [5, 10, 15, 25, 50]

    @abstractmethod
    def select_rate(
        self,
        state: PolicyState,
        window: NDArray[np.float32] | None = None,
    ) -> PolicyAction:
        """Select the sampling rate based on current state.

        Args:
            state: Current policy state with context information.
            window: Optional recent sensor window for analysis.

        Returns:
            PolicyAction with selected sampling rate.
        """
        pass

    def reset(self) -> None:
        """Reset the policy to initial state.

        Override in subclasses that maintain internal state.
        """
        pass

    def quantize_rate(self, rate: float) -> float:
        """Quantize rate to nearest available rate.

        Args:
            rate: Desired sampling rate.

        Returns:
            Nearest available sampling rate.
        """
        rate = max(self.min_rate, min(self.max_rate, rate))

        if self.available_rates:
            # Find nearest available rate
            return min(self.available_rates, key=lambda x: abs(x - rate))
        return rate

    def get_config(self) -> dict[str, Any]:
        """Get policy configuration.

        Returns:
            Dictionary of configuration parameters.
        """
        return {
            "min_rate": self.min_rate,
            "max_rate": self.max_rate,
            "available_rates": self.available_rates,
        }


class StatefulPolicy(BaseSamplingPolicy):
    """Base class for policies that maintain internal state.

    Provides common functionality for policies that track history
    or learn from experience.
    """

    def __init__(
        self,
        min_rate: float = 5.0,
        max_rate: float = 50.0,
        available_rates: list[float] | None = None,
        history_size: int = 100,
    ):
        """Initialize the stateful policy.

        Args:
            min_rate: Minimum sampling rate in Hz.
            max_rate: Maximum sampling rate in Hz.
            available_rates: List of available rates.
            history_size: Maximum history entries to keep.
        """
        super().__init__(min_rate, max_rate, available_rates)
        self.history_size = history_size
        self._history: list[tuple[PolicyState, PolicyAction]] = []

    def record_action(self, state: PolicyState, action: PolicyAction) -> None:
        """Record a state-action pair in history.

        Args:
            state: Policy state at decision time.
            action: Action taken.
        """
        self._history.append((state, action))
        if len(self._history) > self.history_size:
            self._history.pop(0)

    def get_history(self) -> list[tuple[PolicyState, PolicyAction]]:
        """Get action history.

        Returns:
            List of (state, action) tuples.
        """
        return self._history.copy()

    def reset(self) -> None:
        """Reset policy state and history."""
        self._history.clear()
