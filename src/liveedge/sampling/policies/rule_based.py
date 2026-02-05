"""Rule-based adaptive sampling policies.

This module provides rule-based policies that adjust sampling rates
based on signal characteristics and classifier confidence.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from liveedge.clustering.clustering import BehaviorCluster
from liveedge.data.features import compute_magnitude
from liveedge.sampling.policies.base import (
    BaseSamplingPolicy,
    PolicyAction,
    PolicyState,
    StatefulPolicy,
)


class VarianceBasedPolicy(BaseSamplingPolicy):
    """Adjust sampling rate based on signal variance.

    Higher variance indicates more dynamic activity, requiring higher sampling rates.

    Example:
        >>> policy = VarianceBasedPolicy(variance_threshold=0.5)
        >>> state = PolicyState(current_rate=25, signal_variance=0.8)
        >>> action = policy.select_rate(state)
    """

    def __init__(
        self,
        min_rate: float = 5.0,
        max_rate: float = 50.0,
        variance_threshold: float = 0.5,
        upscale_factor: float = 2.0,
        downscale_factor: float = 0.5,
        available_rates: list[float] | None = None,
    ):
        """Initialize the variance-based policy.

        Args:
            min_rate: Minimum sampling rate in Hz.
            max_rate: Maximum sampling rate in Hz.
            variance_threshold: Threshold for rate adjustment.
            upscale_factor: Factor to increase rate when variance is high.
            downscale_factor: Factor to decrease rate when variance is low.
            available_rates: List of available sampling rates.
        """
        super().__init__(min_rate, max_rate, available_rates)
        self.variance_threshold = variance_threshold
        self.upscale_factor = upscale_factor
        self.downscale_factor = downscale_factor

    def select_rate(
        self,
        state: PolicyState,
        window: NDArray[np.float32] | None = None,
    ) -> PolicyAction:
        """Select rate based on signal variance.

        Args:
            state: Current policy state.
            window: Optional sensor window for variance calculation.

        Returns:
            PolicyAction with adjusted sampling rate.
        """
        variance = state.signal_variance

        # Calculate variance from window if provided and state variance is 0
        if variance == 0 and window is not None:
            magnitude = compute_magnitude(window)
            variance = float(np.var(magnitude))

        current_rate = state.current_rate

        if variance > self.variance_threshold:
            new_rate = current_rate * self.upscale_factor
            reason = f"High variance ({variance:.3f} > {self.variance_threshold})"
        else:
            new_rate = current_rate * self.downscale_factor
            reason = f"Low variance ({variance:.3f} <= {self.variance_threshold})"

        return PolicyAction(
            sampling_rate=self.quantize_rate(new_rate),
            reason=reason,
        )


class ConfidenceBasedPolicy(BaseSamplingPolicy):
    """Adjust sampling rate based on classifier confidence.

    Low confidence indicates uncertainty, requiring higher sampling rates
    for better classification accuracy.

    Example:
        >>> policy = ConfidenceBasedPolicy(confidence_threshold=0.8)
        >>> state = PolicyState(current_rate=25, confidence=0.6)
        >>> action = policy.select_rate(state)
    """

    def __init__(
        self,
        min_rate: float = 5.0,
        max_rate: float = 50.0,
        confidence_threshold: float = 0.8,
        low_confidence_rate: float | None = None,
        high_confidence_rate: float | None = None,
        available_rates: list[float] | None = None,
    ):
        """Initialize the confidence-based policy.

        Args:
            min_rate: Minimum sampling rate in Hz.
            max_rate: Maximum sampling rate in Hz.
            confidence_threshold: Threshold for rate adjustment.
            low_confidence_rate: Rate when confidence is low (None = max_rate).
            high_confidence_rate: Rate when confidence is high (None = min_rate).
            available_rates: List of available sampling rates.
        """
        super().__init__(min_rate, max_rate, available_rates)
        self.confidence_threshold = confidence_threshold
        self.low_confidence_rate = low_confidence_rate or max_rate
        self.high_confidence_rate = high_confidence_rate or min_rate

    def select_rate(
        self,
        state: PolicyState,
        window: NDArray[np.float32] | None = None,
    ) -> PolicyAction:
        """Select rate based on classifier confidence.

        Args:
            state: Current policy state with confidence.
            window: Sensor window (ignored).

        Returns:
            PolicyAction with confidence-adjusted rate.
        """
        if state.confidence >= self.confidence_threshold:
            new_rate = self.high_confidence_rate
            reason = f"High confidence ({state.confidence:.2f} >= {self.confidence_threshold})"
        else:
            new_rate = self.low_confidence_rate
            reason = f"Low confidence ({state.confidence:.2f} < {self.confidence_threshold})"

        return PolicyAction(
            sampling_rate=self.quantize_rate(new_rate),
            reason=reason,
        )


class CombinedRulePolicy(StatefulPolicy):
    """Combine multiple rules for adaptive sampling.

    Considers behavior, confidence, and variance to determine sampling rate.

    Example:
        >>> policy = CombinedRulePolicy()
        >>> state = PolicyState(
        ...     current_rate=25,
        ...     current_behavior=BehaviorCluster.LOCOMOTION,
        ...     confidence=0.9,
        ...     signal_variance=0.3
        ... )
        >>> action = policy.select_rate(state)
    """

    def __init__(
        self,
        min_rate: float = 5.0,
        max_rate: float = 50.0,
        available_rates: list[float] | None = None,
        behavior_rates: dict[BehaviorCluster, float] | None = None,
        confidence_threshold: float = 0.8,
        variance_threshold: float = 0.5,
        confidence_weight: float = 0.3,
        variance_weight: float = 0.2,
    ):
        """Initialize the combined rule policy.

        Args:
            min_rate: Minimum sampling rate in Hz.
            max_rate: Maximum sampling rate in Hz.
            available_rates: List of available sampling rates.
            behavior_rates: Mapping from behavior to base rate.
            confidence_threshold: Threshold for confidence adjustment.
            variance_threshold: Threshold for variance adjustment.
            confidence_weight: Weight for confidence adjustment.
            variance_weight: Weight for variance adjustment.
        """
        super().__init__(min_rate, max_rate, available_rates)

        self.behavior_rates = behavior_rates or {
            BehaviorCluster.INACTIVE: 5.0,
            BehaviorCluster.RUMINATING: 10.0,
            BehaviorCluster.FEEDING: 15.0,
            BehaviorCluster.LOCOMOTION: 25.0,
            BehaviorCluster.HIGH_ACTIVITY: 50.0,
        }

        self.confidence_threshold = confidence_threshold
        self.variance_threshold = variance_threshold
        self.confidence_weight = confidence_weight
        self.variance_weight = variance_weight

    def select_rate(
        self,
        state: PolicyState,
        window: NDArray[np.float32] | None = None,
    ) -> PolicyAction:
        """Select rate based on combined rules.

        The final rate is computed as:
        rate = base_rate * (1 + confidence_adj + variance_adj)

        Args:
            state: Current policy state.
            window: Optional sensor window.

        Returns:
            PolicyAction with combined-rule rate.
        """
        reasons = []

        # Base rate from behavior
        if state.current_behavior is not None:
            base_rate = self.behavior_rates.get(state.current_behavior, self.max_rate)
            reasons.append(f"Base: {state.current_behavior.name}")
        else:
            base_rate = self.max_rate
            reasons.append("Base: unknown behavior")

        # Confidence adjustment
        confidence_adj = 0.0
        if state.confidence < self.confidence_threshold:
            # Low confidence -> increase rate
            confidence_adj = (1 - state.confidence) * self.confidence_weight
            reasons.append(f"Conf adj: +{confidence_adj:.2f}")

        # Variance adjustment
        variance_adj = 0.0
        variance = state.signal_variance
        if window is not None and variance == 0:
            magnitude = compute_magnitude(window)
            variance = float(np.var(magnitude))

        if variance > self.variance_threshold:
            variance_adj = (variance - self.variance_threshold) * self.variance_weight
            reasons.append(f"Var adj: +{variance_adj:.2f}")

        # Compute final rate
        adjustment = 1.0 + confidence_adj + variance_adj
        new_rate = base_rate * adjustment

        action = PolicyAction(
            sampling_rate=self.quantize_rate(new_rate),
            reason="; ".join(reasons),
        )

        self.record_action(state, action)
        return action


class StateTransitionPolicy(StatefulPolicy):
    """Adjust rate based on state transition probability.

    Increases sampling rate when a state transition is likely,
    based on time in current state and historical patterns.
    """

    def __init__(
        self,
        min_rate: float = 5.0,
        max_rate: float = 50.0,
        available_rates: list[float] | None = None,
        behavior_rates: dict[BehaviorCluster, float] | None = None,
        expected_dwell_times: dict[BehaviorCluster, float] | None = None,
        transition_boost: float = 1.5,
    ):
        """Initialize the state transition policy.

        Args:
            min_rate: Minimum sampling rate in Hz.
            max_rate: Maximum sampling rate in Hz.
            available_rates: List of available sampling rates.
            behavior_rates: Base rates per behavior.
            expected_dwell_times: Expected time in each state (seconds).
            transition_boost: Rate multiplier when transition is likely.
        """
        super().__init__(min_rate, max_rate, available_rates)

        self.behavior_rates = behavior_rates or {
            BehaviorCluster.INACTIVE: 5.0,
            BehaviorCluster.RUMINATING: 10.0,
            BehaviorCluster.FEEDING: 15.0,
            BehaviorCluster.LOCOMOTION: 25.0,
            BehaviorCluster.HIGH_ACTIVITY: 50.0,
        }

        self.expected_dwell_times = expected_dwell_times or {
            BehaviorCluster.INACTIVE: 300.0,  # 5 minutes
            BehaviorCluster.RUMINATING: 600.0,  # 10 minutes
            BehaviorCluster.FEEDING: 900.0,  # 15 minutes
            BehaviorCluster.LOCOMOTION: 120.0,  # 2 minutes
            BehaviorCluster.HIGH_ACTIVITY: 60.0,  # 1 minute
        }

        self.transition_boost = transition_boost

    def select_rate(
        self,
        state: PolicyState,
        window: NDArray[np.float32] | None = None,
    ) -> PolicyAction:
        """Select rate based on transition probability.

        Args:
            state: Current policy state.
            window: Sensor window (ignored).

        Returns:
            PolicyAction with transition-aware rate.
        """
        if state.current_behavior is None:
            return PolicyAction(sampling_rate=self.max_rate, reason="Unknown behavior")

        base_rate = self.behavior_rates.get(state.current_behavior, self.max_rate)
        expected_dwell = self.expected_dwell_times.get(state.current_behavior, 300.0)

        # Calculate transition probability based on time in state
        transition_prob = min(1.0, state.time_in_state / expected_dwell)

        if transition_prob > 0.5:
            # Likely to transition soon, boost rate
            boost = 1.0 + (transition_prob - 0.5) * (self.transition_boost - 1.0) * 2
            new_rate = base_rate * boost
            reason = f"Transition likely (prob={transition_prob:.2f}), boosted"
        else:
            new_rate = base_rate
            reason = f"Stable state (prob={transition_prob:.2f})"

        action = PolicyAction(
            sampling_rate=self.quantize_rate(new_rate),
            reason=reason,
        )

        self.record_action(state, action)
        return action
