"""Fixed-rate sampling policy (baseline).

This module provides a simple fixed-rate sampling policy for baseline comparison.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from liveedge.sampling.policies.base import BaseSamplingPolicy, PolicyAction, PolicyState


class FixedRatePolicy(BaseSamplingPolicy):
    """Fixed-rate sampling policy.

    Always returns the same sampling rate regardless of behavior or context.
    Used as a baseline for comparison with adaptive policies.

    Example:
        >>> policy = FixedRatePolicy(sampling_rate=50.0)
        >>> action = policy.select_rate(state)
        >>> print(action.sampling_rate)  # Always 50.0
    """

    def __init__(self, sampling_rate: float = 50.0):
        """Initialize the fixed-rate policy.

        Args:
            sampling_rate: Fixed sampling rate in Hz.
        """
        super().__init__(min_rate=sampling_rate, max_rate=sampling_rate)
        self.fixed_rate = sampling_rate

    def select_rate(
        self,
        state: PolicyState,
        window: NDArray[np.float32] | None = None,
    ) -> PolicyAction:
        """Return the fixed sampling rate.

        Args:
            state: Current policy state (ignored).
            window: Sensor window (ignored).

        Returns:
            PolicyAction with fixed sampling rate.
        """
        return PolicyAction(
            sampling_rate=self.fixed_rate,
            reason="Fixed rate policy",
        )

    def get_config(self) -> dict[str, float]:
        """Get policy configuration."""
        return {"sampling_rate": self.fixed_rate}


class BehaviorBasedFixedPolicy(BaseSamplingPolicy):
    """Behavior-based fixed-rate policy.

    Returns a fixed rate based on the detected behavior cluster,
    without hysteresis or adaptive adjustments.

    Example:
        >>> policy = BehaviorBasedFixedPolicy()
        >>> state = PolicyState(current_rate=50, current_behavior=BehaviorCluster.INACTIVE)
        >>> action = policy.select_rate(state)
        >>> print(action.sampling_rate)  # 5.0 for INACTIVE
    """

    def __init__(
        self,
        rate_mapping: dict | None = None,
    ):
        """Initialize the behavior-based policy.

        Args:
            rate_mapping: Mapping from BehaviorCluster to sampling rate.
                If None, uses default mapping.
        """
        from liveedge.clustering.clustering import BehaviorCluster

        super().__init__()

        self.rate_mapping = rate_mapping or {
            BehaviorCluster.INACTIVE: 5.0,
            BehaviorCluster.RUMINATING: 10.0,
            BehaviorCluster.FEEDING: 15.0,
            BehaviorCluster.LOCOMOTION: 25.0,
            BehaviorCluster.HIGH_ACTIVITY: 50.0,
        }

    def select_rate(
        self,
        state: PolicyState,
        window: NDArray[np.float32] | None = None,
    ) -> PolicyAction:
        """Select rate based on current behavior.

        Args:
            state: Current policy state with behavior information.
            window: Sensor window (ignored).

        Returns:
            PolicyAction with behavior-appropriate rate.
        """
        if state.current_behavior is None:
            rate = self.max_rate
            reason = "No behavior detected, using max rate"
        else:
            rate = self.rate_mapping.get(state.current_behavior, self.max_rate)
            reason = f"Rate for {state.current_behavior.name}"

        return PolicyAction(
            sampling_rate=self.quantize_rate(rate),
            reason=reason,
        )

    def get_config(self) -> dict:
        """Get policy configuration."""
        return {
            "rate_mapping": {k.name: v for k, v in self.rate_mapping.items()},
        }
