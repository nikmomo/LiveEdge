"""Adaptive sampling module for LiveEdge.

This module provides FSM-based adaptive sampling control and
signal reconstruction utilities.
"""

from liveedge.sampling.fsm import (
    AdaptiveFSM,
    BehaviorFSM,
    FSMConfig,
    FSMState,
)
from liveedge.sampling.interpolation import (
    InterpolationMethod,
    adaptive_to_fixed_rate,
    downsample_signal,
    reconstruct_signal,
    simulate_adaptive_sampling,
    sinc_interpolate,
)
from liveedge.sampling.policies import (
    BasePolicy,
    BaseSamplingPolicy,
    BehaviorBasedFixedPolicy,
    CombinedRulePolicy,
    ConfidenceBasedPolicy,
    DRLPolicy,
    DRLPolicyConfig,
    FixedRatePolicy,
    OptimizationConfig,
    OptimizationResult,
    PolicyAction,
    PolicyComparator,
    PolicyConfig,
    PolicyOptimizer,
    PolicyState,
    SamplingDecision,
    SamplingEnvironment,
    StatefulPolicy,
    StateTransitionPolicy,
    VarianceBasedPolicy,
    compute_policy_score,
    grid_search_policy,
)

__all__ = [
    # FSM
    "FSMConfig",
    "FSMState",
    "BehaviorFSM",
    "AdaptiveFSM",
    # Interpolation
    "InterpolationMethod",
    "reconstruct_signal",
    "sinc_interpolate",
    "adaptive_to_fixed_rate",
    "downsample_signal",
    "simulate_adaptive_sampling",
    # Policies - Base
    "BasePolicy",
    "BaseSamplingPolicy",
    "StatefulPolicy",
    "PolicyState",
    "PolicyAction",
    "PolicyConfig",
    "SamplingDecision",
    # Policies - Fixed
    "FixedRatePolicy",
    "BehaviorBasedFixedPolicy",
    # Policies - Rule-based
    "VarianceBasedPolicy",
    "ConfidenceBasedPolicy",
    "CombinedRulePolicy",
    "StateTransitionPolicy",
    # Policies - DRL
    "DRLPolicy",
    "DRLPolicyConfig",
    "SamplingEnvironment",
    # Policies - Optimization
    "PolicyOptimizer",
    "PolicyComparator",
    "OptimizationConfig",
    "OptimizationResult",
    "grid_search_policy",
    "compute_policy_score",
]
