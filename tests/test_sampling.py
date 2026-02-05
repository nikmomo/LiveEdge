"""Tests for sampling policies and FSM."""

from __future__ import annotations

import numpy as np
import pytest

from liveedge.clustering import BehaviorCluster
from liveedge.sampling import BehaviorFSM, FSMConfig
from liveedge.sampling.policies import (
    BasePolicy,
    BaseSamplingPolicy,
    BehaviorBasedFixedPolicy,
    CombinedRulePolicy,
    ConfidenceBasedPolicy,
    FixedRatePolicy,
    PolicyConfig,
    PolicyState,
    SamplingDecision,
    VarianceBasedPolicy,
)


class TestFSMConfig:
    """Tests for FSM configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FSMConfig()

        assert config.n_confirm == 3
        assert config.min_dwell_time == 2.0
        assert config.initial_state == BehaviorCluster.INACTIVE
        assert len(config.state_frequencies) == 5

    def test_custom_config(self):
        """Test custom configuration."""
        config = FSMConfig(
            n_confirm=5,
            min_dwell_time=3.0,
            initial_state=BehaviorCluster.FEEDING,
        )

        assert config.n_confirm == 5
        assert config.min_dwell_time == 3.0
        assert config.initial_state == BehaviorCluster.FEEDING


class TestBehaviorFSM:
    """Tests for Behavior FSM."""

    @pytest.fixture
    def fsm(self):
        """Create a default FSM."""
        config = FSMConfig(
            n_confirm=3,
            min_dwell_time=1.0,
        )
        return BehaviorFSM(config)

    def test_initial_state(self, fsm):
        """Test FSM starts in correct initial state."""
        assert fsm.current_state == BehaviorCluster.INACTIVE

    def test_state_transition_with_hysteresis(self, fsm):
        """Test state transitions require confirmation."""
        # First prediction - should not change state
        state, rate, changed = fsm.update(BehaviorCluster.FEEDING, 0.0)
        assert state == BehaviorCluster.INACTIVE
        assert not changed

        # Second prediction
        state, rate, changed = fsm.update(BehaviorCluster.FEEDING, 0.5)
        assert state == BehaviorCluster.INACTIVE
        assert not changed

        # Third prediction - should change state (n_confirm=3)
        state, rate, changed = fsm.update(BehaviorCluster.FEEDING, 1.0)
        assert state == BehaviorCluster.FEEDING
        assert changed

    def test_dwell_time_constraint(self, fsm):
        """Test minimum dwell time is enforced."""
        # Transition to FEEDING
        fsm.update(BehaviorCluster.FEEDING, 0.0)
        fsm.update(BehaviorCluster.FEEDING, 0.3)
        fsm.update(BehaviorCluster.FEEDING, 0.6)  # Now in FEEDING

        # Try to transition immediately - should fail due to dwell time
        fsm.update(BehaviorCluster.LOCOMOTION, 0.7)
        fsm.update(BehaviorCluster.LOCOMOTION, 0.8)
        state, _, _ = fsm.update(BehaviorCluster.LOCOMOTION, 0.9)

        # Should still be in FEEDING because min_dwell_time=1.0
        assert state == BehaviorCluster.FEEDING

    def test_sampling_rate_changes(self, fsm):
        """Test sampling rate changes with state."""
        # Get rate for INACTIVE
        _, inactive_rate, _ = fsm.update(BehaviorCluster.INACTIVE, 0.0)

        # Transition to HIGH_ACTIVITY
        fsm.update(BehaviorCluster.HIGH_ACTIVITY, 1.0)
        fsm.update(BehaviorCluster.HIGH_ACTIVITY, 2.0)
        _, active_rate, _ = fsm.update(BehaviorCluster.HIGH_ACTIVITY, 3.0)

        # HIGH_ACTIVITY should have higher rate
        assert active_rate > inactive_rate

    def test_reset(self, fsm):
        """Test FSM reset."""
        # Change state
        fsm.update(BehaviorCluster.FEEDING, 0.0)
        fsm.update(BehaviorCluster.FEEDING, 0.3)
        fsm.update(BehaviorCluster.FEEDING, 0.6)

        # Reset
        fsm.reset()

        assert fsm.current_state == BehaviorCluster.INACTIVE
        stats = fsm.get_statistics()
        assert stats["total_transitions"] == 0

    def test_statistics(self, fsm):
        """Test FSM statistics tracking."""
        # Make some transitions
        for t in np.arange(0, 3, 0.3):
            fsm.update(BehaviorCluster.FEEDING, t)

        for t in np.arange(3, 6, 0.3):
            fsm.update(BehaviorCluster.LOCOMOTION, t)

        stats = fsm.get_statistics()

        assert "total_transitions" in stats
        assert "state_counts" in stats
        assert stats["total_transitions"] >= 0


class TestFixedRatePolicy:
    """Tests for fixed rate policy."""

    def test_fixed_rate(self):
        """Test fixed rate is always returned."""
        policy = FixedRatePolicy(rate=25.0)

        state = PolicyState(current_rate=50.0)
        action = policy.select_rate(state)

        assert action.sampling_rate == 25.0

    def test_different_rates(self):
        """Test different fixed rates."""
        for rate in [5, 10, 15, 25, 50]:
            policy = FixedRatePolicy(rate=float(rate))
            state = PolicyState(current_rate=50.0)
            action = policy.select_rate(state)
            assert action.sampling_rate == rate


class TestBehaviorBasedFixedPolicy:
    """Tests for behavior-based fixed policy."""

    @pytest.fixture
    def policy(self):
        """Create a behavior-based policy."""
        return BehaviorBasedFixedPolicy(
            behavior_rates={
                BehaviorCluster.INACTIVE: 5.0,
                BehaviorCluster.RUMINATING: 10.0,
                BehaviorCluster.FEEDING: 15.0,
                BehaviorCluster.LOCOMOTION: 25.0,
                BehaviorCluster.HIGH_ACTIVITY: 50.0,
            }
        )

    def test_rate_for_each_behavior(self, policy):
        """Test correct rate for each behavior."""
        expected = {
            BehaviorCluster.INACTIVE: 5.0,
            BehaviorCluster.RUMINATING: 10.0,
            BehaviorCluster.FEEDING: 15.0,
            BehaviorCluster.LOCOMOTION: 25.0,
            BehaviorCluster.HIGH_ACTIVITY: 50.0,
        }

        for behavior, expected_rate in expected.items():
            state = PolicyState(
                current_rate=50.0,
                current_behavior=behavior,
            )
            action = policy.select_rate(state)
            assert action.sampling_rate == expected_rate


class TestVarianceBasedPolicy:
    """Tests for variance-based policy."""

    @pytest.fixture
    def policy(self):
        """Create a variance-based policy."""
        return VarianceBasedPolicy(
            variance_thresholds=[0.1, 0.5, 1.0, 2.0],
        )

    def test_low_variance(self, policy):
        """Test low variance gives low rate."""
        state = PolicyState(
            current_rate=50.0,
            signal_variance=0.05,
        )
        action = policy.select_rate(state)

        assert action.sampling_rate <= 10.0

    def test_high_variance(self, policy):
        """Test high variance gives high rate."""
        state = PolicyState(
            current_rate=5.0,
            signal_variance=3.0,
        )
        action = policy.select_rate(state)

        assert action.sampling_rate >= 25.0


class TestConfidenceBasedPolicy:
    """Tests for confidence-based policy."""

    @pytest.fixture
    def policy(self):
        """Create a confidence-based policy."""
        return ConfidenceBasedPolicy(
            low_confidence_threshold=0.6,
            high_confidence_threshold=0.9,
        )

    def test_low_confidence(self, policy):
        """Test low confidence increases rate."""
        state = PolicyState(
            current_rate=10.0,
            confidence=0.5,
        )
        action = policy.select_rate(state)

        # Low confidence should increase rate
        assert action.sampling_rate >= state.current_rate

    def test_high_confidence(self, policy):
        """Test high confidence allows lower rate."""
        state = PolicyState(
            current_rate=50.0,
            confidence=0.95,
        )
        action = policy.select_rate(state)

        # High confidence can have lower rate
        assert action.sampling_rate <= state.current_rate


class TestCombinedRulePolicy:
    """Tests for combined rule policy."""

    @pytest.fixture
    def policy(self):
        """Create a combined policy."""
        return CombinedRulePolicy(
            variance_weight=0.5,
            confidence_weight=0.5,
        )

    def test_combines_variance_and_confidence(self, policy):
        """Test policy considers both variance and confidence."""
        # Low variance, high confidence - should have low rate
        state1 = PolicyState(
            current_rate=25.0,
            signal_variance=0.1,
            confidence=0.95,
        )
        action1 = policy.select_rate(state1)

        # High variance, low confidence - should have high rate
        state2 = PolicyState(
            current_rate=25.0,
            signal_variance=2.0,
            confidence=0.5,
        )
        action2 = policy.select_rate(state2)

        assert action1.sampling_rate < action2.sampling_rate


class TestPolicyConfig:
    """Tests for PolicyConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = PolicyConfig()

        assert config.available_rates == [5, 10, 15, 25, 50]
        assert config.min_rate == 5
        assert config.max_rate == 50
        assert len(config.state_frequencies) == 5

    def test_custom_rates(self):
        """Test custom rate configuration."""
        config = PolicyConfig(
            available_rates=[10, 20, 30],
            min_rate=10,
            max_rate=30,
        )

        assert config.available_rates == [10, 20, 30]
        assert config.min_rate == 10
        assert config.max_rate == 30


class TestSamplingDecision:
    """Tests for SamplingDecision."""

    def test_decision_creation(self):
        """Test creating a sampling decision."""
        decision = SamplingDecision(
            rate=25,
            state=BehaviorCluster.FEEDING,
            confidence=0.9,
            metadata={"policy": "test"},
        )

        assert decision.rate == 25
        assert decision.state == BehaviorCluster.FEEDING
        assert decision.confidence == 0.9
        assert decision.metadata["policy"] == "test"

    def test_default_values(self):
        """Test default values."""
        decision = SamplingDecision(
            rate=10,
            state=BehaviorCluster.INACTIVE,
        )

        assert decision.confidence == 0.0
        assert decision.metadata == {}


class TestInterpolation:
    """Tests for signal interpolation."""

    def test_linear_interpolation(self):
        """Test linear interpolation."""
        from liveedge.sampling.interpolation import LinearInterpolator

        # Create sparse signal
        t_sparse = np.array([0, 2, 4, 6, 8])
        signal_sparse = np.array([0, 2, 4, 2, 0], dtype=np.float32)

        # Interpolate to dense
        t_dense = np.linspace(0, 8, 17)

        interpolator = LinearInterpolator()
        signal_dense = interpolator.interpolate(t_sparse, signal_sparse, t_dense)

        assert len(signal_dense) == len(t_dense)
        # Check known points
        assert signal_dense[0] == 0
        assert signal_dense[8] == 4  # Middle point

    def test_cubic_interpolation(self):
        """Test cubic spline interpolation."""
        from liveedge.sampling.interpolation import CubicSplineInterpolator

        t_sparse = np.array([0, 2, 4, 6, 8], dtype=np.float64)
        signal_sparse = np.sin(t_sparse * np.pi / 8).astype(np.float32)

        t_dense = np.linspace(0, 8, 33)

        interpolator = CubicSplineInterpolator()
        signal_dense = interpolator.interpolate(t_sparse, signal_sparse, t_dense)

        assert len(signal_dense) == len(t_dense)

    def test_interpolation_error_metrics(self):
        """Test interpolation error computation."""
        from liveedge.sampling.interpolation import compute_reconstruction_error

        original = np.sin(np.linspace(0, 2 * np.pi, 100)).astype(np.float32)
        reconstructed = original + np.random.randn(100).astype(np.float32) * 0.1

        error = compute_reconstruction_error(original, reconstructed)

        assert "mse" in error
        assert "mae" in error
        assert "nmse" in error
        assert error["mse"] > 0
