"""Finite State Machine for behavior-driven adaptive sampling.

This module provides an FSM implementation with hysteresis-based state
transitions to prevent rapid oscillation between sampling rates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from liveedge.clustering.clustering import BehaviorCluster


@dataclass
class FSMConfig:
    """Configuration for behavior FSM.

    Attributes:
        n_confirm: Number of consecutive confirmations needed for transition.
        min_dwell_time: Minimum seconds before allowing a new transition.
        initial_state: Initial behavioral state.
        state_frequencies: Mapping from state to sampling rate in Hz.
    """

    n_confirm: int = 3
    min_dwell_time: float = 5.0
    initial_state: BehaviorCluster = BehaviorCluster.LOCOMOTION
    state_frequencies: dict[BehaviorCluster, int] = field(
        default_factory=lambda: {
            BehaviorCluster.INACTIVE: 5,
            BehaviorCluster.RUMINATING: 10,
            BehaviorCluster.FEEDING: 15,
            BehaviorCluster.LOCOMOTION: 25,
            BehaviorCluster.HIGH_ACTIVITY: 50,
        }
    )

    def get_frequency(self, state: BehaviorCluster) -> int:
        """Get sampling frequency for a state.

        Args:
            state: Behavioral state.

        Returns:
            Sampling frequency in Hz.
        """
        return self.state_frequencies.get(state, 50)


@dataclass
class FSMState:
    """Current state of the FSM.

    Attributes:
        current_state: Current behavioral state.
        current_frequency: Current sampling frequency.
        candidate_state: State being considered for transition.
        confirm_count: Number of confirmations for candidate state.
        last_transition_time: Timestamp of last state transition.
        total_transitions: Total number of state transitions.
        state_history: History of (timestamp, state) tuples.
    """

    current_state: BehaviorCluster
    current_frequency: int
    candidate_state: BehaviorCluster | None = None
    confirm_count: int = 0
    last_transition_time: float = 0.0
    total_transitions: int = 0
    state_history: list[tuple[float, BehaviorCluster]] = field(default_factory=list)


class BehaviorFSM:
    """Finite State Machine for behavior-driven adaptive sampling.

    Implements hysteresis-based state transitions to prevent rapid oscillation.
    A state change only occurs when the classifier consistently predicts a
    different state for n_confirm consecutive windows, and after a minimum
    dwell time has passed.

    Example:
        >>> config = FSMConfig(n_confirm=3, min_dwell_time=5.0)
        >>> fsm = BehaviorFSM(config)
        >>> state, freq, changed = fsm.update(BehaviorCluster.FEEDING, current_time=0.0)
        >>> print(f"State: {state.name}, Frequency: {freq} Hz")
    """

    def __init__(self, config: FSMConfig):
        """Initialize the FSM.

        Args:
            config: FSM configuration.
        """
        self.config = config
        self.state = FSMState(
            current_state=config.initial_state,
            current_frequency=config.get_frequency(config.initial_state),
        )

    def update(
        self,
        predicted_state: BehaviorCluster,
        current_time: float,
    ) -> tuple[BehaviorCluster, int, bool]:
        """Update FSM based on classifier prediction.

        Args:
            predicted_state: Classifier's predicted behavioral state.
            current_time: Current timestamp in seconds.

        Returns:
            Tuple of (current_state, sampling_frequency, state_changed).
        """
        state_changed = False

        # Check if prediction matches current state
        if predicted_state == self.state.current_state:
            # Reset candidate tracking
            self.state.candidate_state = None
            self.state.confirm_count = 0

        else:
            # Check if we're tracking this candidate
            if predicted_state == self.state.candidate_state:
                self.state.confirm_count += 1
            else:
                # New candidate
                self.state.candidate_state = predicted_state
                self.state.confirm_count = 1

            # Check if we should transition
            if self.state.confirm_count >= self.config.n_confirm:
                # Check minimum dwell time
                time_in_state = current_time - self.state.last_transition_time
                if time_in_state >= self.config.min_dwell_time:
                    # Perform transition
                    self._transition_to(predicted_state, current_time)
                    state_changed = True

        return self.state.current_state, self.state.current_frequency, state_changed

    def _transition_to(self, new_state: BehaviorCluster, current_time: float) -> None:
        """Perform state transition.

        Args:
            new_state: New behavioral state.
            current_time: Current timestamp.
        """
        self.state.current_state = new_state
        self.state.current_frequency = self.config.get_frequency(new_state)
        self.state.candidate_state = None
        self.state.confirm_count = 0
        self.state.last_transition_time = current_time
        self.state.total_transitions += 1
        self.state.state_history.append((current_time, new_state))

    def force_transition(
        self,
        new_state: BehaviorCluster,
        current_time: float,
    ) -> tuple[BehaviorCluster, int]:
        """Force an immediate state transition (bypasses hysteresis).

        Args:
            new_state: New behavioral state.
            current_time: Current timestamp.

        Returns:
            Tuple of (new_state, sampling_frequency).
        """
        self._transition_to(new_state, current_time)
        return self.state.current_state, self.state.current_frequency

    def reset(self) -> None:
        """Reset FSM to initial state."""
        self.state = FSMState(
            current_state=self.config.initial_state,
            current_frequency=self.config.get_frequency(self.config.initial_state),
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get FSM statistics.

        Returns:
            Dictionary with transition count, state durations, etc.
        """
        stats = {
            "current_state": self.state.current_state.name,
            "current_frequency": self.state.current_frequency,
            "total_transitions": self.state.total_transitions,
            "candidate_state": (
                self.state.candidate_state.name if self.state.candidate_state else None
            ),
            "confirm_count": self.state.confirm_count,
        }

        # Calculate state durations if we have history
        if self.state.state_history:
            state_durations: dict[str, float] = {}
            for i, (time, state) in enumerate(self.state.state_history):
                if i < len(self.state.state_history) - 1:
                    next_time = self.state.state_history[i + 1][0]
                    duration = next_time - time
                else:
                    duration = 0  # Current state

                state_name = state.name
                state_durations[state_name] = state_durations.get(state_name, 0) + duration

            stats["state_durations"] = state_durations

        return stats

    def get_transition_rate(self, total_time: float) -> float:
        """Calculate transition rate per hour.

        Args:
            total_time: Total observation time in seconds.

        Returns:
            Transitions per hour.
        """
        if total_time <= 0:
            return 0.0
        return self.state.total_transitions * 3600.0 / total_time


class AdaptiveFSM(BehaviorFSM):
    """Extended FSM with adaptive hysteresis based on confidence.

    Adjusts confirmation threshold based on classifier confidence.
    High confidence predictions require fewer confirmations.
    """

    def __init__(
        self,
        config: FSMConfig,
        confidence_thresholds: tuple[float, float] = (0.7, 0.9),
        confirm_multipliers: tuple[float, float] = (1.0, 0.5),
    ):
        """Initialize the adaptive FSM.

        Args:
            config: FSM configuration.
            confidence_thresholds: (low, high) confidence thresholds.
            confirm_multipliers: Confirmation multipliers for (low, high) confidence.
        """
        super().__init__(config)
        self.confidence_thresholds = confidence_thresholds
        self.confirm_multipliers = confirm_multipliers

    def update_with_confidence(
        self,
        predicted_state: BehaviorCluster,
        confidence: float,
        current_time: float,
    ) -> tuple[BehaviorCluster, int, bool]:
        """Update FSM with confidence-adjusted hysteresis.

        Args:
            predicted_state: Classifier's predicted behavioral state.
            confidence: Prediction confidence (0-1).
            current_time: Current timestamp in seconds.

        Returns:
            Tuple of (current_state, sampling_frequency, state_changed).
        """
        state_changed = False

        # Adjust confirmation threshold based on confidence
        if confidence >= self.confidence_thresholds[1]:
            n_confirm_adjusted = max(1, int(self.config.n_confirm * self.confirm_multipliers[1]))
        elif confidence >= self.confidence_thresholds[0]:
            n_confirm_adjusted = self.config.n_confirm
        else:
            n_confirm_adjusted = int(self.config.n_confirm * self.confirm_multipliers[0] * 1.5)

        # Check if prediction matches current state
        if predicted_state == self.state.current_state:
            self.state.candidate_state = None
            self.state.confirm_count = 0
        else:
            if predicted_state == self.state.candidate_state:
                self.state.confirm_count += 1
            else:
                self.state.candidate_state = predicted_state
                self.state.confirm_count = 1

            if self.state.confirm_count >= n_confirm_adjusted:
                time_in_state = current_time - self.state.last_transition_time
                if time_in_state >= self.config.min_dwell_time:
                    self._transition_to(predicted_state, current_time)
                    state_changed = True

        return self.state.current_state, self.state.current_frequency, state_changed
