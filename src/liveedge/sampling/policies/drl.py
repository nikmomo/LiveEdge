"""Deep Reinforcement Learning based sampling policy.

This module implements DRL-based adaptive sampling using stable-baselines3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from liveedge.clustering.clustering import BehaviorCluster
from liveedge.sampling.policies.base import (
    BasePolicy,
    PolicyConfig,
    SamplingDecision,
)


@dataclass
class DRLPolicyConfig(PolicyConfig):
    """Configuration for DRL-based policy.

    Attributes:
        model_type: RL algorithm type (PPO, SAC, DQN).
        learning_rate: Learning rate for training.
        gamma: Discount factor.
        n_steps: Steps per update for on-policy algorithms.
        batch_size: Batch size for training.
        hidden_sizes: Hidden layer sizes for policy network.
        energy_weight: Weight for energy penalty in reward.
        accuracy_weight: Weight for accuracy reward.
        state_dim: Dimension of state space.
        checkpoint_path: Path to load/save model checkpoints.
    """

    model_type: str = "PPO"
    learning_rate: float = 3e-4
    gamma: float = 0.99
    n_steps: int = 2048
    batch_size: int = 64
    hidden_sizes: list[int] = field(default_factory=lambda: [256, 256])
    energy_weight: float = 0.3
    accuracy_weight: float = 0.7
    state_dim: int = 10
    checkpoint_path: str | None = None


class SamplingEnvironment:
    """Gymnasium-compatible environment for adaptive sampling.

    The agent learns to select sampling rates based on the current
    behavioral state and sensor readings.
    """

    def __init__(
        self,
        available_rates: list[int],
        state_dim: int = 10,
        energy_weight: float = 0.3,
        accuracy_weight: float = 0.7,
        max_steps: int = 1000,
    ):
        """Initialize the environment.

        Args:
            available_rates: List of available sampling rates.
            state_dim: Dimension of observation space.
            energy_weight: Weight for energy penalty.
            accuracy_weight: Weight for accuracy reward.
            max_steps: Maximum steps per episode.
        """
        self.available_rates = available_rates
        self.state_dim = state_dim
        self.energy_weight = energy_weight
        self.accuracy_weight = accuracy_weight
        self.max_steps = max_steps

        self.n_actions = len(available_rates)
        self.current_step = 0
        self.current_state: NDArray[np.float32] | None = None
        self.episode_data: list[dict[str, Any]] = []

        # Energy consumption per rate (normalized)
        self._rate_energy = {
            rate: rate / max(available_rates) for rate in available_rates
        }

    @property
    def observation_space_shape(self) -> tuple[int, ...]:
        """Get observation space shape."""
        return (self.state_dim,)

    @property
    def action_space_n(self) -> int:
        """Get number of actions."""
        return self.n_actions

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Reset the environment.

        Args:
            seed: Random seed.
            options: Additional options.

        Returns:
            Initial observation and info dict.
        """
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.episode_data = []
        self.current_state = np.zeros(self.state_dim, dtype=np.float32)

        return self.current_state, {}

    def step(
        self,
        action: int,
        true_label: int | None = None,
        predicted_label: int | None = None,
        next_features: NDArray[np.float32] | None = None,
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Take a step in the environment.

        Args:
            action: Action index (sampling rate selection).
            true_label: Ground truth label (for reward computation).
            predicted_label: Model prediction.
            next_features: Features for next state.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        selected_rate = self.available_rates[action]

        # Compute reward
        reward = self._compute_reward(
            selected_rate,
            true_label,
            predicted_label,
        )

        # Update state
        if next_features is not None:
            self.current_state = next_features.astype(np.float32)
        else:
            self.current_state = np.random.randn(self.state_dim).astype(np.float32)

        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps

        info = {
            "selected_rate": selected_rate,
            "energy_cost": self._rate_energy[selected_rate],
        }

        self.episode_data.append({
            "step": self.current_step,
            "action": action,
            "rate": selected_rate,
            "reward": reward,
        })

        return self.current_state, reward, terminated, truncated, info

    def _compute_reward(
        self,
        selected_rate: int,
        true_label: int | None,
        predicted_label: int | None,
    ) -> float:
        """Compute reward for the current step.

        Args:
            selected_rate: Selected sampling rate.
            true_label: Ground truth label.
            predicted_label: Model prediction.

        Returns:
            Reward value.
        """
        # Energy penalty (normalized)
        energy_penalty = -self._rate_energy[selected_rate]

        # Accuracy reward
        if true_label is not None and predicted_label is not None:
            accuracy_reward = 1.0 if true_label == predicted_label else -0.5
        else:
            accuracy_reward = 0.0

        reward = (
            self.energy_weight * energy_penalty
            + self.accuracy_weight * accuracy_reward
        )

        return reward


class DRLPolicy(BasePolicy):
    """Deep Reinforcement Learning based sampling policy.

    Uses PPO, SAC, or DQN to learn optimal sampling rate selection.
    """

    def __init__(self, config: DRLPolicyConfig | None = None):
        """Initialize the DRL policy.

        Args:
            config: Policy configuration.
        """
        if config is None:
            config = DRLPolicyConfig()
        super().__init__(config)

        self.config: DRLPolicyConfig = config
        self.model = None
        self.env: SamplingEnvironment | None = None
        self._is_trained = False

    def _create_model(self) -> Any:
        """Create the RL model.

        Returns:
            Stable-baselines3 model instance.
        """
        try:
            from stable_baselines3 import DQN, PPO, SAC
            from stable_baselines3.common.env_util import make_vec_env
            import gymnasium as gym
        except ImportError:
            raise ImportError(
                "stable-baselines3 and gymnasium are required for DRL policy. "
                "Install with: pip install stable-baselines3 gymnasium"
            )

        # Create gymnasium wrapper
        env = self._create_gym_env()

        policy_kwargs = dict(
            net_arch=self.config.hidden_sizes,
        )

        model_classes = {
            "PPO": PPO,
            "SAC": SAC,
            "DQN": DQN,
        }

        model_class = model_classes.get(self.config.model_type)
        if model_class is None:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

        if self.config.model_type == "PPO":
            model = model_class(
                "MlpPolicy",
                env,
                learning_rate=self.config.learning_rate,
                gamma=self.config.gamma,
                n_steps=self.config.n_steps,
                batch_size=self.config.batch_size,
                policy_kwargs=policy_kwargs,
                verbose=0,
            )
        elif self.config.model_type == "DQN":
            model = model_class(
                "MlpPolicy",
                env,
                learning_rate=self.config.learning_rate,
                gamma=self.config.gamma,
                batch_size=self.config.batch_size,
                policy_kwargs=policy_kwargs,
                verbose=0,
            )
        else:  # SAC
            model = model_class(
                "MlpPolicy",
                env,
                learning_rate=self.config.learning_rate,
                gamma=self.config.gamma,
                batch_size=self.config.batch_size,
                policy_kwargs=policy_kwargs,
                verbose=0,
            )

        return model

    def _create_gym_env(self) -> Any:
        """Create a gymnasium-compatible environment.

        Returns:
            Gymnasium environment.
        """
        import gymnasium as gym
        from gymnasium import spaces

        class AdaptiveSamplingEnv(gym.Env):
            """Gymnasium wrapper for SamplingEnvironment."""

            def __init__(inner_self, sampling_env: SamplingEnvironment):
                super().__init__()
                inner_self.env = sampling_env
                inner_self.observation_space = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=sampling_env.observation_space_shape,
                    dtype=np.float32,
                )
                inner_self.action_space = spaces.Discrete(sampling_env.action_space_n)

            def reset(inner_self, *, seed=None, options=None):
                return inner_self.env.reset(seed=seed, options=options)

            def step(inner_self, action):
                return inner_self.env.step(action)

        self.env = SamplingEnvironment(
            available_rates=self.config.available_rates,
            state_dim=self.config.state_dim,
            energy_weight=self.config.energy_weight,
            accuracy_weight=self.config.accuracy_weight,
        )

        return AdaptiveSamplingEnv(self.env)

    def select_rate(
        self,
        current_state: BehaviorCluster,
        variance: float | None = None,
        confidence: float | None = None,
        features: NDArray[np.float32] | None = None,
        **kwargs: Any,
    ) -> SamplingDecision:
        """Select sampling rate using learned policy.

        Args:
            current_state: Current behavioral state.
            variance: Signal variance.
            confidence: Classification confidence.
            features: Feature vector for policy input.
            **kwargs: Additional arguments.

        Returns:
            SamplingDecision with selected rate and metadata.
        """
        if not self._is_trained and self.model is None:
            # Return default rate if not trained
            default_rate = self.config.state_frequencies.get(
                current_state, self.config.available_rates[len(self.config.available_rates) // 2]
            )
            return SamplingDecision(
                rate=default_rate,
                state=current_state,
                confidence=confidence or 0.0,
                metadata={"policy": "default", "reason": "model_not_trained"},
            )

        # Build observation
        obs = self._build_observation(current_state, variance, confidence, features)

        # Get action from model
        action, _ = self.model.predict(obs, deterministic=True)
        selected_rate = self.config.available_rates[int(action)]

        return SamplingDecision(
            rate=selected_rate,
            state=current_state,
            confidence=confidence or 0.0,
            metadata={
                "policy": "drl",
                "action": int(action),
                "model_type": self.config.model_type,
            },
        )

    def _build_observation(
        self,
        current_state: BehaviorCluster,
        variance: float | None,
        confidence: float | None,
        features: NDArray[np.float32] | None,
    ) -> NDArray[np.float32]:
        """Build observation vector for the policy.

        Args:
            current_state: Current behavioral state.
            variance: Signal variance.
            confidence: Classification confidence.
            features: Additional features.

        Returns:
            Observation vector.
        """
        obs = np.zeros(self.config.state_dim, dtype=np.float32)

        # One-hot encode state (first 5 dimensions)
        state_idx = list(BehaviorCluster).index(current_state)
        obs[state_idx] = 1.0

        # Add variance and confidence (dimensions 5-6)
        if variance is not None:
            obs[5] = np.clip(variance, 0, 10) / 10.0
        if confidence is not None:
            obs[6] = confidence

        # Add additional features if provided
        if features is not None:
            n_extra = min(len(features), self.config.state_dim - 7)
            obs[7:7 + n_extra] = features[:n_extra]

        return obs

    def train(
        self,
        total_timesteps: int = 100000,
        callback: Any | None = None,
        progress_bar: bool = True,
    ) -> dict[str, Any]:
        """Train the DRL policy.

        Args:
            total_timesteps: Total training timesteps.
            callback: Training callback.
            progress_bar: Show progress bar.

        Returns:
            Training statistics.
        """
        if self.model is None:
            self.model = self._create_model()

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=progress_bar,
        )

        self._is_trained = True

        return {
            "total_timesteps": total_timesteps,
            "model_type": self.config.model_type,
        }

    def save(self, path: str | Path) -> None:
        """Save the trained model.

        Args:
            path: Path to save the model.
        """
        if self.model is None:
            raise ValueError("No model to save. Train first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))

    def load(self, path: str | Path) -> None:
        """Load a trained model.

        Args:
            path: Path to the saved model.
        """
        try:
            from stable_baselines3 import DQN, PPO, SAC
        except ImportError:
            raise ImportError(
                "stable-baselines3 is required. "
                "Install with: pip install stable-baselines3"
            )

        model_classes = {
            "PPO": PPO,
            "SAC": SAC,
            "DQN": DQN,
        }

        model_class = model_classes.get(self.config.model_type)
        if model_class is None:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

        # Create environment for loading
        env = self._create_gym_env()
        self.model = model_class.load(str(path), env=env)
        self._is_trained = True

    def get_statistics(self) -> dict[str, Any]:
        """Get policy statistics.

        Returns:
            Dictionary with policy statistics.
        """
        stats = super().get_statistics()
        stats.update({
            "model_type": self.config.model_type,
            "is_trained": self._is_trained,
            "state_dim": self.config.state_dim,
            "energy_weight": self.config.energy_weight,
            "accuracy_weight": self.config.accuracy_weight,
        })
        return stats
