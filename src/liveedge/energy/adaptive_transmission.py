"""Adaptive transmission strategies for behavior-driven energy optimization.

This module implements behavior-aware adaptive transmission strategies
based on ECE5565 research, integrated with LiveEdge adaptive sampling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable
import numpy as np
from numpy.typing import NDArray

from liveedge.energy.real_hardware import (
    ICM20948Spec,
    NRF52840Spec,
    SX1262Spec,
    RealHardwareSpec,
    SensorMode,
    LoRaSpreadingFactor,
    LoRaBandwidth,
    LIVEEDGE_HARDWARE,
)


class BehaviorCluster(Enum):
    """Behavior clusters for adaptive strategies."""
    INACTIVE = auto()    # lying, standing, drinking (44%)
    MODERATE = auto()    # chewing, licking, grazing+standing (37%)
    ACTIVE = auto()      # walking, grazing+walking (19%)


class TransmissionStrategy(Enum):
    """Transmission strategy types."""
    FIXED = auto()           # Fixed interval (baseline)
    ADAPTIVE = auto()        # Behavior-dependent interval
    EVENT_TRIGGERED = auto() # State change + heartbeat


@dataclass
class BehaviorConfig:
    """Configuration for a behavior cluster.

    Attributes:
        cluster: Behavior cluster type.
        time_fraction: Fraction of time spent in this state (0-1).
        sampling_rate_hz: Adaptive sampling rate for this behavior.
        tx_interval_min: Transmission interval in minutes.
    """
    cluster: BehaviorCluster
    time_fraction: float
    sampling_rate_hz: float
    tx_interval_min: float

    @property
    def tx_per_hour(self) -> float:
        """Transmissions per hour for this behavior."""
        return 60.0 / self.tx_interval_min


# Default behavior configurations based on LiveEdge and ECE5565 research
BEHAVIOR_CONFIGS = {
    BehaviorCluster.INACTIVE: BehaviorConfig(
        cluster=BehaviorCluster.INACTIVE,
        time_fraction=0.44,
        sampling_rate_hz=15.0,
        tx_interval_min=60.0,  # Low priority, transmit once per hour
    ),
    BehaviorCluster.MODERATE: BehaviorConfig(
        cluster=BehaviorCluster.MODERATE,
        time_fraction=0.37,
        sampling_rate_hz=20.0,
        tx_interval_min=20.0,  # Medium priority
    ),
    BehaviorCluster.ACTIVE: BehaviorConfig(
        cluster=BehaviorCluster.ACTIVE,
        time_fraction=0.19,
        sampling_rate_hz=30.0,
        tx_interval_min=5.0,   # High priority, frequent monitoring
    ),
}


@dataclass
class MarkovBehaviorModel:
    """Markov chain model for behavior state transitions.

    Generates realistic behavior sequences based on transition probabilities.
    """
    # Transition matrix (rows: from state, cols: to state)
    # Order: INACTIVE, MODERATE, ACTIVE
    transition_matrix: NDArray[np.float64] = field(default_factory=lambda: np.array([
        [0.985, 0.012, 0.003],  # From INACTIVE
        [0.015, 0.975, 0.010],  # From MODERATE
        [0.005, 0.015, 0.980],  # From ACTIVE
    ]))

    # State order for indexing
    states: list[BehaviorCluster] = field(default_factory=lambda: [
        BehaviorCluster.INACTIVE,
        BehaviorCluster.MODERATE,
        BehaviorCluster.ACTIVE,
    ])

    def _get_state_index(self, state: BehaviorCluster) -> int:
        """Get index for a state."""
        return self.states.index(state)

    def get_next_state(
        self,
        current_state: BehaviorCluster,
        rng: np.random.Generator,
    ) -> BehaviorCluster:
        """Get next state based on transition probabilities.

        Args:
            current_state: Current behavior state.
            rng: Random number generator.

        Returns:
            Next behavior state.
        """
        current_idx = self._get_state_index(current_state)
        probs = self.transition_matrix[current_idx]
        next_idx = rng.choice(len(self.states), p=probs)
        return self.states[next_idx]

    def generate_sequence(
        self,
        duration_minutes: int,
        initial_state: BehaviorCluster = BehaviorCluster.INACTIVE,
        seed: int = 42,
    ) -> list[BehaviorCluster]:
        """Generate a behavior sequence.

        Args:
            duration_minutes: Sequence length in minutes.
            initial_state: Starting state.
            seed: Random seed for reproducibility.

        Returns:
            List of behavior states (one per minute).
        """
        rng = np.random.default_rng(seed)
        sequence = [initial_state]

        current_state = initial_state
        for _ in range(duration_minutes - 1):
            current_state = self.get_next_state(current_state, rng)
            sequence.append(current_state)

        return sequence

    def get_stationary_distribution(self) -> dict[BehaviorCluster, float]:
        """Calculate stationary distribution of the Markov chain.

        Returns:
            Dictionary mapping states to their stationary probabilities.
        """
        # Solve π = πP where P is transition matrix
        # This is the left eigenvector for eigenvalue 1
        eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)

        # Find eigenvector for eigenvalue ~1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()  # Normalize

        return {
            state: prob for state, prob in zip(self.states, stationary)
        }


@dataclass
class TransmissionEvent:
    """Record of a transmission event."""
    timestamp_min: float
    behavior: BehaviorCluster
    trigger: str  # "scheduled", "state_change", "heartbeat"
    energy_uj: float


@dataclass
class AdaptiveTransmissionConfig:
    """Configuration for adaptive transmission strategy."""
    strategy: TransmissionStrategy

    # Fixed strategy parameters
    fixed_interval_s: float = 90.0  # For FIXED strategy

    # Adaptive strategy parameters (per behavior)
    behavior_configs: dict[BehaviorCluster, BehaviorConfig] = field(
        default_factory=lambda: BEHAVIOR_CONFIGS.copy()
    )

    # Event-triggered parameters
    heartbeat_interval_min: float = 60.0  # Maximum time between transmissions

    # LoRa parameters
    payload_bytes: int = 10
    tx_power_dbm: int = 14
    spreading_factor: LoRaSpreadingFactor = LoRaSpreadingFactor.SF7
    bandwidth: LoRaBandwidth = LoRaBandwidth.BW_125


class AdaptiveTransmissionSimulator:
    """Simulator for behavior-aware adaptive transmission.

    Integrates with power decomposition framework to calculate
    system-level energy with adaptive transmission strategies.
    """

    def __init__(
        self,
        hardware: RealHardwareSpec | None = None,
        behavior_model: MarkovBehaviorModel | None = None,
    ):
        """Initialize simulator.

        Args:
            hardware: Hardware specifications.
            behavior_model: Markov behavior model.
        """
        self.hw = hardware or LIVEEDGE_HARDWARE
        self.behavior_model = behavior_model or MarkovBehaviorModel()

    def calculate_tx_energy_uj(
        self,
        config: AdaptiveTransmissionConfig,
    ) -> float:
        """Calculate energy for a single transmission in microjoules.

        Args:
            config: Transmission configuration.

        Returns:
            Energy in microjoules.
        """
        toa_ms = self.hw.radio.calculate_time_on_air_ms(
            config.payload_bytes,
            config.spreading_factor,
            config.bandwidth,
        )
        tx_power_mw = self.hw.radio.get_tx_power_mw(config.tx_power_dbm)

        # Add XOSC warmup (~1ms)
        warmup_energy_uj = self.hw.radio.STANDBY_XOSC_MA * self.hw.radio.vdd * 1.0
        tx_energy_uj = tx_power_mw * toa_ms  # mW * ms = uJ

        return tx_energy_uj + warmup_energy_uj

    def simulate_fixed_transmission(
        self,
        duration_minutes: int,
        interval_s: float,
        behavior_sequence: list[BehaviorCluster],
        config: AdaptiveTransmissionConfig,
    ) -> list[TransmissionEvent]:
        """Simulate fixed interval transmission.

        Args:
            duration_minutes: Simulation duration.
            interval_s: Fixed transmission interval in seconds.
            behavior_sequence: Behavior state sequence.
            config: Transmission configuration.

        Returns:
            List of transmission events.
        """
        events = []
        tx_energy = self.calculate_tx_energy_uj(config)

        # Calculate transmission times
        duration_s = duration_minutes * 60
        n_transmissions = int(duration_s / interval_s)

        for i in range(n_transmissions):
            tx_time_s = (i + 1) * interval_s
            tx_time_min = tx_time_s / 60.0

            # Get behavior at transmission time
            minute_idx = min(int(tx_time_min), len(behavior_sequence) - 1)
            behavior = behavior_sequence[minute_idx]

            events.append(TransmissionEvent(
                timestamp_min=tx_time_min,
                behavior=behavior,
                trigger="scheduled",
                energy_uj=tx_energy,
            ))

        return events

    def simulate_adaptive_transmission(
        self,
        duration_minutes: int,
        behavior_sequence: list[BehaviorCluster],
        config: AdaptiveTransmissionConfig,
    ) -> list[TransmissionEvent]:
        """Simulate behavior-adaptive transmission.

        Transmission interval depends on current behavior state.

        Args:
            duration_minutes: Simulation duration.
            behavior_sequence: Behavior state sequence.
            config: Transmission configuration.

        Returns:
            List of transmission events.
        """
        events = []
        tx_energy = self.calculate_tx_energy_uj(config)

        current_time_min = 0.0
        last_tx_time_min = 0.0

        while current_time_min < duration_minutes:
            minute_idx = min(int(current_time_min), len(behavior_sequence) - 1)
            current_behavior = behavior_sequence[minute_idx]

            # Get transmission interval for current behavior
            behavior_cfg = config.behavior_configs[current_behavior]
            tx_interval_min = behavior_cfg.tx_interval_min

            # Check if it's time to transmit
            time_since_last_tx = current_time_min - last_tx_time_min

            if time_since_last_tx >= tx_interval_min:
                events.append(TransmissionEvent(
                    timestamp_min=current_time_min,
                    behavior=current_behavior,
                    trigger="scheduled",
                    energy_uj=tx_energy,
                ))
                last_tx_time_min = current_time_min

            # Advance time by 1 minute
            current_time_min += 1.0

        return events

    def simulate_event_triggered_transmission(
        self,
        duration_minutes: int,
        behavior_sequence: list[BehaviorCluster],
        config: AdaptiveTransmissionConfig,
    ) -> list[TransmissionEvent]:
        """Simulate event-triggered transmission.

        Transmit on:
        1. Behavior state change
        2. Heartbeat (max interval exceeded)

        Args:
            duration_minutes: Simulation duration.
            behavior_sequence: Behavior state sequence.
            config: Transmission configuration.

        Returns:
            List of transmission events.
        """
        events = []
        tx_energy = self.calculate_tx_energy_uj(config)

        last_tx_time_min = 0.0
        last_behavior = behavior_sequence[0]

        for minute_idx, current_behavior in enumerate(behavior_sequence):
            current_time_min = float(minute_idx)
            time_since_last_tx = current_time_min - last_tx_time_min

            should_transmit = False
            trigger = ""

            # Check for state change
            if current_behavior != last_behavior:
                should_transmit = True
                trigger = "state_change"

            # Check for heartbeat
            elif time_since_last_tx >= config.heartbeat_interval_min:
                should_transmit = True
                trigger = "heartbeat"

            if should_transmit:
                events.append(TransmissionEvent(
                    timestamp_min=current_time_min,
                    behavior=current_behavior,
                    trigger=trigger,
                    energy_uj=tx_energy,
                ))
                last_tx_time_min = current_time_min

            last_behavior = current_behavior

        return events

    def simulate_transmission(
        self,
        duration_minutes: int,
        behavior_sequence: list[BehaviorCluster],
        config: AdaptiveTransmissionConfig,
    ) -> list[TransmissionEvent]:
        """Simulate transmission based on strategy.

        Args:
            duration_minutes: Simulation duration.
            behavior_sequence: Behavior state sequence.
            config: Transmission configuration.

        Returns:
            List of transmission events.
        """
        if config.strategy == TransmissionStrategy.FIXED:
            return self.simulate_fixed_transmission(
                duration_minutes,
                config.fixed_interval_s,
                behavior_sequence,
                config,
            )
        elif config.strategy == TransmissionStrategy.ADAPTIVE:
            return self.simulate_adaptive_transmission(
                duration_minutes,
                behavior_sequence,
                config,
            )
        elif config.strategy == TransmissionStrategy.EVENT_TRIGGERED:
            return self.simulate_event_triggered_transmission(
                duration_minutes,
                behavior_sequence,
                config,
            )
        else:
            raise ValueError(f"Unknown strategy: {config.strategy}")


@dataclass
class SamplingConfig:
    """Configuration for sampling strategy."""
    fixed: bool = True  # True for fixed sampling, False for adaptive
    fixed_rate_hz: float = 50.0  # For fixed sampling
    behavior_configs: dict[BehaviorCluster, BehaviorConfig] = field(
        default_factory=lambda: BEHAVIOR_CONFIGS.copy()
    )
    window_duration_s: float = 1.5
    cycle_duration_s: float = 90.0  # Time between classification cycles


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str
    description: str
    sampling: SamplingConfig
    transmission: AdaptiveTransmissionConfig
    duration_hours: float = 24.0
    num_trials: int = 5
    seeds: list[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])


@dataclass
class ComponentEnergy:
    """Energy breakdown by component."""
    imu_uj: float = 0.0
    mcu_sampling_uj: float = 0.0
    mcu_inference_uj: float = 0.0
    mcu_idle_uj: float = 0.0
    radio_tx_uj: float = 0.0
    radio_sleep_uj: float = 0.0

    @property
    def total_uj(self) -> float:
        """Total energy in microjoules."""
        return (
            self.imu_uj +
            self.mcu_sampling_uj +
            self.mcu_inference_uj +
            self.mcu_idle_uj +
            self.radio_tx_uj +
            self.radio_sleep_uj
        )

    @property
    def total_mj(self) -> float:
        """Total energy in millijoules."""
        return self.total_uj / 1000.0

    def get_percentages(self) -> dict[str, float]:
        """Get energy breakdown as percentages."""
        total = self.total_uj
        if total <= 0:
            return {}

        return {
            "imu_pct": self.imu_uj / total * 100,
            "mcu_sampling_pct": self.mcu_sampling_uj / total * 100,
            "mcu_inference_pct": self.mcu_inference_uj / total * 100,
            "mcu_idle_pct": self.mcu_idle_uj / total * 100,
            "radio_tx_pct": self.radio_tx_uj / total * 100,
            "radio_sleep_pct": self.radio_sleep_uj / total * 100,
        }

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "imu_uj": self.imu_uj,
            "mcu_sampling_uj": self.mcu_sampling_uj,
            "mcu_inference_uj": self.mcu_inference_uj,
            "mcu_idle_uj": self.mcu_idle_uj,
            "radio_tx_uj": self.radio_tx_uj,
            "radio_sleep_uj": self.radio_sleep_uj,
            "total_uj": self.total_uj,
            "total_mj": self.total_mj,
        }


@dataclass
class TrialResult:
    """Result from a single experiment trial."""
    experiment_id: str
    trial_id: int
    seed: int
    duration_hours: float

    # Behavior statistics
    behavior_sequence: list[BehaviorCluster]
    behavior_distribution: dict[BehaviorCluster, float]
    transition_count: int

    # Energy results
    energy: ComponentEnergy

    # Transmission statistics
    tx_count: int
    tx_events: list[TransmissionEvent]
    tx_per_hour: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "experiment_id": self.experiment_id,
            "trial_id": self.trial_id,
            "seed": self.seed,
            "duration_hours": self.duration_hours,
            "behavior_distribution": {
                b.name: v for b, v in self.behavior_distribution.items()
            },
            "transition_count": self.transition_count,
            "energy": self.energy.to_dict(),
            "energy_percentages": self.energy.get_percentages(),
            "tx_count": self.tx_count,
            "tx_per_hour": self.tx_per_hour,
        }


@dataclass
class ExperimentSummary:
    """Summary statistics across multiple trials."""
    experiment_id: str
    config: ExperimentConfig
    num_trials: int

    # Energy statistics
    energy_mean_mj: float
    energy_std_mj: float
    energy_ci_95: tuple[float, float]

    # Component breakdown (mean percentages)
    energy_breakdown_pct: dict[str, float]

    # Transmission statistics
    tx_per_hour_mean: float
    tx_per_hour_std: float

    # Battery life
    battery_life_days: float

    # Individual trials
    trials: list[TrialResult]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "config_name": self.config.name,
            "config_description": self.config.description,
            "num_trials": self.num_trials,
            "energy_mean_mj": self.energy_mean_mj,
            "energy_std_mj": self.energy_std_mj,
            "energy_ci_95_lower": self.energy_ci_95[0],
            "energy_ci_95_upper": self.energy_ci_95[1],
            "energy_breakdown_pct": self.energy_breakdown_pct,
            "tx_per_hour_mean": self.tx_per_hour_mean,
            "tx_per_hour_std": self.tx_per_hour_std,
            "battery_life_days": self.battery_life_days,
        }


class ValidationExperimentRunner:
    """Runner for adaptive transmission validation experiments.

    Implements T7.1-T7.9 scenarios from the experiment design.
    """

    def __init__(
        self,
        hardware: RealHardwareSpec | None = None,
    ):
        """Initialize experiment runner.

        Args:
            hardware: Hardware specifications.
        """
        self.hw = hardware or LIVEEDGE_HARDWARE
        self.behavior_model = MarkovBehaviorModel()
        self.tx_simulator = AdaptiveTransmissionSimulator(self.hw, self.behavior_model)

    def calculate_imu_energy(
        self,
        sampling_config: SamplingConfig,
        behavior_sequence: list[BehaviorCluster],
        duration_hours: float,
    ) -> float:
        """Calculate IMU energy consumption.

        Args:
            sampling_config: Sampling configuration.
            behavior_sequence: Behavior sequence.
            duration_hours: Simulation duration.

        Returns:
            IMU energy in microjoules.
        """
        duration_s = duration_hours * 3600

        if sampling_config.fixed:
            # Fixed sampling rate
            power_mw = self.hw.imu.get_power_mw(
                SensorMode.ACC_ONLY_LP,
                sampling_config.fixed_rate_hz
            )
            energy_uj = power_mw * duration_s * 1000  # mW * s * 1000 = uJ
        else:
            # Adaptive sampling - calculate based on behavior time fractions
            energy_uj = 0.0
            for behavior in [BehaviorCluster.INACTIVE, BehaviorCluster.MODERATE, BehaviorCluster.ACTIVE]:
                # Count minutes in this behavior
                minutes_in_behavior = sum(1 for b in behavior_sequence if b == behavior)
                duration_in_behavior_s = minutes_in_behavior * 60

                # Get sampling rate for this behavior
                rate_hz = sampling_config.behavior_configs[behavior].sampling_rate_hz
                power_mw = self.hw.imu.get_power_mw(SensorMode.ACC_ONLY_LP, rate_hz)

                energy_uj += power_mw * duration_in_behavior_s * 1000

        return energy_uj

    def calculate_mcu_sampling_energy(
        self,
        sampling_config: SamplingConfig,
        behavior_sequence: list[BehaviorCluster],
        duration_hours: float,
    ) -> float:
        """Calculate MCU energy for sampling operations.

        Args:
            sampling_config: Sampling configuration.
            behavior_sequence: Behavior sequence.
            duration_hours: Simulation duration.

        Returns:
            MCU sampling energy in microjoules.
        """
        duration_s = duration_hours * 3600
        n_cycles = int(duration_s / sampling_config.cycle_duration_s)

        # Time per sample read (I2C transaction)
        sample_read_time_ms = 1.0  # 1ms per sample

        if sampling_config.fixed:
            # Fixed sampling
            samples_per_cycle = int(
                sampling_config.fixed_rate_hz * sampling_config.window_duration_s
            )
            total_samples = n_cycles * samples_per_cycle
        else:
            # Adaptive sampling - estimate based on average rate
            avg_rate = sum(
                cfg.sampling_rate_hz * cfg.time_fraction
                for cfg in sampling_config.behavior_configs.values()
            )
            samples_per_cycle = int(avg_rate * sampling_config.window_duration_s)
            total_samples = n_cycles * samples_per_cycle

        # MCU active time for sampling
        sample_time_s = total_samples * sample_read_time_ms / 1000.0
        mcu_power_mw = self.hw.mcu.get_power_mw(active=True, with_peripherals=True)

        return mcu_power_mw * sample_time_s * 1000  # uJ

    def calculate_mcu_inference_energy(
        self,
        n_inferences: int,
        model_name: str = "random_forest_20",
    ) -> float:
        """Calculate MCU energy for inference operations.

        Args:
            n_inferences: Number of inference operations.
            model_name: ML model name.

        Returns:
            MCU inference energy in microjoules.
        """
        # Feature extraction + inference
        fe_energy = self.hw.mcu.get_inference_energy_uj("feature_extraction")
        inf_energy = self.hw.mcu.get_inference_energy_uj(model_name)

        return (fe_energy + inf_energy) * n_inferences

    def calculate_mcu_idle_energy(
        self,
        duration_hours: float,
        active_time_s: float,
    ) -> float:
        """Calculate MCU idle energy.

        Args:
            duration_hours: Total duration.
            active_time_s: Time MCU is active.

        Returns:
            MCU idle energy in microjoules.
        """
        duration_s = duration_hours * 3600
        idle_time_s = duration_s - active_time_s

        idle_power_uw = self.hw.mcu.SYSTEM_ON_IDLE_UA * self.hw.mcu.vdd
        return idle_power_uw * idle_time_s  # uW * s = uJ

    def calculate_radio_sleep_energy(
        self,
        duration_hours: float,
        tx_time_s: float,
    ) -> float:
        """Calculate radio sleep energy.

        Args:
            duration_hours: Total duration.
            tx_time_s: Total transmission time.

        Returns:
            Radio sleep energy in microjoules.
        """
        duration_s = duration_hours * 3600
        sleep_time_s = duration_s - tx_time_s

        sleep_power_uw = self.hw.radio.get_sleep_power_uw(warm_start=True)
        return sleep_power_uw * sleep_time_s  # uW * s = uJ

    def run_trial(
        self,
        config: ExperimentConfig,
        trial_id: int,
        seed: int,
    ) -> TrialResult:
        """Run a single experiment trial.

        Args:
            config: Experiment configuration.
            trial_id: Trial number.
            seed: Random seed.

        Returns:
            Trial result.
        """
        duration_minutes = int(config.duration_hours * 60)

        # Generate behavior sequence
        behavior_sequence = self.behavior_model.generate_sequence(
            duration_minutes,
            initial_state=BehaviorCluster.INACTIVE,
            seed=seed,
        )

        # Calculate behavior distribution
        behavior_counts = {b: 0 for b in BehaviorCluster}
        for b in behavior_sequence:
            behavior_counts[b] += 1
        behavior_distribution = {
            b: count / len(behavior_sequence)
            for b, count in behavior_counts.items()
        }

        # Count transitions
        transition_count = sum(
            1 for i in range(1, len(behavior_sequence))
            if behavior_sequence[i] != behavior_sequence[i-1]
        )

        # Simulate transmission
        tx_events = self.tx_simulator.simulate_transmission(
            duration_minutes,
            behavior_sequence,
            config.transmission,
        )
        tx_count = len(tx_events)
        tx_per_hour = tx_count / config.duration_hours

        # Calculate energy components
        imu_energy = self.calculate_imu_energy(
            config.sampling,
            behavior_sequence,
            config.duration_hours,
        )

        mcu_sampling_energy = self.calculate_mcu_sampling_energy(
            config.sampling,
            behavior_sequence,
            config.duration_hours,
        )

        # One inference per transmission
        mcu_inference_energy = self.calculate_mcu_inference_energy(tx_count)

        # Calculate active time
        duration_s = config.duration_hours * 3600
        n_cycles = int(duration_s / config.sampling.cycle_duration_s)

        if config.sampling.fixed:
            samples_per_cycle = int(
                config.sampling.fixed_rate_hz * config.sampling.window_duration_s
            )
        else:
            avg_rate = sum(
                cfg.sampling_rate_hz * cfg.time_fraction
                for cfg in config.sampling.behavior_configs.values()
            )
            samples_per_cycle = int(avg_rate * config.sampling.window_duration_s)

        sample_time_s = n_cycles * samples_per_cycle * 0.001  # 1ms per sample
        inference_time_s = tx_count * 12.0 / 1000.0  # 12ms per inference
        active_time_s = sample_time_s + inference_time_s

        mcu_idle_energy = self.calculate_mcu_idle_energy(
            config.duration_hours,
            active_time_s,
        )

        # Radio energy
        radio_tx_energy = sum(e.energy_uj for e in tx_events)

        toa_s = self.hw.radio.calculate_time_on_air_ms(
            config.transmission.payload_bytes,
            config.transmission.spreading_factor,
            config.transmission.bandwidth,
        ) / 1000.0
        total_tx_time_s = tx_count * toa_s

        radio_sleep_energy = self.calculate_radio_sleep_energy(
            config.duration_hours,
            total_tx_time_s,
        )

        energy = ComponentEnergy(
            imu_uj=imu_energy,
            mcu_sampling_uj=mcu_sampling_energy,
            mcu_inference_uj=mcu_inference_energy,
            mcu_idle_uj=mcu_idle_energy,
            radio_tx_uj=radio_tx_energy,
            radio_sleep_uj=radio_sleep_energy,
        )

        return TrialResult(
            experiment_id=config.name,
            trial_id=trial_id,
            seed=seed,
            duration_hours=config.duration_hours,
            behavior_sequence=behavior_sequence,
            behavior_distribution=behavior_distribution,
            transition_count=transition_count,
            energy=energy,
            tx_count=tx_count,
            tx_events=tx_events,
            tx_per_hour=tx_per_hour,
        )

    def run_experiment(
        self,
        config: ExperimentConfig,
    ) -> ExperimentSummary:
        """Run a complete experiment with multiple trials.

        Args:
            config: Experiment configuration.

        Returns:
            Experiment summary with statistics.
        """
        trials = []

        for trial_id, seed in enumerate(config.seeds[:config.num_trials], 1):
            trial = self.run_trial(config, trial_id, seed)
            trials.append(trial)

        # Calculate statistics
        energies_mj = [t.energy.total_mj for t in trials]
        energy_mean = np.mean(energies_mj)
        energy_std = np.std(energies_mj, ddof=1)

        # 95% CI
        se = energy_std / np.sqrt(len(energies_mj))
        ci_95 = (energy_mean - 1.96 * se, energy_mean + 1.96 * se)

        # Average breakdown percentages
        breakdown_pcts = [t.energy.get_percentages() for t in trials]
        avg_breakdown = {}
        if breakdown_pcts:
            keys = breakdown_pcts[0].keys()
            for key in keys:
                avg_breakdown[key] = np.mean([bp.get(key, 0) for bp in breakdown_pcts])

        # TX statistics
        tx_rates = [t.tx_per_hour for t in trials]
        tx_mean = np.mean(tx_rates)
        tx_std = np.std(tx_rates, ddof=1)

        # Battery life (based on mean daily energy)
        daily_energy_mj = energy_mean
        daily_energy_j = daily_energy_mj / 1000.0
        battery_capacity_j = self.hw.battery_capacity_j
        battery_life_days = battery_capacity_j / daily_energy_j

        return ExperimentSummary(
            experiment_id=config.name,
            config=config,
            num_trials=len(trials),
            energy_mean_mj=energy_mean,
            energy_std_mj=energy_std,
            energy_ci_95=ci_95,
            energy_breakdown_pct=avg_breakdown,
            tx_per_hour_mean=tx_mean,
            tx_per_hour_std=tx_std,
            battery_life_days=battery_life_days,
            trials=trials,
        )


# Pre-defined experiment configurations

def create_t71_config() -> ExperimentConfig:
    """T7.1: Fixed sampling (50Hz) + Fixed transmission (90s)."""
    return ExperimentConfig(
        name="T7.1",
        description="Fixed 50Hz sampling + Fixed 90s transmission (Baseline)",
        sampling=SamplingConfig(
            fixed=True,
            fixed_rate_hz=50.0,
        ),
        transmission=AdaptiveTransmissionConfig(
            strategy=TransmissionStrategy.FIXED,
            fixed_interval_s=90.0,
        ),
    )


def create_t72_config() -> ExperimentConfig:
    """T7.2: Adaptive sampling + Fixed transmission (90s)."""
    return ExperimentConfig(
        name="T7.2",
        description="Adaptive 15-30Hz sampling + Fixed 90s transmission",
        sampling=SamplingConfig(
            fixed=False,
        ),
        transmission=AdaptiveTransmissionConfig(
            strategy=TransmissionStrategy.FIXED,
            fixed_interval_s=90.0,
        ),
    )


def create_t77_config() -> ExperimentConfig:
    """T7.7: Fixed sampling (50Hz) + Adaptive transmission."""
    return ExperimentConfig(
        name="T7.7",
        description="Fixed 50Hz sampling + Adaptive transmission (5-60min)",
        sampling=SamplingConfig(
            fixed=True,
            fixed_rate_hz=50.0,
        ),
        transmission=AdaptiveTransmissionConfig(
            strategy=TransmissionStrategy.ADAPTIVE,
        ),
    )


def create_t78_config() -> ExperimentConfig:
    """T7.8: Adaptive sampling + Adaptive transmission (Full optimization)."""
    return ExperimentConfig(
        name="T7.8",
        description="Adaptive 15-30Hz sampling + Adaptive transmission (Full optimization)",
        sampling=SamplingConfig(
            fixed=False,
        ),
        transmission=AdaptiveTransmissionConfig(
            strategy=TransmissionStrategy.ADAPTIVE,
        ),
    )


def create_t79_config() -> ExperimentConfig:
    """T7.9: Adaptive sampling + Event-triggered transmission (Extreme optimization)."""
    return ExperimentConfig(
        name="T7.9",
        description="Adaptive 15-30Hz sampling + Event-triggered transmission (Extreme)",
        sampling=SamplingConfig(
            fixed=False,
        ),
        transmission=AdaptiveTransmissionConfig(
            strategy=TransmissionStrategy.EVENT_TRIGGERED,
            heartbeat_interval_min=60.0,
        ),
    )


def get_all_experiment_configs() -> list[ExperimentConfig]:
    """Get all experiment configurations."""
    return [
        create_t71_config(),
        create_t72_config(),
        create_t77_config(),
        create_t78_config(),
        create_t79_config(),
    ]
