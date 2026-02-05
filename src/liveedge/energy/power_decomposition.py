"""Power decomposition simulation framework.

This module provides a comprehensive simulation framework for analyzing
system-level power consumption with different sensor configurations and
operating modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any
import numpy as np

from liveedge.energy.real_hardware import (
    ICM20948Spec,
    NRF52840Spec,
    SX1262Spec,
    RealHardwareSpec,
    SensorMode,
    LoRaSpreadingFactor,
    LoRaBandwidth,
)


class WorkPhase(Enum):
    """System work phases during a duty cycle."""
    SLEEP = auto()         # All components in low-power mode
    SENSING = auto()       # IMU actively sampling
    PROCESSING = auto()    # MCU running inference
    TRANSMITTING = auto()  # Radio transmitting
    RECEIVING = auto()     # Radio receiving
    IDLE = auto()          # MCU idle, waiting


@dataclass
class SensorConfiguration:
    """Configuration for sensor data collection.

    Defines which sensors are enabled and at what rates.
    """
    name: str
    mode: SensorMode
    acc_enabled: bool = True
    gyro_enabled: bool = False
    mag_enabled: bool = False
    acc_odr_hz: float = 50.0
    gyro_odr_hz: float = 50.0
    mag_odr_hz: float = 8.0  # AK09916: 8Hz or 100Hz only

    @property
    def effective_odr_hz(self) -> float:
        """Get the effective (highest) ODR for timing calculations."""
        odrs = []
        if self.acc_enabled:
            odrs.append(self.acc_odr_hz)
        if self.gyro_enabled:
            odrs.append(self.gyro_odr_hz)
        if self.mag_enabled:
            odrs.append(self.mag_odr_hz)
        return max(odrs) if odrs else 50.0

    @property
    def bytes_per_sample(self) -> int:
        """Calculate bytes per sample based on enabled sensors.

        Each axis is 2 bytes (16-bit ADC).
        """
        bytes_count = 0
        if self.acc_enabled:
            bytes_count += 6  # 3 axes × 2 bytes
        if self.gyro_enabled:
            bytes_count += 6
        if self.mag_enabled:
            bytes_count += 6
        return bytes_count


# Pre-defined sensor configurations
SENSOR_CONFIG_ACC_ONLY_5HZ = SensorConfiguration(
    name="ACC_ONLY_5Hz",
    mode=SensorMode.ACC_ONLY_LP,
    acc_enabled=True,
    gyro_enabled=False,
    mag_enabled=False,
    acc_odr_hz=5.0,
)

SENSOR_CONFIG_ACC_ONLY_15HZ = SensorConfiguration(
    name="ACC_ONLY_15Hz",
    mode=SensorMode.ACC_ONLY_LP,
    acc_enabled=True,
    gyro_enabled=False,
    mag_enabled=False,
    acc_odr_hz=15.0,
)

SENSOR_CONFIG_ACC_ONLY_25HZ = SensorConfiguration(
    name="ACC_ONLY_25Hz",
    mode=SensorMode.ACC_ONLY_LP,
    acc_enabled=True,
    gyro_enabled=False,
    mag_enabled=False,
    acc_odr_hz=25.0,
)

SENSOR_CONFIG_ACC_ONLY_50HZ = SensorConfiguration(
    name="ACC_ONLY_50Hz",
    mode=SensorMode.ACC_ONLY_LP,
    acc_enabled=True,
    gyro_enabled=False,
    mag_enabled=False,
    acc_odr_hz=50.0,
)

SENSOR_CONFIG_ACC_ONLY_100HZ = SensorConfiguration(
    name="ACC_ONLY_100Hz",
    mode=SensorMode.ACC_ONLY_LP,
    acc_enabled=True,
    gyro_enabled=False,
    mag_enabled=False,
    acc_odr_hz=100.0,
)

SENSOR_CONFIG_ACC_GYRO_50HZ = SensorConfiguration(
    name="ACC_GYRO_50Hz",
    mode=SensorMode.ACC_GYRO_LP,
    acc_enabled=True,
    gyro_enabled=True,
    mag_enabled=False,
    acc_odr_hz=50.0,
    gyro_odr_hz=50.0,
)

SENSOR_CONFIG_9AXIS_50HZ = SensorConfiguration(
    name="9AXIS_50Hz",
    mode=SensorMode.NINE_AXIS_LN,
    acc_enabled=True,
    gyro_enabled=True,
    mag_enabled=True,
    acc_odr_hz=50.0,
    gyro_odr_hz=50.0,
    mag_odr_hz=100.0,
)

# Adaptive configurations (variable ODR)
SENSOR_CONFIG_ADAPTIVE_LOW = SensorConfiguration(
    name="ADAPTIVE_LOW",
    mode=SensorMode.ACC_ONLY_LP,
    acc_enabled=True,
    acc_odr_hz=15.0,  # Average for low activity
)

SENSOR_CONFIG_ADAPTIVE_MEDIUM = SensorConfiguration(
    name="ADAPTIVE_MEDIUM",
    mode=SensorMode.ACC_ONLY_LP,
    acc_enabled=True,
    acc_odr_hz=25.0,  # Average for medium activity
)


@dataclass
class TransmissionConfiguration:
    """Configuration for data transmission."""
    name: str
    payload_type: str  # "raw_data", "features", "classification"
    payload_bytes: int
    tx_power_dbm: int = 14
    spreading_factor: LoRaSpreadingFactor = LoRaSpreadingFactor.SF7
    bandwidth: LoRaBandwidth = LoRaBandwidth.BW_125
    tx_interval_s: float = 90.0  # Default 90 second interval


# Pre-defined transmission configurations
TX_CONFIG_CLASSIFICATION = TransmissionConfiguration(
    name="CLASSIFICATION_RESULT",
    payload_type="classification",
    payload_bytes=10,  # Timestamp (4) + behavior (1) + confidence (4) + checksum (1)
    tx_power_dbm=14,
    spreading_factor=LoRaSpreadingFactor.SF7,
    tx_interval_s=90.0,
)

TX_CONFIG_FEATURES = TransmissionConfiguration(
    name="FEATURES",
    payload_type="features",
    payload_bytes=50,  # ~12 features × 4 bytes + overhead
    tx_power_dbm=14,
    spreading_factor=LoRaSpreadingFactor.SF7,
    tx_interval_s=90.0,
)

TX_CONFIG_RAW_DATA = TransmissionConfiguration(
    name="RAW_DATA",
    payload_type="raw_data",
    payload_bytes=100,  # ~33 samples × 3 bytes per axis
    tx_power_dbm=14,
    spreading_factor=LoRaSpreadingFactor.SF7,
    tx_interval_s=90.0,
)

TX_CONFIG_RAW_SF10 = TransmissionConfiguration(
    name="RAW_DATA_SF10",
    payload_type="raw_data",
    payload_bytes=100,
    tx_power_dbm=14,
    spreading_factor=LoRaSpreadingFactor.SF10,
    tx_interval_s=90.0,
)

TX_CONFIG_RAW_SF12 = TransmissionConfiguration(
    name="RAW_DATA_SF12",
    payload_type="raw_data",
    payload_bytes=100,
    tx_power_dbm=14,
    spreading_factor=LoRaSpreadingFactor.SF12,
    tx_interval_s=90.0,
)


@dataclass
class DutyCycleResult:
    """Results from a single duty cycle simulation."""
    duration_s: float
    sensor_config: SensorConfiguration
    tx_config: TransmissionConfiguration

    # Energy breakdown in microjoules (uJ)
    imu_energy_uj: float = 0.0
    mcu_sampling_energy_uj: float = 0.0
    mcu_inference_energy_uj: float = 0.0
    mcu_idle_energy_uj: float = 0.0
    radio_tx_energy_uj: float = 0.0
    radio_sleep_energy_uj: float = 0.0
    system_sleep_energy_uj: float = 0.0

    @property
    def total_energy_uj(self) -> float:
        """Total energy in microjoules."""
        return (
            self.imu_energy_uj +
            self.mcu_sampling_energy_uj +
            self.mcu_inference_energy_uj +
            self.mcu_idle_energy_uj +
            self.radio_tx_energy_uj +
            self.radio_sleep_energy_uj +
            self.system_sleep_energy_uj
        )

    @property
    def total_energy_mj(self) -> float:
        """Total energy in millijoules."""
        return self.total_energy_uj / 1000.0

    @property
    def avg_power_uw(self) -> float:
        """Average power in microwatts."""
        if self.duration_s <= 0:
            return 0.0
        return self.total_energy_uj / self.duration_s

    @property
    def avg_power_mw(self) -> float:
        """Average power in milliwatts."""
        return self.avg_power_uw / 1000.0

    def get_breakdown_dict(self) -> dict[str, float]:
        """Get energy breakdown as dictionary."""
        return {
            "imu_energy_uj": self.imu_energy_uj,
            "mcu_sampling_energy_uj": self.mcu_sampling_energy_uj,
            "mcu_inference_energy_uj": self.mcu_inference_energy_uj,
            "mcu_idle_energy_uj": self.mcu_idle_energy_uj,
            "radio_tx_energy_uj": self.radio_tx_energy_uj,
            "radio_sleep_energy_uj": self.radio_sleep_energy_uj,
            "system_sleep_energy_uj": self.system_sleep_energy_uj,
            "total_energy_uj": self.total_energy_uj,
        }

    def get_breakdown_percentages(self) -> dict[str, float]:
        """Get energy breakdown as percentages."""
        total = self.total_energy_uj
        if total <= 0:
            return {}

        return {
            "imu_pct": self.imu_energy_uj / total * 100,
            "mcu_sampling_pct": self.mcu_sampling_energy_uj / total * 100,
            "mcu_inference_pct": self.mcu_inference_energy_uj / total * 100,
            "mcu_idle_pct": self.mcu_idle_energy_uj / total * 100,
            "radio_tx_pct": self.radio_tx_energy_uj / total * 100,
            "radio_sleep_pct": self.radio_sleep_energy_uj / total * 100,
            "system_sleep_pct": self.system_sleep_energy_uj / total * 100,
        }


class PowerDecompositionSimulator:
    """Simulator for system-level power decomposition.

    Simulates energy consumption for different sensor configurations,
    transmission strategies, and operating modes.
    """

    def __init__(self, hardware: RealHardwareSpec | None = None):
        """Initialize simulator with hardware specifications.

        Args:
            hardware: Hardware specifications. Defaults to LIVEEDGE_HARDWARE.
        """
        from liveedge.energy.real_hardware import LIVEEDGE_HARDWARE
        self.hw = hardware or LIVEEDGE_HARDWARE

    def simulate_sensing_phase(
        self,
        sensor_config: SensorConfiguration,
        window_duration_s: float = 1.5,
    ) -> tuple[float, float]:
        """Simulate sensing phase energy consumption.

        Args:
            sensor_config: Sensor configuration.
            window_duration_s: Window duration in seconds.

        Returns:
            Tuple of (IMU energy in uJ, MCU sampling energy in uJ).
        """
        # IMU energy
        imu_power_mw = self.hw.imu.get_power_mw(
            sensor_config.mode,
            sensor_config.effective_odr_hz
        )
        imu_energy_uj = imu_power_mw * window_duration_s * 1000  # mW * s = mJ, *1000 = uJ

        # MCU energy for sampling (active during I2C/SPI reads)
        n_samples = int(sensor_config.effective_odr_hz * window_duration_s)
        # Assume 500us per sample read (I2C transaction)
        sample_read_time_s = n_samples * 500e-6

        mcu_power_mw = self.hw.mcu.get_power_mw(active=True, with_peripherals=True)
        mcu_sampling_energy_uj = mcu_power_mw * sample_read_time_s * 1000

        return imu_energy_uj, mcu_sampling_energy_uj

    def simulate_processing_phase(
        self,
        model_name: str = "random_forest_20",
        include_feature_extraction: bool = True,
    ) -> float:
        """Simulate processing phase energy consumption.

        Args:
            model_name: ML model name for inference timing.
            include_feature_extraction: Whether to include feature extraction time.

        Returns:
            MCU inference energy in uJ.
        """
        total_energy_uj = 0.0

        if include_feature_extraction:
            fe_time_ms = self.hw.mcu.get_inference_time_ms("feature_extraction")
            fe_power_mw = self.hw.mcu.get_power_mw(active=True)
            total_energy_uj += fe_power_mw * fe_time_ms  # mW * ms = uJ

        inference_energy_uj = self.hw.mcu.get_inference_energy_uj(model_name)
        total_energy_uj += inference_energy_uj

        return total_energy_uj

    def simulate_transmission_phase(
        self,
        tx_config: TransmissionConfiguration,
    ) -> float:
        """Simulate transmission phase energy consumption.

        Args:
            tx_config: Transmission configuration.

        Returns:
            Radio TX energy in uJ.
        """
        # Calculate time-on-air
        toa_ms = self.hw.radio.calculate_time_on_air_ms(
            tx_config.payload_bytes,
            tx_config.spreading_factor,
            tx_config.bandwidth,
        )

        # TX power consumption
        tx_power_mw = self.hw.radio.get_tx_power_mw(tx_config.tx_power_dbm)

        # Energy in uJ (mW * ms = uJ)
        tx_energy_uj = tx_power_mw * toa_ms

        # Add standby time for XOSC warmup (~1ms)
        standby_energy_uj = self.hw.radio.STANDBY_XOSC_MA * self.hw.radio.vdd * 1.0

        return tx_energy_uj + standby_energy_uj

    def simulate_idle_phase(
        self,
        duration_s: float,
    ) -> tuple[float, float]:
        """Simulate idle/sleep phase energy consumption.

        Args:
            duration_s: Idle duration in seconds.

        Returns:
            Tuple of (MCU idle energy in uJ, Radio sleep energy in uJ).
        """
        # MCU in System ON idle (RAM retained)
        mcu_idle_power_uw = self.hw.mcu.SYSTEM_ON_IDLE_UA * self.hw.mcu.vdd
        mcu_idle_energy_uj = mcu_idle_power_uw * duration_s

        # Radio in warm sleep
        radio_sleep_power_uw = self.hw.radio.get_sleep_power_uw(warm_start=True)
        radio_sleep_energy_uj = radio_sleep_power_uw * duration_s

        return mcu_idle_energy_uj, radio_sleep_energy_uj

    def simulate_duty_cycle(
        self,
        sensor_config: SensorConfiguration,
        tx_config: TransmissionConfiguration,
        model_name: str = "random_forest_20",
        window_duration_s: float = 1.5,
    ) -> DutyCycleResult:
        """Simulate a complete duty cycle.

        A duty cycle consists of:
        1. Sensing phase: IMU sampling for window_duration_s
        2. Processing phase: Feature extraction + inference
        3. Transmission phase: Send result/data via LoRa
        4. Sleep phase: Remaining time until next TX

        Args:
            sensor_config: Sensor configuration.
            tx_config: Transmission configuration.
            model_name: ML model name.
            window_duration_s: Classification window duration.

        Returns:
            DutyCycleResult with energy breakdown.
        """
        # Phase 1: Sensing
        imu_energy, mcu_sampling_energy = self.simulate_sensing_phase(
            sensor_config, window_duration_s
        )

        # Phase 2: Processing
        mcu_inference_energy = self.simulate_processing_phase(
            model_name, include_feature_extraction=True
        )

        # Phase 3: Transmission
        radio_tx_energy = self.simulate_transmission_phase(tx_config)

        # Calculate time spent in active phases
        inference_time_s = (
            self.hw.mcu.get_inference_time_ms(model_name) +
            self.hw.mcu.get_inference_time_ms("feature_extraction")
        ) / 1000.0

        toa_s = self.hw.radio.calculate_time_on_air_ms(
            tx_config.payload_bytes,
            tx_config.spreading_factor,
            tx_config.bandwidth,
        ) / 1000.0

        # Total active time
        active_time_s = window_duration_s + inference_time_s + toa_s

        # Sleep time (remaining time in cycle)
        sleep_time_s = max(0, tx_config.tx_interval_s - active_time_s)

        # Phase 4: Idle/Sleep
        mcu_idle_energy, radio_sleep_energy = self.simulate_idle_phase(sleep_time_s)

        # IMU sleep during non-sensing time
        imu_sleep_time_s = tx_config.tx_interval_s - window_duration_s
        imu_sleep_power_uw = self.hw.imu.get_power_mw(SensorMode.SLEEP) * 1000
        system_sleep_energy = imu_sleep_power_uw * imu_sleep_time_s

        return DutyCycleResult(
            duration_s=tx_config.tx_interval_s,
            sensor_config=sensor_config,
            tx_config=tx_config,
            imu_energy_uj=imu_energy + system_sleep_energy,
            mcu_sampling_energy_uj=mcu_sampling_energy,
            mcu_inference_energy_uj=mcu_inference_energy,
            mcu_idle_energy_uj=mcu_idle_energy,
            radio_tx_energy_uj=radio_tx_energy,
            radio_sleep_energy_uj=radio_sleep_energy,
            system_sleep_energy_uj=0,  # Already included in idle
        )

    def simulate_continuous_sensing(
        self,
        sensor_config: SensorConfiguration,
        tx_config: TransmissionConfiguration,
        duration_hours: float = 24.0,
        model_name: str = "random_forest_20",
        window_duration_s: float = 1.5,
    ) -> dict[str, Any]:
        """Simulate continuous sensing over a period.

        Args:
            sensor_config: Sensor configuration.
            tx_config: Transmission configuration.
            duration_hours: Total simulation duration in hours.
            model_name: ML model name.
            window_duration_s: Classification window duration.

        Returns:
            Dictionary with simulation results.
        """
        duration_s = duration_hours * 3600
        n_cycles = int(duration_s / tx_config.tx_interval_s)

        # Simulate one cycle and multiply
        cycle_result = self.simulate_duty_cycle(
            sensor_config, tx_config, model_name, window_duration_s
        )

        total_energy_mj = cycle_result.total_energy_mj * n_cycles
        avg_power_mw = cycle_result.avg_power_mw

        # Battery life estimation
        battery_life_days = self.hw.estimate_battery_life_days(avg_power_mw)

        return {
            "duration_hours": duration_hours,
            "n_cycles": n_cycles,
            "sensor_config": sensor_config.name,
            "tx_config": tx_config.name,
            "model_name": model_name,
            "total_energy_mj": total_energy_mj,
            "avg_power_mw": avg_power_mw,
            "avg_current_ua": avg_power_mw * 1000 / self.hw.battery_voltage,
            "battery_life_days": battery_life_days,
            "energy_breakdown": cycle_result.get_breakdown_dict(),
            "energy_percentages": cycle_result.get_breakdown_percentages(),
        }

    def compare_configurations(
        self,
        sensor_configs: list[SensorConfiguration],
        tx_configs: list[TransmissionConfiguration],
        model_name: str = "random_forest_20",
    ) -> list[dict[str, Any]]:
        """Compare multiple configurations.

        Args:
            sensor_configs: List of sensor configurations to compare.
            tx_configs: List of transmission configurations to compare.
            model_name: ML model name.

        Returns:
            List of comparison results.
        """
        results = []

        for sensor_cfg in sensor_configs:
            for tx_cfg in tx_configs:
                result = self.simulate_duty_cycle(sensor_cfg, tx_cfg, model_name)
                results.append({
                    "sensor_config": sensor_cfg.name,
                    "tx_config": tx_cfg.name,
                    "odr_hz": sensor_cfg.effective_odr_hz,
                    "payload_bytes": tx_cfg.payload_bytes,
                    "sf": tx_cfg.spreading_factor.name,
                    "total_energy_uj": result.total_energy_uj,
                    "avg_power_uw": result.avg_power_uw,
                    "battery_life_days": self.hw.estimate_battery_life_days(
                        result.avg_power_mw
                    ),
                    "breakdown": result.get_breakdown_percentages(),
                })

        return results


def run_power_sweep_experiment(
    odr_range: list[float] | None = None,
    payload_range: list[int] | None = None,
    sf_range: list[LoRaSpreadingFactor] | None = None,
) -> dict[str, Any]:
    """Run a parameter sweep experiment.

    Args:
        odr_range: List of ODRs to test.
        payload_range: List of payload sizes to test.
        sf_range: List of spreading factors to test.

    Returns:
        Dictionary with sweep results.
    """
    if odr_range is None:
        odr_range = [5.0, 10.0, 15.0, 25.0, 50.0, 100.0]
    if payload_range is None:
        payload_range = [10, 20, 50, 100]
    if sf_range is None:
        sf_range = [LoRaSpreadingFactor.SF7, LoRaSpreadingFactor.SF10, LoRaSpreadingFactor.SF12]

    simulator = PowerDecompositionSimulator()
    results = {
        "odr_sweep": [],
        "payload_sweep": [],
        "sf_sweep": [],
    }

    # ODR sweep (fixed payload and SF)
    tx_cfg = TX_CONFIG_CLASSIFICATION
    for odr in odr_range:
        sensor_cfg = SensorConfiguration(
            name=f"ACC_{odr}Hz",
            mode=SensorMode.ACC_ONLY_LP,
            acc_enabled=True,
            acc_odr_hz=odr,
        )
        result = simulator.simulate_duty_cycle(sensor_cfg, tx_cfg)
        results["odr_sweep"].append({
            "odr_hz": odr,
            "energy_uj": result.total_energy_uj,
            "power_uw": result.avg_power_uw,
            "imu_pct": result.get_breakdown_percentages().get("imu_pct", 0),
        })

    # Payload sweep (fixed ODR and SF)
    sensor_cfg = SENSOR_CONFIG_ACC_ONLY_50HZ
    for payload in payload_range:
        tx_cfg = TransmissionConfiguration(
            name=f"PAYLOAD_{payload}B",
            payload_type="variable",
            payload_bytes=payload,
            tx_power_dbm=14,
            spreading_factor=LoRaSpreadingFactor.SF7,
            tx_interval_s=90.0,
        )
        result = simulator.simulate_duty_cycle(sensor_cfg, tx_cfg)
        toa_ms = simulator.hw.radio.calculate_time_on_air_ms(payload, LoRaSpreadingFactor.SF7)
        results["payload_sweep"].append({
            "payload_bytes": payload,
            "toa_ms": toa_ms,
            "energy_uj": result.total_energy_uj,
            "power_uw": result.avg_power_uw,
            "radio_pct": result.get_breakdown_percentages().get("radio_tx_pct", 0),
        })

    # SF sweep (fixed ODR and payload)
    sensor_cfg = SENSOR_CONFIG_ACC_ONLY_50HZ
    for sf in sf_range:
        tx_cfg = TransmissionConfiguration(
            name=f"SF{sf.value}",
            payload_type="classification",
            payload_bytes=20,
            tx_power_dbm=14,
            spreading_factor=sf,
            tx_interval_s=90.0,
        )
        result = simulator.simulate_duty_cycle(sensor_cfg, tx_cfg)
        toa_ms = simulator.hw.radio.calculate_time_on_air_ms(20, sf)
        results["sf_sweep"].append({
            "sf": sf.value,
            "toa_ms": toa_ms,
            "energy_uj": result.total_energy_uj,
            "power_uw": result.avg_power_uw,
            "radio_pct": result.get_breakdown_percentages().get("radio_tx_pct", 0),
        })

    return results
