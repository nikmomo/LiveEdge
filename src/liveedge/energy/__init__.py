"""Energy modeling module for LiveEdge.

This module provides energy consumption modeling for edge devices.
"""

from liveedge.energy.hardware import (
    PROFILE_HIGH_PERFORMANCE,
    PROFILE_LOW_POWER,
    PROFILE_STANDARD,
    BatterySpec,
    HardwareSpec,
    MCUSpec,
    RadioSpec,
    SensorSpec,
)
from liveedge.energy.models import (
    EnergyBreakdown,
    EnergyModel,
    compute_energy_reduction,
    estimate_sampling_rate_from_energy_budget,
)
from liveedge.energy.real_hardware import (
    ICM20948Spec,
    NRF52840Spec,
    SX1262Spec,
    RealHardwareSpec,
    SensorMode,
    LoRaSpreadingFactor,
    LoRaBandwidth,
    LIVEEDGE_HARDWARE,
    LIVEEDGE_LOW_POWER,
    LIVEEDGE_HIGH_ACCURACY,
)
from liveedge.energy.power_decomposition import (
    PowerDecompositionSimulator,
    SensorConfiguration,
    TransmissionConfiguration,
    DutyCycleResult,
    WorkPhase,
    SENSOR_CONFIG_ACC_ONLY_5HZ,
    SENSOR_CONFIG_ACC_ONLY_15HZ,
    SENSOR_CONFIG_ACC_ONLY_25HZ,
    SENSOR_CONFIG_ACC_ONLY_50HZ,
    SENSOR_CONFIG_ACC_ONLY_100HZ,
    SENSOR_CONFIG_ACC_GYRO_50HZ,
    SENSOR_CONFIG_9AXIS_50HZ,
    TX_CONFIG_CLASSIFICATION,
    TX_CONFIG_FEATURES,
    TX_CONFIG_RAW_DATA,
    run_power_sweep_experiment,
)

__all__ = [
    # Hardware specs (legacy)
    "SensorSpec",
    "MCUSpec",
    "RadioSpec",
    "BatterySpec",
    "HardwareSpec",
    # Profiles (legacy)
    "PROFILE_LOW_POWER",
    "PROFILE_STANDARD",
    "PROFILE_HIGH_PERFORMANCE",
    # Energy models
    "EnergyBreakdown",
    "EnergyModel",
    "compute_energy_reduction",
    "estimate_sampling_rate_from_energy_budget",
    # Real hardware specs
    "ICM20948Spec",
    "NRF52840Spec",
    "SX1262Spec",
    "RealHardwareSpec",
    "SensorMode",
    "LoRaSpreadingFactor",
    "LoRaBandwidth",
    "LIVEEDGE_HARDWARE",
    "LIVEEDGE_LOW_POWER",
    "LIVEEDGE_HIGH_ACCURACY",
    # Power decomposition
    "PowerDecompositionSimulator",
    "SensorConfiguration",
    "TransmissionConfiguration",
    "DutyCycleResult",
    "WorkPhase",
    "SENSOR_CONFIG_ACC_ONLY_5HZ",
    "SENSOR_CONFIG_ACC_ONLY_15HZ",
    "SENSOR_CONFIG_ACC_ONLY_25HZ",
    "SENSOR_CONFIG_ACC_ONLY_50HZ",
    "SENSOR_CONFIG_ACC_ONLY_100HZ",
    "SENSOR_CONFIG_ACC_GYRO_50HZ",
    "SENSOR_CONFIG_9AXIS_50HZ",
    "TX_CONFIG_CLASSIFICATION",
    "TX_CONFIG_FEATURES",
    "TX_CONFIG_RAW_DATA",
    "run_power_sweep_experiment",
]
