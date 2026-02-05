"""Hardware specifications for energy modeling.

This module provides dataclasses for specifying hardware characteristics
used in energy consumption calculations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SensorSpec:
    """Sensor specifications.

    Attributes:
        active_power_mw: Power consumption when actively sampling (mW).
        sleep_power_uw: Power consumption in sleep mode (uW).
        wakeup_time_us: Time to wake up from sleep (us).
        sample_time_us: Time to take one sample (us).
    """

    active_power_mw: float = 1.0
    sleep_power_uw: float = 5.0
    wakeup_time_us: float = 100.0
    sample_time_us: float = 200.0


@dataclass
class MCUSpec:
    """MCU (Microcontroller Unit) specifications.

    Attributes:
        active_power_mw: Power consumption when active (mW).
        idle_power_mw: Power consumption in idle mode (mW).
        sleep_power_uw: Power consumption in deep sleep (uW).
        wake_time_ms: Time to wake from sleep (ms).
        clock_freq_mhz: CPU clock frequency (MHz).
    """

    active_power_mw: float = 15.0
    idle_power_mw: float = 2.0
    sleep_power_uw: float = 2.0
    wake_time_ms: float = 2.0
    clock_freq_mhz: float = 64.0


@dataclass
class RadioSpec:
    """Radio (BLE/LoRa/etc.) specifications.

    Attributes:
        tx_power_mw: Power consumption during transmission (mW).
        rx_power_mw: Power consumption during reception (mW).
        idle_power_mw: Power consumption in idle mode (mW).
        data_rate_kbps: Data rate (kbps).
        protocol: Radio protocol name.
    """

    tx_power_mw: float = 18.0
    rx_power_mw: float = 17.0
    idle_power_mw: float = 0.01
    data_rate_kbps: float = 2000.0
    protocol: str = "BLE5"


@dataclass
class BatterySpec:
    """Battery specifications.

    Attributes:
        capacity_mah: Battery capacity (mAh).
        voltage: Nominal voltage (V).
        min_voltage: Minimum operating voltage (V).
    """

    capacity_mah: float = 500.0
    voltage: float = 3.7
    min_voltage: float = 3.0

    @property
    def capacity_mwh(self) -> float:
        """Battery capacity in mWh."""
        return self.capacity_mah * self.voltage


@dataclass
class HardwareSpec:
    """Complete hardware specifications for energy modeling.

    Aggregates specifications for all hardware components.
    """

    sensor: SensorSpec = field(default_factory=SensorSpec)
    mcu: MCUSpec = field(default_factory=MCUSpec)
    radio: RadioSpec = field(default_factory=RadioSpec)
    battery: BatterySpec = field(default_factory=BatterySpec)

    # Inference timing (ms per inference for different models)
    inference_time_ms: dict[str, float] = field(
        default_factory=lambda: {
            "decision_tree": 0.3,
            "random_forest": 3.0,
            "svm": 2.0,
            "cnn_1d": 15.0,
            "tcn": 30.0,
        }
    )

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "HardwareSpec":
        """Create HardwareSpec from configuration dictionary.

        Args:
            config: Dictionary with hardware parameters.

        Returns:
            HardwareSpec instance.
        """
        return cls(
            sensor=SensorSpec(
                active_power_mw=config.get("sensor_active_power_mw", 1.0),
                sleep_power_uw=config.get("sensor_sleep_power_uw", 5.0),
            ),
            mcu=MCUSpec(
                active_power_mw=config.get("mcu_active_power_mw", 15.0),
                idle_power_mw=config.get("mcu_idle_power_mw", 2.0),
                sleep_power_uw=config.get("mcu_sleep_power_uw", 2.0),
                wake_time_ms=config.get("wake_time_ms", 2.0),
            ),
            radio=RadioSpec(
                tx_power_mw=config.get("radio_tx_power_mw", 18.0),
                rx_power_mw=config.get("radio_rx_power_mw", 17.0),
                idle_power_mw=config.get("radio_idle_power_mw", 0.01),
                data_rate_kbps=config.get("data_rate_kbps", 2000.0),
            ),
            battery=BatterySpec(
                capacity_mah=config.get("battery_capacity_mah", 500.0),
                voltage=config.get("battery_voltage", 3.7),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation of hardware specs.
        """
        return {
            "sensor_active_power_mw": self.sensor.active_power_mw,
            "sensor_sleep_power_uw": self.sensor.sleep_power_uw,
            "mcu_active_power_mw": self.mcu.active_power_mw,
            "mcu_idle_power_mw": self.mcu.idle_power_mw,
            "mcu_sleep_power_uw": self.mcu.sleep_power_uw,
            "radio_tx_power_mw": self.radio.tx_power_mw,
            "radio_rx_power_mw": self.radio.rx_power_mw,
            "battery_capacity_mah": self.battery.capacity_mah,
            "battery_voltage": self.battery.voltage,
        }


# Pre-defined hardware profiles
PROFILE_LOW_POWER = HardwareSpec(
    sensor=SensorSpec(active_power_mw=0.5, sleep_power_uw=1.0),
    mcu=MCUSpec(active_power_mw=8.0, idle_power_mw=1.0, sleep_power_uw=0.5),
    radio=RadioSpec(tx_power_mw=10.0, data_rate_kbps=1000.0),
    battery=BatterySpec(capacity_mah=250.0),
)

PROFILE_STANDARD = HardwareSpec()

PROFILE_HIGH_PERFORMANCE = HardwareSpec(
    sensor=SensorSpec(active_power_mw=2.0, sleep_power_uw=10.0),
    mcu=MCUSpec(active_power_mw=30.0, idle_power_mw=5.0, sleep_power_uw=5.0),
    radio=RadioSpec(tx_power_mw=25.0, data_rate_kbps=4000.0),
    battery=BatterySpec(capacity_mah=1000.0),
)
