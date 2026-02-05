"""Real hardware specifications based on component datasheets.

This module provides accurate hardware specifications for:
- ICM-20948 9-axis IMU (TDK InvenSense) - DS-000189 Rev 1.3
- nRF52840 MCU (Nordic Semiconductor) - Product Specification v1.1
- SX1262 LoRa Transceiver (Semtech) - DS.SX1261-2.W.APP Rev 2.1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any
import math


class SensorMode(Enum):
    """ICM-20948 sensor operating modes."""
    SLEEP = auto()           # Full chip sleep
    ACC_ONLY_LP = auto()     # Accelerometer only, low-power mode
    ACC_ONLY_LN = auto()     # Accelerometer only, low-noise mode
    ACC_GYRO_LP = auto()     # ACC + Gyroscope, low-power mode
    ACC_GYRO_LN = auto()     # ACC + Gyroscope, low-noise mode
    NINE_AXIS_LP = auto()    # 9-axis (ACC + GYRO + MAG), low-power
    NINE_AXIS_LN = auto()    # 9-axis, low-noise mode


class LoRaSpreadingFactor(Enum):
    """LoRa spreading factors."""
    SF7 = 7
    SF8 = 8
    SF9 = 9
    SF10 = 10
    SF11 = 11
    SF12 = 12


class LoRaBandwidth(Enum):
    """LoRa bandwidth options in kHz."""
    BW_125 = 125
    BW_250 = 250
    BW_500 = 500


@dataclass
class ICM20948Spec:
    """ICM-20948 9-axis IMU specifications from datasheet.

    Reference: DS-000189 Rev 1.3 Table 4: D.C. Electrical Characteristics

    Attributes:
        vdd: Supply voltage (V). Default 1.8V.
        mode: Current operating mode.
    """
    vdd: float = 1.8  # Supply voltage

    # Current draw for different modes (from Table 4)
    # All values in microamps unless otherwise noted

    # Sleep mode
    SLEEP_CURRENT_UA: float = 8.0

    # Low-Power mode currents at different configurations
    # Accelerometer only, LP mode, 102.3Hz, 1x averaging
    ACC_LP_CURRENT_UA: float = 68.9

    # Gyroscope only, LP mode
    GYRO_LP_CURRENT_UA: float = 1230.0  # 1.23 mA

    # Magnetometer (AK09916) current at different ODRs
    MAG_8HZ_CURRENT_UA: float = 90.0
    MAG_100HZ_CURRENT_UA: float = 280.0

    # Low-Noise mode currents (higher power, better performance)
    ACC_LN_CURRENT_UA: float = 230.0
    GYRO_LN_CURRENT_UA: float = 3200.0  # 3.2 mA

    # Full 9-axis mode currents
    NINE_AXIS_DMP_DISABLED_CURRENT_UA: float = 3110.0  # 3.11 mA
    NINE_AXIS_DMP_ENABLED_CURRENT_UA: float = 5300.0   # 5.3 mA (with DMP)

    # Startup times (from datasheet)
    GYRO_STARTUP_MS: float = 35.0
    ACC_STARTUP_FROM_SLEEP_MS: float = 20.0
    ACC_COLD_START_MS: float = 30.0

    # Sampling rate limits
    MAX_ACC_ODR_HZ: float = 1125.0
    MAX_GYRO_LN_ODR_HZ: float = 1125.0
    MAX_GYRO_LP_ODR_HZ: float = 562.5

    def get_current_ua(self, mode: SensorMode, odr_hz: float = 50.0) -> float:
        """Get current draw for a given mode and ODR.

        Args:
            mode: Sensor operating mode.
            odr_hz: Output data rate in Hz.

        Returns:
            Current in microamps.
        """
        if mode == SensorMode.SLEEP:
            return self.SLEEP_CURRENT_UA
        elif mode == SensorMode.ACC_ONLY_LP:
            # Scale with ODR (approximate linear scaling)
            base_current = self.ACC_LP_CURRENT_UA  # at 102.3 Hz
            return base_current * (odr_hz / 102.3) * 0.7 + base_current * 0.3
        elif mode == SensorMode.ACC_ONLY_LN:
            return self.ACC_LN_CURRENT_UA
        elif mode == SensorMode.ACC_GYRO_LP:
            return self.ACC_LP_CURRENT_UA + self.GYRO_LP_CURRENT_UA
        elif mode == SensorMode.ACC_GYRO_LN:
            return self.ACC_LN_CURRENT_UA + self.GYRO_LN_CURRENT_UA
        elif mode == SensorMode.NINE_AXIS_LP:
            return self.ACC_LP_CURRENT_UA + self.GYRO_LP_CURRENT_UA + self.MAG_8HZ_CURRENT_UA
        elif mode == SensorMode.NINE_AXIS_LN:
            return self.NINE_AXIS_DMP_DISABLED_CURRENT_UA
        else:
            return self.ACC_LP_CURRENT_UA

    def get_power_mw(self, mode: SensorMode, odr_hz: float = 50.0) -> float:
        """Get power consumption in milliwatts.

        Args:
            mode: Sensor operating mode.
            odr_hz: Output data rate in Hz.

        Returns:
            Power in milliwatts.
        """
        current_ua = self.get_current_ua(mode, odr_hz)
        return current_ua * self.vdd / 1000.0

    def get_odr_divider(self, target_odr_hz: float) -> int:
        """Calculate SMPLRT_DIV register value for target ODR.

        Formula: ODR = 1125 Hz / (1 + SMPLRT_DIV)

        Args:
            target_odr_hz: Target output data rate.

        Returns:
            SMPLRT_DIV register value (0-4095 for ACC, 0-255 for GYRO).
        """
        if target_odr_hz <= 0:
            return 255
        divider = int(1125.0 / target_odr_hz) - 1
        return max(0, min(divider, 4095))


@dataclass
class NRF52840Spec:
    """nRF52840 MCU specifications from datasheet.

    Reference: nRF52840 Product Specification v1.1

    Attributes:
        vdd: Supply voltage (V). Default 3.0V (typical).
    """
    vdd: float = 3.0  # Supply voltage (1.7V - 5.5V supported)

    # Current draw for different modes (at 3.0V)
    SYSTEM_OFF_UA: float = 0.4        # No RAM retention
    SYSTEM_OFF_RAM_UA: float = 1.5    # Full RAM retention
    SYSTEM_ON_IDLE_UA: float = 2.35   # RAM retained, no peripherals
    CPU_ACTIVE_MA: float = 3.0        # 64MHz, no radio
    CPU_ACTIVE_FLASH_MA: float = 3.7  # 64MHz with Flash access

    # Peripheral currents (typical)
    SPI_MASTER_UA: float = 50.0   # 8MHz
    I2C_MASTER_UA: float = 50.0   # 400kHz
    UART_UA: float = 40.0         # 115200 baud
    ADC_UA: float = 400.0         # During sampling
    GPIO_STATIC_UA: float = 1.0   # Static GPIO

    # Wake-up times
    SYSTEM_OFF_TO_ON_MS: float = 0.5
    IDLE_TO_ACTIVE_US: float = 1.0

    # Clock frequency
    CLOCK_FREQ_MHZ: float = 64.0

    # Inference timing estimates (ms) for different models on Cortex-M4F
    INFERENCE_TIME_MS: dict = field(default_factory=lambda: {
        "decision_tree": 0.3,
        "random_forest_10": 2.0,
        "random_forest_20": 4.0,
        "random_forest_50": 10.0,
        "svm_linear": 1.5,
        "svm_rbf": 5.0,
        "cnn_1d_simple": 30.0,
        "cnn_1d_optimized": 15.0,
        "tcn_small": 50.0,
        "feature_extraction": 8.0,
    })

    def get_current_ua(self, active: bool = False, with_peripherals: bool = False) -> float:
        """Get current draw based on state.

        Args:
            active: Whether CPU is actively computing.
            with_peripherals: Whether peripherals are active.

        Returns:
            Current in microamps.
        """
        if not active:
            return self.SYSTEM_ON_IDLE_UA

        base_current = self.CPU_ACTIVE_MA * 1000  # Convert to uA
        if with_peripherals:
            base_current += self.SPI_MASTER_UA + self.I2C_MASTER_UA
        return base_current

    def get_power_mw(self, active: bool = False, with_peripherals: bool = False) -> float:
        """Get power consumption in milliwatts."""
        current_ua = self.get_current_ua(active, with_peripherals)
        return current_ua * self.vdd / 1000.0

    def get_inference_time_ms(self, model_name: str) -> float:
        """Get inference time for a model."""
        return self.INFERENCE_TIME_MS.get(model_name, 5.0)

    def get_inference_energy_uj(self, model_name: str) -> float:
        """Get energy for one inference in microjoules."""
        time_ms = self.get_inference_time_ms(model_name)
        power_mw = self.get_power_mw(active=True, with_peripherals=False)
        return power_mw * time_ms  # mW * ms = uJ


@dataclass
class SX1262Spec:
    """SX1262 LoRa transceiver specifications from datasheet.

    Reference: DS.SX1261-2.W.APP Rev 2.1

    Attributes:
        vdd: Supply voltage (V). Default 3.3V.
    """
    vdd: float = 3.3  # Supply voltage (1.8V - 3.7V)

    # Current draw for different modes (DC-DC mode, typical values)
    SLEEP_COLD_NA: float = 160.0    # nA, cold start (RC off)
    SLEEP_WARM_NA: float = 600.0    # nA, warm start (config retained)
    STANDBY_RC_UA: float = 1.2      # RC13M enabled
    STANDBY_XOSC_MA: float = 1.6    # Crystal oscillator enabled
    RX_BOOSTED_MA: float = 4.6      # RX with boosted gain
    RX_DCDC_MA: float = 4.2         # RX with DC-DC

    # TX current at different power levels (DC-DC mode)
    TX_POWER_CURRENT_MA: dict = field(default_factory=lambda: {
        14: 22.0,   # +14 dBm
        17: 45.0,   # +17 dBm
        20: 84.0,   # +20 dBm
        22: 118.0,  # +22 dBm (PA from VBAT)
    })

    # Time-on-Air calculation parameters
    PREAMBLE_SYMBOLS: int = 8

    def get_symbol_time_ms(
        self,
        sf: LoRaSpreadingFactor,
        bw: LoRaBandwidth = LoRaBandwidth.BW_125
    ) -> float:
        """Calculate symbol time in milliseconds.

        Formula: T_symbol = 2^SF / BW

        Args:
            sf: Spreading factor.
            bw: Bandwidth.

        Returns:
            Symbol time in milliseconds.
        """
        return (2 ** sf.value) / (bw.value * 1000) * 1000

    def calculate_time_on_air_ms(
        self,
        payload_bytes: int,
        sf: LoRaSpreadingFactor = LoRaSpreadingFactor.SF7,
        bw: LoRaBandwidth = LoRaBandwidth.BW_125,
        coding_rate: int = 1,  # CR 4/5 = 1, 4/6 = 2, etc.
        explicit_header: bool = True,
        crc_enabled: bool = True,
        low_data_rate_optimize: bool = False,
    ) -> float:
        """Calculate time-on-air for a LoRa packet.

        Based on Semtech SX1262 datasheet formulas.

        Args:
            payload_bytes: Payload length in bytes.
            sf: Spreading factor (SF7-SF12).
            bw: Bandwidth.
            coding_rate: 1 for 4/5, 2 for 4/6, 3 for 4/7, 4 for 4/8.
            explicit_header: True for explicit header mode.
            crc_enabled: True if CRC is enabled.
            low_data_rate_optimize: True for SF11/SF12 at 125kHz.

        Returns:
            Time-on-air in milliseconds.
        """
        t_symbol = self.get_symbol_time_ms(sf, bw)

        # Preamble time
        t_preamble = (self.PREAMBLE_SYMBOLS + 4.25) * t_symbol

        # Payload calculation
        h = 0 if explicit_header else 1
        crc = 16 if crc_enabled else 0
        de = 1 if low_data_rate_optimize else 0

        # Number of payload symbols
        numerator = 8 * payload_bytes - 4 * sf.value + 28 + crc - 20 * h
        denominator = 4 * (sf.value - 2 * de)

        if denominator <= 0:
            denominator = 4 * sf.value

        n_payload = 8 + max(math.ceil(numerator / denominator) * (coding_rate + 4), 0)

        t_payload = n_payload * t_symbol

        return t_preamble + t_payload

    def get_tx_current_ma(self, power_dbm: int = 14) -> float:
        """Get TX current for a given power level.

        Args:
            power_dbm: TX power in dBm (14, 17, 20, 22).

        Returns:
            Current in milliamps.
        """
        # Find closest power level
        powers = sorted(self.TX_POWER_CURRENT_MA.keys())
        for p in powers:
            if power_dbm <= p:
                return self.TX_POWER_CURRENT_MA[p]
        return self.TX_POWER_CURRENT_MA[powers[-1]]

    def get_tx_power_mw(self, power_dbm: int = 14) -> float:
        """Get TX power consumption in milliwatts."""
        return self.get_tx_current_ma(power_dbm) * self.vdd

    def get_tx_energy_mj(
        self,
        payload_bytes: int,
        power_dbm: int = 14,
        sf: LoRaSpreadingFactor = LoRaSpreadingFactor.SF7,
        bw: LoRaBandwidth = LoRaBandwidth.BW_125,
    ) -> float:
        """Calculate transmission energy in millijoules.

        Args:
            payload_bytes: Payload size in bytes.
            power_dbm: TX power level.
            sf: Spreading factor.
            bw: Bandwidth.

        Returns:
            Energy in millijoules.
        """
        toa_ms = self.calculate_time_on_air_ms(payload_bytes, sf, bw)
        power_mw = self.get_tx_power_mw(power_dbm)
        return power_mw * toa_ms / 1000.0  # Convert ms to s for proper unit

    def get_rx_power_mw(self, boosted: bool = False) -> float:
        """Get RX power consumption in milliwatts."""
        current = self.RX_BOOSTED_MA if boosted else self.RX_DCDC_MA
        return current * self.vdd

    def get_sleep_power_uw(self, warm_start: bool = True) -> float:
        """Get sleep power in microwatts."""
        current_na = self.SLEEP_WARM_NA if warm_start else self.SLEEP_COLD_NA
        return current_na * self.vdd / 1000.0


@dataclass
class RealHardwareSpec:
    """Complete real hardware specifications for ICM-20948 + nRF52840 + SX1262.

    This class combines all component specifications for system-level energy modeling.
    """

    imu: ICM20948Spec = field(default_factory=ICM20948Spec)
    mcu: NRF52840Spec = field(default_factory=NRF52840Spec)
    radio: SX1262Spec = field(default_factory=SX1262Spec)

    # Battery specifications
    battery_capacity_mah: float = 2000.0
    battery_voltage: float = 3.7

    # Default operating parameters
    default_sensor_mode: SensorMode = SensorMode.ACC_ONLY_LP
    default_odr_hz: float = 50.0
    default_tx_power_dbm: int = 14
    default_sf: LoRaSpreadingFactor = LoRaSpreadingFactor.SF7

    @property
    def battery_capacity_mwh(self) -> float:
        """Battery capacity in milliwatt-hours."""
        return self.battery_capacity_mah * self.battery_voltage

    @property
    def battery_capacity_j(self) -> float:
        """Battery capacity in joules."""
        return self.battery_capacity_mwh * 3.6  # 1 mWh = 3.6 J

    def get_system_sleep_power_uw(self) -> float:
        """Get total system sleep power in microwatts.

        Assumes:
        - IMU in sleep mode
        - MCU in System ON idle (RAM retained)
        - Radio in warm sleep
        """
        imu_power = self.imu.get_power_mw(SensorMode.SLEEP) * 1000  # to uW
        mcu_power = self.mcu.SYSTEM_ON_IDLE_UA * self.mcu.vdd  # uA * V = uW
        radio_power = self.radio.get_sleep_power_uw(warm_start=True)

        return imu_power + mcu_power + radio_power

    def estimate_battery_life_days(
        self,
        avg_power_mw: float,
    ) -> float:
        """Estimate battery life in days.

        Args:
            avg_power_mw: Average system power in milliwatts.

        Returns:
            Estimated battery life in days.
        """
        if avg_power_mw <= 0:
            return float("inf")

        hours = self.battery_capacity_mwh / avg_power_mw
        return hours / 24.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "imu": {
                "model": "ICM-20948",
                "vdd": self.imu.vdd,
                "sleep_current_ua": self.imu.SLEEP_CURRENT_UA,
                "acc_lp_current_ua": self.imu.ACC_LP_CURRENT_UA,
                "nine_axis_current_ua": self.imu.NINE_AXIS_DMP_DISABLED_CURRENT_UA,
            },
            "mcu": {
                "model": "nRF52840",
                "vdd": self.mcu.vdd,
                "active_current_ma": self.mcu.CPU_ACTIVE_MA,
                "idle_current_ua": self.mcu.SYSTEM_ON_IDLE_UA,
            },
            "radio": {
                "model": "SX1262",
                "vdd": self.radio.vdd,
                "tx_14dbm_current_ma": self.radio.TX_POWER_CURRENT_MA.get(14, 22.0),
                "rx_current_ma": self.radio.RX_DCDC_MA,
            },
            "battery": {
                "capacity_mah": self.battery_capacity_mah,
                "voltage": self.battery_voltage,
            },
        }


# Pre-configured hardware profiles
LIVEEDGE_HARDWARE = RealHardwareSpec()

LIVEEDGE_LOW_POWER = RealHardwareSpec(
    battery_capacity_mah=500.0,
    default_sensor_mode=SensorMode.ACC_ONLY_LP,
    default_odr_hz=25.0,
    default_tx_power_dbm=14,
    default_sf=LoRaSpreadingFactor.SF7,
)

LIVEEDGE_HIGH_ACCURACY = RealHardwareSpec(
    battery_capacity_mah=2000.0,
    default_sensor_mode=SensorMode.ACC_GYRO_LP,
    default_odr_hz=50.0,
    default_tx_power_dbm=17,
    default_sf=LoRaSpreadingFactor.SF10,
)
