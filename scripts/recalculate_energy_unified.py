#!/usr/bin/env python3
"""
Unified energy model verification script.

Computes all energy values from first principles using datasheet parameters
and Renode cycle-accurate inference time measurement. This script verifies
consistency with verified_simulation.py and the paper's energy claims.

Datasheet sources:
  - BMI160 (BST-BMI160-DS000): Table 8, low-power accel mode, gyro suspended
  - nRF52832 (PS v1.8): CPU current (DCDC, 64MHz), System ON idle (DCDC)
  - CR2032: Maxell spec, 220 mAh nominal
"""

# =============================================================================
# Hardware Parameters (all from datasheets)
# =============================================================================

# --- Battery: CR2032 (Maxell datasheet) ---
C_BAT = 220       # mAh (typical capacity)
V = 3.0            # V (nominal voltage)
ETA = 0.8          # Usable capacity fraction (standard for CR2032)
BATTERY_ENERGY = C_BAT * V * 3600 * ETA  # mJ = 1,900,800 mJ

# --- IMU: BMI160 (BST-BMI160-DS000, Table 8) ---
# Low-power accelerometer mode, gyro suspended, 1 averaging cycle
# Values in uA at VDD = 3.0V
BMI160_CURRENT = {
    6.25: 7,       # ~7 uA (datasheet: 6 uA typ, +1 uA for I2C interface)
    10: 10,        # interpolated (not a native BMI160 rate)
    12.5: 10,      # ~10 uA
    25: 17,        # ~17 uA
    50: 27,        # ~27 uA
}
# Note: Normal mode (accel only, gyro suspended) = 180 uA (Table 7)
# We use low-power mode which duty-cycles the sensor.

# --- MCU: nRF52832 (Product Specification v1.8) ---
# DCDC mode at VDD = 3.0V, 64 MHz
I_MCU_ACTIVE = 2100   # uA (2.1 mA) - CPU running from flash, DCDC, 64MHz
                       # = 33 uA/MHz x 64 MHz (Product Brief v1.2, DCDC mode)
I_MCU_IDLE = 1.9       # uA - System ON idle + full RAM retention + RTC (DCDC)
                       # Breakdown: base 0.7 uA + RAM ~0.5 uA + RTC ~0.7 uA

# --- BLE Radio: nRF52832 (Product Specification v1.8, Table 50) ---
I_TX = 5300    # uA (5.3 mA) - Radio TX at 0 dBm, 1 Mbps
I_RX = 5400    # uA (5.4 mA) - Radio RX at 1 Mbps
T_TX = 0.0015  # s - TX duration per BLE advertising packet
T_RX = 0.001   # s - RX duration per BLE advertising packet

# --- System overhead: nRF52832 ---
I_OTHER = 2.05  # uA - RTC oscillator + voltage regulator quiescent + leakage

# --- Timing ---
T_DAY = 86400     # s
T_WINDOW = 1.5    # s (window duration)
N_WINDOWS_DAY = T_DAY / T_WINDOW  # 57,600 windows/day
TX_INTERVAL = 60  # s (1 BLE packet per minute)

# --- Inference: Renode DWT cycle-accurate measurement ---
T_INFERENCE = 0.001976  # s (1.976 ms = 126,469 DWT cycles @ 64 MHz)
# Covers: 24-feature extraction + RF inference (30 trees, depth 8) + FSM update
# Measured via DWT_CYCCNT in Renode with dwt: Miscellaneous.DWT @ sysbus 0xE0001000


# =============================================================================
# Energy computation functions (matches verified_simulation.py exactly)
# =============================================================================

def compute_energy(odr_distribution, inference_rate, tx_interval=TX_INTERVAL):
    """Compute daily energy from first principles.

    Args:
        odr_distribution: dict {odr_hz: fraction} summing to 1.0
        inference_rate: fraction of windows requiring inference (0-1)
        tx_interval: BLE TX interval in seconds

    Returns:
        dict with e_imu, e_mcu, e_ble, e_other, e_total (mJ/day), battery_days
    """
    # E_IMU: weighted average IMU current x voltage x time
    i_imu_avg = sum(BMI160_CURRENT.get(odr, 10) * frac
                    for odr, frac in odr_distribution.items())
    e_imu = i_imu_avg * V * T_DAY / 1000  # uAxVxs / 1000 = mJ

    # E_MCU: active during inference + idle rest of time
    n_inferences = N_WINDOWS_DAY * inference_rate
    t_active = n_inferences * T_INFERENCE  # seconds of CPU active time
    t_idle = T_DAY - t_active
    e_mcu = (t_active * I_MCU_ACTIVE + t_idle * I_MCU_IDLE) * V / 1000  # mJ

    # E_BLE: periodic advertising
    n_tx = T_DAY / tx_interval
    e_ble = n_tx * (T_TX * I_TX + T_RX * I_RX) * V / 1000  # mJ

    # E_other: RTC, regulator, leakage
    e_other = I_OTHER * V * T_DAY / 1000  # mJ

    e_total = e_imu + e_mcu + e_ble + e_other
    battery_days = BATTERY_ENERGY / e_total

    return {
        'e_imu': round(e_imu, 1),
        'e_mcu': round(e_mcu, 1),
        'e_ble': round(e_ble, 1),
        'e_other': round(e_other, 1),
        'e_total': round(e_total, 1),
        'i_imu_avg': round(i_imu_avg, 2),
        'battery_days': round(battery_days, 1),
    }


# =============================================================================
# Main computation
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("UNIFIED ENERGY MODEL (First-Principles Computation)")
    print("=" * 70)

    # --- Verify overhead components ---
    n_tx = T_DAY / TX_INTERVAL
    e_ble = n_tx * (T_TX * I_TX + T_RX * I_RX) * V / 1000
    e_other = I_OTHER * V * T_DAY / 1000
    e_overhead = e_ble + e_other

    print(f"\nBattery capacity: {BATTERY_ENERGY:,.0f} mJ ({C_BAT} mAh x {V}V x {ETA} efficiency)")
    print(f"T_inference: {T_INFERENCE*1000:.3f} ms (Renode DWT: 126,469 cycles @ 64MHz)")
    print(f"Windows/day: {N_WINDOWS_DAY:,.0f}")
    print(f"\n--- Overhead Components ---")
    print(f"  E_BLE: {e_ble:.1f} mJ/day ({n_tx:.0f} pkts x ({T_TX*1000:.1f}ms TX @ {I_TX/1000:.1f}mA + {T_RX*1000:.1f}ms RX @ {I_RX/1000:.1f}mA))")
    print(f"  E_other: {e_other:.1f} mJ/day ({I_OTHER} uA continuous)")
    print(f"  E_overhead total: {e_overhead:.0f} mJ/day")

    # =========================================================================
    # Fixed-Rate Baselines (100% inference)
    # =========================================================================
    print("\n" + "=" * 70)
    print("FIXED-RATE BASELINES (100% inference)")
    print("=" * 70)

    print(f"\n{'ODR':<10} {'E_IMU':>8} {'E_MCU':>8} {'E_BLE':>8} {'E_other':>8} {'E_total':>8} {'Battery':>8} {'Savings':>8}")
    print("-" * 78)

    baselines = {}
    for odr in [50, 25, 12.5, 10, 6.25]:
        result = compute_energy({odr: 1.0}, inference_rate=1.0)
        baselines[odr] = result
        savings = (1 - result['e_total'] / baselines[50]['e_total']) * 100 if odr != 50 else 0
        print(f"{odr:<10} {result['e_imu']:>8.0f} {result['e_mcu']:>8.0f} "
              f"{result['e_ble']:>8.0f} {result['e_other']:>8.0f} "
              f"{result['e_total']:>8.0f} {result['battery_days']:>7.0f}d {savings:>+7.1f}%")

    # =========================================================================
    # Ablation Study (Table 6)
    # =========================================================================
    print("\n" + "=" * 70)
    print("ABLATION STUDY (Table 6)")
    print("=" * 70)

    # LiveEdge ODR distribution from compressed model FSM (verified_simulation.py)
    # Derived from verified_results.json: eff_odr=13.37Hz, E_IMU=2782mJ
    liveedge_odr_dist = {6.25: 0.4916, 12.5: 0.1929, 25.0: 0.3154}
    liveedge_inf_rate = 0.3808  # 38.1% from compressed model

    eff_odr = sum(odr * frac for odr, frac in liveedge_odr_dist.items())
    print(f"\nLiveEdge ODR distribution: {liveedge_odr_dist}")
    print(f"Effective ODR: {eff_odr:.2f} Hz")
    print(f"Inference rate: {liveedge_inf_rate*100:.1f}%")

    # A0: Baseline (50Hz, 100% inference)
    a0 = compute_energy({50: 1.0}, inference_rate=1.0)

    # A1: Adaptive ODR only (100% inference)
    a1 = compute_energy(liveedge_odr_dist, inference_rate=1.0)

    # A2: Fixed 50Hz + stability-based inference skip
    a2 = compute_energy({50: 1.0}, inference_rate=liveedge_inf_rate)

    # A3: LiveEdge (Adaptive ODR + inference skip)
    a3 = compute_energy(liveedge_odr_dist, inference_rate=liveedge_inf_rate)

    print(f"\n{'Config':<6} | {'E_IMU':>8} | {'E_MCU':>8} | {'E_total':>8} | {'Savings':>8} | {'Battery':>8}")
    print("-" * 60)
    print(f"A0     | {a0['e_imu']:>8.0f} | {a0['e_mcu']:>8.0f} | {a0['e_total']:>8.0f} |      --- | {a0['battery_days']:>7.0f}d")
    for label, cfg in [('A1', a1), ('A2', a2), ('A3', a3)]:
        savings = (1 - cfg['e_total'] / a0['e_total']) * 100
        print(f"{label:<6} | {cfg['e_imu']:>8.0f} | {cfg['e_mcu']:>8.0f} | {cfg['e_total']:>8.0f} | "
              f"{savings:>6.1f}% | {cfg['battery_days']:>7.0f}d")

    # Component analysis
    print(f"\n--- Component Contributions ---")
    total_saved = a0['e_total'] - a3['e_total']
    imu_saved = a0['e_imu'] - a3['e_imu']
    mcu_saved = a0['e_mcu'] - a3['e_mcu']
    print(f"  IMU reduction: {a0['e_imu']:.0f} -> {a3['e_imu']:.0f} mJ ({imu_saved:.0f} mJ, {imu_saved/total_saved*100:.1f}% of savings)")
    print(f"  MCU reduction: {a0['e_mcu']:.0f} -> {a3['e_mcu']:.0f} mJ ({mcu_saved:.0f} mJ, {mcu_saved/total_saved*100:.1f}% of savings)")
    print(f"  MCU fraction of total: {a0['e_mcu']/a0['e_total']*100:.1f}% (A0) -> {a3['e_mcu']/a3['e_total']*100:.1f}% (A3)")

    # =========================================================================
    # Cross-check with paper values
    # =========================================================================
    print("\n" + "=" * 70)
    print("CROSS-CHECK WITH PAPER VALUES")
    print("=" * 70)

    paper_values = {
        'B1_50Hz':   {'energy': 8796, 'battery': 216},
        'B2_25Hz':   {'energy': 6204, 'battery': 306},
        'B3_12.5Hz': {'energy': 4390, 'battery': 433},
        'B4_10Hz':   {'energy': 4390, 'battery': 433},
        'B5_6.25Hz': {'energy': 3612, 'battery': 526},
        'LiveEdge':  {'energy': 4136, 'battery': 461},
    }

    computed = {
        'B1_50Hz':   baselines[50],
        'B2_25Hz':   baselines[25],
        'B3_12.5Hz': baselines[12.5],
        'B4_10Hz':   baselines[10],
        'B5_6.25Hz': baselines[6.25],
        'LiveEdge':  a3,
    }

    print(f"\n{'Method':<15} | {'Paper E':>8} | {'Computed E':>10} | {'Diff':>6} | {'Paper Batt':>10} | {'Computed Batt':>13} | {'Status':>6}")
    print("-" * 85)

    all_ok = True
    for name in paper_values:
        pe = paper_values[name]['energy']
        pb = paper_values[name]['battery']
        ce = round(computed[name]['e_total'])
        cb = round(computed[name]['battery_days'])
        diff_e = ce - pe
        ok = abs(diff_e) <= 2  # allow ±2 mJ rounding
        if not ok:
            all_ok = False
        print(f"{name:<15} | {pe:>8} | {ce:>10} | {diff_e:>+5} | {pb:>9}d | {cb:>12}d | {'OK' if ok else 'DIFF':>6}")

    print(f"\nOverall: {'ALL MATCH' if all_ok else 'DISCREPANCIES FOUND'}")

    # =========================================================================
    # Datasheet Parameter Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("DATASHEET PARAMETER REFERENCE")
    print("=" * 70)
    print(f"""
Component       | Parameter              | Value      | Source
----------------|------------------------|------------|----------------------------------
CR2032          | Capacity               | {C_BAT} mAh    | Maxell CR2032 datasheet
                | Voltage                | {V}V       |
                | Usable fraction        | {ETA*100:.0f}%       | Standard derating
BMI160          | I_accel @6.25Hz (LP)   | 7 uA       | BST-BMI160-DS000 Table 8
                | I_accel @12.5Hz (LP)   | 10 uA      | BST-BMI160-DS000 Table 8
                | I_accel @25Hz (LP)     | 17 uA      | BST-BMI160-DS000 Table 8
                | I_accel @50Hz (LP)     | 27 uA      | BST-BMI160-DS000 Table 8
                | I_accel normal mode    | 180 uA     | BST-BMI160-DS000 Table 7 (unused)
nRF52832        | I_CPU @64MHz (DCDC)    | {I_MCU_ACTIVE} uA   | PS v1.8 (33 uA/MHz DCDC mode)
                | I_idle (SysON+RTC+RAM) | {I_MCU_IDLE} uA     | PS v1.8 (DCDC, full RAM+RTC)
                | I_TX @0dBm             | {I_TX} uA   | PS v1.8 Table 50
                | I_RX @1Mbps            | {I_RX} uA   | PS v1.8 Table 50
                | I_other (RTC+reg)      | {I_OTHER} uA    | PS v1.8
Renode DWT      | T_inference            | {T_INFERENCE*1000:.3f} ms  | 126,469 cycles @ 64MHz
""")
