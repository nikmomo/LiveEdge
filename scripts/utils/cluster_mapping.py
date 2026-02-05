#!/usr/bin/env python3
"""Cluster mapping utilities for LiveEdge."""

from typing import Dict

# Behavior to cluster mapping
BEHAVIOR_TO_CLUSTER: Dict[str, str] = {
    'Lying': 'STABLE',
    'Eating': 'STABLE',
    'Standing': 'ACTIVE',
    'Walking': 'MODERATE',
    'Drinking': 'ACTIVE',
}

# Cluster to ODR mapping
CLUSTER_ODR: Dict[str, float] = {
    'STABLE': 6.25,
    'MODERATE': 12.5,
    'ACTIVE': 25.0,
}

# BMI160 IMU power consumption (μA at each ODR)
BMI160_CURRENT: Dict[float, float] = {
    6.25: 7,
    12.5: 10,
    25: 17,
    50: 27,
}

# Energy model parameters
ENERGY_PARAMS = {
    'V': 3.0,
    'T_DAY': 86400,
    'T_WINDOW': 1.5,
    'I_MCU_ACTIVE': 2100,
    'I_MCU_IDLE': 1.9,
    'T_INFERENCE': 0.001976,  # Renode DWT: 126,469 cycles @ 64MHz
    'I_TX': 5300,
    'I_RX': 5400,
    'T_TX': 0.0015,
    'T_RX': 0.001,
    'I_OTHER': 2.05,
    'BATTERY_CAPACITY': 220,
    'BATTERY_EFFICIENCY': 0.8,
}

# Pre-computed baseline values (mJ/day, days) with T_inf=1.976ms
BASELINE_ENERGY: Dict[float, int] = {
    50: 8796, 25: 6204, 12.5: 4390, 10: 4390, 6.25: 3612,
}

BASELINE_BATTERY: Dict[float, int] = {
    50: 216, 25: 306, 12.5: 433, 10: 433, 6.25: 526,
}

BEHAVIORS_5CLASS = ['Lying', 'Eating', 'Standing', 'Walking', 'Drinking']


def get_cluster_for_behavior(behavior: str) -> str:
    """Get the cluster for a given behavior."""
    return BEHAVIOR_TO_CLUSTER[behavior]


def get_odr_for_cluster(cluster: str) -> float:
    """Get the ODR for a given cluster."""
    return CLUSTER_ODR[cluster]


def get_odr_for_behavior(behavior: str) -> float:
    """Get the ODR for a given behavior."""
    return CLUSTER_ODR[BEHAVIOR_TO_CLUSTER[behavior]]


def compute_energy(
    odr_distribution: Dict[float, float],
    inference_rate: float,
    tx_interval: float = 60.0,
) -> Dict:
    """Compute energy consumption based on ODR distribution and inference rate."""
    p = ENERGY_PARAMS

    # IMU energy (mJ/day)
    i_imu_avg = sum(BMI160_CURRENT.get(odr, 10) * frac
                   for odr, frac in odr_distribution.items())
    e_imu = i_imu_avg * p['V'] * p['T_DAY'] / 1000

    # MCU energy (mJ/day)
    n_windows = p['T_DAY'] / p['T_WINDOW']
    n_inferences = n_windows * inference_rate
    t_active = n_inferences * p['T_INFERENCE']
    t_idle = p['T_DAY'] - t_active
    e_mcu = (t_active * p['I_MCU_ACTIVE'] + t_idle * p['I_MCU_IDLE']) * p['V'] / 1000

    # BLE energy (mJ/day)
    n_tx = p['T_DAY'] / tx_interval
    e_ble = n_tx * (p['T_TX'] * p['I_TX'] + p['T_RX'] * p['I_RX']) * p['V'] / 1000

    # Other peripherals (mJ/day)
    e_other = p['I_OTHER'] * p['V'] * p['T_DAY'] / 1000

    e_total = e_imu + e_mcu + e_ble + e_other

    # Battery life (days)
    battery_energy = (p['BATTERY_CAPACITY'] * p['V'] * 3600 * p['BATTERY_EFFICIENCY'])
    battery_days = battery_energy / e_total

    return {
        'e_imu': round(e_imu, 1),
        'e_mcu': round(e_mcu, 1),
        'e_ble': round(e_ble, 1),
        'e_other': round(e_other, 1),
        'e_total': round(e_total, 1),
        'i_imu_avg': round(i_imu_avg, 2),
        'battery_days': round(battery_days, 1),
        'savings_vs_50hz': round((1 - e_total / BASELINE_ENERGY[50]) * 100, 1),
    }


def get_cluster_distribution(behavior_counts: Dict[str, int]) -> Dict[str, float]:
    """Compute cluster distribution from behavior counts."""
    total = sum(behavior_counts.values())
    if total == 0:
        return {'STABLE': 0.0, 'MODERATE': 0.0, 'ACTIVE': 0.0}

    cluster_counts = {'STABLE': 0, 'MODERATE': 0, 'ACTIVE': 0}
    for behavior, count in behavior_counts.items():
        if behavior in BEHAVIOR_TO_CLUSTER:
            cluster_counts[BEHAVIOR_TO_CLUSTER[behavior]] += count

    return {cluster: count / total for cluster, count in cluster_counts.items()}


def get_odr_distribution(behavior_counts: Dict[str, int]) -> Dict[float, float]:
    """Compute ODR distribution from behavior counts."""
    cluster_dist = get_cluster_distribution(behavior_counts)
    return {CLUSTER_ODR[cluster]: frac for cluster, frac in cluster_dist.items()}


if __name__ == "__main__":
    print("Cluster Mapping Configuration")
    print("=" * 50)
    for behavior, cluster in BEHAVIOR_TO_CLUSTER.items():
        odr = CLUSTER_ODR[cluster]
        print(f"  {behavior:10} -> {cluster:10} ({odr:5.2f} Hz)")
