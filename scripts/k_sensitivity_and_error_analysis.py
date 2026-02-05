#!/usr/bin/env python3
"""
k Sensitivity Analysis and Cluster Misclassification Impact Analysis

Methodology aligned with verified_simulation.py:
- Cluster classifier: compressed RF (30 trees, depth 8, balanced)
- Behavior classifier: RF (100, depth 20, min_samples_split=5, min_samples_leaf=2, NO balanced)
- FSM uses current-ODR features for cluster prediction (not fixed 50Hz)
- Scalers fitted per-fold on 50Hz training features
- 5-fold temporal-order CV (no shuffle)
"""

import sys
from pathlib import Path
import numpy as np
import json
from typing import Dict, List, Tuple
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root / 'src'))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

from liveedge.data import extract_24_features_batch

# =============================================================================
# Configuration (matches verified_simulation.py)
# =============================================================================

DATA_DIR = Path("L:/GitHub/LiveEdge/data/processed")
OUTPUT_DIR = Path("L:/GitHub/LiveEdge/outputs/k_sensitivity_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEHAVIORS_5CLASS = ['Lying', 'Eating', 'Standing', 'Walking', 'Drinking']

BEHAVIOR_TO_CLUSTER = {
    'Lying': 'STABLE',
    'Eating': 'STABLE',
    'Standing': 'ACTIVE',
    'Walking': 'MODERATE',
    'Drinking': 'ACTIVE',
}

CLUSTER_ODR = {'STABLE': 6.25, 'MODERATE': 12.5, 'ACTIVE': 25.0}

# Energy model (matches verified_simulation.py exactly)
BMI160_CURRENT = {6.25: 7, 12.5: 10, 25: 17, 50: 27}  # µA
V = 3.0
T_DAY = 86400
T_WINDOW = 1.5  # seconds
WINDOWS_PER_DAY = T_DAY / T_WINDOW  # 57,600
T_INFERENCE = 0.001976  # seconds (cycle-accurate from Renode DWT)
I_MCU_ACTIVE = 2100  # µA
I_MCU_IDLE = 1.9     # µA
I_TX, I_RX = 5300, 5400  # µA
T_TX, T_RX = 0.0015, 0.001  # seconds
I_OTHER = 2.05  # µA
BATTERY_CAPACITY = 220 * 3.0 * 3600 * 0.8  # mJ (CR2032, 80% usable)

# Baseline total energy at 50Hz, 100% inference
BASELINE_ENERGY_50HZ = 8796  # mJ/day


def compute_total_energy(odr_distribution: Dict[float, float],
                         inference_rate: float) -> float:
    """Compute total daily energy (IMU + MCU + overhead) in mJ.
    Matches verified_simulation.py compute_energy()."""
    # IMU
    i_imu_avg = sum(BMI160_CURRENT.get(odr, 10) * frac
                    for odr, frac in odr_distribution.items())
    e_imu = i_imu_avg * V * T_DAY / 1000

    # MCU
    n_windows = T_DAY / T_WINDOW
    n_inf = n_windows * inference_rate
    t_active = n_inf * T_INFERENCE
    t_idle = T_DAY - t_active
    e_mcu = (t_active * I_MCU_ACTIVE + t_idle * I_MCU_IDLE) * V / 1000

    # BLE + other overhead
    n_tx = T_DAY / 60  # once per minute
    e_ble = n_tx * (T_TX * I_TX + T_RX * I_RX) * V / 1000
    e_other = I_OTHER * V * T_DAY / 1000

    return e_imu + e_mcu + e_ble + e_other

# =============================================================================
# Data Loading and Feature Extraction
# =============================================================================

def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load full processed data (no subsampling)."""
    path = DATA_DIR / '50hz'
    X = np.load(path / 'windows.npy')[:, :, :3]
    y = np.load(path / 'labels.npy', allow_pickle=True)

    mask = np.isin(y, BEHAVIORS_5CLASS)
    X, y = X[mask], y[mask]

    return X, y


def extract_features(X: np.ndarray) -> np.ndarray:
    """Extract 24 statistical features."""
    return extract_24_features_batch(X)


def resample_batch(X: np.ndarray, target_odr: float, source_odr: float = 50) -> np.ndarray:
    """Resample windows to target ODR."""
    if target_odr >= source_odr:
        return X
    factor = int(source_odr / target_odr)
    return X[:, ::factor, :]

# =============================================================================
# FSM Simulation (matches verified_simulation.py exactly)
# =============================================================================

def simulate_fsm(cluster_preds_cache: Dict[float, np.ndarray],
                 behavior_preds_cache: Dict[float, np.ndarray],
                 y_true_clusters: np.ndarray,
                 y_true_behavior: np.ndarray,
                 k: int, n_samples: int) -> Dict:
    """Simulate FSM with given k, using ODR-dependent predictions.

    Matches verified_simulation.py run_liveedge_simulation() exactly:
    - Cluster prediction uses features at current ODR
    - Initial ODR = 25.0 Hz
    - Behavior prediction uses features at current ODR
    """
    current_cluster = None
    current_odr = 25.0
    last_pred_cluster = None
    consecutive_same = 0

    pred_clusters = []
    odr_history = []
    inference_flags = []
    predictions = []
    transitions = 0

    for i in range(n_samples):
        # Cluster prediction at current ODR
        pred_cluster = cluster_preds_cache[current_odr][i]

        if pred_cluster == last_pred_cluster:
            consecutive_same += 1
        else:
            consecutive_same = 1
        last_pred_cluster = pred_cluster

        should_transition = (consecutive_same >= k)

        if current_cluster is None or (pred_cluster != current_cluster and should_transition):
            if current_cluster is not None and pred_cluster != current_cluster:
                transitions += 1
            current_cluster = pred_cluster
            current_odr = CLUSTER_ODR[pred_cluster]
            inference_flags.append(True)
        else:
            inference_flags.append(consecutive_same <= k)

        pred_clusters.append(current_cluster)
        odr_history.append(current_odr)
        predictions.append(behavior_preds_cache[current_odr][i])

    # Metrics
    cluster_accuracy = accuracy_score(y_true_clusters, pred_clusters)
    behavior_accuracy = accuracy_score(y_true_behavior, predictions)
    inference_rate = np.mean(inference_flags)

    # ODR distribution
    odr_arr = np.array(odr_history)
    unique_odrs, counts = np.unique(odr_arr, return_counts=True)
    odr_distribution = {float(odr): float(cnt / len(odr_arr))
                        for odr, cnt in zip(unique_odrs, counts)}

    # Energy
    total_energy = compute_total_energy(odr_distribution, inference_rate)
    total_savings = (1 - total_energy / BASELINE_ENERGY_50HZ) * 100

    # Transitions per day
    transitions_per_day = transitions * (WINDOWS_PER_DAY / n_samples)

    # Effective ODR
    eff_odr = np.mean(odr_arr)

    # Detection delay: time from true cluster change to FSM matching
    detection_delays = []
    current_true = y_true_clusters[0]

    for i in range(1, n_samples):
        if y_true_clusters[i] != current_true:
            current_true = y_true_clusters[i]
            true_change_idx = i
            for j in range(i, min(i + 50, n_samples)):
                if pred_clusters[j] == current_true:
                    delay = (j - true_change_idx) * T_WINDOW
                    detection_delays.append(delay)
                    break

    mean_delay = np.mean(detection_delays) if detection_delays else 0
    max_delay = np.max(detection_delays) if detection_delays else 0

    return {
        'k': k,
        'cluster_accuracy': cluster_accuracy,
        'behavior_accuracy': behavior_accuracy,
        'inference_rate': inference_rate,
        'total_energy': round(total_energy),
        'total_savings_pct': total_savings,
        'transitions_per_day': round(transitions_per_day),
        'n_transitions_raw': transitions,
        'mean_delay_s': round(mean_delay, 1),
        'max_delay_s': round(max_delay, 1),
        'eff_odr': round(eff_odr, 2),
    }

# =============================================================================
# Experiment 1: k Sensitivity Analysis
# =============================================================================

def run_k_sensitivity_analysis(X: np.ndarray, y: np.ndarray,
                                k_values: List[int] = [1, 2, 3, 4, 5]) -> Dict:
    """Analyze sensitivity of k on energy-accuracy-delay trade-off.

    Methodology matches verified_simulation.py:
    - Scalers fitted per-fold on 50Hz train features
    - Behavior RF: 100 trees, depth 20, NO balanced
    - Cluster RF: 30 trees, depth 8, balanced (compressed)
    - FSM uses current-ODR features for prediction
    """
    print("\n" + "=" * 70)
    print("Experiment 1: k Sensitivity Analysis")
    print("=" * 70)

    y_cluster = np.array([BEHAVIOR_TO_CLUSTER[b] for b in y])

    le_cluster = LabelEncoder()
    y_cluster_encoded = le_cluster.fit_transform(y_cluster)

    le_behavior = LabelEncoder()
    y_behavior_encoded = le_behavior.fit_transform(y)

    # 5-fold temporal-order CV (no shuffle)
    cv = StratifiedKFold(n_splits=5, shuffle=False)

    fold_results = {k: [] for k in k_values}

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y_cluster_encoded)):
        X_train, X_val = X[train_idx], X[val_idx]

        # Extract 50Hz training features
        X_train_features = extract_features(X_train)

        # Fit scalers per fold on 50Hz training features
        scaler_behavior = StandardScaler()
        scaler_behavior.fit(X_train_features)

        scaler_cluster = StandardScaler()
        scaler_cluster.fit(X_train_features)

        # Train behavior classifier (matches verified_simulation.py: NO balanced)
        clf_behavior = RandomForestClassifier(
            n_estimators=100, max_depth=20,
            min_samples_split=5, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        )
        clf_behavior.fit(scaler_behavior.transform(X_train_features),
                         y_behavior_encoded[train_idx])

        # Train cluster classifier (compressed, matches verified_simulation.py)
        clf_cluster = RandomForestClassifier(
            n_estimators=30, max_depth=8,
            class_weight='balanced',
            random_state=42, n_jobs=-1
        )
        clf_cluster.fit(scaler_cluster.transform(X_train_features),
                        y_cluster_encoded[train_idx])

        # Pre-compute predictions at all ODRs (matches verified_simulation.py)
        cluster_preds_cache = {}
        behavior_preds_cache = {}
        for odr in CLUSTER_ODR.values():
            X_resampled = resample_batch(X_val, odr)
            features = extract_features(X_resampled)

            cluster_pred_encoded = clf_cluster.predict(scaler_cluster.transform(features))
            cluster_preds_cache[odr] = le_cluster.inverse_transform(cluster_pred_encoded)

            behavior_pred_encoded = clf_behavior.predict(scaler_behavior.transform(features))
            behavior_preds_cache[odr] = le_behavior.inverse_transform(behavior_pred_encoded)

        y_true_clusters = y_cluster[val_idx]
        y_true_behavior = y[val_idx]

        # Run FSM for each k
        for k in k_values:
            result = simulate_fsm(
                cluster_preds_cache, behavior_preds_cache,
                y_true_clusters, y_true_behavior,
                k, len(val_idx)
            )
            fold_results[k].append(result)

        print(f"    Fold {fold+1}/5 done")

    # Average across folds
    results = {}
    for k in k_values:
        avg = {}
        for key in fold_results[k][0]:
            if key == 'k':
                avg[key] = k
            elif isinstance(fold_results[k][0][key], (int, float, np.floating)):
                avg[key] = float(np.mean([r[key] for r in fold_results[k]]))
        avg['transitions_per_day'] = round(avg['transitions_per_day'])
        avg['total_energy'] = round(avg['total_energy'])
        avg['mean_delay_s'] = round(avg['mean_delay_s'], 1)
        avg['max_delay_s'] = round(avg['max_delay_s'], 1)
        avg['eff_odr'] = round(avg['eff_odr'], 2)
        results[k] = avg

    # Print results
    print(f"\n  {'k':>3s} | {'Cl Acc':>8s} | {'Bh Acc':>8s} | {'Inf Rate':>8s} | "
          f"{'Energy':>7s} | {'Savings':>8s} | {'Trans/d':>8s} | {'Delay':>6s}")
    print("  " + "-" * 80)

    for k in k_values:
        r = results[k]
        print(f"  {k:>3d} | {r['cluster_accuracy']:>8.2%} | "
              f"{r['behavior_accuracy']:>8.2%} | "
              f"{r['inference_rate']:>8.1%} | "
              f"{r['total_energy']:>6d} | "
              f"{r['total_savings_pct']:>7.1f}% | "
              f"{r['transitions_per_day']:>8,d} | "
              f"{r['mean_delay_s']:>5.1f}s")

    return {
        'k_results': results,
    }

# =============================================================================
# Experiment 2: Cluster Misclassification Impact Analysis
# =============================================================================

def analyze_misclassification_impact(X: np.ndarray, y: np.ndarray) -> Dict:
    """Analyze cascading effects of cluster misclassification.

    Uses 50Hz features for raw classifier confusion matrix analysis.
    This measures classifier capability, not FSM runtime performance.
    """
    print("\n" + "=" * 70)
    print("Experiment 2: Cluster Misclassification Impact Analysis")
    print("=" * 70)

    y_cluster = np.array([BEHAVIOR_TO_CLUSTER[b] for b in y])

    # Extract features
    X_features = extract_features(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)

    le_cluster = LabelEncoder()
    y_cluster_encoded = le_cluster.fit_transform(y_cluster)
    cluster_names = le_cluster.classes_

    # Compressed cluster classifier (matching paper)
    clf = RandomForestClassifier(
        n_estimators=30, max_depth=8,
        class_weight='balanced', random_state=42, n_jobs=-1
    )

    cv = StratifiedKFold(n_splits=5, shuffle=False)
    y_pred_all = np.zeros_like(y_cluster_encoded)

    for train_idx, val_idx in cv.split(X_scaled, y_cluster_encoded):
        clf.fit(X_scaled[train_idx], y_cluster_encoded[train_idx])
        y_pred_all[val_idx] = clf.predict(X_scaled[val_idx])

    y_pred_clusters = le_cluster.inverse_transform(y_pred_all)

    # Analyze misclassification patterns
    print("\n  Misclassification Analysis:")
    print(f"  {'True -> Pred':>25s} | {'Count':>8s} | {'Rate':>8s} | "
          f"{'ODR Impact':>12s} | {'Energy Impact':>15s}")
    print("  " + "-" * 80)

    misclass_analysis = {}
    for true_cluster in cluster_names:
        for pred_cluster in cluster_names:
            if true_cluster == pred_cluster:
                continue

            mask = (y_cluster == true_cluster) & (y_pred_clusters == pred_cluster)
            count = np.sum(mask)
            total_true = np.sum(y_cluster == true_cluster)

            if count > 0:
                rate = count / total_true
                true_odr = CLUSTER_ODR[true_cluster]
                pred_odr = CLUSTER_ODR[pred_cluster]
                odr_diff = pred_odr - true_odr
                energy_impact = "Under-sample" if pred_odr < true_odr else "Over-sample"

                key = f"{true_cluster}->{pred_cluster}"
                misclass_analysis[key] = {
                    'count': int(count),
                    'rate': rate,
                    'true_odr': true_odr,
                    'pred_odr': pred_odr,
                    'odr_diff': odr_diff,
                    'energy_impact': energy_impact
                }

                print(f"  {key:>25s} | {count:>8d} | {rate:>7.1%} | "
                      f"{odr_diff:>+10.2f}Hz | {energy_impact:>15s}")

    # Analyze cascading effects: consecutive misclassifications
    print("\n  Cascading Effect Analysis (consecutive misclassifications):")

    is_correct = y_cluster == y_pred_clusters
    run_lengths = []
    current_run = 0

    for correct in is_correct:
        if not correct:
            current_run += 1
        else:
            if current_run > 0:
                run_lengths.append(current_run)
            current_run = 0

    if current_run > 0:
        run_lengths.append(current_run)

    if run_lengths:
        run_lengths = np.array(run_lengths)
        print(f"    Total error runs: {len(run_lengths)}")
        print(f"    Mean run length: {np.mean(run_lengths):.1f} windows "
              f"({np.mean(run_lengths)*T_WINDOW:.1f}s)")
        print(f"    Max run length: {np.max(run_lengths)} windows "
              f"({np.max(run_lengths)*T_WINDOW:.1f}s)")
        print(f"    Runs > 3 windows: {np.sum(run_lengths > 3)} "
              f"({np.sum(run_lengths > 3)/len(run_lengths)*100:.1f}%)")

        cascading_stats = {
            'n_error_runs': len(run_lengths),
            'mean_run_length': float(np.mean(run_lengths)),
            'max_run_length': int(np.max(run_lengths)),
            'runs_gt_3': int(np.sum(run_lengths > 3))
        }
    else:
        cascading_stats = {'n_error_runs': 0}

    # Critical misclassification: ACTIVE -> STABLE
    critical_mask = (y_cluster == 'ACTIVE') & (y_pred_clusters == 'STABLE')
    critical_count = np.sum(critical_mask)
    critical_rate = (critical_count / np.sum(y_cluster == 'ACTIVE')
                     if np.sum(y_cluster == 'ACTIVE') > 0 else 0)

    print(f"\n  Critical Misclassification (ACTIVE -> STABLE):")
    print(f"    Count: {critical_count}")
    print(f"    Rate: {critical_rate:.2%}")
    print(f"    Impact: Under-sampling by {25.0 - 6.25:.2f} Hz")

    return {
        'misclassification_patterns': misclass_analysis,
        'cascading_effects': cascading_stats,
        'critical_misclass': {
            'active_to_stable_count': int(critical_count),
            'active_to_stable_rate': critical_rate
        },
    }

# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("k Sensitivity & Cluster Misclassification Analysis")
    print("=" * 70)

    # Load data (full dataset, no subsampling)
    print("\n[Loading Data]...")
    X, y = load_data()
    print(f"  Loaded {len(X)} samples")

    # Experiment 1: k sensitivity (k=1 to k=6)
    k_results = run_k_sensitivity_analysis(X, y, k_values=[1, 2, 3, 4, 5, 6])

    # Experiment 2: Misclassification impact
    misclass_results = analyze_misclassification_impact(X, y)

    # Compile results
    all_results = {
        'k_sensitivity': k_results,
        'misclassification_impact': misclass_results
    }

    # Convert numpy types
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64, np.floating)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64, np.integer)):
            return int(obj)
        elif isinstance(obj, dict):
            return {str(k): convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj

    all_results = convert_numpy(all_results)

    # Save
    with open(OUTPUT_DIR / 'k_sensitivity_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print(f"\n  k Sensitivity:")
    for k in [1, 3, 5]:
        r = k_results['k_results'][k]
        print(f"    k={k}: ClAcc={r['cluster_accuracy']:.1%}, "
              f"BhAcc={r['behavior_accuracy']:.1%}, "
              f"Inf={r['inference_rate']:.1%}, "
              f"Energy={r['total_energy']}mJ, "
              f"Savings={r['total_savings_pct']:.1f}%, "
              f"Trans/day={r['transitions_per_day']:,}, "
              f"Delay={r['mean_delay_s']:.1f}s")

    print(f"\n  Cluster Misclassification:")
    print(f"    ACTIVE->STABLE (critical): "
          f"{misclass_results['critical_misclass']['active_to_stable_rate']:.2%}")
    print(f"    Mean error run: "
          f"{misclass_results['cascading_effects'].get('mean_run_length', 0):.1f} windows")

    print(f"\n[Results saved to: {OUTPUT_DIR}]")

    return all_results


if __name__ == "__main__":
    main()
