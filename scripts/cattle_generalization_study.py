#!/usr/bin/env python3
"""Cattle generalization study for cross-species validation (Table 9).

Uses the SAME methodology as verified_simulation.py:
- Temporal-order CV (no shuffle) for valid FSM simulation
- Single behavior classifier RF(100, d=20) + deterministic cluster mapping
- k=3 hysteresis FSM with adaptive ODR and inference scheduling
- Cycle-accurate energy model (T_inf=1.976ms)

Pig reference values come directly from Table 5 main results.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json
from typing import Dict, Tuple
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
# Configuration
# =============================================================================

RAW_DATA_PATH = Path("L:/GitHub/LiveEdge/data/raw/imu_cow_dataset_fall-2022.csv")
OUTPUT_DIR = Path("L:/GitHub/LiveEdge/outputs/cattle_generalization")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLING_RATE = 50  # Hz
WINDOW_SIZE = 1.5   # seconds
WINDOW_SAMPLES = int(SAMPLING_RATE * WINDOW_SIZE)  # 75 samples
OVERLAP = 0.5       # 50% overlap

# Cattle behavior mapping (14 raw → 5 standardized)
CATTLE_BEHAVIOR_MAP = {
    # STABLE cluster (low activity) → 6.25 Hz
    'lying': 'Lying',
    'lying to standing': 'Lying',
    'staning to lying': 'Lying',
    'chewing': 'Eating',
    'grazing': 'Eating',
    'grazing and standing': 'Eating',
    'licking': 'Eating',
    # MODERATE cluster (locomotion) → 12.5 Hz
    'walking': 'Walking',
    'grazing and walking': 'Walking',
    'grazing to walking or walking to grazing': 'Walking',
    # ACTIVE cluster (postural/rapid) → 25 Hz
    'standing': 'Standing',
    'drinking': 'Drinking',
    'cow interactions': 'Interacting',
    'running': 'Running',
}

BEHAVIOR_TO_CLUSTER = {
    'Lying': 'STABLE',
    'Eating': 'STABLE',
    'Walking': 'MODERATE',
    'Standing': 'ACTIVE',
    'Drinking': 'ACTIVE',
}

CLUSTER_ODR = {'STABLE': 6.25, 'MODERATE': 12.5, 'ACTIVE': 25.0}
BMI160_CURRENT = {6.25: 7, 12.5: 10, 25: 17, 50: 27}

# 5-class behaviors (excluding rare Interacting/Running)
BEHAVIORS_5CLASS = ['Lying', 'Eating', 'Walking', 'Standing', 'Drinking']

# Energy model parameters (same as verified_simulation.py)
V = 3.0
T_DAY = 86400
T_WINDOW = 1.5
I_MCU_ACTIVE = 2100
I_MCU_IDLE = 1.9
T_INFERENCE = 0.001976
I_TX, I_RX = 5300, 5400
T_TX, T_RX = 0.0015, 0.001
I_OTHER = 2.05


# =============================================================================
# Data Loading (temporal order preserved)
# =============================================================================

def load_and_prepare_cattle_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load cattle CSV, map behaviors, create temporally-ordered windows."""
    print("[Loading Cattle Data]")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"  Total samples: {len(df):,}")

    # Map behaviors
    df['mapped_label'] = df['label'].map(CATTLE_BEHAVIOR_MAP)
    df = df.dropna(subset=['mapped_label'])

    # Filter to 5 core classes
    df = df[df['mapped_label'].isin(BEHAVIORS_5CLASS)].reset_index(drop=True)
    print(f"  After filtering to 5-class: {len(df):,} samples")

    # Print distribution
    print(f"  Label distribution:")
    for label in BEHAVIORS_5CLASS:
        count = (df['mapped_label'] == label).sum()
        print(f"    {label}: {count:,} ({count/len(df)*100:.1f}%)")

    # Create temporally-ordered windows within behavior segments
    print(f"\n[Creating Windows (temporal order)]")
    all_X = []
    all_y = []

    step = int(WINDOW_SAMPLES * (1 - OVERLAP))

    # Find behavior segments (consecutive rows with same mapped label)
    labels = df['mapped_label'].values
    accel = df[['ax', 'ay', 'az']].values

    segment_starts = [0]
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1]:
            segment_starts.append(i)
    segment_starts.append(len(labels))  # sentinel

    n_segments = len(segment_starts) - 1
    print(f"  Total behavior segments: {n_segments:,}")

    for seg_idx in range(n_segments):
        start = segment_starts[seg_idx]
        end = segment_starts[seg_idx + 1]
        seg_label = labels[start]
        seg_data = accel[start:end]

        n_windows = (len(seg_data) - WINDOW_SAMPLES) // step + 1
        for w in range(n_windows):
            w_start = w * step
            w_end = w_start + WINDOW_SAMPLES
            if w_end <= len(seg_data):
                all_X.append(seg_data[w_start:w_end])
                all_y.append(seg_label)

    X = np.array(all_X)
    y = np.array(all_y)
    print(f"  Total windows: {len(X):,} (temporal order preserved)")

    # Print cluster distribution
    cluster_labels = np.array([BEHAVIOR_TO_CLUSTER[b] for b in y])
    for cluster in ['STABLE', 'MODERATE', 'ACTIVE']:
        count = np.sum(cluster_labels == cluster)
        print(f"  {cluster}: {count:,} ({count/len(y)*100:.1f}%)")

    return X, y


# =============================================================================
# Energy Computation (same as verified_simulation.py)
# =============================================================================

def compute_energy(odr_distribution: Dict[float, float],
                   inference_rate: float) -> Dict:
    """Compute daily energy from first principles."""
    i_imu_avg = sum(BMI160_CURRENT.get(odr, 10) * frac
                    for odr, frac in odr_distribution.items())
    e_imu = i_imu_avg * V * T_DAY / 1000

    n_windows = T_DAY / T_WINDOW
    n_inferences = n_windows * inference_rate
    t_active = n_inferences * T_INFERENCE
    t_idle = T_DAY - t_active
    e_mcu = (t_active * I_MCU_ACTIVE + t_idle * I_MCU_IDLE) * V / 1000

    n_tx = T_DAY / 60
    e_ble = n_tx * (T_TX * I_TX + T_RX * I_RX) * V / 1000
    e_other = I_OTHER * V * T_DAY / 1000

    e_total = e_imu + e_mcu + e_ble + e_other
    battery_energy = 220 * 3.0 * 3600 * 0.8
    battery_days = battery_energy / e_total

    return {
        'e_imu': round(e_imu, 1),
        'e_mcu': round(e_mcu, 1),
        'e_total': round(e_total, 1),
        'battery_days': round(battery_days, 1),
        'savings_vs_50hz': round((1 - e_total / 8796.0) * 100, 1),
    }


# =============================================================================
# FSM Simulation (same logic as verified_simulation.py)
# =============================================================================

def resample_batch(X: np.ndarray, target_odr: float, source_odr: float = 50) -> np.ndarray:
    """Resample all windows to target ODR."""
    if target_odr >= source_odr:
        return X
    factor = int(source_odr / target_odr)
    return X[:, ::factor, :]


def run_fsm_simulation(X_test, y_test_behavior_str,
                       behavior_classifier, scaler,
                       le_behavior,
                       k_stability=3):
    """Run LiveEdge FSM simulation (same as verified_simulation.py).

    Single behavior classifier + deterministic cluster mapping.
    """
    n_samples = len(X_test)

    # Pre-compute features and predictions at all ODRs
    cluster_preds_cache = {}
    behavior_preds_cache = {}
    for odr in CLUSTER_ODR.values():
        X_resampled = resample_batch(X_test, odr)
        features = extract_24_features_batch(X_resampled)
        features_scaled = scaler.transform(features)
        pred_indices = behavior_classifier.predict(features_scaled)
        behavior_preds = le_behavior.inverse_transform(pred_indices)
        behavior_preds_cache[odr] = behavior_preds
        cluster_preds_cache[odr] = np.array([BEHAVIOR_TO_CLUSTER[b] for b in behavior_preds])

    # FSM State
    current_cluster = None
    current_odr = 25.0
    last_pred_cluster = None
    consecutive_same = 0

    predictions = []
    pred_clusters = []
    odr_history = []
    inference_flags = []

    for i in range(n_samples):
        pred_cluster = cluster_preds_cache[current_odr][i]

        if pred_cluster == last_pred_cluster:
            consecutive_same += 1
        else:
            consecutive_same = 1
        last_pred_cluster = pred_cluster

        should_transition = (consecutive_same >= k_stability)

        if current_cluster is None or (pred_cluster != current_cluster and should_transition):
            current_cluster = pred_cluster
            current_odr = CLUSTER_ODR[pred_cluster]
            inference_flags.append(True)
        else:
            inference_flags.append(consecutive_same <= k_stability)

        pred_clusters.append(current_cluster)
        odr_history.append(current_odr)
        predictions.append(behavior_preds_cache[current_odr][i])

    # Metrics
    accuracy = accuracy_score(y_test_behavior_str, predictions)
    true_clusters = [BEHAVIOR_TO_CLUSTER[b] for b in y_test_behavior_str]
    cluster_accuracy = accuracy_score(true_clusters, pred_clusters)
    cluster_f1 = f1_score(true_clusters, pred_clusters, average='macro',
                          labels=['STABLE', 'MODERATE', 'ACTIVE'])

    odr_dist = {odr: odr_history.count(odr) / len(odr_history) for odr in set(odr_history)}
    eff_odr = np.mean(odr_history)
    inference_rate = sum(inference_flags) / len(inference_flags)
    energy = compute_energy(odr_dist, inference_rate)

    return {
        'accuracy': round(accuracy, 4),
        'cluster_accuracy': round(cluster_accuracy, 4),
        'cluster_f1': round(cluster_f1, 4),
        'eff_odr': round(eff_odr, 2),
        'inference_rate': round(inference_rate, 4),
        'odr_distribution': {str(k): round(v, 4) for k, v in odr_dist.items()},
        'energy': energy,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("Cattle Generalization Study (Table 9)")
    print("Methodology: Temporal CV + FSM Simulation (matches verified_simulation.py)")
    print("=" * 70)

    # Load data
    X, y = load_and_prepare_cattle_data()

    # Encode labels
    le_behavior = LabelEncoder()
    y_behavior_encoded = le_behavior.fit_transform(y)

    y_cluster_labels = np.array([BEHAVIOR_TO_CLUSTER[b] for b in y])
    le_cluster = LabelEncoder()
    y_cluster_encoded = le_cluster.fit_transform(y_cluster_labels)

    print(f"\n  Behaviors: {list(le_behavior.classes_)}")
    print(f"  Clusters: {list(le_cluster.classes_)}")

    # =========================================================================
    # 5-fold temporal-order CV with FSM simulation
    # =========================================================================
    print(f"\n[Running 5-fold Temporal-Order CV + FSM Simulation]")
    cv = StratifiedKFold(n_splits=5, shuffle=False)

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y_cluster_encoded)):
        print(f"\n  --- Fold {fold + 1}/5 ---")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train_bh = y_behavior_encoded[train_idx]
        y_test_bh_str = le_behavior.inverse_transform(y_behavior_encoded[test_idx])

        # Extract features
        X_train_features = extract_24_features_batch(X_train)

        # Single behavior classifier (same as verified_simulation.py)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_features)

        behavior_clf = RandomForestClassifier(
            n_estimators=100, max_depth=20,
            min_samples_split=5, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        )
        behavior_clf.fit(X_train_scaled, y_train_bh)

        # Baseline: behavior accuracy @50Hz
        X_test_features = extract_24_features_batch(X_test)
        X_test_scaled = scaler.transform(X_test_features)
        y_pred_bh = le_behavior.inverse_transform(behavior_clf.predict(X_test_scaled))
        baseline_acc = accuracy_score(y_test_bh_str, y_pred_bh)

        # Cluster accuracy @50Hz (via behavior→cluster mapping)
        y_pred_cl = np.array([BEHAVIOR_TO_CLUSTER[b] for b in y_pred_bh])
        true_clusters = [BEHAVIOR_TO_CLUSTER[b] for b in y_test_bh_str]
        cluster_acc_50hz = accuracy_score(true_clusters, y_pred_cl)

        print(f"    Baseline @50Hz: BhAcc={baseline_acc*100:.2f}%, ClAcc={cluster_acc_50hz*100:.2f}%")

        # FSM simulation
        result = run_fsm_simulation(
            X_test, y_test_bh_str,
            behavior_clf, scaler,
            le_behavior,
            k_stability=3
        )

        result['baseline_accuracy'] = round(baseline_acc, 4)
        result['cluster_acc_50hz'] = round(cluster_acc_50hz, 4)
        fold_results.append(result)

        print(f"    LiveEdge: BhAcc={result['accuracy']*100:.2f}%, "
              f"ClAcc={result['cluster_accuracy']*100:.2f}%, "
              f"ODR={result['eff_odr']:.1f}Hz, "
              f"Inf={result['inference_rate']*100:.1f}%, "
              f"Savings={result['energy']['savings_vs_50hz']:.1f}%")

    # =========================================================================
    # Aggregate results
    # =========================================================================
    print("\n" + "=" * 70)
    print("CATTLE RESULTS (5-fold Temporal-Order CV)")
    print("=" * 70)

    mean_bh_acc = np.mean([r['accuracy'] for r in fold_results]) * 100
    std_bh_acc = np.std([r['accuracy'] for r in fold_results]) * 100
    mean_cl_acc = np.mean([r['cluster_accuracy'] for r in fold_results]) * 100
    mean_cl_f1 = np.mean([r['cluster_f1'] for r in fold_results]) * 100
    mean_eff_odr = np.mean([r['eff_odr'] for r in fold_results])
    mean_inf_rate = np.mean([r['inference_rate'] for r in fold_results]) * 100
    mean_energy = np.mean([r['energy']['e_total'] for r in fold_results])
    mean_battery = np.mean([r['energy']['battery_days'] for r in fold_results])
    mean_savings = np.mean([r['energy']['savings_vs_50hz'] for r in fold_results])
    mean_baseline = np.mean([r['baseline_accuracy'] for r in fold_results]) * 100

    print(f"\n  Baseline @50Hz Behavior Acc: {mean_baseline:.2f}% +/- {std_bh_acc:.2f}%")
    print(f"  LiveEdge Behavior Acc:       {mean_bh_acc:.2f}% +/- {std_bh_acc:.2f}%")
    print(f"  Runtime Cluster Acc:         {mean_cl_acc:.2f}%")
    print(f"  Cluster Macro-F1:            {mean_cl_f1:.2f}%")
    print(f"  Effective ODR:               {mean_eff_odr:.2f} Hz")
    print(f"  Inference Rate:              {mean_inf_rate:.1f}%")
    print(f"  Energy:                      {mean_energy:.0f} mJ/day")
    print(f"  Battery:                     {mean_battery:.0f} days")
    print(f"  Energy Savings:              {mean_savings:.1f}%")

    # =========================================================================
    # Compare with pig (Table 5 main results)
    # =========================================================================
    print("\n" + "=" * 70)
    print("CROSS-SPECIES COMPARISON (Table 9)")
    print("=" * 70)

    # Pig reference from Table 5 (verified_simulation.py, full model, temporal CV)
    pig = {
        'behavior_acc': 66.85,
        'cluster_acc': 74.31,
        'eff_odr': 8.67,
        'inference_rate': 31.7,
        'energy': 3450,
        'battery': 554,
        'savings': 60.8,
    }

    cattle = {
        'behavior_acc': mean_bh_acc,
        'cluster_acc': mean_cl_acc,
        'eff_odr': mean_eff_odr,
        'inference_rate': mean_inf_rate,
        'energy': mean_energy,
        'battery': mean_battery,
        'savings': mean_savings,
    }

    print(f"\n  {'Metric':<25} | {'Pig':>10} | {'Cattle':>10} | {'Diff':>10}")
    print(f"  {'-'*60}")
    print(f"  {'Behavior Accuracy':<25} | {pig['behavior_acc']:>9.2f}% | {cattle['behavior_acc']:>9.2f}% | {cattle['behavior_acc']-pig['behavior_acc']:>+9.2f}%")
    print(f"  {'Cluster Accuracy':<25} | {pig['cluster_acc']:>9.2f}% | {cattle['cluster_acc']:>9.2f}% | {cattle['cluster_acc']-pig['cluster_acc']:>+9.2f}%")
    print(f"  {'Effective ODR':<25} | {pig['eff_odr']:>8.2f}Hz | {cattle['eff_odr']:>8.2f}Hz | {cattle['eff_odr']-pig['eff_odr']:>+8.2f}Hz")
    print(f"  {'Inference Rate':<25} | {pig['inference_rate']:>9.1f}% | {cattle['inference_rate']:>9.1f}% | {cattle['inference_rate']-pig['inference_rate']:>+9.1f}%")
    print(f"  {'Energy (mJ/day)':<25} | {pig['energy']:>10.0f} | {cattle['energy']:>10.0f} | {cattle['energy']-pig['energy']:>+10.0f}")
    print(f"  {'Battery (days)':<25} | {pig['battery']:>10.0f} | {cattle['battery']:>10.0f} | {cattle['battery']-pig['battery']:>+10.0f}")
    print(f"  {'Energy Savings':<25} | {pig['savings']:>9.1f}% | {cattle['savings']:>9.1f}% | {cattle['savings']-pig['savings']:>+9.1f}%")

    # Cluster distribution
    all_clusters = [BEHAVIOR_TO_CLUSTER[b] for b in y]
    total = len(all_clusters)
    print(f"\n  Cluster Distribution:")
    print(f"    {'Cluster':<10} | {'Pig':>6} | {'Cattle':>6}")
    print(f"    {'-'*28}")
    print(f"    {'STABLE':<10} | {'75%':>6} | {sum(1 for c in all_clusters if c=='STABLE')/total*100:>5.0f}%")
    print(f"    {'MODERATE':<10} | {'11%':>6} | {sum(1 for c in all_clusters if c=='MODERATE')/total*100:>5.0f}%")
    print(f"    {'ACTIVE':<10} | {'14%':>6} | {sum(1 for c in all_clusters if c=='ACTIVE')/total*100:>5.0f}%")

    # Save results
    save_results = {
        'cattle': {
            'behavior_accuracy': round(mean_bh_acc, 2),
            'behavior_accuracy_std': round(std_bh_acc, 2),
            'cluster_accuracy': round(mean_cl_acc, 2),
            'cluster_f1': round(mean_cl_f1, 2),
            'effective_odr': round(mean_eff_odr, 2),
            'inference_rate': round(mean_inf_rate, 1),
            'energy_mj_day': round(mean_energy),
            'battery_days': round(mean_battery),
            'savings_pct': round(mean_savings, 1),
            'fold_results': [{k: float(v) if isinstance(v, (np.floating, float)) else v
                             for k, v in r.items()} for r in fold_results],
        },
        'pig_reference': pig,
        'methodology': 'temporal-order CV, FSM simulation k=3, single behavior RF(100,d20) + cluster mapping',
    }

    output_file = OUTPUT_DIR / 'cattle_generalization_results.json'
    with open(output_file, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_file}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)

    return save_results


if __name__ == "__main__":
    main()
