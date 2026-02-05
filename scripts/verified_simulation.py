#!/usr/bin/env python3
"""LiveEdge simulation with FSM controller and k-hysteresis."""

import sys
from pathlib import Path
import numpy as np
import json
from typing import Dict
from collections import defaultdict

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root / 'src'))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("L:/GitHub/LiveEdge/data/processed")
OUTPUT_DIR = Path("L:/GitHub/LiveEdge/outputs/verified_simulation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEHAVIOR_TO_CLUSTER = {
    'Lying': 'STABLE',
    'Eating': 'MODERATE',
    'Standing': 'ACTIVE',
    'Walking': 'ACTIVE',
    'Drinking': 'ACTIVE',
}

CLUSTER_ODR = {'STABLE': 6.25, 'MODERATE': 12.5, 'ACTIVE': 25.0}
BMI160_CURRENT = {6.25: 7, 12.5: 10, 25: 17, 50: 27}

V = 3.0
T_DAY = 86400
T_WINDOW = 1.5
I_MCU_ACTIVE = 2100
I_MCU_IDLE = 1.9
T_INFERENCE = 0.020
I_TX, I_RX = 5300, 5400
T_TX, T_RX = 0.0015, 0.001
I_OTHER = 2.05

from liveedge.data import extract_24_features_batch


def extract_features(X: np.ndarray) -> np.ndarray:
    """Extract 24 statistical features."""
    return extract_24_features_batch(X)


def resample_batch(X: np.ndarray, target_odr: float, source_odr: float = 50) -> np.ndarray:
    """Resample all windows to target ODR."""
    if target_odr >= source_odr:
        return X
    factor = int(source_odr / target_odr)
    return X[:, ::factor, :]


def compute_energy(odr_distribution: Dict[float, float],
                   inference_rate: float,
                   tx_interval: float = 60) -> Dict:
    """Compute energy consumption."""
    i_imu_avg = sum(BMI160_CURRENT.get(odr, 10) * frac
                   for odr, frac in odr_distribution.items())
    e_imu = i_imu_avg * V * T_DAY / 1000

    n_windows = T_DAY / T_WINDOW
    n_inferences = n_windows * inference_rate
    t_active = n_inferences * T_INFERENCE
    t_idle = T_DAY - t_active
    e_mcu = (t_active * I_MCU_ACTIVE + t_idle * I_MCU_IDLE) * V / 1000

    n_tx = T_DAY / tx_interval
    e_ble = n_tx * (T_TX * I_TX + T_RX * I_RX) * V / 1000
    e_other = I_OTHER * V * T_DAY / 1000

    e_total = e_imu + e_mcu + e_ble + e_other
    battery_energy = 220 * 3.0 * 3600 * 0.8
    battery_days = battery_energy / e_total

    return {
        'e_imu': round(e_imu, 1),
        'e_mcu': round(e_mcu, 1),
        'e_ble': round(e_ble, 1),
        'e_other': round(e_other, 1),
        'e_total': round(e_total, 1),
        'i_imu_avg': round(i_imu_avg, 2),
        'battery_days': round(battery_days, 1)
    }


def run_liveedge_simulation(X_test, y_test_behavior_str,
                            behavior_classifier, cluster_classifier,
                            scaler_behavior, scaler_cluster,
                            label_encoder_behavior, label_encoder_cluster,
                            k_stability=3):
    """Run LiveEdge simulation with FSM controller."""
    n_samples = len(X_test)

    # Pre-compute features at all ODRs
    print("    Pre-computing features at all ODRs...", end=" ", flush=True)
    features_cache = {}
    for odr in CLUSTER_ODR.values():
        X_resampled = resample_batch(X_test, odr)
        features = extract_features(X_resampled)
        features_cache[odr] = {
            'cluster': scaler_cluster.transform(features),
            'behavior': scaler_behavior.transform(features)
        }

    cluster_preds_cache = {}
    behavior_preds_cache = {}
    for odr, feats in features_cache.items():
        pred_indices = cluster_classifier.predict(feats['cluster'])
        cluster_preds_cache[odr] = label_encoder_cluster.inverse_transform(pred_indices)
        pred_indices = behavior_classifier.predict(feats['behavior'])
        behavior_preds_cache[odr] = label_encoder_behavior.inverse_transform(pred_indices)
    print("Done")

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

    accuracy = accuracy_score(y_test_behavior_str, predictions)
    macro_f1 = f1_score(y_test_behavior_str, predictions, average='macro')

    true_clusters = [BEHAVIOR_TO_CLUSTER[b] for b in y_test_behavior_str]
    cluster_accuracy = accuracy_score(true_clusters, pred_clusters)

    odr_dist = {odr: odr_history.count(odr) / len(odr_history) for odr in set(odr_history)}
    eff_odr = np.mean(odr_history)
    inference_rate = sum(inference_flags) / len(inference_flags)
    energy = compute_energy(odr_dist, inference_rate)

    classes = label_encoder_behavior.classes_
    per_class_f1 = {}
    f1_scores = f1_score(y_test_behavior_str, predictions, average=None, labels=classes)
    for i, cls in enumerate(classes):
        per_class_f1[cls] = round(f1_scores[i], 4)

    return {
        'accuracy': round(accuracy, 4),
        'macro_f1': round(macro_f1, 4),
        'cluster_accuracy': round(cluster_accuracy, 4),
        'per_class_f1': per_class_f1,
        'eff_odr': round(eff_odr, 2),
        'odr_distribution': {str(k): round(v, 4) for k, v in odr_dist.items()},
        'inference_rate': round(inference_rate, 4),
        'energy': energy,
        'k_stability': k_stability
    }


def main():
    print("=" * 70)
    print("LiveEdge Simulation")
    print("=" * 70)

    print("\n[1] Loading data...")
    X = np.load(DATA_DIR / '50hz' / 'windows.npy')[:, :, :3]
    y_labels = np.load(DATA_DIR / '50hz' / 'labels.npy', allow_pickle=True)

    valid_mask = np.array([b != 'Interacting' for b in y_labels])
    X = X[valid_mask]
    y = y_labels[valid_mask]

    print(f"  Total samples: {len(X)}")

    label_encoder_behavior = LabelEncoder()
    y_behavior_encoded = label_encoder_behavior.fit_transform(y)

    y_cluster_labels = np.array([BEHAVIOR_TO_CLUSTER[b] for b in y])
    label_encoder_cluster = LabelEncoder()
    y_cluster_encoded = label_encoder_cluster.fit_transform(y_cluster_labels)

    print(f"  Behaviors: {list(label_encoder_behavior.classes_)}")
    print(f"  Clusters: {list(label_encoder_cluster.classes_)}")

    print("\n[2] Running 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=False)

    all_results = {
        'baseline_50hz': [], 'baseline_25hz': [], 'baseline_12.5hz': [],
        'baseline_10hz': [], 'baseline_6.25hz': [],
        'liveedge_k3': [], 'cluster_accuracy_50hz': []
    }

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y_cluster_encoded)):
        print(f"\n  --- Fold {fold + 1}/5 ---")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train_behavior = y_behavior_encoded[train_idx]
        y_test_behavior = y_behavior_encoded[test_idx]
        y_train_cluster = y_cluster_encoded[train_idx]
        y_test_behavior_str = label_encoder_behavior.inverse_transform(y_test_behavior)

        X_train_features = extract_features(X_train)
        scaler_behavior = StandardScaler()
        X_train_scaled = scaler_behavior.fit_transform(X_train_features)

        behavior_classifier = RandomForestClassifier(
            n_estimators=100, max_depth=20,
            min_samples_split=5, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        )
        behavior_classifier.fit(X_train_scaled, y_train_behavior)

        scaler_cluster = StandardScaler()
        X_train_cluster_scaled = scaler_cluster.fit_transform(X_train_features)

        cluster_classifier = RandomForestClassifier(
            n_estimators=100, max_depth=15,
            class_weight='balanced',
            random_state=42, n_jobs=-1
        )
        cluster_classifier.fit(X_train_cluster_scaled, y_train_cluster)

        X_test_features = extract_features(X_test)
        X_test_cluster_scaled = scaler_cluster.transform(X_test_features)
        pred_cluster_indices = cluster_classifier.predict(X_test_cluster_scaled)
        pred_clusters = label_encoder_cluster.inverse_transform(pred_cluster_indices)
        true_clusters = [BEHAVIOR_TO_CLUSTER[b] for b in y_test_behavior_str]
        cluster_acc_50hz = accuracy_score(true_clusters, pred_clusters)
        all_results['cluster_accuracy_50hz'].append(cluster_acc_50hz)
        print(f"    Cluster Acc @50Hz: {cluster_acc_50hz*100:.2f}%")

        for odr, name in [(50, 'baseline_50hz'), (25, 'baseline_25hz'),
                          (12.5, 'baseline_12.5hz'), (10, 'baseline_10hz'),
                          (6.25, 'baseline_6.25hz')]:
            X_resampled = resample_batch(X_test, odr)
            features = extract_features(X_resampled)
            features_scaled = scaler_behavior.transform(features)
            pred_indices = behavior_classifier.predict(features_scaled)
            predictions = label_encoder_behavior.inverse_transform(pred_indices)
            acc = accuracy_score(y_test_behavior_str, predictions)
            all_results[name].append(acc)

        result = run_liveedge_simulation(
            X_test, y_test_behavior_str,
            behavior_classifier, cluster_classifier,
            scaler_behavior, scaler_cluster,
            label_encoder_behavior, label_encoder_cluster,
            k_stability=3
        )
        all_results['liveedge_k3'].append(result)
        print(f"    LiveEdge k=3: Acc={result['accuracy']*100:.2f}%, "
              f"Cluster={result['cluster_accuracy']*100:.2f}%, "
              f"ODR={result['eff_odr']:.1f}Hz, "
              f"Energy={result['energy']['e_total']:.0f}mJ")

    print("\n" + "=" * 70)
    print("RESULTS (5-fold CV)")
    print("=" * 70)

    summary = {
        'cluster_mapping': BEHAVIOR_TO_CLUSTER,
        'cluster_odr': CLUSTER_ODR,
        'baselines': {},
        'liveedge': {},
        'cluster_accuracy_at_50hz': {
            'mean': round(np.mean(all_results['cluster_accuracy_50hz']), 4),
            'std': round(np.std(all_results['cluster_accuracy_50hz']), 4)
        }
    }

    baseline_energy = {50: 15331, 25: 12739, 12.5: 10924, 10: 10665, 6.25: 10147}
    baseline_battery = {50: 124, 25: 149, 12.5: 174, 10: 178, 6.25: 187}

    print(f"\n{'Method':<20} | {'Accuracy':>12} | {'Energy':>10} | {'Battery':>10}")
    print("-" * 60)

    for odr, name in [(50, 'baseline_50hz'), (25, 'baseline_25hz'),
                      (12.5, 'baseline_12.5hz'), (10, 'baseline_10hz'),
                      (6.25, 'baseline_6.25hz')]:
        accs = all_results[name]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        summary['baselines'][name] = {
            'accuracy_mean': round(mean_acc, 4),
            'accuracy_std': round(std_acc, 4),
            'energy': baseline_energy[odr],
            'battery': baseline_battery[odr],
            'odr': odr
        }
        print(f"B_{odr}Hz{'':<12} | {mean_acc*100:>6.2f}% +/- {std_acc*100:.2f}% | "
              f"{baseline_energy[odr]:>8} mJ | {baseline_battery[odr]:>8}d")

    print("-" * 60)

    results = all_results['liveedge_k3']
    accs = [r['accuracy'] for r in results]
    cluster_accs = [r['cluster_accuracy'] for r in results]
    odrs = [r['eff_odr'] for r in results]
    energies = [r['energy']['e_total'] for r in results]
    batteries = [r['energy']['battery_days'] for r in results]
    inference_rates = [r['inference_rate'] for r in results]

    summary['liveedge'] = {
        'accuracy_mean': round(np.mean(accs), 4),
        'accuracy_std': round(np.std(accs), 4),
        'cluster_accuracy_mean': round(np.mean(cluster_accs), 4),
        'cluster_accuracy_std': round(np.std(cluster_accs), 4),
        'eff_odr_mean': round(np.mean(odrs), 2),
        'energy_mean': round(np.mean(energies), 1),
        'battery_mean': round(np.mean(batteries), 1),
        'inference_rate_mean': round(np.mean(inference_rates), 4),
        'savings_vs_50hz': round((1 - np.mean(energies) / 15331) * 100, 1),
        'k': 3
    }

    print(f"LiveEdge k=3{'':<8} | {np.mean(accs)*100:>6.2f}% +/- {np.std(accs)*100:.2f}% | "
          f"{np.mean(energies):>8.0f} mJ | {np.mean(batteries):>8.0f}d")

    print("\n" + "=" * 70)
    print("KEY METRICS")
    print("=" * 70)

    le = summary['liveedge']
    print(f"\n  Cluster Accuracy @50Hz: {summary['cluster_accuracy_at_50hz']['mean']*100:.2f}%")
    print(f"  Behavior Accuracy: {le['accuracy_mean']*100:.2f}%")
    print(f"  Runtime Cluster Accuracy: {le['cluster_accuracy_mean']*100:.2f}%")
    print(f"  Effective ODR: {le['eff_odr_mean']:.2f} Hz")
    print(f"  Inference Rate: {le['inference_rate_mean']*100:.1f}%")
    print(f"  Energy: {le['energy_mean']:.0f} mJ/day")
    print(f"  Battery Life: {le['battery_mean']:.0f} days")
    print(f"  Energy Savings: {le['savings_vs_50hz']:.1f}%")

    with open(OUTPUT_DIR / 'verified_results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Results saved to: {OUTPUT_DIR / 'verified_results.json'}]")
    return summary


if __name__ == "__main__":
    main()
