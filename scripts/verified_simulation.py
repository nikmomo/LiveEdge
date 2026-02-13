#!/usr/bin/env python3
"""LiveEdge simulation with FSM controller and k-hysteresis.

Architecture: Single behavior classifier → deterministic cluster mapping.
- Full model RF(100, d=20) for baseline evaluation
- Compressed model RF(30, d=8, balanced) for MCU deployment + FSM simulation
- Cluster assignments derived from behavior predictions via deterministic mapping
"""

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
    'Eating': 'STABLE',
    'Standing': 'ACTIVE',
    'Walking': 'MODERATE',
    'Drinking': 'ACTIVE',
}

CLUSTER_ODR = {'STABLE': 6.25, 'MODERATE': 12.5, 'ACTIVE': 25.0}
BMI160_CURRENT = {6.25: 7, 12.5: 10, 25: 17, 50: 27}

V = 3.0
T_DAY = 86400
T_WINDOW = 1.5
I_MCU_ACTIVE = 2100
I_MCU_IDLE = 1.9
T_INFERENCE = 0.001976  # seconds (cycle-accurate from Renode DWT: 126,469 cycles @ 64MHz)
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
                            behavior_classifier, scaler,
                            label_encoder_behavior,
                            k_stability=3):
    """Run LiveEdge simulation with single behavior classifier + deterministic cluster mapping."""
    n_samples = len(X_test)

    # Pre-compute features and behavior predictions at all ODRs
    print("    Pre-computing features at all ODRs...", end=" ", flush=True)
    behavior_preds_cache = {}
    cluster_preds_cache = {}
    for odr in CLUSTER_ODR.values():
        X_resampled = resample_batch(X_test, odr)
        features = extract_features(X_resampled)
        features_scaled = scaler.transform(features)
        pred_indices = behavior_classifier.predict(features_scaled)
        behavior_preds = label_encoder_behavior.inverse_transform(pred_indices)
        behavior_preds_cache[odr] = behavior_preds
        # Deterministic cluster mapping from behavior predictions
        cluster_preds_cache[odr] = np.array([BEHAVIOR_TO_CLUSTER[b] for b in behavior_preds])
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
    f1_scores_arr = f1_score(y_test_behavior_str, predictions, average=None, labels=classes)
    for i, cls in enumerate(classes):
        per_class_f1[cls] = round(f1_scores_arr[i], 4)

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
    print("LiveEdge Simulation (Single Classifier Architecture)")
    print("  Behavior classifier → deterministic cluster mapping")
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

    # Cluster labels only needed for stratification and evaluation
    y_cluster_labels = np.array([BEHAVIOR_TO_CLUSTER[b] for b in y])

    print(f"  Behaviors: {list(label_encoder_behavior.classes_)}")

    print("\n[2] Running 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=False)

    all_results = defaultdict(list)

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y_cluster_labels)):
        print(f"\n  --- Fold {fold + 1}/5 ---")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train_behavior = y_behavior_encoded[train_idx]
        y_test_behavior = y_behavior_encoded[test_idx]
        y_test_behavior_str = label_encoder_behavior.inverse_transform(y_test_behavior)

        X_train_features = extract_features(X_train)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_features)

        # Full behavior classifier (for baseline evaluation)
        behavior_clf_full = RandomForestClassifier(
            n_estimators=100, max_depth=20,
            min_samples_split=5, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        )
        behavior_clf_full.fit(X_train_scaled, y_train_behavior)

        # Compressed behavior classifier (for MCU deployment / FSM simulation)
        behavior_clf_compressed = RandomForestClassifier(
            n_estimators=30, max_depth=8,
            class_weight='balanced',
            random_state=42, n_jobs=-1
        )
        behavior_clf_compressed.fit(X_train_scaled, y_train_behavior)

        X_test_features = extract_features(X_test)
        X_test_scaled = scaler.transform(X_test_features)
        true_clusters = [BEHAVIOR_TO_CLUSTER[b] for b in y_test_behavior_str]

        # Cluster accuracy @50Hz via behavior→cluster mapping
        pred_bh_full = label_encoder_behavior.inverse_transform(
            behavior_clf_full.predict(X_test_scaled))
        cluster_acc_full = accuracy_score(
            true_clusters, [BEHAVIOR_TO_CLUSTER[b] for b in pred_bh_full])
        all_results['cluster_accuracy_50hz_full'].append(cluster_acc_full)

        pred_bh_comp = label_encoder_behavior.inverse_transform(
            behavior_clf_compressed.predict(X_test_scaled))
        cluster_acc_comp = accuracy_score(
            true_clusters, [BEHAVIOR_TO_CLUSTER[b] for b in pred_bh_comp])
        all_results['cluster_accuracy_50hz'].append(cluster_acc_comp)

        print(f"    Cluster Acc @50Hz (mapped): compressed={cluster_acc_comp*100:.2f}%, "
              f"full={cluster_acc_full*100:.2f}%")

        # Baselines at each ODR (using full behavior classifier)
        for odr, name in [(50, 'baseline_50hz'), (25, 'baseline_25hz'),
                          (12.5, 'baseline_12.5hz'), (10, 'baseline_10hz'),
                          (6.25, 'baseline_6.25hz')]:
            X_resampled = resample_batch(X_test, odr)
            features = extract_features(X_resampled)
            features_scaled = scaler.transform(features)
            # Behavior accuracy (full model for baselines)
            pred_indices = behavior_clf_full.predict(features_scaled)
            predictions = label_encoder_behavior.inverse_transform(pred_indices)
            acc = accuracy_score(y_test_behavior_str, predictions)
            all_results[name].append(acc)
            # Cluster accuracy via behavior→cluster mapping
            pred_clusters_mapped = [BEHAVIOR_TO_CLUSTER[b] for b in predictions]
            cluster_acc = accuracy_score(true_clusters, pred_clusters_mapped)
            all_results[f'{name}_cluster'].append(cluster_acc)

        # LiveEdge FSM with full behavior classifier (reference)
        result_full = run_liveedge_simulation(
            X_test, y_test_behavior_str,
            behavior_clf_full, scaler, label_encoder_behavior,
            k_stability=3
        )
        all_results['liveedge_k3'].append(result_full)
        print(f"    LiveEdge k=3 (full): Acc={result_full['accuracy']*100:.2f}%, "
              f"Cluster={result_full['cluster_accuracy']*100:.2f}%, "
              f"ODR={result_full['eff_odr']:.1f}Hz, "
              f"Energy={result_full['energy']['e_total']:.0f}mJ")

        # LiveEdge FSM with compressed behavior classifier (deployment model)
        result_compressed = run_liveedge_simulation(
            X_test, y_test_behavior_str,
            behavior_clf_compressed, scaler, label_encoder_behavior,
            k_stability=3
        )
        all_results['liveedge_k3_compressed'].append(result_compressed)
        print(f"    LiveEdge k=3 (compressed): Acc={result_compressed['accuracy']*100:.2f}%, "
              f"Cluster={result_compressed['cluster_accuracy']*100:.2f}%, "
              f"ODR={result_compressed['eff_odr']:.1f}Hz, "
              f"Energy={result_compressed['energy']['e_total']:.0f}mJ")

    print("\n" + "=" * 70)
    print("RESULTS (5-fold CV)")
    print("=" * 70)

    summary = {
        'architecture': 'single behavior classifier + deterministic cluster mapping',
        'cluster_mapping': BEHAVIOR_TO_CLUSTER,
        'cluster_odr': CLUSTER_ODR,
        'baselines': {},
        'liveedge': {},
        'cluster_accuracy_at_50hz': {
            'compressed_mapped': round(np.mean(all_results['cluster_accuracy_50hz']), 4),
            'full_mapped': round(np.mean(all_results['cluster_accuracy_50hz_full']), 4),
        }
    }

    # Compute baseline energy
    baseline_energy = {}
    baseline_battery = {}
    for odr in [50, 25, 12.5, 10, 6.25]:
        e = compute_energy({odr: 1.0}, inference_rate=1.0)
        baseline_energy[odr] = round(e['e_total'])
        baseline_battery[odr] = round(e['battery_days'])

    print(f"\n{'Method':<15} | {'Behav Acc':>12} | {'Cl(mapped)':>10} | {'Energy':>8} | {'Batt':>6}")
    print("-" * 65)

    for odr, name in [(50, 'baseline_50hz'), (25, 'baseline_25hz'),
                      (12.5, 'baseline_12.5hz'), (10, 'baseline_10hz'),
                      (6.25, 'baseline_6.25hz')]:
        accs = all_results[name]
        cluster_map = all_results[f'{name}_cluster']
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        mean_cl_map = np.mean(cluster_map)
        summary['baselines'][name] = {
            'accuracy_mean': round(mean_acc, 4),
            'accuracy_std': round(std_acc, 4),
            'cluster_accuracy_mapped': round(mean_cl_map, 4),
            'energy': baseline_energy[odr],
            'battery': baseline_battery[odr],
            'odr': odr
        }
        print(f"B_{odr}Hz{'':<7} | {mean_acc*100:>6.2f}% ±{std_acc*100:.2f}% | "
              f"{mean_cl_map*100:>8.2f}% | "
              f"{baseline_energy[odr]:>6} mJ | {baseline_battery[odr]:>4}d")

    print("-" * 65)

    # Full model LiveEdge results (reference)
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
        'eff_odr_mean': round(np.mean(odrs), 2),
        'energy_mean': round(np.mean(energies), 1),
        'battery_mean': round(np.mean(batteries), 1),
        'inference_rate_mean': round(np.mean(inference_rates), 4),
        'savings_vs_50hz': round((1 - np.mean(energies) / baseline_energy[50]) * 100, 1),
        'k': 3
    }

    print(f"LiveEdge (full)  | {np.mean(accs)*100:>6.2f}% ±{np.std(accs)*100:.2f}% | "
          f"{np.mean(cluster_accs)*100:>8.2f}% | "
          f"{np.mean(energies):>6.0f} mJ | {np.mean(batteries):>4.0f}d")

    # Compressed model results (for main paper results)
    results_comp = all_results['liveedge_k3_compressed']
    comp_accs = [r['accuracy'] for r in results_comp]
    comp_cluster = [r['cluster_accuracy'] for r in results_comp]
    comp_odrs = [r['eff_odr'] for r in results_comp]
    comp_energies = [r['energy']['e_total'] for r in results_comp]
    comp_batteries = [r['energy']['battery_days'] for r in results_comp]
    comp_inf_rates = [r['inference_rate'] for r in results_comp]

    summary['liveedge_compressed'] = {
        'accuracy_mean': round(np.mean(comp_accs), 4),
        'accuracy_std': round(np.std(comp_accs), 4),
        'cluster_accuracy_mean': round(np.mean(comp_cluster), 4),
        'eff_odr_mean': round(np.mean(comp_odrs), 2),
        'energy_mean': round(np.mean(comp_energies), 1),
        'battery_mean': round(np.mean(comp_batteries), 1),
        'inference_rate_mean': round(np.mean(comp_inf_rates), 4),
        'savings_vs_50hz': round((1 - np.mean(comp_energies) / baseline_energy[50]) * 100, 1),
    }
    print(f"LiveEdge (comp)  | {np.mean(comp_accs)*100:>6.2f}% ±{np.std(comp_accs)*100:.2f}% | "
          f"{np.mean(comp_cluster)*100:>8.2f}% | "
          f"{np.mean(comp_energies):>6.0f} mJ | {np.mean(comp_batteries):>4.0f}d")

    # ========================================================================
    # Ablation Analysis (Table 6)
    # ========================================================================
    print("\n" + "=" * 70)
    print("ABLATION STUDY (Table 6)")
    print("=" * 70)

    E_OVERHEAD = 589  # mJ/day (BLE + RTC + regulator)
    E_IMU_50 = 27 * 3.0 * 86400 / 1000  # 6998 mJ/day

    n_windows_day = T_DAY / T_WINDOW
    t_active_baseline = n_windows_day * T_INFERENCE
    t_idle_baseline = T_DAY - t_active_baseline
    E_MCU_BASELINE = (t_active_baseline * I_MCU_ACTIVE + t_idle_baseline * I_MCU_IDLE) * V / 1000

    a0_total = round(E_IMU_50 + E_MCU_BASELINE + E_OVERHEAD)

    # LiveEdge ODR distribution (from full model FSM, matches Table 6 caption)
    le_odr_dists = [r['odr_distribution'] for r in results]
    avg_odr_dist = {}
    for dist in le_odr_dists:
        for odr_str, frac in dist.items():
            odr_val = float(odr_str)
            avg_odr_dist[odr_val] = avg_odr_dist.get(odr_val, 0) + frac / len(le_odr_dists)

    # A1: Adaptive ODR only (100% inference)
    i_imu_avg = sum(BMI160_CURRENT.get(odr, 10) * frac for odr, frac in avg_odr_dist.items())
    a1_e_imu = i_imu_avg * 3.0 * 86400 / 1000
    a1_e_mcu = E_MCU_BASELINE
    a1_total = a1_e_imu + a1_e_mcu + E_OVERHEAD

    # A2: Fixed 50Hz + stability-based inference
    avg_inf_rate = np.mean(inference_rates)
    n_windows = 86400 / 1.5
    n_inf = n_windows * avg_inf_rate
    t_active = n_inf * T_INFERENCE
    t_idle = 86400 - t_active
    a2_e_mcu = (t_active * I_MCU_ACTIVE + t_idle * I_MCU_IDLE) * V / 1000
    a2_total = E_IMU_50 + a2_e_mcu + E_OVERHEAD

    # A3: LiveEdge (Adaptive ODR + inference)
    a3_e_imu = a1_e_imu
    a3_e_mcu = a2_e_mcu
    a3_total = a3_e_imu + a3_e_mcu + E_OVERHEAD

    print(f"\n{'Config':<6} | {'E_IMU':>8} | {'E_MCU':>8} | {'E_total':>8} | {'Savings':>8}")
    print("-" * 50)
    print(f"A0     | {E_IMU_50:>8.0f} | {E_MCU_BASELINE:>8.0f} | {a0_total:>8.0f} |      ---")
    print(f"A1     | {a1_e_imu:>8.0f} | {a1_e_mcu:>8.0f} | {a1_total:>8.0f} | {(1-a1_total/a0_total)*100:>6.1f}%")
    print(f"A2     | {E_IMU_50:>8.0f} | {a2_e_mcu:>8.0f} | {a2_total:>8.0f} | {(1-a2_total/a0_total)*100:>6.1f}%")
    print(f"A3     | {a3_e_imu:>8.0f} | {a3_e_mcu:>8.0f} | {a3_total:>8.0f} | {(1-a3_total/a0_total)*100:>6.1f}%")

    summary['ablation'] = {
        'A0': {'e_imu': round(E_IMU_50), 'e_mcu': round(E_MCU_BASELINE), 'e_total': a0_total},
        'A1': {'e_imu': round(a1_e_imu), 'e_mcu': round(a1_e_mcu), 'e_total': round(a1_total)},
        'A2': {'e_imu': round(E_IMU_50), 'e_mcu': round(a2_e_mcu), 'e_total': round(a2_total)},
        'A3': {'e_imu': round(a3_e_imu), 'e_mcu': round(a3_e_mcu), 'e_total': round(a3_total)},
    }

    # ========================================================================
    # Key Metrics Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("KEY METRICS")
    print("=" * 70)

    le = summary['liveedge']
    lc = summary['liveedge_compressed']
    print(f"\n  Cluster Acc @50Hz (compressed, mapped): "
          f"{summary['cluster_accuracy_at_50hz']['compressed_mapped']*100:.2f}%")
    print(f"  Cluster Acc @50Hz (full, mapped): "
          f"{summary['cluster_accuracy_at_50hz']['full_mapped']*100:.2f}%")

    print(f"\n  --- Compressed Model (MCU Deployment, Table 10/11) ---")
    print(f"  Behavior Accuracy: {lc['accuracy_mean']*100:.2f}% ± {lc.get('accuracy_std',0)*100:.2f}%")
    print(f"  Runtime Cluster Accuracy: {lc['cluster_accuracy_mean']*100:.2f}%")
    print(f"  Effective ODR: {lc['eff_odr_mean']:.2f} Hz")
    print(f"  Inference Rate: {lc['inference_rate_mean']*100:.1f}%")
    print(f"  Energy: {lc['energy_mean']:.0f} mJ/day")
    print(f"  Battery Life: {lc['battery_mean']:.0f} days")
    print(f"  Energy Savings: {lc['savings_vs_50hz']:.1f}%")

    print(f"\n  --- Full Model (Reference) ---")
    print(f"  Behavior Accuracy: {le['accuracy_mean']*100:.2f}% ± {le['accuracy_std']*100:.2f}%")
    print(f"  Runtime Cluster Accuracy: {le['cluster_accuracy_mean']*100:.2f}%")
    print(f"  Effective ODR: {le['eff_odr_mean']:.2f} Hz")
    print(f"  Energy: {le['energy_mean']:.0f} mJ/day")
    print(f"  Energy Savings: {le['savings_vs_50hz']:.1f}%")

    print("\n" + "=" * 70)
    print("ENERGY MODEL PARAMETERS")
    print("=" * 70)
    print(f"  T_INFERENCE = {T_INFERENCE*1000:.3f} ms (cycle-accurate from Renode DWT)")
    print(f"  Baseline 50Hz energy: {baseline_energy[50]} mJ/day, {baseline_battery[50]} days")

    with open(OUTPUT_DIR / 'verified_results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Results saved to: {OUTPUT_DIR / 'verified_results.json'}]")
    return summary


if __name__ == "__main__":
    main()
