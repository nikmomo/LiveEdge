#!/usr/bin/env python3
"""
Related Work Comparison: LiveEdge vs. Signal-Driven Baselines

Implements and compares:
1. Fixed-rate baselines (B_50Hz, B_25Hz, B_12.5Hz, B_6.25Hz)
2. Signal-driven adaptive ODR (Jeong 2019, Cheng 2018 style)
3. Variance-threshold inference skipping
4. Confidence-based inference skipping
5. LiveEdge (behavior-driven dual mechanism)

References:
- Jeong 2019: Signal-variance triggered sampling, 19-day battery
- Cheng 2018: MDP-based ODR selection
- SmartAPM 2025: DRL component control
- Algabroun 2025: Parametric sampling for slow-varying signals
"""

import sys
from pathlib import Path
import numpy as np
import json
from typing import Dict, List, Tuple
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

from liveedge.data import extract_24_features_batch

# Configuration
DATA_DIR = Path("L:/GitHub/LiveEdge/data/processed")
OUTPUT_DIR = Path("L:/GitHub/LiveEdge/outputs/related_work_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cluster mapping (verified)
BEHAVIOR_TO_CLUSTER = {
    'Lying': 'STABLE',
    'Eating': 'STABLE',
    'Standing': 'ACTIVE',
    'Walking': 'MODERATE',
    'Drinking': 'ACTIVE',
}

CLUSTER_ODR = {'STABLE': 6.25, 'MODERATE': 12.5, 'ACTIVE': 25.0}

# Energy model
BMI160_CURRENT = {6.25: 7, 12.5: 10, 25: 17, 50: 27}  # uA
V = 3.0
T_DAY = 86400
T_WINDOW = 1.5
I_MCU_ACTIVE = 2100  # uA
I_MCU_IDLE = 1.9     # uA
T_INFERENCE = 0.001976  # seconds (cycle-accurate from Renode DWT: 126,469 cycles @ 64MHz)
I_TX, I_RX = 5300, 5400  # uA
T_TX, T_RX = 0.0015, 0.001  # seconds
I_OTHER = 2.05  # uA
BATTERY_CAPACITY = 220 * 3.0 * 3600 * 0.8  # mJ (CR2032, 80% usable)


def extract_features(X: np.ndarray) -> np.ndarray:
    """Extract 24 statistical features."""
    return extract_24_features_batch(X)


def resample_batch(X: np.ndarray, target_odr: float, source_odr: float = 50) -> np.ndarray:
    """Resample windows to target ODR."""
    if target_odr >= source_odr:
        return X
    factor = int(source_odr / target_odr)
    return X[:, ::factor, :]


def compute_variance(window: np.ndarray) -> float:
    """Compute motion variance (magnitude variance)."""
    magnitude = np.sqrt(np.sum(window ** 2, axis=1))
    return np.var(magnitude)


def compute_energy(odr_distribution: Dict[float, float],
                   inference_rate: float,
                   tx_interval: float = 60) -> Dict:
    """Compute daily energy consumption."""
    # IMU energy
    i_imu_avg = sum(BMI160_CURRENT.get(odr, 10) * frac
                   for odr, frac in odr_distribution.items())
    e_imu = i_imu_avg * V * T_DAY / 1000

    # MCU energy
    n_windows = T_DAY / T_WINDOW
    n_inferences = n_windows * inference_rate
    t_active = n_inferences * T_INFERENCE
    t_idle = T_DAY - t_active
    e_mcu = (t_active * I_MCU_ACTIVE + t_idle * I_MCU_IDLE) * V / 1000

    # BLE energy
    n_tx = T_DAY / tx_interval
    e_ble = n_tx * (T_TX * I_TX + T_RX * I_RX) * V / 1000

    # Other
    e_other = I_OTHER * V * T_DAY / 1000

    e_total = e_imu + e_mcu + e_ble + e_other
    battery_days = BATTERY_CAPACITY / e_total

    return {
        'e_imu': round(e_imu, 1),
        'e_mcu': round(e_mcu, 1),
        'e_ble': round(e_ble, 1),
        'e_other': round(e_other, 1),
        'e_total': round(e_total, 1),
        'battery_days': round(battery_days, 1)
    }


# =============================================================================
# BASELINE METHODS
# =============================================================================

def fixed_rate_baseline(X_test, y_test_behavior_str, classifier, scaler,
                        label_encoder, odr: float) -> Dict:
    """
    Fixed-rate baseline: constant ODR, 100% inference rate.
    This represents traditional approaches like Versluijs 2023.
    """
    X_resampled = resample_batch(X_test, odr)
    features = extract_features(X_resampled)
    features_scaled = scaler.transform(features)
    pred_indices = classifier.predict(features_scaled)
    predictions = label_encoder.inverse_transform(pred_indices)

    accuracy = accuracy_score(y_test_behavior_str, predictions)
    macro_f1 = f1_score(y_test_behavior_str, predictions, average='macro')

    # Energy: fixed ODR, 100% inference
    odr_dist = {odr: 1.0}
    energy = compute_energy(odr_dist, inference_rate=1.0)

    return {
        'method': f'Fixed_{odr}Hz',
        'accuracy': round(accuracy, 4),
        'macro_f1': round(macro_f1, 4),
        'eff_odr': odr,
        'inference_rate': 1.0,
        'energy': energy
    }


def variance_threshold_adaptive(X_test, y_test_behavior_str,
                                 classifier, scaler, label_encoder,
                                 variance_threshold: float = 0.1,
                                 low_odr: float = 6.25,
                                 high_odr: float = 25.0) -> Dict:
    """
    Signal-driven adaptive ODR (Jeong 2019 / Cheng 2018 style).

    Uses motion variance to switch between low and high ODR:
    - variance < threshold → use low_odr (static behavior assumed)
    - variance >= threshold → use high_odr (active behavior)

    This is a signal-driven approach that doesn't use behavioral semantics.
    """
    n_samples = len(X_test)

    # Pre-compute features at both ODRs
    features_low = extract_features(resample_batch(X_test, low_odr))
    features_high = extract_features(resample_batch(X_test, high_odr))
    features_low_scaled = scaler.transform(features_low)
    features_high_scaled = scaler.transform(features_high)

    # Pre-compute predictions
    preds_low = label_encoder.inverse_transform(classifier.predict(features_low_scaled))
    preds_high = label_encoder.inverse_transform(classifier.predict(features_high_scaled))

    # Compute variance for each window (using high-rate data for accurate variance)
    variances = np.array([compute_variance(X_test[i]) for i in range(n_samples)])

    # Adaptive ODR selection based on variance
    predictions = []
    odr_history = []

    for i in range(n_samples):
        if variances[i] < variance_threshold:
            predictions.append(preds_low[i])
            odr_history.append(low_odr)
        else:
            predictions.append(preds_high[i])
            odr_history.append(high_odr)

    accuracy = accuracy_score(y_test_behavior_str, predictions)
    macro_f1 = f1_score(y_test_behavior_str, predictions, average='macro')
    eff_odr = np.mean(odr_history)

    # ODR distribution
    odr_dist = {odr: odr_history.count(odr) / len(odr_history)
                for odr in set(odr_history)}

    energy = compute_energy(odr_dist, inference_rate=1.0)

    return {
        'method': f'Variance_Adaptive(tau={variance_threshold})',
        'accuracy': round(accuracy, 4),
        'macro_f1': round(macro_f1, 4),
        'eff_odr': round(eff_odr, 2),
        'inference_rate': 1.0,
        'odr_distribution': {str(k): round(v, 4) for k, v in odr_dist.items()},
        'variance_threshold': variance_threshold,
        'energy': energy
    }


def variance_inference_skip(X_test, y_test_behavior_str,
                            classifier, scaler, label_encoder,
                            variance_threshold: float = 0.05,
                            odr: float = 25.0) -> Dict:
    """
    Signal-driven inference skipping (Jeong 2019 style).

    Fixed ODR, but skip inference when variance is below threshold:
    - variance < threshold → reuse last prediction (skip inference)
    - variance >= threshold → run inference

    This saves MCU energy but not IMU energy.
    """
    n_samples = len(X_test)

    # Pre-compute features and predictions
    X_resampled = resample_batch(X_test, odr)
    features = extract_features(X_resampled)
    features_scaled = scaler.transform(features)
    all_preds = label_encoder.inverse_transform(classifier.predict(features_scaled))

    # Compute variance for each window
    variances = np.array([compute_variance(X_test[i]) for i in range(n_samples)])

    # Inference skipping based on variance
    predictions = []
    inference_flags = []
    last_pred = all_preds[0]  # Initialize with first prediction

    for i in range(n_samples):
        if variances[i] < variance_threshold:
            # Skip inference, reuse last prediction
            predictions.append(last_pred)
            inference_flags.append(False)
        else:
            # Run inference
            predictions.append(all_preds[i])
            last_pred = all_preds[i]
            inference_flags.append(True)

    accuracy = accuracy_score(y_test_behavior_str, predictions)
    macro_f1 = f1_score(y_test_behavior_str, predictions, average='macro')
    inference_rate = sum(inference_flags) / len(inference_flags)

    odr_dist = {odr: 1.0}
    energy = compute_energy(odr_dist, inference_rate)

    return {
        'method': f'Variance_Skip(tau={variance_threshold})',
        'accuracy': round(accuracy, 4),
        'macro_f1': round(macro_f1, 4),
        'eff_odr': odr,
        'inference_rate': round(inference_rate, 4),
        'variance_threshold': variance_threshold,
        'energy': energy
    }


def confidence_inference_skip(X_test, y_test_behavior_str,
                               classifier, scaler, label_encoder,
                               confidence_threshold: float = 0.8,
                               odr: float = 25.0) -> Dict:
    """
    Confidence-based inference skipping.

    Skip inference when classifier confidence exceeds threshold:
    - confidence >= threshold → reuse last prediction
    - confidence < threshold → run inference

    This is a model-based approach (vs signal-driven).
    """
    n_samples = len(X_test)

    # Pre-compute features and probabilities
    X_resampled = resample_batch(X_test, odr)
    features = extract_features(X_resampled)
    features_scaled = scaler.transform(features)
    all_probs = classifier.predict_proba(features_scaled)
    all_preds = label_encoder.inverse_transform(classifier.predict(features_scaled))

    # Inference decision based on confidence
    predictions = []
    inference_flags = []
    last_pred = all_preds[0]
    last_confidence = np.max(all_probs[0])

    for i in range(n_samples):
        current_confidence = np.max(all_probs[i])

        if last_confidence >= confidence_threshold:
            # High confidence in last prediction → skip
            predictions.append(last_pred)
            inference_flags.append(False)
        else:
            # Low confidence → run inference
            predictions.append(all_preds[i])
            last_pred = all_preds[i]
            inference_flags.append(True)

        last_confidence = current_confidence

    accuracy = accuracy_score(y_test_behavior_str, predictions)
    macro_f1 = f1_score(y_test_behavior_str, predictions, average='macro')
    inference_rate = sum(inference_flags) / len(inference_flags)

    odr_dist = {odr: 1.0}
    energy = compute_energy(odr_dist, inference_rate)

    return {
        'method': f'Confidence_Skip(gamma={confidence_threshold})',
        'accuracy': round(accuracy, 4),
        'macro_f1': round(macro_f1, 4),
        'eff_odr': odr,
        'inference_rate': round(inference_rate, 4),
        'confidence_threshold': confidence_threshold,
        'energy': energy
    }


def liveedge_dual_mechanism(X_test, y_test_behavior_str,
                            behavior_classifier, scaler,
                            label_encoder_behavior,
                            k_stability: int = 3) -> Dict:
    """
    LiveEdge: Behavior-driven dual-mechanism optimization.

    Single behavior classifier + deterministic cluster mapping.
    1. ODR Optimization: Adjust ODR based on predicted behavioral cluster
    2. Inference Skipping: Skip inference in stable states

    Key difference from signal-driven: uses behavioral SEMANTICS, not raw signal.
    """
    n_samples = len(X_test)

    # Pre-compute features and predictions at all ODRs
    cluster_preds_cache = {}
    behavior_preds_cache = {}
    for odr in CLUSTER_ODR.values():
        X_resampled = resample_batch(X_test, odr)
        features = extract_features(X_resampled)
        features_scaled = scaler.transform(features)
        pred_indices = behavior_classifier.predict(features_scaled)
        behavior_preds = label_encoder_behavior.inverse_transform(pred_indices)
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

        # Track consecutive same predictions
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

    odr_dist = {odr: odr_history.count(odr) / len(odr_history)
                for odr in set(odr_history)}
    eff_odr = np.mean(odr_history)
    inference_rate = sum(inference_flags) / len(inference_flags)

    energy = compute_energy(odr_dist, inference_rate)

    return {
        'method': f'LiveEdge(k={k_stability})',
        'accuracy': round(accuracy, 4),
        'macro_f1': round(macro_f1, 4),
        'cluster_accuracy': round(cluster_accuracy, 4),
        'eff_odr': round(eff_odr, 2),
        'inference_rate': round(inference_rate, 4),
        'odr_distribution': {str(k): round(v, 4) for k, v in odr_dist.items()},
        'k_stability': k_stability,
        'energy': energy
    }


def main():
    print("=" * 80)
    print("RELATED WORK COMPARISON")
    print("LiveEdge vs. Signal-Driven and Fixed-Rate Baselines")
    print("=" * 80)

    # Load data
    print("\n[1] Loading data...")
    X = np.load(DATA_DIR / '50hz' / 'windows.npy')[:, :, :3]
    y_labels = np.load(DATA_DIR / '50hz' / 'labels.npy', allow_pickle=True)

    # Filter out Interacting
    valid_mask = np.array([b != 'Interacting' for b in y_labels])
    X = X[valid_mask]
    y = y_labels[valid_mask]
    print(f"  Samples: {len(X)}")

    # Encode labels
    le_behavior = LabelEncoder()
    y_behavior = le_behavior.fit_transform(y)

    y_cluster = np.array([BEHAVIOR_TO_CLUSTER[b] for b in y])
    le_cluster = LabelEncoder()
    y_cluster_encoded = le_cluster.fit_transform(y_cluster)

    # 5-fold CV (no shuffle to preserve temporal order for FSM simulation)
    print("\n[2] Running 5-fold cross-validation (temporal order)...")
    cv = StratifiedKFold(n_splits=5, shuffle=False)

    all_results = defaultdict(list)

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y_cluster_encoded)):
        print(f"\n  --- Fold {fold + 1}/5 ---")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train_behavior = y_behavior[train_idx]
        y_test_behavior_str = le_behavior.inverse_transform(y_behavior[test_idx])

        # Train single behavior classifier
        X_train_features = extract_features(X_train)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_features)
        behavior_classifier = RandomForestClassifier(
            n_estimators=100, max_depth=20,
            min_samples_split=5, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        )
        behavior_classifier.fit(X_train_scaled, y_train_behavior)

        # === FIXED-RATE BASELINES ===
        for odr in [50, 25, 12.5, 6.25]:
            result = fixed_rate_baseline(X_test, y_test_behavior_str,
                                         behavior_classifier, scaler,
                                         le_behavior, odr)
            all_results[result['method']].append(result)

        # === SIGNAL-DRIVEN ADAPTIVE ODR ===
        for tau in [0.05, 0.1, 0.2]:
            result = variance_threshold_adaptive(X_test, y_test_behavior_str,
                                                  behavior_classifier, scaler,
                                                  le_behavior, variance_threshold=tau)
            all_results[result['method']].append(result)

        # === SIGNAL-DRIVEN INFERENCE SKIP ===
        for tau in [0.02, 0.05, 0.1]:
            result = variance_inference_skip(X_test, y_test_behavior_str,
                                              behavior_classifier, scaler,
                                              le_behavior, variance_threshold=tau)
            all_results[result['method']].append(result)

        # === CONFIDENCE-BASED INFERENCE SKIP ===
        for gamma in [0.7, 0.8, 0.9]:
            result = confidence_inference_skip(X_test, y_test_behavior_str,
                                                behavior_classifier, scaler,
                                                le_behavior, confidence_threshold=gamma)
            all_results[result['method']].append(result)

        # === LIVEEDGE ===
        result = liveedge_dual_mechanism(X_test, y_test_behavior_str,
                                          behavior_classifier, scaler,
                                          le_behavior, k_stability=3)
        all_results[result['method']].append(result)

        print(f"    LiveEdge: Acc={result['accuracy']*100:.1f}%, "
              f"Energy={result['energy']['e_total']:.0f}mJ, "
              f"Battery={result['energy']['battery_days']:.0f}d")

    # Aggregate and summarize
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY (5-fold CV mean)")
    print("=" * 80)

    summary = {}
    print(f"\n{'Method':<35} | {'Acc':>6} | {'F1':>6} | {'ODR':>6} | {'Inf%':>6} | "
          f"{'Energy':>8} | {'Battery':>8}")
    print("-" * 95)

    # Define method order for nice output
    method_order = [
        'Fixed_50Hz', 'Fixed_25Hz', 'Fixed_12.5Hz', 'Fixed_6.25Hz',
        'Variance_Adaptive(tau=0.05)', 'Variance_Adaptive(tau=0.1)', 'Variance_Adaptive(tau=0.2)',
        'Variance_Skip(tau=0.02)', 'Variance_Skip(tau=0.05)', 'Variance_Skip(tau=0.1)',
        'Confidence_Skip(gamma=0.7)', 'Confidence_Skip(gamma=0.8)', 'Confidence_Skip(gamma=0.9)',
        'LiveEdge(k=3)'
    ]

    for method in method_order:
        if method not in all_results:
            continue

        results = all_results[method]
        acc = np.mean([r['accuracy'] for r in results])
        f1 = np.mean([r['macro_f1'] for r in results])
        odr = np.mean([r['eff_odr'] for r in results])
        inf_rate = np.mean([r['inference_rate'] for r in results])
        energy = np.mean([r['energy']['e_total'] for r in results])
        battery = np.mean([r['energy']['battery_days'] for r in results])

        summary[method] = {
            'accuracy': round(acc, 4),
            'macro_f1': round(f1, 4),
            'eff_odr': round(odr, 2),
            'inference_rate': round(inf_rate, 4),
            'energy_mJ': round(energy, 1),
            'battery_days': round(battery, 1)
        }

        print(f"{method:<35} | {acc*100:>5.1f}% | {f1*100:>5.1f}% | {odr:>5.1f} | "
              f"{inf_rate*100:>5.1f}% | {energy:>7.0f} | {battery:>7.0f}d")

    # Key comparisons for paper
    print("\n" + "=" * 80)
    print("KEY COMPARISONS FOR PAPER")
    print("=" * 80)

    liveedge = summary.get('LiveEdge(k=3)', {})
    fixed_25 = summary.get('Fixed_25Hz', {})
    var_adapt = summary.get('Variance_Adaptive(tau=0.1)', {})
    var_skip = summary.get('Variance_Skip(tau=0.05)', {})
    conf_skip = summary.get('Confidence_Skip(gamma=0.8)', {})

    print(f"\nLiveEdge vs Fixed 25Hz (similar accuracy):")
    if liveedge and fixed_25:
        acc_diff = (liveedge['accuracy'] - fixed_25['accuracy']) * 100
        energy_save = (1 - liveedge['energy_mJ'] / fixed_25['energy_mJ']) * 100
        battery_gain = liveedge['battery_days'] - fixed_25['battery_days']
        print(f"  Accuracy: {liveedge['accuracy']*100:.1f}% vs {fixed_25['accuracy']*100:.1f}% ({acc_diff:+.1f}%)")
        print(f"  Energy: {liveedge['energy_mJ']:.0f} vs {fixed_25['energy_mJ']:.0f} mJ ({energy_save:.1f}% savings)")
        print(f"  Battery: {liveedge['battery_days']:.0f} vs {fixed_25['battery_days']:.0f} days (+{battery_gain:.0f} days)")

    print(f"\nLiveEdge vs Best Signal-Driven Adaptive:")
    if liveedge and var_adapt:
        acc_diff = (liveedge['accuracy'] - var_adapt['accuracy']) * 100
        energy_save = (1 - liveedge['energy_mJ'] / var_adapt['energy_mJ']) * 100
        print(f"  Accuracy: {liveedge['accuracy']*100:.1f}% vs {var_adapt['accuracy']*100:.1f}% ({acc_diff:+.1f}%)")
        print(f"  Energy: {liveedge['energy_mJ']:.0f} vs {var_adapt['energy_mJ']:.0f} mJ ({energy_save:.1f}% savings)")
        print(f"  Battery: {liveedge['battery_days']:.0f} vs {var_adapt['battery_days']:.0f} days")

    print(f"\nLiveEdge vs Variance-based Inference Skip:")
    if liveedge and var_skip:
        acc_diff = (liveedge['accuracy'] - var_skip['accuracy']) * 100
        energy_save = (1 - liveedge['energy_mJ'] / var_skip['energy_mJ']) * 100
        print(f"  Accuracy: {liveedge['accuracy']*100:.1f}% vs {var_skip['accuracy']*100:.1f}% ({acc_diff:+.1f}%)")
        print(f"  Energy: {liveedge['energy_mJ']:.0f} vs {var_skip['energy_mJ']:.0f} mJ ({energy_save:.1f}% savings)")
        print(f"  Inference Rate: {liveedge['inference_rate']*100:.1f}% vs {var_skip['inference_rate']*100:.1f}%")

    # Save detailed results
    output_data = {
        'summary': summary,
        'raw_results': {k: [dict(r) for r in v] for k, v in all_results.items()},
        'energy_model': {
            'V': V,
            'I_MCU_ACTIVE_uA': I_MCU_ACTIVE,
            'I_MCU_IDLE_uA': I_MCU_IDLE,
            'T_INFERENCE_s': T_INFERENCE,
            'BMI160_CURRENT_uA': BMI160_CURRENT
        },
        'methods_description': {
            'Fixed_*Hz': 'Traditional fixed-rate sampling (Versluijs 2023 style)',
            'Variance_Adaptive': 'Signal-driven ODR selection based on motion variance (Jeong 2019, Cheng 2018 style)',
            'Variance_Skip': 'Signal-driven inference skipping based on motion variance (Jeong 2019 style)',
            'Confidence_Skip': 'Model confidence-based inference skipping',
            'LiveEdge': 'Behavior-driven dual mechanism (ODR + inference) optimization'
        }
    }

    with open(OUTPUT_DIR / 'related_work_comparison.json', 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n[Results saved to: {OUTPUT_DIR / 'related_work_comparison.json'}]")

    # Generate LaTeX table for paper
    print("\n" + "=" * 80)
    print("LATEX TABLE FOR PAPER")
    print("=" * 80)
    print("""
\\begin{table}[t]
\\centering
\\caption{Comparison with Signal-Driven and Fixed-Rate Baselines}
\\label{tab:related_work_comparison}
\\begin{tabular}{lcccccc}
\\toprule
Method & Type & Acc (\\%) & ODR (Hz) & Inf (\\%) & Energy & Battery \\\\
\\midrule""")

    for method in ['Fixed_50Hz', 'Fixed_25Hz', 'Fixed_6.25Hz',
                   'Variance_Adaptive(tau=0.1)', 'Variance_Skip(tau=0.05)',
                   'Confidence_Skip(gamma=0.8)', 'LiveEdge(k=3)']:
        if method not in summary:
            continue
        s = summary[method]
        method_type = 'Fixed' if 'Fixed' in method else ('Signal' if 'Variance' in method else ('Model' if 'Confidence' in method else 'Behavior'))
        short_name = method.replace('_', '\\_').replace('(', ' (')
        print(f"{short_name:<30} & {method_type} & {s['accuracy']*100:.1f} & {s['eff_odr']:.1f} & "
              f"{s['inference_rate']*100:.0f} & {s['energy_mJ']:.0f} & {s['battery_days']:.0f}d \\\\")

    print("""\\bottomrule
\\end{tabular}
\\end{table}
""")

    return summary


if __name__ == "__main__":
    main()
