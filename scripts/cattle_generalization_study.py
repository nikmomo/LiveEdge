#!/usr/bin/env python3
"""Cattle generalization study for cross-species validation."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root / 'src'))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

from liveedge.data import extract_24_features_batch

# Configuration

RAW_DATA_PATH = Path("L:/GitHub/LiveEdge/data/raw/imu_cow_dataset_fall-2022.csv")
OUTPUT_DIR = Path("L:/GitHub/LiveEdge/outputs/cattle_generalization")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLING_RATE = 50  # Hz
WINDOW_SIZE = 1.5  # seconds
WINDOW_SAMPLES = int(SAMPLING_RATE * WINDOW_SIZE)  # 75 samples
OVERLAP = 0.5  # 50% overlap

# Behavior mapping for cattle (14 classes -> 5 core + mapping to 3 clusters)
# Core behaviors analogous to pig study
CATTLE_BEHAVIOR_MAP = {
    # STABLE cluster (low activity) -> 6.25 Hz
    'lying': 'Lying',
    'lying to standing': 'Lying',  # Transition, treat as lying
    'staning to lying': 'Lying',   # Typo in data, transition
    'chewing': 'Eating',  # Cattle equivalent of pig eating
    'grazing': 'Eating',  # Cattle-specific feeding behavior
    'grazing and standing': 'Eating',  # Compound: treat as eating
    'licking': 'Eating',  # Oral behavior, treat as eating

    # MODERATE cluster (locomotion) -> 12.5 Hz
    'walking': 'Walking',
    'grazing and walking': 'Walking',  # Compound: treat as walking
    'grazing to walking or walking to grazing': 'Walking',  # Transition

    # ACTIVE cluster (postural/rapid) -> 25 Hz
    'standing': 'Standing',
    'drinking': 'Drinking',
    'cow interactions': 'Interacting',
    'running': 'Running',  # High activity
}

# 3-Cluster configuration
CLUSTER_CONFIG = {
    'STABLE': {'behaviors': ['Lying', 'Eating'], 'odr': 6.25},
    'MODERATE': {'behaviors': ['Walking'], 'odr': 12.5},
    'ACTIVE': {'behaviors': ['Standing', 'Drinking', 'Interacting', 'Running'], 'odr': 25.0}
}

# For comparison with pig: use 5-class (excluding Interacting/Running due to rarity)
BEHAVIORS_5CLASS = ['Lying', 'Eating', 'Walking', 'Standing', 'Drinking']


def load_cattle_data(max_samples_per_class: int = None) -> pd.DataFrame:
    """Load raw cattle IMU data (full dataset by default)."""
    print(f"  Loading cattle data from {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"  Total samples in file: {len(df):,}")

    # If max_samples_per_class is specified, do stratified sampling
    if max_samples_per_class is not None:
        sampled_dfs = []
        for label in df['label'].unique():
            label_df = df[df['label'] == label]
            if len(label_df) > max_samples_per_class:
                start_idx = len(label_df) // 4
                label_df = label_df.iloc[start_idx:start_idx + max_samples_per_class]
            sampled_dfs.append(label_df)
        df = pd.concat(sampled_dfs, ignore_index=True)
        print(f"  Stratified sample: {len(df):,} samples")
    else:
        print(f"  Using full dataset: {len(df):,} samples")

    return df

def map_behaviors(df: pd.DataFrame) -> pd.DataFrame:
    """Map cattle behaviors to standardized labels."""
    df = df.copy()
    df['mapped_label'] = df['label'].map(CATTLE_BEHAVIOR_MAP)

    # Remove unmapped behaviors
    unmapped = df[df['mapped_label'].isna()]['label'].unique()
    if len(unmapped) > 0:
        print(f"  Warning: Unmapped behaviors: {unmapped}")

    df = df.dropna(subset=['mapped_label'])
    return df

def create_windows(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding windows from data, grouped by behavior segments."""
    # Group by original label to create windows within each behavior segment
    all_X = []
    all_y = []

    step = int(WINDOW_SAMPLES * (1 - OVERLAP))

    # Process each behavior segment separately to avoid cross-behavior windows
    for label in df['mapped_label'].unique():
        label_df = df[df['mapped_label'] == label].reset_index(drop=True)
        data = label_df[['ax', 'ay', 'az']].values

        n_windows = (len(data) - WINDOW_SAMPLES) // step + 1

        for i in range(n_windows):
            start = i * step
            end = start + WINDOW_SAMPLES
            if end <= len(data):
                all_X.append(data[start:end])
                all_y.append(label)

    X = np.array(all_X)
    y = np.array(all_y)

    # Shuffle to mix behaviors
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    return X, y

def extract_features(X: np.ndarray) -> np.ndarray:
    """Extract 24 statistical features (unified with pig study)."""
    return extract_24_features_batch(X)


def evaluate_5class_classification(X: np.ndarray, y: np.ndarray) -> Dict:
    """Evaluate 5-class behavior classification (analogous to pig study)."""
    print("\n" + "=" * 70)
    print("Experiment 1: 5-Class Behavior Classification")
    print("=" * 70)

    # Filter to 5 core classes
    mask = np.isin(y, BEHAVIORS_5CLASS)
    X_filtered, y_filtered = X[mask], y[mask]

    print(f"  Samples after filtering: {len(X_filtered):,}")
    print(f"  Class distribution:")
    for label in BEHAVIORS_5CLASS:
        count = np.sum(y_filtered == label)
        print(f"    {label}: {count:,} ({count/len(y_filtered)*100:.1f}%)")

    # Extract features
    print(f"\n  Extracting 24 statistical features...")
    X_features = extract_features(X_filtered)

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_filtered)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)

    # Train classifier (same hyperparameters as pig study)
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=20,
        class_weight='balanced', random_state=42, n_jobs=-1
    )

    # 5-fold cross-validation (shuffled - standard for classifier evaluation)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accuracies = []
    f1_scores = []

    print(f"\n  Running 5-fold cross-validation...")
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y_encoded)):
        clf.fit(X_scaled[train_idx], y_encoded[train_idx])
        y_pred = clf.predict(X_scaled[val_idx])

        acc = accuracy_score(y_encoded[val_idx], y_pred)
        f1 = f1_score(y_encoded[val_idx], y_pred, average='macro')

        accuracies.append(acc)
        f1_scores.append(f1)
        print(f"    Fold {fold+1}: Accuracy={acc:.2%}, Macro-F1={f1:.2%}")

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    mean_f1 = np.mean(f1_scores)

    print(f"\n  Results:")
    print(f"    5-Class Accuracy: {mean_acc:.2%} ± {std_acc:.2%}")
    print(f"    5-Class Macro-F1: {mean_f1:.2%}")

    return {
        'accuracy': mean_acc,
        'accuracy_std': std_acc,
        'f1_macro': mean_f1,
        'n_samples': len(X_filtered),
        'class_distribution': {label: int(np.sum(y_filtered == label)) for label in BEHAVIORS_5CLASS}
    }

def evaluate_3cluster_classification(X: np.ndarray, y: np.ndarray) -> Dict:
    """Evaluate 3-cluster classification for ODR control."""
    print("\n" + "=" * 70)
    print("Experiment 2: 3-Cluster Classification (ODR Control)")
    print("=" * 70)

    # Map behaviors to clusters
    behavior_to_cluster = {}
    for cluster, info in CLUSTER_CONFIG.items():
        for b in info['behaviors']:
            behavior_to_cluster[b] = cluster

    # Filter valid behaviors and map to clusters
    valid_behaviors = list(behavior_to_cluster.keys())
    mask = np.isin(y, valid_behaviors)
    X_filtered, y_filtered = X[mask], y[mask]
    y_cluster = np.array([behavior_to_cluster[b] for b in y_filtered])

    print(f"  Samples: {len(X_filtered):,}")
    print(f"  Cluster distribution:")
    for cluster in ['STABLE', 'MODERATE', 'ACTIVE']:
        count = np.sum(y_cluster == cluster)
        print(f"    {cluster}: {count:,} ({count/len(y_cluster)*100:.1f}%)")

    # Extract features
    print(f"\n  Extracting 24 statistical features...")
    X_features = extract_features(X_filtered)

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_cluster)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)

    # Train classifier
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=20,
        class_weight='balanced', random_state=42, n_jobs=-1
    )

    # 5-fold cross-validation (shuffled - standard for classifier evaluation)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accuracies = []
    f1_scores = []

    print(f"\n  Running 5-fold cross-validation...")
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y_encoded)):
        clf.fit(X_scaled[train_idx], y_encoded[train_idx])
        y_pred = clf.predict(X_scaled[val_idx])

        acc = accuracy_score(y_encoded[val_idx], y_pred)
        f1 = f1_score(y_encoded[val_idx], y_pred, average='macro')

        accuracies.append(acc)
        f1_scores.append(f1)
        print(f"    Fold {fold+1}: Accuracy={acc:.2%}, Macro-F1={f1:.2%}")

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    mean_f1 = np.mean(f1_scores)

    print(f"\n  Results:")
    print(f"    3-Cluster Accuracy: {mean_acc:.2%} ± {std_acc:.2%}")
    print(f"    3-Cluster Macro-F1: {mean_f1:.2%}")

    return {
        'cluster_accuracy': mean_acc,
        'cluster_accuracy_std': std_acc,
        'cluster_f1': mean_f1,
        'n_samples': len(X_filtered),
        'cluster_distribution': {cluster: int(np.sum(y_cluster == cluster))
                                  for cluster in ['STABLE', 'MODERATE', 'ACTIVE']}
    }

def evaluate_energy_savings(X: np.ndarray, y: np.ndarray) -> Dict:
    """Evaluate energy savings with adaptive ODR."""
    print("\n" + "=" * 70)
    print("Experiment 3: Energy Savings Analysis")
    print("=" * 70)

    # Map behaviors to clusters
    behavior_to_cluster = {}
    for cluster, info in CLUSTER_CONFIG.items():
        for b in info['behaviors']:
            behavior_to_cluster[b] = cluster

    valid_behaviors = list(behavior_to_cluster.keys())
    mask = np.isin(y, valid_behaviors)
    y_filtered = y[mask]
    y_cluster = np.array([behavior_to_cluster[b] for b in y_filtered])

    # Compute behavior distribution for energy calculation
    cluster_distribution = {}
    for cluster in ['STABLE', 'MODERATE', 'ACTIVE']:
        cluster_distribution[cluster] = np.sum(y_cluster == cluster) / len(y_cluster)

    print(f"  Cluster Distribution:")
    for cluster, ratio in cluster_distribution.items():
        odr = CLUSTER_CONFIG[cluster]['odr']
        print(f"    {cluster}: {ratio:.1%} @ {odr} Hz")

    # Effective ODR
    effective_odr = sum(
        cluster_distribution[c] * CLUSTER_CONFIG[c]['odr']
        for c in cluster_distribution
    )
    print(f"\n  Effective ODR: {effective_odr:.2f} Hz")

    # Full energy model using compute_energy from cluster_mapping.py
    import importlib.util
    _cm_path = Path(__file__).parent / 'utils' / 'cluster_mapping.py'
    _spec = importlib.util.spec_from_file_location('cluster_mapping', _cm_path)
    _cm = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_cm)
    compute_energy = _cm.compute_energy
    BASELINE_ENERGY = _cm.BASELINE_ENERGY
    BASELINE_BATTERY = _cm.BASELINE_BATTERY

    # Build ODR distribution from cluster distribution
    odr_map = {c: CLUSTER_CONFIG[c]['odr'] for c in CLUSTER_CONFIG}
    odr_distribution = {}
    for cluster, frac in cluster_distribution.items():
        odr = odr_map[cluster]
        odr_distribution[odr] = odr_distribution.get(odr, 0.0) + frac

    # Compute adaptive energy (100% inference since no FSM simulation for cattle)
    adaptive_result = compute_energy(odr_distribution, inference_rate=1.0)
    baseline_total = BASELINE_ENERGY[50]
    baseline_battery = BASELINE_BATTERY[50]
    adaptive_total = round(adaptive_result['e_total'])
    adaptive_battery = round(adaptive_result['battery_days'])
    total_savings = adaptive_result['savings_vs_50hz']

    print(f"\n  Energy Analysis (full model, T_inf=1.976ms):")
    print(f"    Baseline (50Hz): {baseline_total} mJ/day, {baseline_battery} days")
    print(f"    Adaptive: {adaptive_total} mJ/day")
    print(f"    Total Savings: {total_savings:.1f}%")
    print(f"    Battery Life: {baseline_battery} -> {adaptive_battery} days")

    return {
        'effective_odr': effective_odr,
        'cluster_distribution': cluster_distribution,
        'baseline_energy_mj': baseline_total,
        'adaptive_energy_mj': adaptive_total,
        'total_savings_pct': total_savings,
        'baseline_battery_days': baseline_battery,
        'adaptive_battery_days': adaptive_battery
    }

def compare_with_pig_study() -> Dict:
    """Compare cattle results with pig study results (from paper Table 8)."""
    # Pig reference values from verified_simulation.py (T_inf=1.976ms)
    pig_results = {
        '5class_accuracy': 0.7798,
        '3cluster_accuracy': 0.8237,
        'cluster_f1': 0.6048,
        'total_savings_pct': 53.0,  # From FSM simulation (Table 4), compressed model, T_inf=1.976ms
        'effective_odr': 13.4,  # From FSM simulation (Table 4), compressed model
        'battery_days': 461  # From FSM simulation (Table 4), compressed model, T_inf=1.976ms
    }
    return pig_results


def main():
    print("=" * 70)
    print("Cattle Generalization Study")
    print("=" * 70)
    print("\nValidating LiveEdge cross-species applicability on cattle data")

    # Load full dataset (no subsampling for proper comparison with pig)
    print("\n[Loading Data]")
    df = load_cattle_data(max_samples_per_class=None)  # Use full dataset

    # Map behaviors
    print("\n[Mapping Behaviors]")
    df = map_behaviors(df)
    print(f"  Mapped labels distribution:")
    print(df['mapped_label'].value_counts())

    # Create windows
    print("\n[Creating Windows]")
    X, y = create_windows(df)
    print(f"  Created {len(X):,} windows")

    # Run experiments
    results_5class = evaluate_5class_classification(X, y)
    results_3cluster = evaluate_3cluster_classification(X, y)
    results_energy = evaluate_energy_savings(X, y)
    pig_results = compare_with_pig_study()

    # Compile results
    all_results = {
        'cattle': {
            '5class': results_5class,
            '3cluster': results_3cluster,
            'energy': results_energy
        },
        'pig_reference': pig_results,
        'metadata': {
            'source': str(RAW_DATA_PATH),
            'samples_used': len(df),
            'windows_created': len(X),
            'window_size_s': WINDOW_SIZE,
            'overlap': OVERLAP
        }
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

    # Save results
    with open(OUTPUT_DIR / 'cattle_generalization_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print comparison summary
    print("\n" + "=" * 70)
    print("Cross-Species Comparison Summary")
    print("=" * 70)

    print(f"\n  {'Metric':<25s} | {'Pig':<12s} | {'Cattle':<12s} | {'Diff':<10s}")
    print("  " + "-" * 65)

    # 5-class accuracy
    pig_5class = pig_results['5class_accuracy'] * 100
    cattle_5class = results_5class['accuracy'] * 100
    diff_5class = cattle_5class - pig_5class
    print(f"  {'5-Class Accuracy':<25s} | {pig_5class:>10.1f}% | {cattle_5class:>10.1f}% | {diff_5class:>+8.1f}%")

    # 3-cluster accuracy
    pig_3cluster = pig_results['3cluster_accuracy'] * 100
    cattle_3cluster = results_3cluster['cluster_accuracy'] * 100
    diff_3cluster = cattle_3cluster - pig_3cluster
    print(f"  {'3-Cluster Accuracy':<25s} | {pig_3cluster:>10.1f}% | {cattle_3cluster:>10.1f}% | {diff_3cluster:>+8.1f}%")

    # Energy savings
    pig_savings = pig_results['total_savings_pct']
    cattle_savings = results_energy['total_savings_pct']
    diff_savings = cattle_savings - pig_savings
    print(f"  {'Energy Savings':<25s} | {pig_savings:>10.1f}% | {cattle_savings:>10.1f}% | {diff_savings:>+8.1f}%")

    # Effective ODR
    pig_odr = pig_results['effective_odr']
    cattle_odr = results_energy['effective_odr']
    diff_odr = cattle_odr - pig_odr
    print(f"  {'Effective ODR':<25s} | {pig_odr:>9.1f}Hz | {cattle_odr:>9.1f}Hz | {diff_odr:>+7.1f}Hz")

    # Battery life
    pig_battery = pig_results['battery_days']
    cattle_battery = results_energy['adaptive_battery_days']
    diff_battery = cattle_battery - pig_battery
    print(f"  {'Battery Life':<25s} | {pig_battery:>9.0f}d | {cattle_battery:>9.0f}d | {diff_battery:>+7.0f}d")

    print(f"\n[Results saved to: {OUTPUT_DIR}]")
    print("\n" + "=" * 70)
    print("Generalization Study Complete!")
    print("=" * 70)

    return all_results

if __name__ == "__main__":
    results = main()
