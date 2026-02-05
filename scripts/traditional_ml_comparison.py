#!/usr/bin/env python3
"""
Traditional ML Model Comparison for LiveEdge

Compare MCU-compatible traditional ML classifiers:
- Random Forest
- XGBoost
- SVM (RBF kernel)
- SVM (Linear kernel)
- Logistic Regression
- Decision Tree

All models use RFE-20 features for fair comparison.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
import time
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not available, skipping...")

# Paths
DATA_DIR = Path("L:/GitHub/LiveEdge/data/processed/50hz")
OUTPUT_DIR = Path("L:/GitHub/LiveEdge/outputs/ml_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# RFE-20 feature indices (from paper)
RFE_20_FEATURES = [
    'acc_magnitude_mean', 'acc_magnitude_range', 'acc_magnitude_std',
    'acc_x_mean', 'acc_x_std', 'acc_y_mean', 'acc_y_std',
    'acc_z_mean', 'acc_z_std', 'interquartile_range',
    'zero_crossing_rate', 'band_power_0_2', 'band_power_2_5',
    'high_freq_ratio', 'jerk_mean', 'jerk_x_std',
    'jerk_y_std', 'jerk_z_std', 'energy', 'signal_magnitude_area'
]


def extract_features(windows):
    """Extract RFE-20 features from raw windows."""
    features = []

    for window in windows:
        # window shape: (75, 6) - 75 samples, 6 axes (ax, ay, az, gx, gy, gz)
        acc = window[:, :3]  # Accelerometer only

        # Magnitude
        magnitude = np.sqrt(np.sum(acc**2, axis=1))

        # Time domain features
        feat = {
            'acc_magnitude_mean': np.mean(magnitude),
            'acc_magnitude_range': np.max(magnitude) - np.min(magnitude),
            'acc_magnitude_std': np.std(magnitude),
            'acc_x_mean': np.mean(acc[:, 0]),
            'acc_x_std': np.std(acc[:, 0]),
            'acc_y_mean': np.mean(acc[:, 1]),
            'acc_y_std': np.std(acc[:, 1]),
            'acc_z_mean': np.mean(acc[:, 2]),
            'acc_z_std': np.std(acc[:, 2]),
            'interquartile_range': np.percentile(magnitude, 75) - np.percentile(magnitude, 25),
        }

        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(magnitude - np.mean(magnitude))) != 0)
        feat['zero_crossing_rate'] = zero_crossings / len(magnitude)

        # Frequency domain features (simplified)
        fft = np.abs(np.fft.rfft(magnitude))
        freqs = np.fft.rfftfreq(len(magnitude), 1/50)  # 50 Hz sampling

        # Band powers
        feat['band_power_0_2'] = np.sum(fft[(freqs >= 0) & (freqs < 2)]**2)
        feat['band_power_2_5'] = np.sum(fft[(freqs >= 2) & (freqs < 5)]**2)
        total_power = np.sum(fft**2) + 1e-10
        high_freq_power = np.sum(fft[freqs >= 10]**2)
        feat['high_freq_ratio'] = high_freq_power / total_power

        # Jerk features
        jerk = np.diff(acc, axis=0)
        jerk_magnitude = np.sqrt(np.sum(jerk**2, axis=1))
        feat['jerk_mean'] = np.mean(jerk_magnitude)
        feat['jerk_x_std'] = np.std(jerk[:, 0])
        feat['jerk_y_std'] = np.std(jerk[:, 1])
        feat['jerk_z_std'] = np.std(jerk[:, 2])

        # Energy and SMA
        feat['energy'] = np.sum(magnitude**2)
        feat['signal_magnitude_area'] = np.sum(np.abs(acc))

        features.append([feat[f] for f in RFE_20_FEATURES])

    return np.array(features)


def load_data(max_samples=30000):
    """Load and prepare data with RFE-20 features."""
    print("[Loading Data]")
    X_raw = np.load(DATA_DIR / "windows.npy")
    y_str = np.load(DATA_DIR / "labels.npy")

    # Exclude Interacting
    valid_behaviors = ['Lying', 'Eating', 'Walking', 'Standing', 'Drinking']
    mask = np.isin(y_str, valid_behaviors)
    X_raw = X_raw[mask]
    y_str = y_str[mask]

    print(f"  Samples after filtering: {len(X_raw)}")

    # Subsample
    if max_samples and len(X_raw) > max_samples:
        from sklearn.model_selection import train_test_split
        X_raw, _, y_str, _ = train_test_split(
            X_raw, y_str, train_size=max_samples, stratify=y_str, random_state=42
        )
        print(f"  Subsampled to: {len(X_raw)}")

    # Extract features
    print("  Extracting RFE-20 features...")
    X = extract_features(X_raw)
    print(f"  Feature matrix shape: {X.shape}")

    # Convert labels
    behavior_names = sorted(valid_behaviors)
    label_to_idx = {name: i for i, name in enumerate(behavior_names)}
    y = np.array([label_to_idx[label] for label in y_str])

    # Cluster labels
    behavior_to_cluster = {
        'Lying': 0, 'Eating': 1, 'Standing': 1, 'Walking': 2, 'Drinking': 2
    }
    y_cluster = np.array([behavior_to_cluster[label] for label in y_str])

    print(f"  Behavior distribution: {Counter(y_str)}")

    return X, y, y_cluster, behavior_names


def get_models():
    """Get all MCU-compatible models to compare."""
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=20,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=20, class_weight='balanced', random_state=42
        ),
        'SVM (RBF)': SVC(
            kernel='rbf', C=1.0, gamma='scale',
            class_weight='balanced', random_state=42
        ),
        'SVM (Linear)': SVC(
            kernel='linear', C=1.0,
            class_weight='balanced', random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42, n_jobs=-1
        ),
    }

    if HAS_XGBOOST:
        models['XGBoost'] = XGBClassifier(
            n_estimators=100, max_depth=6,
            learning_rate=0.1, random_state=42,
            use_label_encoder=False, eval_metric='mlogloss'
        )

    return models


def estimate_mcu_metrics(model_name):
    """Estimate MCU deployment metrics for each model."""
    # Approximate metrics based on model characteristics
    metrics = {
        'Random Forest': {
            'flash_kb': 150,  # Tree structure storage
            'ram_kb': 20,     # Inference buffer
            'inference_ms': 15,
            'mcu_compatible': True,
            'notes': 'Tree traversal, no matrix ops'
        },
        'Decision Tree': {
            'flash_kb': 10,
            'ram_kb': 5,
            'inference_ms': 2,
            'mcu_compatible': True,
            'notes': 'Single tree, very efficient'
        },
        'XGBoost': {
            'flash_kb': 200,
            'ram_kb': 30,
            'inference_ms': 20,
            'mcu_compatible': True,
            'notes': 'Gradient boosted trees'
        },
        'SVM (RBF)': {
            'flash_kb': 300,  # Support vectors
            'ram_kb': 50,
            'inference_ms': 50,
            'mcu_compatible': False,  # Kernel computation heavy
            'notes': 'Kernel computation overhead'
        },
        'SVM (Linear)': {
            'flash_kb': 50,
            'ram_kb': 10,
            'inference_ms': 5,
            'mcu_compatible': True,
            'notes': 'Simple dot product'
        },
        'Logistic Regression': {
            'flash_kb': 20,
            'ram_kb': 5,
            'inference_ms': 3,
            'mcu_compatible': True,
            'notes': 'Matrix-vector multiply'
        },
    }
    return metrics.get(model_name, {})


def main():
    print("=" * 70)
    print("Traditional ML Model Comparison")
    print("=" * 70)

    # Load data
    X, y_behavior, y_cluster, behavior_names = load_data(max_samples=30000)

    # Get models
    models = get_models()

    # Cross-validation (shuffled - standard for classifier evaluation)
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    results = {}

    print(f"\n[Running {n_folds}-Fold Cross-Validation]")

    for model_name, model in models.items():
        print(f"\n--- {model_name} ---")

        behavior_accs = []
        behavior_f1s = []
        cluster_accs = []
        cluster_f1s = []
        train_times = []
        inference_times = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_behavior)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train_b, y_test_b = y_behavior[train_idx], y_behavior[test_idx]
            y_train_c, y_test_c = y_cluster[train_idx], y_cluster[test_idx]

            # Train behavior classifier
            model_b = type(model)(**model.get_params())
            start = time.time()
            model_b.fit(X_train, y_train_b)
            train_time = time.time() - start

            start = time.time()
            y_pred_b = model_b.predict(X_test)
            inference_time = (time.time() - start) / len(X_test)

            behavior_acc = accuracy_score(y_test_b, y_pred_b)
            behavior_f1 = f1_score(y_test_b, y_pred_b, average='macro')

            # Train cluster classifier
            model_c = type(model)(**model.get_params())
            model_c.fit(X_train, y_train_c)
            y_pred_c = model_c.predict(X_test)

            cluster_acc = accuracy_score(y_test_c, y_pred_c)
            cluster_f1 = f1_score(y_test_c, y_pred_c, average='macro')

            behavior_accs.append(behavior_acc)
            behavior_f1s.append(behavior_f1)
            cluster_accs.append(cluster_acc)
            cluster_f1s.append(cluster_f1)
            train_times.append(train_time)
            inference_times.append(inference_time)

        mcu_metrics = estimate_mcu_metrics(model_name)

        results[model_name] = {
            'behavior_accuracy': np.mean(behavior_accs) * 100,
            'behavior_accuracy_std': np.std(behavior_accs) * 100,
            'behavior_f1': np.mean(behavior_f1s) * 100,
            'cluster_accuracy': np.mean(cluster_accs) * 100,
            'cluster_accuracy_std': np.std(cluster_accs) * 100,
            'cluster_f1': np.mean(cluster_f1s) * 100,
            'train_time': np.mean(train_times),
            'inference_time_ms': np.mean(inference_times) * 1000,
            **mcu_metrics
        }

        print(f"  Behavior Acc: {results[model_name]['behavior_accuracy']:.2f}% +/- {results[model_name]['behavior_accuracy_std']:.2f}%")
        print(f"  Cluster Acc:  {results[model_name]['cluster_accuracy']:.2f}% +/- {results[model_name]['cluster_accuracy_std']:.2f}%")
        print(f"  MCU Compatible: {mcu_metrics.get('mcu_compatible', 'N/A')}")

    # Summary table
    print("\n" + "=" * 70)
    print("Summary: Traditional ML Comparison (RFE-20 Features)")
    print("=" * 70)

    print(f"\n{'Model':<20} | {'Behavior Acc':>12} | {'Cluster Acc':>12} | {'MCU OK':>8} | {'Notes':<25}")
    print("-" * 85)

    # Sort by behavior accuracy
    sorted_models = sorted(results.items(), key=lambda x: x[1]['behavior_accuracy'], reverse=True)

    for model_name, r in sorted_models:
        mcu_ok = "Yes" if r.get('mcu_compatible', False) else "No"
        notes = r.get('notes', '')[:25]
        print(f"{model_name:<20} | {r['behavior_accuracy']:>11.2f}% | {r['cluster_accuracy']:>11.2f}% | {mcu_ok:>8} | {notes:<25}")

    # Find best MCU-compatible model
    mcu_compatible = [(n, r) for n, r in results.items() if r.get('mcu_compatible', False)]
    if mcu_compatible:
        best_mcu = max(mcu_compatible, key=lambda x: x[1]['behavior_accuracy'])
        print(f"\nBest MCU-compatible model: {best_mcu[0]} ({best_mcu[1]['behavior_accuracy']:.2f}%)")

    # Save results
    with open(OUTPUT_DIR / 'traditional_ml_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[Results saved to: {OUTPUT_DIR / 'traditional_ml_comparison.json'}]")

    return results


if __name__ == "__main__":
    results = main()
