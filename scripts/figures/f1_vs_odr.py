#!/usr/bin/env python3
"""Per-behavior F1 vs sampling rate figure."""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent
sys.path.insert(0, str(_project_root / 'src'))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

OUTPUT_DIR = Path("L:/GitHub/LiveEdge/outputs/paper_revision/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("L:/GitHub/LiveEdge/data/processed/50hz")

BEHAVIORS_5CLASS = ['Lying', 'Eating', 'Standing', 'Walking', 'Drinking']
ODRS = [50.0, 25.0, 12.5, 10.0, 6.25]

BEHAVIOR_COLORS = {
    'Lying': '#2ecc71',      # Green
    'Eating': '#f39c12',     # Orange
    'Standing': '#3498db',   # Blue
    'Walking': '#e74c3c',    # Red
    'Drinking': '#9b59b6',   # Purple
}

BEHAVIOR_TO_CLUSTER = {
    'Lying': 'STABLE',
    'Eating': 'STABLE',
    'Standing': 'ACTIVE',
    'Walking': 'MODERATE',
    'Drinking': 'ACTIVE',
}

# fmin values and corresponding ODRs (from Table 1)
# Based on tau=5% threshold from 24-feature fmin analysis
FMIN_ODRS = {
    'Lying': 6.25,     # STABLE cluster
    'Eating': 6.25,    # STABLE cluster
    'Standing': 25.0,  # ACTIVE cluster
    'Walking': 12.5,   # MODERATE cluster
    'Drinking': 25.0,  # ACTIVE cluster
}

# Threshold for fmin determination
TAU = 0.05  # 5%


def extract_features(X: np.ndarray) -> np.ndarray:
    """Extract 24 simple features."""
    if X.ndim == 2:
        X = X[np.newaxis, ...]

    n_windows, n_samples, n_channels = X.shape
    n_features = n_channels * 7 + 3
    features = np.zeros((n_windows, n_features))

    idx = 0
    for ch in range(n_channels):
        channel_data = X[:, :, ch]
        features[:, idx] = np.mean(channel_data, axis=1)
        features[:, idx+1] = np.std(channel_data, axis=1)
        features[:, idx+2] = np.min(channel_data, axis=1)
        features[:, idx+3] = np.max(channel_data, axis=1)
        features[:, idx+4] = np.percentile(channel_data, 25, axis=1)
        features[:, idx+5] = np.percentile(channel_data, 75, axis=1)
        features[:, idx+6] = np.sqrt(np.mean(channel_data**2, axis=1))
        idx += 7

    acc_mag = np.sqrt(X[:, :, 0]**2 + X[:, :, 1]**2 + X[:, :, 2]**2)
    features[:, idx] = np.mean(acc_mag, axis=1)
    features[:, idx+1] = np.std(acc_mag, axis=1)
    features[:, idx+2] = np.max(acc_mag, axis=1) - np.min(acc_mag, axis=1)

    return features


def resample_batch(X: np.ndarray, target_odr: float, source_odr: float = 50) -> np.ndarray:
    """Resample windows to target ODR."""
    if target_odr >= source_odr:
        return X
    factor = int(source_odr / target_odr)
    return X[:, ::factor, :]


def compute_f1_by_odr(X: np.ndarray, y: np.ndarray, n_folds: int = 5):
    """Compute per-behavior F1 at each ODR."""
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classes = list(le.classes_)

    results = {behavior: {str(odr): [] for odr in ODRS} for behavior in classes}

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)  # shuffled for classifier eval

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y_encoded)):
        print(f"    Fold {fold + 1}/{n_folds}...", flush=True)

        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y_encoded[train_idx]
        y_test = y_encoded[test_idx]

        for odr in ODRS:
            # Resample
            X_train_rs = resample_batch(X_train, odr)
            X_test_rs = resample_batch(X_test, odr)

            # Extract features
            X_train_feat = extract_features(X_train_rs)
            X_test_feat = extract_features(X_test_rs)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_feat)
            X_test_scaled = scaler.transform(X_test_feat)

            # Train classifier
            clf = RandomForestClassifier(
                n_estimators=100, max_depth=20,
                class_weight='balanced', random_state=42, n_jobs=-1
            )
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)

            # Per-class F1
            f1s = f1_score(y_test, y_pred, average=None, labels=range(len(classes)))

            for i, behavior in enumerate(classes):
                results[behavior][str(odr)].append(f1s[i])

    # Average across folds
    avg_results = {}
    for behavior in classes:
        avg_results[behavior] = {}
        for odr in ODRS:
            f1_list = results[behavior][str(odr)]
            avg_results[behavior][str(odr)] = {
                'mean': round(np.mean(f1_list), 4),
                'std': round(np.std(f1_list), 4),
            }

    return avg_results


def create_f1_vs_odr_figure(results):
    """Create the F1 vs ODR figure."""
    fig, ax = plt.subplots(figsize=(12, 7))

    for behavior in BEHAVIORS_5CLASS:
        odrs = []
        f1_means = []
        f1_stds = []

        for odr in ODRS:
            odrs.append(odr)
            f1_means.append(results[behavior][str(odr)]['mean'] * 100)
            f1_stds.append(results[behavior][str(odr)]['std'] * 100)

        odrs = np.array(odrs)
        f1_means = np.array(f1_means)
        f1_stds = np.array(f1_stds)

        # Sort by ODR
        sort_idx = np.argsort(odrs)
        odrs = odrs[sort_idx]
        f1_means = f1_means[sort_idx]
        f1_stds = f1_stds[sort_idx]

        ax.plot(odrs, f1_means, 'o-', label=f'{behavior} (fmin={FMIN_ODRS[behavior]}Hz)',
                color=BEHAVIOR_COLORS[behavior], linewidth=2, markersize=8)
        ax.fill_between(odrs, f1_means - f1_stds, f1_means + f1_stds,
                        alpha=0.2, color=BEHAVIOR_COLORS[behavior])

        # Mark fmin point
        fmin_odr = FMIN_ODRS[behavior]
        fmin_f1 = results[behavior][str(fmin_odr)]['mean'] * 100
        ax.scatter([fmin_odr], [fmin_f1], marker='s', s=150,
                   color=BEHAVIOR_COLORS[behavior], edgecolors='black', linewidths=2,
                   zorder=10)

    # Reference lines for ODR levels
    ax.axvline(6.25, color='green', linestyle='--', alpha=0.5, label='STABLE (6.25 Hz)')
    ax.axvline(12.5, color='orange', linestyle='--', alpha=0.5, label='MODERATE (12.5 Hz)')
    ax.axvline(25, color='red', linestyle='--', alpha=0.5, label='ACTIVE (25 Hz)')

    ax.set_xlabel('Sampling Rate (Hz)', fontsize=12)
    ax.set_ylabel('F1 Score (%)', fontsize=12)
    ax.set_title('Per-Behavior F1 Score vs Sampling Rate', fontsize=14)
    ax.legend(loc='lower right', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 55)
    ax.set_xticks([6.25, 10, 12.5, 25, 50])

    # Add tau threshold annotation
    ax.text(0.02, 0.02, f'τ = {TAU*100:.0f}% threshold for fmin determination',
            transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / 'f1_vs_odr.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'f1_vs_odr.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {OUTPUT_DIR / 'f1_vs_odr.pdf'}")
    print(f"Saved: {OUTPUT_DIR / 'f1_vs_odr.png'}")


def compute_fmin_analysis(results):
    """Analyze fmin determination."""
    analysis = {}

    for behavior in BEHAVIORS_5CLASS:
        # Get F1 at 50Hz (reference)
        f1_50 = results[behavior]['50.0']['mean']

        # Check each ODR
        analysis[behavior] = {
            'f1_at_50hz': round(f1_50 * 100, 2),
            'odr_analysis': {},
        }

        for odr in ODRS:
            f1_odr = results[behavior][str(odr)]['mean']
            delta = f1_50 - f1_odr

            analysis[behavior]['odr_analysis'][str(odr)] = {
                'f1': round(f1_odr * 100, 2),
                'delta_from_50hz': round(delta * 100, 2),
                'exceeds_tau': delta > TAU,
            }

        # Determine fmin
        for odr in sorted(ODRS):
            delta = results[behavior]['50.0']['mean'] - results[behavior][str(odr)]['mean']
            if delta <= TAU:
                analysis[behavior]['fmin_hz'] = odr
                break
        else:
            analysis[behavior]['fmin_hz'] = 50.0

        analysis[behavior]['assigned_cluster'] = BEHAVIOR_TO_CLUSTER[behavior]

    return analysis


def main():
    print("=" * 70)
    print("Figure G5: Per-Behavior F1 vs Sampling Rate")
    print("=" * 70)

    # Load data
    print("\n[1] Loading data...")
    X = np.load(DATA_DIR / 'windows.npy')[:, :, :3]
    y_labels = np.load(DATA_DIR / 'labels.npy', allow_pickle=True)

    mask = np.isin(y_labels, BEHAVIORS_5CLASS)
    X = X[mask]
    y_labels = y_labels[mask]

    print(f"  Samples: {len(X)}")

    # Compute F1 by ODR
    print("\n[2] Computing F1 scores...")
    results = compute_f1_by_odr(X, y_labels)

    # Print results
    print("\n[3] Results:")
    print(f"\n  {'Behavior':<12} | " + " | ".join([f'{odr:>7.1f}Hz' for odr in sorted(ODRS)]))
    print("  " + "-" * 70)

    for behavior in BEHAVIORS_5CLASS:
        f1s = [f"{results[behavior][str(odr)]['mean']*100:.1f}%" for odr in sorted(ODRS)]
        print(f"  {behavior:<12} | " + " | ".join([f'{f:>8}' for f in f1s]))

    # fmin analysis
    print("\n[4] fmin Analysis:")
    analysis = compute_fmin_analysis(results)

    for behavior in BEHAVIORS_5CLASS:
        fmin = analysis[behavior]['fmin_hz']
        cluster = analysis[behavior]['assigned_cluster']
        print(f"  {behavior:<12}: fmin={fmin:.2f}Hz -> {cluster}")

    # Create figure
    print("\n[5] Creating figure...")
    create_f1_vs_odr_figure(results)

    # Save data
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy(i) for i in obj]
        elif hasattr(obj, 'item'):  # Catch any remaining numpy scalar types
            return obj.item()
        return obj

    output = convert_numpy({
        'f1_by_odr': results,
        'fmin_analysis': analysis,
        'tau_threshold': TAU,
    })

    with open(OUTPUT_DIR / 'f1_vs_odr_data.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved: {OUTPUT_DIR / 'f1_vs_odr_data.json'}")


if __name__ == "__main__":
    main()
