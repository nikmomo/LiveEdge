#!/usr/bin/env python3
"""ODR state trajectory figure.

Uses single behavior classifier RF(100, d=20) + deterministic cluster mapping,
matching the main paper architecture.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import json

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent
sys.path.insert(0, str(_project_root / 'src'))

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from liveedge.data import extract_24_features_batch

OUTPUT_DIR = Path("L:/GitHub/LiveEdge/outputs/paper_revision/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("L:/GitHub/LiveEdge/data/processed/50hz")

BEHAVIORS_5CLASS = ['Lying', 'Eating', 'Standing', 'Walking', 'Drinking']

BEHAVIOR_TO_CLUSTER = {
    'Lying': 'STABLE',
    'Eating': 'STABLE',
    'Standing': 'ACTIVE',
    'Walking': 'MODERATE',
    'Drinking': 'ACTIVE',
}

CLUSTER_ODR = {
    'STABLE': 6.25,
    'MODERATE': 12.5,
    'ACTIVE': 25.0,
}

CLUSTER_COLORS = {
    'STABLE': '#2ecc71',    # Green
    'MODERATE': '#f39c12',  # Orange
    'ACTIVE': '#e74c3c',    # Red
}

WINDOW_DURATION = 1.5  # seconds


def simulate_trajectory(X: np.ndarray, y_labels: np.ndarray, k_stability: int = 3):
    """
    Simulate LiveEdge trajectory on data.

    Uses single behavior classifier + deterministic cluster mapping.
    Returns ground truth clusters, predicted clusters, and ODR settings.
    """
    # Encode labels
    le_behavior = LabelEncoder()
    y_behavior_encoded = le_behavior.fit_transform(y_labels)

    y_cluster = np.array([BEHAVIOR_TO_CLUSTER[b] for b in y_labels])

    # Use 80% for training
    n_train = int(len(X) * 0.8)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train_behavior = y_behavior_encoded[:n_train]
    y_test_cluster_str = y_cluster[n_train:]

    # Train single behavior classifier
    X_train_feat = extract_24_features_batch(X_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)

    behavior_clf = RandomForestClassifier(
        n_estimators=100, max_depth=20,
        min_samples_split=5, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    )
    behavior_clf.fit(X_train_scaled, y_train_behavior)

    # Predict behaviors, map to clusters
    X_test_feat = extract_24_features_batch(X_test)
    X_test_scaled = scaler.transform(X_test_feat)

    pred_behavior_encoded = behavior_clf.predict(X_test_scaled)
    pred_behavior_str = le_behavior.inverse_transform(pred_behavior_encoded)
    pred_cluster_str = np.array([BEHAVIOR_TO_CLUSTER[b] for b in pred_behavior_str])

    # Apply FSM with k-stability
    current_cluster = None
    last_pred = None
    consecutive_same = 0

    fsm_clusters = []
    odr_settings = []

    for pred in pred_cluster_str:
        if pred == last_pred:
            consecutive_same += 1
        else:
            consecutive_same = 1
        last_pred = pred

        if current_cluster is None or (pred != current_cluster and consecutive_same >= k_stability):
            current_cluster = pred

        fsm_clusters.append(current_cluster)
        odr_settings.append(CLUSTER_ODR[current_cluster])

    return y_test_cluster_str, np.array(fsm_clusters), np.array(odr_settings)


def create_trajectory_figure(gt_clusters, pred_clusters, odr_settings, n_hours: float = 4):
    """Create the trajectory visualization."""
    # Calculate number of windows for n_hours
    windows_per_hour = 3600 / WINDOW_DURATION
    n_windows = min(int(n_hours * windows_per_hour), len(gt_clusters))

    gt = gt_clusters[:n_windows]
    pred = pred_clusters[:n_windows]
    odr = odr_settings[:n_windows]

    # Time axis in hours
    time_hours = np.arange(n_windows) * WINDOW_DURATION / 3600

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    # Map clusters to numeric for plotting
    cluster_to_num = {'STABLE': 0, 'MODERATE': 1, 'ACTIVE': 2}

    gt_num = np.array([cluster_to_num[c] for c in gt])
    pred_num = np.array([cluster_to_num[c] for c in pred])

    # --- Top: Ground Truth ---
    ax1 = axes[0]
    for cluster, num in cluster_to_num.items():
        mask = gt_num == num
        if mask.any():
            for start, end in get_contiguous_regions(mask):
                ax1.axvspan(time_hours[start], time_hours[end],
                           color=CLUSTER_COLORS[cluster], alpha=0.7)

    ax1.set_ylabel('Ground Truth\nCluster', fontsize=11)
    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels(['STABLE', 'MODERATE', 'ACTIVE'])
    ax1.set_ylim(-0.5, 2.5)
    ax1.set_title(f'LiveEdge State Trajectory ({n_hours:.0f} Hours)', fontsize=14)

    # --- Middle: Predicted (with FSM) ---
    ax2 = axes[1]
    for cluster, num in cluster_to_num.items():
        mask = pred_num == num
        if mask.any():
            for start, end in get_contiguous_regions(mask):
                ax2.axvspan(time_hours[start], time_hours[end],
                           color=CLUSTER_COLORS[cluster], alpha=0.7)

    ax2.set_ylabel('Predicted\nCluster (FSM)', fontsize=11)
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['STABLE', 'MODERATE', 'ACTIVE'])
    ax2.set_ylim(-0.5, 2.5)

    # --- Bottom: ODR Setting ---
    ax3 = axes[2]
    ax3.step(time_hours, odr, where='post', linewidth=2, color='#3498db')
    ax3.fill_between(time_hours, odr, step='post', alpha=0.3, color='#3498db')

    ax3.set_ylabel('ODR (Hz)', fontsize=11)
    ax3.set_xlabel('Time (hours)', fontsize=12)
    ax3.set_yticks([6.25, 12.5, 25])
    ax3.set_ylim(0, 30)
    ax3.grid(True, alpha=0.3)

    # Add ODR labels
    for odr_val, cluster in [(6.25, 'STABLE'), (12.5, 'MODERATE'), (25, 'ACTIVE')]:
        ax3.axhline(y=odr_val, color=CLUSTER_COLORS[cluster], linestyle='--', alpha=0.5)

    # Legend
    legend_elements = [
        Patch(facecolor=CLUSTER_COLORS['STABLE'], label='STABLE (6.25 Hz)'),
        Patch(facecolor=CLUSTER_COLORS['MODERATE'], label='MODERATE (12.5 Hz)'),
        Patch(facecolor=CLUSTER_COLORS['ACTIVE'], label='ACTIVE (25 Hz)'),
    ]
    axes[0].legend(handles=legend_elements, loc='upper right', ncol=3)

    plt.tight_layout()

    # Save
    fig.savefig(OUTPUT_DIR / 'odr_trajectory.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'odr_trajectory.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {OUTPUT_DIR / 'odr_trajectory.pdf'}")
    print(f"Saved: {OUTPUT_DIR / 'odr_trajectory.png'}")


def get_contiguous_regions(mask):
    """Get start and end indices of contiguous True regions."""
    regions = []
    in_region = False
    start = 0

    for i, val in enumerate(mask):
        if val and not in_region:
            start = i
            in_region = True
        elif not val and in_region:
            regions.append((start, i-1))
            in_region = False

    if in_region:
        regions.append((start, len(mask)-1))

    return regions


def compute_trajectory_stats(gt_clusters, pred_clusters, odr_settings):
    """Compute statistics about the trajectory."""
    n_windows = len(gt_clusters)

    # Transitions
    gt_transitions = np.sum(gt_clusters[1:] != gt_clusters[:-1])
    pred_transitions = np.sum(pred_clusters[1:] != pred_clusters[:-1])

    # ODR distribution
    unique_odr, counts = np.unique(odr_settings, return_counts=True)
    odr_dist = dict(zip(unique_odr, counts / n_windows))

    # Effective ODR
    eff_odr = np.mean(odr_settings)

    # Cluster match rate
    match_rate = np.mean(gt_clusters == pred_clusters)

    return {
        'n_windows': n_windows,
        'duration_hours': n_windows * WINDOW_DURATION / 3600,
        'gt_transitions': int(gt_transitions),
        'pred_transitions': int(pred_transitions),
        'odr_distribution': {str(k): round(v, 4) for k, v in odr_dist.items()},
        'effective_odr': round(eff_odr, 2),
        'cluster_match_rate': round(match_rate, 4),
    }


def main():
    print("=" * 70)
    print("Figure G2: 24-Hour ODR State Trajectory")
    print("=" * 70)

    # Load data
    print("\n[1] Loading data...")
    X = np.load(DATA_DIR / 'windows.npy')[:, :, :3]
    y_labels = np.load(DATA_DIR / 'labels.npy', allow_pickle=True)

    mask = np.isin(y_labels, BEHAVIORS_5CLASS)
    X = X[mask]
    y_labels = y_labels[mask]

    print(f"  Total samples: {len(X)}")
    print(f"  Total duration: {len(X) * WINDOW_DURATION / 3600:.1f} hours")

    # Simulate trajectory
    print("\n[2] Simulating trajectory...")
    gt_clusters, pred_clusters, odr_settings = simulate_trajectory(X, y_labels, k_stability=3)

    print(f"  Test samples: {len(gt_clusters)}")

    # Create figure
    print("\n[3] Creating figure...")
    create_trajectory_figure(gt_clusters, pred_clusters, odr_settings, n_hours=4)

    # Compute stats
    stats = compute_trajectory_stats(gt_clusters, pred_clusters, odr_settings)

    print(f"\n  Statistics:")
    print(f"    Duration: {stats['duration_hours']:.1f} hours")
    print(f"    GT transitions: {stats['gt_transitions']}")
    print(f"    Pred transitions: {stats['pred_transitions']}")
    print(f"    Effective ODR: {stats['effective_odr']:.2f} Hz")
    print(f"    Cluster match rate: {stats['cluster_match_rate']*100:.1f}%")

    # Save stats
    with open(OUTPUT_DIR / 'odr_trajectory_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved: {OUTPUT_DIR / 'odr_trajectory_stats.json'}")


if __name__ == "__main__":
    main()
