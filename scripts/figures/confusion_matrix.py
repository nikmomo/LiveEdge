#!/usr/bin/env python3
"""
Cluster confusion matrix figure.

Computes confusion matrix from compressed RF (30 trees, depth 8)
using 5-fold temporal-order CV, matching the main paper results.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent
sys.path.insert(0, str(_project_root / 'src'))

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

CLUSTER_ORDER = ['STABLE', 'MODERATE', 'ACTIVE']


def compute_confusion_matrix():
    """Compute cluster confusion matrix using compressed RF + temporal CV."""
    print("[1] Loading data...")
    X = np.load(DATA_DIR / 'windows.npy')[:, :, :3]
    y = np.load(DATA_DIR / 'labels.npy', allow_pickle=True)

    mask = np.isin(y, BEHAVIORS_5CLASS)
    X, y = X[mask], y[mask]
    print(f"  Loaded {len(X)} samples")

    y_cluster = np.array([BEHAVIOR_TO_CLUSTER[b] for b in y])

    print("[2] Extracting 24 features...")
    X_features = extract_24_features_batch(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_cluster)

    print("[3] Running 5-fold temporal CV (compressed RF: 30 trees, depth 8)...")
    cv = StratifiedKFold(n_splits=5, shuffle=False)
    y_pred_all = np.zeros_like(y_encoded)

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y_encoded)):
        clf = RandomForestClassifier(
            n_estimators=30, max_depth=8,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        clf.fit(X_scaled[train_idx], y_encoded[train_idx])
        y_pred_all[val_idx] = clf.predict(X_scaled[val_idx])
        print(f"    Fold {fold+1}/5 done")

    y_pred_clusters = le.inverse_transform(y_pred_all)

    # Build confusion matrix in CLUSTER_ORDER
    cm = confusion_matrix(y_cluster, y_pred_clusters, labels=CLUSTER_ORDER)

    accuracy = np.mean(y_cluster == y_pred_clusters)
    print(f"\n  Overall weighted accuracy: {accuracy:.2%}")

    return cm, y_cluster, y_pred_clusters


def create_confusion_matrix_figure(cm):
    """Create single-panel confusion matrix heatmap for paper."""
    fig, ax = plt.subplots(figsize=(7, 6))

    # Normalize by row (true class)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    # Create annotations with count and percentage
    annot = []
    for i in range(len(CLUSTER_ORDER)):
        row = []
        for j in range(len(CLUSTER_ORDER)):
            count = cm[i, j]
            pct = cm_normalized[i, j]
            row.append(f'{count:,}\n({pct:.1f}%)')
        annot.append(row)

    sns.heatmap(cm_normalized, annot=annot, fmt='', cmap='Blues',
                xticklabels=CLUSTER_ORDER, yticklabels=CLUSTER_ORDER,
                ax=ax, cbar_kws={'label': 'Percentage (%)'})

    ax.set_xlabel('Predicted Cluster', fontsize=12)
    ax.set_ylabel('True Cluster', fontsize=12)
    ax.set_title('Cluster Classification Confusion Matrix', fontsize=14)

    # Highlight diagonal (correct)
    for i in range(len(CLUSTER_ORDER)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                                   edgecolor='green', linewidth=3))

    # Highlight critical error: ACTIVE -> STABLE (row=2, col=0)
    ax.add_patch(plt.Rectangle((0, 2), 1, 1, fill=False,
                               edgecolor='red', linewidth=3, linestyle='--'))

    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / 'confusion_matrix.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nSaved: {OUTPUT_DIR / 'confusion_matrix.pdf'}")
    print(f"Saved: {OUTPUT_DIR / 'confusion_matrix.png'}")


def compute_error_analysis(cm):
    """Compute detailed error analysis."""
    total = cm.sum()
    correct = np.trace(cm)
    errors = total - correct

    # Per-class accuracy
    per_class_acc = {}
    for i, name in enumerate(CLUSTER_ORDER):
        row_total = cm[i].sum()
        per_class_acc[name] = cm[i, i] / row_total if row_total > 0 else 0

    # Critical: ACTIVE -> STABLE (row=2, col=0)
    active_total = cm[2].sum()
    active_to_stable = cm[2, 0]
    active_to_stable_rate_conditional = active_to_stable / active_total if active_total > 0 else 0
    active_to_stable_rate_global = active_to_stable / total

    # ACTIVE -> MODERATE (row=2, col=1)
    active_to_moderate = cm[2, 1]
    active_to_moderate_rate = active_to_moderate / active_total if active_total > 0 else 0

    return {
        'total_samples': int(total),
        'correct': int(correct),
        'accuracy': round(correct / total, 4),
        'total_errors': int(errors),
        'error_rate': round(errors / total, 4),
        'per_class_accuracy': {k: round(v, 4) for k, v in per_class_acc.items()},
        'active_to_stable_count': int(active_to_stable),
        'active_to_stable_conditional': round(active_to_stable_rate_conditional, 4),
        'active_to_stable_global': round(active_to_stable_rate_global, 4),
        'active_to_moderate_count': int(active_to_moderate),
        'active_to_moderate_conditional': round(active_to_moderate_rate, 4),
    }


def main():
    print("=" * 70)
    print("Cluster Confusion Matrix (Compressed RF, Temporal CV)")
    print("=" * 70)

    cm, y_true, y_pred = compute_confusion_matrix()

    # Print confusion matrix
    print("\n[4] Confusion Matrix:")
    print(f"      {''.join([f'{c:>12}' for c in CLUSTER_ORDER])}")
    for i, row_name in enumerate(CLUSTER_ORDER):
        row = ''.join([f'{cm[i,j]:>12,}' for j in range(3)])
        print(f"  {row_name:<10}{row}")

    # Normalized (row percentages)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    print(f"\n  Row-normalized (%):")
    print(f"      {''.join([f'{c:>12}' for c in CLUSTER_ORDER])}")
    for i, row_name in enumerate(CLUSTER_ORDER):
        row = ''.join([f'{cm_norm[i,j]:>11.1f}%' for j in range(3)])
        print(f"  {row_name:<10}{row}")

    # Error analysis
    print("\n[5] Error Analysis:")
    analysis = compute_error_analysis(cm)

    print(f"    Total samples: {analysis['total_samples']:,}")
    print(f"    Overall accuracy: {analysis['accuracy']*100:.2f}%")
    print(f"    Per-class accuracy:")
    for name, acc in analysis['per_class_accuracy'].items():
        print(f"      {name}: {acc*100:.1f}%")
    print(f"    ACTIVE→STABLE (critical):")
    print(f"      Count: {analysis['active_to_stable_count']:,}")
    print(f"      Conditional rate (of ACTIVE): {analysis['active_to_stable_conditional']*100:.1f}%")
    print(f"      Global rate (of all samples): {analysis['active_to_stable_global']*100:.1f}%")
    print(f"    ACTIVE→MODERATE:")
    print(f"      Count: {analysis['active_to_moderate_count']:,}")
    print(f"      Conditional rate (of ACTIVE): {analysis['active_to_moderate_conditional']*100:.1f}%")

    # Create figure
    print("\n[6] Creating figure...")
    create_confusion_matrix_figure(cm)

    # Save analysis
    analysis['confusion_matrix'] = cm.tolist()
    analysis['cluster_order'] = CLUSTER_ORDER
    with open(OUTPUT_DIR / 'confusion_matrix_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"Saved: {OUTPUT_DIR / 'confusion_matrix_analysis.json'}")


if __name__ == "__main__":
    main()
