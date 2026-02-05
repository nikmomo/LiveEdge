#!/usr/bin/env python3
"""Cluster confusion matrix figure."""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

OUTPUT_DIR = Path("L:/GitHub/LiveEdge/outputs/paper_revision/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("L:/GitHub/LiveEdge/outputs/paper_revision")

CLUSTER_ORDER = ['STABLE', 'MODERATE', 'ACTIVE']

# Cost matrix: higher cost for under-sampling
COST_MATRIX = np.array([
    [0, 1, 1],   # STABLE -> {STABLE, MODERATE, ACTIVE}
    [2, 0, 1],   # MODERATE -> under-sampling is cost 2
    [3, 2, 0],   # ACTIVE -> STABLE is critical (3), MODERATE is moderate (2)
])


def load_confusion_matrix():
    """Load confusion matrix from comprehensive statistics."""
    with open(DATA_DIR / 'comprehensive_statistics.json', 'r') as f:
        data = json.load(f)

    cm = np.array(data['classification_results']['cluster']['confusion_matrix'])
    class_names = data['classification_results']['cluster']['class_names']

    return cm, class_names


def reorder_confusion_matrix(cm, class_names):
    """Reorder confusion matrix to match CLUSTER_ORDER."""
    order = [class_names.index(c) for c in CLUSTER_ORDER]
    cm_reordered = cm[order][:, order]
    return cm_reordered


def create_confusion_matrix_figure(cm, cost_matrix=None):
    """Create confusion matrix heatmap."""
    fig, axes = plt.subplots(1, 2 if cost_matrix is not None else 1,
                              figsize=(14 if cost_matrix is not None else 8, 6))

    if cost_matrix is not None:
        ax1, ax2 = axes
    else:
        ax1 = axes

    # --- Standard Confusion Matrix (percentages) ---
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
                ax=ax1, cbar_kws={'label': 'Percentage (%)'})

    ax1.set_xlabel('Predicted Cluster', fontsize=12)
    ax1.set_ylabel('True Cluster', fontsize=12)
    ax1.set_title('(a) Cluster Confusion Matrix', fontsize=14)

    # Highlight diagonal
    for i in range(len(CLUSTER_ORDER)):
        ax1.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='green', linewidth=3))

    # --- Cost-Weighted Matrix ---
    if cost_matrix is not None:
        # Weight by cost
        weighted = cm * cost_matrix

        # Create annotations
        annot_weighted = []
        for i in range(len(CLUSTER_ORDER)):
            row = []
            for j in range(len(CLUSTER_ORDER)):
                count = cm[i, j]
                cost = cost_matrix[i, j]
                weighted_val = weighted[i, j]
                row.append(f'{count:,}\n(cost={cost})')
            annot_weighted.append(row)

        # Custom colormap: green (0) to red (high cost)
        cmap = sns.diverging_palette(130, 10, as_cmap=True)

        sns.heatmap(weighted, annot=annot_weighted, fmt='', cmap='YlOrRd',
                    xticklabels=CLUSTER_ORDER, yticklabels=CLUSTER_ORDER,
                    ax=ax2, cbar_kws={'label': 'Weighted Error'})

        ax2.set_xlabel('Predicted Cluster', fontsize=12)
        ax2.set_ylabel('True Cluster', fontsize=12)
        ax2.set_title('(b) Cost-Weighted Confusion Matrix', fontsize=14)

        # Highlight critical errors (cost >= 2)
        for i in range(len(CLUSTER_ORDER)):
            for j in range(len(CLUSTER_ORDER)):
                if cost_matrix[i, j] >= 2:
                    ax2.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                                edgecolor='red', linewidth=3))

    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / 'confusion_matrix.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {OUTPUT_DIR / 'confusion_matrix.pdf'}")
    print(f"Saved: {OUTPUT_DIR / 'confusion_matrix.png'}")


def create_behavior_confusion_matrix():
    """Create behavior-level confusion matrix."""
    with open(DATA_DIR / 'comprehensive_statistics.json', 'r') as f:
        data = json.load(f)

    cm = np.array(data['classification_results']['behavior']['confusion_matrix'])
    class_names = data['classification_results']['behavior']['class_names']

    fig, ax = plt.subplots(figsize=(10, 8))

    # Normalize
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    # Create annotations
    annot = []
    for i in range(len(class_names)):
        row = []
        for j in range(len(class_names)):
            pct = cm_normalized[i, j]
            row.append(f'{pct:.1f}%')
        annot.append(row)

    sns.heatmap(cm_normalized, annot=annot, fmt='', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Percentage (%)'})

    ax.set_xlabel('Predicted Behavior', fontsize=12)
    ax.set_ylabel('True Behavior', fontsize=12)
    ax.set_title('Behavior Confusion Matrix', fontsize=14)

    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / 'behavior_confusion_matrix.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'behavior_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {OUTPUT_DIR / 'behavior_confusion_matrix.pdf'}")


def compute_error_analysis(cm, cost_matrix):
    """Compute detailed error analysis."""
    total = cm.sum()
    correct = np.trace(cm)
    errors = total - correct

    # Under-sampling errors (cost >= 2)
    under_sampling = 0
    over_sampling = 0

    for i in range(3):
        for j in range(3):
            if cost_matrix[i, j] >= 2:
                under_sampling += cm[i, j]
            elif cost_matrix[i, j] == 1:
                over_sampling += cm[i, j]

    # Critical errors (cost = 3)
    critical = cm[2, 0]  # ACTIVE -> STABLE

    # Weighted error
    weighted_total = (cm * cost_matrix).sum()

    return {
        'total_samples': int(total),
        'correct': int(correct),
        'accuracy': round(correct / total, 4),
        'total_errors': int(errors),
        'error_rate': round(errors / total, 4),
        'under_sampling_errors': int(under_sampling),
        'under_sampling_rate': round(under_sampling / total, 4),
        'over_sampling_errors': int(over_sampling),
        'over_sampling_rate': round(over_sampling / total, 4),
        'critical_errors': int(critical),
        'critical_rate': round(critical / total, 4),
        'weighted_error': int(weighted_total),
        'normalized_weighted_error': round(weighted_total / (total * 3), 4),
    }


def main():
    print("=" * 70)
    print("Figure G4: Cluster Confusion Matrix")
    print("=" * 70)

    # Load data
    print("\n[1] Loading confusion matrix...")
    try:
        cm, class_names = load_confusion_matrix()
        cm = reorder_confusion_matrix(cm, class_names)
        print(f"  Loaded {cm.sum():,} samples")
    except FileNotFoundError:
        print("  Pre-computed file not found. Run comprehensive_statistics.py first.")
        return

    # Print confusion matrix
    print("\n[2] Confusion Matrix:")
    print(f"      {''.join([f'{c:>12}' for c in CLUSTER_ORDER])}")
    for i, row_name in enumerate(CLUSTER_ORDER):
        row = ''.join([f'{cm[i,j]:>12,}' for j in range(3)])
        print(f"  {row_name:<8}{row}")

    # Error analysis
    print("\n[3] Error Analysis:")
    analysis = compute_error_analysis(cm, COST_MATRIX)

    print(f"    Total samples: {analysis['total_samples']:,}")
    print(f"    Accuracy: {analysis['accuracy']*100:.2f}%")
    print(f"    Under-sampling errors: {analysis['under_sampling_errors']:,} ({analysis['under_sampling_rate']*100:.2f}%)")
    print(f"    Over-sampling errors: {analysis['over_sampling_errors']:,} ({analysis['over_sampling_rate']*100:.2f}%)")
    print(f"    Critical errors (ACTIVE->STABLE): {analysis['critical_errors']:,} ({analysis['critical_rate']*100:.2f}%)")
    print(f"    Normalized weighted error: {analysis['normalized_weighted_error']*100:.2f}%")

    # Create figures
    print("\n[4] Creating figures...")
    create_confusion_matrix_figure(cm, COST_MATRIX)
    create_behavior_confusion_matrix()

    # Save analysis
    with open(OUTPUT_DIR / 'confusion_matrix_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"\nSaved: {OUTPUT_DIR / 'confusion_matrix_analysis.json'}")


if __name__ == "__main__":
    main()
