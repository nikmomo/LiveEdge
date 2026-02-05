#!/usr/bin/env python3
"""
Generate supplementary figures for the LiveEdge paper.

Figures to generate:
1. Pareto chart (Energy vs Accuracy) - supplement to Table 4
2. k-sensitivity line chart - can replace Table 6
3. Classifier comparison bar chart - supplement to Table 7
4. Confusion matrix heatmap - new figure
5. F1 vs ODR curves - visualize fmin determination
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import json

# Output directory
OUTPUT_DIR = Path("L:/GitHub/LiveEdge/docs/paper/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set style for publication
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# =============================================================================
# Data from paper (verified experimental results)
# =============================================================================

# Main results (Table 4)
MAIN_RESULTS = {
    'B1_50Hz': {'energy': 17600, 'battery': 108, 'cluster_acc': 73.75, 'behavior_acc': 71.24, 'odr': 50.0},
    'B2_25Hz': {'energy': 15008, 'battery': 127, 'cluster_acc': 72.89, 'behavior_acc': 70.73, 'odr': 25.0},
    'B3_12.5Hz': {'energy': 13194, 'battery': 144, 'cluster_acc': 71.94, 'behavior_acc': 69.78, 'odr': 12.5},
    'B4_10Hz': {'energy': 13194, 'battery': 144, 'cluster_acc': 71.27, 'behavior_acc': 69.11, 'odr': 10.0},
    'B5_6.25Hz': {'energy': 12416, 'battery': 153, 'cluster_acc': 70.02, 'behavior_acc': 67.86, 'odr': 6.25},
    'LiveEdge': {'energy': 5266, 'battery': 361, 'cluster_acc': 79.92, 'behavior_acc': 70.09, 'odr': 11.3},
}

# k-sensitivity (Table 6)
K_SENSITIVITY = {
    1: {'transitions': 10995, 'savings': 66.0, 'delay': 2.6},
    2: {'transitions': 4634, 'savings': 68.2, 'delay': 2.7},
    3: {'transitions': 1701, 'savings': 68.6, 'delay': 3.1},
    4: {'transitions': 679, 'savings': 68.1, 'delay': 3.0},
    5: {'transitions': 303, 'savings': 67.9, 'delay': 2.9},
}

# Classifier comparison (Table 7)
CLASSIFIERS = {
    'Random Forest': {'behavior': 77.98, 'cluster': 82.37, 'f1': 60.48, 'mcu': True},
    'Decision Tree': {'behavior': 72.21, 'cluster': 78.39, 'f1': 51.23, 'mcu': True},
    'SVM (Linear)': {'behavior': 69.82, 'cluster': 72.45, 'f1': 52.41, 'mcu': True},
    'Logistic Reg.': {'behavior': 65.34, 'cluster': 68.91, 'f1': 49.18, 'mcu': True},
    '1D-CNN': {'behavior': 80.91, 'cluster': 86.78, 'f1': 66.11, 'mcu': False},
    'DeepConvLSTM': {'behavior': 80.46, 'cluster': 86.63, 'f1': 64.67, 'mcu': False},
    'GRU': {'behavior': 80.18, 'cluster': 86.30, 'f1': 64.21, 'mcu': False},
    'ResNet-1D': {'behavior': 80.23, 'cluster': 86.07, 'f1': 65.02, 'mcu': False},
    'LSTM': {'behavior': 79.83, 'cluster': 86.03, 'f1': 63.25, 'mcu': False},
    'Transformer': {'behavior': 76.12, 'cluster': 80.73, 'f1': 58.62, 'mcu': False},
}

# Ablation study (Table 5)
ABLATION = {
    'A0': {'name': 'Baseline\n(50Hz, 100%)', 'e_imu': 6998, 'e_mcu': 10013, 'e_total': 17600},
    'A1': {'name': 'Adaptive ODR\nonly', 'e_imu': 2396, 'e_mcu': 10013, 'e_total': 12998},
    'A2': {'name': 'Adaptive Inf.\nonly', 'e_imu': 6998, 'e_mcu': 3281, 'e_total': 10868},
    'A3': {'name': 'LiveEdge\n(Both)', 'e_imu': 2396, 'e_mcu': 2281, 'e_total': 5266},
}

# Confusion matrix data (from k_sensitivity_results.json)
CONFUSION_MATRIX = np.array([
    [93.0, 6.2, 0.7],   # STABLE: correctly predicted, to MODERATE, to ACTIVE
    [5.4, 86.5, 8.1],   # MODERATE
    [3.1, 49.3, 47.6],  # ACTIVE (note: many misclassified as MODERATE)
])

# F1 vs ODR data (from fmin analysis)
F1_VS_ODR = {
    'Lying': {50: 93.93, 25: 92.81, 12.5: 91.62, 6.25: 90.48},
    'Eating': {50: 79.28, 25: 78.45, 12.5: 77.01, 6.25: 72.13},
    'Standing': {50: 41.61, 25: 40.24, 12.5: 36.12, 6.25: 28.45},
    'Walking': {50: 52.38, 25: 50.45, 12.5: 45.23, 6.25: 38.67},
    'Drinking': {50: 44.33, 25: 44.00, 12.5: 41.52, 6.25: 35.21},
}


# =============================================================================
# Figure 1: Pareto Chart (Energy vs Accuracy)
# =============================================================================

def create_pareto_chart():
    """Create Pareto chart showing energy-accuracy trade-off."""
    fig, ax = plt.subplots(figsize=(5, 4))

    # Plot baselines
    for name, data in MAIN_RESULTS.items():
        if name == 'LiveEdge':
            continue
        ax.scatter(data['energy'] / 1000, data['cluster_acc'],
                   s=80, c='#1f77b4', marker='o', alpha=0.7, zorder=2)
        # Label with adjusted positions to avoid overlap
        if name == 'B5_6.25Hz':
            offset = (5, -12)
        elif name == 'B4_10Hz':
            offset = (-45, -12)  # Move to left side
        elif name == 'B3_12.5Hz':
            offset = (5, 8)
        elif name == 'B2_25Hz':
            offset = (5, -12)
        else:
            offset = (5, 5)
        ax.annotate(name.replace('_', ' '),
                    (data['energy'] / 1000, data['cluster_acc']),
                    xytext=offset, textcoords='offset points',
                    fontsize=8, ha='left')

    # Plot LiveEdge (highlighted)
    le = MAIN_RESULTS['LiveEdge']
    ax.scatter(le['energy'] / 1000, le['cluster_acc'],
               s=200, c='#d62728', marker='*', zorder=3,
               label='LiveEdge', edgecolors='black', linewidths=0.5)
    ax.annotate('LiveEdge', (le['energy'] / 1000, le['cluster_acc']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=9, fontweight='bold', ha='left')

    ax.set_xlabel('Daily Energy Consumption (J/day)')
    ax.set_ylabel('Runtime Cluster Accuracy (%)')
    ax.set_xlim(3, 20)
    ax.set_ylim(68, 82)
    ax.grid(True, alpha=0.3)

    # Legend
    baseline_patch = plt.Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor='#1f77b4', markersize=8,
                                 label='Fixed-rate baselines')
    liveedge_patch = plt.Line2D([0], [0], marker='*', color='w',
                                 markerfacecolor='#d62728', markersize=12,
                                 label='LiveEdge (ours)')
    ax.legend(handles=[baseline_patch, liveedge_patch], loc='lower left')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'pareto_energy_accuracy.pdf')
    plt.savefig(OUTPUT_DIR / 'pareto_energy_accuracy.png')
    plt.close()
    print(f"Created: pareto_energy_accuracy.png")


# =============================================================================
# Figure 2: k-Sensitivity (replaces Table 6)
# =============================================================================

def create_k_sensitivity_chart():
    """Create k-sensitivity chart with dual y-axes."""
    fig, ax1 = plt.subplots(figsize=(5, 3.5))

    k_values = list(K_SENSITIVITY.keys())
    transitions = [K_SENSITIVITY[k]['transitions'] for k in k_values]
    savings = [K_SENSITIVITY[k]['savings'] for k in k_values]

    # Left axis: Transitions (log scale)
    color1 = '#1f77b4'
    ax1.semilogy(k_values, transitions, 'o-', color=color1, linewidth=2,
                  markersize=8, label='Transitions/day')
    ax1.set_xlabel('Stability Threshold $k$')
    ax1.set_ylabel('Daily State Transitions', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(100, 20000)

    # Right axis: Energy savings
    ax2 = ax1.twinx()
    color2 = '#d62728'
    ax2.plot(k_values, savings, 's--', color=color2, linewidth=2,
             markersize=8, label='Energy Savings')
    ax2.set_ylabel('Energy Savings (%)', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(64, 70)

    # Highlight k=3
    ax1.axvline(x=3, color='green', linestyle=':', alpha=0.7, linewidth=1.5)
    ax1.annotate('$k$=3\n(selected)', xy=(3, 1701), xytext=(3.5, 4000),
                 fontsize=9, color='green',
                 arrowprops=dict(arrowstyle='->', color='green', lw=1))

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    ax1.set_xticks(k_values)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'k_sensitivity.pdf')
    plt.savefig(OUTPUT_DIR / 'k_sensitivity.png')
    plt.close()
    print(f"Created: k_sensitivity.pdf")


# =============================================================================
# Figure 3: Classifier Comparison Bar Chart
# =============================================================================

def create_classifier_comparison_chart():
    """Create grouped bar chart for classifier comparison."""
    fig, ax = plt.subplots(figsize=(7, 4))

    # Separate MCU-compatible and GPU-required
    mcu_names = [n for n, d in CLASSIFIERS.items() if d['mcu']]
    gpu_names = [n for n, d in CLASSIFIERS.items() if not d['mcu']]
    all_names = mcu_names + gpu_names

    x = np.arange(len(all_names))
    width = 0.35

    behavior_acc = [CLASSIFIERS[n]['behavior'] for n in all_names]
    cluster_acc = [CLASSIFIERS[n]['cluster'] for n in all_names]

    # Create bars
    bars1 = ax.bar(x - width/2, behavior_acc, width, label='Behavior Accuracy',
                   color='#1f77b4', alpha=0.8)
    bars2 = ax.bar(x + width/2, cluster_acc, width, label='Cluster Accuracy',
                   color='#ff7f0e', alpha=0.8)

    # Highlight Random Forest
    bars1[0].set_edgecolor('red')
    bars1[0].set_linewidth(2)
    bars2[0].set_edgecolor('red')
    bars2[0].set_linewidth(2)

    # Add separator line between MCU and GPU
    ax.axvline(x=len(mcu_names) - 0.5, color='gray', linestyle='--', alpha=0.5)
    ax.text(len(mcu_names)/2 - 0.5, 88, 'MCU-compatible', ha='center',
            fontsize=9, style='italic', color='green')
    ax.text(len(mcu_names) + len(gpu_names)/2 - 0.5, 88, 'GPU-required',
            ha='center', fontsize=9, style='italic', color='gray')

    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(all_names, rotation=45, ha='right')
    ax.set_ylim(60, 90)
    ax.legend(loc='lower right')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'classifier_comparison.pdf')
    plt.savefig(OUTPUT_DIR / 'classifier_comparison.png')
    plt.close()
    print(f"Created: classifier_comparison.pdf")


# =============================================================================
# Figure 4: Confusion Matrix Heatmap
# =============================================================================

def create_confusion_matrix():
    """Create confusion matrix heatmap for cluster classification."""
    fig, ax = plt.subplots(figsize=(4, 3.5))

    clusters = ['STABLE', 'MODERATE', 'ACTIVE']

    # Create heatmap without grid lines
    im = ax.imshow(CONFUSION_MATRIX, cmap='Blues', vmin=0, vmax=100)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Percentage (%)', rotation=-90, va="bottom")

    # Add text annotations
    for i in range(3):
        for j in range(3):
            value = CONFUSION_MATRIX[i, j]
            color = 'white' if value > 50 else 'black'
            ax.text(j, i, f'{value:.1f}%', ha='center', va='center',
                    color=color, fontsize=10)

    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(clusters)
    ax.set_yticklabels(clusters)
    ax.set_xlabel('Predicted Cluster')
    ax.set_ylabel('True Cluster')

    # Remove grid, keep only outer border
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('black')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confusion_matrix.pdf')
    plt.savefig(OUTPUT_DIR / 'confusion_matrix.png')
    plt.close()
    print(f"Created: confusion_matrix.png")


# =============================================================================
# Figure 5: F1 vs ODR (fmin visualization)
# =============================================================================

def create_f1_vs_odr_chart():
    """Create F1 vs ODR chart showing fmin determination."""
    fig, ax = plt.subplots(figsize=(6, 4))

    odrs = [6.25, 12.5, 25, 50]
    colors = {'Lying': '#2ca02c', 'Eating': '#ff7f0e',
              'Standing': '#d62728', 'Walking': '#9467bd', 'Drinking': '#8c564b'}
    markers = {'Lying': 'o', 'Eating': 's', 'Standing': '^',
               'Walking': 'D', 'Drinking': 'v'}

    for behavior, data in F1_VS_ODR.items():
        f1_values = [data[odr] for odr in odrs]
        ax.plot(odrs, f1_values, marker=markers[behavior],
                color=colors[behavior], linewidth=1.5, markersize=6,
                label=behavior)

    ax.set_xlabel('Sampling Rate (Hz)')
    ax.set_ylabel('F1 Score (%)')
    ax.set_xscale('log', base=2)
    ax.set_xticks(odrs)
    ax.set_xticklabels(['6.25', '12.5', '25', '50'])
    ax.set_xlim(5, 60)
    ax.set_ylim(25, 100)
    ax.legend(loc='center right', fontsize=8, bbox_to_anchor=(1.0, 0.5))
    ax.grid(True, alpha=0.3)

    # Add cluster background colors
    ax.axvspan(5, 9, alpha=0.08, color='green')
    ax.axvspan(9, 18, alpha=0.08, color='orange')
    ax.axvspan(18, 60, alpha=0.08, color='red')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'f1_vs_odr.pdf')
    plt.savefig(OUTPUT_DIR / 'f1_vs_odr.png')
    plt.close()
    print(f"Created: f1_vs_odr.png")


# =============================================================================
# Figure 6: Ablation Energy Breakdown (Stacked Bar)
# =============================================================================

def create_ablation_chart():
    """Create stacked bar chart for ablation study."""
    fig, ax = plt.subplots(figsize=(5, 4))

    configs = list(ABLATION.keys())
    names = [ABLATION[c]['name'] for c in configs]
    e_imu = [ABLATION[c]['e_imu'] / 1000 for c in configs]  # Convert to J
    e_mcu = [ABLATION[c]['e_mcu'] / 1000 for c in configs]

    x = np.arange(len(configs))
    width = 0.6

    # Stacked bars
    bars1 = ax.bar(x, e_imu, width, label='$E_{IMU}$', color='#1f77b4')
    bars2 = ax.bar(x, e_mcu, width, bottom=e_imu, label='$E_{MCU}$', color='#ff7f0e')

    # Add total energy labels on top
    for i, (imu, mcu) in enumerate(zip(e_imu, e_mcu)):
        total = imu + mcu
        ax.text(i, total + 0.3, f'{total:.1f}J', ha='center', fontsize=9)

    ax.set_ylabel('Daily Energy (J/day)')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 20)
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ablation_energy.pdf')
    plt.savefig(OUTPUT_DIR / 'ablation_energy.png')
    plt.close()
    print(f"Created: ablation_energy.png")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Generating paper figures...")
    print("=" * 60)
    print()

    create_pareto_chart()
    create_k_sensitivity_chart()
    create_classifier_comparison_chart()
    create_confusion_matrix()
    create_f1_vs_odr_chart()
    create_ablation_chart()

    print()
    print("=" * 60)
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("=" * 60)
