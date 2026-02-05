#!/usr/bin/env python3
"""k-sensitivity analysis figure."""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json

OUTPUT_DIR = Path("L:/GitHub/LiveEdge/outputs/paper_revision/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VERIFIED_RESULTS = Path("L:/GitHub/LiveEdge/outputs/verified_simulation/verified_results.json")


def load_verified_results():
    """Load results from verified_simulation.py (preserves temporal ordering)."""
    with open(VERIFIED_RESULTS, 'r') as f:
        data = json.load(f)
    return data


def create_k_sensitivity_figure(data):
    """Create the k-sensitivity figure using verified results."""

    # Extract data for k=1,2,3 from verified_results
    k_values = []
    behavior_acc = []
    cluster_acc = []
    energy = []
    battery = []
    savings = []

    for k_key in ['k1', 'k2', 'k3']:
        if k_key in data['liveedge']:
            k = int(k_key[1])
            k_values.append(k)
            behavior_acc.append(data['liveedge'][k_key]['accuracy_mean'] * 100)
            cluster_acc.append(data['liveedge'][k_key]['cluster_accuracy_mean'] * 100)
            energy.append(data['liveedge'][k_key]['energy_mean'])
            battery.append(data['liveedge'][k_key]['battery_mean'])
            savings.append(data['liveedge'][k_key]['savings_vs_50hz'])

    k_values = np.array(k_values)
    behavior_acc = np.array(behavior_acc)
    cluster_acc = np.array(cluster_acc)
    energy = np.array(energy)
    battery = np.array(battery)
    savings = np.array(savings)

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Accuracy vs k
    ax1 = axes[0, 0]
    ax1.plot(k_values, cluster_acc, 'o-', color='#2ecc71', linewidth=2,
             markersize=10, label='Cluster Accuracy')
    ax1.plot(k_values, behavior_acc, 's--', color='#3498db', linewidth=2,
             markersize=10, label='Behavior Accuracy')
    ax1.axhline(y=data['cluster_accuracy_at_50hz']['mean'] * 100,
                color='gray', linestyle=':', alpha=0.7, label='50Hz Ceiling (84.14%)')
    ax1.set_xlabel('k (stability threshold)', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('(a) Accuracy vs k', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(k_values)
    ax1.set_ylim([78, 86])

    # Annotate optimal point
    best_k_idx = np.argmax(cluster_acc)
    ax1.annotate(f'Best: k={k_values[best_k_idx]}\n{cluster_acc[best_k_idx]:.1f}%',
                xy=(k_values[best_k_idx], cluster_acc[best_k_idx]),
                xytext=(k_values[best_k_idx]+0.3, cluster_acc[best_k_idx]-1.5),
                fontsize=10, ha='left',
                arrowprops=dict(arrowstyle='->', color='#2ecc71'))

    # Plot 2: Energy vs k
    ax2 = axes[0, 1]
    ax2.plot(k_values, energy, 'o-', color='#e74c3c', linewidth=2, markersize=10)
    ax2.axhline(y=data['baselines']['baseline_50hz']['energy'],
                color='gray', linestyle=':', alpha=0.7, label='50Hz Baseline (15,331 mJ)')
    ax2.set_xlabel('k (stability threshold)', fontsize=12)
    ax2.set_ylabel('Energy (mJ/day)', fontsize=12)
    ax2.set_title('(b) Energy Consumption vs k', fontsize=14)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(k_values)

    # Plot 3: Battery Life vs k
    ax3 = axes[1, 0]
    ax3.plot(k_values, battery, 'o-', color='#9b59b6', linewidth=2, markersize=10)
    ax3.axhline(y=data['baselines']['baseline_50hz']['battery'],
                color='gray', linestyle=':', alpha=0.7, label='50Hz Baseline (124 days)')
    ax3.set_xlabel('k (stability threshold)', fontsize=12)
    ax3.set_ylabel('Battery Life (days)', fontsize=12)
    ax3.set_title('(c) Battery Life vs k', fontsize=14)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(k_values)

    # Plot 4: Energy Savings vs k
    ax4 = axes[1, 1]
    ax4.bar(k_values, savings, color='#f39c12', alpha=0.8, edgecolor='black')
    ax4.set_xlabel('k (stability threshold)', fontsize=12)
    ax4.set_ylabel('Energy Savings vs 50Hz (%)', fontsize=12)
    ax4.set_title('(d) Energy Savings vs k', fontsize=14)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_xticks(k_values)
    ax4.set_ylim([0, 80])

    # Add value labels on bars
    for i, (k, s) in enumerate(zip(k_values, savings)):
        ax4.text(k, s + 2, f'{s:.1f}%', ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / 'k_sensitivity.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'k_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {OUTPUT_DIR / 'k_sensitivity.pdf'}")
    print(f"Saved: {OUTPUT_DIR / 'k_sensitivity.png'}")


def generate_latex_table(data):
    """Generate extended Table 6 with all k-sensitivity metrics."""

    latex = r"""\begin{table}[h]
\centering
\caption{k-Sensitivity Analysis (Extended Table 6)}
\label{tab:k_sensitivity_extended}
\begin{tabular}{lccccc}
\toprule
k & Behavior Acc & Cluster Acc & Energy (mJ) & Battery (days) & Savings \\
\midrule
"""

    for k_key in ['k1', 'k2', 'k3']:
        if k_key in data['liveedge']:
            k = int(k_key[1])
            d = data['liveedge'][k_key]
            latex += f"{k} & {d['accuracy_mean']*100:.2f}\\% & {d['cluster_accuracy_mean']*100:.2f}\\% & "
            latex += f"{d['energy_mean']:.0f} & {d['battery_mean']:.0f} & {d['savings_vs_50hz']:.1f}\\% \\\\\n"

    latex += r"""\midrule
50Hz & 80.88\% & 84.14\% & 15,331 & 124 & -- \\
\bottomrule
\end{tabular}
\end{table}
"""

    with open(OUTPUT_DIR / 'table_6_extended.tex', 'w', encoding='utf-8') as f:
        f.write(latex)

    print(f"Saved: {OUTPUT_DIR / 'table_6_extended.tex'}")


def main():
    print("=" * 70)
    print("Figure G6: k-Sensitivity Multi-Metric Analysis")
    print("Using verified_simulation results (preserves temporal ordering)")
    print("=" * 70)

    # Load verified results
    print("\n[1] Loading verified_simulation results...")
    data = load_verified_results()

    # Print results
    print("\n[2] k-Sensitivity Results (CORRECT - temporal order preserved):")
    print(f"\n    k | Behavior Acc | Cluster Acc |   Energy |  Battery |  Savings")
    print("  " + "-" * 70)

    for k_key in ['k1', 'k2', 'k3']:
        if k_key in data['liveedge']:
            k = int(k_key[1])
            d = data['liveedge'][k_key]
            print(f"    {k} |      {d['accuracy_mean']*100:.2f}% |      {d['cluster_accuracy_mean']*100:.2f}% | "
                  f"{d['energy_mean']:7.0f} |    {d['battery_mean']:.0f}d |   {d['savings_vs_50hz']:.1f}%")

    print(f"\n  50Hz baseline: 84.14% cluster acc, 15,331 mJ, 124 days")

    # Create figure
    print("\n[3] Creating figure...")
    create_k_sensitivity_figure(data)

    # Generate LaTeX table
    print("\n[4] Generating LaTeX table...")
    generate_latex_table(data)

    # Save JSON data
    output_data = {
        'source': 'verified_simulation (temporal order preserved)',
        'note': 'CV-based FSM simulation gives incorrect results for k>1 due to shuffled temporal order',
        'k_sensitivity': {},
        'baseline_50hz': data['baselines']['baseline_50hz'],
    }

    for k_key in ['k1', 'k2', 'k3']:
        if k_key in data['liveedge']:
            output_data['k_sensitivity'][k_key] = data['liveedge'][k_key]

    with open(OUTPUT_DIR / 'k_sensitivity_data.json', 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved: {OUTPUT_DIR / 'k_sensitivity_data.json'}")


if __name__ == "__main__":
    main()
