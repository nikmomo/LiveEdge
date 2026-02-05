#!/usr/bin/env python3
"""k-sensitivity analysis figure."""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json

OUTPUT_DIR = Path("L:/GitHub/LiveEdge/outputs/paper_revision/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PAPER_FIGURES_DIR = Path("L:/GitHub/LiveEdge/docs/paper/figures")
PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

K_SENSITIVITY_RESULTS = Path("L:/GitHub/LiveEdge/outputs/k_sensitivity_analysis/k_sensitivity_results.json")
BATTERY_CAPACITY = 220 * 3.0 * 3600 * 0.8  # mJ (CR2032, 80% usable)


def load_k_sensitivity_results():
    """Load results from k_sensitivity_and_error_analysis.py."""
    with open(K_SENSITIVITY_RESULTS, 'r') as f:
        data = json.load(f)
    return data


def create_k_sensitivity_figure(data):
    """Create the k-sensitivity figure using k-sensitivity analysis results."""

    # Extract data for k=1 to 6
    k_results = data['k_sensitivity']['k_results']
    k_values = []
    behavior_acc = []
    cluster_acc = []
    energy = []
    battery = []
    savings = []
    transitions = []
    delay = []

    for k_str in sorted(k_results.keys(), key=int):
        k_data = k_results[k_str]
        k_values.append(k_data['k'])
        behavior_acc.append(k_data['behavior_accuracy'] * 100)
        cluster_acc.append(k_data['cluster_accuracy'] * 100)
        energy.append(k_data['total_energy'])
        battery.append(BATTERY_CAPACITY / k_data['total_energy'])
        savings.append(k_data['total_savings_pct'])
        transitions.append(k_data['transitions_per_day'])
        delay.append(k_data['mean_delay_s'])

    k_values = np.array(k_values)
    behavior_acc = np.array(behavior_acc)
    cluster_acc = np.array(cluster_acc)
    energy = np.array(energy)
    battery = np.array(battery)
    savings = np.array(savings)
    transitions = np.array(transitions)
    delay = np.array(delay)

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Accuracy vs k
    ax1 = axes[0, 0]
    ax1.plot(k_values, cluster_acc, 'o-', color='#2ecc71', linewidth=2,
             markersize=10, label='Cluster Accuracy')
    ax1.plot(k_values, behavior_acc, 's--', color='#3498db', linewidth=2,
             markersize=10, label='Behavior Accuracy')
    ax1.set_xlabel('k (stability threshold)', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('(a) Accuracy vs k', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(k_values)
    ax1.set_ylim([56, 70])

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
    baseline_energy = 8796  # mJ (from verified_results.json)
    ax2.axhline(y=baseline_energy, color='gray', linestyle=':', alpha=0.7,
                label=f'50Hz Baseline ({baseline_energy} mJ)')
    ax2.set_xlabel('k (stability threshold)', fontsize=12)
    ax2.set_ylabel('Energy (mJ/day)', fontsize=12)
    ax2.set_title('(b) Energy Consumption vs k', fontsize=14)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(k_values)

    # Plot 3: Transitions/day vs k
    ax3 = axes[1, 0]
    ax3.plot(k_values, transitions, 'o-', color='#9b59b6', linewidth=2, markersize=10)
    ax3.set_xlabel('k (stability threshold)', fontsize=12)
    ax3.set_ylabel('Transitions per Day', fontsize=12)
    ax3.set_title('(c) State Transitions vs k', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(k_values)
    ax3.set_yscale('log')

    # Plot 4: Detection Delay vs k
    ax4 = axes[1, 1]
    ax4.plot(k_values, delay, 'o-', color='#f39c12', linewidth=2, markersize=10)
    ax4.set_xlabel('k (stability threshold)', fontsize=12)
    ax4.set_ylabel('Mean Detection Delay (s)', fontsize=12)
    ax4.set_title('(d) Detection Delay vs k', fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(k_values)

    plt.tight_layout()

    # Save to both directories
    for out_dir in [OUTPUT_DIR, PAPER_FIGURES_DIR]:
        fig.savefig(out_dir / 'k_sensitivity.pdf', dpi=300, bbox_inches='tight')
        fig.savefig(out_dir / 'k_sensitivity.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {out_dir / 'k_sensitivity.pdf'}")
        print(f"Saved: {out_dir / 'k_sensitivity.png'}")
    plt.close()

    print(f"Saved: {OUTPUT_DIR / 'k_sensitivity.pdf'}")
    print(f"Saved: {OUTPUT_DIR / 'k_sensitivity.png'}")


def generate_latex_table(data):
    """Generate extended Table 6 with all k-sensitivity metrics."""

    latex = r"""\begin{table}[h]
\centering
\caption{k-Sensitivity Analysis (Extended Table 6)}
\label{tab:k_sensitivity_extended}
\begin{tabular}{lcccccc}
\toprule
k & Behavior & Cluster & Energy & Savings & Trans/day & Delay \\
  & Acc (\%) & Acc (\%) & (mJ) & (\%) & & (s) \\
\midrule
"""

    k_results = data['k_sensitivity']['k_results']
    for k_str in sorted(k_results.keys(), key=int):
        d = k_results[k_str]
        latex += f"{d['k']} & {d['behavior_accuracy']*100:.1f} & {d['cluster_accuracy']*100:.1f} & "
        latex += f"{d['total_energy']:,} & {d['total_savings_pct']:.1f} & "
        latex += f"{d['transitions_per_day']:,} & {d['mean_delay_s']:.1f} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open(OUTPUT_DIR / 'table_6_extended.tex', 'w', encoding='utf-8') as f:
        f.write(latex)

    print(f"Saved: {OUTPUT_DIR / 'table_6_extended.tex'}")


def main():
    print("=" * 70)
    print("Figure G6: k-Sensitivity Multi-Metric Analysis")
    print("Using k_sensitivity_and_error_analysis results (k=1 to 6)")
    print("=" * 70)

    # Load k-sensitivity results
    print("\n[1] Loading k-sensitivity results...")
    data = load_k_sensitivity_results()

    # Print results
    print("\n[2] k-Sensitivity Results (k=1 to 6, temporal order preserved):")
    print(f"\n    k | Behavior | Cluster |   Energy |  Savings | Trans/d | Delay")
    print("  " + "-" * 75)

    k_results = data['k_sensitivity']['k_results']
    for k_str in sorted(k_results.keys(), key=int):
        d = k_results[k_str]
        battery_days = BATTERY_CAPACITY / d['total_energy']
        print(f"    {d['k']} |   {d['behavior_accuracy']*100:5.1f}% |  {d['cluster_accuracy']*100:5.1f}% | "
              f"{d['total_energy']:7,} |   {d['total_savings_pct']:5.1f}% | {d['transitions_per_day']:7,} | {d['mean_delay_s']:5.1f}s")

    print(f"\n  50Hz baseline: 69.4% behavior, 8,796 mJ, 216 days")

    # Create figure
    print("\n[3] Creating figure...")
    create_k_sensitivity_figure(data)

    # Generate LaTeX table
    print("\n[4] Generating LaTeX table...")
    generate_latex_table(data)

    print(f"\nAll outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
