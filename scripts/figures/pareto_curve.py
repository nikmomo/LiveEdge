#!/usr/bin/env python3
"""Accuracy-energy pareto curve figure."""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent

OUTPUT_DIR = Path("L:/GitHub/LiveEdge/outputs/paper_revision/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load actual verified results
VERIFIED_RESULTS_PATH = _project_root / "outputs/verified_simulation/verified_results.json"

def load_verified_data():
    """Load actual simulation results from verified_results.json."""
    with open(VERIFIED_RESULTS_PATH, 'r') as f:
        data = json.load(f)

    # Extract baselines
    baselines = {}
    for key, val in data['baselines'].items():
        odr_str = key.replace('baseline_', '').replace('hz', 'Hz')
        baselines[odr_str] = {
            'energy': val['energy'],
            'behavior_acc': val['accuracy_mean'] * 100,  # Convert to percentage
            'cluster_acc': val['cluster_accuracy_mapped'] * 100,
            'battery': val['battery'],
        }

    # Extract LiveEdge compressed model results (k=3)
    liveedge = {
        'k=3': {
            'energy': data['liveedge_compressed']['energy_mean'],
            'behavior_acc': data['liveedge_compressed']['accuracy_mean'] * 100,
            'cluster_acc': data['liveedge_compressed']['cluster_accuracy_mean'] * 100,
            'battery': data['liveedge_compressed']['battery_mean'],
        }
    }

    return baselines, liveedge

BASELINES, LIVEEDGE = load_verified_data()

# Signal-driven baselines (placeholder - to be computed from related_work_comparison.py)
# These are NOT actual values, just placeholders for figure layout
SIGNAL_DRIVEN = {}


def create_pareto_figure():
    """Create the Pareto curve figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left plot: Energy vs Accuracy ---
    # Fixed-rate baselines (sorted by energy for frontier)
    baseline_items = sorted(BASELINES.items(), key=lambda x: x[1]['energy'])
    baseline_energies = [data['energy'] for _, data in baseline_items]
    baseline_accs = [data['behavior_acc'] for _, data in baseline_items]

    ax1.scatter(baseline_energies, baseline_accs, marker='s', s=100, c='gray',
                label='Fixed-ODR Baselines', zorder=5)

    for name, data in BASELINES.items():
        ax1.annotate(f'{name}', (data['energy'], data['behavior_acc']),
                     textcoords='offset points', xytext=(5, 5), fontsize=9)

    # LiveEdge
    le_energies = [LIVEEDGE[k]['energy'] for k in LIVEEDGE]
    le_accs = [LIVEEDGE[k]['behavior_acc'] for k in LIVEEDGE]
    ax1.scatter(le_energies, le_accs, marker='*', s=300, c='red',
                label='LiveEdge', zorder=10, edgecolors='black')

    for name, data in LIVEEDGE.items():
        ax1.annotate(f'LiveEdge {name}', (data['energy'], data['behavior_acc']),
                     textcoords='offset points', xytext=(-60, 10), fontsize=9)

    # Pareto frontier (computed from actual data)
    # Include LiveEdge k=3 and fixed baselines
    pareto_points = [(data['energy'], data['behavior_acc']) for data in BASELINES.values()]
    pareto_points.extend([(data['energy'], data['behavior_acc']) for data in LIVEEDGE.values()])
    pareto_points_sorted = sorted(pareto_points, key=lambda x: x[0])

    # Compute actual Pareto frontier (non-dominated points)
    pareto_frontier = []
    max_acc_so_far = -float('inf')
    for energy, acc in pareto_points_sorted:
        if acc > max_acc_so_far:
            pareto_frontier.append((energy, acc))
            max_acc_so_far = acc

    if pareto_frontier:
        px, py = zip(*pareto_frontier)
        ax1.plot(px, py, 'r--', alpha=0.5, linewidth=2, label='Pareto Frontier')

    ax1.set_xlabel('Daily Energy Consumption (mJ)', fontsize=12)
    ax1.set_ylabel('Behavior Classification Accuracy (%)', fontsize=12)
    ax1.set_title('(a) Accuracy vs Energy', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Add arrow showing LiveEdge improvement (computed from actual data)
    le_k3_energy = LIVEEDGE['k=3']['energy']
    le_k3_acc = LIVEEDGE['k=3']['behavior_acc']
    baseline_6_25_energy = BASELINES['6.25Hz']['energy']
    baseline_6_25_acc = BASELINES['6.25Hz']['behavior_acc']

    savings_pct = ((baseline_6_25_energy - le_k3_energy) / baseline_6_25_energy) * 100
    acc_diff = le_k3_acc - baseline_6_25_acc

    ax1.annotate('', xy=(le_k3_energy, le_k3_acc), xytext=(baseline_6_25_energy, baseline_6_25_acc),
                 arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax1.text((le_k3_energy + baseline_6_25_energy) / 2, (le_k3_acc + baseline_6_25_acc) / 2 - 0.5,
             f'{savings_pct:.1f}% less energy\n{acc_diff:+.1f}% accuracy',
             fontsize=10, color='green', ha='center')

    # --- Right plot: Battery Life vs Accuracy ---
    baseline_batteries = [BASELINES[k]['battery'] for k in BASELINES]
    ax2.scatter(baseline_batteries, baseline_accs, marker='s', s=100, c='gray',
                label='Fixed-ODR Baselines', zorder=5)

    for name, data in BASELINES.items():
        ax2.annotate(f'{name}', (data['battery'], data['behavior_acc']),
                     textcoords='offset points', xytext=(5, 5), fontsize=9)

    le_batteries = [LIVEEDGE[k]['battery'] for k in LIVEEDGE]
    ax2.scatter(le_batteries, le_accs, marker='*', s=300, c='red',
                label='LiveEdge', zorder=10, edgecolors='black')

    for name, data in LIVEEDGE.items():
        ax2.annotate(f'LiveEdge {name}\n({data["battery"]:.0f} days)',
                     (data['battery'], data['behavior_acc']),
                     textcoords='offset points', xytext=(-70, 10), fontsize=9)

    # Target: 1 year
    ax2.axvline(x=365, color='green', linestyle='--', alpha=0.7, label='1-Year Target')

    ax2.set_xlabel('Battery Life (days)', fontsize=12)
    ax2.set_ylabel('Behavior Classification Accuracy (%)', fontsize=12)
    ax2.set_title('(b) Accuracy vs Battery Life', fontsize=14)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    fig.savefig(OUTPUT_DIR / 'pareto_curve.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'pareto_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {OUTPUT_DIR / 'pareto_curve.pdf'}")
    print(f"Saved: {OUTPUT_DIR / 'pareto_curve.png'}")


def create_pareto_data_json():
    """Save Pareto curve data as JSON."""
    data = {
        'baselines': BASELINES,
        'liveedge': LIVEEDGE,
        'signal_driven': SIGNAL_DRIVEN,
    }

    with open(OUTPUT_DIR / 'pareto_curve_data.json', 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved: {OUTPUT_DIR / 'pareto_curve_data.json'}")


def main():
    print("=" * 70)
    print("Figure G1: Accuracy-Energy Pareto Curve")
    print("=" * 70)

    create_pareto_figure()
    create_pareto_data_json()


if __name__ == "__main__":
    main()
