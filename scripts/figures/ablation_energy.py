#!/usr/bin/env python3
"""Ablation study energy breakdown figure."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path("L:/GitHub/LiveEdge/outputs/paper_revision/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PAPER_FIGURES_DIR = Path("L:/GitHub/LiveEdge/docs/paper/figures")
PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

VERIFIED_RESULTS = Path("L:/GitHub/LiveEdge/outputs/verified_simulation/verified_results.json")

# Load ablation data from verified_results.json
with open(VERIFIED_RESULTS, 'r') as f:
    data = json.load(f)

ablation = data['ablation']

# Configuration names
configs = {
    'A0': 'Baseline\n(50Hz, 100% inf)',
    'A1': 'Adaptive ODR\nonly',
    'A2': 'Adaptive Inf.\nonly',
    'A3': 'LiveEdge\n(Both)',
}

# Extract data
config_names = ['A0', 'A1', 'A2', 'A3']
e_imu = [ablation[c]['e_imu'] for c in config_names]
e_mcu = [ablation[c]['e_mcu'] for c in config_names]
e_total = [ablation[c]['e_total'] for c in config_names]

# BLE + overhead (constant across all configs)
e_overhead = 589  # mJ/day (from recalculate_energy_unified.py)

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(config_names))
width = 0.6

# Stacked bars
p1 = ax.bar(x, e_imu, width, label='IMU', color='#3498db')
p2 = ax.bar(x, e_mcu, width, bottom=e_imu, label='MCU', color='#e74c3c')
p3 = ax.bar(x, [e_overhead]*len(config_names), width,
            bottom=[e_imu[i] + e_mcu[i] for i in range(len(config_names))],
            label='Overhead (BLE+RTC)', color='#95a5a6')

# Add total energy labels on top
for i, (e_i, e_m, e_t) in enumerate(zip(e_imu, e_mcu, e_total)):
    ax.text(i, e_t + 200, f'{e_t:,} mJ', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add component percentages
    imu_pct = (e_i / e_t) * 100
    mcu_pct = (e_m / e_t) * 100
    oh_pct = (e_overhead / e_t) * 100

    # IMU label
    if imu_pct > 10:
        ax.text(i, e_i/2, f'{imu_pct:.0f}%', ha='center', va='center',
                fontsize=9, color='white', fontweight='bold')

    # MCU label
    if mcu_pct > 5:
        ax.text(i, e_i + e_m/2, f'{mcu_pct:.0f}%', ha='center', va='center',
                fontsize=9, color='white', fontweight='bold')

# Styling
ax.set_ylabel('Daily Energy Consumption (mJ)', fontsize=12)
ax.set_xlabel('Configuration', fontsize=12)
ax.set_title('Energy Breakdown by Component (Ablation Study)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([configs[c] for c in config_names], fontsize=11)
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 11000)

# Add savings annotations
savings_a1 = ((e_total[0] - e_total[1]) / e_total[0]) * 100
savings_a2 = ((e_total[0] - e_total[2]) / e_total[0]) * 100
savings_a3 = ((e_total[0] - e_total[3]) / e_total[0]) * 100

ax.annotate(f'-{savings_a1:.1f}%', xy=(1, e_total[1]), xytext=(1, e_total[0]),
            ha='center', va='bottom', fontsize=10, color='green', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

ax.annotate(f'-{savings_a2:.1f}%', xy=(2, e_total[2]), xytext=(2, e_total[0]),
            ha='center', va='bottom', fontsize=10, color='green', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

ax.annotate(f'-{savings_a3:.1f}%', xy=(3, e_total[3]), xytext=(3, e_total[0]),
            ha='center', va='bottom', fontsize=10, color='green', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

plt.tight_layout()

# Save
output_paths = [
    OUTPUT_DIR / "ablation_energy.pdf",
    OUTPUT_DIR / "ablation_energy.png",
    PAPER_FIGURES_DIR / "ablation_energy.pdf",
    PAPER_FIGURES_DIR / "ablation_energy.png",
]

for path in output_paths:
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved: {path}")

# Save data
ablation_data = {
    'configs': configs,
    'e_imu': e_imu,
    'e_mcu': e_mcu,
    'e_overhead': e_overhead,
    'e_total': e_total,
    'savings': {
        'A1': savings_a1,
        'A2': savings_a2,
        'A3': savings_a3,
    }
}

output_json = OUTPUT_DIR / "ablation_energy_data.json"
with open(output_json, 'w') as f:
    json.dump(ablation_data, f, indent=2)
print(f"Saved: {output_json}")

print("\n=== Ablation Energy Breakdown ===")
for i, c in enumerate(config_names):
    print(f"{configs[c]:20s}: IMU={e_imu[i]:4d} + MCU={e_mcu[i]:4d} + OH={e_overhead:3d} = {e_total[i]:5d} mJ")
print(f"\nSavings: A1={savings_a1:.1f}%, A2={savings_a2:.1f}%, A3={savings_a3:.1f}%")
