"""Visualization utilities for power decomposition analysis.

This module provides plotting functions for analyzing and visualizing
system-level power consumption.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def set_plot_style():
    """Set consistent plot style."""
    if HAS_MATPLOTLIB:
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'figure.dpi': 100,
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 13,
            'legend.fontsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'lines.linewidth': 2,
            'axes.grid': True,
            'grid.alpha': 0.3,
        })
        if HAS_SEABORN:
            sns.set_palette("husl")


def plot_power_breakdown_pie(
    breakdown: dict[str, float],
    title: str = "System Power Breakdown",
    save_path: str | Path | None = None,
) -> Any:
    """Create a pie chart showing power breakdown by component.

    Args:
        breakdown: Dictionary with component names and percentages.
        title: Plot title.
        save_path: Path to save figure.

    Returns:
        matplotlib Figure object.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return None

    set_plot_style()

    # Filter out zero/tiny values
    filtered = {k: v for k, v in breakdown.items() if v > 0.5}

    # Define colors for components
    colors = {
        'imu': '#2ecc71',       # Green
        'mcu_sampling': '#3498db',  # Blue
        'mcu_inference': '#9b59b6',  # Purple
        'mcu_idle': '#95a5a6',      # Gray
        'radio_tx': '#e74c3c',      # Red
        'radio_sleep': '#f39c12',   # Orange
        'system_sleep': '#1abc9c',  # Teal
    }

    # Prepare data
    labels = []
    sizes = []
    plot_colors = []

    for key, value in filtered.items():
        # Clean up label
        label = key.replace('_pct', '').replace('_', ' ').title()
        labels.append(f"{label}\n({value:.1f}%)")
        sizes.append(value)

        # Find matching color
        color_key = key.replace('_pct', '')
        plot_colors.append(colors.get(color_key, '#7f8c8d'))

    fig, ax = plt.subplots(figsize=(10, 8))

    wedges, texts = ax.pie(
        sizes,
        labels=labels,
        colors=plot_colors,
        startangle=90,
        wedgeprops=dict(width=0.7, edgecolor='white', linewidth=2),
    )

    # Add center text
    ax.text(0, 0, 'Power\nBreakdown', ha='center', va='center', fontsize=14, fontweight='bold')

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_odr_power_curve(
    odr_data: list[dict[str, float]],
    title: str = "IMU Power vs Sampling Rate",
    save_path: str | Path | None = None,
) -> Any:
    """Plot power consumption vs ODR curve.

    Args:
        odr_data: List of dicts with 'odr_hz' and 'power_uw' keys.
        title: Plot title.
        save_path: Path to save figure.

    Returns:
        matplotlib Figure object.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return None

    set_plot_style()

    odrs = [d['odr_hz'] for d in odr_data]
    powers = [d['power_uw'] for d in odr_data]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(odrs, powers, 'o-', markersize=8, linewidth=2, color='#3498db')
    ax.fill_between(odrs, powers, alpha=0.2, color='#3498db')

    ax.set_xlabel('Sampling Rate (Hz)')
    ax.set_ylabel('Power (µW)')
    ax.set_title(title, fontweight='bold')

    # Add annotations for key points
    for d in odr_data:
        if d['odr_hz'] in [5, 25, 50, 100]:
            ax.annotate(
                f"{d['power_uw']:.0f} µW",
                xy=(d['odr_hz'], d['power_uw']),
                xytext=(5, 10),
                textcoords='offset points',
                fontsize=9,
            )

    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_configuration_comparison(
    configs: list[dict[str, Any]],
    metric: str = "avg_power_uw",
    title: str = "Configuration Comparison",
    save_path: str | Path | None = None,
) -> Any:
    """Plot bar chart comparing different configurations.

    Args:
        configs: List of configuration results.
        metric: Metric to compare.
        title: Plot title.
        save_path: Path to save figure.

    Returns:
        matplotlib Figure object.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return None

    set_plot_style()

    names = [c.get('name', c.get('sensor_config', 'Unknown')) for c in configs]
    values = [c.get(metric, 0) for c in configs]

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(range(len(names)), values, color='#3498db', edgecolor='white', linewidth=1.5)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f'{value:.1f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=9,
        )

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_energy_breakdown_stacked(
    duty_cycles: list[dict[str, Any]],
    title: str = "Energy Breakdown by Configuration",
    save_path: str | Path | None = None,
) -> Any:
    """Plot stacked bar chart of energy breakdown.

    Args:
        duty_cycles: List of duty cycle results.
        title: Plot title.
        save_path: Path to save figure.

    Returns:
        matplotlib Figure object.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return None

    set_plot_style()

    # Extract data
    names = [dc['name'][:25] for dc in duty_cycles]  # Truncate long names
    components = ['imu_pct', 'mcu_sampling_pct', 'mcu_inference_pct', 'mcu_idle_pct', 'radio_tx_pct', 'radio_sleep_pct']
    labels = ['IMU', 'MCU Sampling', 'MCU Inference', 'MCU Idle', 'Radio TX', 'Radio Sleep']
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#95a5a6', '#e74c3c', '#f39c12']

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(names))
    width = 0.6

    bottom = np.zeros(len(names))

    for component, label, color in zip(components, labels, colors):
        values = [dc.get('breakdown', {}).get(component, 0) for dc in duty_cycles]
        ax.bar(x, values, width, label=label, bottom=bottom, color=color, edgecolor='white', linewidth=0.5)
        bottom += np.array(values)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Energy Percentage (%)')
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_battery_life_comparison(
    configs: list[dict[str, Any]],
    title: str = "Estimated Battery Life",
    save_path: str | Path | None = None,
) -> Any:
    """Plot battery life comparison.

    Args:
        configs: List of configuration results with battery_life_days.
        title: Plot title.
        save_path: Path to save figure.

    Returns:
        matplotlib Figure object.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return None

    set_plot_style()

    names = [c.get('name', 'Unknown')[:25] for c in configs]
    days = [c.get('battery_life_days', 0) for c in configs]

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.barh(range(len(names)), days, color='#27ae60', edgecolor='white', linewidth=1.5)

    # Add value labels
    for bar, value in zip(bars, days):
        width = bar.get_width()
        ax.annotate(
            f'{value:.0f} days',
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0),
            textcoords='offset points',
            ha='left',
            va='center',
            fontsize=10,
            fontweight='bold',
        )

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel('Battery Life (days)')
    ax.set_title(title, fontweight='bold')

    # Add vertical line for reference (e.g., 1 year)
    ax.axvline(x=365, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.7, label='1 Year')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_lora_toa_vs_sf(
    sf_data: list[dict[str, Any]],
    title: str = "LoRa Time-on-Air vs Spreading Factor",
    save_path: str | Path | None = None,
) -> Any:
    """Plot LoRa time-on-air vs spreading factor.

    Args:
        sf_data: List of dicts with 'sf' and 'toa_ms' keys.
        title: Plot title.
        save_path: Path to save figure.

    Returns:
        matplotlib Figure object.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return None

    set_plot_style()

    sfs = [d['sf'] for d in sf_data]
    toas = [d['toa_ms'] for d in sf_data]
    energies = [d.get('energy_uj', 0) for d in sf_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ToA plot
    ax1.bar(sfs, toas, color='#3498db', edgecolor='white', linewidth=1.5)
    ax1.set_xlabel('Spreading Factor')
    ax1.set_ylabel('Time-on-Air (ms)')
    ax1.set_title('Time-on-Air', fontweight='bold')
    ax1.set_yscale('log')

    for i, (sf, toa) in enumerate(zip(sfs, toas)):
        ax1.annotate(f'{toa:.0f}ms', xy=(sf, toa), ha='center', va='bottom', fontsize=9)

    # Energy plot
    ax2.bar(sfs, energies, color='#e74c3c', edgecolor='white', linewidth=1.5)
    ax2.set_xlabel('Spreading Factor')
    ax2.set_ylabel('Energy (µJ)')
    ax2.set_title('TX Energy', fontweight='bold')
    ax2.set_yscale('log')

    for i, (sf, energy) in enumerate(zip(sfs, energies)):
        ax2.annotate(f'{energy:.0f}µJ', xy=(sf, energy), ha='center', va='bottom', fontsize=9)

    fig.suptitle(title, fontweight='bold', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def generate_power_report_figures(
    results: dict[str, Any],
    output_dir: str | Path = "outputs/power_decomposition/figures",
) -> list[Path]:
    """Generate all figures for power decomposition report.

    Args:
        results: Full experiment results dictionary.
        output_dir: Directory to save figures.

    Returns:
        List of saved figure paths.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, cannot generate figures")
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    scenarios = {s['scenario_id']: s for s in results.get('scenarios', [])}

    # T1: System standby
    if 'T1' in scenarios:
        t1 = scenarios['T1']['results']
        components = t1.get('components', {})
        breakdown = {
            f"{k}_pct": v['percentage']
            for k, v in components.items()
        }
        path = output_dir / "t1_standby_breakdown.png"
        plot_power_breakdown_pie(breakdown, "T1: System Standby Power Breakdown", path)
        saved_files.append(path)

    # T2: ODR sweep
    if 'T2' in scenarios:
        t2 = scenarios['T2']['results']
        odr_sweep = t2.get('odr_sweep', [])
        if odr_sweep:
            path = output_dir / "t2_odr_power_curve.png"
            plot_odr_power_curve(odr_sweep, "T2: IMU Power vs Sampling Rate", path)
            saved_files.append(path)

    # T5: LoRa SF sweep
    if 'T5' in scenarios:
        t5 = scenarios['T5']['results']
        sf_sweep = t5.get('sf_sweep', [])
        if sf_sweep:
            path = output_dir / "t5_lora_sf_sweep.png"
            plot_lora_toa_vs_sf(sf_sweep, "T5: LoRa Time-on-Air vs Spreading Factor", path)
            saved_files.append(path)

    # T7: Duty cycle comparison
    if 'T7' in scenarios:
        t7 = scenarios['T7']['results']
        duty_cycles = t7.get('duty_cycles', [])
        if duty_cycles:
            path = output_dir / "t7_energy_breakdown_stacked.png"
            plot_energy_breakdown_stacked(duty_cycles, "T7: Energy Breakdown by Configuration", path)
            saved_files.append(path)

            path = output_dir / "t7_battery_life.png"
            plot_battery_life_comparison(duty_cycles, "T7: Estimated Battery Life", path)
            saved_files.append(path)

            path = output_dir / "t7_power_comparison.png"
            plot_configuration_comparison(
                duty_cycles,
                metric='avg_power_uw',
                title="T7: Average Power Comparison",
                save_path=path,
            )
            saved_files.append(path)

    # Sensor configuration comparison
    if 'SENSOR_COMPARE' in scenarios:
        sensor = scenarios['SENSOR_COMPARE']['results']
        configs = sensor.get('configurations', [])
        if configs:
            path = output_dir / "sensor_power_comparison.png"
            plot_configuration_comparison(
                configs,
                metric='power_mw',
                title="Sensor Configuration Power Comparison",
                save_path=path,
            )
            saved_files.append(path)

    print(f"\nGenerated {len(saved_files)} figures in {output_dir}")
    return saved_files
