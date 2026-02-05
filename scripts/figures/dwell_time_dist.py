#!/usr/bin/env python3
"""Dwell time distribution figure."""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json

OUTPUT_DIR = Path("L:/GitHub/LiveEdge/outputs/paper_revision/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("L:/GitHub/LiveEdge/outputs/paper_revision")

CLUSTER_COLORS = {
    'STABLE': '#2ecc71',    # Green
    'MODERATE': '#f39c12',  # Orange
    'ACTIVE': '#e74c3c',    # Red
}

CLUSTER_ORDER = ['STABLE', 'MODERATE', 'ACTIVE']


def load_dwell_times():
    """Load pre-computed dwell times."""
    with open(DATA_DIR / 'dwell_times_raw.json', 'r') as f:
        data = json.load(f)
    return data


def create_histogram_figure(dwell_times):
    """Create histogram of dwell times for each cluster."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for idx, cluster in enumerate(CLUSTER_ORDER):
        ax = axes[idx]
        times = np.array(dwell_times['clusters'].get(cluster, []))

        if len(times) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{cluster}')
            continue

        # Cap at 300 seconds for visualization
        times_capped = np.clip(times, 0, 300)

        # Histogram
        bins = np.linspace(0, 300, 31)
        ax.hist(times_capped, bins=bins, color=CLUSTER_COLORS[cluster],
                alpha=0.7, edgecolor='black', linewidth=0.5)

        # Statistics
        mean_t = np.mean(times)
        median_t = np.median(times)
        p25 = np.percentile(times, 25)
        p75 = np.percentile(times, 75)

        # Add vertical lines for mean and median
        ax.axvline(mean_t, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_t:.1f}s')
        ax.axvline(median_t, color='blue', linestyle=':', linewidth=2, label=f'Median: {median_t:.1f}s')

        # 30-second threshold reference
        ax.axvline(30, color='green', linestyle='-', alpha=0.7, linewidth=2)
        ax.text(30, ax.get_ylim()[1] * 0.9, '30s threshold', rotation=90,
                va='top', ha='right', fontsize=9, color='green')

        ax.set_xlabel('Dwell Time (seconds)', fontsize=11)
        ax.set_ylabel('Frequency' if idx == 0 else '', fontsize=11)
        ax.set_title(f'{cluster}\n(n={len(times)}, mean={mean_t:.1f}s)', fontsize=12)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_xlim(0, 310)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Cluster Dwell Time Distribution', fontsize=14, y=1.02)
    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / 'dwell_time_histogram.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'dwell_time_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {OUTPUT_DIR / 'dwell_time_histogram.pdf'}")


def create_cdf_figure(dwell_times):
    """Create CDF of dwell times."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for cluster in CLUSTER_ORDER:
        times = np.array(dwell_times['clusters'].get(cluster, []))

        if len(times) == 0:
            continue

        # Sort for CDF
        sorted_times = np.sort(times)
        cdf = np.arange(1, len(sorted_times) + 1) / len(sorted_times)

        ax.plot(sorted_times, cdf, label=f'{cluster} (n={len(times)})',
                color=CLUSTER_COLORS[cluster], linewidth=2)

        # Mark median
        median = np.median(times)
        ax.plot(median, 0.5, 'o', color=CLUSTER_COLORS[cluster], markersize=8)

    # Reference lines
    ax.axvline(30, color='gray', linestyle='--', alpha=0.7, label='30s threshold')
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)

    ax.set_xlabel('Dwell Time (seconds)', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title('Cluster Dwell Time CDF', fontsize=14)
    ax.legend(loc='lower right')
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / 'dwell_time_cdf.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'dwell_time_cdf.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {OUTPUT_DIR / 'dwell_time_cdf.pdf'}")


def create_combined_figure(dwell_times):
    """Create combined histogram and CDF figure."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))

    # Top row: Histograms
    for idx, cluster in enumerate(CLUSTER_ORDER):
        ax = axes[0, idx]
        times = np.array(dwell_times['clusters'].get(cluster, []))

        if len(times) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        times_capped = np.clip(times, 0, 300)
        bins = np.linspace(0, 300, 31)
        ax.hist(times_capped, bins=bins, color=CLUSTER_COLORS[cluster],
                alpha=0.7, edgecolor='black', linewidth=0.5)

        mean_t = np.mean(times)
        median_t = np.median(times)

        ax.axvline(mean_t, color='red', linestyle='--', linewidth=2)
        ax.axvline(median_t, color='blue', linestyle=':', linewidth=2)
        ax.axvline(30, color='green', linestyle='-', alpha=0.7, linewidth=2)

        ax.set_xlabel('Dwell Time (s)')
        ax.set_ylabel('Frequency' if idx == 0 else '')
        ax.set_title(f'{cluster}', fontsize=12, fontweight='bold')

        # Add stats text
        ax.text(0.95, 0.95, f'n={len(times)}\nmean={mean_t:.1f}s\nmedian={median_t:.1f}s',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Bottom row: CDFs
    for idx, cluster in enumerate(CLUSTER_ORDER):
        ax = axes[1, idx]
        times = np.array(dwell_times['clusters'].get(cluster, []))

        if len(times) == 0:
            continue

        sorted_times = np.sort(times)
        cdf = np.arange(1, len(sorted_times) + 1) / len(sorted_times)

        ax.plot(sorted_times, cdf, color=CLUSTER_COLORS[cluster], linewidth=2)
        ax.fill_between(sorted_times, cdf, alpha=0.3, color=CLUSTER_COLORS[cluster])

        # Mark percentiles
        for p, style in [(25, ':'), (50, '--'), (75, ':')]:
            pval = np.percentile(times, p)
            ax.axvline(pval, color='gray', linestyle=style, alpha=0.5)

        ax.axvline(30, color='green', linestyle='-', alpha=0.7, linewidth=2)

        ax.set_xlabel('Dwell Time (s)')
        ax.set_ylabel('CDF' if idx == 0 else '')
        ax.set_xlim(0, 300)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        # Mark fraction above 30s
        frac_above_30 = np.mean(times > 30)
        ax.text(0.95, 0.05, f'{frac_above_30*100:.1f}% > 30s',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Legend
    legend_elements = [
        Line2D([0], [0], color='red', linestyle='--', label='Mean'),
        Line2D([0], [0], color='blue', linestyle=':', label='Median'),
        Line2D([0], [0], color='green', linestyle='-', label='30s Threshold'),
    ]
    axes[0, 2].legend(handles=legend_elements, loc='upper right')

    plt.suptitle('Cluster Dwell Time Analysis', fontsize=14, y=1.01)
    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / 'dwell_time_dist.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'dwell_time_dist.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {OUTPUT_DIR / 'dwell_time_dist.pdf'}")


def compute_dwell_statistics(dwell_times):
    """Compute and print dwell time statistics."""
    stats = {}

    for cluster in CLUSTER_ORDER:
        times = np.array(dwell_times['clusters'].get(cluster, []))

        if len(times) == 0:
            continue

        stats[cluster] = {
            'n_segments': len(times),
            'mean': round(np.mean(times), 2),
            'median': round(np.median(times), 2),
            'std': round(np.std(times), 2),
            'min': round(np.min(times), 2),
            'max': round(np.max(times), 2),
            'p25': round(np.percentile(times, 25), 2),
            'p75': round(np.percentile(times, 75), 2),
            'frac_above_30s': round(np.mean(times > 30), 4),
        }

    return stats


def main():
    print("=" * 70)
    print("Figure G3: Dwell Time Distribution")
    print("=" * 70)

    # Load data
    print("\n[1] Loading dwell times...")
    try:
        dwell_times = load_dwell_times()
        print("  Loaded from pre-computed file")
    except FileNotFoundError:
        print("  Pre-computed file not found. Run comprehensive_statistics.py first.")
        return

    # Compute statistics
    print("\n[2] Computing statistics...")
    stats = compute_dwell_statistics(dwell_times)

    print("\n  Dwell Time Statistics:")
    print(f"  {'Cluster':<10} | {'N':<6} | {'Mean':>8} | {'Median':>8} | {'%>30s':>8}")
    print("  " + "-" * 55)

    for cluster in CLUSTER_ORDER:
        if cluster in stats:
            s = stats[cluster]
            print(f"  {cluster:<10} | {s['n_segments']:<6} | {s['mean']:>7.1f}s | "
                  f"{s['median']:>7.1f}s | {s['frac_above_30s']*100:>7.1f}%")

    # Validate assumption
    print("\n  Validation: Mean dwell time > 30s?")
    for cluster in CLUSTER_ORDER:
        if cluster in stats:
            mean = stats[cluster]['mean']
            valid = "YES" if mean > 30 else "NO"
            print(f"    {cluster}: {mean:.1f}s ({valid})")

    # Create figures
    print("\n[3] Creating figures...")
    create_histogram_figure(dwell_times)
    create_cdf_figure(dwell_times)
    create_combined_figure(dwell_times)

    # Save statistics
    with open(OUTPUT_DIR / 'dwell_time_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved: {OUTPUT_DIR / 'dwell_time_stats.json'}")


if __name__ == "__main__":
    main()
