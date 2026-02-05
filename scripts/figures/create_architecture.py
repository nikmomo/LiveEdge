#!/usr/bin/env python3
"""
Create LiveEdge system architecture diagram.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Output directory
OUTPUT_DIR = "L:/GitHub/LiveEdge/docs/paper/figures"

# Set style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

def create_architecture():
    """Create the LiveEdge system architecture diagram."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Colors
    color_sensor = '#d4e6f1'      # Light blue - IMU
    color_processing = '#fdebd0'  # Light yellow - processing blocks
    color_output = '#d5f5e3'      # Light green - outputs
    color_control = '#fadbd8'     # Light pink - control blocks

    # Box dimensions
    box_width = 1.4
    box_height = 0.8

    # Positions (x, y) for center of each box
    positions = {
        'imu': (1, 4),
        'feature': (3.2, 4),
        'classifier': (5.4, 4),
        'behavior': (7.6, 4),
        'cluster': (5.4, 2.5),
        'fsm': (3.2, 1),
        'odr': (1, 1),
        'cache': (7.6, 1),
    }

    # Box colors
    colors = {
        'imu': color_sensor,
        'feature': color_processing,
        'classifier': color_processing,
        'behavior': color_output,
        'cluster': color_processing,
        'fsm': color_control,
        'odr': color_control,
        'cache': color_output,
    }

    # Box labels
    labels = {
        'imu': 'IMU\nSensor',
        'feature': 'Feature\nExtraction',
        'classifier': 'RF\nClassifier',
        'behavior': 'Behavior\nPrediction',
        'cluster': 'Cluster\nMapping',
        'fsm': 'FSM\nController',
        'odr': 'ODR\nControl',
        'cache': 'Prediction\nCache',
    }

    # Draw boxes
    boxes = {}
    for name, (cx, cy) in positions.items():
        x = cx - box_width / 2
        y = cy - box_height / 2

        # Determine if box should have dashed border (output boxes)
        if name in ['behavior', 'cache']:
            linestyle = '--'
        else:
            linestyle = '-'

        box = FancyBboxPatch(
            (x, y), box_width, box_height,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=colors[name],
            edgecolor='black',
            linewidth=1.5,
            linestyle=linestyle
        )
        ax.add_patch(box)
        boxes[name] = (cx, cy)

        # Add label
        ax.text(cx, cy, labels[name], ha='center', va='center',
                fontsize=11, fontweight='bold')

    # Arrow style
    arrow_style = dict(
        arrowstyle='-|>',
        color='black',
        lw=1.5,
        mutation_scale=15
    )

    dashed_arrow_style = dict(
        arrowstyle='-|>',
        color='black',
        lw=1.5,
        mutation_scale=15,
        linestyle='--'
    )

    # Helper function to draw arrows with labels
    def draw_arrow(start, end, label=None, label_pos='above', style='solid', connectionstyle='arc3,rad=0'):
        if style == 'dashed':
            props = dashed_arrow_style.copy()
        else:
            props = arrow_style.copy()
        props['connectionstyle'] = connectionstyle

        arrow = FancyArrowPatch(start, end, **props)
        ax.add_patch(arrow)

        if label:
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            offset = 0.15 if label_pos == 'above' else -0.15
            # Add white background to prevent arrow crossing through text
            ax.text(mid_x, mid_y + offset, label, ha='center', va='center',
                    fontsize=9, style='italic',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                              edgecolor='none', alpha=0.9))

    # Draw horizontal arrows (top row)
    # IMU → Feature Extraction
    draw_arrow(
        (positions['imu'][0] + box_width/2, positions['imu'][1]),
        (positions['feature'][0] - box_width/2, positions['feature'][1]),
        label='data'
    )

    # Feature Extraction → RF Classifier
    draw_arrow(
        (positions['feature'][0] + box_width/2, positions['feature'][1]),
        (positions['classifier'][0] - box_width/2, positions['classifier'][1]),
        label='features'
    )

    # RF Classifier → Behavior Prediction
    draw_arrow(
        (positions['classifier'][0] + box_width/2, positions['classifier'][1]),
        (positions['behavior'][0] - box_width/2, positions['behavior'][1]),
        label='behavior'
    )

    # RF Classifier → Cluster Mapping (down)
    draw_arrow(
        (positions['classifier'][0], positions['classifier'][1] - box_height/2),
        (positions['cluster'][0], positions['cluster'][1] + box_height/2),
        label='cluster', label_pos='above'
    )

    # Cluster Mapping → FSM Controller (down with right angle)
    # First go down, then left
    cluster_bottom = (positions['cluster'][0], positions['cluster'][1] - box_height/2)
    fsm_top = (positions['fsm'][0], positions['fsm'][1] + box_height/2)

    # Draw as two segments with a corner
    corner_y = (cluster_bottom[1] + fsm_top[1]) / 2
    ax.plot([cluster_bottom[0], cluster_bottom[0]], [cluster_bottom[1], corner_y],
            'k-', lw=1.5)
    ax.plot([cluster_bottom[0], fsm_top[0]], [corner_y, corner_y],
            'k-', lw=1.5)
    draw_arrow(
        (fsm_top[0], corner_y),
        fsm_top
    )

    # FSM Controller → ODR Control (left)
    draw_arrow(
        (positions['fsm'][0] - box_width/2, positions['fsm'][1]),
        (positions['odr'][0] + box_width/2, positions['odr'][1])
    )

    # ODR Control → IMU Sensor (up)
    draw_arrow(
        (positions['odr'][0], positions['odr'][1] + box_height/2),
        (positions['imu'][0], positions['imu'][1] - box_height/2),
        label='ODR', label_pos='above'
    )

    # FSM Controller → Prediction Cache (right) - straight line with label
    draw_arrow(
        (positions['fsm'][0] + box_width/2, positions['fsm'][1]),
        (positions['cache'][0] - box_width/2, positions['cache'][1]),
        label='skip/run'
    )

    # Prediction Cache → Behavior Prediction (up) - dashed line
    # Use straight vertical line
    cache_top = (positions['cache'][0], positions['cache'][1] + box_height/2)
    behavior_bottom = (positions['behavior'][0], positions['behavior'][1] - box_height/2)

    # Draw dashed arrow going up
    ax.annotate('',
                xy=behavior_bottom,
                xytext=cache_top,
                arrowprops=dict(arrowstyle='-|>', color='black', lw=1.5,
                               linestyle='--', mutation_scale=15))

    # Add "cached" label to the right of the dashed line
    ax.text(positions['cache'][0] + 0.3, (cache_top[1] + behavior_bottom[1]) / 2,
            'cached', ha='left', va='center', fontsize=9, style='italic', rotation=90)

    # Set axis properties
    ax.set_xlim(-0.5, 9)
    ax.set_ylim(-0.3, 5.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Add color legend
    legend_elements = [
        mpatches.Patch(facecolor=color_sensor, edgecolor='black', label='Sensor'),
        mpatches.Patch(facecolor=color_processing, edgecolor='black', label='Processing'),
        mpatches.Patch(facecolor=color_control, edgecolor='black', label='Control'),
        mpatches.Patch(facecolor=color_output, edgecolor='black', linestyle='--', label='Output'),
    ]
    ax.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=9,
              frameon=True, fancybox=True, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/architecture.pdf')
    plt.savefig(f'{OUTPUT_DIR}/architecture.png')
    plt.close()
    print("Created: architecture.png")


if __name__ == '__main__':
    create_architecture()
