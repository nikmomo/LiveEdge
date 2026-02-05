"""Visualization utilities for plotting results.

This module provides functions for creating publication-quality figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from numpy.typing import NDArray


def setup_plotting_style(style: str = "seaborn-v0_8-whitegrid") -> None:
    """Set up matplotlib style for publication-quality figures.

    Args:
        style: Matplotlib style to use.
    """
    try:
        plt.style.use(style)
    except OSError:
        plt.style.use("seaborn-whitegrid")

    plt.rcParams.update(
        {
            "figure.figsize": (8, 6),
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )


def plot_confusion_matrix(
    cm: NDArray[np.int64],
    class_names: list[str],
    normalize: bool = True,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    figsize: tuple[float, float] = (8, 6),
) -> Figure:
    """Plot a confusion matrix.

    Args:
        cm: Confusion matrix array.
        class_names: List of class names.
        normalize: Whether to normalize the matrix.
        title: Plot title.
        cmap: Colormap name.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if normalize:
        cm_display = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar=True,
        square=True,
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)

    plt.tight_layout()
    return fig


def plot_sampling_timeline(
    timestamps: NDArray[np.float64],
    sampling_rates: NDArray[np.float64],
    states: list[str] | None = None,
    true_labels: list[str] | None = None,
    title: str = "Adaptive Sampling Timeline",
    figsize: tuple[float, float] = (12, 6),
) -> Figure:
    """Plot sampling rate changes over time.

    Args:
        timestamps: Array of timestamps.
        sampling_rates: Array of sampling rates.
        states: Optional list of behavior states.
        true_labels: Optional list of true behavior labels.
        title: Plot title.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    n_plots = 1 + (states is not None) + (true_labels is not None)
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)

    if n_plots == 1:
        axes = [axes]

    # Sampling rate plot
    ax = axes[0]
    ax.step(timestamps, sampling_rates, where="post", linewidth=1.5)
    ax.fill_between(timestamps, sampling_rates, step="post", alpha=0.3)
    ax.set_ylabel("Sampling Rate (Hz)")
    ax.set_title(title)

    plot_idx = 1

    # State plot
    if states is not None:
        ax = axes[plot_idx]
        unique_states = sorted(set(states))
        state_to_idx = {s: i for i, s in enumerate(unique_states)}
        state_indices = [state_to_idx[s] for s in states]
        ax.step(timestamps, state_indices, where="post", linewidth=1.5, color="green")
        ax.set_ylabel("Predicted State")
        ax.set_yticks(range(len(unique_states)))
        ax.set_yticklabels(unique_states)
        plot_idx += 1

    # True labels plot
    if true_labels is not None:
        ax = axes[plot_idx]
        unique_labels = sorted(set(true_labels))
        label_to_idx = {s: i for i, s in enumerate(unique_labels)}
        label_indices = [label_to_idx[s] for s in true_labels]
        ax.step(timestamps, label_indices, where="post", linewidth=1.5, color="orange")
        ax.set_ylabel("True Label")
        ax.set_yticks(range(len(unique_labels)))
        ax.set_yticklabels(unique_labels)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    return fig


def plot_energy_breakdown(
    energy_dict: dict[str, float],
    title: str = "Energy Consumption Breakdown",
    colors: list[str] | None = None,
    figsize: tuple[float, float] = (8, 8),
) -> Figure:
    """Plot energy consumption breakdown as a pie chart.

    Args:
        energy_dict: Dictionary of component name to energy value.
        title: Plot title.
        colors: Optional list of colors.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    labels = list(energy_dict.keys())
    values = list(energy_dict.values())

    if colors is None:
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        pctdistance=0.75,
    )

    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_frequency_accuracy_curve(
    frequencies: list[float],
    accuracies: dict[str, list[float]],
    title: str = "Frequency-Accuracy Trade-off",
    figsize: tuple[float, float] = (10, 6),
) -> Figure:
    """Plot accuracy vs sampling frequency curves.

    Args:
        frequencies: List of sampling frequencies.
        accuracies: Dictionary mapping behavior/method name to accuracy values.
        title: Plot title.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for name, acc in accuracies.items():
        ax.plot(frequencies, acc, marker="o", linewidth=2, label=name)

    ax.set_xlabel("Sampling Frequency (Hz)")
    ax.set_ylabel("Classification Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_feature_importance(
    feature_names: list[str],
    importances: NDArray[np.float64],
    top_k: int = 20,
    title: str = "Feature Importance",
    figsize: tuple[float, float] = (10, 8),
) -> Figure:
    """Plot feature importance as a horizontal bar chart.

    Args:
        feature_names: List of feature names.
        importances: Array of importance values.
        top_k: Number of top features to show.
        title: Plot title.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_k]
    top_names = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    # Plot horizontal bars
    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_importances, align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(title)

    plt.tight_layout()
    return fig


def save_figure(
    fig: Figure,
    path: str | Path,
    formats: list[str] | None = None,
    dpi: int = 300,
) -> None:
    """Save figure to file(s).

    Args:
        fig: Matplotlib figure.
        path: Output path (without extension).
        formats: List of formats to save (default: ["pdf", "png"]).
        dpi: Resolution for raster formats.
    """
    if formats is None:
        formats = ["pdf", "png"]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        fig.savefig(path.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")


def plot_learning_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_metrics: dict[str, list[float]] | None = None,
    val_metrics: dict[str, list[float]] | None = None,
    title: str = "Learning Curves",
    figsize: tuple[float, float] = (12, 5),
) -> Figure:
    """Plot training and validation learning curves.

    Args:
        train_losses: List of training losses per epoch.
        val_losses: List of validation losses per epoch.
        train_metrics: Optional dict of training metrics per epoch.
        val_metrics: Optional dict of validation metrics per epoch.
        title: Plot title.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    n_plots = 1 + (train_metrics is not None or val_metrics is not None)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    if n_plots == 1:
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)

    # Loss plot
    ax = axes[0]
    ax.plot(epochs, train_losses, label="Train", linewidth=2)
    ax.plot(epochs, val_losses, label="Validation", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()

    # Metrics plot
    if n_plots > 1:
        ax = axes[1]
        if train_metrics:
            for name, values in train_metrics.items():
                ax.plot(epochs, values, label=f"Train {name}", linewidth=2)
        if val_metrics:
            for name, values in val_metrics.items():
                ax.plot(epochs, values, label=f"Val {name}", linewidth=2, linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Metric Value")
        ax.set_title("Metrics")
        ax.legend()

    fig.suptitle(title)
    plt.tight_layout()
    return fig
