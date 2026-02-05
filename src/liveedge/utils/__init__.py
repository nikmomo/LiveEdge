"""Utility functions for LiveEdge.

This module provides configuration, logging, reproducibility, and visualization utilities.
"""

from liveedge.utils.config import (
    get_nested,
    load_config,
    merge_configs,
    print_config,
    save_config,
    to_dict,
    validate_config,
)
from liveedge.utils.logging import (
    LoggerAdapter,
    get_logger,
    log_dict,
    log_metrics,
    setup_logging,
)
from liveedge.utils.seed import (
    SeedManager,
    get_rng,
    set_seed,
    worker_init_fn,
)
from liveedge.utils.visualization import (
    plot_confusion_matrix,
    plot_energy_breakdown,
    plot_feature_importance,
    plot_frequency_accuracy_curve,
    plot_learning_curves,
    plot_sampling_timeline,
    save_figure,
    setup_plotting_style,
)

__all__ = [
    # config
    "load_config",
    "merge_configs",
    "to_dict",
    "validate_config",
    "get_nested",
    "save_config",
    "print_config",
    # logging
    "setup_logging",
    "get_logger",
    "LoggerAdapter",
    "log_dict",
    "log_metrics",
    # seed
    "set_seed",
    "get_rng",
    "SeedManager",
    "worker_init_fn",
    # visualization
    "setup_plotting_style",
    "plot_confusion_matrix",
    "plot_sampling_timeline",
    "plot_energy_breakdown",
    "plot_frequency_accuracy_curve",
    "plot_feature_importance",
    "save_figure",
    "plot_learning_curves",
]
