"""Reproducibility utilities for setting random seeds.

This module provides utilities for ensuring reproducible experiments.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seeds for reproducibility.

    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (if available)
    - TensorFlow (if available)

    Args:
        seed: Random seed value.
        deterministic: If True, enable deterministic mode in PyTorch (may reduce performance).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True)
    except ImportError:
        pass

    # TensorFlow
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def get_rng(seed: int | None = None) -> np.random.Generator:
    """Get a NumPy random number generator.

    Args:
        seed: Optional seed for the generator.

    Returns:
        NumPy random number generator.
    """
    return np.random.default_rng(seed)


class SeedManager:
    """Context manager for temporary seed changes.

    Example:
        >>> with SeedManager(42):
        ...     # Code with seed 42
        ...     result = np.random.rand()
        >>> # Original random state restored
    """

    def __init__(self, seed: int):
        """Initialize the seed manager.

        Args:
            seed: Seed to use within the context.
        """
        self.seed = seed
        self._random_state: Any = None
        self._numpy_state: Any = None
        self._torch_state: Any = None

    def __enter__(self) -> "SeedManager":
        """Enter the context and save current random states."""
        # Save Python random state
        self._random_state = random.getstate()

        # Save NumPy state
        self._numpy_state = np.random.get_state()

        # Save PyTorch state if available
        try:
            import torch

            self._torch_state = torch.get_rng_state()
        except ImportError:
            pass

        # Set new seed
        set_seed(self.seed)
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit the context and restore random states."""
        # Restore Python random state
        random.setstate(self._random_state)

        # Restore NumPy state
        np.random.set_state(self._numpy_state)

        # Restore PyTorch state if available
        try:
            import torch

            if self._torch_state is not None:
                torch.set_rng_state(self._torch_state)
        except ImportError:
            pass


def worker_init_fn(worker_id: int) -> None:
    """Worker initialization function for PyTorch DataLoader.

    Ensures each worker has a different but reproducible seed.

    Args:
        worker_id: ID of the worker process.

    Example:
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(dataset, num_workers=4, worker_init_fn=worker_init_fn)
    """
    try:
        import torch

        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    except ImportError:
        pass
