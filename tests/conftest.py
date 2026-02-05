"""Pytest fixtures for LiveEdge tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_window():
    """Generate synthetic accelerometer window (1.5s @ 50Hz)."""
    np.random.seed(42)
    return np.random.randn(75, 3).astype(np.float32)


@pytest.fixture
def periodic_window():
    """Generate periodic signal for testing (2 Hz sine wave)."""
    t = np.linspace(0, 1.5, 75)
    x = np.sin(2 * np.pi * 2 * t)  # 2 Hz sine wave
    y = np.sin(2 * np.pi * 2 * t + np.pi / 4)
    z = np.cos(2 * np.pi * 2 * t)
    return np.stack([x, y, z], axis=1).astype(np.float32)


@pytest.fixture
def constant_window():
    """Generate constant signal for testing."""
    return np.ones((75, 3), dtype=np.float32)


@pytest.fixture
def high_activity_window():
    """Generate high-activity signal with rapid changes."""
    np.random.seed(123)
    t = np.linspace(0, 1.5, 75)
    # High frequency components
    x = np.sin(2 * np.pi * 10 * t) + 0.5 * np.random.randn(75)
    y = np.sin(2 * np.pi * 10 * t + np.pi / 3) + 0.5 * np.random.randn(75)
    z = np.sin(2 * np.pi * 10 * t + 2 * np.pi / 3) + 0.5 * np.random.randn(75)
    return np.stack([x, y, z], axis=1).astype(np.float32)


@pytest.fixture
def sample_raw_data():
    """Generate sample raw data DataFrame."""
    np.random.seed(42)
    n_samples = 1000

    data = pd.DataFrame(
        {
            "timestamp": np.linspace(0, 20, n_samples),
            "acc_x": np.random.randn(n_samples),
            "acc_y": np.random.randn(n_samples),
            "acc_z": np.random.randn(n_samples) + 9.8,
            "behavior": np.random.choice(
                ["walking", "standing", "lying"], n_samples
            ),
            "subject_id": np.random.choice(["cow_1", "cow_2", "cow_3"], n_samples),
        }
    )
    return data


@pytest.fixture
def sample_windows():
    """Generate batch of sample windows."""
    np.random.seed(42)
    return np.random.randn(100, 75, 3).astype(np.float32)


@pytest.fixture
def sample_labels():
    """Generate sample labels."""
    np.random.seed(42)
    return np.random.randint(0, 5, 100).astype(np.int64)


@pytest.fixture
def continuous_data():
    """Generate continuous accelerometer data with labels."""
    np.random.seed(42)
    n_samples = 500

    data = np.random.randn(n_samples, 3).astype(np.float32)
    # Create labels that change over time
    labels = np.zeros(n_samples, dtype=np.int64)
    labels[100:200] = 1
    labels[200:350] = 2
    labels[350:450] = 1
    labels[450:] = 3

    timestamps = np.linspace(0, 10, n_samples).astype(np.float64)

    return data, labels, timestamps
