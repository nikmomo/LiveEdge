"""PyTorch Dataset classes for accelerometer data.

This module provides Dataset classes for training behavior classification models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset


class AccelerometerDataset(Dataset):
    """PyTorch Dataset for accelerometer windows.

    Supports both raw windows (for deep learning) and feature vectors (for traditional ML).

    Attributes:
        data: Data array (windows or features).
        labels: Label array.
        transform: Optional transform to apply to data.
    """

    def __init__(
        self,
        data: NDArray[np.float32],
        labels: NDArray[np.int64],
        transform: Callable[[NDArray[np.float32]], NDArray[np.float32]] | None = None,
    ):
        """Initialize the dataset.

        Args:
            data: Data array of shape (n_samples, ...).
            labels: Labels of shape (n_samples,).
            transform: Optional transform to apply.
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a sample by index.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (data_tensor, label_tensor).
        """
        x = self.data[idx]
        y = self.labels[idx]

        if self.transform is not None:
            x = self.transform(x)

        return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.long)


class StreamingDataset(Dataset):
    """Dataset for streaming simulation.

    Provides data in sequential order for simulating real-time inference.

    Attributes:
        data: Raw data array of shape (n_samples, n_channels).
        labels: Labels of shape (n_samples,).
        timestamps: Timestamps of shape (n_samples,).
        window_size: Window size in samples.
        stride: Stride between consecutive windows.
    """

    def __init__(
        self,
        data: NDArray[np.float32],
        labels: NDArray[np.int64],
        timestamps: NDArray[np.float64],
        window_size: int,
        stride: int = 1,
    ):
        """Initialize the dataset.

        Args:
            data: Raw data of shape (n_samples, n_channels).
            labels: Labels of shape (n_samples,).
            timestamps: Timestamps of shape (n_samples,).
            window_size: Window size in samples.
            stride: Stride between windows.
        """
        self.data = data
        self.labels = labels
        self.timestamps = timestamps
        self.window_size = window_size
        self.stride = stride

        # Calculate number of windows
        self.n_windows = max(0, (len(data) - window_size) // stride + 1)

    def __len__(self) -> int:
        """Return the number of windows."""
        return self.n_windows

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a window by index.

        Args:
            idx: Window index.

        Returns:
            Dictionary with keys: window, label, timestamp, index.
        """
        start = idx * self.stride
        end = start + self.window_size

        window = self.data[start:end]
        # Majority label for the window
        window_labels = self.labels[start:end]
        label = np.bincount(window_labels).argmax()
        timestamp = self.timestamps[start]

        return {
            "window": torch.from_numpy(window).float(),
            "label": torch.tensor(label, dtype=torch.long),
            "timestamp": timestamp,
            "index": idx,
        }


class MultiSubjectDataset(Dataset):
    """Dataset that maintains subject information.

    Useful for Leave-One-Subject-Out cross-validation.

    Attributes:
        data: Data array.
        labels: Label array.
        subject_ids: Subject ID for each sample.
        transform: Optional transform.
    """

    def __init__(
        self,
        data: NDArray[np.float32],
        labels: NDArray[np.int64],
        subject_ids: NDArray[np.object_],
        transform: Callable[[NDArray[np.float32]], NDArray[np.float32]] | None = None,
    ):
        """Initialize the dataset.

        Args:
            data: Data array of shape (n_samples, ...).
            labels: Labels of shape (n_samples,).
            subject_ids: Subject IDs of shape (n_samples,).
            transform: Optional transform.
        """
        self.data = data
        self.labels = labels
        self.subject_ids = subject_ids
        self.transform = transform

        # Index samples by subject
        self.subjects = np.unique(subject_ids)
        self.subject_indices = {
            subj: np.where(subject_ids == subj)[0] for subj in self.subjects
        }

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Get a sample by index.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (data_tensor, label_tensor, subject_id).
        """
        x = self.data[idx]
        y = self.labels[idx]
        subject = self.subject_ids[idx]

        if self.transform is not None:
            x = self.transform(x)

        return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.long), subject

    def get_subject_split(
        self,
        test_subjects: list[str],
        val_subjects: list[str] | None = None,
    ) -> dict[str, "MultiSubjectDataset"]:
        """Split dataset by subject.

        Args:
            test_subjects: Subjects for test set.
            val_subjects: Optional subjects for validation set.

        Returns:
            Dictionary with "train", "val" (if provided), and "test" datasets.
        """
        test_mask = np.isin(self.subject_ids, test_subjects)
        val_mask = (
            np.isin(self.subject_ids, val_subjects) if val_subjects else np.zeros_like(test_mask)
        )
        train_mask = ~(test_mask | val_mask)

        result = {
            "train": MultiSubjectDataset(
                self.data[train_mask],
                self.labels[train_mask],
                self.subject_ids[train_mask],
                self.transform,
            ),
            "test": MultiSubjectDataset(
                self.data[test_mask],
                self.labels[test_mask],
                self.subject_ids[test_mask],
                self.transform,
            ),
        }

        if val_subjects:
            result["val"] = MultiSubjectDataset(
                self.data[val_mask],
                self.labels[val_mask],
                self.subject_ids[val_mask],
                self.transform,
            )

        return result


def load_processed_data(
    data_path: str | Path,
) -> dict[str, NDArray[np.float32] | NDArray[np.int64] | list[str]]:
    """Load processed data from NPZ file.

    Args:
        data_path: Path to the processed data file.

    Returns:
        Dictionary with keys:
        - X: Data array
        - y: Labels
        - class_names: List of class names
        - subject_ids: Optional subject IDs
    """
    data_path = Path(data_path)

    if data_path.suffix == ".npz":
        loaded = np.load(data_path, allow_pickle=True)
        result = {
            "X": loaded["X"],
            "y": loaded["y"],
            "class_names": loaded["class_names"].tolist(),
        }
        if "subject_ids" in loaded:
            result["subject_ids"] = loaded["subject_ids"]
        return result
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")


def save_processed_data(
    X: NDArray[np.float32],
    y: NDArray[np.int64],
    class_names: list[str],
    output_path: str | Path,
    subject_ids: NDArray[np.object_] | None = None,
) -> None:
    """Save processed data to NPZ file.

    Args:
        X: Data array.
        y: Labels.
        class_names: List of class names.
        output_path: Output file path.
        subject_ids: Optional subject IDs.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "X": X,
        "y": y,
        "class_names": np.array(class_names),
    }
    if subject_ids is not None:
        save_dict["subject_ids"] = subject_ids

    np.savez_compressed(output_path, **save_dict)


def create_data_loaders(
    train_data: tuple[NDArray[np.float32], NDArray[np.int64]],
    val_data: tuple[NDArray[np.float32], NDArray[np.int64]] | None,
    test_data: tuple[NDArray[np.float32], NDArray[np.int64]],
    batch_size: int = 64,
    num_workers: int = 0,
    train_transform: Callable | None = None,
) -> dict[str, torch.utils.data.DataLoader]:
    """Create PyTorch DataLoaders.

    Args:
        train_data: Tuple of (X_train, y_train).
        val_data: Optional tuple of (X_val, y_val).
        test_data: Tuple of (X_test, y_test).
        batch_size: Batch size.
        num_workers: Number of worker processes.
        train_transform: Optional transform for training data.

    Returns:
        Dictionary with "train", "val" (if provided), and "test" DataLoaders.
    """
    from liveedge.utils.seed import worker_init_fn

    train_dataset = AccelerometerDataset(train_data[0], train_data[1], train_transform)
    test_dataset = AccelerometerDataset(test_data[0], test_data[1])

    loaders = {
        "train": torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            pin_memory=True,
        ),
        "test": torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    if val_data is not None:
        val_dataset = AccelerometerDataset(val_data[0], val_data[1])
        loaders["val"] = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return loaders
