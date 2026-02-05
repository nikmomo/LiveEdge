"""Abstract base classes for behavior classifiers.

This module defines the interface that all classifiers must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


class BaseClassifier(ABC):
    """Abstract base class for behavior classifiers.

    All classifiers must implement fit, predict, and predict_proba methods
    to ensure consistent interface across traditional ML and deep learning models.

    Attributes:
        num_classes: Number of output classes.
        is_fitted: Whether the model has been trained.
    """

    def __init__(self, num_classes: int, **kwargs: Any):
        """Initialize the classifier.

        Args:
            num_classes: Number of output classes.
            **kwargs: Additional model-specific parameters.
        """
        self.num_classes = num_classes
        self.is_fitted = False
        self._config = kwargs

    @abstractmethod
    def fit(
        self,
        X: NDArray[np.float32],
        y: NDArray[np.int64],
        X_val: NDArray[np.float32] | None = None,
        y_val: NDArray[np.int64] | None = None,
    ) -> "BaseClassifier":
        """Train the classifier.

        Args:
            X: Training features of shape (n_samples, n_features) or
               (n_samples, seq_len, n_channels) for sequential models.
            y: Training labels of shape (n_samples,).
            X_val: Optional validation features.
            y_val: Optional validation labels.

        Returns:
            Self for method chaining.
        """
        pass

    @abstractmethod
    def predict(self, X: NDArray[np.float32]) -> NDArray[np.int64]:
        """Predict class labels.

        Args:
            X: Input features.

        Returns:
            Predicted class labels of shape (n_samples,).
        """
        pass

    @abstractmethod
    def predict_proba(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Predict class probabilities.

        Args:
            X: Input features.

        Returns:
            Class probabilities of shape (n_samples, n_classes).
        """
        pass

    def get_confidence(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Get prediction confidence (max probability).

        Args:
            X: Input features.

        Returns:
            Confidence scores of shape (n_samples,).
        """
        proba = self.predict_proba(X)
        return np.max(proba, axis=1).astype(np.float32)

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save the model to disk.

        Args:
            path: Path to save the model.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "BaseClassifier":
        """Load a model from disk.

        Args:
            path: Path to the saved model.

        Returns:
            Loaded classifier instance.
        """
        pass

    def get_config(self) -> dict[str, Any]:
        """Get model configuration.

        Returns:
            Dictionary of model configuration.
        """
        return {
            "num_classes": self.num_classes,
            **self._config,
        }


class DeepLearningClassifier(BaseClassifier):
    """Base class for deep learning classifiers.

    Provides additional methods specific to neural network models.
    """

    @abstractmethod
    def get_model(self) -> Any:
        """Get the underlying neural network model.

        Returns:
            The neural network model (e.g., PyTorch Module).
        """
        pass

    @abstractmethod
    def get_feature_extractor(self) -> Any:
        """Get feature extraction layers (excluding classification head).

        Returns:
            Feature extraction portion of the model.
        """
        pass

    def count_parameters(self) -> int:
        """Count trainable parameters.

        Returns:
            Number of trainable parameters.
        """
        import torch

        model = self.get_model()
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def to_device(self, device: str) -> "DeepLearningClassifier":
        """Move model to specified device.

        Args:
            device: Device string (e.g., "cuda", "cpu").

        Returns:
            Self for method chaining.
        """
        import torch

        model = self.get_model()
        model.to(torch.device(device))
        return self
