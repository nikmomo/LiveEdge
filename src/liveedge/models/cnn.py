"""1D Convolutional Neural Network for sequence classification.

This module provides a 1D-CNN architecture for accelerometer-based
behavior classification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray

from liveedge.models.base import DeepLearningClassifier


class Conv1DBlock(nn.Module):
    """1D Convolutional block with batch normalization and dropout.

    Architecture: Conv1D -> BatchNorm -> ReLU -> MaxPool -> Dropout
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        pool_size: int = 2,
        dropout: float = 0.3,
        batch_norm: bool = True,
    ):
        """Initialize the convolutional block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size for convolution.
            pool_size: Pool size for max pooling.
            dropout: Dropout rate.
            batch_norm: Whether to use batch normalization.
        """
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
        )
        self.batch_norm = nn.BatchNorm1d(out_channels) if batch_norm else nn.Identity()
        self.pool = nn.MaxPool1d(pool_size) if pool_size > 1 else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, seq_len).

        Returns:
            Output tensor.
        """
        x = self.conv(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class CNN1D(nn.Module):
    """1D Convolutional Neural Network for sequence classification.

    Architecture:
        Input (batch, seq_len, channels) -> Transpose
        -> Conv1DBlock x N -> GlobalAvgPool -> FC layers -> Output
    """

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 5,
        conv_channels: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        pool_sizes: list[int] | None = None,
        fc_dims: list[int] | None = None,
        dropout: float = 0.3,
        batch_norm: bool = True,
    ):
        """Initialize the CNN1D model.

        Args:
            input_channels: Number of input channels (accelerometer axes).
            num_classes: Number of output classes.
            conv_channels: List of channel sizes for conv layers.
            kernel_sizes: List of kernel sizes for conv layers.
            pool_sizes: List of pool sizes for conv layers.
            fc_dims: List of hidden dimensions for FC layers.
            dropout: Dropout rate.
            batch_norm: Whether to use batch normalization.
        """
        super().__init__()

        if conv_channels is None:
            conv_channels = [32, 64, 128]
        if kernel_sizes is None:
            kernel_sizes = [5, 5, 3]
        if pool_sizes is None:
            pool_sizes = [2, 2, 2]
        if fc_dims is None:
            fc_dims = [64]

        self.input_channels = input_channels
        self.num_classes = num_classes

        # Build convolutional layers
        conv_layers = []
        in_ch = input_channels
        for out_ch, ks, ps in zip(conv_channels, kernel_sizes, pool_sizes):
            conv_layers.append(
                Conv1DBlock(in_ch, out_ch, ks, ps, dropout, batch_norm)
            )
            in_ch = out_ch

        self.conv_layers = nn.Sequential(*conv_layers)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Build FC layers
        fc_layers = []
        fc_in = conv_channels[-1]
        for fc_dim in fc_dims:
            fc_layers.extend(
                [
                    nn.Linear(fc_in, fc_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            fc_in = fc_dim

        # Output layer
        fc_layers.append(nn.Linear(fc_in, num_classes))

        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_channels).

        Returns:
            Logits of shape (batch, num_classes).
        """
        # Transpose to (batch, channels, seq_len) for Conv1d
        x = x.transpose(1, 2)

        # Convolutional layers
        x = self.conv_layers(x)

        # Global average pooling
        x = self.global_pool(x)
        x = x.squeeze(-1)

        # FC layers
        x = self.fc_layers(x)

        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get features before classification head.

        Args:
            x: Input tensor of shape (batch, seq_len, input_channels).

        Returns:
            Feature tensor of shape (batch, feature_dim).
        """
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        return x


class CNN1DClassifier(DeepLearningClassifier):
    """Classifier wrapper for CNN1D model.

    Provides training, inference, and persistence functionality.
    """

    def __init__(
        self,
        num_classes: int = 5,
        input_channels: int = 3,
        conv_channels: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        pool_sizes: list[int] | None = None,
        fc_dims: list[int] | None = None,
        dropout: float = 0.3,
        batch_norm: bool = True,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        device: str | None = None,
        **kwargs: Any,
    ):
        """Initialize the CNN1D classifier.

        Args:
            num_classes: Number of output classes.
            input_channels: Number of input channels.
            conv_channels: Channel sizes for conv layers.
            kernel_sizes: Kernel sizes for conv layers.
            pool_sizes: Pool sizes for conv layers.
            fc_dims: Hidden dimensions for FC layers.
            dropout: Dropout rate.
            batch_norm: Whether to use batch normalization.
            learning_rate: Learning rate for optimizer.
            weight_decay: Weight decay for regularization.
            device: Device to use ("cuda", "cpu", or None for auto).
            **kwargs: Additional parameters.
        """
        super().__init__(num_classes, **kwargs)

        self.input_channels = input_channels
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Create model
        self.model = CNN1D(
            input_channels=input_channels,
            num_classes=num_classes,
            conv_channels=conv_channels,
            kernel_sizes=kernel_sizes,
            pool_sizes=pool_sizes,
            fc_dims=fc_dims,
            dropout=dropout,
            batch_norm=batch_norm,
        ).to(self.device)

        self._config.update(
            {
                "input_channels": input_channels,
                "conv_channels": conv_channels or [32, 64, 128],
                "kernel_sizes": kernel_sizes or [5, 5, 3],
                "pool_sizes": pool_sizes or [2, 2, 2],
                "fc_dims": fc_dims or [64],
                "dropout": dropout,
                "batch_norm": batch_norm,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
            }
        )

    def fit(
        self,
        X: NDArray[np.float32],
        y: NDArray[np.int64],
        X_val: NDArray[np.float32] | None = None,
        y_val: NDArray[np.int64] | None = None,
        epochs: int = 100,
        batch_size: int = 64,
        early_stopping_patience: int = 10,
        verbose: bool = True,
    ) -> "CNN1DClassifier":
        """Train the CNN1D classifier.

        Args:
            X: Training data of shape (n_samples, seq_len, input_channels).
            y: Training labels.
            X_val: Optional validation data.
            y_val: Optional validation labels.
            epochs: Maximum number of training epochs.
            batch_size: Training batch size.
            early_stopping_patience: Epochs to wait before early stopping.
            verbose: Whether to print training progress.

        Returns:
            Self for method chaining.
        """
        from torch.utils.data import DataLoader, TensorDataset

        # Create datasets
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).long()
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.from_numpy(X_val).float()
            y_val_tensor = torch.from_numpy(y_val).long()
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Optimizer and loss
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            val_loss = 0.0
            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                val_loss /= len(val_loader)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        break

            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}"
                if val_loader is not None:
                    msg += f" - Val Loss: {val_loss:.4f}"
                print(msg)

        self.is_fitted = True
        return self

    def predict(self, X: NDArray[np.float32]) -> NDArray[np.int64]:
        """Predict class labels."""
        self.model.eval()
        X_tensor = torch.from_numpy(X).float().to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)

        return predictions.cpu().numpy().astype(np.int64)

    def predict_proba(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Predict class probabilities."""
        self.model.eval()
        X_tensor = torch.from_numpy(X).float().to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = F.softmax(outputs, dim=1)

        return probabilities.cpu().numpy().astype(np.float32)

    def get_model(self) -> CNN1D:
        """Get the underlying CNN1D model."""
        return self.model

    def get_feature_extractor(self) -> nn.Sequential:
        """Get feature extraction layers."""
        return self.model.conv_layers

    def save(self, path: str | Path) -> None:
        """Save the model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "num_classes": self.num_classes,
                "config": self._config,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "CNN1DClassifier":
        """Load a model from disk."""
        data = torch.load(path, map_location="cpu")

        instance = cls(num_classes=data["num_classes"], **data["config"])
        instance.model.load_state_dict(data["model_state_dict"])
        instance.is_fitted = True
        return instance
