"""Temporal Convolutional Network for sequence classification.

This module provides a TCN architecture optimized for edge deployment,
featuring dilated causal convolutions for capturing temporal dependencies.
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


class CausalConv1d(nn.Module):
    """Causal 1D convolution with optional dilation.

    Ensures output at time t only depends on inputs at times <= t.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ):
        """Initialize causal convolution.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size.
            dilation: Dilation factor.
        """
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=self.padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with causal padding removal.

        Args:
            x: Input tensor of shape (batch, channels, seq_len).

        Returns:
            Output tensor with same sequence length.
        """
        x = self.conv(x)
        # Remove future padding to maintain causality
        if self.padding > 0:
            x = x[:, :, : -self.padding]
        return x


class TemporalBlock(nn.Module):
    """Single temporal block with dilated causal convolution.

    Architecture:
        CausalConv -> WeightNorm -> ReLU -> Dropout
        -> CausalConv -> WeightNorm -> ReLU -> Dropout
        + Residual Connection
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        """Initialize temporal block.

        Args:
            n_inputs: Number of input channels.
            n_outputs: Number of output channels.
            kernel_size: Kernel size for convolutions.
            dilation: Dilation factor.
            dropout: Dropout rate.
        """
        super().__init__()

        # First causal conv
        self.conv1 = nn.utils.parametrizations.weight_norm(
            CausalConv1d(n_inputs, n_outputs, kernel_size, dilation)
        )
        self.dropout1 = nn.Dropout(dropout)

        # Second causal conv
        self.conv2 = nn.utils.parametrizations.weight_norm(
            CausalConv1d(n_outputs, n_outputs, kernel_size, dilation)
        )
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection (1x1 conv if channels differ)
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, seq_len).

        Returns:
            Output tensor.
        """
        out = self.conv1.conv(x)
        if self.conv1.conv.padding > 0:
            out = out[:, :, : -self.conv1.conv.padding]
        out = self.relu(out)
        out = self.dropout1(out)

        out = self.conv2.conv(out)
        if self.conv2.conv.padding > 0:
            out = out[:, :, : -self.conv2.conv.padding]
        out = self.relu(out)
        out = self.dropout2(out)

        # Residual
        res = x if self.downsample is None else self.downsample(x)

        return self.relu(out + res)


class TCN(nn.Module):
    """Temporal Convolutional Network for sequence classification.

    Architecture optimized for edge deployment:
    - Limited depth (3-4 layers)
    - Small channel sizes (8-32)
    - Total parameters < 50K for quantization
    """

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 5,
        num_channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        use_skip_connections: bool = True,
    ):
        """Initialize TCN model.

        Args:
            input_channels: Number of input channels (accelerometer axes).
            num_classes: Number of output classes.
            num_channels: List of channel sizes for each temporal layer.
            kernel_size: Kernel size for temporal convolutions.
            dropout: Dropout rate.
            use_skip_connections: Whether to use skip connections to output.
        """
        super().__init__()

        if num_channels is None:
            num_channels = [16, 32, 32]

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.use_skip_connections = use_skip_connections

        # Build temporal layers
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation = 2**i
            in_ch = input_channels if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]

            layers.append(
                TemporalBlock(
                    in_ch, out_ch, kernel_size, dilation, dropout
                )
            )

        self.network = nn.ModuleList(layers)

        # Output layers
        if use_skip_connections:
            # Sum of all channel outputs
            total_channels = sum(num_channels)
            self.fc = nn.Linear(total_channels, num_classes)
        else:
            self.fc = nn.Linear(num_channels[-1], num_classes)

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_channels).

        Returns:
            Logits of shape (batch, num_classes).
        """
        # Transpose to (batch, channels, seq_len)
        x = x.transpose(1, 2)

        if self.use_skip_connections:
            skip_outputs = []
            for layer in self.network:
                x = layer(x)
                skip_outputs.append(self.global_pool(x).squeeze(-1))
            x = torch.cat(skip_outputs, dim=1)
        else:
            for layer in self.network:
                x = layer(x)
            x = self.global_pool(x).squeeze(-1)

        return self.fc(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get features before classification head.

        Args:
            x: Input tensor of shape (batch, seq_len, input_channels).

        Returns:
            Feature tensor.
        """
        x = x.transpose(1, 2)

        if self.use_skip_connections:
            skip_outputs = []
            for layer in self.network:
                x = layer(x)
                skip_outputs.append(self.global_pool(x).squeeze(-1))
            return torch.cat(skip_outputs, dim=1)
        else:
            for layer in self.network:
                x = layer(x)
            return self.global_pool(x).squeeze(-1)


class TCNClassifier(DeepLearningClassifier):
    """Classifier wrapper for TCN model.

    Provides training, inference, and persistence functionality.
    """

    def __init__(
        self,
        num_classes: int = 5,
        input_channels: int = 3,
        num_channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        use_skip_connections: bool = True,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        gradient_clip_val: float = 1.0,
        device: str | None = None,
        **kwargs: Any,
    ):
        """Initialize the TCN classifier.

        Args:
            num_classes: Number of output classes.
            input_channels: Number of input channels.
            num_channels: Channel sizes for temporal layers.
            kernel_size: Kernel size for temporal convolutions.
            dropout: Dropout rate.
            use_skip_connections: Whether to use skip connections.
            learning_rate: Learning rate for optimizer.
            weight_decay: Weight decay for regularization.
            gradient_clip_val: Gradient clipping value.
            device: Device to use.
            **kwargs: Additional parameters.
        """
        super().__init__(num_classes, **kwargs)

        self.input_channels = input_channels
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip_val = gradient_clip_val

        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Create model
        self.model = TCN(
            input_channels=input_channels,
            num_classes=num_classes,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            use_skip_connections=use_skip_connections,
        ).to(self.device)

        self._config.update(
            {
                "input_channels": input_channels,
                "num_channels": num_channels or [16, 32, 32],
                "kernel_size": kernel_size,
                "dropout": dropout,
                "use_skip_connections": use_skip_connections,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "gradient_clip_val": gradient_clip_val,
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
    ) -> "TCNClassifier":
        """Train the TCN classifier.

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
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

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

                # Gradient clipping
                if self.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_val
                    )

                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            scheduler.step()

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

                # Early stopping with best model checkpoint
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
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

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

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

    def get_model(self) -> TCN:
        """Get the underlying TCN model."""
        return self.model

    def get_feature_extractor(self) -> nn.ModuleList:
        """Get feature extraction layers (temporal blocks)."""
        return self.model.network

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
    def load(cls, path: str | Path) -> "TCNClassifier":
        """Load a model from disk."""
        data = torch.load(path, map_location="cpu")

        instance = cls(num_classes=data["num_classes"], **data["config"])
        instance.model.load_state_dict(data["model_state_dict"])
        instance.is_fitted = True
        return instance
