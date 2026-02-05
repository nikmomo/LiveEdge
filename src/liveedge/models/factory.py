"""Model factory for configuration-based instantiation.

This module provides factory functions for creating classifiers
from configuration dictionaries or files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from liveedge.models.base import BaseClassifier


def create_model(
    model_type: str,
    num_classes: int,
    **kwargs: Any,
) -> BaseClassifier:
    """Create a classifier from type name and parameters.

    Args:
        model_type: Type of model ("random_forest", "xgboost", "svm", "cnn_1d", "tcn").
        num_classes: Number of output classes.
        **kwargs: Model-specific parameters.

    Returns:
        Instantiated classifier.

    Raises:
        ValueError: If unknown model type is specified.

    Example:
        >>> model = create_model("random_forest", num_classes=5, n_estimators=100)
        >>> model = create_model("tcn", num_classes=5, num_channels=[16, 32])
    """
    model_type = model_type.lower().replace("-", "_")

    if model_type == "random_forest":
        from liveedge.models.traditional import RandomForestWrapper

        return RandomForestWrapper(num_classes=num_classes, **kwargs)

    elif model_type == "xgboost":
        from liveedge.models.traditional import XGBoostWrapper

        return XGBoostWrapper(num_classes=num_classes, **kwargs)

    elif model_type == "svm":
        from liveedge.models.traditional import SVMWrapper

        return SVMWrapper(num_classes=num_classes, **kwargs)

    elif model_type in ("cnn_1d", "cnn1d", "cnn"):
        from liveedge.models.cnn import CNN1DClassifier

        return CNN1DClassifier(num_classes=num_classes, **kwargs)

    elif model_type == "tcn":
        from liveedge.models.tcn import TCNClassifier

        return TCNClassifier(num_classes=num_classes, **kwargs)

    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Supported types: random_forest, xgboost, svm, cnn_1d, tcn"
        )


def create_model_from_config(
    config: DictConfig | dict[str, Any],
    num_classes: int | None = None,
) -> BaseClassifier:
    """Create a classifier from a configuration object.

    Args:
        config: Configuration dictionary or DictConfig with model settings.
            Must contain "name" or "type" key specifying model type.
            May contain "params" key with model parameters.
        num_classes: Number of classes (overrides config if provided).

    Returns:
        Instantiated classifier.

    Raises:
        ValueError: If model type is not specified in config.

    Example:
        >>> config = {
        ...     "name": "random_forest",
        ...     "params": {"n_estimators": 100, "max_depth": 20}
        ... }
        >>> model = create_model_from_config(config, num_classes=5)
    """
    if isinstance(config, DictConfig):
        from omegaconf import OmegaConf

        config = OmegaConf.to_container(config, resolve=True)

    # Get model type
    model_type = config.get("name") or config.get("type") or config.get("algorithm")
    if model_type is None:
        raise ValueError("Model configuration must specify 'name', 'type', or 'algorithm'")

    # Get parameters
    params = config.get("params", {})
    if not params:
        # Look for parameters at top level (excluding metadata keys)
        exclude_keys = {"name", "type", "algorithm", "params", "architecture", "training"}
        params = {k: v for k, v in config.items() if k not in exclude_keys}

    # Handle nested architecture parameters
    if "architecture" in config:
        params.update(config["architecture"])

    # Get num_classes
    if num_classes is None:
        num_classes = (
            params.pop("num_classes", None)
            or config.get("num_classes")
            or config.get("architecture", {}).get("num_classes")
        )
        if num_classes is None:
            raise ValueError("num_classes must be specified in config or as argument")

    return create_model(model_type, num_classes=num_classes, **params)


def load_model(path: str | Path) -> BaseClassifier:
    """Load a classifier from a saved file.

    Automatically detects model type from file extension and contents.

    Args:
        path: Path to the saved model file.

    Returns:
        Loaded classifier.

    Raises:
        ValueError: If model type cannot be determined.
    """
    import pickle

    import torch

    path = Path(path)

    # Try loading as PyTorch model
    if path.suffix in (".pt", ".pth"):
        data = torch.load(path, map_location="cpu")

        # Determine model type from config
        config = data.get("config", {})
        if "conv_channels" in config:
            from liveedge.models.cnn import CNN1DClassifier

            return CNN1DClassifier.load(path)
        elif "num_channels" in config and "kernel_size" in config:
            from liveedge.models.tcn import TCNClassifier

            return TCNClassifier.load(path)
        else:
            raise ValueError(f"Cannot determine deep learning model type from {path}")

    # Try loading as pickle (traditional ML)
    elif path.suffix in (".pkl", ".pickle"):
        with open(path, "rb") as f:
            data = pickle.load(f)

        model = data.get("model")
        if model is None:
            raise ValueError(f"Invalid model file: {path}")

        # Determine model type
        model_class = type(model).__name__

        if "RandomForest" in model_class:
            from liveedge.models.traditional import RandomForestWrapper

            return RandomForestWrapper.load(path)
        elif "XGB" in model_class:
            from liveedge.models.traditional import XGBoostWrapper

            return XGBoostWrapper.load(path)
        elif "SVC" in model_class or "SVM" in model_class:
            from liveedge.models.traditional import SVMWrapper

            return SVMWrapper.load(path)
        else:
            raise ValueError(f"Unknown model type in {path}: {model_class}")

    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")


def get_available_models() -> list[str]:
    """Get list of available model types.

    Returns:
        List of model type names.
    """
    return ["random_forest", "xgboost", "svm", "cnn_1d", "tcn"]


def get_model_info(model_type: str) -> dict[str, Any]:
    """Get information about a model type.

    Args:
        model_type: Type of model.

    Returns:
        Dictionary with model information.
    """
    info = {
        "random_forest": {
            "name": "Random Forest",
            "type": "traditional",
            "input_format": "features",
            "supports_proba": True,
            "edge_friendly": True,
            "typical_inference_ms": 3.0,
        },
        "xgboost": {
            "name": "XGBoost",
            "type": "traditional",
            "input_format": "features",
            "supports_proba": True,
            "edge_friendly": True,
            "typical_inference_ms": 5.0,
        },
        "svm": {
            "name": "Support Vector Machine",
            "type": "traditional",
            "input_format": "features",
            "supports_proba": True,
            "edge_friendly": True,
            "typical_inference_ms": 2.0,
        },
        "cnn_1d": {
            "name": "1D Convolutional Neural Network",
            "type": "deep_learning",
            "input_format": "sequences",
            "supports_proba": True,
            "edge_friendly": True,
            "typical_inference_ms": 15.0,
        },
        "tcn": {
            "name": "Temporal Convolutional Network",
            "type": "deep_learning",
            "input_format": "sequences",
            "supports_proba": True,
            "edge_friendly": True,
            "typical_inference_ms": 30.0,
        },
    }

    model_type = model_type.lower().replace("-", "_")
    if model_type not in info:
        raise ValueError(f"Unknown model type: {model_type}")

    return info[model_type]
