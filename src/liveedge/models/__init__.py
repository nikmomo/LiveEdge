"""Classification models module for LiveEdge.

This module provides behavior classification models including traditional ML
and deep learning architectures.
"""

from liveedge.models.base import BaseClassifier, DeepLearningClassifier
from liveedge.models.cnn import CNN1D, CNN1DClassifier
from liveedge.models.factory import (
    create_model,
    create_model_from_config,
    get_available_models,
    get_model_info,
    load_model,
)
from liveedge.models.tcn import TCN, TCNClassifier
from liveedge.models.traditional import (
    RandomForestWrapper,
    SVMWrapper,
    XGBoostWrapper,
)

__all__ = [
    # Base classes
    "BaseClassifier",
    "DeepLearningClassifier",
    # Traditional ML
    "RandomForestWrapper",
    "XGBoostWrapper",
    "SVMWrapper",
    # Deep Learning
    "CNN1D",
    "CNN1DClassifier",
    "TCN",
    "TCNClassifier",
    # Factory
    "create_model",
    "create_model_from_config",
    "load_model",
    "get_available_models",
    "get_model_info",
]
