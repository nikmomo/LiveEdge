"""Data processing module for LiveEdge.

This module provides tools for loading, preprocessing, feature extraction,
and dataset management for accelerometer data.
"""

from liveedge.data.dataset import (
    AccelerometerDataset,
    MultiSubjectDataset,
    StreamingDataset,
    create_data_loaders,
    load_processed_data,
    save_processed_data,
)
from liveedge.data.features import (
    compute_magnitude,
    extract_all_features,
    extract_dynamics_features,
    extract_features_batch,
    extract_frequency_domain_features,
    extract_periodicity_features,
    extract_statistical_features,
    extract_time_domain_features,
    get_feature_names,
)
from liveedge.data.features_24 import (
    FEATURE_NAMES_24,
    extract_24_features,
    extract_24_features_batch,
    get_feature_names_24,
)
from liveedge.data.filters import (
    apply_filter,
    behavior_specific_filter,
    hampel_filter,
    lowpass_filter,
    median_filter,
    wavelet_denoise,
)
from liveedge.data.preprocessing import (
    NormalizationParams,
    WindowConfig,
    apply_lowpass_filter,
    compute_class_weights,
    create_windows,
    create_windows_by_subject,
    load_raw_data,
    normalize,
    remove_outliers,
    resample_data,
    split_by_subject,
)
from liveedge.data.transforms import (
    AddNoise,
    BaseTransform,
    ChannelDropout,
    Compose,
    MagnitudeWarp,
    Permutation,
    RandomApply,
    RandomRotation,
    RandomScale,
    TimeWarp,
    get_default_transforms,
)

__all__ = [
    # Preprocessing
    "WindowConfig",
    "NormalizationParams",
    "load_raw_data",
    "remove_outliers",
    "normalize",
    "apply_lowpass_filter",
    "resample_data",
    "create_windows",
    "create_windows_by_subject",
    "split_by_subject",
    "compute_class_weights",
    # Features
    "compute_magnitude",
    "extract_time_domain_features",
    "extract_frequency_domain_features",
    "extract_dynamics_features",
    "extract_periodicity_features",
    "extract_statistical_features",
    "extract_all_features",
    "extract_features_batch",
    "get_feature_names",
    # 24-Feature (unified MCU-friendly features)
    "FEATURE_NAMES_24",
    "extract_24_features",
    "extract_24_features_batch",
    "get_feature_names_24",
    # Dataset
    "AccelerometerDataset",
    "StreamingDataset",
    "MultiSubjectDataset",
    "load_processed_data",
    "save_processed_data",
    "create_data_loaders",
    # Transforms
    "BaseTransform",
    "Compose",
    "RandomApply",
    "AddNoise",
    "RandomScale",
    "RandomRotation",
    "TimeWarp",
    "Permutation",
    "ChannelDropout",
    "MagnitudeWarp",
    "get_default_transforms",
]
