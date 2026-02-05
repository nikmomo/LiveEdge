"""Behavior clustering module for LiveEdge.

This module provides tools for clustering fine-grained behaviors into
adaptive sampling groups based on motion dynamics.
"""

from liveedge.clustering.clustering import (
    DEFAULT_CATTLE_CLUSTERING,
    BehaviorCluster,
    ClusteringResult,
    cluster_behaviors,
    create_manual_clustering,
    update_cluster_frequencies,
)
from liveedge.clustering.feature_extraction import (
    compute_behavior_dynamics,
    create_clustering_feature_matrix,
    estimate_minimum_sampling_rate,
    normalize_clustering_features,
)
from liveedge.clustering.validation import (
    ClusterValidationMetrics,
    compute_cluster_separation,
    compute_clustering_metrics,
    compute_variance_metrics,
    find_optimal_n_clusters,
    validate_cluster_consistency,
)

__all__ = [
    # Clustering
    "BehaviorCluster",
    "ClusteringResult",
    "cluster_behaviors",
    "create_manual_clustering",
    "update_cluster_frequencies",
    "DEFAULT_CATTLE_CLUSTERING",
    # Feature extraction
    "compute_behavior_dynamics",
    "create_clustering_feature_matrix",
    "normalize_clustering_features",
    "estimate_minimum_sampling_rate",
    # Validation
    "ClusterValidationMetrics",
    "compute_clustering_metrics",
    "compute_variance_metrics",
    "validate_cluster_consistency",
    "find_optimal_n_clusters",
    "compute_cluster_separation",
]
