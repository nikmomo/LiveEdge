"""Behavior clustering algorithms.

This module provides clustering algorithms for grouping fine-grained behaviors
into adaptive sampling clusters based on their motion dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score


class BehaviorCluster(Enum):
    """Behavioral cluster definitions for adaptive sampling.

    Each cluster represents behaviors with similar motion dynamics
    and therefore similar sampling requirements.
    """

    INACTIVE = auto()  # Lying, sleeping, standing idle
    RUMINATING = auto()  # Ruminating (lying/standing)
    FEEDING = auto()  # Grazing, eating, drinking
    LOCOMOTION = auto()  # Walking, exploring
    HIGH_ACTIVITY = auto()  # Running, playing, social

    @classmethod
    def from_name(cls, name: str) -> "BehaviorCluster":
        """Get cluster from name string.

        Args:
            name: Cluster name (case insensitive).

        Returns:
            BehaviorCluster enum value.

        Raises:
            ValueError: If name is not a valid cluster.
        """
        name_upper = name.upper()
        for cluster in cls:
            if cluster.name == name_upper:
                return cluster
        raise ValueError(f"Unknown cluster name: {name}")

    @property
    def default_sampling_rate(self) -> int:
        """Get default sampling rate for this cluster in Hz."""
        rates = {
            BehaviorCluster.INACTIVE: 5,
            BehaviorCluster.RUMINATING: 10,
            BehaviorCluster.FEEDING: 15,
            BehaviorCluster.LOCOMOTION: 25,
            BehaviorCluster.HIGH_ACTIVITY: 50,
        }
        return rates[self]


@dataclass
class ClusteringResult:
    """Results from behavior clustering.

    Attributes:
        behavior_to_cluster: Mapping from behavior name to cluster.
        cluster_to_behaviors: Mapping from cluster to list of behaviors.
        feature_matrix: Feature matrix used for clustering.
        silhouette_score: Silhouette score of the clustering.
        cluster_frequencies: Recommended sampling rates per cluster.
        linkage_matrix: Linkage matrix (for hierarchical clustering).
        cluster_labels: Cluster labels for each behavior.
        behavior_names: List of behavior names in order.
    """

    behavior_to_cluster: dict[str, BehaviorCluster]
    cluster_to_behaviors: dict[BehaviorCluster, list[str]]
    feature_matrix: NDArray[np.float32]
    silhouette_score: float
    cluster_frequencies: dict[BehaviorCluster, float]
    linkage_matrix: NDArray[np.float64] | None = None
    cluster_labels: NDArray[np.int64] = field(default_factory=lambda: np.array([]))
    behavior_names: list[str] = field(default_factory=list)

    def get_cluster(self, behavior: str) -> BehaviorCluster:
        """Get cluster for a behavior.

        Args:
            behavior: Behavior name.

        Returns:
            BehaviorCluster for the behavior.

        Raises:
            KeyError: If behavior is not in the clustering.
        """
        return self.behavior_to_cluster[behavior]

    def get_sampling_rate(self, behavior: str) -> float:
        """Get recommended sampling rate for a behavior.

        Args:
            behavior: Behavior name.

        Returns:
            Recommended sampling rate in Hz.
        """
        cluster = self.get_cluster(behavior)
        return self.cluster_frequencies[cluster]


def cluster_behaviors(
    behavior_features: dict[str, dict[str, float]],
    n_clusters: int = 5,
    method: str = "hierarchical",
    linkage_method: str = "ward",
) -> ClusteringResult:
    """Cluster behaviors based on motion dynamics features.

    Args:
        behavior_features: Dictionary mapping behavior name to feature dict.
        n_clusters: Number of clusters to create.
        method: Clustering method ("hierarchical", "kmeans", "spectral").
        linkage_method: Linkage method for hierarchical clustering.

    Returns:
        ClusteringResult with clustering information.

    Raises:
        ValueError: If unknown clustering method is specified.
    """
    from liveedge.clustering.feature_extraction import (
        create_clustering_feature_matrix,
        normalize_clustering_features,
    )

    # Create and normalize feature matrix
    feature_matrix, behavior_names, feature_names = create_clustering_feature_matrix(
        behavior_features
    )
    normalized_features = normalize_clustering_features(feature_matrix)

    # Perform clustering
    linkage_matrix = None

    if method == "hierarchical":
        linkage_matrix = linkage(normalized_features, method=linkage_method)
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion="maxclust") - 1

    elif method == "kmeans":
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(normalized_features)

    elif method == "spectral":
        spectral = SpectralClustering(
            n_clusters=n_clusters, random_state=42, affinity="rbf"
        )
        cluster_labels = spectral.fit_predict(normalized_features)

    else:
        raise ValueError(f"Unknown clustering method: {method}")

    cluster_labels = cluster_labels.astype(np.int64)

    # Calculate silhouette score
    if len(np.unique(cluster_labels)) > 1:
        sil_score = float(silhouette_score(normalized_features, cluster_labels))
    else:
        sil_score = 0.0

    # Map cluster indices to BehaviorCluster enum
    # Sort clusters by average activity level (based on jerk/frequency)
    cluster_activity = {}
    for i in range(n_clusters):
        mask = cluster_labels == i
        if mask.any():
            # Use mean of normalized features as activity proxy
            cluster_activity[i] = float(np.mean(normalized_features[mask]))

    # Sort clusters by activity level
    sorted_clusters = sorted(cluster_activity.keys(), key=lambda x: cluster_activity[x])

    # Map to BehaviorCluster (lowest activity -> INACTIVE, highest -> HIGH_ACTIVITY)
    cluster_enum_values = list(BehaviorCluster)[:n_clusters]
    cluster_index_to_enum = {
        idx: cluster_enum_values[rank] for rank, idx in enumerate(sorted_clusters)
    }

    # Create mappings
    behavior_to_cluster: dict[str, BehaviorCluster] = {}
    cluster_to_behaviors: dict[BehaviorCluster, list[str]] = {c: [] for c in BehaviorCluster}

    for i, behavior in enumerate(behavior_names):
        cluster_idx = cluster_labels[i]
        cluster_enum = cluster_index_to_enum[cluster_idx]
        behavior_to_cluster[behavior] = cluster_enum
        cluster_to_behaviors[cluster_enum].append(behavior)

    # Set default cluster frequencies
    cluster_frequencies = {cluster: float(cluster.default_sampling_rate) for cluster in BehaviorCluster}

    return ClusteringResult(
        behavior_to_cluster=behavior_to_cluster,
        cluster_to_behaviors=cluster_to_behaviors,
        feature_matrix=feature_matrix,
        silhouette_score=sil_score,
        cluster_frequencies=cluster_frequencies,
        linkage_matrix=linkage_matrix,
        cluster_labels=cluster_labels,
        behavior_names=behavior_names,
    )


def update_cluster_frequencies(
    result: ClusteringResult,
    behavior_min_rates: dict[str, float],
) -> ClusteringResult:
    """Update cluster frequencies based on behavior-specific minimum rates.

    Takes the maximum of all behavior minimum rates within each cluster.

    Args:
        result: Existing ClusteringResult.
        behavior_min_rates: Dictionary mapping behavior name to minimum rate.

    Returns:
        Updated ClusteringResult with new cluster_frequencies.
    """
    new_frequencies = {}

    for cluster in BehaviorCluster:
        behaviors_in_cluster = result.cluster_to_behaviors.get(cluster, [])
        if behaviors_in_cluster:
            # Take maximum of behavior minimum rates
            max_rate = max(
                behavior_min_rates.get(b, cluster.default_sampling_rate)
                for b in behaviors_in_cluster
            )
            new_frequencies[cluster] = max_rate
        else:
            new_frequencies[cluster] = float(cluster.default_sampling_rate)

    return ClusteringResult(
        behavior_to_cluster=result.behavior_to_cluster,
        cluster_to_behaviors=result.cluster_to_behaviors,
        feature_matrix=result.feature_matrix,
        silhouette_score=result.silhouette_score,
        cluster_frequencies=new_frequencies,
        linkage_matrix=result.linkage_matrix,
        cluster_labels=result.cluster_labels,
        behavior_names=result.behavior_names,
    )


def create_manual_clustering(
    behavior_assignments: dict[str, str],
) -> ClusteringResult:
    """Create a clustering result from manual behavior assignments.

    Args:
        behavior_assignments: Dictionary mapping behavior name to cluster name.
            Cluster names should match BehaviorCluster enum names.

    Returns:
        ClusteringResult with manual assignments.

    Example:
        >>> assignments = {
        ...     "lying_lateral": "INACTIVE",
        ...     "lying_sternal": "INACTIVE",
        ...     "walking": "LOCOMOTION",
        ...     "running": "HIGH_ACTIVITY",
        ... }
        >>> result = create_manual_clustering(assignments)
    """
    behavior_to_cluster: dict[str, BehaviorCluster] = {}
    cluster_to_behaviors: dict[BehaviorCluster, list[str]] = {c: [] for c in BehaviorCluster}

    for behavior, cluster_name in behavior_assignments.items():
        cluster = BehaviorCluster.from_name(cluster_name)
        behavior_to_cluster[behavior] = cluster
        cluster_to_behaviors[cluster].append(behavior)

    # Create dummy feature matrix
    n_behaviors = len(behavior_assignments)
    feature_matrix = np.zeros((n_behaviors, 1), dtype=np.float32)

    cluster_frequencies = {cluster: float(cluster.default_sampling_rate) for cluster in BehaviorCluster}

    return ClusteringResult(
        behavior_to_cluster=behavior_to_cluster,
        cluster_to_behaviors=cluster_to_behaviors,
        feature_matrix=feature_matrix,
        silhouette_score=0.0,
        cluster_frequencies=cluster_frequencies,
        linkage_matrix=None,
        cluster_labels=np.zeros(n_behaviors, dtype=np.int64),
        behavior_names=list(behavior_assignments.keys()),
    )


# Default clustering for cattle behaviors
DEFAULT_CATTLE_CLUSTERING = {
    "lying_lateral": "INACTIVE",
    "lying_sternal": "INACTIVE",
    "standing_idle": "INACTIVE",
    "ruminating_lying": "RUMINATING",
    "ruminating_standing": "RUMINATING",
    "grazing": "FEEDING",
    "eating": "FEEDING",
    "drinking": "FEEDING",
    "walking": "LOCOMOTION",
    "exploring": "LOCOMOTION",
    "running": "HIGH_ACTIVITY",
    "playing": "HIGH_ACTIVITY",
    "social": "HIGH_ACTIVITY",
}
