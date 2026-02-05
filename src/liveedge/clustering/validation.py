"""Cluster validation metrics.

This module provides functions for validating behavior clustering quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)

from liveedge.clustering.clustering import BehaviorCluster, ClusteringResult


@dataclass
class ClusterValidationMetrics:
    """Validation metrics for clustering.

    Attributes:
        silhouette_score: Overall silhouette score (-1 to 1, higher is better).
        silhouette_per_sample: Silhouette score for each sample.
        calinski_harabasz_score: Calinski-Harabasz index (higher is better).
        davies_bouldin_score: Davies-Bouldin index (lower is better).
        within_cluster_variance: Average variance within clusters.
        between_cluster_variance: Variance between cluster centers.
    """

    silhouette_score: float
    silhouette_per_sample: NDArray[np.float64] | None = None
    calinski_harabasz_score: float | None = None
    davies_bouldin_score: float | None = None
    within_cluster_variance: float | None = None
    between_cluster_variance: float | None = None


def compute_clustering_metrics(
    feature_matrix: NDArray[np.float32],
    cluster_labels: NDArray[np.int64],
) -> ClusterValidationMetrics:
    """Compute comprehensive clustering validation metrics.

    Args:
        feature_matrix: Feature matrix of shape (n_samples, n_features).
        cluster_labels: Cluster labels of shape (n_samples,).

    Returns:
        ClusterValidationMetrics with computed values.
    """
    n_unique = len(np.unique(cluster_labels))

    # Silhouette score
    if n_unique > 1:
        sil_score = float(silhouette_score(feature_matrix, cluster_labels))
        sil_per_sample = silhouette_samples(feature_matrix, cluster_labels)
    else:
        sil_score = 0.0
        sil_per_sample = np.zeros(len(cluster_labels))

    # Calinski-Harabasz and Davies-Bouldin
    if n_unique > 1:
        ch_score = float(calinski_harabasz_score(feature_matrix, cluster_labels))
        db_score = float(davies_bouldin_score(feature_matrix, cluster_labels))
    else:
        ch_score = None
        db_score = None

    # Within and between cluster variance
    within_var, between_var = compute_variance_metrics(feature_matrix, cluster_labels)

    return ClusterValidationMetrics(
        silhouette_score=sil_score,
        silhouette_per_sample=sil_per_sample,
        calinski_harabasz_score=ch_score,
        davies_bouldin_score=db_score,
        within_cluster_variance=within_var,
        between_cluster_variance=between_var,
    )


def compute_variance_metrics(
    feature_matrix: NDArray[np.float32],
    cluster_labels: NDArray[np.int64],
) -> tuple[float, float]:
    """Compute within-cluster and between-cluster variance.

    Args:
        feature_matrix: Feature matrix of shape (n_samples, n_features).
        cluster_labels: Cluster labels of shape (n_samples,).

    Returns:
        Tuple of (within_cluster_variance, between_cluster_variance).
    """
    unique_labels = np.unique(cluster_labels)
    n_clusters = len(unique_labels)

    if n_clusters <= 1:
        total_var = float(np.var(feature_matrix))
        return total_var, 0.0

    # Overall centroid
    global_centroid = np.mean(feature_matrix, axis=0)

    # Compute cluster centroids and variances
    cluster_variances = []
    cluster_sizes = []
    cluster_centroids = []

    for label in unique_labels:
        mask = cluster_labels == label
        cluster_data = feature_matrix[mask]

        if len(cluster_data) > 0:
            cluster_centroid = np.mean(cluster_data, axis=0)
            cluster_var = np.mean(np.sum((cluster_data - cluster_centroid) ** 2, axis=1))

            cluster_variances.append(cluster_var)
            cluster_sizes.append(len(cluster_data))
            cluster_centroids.append(cluster_centroid)

    # Within-cluster variance (weighted average)
    total_samples = sum(cluster_sizes)
    within_var = sum(v * s for v, s in zip(cluster_variances, cluster_sizes)) / total_samples

    # Between-cluster variance
    between_var = 0.0
    for centroid, size in zip(cluster_centroids, cluster_sizes):
        between_var += size * np.sum((centroid - global_centroid) ** 2)
    between_var /= total_samples

    return float(within_var), float(between_var)


def validate_cluster_consistency(
    result: ClusteringResult,
    behavior_min_freqs: dict[str, float],
    cv_threshold: float = 0.3,
) -> dict[BehaviorCluster, bool]:
    """Validate that behaviors within clusters have consistent sampling needs.

    A cluster is considered consistent if the coefficient of variation (CV)
    of minimum sampling rates is below the threshold.

    Args:
        result: Clustering result to validate.
        behavior_min_freqs: Minimum sampling rates per behavior.
        cv_threshold: Maximum allowed coefficient of variation.

    Returns:
        Dictionary mapping cluster to validity (True if consistent).
    """
    validity = {}

    for cluster in BehaviorCluster:
        behaviors = result.cluster_to_behaviors.get(cluster, [])

        if not behaviors:
            validity[cluster] = True
            continue

        # Get sampling rates for behaviors in this cluster
        rates = [behavior_min_freqs.get(b, cluster.default_sampling_rate) for b in behaviors]

        if len(rates) == 1:
            validity[cluster] = True
            continue

        # Compute coefficient of variation
        mean_rate = np.mean(rates)
        std_rate = np.std(rates)

        if mean_rate > 0:
            cv = std_rate / mean_rate
            validity[cluster] = cv <= cv_threshold
        else:
            validity[cluster] = True

    return validity


def find_optimal_n_clusters(
    feature_matrix: NDArray[np.float32],
    min_clusters: int = 2,
    max_clusters: int = 8,
    method: str = "silhouette",
) -> tuple[int, dict[int, float]]:
    """Find optimal number of clusters using elbow method or silhouette.

    Args:
        feature_matrix: Feature matrix of shape (n_samples, n_features).
        min_clusters: Minimum number of clusters to try.
        max_clusters: Maximum number of clusters to try.
        method: Evaluation method ("silhouette", "calinski_harabasz", "davies_bouldin").

    Returns:
        Tuple of (optimal_n_clusters, scores_dict).
    """
    from sklearn.cluster import KMeans

    max_clusters = min(max_clusters, len(feature_matrix) - 1)
    scores: dict[int, float] = {}

    for n in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
        labels = kmeans.fit_predict(feature_matrix)

        if method == "silhouette":
            score = silhouette_score(feature_matrix, labels)
        elif method == "calinski_harabasz":
            score = calinski_harabasz_score(feature_matrix, labels)
        elif method == "davies_bouldin":
            # Lower is better, so negate for consistent comparison
            score = -davies_bouldin_score(feature_matrix, labels)
        else:
            raise ValueError(f"Unknown method: {method}")

        scores[n] = float(score)

    # Find optimal (maximum score)
    optimal_n = max(scores, key=scores.get)

    return optimal_n, scores


def compute_cluster_separation(
    feature_matrix: NDArray[np.float32],
    cluster_labels: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Compute pairwise cluster separation matrix.

    Separation is measured as the distance between cluster centroids
    divided by the average within-cluster standard deviation.

    Args:
        feature_matrix: Feature matrix of shape (n_samples, n_features).
        cluster_labels: Cluster labels of shape (n_samples,).

    Returns:
        Separation matrix of shape (n_clusters, n_clusters).
    """
    unique_labels = np.unique(cluster_labels)
    n_clusters = len(unique_labels)

    # Compute centroids and standard deviations
    centroids = []
    stds = []

    for label in unique_labels:
        mask = cluster_labels == label
        cluster_data = feature_matrix[mask]

        centroids.append(np.mean(cluster_data, axis=0))
        stds.append(np.mean(np.std(cluster_data, axis=0)))

    centroids = np.array(centroids)
    stds = np.array(stds)

    # Compute separation matrix
    separation = np.zeros((n_clusters, n_clusters))

    for i in range(n_clusters):
        for j in range(n_clusters):
            if i != j:
                distance = np.linalg.norm(centroids[i] - centroids[j])
                avg_std = (stds[i] + stds[j]) / 2
                if avg_std > 0:
                    separation[i, j] = distance / avg_std
                else:
                    separation[i, j] = distance

    return separation
