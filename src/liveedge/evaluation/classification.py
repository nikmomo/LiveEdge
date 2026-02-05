"""Classification evaluation metrics.

This module provides metrics for evaluating behavior classification performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class ClassificationMetrics:
    """Classification evaluation metrics.

    Attributes:
        accuracy: Overall accuracy.
        macro_f1: Macro-averaged F1 score.
        weighted_f1: Weighted F1 score.
        per_class_f1: Per-class F1 scores.
        per_class_precision: Per-class precision scores.
        per_class_recall: Per-class recall scores.
        confusion_matrix: Confusion matrix.
        transition_accuracy: Accuracy on transition windows.
        class_names: List of class names.
    """

    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_f1: dict[str, float]
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    confusion_matrix: NDArray[np.int64]
    transition_accuracy: float | None = None
    class_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "per_class_f1": self.per_class_f1,
            "per_class_precision": self.per_class_precision,
            "per_class_recall": self.per_class_recall,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "transition_accuracy": self.transition_accuracy,
        }

    def summary(self) -> str:
        """Get summary string."""
        lines = [
            f"Accuracy: {self.accuracy:.4f}",
            f"Macro F1: {self.macro_f1:.4f}",
            f"Weighted F1: {self.weighted_f1:.4f}",
        ]
        if self.transition_accuracy is not None:
            lines.append(f"Transition Accuracy: {self.transition_accuracy:.4f}")
        return "\n".join(lines)


def compute_classification_metrics(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    class_names: list[str] | None = None,
    compute_transition: bool = True,
) -> ClassificationMetrics:
    """Compute comprehensive classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: Optional list of class names.
        compute_transition: Whether to compute transition accuracy.

    Returns:
        ClassificationMetrics with all computed values.
    """
    if class_names is None:
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        class_names = [f"class_{i}" for i in unique_labels]

    n_classes = len(class_names)

    # Basic metrics
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Per-class metrics
    f1_scores = f1_score(y_true, y_pred, average=None, zero_division=0)
    precision_scores = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_scores = recall_score(y_true, y_pred, average=None, zero_division=0)

    per_class_f1 = {
        class_names[i]: float(f1_scores[i]) for i in range(min(len(class_names), len(f1_scores)))
    }
    per_class_precision = {
        class_names[i]: float(precision_scores[i])
        for i in range(min(len(class_names), len(precision_scores)))
    }
    per_class_recall = {
        class_names[i]: float(recall_scores[i])
        for i in range(min(len(class_names), len(recall_scores)))
    }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))

    # Transition accuracy
    transition_acc = None
    if compute_transition and len(y_true) > 1:
        transition_acc = compute_transition_accuracy(y_true, y_pred)

    return ClassificationMetrics(
        accuracy=accuracy,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        per_class_f1=per_class_f1,
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        confusion_matrix=cm,
        transition_accuracy=transition_acc,
        class_names=class_names,
    )


def compute_transition_accuracy(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    window_size: int = 5,
) -> float:
    """Compute accuracy specifically at state transitions.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        window_size: Window around transitions to consider.

    Returns:
        Accuracy at transition points.
    """
    # Find transition points
    transitions = np.where(np.diff(y_true) != 0)[0] + 1

    if len(transitions) == 0:
        return 1.0  # No transitions, perfect by default

    # Collect indices around transitions
    transition_indices = set()
    for t in transitions:
        for i in range(max(0, t - window_size), min(len(y_true), t + window_size + 1)):
            transition_indices.add(i)

    transition_indices = list(transition_indices)

    if len(transition_indices) == 0:
        return 1.0

    # Compute accuracy at transitions
    correct = sum(y_true[i] == y_pred[i] for i in transition_indices)
    return correct / len(transition_indices)


def compute_per_class_support(
    y_true: NDArray[np.int64],
    class_names: list[str],
) -> dict[str, int]:
    """Compute number of samples per class.

    Args:
        y_true: Ground truth labels.
        class_names: List of class names.

    Returns:
        Dictionary mapping class name to sample count.
    """
    unique, counts = np.unique(y_true, return_counts=True)
    support = {}
    for i, name in enumerate(class_names):
        if i in unique:
            idx = np.where(unique == i)[0][0]
            support[name] = int(counts[idx])
        else:
            support[name] = 0
    return support
