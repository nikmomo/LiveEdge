"""Traditional machine learning classifiers.

This module provides wrappers for traditional ML algorithms (Random Forest,
XGBoost, SVM) that implement the BaseClassifier interface.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from liveedge.models.base import BaseClassifier


class RandomForestWrapper(BaseClassifier):
    """Wrapper for scikit-learn Random Forest classifier.

    Attributes:
        model: The underlying RandomForestClassifier.
    """

    def __init__(
        self,
        num_classes: int,
        n_estimators: int = 100,
        max_depth: int | None = 20,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        max_features: str | int | float = "sqrt",
        class_weight: str | dict | None = "balanced",
        n_jobs: int = -1,
        random_state: int | None = 42,
        **kwargs: Any,
    ):
        """Initialize the Random Forest classifier.

        Args:
            num_classes: Number of output classes.
            n_estimators: Number of trees in the forest.
            max_depth: Maximum depth of trees.
            min_samples_split: Minimum samples required to split a node.
            min_samples_leaf: Minimum samples required at a leaf node.
            max_features: Number of features to consider for best split.
            class_weight: Weights for classes.
            n_jobs: Number of parallel jobs.
            random_state: Random seed for reproducibility.
            **kwargs: Additional parameters.
        """
        super().__init__(num_classes, **kwargs)

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            n_jobs=n_jobs,
            random_state=random_state,
        )

        self._config.update(
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
                "max_features": max_features,
                "class_weight": class_weight,
            }
        )

    def fit(
        self,
        X: NDArray[np.float32],
        y: NDArray[np.int64],
        X_val: NDArray[np.float32] | None = None,
        y_val: NDArray[np.int64] | None = None,
    ) -> "RandomForestWrapper":
        """Train the Random Forest classifier.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Training labels of shape (n_samples,).
            X_val: Ignored (no validation in RF).
            y_val: Ignored (no validation in RF).

        Returns:
            Self for method chaining.
        """
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: NDArray[np.float32]) -> NDArray[np.int64]:
        """Predict class labels.

        Args:
            X: Input features.

        Returns:
            Predicted class labels.
        """
        return self.model.predict(X).astype(np.int64)

    def predict_proba(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Predict class probabilities.

        Args:
            X: Input features.

        Returns:
            Class probabilities.
        """
        return self.model.predict_proba(X).astype(np.float32)

    def get_feature_importance(self) -> NDArray[np.float64]:
        """Get feature importance scores.

        Returns:
            Feature importance array.
        """
        return self.model.feature_importances_

    def save(self, path: str | Path) -> None:
        """Save the model to disk.

        Args:
            path: Path to save the model.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "num_classes": self.num_classes,
                    "config": self._config,
                },
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> "RandomForestWrapper":
        """Load a model from disk.

        Args:
            path: Path to the saved model.

        Returns:
            Loaded classifier instance.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        instance = cls(num_classes=data["num_classes"], **data["config"])
        instance.model = data["model"]
        instance.is_fitted = True
        return instance


class XGBoostWrapper(BaseClassifier):
    """Wrapper for XGBoost classifier.

    Attributes:
        model: The underlying XGBClassifier.
    """

    def __init__(
        self,
        num_classes: int,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        gamma: float = 0.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        scale_pos_weight: float = 1.0,
        n_jobs: int = -1,
        random_state: int | None = 42,
        **kwargs: Any,
    ):
        """Initialize the XGBoost classifier.

        Args:
            num_classes: Number of output classes.
            n_estimators: Number of boosting rounds.
            max_depth: Maximum tree depth.
            learning_rate: Boosting learning rate.
            subsample: Subsample ratio of training instances.
            colsample_bytree: Subsample ratio of columns.
            gamma: Minimum loss reduction for split.
            reg_alpha: L1 regularization.
            reg_lambda: L2 regularization.
            scale_pos_weight: Balance of positive and negative weights.
            n_jobs: Number of parallel jobs.
            random_state: Random seed.
            **kwargs: Additional parameters.
        """
        super().__init__(num_classes, **kwargs)

        # Import here to make XGBoost optional
        import xgboost as xgb

        objective = "multi:softprob" if num_classes > 2 else "binary:logistic"

        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            scale_pos_weight=scale_pos_weight,
            n_jobs=n_jobs,
            random_state=random_state,
            objective=objective,
            num_class=num_classes if num_classes > 2 else None,
            use_label_encoder=False,
            eval_metric="mlogloss" if num_classes > 2 else "logloss",
        )

        self._config.update(
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
            }
        )

    def fit(
        self,
        X: NDArray[np.float32],
        y: NDArray[np.int64],
        X_val: NDArray[np.float32] | None = None,
        y_val: NDArray[np.int64] | None = None,
    ) -> "XGBoostWrapper":
        """Train the XGBoost classifier.

        Args:
            X: Training features.
            y: Training labels.
            X_val: Optional validation features for early stopping.
            y_val: Optional validation labels.

        Returns:
            Self for method chaining.
        """
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        self.model.fit(
            X,
            y,
            eval_set=eval_set,
            verbose=False,
        )
        self.is_fitted = True
        return self

    def predict(self, X: NDArray[np.float32]) -> NDArray[np.int64]:
        """Predict class labels."""
        return self.model.predict(X).astype(np.int64)

    def predict_proba(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Predict class probabilities."""
        return self.model.predict_proba(X).astype(np.float32)

    def get_feature_importance(self) -> NDArray[np.float64]:
        """Get feature importance scores."""
        return self.model.feature_importances_

    def save(self, path: str | Path) -> None:
        """Save the model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "num_classes": self.num_classes,
                    "config": self._config,
                },
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostWrapper":
        """Load a model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        instance = cls(num_classes=data["num_classes"], **data["config"])
        instance.model = data["model"]
        instance.is_fitted = True
        return instance


class SVMWrapper(BaseClassifier):
    """Wrapper for Support Vector Machine classifier.

    Attributes:
        model: The underlying SVC.
    """

    def __init__(
        self,
        num_classes: int,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: str | float = "scale",
        class_weight: str | dict | None = "balanced",
        probability: bool = True,
        random_state: int | None = 42,
        **kwargs: Any,
    ):
        """Initialize the SVM classifier.

        Args:
            num_classes: Number of output classes.
            kernel: Kernel type ("linear", "poly", "rbf", "sigmoid").
            C: Regularization parameter.
            gamma: Kernel coefficient.
            class_weight: Weights for classes.
            probability: Whether to enable probability estimates.
            random_state: Random seed.
            **kwargs: Additional parameters.
        """
        super().__init__(num_classes, **kwargs)

        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            class_weight=class_weight,
            probability=probability,
            random_state=random_state,
        )

        self._config.update(
            {
                "kernel": kernel,
                "C": C,
                "gamma": gamma,
                "class_weight": class_weight,
            }
        )

    def fit(
        self,
        X: NDArray[np.float32],
        y: NDArray[np.int64],
        X_val: NDArray[np.float32] | None = None,
        y_val: NDArray[np.int64] | None = None,
    ) -> "SVMWrapper":
        """Train the SVM classifier."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: NDArray[np.float32]) -> NDArray[np.int64]:
        """Predict class labels."""
        return self.model.predict(X).astype(np.int64)

    def predict_proba(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Predict class probabilities."""
        return self.model.predict_proba(X).astype(np.float32)

    def save(self, path: str | Path) -> None:
        """Save the model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "num_classes": self.num_classes,
                    "config": self._config,
                },
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> "SVMWrapper":
        """Load a model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        instance = cls(num_classes=data["num_classes"], **data["config"])
        instance.model = data["model"]
        instance.is_fitted = True
        return instance
