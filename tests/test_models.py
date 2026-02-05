"""Tests for classification models."""

from __future__ import annotations

import numpy as np
import pytest

from liveedge.models import (
    BaseClassifier,
    CNN1DClassifier,
    RandomForestWrapper,
    SVMWrapper,
    TCNClassifier,
    XGBoostWrapper,
    create_model_from_config,
)


class TestBaseClassifier:
    """Tests for BaseClassifier interface."""

    def test_abstract_methods(self):
        """Test that BaseClassifier is abstract."""
        with pytest.raises(TypeError):
            BaseClassifier()


class TestRandomForestWrapper:
    """Tests for Random Forest wrapper."""

    @pytest.fixture
    def model(self):
        """Create a Random Forest model."""
        return RandomForestWrapper(
            num_classes=5,
            n_estimators=10,
            max_depth=5,
            random_state=42,
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample training/test data."""
        np.random.seed(42)
        n_train = 100
        n_test = 20
        n_features = 50
        n_classes = 5

        X_train = np.random.randn(n_train, n_features).astype(np.float32)
        y_train = np.random.randint(0, n_classes, n_train).astype(np.int64)

        X_test = np.random.randn(n_test, n_features).astype(np.float32)
        y_test = np.random.randint(0, n_classes, n_test).astype(np.int64)

        return X_train, y_train, X_test, y_test

    def test_fit(self, model, sample_data):
        """Test model fitting."""
        X_train, y_train, _, _ = sample_data
        model.fit(X_train, y_train)
        assert model._is_fitted

    def test_predict(self, model, sample_data):
        """Test prediction."""
        X_train, y_train, X_test, _ = sample_data
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        assert predictions.shape == (len(X_test),)
        assert predictions.dtype == np.int64
        assert all(0 <= p < 5 for p in predictions)

    def test_predict_proba(self, model, sample_data):
        """Test probability prediction."""
        X_train, y_train, X_test, _ = sample_data
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)

        assert proba.shape == (len(X_test), 5)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert all(proba.flatten() >= 0)
        assert all(proba.flatten() <= 1)

    def test_save_load(self, model, sample_data, tmp_path):
        """Test model save/load."""
        X_train, y_train, X_test, _ = sample_data
        model.fit(X_train, y_train)

        original_predictions = model.predict(X_test)

        # Save model
        model_path = tmp_path / "rf_model.pt"
        model.save(model_path)

        # Load model
        loaded_model = RandomForestWrapper(num_classes=5)
        loaded_model.load(model_path)

        loaded_predictions = loaded_model.predict(X_test)

        assert np.array_equal(original_predictions, loaded_predictions)


class TestXGBoostWrapper:
    """Tests for XGBoost wrapper."""

    @pytest.fixture
    def model(self):
        """Create an XGBoost model."""
        return XGBoostWrapper(
            num_classes=5,
            n_estimators=10,
            max_depth=3,
            random_state=42,
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        n_train = 100
        n_features = 50
        n_classes = 5

        X_train = np.random.randn(n_train, n_features).astype(np.float32)
        y_train = np.random.randint(0, n_classes, n_train).astype(np.int64)

        X_test = np.random.randn(20, n_features).astype(np.float32)

        return X_train, y_train, X_test

    def test_fit_predict(self, model, sample_data):
        """Test fit and predict."""
        X_train, y_train, X_test = sample_data
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        assert predictions.shape == (len(X_test),)
        assert all(0 <= p < 5 for p in predictions)


class TestSVMWrapper:
    """Tests for SVM wrapper."""

    @pytest.fixture
    def model(self):
        """Create an SVM model."""
        return SVMWrapper(
            num_classes=5,
            kernel="rbf",
            random_state=42,
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        n_train = 100
        n_features = 50
        n_classes = 5

        X_train = np.random.randn(n_train, n_features).astype(np.float32)
        y_train = np.random.randint(0, n_classes, n_train).astype(np.int64)

        X_test = np.random.randn(20, n_features).astype(np.float32)

        return X_train, y_train, X_test

    def test_fit_predict(self, model, sample_data):
        """Test fit and predict."""
        X_train, y_train, X_test = sample_data
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        assert predictions.shape == (len(X_test),)


class TestCNN1DClassifier:
    """Tests for 1D CNN classifier."""

    @pytest.fixture
    def model(self):
        """Create a CNN model."""
        return CNN1DClassifier(
            num_classes=5,
            input_channels=3,
            hidden_channels=[16, 32],
            kernel_size=3,
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample window data."""
        np.random.seed(42)
        n_train = 50
        n_test = 10
        window_size = 75
        n_channels = 3
        n_classes = 5

        X_train = np.random.randn(n_train, window_size, n_channels).astype(np.float32)
        y_train = np.random.randint(0, n_classes, n_train).astype(np.int64)

        X_test = np.random.randn(n_test, window_size, n_channels).astype(np.float32)

        return X_train, y_train, X_test

    def test_fit(self, model, sample_data):
        """Test model fitting."""
        X_train, y_train, _ = sample_data
        model.fit(X_train, y_train, epochs=2, batch_size=16, verbose=False)
        assert model._is_fitted

    def test_predict(self, model, sample_data):
        """Test prediction."""
        X_train, y_train, X_test = sample_data
        model.fit(X_train, y_train, epochs=2, batch_size=16, verbose=False)

        predictions = model.predict(X_test)

        assert predictions.shape == (len(X_test),)
        assert predictions.dtype == np.int64

    def test_predict_proba(self, model, sample_data):
        """Test probability prediction."""
        X_train, y_train, X_test = sample_data
        model.fit(X_train, y_train, epochs=2, batch_size=16, verbose=False)

        proba = model.predict_proba(X_test)

        assert proba.shape == (len(X_test), 5)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


class TestTCNClassifier:
    """Tests for TCN classifier."""

    @pytest.fixture
    def model(self):
        """Create a TCN model."""
        return TCNClassifier(
            num_classes=5,
            input_channels=3,
            num_channels=[8, 16],
            kernel_size=3,
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample window data."""
        np.random.seed(42)
        n_train = 50
        n_test = 10
        window_size = 75
        n_channels = 3
        n_classes = 5

        X_train = np.random.randn(n_train, window_size, n_channels).astype(np.float32)
        y_train = np.random.randint(0, n_classes, n_train).astype(np.int64)

        X_test = np.random.randn(n_test, window_size, n_channels).astype(np.float32)

        return X_train, y_train, X_test

    def test_fit(self, model, sample_data):
        """Test model fitting."""
        X_train, y_train, _ = sample_data
        model.fit(X_train, y_train, epochs=2, batch_size=16, verbose=False)
        assert model._is_fitted

    def test_predict(self, model, sample_data):
        """Test prediction."""
        X_train, y_train, X_test = sample_data
        model.fit(X_train, y_train, epochs=2, batch_size=16, verbose=False)

        predictions = model.predict(X_test)

        assert predictions.shape == (len(X_test),)


class TestModelFactory:
    """Tests for model factory."""

    def test_create_random_forest(self):
        """Test creating Random Forest from config."""
        from omegaconf import OmegaConf

        config = OmegaConf.create({
            "name": "random_forest",
            "type": "traditional",
            "n_estimators": 10,
            "max_depth": 5,
        })

        model = create_model_from_config(config, num_classes=5)

        assert isinstance(model, RandomForestWrapper)

    def test_create_xgboost(self):
        """Test creating XGBoost from config."""
        from omegaconf import OmegaConf

        config = OmegaConf.create({
            "name": "xgboost",
            "type": "traditional",
            "n_estimators": 10,
            "max_depth": 3,
        })

        model = create_model_from_config(config, num_classes=5)

        assert isinstance(model, XGBoostWrapper)

    def test_create_cnn(self):
        """Test creating CNN from config."""
        from omegaconf import OmegaConf

        config = OmegaConf.create({
            "name": "cnn1d",
            "type": "deep_learning",
            "input_channels": 3,
            "hidden_channels": [16, 32],
            "kernel_size": 3,
        })

        model = create_model_from_config(config, num_classes=5)

        assert isinstance(model, CNN1DClassifier)

    def test_create_tcn(self):
        """Test creating TCN from config."""
        from omegaconf import OmegaConf

        config = OmegaConf.create({
            "name": "tcn",
            "type": "deep_learning",
            "input_channels": 3,
            "num_channels": [16, 32],
            "kernel_size": 3,
        })

        model = create_model_from_config(config, num_classes=5)

        assert isinstance(model, TCNClassifier)

    def test_invalid_model_name(self):
        """Test error on invalid model name."""
        from omegaconf import OmegaConf

        config = OmegaConf.create({
            "name": "invalid_model",
            "type": "traditional",
        })

        with pytest.raises(ValueError):
            create_model_from_config(config, num_classes=5)
