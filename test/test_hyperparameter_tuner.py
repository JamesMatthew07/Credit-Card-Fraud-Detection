"""
Unit tests for hyperparameter_tuner module.
"""
import pytest
import numpy as np
from pathlib import Path
import sys
from sklearn.ensemble import RandomForestClassifier

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hyperparameter_tuner import tune_hyperparameters


class TestTuneHyperparameters:
    """Test suite for tune_hyperparameters function."""

    @pytest.fixture
    def small_param_grid(self):
        """Create a small parameter grid for faster testing."""
        return {
            'n_estimators': [10, 20],
            'max_depth': [3, 5]
        }

    @pytest.fixture
    def single_param_grid(self):
        """Create a parameter grid with single values."""
        return {
            'n_estimators': [10],
            'max_depth': [5]
        }

    def test_tune_returns_model_and_params(self, sample_features, sample_target, small_param_grid):
        """Test that tuning returns model and parameters."""
        best_model, best_params = tune_hyperparameters(
            sample_features, sample_target, small_param_grid, cv=2
        )

        assert isinstance(best_model, RandomForestClassifier)
        assert isinstance(best_params, dict)

    def test_tune_best_params_in_grid(self, sample_features, sample_target, small_param_grid):
        """Test that best parameters are from the grid."""
        best_model, best_params = tune_hyperparameters(
            sample_features, sample_target, small_param_grid, cv=2
        )

        # Check that best params are in the grid
        assert best_params['n_estimators'] in small_param_grid['n_estimators']
        assert best_params['max_depth'] in small_param_grid['max_depth']

    def test_tune_model_is_fitted(self, sample_features, sample_target, small_param_grid):
        """Test that returned model is already fitted."""
        best_model, best_params = tune_hyperparameters(
            sample_features, sample_target, small_param_grid, cv=2
        )

        # Fitted model should have estimators_
        assert hasattr(best_model, 'estimators_')
        assert len(best_model.estimators_) > 0

    def test_tune_can_predict(self, sample_features, sample_target, small_param_grid):
        """Test that tuned model can make predictions."""
        best_model, best_params = tune_hyperparameters(
            sample_features, sample_target, small_param_grid, cv=2
        )

        predictions = best_model.predict(sample_features)
        assert len(predictions) == len(sample_target)
        assert set(predictions).issubset({0, 1})

    def test_tune_with_recall_scoring(self, sample_features, sample_target, small_param_grid):
        """Test tuning with recall scoring metric."""
        best_model, best_params = tune_hyperparameters(
            sample_features, sample_target, small_param_grid, scoring='recall', cv=2
        )

        assert isinstance(best_model, RandomForestClassifier)

    def test_tune_with_precision_scoring(self, sample_features, sample_target, small_param_grid):
        """Test tuning with precision scoring metric."""
        best_model, best_params = tune_hyperparameters(
            sample_features, sample_target, small_param_grid, scoring='precision', cv=2
        )

        assert isinstance(best_model, RandomForestClassifier)

    def test_tune_with_f1_scoring(self, sample_features, sample_target, small_param_grid):
        """Test tuning with F1 scoring metric."""
        best_model, best_params = tune_hyperparameters(
            sample_features, sample_target, small_param_grid, scoring='f1', cv=2
        )

        assert isinstance(best_model, RandomForestClassifier)

    def test_tune_with_different_cv_folds(self, sample_features, sample_target, small_param_grid):
        """Test tuning with different number of CV folds."""
        # Test with 2 folds
        best_model_2, _ = tune_hyperparameters(
            sample_features, sample_target, small_param_grid, cv=2
        )

        # Test with 3 folds
        best_model_3, _ = tune_hyperparameters(
            sample_features, sample_target, small_param_grid, cv=3
        )

        assert isinstance(best_model_2, RandomForestClassifier)
        assert isinstance(best_model_3, RandomForestClassifier)

    def test_tune_single_parameter_grid(self, sample_features, sample_target, single_param_grid):
        """Test tuning with single parameter values."""
        best_model, best_params = tune_hyperparameters(
            sample_features, sample_target, single_param_grid, cv=2
        )

        # Should return those exact parameters
        assert best_params['n_estimators'] == 10
        assert best_params['max_depth'] == 5

    def test_tune_params_dict_structure(self, sample_features, sample_target, small_param_grid):
        """Test that best_params dictionary has correct structure."""
        best_model, best_params = tune_hyperparameters(
            sample_features, sample_target, small_param_grid, cv=2
        )

        # Should have all parameters from grid
        assert 'n_estimators' in best_params
        assert 'max_depth' in best_params

    def test_tune_with_larger_grid(self, sample_features, sample_target):
        """Test tuning with a larger parameter grid."""
        param_grid = {
            'n_estimators': [10, 20, 30],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5]
        }

        best_model, best_params = tune_hyperparameters(
            sample_features, sample_target, param_grid, cv=2
        )

        assert 'n_estimators' in best_params
        assert 'max_depth' in best_params
        assert 'min_samples_split' in best_params

    def test_tune_model_performance(self, sample_features, sample_target, small_param_grid):
        """Test that tuned model has reasonable performance."""
        best_model, best_params = tune_hyperparameters(
            sample_features, sample_target, small_param_grid, cv=2
        )

        # Model should achieve some level of accuracy
        predictions = best_model.predict(sample_features)
        accuracy = (predictions == sample_target.values).mean()

        # Should be better than random guessing
        assert accuracy > 0.5

    def test_tune_with_imbalanced_data(self, sample_features, imbalanced_target, small_param_grid):
        """Test tuning with imbalanced data."""
        best_model, best_params = tune_hyperparameters(
            sample_features, imbalanced_target, small_param_grid, scoring='recall', cv=2
        )

        assert isinstance(best_model, RandomForestClassifier)

    def test_tune_consistent_with_random_state(self, sample_features, sample_target):
        """Test that tuning is consistent when random_state is controlled."""
        param_grid = {
            'n_estimators': [10, 20],
            'max_depth': [3, 5],
            'random_state': [42]  # Fixed random state
        }

        _, best_params_1 = tune_hyperparameters(
            sample_features, sample_target, param_grid, cv=2
        )

        _, best_params_2 = tune_hyperparameters(
            sample_features, sample_target, param_grid, cv=2
        )

        # Results should be consistent
        assert best_params_1 == best_params_2

    def test_tune_with_min_samples_parameters(self, sample_features, sample_target):
        """Test tuning with min_samples_split and min_samples_leaf."""
        param_grid = {
            'n_estimators': [10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        best_model, best_params = tune_hyperparameters(
            sample_features, sample_target, param_grid, cv=2
        )

        assert 'min_samples_split' in best_params
        assert 'min_samples_leaf' in best_params

    def test_tune_preserves_data(self, sample_features, sample_target, small_param_grid):
        """Test that tuning doesn't modify input data."""
        X_original = sample_features.copy()
        y_original = sample_target.copy()

        tune_hyperparameters(sample_features, sample_target, small_param_grid, cv=2)

        # Data should be unchanged
        assert sample_features.equals(X_original)
        assert sample_target.equals(y_original)

    def test_tune_param_values_types(self, sample_features, sample_target, small_param_grid):
        """Test that best parameter values have correct types."""
        best_model, best_params = tune_hyperparameters(
            sample_features, sample_target, small_param_grid, cv=2
        )

        assert isinstance(best_params['n_estimators'], (int, np.integer))
        assert isinstance(best_params['max_depth'], (int, np.integer))

    def test_tune_empty_param_grid_works(self, sample_features, imbalanced_target):
        """Test that empty parameter grid uses default parameters."""
        param_grid = {}

        # Empty param_grid means GridSearchCV will use model defaults (1 candidate)
        # This should work but return the default model
        best_model, best_params = tune_hyperparameters(sample_features, imbalanced_target, param_grid, cv=2)

        # Should return a model (with defaults)
        assert isinstance(best_model, RandomForestClassifier)
        assert isinstance(best_params, dict)

    def test_tune_model_attributes(self, sample_features, sample_target, small_param_grid):
        """Test that tuned model has correct attributes set."""
        best_model, best_params = tune_hyperparameters(
            sample_features, sample_target, small_param_grid, cv=2
        )

        # Check that parameters were actually set
        assert best_model.n_estimators == best_params['n_estimators']
        assert best_model.max_depth == best_params['max_depth']
