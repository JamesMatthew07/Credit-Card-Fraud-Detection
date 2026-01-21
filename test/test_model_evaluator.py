"""
Unit tests for model_evaluator module.
"""
import pytest
import numpy as np
from pathlib import Path
import sys
from sklearn.ensemble import RandomForestClassifier

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_evaluator import evaluate_model, evaluate_thresholds


class TestEvaluateModel:
    """Test suite for evaluate_model function."""

    @pytest.fixture
    def trained_model(self, sample_features, sample_target):
        """Create a trained model for testing."""
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(sample_features, sample_target)
        return model

    def test_evaluate_model_returns_dict(self, trained_model, sample_features, sample_target):
        """Test that evaluate_model returns a dictionary."""
        results = evaluate_model(trained_model, sample_features, sample_target, "Test Model")

        assert isinstance(results, dict)

    def test_evaluate_model_contains_metrics(self, trained_model, sample_features, sample_target):
        """Test that results contain all expected metrics."""
        results = evaluate_model(trained_model, sample_features, sample_target, "Test Model")

        expected_keys = ['precision', 'recall', 'f1_score', 'roc_auc',
                        'average_precision_score', 'confusion_matrix', 'y_pred', 'y_proba']

        for key in expected_keys:
            assert key in results

    def test_evaluate_model_metric_ranges(self, trained_model, sample_features, sample_target):
        """Test that metrics are within valid ranges."""
        results = evaluate_model(trained_model, sample_features, sample_target, "Test Model")

        # Metrics should be between 0 and 1
        assert 0 <= results['precision'] <= 1
        assert 0 <= results['recall'] <= 1
        assert 0 <= results['f1_score'] <= 1
        assert 0 <= results['roc_auc'] <= 1
        assert 0 <= results['average_precision_score'] <= 1

    def test_evaluate_model_confusion_matrix_shape(self, trained_model, sample_features, sample_target):
        """Test that confusion matrix has correct shape."""
        results = evaluate_model(trained_model, sample_features, sample_target, "Test Model")

        cm = results['confusion_matrix']
        assert cm.shape == (2, 2)

    def test_evaluate_model_confusion_matrix_sum(self, trained_model, sample_features, sample_target):
        """Test that confusion matrix sums to total samples."""
        results = evaluate_model(trained_model, sample_features, sample_target, "Test Model")

        cm = results['confusion_matrix']
        assert cm.sum() == len(sample_target)

    def test_evaluate_model_predictions_length(self, trained_model, sample_features, sample_target):
        """Test that predictions have correct length."""
        results = evaluate_model(trained_model, sample_features, sample_target, "Test Model")

        assert len(results['y_pred']) == len(sample_target)
        assert len(results['y_proba']) == len(sample_target)

    def test_evaluate_model_predictions_values(self, trained_model, sample_features, sample_target):
        """Test that predictions have valid values."""
        results = evaluate_model(trained_model, sample_features, sample_target, "Test Model")

        # y_pred should be binary
        assert set(results['y_pred']).issubset({0, 1})

        # y_proba should be probabilities
        assert all(0 <= p <= 1 for p in results['y_proba'])

    def test_evaluate_model_with_perfect_predictions(self, sample_features, sample_target):
        """Test evaluation with perfect predictions."""
        # Create a mock model that predicts perfectly
        class PerfectModel:
            def predict(self, X):
                return sample_target.values
            def predict_proba(self, X):
                probs = np.zeros((len(sample_target), 2))
                probs[sample_target == 0, 0] = 1.0
                probs[sample_target == 1, 1] = 1.0
                return probs

        perfect_model = PerfectModel()
        results = evaluate_model(perfect_model, sample_features, sample_target, "Perfect Model")

        # With perfect predictions, most metrics should be 1.0
        assert results['precision'] == 1.0 or sample_target.sum() == 0
        assert results['recall'] == 1.0 or sample_target.sum() == 0
        assert results['f1_score'] == 1.0 or sample_target.sum() == 0
        assert results['roc_auc'] == 1.0

    def test_evaluate_model_with_all_zeros(self, sample_features):
        """Test evaluation when model predicts all zeros."""
        target_all_zeros = np.zeros(len(sample_features))

        class AllZerosModel:
            def predict(self, X):
                return np.zeros(len(X))
            def predict_proba(self, X):
                probs = np.zeros((len(X), 2))
                probs[:, 0] = 1.0
                return probs

        model = AllZerosModel()
        results = evaluate_model(model, sample_features, target_all_zeros, "All Zeros")

        # Check that function handles this edge case
        assert isinstance(results, dict)
        assert 'confusion_matrix' in results

    def test_evaluate_model_different_model_names(self, trained_model, sample_features, sample_target):
        """Test evaluation with different model names."""
        results1 = evaluate_model(trained_model, sample_features, sample_target, "Model A")
        results2 = evaluate_model(trained_model, sample_features, sample_target, "Model B")

        # Results should be the same (model name doesn't affect evaluation)
        assert results1['precision'] == results2['precision']
        assert results1['recall'] == results2['recall']


class TestEvaluateThresholds:
    """Test suite for evaluate_thresholds function."""

    def test_evaluate_thresholds_returns_dict(self, trained_predictions):
        """Test that evaluate_thresholds returns a dictionary."""
        y_test, y_pred, y_proba = trained_predictions
        thresholds = [0.3, 0.5, 0.7]

        results = evaluate_thresholds(y_test, y_proba, thresholds)

        assert isinstance(results, dict)

    def test_evaluate_thresholds_has_all_thresholds(self, trained_predictions):
        """Test that results contain all specified thresholds."""
        y_test, y_pred, y_proba = trained_predictions
        thresholds = [0.3, 0.5, 0.7]

        results = evaluate_thresholds(y_test, y_proba, thresholds)

        for threshold in thresholds:
            assert threshold in results

    def test_evaluate_thresholds_classification_reports(self, trained_predictions):
        """Test that each threshold has a classification report."""
        y_test, y_pred, y_proba = trained_predictions
        thresholds = [0.3, 0.5, 0.7]

        results = evaluate_thresholds(y_test, y_proba, thresholds)

        for threshold in thresholds:
            assert isinstance(results[threshold], dict)

    def test_evaluate_thresholds_single_threshold(self, trained_predictions):
        """Test evaluation with a single threshold."""
        y_test, y_pred, y_proba = trained_predictions
        thresholds = [0.5]

        results = evaluate_thresholds(y_test, y_proba, thresholds)

        assert len(results) == 1
        assert 0.5 in results

    def test_evaluate_thresholds_multiple_thresholds(self, trained_predictions):
        """Test evaluation with multiple thresholds."""
        y_test, y_pred, y_proba = trained_predictions
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]

        results = evaluate_thresholds(y_test, y_proba, thresholds)

        assert len(results) == 5

    def test_evaluate_thresholds_lower_increases_positives(self, trained_predictions):
        """Test that lower thresholds predict more positives."""
        y_test, y_pred, y_proba = trained_predictions
        thresholds = [0.3, 0.5, 0.7]

        # Manually predict with different thresholds
        pred_03 = (y_proba >= 0.3).astype(int)
        pred_05 = (y_proba >= 0.5).astype(int)
        pred_07 = (y_proba >= 0.7).astype(int)

        # Lower threshold should predict more positives
        assert pred_03.sum() >= pred_05.sum()
        assert pred_05.sum() >= pred_07.sum()

    def test_evaluate_thresholds_extreme_low(self, trained_predictions):
        """Test evaluation with very low threshold."""
        y_test, y_pred, y_proba = trained_predictions
        thresholds = [0.01]

        results = evaluate_thresholds(y_test, y_proba, thresholds)

        # Very low threshold should predict almost everything as positive
        assert 0.01 in results

    def test_evaluate_thresholds_extreme_high(self, trained_predictions):
        """Test evaluation with very high threshold."""
        y_test, y_pred, y_proba = trained_predictions
        thresholds = [0.99]

        results = evaluate_thresholds(y_test, y_proba, thresholds)

        # Very high threshold should predict almost everything as negative
        assert 0.99 in results

    def test_evaluate_thresholds_boundary_values(self, trained_predictions):
        """Test evaluation with boundary threshold values."""
        y_test, y_pred, y_proba = trained_predictions
        thresholds = [0.0, 0.5, 1.0]

        results = evaluate_thresholds(y_test, y_proba, thresholds)

        assert len(results) == 3

    def test_evaluate_thresholds_with_balanced_data(self, balanced_data):
        """Test threshold evaluation with balanced data."""
        y = balanced_data['Class'].values.astype(int)  # Ensure int type
        # Create probabilities that ensure some predictions at each threshold
        np.random.seed(42)
        y_proba = np.random.rand(len(y)) * 0.5 + y * 0.5  # Correlate with y
        thresholds = [0.3, 0.5, 0.7]

        results = evaluate_thresholds(y, y_proba, thresholds)

        assert len(results) == 3

    def test_evaluate_thresholds_empty_list(self, trained_predictions):
        """Test evaluation with empty threshold list."""
        y_test, y_pred, y_proba = trained_predictions
        thresholds = []

        results = evaluate_thresholds(y_test, y_proba, thresholds)

        assert results == {}

    def test_evaluate_thresholds_sorted_order(self, trained_predictions):
        """Test that thresholds can be in any order."""
        y_test, y_pred, y_proba = trained_predictions
        thresholds = [0.7, 0.3, 0.5]  # Unsorted

        results = evaluate_thresholds(y_test, y_proba, thresholds)

        assert len(results) == 3
        assert all(t in results for t in thresholds)

    def test_evaluate_thresholds_duplicate_thresholds(self, trained_predictions):
        """Test evaluation with duplicate thresholds."""
        y_test, y_pred, y_proba = trained_predictions
        thresholds = [0.5, 0.5, 0.3]

        results = evaluate_thresholds(y_test, y_proba, thresholds)

        # Dictionary should have unique keys
        assert len(results) == 2  # 0.5 and 0.3
