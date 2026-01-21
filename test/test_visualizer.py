"""
Unit tests for visualizer module.
"""
import pytest
import numpy as np
from pathlib import Path
import sys
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualizer import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
    plot_all_metrics
)


class TestPlotConfusionMatrix:
    """Test suite for plot_confusion_matrix function."""

    @pytest.fixture
    def sample_confusion_matrix(self):
        """Create a sample confusion matrix."""
        return np.array([[80, 5], [10, 5]])

    def test_plot_creates_file(self, sample_confusion_matrix, temp_output_dir):
        """Test that confusion matrix plot creates a file."""
        plot_confusion_matrix(sample_confusion_matrix, "Test Model", temp_output_dir)

        expected_file = os.path.join(temp_output_dir, "confusion_matrix_test_model.png")
        assert os.path.exists(expected_file)

    def test_plot_file_naming(self, sample_confusion_matrix, temp_output_dir):
        """Test that plot file follows naming convention."""
        model_name = "Random Forest"
        plot_confusion_matrix(sample_confusion_matrix, model_name, temp_output_dir)

        expected_file = os.path.join(temp_output_dir, "confusion_matrix_random_forest.png")
        assert os.path.exists(expected_file)

    def test_plot_creates_directory(self, sample_confusion_matrix, tmp_path):
        """Test that plot creates output directory if it doesn't exist."""
        new_dir = os.path.join(tmp_path, "new_plots")
        plot_confusion_matrix(sample_confusion_matrix, "Test Model", new_dir)

        assert os.path.exists(new_dir)

    def test_plot_with_different_models(self, sample_confusion_matrix, temp_output_dir):
        """Test creating plots for different models."""
        models = ["Logistic Regression", "Random Forest", "XGBoost"]

        for model_name in models:
            plot_confusion_matrix(sample_confusion_matrix, model_name, temp_output_dir)

        for model_name in models:
            filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
            expected_file = os.path.join(temp_output_dir, filename)
            assert os.path.exists(expected_file)

    def test_plot_overwrites_existing(self, sample_confusion_matrix, temp_output_dir):
        """Test that plotting overwrites existing files."""
        # Create plot twice
        plot_confusion_matrix(sample_confusion_matrix, "Test Model", temp_output_dir)
        plot_confusion_matrix(sample_confusion_matrix, "Test Model", temp_output_dir)

        expected_file = os.path.join(temp_output_dir, "confusion_matrix_test_model.png")
        assert os.path.exists(expected_file)

    def test_plot_with_zeros(self, temp_output_dir):
        """Test plotting confusion matrix with zeros."""
        cm_with_zeros = np.array([[100, 0], [0, 0]])
        plot_confusion_matrix(cm_with_zeros, "Zero Model", temp_output_dir)

        expected_file = os.path.join(temp_output_dir, "confusion_matrix_zero_model.png")
        assert os.path.exists(expected_file)

    def test_plot_file_is_not_empty(self, sample_confusion_matrix, temp_output_dir):
        """Test that created plot file is not empty."""
        plot_confusion_matrix(sample_confusion_matrix, "Test Model", temp_output_dir)

        expected_file = os.path.join(temp_output_dir, "confusion_matrix_test_model.png")
        assert os.path.getsize(expected_file) > 0


class TestPlotRocCurve:
    """Test suite for plot_roc_curve function."""

    @pytest.fixture
    def sample_roc_data(self):
        """Create sample data for ROC curve."""
        y_test = np.array([0] * 80 + [1] * 20)
        y_proba = np.random.rand(100)
        y_proba[y_test == 1] += 0.5  # Make fraud cases have higher probabilities
        y_proba = np.clip(y_proba, 0, 1)
        return y_test, y_proba

    def test_roc_creates_file(self, sample_roc_data, temp_output_dir):
        """Test that ROC curve plot creates a file."""
        y_test, y_proba = sample_roc_data
        plot_roc_curve(y_test, y_proba, "Test Model", temp_output_dir)

        expected_file = os.path.join(temp_output_dir, "roc_curve_test_model.png")
        assert os.path.exists(expected_file)

    def test_roc_file_naming(self, sample_roc_data, temp_output_dir):
        """Test that ROC curve file follows naming convention."""
        y_test, y_proba = sample_roc_data
        model_name = "Random Forest"
        plot_roc_curve(y_test, y_proba, model_name, temp_output_dir)

        expected_file = os.path.join(temp_output_dir, "roc_curve_random_forest.png")
        assert os.path.exists(expected_file)

    def test_roc_creates_directory(self, sample_roc_data, tmp_path):
        """Test that ROC plot creates output directory if it doesn't exist."""
        y_test, y_proba = sample_roc_data
        new_dir = os.path.join(tmp_path, "new_roc_plots")
        plot_roc_curve(y_test, y_proba, "Test Model", new_dir)

        assert os.path.exists(new_dir)

    def test_roc_with_perfect_predictions(self, temp_output_dir):
        """Test ROC curve with perfect predictions."""
        y_test = np.array([0] * 50 + [1] * 50)
        y_proba = y_test.astype(float)  # Perfect predictions

        plot_roc_curve(y_test, y_proba, "Perfect Model", temp_output_dir)

        expected_file = os.path.join(temp_output_dir, "roc_curve_perfect_model.png")
        assert os.path.exists(expected_file)

    def test_roc_file_is_not_empty(self, sample_roc_data, temp_output_dir):
        """Test that created ROC plot file is not empty."""
        y_test, y_proba = sample_roc_data
        plot_roc_curve(y_test, y_proba, "Test Model", temp_output_dir)

        expected_file = os.path.join(temp_output_dir, "roc_curve_test_model.png")
        assert os.path.getsize(expected_file) > 0


class TestPlotPrCurve:
    """Test suite for plot_pr_curve function."""

    @pytest.fixture
    def sample_pr_data(self):
        """Create sample data for PR curve."""
        y_test = np.array([0] * 80 + [1] * 20)
        y_proba = np.random.rand(100)
        y_proba[y_test == 1] += 0.5
        y_proba = np.clip(y_proba, 0, 1)
        return y_test, y_proba

    def test_pr_creates_file(self, sample_pr_data, temp_output_dir):
        """Test that PR curve plot creates a file."""
        y_test, y_proba = sample_pr_data
        plot_pr_curve(y_test, y_proba, "Test Model", temp_output_dir)

        expected_file = os.path.join(temp_output_dir, "pr_curve_test_model.png")
        assert os.path.exists(expected_file)

    def test_pr_file_naming(self, sample_pr_data, temp_output_dir):
        """Test that PR curve file follows naming convention."""
        y_test, y_proba = sample_pr_data
        model_name = "XGBoost"
        plot_pr_curve(y_test, y_proba, model_name, temp_output_dir)

        expected_file = os.path.join(temp_output_dir, "pr_curve_xgboost.png")
        assert os.path.exists(expected_file)

    def test_pr_creates_directory(self, sample_pr_data, tmp_path):
        """Test that PR plot creates output directory if it doesn't exist."""
        y_test, y_proba = sample_pr_data
        new_dir = os.path.join(tmp_path, "new_pr_plots")
        plot_pr_curve(y_test, y_proba, "Test Model", new_dir)

        assert os.path.exists(new_dir)

    def test_pr_with_perfect_predictions(self, temp_output_dir):
        """Test PR curve with perfect predictions."""
        y_test = np.array([0] * 50 + [1] * 50)
        y_proba = y_test.astype(float)

        plot_pr_curve(y_test, y_proba, "Perfect Model", temp_output_dir)

        expected_file = os.path.join(temp_output_dir, "pr_curve_perfect_model.png")
        assert os.path.exists(expected_file)

    def test_pr_file_is_not_empty(self, sample_pr_data, temp_output_dir):
        """Test that created PR plot file is not empty."""
        y_test, y_proba = sample_pr_data
        plot_pr_curve(y_test, y_proba, "Test Model", temp_output_dir)

        expected_file = os.path.join(temp_output_dir, "pr_curve_test_model.png")
        assert os.path.getsize(expected_file) > 0


class TestPlotAllMetrics:
    """Test suite for plot_all_metrics function."""

    @pytest.fixture
    def sample_results(self):
        """Create sample evaluation results."""
        y_test = np.array([0] * 80 + [1] * 20)
        y_proba = np.random.rand(100)
        y_proba[y_test == 1] += 0.5
        y_proba = np.clip(y_proba, 0, 1)

        cm = np.array([[75, 5], [10, 10]])

        return {
            'precision': 0.67,
            'recall': 0.50,
            'f1_score': 0.57,
            'roc_auc': 0.85,
            'confusion_matrix': cm,
            'y_proba': y_proba
        }, y_test

    def test_all_metrics_creates_all_files(self, sample_results, temp_output_dir):
        """Test that plot_all_metrics creates all three plot files."""
        results, y_test = sample_results
        plot_all_metrics(results, y_test, "Test Model", temp_output_dir)

        # Check all three files exist
        cm_file = os.path.join(temp_output_dir, "confusion_matrix_test_model.png")
        roc_file = os.path.join(temp_output_dir, "roc_curve_test_model.png")
        pr_file = os.path.join(temp_output_dir, "pr_curve_test_model.png")

        assert os.path.exists(cm_file)
        assert os.path.exists(roc_file)
        assert os.path.exists(pr_file)

    def test_all_metrics_with_different_model_names(self, sample_results, temp_output_dir):
        """Test plotting all metrics for different model names."""
        results, y_test = sample_results
        models = ["Logistic Regression", "Random Forest"]

        for model_name in models:
            plot_all_metrics(results, y_test, model_name, temp_output_dir)

        # Check files for both models
        for model_name in models:
            name_lower = model_name.lower().replace(' ', '_')
            cm_file = os.path.join(temp_output_dir, f"confusion_matrix_{name_lower}.png")
            roc_file = os.path.join(temp_output_dir, f"roc_curve_{name_lower}.png")
            pr_file = os.path.join(temp_output_dir, f"pr_curve_{name_lower}.png")

            assert os.path.exists(cm_file)
            assert os.path.exists(roc_file)
            assert os.path.exists(pr_file)

    def test_all_metrics_creates_directory(self, sample_results, tmp_path):
        """Test that plot_all_metrics creates output directory."""
        results, y_test = sample_results
        new_dir = os.path.join(tmp_path, "all_metrics_plots")
        plot_all_metrics(results, y_test, "Test Model", new_dir)

        assert os.path.exists(new_dir)

    def test_all_metrics_files_not_empty(self, sample_results, temp_output_dir):
        """Test that all created plot files are not empty."""
        results, y_test = sample_results
        plot_all_metrics(results, y_test, "Test Model", temp_output_dir)

        cm_file = os.path.join(temp_output_dir, "confusion_matrix_test_model.png")
        roc_file = os.path.join(temp_output_dir, "roc_curve_test_model.png")
        pr_file = os.path.join(temp_output_dir, "pr_curve_test_model.png")

        assert os.path.getsize(cm_file) > 0
        assert os.path.getsize(roc_file) > 0
        assert os.path.getsize(pr_file) > 0

    def test_all_metrics_handles_missing_keys_gracefully(self, temp_output_dir):
        """Test that plot_all_metrics handles incomplete results."""
        y_test = np.array([0] * 80 + [1] * 20)
        y_proba = np.random.rand(100)
        cm = np.array([[75, 5], [10, 10]])

        # Minimal results
        results = {
            'confusion_matrix': cm,
            'y_proba': y_proba
        }

        # Should not raise an error
        try:
            plot_all_metrics(results, y_test, "Minimal Model", temp_output_dir)
        except KeyError:
            pytest.fail("plot_all_metrics should handle missing keys")
