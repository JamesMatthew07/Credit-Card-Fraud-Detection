"""
Shared pytest fixtures for credit card fraud detection tests.
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import yaml
from pathlib import Path


@pytest.fixture
def sample_data():
    """Create sample credit card transaction data for testing."""
    np.random.seed(42)
    n_samples = 1000

    # Create feature columns V1-V28
    data = {f'V{i}': np.random.randn(n_samples) for i in range(1, 29)}

    # Add Time and Amount columns
    data['Time'] = np.random.randint(0, 172800, n_samples)
    data['Amount'] = np.random.exponential(scale=50, size=n_samples)

    # Add Class column (imbalanced: ~2% fraud)
    data['Class'] = np.random.choice([0, 1], size=n_samples, p=[0.98, 0.02])

    return pd.DataFrame(data)


@pytest.fixture
def balanced_data():
    """Create balanced dataset for testing."""
    np.random.seed(42)
    n_samples = 500

    data = {f'V{i}': np.random.randn(n_samples) for i in range(1, 29)}
    data['Time'] = np.random.randint(0, 172800, n_samples)
    data['Amount'] = np.random.exponential(scale=50, size=n_samples)
    data['Class'] = np.concatenate([np.zeros(250), np.ones(250)])

    return pd.DataFrame(data)


@pytest.fixture
def sample_features():
    """Create sample feature DataFrame without target column."""
    np.random.seed(42)
    n_samples = 100

    data = {f'V{i}': np.random.randn(n_samples) for i in range(1, 29)}
    data['Time'] = np.random.randint(0, 172800, n_samples)
    data['Amount'] = np.random.exponential(scale=50, size=n_samples)

    return pd.DataFrame(data)


@pytest.fixture
def sample_target():
    """Create sample target Series."""
    np.random.seed(42)
    return pd.Series(np.random.choice([0, 1], size=100, p=[0.98, 0.02]), name='Class')


@pytest.fixture
def imbalanced_target():
    """Create imbalanced target with enough fraud cases for SMOTE (needs at least 6)."""
    return pd.Series([0] * 90 + [1] * 10, name='Class')


@pytest.fixture
def temp_csv_file(sample_data, tmp_path):
    """Create a temporary CSV file with sample data."""
    filepath = tmp_path / "test_data.csv"
    sample_data.to_csv(filepath, index=False)
    return str(filepath)


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary config YAML file."""
    config = {
        'data': {
            'filepath': 'dataset/creditcard.csv',
            'test_size': 0.2,
            'random_state': 42
        },
        'models': {
            'logistic_regression': {
                'enabled': True,
                'max_iter': 1000,
                'random_state': 42
            },
            'random_forest': {
                'enabled': True,
                'n_estimators': 10,
                'random_state': 42
            },
            'xgboost': {
                'enabled': True,
                'n_estimators': 10,
                'learning_rate': 0.1,
                'random_state': 42
            }
        },
        'evaluation': {
            'metrics': ['precision', 'recall', 'f1_score', 'roc_auc']
        },
        'preprocessing': {
            'stratify': True,
            'scaling': True,
            'oversampling': True,
            'amount_threshold': 200
        },
        'visualization': {
            'show_plots': False,
            'figure_width': 12,
            'figure_height': 8,
            'output_dir': 'plots/results'
        }
    }

    filepath = tmp_path / "test_config.yaml"
    with open(filepath, 'w') as f:
        yaml.dump(config, f)

    return str(filepath)


@pytest.fixture
def default_config():
    """Return default configuration dictionary."""
    return {
        'data': {
            'filepath': 'dataset/creditcard.csv',
            'test_size': 0.2,
            'random_state': 42
        },
        'models': {
            'logistic_regression': {
                'enabled': True,
                'max_iter': 1000,
                'random_state': 42
            },
            'random_forest': {
                'enabled': True,
                'n_estimators': 100,
                'random_state': 42
            },
            'xgboost': {
                'enabled': True,
                'n_estimators': 100,
                'learning_rate': 0.1,
                'random_state': 42
            }
        },
        'evaluation': {
            'metrics': ['precision', 'recall', 'f1_score', 'roc_auc']
        },
        'preprocessing': {
            'stratify': True,
            'scaling': True,
            'oversampling': True,
            'amount_threshold': 200
        },
        'visualization': {
            'show_plots': False,
            'figure_width': 12,
            'figure_height': 8,
            'output_dir': 'plots/results'
        }
    }


@pytest.fixture
def trained_predictions():
    """Create sample predictions and probabilities for testing."""
    np.random.seed(42)
    y_test = np.array([0] * 80 + [1] * 20)
    y_pred = y_test.copy()
    # Add some errors
    y_pred[5] = 1  # False positive
    y_pred[85] = 0  # False negative

    y_proba = np.random.rand(100)
    y_proba[y_test == 1] += 0.5  # Make fraud cases have higher probabilities
    y_proba = np.clip(y_proba, 0, 1)

    return y_test, y_pred, y_proba


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory for plots."""
    output_dir = tmp_path / "test_plots"
    output_dir.mkdir(exist_ok=True)
    return str(output_dir)
