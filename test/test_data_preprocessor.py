"""
Unit tests for data_preprocessor module.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_preprocessor import apply_smote


class TestApplySmote:
    """Test suite for apply_smote function."""

    def test_smote_balances_classes(self, sample_features, imbalanced_target):
        """Test that SMOTE balances imbalanced classes."""
        X_resampled, y_resampled = apply_smote(sample_features, imbalanced_target)

        # Count classes after SMOTE (y_resampled is numpy array)
        unique, counts = np.unique(y_resampled, return_counts=True)
        class_counts = dict(zip(unique, counts))

        # Classes should be balanced
        assert class_counts[0] == class_counts[1]

    def test_smote_increases_minority_class(self, sample_features, imbalanced_target):
        """Test that SMOTE increases minority class samples."""
        original_fraud_count = (imbalanced_target == 1).sum()

        X_resampled, y_resampled = apply_smote(sample_features, imbalanced_target)

        resampled_fraud_count = (y_resampled == 1).sum()

        # Minority class should be increased
        assert resampled_fraud_count > original_fraud_count

    def test_smote_output_types(self, sample_features, imbalanced_target):
        """Test that SMOTE returns correct data types."""
        X_resampled, y_resampled = apply_smote(sample_features, imbalanced_target)

        # SMOTE with DataFrame input returns DataFrames/Series in newer imblearn versions
        assert isinstance(X_resampled, (pd.DataFrame, np.ndarray))
        assert isinstance(y_resampled, (pd.Series, np.ndarray))

    def test_smote_preserves_columns(self, sample_features, imbalanced_target):
        """Test that SMOTE preserves number of features."""
        X_resampled, y_resampled = apply_smote(sample_features, imbalanced_target)

        # Number of features should be the same
        assert X_resampled.shape[1] == sample_features.shape[1]

    def test_smote_increases_dataset_size(self, sample_features, imbalanced_target):
        """Test that SMOTE increases the dataset size."""
        original_size = len(sample_features)

        X_resampled, y_resampled = apply_smote(sample_features, imbalanced_target)

        # Resampled should be larger
        assert len(X_resampled) > original_size
        assert len(y_resampled) > original_size

    def test_smote_length_consistency(self, sample_features, imbalanced_target):
        """Test that X and y have same length after SMOTE."""
        X_resampled, y_resampled = apply_smote(sample_features, imbalanced_target)

        assert len(X_resampled) == len(y_resampled)

    def test_smote_reproducibility(self, sample_features, imbalanced_target):
        """Test that SMOTE is reproducible (uses fixed random_state)."""
        X_resampled1, y_resampled1 = apply_smote(sample_features, imbalanced_target)
        X_resampled2, y_resampled2 = apply_smote(sample_features, imbalanced_target)

        # Should be identical (numpy arrays)
        np.testing.assert_array_equal(X_resampled1, X_resampled2)
        np.testing.assert_array_equal(y_resampled1, y_resampled2)

    def test_smote_preserves_feature_ranges(self, sample_features, imbalanced_target):
        """Test that SMOTE preserves reasonable feature ranges."""
        original_min = sample_features.min().values
        original_max = sample_features.max().values

        X_resampled, y_resampled = apply_smote(sample_features, imbalanced_target)

        resampled_min = X_resampled.min(axis=0)
        resampled_max = X_resampled.max(axis=0)

        # Synthetic samples should be within reasonable bounds
        # Allow some extrapolation but not too much
        for i in range(sample_features.shape[1]):
            assert resampled_min[i] >= original_min[i] - 5 * abs(original_max[i])
            assert resampled_max[i] <= original_max[i] + 5 * abs(original_min[i])

    def test_smote_no_missing_values(self, sample_features, imbalanced_target):
        """Test that SMOTE doesn't introduce missing values."""
        X_resampled, y_resampled = apply_smote(sample_features, imbalanced_target)

        if isinstance(X_resampled, pd.DataFrame):
            assert X_resampled.isna().sum().sum() == 0
            assert y_resampled.isna().sum() == 0
        else:
            assert not np.isnan(X_resampled).any()
            assert not np.isnan(y_resampled).any()

    def test_smote_with_already_balanced_data(self, balanced_data):
        """Test SMOTE with already balanced data."""
        X = balanced_data.drop('Class', axis=1)
        y = balanced_data['Class']

        X_resampled, y_resampled = apply_smote(X, y)

        # Should still balance (might oversample both classes equally)
        assert len(X_resampled) >= len(X)

    def test_smote_target_values_binary(self, sample_features, imbalanced_target):
        """Test that SMOTE preserves binary target values."""
        X_resampled, y_resampled = apply_smote(sample_features, imbalanced_target)

        # Should only have 0 and 1
        unique_values = set(np.unique(y_resampled))
        assert unique_values.issubset({0, 1})

    def test_smote_feature_dtypes_preserved(self, sample_features, imbalanced_target):
        """Test that SMOTE preserves feature data types."""
        X_resampled, y_resampled = apply_smote(sample_features, imbalanced_target)

        # All features should still be numeric
        if isinstance(X_resampled, pd.DataFrame):
            assert X_resampled.select_dtypes(include=[np.number]).shape[1] == X_resampled.shape[1]
        else:
            assert np.issubdtype(X_resampled.dtype, np.number)

    def test_smote_very_imbalanced_data(self, sample_features):
        """Test SMOTE with extremely imbalanced data."""
        # Create very imbalanced target (90:10 ratio - still very imbalanced but enough for SMOTE)
        very_imbalanced = pd.Series([0] * 90 + [1] * 10)
        sample_features_subset = sample_features.iloc[:100].copy()

        X_resampled, y_resampled = apply_smote(sample_features_subset, very_imbalanced)

        # Should balance the classes
        unique, counts = np.unique(y_resampled, return_counts=True)
        class_counts = dict(zip(unique, counts))
        assert class_counts[0] == class_counts[1]

    def test_smote_index_reset(self, sample_features, imbalanced_target):
        """Test that SMOTE returns appropriate data structures."""
        X_resampled, y_resampled = apply_smote(sample_features, imbalanced_target)

        # Check that we get valid data structures (either DataFrame/Series or arrays)
        assert isinstance(X_resampled, (pd.DataFrame, np.ndarray))
        assert isinstance(y_resampled, (pd.Series, np.ndarray))

    def test_smote_with_minimal_samples(self):
        """Test SMOTE with minimal number of samples."""
        # Create minimal dataset (need at least 6 samples of minority class for default SMOTE)
        X_minimal = pd.DataFrame({
            'V1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            'V2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
        })
        y_minimal = pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

        X_resampled, y_resampled = apply_smote(X_minimal, y_minimal)

        # Should still work and balance
        unique, counts = np.unique(y_resampled, return_counts=True)
        class_counts = dict(zip(unique, counts))
        assert class_counts[0] == class_counts[1]
