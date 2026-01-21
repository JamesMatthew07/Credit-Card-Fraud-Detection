"""
Unit tests for data_loader module.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_data, preprocess_data, split_data


class TestLoadData:
    """Test suite for load_data function."""

    def test_load_valid_csv(self, temp_csv_file):
        """Test loading a valid CSV file."""
        X, y = load_data(temp_csv_file)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert 'Class' not in X.columns
        assert y.name == 'Class'

    def test_load_data_shape(self, temp_csv_file):
        """Test that loaded data has correct shape."""
        X, y = load_data(temp_csv_file)

        # Should have 30 features (V1-V28, Time, Amount)
        assert X.shape[1] == 30
        assert X.shape[0] == 1000
        assert len(y) == 1000

    def test_load_data_columns(self, temp_csv_file):
        """Test that loaded data has correct columns."""
        X, y = load_data(temp_csv_file)

        # Check for V1-V28
        for i in range(1, 29):
            assert f'V{i}' in X.columns

        # Check for Time and Amount
        assert 'Time' in X.columns
        assert 'Amount' in X.columns

    def test_load_nonexistent_file(self):
        """Test loading a non-existent file raises SystemExit."""
        with pytest.raises(SystemExit):
            load_data('nonexistent_file.csv')

    def test_load_empty_csv(self, tmp_path):
        """Test loading an empty CSV file raises SystemExit."""
        empty_file = tmp_path / "empty.csv"
        empty_file.touch()

        with pytest.raises(SystemExit):
            load_data(str(empty_file))

    def test_load_csv_without_class_column(self, tmp_path):
        """Test loading CSV without 'Class' column raises SystemExit."""
        data = pd.DataFrame({
            'V1': [1, 2, 3],
            'Amount': [10, 20, 30]
        })
        no_class_file = tmp_path / "no_class.csv"
        data.to_csv(no_class_file, index=False)

        with pytest.raises(SystemExit):
            load_data(str(no_class_file))

    def test_load_data_class_distribution(self, temp_csv_file):
        """Test that class distribution is preserved."""
        X, y = load_data(temp_csv_file)

        # Check that we have both classes
        assert 0 in y.values
        assert 1 in y.values

        # Check class counts
        class_counts = y.value_counts()
        assert len(class_counts) == 2

    def test_load_data_dtypes(self, temp_csv_file):
        """Test that loaded data has correct data types."""
        X, y = load_data(temp_csv_file)

        # Features should be numeric
        assert X.select_dtypes(include=[np.number]).shape[1] == X.shape[1]

        # Target should be integer-like
        assert y.dtype in [np.int64, np.int32, int]

    def test_load_data_no_missing_values(self, temp_csv_file):
        """Test that loaded data has no missing values."""
        X, y = load_data(temp_csv_file)

        assert X.isna().sum().sum() == 0
        assert y.isna().sum() == 0


class TestPreprocessData:
    """Test suite for preprocess_data function."""

    def test_preprocess_scales_amount(self, sample_features):
        """Test that Amount column is scaled."""
        X_scaled = preprocess_data(sample_features)

        # Scaled Amount should have mean ~0 and std ~1
        assert abs(X_scaled['Amount'].mean()) < 0.1
        assert abs(X_scaled['Amount'].std() - 1.0) < 0.1

    def test_preprocess_scales_time(self, sample_features):
        """Test that Time column is scaled."""
        X_scaled = preprocess_data(sample_features)

        # Scaled Time should have mean ~0 and std ~1
        assert abs(X_scaled['Time'].mean()) < 0.1
        assert abs(X_scaled['Time'].std() - 1.0) < 0.1

    def test_preprocess_preserves_other_columns(self, sample_features):
        """Test that other columns remain unchanged."""
        original_v1 = sample_features['V1'].copy()
        X_scaled = preprocess_data(sample_features)

        # V1 should be unchanged
        pd.testing.assert_series_equal(X_scaled['V1'], original_v1)

    def test_preprocess_output_shape(self, sample_features):
        """Test that output shape matches input shape."""
        X_scaled = preprocess_data(sample_features)

        assert X_scaled.shape == sample_features.shape

    def test_preprocess_output_columns(self, sample_features):
        """Test that output has same columns as input."""
        X_scaled = preprocess_data(sample_features)

        assert list(X_scaled.columns) == list(sample_features.columns)

    def test_preprocess_does_not_modify_original(self, sample_features):
        """Test that preprocessing does not modify the original DataFrame."""
        original_amount = sample_features['Amount'].copy()
        X_scaled = preprocess_data(sample_features)

        # Original should be unchanged
        pd.testing.assert_series_equal(sample_features['Amount'], original_amount)

    def test_preprocess_with_missing_columns(self):
        """Test preprocessing with missing Amount or Time columns."""
        # Create data without Amount
        data = pd.DataFrame({
            'V1': [1, 2, 3],
            'Time': [100, 200, 300]
        })

        with pytest.raises(KeyError):
            preprocess_data(data)

    def test_preprocess_consistent_scaling(self, sample_features):
        """Test that preprocessing is consistent across calls."""
        X_scaled1 = preprocess_data(sample_features)
        X_scaled2 = preprocess_data(sample_features)

        pd.testing.assert_frame_equal(X_scaled1, X_scaled2)


class TestSplitData:
    """Test suite for split_data function."""

    def test_split_default_ratio(self, sample_features, sample_target):
        """Test splitting with default 80/20 ratio."""
        # Use stratify=False to avoid issues with very imbalanced small dataset
        X_train, X_test, y_train, y_test = split_data(
            sample_features, sample_target, stratify=False
        )

        # Check sizes (80/20 split of 100 samples)
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20

    def test_split_custom_ratio(self, sample_features, sample_target):
        """Test splitting with custom ratio."""
        X_train, X_test, y_train, y_test = split_data(
            sample_features, sample_target, test_size=0.3, stratify=False
        )

        # Check sizes (70/30 split of 100 samples)
        assert len(X_train) == 70
        assert len(X_test) == 30

    def test_split_reproducibility(self, sample_features, sample_target):
        """Test that split is reproducible with same random_state."""
        X_train1, X_test1, y_train1, y_test1 = split_data(
            sample_features, sample_target, random_state=42, stratify=False
        )

        X_train2, X_test2, y_train2, y_test2 = split_data(
            sample_features, sample_target, random_state=42, stratify=False
        )

        # Should be identical
        pd.testing.assert_frame_equal(X_train1, X_train2)
        pd.testing.assert_frame_equal(X_test1, X_test2)
        pd.testing.assert_series_equal(y_train1, y_train2)
        pd.testing.assert_series_equal(y_test1, y_test2)

    def test_split_stratified(self, sample_features, imbalanced_target):
        """Test that stratified split maintains class distribution."""
        X_train, X_test, y_train, y_test = split_data(
            sample_features, imbalanced_target, stratify=True, random_state=42
        )

        # Calculate fraud percentage
        train_fraud_pct = (y_train == 1).sum() / len(y_train)
        test_fraud_pct = (y_test == 1).sum() / len(y_test)
        original_fraud_pct = (imbalanced_target == 1).sum() / len(imbalanced_target)

        # Percentages should be similar (within reasonable tolerance)
        assert abs(train_fraud_pct - original_fraud_pct) < 0.1
        assert abs(test_fraud_pct - original_fraud_pct) < 0.2

    def test_split_non_stratified(self, sample_features, sample_target):
        """Test non-stratified split."""
        X_train, X_test, y_train, y_test = split_data(
            sample_features, sample_target, stratify=False, random_state=42
        )

        # Should still split correctly
        assert len(X_train) == 80
        assert len(X_test) == 20

    def test_split_return_types(self, sample_features, sample_target):
        """Test that split returns correct types."""
        X_train, X_test, y_train, y_test = split_data(
            sample_features, sample_target, stratify=False
        )

        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)

    def test_split_no_overlap(self, sample_features, sample_target):
        """Test that train and test sets don't overlap."""
        X_train, X_test, y_train, y_test = split_data(
            sample_features, sample_target, random_state=42, stratify=False
        )

        # Indices should not overlap
        train_indices = set(X_train.index)
        test_indices = set(X_test.index)

        assert len(train_indices.intersection(test_indices)) == 0

    def test_split_covers_all_data(self, sample_features, sample_target):
        """Test that train and test sets cover all original data."""
        X_train, X_test, y_train, y_test = split_data(
            sample_features, sample_target, stratify=False
        )

        # Combined should equal original size
        assert len(X_train) + len(X_test) == len(sample_features)
        assert len(y_train) + len(y_test) == len(sample_target)

    def test_split_preserves_columns(self, sample_features, sample_target):
        """Test that split preserves all columns."""
        X_train, X_test, y_train, y_test = split_data(
            sample_features, sample_target, stratify=False
        )

        assert list(X_train.columns) == list(sample_features.columns)
        assert list(X_test.columns) == list(sample_features.columns)

    def test_split_with_different_random_states(self, sample_features, sample_target):
        """Test that different random states produce different splits."""
        X_train1, X_test1, _, _ = split_data(
            sample_features, sample_target, random_state=42, stratify=False
        )

        X_train2, X_test2, _, _ = split_data(
            sample_features, sample_target, random_state=99, stratify=False
        )

        # Should be different
        assert not X_train1.equals(X_train2)
