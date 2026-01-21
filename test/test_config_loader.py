"""
Unit tests for config_loader module.
"""
import pytest
import yaml
import os
from pathlib import Path
import sys

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config


class TestLoadConfig:
    """Test suite for load_config function."""

    def test_load_valid_config(self, temp_config_file):
        """Test loading a valid YAML configuration file."""
        config = load_config(temp_config_file)

        assert isinstance(config, dict)
        assert 'data' in config
        assert 'models' in config
        assert 'evaluation' in config
        assert 'preprocessing' in config
        assert 'visualization' in config

    def test_config_data_section(self, temp_config_file):
        """Test data section of configuration."""
        config = load_config(temp_config_file)

        assert config['data']['filepath'] == 'dataset/creditcard.csv'
        assert config['data']['test_size'] == 0.2
        assert config['data']['random_state'] == 42

    def test_config_models_section(self, temp_config_file):
        """Test models section of configuration."""
        config = load_config(temp_config_file)

        # Check logistic regression
        assert 'logistic_regression' in config['models']
        assert config['models']['logistic_regression']['enabled'] is True
        assert config['models']['logistic_regression']['max_iter'] == 1000

        # Check random forest
        assert 'random_forest' in config['models']
        assert config['models']['random_forest']['enabled'] is True
        assert config['models']['random_forest']['n_estimators'] == 10

        # Check xgboost
        assert 'xgboost' in config['models']
        assert config['models']['xgboost']['enabled'] is True

    def test_config_evaluation_section(self, temp_config_file):
        """Test evaluation section of configuration."""
        config = load_config(temp_config_file)

        assert 'metrics' in config['evaluation']
        assert isinstance(config['evaluation']['metrics'], list)
        assert 'precision' in config['evaluation']['metrics']
        assert 'recall' in config['evaluation']['metrics']

    def test_config_preprocessing_section(self, temp_config_file):
        """Test preprocessing section of configuration."""
        config = load_config(temp_config_file)

        assert config['preprocessing']['stratify'] is True
        assert config['preprocessing']['scaling'] is True
        assert config['preprocessing']['oversampling'] is True
        assert config['preprocessing']['amount_threshold'] == 200

    def test_config_visualization_section(self, temp_config_file):
        """Test visualization section of configuration."""
        config = load_config(temp_config_file)

        assert config['visualization']['show_plots'] is False
        assert config['visualization']['figure_width'] == 12
        assert config['visualization']['figure_height'] == 8
        assert config['visualization']['output_dir'] == 'plots/results'

    def test_load_nonexistent_file(self):
        """Test loading a non-existent configuration file returns default config."""
        config = load_config('nonexistent_config.yaml')

        # Should return default configuration
        assert isinstance(config, dict)
        assert 'data' in config
        assert 'models' in config

    def test_load_invalid_yaml(self, tmp_path):
        """Test loading an invalid YAML file returns default config."""
        invalid_file = tmp_path / "invalid.yaml"
        with open(invalid_file, 'w') as f:
            f.write("invalid: yaml: content: ][")

        config = load_config(str(invalid_file))

        # Should return default configuration
        assert isinstance(config, dict)
        assert 'data' in config

    def test_default_config_structure(self):
        """Test that default configuration has required structure."""
        config = load_config('nonexistent.yaml')

        # Verify all required sections exist
        assert 'data' in config
        assert 'models' in config
        assert 'evaluation' in config
        assert 'preprocessing' in config
        assert 'visualization' in config

        # Verify data defaults
        assert 'filepath' in config['data']
        assert 'test_size' in config['data']
        assert 'random_state' in config['data']

    def test_config_immutability(self, temp_config_file):
        """Test that loaded config can be modified without affecting file."""
        config1 = load_config(temp_config_file)
        original_value = config1['data']['test_size']

        # Modify config
        config1['data']['test_size'] = 0.5

        # Load again
        config2 = load_config(temp_config_file)

        # Should have original value
        assert config2['data']['test_size'] == original_value

    def test_empty_config_file(self, tmp_path):
        """Test loading an empty YAML file."""
        empty_file = tmp_path / "empty.yaml"
        empty_file.touch()

        config = load_config(str(empty_file))

        # Empty file parses to None - check if this is handled
        assert config is None or isinstance(config, dict)

    def test_partial_config_file(self, tmp_path):
        """Test loading a partial configuration file."""
        partial_config = {
            'data': {
                'filepath': 'custom/path.csv'
            }
        }

        partial_file = tmp_path / "partial.yaml"
        with open(partial_file, 'w') as f:
            yaml.dump(partial_config, f)

        config = load_config(str(partial_file))

        # Should load the partial config
        assert config['data']['filepath'] == 'custom/path.csv'

    def test_config_with_extra_fields(self, tmp_path):
        """Test loading configuration with extra fields."""
        extra_config = {
            'data': {'filepath': 'test.csv'},
            'extra_section': {'custom_field': 'value'}
        }

        extra_file = tmp_path / "extra.yaml"
        with open(extra_file, 'w') as f:
            yaml.dump(extra_config, f)

        config = load_config(str(extra_file))

        # Should load without errors
        assert 'extra_section' in config
        assert config['extra_section']['custom_field'] == 'value'
