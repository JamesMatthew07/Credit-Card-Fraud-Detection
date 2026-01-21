"""
Unit tests for model_builder module.
"""
import pytest
from pathlib import Path
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_builder import build_model


class TestBuildModel:
    """Test suite for build_model function."""

    def test_build_all_models(self, default_config):
        """Test building all enabled models."""
        models = build_model(default_config)

        assert isinstance(models, dict)
        assert 'Logistic Regression' in models
        assert 'Random Forest' in models
        assert 'XGBoost' in models

    def test_logistic_regression_created(self, default_config):
        """Test that Logistic Regression model is created correctly."""
        models = build_model(default_config)

        assert 'Logistic Regression' in models
        assert isinstance(models['Logistic Regression'], LogisticRegression)

    def test_random_forest_created(self, default_config):
        """Test that Random Forest model is created correctly."""
        models = build_model(default_config)

        assert 'Random Forest' in models
        assert isinstance(models['Random Forest'], RandomForestClassifier)

    def test_xgboost_created(self, default_config):
        """Test that XGBoost model is created correctly."""
        models = build_model(default_config)

        assert 'XGBoost' in models
        assert isinstance(models['XGBoost'], XGBClassifier)

    def test_model_parameters_logistic_regression(self, default_config):
        """Test that Logistic Regression has correct parameters."""
        models = build_model(default_config)
        lr_model = models['Logistic Regression']

        assert lr_model.max_iter == 1000
        assert lr_model.random_state == 42

    def test_model_parameters_random_forest(self, default_config):
        """Test that Random Forest has correct parameters."""
        models = build_model(default_config)
        rf_model = models['Random Forest']

        assert rf_model.n_estimators == 100
        assert rf_model.random_state == 42

    def test_model_parameters_xgboost(self, default_config):
        """Test that XGBoost has correct parameters."""
        models = build_model(default_config)
        xgb_model = models['XGBoost']

        assert xgb_model.n_estimators == 100
        assert xgb_model.learning_rate == 0.1
        assert xgb_model.random_state == 42

    def test_disabled_models_not_built(self, default_config):
        """Test that disabled models are not built."""
        # Disable Random Forest
        default_config['models']['random_forest']['enabled'] = False

        models = build_model(default_config)

        assert 'Logistic Regression' in models
        assert 'Random Forest' not in models
        assert 'XGBoost' in models

    def test_only_logistic_regression_enabled(self, default_config):
        """Test building only Logistic Regression."""
        default_config['models']['random_forest']['enabled'] = False
        default_config['models']['xgboost']['enabled'] = False

        models = build_model(default_config)

        assert len(models) == 1
        assert 'Logistic Regression' in models

    def test_only_random_forest_enabled(self, default_config):
        """Test building only Random Forest."""
        default_config['models']['logistic_regression']['enabled'] = False
        default_config['models']['xgboost']['enabled'] = False

        models = build_model(default_config)

        assert len(models) == 1
        assert 'Random Forest' in models

    def test_only_xgboost_enabled(self, default_config):
        """Test building only XGBoost."""
        default_config['models']['logistic_regression']['enabled'] = False
        default_config['models']['random_forest']['enabled'] = False

        models = build_model(default_config)

        assert len(models) == 1
        assert 'XGBoost' in models

    def test_no_models_enabled(self, default_config):
        """Test building with no models enabled."""
        default_config['models']['logistic_regression']['enabled'] = False
        default_config['models']['random_forest']['enabled'] = False
        default_config['models']['xgboost']['enabled'] = False

        models = build_model(default_config)

        assert len(models) == 0
        assert models == {}

    def test_custom_parameters_logistic_regression(self, default_config):
        """Test Logistic Regression with custom parameters."""
        default_config['models']['logistic_regression']['max_iter'] = 500
        default_config['models']['logistic_regression']['C'] = 0.5

        models = build_model(default_config)
        lr_model = models['Logistic Regression']

        assert lr_model.max_iter == 500
        assert lr_model.C == 0.5

    def test_custom_parameters_random_forest(self, default_config):
        """Test Random Forest with custom parameters."""
        default_config['models']['random_forest']['n_estimators'] = 200

        models = build_model(default_config)
        rf_model = models['Random Forest']

        assert rf_model.n_estimators == 200

    def test_custom_parameters_xgboost(self, default_config):
        """Test XGBoost with custom parameters."""
        default_config['models']['xgboost']['n_estimators'] = 50
        default_config['models']['xgboost']['learning_rate'] = 0.05

        models = build_model(default_config)
        xgb_model = models['XGBoost']

        assert xgb_model.n_estimators == 50
        assert xgb_model.learning_rate == 0.05

    def test_models_are_untrained(self, default_config):
        """Test that built models are untrained."""
        models = build_model(default_config)

        # Check that models haven't been fitted
        # (they won't have certain attributes until fitted)
        lr = models['Logistic Regression']
        rf = models['Random Forest']
        xgb = models['XGBoost']

        assert not hasattr(lr, 'coef_') or lr.coef_ is None
        assert not hasattr(rf, 'estimators_') or len(rf.estimators_) == 0
        assert not hasattr(xgb, 'feature_importances_') or xgb.feature_importances_ is None

    def test_models_can_be_fitted(self, default_config, sample_features, sample_target):
        """Test that built models can be fitted."""
        models = build_model(default_config)

        # Try fitting each model
        for model_name, model in models.items():
            model.fit(sample_features, sample_target)
            predictions = model.predict(sample_features)
            assert len(predictions) == len(sample_target)

    def test_logistic_regression_solver_parameter(self, default_config):
        """Test Logistic Regression with solver parameter."""
        default_config['models']['logistic_regression']['solver'] = 'liblinear'

        models = build_model(default_config)
        lr_model = models['Logistic Regression']

        assert lr_model.solver == 'liblinear'

    def test_config_missing_parameters(self, default_config):
        """Test that missing optional parameters use sklearn defaults."""
        # Remove optional parameter if it exists
        if 'C' in default_config['models']['logistic_regression']:
            del default_config['models']['logistic_regression']['C']

        models = build_model(default_config)

        # Should still build successfully (sklearn will use its default C)
        assert 'Logistic Regression' in models
        # Just check the model exists, default C might vary by sklearn version
        assert isinstance(models['Logistic Regression'], LogisticRegression)

    def test_multiple_random_states(self, default_config):
        """Test models with different random states."""
        default_config['models']['logistic_regression']['random_state'] = 10
        default_config['models']['random_forest']['random_state'] = 20
        default_config['models']['xgboost']['random_state'] = 30

        models = build_model(default_config)

        assert models['Logistic Regression'].random_state == 10
        assert models['Random Forest'].random_state == 20
        assert models['XGBoost'].random_state == 30

    def test_model_dictionary_keys(self, default_config):
        """Test that model dictionary has correct keys."""
        models = build_model(default_config)

        expected_keys = {'Logistic Regression', 'Random Forest', 'XGBoost'}
        assert set(models.keys()) == expected_keys

    def test_empty_config_section(self):
        """Test with empty models configuration."""
        empty_config = {'models': {}}

        models = build_model(empty_config)

        assert models == {}
