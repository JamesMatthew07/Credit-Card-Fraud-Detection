import logging

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from typing import Dict, Any

def build_model(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build ML models based on configuration.
    
    Args:
        config: Configuration dictionary with model parameters
        
    Returns:
        Dictionary of model name -> model instance
    """
    models = {}
    model_config = config.get('models', {})

    # Logistic Regression
    if model_config.get('logistic_regression', {}).get('enabled', False):
        lr_params = model_config['logistic_regression']
        models['Logistic Regression'] = LogisticRegression(
            max_iter=lr_params.get('max_iter', 1000),
            random_state=lr_params.get('random_state', 42),
            class_weight=lr_params.get('class_weight', 'balanced')
        )
    
    #Random Forest
    if model_config.get('random_forest', {}).get('enabled', False):
        rf_params = model_config['random_forest']
        models['Random Forest'] = RandomForestClassifier(
            n_estimators=rf_params.get('n_estimators', 150),
            random_state=rf_params.get('random_state', 42),
            class_weight=rf_params.get('class_weight', 'balanced')
        )

    # XGBoost
    if model_config.get('xgboost', {}).get('enabled', False):
        xgb_params = model_config['xgboost']
        models['XGBoost'] = XGBClassifier(
            n_estimators=xgb_params.get('n_estimators', 100),
            learning_rate=xgb_params.get('learning_rate',0.1),
            random_state=xgb_params.get('random_state', 42),
            scale_pos_weight=xgb_params.get('scale_pos_weight', 289)
        )

    logging.info(f"Built {len(models)} models: {list(models.keys())}")
    return models