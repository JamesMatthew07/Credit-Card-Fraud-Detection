from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import logging
from typing import Dict, Any, Tuple

def tune_hyperparameters(X_train, y_train, param_grid: Dict[str, list], scoring: str = 'recall', cv: int = 3) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Tune Random Forest hyperparameters using GridSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training labels
        param_grid: Dictionary of hyperparameters to search
        scoring: Metric to optimize (default: recall)
        cv: Number of cross-validation folds
        
    Returns:
        Tuple of (best_model, best_params)
    """

    logging.info("Starting hyperparameter tuning using GridSearchCV")

    rf = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV( estimator=rf, param_grid=param_grid, scoring = scoring, cv=cv, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    logging.info(f"Best hyperparameters found: {grid_search.best_params_}")
    logging.info(f"Best CV score ({scoring}): {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_