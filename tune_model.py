"""
Test script for hyperparameter tuning.
"""

import logging
from src.config_loader import load_config
from src.data_loader import load_data, preprocess_data, split_data
from src.data_preprocessor import apply_smote
from src.hyperparameter_tuner import tune_hyperparameters
from src.model_evaluator import evaluate_model


def main():
    """Test hyperparameter tuning pipeline."""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info("=" * 60)
    logging.info("Hyperparameter Tuning Test")
    logging.info("=" * 60)

    # 1. Load and preprocess data
    config = load_config()
    X, y = load_data(config['data']['filepath'])
    X_processed = preprocess_data(X)
    X_train, X_test, y_train, y_test = split_data(X_processed, y)

    # 2. Apply SMOTE
    X_resampled, y_resampled = apply_smote(X_train, y_train)

    # 3. Define param_grid
    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [10, 20],
    }

    # 4. Tune hyperparameters
    logging.info("Starting hyperparameter tuning...")
    best_model, best_params = tune_hyperparameters(X_resampled, y_resampled, param_grid)

    print(f"\nBest Parameters: {best_params}")

    # 5. Evaluate the tuned model
    logging.info("Evaluating tuned model...")
    results = evaluate_model(best_model, X_test, y_test, "Tuned Random Forest")

    print(f"\nTuned Random Forest Results:")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1 Score:  {results['f1_score']:.4f}")
    print(f"  ROC-AUC:   {results['roc_auc']:.4f}")
    print(f"  Avg Precision Score: {results['average_precision_score']:.4f}")

    logging.info("Tuning test completed!")


if __name__ == "__main__":
    main()
