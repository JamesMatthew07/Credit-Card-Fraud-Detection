"""
Credit Card Fraud Detection - Main Pipeline
============================================
Run this script to train and evaluate fraud detection models.

Usage:
    python main.py
"""

import logging
from src.config_loader import load_config
from src.data_loader import load_data, preprocess_data, split_data
from src.model_builder import build_model
from src.model_evaluator import evaluate_model
from src.visualizer import plot_all_metrics
from src.data_preprocessor import apply_smote


def main():
    """Main pipeline for credit card fraud detection."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logging.info("=" * 60)
    logging.info("Credit Card Fraud Detection Pipeline")
    logging.info("=" * 60)
    
    # 1. Load configuration
    config = load_config()
    
    # 2. Load and preprocess data
    X, y = load_data(config['data']['filepath'])
    X_processed = preprocess_data(X)
    X_train, X_test, y_train, y_test = split_data(X_processed, y)
    X_resampled, y_resampled = apply_smote(X_train, y_train)
    
    # 3. Build models
    models = build_model(config)
    
    # 4. Train and evaluate each model
    all_results = {}
    
    for name, model in models.items():
        logging.info(f"\n{'='*60}")
        logging.info(f"Training {name}...")
        logging.info(f"{'='*60}")
        
        # Train
        model.fit(X_resampled, y_resampled)
        
        # Evaluate
        results = evaluate_model(model, X_test, y_test, name)
        all_results[name] = results
        
        # Visualize
        plot_all_metrics(results, y_test, name, config['visualization']['output_dir'])
        
        # Print results
        print(f"\n{name} Results:")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        print(f"  F1 Score:  {results['f1_score']:.4f}")
        print(f"  ROC-AUC:   {results['roc_auc']:.4f}")
    
    # 5. Summary comparison
    logging.info(f"\n{'='*60}")
    logging.info("MODEL COMPARISON SUMMARY")
    logging.info(f"{'='*60}")
    
    print("\n" + "=" * 70)
    print(f"{'Model':<25} {'Precision':<12} {'Recall':<12} {'F1':<12} {'ROC-AUC':<12}")
    print("=" * 70)
    
    for name, results in all_results.items():
        print(f"{name:<25} {results['precision']:<12.4f} {results['recall']:<12.4f} {results['f1_score']:<12.4f} {results['roc_auc']:<12.4f}")
    
    print("=" * 70)
    logging.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
