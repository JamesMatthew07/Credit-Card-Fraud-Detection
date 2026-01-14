import logging
import numpy as np

from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, 
    confusion_matrix, classification_report, average_precision_score
)

from typing import Dict, Any


def evaluate_model(model, X_test, y_test, model_name: str) -> Dict[str, Any]:
    """Evaluate a trained model on test data.
    
    Args:
        model: Trained model with predict and predict_proba methods
        X_test: Test features
        y_test: True labels
        model_name: Name of the model for logging
        
    Returns:
        Dictionary with evaluation metrics
    """
    logging.info(f"Evaluating model: {model_name}")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] # Probability of class 1 (fraud)

    results = {
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'average_precision_score': average_precision_score(y_test, y_proba),
        'y_pred': y_pred,
        'y_proba': y_proba
    }

    return results

def evaluate_thresholds(y_test, y_proba, thresholds: list) -> dict:
    """Evaluate model performance at different classification thresholds.
    
    Args:
        y_test: True labels
        y_proba: Predicted probabilities for the positive class
        thresholds: List of thresholds to evaluate
        
    Returns:
        Dictionary mapping threshold to classification report
    """
    results = {}
    for threshold in thresholds:
        y_pred_thresholded = (y_proba >= threshold).astype(int)
        report = classification_report(y_test, y_pred_thresholded, output_dict=True)
        results[threshold] = report
        logging.info(f"Threshold: {threshold:.2f} - Precision: {report['1']['precision']:.4f}, Recall: {report['1']['recall']:.4f}, F1-Score: {report['1']['f1-score']:.4f}")
    return results