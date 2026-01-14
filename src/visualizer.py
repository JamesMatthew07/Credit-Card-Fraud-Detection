import logging
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve
from typing import Dict, Any


def plot_confusion_matrix(cm, model_name: str, output_dir: str = 'plots/results'):
    """Plot confusion matrix as heatmap."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')
    
    # Labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Legitimate', 'Fraud'])
    ax.set_yticklabels(['Legitimate', 'Fraud'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {model_name}')
    
    # Add values in cells
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14)
    
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'))
    plt.close()
    logging.info(f"Saved confusion matrix for {model_name}")


def plot_roc_curve(y_test, y_proba, model_name: str, output_dir: str = 'plots/results'):
    """Plot ROC curve."""
    os.makedirs(output_dir, exist_ok=True)
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name}')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'roc_curve_{model_name.lower().replace(" ", "_")}.png'))
    plt.close()
    logging.info(f"Saved ROC curve for {model_name}")


def plot_all_metrics(results: Dict[str, Any], y_test, model_name: str, output_dir: str = 'plots/results'):
    """Generate all visualization plots for a model."""
    logging.info(f"Generating visualizations for {model_name}")
    
    plot_confusion_matrix(results['confusion_matrix'], model_name, output_dir)
    plot_roc_curve(y_test, results['y_proba'], model_name, output_dir)
    
    logging.info(f"All visualizations saved to {output_dir}")
