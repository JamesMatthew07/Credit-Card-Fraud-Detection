# Credit Card Fraud Detection

A machine learning project for detecting fraudulent credit card transactions using multiple classification models. The system handles class imbalance using SMOTE and provides comprehensive evaluation metrics and visualizations.

## Overview

This project implements a complete ML pipeline for credit card fraud detection, featuring:
- Multiple classification models (Logistic Regression, Random Forest, XGBoost)
- Handling of highly imbalanced datasets using SMOTE
- Comprehensive evaluation metrics and visualizations
- Modular, configuration-driven architecture
- Extensive test coverage

## Features

- **Multiple ML Models**: Compare performance across Logistic Regression, Random Forest, and XGBoost
- **Class Imbalance Handling**: SMOTE (Synthetic Minority Over-sampling Technique) for balancing fraud/legitimate transactions
- **Configuration-Driven**: YAML-based configuration for easy parameter tuning
- **Comprehensive Evaluation**: Precision, recall, F1-score, ROC-AUC, and average precision metrics
- **Rich Visualizations**: Confusion matrices, ROC curves, and precision-recall curves
- **Hyperparameter Tuning**: GridSearchCV for model optimization
- **Exploratory Data Analysis**: EDA scripts for data understanding
- **Extensive Testing**: Pytest suite with comprehensive fixtures

## Project Structure

```
credit_card_fraud_detection/
├── src/                          # Source code modules
│   ├── config_loader.py         # YAML configuration management
│   ├── data_loader.py           # Data loading and splitting
│   ├── data_preprocessor.py     # SMOTE resampling
│   ├── model_builder.py         # ML model instantiation
│   ├── model_evaluator.py       # Model evaluation
│   ├── visualizer.py            # Visualization generation
│   └── hyperparameter_tuner.py  # GridSearchCV tuning
├── test/                         # Test suite
│   ├── conftest.py              # Pytest fixtures
│   ├── test_config_loader.py
│   ├── test_data_loader.py
│   ├── test_data_preprocessor.py
│   ├── test_hyperparameter_tuner.py
│   ├── test_model_builder.py
│   ├── test_model_evaluator.py
│   └── test_visualizer.py
├── dataset/                      # Data directory
│   ├── creditcard.csv           # Credit card transactions dataset
│   └── dataset.md               # Dataset documentation
├── plots/                        # Generated visualizations
│   ├── results/                 # Model-specific plots
│   ├── class_distribution.png
│   ├── amount_by_class_boxplot.png
│   └── log_amount_distribution.png
├── main.py                       # Main ML pipeline
├── tune_model.py                 # Hyperparameter tuning script
├── eda_exploratory.py            # Exploratory Data Analysis
└── config.yaml                   # Configuration file
```

## Installation

### Prerequisites

- Python 3.7+
- pip

### Dependencies

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib pyyaml pytest
```

Or install from requirements file (if available):

```bash
pip install -r requirements.txt
```

## Dataset

The project uses the Credit Card Fraud Detection dataset:
- **Source**: Kaggle Credit Card Fraud Detection Dataset
- **Size**: 284,808 transactions
- **Features**:
  - V1-V28: PCA-transformed features (confidential)
  - Time: Seconds elapsed between transactions
  - Amount: Transaction amount
- **Target**: Class (0 = Legitimate, 1 = Fraud)
- **Class Distribution**: Highly imbalanced (~0.17% fraud transactions)

Place the dataset at [dataset/creditcard.csv](dataset/creditcard.csv).

## Configuration

Edit [config.yaml](config.yaml) to customize the pipeline:

```yaml
data:
  filepath: "dataset/creditcard.csv"
  test_size: 0.2
  random_state: 42

models:
  logistic_regression:
    enabled: true
    max_iter: 1000
  random_forest:
    enabled: true
    n_estimators: 150
  xgboost:
    enabled: true
    n_estimators: 100
    learning_rate: 0.1

evaluation:
  metrics: [precision, recall, f1_score, roc_auc]

preprocessing:
  stratify: true
  scaling: true
  oversampling: true
  amount_threshold: 200
```

## Usage

### 1. Run Complete ML Pipeline

Execute the main pipeline to train and evaluate all models:

```bash
python main.py
```

This will:
- Load and preprocess the data
- Apply SMOTE resampling to balance classes
- Train enabled models (Logistic Regression, Random Forest, XGBoost)
- Evaluate models and compute metrics
- Generate visualizations in [plots/results/](plots/results/)
- Display comparison summary

### 2. Exploratory Data Analysis

Run EDA to understand the dataset:

```bash
python eda_exploratory.py
```

Generates:
- Class distribution plots
- Transaction amount boxplots
- Log-scaled amount distributions
- Basic statistics

### 3. Hyperparameter Tuning

Optimize model parameters using GridSearchCV:

```bash
python tune_model.py
```

Tunes Random Forest hyperparameters:
- n_estimators: [100, 150, 200]
- max_depth: [10, 20, None]
- Optimizes for recall metric

### 4. Run Tests

Execute the test suite:

```bash
pytest test/
```

For verbose output:

```bash
pytest test/ -v
```

## Models

### 1. Logistic Regression
- Simple linear classifier
- Fast training and inference
- Good baseline model
- max_iter: 1000

### 2. Random Forest
- Ensemble of decision trees
- Handles non-linear relationships
- Feature importance analysis
- n_estimators: 150

### 3. XGBoost
- Gradient boosting algorithm
- High performance on imbalanced data
- Advanced regularization
- n_estimators: 100, learning_rate: 0.1

## Evaluation Metrics

The pipeline evaluates models using:

- **Precision**: Of predicted fraud, how many are actually fraud?
- **Recall**: Of actual fraud, how many are detected?
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve (TPR vs FPR)
- **Average Precision**: Area under precision-recall curve

### Why These Metrics Matter for Fraud Detection

- **Recall** is critical: Missing fraud (false negatives) is costly
- **Precision** prevents alert fatigue: Too many false alarms reduce trust
- **F1-Score** balances both concerns
- **ROC-AUC** shows overall discrimination ability
- **PR-AUC** is more informative for imbalanced datasets

## Visualizations

The pipeline generates three types of plots for each model:

### 1. Confusion Matrix
- True Positives, True Negatives, False Positives, False Negatives
- Saved as [confusion_matrix_*.png](plots/results/)

### 2. ROC Curve
- True Positive Rate vs False Positive Rate
- Includes AUC score
- Saved as [roc_curve_*.png](plots/results/)

### 3. Precision-Recall Curve
- Precision vs Recall trade-off
- More informative for imbalanced data
- Saved as [pr_curve_*.png](plots/results/)

## Class Imbalance Handling

The dataset is highly imbalanced (~0.17% fraud). The project addresses this using:

### SMOTE (Synthetic Minority Over-sampling Technique)
- Generates synthetic fraud examples
- Balances training data
- Applied only to training set (not test set)
- Configured in [config.yaml](config.yaml) under `preprocessing.oversampling`

## Pipeline Workflow

1. **Load Configuration**: Parse [config.yaml](config.yaml)
2. **Load Data**: Read [dataset/creditcard.csv](dataset/creditcard.csv)
3. **Preprocess**: Scale Amount and Time features using StandardScaler
4. **Split**: Stratified 80-20 train-test split
5. **Resample**: Apply SMOTE to training data
6. **Build Models**: Instantiate enabled classifiers
7. **Train**: Fit models on resampled training data
8. **Evaluate**: Compute metrics on test set
9. **Visualize**: Generate confusion matrices and curves
10. **Compare**: Display summary comparison table

## Testing

The project includes comprehensive pytest coverage:

### Test Fixtures ([test/conftest.py](test/conftest.py))
- `sample_data`: 1000 imbalanced transactions
- `balanced_data`: 500 balanced transactions
- `temp_csv_file`: Temporary test datasets
- `temp_config_file`: Temporary configurations
- `trained_predictions`: Sample model outputs
- `temp_output_dir`: Temporary plot directories

### Test Modules
- `test_config_loader.py`: Configuration loading
- `test_data_loader.py`: Data loading and splitting
- `test_data_preprocessor.py`: SMOTE resampling
- `test_hyperparameter_tuner.py`: GridSearchCV
- `test_model_builder.py`: Model instantiation
- `test_model_evaluator.py`: Metrics computation
- `test_visualizer.py`: Plot generation

Run tests with:
```bash
pytest test/ -v
```

## Module Documentation

### [src/config_loader.py](src/config_loader.py)
Loads and parses YAML configuration with fallback to defaults.

### [src/data_loader.py](src/data_loader.py)
- `load_data()`: Load CSV and separate features/target
- `preprocess_data()`: Scale Amount and Time features
- `split_data()`: Stratified train-test split

### [src/data_preprocessor.py](src/data_preprocessor.py)
- `apply_smote()`: Balance training data using SMOTE

### [src/model_builder.py](src/model_builder.py)
- `build_models()`: Instantiate classifiers based on config

### [src/model_evaluator.py](src/model_evaluator.py)
- `evaluate_model()`: Compute all metrics
- `evaluate_thresholds()`: Test performance at different thresholds

### [src/visualizer.py](src/visualizer.py)
- `plot_confusion_matrix()`: Generate confusion matrix heatmap
- `plot_roc_curve()`: Generate ROC curve
- `plot_pr_curve()`: Generate precision-recall curve

### [src/hyperparameter_tuner.py](src/hyperparameter_tuner.py)
- `tune_hyperparameters()`: GridSearchCV for Random Forest optimization

## Best Practices

1. **Always use stratified splitting** to maintain class distribution
2. **Apply SMOTE only to training data** to prevent data leakage
3. **Monitor recall** as primary metric for fraud detection
4. **Review confusion matrices** to understand error types
5. **Use PR curves** instead of ROC for imbalanced datasets
6. **Tune thresholds** based on business requirements (cost of false positives vs false negatives)

## Future Enhancements

- [ ] Add neural network models (deep learning)
- [ ] Implement ensemble methods (voting, stacking)
- [ ] Add feature engineering pipeline
- [ ] Implement real-time prediction API
- [ ] Add model persistence (save/load trained models)
- [ ] Create web dashboard for monitoring
- [ ] Add time-series analysis for fraud trends
- [ ] Implement cost-sensitive learning
- [ ] Add explainability (SHAP, LIME)

## Troubleshooting

### Memory Issues
If dataset is too large, consider:
- Using sampling for development
- Implementing batch processing
- Using incremental learning algorithms

### Class Imbalance Still Affecting Results
- Try different resampling ratios in SMOTE
- Experiment with class weights in models
- Use ensemble methods
- Try anomaly detection approaches

### Poor Model Performance
- Check data preprocessing (scaling, normalization)
- Verify SMOTE is applied correctly
- Tune hyperparameters using GridSearchCV
- Try different algorithms
- Add feature engineering

## Acknowledgments

- Dataset: Kaggle Credit Card Fraud Detection Dataset
- SMOTE: imbalanced-learn library
- ML Framework: scikit-learn
- Gradient Boosting: XGBoost
