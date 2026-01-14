from imblearn.over_sampling import SMOTE
import pandas as pd
import logging

def apply_smote(X: pd.DataFrame, y: pd.Series) ->  tuple[pd.DataFrame, pd.Series]:
    """ Apply SMOTE to balance the dataset.
    
    Args:
        X (pd.DataFrame): Feature set.
        y (pd.Series): Target variable. 

    Returns:
        tuple[pd.DataFrame, pd.Series]: Resampled feature set and target variable.
    
    """
    smote = SMOTE(random_state=42)
    logging.info(f"Original dataset shape: {X.shape}, {y.value_counts().to_dict()}")

    X_resampled, y_resampled = smote.fit_resample(X, y)

    logging.info(f"Resampled dataset shape: {X_resampled.shape}, {y_resampled.value_counts().to_dict()}")
   
    return X_resampled, y_resampled