import pandas as pd
import logging
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def load_data(filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load dataset from a CSV file.
    
    Args:
        filepath: Path to the CSV file.
    
    Returns:
        A tuple containing features DataFrame and target Series.
    """
    try:
        data = pd.read_csv(filepath)
        X = data.drop('Class', axis=1)
        y = data['Class']

        logging.info(f"Loading data from {filepath}")
        logging.info(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        logging.info(f"Class distribution: {y.value_counts().to_dict()}")

        return X, y

    except FileNotFoundError:
        logging.error(f"Error: The file {filepath} was not found")
        sys.exit(1)

    except pd.errors.EmptyDataError:
        logging.error("Error: The provided CSV file is empty")
        sys.exit(1)

    except KeyError as e:
        logging.error(f"Error: Missing expected column in the dataset - {e}")
        sys.exit(1)

def preprocess_data(X:pd.DataFrame) -> pd.DataFrame:
    """Preprocess the features by scaling them.
    
    Args:
        X: Features DataFrame.
        
    Returns:
        Scaled features DataFrame.
    """
    logging.info("Preprocessing data: Scaling Amount and Time")

    X_processed = X.copy() #Do not modify original data

    scaler = StandardScaler()
    X_processed[['Amount', 'Time']] = scaler.fit_transform(X_processed[['Amount', 'Time']])

    logging.info("Data preprocessing completed")

    return X_processed


def split_data(X:pd.DataFrame, y:pd.Series, test_size: float = 0.2, random_state: int = 42, stratify: bool = True) -> Tuple:
    """Split data into train and test sets with stratification.
    
    Args:
        X: Features DataFrame
        y: Target Series
        test_size: Proportion of data for testing (default 0.2)
        random_state: Random seed for reproducibility (default 42)
        stratify: Whether to maintain class distribution in splits (default True)
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """

    logging.info(f"Splitting data: {test_size*100:.0f}% test, stratify={stratify}")

    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )  
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    logging.info(f"Train set: {X_train.shape[0]} samples")
    logging.info(f"Test set: {X_test.shape[0]} samples")
    logging.info(f"Train fraud %: {(y_train.sum()/len(y_train))*100:.2f}%")
    logging.info(f"Test fraud %: {(y_test.sum()/len(y_test))*100:.2f}%")

    return X_train, X_test, y_train, y_test
    
    
