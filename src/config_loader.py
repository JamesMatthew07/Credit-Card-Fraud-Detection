import yaml
import logging

def load_config(config_path: str = 'config.yaml') -> dict:
    """Load configuration from YAML file or return default configuration.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    
    Returns:
        A dictionary containing configuration parameters.
    """

    #Default configuration
    default_config = {
        'data': {'filepath': 'dataset/creditcard.csv', 'test_size': 0.2, 'random_state': 42},
        'models': {
            'logistic_regression': {'enabled': True, 'max_iter': 1000, 'random_state': 42 },
            'random_forest': {'enabled': True, 'n_estimators': 150, 'random_state': 42},
            'xgboost': {'enabled': True, 'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42}
        },
        'evaluation': {'metrics': ['precision', 'recall', 'f1_score', 'roc_auc']},
        'preprocessing': {'stratify': True, 'scaling': True, 'oversampling': True, 'amount_threshold': 200},
        'visualization': {'show_plots': True, 'figure_width': 12, 'figure_height': 8, 'output_dir': 'plots/results'}
    }

    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            return config
    except FileNotFoundError:
        logging.warning(f"Configuration file {config_path} not found. Using default configuration.")
        return default_config 
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}. Using default configuration.")
        return default_config