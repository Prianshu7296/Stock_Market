import yaml
import joblib

def load_config(config_path):
    """
    Loads the YAML config file.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_model(model, file_path):
    """
    Save model to a file.
    """
    joblib.dump(model, file_path)

def load_model(file_path):
    """
    Load model from a file.
    """
    return joblib.load(file_path)
