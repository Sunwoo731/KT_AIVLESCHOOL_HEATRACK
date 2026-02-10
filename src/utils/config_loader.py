import yaml
import os

def load_config(config_path="configs/config.yaml"):
    """Load YAML configuration."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}. Copy configs/config.example.yaml to configs/config.yaml")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def ensure_dirs(config):
    """Ensure all directories in paths exist."""
    for key, path in config['paths'].items():
        if key == 'simulated_pipes': continue # File path
        os.makedirs(path, exist_ok=True)
