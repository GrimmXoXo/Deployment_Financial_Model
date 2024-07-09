# src/utils.py
from pathlib import Path
import yaml

def load_config(filename):
    config_path = Path(__file__).parent.parent / filename
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as exc:
        print(f"Error reading YAML file: {exc}")
    return None
