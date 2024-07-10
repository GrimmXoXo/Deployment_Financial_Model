# # src/utils.py
# from pathlib import Path
# import yaml

# def load_config(filename):
#     config_path = Path(__file__).parent.parent / filename
#     print(config_path)
#     try:
#         with open(config_path, 'r') as file:
#             return yaml.safe_load(file)
#     except FileNotFoundError:
#         print(f"Configuration file not found: {config_path}")
#     except yaml.YAMLError as exc:
#         print(f"Error reading YAML file: {exc}")
#     return None

    
# load_config('../config/config.yaml')


# src/utils.py

import os
import yaml
from pkg_resources import resource_filename

def load_config(filename='config.yaml'):
    config_path = resource_filename(__name__, os.path.join('config', filename))

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"The configuration file '{config_path}' does not exist.")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    return config

