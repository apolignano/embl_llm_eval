import os
import yaml
import logging
import pandas as pd
import importlib

def validate_task_folder(task_folder):
    """Check if the task folder exists and contains the required files."""
    if not os.path.isdir(task_folder):
        raise FileNotFoundError(f"Task folder '{task_folder}' does not exist.")
    
    config_path = os.path.join(task_folder, "config.yml")
    dataset_path = os.path.join(task_folder, "dataset.csv")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Missing 'config.yml' in '{task_folder}'.")
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Missing 'dataset.csv' in '{task_folder}'.")
    
    return config_path, dataset_path

def create_results_folder(results_folder):
    """Create results folder if it doesn't exist."""
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
        logging.info(f"Created results folder: {results_folder}")

def validate_config(config_path):
    """Check if task_config.yml contains the required parameters and return their values."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    required_keys = {"models_list", "scoring", "prompt", "dataset_separator"}
    missing_keys = required_keys - config.keys()
    
    if missing_keys:
        raise ValueError(f"task_config.yml is missing required keys: {missing_keys}")

    # Return only the required keys and their values
    return {key: config[key] for key in config.keys()}


def validate_dataset(dataset_path, sep=","):
    """Check if dataset.csv has the required 'input' and 'target' columns, then return both."""
    df = pd.read_csv(dataset_path, sep=sep)

    if "input" not in df.columns:
        raise ValueError("'dataset.csv' must contain an 'input' column.")
        
    if "target" not in df.columns:
        raise ValueError("'dataset.csv' must contain a 'target' column.")
    
    return df["input"].tolist(), df["target"].tolist()

def import_task_func(task_name: str, func_type: str, func_name: str):

    if "." in func_name: 
        module_to_import = f"{task_name}.functions"
        func_name = func_name.split(".")[1]
    else:
        module_to_import = func_type

    module_functions = importlib.import_module(module_to_import)
    preprocess_func = getattr(module_functions, func_name)
    
    return preprocess_func

