# file: utils/config.py

"""
Configuration Management Utilities

This module provides functions for loading, validating, and managing configuration
settings from YAML files, with support for path resolution and placeholder substitution.

Functions:
    replace_placeholders: Recursively replace placeholders in config with actual values
    load_config: Load and validate configuration from config.yaml with path resolution
"""

import yaml
from pathlib import Path
import glob
import os

def replace_placeholders(config_dict, placeholder, replacement):
    """
    Recursively replace placeholders in config with actual values.
    
    Traverses through nested dictionaries and lists to replace all occurrences
    of a placeholder string with the specified replacement value.
    
    Parameters:
    -----------
    config_dict : dict, list, or str
        The configuration data structure to process
    placeholder : str
        The placeholder string to replace (e.g., '[project_root]')
    replacement : str
        The actual value to substitute for the placeholder
        
    Returns:
    --------
    dict, list, or str
        Configuration with all placeholders replaced
    """
    if isinstance(config_dict, dict):
        return {k: replace_placeholders(v, placeholder, replacement) for k, v in config_dict.items()}
    elif isinstance(config_dict, list):
        return [replace_placeholders(item, placeholder, replacement) for item in config_dict]
    elif isinstance(config_dict, str):
        return config_dict.replace(placeholder, str(replacement))
    return config_dict

def load_config(config_path="code/configs/config.yaml"):
    """
    Load and validate configuration from config.yaml.
    
    Loads the main configuration file, resolves project root paths, replaces
    placeholders, validates required sections, and ensures output directories exist.
    
    Args:
        config_path (str): Path to config.yaml, relative to project root.
    
    Returns:
        dict: Configuration dictionary with resolved paths and validated structure.
    
    Raises:
        FileNotFoundError: If config file or sample sheet not found
        ValueError: If required configuration sections are missing
    """
    # Resolve project root (directory containing 'code', 'data', 'output')
    project_root = Path(__file__).parent.parent.parent.resolve()
    
    # Load config.yaml
    config_file = project_root / config_path
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Replace [project_root] placeholders with actual path
    config = replace_placeholders(config, '[project_root]', project_root)
    
    # Validate required sections exist in configuration
    required_sections = ['project', 'paths', 'preprocessing', 'logging']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section '{section}' in config.yaml")
    
    # Resolve and validate paths section
    paths = config['paths']
    
    # Convert string paths to Path objects for easier manipulation
    for key, value in paths.items():
        if isinstance(value, str) and not key.endswith('_glob'):
            paths[key] = Path(value)
    
    # Handle sample_sheet glob pattern - find matching files
    sample_sheet_glob = str(paths['sample_sheet'])
    sample_sheets = glob.glob(sample_sheet_glob)
    if not sample_sheets:
        raise FileNotFoundError(f"No sample sheet found matching: {sample_sheet_glob}")
    if len(sample_sheets) > 1:
        print(f"Warning: Multiple sample sheets found, using first: {sample_sheets[0]}")
    paths['sample_sheet'] = Path(sample_sheets[0])
    
    # Create output directories if they don't exist
    for key in paths:
        if key.startswith('output_') and paths[key].is_dir():
            paths[key].mkdir(parents=True, exist_ok=True)
    
    return config