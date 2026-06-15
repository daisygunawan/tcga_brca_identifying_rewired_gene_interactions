# file: utils/file.py

"""
File System and Path Management Utilities

This module provides functions for file system operations, path management,
and automatic output directory generation for the co-expression analysis pipeline.

Functions:
    ensure_dir: Create directory if it doesn't exist
    get_relative_path: Compute relative path from full path relative to base directory
    get_auto_output_path: Generate automatic output path based on current script filename
"""

from pathlib import Path

def ensure_dir(path):
    """
    Create directory if it doesn't exist.
    
    Creates the specified directory path including any necessary parent directories.
    Safe to call multiple times - will not raise error if directory already exists.
    
    Parameters:
    -----------
    path : str or Path
        Directory path to create
        
    Returns:
    --------
    Path
        Created directory path as Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_relative_path(full_path, base_dir=None):
    """
    Compute a relative path string from a full path relative to a base directory.
    
    Used for generating cleaner log messages and reports by showing paths
    relative to project root instead of full absolute paths.
    
    Args:
        full_path (str or Path): The absolute or relative path to convert.
        base_dir (str or Path, optional): Base directory to compute relative to.
            Defaults to project root if not provided (assumes PROJECT_ROOT global or config).
    
    Returns:
        str: Relative path string (e.g., 'output/00_a_analysis_raw/file.json').
    
    Raises:
        ValueError: If relative computation fails.
    """
    full_path = Path(full_path)
    if base_dir is None:
        # Assume PROJECT_ROOT is available; fallback to cwd if not
        try:
            from utils.config import load_config
            config = load_config()
            base_dir = Path(config['paths']['project_root'])
        except (ImportError, KeyError):
            base_dir = Path.cwd()
    else:
        base_dir = Path(base_dir)
    
    try:
        rel = full_path.relative_to(base_dir)
        return str(rel)
    except ValueError:
        # Fallback to str(full_path) if not relative
        return str(full_path)

def get_auto_output_path(file_path, project_root=None):
    """
    Generate automatic output path based on current script filename.
    
    Automatically determines the appropriate output directory for a script
    by using the script's filename (without extension) as the output subdirectory.
    This ensures organized output structure matching the analysis pipeline structure.
    
    Args:
        file_path (str or Path): Path to the current script file (usually __file__)
        project_root (str or Path, optional): Project root directory. 
            If None, detects project root by looking for config.yaml or goes up from code/ dir.
    
    Returns:
        Path: Output directory path (e.g., project_root/output/00_a_analyse_raw_data)
    
    Example:
        # For script: code/00_a_analyse_raw_data.py
        # Returns: project_root/output/00_a_analyse_raw_data
    """
    file_path = Path(file_path)
    script_name = file_path.stem  # '00_a_analyse_raw_data' from '00_a_analyse_raw_data.py'
    
    if project_root is None:
        # Try to detect project root automatically
        current_dir = file_path.parent
        
        # Look for config.yaml in current or parent directories
        config_candidates = [
            current_dir / "config.yaml",
            current_dir.parent / "config.yaml", 
            current_dir.parent.parent / "config.yaml"
        ]
        
        for config_path in config_candidates:
            if config_path.exists():
                project_root = config_path.parent
                break
        else:
            # Fallback: if we're in a code/ directory, go up one level
            if current_dir.name == "code":
                project_root = current_dir.parent
            else:
                project_root = Path.cwd()
    
    project_root = Path(project_root)
    return project_root / "output" / script_name


def validate_input_paths(required_files, logger):
    """
    Validate that all required input files exist before analysis.
    
    Args:
        required_files (dict): {description: Path} pairs
        logger: Logger instance
        
    Returns:
        bool: True if all exist, False otherwise
        
    Raises:
        FileNotFoundError: With detailed message listing missing files
    """
    missing = []
    for desc, path in required_files.items():
        if not Path(path).exists():
            missing.append(f"  - {desc}: {path}")
    
    if missing:
        error_msg = "Missing required input files:\n" + "\n".join(missing)
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    logger.info(f"✓ Validated {len(required_files)} input paths")
    return True