"""
01_a_build_correlation_matrices.py

Script Purpose:
This script loads preprocessed gene expression matrices (tumor and normal) from 00_b, transposes them for correlation computation, calculates correlation matrices using configurable methods (e.g., Spearman, Pearson), saves them in compressed NPZ format (matrix values + gene indices), and generates a summary JSON with stats (shapes, mean/std/min/max/median of upper triangle correlations). It focuses on efficient handling of large matrices for downstream network analysis in cancer genomics workflows.

Summary Logic:
1. Load config, set up logging, verify/ load transposed expression matrices (genes as columns).
2. For each correlation method in config: compute corr matrix (tumor/normal) with progress bars.
3. Save matrices as NPZ (compressed, with genes array); compute upper-triangle stats (exclude diagonal=1).
4. Aggregate run summary (inputs/outputs, timings, matrix stats) and save as JSON.
5. Use tqdm for progress, relative paths for logs, error handling for missing files.

Key Features:
- Configurable methods (e.g., 'spearman', 'pearson') via config['network_analysis']['correlation_methods'].
- Memory-efficient: Transpose once, use np.savez_compressed for large matrices.
- Progress tracking: tqdm bars for long computations; timings logged.
- Summary JSON: Relative paths, stats on correlations (focus on off-diagonal for relevance).
- Error-tolerant: Checks inputs, logs exceptions with traceback.

Dependencies: See imports below. Assumes utils.config, utils.file; inputs from 00_b_data_preprocess.py.
"""

import pandas as pd
import numpy as np
import logging
import json
import time
from pathlib import Path
from tqdm import tqdm
from utils.config import load_config
from utils.file import ensure_dir, get_relative_path, get_auto_output_path


def setup_logging(config, output_dir):
    """
    Set up logging with a clean format for console output.
    
    Configures file handler (detailed, timestamped) and optional console (simple);
    clears existing handlers; creates logs/ dir.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(config['logging']['level'])

    if logger.hasHandlers():
        logger.handlers.clear()

    # Create log directory in output folder
    log_dir = output_dir / 'logs'
    ensure_dir(log_dir)
    log_file = log_dir / 'correlation_matrices.log'
    
    # File handler with full timestamped format
    file_handler = logging.FileHandler(log_file, mode='w')
    file_formatter = logging.Formatter(config['logging']['format'])
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler with simple, clean messages
    if config['logging']['console_log']:
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
    return logger


def sort_nested_dict(obj):
    """
    Recursively sort nested dictionaries by keys for consistent JSON output.
    Returns a new sorted dictionary.
    """
    if isinstance(obj, dict):
        # Sort dictionary keys and recursively sort values
        return {k: sort_nested_dict(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, list):
        # For lists, sort items if they're dictionaries
        return [sort_nested_dict(item) if isinstance(item, (dict, list)) else item for item in obj]
    else:
        # Return non-dict, non-list items as-is
        return obj


def create_summary_json(summary_data, output_path, project_root):
    """Creates a JSON file with summary statistics of the run."""
    # Convert all Path objects to clean, relative string paths for the JSON output
    for section in ['inputs', 'outputs']:
        for key, value in summary_data[section].items():
            if isinstance(value, Path):
                summary_data[section][key] = str(value.relative_to(project_root))
    
    # Sort the entire summary data structure for consistent output
    sorted_summary_data = sort_nested_dict(summary_data)
    
    with open(output_path, 'w') as f:
        json.dump(sorted_summary_data, f, indent=4)


def main():
    """
    Loads preprocessed expression matrices, calculates multiple types of
    correlation matrices, and saves them in a compressed format with a summary.
    """
    config = load_config()
    
    # Set up paths using the established pattern
    PROJECT_ROOT = Path(config['paths']['project_root'])
    INPUT_PREPROCESSED = Path(config['paths']['preprocessed'])  # Input from 00_b
    OUTPUT_DIR = get_auto_output_path(__file__, PROJECT_ROOT)  # Auto-generated output
    
    logger = setup_logging(config, OUTPUT_DIR)
    
    logger.info(f"00_b input directory: {get_relative_path(INPUT_PREPROCESSED)}")
    logger.info(f"01_a output directory: {get_relative_path(OUTPUT_DIR)}")
    logger.info("Starting script: 01_a_build_correlation_matrices.py")
    logger.info("-" * 50)
    
    # Initialize a dictionary to hold summary stats for the JSON report.
    summary_stats = {
        "script": "01_a_build_correlation_matrices",
        "start_time": time.strftime('%Y-%m-%d %H:%M:%S'),
        "parameters": {
            "correlation_methods": config['network_analysis']['correlation_methods']
        },
        "inputs": {
            "tumor_matrix": INPUT_PREPROCESSED / 'matrices' / 'tumor_matrix.tsv',
            "normal_matrix": INPUT_PREPROCESSED / 'matrices' / 'normal_matrix.tsv'
        },
        "outputs": {},
        "matrix_stats": {}
    }

    try:
        # Verify input files exist
        for file_type, file_path in summary_stats['inputs'].items():
            if not file_path.exists():
                logger.error(f"Required input file not found: {get_relative_path(file_path)}")
                logger.error("Please run 00_b_data_preprocess.py first")
                return

        # --- 1. Load Preprocessed Data ---
        logger.info("Loading preprocessed expression matrices...")
        
        tumor_df = pd.read_csv(summary_stats['inputs']['tumor_matrix'], sep='\t', index_col='gene_key')
        normal_df = pd.read_csv(summary_stats['inputs']['normal_matrix'], sep='\t', index_col='gene_key')
        
        logger.info(f"Loaded tumor matrix with shape: {tumor_df.shape}")
        logger.info(f"Loaded normal matrix with shape: {normal_df.shape}")
        logger.info("")

        # --- 2. Transpose Matrices ---
        # Correlation is calculated between columns, so genes must become columns.
        logger.info("Transposing matrices...")
        tumor_df_T = tumor_df.transpose()
        normal_df_T = normal_df.transpose()
        logger.info("")

        # --- 3. Calculate and Save Correlation Matrices ---
        # Loop through all methods specified in the config (e.g., "spearman", "pearson").
        methods = config['network_analysis'].get('correlation_methods', ['spearman'])
        
        # Ensure output directory structure
        matrices_dir = OUTPUT_DIR / 'matrices'
        ensure_dir(matrices_dir)
        
        for method in methods:
            logger.info(f"--- Calculating correlations using '{method}' method ---")
            
            # IMPROVEMENT: Added progress bars for correlation calculations
            logger.info("Processing tumor data... (this may take several minutes)")
            start_time = time.time()
            
            # Progress bar for tumor correlation
            with tqdm(total=1, desc="Tumor correlation", bar_format='{l_bar}{bar}| {elapsed} elapsed') as pbar:
                tumor_corr_matrix = tumor_df_T.corr(method=method)
                pbar.update(1)
            
            tumor_time = time.time() - start_time
            logger.info(f"Tumor correlation calculated in {tumor_time:.2f} seconds.")

            logger.info("Processing normal data...")
            start_time = time.time()
            
            # Progress bar for normal correlation
            with tqdm(total=1, desc="Normal correlation", bar_format='{l_bar}{bar}| {elapsed} elapsed') as pbar:
                normal_corr_matrix = normal_df_T.corr(method=method)
                pbar.update(1)
            
            normal_time = time.time() - start_time
            logger.info(f"Normal correlation calculated in {normal_time:.2f} seconds.")
            logger.info("")

            logger.info(f"Saving '{method}' correlation matrices to compressed .npz format...")
            
            tumor_output_path = matrices_dir / f"tumor_corr_{method}.npz"
            normal_output_path = matrices_dir / f"normal_corr_{method}.npz"
            
            # NPZ is an efficient format for storing large NumPy arrays. We save both the
            # matrix values and the gene names that serve as the index/columns.
            np.savez_compressed(tumor_output_path, matrix=tumor_corr_matrix.values, genes=tumor_corr_matrix.index)
            np.savez_compressed(normal_output_path, matrix=normal_corr_matrix.values, genes=normal_corr_matrix.index)
            
            # IMPROVEMENT: Report relative paths for cleaner logs.
            logger.info(f"Successfully saved tumor matrix to: {get_relative_path(tumor_output_path)}")
            logger.info(f"Successfully saved normal matrix to: {get_relative_path(normal_output_path)}")
            
            # Collect paths and stats for the summary file.
            summary_stats['outputs'][f"tumor_{method}_matrix"] = tumor_output_path
            summary_stats['outputs'][f"normal_{method}_matrix"] = normal_output_path
            
            # Calculate stats on the upper triangle of the matrix to avoid self-correlation (diagonal = 1).
            tumor_vals = tumor_corr_matrix.values[np.triu_indices_from(tumor_corr_matrix, k=1)]
            normal_vals = normal_corr_matrix.values[np.triu_indices_from(normal_corr_matrix, k=1)]
            
            summary_stats['matrix_stats'][f"tumor_{method}"] = {
                'shape': list(tumor_corr_matrix.shape),
                'mean': float(np.mean(tumor_vals)),
                'std': float(np.std(tumor_vals)),
                'min': float(np.min(tumor_vals)),
                'max': float(np.max(tumor_vals)),
                'median': float(np.median(tumor_vals))
            }
            summary_stats['matrix_stats'][f"normal_{method}"] = {
                'shape': list(normal_corr_matrix.shape),
                'mean': float(np.mean(normal_vals)),
                'std': float(np.std(normal_vals)),
                'min': float(np.min(normal_vals)),
                'max': float(np.max(normal_vals)),
                'median': float(np.median(normal_vals))
            }
            logger.info("")

    except FileNotFoundError as e:
        logger.error(f"ERROR: Input file not found. Have you run the preprocessing script?\n{e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        # --- 4. Save Summary JSON ---
        summary_stats['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        summary_path = OUTPUT_DIR / "01_a_result_info.json"
        create_summary_json(summary_stats, summary_path, PROJECT_ROOT)
        logger.info(f"Saved summary stats to: {get_relative_path(summary_path)}")
        logger.info("-" * 50)
        logger.info("Script finished.")

if __name__ == "__main__":
    main()