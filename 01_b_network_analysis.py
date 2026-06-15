"""
01_b_network_analysis.py

Script Purpose:
This script loads correlation matrices from 01_a, builds undirected NetworkX graphs for tumor/normal tissues at configurable thresholds (absolute correlation > threshold), computes graph metrics (nodes, edges, density, average clustering coefficient), and saves networks in GML (human-readable, tool-compatible) and Pickle (fast Python loading) formats. It generates a global metrics JSON for comparison across methods/thresholds and a summary JSON with run details, emphasizing primary networks (e.g., Spearman at primary_threshold) for downstream analysis in cancer gene co-expression networks.

Summary Logic:
1. Load config, verify/load NPZ correlation matrices (tumor/normal per method); log shapes/pairs.
2. For each method/threshold: Build graphs (upper triangle edges > threshold) with progress; compute metrics (density, clustering with per-node progress/estimates).
3. Save all networks (GML/Pickle) in subdirs; mark/track primary; aggregate metrics.
4. Save global_metrics_comparison.json (per-method/threshold stats) and 01_b_result_info.json (run summary with relative paths).
5. Use tqdm for all long ops; time estimates for large networks; final log of saved files.

Key Features:
- Configurable: Methods/thresholds via config['network_analysis']; primary method/threshold highlighted.
- Efficient: Upper-triangle only (no self-loops/duplicates); compressed saves; progress with rates/postfix.
- Metrics: Density (edges/max possible), avg clustering (local triangle density per node).
- Formats: GML for visualization (Cytoscape/Gephi), Pickle for Python (nx.load).
- Logging: Detailed file, simple console; estimates/warnings for large graphs (>1M edges).

Dependencies: See imports below. Assumes utils.config, utils.file; inputs from 01_a_build_correlation_matrices.py; requires networkx.
"""

import pandas as pd
import numpy as np
import networkx as nx
import json
import logging
import time
import pickle
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
    log_file = log_dir / 'network_analysis.log'
    
    file_handler = logging.FileHandler(log_file, mode='w')
    file_formatter = logging.Formatter(config['logging']['format'])
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

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


def build_network(corr_matrix, genes, threshold, method, logger):
    """
    Builds a NetworkX graph from a correlation matrix and a given threshold.
    
    Adds nodes first; iterates upper triangle (triu_indices) for edges where abs(corr) >= threshold;
    weights edges with corr value; uses tqdm with postfix for edges count.
    """
    G = nx.Graph()
    G.add_nodes_from(genes)
    
    # Get row and column indices of the upper triangle of the matrix to avoid duplicate checks and self-loops.
    rows, cols = np.triu_indices_from(corr_matrix, k=1)
    total_pairs = len(rows)
    
    logger.info(f"Checking {total_pairs:,} potential gene pairs...")
    
    # Use tqdm for a progress bar with detailed information
    progress_desc = f"Building network ({method}, thr={threshold})"
    
    # Count edges added for progress reporting
    edges_added = 0
    
    # Create a more informative progress bar
    with tqdm(total=total_pairs, desc=progress_desc, 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
              postfix=f"Edges: 0") as pbar:
        
        for i, j in zip(rows, cols):
            corr_value = corr_matrix[i, j]
            if abs(corr_value) >= threshold:
                G.add_edge(genes[i], genes[j], weight=corr_value)
                edges_added += 1
                
                # Update postfix every 1000 edges to reduce overhead
                if edges_added % 1000 == 0:
                    pbar.set_postfix_str(f"Edges: {edges_added:,}")
            
            pbar.update(1)
    
    logger.info(f"Network built with {edges_added:,} edges ({(edges_added/total_pairs)*100:.2f}% of possible pairs)")
    return G


def calculate_clustering_with_clean_progress(network, network_type, logger):
    """Calculate clustering with clean, single progress bar and accurate time estimates."""
    logger.info(f"Calculating clustering coefficient for {network_type} network...")
    
    n_nodes = network.number_of_nodes()
    n_edges = network.number_of_edges()
    
    logger.info(f"  Network has {n_nodes:,} nodes and {n_edges:,} edges")
    
    # Provide time estimate based on network size
    # Empirical formula based on actual runtime data:
    # - Small networks (<1M edges): ~0.01 sec/node
    # - Medium networks (1-5M edges): ~0.05 sec/node  
    # - Large networks (>5M edges): ~0.5-3 sec/node (scales with avg_degree)
    
    avg_degree = (2 * n_edges) / n_nodes if n_nodes > 0 else 0
    
    # More accurate empirical estimate based on average degree
    if avg_degree < 100:
        sec_per_node = 0.01
    elif avg_degree < 500:
        sec_per_node = 0.05
    elif avg_degree < 1000:
        sec_per_node = 0.2
    elif avg_degree < 2000:
        sec_per_node = 1.0
    else:
        # For very dense networks: time scales roughly linearly with degree
        sec_per_node = avg_degree / 2000
    
    estimated_seconds = n_nodes * sec_per_node
    
    if estimated_seconds > 7200:  # > 2 hours
        estimated_hours = estimated_seconds / 3600
        logger.info(f"  ⏰ ESTIMATE: This may take {estimated_hours:.1f} hours (long run)")
    elif estimated_seconds > 300:  # > 5 minutes
        estimated_minutes = estimated_seconds / 60
        logger.info(f"  ⏰ ESTIMATE: This may take {estimated_minutes:.1f} minutes")
    else:
        logger.info(f"  ⏰ ESTIMATE: This should complete quickly (< 5 minutes)")
    
    # Use a single, clean progress bar without duplicate logging
    with tqdm(total=n_nodes, desc=f"Clustering {network_type}", 
             bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] | Speed: {rate_fmt} | Current: {postfix}',
             postfix="Starting...") as pbar:
        
        clustering_values = []
        nodes_processed = 0
        start_time = time.time()
        
        nodes_list = list(network.nodes())
        
        for i, node in enumerate(nodes_list):
            try:
                # Calculate clustering for this node
                clustering_val = nx.clustering(network, node)
                clustering_values.append(clustering_val)
                
                # Update progress after every node
                nodes_processed += 1
                pbar.update(1)
                
                # Update postfix with current node (truncated if long)
                current_node = str(node)
                if len(current_node) > 20:
                    current_node = current_node[:17] + "..."
                pbar.set_postfix_str(current_node)
                
            except Exception as e:
                logger.warning(f"  Error calculating clustering for node {node}: {e}")
                clustering_values.append(0.0)
                pbar.update(1)
        
        # Final calculation
        avg_clustering = np.mean(clustering_values) if clustering_values else 0.0
        total_time = time.time() - start_time
        
        logger.info(f"  ✅ COMPLETED: Average clustering coefficient: {avg_clustering:.6f}")
        logger.info(f"  ⏱️ Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        logger.info(f"  📊 Final stats: {nodes_processed:,} nodes processed | {nodes_processed/total_time:.1f} nodes/sec")
        
        return avg_clustering


def create_summary_json(summary_data, output_path, project_root):
    """Creates a JSON file with summary statistics of the run."""
    # Convert all Path objects to clean, relative string paths
    for section in ['inputs', 'outputs']:
        if section in summary_data:
            for key, value in summary_data[section].items():
                if isinstance(value, Path):
                    summary_data[section][key] = str(value.relative_to(project_root))
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (Path, dict)):
                            if isinstance(sub_value, Path):
                                summary_data[section][key][sub_key] = str(sub_value.relative_to(project_root))
                            elif isinstance(sub_value, dict):
                                for sub_sub_key, sub_sub_value in sub_value.items():
                                    if isinstance(sub_sub_value, Path):
                                        summary_data[section][key][sub_key][sub_sub_key] = str(sub_sub_value.relative_to(project_root))
    
    # Sort the entire summary data structure for consistent output
    sorted_summary_data = sort_nested_dict(summary_data)

    with open(output_path, 'w') as f:
        json.dump(sorted_summary_data, f, indent=4)


def main():
    """Main execution function: Orchestrates loading, network building, metrics, saving, and summarization."""
    config = load_config()
    
    # Set up paths using the established pattern
    PROJECT_ROOT = Path(config['paths']['project_root'])
    INPUT_CORRELATION = Path(config['paths']['correlation_matrices'])  # Input from 01_a
    OUTPUT_DIR = get_auto_output_path(__file__, PROJECT_ROOT)  # Auto-generated output
    
    net_params = config['network_analysis']
    
    # FIXED: Read primary_threshold from config with fallback to default
    primary_threshold = net_params.get('primary_threshold', 0.7)

    logger = setup_logging(config, OUTPUT_DIR)
    
    logger.info(f"01_a input directory: {get_relative_path(INPUT_CORRELATION)}")
    logger.info(f"01_b output directory: {get_relative_path(OUTPUT_DIR)}")
    logger.info("Starting script: 01_b_network_analysis.py")
    logger.info("-" * 50)

    summary_stats = {
        "script": "01_b_network_analysis",
        "format_notes": {
            "pickle": "Pickle format preserves exact NetworkX graph objects with all attributes. Use for fast loading in Python analysis workflows.",
            "gml": "GML format is human-readable and compatible with external tools (Cytoscape, Gephi). Use for archival, sharing, and visualization."
        },
        "start_time": time.strftime('%Y-%m-%d %H:%M:%S'),
        "parameters": {
            "correlation_methods": net_params['correlation_methods'],
            "correlation_thresholds": net_params['correlation_thresholds'],
            "primary_threshold": primary_threshold
        },
        "inputs": {},
        "outputs": {
            "primary_networks": {},
            "all_networks": {}
        },
        "global_metrics": {}
    }
    
    try:
        # Verify input files exist
        methods = net_params.get('correlation_methods', ['spearman'])
        required_files = []
        for method in methods:
            required_files.extend([
                INPUT_CORRELATION / 'matrices' / f"tumor_corr_{method}.npz",
                INPUT_CORRELATION / 'matrices' / f"normal_corr_{method}.npz"
            ])
        
        for file_path in required_files:
            if not file_path.exists():
                logger.error(f"Required input file not found: {get_relative_path(file_path)}")
                logger.error("Please run 01_a_build_correlation_matrices.py first")
                return

        # Create subdirectories for different formats
        gml_dir = ensure_dir(OUTPUT_DIR / "gml")
        pickle_dir = ensure_dir(OUTPUT_DIR / "pickle")
        
        # Loop through each correlation method calculated in the previous script.
        primary_method = net_params['primary_correlation_method']
        
        for method in methods:
            logger.info(f"\n===== Processing Correlation Method: {method.upper()} =====")
            
            # --- 1. Load Correlation Matrices ---
            tumor_corr_path = INPUT_CORRELATION / 'matrices' / f"tumor_corr_{method}.npz"
            normal_corr_path = INPUT_CORRELATION / 'matrices' / f"normal_corr_{method}.npz"
            
            summary_stats['inputs'][f"tumor_{method}_matrix"] = tumor_corr_path
            summary_stats['inputs'][f"normal_{method}_matrix"] = normal_corr_path

            # FIXED: Add allow_pickle=True for loading gene name arrays
            with np.load(tumor_corr_path, allow_pickle=True) as data:
                tumor_corr_matrix = data['matrix']
                tumor_genes = data['genes']
            with np.load(normal_corr_path, allow_pickle=True) as data:
                normal_corr_matrix = data['matrix']
                normal_genes = data['genes']
                
            logger.info(f"Loaded '{method}' correlation matrices.")
            logger.info(f"Tumor matrix shape: {tumor_corr_matrix.shape}")
            logger.info(f"Normal matrix shape: {normal_corr_matrix.shape}")
            logger.info(f"Number of genes: {len(tumor_genes)}")
            total_pairs = (len(tumor_genes) * (len(tumor_genes) - 1)) // 2
            logger.info(f"Total possible gene pairs: {total_pairs:,}")
            logger.info("")

            # --- 2. Build Networks and Calculate Metrics for each threshold ---
            thresholds = net_params['correlation_thresholds']
            logger.info(f"Building networks for thresholds: {thresholds}")
            
            method_metrics = {}
            for threshold in thresholds:
                logger.info(f"\n--- Processing threshold: {threshold} ---")
                
                # Build networks with progress bars - pass logger for status updates
                logger.info("Building tumor network...")
                start_time = time.time()
                tumor_network = build_network(tumor_corr_matrix, tumor_genes, threshold, method, logger)
                tumor_time = time.time() - start_time
                logger.info(f"✓ Tumor network built in {tumor_time:.2f} seconds")
                
                logger.info("")
                
                logger.info("Building normal network...")
                start_time = time.time()
                normal_network = build_network(normal_corr_matrix, normal_genes, threshold, method, logger)
                normal_time = time.time() - start_time
                logger.info(f"✓ Normal network built in {normal_time:.2f} seconds")
                
                logger.info("")
                
                # Calculate key graph theory metrics
                logger.info("Calculating network metrics...")
                with tqdm(total=4, desc="Basic metrics", bar_format='{l_bar}{bar}| {elapsed} elapsed') as metric_pbar:
                    # Tumor metrics
                    tumor_nodes = tumor_network.number_of_nodes()
                    metric_pbar.update(1)
                    tumor_edges = tumor_network.number_of_edges()
                    metric_pbar.update(1)
                    # Normal metrics
                    normal_nodes = normal_network.number_of_nodes()
                    metric_pbar.update(1)
                    normal_edges = normal_network.number_of_edges()
                    metric_pbar.update(1)
                
                # Calculate density
                tumor_density = nx.density(tumor_network)
                normal_density = nx.density(normal_network)
                
                # Calculate clustering coefficients with clean progress tracking
                logger.info("Calculating clustering coefficients...")
                tumor_clustering = calculate_clustering_with_clean_progress(tumor_network, "tumor", logger)
                logger.info("")
                normal_clustering = calculate_clustering_with_clean_progress(normal_network, "normal", logger)
                logger.info("")
                
                metrics = {
                    'tumor': {
                        'nodes': tumor_nodes,
                        'edges': tumor_edges,
                        'density': tumor_density,
                        'avg_clustering_coeff': tumor_clustering
                    },
                    'normal': {
                        'nodes': normal_nodes,
                        'edges': normal_edges,
                        'density': normal_density,
                        'avg_clustering_coeff': normal_clustering
                    }
                }
                method_metrics[str(threshold)] = metrics
                
                logger.info(f"✓ Tumor Network: {metrics['tumor']['nodes']} nodes, {metrics['tumor']['edges']} edges, density: {metrics['tumor']['density']:.6f}")
                logger.info(f"✓ Normal Network: {metrics['normal']['nodes']} nodes, {metrics['normal']['edges']} edges, density: {metrics['normal']['density']:.6f}")

                # Save ALL networks in both formats
                tumor_gml_path = gml_dir / f"tumor_network_{method}_{threshold}.gml"
                normal_gml_path = gml_dir / f"normal_network_{method}_{threshold}.gml"
                tumor_pkl_path = pickle_dir / f"tumor_network_{method}_{threshold}.pkl"
                normal_pkl_path = pickle_dir / f"normal_network_{method}_{threshold}.pkl"
                
                logger.info("Saving tumor network (GML)...")
                with tqdm(total=1, desc="Saving GML", bar_format='{l_bar}{bar}| {elapsed} elapsed') as save_pbar:
                    nx.write_gml(tumor_network, str(tumor_gml_path))
                    save_pbar.update(1)
                logger.info(f"✓ Saved tumor network to: {get_relative_path(tumor_gml_path)}")
                
                logger.info("Saving tumor network (Pickle)...")
                with tqdm(total=1, desc="Saving Pickle", bar_format='{l_bar}{bar}| {elapsed} elapsed') as save_pbar:
                    with open(tumor_pkl_path, 'wb') as f:
                        pickle.dump(tumor_network, f, protocol=pickle.HIGHEST_PROTOCOL)
                    save_pbar.update(1)
                logger.info(f"✓ Saved tumor network to: {get_relative_path(tumor_pkl_path)}")
                
                logger.info("Saving normal network (GML)...")
                with tqdm(total=1, desc="Saving GML", bar_format='{l_bar}{bar}| {elapsed} elapsed') as save_pbar:
                    nx.write_gml(normal_network, str(normal_gml_path))
                    save_pbar.update(1)
                logger.info(f"✓ Saved normal network to: {get_relative_path(normal_gml_path)}")
                
                logger.info("Saving normal network (Pickle)...")
                with tqdm(total=1, desc="Saving Pickle", bar_format='{l_bar}{bar}| {elapsed} elapsed') as save_pbar:
                    with open(normal_pkl_path, 'wb') as f:
                        pickle.dump(normal_network, f, protocol=pickle.HIGHEST_PROTOCOL)
                    save_pbar.update(1)
                logger.info(f"✓ Saved normal network to: {get_relative_path(normal_pkl_path)}")
                
                # Track all saved networks
                network_key = f"{method}_{threshold}"
                summary_stats['outputs']['all_networks'][network_key] = {
                    'tumor_gml': tumor_gml_path,
                    'tumor_pkl': tumor_pkl_path,
                    'normal_gml': normal_gml_path,
                    'normal_pkl': normal_pkl_path
                }
                
                # Mark primary networks
                if method == primary_method and threshold == primary_threshold:
                    summary_stats['outputs']['primary_networks'] = {
                        'tumor_gml': tumor_gml_path,
                        'tumor_pkl': tumor_pkl_path,
                        'normal_gml': normal_gml_path,
                        'normal_pkl': normal_pkl_path
                    }
                    logger.info("⭐ Marked as primary network for downstream analysis")
            
            summary_stats['global_metrics'][method] = method_metrics

        # --- 3. Save Summary Files ---
        metrics_path = OUTPUT_DIR / "global_metrics_comparison.json"
        logger.info("")
        logger.info("Saving global metrics...")
        with tqdm(total=1, desc="Writing JSON", bar_format='{l_bar}{bar}| {elapsed} elapsed') as json_pbar:
            # Sort the global metrics before saving
            sorted_global_metrics = sort_nested_dict(summary_stats['global_metrics'])
            with open(metrics_path, 'w') as f:
                json.dump(sorted_global_metrics, f, indent=4)
            json_pbar.update(1)
        summary_stats['outputs']['global_metrics_comparison_json'] = metrics_path
        logger.info(f"✓ Saved global network metrics to: {get_relative_path(metrics_path)}")

    except FileNotFoundError as e:
        logger.error(f"ERROR: Input .npz file not found. Have you run script 01_a first?\n{e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        summary_stats['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        summary_path = OUTPUT_DIR / "01_b_result_info.json"
        logger.info("")
        logger.info("Saving final summary...")
        with tqdm(total=1, desc="Writing summary", bar_format='{l_bar}{bar}| {elapsed} elapsed') as summary_pbar:
            create_summary_json(summary_stats, summary_path, PROJECT_ROOT)
            summary_pbar.update(1)
        logger.info(f"✓ Saved summary stats to: {get_relative_path(summary_path)}")
        
        # Print summary of saved files
        logger.info("\n" + "="*50)
        logger.info("NETWORK FILES SAVED:")
        logger.info("="*50)
        methods = net_params['correlation_methods']
        thresholds = net_params['correlation_thresholds']
        for method in methods:
            for threshold in thresholds:
                primary_mark = " (PRIMARY)" if method == primary_method and threshold == primary_threshold else ""
                logger.info(f"• GML: tumor_network_{method}_{threshold}.gml{primary_mark}")
                logger.info(f"• GML: normal_network_{method}_{threshold}.gml{primary_mark}")
                logger.info(f"• PKL: tumor_network_{method}_{threshold}.pkl{primary_mark}")
                logger.info(f"• PKL: normal_network_{method}_{threshold}.pkl{primary_mark}")
        logger.info("="*50)
        logger.info("-" * 50)
        logger.info("Script finished.")

if __name__ == "__main__":
    main()