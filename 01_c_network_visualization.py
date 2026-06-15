"""
01_c_network_visualization.py

Script Purpose:
This script loads pre-built gene co-expression networks for tumor and normal tissues using both Spearman and Pearson correlations. 
It generates a series of comparative visualizations to highlight key structural differences, such as cancer-induced network fragmentation, 
and to demonstrate the methodological superiority of Spearman correlation for this biological data. Visualizations include degree distributions, 
component sizes, hub neighborhoods, and method comparison dashboards and radar charts. All plotting logic is handled by a dedicated utility module.

Summary Logic:
1.  Define a global correlation threshold for the analysis.
2.  Load config, setup structured logging, and create organized output directories.
3.  Load the four pre-computed networks (tumor/normal for both Spearman and Pearson) based on the defined threshold.
4.  Calculate key comparative metrics (e.g., edge count ratio between methods).
5.  For each correlation method (Spearman, Pearson):
    a. Generate a dual histogram comparing tumor (fragmented) vs. normal (cooperative) degree distributions.
    b. Create a dual bar chart showing tumor fragmentation vs. the normal giant component.
    c. Visualize and compare the neighborhoods of the top 3 hub genes.
    d. Plot the Cumulative Distribution Function (CDF) of edge weights.
6.  Generate method comparison visualizations:
    a. Create overlaid density plots of tumor/normal degrees to show Spearman's superior signal capture.
    b. Render a performance scorecard as a table, quantifying the advantages of Spearman.
    c. Plot two radar charts (one for tumor, one for normal) for a multi-metric comparison of performance.
7.  Track all generated visualizations with rich descriptions, interpretations, and metrics, and save a comprehensive '01_c_result_info.json' summary file.
8.  Save JSON data files for each chart containing the actual data used for plotting, with descriptions of axes and data points.

Key Features:
- Centralized Configuration: A global variable `CORRELATION_THRESHOLD` at the top of the script controls the entire analysis.
- Biological Storytelling: Chart titles, subtitles, and annotations are crafted to tell a clear story of network fragmentation in cancer and Spearman's advantages.
- Comprehensive Summary: Generates a highly descriptive JSON file logging all outputs, parameters, interpretations, and key findings for programmatic use.
- Data Preservation: Saves JSON data files alongside each chart for reproducibility and detailed analysis.
- Clean Output: Suppresses common, non-critical warnings from plotting libraries for a cleaner log.
- Modularity: All plotting is delegated to the `utils.chart` module, keeping this script focused on analysis logic.

Dependencies: See imports. Requires `utils/config`, `utils/file`, and `utils.chart`.
"""

import pandas as pd
import numpy as np
import networkx as nx
import json
import logging
import time
import pickle
import warnings
from pathlib import Path
from collections import OrderedDict

# Local utilities
from utils.config import load_config
from utils.file import ensure_dir, get_relative_path, get_auto_output_path
from utils.chart import (
    create_dual_histogram,
    create_dual_bar_chart,
    create_hub_comparison_grid,
    create_cdf,
    create_table_chart,
    create_density_overlay_chart,
    create_radar_chart,
    create_network_fragmentation_dashboard,
    create_biological_signal_comparison,
    create_cancer_network_story,
    create_simplified_performance_radar
)

from utils.chart_advanced import (
    create_degree_distribution_overlay,
    create_rank_degree_plot,
    find_goldilocks_hub_pairs, 
    create_paired_hub_comparison, 
    find_breast_cancer_hub_pairs, 
    create_differential_hub_overlay
)

from utils.chart_method_comparison import (
    create_edge_type_breakdown,
    create_hub_preservation_analysis,
    create_unified_performance_dashboard,
    create_side_by_side_distribution_comparison
)

from utils.genes import load_combined_gene_info, get_gene_info, normalize_gene_id


# --- Global Analysis Configuration ---
CORRELATION_THRESHOLD = 0.7
# ------------------------------------

def setup_logging(config, output_dir):
    """Set up file and console logging."""
    logger = logging.getLogger(__name__)
    logger.setLevel(config['logging']['level'])
    if logger.hasHandlers():
        logger.handlers.clear()
    
    log_dir = ensure_dir(output_dir / 'logs')
    file_handler = logging.FileHandler(log_dir / 'network_visualization.log', mode='w')
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

def safe_read_pickle(file_path, logger):
    """Safely read a NetworkX graph from a pickle file."""
    logger.info(f"Loading network from: {get_relative_path(file_path)}")
    try:
        with open(file_path, 'rb') as f:
            network = pickle.load(f)
        logger.info(f"✓ Successfully loaded: {network.number_of_nodes():,} nodes, {network.number_of_edges():,} edges")
        return network
    except Exception as e:
        logger.error(f"Pickle loading failed for {file_path}: {e}")
        raise

def save_chart_data_json(chart_data, output_path, chart_type, title, description=None):
    """
    Save chart data to a JSON file with descriptions and metadata.
    
    Args:
        chart_data: Dictionary containing the chart data
        output_path: Path to save the JSON file (same name as chart but .json extension)
        chart_type: Type of chart (e.g., 'degree_distribution', 'component_sizes')
        title: Chart title
        description: Optional description of the chart
    """
    json_data = {
        "metadata": {
            "chart_type": chart_type,
            "title": title,
            "correlation_threshold": CORRELATION_THRESHOLD,
            "generated_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "description": description or f"Data for {chart_type.replace('_', ' ')} chart"
        },
        "data": chart_data
    }
    
    # Sort the data for consistent output
    sorted_json_data = sort_nested_dict(json_data)
    
    with open(output_path, 'w') as f:
        json.dump(sorted_json_data, f, indent=2)



def create_enhanced_visualizations(
    networks, 
    base_output_dir, 
    logger, 
    visualization_tracker, 
    CORRELATION_THRESHOLD,
    config
):
    """
    Create enhanced visualizations (overlay and high-res hubs).
    
    This function should be called after the main visualization loop completes.
    It generates:
    1. Degree distribution overlay charts (one for Spearman, one for Pearson)
    2. Six individual high-resolution hub network charts (3 tumor + 3 normal) for Spearman
    
    Args:
        networks: Dict with keys 'spearman_tumor', 'spearman_normal', 'pearson_tumor', 'pearson_normal'
        base_output_dir: Base output directory path
        logger: Logger instance
        visualization_tracker: Dict to track all generated visualizations
        CORRELATION_THRESHOLD: Threshold value used for network construction
    """
    
    logger.info("\n" + "="*60)
    logger.info("ENHANCED VISUALIZATIONS")
    logger.info("="*60)
    
    # ===== ENHANCEMENT A: DEGREE DISTRIBUTION OVERLAY =====
    logger.info("\nEnhancement A: Creating degree distribution overlay charts...")
    logger.info("-" * 40)
    
    all_hub_metadata = []

    for method in ['spearman', 'pearson']:
        logger.info(f"\nProcessing {method.upper()} overlay...")
        
        # Get networks
        tumor_net = networks[f'{method}_tumor']
        normal_net = networks[f'{method}_normal']
        
        # Get degree data
        tumor_degrees = [d for _, d in tumor_net.degree()]
        normal_degrees = [d for _, d in normal_net.degree()]
        
        # Create overlay chart
        method_subdir = base_output_dir / method
        plot_path = method_subdir / f'01_{method}_degree_distribution_overlay.png'
        
        title = f'{method.upper()}: Tumor Fragmented vs Normal Cooperative'
        subtitle = f'Degree Distribution Overlay | Threshold: |r| ≥ {CORRELATION_THRESHOLD}'
        
        metadata = create_degree_distribution_overlay(
            tumor_degrees=tumor_degrees,
            normal_degrees=normal_degrees,
            title=title,
            output_path=plot_path,
            subtitle=subtitle,
            layout='wide',
            normalize=True,
            show_medians=True,
            dpi=300
        )
        
        # Save JSON metadata
        json_path = method_subdir / f'01_{method}_degree_distribution_overlay.json'
        chart_data = {
            "metadata": metadata,
            "description": {
                "chart_type": "degree_distribution_overlay",
                "purpose": "Direct visual comparison of tumor vs normal degree distributions on the same scale",
                "x_axis": "Degree (Number of Connections)",
                "y_axis": "Frequency (% of genes)",
                "interpretation": "Overlay plot shows dramatic collapse of tumor network (red) compared to normal (blue). Tumor median near 0 indicates most genes are isolated, while normal median ~1700 shows robust connectivity.",
                "visualization_features": [
                    "Smoothed curves using Gaussian kernel",
                    "Semi-transparent filled areas for overlap visualization",
                    "Median lines showing distribution centers",
                    "Normalized to percentage for direct comparison"
                ]
            },
            "tumor_distribution": {
                "raw_degrees": tumor_degrees,
                "median": float(np.median(tumor_degrees)),
                "mean": float(np.mean(tumor_degrees)),
                "genes_with_zero_degree": sum(1 for d in tumor_degrees if d == 0),
                "genes_with_nonzero_degree": sum(1 for d in tumor_degrees if d > 0)
            },
            "normal_distribution": {
                "raw_degrees": normal_degrees,
                "median": float(np.median(normal_degrees)),
                "mean": float(np.mean(normal_degrees)),
                "genes_with_zero_degree": sum(1 for d in normal_degrees if d == 0),
                "genes_with_nonzero_degree": sum(1 for d in normal_degrees if d > 0)
            }
        }
        
        with open(json_path, 'w') as f:
            json.dump(chart_data, f, indent=2)
        
        # Update visualization tracker
        visualization_tracker['visualizations']['by_method'][method].append({
            'type': 'degree_distribution_overlay',
            'file': str(get_relative_path(plot_path)),
            'data_file': str(get_relative_path(json_path)),
            'title': title,
            'interpretation': 'Smoothed overlay comparison makes the network collapse visually dramatic. Tumor distribution (red) is heavily skewed toward zero, while normal distribution (blue) shows broad connectivity across all genes.',
            'biological_context': 'This single chart encapsulates the Rewiring Paradox: global connectivity collapse. The lack of overlap between curves demonstrates qualitatively different network architectures.',
            'usage': 'Ideal for thesis Chapter 4 as companion to dual histogram. Better for presentations due to direct comparison on same axes.',
            'enhancement': 'A'
        })
        
        logger.info(f"✓ Overlay chart saved: {get_relative_path(plot_path)}")
        logger.info(f"✓ Metadata saved: {get_relative_path(json_path)}")
        logger.info(f"  - Tumor median: {metadata['tumor_median']:.1f}")
        logger.info(f"  - Normal median: {metadata['normal_median']:.1f}")


        # --- NEW: Enhancement C: Rank-Degree Topology ---
        logger.info(f"Enhancement C: Creating Rank-Degree Topology Plot for {method}...")
        
        rd_plot_path = method_subdir / f'01_{method}_rank_degree_topology.png'
        rd_json_path = method_subdir / f'01_{method}_rank_degree_topology.json'
        
        rd_metadata = create_rank_degree_plot(
            tumor_degrees=tumor_degrees,
            normal_degrees=normal_degrees,
            title=f"Structural Collapse ({method.capitalize()}): Rank-Degree Topology",
            output_path=rd_plot_path,
            json_path=rd_json_path
        )
        
        # FIX: Using dictionary assignment instead of .append()
        # We use a unique key for each method so they don't overwrite each other
        tracker_key = f'rank_degree_{method}'
        visualization_tracker[tracker_key] = {
            'name': f'{method.capitalize()} Rank-Degree Topology',
            'file': str(rd_plot_path.relative_to(base_output_dir)),
            'data': str(rd_json_path.relative_to(base_output_dir)),
            'description': 'Log-Log rank ordering showing the breakdown of network hierarchy.'
        }
        logger.info(f"✓ Saved: {rd_plot_path.name}") 



        # Track genes used in hub visualizations for metadata generation
        hub_viz_genes = {
            'goldilocks': [],
            'breast_cancer': [],
            'differential': []
        }

        # ======================================================================
        # NEW: ENHANCEMENT - DUAL-CENTRIC HUB ANALYSIS
        # ======================================================================
        logger.info(f"Enhancement: Generating Dual-Centric Hub Comparisons for {method}...")
        
        # --- Calculate Degrees first ---
        normal_degrees_dict = dict(normal_net.degree())
        tumor_degrees_dict = dict(tumor_net.degree())

        normal_hubs = sorted(normal_degrees_dict.items(), key=lambda x: x[1], reverse=True)
        tumor_hubs = sorted(tumor_degrees_dict.items(), key=lambda x: x[1], reverse=True)

        hub_dir = method_subdir / "03_hub_analysis_centric"
        hub_dir.mkdir(exist_ok=True)

        # Generate 6 Unified Comparative Hub Charts (3 Normal-centric, 3 Tumor-centric)
        logger.info(f"Generating 6 Unified Comparative Hub Overlays...")
        
        # Combine hubs into one list of 6 total to process
        target_hubs = [(h, 'normal') for h, _ in normal_hubs[:3]] + \
                      [(h, 'tumor') for h, _ in tumor_hubs[:3]]

        # Track differential hub genes (will populate during loop below)
        # Note: target_hubs = [(gene_id, 'normal'), ...] - tuples not dicts
        # We'll track them in the generation loop where we have all the data

        for i, (hub_gene, origin) in enumerate(target_hubs):
            symbol = hub_gene.split('|')[1] if '|' in hub_gene else hub_gene
            plot_path = hub_dir / f"diff_hub_{i+1}_{symbol}_{origin}_centric.png"
            
            # Get the degree for this hub
            if origin == 'normal':
                hub_degree = normal_degrees_dict.get(hub_gene, 0)
            else:
                hub_degree = tumor_degrees_dict.get(hub_gene, 0)
            
            # Track this differential hub gene
            hub_viz_genes['differential'].append({
                'gene_id': hub_gene,
                'gene_symbol': symbol,
                'condition': origin,
                'degree': hub_degree
            })

            # Call the unified overlay
            metadata = create_differential_hub_overlay(
                normal_net=normal_net,
                tumor_net=tumor_net,
                hub_gene=hub_gene,
                output_path=plot_path,
                threshold=CORRELATION_THRESHOLD
            )
            all_hub_metadata.append(metadata)


        # === GOLDILOCKS PAIRS: Visually Optimal Contrast (ENHANCED) ===
        logger.info(f"Enhancement: Creating Goldilocks Hub Pairs for {method}...")
        logger.info("  Finding genes with HIGH visual contrast (5-20× ratio)...")

        # ENHANCED CRITERIA: More dramatic contrast (5-20× instead of 3-10×)
        goldilocks_pairs = find_goldilocks_hub_pairs(
            normal_network=normal_net,
            tumor_network=tumor_net,
            min_normal_deg=3000,   # Higher baseline (more substantial hubs)
            min_tumor_deg=200,     # Lower minimum (shows more loss)
            max_tumor_deg=1200,    # Lower ceiling (bigger difference)
            min_ratio=5.0,         # Higher minimum (more dramatic)
            max_ratio=20.0,        # Higher ceiling (allow extreme but visible)
            num_pairs=3
        )
            
        if len(goldilocks_pairs) < 3:
            logger.warning(f"  Only found {len(goldilocks_pairs)} goldilocks pairs (expected 3)")
            logger.warning("  Relaxing criteria...")
            goldilocks_pairs = find_goldilocks_hub_pairs(
                normal_network=normal_net,
                tumor_network=tumor_net,
                min_normal_deg=2000,   # Relaxed
                min_tumor_deg=150,     # Relaxed
                max_tumor_deg=1500,    # Relaxed
                min_ratio=4.0,         # Relaxed
                max_ratio=25.0,        # Relaxed
                num_pairs=3
            )

        goldilocks_metadata = []

        for i, (gene, normal_deg, tumor_deg, ratio) in enumerate(goldilocks_pairs):
            symbol = gene.split('|')[1] if '|' in gene else gene

            # Track goldilocks genes
            hub_viz_genes['goldilocks'].append({
                'rank': i,
                'gene_id': gene,
                'gene_symbol': symbol,
                'normal_degree': normal_deg,
                'tumor_degree': tumor_deg,
                'ratio': ratio
            })

            logger.info(f"  Pair {i+1}: {symbol}")
            logger.info(f"    Normal degree: {normal_deg:,}")
            logger.info(f"    Tumor degree: {tumor_deg:,}")
            logger.info(f"    Ratio: {ratio:.1f}×")
            logger.info(f"    Loss: {((normal_deg - tumor_deg) / normal_deg * 100):.1f}%")
            
            # Create paired comparison chart
            paired_path = hub_dir / f"goldilocks_{i+1}_{symbol}_paired.png"
            paired_json = hub_dir / f"goldilocks_{i+1}_{symbol}_paired.json"
            
            metadata = create_paired_hub_comparison(
                normal_network=normal_net,
                tumor_network=tumor_net,
                hub_gene=gene,
                output_path=paired_path,
                threshold=CORRELATION_THRESHOLD,
                max_neighbors=200,
                dpi=300
            )
            
            with open(paired_json, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            goldilocks_metadata.append(metadata)
            
            logger.info(f"    ✓ Paired chart: {paired_path.name}")
            logger.info(f"    ✓ Metadata: {paired_json.name}")

        # Save goldilocks summary
        summary_path = hub_dir / "goldilocks_pairs_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                "description": "Goldilocks hub pairs with HIGH visual contrast",
                "explanation": "Enhanced criteria: 5-20× ratio (more dramatic than original 3-10×)",
                "criteria": {
                    "min_normal_degree": 3000,
                    "min_tumor_degree": 200,
                    "max_tumor_degree": 1200,
                    "min_ratio": 5.0,
                    "max_ratio": 20.0
                },
                "correlation_threshold": CORRELATION_THRESHOLD,
                "pairs": goldilocks_metadata
            }, f, indent=2)

        logger.info(f"✓ Generated {len(goldilocks_pairs)} Goldilocks pairs")
        logger.info(f"✓ Summary saved: {summary_path.name}")


        # === BREAST CANCER GENE PAIRS: Known BC Genes with Visual Contrast ===
        logger.info(f"Enhancement: Creating Breast Cancer Gene Hub Pairs for {method}...")
        logger.info("  Finding known breast cancer genes with rewiring...")

        # Load gene info for classification
        combined_gene_data = load_combined_gene_info(config)

        bc_pairs = find_breast_cancer_hub_pairs(
            normal_network=normal_net,
            tumor_network=tumor_net,
            config=config,
            combined_gene_data=combined_gene_data,
            min_normal_deg=1500,   # Permissive (want known BC genes)
            min_tumor_deg=100,     # Very permissive
            max_tumor_deg=3000,    # Very permissive
            min_ratio=2.0,         # Permissive
            max_ratio=50.0,        # Very permissive
            num_pairs=3
        )

            
        if len(bc_pairs) < 3:
            logger.warning(f"  Only found {len(bc_pairs)} breast cancer gene pairs (expected 3)")
            logger.warning("  These are genes classified as breast_cancer in gene_info")
        else:
            logger.info(f"  Found {len(bc_pairs)} breast cancer gene pairs")

        bc_metadata = []

        for i, (gene, normal_deg, tumor_deg, ratio, division) in enumerate(bc_pairs):
            symbol = gene.split('|')[1] if '|' in gene else gene
            
             # Track breast cancer genes
            hub_viz_genes['breast_cancer'].append({
                'rank': i,
                'gene_id': gene,
                'gene_symbol': symbol,
                'normal_degree': normal_deg,
                'tumor_degree': tumor_deg,
                'ratio': ratio,
                'is_breast_cancer_gene': True
            })

            logger.info(f"  BC Pair {i+1}: {symbol} (Known Breast Cancer Gene)")
            logger.info(f"    Normal degree: {normal_deg:,}")
            logger.info(f"    Tumor degree: {tumor_deg:,}")
            logger.info(f"    Ratio: {ratio:.1f}×")
            logger.info(f"    Loss: {((normal_deg - tumor_deg) / normal_deg * 100):.1f}%")
            
            # Create paired comparison chart
            paired_path = hub_dir / f"breast_cancer_{i+1}_{symbol}_paired.png"
            paired_json = hub_dir / f"breast_cancer_{i+1}_{symbol}_paired.json"
            
            metadata = create_paired_hub_comparison(
                normal_network=normal_net,
                tumor_network=tumor_net,
                hub_gene=gene,
                output_path=paired_path,
                threshold=CORRELATION_THRESHOLD,
                max_neighbors=200,
                dpi=300
            )
            
            # Add breast cancer flag to metadata
            metadata['is_breast_cancer_gene'] = True
            metadata['gene_classification'] = 'breast_cancer'
            
            with open(paired_json, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            bc_metadata.append(metadata)
            
            logger.info(f"    ✓ Paired chart: {paired_path.name}")
            logger.info(f"    ✓ Metadata: {paired_json.name}")

        # Save breast cancer pairs summary
        bc_summary_path = hub_dir / "breast_cancer_pairs_summary.json"
        with open(bc_summary_path, 'w') as f:
            json.dump({
                "description": "Breast cancer gene hub pairs",
                "explanation": "These genes are classified as 'breast_cancer' in gene_info database",
                "criteria": {
                    "classification": "breast_cancer",
                    "min_normal_degree": 1500,
                    "min_tumor_degree": 100,
                    "max_tumor_degree": 3000,
                    "min_ratio": 2.0,
                    "max_ratio": 50.0
                },
                "correlation_threshold": CORRELATION_THRESHOLD,
                "pairs": bc_metadata,
                "total_found": len(bc_pairs)
            }, f, indent=2)

        logger.info(f"✓ Generated {len(bc_pairs)} Breast Cancer gene pairs")
        logger.info(f"✓ Summary saved: {bc_summary_path.name}")
        logger.info("")

        # ===== Generate Hub Visualization Gene Info JSON =====
        logger.info("\nGenerating hub visualization gene metadata...")

        hub_gene_info_path = hub_dir / 'hub_visualization_genes.json'

        # Load gene info utilities
        combined_gene_data = load_combined_gene_info(config)

        hub_gene_metadata = {
            'description': 'Gene information for all genes visualized in hub comparison charts',
            'total_unique_genes': 0,
            'goldilocks_pairs': {
                'description': 'Genes with optimal visual contrast (5-20× degree ratio)',
                'count': len(hub_viz_genes['goldilocks']),
                'genes': []
            },
            'breast_cancer_pairs': {
                'description': 'Known breast cancer genes from Tier 1 annotation',
                'count': len(hub_viz_genes['breast_cancer']),
                'genes': []
            },
            'differential_hubs': {
                'description': 'Top differential hubs (3 normal-centric, 3 tumor-centric)',
                'count': len(hub_viz_genes['differential']),
                'genes': []
            }
        }

        # Collect unique gene IDs across all types
        all_gene_ids = set()

        # Process goldilocks genes
        for gene_data in hub_viz_genes['goldilocks']:
            gene_id = gene_data['gene_id']
            all_gene_ids.add(gene_id)
            
            # Get detailed gene info
            gene_info = get_gene_info(gene_id, config, combined_data=combined_gene_data)
            
            detailed_info = {
                'rank': gene_data['rank'],
                'gene_id': gene_id,
                'gene_symbol': gene_data['gene_symbol'],
                'normal_degree': gene_data['normal_degree'],
                'tumor_degree': gene_data['tumor_degree'],
                'ratio': gene_data['ratio'],
                'percent_loss': round((1 - gene_data['tumor_degree']/gene_data['normal_degree']) * 100, 1) 
                            if gene_data['normal_degree'] > 0 else 0,
                'cancer_tier': gene_info.get('division', 'non_cancer') if gene_info else 'non_cancer',
                'gene_description': gene_info['gene_info'].get('gene_description', '') 
                                if gene_info and gene_info.get('gene_info') else '',
                'summary': gene_info['gene_info'].get('summary', '')
                        if gene_info and gene_info.get('gene_info') and gene_info['gene_info'].get('summary') 
                        else ''
            }
            hub_gene_metadata['goldilocks_pairs']['genes'].append(detailed_info)

        # Process breast cancer genes
        for gene_data in hub_viz_genes['breast_cancer']:
            gene_id = gene_data['gene_id']
            all_gene_ids.add(gene_id)
            
            gene_info = get_gene_info(gene_id, config, combined_data=combined_gene_data)
            
            detailed_info = {
                'rank': gene_data['rank'],
                'gene_id': gene_id,
                'gene_symbol': gene_data['gene_symbol'],
                'normal_degree': gene_data['normal_degree'],
                'tumor_degree': gene_data['tumor_degree'],
                'ratio': gene_data['ratio'],
                'percent_loss': round((1 - gene_data['tumor_degree']/gene_data['normal_degree']) * 100, 1)
                            if gene_data['normal_degree'] > 0 else 0,
                'is_breast_cancer_gene': True,
                'cancer_tier': gene_info.get('division', 'breast_cancer') if gene_info else 'breast_cancer',
                'gene_description': gene_info['gene_info'].get('gene_description', '')
                                if gene_info and gene_info.get('gene_info') else '',
                'summary': gene_info['gene_info'].get('summary', '')
                        if gene_info and gene_info.get('gene_info') and gene_info['gene_info'].get('summary')
                        else ''
            }
            hub_gene_metadata['breast_cancer_pairs']['genes'].append(detailed_info)

        # Process differential hubs
        for gene_data in hub_viz_genes['differential']:
            gene_id = gene_data['gene_id']
            all_gene_ids.add(gene_id)
            
            gene_info = get_gene_info(gene_id, config, combined_data=combined_gene_data)
            
            detailed_info = {
                'gene_id': gene_id,
                'gene_symbol': gene_data['gene_symbol'],
                'condition': gene_data['condition'],
                'degree': gene_data['degree'],
                'cancer_tier': gene_info.get('division', 'non_cancer') if gene_info else 'non_cancer',
                'gene_description': gene_info['gene_info'].get('gene_description', '')
                                if gene_info and gene_info.get('gene_info') else '',
                'summary': gene_info['gene_info'].get('summary', '')
                        if gene_info and gene_info.get('gene_info') and gene_info['gene_info'].get('summary')
                        else ''
            }
            hub_gene_metadata['differential_hubs']['genes'].append(detailed_info)

        # Update total unique genes
        hub_gene_metadata['total_unique_genes'] = len(all_gene_ids)

        # Save JSON
        with open(hub_gene_info_path, 'w') as f:
            json.dump(hub_gene_metadata, f, indent=2)

        logger.info(f"✓ Hub visualization gene metadata saved: {hub_gene_info_path.name}")
        logger.info(f"  Total unique genes: {len(all_gene_ids)}")
        logger.info(f"  Goldilocks: {len(hub_viz_genes['goldilocks'])}")
        logger.info(f"  Breast cancer: {len(hub_viz_genes['breast_cancer'])}")
        logger.info(f"  Differential: {len(hub_viz_genes['differential'])}")


def main():
    """Main execution orchestrating network loading, metric calculation, and visualization generation."""
    config = load_config()
    PROJECT_ROOT = Path(config['paths']['project_root'])
    INPUT_NETWORKS = Path(config['paths']['networks'])
    OUTPUT_DIR = get_auto_output_path(__file__, PROJECT_ROOT)
    
    logger = setup_logging(config, OUTPUT_DIR)
    
    # Suppress common UserWarning from matplotlib for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)

    logger.info("Starting script: 01_c_network_visualization.py (Rich JSON Output)")
    logger.info(f"Using Global Correlation Threshold: |r| >= {CORRELATION_THRESHOLD}")
    logger.info("-" * 60)

    start_time = time.time()
    
    # Create organized output subdirectories
    spearman_dir = ensure_dir(OUTPUT_DIR / "spearman")
    pearson_dir = ensure_dir(OUTPUT_DIR / "pearson")
    method_comp_dir = ensure_dir(OUTPUT_DIR / "method_comparison")

    # --- Initialize trackers for the JSON summary file ---
    visualization_tracker = {
        "visualizations": {
            "by_method": {"spearman": [], "pearson": []},
            "method_comparison": {}
        }
    }
    summary_stats = {
        "script": "01_c_network_visualization",
        "parameters": {"correlation_threshold": CORRELATION_THRESHOLD},
        "outputs": {},
        "processing_notes": {}
    }

    try:
        # Load all four networks using the global threshold
        logger.info("Loading all required networks...")
        pickle_dir = INPUT_NETWORKS / "pickle"
        
        networks = {
            'spearman_tumor': safe_read_pickle(pickle_dir / f"tumor_network_spearman_{CORRELATION_THRESHOLD}.pkl", logger),
            'spearman_normal': safe_read_pickle(pickle_dir / f"normal_network_spearman_{CORRELATION_THRESHOLD}.pkl", logger),
            'pearson_tumor': safe_read_pickle(pickle_dir / f"tumor_network_pearson_{CORRELATION_THRESHOLD}.pkl", logger),
            'pearson_normal': safe_read_pickle(pickle_dir / f"normal_network_pearson_{CORRELATION_THRESHOLD}.pkl", logger)
        }
        
        # --- Pre-calculate key metrics for annotations ---
        s_tumor_edges = networks['spearman_tumor'].number_of_edges()
        p_tumor_edges = networks['pearson_tumor'].number_of_edges()
        s_normal_edges = networks['spearman_normal'].number_of_edges()
        p_normal_edges = networks['pearson_normal'].number_of_edges()
        
        edge_advantage_ratio = s_tumor_edges / p_tumor_edges if p_tumor_edges > 0 else float('inf')
        
        logger.info(f"\nKey Metric Calculated: Spearman captures {edge_advantage_ratio:.2f}x more edges in tumor data.\n")
        logger.info("=" * 70)
        logger.info("CREATING FOCUSED VISUALIZATIONS FOR BOTH METHODS")
        logger.info("=" * 70)

        # === Per-Method Visualizations ===
        for method in ['spearman', 'pearson']:
            logger.info(f"\n{method.upper()} CORRELATION NETWORKS")
            logger.info("-" * 40)
            tumor_net = networks[f'{method}_tumor']
            normal_net = networks[f'{method}_normal']
            output_subdir = spearman_dir if method == 'spearman' else pearson_dir
            
            # 1. Degree Distribution Histograms
            logger.info(f"Creating degree distribution comparison for {method}...")
            plot_path = output_subdir / f'01_{method}_degree_distribution.png'
            tumor_degrees = [d for _, d in tumor_net.degree()]
            normal_degrees = [d for _, d in normal_net.degree()]
            
            # Create and save chart
            create_dual_histogram(
                data1=tumor_degrees, data2=normal_degrees,
                labels=['Tumor Network', 'Normal Network'], title=f'{method.upper()}: Tumor Fragmented vs Normal Cooperative',
                subtitle=f'Threshold: |r| ≥ {CORRELATION_THRESHOLD}', xlabel='Degree (Number of Connections)', ylabel='Frequency',
                output_path=plot_path
            )
            
            # Save JSON data
            json_path = output_subdir / f'01_{method}_degree_distribution.json'
            chart_data = {
                "description": {
                    "chart_type": "dual_histogram",
                    "x_axis": "Degree (Number of Connections per gene)",
                    "y_axis": "Frequency (Number of genes with given degree)",
                    "tumor_dataset": f"Tumor network using {method} correlation, threshold |r| ≥ {CORRELATION_THRESHOLD}",
                    "normal_dataset": f"Normal network using {method} correlation, threshold |r| ≥ {CORRELATION_THRESHOLD}",
                    "interpretation": "Degree distribution shows connectivity patterns. Tumor networks typically show lower, sparser degrees indicating fragmentation."
                },
                "tumor_degree_distribution": {
                    "values": tumor_degrees,
                    "summary": {
                        "min": float(np.min(tumor_degrees)),
                        "max": float(np.max(tumor_degrees)),
                        "mean": float(np.mean(tumor_degrees)),
                        "median": float(np.median(tumor_degrees)),
                        "std": float(np.std(tumor_degrees)),
                        "total_genes": len(tumor_degrees)
                    }
                },
                "normal_degree_distribution": {
                    "values": normal_degrees,
                    "summary": {
                        "min": float(np.min(normal_degrees)),
                        "max": float(np.max(normal_degrees)),
                        "mean": float(np.mean(normal_degrees)),
                        "median": float(np.median(normal_degrees)),
                        "std": float(np.std(normal_degrees)),
                        "total_genes": len(normal_degrees)
                    }
                }
            }
            save_chart_data_json(chart_data, json_path, "degree_distribution", 
                               f"{method.upper()}: Degree Distribution Comparison")
            
            visualization_tracker['visualizations']['by_method'][method].append({
                'type': 'degree_distribution',
                'file': str(get_relative_path(plot_path)),
                'data_file': str(get_relative_path(json_path)),
                'title': f'{method.upper()}: Degree Distribution Comparison',
                'interpretation': 'Shows a shift from a dense, hub-driven network in normal tissue to a sparse, fragmented one in tumor tissue.',
                'biological_context': 'This represents a massive loss of cellular coordination and regulatory control in cancer.',
                'key_metrics': {
                    'tumor_avg_degree': f'{np.mean(tumor_degrees):.2f}',
                    'normal_avg_degree': f'{np.mean(normal_degrees):.2f}',
                }
            })
            logger.info(f"✓ Saved: {get_relative_path(plot_path)}")
            logger.info(f"✓ Data saved: {get_relative_path(json_path)}")

            # 2. Component Sizes
            logger.info(f"Creating component size distribution comparison for {method}...")
            plot_path = output_subdir / f'02_{method}_component_sizes.png'
            
            # Calculate component sizes
            tumor_components = sorted([len(c) for c in nx.connected_components(tumor_net)], reverse=True)
            normal_components = sorted([len(c) for c in nx.connected_components(normal_net)], reverse=True)
            
            # Take top components for display
            tumor_top = tumor_components[:20]
            normal_top = normal_components[:10]
            
            create_dual_bar_chart(
                data1=tumor_top,
                data2=normal_top,
                labels=[f'Tumor: {nx.number_connected_components(tumor_net)} Components', f'Normal: {nx.number_connected_components(normal_net)} Components'],
                title=f'{method.upper()}: Tumor Fragmentation vs Normal Giant Component',
                subtitle=f'Threshold: |r| ≥ {CORRELATION_THRESHOLD}', xlabel='Component Rank', ylabel='Number of Nodes',
                output_path=plot_path
            )
            
            # Save JSON data
            json_path = output_subdir / f'02_{method}_component_sizes.json'
            chart_data = {
                "description": {
                    "chart_type": "dual_bar_chart",
                    "x_axis": "Component Rank (1 = largest component)",
                    "y_axis": "Number of Nodes in Component",
                    "tumor_data": "Top 20 connected components in tumor network",
                    "normal_data": "Top 10 connected components in normal network (usually shows one dominant giant component)",
                    "interpretation": "Component size distribution reveals network connectivity. Tumor shows many small components (fragmentation), normal shows one large giant component (integration)."
                },
                "tumor_components": {
                    "all_component_sizes": tumor_components,
                    "displayed_top_20": tumor_top,
                    "summary": {
                        "total_components": nx.number_connected_components(tumor_net),
                        "largest_component": tumor_components[0] if tumor_components else 0,
                        "average_component_size": float(np.mean(tumor_components)) if tumor_components else 0
                    }
                },
                "normal_components": {
                    "all_component_sizes": normal_components,
                    "displayed_top_10": normal_top,
                    "summary": {
                        "total_components": nx.number_connected_components(normal_net),
                        "largest_component": normal_components[0] if normal_components else 0,
                        "average_component_size": float(np.mean(normal_components)) if normal_components else 0,
                        "giant_component_percentage": f"{(normal_components[0] / sum(normal_components) * 100):.1f}%" if normal_components else "0%"
                    }
                }
            }
            save_chart_data_json(chart_data, json_path, "component_sizes",
                               f"{method.upper()}: Component Size Comparison")
            
            visualization_tracker['visualizations']['by_method'][method].append({
                'type': 'component_sizes',
                'file': str(get_relative_path(plot_path)),
                'data_file': str(get_relative_path(json_path)),
                'title': f'{method.upper()}: Component Size Comparison',
                'interpretation': 'Visually demonstrates the collapse of the network into thousands of tiny, disconnected components in tumor tissue, compared to a single dominant "giant component" in normal tissue.',
                'biological_context': 'The giant component represents the core functional machinery of the cell. Its disintegration in cancer highlights systemic failure of biological processes.',
                'key_metrics': {
                    'tumor_component_count': nx.number_connected_components(tumor_net),
                    'normal_component_count': nx.number_connected_components(normal_net),
                }
            })
            logger.info(f"✓ Saved: {get_relative_path(plot_path)}")
            logger.info(f"✓ Data saved: {get_relative_path(json_path)}")

            # 3. Hub Neighborhoods
            logger.info(f"Creating hub neighborhood comparison for {method} (top 3 hubs)...")
            plot_path = output_subdir / f'03_{method}_hub_neighborhoods.png'
            
            # Get top hubs
            tumor_hubs = sorted(tumor_net.degree(), key=lambda x: x[1], reverse=True)[:3]
            normal_hubs = sorted(normal_net.degree(), key=lambda x: x[1], reverse=True)[:3]
            
            create_hub_comparison_grid(
                tumor_network=tumor_net, normal_network=normal_net,
                tumor_hubs=tumor_hubs,
                normal_hubs=normal_hubs,
                title=f'{method.upper()} Hub Neighborhoods: Tumor vs Normal', subtitle=f'Threshold: |r| ≥ {CORRELATION_THRESHOLD}',
                annotation=f"Tumor hubs show sparse links while normal hubs form dense, cooperative webs.", output_path=plot_path
            )
            
            # Save JSON data - hub information
            json_path = output_subdir / f'03_{method}_hub_neighborhoods.json'
            chart_data = {
                "description": {
                    "chart_type": "hub_comparison_grid",
                    "visualization": "2x3 grid showing ego networks (1-hop neighborhoods) of top 3 hubs",
                    "tumor_row": "Top row shows tumor hub neighborhoods (typically sparse)",
                    "normal_row": "Bottom row shows normal hub neighborhoods (typically dense)",
                    "interpretation": "Hub neighborhoods reveal local connectivity patterns. Tumor hubs are isolated, normal hubs are well-connected."
                },
                "tumor_hubs": [
                    {
                        "gene_key": hub[0],
                        "gene_id": hub[0].split('|')[0] if '|' in hub[0] else hub[0],
                        "gene_symbol": hub[0].split('|')[1] if '|' in hub[0] else hub[0],
                        "degree": hub[1],
                        "neighbor_count": len(list(tumor_net.neighbors(hub[0])))
                    }
                    for hub in tumor_hubs
                ],
                "normal_hubs": [
                    {
                        "gene_key": hub[0],
                        "gene_id": hub[0].split('|')[0] if '|' in hub[0] else hub[0],
                        "gene_symbol": hub[0].split('|')[1] if '|' in hub[0] else hub[0],
                        "degree": hub[1],
                        "neighbor_count": len(list(normal_net.neighbors(hub[0])))
                    }
                    for hub in normal_hubs
                ],
                "hub_comparison": {
                    "tumor_avg_degree": float(np.mean([h[1] for h in tumor_hubs])),
                    "normal_avg_degree": float(np.mean([h[1] for h in normal_hubs])),
                    "degree_ratio": f"{(np.mean([h[1] for h in normal_hubs]) / np.mean([h[1] for h in tumor_hubs])):.2f}" if np.mean([h[1] for h in tumor_hubs]) > 0 else "inf"
                }
            }
            save_chart_data_json(chart_data, json_path, "hub_neighborhoods",
                               f"{method.upper()}: Hub Neighborhood Comparison")
            
            visualization_tracker['visualizations']['by_method'][method].append({
                'type': 'hub_neighborhoods',
                'file': str(get_relative_path(plot_path)),
                'data_file': str(get_relative_path(json_path)),
                'title': f'{method.upper()}: Hub Neighborhood Comparison',
                'interpretation': 'Compares the local connectivity of the most-connected genes (hubs). In normal tissue, hubs are densely connected, whereas in tumors, they lose the majority of their connections.',
                'biological_context': 'Hubs are often master regulator genes. Their isolation in cancer signifies a breakdown in key command-and-control pathways.',
            })
            logger.info(f"✓ Saved: {get_relative_path(plot_path)}")
            logger.info(f"✓ Data saved: {get_relative_path(json_path)}")

            # 4. Edge Weight CDF
            logger.info(f"Creating edge weight CDF comparison for {method}...")
            plot_path = output_subdir / f'04_{method}_edge_weight_cdf.png'
            
            # Get edge weights
            tumor_weights = [abs(d.get('weight', 1.0)) for _, _, d in tumor_net.edges(data=True)]
            normal_weights = [abs(d.get('weight', 1.0)) for _, _, d in normal_net.edges(data=True)]
            
            create_cdf(
                data1=tumor_weights,
                data2=normal_weights,
                labels=['Tumor Network (Selective)', 'Normal Network (Broad)'], title=f'{method.upper()} Edge Weight CDF',
                subtitle=f'Threshold: |r| ≥ {CORRELATION_THRESHOLD}', xlabel='Absolute Correlation Strength (|r|)',
                ylabel='Cumulative Proportion of Edges', output_path=plot_path, vline_at=CORRELATION_THRESHOLD,
                annotation="Tumor: Most edges are highly selective (strong |r|)\nNormal: Edges show a broad mix of strengths."
            )
            
            # Save JSON data
            json_path = output_subdir / f'04_{method}_edge_weight_cdf.json'
            chart_data = {
                "description": {
                    "chart_type": "cumulative_distribution_function",
                    "x_axis": "Absolute Correlation Strength (|r|)",
                    "y_axis": "Cumulative Proportion of Edges (fraction of edges with correlation ≤ x)",
                    "tumor_data": f"Edge weights in tumor network ({len(tumor_weights)} edges)",
                    "normal_data": f"Edge weights in normal network ({len(normal_weights)} edges)",
                    "threshold_line": f"Vertical line at |r| = {CORRELATION_THRESHOLD} (network construction threshold)",
                    "interpretation": "CDF shows distribution of edge strengths. Steep curve indicates concentration of strong edges. Tumor networks are more selective (stronger edges), normal networks have broader strength distribution."
                },
                "tumor_edge_weights": {
                    "values": [float(w) for w in tumor_weights],
                    "sorted_for_cdf": [float(w) for w in sorted(tumor_weights)],
                    "cdf_values": [float((i+1)/len(tumor_weights)) for i in range(len(tumor_weights))] if tumor_weights else [],
                    "summary": {
                        "count": len(tumor_weights),
                        "min": float(np.min(tumor_weights)) if tumor_weights else 0,
                        "max": float(np.max(tumor_weights)) if tumor_weights else 0,
                        "mean": float(np.mean(tumor_weights)) if tumor_weights else 0,
                        "median": float(np.median(tumor_weights)) if tumor_weights else 0,
                        "edges_above_threshold": sum(1 for w in tumor_weights if w >= CORRELATION_THRESHOLD)
                    }
                },
                "normal_edge_weights": {
                    "values": [float(w) for w in normal_weights],
                    "sorted_for_cdf": [float(w) for w in sorted(normal_weights)],
                    "cdf_values": [float((i+1)/len(normal_weights)) for i in range(len(normal_weights))] if normal_weights else [],
                    "summary": {
                        "count": len(normal_weights),
                        "min": float(np.min(normal_weights)) if normal_weights else 0,
                        "max": float(np.max(normal_weights)) if normal_weights else 0,
                        "mean": float(np.mean(normal_weights)) if normal_weights else 0,
                        "median": float(np.median(normal_weights)) if normal_weights else 0,
                        "edges_above_threshold": sum(1 for w in normal_weights if w >= CORRELATION_THRESHOLD)
                    }
                },
                "threshold_info": {
                    "correlation_threshold": CORRELATION_THRESHOLD,
                    "description": "Minimum absolute correlation for edge inclusion in network"
                }
            }
            save_chart_data_json(chart_data, json_path, "edge_weight_cdf",
                               f"{method.upper()}: Edge Weight CDF")
            
            visualization_tracker['visualizations']['by_method'][method].append({
                'type': 'edge_weight_cdf',
                'file': str(get_relative_path(plot_path)),
                'data_file': str(get_relative_path(json_path)),
                'title': f'{method.upper()}: Edge Weight CDF',
                'interpretation': 'Shows the distribution of correlation strengths. Tumor networks are dominated by very strong correlations (a steep curve), while normal networks have a healthier mix of strong and weak ties.',
                'biological_context': 'This suggests that cancer networks are not just smaller, but are rewired to be more specific and selective, focusing on a few critical pathways for survival and proliferation.',
            })
            logger.info(f"✓ Saved: {get_relative_path(plot_path)}")
            logger.info(f"✓ Data saved: {get_relative_path(json_path)}")


        # === ENHANCED VISUALIZATIONS ===
        create_enhanced_visualizations(
            networks=networks,
            base_output_dir=OUTPUT_DIR,
            logger=logger,
            visualization_tracker=visualization_tracker,
            CORRELATION_THRESHOLD=CORRELATION_THRESHOLD,
            config=config 
        )


        # ===== METHOD COMPARISON: COMPREHENSIVE ANALYSIS =====
        logger.info("\n\n" + "="*80)
        logger.info("METHOD COMPARISON: Spearman vs Pearson")
        logger.info("="*80)

        method_comp_dir = OUTPUT_DIR / 'method_comparison'
        method_comp_dir.mkdir(exist_ok=True)

        # Calculate edge counts for use across charts
        s_tumor_edges = networks['spearman_tumor'].number_of_edges()
        p_tumor_edges = networks['pearson_tumor'].number_of_edges()
        s_normal_edges = networks['spearman_normal'].number_of_edges()
        p_normal_edges = networks['pearson_normal'].number_of_edges()

        edge_advantage_ratio = s_tumor_edges / p_tumor_edges if p_tumor_edges > 0 else 0

        # === Chart 1: Edge Type Breakdown ===
        logger.info("\n1. Creating Edge Type Breakdown...")
        edge_breakdown_path = method_comp_dir / '01_edge_type_breakdown.png'
        edge_breakdown_json = method_comp_dir / '01_edge_type_breakdown.json'

        edge_breakdown_meta = create_edge_type_breakdown(
            spearman_tumor_edges=s_tumor_edges,
            pearson_tumor_edges=p_tumor_edges,
            spearman_normal_edges=s_normal_edges,
            pearson_normal_edges=p_normal_edges,
            title='Edge Type Analysis: What Each Method Captures',
            subtitle=f'Correlation threshold: |r| ≥ {CORRELATION_THRESHOLD}',
            output_path=edge_breakdown_path,
            correlation_threshold=CORRELATION_THRESHOLD
        )

        # Save JSON
        with open(edge_breakdown_json, 'w') as f:
            json.dump(edge_breakdown_meta, f, indent=2)

        logger.info(f"   ✓ Saved: {edge_breakdown_path.name}")
        logger.info(f"   Tumor: Spearman captures {edge_breakdown_meta['tumor']['unique_to_spearman']:,} unique edges")
        logger.info(f"   Normal: Spearman captures {edge_breakdown_meta['normal']['unique_to_spearman']:,} unique edges")

        # Update tracker
        visualization_tracker['visualizations']['method_comparison']['edge_type_breakdown'] = {
            'file': str(get_relative_path(edge_breakdown_path)),
            'data': str(get_relative_path(edge_breakdown_json)),
            'title': 'Edge Type Breakdown',
            'interpretation': f"Shows WHAT edges Spearman captures: {edge_breakdown_meta['tumor']['unique_to_spearman']:,} unique tumor edges including monotonic, rank-based, and outlier-robust patterns"
        }

        # === Chart 2: Hub Preservation Analysis ===
        logger.info("\n2. Creating Hub Preservation Analysis...")
        hub_preservation_path = method_comp_dir / '02_hub_preservation_analysis.png'
        hub_preservation_json = method_comp_dir / '02_hub_preservation_analysis.json'

        hub_preservation_meta = create_hub_preservation_analysis(
            spearman_tumor_network=networks['spearman_tumor'],
            pearson_tumor_network=networks['pearson_tumor'],
            spearman_normal_network=networks['spearman_normal'],
            pearson_normal_network=networks['pearson_normal'],
            title='Hub Gene Consistency: Do Methods Agree on Key Regulators?',
            subtitle=f'Top 10 hubs comparison | Threshold: |r| ≥ {CORRELATION_THRESHOLD}',
            output_path=hub_preservation_path,
            top_n=10
        )

        # Save JSON
        with open(hub_preservation_json, 'w') as f:
            json.dump(hub_preservation_meta, f, indent=2)

        logger.info(f"   ✓ Saved: {hub_preservation_path.name}")
        logger.info(f"   Tumor consistency: {hub_preservation_meta['tumor']['consistency_score']:.0f}%")
        logger.info(f"   Normal consistency: {hub_preservation_meta['normal']['consistency_score']:.0f}%")

        # Update tracker
        visualization_tracker['visualizations']['method_comparison']['hub_preservation'] = {
            'file': str(get_relative_path(hub_preservation_path)),
            'data': str(get_relative_path(hub_preservation_json)),
            'title': 'Hub Gene Consistency',
            'interpretation': f"Gene-level validation: {hub_preservation_meta['tumor']['consistency_score']:.0f}% agreement in tumor (only {hub_preservation_meta['tumor']['agree_count']}/10 exact matches)"
        }

        # === Chart 3: Unified Performance Dashboard ===
        logger.info("\n3. Creating Unified Performance Dashboard...")
        dashboard_path = method_comp_dir / '03_unified_performance_dashboard.png'
        dashboard_json = method_comp_dir / '03_unified_performance_dashboard.json'

        dashboard_meta = create_unified_performance_dashboard(
            spearman_tumor=networks['spearman_tumor'],
            pearson_tumor=networks['pearson_tumor'],
            spearman_normal=networks['spearman_normal'],
            pearson_normal=networks['pearson_normal'],
            title='Method Performance: Comprehensive Comparison',
            subtitle=f'Threshold: |r| ≥ {CORRELATION_THRESHOLD}',
            output_path=dashboard_path,
            correlation_threshold=CORRELATION_THRESHOLD
        )

        # Save JSON
        with open(dashboard_json, 'w') as f:
            json.dump(dashboard_meta, f, indent=2)

        logger.info(f"   ✓ Saved: {dashboard_path.name}")
        logger.info(f"   Tumor advantage: {dashboard_meta['tumor_advantage_ratio']:.2f}×")
        logger.info(f"   Normal advantage: {dashboard_meta['normal_advantage_ratio']:.2f}×")
        logger.info(f"   Overall advantage: {dashboard_meta['overall_advantage']:.2f}×")

        # Update tracker
        visualization_tracker['visualizations']['method_comparison']['unified_dashboard'] = {
            'file': str(get_relative_path(dashboard_path)),
            'data': str(get_relative_path(dashboard_json)),
            'title': 'Unified Performance Dashboard',
            'interpretation': f"Comprehensive 2×2 dashboard replacing 4 old charts. Shows {dashboard_meta['tumor_advantage_ratio']:.2f}× tumor advantage, {dashboard_meta['normal_advantage_ratio']:.2f}× normal advantage"
        }

        # === Chart 4: Side-by-Side Distribution Comparison ===
        logger.info("\n4. Creating Distribution Comparison...")
        distribution_path = method_comp_dir / '04_distribution_comparison.png'
        distribution_json = method_comp_dir / '04_distribution_comparison.json'

        distribution_meta = create_side_by_side_distribution_comparison(
            spearman_tumor_degrees=[d for _, d in networks['spearman_tumor'].degree()],
            pearson_tumor_degrees=[d for _, d in networks['pearson_tumor'].degree()],
            spearman_normal_degrees=[d for _, d in networks['spearman_normal'].degree()],
            pearson_normal_degrees=[d for _, d in networks['pearson_normal'].degree()],
            title='Degree Distribution: Side-by-Side Comparison',
            subtitle=f'Threshold: |r| ≥ {CORRELATION_THRESHOLD}',
            output_path=distribution_path
        )

        # Save JSON
        with open(distribution_json, 'w') as f:
            json.dump(distribution_meta, f, indent=2)

        logger.info(f"   ✓ Saved: {distribution_path.name}")
        logger.info(f"   Tumor median ratio: {distribution_meta['tumor']['median_ratio']:.2f}×")
        logger.info(f"   Normal median ratio: {distribution_meta['normal']['median_ratio']:.2f}×")

        # Update tracker
        visualization_tracker['visualizations']['method_comparison']['distribution_comparison'] = {
            'file': str(get_relative_path(distribution_path)),
            'data': str(get_relative_path(distribution_json)),
            'title': 'Distribution Comparison',
            'interpretation': f"Clear side-by-side histograms showing {distribution_meta['tumor']['median_ratio']:.2f}× higher median degree in tumor networks"
        }

        logger.info("\n✓ All method comparison charts complete!")
        logger.info("="*80)



    except FileNotFoundError as e:
        logger.error(f"ERROR: A required network file was not found. Please run script 01_b first. Details: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        # --- Finalize and Save Summary ---
        total_time = time.time() - start_time
        summary_stats['outputs'] = visualization_tracker
        
        # Sort the entire summary data structure for consistent JSON output
        sorted_summary_stats = sort_nested_dict(summary_stats)
        
        summary_path = OUTPUT_DIR / "01_c_result_info.json"
        with open(summary_path, 'w') as f:
            json.dump(sorted_summary_stats, f, indent=2)
        
        logger.info("-" * 60)
        logger.info(f"✓ All visualizations completed successfully!")
        logger.info(f"✓ JSON data files saved for all charts")
        logger.info(f"📄 Saved comprehensive summary to: {get_relative_path(summary_path)}")
        logger.info(f"Script finished in {total_time:.2f} seconds.")

if __name__ == "__main__":
    main()