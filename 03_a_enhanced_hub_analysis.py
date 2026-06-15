"""
03_a_enhanced_hub_analysis.py

Script Purpose:
Hybrid hub analysis integrating pre-computed centralities from 01_b, DCEA rewiring deltas from 02_b,
and validated feature importance from 02_c classification. Identifies key network players and potential 
drivers in BRCA through comprehensive multi-modal scoring and ranking.

DEPENDENCIES & WORKFLOW PIPELINE:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   01_b_network  │    │ 02_b_dcea_viz   │    │ 02_c_sample     │
│    _analysis    │    │   _enrich       │    │ _classification │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • Network       │    │ • Delta         │    │ • Feature       │
│   centralities  │    │   connectivity  │    │   importance    │
│ • Node metrics  │    │ • Cancer        │    │ • Validated     │
│                 │    │   annotations   │    │   gene scores   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                         ┌─────────────────┐
                         │ 03_a_enhanced   │
                         │ hub_analysis    │
                         ├─────────────────┤
                         │ • Hybrid scoring│
                         │ • Multi-modal   │
                         │   integration   │
                         │ • Driver        │
                         │   identification│
                         └─────────────────┘
                                 │
                         ┌─────────────────┐
                         │ 03_b_functional │
                         │ characterization│
                         └─────────────────┘

Summary Logic:
1. Load pre-computed centralities from 01_b networks (avoid re-calculation)
2. Integrate DCEA deltas from 02_b for biologically meaningful enhancement
3. Integrate validated feature importance from 02_c classification results
4. Apply hybrid scoring combining predictive power and network disruption
5. Apply cancer relevance weights from 00_c gene annotations
6. Rank hubs by enhanced scoring with known BRCA gene highlighting
7. Generate comprehensive visualizations and outputs for 03_b

Key Features:
- Efficiency: Uses pre-computed centralities from 01_b
- Validation: Incorporates proven predictive power from 02_c
- Biological Relevance: Meaningful DCEA integration with cancer context
- Hybrid Scoring: Combines classification importance with network topology
- Comprehensive Outputs: Single authoritative hub ranking
- Static Visualizations: Publication-ready figures
"""

import pandas as pd
import numpy as np
import json
import logging
import time
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from utils.config import load_config
from utils.file import ensure_dir, get_relative_path, get_auto_output_path
from utils.genes import load_combined_gene_info, get_gene_info

# These should already be there, but verify:
from utils.genes import normalize_gene_id, extract_gene_symbol, get_gene_info, load_combined_gene_info
from utils.chart_advanced import create_paired_hub_comparison

# --- Global Analysis Configuration ---
CORRELATION_THRESHOLD = 0.7
# ------------------------------------

def setup_logging(config, output_dir):
    """Set up structured logging for hub analysis."""
    logger = logging.getLogger(__name__)
    logger.setLevel(config['logging']['level'])

    if logger.hasHandlers():
        logger.handlers.clear()

    log_dir = output_dir / 'logs'
    ensure_dir(log_dir)
    log_file = log_dir / '03_a_enhanced_hub_analysis.log'
    
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


def create_summary_json(summary_data, output_path, project_root):
    """Create standardized result info JSON with relative paths."""
    for section in ['inputs', 'outputs']:
        if section in summary_data:
            for key, value in summary_data[section].items():
                if isinstance(value, Path):
                    summary_data[section][key] = str(value.relative_to(project_root))

    with open(output_path, 'w') as f:
        json.dump(summary_data, f, indent=2)


# Add this function after select_consensus_hubs (around line 240):

def match_gene_to_network(gene_from_tsv, normal_network=None, tumor_network=None):
    """
    Match a gene ID from TSV to actual network node format.
    
    TSV has normalized IDs (no version): ENSG00000180660|MAB21L1
    Networks have versioned IDs: ENSG00000180660.14|MAB21L1
    
    Args:
        gene_from_tsv: Gene ID from enhanced_hub_ranking.tsv
        normal_network: NetworkX graph (optional, for lookup)
        tumor_network: NetworkX graph (optional, for lookup)
        
    Returns:
        str: Matching network node ID or normalized fallback
    """
    gene_normalized = normalize_gene_id(gene_from_tsv)
    
    # Try to find exact match in networks (with version number)
    if normal_network:
        for node in normal_network.nodes():
            if normalize_gene_id(node) == gene_normalized:
                return node
    
    if tumor_network:
        for node in tumor_network.nodes():
            if normalize_gene_id(node) == gene_normalized:
                return node
    
    # Fallback to normalized version
    return gene_normalized

def select_consensus_hubs(logger, hub_ranking_df, config, combined_gene_data=None, top_100=True, normal_network=None, tumor_network=None):
    """
    Select 9 consensus hubs for visualization.
    
    Strategy:
    - Top 3 overall (any tier, highest composite scores)
    - Top 3 breast cancer (Tier 1 only from top 100)
    - Top 3 general cancer (Tier 2 only from top 100)
    
    Args:
        hub_ranking_df: DataFrame with enhanced hub ranking
        config: Project configuration dict
        combined_gene_data: Pre-loaded gene info (optional)
        top_100: Only consider top 100 hubs; changed to 1000 because breast cancer first apper in 100+ position
        
    Returns:
        List of dicts with hub info for visualization
    """
    from utils.genes import get_gene_info, normalize_gene_id, extract_gene_symbol
    
    # Work with top 100 only
    if top_100:
        df = hub_ranking_df.head(1000).copy()
        logger.info(f"Available columns in df for consensus: {df.columns.tolist()}")
    else:
        df = hub_ranking_df.copy()
    
    selected = []
    
    # Category 1: Top 3 overall (any tier)
    logger.info("\n  Selecting top 3 OVERALL hubs...")
    for rank, (idx, row) in enumerate(df.head(3).iterrows(), 1):
        gene_id = match_gene_to_network(row['gene'], normal_network, tumor_network)
        symbol = extract_gene_symbol(row['gene'])
        
        # In select_consensus_hubs, after line 150, add debug:
        if normal_network and tumor_network:
            logger.info(f"DEBUG: Looking for gene_id='{gene_id}' (symbol='{symbol}') in networks")
            logger.info(f"  Sample network nodes: {list(normal_network.nodes())[:5]}")
            logger.info(f"  Gene ID in normal? {gene_id in normal_network}")
            logger.info(f"  Gene ID in tumor? {gene_id in tumor_network}")
            logger.info(f"  Symbol in normal? {symbol in normal_network}")
            logger.info(f"  Symbol in tumor? {symbol in tumor_network}")
            if gene_id in normal_network:
                logger.info(f"  Actual normal degree: {normal_network.degree(gene_id)}")
            if gene_id in tumor_network:
                logger.info(f"  Actual tumor degree: {tumor_network.degree(gene_id)}")
        else:
            logger.info(f"DEBUG: Networks not provided to function")


        # Get detailed gene info
        gene_info = get_gene_info(gene_id, config, combined_data=combined_gene_data)
        
        selected.append({
            'gene_id': gene_id,
            'gene_symbol': symbol,
            'category': 'overall',
            'category_label': 'Top Overall',
            'rank_in_category': idx + 1,
            'rank_in_top100': row['rank'],
            'composite_score': row['enhanced_score'],
            'cancer_relevance': row.get('cancer_relevance', 'non_cancer'),
            'feature_importance': row.get('feature_importance', 0),
            'delta_connectivity': row.get('delta_connectivity', 0),
            'gene_description': gene_info['gene_info'].get('gene_description', '') 
                               if gene_info and gene_info.get('gene_info') else '',
            'tier_from_annotation': gene_info.get('division', 'non_cancer') 
                                   if gene_info else 'non_cancer'
        })
        logger.info(f"    #{rank}: {symbol} (score={row['enhanced_score']:.1f}, tier={row.get('cancer_relevance','N/A')})")
    
    # Category 2: Top 3 breast cancer (Tier 1)
    logger.info("\n  Selecting top 3 BREAST CANCER hubs...")
    tier1 = df[df['cancer_relevance'] == 'breast_cancer']
    
    if len(tier1) == 0:
        logger.warning("    ⚠️  No Tier 1 (breast cancer) genes found in top 100!")
    else:
        for rank, (idx, row) in enumerate(tier1.head(3).iterrows(), 1):
            gene_id = match_gene_to_network(row['gene'], normal_network, tumor_network)
            symbol = extract_gene_symbol(row['gene'])
            gene_info = get_gene_info(gene_id, config, combined_data=combined_gene_data)
            
            selected.append({
                'gene_id': gene_id,
                'gene_symbol': symbol,
                'category': 'breast_cancer',
                'category_label': 'Breast Cancer',
                'rank_in_category': idx + 1,
                'rank_in_top100': row['rank'],
                'composite_score': row['enhanced_score'],
                'cancer_relevance': 'breast_cancer',
                'feature_importance': row.get('feature_importance', 0),
                'delta_connectivity': row.get('delta_connectivity', 0),
                'gene_description': gene_info['gene_info'].get('gene_description', '') 
                                   if gene_info and gene_info.get('gene_info') else '',
                'tier_from_annotation': 'breast_cancer'
            })
            logger.info(f"    #{rank}: {symbol} (score={row['enhanced_score']:.1f}, overall rank={idx + 1})")
        
        if len(tier1) < 3:
            logger.warning(f"    ⚠️  Only {len(tier1)} Tier 1 genes available (requested 3)")
    
    # Category 3: Top 3 general cancer (Tier 2)
    logger.info("\n  Selecting top 3 GENERAL CANCER hubs...")
    tier2 = df[df['cancer_relevance'] == 'cancer']
    
    if len(tier2) == 0:
        logger.warning("    ⚠️  No Tier 2 (general cancer) genes found in top 100!")
    else:
        for rank, (idx, row) in enumerate(tier2.head(3).iterrows(), 1):
            gene_id = match_gene_to_network(row['gene'], normal_network, tumor_network)
            symbol = extract_gene_symbol(row['gene'])
            gene_info = get_gene_info(gene_id, config, combined_data=combined_gene_data)
            
            selected.append({
                'gene_id': gene_id,
                'gene_symbol': symbol,
                'category': 'cancer',
                'category_label': 'General Cancer',
                'rank_in_category': idx + 1,
                'rank_in_top100': row['rank'],
                'composite_score': row['enhanced_score'],
                'cancer_relevance': 'cancer',
                'feature_importance': row.get('feature_importance', 0),
                'delta_connectivity': row.get('delta_connectivity', 0),
                'gene_description': gene_info['gene_info'].get('gene_description', '') 
                                   if gene_info and gene_info.get('gene_info') else '',
                'tier_from_annotation': 'cancer'
            })
            logger.info(f"    #{rank}: {symbol} (score={row['enhanced_score']:.1f}, overall rank={idx + 1})")
        
        if len(tier2) < 3:
            logger.warning(f"    ⚠️  Only {len(tier2)} Tier 2 genes available (requested 3)")
    
    logger.info(f"\n  ✓ Selected {len(selected)} consensus hubs total")
    return selected



def load_precomputed_centralities(config, project_root, logger):
    """
    Load pre-computed centrality metrics from 01_b network analysis.
    Uses global_metrics_comparison.json and network pickle files.
    
    DEPENDENCY: Requires 01_b_network_analysis to complete successfully
    """
    net_dir = Path(project_root) / config['paths']['networks']
    metrics_path = net_dir / 'global_metrics_comparison.json'
    
    if not metrics_path.exists():
        raise FileNotFoundError(f"Network metrics missing: {metrics_path}. Run 01_b first.")
    
    # Load global metrics
    with open(metrics_path, 'r') as f:
        metrics_data = json.load(f)
    
    # Extract primary method and threshold - USE CONFIG THRESHOLD
    primary_method = config['network_analysis']['primary_correlation_method']
    # Use the threshold from config with explicit index for clarity
    threshold_index = 1  # 0.7 threshold for balanced sensitivity analysis
    threshold = config['network_analysis']['correlation_thresholds'][threshold_index]
    
    logger.info(f"Using {primary_method} networks with threshold {threshold} (balanced sensitivity)")
    
    # Load node-level centralities from pickle files
    tumor_pkl_path = net_dir / 'pickle' / f'tumor_network_{primary_method}_{threshold}.pkl'
    
    if not tumor_pkl_path.exists():
        # Fallback: try to compute from GML or use alternative approach
        logger.warning(f"Network pickle missing: {tumor_pkl_path}")
        logger.info("Attempting to load from GML or using simplified approach...")
        return load_centralities_fallback(net_dir, primary_method, threshold, logger)
    
    try:
        import pickle
        logger.info(f"Loading network from: {tumor_pkl_path.name}")
        with open(tumor_pkl_path, 'rb') as f:
            tumor_network = pickle.load(f)
        
        # Extract centralities - handle different network object types
        logger.info("Extracting degree centralities...")
        if hasattr(tumor_network, 'degree'):
            tumor_degrees = dict(tumor_network.degree())
        else:
            # Fallback: create simple degree dictionary
            tumor_degrees = {node: len(neighbors) for node, neighbors in tumor_network.adjacency()} if hasattr(tumor_network, 'adjacency') else {}
        
        # Try to get betweenness, but provide fallback
        logger.info("Computing betweenness centrality (this may take a while for large networks)...")
        try:
            if len(tumor_network) > 1000:
                logger.info("Large network detected, using sampling for betweenness...")
                tumor_betweenness = nx.betweenness_centrality(tumor_network, k=min(500, len(tumor_network)))
            else:
                tumor_betweenness = nx.betweenness_centrality(tumor_network)
        except Exception as e:
            logger.warning(f"Betweenness computation failed: {e}, using degree-only approach")
            tumor_betweenness = {}
        
        logger.info(f"✓ Loaded centralities for {len(tumor_degrees)} nodes from 01_b")
        
        return {
            'degrees': tumor_degrees,
            'betweenness': tumor_betweenness,
            'global_metrics': metrics_data,
            'primary_method': primary_method,
            'threshold': threshold
        }
        
    except Exception as e:
        logger.error(f"Failed to load pickle: {e}")
        return load_centralities_fallback(net_dir, primary_method, threshold, logger)


def load_centralities_fallback(net_dir, primary_method, threshold, logger):
    """
    Fallback method to load or compute centralities when pickles are unavailable.
    This ensures analysis can continue even if some 01_b outputs are missing.
    """
    logger.info("Using fallback method to obtain centralities...")
    
    # Try to load from GML
    tumor_gml_path = net_dir / f"tumor_network_{primary_method}_{threshold}.gml"
    if tumor_gml_path.exists():
        try:
            logger.info(f"Loading network from GML: {tumor_gml_path.name}")
            tumor_network = nx.read_gml(str(tumor_gml_path))
            
            logger.info("Computing degree centralities...")
            tumor_degrees = dict(tumor_network.degree())
            
            # For large networks, skip betweenness or sample
            logger.info("Computing betweenness centrality...")
            if len(tumor_network) > 1000:
                logger.info("Large network, using sampling for betweenness computation")
                tumor_betweenness = nx.betweenness_centrality(tumor_network, k=min(500, len(tumor_network)))
            else:
                tumor_betweenness = nx.betweenness_centrality(tumor_network)
            
            logger.info(f"✓ Loaded centralities from GML: {len(tumor_degrees)} nodes")
            return {
                'degrees': tumor_degrees,
                'betweenness': tumor_betweenness,
                'global_metrics': {},
                'primary_method': primary_method,
                'threshold': threshold
            }
        except Exception as e:
            logger.error(f"GML load failed: {e}")
    
    # Ultimate fallback: create from connectivity data
    logger.info("Using connectivity-based fallback...")
    return create_synthetic_centralities(primary_method, threshold, logger)


def create_synthetic_centralities(primary_method, threshold, logger):
    """
    Create synthetic centrality data as last resort fallback.
    This should rarely be needed if previous scripts ran correctly.
    """
    logger.warning("Creating synthetic centrality data - results may be limited")
    
    # This would ideally load from differential_connectivity.tsv
    # For now, create minimal synthetic data
    synthetic_degrees = {'gene1': 100, 'gene2': 80, 'gene3': 60}  # Example
    synthetic_betweenness = {'gene1': 0.1, 'gene2': 0.05, 'gene3': 0.02}
    
    return {
        'degrees': synthetic_degrees,
        'betweenness': synthetic_betweenness,
        'global_metrics': {},
        'primary_method': primary_method,
        'threshold': threshold
    }


def load_dcea_enhancements(config, project_root, logger):
    """
    Load DCEA deltas and annotations from 02_b with flexible fallbacks.
    
    DEPENDENCY: Requires 02_b_dcea_viz_enrich to complete successfully
    Falls back to 02_a_differential_analysis if 02_b outputs missing
    """
    dcea_dir = Path(project_root) / config['paths']['dcea_viz_enrich']
    diff_dir = Path(project_root) / config['paths']['differential_analysis']
    
    # Check what files are available
    # logger.info(f"Checking 02_b directory: {dcea_dir}")
    dcea_files = list(dcea_dir.glob('*'))
    logger.info(f"Files in 02_b: {[f.name for f in dcea_files]}")
    
    # logger.info(f"Checking 02_a directory: {diff_dir}") 
    diff_files = list(diff_dir.glob('*'))
    logger.info(f"Files in 02_a: {[f.name for f in diff_files]}")
    
    delta_dict = {}
    cancer_relevance = {}
    
    # Try multiple possible locations for connectivity data
    possible_conn_paths = [
        diff_dir / 'differential_connectivity.tsv',  # Primary location (02_a)
        dcea_dir / 'differential_connectivity.tsv',  # Secondary location (02_b)
    ]
    
    conn_path = None
    for path in possible_conn_paths:
        if path.exists():
            conn_path = path
            break
    
    if conn_path:
        # logger.info(f"Loading DCEA deltas from: {conn_path}")
        conn_df = pd.read_csv(conn_path, sep='\t')
        delta_dict = dict(zip(conn_df['gene'], conn_df['delta_connectivity']))
        logger.info(f"✓ Loaded {len(delta_dict)} DCEA deltas from {conn_path.parent.name}")
    else:
        logger.warning("No connectivity delta file found. Using degree-only scoring.")
        # Create empty delta dict - will use degree-only scoring
    
    # Load annotated hubs for cancer relevance
    annotated_path = dcea_dir / 'annotated_hubs.json'
    if annotated_path.exists():
        logger.info(f"Loading cancer annotations from: {annotated_path.name}")
        with open(annotated_path, 'r') as f:
            hubs = json.load(f)
        cancer_relevance = {h['gene']: h['cancer_relevance'] for h in hubs}
        logger.info(f"✓ Loaded {len(cancer_relevance)} cancer annotations")
    else:
        logger.warning("No annotated hubs found. Using default cancer relevance.")
        # Will use 'non_cancer' as default for all genes
    
    return delta_dict, cancer_relevance


def load_classification_features(config, project_root, logger):
    """
    Load feature importance from consensus JSON with robust error handling.
    Prioritizes consensus data over legacy single-method files.
    
    RATIONALE:
    Consensus feature importance (FI) represents genes that are consistently 
    predictive of the tumor/normal phenotype across different sampling methods
    (e.g., median-based vs cluster-based). This reduces methodological bias.
    """
    class_dir = Path(project_root) / config['paths']['sample_classification']
    consensus_path = class_dir / 'sampling_comparison' / 'consensus_predictive_ranking.json'
    
    # Issue 1: Robust path relativity check
    # Ensures the error message is readable regardless of OS or execution context
    try:
        rel_path = consensus_path.relative_to(project_root)
    except ValueError:
        rel_path = consensus_path

    if not consensus_path.exists():
        logger.error("=" * 60)
        logger.error("CONSENSUS FILE NOT FOUND")
        logger.error("=" * 60)
        logger.error(f"Missing: {rel_path}")
        logger.error("")
        logger.error("REQUIRED: This file is generated by 02_c enhancements module")
        logger.error("TO FIX: Run 02_c_sample_classification.py with enhancements enabled")
        logger.error("")
        logger.error("IMPACT: Falling back to network-only scoring")
        logger.error("        Results will lack classification validation")
        logger.error("=" * 60)
        return {}, {'source_type': 'none', 'fallback_reason': 'consensus_missing'}

    try:
        with open(consensus_path, 'r') as f:
            data = json.load(f)
        
        # Validate required JSON structure (Refinement 3)
        # We need both the ranking array and metadata for provenance tracking
        if 'ranking' not in data or 'metadata' not in data:
            raise ValueError(f"Consensus JSON missing required keys: {list(data.keys())}")
        
        ranking = data['ranking']
        if not ranking:
            raise ValueError("Consensus ranking array is empty")

        # Issue 2: Defensive FI extraction & validation (Refinement 2)
        # Biological sanity check: FI values should typically be small decimals
        fi_values = [float(item['avg_feature_importance']) for item in ranking]
        if not fi_values:
            raise ValueError("No feature importance values extracted from ranking items")

        feature_dict = {}
        cancer_relevance_dict = {}
        for item in ranking:
            gene_id = item['gene']
            val = float(item['avg_feature_importance'])
            cancer_rel = item.get('cancer_relevance', 'non_cancer')
            
            # Map by full gene ID
            feature_dict[gene_id] = val
            cancer_relevance_dict[gene_id] = cancer_rel
            
            # Map by symbol (try both methods)
            if 'gene_symbol' in item:
                symbol = item['gene_symbol']
                feature_dict[symbol] = val
                cancer_relevance_dict[symbol] = cancer_rel
            elif '|' in gene_id:
                symbol = gene_id.split('|')[1]
                feature_dict[symbol] = val
                cancer_relevance_dict[symbol] = cancer_rel


        min_fi, max_fi = min(fi_values), max(fi_values)
        if max_fi > 1.0:
            logger.warning(f"⚠ Unexpected FI range: {min_fi:.6f} - {max_fi:.6f}")
            logger.warning("   Expected: 0.0 - 0.3 for feature importance")
            logger.warning("   Check if correct field ('avg_feature_importance') was loaded")

        # Issue 4: Metadata fallback for source_type (Refinement 4)
        # Identifies which sampling methods contributed to this specific score
        methods = data['metadata'].get('methods_included', [])
        if not methods:
            logger.warning("Metadata 'methods_included' is empty or missing")
            methods = ['unknown']
        source_type = f"consensus_{'+'.join(methods)}"

        source_metadata = {
            'source_file': str(rel_path),
            'source_type': source_type,
            'score_field': 'avg_feature_importance',
            'n_genes': len(ranking),
            'score_range': f"{min_fi:.6f} - {max_fi:.6f}",
            'methods_included': methods,
            'generated_on': data['metadata'].get('generated_on', 'unknown'),
            'validation_quality': 'high'  # High because it is cross-validated consensus
        }

        logger.info("=" * 60)
        logger.info(f"✓ CONSENSUS MODE ACTIVATED: {source_type}")
        logger.info(f"  Methods: {', '.join(methods)}")
        logger.info(f"  Range: {source_metadata['score_range']} | Genes: {len(ranking)}")
        logger.info("=" * 60)

        return feature_dict, source_metadata, cancer_relevance_dict

    except Exception as e:
        logger.error(f"Failed to parse consensus JSON: {e}")
        return {}, {'source_type': 'error', 'error_msg': str(e)}


def compute_enhanced_scores(centralities, delta_dict, cancer_relevance, 
                           feature_importance_dict, source_metadata, config, logger):
    """
    Enhanced scoring with hybrid approach combining classification and network metrics.
    
    HYBRID SCORING RATIONALE:
    - Combines validated predictive power (from 02_c) with network disruption (from 02_b)
    - Feature Importance (FI): Genes that successfully classify tumor vs. normal
    - Delta Connectivity: Genes whose network position changes in cancer
    - Intersection: Genes that are BOTH predictive AND mechanistically disrupted
    
    SCORING FORMULA:
        1. Normalize FI across all genes -> FI_norm [0,1]
        2. Normalize |delta| across all genes -> delta_norm [0,1]
        3. base_score = (FI_norm * 0.6) + (delta_norm * 0.4)
        4. Apply biological context: base_score * cancer_weight * driver_bonus * 1000
    """
    
    # Extract weights and multipliers from config
    fi_weight = config['hub_analysis']['feature_importance_weight']
    delta_weight = config['hub_analysis']['delta_connectivity_weight']
    cancer_bonus = config['hub_analysis']['cancer_bonus_multiplier']
    
    has_fi_data = bool(feature_importance_dict)
    fi_type = source_metadata.get('source_type', 'none')
    genes_to_process = list(centralities['degrees'].keys())
    
        
    # Expand annotation coverage with curated gene lists
    from utils.cancer_gene_lists import classify_gene_id

    logger.info("Expanding annotations with curated gene lists...")
    expanded = 0
    for gene in genes_to_process:
        if gene not in cancer_relevance:
            curated_class = classify_gene_id(gene)
            if curated_class != 'non_cancer':
                cancer_relevance[gene] = curated_class
                expanded += 1

    logger.info(f"✓ Added {expanded} annotations from curated lists")


    # Extract values for global normalization
    all_deltas = [abs(v) for v in delta_dict.values()] if delta_dict else [0.0]
    all_fis = [v for v in feature_importance_dict.values()] if has_fi_data else [0.0]
    
    # Issue 3: Safe max extraction with fallback (Refinement Logic)
    # Prevents DivisionByZero while logging the ranges for debug validation
    max_delta = max(all_deltas) if all_deltas and max(all_deltas) > 0 else 1.0
    max_fi = max(all_fis) if all_fis and max(all_fis) > 0 else 1.0
    logger.info(f"Normalization factors: max_delta={max_delta:.2f}, max_fi={max_fi:.6f}")

    # == DEBUG: Check what cancer genes are actually in the network
    cancer_genes_in_network = []
    for gene in genes_to_process:
        if gene in cancer_relevance or (gene.split('|')[1] if '|' in gene else gene) in cancer_relevance:
            for key in cancer_relevance:
                if gene in key or (gene.split('|')[1] if '|' in gene else gene) in key:
                    cancer_genes_in_network.append(gene)
                    break

    logger.info(f"DEBUG: Found {len(cancer_genes_in_network)} cancer-annotated genes in the network")
    if cancer_genes_in_network:
        logger.info("DEBUG: Sample cancer genes in network:")
        for gene in cancer_genes_in_network[:10]:
            symbol = gene.split('|')[1] if '|' in gene else gene
            logger.info(f"  {symbol}")
    # == END DEBUG

    hub_data = []
    
    for gene in genes_to_process:
        # A. Basic Network Metrics (Topology)
        degree = centralities['degrees'].get(gene, 0)
        betweenness = centralities['betweenness'].get(gene, 0)
        gene_symbol = gene.split('|')[1] if '|' in gene else gene
        
        # B. Connectivity Delta (Network Rewiring/Mechanistic Shift)
        delta = delta_dict.get(gene, 0.0)
        delta_norm = abs(delta) / max_delta
        
        # C. Feature Importance (Predictive Power)
        fi_val = feature_importance_dict.get(gene, feature_importance_dict.get(gene_symbol, 0.0))
        fi_norm = fi_val / max_fi
        
        # D. Biological Relevance Weighting 
        relevance = None
        gene_symbol = gene.split('|')[1] if '|' in gene else gene

        # Try multiple matching strategies
        if gene in cancer_relevance:
            relevance = cancer_relevance[gene]
        elif gene_symbol in cancer_relevance:
            relevance = cancer_relevance[gene_symbol]
        else:
            # Check if any key in cancer_relevance contains this gene symbol
            for key in cancer_relevance:
                if '|' in key and key.endswith(f"|{gene_symbol}"):
                    relevance = cancer_relevance[key]
                    break
                elif key == gene_symbol:
                    relevance = cancer_relevance[key]
                    break
            
            # If still not found, try case-insensitive matching
            if relevance is None:
                for key in cancer_relevance:
                    if gene_symbol.lower() == key.lower():
                        relevance = cancer_relevance[key]
                        break
            
            # Final fallback
            if relevance is None:
                relevance = 'non_cancer'

            # Apply multiplier for breast cancer genes from curated lists
            if relevance == 'breast_cancer':
                cancer_multiplier = 1.2
            elif relevance == 'cancer':
                cancer_multiplier = 1.1
            else:
                cancer_multiplier = 1.0

            
        # E. Hybrid Score Integration
        if has_fi_data:
            base_score = (fi_norm * fi_weight) + (delta_norm * delta_weight)
            scoring_method = "hybrid_fi_delta"
        else:
            base_score = delta_norm 
            scoring_method = "network_only_delta"

        # Apply simplified multiplier
        enhanced_score = base_score * cancer_multiplier * 1000
        
        hub_data.append({
            'gene': gene,
            'gene_symbol': gene_symbol,
            'degree': degree,
            'betweenness': betweenness,
            'delta_connectivity': delta,
            'feature_importance': fi_val,
            'cancer_relevance': relevance,
            'base_score': base_score,
            'enhanced_score': enhanced_score,
            'fi_norm': fi_norm,
            'fi_weight': fi_weight,
            'delta_norm': delta_norm,
            'delta_weight': delta_weight,
            'fi_source': fi_type,
            'scoring_method': scoring_method,
            # REWIRING THRESHOLD: |delta| > 0.1 considered significant network reorganization
            'is_rewired': abs(delta) > 0.1 # Logical threshold for DCEA significance
        })

    df_hubs = pd.DataFrame(hub_data)
    
    if not df_hubs.empty:
        # Sort by integrated priority and assign ordinal rank
        df_hubs = df_hubs.sort_values('enhanced_score', ascending=False).reset_index(drop=True)
        df_hubs['rank'] = df_hubs.index + 1
        
        # Add 0-1 normalized score for easier comparison across different network builds
        max_score = df_hubs['enhanced_score'].max()
        df_hubs['normalized_score'] = df_hubs['enhanced_score'] / max_score if max_score > 0 else 0


    # == DEBUG: Show breast_cancer gene scores
    breast_cancer_genes = df_hubs[df_hubs['cancer_relevance'] == 'breast_cancer']
    if not breast_cancer_genes.empty:
        logger.info(f"DEBUG: Found {len(breast_cancer_genes)} breast_cancer genes in df_hubs")
        logger.info("DEBUG: Top 10 breast_cancer genes by enhanced_score:")
        for _, row in breast_cancer_genes.sort_values('enhanced_score', ascending=False).head(10).iterrows():
            logger.info(f"  {row['gene_symbol']:15s}: score={row['enhanced_score']:.2f}, rank={row['rank']}")
    else:
        logger.warning("DEBUG: No breast_cancer genes found in df_hubs!") 
    # == END DEBUG 
            
    # Additional statistics on annotation coverage
    logger.info("\n" + "="*60)
    logger.info("ANNOTATION COVERAGE ANALYSIS")
    logger.info("="*60)

    # Genes with any annotation (from 02_b or consensus)
    annotated_genes = set(cancer_relevance.keys())
    genes_in_network = set(genes_to_process)  # Use genes_to_process instead of df_hubs

    # Intersection
    annotated_in_network = annotated_genes & genes_in_network
    unannotated_in_network = genes_in_network - annotated_genes

    logger.info(f"Total genes in network: {len(genes_in_network)}")
    logger.info(f"Genes with cancer annotation: {len(annotated_in_network)} ({len(annotated_in_network)/len(genes_in_network)*100:.1f}%)")
    logger.info(f"Unannotated genes (default to non-cancer): {len(unannotated_in_network)} ({len(unannotated_in_network)/len(genes_in_network)*100:.1f}%)")
    logger.info(f"")
    logger.info(f"Annotated genes by type:")
    for relevance_type in ['breast_cancer', 'cancer', 'non_cancer']:
        count = len([g for g in annotated_in_network if cancer_relevance.get(g) == relevance_type])
        if len(annotated_in_network) > 0:
            logger.info(f"  {relevance_type:20s}: {count:5d} ({count/len(annotated_in_network)*100:5.1f}% of annotated)")
    logger.info("="*60 + "\n")

    return df_hubs


def create_stratified_lists(df_hubs, config, logger):
    """
    Create 4 stratified gene lists from enhanced hub ranking with detailed statistics.
    
    Returns:
        dict: {
            'overall': [gene1, gene2, ...],
            'breast_cancer': [...],
            'general_cancer': [...],
            'novel_predictive': [...],
            'network_rewired': [...]
        }
    """
    top_n = config.get('hub_analysis', {}).get('top_hubs_count', 250)
    logger.info(f"Creating 5 stratified gene lists (top {top_n} each)")
    
    # First, print comprehensive statistics
    logger.info("\n" + "="*60)
    logger.info("STRATIFICATION STATISTICS")
    logger.info("="*60)
    
    total_genes = len(df_hubs)
    breast_cancer_count = len(df_hubs[df_hubs['cancer_relevance'] == 'breast_cancer'])
    general_cancer_count = len(df_hubs[df_hubs['cancer_relevance'] == 'cancer'])
    non_cancer_count = len(df_hubs[df_hubs['cancer_relevance'] == 'non_cancer'])
    
    logger.info(f"Total genes in network: {total_genes}")
    logger.info(f"")
    logger.info(f"Cancer Annotations:")
    logger.info(f"  Breast cancer genes:  {breast_cancer_count:5d} ({breast_cancer_count/total_genes*100:5.2f}%)")
    logger.info(f"  General cancer genes: {general_cancer_count:5d} ({general_cancer_count/total_genes*100:5.2f}%)")
    logger.info(f"  Non-cancer/unknown:   {non_cancer_count:5d} ({non_cancer_count/total_genes*100:5.2f}%)")
    logger.info(f"")
    logger.info(f"Note: 'Non-cancer/unknown' includes:")
    logger.info(f"  - Genes explicitly classified as non-cancer from curated lists")
    logger.info(f"  - Genes without cancer annotation (unannotated)")
    logger.info(f"  - Novel genes potentially relevant but not yet characterized")
    logger.info("="*60 + "\n")
    
    # Ensure base_score exists for unbiased ranking
    if 'base_score' not in df_hubs.columns:
        logger.error("base_score column missing - cannot create unbiased lists")
        return {}
    
    stratified_lists = {}
    
    # 1. Overall ranking (biased ranking with cancer multipliers)
    logger.info("1. Overall Top Predictive Hubs (with 1.2x boost for breast cancer genes)...")
    overall_genes = df_hubs.sort_values('enhanced_score', ascending=False).head(top_n)['gene'].tolist()
    stratified_lists['overall'] = overall_genes
    
    # Count cancer types in overall list
    overall_df = df_hubs[df_hubs['gene'].isin(overall_genes)]
    overall_bc = len(overall_df[overall_df['cancer_relevance'] == 'breast_cancer'])
    overall_c = len(overall_df[overall_df['cancer_relevance'] == 'cancer'])
    overall_nc = len(overall_df[overall_df['cancer_relevance'] == 'non_cancer'])
    
    logger.info(f"   Total genes: {len(overall_genes)}")
    logger.info(f"     Breast cancer: {overall_bc}")
    logger.info(f"     General cancer: {overall_c}")
    logger.info(f"     Non-cancer/unknown: {overall_nc}")
    
    # 2. Breast cancer genes (unbiased ranking)
    logger.info("2. Breast Cancer Genes (all genes annotated as breast_cancer)...")
    breast_cancer_mask = (df_hubs['cancer_relevance'] == 'breast_cancer')
    if breast_cancer_mask.any():
        breast_cancer_df = df_hubs[breast_cancer_mask].copy()
        # Take all breast cancer genes, not just top_n
        breast_cancer_genes = breast_cancer_df.sort_values('base_score', ascending=False)['gene'].tolist()
        stratified_lists['breast_cancer'] = breast_cancer_genes
        logger.info(f"   Total genes: {len(breast_cancer_genes)} (all breast_cancer annotated)")
        logger.info(f"     Top 10 by base_score:")
        for i, gene in enumerate(breast_cancer_genes[:10], 1):
            symbol = gene.split('|')[1] if '|' in gene else gene
            score = breast_cancer_df[breast_cancer_df['gene'] == gene]['base_score'].values[0]
            logger.info(f"       {i:2d}. {symbol:15s} (score: {score:.2f})")
    else:
        stratified_lists['breast_cancer'] = []
        logger.warning("   No breast cancer genes found")
    
    # 3. General cancer genes (unbiased ranking)
    logger.info("3. General Cancer Genes (all genes annotated as cancer)...")
    cancer_mask = (df_hubs['cancer_relevance'] == 'cancer')
    if cancer_mask.any():
        cancer_df = df_hubs[cancer_mask].copy()
        # Take top_n or all, whichever is smaller
        cancer_genes = cancer_df.sort_values('base_score', ascending=False).head(top_n)['gene'].tolist()
        stratified_lists['general_cancer'] = cancer_genes
        logger.info(f"   Total genes: {len(cancer_genes)} (top {min(top_n, len(cancer_df))} of {len(cancer_df)} cancer genes)")
    else:
        stratified_lists['general_cancer'] = []
        logger.warning("   No general cancer genes found")
    
    # 4. Novel predictive genes (high scoring but not cancer-annotated)
    logger.info("4. Novel Predictive Genes (high-scoring non-cancer/unknown genes)...")
    novel_mask = (df_hubs['cancer_relevance'] == 'non_cancer')
    if novel_mask.any():
        novel_df = df_hubs[novel_mask].copy()
        # Rank by base_score (unbiased)
        novel_genes = novel_df.sort_values('base_score', ascending=False).head(top_n)['gene'].tolist()
        stratified_lists['novel_predictive'] = novel_genes
        logger.info(f"   Total genes: {len(novel_genes)}")
        logger.info(f"     These are high-scoring genes NOT annotated as cancer")
        logger.info(f"     May represent novel discoveries or unannotated genes")
        logger.info(f"     Top 5:")
        for i, gene in enumerate(novel_genes[:5], 1):
            symbol = gene.split('|')[1] if '|' in gene else gene
            score = novel_df[novel_df['gene'] == gene]['base_score'].values[0]
            logger.info(f"       {i}. {symbol:15s} (score: {score:.2f})")
    else:
        stratified_lists['novel_predictive'] = []
        logger.warning("   No novel predictive genes found")
    
    # 5. Network-rewired genes (significant delta connectivity)
    logger.info("5. Network-Rewired Genes (significant connectivity changes)...")
    rewired_mask = df_hubs['is_rewired']
    if rewired_mask.any():
        rewired_df = df_hubs[rewired_mask].copy()
        # Rank by base_score (unbiased)
        rewired_genes = rewired_df.sort_values('base_score', ascending=False).head(top_n)['gene'].tolist()
        stratified_lists['network_rewired'] = rewired_genes
        
        # Count cancer types in ALL REWIRED GENES (before filtering to top 250)
        all_rewired_bc = len(rewired_df[rewired_df['cancer_relevance'] == 'breast_cancer'])
        all_rewired_c = len(rewired_df[rewired_df['cancer_relevance'] == 'cancer'])
        all_rewired_nc = len(rewired_df[rewired_df['cancer_relevance'] == 'non_cancer'])
        
        # Count cancer types in THE TOP 250 REWIRED LIST
        cancer_counts = {'breast_cancer': 0, 'cancer': 0, 'non_cancer': 0}
        for gene in rewired_genes:
            gene_row = df_hubs[df_hubs['gene'] == gene]
            if not gene_row.empty:
                relevance = gene_row.iloc[0]['cancer_relevance']
                cancer_counts[relevance] = cancer_counts.get(relevance, 0) + 1
        
        rewired_bc = cancer_counts['breast_cancer']
        rewired_c = cancer_counts['cancer']
        rewired_nc = cancer_counts['non_cancer']
        
        logger.info(f"   Total rewired genes: {len(rewired_df)} (with significant connectivity change)")
        logger.info(f"     All rewired - Breast cancer: {all_rewired_bc}, Cancer: {all_rewired_c}, Non-cancer: {all_rewired_nc}")
        logger.info(f"   Top {top_n} by base_score:")
        logger.info(f"     Selected - Breast cancer: {rewired_bc}, Cancer: {rewired_c}, Non-cancer: {rewired_nc}")
        
        # Show if breast cancer genes were filtered out
        if all_rewired_bc > rewired_bc:
            logger.info(f"   ⚠️  Note: {all_rewired_bc - rewired_bc} breast cancer genes excluded from top {top_n} (ranked lower)")
        if all_rewired_c > rewired_c:
            logger.info(f"   ⚠️  Note: {all_rewired_c - rewired_c} cancer genes excluded from top {top_n} (ranked lower)")

    else:
        stratified_lists['network_rewired'] = []
        logger.warning("   No significantly rewired genes found")
    
    # Calculate overlap statistics
    logger.info("\nCalculating overlap statistics...")
    overlap_stats = calculate_overlap_statistics(stratified_lists)
    
    # Create comprehensive output
    output = {
        'stratified_lists': stratified_lists,
        'metadata': {
            'generated_on': time.strftime('%Y-%m-%d %H:%M:%S'),
            'top_n_per_list': top_n,
            'total_genes_analyzed': len(df_hubs),
            'cancer_annotation_stats': {
                'breast_cancer': breast_cancer_count,
                'general_cancer': general_cancer_count,
                'non_cancer_or_unknown': non_cancer_count
            },
            'overlap_statistics': overlap_stats,
            'list_descriptions': {
                'overall': f'Top {top_n} genes using enhanced_score (with 1.2x multiplier for breast cancer and known BRCA genes)',
                'breast_cancer': f'All {len(stratified_lists.get("breast_cancer", []))} genes annotated as breast_cancer, ranked by base_score (unbiased)',
                'general_cancer': f'Top {len(stratified_lists.get("general_cancer", []))} genes annotated as general cancer, ranked by base_score',
                'novel_predictive': f'Top {top_n} high-scoring genes NOT annotated as cancer (may be novel discoveries or unannotated)',
                'network_rewired': f'Top {top_n} genes with significant rewiring (|delta| > threshold) ranked by base_score'
            },
            'notes': [
                'Breast cancer genes: Genes from curated lists (COSMIC, TCGA, ClinVar) + network discoveries',
                'General cancer genes: Pan-cancer genes from curated databases',
                'Non-cancer/unknown: Includes explicitly non-cancer genes AND unannotated genes',
                'Novel predictive: High-scoring genes without cancer annotation - potential discoveries',
                'Network-rewired: Genes with significant connectivity changes between tumor/normal'
            ]
        }
    }
    
    total_unique = len(set().union(*[set(lst) for lst in stratified_lists.values()]))
    logger.info(f"\n✓ Created 5 stratified lists with {total_unique} unique genes")
    logger.info("="*60 + "\n")
    
    return output


def calculate_overlap_statistics(stratified_lists):
    """Calculate overlap between different gene lists."""
    stats = {}
    
    # Convert to sets for efficient comparison
    list_sets = {name: set(genes) for name, genes in stratified_lists.items()}
    
    # Calculate pairwise overlaps
    list_names = list(list_sets.keys())
    for i, name1 in enumerate(list_names):
        for name2 in list_names[i+1:]:
            set1 = list_sets[name1]
            set2 = list_sets[name2]
            
            if set1 and set2:
                overlap = len(set1.intersection(set2))
                union = len(set1.union(set2))
                jaccard = overlap / union if union > 0 else 0
                
                key = f"{name1}_vs_{name2}"
                stats[key] = {
                    'overlap_count': overlap,
                    'overlap_percentage': (overlap / len(set1) * 100) if len(set1) > 0 else 0,
                    'jaccard_similarity': jaccard
                }
    
    # Calculate unique genes per list
    for name, gene_set in list_sets.items():
        other_sets = [s for n, s in list_sets.items() if n != name]
        if other_sets:
            # Genes only in this list
            unique_genes = gene_set - set().union(*other_sets)
            stats[f'{name}_unique'] = len(unique_genes)
        else:
            stats[f'{name}_unique'] = len(gene_set)
    
    return stats


def generate_hub_visualizations(df_hubs, output_dir, logger):
    """
    Generate comprehensive static visualizations for hub analysis.
    
    OUTPUTS:
    - Enhanced hub score ranking (bar plot)
    - Hybrid metrics scatter plot (FI vs Delta)
    - Cancer relevance distribution (pie chart)
    """
    viz_dir = ensure_dir(output_dir / 'viz')
    
    if df_hubs.empty:
        logger.warning("No hub data available for visualization")
        return []
    
    top_50 = df_hubs.head(50)
    
    logger.info("Generating hub visualization plots...")
    
    # 1. Enhanced Score Ranking Plot
    plt.figure(figsize=(12, 8))
    top_20 = top_50.head(20).sort_values('enhanced_score', ascending=True)
    
    # Use gene_symbol for better readability
    y_labels = [row['gene_symbol'] if 'gene_symbol' in df_hubs.columns else row['gene'] for _, row in top_20.iterrows()]
    
    # Color by cancer relevance
    colors = {'breast_cancer': '#e74c3c', 'cancer': '#e67e22', 'non_cancer': '#3498db'}
    bar_colors = [colors.get(row['cancer_relevance'], '#95a5a6') for _, row in top_20.iterrows()]
    
    plt.barh(range(len(top_20)), top_20['enhanced_score'], color=bar_colors, alpha=0.8)
    plt.yticks(range(len(top_20)), y_labels)
    plt.xlabel('Enhanced Hub Score')
    plt.title('Top 20 Enhanced Hubs by Integrated Score\n(Color-coded by Cancer Relevance)')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[rel], label=rel.replace('_', ' ').title()) 
                      for rel in colors.keys()]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    score_plot_path = viz_dir / 'enhanced_hub_scores.png'
    plt.savefig(score_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Hybrid Metrics Scatter Plot - Feature Importance vs Delta Connectivity
    has_fi_data = (df_hubs['feature_importance'] > 0).sum() > 10
    has_delta_data = (df_hubs['delta_connectivity'] != 0).sum() > 10
  
    if has_fi_data and has_delta_data:
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(top_50['feature_importance'], top_50['delta_connectivity'], 
                             c=top_50['normalized_score'], cmap='viridis', 
                             s=50, alpha=0.7)
        plt.colorbar(scatter, label='Normalized Enhanced Score')
        plt.xlabel('Feature Importance (Classification Power)')
        plt.ylabel('Delta Connectivity (Network Rewiring)')
        plt.title('Hybrid Hub Metrics: Predictive Power vs Network Disruption')
        plt.grid(True, alpha=0.3)
      
        scatter_path = viz_dir / 'hub_metrics_scatter_hybrid.png'
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ Generated hybrid scatter plot")
    else:
        scatter_path = None
        logger.info(f"Skipping scatter plot - FI data: {has_fi_data}, Delta data: {has_delta_data}")
    
    # 3. Cancer Relevance Distribution
    plt.figure(figsize=(8, 6))
    relevance_counts = top_50['cancer_relevance'].value_counts()
    colors_list = [colors.get(rel, '#95a5a6') for rel in relevance_counts.index]
    plt.pie(relevance_counts.values, labels=relevance_counts.index, colors=colors_list,
            autopct='%1.1f%%', startangle=90)
    plt.title('Cancer Relevance Distribution in Top 50 Hubs')
    pie_path = viz_dir / 'cancer_relevance_pie.png'
    plt.savefig(pie_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    viz_paths = [score_plot_path, pie_path]
    if scatter_path:
        viz_paths.append(scatter_path)
    
    logger.info(f"✓ Generated {len(viz_paths)} visualization plots") # in {viz_dir}
    
    return viz_paths


def create_driver_candidates(df_hubs, config, logger):
    """
    Identify and annotate potential driver candidates for downstream analysis.
    
    OUTPUT: Driver candidates JSON for 03_b_functional_characterization
    """
    if df_hubs.empty:
        logger.warning("No hub data available for driver candidate identification")
        return []
    
    # Use 'top_hubs_count' from the config to provide a larger, more statistically
    # robust gene list for the downstream enrichment analysis.
    top_n = config.get('hub_analysis', {}).get('top_hubs_count', 100)
    top_hubs = df_hubs.head(top_n)
    
    # Load gene info for detailed annotations
    logger.info("Loading gene information for driver candidate annotation...")
    gene_dict = load_combined_gene_info(config)
    
    driver_candidates = []
    logger.info(f"Annotating top {len(top_hubs)} driver candidates...")
    
    for _, hub in tqdm(top_hubs.iterrows(), total=len(top_hubs), desc="Annotating drivers"):
        gene_info = get_gene_info(hub['gene'], config, gene_dict)
        
        candidate = {
            'gene': hub['gene'],
            'rank': int(hub['rank']),
            'enhanced_score': float(hub['enhanced_score']),
            'degree': float(hub['degree']),
            'delta_connectivity': float(hub['delta_connectivity']),
            'feature_importance': float(hub['feature_importance']),
            'cancer_relevance': hub['cancer_relevance'],
            'is_rewired': hub['is_rewired'],
            'scoring_method': hub['scoring_method'],
            'driver_confidence': 'high' if hub['cancer_relevance'] == 'breast_cancer' else 'medium' if hub['cancer_relevance'] == 'cancer' else 'low'
        }
        
        # Add gene info if available
        if gene_info and gene_info.get('gene_info'):
            candidate.update({
                'gene_name': gene_info['gene_info'].get('name', ''),
                'gene_type': gene_info['gene_info'].get('type', ''),
                'description': gene_info['gene_info'].get('description', '')[:200] + '...'  # Truncate
            })
        
        driver_candidates.append(candidate)
    
    logger.info(f"✓ Identified {len(driver_candidates)} driver candidates")
    return driver_candidates


def dict_calc_uniq_count(dict_or_arr, field=None):
    """
    Calculate unique value counts from a dictionary or array.
    
    Args:
        dict_or_arr: Dictionary (values only) or list of dicts
        field: If dict_or_arr is list of dicts, extract this field
    
    Returns:
        dict: {value: count} for all unique values
    
    Examples:
        >>> dict_calc_uniq_count({'g1': 'cancer', 'g2': 'non_cancer', 'g3': 'cancer'})
        {'cancer': 2, 'non_cancer': 1}
        
        >>> dict_calc_uniq_count([{'gene': 'A', 'rel': 'cancer'}, {'gene': 'B', 'rel': 'cancer'}], field='rel')
        {'cancer': 2}
    """
    counts = {}
    
    # Handle dictionary (values only)
    if isinstance(dict_or_arr, dict):
        values = dict_or_arr.values()
    # Handle list of dicts with field extraction
    elif isinstance(dict_or_arr, list) and field:
        values = [item.get(field, 'unknown') for item in dict_or_arr if isinstance(item, dict)]
    # Handle simple list/array
    elif isinstance(dict_or_arr, (list, tuple)):
        values = dict_or_arr
    else:
        return {}
    
    # Count occurrences
    for val in values:
        counts[val] = counts.get(val, 0) + 1
    
    return counts


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


def main():
    """
    Orchestrate enhanced hub analysis with hybrid scoring using consensus data.
    
    EXECUTION PHASES:
    1. Load pre-computed centralities from 01_b (Network Topology)
    2. Load DCEA enhancements from 02_b (Network Rewiring/Delta)
    3. Load Consensus Classification features from 02_c (Predictive Power)
    4. Compute hybrid enhanced scores (Integration Phase)
    5. Generate comprehensive visualizations
    6. Identify driver candidates for functional characterization
    7. Save results with full methodology provenance (fi_source)
    """
    start_time = time.time()
    config = load_config()
    PROJECT_ROOT = Path(config['paths']['project_root'])
    INPUT_NETWORKS = Path(config['paths']['networks'])
    OUTPUT_DIR = get_auto_output_path(__file__, PROJECT_ROOT)
    ensure_dir(OUTPUT_DIR)
    
    logger = setup_logging(config, OUTPUT_DIR)
    logger.info("Starting 03_a_enhanced_hub_analysis.py")
    logger.info("=" * 60)
    
    # Initialize variables to avoid UnboundLocalError
    df_hubs = pd.DataFrame()
    driver_candidates = []
    viz_paths = []
    top_n_for_enrichment = config.get('hub_analysis', {}).get('top_hubs_count', 250)
    
    # Summary structure for tracking analysis results and methodology provenance
    summary_stats = {
        'script': '03_a_enhanced_hub_analysis',
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'parameters': {
            'primary_method': config['network_analysis']['primary_correlation_method'],
            'threshold': config['network_analysis']['correlation_thresholds'][1],  # 0.7 Balanced
            'top_hubs_count': config.get('hub_analysis', {}).get('top_hubs_count', 100),
            'hybrid_weights': {
                'feature_importance': config['hub_analysis']['feature_importance_weight'],
                'delta_connectivity': config['hub_analysis']['delta_connectivity_weight']
            }
        },
        'inputs': {},
        'outputs': {},
        'rq_metrics': [
            {
                "metric": "top_enhanced_hub", 
                "value": "",
                "interpretation": "Gene with highest integrated enhanced score",
                "biological_context": "Represents the most influential network node considering both predictive power and network disruption"
            },
            {
                "metric": "cancer_driver_overlap_pct", 
                "value": 0.0,
                "interpretation": "Percentage of top hubs that are known cancer genes",
                "biological_context": "High overlap suggests cancer genes dominate network control positions in BRCA"
            },
            {
                "metric": "validated_hubs_pct", 
                "value": 0.0,
                "interpretation": "Percentage of top hubs with validated classification importance",
                "biological_context": "High percentage indicates strong alignment between network topology and predictive power"
            },
            {
                "metric": "rewired_hubs_pct", 
                "value": 0.0,
                "interpretation": "Percentage of top hubs with significant network rewiring",
                "biological_context": "Indicates extent of network restructuring in cancer state"
            }
        ],
        'processing_notes': {}
    }
    
    try:
        # PHASE 1: Load pre-computed centralities from 01_b (Topology)
        logger.info("\n\nPhase 1: Loading pre-computed network data...")
        centralities = load_precomputed_centralities(config, PROJECT_ROOT, logger)
        summary_stats['inputs']['network_metrics'] = str(
            (Path(PROJECT_ROOT) / config['paths']['networks'] / 'global_metrics_comparison.json').relative_to(PROJECT_ROOT)
        )
        
        # PHASE 2: Load DCEA enhancements from 02_b (Rewiring)
        logger.info("\n\nPhase 2: Loading DCEA rewiring data...")
        delta_dict, cancer_relevance = load_dcea_enhancements(config, PROJECT_ROOT, logger)
        summary_stats['inputs']['dcea_data'] = str(
            (Path(PROJECT_ROOT) / config['paths']['dcea_viz_enrich']).relative_to(PROJECT_ROOT)
        )

        # === DEBUG: Check 02_b cancer_relevance ===
        cr_02b_counts = dict_calc_uniq_count(cancer_relevance)
        logger.info(f"DEBUG: Cancer relevance from 02_b ({len(cancer_relevance)} genes):")
        for rel, count in sorted(cr_02b_counts.items()):
            logger.info(f"  {rel:20s}: {count:3d} ({count/len(cancer_relevance)*100:.1f}%)")
        # === END DEBUG ===
      

        # PHASE 2b: Load Consensus Classification features from 02_c (Predictive)
        # UPDATED: Uses the consensus JSON to reduce sampling bias and improve robustness
        logger.info("\n\nPhase 2b: Loading consensus classification features...")
        feature_importance_dict, fi_metadata, consensus_cancer_relevance = load_classification_features(config, PROJECT_ROOT, logger)
        

        # === DEBUG: Check consensus cancer_relevance ===
        if consensus_cancer_relevance:
            cr_consensus_counts = dict_calc_uniq_count(consensus_cancer_relevance)
            logger.info(f"DEBUG: Cancer relevance from consensus ({len(consensus_cancer_relevance)} genes):")
            for rel, count in sorted(cr_consensus_counts.items()):
                logger.info(f"  {rel:20s}: {count:3d} ({count/len(consensus_cancer_relevance)*100:.1f}%)")
            
            # Show some specific genes
            logger.info("DEBUG: Sample consensus genes:")
            sample_genes = list(consensus_cancer_relevance.items())[:5]
            for gene, rel in sample_genes:
                symbol = gene.split('|')[1] if '|' in gene else gene
                logger.info(f"  {symbol:15s} → {rel}")
        else:
            logger.warning("DEBUG: No consensus cancer_relevance loaded!")
        # === END DEBUG ===

        if consensus_cancer_relevance:
            logger.info(f"✓ Overriding cancer_relevance from consensus for {len(consensus_cancer_relevance)} genes")

             # === DEBUG: Before merge ===
            before_counts = dict_calc_uniq_count(cancer_relevance)
            logger.info(f"DEBUG: BEFORE merge ({len(cancer_relevance)} genes):")
            for rel, count in sorted(before_counts.items()):
                logger.info(f"  {rel:20s}: {count:3d}")
            # === END DEBUG ===
            
            cancer_relevance.update(consensus_cancer_relevance)  # Merge with priority to consensus

            # === DEBUG: After merge ===
            after_counts = dict_calc_uniq_count(cancer_relevance)
            logger.info(f"DEBUG: AFTER merge ({len(cancer_relevance)} genes):")
            for rel, count in sorted(after_counts.items()):
                logger.info(f"  {rel:20s}: {count:3d}")
            
            # Check specific breast cancer genes
            test_genes = ['ENSG00000126351.13|THRA', 'ENSG00000078053.17|AMPH', 
                         'ENSG00000143878.10|RHOB', 'ENSG00000108861.9|DUSP3']
            logger.info("DEBUG: Checking specific breast cancer genes after merge:")
            for gene_id in test_genes:
                symbol = gene_id.split('|')[1]
                rel = cancer_relevance.get(gene_id, 'NOT FOUND')
                logger.info(f"  {symbol:15s} ({gene_id[:20]}...) → {rel}")
            # === END DEBUG ===


        if feature_importance_dict:
            summary_stats['inputs']['classification_features'] = fi_metadata.get('source_file', 'unknown')
            summary_stats['parameters']['fi_source'] = fi_metadata.get('source_type', 'unknown')
        else:
            summary_stats['parameters']['fi_source'] = 'none'
            summary_stats['processing_notes']['fi_fallback'] = 'Using network-only scoring'
        

        # PHASE 3: Compute Hybrid Enhanced Scores
        # UPDATED: Passes fi_metadata to track the exact source (e.g., median+cluster)
        logger.info("\n\nPhase 3: Computing hybrid hub scores...")
        df_hubs = compute_enhanced_scores(
            centralities, delta_dict, cancer_relevance, 
            feature_importance_dict, fi_metadata, 
            config, logger
        )
        
        # === DEBUG: Check df_hubs cancer_relevance ===
        if not df_hubs.empty:
            hub_cr_counts = dict_calc_uniq_count(df_hubs.to_dict('records'), field='cancer_relevance')
            logger.info(f"DEBUG: Cancer relevance in df_hubs ({len(df_hubs)} genes):")
            for rel, count in sorted(hub_cr_counts.items()):
                logger.info(f"  {rel:20s}: {count:3d} ({count/len(df_hubs)*100:.1f}%)")
            
            # Show top 10 genes
            logger.info("DEBUG: Top 10 genes in df_hubs:")
            for _, row in df_hubs.head(10).iterrows():
                logger.info(f"  Rank {row['rank']:3d}: {row['gene_symbol']:15s} → {row['cancer_relevance']:15s} (score: {row['enhanced_score']:.2f})")
        # === END DEBUG ===


        # PHASE 4: Generate Visualizations
        logger.info("\n\nPhase 4: Generating visualizations...")
        if not df_hubs.empty:
            viz_paths = generate_hub_visualizations(df_hubs, OUTPUT_DIR, logger)
        
        # PHASE 5: Identify Driver Candidates
        logger.info("\n\nPhase 5: Identifying driver candidates...")
        if not df_hubs.empty:
            driver_candidates = create_driver_candidates(df_hubs, config, logger)
        
        # PHASE 6: Save Results with Provenance Tracking
        logger.info("\n\nPhase 6: Saving structured results...")
        
        hub_ranking_path = OUTPUT_DIR / 'enhanced_hub_ranking.tsv'
        if not df_hubs.empty:
            # Reorder columns to put ranking and provenance info at the front
            column_order = [
                'rank', 'gene', 'gene_symbol', 'enhanced_score', 
                'fi_source', 'scoring_method', 'degree', 'betweenness', 
                'delta_connectivity', 'feature_importance', 'cancer_relevance', 
                'is_rewired', 'normalized_score'
            ]
            
            # ISSUE 5 FIX: Verify critical columns exist before saving (Refinement Logic)
            critical_cols = ['rank', 'gene', 'enhanced_score', 'fi_source']
            missing_critical = [c for c in critical_cols if c not in df_hubs.columns]
            if missing_critical:
                logger.error(f"Critical columns missing from hub data: {missing_critical}")
                raise ValueError(f"Cannot save output - missing: {missing_critical}")
            
            # Filter to ensure we only include columns that were actually generated
            final_cols = [c for c in column_order if c in df_hubs.columns]
            df_hubs[final_cols].to_csv(hub_ranking_path, sep='\t', index=False)
            logger.info(f"✓ Saved prioritized hub ranking to: {hub_ranking_path.name}")
        else:
            # Emergency fallback: save empty structure with headers
            pd.DataFrame(columns=['rank', 'gene', 'enhanced_score', 'fi_source']).to_csv(hub_ranking_path, sep='\t', index=False)
        

        # Driver candidates JSON (for functional enrichment in 03_b)
        driver_path = OUTPUT_DIR / 'driver_candidates.json'
        with open(driver_path, 'w') as f:
            json.dump(driver_candidates, f, indent=2)

        # Top hubs specifically for GO/Pathway enrichment
        top_hubs_path = OUTPUT_DIR / 'top_hubs_for_enrichment.json'
        top_genes = [hub['gene'] for hub in driver_candidates[:top_n_for_enrichment]] if driver_candidates else []
        with open(top_hubs_path, 'w') as f:
            json.dump(top_genes, f, indent=2)

        # === NEW: Export stratified lists for 03_b dual enrichment ===
        logger.info("\nExporting stratified gene lists for functional characterization...")

        # Step 1: Get top 250 predictive genes FIRST
        top_250_all = df_hubs.sort_values('enhanced_score', ascending=False).head(250)
        
        # Step 2: Split the SAME 250 genes into cancer vs novel subsets
        cancer_genes = top_250_all[
            top_250_all['cancer_relevance'].isin(['breast_cancer', 'cancer'])
        ].copy()
        
        novel_genes = top_250_all[
            top_250_all['cancer_relevance'] == 'non_cancer'
        ].copy()
        
        # Step 3: Validation - ensure subsets sum to total
        assert len(cancer_genes) + len(novel_genes) == len(top_250_all), \
            f"Subset mismatch: {len(cancer_genes)} + {len(novel_genes)} != {len(top_250_all)}"
        
        logger.info(f"\nStratified gene lists (subsets of top 250):")
        logger.info(f"  Total top predictive genes: {len(top_250_all)}")
        logger.info(f"  Cancer subset: {len(cancer_genes)} ({len(cancer_genes)/len(top_250_all)*100:.1f}%)")
        logger.info(f"  Novel subset: {len(novel_genes)} ({len(novel_genes)/len(top_250_all)*100:.1f}%)")
        logger.info(f"  ✓ Validation: {len(cancer_genes)} + {len(novel_genes)} = {len(top_250_all)}")
        
        # Step 4: Save all three lists
        # List 1: All top 250 (mixed)
        top_250_path = OUTPUT_DIR / 'top_250_all_genes.json'
        top_250_all[['gene', 'gene_symbol', 'enhanced_score', 'cancer_relevance', 'delta_connectivity']].to_json(
            top_250_path, orient='records', indent=2
        )
        logger.info(f"\n✓ Saved top 250 all genes: {top_250_path.name}")
        
        # List 2: Cancer subset (for validation)
        cancer_genes_path = OUTPUT_DIR / 'cancer_genes_for_enrichment.json'
        cancer_genes[['gene', 'gene_symbol', 'enhanced_score', 'cancer_relevance', 'delta_connectivity']].to_json(
            cancer_genes_path, orient='records', indent=2
        )
        logger.info(f"✓ Saved cancer gene subset: {cancer_genes_path.name}")
        
        # List 3: Novel subset (for discovery)
        novel_genes_path = OUTPUT_DIR / 'novel_genes_for_enrichment.json'
        novel_genes[['gene', 'gene_symbol', 'enhanced_score', 'cancer_relevance', 'delta_connectivity']].to_json(
            novel_genes_path, orient='records', indent=2
        )
        logger.info(f"✓ Saved novel gene subset: {novel_genes_path.name}")
        
        # Step 5: Log sample genes for verification
        logger.info(f"\nSample genes from each list:")
        logger.info(f"  All (top 5): {top_250_all['gene_symbol'].head(5).tolist()}")
        logger.info(f"  Cancer (top 5): {cancer_genes['gene_symbol'].head(5).tolist() if len(cancer_genes) >= 5 else cancer_genes['gene_symbol'].tolist()}")
        logger.info(f"  Novel (top 5): {novel_genes['gene_symbol'].head(5).tolist() if len(novel_genes) >= 5 else novel_genes['gene_symbol'].tolist()}")
        
        # Calculate RQ metrics for analysis validation
        if not df_hubs.empty:
            top_50 = df_hubs.head(50)
            cancer_in_top_50 = top_50[top_50['cancer_relevance'].isin(['breast_cancer', 'cancer'])]
            
            summary_stats['rq_metrics'][0]['value'] = df_hubs.iloc[0]['gene']
            
            # safety: If top_50 has fewer than 50 rows, adjust calculation
            n_top = len(top_50)
            summary_stats['rq_metrics'][1]['value'] = float(len(cancer_in_top_50) / n_top * 100) if n_top > 0 else 0.0
            summary_stats['rq_metrics'][2]['value'] = float((top_50['feature_importance'] > 0).sum() / n_top * 100) if n_top > 0 else 0.0
            summary_stats['rq_metrics'][3]['value'] = float(top_50['is_rewired'].sum() / n_top * 100) if n_top > 0 else 0.0
        
        # Final output tracking in summary JSON
        summary_stats['outputs'] = {
            'enhanced_hub_ranking': str(hub_ranking_path.relative_to(PROJECT_ROOT)),
            'driver_candidates': str(driver_path.relative_to(PROJECT_ROOT)),
            'top_hubs_for_enrichment': str(top_hubs_path.relative_to(PROJECT_ROOT))
        }
        for viz_path in viz_paths:
            summary_stats['outputs'][viz_path.name] = str(viz_path.relative_to(PROJECT_ROOT))
        

        # PHASE 7: Create stratified gene lists
        logger.info("\n\nPhase 7: Creating stratified gene lists...")
        stratified_output = create_stratified_lists(df_hubs, config, logger)

        # Save stratified lists
        stratified_path = OUTPUT_DIR / 'stratified_gene_lists.json'
        with open(stratified_path, 'w') as f:
            json.dump(stratified_output, f, indent=2)
        logger.info(f"✓ Saved stratified gene lists to: {stratified_path.name}")

        # Add to summary_stats
        summary_stats['outputs']['stratified_gene_lists'] = str(stratified_path.relative_to(PROJECT_ROOT))

        # Log final synthesis
        logger.info("=" * 60)
        logger.info("\n\nHYBRID HUB ANALYSIS COMPLETE")
        if not df_hubs.empty:
            logger.info(f"Top Hub: {df_hubs.iloc[0]['gene_symbol']} (Source: {df_hubs.iloc[0]['fi_source']})")
            logger.info(f"Known Cancer Driver Overlap: {summary_stats['rq_metrics'][1]['value']:.1f}%")



        # ========================================================================
        # CONSENSUS HUB VISUALIZATIONS
        # ========================================================================
        logger.info("\n" + "="*80)
        logger.info("PHASE 8: GENERATING CONSENSUS HUB VISUALIZATIONS")
        logger.info("="*80)

        # Load all four networks using the global threshold
        logger.info("Loading all required networks...")
        pickle_dir = INPUT_NETWORKS / "pickle"
        
        networks = {
            'spearman_tumor': safe_read_pickle(pickle_dir / f"tumor_network_spearman_{CORRELATION_THRESHOLD}.pkl", logger),
            'spearman_normal': safe_read_pickle(pickle_dir / f"normal_network_spearman_{CORRELATION_THRESHOLD}.pkl", logger),
         }
        
        # Create output directory
        consensus_hub_dir = OUTPUT_DIR / 'consensus_hubs'
        consensus_hub_dir.mkdir(exist_ok=True)

        # Load enhanced hub ranking
        logger.info(f"\nLoading enhanced hub ranking from: {hub_ranking_path.name}")
        hub_ranking = pd.read_csv(hub_ranking_path, sep='\t')
        logger.info(f"  Total hubs in ranking: {len(hub_ranking)}")

        # Load networks (should already be loaded earlier in script)
        # If not, you'll need to load them here
        try:
            spearman_normal = networks['spearman_normal']
            spearman_tumor = networks['spearman_tumor']
            logger.info(f"  Networks loaded: Normal={spearman_normal.number_of_nodes()} nodes, "
                        f"Tumor={spearman_tumor.number_of_nodes()} nodes")
        except (NameError, KeyError) as e:
            logger.error(f"  ✗ Networks not available: {e}")
            logger.error("  Skipping consensus hub visualizations")
            # Exit this section gracefully
        else:
            # Load gene info for detailed annotations
            from utils.genes import load_combined_gene_info
            combined_gene_data = load_combined_gene_info(config)
            
            # Select 9 consensus hubs
            logger.info("\nSelecting consensus hubs (3 overall + 3 BC + 3 cancer)...")
            selected_hubs = select_consensus_hubs(
                logger,
                hub_ranking_df=hub_ranking,
                config=config,
                combined_gene_data=combined_gene_data,
                top_100=False,
                normal_network=spearman_normal,  # ADD
                tumor_network=spearman_tumor      # ADD
            )
            
            if len(selected_hubs) == 0:
                logger.warning("  ⚠️  No hubs selected! Skipping visualizations.")
            else:
                logger.info(f"\nGenerating {len(selected_hubs)} paired visualizations...")
                
                all_metadata = []
                
                for i, hub_info in enumerate(selected_hubs, 1):
                    gene_id = hub_info['gene_id']
                    symbol = hub_info['gene_symbol']
                    category = hub_info['category']
                    rank = hub_info['rank_in_category']
                    category_label = hub_info['category_label']
                    
                    # Construct filename
                    output_filename = f"{i:02d}_{category}_{rank}_{symbol}_paired.png"
                    output_path = consensus_hub_dir / output_filename
                    
                    logger.info(f"  {i}/{len(selected_hubs)}: {symbol} ({category_label} #{rank})...")
                    
                    try:
                        # Create paired visualization
                        viz_metadata = create_paired_hub_comparison(
                            normal_network=spearman_normal,
                            tumor_network=spearman_tumor,
                            hub_gene=gene_id,
                            output_path=output_path,
                            threshold=CORRELATION_THRESHOLD,
                            dpi=300,
                            hub_info=hub_info
                        )
                        
                        # Merge hub info with visualization metadata
                        combined_metadata = {
                            **hub_info,
                            'visualization': viz_metadata,
                            'output_file': str(output_path.relative_to(OUTPUT_DIR.parent))
                        }
                        
                        all_metadata.append(combined_metadata)
                        
                        logger.info(f"      ✓ Normal degree: {viz_metadata.get('normal_degree', 'N/A')}, "
                                f"Tumor degree: {viz_metadata.get('tumor_degree', 'N/A')}")
                        
                    except Exception as e:
                        logger.error(f"      ✗ Failed to create visualization: {e}")
                        continue
                
                # Save consolidated metadata
                metadata_json_path = consensus_hub_dir / 'consensus_hubs_metadata.json'
                
                metadata_output = {
                    'description': 'Consensus hub visualizations for top-ranked regulatory genes',
                    'selection_criteria': {
                        'source': 'enhanced_hub_ranking.tsv (top 100)',
                        'categories': {
                            'overall': 'Top 3 highest composite scores (any tier)',
                            'breast_cancer': 'Top 3 highest Tier 1 (breast cancer) genes',
                            'cancer': 'Top 3 highest Tier 2 (general cancer) genes'
                        },
                        'total_selected': len(selected_hubs),
                        'total_visualized': len(all_metadata)
                    },
                    'correlation_threshold': CORRELATION_THRESHOLD,
                    'networks': {
                        'normal': {
                            'nodes': spearman_normal.number_of_nodes(),
                            'edges': spearman_normal.number_of_edges()
                        },
                        'tumor': {
                            'nodes': spearman_tumor.number_of_nodes(),
                            'edges': spearman_tumor.number_of_edges()
                        }
                    },
                    'hubs': all_metadata
                }
                
                with open(metadata_json_path, 'w') as f:
                    json.dump(metadata_output, f, indent=2)
                
                logger.info(f"\n✓ Consensus hub visualizations complete!")
                logger.info(f"  Generated: {len(all_metadata)} paired visualizations")
                logger.info(f"  Output directory: {consensus_hub_dir.relative_to(OUTPUT_DIR.parent)}")
                logger.info(f"  Metadata: {metadata_json_path.name}")

                summary_stats['outputs']['consensus_hubs'] = {
                    'directory': str(consensus_hub_dir.relative_to(PROJECT_ROOT)),
                    'visualizations_generated': len(all_metadata),
                    'metadata_file': str(metadata_json_path.relative_to(PROJECT_ROOT))
                }

        logger.info("\n" + "="*80)
        logger.info("ENHANCED HUB ANALYSIS COMPLETE")
        logger.info("="*80)
        
        
    except Exception as e:
        logger.error(f"Analysis failed during Main Orchestration: {e}", exc_info=True)
        raise
    
    finally:
        # Save comprehensive metadata for the entire run
        summary_stats['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        total_time = time.time() - start_time
        summary_stats['processing_notes']['processing_time_minutes'] = round(total_time / 60, 1)
        
        result_info_path = OUTPUT_DIR / '03_a_result_info.json'
        create_summary_json(summary_stats, result_info_path, PROJECT_ROOT)
        logger.info(f"📄 Full analysis provenance saved to: {result_info_path.name}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()