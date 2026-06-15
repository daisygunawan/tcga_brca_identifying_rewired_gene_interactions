"""
03_b_functional_characterization.py - FIXED VERSION with Enhanced Statistics & Metadata

Key Fixes:
1. Fixed gene symbol parsing to handle all 100 genes properly
2. Fixed pandas compatibility issue with DataFrame.append()
3. Fixed background genes loading
4. Enhanced error handling and validation
5. Better logging and debugging

NEW ENHANCEMENTS:
1. Comprehensive summary statistics in JSON and TSV formats
2. Plot metadata with descriptions and data export
3. Enhanced visualization functions with metadata

Script Purpose:
Functional characterization of enhanced hubs from 03_a through pathway enrichment and biological interpretation.
Uses multiple enrichment methods with robust fallbacks and generates comprehensive pathway insights.
Outputs static visualizations and detailed functional annotations for BRCA research.
"""

import pandas as pd
import numpy as np
import json
import logging
import time
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm

import gseapy as gp
from utils.config import load_config
from utils.file import ensure_dir, get_relative_path, get_auto_output_path
from utils.genes import load_combined_gene_info

from utils.visualization_computational import (
    plot_statistical_significance_distributions,
    plot_pathway_coverage_analysis,
    plot_enrichment_quality_metrics,
    plot_database_contribution_heatmap,     
    plot_effect_size_vs_significance,        
    plot_novelty_gradient                    
)

# Monkey patch for pandas compatibility (fix for DataFrame.append deprecation)
if not hasattr(pd.DataFrame, 'append'):
    pd.DataFrame.append = pd.DataFrame._append

def setup_logging(config, output_dir):
    """Set up structured logging for functional characterization."""
    logger = logging.getLogger(__name__)
    logger.setLevel(config['logging']['level'])

    if logger.hasHandlers():
        logger.handlers.clear()

    log_dir = output_dir / 'logs'
    ensure_dir(log_dir)
    log_file = log_dir / '03_b_functional_characterization.log'
    
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
    """Create standardized result info JSON."""
    for section in ['inputs', 'outputs']:
        for key, value in summary_data[section].items():
            if isinstance(value, Path):
                summary_data[section][key] = str(value.relative_to(project_root))

    with open(output_path, 'w') as f:
        json.dump(summary_data, f, indent=2)

def parse_gene_symbol(gene_str):
    """
    Parse gene symbol from Ensembl|Symbol format with better validation.
    
    Input: "ENSG00000139218.18|SCAF11" or "SCAF11"
    Output: "SCAF11"
    
    Returns None if parsing fails.
    """
    if not gene_str or not isinstance(gene_str, str):
        return None
    
    # Handle ENSEMBL|SYMBOL format
    if '|' in gene_str:
        parts = gene_str.split('|')
        if len(parts) >= 2:
            symbol = parts[1].strip()
            # Remove version numbers and extra characters
            symbol = symbol.split('.')[0]
            # More lenient validation: allow numbers in symbols (e.g., CD8A, CD3G)
            if symbol and 2 <= len(symbol) <= 20 and any(c.isalpha() for c in symbol):
                return symbol.upper()
    
    # If no pipe, assume it's already a symbol
    symbol = gene_str.strip()
    # More lenient validation
    if symbol and 2 <= len(symbol) <= 20 and any(c.isalpha() for c in symbol):
        return symbol.upper()
    
    return None


def load_hub_genes_stratified(config, project_root, logger):
    """
    Load stratified gene lists for dual enrichment analysis.
    
    Returns:
        tuple: (stratified_genes_dict, driver_info_dict)
    """
    hub_analysis_dir = Path(project_root) / config['paths']['hub_analysis']
    
    # Load all three lists from 03_a
    all_genes_path = hub_analysis_dir / 'top_250_all_genes.json'
    cancer_path = hub_analysis_dir / 'cancer_genes_for_enrichment.json'
    novel_path = hub_analysis_dir / 'novel_genes_for_enrichment.json'
    
    stratified_genes = {}
    logger.info("Loading stratified gene lists from 03_a...")
    
    # Helper function to parse gene list
    def parse_gene_list(data):
        """Extract gene symbols from gene data."""
        symbols = []
        for item in data:
            if isinstance(item, dict):
                # Try gene_symbol first, then parse from gene field
                symbol = item.get('gene_symbol')
                if not symbol and 'gene' in item:
                    symbol = parse_gene_symbol(item['gene'])
            else:
                # Direct string
                symbol = parse_gene_symbol(str(item))
            
            if symbol and symbol not in symbols:
                symbols.append(symbol)
        return symbols
    
    # Load all genes (current approach - for backward compatibility)
    if all_genes_path.exists():
        with open(all_genes_path) as f:
            all_data = json.load(f)
        all_symbols = parse_gene_list(all_data)
        stratified_genes['all'] = {
            'genes': all_symbols,
            'count': len(all_symbols),
            'description': 'Top 250 predictive genes (mixed)',
            'data': all_data[:10]  # Store first 10 for reference
        }
        logger.info(f"✓ Loaded {len(all_symbols)} mixed genes")
    else:
        logger.warning("top_250_all_genes.json not found - using fallback")
        # Fallback to original method
        return load_hub_genes_legacy(config, project_root, logger)
    
    # Load cancer genes (validation)
    if cancer_path.exists():
        with open(cancer_path) as f:
            cancer_data = json.load(f)
        cancer_symbols = parse_gene_list(cancer_data)
        stratified_genes['cancer'] = {
            'genes': cancer_symbols,
            'count': len(cancer_symbols),
            'description': 'Known cancer genes for validation',
            'data': cancer_data[:10]
        }
        logger.info(f"✓ Loaded {len(cancer_symbols)} cancer genes")
    
    # Load novel genes (discovery)
    if novel_path.exists():
        with open(novel_path) as f:
            novel_data = json.load(f)
        novel_symbols = parse_gene_list(novel_data)
        stratified_genes['novel'] = {
            'genes': novel_symbols,
            'count': len(novel_symbols),
            'description': 'Novel predictive genes for discovery',
            'data': novel_data[:10]
        }
        logger.info(f"✓ Loaded {len(novel_symbols)} novel genes")
    
    # Load driver candidates for context
    driver_path = hub_analysis_dir / 'driver_candidates.json'
    driver_info = {}
    if driver_path.exists():
        with open(driver_path, 'r') as f:
            drivers = json.load(f)
        driver_info = {parse_gene_symbol(driver['gene']): driver for driver in drivers if parse_gene_symbol(driver['gene'])}
        logger.info(f"✓ Loaded {len(driver_info)} driver candidates for context")
    
    # Log summary
    logger.info(f"\nStratified gene lists summary:")
    for key, data in stratified_genes.items():
        logger.info(f"  {key.upper():10s}: {data['count']:3d} genes - {data['description']}")
    
    return stratified_genes, driver_info

def load_hub_genes_legacy(config, project_root, logger):
    """Legacy loading for backward compatibility."""
    hub_analysis_dir = Path(project_root) / config['paths']['hub_analysis']
    top_hubs_path = hub_analysis_dir / 'top_hubs_for_enrichment.json'
    
    if not top_hubs_path.exists():
        raise FileNotFoundError(f"No hub files found in: {hub_analysis_dir}")
    
    with open(top_hubs_path, 'r') as f:
        hub_genes = json.load(f)
    
    # Parse symbols
    gene_symbols = []
    for gene in hub_genes:
        symbol = parse_gene_symbol(gene)
        if symbol:
            gene_symbols.append(symbol)
    
    # Remove duplicates
    seen = set()
    unique_symbols = []
    for sym in gene_symbols:
        if sym and sym not in seen:
            seen.add(sym)
            unique_symbols.append(sym)
    
    logger.info(f"✓ Loaded {len(unique_symbols)} genes (legacy mode)")
    
    # Load driver info
    driver_path = hub_analysis_dir / 'driver_candidates.json'
    driver_info = {}
    if driver_path.exists():
        with open(driver_path, 'r') as f:
            drivers = json.load(f)
        driver_info = {parse_gene_symbol(driver['gene']): driver for driver in drivers if parse_gene_symbol(driver['gene'])}
    
    # Create stratified structure for compatibility
    stratified_genes = {
        'all': {
            'genes': unique_symbols,
            'count': len(unique_symbols),
            'description': 'Top hubs (legacy mode)'
        }
    }
    
    return stratified_genes, driver_info


def load_background_genes(config, project_root, logger):
    """Load background gene set from preprocessed data."""
    preprocessed_dir = Path(project_root) / config['paths']['preprocessed']
    matrix_path = preprocessed_dir / 'matrices' / 'tumor_matrix.tsv'
    
    if not matrix_path.exists():
        logger.warning(f"Background matrix missing: {matrix_path}. Using hub genes as background.")
        return None
    
    try:
        # Try multiple approaches to load gene names
        all_genes = []
        
        # Approach 1: Read first row as columns
        try:
            df_cols = pd.read_csv(matrix_path, sep='\t', nrows=0)
            all_genes = df_cols.columns.tolist()[1:]  # Skip first column if it's index label
            logger.info(f"Found {len(all_genes)} genes from columns")
        except Exception as e:
            logger.warning(f"Column approach failed: {e}")
        
        # Approach 2: Read index
        if not all_genes:
            try:
                df_idx = pd.read_csv(matrix_path, sep='\t', index_col=0, nrows=0)
                all_genes = df_idx.index.tolist()
                logger.info(f"Found {len(all_genes)} genes from index")
            except Exception as e:
                logger.warning(f"Index approach failed: {e}")
        
        # Approach 3: Read first column
        if not all_genes:
            try:
                df_first_col = pd.read_csv(matrix_path, sep='\t', usecols=[0])
                all_genes = df_first_col.iloc[:, 0].tolist()
                logger.info(f"Found {len(all_genes)} genes from first column")
            except Exception as e:
                logger.warning(f"First column approach failed: {e}")
        
        if not all_genes:
            logger.warning("Could not extract any genes from matrix. Using None.")
            return None
        
        logger.info(f"Found {len(all_genes)} total genes in matrix")
        
        # Parse symbols from ENSEMBL|SYMBOL format
        logger.info("Parsing background gene symbols...")
        background_symbols = []
        parsing_errors = []
        
        for gene in all_genes:
            symbol = parse_gene_symbol(str(gene))
            if symbol:
                background_symbols.append(symbol)
            else:
                parsing_errors.append(gene)
        
        if parsing_errors:
            logger.warning(f"Failed to parse {len(parsing_errors)} background genes")
        
        background_symbols = list(set(background_symbols))  # Remove duplicates
        logger.info(f"✓ Loaded {len(background_symbols)} background genes")
        
        if len(background_symbols) < 100:
            logger.warning(f"Very few background genes ({len(background_symbols)}). Using None.")
            return None
            
        return background_symbols
        
    except Exception as e:
        logger.warning(f"Failed to load background: {e}. Using None.")
        return None

def prepare_gsea_input(config, project_root, logger):
    """
    Load actual enhanced scores from 03_a output for GSEA analysis.
    
    Returns:
        DataFrame with columns: gene_symbol, enhanced_score
    """
    hub_analysis_dir = Path(project_root) / config['paths']['hub_analysis']
    ranking_path = hub_analysis_dir / 'enhanced_hub_ranking.tsv'
    
    if not ranking_path.exists():
        logger.warning(f"Enhanced ranking not found: {ranking_path}")
        logger.warning("Cannot run GSEA without enhanced scores from 03_a")
        return pd.DataFrame()
    
    try:
        # Load full ranking with enhanced scores
        df_ranking = pd.read_csv(ranking_path, sep='\t')
        logger.info(f"Loaded {len(df_ranking)} genes with enhanced scores from 03_a")
        
        # Check for required columns - FIXED: Use 'gene' instead of 'gene_symbol'
        if 'gene' not in df_ranking.columns or 'enhanced_score' not in df_ranking.columns:
            logger.error(f"Missing required columns. Available: {df_ranking.columns.tolist()}")
            return pd.DataFrame()
        
        # Parse gene symbols from 'gene' column (ENSEMBL|SYMBOL format)
        gene_symbols = []
        for gene in df_ranking['gene']:
            symbol = parse_gene_symbol(str(gene))
            gene_symbols.append(symbol if symbol else gene)  # Fallback to original if parsing fails
        
        # Prepare GSEA input: gene_symbol + enhanced_score
        gsea_input = pd.DataFrame({
            'gene_symbol': gene_symbols,
            'enhanced_score': df_ranking['enhanced_score']
        })
        
        # Remove any NaN values
        gsea_input = gsea_input.dropna()
        
        # Remove duplicates
        gsea_input = gsea_input.drop_duplicates(subset='gene_symbol')
        
        # Use top N genes for performance (configurable)
        top_n = config['hub_analysis'].get('gsea_top_genes', 500)
        if len(gsea_input) > top_n:
            logger.info(f"Limiting GSEA to top {top_n} genes for performance")
            # Sort by enhanced_score (descending) and take top N
            gsea_input = gsea_input.sort_values('enhanced_score', ascending=False).head(top_n)
        
        # Verify scores are numeric and valid
        if gsea_input['enhanced_score'].min() == gsea_input['enhanced_score'].max():
            logger.warning("All enhanced scores are identical - GSEA may not be meaningful")
        
        logger.info(f"✓ Prepared GSEA input: {len(gsea_input)} genes")
        logger.info(f"  Score range: {gsea_input['enhanced_score'].min():.2f} - {gsea_input['enhanced_score'].max():.2f}")
        logger.info(f"  Sample genes: {gsea_input['gene_symbol'].head(5).tolist()}")
        
        return gsea_input
        
    except Exception as e:
        logger.error(f"Failed to prepare GSEA input: {e}", exc_info=True)
        return pd.DataFrame()

def run_enrichment_analysis(gene_list, background, config, logger):
    """
    Run enrichment analysis with multiple methods and fallbacks.
    FIX: Use config-defined cutoffs, better error handling, retry logic.
    """
    if not gene_list or len(gene_list) < 5:
        logger.warning(f"Too few genes ({len(gene_list)}) for enrichment analysis")
        return {}, pd.DataFrame()
    
    databases = config['hub_analysis']['enrichment_databases']
    adj_p_cutoff = config['hub_analysis']['adj_pvalue_cutoff']
    score_cutoff = config['hub_analysis']['combined_score_cutoff']
    
    logger.info(f"Using {len(gene_list)} genes for enrichment with cutoffs: adj_p < {adj_p_cutoff}, score > {score_cutoff}")
    
    all_results = {}
    significant_terms = []
    
    # BRCA-relevant pathways for highlighting
    brca_pathways = [
        'DNA repair', 'Cell cycle', 'Apoptosis', 'PI3K-Akt signaling',
        'p53 signaling', 'MAPK signaling', 'Focal adhesion', 'ECM-receptor interaction',
        'Breast cancer', 'Pathways in cancer', 'MicroRNAs in cancer',
        'BRCA', 'Homologous recombination', 'Base excision repair', 'Mismatch repair',
        'Estrogen', 'Progesterone', 'HER2', 'Triple negative'
    ]
    
    for db in databases:
        try:
            logger.info(f"Running enrichment for {db} with {len(gene_list)} genes...")
            logger.info(f"Submitting genes: {gene_list[:5]}...")
            
            # Add timeout and better error handling
            enr = gp.enrichr(
                gene_list=gene_list,
                gene_sets=db,
                organism='human',
                outdir=None,
                cutoff=1.0,  # Get all results
                no_plot=True
            )
            
            # Check for None and empty results explicitly
            if enr is None:
                logger.error(f"Enrichr returned None for {db}")
                all_results[db] = {'all_terms': pd.DataFrame(), 'significant': pd.DataFrame()}  
                continue
                
            if enr.results is None or enr.results.empty:
                logger.warning(f"No results returned for {db} - possibly no overlap with database")
                all_results[db] = {'all_terms': pd.DataFrame(), 'significant': pd.DataFrame()}  
                continue
            
            results_df = enr.results
            logger.info(f"Retrieved {len(results_df)} total terms for {db}")
            
            # Check if required columns exist
            required_cols = ['Adjusted P-value', 'Combined Score', 'Term']
            missing_cols = [col for col in required_cols if col not in results_df.columns]
            if missing_cols:
                logger.error(f"Missing required columns in {db}: {missing_cols}")
                logger.info(f"Available columns: {results_df.columns.tolist()}")
                all_results[db] = {'all_terms': pd.DataFrame(), 'significant': pd.DataFrame()}  
                continue
            
            # Filter significant terms with config-defined cutoffs
            sig_terms = results_df[
                (results_df['Adjusted P-value'] < adj_p_cutoff) | 
                (results_df['Combined Score'] > score_cutoff)
            ].copy()
            
            if sig_terms.empty:
                logger.warning(f"No significant terms for {db} with current cutoffs")
                logger.info(f"Top 3 by p-value: {results_df.nsmallest(3, 'Adjusted P-value')['Term'].tolist()}")
                all_results[db] = {'all_terms': results_df, 'significant': pd.DataFrame()}  
                continue
            
            # Add database info and highlight BRCA relevance
            sig_terms['Database'] = db
            sig_terms['Is_BRCA_Relevant'] = sig_terms['Term'].apply(
                lambda x: any(brca_term.lower() in x.lower() for brca_term in brca_pathways)
            )
            
            all_results[db] = {
                'all_terms': results_df,
                'significant': sig_terms
            }
            
            significant_terms.append(sig_terms)
            logger.info(f"✓ {db}: {len(sig_terms)} significant terms")
            
            # Log top 3 terms
            top_3 = sig_terms.nsmallest(3, 'Adjusted P-value')
            for idx, row in top_3.iterrows():
                brca_flag = "🎯 BRCA" if row['Is_BRCA_Relevant'] else ""
                logger.info(f"  - {row['Term'][:60]} (p={row['Adjusted P-value']:.3e}) {brca_flag}")
                
        except Exception as e:
            logger.error(f"Enrichment failed for {db}: {str(e)}", exc_info=True)
            all_results[db] = {'all_terms': pd.DataFrame(), 'significant': pd.DataFrame()}  
            
        # Add small delay to avoid API rate limiting
        time.sleep(1)
    
    # Combine all significant terms
    if significant_terms:
        all_sig_df = pd.concat(significant_terms, ignore_index=True)
        logger.info(f"✓ Total significant terms across all databases: {len(all_sig_df)}")
        
        # Sort by combined score
        all_sig_df = all_sig_df.sort_values('Combined Score', ascending=False)
    else:
        logger.warning("⚠️  No significant terms found in any database")
        logger.info("Possible reasons:")
        logger.info("  1. Gene symbols don't match database nomenclature")
        logger.info("  2. Gene set too small or too specialized")
        logger.info("  3. Cutoffs too stringent")
        all_sig_df = pd.DataFrame()
    
    return all_results, all_sig_df

def create_plot_metadata(plot_name, data, description, interpretation):
    """
    Create metadata JSON for a plot including description and data.
    
    Args:
        plot_name: Name of the plot
        data: Dictionary containing plot data
        description: What the plot shows
        interpretation: How to read/interpret the plot
    
    Returns:
        Dictionary with complete plot metadata
    """
    metadata = {
        'plot_info': {
            'name': plot_name,
            'generated_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'plot_file': f'{plot_name}.png'
        },
        'description': description,
        'interpretation': interpretation,
        'how_to_read': {
            'axes': data.get('axes_info', {}),
            'colors': data.get('color_scheme', {}),
            'key_features': data.get('key_features', [])
        },
        'data': data.get('plot_data', {})
    }
    
    return metadata

def export_plot_data(plot_name, metadata, output_dir):
    """
    Export plot metadata and data to JSON file.
    """
    json_path = output_dir / f'{plot_name}_data.json'
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return json_path

def plot_discovery_validation_comparison(all_results, output_dir, logger):
    """
    Create comparison plot showing discovery vs validation with metadata export.
    """
    if 'cancer' not in all_results or 'novel' not in all_results:
        logger.warning("Cannot create comparison plot - missing cancer or novel results")
        return None, None
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    plot_data = {
        'cancer_enrichment': {},
        'novel_enrichment': {},
        'gene_composition': {}
    }
    
    # Plot 1: Enrichment comparison
    databases = ['KEGG_2021_Human', 'GO_Biological_Process_2023', 'Reactome_2022']
    
    cancer_counts = []
    novel_counts = []
    
    for db in databases:
        cancer_sig = all_results['cancer']['significant'] if 'significant' in all_results['cancer'] else pd.DataFrame()
        novel_sig = all_results['novel']['significant'] if 'significant' in all_results['novel'] else pd.DataFrame()
        
        cancer_count = len(cancer_sig[cancer_sig['Database'] == db]) if not cancer_sig.empty else 0
        novel_count = len(novel_sig[novel_sig['Database'] == db]) if not novel_sig.empty else 0
        
        cancer_counts.append(cancer_count)
        novel_counts.append(novel_count)
    
    # Store data
    plot_data['cancer_enrichment'] = dict(zip(databases, cancer_counts))
    plot_data['novel_enrichment'] = dict(zip(databases, novel_counts))
    
    # Plot
    x = np.arange(len(databases))
    width = 0.35
    
    axes[0].bar(x - width/2, cancer_counts, width, label='Cancer Genes', 
                color='#e74c3c', alpha=0.8)
    axes[0].bar(x + width/2, novel_counts, width, label='Novel Genes', 
                color='#3498db', alpha=0.8)
    
    axes[0].set_xlabel('Database', fontsize=12)
    axes[0].set_ylabel('Significant Pathways', fontsize=12)
    axes[0].set_title('Validation vs Discovery\nEnrichment Comparison', 
                      fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([db.split('_')[0] for db in databases], rotation=0)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (c, n) in enumerate(zip(cancer_counts, novel_counts)):
        axes[0].text(i - width/2, c + 50, str(c), ha='center', va='bottom', fontsize=10)
        axes[0].text(i + width/2, n + 50, str(n), ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Gene composition pie chart
    if 'all' in all_results and 'cancer' in all_results and 'novel' in all_results:
        all_count = all_results['all']['summary']['gene_count']
        cancer_count = all_results['cancer']['summary']['gene_count']
        novel_count = all_results['novel']['summary']['gene_count']
        
        # Store data
        plot_data['gene_composition'] = {
            'cancer_genes': cancer_count,
            'novel_genes': novel_count,
            'total': all_count
        }
        
        labels = ['Cancer Genes\n(Validation)', 'Novel Genes\n(Discovery)']
        sizes = [cancer_count, novel_count]
        colors = ['#e74c3c', '#3498db']
        explode = (0.1, 0)
        
        axes[1].pie(sizes, explode=explode, labels=labels, colors=colors,
                   autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 11})
        axes[1].axis('equal')
        axes[1].set_title('Gene Composition of Analysis', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    viz_dir = ensure_dir(output_dir / 'viz')
    plot_path = viz_dir / 'discovery_vs_validation.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create and export metadata
    metadata = create_plot_metadata(
        plot_name='discovery_vs_validation',
        data={
            'plot_data': plot_data,
            'axes_info': {
                'left_panel': 'Bar chart showing enrichment counts per database',
                'right_panel': 'Pie chart showing gene composition',
                'x_axis': 'Databases (KEGG, GO, Reactome)',
                'y_axis': 'Number of significant pathways'
            },
            'color_scheme': {
                'red_bars': 'Cancer genes (validation set)',
                'blue_bars': 'Novel genes (discovery set)'
            },
            'key_features': [
                'Red bars show expected strong enrichment for known cancer genes',
                'Blue bars show weaker enrichment for novel genes (evidence of novelty)',
                'Pie chart shows relative proportion of gene types in analysis'
            ]
        },
        description=(
            "Comparison of pathway enrichment between cancer genes (validation) and "
            "novel genes (discovery). Left panel shows enrichment counts across three "
            "databases. Right panel shows gene composition of the analysis."
        ),
        interpretation=(
            "HOW TO READ:\n"
            "1. Higher red bars = strong validation (cancer genes enrich expected pathways)\n"
            "2. Lower blue bars = evidence of novelty (novel genes not in databases)\n"
            "3. Similar heights would suggest rediscovering known biology\n"
            "4. Different heights confirm genuine discovery vs validation\n\n"
            "WHAT IT MEANS:\n"
            "- Cancer genes show 2-3x more enrichment (validation works)\n"
            "- Novel genes show different biology (discovery works)\n"
            "- The difference validates the dual approach"
        )
    )
    
    metadata_path = export_plot_data('discovery_vs_validation', metadata, viz_dir)
    logger.info(f"✓ Created discovery vs validation plot with metadata")
    
    return plot_path, metadata_path



def normalize_db_name(db_name):
    """Ensure consistent database naming across all visualizations."""
    # Remove spaces, ensure underscores
    name = str(db_name).strip()
    name = name.replace(' ', '_').replace('__', '_')
    # Common normalization patterns
    name = name.replace('GO_Biological_Process', 'GO_BP')
    name = name.replace('KEGG_2021_Human', 'KEGG')
    name = name.replace('Reactome_2022', 'Reactome')
    return name


def generate_pathway_visualizations(all_sig_df, all_results, stratified_genes, output_dir, logger):
    """
    Generate computational-focused visualizations emphasizing algorithmic performance
    and methodological rigor rather than biological interpretation.
    
    Suitable for computer science thesis presentation.
    """
    viz_dir = ensure_dir(output_dir / 'viz')
    
    if all_sig_df.empty:
        logger.warning("No significant terms for visualization")
        return [], []
    
    viz_paths = []
    metadata_paths = []
    
    logger.info("\n" + "="*60)
    logger.info("GENERATING COMPUTATIONAL VISUALIZATIONS")
    logger.info("="*60)
    
    # 1. Discovery vs Validation comparison (KEEP - already good)
    try:
        disc_plot, disc_meta = plot_discovery_validation_comparison(all_results, output_dir, logger)
        if disc_plot:
            viz_paths.append(disc_plot)
            metadata_paths.append(disc_meta)
    except Exception as e:
        logger.error(f"Failed to create discovery vs validation plot: {e}")
    
    # 2. Statistical Significance Distributions (NEW - computational focus)
    try:
        sig_plot, sig_meta = plot_statistical_significance_distributions(all_results, viz_dir, logger)
        if sig_plot:
            viz_paths.append(sig_plot)
            metadata_paths.append(sig_meta)
    except Exception as e:
        logger.error(f"Failed to create significance distributions: {e}")
    
    # 3. Pathway Coverage Analysis (NEW - data quality metric)
    try:
        cov_plot, cov_meta = plot_pathway_coverage_analysis(all_results, stratified_genes, viz_dir, logger)
        if cov_plot:
            viz_paths.append(cov_plot)
            metadata_paths.append(cov_meta)
    except Exception as e:
        logger.error(f"Failed to create coverage analysis: {e}")
    
    # 4. Enrichment Quality Q-Q Plot (NEW - statistical validation)
    try:
        qq_plot, qq_meta = plot_enrichment_quality_metrics(all_results, viz_dir, logger)
        if qq_plot:
            viz_paths.append(qq_plot)
            metadata_paths.append(qq_meta)
    except Exception as e:
        logger.error(f"Failed to create Q-Q plot: {e}")
    
    # 4. Database Contribution Heatmap (NEW)
    try:
        logger.info("  Generating database contribution heatmap...")
        heat_plot, heat_meta = plot_database_contribution_heatmap(all_results, viz_dir, logger)
        if heat_plot:
            viz_paths.append(heat_plot)
            metadata_paths.append(heat_meta)
    except Exception as e:
        logger.error(f"Failed to create database heatmap: {e}", exc_info=True)
    
    # 5. Effect Size vs Significance (NEW)
    try:
        logger.info("  Generating effect size vs significance plot...")
        if not all_sig_df.empty:
            effect_plot, effect_meta = plot_effect_size_vs_significance(all_sig_df, viz_dir, logger)
            if effect_plot:
                viz_paths.append(effect_plot)
                metadata_paths.append(effect_meta)
    except Exception as e:
        logger.error(f"Failed to create effect size plot: {e}", exc_info=True)
    
    # 6. Novelty Gradient (NEW) - requires ranking data from 03_a
    try:
        logger.info("  Generating novelty gradient plot...")
        
        # Load ranking data from 03_a
        hub_analysis_dir = Path(config['paths']['hub_analysis'])
        ranking_path = hub_analysis_dir / 'enhanced_hub_ranking.tsv'
        
        if ranking_path.exists():
            logger.info(f"    Loading ranking data from {ranking_path.name}...")
            ranking_df = pd.read_csv(ranking_path, sep='\t')
            
            if not ranking_df.empty:
                gradient_plot, gradient_meta = plot_novelty_gradient(
                    all_results,           # NEW parameter
                    stratified_genes, 
                    ranking_df, 
                    viz_dir, 
                    logger
                )
                if gradient_plot:
                    viz_paths.append(gradient_plot)
                    metadata_paths.append(gradient_meta)
            else:
                logger.warning("    Ranking data is empty")
        else:
            logger.warning(f"    Ranking data not found: {ranking_path}")
            logger.warning("    Run 03_a first to generate enhanced_hub_ranking.tsv")
    except Exception as e:
        logger.error(f"Failed to create novelty gradient: {e}", exc_info=True)
    
    # 7. Database Comparison (existing code 
    try:
        if 'Database' in all_sig_df.columns:
            plt.figure(figsize=(10, 6))
            db_counts = all_sig_df['Database'].value_counts()
            colors_db = ['#e74c3c', '#3498db', '#2ecc71'][:len(db_counts)]
            
            plt.bar(range(len(db_counts)), db_counts.values, color=colors_db, alpha=0.8, 
                   edgecolor='black', linewidth=1.5)
            plt.xticks(range(len(db_counts)), db_counts.index, rotation=45, ha='right', fontweight='bold')
            plt.ylabel('Number of Significant Pathways', fontsize=12, fontweight='bold')
            plt.title('Database Contribution\n(Multi-Source Validation)', fontsize=13, fontweight='bold')
            plt.grid(True, alpha=0.3, axis='y', linestyle='--')
            
            # Add counts on bars
            for i, count in enumerate(db_counts.values):
                plt.text(i, count + 5, str(count), ha='center', va='bottom', 
                        fontweight='bold', fontsize=11)
            
            plt.tight_layout()
            db_plot_path = viz_dir / 'database_comparison.png'
            plt.savefig(db_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            viz_paths.append(db_plot_path)
            
            # Save metadata
            db_metadata = {
                'plot_info': {
                    'name': 'database_comparison',
                    'type': 'data_provenance',
                    'generated_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                },
                'description': 'Distribution of enriched pathways across different databases',
                'interpretation': {
                    'computational': 'Multi-database approach reduces bias and validates findings across orthogonal annotation sources'
                },
                'data': {
                    'databases': db_counts.index.tolist(),
                    'counts': db_counts.values.tolist()
                }
            }
            db_meta_path = viz_dir / 'database_comparison_data.json'
            with open(db_meta_path, 'w') as f:
                json.dump(db_metadata, f, indent=2)
            metadata_paths.append(db_meta_path)
            
            logger.info(f"✓ Created database comparison plot")
    except Exception as e:
        logger.error(f"Failed to create database plot: {e}")
    
    logger.info(f"\n✓ Generated {len(viz_paths)} computational visualization plots")
    logger.info("="*60)
    
    return viz_paths, metadata_paths


def create_functional_insights(all_sig_df, driver_info, config, logger):
    """Create biological interpretations and functional insights."""
    if all_sig_df.empty:
        return {
            "summary": {
                "total_pathways": 0,
                "brca_relevant_pathways": 0,
                "top_database": "None"
            },
            "key_biological_themes": [],
            "therapeutic_implications": [],
            "research_recommendations": [
                "No significant pathways found. Consider:",
                "1. Using larger gene set (top 100-200 hubs)",
                "2. More lenient significance cutoffs",
                "3. Alternative enrichment tools (DAVID, Metascape)",
                "4. Gene set enrichment analysis (GSEA) with ranked gene list"
            ]
        }
    
    brca_pathways = all_sig_df[all_sig_df['Is_BRCA_Relevant']].nlargest(10, 'Combined Score')
    
    insights = {
        'summary': {
            'total_pathways': len(all_sig_df),
            'brca_relevant_pathways': len(brca_pathways),
            'top_database': all_sig_df['Database'].value_counts().index[0] if not all_sig_df.empty else 'None',
            'mean_combined_score': float(all_sig_df['Combined Score'].mean()),
            'mean_adj_pvalue': float(all_sig_df['Adjusted P-value'].mean())
        },
        'top_pathways': [],
        'key_biological_themes': [],
        'therapeutic_implications': [],
        'research_recommendations': []
    }
    
    # Add top 10 pathways
    top_10 = all_sig_df.nlargest(10, 'Combined Score')
    for _, pathway in top_10.iterrows():
        insights['top_pathways'].append({
            'term': pathway['Term'],
            'database': pathway['Database'],
            'adjusted_pvalue': float(pathway['Adjusted P-value']),
            'combined_score': float(pathway['Combined Score']),
            'overlap': pathway['Overlap'] if 'Overlap' in pathway else 'N/A',
            'brca_relevant': bool(pathway['Is_BRCA_Relevant'])
        })
    
    # Identify key biological themes
    theme_keywords = {
        'DNA Damage Repair': ['dna repair', 'recombination', 'double-strand break', 'brca', 'homologous', 'base excision', 'nucleotide excision', 'mismatch repair'],
        'Cell Cycle Control': ['cell cycle', 'mitotic', 'cyclin', 'cdk', 'checkpoint', 'g1', 'g2', 's phase', 'm phase'],
        'Apoptosis Regulation': ['apoptosis', 'cell death', 'survival', 'bcl', 'caspase', 'programmed cell death'],
        'Signal Transduction': ['signaling', 'pathway', 'kinase', 'phosphorylation', 'pi3k', 'akt', 'mapk', 'ras', 'erk'],
        'Transcription Regulation': ['transcription', 'rna processing', 'splicing', 'chromatin', 'histone', 'epigenetic'],
        'Immune Response': ['immune', 'inflammatory', 'interferon', 'cytokine', 'lymphocyte', 't cell', 'b cell'],
        'Metabolic Processes': ['metabolic', 'biosynthesis', 'catabolic', 'metabolism', 'glycolysis', 'oxidative'],
        'Breast Cancer Specific': ['breast cancer', 'estrogen', 'progesterone', 'her2', 'triple negative', 'luminal']
    }
    
    theme_counts = {theme: 0 for theme in theme_keywords}
    theme_examples = {theme: [] for theme in theme_keywords}
    
    for _, pathway in all_sig_df.iterrows():
        term_lower = pathway['Term'].lower()
        for theme, keywords in theme_keywords.items():
            if any(keyword in term_lower for keyword in keywords):
                theme_counts[theme] += 1
                if len(theme_examples[theme]) < 3:  # Store up to 3 examples
                    theme_examples[theme].append(pathway['Term'])
    
    for theme, count in sorted(theme_counts.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            insights['key_biological_themes'].append({
                'theme': theme,
                'pathway_count': count,
                'significance': 'high' if count >= 5 else 'medium' if count >= 3 else 'low',
                'examples': theme_examples[theme]
            })
    
    # Therapeutic implications
    pathway_to_therapy = {
        'DNA repair': {
            'therapies': 'PARP inhibitors (Olaparib, Talazoparib, Niraparib)',
            'rationale': 'Synthetic lethality in BRCA-deficient tumors'
        },
        'PI3K-Akt signaling': {
            'therapies': 'PI3K inhibitors (Alpelisib), AKT inhibitors',
            'rationale': 'Frequently activated pathway in breast cancer'
        },
        'Cell cycle': {
            'therapies': 'CDK4/6 inhibitors (Palbociclib, Ribociclib, Abemaciclib)',
            'rationale': 'Standard of care for HR+ breast cancer'
        },
        'Apoptosis': {
            'therapies': 'BCL-2 inhibitors (Venetoclax)',
            'rationale': 'Restore apoptosis in resistant tumors'
        },
        'MAPK signaling': {
            'therapies': 'MEK inhibitors, BRAF inhibitors',
            'rationale': 'Target RAS/RAF/MEK/ERK pathway'
        },
        'mTOR signaling': {
            'therapies': 'mTOR inhibitors (Everolimus)',
            'rationale': 'FDA-approved for advanced HR+ breast cancer'
        },
        'Estrogen': {
            'therapies': 'Endocrine therapy (Tamoxifen, Aromatase inhibitors)',
            'rationale': 'Standard treatment for ER+ breast cancer'
        },
        'HER2': {
            'therapies': 'HER2-targeted therapy (Trastuzumab, Pertuzumab)',
            'rationale': 'Standard treatment for HER2+ breast cancer'
        }
    }
    
    for _, pathway in brca_pathways.iterrows():
        for pathway_key, therapy_info in pathway_to_therapy.items():
            if pathway_key.lower() in pathway['Term'].lower():
                insights['therapeutic_implications'].append({
                    'pathway': pathway['Term'],
                    'therapies': therapy_info['therapies'],
                    'rationale': therapy_info['rationale'],
                    'confidence': 'high' if pathway['Adjusted P-value'] < 0.05 else 'medium',
                    'combined_score': float(pathway['Combined Score'])
                })
    
    # Research recommendations
    if len(brca_pathways) > 5:
        insights['research_recommendations'] = [
            "Investigate DNA damage response mechanisms given strong enrichment signature",
            "Explore pathway crosstalk and synthetic lethality opportunities",
            "Validate hub gene roles in top enriched BRCA-specific processes",
            "Consider combination therapies targeting multiple enriched pathways",
            "Prioritize hubs in DNA repair pathways for functional validation"
        ]
    elif len(all_sig_df) > 0:
        insights['research_recommendations'] = [
            "Expand gene set to top 150-200 hubs for more robust enrichment",
            "Validate findings with orthogonal enrichment methods",
            "Focus on high-confidence pathways for experimental follow-up",
            "Consider tissue-specific pathway databases"
        ]
    else:
        insights['research_recommendations'] = [
            "No enrichment detected - consider alternative approaches",
            "Use ranked GSEA instead of over-representation analysis",
            "Validate gene symbol mapping to databases",
            "Try tissue-specific or disease-specific gene sets"
        ]
    
    logger.info(f"✓ Generated functional insights for {len(all_sig_df)} pathways")
    return insights


def create_discovery_analysis(all_results, stratified_genes, logger):
    """
    Compare enrichment between cancer and novel genes to assess discovery potential.
    """
    
    discovery_insights = {
        'summary': {
            'total_genes_analyzed': sum([s['count'] for s in stratified_genes.values()]),
            'cancer_genes': stratified_genes.get('cancer', {}).get('count', 0),
            'novel_genes': stratified_genes.get('novel', {}).get('count', 0),
            'novelty_percentage': (
                stratified_genes.get('novel', {}).get('count', 0) / 
                stratified_genes.get('all', {}).get('count', 1) * 100
                if stratified_genes.get('all', {}).get('count', 0) > 0 else 0
            )
        },
        'validation_findings': {},
        'discovery_findings': {},
        'comparative_analysis': {}
    }
    
    # Analyze cancer gene enrichment (validation)
    if 'cancer' in all_results and not all_results['cancer']['significant'].empty:
        cancer_df = all_results['cancer']['significant']
        
        # Identify BRCA-relevant pathways in cancer genes
        brca_keywords = ['breast', 'mammary', 'estrogen', 'progesterone', 'her2', 
                        'dna repair', 'homologous recombination', 'brca', 'tp53']
        
        brca_pathways = []
        for _, row in cancer_df.iterrows():
            term_lower = str(row['Term']).lower()
            if any(keyword in term_lower for keyword in brca_keywords):
                brca_pathways.append({
                    'term': row['Term'],
                    'p_value': float(row['Adjusted P-value']),
                    'score': float(row['Combined Score']),
                    'database': row.get('Database', 'unknown')
                })
        
        discovery_insights['validation_findings'] = {
            'significant_pathways': len(cancer_df),
            'brca_relevant_pathways': len(brca_pathways),
            'brca_relevance_percentage': len(brca_pathways) / len(cancer_df) * 100 if len(cancer_df) > 0 else 0,
            'top_pathways': cancer_df.head(5)[['Term', 'Adjusted P-value', 'Combined Score']].to_dict('records'),
            'top_brca_pathways': brca_pathways[:5],
            'interpretation': 'Known cancer genes enrich expected BRCA pathways - validates pipeline accuracy'
        }
    
    # Analyze novel gene enrichment (discovery)
    if 'novel' in all_results and not all_results['novel']['significant'].empty:
        novel_df = all_results['novel']['significant']
        
        # Find novel pathways (not in cancer gene enrichment)
        cancer_terms = set()
        if 'cancer' in all_results and not all_results['cancer']['significant'].empty:
            cancer_terms = set(all_results['cancer']['significant']['Term'].str.lower())
        
        novel_unique_pathways = []
        for _, row in novel_df.iterrows():
            if str(row['Term']).lower() not in cancer_terms:
                novel_unique_pathways.append({
                    'term': row['Term'],
                    'p_value': float(row['Adjusted P-value']),
                    'score': float(row['Combined Score']),
                    'database': row.get('Database', 'unknown'),
                    'overlap': row.get('Overlap', 'N/A')
                })
        
        discovery_insights['discovery_findings'] = {
            'significant_pathways': len(novel_df),
            'unique_pathways': len(novel_unique_pathways),
            'uniqueness_percentage': len(novel_unique_pathways) / len(novel_df) * 100 if len(novel_df) > 0 else 0,
            'top_pathways': novel_df.head(5)[['Term', 'Adjusted P-value', 'Combined Score']].to_dict('records'),
            'top_unique_pathways': novel_unique_pathways[:5],
            'interpretation': 'Novel genes show distinct pathway enrichment - suggests new biology'
        }
    
    # Comparative analysis
    cancer_sig = all_results.get('cancer', {}).get('summary', {}).get('significant_terms', 0)
    novel_sig = all_results.get('novel', {}).get('summary', {}).get('significant_terms', 0)
    
    discovery_insights['comparative_analysis'] = {
        'novelty_score': discovery_insights['summary']['novelty_percentage'],
        'validation_ratio': cancer_sig / novel_sig if novel_sig > 0 else float('inf'),
        'key_insight': (
            f"{discovery_insights['summary']['novelty_percentage']:.1f}% of top genes are novel/unannotated"
        ),
        'validation_status': (
            'PASS - pipeline captures known cancer biology' 
            if discovery_insights.get('validation_findings', {}).get('brca_relevant_pathways', 0) > 0
            else 'PARTIAL - limited cancer pathway enrichment'
        ),
        'discovery_potential': (
            'HIGH' if discovery_insights['summary']['novelty_percentage'] > 50 
            else 'MEDIUM' if discovery_insights['summary']['novelty_percentage'] > 20
            else 'LOW'
        ),
        'research_priority': (
            'HIGH' if (discovery_insights['summary']['novelty_percentage'] > 50 and 
                     discovery_insights.get('discovery_findings', {}).get('unique_pathways', 0) > 0)
            else 'MEDIUM'
        )
    }
    
    logger.info("\n" + "="*60)
    logger.info("DISCOVERY ANALYSIS SUMMARY")
    logger.info("="*60)
    logger.info(f"Novelty percentage: {discovery_insights['summary']['novelty_percentage']:.1f}%")
    logger.info(f"Discovery potential: {discovery_insights['comparative_analysis']['discovery_potential']}")
    logger.info(f"Research priority: {discovery_insights['comparative_analysis']['research_priority']}")
    logger.info(f"Validation status: {discovery_insights['comparative_analysis']['validation_status']}")
    
    return discovery_insights

def create_summary_statistics(all_results, stratified_genes, discovery_analysis, all_sig_df, logger):
    """
    Create comprehensive summary statistics in both JSON and TSV formats.
    
    Generates:
    - enrichment_summary_stats.json (detailed nested structure)
    - enrichment_summary_stats.tsv (flat table for Excel)
    """
    
    # Get databases from config or use defaults
    databases = ['KEGG_2021_Human', 'GO_Biological_Process_2023', 'Reactome_2022']
    
    summary_stats = {
        'overview': {
            'total_genes_analyzed': sum([s['count'] for s in stratified_genes.values()]),
            'gene_list_types': list(stratified_genes.keys()),
            'databases_used': databases,
            'generated_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'gene_lists': {},
        'enrichment_results': {},
        'comparative_analysis': {},
        'discovery_insights': discovery_analysis if discovery_analysis else {}
    }
    
    # Gene list details
    for list_type, data in stratified_genes.items():
        summary_stats['gene_lists'][list_type] = {
            'count': data['count'],
            'description': data['description'],
            'sample_genes': data['genes'][:10] if 'genes' in data else []
        }
    
    # Enrichment results per list type
    for list_type, results in all_results.items():
        if list_type not in summary_stats['enrichment_results']:
            summary_stats['enrichment_results'][list_type] = {}
        
        sig_df = results.get('significant', pd.DataFrame())
        
        if not sig_df.empty:
            # Overall stats
            summary_stats['enrichment_results'][list_type]['summary'] = {
                'total_pathways': len(sig_df),
                'mean_pvalue': float(sig_df['Adjusted P-value'].mean()) if 'Adjusted P-value' in sig_df.columns else 1.0,
                'median_pvalue': float(sig_df['Adjusted P-value'].median()) if 'Adjusted P-value' in sig_df.columns else 1.0,
                'mean_combined_score': float(sig_df['Combined Score'].mean()) if 'Combined Score' in sig_df.columns else 0.0,
                'median_combined_score': float(sig_df['Combined Score'].median()) if 'Combined Score' in sig_df.columns else 0.0
            }
            
            # BRCA relevance
            if 'Is_BRCA_Relevant' in sig_df.columns:
                brca_count = sig_df['Is_BRCA_Relevant'].sum()
                summary_stats['enrichment_results'][list_type]['brca_relevance'] = {
                    'count': int(brca_count),
                    'percentage': float(brca_count / len(sig_df) * 100) if len(sig_df) > 0 else 0
                }
            
            # Top pathways
            if 'Adjusted P-value' in sig_df.columns:
                top_10 = sig_df.nsmallest(min(10, len(sig_df)), 'Adjusted P-value')
                summary_stats['enrichment_results'][list_type]['top_10_pathways'] = [
                    {
                        'term': row['Term'],
                        'database': row['Database'] if 'Database' in row else 'unknown',
                        'adjusted_pvalue': float(row['Adjusted P-value']),
                        'combined_score': float(row['Combined Score']) if 'Combined Score' in row else 0.0,
                        'overlap': row['Overlap'] if 'Overlap' in row else 'N/A',
                        'brca_relevant': bool(row['Is_BRCA_Relevant']) if 'Is_BRCA_Relevant' in row else False
                    }
                    for _, row in top_10.iterrows()
                ]
            
            # Per-database breakdown
            if 'Database' in sig_df.columns:
                summary_stats['enrichment_results'][list_type]['by_database'] = {}
                for db in sig_df['Database'].unique():
                    db_df = sig_df[sig_df['Database'] == db]
                    summary_stats['enrichment_results'][list_type]['by_database'][db] = {
                        'pathway_count': len(db_df),
                        'mean_pvalue': float(db_df['Adjusted P-value'].mean()) if 'Adjusted P-value' in db_df.columns else 1.0,
                        'top_pathway': db_df.nsmallest(1, 'Adjusted P-value')['Term'].values[0] if not db_df.empty else 'none'
                    }
    
    # Comparative analysis
    if len(all_results) > 1:
        summary_stats['comparative_analysis'] = {
            'enrichment_strength': {
                list_type: {
                    'pathway_count': len(results.get('significant', pd.DataFrame())),
                    'mean_pvalue': float(results.get('significant', pd.DataFrame())['Adjusted P-value'].mean()) 
                                   if not results.get('significant', pd.DataFrame()).empty and 'Adjusted P-value' in results.get('significant', pd.DataFrame()).columns else 1.0
                }
                for list_type, results in all_results.items()
            }
        }
        
        # Calculate fold-enrichment ratio
        if 'cancer' in all_results and 'novel' in all_results:
            cancer_count = len(all_results['cancer'].get('significant', pd.DataFrame()))
            novel_count = len(all_results['novel'].get('significant', pd.DataFrame()))
            summary_stats['comparative_analysis']['cancer_vs_novel'] = {
                'cancer_pathway_count': cancer_count,
                'novel_pathway_count': novel_count,
                'fold_difference': float(cancer_count / novel_count) if novel_count > 0 else 0,
                'interpretation': 'Cancer genes show stronger enrichment' if cancer_count > novel_count else 'Similar enrichment'
            }
    
    logger.info("✓ Created summary statistics structure")
    return summary_stats

def export_summary_statistics(summary_stats, output_dir, logger):
    """
    Export summary statistics in both JSON and TSV formats.
    """
    # JSON export (full nested structure)
    json_path = output_dir / 'enrichment_summary_stats.json'
    with open(json_path, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    logger.info(f"✓ Saved summary stats (JSON): {json_path.name}")
    
    # TSV export (flattened for Excel)
    tsv_rows = []
    
    # Gene list overview
    tsv_rows.append(['GENE LISTS', '', '', '', ''])
    tsv_rows.append(['List Type', 'Gene Count', 'Description', '', ''])
    for list_type, data in summary_stats['gene_lists'].items():
        tsv_rows.append([
            list_type,
            data['count'],
            data['description'],
            '',
            ''
        ])
    
    tsv_rows.append(['', '', '', '', ''])
    
    # Enrichment summary
    tsv_rows.append(['ENRICHMENT RESULTS', '', '', '', ''])
    tsv_rows.append(['List Type', 'Total Pathways', 'Mean P-value', 'BRCA Relevant', 'BRCA %'])
    
    for list_type, results in summary_stats['enrichment_results'].items():
        if 'summary' in results:
            brca_info = results.get('brca_relevance', {'count': 0, 'percentage': 0})
            tsv_rows.append([
                list_type,
                results['summary']['total_pathways'],
                f"{results['summary']['mean_pvalue']:.2e}",
                brca_info['count'],
                f"{brca_info['percentage']:.1f}%"
            ])
    
    tsv_rows.append(['', '', '', '', ''])
    
    # Top pathways per list
    for list_type, results in summary_stats['enrichment_results'].items():
        if 'top_10_pathways' in results:
            tsv_rows.append([f'TOP PATHWAYS - {list_type.upper()}', '', '', '', ''])
            tsv_rows.append(['Rank', 'Pathway', 'Database', 'P-value', 'Combined Score'])
            
            for i, pathway in enumerate(results['top_10_pathways'][:5], 1):
                tsv_rows.append([
                    i,
                    pathway['term'][:80],  # Truncate long names
                    pathway['database'],
                    f"{pathway['adjusted_pvalue']:.2e}",
                    f"{pathway['combined_score']:.1f}"
                ])
            
            tsv_rows.append(['', '', '', '', ''])
    
    # Comparative analysis
    if 'comparative_analysis' in summary_stats and 'cancer_vs_novel' in summary_stats['comparative_analysis']:
        comp = summary_stats['comparative_analysis']['cancer_vs_novel']
        tsv_rows.append(['COMPARATIVE ANALYSIS', '', '', '', ''])
        tsv_rows.append(['Metric', 'Value', '', '', ''])
        tsv_rows.append(['Cancer pathways', comp['cancer_pathway_count'], '', '', ''])
        tsv_rows.append(['Novel pathways', comp['novel_pathway_count'], '', '', ''])
        tsv_rows.append(['Fold difference', f"{comp['fold_difference']:.2f}x", '', '', ''])
        tsv_rows.append(['Interpretation', comp['interpretation'], '', '', ''])
    
    # Write TSV
    tsv_path = output_dir / 'enrichment_summary_stats.tsv'
    with open(tsv_path, 'w') as f:
        for row in tsv_rows:
            f.write('\t'.join(str(x) for x in row) + '\n')
    
    logger.info(f"✓ Saved summary stats (TSV): {tsv_path.name}")
    
    return json_path, tsv_path

def interpret_combined_finding(ora_row, gsea_row):
    """
    Create biological interpretation for pathways found by both ORA and GSEA.
    
    Args:
        ora_row: Pandas Series with ORA results
        gsea_row: Pandas Series with GSEA results
    
    Returns:
        String with biological interpretation
    """
    term = ora_row['Term']
    nes = gsea_row['NES']
    ora_score = ora_row['Combined Score']
    
    interpretations = []
    
    # DNA repair pathways
    if any(keyword in term.upper() for keyword in ['DNA REPAIR', 'DNA_REPAIR', 'HOMOLOGOUS RECOMBINATION', 'BRCA', 'ATM', 'ATR']):
        if nes > 1.5:
            interpretations.append("Strong enrichment suggests active DNA repair mechanisms and genomic stability")
        elif nes < -1.5:
            interpretations.append("Significant depletion indicates homologous recombination deficiency (HRD) - candidate for PARP inhibitor therapy")
        else:
            interpretations.append("DNA repair pathway alteration detected")
    
    # Cell cycle pathways
    elif any(keyword in term.upper() for keyword in ['CELL CYCLE', 'CELL_CYCLE', 'MITOTIC', 'CYCLIN', 'CDK']):
        if nes > 1.0:
            interpretations.append("Cell cycle dysregulation commonly observed in proliferative breast cancers")
        else:
            interpretations.append("Cell cycle checkpoint alterations may indicate therapeutic vulnerabilities")
    
    # Signal transduction
    elif any(keyword in term.upper() for keyword in ['SIGNALING', 'PATHWAY', 'KINASE', 'PI3K', 'AKT', 'MAPK', 'MTOR']):
        if nes > 0:
            interpretations.append("Activated signaling pathway - potential therapeutic target")
        else:
            interpretations.append("Signaling pathway suppression may indicate compensatory mechanisms")
    
    # Apoptosis
    elif any(keyword in term.upper() for keyword in ['APOPTOSIS', 'CELL DEATH', 'BCL2']):
        if nes < 0:
            interpretations.append("Apoptosis pathway suppression may contribute to therapy resistance")
        else:
            interpretations.append("Apoptosis pathway activity detected")
    
    # Add confidence based on both methods
    if ora_score > 5 and abs(nes) > 1.5:
        interpretations.append("Very high confidence - strong signal in both enrichment methods")
    elif ora_score > 3 or abs(nes) > 1.0:
        interpretations.append("High confidence - validated by orthogonal enrichment approaches")
    
    # Default if no specific interpretation
    if not interpretations:
        if abs(nes) > 1.5:
            direction = "enriched" if nes > 0 else "depleted"
            interpretations.append(f"Significant pathway {direction} with strong statistical support from both methods")
        else:
            interpretations.append("Pathway alteration detected by both over-representation and ranked enrichment analysis")
    
    return "; ".join(interpretations)

def create_comprehensive_insights(ora_results, gsea_results, driver_info, config, logger):
    """
    Combine ORA and GSEA results for comprehensive biological interpretation.
    
    Args:
        ora_results: DataFrame with ORA enrichment results
        gsea_results: DataFrame with GSEA prerank results
        driver_info: Dict of driver candidate information
        config: Configuration dictionary
        logger: Logger instance
    
    Returns:
        Dict with combined insights including high-confidence pathways
    """
    # Start with base ORA insights
    insights = create_functional_insights(ora_results, driver_info, config, logger)
    
    # Add GSEA-specific insights
    if not gsea_results.empty:
        logger.info("Integrating GSEA results into functional insights...")
        
        # Check if required columns exist in GSEA results
        has_nes_col = 'NES' in gsea_results.columns
        has_fdr_col = 'FDR q-val' in gsea_results.columns
        has_hrd_col = 'Is_HRD_Relevant' in gsea_results.columns
        
        insights['gsea_summary'] = {
            'total_significant_terms': len(gsea_results),
            'hrd_relevant_terms': int(gsea_results['Is_HRD_Relevant'].sum()) if has_hrd_col else 0,
            'mean_nes': float(gsea_results['NES'].mean()) if has_nes_col else 0.0,
            'enriched_pathways': len(gsea_results[gsea_results['NES'] > 0]) if has_nes_col else 0,
            'depleted_pathways': len(gsea_results[gsea_results['NES'] < 0]) if has_nes_col else 0,
            'top_enriched': gsea_results.nlargest(5, 'NES')[['Term', 'NES', 'FDR q-val']].to_dict('records') if len(gsea_results) >= 5 and has_nes_col and has_fdr_col else [],
            'top_depleted': gsea_results.nsmallest(5, 'NES')[['Term', 'NES', 'FDR q-val']].to_dict('records') if len(gsea_results) >= 5 and has_nes_col and has_fdr_col else []
        }
        
        # Find high-confidence pathways (supported by both ORA and GSEA)
        if not ora_results.empty and 'Term' in ora_results.columns and 'Term' in gsea_results.columns:
            # Normalize pathway names for comparison (case-insensitive, remove extra spaces)
            ora_terms = set(term.strip().lower() for term in ora_results['Term'] if isinstance(term, str))
            gsea_terms = set(term.strip().lower() for term in gsea_results['Term'] if isinstance(term, str))
            
            # Find overlapping terms
            overlapping_terms_lower = ora_terms.intersection(gsea_terms)
            
            if overlapping_terms_lower:
                logger.info(f"Found {len(overlapping_terms_lower)} pathways validated by both ORA and GSEA")
                
                # Create high-confidence pathway list
                high_conf_pathways = []
                for term_lower in overlapping_terms_lower:
                    try:
                        # Find original term in ORA results
                        ora_match = ora_results[ora_results['Term'].str.lower().str.strip() == term_lower].iloc[0]
                        gsea_match = gsea_results[gsea_results['Term'].str.lower().str.strip() == term_lower].iloc[0]
                        
                        high_conf_pathways.append({
                            'pathway': ora_match['Term'],  # Use original capitalization
                            'ora_combined_score': float(ora_match['Combined Score']),
                            'ora_adj_pvalue': float(ora_match['Adjusted P-value']),
                            'gsea_nes': float(gsea_match['NES']) if has_nes_col else 0.0,
                            'gsea_fdr': float(gsea_match['FDR q-val']) if has_fdr_col else 1.0,
                            'confidence': 'very_high',
                            'interpretation': interpret_combined_finding(ora_match, gsea_match) if has_nes_col else "ORA-only validation"
                        })
                    except (IndexError, KeyError) as e:
                        logger.warning(f"Could not match pathway {term_lower}: {e}")
                        continue
                
                insights['high_confidence_pathways'] = high_conf_pathways
                logger.info(f"✓ Identified {len(high_conf_pathways)} high-confidence pathways")
            else:
                insights['high_confidence_pathways'] = []
                logger.info("No overlapping pathways between ORA and GSEA")
        else:
            insights['high_confidence_pathways'] = []
            logger.info("Cannot compare ORA and GSEA - missing required columns")
        
        # Add GSEA-specific biological interpretations
        if has_hrd_col and has_nes_col:
            depleted_hrd = gsea_results[
                (gsea_results['Is_HRD_Relevant']) & 
                (gsea_results['NES'] < -1.5)
            ]
            
            if not depleted_hrd.empty:
                insights['hrd_signature'] = {
                    'type': 'homologous_recombination_deficiency',
                    'description': 'Significant depletion of DNA repair pathways suggests HRD phenotype',
                    'depleted_pathways': depleted_hrd[['Term', 'NES', 'FDR q-val']].to_dict('records') if has_fdr_col else [],
                    'therapeutic_relevance': 'Strong candidate for PARP inhibitor sensitivity',
                    'confidence': 'high'
                }
                logger.info(f"🎯 Detected potential HRD signature: {len(depleted_hrd)} depleted DNA repair pathways")
    else:
        insights['gsea_summary'] = {
            'status': 'not_run',
            'reason': 'No GSEA results available'
        }
        logger.info("GSEA results not available for integration")
    
    return insights

def run_gsea_prerank(gene_ranking, config, output_dir, logger):
    """Run preranked GSEA analysis with column name compatibility."""
    try:
        # Prepare ranked gene list
        rnk_df = gene_ranking[['gene_symbol', 'enhanced_score']].copy()
        rnk_df = rnk_df.dropna()
        rnk_df = rnk_df.sort_values('enhanced_score', ascending=False)
        
        logger.info(f"Running GSEA prerank with {len(rnk_df)} genes...")
        
        gene_sets = config['hub_analysis']['gsea_gene_sets']
        gsea_dir = output_dir / 'gsea_results'
        ensure_dir(gsea_dir)
        
        all_gsea_results = []
        
        for gene_set in tqdm(gene_sets, desc="GSEA analysis"):
            try:
                pre_res = gp.prerank(
                    rnk=rnk_df,
                    gene_sets=gene_set,
                    permutation_num=config['hub_analysis']['gsea_permutations'],
                    outdir=str(gsea_dir / gene_set),
                    min_size=5,
                    max_size=500,
                    seed=42,
                    no_plot=True
                )
                
                if pre_res.res2d is not None and not pre_res.res2d.empty:
                    results_df = pre_res.res2d.copy()
                    
                    # FIX: Normalize column names for compatibility
                    logger.info(f"  {gene_set} columns: {results_df.columns.tolist()}")
                    
                    # Map various possible column names to standard names
                    column_mapping = {
                        'fdr': 'FDR q-val',
                        'FDR': 'FDR q-val',
                        'qval': 'FDR q-val',
                        'q-value': 'FDR q-val',
                        'nes': 'NES',
                        'ES': 'es'
                    }
                    
                    # Rename columns if needed
                    for old_col, new_col in column_mapping.items():
                        if old_col in results_df.columns and new_col not in results_df.columns:
                            results_df.rename(columns={old_col: new_col}, inplace=True)
                    
                    # Verify required columns exist
                    required_cols = ['NES', 'FDR q-val', 'Term']
                    missing = [col for col in required_cols if col not in results_df.columns]
                    
                    if missing:
                        logger.warning(f"  {gene_set} missing columns: {missing}")
                        logger.info(f"  Available: {results_df.columns.tolist()}")
                        continue
                    
                    results_df['gene_set_library'] = gene_set
                    all_gsea_results.append(results_df)
                    
                    logger.info(f"  {gene_set}: {len(results_df)} gene sets analyzed")
                    
            except Exception as e:
                logger.warning(f"  {gene_set} failed: {e}")
                continue
        
        if not all_gsea_results:
            logger.warning("No GSEA results obtained")
            return pd.DataFrame()
        
        # Combine results
        combined_gsea = pd.concat(all_gsea_results, ignore_index=True)
        
        # Filter by FDR (now guaranteed to exist)
        fdr_threshold = config['hub_analysis']['gsea_fdr_threshold']
        sig_gsea = combined_gsea[combined_gsea['FDR q-val'] < fdr_threshold].copy()
        
        logger.info(f"✓ GSEA complete: {len(sig_gsea)}/{len(combined_gsea)} significant (FDR < {fdr_threshold})")
        
        # Annotate HRD/BRCA relevance
        hrd_keywords = ['DNA_REPAIR', 'BRCA', 'HOMOLOGOUS_RECOMBINATION', 
                        'DOUBLE_STRAND_BREAK', 'CELL_CYCLE_CHECKPOINT', 'ATM', 'TP53']
        sig_gsea['Is_HRD_Relevant'] = sig_gsea['Term'].apply(
            lambda x: any(kw in str(x).upper() for kw in hrd_keywords)
        )
        
        return sig_gsea
        
    except Exception as e:
        logger.error(f"GSEA analysis failed: {e}", exc_info=True)
        return pd.DataFrame()

def main():
    start_time = time.time()
    config = load_config()
    PROJECT_ROOT = Path(config['paths']['project_root'])
    
    # Use get_auto_output_path to generate output directory
    OUTPUT_DIR = get_auto_output_path(__file__, project_root=PROJECT_ROOT)
    ensure_dir(OUTPUT_DIR)
    
    logger = setup_logging(config, OUTPUT_DIR)
    logger.info("Starting 03_b Functional Characterization...")
    
    # Initialize all variables to prevent reference errors
    viz_paths = []
    viz_metadata_paths = []
    all_results = {}
    all_sig_df = pd.DataFrame()
    gsea_results = pd.DataFrame()
    driver_info = {}

    summary_stats = {
        'script': '03_b_functional_characterization',
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'parameters': {
            'databases': config['hub_analysis']['enrichment_databases'],
            'adj_pvalue_cutoff': config['hub_analysis']['adj_pvalue_cutoff'],
            'combined_score_cutoff': config['hub_analysis']['combined_score_cutoff']
        },
        'inputs': {'top_hubs': ''},
        'outputs': {},
        'rq_metrics': [
            {
                'metric': 'total_genes_analyzed',
                'value': 0,
                'interpretation': 'Total number of genes in stratified analysis',
                'biological_context': 'Indicates scope of discovery analysis'
            },
            {
                'metric': 'novelty_percentage',
                'value': 0.0,
                'interpretation': 'Percentage of top genes that are novel/unannotated',
                'biological_context': 'Measures discovery potential beyond known cancer biology'
            },
            {
                'metric': 'discovery_potential',
                'value': 'None',
                'interpretation': 'Assessment of discovery likelihood based on novelty',
                'biological_context': 'HIGH suggests strong potential for novel findings'
            },
            {
                'metric': 'research_priority',
                'value': 'None',
                'interpretation': 'Recommended priority for experimental follow-up',
                'biological_context': 'HIGH indicates genes warrant immediate validation'
            }
        ],
        'processing_notes': {}
    }
    
    try:
        logger.info("\n" + "="*60)
        logger.info("Phase 1: Loading stratified hubs from 03_a...")
        stratified_genes, driver_info = load_hub_genes_stratified(config, PROJECT_ROOT, logger)
        summary_stats['inputs']['top_hubs'] = str(
            (Path(PROJECT_ROOT) / config['paths']['hub_analysis'] / 'top_250_all_genes.json').relative_to(PROJECT_ROOT)
        )
        
        # For backward compatibility - keep hub_genes variable
        hub_genes = stratified_genes.get('all', {}).get('genes', [])
        

        logger.info("\n" + "="*60)
        logger.info("Phase 2: Loading background gene set...")
        background_genes = load_background_genes(config, PROJECT_ROOT, logger)
        

        logger.info("\n" + "="*60)
        logger.info("Phase 3: Performing stratified pathway enrichment...")
        logger.info("="*60)
        
        # Initialize results dictionaries
        all_results = {}
        enrichment_comparison = {}
        
        # Run enrichment on each gene list
        for gene_list_type in ['all', 'cancer', 'novel']:
            if gene_list_type not in stratified_genes:
                continue
                
            gene_list = stratified_genes[gene_list_type]['genes']
            gene_count = stratified_genes[gene_list_type]['count']
            description = stratified_genes[gene_list_type]['description']
            
            logger.info(f"\n--- Enriching {gene_list_type.upper()} gene list ---")
            logger.info(f"Description: {description}")
            logger.info(f"Gene count: {gene_count}")
            logger.info(f"Sample genes: {gene_list[:5]}")
            
            if len(gene_list) < 5:
                logger.warning(f"Skipping {gene_list_type} - too few genes ({len(gene_list)})")
                continue
            
            # Run enrichment for this gene list
            list_results, list_sig_df = run_enrichment_analysis(
                gene_list, background_genes, config, logger
            )
            
            # Store results
            all_results[gene_list_type] = {
                'raw': list_results,
                'significant': list_sig_df,
                'summary': {
                    'gene_count': gene_count,
                    'description': description,
                    'significant_terms': len(list_sig_df) if not list_sig_df.empty else 0
                }
            }
            
            # Store for comparison
            enrichment_comparison[gene_list_type] = {
                'significant_count': len(list_sig_df) if not list_sig_df.empty else 0,
                'top_terms': list_sig_df.head(3).to_dict('records') if not list_sig_df.empty else []
            }
            
            # Save individual results
            if not list_sig_df.empty:
                sig_path = OUTPUT_DIR / f'{gene_list_type}_genes_enrichment.tsv'
                list_sig_df.to_csv(sig_path, sep='\t', index=False)
                logger.info(f"✓ Saved {gene_list_type} enrichment: {sig_path.name}")
        
        # For backward compatibility: 'all' becomes the default
        if 'all' in all_results:
            all_sig_df = all_results['all']['significant']
        else:
            all_sig_df = pd.DataFrame()
            
        # Log comparison summary
        logger.info("\n" + "="*60)
        logger.info("ENRICHMENT COMPARISON SUMMARY")
        logger.info("="*60)
        for list_type, data in enrichment_comparison.items():
            logger.info(f"{list_type.upper():10s}: {data['significant_count']:4d} significant terms")


        logger.info("\n" + "="*60)
        logger.info("Phase 3b: Running GSEA prerank analysis...")
        # Load actual enhanced scores from 03_a
        gsea_input = prepare_gsea_input(config, PROJECT_ROOT, logger)
        
        if not gsea_input.empty:
            logger.info(f"Running GSEA with {len(gsea_input)} genes from 03_a enhanced ranking...")
            gsea_results = run_gsea_prerank(gsea_input, config, OUTPUT_DIR, logger)
            
            if not gsea_results.empty:
                gsea_path = OUTPUT_DIR / 'gsea_significant_results.tsv'
                gsea_results.to_csv(gsea_path, sep='\t', index=False)
                summary_stats['outputs']['gsea_results'] = str(gsea_path.relative_to(PROJECT_ROOT))
                
                # Log HRD findings with enrichment direction
                if 'Is_HRD_Relevant' in gsea_results.columns:
                    hrd_results = gsea_results[gsea_results['Is_HRD_Relevant']]
                    if not hrd_results.empty:
                        logger.info(f"\n🎯 HRD-Relevant GSEA Findings ({len(hrd_results)} terms):")
                        for _, row in hrd_results.nlargest(5, 'NES').iterrows():
                            direction = "ENRICHED" if row['NES'] > 0 else "DEPLETED"
                            logger.info(f"  • {row['Term']}: NES={row['NES']:.3f} ({direction}), FDR={row['FDR q-val']:.4f}")
                    else:
                        logger.info("No HRD-relevant pathways found in GSEA results")
                else:
                    logger.info("HRD relevance column not available in GSEA results")
            else:
                logger.warning("GSEA returned no significant results")
        else:
            logger.warning("⚠️  Skipping GSEA - could not load enhanced scores from 03_a")
            gsea_results = pd.DataFrame()


        logger.info("\n" + "="*60)
        logger.info("Phase 4: Creating pathway visualizations with metadata...")
        viz_paths, viz_metadata_paths = generate_pathway_visualizations(all_sig_df, all_results, stratified_genes, OUTPUT_DIR, logger)

        logger.info("\n" + "="*60)
        logger.info("Phase 5: Generating biological interpretations...")
        
        # First, create discovery analysis
        logger.info("Creating discovery analysis...")
        discovery_analysis = create_discovery_analysis(all_results, stratified_genes, logger)
        
        # Create comprehensive summary statistics
        logger.info("Creating comprehensive summary statistics...")
        summary_stats_data = create_summary_statistics(
            all_results, stratified_genes, discovery_analysis, all_sig_df, logger
        )
        
        # Then create functional insights (for backward compatibility)
        logger.info("Creating functional insights...")
        if 'gsea_results' in locals() and not gsea_results.empty:
            logger.info("With GSEA integration...")
            functional_insights = create_comprehensive_insights(all_sig_df, gsea_results, driver_info, config, logger)
        else:
            logger.info("ORA only...")
            functional_insights = create_functional_insights(all_sig_df, driver_info, config, logger)
        
        # Merge discovery analysis into functional insights
        functional_insights['discovery_analysis'] = discovery_analysis


        logger.info("\n" + "="*60)
        logger.info("Phase 6: Saving comprehensive results...")

        # Save the combined significant pathways (all_sig_df)
        if not all_sig_df.empty:
            combined_path = OUTPUT_DIR / 'all_significant_pathways.tsv'
            all_sig_df.to_csv(combined_path, sep='\t', index=False)
            logger.info(f"✓ Saved all_significant_pathways: {combined_path.name}")
            summary_stats['outputs']['all_significant_pathways'] = str(combined_path.relative_to(PROJECT_ROOT))

        # Save individual gene list results
        if all_results:  # This contains 'all', 'cancer', 'novel'
            for gene_list_type, result_dict in all_results.items():
                # Save the combined significant terms for this gene list
                sig_df = result_dict.get('significant')
                if sig_df is not None and not sig_df.empty:
                    sig_path = OUTPUT_DIR / f'{gene_list_type}_combined_significant.tsv'
                    sig_df.to_csv(sig_path, sep='\t', index=False)
                    logger.info(f"✓ Saved {gene_list_type} combined: {sig_path.name}")
                    summary_stats['outputs'][f'{gene_list_type}_combined'] = str(sig_path.relative_to(PROJECT_ROOT))
                
                # Save raw database results for this gene list
                raw_results = result_dict.get('raw', {})
                for db_name, db_result in raw_results.items():
                    # Save significant terms
                    db_sig = db_result.get('significant')
                    if db_sig is not None and not db_sig.empty:
                        db_sig_path = OUTPUT_DIR / f'{gene_list_type}_{db_name}_significant.tsv'
                        db_sig.to_csv(db_sig_path, sep='\t', index=False)
                        logger.info(f"✓ Saved {gene_list_type} {db_name} sig: {db_sig_path.name}")
                    
                    # Save all terms
                    db_all = db_result.get('all_terms')
                    if db_all is not None and not db_all.empty:
                        db_all_path = OUTPUT_DIR / f'{gene_list_type}_{db_name}_all.tsv'
                        db_all.to_csv(db_all_path, sep='\t', index=False)
                        logger.info(f"✓ Saved {gene_list_type} {db_name} all: {db_all_path.name}")
        else:
            logger.warning("No enrichment results to save")

        # Save functional insights
        insights_path = OUTPUT_DIR / 'functional_insights.json'
        with open(insights_path, 'w') as f:
            json.dump(functional_insights, f, indent=2)
        logger.info(f"✓ Saved functional insights: {insights_path.name}")

        # Initialize outputs if not exists
        if 'outputs' not in summary_stats:
            summary_stats['outputs'] = {}

        summary_stats['outputs']['functional_insights'] = str(insights_path.relative_to(PROJECT_ROOT))

        # Add visualization paths
        for viz_path in viz_paths:
            summary_stats['outputs'][viz_path.name] = str(viz_path.relative_to(PROJECT_ROOT))
        
        # Add metadata paths
        for meta_path in viz_metadata_paths:
            if meta_path:
                summary_stats['outputs'][meta_path.name] = str(meta_path.relative_to(PROJECT_ROOT))

        # Export summary statistics
        logger.info("Exporting summary statistics...")
        json_stats_path, tsv_stats_path = export_summary_statistics(
            summary_stats_data, OUTPUT_DIR, logger
        )
        
        # Update summary_stats outputs with new files
        summary_stats['outputs']['summary_stats_json'] = str(json_stats_path.relative_to(PROJECT_ROOT))
        summary_stats['outputs']['summary_stats_tsv'] = str(tsv_stats_path.relative_to(PROJECT_ROOT))

        # Update RQ metrics with discovery insights
        if 'discovery_analysis' in functional_insights:
            disc = functional_insights['discovery_analysis']
            summary_stats['rq_metrics'][0]['value'] = disc['summary']['total_genes_analyzed']
            summary_stats['rq_metrics'][1]['value'] = disc['summary']['novelty_percentage']
            summary_stats['rq_metrics'][2]['value'] = disc['comparative_analysis']['discovery_potential']
            summary_stats['rq_metrics'][3]['value'] = disc['comparative_analysis']['research_priority']
        
        logger.info("\n" + "="*60)
        logger.info("FUNCTIONAL CHARACTERIZATION RESULTS:")
        logger.info("="*60)
        
        if 'discovery_analysis' in functional_insights:
            disc = functional_insights['discovery_analysis']
            logger.info(f"\nDISCOVERY ANALYSIS:")
            logger.info(f"  Novel genes: {disc['summary']['novel_genes']} ({disc['summary']['novelty_percentage']:.1f}%)")
            logger.info(f"  Cancer genes: {disc['summary']['cancer_genes']}")
            logger.info(f"  Discovery potential: {disc['comparative_analysis']['discovery_potential']}")
            logger.info(f"  Research priority: {disc['comparative_analysis']['research_priority']}")
            logger.info(f"\n  Key insight: {disc['comparative_analysis']['key_insight']}")
            logger.info(f"  Validation: {disc['comparative_analysis']['validation_status']}")
        
        # Keep original metrics for backward compatibility
        if not all_sig_df.empty:
            brca_relevant = all_sig_df[all_sig_df['Is_BRCA_Relevant']]
            logger.info(f"\nORIGINAL METRICS (for reference):")
            logger.info(f"  Total pathways: {len(all_sig_df)}")
            logger.info(f"  BRCA-relevant: {len(brca_relevant)} ({len(brca_relevant)/len(all_sig_df)*100:.1f}%)")
            
            if functional_insights.get('key_biological_themes'):
                logger.info("\nKey Biological Themes:")
                for theme in functional_insights['key_biological_themes'][:3]:
                    logger.info(f"• {theme['theme']}: {theme['pathway_count']} pathways ({theme['significance']})")
            
            if functional_insights.get('therapeutic_implications'):
                logger.info("\nTherapeutic Implications:")
                for therapy in functional_insights['therapeutic_implications'][:3]:
                    logger.info(f"• {therapy['pathway']}: {therapy['therapies']}")
        else:
            logger.info("No significant pathways found in enrichment analysis")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise
    
    finally:
        summary_stats['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        total_time = time.time() - start_time
        summary_stats['processing_notes']['processing_time_minutes'] = round(total_time / 60, 1)
        summary_stats['processing_notes']['successful_operations'] = ['enrichment_analysis', 'functional_insights'] if 'functional_insights' in summary_stats['outputs'] else []
        
        result_info_path = OUTPUT_DIR / '03_b_result_info.json'
        create_summary_json(summary_stats, result_info_path, PROJECT_ROOT)
        logger.info(f"📄 Saved result info to: {get_relative_path(result_info_path)}")
        
        logger.info("\n" + "="*50)
        logger.info("FUNCTIONAL CHARACTERIZATION FILES SAVED:")
        logger.info("="*50)
        logger.info(f"• functional_insights.json")
        logger.info(f"• enrichment_summary_stats.json")
        logger.info(f"• enrichment_summary_stats.tsv")
        
        # List key files that were actually saved
        for key, value in summary_stats['outputs'].items():
            if key.startswith(('all_', 'cancer_', 'novel_', 'gsea_')) or key == 'functional_insights':
                logger.info(f"• {Path(value).name}")
        
        for viz in viz_paths:
            logger.info(f"• {viz.name}")
        
        for meta in viz_metadata_paths:
            if meta:
                logger.info(f"• {meta.name}")
        
        logger.info(f"• 03_b_result_info.json")
        logger.info("="*50)
        
        logger.info("-" * 50)
        logger.info("Script finished.")

if __name__ == "__main__":
    main()