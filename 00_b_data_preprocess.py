"""
00_b_data_preprocess.py

Script Purpose:
This script preprocesses raw gene expression data from the previous analysis (00_a) by filtering genes based on configurable criteria (e.g., protein-coding only, expression thresholds), extracting TPM values in parallel from TSV files, constructing tumor and normal expression matrices, saving metadata (e.g., sample mappings, gene info, filtered genes), and generating a verification report with previews and summaries. It ensures data quality for downstream analyses like differential expression.

Summary Logic:
1. Load inputs from 00_a (samples, genes, tumor/normal files); set up logging and output structure.
2. Filter gene keys based on config (e.g., protein-coding, then expression thresholds across samples).
3. Parallel-process TSV files to extract TPM for filtered genes.
4. Build transposed matrices (genes x samples) for tumor and normal, save as TSV.
5. Save metadata JSONs (sample mapping, gene metadata, enhanced filtering summary).
6. Generate verification report (previews of files, matrices) and log completion.

Key Features:
- Configurable filtering (gene types, min % samples expressed > threshold).
- Parallel processing with multiprocessing for efficiency on large datasets.
- Memory-efficient: reads only needed columns, uses float32 for TPM.
- Comprehensive logging and verification: previews, summaries, error handling.
- Enhanced filtering summary with expression range distribution and percentages.
- Output structure: matrices/, metadata/, logs/ for easy access.
- Reproducible JSON output with sorted keys for consistent comparisons.

Dependencies: See imports below. Assumes utils.config, utils.file; inputs from 00_a_analyse_raw_data.py.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
import datetime
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
from utils.config import load_config
from utils.file import get_relative_path, ensure_dir, get_auto_output_path

# Load configuration and set up paths
config = load_config()
PROJECT_ROOT = Path(config['paths']['project_root'])
DATA_DIR = Path(config['paths']['data_dir'])
RAW_DATA_DIR = Path(config['paths']['raw_data'])
INPUT_ANALYSIS_RAW = Path(config['paths']['analysis_raw'])  # Input from 00_a

# Use auto-generated output path
OUTPUT_DIR = get_auto_output_path(__file__, PROJECT_ROOT)

# Get processing parameters from config
PREPROCESSING_CONFIG = config['preprocessing']
PARALLEL_CONFIG = PREPROCESSING_CONFIG['parallel_processing']


def setup_logging(config, output_dir):
    """
    Set up logging with different formats for file and console.
    
    Creates a log directory, configures file handler (detailed) and optional console (simple),
    clears existing handlers to avoid duplicates.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(config['logging']['level'])
    
    # Avoid adding handlers if they already exist
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create log directory in output folder
    log_dir = output_dir / 'logs'
    ensure_dir(log_dir)
    log_file = log_dir / 'preprocessing.log'
    
    # Handler for the log file (with full details)
    file_handler = logging.FileHandler(log_file, mode='w')
    file_formatter = logging.Formatter(config['logging']['format'])
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Handler for the console (with clean, simple messages)
    if config['logging']['console_log']:
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
    return logger


def process_tsv_file(file_path, gene_keys_to_keep):
    """
    Process a single TSV file to extract TPM for specified genes.
    
    Optimizations applied:
    - Reads only required columns (reduces memory by 50-80%)
    - Uses float32 for TPM values (halves memory usage)
    - Uses dict comprehension for fast lookup
    
    Reads only required columns, creates gene keys, filters to kept genes, returns dict of TPM values.
    Handles errors gracefully with logging.
    """
    file_id = file_path.parent.name
    try:
        # Read only needed columns with optimal dtypes
        df = pd.read_csv(
            file_path, 
            sep='\t', 
            usecols=['gene_id', 'gene_name', 'tpm_unstranded'],
            dtype={'gene_id': str, 'gene_name': str, 'tpm_unstranded': np.float32},
            comment='#'
        )
        
        # Create gene keys and filter using vectorized operations
        df['gene_key'] = df['gene_id'] + '|' + df['gene_name']
        
        # Filter efficiently using boolean indexing (vectorized)
        mask = df['gene_key'].isin(gene_keys_to_keep)
        filtered_df = df[mask]
        
        # Use dict comprehension for faster creation
        tpm_dict = dict(zip(filtered_df['gene_key'], filtered_df['tpm_unstranded']))
        return {'file_id': file_id, 'tpm': tpm_dict}
    except Exception as e:
        logging.getLogger(__name__).error(f"Error processing {file_path}: {e}")
        return {'file_id': file_id, 'error': str(e)}


def _write_file_preview(summary_handle, title, file_path, num_lines):
    """Helper to write a preview of a text-based file to the summary."""
    summary_handle.write(f"### {title} ({file_path.name}) ###\n")
    try:
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= num_lines:
                    break
                summary_handle.write(line)
        summary_handle.write("\n" + "-"*60 + "\n\n")
    except FileNotFoundError:
        summary_handle.write(f"ERROR: File not found at {file_path}\n\n")


def _write_matrix_summary(summary_handle, title, file_path):
    """Helper to write a matrix preview and line count in a single pass."""
    summary_handle.write(f"### {title} ({file_path.name}) ###\n")
    try:
        line_count = 0
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if i < 2:
                    summary_handle.write(line)
                line_count += 1
        summary_handle.write(f"\nTotal rows (including header): {line_count}\n")
        summary_handle.write("\n" + "-"*60 + "\n\n")
    except FileNotFoundError:
        summary_handle.write(f"ERROR: File not found at {file_path}\n\n")


def _create_preview_matrices(preprocessed_dir, num_rows=25):
    """Helper to create preview copies of the large matrix files."""
    preview_dir = preprocessed_dir / 'matrices_preview'
    ensure_dir(preview_dir)
    
    for matrix_type in ['tumor', 'normal']:
        input_path = preprocessed_dir / 'matrices' / f'{matrix_type}_matrix.tsv'
        output_path = preview_dir / f'{matrix_type}_matrix_preview.tsv'
        
        if not input_path.exists():
            logging.warning(f"Matrix file not found, skipping preview creation: {input_path}")
            continue

        with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
            for i, line in enumerate(infile):
                if i >= num_rows:
                    break
                outfile.write(line)


def verify_preprocessing_results(output_dir):
    """
    Creates a summary report and preview matrices to verify outputs.
    
    Writes a timestamped report with previews of logs, JSONs, and matrix summaries;
    also creates truncated matrix previews for quick inspection.
    """
    logger = logging.getLogger(__name__)
    output_file = output_dir / '00_b_result_info.txt'
    
    logger.info(f"Generating verification report to {get_relative_path(output_file)}...")
    with open(output_file, 'w') as f:
        f.write(f"Verification Report for Preprocessed Files\n")
        f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        _write_file_preview(f, "Log Preview", output_dir / 'logs/preprocessing.log', 20)
        _write_file_preview(f, "Filtered Genes Summary", output_dir / 'metadata/filtered_genes.json', 50)
        _write_file_preview(f, "Gene Metadata Preview", output_dir / 'metadata/gene_metadata.json', 20)
        _write_file_preview(f, "Sample Mapping Preview", output_dir / 'metadata/sample_mapping.json', 20)
        _write_matrix_summary(f, "Tumor Matrix Summary", output_dir / 'matrices/tumor_matrix.tsv')
        _write_matrix_summary(f, "Normal Matrix Summary", output_dir / 'matrices/normal_matrix.tsv')

    logger.info(f"Verification report saved to {get_relative_path(output_file)}")
    
    try:
        _create_preview_matrices(output_dir)
        logger.info(f"Created preview matrices in {get_relative_path(output_dir / 'matrices_preview')}")
    except Exception as e:
        logger.error(f"Failed to create preview matrices: {e}")


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


def main():
    """Main execution function: Orchestrates preprocessing, filtering, matrix building, and verification."""
    # Setup logging with auto-generated output directory
    logger = setup_logging(config, OUTPUT_DIR)
    
    # Ensure output directory structure
    ensure_dir(OUTPUT_DIR / 'matrices')
    ensure_dir(OUTPUT_DIR / 'metadata')
    
    logger.info(f"00_a input directory: {get_relative_path(INPUT_ANALYSIS_RAW)}")
    logger.info(f"00_b output directory: {get_relative_path(OUTPUT_DIR)}")
    logger.info("Starting preprocessing script...")
    logger.info("")

    # Load inputs from previous script (00_a)
    try:
        with open(INPUT_ANALYSIS_RAW / 'samples_list.json', 'r') as f:
            samples_dict = json.load(f)
        with open(INPUT_ANALYSIS_RAW / 'genes_list.json', 'r') as f:
            genes_dict = json.load(f)
        with open(INPUT_ANALYSIS_RAW / 'files_tumor.json', 'r') as f:
            tumor_files_info = json.load(f)
        with open(INPUT_ANALYSIS_RAW / 'files_normal.json', 'r') as f:
            normal_files_info = json.load(f)
    except FileNotFoundError as e:
        logger.error(f"Missing input file from 00_a: {e}")
        logger.error("Please run 00_a_analyse_raw_data.py first")
        return

    # Initial gene filtering based on config (e.g., protein-coding only)
    logger.info("Filtering genes based on config rules...")
    gene_filters = PREPROCESSING_CONFIG['gene_filters']
    
    # Use dictionary comprehension for faster filtering
    if "protein_coding" in gene_filters.get('keep_gene_types', []):
        # Vectorized filtering: use dict comprehension with conditional
        initial_gene_keys = {
            key for key, info in genes_dict.items() if info.get('is_protein_coding', False)
        }
        logger.info(f"Filtered for {len(initial_gene_keys)} protein-coding genes.")
    else:
        # Use set for O(1) lookups during processing
        initial_gene_keys = set(genes_dict.keys())
        logger.info(f"Keeping all {len(initial_gene_keys)} gene types.")
    logger.info("")

    # Extract TPM data in parallel from all valid TSV files
    logger.info("Extracting TPM data from all TSV files in parallel...")
    
    # Pre-compute file paths list comprehension (faster than repeated checks)
    tumor_file_ids = list(tumor_files_info['list'].keys())
    normal_file_ids = list(normal_files_info['list'].keys())
    
    all_file_paths = [
        RAW_DATA_DIR / file_id / samples_dict[file_id]['file_name']
        for file_id in tumor_file_ids + normal_file_ids
        if file_id in samples_dict  # Fast dict lookup (O(1))
    ]
    
    # Filter paths that exist in a single pass
    all_file_paths = [p for p in all_file_paths if p.exists()]
    
    # Configure parallel processing
    process_func = partial(process_tsv_file, gene_keys_to_keep=initial_gene_keys)
    
    # Get n_jobs inside main function
    n_jobs = PARALLEL_CONFIG['n_jobs']
    if n_jobs == -1: 
        n_jobs = cpu_count()
        
    # Process files in parallel - optimal for CPU-bound file reading
    with Pool(n_jobs) as pool:
        results = list(tqdm(pool.imap_unordered(process_func, all_file_paths), 
                          total=len(all_file_paths), 
                          desc="Extracting TPM data"))
    
    # Aggregate successful TPM data, log errors
    # Use dict comprehension for faster aggregation
    tpm_data = {r['file_id']: r['tpm'] for r in results if 'error' not in r}
    
    error_count = sum(1 for r in results if 'error' in r)
    if error_count > 0:
        logger.warning(f"Errors in {error_count} files.")
    logger.info("")

    # Secondary filtering: expression threshold across samples
    logger.info("Filtering genes based on expression thresholds...")
    min_sample_pct = gene_filters['min_sample_percentage']
    tpm_threshold = gene_filters['expression_threshold']
    total_samples = len(tpm_data)
    min_samples_expressed = int(total_samples * min_sample_pct)

    # Pre-compute gene list for iteration
    initial_gene_list = list(initial_gene_keys)
    genes_kept = []
    genes_excluded_details = []
    
    # Optimize by pre-computing sample list
    sample_ids = list(tpm_data.keys())
    
    # Process genes with vectorized operations where possible
    for gene_key in tqdm(initial_gene_list, desc="Filtering by expression"):
        # Count samples where gene is expressed above threshold
        expressed_count = 0
        for sample_id in sample_ids:
            if tpm_data[sample_id].get(gene_key, 0.0) > tpm_threshold:
                expressed_count += 1
        
        expression_pct = (expressed_count / total_samples * 100) if total_samples > 0 else 0
        
        if expressed_count >= min_samples_expressed:
            genes_kept.append(gene_key)
        else:
            genes_excluded_details.append({
                'gene_key': gene_key,
                'expressed_in_samples': expressed_count,
                'expression_percentage': round(expression_pct, 1),
                'reason': f'Expressed in {expressed_count}/{total_samples} samples ({expression_pct:.1f}%)'
            })
    
    logger.info(f"Kept {len(genes_kept)} genes. Excluded {len(genes_excluded_details)}.")
    logger.info("")

    # Build expression range distribution summary
    logger.info("Calculating expression range distribution...")
    range_buckets = {
        '0-10%': {'genes_count': 0, 'range': '0.0-10.0% of samples'},
        '10-20%': {'genes_count': 0, 'range': '10.0-20.0% of samples'},
        '20-30%': {'genes_count': 0, 'range': '20.0-30.0% of samples'},
        '30-40%': {'genes_count': 0, 'range': '30.0-40.0% of samples'},
        '40-50%': {'genes_count': 0, 'range': '40.0-50.0% of samples'}
    }
    
    # Vectorized distribution calculation
    for gene_info in genes_excluded_details:
        pct = gene_info['expression_percentage']
        if pct < 10:
            range_buckets['0-10%']['genes_count'] += 1
        elif pct < 20:
            range_buckets['10-20%']['genes_count'] += 1
        elif pct < 30:
            range_buckets['20-30%']['genes_count'] += 1
        elif pct < 40:
            range_buckets['30-40%']['genes_count'] += 1
        else:  # 40-50%
            range_buckets['40-50%']['genes_count'] += 1
    
    # Calculate percentages efficiently
    total_genes = len(genes_kept) + len(genes_excluded_details)
    excluded_count = len(genes_excluded_details)
    
    # Pre-compute percentages to avoid repeated division
    excluded_div = excluded_count if excluded_count > 0 else 1
    total_div = total_genes if total_genes > 0 else 1
    
    for bucket_key in range_buckets:
        count = range_buckets[bucket_key]['genes_count']
        range_buckets[bucket_key]['pct_of_excluded'] = round((count / excluded_div * 100), 1)
        range_buckets[bucket_key]['pct_of_total_genes'] = round((count / total_div * 100), 1)

    # Construct matrices: genes (rows) x samples (columns), with sample IDs prefixed
    logger.info("Constructing tumor and normal expression matrices...")
    
    # Use list comprehensions for faster sample ID generation
    tumor_sample_ids = tumor_file_ids
    normal_sample_ids = normal_file_ids
    
    # Build matrices using vectorized operations where possible
    # Convert genes_kept to list for consistent ordering
    genes_kept_list = list(genes_kept)
    
    # Build tumor matrix using dictionary comprehension
    tumor_data = {}
    for i, file_id in enumerate(tumor_sample_ids):
        sample_tpm = tpm_data.get(file_id, {})
        # Use dict comprehension with fallback value
        tumor_data[f"t{i+1:04d}"] = {
            gene: sample_tpm.get(gene, 0.0) for gene in genes_kept_list
        }
    
    # Build normal matrix using dictionary comprehension
    normal_data = {}
    for i, file_id in enumerate(normal_sample_ids):
        sample_tpm = tpm_data.get(file_id, {})
        normal_data[f"n{i+1:04d}"] = {
            gene: sample_tpm.get(gene, 0.0) for gene in genes_kept_list
        }
    
    # Create DataFrames with orientation optimization
    tumor_matrix = pd.DataFrame.from_dict(tumor_data, orient='columns')
    normal_matrix = pd.DataFrame.from_dict(normal_data, orient='columns')
    tumor_matrix.index.name, normal_matrix.index.name = "gene_key", "gene_key"
    
    # Save matrices as tab-separated TSV with optimal settings
    tumor_matrix.to_csv(OUTPUT_DIR / 'matrices/tumor_matrix.tsv', sep='\t', index=True)
    normal_matrix.to_csv(OUTPUT_DIR / 'matrices/normal_matrix.tsv', sep='\t', index=True)
    logger.info(f"Wrote tumor_matrix.tsv ({tumor_matrix.shape[0]}x{tumor_matrix.shape[1]})")
    logger.info(f"Wrote normal_matrix.tsv ({normal_matrix.shape[0]}x{normal_matrix.shape[1]})")
    logger.info("")

    # Save metadata: mappings and summaries
    logger.info("Saving metadata files (with sorted keys for reproducibility)...")
    sample_format, sample_mapping = PREPROCESSING_CONFIG['matrix_format'], {}
    
    # Build sample mapping with single loop per type
    for i, file_id in enumerate(tumor_sample_ids):
        sample_mapping[f"{sample_format['sample_prefix_tumor']}{i+1:04d}"] = samples_dict[file_id]
    for i, file_id in enumerate(normal_sample_ids):
        sample_mapping[f"{sample_format['sample_prefix_normal']}{i+1:04d}"] = samples_dict[file_id]
    
    # Sort sample mapping by sample ID (t001, t002, ..., n001, n002, ...)
    sorted_sample_mapping = dict(sorted(sample_mapping.items()))
    with open(OUTPUT_DIR / 'metadata/sample_mapping.json', 'w') as f:
        json.dump(sort_nested_dict(sorted_sample_mapping), f, indent=2)
    
    # Filter gene metadata efficiently and sort by gene key
    filtered_gene_metadata = {key: genes_dict[key] for key in genes_kept_list if key in genes_dict}
    sorted_gene_metadata = dict(sorted(filtered_gene_metadata.items()))
    with open(OUTPUT_DIR / 'metadata/gene_metadata.json', 'w') as f:
        json.dump(sort_nested_dict(sorted_gene_metadata), f, indent=2)
    
    # Save enhanced filtered genes summary (already sorted via sort_nested_dict)
    filtered_genes_summary = {
        'genes_kept': len(genes_kept_list),
        'genes_kept_pct': round((len(genes_kept_list) / total_div * 100), 1),
        'genes_excluded': excluded_count,
        'genes_excluded_pct': round((excluded_count / total_div * 100), 1),
        'filtering_criteria': {
            'expression_threshold_tpm': tpm_threshold,
            'min_sample_percentage': min_sample_pct * 100,  # Convert 0.5 to 50.0
            'total_samples': total_samples,
            'min_samples_required': min_samples_expressed
        },
        'excluded_summary_by_expression_range': range_buckets,
        'excluded_list': sorted(genes_excluded_details, key=lambda x: x['gene_key'])
    }
    
    with open(OUTPUT_DIR / 'metadata/filtered_genes.json', 'w') as f:
        json.dump(sort_nested_dict(filtered_genes_summary), f, indent=2)
    
    logger.info("Metadata files saved (with consistent ordering for reproducibility).")
    logger.info("")

    # Final verification and report
    verify_preprocessing_results(OUTPUT_DIR)
    logger.info("\nPreprocessing complete.")


if __name__ == "__main__":
    main()