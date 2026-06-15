# File: code/00_d_refresh_gene_classification.py
"""
00_d_refresh_gene_classification.py

Script Purpose:
This script reclassifies genes using the updated curated cancer gene lists
and enhance_cancer_detection() function without re-running API calls. It reads 
existing gene files from 00_c, applies the new classification logic with priority:
1. Curated gene lists (COSMIC, TCGA, ClinVar, etc.)
2. Pattern-based detection (fallback)

Use case: When enhance_cancer_detection() is updated OR curated lists are modified,
run this script to refresh classifications instead of re-running the entire 
00_c_gene_info.py pipeline.

Key Features:
- Uses curated gene lists (260+ genes from authoritative sources)
- Preserves all existing gene data (no API calls)
- Creates backup of original combined file
- Generates detailed reclassification report with source tracking
- Updates summary statistics
- Maintains alphabetical sorting in combined file
- Exports updated cancer_gene_classification.json

Dependencies: Requires 00_c_gene_info.py output to exist.

Usage:
    python code/00_d_refresh_gene_classification.py

Output:
    - Updates gene_info_combined.json with new classifications
    - Creates backup: gene_info_combined.json.backup
    - Generates refresh_classification_report.json
    - Exports cancer_gene_classification.json (if curated lists available)
"""

import json
import logging
from pathlib import Path
from tqdm import tqdm
import shutil
import sys

from utils.config import load_config
from utils.file import ensure_dir, get_relative_path, get_auto_output_path
from utils.genes import enhance_cancer_detection

# Import curated cancer gene lists
try:
    from utils.cancer_gene_lists import (
        classify_gene_symbol,
        get_cancer_gene_dict,
        get_gene_classification_with_source,
        export_gene_lists_json
    )
    USE_CURATED_LISTS = True
    print("✓ Loaded curated cancer gene lists (260 genes)")
except ImportError as e:
    print(f"⚠️  Warning: Could not load curated gene lists: {e}")
    print("   Falling back to pattern-based detection only")
    USE_CURATED_LISTS = False


def setup_logging(config, output_dir):
    """
    Set up logging for the refresh script.
    
    Args:
        config: Configuration dictionary
        output_dir: Output directory path
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG if config['logging'].get('level') == 'DEBUG' else logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()

    # File log
    log_dir = output_dir / 'logs'
    ensure_dir(log_dir)
    file_handler = logging.FileHandler(log_dir / 'refresh_classification.log', mode='w')
    file_formatter = logging.Formatter(config['logging']['format'])
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console log
    if config['logging']['console_log']:
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
    return logger


def classify_gene_cancer_relevance(gene_data: dict, gene_key: str, logger) -> tuple:
    """
    Classify gene cancer relevance using curated lists + pattern matching.
    
    Priority:
    1. Curated gene lists (exact match on symbol) - HIGHEST PRIORITY
    2. Pattern-based detection (fallback for new discoveries)
    
    Args:
        gene_data: Merged gene information dictionary
        gene_key: Gene key in format "ENSG00000141510|TP53"
        logger: Logger instance
    
    Returns:
        tuple: (classification, source, confidence)
            - classification: 'breast_cancer', 'cancer', or 'non_cancer'
            - source: String describing the classification source
            - confidence: 'high', 'medium', or 'low'
    """
    # Extract gene symbol from key
    if '|' in gene_key:
        gene_symbol = gene_key.split('|')[1]
    else:
        gene_symbol = gene_data.get('gene_name') or gene_data.get('gene_symbol', '')
    
    # PRIORITY 1: Check curated lists (if available)
    if USE_CURATED_LISTS and gene_symbol:
        try:
            result = get_gene_classification_with_source(gene_symbol)
            classification = result['classification']
            source = result['source']
            confidence = result['confidence']
            
            # If found in curated lists, return immediately
            if classification != 'non_cancer':
                logger.debug(f"  {gene_symbol:15s} → {classification:15s} ({source})")
                return classification, source, confidence
        except Exception as e:
            logger.warning(f"  Error using curated list for {gene_symbol}: {e}")
    
    # PRIORITY 2: Pattern-based detection (fallback)
    classification = enhance_cancer_detection(gene_data)
    source = "Pattern matching (description/summary)"
    confidence = "medium" if classification != 'non_cancer' else "low"
    
    if classification != 'non_cancer':
        logger.debug(f"  {gene_symbol:15s} → {classification:15s} ({source})")
    
    return classification, source, confidence


def load_existing_data(combined_path):
    """
    Load existing combined gene data.
    
    Args:
        combined_path: Path to gene_info_combined.json
    
    Returns:
        dict: Combined gene data or None if loading fails
    """
    logger = logging.getLogger(__name__)
    
    if not combined_path.exists():
        logger.error(f"Combined file not found: {get_relative_path(combined_path)}")
        logger.error("Please run 00_c_gene_info.py first")
        return None
    
    try:
        with open(combined_path, 'r') as f:
            data = json.load(f)
        
        # Check if it has the old structure (summary.total_found) or new structure
        total_genes = data.get('summary', {}).get('total_found', 0)
        if total_genes == 0:
            # Try counting from divisions directly
            total_genes = sum([
                len(data.get('breast_cancer', {})),
                len(data.get('cancer', {})),
                len(data.get('non_cancer', {}))
            ])
        
        logger.info(f"Loaded existing combined data with {total_genes} genes")
        return data
    except Exception as e:
        logger.error(f"Error loading combined data: {e}")
        return None


def reclassify_genes(gene_dir, combined_data):
    """
    Reclassify all genes using the updated classification function with curated lists.
    
    Args:
        gene_dir: Directory containing individual gene JSON files
        combined_data: Existing combined gene data
    
    Returns:
        tuple: (new_classifications, reclassification_stats, sample_reclassified, source_stats)
            - new_classifications: dict with breast_cancer/cancer/non_cancer divisions
            - reclassification_stats: Statistics on changes
            - sample_reclassified: List of sample reclassified genes
            - source_stats: Statistics by classification source
    """
    logger = logging.getLogger(__name__)
    
    gene_files = list(gene_dir.glob('*.json'))
    if not gene_files:
        logger.error(f"No gene files found in {get_relative_path(gene_dir)}")
        return None, None, None, None
    
    logger.info(f"Found {len(gene_files)} gene files for reclassification")
    
    # Track new classifications
    new_classifications = {
        'breast_cancer': {},
        'cancer': {},
        'non_cancer': {}
    }
    
    # Track classification sources
    source_stats = {}
    
    # Track reclassifications
    reclassification_stats = {
        'total_genes': len(gene_files),
        'unchanged': 0,
        'changed': 0,
        'by_category': {
            'breast_cancer': {'from': [], 'to': []},
            'cancer': {'from': [], 'to': []},
            'non_cancer': {'from': [], 'to': []}
        }
    }
    
    # Sample of reclassified genes (first 20)
    sample_reclassified = []
    
    # Process each gene file
    for gene_file in tqdm(gene_files, desc="Reclassifying genes", unit="gene"):
        try:
            with open(gene_file, 'r') as f:
                gene_data = json.load(f)
        except Exception as e:
            logger.warning(f"Error reading {gene_file.name}: {e}")
            continue
        
        gene_key = gene_data.get('key', gene_file.stem)
        
        # Get old classification from combined data
        old_class = None
        for division in ['breast_cancer', 'cancer', 'non_cancer']:
            if gene_key in combined_data.get(division, {}):
                old_class = division
                break
        
        if old_class is None:
            old_class = 'non_cancer'  # Default for genes not in combined data
        
        # Get new classification using updated function with curated lists
        new_class, source, confidence = classify_gene_cancer_relevance(gene_data, gene_key, logger)
        
        # Store in new classifications
        relative_path = f"00_c_gene_info/gene/{gene_file.name}"
        new_classifications[new_class][gene_key] = relative_path
        
        # Track sources
        if source not in source_stats:
            source_stats[source] = {'breast_cancer': 0, 'cancer': 0, 'non_cancer': 0}
        source_stats[source][new_class] += 1
        
        # Track reclassifications
        if old_class == new_class:
            reclassification_stats['unchanged'] += 1
        else:
            reclassification_stats['changed'] += 1
            if old_class in reclassification_stats['by_category']:
                reclassification_stats['by_category'][old_class]['from'].append(gene_key)
            reclassification_stats['by_category'][new_class]['to'].append(gene_key)
            
            # Add to sample if we have less than 20
            if len(sample_reclassified) < 20:
                gene_symbol = gene_key.split('|')[1] if '|' in gene_key else gene_key
                sample_reclassified.append({
                    'gene_key': gene_key,
                    'gene_symbol': gene_symbol,
                    'old': old_class,
                    'new': new_class,
                    'source': source,
                    'confidence': confidence
                })
    
    logger.info(f"Reclassification complete: {reclassification_stats['unchanged']} unchanged, "
                f"{reclassification_stats['changed']} changed")
    
    # Log source statistics
    if USE_CURATED_LISTS:
        logger.info("\nClassification sources:")
        for source, counts in sorted(source_stats.items()):
            total = sum(counts.values())
            logger.info(f"  {source}:")
            logger.info(f"    breast_cancer: {counts['breast_cancer']}")
            logger.info(f"    cancer: {counts['cancer']}")
            logger.info(f"    non_cancer: {counts['non_cancer']}")
            logger.info(f"    total: {total}")
    
    return new_classifications, reclassification_stats, sample_reclassified, source_stats


def update_combined_data(combined_data, new_classifications):
    """
    Update combined data with new classifications.
    
    Args:
        combined_data: Existing combined data dictionary
        new_classifications: New classifications by division
    
    Returns:
        dict: Updated combined data
    """
    logger = logging.getLogger(__name__)
    
    # Clear old classifications
    for division in ['breast_cancer', 'cancer', 'non_cancer']:
        combined_data[division] = {}
    
    # Add new classifications (sorted alphabetically)
    for division in new_classifications:
        sorted_items = dict(sorted(new_classifications[division].items()))
        combined_data[division] = sorted_items
        
        # Update summary counts
        if 'summary' not in combined_data:
            combined_data['summary'] = {}
        if division not in combined_data['summary']:
            combined_data['summary'][division] = {}
        
        combined_data['summary'][division]['count'] = len(sorted_items)
    
    # Update total found count
    total_found = (
        len(new_classifications['breast_cancer']) +
        len(new_classifications['cancer']) + 
        len(new_classifications['non_cancer'])
    )
    combined_data['summary']['total_found'] = total_found
    
    # Update total_missing if total_input exists
    if 'total_input' in combined_data['summary']:
        combined_data['summary']['total_missing'] = (
            combined_data['summary']['total_input'] - total_found
        )
    
    logger.info(f"Updated summary: {total_found} total genes")
    logger.info(f"  Breast cancer: {combined_data['summary']['breast_cancer']['count']}")
    logger.info(f"  Other cancer: {combined_data['summary']['cancer']['count']}")
    logger.info(f"  Non-cancer: {combined_data['summary']['non_cancer']['count']}")
    
    return combined_data


def create_backup(original_path):
    """
    Create a backup of the original file.
    
    Args:
        original_path: Path to file to backup
    
    Returns:
        Path: Backup file path or None if backup failed
    """
    logger = logging.getLogger(__name__)
    
    backup_path = original_path.with_suffix('.json.backup')
    backup_count = 1
    while backup_path.exists():
        backup_path = original_path.with_suffix(f'.json.backup{backup_count}')
        backup_count += 1
    
    try:
        shutil.copy2(original_path, backup_path)
        logger.info(f"Created backup at: {get_relative_path(backup_path)}")
        return backup_path
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        return None


def save_refresh_report(output_dir, reclassification_stats, sample_reclassified, source_stats):
    """
    Save a detailed reclassification report.
    
    Args:
        output_dir: Output directory
        reclassification_stats: Statistics on reclassifications
        sample_reclassified: Sample of reclassified genes
        source_stats: Statistics by classification source
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    report_path = output_dir / 'refresh_classification_report.json'
    
    # Calculate reclassification flows
    reclass_details = {}
    for old_class in ['breast_cancer', 'cancer', 'non_cancer']:
        reclass_details[f'from_{old_class}'] = {
            'count': len(reclassification_stats['by_category'][old_class]['from'])
        }
        
        # Count transitions
        for new_class in ['breast_cancer', 'cancer', 'non_cancer']:
            if old_class != new_class:
                transition_count = len([
                    g for g in reclassification_stats['by_category'][new_class]['to']
                    if g in reclassification_stats['by_category'][old_class]['from']
                ])
                reclass_details[f'from_{old_class}'][f'to_{new_class}'] = transition_count
    
    report_data = {
        'summary': {
            'total_genes_processed': reclassification_stats['total_genes'],
            'genes_unchanged': reclassification_stats['unchanged'],
            'genes_reclassified': reclassification_stats['changed'],
            'reclassification_rate': (
                reclassification_stats['changed'] / reclassification_stats['total_genes'] * 100
                if reclassification_stats['total_genes'] > 0 else 0
            ),
            'curated_lists_used': USE_CURATED_LISTS
        },
        'classification_sources': source_stats if USE_CURATED_LISTS else None,
        'reclassification_details': reclass_details,
        'sample_reclassified_genes': sample_reclassified
    }
    
    try:
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        logger.info(f"Refresh report saved to: {get_relative_path(report_path)}")
        return True
    except Exception as e:
        logger.error(f"Failed to save refresh report: {e}")
        return False


def main():
    """Main execution function."""
    # Load configuration
    config = load_config()
    PROJECT_ROOT = Path(config['paths']['project_root'])
    
    # Use same output directory as 00_c
    OUTPUT_DIR = PROJECT_ROOT / 'output' / '00_c_gene_info'
    
    # Setup logging
    logger = setup_logging(config, OUTPUT_DIR)
    
    logger.info("=" * 70)
    logger.info("00_d: REFRESH GENE CLASSIFICATION WITH CURATED LISTS")
    logger.info("=" * 70)
    logger.info("Script purpose: Reclassify genes using curated gene lists + pattern detection")
    logger.info(f"Curated lists enabled: {USE_CURATED_LISTS}")
    logger.info(f"Output directory: {get_relative_path(OUTPUT_DIR)}")
    
    # Verify required files exist
    GENE_DIR = OUTPUT_DIR / 'gene'
    COMBINED_PATH = OUTPUT_DIR / 'gene_info_combined.json'
    
    required_paths = [GENE_DIR, COMBINED_PATH]
    for path in required_paths:
        if not path.exists():
            logger.error(f"Required path not found: {get_relative_path(path)}")
            logger.error("Please run 00_c_gene_info.py first")
            return
    
    # Load existing data
    combined_data = load_existing_data(COMBINED_PATH)
    if not combined_data:
        return
    
    # Create backup
    backup_path = create_backup(COMBINED_PATH)
    if not backup_path:
        logger.warning("Proceeding without backup (risky!)")
    
    # Reclassify genes
    new_classifications, reclass_stats, sample_reclassified, source_stats = reclassify_genes(
        GENE_DIR, combined_data
    )
    
    if not new_classifications:
        logger.error("Failed to reclassify genes")
        return
    
    # Update combined data
    updated_data = update_combined_data(combined_data, new_classifications)
    
    # Save updated combined data
    try:
        with open(COMBINED_PATH, 'w') as f:
            json.dump(updated_data, f, indent=2, sort_keys=True)
        logger.info(f"Updated combined data saved to: {get_relative_path(COMBINED_PATH)}")
    except Exception as e:
        logger.error(f"Failed to save updated combined data: {e}")
        # Restore from backup if available
        if backup_path and backup_path.exists():
            try:
                shutil.copy2(backup_path, COMBINED_PATH)
                logger.info(f"Restored original from backup: {get_relative_path(backup_path)}")
            except Exception as restore_error:
                logger.error(f"Failed to restore from backup: {restore_error}")
        return
    
    # Save refresh report
    save_refresh_report(OUTPUT_DIR, reclass_stats, sample_reclassified, source_stats)
    
    # Export updated cancer gene dictionary (if using curated lists)
    if USE_CURATED_LISTS:
        logger.info("\n" + "="*60)
        logger.info("Exporting updated cancer gene dictionary...")
        
        try:
            dict_export_path = OUTPUT_DIR / 'cancer_gene_classification.json'
            export_gene_lists_json(str(dict_export_path))
            logger.info(f"✓ Exported to: {get_relative_path(dict_export_path)}")
        except Exception as e:
            logger.error(f"Failed to export gene dictionary: {e}")
    
    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("REFRESH COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total genes processed: {reclass_stats['total_genes']}")
    logger.info(f"Genes reclassified: {reclass_stats['changed']} "
                f"({reclass_stats['changed']/reclass_stats['total_genes']*100:.1f}%)")
    logger.info(f"Genes unchanged: {reclass_stats['unchanged']}")
    
    if sample_reclassified:
        logger.info(f"\nSample of reclassified genes (first {len(sample_reclassified)}):")
        for item in sample_reclassified:
            logger.info(f"  {item['gene_symbol']:15s}: {item['old']:15s} → {item['new']:15s} "
                       f"(source: {item['source']}, confidence: {item['confidence']})")
    
    if backup_path:
        logger.info(f"\nBackup saved at: {get_relative_path(backup_path)}")
    
    logger.info("\nTo verify the classification:")
    logger.info("  - Check: output/00_c_gene_info/refresh_classification_report.json")
    logger.info("  - Check: output/00_c_gene_info/gene_info_combined.json")
    if USE_CURATED_LISTS:
        logger.info("  - Check: output/00_c_gene_info/cancer_gene_classification.json")
    
    logger.info("\n" + "=" * 70)


if __name__ == "__main__":
    main()