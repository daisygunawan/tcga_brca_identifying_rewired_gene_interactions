"""
00_c_gene_info.py

Script Purpose:
This script enriches preprocessed gene metadata (from 00_b) by querying external APIs (MyGene.info for summaries/diseases, Ensembl for genomic details/descriptions). It merges results into structured JSONs, classifies genes by cancer relevance (breast cancer, other cancer, non-cancer), saves individual gene files conditionally (symbol match required), a grouped global list with relative paths, and verification reports with stats on successes/failures/not-founds. Designed for cancer genomics workflows, focusing on TCGA-BRCA-like data.

Summary Logic:
1. Load and clean gene list from 00_b's gene_metadata.json (strip Ensembl versions, create keys).
2. Batch-query MyGene.info API for layman info (name, summary, diseases); cache per batch.
3. Per-gene query Ensembl REST API for functional/genomic data; with timeout retries and per-gene caching.
4. Merge API results by gene_key, structure as core fields + misc_info dict.
5. Save individual JSONs only if API symbol matches original; build grouped JSON by cancer classification.
6. Generate verification JSON with stats, samples of not-founds, and grouped summary.
7. Use logging (file/console/debug) and relative paths for traceability; handle errors gracefully.

Key Features:
- API efficiency: Batching (MyGene), per-gene caching (Ensembl), retries/backoff for reliability.
- Data integrity: Conditional saving on symbol match; cancer detection via utils.genes.enhance_cancer_detection.
- Output organization: gene/ dir for individuals, api_responses/ for caches, logs/debug/ for auditing.
- Error-tolerant: Continues on failures, tracks not-founds/failed, logs warnings for mismatches.
- Configurable: Logging levels, batch sizes, timeouts via config.

Dependencies: See imports below. Assumes utils.config, utils.file, utils.genes; inputs from 00_b_data_preprocess.py; requires internet for APIs.
"""

import requests
import json
import logging
from pathlib import Path
from tqdm import tqdm
from utils.config import load_config
from utils.file import ensure_dir, get_relative_path, get_auto_output_path
from utils.genes import enhance_cancer_detection

# Import curated cancer gene lists
try:
    from utils.cancer_gene_lists import (
        classify_gene_symbol,
        classify_gene_id,
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


import time  # For retry backoff

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


def setup_logging(config, output_dir):
    """
    Set up logging with different formats for file, console, and debug log.
    
    Configures multiple handlers: detailed file log, simple console (if enabled),
    and debug log for API traces; clears existing handlers to prevent duplicates.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG if config['logging'].get('level') == 'DEBUG' else logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()

    # File log (append mode for continuity)
    log_dir = output_dir / 'logs'
    ensure_dir(log_dir)
    file_handler = logging.FileHandler(log_dir / 'gene_info.log', mode='a')
    file_formatter = logging.Formatter(config['logging']['format'])
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console log (simple messages only)
    if config['logging']['console_log']:
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
    # Debug log for API requests/responses (timestamped)
    debug_dir = output_dir / 'debug'
    ensure_dir(debug_dir)
    debug_handler = logging.FileHandler(debug_dir / 'api_debug.log', mode='a')
    debug_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    debug_handler.setFormatter(debug_formatter)
    logger.addHandler(debug_handler)
        
    return logger


def load_gene_list(input_preprocessed):
    """
    Load gene list from gene_metadata.json (output from 00_b).
    
    Parses JSON, strips Ensembl version numbers (post-dot), creates clean keys,
    builds list of dicts with id/symbol/key/original_key; logs warnings for malforms.
    """
    logger = logging.getLogger(__name__)
    input_path = input_preprocessed / 'metadata' / 'gene_metadata.json'
    try:
        with open(input_path, 'r') as f:
            gene_data = json.load(f)
        
        # Create gene_info list from the filtered gene metadata
        gene_info = []
        for gene_key, gene_info_data in gene_data.items():
            parts = gene_key.split('|')
            if len(parts) >= 2:
                # Remove version number from Ensembl ID (everything after period)
                ensembl_id_base = parts[0].split('.')[0]
                symbol = parts[1]
                # Create new key without version number
                new_key = f"{ensembl_id_base}|{symbol}"
                gene_info.append({
                    "id": ensembl_id_base,  # Use base ID without version
                    "symbol": symbol,
                    "key": new_key,  # Use new key without version
                    "original_key": gene_key  # Keep original for reference
                })
            else:
                logger.warning(f"Skipping malformed gene key: {gene_key}")
        
        # Sort gene_info by key for consistent processing
        gene_info.sort(key=lambda x: x['key'])
        
        logger.debug(f"Created gene_info: type={type(gene_info)}, length={len(gene_info)}, first few={gene_info[:5]}")
        return gene_info
    except Exception as e:
        logger.error(f"Error reading {input_path}: {e}")
        return []


def load_api_response(output_dir, api_type, batch_number):
    """Load existing API response from file if it exists."""
    logger = logging.getLogger(__name__)
    api_dir = output_dir / 'api_responses' / api_type
    output_path = api_dir / f"batch_{batch_number}.json"
    project_root = output_dir.parent
    try:
        if output_path.exists():
            with open(output_path, 'r') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    logger.error(f"Invalid cached {api_type} response for batch {batch_number}: expected list, got {type(data)}")
                    return None
                logger.info(f"Loaded existing {api_type} response for batch {batch_number}")
                return data
        return None
    except Exception as e:
        relative_path = output_path.relative_to(project_root)
        logger.error(f"Error loading {api_type} response for batch {batch_number} from {relative_path}: {e}")
        return None


def save_api_response(output_dir, api_type, batch_number, response_data):
    """Save API response to a JSON file with relative path logging."""
    logger = logging.getLogger(__name__)
    api_dir = output_dir / 'api_responses' / api_type
    ensure_dir(api_dir)
    output_path = api_dir / f"batch_{batch_number}.json"
    project_root = output_dir.parent
    try:
        with open(output_path, 'w') as f:
            # Sort the response data if it's a list of dictionaries
            if isinstance(response_data, list):
                # Sort by query field if available, otherwise by _id
                try:
                    response_data.sort(key=lambda x: str(x.get('query', x.get('_id', ''))).lower())
                except (AttributeError, KeyError):
                    # If sorting fails, keep original order
                    pass
            json.dump(response_data, f, indent=2, sort_keys=False)
        relative_path = output_path.relative_to(project_root)
        logger.info(f"Saved {api_type} response for batch {batch_number} to {relative_path}")
        return True
    except Exception as e:
        relative_path = output_path.relative_to(project_root)
        logger.error(f"Error saving {api_type} response for batch {batch_number} to {relative_path}: {e}")
        return False


def query_mygene_info(config, output_dir, gene_ids, batch_size=500):
    """
    Query MyGene.info API in batches for layman-friendly info.
    
    Uses POST for batch efficiency; fields: symbol,name,summary,reactome,pantherdb,disease.
    Loads/saves cache per batch; tracks stats (calls,successes,failures), not-founds.
    """
    logger = logging.getLogger(__name__)
    # Limit batch size for memory safety and better error handling
    batch_size = min(int(batch_size), 200)
    base_url = "http://mygene.info/v3/gene"
    fields = "symbol,name,summary,reactome,pantherdb,disease"
    results = {}
    not_found_genes = []
    
    stats = {
        'total_calls': 0,
        'success_count': 0,
        'failed_count': 0
    }

    if not isinstance(gene_ids, list):
        logger.error(f"gene_ids is not a list: type={type(gene_ids)}")
        return results, not_found_genes, stats
    
    # Ensure gene_ids are sorted for consistent batching
    gene_ids.sort()
    
    for i in tqdm(range(0, len(gene_ids), batch_size), desc="Querying MyGene.info", unit="batch"):
        batch = gene_ids[i:i + batch_size]
        batch_number = i // batch_size + 1
        stats['total_calls'] += 1
        
        cached_response = load_api_response(output_dir, "mygene", f"{batch_number:03d}")
        if cached_response:
            batch_results = cached_response
        else:
            try:
                payload = {'ids': ','.join(batch), 'fields': fields, 'species': 'human'}
                response = requests.post(base_url, data=payload, timeout=30)
                response.raise_for_status()
                batch_results = response.json()
                # Sort batch results for consistent caching
                batch_results.sort(key=lambda x: str(x.get('query', x.get('_id', ''))).lower())
                save_api_response(output_dir, "mygene", f"{batch_number:03d}", batch_results)
                stats['success_count'] += 1
            except requests.RequestException as e:
                logger.error(f"POST API request failed for batch {batch_number}: {e}")
                batch_results = []
                stats['failed_count'] += 1
                continue
        
        for result in batch_results:
            if 'notfound' in result:
                query_id = result.get('query')
                if query_id:
                    not_found_genes.append(query_id)
                continue
            gene_id = result.get('query', result.get('_id'))
            if gene_id:
                results[gene_id] = result
    
    # Sort not_found_genes for consistent output
    not_found_genes.sort()
    
    logger.info(f"MyGene.info query complete: {len(results)} found, {len(not_found_genes)} not found")
    return results, not_found_genes, stats


def query_ensembl_lookup(config, output_dir, gene_info, max_retries=3, base_timeout=10):
    """
    Query Ensembl REST API for detailed functional info with retries for timeouts and handling 400 errors.
    
    Uses /lookup/id/{gene_id} per gene; caches individually; exponential backoff on timeouts.
    Handles 400 (invalid ID) as failure without retry; tracks stats and failed genes.
    """
    logger = logging.getLogger(__name__)
    base_url = "https://rest.ensembl.org"
    results = {}
    failed_genes = []
    stats = {
        'total_calls': 0,
        'success_count': 0,
        'failed_count': 0
    }
    
    # Sort gene_info by ID for consistent processing
    sorted_gene_info = sorted(gene_info, key=lambda x: x['id'])
    
    for gene in tqdm(sorted_gene_info, desc="Querying Ensembl", unit="gene"):
        gene_id = gene['id']
        stats['total_calls'] += 1
        
        # Check cache first
        cache_dir = output_dir / 'api_responses' / 'ensembl'
        ensure_dir(cache_dir)
        cache_path = cache_dir / f"{gene_id}.json"
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    result = json.load(f)
                results[gene_id] = result
                stats['success_count'] += 1
                continue
            except Exception as e:
                logger.warning(f"Cache invalid for {gene_id}: {e}")
        
        retry_count = 0
        success = False
        while retry_count < max_retries and not success:
            try:
                # Use /lookup/id/{gene_id} endpoint
                endpoint = f"/lookup/id/{gene_id}"
                current_timeout = base_timeout + (retry_count * 5)  # Exponential backoff: 10, 15, 20s
                response = requests.get(
                    f"{base_url}{endpoint}", 
                    headers={"Content-Type": "application/json"}, 
                    params={"content-type": "application/json"}, 
                    timeout=current_timeout
                )
                if response.status_code == 400:
                    # Bad Request: Likely invalid/deprecated ID; treat as failed
                    logger.warning(f"Ensembl 400 Bad Request for {gene_id} (possibly invalid ID); skipping.")
                    failed_genes.append(gene_id)
                    stats['failed_count'] += 1
                    break
                response.raise_for_status()
                result = response.json()
                # Save to cache
                with open(cache_path, 'w') as f:
                    json.dump(result, f, indent=2, sort_keys=True)
                results[gene_id] = result
                stats['success_count'] += 1
                success = True
            except requests.exceptions.Timeout:
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = 2 ** retry_count  # Backoff wait
                    logger.warning(f"Timeout for {gene_id} (attempt {retry_count}/{max_retries}); retrying in {wait_time}s.")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Ensembl timeout exhausted for {gene_id}; marking as failed.")
                    failed_genes.append(gene_id)
                    stats['failed_count'] += 1
            except requests.RequestException as e:
                logger.error(f"Ensembl API failed for {gene_id}: {e}")
                failed_genes.append(gene_id)
                stats['failed_count'] += 1
                break  # No retry for non-timeout errors
            
            # Add small delay between requests to respect rate limits
            if success:
                time.sleep(0.1)  # Rate limit to ~10 requests/second
                break
    
    # Sort failed genes for consistent output
    failed_genes.sort()
    
    logger.info(f"Ensembl query complete: {len(results)} found, {len(failed_genes)} failed")
    return results, failed_genes, stats


def merge_results(mygene_results, ensembl_results, gene_info):
    """
    Merge results from different APIs and re-key the dictionary by gene_key.
    
    Structures output: core fields (key, gene_id, gene_name, gene_description, biotype, summary, name),
    then misc_info dict (genomic_location, canonical_transcript, etc., disease_associations).
    Skips if no matching key; logs warnings for mismatches.
    """
    logger = logging.getLogger(__name__)
    
    # Use gene_key (id|symbol) as the primary key for the final results dict.
    results_by_key = {}
    ensembl_to_key = {g['id']: g['key'] for g in gene_info}
    
    for gene_id, data in mygene_results.items():
        gene_key = ensembl_to_key.get(gene_id)
        if gene_key:
            # Build the structured dict in desired order
            gene_data = {}
            
            # Core fields first
            gene_data['key'] = gene_key
            gene_data['gene_id'] = gene_id
            if 'symbol' in data:
                gene_data['gene_name'] = data['symbol']
            
            # Merge Ensembl data if available
            ensembl_desc = ''
            misc_info = {}
            if gene_id in ensembl_results:
                ensembl_data = ensembl_results[gene_id]
                if ensembl_data:
                    ensembl_desc = ensembl_data.get('description', '')
                    
                    # Add to misc_info
                    misc_info['genomic_location'] = {
                        'chromosome': ensembl_data.get('seq_region_name', ''),
                        'start': ensembl_data.get('start', 0),
                        'end': ensembl_data.get('end', 0),
                        'strand': ensembl_data.get('strand', 0)
                    }
                    misc_info['canonical_transcript'] = ensembl_data.get('canonical_transcript', '')
                    misc_info['ensembl_version'] = ensembl_data.get('version', 0)
                    misc_info['genome_assembly'] = ensembl_data.get('assembly_name', '')
                    misc_info['biotype'] = ensembl_data.get('biotype', '')
            
            # Rename function to gene_description
            gene_data['gene_description'] = ensembl_desc
            
            # Biotype from Ensembl (already in misc_info)
            gene_data['biotype'] = misc_info.get('biotype', '')
            
            # Summary from MyGene
            gene_data['summary'] = data.get('summary', '')
            
            # Name from MyGene (additional field)
            if 'name' in data:
                gene_data['name'] = data['name']
            
            # Misc info (MyGene metadata + Ensembl extras)
            misc_info['query'] = data.get('query', '')
            misc_info['_id'] = data.get('_id', '')
            misc_info['_version'] = data.get('_version', 0)
            
            # Add disease info if available
            if 'disease' in data:
                # Sort disease associations for consistent output
                disease_data = data['disease']
                if isinstance(disease_data, list):
                    # Sort by disease name if available
                    disease_data.sort(key=lambda x: str(x.get('disease_name', '')).lower())
                misc_info['disease_associations'] = disease_data
                
            gene_data['misc_info'] = misc_info
            
            # Skip pantherdb and other extras for leanness
            if 'pantherdb' in data:
                del data['pantherdb']
            
            results_by_key[gene_key] = gene_data
        else:
            logger.warning(f"Could not find a gene_key for Ensembl ID {gene_id}. Skipping.")
            
    logger.debug(f"Merged results: {len(results_by_key)} genes processed")
    return results_by_key


def verify_results(output_dir, results, not_found_genes, mygene_stats, ensembl_stats, grouped_summary):
    """Save a verification report."""
    logger = logging.getLogger(__name__)
    output_file = output_dir / '00_c_result_info.json'
    project_root = output_dir.parent
    
    # Sort results keys for consistent verification output
    sorted_results_keys = sorted(results.keys())
    
    stats = {
        'mygene_stats': mygene_stats,
        'ensembl_stats': ensembl_stats,
        'not_found_genes_count': len(not_found_genes),
        'not_found_genes_sample': not_found_genes[:10],
        'grouped_summary': grouped_summary,
        'total_genes_processed': len(results),
        'sample_processed_genes': sorted_results_keys[:20]  # First 20 for quick reference
    }
    
    try:
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2, sort_keys=True)
        relative_path = output_file.relative_to(project_root)
        logger.info(f"Verification report saved to {relative_path}")
    except Exception as e:
        relative_path = output_file.relative_to(project_root)
        logger.error(f"Failed to create verification report to {relative_path}: {e}")


def save_results(output_dir, results, gene_info, original_symbols_map):
    """
    Save individual gene JSON files and a grouped global list.
    
    Saves individuals only if API symbol matches original (integrity check);
    builds grouped JSON by cancer classification (sorted alphabetically),
    with relative paths; returns summary dict for verification.
    """
    logger = logging.getLogger(__name__)
    gene_dir = output_dir / 'gene'
    ensure_dir(gene_dir)
    project_root = output_dir.parent
    
    # Sort results by key for consistent processing and output
    sorted_results_keys = sorted(results.keys())
    
    # Conditional saving of individual gene files
    failed_writes = []
    skipped_mismatch_count = 0
    mismatch_details = []  # Track mismatch details for logging
    
    for key in tqdm(sorted_results_keys, desc="Saving individual gene files", unit="gene"):
        gene_data = results[key]
        original_symbol = original_symbols_map.get(key)
        api_gene_name = gene_data.get('gene_name')

        # Only save the file if the original symbol matches the one from the API (case-insensitive)
        if original_symbol and api_gene_name and original_symbol.upper() == api_gene_name.upper():
            # Use the key (already without version) for filename
            output_path = gene_dir / f"{key}.json"
            try:
                with open(output_path, 'w') as f:
                    # Sort the gene data for consistent output
                    # Define field order for readability
                    sorted_gene_data = {}
                    # Add core fields in specific order
                    for field in ['key', 'gene_id', 'gene_name', 'gene_description', 
                                  'biotype', 'summary', 'name']:
                        if field in gene_data:
                            sorted_gene_data[field] = gene_data[field]
                    
                    # Add misc_info with sorted keys
                    if 'misc_info' in gene_data:
                        sorted_misc_info = dict(sorted(gene_data['misc_info'].items()))
                        # Ensure nested dictionaries are also sorted
                        for misc_key, misc_value in sorted_misc_info.items():
                            if isinstance(misc_value, dict):
                                sorted_misc_info[misc_key] = dict(sorted(misc_value.items()))
                        sorted_gene_data['misc_info'] = sorted_misc_info
                    
                    json.dump(sorted_gene_data, f, indent=2, sort_keys=False)
            except Exception as e:
                relative_path = output_path.relative_to(project_root)
                logger.error(f"Error saving {relative_path} for gene {key}: {e}")
                failed_writes.append(key)
        else:
            skipped_mismatch_count += 1
            mismatch_detail = f"{key}: original='{original_symbol}', API='{api_gene_name}'"
            mismatch_details.append(mismatch_detail)
            
    if skipped_mismatch_count > 0:
        logger.info(f"Skipped saving {skipped_mismatch_count} files due to symbol mismatches.")
        if mismatch_details:
            logger.info(f"First 10 symbol mismatches: {mismatch_details[:10]}")

    # Build summary and grouped data using gene_key
    global_path = output_dir / 'gene_info_combined.json'
    
    total_input = len(gene_info)
    total_found = len(results)
    
    grouped_data = {
        "summary": {
            "total_input": total_input,
            "total_found": total_found,
            "total_missing": total_input - total_found,
            "breast_cancer": {
                "count": 0,
                "description": "Genes with breast cancer-related terms in description or summary"
            },
            "cancer": {
                "count": 0,
                "description": "Genes with cancer-related terms (excluding breast cancer)"
            },
            "non_cancer": {
                "count": 0,
                "description": "Genes without cancer-related terms"
            }
        },
        "breast_cancer": {},
        "cancer": {},
        "non_cancer": {}
    }
    
    # Collect paths before sorting - USE ENHANCED CANCER DETECTION
    temp_grouped = {"breast_cancer": {}, "cancer": {}, "non_cancer": {}}
    for key in sorted_results_keys:
        gene_data = results[key]
        relative_path = str((gene_dir / f"{key}.json").relative_to(project_root))
        
        # Use enhanced cancer detection with curated lists
        classification, source, confidence = classify_gene_cancer_relevance(gene_data, key, logger)

        temp_grouped[classification][key] = relative_path
        grouped_data["summary"][classification]["count"] += 1

        # Optional: Track classification sources in summary
        if "classification_sources" not in grouped_data["summary"][classification]:
            grouped_data["summary"][classification]["classification_sources"] = {}
        if source not in grouped_data["summary"][classification]["classification_sources"]:
            grouped_data["summary"][classification]["classification_sources"][source] = 0
        grouped_data["summary"][classification]["classification_sources"][source] += 1

    
    # Sort keys alphabetically in each division
    for division in temp_grouped:
        sorted_keys = sorted(temp_grouped[division].keys())
        grouped_data[division] = {k: temp_grouped[division][k] for k in sorted_keys}
    
    try:
        with open(global_path, 'w') as f:
            json.dump(grouped_data, f, indent=2, sort_keys=True)
        relative_path = global_path.relative_to(project_root)
        logger.info(f"Global list saved to {relative_path}")
        logger.info(f"Cancer gene classification: {grouped_data['summary']['breast_cancer']['count']} breast cancer, "
                   f"{grouped_data['summary']['cancer']['count']} other cancer, "
                   f"{grouped_data['summary']['non_cancer']['count']} non-cancer")
    except Exception as e:
        relative_path = global_path.relative_to(project_root)
        logger.error(f"Error saving global list to {relative_path}: {e}")
    
    if failed_writes:
        logger.error(f"Failed to write {len(failed_writes)} gene files: {', '.join(failed_writes[:5])}")
    
    return grouped_data["summary"]


def main():
    """Main execution function: Orchestrates loading, API queries, merging, saving, and verification."""
    # Load configuration and set up paths
    config = load_config()
    PROJECT_ROOT = Path(config['paths']['project_root'])
    INPUT_PREPROCESSED = Path(config['paths']['preprocessed'])  # Input from 00_b
    OUTPUT_DIR = get_auto_output_path(__file__, PROJECT_ROOT)  # Auto-generated output
    
    # Setup logging with auto-generated output directory
    logger = setup_logging(config, OUTPUT_DIR)
    
    logger.info(f"00_b input directory: {get_relative_path(INPUT_PREPROCESSED)}")
    logger.info(f"00_c output directory: {get_relative_path(OUTPUT_DIR)}")
    logger.info("Starting gene information query script...")
    logger.info("Sorting all outputs for consistent comparison between runs.")
    
    ensure_dir(OUTPUT_DIR)
    
    # Verify input files exist
    required_files = [
        INPUT_PREPROCESSED / 'metadata' / 'gene_metadata.json'
    ]
    
    for file_path in required_files:
        if not file_path.exists():
            logger.error(f"Required input file not found: {get_relative_path(file_path)}")
            logger.error("Please run 00_b_data_preprocess.py first")
            return
    
    # Load and prepare gene list
    gene_info = load_gene_list(INPUT_PREPROCESSED)
    if not gene_info:
        logger.error("No genes loaded. Exiting.")
        return
    
    logger.info(f"Loaded {len(gene_info)} genes for querying.")
    
    # Create a map for original symbols for later comparison
    original_symbols_map = {g['key']: g['symbol'] for g in gene_info}
    
    # Query APIs
    mygene_results, mygene_not_found, mygene_stats = query_mygene_info(config, OUTPUT_DIR, [g["id"] for g in gene_info])
    ensembl_results, ensembl_failed, ensembl_stats = query_ensembl_lookup(config, OUTPUT_DIR, gene_info)
    
    # Merge and re-key results
    results = merge_results(mygene_results, ensembl_results, gene_info)
    
    not_found_genes = list(set(mygene_not_found + ensembl_failed))
    
    # Save results conditionally and get summary
    grouped_summary = save_results(OUTPUT_DIR, results, gene_info, original_symbols_map)
    
    # Verify and save final report
    verify_results(OUTPUT_DIR, results, not_found_genes, mygene_stats, ensembl_stats, grouped_summary)
    
    # === NEW: Export curated gene dictionary ===
    if USE_CURATED_LISTS:
        logger.info("\n" + "="*60)
        logger.info("Exporting curated cancer gene dictionary...")
        logger.info("="*60)
        
        try:
            dict_export_path = OUTPUT_DIR / 'cancer_gene_classification.json'
            export_gene_lists_json(str(dict_export_path))
            
            relative_path = get_relative_path(dict_export_path)
            logger.info(f"✓ Exported to: {relative_path}")
            
            # Update existing verification report
            verification_path = OUTPUT_DIR / '00_c_result_info.json'
            if verification_path.exists():
                with open(verification_path, 'r') as f:
                    verification_data = json.load(f)
                
                verification_data['curated_gene_lists'] = {
                    'enabled': True,
                    'export_path': str(relative_path),
                    'total_genes': len(get_cancer_gene_dict()),
                    'breast_cancer_genes': len([g for g, c in get_cancer_gene_dict().items() if c == 'breast_cancer']),
                    'general_cancer_genes': len([g for g, c in get_cancer_gene_dict().items() if c == 'cancer']),
                }
                
                with open(verification_path, 'w') as f:
                    json.dump(verification_data, f, indent=2, sort_keys=True)
                
                logger.info(f"✓ Updated verification report")
            
        except Exception as e:
            logger.error(f"Failed to export gene dictionary: {e}")
    # === END NEW ===
    
    logger.info("Gene information query complete.")
    logger.info("All outputs have been sorted for easy comparison between runs.")


if __name__ == "__main__":
    main()