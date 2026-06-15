# File: code/utils/genes.py (Fixed path resolution)

"""
Gene Information and Processing Utilities

This module provides functions for gene ID normalization, information retrieval,
and cancer relevance classification for co-expression network analysis.

Functions:
    normalize_gene_id: Normalize gene ID by removing version numbers and artifacts
    get_gene_info: Retrieve detailed information for a specific gene from combined results
    extract_gene_symbol: Extract gene symbol from various ID formats
    get_batch_gene_info: Batch fetch gene info for multiple genes with optional limit
    load_combined_gene_info: Load the combined gene info JSON file
    enhance_cancer_detection: Enhanced cancer gene detection with broader keywords and patterns
"""

import json
from pathlib import Path


def normalize_gene_id(gene_id: str) -> str:
    """
    Normalize gene ID by removing version numbers and other artifacts.
    
    Standardizes gene identifiers to a consistent format for reliable lookup
    across different data sources and analysis stages.
    
    Examples:
        ENSG00000122122.10|SASH3 -> ENSG00000122122|SASH3
        ENSG00000122122.10 -> ENSG00000122122
        ENSG00000122122.10_SASH3 -> ENSG00000122122|SASH3
        
    Parameters:
    -----------
    gene_id : str
        Raw gene identifier in various formats
        
    Returns:
    --------
    str
        Normalized gene identifier in consistent format
    """
    if not gene_id:
        return gene_id
    
    # Handle underscore format (common in some data sources)
    if '_' in gene_id and '|' not in gene_id:
        parts = gene_id.split('_')
        if parts[0].startswith('ENSG') and '.' in parts[0]:
            parts[0] = parts[0].split('.')[0]
        return '|'.join(parts)
    
    # Handle pipe format (standard format in this project)
    parts = gene_id.split('|')
    
    # Remove version number from Ensembl ID (the part before pipe)
    if parts[0].startswith('ENSG') and '.' in parts[0]:
        parts[0] = parts[0].split('.')[0]
    
    # Rejoin with pipe if there was a gene symbol
    return '|'.join(parts)


def get_gene_info(gene_key: str, config: dict, combined_data: dict = None):
    """
    Retrieves detailed information for a specific gene using the combined results file.
    Automatically normalizes gene IDs by removing version numbers.

    Args:
        gene_key (str): The identifier for the gene to retrieve, in 'gene_id|gene_name' format.
        config (dict): The loaded project configuration dictionary.
        combined_data (dict, optional): A pre-loaded 'gene_info_combined.json' dictionary
                                        to avoid re-reading the file on multiple calls.

    Returns:
        dict: A dictionary containing the search term, the division (category) the gene
              was found in, and the detailed gene information.
    """
    # Normalize the gene key for consistent lookup
    normalized_key = normalize_gene_id(gene_key)
    
    project_root = Path(config['paths']['project_root'])
    genes_info_dir = Path(config['paths']['genes_info'])
    combined_json_path = genes_info_dir / 'gene_info_combined.json'

    # Load the combined data if not provided (caching for performance)
    if combined_data is None:
        try:
            with open(combined_json_path, 'r') as f:
                combined_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Combined gene info file not found at {combined_json_path}")
            return {'search_term': gene_key, 'division': None, 'gene_info': None}
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {combined_json_path}")
            return {'search_term': gene_key, 'division': None, 'gene_info': None}

    # Search for the normalized gene_key in the three divisions (breast_cancer, cancer, non_cancer)
    found_division = None
    relative_path_str = None
    for division in ['breast_cancer', 'cancer', 'non_cancer']:
        if normalized_key in combined_data.get(division, {}):
            found_division = division
            relative_path_str = combined_data[division][normalized_key]
            break

    if not found_division:
        # Try with original key as fallback (for edge cases)
        for division in ['breast_cancer', 'cancer', 'non_cancer']:
            if gene_key in combined_data.get(division, {}):
                found_division = division
                relative_path_str = combined_data[division][gene_key]
                break
    
    if not found_division:
        return {'search_term': gene_key, 'division': None, 'gene_info': None}

    # FIXED: Correct path resolution - paths in combined.json are relative to project_root/output/
    if relative_path_str:
        # The paths in gene_info_combined.json are like "00_c_gene_info/gene/ENSG00000002587|HS3ST1.json"
        # These need to be resolved relative to project_root/output/
        individual_gene_path = project_root / 'output' / relative_path_str
        
        # Verify the path exists
        if not individual_gene_path.exists():
            print(f"Warning: Gene file not found at primary path: {individual_gene_path}")
            # Try alternative: remove any duplicate prefix and use genes_info_dir as base
            alt_path = genes_info_dir / Path(relative_path_str).relative_to('00_c_gene_info')
            if alt_path.exists():
                individual_gene_path = alt_path
                print(f"Using alternative path: {individual_gene_path}")
            else:
                return {'search_term': gene_key, 'division': found_division, 'gene_info': {'error': 'file not found'}}
    
    try:
        with open(individual_gene_path, 'r') as f:
            gene_info_details = json.load(f)
    except FileNotFoundError:
        print(f"Error: Found gene '{normalized_key}' in '{found_division}' but its file is missing at {individual_gene_path}")
        return {'search_term': gene_key, 'division': found_division, 'gene_info': {'error': 'file not found'}}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from individual gene file: {individual_gene_path}")
        return {'search_term': gene_key, 'division': found_division, 'gene_info': {'error': 'invalid json'}}

    return {
        'search_term': gene_key,
        'normalized_key': normalized_key,
        'division': found_division,
        'gene_info': gene_info_details
    }


def extract_gene_symbol(gene_id: str) -> str:
    """
    Extract gene symbol from various ID formats (agnostic).
    
    Handles multiple gene identifier formats commonly found in genomic data:
    - ENSG00000122122.10_SASH3 → SASH3
    - ENSG00000141510|TP53 → TP53
    - ENSG00000012048.10 → ENSG00000012048
    - ENSG00000122122.10|SASH3 → SASH3
    
    Args:
        gene_id (str): Gene identifier in various formats
        
    Returns:
        str: Extracted gene symbol or normalized ID if no symbol found
    """
    if not gene_id:
        return ""
    
    # Normalize first, then extract
    normalized = normalize_gene_id(gene_id)
    
    if '|' in normalized:
        return normalized.split('|')[-1]
    elif '_' in normalized:
        return normalized.split('_')[-1]
    else:
        return normalized


def get_batch_gene_info(gene_keys: list, config: dict, combined_data: dict = None, fetch_limit: int = None) -> dict:
    """
    Batch fetch gene info for multiple genes with optional limit.
    Automatically normalizes all gene IDs before lookup.
    
    This function efficiently retrieves gene information for multiple genes at once,
    with optional caching and a limit to prevent memory overload. Used for bulk
    processing in network analysis and visualization.
    
    Args:
        gene_keys (list): List of gene identifiers to fetch
        config (dict): Project configuration dictionary
        combined_data (dict, optional): Pre-loaded combined gene info data
        fetch_limit (int, optional): Maximum number of genes to process
        
    Returns:
        dict: Dictionary mapping original gene_key to {'name': str, 'division': str}
    """
    if fetch_limit and len(gene_keys) > fetch_limit:
        gene_keys = gene_keys[:fetch_limit]
    
    batch_info = {}
    
    if combined_data:
        # Union all division dicts for efficient lookup
        all_genes = {}
        for div in ['breast_cancer', 'cancer', 'non_cancer']:
            if div in combined_data:
                all_genes.update(combined_data[div])
        
        for key in gene_keys:
            # Normalize the key for lookup
            normalized_key = normalize_gene_id(key)
            
            if normalized_key in all_genes:
                # Use get_gene_info for full info (it also normalizes internally)
                info = get_gene_info(key, config, combined_data=combined_data)
                if info and 'gene_info' in info and info['gene_info'] and 'error' not in info['gene_info']:
                    batch_info[key] = {
                        'name': info['gene_info'].get('gene_name', extract_gene_symbol(key)),
                        'division': info['division']
                    }
                else:
                    # Fallback: use normalized key directly
                    batch_info[key] = {'name': extract_gene_symbol(key), 'division': 'non_cancer'}
            else:
                # Try original key as fallback
                if key in all_genes:
                    info = get_gene_info(key, config, combined_data=combined_data)
                    if info and 'gene_info' in info and info['gene_info'] and 'error' not in info['gene_info']:
                        batch_info[key] = {
                            'name': info['gene_info'].get('gene_name', extract_gene_symbol(key)),
                            'division': info['division']
                        }
                    else:
                        batch_info[key] = {'name': extract_gene_symbol(key), 'division': 'non_cancer'}
                else:
                    batch_info[key] = {'name': extract_gene_symbol(key), 'division': 'non_cancer'}
    else:
        # Fallback to simple symbol extraction when combined data unavailable
        for key in gene_keys:
            batch_info[key] = {'name': extract_gene_symbol(key), 'division': 'non_cancer'}
    
    return batch_info


def load_combined_gene_info(config: dict) -> dict:
    """
    Load the combined gene info JSON file.
    
    Loads the pre-processed gene information file that contains mappings
    from gene IDs to their detailed information files and cancer classifications.
    
    Args:
        config (dict): Project configuration dictionary
        
    Returns:
        dict: Combined gene info data or None if loading fails
    """
    genes_info_dir = Path(config['paths']['genes_info'])
    combined_json_path = genes_info_dir / 'gene_info_combined.json'
    try:
        with open(combined_json_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load combined gene info: {e}")
        return None


def enhance_cancer_detection(gene_data: dict) -> str:
    """
    Enhanced cancer gene detection with stricter field checking and pattern matching.
    
    Only checks specific fields (gene_description, summary, name) and uses
    more precise cancer term matching with word boundaries. Skips misc_info 
    to avoid disease association over-classification.
    
    Args:
        gene_data (dict): Gene information dictionary
        
    Returns:
        str: Classification - 'breast_cancer', 'cancer', or 'non_cancer'
    """
    if not gene_data:
        return 'non_cancer'
    
    # ONLY check specific fields - NOT misc_info or disease_associations
    text_fields = []
    
    # Primary fields to check (from structured gene info)
    primary_fields = ['gene_description', 'summary', 'name']
    for field in primary_fields:
        if gene_data.get(field):
            text_fields.append(str(gene_data[field]))
    
    full_text = ' '.join(text_fields).lower()
    
    if not full_text:
        return 'non_cancer'
    
    import re
    
    # Breast cancer specific terms - MORE SPECIFIC to breast context
    breast_cancer_patterns = [
        r'\bbreast cancer\b',
        r'\bbrca1\b', r'\bbrca2\b', 
        r'\bher2\b', r'\bher-2\b', r'\berbb2\b', r'\berb-b2\b',
        r'\bestrogen receptor\b', r'\bprogesterone receptor\b',
        r'\btriple-negative\b', r'\btriple negative\b',
        # REMOVED: r'\bductal carcinoma\b', r'\blobular carcinoma\b' (can be pancreatic, etc.)
        r'\bmammary\b',
        r'\bbreast tumor\b', r'\bbreast carcinoma\b', r'\bbreast neoplasm\b',
        r'\bbreast malignancy\b', r'\bbreast adenocarcinoma\b',
        # REMOVED: r'\binvasive ductal carcinoma\b', r'\binvasive lobular carcinoma\b'
        r'\bhormone receptor\b', r'\bhr\+\b', r'\bher2\+\b', r'\btnbc\b',
        r'\bbrca\b', r'\bpalb2\b', r'\bchek2\b', r'\batm\b'
    ]
    
    # Check breast cancer patterns
    for pattern in breast_cancer_patterns:
        if re.search(pattern, full_text):
            return 'breast_cancer'
    
    # General cancer terms (with word boundary checking)
    cancer_patterns = [
        # High specificity terms
        r'\boncogene\b', r'\btumor suppressor\b', r'\btumour suppressor\b',
        r'\bmetastasis\b', r'\bmetastatic\b', r'\bmalignant\b',
        r'\bcarcinoma\b', r'\bsarcoma\b', r'\bmelanoma\b',
        r'\bglioblastoma\b', r'\bleukemia\b', r'\bleukaemia\b', r'\blymphoma\b',
        r'\bcarcinogenesis\b', r'\btumorigenesis\b', r'\btumorigenic\b',
        
        # Gene-specific cancer markers (as whole words only)
        r'\bp53\b', r'\btp53\b', r'\bras\b', r'\bkras\b', r'\bpten\b', 
        r'\bpi3k\b', r'\bakt\b',
        
        # Cancer context terms
        r'\bcancer\b', r'\btumor\b', r'\btumour\b', r'\bmalignancy\b', r'\bneoplasm\b'
    ]
    
    # Check cancer patterns
    for pattern in cancer_patterns:
        if re.search(pattern, full_text):
            return 'cancer'
    
    return 'non_cancer'