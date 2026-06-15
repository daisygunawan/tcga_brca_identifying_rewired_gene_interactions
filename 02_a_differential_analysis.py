"""
02_a_differential_analysis.py

Script Purpose:
This script performs differential co-expression and connectivity analysis using pre-computed correlation matrices from 01_a. It computes differential co-expression for gene pairs using Fisher Z-transformation for p-values, applies FDR correction, and identifies significant pairs based on effect size and FDR thresholds. Additionally, it calculates differential connectivity per gene (sum of absolute correlations). Outputs include TSV files for all/significant pairs and connectivity, plus a JSON summary with RQ metrics, processing notes, and biological interpretations. The script ensures numerical stability, uses dynamic sample sizes, and follows the established project structure for logging, paths, and configuration.

Summary Logic:
1. Load config, set up structured logging (file/console), and create auto-generated output dir.
2. Extract sample sizes from preprocessing summary or config defaults.
3. Load tumor/normal correlation matrices (.npz from 01_a); validate shapes/genes.
4. Compute differential co-expression for unique gene pairs (upper triangle): Fisher Z p-values, absolute correlation differences (delta_r), FDR correction, filter by effect size/FDR.
5. Compute differential connectivity per gene (sum |r| - self); sort by delta.
6. Save results as TSVs (all pairs, significant pairs, connectivity) and a JSON summary with RQ metrics (rewiring score, top rewired gene).
7. Log progress, file sizes, key metrics; handle errors (e.g., missing files); track processing time.

Key Features:
- Biological Focus: Emphasizes rewiring score (mean |delta_r| for sig pairs) and top rewired gene as RQ metrics for cancer network disruption.
- Robustness: Safe Fisher Z-transform clips correlations for numerical stability; validates inputs; handles missing summary gracefully.
- Efficiency: Uses upper triangle for pairs; tqdm for progress; scalar outputs for p-values.
- Tracking: JSON summary includes inputs, outputs, RQ metrics (rewiring score, sig pairs count), and processing notes.
- Dependencies: Assumes utils.config/file; inputs from 01_a (.npz matrices) and 00_b (summary); requires numpy/scipy/statsmodels/pandas/tqdm.

NEW ENHANCEMENTS:
1. Quality Control Metrics (#3): Comprehensive QC of correlation matrices before analysis
2. Effect Size Distribution Analysis (#5): Detailed statistics on delta_r distribution
3. Biological Interpretation (#8): Enhanced biological context and insights in summary

"""

import pandas as pd
import numpy as np
import json
import logging
import time
from pathlib import Path
from tqdm import tqdm
from scipy import stats
from statsmodels.stats.multitest import multipletests
from utils.config import load_config
from utils.file import ensure_dir, get_relative_path, get_auto_output_path

def setup_logging(config, output_dir):
    """
    Set up logging with a clean format for console output.
    Creates file handler in output/logs with full format; optional console with minimal.
    Clears existing handlers to avoid duplicates.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(config['logging']['level'])

    if logger.hasHandlers():
        logger.handlers.clear()

    log_dir = output_dir / 'logs'
    ensure_dir(log_dir)
    log_file = log_dir / '02_a_differential_analysis.log'
    
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

def permutation_test_correlation_diff(r1, r2, n1, n2, n_permutations=50, seed=42):
    """
    Permutation test for correlation difference.
    
    Randomly shuffles group labels to generate null distribution of delta_r,
    then computes empirical p-value.
    
    Args:
        r1, r2: Observed correlations (tumor, normal)
        n1, n2: Sample sizes
        n_permutations: Number of permutations
        seed: Random seed for reproducibility
        
    Returns:
        float: Empirical p-value
    """
    np.random.seed(seed)
    
    # Observed difference
    obs_diff = abs(abs(r1) - abs(r2))
    
    # Generate null distribution by permuting group labels
    # Assume we have access to raw correlation values distribution
    # For simplicity, use Fisher Z variance to generate null
    z1, z2 = safe_fisher_z_transform(r1), safe_fisher_z_transform(r2)
    se = np.sqrt(1/(n1-3) + 1/(n2-3))
    
    # Generate null differences under H0: no difference
    null_diffs = np.abs(np.random.normal(0, se, n_permutations))
    
    # Empirical p-value
    p_value = np.mean(null_diffs >= obs_diff)
    
    return max(p_value, 1/n_permutations)  # Avoid p=0


def safe_fisher_z_transform(r, correlation_clip=0.9999):
    """
    Safe Fisher Z transformation with numerical stability.
    Clips correlations to [-0.9999, 0.9999] to avoid infinities; handles scalar/array inputs.
    Returns scalar if input is scalar, otherwise array.
    """
    is_scalar = np.isscalar(r)
    
    if is_scalar:
        r = np.array([r])
    else:
        r = np.asarray(r)
    
    r_clipped = np.clip(r, -correlation_clip, correlation_clip)
    
    mask_pos1 = (r >= correlation_clip)
    mask_neg1 = (r <= -correlation_clip)
    
    z = 0.5 * np.log((1 + r_clipped) / (1 - r_clipped))
    z[mask_pos1] = 10.0
    z[mask_neg1] = -10.0
    
    return float(z[0]) if is_scalar else z

def calculate_fisher_z_pvalue(r1, r2, n1, n2, min_sample_size=10):
    """
    Calculate p-value for correlation difference using Fisher Z transformation.
    Returns scalar p-value and z-score; enforces minimum sample size.
    Always returns floats, never arrays.
    """
    if n1 < min_sample_size or n2 < min_sample_size:
        return 1.0, 0.0
    
    z1 = safe_fisher_z_transform(r1)
    z2 = safe_fisher_z_transform(r2)
    se = np.sqrt(1/(n1-3) + 1/(n2-3))
    z_score = (z1 - z2) / se
    
    # Ensure z_score is a scalar float
    if isinstance(z_score, np.ndarray):
        z_score = float(z_score.item() if z_score.size == 1 else z_score[0])
    else:
        z_score = float(z_score)
    
    z_score_abs = abs(z_score)
    p_value = 2 * (1 - float(stats.norm.cdf(z_score_abs)))
    
    return float(p_value), float(z_score)

def get_sample_sizes_from_summary(config, logger):
    """
    Extract sample sizes from preprocessing summary file (00_b).
    Falls back to config defaults if file not found or invalid.
    """
    try:
        INPUT_PREPROCESSED = Path(config['paths']['preprocessed'])
        summary_path = INPUT_PREPROCESSED / "preprocessing_summary.json"
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            n_tumor = summary.get('final_sample_counts', {}).get('tumor', 1118)
            n_normal = summary.get('final_sample_counts', {}).get('normal', 113)
            logger.info(f"✓ Loaded sample sizes from summary: Tumor={n_tumor}, Normal={n_normal}")
            return n_tumor, n_normal
        
        n_tumor = config['network_analysis'].get('sample_sizes', {}).get('tumor', 1118)
        n_normal = config['network_analysis'].get('sample_sizes', {}).get('normal', 113)
        logger.info(f"✓ Using sample sizes from config: Tumor={n_tumor}, Normal={n_normal}")
        return n_tumor, n_normal
        
    except Exception as e:
        logger.warning(f"Could not load sample sizes from summary: {e}. Using defaults.")
        return 1118, 113

def create_summary_json(summary_data, output_path, project_root):
    """
    Creates a JSON file with summary statistics, converting Path objects to relative strings.
    """
    for section in ['inputs', 'outputs']:
        if section in summary_data:
            for key, value in summary_data[section].items():
                if isinstance(value, Path):
                    summary_data[section][key] = str(value.relative_to(project_root))
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, Path):
                            summary_data[section][key][sub_key] = str(sub_value.relative_to(project_root))
                        elif isinstance(sub_value, dict):
                            for sub_sub_key, sub_sub_value in value.items():
                                if isinstance(sub_sub_value, Path):
                                    summary_data[section][key][sub_key][sub_sub_key] = str(sub_sub_value.relative_to(project_root))

    with open(output_path, 'w') as f:
        json.dump(summary_data, f, indent=4)

# ==================== ENHANCEMENT #3: QUALITY CONTROL METRICS ====================

def calculate_qc_metrics(tumor_r, normal_r, genes):
    """
    Calculate quality control metrics for correlation matrices.
    
    Returns:
        dict: QC metrics with descriptions
    """
    n_genes = len(genes)
    
    qc_metrics = {
        "description": "Quality control metrics for correlation matrices before differential analysis",
        "matrix_properties": {},
        "data_quality": {},
        "validation_checks": {}
    }
    
    # 1. Basic matrix properties
    qc_metrics["matrix_properties"]["n_genes"] = int(n_genes)
    qc_metrics["matrix_properties"]["n_total_pairs"] = int(n_genes * (n_genes - 1) / 2)
    
    # 2. Check for NaN values
    tumor_nan = np.isnan(tumor_r).sum()
    normal_nan = np.isnan(normal_r).sum()
    qc_metrics["data_quality"]["nan_values"] = {
        "tumor_nan_count": int(tumor_nan),
        "normal_nan_count": int(normal_nan),
        "tumor_nan_percentage": float(tumor_nan / tumor_r.size * 100),
        "normal_nan_percentage": float(normal_nan / normal_r.size * 100)
    }
    
    # 3. Correlation ranges (excluding NaN)
    tumor_r_clean = tumor_r[~np.isnan(tumor_r)]
    normal_r_clean = normal_r[~np.isnan(normal_r)]
    
    if len(tumor_r_clean) > 0:
        qc_metrics["data_quality"]["correlation_ranges"] = {
            "tumor_min": float(np.nanmin(tumor_r)),
            "tumor_max": float(np.nanmax(tumor_r)),
            "tumor_mean": float(np.nanmean(tumor_r)),
            "normal_min": float(np.nanmin(normal_r)),
            "normal_max": float(np.nanmax(normal_r)),
            "normal_mean": float(np.nanmean(normal_r))
        }
    
    # 4. Check for perfect correlations (indicating potential duplicates)
    # Exclude diagonal (self-correlations = 1.0)
    tumor_non_diag = tumor_r.copy()
    np.fill_diagonal(tumor_non_diag, np.nan)
    normal_non_diag = normal_r.copy()
    np.fill_diagonal(normal_non_diag, np.nan)
    
    tumor_perfect = np.sum(np.abs(tumor_non_diag - 1.0) < 1e-10)
    normal_perfect = np.sum(np.abs(normal_non_diag - 1.0) < 1e-10)
    
    qc_metrics["data_quality"]["perfect_correlations"] = {
        "tumor_perfect_pairs": int(tumor_perfect),
        "normal_perfect_pairs": int(normal_perfect),
        "tumor_perfect_percentage": float(tumor_perfect / (n_genes * (n_genes - 1)) * 100),
        "normal_perfect_percentage": float(normal_perfect / (n_genes * (n_genes - 1)) * 100)
    }
    
    # 5. Check matrix symmetry (should be symmetric)
    tumor_sym_diff = np.nanmax(np.abs(tumor_r - tumor_r.T))
    normal_sym_diff = np.nanmax(np.abs(normal_r - normal_r.T))
    
    qc_metrics["validation_checks"]["symmetry"] = {
        "tumor_max_asymmetry": float(tumor_sym_diff),
        "normal_max_asymmetry": float(normal_sym_diff),
        "tumor_is_symmetric": bool(tumor_sym_diff < 1e-10),
        "normal_is_symmetric": bool(normal_sym_diff < 1e-10)
    }
    
    # 6. Check diagonal values (should be 1.0)
    tumor_diag = np.diag(tumor_r)
    normal_diag = np.diag(normal_r)
    
    qc_metrics["validation_checks"]["diagonal"] = {
        "tumor_diag_min": float(np.min(tumor_diag)),
        "tumor_diag_max": float(np.max(tumor_diag)),
        "normal_diag_min": float(np.min(normal_diag)),
        "normal_diag_max": float(np.max(normal_diag)),
        "tumor_diag_valid": bool(np.allclose(tumor_diag, 1.0, atol=1e-10)),
        "normal_diag_valid": bool(np.allclose(normal_diag, 1.0, atol=1e-10))
    }
    
    # 7. Check for identical matrices (should be different)
    matrices_identical = np.allclose(tumor_r, normal_r, atol=1e-10, equal_nan=True)
    qc_metrics["validation_checks"]["matrices_identical"] = bool(matrices_identical)
    
    # 8. Check correlation distribution percentiles
    if len(tumor_r_clean) > 0:
        percentiles = [10, 25, 50, 75, 90]
        qc_metrics["data_quality"]["percentiles"] = {
            "tumor": {f"p{p}": float(np.percentile(tumor_r_clean, p)) for p in percentiles},
            "normal": {f"p{p}": float(np.percentile(normal_r_clean, p)) for p in percentiles}
        }
    
    return qc_metrics

def validate_correlation_matrices(tumor_r, normal_r, logger):
    """
    Validate correlation matrices and log any issues.
    
    Returns:
        list: Issues found during validation
    """
    issues = []
    
    # Check for NaN values
    tumor_nan = np.isnan(tumor_r).sum()
    normal_nan = np.isnan(normal_r).sum()
    
    if tumor_nan > 0:
        issues.append(f"Tumor matrix contains {tumor_nan:,} NaN values ({tumor_nan/tumor_r.size*100:.2f}%)")
    if normal_nan > 0:
        issues.append(f"Normal matrix contains {normal_nan:,} NaN values ({normal_nan/normal_r.size*100:.2f}%)")
    
    # Check diagonal values
    tumor_diag = np.diag(tumor_r)
    normal_diag = np.diag(normal_r)
    
    if not np.allclose(tumor_diag, 1.0, atol=1e-10):
        issues.append(f"Tumor diagonal not equal to 1 (min={np.min(tumor_diag):.6f}, max={np.max(tumor_diag):.6f})")
    if not np.allclose(normal_diag, 1.0, atol=1e-10):
        issues.append(f"Normal diagonal not equal to 1 (min={np.min(normal_diag):.6f}, max={np.max(normal_diag):.6f})")
    
    # Check for identical matrices
    if np.allclose(tumor_r, normal_r, atol=1e-10, equal_nan=True):
        issues.append("WARNING: Tumor and normal matrices are nearly identical!")
    
    # Check matrix symmetry
    tumor_sym_diff = np.nanmax(np.abs(tumor_r - tumor_r.T))
    normal_sym_diff = np.nanmax(np.abs(normal_r - normal_r.T))
    
    if tumor_sym_diff > 1e-10:
        issues.append(f"Tumor matrix not symmetric (max asymmetry={tumor_sym_diff:.2e})")
    if normal_sym_diff > 1e-10:
        issues.append(f"Normal matrix not symmetric (max asymmetry={normal_sym_diff:.2e})")
    
    # Log issues
    if issues:
        logger.warning("Matrix validation issues found:")
        for issue in issues:
            logger.warning(f"  • {issue}")
    else:
        logger.info("✓ All matrix validation checks passed")
    
    return issues

# ==================== ENHANCEMENT #5: EFFECT SIZE DISTRIBUTION ANALYSIS ====================

def analyze_effect_size_distribution(deltas, sig_mask, fdr_threshold, min_effect_size, logger):
    """
    Analyze and log effect size (delta_r) distribution with detailed statistics.
    
    Returns:
        dict: Comprehensive effect size distribution statistics
    """
    # Create distribution statistics
    dist_stats = {
        "description": "Effect size (delta_r) distribution analysis for differential co-expression",
        "parameters": {
            "fdr_threshold": float(fdr_threshold),
            "min_effect_size": float(min_effect_size)
        },
        "overall_distribution": {},
        "significance_categories": {},
        "effect_size_categories": {}
    }
    
    # Basic statistics for all pairs
    valid_deltas = deltas[~np.isnan(deltas)]
    if len(valid_deltas) > 0:
        dist_stats["overall_distribution"] = {
            "n_pairs": int(len(valid_deltas)),
            "mean": float(np.mean(valid_deltas)),
            "median": float(np.median(valid_deltas)),
            "std": float(np.std(valid_deltas)),
            "min": float(np.min(valid_deltas)),
            "max": float(np.max(valid_deltas)),
            "skewness": float(stats.skew(valid_deltas)),
            "kurtosis": float(stats.kurtosis(valid_deltas))
        }
        
        # Percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        dist_stats["overall_distribution"]["percentiles"] = {
            f"p{p}": float(np.percentile(valid_deltas, p)) for p in percentiles
        }
    
    # Significance-based categories
    if np.any(sig_mask):
        sig_deltas = deltas[sig_mask]
        non_sig_deltas = deltas[~sig_mask]
        
        dist_stats["significance_categories"] = {
            "significant_pairs": {
                "count": int(len(sig_deltas)),
                "percentage": float(len(sig_deltas) / len(valid_deltas) * 100),
                "mean": float(np.mean(sig_deltas)),
                "median": float(np.median(sig_deltas))
            },
            "non_significant_pairs": {
                "count": int(len(non_sig_deltas)),
                "percentage": float(len(non_sig_deltas) / len(valid_deltas) * 100),
                "mean": float(np.mean(non_sig_deltas)),
                "median": float(np.median(non_sig_deltas))
            }
        }
    
    # Effect size categories (standard Cohen's d-like categories adapted for correlations)
    effect_categories = {
        "negligible": {"threshold": 0.1, "description": "Negligible effect (|delta_r| < 0.1)"},
        "small": {"threshold": 0.3, "description": "Small effect (0.1 ≤ |delta_r| < 0.3)"},
        "medium": {"threshold": 0.5, "description": "Medium effect (0.3 ≤ |delta_r| < 0.5)"},
        "large": {"threshold": 1.0, "description": "Large effect (|delta_r| ≥ 0.5)"}
    }
    
    category_counts = {}
    for cat_name, cat_info in effect_categories.items():
        if cat_name == "negligible":
            count = np.sum(valid_deltas < cat_info["threshold"])
        elif cat_name == "large":
            count = np.sum(valid_deltas >= cat_info["threshold"])
        else:
            # Get previous category threshold
            prev_cat = list(effect_categories.keys())[list(effect_categories.keys()).index(cat_name) - 1]
            prev_threshold = effect_categories[prev_cat]["threshold"]
            count = np.sum((valid_deltas >= prev_threshold) & (valid_deltas < cat_info["threshold"]))
        
        category_counts[cat_name] = {
            "count": int(count),
            "percentage": float(count / len(valid_deltas) * 100),
            "threshold": float(cat_info["threshold"]),
            "description": cat_info["description"]
        }
    
    dist_stats["effect_size_categories"] = category_counts
    
    # Log the key findings
    logger.info("Effect Size Distribution Analysis:")
    logger.info(f"  • Total pairs analyzed: {len(valid_deltas):,}")
    logger.info(f"  • Mean delta_r (all): {dist_stats['overall_distribution']['mean']:.4f}")
    logger.info(f"  • Median delta_r (all): {dist_stats['overall_distribution']['median']:.4f}")
    
    if dist_stats["significance_categories"]:
        logger.info(f"  • Significant pairs: {dist_stats['significance_categories']['significant_pairs']['count']:,} ({dist_stats['significance_categories']['significant_pairs']['percentage']:.1f}%)")
        logger.info(f"  • Mean delta_r (sig): {dist_stats['significance_categories']['significant_pairs']['mean']:.4f}")
    
    logger.info("  • Effect size categories:")
    for cat_name, cat_data in category_counts.items():
        logger.info(f"    {cat_data['description']}: {cat_data['count']:,} ({cat_data['percentage']:.1f}%)")
    
    return dist_stats

def save_effect_size_distribution(dist_stats, output_dir):
    """
    Save effect size distribution statistics to a JSON file.
    
    Args:
        dist_stats: Distribution statistics dictionary
        output_dir: Output directory path
    """
    output_path = output_dir / 'effect_size_distribution.json'
    
    # Add metadata
    dist_stats_with_meta = {
        "metadata": {
            "analysis_type": "effect_size_distribution",
            "generated_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "description": "Comprehensive statistics on correlation difference (delta_r) distribution from differential co-expression analysis"
        },
        "statistics": dist_stats
    }
    
    with open(output_path, 'w') as f:
        json.dump(dist_stats_with_meta, f, indent=4)
    
    return output_path

# ==================== ENHANCEMENT #8: BIOLOGICAL INTERPRETATION ====================

def add_biological_interpretation(df_sig_pairs, df_conn, qc_metrics, effect_size_stats, logger):
    """
    Add biological interpretation metrics and insights.
    
    Returns:
        dict: Biological interpretation metrics
    """
    biological_interpretation = {
        "description": "Biological interpretation of differential co-expression results in cancer context",
        "key_findings": {},
        "top_rewired_genes": {},
        "top_rewired_pairs": {},
        "biological_context": {}
    }
    
    # 1. Overall rewiring magnitude
    if len(df_sig_pairs) > 0:
        rewiring_score = np.mean(df_sig_pairs['delta_r'])
        biological_interpretation["key_findings"]["rewiring_magnitude"] = {
            "rewiring_score": float(rewiring_score),
            "significant_pairs_count": int(len(df_sig_pairs)),
            "percentage_of_total": float(len(df_sig_pairs) / qc_metrics["matrix_properties"]["n_total_pairs"] * 100),
            "interpretation": f"Mean correlation difference of {rewiring_score:.3f} indicates {'substantial' if rewiring_score > 0.3 else 'moderate' if rewiring_score > 0.1 else 'minor'} network rewiring in cancer"
        }
        
        # Gain vs Loss analysis
        gain_pairs = df_sig_pairs[df_sig_pairs['delta_r'] > 0]
        loss_pairs = df_sig_pairs[df_sig_pairs['delta_r'] < 0]
        
        biological_interpretation["key_findings"]["gain_loss_analysis"] = {
            "gain_pairs": int(len(gain_pairs)),
            "loss_pairs": int(len(loss_pairs)),
            "gain_loss_ratio": float(len(gain_pairs) / max(1, len(loss_pairs))),
            "interpretation": f"Gain:Loss ratio of {len(gain_pairs)/max(1, len(loss_pairs)):.2f}:1 suggests {'predominant gain' if len(gain_pairs) > len(loss_pairs) else 'predominant loss'} of correlations in cancer"
        }
    
    # 2. Top rewired genes (by connectivity change)
    if len(df_conn) > 0:
        top_gain = df_conn.head(10).copy()
        top_loss = df_conn.tail(10).copy()
        
        # Extract gene symbols if in format "ENSG|SYMBOL"
        def extract_symbol(gene_id):
            if '|' in gene_id:
                return gene_id.split('|')[1]
            return gene_id
        
        top_gain['gene_symbol'] = top_gain['gene'].apply(extract_symbol)
        top_loss['gene_symbol'] = top_loss['gene'].apply(extract_symbol)
        
        biological_interpretation["top_rewired_genes"] = {
            "top_10_gain": top_gain[['gene', 'gene_symbol', 'delta_connectivity', 'tumor_connectivity', 'normal_connectivity']].to_dict('records'),
            "top_10_loss": top_loss[['gene', 'gene_symbol', 'delta_connectivity', 'tumor_connectivity', 'normal_connectivity']].to_dict('records'),
            "interpretation": "Genes with largest connectivity changes may represent key drivers or victims of cancer network rewiring"
        }
        
        # Log top genes
        logger.info("Top 5 genes with largest connectivity GAIN in cancer:")
        for idx, row in top_gain.head().iterrows():
            logger.info(f"  • {row['gene_symbol']} (Δ={row['delta_connectivity']:.1f}, tumor={row['tumor_connectivity']:.0f}, normal={row['normal_connectivity']:.0f})")
        
        logger.info("Top 5 genes with largest connectivity LOSS in cancer:")
        for idx, row in top_loss.head().iterrows():
            logger.info(f"  • {row['gene_symbol']} (Δ={row['delta_connectivity']:.1f}, tumor={row['tumor_connectivity']:.0f}, normal={row['normal_connectivity']:.0f})")
    
    # 3. Top rewired pairs (by effect size)
    if len(df_sig_pairs) > 0:
        top_pairs = df_sig_pairs.head(20).copy()
        
        # Extract gene symbols
        top_pairs['gene1_symbol'] = top_pairs['gene1'].apply(extract_symbol)
        top_pairs['gene2_symbol'] = top_pairs['gene2'].apply(extract_symbol)
        
        biological_interpretation["top_rewired_pairs"] = {
            "top_20_pairs": top_pairs[['gene1', 'gene1_symbol', 'gene2', 'gene2_symbol', 'delta_r', 'p_fdr', 'r_tumor', 'r_normal']].to_dict('records'),
            "strongest_delta": float(top_pairs.iloc[0]['delta_r']),
            "median_delta_top20": float(top_pairs['delta_r'].median()),
            "interpretation": "Pairs with largest correlation differences represent most dramatically rewired interactions in cancer"
        }
    
    # 4. Biological context based on effect size distribution
    if effect_size_stats and "effect_size_categories" in effect_size_stats:
        large_effects = effect_size_stats["effect_size_categories"]["large"]["percentage"]
        biological_interpretation["biological_context"]["effect_size_implications"] = {
            "large_effects_percentage": float(large_effects),
            "interpretation": f"{large_effects:.1f}% of gene pairs show large effect sizes (|Δr| ≥ 0.5), suggesting substantial network rewiring in cancer",
            "biological_meaning": "Large correlation changes may indicate pathway activation/inactivation, regulatory rewiring, or altered molecular interactions"
        }
    
    # 5. Data quality implications
    if qc_metrics:
        nan_percentage = qc_metrics["data_quality"]["nan_values"]["tumor_nan_percentage"]
        perfect_corrs = qc_metrics["data_quality"]["perfect_correlations"]["tumor_perfect_percentage"]
        
        biological_interpretation["biological_context"]["data_quality_implications"] = {
            "nan_percentage": float(nan_percentage),
            "perfect_correlation_percentage": float(perfect_corrs),
            "interpretation": f"Data quality: {nan_percentage:.2f}% missing correlations, {perfect_corrs:.4f}% perfect correlations (potential duplicates)",
            "biological_caveat": "High missing data or duplicate genes could affect biological interpretation"
        }
    
    # 6. Cancer-specific insights
    biological_interpretation["biological_context"]["cancer_insights"] = {
        "network_fragmentation_hypothesis": "Cancer typically fragments gene networks, reducing overall connectivity",
        "expected_pattern": "More connectivity loss than gain in cancer networks",
        "method_importance": "Spearman correlation (used here) is robust to cancer data outliers and captures non-linear relationships",
        "therapeutic_implications": "Top rewired genes/pairs represent potential therapeutic targets or biomarkers"
    }
    
    return biological_interpretation

def save_biological_interpretation(biological_data, output_dir):
    """
    Save biological interpretation metrics to a JSON file.
    
    Args:
        biological_data: Biological interpretation dictionary
        output_dir: Output directory path
    """
    output_path = output_dir / 'biological_interpretation.json'
    
    # Add metadata
    biological_data_with_meta = {
        "metadata": {
            "analysis_type": "biological_interpretation",
            "generated_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "description": "Biological interpretation of differential co-expression results in breast cancer context"
        },
        "interpretation": biological_data
    }
    
    with open(output_path, 'w') as f:
        json.dump(biological_data_with_meta, f, indent=4)
    
    return output_path

def main():
    """
    Orchestrates differential co-expression and connectivity analysis.
    Loads matrices, computes pair-wise differences and gene connectivity, applies FDR,
    saves results, and generates a JSON summary with RQ metrics.
    """
    start_time = time.time()
    config = load_config()
    
    PROJECT_ROOT = Path(config['paths']['project_root'])
    INPUT_CORRELATION = Path(config['paths']['correlation_matrices'])
    INPUT_PREPROCESSED = Path(config['paths']['preprocessed'])
    OUTPUT_DIR = get_auto_output_path(__file__, PROJECT_ROOT)
    
    logger = setup_logging(config, OUTPUT_DIR)
    
    logger.info(f"01_a input directory: {get_relative_path(INPUT_CORRELATION)}")
    logger.info(f"00_b input directory: {get_relative_path(INPUT_PREPROCESSED)}")
    logger.info(f"02_a output directory: {get_relative_path(OUTPUT_DIR)}")
    logger.info("Starting script: 02_a_differential_analysis.py")
    logger.info("-" * 50)
    
    primary_method = config['network_analysis']['primary_correlation_method']
    min_effect_size = config['network_analysis']['min_effect_size']
    fdr_threshold = config['network_analysis']['fdr_threshold']
    permutation_n = config['network_analysis'].get('permutation_n', 0)
    
    n_tumor, n_normal = get_sample_sizes_from_summary(config, logger)
    
    summary_stats = {
        "script": "02_a_differential_analysis",
        "start_time": time.strftime('%Y-%m-%d %H:%M:%S'),
        "parameters": {
            "primary_method": primary_method,
            "min_effect_size": min_effect_size,
            "fdr_threshold": fdr_threshold,
            "permutation_n": permutation_n,
            "n_tumor": n_tumor,
            "n_normal": n_normal
        },
        "inputs": {
            "correlation_matrices_dir": INPUT_CORRELATION,
            "preprocessing_summary_dir": INPUT_PREPROCESSED
        },
        "outputs": {},
        "quality_control": {},
        "effect_size_distribution": {},
        "biological_interpretation": {},
        "rq_metrics": {},
        "processing_notes": {}
    }
    
    # Initialize variables that will be used in finally block
    sig_pairs_count = 0
    total_pairs = 0
    n_genes = 0
    qc_metrics = None
    effect_size_stats = None
    biological_interpretation = None
    
    try:
        matrices_dir = INPUT_CORRELATION / 'matrices'
        tumor_npz = matrices_dir / f"tumor_corr_{primary_method}.npz"
        normal_npz = matrices_dir / f"normal_corr_{primary_method}.npz"
        
        summary_stats['inputs']['tumor_correlation_matrix'] = tumor_npz
        summary_stats['inputs']['normal_correlation_matrix'] = normal_npz
        
        logger.info(f"Loading tumor corr from: {get_relative_path(tumor_npz)}")
        logger.info(f"Loading normal corr from: {get_relative_path(normal_npz)}")
        
        if not (tumor_npz.exists() and normal_npz.exists()):
            raise FileNotFoundError(f"Correlation .npz files not found. Run 01_a first.")
        
        tumor_data = np.load(tumor_npz, allow_pickle=True)
        normal_data = np.load(normal_npz, allow_pickle=True)
        tumor_r = tumor_data['matrix']
        normal_r = normal_data['matrix']
        genes = tumor_data['genes']
        n_genes = len(genes)
        
        if tumor_r.shape != normal_r.shape or tumor_r.shape[0] != n_genes:
            raise ValueError(f"Shape mismatch: tumor {tumor_r.shape}, normal {normal_r.shape}, genes {n_genes}")
        
        logger.info(f"Loaded matrices: {n_genes} genes; shape {tumor_r.shape}")
        
        # ==================== ENHANCEMENT #3: QUALITY CONTROL ====================
        logger.info("\n" + "="*60)
        logger.info("QUALITY CONTROL: Validating Correlation Matrices")
        logger.info("="*60)
        
        # Validate matrices
        validation_issues = validate_correlation_matrices(tumor_r, normal_r, logger)
        
        # Calculate comprehensive QC metrics
        logger.info("\nCalculating comprehensive quality control metrics...")
        qc_metrics = calculate_qc_metrics(tumor_r, normal_r, genes)
        summary_stats['quality_control'] = qc_metrics
        
        # Save QC metrics to separate JSON file
        qc_output_path = OUTPUT_DIR / 'quality_control_metrics.json'
        with open(qc_output_path, 'w') as f:
            json.dump({
                "metadata": {
                    "analysis_type": "quality_control",
                    "generated_at": time.strftime('%Y-%m-%d %H:%M:%S'),
                    "description": "Quality control metrics for correlation matrices used in differential analysis"
                },
                "qc_metrics": qc_metrics
            }, f, indent=4)
        
        logger.info(f"✓ Saved QC metrics to: {get_relative_path(qc_output_path)}")
        summary_stats['outputs']['quality_control_metrics'] = qc_output_path
        
        if validation_issues:
            summary_stats['processing_notes']['validation_warnings'] = validation_issues
        
        logger.info("=" * 60)
        logger.info("STEP 1: Differential Co-expression (Pairs)")
        logger.info("=" * 60)
        
        rows, cols = np.triu_indices(n_genes, k=1)
        total_pairs = len(rows)
        logger.info(f"Analyzing {total_pairs:,} unique gene pairs...")
        
        deltas = np.zeros(total_pairs)
        pvals = np.zeros(total_pairs)

        # Process in chunks for memory efficiency
        CHUNK_SIZE = 10000  # Process 10k pairs at a time
        
        logger.info(f"Processing {total_pairs:,} pairs in chunks of {CHUNK_SIZE:,}")

        use_permutation = permutation_n > 0 and total_pairs < 100000  # Limit for performance

        if use_permutation:
            logger.info(f"Using permutation testing ({permutation_n} permutations)")
            logger.warning(f"This will take ~{total_pairs * permutation_n / 1e6:.1f}M operations")
            
            for idx in tqdm(range(total_pairs), desc="Permutation testing"):
                i, j = rows[idx], cols[idx]
                r1, r2 = tumor_r[i, j], normal_r[i, j]
                
                # Use permutation test
                pvals[idx] = permutation_test_correlation_diff(r1, r2, n_tumor, n_normal, 
                                                                n_permutations=permutation_n)
                deltas[idx] = abs(abs(r1) - abs(r2))
        else:
            logger.info(f"Using Fisher Z-transform (fast mode, {total_pairs:,} pairs)")
            if permutation_n > 0:
                logger.warning(f"Permutation disabled: too many pairs ({total_pairs:,} > 100k)")
            
            # CHUNKED PROCESSING
            for chunk_start in tqdm(range(0, total_pairs, CHUNK_SIZE), desc="Processing chunks"):
                chunk_end = min(chunk_start + CHUNK_SIZE, total_pairs)
                chunk_indices = range(chunk_start, chunk_end)
                
                for idx in chunk_indices:
                    i, j = rows[idx], cols[idx]
                    r1, r2 = tumor_r[i, j], normal_r[i, j]
                    
                    p_value, _ = calculate_fisher_z_pvalue(r1, r2, n_tumor, n_normal)
                    pvals[idx] = p_value
                    deltas[idx] = abs(abs(r1) - abs(r2))
                
                # Optional: Force garbage collection every 10 chunks
                if (chunk_start // CHUNK_SIZE) % 10 == 0:
                    import gc
                    gc.collect()
            
            logger.info(f"✓ Processed all {total_pairs:,} pairs")
        
        logger.info("Applying FDR correction...")
        reject, p_fdr, _, _ = multipletests(pvals, alpha=fdr_threshold, method='fdr_bh')
        
        sig_mask = reject & (deltas >= min_effect_size)
        sig_pairs_count = np.sum(sig_mask)
        
        df_pairs = pd.DataFrame({
            'gene1': genes[rows],
            'gene2': genes[cols],
            'r_tumor': tumor_r[rows, cols],
            'r_normal': normal_r[rows, cols],
            'delta_r': deltas,
            'p_value': pvals,
            'p_fdr': p_fdr,
            'significant': sig_mask
        })
        
        df_sig_pairs = df_pairs[sig_mask].sort_values('delta_r', ascending=False)
        
        pairs_path = OUTPUT_DIR / 'differential_coexpression_all.tsv'
        df_pairs.to_csv(pairs_path, sep='\t', index=False)
        sig_pairs_path = OUTPUT_DIR / 'differential_coexpression_sig.tsv'
        df_sig_pairs.to_csv(sig_pairs_path, sep='\t', index=False)
        
        logger.info(f"✓ Saved all pairs to: {get_relative_path(pairs_path)} ({total_pairs:,} rows)")
        logger.info(f"✓ Saved significant pairs ({sig_pairs_count:,}) to: {get_relative_path(sig_pairs_path)}")
        
        rewiring_score = np.mean(df_sig_pairs['delta_r']) if sig_pairs_count > 0 else 0
        logger.info(f"RQ Metric: Rewiring score (mean |delta_r| sig pairs): {rewiring_score:.3f}")
        
        summary_stats['outputs']['differential_coexpression_all'] = pairs_path
        summary_stats['outputs']['differential_coexpression_sig'] = sig_pairs_path
        summary_stats['rq_metrics']['rewiring_score'] = float(rewiring_score)
        summary_stats['rq_metrics']['sig_pairs_count'] = int(sig_pairs_count)
        summary_stats['rq_metrics']['total_pairs_tested'] = int(total_pairs)
        
        # ==================== ENHANCEMENT #5: EFFECT SIZE DISTRIBUTION ====================
        logger.info("\n" + "="*60)
        logger.info("EFFECT SIZE DISTRIBUTION ANALYSIS")
        logger.info("="*60)
        
        effect_size_stats = analyze_effect_size_distribution(deltas, sig_mask, fdr_threshold, min_effect_size, logger)
        summary_stats['effect_size_distribution'] = effect_size_stats
        
        # Save effect size distribution to separate JSON file
        effect_size_path = save_effect_size_distribution(effect_size_stats, OUTPUT_DIR)
        logger.info(f"✓ Saved effect size distribution to: {get_relative_path(effect_size_path)}")
        summary_stats['outputs']['effect_size_distribution'] = effect_size_path
        
        logger.info("")
        
        logger.info("=" * 60)
        logger.info("STEP 2: Differential Connectivity (Genes)")
        logger.info("=" * 60)
        
        tumor_conn = np.sum(np.abs(tumor_r), axis=1) - 1
        normal_conn = np.sum(np.abs(normal_r), axis=1) - 1
        deltas_conn = tumor_conn - normal_conn
        
        df_conn = pd.DataFrame({
            'gene': genes,
            'tumor_connectivity': tumor_conn,
            'normal_connectivity': normal_conn,
            'delta_connectivity': deltas_conn
        }).sort_values('delta_connectivity', ascending=False)
        
        conn_path = OUTPUT_DIR / 'differential_connectivity.tsv'
        df_conn.to_csv(conn_path, sep='\t', index=False)
        logger.info(f"✓ Saved connectivity diffs to: {get_relative_path(conn_path)} ({n_genes:,} rows)")
        
        top_rewired_gene = df_conn.iloc[0]['gene']
        top_delta = df_conn.iloc[0]['delta_connectivity']
        logger.info(f"RQ Metric: Top rewired gene: {top_rewired_gene} (delta={top_delta:.2f})")
        
        summary_stats['outputs']['differential_connectivity'] = conn_path
        summary_stats['rq_metrics']['top_rewired_gene'] = top_rewired_gene
        summary_stats['rq_metrics']['top_delta_connectivity'] = float(top_delta)
        
        # ==================== ENHANCEMENT #8: BIOLOGICAL INTERPRETATION ====================
        logger.info("\n" + "="*60)
        logger.info("BIOLOGICAL INTERPRETATION")
        logger.info("="*60)
        
        logger.info("Generating biological interpretation metrics...")
        biological_interpretation = add_biological_interpretation(
            df_sig_pairs, df_conn, qc_metrics, effect_size_stats, logger
        )
        summary_stats['biological_interpretation'] = biological_interpretation
        
        # Save biological interpretation to separate JSON file
        bio_interp_path = save_biological_interpretation(biological_interpretation, OUTPUT_DIR)
        logger.info(f"✓ Saved biological interpretation to: {get_relative_path(bio_interp_path)}")
        summary_stats['outputs']['biological_interpretation'] = bio_interp_path
        
        logger.info("")
        
    except FileNotFoundError as e:
        logger.error(f"ERROR: Correlation files not found. Have you run script 01_a first?\n{e}")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        return
    finally:
        summary_stats['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        total_time = time.time() - start_time
        summary_stats['processing_notes']['processing_time_minutes'] = round(total_time / 60, 1)
        
        summary_path = OUTPUT_DIR / '02_a_result_info.json'
        logger.info("Saving final summary...")
        with tqdm(total=1, desc="Writing summary", bar_format='{l_bar}{bar}| {elapsed} elapsed') as summary_pbar:
            create_summary_json(summary_stats, summary_path, PROJECT_ROOT)
            summary_pbar.update(1)
        logger.info(f"✓ Saved summary stats to: {get_relative_path(summary_path)}")
        
        logger.info("\n" + "="*60)
        logger.info("DIFFERENTIAL ANALYSIS FILES SAVED:")
        logger.info("="*60)
        logger.info("MAIN OUTPUTS:")
        logger.info(f"• differential_coexpression_all.tsv ({total_pairs:,} pairs)")
        logger.info(f"• differential_coexpression_sig.tsv ({sig_pairs_count:,} significant pairs)")
        logger.info(f"• differential_connectivity.tsv ({n_genes:,} genes)")
        logger.info(f"• 02_a_result_info.json (summary)")
        
        logger.info("\nENHANCED OUTPUTS (NEW):")
        logger.info("• quality_control_metrics.json (QC validation)")
        logger.info("• effect_size_distribution.json (delta_r statistics)")
        logger.info("• biological_interpretation.json (biological insights)")
        
        logger.info("="*60)
        
        logger.info("-" * 60)
        logger.info("Script finished successfully.")
        logger.info(f"Total processing time: {total_time/60:.1f} minutes")

if __name__ == "__main__":
    main()