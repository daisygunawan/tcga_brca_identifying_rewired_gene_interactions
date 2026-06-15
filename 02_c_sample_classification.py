"""
02_c_sample_classification.py (Refactored for Multi-Method Comparison)

Script Purpose:
This script performs tumor vs. normal sample classification using real TCGA-BRCA expression data
with THREE different sampling strategies for comprehensive comparison.

NEW: Flag-based modular sampling system - enable/disable methods via SAMPLING_METHOD_FLAGS

Sampling Methods Compared:
1. MEDIAN: Selects 100 tumor samples closest to median expression profile
2. STRATIFIED_RANDOM: Randomly selects tumors to match normal count (113 vs 113)
3. CLUSTER_BASED: Clusters tumors into subtypes and samples proportionally

Each method is evaluated with:
- Regularized Random Forest and XGBoost
- Soft and Hard voting ensembles
- Nested cross-validation
- Feature stability analysis

Results are saved in separate directories with comprehensive comparison reporting.
"""

import pandas as pd
import numpy as np
import json
import logging
import time
from pathlib import Path
from math import gcd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef, roc_auc_score, roc_curve, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.cluster import KMeans

import math
import warnings
warnings.filterwarnings('ignore')

from utils.config import load_config
from utils.file import ensure_dir, get_relative_path, get_auto_output_path
from utils.classification_enhancements import run_all_enhancements

# ============================================================================
# SAMPLING METHODS CONFIGURATION
# ============================================================================

# Enable/Disable sampling methods
SAMPLING_METHOD_FLAGS = {
    'median': True,           # Median-based selection (best performance)
    'stratified_random': False,  # Disabled: poor performance, no ensemble benefit
    'cluster_based': True     # Cluster-based proportional sampling (best generalization)
}

# Get list of enabled methods
SAMPLING_METHODS = [method for method, enabled in SAMPLING_METHOD_FLAGS.items() if enabled]

# Validate at least one method is enabled
if len(SAMPLING_METHODS) == 0:
    raise ValueError("At least one sampling method must be enabled in SAMPLING_METHOD_FLAGS")

# Classification constants defaults
# CAREFUL :: can be OVERRIDEN by CONFIG.yaml, classification
DEFAULT_TOP_FEATURES = 100 # top_n_features
DEFAULT_TUMOR_SAMPLE_SELECT = 113 # same as normal samples
TEST_SIZE = 0.3  # test_size: Increased from 0.2 for better validation
CV_FOLDS = 5 # cv_folds
N_ESTIMATORS = 500 #n_estimators
RANDOM_STATE = 42 # random_state

# Sampling method constants
CLUSTER_N = 5  # Number of clusters for cluster-based sampling

# ============================================================================
# WEIGHTED ENSEMBLE SENSITIVITY TEST CONFIGURATION
# ============================================================================

# Enable / disable the entire weighted sensitivity sweep
ENABLE_WEIGHTED_SENSITIVITY = True

# Weight grid: 9 points from pure XGBoost (rf=0) to pure RF (rf=1)
# rf=0.0  ≡  XGBoost alone   (redundant with individual model, kept for curve continuity)
# rf=0.5  ≡  equal soft vote  (redundant with Ensemble_Soft, kept for curve continuity)
# rf=1.0  ≡  RF alone         (redundant with individual model, kept for curve continuity)
_RF_WEIGHT_GRID = [0.000, 0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875, 1.000]


def _format_weight_name(rf_weight: float, xgb_weight: float) -> str:
    """
    Return a fixed-length 23-character weight label for filenames/keys.

    Converts each weight to an integer in [0, 10000] and zero-pads to
    exactly 5 digits so all 9 names sort correctly alphabetically.

    Examples
    --------
    >>> _format_weight_name(0.000, 1.000)
    'rf_00000_xgb_10000'
    >>> _format_weight_name(0.125, 0.875)
    'rf_01250_xgb_08750'
    >>> _format_weight_name(0.500, 0.500)
    'rf_05000_xgb_05000'
    >>> _format_weight_name(1.000, 0.000)
    'rf_10000_xgb_00000'
    """
    rf_int  = int(round(rf_weight  * 10_000))
    xgb_int = int(round(xgb_weight * 10_000))
    return f"rf_{rf_int:05d}_xgb_{xgb_int:05d}"


def _ratio_description(rf_w: float, xgb_w: float) -> str:
    """
    Human-readable ratio string derived from integer arithmetic (no float Fraction risk).

    Uses eighths representation (grid step = 0.125 = 1/8) so all 9 points reduce cleanly.

    Examples: 0.000/1.000 -> 'Pure XGBoost', 0.125/0.875 -> '1:7', 0.500/0.500 -> '1:1 Equal'
    """
    if rf_w == 0.0:
        return 'Pure XGBoost'
    if xgb_w == 0.0:
        return 'Pure RandomForest'

    rf_eighths  = int(round(rf_w  * 8))
    xgb_eighths = int(round(xgb_w * 8))

    g = gcd(rf_eighths, xgb_eighths)
    n, d = rf_eighths // g, xgb_eighths // g

    if n == d:
        return '1:1 Equal'
    return f'{n}:{d}'


# Build the 9-combination list once at import time
WEIGHT_COMBINATIONS = []
for _rf in _RF_WEIGHT_GRID:
    _xgb = round(1.0 - _rf, 4)
    _rf  = round(_rf, 4)
    WEIGHT_COMBINATIONS.append({
        'rf':                _rf,
        'xgb':               _xgb,
        'name':              _format_weight_name(_rf, _xgb),
        'ratio_description': _ratio_description(_rf, _xgb),
    })

assert len(WEIGHT_COMBINATIONS) == 9, \
    f"Expected 9 weight combinations, got {len(WEIGHT_COMBINATIONS)}"

# ============================================================================
# DIVERSIFIED STACKING META-LEARNER SENSITIVITY TEST CONFIGURATION
# ============================================================================

# Enable / disable the entire stacking sensitivity sweep
ENABLE_STACKING_SENSITIVITY = True

# LightGBM is optional — imported lazily inside the function.
# If not installed, the sweep runs with 4 meta-learners instead of 5.
# To install: pip install lightgbm

def _build_stacking_meta_learners() -> list:
    """
    Return the ordered list of (key, display_name, callable) tuples used by
    run_stacking_sensitivity_test().

    LogisticRegression (L2) is intentionally first — it matches the existing
    Ensemble_Stacking in the main CV block and serves as an anchor for comparison.

    Each entry is:
        key          : short identifier used in filenames / JSON keys
        display_name : human-readable label for logs and plots
        factory      : zero-arg callable returning a fresh unfitted estimator
    """
    meta_learners = [
        (
            'LR_L2',
            'LogisticRegression (L2)',
            lambda: LogisticRegression(
                penalty='l2', C=1.0, solver='lbfgs',
                max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced'
            ),
        ),
        (
            'LR_L1',
            'LogisticRegression (L1/Lasso)',
            lambda: LogisticRegression(
                penalty='l1', C=0.5, solver='liblinear',
                max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced'
            ),
        ),
        (
            'LR_EN',
            'LogisticRegression (ElasticNet)',
            lambda: LogisticRegression(
                penalty='elasticnet', l1_ratio=0.5, C=0.5, solver='saga',
                max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced'
            ),
        ),
        (
            'SVC_RBF',
            'SVC (RBF kernel)',
            lambda: SVC(
                probability=True, kernel='rbf', C=1.0,
                random_state=RANDOM_STATE, class_weight='balanced'
            ),
        ),
    ]

    # Soft import of LightGBM — degrade gracefully if absent
    try:
        from lightgbm import LGBMClassifier
        meta_learners.append((
            'LightGBM',
            'LightGBM',
            lambda: LGBMClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=RANDOM_STATE, class_weight='balanced',
                verbose=-1,
            ),
        ))
    except ImportError:
        pass  # LightGBM unavailable — sweep continues with 4 meta-learners

    return meta_learners

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def convert_to_json_serializable(obj):
    """
    Convert numpy/pandas types to JSON-serializable Python types.
    """
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        # Convert all keys to strings and values recursively
        return {str(k): convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif pd.isna(obj):  # Handle NaN values
        return None
    else:
        return obj


def analyze_cancer_enrichment_by_threshold(feature_importance_df, 
                                           annotated_hubs,
                                           thresholds=[10, 20, 30, 50]):
    """
    Analyze cancer gene enrichment at multiple rank thresholds.
    
    Args:
        feature_importance_df: DataFrame with 'feature', 'importance', 'cancer_relevance'
        annotated_hubs: List of hub dictionaries
        thresholds: List of rank cutoffs to analyze
    
    Returns:
        Dictionary with enrichment stats at each threshold
    """
    gene_info = {hub['gene']: hub['cancer_relevance'] for hub in annotated_hubs}
    
    results = {
        'by_rank': {},
        'by_cumulative_importance': {}
    }
    
    # Analysis by rank threshold
    for n in thresholds:
        if n > len(feature_importance_df):
            continue
            
        top_n = feature_importance_df.head(n)
        cancer_count = sum(
            1 for f in top_n['feature']
            if gene_info.get(f, 'non_cancer') in ['breast_cancer', 'cancer']
        )
        
        results['by_rank'][f'top_{n}'] = {
            'count': int(cancer_count),
            'total': int(n),
            'percentage': float((cancer_count / n) * 100),
            'cumulative_importance': float(top_n['importance'].sum()),
            'cancer_genes': [
                f for f in top_n['feature']
                if gene_info.get(f, 'non_cancer') in ['breast_cancer', 'cancer']
            ]
        }
    
    # Analysis by cumulative importance threshold
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        features_needed = 0
        cum_imp = 0
        for _, row in feature_importance_df.iterrows():
            features_needed += 1
            cum_imp += row['importance']
            if cum_imp >= threshold:
                break
        
        top_for_threshold = feature_importance_df.head(features_needed)
        cancer_count = sum(
            1 for f in top_for_threshold['feature']
            if gene_info.get(f, 'non_cancer') in ['breast_cancer', 'cancer']
        )
        
        results['by_cumulative_importance'][f'{int(threshold*100)}pct'] = {
            'features_needed': int(features_needed),
            'cancer_count': int(cancer_count),
            'cancer_percentage': float((cancer_count / features_needed) * 100) if features_needed > 0 else 0,
            'importance_threshold': float(threshold)
        }
    
    return results


# ============================================================================
# BASE SAMPLING METHOD CLASS
# ============================================================================

class SamplingMethod:
    """
    Base class for all sampling methods.
    Provides standardized interface for sampling, analysis, and reporting.
    """
    
    def __init__(self, method_name, config, logger):
        """
        Initialize sampling method.
        
        Args:
            method_name: Name of the sampling method
            config: Configuration dictionary
            logger: Logger instance
        """
        self.method_name = method_name
        self.config = config
        self.logger = logger
        self.method_info = {}
        self.X = None
        self.y = None
        self.valid_features = None
        
    def configure(self, **params):
        """
        Configure method-specific parameters.
        Override in subclasses.
        """
        self.params = params
        return self
    
    def sample(self, tumor_df, normal_df, feature_genes):
        """
        Perform sampling to create balanced dataset.
        Must be implemented in subclasses.
        
        Returns:
            tuple: (X, y, valid_features, method_info)
        """
        raise NotImplementedError("Subclasses must implement sample()")
    
    def analyze(self, results):
        """
        Perform method-specific analysis on results.
        Optional - override in subclasses if needed.
        
        Args:
            results: Classification results dictionary
        """
        self.logger.info(f"  No additional analysis for {self.method_name}")
        return {}
    
    def report(self, output_dir):
        """
        Generate method-specific visualizations.
        Optional - override in subclasses if needed.
        
        Args:
            output_dir: Directory to save reports
        """
        self.logger.info(f"  No additional reports for {self.method_name}")
        return []
    
    def get_description(self):
        """
        Get method description for documentation.
        Override in subclasses.
        """
        return {
            'name': self.method_name,
            'description': 'Base sampling method',
            'parameters': self.params if hasattr(self, 'params') else {}
        }


# ============================================================================
# MEDIAN SAMPLING METHOD
# ============================================================================

class MedianSamplingMethod(SamplingMethod):
    """
    Median-Based Selection Method.
    Selects tumor samples closest to median expression profile.
    """
    
    def __init__(self, config, logger):
        super().__init__('median', config, logger)
        
    def configure(self, n_tumor_samples=100):
        """Configure median sampling parameters."""
        self.n_tumor_samples = n_tumor_samples
        self.params = {'n_tumor_samples': n_tumor_samples}
        return self
    
    def sample(self, tumor_df, normal_df, feature_genes):
        """Perform median-based sampling."""
        self.logger.info("Applying MEDIAN sampling: Selecting tumors closest to median profile...")
        
        # Filter to feature genes
        valid_features = [f for f in feature_genes if f in tumor_df.columns and f in normal_df.columns]
        tumor_features = tumor_df[valid_features]
        normal_features = normal_df[valid_features]
        
        # Calculate median tumor profile
        median_profile = tumor_features.median(axis=0)
        
        # Calculate distance for each tumor sample
        distances = tumor_features.apply(lambda row: euclidean(row, median_profile), axis=1)
        
        # Select samples with smallest distance
        n_select = self.n_tumor_samples
        top_indices = distances.nsmallest(n_select).index
        selected_tumor_df = tumor_features.loc[top_indices]
        
        # Combine with normal samples
        self.X = pd.concat([selected_tumor_df, normal_features])
        self.y = np.array([1] * len(selected_tumor_df) + [0] * len(normal_features))
        self.valid_features = valid_features
        
        # Store method info
        self.method_info = {
            'method': 'median',
            'n_tumor_selected': int(len(selected_tumor_df)),
            'n_normal': int(len(normal_features)),
            'distance_metric': 'euclidean',
            'selection_criterion': 'closest_to_median',
            'distance_stats': {
                'min': float(distances.loc[top_indices].min()),
                'max': float(distances.loc[top_indices].max()),
                'mean': float(distances.loc[top_indices].mean())
            }
        }
        
        self.logger.info(f"  Selected {len(selected_tumor_df)} tumor samples closest to median")
        self.logger.info(f"  Distance range: {self.method_info['distance_stats']['min']:.2f} to "
                        f"{self.method_info['distance_stats']['max']:.2f}")
        self.logger.info(f"  Final dataset: {len(selected_tumor_df)} tumor, {len(normal_features)} normal")
        
        return self.X, self.y, self.valid_features, self.method_info
    
    def analyze(self, results):
        """Analyze median sampling characteristics."""
        analysis = {
            'homogeneity_score': self.method_info['distance_stats']['max'] - self.method_info['distance_stats']['min'],
            'selection_efficiency': self.method_info['distance_stats']['mean'] / self.method_info['distance_stats']['max'],
            'interpretation': 'High homogeneity - selected samples are very similar to median'
        }
        return analysis
    
    def report(self, output_dir):
        """Generate median sampling visualization."""
        # Could add distance distribution plot here
        return []
    
    def get_description(self):
        return {
            'name': 'Median-Based Selection',
            'description': 'Selects tumor samples closest to median expression profile',
            'rationale': 'Captures most representative tumor phenotype',
            'advantages': [
                'Reduces outlier influence',
                'Creates clear separation',
                'Identifies core tumor features'
            ],
            'disadvantages': [
                'Oversimplifies tumor heterogeneity',
                'May create artificial separability',
                'Risk of overfitting'
            ],
            'parameters': self.params
        }


# ============================================================================
# STRATIFIED RANDOM SAMPLING METHOD
# ============================================================================

class StratifiedRandomSamplingMethod(SamplingMethod):
    """
    Stratified Random Sampling Method.
    Randomly selects tumor samples to match normal count.
    """
    
    def __init__(self, config, logger):
        super().__init__('stratified_random', config, logger)
        
    def configure(self, random_state=42):
        """Configure stratified random sampling parameters."""
        self.random_state = random_state
        self.params = {'random_state': random_state}
        return self
    
    def sample(self, tumor_df, normal_df, feature_genes):
        """Perform stratified random sampling."""
        self.logger.info("Applying STRATIFIED RANDOM sampling: Random tumor selection...")
        
        # Filter to feature genes
        valid_features = [f for f in feature_genes if f in tumor_df.columns and f in normal_df.columns]
        tumor_features = tumor_df[valid_features]
        normal_features = normal_df[valid_features]
        
        # Match normal sample count
        n_select = len(normal_features)
        
        # Random selection without replacement
        np.random.seed(self.random_state)
        selected_indices = np.random.choice(tumor_features.index, n_select, replace=False)
        selected_tumor_df = tumor_features.loc[selected_indices]
        
        # Combine with normal samples
        self.X = pd.concat([selected_tumor_df, normal_features])
        self.y = np.array([1] * len(selected_tumor_df) + [0] * len(normal_features))
        self.valid_features = valid_features
        
        # Store method info
        self.method_info = {
            'method': 'stratified_random',
            'n_tumor_selected': int(len(selected_tumor_df)),
            'n_normal': int(len(normal_features)),
            'selection_method': 'random_without_replacement',
            'random_seed': int(self.random_state),
            'balance_ratio': float(len(selected_tumor_df) / len(normal_features))
        }
        
        self.logger.info(f"  Randomly selected {len(selected_tumor_df)} tumor samples")
        self.logger.info(f"  Balanced dataset: {len(selected_tumor_df)} tumor, {len(normal_features)} normal")
        
        return self.X, self.y, self.valid_features, self.method_info
    
    def get_description(self):
        return {
            'name': 'Stratified Random Sampling',
            'description': 'Random selection of tumor samples matching normal sample count',
            'rationale': 'Preserves natural tumor heterogeneity',
            'advantages': [
                'Maintains natural variability',
                'Better generalization',
                'Statistically unbiased'
            ],
            'disadvantages': [
                'May include outliers',
                'More variable performance',
                'Randomness requires multiple runs'
            ],
            'parameters': self.params
        }


# ============================================================================
# CLUSTER-BASED SAMPLING METHOD
# ============================================================================

class ClusterBasedSamplingMethod(SamplingMethod):
    """
    Cluster-Based Proportional Sampling Method.
    Clusters tumors into subtypes and samples proportionally.
    """
    
    def __init__(self, config, logger):
        super().__init__('cluster_based', config, logger)
        
    def configure(self, n_clusters=5, random_state=42):
        """Configure cluster-based sampling parameters."""
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.params = {
            'n_clusters': n_clusters,
            'random_state': random_state
        }
        return self
    
    def sample(self, tumor_df, normal_df, feature_genes):
        """Perform cluster-based proportional sampling."""
        self.logger.info("Applying CLUSTER-BASED sampling: Proportional sampling from tumor clusters...")
        
        # Filter to feature genes
        valid_features = [f for f in feature_genes if f in tumor_df.columns and f in normal_df.columns]
        tumor_features = tumor_df[valid_features]
        normal_features = normal_df[valid_features]
        
        # Perform K-means clustering on tumors
        self.logger.info(f"  Clustering {len(tumor_features)} tumors into {self.n_clusters} subtypes...")
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(tumor_features)
        
        # Calculate cluster sizes
        unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
        cluster_sizes_dict = {int(cluster_id): int(count) for cluster_id, count in zip(unique_clusters, cluster_counts)}
        self.logger.info(f"  Cluster sizes: {cluster_sizes_dict}")
        
        # Determine samples per cluster (proportional to cluster size)
        target_n_tumor = len(normal_features)  # Match normal count
        samples_per_cluster = {}
        selected_indices = []
        
        for cluster_id in unique_clusters:
            cluster_indices = tumor_features.index[cluster_labels == cluster_id]
            # Proportional sampling
            n_from_cluster = max(1, int((cluster_counts[cluster_id] / len(tumor_features)) * target_n_tumor))
            
            # Random sample from this cluster
            np.random.seed(self.random_state + int(cluster_id))
            sampled_idx = np.random.choice(cluster_indices, min(n_from_cluster, len(cluster_indices)), replace=False)
            selected_indices.extend(sampled_idx)
            
            samples_per_cluster[int(cluster_id)] = int(len(sampled_idx))
        
        # If we need more samples, distribute evenly
        while len(selected_indices) < target_n_tumor:
            for cluster_id in unique_clusters:
                cluster_indices = tumor_features.index[cluster_labels == cluster_id]
                available = set(cluster_indices) - set(selected_indices)
                if len(available) > 0:
                    selected_indices.append(np.random.choice(list(available)))
                    samples_per_cluster[int(cluster_id)] += 1
                    if len(selected_indices) >= target_n_tumor:
                        break
        
        # Trim if we have too many
        selected_indices = selected_indices[:target_n_tumor]
        selected_tumor_df = tumor_features.loc[selected_indices]
        
        # Combine with normal samples
        self.X = pd.concat([selected_tumor_df, normal_features])
        self.y = np.array([1] * len(selected_tumor_df) + [0] * len(normal_features))
        self.valid_features = valid_features
        
        # Store method info
        self.method_info = {
            'method': 'cluster_based',
            'n_tumor_selected': int(len(selected_tumor_df)),
            'n_normal': int(len(normal_features)),
            'n_clusters': int(self.n_clusters),
            'clustering_method': 'kmeans',
            'samples_per_cluster': samples_per_cluster,
            'cluster_sizes': cluster_sizes_dict,
            'sampling_strategy': 'proportional_to_cluster_size'
        }
        
        self.logger.info(f"  Proportional sampling from clusters: {samples_per_cluster}")
        self.logger.info(f"  Selected {len(selected_tumor_df)} tumor samples from {self.n_clusters} clusters")
        self.logger.info(f"  Final dataset: {len(selected_tumor_df)} tumor, {len(normal_features)} normal")
        
        return self.X, self.y, self.valid_features, self.method_info
    
    def analyze(self, results):
        """Analyze cluster-based sampling characteristics."""
        analysis = {
            'cluster_diversity': len(self.method_info['samples_per_cluster']),
            'sampling_balance': min(self.method_info['samples_per_cluster'].values()) / 
                              max(self.method_info['samples_per_cluster'].values()),
            'interpretation': 'Preserves tumor subtype diversity through proportional sampling'
        }
        return analysis
    
    def get_description(self):
        return {
            'name': 'Cluster-Based Proportional Sampling',
            'description': 'Samples tumors proportionally from identified expression clusters',
            'rationale': 'Captures tumor subtype diversity',
            'advantages': [
                'Preserves subtype diversity',
                'Biologically meaningful',
                'Balances representation'
            ],
            'disadvantages': [
                'Computationally intensive',
                'Sensitive to clustering parameters',
                'Complex implementation'
            ],
            'parameters': self.params
        }


# ============================================================================
# SAMPLING METHOD FACTORY
# ============================================================================

def get_sampling_method_instance(method_name, config, logger):
    """
    Factory function to create sampling method instance.
    
    Args:
        method_name: Name of the sampling method
        config: Configuration dictionary
        logger: Logger instance
    
    Returns:
        SamplingMethod instance
    """
    method_classes = {
        'median': MedianSamplingMethod,
        'stratified_random': StratifiedRandomSamplingMethod,
        'cluster_based': ClusterBasedSamplingMethod
    }
    
    if method_name not in method_classes:
        raise ValueError(f"Unknown sampling method: {method_name}. Available: {list(method_classes.keys())}")
    
    # Check if method is enabled
    if not SAMPLING_METHOD_FLAGS.get(method_name, False):
        raise ValueError(f"Sampling method '{method_name}' is disabled in SAMPLING_METHOD_FLAGS")
    
    # Create and configure instance
    method_class = method_classes[method_name]
    instance = method_class(config, logger)
    
    # Configure with default parameters
    if method_name == 'median':
        n_tumor_samples = config['classification'].get('n_tumor_samples_select', DEFAULT_TUMOR_SAMPLE_SELECT)
        instance.configure(n_tumor_samples=n_tumor_samples)
    elif method_name == 'stratified_random':
        instance.configure(random_state=RANDOM_STATE)
    elif method_name == 'cluster_based':
        instance.configure(n_clusters=CLUSTER_N, random_state=RANDOM_STATE)
    
    return instance


# ============================================================================
# CORE CLASSIFICATION FUNCTIONS
# ============================================================================

def setup_logging(config, output_dir, method_name):
    """
    Set up logging with method-specific log file.
    """
    logger = logging.getLogger(f'{__name__}_{method_name}')
    logger.setLevel(config['logging']['level'])
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    log_dir = output_dir / 'logs'
    ensure_dir(log_dir)
    log_file = log_dir / f'02_c_{method_name}_classification.log'
    
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


def setup_master_logger(config, comparison_dir):
    """
    Set up master logger for the comparison.
    """
    master_logger = logging.getLogger('master_comparison')
    master_logger.setLevel(logging.INFO)
    
    if master_logger.hasHandlers():
        master_logger.handlers.clear()
    
    master_log_file = comparison_dir / 'master_comparison.log'
    file_handler = logging.FileHandler(master_log_file, mode='w')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    master_logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    master_logger.addHandler(console_handler)
    
    return master_logger


def load_top_feature_genes(config, project_root, top_n, logger):
    """
    Load annotated hubs from 02_b and select top N genes ranked by delta_connectivity.
    """
    logger.info("=" * 60)
    logger.info("STEP 1: Loading and Ranking Feature Genes from 02_b")
    logger.info("=" * 60)
    
    # Load annotated hubs from 02_b
    dcea_dir = Path(project_root) / config['paths']['dcea_viz_enrich']
    annotated_path = dcea_dir / 'annotated_hubs.json'
    
    if not annotated_path.exists():
        raise FileNotFoundError(
            f"Annotated hubs missing at {annotated_path}. "
            f"Please run 02_b_dcea_viz_enrich.py first."
        )
    
    logger.info(f"Loading annotated hubs from: {get_relative_path(annotated_path)}")
    with open(annotated_path, 'r') as f:
        annotated_hubs = json.load(f)
    
    # Convert to DataFrame for easy ranking
    hubs_df = pd.DataFrame(annotated_hubs)
    logger.info(f"✓ Loaded {len(hubs_df)} annotated rewired hubs")
    
    # Rank by delta_connectivity (descending)
    hubs_df = hubs_df.sort_values('delta_connectivity', ascending=False)
    logger.info("✓ Ranked hubs by delta_connectivity (descending)")
    
    # Select top N genes
    top_hubs = hubs_df.head(top_n)
    top_feature_genes = top_hubs['gene'].tolist()
    
    logger.info(f"✓ Selected top {len(top_feature_genes)} genes with highest network disruption")
    logger.info(f"\nTop 5 feature genes:")
    for i, (_, row) in enumerate(top_hubs.head(5).iterrows(), 1):
        logger.info(
            f"  {i}. {row['gene']} "
            f"(Δ={row['delta_connectivity']:.1f}, {row['cancer_relevance']})"
        )
    
    return top_feature_genes, annotated_hubs, hubs_df


def load_real_data(config, project_root, logger):
    """
    Load real TCGA-BRCA expression data.
    """
    logger.info("\n" + "=" * 60)
    logger.info("Loading Real TCGA-BRCA Expression Data")
    logger.info("=" * 60)
    
    preprocessed_dir = Path(project_root) / config['paths']['preprocessed']
    tumor_matrix_path = preprocessed_dir / 'matrices' / 'tumor_matrix.tsv'
    normal_matrix_path = preprocessed_dir / 'matrices' / 'normal_matrix.tsv'
    
    if not tumor_matrix_path.exists() or not normal_matrix_path.exists():
        raise FileNotFoundError(
            f"Expression matrices not found in {preprocessed_dir}. "
            f"Please run 00_b_data_preprocess.py first."
        )
    
    logger.info(f"Loading tumor expression data from: {tumor_matrix_path.name}")
    tumor_df = pd.read_csv(tumor_matrix_path, sep='\t', index_col=0)
    logger.info(f"  Dimensions: {tumor_df.shape[0]} genes × {tumor_df.shape[1]} samples")
    
    logger.info(f"Loading normal expression data from: {normal_matrix_path.name}")
    normal_df = pd.read_csv(normal_matrix_path, sep='\t', index_col=0)
    logger.info(f"  Dimensions: {normal_df.shape[0]} genes × {normal_df.shape[1]} samples")
    
    logger.info(f"✓ Loaded {tumor_df.shape[1]} tumor samples and {normal_df.shape[1]} normal samples")
    
    # Transpose so samples are in rows (needed for classification)
    tumor_df = tumor_df.T
    normal_df = normal_df.T
    logger.info("✓ Transposed matrices (samples now in rows)")
    
    return tumor_df, normal_df


def plot_xgboost_learning_curve(xgb_model, output_dir, sampling_label, logger):
    """
    Plot XGBoost log loss across boosting rounds for training and validation sets.

    Requires the model to have been trained with eval_set and eval_metric='logloss'.
    Saves both a PNG plot and a JSON data file for reproducibility/reporting.

    Args:
        xgb_model:      Trained XGBClassifier with evals_result available.
        output_dir:     Path object — directory where outputs are saved.
        sampling_label: String label for the sampling strategy (used in title/filename).
        logger:         Logger instance.

    Returns:
        plot_path (Path) if successful, else None.
    """
    results = xgb_model.evals_result()

    if not results:
        logger.warning("No eval results found — model was not trained with eval_set. Skipping learning curve.")
        return None

    train_loss = results['validation_0']['logloss']
    val_loss   = results['validation_1']['logloss']
    n_rounds   = len(train_loss)
    rounds     = list(range(1, n_rounds + 1))

    best_round    = int(np.argmin(val_loss)) + 1
    best_val_loss = float(min(val_loss))

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(rounds, train_loss, color='#e74c3c', linewidth=2,
            label='Training Log Loss', alpha=0.85)
    ax.plot(rounds, val_loss,   color='#3498db', linewidth=2,
            label='Validation Log Loss', alpha=0.85)
    ax.axvline(x=best_round, color='#2ecc71', linestyle='--', linewidth=1.5,
               label=f'Best round: {best_round} (val loss = {best_val_loss:.4f})')

    ax.set_xlabel('Boosting Round', fontsize=12, fontweight='bold')
    ax.set_ylabel('Log Loss', fontsize=12, fontweight='bold')
    ax.set_title(
        f'XGBoost Learning Curve — {sampling_label.replace("_", " ").title()} Sampling',
        fontsize=13, fontweight='bold'
    )
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xlim(1, n_rounds)

    plt.tight_layout()

    label_slug = sampling_label.replace(' ', '_').lower()
    plot_path  = output_dir / f'xgb_learning_curve_{label_slug}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ XGBoost learning curve saved: {plot_path.name} (best round: {best_round})")

    # --- JSON companion (for table / reproducibility) ---
    json_data = {
        'sampling_label': sampling_label,
        'n_rounds':       n_rounds,
        'best_round':     best_round,
        'best_val_loss':  best_val_loss,
        'final_train_loss': float(train_loss[-1]),
        'final_val_loss':   float(val_loss[-1]),
        'train_loss': [round(float(v), 6) for v in train_loss],
        'val_loss':   [round(float(v), 6) for v in val_loss],
    }
    json_path = output_dir / f'xgb_learning_curve_{label_slug}.json'
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    logger.info(f"✓ XGBoost learning curve data saved: {json_path.name}")

    return plot_path


def plot_confusion_matrices(
    y_test,
    predictions_dict,
    probas_dict,
    output_dir,
    sampling_label,
    logger
):
    """
    Generate a 2x2 panel of annotated confusion matrices (one per model),
    save as PNG + companion JSON.

    Layout (2 rows x 2 cols):
        RandomForest   | XGBoost
        Ensemble_Soft  | Ensemble_Hard

    Each cell shows:
        - Raw count (large, bold)
        - Row-normalised % in parentheses
        - Cell label (True Negative / False Positive / etc.)
        - Colour intensity from Blues cmap tied to count (matched to colorbar)
        - White text on dark cells, dark text on pale cells

    Footer per matrix:
        Sens, Spec, PPV, NPV, MCC
        Total errors, FP count, FN count

    Args:
        y_test            : array-like, true labels (0=normal, 1=tumor)
        predictions_dict  : dict {model_name: y_pred array}
        probas_dict       : dict {model_name: y_proba array}  (kept for future use / MCC)
        output_dir        : pathlib.Path, directory to write outputs
        sampling_label    : str, e.g. 'cluster_based' or 'median' (used in filenames + title)
        logger            : logging.Logger instance

    Returns:
        png_path   : pathlib.Path to saved PNG
        json_path  : pathlib.Path to saved JSON
        summary    : dict of per-model metrics
    """
    import json
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    from matplotlib.patches import Rectangle
    from sklearn.metrics import confusion_matrix, matthews_corrcoef

    model_order = ['RandomForest', 'XGBoost', 'Ensemble_Soft', 'Ensemble_Hard']
    titles      = ['Random Forest', 'XGBoost', 'Soft-Voting Ensemble', 'Hard-Voting Ensemble']

    # ── Collect all counts to set a shared colorbar range ────────────────────
    all_counts = []
    cms = {}
    for name in model_order:
        y_pred = predictions_dict[name]
        cm_arr = confusion_matrix(y_test, y_pred)   # [[TN, FP], [FN, TP]]
        cms[name] = cm_arr
        all_counts += cm_arr.ravel().tolist()

    vmin = 0
    vmax = max(all_counts)
    cmap = plt.cm.Blues

    # ── Figure + GridSpec ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    gs  = fig.add_gridspec(
        2, 3,
        width_ratios=[1, 1, 0.045],
        hspace=0.52,
        wspace=0.38,
        left=0.07, right=0.94,
        top=0.90,  bottom=0.07
    )

    label_str = sampling_label.replace('_', ' ').title()
    fig.suptitle(
        f'Confusion Matrices — Tumor vs Normal Classification\n'
        f'({label_str} Sampling  |  Test Set n={len(y_test)}  |  '
        f'{int((y_test == 0).sum())} Normal + {int((y_test == 1).sum())} Tumor)',
        fontsize=15, fontweight='bold', y=0.97
    )

    ax_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    summary = {}

    for idx, (name, title) in enumerate(zip(model_order, titles)):
        ri, ci = ax_positions[idx]
        ax = fig.add_subplot(gs[ri, ci])

        y_pred  = predictions_dict[name]
        cm_arr  = cms[name]
        TN, FP, FN, TP = cm_arr.ravel()
        total   = int(TN + FP + FN + TP)

        # ── Derived metrics ───────────────────────────────────────────────────
        sensitivity = TP / (TP + FN) if (TP + FN) else 0.0
        specificity = TN / (TN + FP) if (TN + FP) else 0.0
        ppv         = TP / (TP + FP) if (TP + FP) else 0.0
        npv         = TN / (TN + FN) if (TN + FN) else 0.0
        mcc_val     = matthews_corrcoef(y_test, y_pred)
        accuracy    = (TP + TN) / total

        summary[name] = {
            'TP':          int(TP),
            'TN':          int(TN),
            'FP':          int(FP),
            'FN':          int(FN),
            'sensitivity': round(float(sensitivity), 4),
            'specificity': round(float(specificity), 4),
            'ppv':         round(float(ppv),         4),
            'npv':         round(float(npv),         4),
            'mcc':         round(float(mcc_val),     4),
            'accuracy':    round(float(accuracy),    4),
            'total_errors': int(FP + FN)
        }

        # Row totals for row-normalised %
        n_row = int(TN + FP)   # actual normal row
        t_row = int(FN + TP)   # actual tumor row

        # [actual_row][pred_col] → (count, row_pct, is_diagonal)
        cells = [
            [(TN, TN / n_row * 100, True),  (FP, FP / n_row * 100, False)],
            [(FN, FN / t_row * 100, False),  (TP, TP / t_row * 100, True)],
        ]

        # ── Draw cells ────────────────────────────────────────────────────────
        for r in range(2):
            for c in range(2):
                count, pct, diagonal = cells[r][c]
                norm_val = count / vmax if vmax > 0 else 0.5
                rgba     = cmap(norm_val)
                ax.add_patch(Rectangle((c, 1 - r), 1, 1, color=rgba))

                # Text colours: white on dark cells, dark on pale cells
                light     = norm_val < 0.45
                txt_main  = '#1a1a2e' if light else 'white'
                txt_sub   = '#555555' if light else '#dddddd'
                txt_label = '#aaaaaa' if light else '#cccccc'

                # Count — dominant
                ax.text(
                    c + 0.5, 1 - r + 0.62, str(count),
                    ha='center', va='center',
                    fontsize=34, fontweight='bold', color=txt_main
                )
                # Row-normalised percentage
                ax.text(
                    c + 0.5, 1 - r + 0.38, f'({pct:.1f}%)',
                    ha='center', va='center',
                    fontsize=13, color=txt_sub
                )
                # Cell type label
                cell_labels = {
                    (0, 0): 'True Negative',  (0, 1): 'False Positive',
                    (1, 0): 'False Negative', (1, 1): 'True Positive'
                }
                ax.text(
                    c + 0.5, 1 - r + 0.16, cell_labels[(r, c)],
                    ha='center', va='center',
                    fontsize=9.5, color=txt_label, style='italic'
                )

        # ── Axes cosmetics ────────────────────────────────────────────────────
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)

        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(
            ['Predicted\nNormal', 'Predicted\nTumor'],
            fontsize=11, fontweight='bold'
        )
        ax.xaxis.set_ticks_position('bottom')

        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(
            ['Actual\nTumor', 'Actual\nNormal'],
            fontsize=11, fontweight='bold'
        )
        ax.yaxis.set_ticks_position('left')

        ax.tick_params(length=0, pad=6)
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_title(
            f'{title}\nAccuracy = {accuracy * 100:.2f}%',
            fontsize=13, fontweight='bold', pad=10
        )

        # ── Footer ────────────────────────────────────────────────────────────
        metrics_line = (
            f'Sens={sensitivity:.3f}  Spec={specificity:.3f}  '
            f'PPV={ppv:.3f}  NPV={npv:.3f}  MCC={mcc_val:.3f}'
        )
        errors_line = f'Errors: {int(FP + FN)}/68    (FP={int(FP)}  FN={int(FN)})'

        # ax.text(
        #     1.0, -0.055, metrics_line,
        #     transform=ax.transAxes, ha='center', va='top',
        #     fontsize=9, color='#333333', family='monospace'
        # )
        # ax.text(
        #     1.0, -0.125, errors_line,
        #     transform=ax.transAxes, ha='center', va='top',
        #     fontsize=9, color='#555555', family='monospace'
        # )

    # ── Colorbar — spans both rows in the third column ────────────────────────
    cbar_ax = fig.add_subplot(gs[:, 2])
    norm    = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm      = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Count', fontsize=11, fontweight='bold', labelpad=10)
    cbar.ax.tick_params(labelsize=9)

    # # Ticks only at values that actually appear in the data
    # tick_vals = sorted(set(all_counts + [0]))
    # cbar.set_ticks(tick_vals)
    # cbar.set_ticklabels([str(v) for v in tick_vals])

    # Ticks at multiples of 5 from 0 to ceiling of vmax rounded to next multiple of 5
    tick_max  = math.ceil(vmax / 5) * 5           # e.g. 33 → 35
    tick_vals = list(range(0, tick_max + 1, 5))   # [0, 5, 10, 15, 20, 25, 30, 35]

    # Use tick_max as the norm ceiling so 33–35 maps to dark blue, not white
    norm = mcolors.Normalize(vmin=0, vmax=tick_max)
    sm   = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Count', fontsize=11, fontweight='bold', labelpad=10)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels([str(v) for v in tick_vals])

    # ── Save PNG ──────────────────────────────────────────────────────────────
    png_path = output_dir / f'classification_confusion_matrices_{sampling_label}.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved confusion matrix plot: {png_path.name}")

    # ── Save companion JSON ───────────────────────────────────────────────────
    json_path = output_dir / f'confusion_matrix_summary_{sampling_label}.json'
    with open(json_path, 'w') as f:
        json.dump({
            'sampling_method': sampling_label,
            'test_set_size':   int(len(y_test)),
            'class_balance': {
                'normal': int((y_test == 0).sum()),
                'tumor':  int((y_test == 1).sum())
            },
            'models': summary
        }, f, indent=2)
    logger.info(f"✓ Saved confusion matrix JSON: {json_path.name}")

    return png_path, json_path, summary


def plot_confusion_matrices_stacking(
    y_test,
    stacking_records,
    X_test_scaled,
    output_dir,
    sampling_label,
    logger,
):
    """
    Generate a 2×3 panel of annotated confusion matrices — one per stacking
    meta-learner (up to 5), with the 6th cell left blank — plus a shared
    colorbar.  Saves PNG + companion JSON to stacking_sensitivity/.

    Layout (row-major, 2 rows × 3 cols):
        LR_L2        | LR_L1        | LR_EN
        SVC_RBF      | LightGBM     | [empty]

    Each cell mirrors the style of plot_confusion_matrices:
        - Raw count (large, bold)
        - Row-normalised % in parentheses
        - Cell label (True Negative / False Positive / etc.)
        - Blues cmap intensity tied to count (shared colorbar)
        - White text on dark cells, dark text on pale cells

    Args:
        y_test           : array-like, true labels (0=normal, 1=tumor)
        stacking_records : list of per-meta-learner dicts from
                           run_stacking_sensitivity_test()
        X_test_scaled    : np.ndarray — scaled held-out features
                           (used to re-predict so we have y_pred per learner)
        output_dir       : pathlib.Path — method directory (e.g. .../median/)
        sampling_label   : str, e.g. 'cluster_based'
        logger           : logging.Logger

    Returns:
        png_path  : pathlib.Path to saved PNG
        json_path : pathlib.Path to saved JSON
        summary   : dict of per-meta-learner metrics
    """
    import json
    import math
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    from matplotlib.patches import Rectangle
    from sklearn.metrics import confusion_matrix, matthews_corrcoef

    # ── Filter to successful records only ────────────────────────────────────
    ok_records = [r for r in stacking_records if r.get('status') == 'ok']
    if not ok_records:
        logger.warning("plot_confusion_matrices_stacking: no successful records – skipping")
        return None, None, {}

    # ── We need y_pred per meta-learner.  The stacking sweep already stored
    #    test_acc / test_auc but not the raw predictions.  Re-derive them
    #    from the stored stacking_clf objects if available; otherwise skip
    #    gracefully.  However, the sweep does NOT return fitted clf objects.
    #
    #    Practical approach: we rely on the fact that each record stores
    #    test_acc (accuracy).  To get a confusion matrix we need y_pred.
    #    We refit each meta-learner on X_train / y_train here — but those
    #    are not passed into this function.
    #
    #    Instead we retrieve the predictions from the JSON that was saved
    #    alongside the sweep.  But that JSON also doesn't store y_pred.
    #
    #    ➜  The cleanest solution: accept an optional predictions_dict kwarg.
    #       When it is absent we skip and log a warning.  The caller
    #       (run_stacking_sensitivity_test) will pass it.
    #
    #    Here we expect predictions_dict to be injected via the
    #    `_stacking_predictions` attribute on each record (set by caller).
    # ─────────────────────────────────────────────────────────────────────────

    # Build ordered list: up to 5 learners in canonical order
    canonical_order = ['LR_L2', 'LR_L1', 'LR_EN', 'SVC_RBF', 'LightGBM']
    key_to_rec = {r['key']: r for r in ok_records}

    ordered = []
    for key in canonical_order:
        if key in key_to_rec:
            ordered.append(key_to_rec[key])
    # Append any unexpected keys not in canonical list
    for r in ok_records:
        if r['key'] not in canonical_order:
            ordered.append(r)

    n_learners = len(ordered)

    # Check we have y_pred stored in records
    missing_pred = [r['key'] for r in ordered if '_y_pred' not in r]
    if missing_pred:
        logger.warning(
            f"plot_confusion_matrices_stacking: y_pred missing for "
            f"{missing_pred}. Call this function from inside "
            f"run_stacking_sensitivity_test after storing r['_y_pred']. Skipping."
        )
        return None, None, {}

    # ── Collect all counts for shared colorbar range ──────────────────────────
    all_counts = []
    cms = {}
    for rec in ordered:
        y_pred = rec['_y_pred']
        cm_arr = confusion_matrix(y_test, y_pred)
        cms[rec['key']] = cm_arr
        all_counts += cm_arr.ravel().tolist()

    vmin = 0
    vmax = max(all_counts)
    cmap = plt.cm.Blues

    # ── Figure + GridSpec  (2 rows × 3 cols + narrow colorbar column) ─────────
    # Positions: [0,0] [0,1] [0,2] [1,0] [1,1]  →  5 matrices
    #            [1,2] → empty cell
    #            [:, 3] → colorbar
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(
        2, 4,
        width_ratios=[1, 1, 1, 0.045],
        hspace=0.52,
        wspace=0.38,
        left=0.06, right=0.95,
        top=0.90,  bottom=0.07,
    )

    label_str = sampling_label.replace('_', ' ').title()
    fig.suptitle(
        f'Confusion Matrices — Stacking Meta-Learner Sensitivity\n'
        f'({label_str} Sampling  |  Test Set n={len(y_test)}  |  '
        f'{int((y_test == 0).sum())} Normal + {int((y_test == 1).sum())} Tumor)',
        fontsize=15, fontweight='bold', y=0.97,
    )

    # Grid positions for up to 6 cells (row, col)
    cell_positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    summary = {}

    for idx, rec in enumerate(ordered):
        ri, ci = cell_positions[idx]
        ax = fig.add_subplot(gs[ri, ci])

        key     = rec['key']
        name    = rec.get('display_name', key)
        y_pred  = rec['_y_pred']
        cm_arr  = cms[key]
        TN, FP, FN, TP = cm_arr.ravel()
        total   = int(TN + FP + FN + TP)

        # ── Derived metrics ───────────────────────────────────────────────
        sensitivity = TP / (TP + FN) if (TP + FN) else 0.0
        specificity = TN / (TN + FP) if (TN + FP) else 0.0
        ppv         = TP / (TP + FP) if (TP + FP) else 0.0
        npv         = TN / (TN + FN) if (TN + FN) else 0.0
        mcc_val     = matthews_corrcoef(y_test, y_pred)
        accuracy    = (TP + TN) / total

        summary[key] = {
            'display_name': name,
            'TP':           int(TP),
            'TN':           int(TN),
            'FP':           int(FP),
            'FN':           int(FN),
            'sensitivity':  round(float(sensitivity), 4),
            'specificity':  round(float(specificity), 4),
            'ppv':          round(float(ppv),         4),
            'npv':          round(float(npv),         4),
            'mcc':          round(float(mcc_val),     4),
            'accuracy':     round(float(accuracy),    4),
            'total_errors': int(FP + FN),
        }

        # Row totals for row-normalised %
        n_row = int(TN + FP)
        t_row = int(FN + TP)

        cells = [
            [(TN, TN / n_row * 100, True),  (FP, FP / n_row * 100, False)],
            [(FN, FN / t_row * 100, False),  (TP, TP / t_row * 100, True)],
        ]

        # ── Draw cells ────────────────────────────────────────────────────
        tick_max  = math.ceil(vmax / 5) * 5
        norm_cb   = mcolors.Normalize(vmin=0, vmax=tick_max)

        for r in range(2):
            for c in range(2):
                count, pct, diagonal = cells[r][c]
                norm_val = count / tick_max if tick_max > 0 else 0.5
                rgba     = cmap(norm_val)
                ax.add_patch(Rectangle((c, 1 - r), 1, 1, color=rgba))

                light     = norm_val < 0.45
                txt_main  = '#1a1a2e' if light else 'white'
                txt_sub   = '#555555' if light else '#dddddd'
                txt_label = '#aaaaaa' if light else '#cccccc'

                ax.text(
                    c + 0.5, 1 - r + 0.62, str(count),
                    ha='center', va='center',
                    fontsize=28, fontweight='bold', color=txt_main,
                )
                ax.text(
                    c + 0.5, 1 - r + 0.38, f'({pct:.1f}%)',
                    ha='center', va='center',
                    fontsize=11, color=txt_sub,
                )
                cell_labels = {
                    (0, 0): 'True Negative',  (0, 1): 'False Positive',
                    (1, 0): 'False Negative', (1, 1): 'True Positive',
                }
                ax.text(
                    c + 0.5, 1 - r + 0.16, cell_labels[(r, c)],
                    ha='center', va='center',
                    fontsize=8.5, color=txt_label, style='italic',
                )

        # ── Axes cosmetics ────────────────────────────────────────────────
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(
            ['Predicted\nNormal', 'Predicted\nTumor'],
            fontsize=10, fontweight='bold',
        )
        ax.xaxis.set_ticks_position('bottom')
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(
            ['Actual\nTumor', 'Actual\nNormal'],
            fontsize=10, fontweight='bold',
        )
        ax.yaxis.set_ticks_position('left')
        ax.tick_params(length=0, pad=6)
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_title(
            f'{name}\nAccuracy = {accuracy * 100:.2f}%',
            fontsize=11, fontweight='bold', pad=10,
        )

    # ── Leave cell [1,2] empty (turn off axes) ────────────────────────────────
    if n_learners < 6:
        ax_empty = fig.add_subplot(gs[1, 2])
        ax_empty.set_visible(False)

    # ── Shared colorbar ───────────────────────────────────────────────────────
    tick_max  = math.ceil(vmax / 5) * 5
    tick_vals = list(range(0, tick_max + 1, 5))
    norm_cb   = mcolors.Normalize(vmin=0, vmax=tick_max)
    sm        = cm.ScalarMappable(cmap=cmap, norm=norm_cb)
    sm.set_array([])
    cbar_ax   = fig.add_subplot(gs[:, 3])
    cbar      = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Count', fontsize=11, fontweight='bold', labelpad=10)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels([str(v) for v in tick_vals])

    # ── Save PNG ──────────────────────────────────────────────────────────────
    stacking_dir = output_dir / 'stacking_sensitivity'
    stacking_dir.mkdir(parents=True, exist_ok=True)
    png_path = stacking_dir / f'stacking_confusion_matrices_{sampling_label}.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved stacking confusion matrix plot: {png_path.name}")

    # ── Save companion JSON ───────────────────────────────────────────────────
    json_path = stacking_dir / f'stacking_confusion_matrix_summary_{sampling_label}.json'
    with open(json_path, 'w') as f:
        json.dump({
            'sampling_method': sampling_label,
            'test_set_size':   int(len(y_test)),
            'class_balance': {
                'normal': int((y_test == 0).sum()),
                'tumor':  int((y_test == 1).sum()),
            },
            'meta_learners': summary,
        }, f, indent=2)
    logger.info(f"✓ Saved stacking confusion matrix JSON: {json_path.name}")

    return png_path, json_path, summary


def plot_confusion_matrices_weighted(
    y_test,
    combo_results,
    rf,
    xgb,
    X_test_scaled,
    output_dir,
    sampling_label,
    logger,
):
    """
    Generate a 3×3 panel of annotated confusion matrices — one per RF/XGB
    weight combination (9 total) — plus a shared colorbar.
    Saves PNG + companion JSON to weighted_sensitivity/.

    Layout (row-major, 3 rows × 3 cols):
        Pure XGBoost | 1:7  | 1:3
        3:5          | 1:1  | 5:3
        3:1          | 7:1  | Pure RF

    Each cell mirrors the style of plot_confusion_matrices.

    Args:
        y_test        : array-like, true labels (0=normal, 1=tumor)
        combo_results : list of 9 dicts from run_weighted_sensitivity_test()
                        (must contain 'rf_weight', 'xgb_weight',
                         'ratio_description', 'auc', 'acc')
        rf            : fitted RandomForestClassifier
        xgb           : fitted XGBClassifier
        X_test_scaled : np.ndarray — scaled held-out features
        output_dir    : pathlib.Path — method directory
        sampling_label: str, e.g. 'median'
        logger        : logging.Logger

    Returns:
        png_path  : pathlib.Path to saved PNG
        json_path : pathlib.Path to saved JSON
        summary   : dict of per-combo metrics keyed by ratio_description
    """
    import json
    import math
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    from matplotlib.patches import Rectangle
    from sklearn.metrics import confusion_matrix, matthews_corrcoef

    if not combo_results:
        logger.warning("plot_confusion_matrices_weighted: no combo results – skipping")
        return None, None, {}

    # ── Pre-compute base probabilities once ───────────────────────────────────
    rf_proba  = rf.predict_proba(X_test_scaled)[:, 1]
    xgb_proba = xgb.predict_proba(X_test_scaled)[:, 1]

    # ── Collect all counts for shared colorbar range ──────────────────────────
    all_counts = []
    cms        = {}  # key = ratio_description
    y_preds    = {}

    for combo in combo_results:
        rf_w    = combo['rf_weight']
        xgb_w   = combo['xgb_weight']
        label   = combo['ratio_description']
        p_blend = rf_w * rf_proba + xgb_w * xgb_proba
        y_pred  = (p_blend >= 0.5).astype(int)
        cm_arr  = confusion_matrix(y_test, y_pred)
        cms[label]    = cm_arr
        y_preds[label] = y_pred
        all_counts += cm_arr.ravel().tolist()

    vmin = 0
    vmax = max(all_counts)
    cmap = plt.cm.Blues

    # ── Figure + GridSpec  (3 rows × 3 cols + narrow colorbar column) ─────────
    fig = plt.figure(figsize=(22, 20))
    gs = fig.add_gridspec(
        3, 4,
        width_ratios=[1, 1, 1, 0.045],
        hspace=0.52,
        wspace=0.38,
        left=0.06, right=0.95,
        top=0.92,  bottom=0.05,
    )

    label_str = sampling_label.replace('_', ' ').title()
    fig.suptitle(
        f'Confusion Matrices — Weighted Ensemble Sensitivity\n'
        f'({label_str} Sampling  |  Test Set n={len(y_test)}  |  '
        f'{int((y_test == 0).sum())} Normal + {int((y_test == 1).sum())} Tumor)',
        fontsize=15, fontweight='bold', y=0.96,
    )

    # 9 cell positions in row-major order
    cell_positions = [
        (0, 0), (0, 1), (0, 2),
        (1, 0), (1, 1), (1, 2),
        (2, 0), (2, 1), (2, 2),
    ]

    summary = {}

    tick_max  = math.ceil(vmax / 5) * 5

    for idx, combo in enumerate(combo_results):
        ri, ci  = cell_positions[idx]
        ax      = fig.add_subplot(gs[ri, ci])

        rf_w    = combo['rf_weight']
        xgb_w   = combo['xgb_weight']
        ratio   = combo['ratio_description']
        y_pred  = y_preds[ratio]
        cm_arr  = cms[ratio]
        TN, FP, FN, TP = cm_arr.ravel()
        total   = int(TN + FP + FN + TP)

        # ── Derived metrics ───────────────────────────────────────────────
        sensitivity = TP / (TP + FN) if (TP + FN) else 0.0
        specificity = TN / (TN + FP) if (TN + FP) else 0.0
        ppv         = TP / (TP + FP) if (TP + FP) else 0.0
        npv         = TN / (TN + FN) if (TN + FN) else 0.0
        mcc_val     = matthews_corrcoef(y_test, y_pred)
        accuracy    = (TP + TN) / total

        summary[ratio] = {
            'rf_weight':    rf_w,
            'xgb_weight':   xgb_w,
            'TP':           int(TP),
            'TN':           int(TN),
            'FP':           int(FP),
            'FN':           int(FN),
            'sensitivity':  round(float(sensitivity), 4),
            'specificity':  round(float(specificity), 4),
            'ppv':          round(float(ppv),         4),
            'npv':          round(float(npv),         4),
            'mcc':          round(float(mcc_val),     4),
            'accuracy':     round(float(accuracy),    4),
            'total_errors': int(FP + FN),
        }

        # Row totals for row-normalised %
        n_row = int(TN + FP)
        t_row = int(FN + TP)

        cells = [
            [(TN, TN / n_row * 100, True),  (FP, FP / n_row * 100, False)],
            [(FN, FN / t_row * 100, False),  (TP, TP / t_row * 100, True)],
        ]

        # ── Draw cells ────────────────────────────────────────────────────
        for r in range(2):
            for c in range(2):
                count, pct, diagonal = cells[r][c]
                norm_val = count / tick_max if tick_max > 0 else 0.5
                rgba     = cmap(norm_val)
                ax.add_patch(Rectangle((c, 1 - r), 1, 1, color=rgba))

                light     = norm_val < 0.45
                txt_main  = '#1a1a2e' if light else 'white'
                txt_sub   = '#555555' if light else '#dddddd'
                txt_label = '#aaaaaa' if light else '#cccccc'

                ax.text(
                    c + 0.5, 1 - r + 0.62, str(count),
                    ha='center', va='center',
                    fontsize=28, fontweight='bold', color=txt_main,
                )
                ax.text(
                    c + 0.5, 1 - r + 0.38, f'({pct:.1f}%)',
                    ha='center', va='center',
                    fontsize=11, color=txt_sub,
                )
                cell_labels = {
                    (0, 0): 'True Negative',  (0, 1): 'False Positive',
                    (1, 0): 'False Negative', (1, 1): 'True Positive',
                }
                ax.text(
                    c + 0.5, 1 - r + 0.16, cell_labels[(r, c)],
                    ha='center', va='center',
                    fontsize=8.5, color=txt_label, style='italic',
                )

        # ── Axes cosmetics ────────────────────────────────────────────────
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(
            ['Predicted\nNormal', 'Predicted\nTumor'],
            fontsize=9, fontweight='bold',
        )
        ax.xaxis.set_ticks_position('bottom')
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(
            ['Actual\nTumor', 'Actual\nNormal'],
            fontsize=9, fontweight='bold',
        )
        ax.yaxis.set_ticks_position('left')
        ax.tick_params(length=0, pad=6)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Title: ratio label + RF:XGB weights + accuracy
        ax.set_title(
            f'{ratio}\nRF={rf_w:.3f} | XGB={xgb_w:.3f}\nAccuracy = {accuracy * 100:.2f}%',
            fontsize=9, fontweight='bold', pad=8,
        )

    # ── Shared colorbar ───────────────────────────────────────────────────────
    tick_vals = list(range(0, tick_max + 1, 5))
    norm_cb   = mcolors.Normalize(vmin=0, vmax=tick_max)
    sm        = cm.ScalarMappable(cmap=cmap, norm=norm_cb)
    sm.set_array([])
    cbar_ax   = fig.add_subplot(gs[:, 3])
    cbar      = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Count', fontsize=11, fontweight='bold', labelpad=10)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels([str(v) for v in tick_vals])

    # ── Save PNG ──────────────────────────────────────────────────────────────
    weighted_dir = output_dir / 'weighted_sensitivity'
    weighted_dir.mkdir(parents=True, exist_ok=True)
    png_path = weighted_dir / f'weighted_confusion_matrices_{sampling_label}.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved weighted confusion matrix plot: {png_path.name}")

    # ── Save companion JSON ───────────────────────────────────────────────────
    json_path = weighted_dir / f'weighted_confusion_matrix_summary_{sampling_label}.json'
    with open(json_path, 'w') as f:
        json.dump({
            'sampling_method': sampling_label,
            'test_set_size':   int(len(y_test)),
            'class_balance': {
                'normal': int((y_test == 0).sum()),
                'tumor':  int((y_test == 1).sum()),
            },
            'weight_combinations': summary,
        }, f, indent=2)
    logger.info(f"✓ Saved weighted confusion matrix JSON: {json_path.name}")

    return png_path, json_path, summary


def train_and_evaluate_models(X, y, valid_features, annotated_hubs, output_dir, logger):
    """
    Train and evaluate multiple classifiers with regularization.
    """
    logger.info("\n" + "=" * 60)
    logger.info("Training and Evaluating Multiple Classifiers")
    logger.info("=" * 60)
    
    # Scale features (ONLY on training data later)
    logger.info("Preparing features...")
    scaler = StandardScaler()
    
    # Split data with stratification
    logger.info(f"Splitting data into train/test sets ({int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} split)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    
    # Scale features (fit ONLY on training data)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info(f"✓ Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    
    # === TRAIN REGULARIZED MODELS ===
    logger.info("\nTraining Random Forest (with regularization)...")
    rf = RandomForestClassifier(
        n_estimators=200,  # Reduced from 500
        max_depth=10,      # Limited depth
        min_samples_split=5,
        min_samples_leaf=3,
        max_features='sqrt',  # Feature sampling
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    rf_proba = rf.predict_proba(X_test_scaled)[:, 1]
    logger.info("✓ Random Forest trained")
    
    logger.info("Training XGBoost (with regularization)...")
    xgb = XGBClassifier(
        n_estimators=200,  # Reduced from 500
        max_depth=4,       # Reduced from 6
        learning_rate=0.05, # Lower learning rate
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,     # L1 regularization
        reg_lambda=1.0,    # L2 regularization
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric='logloss',
        scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1])  # Handle imbalance
    )
    xgb.fit(
        X_train_scaled, y_train,
        eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
        verbose=False
    )
    xgb_pred = xgb.predict(X_test_scaled)
    xgb_proba = xgb.predict_proba(X_test_scaled)[:, 1]
    logger.info("✓ XGBoost trained")

    # Generate learning curve plot + JSON companion
    plot_xgboost_learning_curve(xgb, output_dir, sampling_label=output_dir.name, logger=logger)
    
    # SOFT VOTING ENSEMBLE
    logger.info("Training RF + XGBoost Ensemble (soft voting)...")
    ensemble_soft = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb)],
        voting='soft'
    )
    ensemble_soft.fit(X_train_scaled, y_train)
    ensemble_soft_pred = ensemble_soft.predict(X_test_scaled)
    ensemble_soft_proba = ensemble_soft.predict_proba(X_test_scaled)[:, 1]
    logger.info("✓ Soft voting ensemble trained")
    
    # HARD VOTING ENSEMBLE
    logger.info("Training RF + XGBoost Ensemble (hard voting)...")
    ensemble_hard = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb)],
        voting='hard'
    )
    ensemble_hard.fit(X_train_scaled, y_train)
    ensemble_hard_pred = ensemble_hard.predict(X_test_scaled)
    ensemble_hard_proba = (rf_proba + xgb_proba) / 2  # Average for AUC calculation
    logger.info("✓ Hard voting ensemble trained")
    
    # === EVALUATE ALL MODELS ===
    results = {
        'RandomForest': {
            'acc': accuracy_score(y_test, rf_pred), 
            'auc': roc_auc_score(y_test, rf_proba), 
            'f1': f1_score(y_test, rf_pred)
        },
        'XGBoost': {
            'acc': accuracy_score(y_test, xgb_pred), 
            'auc': roc_auc_score(y_test, xgb_proba), 
            'f1': f1_score(y_test, xgb_pred)
        },
        'Ensemble_Soft': {
            'acc': accuracy_score(y_test, ensemble_soft_pred), 
            'auc': roc_auc_score(y_test, ensemble_soft_proba), 
            'f1': f1_score(y_test, ensemble_soft_pred)
        },
        'Ensemble_Hard': {
            'acc': accuracy_score(y_test, ensemble_hard_pred), 
            'auc': roc_auc_score(y_test, ensemble_hard_proba), 
            'f1': f1_score(y_test, ensemble_hard_pred)
        }
    }
    
    # Convert results to serializable types
    serializable_results = {}
    for model_name, metrics in results.items():
        serializable_results[model_name] = {
            k: float(v) for k, v in metrics.items()
        }

    # Save performance comparison
    results_df = pd.DataFrame(serializable_results).T.round(4)
    results_df = results_df[['acc', 'auc', 'f1']]
    results_df.to_csv(output_dir / 'classification_performance_comparison.tsv', sep='\t')
    logger.info(f"\nModel Performance Comparison:")
    for model, metrics in results.items():
        logger.info(f"  {model:16s} - Acc: {metrics['acc']:.4f}, AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
    
    # === CROSS-VALIDATION (all models + all ensemble types) ===
    logger.info("\nPerforming 5-fold cross-validation on all models and ensembles...")

    # Prepare full dataset for CV
    X_full_scaled = scaler.fit_transform(X)
    skf = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Build all 4 ensemble variants for CV
    # Note: base estimators (rf, xgb) are already fitted; VotingClassifier/StackingClassifier
    # will refit them internally during cross_val_score as expected.
    ens_cv_soft = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb)], voting='soft')
    ens_cv_hard = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb)], voting='hard')
    ens_cv_weighted = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb)], voting='soft', weights=[2, 1])
    ens_cv_stacking = StackingClassifier(
        estimators=[('rf', rf), ('xgb', xgb)],
        final_estimator=LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced'),
        cv=5, passthrough=False)

    # All 6 models to evaluate: (label, model, scoring_metric)
    # Hard voting: predict_proba is still available (RF + XGBoost both support it),
    # so roc_auc scoring works. Hard voting only affects predict(), not predict_proba().
    _cv_models = [
        ('RandomForest',      rf,              'roc_auc'),
        ('XGBoost',           xgb,             'roc_auc'),
        ('Ensemble_Soft',     ens_cv_soft,     'roc_auc'),
        ('Ensemble_Hard',     ens_cv_hard,     'accuracy'),  # hard vote → majority label → accuracy
        ('Ensemble_Weighted', ens_cv_weighted, 'roc_auc'),
        ('Ensemble_Stacking', ens_cv_stacking, 'roc_auc'),
    ]

    cv_summary = {'cv_folds': CV_FOLDS}
    cv_scores_rf = cv_scores_xgb = None  # preserved for existing return statement

    logger.info(f"\n  {'Model':<22} {'Metric':>9}  {'Train CV Mean':>7}  {'Std (ddof=1)':>13}  {'Duration':>10}")
    logger.info("  " + "-" * 68)

    for _name, _model, _scoring in _cv_models:
        _t0 = time.time()
        _scores = cross_val_score(_model, X_full_scaled, y, cv=skf,
                                  scoring=_scoring, n_jobs=-1)
        _dur = round(time.time() - _t0, 2)
        _mean = float(np.mean(_scores))
        _std  = float(np.std(_scores, ddof=1))
        cv_summary[_name] = {
            'scoring':     _scoring,
            'mean':        round(_mean, 4),
            'std':         round(_std,  4),
            'fold_scores': [round(float(s), 4) for s in _scores],
            'duration_s':  _dur,
        }
        logger.info(f"  {_name:<22} {_scoring:>9}  {_mean:>7.4f}  {_std:>13.4f}  {_dur:>8.2f}s")
        if _name == 'RandomForest':
            cv_scores_rf  = _scores
        elif _name == 'XGBoost':
            cv_scores_xgb = _scores

    # Add explanatory notes per ensemble type
    cv_summary['Ensemble_Hard']['note'] = (
        'Scored on accuracy (not AUC): voting=hard disables predict_proba in sklearn; '
        'holdout AUC (in results dict) uses manual (rf_proba+xgb_proba)/2 averaging')
    cv_summary['Ensemble_Weighted']['note'] = 'RF weight=2, XGBoost weight=1'
    cv_summary['Ensemble_Stacking']['note'] = (
        'Meta-learner: LogisticRegression(balanced), inner cv=5; '
        'duration includes nested CV overhead')

    # Best ensemble by Train CV AUC(all 4 use roc_auc so direct comparison is valid)
    _ens_names = ['Ensemble_Soft', 'Ensemble_Hard', 'Ensemble_Weighted', 'Ensemble_Stacking']
    _best_ens = max(_ens_names, key=lambda k: cv_summary[k]['mean'])
    cv_summary['recommendation'] = {
        'best_ensemble_by_cv_auc': _best_ens,
        'note': 'All four ensembles scored on roc_auc under identical 5-fold stratified Train CV'
    }
    logger.info(f"\n  → Best ensemble by mean Train CV AUC: {_best_ens}")

    cv_json_path = output_dir / 'cv_summary.json'
    with open(cv_json_path, 'w') as f:
        json.dump(cv_summary, f, indent=2)
    logger.info(f"✓ Saved full CV summary (all 6 models, mean ± std, fold scores) to: {cv_json_path.name}")

    # --- Augment serializable_results with CV mean/std for the 4 existing holdout models ---
    for model_key in ['RandomForest', 'XGBoost', 'Ensemble_Soft']:
        serializable_results[model_key]['cv_auc_mean'] = cv_summary[model_key]['mean']
        serializable_results[model_key]['cv_auc_std'] = cv_summary[model_key]['std']

    # Handle Ensemble_Hard separately since it uses accuracy for CV, not AUC
    serializable_results['Ensemble_Hard']['cv_acc_mean'] = cv_summary['Ensemble_Hard']['mean']
    serializable_results['Ensemble_Hard']['cv_acc_std'] = cv_summary['Ensemble_Hard']['std']

    # Re-save TSV with cv columns appended
    results_df_cv = pd.DataFrame(serializable_results).T.round(4)
    results_df_cv = results_df_cv[['acc', 'auc', 'f1', 'cv_auc_mean', 'cv_auc_std']]
    results_df_cv.to_csv(output_dir / 'classification_performance_comparison.tsv', sep='\t')
    logger.info("✓ Updated classification_performance_comparison.tsv with cv_auc_mean / cv_auc_std")
    
    # === FEATURE IMPORTANCE ===
    logger.info("\nExtracting feature importances...")
    feature_importance_dict = {}
    
    rf_importance = dict(zip(valid_features, rf.feature_importances_))
    xgb_importance = dict(zip(valid_features, xgb.feature_importances_))
    
    # Create ensemble mean importance
    for gene in valid_features:
        feature_importance_dict[gene] = (rf_importance[gene] + xgb_importance[gene]) / 2
    
    feature_importance_df = pd.DataFrame({
        'feature': valid_features,
        'importance': [feature_importance_dict[gene] for gene in valid_features]
    }).sort_values('importance', ascending=False)


    # AFTER creating feature_importance_df but BEFORE saving
    # Create cancer_relevance mapping from annotated_hubs
    cancer_mapping = {}
    for hub in annotated_hubs:
        gene_key = hub['gene']
        cancer_mapping[gene_key] = hub['cancer_relevance']
        # Also map by symbol if available
        if '|' in gene_key:
            symbol = gene_key.split('|')[1]
            cancer_mapping[symbol] = hub['cancer_relevance']

    # Add cancer_relevance column
    feature_importance_df['cancer_relevance'] = feature_importance_df['feature'].apply(
        lambda x: cancer_mapping.get(x, 'non_cancer')
    )


    # Save importance files
    feature_importance_df.to_csv(output_dir / 'classification_feature_importance.tsv', sep='\t', index=False)
    pd.DataFrame({'feature': valid_features, 'importance': rf.feature_importances_}).to_csv(
        output_dir / 'feature_importance_rf.tsv', sep='\t', index=False)
    pd.DataFrame({'feature': valid_features, 'importance': xgb.feature_importances_}).to_csv(
        output_dir / 'feature_importance_xgb.tsv', sep='\t', index=False)
    
    logger.info("✓ Saved feature importance files")
    
    # === GENERATE ROC CURVE ===
    logger.info("\nGenerating ROC curve comparison plot...")
    plt.figure(figsize=(12, 8))
    colors = {
        'RandomForest': '#3498db', 
        'XGBoost': '#e74c3c', 
        'Ensemble_Soft': '#2ecc71',
        'Ensemble_Hard': '#9b59b6'
    }
    
    for name, proba in [
        ('RandomForest', rf_proba), 
        ('XGBoost', xgb_proba), 
        ('Ensemble_Soft', ensemble_soft_proba),
        ('Ensemble_Hard', ensemble_hard_proba)
    ]:
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc_score = roc_auc_score(y_test, proba)
        linestyle = '-' if 'Ensemble' in name else '--'
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.4f})", 
                 color=colors[name], linewidth=2.5, linestyle=linestyle)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random (AUC = 0.5)')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves: Tumor vs Normal Classification', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    
    roc_path = output_dir / 'classification_roc_curves.png'
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved ROC curve plot: {roc_path.name}")
    

    # ── NEW: CONFUSION MATRICES ──────────────────────────────────────────────
    logger.info("\nGenerating confusion matrix panel...")
    _predictions = {
        'RandomForest':   rf_pred,
        'XGBoost':        xgb_pred,
        'Ensemble_Soft':  ensemble_soft_pred,
        'Ensemble_Hard':  ensemble_hard_pred,
    }
    _probas = {
        'RandomForest':   rf_proba,
        'XGBoost':        xgb_proba,
        'Ensemble_Soft':  ensemble_soft_proba,
        'Ensemble_Hard':  ensemble_hard_proba,
    }
    cm_png, cm_json, cm_summary = plot_confusion_matrices(
        y_test        = y_test,
        predictions_dict = _predictions,
        probas_dict      = _probas,
        output_dir    = output_dir,
        sampling_label = output_dir.name,   # 'cluster_based' or 'median'
        logger        = logger
    )
    # ── END NEW ──────────────────────────────────────────────────────────────

    # --- GENERIC EVALUATION DATA EXPORT ---
    # Save raw test results for external visualization/reporting
    eval_data = {
        'y_true':  y_test.tolist(),
        'y_proba': ensemble_soft_proba.tolist(),
        'classes': [int(c) for c in ensemble_soft.classes_],
        'confusion_matrices': cm_summary, 
        'cv_auc': {
            k: {
                'mean':       cv_summary[k]['mean'],
                'std':        cv_summary[k]['std'],
                'duration_s': cv_summary[k]['duration_s']
            }
            for k in ['RandomForest', 'XGBoost',
                      'Ensemble_Soft', 'Ensemble_Hard',
                      'Ensemble_Weighted', 'Ensemble_Stacking']
        }
    }
    eval_path = output_dir / 'model_evaluation_data.json'
    with open(eval_path, 'w') as f:
        json.dump(eval_data, f)
    logger.info(f"✓ Saved generic evaluation metadata to {eval_path.name}")

    return serializable_results, feature_importance_df, {
        'rf': rf, 'xgb': xgb, 'ensemble_soft': ensemble_soft, 'ensemble_hard': ensemble_hard,
        'cv_scores_rf': cv_scores_rf.tolist(), 'cv_scores_xgb': cv_scores_xgb.tolist(),
        'X_test': X_test, 'y_test': y_test,
        'X_test_scaled':  X_test_scaled,
        'X_train_scaled': X_train_scaled,
        'X_full_scaled':  X_full_scaled,
        'y_train':        y_train,
        'y':              y,
        'skf':            skf,
    }


def run_stacking_sensitivity_test(
    rf,
    xgb,
    X_train_scaled,
    X_test_scaled,
    X_full_scaled,
    y_train,
    y_test,
    y,
    skf,
    output_dir,
    logger,
) -> dict:
    """
    Sweep all configured stacking meta-learners and record holdout + CV metrics.

    Base models (rf, xgb) are already trained and frozen. Each StackingClassifier
    is built fresh with those base estimators, fitted on the training split, and
    evaluated on the held-out test set. A single outer CV pass per meta-learner
    (inner cv=3 to control runtime) provides generalisation estimates.

    The meta-learners are defined in _build_stacking_meta_learners() and include:
        LR_L2      - LogisticRegression L2 (matches existing Ensemble_Stacking baseline)
        LR_L1      - LogisticRegression L1 / Lasso
        LR_EN      - LogisticRegression ElasticNet
        SVC_RBF    - Support Vector Classifier, RBF kernel
        LightGBM   - LightGBM (optional; skipped if package not installed)

    Parameters
    ----------
    rf             : fitted RandomForestClassifier
    xgb            : fitted XGBClassifier
    X_train_scaled : np.ndarray -- scaled training features
    X_test_scaled  : np.ndarray -- scaled held-out test features
    X_full_scaled  : np.ndarray -- scaled full dataset (for outer CV)
    y_train        : np.ndarray -- training labels
    y_test         : np.ndarray -- held-out test labels
    y              : np.ndarray -- full label array (for outer CV)
    skf            : StratifiedKFold instance (shared with main CV block)
    output_dir     : pathlib.Path -- method directory (e.g. .../median/)
    logger         : logging.Logger

    Returns
    -------
    dict with keys:
        'meta_learners'  : list of per-meta-learner result dicts
        'best_by_test_auc' : key of the meta-learner with highest test AUC
        'best_by_cv_auc'   : key of the meta-learner with highest Train CV AUCmean
        'n_meta_learners'  : int
        'method_note'      : str
    """
    meta_learners = _build_stacking_meta_learners()
    n = len(meta_learners)
    base_estimators = [('rf', rf), ('xgb', xgb)]

    logger.info("\n" + "=" * 60)
    logger.info("STACKING META-LEARNER SENSITIVITY TEST")
    logger.info(f"Testing {n} meta-learner(s): " +
                ", ".join(key for key, _, __ in meta_learners))
    logger.info("=" * 60)

    stacking_records = []

    for i, (key, display_name, factory) in enumerate(meta_learners, 1):
        logger.info(f"\n  [{i}/{n}] {display_name}")
        try:
            stacking_clf = StackingClassifier(
                estimators=base_estimators,
                final_estimator=factory(),
                cv=3,           # inner CV=3: sensitivity sweep, not final model
                passthrough=False,
                n_jobs=-1,
            )

            # Fit on training split
            stacking_clf.fit(X_train_scaled, y_train)

            # Holdout evaluation
            y_pred  = stacking_clf.predict(X_test_scaled)
            y_proba = stacking_clf.predict_proba(X_test_scaled)[:, 1]

            test_auc = float(roc_auc_score(y_test, y_proba))
            test_acc = float(accuracy_score(y_test, y_pred))
            test_f1  = float(f1_score(y_test, y_pred))

            # Outer CV — rebuild classifier with same params to avoid data leakage
            stacking_cv = StackingClassifier(
                estimators=base_estimators,
                final_estimator=factory(),
                cv=3,
                passthrough=False,
                n_jobs=-1,
            )
            _t0 = time.time()
            cv_scores = cross_val_score(
                stacking_cv, X_full_scaled, y,
                cv=skf, scoring='roc_auc', n_jobs=-1,
            )
            cv_dur  = round(time.time() - _t0, 2)
            cv_mean = float(np.mean(cv_scores))
            cv_std  = float(np.std(cv_scores, ddof=1))

            record = {
                'key':          key,
                'display_name': display_name,
                'test_auc':     round(test_auc, 4),
                'test_acc':     round(test_acc, 4),
                'test_f1':      round(test_f1,  4),
                'cv_auc_mean':  round(cv_mean,  4),
                'cv_auc_std':   round(cv_std,   4),
                'cv_fold_scores': [round(float(s), 4) for s in cv_scores],
                'cv_duration_s': cv_dur,
                'status':       'ok',
                '_y_pred':      y_pred.tolist(),   # ← NEW: stored for confusion matrix
            }
            logger.info(f"      Test  AUC: {test_auc:.4f}  Acc: {test_acc:.4f}  F1: {test_f1:.4f}")
            logger.info(f"      Train CV AUC: {cv_mean:.4f} +/- {cv_std:.4f}  ({cv_dur:.1f}s)")

        except Exception as exc:
            logger.warning(f"      FAILED: {exc}")
            record = {
                'key':          key,
                'display_name': display_name,
                'test_auc':     None,
                'test_acc':     None,
                'test_f1':      None,
                'cv_auc_mean':  None,
                'cv_auc_std':   None,
                'cv_fold_scores': [],
                'cv_duration_s': None,
                'status':       f'failed: {exc}',
            }

        stacking_records.append(record)

    # ── Summary ───────────────────────────────────────────────────────────
    ok_records = [r for r in stacking_records if r['status'] == 'ok']

    best_test = max(ok_records, key=lambda r: r['test_auc']) if ok_records else None
    best_cv   = max(ok_records, key=lambda r: r['cv_auc_mean']) if ok_records else None

    logger.info("\n" + "-" * 60)
    if best_test:
        logger.info(f"  Best test AUC -> {best_test['key']}  "
                    f"(AUC={best_test['test_auc']:.4f})")
    if best_cv:
        logger.info(f"  Best Train CV AUC  -> {best_cv['key']}  "
                    f"(Train CV AUC={best_cv['cv_auc_mean']:.4f} +/- {best_cv['cv_auc_std']:.4f})")
    logger.info("=" * 60)

    # ── Save ──────────────────────────────────────────────────────────────
    stacking_dir = output_dir / 'stacking_sensitivity'
    stacking_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        'meta_learners':    stacking_records,
        'best_by_test_auc': best_test['key'] if best_test else None,
        'best_by_cv_auc':   best_cv['key']   if best_cv   else None,
        'n_meta_learners':  len(stacking_records),
        'method_note': (
            'Post-hoc stacking sweep on frozen RF + XGBoost base models. '
            'Inner CV=3 (sensitivity sweep; use cv=5 for final model selection). '
            'LR_L2 matches the existing Ensemble_Stacking baseline in cv_summary.json.'
        ),
    }

    json_path = stacking_dir / 'stacking_sensitivity_results.json'
    with open(json_path, 'w') as fh:
        json.dump(payload, fh, indent=2)
    logger.info(f"  Saved stacking sensitivity results: {json_path.name}")

    tsv_path = stacking_dir / 'stacking_sensitivity_results.tsv'
    tsv_cols = ['key', 'display_name', 'test_auc', 'test_acc', 'test_f1',
                'cv_auc_mean', 'cv_auc_std', 'cv_duration_s', 'status']
    pd.DataFrame(stacking_records)[tsv_cols].to_csv(tsv_path, sep='\t', index=False)
    logger.info(f"  Saved stacking sensitivity table:   {tsv_path.name}")

    _plot_stacking_sensitivity(stacking_records, stacking_dir, logger)

    # ── NEW: Confusion matrix panel for all meta-learners ─────────────────────
    import numpy as _np_stk
    for rec in stacking_records:
        if rec.get('status') == 'ok' and isinstance(rec.get('_y_pred'), list):
            rec['_y_pred'] = _np_stk.array(rec['_y_pred'])

    plot_confusion_matrices_stacking(
        y_test           = y_test,
        stacking_records = stacking_records,
        X_test_scaled    = X_test_scaled,
        output_dir       = output_dir,
        sampling_label   = output_dir.name,
        logger           = logger,
    )
    # ── END NEW ───────────────────────────────────────────────────────────────

    return payload


def _plot_stacking_sensitivity(stacking_records: list, stacking_dir, logger) -> None:
    """
    Grouped bar chart: test AUC and Train CV AUC(with std error bars) per meta-learner.

    Only records with status='ok' are plotted. Failed meta-learners are annotated
    as absent bars so the axis labels remain consistent.
    """
    ok  = [r for r in stacking_records if r['status'] == 'ok']
    all_keys   = [r['key']          for r in stacking_records]
    all_labels = [r['display_name'] for r in stacking_records]

    # Build value arrays aligned to all_keys (None for failed)
    key_to_rec = {r['key']: r for r in ok}
    test_aucs  = [key_to_rec[k]['test_auc']    if k in key_to_rec else 0.0 for k in all_keys]
    cv_means   = [key_to_rec[k]['cv_auc_mean'] if k in key_to_rec else 0.0 for k in all_keys]
    cv_stds    = [key_to_rec[k]['cv_auc_std']  if k in key_to_rec else 0.0 for k in all_keys]
    failed_mask = [k not in key_to_rec for k in all_keys]

    x      = np.arange(len(all_keys))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(all_keys) * 1.8), 6))

    bars_test = ax.bar(x - width / 2, test_aucs, width,
                       label='Test AUC', color='#3498db', alpha=0.85)
    bars_cv   = ax.bar(x + width / 2, cv_means,  width,
                       label='Train CV AUC(mean)', color='#2ecc71', alpha=0.85,
                       yerr=cv_stds, capsize=4, error_kw={'linewidth': 1.2})

    # Annotate failed bars
    for j, failed in enumerate(failed_mask):
        if failed:
            ax.text(x[j], 0.02, 'FAILED', ha='center', va='bottom',
                    fontsize=8, color='#e74c3c', rotation=90)

    # Value labels on bars
    for bar in bars_test:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                    f'{h:.4f}', ha='center', va='bottom', fontsize=7.5)
    for bar in bars_cv:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                    f'{h:.4f}', ha='center', va='bottom', fontsize=7.5)

    all_vals = [v for v in test_aucs + cv_means if v > 0]
    y_min = max(0.5, min(all_vals) - 0.05) if all_vals else 0.5
    y_max = min(1.01, max(all_vals) + 0.04) if all_vals else 1.01
    ax.set_ylim([y_min, y_max])

    ax.set_xlabel('Meta-Learner', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('Stacking Meta-Learner Sensitivity: Test AUC vs Train CV AUC',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=20, ha='right', fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plot_path = stacking_dir / 'stacking_sensitivity_plot.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved stacking sensitivity plot:    {plot_path.name}")


def run_weighted_sensitivity_test(
    rf,
    xgb,
    X_test_scaled,
    y_test,
    output_dir,
    logger,
) -> dict:
    """
    Sweep all 9 RF/XGBoost weight combinations and record holdout metrics.

    This is a *post-hoc* sensitivity test: the base models (rf, xgb) are
    already trained and frozen. Each combination is evaluated by blending
    the stored predict_proba outputs — no retraining occurs, so the sweep
    is fast and the comparison is apples-to-apples.

    Weights are applied as a convex combination of the two probability vectors:
        p_ensemble = rf_weight * p_rf  +  xgb_weight * p_xgb

    This is mathematically equivalent to VotingClassifier(voting='soft',
    weights=[rf_weight, xgb_weight]) on the held-out set.

    Parameters
    ----------
    rf            : fitted RandomForestClassifier
    xgb           : fitted XGBClassifier
    X_test_scaled : np.ndarray -- already-scaled held-out feature matrix
    y_test        : np.ndarray -- held-out labels
    output_dir    : pathlib.Path -- method directory (e.g. .../median/)
    logger        : logging.Logger

    Returns
    -------
    dict with keys:
        'combinations'    : list of per-combination result dicts
        'best_by_auc'     : weight name of the combination with highest AUC
        'best_by_f1'      : weight name of the combination with highest F1
        'n_combinations'  : int
        'method_note'     : str
    """
    logger.info("\n" + "=" * 60)
    logger.info("WEIGHTED ENSEMBLE SENSITIVITY TEST")
    logger.info(f"Testing {len(WEIGHT_COMBINATIONS)} weight combinations")
    logger.info("=" * 60)

    # ── Print configuration table ──────────────────────────────────────────
    logger.info(f"\n  {'#':<4} {'Weight name':<25} {'RF':>6} {'XGB':>6}  Ratio")
    logger.info("  " + "-" * 55)
    for i, combo in enumerate(WEIGHT_COMBINATIONS, 1):
        logger.info(
            f"  [{i}/9]  {combo['name']:<25} "
            f"{combo['rf']:>6.4f} {combo['xgb']:>6.4f}  {combo['ratio_description']}"
        )
    logger.info("")

    # ── Pre-compute base probabilities once ───────────────────────────────
    rf_proba  = rf.predict_proba(X_test_scaled)[:, 1]
    xgb_proba = xgb.predict_proba(X_test_scaled)[:, 1]

    # ── Sweep ─────────────────────────────────────────────────────────────
    combo_results = []

    for i, combo in enumerate(WEIGHT_COMBINATIONS, 1):
        rf_w, xgb_w = combo['rf'], combo['xgb']
        name        = combo['name']

        p_blend = rf_w * rf_proba + xgb_w * xgb_proba
        y_pred  = (p_blend >= 0.5).astype(int)

        auc = float(roc_auc_score(y_test, p_blend))
        acc = float(accuracy_score(y_test, y_pred))
        f1  = float(f1_score(y_test, y_pred))

        entry = {
            'rank':              i,
            'name':              name,
            'rf_weight':         rf_w,
            'xgb_weight':        xgb_w,
            'ratio_description': combo['ratio_description'],
            'auc':               round(auc, 4),
            'acc':               round(acc, 4),
            'f1':                round(f1,  4),
        }
        combo_results.append(entry)

        logger.info(
            f"  [{i}/9] {name}  "
            f"Ratio: {combo['ratio_description']:<20} | "
            f"AUC: {auc:.4f}  Acc: {acc:.4f}  F1: {f1:.4f}"
        )

    # ── Summary ───────────────────────────────────────────────────────────
    best_auc_entry = max(combo_results, key=lambda r: r['auc'])
    best_f1_entry  = max(combo_results, key=lambda r: r['f1'])

    logger.info("\n" + "-" * 60)
    logger.info(f"  Best AUC -> {best_auc_entry['name']}  "
                f"(RF={best_auc_entry['rf_weight']:.4f}  "
                f"AUC={best_auc_entry['auc']:.4f})")
    logger.info(f"  Best F1  -> {best_f1_entry['name']}  "
                f"(RF={best_f1_entry['rf_weight']:.4f}  "
                f"F1={best_f1_entry['f1']:.4f})")
    logger.info("=" * 60)

    # ── Save ──────────────────────────────────────────────────────────────
    weighted_dir = output_dir / 'weighted_sensitivity'
    weighted_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        'combinations':   combo_results,
        'best_by_auc':    best_auc_entry['name'],
        'best_by_f1':     best_f1_entry['name'],
        'n_combinations': len(combo_results),
        'method_note': (
            'Post-hoc blend of frozen RF and XGBoost predict_proba outputs. '
            'rf=0.0 == pure XGBoost, rf=1.0 == pure RandomForest, '
            'rf=0.5 == equal soft vote (same as Ensemble_Soft).' # All scores are on held-out test set; Train CV not applicable for post-hoc weight sweep.'
        ),
    }

    json_path = weighted_dir / 'weighted_sensitivity_results.json'
    with open(json_path, 'w') as fh:
        json.dump(payload, fh, indent=2)
    logger.info(f"  Saved weighted sensitivity results: {json_path.name}")

    tsv_path = weighted_dir / 'weighted_sensitivity_results.tsv'
    pd.DataFrame(combo_results).to_csv(tsv_path, sep='\t', index=False)
    logger.info(f"  Saved weighted sensitivity table:   {tsv_path.name}")

    _plot_sensitivity_curve(combo_results, weighted_dir, logger)

    # ── NEW: Confusion matrix panel for all weight combinations ───────────────
    plot_confusion_matrices_weighted(
        y_test         = y_test,
        combo_results  = combo_results,
        rf             = rf,
        xgb            = xgb,
        X_test_scaled  = X_test_scaled,
        output_dir     = output_dir,
        sampling_label = output_dir.name,
        logger         = logger,
    )
    # ── END NEW ───────────────────────────────────────────────────────────────

    return payload


def _plot_sensitivity_curve(combo_results: list, weighted_dir, logger) -> None:
    """
    Line plot of AUC, Accuracy, and F1 across the 9 RF weight values.

    Vertical reference lines mark the three degenerate cases:
    pure XGBoost (rf=0), equal soft vote (rf=0.5), pure RF (rf=1).
    Y-axis limits are set dynamically from actual score range.
    """
    rf_weights = [r['rf_weight'] for r in combo_results]
    aucs       = [r['auc']       for r in combo_results]
    accs       = [r['acc']       for r in combo_results]
    f1s        = [r['f1']        for r in combo_results]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(rf_weights, aucs, marker='o', linewidth=2.0, color='#2ecc71', label='AUC')
    ax.plot(rf_weights, accs, marker='s', linewidth=2.0, color='#3498db', label='Accuracy')
    ax.plot(rf_weights, f1s,  marker='^', linewidth=2.0, color='#e74c3c', label='F1')

    ref_lines = {
        0.000: ('Pure XGBoost', '#e67e22', '--'),
        0.500: ('Equal (Soft)',  '#9b59b6', ':'),
        1.000: ('Pure RF',       '#1abc9c', '--'),
    }
    for xval, (label, color, ls) in ref_lines.items():
        ax.axvline(x=xval, color=color, linestyle=ls, linewidth=1.2,
                   alpha=0.7, label=label)

    ax.set_xlabel('RF Weight  (XGB weight = 1 - RF weight)', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Weighted Ensemble Sensitivity: RF vs XGBoost Weight',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(rf_weights)
    ax.set_xticklabels([f'{w:.3f}' for w in rf_weights], rotation=45, ha='right')

    all_scores = aucs + accs + f1s
    y_min = max(0.5,  min(all_scores) - 0.05)
    y_max = min(1.01, max(all_scores) + 0.02)
    ax.set_ylim([y_min, y_max])

    ax.legend(fontsize=9, loc='lower center', ncol=3)
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plot_path = weighted_dir / 'weighted_sensitivity_curve.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved weighted sensitivity plot:    {plot_path.name}")


def analyze_feature_categories(feature_importance_df, annotated_hubs, logger):
    """
    Analyze feature importance by biological categories.
    """
    logger.info("\n" + "=" * 60)
    logger.info("Analyzing Feature Importance by Category")
    logger.info("=" * 60)
    
    # Create mapping from feature names to gene info
    gene_info = {}
    for hub in annotated_hubs:
        gene_key = hub['gene']
        gene_info[gene_key] = {
            'cancer_relevance': hub['cancer_relevance'],
            'delta_connectivity': hub['delta_connectivity']
        }
    
    # Add biological context to feature importance
    feature_importance_df['cancer_relevance'] = 'unknown'
    feature_importance_df['delta_connectivity'] = 0.0
    
    for idx, row in feature_importance_df.iterrows():
        feat_name = row['feature']
        if feat_name in gene_info:
            gene_data = gene_info[feat_name]
            feature_importance_df.at[idx, 'cancer_relevance'] = gene_data['cancer_relevance']
            feature_importance_df.at[idx, 'delta_connectivity'] = gene_data['delta_connectivity']
    
    # Analyze by cancer relevance category
    category_analysis = {}
    for relevance in ['breast_cancer', 'cancer', 'non_cancer']:
        rel_feats = feature_importance_df[feature_importance_df['cancer_relevance'] == relevance]
        if len(rel_feats) > 0:
            top_rel = rel_feats.nlargest(5, 'importance')
            category_analysis[f'top_{relevance}'] = top_rel[['feature', 'importance', 'delta_connectivity']].to_dict('records')
            logger.info(f"✓ Found {len(rel_feats)} {relevance.replace('_', ' ')} genes")
    
    return category_analysis, feature_importance_df


def generate_plots(feature_importance_df, category_analysis, output_dir, method_name, logger):
    """
    Generate visualization plots.
    """
    logger.info("\n" + "=" * 60)
    logger.info("Generating Visualizations")
    logger.info("=" * 60)
    
    plot_paths = []
    
    # 1. Top 20 Feature Importance Plot
    plt.figure(figsize=(12, 8))
    top_20 = feature_importance_df.nlargest(20, 'importance').sort_values('importance', ascending=True)
    
    colors = {
        'breast_cancer': '#e74c3c',
        'cancer': '#e67e22', 
        'non_cancer': '#3498db',
        'unknown': '#95a5a6'
    }
    
    bar_colors = [colors.get(row['cancer_relevance'], '#95a5a6') for _, row in top_20.iterrows()]
    
    plt.barh(range(len(top_20)), top_20['importance'], color=bar_colors, alpha=0.8)
    
    gene_labels = [g.split('|')[1] if '|' in g else g for g in top_20['feature']]
    plt.yticks(range(len(top_20)), gene_labels)
    
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title(f'Top 20 Predictive Features: {method_name.replace("_", " ").title()} Sampling', fontsize=14)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['breast_cancer'], label='Breast Cancer Associated'),
        Patch(facecolor=colors['cancer'], label='Cancer Associated'),
        Patch(facecolor=colors['non_cancer'], label='Non-Cancer'),
        Patch(facecolor=colors['unknown'], label='Unknown')
    ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plot_path = output_dir / 'feature_importance_colored.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved colored feature importance plot: {plot_path.name}")
    plot_paths.append(plot_path)
    
    # 2. Feature Importance vs Delta Connectivity Scatter
    plt.figure(figsize=(10, 6))
    scatter_df = feature_importance_df[feature_importance_df['delta_connectivity'] != 0]
    
    if len(scatter_df) > 0:
        scatter_colors = [colors.get(rel, '#95a5a6') for rel in scatter_df['cancer_relevance']]
        plt.scatter(scatter_df['delta_connectivity'], scatter_df['importance'], 
                   c=scatter_colors, alpha=0.6, s=60)
        plt.xlabel('Delta Connectivity (Network Disruption from 02_b)', fontsize=12)
        plt.ylabel('Feature Importance (Classification Power)', fontsize=12)
        plt.title(f'Network Rewiring vs. Classification Power: {method_name.replace("_", " ").title()}', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        if len(scatter_df) > 1:
            corr = np.corrcoef(scatter_df['delta_connectivity'], scatter_df['importance'])[0,1]
            plt.annotate(f'Correlation: {corr:.3f}', xy=(0.05, 0.95), 
                        xycoords='axes fraction', fontsize=11,
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        scatter_path = output_dir / 'feature_vs_connectivity_scatter.png'
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved feature-connectivity scatter plot: {scatter_path.name}")
        plot_paths.append(scatter_path)
    
    return plot_paths


def create_sampling_methods_descriptions(output_dir, config, logger):
    """
    Create comprehensive documentation of ENABLED sampling methods.
    """
    descriptions = {
        'metadata': {
            'generated_on': time.strftime('%Y-%m-%d %H:%M:%S'),
            'purpose': 'Documentation of sampling methods used in classification comparison',
            'total_methods': len(SAMPLING_METHODS),
            'enabled_methods': SAMPLING_METHODS,
            'disabled_methods': [m for m, enabled in SAMPLING_METHOD_FLAGS.items() if not enabled],
            'version': '1.0'
        },
        'methods': {}
    }
    
    # Generate descriptions only for ENABLED methods
    for method_name in SAMPLING_METHODS:
        try:
            # Create instance to get description
            method = get_sampling_method_instance(method_name, config, logger)
            descriptions['methods'][method_name] = method.get_description()
        except Exception as e:
            logger.warning(f"Could not get description for {method_name}: {e}")
    
    # Save descriptions
    desc_path = output_dir / 'sampling_methods_descriptions.json'
    with open(desc_path, 'w') as f:
        json.dump(descriptions, f, indent=2)
    
    return desc_path


def run_sampling_method(method_name, config, PROJECT_ROOT, top_feature_genes, annotated_hubs, 
                       tumor_df, normal_df, comparison_dir):
    """
    Run a single sampling method and save results.
    Now uses object-oriented sampling method classes.
    """
    start_time = time.time()
    
    # Create method-specific output directory
    method_dir = comparison_dir / method_name
    ensure_dir(method_dir)
    ensure_dir(method_dir / 'logs')
    
    # Setup logging for this method
    logger = setup_logging(config, method_dir, method_name)
    logger.info("=" * 70)
    logger.info(f" TCGA-BRCA Classification: {method_name.replace('_', ' ').title()} Sampling")
    logger.info("=" * 70)
    
    # Summary statistics for this method
    method_stats = {
        'method': method_name,
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'parameters': {
            'top_n_features': len(top_feature_genes),
            'test_size': TEST_SIZE,
            'random_state': RANDOM_STATE,
            'cv_folds': CV_FOLDS
        }
    }
    
    try:
        # Get sampling method instance
        sampling_method = get_sampling_method_instance(method_name, config, logger)
        
        # Apply sampling method
        X, y, valid_features, method_info = sampling_method.sample(
            tumor_df, normal_df, top_feature_genes
        )
        
        # Update method stats
        method_stats.update(method_info)
        method_stats['n_features_used'] = len(valid_features)
        method_stats['n_samples_total'] = int(len(X))
        method_stats['n_tumor_samples'] = int(np.sum(y == 1))
        method_stats['n_normal_samples'] = int(np.sum(y == 0))
        method_stats['class_balance'] = f"{np.sum(y==1)} tumor, {np.sum(y==0)} normal"
        
        # Train and evaluate models
        results, feature_importance_df, model_info = train_and_evaluate_models(
            X, y, valid_features, annotated_hubs, method_dir, logger
        )

        # === WEIGHTED ENSEMBLE SENSITIVITY TEST ===
        if ENABLE_WEIGHTED_SENSITIVITY:
            weighted_results = run_weighted_sensitivity_test(
                rf=model_info['rf'],
                xgb=model_info['xgb'],
                X_test_scaled=model_info['X_test_scaled'],
                y_test=model_info['y_test'],
                output_dir=method_dir,
                logger=logger,
            )
            method_stats['weighted_sensitivity'] = weighted_results

        # === STACKING META-LEARNER SENSITIVITY TEST ===
        if ENABLE_STACKING_SENSITIVITY:
            stacking_results = run_stacking_sensitivity_test(
                rf=model_info['rf'],
                xgb=model_info['xgb'],
                X_train_scaled=model_info['X_train_scaled'],
                X_test_scaled=model_info['X_test_scaled'],
                X_full_scaled=model_info['X_full_scaled'],
                y_train=model_info['y_train'],
                y_test=model_info['y_test'],
                y=model_info['y'],
                skf=model_info['skf'],
                output_dir=method_dir,
                logger=logger,
            )
            method_stats['stacking_sensitivity'] = stacking_results

        # Perform method-specific analysis
        method_analysis = sampling_method.analyze(results)
        method_stats['method_specific_analysis'] = method_analysis
        
        # Generate method-specific reports
        method_reports = sampling_method.report(method_dir)
        method_stats['method_reports_generated'] = len(method_reports)
        
        # Analyze feature categories
        category_analysis, feature_importance_df = analyze_feature_categories(
            feature_importance_df, annotated_hubs, logger
        )
        
        # Generate plots
        plot_paths = generate_plots(
            feature_importance_df, category_analysis, method_dir, method_name, logger
        )
        
        # Calculate cancer hub predictive power
        cancer_hubs_in_top = len([
            f for f in feature_importance_df.head(10)['feature'] 
            if any(hub['gene'] == f for hub in annotated_hubs 
                  if hub['cancer_relevance'] in ['breast_cancer', 'cancer'])
        ])
        cancer_hub_predictive_power = (cancer_hubs_in_top / 10) * 100
        
        # === Multi-threshold cancer enrichment analysis ===
        logger.info("\nAnalyzing cancer enrichment across multiple thresholds...")
        enrichment_analysis = analyze_cancer_enrichment_by_threshold(
            feature_importance_df, 
            annotated_hubs,
            thresholds=[10, 20, 30, 50]
        )
        
        # Log the results
        logger.info("Cancer gene enrichment by rank:")
        for threshold, stats in enrichment_analysis['by_rank'].items():
            logger.info(f"  {threshold}: {stats['count']}/{stats['total']} = {stats['percentage']:.1f}%")
        
        logger.info("\nCancer gene enrichment by cumulative importance:")
        for threshold, stats in enrichment_analysis['by_cumulative_importance'].items():
            logger.info(f"  {threshold} importance: {stats['cancer_count']}/{stats['features_needed']} features = {stats['cancer_percentage']:.1f}%")
        

        # Calculate correlation between delta_connectivity and importance
        scatter_df = feature_importance_df[feature_importance_df['delta_connectivity'] != 0]
        if len(scatter_df) > 1:
            corr = np.corrcoef(scatter_df['delta_connectivity'], scatter_df['importance'])[0,1]
        else:
            corr = 0.0
        
        # Compile performance metrics
        best_model = max(results.keys(), key=lambda x: results[x]['auc'])
        
        method_stats['performance_metrics'] = {
            'best_model': best_model,
            'best_model_auc': float(results[best_model]['auc']),
            'best_model_accuracy': float(results[best_model]['acc']),
            'best_model_f1': float(results[best_model]['f1']),
            'all_models': results,
            'cv_auc_rf': {
                'mean': float(np.mean(model_info['cv_scores_rf'])),
                'std': float(np.std(model_info['cv_scores_rf'])),
                'per_fold': model_info['cv_scores_rf']
            },
            'cv_auc_xgb': {
                'mean': float(np.mean(model_info['cv_scores_xgb'])),
                'std': float(np.std(model_info['cv_scores_xgb'])),
                'per_fold': model_info['cv_scores_xgb']
            },
            'generalization_gap_rf': float(results['RandomForest']['auc'] - np.mean(model_info['cv_scores_rf'])),
            'generalization_gap_xgb': float(results['XGBoost']['auc'] - np.mean(model_info['cv_scores_xgb']))
        }
        
        method_stats['feature_analysis'] = {
            'top_10_features': convert_to_json_serializable(feature_importance_df.head(10).to_dict('records')),
            'top_100_features': convert_to_json_serializable(feature_importance_df.head(100).to_dict('records')), 
            'category_breakdown': convert_to_json_serializable(category_analysis),
            'cancer_hub_predictive_power': float(cancer_hub_predictive_power),
            'cancer_hubs_in_top_10': int(cancer_hubs_in_top),
            'delta_connectivity_correlation': float(corr),
            'n_features_analyzed': int(len(feature_importance_df)),
            # === multi-threshold analysis ===
            'enrichment_analysis': convert_to_json_serializable(enrichment_analysis)
        }
        
        method_stats['biological_interpretation'] = {
            "summary": f"Classification using {method_name} sampling method",
            "key_insights": [
                f"Best model: {best_model} with AUC={results[best_model]['auc']:.3f}",
                f"Cancer hub predictive power: {cancer_hub_predictive_power:.1f}%",
                f"Correlation (Δ-connectivity vs importance): {corr:.3f}",
                f"Dataset: {np.sum(y==1)} tumor vs {np.sum(y==0)} normal samples"
            ]
        }
        
        # Convert all numpy types to Python types for JSON serialization
        method_stats = convert_to_json_serializable(method_stats)
        
        # Save method-specific info
        method_info_path = method_dir / 'method_info.json'
        with open(method_info_path, 'w') as f:
            json.dump(method_stats, f, indent=2)
        
        logger.info(f"\n✓ Method completed successfully")
        logger.info(f"  Best model: {best_model} (AUC={results[best_model]['auc']:.4f})")
        logger.info(f"  Cancer hub predictive power: {cancer_hub_predictive_power:.1f}%")
        logger.info(f"  Correlation (Δ-connectivity vs importance): {corr:.3f}")
        
        # Log top 5 features
        logger.info(f"\nTop 5 Predictive Features:")
        for i, (_, row) in enumerate(feature_importance_df.head(5).iterrows(), 1):
            gene_display = row['feature'].split('|')[1] if '|' in row['feature'] else row['feature']
            logger.info(f"  {i}. {gene_display}: importance={row['importance']:.4f} [{row['cancer_relevance']}]")
        
        return method_stats
        
    except Exception as e:
        logger.error(f"\n{'='*60}")
        logger.error(f"ERROR: {method_name} method failed")
        logger.error(f"{'='*60}")
        logger.error(f"{e}", exc_info=True)
        raise
    
    finally:
        # Log completion time
        total_time = time.time() - start_time
        logger.info(f"\n✓ {method_name} completed in {total_time:.1f} seconds ({total_time/60:.2f} minutes)")
        logger.info("=" * 70)


def run_comparison_analysis(all_methods_results, comparison_dir, logger):
    """
    Run comprehensive comparison across all sampling methods.
    """
    logger.info("\n" + "=" * 70)
    logger.info(" COMPREHENSIVE SAMPLING METHOD COMPARISON")
    logger.info("=" * 70)
    
    # Create comparison report directory
    report_dir = comparison_dir / 'comparison_report'
    ensure_dir(report_dir)
    
    comparison_results = {
        'metadata': {
            'generated_on': time.strftime('%Y-%m-%d %H:%M:%S'),
            'methods_compared': list(all_methods_results.keys()),
            'methods_enabled': SAMPLING_METHODS,
            'methods_disabled': [m for m in SAMPLING_METHOD_FLAGS if not SAMPLING_METHOD_FLAGS[m]],
            'purpose': 'Comparative analysis of sampling methods for tumor classification'
        },
        'performance_comparison': {},
        'biological_insight_comparison': {},
        'method_recommendations': {}
    }
    
    # Extract performance metrics for comparison
    for method_name, method_stats in all_methods_results.items():
        comparison_results['performance_comparison'][method_name] = {
            'best_auc': method_stats['performance_metrics']['best_model_auc'],
            'best_accuracy': method_stats['performance_metrics']['best_model_accuracy'],
            'cv_auc_rf_mean': method_stats['performance_metrics']['cv_auc_rf']['mean'],
            'cv_auc_rf_std': method_stats['performance_metrics']['cv_auc_rf']['std'],
            'generalization_gap_rf': method_stats['performance_metrics']['generalization_gap_rf'],
            'generalization_gap_xgb': method_stats['performance_metrics']['generalization_gap_xgb'],
            'n_samples': method_stats['n_samples_total'],
            'n_tumor': method_stats['n_tumor_samples'],
            'n_normal': method_stats['n_normal_samples'],
            'class_balance_ratio': method_stats['n_tumor_samples'] / method_stats['n_normal_samples']
        }
        
        comparison_results['biological_insight_comparison'][method_name] = {
            'cancer_hub_predictive_power': method_stats['feature_analysis']['cancer_hub_predictive_power'],
            'delta_connectivity_correlation': method_stats['feature_analysis']['delta_connectivity_correlation'],
            'cancer_hubs_in_top_10': method_stats['feature_analysis']['cancer_hubs_in_top_10'],
            'top_feature_cancer_relevance': [
                feat['cancer_relevance'] for feat in method_stats['feature_analysis']['top_10_features']
            ],
            # === Add enrichment trends ===
            'enrichment_by_rank': {
                k: v['percentage'] 
                for k, v in method_stats['feature_analysis']['enrichment_analysis']['by_rank'].items()
            }
        }
    
    # Generate rankings
    comparison_results['rankings'] = {
        'by_auc': sorted(
            all_methods_results.items(),
            key=lambda x: x[1]['performance_metrics']['best_model_auc'],
            reverse=True
        ),
        'by_generalization': sorted(
            all_methods_results.items(),
            key=lambda x: abs(x[1]['performance_metrics']['generalization_gap_rf']),  # Smaller gap = better
            reverse=False
        ),
        'by_biological_relevance': sorted(
            all_methods_results.items(),
            key=lambda x: x[1]['feature_analysis']['cancer_hub_predictive_power'],
            reverse=True
        )
    }
    
    # Generate insights and recommendations
    insights = []
    recommendations = []
    
    # Insight 1: Performance comparison
    best_by_auc = comparison_results['rankings']['by_auc'][0] if comparison_results['rankings']['by_auc'] else None
    if best_by_auc:
        insights.append(
            f"{best_by_auc[0].replace('_', ' ').title()} achieved highest AUC ({best_by_auc[1]['performance_metrics']['best_model_auc']:.4f})"
        )
    
    # Insight 2: Generalization comparison
    best_by_gen = comparison_results['rankings']['by_generalization'][0] if comparison_results['rankings']['by_generalization'] else None
    if best_by_gen:
        insights.append(
            f"{best_by_gen[0].replace('_', ' ').title()} shows best generalization (gap: {abs(best_by_gen[1]['performance_metrics']['generalization_gap_rf']):.4f})"
        )
    
    # Insight 3: Biological relevance comparison
    best_by_bio = comparison_results['rankings']['by_biological_relevance'][0] if comparison_results['rankings']['by_biological_relevance'] else None
    if best_by_bio:
        insights.append(
            f"{best_by_bio[0].replace('_', ' ').title()} yields most biologically relevant features ({best_by_bio[1]['feature_analysis']['cancer_hub_predictive_power']:.1f}% cancer hubs in top 10)"
        )
    
    # Recommendations based on use case
    if best_by_auc:
        recommendations.append({
            'use_case': 'Maximum performance (benchmarking)',
            'recommended_method': best_by_auc[0],
            'rationale': 'Highest test set performance'
        })
    
    if best_by_gen:
        recommendations.append({
            'use_case': 'Generalizable models (real-world application)',
            'recommended_method': best_by_gen[0],
            'rationale': 'Smallest generalization gap between train and test'
        })
    
    if best_by_bio:
        recommendations.append({
            'use_case': 'Biological insight discovery',
            'recommended_method': best_by_bio[0],
            'rationale': 'Highest proportion of cancer-associated features'
        })
    
    comparison_results['summary_findings'] = {
        'key_insights': insights,
        'method_recommendations': recommendations,
        'overall_assessment': f'All {len(all_methods_results)} enabled methods provide different trade-offs between performance, generalization, and biological relevance.'
    }
    
    # Save comparison results
    comparison_path = report_dir / 'sampling_methods_comparison.json'
    with open(comparison_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    logger.info(f"\n✓ Comparison analysis completed")
    logger.info(f"✓ Results saved to: {comparison_path.relative_to(comparison_dir.parent.parent)}")
    
    # Generate comparison visualization
    generate_comparison_visualization(comparison_results, report_dir, logger)
    
    return comparison_results


def generate_comparison_visualization(comparison_results, report_dir, logger):
    """
    Generate visual comparison of sampling methods.
    """
    logger.info("\nGenerating comparison visualizations...")
    
    methods = list(comparison_results['performance_comparison'].keys())
    if not methods:
        logger.info("No enabled methods to compare")
        return
    
    method_names = [m.replace('_', ' ').title() for m in methods]
    
    # 1. Performance Comparison Bar Chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # AUC comparison
    auc_values = [comparison_results['performance_comparison'][m]['best_auc'] for m in methods]
    axes[0, 0].bar(method_names, auc_values, color=['#3498db', '#2ecc71', '#9b59b6'][:len(methods)])
    axes[0, 0].set_title('Best Model AUC by Sampling Method', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('AUC-ROC', fontsize=10)
    axes[0, 0].set_ylim([0, 1.05])
    for i, v in enumerate(auc_values):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    
    # Generalization gap comparison
    gap_values = [abs(comparison_results['performance_comparison'][m]['generalization_gap_rf']) for m in methods]
    axes[0, 1].bar(method_names, gap_values, color=['#3498db', '#2ecc71', '#9b59b6'][:len(methods)])
    axes[0, 1].set_title('Generalization Gap (|Test AUC - Train CV AUC|)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Gap Size', fontsize=10)
    for i, v in enumerate(gap_values):
        axes[0, 1].text(i, v + 0.001, f'{v:.3f}', ha='center', fontsize=9)
    
    # Biological relevance comparison
    bio_values = [comparison_results['biological_insight_comparison'][m]['cancer_hub_predictive_power'] for m in methods]
    axes[1, 0].bar(method_names, bio_values, color=['#3498db', '#2ecc71', '#9b59b6'][:len(methods)])
    axes[1, 0].set_title('Biological Relevance (% Cancer Hubs in Top 10)', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Percentage', fontsize=10)
    axes[1, 0].set_ylim([0, 100])
    for i, v in enumerate(bio_values):
        axes[1, 0].text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=9)
    
    # Correlation comparison
    corr_values = [comparison_results['biological_insight_comparison'][m]['delta_connectivity_correlation'] for m in methods]
    axes[1, 1].bar(method_names, corr_values, color=['#3498db', '#2ecc71', '#9b59b6'][:len(methods)])
    axes[1, 1].set_title('Δ-Connectivity vs Feature Importance Correlation', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Correlation Coefficient', fontsize=10)
    axes[1, 1].set_ylim([-1, 1])
    for i, v in enumerate(corr_values):
        axes[1, 1].text(i, v + 0.05 if v >= 0 else v - 0.08, f'{v:.3f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    comparison_plot_path = report_dir / 'sampling_methods_comparison.png'
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved comparison visualization: {comparison_plot_path.name}")
    
    # 2. Create summary table
    summary_data = []
    for method in methods:
        perf = comparison_results['performance_comparison'][method]
        bio = comparison_results['biological_insight_comparison'][method]
        
        summary_data.append({
            'Method': method.replace('_', ' ').title(),
            'Best AUC': f"{perf['best_auc']:.3f}",
            'Train CV AUC(RF)': f"{perf['cv_auc_rf_mean']:.3f} ± {perf['cv_auc_rf_std']:.3f}",
            'Generalization Gap': f"{perf['generalization_gap_rf']:.3f}",
            'Cancer Hubs in Top 10': f"{bio['cancer_hubs_in_top_10']}/10",
            'Δ-Connectivity Correlation': f"{bio['delta_connectivity_correlation']:.3f}",
            'Samples (T/N)': f"{perf['n_tumor']}/{perf['n_normal']}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = report_dir / 'sampling_methods_summary.csv'
    summary_df.to_csv(summary_csv_path, index=False)
    
    logger.info(f"✓ Saved summary table: {summary_csv_path.name}")


def main():
    """
    Main function orchestrating the multi-method comparison.
    """
    start_time = time.time()
    config = load_config()
    PROJECT_ROOT = Path(config['paths']['project_root'])
    
    # Create main output directory
    OUTPUT_DIR = get_auto_output_path(__file__, PROJECT_ROOT)
    ensure_dir(OUTPUT_DIR)
    
    # Create sampling comparison directory
    COMPARISON_DIR = OUTPUT_DIR / 'sampling_comparison'
    ensure_dir(COMPARISON_DIR)
    
    # Setup master logger
    master_logger = setup_master_logger(config, COMPARISON_DIR)
    
    master_logger.info("=" * 70)
    master_logger.info(" TCGA-BRCA Multi-Method Classification Comparison")
    master_logger.info("=" * 70)
    
    # Display enabled methods
    master_logger.info(f"\nSampling Methods Configuration:")
    for method, enabled in SAMPLING_METHOD_FLAGS.items():
        status = "✓ ENABLED" if enabled else "⊗ DISABLED"
        master_logger.info(f"  {method.replace('_', ' ').title():25s}: {status}")
    
    master_logger.info(f"\nMethods to compare: {', '.join([m.replace('_', ' ').title() for m in SAMPLING_METHODS])}")
    master_logger.info(f"Output directory: {COMPARISON_DIR.relative_to(PROJECT_ROOT)}")
    
    # Create method descriptions (only for enabled methods)
    desc_path = create_sampling_methods_descriptions(COMPARISON_DIR, config, master_logger)
    master_logger.info(f"\n📚 Created sampling methods documentation: {desc_path.relative_to(PROJECT_ROOT)}")
    
    try:
        # Load feature genes once (shared by all methods)
        top_n_features = config['classification'].get('top_n_features', DEFAULT_TOP_FEATURES)
        top_feature_genes, annotated_hubs, hubs_df = load_top_feature_genes(
            config, PROJECT_ROOT, top_n_features, master_logger
        )
        
        # Load real data (shared by all methods)
        tumor_df, normal_df = load_real_data(config, PROJECT_ROOT, master_logger)
        
        # Dictionary to store all methods results
        all_methods_results = {}
        
        # Run each sampling method (only enabled ones)
        for method_name in SAMPLING_METHODS:
            master_logger.info(f"\n{'='*70}")
            master_logger.info(f" Running {method_name.replace('_', ' ').title()} Sampling Method")
            master_logger.info(f"{'='*70}")
            
            method_stats = run_sampling_method(
                method_name, config, PROJECT_ROOT, top_feature_genes, 
                annotated_hubs, tumor_df, normal_df, COMPARISON_DIR
            )
            
            if method_stats is not None:  # Only if method was enabled
                all_methods_results[method_name] = method_stats
                master_logger.info(f"✓ Completed {method_name} method")
                master_logger.info(f"  Best AUC: {method_stats['performance_metrics']['best_model_auc']:.4f}")
                master_logger.info(f"  Samples: {method_stats['n_tumor_samples']} tumor, {method_stats['n_normal_samples']} normal")
        
        # Run comprehensive comparison analysis
        master_logger.info(f"\n{'='*70}")
        master_logger.info(" Generating Comprehensive Comparison Report")
        master_logger.info(f"{'='*70}")
        
        comparison_results = run_comparison_analysis(
            all_methods_results, COMPARISON_DIR, master_logger
        )
        
        # Run enhanced analytics module
        master_logger.info(f"\n{'='*70}")
        master_logger.info(" Running Enhanced Analytics Module")
        master_logger.info(f"{'='*70}")
        
        try:
            enhancement_results = run_all_enhancements(
                all_methods_results, COMPARISON_DIR, master_logger, annotated_hubs
            )
            master_logger.info("✓ Enhanced analytics completed successfully")
        except Exception as e:
            master_logger.warning(f"⚠ Enhanced analytics failed: {e}")
            master_logger.warning("  Core analysis completed - enhancements are optional")
        
        # Final summary
        total_time = time.time() - start_time
        master_logger.info(f"\n{'='*70}")
        master_logger.info(" COMPARISON COMPLETED SUCCESSFULLY")
        master_logger.info(f"{'='*70}")
        
        # Print key findings
        master_logger.info("\nKEY FINDINGS:")
        if comparison_results.get('summary_findings', {}).get('key_insights'):
            for insight in comparison_results['summary_findings']['key_insights']:
                master_logger.info(f"  • {insight}")
        else:
            master_logger.info("  No comparison insights available")
        
        master_logger.info("\nRECOMMENDATIONS:")
        if comparison_results.get('summary_findings', {}).get('method_recommendations'):
            for rec in comparison_results['summary_findings']['method_recommendations']:
                master_logger.info(f"  • {rec['use_case']}: {rec['recommended_method'].replace('_', ' ').title()}")
                master_logger.info(f"    {rec['rationale']}")
        else:
            master_logger.info("  No recommendations available")
        
        # master_logger.info(f"\nOutput Structure:")
        # master_logger.info(f"  {COMPARISON_DIR.relative_to(PROJECT_ROOT)}/")
        # master_logger.info(f"    ├── sampling_methods_descriptions.json")
        # master_logger.info(f"    ├── master_comparison.log")
        # for method in SAMPLING_METHODS:
        #     if SAMPLING_METHOD_FLAGS.get(method, False):
        #         master_logger.info(f"    ├── {method}/")
        #         master_logger.info(f"    │   ├── logs/")
        #         master_logger.info(f"    │   └── method_info.json")
        # master_logger.info(f"    ├── comparison_report/")
        # master_logger.info(f"    │   ├── sampling_methods_comparison.json")
        # master_logger.info(f"    │   ├── sampling_methods_comparison.png")
        # master_logger.info(f"    │   └── sampling_methods_summary.csv")
        # master_logger.info(f"    └── enhancements/")
        # master_logger.info(f"        ├── enhancements_summary.json")
        # master_logger.info(f"        ├── model_performance_heatmap.png")
        # master_logger.info(f"        └── ensemble_benefits_analysis.png")
        
        master_logger.info(f"\n⏱Total execution time: {total_time:.1f} seconds ({total_time/60:.2f} minutes)")
        master_logger.info("="*70)
        

    except FileNotFoundError as e:
        master_logger.error(f"\n{'='*60}")
        master_logger.error(f"ERROR: Required input files missing")
        master_logger.error(f"{'='*60}")
        master_logger.error(f"{e}")
        master_logger.error(f"\nPlease ensure you have run the following scripts:")
        master_logger.error(f"  1. 00_b_data_preprocess.py (for expression matrices)")
        master_logger.error(f"  2. 02_b_dcea_viz_enrich.py (for annotated hubs)")
        raise
        
    except Exception as e:
        master_logger.error(f"\n{'='*60}")
        master_logger.error(f"ERROR: Comparison analysis failed")
        master_logger.error(f"{'='*60}")
        master_logger.error(f"{e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()