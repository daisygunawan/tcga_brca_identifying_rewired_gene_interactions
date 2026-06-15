"""
utils/classification_enhancements.py

Comprehensive enhancements module for 02_c_sample_classification.py
Adds 10 advanced analytics and visualization features while preserving original code.

Usage:
    from utils.classification_enhancements import run_all_enhancements
    
    # After main comparison analysis:
    enhancement_results = run_all_enhancements(
        all_methods_results, COMPARISON_DIR, logger
    )
"""

import pandas as pd
import numpy as np
import json
import logging
import time
import traceback
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

from utils.file import ensure_dir

# Module constants
RANDOM_STATE = 42

# ============================================================================
# ENHANCEMENT 1: Model Comparison Across Sampling Methods
# ============================================================================

def create_model_across_methods_heatmap(all_methods_results, report_dir, logger):
    """
    Creates a heatmap showing model performance across different sampling methods.
    """
    logger.info("\nCreating model performance heatmap across sampling methods...")
    
    methods = list(all_methods_results.keys())
    method_names = [m.replace('_', ' ').title() for m in methods]
    models = ['RandomForest', 'XGBoost', 'Ensemble_Soft', 'Ensemble_Hard']
    
    # Create performance matrix
    performance_matrix = np.zeros((len(models), len(methods)))
    
    for i, model in enumerate(models):
        for j, method in enumerate(methods):
            perf = all_methods_results[method]['performance_metrics']['all_models'][model]
            performance_matrix[i, j] = perf['auc']
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(performance_matrix, 
                annot=True, 
                fmt='.3f',
                cmap='RdYlGn',
                vmin=0.95, 
                vmax=1.0,
                xticklabels=method_names,
                yticklabels=[m.replace('_', ' ') for m in models],
                cbar_kws={'label': 'AUC-ROC'})
    
    plt.title('Model Performance Across Sampling Methods\n(Higher is Better)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Sampling Method', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.tight_layout()
    
    heatmap_path = report_dir / 'model_performance_heatmap.png'
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved model performance heatmap: {heatmap_path.name}")
    
    return performance_matrix


# ============================================================================
# ENHANCEMENT 2: Ensemble Benefit Analysis
# ============================================================================

def analyze_ensemble_benefits(all_methods_results, report_dir, logger):
    """
    Analyzes when and how much ensembles improve over base models.
    """
    logger.info("\nAnalyzing ensemble benefits across sampling methods...")
    
    ensemble_analysis = {
        'metadata': {
            'generated_on': time.strftime('%Y-%m-%d %H:%M:%S'),
            'purpose': 'Analysis of ensemble learning benefits across sampling methods'
        },
        'methods': {}
    }
    
    for method_name, method_stats in all_methods_results.items():
        perf = method_stats['performance_metrics']
        
        # Best base model AUC
        rf_auc = perf['all_models']['RandomForest']['auc']
        xgb_auc = perf['all_models']['XGBoost']['auc']
        best_base_auc = max(rf_auc, xgb_auc)
        best_base_model = 'RandomForest' if rf_auc >= xgb_auc else 'XGBoost'
        
        # Best ensemble AUC
        soft_auc = perf['all_models']['Ensemble_Soft']['auc']
        hard_auc = perf['all_models']['Ensemble_Hard']['auc']
        best_ensemble_auc = max(soft_auc, hard_auc)
        best_ensemble = 'Ensemble_Soft' if soft_auc >= hard_auc else 'Ensemble_Hard'
        
        # Calculate improvement metrics
        improvement_ratio = best_ensemble_auc / best_base_auc
        value_added = best_ensemble_auc - best_base_auc
        relative_improvement = (value_added / best_base_auc) * 100
        
        ensemble_analysis['methods'][method_name] = {
            'best_base_model': best_base_model,
            'best_base_auc': float(best_base_auc),
            'best_ensemble': best_ensemble,
            'best_ensemble_auc': float(best_ensemble_auc),
            'improvement_ratio': float(improvement_ratio),
            'value_added': float(value_added),
            'relative_improvement_percent': float(relative_improvement),
            'ensemble_benefits': 'Yes' if value_added > 0 else 'No'
        }
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    methods = list(all_methods_results.keys())
    method_names = [m.replace('_', ' ').title() for m in methods]
    ratios = [ensemble_analysis['methods'][m]['improvement_ratio'] for m in methods]
    
    bars1 = axes[0].bar(method_names, ratios, color=['#3498db', '#2ecc71', '#9b59b6'])
    axes[0].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='No Improvement')
    axes[0].set_title('Ensemble Improvement Ratio\n(Ratio > 1 indicates benefit)', 
                     fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Ensemble AUC / Best Base AUC', fontsize=10)
    axes[0].set_ylim([0.995, 1.005])
    axes[0].legend()
    
    for bar, ratio in zip(bars1, ratios):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005, 
                    f'{ratio:.4f}', ha='center', va='bottom', fontsize=9)
    
    values = [ensemble_analysis['methods'][m]['value_added'] for m in methods]
    
    bars2 = axes[1].bar(method_names, values, color=['#3498db', '#2ecc71', '#9b59b6'])
    axes[1].axhline(y=0.0, color='r', linestyle='--', alpha=0.5, label='No Value Added')
    axes[1].set_title('Ensemble Value Added\n(Positive = Improvement)', 
                     fontsize=12, fontweight='bold')
    axes[1].set_ylabel('AUC Improvement', fontsize=10)
    axes[1].set_ylim([-0.01, 0.01])
    axes[1].legend()
    
    for bar, value in zip(bars2, values):
        color = 'green' if value >= 0 else 'red'
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005, 
                    f'{value:.4f}', ha='center', va='bottom', fontsize=9, color=color)
    
    plt.tight_layout()
    ensemble_plot_path = report_dir / 'ensemble_benefits_analysis.png'
    plt.savefig(ensemble_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    analysis_path = report_dir / 'ensemble_benefits_analysis.json'
    with open(analysis_path, 'w') as f:
        json.dump(ensemble_analysis, f, indent=2)
    
    logger.info(f"✓ Saved ensemble benefits analysis: {analysis_path.name}")
    logger.info(f"✓ Saved ensemble benefits visualization: {ensemble_plot_path.name}")
    
    logger.info("\nEnsemble Benefits Summary:")
    for method in methods:
        analysis = ensemble_analysis['methods'][method]
        logger.info(f"  {method.replace('_', ' ').title():20s}: {analysis['ensemble_benefits']} " 
                   f"(Value added: {analysis['value_added']:.4f})")
    
    return ensemble_analysis


# ============================================================================
# ENHANCEMENT 3: Predictive Hub Rankings (Option A & B)
# ============================================================================

def create_predictive_hub_ranking_simple(feature_importance_df, annotated_hubs, 
                                        method_name, output_dir, logger):
    """
    Option A: Simple predictive hub ranking by combining feature importance with DCEA data.
    """
    logger.info(f"Creating simple predictive hub ranking for {method_name}...")
    
    # Create DCEA rank mapping from annotated_hubs
    dcea_rank_map = {}

    for i, hub in enumerate(annotated_hubs):
        gene_key = hub['gene']
        dcea_rank_map[gene_key] = {
            'rank': int(i + 1),  # CONVERT TO INT
            'delta_connectivity': float(hub['delta_connectivity']),  # CONVERT TO FLOAT
            'cancer_relevance': hub['cancer_relevance']
        }
        # Also map by gene symbol
        if '|' in gene_key:
            gene_symbol = gene_key.split('|')[1]
            dcea_rank_map[gene_symbol] = dcea_rank_map[gene_key]
    
    # Build enriched ranking
    ranking = []
    for idx, row in feature_importance_df.iterrows():
        gene = row['feature']
        gene_symbol = gene.split('|')[1] if '|' in gene else gene
        
        hub_entry = {
            'gene': str(gene),  # CONVERT TO STRING
            'gene_symbol': str(gene_symbol),  # CONVERT TO STRING
            'feature_importance': float(row['importance']),  # CONVERT TO FLOAT
            'feature_importance_rank': int(idx + 1)  # CONVERT TO INT
        }
        
        # Add DCEA data if available
        if gene in dcea_rank_map:
            dcea_data = dcea_rank_map[gene]
            hub_entry.update({
                'delta_connectivity': float(dcea_data['delta_connectivity']),
                'dcea_rank': int(dcea_data['rank']),
                'cancer_relevance': dcea_data['cancer_relevance'],
                'has_dcea_data': True
            })
        elif gene_symbol in dcea_rank_map:
            dcea_data = dcea_rank_map[gene_symbol]
            hub_entry.update({
                'delta_connectivity': float(dcea_data['delta_connectivity']),
                'dcea_rank': int(dcea_data['rank']),
                'cancer_relevance': dcea_data['cancer_relevance'],
                'has_dcea_data': True
            })
        else:
            hub_entry.update({
                'delta_connectivity': 0.0,
                'dcea_rank': None,
                'cancer_relevance': 'unknown',
                'has_dcea_data': False
            })
        
        # Calculate composite score
        if hub_entry['has_dcea_data']:
            composite = (hub_entry['feature_importance'] * 1000) + hub_entry['delta_connectivity']
        else:
            composite = hub_entry['feature_importance'] * 1000
        hub_entry['composite_score'] = float(composite)
        
        ranking.append(hub_entry)
    
    # Sort by composite score
    ranking_df = pd.DataFrame(ranking)
    ranking_df = ranking_df.sort_values('composite_score', ascending=False)
    ranking_df['composite_rank'] = range(1, len(ranking_df) + 1)
    ranking_df['composite_rank'] = ranking_df['composite_rank'].astype(int)  # CONVERT TO INT
    
    # Convert to JSON-serializable list
    final_ranking = []
    for _, row in ranking_df.iterrows():
        final_ranking.append({
            'gene': str(row['gene']),
            'gene_symbol': str(row['gene_symbol']),
            'feature_importance': float(row['feature_importance']),
            'feature_importance_rank': int(row['feature_importance_rank']),
            'delta_connectivity': float(row['delta_connectivity']),
            'dcea_rank': int(row['dcea_rank']) if pd.notna(row['dcea_rank']) else None,
            'cancer_relevance': str(row['cancer_relevance']),
            'composite_score': float(row['composite_score']),
            'composite_rank': int(row['composite_rank']),
            'has_dcea_data': bool(row['has_dcea_data'])
        })
    
    # Create metadata
    result = {
        'metadata': {
            'method': method_name,
            'ranking_type': 'predictive_simple',
            'description': 'Combines classification feature importance (from 02_c) with DCEA delta connectivity (from 02_b) to identify genes that are both predictive and network-disrupted.',
            'potential_uses': [
                'Identify novel cancer drivers that are both predictive and network-hubs',
                'Validate network disruption findings with predictive power',
                'Prioritize genes for functional validation studies',
                'Compare with pure DCEA ranking from 02_b'
            ],
            'composite_score_formula': 'composite = (feature_importance × 1000) + delta_connectivity',
            'generated_on': time.strftime('%Y-%m-%d %H:%M:%S'),
            'n_genes': int(len(final_ranking)),
            'n_with_dcea_data': int(sum(1 for item in final_ranking if item['has_dcea_data']))
        },
        'ranking': final_ranking
    }
    
    # Save to file
    output_path = output_dir / 'predictive_hub_ranking.json'
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"✓ Created simple predictive hub ranking: {len(final_ranking)} genes")
    logger.info(f"  Top gene: {final_ranking[0]['gene_symbol']} (composite: {final_ranking[0]['composite_score']:.1f})")
    
    return result


def create_method_specific_ranking(feature_importance_df, method_name, output_dir, logger):
    """
    Option B - Method-specific: Ranking based solely on feature importance.
    """
    logger.info(f"Creating method-specific ranking for {method_name}...")
    
    # Calculate cumulative importance
    total_importance = float(feature_importance_df['importance'].sum())  # CONVERT TO FLOAT
    feature_importance_df['importance_percentage'] = (feature_importance_df['importance'] / total_importance) * 100
    feature_importance_df['cumulative_importance'] = feature_importance_df['importance'].cumsum()
    
    # Create ranking
    ranking = []
    for idx, row in feature_importance_df.iterrows():
        gene = row['feature']
        gene_symbol = gene.split('|')[1] if '|' in gene else gene
        
        ranking.append({
            'gene': str(gene),  # CONVERT TO STRING
            'gene_symbol': str(gene_symbol),  # CONVERT TO STRING
            'importance': float(row['importance']),  # CONVERT TO FLOAT
            'importance_percentage': float(row['importance_percentage']),  # CONVERT TO FLOAT
            'cumulative_importance': float(row['cumulative_importance']),  # CONVERT TO FLOAT
            'rank': int(idx + 1)  # CONVERT TO INT
        })
    
    # Calculate summary statistics
    top_10 = feature_importance_df.head(10)
    top_20 = feature_importance_df.head(20)
    
    # Count cancer genes in top 10 (simplified - would need actual cancer_relevance)
    cancer_count_top_10 = 0  # Would be populated with actual data
    
    result = {
        'metadata': {
            'method': method_name,
            'ranking_type': 'method_specific',
            'description': f'Pure classification-based ranking from {method_name} sampling method. Ranks genes by their importance in distinguishing tumor vs normal samples.',
            'usefulness': [
                'Identify most discriminative features for tumor classification',
                f'Compare feature importance distributions across sampling methods',
                'Understand contribution of individual genes to classification accuracy',
                'Select top features for simplified diagnostic models'
            ],
            'generated_on': time.strftime('%Y-%m-%d %H:%M:%S'),
            'n_genes': int(len(ranking))
        },
        'ranking': ranking,
        'summary_stats': {
            'total_importance': float(total_importance),
            'importance_top_10': float(top_10['importance'].sum()),
            'importance_percentage_top_10': float((top_10['importance'].sum() / total_importance) * 100),
            'importance_top_20': float(top_20['importance'].sum()),
            'importance_percentage_top_20': float((top_20['importance'].sum() / total_importance) * 100),
            'cancer_genes_in_top_10': int(cancer_count_top_10),
            'cancer_percentage_top_10': float((cancer_count_top_10 / 10) * 100) if len(top_10) > 0 else 0.0
        }
    }
    
    # Save to file
    output_path = output_dir / 'method_specific_ranking.json'
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"✓ Created method-specific ranking: {len(ranking)} genes")
    logger.info(f"  Top gene: {ranking[0]['gene_symbol']} ({ranking[0]['importance_percentage']:.1f}% of total)")
    
    return result


def create_consensus_predictive_ranking(all_method_results_dict, output_dir, logger):
    """
    Option B - Consensus: Creates consensus ranking across multiple sampling methods.
    Enhanced with composite scores and feature importances from each method.
    """
    logger.info("Creating consensus predictive ranking across methods...")
    
    methods = list(all_method_results_dict.keys())
    
    # Load composite scores and feature importances from predictive_hub_ranking.json
    logger.info("Loading composite scores and feature importances from method rankings...")
    method_composite_scores = {}
    method_feature_importances = {}
    delta_connectivity_map = {}
    
    files_loaded = []

    cancer_relevance_map = {}
    project_root = output_dir.parent.parent.parent
    annotated_path = project_root / 'output' / '02_b_dcea_viz_enrich' / 'annotated_hubs.json'

    with open(annotated_path, 'r') as f:
        annotated_hubs = json.load(f)

    for hub in annotated_hubs:
        gene_id = hub['gene']
        cancer_relevance_map[gene_id] = hub['cancer_relevance']
        if '|' in gene_id:
            symbol = gene_id.split('|')[1]
            cancer_relevance_map[symbol] = hub['cancer_relevance']

     
    for method_name in methods:
        method_dir = output_dir / method_name
        pred_json_path = method_dir / 'predictive_hub_ranking.json'
        
        if pred_json_path.exists():
            try:
                with open(pred_json_path, 'r') as f:
                    data = json.load(f)
                
                if 'ranking' in data:
                    loaded_genes = 0
                    for item in data['ranking']:
                        gene = item.get('gene', '')
                        cancer_relevance = item.get('cancer_relevance', '')
                        if not gene:
                            continue
                        
                        # Get gene symbol for alternative lookup
                        gene_symbol = item.get('gene_symbol', '')
                        if '|' in gene and not gene_symbol:
                            gene_symbol = gene.split('|')[1]
                        
                        # Initialize dictionaries if needed
                        if gene not in method_composite_scores:
                            method_composite_scores[gene] = {}
                            method_feature_importances[gene] = {}
                        
                        # ALSO store using gene symbol for lookup flexibility
                        if gene_symbol and gene_symbol not in method_composite_scores:
                            method_composite_scores[gene_symbol] = {}
                            method_feature_importances[gene_symbol] = {}
                        
                        # Extract values
                        composite = float(item.get('composite_score', 0.0))
                        feature_importance = float(item.get('feature_importance', 0.0))
                        delta_conn = float(item.get('delta_connectivity', 0.0))
                        
                        # Store using BOTH full gene ID AND gene symbol
                        method_composite_scores[gene][method_name] = composite
                        method_feature_importances[gene][method_name] = feature_importance
                        
                        if gene_symbol:
                            method_composite_scores[gene_symbol][method_name] = composite
                            method_feature_importances[gene_symbol][method_name] = feature_importance
                        
                        # Delta connectivity is the same for all methods, store once for each key
                        if gene not in delta_connectivity_map:
                            delta_connectivity_map[gene] = delta_conn
                        if gene_symbol and gene_symbol not in delta_connectivity_map:
                            delta_connectivity_map[gene_symbol] = delta_conn
                        
                        loaded_genes += 1
                    
                    files_loaded.append(method_name)
                    logger.info(f"  ✓ {method_name}: Loaded {loaded_genes} genes with scores")
                else:
                    logger.warning(f"  {method_name}: No 'ranking' key in JSON file")
                    
            except Exception as e:
                logger.error(f"  ❌ {method_name}: Failed to load JSON: {str(e)}")
                logger.error(f"  Error details: {traceback.format_exc()}")
        else:
            logger.warning(f"  {method_name}: predictive_hub_ranking.json not found at {pred_json_path}")
    
    if not files_loaded:
        logger.warning("⚠️  No score data loaded - consensus will only include rank information")
    else:
        logger.info(f"✓ Loaded score data from {len(files_loaded)} methods: {', '.join(files_loaded)}")
    
    # Collect rankings from each method
    method_rankings = {}
    for method_name, method_stats in all_method_results_dict.items():
        # Extract top genes from method (using feature_analysis data)
        top_genes = []
        
        # Try to get from method_stats structure
        if 'feature_analysis' in method_stats and 'top_100_features' in method_stats['feature_analysis']:
            top_features = method_stats['feature_analysis']['top_100_features']
            top_genes = [item['feature'] for item in top_features if 'feature' in item]
        
        TOP_N_CONSENSUS = 100  # Configurable limit

        # If not found, try alternative approach
        if not top_genes:
            # Load from file
            method_dir = output_dir.parent / method_name
            fi_path = method_dir / 'classification_feature_importance.tsv'
            if fi_path.exists():
                fi_df = pd.read_csv(fi_path, sep='\t')
                top_genes = fi_df.head(TOP_N_CONSENSUS)['feature'].tolist()
        
        method_ranking = {}
        for i, gene in enumerate(top_genes[:TOP_N_CONSENSUS]):
            if gene and isinstance(gene, str):
                method_ranking[str(gene)] = int(i + 1)
        
        method_rankings[method_name] = method_ranking
        logger.info(f"  Collected {len(method_ranking)} gene rankings from {method_name}")
    
    # Calculate consensus scores using rank sum (lower sum = better)
    gene_scores = {}
    for gene in set().union(*[set(r.keys()) for r in method_rankings.values()]):
        ranks = []
        for method in methods:
            if gene in method_rankings[method]:
                ranks.append(int(method_rankings[method][gene]))
            else:
                # Penalty for missing: assign rank = max_rank + 1
                max_rank = len(method_rankings[method])
                ranks.append(int(max_rank + 1))
        
        # Calculate consensus metrics
        gene_scores[gene] = {
            'gene': str(gene),
            'gene_symbol': str(gene.split('|')[1]) if '|' in gene else str(gene),
            'cancer_relevance': cancer_relevance_map.get(gene, 'non_cancer'), 
            'rank_sum': int(sum(ranks)),
            'average_rank': float(np.mean(ranks)),
            'rank_std': float(np.std(ranks)) if len(ranks) > 1 else 0.0,
            'method_ranks': {method: int(method_rankings[method].get(gene)) if gene in method_rankings[method] else None for method in methods},
            'appears_in_methods': [method for method in methods if gene in method_rankings[method]],
            'n_methods': int(sum(1 for method in methods if gene in method_rankings[method]))
        }
    
    # STEP 1: Populate all scores in gene_scores dictionary
    logger.info("Enhancing consensus ranking with composite scores...")
    
    for gene_id, gene_data in gene_scores.items():
        gene = gene_data['gene']
        gene_symbol = gene_data['gene_symbol']
        
        # Initialize score fields with defaults
        gene_data['avg_composite_score'] = 0.0
        gene_data['avg_feature_importance'] = 0.0
        gene_data['delta_connectivity'] = 0.0
        
        # Initialize method-specific fields
        for method in methods:
            gene_data[f'{method}_composite_score'] = 0.0
            gene_data[f'{method}_feature_importance'] = 0.0
        
        # Get composite scores if available - try multiple lookup strategies
        found_scores = False
        comp_scores = []
        fi_scores = []
        
        # Strategy 1: Try full gene ID first
        lookup_key = gene
        if gene in method_composite_scores and gene in method_feature_importances:
            found_scores = True
            lookup_key = gene

        # Strategy 2: Try gene symbol if full ID not found
        elif gene_symbol in method_composite_scores and gene_symbol in method_feature_importances:
            found_scores = True
            lookup_key = gene_symbol
        
        if found_scores:
            for method in methods:
                # Get scores for this method
                comp_score = method_composite_scores[lookup_key].get(method, 0.0)
                fi_score = method_feature_importances[lookup_key].get(method, 0.0)
                
                # Store method-specific scores
                gene_data[f'{method}_composite_score'] = float(comp_score)
                gene_data[f'{method}_feature_importance'] = float(fi_score)
                
                # Only include in averages if we have valid data (> 0)
                if comp_score > 0:
                    comp_scores.append(comp_score)
                if fi_score > 0:
                    fi_scores.append(fi_score)
            
            # Calculate averages (only for methods that have data)
            if comp_scores:
                gene_data['avg_composite_score'] = float(np.mean(comp_scores))
            if fi_scores:
                gene_data['avg_feature_importance'] = float(np.mean(fi_scores))
        
        # Add delta connectivity (same for both methods)
        if gene in delta_connectivity_map:
            gene_data['delta_connectivity'] = float(delta_connectivity_map[gene])
        elif gene_symbol in delta_connectivity_map:
            gene_data['delta_connectivity'] = float(delta_connectivity_map[gene_symbol])
    
    # STEP 2: Now sort with all scores populated
    # sort based on avg_composite_score, BUT on 03a calculation this cant be used because delta connectivity would be doubly embeded
    # 03a gets delta coon from 02b but our composite value combines delta conn and feature importance
    # consensus_list = sorted(
    #     gene_scores.values(),
    #     key=lambda x: (-x['avg_composite_score'], x['rank_std'], x['rank_sum'])  # Descending composite, then consistency
    # )
    # sort based on feature importance to avoid potential conflict with formula in 03a
    consensus_list = sorted(
        gene_scores.values(),
        key=lambda x: (-x['avg_feature_importance'], x['avg_composite_score'], x['rank_std'], x['rank_sum'])  # Descending FI, then consistency
    )

    # Add consensus_rank
    for i, gene_data in enumerate(consensus_list):
        gene_data['consensus_rank'] = int(i + 1)
    
    # Calculate method contributions
    method_contributions = {}
    top_20_consensus = {gene_data['gene'] for gene_data in consensus_list[:20]}
    
    for method in methods:
        top_20_method = set(list(method_rankings[method].keys())[:20])
        overlap = len(top_20_consensus.intersection(top_20_method))
        unique = len(top_20_method - top_20_consensus)
        
        method_contributions[method] = {
            'top_20_overlap_with_consensus': int(overlap),
            'unique_genes_in_top_20': int(unique),
            'overlap_percentage': float((overlap / 20) * 100) if len(top_20_method) > 0 else 0.0
        }
    
    # Calculate consensus quality metrics
    avg_rank_std = 0.0
    if len(consensus_list[:20]) > 0:
        avg_rank_std = float(np.mean([g['rank_std'] for g in consensus_list[:20]]))
    
    method_agreement = 0.0
    if len(consensus_list[:10]) > 0:
        method_agreement = float(len([g for g in consensus_list[:10] if g['n_methods'] == len(methods)]) / 10 * 100)
    
    # Calculate composite score statistics for top genes
    top_avg_composite_scores = []
    
    if consensus_list[:10]:
        for g in consensus_list[:10]:
            if g.get('avg_composite_score', 0) > 0:
                top_avg_composite_scores.append(g['avg_composite_score'])
    
    if top_avg_composite_scores:
        min_comp = min(top_avg_composite_scores)
        max_comp = max(top_avg_composite_scores)
        avg_comp = np.mean(top_avg_composite_scores)
    else:
        min_comp = max_comp = avg_comp = 0.0
    
    result = {
        'metadata': {
            'methods_included': methods,
            'ranking_type': 'consensus_predictive',
            'consensus_method': 'rank_sum_lower_better',
            'description': 'Consensus ranking across multiple sampling methods. Enhanced with composite scores (feature_importance × 1000 + delta_connectivity).',
            'potential_uses': [
                'Identify genes consistently important across methodological variations',
                'Reduce bias from any single sampling method',
                'Prioritize most robust candidates for experimental validation',
                'Compare rank-based consensus with score-based metrics'
            ],
            'scoring_method': 'Genes ranked by sum of ranks across methods (lower sum = better consensus)',
            'composite_score_formula': 'composite_score = (feature_importance × 1000) + delta_connectivity',
            'sort_order': '1) rank_sum (lower=better), 2) rank_std (lower=more consistent), 3) avg_composite_score (higher=better)',
            'generated_on': time.strftime('%Y-%m-%d %H:%M:%S'),
            'n_genes_consensus': int(len(consensus_list)),
            'n_methods_compared': int(len(methods)),
            'top_10_avg_composite_range': f"{min_comp:.1f}-{max_comp:.1f}",
            'top_10_avg_composite': float(avg_comp) if avg_comp else 0.0,
            'score_data_loaded': len(files_loaded) > 0,
            'methods_with_scores': files_loaded
        },
        'method_contributions': method_contributions,
        'consensus_quality_metrics': {
            'average_rank_std': avg_rank_std,
            'method_agreement_top_10': method_agreement,
            'unique_genes_across_methods': int(len(gene_scores))
        },
        'ranking': consensus_list
    }
    
    # Save JSON file
    output_path = output_dir / 'consensus_predictive_ranking.json'
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"✓ Created enhanced consensus ranking across {len(methods)} methods")
    if consensus_list:
        top_gene = consensus_list[0]
        logger.info(f"  Top consensus gene: {top_gene['gene_symbol']}")
        logger.info(f"    Rank sum: {top_gene['rank_sum']}")
        logger.info(f"    Avg composite: {top_gene['avg_composite_score']:.1f}")
        logger.info(f"    Avg feature importance: {top_gene['avg_feature_importance']:.6f}")
        logger.info(f"    Delta connectivity: {top_gene['delta_connectivity']:.1f}")
    
    # Also save as TSV with ordered columns
    logger.info("Creating consensus TSV with ordered columns...")
    tsv_path = output_dir / 'consensus_predictive_ranking.tsv'
    
    # Define column order with averages first
    column_order = [
        'consensus_rank',
        'gene',
        'gene_symbol',
        'avg_composite_score',
        'avg_feature_importance',
        'delta_connectivity',
        'rank_sum',
        'average_rank',
        'rank_std',
    ]
    
    # Add method-specific columns
    for method in methods:
        column_order.extend([f'{method}_composite_score', f'{method}_feature_importance'])
    
    # Add remaining columns
    column_order.extend([
        'n_methods',
        'appears_in_methods'
    ])
    
    # Convert to DataFrame
    tsv_df = pd.DataFrame(consensus_list)
    
    # Only include columns that exist
    existing_columns = [col for col in column_order if col in tsv_df.columns]
    tsv_df = tsv_df[existing_columns]
    
    # Format float columns
    float_cols = [col for col in tsv_df.columns if 'composite' in col or 'importance' in col or 'rank_avg' in col or 'rank_std' in col]
    for col in float_cols:
        if col in tsv_df.columns:
            tsv_df[col] = pd.to_numeric(tsv_df[col], errors='coerce')
    
    tsv_df.to_csv(tsv_path, sep='\t', index=False, float_format='%.6f')
    
    logger.info(f"✓ Saved enhanced consensus TSV: {tsv_path.name}")
    logger.info(f"  Columns: {len(tsv_df.columns)}")
    logger.info(f"  Genes: {len(tsv_df)}")
    
    # Show sample of data
    if len(tsv_df) > 0:
        logger.info("\nSample of consensus data:")
        sample = tsv_df.head(3)
        for _, row in sample.iterrows():
            logger.info(f"  {row['gene_symbol']}: Rank {row['consensus_rank']}, "
                       f"Composite: {row.get('avg_composite_score', 0):.1f}, "
                       f"Rank sum: {row['rank_sum']}")
    
    return result


def create_all_predictive_rankings(all_methods_results, annotated_hubs, comparison_dir, logger):
    """
    Creates all 4 predictive ranking JSON files (Option A, Option B × 3).
    
    FILES GENERATED:
    1. sampling_comparison/median/predictive_hub_ranking.json           (Option A)
    2. sampling_comparison/median/method_specific_ranking.json          (Option B)
    3. sampling_comparison/cluster_based/predictive_hub_ranking.json    (Option A)
    4. sampling_comparison/cluster_based/method_specific_ranking.json   (Option B)
    5. consensus_predictive_ranking.json                                (Option B - Consensus)
    
    RETURNS: Dictionary with paths to all generated files
    """
    logger.info("\n" + "="*70)
    logger.info(" CREATING PREDICTIVE HUB RANKINGS (4 JSON FILES)")
    logger.info("="*70)
    
    start_time = time.time()
    rankings_info = {
        'metadata': {
            'generated_on': time.strftime('%Y-%m-%d %H:%M:%S'),
            'rankings_created': []
        },
        'files': {}
    }
    
    try:
        # Create rankings for each method
        for method_name in ['median', 'cluster_based']:
            if method_name in all_methods_results:
                method_dir = comparison_dir / method_name
                
                # Load feature importance for this method
                fi_path = method_dir / 'classification_feature_importance.tsv'
                if fi_path.exists():
                    fi_df = pd.read_csv(fi_path, sep='\t')
                    
                    # Option A: Simple predictive hub ranking
                    logger.info(f"\n📊 Creating Option A ranking for {method_name}...")
                    option_a_result = create_predictive_hub_ranking_simple(
                        fi_df, annotated_hubs, method_name, method_dir, logger
                    )
                    rankings_info['files'][f'{method_name}_predictive_hub_ranking'] = str(
                        (method_dir / 'predictive_hub_ranking.json').relative_to(comparison_dir.parent.parent)
                    )
                    rankings_info['metadata']['rankings_created'].append(f'{method_name}_option_a')
                    
                    # Option B: Method-specific ranking
                    logger.info(f"📊 Creating Option B (method-specific) ranking for {method_name}...")
                    option_b_result = create_method_specific_ranking(
                        fi_df, method_name, method_dir, logger
                    )
                    rankings_info['files'][f'{method_name}_method_specific_ranking'] = str(
                        (method_dir / 'method_specific_ranking.json').relative_to(comparison_dir.parent.parent)
                    )
                    rankings_info['metadata']['rankings_created'].append(f'{method_name}_option_b')
                else:
                    logger.warning(f"Feature importance file not found for {method_name}: {fi_path}")
        
        # Save summary info
        summary_path = comparison_dir / 'predictive_rankings_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(rankings_info, f, indent=2)
        
        # In create_all_predictive_rankings, after creating JSON files:
        time.sleep(0.1)  # Small delay to ensure files are written

        total_time = time.time() - start_time
        
        logger.info("\n" + "="*70)
        logger.info(" PREDICTIVE RANKINGS CREATED SUCCESSFULLY")
        logger.info("="*70)
        logger.info(f"\n✅ Generated 4+ JSON files in {total_time:.1f} seconds")
        
        logger.info("\n📄 Generated Ranking Files:")
        logger.info(f"  1. Median method - Option A: predictive_hub_ranking.json")
        logger.info(f"  2. Median method - Option B: method_specific_ranking.json")
        logger.info(f"  3. Cluster-based - Option A: predictive_hub_ranking.json")
        logger.info(f"  4. Cluster-based - Option B: method_specific_ranking.json")
        logger.info(f"  5. Consensus across methods: consensus_predictive_ranking.json")
        logger.info(f"  6. Summary: predictive_rankings_summary.json")
        
        return rankings_info
        
    except Exception as e:
        logger.error(f"\n❌ Error creating predictive rankings: {e}")
        logger.error(traceback.format_exc())
        raise

    
def save_ranking_as_tsv(ranking_data, output_path, logger=None):
    """
    Simple function to save ranking data (JSON list) as TSV file.
    
    Args:
        ranking_data: List of dictionaries (JSON array)
        output_path: Path where TSV should be saved
        logger: Optional logger for messages
    """
    if not ranking_data:
        if logger:
            logger.warning(f"⚠️  No data to save for {output_path.name}")
        return False
    
    try:
        df = pd.DataFrame(ranking_data)
        
        # Reorder important columns to front if they exist
        priority_cols = ['gene', 'gene_symbol', 'rank', 'importance', 'composite_score']
        existing_cols = [col for col in priority_cols if col in df.columns]
        other_cols = [col for col in df.columns if col not in existing_cols]
        df = df[existing_cols + other_cols]
        
        df.to_csv(output_path, sep='\t', index=False, float_format='%.6f')
        
        if logger:
            logger.info(f"✓ Created: {output_path.name} ({len(df)} rows)")
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Failed to save {output_path.name}: {str(e)[:100]}")
        return False


def create_consolidated_ranking_table(comparison_dir, annotated_hubs, logger):
    """
    Creates ONE master table consolidating all ranking TSV files into JSON and TSV.
    """
    logger.info("\n" + "="*70)
    logger.info(" CREATING CONSOLIDATED RANKING TABLE")
    logger.info("="*70)
    
    start_time = time.time()
    
    # Define all TSV files to consolidate
    tsv_files = {
        'dcea': comparison_dir / '02_b_annotated_hubs.tsv',
        'median_predictive': comparison_dir / 'median' / 'predictive_hub_ranking.tsv',
        'median_importance': comparison_dir / 'median' / 'method_specific_ranking.tsv',
        'cluster_predictive': comparison_dir / 'cluster_based' / 'predictive_hub_ranking.tsv',
        'cluster_importance': comparison_dir / 'cluster_based' / 'method_specific_ranking.tsv',
        'consensus': comparison_dir / 'consensus_predictive_ranking.tsv'
    }
    
    # Check which files exist
    existing_files = {}
    for key, path in tsv_files.items():
        if path.exists():
            existing_files[key] = path
            logger.info(f"✓ Found: {path.relative_to(comparison_dir.parent.parent)}")
        else:
            logger.warning(f"⚠️  Missing: {path.name}")
    
    if len(existing_files) < 3:
        logger.error("❌ Not enough ranking files to consolidate")
        return None
    
    # Load each TSV into DataFrames
    data_frames = {}
    for key, path in existing_files.items():
        try:
            df = pd.read_csv(path, sep='\t')
            data_frames[key] = df
            logger.info(f"  Loaded {len(df)} rows from {key}")
        except Exception as e:
            logger.error(f"❌ Failed to load {path.name}: {str(e)[:100]}")
    
    # Create gene mapping from annotated_hubs with proper description
    gene_info = {}
    if annotated_hubs:
        for hub in annotated_hubs:
            gene_id = hub.get('gene', '')
            if gene_id:
                gene_symbol = gene_id.split('|')[1] if '|' in gene_id else gene_id
                
                # Try to get description from multiple possible keys
                description = ''
                possible_desc_keys = ['description', 'gene_description', 'function', 'annotation']
                for key in possible_desc_keys:
                    if key in hub and hub[key]:
                        description = hub[key]
                        break
                
                # If no description found, create a basic one
                if not description:
                    description = f"{gene_symbol} gene"
                
                gene_info[gene_id] = {
                    'symbol': gene_symbol,
                    'cancer_tag': hub.get('cancer_relevance', 'unknown'),
                    'delta_conn': hub.get('delta_connectivity', 0.0),
                    'description': description
                }
    
    # Helper function to safely extract values
    def safe_get(df, gene_id, col_name, default=''):
        """Safely get value from DataFrame with error handling."""
        try:
            if 'gene' in df.columns:
                row = df[df['gene'] == gene_id]
            elif 'gene_id' in df.columns:
                row = df[df['gene_id'] == gene_id]
            else:
                return default
            
            if not row.empty and col_name in row.columns:
                value = row.iloc[0][col_name]
                return value if not pd.isna(value) else default
            return default
        except:
            return default
    
    # Build consolidated data
    consolidated = {}
    all_gene_ids = set()
    
    # Collect all unique gene IDs
    for key, df in data_frames.items():
        if 'gene' in df.columns:
            all_gene_ids.update(df['gene'].astype(str).tolist())
        elif 'gene_id' in df.columns:
            all_gene_ids.update(df['gene_id'].astype(str).tolist())
    
    logger.info(f"\nFound {len(all_gene_ids)} unique genes across all rankings")
    
    # Process each gene
    for gene_id in all_gene_ids:
        gene_entry = {
            'gene_id': gene_id,
            'symbol': gene_id.split('|')[1] if '|' in gene_id else gene_id,
            'rankings': {},
            'scores': {},
            'context': {
                'cancer_tag': 'unknown',
                'description': ''
            }
        }
        
        # Extract data from each source
        if 'dcea' in data_frames:
            gene_entry['rankings']['dcea'] = int(safe_get(data_frames['dcea'], gene_id, 'rank', 999))
            gene_entry['scores']['delta_connectivity'] = float(safe_get(data_frames['dcea'], gene_id, 'delta_connectivity', 0.0))
        
        if 'median_importance' in data_frames:
            gene_entry['rankings']['median_classf'] = int(safe_get(data_frames['median_importance'], gene_id, 'rank', 999))
            gene_entry['scores']['median_classf_importance'] = float(safe_get(data_frames['median_importance'], gene_id, 'importance', 0.0))
        
        if 'cluster_importance' in data_frames:
            gene_entry['rankings']['cluster_classf'] = int(safe_get(data_frames['cluster_importance'], gene_id, 'rank', 999))
            gene_entry['scores']['cluster_classf_importance'] = float(safe_get(data_frames['cluster_importance'], gene_id, 'importance', 0.0))
        
        if 'consensus' in data_frames:
            gene_entry['rankings']['consensus'] = int(safe_get(data_frames['consensus'], gene_id, 'consensus_rank', 999))
            gene_entry['scores']['consensus_score'] = float(safe_get(data_frames['consensus'], gene_id, 'rank_sum', 0.0))
        
        # Add gene info from annotated_hubs
        if gene_id in gene_info:
            gene_entry['context']['cancer_tag'] = gene_info[gene_id]['cancer_tag']
            gene_entry['context']['description'] = gene_info[gene_id]['description']
        else:
            # Fallback description
            gene_entry['context']['description'] = f"{gene_entry['symbol']} gene"
        
        consolidated[gene_id] = gene_entry
    
    # Calculate derived metrics
    logger.info("\nCalculating derived metrics and normalized ranking...")
    
    all_rank_totals = []
    all_rank_avgs = []
    gene_rank_data = {}
    
    for gene_id, entry in consolidated.items():
        rankings = entry['rankings']
        
        # Get the 4 main ranks
        main_ranks = []
        for rank_key in ['dcea', 'median_classf', 'cluster_classf', 'consensus']:
            if rank_key in rankings:
                main_ranks.append(rankings[rank_key])
            else:
                main_ranks.append(999)
        
        rank_total = sum(main_ranks)
        rank_avg = sum(main_ranks) / len(main_ranks)
        
        valid_ranks = [r for r in main_ranks if r < 999]
        if len(valid_ranks) > 1:
            rank_std = float(np.std(valid_ranks))
        else:
            rank_std = 0.0
        
        all_rank_totals.append(rank_total)
        all_rank_avgs.append(rank_avg)
        gene_rank_data[gene_id] = {
            'rank_total': rank_total,
            'rank_avg': rank_avg,
            'rank_std': rank_std,
            'main_ranks': main_ranks,
            'dcea_rank': rankings.get('dcea', 999)  # Store dcea_rank for tie-breaking
        }
    
    # Calculate min and max rank_avg before using them
    min_avg = min(all_rank_avgs) if all_rank_avgs else 0
    max_avg = max(all_rank_avgs) if all_rank_avgs else 0
    
    logger.info(f"Rank average range: {min_avg:.2f} (best) to {max_avg:.2f} (worst)")
    
    # Use sequential ranking based on rank_avg with dcea_rank tie-breaking
    if all_rank_avgs:
        # Sort genes by: 1) rank_avg (ascending), 2) dcea_rank (ascending for tie-breaking)
        sorted_genes = sorted(
            gene_rank_data.keys(),
            key=lambda g: (gene_rank_data[g]['rank_avg'], gene_rank_data[g]['dcea_rank'])
        )
        
        # Assign sequential ranks: 1, 2, 3, ...
        current_rank = 1
        last_rank_avg = None
        last_dcea_rank = None
        
        for gene_id in sorted_genes:
            rank_info = gene_rank_data[gene_id]
            rank_avg = rank_info['rank_avg']
            dcea_rank = rank_info['dcea_rank']
            
            # Only increment rank if different from previous gene
            if last_rank_avg is not None and last_dcea_rank is not None:
                if not (abs(rank_avg - last_rank_avg) < 0.0001 and dcea_rank == last_dcea_rank):
                    current_rank += 1
            
            consolidated[gene_id]['rank'] = current_rank
            consolidated[gene_id]['derived'] = {
                'rank_total': int(rank_info['rank_total']),
                'rank_avg': float(rank_avg),
                'rank_std': float(rank_info['rank_std'])
            }
            
            last_rank_avg = rank_avg
            last_dcea_rank = dcea_rank
        
        # Sort genes by sequential rank (ascending)
        sorted_gene_ids = sorted(
            consolidated.keys(),
            key=lambda g: consolidated[g]['rank']
        )
    else:
        sorted_gene_ids = list(consolidated.keys())
    
    # Create sorted consolidated dict
    sorted_consolidated = {gene_id: consolidated[gene_id] for gene_id in sorted_gene_ids}
    
    # Save JSON
    json_path = comparison_dir / 'consolidated_rankings.json'
    with open(json_path, 'w') as f:
        json.dump({
            'metadata': {
                'generated_on': time.strftime('%Y-%m-%d %H:%M:%S'),
                'n_genes': len(sorted_consolidated),
                'files_consolidated': list(existing_files.keys()),
                'rank_avg_range': f"{min_avg:.2f}-{max_avg:.2f}",
                'description': 'Consolidated ranking table sorted by overall rank (ascending)',
                'sort_order': 'ascending (lower rank = better)',
                'ranking_method': 'sequential_by_rank_avg_with_dcea_tiebreak'
            },
            'genes': sorted_consolidated
        }, f, indent=2)
    
    logger.info(f"✓ Saved consolidated JSON: {json_path.name}")
    
    # Create TSV - sorted by rank
    tsv_path = comparison_dir / 'consolidated_rankings.tsv'
    tsv_rows = []
    
    for gene_id in sorted_gene_ids:
        entry = consolidated[gene_id]
        row = {
            'rank': entry.get('rank', 999),
            'gene_id': gene_id,
            'gene_symbol': entry.get('symbol', ''),
            'dcea_rank': entry.get('rankings', {}).get('dcea', ''),
            'median_classf_rank': entry.get('rankings', {}).get('median_classf', ''),
            'cluster_classf_rank': entry.get('rankings', {}).get('cluster_classf', ''),
            'consensus_rank': entry.get('rankings', {}).get('consensus', ''),
            'delta_connectivity': entry.get('scores', {}).get('delta_connectivity', ''),
            'median_classf_importance': entry.get('scores', {}).get('median_classf_importance', ''),
            'cluster_classf_importance': entry.get('scores', {}).get('cluster_classf_importance', ''),
            'rank_total': entry.get('derived', {}).get('rank_total', ''),
            'rank_avg': entry.get('derived', {}).get('rank_avg', ''),
            'rank_std': entry.get('derived', {}).get('rank_std', ''),
            'cancer_relevance_tag': entry.get('context', {}).get('cancer_tag', 'unknown'),
            'description': entry.get('context', {}).get('description', '')
        }
        tsv_rows.append(row)
    
    tsv_df = pd.DataFrame(tsv_rows)
    
    # Column order with rank as first column
    column_order = [
        'rank', 'gene_id', 'gene_symbol',
        'dcea_rank', 'median_classf_rank', 'cluster_classf_rank', 'consensus_rank',
        'delta_connectivity', 'median_classf_importance', 'cluster_classf_importance',
        'rank_total', 'rank_avg', 'rank_std',
        'cancer_relevance_tag', 'description'
    ]
    
    tsv_df = tsv_df[column_order]
    tsv_df.to_csv(tsv_path, sep='\t', index=False, float_format='%.6f')
    
    total_time = time.time() - start_time
    
    logger.info(f"✓ Saved consolidated TSV: {tsv_path.name}")
    logger.info(f"  Total genes: {len(tsv_df)}")
    logger.info(f"  Columns: {len(tsv_df.columns)}")
    logger.info(f"  Sorted by: rank (ascending, lower = better)")
    
    # Show top 5 genes
    logger.info("\n🏆 Top 5 Ranked Genes:")
    for i in range(min(5, len(tsv_df))):
        row = tsv_df.iloc[i]
        logger.info(f"  {i+1}. {row['gene_symbol']} (Rank: {row['rank']}, Avg: {row['rank_avg']:.2f}, Std: {row['rank_std']:.2f})")
    
    logger.info(f"\n✅ Consolidation completed in {total_time:.1f} seconds")
    logger.info("="*70)
    
    return {
        'json_path': str(json_path),
        'tsv_path': str(tsv_path),
        'n_genes': len(sorted_consolidated),
        'files_used': list(existing_files.keys()),
        'rank_avg_range': f"{min_avg:.2f}-{max_avg:.2f}",
        'sort_order': 'ascending_by_rank',
        'ranking_method': 'sequential_by_rank_avg_with_dcea_tiebreak'
    }


# ============================================================================
# INTEGRATION INTO run_all_enhancements()
# ============================================================================

def run_all_enhancements(all_methods_results, comparison_dir, logger, annotated_hubs=None):
    """
    Main function to run all enhancements including new predictive rankings.
    
    This should be called AFTER the original comparison analysis is complete.
    All original outputs are preserved - this adds NEW analyses.
    """
    logger.info("\n" + "="*70)
    logger.info(" RUNNING COMPREHENSIVE ENHANCEMENTS MODULE")
    logger.info("="*70)
    
    start_time = time.time()
    
    # Create enhancements directory
    enhancements_dir = comparison_dir / 'enhancements'
    ensure_dir(enhancements_dir)
    
    enhancement_results = {
        'metadata': {
            'generated_on': time.strftime('%Y-%m-%d %H:%M:%S'),
            'enhancements_completed': []
        }
    }
    
    try:
        # 1. Model comparison across sampling methods
        logger.info("\n📊 Enhancement 1: Model comparison across sampling methods...")
        performance_matrix = create_model_across_methods_heatmap(
            all_methods_results, enhancements_dir, logger
        )
        enhancement_results['model_comparison'] = performance_matrix.tolist()
        enhancement_results['metadata']['enhancements_completed'].append('model_comparison_heatmap')
        
        # 2. Ensemble benefit analysis
        logger.info("\n🤝 Enhancement 2: Ensemble benefit analysis...")
        ensemble_analysis = analyze_ensemble_benefits(
            all_methods_results, enhancements_dir, logger
        )
        enhancement_results['ensemble_analysis'] = ensemble_analysis
        enhancement_results['metadata']['enhancements_completed'].append('ensemble_benefits')
        
        # 3. Create all predictive rankings (4+ JSON files)
        if annotated_hubs:
            logger.info("\n🎯 Enhancement 3: Creating predictive hub rankings (4 JSON files)...")
            rankings_info = create_all_predictive_rankings(
                all_methods_results, annotated_hubs, comparison_dir, logger
            )
            enhancement_results['predictive_rankings'] = rankings_info
            enhancement_results['metadata']['enhancements_completed'].append('predictive_rankings')

            # NEW: Add consensus ranking creation HERE, after individual rankings are created
            logger.info("\n🤝 Creating consensus predictive ranking across methods...")
            consensus_result = create_consensus_predictive_ranking(
                all_methods_results, comparison_dir, logger
            )
            enhancement_results['consensus_ranking'] = consensus_result
            enhancement_results['metadata']['enhancements_completed'].append('consensus_ranking')
        else:
            logger.warning("⚠️  Skipping predictive rankings: annotated_hubs not provided")
        
        # 4. NEW: Convert all ranking JSONs to TSV
        logger.info("\n📊 Enhancement 4: Converting ranking JSONs to TSV format...")
        tsv_files_created = []
        
        for method in ['median', 'cluster_based']:
            method_dir = comparison_dir / method
            
            # Convert predictive_hub_ranking
            pred_json_path = method_dir / 'predictive_hub_ranking.json'
            if pred_json_path.exists():
                with open(pred_json_path) as f:
                    data = json.load(f)
                if 'ranking' in data:
                    tsv_path = method_dir / 'predictive_hub_ranking.tsv'
                    if save_ranking_as_tsv(data['ranking'], tsv_path, logger):
                        tsv_files_created.append(tsv_path.relative_to(comparison_dir.parent.parent))
            
            # Convert method_specific_ranking
            method_json_path = method_dir / 'method_specific_ranking.json'
            if method_json_path.exists():
                with open(method_json_path) as f:
                    data = json.load(f)
                if 'ranking' in data:
                    tsv_path = method_dir / 'method_specific_ranking.tsv'
                    if save_ranking_as_tsv(data['ranking'], tsv_path, logger):
                        tsv_files_created.append(tsv_path.relative_to(comparison_dir.parent.parent))
        
        # Convert consensus ranking
        consensus_path = comparison_dir / 'consensus_predictive_ranking.json'
        if consensus_path.exists():
            with open(consensus_path) as f:
                data = json.load(f)
            # The key is 'ranking' (you already fixed this in create_consensus_predictive_ranking)
            if 'ranking' in data:
                tsv_path = comparison_dir / 'consensus_predictive_ranking.tsv'
                if save_ranking_as_tsv(data['ranking'], tsv_path, logger):
                    tsv_files_created.append(tsv_path.relative_to(comparison_dir.parent.parent))
        
        # Convert annotated_hubs (already in memory from step 3)
        if annotated_hubs:
            # Save in comparison directory for easy access
            tsv_path = comparison_dir / '02_b_annotated_hubs.tsv'
            if save_ranking_as_tsv(annotated_hubs, tsv_path, logger):
                tsv_files_created.append(tsv_path.relative_to(comparison_dir.parent.parent))
                logger.info(f"✨ BONUS: Annotated hubs from 02_b also saved as TSV")
        
        # 5. NEW: Create consolidated ranking table
        logger.info("\n🎯 Enhancement 5: Creating consolidated ranking table...")
        if annotated_hubs and len(tsv_files_created) >= 3:
            consolidation_result = create_consolidated_ranking_table(
                comparison_dir, annotated_hubs, logger
            )
            if consolidation_result:
                enhancement_results['consolidated_rankings'] = consolidation_result
                enhancement_results['metadata']['enhancements_completed'].append('consolidated_rankings')
                logger.info(f"✅ Consolidated table created with {consolidation_result['n_genes']} genes")
        else:
            logger.warning("⚠️  Skipping consolidation: need annotated_hubs and TSV files")


        # Save all enhancement results
        enhancements_path = enhancements_dir / 'enhancements_summary.json'
        with open(enhancements_path, 'w') as f:
            json.dump(enhancement_results, f, indent=2)
        
        total_time = time.time() - start_time
        
        logger.info("\n" + "="*70)
        logger.info(" ENHANCEMENTS COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        logger.info(f"\n✅ Enhancements completed in {total_time:.1f} seconds")
        logger.info(f"📁 Output directory: {enhancements_dir.relative_to(comparison_dir.parent.parent)}")
        
        logger.info("\n📄 Generated Files:")
        for file_path in sorted(enhancements_dir.glob('*')):
            if file_path.is_file():
                logger.info(f"  • {file_path.name}")
        
        # List TSV files created
        if tsv_files_created:
            logger.info("\n📊 TSV Files Created:")
            for tsv_file in sorted(tsv_files_created):
                logger.info(f"  • {tsv_file} ← NEW")
        
        logger.info("="*70)
        
        return enhancement_results
        
    except Exception as e:
        logger.error(f"\n❌ Error running enhancements: {e}")
        logger.error(traceback.format_exc())
        raise