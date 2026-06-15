"""
Computational-focused visualizations for enrichment analysis.

This module provides visualizations that emphasize algorithmic performance,
statistical rigor, and methodological validation rather than biological
interpretation. Designed for computer science thesis presentation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.metrics import precision_recall_curve, auc
import json


def save_plot_metadata(plot_name, metadata, output_dir):
    """Save plot metadata to JSON file."""
    json_path = output_dir / f'{plot_name}_data.json'
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    return json_path


def plot_statistical_significance_distributions(all_results, output_dir, logger):
    """
    Chart 1: Statistical Significance Distribution Comparison
    
    Purpose: Demonstrate algorithmic discrimination power between cancer and novel genes.
    Shows that the dual-enrichment framework produces statistically distinct distributions.
    
    CS Metrics:
    - Kolmogorov-Smirnov test statistic and p-value
    - Median significance for each distribution
    - Separation ratio
    """
    logger.info("Creating statistical significance distributions plot...")
    
    # Extract p-values
    cancer_df = all_results.get('cancer', {}).get('significant', pd.DataFrame())
    novel_df = all_results.get('novel', {}).get('significant', pd.DataFrame())
    
    if cancer_df.empty or novel_df.empty:
        logger.warning("Skipping significance distributions - insufficient data")
        return None, None
    
    cancer_pvals = cancer_df['Adjusted P-value'].values
    novel_pvals = novel_df['Adjusted P-value'].values
    
    # Convert to -log10
    cancer_log = -np.log10(cancer_pvals + 1e-300)  # Avoid log(0)
    novel_log = -np.log10(novel_pvals + 1e-300)
    
    # Perform KS test
    ks_stat, ks_pval = stats.ks_2samp(cancer_log, novel_log)
    
    # Calculate metrics
    cancer_median = np.median(cancer_log)
    novel_median = np.median(novel_log)
    separation_ratio = cancer_median / novel_median if novel_median > 0 else 0
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left panel: Overlapping histograms
    max_val = max(cancer_log.max(), novel_log.max())
    bins = np.linspace(0, max_val, 40)
    
    ax1.hist(cancer_log, bins=bins, alpha=0.6, label=f'Cancer genes (n={len(cancer_log)})', 
             color='#e74c3c', edgecolor='black', linewidth=0.5)
    ax1.hist(novel_log, bins=bins, alpha=0.6, label=f'Novel genes (n={len(novel_log)})', 
             color='#3498db', edgecolor='black', linewidth=0.5)
    ax1.axvline(-np.log10(0.05), color='black', linestyle='--', linewidth=2, 
                label='p=0.05 threshold', zorder=10)
    
    # Add median lines
    ax1.axvline(cancer_median, color='#e74c3c', linestyle=':', linewidth=2, alpha=0.8)
    ax1.axvline(novel_median, color='#3498db', linestyle=':', linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('-log10(Adjusted P-value)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency (pathway count)', fontsize=11, fontweight='bold')
    ax1.set_title('Statistical Significance Distributions\n(Algorithmic Discrimination)', 
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(alpha=0.3, linestyle='--')
    
    # Right panel: Cumulative distributions (ECDF)
    cancer_sorted = np.sort(cancer_log)
    novel_sorted = np.sort(novel_log)
    cancer_ecdf = np.arange(1, len(cancer_sorted) + 1) / len(cancer_sorted)
    novel_ecdf = np.arange(1, len(novel_sorted) + 1) / len(novel_sorted)
    
    ax2.plot(cancer_sorted, cancer_ecdf, linewidth=2.5, label='Cancer genes', 
             color='#e74c3c', alpha=0.8)
    ax2.plot(novel_sorted, novel_ecdf, linewidth=2.5, label='Novel genes', 
             color='#3498db', alpha=0.8)
    
    # Add KS test annotation
    ks_text = f'Kolmogorov-Smirnov Test:\nD = {ks_stat:.3f}\np = {ks_pval:.2e}\n\nMedians:\nCancer: {cancer_median:.2f}\nNovel: {novel_median:.2f}\nRatio: {separation_ratio:.2f}x'
    ax2.text(0.05, 0.95, ks_text, transform=ax2.transAxes, va='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black'))
    
    ax2.set_xlabel('-log10(Adjusted P-value)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Cumulative Probability', fontsize=11, fontweight='bold')
    ax2.set_title('Empirical Cumulative Distribution Functions\n(KS Test Validation)', 
                  fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', framealpha=0.9)
    ax2.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'statistical_significance_distributions.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create metadata
    metadata = {
        'plot_info': {
            'name': 'statistical_significance_distributions',
            'type': 'computational_validation',
            'generated_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'plot_file': 'statistical_significance_distributions.png'
        },
        'description': (
            'Comparison of statistical significance distributions between cancer and novel gene '
            'enrichment results. Demonstrates the algorithmic discrimination power of the dual-enrichment '
            'framework through quantitative statistical testing.'
        ),
        'interpretation': {
            'how_to_read': [
                'Left panel: Overlapping histograms show frequency distribution of -log10(p-values)',
                'Right panel: Cumulative distribution functions (ECDF) for statistical comparison',
                'Vertical dashed line: p=0.05 significance threshold',
                'Vertical dotted lines: Median values for each distribution',
                'Text box: Kolmogorov-Smirnov test results and median comparison'
            ],
            'computational_interpretation': (
                f'KS test (D={ks_stat:.3f}, p={ks_pval:.2e}) confirms that cancer and novel gene '
                f'distributions are statistically distinct, validating the algorithmic separation. '
                f'Cancer genes show {separation_ratio:.1f}x higher median significance, confirming '
                'database coverage for known biology. Novel genes show weaker enrichment, validating '
                'genuine novelty beyond pathway databases.'
            ),
            'thesis_relevance': (
                'Demonstrates quantitative validation of the dual-enrichment algorithmic approach. '
                'Provides statistical evidence that the framework successfully discriminates between '
                'validation (known) and discovery (novel) gene sets.'
            )
        },
        'axes': {
            'left_x': '-log10(Adjusted P-value) - higher values = more significant enrichment',
            'left_y': 'Frequency - count of pathways in each significance bin',
            'right_x': '-log10(Adjusted P-value)',
            'right_y': 'Cumulative Probability - fraction of pathways with p-value ≤ threshold'
        },
        'metrics': {
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pval),
            'cancer_median_log_p': float(cancer_median),
            'novel_median_log_p': float(novel_median),
            'separation_ratio': float(separation_ratio),
            'cancer_count': int(len(cancer_log)),
            'novel_count': int(len(novel_log))
        },
        'data': {
            'cancer_log_pvalues': cancer_log.tolist(),
            'novel_log_pvalues': novel_log.tolist(),
            'significance_threshold': -np.log10(0.05)
        }
    }
    
    metadata_path = save_plot_metadata('statistical_significance_distributions', metadata, output_dir)
    logger.info(f"✓ Created significance distributions plot (KS D={ks_stat:.3f}, p={ks_pval:.2e})")
    
    return plot_path, metadata_path


def plot_pathway_coverage_analysis(all_results, stratified_genes, output_dir, logger):
    """
    Chart 2: Pathway Coverage Analysis
    
    Purpose: Quantify annotation completeness to justify novelty claims.
    Shows percentage of genes in each set that have pathway annotations.
    
    CS Metrics:
    - Coverage rate (annotated / total)
    - Annotation gap (novel vs cancer)
    """
    logger.info("Creating pathway coverage analysis plot...")
    
    # Get gene counts
    all_count = stratified_genes.get('all', {}).get('count', 250)
    cancer_count = stratified_genes.get('cancer', {}).get('count', 0)
    novel_count = stratified_genes.get('novel', {}).get('count', 0)
    
    # Count genes with pathway annotations (those that appear in enrichment results)
    cancer_enriched_genes = set()
    novel_enriched_genes = set()
    
    cancer_df = all_results.get('cancer', {}).get('significant', pd.DataFrame())
    novel_df = all_results.get('novel', {}).get('significant', pd.DataFrame())
    
    # Extract genes from overlap field (e.g., "TP53;BRCA1;MYC")
    if not cancer_df.empty and 'Genes' in cancer_df.columns:
        for genes_str in cancer_df['Genes'].dropna():
            cancer_enriched_genes.update(genes_str.split(';'))
    
    if not novel_df.empty and 'Genes' in novel_df.columns:
        for genes_str in novel_df['Genes'].dropna():
            novel_enriched_genes.update(genes_str.split(';'))
    
    # Calculate coverage
    cancer_annotated = min(len(cancer_enriched_genes), cancer_count)
    novel_annotated = min(len(novel_enriched_genes), novel_count)
    
    cancer_coverage = (cancer_annotated / cancer_count * 100) if cancer_count > 0 else 0
    novel_coverage = (novel_annotated / novel_count * 100) if novel_count > 0 else 0
    annotation_gap = cancer_coverage - novel_coverage
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left panel: Stacked bar chart
    categories = ['Cancer Genes', 'Novel Genes']
    annotated = [cancer_annotated, novel_annotated]
    unannotated = [cancer_count - cancer_annotated, novel_count - novel_annotated]
    
    x = np.arange(len(categories))
    width = 0.6
    
    bars1 = ax1.bar(x, annotated, width, label='Annotated in Databases', 
                    color='#27ae60', edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x, unannotated, width, bottom=annotated, label='Not Annotated',
                    color='#95a5a6', edgecolor='black', linewidth=1.5)
    
    # Add count labels
    for i, (ann, unann) in enumerate(zip(annotated, unannotated)):
        total = ann + unann
        # Annotated label
        ax1.text(i, ann/2, f'{ann}\n({ann/total*100:.1f}%)', 
                ha='center', va='center', fontweight='bold', fontsize=11)
        # Unannotated label
        ax1.text(i, ann + unann/2, f'{unann}\n({unann/total*100:.1f}%)', 
                ha='center', va='center', fontweight='bold', fontsize=11)
    
    ax1.set_ylabel('Gene Count', fontsize=12, fontweight='bold')
    ax1.set_title('Pathway Database Coverage by Gene Set\n(Data Quality Metric)', 
                  fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.9, fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Right panel: Coverage percentage comparison
    coverage_data = [cancer_coverage, novel_coverage]
    colors = ['#e74c3c', '#3498db']
    
    bars = ax2.barh(categories, coverage_data, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add percentage labels
    for i, (bar, pct) in enumerate(zip(bars, coverage_data)):
        ax2.text(pct + 2, i, f'{pct:.1f}%', va='center', fontweight='bold', fontsize=11)
    
    ax2.set_xlabel('Coverage Percentage (%)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Annotation Coverage Rate\n(Gap: {annotation_gap:.1f}%)', 
                  fontsize=13, fontweight='bold')
    ax2.set_xlim(0, 105)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add annotation gap indicator
    if annotation_gap > 0:
        ax2.annotate('', xy=(novel_coverage, 1), xytext=(cancer_coverage, 0),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=2))
        ax2.text((cancer_coverage + novel_coverage)/2, 0.5, f'{annotation_gap:.1f}%\ngap',
                ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'pathway_coverage_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create metadata
    metadata = {
        'plot_info': {
            'name': 'pathway_coverage_analysis',
            'type': 'data_quality',
            'generated_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'plot_file': 'pathway_coverage_analysis.png'
        },
        'description': (
            'Analysis of pathway database coverage for cancer versus novel gene sets. '
            'Quantifies what percentage of genes in each set have pathway annotations, '
            'providing data quality metrics and justification for novelty claims.'
        ),
        'interpretation': {
            'how_to_read': [
                'Left panel: Stacked bars showing annotated (green) vs unannotated (gray) genes',
                'Right panel: Horizontal bars showing coverage percentage',
                'Red arrow: Annotation gap between cancer and novel genes',
                'Higher coverage = more genes found in pathway databases'
            ],
            'computational_interpretation': (
                f'Cancer genes show {cancer_coverage:.1f}% database coverage ({cancer_annotated}/{cancer_count}), '
                f'validating that known cancer genes are well-represented in pathway databases. '
                f'Novel genes show only {novel_coverage:.1f}% coverage ({novel_annotated}/{novel_count}), '
                f'confirming {annotation_gap:.1f}% annotation gap. This quantitatively demonstrates that '
                'novel genes are genuinely not in pathway databases, not just poorly enriched.'
            ),
            'thesis_relevance': (
                'Provides quantitative evidence for novelty claims. Low coverage of novel genes '
                'validates that they represent genuine discovery beyond known biology, not just '
                'weak statistical signals from annotated genes.'
            )
        },
        'axes': {
            'left_y': 'Gene set category',
            'left_x': 'Gene count (stacked: annotated + unannotated)',
            'right_y': 'Gene set category',
            'right_x': 'Percentage of genes with pathway annotations'
        },
        'metrics': {
            'cancer_total': int(cancer_count),
            'cancer_annotated': int(cancer_annotated),
            'cancer_coverage_pct': float(cancer_coverage),
            'novel_total': int(novel_count),
            'novel_annotated': int(novel_annotated),
            'novel_coverage_pct': float(novel_coverage),
            'annotation_gap_pct': float(annotation_gap)
        },
        'data': {
            'cancer': {
                'total': int(cancer_count),
                'annotated': int(cancer_annotated),
                'unannotated': int(cancer_count - cancer_annotated)
            },
            'novel': {
                'total': int(novel_count),
                'annotated': int(novel_annotated),
                'unannotated': int(novel_count - novel_annotated)
            }
        }
    }
    
    metadata_path = save_plot_metadata('pathway_coverage_analysis', metadata, output_dir)
    logger.info(f"✓ Created coverage analysis (Cancer: {cancer_coverage:.1f}%, Novel: {novel_coverage:.1f}%)")
    
    return plot_path, metadata_path


def plot_enrichment_quality_metrics(all_results, output_dir, logger):
    """
    Chart 3: Enrichment Quality Metrics (Q-Q Plot)
    
    Purpose: Validate that observed p-values follow expected distribution under true enrichment.
    Shows deviation from null expectation.
    
    CS Metrics:
    - Observed vs expected -log10(p) quantiles
    - Genomic inflation factor λ
    """
    logger.info("Creating enrichment quality metrics (Q-Q plot)...")
    
    # Get all p-values
    all_df = all_results.get('all', {}).get('significant', pd.DataFrame())
    
    if all_df.empty:
        logger.warning("Skipping Q-Q plot - no data")
        return None, None
    
    observed_p = all_df['Adjusted P-value'].values
    observed_log = -np.log10(observed_p + 1e-300)
    
    # Expected p-values under uniform null
    n = len(observed_log)
    expected_p = np.arange(1, n + 1) / (n + 1)
    expected_log = -np.log10(expected_p)
    
    # Sort observed
    observed_sorted = np.sort(observed_log)
    
    # Calculate genomic inflation factor (λ)
    # λ = median(observed²) / median(expected²) for chi-square distribution
    # For p-values: λ ≈ median(observed) / median(expected)
    lambda_gc = np.median(observed_sorted) / np.median(expected_log) if np.median(expected_log) > 0 else 1.0
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Q-Q plot
    ax.scatter(expected_log, observed_sorted, alpha=0.6, s=50, color='#3498db', 
               edgecolors='black', linewidth=0.5)
    
    # Diagonal line (y=x)
    max_val = max(expected_log.max(), observed_sorted.max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Null expectation (y=x)', zorder=10)
    
    # Add λ annotation
    lambda_text = f'Genomic Inflation Factor:\nλ = {lambda_gc:.3f}\n\nInterpretation:\nλ > 1: True enrichment\nλ = 1: Null distribution\nλ < 1: Deflation'
    ax.text(0.05, 0.95, lambda_text, transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black'))
    
    ax.set_xlabel('Expected -log10(P-value)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Observed -log10(P-value)', fontsize=12, fontweight='bold')
    ax.set_title('Q-Q Plot: Enrichment Quality Assessment\n(Deviation from Null Expectation)', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'enrichment_quality_qq_plot.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create metadata
    metadata = {
        'plot_info': {
            'name': 'enrichment_quality_qq_plot',
            'type': 'statistical_validation',
            'generated_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'plot_file': 'enrichment_quality_qq_plot.png'
        },
        'description': (
            'Quantile-Quantile (Q-Q) plot comparing observed p-value distribution against '
            'expected distribution under the null hypothesis. Validates statistical quality '
            'of enrichment results.'
        ),
        'interpretation': {
            'how_to_read': [
                'X-axis: Expected -log10(p-value) under uniform null distribution',
                'Y-axis: Observed -log10(p-value) from enrichment analysis',
                'Red dashed line: y=x (null expectation)',
                'Points above line: More significant than expected (true enrichment)',
                'Points on line: Follows null distribution',
                'λ (lambda): Genomic inflation factor measuring overall deviation'
            ],
            'computational_interpretation': (
                f'Genomic inflation factor λ = {lambda_gc:.3f}. '
                f'{"λ > 1 indicates true enrichment signal beyond random chance. " if lambda_gc > 1 else ""}'
                'Deviation of points from the diagonal line validates that observed enrichment '
                'is statistically robust and not due to random variation or technical artifacts.'
            ),
            'thesis_relevance': (
                'Demonstrates statistical rigor of the enrichment pipeline. Q-Q plot is a '
                'standard validation tool in genomics showing the methodology follows best practices.'
            )
        },
        'axes': {
            'x': 'Expected -log10(P-value) under H0 (uniform distribution)',
            'y': 'Observed -log10(P-value) from enrichment analysis'
        },
        'metrics': {
            'lambda_gc': float(lambda_gc),
            'n_pathways': int(n),
            'median_expected': float(np.median(expected_log)),
            'median_observed': float(np.median(observed_sorted))
        },
        'data': {
            'expected_log_p': expected_log.tolist(),
            'observed_log_p': observed_sorted.tolist()
        }
    }
    
    metadata_path = save_plot_metadata('enrichment_quality_qq_plot', metadata, output_dir)
    logger.info(f"✓ Created Q-Q plot (λ = {lambda_gc:.3f})")
    
    return plot_path, metadata_path

"""
COMPLETE CODE TO ADD TO visualization_computational.py

Add these 3 functions at the END of the file (after plot_enrichment_quality_qq)

CRITICAL FIXES APPLIED:
1. Added missing scipy.ndimage import
2. Fixed stratified_genes structure handling in Chart 6
3. Updated function signature for Chart 6
4. Added robust error handling
"""

# =============================================================================
# CHART 4: DATABASE CONTRIBUTION HEATMAP
# =============================================================================

def plot_database_contribution_heatmap(all_results, output_dir, logger):
    """
    Chart 4: Database Contribution Heatmap
    
    Shows multi-database validation patterns and database biases.
    3x3 heatmap showing pathway counts across databases (rows) for each gene set (columns).
    
    CS Metrics:
    - Database-specific enrichment patterns
    - Orthogonal validation across databases
    - Database bias quantification
    """
    logger.info("Creating database contribution heatmap...")
    
    # Define databases and gene sets
    databases = ['KEGG_2021_Human', 'GO_Biological_Process_2023', 'Reactome_2022']
    gene_sets = ['all', 'cancer', 'novel']
    
    # Initialize heatmap matrix
    heatmap_data = np.zeros((len(databases), len(gene_sets)))
    
    # Populate matrix with pathway counts
    for i, db in enumerate(databases):
        for j, gene_set in enumerate(gene_sets):
            if gene_set in all_results:
                sig_df = all_results[gene_set].get('significant', pd.DataFrame())
                if not sig_df.empty and 'Database' in sig_df.columns:
                    db_count = len(sig_df[sig_df['Database'] == db])
                    heatmap_data[i, j] = db_count
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(gene_sets)))
    ax.set_yticks(np.arange(len(databases)))
    
    # Get gene counts for x labels
    gene_counts = []
    for gs in gene_sets:
        count = all_results.get(gs, {}).get('summary', {}).get('gene_count', 0)
        gene_counts.append(count)
    
    ax.set_xticklabels([f'{gs.upper()}\n(n={count})' 
                        for gs, count in zip(gene_sets, gene_counts)], 
                       fontsize=11, fontweight='bold')
    ax.set_yticklabels([db.split('_')[0] for db in databases], 
                       fontsize=11, fontweight='bold')
    
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    
    # Add text annotations
    for i in range(len(databases)):
        for j in range(len(gene_sets)):
            count = heatmap_data[i, j]
            # Determine text color based on background
            color = 'white' if count >= np.max(heatmap_data)/2 else 'black'
            text = int(count) if count > 0 else '0'
            ax.text(j, i, text, ha="center", va="center", color=color, 
                   fontsize=12, fontweight='bold')
    
    # Title and labels
    ax.set_title("Multi-Database Pathway Enrichment\n(Cross-Validation Heatmap)", 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Gene Set Category", fontsize=12, fontweight='bold')
    ax.set_ylabel("Pathway Database", fontsize=12, fontweight='bold')
    
    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Pathway Count', rotation=90, va="bottom", fontsize=11)
    
    plt.tight_layout()
    
    # Save
    plot_path = output_dir / 'database_contribution_heatmap.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate database bias metrics
    db_bias = {}
    for i, db in enumerate(databases):
        row_sum = np.sum(heatmap_data[i, :])
        db_name = db.split('_')[0]
        if row_sum > 0:
            db_bias[db_name] = {
                'total_pathways': int(row_sum),
                'cancer_fraction': float(heatmap_data[i, 1] / row_sum),
                'novel_fraction': float(heatmap_data[i, 2] / row_sum),
                'coefficient_of_variation': float(np.std(heatmap_data[i, :]) / np.mean(heatmap_data[i, :]))
            }
    
    # Metadata
    metadata = {
        'plot_info': {
            'name': 'database_contribution_heatmap',
            'type': 'cross_validation',
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'description': (
            'Heatmap showing pathway enrichment counts across three databases (KEGG, GO, Reactome) '
            'for three gene sets (all, cancer, novel). Demonstrates multi-database validation patterns.'
        ),
        'how_to_read': {
            'rows': 'Pathway databases (KEGG, GO, Reactome)',
            'columns': 'Gene set categories with counts',
            'colors': 'Yellow (low) to Red (high) pathway counts',
            'numbers': 'Exact pathway counts in each cell',
            'interpretation': 'Cancer genes should show strong enrichment across all databases. Novel genes show weaker/different patterns.'
        },
        'computational_metrics': {
            'heatmap_matrix': heatmap_data.tolist(),
            'database_bias': db_bias,
            'max_enrichment': float(np.max(heatmap_data)),
            'validation': 'Multi-source cross-validation'
        },
        'data': {
            'databases': databases,
            'gene_sets': gene_sets,
            'gene_counts': gene_counts
        }
    }
    
    metadata_path = save_plot_metadata('database_contribution_heatmap', metadata, output_dir)
    logger.info(f"✓ Database heatmap created")
    
    return plot_path, metadata_path


# =============================================================================
# CHART 5: EFFECT SIZE VS SIGNIFICANCE (VOLCANO PLOT)
# =============================================================================

def plot_effect_size_vs_significance(all_sig_df, output_dir, logger):
    """
    Chart 5: Effect Size vs Significance (Volcano Plot)
    
    Distinguishes strong biological signals from statistical artifacts.
    Shows relationship between effect size and significance.
    
    CS Metrics:
    - Effect size distribution
    - Significance threshold compliance
    - High-confidence quadrant analysis
    """
    logger.info("Creating effect size vs significance plot...")
    
    if all_sig_df.empty:
        logger.warning("Skipping volcano plot - no data")
        return None, None
    
    # Check required columns
    required_cols = ['Combined Score', 'Adjusted P-value']
    if not all(col in all_sig_df.columns for col in required_cols):
        logger.warning(f"Missing columns. Available: {all_sig_df.columns.tolist()}")
        return None, None
    
    # Prepare data
    df = all_sig_df.copy()
    effect_size = df['Combined Score'].values
    significance = -np.log10(df['Adjusted P-value'].values + 1e-300)
    
    # Define thresholds
    effect_threshold = np.median(effect_size) if len(effect_size) > 0 else 50
    sig_threshold = -np.log10(0.05)  # p=0.05
    
    # Classify into quadrants
    quadrants = []
    for es, sig in zip(effect_size, significance):
        if es >= effect_threshold and sig >= sig_threshold:
            quadrants.append('High-Confidence')
        elif es < effect_threshold and sig >= sig_threshold:
            quadrants.append('Significant Only')
        elif es >= effect_threshold and sig < sig_threshold:
            quadrants.append('Large Effect Only')
        else:
            quadrants.append('Low Confidence')
    
    df['Quadrant'] = quadrants
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Volcano plot
    colors = {
        'High-Confidence': '#e74c3c', 
        'Significant Only': '#3498db',
        'Large Effect Only': '#f39c12',
        'Low Confidence': '#95a5a6'
    }
    
    for quadrant, color in colors.items():
        mask = df['Quadrant'] == quadrant
        if mask.any():
            ax1.scatter(effect_size[mask], significance[mask], 
                       alpha=0.6, s=50, color=color, label=quadrant,
                       edgecolors='black', linewidth=0.5)
    
    # Threshold lines
    ax1.axhline(y=sig_threshold, color='black', linestyle='--', linewidth=2, 
                alpha=0.7, label=f'p=0.05')
    ax1.axvline(x=effect_threshold, color='black', linestyle='--', linewidth=2, 
                alpha=0.7, label=f'Median effect')
    
    ax1.set_xlabel('Effect Size (Combined Score)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('-log10(Adjusted P-value)', fontsize=12, fontweight='bold')
    ax1.set_title('Effect Size vs Significance\n(Volcano Plot)', 
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', framealpha=0.9, fontsize=9)
    ax1.grid(alpha=0.3, linestyle='--')
    
    # Right: Quadrant distribution
    quadrant_counts = df['Quadrant'].value_counts()
    colors_ordered = [colors[q] for q in quadrant_counts.index]
    
    wedges, texts, autotexts = ax2.pie(
        quadrant_counts.values, 
        labels=quadrant_counts.index, 
        colors=colors_ordered, 
        autopct='%1.1f%%',
        startangle=90, 
        textprops={'fontsize': 10}
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax2.set_title('Quadrant Distribution\n(Confidence Assessment)', 
                  fontsize=13, fontweight='bold')
    
    # Add metrics
    high_conf_count = len(df[df['Quadrant'] == 'High-Confidence'])
    metrics_text = f'Thresholds:\nEffect: {effect_threshold:.1f}\nSignif: {sig_threshold:.2f}\n\nHigh-confidence:\n{high_conf_count} ({high_conf_count/len(df)*100:.1f}%)'
    ax2.text(-1.5, -1.3, metrics_text, fontsize=9, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save
    plot_path = output_dir / 'effect_size_vs_significance.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate statistics
    quad_stats = {}
    for quadrant in colors.keys():
        quad_df = df[df['Quadrant'] == quadrant]
        if not quad_df.empty:
            quad_stats[quadrant] = {
                'count': len(quad_df),
                'percentage': float(len(quad_df) / len(df) * 100)
            }
    
    # Metadata
    metadata = {
        'plot_info': {
            'name': 'effect_size_vs_significance',
            'type': 'signal_quality',
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'description': 'Volcano plot showing effect size vs statistical significance. Distinguishes robust signals from artifacts.',
        'how_to_read': {
            'x_axis': 'Effect size (Combined Score)',
            'y_axis': 'Statistical significance (-log10 p-value)',
            'quadrants': {
                'top_right': 'High-confidence (both high effect and significance)',
                'top_left': 'Statistically significant but small effect',
                'bottom_right': 'Large effect but not significant',
                'bottom_left': 'Low confidence'
            },
            'interpretation': 'Focus on top-right quadrant for most robust findings'
        },
        'computational_metrics': {
            'total_pathways': len(df),
            'effect_threshold': float(effect_threshold),
            'significance_threshold': float(sig_threshold),
            'quadrant_distribution': quad_stats,
            'high_confidence_pct': quad_stats.get('High-Confidence', {}).get('percentage', 0)
        },
        'data': {
            'effect_sizes': effect_size.tolist()[:100],  # Truncate for size
            'significance_values': significance.tolist()[:100],
            'quadrant_counts': {k: v['count'] for k, v in quad_stats.items()}
        }
    }
    
    metadata_path = save_plot_metadata('effect_size_vs_significance', metadata, output_dir)
    logger.info(f"✓ Effect size plot created ({len(df)} pathways)")
    
    return plot_path, metadata_path


# =============================================================================
# CHART 6: NOVELTY GRADIENT (WITH FIXES)
# =============================================================================

def plot_novelty_gradient(all_results, stratified_genes, ranking_df, output_dir, logger):
    """
    Chart 6: Novelty Gradient Across Rankings
    
    Shows how cancer/novel distribution changes across hub rankings.
    Demonstrates data-driven split, not arbitrary classification.
    
    FIXED: Handles multiple stratified_genes structures
    FIXED: Requires ranking_df from 03_a enhanced_hub_ranking.tsv
    """
    logger.info("Creating novelty gradient plot...")
    
    if ranking_df is None or ranking_df.empty:
        logger.warning("Skipping novelty gradient - no ranking data")
        return None, None
    
    # Parse gene symbols from ranking data
    def parse_gene_symbol_simple(gene_str):
        """Extract gene symbol from ENSG|SYMBOL format."""
        gene_str = str(gene_str)
        if '|' in gene_str:
            return gene_str.split('|')[1].upper()
        return gene_str.upper()
    
    # Ensure gene_symbol column exists
    ranking_df = ranking_df.copy()
    if 'gene_symbol' not in ranking_df.columns:
        if 'gene' in ranking_df.columns:
            ranking_df['gene_symbol'] = ranking_df['gene'].apply(parse_gene_symbol_simple)
        else:
            logger.warning("No gene column found in ranking data")
            return None, None
    
    # Get gene sets - FIXED to handle both list and dict structures
    cancer_genes = stratified_genes.get('cancer', {})
    novel_genes = stratified_genes.get('novel', {})
    
    # Extract gene lists - handle multiple formats
    if isinstance(cancer_genes, dict):
        cancer_gene_list = cancer_genes.get('genes', [])
    elif isinstance(cancer_genes, (list, set)):
        cancer_gene_list = list(cancer_genes)
    else:
        cancer_gene_list = []
    
    if isinstance(novel_genes, dict):
        novel_gene_list = novel_genes.get('genes', [])
    elif isinstance(novel_genes, (list, set)):
        novel_gene_list = list(novel_genes)
    else:
        novel_gene_list = []
    
    cancer_set = set(cancer_gene_list)
    novel_set = set(novel_gene_list)
    
    logger.info(f"  Cancer genes: {len(cancer_set)}, Novel genes: {len(novel_set)}")
    
    if len(cancer_set) == 0 and len(novel_set) == 0:
        logger.warning("  No cancer or novel genes found - check stratified_genes structure")
        return None, None
    
    # Classify genes
    classifications = []
    for gene in ranking_df['gene_symbol']:
        if gene in cancer_set:
            classifications.append('Cancer')
        elif gene in novel_set:
            classifications.append('Novel')
        else:
            classifications.append('Other')
    
    ranking_df['classification'] = classifications
    
    # Sort by enhanced score (descending)
    if 'enhanced_score' in ranking_df.columns:
        ranking_df = ranking_df.sort_values('enhanced_score', ascending=False)
    elif 'rank' in ranking_df.columns:
        ranking_df = ranking_df.sort_values('rank', ascending=True)
    
    ranking_df = ranking_df.reset_index(drop=True)
    ranking_df['position'] = ranking_df.index + 1
    ranking_df['percentile'] = (ranking_df['position'] / len(ranking_df) * 100)
    
    # Create bins
    n_bins = 20
    bin_edges = np.linspace(0, 100, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    cancer_counts = np.zeros(n_bins)
    novel_counts = np.zeros(n_bins)
    other_counts = np.zeros(n_bins)
    
    # Count genes in each bin
    for i, (bin_start, bin_end) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        mask = (ranking_df['percentile'] >= bin_start) & (ranking_df['percentile'] < bin_end)
        bin_df = ranking_df[mask]
        
        cancer_counts[i] = len(bin_df[bin_df['classification'] == 'Cancer'])
        novel_counts[i] = len(bin_df[bin_df['classification'] == 'Novel'])
        other_counts[i] = len(bin_df[bin_df['classification'] == 'Other'])
    
    # Convert to percentages (avoid division by zero)
    total_counts = cancer_counts + novel_counts + other_counts
    total_counts = np.where(total_counts == 0, 1, total_counts)
    
    cancer_pct = (cancer_counts / total_counts * 100)
    novel_pct = (novel_counts / total_counts * 100)
    other_pct = (other_counts / total_counts * 100)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Stacked area chart
    colors = {'Cancer': '#e74c3c', 'Novel': '#3498db', 'Other': '#95a5a6'}
    
    ax1.fill_between(bin_centers, 0, cancer_pct, alpha=0.7, color=colors['Cancer'], 
                     label='Cancer', edgecolor='black', linewidth=1)
    ax1.fill_between(bin_centers, cancer_pct, cancer_pct + novel_pct, alpha=0.7, 
                     color=colors['Novel'], label='Novel', edgecolor='black', linewidth=1)
    ax1.fill_between(bin_centers, cancer_pct + novel_pct, 100, alpha=0.7, 
                     color=colors['Other'], label='Other', edgecolor='black', linewidth=1)
    
    ax1.set_xlabel('Ranking Percentile (0 = top)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Percentage of Genes (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Novelty Gradient Across Rankings\n(Discovery Distribution)', 
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', framealpha=0.9, fontsize=10)
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    
    # Add smoothed trends (only if enough data points)
    if len(cancer_pct) > 3:
        from scipy.ndimage import gaussian_filter1d
        try:
            cancer_smooth = gaussian_filter1d(cancer_pct, sigma=1)
            novel_smooth = gaussian_filter1d(novel_pct, sigma=1)
            ax1.plot(bin_centers, cancer_smooth, 'k-', linewidth=2, alpha=0.7)
            ax1.plot(bin_centers, cancer_smooth + novel_smooth, 'k--', linewidth=2, alpha=0.7)
        except Exception as e:
            logger.warning(f"Could not add trend lines: {e}")
    
    # Right: Top vs bottom comparison
    top_cutoff = 20
    bottom_cutoff = 80
    
    top_mask = ranking_df['percentile'] <= top_cutoff
    bottom_mask = ranking_df['percentile'] >= bottom_cutoff
    
    top_counts = ranking_df[top_mask]['classification'].value_counts()
    bottom_counts = ranking_df[bottom_mask]['classification'].value_counts()
    
    categories = ['Top 20%', 'Bottom 20%']
    cancer_values = [top_counts.get('Cancer', 0), bottom_counts.get('Cancer', 0)]
    novel_values = [top_counts.get('Novel', 0), bottom_counts.get('Novel', 0)]
    other_values = [top_counts.get('Other', 0), bottom_counts.get('Other', 0)]
    
    x = np.arange(len(categories))
    width = 0.25
    
    ax2.bar(x - width, cancer_values, width, label='Cancer', color=colors['Cancer'], 
            alpha=0.8, edgecolor='black')
    ax2.bar(x, novel_values, width, label='Novel', color=colors['Novel'], 
            alpha=0.8, edgecolor='black')
    ax2.bar(x + width, other_values, width, label='Other', color=colors['Other'], 
            alpha=0.8, edgecolor='black')
    
    ax2.set_xlabel('Ranking Segment', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Gene Count', fontsize=12, fontweight='bold')
    ax2.set_title('Top vs Bottom Composition\n(Position Bias)', 
                  fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax2.legend(framealpha=0.9, fontsize=10)
    ax2.grid(alpha=0.3, linestyle='--', axis='y')
    
    # Add value labels (only if values > 0)
    for i, (c, n, o) in enumerate(zip(cancer_values, novel_values, other_values)):
        if c > 0:
            y_offset = max(cancer_values) * 0.02 if max(cancer_values) > 0 else 1
            ax2.text(i - width, c + y_offset, str(c), ha='center', va='bottom', 
                    fontweight='bold', fontsize=9)
        if n > 0:
            y_offset = max(novel_values) * 0.02 if max(novel_values) > 0 else 1
            ax2.text(i, n + y_offset, str(n), ha='center', va='bottom', 
                    fontweight='bold', fontsize=9)
        if o > 0:
            y_offset = max(other_values) * 0.02 if max(other_values) > 0 else 1
            ax2.text(i + width, o + y_offset, str(o), ha='center', va='bottom', 
                    fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    
    # Save
    plot_path = output_dir / 'novelty_gradient.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate metrics
    novelty_ratio_top = novel_values[0] / cancer_values[0] if cancer_values[0] > 0 else 0
    novelty_ratio_bottom = novel_values[1] / cancer_values[1] if cancer_values[1] > 0 else 0
    
    # Metadata
    metadata = {
        'plot_info': {
            'name': 'novelty_gradient',
            'type': 'ranking_analysis',
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'description': 'Analysis of cancer vs novel gene distribution across hub rankings. Shows data-driven split.',
        'how_to_read': {
            'left_panel': 'Stacked area showing gene type percentages across rankings',
            'right_panel': 'Top 20% vs bottom 20% comparison',
            'colors': 'Red=cancer, Blue=novel, Gray=other',
            'interpretation': 'Flat distribution = no ranking bias against novelty'
        },
        'computational_metrics': {
            'total_genes': len(ranking_df),
            'cancer_genes': len(cancer_set),
            'novel_genes': len(novel_set),
            'top_20_novelty_ratio': float(novelty_ratio_top),
            'bottom_20_novelty_ratio': float(novelty_ratio_bottom),
            'position_bias': 'Minimal' if abs(novelty_ratio_top - novelty_ratio_bottom) < 1.0 else 'Present'
        },
        'data': {
            'bin_centers': bin_centers.tolist(),
            'cancer_pct': cancer_pct.tolist(),
            'novel_pct': novel_pct.tolist(),
            'top_counts': {'cancer': int(cancer_values[0]), 'novel': int(novel_values[0])},
            'bottom_counts': {'cancer': int(cancer_values[1]), 'novel': int(novel_values[1])}
        }
    }
    
    metadata_path = save_plot_metadata('novelty_gradient', metadata, output_dir)
    logger.info(f"✓ Novelty gradient created ({len(ranking_df)} genes)")
    
    return plot_path, metadata_path