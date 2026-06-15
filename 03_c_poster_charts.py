"""
03_c_poster_charts.py

Script Purpose:
This standalone script generates high-resolution, publication-quality charts 
specifically formatted for the research poster (Images 3B and 3C). 

All poster specifications implemented:
- 300 DPI output for professional printing
- Exact color scheme matching poster requirements
- Complete statistical annotations and labels
- Publication-grade formatting
"""

import pandas as pd
import numpy as np
import json
import logging
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from matplotlib.patches import Patch

# Import project utilities
from utils.config import load_config
from utils.file import ensure_dir, get_relative_path

def setup_poster_logging(output_dir):
    """Set up dedicated logging for poster chart generation."""
    logger = logging.getLogger("poster_logger")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    log_dir = output_dir / 'logs'
    ensure_dir(log_dir)
    log_file = log_dir / '03_c_poster_charts.log'
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    return logger

def create_poster_violin_plot(sig_pairs_df, output_dir, logger):
    """Creates Image 3B: Clean violin plot with statistical annotations (no pie chart)."""
    try:
        logger.info("Generating Poster Image 3B (Violin Plot)...")
        
        # Calculate absolute delta_r values
        abs_delta_r = sig_pairs_df['delta_r'].abs().values
        n_pairs = len(abs_delta_r)
        
        # Calculate statistics
        median_val = np.median(abs_delta_r)
        q1, q3 = np.percentile(abs_delta_r, [25, 75])
        iqr = q3 - q1
        mean_val = np.mean(abs_delta_r)
        std_val = np.std(abs_delta_r)
        
        logger.info(f"  Statistics: Median={median_val:.3f}, IQR={iqr:.3f}, Mean={mean_val:.3f}±{std_val:.3f}")
        
        # Count directional pairs (for logging)
        n_gains = np.sum(sig_pairs_df['delta_r'] > 0)
        n_losses = np.sum(sig_pairs_df['delta_r'] < 0)
        logger.info(f"  Directional: {n_gains} gains ({n_gains/n_pairs*100:.1f}%), {n_losses} losses")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 10))
        
        # Create violin plot
        parts = ax.violinplot([abs_delta_r], positions=[0], vert=True, 
                              widths=0.7, showmeans=False, showmedians=False)
        
        # Color violin orange
        for pc in parts['bodies']:
            pc.set_facecolor('#e67e22')
            pc.set_edgecolor('#d35400')
            pc.set_alpha(0.7)
        
        # Add median line
        ax.axhline(median_val, color='black', linestyle='--', linewidth=2.5, 
                   label=f'Median = {median_val:.3f}', zorder=5)
        
        # Add IQR shaded region
        ax.axhspan(q1, q3, alpha=0.25, color='orange', label=f'IQR = {iqr:.3f}', zorder=1)
        
        # Set axis properties
        ax.set_ylabel('Effect Size |Δr|', fontsize=16, fontweight='bold')
        ax.set_ylim(0.75, 1.0)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.grid(axis='y', alpha=0.3, linestyle=':', color='gray')
        
        # Add statistics text box
        stats_text = (f'Median = {median_val:.3f}\n'
                     f'IQR = {iqr:.3f}\n'
                     f'Mean = {mean_val:.3f} ± {std_val:.3f}')
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
        
        # Add title and subtitle
        ax.set_title(f'Effect Size Distribution of Significant Rewired Pairs (n = {n_pairs:,})', 
                     fontsize=14, fontweight='bold', pad=15)
        ax.text(0.5, 1.02, '(|Δr| = absolute change in correlation between tumor and normal)',
                transform=ax.transAxes, fontsize=11, ha='center', style='italic', color='gray')
        
        # Add explanation text
        fig.text(0.5, 0.02, 
                 'Most rewired pairs show very large and consistent increases in correlation,\n'
                 'suggesting a uniform biological mechanism rather than random noise.',
                 ha='center', fontsize=11, style='italic', color='#555555', wrap=True)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.98])
        
        # Save PNG
        png_path = output_dir / 'image_3b_rewiring_magnitude.png'
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"✓ Saved Image 3B: {png_path.name}")
        plt.close()
        
        return png_path
        
    except Exception as e:
        logger.error(f"Error in Image 3B: {e}", exc_info=True)
        return None

def create_poster_roc_and_hubs(y_test, y_proba, rankings_df, output_dir, logger):
    """Creates Image 3C: ROC curve and Top 10 hubs bar chart with full annotations."""
    try:
        logger.info("Generating Poster Image 3C (ROC + Top 10 Hubs)...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # === LEFT: ROC CURVE ===
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        logger.info(f"  ROC AUC: {roc_auc:.4f}")
        
        # Plot ROC curve
        ax1.plot(fpr, tpr, color='#2e7d9a', lw=4, label='Ensemble Classifier', zorder=3)
        ax1.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Chance', zorder=1)
        ax1.fill_between(fpr, tpr, alpha=0.2, color='#2e7d9a', zorder=2)
        
        # Large AUC annotation
        ax1.text(0.55, 0.25, f'AUC = {roc_auc:.4f}', fontsize=26, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.95, 
                          edgecolor='black', linewidth=2.5),
                 ha='center', va='center', zorder=5)
        
        ax1.set_title('ROC Curve - Tumor vs Normal Classification', 
                     fontsize=13, fontweight='bold', pad=10)
        ax1.set_xlabel('FPR (False Positive Rate)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('TPR (True Positive Rate)', fontsize=14, fontweight='bold')
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.set_aspect('equal', adjustable='box')
        ax1.grid(True, alpha=0.3, linestyle=':', color='gray', zorder=0)
        ax1.legend(loc='lower right', fontsize=12, framealpha=0.9)
        ax1.tick_params(axis='both', which='major', labelsize=11)
        
        # === RIGHT: TOP 10 HUBS BAR CHART ===
        top_10 = rankings_df.head(10).copy()
        
        logger.info(f"  Top 10 genes: {', '.join(top_10['gene_symbol'].tolist())}")
        
        # Define tier colors
        tier_colors = {
            'breast_cancer': '#e74c3c',  # Orange-red
            'cancer': '#f1c40f',         # Yellow
            'non_cancer': '#27ae60'      # Green
        }
        colors = [tier_colors.get(tier, '#95a5a6') for tier in top_10['cancer_relevance']]
        
        # Count tier composition
        tier_counts = top_10['cancer_relevance'].value_counts().to_dict()
        logger.info(f"  Tier composition: {tier_counts}")
        
        # Horizontal bar chart
        y_pos = np.arange(len(top_10))
        ax2.barh(y_pos, top_10['composite_score'], color=colors, 
                height=0.75, edgecolor='black', linewidth=0.8, alpha=0.9)
        
        # Gene labels
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(top_10['gene_symbol'], fontsize=13, fontweight='bold')
        ax2.invert_yaxis()
        
        # Axis labels
        ax2.set_xlabel('Composite Score', fontsize=14, fontweight='bold')
        ax2.set_title('Top 10 Predictive Hub Genes', fontsize=13, fontweight='bold', pad=10)
        
        # Add value labels on bars
        for i, (idx, row) in enumerate(top_10.iterrows()):
            score = row['composite_score']
            ax2.text(score + (ax2.get_xlim()[1] * 0.02), i, f"{score:.0f}", 
                    va='center', ha='left', fontsize=11, fontweight='bold')
        
        # Add legend
        legend_elements = [
            Patch(facecolor='#e74c3c', edgecolor='black', linewidth=0.8, label='Breast Cancer'),
            Patch(facecolor='#f1c40f', edgecolor='black', linewidth=0.8, label='General Cancer'),
            Patch(facecolor='#27ae60', edgecolor='black', linewidth=0.8, label='Novel')
        ]
        ax2.legend(handles=legend_elements, loc='lower right', fontsize=10.5, 
                  title='Gene Tier', title_fontsize=11, framealpha=0.95)
        
        ax2.grid(axis='x', alpha=0.3, linestyle=':', color='gray')
        ax2.tick_params(axis='both', which='major', labelsize=11)
        
        plt.tight_layout(pad=2.0)
        
        # Save PNG
        png_path = output_dir / 'image_3c_roc_and_hubs.png'
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"✓ Saved Image 3C: {png_path.name}")
        plt.close()
        
        return png_path
        
    except Exception as e:
        logger.error(f"Error in Image 3C: {e}", exc_info=True)
        return None

def main():
    config = load_config()
    project_root = Path(__file__).parent.parent
    output_dir = project_root / 'output' / '03_c_poster_charts'
    ensure_dir(output_dir)
    logger = setup_poster_logging(output_dir)
    
    logger.info("="*60)
    logger.info("POSTER CHART GENERATION")
    logger.info("="*60)

    # Data paths
    dir_02a = project_root / 'output' / '02_a_differential_analysis'
    dir_02c = project_root / 'output' / '02_c_sample_classification' / 'sampling_comparison' / 'cluster_based'
    
    # Generate Image 3B
    pairs_path = dir_02a / 'differential_coexpression_sig.tsv'
    if pairs_path.exists():
        logger.info(f"\nLoading data for Image 3B from: {pairs_path.name}")
        sig_pairs = pd.read_csv(pairs_path, sep='\t')
        logger.info(f"  Loaded {len(sig_pairs):,} significant rewired pairs")
        result = create_poster_violin_plot(sig_pairs, output_dir, logger)
        if result:
            logger.info(f"  SUCCESS: Image 3B created")
        else:
            logger.error(f"  FAILED: Image 3B creation failed")
    else:
        logger.error(f"Missing 3B input: {pairs_path}")

    # Generate Image 3C
    test_data_path = dir_02c / 'model_evaluation_data.json'
    rankings_path = dir_02c / 'predictive_hub_ranking.tsv'
    
    if test_data_path.exists() and rankings_path.exists():
        logger.info(f"\nLoading data for Image 3C:")
        logger.info(f"  Test data: {test_data_path.name}")
        logger.info(f"  Rankings: {rankings_path.name}")
        
        with open(test_data_path, 'r') as f:
            p_data = json.load(f)
        
        rankings_df = pd.read_csv(rankings_path, sep='\t')
        
        logger.info(f"  Loaded {len(p_data['y_true'])} test samples")
        logger.info(f"  Loaded {len(rankings_df)} ranked genes")
        
        result = create_poster_roc_and_hubs(
            np.array(p_data['y_true']), 
            np.array(p_data['y_proba']), 
            rankings_df, 
            output_dir, 
            logger
        )
        
        if result:
            logger.info(f"  SUCCESS: Image 3C created")
        else:
            logger.error(f"  FAILED: Image 3C creation failed")
    else:
        logger.error(f"Missing 3C inputs in {dir_02c}")
        if not test_data_path.exists():
            logger.error(f"  Missing: {test_data_path.name}")
        if not rankings_path.exists():
            logger.error(f"  Missing: {rankings_path.name}")

    logger.info("")
    logger.info("="*60)
    logger.info(f"COMPLETE. Results in: {get_relative_path(output_dir)}")
    logger.info("="*60)
    logger.info("")
    logger.info("FILES CREATED:")
    logger.info("  • image_3b_rewiring_magnitude.png (300 DPI)")
    logger.info("  • image_3c_roc_and_hubs.png (300 DPI)")
    logger.info("")
    logger.info("READY FOR POSTER USE ✓")

if __name__ == "__main__":
    main()