# utils/chart_advanced.py

# utils/chart_advanced.py

"""
Advanced Chart Visualization Utilities for Network Analysis

This module provides specialized visualization functions for co-expression network
analysis, designed for publication-quality output and thesis presentation. Functions
support high-resolution rendering, side-by-side comparisons, and biologically-informed
hub gene selection.

Key Features:
    - Smoothed degree distribution overlays with KDE
    - High-resolution individual hub ego network visualizations
    - Side-by-side paired comparisons (Normal | Tumor)
    - Rank-degree topology plots (log-log scale)
    - "Goldilocks" hub pair selection (optimal visual contrast)
    - Breast cancer gene hub identification using gene_info classification
    - Differential hub overlay with unified side-by-side layout

Functions:
    smooth_degree_distribution: Apply KDE smoothing to degree distribution data
    create_degree_distribution_overlay: Smoothed overlay comparison of tumor vs normal degrees
    create_rank_degree_plot: Log-log rank-degree plot showing network hierarchy
    get_hub_ego_network: Extract k-hop ego network subgraph for a hub gene
    find_goldilocks_hub_pairs: Find hub genes with optimal visual contrast (5-20× ratio)
    create_paired_hub_comparison: Create side-by-side comparison (Normal | Tumor)
    find_breast_cancer_hub_pairs: Find known breast cancer hub genes with rewiring
    create_differential_hub_overlay: Side-by-side differential comparison for any hub

Design Philosophy:
    - All paired comparisons use consistent side-by-side layout
    - Light gray edges (#CCCCCC) for clean professional appearance
    - 300 DPI for paired charts (screen quality, reasonable file size)
    - Unified node colors: Blue (normal), Orange/Red (tumor)
    - JSON metadata saved alongside all visualizations

Usage:
    These functions are called by 01_c_network_visualization.py to generate
    comparative hub visualizations showing network rewiring in cancer.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
import json

# utils/chart_advanced.py - ADD THIS IMPORT AT THE TOP
from utils.color_scheme import (
    NORMAL_NODE, NORMAL_HUB, NORMAL_EDGE, NORMAL_FILL, NORMAL_LINE, NORMAL_VLINE, NORMAL_TITLE,
    TUMOR_NODE, TUMOR_HUB, TUMOR_EDGE, TUMOR_FILL, TUMOR_LINE, TUMOR_VLINE, TUMOR_TITLE,
    EDGE_GRAY, TIER_BREAST, TIER_CANCER, TIER_NOVEL
)

def smooth_degree_distribution(degrees, bandwidth=50, num_points=500):
    """
    Apply kernel density estimation (KDE) smoothing to degree distribution.
    
    Args:
        degrees: List or array of degree values
        bandwidth: KDE bandwidth parameter (higher = smoother)
        num_points: Number of points for the smoothed curve
        
    Returns:
        Tuple of (x_smooth, y_smooth) arrays for plotting
    """
    if len(degrees) == 0:
        return np.array([]), np.array([])
    
    # Remove zeros for better KDE estimation
    degrees_nonzero = [d for d in degrees if d > 0]
    if len(degrees_nonzero) == 0:
        degrees_nonzero = [0.1]  # Fallback
    
    # Create KDE
    kde = stats.gaussian_kde(degrees_nonzero, bw_method=bandwidth/np.std(degrees_nonzero))
    
    # Generate smooth curve
    x_min, x_max = min(degrees_nonzero), max(degrees_nonzero)
    x_smooth = np.linspace(x_min, x_max, num_points)
    y_smooth = kde(x_smooth)
    
    return x_smooth, y_smooth



def create_degree_distribution_overlay(
    tumor_degrees, 
    normal_degrees, 
    title, 
    output_path,
    subtitle=None,
    layout='wide',
    normalize=True,
    show_medians=True,
    dpi=300
):
    """
    Create smoothed overlay comparison chart for degree distributions.
    
    This function generates a publication-quality overlay plot showing tumor vs normal
    degree distributions on the same scale, making direct comparison easy.
    
    Args:
        tumor_degrees: List of degree values for tumor network
        normal_degrees: List of degree values for normal network
        title: Main plot title
        output_path: Path to save PNG file
        subtitle: Optional subtitle text
        layout: 'wide' (16:9) or 'tall' (4:3) aspect ratio
        normalize: If True, normalize frequencies to percentage
        show_medians: If True, add vertical lines at median values
        dpi: Resolution for output (300 = publication quality)
        
    Returns:
        Dict with metadata about the generated chart
    """
    # Determine figure size based on layout
    if layout == 'wide':
        figsize = (12, 6)
    else:
        figsize = (10, 8)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Calculate degree ranges
    tumor_nonzero = [d for d in tumor_degrees if d > 0]
    normal_nonzero = [d for d in normal_degrees if d > 0]
    
    # Determine common x-axis range
    max_degree = max(
        max(tumor_nonzero) if tumor_nonzero else 600,
        min(max(normal_nonzero) if normal_nonzero else 6000, 6000)
    )
    
    # Create histograms for area fill
    bins = np.linspace(0, max_degree, 100)
    
    # Tumor distribution
    tumor_hist, tumor_bin_edges = np.histogram(tumor_degrees, bins=bins)
    tumor_bin_centers = (tumor_bin_edges[:-1] + tumor_bin_edges[1:]) / 2
    
    # Normal distribution  
    normal_hist, normal_bin_edges = np.histogram(normal_degrees, bins=bins)
    normal_bin_centers = (normal_bin_edges[:-1] + normal_bin_edges[1:]) / 2
    
    # Normalize if requested
    if normalize:
        tumor_hist = tumor_hist / len(tumor_degrees) * 100
        normal_hist = normal_hist / len(normal_degrees) * 100
        ylabel = 'Frequency (%)'
    else:
        ylabel = 'Frequency (Count)'
    
    # Smooth the histograms for better visualization
    tumor_smooth = gaussian_filter1d(tumor_hist, sigma=2)
    normal_smooth = gaussian_filter1d(normal_hist, sigma=2)
    
    # Plot filled areas with transparency - USING COLOR CONSTANTS
    ax.fill_between(tumor_bin_centers, tumor_smooth, alpha=0.3, color=TUMOR_FILL, label='Tumor Network')
    ax.fill_between(normal_bin_centers, normal_smooth, alpha=0.3, color=NORMAL_FILL, label='Normal Network')
    
    # Plot lines on top for clarity
    ax.plot(tumor_bin_centers, tumor_smooth, color=TUMOR_LINE, linewidth=2.5, alpha=0.9)
    ax.plot(normal_bin_centers, normal_smooth, color=NORMAL_LINE, linewidth=2.5, alpha=0.9)
    
    # Add median lines if requested
    if show_medians:
        tumor_median = np.median(tumor_degrees)
        normal_median = np.median(normal_degrees)
        
        ax.axvline(tumor_median, color=TUMOR_VLINE, linestyle='--', linewidth=2, alpha=0.7,
                   label=f'Tumor Median: {tumor_median:.0f}')
        ax.axvline(normal_median, color=NORMAL_VLINE, linestyle='--', linewidth=2, alpha=0.7,
                   label=f'Normal Median: {normal_median:.0f}')
    
    # Styling
    ax.set_xlabel('Degree (Number of Connections)', fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    if subtitle:
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, 
                ha='center', fontsize=12, style='italic')
    
    # Legend
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Set x-axis limit to focus on tumor range primarily
    ax.set_xlim(0, max_degree)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # Return metadata
    return {
        'chart_type': 'degree_distribution_overlay',
        'tumor_median': float(np.median(tumor_degrees)),
        'normal_median': float(np.median(normal_degrees)),
        'tumor_mean': float(np.mean(tumor_degrees)),
        'normal_mean': float(np.mean(normal_degrees)),
        'max_degree_shown': int(max_degree),
        'normalized': normalize,
        'layout': layout
    }



def get_hub_ego_network(network, hub_gene, k=1, max_neighbors=None, weight_based=True):
    """
    Extract k-hop ego network subgraph for a hub gene.
    
    Args:
        network: NetworkX graph object
        hub_gene: Gene identifier (node) to extract ego network for
        k: Number of hops (1 = direct neighbors only)
        max_neighbors: Maximum number of neighbors to include (None = all)
        weight_based: If True and max_neighbors set, select highest-weight neighbors
        
    Returns:
        NetworkX subgraph representing the ego network
    """
    if hub_gene not in network:
        solo_g = nx.Graph()
        solo_g.add_node(hub_gene)
        return solo_g
    
    neighbors = list(network.neighbors(hub_gene))
    
    if not neighbors:
        solo_g = nx.Graph()
        solo_g.add_node(hub_gene)
        return solo_g

    # Filter to top N neighbors by weight if requested
    if max_neighbors and len(neighbors) > max_neighbors:
        first_neighbor = neighbors[0]
        has_weights = 'weight' in network[hub_gene][first_neighbor]
        
        if weight_based and has_weights:
            neighbors_with_weights = []
            for n in neighbors:
                weight = network[hub_gene][n].get('weight', 1.0)
                neighbors_with_weights.append((n, abs(weight)))
            top_neighbors = sorted(neighbors_with_weights, key=lambda x: x[1], reverse=True)[:max_neighbors]
            neighbors_to_keep = [n for n, w in top_neighbors]
        else:
            import random
            neighbors_to_keep = random.sample(neighbors, max_neighbors)
    else:
        neighbors_to_keep = neighbors

    nodes_to_keep = [hub_gene] + neighbors_to_keep
    return network.subgraph(nodes_to_keep).copy()


def create_rank_degree_plot(tumor_degrees, normal_degrees, title, output_path, json_path, dpi=300):
    """
    Generates a Rank-Degree (Log-Log) plot and saves requested metadata JSON.
    Includes the actual data points (ranks and degrees) in the JSON.
    """
    import json
    import numpy as np
    import matplotlib.pyplot as plt

    # 1. Sort degrees descending for rank-ordering
    n_sorted = np.sort(normal_degrees)[::-1]
    t_sorted = np.sort(tumor_degrees)[::-1]
    
    # 2. Generate Rank indices (1 to N)
    n_ranks = np.arange(1, len(n_sorted) + 1)
    t_ranks = np.arange(1, len(t_sorted) + 1)
    
    # 3. Create the Log-Log Plot - USING COLOR CONSTANTS
    plt.figure(figsize=(10, 7), dpi=dpi)
    plt.loglog(n_ranks, n_sorted, label='Normal (Healthy Hierarchy)', color=NORMAL_LINE, linewidth=3, alpha=0.8)
    plt.loglog(t_ranks, t_sorted, label='Tumor (Structural Collapse)', color=TUMOR_LINE, linewidth=3, alpha=0.8)
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Gene Rank (Log10 scale)', fontsize=14)
    plt.ylabel('Degree (Log10 scale)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()

    # 4. Save comprehensive metadata + Plot Data
    metadata = {
        "metadata": {
            "chart_type": "rank_degree_distribution_log_log",
            "normal_median": float(np.median(normal_degrees)),
            "tumor_median": float(np.median(tumor_degrees)),
            "connectivity_loss_ratio": float(np.mean(normal_degrees) / np.mean(tumor_degrees)) if np.mean(tumor_degrees) > 0 else 0
        },
        "description": {
            "chart_type": "rank_degree_distribution_log_log",
            "purpose": "Topological assessment of network hierarchy using Zipf-style rank ordering",
            "interpretation": "The 'Topological Gap' between the blue and red lines quantifies the systemic loss of network coordination in cancer.",
            "visualization_features": [
                "Log-Log scale for power-law detection",
                "Descending rank-order sorting"
            ]
        },
        "plot_data": {
            "normal": {
                "ranks": n_ranks.tolist(),
                "degrees": n_sorted.tolist()
            },
            "tumor": {
                "ranks": t_ranks.tolist(),
                "degrees": t_sorted.tolist()
            }
        }
    }
    
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    return metadata



def find_goldilocks_hub_pairs(normal_network, tumor_network, 
                               min_normal_deg=2000,
                               min_tumor_deg=300,
                               max_tumor_deg=2000,
                               min_ratio=3.0,
                               max_ratio=10.0,
                               num_pairs=3):
    """
    Find hub genes with visually optimal contrast across conditions.
    
    "Goldilocks" criteria: Not too extreme (5000 vs 5), not too similar (5000 vs 4800),
    but just right for clear visual comparison (e.g., 5000 vs 800 = 6.25× ratio).
    
    Args:
        normal_network: NetworkX graph for normal tissue
        tumor_network: NetworkX graph for tumor tissue
        min_normal_deg: Minimum degree in normal (must be substantial hub)
        min_tumor_deg: Minimum degree in tumor (must be visible, not isolated)
        max_tumor_deg: Maximum degree in tumor (avoid too similar to normal)
        min_ratio: Minimum normal/tumor ratio (avoid too similar)
        max_ratio: Maximum normal/tumor ratio (avoid too extreme)
        num_pairs: Number of pairs to return
        
    Returns:
        List of (gene, normal_deg, tumor_deg, ratio) tuples, sorted by ratio descending
    """
    candidates = []
    
    # Find genes that exist in both networks
    common_genes = set(normal_network.nodes()) & set(tumor_network.nodes())
    
    for gene in common_genes:
        normal_deg = normal_network.degree(gene)
        tumor_deg = tumor_network.degree(gene)
        
        # Apply Goldilocks criteria
        if (normal_deg >= min_normal_deg and 
            min_tumor_deg <= tumor_deg <= max_tumor_deg):
            
            ratio = normal_deg / tumor_deg if tumor_deg > 0 else float('inf')
            
            if min_ratio <= ratio <= max_ratio:
                candidates.append((gene, normal_deg, tumor_deg, ratio))
    
    # Sort by ratio descending (most dramatic but still reasonable)
    candidates.sort(key=lambda x: x[3], reverse=True)
    
    return candidates[:num_pairs]


def create_paired_hub_comparison(
    normal_network,
    tumor_network,
    hub_gene,
    output_path,
    threshold=0.7,
    max_neighbors=200,
    dpi=300,
    hub_info=None
):
    """
    Create side-by-side comparison of same gene in both conditions.
    
    Layout: [Normal Network] | [Tumor Network]
    
    This shows the same hub gene in both conditions on a single chart,
    making visual comparison immediate and clear.
    
    Args:
        normal_network: NetworkX graph for normal tissue
        tumor_network: NetworkX graph for tumor tissue
        hub_gene: Gene identifier (format: ENSG...|SYMBOL)
        output_path: Path to save PNG file
        threshold: Correlation threshold used
        max_neighbors: Maximum neighbors to show
        dpi: Resolution (300 = good screen quality, lighter files)
        
    Returns:
        Dict with metadata including degree loss %, ratio, etc.
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    from pathlib import Path
    
    gene_symbol = hub_gene.split('|')[1] if '|' in hub_gene else hub_gene
    
    # Get ego networks for both conditions
    ego_normal = get_hub_ego_network(normal_network, hub_gene, k=1, max_neighbors=max_neighbors)
    ego_tumor = get_hub_ego_network(tumor_network, hub_gene, k=1, max_neighbors=max_neighbors)
    
    # Get degrees
    normal_deg = normal_network.degree(hub_gene) if hub_gene in normal_network else 0
    tumor_deg = tumor_network.degree(hub_gene) if hub_gene in tumor_network else 0
    
    # Create unified layout based on normal network (usually larger)
    if ego_normal.number_of_nodes() > 1:
        pos_normal = nx.spring_layout(ego_normal, k=0.5, seed=42)
    else:
        pos_normal = {hub_gene: (0, 0)}
    
    # Apply same positions to tumor network (for matching nodes)
    pos_tumor = {n: pos_normal[n] for n in ego_tumor.nodes() if n in pos_normal}
    missing = [n for n in ego_tumor.nodes() if n not in pos_tumor]
    if missing:
        pos_tumor = nx.spring_layout(ego_tumor, pos=pos_tumor, fixed=list(pos_tumor.keys()), k=0.5, seed=42)
    
    # Create figure with 2 subplots side-by-side
    fig, (ax_normal, ax_tumor) = plt.subplots(1, 2, figsize=(16, 8), dpi=dpi)
    
    # === LEFT: NORMAL NETWORK ===
    # Draw edges (light gray as requested)
    if ego_normal.number_of_edges() > 0:
        nx.draw_networkx_edges(ego_normal, pos_normal, ax=ax_normal, 
                              edge_color=EDGE_GRAY, width=0.5, alpha=0.5)
    
    # Draw neighbor nodes - USING NORMAL_NODE CONSTANT
    others_n = [n for n in ego_normal.nodes() if n != hub_gene]
    if others_n:
        nx.draw_networkx_nodes(ego_normal, pos_normal, nodelist=others_n, ax=ax_normal,
                               node_color=NORMAL_NODE, node_size=80, alpha=0.8, edgecolors='white')
    
    # Draw hub node - USING NORMAL_HUB CONSTANT
    if hub_gene in ego_normal.nodes():
        nx.draw_networkx_nodes(ego_normal, pos_normal, nodelist=[hub_gene], ax=ax_normal,
                               node_color=NORMAL_HUB, node_size=400, alpha=0.95, 
                               edgecolors='white', linewidths=2)
    
    ax_normal.set_xlim(-1.3, 1.3)
    ax_normal.set_ylim(-1.3, 1.3)
    ax_normal.set_title(f'NORMAL: {gene_symbol}\nDegree: {normal_deg:,}', 
                        fontsize=16, fontweight='bold', pad=20, color=NORMAL_TITLE)
    ax_normal.axis('off')
    
    # === RIGHT: TUMOR NETWORK ===
    # Draw edges (light gray as requested)
    if ego_tumor.number_of_edges() > 0:
        nx.draw_networkx_edges(ego_tumor, pos_tumor, ax=ax_tumor,
                              edge_color=EDGE_GRAY, width=0.5, alpha=0.5)
    
    # Draw neighbor nodes - USING TUMOR_NODE CONSTANT (updated from TUMOR_NODE = #e8623a)
    others_t = [n for n in ego_tumor.nodes() if n != hub_gene]
    if others_t:
        nx.draw_networkx_nodes(ego_tumor, pos_tumor, nodelist=others_t, ax=ax_tumor,
                               node_color=TUMOR_NODE, node_size=80, alpha=0.8, edgecolors='white')
    
    # Draw hub node - USING TUMOR_HUB CONSTANT (keeping warm red #c0392b)
    if hub_gene in ego_tumor.nodes():
        nx.draw_networkx_nodes(ego_tumor, pos_tumor, nodelist=[hub_gene], ax=ax_tumor,
                               node_color=TUMOR_HUB, node_size=400, alpha=0.95,
                               edgecolors='white', linewidths=2)
    
    ax_tumor.set_xlim(-1.3, 1.3)
    ax_tumor.set_ylim(-1.3, 1.3)
    ax_tumor.set_title(f'TUMOR: {gene_symbol}\nDegree: {tumor_deg:,}', 
                       fontsize=16, fontweight='bold', pad=20, color=TUMOR_TITLE)  # Now uses #c0392b for consistency
    ax_tumor.axis('off')
    
    # Determine appropriate title based on degree patterns
    if normal_deg == 0 and tumor_deg == 0:
        chart_title = f'{gene_symbol}: Rewired Below Threshold (δ-connectivity driven)'
    elif normal_deg == 0 and tumor_deg > 0:
        chart_title = f'{gene_symbol}: Ectopic Activation (0 → {tumor_deg} connections)'
    elif tumor_deg == 0 and normal_deg > 0:
        chart_title = f'{gene_symbol}: Connectivity Collapse (100.0% Loss)'
    else:
        change_pct = (tumor_deg - normal_deg) / normal_deg * 100
        if change_pct >= 0:
            chart_title = f'{gene_symbol}: Connectivity Gain (+{change_pct:.1f}%)'
        else:
            chart_title = f'{gene_symbol}: Connectivity Collapse ({abs(change_pct):.1f}% Loss)'

    fig.suptitle(chart_title, fontsize=20, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # Determine chart type, signed change, and interpretation
    if normal_deg == 0 and tumor_deg == 0:
        chart_type = 'below_threshold_rewired'
        degree_change = 0
        percent_change = None
        interpretation = (
            f"{gene_symbol} has no co-expression edges above the |r|≥{threshold} threshold "
            f"in either normal or tumor tissue. It is not visually connected in either network "
            f"but ranks highly because its pairwise correlations change substantially across "
            f"many gene pairs (delta_connectivity={hub_info.get('delta_connectivity', 'N/A') if hub_info else 'N/A'}), "
            f"making it highly discriminatory in the ML model despite being 'invisible' "
            f"in the thresholded network graph."
        )
    elif normal_deg == 0 and tumor_deg > 0:
        chart_type = 'ectopic_activation'
        degree_change = tumor_deg
        percent_change = None
        interpretation = (
            f"{gene_symbol} is silent in normal breast tissue (0 co-expression partners "
            f"above |r|≥{threshold}) but gains {tumor_deg} significant co-expression "
            f"connections in tumor tissue. This ectopic activation pattern — expression "
            f"in an inappropriate tissue context — suggests the gene becomes incorporated "
            f"into the tumor regulatory network during carcinogenesis."
        )
    elif tumor_deg == 0 and normal_deg > 0:
        chart_type = 'connectivity_collapse'
        degree_change = -normal_deg
        percent_change = -100.0
        interpretation = (
            f"{gene_symbol} maintains {normal_deg} co-expression partners in normal tissue "
            f"(above |r|≥{threshold}) but loses all connections in tumor tissue. "
            f"This complete connectivity collapse suggests the gene's regulatory relationships "
            f"are entirely disrupted during carcinogenesis — it is no longer coordinated "
            f"with its normal co-expression partners."
        )
    else:
        degree_change = tumor_deg - normal_deg
        percent_change = round(degree_change / normal_deg * 100, 2)
        if degree_change > 0:
            chart_type = 'connectivity_gain'
            interpretation = (
                f"{gene_symbol} gains {degree_change} co-expression connections in tumor "
                f"tissue compared to normal ({normal_deg} → {tumor_deg} partners above "
                f"|r|≥{threshold}, +{percent_change:.1f}%). This increased connectivity "
                f"suggests the gene becomes more centrally embedded in the tumor network."
            )
        else:
            chart_type = 'connectivity_loss'
            interpretation = (
                f"{gene_symbol} loses {abs(degree_change)} co-expression connections in tumor "
                f"tissue compared to normal ({normal_deg} → {tumor_deg} partners above "
                f"|r|≥{threshold}, {percent_change:.1f}%). This partial connectivity loss "
                f"suggests disrupted but not eliminated regulatory coordination."
            )

    # JSON-safe ratio (never Infinity)
    ratio = round(normal_deg / tumor_deg, 3) if tumor_deg > 0 else None

    return {
        # Gene identity
        "hub_gene":           hub_gene,
        "gene_symbol":        gene_symbol,
        "gene_description":   hub_info.get('gene_description', '') if hub_info else '',

        # Chart classification
        "chart_type":         chart_type,
        "interpretation":     interpretation,

        # Degree metrics (signed, JSON-safe)
        "normal_degree":      int(normal_deg),
        "tumor_degree":       int(tumor_deg),
        "degree_change":      int(degree_change),      # signed: negative=loss, positive=gain
        "percent_change":     percent_change,          # None if normal_deg==0 (undefined)
        "ratio":              ratio,                   # None if tumor_deg==0 (avoids Infinity)

        # Scoring context
        "delta_connectivity": round(hub_info.get('delta_connectivity', 0), 4) if hub_info else None,
        "feature_importance": round(hub_info.get('feature_importance', 0), 6) if hub_info else None,
        "composite_score":    round(hub_info.get('composite_score', 0), 4) if hub_info else None,
        "cancer_relevance":   hub_info.get('cancer_relevance', '') if hub_info else '',

        # Visualization metadata
        "correlation_threshold": float(threshold),
        "normal_neighbors_shown": len(others_n) if 'others_n' in locals() else None,
        "tumor_neighbors_shown":  len(others_t) if 'others_t' in locals() else None,
        "output_file":        str(output_path),
        "dpi":                int(dpi)
    }



def find_breast_cancer_hub_pairs(
    normal_network, 
    tumor_network,
    config,
    combined_gene_data=None,
    min_normal_deg=1500,
    min_tumor_deg=100,
    max_tumor_deg=3000,
    min_ratio=2.0,
    max_ratio=50.0,
    num_pairs=3
):
    """
    Find hub genes that are known breast cancer genes with visual contrast.
    
    This prioritizes genes that:
    1. Are classified as breast_cancer in gene_info
    2. Show substantial degree change (rewiring)
    3. Have reasonable visual contrast
    
    Args:
        normal_network: NetworkX graph for normal tissue
        tumor_network: NetworkX graph for tumor tissue
        config: Project configuration dict
        combined_gene_data: Pre-loaded gene info (optional)
        min_normal_deg: Minimum degree in normal
        min_tumor_deg: Minimum degree in tumor (very permissive)
        max_tumor_deg: Maximum degree in tumor (very permissive)
        min_ratio: Minimum ratio (permissive for known genes)
        max_ratio: Maximum ratio (very permissive)
        num_pairs: Number of pairs to return
        
    Returns:
        List of (gene, normal_deg, tumor_deg, ratio, division) tuples
    """
    from utils.genes import load_combined_gene_info, normalize_gene_id, get_gene_info
    
    # Load gene info if not provided
    if combined_gene_data is None:
        combined_gene_data = load_combined_gene_info(config)
    
    if not combined_gene_data:
        print("Warning: Could not load gene info, skipping breast cancer gene selection")
        return []
    
    # Get all breast cancer genes from gene_info
    breast_cancer_genes = set()
    if 'breast_cancer' in combined_gene_data:
        for gene_key in combined_gene_data['breast_cancer'].keys():
            normalized = normalize_gene_id(gene_key)
            breast_cancer_genes.add(normalized)
            breast_cancer_genes.add(gene_key)  # Keep both versions
    
    # Find candidates
    candidates = []
    common_genes = set(normal_network.nodes()) & set(tumor_network.nodes())
    
    for gene in common_genes:
        # Check if this gene is a known breast cancer gene
        normalized = normalize_gene_id(gene)
        
        is_breast_cancer = (gene in breast_cancer_genes or 
                           normalized in breast_cancer_genes)
        
        if not is_breast_cancer:
            continue
        
        normal_deg = normal_network.degree(gene)
        tumor_deg = tumor_network.degree(gene)
        
        # Apply VERY PERMISSIVE criteria for breast cancer genes
        if (normal_deg >= min_normal_deg and 
            min_tumor_deg <= tumor_deg <= max_tumor_deg):
            
            ratio = normal_deg / tumor_deg if tumor_deg > 0 else float('inf')
            
            if min_ratio <= ratio <= max_ratio:
                candidates.append((gene, normal_deg, tumor_deg, ratio, 'breast_cancer'))
    
    # Sort by a combination of ratio and normal degree
    # Prioritize genes with both high normal degree AND high ratio
    candidates.sort(key=lambda x: (x[3] * (x[1] / 1000)), reverse=True)
    
    return candidates[:num_pairs]


def create_differential_hub_overlay(
    normal_net, 
    tumor_net, 
    hub_gene, 
    output_path,
    threshold=0.7, 
    max_neighbors=200,
    dpi=300,
    hub_info=None
):
    """
    Side-by-side comparison: Normal | Tumor (same as goldilocks layout).
    
    This replaces the confusing 3D overlay with a clean side-by-side view
    matching the goldilocks and breast cancer chart style.
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    from pathlib import Path
    
    gene_symbol = hub_gene.split('|')[1] if '|' in hub_gene else hub_gene
    
    # Get ego networks for both conditions
    ego_normal = get_hub_ego_network(normal_net, hub_gene, k=1, max_neighbors=max_neighbors)
    ego_tumor = get_hub_ego_network(tumor_net, hub_gene, k=1, max_neighbors=max_neighbors)
    
    # Get degrees
    normal_deg = normal_net.degree(hub_gene) if hub_gene in normal_net else 0
    tumor_deg = tumor_net.degree(hub_gene) if hub_gene in tumor_net else 0
    
    # Create unified layout based on normal network (usually larger)
    if ego_normal.number_of_nodes() > 1:
        pos_normal = nx.spring_layout(ego_normal, k=0.5, seed=42)
    else:
        pos_normal = {hub_gene: (0, 0)}
    
    # Apply same positions to tumor network (for matching nodes)
    pos_tumor = {n: pos_normal[n] for n in ego_tumor.nodes() if n in pos_normal}
    missing = [n for n in ego_tumor.nodes() if n not in pos_tumor]
    if missing:
        pos_tumor = nx.spring_layout(ego_tumor, pos=pos_tumor, 
                                     fixed=list(pos_tumor.keys()), k=0.5, seed=42)
    
    # Create figure with 2 subplots side-by-side
    fig, (ax_normal, ax_tumor) = plt.subplots(1, 2, figsize=(16, 8), dpi=dpi)
    
    # === LEFT: NORMAL NETWORK ===
    if ego_normal.number_of_edges() > 0:
        nx.draw_networkx_edges(ego_normal, pos_normal, ax=ax_normal, 
                              edge_color=EDGE_GRAY, width=0.5, alpha=0.5)
    
    others_n = [n for n in ego_normal.nodes() if n != hub_gene]
    if others_n:
        nx.draw_networkx_nodes(ego_normal, pos_normal, nodelist=others_n, ax=ax_normal,
                               node_color=NORMAL_NODE, node_size=80, alpha=0.8, 
                               edgecolors='white')
    
    if hub_gene in ego_normal.nodes():
        nx.draw_networkx_nodes(ego_normal, pos_normal, nodelist=[hub_gene], ax=ax_normal,
                               node_color=NORMAL_HUB, node_size=400, alpha=0.95, 
                               edgecolors='white', linewidths=2)
    
    ax_normal.set_xlim(-1.3, 1.3)
    ax_normal.set_ylim(-1.3, 1.3)
    ax_normal.set_title(f'NORMAL: {gene_symbol}\nDegree: {normal_deg:,}', 
                        fontsize=16, fontweight='bold', pad=20, color=NORMAL_TITLE)
    ax_normal.axis('off')
    
    # === RIGHT: TUMOR NETWORK ===
    if ego_tumor.number_of_edges() > 0:
        nx.draw_networkx_edges(ego_tumor, pos_tumor, ax=ax_tumor,
                              edge_color=EDGE_GRAY, width=0.5, alpha=0.5)
    
    others_t = [n for n in ego_tumor.nodes() if n != hub_gene]
    if others_t:
        nx.draw_networkx_nodes(ego_tumor, pos_tumor, nodelist=others_t, ax=ax_tumor,
                               node_color=TUMOR_NODE, node_size=80, alpha=0.8, 
                               edgecolors='white')
    
    if hub_gene in ego_tumor.nodes():
        nx.draw_networkx_nodes(ego_tumor, pos_tumor, nodelist=[hub_gene], ax=ax_tumor,
                               node_color=TUMOR_HUB, node_size=400, alpha=0.95,
                               edgecolors='white', linewidths=2)
    
    ax_tumor.set_xlim(-1.3, 1.3)
    ax_tumor.set_ylim(-1.3, 1.3)
    ax_tumor.set_title(f'TUMOR: {gene_symbol}\nDegree: {tumor_deg:,}', 
                       fontsize=16, fontweight='bold', pad=20, color=TUMOR_TITLE)
    ax_tumor.axis('off')
    
    # Determine appropriate title based on degree patterns
    if normal_deg == 0 and tumor_deg == 0:
        chart_title = f'{gene_symbol}: Rewired Below Threshold (δ-connectivity driven)'
    elif normal_deg == 0 and tumor_deg > 0:
        chart_title = f'{gene_symbol}: Ectopic Activation (0 → {tumor_deg} connections)'
    elif tumor_deg == 0 and normal_deg > 0:
        chart_title = f'{gene_symbol}: Connectivity Collapse (100.0% Loss)'
    else:
        change_pct = (tumor_deg - normal_deg) / normal_deg * 100
        if change_pct >= 0:
            chart_title = f'{gene_symbol}: Connectivity Gain (+{change_pct:.1f}%)'
        else:
            chart_title = f'{gene_symbol}: Connectivity Collapse ({abs(change_pct):.1f}% Loss)'

    fig.suptitle(chart_title, fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # Determine chart type, signed change, and interpretation
    if normal_deg == 0 and tumor_deg == 0:
        chart_type = 'below_threshold_rewired'
        degree_change = 0
        percent_change = None
        interpretation = (
            f"{gene_symbol} has no co-expression edges above the |r|≥{threshold} threshold "
            f"in either normal or tumor tissue. It is not visually connected in either network "
            f"but ranks highly because its pairwise correlations change substantially across "
            f"many gene pairs (delta_connectivity={hub_info.get('delta_connectivity', 'N/A') if hub_info else 'N/A'}), "
            f"making it highly discriminatory in the ML model despite being 'invisible' "
            f"in the thresholded network graph."
        )
    elif normal_deg == 0 and tumor_deg > 0:
        chart_type = 'ectopic_activation'
        degree_change = tumor_deg
        percent_change = None
        interpretation = (
            f"{gene_symbol} is silent in normal breast tissue (0 co-expression partners "
            f"above |r|≥{threshold}) but gains {tumor_deg} significant co-expression "
            f"connections in tumor tissue. This ectopic activation pattern — expression "
            f"in an inappropriate tissue context — suggests the gene becomes incorporated "
            f"into the tumor regulatory network during carcinogenesis."
        )
    elif tumor_deg == 0 and normal_deg > 0:
        chart_type = 'connectivity_collapse'
        degree_change = -normal_deg
        percent_change = -100.0
        interpretation = (
            f"{gene_symbol} maintains {normal_deg} co-expression partners in normal tissue "
            f"(above |r|≥{threshold}) but loses all connections in tumor tissue. "
            f"This complete connectivity collapse suggests the gene's regulatory relationships "
            f"are entirely disrupted during carcinogenesis — it is no longer coordinated "
            f"with its normal co-expression partners."
        )
    else:
        degree_change = tumor_deg - normal_deg
        percent_change = round(degree_change / normal_deg * 100, 2)
        if degree_change > 0:
            chart_type = 'connectivity_gain'
            interpretation = (
                f"{gene_symbol} gains {degree_change} co-expression connections in tumor "
                f"tissue compared to normal ({normal_deg} → {tumor_deg} partners above "
                f"|r|≥{threshold}, +{percent_change:.1f}%). This increased connectivity "
                f"suggests the gene becomes more centrally embedded in the tumor network."
            )
        else:
            chart_type = 'connectivity_loss'
            interpretation = (
                f"{gene_symbol} loses {abs(degree_change)} co-expression connections in tumor "
                f"tissue compared to normal ({normal_deg} → {tumor_deg} partners above "
                f"|r|≥{threshold}, {percent_change:.1f}%). This partial connectivity loss "
                f"suggests disrupted but not eliminated regulatory coordination."
            )

    # JSON-safe ratio (never Infinity)
    ratio = round(normal_deg / tumor_deg, 3) if tumor_deg > 0 else None

    return {
        # Gene identity
        "hub_gene":           hub_gene,
        "gene_symbol":        gene_symbol,
        "gene_description":   hub_info.get('gene_description', '') if hub_info else '',

        # Chart classification
        "chart_type":         chart_type,
        "interpretation":     interpretation,

        # Degree metrics (signed, JSON-safe)
        "normal_degree":      int(normal_deg),
        "tumor_degree":       int(tumor_deg),
        "degree_change":      int(degree_change),      # signed: negative=loss, positive=gain
        "percent_change":     percent_change,          # None if normal_deg==0 (undefined)
        "ratio":              ratio,                   # None if tumor_deg==0 (avoids Infinity)

        # Scoring context
        "delta_connectivity": round(hub_info.get('delta_connectivity', 0), 4) if hub_info else None,
        "feature_importance": round(hub_info.get('feature_importance', 0), 6) if hub_info else None,
        "composite_score":    round(hub_info.get('composite_score', 0), 4) if hub_info else None,
        "cancer_relevance":   hub_info.get('cancer_relevance', '') if hub_info else '',

        # Visualization metadata
        "correlation_threshold": float(threshold),
        "normal_neighbors_shown": len(others_n) if 'others_n' in locals() else None,
        "tumor_neighbors_shown":  len(others_t) if 'others_t' in locals() else None,
        "output_file":        str(output_path),
        "dpi":                int(dpi)
    }

