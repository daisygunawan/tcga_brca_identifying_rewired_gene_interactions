# utils/color_scheme.py - UPDATED LINE 28
"""
Centralized color scheme for the entire project.
This ensures consistency across all visualizations and makes future changes easy.
"""

# ── TISSUE-LEVEL COLORS (Normal vs Tumor) ──────────────────────────────────
# These colors represent biological tissue types in network visualizations
# Normal = cool blue, Tumor = warm orange-red (as specified in poster)

# Normal tissue colors
NORMAL_NODE      = '#2e86c1'   # peripheral nodes, normal network
NORMAL_HUB       = '#1a5276'   # hub node, normal network  
NORMAL_EDGE      = '#2e86c1'   # edges, normal network
NORMAL_FILL      = '#2e86c1'   # area fill, normal distributions
NORMAL_LINE      = '#2980b9'   # line plots, normal distributions
NORMAL_VLINE     = '#2980b9'   # vertical lines (medians, etc.)
NORMAL_TITLE     = '#1976D2'   # title text for normal panels
NORMAL_BAR       = '#2e86c1'   # bar chart fill for normal

# Tumor tissue colors
TUMOR_NODE       = '#e8623a'   # peripheral nodes, tumor network
TUMOR_HUB        = '#c0392b'   # hub node, tumor network
TUMOR_EDGE       = '#e8623a'   # edges, tumor network
TUMOR_FILL       = '#e8623a'   # area fill, tumor distributions
TUMOR_LINE       = '#c0392b'   # line plots, tumor distributions
TUMOR_VLINE      = '#c0392b'   # vertical lines (medians, etc.)
TUMOR_TITLE      = '#c0392b'   # title text for tumor panels (updated for consistency)
TUMOR_BAR        = '#e8623a'   # bar chart fill for tumor

# Edge color (common for both tissues in certain contexts)
EDGE_GRAY        = '#CCCCCC'   # light gray edges for clean appearance


# ── TIER/CATEGORY COLORS (separate semantic meaning - DO NOT USE FOR TISSUE) ─
# These represent different gene categories, not tissue types
# IMPORTANT: These should NEVER be used for normal vs tumor comparisons

TIER_BREAST      = '#e74c3c'   # breast cancer specific genes
TIER_CANCER      = '#e67e22'   # general cancer genes
TIER_NOVEL       = '#3498db'   # novel/non-cancer genes
TIER_OTHER       = '#95a5a6'   # other/non-cancer genes (light gray)

# Method comparison colors (Spearman vs Pearson)
METHOD_SPEARMAN  = '#2ecc71'   # Spearman correlation (green)
METHOD_PEARSON   = '#95a5a6'   # Pearson correlation (gray)

# Quadrant/group colors for specialized visualizations
QUADRANT_Q1      = '#FF6B6B'   # Gain + Strong baseline
QUADRANT_Q2      = '#3498db'   # Loss + Strong baseline
QUADRANT_Q3      = '#95a5a6'   # Loss + Weak baseline
QUADRANT_Q4      = '#2ecc71'   # Gain + Weak baseline

# Statistical significance
SIG_THRESHOLD    = '#e74c3c'   # significance line/annotation
SIG_HIGHLIGHT    = '#ffffcc'   # highlight background for stats


# ── HELPER FUNCTIONS ────────────────────────────────────────────────────────

def get_tissue_colors(tissue_type):
    """
    Get a complete color dictionary for a tissue type.
    
    Args:
        tissue_type: 'normal' or 'tumor'
        
    Returns:
        Dictionary with all color roles for that tissue
    """
    if tissue_type.lower() == 'normal':
        return {
            'node': NORMAL_NODE,
            'hub': NORMAL_HUB,
            'edge': NORMAL_EDGE,
            'fill': NORMAL_FILL,
            'line': NORMAL_LINE,
            'vline': NORMAL_VLINE,
            'title': NORMAL_TITLE,
            'bar': NORMAL_BAR
        }
    elif tissue_type.lower() == 'tumor':
        return {
            'node': TUMOR_NODE,
            'hub': TUMOR_HUB,
            'edge': TUMOR_EDGE,
            'fill': TUMOR_FILL,
            'line': TUMOR_LINE,
            'vline': TUMOR_VLINE,
            'title': TUMOR_TITLE,
            'bar': TUMOR_BAR
        }
    else:
        raise ValueError(f"Unknown tissue type: {tissue_type}. Use 'normal' or 'tumor'.")


def get_gene_category_colors():
    """
    Get colors for gene categories (cancer relevance).
    These are for tier-based coloring, NOT tissue comparison.
    """
    return {
        'breast_cancer': TIER_BREAST,
        'cancer': TIER_CANCER,
        'non_cancer': TIER_NOVEL,
        'unknown': TIER_OTHER
    }


def print_color_scheme():
    """Utility to print the current color scheme for documentation."""
    print("\n" + "="*60)
    print("PROJECT COLOR SCHEME")
    print("="*60)
    
    print("\nTISSUE COLORS:")
    print("-"*40)
    print(f"Normal:  Node: {NORMAL_NODE}  Hub: {NORMAL_HUB}  Title: {NORMAL_TITLE}")
    print(f"Tumor:   Node: {TUMOR_NODE}  Hub: {TUMOR_HUB}  Title: {TUMOR_TITLE}")
    print(f"Edges:   {EDGE_GRAY} (common)")
    
    print("\nGENE CATEGORY COLORS:")
    print("-"*40)
    print(f"Breast Cancer: {TIER_BREAST}")
    print(f"Other Cancer:  {TIER_CANCER}")
    print(f"Novel/Other:   {TIER_NOVEL}")
    print(f"Unknown:       {TIER_OTHER}")
    
    print("\nMETHOD COMPARISON:")
    print("-"*40)
    print(f"Spearman: {METHOD_SPEARMAN}")
    print(f"Pearson:  {METHOD_PEARSON}")
    
    print("\nQUADRANT COLORS:")
    print("-"*40)
    print(f"Q1 (Gain + Strong): {QUADRANT_Q1}")
    print(f"Q2 (Loss + Strong): {QUADRANT_Q2}")
    print(f"Q3 (Loss + Weak):   {QUADRANT_Q3}")
    print(f"Q4 (Gain + Weak):   {QUADRANT_Q4}")
    print("="*60)


if __name__ == "__main__":
    print_color_scheme()