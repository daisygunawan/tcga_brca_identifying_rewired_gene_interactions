"""
resize_imgs_for_review.py
--------------------------
Resizes Chapter 4 figures to max 1200px (longest side) for Claude review.
Output saved to: code/tests/resized_imgs/

Run from: code/tests/
    python resize_imgs_for_review.py

Output filenames use the full relative path with slashes replaced by underscores,
so the source location is identifiable from the filename alone.
e.g. output/03_a.../cancer_relevance_pie.png
  -> output_03_a_enhanced_hub_analysis_viz_cancer_relevance_pie.png
"""

from pathlib import Path
from PIL import Image

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# Run from code/tests/ — project root is two levels up
PROJECT_ROOT = Path(__file__).parent / ".." / ".."

OUTPUT_DIR = Path(__file__).parent / "resized_imgs"
MAX_DIMENSION = 1200  # px, longest side

IMAGES = [
    # Figure 4.1 — Cancer relevance tier pie chart
    "output/03_a_enhanced_hub_analysis/viz/cancer_relevance_pie.png",

    # Figure 4.2A — Degree distribution overlay (Spearman)
    "output/01_c_network_visualization/spearman/01_spearman_degree_distribution_overlay.png",

    # Figure 4.2B — Rank-degree topology (Spearman)
    "output/01_c_network_visualization/spearman/01_spearman_rank_degree_topology.png",

    # Figure 4.3A — Edge type breakdown (Spearman vs Pearson)
    "output/01_c_network_visualization/method_comparison/01_edge_type_breakdown.png",

    # Figure 4.3B — Hub preservation / consistency analysis
    "output/01_c_network_visualization/method_comparison/02_hub_preservation_analysis.png",

    # Figure 4.3C — Unified performance dashboard
    "output/01_c_network_visualization/method_comparison/03_unified_performance_dashboard.png",

    # Figure 4.3D — Side-by-side distribution comparison
    "output/01_c_network_visualization/method_comparison/04_distribution_comparison.png",

    # Figure 4.4 — Effect size distribution and directional bias
    "output/02_b_dcea_viz_enrich/viz/02_effect_size_distribution_chart.png",

    # Figure 4.5 — Rewiring mechanism quadrant / scatter
    "output/02_b_dcea_viz_enrich/viz/06_rewired_edge_scatter.png",

    # Figure 4.6 — ML sampling method comparison
    "output/02_c_sample_classification/sampling_comparison/comparison_report/sampling_methods_comparison.png",

    # Figure 4.7 — Top 20 genes by delta connectivity (bar chart)
    "output/02_b_dcea_viz_enrich/viz/04_delta_connectivity_bar.png",

    # Figure 4.8 — Hub neighbourhood comparison (2x3 grid, top tumor vs normal hubs)
    "output/01_c_network_visualization/spearman/03_spearman_hub_neighborhoods.png",

    # Figure 4.9 — MGA goldilocks pair (connectivity collapse example)
    "output/01_c_network_visualization/spearman/03_hub_analysis_centric/goldilocks_3_MGA_paired.png",

    # Figure 4.10 — FANCI breast cancer gene pair (connectivity collapse)
    "output/01_c_network_visualization/spearman/03_hub_analysis_centric/breast_cancer_2_FANCI_paired.png",

    # Figures 4.11a-i — Nine consensus hub paired visualizations
    "output/03_a_enhanced_hub_analysis/consensus_hubs/01_overall_1_MAB21L1_paired.png",
    "output/03_a_enhanced_hub_analysis/consensus_hubs/02_overall_2_GSTM5_paired.png",
    "output/03_a_enhanced_hub_analysis/consensus_hubs/03_overall_3_SOBP_paired.png",
    "output/03_a_enhanced_hub_analysis/consensus_hubs/04_breast_cancer_121_GRB7_paired.png",
    "output/03_a_enhanced_hub_analysis/consensus_hubs/05_breast_cancer_155_TP53_paired.png",
    "output/03_a_enhanced_hub_analysis/consensus_hubs/06_breast_cancer_201_CDH1_paired.png",
    "output/03_a_enhanced_hub_analysis/consensus_hubs/07_cancer_13_TMPRSS2_paired.png",
    "output/03_a_enhanced_hub_analysis/consensus_hubs/08_cancer_156_ETV4_paired.png",
    "output/03_a_enhanced_hub_analysis/consensus_hubs/09_cancer_304_DNMT3B_paired.png",

    # Figure 4.12 — Enrichment comparison (all vs cancer vs novel gene sets)
    "output/03_b_functional_characterization/viz/discovery_vs_validation.png",
]

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def path_to_flat_name(relative_path: str) -> str:
    """
    Convert a relative path to a flat filename by replacing separators
    with underscores. Preserves the full path so source is identifiable.

    e.g. "output/03_a_enhanced_hub_analysis/viz/cancer_relevance_pie.png"
      -> "output_03_a_enhanced_hub_analysis_viz_cancer_relevance_pie.png"
    """
    return relative_path.replace("/", "_").replace("\\", "_")


def resize_image(src: Path, dst: Path, max_dim: int) -> tuple[int, int, int, int]:
    """
    Resize image so its longest side = max_dim, preserving aspect ratio.
    Saves as PNG regardless of input format.
    Returns (orig_w, orig_h, new_w, new_h).
    """
    with Image.open(src) as img:
        orig_w, orig_h = img.size
        scale = min(max_dim / orig_w, max_dim / orig_h, 1.0)  # never upscale
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        if scale < 1.0:
            resized = img.resize((new_w, new_h), Image.LANCZOS)
        else:
            resized = img.copy()
        resized.save(dst, format="PNG")
    return orig_w, orig_h, new_w, new_h


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory : {OUTPUT_DIR.resolve()}")
    print(f"Max dimension    : {MAX_DIMENSION}px (longest side)")
    print(f"Images to process: {len(IMAGES)}\n")
    print(f"{'#':<4} {'Status':<8} {'Orig (WxH)':<18} {'New (WxH)':<18} Output filename")
    print("-" * 100)

    ok = skipped = errors = 0

    for i, rel_path in enumerate(IMAGES, 1):
        src = (PROJECT_ROOT / rel_path).resolve()
        out_name = path_to_flat_name(rel_path)
        dst = OUTPUT_DIR / out_name

        if not src.exists():
            print(f"{i:<4} {'MISSING':<8} {'—':<18} {'—':<18} {rel_path}")
            errors += 1
            continue

        try:
            orig_w, orig_h, new_w, new_h = resize_image(src, dst, MAX_DIMENSION)
            if (orig_w, orig_h) == (new_w, new_h):
                status = "KEPT"
                skipped += 1
            else:
                status = "OK"
                ok += 1
            print(f"{i:<4} {status:<8} {f'{orig_w}x{orig_h}':<18} {f'{new_w}x{new_h}':<18} {out_name}")
        except Exception as e:
            print(f"{i:<4} {'ERROR':<8} {'—':<18} {'—':<18} {rel_path}  [{e}]")
            errors += 1

    print("-" * 100)
    print(f"\nDone. Resized: {ok}  |  Already small (kept): {skipped}  |  Errors/Missing: {errors}")
    print(f"Files saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()