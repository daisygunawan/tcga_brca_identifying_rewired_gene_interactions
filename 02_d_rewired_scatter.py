"""
02_d_rewired_scatter.py

Regenerates Figures 18 and 38 (both are the rewired_edge_scatter) with:
  - Descriptive two-line axis labels (C6)
  - Interpretation text box inside the figure (C6)
  - Full distribution sample instead of top-200 filter (C7)

Also produces a focused companion plot (06_rewired_dual_mechanism.png) that
presents the same data without quadrant machinery — designed to communicate
the dual-mechanism finding directly to a non-specialist reader.

Reads: output/02_a_differential_analysis/differential_coexpression_sig.tsv
Confirmed columns: gene1, gene2, r_tumor, r_normal, delta_r, p_value, p_fdr, significant

Outputs: output/02_d_rewired_scatter/
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_02A      = PROJECT_ROOT / 'output' / '02_a_differential_analysis'
OUT_DIR      = PROJECT_ROOT / 'output' / '02_d_rewired_scatter'
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI         = 300
RANDOM_SEED = 42


def _save_table(df, out_path_stem):
    """Save DataFrame as both TSV and JSON."""
    tsv_path  = out_path_stem.with_suffix('.tsv')
    json_path = out_path_stem.with_suffix('.json')
    df.to_csv(tsv_path, sep='\t', index=False)
    df.to_json(json_path, orient='records', indent=2)
    print(f"  ✓ Table → {tsv_path.name} + {json_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Original full quadrant scatter (existing, unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def create_rewired_scatter(sig_pairs_df, output_dir):
    """
    C6: Descriptive two-line axis labels + interpretation text box.
    C7: Full distribution (up to 2,000 sampled pairs) replacing nlargest(200).

    Confirmed columns: gene1, gene2, r_tumor, r_normal, delta_r, p_value, p_fdr
    avg_abs_r computed here as (|r_tumor| + |r_normal|) / 2.
    """
    print(f"  Total significant pairs: {len(sig_pairs_df):,}")

    df = sig_pairs_df.copy()

    # Compute avg_abs_r from confirmed columns
    df['avg_abs_r']   = (df['r_tumor'].abs() + df['r_normal'].abs()) / 2
    df['abs_delta_r'] = df['delta_r'].abs()

    # ── C7: Full distribution sample ─────────────────────────────────────────
    if len(df) > 2000:
        plot_df     = df.sample(n=2000, random_state=RANDOM_SEED)
        sample_note = f"Random sample of 2,000 from {len(df):,} significant pairs"
    else:
        plot_df     = df.copy()
        sample_note = f"All {len(df):,} significant pairs"
    print(f"  Plotting: {sample_note}")

    # ── Quadrant assignment ───────────────────────────────────────────────────
    median_avg = df['avg_abs_r'].median()

    def quadrant(dr, avg):
        if   dr >= 0 and avg >= median_avg: return 'Q1'
        elif dr <  0 and avg >= median_avg: return 'Q2'
        elif dr <  0 and avg <  median_avg: return 'Q3'
        else:                               return 'Q4'

    plot_df = plot_df.copy()
    plot_df['quadrant'] = [quadrant(r, a)
                           for r, a in zip(plot_df['delta_r'], plot_df['avg_abs_r'])]

    q_colors = {'Q1': '#e74c3c', 'Q2': '#3498db',
                 'Q3': '#95a5a6', 'Q4': '#2ecc71'}
    q_labels = {
        'Q1': 'Q1: Gain + Strong baseline',
        'Q2': 'Q2: Loss + Strong baseline',
        'Q3': 'Q3: Loss + Weak baseline',
        'Q4': 'Q4: Gain + Weak baseline',
    }

    q_counts = plot_df['quadrant'].value_counts()
    total    = len(plot_df)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 9), dpi=DPI)

    for quad, grp in plot_df.groupby('quadrant'):
        sizes = (grp['abs_delta_r'] * 60).clip(10, 120)
        ax.scatter(grp['delta_r'], grp['avg_abs_r'],
                   c=q_colors.get(quad, '#aaaaaa'), s=sizes,
                   alpha=0.55, edgecolors='none',
                   label=q_labels.get(quad, quad))

    ax.axvline(x=0,          color='black', linewidth=1.0, linestyle='--', alpha=0.4)
    ax.axhline(y=median_avg, color='black', linewidth=1.0, linestyle='--', alpha=0.4)

    # ── C6: Descriptive axis labels ───────────────────────────────────────────
    ax.set_xlabel(
        'Δr  =  r_tumor − r_normal\n'
        '(positive = correlation gained in tumor;  negative = lost)',
        fontweight='bold', fontsize=12
    )
    ax.set_ylabel(
        'Average |r|  =  (|r_tumor| + |r_normal|) / 2\n'
        '(higher = stronger baseline co-expression before rewiring)',
        fontweight='bold', fontsize=12
    )
    ax.set_title(
        f'Rewired Pairs: Δr vs Baseline Correlation — Full Distribution\n'
        f'({sample_note}  |  point size ∝ |Δr| magnitude)',
        fontweight='bold', fontsize=13
    )

    # ── C6: Interpretation guide box ─────────────────────────────────────────
    guide = (
        "AXES GUIDE\n"
        "X-axis: rewiring direction and magnitude\n"
        "  right (+) = new connection formed in tumor\n"
        "  left  (−) = existing connection lost in tumor\n"
        "Y-axis: pre-rewiring co-expression strength\n"
        "  high  = genes strongly correlated before\n"
        "  low   = genes weakly/not correlated before"
    )
    ax.text(0.15, 0.64, guide, transform=ax.transAxes,
            fontsize=11, family='monospace', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                      edgecolor='gray', alpha=0.92))

    # ── Quadrant count corner labels ──────────────────────────────────────────
    corners = {
        'Q1': (0.97, 0.97, 'right', 'top'),
        'Q2': (0.03, 0.97, 'left',  'top'),
        'Q3': (0.03, 0.03, 'left',  'bottom'),
        'Q4': (0.97, 0.03, 'right', 'bottom'),
    }
    for quad, (tx, ty, ha, va) in corners.items():
        n   = q_counts.get(quad, 0)
        pct = 100 * n / total if total > 0 else 0
        ax.text(tx, ty, f'{quad}\nn={n:,} ({pct:.1f}%)',
                transform=ax.transAxes, fontsize=9.5, ha=ha, va=va,
                color=q_colors[quad], fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.75))

    ax.legend(fontsize=13, loc='center left', framealpha=0.9,
              bbox_to_anchor=(0.015, 0.82))

    # ── Q2/Q3 empty explanation annotation ───────────────────────────────────
    q2q3_note = (
        "Why Q2 & Q3 are empty:\n"
        "All 38,845 significant pairs show\n"
        "correlation GAINED in tumor (Δr > 0).\n"
        "Loss events (Δr < 0) do not reach\n"
        "the |Δr| ≥ 0.8 threshold because\n"
        "the normal network was already dense\n"
        "— no single loss is large enough."
    )
    ax.text(0.15, 0.40, q2q3_note,
            transform=ax.transAxes,
            fontsize=11, family='monospace', va='center',
            color='#555555',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0',
                      edgecolor='#aaaaaa', alpha=0.88))

    ax.grid(True, linestyle='--', alpha=0.2)
    ax.tick_params(labelsize=11)
    plt.tight_layout()

    out = output_dir / '06_rewired_edge_scatter.png'
    plt.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Scatter (Figs 18 + 38) → {out.name}")

    # ── Companion JSON ────────────────────────────────────────────────────────
    meta = {
        'sample_note':          sample_note,
        'n_pairs_significant':  len(df),
        'n_pairs_plotted':      len(plot_df),
        'median_avg_abs_r':     round(float(median_avg), 4),
        'quadrant_counts':      {k: int(v) for k, v in q_counts.to_dict().items()},
        'quadrant_pcts':        {k: round(100*v/total, 1)
                                 for k, v in q_counts.to_dict().items()},
        'x_axis_description': (
            'Delta-r = r_tumor minus r_normal; '
            'positive = gained in tumor, negative = lost'
        ),
        'y_axis_description': (
            'Average |r| = (|r_tumor| + |r_normal|) / 2; '
            'measures baseline co-expression strength before rewiring'
        ),
        'c6_applied': True,
        'c7_applied': True,
        'seed': RANDOM_SEED,
    }
    with open(output_dir / '06_rewired_edge_scatter_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  ✓ Companion JSON → 06_rewired_edge_scatter_meta.json")

    # ── Data table: plotted pairs ─────────────────────────────────────────────
    table_cols = ['gene1', 'gene2', 'r_tumor', 'r_normal',
                  'delta_r', 'avg_abs_r', 'quadrant', 'p_fdr']
    _save_table(
        plot_df[[c for c in table_cols if c in plot_df.columns]].round(6),
        output_dir / '06_rewired_edge_scatter_pairs'
    )

    # ── Data table: quadrant summary (all 38,845 pairs) ──────────────────────
    df['quadrant'] = [quadrant(r, a)
                      for r, a in zip(df['delta_r'], df['avg_abs_r'])]
    q_summary = (df.groupby('quadrant')
                   .agg(n_pairs=('delta_r', 'count'),
                        mean_delta_r=('delta_r', 'mean'),
                        mean_avg_abs_r=('avg_abs_r', 'mean'))
                   .reset_index()
                   .round(4))
    _save_table(q_summary, output_dir / '06_rewired_edge_scatter_quadrant_summary')

    # Return enriched df for use in companion plot
    return df, median_avg


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Focused dual-mechanism plot (new companion)
# ─────────────────────────────────────────────────────────────────────────────
def create_dual_mechanism_plot(df, median_avg, output_dir):
    """
    Focused companion scatter that communicates the dual-mechanism finding
    directly — without quadrant labels, Q2/Q3 machinery, or axes guides.

    Design decisions:
      - x-axis: Δr (zoomed to where data actually is)
      - y-axis: avg_abs_r (baseline co-expression strength)
      - Color: continuous RdYlGn_r colormap driven by avg_abs_r
               green = low baseline (new connections)
               red   = high baseline (amplified connections)
      - Red annotation box: top-right INSIDE axes, arrow to red dots (high avg|r|)
      - Green annotation box: bottom-right INSIDE axes, arrow to green dots (low avg|r|)
      - Summary box: top-right corner
      - Colorbar: right side
      - No Q1/Q2/Q3/Q4 labels anywhere

    Output: 06_rewired_dual_mechanism.png + _data.tsv/.json
    """
    print(f"\n  Generating dual-mechanism companion plot...")

    # Sample for rendering — different seed from plot 1
    if len(df) > 2000:
        plot_df = df.sample(n=2000, random_state=RANDOM_SEED + 1).copy()
    else:
        plot_df = df.copy()

    # ── Colormap ─────────────────────────────────────────────────────────────
    cmap = plt.cm.RdYlGn_r
    norm = mcolors.Normalize(
        vmin=plot_df['avg_abs_r'].quantile(0.02),
        vmax=plot_df['avg_abs_r'].quantile(0.98)
    )

    sizes = (plot_df['abs_delta_r'] * 55).clip(15, 110)

    fig, ax = plt.subplots(figsize=(13, 9), dpi=DPI)

    sc = ax.scatter(
        plot_df['delta_r'],
        plot_df['avg_abs_r'],
        c=plot_df['avg_abs_r'],
        cmap=cmap, norm=norm,
        s=sizes, alpha=0.55, edgecolors='none'
    )

    # ── Median dividing line ──────────────────────────────────────────────────
    ax.axhline(y=median_avg, color='#444444', linewidth=1.5,
               linestyle='--', alpha=0.7, zorder=3)

    # ── Axis limits — set BEFORE annotations so data coords are stable ────────
    x_pad  = 0.01
    x_min  = plot_df['delta_r'].min() - x_pad
    x_max  = plot_df['delta_r'].max() + x_pad
    y_min  = plot_df['avg_abs_r'].min() - 0.005
    y_max  = plot_df['avg_abs_r'].max() + 0.015
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # ── Target coordinates for arrows ────────────────────────────────────────
    # Red arrow: point to a genuine red dot — 90th pct of avg_abs_r,
    # among pairs with delta_r in the lower half (where red dots cluster left)
    red_candidates = plot_df[
        (plot_df['avg_abs_r'] >= plot_df['avg_abs_r'].quantile(0.88)) &
        (plot_df['delta_r']   <= plot_df['delta_r'].quantile(0.30))
    ]
    if len(red_candidates) > 0:
        arrow_red_x = float(red_candidates['delta_r'].median())
        arrow_red_y = float(red_candidates['avg_abs_r'].median())
    else:
        arrow_red_x = float(plot_df['delta_r'].quantile(0.15))
        arrow_red_y = float(plot_df['avg_abs_r'].quantile(0.90))

    # Green arrow: point to a genuine green dot — 10th pct of avg_abs_r,
    # among pairs with delta_r in the right half (dense green triangle)
    grn_candidates = plot_df[
        (plot_df['avg_abs_r'] <= plot_df['avg_abs_r'].quantile(0.12)) &
        (plot_df['delta_r']   >= plot_df['delta_r'].quantile(0.55))
    ]
    if len(grn_candidates) > 0:
        arrow_grn_x = float(grn_candidates['delta_r'].median())
        arrow_grn_y = float(grn_candidates['avg_abs_r'].median())
    else:
        arrow_grn_x = float(plot_df['delta_r'].quantile(0.75))
        arrow_grn_y = float(plot_df['avg_abs_r'].quantile(0.05))

    # #Some hardcoded override
    arrow_red_x = 0.85
    arrow_grn_x = 0.85  # same middle x position

    # ── Red annotation box — top right INSIDE axes ────────────────────────────
    ax.annotate(
        'Amplified existing connections\n'
        '(genes that were already co-expressed\n'
        ' became even more strongly linked\n'
        ' in tumor)',
        xy=(arrow_red_x, arrow_red_y),           # arrow tip = real red dot
        xytext=(x_max - 0.001, y_max - 0.002),   # box anchored top-right inside
        fontsize=11, ha='right', va='top',
        color='#c0392b', fontweight='bold',
        arrowprops=dict(
            arrowstyle='->', color='#c0392b', lw=1.8,
            linestyle='dashed', connectionstyle='arc3,rad=0.0'  # straight line
            # connectionstyle='arc3,rad=-0.25'
        ),
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor='#c0392b', alpha=0.92),
        xycoords='data', textcoords='data'
    )

    # ── Green annotation box — bottom right INSIDE axes ───────────────────────
    ax.annotate(
        'New connections formed\n'
        '(genes with no prior relationship\n'
        ' became strongly co-expressed\n'
        ' in tumor)',
        xy=(arrow_grn_x, arrow_grn_y),           # arrow tip = real green dot
        xytext=(x_max - 0.001, y_min + 0.002),   # box anchored bottom-right inside
        fontsize=11, ha='right', va='bottom',
        color='#27ae60', fontweight='bold',
        arrowprops=dict(
            arrowstyle='->', color='#27ae60', lw=1.8,
            linestyle='dashed', connectionstyle='arc3,rad=0.0'  # straight line
            # connectionstyle='arc3,rad=0.25'
        ),
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor='#27ae60', alpha=0.92),
        xycoords='data', textcoords='data'
    )

    # ── Summary box — upper centre-left, clear of both annotation boxes ───────
    ax.text(
        0.38, 0.97,
        f'38,845 gene pairs\n'
        f'All gained co-expression in tumor (Δr > 0)\n'
        f'Split ~50/50 between the two mechanisms',
        transform=ax.transAxes, fontsize=12, 
        ha='center', va='top', color='#333333',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f9f9f9',
                  edgecolor='#cccccc', alpha=0.92)
    )

    # ── Colorbar ──────────────────────────────────────────────────────────────
    cbar = fig.colorbar(sc, ax=ax, pad=0.01, fraction=0.025)
    cbar.set_label(
        'Baseline co-expression strength\n(avg |r| before rewiring)',
        fontsize=11, fontweight='bold'
    )
    cbar.ax.tick_params(labelsize=10)

    # ── Axes labels and title ─────────────────────────────────────────────────
    ax.set_xlabel(
        'Rewiring magnitude  (Δr = r_tumor − r_normal)\n'
        'All values positive: every significant pair gained co-expression in tumor',
        fontweight='bold', fontsize=12
    )
    ax.set_ylabel(
        'Baseline co-expression strength\n'
        '(average |r| across both conditions before rewiring)',
        fontweight='bold', fontsize=12
    )
    ax.set_title(
        'Two Mechanisms of Tumor-Specific Co-expression Rewiring\n'
        'in 38,845 Significantly Rewired Gene Pairs (TCGA-BRCA)',
        fontweight='bold', fontsize=14
    )
    ax.grid(True, linestyle='--', alpha=0.15)
    ax.tick_params(labelsize=11)
    plt.tight_layout()

    out = output_dir / '06_rewired_dual_mechanism.png'
    plt.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Dual mechanism plot → {out.name}")

    # ── Data tables ───────────────────────────────────────────────────────────
    # Split by mechanism for the table
    plot_df = plot_df.copy()
    plot_df['mechanism'] = plot_df['avg_abs_r'].apply(
        lambda v: 'Amplified existing connection' if v >= median_avg
                  else 'New connection formed'
    )

    # Plotted pairs table
    dm_cols = ['gene1', 'gene2', 'r_tumor', 'r_normal',
               'delta_r', 'avg_abs_r', 'abs_delta_r', 'mechanism', 'p_fdr']
    _save_table(
        plot_df[[c for c in dm_cols if c in plot_df.columns]].round(6),
        output_dir / '06_rewired_dual_mechanism_pairs'
    )

    # Mechanism summary from full dataset
    df_full = df.copy()
    df_full['mechanism'] = df_full['avg_abs_r'].apply(
        lambda v: 'Amplified existing connection' if v >= median_avg
                  else 'New connection formed'
    )
    dm_summary = (
        df_full.groupby('mechanism')
               .agg(n_pairs       =('delta_r',    'count'),
                    mean_delta_r  =('delta_r',    'mean'),
                    mean_avg_abs_r=('avg_abs_r',  'mean'),
                    min_delta_r   =('delta_r',    'min'),
                    max_delta_r   =('delta_r',    'max'))
               .reset_index()
               .round(4)
    )
    _save_table(dm_summary, output_dir / '06_rewired_dual_mechanism_summary')
    print(f"  ✓ Dual mechanism summary (all {len(df):,} pairs):")
    for _, row in dm_summary.iterrows():
        pct = 100 * row['n_pairs'] / len(df)
        print(f"    {row['mechanism']}: n={int(row['n_pairs']):,} "
              f"({pct:.1f}%), mean Δr={row['mean_delta_r']:.3f}, "
              f"mean avg|r|={row['mean_avg_abs_r']:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("02_d_rewired_scatter.py")
    print("Fixing Figures 18 and 38 (rewired edge scatter)")
    print("+ Companion dual-mechanism plot")
    print("=" * 60)

    sig_path = OUT_02A / 'differential_coexpression_sig.tsv'
    if not sig_path.exists():
        print(f"ERROR: {sig_path} not found. Run 02_a first.")
        raise SystemExit(1)

    sig_pairs = pd.read_csv(sig_path, sep='\t')
    print(f"Loaded {len(sig_pairs):,} pairs from {sig_path.name}")
    print(f"Columns: {list(sig_pairs.columns)}")

    # Plot 1 — original quadrant scatter (Figs 18 + 38)
    enriched_df, median_avg = create_rewired_scatter(sig_pairs, OUT_DIR)

    # Plot 2 — focused dual-mechanism companion (new)
    create_dual_mechanism_plot(enriched_df, median_avg, OUT_DIR)

    print("\nDone. Outputs in: output/02_d_rewired_scatter/")
    print("  06_rewired_edge_scatter.png                      ← Figs 18 + 38 (quadrant plot)")
    print("  06_rewired_edge_scatter_meta.json                ← metadata + quadrant counts")
    print("  06_rewired_edge_scatter_pairs.tsv/.json          ← 2,000 plotted pairs")
    print("  06_rewired_edge_scatter_quadrant_summary.tsv/.json ← all 38,845 pairs summary")
    print("  06_rewired_dual_mechanism.png                    ← companion focused plot")
    print("  06_rewired_dual_mechanism_pairs.tsv/.json        ← 2,000 plotted pairs with mechanism label")
    print("  06_rewired_dual_mechanism_summary.tsv/.json      ← all 38,845 pairs by mechanism")