"""
01_d_regenerate_figures.py

Regenerates Figures 11, 13, and 15 at corrected sizes and with panel labels.
Reads pre-computed JSON data files from 01_c output only.
Does NOT rerun network construction, correlation loading, or graph metrics.

Outputs: output/01_d_regenerate_figures/  (originals untouched)
Fixes:   C8 (Fig 11 figsize), C9 (Fig 13 figsize), C10 (Fig 15 panel labels)

Confirmed JSON structures:
  Fig11 (01_spearman_degree_distribution_overlay.json):
    metadata.tumor_median, metadata.normal_median, metadata.normalized
    tumor_distribution.raw_degrees  — list[int]
    normal_distribution.raw_degrees — list[int]

  Fig13 (01_edge_type_breakdown.json):
    tumor.spearman_breakdown  — {label: count}
    tumor.spearman_total, tumor.pearson_total
    normal.spearman_breakdown — {label: count}
    normal.spearman_total, normal.pearson_total

  Fig15 (03_unified_performance_dashboard.json):
    metrics.spearman.{tumor_edges, normal_edges, ...}
    metrics.pearson.{tumor_edges, normal_edges, ...}
    tumor_advantage_ratio, normal_advantage_ratio, overall_advantage
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_01C      = PROJECT_ROOT / 'output' / '01_c_network_visualization'
OUT_DIR      = PROJECT_ROOT / 'output' / '01_d_regenerate_figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300


def _load_json(path, label):
    if not path.exists():
        print(f"  SKIP {label}: {path.name} not found")
        return None
    with open(path) as f:
        data = json.load(f)
    print(f"  [{label}] loaded — top-level keys: {list(data.keys())}")
    return data


def _save_table(records, out_path_stem):
    """Save a list-of-dicts as both TSV and JSON for easy reuse."""
    import csv
    tsv_path  = out_path_stem.with_suffix('.tsv')
    json_path = out_path_stem.with_suffix('.json')
    if records:
        keys = list(records[0].keys())
        with open(tsv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys, delimiter='\t')
            w.writeheader()
            w.writerows(records)
        with open(json_path, 'w') as f:
            json.dump(records, f, indent=2)
        print(f"  ✓ Table → {tsv_path.name} + {json_path.name}")


# ── Figure 11 — Degree Distribution Overlay (enlarged) ───────────────────────
def fix_figure_11():
    """
    Reads raw_degrees arrays from nested tumor_distribution / normal_distribution.
    Applies Gaussian smoothing and normalisation (% of genes) to match original.
    Outputs PNG + data TSV/JSON.
    """
    json_path = OUT_01C / 'spearman' / '01_spearman_degree_distribution_overlay.json'
    data = _load_json(json_path, 'Fig11')
    if data is None:
        return

    meta       = data.get('metadata', {})
    tumor_arr  = np.array(data['tumor_distribution']['raw_degrees'])
    normal_arr = np.array(data['normal_distribution']['raw_degrees'])
    tumor_med  = meta.get('tumor_median',  float(np.median(tumor_arr)))
    normal_med = meta.get('normal_median', float(np.median(normal_arr)))
    normalized = meta.get('normalized', True)

    fig, ax = plt.subplots(figsize=(18, 8), dpi=DPI)   # was (12, 6)

    def plot_dist(arr, med, color, ls, label_curve, label_med):
        max_d  = int(arr.max()) + 1
        bins   = min(300, max_d)
        counts, edges = np.histogram(arr, bins=bins, range=(0, max_d))
        centers = (edges[:-1] + edges[1:]) / 2
        if normalized:
            counts = counts / counts.sum() * 100
        smooth = gaussian_filter1d(counts.astype(float), sigma=4)
        ax.plot(centers, smooth, color=color, linestyle=ls,
                linewidth=2.5, label=label_curve)
        ax.fill_between(centers, smooth, alpha=0.18, color=color)
        ax.axvline(med, color=color, linestyle=':', linewidth=2.0, alpha=0.85,
                   label=label_med)

    plot_dist(tumor_arr,  tumor_med,  '#e74c3c', '-',  'Tumor network',
              f'Tumor median = {int(tumor_med)}')
    plot_dist(normal_arr, normal_med, '#3498db', '--', 'Normal network',
              f'Normal median = {int(normal_med)}')

    ax.set_xlabel('Degree (Number of Connections)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Frequency (% of genes)' if normalized else 'Frequency',
                  fontsize=16, fontweight='bold')
    ax.set_title(
        'Degree Distribution Overlay: Tumor vs Normal Network\n'
        f'(Spearman |r| ≥ 0.7, {len(tumor_arr):,} genes)',
        fontsize=15, fontweight='bold'
    )
    ax.legend(fontsize=13, framealpha=0.9)
    ax.tick_params(labelsize=13)
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    out = OUT_DIR / '01_spearman_degree_distribution_overlay.png'
    plt.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure 11 → {out.name}")

    # Data table
    records = [
        {'condition': 'tumor',  'median_degree': tumor_med,
         'mean_degree': float(tumor_arr.mean()),  'n_genes': len(tumor_arr)},
        {'condition': 'normal', 'median_degree': normal_med,
         'mean_degree': float(normal_arr.mean()), 'n_genes': len(normal_arr)},
    ]
    _save_table(records, OUT_DIR / '01_spearman_degree_distribution_overlay_summary')


# ── Figure 13 — Edge Type Breakdown (enlarged) ───────────────────────────────
def fix_figure_13():
    """
    Panel 1: Spearman edge type breakdown — grouped bars per type, tumor vs normal.
             Uses log scale because normal (~4.5M) dwarfs tumor (~54K).
    Panel 2: Spearman total vs Pearson total per condition (also log scale).
    Outputs PNG + data TSV/JSON.
    """
    json_path = OUT_01C / 'method_comparison' / '01_edge_type_breakdown.json'
    data = _load_json(json_path, 'Fig13')
    if data is None:
        return

    td = data['tumor']
    nd = data['normal']
    sp_t = td.get('spearman_breakdown', {})
    sp_n = nd.get('spearman_breakdown', {})
    categories  = list(sp_t.keys())
    tumor_vals  = [sp_t.get(c, 0) for c in categories]
    normal_vals = [sp_n.get(c, 0) for c in categories]

    method_labels  = ['Spearman\nTumor', 'Pearson\nTumor',
                      'Spearman\nNormal', 'Pearson\nNormal']
    method_vals    = [td.get('spearman_total', 0), td.get('pearson_total', 0),
                      nd.get('spearman_total', 0), nd.get('pearson_total', 0)]
    method_colors  = ['#e74c3c', '#e74c3c', '#2ecc71', '#2ecc71']
    method_alphas  = [0.9, 0.5, 0.9, 0.5]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10), dpi=DPI)   # was (16, 7)
    w = 0.35
    x = np.arange(len(categories))

    # ── Panel 1: Spearman breakdown, log scale ─────────────────────────────
    bars1 = ax1.bar(x - w/2, tumor_vals,  w, label='Tumor',  color='#e74c3c', alpha=0.9)
    bars2 = ax1.bar(x + w/2, normal_vals, w, label='Normal', color='#2ecc71', alpha=0.9)
    ax1.set_yscale('log')

    for bar, val in zip(list(bars1) + list(bars2), tumor_vals + normal_vals):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     val * 1.15, f'{val:,.0f}',
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_xlabel('Edge Type', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Edge Count (log scale)', fontsize=14, fontweight='bold')
    ax1.set_title('Spearman Edge Type Breakdown\nby Condition (log scale)',
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=11, rotation=15, ha='right')
    ax1.legend(fontsize=12)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax1.tick_params(labelsize=12)

    # ── Panel 2: Method totals, log scale ─────────────────────────────────
    x2 = np.arange(len(method_labels))
    for i, (lbl, val, col, alp) in enumerate(
            zip(method_labels, method_vals, method_colors, method_alphas)):
        bar = ax2.bar(i, val, color=col, alpha=alp, width=0.55,
                      label=lbl if i < 2 else None)
        if val > 0:
            ax2.text(i, val * 1.15, f'{val:,.0f}',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_yscale('log')
    ax2.set_xlabel('Method × Condition', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Total Edge Count (log scale)', fontsize=14, fontweight='bold')
    ax2.set_title('Total Edges: Spearman vs Pearson\nby Condition (log scale)',
                  fontsize=14, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(method_labels, fontsize=12)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax2.tick_params(labelsize=12)

    # Unique-to-Spearman annotation
    uniq_t = td.get('unique_to_spearman', 0)
    uniq_n = nd.get('unique_to_spearman', 0)
    ax2.annotate(f'Unique to Spearman:\nTumor={uniq_t:,.0f}\nNormal={uniq_n:,.0f}',
                 xy=(0.98, 0.05), xycoords='axes fraction',
                 ha='right', va='bottom', fontsize=11,
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                           edgecolor='gray', alpha=0.9))

    plt.tight_layout()
    out = OUT_DIR / '01_edge_type_breakdown.png'
    plt.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure 13 → {out.name}")

    # Data tables
    records_panel1 = [
        {'edge_type': c, 'tumor_count': tv, 'normal_count': nv}
        for c, tv, nv in zip(categories, tumor_vals, normal_vals)
    ]
    _save_table(records_panel1, OUT_DIR / '01_edge_type_breakdown_spearman')

    records_panel2 = [
        {'condition': 'tumor',  'spearman_total': td.get('spearman_total', 0),
         'pearson_total': td.get('pearson_total', 0),
         'unique_to_spearman': td.get('unique_to_spearman', 0)},
        {'condition': 'normal', 'spearman_total': nd.get('spearman_total', 0),
         'pearson_total': nd.get('pearson_total', 0),
         'unique_to_spearman': nd.get('unique_to_spearman', 0)},
    ]
    _save_table(records_panel2, OUT_DIR / '01_edge_type_breakdown_method_totals')


# ── Figure 15 — Unified Performance Dashboard with panel labels ───────────────
def fix_figure_15():
    """
    Panel (a): Edge counts — 4 bars: Spearman/Pearson × Tumor/Normal (log scale).
    Panel (b): Spearman/Pearson coverage ratio per condition.
    Adds (a)/(b) panel labels. Outputs PNG + data TSV/JSON.
    """
    json_path = OUT_01C / 'method_comparison' / '03_unified_performance_dashboard.json'
    data = _load_json(json_path, 'Fig15')
    if data is None:
        return

    sp = data['metrics']['spearman']
    pe = data['metrics']['pearson']
    t_ratio  = data.get('tumor_advantage_ratio',  0)
    n_ratio  = data.get('normal_advantage_ratio', 0)
    overall  = data.get('overall_advantage', 0)
    thresh   = data.get('threshold', 0.7)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=DPI)

    # ── Panel (a): edge counts — 4 bars ──────────────────────────────────
    labels = ['Spearman\nTumor', 'Pearson\nTumor',
              'Spearman\nNormal', 'Pearson\nNormal']
    values = [sp['tumor_edges'],  pe['tumor_edges'],
              sp['normal_edges'], pe['normal_edges']]
    colors = ['#e74c3c', '#e74c3c', '#2ecc71', '#2ecc71']
    alphas = [0.9, 0.5, 0.9, 0.5]

    for i, (lbl, val, col, alp) in enumerate(zip(labels, values, colors, alphas)):
        ax1.bar(i, val, color=col, alpha=alp, width=0.6)
        ax1.text(i, val * 1.12, f'{val:,.0f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_yscale('log')
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=12)
    ax1.set_ylabel('Edge Count (log scale)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Edge Count: Spearman vs Pearson\n(threshold |r| ≥ {thresh})',
                  fontsize=12, fontweight='bold')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax1.tick_params(labelsize=11)
    ax1.text(0.02, 0.97, '(a)', transform=ax1.transAxes,
             fontsize=14, fontweight='bold', va='top',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # ── Panel (b): coverage ratio ─────────────────────────────────────────
    conditions = ['Tumor', 'Normal']
    ratios     = [t_ratio, n_ratio]
    bar_colors = ['#e74c3c', '#2ecc71']

    for i, (cond, ratio, col) in enumerate(zip(conditions, ratios, bar_colors)):
        ax2.bar(i, ratio, color=col, alpha=0.85, width=0.5)
        ax2.text(i, ratio + 0.04, f'{ratio:.2f}x',
                 ha='center', va='bottom', fontsize=13, fontweight='bold')

    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5,
                label='Equal coverage (ratio = 1)')
    ax2.set_xticks(range(len(conditions)))
    ax2.set_xticklabels(conditions, fontsize=13)
    ax2.set_ylabel('Spearman / Pearson Edge Ratio', fontsize=12, fontweight='bold')
    ax2.set_title(f'Spearman vs Pearson Coverage Ratio\n(Overall: {overall:.2f}x)',
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax2.tick_params(labelsize=12)
    ax2.text(0.02, 0.97, '(b)', transform=ax2.transAxes,
             fontsize=14, fontweight='bold', va='top',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    plt.suptitle('Figure 15. Spearman vs Pearson Correlation Comparison',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    out = OUT_DIR / '15_spearman_vs_pearson_comparison.png'
    plt.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure 15 → {out.name}")

    # Data table
    records = [
        {'metric': 'tumor_spearman_edges',  'value': sp['tumor_edges']},
        {'metric': 'tumor_pearson_edges',   'value': pe['tumor_edges']},
        {'metric': 'normal_spearman_edges', 'value': sp['normal_edges']},
        {'metric': 'normal_pearson_edges',  'value': pe['normal_edges']},
        {'metric': 'tumor_advantage_ratio', 'value': round(t_ratio, 4)},
        {'metric': 'normal_advantage_ratio','value': round(n_ratio, 4)},
        {'metric': 'overall_advantage',     'value': round(overall, 4)},
        {'metric': 'threshold',             'value': thresh},
    ]
    _save_table(records, OUT_DIR / '15_spearman_vs_pearson_comparison_metrics')


if __name__ == '__main__':
    print("=" * 60)
    print("01_d_regenerate_figures.py")
    print("Regenerating Figures 11, 13, 15 + companion data tables")
    print("=" * 60)
    fix_figure_11()
    fix_figure_13()
    fix_figure_15()
    print("\nDone. Outputs in: output/01_d_regenerate_figures/")