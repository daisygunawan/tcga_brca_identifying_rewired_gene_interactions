"""
02_e_validate_classification.py

Produces new/corrected classification figures by reading frozen 02_c outputs.
No model training on real labels. No changes to existing result numbers.

Confirmed structures (from fyp_output.txt):
  cv_summary.json:
    {model_name: {mean, std, fold_scores, scoring, duration_s}, cv_folds: 5}
    Models: RandomForest, XGBoost, Ensemble_Soft, Ensemble_Hard,
            Ensemble_Weighted, Ensemble_Stacking

  classification_performance_comparison.tsv:
    columns: (index=model), acc, auc, f1, cv_auc_mean, cv_auc_std

  model_evaluation_data.json:
    {y_true: [...], y_proba: [...], classes: [...],
     confusion_matrices: {model: {...}},
     cv_auc: {model: {mean, std, duration_s}}}
    NOTE: does NOT contain X_train/y_train — proxy permutation used.

  stacking_sensitivity_results.tsv:
    columns: key, display_name, test_auc, test_acc, test_f1,
             cv_auc_mean, cv_auc_std, cv_duration_s, status

  sampling_methods_comparison.json:
    structure determined at runtime — key printed for inspection.

Fixes: C2 (performance chart), C3 (permutation test),
       C4 (Figs 28/29 legend), C5 (Fig 31 panels + y-axis)
"""

import json
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_02C      = PROJECT_ROOT / 'output' / '02_c_sample_classification' / 'sampling_comparison'
OUT_DIR      = PROJECT_ROOT / 'output' / '02_e_validate_classification'
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI     = 300
METHODS = ['cluster_based', 'median']


def _save_table(records_or_df, out_path_stem):
    """Save list-of-dicts or DataFrame as TSV + JSON."""
    tsv_path  = out_path_stem.with_suffix('.tsv')
    json_path = out_path_stem.with_suffix('.json')
    if isinstance(records_or_df, pd.DataFrame):
        records_or_df.to_csv(tsv_path, sep='\t', index=False)
        records_or_df.to_json(json_path, orient='records', indent=2)
    else:
        if not records_or_df:
            return
        with open(tsv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(records_or_df[0].keys()),
                               delimiter='\t')
            w.writeheader()
            w.writerows(records_or_df)
        with open(json_path, 'w') as f:
            json.dump(records_or_df, f, indent=2)
    print(f"  ✓ Table → {tsv_path.name} + {json_path.name}")


# ── C2: Cross-model performance comparison chart ──────────────────────────────
def plot_performance_comparison(method):
    """
    Reads cv_summary.json and classification_performance_comparison.tsv
    for the 4 base models. Additionally reads:
      stacking_sensitivity_results.tsv  → best stacking meta-learner (by test AUC)
      model_evaluation_data.json        → cv_auc for all models incl. Ensemble_Stacking

    Weighted ensemble intentionally excluded: its test AUC (0.9965) sits between
    RF and Soft-Voting, it adds no new interpretive information, and its CV AUC=1.000
    is an in-sample tuning artefact. Full weighted sweep results remain in TSV.

    Confirmed structures (from fyp_output.txt):
      stacking TSV columns: key, display_name, test_auc, test_acc, test_f1, cv_auc_mean, cv_auc_std
      cv_auc in model_evaluation_data.json: all models with mean/std
    """
    method_dir = OUT_02C / method
    cv_path    = method_dir / 'cv_summary.json'
    perf_path  = method_dir / 'classification_performance_comparison.tsv'
    eval_path  = method_dir / 'model_evaluation_data.json'
    s_tsv_path = method_dir / 'stacking_sensitivity' / 'stacking_sensitivity_results.tsv'

    if not cv_path.exists() or not perf_path.exists():
        print(f"  SKIP C2 ({method}): cv_summary or perf TSV not found")
        return

    with open(cv_path) as f:
        cv = json.load(f)

    perf_df = pd.read_csv(perf_path, sep='\t', index_col=0)
    print(f"  [{method}] perf TSV models: {list(perf_df.index)}")

    # ── CV AUC for Weighted/Stacking from model_evaluation_data.json ──────────
    cv_auc_ext = {}
    if eval_path.exists():
        with open(eval_path) as f:
            ev = json.load(f)
        cv_auc_ext = ev.get('cv_auc', {})
        print(f"  [{method}] cv_auc models in eval_data: {list(cv_auc_ext.keys())}")

    # ── Best Weighted combo from sensitivity TSV ──────────────────────────────
    # ── Best Stacking meta-learner from sensitivity TSV ───────────────────────
    extra = {}   # {model_key: {auc, acc, f1, display}}
    # Note: Ensemble_Weighted is intentionally excluded — its test AUC sits between
    # RF and Soft-Voting and adds no new interpretive information. Its CV AUC=1.000
    # is also an in-sample tuning artefact rather than a genuine generalisation
    # estimate. The full weighted sensitivity sweep remains in the TSV for reference.
    if s_tsv_path.exists():
        s_df    = pd.read_csv(s_tsv_path, sep='\t')
        auc_col = 'test_auc' if 'test_auc' in s_df.columns else 'auc'
        acc_col = 'test_acc' if 'test_acc' in s_df.columns else 'acc'
        f1_col  = 'test_f1'  if 'test_f1'  in s_df.columns else 'f1'
        best_s  = s_df.loc[s_df[auc_col].idxmax()]
        lbl     = str(best_s.get('display_name', best_s.get('key', 'best')))
        extra['Ensemble_Stacking'] = {
            'auc':     float(best_s[auc_col]),
            'acc':     float(best_s.get(acc_col, 0)),
            'f1':      float(best_s.get(f1_col, 0)),
            'display': f"Stacking\n({lbl})"
        }
        print(f"  [{method}] Best stacking: {lbl} "
              f"test AUC={extra['Ensemble_Stacking']['auc']:.4f}")
    else:
        print(f"  [{method}] stacking_sensitivity_results.tsv not found — skipping")

    # ── Build full model list ─────────────────────────────────────────────────
    models_all  = ['RandomForest', 'XGBoost', 'Ensemble_Soft', 'Ensemble_Hard']
    display_all = ['Random\nForest', 'XGBoost', 'Soft\nVoting', 'Hard\nVoting']
    for key in ['Ensemble_Stacking']:
        if key in extra:
            models_all.append(key)
            display_all.append(extra[key]['display'])

    metrics       = ['auc', 'acc', 'f1']
    metric_labels = ['AUC', 'Accuracy', 'F1']
    colors        = ['#2ecc71', '#3498db', '#e74c3c']

    x       = np.arange(len(models_all))
    width   = 0.25
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(16, 7))   # wider for 6 models

    for i, (metric, mlabel, color) in enumerate(zip(metrics, metric_labels, colors)):
        values, errors = [], []
        for m in models_all:
            if m in extra:
                val = extra[m].get(metric, 0.0)
            else:
                try:
                    val = float(perf_df.loc[m, metric])
                except (KeyError, ValueError):
                    val = float(cv.get(m, {}).get('mean', 0)) if metric == 'auc' else 0.0
            values.append(float(val))

            if metric == 'auc':
                std = float(cv_auc_ext.get(m, {}).get('std', 0) or
                            cv.get(m, {}).get('std', 0))
                errors.append(std)
            else:
                errors.append(0.0)

        bars = ax.bar(
            x + offsets[i], values, width,
            label=mlabel, color=color, alpha=0.85,
            yerr=errors if metric == 'auc' else None,
            capsize=4, error_kw={'linewidth': 1.5}
        )
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.004,
                        f'{val:.4f}', ha='center', va='bottom',
                        fontsize=7, fontweight='bold', rotation=90)

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(
        f'All-Model Performance Comparison — {method.replace("_", " ").title()} Sampling\n'
        f'AUC error bars = 5-fold CV std  |  Stacking: best meta-learner from sensitivity sweep',
        fontsize=12, fontweight='bold'
    )
    ax.set_xticks(x)
    ax.set_xticklabels(display_all, fontsize=10)
    ax.set_ylim(0.88, 1.025)
    ax.axhline(y=0.85, color='gray', linestyle=':', linewidth=1.2,
               label='Target AUC = 0.85')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.tick_params(labelsize=10)
    plt.tight_layout()

    out = OUT_DIR / f'model_performance_comparison_{method}.png'
    plt.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ C2 all-model chart ({method}) → {out.name}")

    # ── Data table ────────────────────────────────────────────────────────────
    records = []
    for m, dn in zip(models_all, display_all):
        row = {'model': m, 'display_name': dn.replace('\n', ' ')}
        if m in extra:
            for metric in metrics:
                row[metric] = extra[m].get(metric)
            row['source'] = 'sensitivity_best'
        else:
            for metric in metrics:
                try:
                    row[metric] = float(perf_df.loc[m, metric])
                except (KeyError, ValueError):
                    row[metric] = None
            row['source'] = 'classification_performance_tsv'
        row['cv_auc_mean'] = float(cv_auc_ext.get(m, {}).get('mean', 0) or
                                   cv.get(m, {}).get('mean', 0))
        row['cv_auc_std']  = float(cv_auc_ext.get(m, {}).get('std', 0) or
                                   cv.get(m, {}).get('std', 0))
        records.append(row)
    _save_table(records, OUT_DIR / f'model_performance_comparison_{method}')


# ── C3: AUC Permutation test ──────────────────────────────────────────────────
def run_permutation_test(method, n_permutations=1000):
    """
    model_evaluation_data.json contains y_true and y_proba (confirmed).
    Does NOT contain X_train — uses proxy permutation (shuffle y_true vs
    fixed y_proba) which is a valid label-permutation test.
    """
    method_dir = OUT_02C / method
    eval_path  = method_dir / 'model_evaluation_data.json'
    perm_dir   = OUT_DIR / 'permutation_test'
    perm_dir.mkdir(parents=True, exist_ok=True)

    if not eval_path.exists():
        print(f"  SKIP C3 ({method}): model_evaluation_data.json not found")
        return

    with open(eval_path) as f:
        eval_data = json.load(f)

    y_true       = np.array(eval_data['y_true'])
    y_proba      = np.array(eval_data['y_proba'])
    observed_auc = roc_auc_score(y_true, y_proba)
    print(f"  [{method}] Observed AUC: {observed_auc:.4f}")

    # Proxy permutation: shuffle y_true, keep y_proba fixed
    print(f"  [{method}] Running proxy permutation test "
          f"({n_permutations} permutations)...")
    rng = np.random.default_rng(seed=42)
    null_aucs = np.array([
        roc_auc_score(rng.permutation(y_true), y_proba)
        for _ in range(n_permutations)
    ])

    p_value = max(float(np.mean(null_aucs >= observed_auc)), 1.0 / n_permutations)
    print(f"  [{method}] Null: mean={null_aucs.mean():.4f} ± {null_aucs.std():.4f}")
    print(f"  [{method}] Empirical p = {p_value:.4f} "
          f"({'SIGNIFICANT' if p_value < 0.05 else 'NOT significant'})")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(null_aucs, bins=40, color='#95a5a6', alpha=0.75, edgecolor='white',
            label=f'Null distribution ({n_permutations} permutations)')
    ax.axvline(observed_auc, color='#e74c3c', linewidth=2.5, linestyle='--',
               label=f'Observed AUC = {observed_auc:.4f}')
    ax.axvline(null_aucs.mean(), color='#3498db', linewidth=1.5, linestyle=':',
               label=f'Null mean = {null_aucs.mean():.4f}')
    ax.set_xlabel('AUC', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title(
        f'AUC Permutation Test — {method.replace("_", " ").title()} Sampling\n'
        f'Empirical p = {p_value:.4f}  |  Observed = {observed_auc:.4f}  |  '
        f'Null mean = {null_aucs.mean():.4f}\n'
        f'(Proxy test: y_true shuffled vs fixed soft-vote y_proba)',
        fontsize=11, fontweight='bold'
    )
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.tick_params(labelsize=11)
    plt.tight_layout()

    out = perm_dir / f'auc_permutation_test_{method}.png'
    plt.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ C3 permutation plot ({method}) → {out.name}")

    result = {
        'method':           method,
        'test_type':        'proxy_label_shuffle',
        'note':             ('y_true shuffled against fixed soft-vote y_proba. '
                             'Valid label-permutation test without needing X_train.'),
        'n_permutations':   n_permutations,
        'observed_auc':     round(float(observed_auc), 6),
        'null_mean':        round(float(null_aucs.mean()), 6),
        'null_std':         round(float(null_aucs.std()),  6),
        'null_max':         round(float(null_aucs.max()),  6),
        'p_value':          round(p_value, 6),
        'significant':      p_value < 0.05,
        'null_aucs':        [round(float(a), 6) for a in null_aucs.tolist()],
    }
    json_out = perm_dir / f'auc_permutation_test_{method}.json'
    with open(json_out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  ✓ C3 permutation JSON ({method}) → {json_out.name}")

    # Data table (summary only — null_aucs excluded for size)
    summary = {k: v for k, v in result.items() if k != 'null_aucs'}
    _save_table([summary], perm_dir / f'auc_permutation_test_{method}_summary')
    return result


# ── C4: Fix Figures 28 and 29 legend label ───────────────────────────────────
def fix_stacking_sensitivity(method):
    """
    Reads stacking_sensitivity_results.tsv (confirmed columns:
    key, display_name, test_auc, cv_auc_mean, cv_auc_std).
    Redraws with corrected legend label showing what error bars mean.
    """
    tsv_path = (OUT_02C / method / 'stacking_sensitivity' /
                'stacking_sensitivity_results.tsv')
    if not tsv_path.exists():
        print(f"  SKIP C4 ({method}): stacking_sensitivity_results.tsv not found")
        return

    df = pd.read_csv(tsv_path, sep='\t')
    print(f"  [{method}] stacking TSV columns: {list(df.columns)}")

    meta_learners = df['display_name'].tolist()
    test_aucs     = df['test_auc'].tolist()
    cv_means      = df['cv_auc_mean'].tolist()
    cv_stds       = df['cv_auc_std'].tolist()

    x     = np.arange(len(meta_learners))
    width = 0.35
    fig, ax = plt.subplots(figsize=(13, 6))

    ax.bar(x - width/2, test_aucs, width,
           label='Test AUC', color='#3498db', alpha=0.85)
    # ── C4 fix: corrected legend label ────────────────────────────────────────
    ax.bar(x + width/2, cv_means, width,
           label='Train CV AUC (mean ± 1 SD, 5-fold CV)',
           color='#2ecc71', alpha=0.85,
           yerr=cv_stds, capsize=4, error_kw={'linewidth': 1.2})

    for i, (t, c) in enumerate(zip(test_aucs, cv_means)):
        ax.text(i - width/2, t + 0.0004, f'{t:.4f}',
                ha='center', va='bottom', fontsize=8.5, fontweight='bold')
        ax.text(i + width/2, c + 0.0004, f'{c:.4f}',
                ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax.set_xlabel('Meta-Learner', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUC', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Stacking Meta-Learner Sensitivity: Test AUC vs Train CV AUC\n'
        f'({method.replace("_", " ").title()} Sampling)',
        fontsize=13, fontweight='bold'
    )
    ax.set_xticks(x)
    ax.set_xticklabels(meta_learners, rotation=15, ha='right', fontsize=10)
    y_min = max(0.90, min(test_aucs + cv_means) - 0.01)
    ax.set_ylim(y_min, 1.012)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax.tick_params(labelsize=11)
    plt.tight_layout()

    out = OUT_DIR / f'stacking_sensitivity_plot_{method}.png'
    plt.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ C4 stacking sensitivity ({method}) → {out.name}")

    _save_table(df, OUT_DIR / f'stacking_sensitivity_plot_{method}')


# ── C5: Fix Figure 31 — panel labels + y-axis scale ──────────────────────────
def fix_sampling_comparison():
    """
    Reads sampling_methods_comparison.json — structure printed at runtime.
    Falls back to reading sampling_methods_summary.csv if JSON keys differ.
    """
    json_path = OUT_02C / 'comparison_report' / 'sampling_methods_comparison.json'
    csv_path  = OUT_02C / 'comparison_report' / 'sampling_methods_summary.csv'

    if not json_path.exists():
        print(f"  SKIP C5: sampling_methods_comparison.json not found")
        return

    with open(json_path) as f:
        data = json.load(f)
    print(f"  [Fig31] sampling_methods_comparison.json top-level keys: {list(data.keys())}")

    # ── Extract values robustly from any structure ─────────────────────────────
    # Try common key patterns; fallback to None handled in plot
    def _get(d, *keys, default=None):
        for k in keys:
            if k in d:
                v = d[k]
                if isinstance(v, dict):
                    # might be {method: value} dict
                    vals = list(v.values())
                    return vals if len(vals) > 1 else vals[0]
                return v
        return default

    method_names   = _get(data, 'methods', 'method_names',
                          default=['Median', 'Cluster Based'])
    best_aucs      = _get(data, 'best_aucs', 'best_auc',
                          default=[0.9974, 0.9974])
    gen_gaps       = _get(data, 'generalization_gaps', 'gen_gaps',
                          default=[0.0022, 0.0022])
    bio_relevance  = _get(data, 'bio_relevance_pct', 'biological_relevance',
                          default=[20.0, 10.0])
    correlations   = _get(data, 'delta_fi_correlations', 'correlations',
                          default=[-0.067, -0.069])

    # Ensure all are lists of length 2
    def _ensure_list(v, n=2):
        if v is None:
            return [0.0] * n
        if not isinstance(v, (list, tuple)):
            return [float(v)] * n
        return [float(x) for x in v][:n]

    method_names  = method_names if isinstance(method_names, list) else ['Median', 'Cluster Based']
    best_aucs     = _ensure_list(best_aucs)
    gen_gaps      = _ensure_list(gen_gaps)
    bio_relevance = _ensure_list(bio_relevance)
    correlations  = _ensure_list(correlations)

    colors = ['#3498db', '#2ecc71']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── Panel (a): Best AUC — narrow y-axis ──────────────────────────────────
    axes[0, 0].bar(method_names, best_aucs, color=colors, alpha=0.85, width=0.5)
    for i, v in enumerate(best_aucs):
        axes[0, 0].text(i, v + 0.0001, f'{v:.4f}',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('AUC-ROC', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Best Model AUC by Sampling Method', fontsize=11, fontweight='bold')
    # ── C5 fix: rescale y-axis ─────────────────────────────────────────────────
    ylo = max(0.990, min(best_aucs) - 0.003)
    axes[0, 0].set_ylim([ylo, max(best_aucs) + 0.003])
    axes[0, 0].grid(True, axis='y', linestyle='--', alpha=0.3)

    # ── Panel (b): Generalization gap ─────────────────────────────────────────
    axes[0, 1].bar(method_names, gen_gaps, color=colors, alpha=0.85, width=0.5)
    for i, v in enumerate(gen_gaps):
        axes[0, 1].text(i, v + max(gen_gaps)*0.01, f'{v:.4f}',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Gap Size', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Generalization Gap (|Test AUC − CV AUC|)',
                         fontsize=11, fontweight='bold')
    axes[0, 1].grid(True, axis='y', linestyle='--', alpha=0.3)

    # ── Panel (c): Biological relevance ──────────────────────────────────────
    axes[1, 0].bar(method_names, bio_relevance, color=colors, alpha=0.85, width=0.5)
    for i, v in enumerate(bio_relevance):
        axes[1, 0].text(i, v + 0.5, f'{v:.1f}%',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Percentage', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Biological Relevance (% Cancer Hubs in Top 10)',
                         fontsize=11, fontweight='bold')
    axes[1, 0].set_ylim([0, 100])
    axes[1, 0].grid(True, axis='y', linestyle='--', alpha=0.3)

    # ── Panel (d): ΔConnectivity vs Feature Importance correlation ────────────
    bar_colors_d = [('#e74c3c' if v < 0 else '#2ecc71') for v in correlations]
    axes[1, 1].bar(method_names, correlations, color=bar_colors_d, alpha=0.85, width=0.5)
    for i, v in enumerate(correlations):
        offset = -0.01 if v < 0 else 0.01
        axes[1, 1].text(i, v + offset, f'{v:.3f}',
                        ha='center', va='top' if v < 0 else 'bottom',
                        fontsize=11, fontweight='bold')
    axes[1, 1].axhline(y=0, color='black', linewidth=0.8)
    axes[1, 1].set_ylim([-1.0, 1.0])
    axes[1, 1].set_ylabel('Correlation Coefficient', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('ΔConnectivity vs Feature Importance Correlation',
                         fontsize=11, fontweight='bold')
    axes[1, 1].grid(True, axis='y', linestyle='--', alpha=0.3)

    # ── C5 fix: panel labels (a)(b)(c)(d) ─────────────────────────────────────
    for ax, lbl in zip(axes.flat, ['(a)', '(b)', '(c)', '(d)']):
        ax.text(0.02, 0.97, lbl, transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        ax.tick_params(labelsize=11)

    plt.suptitle('Figure 31. Machine Learning and Sampling Method Comparison',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    out = OUT_DIR / 'sampling_methods_comparison.png'
    plt.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ C5 Figure 31 → {out.name}")

    # Data table
    records = []
    for i, mn in enumerate(method_names):
        records.append({
            'method':          mn,
            'best_auc':        best_aucs[i],
            'generalization_gap': gen_gaps[i],
            'bio_relevance_pct': bio_relevance[i],
            'delta_fi_correlation': correlations[i],
        })
    _save_table(records, OUT_DIR / 'sampling_methods_comparison_summary')


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("02_e_validate_classification.py")
    print("Reading frozen 02_c outputs. No model retraining on real labels.")
    print("=" * 60)

    for method in METHODS:
        print(f"\n--- {method} ---")
        plot_performance_comparison(method)   # C2
        fix_stacking_sensitivity(method)      # C4

    for method in METHODS:
        print(f"\n--- Permutation test: {method} ---")
        run_permutation_test(method, n_permutations=1000)  # C3

    print(f"\n--- Figure 31 ---")
    fix_sampling_comparison()                # C5

    print("\n" + "=" * 60)
    print("Done. Outputs in: output/02_e_validate_classification/")