"""
03_d_candidate_tables.py

Produces four formatted gene tables from the frozen 03_a hub ranking:
  1. known_cancer_genes_in_hubs.tsv  — All Tier 1 + Tier 2 with established functions
  2. top10_novel_candidates.tsv      — Top 10 Tier 3 genes with full annotation
  3. simple_top10_known.tsv          — Top 10 known (Tier 1+2) simplified
  4. simple_top10_novel.tsv          — Top 10 novel (Tier 3) simplified
"""

import json
import csv
import pandas as pd
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
HUB_RANKING   = (PROJECT_ROOT / 'output' / '03_a_enhanced_hub_analysis' /
                 'enhanced_hub_ranking.tsv')
GENE_INFO     = PROJECT_ROOT / 'output' / '00_c_gene_info' / 'gene_info_combined.json'
CANCER_CLASS  = (PROJECT_ROOT / 'output' / '00_c_gene_info' /
                 'cancer_gene_classification.json')
OUT_DIR       = PROJECT_ROOT / 'output' / '03_d_candidate_tables'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Curated biological function descriptions for top known cancer genes ───────
KNOWN_FUNCTIONS = {
    # Commented out - using gene_info_combined.json as primary source
}


def load_gene_descriptions(gene_info_path):
    """
    Loads descriptions from individual gene JSON files referenced by
    gene_info_combined.json.
    
    Priority order for descriptions:
    1. 'summary' (MyGene.info) — most readable, plain-text description
    2. 'description' (Ensembl) — functional description
    3. 'gene_description' (Ensembl) — same as description, fallback
    
    Returns:
        dict: Mapping from gene_key or gene_symbol to description string
    """
    desc_map = {}
    
    if not gene_info_path.exists():
        print(f"  WARNING: {gene_info_path.name} not found")
        return desc_map

    print(f"  Loading {gene_info_path.name}...")
    with open(gene_info_path, 'r') as f:
        gene_info = json.load(f)
    
    output_dir = gene_info_path.parent.parent
    
    # Collect all gene file paths from the three divisions
    gene_file_paths = {}
    for division in ['breast_cancer', 'cancer', 'non_cancer']:
        if division in gene_info:
            for gene_key, relative_path in gene_info[division].items():
                abs_path = output_dir / relative_path
                if abs_path.exists():
                    gene_file_paths[gene_key] = abs_path
                else:
                    alt_path = output_dir / '00_c_gene_info' / 'gene' / f"{gene_key}.json"
                    if alt_path.exists():
                        gene_file_paths[gene_key] = alt_path
    
    print(f"  Found {len(gene_file_paths)} individual gene JSON files")
    
    def clean_description(desc):
        """Remove source tags like [Source:HGNC...] and clean up."""
        if not desc or not isinstance(desc, str):
            return desc
        import re
        # Remove [Source:...] tags
        desc = re.sub(r'\[Source:.*?\]', '', desc).strip()
        # Remove trailing periods if they create double punctuation
        desc = re.sub(r'\.\.+', '.', desc)
        return desc
    
    def capitalize_first(desc):
        """Capitalize first letter of description."""
        if desc and isinstance(desc, str) and len(desc) > 0:
            if desc[0].islower():
                return desc[0].upper() + desc[1:]
        return desc
    
    loaded_count = 0
    for gene_key, json_path in gene_file_paths.items():
        try:
            with open(json_path, 'r') as f:
                gene_data = json.load(f)
            
            desc = None
            
            if isinstance(gene_data, dict):
                if 'gene_info' in gene_data and isinstance(gene_data['gene_info'], dict):
                    gi = gene_data['gene_info']
                    desc = (gi.get('summary') or
                            gi.get('description') or
                            gi.get('gene_description'))
                
                if not desc:
                    desc = (gene_data.get('summary') or
                            gene_data.get('description') or
                            gene_data.get('gene_description'))
            
            if desc and isinstance(desc, str) and len(desc.strip()) > 0:
                desc = desc.strip()
                desc = clean_description(desc)
                desc = capitalize_first(desc)
                
                desc_map[gene_key] = desc
                if '|' in gene_key:
                    symbol = gene_key.split('|')[1]
                    desc_map[symbol] = desc
                
                bare_ensg = gene_key.split('|')[0].split('.')[0]
                desc_map[bare_ensg] = desc
                loaded_count += 1
                
        except Exception as e:
            continue
    
    print(f"  Loaded descriptions for {loaded_count:,} genes")
    print(f"  Total entries in desc_map: {len(desc_map):,}")
    
    return desc_map


def get_description(gene_id, gene_symbol, desc_map):
    """Lookup description with fallback chain."""
    # 1. Curated dict (commented out, but kept for structure)
    if gene_symbol in KNOWN_FUNCTIONS:
        return KNOWN_FUNCTIONS[gene_symbol]
    
    # 2. gene_info lookup
    for key in [gene_symbol, gene_id, gene_id.split('.')[0].split('|')[0]]:
        if key in desc_map:
            return desc_map[key]
    
    # 3. Fallback
    return 'No established function description available.'


def extract_symbol(gene_id_str):
    """Extract gene symbol from 'ENSG..|SYMBOL' format."""
    s = str(gene_id_str)
    return s.split('|')[1] if '|' in s else s


def save_table(records, stem):
    """Save list-of-dicts as TSV + JSON."""
    if not records:
        print(f"  WARNING: no records to save for {stem.name}")
        return
    tsv_path = stem.with_suffix('.tsv')
    json_path = stem.with_suffix('.json')
    keys = list(records[0].keys())
    with open(tsv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys, delimiter='\t')
        w.writeheader()
        w.writerows(records)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"  ✓ {tsv_path.name}  ({len(records)} rows)")
    print(f"  ✓ {json_path.name}")


def save_simple_table(records, stem):
    """Save simple table as TSV only (clean format, no JSON)."""
    if not records:
        print(f"  WARNING: no records to save for {stem.name}")
        return
    tsv_path = stem.with_suffix('.tsv')
    keys = ['rank', 'original_rank', 'gene_symbol', 'ensembl_id', 'type', 'established_function']
    with open(tsv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys, delimiter='\t')
        w.writeheader()
        w.writerows(records)
    print(f"  ✓ {tsv_path.name}  ({len(records)} rows)")


def generate_tables(hub_df, desc_map):
    """Generate all four tables (existing + simple)."""
    print(f"\n  Hub ranking columns: {list(hub_df.columns)}")

    rel_col = next(
        (c for c in hub_df.columns if 'cancer_relevance' in c.lower()), None
    ) or next(
        (c for c in hub_df.columns if 'relevance' in c.lower()), None
    )
    score_col = next(
        (c for c in hub_df.columns
         if 'composite_score' in c.lower() or 'enhanced_score' in c.lower()), None
    )
    gene_col = next(
        (c for c in hub_df.columns if c.lower() == 'gene'), 'gene'
    )
    sym_col = next(
        (c for c in hub_df.columns if 'symbol' in c.lower()), None
    )
    rank_col = next(
        (c for c in hub_df.columns if 'composite_rank' in c.lower() or 'rank' in c.lower()), None
    )

    if rel_col is None:
        print(f"  ERROR: cannot find cancer_relevance column")
        return

    print(f"  Using: gene='{gene_col}', symbol='{sym_col}', "
          f"relevance='{rel_col}', score='{score_col}', rank='{rank_col}'")

    def build_full_row(row, tier_label):
        """Build row for full table (with scores)."""
        gene_id = str(row[gene_col])
        symbol = (str(row[sym_col]) if sym_col and sym_col in row
                  else extract_symbol(gene_id))
        desc = get_description(gene_id, symbol, desc_map)
        return {
            'rank': int(row[rank_col]) if rank_col and rank_col in row
                    else int(row.name) + 1,
            'gene_symbol': symbol,
            'ensembl_id': gene_id,
            'tier': tier_label,
            'composite_score': round(float(row[score_col]), 4) if score_col else None,
            'feature_importance': round(float(row['feature_importance']), 6)
                                  if 'feature_importance' in row else None,
            'delta_connectivity': round(float(row['delta_connectivity']), 2)
                                  if 'delta_connectivity' in row else None,
            'established_function': desc,
        }

    def build_simple_row(row, idx, type_label):
        """Build row for simple table (no scores, includes original_rank)."""
        gene_id = str(row[gene_col])
        symbol = (str(row[sym_col]) if sym_col and sym_col in row
                  else extract_symbol(gene_id))
        desc = get_description(gene_id, symbol, desc_map)
        # Get original rank from the ranking
        original_rank = int(row[rank_col]) if rank_col and rank_col in row else int(row.name) + 1
        return {
            'rank': idx + 1,
            'original_rank': original_rank,
            'gene_symbol': symbol,
            'ensembl_id': gene_id,
            'type': type_label,
            'established_function': desc,
        }

    def get_simple_type(cancer_relevance):
        """Convert cancer_relevance to simple type label."""
        if cancer_relevance == 'breast_cancer':
            return 'Breast Cancer'
        elif cancer_relevance == 'cancer':
            return 'General Cancer'
        else:
            return 'Unannotated'

    # ==================== EXISTING TABLES (FULL) ====================
    
    # Table 1: All known cancer genes (Tier 1 + Tier 2) - FULL
    known = hub_df[hub_df[rel_col].isin(['breast_cancer', 'cancer'])].copy()
    tier_map = {'breast_cancer': 'Tier 1 — Breast Cancer',
                'cancer': 'Tier 2 — General Cancer'}

    known_records = []
    for _, row in known.iterrows():
        known_records.append(build_full_row(row, tier_map.get(row[rel_col], row[rel_col])))

    known_records.sort(key=lambda x: x['composite_score'] or 0, reverse=True)
    print(f"\n  Known cancer genes found (full): {len(known_records)}")
    save_table(known_records, OUT_DIR / 'known_cancer_genes_in_hubs')

    # Table 2: Top 10 novel candidates - FULL
    novel = hub_df[hub_df[rel_col] == 'non_cancer'].copy()
    if score_col:
        novel = novel.sort_values(score_col, ascending=False)
    novel = novel.head(10)

    novel_records = []
    for _, row in novel.iterrows():
        novel_records.append(build_full_row(row, 'Tier 3 — Novel / Unannotated'))

    print(f"\n  Top 10 novel candidates (full):")
    for r in novel_records:
        print(f"    {r['rank']:>3}. {r['gene_symbol']:<12} "
              f"score={r['composite_score']}  delta_conn={r['delta_connectivity']}")
    save_table(novel_records, OUT_DIR / 'top10_novel_candidates')

    # ==================== NEW SIMPLE TABLES (TOP 10 ONLY) ====================
    
    # Simple Table 1: Top 10 known cancer genes (by composite score)
    known_top10 = known.sort_values(score_col, ascending=False).head(10)
    simple_known_records = []
    for idx, (_, row) in enumerate(known_top10.iterrows()):
        simple_known_records.append(
            build_simple_row(row, idx, get_simple_type(row[rel_col]))
        )
    
    print(f"\n  Simple top 10 known cancer genes:")
    for r in simple_known_records:
        print(f"    {r['rank']}. {r['gene_symbol']} (original rank: {r['original_rank']}) - {r['type']}")
    save_simple_table(simple_known_records, OUT_DIR / 'simple_top10_known')

    # Simple Table 2: Top 10 novel candidates (by composite score)
    simple_novel_records = []
    for idx, (_, row) in enumerate(novel.iterrows()):
        simple_novel_records.append(
            build_simple_row(row, idx, 'Unannotated')
        )
    
    print(f"\n  Simple top 10 novel candidates:")
    for r in simple_novel_records:
        print(f"    {r['rank']}. {r['gene_symbol']} (original rank: {r['original_rank']})")
    save_simple_table(simple_novel_records, OUT_DIR / 'simple_top10_novel')


if __name__ == '__main__':
    print("=" * 60)
    print("03_d_candidate_tables.py")
    print("Generating gene tables from frozen 03_a hub ranking")
    print("=" * 60)

    if not HUB_RANKING.exists():
        print(f"ERROR: {HUB_RANKING} not found. Run 03_a first.")
        raise SystemExit(1)

    hub_df = pd.read_csv(HUB_RANKING, sep='\t')
    print(f"  Loaded {len(hub_df):,} genes from {HUB_RANKING.name}")

    desc_map = load_gene_descriptions(GENE_INFO)
    generate_tables(hub_df, desc_map)

    print("\nDone. Outputs:")
    print(f"  {OUT_DIR / 'known_cancer_genes_in_hubs.tsv'} (all known, full details)")
    print(f"  {OUT_DIR / 'top10_novel_candidates.tsv'} (top 10 novel, full details)")
    print(f"  {OUT_DIR / 'simple_top10_known.tsv'} (top 10 known, simplified, with original_rank)")
    print(f"  {OUT_DIR / 'simple_top10_novel.tsv'} (top 10 novel, simplified, with original_rank)")