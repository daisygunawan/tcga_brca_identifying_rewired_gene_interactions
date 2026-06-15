# TCGA-BRCA Differential Co-expression Network Analysis
### Identifying Rewired Gene Interactions in Breast Cancer Using Graph-Based ML Classification

**Daisy Christina Gunawan** · Sampoerna University · Faculty of Engineering and Technology

---

## Overview

This pipeline performs a systematic, architecture-wide comparison of gene co-expression networks between breast tumour and adjacent-normal tissue using TCGA-BRCA RNA-Seq data. It combines Spearman-based network construction, differential co-expression analysis (DCEA), ensemble machine learning classification, and pathway enrichment to identify and prioritise candidate regulatory driver genes.

**Key findings from this pipeline:**
- 98.8% collapse in global co-expression connectivity in tumour tissue (11.4M → 137K edges at |r| ≥ 0.7)
- 38,845 significantly rewired gene pairs (|Δr| ≥ 0.8, FDR < 0.05), 100% gain-directional
- 97.1% tumour/normal classification accuracy (AUC 0.9983) from hub connectivity alone
- ~92% of diagnostic signal from previously unannotated genes (no prior breast cancer annotation)

---

## Requirements

### System
- Python **3.8 or higher**
- Minimum **16 GB RAM** recommended (correlation matrix computation for ~13K genes is memory-intensive)
- macOS, Linux, or Windows (macOS users: see XGBoost note below)

### Python Dependencies

Install all dependencies via:
```bash
pip install -r requirements.txt
```

Core packages used across the pipeline:

| Package | Purpose |
|---|---|
| `pandas`, `numpy` | Data loading, matrix operations |
| `scipy` | Fisher Z-transformation, statistical tests |
| `statsmodels` | FDR (Benjamini–Hochberg) correction |
| `networkx` | Graph construction and network metrics |
| `scikit-learn` | Random Forest, cross-validation, scaling |
| `xgboost` | Gradient boosting classifier |
| `matplotlib`, `seaborn` | Plotting and visualisation |
| `tqdm` | Progress bars for long operations |
| `gseapy` | Pathway enrichment (Enrichr / GSEA) |
| `requests` | Ensembl and MyGene.info API calls (script 00_c) |

### macOS — XGBoost dependency
```bash
brew install libomp
```

### Internet access
Script `00_c_gene_info.py` queries **Ensembl REST API** and **MyGene.info** to annotate genes. An internet connection is required for this step. All responses are cached locally so subsequent runs do not repeat API calls.

---

## Data Setup

1. Download TCGA-BRCA RNA-Seq data from the [GDC Data Portal](https://portal.gdc.cancer.gov/).
   - Data type: Gene Expression Quantification
   - Workflow: STAR — Counts (TPM values used)
   - Expected: ~1,231 files (1,118 tumour + 113 adjacent-normal)

2. Place files in the following structure:
```
data/
├── files/                          # Individual .tsv RNA-Seq files (one per sample)
│   ├── <uuid>/
│   │   └── *.rna_seq.augmented_star_gene_counts.tsv
│   └── ...
└── gdc_sample_sheet*.tsv           # GDC sample sheet mapping file IDs to metadata
```

3. Validate setup:
```bash
python -c "from utils.config import load_config; load_config()"
```

---

## Configuration

All parameters are controlled from a single file: `code/configs/config.yaml`

Key settings to review before running:

```yaml
network_analysis:
  correlation_methods: ["spearman", "pearson"]   # Methods to compute
  primary_correlation_method: "spearman"          # Used in all downstream steps
  correlation_thresholds: [0.6, 0.7, 0.8]        # Sensitivity range; 0.7 is primary
  primary_threshold: 0.7
  min_effect_size: 0.8                            # |Δr| threshold for rewiring
  fdr_threshold: 0.05
  permutation_n: 0                                # 0 = use Fisher Z only (recommended)

hub_analysis:
  top_hubs_count: 250                             # Genes passed to enrichment
  feature_importance_weight: 0.6                  # Composite score: 60% ML importance
  delta_connectivity_weight: 0.4                  # 40% structural rewiring

classification:
  top_n_features: 100                             # Hub genes used as classifier features
  n_tumor_samples_select: 113                     # Matched to normal sample count
  cv_folds: 5
```

> **Memory note:** Running both Pearson and Spearman at three thresholds (default) stores 12 network files. To reduce memory/disk use, set `correlation_methods: ["spearman"]` and `correlation_thresholds: [0.7]`.

---

## Running the Pipeline

Scripts must be run **in order** — each script depends on outputs from previous steps.

```bash
# Phase 0: Data preparation and gene annotation
python code/00_a_analyse_raw_data.py
python code/00_b_data_preprocess.py
python code/00_c_gene_info.py           # Requires internet
python code/00_d_refresh_gene_classification.py   # Optional: re-run if gene lists updated

# Phase 1: Network construction and visualisation
python code/01_a_build_correlation_matrices.py    # Computationally intensive
python code/01_b_network_analysis.py              # Memory-intensive
python code/01_c_network_visualization.py

# Phase 2: Differential analysis and classification
python code/02_a_differential_analysis.py         # ~89.5M gene pairs tested
python code/02_b_dcea_viz_enrich.py
python code/02_c_sample_classification.py

# Phase 3: Hub prioritisation and functional characterisation
python code/03_a_enhanced_hub_analysis.py
python code/03_b_functional_characterization.py

# Optional: Poster-quality chart export
python code/03_c_poster_charts.py
```

---

## Pipeline Scripts — Summary

### Phase 0 — Data Preparation

#### `00_a_analyse_raw_data.py`
**Purpose:** Initial analysis of all raw TCGA TSV files. Reads the GDC sample sheet, classifies files as tumour or normal, computes per-gene TPM statistics, identifies protein-coding genes, and detects duplicate gene names.

**Inputs:** `data/files/` (raw TSVs), `data/gdc_sample_sheet*.tsv`

**Key outputs (`output/00_a_analyse_raw_data/`):**

| File | Contents |
|---|---|
| `samples_stats.json` | Count of tumour (1,118) and normal (113) samples |
| `genes_protein_coding.json` | List of 19,962 protein-coding gene IDs |
| `genes_list.json` | Full gene universe from GENCODE v36 (60,660 features) |
| `detailed_statistics.json` | Per-gene TPM statistics across all samples |
| `stats.json` | Run summary and sample counts |

---

#### `00_b_data_preprocess.py`
**Purpose:** Filters genes to protein-coding only, applies expression threshold (TPM > 1.0 in ≥50% of samples per condition), and constructs the final expression matrices used throughout the pipeline.

**Inputs:** `output/00_a_analyse_raw_data/`

**Key outputs (`output/00_b_data_preprocess/`):**

| File | Contents |
|---|---|
| `matrices/tumor_matrix.tsv` | Expression matrix: 13,383 genes × 1,118 tumour samples |
| `matrices/normal_matrix.tsv` | Expression matrix: 13,383 genes × 113 normal samples |
| `matrices_preview/` | First-rows previews for quick inspection |
| `metadata/filtered_genes.json` | Final 13,383 high-confidence gene IDs |
| `metadata/sample_mapping.json` | File ID to sample ID mapping |

---

#### `00_c_gene_info.py`
**Purpose:** Annotates all 13,383 genes by querying Ensembl REST API and MyGene.info. Classifies each gene into three cancer relevance tiers: **Tier 1** (breast cancer-specific, 338 genes), **Tier 2** (general cancer, 1,660 genes), **Tier 3** (unannotated/novel, 11,375 genes). Results are cached to avoid repeat API calls.

**Inputs:** `output/00_b_data_preprocess/metadata/`

**Key outputs (`output/00_c_gene_info/`):**

| File | Contents |
|---|---|
| `gene_info_combined.json` | Merged annotation for all 13,373 classified genes |
| `cancer_gene_classification.json` | Tier assignment per gene |
| `gene/` | Individual JSON per gene (symbol, description, cancer relevance) |
| `api_responses/` | Cached raw responses from Ensembl and MyGene.info |

---

#### `00_d_refresh_gene_classification.py` *(optional)*
**Purpose:** Re-applies cancer gene classification using updated curated lists (COSMIC, TCGA, ClinVar) **without** re-running API calls. Use this when the curated gene lists in `utils/cancer_gene_lists.py` are updated.

**Key outputs:** Updated `gene_info_combined.json`, backup copy, `refresh_classification_report.json`

---

### Phase 1 — Network Construction

#### `01_a_build_correlation_matrices.py`
**Purpose:** Computes pairwise Spearman and Pearson correlation matrices for all 13,383 genes × 13,383 genes separately for tumour and normal conditions. Saves as compressed `.npz` files.

**Inputs:** `output/00_b_data_preprocess/matrices/`

**Key outputs (`output/01_a_build_correlation_matrices/matrices/`):**

| File | Contents |
|---|---|
| `normal_corr_spearman.npz` | 13,383 × 13,383 Spearman correlation matrix, normal tissue |
| `tumor_corr_spearman.npz` | 13,383 × 13,383 Spearman correlation matrix, tumour tissue |
| `normal_corr_pearson.npz` | Pearson equivalent (normal) |
| `tumor_corr_pearson.npz` | Pearson equivalent (tumour) |
| `01_a_result_info.json` | Matrix shapes, correlation statistics, run summary |

> ⚠️ Each `.npz` file is ~1.3 GB. Four files total (~5.2 GB). Ensure sufficient disk space.

---

#### `01_b_network_analysis.py`
**Purpose:** Builds undirected NetworkX graphs by thresholding correlation matrices at |r| ≥ 0.6, 0.7, and 0.8. Computes global network metrics (edge count, density, clustering coefficient, connected components, median degree) for each method/threshold combination.

**Inputs:** `output/01_a_build_correlation_matrices/matrices/`

**Key outputs (`output/01_b_network_analysis/`):**

| File | Contents |
|---|---|
| `global_metrics_comparison.json` | Network metrics for all 12 method/threshold combinations |
| `gml/*.gml` | Human-readable network files (12 total; compatible with Cytoscape/Gephi) |
| `pickle/*.pkl` | Fast-loading Python network files (12 total) |
| `01_b_result_info.json` | Run summary, primary network stats |

---

#### `01_c_network_visualization.py`
**Purpose:** Generates comparative visualisations of network structure — degree distributions, rank-degree topology, hub neighbourhood comparisons, edge weight CDFs, and a Spearman vs Pearson performance dashboard. Also produces paired hub visualisations (normal vs tumour) for top breast cancer genes, goldilocks pairs, and differential hub comparisons.

**Inputs:** `output/01_b_network_analysis/`, `output/00_c_gene_info/`

**Key outputs (`output/01_c_network_visualization/`):**

| File | Contents |
|---|---|
| `spearman/01_spearman_degree_distribution_overlay.png` | KDE overlay of tumour vs normal degree (Figure 4.2A) |
| `spearman/01_spearman_rank_degree_topology.png` | Log-log rank-degree collapse plot (Figure 4.2B) |
| `method_comparison/01_edge_type_breakdown.png` | Spearman vs Pearson edge type comparison (Figure 4.3A) |
| `method_comparison/02_hub_preservation_analysis.png` | Top-10 hub consistency between methods (Figure 4.3B) |
| `method_comparison/03_unified_performance_dashboard.png` | Multi-metric comparison dashboard (Figure 4.3C) |
| `spearman/03_hub_analysis_centric/breast_cancer_*_paired.png` | Normal/tumour paired hub plots for DCAF1, FANCI, ARID4B |
| `spearman/03_hub_analysis_centric/goldilocks_*_paired.png` | Goldilocks pair visualisations (e.g., MGA) |
| `*/*.json` | Data files for every chart (for reproducibility) |

---

### Phase 2 — Differential Analysis and Classification

#### `02_a_differential_analysis.py`
**Purpose:** Computes differential co-expression across all ~89.5 million gene pairs using Fisher Z-transformation for p-values and Benjamini–Hochberg correction for FDR. Identifies significantly rewired pairs (|Δr| ≥ 0.8, FDR < 0.05) and computes per-gene differential connectivity (ΔConn = Σ|r_tumor| − Σ|r_normal| across all partners).

**Inputs:** `output/01_a_build_correlation_matrices/matrices/`

**Key outputs (`output/02_a_differential_analysis/`):**

| File | Contents |
|---|---|
| `differential_coexpression_sig.tsv` | 38,845 significantly rewired gene pairs with Δr, p-value, FDR |
| `differential_coexpression_all.tsv` | All tested pairs (large file) |
| `differential_connectivity.tsv` | ΔConnectivity score per gene (all 13,383 genes) |
| `biological_interpretation.json` | Summary: mean Δr, direction counts, gain/loss breakdown |
| `effect_size_distribution.json` | Distribution statistics of |Δr| across all pairs |

---

#### `02_b_dcea_viz_enrich.py`
**Purpose:** Annotates the top 250 rewired hub genes with cancer relevance tier, gene function, and description from `00_c`. Generates 7+ visualisations covering effect size distribution, ΔConnectivity rankings, rewired edge scatter, and rewiring flow summary. Produces the annotated hub files used downstream by `02_c` (feature selection) and `03_a` (composite scoring).

**Inputs:** `output/02_a_differential_analysis/`, `output/00_c_gene_info/`

**Key outputs (`output/02_b_dcea_viz_enrich/`):**

| File | Contents |
|---|---|
| `annotated_hubs.json` / `.tsv` | Top 250 rewired hubs with tier, Δconn, gene metadata |
| `annotated_hubs_breast_cancer.json` | Tier 1 (breast cancer) subset |
| `annotated_hubs_cancer.json` | Tier 2 (general cancer) subset |
| `annotated_hubs_non_cancer.json` | Tier 3 (novel/unannotated) subset |
| `viz/02_effect_size_distribution_chart.png` | Effect size distribution (Figure 4.4) |
| `viz/04_delta_connectivity_bar.png` | Top 20 genes by ΔConnectivity (Figure 4.7) |
| `viz/06_rewired_edge_scatter.png` | Rewiring mechanism quadrant analysis (Figure 4.5) |

---

#### `02_c_sample_classification.py`
**Purpose:** Trains and evaluates four classifiers (Random Forest, XGBoost, Soft Ensemble, Hard Ensemble) using the connectivity profiles of the top 100 hub genes as features. Runs two independent sampling strategies — **Median-Based** and **Cluster-Based** (K-means k=5) — to handle the 10:1 class imbalance and assess generalisability. Generates consensus feature importance rankings used by `03_a`.

**Inputs:** `output/02_b_dcea_viz_enrich/annotated_hubs.json`, `output/00_b_data_preprocess/matrices/`

**Key outputs (`output/02_c_sample_classification/sampling_comparison/`):**

| File | Contents |
|---|---|
| `cluster_based/classification_performance_comparison.tsv` | Acc / AUC / F1 for all 4 models (cluster-based) |
| `median/classification_performance_comparison.tsv` | Acc / AUC / F1 for all 4 models (median) |
| `cluster_based/classification_feature_importance.tsv` | Per-gene importance (consensus RF+XGBoost) |
| `*/classification_roc_curves.png` | ROC curves per sampling method |
| `comparison_report/sampling_methods_comparison.json` | Side-by-side sampling method comparison (Figure 4.6) |
| `enhancements/ensemble_benefits_analysis.json` | Ensemble vs individual model benefit analysis |
| `consensus_predictive_ranking.tsv` / `.json` | Final consensus gene rankings (rank-sum across both methods) |

**Classification results summary:**

| Model | Accuracy | AUC (Cluster-Based) |
|---|---|---|
| Random Forest | **97.06%** | **0.9983** |
| Soft Ensemble | 95.59% | 0.9957 |
| Hard Ensemble | 95.59% | 0.9957 |
| XGBoost | 94.12% | 0.9939 |

---

### Phase 3 — Hub Prioritisation and Functional Characterisation

#### `03_a_enhanced_hub_analysis.py`
**Purpose:** Integrates three evidence streams — ML feature importance (from `02_c`), structural ΔConnectivity (from `02_b`), and cancer relevance tier (from `00_c`) — into a single **Composite Hub Score** per gene. Formula: `Score = (0.6 × norm(FI) + 0.4 × norm(|ΔConn|)) × bonus × 1000`, where bonus = 1.2× for Tier 1, 1.1× for Tier 2, 1.0× for Tier 3. Generates paired hub visualisations (normal vs tumour) for the top 9 consensus genes.

**Inputs:** `output/02_b_dcea_viz_enrich/`, `output/02_c_sample_classification/`, `output/01_b_network_analysis/`

**Key outputs (`output/03_a_enhanced_hub_analysis/`):**

| File | Contents |
|---|---|
| `enhanced_hub_ranking.tsv` | All 250 hubs ranked by composite score with all metrics |
| `driver_candidates.json` | Top prioritised novel driver candidates |
| `viz/cancer_relevance_pie.png` | Tier composition of top 50 hubs (Figure 4.1) |
| `viz/enhanced_hub_scores.png` | Composite score distribution |
| `consensus_hubs/01_overall_1_MAB21L1_paired.png` | Normal/tumour paired plot — MAB21L1 (#1 overall) |
| `consensus_hubs/04_breast_cancer_121_GRB7_paired.png` | Paired plot — GRB7 (#1 breast cancer hub) |
| `consensus_hubs/consensus_hubs_metadata.json` | Degree counts for all 9 visualised consensus hubs |

---

#### `03_b_functional_characterization.py`
**Purpose:** Performs stratified pathway enrichment analysis using Enrichr (via gseapy) across three databases — GO Biological Process 2023, KEGG 2021 Human, and Reactome 2022 — tested separately on all hubs, cancer-annotated hubs (Tiers 1–2), and novel hubs (Tier 3). Also runs GSEA preranked analysis. Enables comparison of known-biology validation against novel discovery signal.

**Inputs:** `output/03_a_enhanced_hub_analysis/`

**Key outputs (`output/03_b_functional_characterization/`):**

| File | Contents |
|---|---|
| `enrichment_summary_stats.tsv` / `.json` | Pathway counts by gene set × database (Table 4.7) |
| `cancer_KEGG_2021_Human_significant.tsv` | Significant KEGG pathways — cancer gene set |
| `novel_GO_Biological_Process_2023_significant.tsv` | Significant GO terms — novel gene set |
| `all_combined_significant.tsv` | All significant pathways across all databases |
| `viz/discovery_vs_validation.png` | Cancer vs novel enrichment comparison (Figure 4.12) |
| `gsea_results/` | GSEA preranked reports per database |
| `functional_insights.json` | Top enriched pathways with biological interpretation |

---

#### `03_c_poster_charts.py` *(optional)*
**Purpose:** Standalone script generating two high-resolution (300 DPI) poster-quality charts — the rewiring magnitude distribution and the ROC curve with hub importance panel.

**Key outputs (`output/03_c_poster_charts/`):**

| File | Contents |
|---|---|
| `image_3b_rewiring_magnitude.png` | Effect size distribution chart for poster |
| `image_3c_roc_and_hubs.png` | ROC curve + top hub importance bar chart |

---

## Output Directory Structure

```
output/
├── 00_a_analyse_raw_data/          # Gene lists, sample counts, raw statistics
├── 00_b_data_preprocess/
│   ├── matrices/                   # tumor_matrix.tsv, normal_matrix.tsv (13,383 × samples)
│   └── metadata/                   # filtered_genes.json, sample_mapping.json
├── 00_c_gene_info/                 # Gene annotations, cancer tier classifications, API cache
├── 01_a_build_correlation_matrices/
│   └── matrices/                   # 4 × .npz correlation matrices (~1.3 GB each)
├── 01_b_network_analysis/
│   ├── gml/                        # 12 × .gml network files (Cytoscape/Gephi compatible)
│   └── pickle/                     # 12 × .pkl network files (Python/NetworkX)
├── 01_c_network_visualization/
│   ├── spearman/                   # Degree distributions, hub plots, paired visualisations
│   ├── pearson/                    # Equivalent plots for Pearson
│   └── method_comparison/          # Spearman vs Pearson comparative charts
├── 02_a_differential_analysis/     # Rewired gene pairs TSVs, ΔConnectivity, effect sizes
├── 02_b_dcea_viz_enrich/
│   ├── annotated_hubs*.json        # Top 250 hubs stratified by cancer tier
│   └── viz/                        # 7 publication-quality chart PNGs + data JSONs
├── 02_c_sample_classification/
│   └── sampling_comparison/
│       ├── median/                 # Results from median-based sampling
│       ├── cluster_based/          # Results from cluster-based sampling
│       ├── comparison_report/      # Side-by-side sampling method comparison
│       └── enhancements/           # Ensemble benefit analysis
├── 03_a_enhanced_hub_analysis/
│   ├── enhanced_hub_ranking.tsv    # Final composite-scored hub ranking
│   ├── consensus_hubs/             # 9 × paired normal/tumour network visualisations
│   └── viz/                        # Hub score distribution charts
├── 03_b_functional_characterization/
│   ├── *_significant.tsv           # Enriched pathways per gene set × database
│   ├── gsea_results/               # GSEA preranked reports
│   └── viz/                        # Enrichment comparison charts
└── 03_c_poster_charts/             # High-resolution poster figures
```

---

## Utils Modules

| Module | Purpose |
|---|---|
| `utils/config.py` | Loads and validates `config.yaml`; resolves path placeholders |
| `utils/file.py` | Directory creation, relative path resolution, auto output path generation |
| `utils/genes.py` | Cancer detection logic, gene symbol parsing, API response merging |
| `utils/cancer_gene_lists.py` | Curated lists of breast cancer, general cancer genes (COSMIC, TCGA, ClinVar sources) |
| `utils/chart.py` | Core plotting functions (degree distributions, hub neighbourhood plots) |
| `utils/chart_advanced.py` | Advanced multi-panel charts, dashboards |
| `utils/chart_method_comparison.py` | Spearman vs Pearson comparison visualisations |
| `utils/classification_enhancements.py` | Ensemble benefit analysis, sampling comparison reporting |
| `utils/analysis.py` | Network metric computation helpers |
| `utils/visualization_computational.py` | Paired hub visualisation renderer |
| `utils/color_scheme.py` | Consistent colour palette across all figures |

---

## Troubleshooting

**`FileNotFoundError: No sample sheet found`**
Ensure `data/gdc_sample_sheet*.tsv` exists. Check `paths.sample_sheet` glob pattern in `config.yaml`.

**`MemoryError` in `01_b` or `02_a`**
Reduce scope in `config.yaml`: use `correlation_methods: ["spearman"]` and `correlation_thresholds: [0.7]` only. Alternatively increase system swap space.

**`00_c` API timeouts**
The script retries with exponential backoff. If it fails repeatedly, run again — cached responses are preserved and already-fetched genes are skipped.

**`MemoryError` in `01_a`**
Increase system RAM or reduce `chunk_size` under `analysis_raw` in `config.yaml`. The full 13K × 13K correlation matrix requires ~1.4 GB per method per condition.

**Permutation tests too slow**
Set `permutation_n: 0` in `config.yaml` to use Fisher Z-transformation only (the default and recommended setting for datasets of this scale).

---

## Citation

If you use this pipeline, please cite:

- **TCGA-BRCA dataset:** Cancer Genome Atlas Network (2012). *Nature*, 490(7418), 61–70. https://doi.org/10.1038/nature11412
- **NetworkX:** Hagberg et al. (2008). SciPy 2008 Proceedings.
- **GSEApy / Enrichr:** Fang et al. (2023). *Bioinformatics*, 39(1). https://doi.org/10.1093/bioinformatics/btac757
- **XGBoost:** Chen & Guestrin (2016). *KDD*, 785–794.
- **Random Forests:** Breiman (2001). *Machine Learning*, 45(1), 5–32.
- **Spearman correlation justification:** See thesis Section 4.4.1 (3.18× advantage over Pearson in cancer co-expression data).

---

## License

MIT