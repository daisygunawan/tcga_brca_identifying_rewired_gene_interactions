# Final Year Project — Breast Cancer Gene Expression (TCGA)

---

## Overview

This pipeline analyses TCGA gene expression data (RNA-seq TPM) to build co-expression networks, identify differentially connected hub genes, classify tumour samples, and perform functional enrichment — all aimed at discovering cancer-relevant and potentially novel biomarker genes in breast cancer.

The pipeline is organised into four sequential stages (00–03), each producing structured outputs consumed by the next. Four supplementary scripts (`01_d`, `02_d`, `02_e`, `03_d`) produce corrected figures and formatted tables from frozen upstream outputs without rerunning any model training or network construction.

---

## Project Structure

```
.
├── code/
│   ├── 00_a_analyse_raw_data.py
│   ├── 00_b_data_preprocess.py
│   ├── 00_c_gene_info.py
│   ├── 00_d_refresh_gene_classification.py
│   ├── 01_a_build_correlation_matrices.py
│   ├── 01_b_network_analysis.py
│   ├── 01_c_network_visualization.py
│   ├── 01_d_regenerate_figures.py          ← supplementary
│   ├── 02_a_differential_analysis.py
│   ├── 02_b_dcea_viz_enrich.py
│   ├── 02_c_sample_classification.py
│   ├── 02_d_rewired_scatter.py             ← supplementary
│   ├── 02_e_validate_classification.py     ← supplementary
│   ├── 03_a_enhanced_hub_analysis.py
│   ├── 03_b_functional_characterization.py
│   ├── 03_c_poster_charts.py
│   ├── 03_d_candidate_tables.py            ← supplementary
│   ├── configs/
│   │   └── config.yaml          # Central configuration (paths, parameters)
│   ├── utils/                   # Shared utility modules
│   └── tests/                   # Test scripts and sample output images
└── output/                      # All generated outputs (auto-structured by script name)
```

---

## Pipeline Flow

```
Raw TCGA TSV files + Sample Sheet
         │
         ▼
[00_a] Analyse Raw Data
   → genes_list.json, files_normal/tumor.json, stats.json, samples_list.json
         │
         ▼
[00_b] Data Preprocessing
   → normal_matrix.tsv, tumor_matrix.tsv, gene_metadata.json, sample_mapping.json
         │
         ▼
[00_c] Gene Info (API enrichment)
   → gene_info_combined.json, cancer_gene_classification.json
         │
   [00_d] Refresh Gene Classification  ← optional re-run to update classifications
         │
         ▼
[01_a] Build Correlation Matrices
   → normal/tumor_corr_pearson/spearman.npz
         │
         ▼
[01_b] Network Analysis
   → GML + pickle networks at thresholds 0.6 / 0.7 / 0.8
   → global_metrics_comparison.json
         │
         ▼
[01_c] Network Visualization
   → Degree distribution plots, hub neighbourhood charts, method comparison charts
         │
   [01_d] Regenerate Figures  ← supplementary; reads frozen 01_c JSON, no rerun
         │
         ▼
[02_a] Differential Connectivity / Expression Analysis (DCEA)
   → delta connectivity scores, hub lists, annotated_hubs.tsv
         │
         ├──────────────────────────────────────────────────┐
         ▼                                                  ▼
[02_b] DCEA Visualisation & Enrichment         [02_c] Sample Classification
   → effect size charts, rewired edge               → RF / XGBoost / ensemble models,
     scatter, null model comparison                   ROC curves, feature importance,
         │                                            stacking sensitivity, sampling
   [02_d] Rewired Scatter  ← supplementary;           comparison
          reads frozen 02_a output                        │
                                                 [02_e] Validate Classification
                                                     ← supplementary; reads frozen
                                                       02_c output
         │
         ▼
[03_a] Enhanced Hub Analysis
   → enhanced_hub_ranking.tsv, consensus_hubs/, driver_candidates.json
   → stratified_gene_lists.json (cancer / novel / other)
         │
         ├──────────────────────────────────────────────────┐
         ▼                                                  ▼
[03_b] Functional Characterization             [03_d] Candidate Tables
   → GO / KEGG / Reactome enrichment TSVs,         ← supplementary; reads frozen
     GSEA results, enrichment visualisations          03_a and 00_c output
         │
         ▼
[03_c] Poster Charts
   → Final publication-ready figures
```

---

## Script Reference

### Stage 00 — Data Ingestion & Preparation

#### `00_a_analyse_raw_data.py`
**Purpose:** Initial quality control and survey of the raw TCGA dataset.

Reads all TSV gene expression files in parallel, classifies them as tumour/normal using the sample sheet, aggregates per-gene TPM statistics, detects duplicate gene names, and computes differential expression summaries (fold changes). All outputs are JSON-sorted for reproducibility.

**Key outputs (`output/00_a_analyse_raw_data/`):**
- `genes_list.json` — full gene catalogue
- `files_normal.json`, `files_tumor.json` — classified file lists
- `genes_protein_coding.json`, `genes_non_coding.json`
- `genes_tpm_diff_protein_coding.json`, `genes_tpm_diff_non_coding.json`
- `stats.json`, `samples_list.json`, `detailed_statistics.json`

**Packages:** `pandas`, `numpy`, `multiprocessing`, `tqdm`, `pathlib`, `json`

---

#### `00_b_data_preprocess.py`
**Purpose:** Constructs clean expression matrices for downstream correlation analysis.

Filters genes by expression and quality criteria, pivots data into sample × gene matrices (separate normal and tumour), and exports metadata for gene-to-ID mappings and sample groupings.

**Inputs from:** `output/00_a_analyse_raw_data/`

**Key outputs (`output/00_b_data_preprocess/`):**
- `matrices/normal_matrix.tsv`, `tumor_matrix.tsv` — gene expression matrices (genes × samples)
- `metadata/gene_metadata.json`, `sample_mapping.json`, `filtered_genes.json`

**Packages:** `pandas`, `numpy`, `tqdm`, `pathlib`

---

#### `00_c_gene_info.py`
**Purpose:** Enriches the gene list with biological annotations via external APIs.

Queries the **MyGene.info** and **Ensembl REST** APIs to fetch gene summaries, chromosomal locations, biotypes, and known cancer associations. Matches genes against curated cancer gene lists (CGC, OncoKB). Results are cached locally in `api_responses/`.

**Inputs from:** `output/00_b_data_preprocess/metadata/gene_metadata.json`

**Key outputs (`output/00_c_gene_info/`):**
- `gene_info_combined.json` — merged annotation for all genes
- `cancer_gene_classification.json` — cancer vs novel vs other labels
- `api_responses/ensembl/`, `api_responses/mygene/` — raw cached responses

**Packages:** `requests`, `pandas`, `json`, `tqdm`, `utils.genes`, `utils.cancer_gene_lists`

---

#### `00_d_refresh_gene_classification.py`
**Purpose:** Utility script to re-classify genes without re-querying APIs.

Re-applies cancer gene list matching logic to the cached `gene_info_combined.json`, useful when the curated cancer gene lists are updated without needing to repeat all API calls.

**Inputs from:** `output/00_c_gene_info/gene_info_combined.json`

**Key outputs:** Updates `cancer_gene_classification.json` and writes `refresh_classification_report.json`

**Packages:** `pandas`, `json`, `utils.cancer_gene_lists`, `utils.genes`

---

### Stage 01 — Network Construction & Topology

#### `01_a_build_correlation_matrices.py`
**Purpose:** Computes pairwise gene–gene correlation matrices for both tissue types.

Calculates both **Pearson** and **Spearman** correlations across all genes in the expression matrices, storing results as compressed sparse `.npz` files for memory efficiency.

**Inputs from:** `output/00_b_data_preprocess/matrices/`

**Key outputs (`output/01_a_build_correlation_matrices/matrices/`):**
- `normal_corr_pearson.npz`, `normal_corr_spearman.npz`
- `tumor_corr_pearson.npz`, `tumor_corr_spearman.npz`

**Packages:** `numpy`, `scipy`, `pandas`, `tqdm`

---

#### `01_b_network_analysis.py`
**Purpose:** Constructs co-expression networks by thresholding the correlation matrices.

Applies absolute correlation thresholds (0.6, 0.7, 0.8) to both Pearson and Spearman matrices, producing 12 network variants (2 tissue × 2 methods × 3 thresholds). Computes global graph topology metrics (density, clustering, degree distribution). Networks are saved as both **GML** (portable) and **pickle** (fast loading).

**Inputs from:** `output/01_a_build_correlation_matrices/matrices/`

**Key outputs (`output/01_b_network_analysis/`):**
- `gml/*.gml` — 12 network files for external tools (e.g. Cytoscape)
- `pickle/*.pkl` — 12 network files for Python analysis
- `global_metrics_comparison.json`

**Packages:** `networkx`, `numpy`, `scipy`, `pandas`, `tqdm`, `pickle`

---

#### `01_c_network_visualization.py`
**Purpose:** Generates exploratory visualisations of the network topology.

Produces degree distribution overlays, rank-degree topology plots, component size histograms, and paired hub neighbourhood ego-graphs comparing normal vs tumour networks. Also includes a method comparison panel (Pearson vs Spearman). All figures are accompanied by JSON companion files containing the underlying data for potential downstream re-rendering.

**Inputs from:** `output/01_b_network_analysis/pickle/`, `output/00_c_gene_info/cancer_gene_classification.json`

**Key outputs (`output/01_c_network_visualization/`):**
- Per-method plots: degree distribution, rank-degree topology
- Hub-centric ego-network pairs (normal + tumour side-by-side)
- `method_comparison/` — Pearson vs Spearman comparison charts
- `spearman/*.json` — companion data files for all figures

**Packages:** `networkx`, `matplotlib`, `numpy`, `pandas`, `utils.chart`, `utils.color_scheme`

---

#### `01_d_regenerate_figures.py` *(supplementary)*
**Purpose:** Regenerates Figures 11, 13, and 15 at corrected sizes and with panel labels, without rerunning any network analysis.

Reads only the frozen JSON companion files produced by `01_c`. Figure 11 (degree distribution overlay) is regenerated at `figsize=(18, 8)`. Figure 13 (edge type breakdown) is regenerated at `figsize=(22, 10)` with a log-scale y-axis, necessary because normal edge counts are approximately 83× larger than tumour counts on a linear scale. Figure 15 (Spearman vs Pearson comparison) is regenerated with `(a)` and `(b)` panel labels added.

**Inputs from:** `output/01_c_network_visualization/spearman/*.json` (frozen; 01_c not rerun)

**Key outputs (`output/01_d_regenerate_figures/`):**
- `01_spearman_degree_distribution_overlay.png` — Figure 11, enlarged
- `01_edge_type_breakdown.png` — Figure 13, enlarged with log scale
- `15_spearman_vs_pearson_comparison.png` — Figure 15, with panel labels
- Companion `.tsv` and `.json` data tables for each figure

**Packages:** `matplotlib`, `numpy`, `scipy`, `json`, `pathlib`

---

### Stage 02 — Differential Analysis & Classification

#### `02_a_differential_analysis.py`
**Purpose:** Core **Differential Co-expression Analysis (DCEA)** between normal and tumour networks.

Computes delta connectivity (change in node degree / edge weight) for every gene, identifies rewired edges, and ranks genes by differential connectivity. Applies permutation testing for statistical validation and compares against a null model.

**Inputs from:** `output/01_b_network_analysis/pickle/`, `output/00_c_gene_info/cancer_gene_classification.json`

**Key outputs (`output/02_a_differential_analysis/`):**
- `annotated_hubs.tsv` — hub genes with delta connectivity scores and cancer labels
- `differential_coexpression_sig.tsv` — all statistically significant rewired gene pairs
- `rewired_edges.tsv`, `hub_network_statistics.json`
- `permutation_results.json` — statistical significance of rewiring

**Packages:** `networkx`, `numpy`, `scipy`, `pandas`, `tqdm`, `utils.analysis`

---

#### `02_b_dcea_viz_enrich.py`
**Purpose:** Visualises the DCEA results and performs initial pathway enrichment on differential hubs.

Generates effect size distribution charts, delta connectivity bar charts, rewired edge scatter plots, rewiring flow summaries, and statistical validation summaries. Runs enrichment on the hub gene set via **gseapy**.

**Inputs from:** `output/02_a_differential_analysis/`

**Key outputs (`output/02_b_dcea_viz_enrich/`):**
- `01_` → `09_` numbered chart pairs (`.png` + `.json` metadata each)
- Enrichment results for hub genes

**Packages:** `matplotlib`, `seaborn`, `numpy`, `pandas`, `gseapy`, `utils.chart_advanced`

---

#### `02_c_sample_classification.py`
**Purpose:** Builds and evaluates ML classifiers to distinguish tumour from normal samples using hub gene expression as features.

Trains **Random Forest** and **XGBoost** base models and four ensemble variants — soft-voting, hard-voting, weighted-voting, and stacking. Evaluates all models with stratified 5-fold cross-validation and on a held-out 30% test set. Produces ROC curves, confusion matrices, feature importance rankings, and a stacking meta-learner sensitivity sweep. Compares two sampling strategies (median-based and cluster-based) and outputs a consensus predictive hub ranking.

**Inputs from:** `output/02_a_differential_analysis/annotated_hubs.tsv`, `output/00_b_data_preprocess/matrices/`

**Key outputs (`output/02_c_sample_classification/sampling_comparison/`):**
- Per-strategy subdirs (`cluster_based/`, `median/`): ROC curves, confusion matrices, feature importance TSVs, stacking sensitivity results
- `cv_summary.json`, `classification_performance_comparison.tsv`, `model_evaluation_data.json`
- `consensus_predictive_ranking.tsv`, `consolidated_rankings.tsv`
- `comparison_report/sampling_methods_comparison.json`
- `enhancements/` — ensemble benefit analysis, model heatmap

**Packages:** `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `pandas`, `numpy`, `utils.classification_enhancements`

---

#### `02_d_rewired_scatter.py` *(supplementary)*
**Purpose:** Regenerates the rewired gene pair scatter plot (Figure 18) with corrected sampling, axis labels, and a new dual-mechanism companion figure, without rerunning the DCEA.

Reads the frozen `differential_coexpression_sig.tsv` from `02_a`. Replaces the previous top-200-filtered version with a random sample of 2,000 pairs drawn from all 38,845 significant pairs. Computes `avg_abs_r = (|r_tumor| + |r_normal|) / 2` inline. Produces two figures: the primary quadrant scatter with descriptive axis labels and a Q2/Q3 absence annotation; and a companion dual-mechanism plot using a continuous RdYlGn_r colormap that encodes the same finding without quadrant notation.

**Inputs from:** `output/02_a_differential_analysis/differential_coexpression_sig.tsv` (frozen; 02_a not rerun)

**Key outputs (`output/02_d_rewired_scatter/`):**
- `06_rewired_edge_scatter.png` — primary quadrant scatter (Figure 18)
- `06_rewired_dual_mechanism.png` — companion continuous colormap figure
- Companion `.tsv` and `.json` data tables for both figures

**Packages:** `matplotlib`, `numpy`, `pandas`, `pathlib`, `json`

---

#### `02_e_validate_classification.py` *(supplementary)*
**Purpose:** Produces corrected and new classification figures by reading frozen `02_c` outputs. No model training is performed on real labels.

Generates four outputs: (1) a cross-model performance comparison bar chart (AUC, Accuracy, F1) for all five reported models with CV standard deviation error bars; (2) a proxy permutation test on the soft-vote ensemble — shuffling true class labels 1,000 times against fixed probability scores to empirically rule out data leakage; (3) corrected stacking sensitivity plots for Figures 28 and 29 with an updated error bar legend; (4) a corrected sampling methods comparison (Figure 31) with panel labels and a rescaled AUC y-axis.

**Inputs from:** `output/02_c_sample_classification/sampling_comparison/` (frozen; 02_c not rerun)
- `cv_summary.json`, `classification_performance_comparison.tsv`
- `model_evaluation_data.json`
- `stacking_sensitivity_results.tsv` (per sampling method)
- `comparison_report/sampling_methods_comparison.json`

**Key outputs (`output/02_e_validate_classification/`):**
- `model_performance_comparison_{method}.png` — cross-model comparison chart
- `permutation_test/auc_permutation_test_{method}.png` — permutation null distribution
- `stacking_sensitivity_plot_{method}.png` — corrected Figures 28/29
- `sampling_methods_comparison.png` — corrected Figure 31
- Companion `.tsv` and `.json` data tables for all figures

**Packages:** `scikit-learn`, `matplotlib`, `numpy`, `pandas`, `pathlib`, `json`

---

### Stage 03 — Hub Prioritisation & Functional Annotation

#### `03_a_enhanced_hub_analysis.py`
**Purpose:** Integrates all hub ranking signals into a single prioritised gene list.

Combines delta connectivity, predictive feature importance, and network topology into a composite **enhanced hub score** using min-max normalisation and configurable weights (default: 0.6 feature importance, 0.4 delta connectivity). Applies annotation-tier multipliers (1.2× breast cancer, 1.1× other cancer). Stratifies genes into cancer-known, novel, and other categories. Identifies consensus hubs across multiple network configurations and flags driver gene candidates.

**Inputs from:**
- `output/02_a_differential_analysis/annotated_hubs.tsv`
- `output/02_c_sample_classification/sampling_comparison/consensus_predictive_ranking.tsv`
- `output/00_c_gene_info/cancer_gene_classification.json`

**Key outputs (`output/03_a_enhanced_hub_analysis/`):**
- `enhanced_hub_ranking.tsv` — final scored gene list (all hub genes, ranked)
- `stratified_gene_lists.json` — cancer / novel / other gene sets
- `driver_candidates.json`, `top_250_all_genes.json`
- `consensus_hubs/` — paired ego-network visualisations for top consensus hubs
- `viz/` — cancer relevance pie chart, hub score scatter plots

**Packages:** `networkx`, `numpy`, `pandas`, `matplotlib`, `utils.analysis`, `utils.chart`

---

#### `03_b_functional_characterization.py`
**Purpose:** Pathway and gene-set enrichment analysis of the prioritised hub genes.

Submits stratified gene sets (all hubs, cancer-labelled, novel) to **Enrichr** via gseapy for over-representation analysis against GO Biological Process 2023, KEGG 2021 Human, and Reactome 2022. Also runs GSEA preranked analysis. Produces comparative visualisations across databases and gene subsets.

**Inputs from:**
- `output/03_a_enhanced_hub_analysis/stratified_gene_lists.json`
- `output/03_a_enhanced_hub_analysis/top_hubs_for_enrichment.json`
- `output/03_a_enhanced_hub_analysis/cancer_genes_for_enrichment.json`
- `output/03_a_enhanced_hub_analysis/novel_genes_for_enrichment.json`

**Key outputs (`output/03_b_functional_characterization/`):**
- `*_GO_Biological_Process_2023_significant.tsv`, `*_KEGG_*`, `*_Reactome_*` — per-subset enrichment tables
- `gsea_results/` — GSEA preranked reports
- `viz/` — database comparison, heatmap, discovery vs validation, Q-Q plots

**Packages:** `gseapy`, `pandas`, `matplotlib`, `seaborn`, `numpy`, `scipy`

---

#### `03_c_poster_charts.py`
**Purpose:** Generates polished, publication/poster-ready composite figures.

Assembles final charts combining rewiring magnitude and classification performance (ROC + hub summary) into layout-optimised images for presentation.

**Inputs from:** `output/02_b_dcea_viz_enrich/`, `output/02_c_sample_classification/`, `output/03_a_enhanced_hub_analysis/`

**Key outputs (`output/03_c_poster_charts/`):**
- `image_3b_rewiring_magnitude.png`
- `image_3c_roc_and_hubs.png`

**Packages:** `matplotlib`, `numpy`, `pandas`, `PIL` (Pillow)

---

#### `03_d_candidate_tables.py` *(supplementary)*
**Purpose:** Produces formatted gene candidate tables from the frozen `03_a` hub ranking for direct insertion into the thesis.

Reads the frozen `enhanced_hub_ranking.tsv` from `03_a` and retrieves gene descriptions from the individual gene JSON files indexed by `gene_info_combined.json` from `00_c`. Splits the ranked gene list by annotation tier, producing four output files: full tables for all known cancer genes (Tier 1 + 2) and top 10 novel candidates (Tier 3) with complete scores, and simplified versions of each with a reduced column set suitable for report tables.

**Inputs from:**
- `output/03_a_enhanced_hub_analysis/enhanced_hub_ranking.tsv` (frozen; 03_a not rerun)
- `output/00_c_gene_info/gene_info_combined.json`

**Key outputs (`output/03_d_candidate_tables/`):**
- `known_cancer_genes_in_hubs.tsv` / `.json` — all Tier 1 + 2 genes with scores and functions
- `top10_novel_candidates.tsv` / `.json` — top 10 Tier 3 genes with full annotation
- `simple_top10_known.tsv` / `.json` — top 10 known genes, report-ready (rank, gene, tier, function)
- `simple_top10_novel.tsv` / `.json` — top 10 novel genes, report-ready (rank, gene, tier, function)

**Packages:** `pandas`, `json`, `csv`, `pathlib`

---

## Utility Modules (`code/utils/`)

| Module | Purpose |
|---|---|
| `config.py` | Loads `configs/config.yaml`; provides path resolution and parameter access for all scripts |
| `file.py` | Path helpers: `get_relative_path`, `ensure_dir`, `get_auto_output_path` — used by every script |
| `genes.py` | Gene lookup, name normalisation, API query wrappers for MyGene.info and Ensembl |
| `cancer_gene_lists.py` | Curated cancer gene reference sets (CGC, OncoKB, etc.) and matching logic |
| `analysis.py` | Core network analysis functions: delta connectivity, permutation testing, hub scoring |
| `chart.py` | Base charting utilities: colour palettes, axis styling, saving helpers |
| `chart_advanced.py` | Advanced/composite chart builders for multi-panel figures |
| `chart_method_comparison.py` | Charts specifically comparing Pearson vs Spearman network results |
| `color_scheme.py` | Centralised colour definitions (normal=blue, tumour=red, cancer gene=orange, novel=green, etc.) |
| `classification_enhancements.py` | ML helpers: stacking classifier, sensitivity analysis, ensemble benefit scoring |
| `visualization_computational.py` | Computational visualisations: null model plots, permutation distribution charts |
| `__init__.py` | Marks `utils/` as a package; exposes key imports |

---

## Configuration (`configs/config.yaml`)

Central YAML file controlling:
- **`paths`** — `project_root`, `data_dir`, `raw_data`, `sample_sheet`
- **`analysis_raw`** — `pool_size` for multiprocessing
- **`network`** — correlation thresholds, method choices
- **`hub_analysis`** — `feature_importance_weight` (0.6), `delta_connectivity_weight` (0.4), `cancer_bonus_multiplier`
- **`classification`** — model hyperparameters, CV folds
- **`enrichment`** — gene set databases, significance cutoffs

All scripts load config via `utils.config.load_config()` and derive output paths automatically using `get_auto_output_path(__file__, PROJECT_ROOT)`.

---

## Key Package Dependencies

| Package | Used For |
|---|---|
| `pandas` | Tabular data, matrix I/O, TSV/CSV read-write |
| `numpy` | Numerical operations, compressed matrix storage (.npz) |
| `scipy` | Spearman correlation, statistical tests, Gaussian smoothing |
| `networkx` | Graph construction, topology metrics, ego-network extraction |
| `matplotlib` / `seaborn` | All visualisations |
| `scikit-learn` | Random Forest, cross-validation, ROC, confusion matrix |
| `xgboost` | XGBoost classifier for sample classification |
| `gseapy` | Enrichr API queries and GSEA preranked analysis |
| `tqdm` | Progress bars for long-running loops |
| `requests` | REST API calls (MyGene.info, Ensembl) |
| `Pillow` | Image compositing for poster charts |
| `multiprocessing` | Parallel TSV file processing in stage 00_a |

---

## Running the Pipeline

Run scripts sequentially from the `code/` directory:

```bash
cd code/

# Stage 00 — Data ingestion
python 00_a_analyse_raw_data.py
python 00_b_data_preprocess.py
python 00_c_gene_info.py          # requires internet (API calls; cached after first run)
# optional:
python 00_d_refresh_gene_classification.py

# Stage 01 — Network construction
python 01_a_build_correlation_matrices.py
python 01_b_network_analysis.py
python 01_c_network_visualization.py
# supplementary (reads frozen 01_c output):
python 01_d_regenerate_figures.py

# Stage 02 — Differential analysis & classification
python 02_a_differential_analysis.py
python 02_b_dcea_viz_enrich.py
python 02_c_sample_classification.py
# supplementary (read frozen upstream outputs):
python 02_d_rewired_scatter.py
python 02_e_validate_classification.py

# Stage 03 — Hub prioritisation & annotation
python 03_a_enhanced_hub_analysis.py
python 03_b_functional_characterization.py
python 03_c_poster_charts.py
# supplementary (reads frozen 03_a and 00_c output):
python 03_d_candidate_tables.py
```

Each script auto-creates its output directory under `output/<script_name>/`. Supplementary scripts never modify the outputs of the scripts they read from.

---

## Tests (`code/tests/`)

| File | Purpose |
|---|---|
| `test_gene_api_lookup.py` | Validates MyGene.info and Ensembl API responses for sample genes |
| `test_gene_classification.py` | Unit tests for cancer gene matching logic in `utils/cancer_gene_lists.py` |
| `test_gene_search.py` | Tests gene name normalisation and search in `utils/genes.py` |
| `test_permutation.py` | Validates permutation testing output distributions |
| `resize_imgs_for_review.py` | Utility to batch-resize output PNG charts for quick visual review |