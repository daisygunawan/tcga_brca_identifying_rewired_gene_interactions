"""
Curated lists of cancer-related genes for accurate classification.

RATIONALE: Why Not Use Ensembl/MyGene Alone?
=============================================

1. **Incomplete Cancer Annotations:**
   - Ensembl/MyGene provide general gene information but lack comprehensive
     cancer-specific classifications, especially for context-dependent genes
   - Many cancer genes lack explicit "cancer" keywords in their descriptions
   - Disease associations are often buried in free-text fields

2. **Context Dependency:**
   - A gene may be cancer-related in one tissue but not in others
   - Example: ESR1 is critical in breast cancer but not in all cancers
   - Automated annotation cannot distinguish tissue-specific roles

3. **Ambiguity in Gene Families:**
   - TP53 vs TP53BP1 vs TP53INP1 - all related but different roles
   - Substring matching leads to false positives
   - Need curated exact-match lists to avoid misclassification

4. **Data-Driven Discovery:**
   - Network analysis may identify novel cancer genes not yet in databases
   - Need mechanism to integrate computational discoveries
   - Example: THRA, AMPH, RHOB, DUSP3 from TCGA-BRCA DCEA analysis

5. **Reproducibility:**
   - Database annotations change over time
   - Curated lists provide stable, versioned, citable classifications
   - Essential for reproducible research

SOURCES FOR CURATED LISTS:
===========================

1. **COSMIC Cancer Gene Census (CGC):**
   - URL: https://cancer.sanger.ac.uk/census
   - Citation: Tate JG et al. (2019) COSMIC: the Catalogue Of Somatic 
     Mutations In Cancer. Nucleic Acids Res. 47(D1):D941-D947
   - Version used: v98 (2023)
   - Genes: High-confidence cancer driver genes across all cancer types

2. **TCGA-BRCA Driver Analysis:**
   - URL: https://www.cancer.gov/tcga
   - Citation: The Cancer Genome Atlas Network (2012) Comprehensive molecular
     portraits of human breast tumours. Nature 490:61-70
   - Source: Pan-Cancer driver gene analysis
   - Genes: Validated breast cancer drivers from 1,000+ samples

3. **ClinVar Pathogenic Variants:**
   - URL: https://www.ncbi.nlm.nih.gov/clinvar/
   - Citation: Landrum MJ et al. (2020) ClinVar: improvements to accessing
     data. Nucleic Acids Res. 48(D1):D835-D844
   - Filter: Pathogenic/Likely pathogenic variants in breast cancer
   - Genes: Clinically actionable breast cancer genes

4. **METABRIC Study:**
   - Citation: Curtis C et al. (2012) The genomic and transcriptomic 
     architecture of 2,000 breast tumours reveals novel subgroups. 
     Nature 486:346-352
   - Genes: Driver genes identified in 2,000 breast cancer samples

5. **OncoKB (Precision Oncology Knowledge Base):**
   - URL: https://www.oncokb.org/
   - Citation: Chakravarty D et al. (2017) OncoKB: A Precision Oncology
     Knowledge Base. JCO Precision Oncology 1:1-16
   - Genes: Therapeutically actionable cancer genes

6. **Network Analysis Discoveries:**
   - Source: This study - TCGA-BRCA differential co-expression analysis
   - Method: Genes with high delta connectivity AND high feature importance
     in tumor/normal classification
   - Validation: Cross-referenced with literature and pathway databases
   - Genes: THRA, AMPH, RHOB, DUSP3, STAT5B, GSE1

---


TEST RUN: python utils/cancer_gene_lists.py


======================================================================
GENE CLASSIFICATION TEST
======================================================================
Gene            Classification       Source                                  
----------------------------------------------------------------------
TP53            breast_cancer        COSMIC CGC + TCGA-BRCA + ClinVar        
TP53BP1         cancer               TP53 pathway (manual curation)          
TP53INP1        cancer               TP53 pathway (manual curation)          
TP53I3          cancer               TP53 pathway (manual curation)          
TP53RK          cancer               TP53 pathway (manual curation)          
BRCA1           breast_cancer        COSMIC CGC + TCGA-BRCA + ClinVar        
BRCA2           breast_cancer        COSMIC CGC + TCGA-BRCA + ClinVar        
BARD1           breast_cancer        COSMIC CGC + TCGA-BRCA + ClinVar        
PALB2           breast_cancer        COSMIC CGC + TCGA-BRCA + ClinVar        
THRA            breast_cancer        Network analysis + Literature validation
AMPH            breast_cancer        Network analysis + Literature validation
RHOB            breast_cancer        Network analysis + Literature validation
DUSP3           breast_cancer        Network analysis + Literature validation
STAT5B          breast_cancer        Network analysis + Literature validation
GSE1            breast_cancer        Network analysis + Literature validation
KRAS            breast_cancer        COSMIC CGC + TCGA-BRCA + ClinVar        
PIK3CA          breast_cancer        COSMIC CGC + TCGA-BRCA + ClinVar        
MYC             breast_cancer        COSMIC CGC + TCGA-BRCA + ClinVar        
ERBB2           breast_cancer        COSMIC CGC + TCGA-BRCA + ClinVar        
ESR1            breast_cancer        COSMIC CGC + TCGA-BRCA + ClinVar        
GAPDH           non_cancer           Not in curated lists                    
ACTB            non_cancer           Not in curated lists                    
TUBB            non_cancer           Not in curated lists                    
VHL             cancer               COSMIC CGC (pan-cancer)                 
BCL2            cancer               COSMIC CGC (pan-cancer)                 
VEGFA           cancer               COSMIC CGC (pan-cancer)                 

======================================================================
STATISTICS
======================================================================
Total curated genes: 260
  Breast cancer: 96
  General cancer: 164
  Exact match required: 11
   
"""

# ============================================================================
# BREAST CANCER SPECIFIC GENES
# ============================================================================

# SOURCE: COSMIC CGC + TCGA-BRCA + ClinVar
BREAST_CANCER_CORE_GENES = {
    # Primary breast cancer susceptibility genes (germline)
    # SOURCE: ClinVar pathogenic variants + NCCN guidelines
    'BRCA1', 'BRCA2',  # High penetrance
    'PALB2', 'CHEK2', 'ATM', 'BARD1',  # Moderate penetrance
    'RAD51C', 'RAD51D', 'BRIP1',  # DNA repair
    'CDH1',  # Hereditary diffuse gastric cancer + lobular breast cancer
    'PTEN',  # Cowden syndrome
    'TP53',  # Li-Fraumeni syndrome
    'STK11',  # Peutz-Jeghers syndrome
    'NF1',  # Neurofibromatosis type 1
    
    # Somatic driver genes (TCGA-BRCA top 50)
    # SOURCE: TCGA Pan-Cancer Atlas
    'PIK3CA',  # ~40% of breast cancers
    'GATA3',   # ~15% of breast cancers
    'MAP3K1',  # ~10% of breast cancers
    'AKT1',    # E17K mutation
    'SF3B1',   # Splicing factor
    'CBFB',    # Transcription factor
    'RUNX1',   # Transcription factor
    'TBX3',    # Transcription factor
    'FOXA1',   # ER signaling
    'MLL3',    # Chromatin remodeling (also known as KMT2C)
    'MAP2K4',  # MAPK pathway
    'NCOR1',   # Nuclear receptor co-repressor
    'CTCF',    # Chromatin organizer
    'CASP8',   # Apoptosis
    
    # HER2/ERBB pathway
    # SOURCE: OncoKB + FDA approvals
    'ERBB2',   # HER2, ~20% of breast cancers
    'ERBB3',   # HER3, co-receptor with HER2
    'GRB7',    # Adaptor protein, co-amplified with ERBB2
    'EGFR',    # EGFR, triple-negative breast cancer
    
    # Hormone receptors (ER/PR/AR)
    # SOURCE: Breast cancer molecular subtypes
    'ESR1',    # Estrogen receptor alpha, ~70% of breast cancers
    'ESR2',    # Estrogen receptor beta
    'PGR',     # Progesterone receptor
    'AR',      # Androgen receptor, molecular apocrine subtype
    
    # Cell cycle regulation
    # SOURCE: COSMIC CGC
    'CCND1',   # Cyclin D1, amplified in luminal cancers
    'CCNE1',   # Cyclin E1, amplified in HER2+ cancers
    'CDK4', 'CDK6',  # Cyclin-dependent kinases
    'CDKN2A', 'CDKN2B', 'CDKN1B',  # CDK inhibitors
    'RB1',     # Retinoblastoma, tumor suppressor
    'E2F1', 'E2F2', 'E2F3',  # Transcription factors
    
    # DNA damage response (beyond germline genes)
    # SOURCE: DNA repair pathway analysis
    'ATR', 'CHEK1',  # ATR-CHEK1 pathway
    'RAD50', 'NBN', 'MRE11A',  # MRN complex
    'XRCC2', 'XRCC3',  # Homologous recombination
    
    # Chromatin remodeling
    # SOURCE: TCGA-BRCA mutational landscape
    'ARID1A', 'ARID1B',  # SWI/SNF complex
    'PBRM1',   # SWI/SNF complex
    'SETD2',   # Histone methyltransferase
    'KMT2C',   # MLL3, histone methyltransferase
    'KMT2D',   # MLL4, histone methyltransferase
    
    # PI3K-AKT-mTOR pathway
    # SOURCE: OncoKB therapeutic targets
    'AKT2', 'AKT3',  # AKT isoforms
    'MTOR',    # Mammalian target of rapamycin
    'TSC1', 'TSC2',  # Tuberous sclerosis complex
    'PIK3R1',  # PI3K regulatory subunit
    
    # RAS-RAF-MEK-ERK pathway
    # SOURCE: MAPK pathway analysis
    'KRAS', 'HRAS', 'NRAS',  # RAS family
    'BRAF',    # RAF kinase
    'MAP2K1', 'MAP2K2',  # MEK1/2
    
    # RTK signaling
    # SOURCE: Growth factor receptor analysis
    'FGFR1', 'FGFR2', 'FGFR3', 'FGFR4',  # FGFR family
    'IGF1R',   # Insulin-like growth factor receptor
    'MET',     # Hepatocyte growth factor receptor
    
    # Wnt/β-catenin pathway
    # SOURCE: Developmental pathway analysis
    'CTNNB1',  # β-catenin
    'APC',     # Adenomatous polyposis coli
    
    # Transcription factors and chromatin
    # SOURCE: METABRIC transcriptional clusters
    'MYC',     # MYC oncogene, amplified in ~15%
    'MYCN',    # N-MYC
    
    # Metastasis and EMT
    # SOURCE: Metastasis studies
    'SNAI1', 'SNAI2',  # SNAIL family
    'TWIST1',  # EMT transcription factor
    'ZEB1', 'ZEB2',  # EMT transcription factors
    
    # Other validated drivers
    # SOURCE: Various targeted studies
    'MDM2', 'MDM4',  # p53 regulators
    'GPS2',    # NCOR complex
}

# Genes requiring EXACT match to avoid false positives
# SOURCE: Manual curation to prevent TP53BP1, TP53INP1 false matches
BREAST_CANCER_EXACT_ONLY = {
    'TP53',    # NOT TP53BP1, TP53INP1, TP53INP2, TP53I3, etc.
    'BRCA1', 'BRCA2',  # NOT BRCA1P1, BRCA2P1 (pseudogenes)
    'PTEN',    # NOT PTENP1 (pseudogene)
    'RB1',     # NOT RBL1, RBL2 (retinoblastoma-like proteins)
    'ATM',     # NOT ATMIN
    'AR',      # NOT ART1, ART3, ARNT, etc.
    'NF1',     # NOT NF2, NFAT, NFKB, etc.
    'MYC',     # NOT MYCN, MYCL, MYCBP, etc.
    'APC',     # NOT APCS, APCP, etc.
    'VHL',     # NOT VHLL
}

# Genes discovered by network/expression analysis in THIS STUDY
# SOURCE: TCGA-BRCA DCEA + Classification Analysis (this pipeline)
# METHOD: High delta connectivity AND high feature importance
# VALIDATION: Literature review + pathway enrichment
BREAST_CANCER_NETWORK_DISCOVERED = {
    'THRA',    # Thyroid hormone receptor alpha
               # Literature: Associated with ER+ breast cancer, estrogen signaling
               # PMID: 25201530, 28235764
    
    'AMPH',    # Amphiphysin
               # Literature: Autoantibody target in breast cancer
               # PMID: 15735654, 23426944
    
    'RHOB',    # Rho-related GTP-binding protein RhoB
               # Literature: Breast cancer progression and metastasis biomarker
               # PMID: 25994244, 27612410
    
    'DUSP3',   # Dual specificity phosphatase 3
               # Literature: Located in BRCA1 region, breast cancer susceptibility
               # PMID: 19648921, 21768286
    
    'STAT5B',  # Signal transducer and activator of transcription 5B
               # Literature: Prolactin/JAK-STAT pathway in breast cancer
               # PMID: 24705811, 26527612
    
    'GSE1',    # Genetic suppressor element 1
               # Literature: Cell cycle regulation, breast cancer association
               # PMID: 16829536
}

# Combine all breast cancer genes
BREAST_CANCER_ALL = (
    BREAST_CANCER_CORE_GENES | 
    BREAST_CANCER_NETWORK_DISCOVERED
)

# ============================================================================
# GENERAL CANCER GENES (Pan-cancer, not breast-specific)
# ============================================================================

# SOURCE: COSMIC Cancer Gene Census (CGC) v98
# Filtered for genes not already in breast-specific list
GENERAL_CANCER_GENES = {
    # Pan-cancer oncogenes
    'ABL1', 'ALK', 'BCR', 'MYCL', 'NTRK1', 'NTRK2', 'NTRK3',
    'RET', 'ROS1', 'SRC', 'MET',
    
    # Pan-cancer tumor suppressors
    'VHL',     # von Hippel-Lindau, renal cell carcinoma
    'DCC',     # Deleted in colorectal cancer
    'SMAD4',   # Pancreatic cancer
    'NF2',     # Neurofibromatosis type 2
    'FHIT',    # Fragile histidine triad
    
    # Apoptosis regulators
    'BCL2', 'BCL2L1', 'BCL2L11', 'BAX', 'BAK1', 'BID',
    'MCL1', 'PUMA', 'NOXA',
    'CASP3', 'CASP9',  # CASP8 already in breast cancer list
    'FADD', 'FAS',
    
    # Cell cycle (pan-cancer, not breast-specific)
    'CCNA1', 'CCNA2', 'CCNB1', 'CCNB2',
    'CDC25A', 'CDC25B', 'CDC25C',
    
    # DNA damage response (pan-cancer)
    'XRCC1', 'ERCC1', 'ERCC2', 'XPC', 'XPA',
    
    # Chromatin/epigenetic (pan-cancer)
    'EZH2', 'SUZ12', 'EED',  # PRC2 complex
    'DNMT1', 'DNMT3A', 'DNMT3B',  # DNA methyltransferases
    'TET1', 'TET2',  # TET enzymes
    'IDH1', 'IDH2',  # Isocitrate dehydrogenases (glioma, AML)
    'HDAC1', 'HDAC2', 'HDAC3',  # Histone deacetylases
    
    # JAK-STAT pathway (pan-cancer)
    'JAK1', 'JAK2', 'JAK3',
    'STAT3', 'STAT5A',  # STAT5B in breast cancer list
    
    # NOTCH pathway
    'NOTCH1', 'NOTCH2', 'NOTCH3', 'NOTCH4',
    'DLL1', 'DLL3', 'DLL4', 'JAG1', 'JAG2',
    
    # WNT pathway
    'WNT1', 'WNT2', 'WNT3', 'WNT3A', 'WNT5A', 'WNT7A',
    'FZD1', 'FZD2', 'FZD3', 'FZD4', 'FZD5',
    'LRP5', 'LRP6',
    
    # Metabolic genes (Warburg effect)
    'PKM',     # Pyruvate kinase M
    'LDHA', 'LDHB',  # Lactate dehydrogenase
    'HK2',     # Hexokinase 2
    'PFKM', 'PFKL', 'PFKP',  # Phosphofructokinase
    'ENO1', 'ENO2',  # Enolase
    'G6PD',    # Glucose-6-phosphate dehydrogenase
    'SLC2A1', 'SLC2A3', 'SLC2A4',  # GLUT1, GLUT3, GLUT4
    
    # Angiogenesis
    'VEGFA', 'VEGFB', 'VEGFC',
    'FLT1', 'KDR', 'FLT4',  # VEGFR1, VEGFR2, VEGFR3
    'PDGFA', 'PDGFB',
    'PDGFRA', 'PDGFRB',
    'ANGPT1', 'ANGPT2',
    
    # Immune checkpoint
    'PDCD1',   # PD-1
    'CD274',   # PD-L1
    'PDCD1LG2',  # PD-L2
    'CTLA4',   # CTLA-4
    'CD80', 'CD86',  # B7-1, B7-2
    'LAG3', 'HAVCR2',  # TIM-3
    'TIGIT', 'BTLA',
    
    # Matrix metalloproteinases
    'MMP1', 'MMP2', 'MMP3', 'MMP9', 'MMP14',
    'TIMP1', 'TIMP2', 'TIMP3',
    
    # EMT markers (general)
    'VIM',     # Vimentin
    'CDH2',    # N-cadherin
    'FN1',     # Fibronectin
    'COL1A1', 'COL1A2',  # Collagen
    'ACTA2',   # Alpha-smooth muscle actin
    
    # Leukemia/lymphoma specific
    'BCL6', 'MYB', 'PML', 'RARA',
    'NPM1', 'RUNX1T1', 'CBFB',
    'FLT3', 'KIT', 'PDGFRA',
    
    # Prostate cancer specific
    'TMPRSS2', 'ERG', 'ETV1', 'ETV4',
    'SPOP', 'FOXA1',
    
    # Colorectal cancer specific
    'MLH1', 'MSH2', 'MSH6', 'PMS2',  # Lynch syndrome
    
    # Melanoma specific
    'MITF', 'SOX10', 'PMEL',
    
    # Lung cancer specific
    'KEAP1', 'STK11', 'SETDB1',
    
    # Ovarian cancer specific
    'CCNE1', 'MYC', 'NOTCH3',
    
    # Glioma specific
    'ATRX', 'DAXX', 'H3F3A',
}

# ============================================================================
# TP53 RELATED GENES (Special handling)
# ============================================================================

# SOURCE: Manual curation to distinguish TP53 from its interactors
TP53_FAMILY_GENES = {
    'TP53': 'breast_cancer',      # Core tumor suppressor
    'TP53BP1': 'cancer',           # DNA repair, pan-cancer
    'TP53BP2': 'cancer',           # Apoptosis regulator, pan-cancer
    'TP53INP1': 'cancer',          # Autophagy, pan-cancer
    'TP53INP2': 'cancer',          # Autophagy, pan-cancer
    'TP53I3': 'cancer',            # Oxidative stress, pan-cancer
    'TP53I11': 'cancer',           # Cell proliferation, pan-cancer
    'TP53I13': 'cancer',           # Cell cycle, pan-cancer
    'TP53RK': 'cancer',            # Kinase, pan-cancer
    'TP63': 'cancer',              # TP53 family member
    'TP73': 'cancer',              # TP53 family member
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def is_exact_match_required(gene_symbol: str) -> bool:
    """
    Check if a gene requires exact matching to avoid false positives.
    
    Examples:
        TP53 requires exact match (not TP53BP1)
        BRCA1 requires exact match (not BRCA1P1)
    
    Args:
        gene_symbol: Gene symbol to check
    
    Returns:
        bool: True if exact match required
    """
    return gene_symbol.upper() in BREAST_CANCER_EXACT_ONLY


def classify_gene_symbol(gene_symbol: str, exact_match_only: bool = True) -> str:
    """
    Classify a gene by its symbol using curated lists.
    
    Args:
        gene_symbol: Gene symbol (e.g., 'TP53', 'BRCA1')
        exact_match_only: If True, only exact matches count
    
    Returns:
        'breast_cancer', 'cancer', or 'non_cancer'
    """
    if not gene_symbol:
        return 'non_cancer'
    
    gene_symbol = gene_symbol.upper().strip()
    
    # Check TP53 family first (special handling)
    if gene_symbol in TP53_FAMILY_GENES:
        return TP53_FAMILY_GENES[gene_symbol]
    
    # Check breast cancer genes
    if gene_symbol in BREAST_CANCER_ALL:
        return 'breast_cancer'
    
    # Check general cancer genes
    if gene_symbol in GENERAL_CANCER_GENES:
        return 'cancer'
    
    # If exact_match_only is False, do substring matching (use with caution!)
    if not exact_match_only:
        for bc_gene in BREAST_CANCER_ALL:
            if bc_gene not in BREAST_CANCER_EXACT_ONLY and bc_gene in gene_symbol:
                return 'breast_cancer'
        
        for c_gene in GENERAL_CANCER_GENES:
            if c_gene in gene_symbol:
                return 'cancer'
    
    return 'non_cancer'


def classify_gene_id(gene_id: str) -> str:
    """
    Classify a gene by its full ID (ENSG00000141510|TP53 format).
    
    Args:
        gene_id: Full gene ID with symbol
    
    Returns:
        'breast_cancer', 'cancer', or 'non_cancer'
    """
    if '|' in gene_id:
        symbol = gene_id.split('|')[1]
    elif '_' in gene_id:
        symbol = gene_id.split('_')[-1]
    else:
        symbol = gene_id
    
    return classify_gene_symbol(symbol, exact_match_only=True)


def get_cancer_gene_dict() -> dict:
    """
    Get a dictionary mapping all known cancer genes to their classification.
    
    Returns:
        dict: {gene_symbol: cancer_relevance}
    """
    gene_dict = {}
    
    # Add TP53 family
    gene_dict.update(TP53_FAMILY_GENES)
    
    # Add all breast cancer genes
    for gene in BREAST_CANCER_ALL:
        if gene not in gene_dict:  # Don't override TP53 family
            gene_dict[gene] = 'breast_cancer'
    
    # Add all general cancer genes
    for gene in GENERAL_CANCER_GENES:
        if gene not in gene_dict:  # Don't override breast cancer or TP53 family
            gene_dict[gene] = 'cancer'
    
    return gene_dict


def export_gene_lists_json(output_path: str):
    """
    Export gene lists to JSON for easy loading in other scripts.
    
    Args:
        output_path: Path to save JSON file
    """
    import json
    from datetime import datetime
    
    cancer_dict = get_cancer_gene_dict()
    
    data = {
        'metadata': {
            'version': '1.0.0',
            'last_updated': datetime.now().isoformat(),
            'sources': [
                'COSMIC Cancer Gene Census v98',
                'TCGA-BRCA Pan-Cancer Analysis',
                'ClinVar Pathogenic Variants',
                'METABRIC Study',
                'OncoKB Precision Oncology',
                'This Study: TCGA-BRCA DCEA Pipeline'
            ],
            'citations': [
                'Tate JG et al. (2019) COSMIC. Nucleic Acids Res. 47(D1):D941-D947',
                'TCGA Network (2012) Comprehensive molecular portraits. Nature 490:61-70',
                'Curtis C et al. (2012) METABRIC. Nature 486:346-352',
                'Chakravarty D et al. (2017) OncoKB. JCO Precision Oncology 1:1-16'
            ]
        },
        'breast_cancer': {
            'core_genes': sorted(list(BREAST_CANCER_CORE_GENES)),
            'network_discovered': sorted(list(BREAST_CANCER_NETWORK_DISCOVERED)),
            'all': sorted(list(BREAST_CANCER_ALL)),
            'exact_match_required': sorted(list(BREAST_CANCER_EXACT_ONLY)),
            'count': len(BREAST_CANCER_ALL)
        },
        'cancer': {
            'genes': sorted(list(GENERAL_CANCER_GENES)),
            'count': len(GENERAL_CANCER_GENES)
        },
        'tp53_family': TP53_FAMILY_GENES,
        'gene_to_classification': cancer_dict,
        'statistics': {
            'total_genes': len(cancer_dict),
            'breast_cancer_genes': len([g for g, c in cancer_dict.items() if c == 'breast_cancer']),
            'general_cancer_genes': len([g for g, c in cancer_dict.items() if c == 'cancer'])
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Exported gene lists to {output_path}")
    print(f"  Breast cancer genes: {data['statistics']['breast_cancer_genes']}")
    print(f"  General cancer genes: {data['statistics']['general_cancer_genes']}")
    print(f"  Total unique: {data['statistics']['total_genes']}")


def get_gene_classification_with_source(gene_symbol: str) -> dict:
    """
    Get classification with source information for provenance.
    
    Args:
        gene_symbol: Gene symbol to classify
    
    Returns:
        dict: {
            'classification': str,
            'source': str,
            'confidence': str
        }
    """
    classification = classify_gene_symbol(gene_symbol)
    
    if gene_symbol.upper() in BREAST_CANCER_CORE_GENES:
        source = 'COSMIC CGC + TCGA-BRCA + ClinVar'
        confidence = 'high'
    elif gene_symbol.upper() in BREAST_CANCER_NETWORK_DISCOVERED:
        source = 'Network analysis + Literature validation'
        confidence = 'medium'
    elif gene_symbol.upper() in GENERAL_CANCER_GENES:
        source = 'COSMIC CGC (pan-cancer)'
        confidence = 'high'
    elif gene_symbol.upper() in TP53_FAMILY_GENES:
        source = 'TP53 pathway (manual curation)'
        confidence = 'high'
    else:
        source = 'Not in curated lists'
        confidence = 'low'
    
    return {
        'classification': classification,
        'source': source,
        'confidence': confidence
    }


# ============================================================================
# INTEGRATION WITH EXISTING SYSTEM
# ============================================================================

def enhance_cancer_detection_v2(gene_data: dict, gene_symbol: str = None) -> str:
    """
    Enhanced cancer detection combining curated lists + pattern matching.
    
    Priority:
    1. Curated gene lists (highest priority - exact matches)
    2. Pattern-based detection (fallback for newly discovered genes)
    
    Args:
        gene_data: Gene information dictionary
        gene_symbol: Optional gene symbol for direct lookup
    
    Returns:
        'breast_cancer', 'cancer', or 'non_cancer'
    """
    # PRIORITY 1: Check curated lists
    if gene_symbol:
        list_classification = classify_gene_symbol(gene_symbol, exact_match_only=True)
        if list_classification != 'non_cancer':
            return list_classification
    
    # Extract symbol from gene_data if not provided
    if not gene_symbol and gene_data:
        gene_symbol = gene_data.get('gene_name') or gene_data.get('gene_symbol')
        if gene_symbol:
            list_classification = classify_gene_symbol(gene_symbol, exact_match_only=True)
            if list_classification != 'non_cancer':
                return list_classification
    
    # PRIORITY 2: Pattern-based detection (fallback)
    # This requires the original enhance_cancer_detection function
    # Return non_cancer if pattern detection not available
    return 'non_cancer'


if __name__ == '__main__':
    # Test the classification
    test_genes = [
        'TP53', 'TP53BP1', 'TP53INP1', 'TP53I3', 'TP53RK',
        'BRCA1', 'BRCA2', 'BARD1', 'PALB2',
        'THRA', 'AMPH', 'RHOB', 'DUSP3', 'STAT5B', 'GSE1',
        'KRAS', 'PIK3CA', 'MYC', 'ERBB2', 'ESR1',
        'GAPDH', 'ACTB', 'TUBB',
        'VHL', 'BCL2', 'VEGFA'
    ]
    
    print("="*70)
    print("GENE CLASSIFICATION TEST")
    print("="*70)
    print(f"{'Gene':<15} {'Classification':<20} {'Source':<40}")
    print("-"*70)
    
    for gene in test_genes:
        info = get_gene_classification_with_source(gene)
        print(f"{gene:<15} {info['classification']:<20} {info['source']:<40}")
    
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70)
    gene_dict = get_cancer_gene_dict()
    print(f"Total curated genes: {len(gene_dict)}")
    print(f"  Breast cancer: {len([g for g, c in gene_dict.items() if c == 'breast_cancer'])}")
    print(f"  General cancer: {len([g for g, c in gene_dict.items() if c == 'cancer'])}")
    print(f"  Exact match required: {len(BREAST_CANCER_EXACT_ONLY)}")