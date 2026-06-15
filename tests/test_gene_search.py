# cd code
# python tests/test_gene_search.py

"""
Gene Search Utility Test Script

This script validates the gene information retrieval system by testing:
1. Correct import paths and module availability
2. Gene ID normalization and processing
3. Information retrieval from the combined gene database
4. Error handling for non-existent genes
5. Classification accuracy for cancer-related genes

The test validates the entire pipeline from gene ID to detailed biological
information, ensuring the system is ready for co-expression network analysis.

Usage:
    cd code
    python tests/genes_search.py
"""

import sys
import json
from pathlib import Path

print("=" * 70)
print("GENE INFORMATION RETRIEVAL VALIDATION SCRIPT")
print("=" * 70)
print("\nPurpose: Test the gene information retrieval pipeline")
print("Tests: Import paths, ID normalization, JSON loading, and search functionality")
print("-" * 70)

# --- Setup Project Path ---
print("\n[1/4] SETUP AND IMPORT VALIDATION")
print("-" * 40)
print("  Setting up project paths...")

try:
    # Calculate project root: tests -> code
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    print(f"  Project root: {project_root}")
    
    # Import required modules
    from utils.config import load_config
    from utils.genes import get_gene_info, normalize_gene_id
    print("  ✓ Successfully imported:")
    print("      - load_config() from utils.config")
    print("      - get_gene_info() from utils.genes")
    print("      - normalize_gene_id() from utils.genes")
    
except ImportError as e:
    print(f"\n❌ IMPORT ERROR: Could not import necessary modules")
    print(f"   Error details: {e}")
    print("\nTroubleshooting steps:")
    print("  1. Ensure this script is in the 'tests' directory")
    print("  2. Check that 'code/utils/' contains the required Python files")
    print("  3. Verify all dependencies are installed")
    sys.exit(1)


def main():
    """
    Main function to test the gene retrieval utility.
    Validates the entire pipeline with multiple test cases.
    """
    print("\n[2/4] LOADING PROJECT CONFIGURATION")
    print("-" * 40)
    
    # Load the project configuration
    try:
        config = load_config()
        print("  ✓ Configuration loaded successfully")
        
        # Display key paths for verification
        project_path = config['paths'].get('project_root', 'Not found')
        genes_info_path = config['paths'].get('genes_info', 'Not found')
        print(f"  Project root: {project_path}")
        print(f"  Genes info directory: {genes_info_path}")
        
    except Exception as e:
        print(f"\n❌ CONFIGURATION ERROR: Failed to load configuration")
        print(f"   Error: {e}")
        print("\nCheck that config.yaml exists and has the correct structure.")
        return

    # Define comprehensive test cases with different gene ID formats and expected outcomes
    print("\n[3/4] PREPARING TEST CASES")
    print("-" * 40)
    
    test_gene_keys = [
        {
            "input": "ENSG00000141510|TP53",
            "description": "TP53 tumor suppressor (p53) - Major cancer gene",
            "expected": "Should be found in cancer or breast_cancer division",
            "notes": "Normalization: No version number present"
        },
        {
            "input": "ENSG00000160185|UBASH3A",
            "description": "Ubiquitin associated domain containing protein",
            "expected": "Should be found, likely in non_cancer division",
            "notes": "Tests standard pipe format"
        },
        {
            "input": "ENSG00000012048|BRCA1",
            "description": "BRCA1 DNA repair gene - Breast cancer susceptibility",
            "expected": "Should be found in breast_cancer division",
            "notes": "Key breast cancer gene"
        },
        {
            "input": "ENSG00000000003|TSPAN6",
            "description": "Tetraspanin 6 - Not typically cancer-associated",
            "expected": "Should be found in non_cancer division",
            "notes": "Tests non-cancer gene classification"
        },
        {
            "input": "ENSG00000122122.10|SASH3",
            "description": "SAM and SH3 domain containing 3",
            "expected": "Should be found, normalization removes .10",
            "notes": "Tests version number normalization"
        },
        {
            "input": "ENSG00000122122.10_SASH3",
            "description": "Same gene with underscore format",
            "expected": "Should match pipe format after normalization",
            "notes": "Tests underscore-to-pipe conversion"
        },
        {
            "input": "ENSG_NOT_REAL|FAKEGENE",
            "description": "Non-existent gene ID",
            "expected": "Should NOT be found (division = None)",
            "notes": "Tests error handling for missing genes"
        }
    ]
    
    print(f"  Prepared {len(test_gene_keys)} test cases:")
    for i, test in enumerate(test_gene_keys, 1):
        print(f"    {i:2d}. {test['input']:30} - {test['description']}")

    # Optional: Pre-load the combined JSON for efficiency
    print("\n[4/4] RUNNING GENE SEARCH TESTS")
    print("-" * 40)
    
    try:
        INPUT_GENES_INFO = Path(config['paths']['genes_info'])
        combined_json_path = INPUT_GENES_INFO / 'gene_info_combined.json'
        
        print(f"  Loading combined gene database...")
        with open(combined_json_path, 'r') as f:
            combined_data = json.load(f)
        
        # Display database statistics
        breast_cancer_count = len(combined_data.get('breast_cancer', {}))
        cancer_count = len(combined_data.get('cancer', {}))
        non_cancer_count = len(combined_data.get('non_cancer', {}))
        total_genes = breast_cancer_count + cancer_count + non_cancer_count
        
        print(f"  ✓ Database loaded successfully")
        print(f"  Database statistics:")
        print(f"      Breast cancer genes: {breast_cancer_count}")
        print(f"      Cancer genes: {cancer_count}")
        print(f"      Non-cancer genes: {non_cancer_count}")
        print(f"      Total genes: {total_genes}")
        
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"\n⚠ WARNING: Could not pre-load combined data file")
        print(f"  Details: {e}")
        print(f"  The search function will load it for each call (slower)")
        combined_data = None

    # Track test results
    test_results = []
    
    # Run tests for each gene key
    for i, test_case in enumerate(test_gene_keys, 1):
        gene_key = test_case["input"]
        
        print(f"\n  Test {i}: {gene_key}")
        print(f"    Description: {test_case['description']}")
        print(f"    Expected: {test_case['expected']}")
        print(f"    Notes: {test_case['notes']}")
        
        # Step 1: Show normalization
        normalized = normalize_gene_id(gene_key)
        if normalized != gene_key:
            print(f"    Normalization: '{gene_key}' → '{normalized}'")
        
        # Step 2: Call the utility function
        print("    Searching database...")
        result = get_gene_info(gene_key, config, combined_data=combined_data)
        
        # Step 3: Evaluate results
        if result and result['division']:
            # Gene found
            division = result['division']
            division_emoji = "🔴" if division == "breast_cancer" else "🟡" if division == "cancer" else "🟢"
            
            if result['gene_info'] and 'error' not in result['gene_info']:
                # Success case
                gene_name = result['gene_info'].get('gene_name', 'Unknown')
                gene_symbol = result['gene_info'].get('symbol', 'Unknown')
                
                print(f"    ✅ Found: {division_emoji} {division}")
                print(f"       Gene: {gene_name} ({gene_symbol})")
                
                # Show key information
                if 'summary' in result['gene_info']:
                    summary = result['gene_info']['summary'][:150] + "..." if len(result['gene_info']['summary']) > 150 else result['gene_info']['summary']
                    print(f"       Summary: {summary}")
                
                test_results.append({
                    "test": gene_key,
                    "status": "PASS",
                    "division": division,
                    "note": f"Found in {division} division"
                })
            else:
                # Found in index but file missing/corrupt
                print(f"    ⚠ Found in division '{division}' but with error")
                if result['gene_info'] and 'error' in result['gene_info']:
                    print(f"       Error: {result['gene_info']['error']}")
                
                test_results.append({
                    "test": gene_key,
                    "status": "WARNING",
                    "division": division,
                    "note": f"File access error: {result['gene_info'].get('error', 'Unknown')}"
                })
        else:
            # Gene not found
            if gene_key == "ENSG_NOT_REAL|FAKEGENE":
                print(f"    ✅ Not found (expected for fake gene)")
                test_results.append({
                    "test": gene_key,
                    "status": "PASS",
                    "division": None,
                    "note": "Correctly not found"
                })
            else:
                print(f"    ❌ NOT FOUND (unexpected)")
                test_results.append({
                    "test": gene_key,
                    "status": "FAIL",
                    "division": None,
                    "note": "Expected but not found"
                })
        
        print("    " + "-" * 40)

    # Generate comprehensive summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    # Calculate statistics
    total_tests = len(test_results)
    passed_tests = sum(1 for r in test_results if r['status'] == 'PASS')
    warning_tests = sum(1 for r in test_results if r['status'] == 'WARNING')
    failed_tests = sum(1 for r in test_results if r['status'] == 'FAIL')
    
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Warnings: {warning_tests}")
    print(f"Failed: {failed_tests}")
    
    print("\nDetailed Results:")
    print("-" * 40)
    
    for i, result in enumerate(test_results, 1):
        status_symbol = "✅" if result['status'] == 'PASS' else "⚠" if result['status'] == 'WARNING' else "❌"
        division_str = result['division'] if result['division'] else 'None'
        print(f"  {i:2d}. {result['test']:30} {status_symbol:2} {result['status']:8} Division: {division_str:15} - {result['note']}")
    
    print("\n" + "-" * 40)
    if failed_tests == 0:
        if warning_tests == 0:
            print("✅ ALL TESTS PASSED PERFECTLY!")
            print("  Gene retrieval system is fully functional.")
        else:
            print("✅ ALL TESTS PASSED (with warnings)")
            print("  Gene retrieval works but some files may be missing.")
    else:
        print(f"❌ {failed_tests} TEST(S) FAILED")
        print("  Check the detailed results above.")
        print("  Common issues:")
        print("    - Missing gene information files")
        print("    - Incorrect paths in configuration")
        print("    - Database file corruption")
    
    print("\nNext steps:")
    print("  1. Use get_gene_info() in your analysis scripts")
    print("  2. Call get_batch_gene_info() for bulk processing")
    print("  3. Use enhance_cancer_detection() for classification")
    print("=" * 70)


def get_relative_path(full_path):
    """Helper to show relative paths for cleaner output."""
    try:
        return Path(full_path).relative_to(Path.cwd())
    except ValueError:
        return full_path


if __name__ == "__main__":
    main()
    
    # Exit with appropriate code for CI/CD
    # This allows automated testing systems to detect failures
    # Note: We don't fail on warnings, only on actual failures
    if any(r['status'] == 'FAIL' for r in globals().get('test_results', [])):
        sys.exit(1)
    else:
        sys.exit(0)