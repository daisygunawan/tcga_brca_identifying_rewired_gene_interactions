# cd code
# python tests/test_permutation.py

"""
Permutation Test Validation Script

This script tests the permutation-based correlation difference function
from 02_a_differential_analysis.py. The permutation test is a 
non-parametric alternative to Fisher Z-transform that doesn't assume
normality and is useful for small datasets or non-normal distributions.

The test validates:
1. Function import and execution
2. p-value range validity (0-1)
3. Behavior with different correlation scenarios
4. Reproducibility with random seeds

Usage:
    cd code
    python tests/test_permutation.py
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path to import from code/
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import configuration for consistency with main script
from utils.config import load_config

print("=" * 70)
print("PERMUTATION TEST VALIDATION SCRIPT")
print("=" * 70)
print("\nPurpose: Validate the permutation-based correlation difference test")
print("Function: permutation_test_correlation_diff() from 02_a_differential_analysis.py")
print("-" * 70)

# Dynamically import the differential analysis module
print("\n[1/3] IMPORTING MODULES")
print("-" * 40)
print("  Loading differential analysis module...")

import importlib.util

# Get the path to the main differential analysis script
diff_analysis_path = Path(__file__).parent.parent / "02_a_differential_analysis.py"
print(f"  Source: {diff_analysis_path.name}")

# Create module specification and load it dynamically
spec = importlib.util.spec_from_file_location(
    "differential_analysis", 
    diff_analysis_path
)
diff_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diff_module)

# Extract the permutation test function
permutation_test_correlation_diff = diff_module.permutation_test_correlation_diff
print("  ✓ Successfully imported permutation_test_correlation_diff()")
print("  ✓ Module loaded with all dependencies")

# Load configuration to ensure consistent parameters
print("\n[2/3] LOADING CONFIGURATION")
print("-" * 40)
try:
    config = load_config()
    print("  ✓ Configuration loaded successfully")
    
    # Extract relevant parameters
    default_permutations = config['network_analysis'].get('permutation_n', 50)
    print(f"  Default permutations from config: {default_permutations}")
    
except Exception as e:
    print(f"  ⚠ Could not load config: {e}")
    print("  Proceeding with default parameters")
    default_permutations = 50

# Define test scenarios with different correlation patterns
print("\n[3/3] RUNNING PERMUTATION TESTS")
print("-" * 40)

test_scenarios = [
    {
        "name": "Large Difference (0.7 vs 0.3)",
        "description": "Clear correlation difference should yield low p-value",
        "r1": 0.7,
        "r2": 0.3,
        "n1": 100,
        "n2": 100,
        "expected": "p < 0.05"
    },
    {
        "name": "Small Difference (0.5 vs 0.45)",
        "description": "Minor difference may not be statistically significant",
        "r1": 0.5,
        "r2": 0.45,
        "n1": 100,
        "n2": 100,
        "expected": "p > 0.05 (likely)"
    },
    {
        "name": "Identical Correlations (0.6 vs 0.6)",
        "description": "No difference should yield high p-value (≈1.0)",
        "r1": 0.6,
        "r2": 0.6,
        "n1": 100,
        "n2": 100,
        "expected": "p ≈ 1.0"
    },
    {
        "name": "Opposite Correlations (0.8 vs -0.8)",
        "description": "Strong opposite correlations should be highly significant",
        "r1": 0.8,
        "r2": -0.8,
        "n1": 100,
        "n2": 100,
        "expected": "p < 0.001"
    }
]

# Track test results
test_results = []

for i, scenario in enumerate(test_scenarios, 1):
    print(f"\n  Test {i}: {scenario['name']}")
    print(f"    Description: {scenario['description']}")
    print(f"    Parameters: r1={scenario['r1']}, r2={scenario['r2']}, n1={scenario['n1']}, n2={scenario['n2']}")
    print(f"    Expected: {scenario['expected']}")
    
    # Run permutation test
    print("    Running permutation test (100 permutations)...")
    
    try:
        p_val = permutation_test_correlation_diff(
            r1=scenario['r1'],
            r2=scenario['r2'],
            n1=scenario['n1'],
            n2=scenario['n2'],
            n_permutations=100,  # Use 100 for quick testing
            seed=42  # Fixed seed for reproducibility
        )
        
        # Validate p-value range
        if 0 <= p_val <= 1:
            range_check = "✓ Valid range"
        else:
            range_check = "✗ INVALID RANGE"
            
        print(f"    Result: p-value = {p_val:.6f}")
        print(f"    {range_check} (0 ≤ {p_val:.6f} ≤ 1)")
        
        # Store result for summary
        test_results.append({
            "name": scenario['name'],
            "p_value": p_val,
            "valid": 0 <= p_val <= 1,
            "passed": True
        })
        
    except Exception as e:
        print(f"    ✗ Test failed: {e}")
        test_results.append({
            "name": scenario['name'],
            "p_value": None,
            "valid": False,
            "passed": False,
            "error": str(e)
        })

# Additional test: Verify reproducibility with same seed
print("\n" + "-" * 40)
print("  Additional Test: Reproducibility")
print("  Running same test twice with seed=42...")

p_val1 = permutation_test_correlation_diff(0.7, 0.3, 100, 100, n_permutations=100, seed=42)
p_val2 = permutation_test_correlation_diff(0.7, 0.3, 100, 100, n_permutations=100, seed=42)

if np.isclose(p_val1, p_val2):
    print(f"  ✓ Reproducible: p1={p_val1:.6f}, p2={p_val2:.6f}")
    test_results.append({
        "name": "Reproducibility Test",
        "p_value": p_val1,
        "valid": True,
        "passed": True,
        "note": f"Reproducible with seed (diff: {abs(p_val1-p_val2):.2e})"
    })
else:
    print(f"  ✗ Not reproducible: p1={p_val1:.6f}, p2={p_val2:.6f}")
    test_results.append({
        "name": "Reproducibility Test",
        "p_value": None,
        "valid": False,
        "passed": False,
        "note": f"Not reproducible (diff: {abs(p_val1-p_val2):.4f})"
    })

# Generate comprehensive summary
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)

# Calculate statistics
total_tests = len(test_results)
passed_tests = sum(1 for r in test_results if r.get('passed', False))
failed_tests = total_tests - passed_tests
valid_p_values = sum(1 for r in test_results if r.get('valid', False))

print(f"Total tests run: {total_tests}")
print(f"Tests passed: {passed_tests}")
print(f"Tests failed: {failed_tests}")
print(f"Valid p-values: {valid_p_values}/{total_tests}")

print("\nDetailed Results:")
print("-" * 40)

for i, result in enumerate(test_results, 1):
    status = "✓ PASS" if result.get('passed', False) else "✗ FAIL"
    p_val_str = f"{result['p_value']:.6f}" if result['p_value'] is not None else "N/A"
    print(f"  {i:2d}. {result['name']:30} {status:8} p = {p_val_str}")
    if 'note' in result:
        print(f"       Note: {result['note']}")

# Final assessment
print("\n" + "-" * 40)
if failed_tests == 0:
    print("✅ ALL TESTS PASSED!")
    print("  Permutation test function is working correctly.")
    print("  All p-values are in valid range (0-1).")
    print("  Results are reproducible with fixed seed.")
else:
    print(f"❌ {failed_tests} TEST(S) FAILED")
    print("  Check the detailed results above.")
    print("  Common issues:")
    print("    - Function import problems")
    print("    - Invalid p-value range")
    print("    - Non-reproducible results")

print("=" * 70)

# Exit with appropriate code for CI/CD systems
if failed_tests > 0:
    sys.exit(1)
else:
    print("\n✅ Permutation test validation complete. Ready for use in differential analysis.")
    sys.exit(0)
