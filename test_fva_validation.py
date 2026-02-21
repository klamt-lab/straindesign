#!/usr/bin/env python3
"""
Validate that Python and Java compression produce equivalent flux spaces.

The compressions may differ in sign conventions but should represent
the same flux space when properly accounting for direction.
"""

import numpy as np
from cobra.io import load_model
from cobra.flux_analysis import flux_variability_analysis
import straindesign.networktools as nt
import warnings
warnings.filterwarnings('ignore')

def normalize_fva_range(r_min, r_max):
    """Normalize FVA range to always be [min, max] regardless of sign."""
    values = sorted([r_min, r_max, -r_min, -r_max])
    # Return the range that covers the actual flux possibilities
    return (min(r_min, -r_max), max(r_max, -r_min))


if __name__ == "__main__":
    print("Validating flux space equivalence")
    print("="*70)

    # Python compression
    print("\nPython compression:")
    model_py = load_model("e_coli_core")
    cmp_map_py = nt.compress_model(model_py, backend='sparse')
    print(f"  Compressed to {len(model_py.reactions)} reactions")

    # Java compression
    print("\nJava compression:")
    model_java = load_model("e_coli_core")
    cmp_map_java = nt.compress_model(model_java, backend='efmtool')
    print(f"  Compressed to {len(model_java.reactions)} reactions")

    # Run FVA
    print("\nRunning FVA...")
    fva_py = flux_variability_analysis(model_py, fraction_of_optimum=0.0, processes=1)
    fva_java = flux_variability_analysis(model_java, fraction_of_optimum=0.0, processes=1)

    common = set(fva_py.index) & set(fva_java.index)
    print(f"\nComparing {len(common)} common reactions:")

    mismatches = []
    sign_differences = []
    true_mismatches = []

    for r_id in sorted(common):
        py_min = fva_py.loc[r_id, 'minimum']
        py_max = fva_py.loc[r_id, 'maximum']
        java_min = fva_java.loc[r_id, 'minimum']
        java_max = fva_java.loc[r_id, 'maximum']

        # Check for direct match
        if abs(py_min - java_min) < 1e-6 and abs(py_max - java_max) < 1e-6:
            continue  # Perfect match

        # Check for sign-flipped match (Python = -Java)
        if abs(py_min - (-java_max)) < 1e-6 and abs(py_max - (-java_min)) < 1e-6:
            sign_differences.append(r_id)
            continue  # Sign convention difference, not a real mismatch

        # True mismatch
        true_mismatches.append({
            'id': r_id,
            'py': (py_min, py_max),
            'java': (java_min, java_max)
        })

    print(f"\n  Direct matches: {len(common) - len(sign_differences) - len(true_mismatches)}")
    print(f"  Sign convention differences: {len(sign_differences)}")
    print(f"  True mismatches: {len(true_mismatches)}")

    if sign_differences:
        print(f"\nReactions with sign differences (mathematically equivalent):")
        for r_id in sign_differences[:5]:
            py_min = fva_py.loc[r_id, 'minimum']
            py_max = fva_py.loc[r_id, 'maximum']
            java_min = fva_java.loc[r_id, 'minimum']
            java_max = fva_java.loc[r_id, 'maximum']
            print(f"  {r_id}:")
            print(f"    Python: [{py_min:.4f}, {py_max:.4f}]")
            print(f"    Java:   [{java_min:.4f}, {java_max:.4f}]")
            print(f"    (Python = -Java, both represent same flux range)")

    if true_mismatches:
        print(f"\nTrue mismatches (potential bugs):")
        for m in true_mismatches:
            print(f"  {m['id']}: Python={m['py']}, Java={m['java']}")

    # Validate that BOTH can achieve the same optimal FBA value
    print("\n" + "="*70)
    print("FBA validation")
    print("="*70)

    # Find biomass reaction
    biomass_py = [r for r in model_py.reactions if 'biomass' in r.id.lower()]
    biomass_java = [r for r in model_java.reactions if 'biomass' in r.id.lower()]

    if biomass_py and biomass_java:
        model_py.objective = biomass_py[0]
        model_java.objective = biomass_java[0]

        sol_py = model_py.optimize()
        sol_java = model_java.optimize()

        print(f"\nPython FBA: {sol_py.objective_value:.6f}")
        print(f"Java FBA:   {sol_java.objective_value:.6f}")

        if abs(sol_py.objective_value - sol_java.objective_value) < 1e-6:
            print("\nSUCCESS: Both compressions achieve the same optimal value!")
        else:
            print("\nWARNING: FBA values differ!")
    else:
        print("\nCould not find biomass reaction for FBA validation")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if len(true_mismatches) == 0:
        print("\nBoth compression methods produce EQUIVALENT flux spaces.")
        print("The sign convention differences are mathematically irrelevant.")
        print("VALIDATION PASSED!")
    else:
        print(f"\nWARNING: Found {len(true_mismatches)} true mismatches!")
        print("VALIDATION FAILED!")
