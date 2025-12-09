#!/usr/bin/env python3
"""Investigate FVA mismatches between Python and Java compression."""

import time
import numpy as np
from cobra.io import load_model
from cobra.flux_analysis import flux_variability_analysis
import straindesign.networktools as nt
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    print("Investigating FVA mismatches")
    print("="*70)

    # Python compression
    print("\nPython compression:")
    model_py = load_model("e_coli_core")
    cmp_map_py = nt.compress_model(model_py, legacy_java_compression=False)
    print(f"  Compressed to {len(model_py.reactions)} reactions")

    # Java compression
    print("\nJava compression:")
    model_java = load_model("e_coli_core")
    cmp_map_java = nt.compress_model(model_java, legacy_java_compression=True)
    print(f"  Compressed to {len(model_java.reactions)} reactions")

    # Compare reaction IDs
    py_reacs = set(r.id for r in model_py.reactions)
    java_reacs = set(r.id for r in model_java.reactions)

    print(f"\nReaction comparison:")
    print(f"  Python-only reactions: {py_reacs - java_reacs}")
    print(f"  Java-only reactions: {java_reacs - py_reacs}")
    print(f"  Common reactions: {len(py_reacs & java_reacs)}")

    # Run FVA on both
    print("\nRunning FVA (single-threaded)...")
    fva_py = flux_variability_analysis(model_py, fraction_of_optimum=0.0, processes=1)
    fva_java = flux_variability_analysis(model_java, fraction_of_optimum=0.0, processes=1)

    # Find mismatches
    common = py_reacs & java_reacs
    print(f"\nFVA comparison for {len(common)} common reactions:")
    print("-"*70)

    mismatches = []
    for r_id in sorted(common):
        py_min = fva_py.loc[r_id, 'minimum']
        py_max = fva_py.loc[r_id, 'maximum']
        java_min = fva_java.loc[r_id, 'minimum']
        java_max = fva_java.loc[r_id, 'maximum']

        min_diff = abs(py_min - java_min)
        max_diff = abs(py_max - java_max)

        if min_diff > 1e-6 or max_diff > 1e-6:
            mismatches.append({
                'id': r_id,
                'py_min': py_min, 'py_max': py_max,
                'java_min': java_min, 'java_max': java_max,
                'min_diff': min_diff, 'max_diff': max_diff
            })

    if mismatches:
        print(f"Found {len(mismatches)} mismatches:\n")
        for m in mismatches:
            print(f"  {m['id']}:")
            print(f"    Python: [{m['py_min']:.6f}, {m['py_max']:.6f}]")
            print(f"    Java:   [{m['java_min']:.6f}, {m['java_max']:.6f}]")
            print(f"    Diff:   min={m['min_diff']:.6f}, max={m['max_diff']:.6f}")
            print()
    else:
        print("No mismatches found!")

    # Also compare reaction stoichiometries for mismatched reactions
    print("\n" + "="*70)
    print("Comparing stoichiometry of mismatched reactions")
    print("="*70)

    for m in mismatches[:3]:  # Just first 3
        r_id = m['id']
        print(f"\n{r_id}:")

        r_py = model_py.reactions.get_by_id(r_id)
        r_java = model_java.reactions.get_by_id(r_id)

        print(f"  Python reaction: {r_py.reaction}")
        print(f"  Java reaction:   {r_java.reaction}")
        print(f"  Python bounds: [{r_py.lower_bound}, {r_py.upper_bound}]")
        print(f"  Java bounds:   [{r_java.lower_bound}, {r_java.upper_bound}]")

    # Check compression maps
    print("\n" + "="*70)
    print("Compression map comparison")
    print("="*70)

    print(f"\nPython compression steps: {len(cmp_map_py)}")
    for i, step in enumerate(cmp_map_py):
        print(f"  Step {i}: {len(step.get('reac_map_exp', {}))} reaction mappings")

    print(f"\nJava compression steps: {len(cmp_map_java)}")
    for i, step in enumerate(cmp_map_java):
        print(f"  Step {i}: {len(step.get('reac_map_exp', {}))} reaction mappings")
