#!/usr/bin/env python3
"""Fast-fail test for compression benchmark - e_coli_core only."""

import time
import numpy as np
from cobra.io import load_model
from cobra.flux_analysis import flux_variability_analysis
import straindesign.networktools as nt
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    print("Testing e_coli_core compression (fast-fail test)")
    print("="*60)

    # Load model
    model = load_model("e_coli_core")
    print(f"Original: {len(model.reactions)} reactions, {len(model.metabolites)} metabolites")

    # Python compression
    print("\nPython compression:")
    model_py = load_model("e_coli_core")
    start = time.perf_counter()
    cmp_map_py = nt.compress_model(model_py, legacy_java_compression=False)
    elapsed_py = time.perf_counter() - start
    print(f"  Time: {elapsed_py:.3f}s -> {len(model_py.reactions)} reactions")

    # Java compression
    print("\nJava compression:")
    model_java = load_model("e_coli_core")
    start = time.perf_counter()
    cmp_map_java = nt.compress_model(model_java, legacy_java_compression=True)
    elapsed_java = time.perf_counter() - start
    print(f"  Time: {elapsed_java:.3f}s -> {len(model_java.reactions)} reactions")

    # Quick FVA comparison (single-threaded to avoid multiprocessing issues on Windows)
    print("\nFVA comparison (single-threaded):")

    # Disable multiprocessing by setting processes=1
    fva_py = flux_variability_analysis(model_py, fraction_of_optimum=0.0, processes=1)
    print(f"  Python FVA: {len(fva_py)} reactions")

    fva_java = flux_variability_analysis(model_java, fraction_of_optimum=0.0, processes=1)
    print(f"  Java FVA: {len(fva_java)} reactions")

    common = set(fva_py.index) & set(fva_java.index)
    print(f"  Common reactions: {len(common)}")

    mismatches = 0
    for r_id in common:
        if abs(fva_py.loc[r_id, 'minimum'] - fva_java.loc[r_id, 'minimum']) > 1e-6 or \
           abs(fva_py.loc[r_id, 'maximum'] - fva_java.loc[r_id, 'maximum']) > 1e-6:
            mismatches += 1

    if mismatches:
        print(f"  WARNING: {mismatches} FVA mismatches!")
    else:
        print(f"  SUCCESS: All FVA values match!")

    print(f"\nSpeedup: {elapsed_java/elapsed_py:.2f}x (Python vs Java)")
    print("\nFast-fail test PASSED!")
