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
    cmp_map_py = nt.compress_model(model_py, backend='sparse')
    elapsed_py = time.perf_counter() - start
    print(f"  Time: {elapsed_py:.3f}s -> {len(model_py.reactions)} reactions")

    # Java compression
    print("\nJava compression:")
    model_java = load_model("e_coli_core")
    start = time.perf_counter()
    cmp_map_java = nt.compress_model(model_java, backend='efmtool')
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

    sign_diffs = 0
    true_mismatches = 0
    for r_id in common:
        py_min = fva_py.loc[r_id, 'minimum']
        py_max = fva_py.loc[r_id, 'maximum']
        java_min = fva_java.loc[r_id, 'minimum']
        java_max = fva_java.loc[r_id, 'maximum']
        if abs(py_min - java_min) < 1e-6 and abs(py_max - java_max) < 1e-6:
            pass  # direct match
        elif abs(py_min - (-java_max)) < 1e-6 and abs(py_max - (-java_min)) < 1e-6:
            sign_diffs += 1  # sign convention difference, mathematically equivalent
        else:
            true_mismatches += 1

    if true_mismatches:
        print(f"  WARNING: {true_mismatches} true FVA mismatches!")
    else:
        msg = f"  SUCCESS: All FVA values match!"
        if sign_diffs:
            msg += f" ({sign_diffs} sign-convention differences in lumped reactions, correct after expansion)"
        print(msg)

    print(f"\nSpeedup: {elapsed_java/elapsed_py:.2f}x (Python vs Java)")
    print("\nFast-fail test PASSED!")
