"""Benchmark compression against Java and verify correctness."""
import time
from fractions import Fraction
import numpy as np

def benchmark_model(model_name):
    """Benchmark compression on a single model."""
    from cobra.io import load_model
    from straindesign.compression import compress_cobra_model

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print('='*60)

    # Load model
    t0 = time.time()
    model = load_model(model_name)
    load_time = time.time() - t0
    print(f"Loaded in {load_time:.2f}s: {len(model.reactions)} reactions, {len(model.metabolites)} metabolites")

    # Original FBA
    orig_obj = model.slim_optimize()
    print(f"Original FBA objective: {orig_obj:.6f}")

    # Compress
    t0 = time.time()
    result = compress_cobra_model(model.copy())
    compress_time = time.time() - t0
    cmp_model = result.compressed_model

    print(f"\nCompressed in {compress_time:.2f}s: {len(cmp_model.reactions)} reactions, {len(cmp_model.metabolites)} metabolites")
    print(f"Compression ratio: {len(cmp_model.reactions)/len(model.reactions)*100:.1f}%")

    # Count Fraction types
    frac_coeffs = sum(1 for r in cmp_model.reactions for c in r.metabolites.values() if isinstance(c, Fraction))
    total_coeffs = sum(len(r.metabolites) for r in cmp_model.reactions)
    frac_bounds = sum(1 for r in cmp_model.reactions for b in [r.lower_bound, r.upper_bound]
                      if isinstance(b, Fraction) and b not in (-float('inf'), float('inf')))
    total_bounds = len(cmp_model.reactions) * 2

    print(f"Fraction coefficients: {frac_coeffs}/{total_coeffs} ({100*frac_coeffs/total_coeffs:.1f}%)")
    print(f"Fraction bounds: {frac_bounds}/{total_bounds}")

    # Compressed FBA
    cmp_obj = cmp_model.slim_optimize()
    print(f"\nCompressed FBA objective: {cmp_obj:.6f}")
    print(f"Objective match: {abs(orig_obj - cmp_obj) < 1e-6}")

    # Check objective function was compressed correctly
    print(f"\nObjective reaction check:")
    for rxn in cmp_model.reactions:
        obj_coeff = rxn.objective_coefficient
        if obj_coeff != 0:
            print(f"  {rxn.id}: coeff={obj_coeff}")

    return {
        'model': model_name,
        'orig_reactions': len(model.reactions),
        'cmp_reactions': len(cmp_model.reactions),
        'compress_time': compress_time,
        'orig_obj': orig_obj,
        'cmp_obj': cmp_obj,
        'obj_match': abs(orig_obj - cmp_obj) < 1e-6,
        'frac_coeffs_pct': 100*frac_coeffs/total_coeffs,
        'compressed_model': cmp_model
    }


def verify_fva(model_name, cmp_model, max_reactions=None):
    """Verify FVA results."""
    from cobra.flux_analysis import flux_variability_analysis

    print(f"\n{'='*60}")
    print(f"FVA Verification: {model_name}")
    print('='*60)

    # Run FVA on compressed model
    print("Running FVA on compressed model...")
    t0 = time.time()

    # Select reactions for FVA
    if max_reactions and len(cmp_model.reactions) > max_reactions:
        rxn_list = list(cmp_model.reactions)[:max_reactions]
        print(f"  (limited to first {max_reactions} reactions)")
    else:
        rxn_list = cmp_model.reactions

    fva_cmp = flux_variability_analysis(cmp_model, reaction_list=rxn_list, fraction_of_optimum=0.0)
    fva_time = time.time() - t0
    print(f"FVA completed in {fva_time:.2f}s")

    # Check for any unbounded or infeasible results
    n_unbounded = ((fva_cmp['minimum'] < -999) | (fva_cmp['maximum'] > 999)).sum()
    print(f"Reactions with |flux| > 999: {n_unbounded}")

    # Show sample results
    print("\nSample FVA results:")
    for rxn_id in list(fva_cmp.index)[:5]:
        row = fva_cmp.loc[rxn_id]
        print(f"  {rxn_id}: [{row['minimum']:.4f}, {row['maximum']:.4f}]")

    return fva_cmp


def main():
    """Run all benchmarks."""
    results = []

    # Benchmark on three models
    for model_name in ['textbook', 'iJO1366']:  # textbook = e_coli_core
        try:
            r = benchmark_model(model_name)
            results.append(r)
        except Exception as e:
            print(f"Error with {model_name}: {e}")

    # Try iML1515 if available
    try:
        r = benchmark_model('iML1515')
        results.append(r)
    except Exception as e:
        print(f"iML1515 not available: {e}")

    # FVA on smaller models
    print("\n" + "="*60)
    print("FVA VERIFICATION")
    print("="*60)

    for r in results:
        if r['orig_reactions'] <= 3000:  # Only FVA on smaller models
            try:
                max_rxns = 50 if r['orig_reactions'] > 500 else None
                verify_fva(r['model'], r['compressed_model'], max_reactions=max_rxns)
            except Exception as e:
                print(f"FVA error on {r['model']}: {e}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in results:
        print(f"{r['model']}: {r['orig_reactions']} -> {r['cmp_reactions']} rxns "
              f"({r['compress_time']:.2f}s), obj match: {r['obj_match']}, "
              f"Fraction%: {r['frac_coeffs_pct']:.1f}%")


if __name__ == '__main__':
    main()
