#!/usr/bin/env python3
"""
Test that FVA results from compressed models can be correctly expanded
back to match FVA results from the original uncompressed model.
"""

import numpy as np
from cobra.io import load_model
from cobra.flux_analysis import flux_variability_analysis
import straindesign.networktools as nt
from fractions import Fraction
import warnings
warnings.filterwarnings('ignore')


def expand_fva_using_compression_map(compressed_fva, cmp_map, original_reaction_ids):
    """
    Expand FVA results from compressed model back to original reaction space.

    Args:
        compressed_fva: DataFrame from flux_variability_analysis on compressed model
        cmp_map: list of compression step dicts, each with 'reac_map_exp'
        original_reaction_ids: list of reaction IDs from the original model

    Returns:
        dict: {original_reaction_id: (min, max)} for expanded FVA results
    """
    # Build mapping from original reactions to compressed reactions with coefficients
    # reac_map_exp: {new_reaction: {old_reaction: coefficient, ...}, ...}
    # We need the inverse: {old_reaction: (new_reaction, coefficient)}

    orig_to_compressed = {}  # {orig_id: (compressed_id, coefficient)}

    for step in cmp_map:
        reac_map_exp = step.get('reac_map_exp', {})
        for new_reac, old_reacs in reac_map_exp.items():
            for old_reac, coeff in old_reacs.items():
                orig_to_compressed[old_reac] = (new_reac, float(coeff))

    expanded = {}

    for orig_id in original_reaction_ids:
        if orig_id in orig_to_compressed:
            comp_id, coeff = orig_to_compressed[orig_id]

            if comp_id in compressed_fva.index:
                comp_min = compressed_fva.loc[comp_id, 'minimum']
                comp_max = compressed_fva.loc[comp_id, 'maximum']

                # Expand: original_flux = coefficient * compressed_flux
                if coeff >= 0:
                    orig_min = coeff * comp_min
                    orig_max = coeff * comp_max
                else:
                    # Negative coefficient flips min/max
                    orig_min = coeff * comp_max
                    orig_max = coeff * comp_min

                expanded[orig_id] = (orig_min, orig_max)
            else:
                # Compressed reaction not found (shouldn't happen)
                expanded[orig_id] = (np.nan, np.nan)
        else:
            # Reaction not in compression map (might have been removed)
            expanded[orig_id] = (np.nan, np.nan)

    return expanded


if __name__ == "__main__":
    print("Testing FVA expansion using compression map")
    print("="*70)

    # Load original model
    model_orig = load_model("e_coli_core")
    original_reaction_ids = [r.id for r in model_orig.reactions]
    print(f"\nOriginal model: {len(model_orig.reactions)} reactions")

    # Run FVA on original model
    print("\nRunning FVA on original model...")
    fva_orig = flux_variability_analysis(model_orig, fraction_of_optimum=0.0, processes=1)
    print(f"  Got FVA for {len(fva_orig)} reactions")

    # Test both Python and Java compression
    for method, use_legacy in [("Python", False), ("Java", True)]:
        print(f"\n{'='*70}")
        print(f"Testing {method} compression expansion")
        print(f"{'='*70}")

        # Load and compress model
        model_comp = load_model("e_coli_core")
        cmp_map = nt.compress_model(model_comp, legacy_java_compression=use_legacy)
        print(f"\n{method} compressed: {len(model_comp.reactions)} reactions")

        # Run FVA on compressed model
        print(f"Running FVA on {method}-compressed model...")
        fva_comp = flux_variability_analysis(model_comp, fraction_of_optimum=0.0, processes=1)
        print(f"  Got FVA for {len(fva_comp)} reactions")

        # Expand FVA results using compression map
        print(f"\nExpanding FVA results using compression map...")
        expanded_fva = expand_fva_using_compression_map(fva_comp, cmp_map, original_reaction_ids)
        print(f"  Expanded to {len(expanded_fva)} reactions")

        # Compare expanded results with original FVA
        print(f"\nComparing expanded vs original FVA:")

        tolerance = 1e-5
        matches = 0
        mismatches = []
        sign_flips = 0
        not_found = 0

        for orig_id in original_reaction_ids:
            orig_min = fva_orig.loc[orig_id, 'minimum']
            orig_max = fva_orig.loc[orig_id, 'maximum']

            if orig_id in expanded_fva:
                exp_min, exp_max = expanded_fva[orig_id]

                if np.isnan(exp_min) or np.isnan(exp_max):
                    not_found += 1
                    continue

                # Check direct match
                if abs(exp_min - orig_min) < tolerance and abs(exp_max - orig_max) < tolerance:
                    matches += 1
                # Check sign-flipped match (exp = -orig)
                elif abs(exp_min - (-orig_max)) < tolerance and abs(exp_max - (-orig_min)) < tolerance:
                    sign_flips += 1
                else:
                    mismatches.append({
                        'id': orig_id,
                        'original': (orig_min, orig_max),
                        'expanded': (exp_min, exp_max)
                    })
            else:
                not_found += 1

        print(f"\n  Results:")
        print(f"    Direct matches:     {matches}")
        print(f"    Sign-flipped:       {sign_flips} (mathematically equivalent)")
        print(f"    True mismatches:    {len(mismatches)}")
        print(f"    Not found/removed:  {not_found}")

        if mismatches:
            print(f"\n  True mismatches (first 5):")
            for m in mismatches[:5]:
                print(f"    {m['id']}:")
                print(f"      Original: [{m['original'][0]:.6f}, {m['original'][1]:.6f}]")
                print(f"      Expanded: [{m['expanded'][0]:.6f}, {m['expanded'][1]:.6f}]")

        # Validation result
        if len(mismatches) == 0:
            print(f"\n  SUCCESS: {method} compression expansion is CORRECT!")
        else:
            print(f"\n  WARNING: {method} compression expansion has {len(mismatches)} mismatches!")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)
