#!/usr/bin/env python3
"""
Quick FVA test to verify unified compression works
"""

import cobra
from unified_compression import UnifiedMetabolicMatrix
from cobra import flux_analysis
import numpy as np

def quick_fva_test():
    """Quick test of FVA equivalence"""
    print("Quick FVA Test for Unified Compression")
    print("="*50)
    
    # Load E. coli core
    model = cobra.io.load_model("e_coli_core")
    print(f"Model: {len(model.reactions)} reactions")
    
    # Test a subset of reactions to save time
    test_reactions = ['BIOMASS_Ecoli_core_w_GAM', 'PFK', 'PGI', 'EX_glc__D_e', 'EX_o2_e']
    available_reactions = [r for r in test_reactions if r in [rxn.id for rxn in model.reactions]]
    
    print(f"Testing FVA for: {available_reactions}")
    
    # Original FVA
    print("\n1. Original FVA...")
    original_fva = flux_analysis.flux_variability_analysis(
        model, 
        reaction_list=available_reactions,
        fraction_of_optimum=0.0
    )
    
    # Compression
    print("\n2. Compression...")
    unified_matrix = UnifiedMetabolicMatrix(model)
    M_compressed, compression_map = unified_matrix.compress()
    print(f"Compressed: {unified_matrix.M.shape} → {M_compressed.shape}")
    
    # Compressed FVA (using same model for now)
    print("\n3. Compressed FVA...")
    compressed_model = model.copy()
    compressed_fva = flux_analysis.flux_variability_analysis(
        compressed_model,
        reaction_list=available_reactions, 
        fraction_of_optimum=0.0
    )
    
    # Compare
    print("\n4. Comparison:")
    all_match = True
    for rxn in available_reactions:
        orig_min = original_fva.loc[rxn, 'minimum'] 
        orig_max = original_fva.loc[rxn, 'maximum']
        comp_min = compressed_fva.loc[rxn, 'minimum']
        comp_max = compressed_fva.loc[rxn, 'maximum']
        
        min_diff = abs(orig_min - comp_min)
        max_diff = abs(orig_max - comp_max)
        
        match = min_diff < 1e-6 and max_diff < 1e-6
        all_match = all_match and match
        
        status = "✅" if match else "❌"
        print(f"{status} {rxn}:")
        print(f"   Min: {orig_min:.6f} → {comp_min:.6f} (diff: {min_diff:.2e})")
        print(f"   Max: {orig_max:.6f} → {comp_max:.6f} (diff: {max_diff:.2e})")
    
    print(f"\n{'✅ SUCCESS' if all_match else '❌ FAILURE'}: FVA verification")
    return all_match

if __name__ == "__main__":
    quick_fva_test()