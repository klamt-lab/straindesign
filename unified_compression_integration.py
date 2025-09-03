#!/usr/bin/env python3
"""
Integration wrapper for unified matrix compression in strain design workflow
"""

import logging
from unified_compression import UnifiedMetabolicMatrix
from straindesign.networktools import compress_model as original_compress_model

def compress_model_unified(model, no_par_compress_reacs=set(), use_unified=True):
    """
    Drop-in replacement for compress_model that can use unified matrix compression
    
    Args:
        model: cobra model to compress
        no_par_compress_reacs: set of reactions to protect from parallel compression
        use_unified: bool, whether to use unified matrix compression
        
    Returns:
        compression_map: list of compression steps compatible with strain design workflow
    """
    
    if not use_unified:
        # Use original compression method
        logging.info('  Using original EFMtool compression method.')
        return original_compress_model(model, no_par_compress_reacs)
    
    # Use unified matrix compression
    logging.info('  Using unified matrix compression method.')
    
    try:
        # Create unified matrix representation
        unified_matrix = UnifiedMetabolicMatrix(model)
        
        # Perform compression
        M_compressed, compression_map = unified_matrix.compress()
        
        # Log compression results
        original_shape = unified_matrix.M.shape
        compressed_shape = M_compressed.shape
        
        constraint_reduction = original_shape[0] - compressed_shape[0]
        variable_reduction = original_shape[1] - compressed_shape[1]
        
        logging.info(f'  Unified compression results:')
        logging.info(f'    Matrix: {original_shape} → {compressed_shape}')
        logging.info(f'    Constraints reduced: {constraint_reduction}')
        logging.info(f'    Variables reduced: {variable_reduction}')
        logging.info(f'    Compression steps: {len(compression_map)}')
        
        # Ensure compression_map format matches expected structure
        if not compression_map:
            # If no compressions were performed, create empty map
            compression_map = []
        
        return compression_map
        
    except Exception as e:
        logging.warning(f'  Unified compression failed: {e}')
        logging.info('  Falling back to original compression method.')
        return original_compress_model(model, no_par_compress_reacs)


def test_unified_integration():
    """Test integration with strain design workflow"""
    
    import cobra
    import straindesign as sd
    
    print("Testing Unified Compression Integration")
    print("="*50)
    
    # Load test model
    model = cobra.io.load_model("e_coli_core")
    print(f"Test model: {len(model.reactions)} reactions, {len(model.metabolites)} metabolites")
    
    # Test 1: Original compression
    print("\n1. Testing original compression...")
    model_orig = model.copy()
    try:
        compression_map_orig = compress_model_unified(model_orig, use_unified=False)
        print(f"   Original compression: {len(compression_map_orig)} steps")
        print(f"   Final model: {len(model_orig.reactions)} reactions")
    except Exception as e:
        print(f"   Original compression failed: {e}")
        compression_map_orig = []
    
    # Test 2: Unified compression  
    print("\n2. Testing unified compression...")
    model_unified = model.copy()
    try:
        compression_map_unified = compress_model_unified(model_unified, use_unified=True)
        print(f"   Unified compression: {len(compression_map_unified)} steps")
        print(f"   Final model: {len(model_unified.reactions)} reactions")
    except Exception as e:
        print(f"   Unified compression failed: {e}")
        compression_map_unified = []
    
    # Test 3: FBA comparison
    print("\n3. Comparing FBA results...")
    
    # Set same objective
    biomass_rxn = 'BIOMASS_Ecoli_core_w_GAM'
    model_orig.objective = biomass_rxn
    model_unified.objective = biomass_rxn
    
    # Solve both
    sol_orig = model_orig.optimize()
    sol_unified = model_unified.optimize()
    
    if sol_orig.status == 'optimal' and sol_unified.status == 'optimal':
        obj_diff = abs(sol_orig.objective_value - sol_unified.objective_value)
        print(f"   Original objective: {sol_orig.objective_value:.6f}")
        print(f"   Unified objective: {sol_unified.objective_value:.6f}")
        print(f"   Difference: {obj_diff:.2e}")
        
        if obj_diff < 1e-8:
            print("   ✅ Objective values match!")
        else:
            print(f"   ❌ Objective values differ by {obj_diff}")
    else:
        print(f"   ❌ Optimization failed: orig={sol_orig.status}, unified={sol_unified.status}")
    
    print("\n" + "="*50)
    print("Integration test completed")
    
    return compression_map_orig, compression_map_unified


if __name__ == "__main__":
    test_unified_integration()