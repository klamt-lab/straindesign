#!/usr/bin/env python3
"""
Line-by-line translation of Java CoupledZero compression method.

This implements exactly the Java StoichMatrixCompressor nullspace workflow
for CoupledZero compression only.
"""

from fractions import Fraction
from typing import List, Dict, Set, Optional, Tuple
import logging

from .rational_math import RationalMath, ZERO, ONE
from .rational_matrix import RationalMatrix
from .java_translation import (
    WorkRecord, NullspaceRecord, CompressionStatistics, 
    nullspace_zero_flux_reactions, identity, cancel, is_zero
)


class CoupledZeroJavaTranslation:
    """
    Line-by-line translation of Java CoupledZero compression method.
    
    This matches the exact workflow from StoichMatrixCompressor.java.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def compress_coupled_zero(self, work_record: WorkRecord) -> bool:
        """
        Line-by-line translation of CoupledZero compression workflow.
        
        This matches lines 203-217 in StoichMatrixCompressor.java:
        - Create NullspaceRecord 
        - Apply nullspaceZeroFluxReactions
        - Clean up unused metabolites
        
        Args:
            work_record: Working data to compress
            
        Returns:
            True if any compressions were applied
        """
        # Line 211: NullspaceRecord nullspaceRecord = new NullspaceRecord(workRecord);
        nullspace_record = NullspaceRecord(work_record)
        
        # Line 212: boolean compressedAny = false;
        compressed_any = False
        
        # Line 213: if (doZer) compressedAny |= nullspaceZeroFluxReactions(nullspaceRecord);
        # For CoupledZero, doZer = True
        compressed_any = compressed_any or nullspace_zero_flux_reactions(nullspace_record)
        
        # Line 215: if (compressedAny) workRecord.removeUnusedMetabolites();
        if compressed_any:
            work_record.remove_unused_metabolites()
        
        # Copy results back to work_record (this happens implicitly in Java through references)
        # But we need to explicitly copy since we created a new NullspaceRecord
        work_record.pre = nullspace_record.pre
        work_record.cmp = nullspace_record.cmp
        work_record.post = nullspace_record.post
        work_record.reversible = nullspace_record.reversible
        work_record.size = nullspace_record.size
        work_record.stats = nullspace_record.stats
        work_record.meta_names = nullspace_record.meta_names
        work_record.reac_names = nullspace_record.reac_names
        
        # Line 216: return compressedAny;
        return compressed_any


def compress_coupled_zero_java_faithful(stoich: RationalMatrix, reversible: List[bool],
                                       meta_names: List[str], reac_names: List[str]) -> Tuple[RationalMatrix, RationalMatrix, RationalMatrix, List[bool], List[str], List[str]]:
    """
    Complete CoupledZero compression with Java-faithful implementation.
    
    Args:
        stoich: Original stoichiometric matrix
        reversible: Reaction reversibilities  
        meta_names: Metabolite names
        reac_names: Reaction names
        
    Returns:
        Tuple of (pre_matrix, compressed_stoich, post_matrix, compressed_reversible, compressed_meta_names, compressed_reac_names)
    """
    # Line 130: final WorkRecord workRecord = new WorkRecord(stoich, reversible, metaNames, reacNames);
    work_record = WorkRecord(stoich, reversible, meta_names, reac_names)
    
    # Apply CoupledZero compression
    compressor = CoupledZeroJavaTranslation()
    compressed = compressor.compress_coupled_zero(work_record)
    
    # Extract final results from work_record 
    # Get truncated matrices (only active region)
    final_record = work_record.get_truncated()
    
    return (final_record.pre, final_record.cmp, final_record.post,
            work_record.reversible[:work_record.size.reacs],
            work_record.meta_names[:work_record.size.metas],
            work_record.reac_names[:work_record.size.reacs])


def test_coupled_zero_java_translation():
    """Test the Java translation against a simple model."""
    import cobra
    import numpy as np
    
    # Load test model
    model = cobra.io.read_sbml_model("tests/model_gpr.xml")
    print(f"Testing CoupledZero on model: {len(model.metabolites)}×{len(model.reactions)}")
    
    # Convert to our format
    S = cobra.util.array.create_stoichiometric_matrix(model, array_type='dense')
    rows, cols = S.shape
    
    stoich = RationalMatrix(rows, cols)
    for i in range(rows):
        for j in range(cols):
            if S[i, j] != 0:
                stoich.set_value_at(i, j, Fraction(S[i, j]).limit_denominator())
    
    reversible = [rxn.reversibility for rxn in model.reactions]
    meta_names = [m.id for m in model.metabolites]
    reac_names = [r.id for r in model.reactions]
    
    print(f"Original: {len(meta_names)}×{len(reac_names)}")
    
    # Apply Java-faithful CoupledZero compression
    pre, cmp, post, rev_compressed, meta_compressed, reac_compressed = compress_coupled_zero_java_faithful(
        stoich, reversible, meta_names, reac_names
    )
    
    print(f"Compressed: {len(meta_compressed)}×{len(reac_compressed)}")
    print(f"Pre matrix: {pre.get_row_count()}×{pre.get_column_count()}")
    print(f"Post matrix: {post.get_row_count()}×{post.get_column_count()}")
    
    # Verify the fundamental equation: Pre × S_orig × Post = S_compressed
    print("\nTesting compression equation...")
    
    try:
        # Transpose Pre for correct equation (Pre should be compressed_rows × original_rows)
        pre_t = pre.transpose()
        temp = pre_t.multiply(stoich)
        reconstructed = temp.multiply(post)
        
        print(f"Equation dimensions: {pre_t.get_row_count()}×{pre_t.get_column_count()} × {stoich.get_row_count()}×{stoich.get_column_count()} × {post.get_row_count()}×{post.get_column_count()}")
        print(f"Reconstructed: {reconstructed.get_row_count()}×{reconstructed.get_column_count()}")
        print(f"Compressed: {cmp.get_row_count()}×{cmp.get_column_count()}")
        
        if (reconstructed.get_row_count() == cmp.get_row_count() and
            reconstructed.get_column_count() == cmp.get_column_count()):
            
            # Check if matrices are equal
            matrices_equal = True
            max_diff = 0.0
            
            for i in range(reconstructed.get_row_count()):
                for j in range(reconstructed.get_column_count()):
                    orig_val = reconstructed.get_value_at(i, j)
                    comp_val = cmp.get_value_at(i, j)
                    diff = abs(float(orig_val - comp_val))
                    max_diff = max(max_diff, diff)
                    
                    if diff > 1e-12:
                        matrices_equal = False
                        print(f"Difference at ({i},{j}): {orig_val} vs {comp_val}")
                        break
                if not matrices_equal:
                    break
            
            print(f"Matrix equation verification: {'✓ PASSED' if matrices_equal else '✗ FAILED'}")
            print(f"Maximum difference: {max_diff:.2e}")
            
            return matrices_equal
        else:
            print(f"✗ FAILED: Dimension mismatch")
            return False
            
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


if __name__ == "__main__":
    success = test_coupled_zero_java_translation()
    print(f"\n{'✅ CoupledZero Java translation successful!' if success else '❌ CoupledZero Java translation failed!'}")