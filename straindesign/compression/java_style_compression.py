#!/usr/bin/env python3
"""
Java-style compression that matches efmtool workflow exactly.

Key insight: Java applies metabolite cleanup AFTER compression, not before.
"""

from fractions import Fraction
from typing import List, Dict, Set, Tuple, Optional
import logging

from .rational_math import RationalMath, ZERO, ONE
from .rational_matrix import RationalMatrix, GaussElimination
from .compression_structures import WorkRecord, NullspaceRecord, CompressionStatistics
from .coupled_compression_complete import CoupledCompressionComplete


def apply_post_compression_metabolite_cleanup(work_record: WorkRecord) -> bool:
    """
    Apply metabolite cleanup AFTER compression, matching Java efmtool workflow.
    
    This removes metabolites that become linearly dependent after reaction compression,
    using the same logic as Java's basic_columns_rat() function.
    
    Args:
        work_record: Working data after compression
        
    Returns:
        True if any metabolites were removed
    """
    logger = logging.getLogger(__name__)
    
    if work_record.size.metas == 0 or work_record.size.reacs == 0:
        return False
    
    # Find basic columns of the transposed matrix (like Java's basic_columns_rat)
    stoich_T = work_record.cmp.transpose()
    basic_metabolites = GaussElimination.basic_columns(stoich_T)
    
    # Identify dependent metabolites
    all_metabolites = set(range(work_record.size.metas))
    independent_metabolites = set(basic_metabolites)
    dependent_metabolites = all_metabolites - independent_metabolites
    
    if not dependent_metabolites:
        logger.debug("No dependent metabolites found after compression")
        return False
    
    logger.info(f"Post-compression cleanup: removing {len(dependent_metabolites)} dependent metabolites")
    logger.debug(f"Keeping {len(independent_metabolites)} independent metabolites")
    
    # Remove dependent metabolites in reverse order to maintain indices
    removed_count = 0
    for meta_idx in sorted(dependent_metabolites, reverse=True):
        if meta_idx < len(work_record.meta_names):
            logger.debug(f"Removing dependent metabolite: {work_record.meta_names[meta_idx]}")
            work_record.remove_metabolite(meta_idx, set_stoich_to_zero=False)
            removed_count += 1
    
    logger.info(f"Removed {removed_count} dependent metabolites after compression")
    return removed_count > 0


def java_style_coupled_compression(work_record: WorkRecord,
                                 do_zero: bool = True,
                                 do_contradicting: bool = True, 
                                 do_combine: bool = True) -> bool:
    """
    Apply coupled compression with Java-style metabolite cleanup.
    
    Java workflow:
    1. Apply reaction compressions (CoupledZero, CoupledContradicting, CoupledCombine)
    2. Apply metabolite cleanup using basic_columns analysis
    3. Clean up unused metabolites
    
    Args:
        work_record: Working data to compress
        do_zero: Apply CoupledZero compression
        do_contradicting: Apply CoupledContradicting compression
        do_combine: Apply CoupledCombine compression
        
    Returns:
        True if any compressions were applied
    """
    logger = logging.getLogger(__name__)
    any_compressed = False
    
    # STEP 1: Apply nullspace-based compressions (NO conservation removal first)
    if work_record.size.metas > 0 and work_record.size.reacs > 0:
        compressor = CoupledCompressionComplete()
        nullspace_record = NullspaceRecord(work_record)
        
        # Check if we have a meaningful nullspace
        if nullspace_record.kernel.get_column_count() > 0:
            logger.debug(f"Found nullspace with {nullspace_record.kernel.get_column_count()} dimensions")
            
            # Apply zero flux compression
            if do_zero:
                compressed = compressor._nullspace_zero_flux_reactions(nullspace_record)
                any_compressed |= compressed
            
            # Apply coupled reactions compression
            if do_contradicting or do_combine:
                compressed = compressor._nullspace_coupled_reactions(
                    nullspace_record, do_contradicting, do_combine, True
                )
                any_compressed |= compressed
            
            # Update work record with changes
            work_record.cmp = nullspace_record.cmp
            work_record.pre = nullspace_record.pre
            work_record.post = nullspace_record.post
            work_record.meta_names = nullspace_record.meta_names
            work_record.reac_names = nullspace_record.reac_names
            work_record.reversible = nullspace_record.reversible
            work_record.size = nullspace_record.size
            work_record.stats = nullspace_record.stats
        else:
            logger.debug("No nullspace found - all reactions are independent")
    
    # STEP 2: Apply metabolite cleanup AFTER compression (Java style)
    if any_compressed:
        metabolite_cleanup_applied = apply_post_compression_metabolite_cleanup(work_record)
        if metabolite_cleanup_applied:
            any_compressed = True
    
    # STEP 3: Clean up any remaining unused metabolites
    if any_compressed:
        work_record.remove_unused_metabolites()
    
    return any_compressed


# Factory functions with Java-style workflow
def java_style_coupled_zero(work_record: WorkRecord) -> bool:
    """Apply only CoupledZero with Java-style metabolite cleanup."""
    return java_style_coupled_compression(
        work_record, do_zero=True, do_contradicting=False, do_combine=False
    )


def java_style_coupled_contradicting(work_record: WorkRecord) -> bool:
    """Apply only CoupledContradicting with Java-style metabolite cleanup."""
    return java_style_coupled_compression(
        work_record, do_zero=False, do_contradicting=True, do_combine=False
    )


def java_style_coupled_combine(work_record: WorkRecord) -> bool:
    """Apply only CoupledCombine with Java-style metabolite cleanup."""
    return java_style_coupled_compression(
        work_record, do_zero=False, do_contradicting=False, do_combine=True
    )


def java_style_all_coupled(work_record: WorkRecord) -> bool:
    """Apply all coupled methods with Java-style metabolite cleanup."""
    return java_style_coupled_compression(
        work_record, do_zero=True, do_contradicting=True, do_combine=True
    )