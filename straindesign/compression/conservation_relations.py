#!/usr/bin/env python3
"""
Conservation relations removal for network compression.

This module implements the removal of conservation relations (dependent metabolites)
which is essential before nullspace-based compression methods.
"""

from fractions import Fraction
from typing import List, Set
import logging

from .rational_math import RationalMath, ZERO, ONE
from .rational_matrix import RationalMatrix, GaussElimination
from .compression_structures import WorkRecord


def remove_conservation_relations(work_record: WorkRecord) -> bool:
    """
    Remove conservation relations from the network.
    
    This finds and removes metabolites that are linear combinations of other
    metabolites, reducing the dimension while preserving the flux space.
    
    This is the exact equivalent of remove_conservation_relations() from 
    test_compress.py, but working directly on WorkRecord.
    
    Args:
        work_record: Working data to modify
        
    Returns:
        True if any metabolites were removed
    """
    logger = logging.getLogger(__name__)
    
    # Get current stoichiometric matrix
    stoich = work_record.cmp
    
    # Find basic columns of the transposed matrix
    # (This identifies linearly independent metabolites)
    stoich_T = stoich.transpose()
    basic_metabolites = GaussElimination.basic_columns(stoich_T)
    
    # Identify dependent metabolites
    all_metabolites = set(range(work_record.size.metas))
    independent_metabolites = set(basic_metabolites)
    dependent_metabolites = all_metabolites - independent_metabolites
    
    if not dependent_metabolites:
        logger.debug("No conservation relations found")
        return False
    
    logger.debug(f"Found {len(dependent_metabolites)} dependent metabolites to remove")
    logger.debug(f"Keeping {len(independent_metabolites)} independent metabolites")
    
    # Remove dependent metabolites in reverse order to maintain indices
    removed_count = 0
    for meta_idx in sorted(dependent_metabolites, reverse=True):
        logger.debug(f"Removing dependent metabolite: {work_record.meta_names[meta_idx]}")
        work_record.remove_metabolite(meta_idx, set_stoich_to_zero=False)
        removed_count += 1
    
    logger.info(f"Removed {removed_count} conservation relations")
    return removed_count > 0


def verify_conservation_removal(work_record: WorkRecord) -> bool:
    """
    Verify that all remaining metabolites are linearly independent.
    
    Args:
        work_record: Work record to verify
        
    Returns:
        True if all metabolites are linearly independent
    """
    if work_record.size.metas == 0:
        return True
        
    stoich_T = work_record.cmp.transpose()
    rank = GaussElimination.row_echelon(stoich_T.copy(), reduced=False)
    
    expected_rank = work_record.size.metas
    actual_rank = rank
    
    logger = logging.getLogger(__name__)
    logger.debug(f"Matrix rank: {actual_rank}, Expected: {expected_rank}")
    
    return actual_rank == expected_rank


def basic_columns_rational(matrix: RationalMatrix) -> List[int]:
    """
    Find basic (linearly independent) columns using exact rational arithmetic.
    
    This is the pure Python equivalent of basic_columns_rat() from test_compress.py.
    
    Args:
        matrix: Matrix to analyze
        
    Returns:
        List of column indices that form a basis
    """
    return GaussElimination.basic_columns(matrix)