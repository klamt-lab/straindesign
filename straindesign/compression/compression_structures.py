#!/usr/bin/env python3
"""
Data structures for network compression.

This module provides the core data structures used during compression,
replacing the internal classes from Java's StoichMatrixCompressor.
"""

from fractions import Fraction
from typing import List, Dict, Optional, Set, Tuple, Any
import numpy as np
from dataclasses import dataclass, field
import copy
import logging

from .rational_math import RationalMath, ZERO, ONE
from .rational_matrix import RationalMatrix, GaussElimination


@dataclass
class CompressionStatistics:
    """
    Track compression statistics and metrics.
    
    Equivalent to the statistics tracking in Java implementation.
    """
    compression_iterations: int = 0
    zero_flux_reactions: int = 0
    contradicting_reactions: int = 0
    coupled_reactions: int = 0
    unique_flow_reactions: int = 0
    dead_end_metabolite_reactions: int = 0
    unused_metabolites: int = 0
    duplicate_genes: int = 0
    duplicate_genes_extended: int = 0
    interchangeable_metabolites: int = 0
    
    # Track sizes at each step
    initial_reactions: int = 0
    initial_metabolites: int = 0
    final_reactions: int = 0
    final_metabolites: int = 0
    
    def inc_compression_iteration(self) -> int:
        """Increment and return compression iteration count."""
        self.compression_iterations += 1
        return self.compression_iterations - 1
    
    def inc_zero_flux_reactions(self, count: int = 1):
        """Increment zero flux reactions count."""
        self.zero_flux_reactions += count
    
    def inc_contradicting_reactions(self, count: int = 1):
        """Increment contradicting reactions count."""
        self.contradicting_reactions += count
    
    def inc_coupled_reactions(self, count: int = 1):
        """Increment coupled reactions count."""
        self.coupled_reactions += count
    
    def inc_unique_flow_reactions(self, count: int = 1):
        """Increment unique flow reactions count."""
        self.unique_flow_reactions += count
    
    def inc_dead_end_metabolite_reactions(self, count: int):
        """Increment dead end metabolite/reactions count."""
        self.dead_end_metabolite_reactions += count
    
    def inc_unused_metabolite(self, count: int = 1):
        """Increment unused metabolites count."""
        self.unused_metabolites += count
    
    def inc_duplicate_genes(self, count: int = 1):
        """Increment duplicate genes count."""
        self.duplicate_genes += count
    
    def inc_duplicate_genes_extended(self, count: int = 1):
        """Increment extended duplicate genes count."""
        self.duplicate_genes_extended += count
    
    def write_to_log(self, logger: Optional[logging.Logger] = None):
        """Write statistics to log."""
        if logger is None:
            logger = logging.getLogger(__name__)
        
        logger.info(f"Compression Statistics:")
        logger.info(f"  Initial: {self.initial_reactions} reactions, {self.initial_metabolites} metabolites")
        logger.info(f"  Final: {self.final_reactions} reactions, {self.final_metabolites} metabolites")
        logger.info(f"  Iterations: {self.compression_iterations}")
        logger.info(f"  Removed:")
        if self.zero_flux_reactions > 0:
            logger.info(f"    Zero flux reactions: {self.zero_flux_reactions}")
        if self.contradicting_reactions > 0:
            logger.info(f"    Contradicting reactions: {self.contradicting_reactions}")
        if self.coupled_reactions > 0:
            logger.info(f"    Coupled reactions: {self.coupled_reactions}")
        if self.unique_flow_reactions > 0:
            logger.info(f"    Unique flow reactions: {self.unique_flow_reactions}")
        if self.dead_end_metabolite_reactions > 0:
            logger.info(f"    Dead end metabolite/reactions: {self.dead_end_metabolite_reactions}")
        if self.unused_metabolites > 0:
            logger.info(f"    Unused metabolites: {self.unused_metabolites}")
        if self.duplicate_genes > 0:
            logger.info(f"    Duplicate genes: {self.duplicate_genes}")


@dataclass
class Size:
    """
    Mutable counter for metabolite/reaction sizes.
    
    Equivalent to the Size inner class in Java StoichMatrixCompressor.
    """
    metas: int
    reacs: int
    
    def clone(self) -> 'Size':
        """Create a copy of this Size."""
        return Size(self.metas, self.reacs)
    
    def __eq__(self, other: 'Size') -> bool:
        """Check equality."""
        if not isinstance(other, Size):
            return False
        return self.metas == other.metas and self.reacs == other.reacs
    
    def __str__(self) -> str:
        """String representation."""
        return f"[metas={self.metas}, reacs={self.reacs}]"


@dataclass
class CompressionRecord:
    """
    Record of compression containing the three transformation matrices.
    
    Contains pre, post, and cmp matrices such that:
    pre * stoich * post == cmp
    
    Where:
    - stoich: Original stoichiometric matrix (m x r)
    - cmp: Compressed stoichiometric matrix (mc x rc)
    - pre: Metabolite mapping (mc x m)
    - post: Reaction mapping (r x rc)
    """
    pre: RationalMatrix
    post: RationalMatrix
    cmp: RationalMatrix
    
    # Optional metadata
    meta_names: Optional[List[str]] = None
    reac_names: Optional[List[str]] = None
    reversible: Optional[List[bool]] = None
    
    def get_compressed_size(self) -> Tuple[int, int]:
        """Get size of compressed network (metabolites, reactions)."""
        return (self.cmp.get_row_count(), self.cmp.get_column_count())
    
    def get_original_size(self) -> Tuple[int, int]:
        """Get size of original network (metabolites, reactions)."""
        return (self.pre.get_column_count(), self.post.get_row_count())


class WorkRecord:
    """
    Mutable working data during compression.
    
    This is the main working structure that gets modified during compression.
    Equivalent to WorkRecord in Java implementation.
    """
    
    def __init__(self, stoich: RationalMatrix, reversible: List[bool],
                 meta_names: List[str], reac_names: List[str]):
        """
        Initialize work record.
        
        Args:
            stoich: Initial stoichiometric matrix
            reversible: Reaction reversibilities
            meta_names: Metabolite names
            reac_names: Reaction names
        """
        # Current compressed matrix (gets modified)
        self.cmp = stoich.copy()
        
        # Transformation matrices
        rows = stoich.get_row_count()
        cols = stoich.get_column_count()
        
        # Pre matrix (metabolite transformation) - initially identity
        self.pre = RationalMatrix(rows, rows, sparse_mode=True)
        for i in range(rows):
            self.pre.set_value_at(i, i, ONE)
        
        # Post matrix (reaction transformation) - initially identity  
        self.post = RationalMatrix(cols, cols, sparse_mode=True)
        for i in range(cols):
            self.post.set_value_at(i, i, ONE)
        
        # Current names and properties (get shortened as compression proceeds)
        self.meta_names = list(meta_names)
        self.reac_names = list(reac_names)
        self.reversible = list(reversible)
        
        # Current size
        self.size = Size(rows, cols)
        
        # Statistics
        self.stats = CompressionStatistics()
        self.stats.initial_metabolites = rows
        self.stats.initial_reactions = cols
    
    def remove_metabolite(self, meta_idx: int, set_stoich_to_zero: bool = True):
        """
        Remove a metabolite from the network.
        
        Args:
            meta_idx: Metabolite index to remove
            set_stoich_to_zero: If True, first set all stoich coefficients to zero
        """
        if set_stoich_to_zero:
            # Set all stoichiometric coefficients for this metabolite to zero
            for reac in range(self.size.reacs):
                self.cmp.set_value_at(meta_idx, reac, ZERO)
        
        # Remove from matrices
        self.cmp.remove_row(meta_idx)
        self.pre.remove_column(meta_idx)
        
        # Remove from names
        del self.meta_names[meta_idx]
        
        # Update size
        self.size.metas -= 1
    
    def remove_reaction(self, reac_idx: int):
        """
        Remove a reaction from the network.
        
        Args:
            reac_idx: Reaction index to remove
        """
        # Remove from matrices
        self.cmp.remove_column(reac_idx)
        self.post.remove_row(reac_idx)
        
        # Remove from names and properties
        del self.reac_names[reac_idx]
        del self.reversible[reac_idx]
        
        # Update size
        self.size.reacs -= 1
    
    def remove_reactions(self, reac_indices: Set[int]):
        """
        Remove multiple reactions from the network.
        
        Args:
            reac_indices: Set of reaction indices to remove
        """
        # Sort in descending order to maintain indices while removing
        for idx in sorted(reac_indices, reverse=True):
            self.remove_reaction(idx)
    
    def remove_unused_metabolites(self) -> bool:
        """
        Remove metabolites that don't participate in any reaction.
        
        Returns:
            True if any metabolites were removed
        """
        removed_any = False
        meta = 0
        
        while meta < self.size.metas:
            # Check if metabolite participates in any reaction
            has_nonzero = False
            for reac in range(self.size.reacs):
                if not RationalMath.is_zero(self.cmp.get_value_at(meta, reac)):
                    has_nonzero = True
                    break
            
            if not has_nonzero:
                # Remove unused metabolite
                self.remove_metabolite(meta, set_stoich_to_zero=False)
                self.stats.inc_unused_metabolite()
                removed_any = True
                # Don't increment meta since indices shifted
            else:
                meta += 1
        
        return removed_any
    
    def get_reaction_names(self, indices: Any) -> List[str]:
        """
        Get reaction names for given indices.
        
        Args:
            indices: List, set, or bitset of reaction indices
            
        Returns:
            List of reaction names
        """
        if hasattr(indices, 'length'):  # IntArray from Java
            return [self.reac_names[indices.get(i)] for i in range(indices.length())]
        elif hasattr(indices, '__iter__'):  # Set or list
            if hasattr(indices, 'nextSetBit'):  # BitSet-like
                result = []
                idx = 0
                while True:
                    try:
                        idx = indices.nextSetBit(idx)
                        if idx < 0:
                            break
                        result.append(self.reac_names[idx])
                        idx += 1
                    except:
                        break
                return result
            else:  # Regular iterable
                return [self.reac_names[i] for i in indices]
        else:
            return [self.reac_names[indices]]
    
    def get_reaction_details(self, reac_idx: int) -> str:
        """
        Get detailed string representation of a reaction.
        
        Args:
            reac_idx: Reaction index
            
        Returns:
            String with reaction details
        """
        parts = []
        
        # Reactants (negative coefficients)
        reactants = []
        for meta in range(self.size.metas):
            coeff = self.cmp.get_value_at(meta, reac_idx)
            if RationalMath.signum(coeff) < 0:
                if coeff == Fraction(-1):
                    reactants.append(self.meta_names[meta])
                else:
                    reactants.append(f"{abs(coeff)} {self.meta_names[meta]}")
        
        # Products (positive coefficients)
        products = []
        for meta in range(self.size.metas):
            coeff = self.cmp.get_value_at(meta, reac_idx)
            if RationalMath.signum(coeff) > 0:
                if coeff == Fraction(1):
                    products.append(self.meta_names[meta])
                else:
                    products.append(f"{coeff} {self.meta_names[meta]}")
        
        # Build reaction string
        reaction_str = " + ".join(reactants) if reactants else "∅"
        reaction_str += " <=> " if self.reversible[reac_idx] else " => "
        reaction_str += " + ".join(products) if products else "∅"
        
        return f"{self.reac_names[reac_idx]}: {reaction_str}"
    
    def log_reaction_details(self, level: int, prefix: str, reac_indices: Any):
        """
        Log reaction details.
        
        Args:
            level: Logging level
            prefix: Prefix string
            reac_indices: Reaction index or indices
        """
        logger = logging.getLogger(__name__)
        
        if isinstance(reac_indices, int):
            logger.log(level, f"{prefix}{self.get_reaction_details(reac_indices)}")
        else:
            for idx in reac_indices:
                logger.log(level, f"{prefix}{self.get_reaction_details(idx)}")
    
    def get_truncated(self) -> CompressionRecord:
        """
        Get the final compression record with properly sized matrices.
        
        Returns:
            CompressionRecord with pre, post, and cmp matrices
        """
        # Update final statistics
        self.stats.final_metabolites = self.size.metas
        self.stats.final_reactions = self.size.reacs
        
        # Create truncated matrices (remove unused rows/columns)
        pre_truncated = RationalMatrix(self.size.metas, self.pre.get_column_count())
        for i in range(self.size.metas):
            for j in range(self.pre.get_column_count()):
                pre_truncated.set_value_at(i, j, self.pre.get_value_at(i, j))
        
        post_truncated = RationalMatrix(self.post.get_row_count(), self.size.reacs)
        for i in range(self.post.get_row_count()):
            for j in range(self.size.reacs):
                post_truncated.set_value_at(i, j, self.post.get_value_at(i, j))
        
        cmp_truncated = RationalMatrix(self.size.metas, self.size.reacs)
        for i in range(self.size.metas):
            for j in range(self.size.reacs):
                cmp_truncated.set_value_at(i, j, self.cmp.get_value_at(i, j))
        
        return CompressionRecord(
            pre=pre_truncated,
            post=post_truncated,
            cmp=cmp_truncated,
            meta_names=self.meta_names[:self.size.metas],
            reac_names=self.reac_names[:self.size.reacs],
            reversible=self.reversible[:self.size.reacs]
        )


class NullspaceRecord(WorkRecord):
    """
    Extended work record with nullspace/kernel computation.
    
    Used for coupled reaction compression methods.
    """
    
    def __init__(self, work_record: WorkRecord):
        """
        Initialize from existing work record and compute nullspace.
        
        Args:
            work_record: Existing WorkRecord to extend
        """
        # Copy all attributes from work_record
        self.cmp = work_record.cmp
        self.pre = work_record.pre
        self.post = work_record.post
        self.meta_names = work_record.meta_names
        self.reac_names = work_record.reac_names
        self.reversible = work_record.reversible
        self.size = work_record.size
        self.stats = work_record.stats
        
        # Compute nullspace/kernel
        self.kernel = GaussElimination.nullspace(self.cmp)
        
        # Store kernel dimensions
        self.kernel_rows = self.kernel.get_row_count()
        self.kernel_cols = self.kernel.get_column_count()