#!/usr/bin/env python3
"""
Fixed compression data structures that properly maintain Pre/Post transformation matrices.

Key insight from Java efmtool:
- Pre and Post matrices maintain original dimensions throughout compression
- Removed rows/columns are swapped to the end, not deleted
- Size counters track the active region of matrices
"""

from fractions import Fraction
from typing import List, Dict, Set, Optional
import logging

from .rational_math import RationalMath, ZERO, ONE
from .rational_matrix import RationalMatrix


class Size:
    """Size tracker for metabolites and reactions."""
    
    def __init__(self, metas: int, reacs: int):
        self.metas = metas
        self.reacs = reacs
        # Store original dimensions
        self.orig_metas = metas
        self.orig_reacs = reacs


class CompressionStatistics:
    """Statistics tracking for compression operations."""
    
    def __init__(self):
        self.initial_metabolites = 0
        self.initial_reactions = 0
        self.zero_flux_reactions = 0
        self.contradicting_reactions = 0
        self.coupled_reactions = 0
        self.unique_flow_reactions = 0
        self.dead_end_metabolite_reactions = 0
        self.unused_metabolites = 0
        self.duplicate_genes = 0
        self.duplicate_genes_extended = 0
        self.compression_iterations = 0
    
    def inc_compression_iteration(self) -> int:
        """Increment and return compression iteration count."""
        self.compression_iterations += 1
        return self.compression_iterations
    
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


class WorkRecord:
    """
    Fixed working data during compression that properly maintains Pre/Post matrices.
    
    Critical changes from original:
    - Pre/Post matrices maintain original dimensions
    - Removed elements are swapped to end, not deleted
    - Size object tracks active region
    """
    
    def __init__(self, stoich: RationalMatrix, reversible: List[bool],
                 meta_names: List[str], reac_names: List[str]):
        """
        Initialize work record with proper Pre/Post matrix dimensions.
        
        Args:
            stoich: Initial stoichiometric matrix
            reversible: Reaction reversibilities
            meta_names: Metabolite names
            reac_names: Reaction names
        """
        # Current compressed matrix (gets modified)
        self.cmp = stoich.copy()
        
        # Get original dimensions
        orig_rows = stoich.get_row_count()
        orig_cols = stoich.get_column_count()
        
        # Pre matrix: maps from original to compressed metabolites
        # Dimensions: orig_rows × current_rows (starts as identity)
        self.pre = RationalMatrix(orig_rows, orig_rows, sparse_mode=True)
        for i in range(orig_rows):
            self.pre.set_value_at(i, i, ONE)
        
        # Post matrix: maps from original to compressed reactions  
        # Dimensions: orig_cols × current_cols (starts as identity)
        self.post = RationalMatrix(orig_cols, orig_cols, sparse_mode=True)
        for i in range(orig_cols):
            self.post.set_value_at(i, i, ONE)
        
        # Current names and properties (get swapped as compression proceeds)
        self.meta_names = list(meta_names)
        self.reac_names = list(reac_names)
        self.reversible = list(reversible)
        
        # Size tracker (critical for maintaining correct dimensions)
        self.size = Size(orig_rows, orig_cols)
        
        # Statistics
        self.stats = CompressionStatistics()
        self.stats.initial_metabolites = orig_rows
        self.stats.initial_reactions = orig_cols
    
    def swap_columns(self, matrix: RationalMatrix, col1: int, col2: int):
        """Swap two columns in a matrix."""
        for row in range(matrix.get_row_count()):
            val1 = matrix.get_value_at(row, col1)
            val2 = matrix.get_value_at(row, col2)
            matrix.set_value_at(row, col1, val2)
            matrix.set_value_at(row, col2, val1)
    
    def swap_rows(self, matrix: RationalMatrix, row1: int, row2: int):
        """Swap two rows in a matrix."""
        for col in range(matrix.get_column_count()):
            val1 = matrix.get_value_at(row1, col)
            val2 = matrix.get_value_at(row2, col)
            matrix.set_value_at(row1, col, val2)
            matrix.set_value_at(row2, col, val1)
    
    def remove_reaction(self, reac_idx: int):
        """
        Remove a reaction by swapping to end (Java efmtool style).
        
        Args:
            reac_idx: Reaction index to remove
        """
        # Zero out the reaction column in cmp
        for meta in range(self.size.metas):
            self.cmp.set_value_at(meta, reac_idx, ZERO)
        
        # Decrement active reaction count
        self.size.reacs -= 1
        
        # Swap removed reaction to end if not already there
        if reac_idx != self.size.reacs:
            # Swap columns in Post and cmp matrices
            self.swap_columns(self.post, reac_idx, self.size.reacs)
            self.swap_columns(self.cmp, reac_idx, self.size.reacs)
            
            # Swap reaction properties
            self.reversible[reac_idx], self.reversible[self.size.reacs] = \
                self.reversible[self.size.reacs], self.reversible[reac_idx]
            self.reac_names[reac_idx], self.reac_names[self.size.reacs] = \
                self.reac_names[self.size.reacs], self.reac_names[reac_idx]
    
    def remove_reactions(self, reac_indices: Set[int]):
        """
        Remove multiple reactions by swapping to end.
        
        Args:
            reac_indices: Set of reaction indices to remove
        """
        # Process in descending order to maintain indices
        for idx in sorted(reac_indices, reverse=True):
            if idx < self.size.reacs:  # Only remove if in active region
                self.remove_reaction(idx)
    
    def remove_metabolite(self, meta_idx: int, set_stoich_to_zero: bool = True):
        """
        Remove a metabolite by swapping to end (Java efmtool style).
        
        Args:
            meta_idx: Metabolite index to remove
            set_stoich_to_zero: Whether to zero out the metabolite row
        """
        # Zero out the metabolite row if requested
        if set_stoich_to_zero:
            for reac in range(self.size.reacs):
                self.cmp.set_value_at(meta_idx, reac, ZERO)
        
        # Decrement active metabolite count
        self.size.metas -= 1
        
        # Swap removed metabolite to end if not already there
        if meta_idx != self.size.metas:
            # Swap rows in Pre and cmp matrices
            self.swap_rows(self.pre, meta_idx, self.size.metas)
            self.swap_rows(self.cmp, meta_idx, self.size.metas)
            
            # Swap metabolite name
            self.meta_names[meta_idx], self.meta_names[self.size.metas] = \
                self.meta_names[self.size.metas], self.meta_names[meta_idx]
    
    def remove_unused_metabolites(self) -> bool:
        """
        Remove metabolites with no non-zero entries.
        
        Returns:
            True if any metabolites were removed
        """
        orig_count = self.size.metas
        meta = 0
        
        while meta < self.size.metas:
            # Check if metabolite has any non-zero entries
            has_nonzero = False
            for reac in range(self.size.reacs):
                if not RationalMath.is_zero(self.cmp.get_numerator_at(meta, reac)):
                    has_nonzero = True
                    break
            
            if not has_nonzero:
                # Remove this metabolite
                self.remove_metabolite(meta, set_stoich_to_zero=False)
                self.stats.inc_unused_metabolite()
                # Don't increment meta since we swapped a new one here
            else:
                meta += 1
        
        return self.size.metas < orig_count
    
    def get_compressed_stoich(self) -> RationalMatrix:
        """
        Get the compressed stoichiometric matrix (active region only).
        
        Returns:
            Matrix with dimensions size.metas × size.reacs
        """
        compressed = RationalMatrix(self.size.metas, self.size.reacs)
        for i in range(self.size.metas):
            for j in range(self.size.reacs):
                val = self.cmp.get_value_at(i, j)
                if not RationalMath.is_zero(val):
                    compressed.set_value_at(i, j, val)
        return compressed
    
    def get_pre_matrix(self) -> RationalMatrix:
        """
        Get the Pre transformation matrix (metabolite mapping).
        
        Returns:
            Matrix with dimensions orig_metas × size.metas
        """
        pre_truncated = RationalMatrix(self.size.orig_metas, self.size.metas)
        for i in range(self.size.orig_metas):
            for j in range(self.size.metas):
                val = self.pre.get_value_at(i, j)
                if not RationalMath.is_zero(val):
                    pre_truncated.set_value_at(i, j, val)
        return pre_truncated
    
    def get_post_matrix(self) -> RationalMatrix:
        """
        Get the Post transformation matrix (reaction mapping).
        
        Returns:
            Matrix with dimensions orig_reacs × size.reacs
        """
        post_truncated = RationalMatrix(self.size.orig_reacs, self.size.reacs)
        for i in range(self.size.orig_reacs):
            for j in range(self.size.reacs):
                val = self.post.get_value_at(i, j)
                if not RationalMath.is_zero(val):
                    post_truncated.set_value_at(i, j, val)
        return post_truncated
    
    def get_active_reversible(self) -> List[bool]:
        """Get reversibilities for active reactions only."""
        return self.reversible[:self.size.reacs]
    
    def get_active_meta_names(self) -> List[str]:
        """Get names for active metabolites only."""
        return self.meta_names[:self.size.metas]
    
    def get_active_reac_names(self) -> List[str]:
        """Get names for active reactions only."""
        return self.reac_names[:self.size.reacs]


class NullspaceRecord(WorkRecord):
    """
    Nullspace-based compression record with fixed Pre/Post handling.
    """
    
    def __init__(self, work_record: WorkRecord):
        """
        Initialize from existing work record.
        
        Args:
            work_record: Base work record to build from
        """
        # Copy base attributes (maintain original dimensions!)
        super().__init__(
            work_record.cmp,
            work_record.reversible,
            work_record.meta_names,
            work_record.reac_names
        )
        
        # Copy transformation matrices (keep original dimensions)
        self.pre = work_record.pre.copy()
        self.post = work_record.post.copy()
        
        # Copy size tracker (critical!)
        self.size = Size(work_record.size.metas, work_record.size.reacs)
        self.size.orig_metas = work_record.size.orig_metas
        self.size.orig_reacs = work_record.size.orig_reacs
        
        # Copy statistics
        self.stats = work_record.stats
        
        # Compute nullspace kernel for active region only
        active_stoich = self.get_compressed_stoich()
        self.kernel = self._compute_kernel(active_stoich)
    
    def _compute_kernel(self, stoich: RationalMatrix) -> RationalMatrix:
        """
        Compute nullspace kernel.
        
        Args:
            stoich: Stoichiometric matrix (active region)
            
        Returns:
            Kernel matrix where columns are nullspace basis vectors
        """
        from .rational_matrix import GaussElimination
        return GaussElimination.nullspace(stoich.transpose())
    
    def remove_reaction(self, reac_idx: int):
        """
        Remove reaction and update kernel.
        
        Args:
            reac_idx: Reaction index to remove
        """
        # Call parent method to handle Pre/Post properly
        super().remove_reaction(reac_idx)
        
        # Also swap kernel row if it exists
        if self.kernel and self.kernel.get_row_count() > reac_idx:
            if reac_idx != self.size.reacs and self.size.reacs < self.kernel.get_row_count():
                self.swap_rows(self.kernel, reac_idx, self.size.reacs)