#!/usr/bin/env python3
"""
Coupled compression methods based on nullspace analysis.

This module implements the three coupled compression methods from efmtool:
1. CoupledZero: Remove reactions with zero flux in all steady states
2. CoupledContradicting: Remove inconsistently coupled reactions
3. CoupledCombine: Merge consistently coupled reactions

These methods analyze the nullspace (kernel) of the stoichiometric matrix
to identify reactions that are coupled through mass balance constraints.
"""

from fractions import Fraction
from typing import List, Dict, Set, Tuple, Optional
import logging
from collections import defaultdict

from .rational_math import RationalMath, ZERO, ONE
from .rational_matrix import RationalMatrix, GaussElimination
from .compression_structures import WorkRecord, NullspaceRecord, CompressionStatistics


class CoupledCompression:
    """
    Coupled compression methods based on nullspace analysis.
    
    These methods port the nullspace-based compression from Java's
    StoichMatrixCompressor.nullspace() and related methods.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize coupled compression.
        
        Args:
            logger: Optional logger for debugging output
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def compress_nullspace(self, work_record: WorkRecord, 
                          do_zero: bool = True,
                          do_contradicting: bool = True, 
                          do_combine: bool = True,
                          include_compression: bool = True) -> bool:
        """
        Main nullspace compression method.
        
        This is the main entry point that orchestrates all three coupled
        compression methods based on nullspace analysis.
        
        Args:
            work_record: Working data to modify
            do_zero: Apply CoupledZero compression
            do_contradicting: Apply CoupledContradicting compression  
            do_combine: Apply CoupledCombine compression
            include_compression: If True, actually merge reactions; if False, only remove
            
        Returns:
            True if any compressions were applied
        """
        # Create nullspace record for kernel computation
        nullspace_record = NullspaceRecord(work_record)
        
        any_compressed = False
        
        # Apply CoupledZero: Remove zero flux reactions
        if do_zero:
            compressed = self._nullspace_zero_flux_reactions(nullspace_record)
            any_compressed |= compressed
        
        # Apply CoupledContradicting and CoupledCombine: Handle coupled reactions
        if do_contradicting or do_combine:
            compressed = self._nullspace_coupled_reactions(
                nullspace_record, 
                do_contradicting, 
                do_combine, 
                include_compression
            )
            any_compressed |= compressed
        
        # Update the original work record with changes
        work_record.cmp = nullspace_record.cmp
        work_record.pre = nullspace_record.pre
        work_record.post = nullspace_record.post
        work_record.meta_names = nullspace_record.meta_names
        work_record.reac_names = nullspace_record.reac_names
        work_record.reversible = nullspace_record.reversible
        work_record.size = nullspace_record.size
        work_record.stats = nullspace_record.stats
        
        # Clean up unused metabolites if anything was compressed
        if any_compressed:
            work_record.remove_unused_metabolites()
        
        return any_compressed
    
    def _nullspace_zero_flux_reactions(self, nullspace_record: NullspaceRecord) -> bool:
        """
        CoupledZero compression: Remove reactions with zero flux.
        
        A reaction has zero flux if its coefficient is zero in all vectors
        of the nullspace (kernel). This means the reaction cannot carry
        any flux in any steady state.
        
        Args:
            nullspace_record: Nullspace record with kernel computed
            
        Returns:
            True if any reactions were removed
        """
        kernel = nullspace_record.kernel
        any_zero_flux = False
        
        # Check each reaction (row in kernel matrix)
        reac = 0
        while reac < nullspace_record.size.reacs:
            # Check if all kernel coefficients are zero for this reaction
            all_zero = True
            for col in range(kernel.get_column_count()):
                if not RationalMath.is_zero(kernel.get_value_at(reac, col)):
                    all_zero = False
                    break
            
            if all_zero:
                # This reaction has zero flux in all steady states
                self.logger.debug(f"Found and removed zero flux reaction: {nullspace_record.reac_names[reac]}")
                
                any_zero_flux = True
                nullspace_record.remove_reaction(reac)
                nullspace_record.stats.inc_zero_flux_reactions()
                
                # Don't increment reac since indices shifted down
            else:
                reac += 1
        
        return any_zero_flux
    
    def _nullspace_coupled_reactions(self, nullspace_record: NullspaceRecord,
                                   do_contradicting: bool,
                                   do_combine: bool, 
                                   include_compression: bool) -> bool:
        """
        Handle coupled reactions (CoupledContradicting and CoupledCombine).
        
        Two reactions are coupled if their fluxes have a constant ratio
        across all vectors in the nullspace. This means they must always
        operate together in fixed proportions.
        
        Args:
            nullspace_record: Nullspace record with kernel computed
            do_contradicting: Remove contradictory coupled reactions
            do_combine: Merge consistent coupled reactions
            include_compression: Actually perform merging (not just removal)
            
        Returns:
            True if any reactions were processed
        """
        kernel = nullspace_record.kernel
        stoich = nullspace_record.cmp
        post = nullspace_record.post
        reversible = nullspace_record.reversible
        
        reacs = nullspace_record.size.reacs
        kernel_cols = kernel.get_column_count()
        
        any_changes = False
        
        # Find coupled reaction groups
        groups, ratios = self._find_coupled_groups(kernel)
        
        # Process each group
        reactions_to_remove = set()
        
        for group in groups:
            if len(group) < 2:
                continue  # Skip single reactions
            
            # Check consistency with reversibilities
            consistent, direction = self._check_coupling_consistency(
                group, ratios, reversible
            )
            
            if not consistent:
                if do_contradicting:
                    # Remove all reactions in this inconsistent group
                    self.logger.debug(
                        f"Found and removed inconsistently coupled reactions: "
                        f"{[nullspace_record.reac_names[r] for r in group]}"
                    )
                    
                    for reac in group:
                        reactions_to_remove.add(reac)
                        nullspace_record.stats.inc_contradicting_reactions()
                    
                    any_changes = True
                else:
                    self.logger.debug(
                        "Ignoring inconsistently coupled reactions due to compression settings"
                    )
            else:
                if do_combine and include_compression:
                    # Combine consistently coupled reactions
                    any_changes |= self._combine_coupled_reactions(
                        group, ratios, nullspace_record
                    )
        
        # Remove marked reactions
        if reactions_to_remove:
            nullspace_record.remove_reactions(reactions_to_remove)
        
        return any_changes
    
    def _find_coupled_groups(self, kernel: RationalMatrix) -> Tuple[List[List[int]], Dict[int, Fraction]]:
        """
        Find groups of coupled reactions by analyzing nullspace ratios.
        
        Two reactions are coupled if they have a constant non-zero ratio
        across all nullspace vectors.
        
        Args:
            kernel: Nullspace matrix (reactions x nullspace_dimension)
            
        Returns:
            Tuple of (groups, ratios) where:
            - groups: List of reaction index lists (each list is a coupled group)
            - ratios: Dict mapping reaction_index -> ratio relative to group master
        """
        reacs = kernel.get_row_count()
        kernel_cols = kernel.get_column_count()
        
        groups = []
        ratios = {}  # reaction_index -> ratio (reacA / reacB where B is master)
        processed = set()
        
        for reac_a in range(reacs):
            if reac_a in processed:
                continue
            
            group = [reac_a]
            
            # Compare with all remaining reactions
            for reac_b in range(reac_a + 1, reacs):
                if reac_b in processed:
                    continue
                
                # Calculate ratio reac_a / reac_b across all nullspace vectors
                ratio = self._calculate_coupling_ratio(kernel, reac_a, reac_b)
                
                if ratio is not None and not RationalMath.is_zero(ratio):
                    # Found coupled reactions
                    ratios[reac_b] = ratio
                    group.append(reac_b)
                    processed.add(reac_b)
            
            if len(group) > 1:
                groups.append(group)
                processed.add(reac_a)
        
        return groups, ratios
    
    def _calculate_coupling_ratio(self, kernel: RationalMatrix, 
                                reac_a: int, reac_b: int) -> Optional[Fraction]:
        """
        Calculate the coupling ratio between two reactions.
        
        Returns the ratio reac_a/reac_b if they are coupled, None if not coupled.
        Two reactions are coupled if they have a constant non-zero ratio across
        all nullspace vectors where both are non-zero.
        
        Args:
            kernel: Nullspace matrix
            reac_a: First reaction index
            reac_b: Second reaction index
            
        Returns:
            Coupling ratio or None if not coupled
        """
        kernel_cols = kernel.get_column_count()
        ratio = None
        
        for col in range(kernel_cols):
            val_a = kernel.get_value_at(reac_a, col)
            val_b = kernel.get_value_at(reac_b, col)
            
            is_zero_a = RationalMath.is_zero(val_a)
            is_zero_b = RationalMath.is_zero(val_b)
            
            if is_zero_a != is_zero_b:
                # One is zero, other is not -> not coupled
                return ZERO
            elif not is_zero_a:  # Both are non-zero
                # Calculate ratio for this nullspace vector
                current_ratio = val_a / val_b
                
                if ratio is None:
                    # First non-zero ratio found
                    ratio = current_ratio
                elif ratio != current_ratio:
                    # Ratio is not constant -> not coupled
                    return ZERO
        
        # If we get here, ratio is constant (or all coefficients were zero)
        return ratio
    
    def _check_coupling_consistency(self, group: List[int], ratios: Dict[int, Fraction], 
                                  reversible: List[bool]) -> Tuple[bool, bool]:
        """
        Check if coupled reactions are consistent with their reversibilities.
        
        Coupled reactions are consistent if there exists a direction where
        all reactions can operate according to their coupling and reversibility.
        
        Args:
            group: List of coupled reaction indices
            ratios: Coupling ratios relative to first reaction
            reversible: Reaction reversibilities
            
        Returns:
            Tuple of (is_consistent, forward_direction)
        """
        master = group[0]
        
        # Try both forward and backward directions
        for forward in [True, False]:
            all_ok = forward or reversible[master]
            
            if all_ok:
                # Check all other reactions in group
                for i in range(1, len(group)):
                    reac = group[i]
                    ratio = ratios[reac]
                    
                    # Determine required direction for this reaction
                    same_direction = (forward == (RationalMath.signum(ratio) > 0))
                    
                    # Check if reaction can operate in required direction
                    if not (same_direction or reversible[reac]):
                        all_ok = False
                        break
            
            if all_ok:
                return True, forward
        
        return False, True
    
    def _combine_coupled_reactions(self, group: List[int], ratios: Dict[int, Fraction],
                                 nullspace_record: NullspaceRecord) -> bool:
        """
        Combine consistently coupled reactions into one.
        
        The first reaction becomes the master, others are merged into it
        according to their coupling ratios.
        
        Args:
            group: List of coupled reaction indices
            ratios: Coupling ratios
            nullspace_record: Working data
            
        Returns:
            True if reactions were combined
        """
        if len(group) < 2:
            return False
        
        master_idx = group[0]
        reactions_to_remove = set()
        
        self.logger.debug(
            f"Found and combined coupled reactions: "
            f"{[nullspace_record.reac_names[r] for r in group]}"
        )
        
        # Combine each slave reaction into the master
        for i in range(1, len(group)):
            slave_idx = group[i]
            ratio = ratios[slave_idx]
            
            # Get the stoichiometric coefficient for the metabolite being eliminated
            # This requires more complex logic based on the specific elimination
            master_stoich = None
            slave_stoich = None
            
            # Find a metabolite where both reactions participate
            for meta in range(nullspace_record.size.metas):
                master_coeff = nullspace_record.cmp.get_value_at(meta, master_idx)
                slave_coeff = nullspace_record.cmp.get_value_at(meta, slave_idx)
                
                if not (RationalMath.is_zero(master_coeff) and RationalMath.is_zero(slave_coeff)):
                    master_stoich = master_coeff
                    slave_stoich = slave_coeff
                    break
            
            if master_stoich is not None:
                # Calculate multipliers for column combination
                # We want: master_col = master_col * kp_mul + slave_col * rm_mul
                # such that the elimination metabolite coefficient becomes zero
                
                kp_mul = RationalMath.abs(slave_stoich)
                if RationalMath.signum(master_stoich) < 0:
                    rm_mul = master_stoich
                else:
                    rm_mul = -master_stoich
                
                # Apply column combination: master += multiplier * slave
                self._add_column_multiple_to(
                    nullspace_record.cmp, slave_idx, rm_mul, master_idx, kp_mul
                )
                self._add_column_multiple_to(
                    nullspace_record.post, slave_idx, rm_mul, master_idx, kp_mul
                )
                
                # Update reversibility (AND operation)
                nullspace_record.reversible[master_idx] &= nullspace_record.reversible[slave_idx]
            
            reactions_to_remove.add(slave_idx)
            nullspace_record.stats.inc_coupled_reactions()
        
        # Remove slave reactions
        nullspace_record.remove_reactions(reactions_to_remove)
        
        return True
    
    def _add_column_multiple_to(self, matrix: RationalMatrix, 
                              source_col: int, source_mult: Fraction,
                              target_col: int, target_mult: Fraction):
        """
        Add multiple of source column to target column.
        
        Performs: target_col = target_mult * target_col + source_mult * source_col
        
        Args:
            matrix: Matrix to modify
            source_col: Source column index  
            source_mult: Multiplier for source column
            target_col: Target column index
            target_mult: Multiplier for target column
        """
        matrix.add_column_mult_to(source_col, source_mult, target_col, target_mult)


# Factory functions for each compression type
def compress_coupled_zero(work_record: WorkRecord) -> bool:
    """
    Apply CoupledZero compression: remove zero flux reactions.
    
    Args:
        work_record: Working data to compress
        
    Returns:
        True if any reactions were removed
    """
    compressor = CoupledCompression()
    return compressor.compress_nullspace(
        work_record,
        do_zero=True,
        do_contradicting=False, 
        do_combine=False
    )


def compress_coupled_contradicting(work_record: WorkRecord) -> bool:
    """
    Apply CoupledContradicting compression: remove inconsistent coupled reactions.
    
    Args:
        work_record: Working data to compress
        
    Returns:
        True if any reactions were removed
    """
    compressor = CoupledCompression()
    return compressor.compress_nullspace(
        work_record,
        do_zero=False,
        do_contradicting=True,
        do_combine=False
    )


def compress_coupled_combine(work_record: WorkRecord) -> bool:
    """
    Apply CoupledCombine compression: merge consistently coupled reactions.
    
    Args:
        work_record: Working data to compress
        
    Returns:
        True if any reactions were combined
    """
    compressor = CoupledCompression()
    return compressor.compress_nullspace(
        work_record,
        do_zero=False,
        do_contradicting=False,
        do_combine=True,
        include_compression=True
    )


def compress_coupled_all(work_record: WorkRecord, 
                        include_compression: bool = True) -> bool:
    """
    Apply all coupled compression methods.
    
    Args:
        work_record: Working data to compress
        include_compression: If True, merge reactions; if False, only remove
        
    Returns:
        True if any compressions were applied
    """
    compressor = CoupledCompression()
    return compressor.compress_nullspace(
        work_record,
        do_zero=True,
        do_contradicting=True,
        do_combine=True,
        include_compression=include_compression
    )