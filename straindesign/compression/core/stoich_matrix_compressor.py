#!/usr/bin/env python3
"""
EFMTool StoichMatrixCompressor Implementation

Python port of ch.javasoft.metabolic.compress.StoichMatrixCompressor
Core compression algorithms for metabolic networks using stoichiometric matrices.

From Java source: efmtool_source/ch/javasoft/metabolic/compress/StoichMatrixCompressor.java
Ported line-by-line for exact algorithmic compatibility
"""

import logging
from typing import List, Set, Optional, TYPE_CHECKING

from .compression_method import CompressionMethod
from .compression_record import CompressionRecord
from .work_record import WorkRecord

if TYPE_CHECKING:
    from ..math.readable_bigint_rational_matrix import ReadableBigIntegerRationalMatrix
    from ..math.bigint_rational_matrix import BigIntegerRationalMatrix
    from ..math.big_fraction import BigFraction


# Set up logging
LOG = logging.getLogger(__name__)


class StoichMatrixCompressor:
    """
    StoichMatrixCompressor compresses metabolic networks using various compression methods.
    
    The compress() method uses stoichiometric matrix and reaction reversibilities as input,
    and returns 3 matrices pre, post and cmp, such that pre * stoich * post == cmp.
    
    Elementary modes can be computed on the compressed matrix and mapped back to 
    the original network using: efm_original = post * efm_compressed
    """
    
    def __init__(self, *compression_methods: CompressionMethod):
        """
        Initialize compressor with specified compression methods.
        
        Args:
            compression_methods: Methods to use for compression. 
                                If none provided, uses CompressionMethod.STANDARD
        """
        if not compression_methods:
            self._compression_methods = CompressionMethod.standard()
        else:
            self._compression_methods = list(compression_methods)

    def compress(self, stoich: 'ReadableBigIntegerRationalMatrix', reversible: List[bool],
                 meta_names: List[str], reac_names: List[str],
                 suppressed_reactions: Optional[Set[str]] = None) -> CompressionRecord:
        """
        Compress the metabolic network returning compression matrices.

        Returns CompressionRecord containing matrices pre, cmp, post such that:
        pre * stoich * post == cmp

        Args:
            stoich: Original stoichiometric matrix (m metabolites x r reactions)
            reversible: Reversibility flags for reactions
            meta_names: Names of metabolites
            reac_names: Names of reactions
            suppressed_reactions: Optional set of reaction names to remove

        Returns:
            CompressionRecord with compression matrices and statistics
        """
        work_record = WorkRecord(stoich, reversible, meta_names, reac_names)

        # Determine compression options from methods
        do_nullspace = CompressionMethod.NULLSPACE in self._compression_methods
        do_recursive = CompressionMethod.RECURSIVE in self._compression_methods

        # Remove suppressed reactions first
        work_record.remove_reactions(suppressed_reactions)

        if do_nullspace:
            # Iterative nullspace-based compression
            while True:
                it_count = work_record.stats.inc_compression_iteration()
                LOG.debug(f"compression iteration {it_count + 1}")

                # Remove unused metabolites
                initial_metas = work_record.size.metas
                work_record.remove_unused_metabolites()
                compressed_any = (work_record.size.metas < initial_metas)

                # Full nullspace compression (zero-flux, contradicting, combine)
                compressed_any |= self._nullspace(work_record)

                if not (compressed_any and do_recursive):
                    break

        # Final cleanup
        work_record.remove_unused_metabolites()

        # Log compression statistics and return result
        work_record.stats.write_to_log()
        return work_record.get_truncated()

    def _nullspace(self, work_record: WorkRecord) -> bool:
        """
        Perform full nullspace-based compression.

        This includes:
        - Removing zero-flux reactions (reactions with all zeros in nullspace)
        - Removing contradicting coupled reactions (inconsistent with reversibility)
        - Merging consistently coupled reactions

        Args:
            work_record: Working compression record

        Returns:
            True if any compression was performed
        """
        # Create nullspace record and perform compressions
        nullspace_record = NullspaceRecord(work_record)
        LOG.debug(f"Nullspace kernel computed: {nullspace_record.kernel.get_row_count()}x{nullspace_record.kernel.get_column_count()}")

        # 1. Remove zero-flux reactions
        compressed_any = self._nullspace_zero_flux_reactions(nullspace_record)

        # 2. Handle coupled reactions (remove contradicting, merge consistent)
        compressed_any |= self._nullspace_coupled_reactions(nullspace_record)

        if compressed_any:
            work_record.remove_unused_metabolites()

        return compressed_any
    
    def _nullspace_zero_flux_reactions(self, nullspace_record: 'NullspaceRecord') -> bool:
        """
        Find and remove reactions that must have zero flux (CoupledZero method).
        
        A reaction has zero flux if its corresponding row in the nullspace kernel
        is all zeros, meaning it cannot participate in any steady-state flux distribution.
        
        Args:
            nullspace_record: Nullspace computation record
            
        Returns:
            True if any zero flux reactions were found and removed
        """
        kernel = nullspace_record.kernel
        size = nullspace_record.size
        kernel_cols = kernel.get_column_count()
        
        LOG.debug(f"Checking {size.reacs} reactions for zero flux in {kernel_cols}-dimensional nullspace")
        
        # First, identify all zero flux reactions (collect indices before removing anything)
        zero_flux_indices = []
        
        for reac_index in range(size.reacs):
            # Check if all kernel entries for this reaction are zero
            all_zero = True
            for col in range(kernel_cols):
                numerator = kernel.get_big_integer_numerator_at(reac_index, col)
                if not self._is_zero(numerator):
                    all_zero = False
                    break
            
            if all_zero:
                zero_flux_indices.append(reac_index)
                LOG.debug(f"identified zero flux reaction: {nullspace_record.reac_names[reac_index]}")
        
        # Remove reactions in reverse order to avoid index shifting issues
        any_zero_flux = len(zero_flux_indices) > 0
        
        for reac_index in reversed(zero_flux_indices):
            LOG.debug(f"removing zero flux reaction: {nullspace_record.reac_names[reac_index]}")
            nullspace_record.remove_reaction(reac_index)
            nullspace_record.stats.inc_zero_flux_reactions()
        
        return any_zero_flux
    
    def _nullspace_coupled_reactions(self, nullspace_record: 'NullspaceRecord') -> bool:
        """
        Find and handle coupled reactions.

        Two reactions are coupled if they have a constant ratio in all nullspace vectors.
        - Contradictory couplings (inconsistent with reversibility) are removed.
        - Valid couplings are combined into single reactions.

        Optimized algorithm using zero-pattern hashing:
        1. Group reactions by their zero/nonzero pattern (O(n*k))
        2. Only check ratio consistency within groups (O(sum of group sizes))
        This reduces O(n²k) to approximately O(n*k) for typical sparse kernels.

        Args:
            nullspace_record: Nullspace computation record

        Returns:
            True if any coupled reactions were processed
        """
        from ..math.big_fraction import BigFraction

        kernel = nullspace_record.kernel
        stoich = nullspace_record.cmp
        post = nullspace_record.post
        reversible = nullspace_record.reversible
        size = nullspace_record.size

        kernel_cols = kernel.get_column_count()
        num_reactions = size.reacs

        # Step 1: Group reactions by zero-pattern signature (exact arithmetic)
        # This is O(n*k) and dramatically reduces the search space for coupled reaction detection
        patterns = {}
        for reac_idx in range(num_reactions):
            pattern = tuple(
                kernel.get_big_integer_numerator_at(reac_idx, col) == 0
                for col in range(kernel_cols)
            )
            if pattern not in patterns:
                patterns[pattern] = []
            patterns[pattern].append(reac_idx)

        # Filter to patterns with multiple reactions (potential couplings)
        potential_groups = [indices for indices in patterns.values() if len(indices) > 1]

        # Step 2: Within each pattern group, find coupled reactions
        # Use the original O(n²) algorithm but only within pattern groups
        groups = []  # List of coupled reaction groups
        ratios = [None] * num_reactions  # ratios[reacB] = ratio of reacA/reacB

        for potential_group in potential_groups:
            # Find non-zero columns for this group's pattern
            ref_reac = potential_group[0]
            nonzero_cols = [col for col in range(kernel_cols)
                           if kernel.get_big_integer_numerator_at(ref_reac, col) != 0]

            if not nonzero_cols:
                continue  # All zeros - skip

            # Within this pattern group, use the original pairwise algorithm
            # to correctly handle transitivity
            local_ratios = [None] * len(potential_group)
            idx_map = {r: i for i, r in enumerate(potential_group)}

            for i, reac_a in enumerate(potential_group):
                if local_ratios[i] is None:  # Not yet in a group
                    group = None

                    for j in range(i + 1, len(potential_group)):
                        reac_b = potential_group[j]
                        # Calculate ratio reac_a / reac_b across non-zero columns
                        ratio = None

                        for col in nonzero_cols:
                            val_a = kernel.get_big_fraction_value_at(reac_a, col)
                            val_b = kernel.get_big_fraction_value_at(reac_b, col)
                            current_ratio = val_a.divide(val_b).reduce()

                            if ratio is None:
                                ratio = current_ratio
                            elif not (ratio == current_ratio):
                                # Inconsistent ratio - not coupled
                                ratio = BigFraction.ZERO
                                break

                        if ratio is not None and not self._is_zero(ratio.get_numerator()):
                            # Found coupled reactions
                            local_ratios[j] = ratio
                            ratios[reac_b] = ratio
                            if group is None:
                                group = [reac_a]
                            group.append(reac_b)

                    if group is not None:
                        groups.append(group)
        
        # Process each coupled group
        reactions_to_remove = set()

        for group in groups:
            # Check consistency with reversibility constraints
            group_valid = self._check_coupling_consistency(group, ratios, reversible)

            if not group_valid:
                # Remove contradictory coupled reactions
                group_names = [nullspace_record.reac_names[r] for r in group]
                LOG.debug(f"found and removed inconsistently coupled reactions: {group_names}")

                for reac_idx in group:
                    reactions_to_remove.add(reac_idx)
                    nullspace_record.stats.inc_contradicting_reactions()
            else:
                # Combine valid coupled reactions
                self._combine_coupled_reactions(group, ratios, nullspace_record)
                # Remove all but the first (master) reaction
                for reac_idx in group[1:]:
                    reactions_to_remove.add(reac_idx)

        # Remove marked reactions
        if reactions_to_remove:
            nullspace_record.work_record.remove_reactions_by_indices(reactions_to_remove)
            return True

        return False
    
    def _check_coupling_consistency(self, group: List[int], ratios: List['BigFraction'],
                                   reversible: List[bool]) -> bool:
        """
        Check if coupled reaction group is consistent with reversibility constraints.
        
        Args:
            group: List of coupled reaction indices
            ratios: Coupling ratios for reactions
            reversible: Reversibility flags for reactions
            
        Returns:
            True if the coupling is consistent with reversibility constraints
        """
        # Try both forward and backward directions
        for forward_direction in [True, False]:
            all_consistent = forward_direction or reversible[group[0]]
            
            for i in range(1, len(group)):
                reac_idx = group[i]
                ratio = ratios[reac_idx]
                
                # Check if reaction direction is consistent with coupling ratio
                ratio_positive = ratio.signum() > 0
                direction_match = forward_direction == ratio_positive
                
                if not (direction_match or reversible[reac_idx]):
                    all_consistent = False
                    break
            
            if all_consistent:
                return True
        
        return False
    
    def _combine_coupled_reactions(self, group: List[int], ratios: List['BigFraction'],
                                  nullspace_record: 'NullspaceRecord') -> None:
        """
        Combine coupled reactions into the first reaction in the group.

        Args:
            group: List of coupled reaction indices (first is master)
            ratios: Coupling ratios for reactions
            nullspace_record: Nullspace computation record
        """
        from ..math.big_fraction import BigFraction

        stoich = nullspace_record.cmp
        post = nullspace_record.post
        master_idx = group[0]

        group_names = [nullspace_record.reac_names[r] for r in group]
        LOG.debug(f"found and combined coupled reactions: {group_names}")

        # Check if using FLINT matrices for optimized path
        use_flint = hasattr(stoich, 'get_flint_value_at')

        if use_flint:
            from flint import fmpq
            # FLINT-optimized path: use native fmpq operations
            for i in range(1, len(group)):
                slave_idx = group[i]
                ratio = ratios[slave_idx]  # master/slave ratio
                # multiplier = 1/ratio
                mult = fmpq(ratio.denominator, ratio.numerator)

                # Update stoichiometric matrix
                for meta in range(stoich.get_row_count()):
                    master_val = stoich.get_flint_value_at(meta, master_idx)
                    slave_val = stoich.get_flint_value_at(meta, slave_idx)
                    stoich.set_flint_value_at(meta, master_idx, master_val + mult * slave_val)

                # Update post matrix
                for orig_reac in range(post.get_row_count()):
                    master_val = post.get_flint_value_at(orig_reac, master_idx)
                    slave_val = post.get_flint_value_at(orig_reac, slave_idx)
                    post.set_flint_value_at(orig_reac, master_idx, master_val + mult * slave_val)
        else:
            # Pure Python path: use BigFraction
            for i in range(1, len(group)):
                slave_idx = group[i]
                ratio = ratios[slave_idx]  # master/slave ratio

                # Add slave reaction to master: master = master + (1/ratio) * slave
                multiplier = BigFraction.ONE.divide(ratio)

                # Update stoichiometric matrix
                for meta in range(stoich.get_row_count()):
                    master_val = stoich.get_big_fraction_value_at(meta, master_idx)
                    slave_val = stoich.get_big_fraction_value_at(meta, slave_idx)
                    combined_val = master_val.add(multiplier.multiply(slave_val))
                    stoich.set_value_at(meta, master_idx, combined_val)

                # Update post matrix
                for orig_reac in range(post.get_row_count()):
                    master_val = post.get_big_fraction_value_at(orig_reac, master_idx)
                    slave_val = post.get_big_fraction_value_at(orig_reac, slave_idx)
                    combined_val = master_val.add(multiplier.multiply(slave_val))
                    post.set_value_at(orig_reac, master_idx, combined_val)

        nullspace_record.stats.inc_coupled_reactions_count(len(group))

    def _add_column_multiple_to(self, matrix: 'BigIntegerRationalMatrix',
                               dst_column: int, dst_factor: 'BigFraction',
                               src_column: int, src_factor: 'BigFraction') -> None:
        """
        Add scaled source column to destination column: dst = dst_factor * dst + src_factor * src

        Args:
            matrix: Matrix to modify
            dst_column: Destination column index
            dst_factor: Scaling factor for destination column
            src_column: Source column index
            src_factor: Scaling factor for source column
        """
        # Check if using FLINT matrices for optimized path
        use_flint = hasattr(matrix, 'get_flint_value_at')

        if use_flint:
            from flint import fmpq
            # Convert factors to FLINT fmpq
            df = fmpq(dst_factor.numerator, dst_factor.denominator)
            sf = fmpq(src_factor.numerator, src_factor.denominator)

            for row in range(matrix.get_row_count()):
                dst_val = matrix.get_flint_value_at(row, dst_column)
                src_val = matrix.get_flint_value_at(row, src_column)
                matrix.set_flint_value_at(row, dst_column, dst_val * df + src_val * sf)
        else:
            for row in range(matrix.get_row_count()):
                dst_val = matrix.get_big_fraction_value_at(row, dst_column)
                src_val = matrix.get_big_fraction_value_at(row, src_column)

                result = (dst_val * dst_factor) + (src_val * src_factor)
                matrix.set_value_at(row, dst_column, result)
    
    @staticmethod
    def _is_zero(numerator) -> bool:
        """Check if BigInteger numerator represents zero"""
        return numerator == 0 or (hasattr(numerator, 'signum') and numerator.signum() == 0)


class NullspaceRecord:
    """
    Helper class for nullspace-based compression operations.
    Contains the nullspace (kernel) computation and related data.
    """
    
    def __init__(self, work_record: WorkRecord):
        """
        Initialize nullspace record with kernel computation.
        
        Args:
            work_record: Current working compression record
        """
        # Store references to work record data
        self.work_record = work_record
        self.size = work_record.size
        self.cmp = work_record.cmp
        self.post = work_record.post
        self.reversible = work_record.reversible
        self.stats = work_record.stats
        self.meta_names = work_record.meta_names
        self.reac_names = work_record.reac_names
        
        # Compute nullspace (kernel) of compressed stoichiometric matrix
        from ..math.gauss import Gauss
        gauss_instance = Gauss.get_rational_instance()
        
        # Compute right nullspace of stoichiometric matrix (for flux vectors)
        self.kernel = gauss_instance.nullspace(work_record.cmp)
        
        LOG.debug(f"nullspace computation: kernel is {self.kernel.get_row_count()}x{self.kernel.get_column_count()}")
    
    def remove_reaction(self, reaction_index: int) -> None:
        """Remove a reaction from the work record"""
        self.work_record.remove_reaction(reaction_index)
    
    def get_reaction_details(self, reaction_index: int) -> str:
        """Get detailed reaction description for logging"""
        return f"reaction {reaction_index}: {self.reac_names[reaction_index]}"