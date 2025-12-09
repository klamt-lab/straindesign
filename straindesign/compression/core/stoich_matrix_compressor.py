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
        
        # Check for problematic methods and warn users
        problematic_methods = {
            CompressionMethod.UNIQUE_FLOWS: "UniqueFlows has severe over-compression bugs - strongly discouraged"
        }
        
        for method in self._compression_methods:
            if method in problematic_methods:
                LOG.warning(f"Compression method {method}: {problematic_methods[method]}")
        
        # Log unsupported methods (warning in Java)
        supported_methods = {
            CompressionMethod.COUPLED_ZERO,
            CompressionMethod.COUPLED_COMBINE, 
            CompressionMethod.COUPLED_CONTRADICTING,
            CompressionMethod.UNIQUE_FLOWS,
            CompressionMethod.DEAD_END,
            CompressionMethod.RECURSIVE
        }
        
        for method in self._compression_methods:
            if method not in supported_methods:
                LOG.warning(f"Compression method {method} is not fully supported")
    
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
        do_rec = CompressionMethod.RECURSIVE in self._compression_methods
        do_zer = CompressionMethod.COUPLED_ZERO in self._compression_methods
        do_con = CompressionMethod.COUPLED_CONTRADICTING in self._compression_methods
        do_com = CompressionMethod.COUPLED_COMBINE in self._compression_methods
        do_unq = CompressionMethod.UNIQUE_FLOWS in self._compression_methods
        do_dea = CompressionMethod.DEAD_END in self._compression_methods
        
        # Group method flags for phases
        do_unq_inc = do_dea  # Dead-end phase
        do_unq_com = do_unq  # Unique flows phase
        do_nul_inc = do_zer or do_con  # Inconsistency phase
        do_nul_com = do_com  # Combination phase
        
        # Phase 1: Remove suppressed reactions and handle dead-ends/inconsistencies
        compressed_any = work_record.remove_reactions(suppressed_reactions)
        
        while True:
            it_count = work_record.stats.inc_compression_iteration()
            LOG.debug(f"compression iteration {it_count + 1} (dead-ends/inconsistencies)")
            
            compressed_any = False
            if do_unq_inc:
                # Store initial size to check if anything was removed
                initial_size = work_record.size.metas
                work_record.remove_unused_metabolites()
                compressed_any |= (work_record.size.metas < initial_size)
                compressed_any |= self._unique(work_record, include_compression=False)
            
            if do_nul_inc:
                # Store initial size to check if anything was removed
                initial_metas = work_record.size.metas
                work_record.remove_unused_metabolites()
                compressed_any |= (work_record.size.metas < initial_metas)
                compressed_any |= self._nullspace(work_record, include_compression=False)
            
            if not (compressed_any and do_rec):
                break
        
        # Phase 2: Full compression - unique flows
        if do_unq_com:
            while True:
                it_count = work_record.stats.inc_compression_iteration()
                LOG.debug(f"compression iteration {it_count + 1} (unique fluxes)")
                
                compressed_any = work_record.remove_unused_metabolites()
                compressed_any |= self._unique(work_record, include_compression=True)
                
                if not (compressed_any and do_rec):
                    break
            
            if compressed_any and not do_rec:
                work_record.remove_unused_metabolites()
        
        # Phase 3: Full compression - nullspace
        if do_nul_com:
            while True:
                it_count = work_record.stats.inc_compression_iteration()
                LOG.debug(f"compression iteration {it_count + 1} (nullspace)")
                
                compressed_any = work_record.remove_unused_metabolites()
                compressed_any |= self._nullspace(work_record, include_compression=True)
                
                if not (compressed_any and do_rec):
                    break
            
            if compressed_any and not do_rec:
                work_record.remove_unused_metabolites()
        
        # Phase 4: Combined full compression
        if do_rec and do_unq_com and do_nul_com:
            while True:
                it_count = work_record.stats.inc_compression_iteration()
                LOG.debug(f"compression iteration {it_count + 1} (unique/nullspace)")
                
                compressed_any = work_record.remove_unused_metabolites()
                compressed_any |= self._unique(work_record, include_compression=True)
                compressed_any |= work_record.remove_unused_metabolites()
                compressed_any |= self._nullspace(work_record, include_compression=True)
                
                if not compressed_any or not do_rec:
                    break
        
        # Log compression statistics and return result
        work_record.stats.write_to_log()
        return work_record.get_truncated()
    
    def _nullspace(self, work_record: WorkRecord, include_compression: bool) -> bool:
        """
        Perform nullspace-based compression methods.
        
        Args:
            work_record: Working compression record
            include_compression: Whether to include CoupledCombine compression
            
        Returns:
            True if any compression was performed
        """
        # Determine compression options
        do_zer = CompressionMethod.COUPLED_ZERO in self._compression_methods
        do_con = CompressionMethod.COUPLED_CONTRADICTING in self._compression_methods
        do_com = CompressionMethod.COUPLED_COMBINE in self._compression_methods
        do_cpl = do_con or do_com
        
        # Create nullspace record and perform compressions
        nullspace_record = NullspaceRecord(work_record)
        LOG.debug(f"Nullspace kernel computed: {nullspace_record.kernel.get_row_count()}x{nullspace_record.kernel.get_column_count()}")
        compressed_any = False
        
        if do_zer:
            compressed_any |= self._nullspace_zero_flux_reactions(nullspace_record)
        
        if do_cpl:
            compressed_any |= self._nullspace_coupled_reactions(nullspace_record, include_compression)
        
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
    
    def _nullspace_coupled_reactions(self, nullspace_record: 'NullspaceRecord',
                                   include_compression: bool) -> bool:
        """
        Find and handle coupled reactions (CoupledContradicting and CoupledCombine methods).
        
        Two reactions are coupled if they have a constant ratio in all nullspace vectors.
        Contradictory couplings (inconsistent with reversibility) are removed.
        Valid couplings can be combined into single reactions.
        
        Args:
            nullspace_record: Nullspace computation record
            include_compression: Whether to combine coupled reactions (CoupledCombine)
            
        Returns:
            True if any coupled reactions were processed
        """
        from ..math.big_fraction import BigFraction
        
        # Determine compression options
        do_con = CompressionMethod.COUPLED_CONTRADICTING in self._compression_methods
        do_com = CompressionMethod.COUPLED_COMBINE in self._compression_methods
        
        kernel = nullspace_record.kernel
        stoich = nullspace_record.cmp
        post = nullspace_record.post
        reversible = nullspace_record.reversible
        size = nullspace_record.size
        
        kernel_cols = kernel.get_column_count()
        num_reactions = size.reacs
        
        # Find coupled reaction groups
        groups = []  # List of reaction index groups (no single-element groups)
        ratios = [None] * num_reactions  # ratios[reacB] = ratio of reacA/reacB for coupled reactions
        
        for reac_a in range(num_reactions):
            if ratios[reac_a] is None:  # Not yet processed
                group = None
                
                for reac_b in range(reac_a + 1, num_reactions):
                    # Calculate ratio reac_a / reac_b across all kernel columns
                    ratio = None
                    
                    for col in range(kernel_cols):
                        val_a_num = kernel.get_big_integer_numerator_at(reac_a, col)
                        val_b_num = kernel.get_big_integer_numerator_at(reac_b, col)
                        is_zero_a = self._is_zero(val_a_num)
                        is_zero_b = self._is_zero(val_b_num)
                        
                        if is_zero_a != is_zero_b:
                            # Different zero patterns - not coupled
                            ratio = BigFraction.ZERO
                            break
                        elif not is_zero_a:
                            # Both non-zero - check ratio consistency
                            val_a = kernel.get_big_fraction_value_at(reac_a, col)
                            val_b = kernel.get_big_fraction_value_at(reac_b, col)
                            current_ratio = val_a.divide(val_b).reduce()
                            
                            if ratio is None:
                                ratio = current_ratio
                            elif not (ratio == current_ratio):
                                # Inconsistent ratio - not coupled
                                ratio = BigFraction.ZERO
                                break
                    
                    if ratio is None:
                        raise RuntimeError("no zero rows expected here")
                    elif not self._is_zero(ratio.get_numerator()):
                        # Found coupled reactions
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
                if do_con:
                    # Remove contradictory coupled reactions
                    group_names = [nullspace_record.reac_names[r] for r in group]
                    LOG.debug(f"found and removed inconsistently coupled reactions: {group_names}")
                    
                    for reac_idx in group:
                        reactions_to_remove.add(reac_idx)
                        nullspace_record.stats.inc_contradicting_reactions()
                else:
                    LOG.debug("ignoring inconsistently coupled reactions due to compression settings")
            elif do_com and include_compression:
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
        
        # Combine each slave reaction into the master
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
    
    def _unique(self, work_record: WorkRecord, include_compression: bool) -> bool:
        """
        Perform unique flow compression methods (UniqueFlows and DeadEnd).
        
        Args:
            work_record: Working compression record
            include_compression: Whether to include UniqueFlows compression
            
        Returns:
            True if any compression was performed
        """
        original_size = work_record.size.clone()
        
        meta_index = 0
        while meta_index < work_record.size.metas:
            previous_meta_count = work_record.size.metas
            self._unique_metabolite(work_record, meta_index, include_compression)
            
            # If metabolite count unchanged, move to next metabolite
            # Otherwise, current index now represents a different metabolite
            if work_record.size.metas == previous_meta_count:
                meta_index += 1
        
        return not (original_size == work_record.size)
    
    def _unique_metabolite(self, work_record: WorkRecord, meta_index: int,
                          include_compression: bool) -> bool:
        """
        Process a single metabolite for unique flow compression.
        
        Args:
            work_record: Working compression record
            meta_index: Index of metabolite to process
            include_compression: Whether to include UniqueFlows compression
            
        Returns:
            True if the metabolite was processed (removed or compressed)
        """
        # Get matrix references
        stoich = work_record.cmp
        post = work_record.post
        reversible = work_record.reversible
        size = work_record.size
        
        # Classify reactions involving this metabolite
        educt_reactions = set()      # Reactions consuming metabolite (negative coefficient)
        product_reactions = set()    # Reactions producing metabolite (positive coefficient)  
        reversible_reactions = set() # Reversible reactions involving metabolite
        
        for reac_index in range(size.reacs):
            coefficient_sign = stoich.get_signum_at(meta_index, reac_index)
            if coefficient_sign != 0:
                if reversible[reac_index]:
                    reversible_reactions.add(reac_index)
                else:
                    if coefficient_sign < 0:
                        educt_reactions.add(reac_index)
                    elif coefficient_sign > 0:
                        product_reactions.add(reac_index)
        
        edu_count = len(educt_reactions)
        pro_count = len(product_reactions)
        rev_count = len(reversible_reactions)
        
        # Check for unused metabolite
        if edu_count == 0 and pro_count == 0 and rev_count == 0:
            LOG.debug(f"found and removed unused metabolite: {work_record.meta_names[meta_index]}")
            work_record.remove_metabolite(meta_index)
            work_record.stats.inc_unused_metabolite()
            return True
        
        # Check for dead-end metabolite
        is_dead_end = (
            (edu_count == 0 and pro_count == 0 and rev_count == 1) or  # Only 1 reversible
            (edu_count == 0 and rev_count == 0) or                     # Only producers
            (pro_count == 0 and rev_count == 0)                        # Only consumers
        )
        
        if is_dead_end:
            all_reactions = educt_reactions | product_reactions | reversible_reactions
            reaction_names = [work_record.reac_names[r] for r in all_reactions]
            
            LOG.debug(f"found and removed dead-end metabolite/reaction(s): "
                     f"{work_record.meta_names[meta_index]} / {reaction_names}")
            
            work_record.remove_metabolite(meta_index)
            work_record.remove_reactions_by_indices(all_reactions)
            work_record.stats.inc_dead_end_metabolite_reactions(len(all_reactions))
            return True
        
        # Unique flow compression (if enabled)
        if include_compression:
            reaction_to_remove = None
            reactions_to_merge = None
            
            # Case 1: No reversible reactions, uniquely consumed or produced
            if rev_count == 0 and (edu_count == 1 or pro_count == 1):
                if edu_count == 1:
                    reaction_to_remove = list(educt_reactions)[0]
                    reactions_to_merge = product_reactions
                    LOG.debug(f"found uniquely consumed metabolite: {work_record.meta_names[meta_index]}")
                else:  # pro_count == 1
                    reaction_to_remove = list(product_reactions)[0]
                    reactions_to_merge = educt_reactions
                    LOG.debug(f"found uniquely produced metabolite: {work_record.meta_names[meta_index]}")
            
            # Case 2: One reversible reaction, all others same direction
            elif rev_count == 1 and (edu_count == 0 or pro_count == 0):
                reaction_to_remove = list(reversible_reactions)[0]
                reactions_to_merge = product_reactions if edu_count == 0 else educt_reactions
                direction = "consumed" if edu_count == 0 else "produced"
                LOG.debug(f"found uniquely (reversibly) {direction} metabolite: {work_record.meta_names[meta_index]}")
            
            # Case 3: Only 2 reversible reactions
            elif rev_count == 2 and edu_count == 0 and pro_count == 0:
                reversible_list = list(reversible_reactions)
                reaction_to_remove = reversible_list[0]
                reactions_to_merge = {reversible_list[1]}
                reaction_names = [work_record.reac_names[r] for r in reversible_reactions]
                LOG.debug(f"found and removed metabolite between 2 reversible reactions: "
                         f"{work_record.meta_names[meta_index]} / {reaction_names}")
            
            # Perform the compression if applicable
            if reactions_to_merge is not None:
                return self._merge_reactions(work_record, meta_index, reaction_to_remove, reactions_to_merge)
        
        return False
    
    def _merge_reactions(self, work_record: WorkRecord, meta_index: int,
                        reaction_to_remove: int, reactions_to_merge: Set[int]) -> bool:
        """
        Merge reactions by eliminating a metabolite (UniqueFlows compression).
        
        Args:
            work_record: Working compression record
            meta_index: Index of metabolite being eliminated
            reaction_to_remove: Index of reaction being removed
            reactions_to_merge: Set of reaction indices to merge with removed reaction
            
        Returns:
            True indicating compression was performed
        """
        stoich = work_record.cmp
        post = work_record.post
        
        # Get stoichiometric coefficient of metabolite in reaction to remove
        from ..math.big_fraction import BigFraction
        rm_stoich = stoich.get_big_fraction_value_at(meta_index, reaction_to_remove)
        kp_multiplier = rm_stoich.abs()
        
        # Merge each reaction with the reaction to remove
        for reaction_index in reactions_to_merge:
            kp_stoich = stoich.get_big_fraction_value_at(meta_index, reaction_index)
            
            # Calculate multiplier to eliminate the metabolite
            # The signs ensure the metabolite coefficient becomes zero
            if rm_stoich.signum() < 0:
                rm_multiplier = kp_stoich
            else:
                rm_multiplier = kp_stoich.negate()
            
            # Add scaled columns: reaction_index = kp_multiplier * reaction_index + rm_multiplier * reaction_to_remove
            self._add_column_multiple_to(stoich, reaction_index, kp_multiplier, reaction_to_remove, rm_multiplier)
            self._add_column_multiple_to(post, reaction_index, kp_multiplier, reaction_to_remove, rm_multiplier)
        
        # Remove the metabolite and reaction
        work_record.remove_metabolite(meta_index)
        work_record.remove_reaction(reaction_to_remove)
        work_record.stats.inc_unique_flow_reactions()
        
        return True
    
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