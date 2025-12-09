#!/usr/bin/env python3
"""
EFMTool DuplicateGeneCompressor Implementation

Python port of ch.javasoft.metabolic.compress.DuplicateGeneCompressor
Compresses metabolic networks by removing duplicate gene reactions with equal stoichiometry.

From Java source: efmtool_source/ch/javasoft/metabolic/compress/DuplicateGeneCompressor.java
Ported line-by-line for exact algorithmic compatibility
"""

import logging
from typing import List, Set, Optional, TYPE_CHECKING

from .compression_record import CompressionRecord
from .work_record import WorkRecord

if TYPE_CHECKING:
    from ..math.readable_bigint_rational_matrix import ReadableBigIntegerRationalMatrix
    from ..math.bigint_rational_matrix import BigIntegerRationalMatrix
    from ..math.big_fraction import BigFraction


# Set up logging
LOG = logging.getLogger(__name__)


class DuplicateGeneCompressor:
    """
    DuplicateGeneCompressor compresses metabolic networks by removing duplicate gene reactions.
    
    Duplicate gene reactions are reactions with equal stoichiometry (proportional columns).
    The compress() method returns matrices dupelim and dupfree such that:
    stoich * dupelim == dupfree
    
    This is a static utility class - no instances should be created.
    """
    
    def __init__(self):
        """Private constructor - this is a static utility class"""
        raise RuntimeError("DuplicateGeneCompressor is a static utility class - do not instantiate")
    
    @staticmethod
    def compress(stoich: 'ReadableBigIntegerRationalMatrix', reversible: List[bool],
                 meta_names: List[str], reac_names: List[str], 
                 extended: bool = False) -> CompressionRecord:
        """
        Compress metabolic network by removing duplicate gene reactions.
        
        Returns compression record containing matrices dupelim, dupfree such that:
        stoich * dupelim == dupfree
        
        Args:
            stoich: Original stoichiometric matrix
            reversible: Reversibility flags for reactions
            meta_names: Metabolite names
            reac_names: Reaction names
            extended: If False, duplicate reactions must have same directionality.
                     If True, allows different directionalities (extended compression)
        
        Returns:
            CompressionRecord with compression matrices and statistics
        """
        work_record = WorkRecord(stoich, reversible, meta_names, reac_names)
        
        # Start compression
        it_count = work_record.stats.inc_compression_iteration()
        LOG.debug(f"compression iteration {it_count + 1} (duplicate genes)")
        
        DuplicateGeneCompressor._compress_duplicate_genes(work_record, extended)
        
        # Log compression statistics and return result
        work_record.stats.write_to_log()
        return work_record.get_truncated()
    
    @staticmethod
    def _compress_duplicate_genes(work_record: WorkRecord, extended: bool) -> bool:
        """
        Perform duplicate gene compression on the work record.
        
        Args:
            work_record: Working compression record
            extended: Whether to allow extended compression (different directionalities)
            
        Returns:
            True if any duplicate genes were found and compressed
        """
        # Aliases for cleaner code
        stoich = work_record.cmp  # Note: uses compressed matrix (same as StoichMatrixCompressor)
        reversible = work_record.reversible
        size = work_record.size
        
        num_metabolites = size.metas
        num_reactions = size.reacs
        
        # Find duplicate gene groups by comparing reaction stoichiometries
        groups = []  # List of reaction index groups 
        ratios = [None] * num_reactions  # ratios[reacB] = reacA/reacB for duplicate reactions
        processed_reactions = set()  # Track reactions already in groups
        
        for reac_a in range(num_reactions):
            if reac_a in processed_reactions:
                continue  # Skip if already processed
            
            group = None
            
            for reac_b in range(reac_a + 1, num_reactions):
                if reac_b in processed_reactions:
                    continue  # Skip if already processed
                # Calculate stoichiometric ratio reac_a / reac_b across all metabolites
                ratio = None
                
                for meta in range(num_metabolites):
                    val_a_num = stoich.get_big_integer_numerator_at(meta, reac_a)
                    val_b_num = stoich.get_big_integer_numerator_at(meta, reac_b)
                    is_zero_a = DuplicateGeneCompressor._is_zero(val_a_num)
                    is_zero_b = DuplicateGeneCompressor._is_zero(val_b_num)
                    
                    if is_zero_a != is_zero_b:
                        # Different zero patterns - not proportional
                        from ..math.big_fraction import BigFraction
                        ratio = BigFraction.ZERO
                        break
                    elif not is_zero_a:
                        # Both non-zero - check proportionality
                        val_a = stoich.get_big_fraction_value_at(meta, reac_a)
                        val_b = stoich.get_big_fraction_value_at(meta, reac_b)
                        current_ratio = val_a.divide(val_b).reduce()
                        
                        if ratio is None:
                            ratio = current_ratio
                        elif ratio != current_ratio:
                            # Inconsistent ratio - not proportional
                            from ..math.big_fraction import BigFraction
                            ratio = BigFraction.ZERO
                            break
                
                if ratio is None:
                    # Both reactions are all-zero (should not happen)
                    LOG.warning(f"zero stoichiometries found: {work_record.get_reaction_names([reac_a, reac_b])}")
                    work_record.log_reaction_details(logging.WARNING, "  ", reac_a)
                    work_record.log_reaction_details(logging.WARNING, "  ", reac_b)
                    raise RuntimeError("no zero stoichiometries expected")
                elif not DuplicateGeneCompressor._is_zero(ratio.get_numerator()):
                    # Found proportional (duplicate gene) reactions
                    ratios[reac_b] = ratio
                    if group is None:
                        group = [reac_a]  # Use Python list instead of IntArray
                        processed_reactions.add(reac_a)  # Mark as processed
                    group.append(reac_b)
                    processed_reactions.add(reac_b)  # Mark as processed
            
            if group is not None:
                groups.append(group)
        
        # Process each duplicate gene group
        reactions_to_remove = set()
        
        for group in groups:
            # Analyze reversibility of duplicate gene reactions
            kept_reaction_idx = group[0]
            kept_reversible = reversible[kept_reaction_idx]  # Will the kept reaction be reversible?
            same_reversibility = kept_reversible  # Track if all have same directionality
            scaled = False  # Track if any ratio != 1 (scaled duplicates)
            
            for i in range(1, len(group)):
                reac_idx = group[i]
                ratio = ratios[reac_idx]
                
                # Update kept reaction reversibility (union of all possibilities)
                kept_reversible = kept_reversible or reversible[reac_idx] or (ratio.signum() < 0)
                
                # Check for scaling
                scaled = scaled or not ratio.is_one()
                
                # Check directionality consistency
                if same_reversibility is not None:
                    current_reversible = reversible[reac_idx]
                    if not ((same_reversibility and current_reversible) or 
                           (not same_reversibility and not current_reversible and ratio.signum() > 0)):
                        same_reversibility = None
                        break
            
            # Decide whether to compress this group
            if same_reversibility is not None or extended:
                # Compress - remove duplicate reactions and keep the first one
                if scaled:
                    LOG.info(f"found and removed duplicate gene reactions (some ratios unequal to one): "
                            f"{work_record.get_reaction_names(group)}")
                elif DuplicateGeneCompressor._log_fine():
                    LOG.debug(f"found and removed duplicate gene reactions: "
                             f"{work_record.get_reaction_names(group)}")
                
                # Log details for kept reaction
                if DuplicateGeneCompressor._log_finer():
                    work_record.log_reaction_details(logging.DEBUG, "   [+] r=1: ", kept_reaction_idx)
                
                # Mark slave reactions for removal and log details
                for i in range(1, len(group)):
                    reac_idx = group[i]
                    if DuplicateGeneCompressor._log_finer():
                        prefix = f"   [-] r={ratios[reac_idx]}: "
                        work_record.log_reaction_details(logging.DEBUG, prefix, reac_idx)
                    reactions_to_remove.add(reac_idx)
                
                # Update kept reaction's reversibility
                reversible[kept_reaction_idx] = kept_reversible
                
                # Track group for statistics
                work_record.groups.append(group)  # Add to work record's groups
                work_record.stats.inc_duplicate_gene_reactions(len(group))
            else:
                LOG.debug(f"ignoring weak duplicate gene reactions (not all have same directionality): "
                         f"{work_record.get_reaction_names(group)}")
        
        # Remove marked reactions
        if reactions_to_remove:
            work_record.remove_reactions_by_indices(reactions_to_remove)
            return True
        
        return False
    
    @staticmethod
    def _is_zero(big_integer_val) -> bool:
        """Check if BigInteger value represents zero"""
        if hasattr(big_integer_val, 'signum'):
            return big_integer_val.signum() == 0
        else:
            return big_integer_val == 0
    
    @staticmethod
    def _log_fine() -> bool:
        """Check if FINE level logging is enabled"""
        return LOG.isEnabledFor(logging.DEBUG)
    
    @staticmethod
    def _log_finer() -> bool:
        """Check if FINER level logging is enabled"""
        return LOG.isEnabledFor(logging.DEBUG)