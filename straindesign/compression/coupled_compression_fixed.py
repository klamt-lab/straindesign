#!/usr/bin/env python3
"""
Coupled compression methods - CORRECTED version matching Java original.

This module implements the exact logic from Java StoichMatrixCompressor
for nullspace-based compression methods.
"""

from fractions import Fraction
from typing import List, Dict, Set, Tuple, Optional
import logging

from .rational_math import RationalMath, ZERO, ONE
from .rational_matrix import RationalMatrix, GaussElimination
from .compression_structures import WorkRecord, NullspaceRecord, CompressionStatistics


class BitSet:
    """Simple BitSet implementation to match Java's BitSet usage."""
    
    def __init__(self, size: int = 64):
        self._bits = set()
        
    def set(self, bit: int):
        """Set a bit to true."""
        self._bits.add(bit)
        
    def clear(self, bit: int):
        """Set a bit to false."""
        self._bits.discard(bit)
        
    def get(self, bit: int) -> bool:
        """Check if a bit is set."""
        return bit in self._bits
        
    def cardinality(self) -> int:
        """Return number of set bits."""
        return len(self._bits)
        
    def isEmpty(self) -> bool:
        """Check if no bits are set."""
        return len(self._bits) == 0
        
    def nextSetBit(self, start: int) -> int:
        """Find next set bit from start position."""
        for bit in sorted(self._bits):
            if bit >= start:
                return bit
        return -1
        
    def __iter__(self):
        """Iterate over set bits."""
        return iter(sorted(self._bits))


class IntArray:
    """Simple IntArray implementation to match Java's IntArray usage."""
    
    def __init__(self):
        self._values = []
        
    def add(self, value: int):
        """Add a value."""
        self._values.append(value)
        
    def get(self, index: int) -> int:
        """Get value at index."""
        return self._values[index]
        
    def first(self) -> int:
        """Get first value."""
        return self._values[0] if self._values else -1
        
    def length(self) -> int:
        """Get length."""
        return len(self._values)
        
    def __iter__(self):
        """Iterate over values."""
        return iter(self._values)


class CoupledCompressionFixed:
    """
    Exact port of Java StoichMatrixCompressor nullspace methods.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def compress_nullspace(self, work_record: WorkRecord,
                          do_zero: bool = True,
                          do_contradicting: bool = True,
                          do_combine: bool = True,
                          include_compression: bool = True) -> bool:
        """
        Main nullspace compression method - exact port of Java nullspace().
        """
        nullspace_record = NullspaceRecord(work_record)
        any_compressed = False
        
        if do_zero:
            any_compressed |= self._nullspace_zero_flux_reactions(nullspace_record)
            
        if do_contradicting or do_combine:
            any_compressed |= self._nullspace_coupled_reactions(
                nullspace_record, do_contradicting, do_combine, include_compression
            )
        
        # Update work record
        work_record.cmp = nullspace_record.cmp
        work_record.pre = nullspace_record.pre  
        work_record.post = nullspace_record.post
        work_record.meta_names = nullspace_record.meta_names
        work_record.reac_names = nullspace_record.reac_names
        work_record.reversible = nullspace_record.reversible
        work_record.size = nullspace_record.size
        work_record.stats = nullspace_record.stats
        
        if any_compressed:
            work_record.remove_unused_metabolites()
            
        return any_compressed
    
    def _nullspace_zero_flux_reactions(self, nullspace_record: NullspaceRecord) -> bool:
        """
        Exact port of Java nullspaceZeroFluxReactions().
        """
        kernel = nullspace_record.kernel
        any_zero_flux = False
        cols = kernel.get_column_count()
        reac = 0
        
        while reac < nullspace_record.size.reacs:
            all_zero = True
            for col in range(cols):
                # Java: isZero(kernel.getBigIntegerNumeratorAt(reac, col))
                if not RationalMath.is_zero(kernel.get_numerator_at(reac, col)):
                    all_zero = False
                    break
                    
            if all_zero:
                self.logger.debug(f"Found and removed zero flux reaction: {nullspace_record.reac_names[reac]}")
                any_zero_flux = True
                nullspace_record.remove_reaction(reac)
                nullspace_record.stats.inc_zero_flux_reactions()
            else:
                reac += 1
                
        return any_zero_flux
    
    def _nullspace_coupled_reactions(self, nullspace_record: NullspaceRecord,
                                   do_contradicting: bool, do_combine: bool,
                                   include_compression: bool) -> bool:
        """
        Exact port of Java nullspaceCoupledReactions().
        """
        kernel = nullspace_record.kernel
        stoich = nullspace_record.cmp
        post = nullspace_record.post
        reversible = nullspace_record.reversible
        size = nullspace_record.size
        
        cols = kernel.get_column_count()
        reacs = size.reacs
        
        # Find coupled groups - exact Java algorithm
        groups = []  # List[IntArray]
        ratios = [None] * reacs  # List[Optional[Fraction]] - ratios[reacB] = reacA/reacB
        
        for reac_a in range(reacs):
            if ratios[reac_a] is None:
                group = None
                
                for reac_b in range(reac_a + 1, reacs):
                    ratio = None  # reac_a / reac_b
                    
                    for col in range(cols):
                        # Java: isZero(kernel.getBigIntegerNumeratorAt(...))
                        is_zero_a = RationalMath.is_zero(kernel.get_numerator_at(reac_a, col))
                        is_zero_b = RationalMath.is_zero(kernel.get_numerator_at(reac_b, col))
                        
                        if is_zero_a != is_zero_b:
                            ratio = ZERO
                            break
                        elif not is_zero_a:
                            val_a = kernel.get_value_at(reac_a, col)
                            val_b = kernel.get_value_at(reac_b, col)
                            cur_ratio = (val_a / val_b)  # Fraction auto-reduces
                            
                            if ratio is None:
                                ratio = cur_ratio
                            elif ratio != cur_ratio:
                                ratio = ZERO
                                break
                    
                    if ratio is None:
                        raise RuntimeError("No zero rows expected here")
                    elif not RationalMath.is_zero(ratio):
                        # Found coupled reactions
                        ratios[reac_b] = ratio
                        if group is None:
                            group = IntArray()
                            group.add(reac_a)
                        group.add(reac_b)
                
                if group is not None:
                    groups.append(group)
        
        # Process groups - exact Java algorithm
        to_remove = BitSet()
        
        for grp in groups:
            # Check consistency with reversibilities - exact Java logic
            all_ok = False
            forward = False
            
            while True:
                forward = not forward
                all_ok = forward or reversible[grp.first()]
                
                for i in range(1, grp.length()):
                    if not all_ok:
                        break
                    reac = grp.get(i)
                    # Java: (forward == ratios[reac].signum() > 0) || reversible[reac]
                    ratio_positive = RationalMath.signum(ratios[reac]) > 0
                    all_ok &= (forward == ratio_positive) or reversible[reac]
                
                if not forward or all_ok:
                    break
            
            if not all_ok:
                if do_contradicting:
                    self.logger.debug(f"Found and removed inconsistently coupled reactions: {[nullspace_record.reac_names[grp.get(i)] for i in range(grp.length())]}")
                    
                    for i in range(grp.length()):
                        reac = grp.get(i)
                        to_remove.set(reac)
                        nullspace_record.stats.inc_contradicting_reactions()
            else:
                if do_combine and include_compression:
                    # Combine coupled reactions - exact Java logic
                    self.logger.debug(f"Found and combined coupled reactions: {[nullspace_record.reac_names[grp.get(i)] for i in range(grp.length())]}")
                    
                    master_reac = grp.first()
                    
                    if not forward:
                        # Negate the master column - exact Java negateColumn()
                        self._negate_column(stoich, master_reac)
                        self._negate_column(post, master_reac)
                    
                    for i in range(1, grp.length()):
                        reac = grp.get(i)
                        ratio = ratios[reac] if forward else -ratios[reac]
                        
                        # Java: addColumnMultipleTo(stoich, masterReac, reac, ratio)
                        self._add_column_multiple_to(stoich, master_reac, reac, ratio)
                        self._add_column_multiple_to(post, master_reac, reac, ratio)
                        
                        reversible[master_reac] &= reversible[reac]
                        to_remove.set(reac)
                        nullspace_record.stats.inc_coupled_reactions()
                    
                    # Check for all-zero column after merging
                    all_zero = True
                    for meta in range(size.metas):
                        if not RationalMath.is_zero(stoich.get_numerator_at(meta, master_reac)):
                            all_zero = False
                            break
                    
                    if all_zero:
                        raise RuntimeError(f"All entries found 0 for reaction after merging: {master_reac}")
        
        # Remove reactions
        reactions_to_remove = set(to_remove)
        if reactions_to_remove:
            nullspace_record.remove_reactions(reactions_to_remove)
        
        return not to_remove.isEmpty()
    
    def _negate_column(self, matrix: RationalMatrix, col: int):
        """
        Exact port of Java negateColumn().
        """
        rows = matrix.get_row_count()
        for row in range(rows):
            val = matrix.get_value_at(row, col)
            matrix.set_value_at(row, col, -val)
    
    def _add_column_multiple_to(self, matrix: RationalMatrix, 
                              dst_col: int, src_col: int, dst_to_src_ratio: Fraction):
        """
        Exact port of Java addColumnMultipleTo(mx, dstCol, srcCol, dstToSrcRatio).
        
        Java logic: dstCol = dstCol + srcCol / dstToSrcRatio
        """
        rows = matrix.get_row_count()
        for row in range(rows):
            src_signum = matrix.get_signum_at(row, src_col)
            if src_signum != 0:
                # Java: add = mx.getBigFractionValueAt(row, srcCol).divide(dstToSrcRatio);
                src_val = matrix.get_value_at(row, src_col)
                dst_val = matrix.get_value_at(row, dst_col)
                add_val = src_val / dst_to_src_ratio
                
                # Java: mx.add(row, dstCol, add.getNumerator(), add.getDenominator());
                new_val = dst_val + add_val
                matrix.set_value_at(row, dst_col, new_val)


def test_fixed_implementation():
    """Test the fixed implementation against specific cases."""
    
    # Test case 1: Simple coupled reactions A->B, B->C
    # These should be coupled 1:1 in steady state
    stoich = RationalMatrix(3, 2)
    stoich.set_value_at(0, 0, Fraction(-1, 1))  # A consumed in R1
    stoich.set_value_at(1, 0, Fraction(1, 1))   # B produced in R1  
    stoich.set_value_at(1, 1, Fraction(-1, 1))  # B consumed in R2
    stoich.set_value_at(2, 1, Fraction(1, 1))   # C produced in R2
    
    work = WorkRecord(
        stoich,
        [True, True],  # Both reversible
        ["A", "B", "C"],
        ["R1_A_to_B", "R2_B_to_C"]
    )
    
    print("Before compression:")
    print(f"Reactions: {work.size.reacs}")
    print(f"Names: {work.reac_names}")
    print("Stoichiometric matrix:")
    for i in range(work.cmp.get_row_count()):
        row = [float(work.cmp.get_value_at(i, j)) for j in range(work.cmp.get_column_count())]
        print(f"  {work.meta_names[i]}: {row}")
    
    # Apply compression
    compressor = CoupledCompressionFixed()
    result = compressor.compress_nullspace(work, do_combine=True, include_compression=True)
    
    print(f"\nCompression result: {result}")
    print(f"Reactions after: {work.size.reacs}")
    print(f"Names after: {work.reac_names}")
    print("Stoichiometric matrix after:")
    for i in range(work.cmp.get_row_count()):
        row = [float(work.cmp.get_value_at(i, j)) for j in range(work.cmp.get_column_count())]
        print(f"  {work.meta_names[i]}: {row}")
    
    print(f"Statistics - Coupled reactions: {work.stats.coupled_reactions}")
    
    return result


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    
    print("Testing Fixed Coupled Compression Implementation")
    print("=" * 60)
    
    test_fixed_implementation()