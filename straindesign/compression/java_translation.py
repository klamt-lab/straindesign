#!/usr/bin/env python3
"""
Line-by-line translation of Java StoichMatrixCompressor.

This is a faithful Python translation of the Java code to ensure exact behavior.
"""

from fractions import Fraction
from typing import List, Dict, Set, Optional, Tuple
import copy
import logging

from .rational_math import RationalMath, ZERO, ONE
from .rational_matrix import RationalMatrix


class Size:
    """Size tracker for metabolites and reactions (exact Java translation)."""
    
    def __init__(self, metas: int, reacs: int):
        self.metas = metas
        self.reacs = reacs


class CompressionStatistics:
    """Compression statistics (exact Java translation)."""
    
    def __init__(self):
        self.initial_metabolites = 0
        self.initial_reactions = 0
        self.zero_flux_reactions = 0
        self.contradicting_reactions = 0
        self.coupled_reactions = 0
        self.unique_flow_reactions = 0
        self.dead_end_metabolite_reactions = 0
        self.unused_metabolites = 0
        self.compression_iterations = 0
    
    def inc_compression_iteration(self) -> int:
        """Increment and return compression iteration count."""
        self.compression_iterations += 1
        return self.compression_iterations - 1  # Java returns old value
    
    def inc_zero_flux_reactions(self, count: int = 1):
        self.zero_flux_reactions += count
    
    def inc_contradicting_reactions(self, count: int = 1):
        self.contradicting_reactions += count
    
    def inc_coupled_reactions(self, count: int = 1):
        self.coupled_reactions += count
    
    def inc_unique_flow_reactions(self, count: int = 1):
        self.unique_flow_reactions += count
    
    def inc_dead_end_metabolite_reactions(self, count: int):
        self.dead_end_metabolite_reactions += count
    
    def inc_unused_metabolite(self, count: int = 1):
        self.unused_metabolites += count


class BitSet:
    """Python equivalent of Java BitSet."""
    
    def __init__(self):
        self._bits = set()
    
    def set(self, bit: int):
        self._bits.add(bit)
    
    def get(self, bit: int) -> bool:
        return bit in self._bits
    
    def clear(self, bit: int):
        self._bits.discard(bit)
    
    def next_set_bit(self, from_index: int) -> int:
        """Return next set bit >= from_index, or -1 if none."""
        for bit in sorted(self._bits):
            if bit >= from_index:
                return bit
        return -1
    
    def clone(self) -> 'BitSet':
        new_set = BitSet()
        new_set._bits = self._bits.copy()
        return new_set
    
    def is_empty(self) -> bool:
        return len(self._bits) == 0


def identity(size: int) -> RationalMatrix:
    """Create identity matrix (exact Java translation)."""
    id_matrix = RationalMatrix(size, size)
    for piv in range(size):
        id_matrix.set_value_at(piv, piv, ONE)
    return id_matrix


def cancel(matrix: RationalMatrix) -> RationalMatrix:
    """Create copy of matrix (Java's cancel just copies and reduces)."""
    mx = matrix.copy()
    # Java calls reduce() here, but our matrices don't need explicit reduction
    return mx


def create_sub_stoich(pre: RationalMatrix, stoich: RationalMatrix, 
                     post: RationalMatrix, reversible: List[bool], size: Size) -> RationalMatrix:
    """Create submatrix for active region only."""
    result = RationalMatrix(size.metas, size.reacs)
    for i in range(size.metas):
        for j in range(size.reacs):
            val = stoich.get_value_at(i, j)
            if not RationalMath.is_zero(val):
                result.set_value_at(i, j, val)
    return result


def swap_array_elements(arr: List, i: int, j: int):
    """Swap two elements in a list (Java Arrays.swap equivalent)."""
    arr[i], arr[j] = arr[j], arr[i]


def is_zero(value) -> bool:
    """Check if value is zero (Java isZero equivalent)."""
    return RationalMath.is_zero(value)


class CompressionRecord:
    """Base class for compression data (exact Java translation)."""
    
    def __init__(self, pre: RationalMatrix, stoich: RationalMatrix, 
                 post: RationalMatrix, reversible: List[bool]):
        self.pre = pre
        self.cmp = stoich
        self.post = post
        self.reversible = reversible


class WorkRecord(CompressionRecord):
    """
    Line-by-line translation of Java WorkRecord class.
    
    This exactly matches the Java behavior including:
    - Matrix dimensions and initialization 
    - Swap-based removal operations
    - Size tracking
    """
    
    def __init__(self, rd_stoich: RationalMatrix, reversible: List[bool], 
                 meta_names: List[str], reac_names: List[str]):
        """Constructor for initial work record (line 595-602)."""
        # Line 596: super(identity(rdStoich.getRowCount()), cancel(rdStoich), identity(rdStoich.getColumnCount()), reversible.clone());
        row_count = rd_stoich.get_row_count()
        col_count = rd_stoich.get_column_count()
        
        super().__init__(
            identity(row_count),           # pre matrix
            cancel(rd_stoich),            # cmp matrix (copied)
            identity(col_count),          # post matrix
            reversible.copy()             # reversible (cloned)
        )
        
        # Line 597: this.stats = new CompressionStatistics();
        self.stats = CompressionStatistics()
        
        # Line 599: this.size = new Size(cmp.getRowCount(), cmp.getColumnCount());
        self.size = Size(self.cmp.get_row_count(), self.cmp.get_column_count())
        
        # Lines 600-601: store names
        self.meta_names = meta_names
        self.reac_names = reac_names
    
    def __init_copy__(self, work_record: 'WorkRecord'):
        """Constructor cloning an existing work record (line 604-610)."""
        super().__init__(work_record.pre, work_record.cmp, work_record.post, work_record.reversible)
        self.stats = work_record.stats
        self.size = work_record.size  
        self.meta_names = work_record.meta_names
        self.reac_names = work_record.reac_names
    
    def get_truncated(self) -> CompressionRecord:
        """Line-by-line translation of getTruncated (lines 611-625)."""
        # Line 612: final BigIntegerRationalMatrix cmpTrunc = createSubStoich(pre, cmp, post, reversible, size);
        cmp_trunc = create_sub_stoich(self.pre, self.cmp, self.post, self.reversible, self.size)
        
        # Lines 613-616: dimensions
        m = self.cmp.get_row_count()
        r = self.cmp.get_column_count()
        mc = cmp_trunc.get_row_count()
        rc = cmp_trunc.get_column_count()
        
        # Lines 617-618: copy reversible array
        rev_trunc = [False] * rc
        for i in range(rc):
            rev_trunc[i] = self.reversible[i]
        
        # Lines 619-624: return truncated record
        return CompressionRecord(
            self.pre.sub_matrix(0, mc, 0, m),       # pre submatrix
            cmp_trunc,                              # compressed matrix
            self.post.sub_matrix(0, r, 0, rc),     # post submatrix  
            rev_trunc                               # truncated reversible
        )
    
    def remove_reaction(self, reac: int):
        """
        Line-by-line translation of removeReaction (lines 631-642).
        
        Removes the specified reaction by swapping to end and decrementing size.
        """
        # Lines 632-634: zero out the reaction column
        for meta in range(self.size.metas):
            self.cmp.set_value_at(meta, reac, ZERO)
        
        # Line 635: decrement size
        self.size.reacs -= 1
        
        # Line 636-641: swap to end if not already there
        if reac != self.size.reacs:
            # Line 637: post.swapColumns(reac, size.reacs)
            self.post.swap_columns(reac, self.size.reacs)
            # Line 638: cmp.swapColumns(reac, size.reacs)
            self.cmp.swap_columns(reac, self.size.reacs)
            # Line 639: Arrays.swap(reversible, reac, size.reacs)
            swap_array_elements(self.reversible, reac, self.size.reacs)
            # Line 640: Arrays.swap(reacNames, reac, size.reacs)
            swap_array_elements(self.reac_names, reac, self.size.reacs)
    
    def remove_reactions_by_name(self, suppressed_reactions: Optional[Set[str]]) -> bool:
        """Line-by-line translation of removeReactions(Set<String>) (lines 649-668)."""
        # Lines 650-652: check for empty set
        if suppressed_reactions is None or len(suppressed_reactions) == 0:
            return False
        
        # Line 653: create BitSet
        index_set = BitSet()
        
        # Lines 654-665: find indices
        for reac in suppressed_reactions:
            index = -1
            for i in range(len(self.reac_names)):
                if self.reac_names[i] == reac:
                    index = i
                    break
            if index < 0:
                raise ValueError(f"no such reaction: {reac}")
            index_set.set(index)
        
        # Line 667: remove by indices  
        self.remove_reactions(index_set)
        return True
    
    def remove_reactions(self, reactions_to_remove: BitSet):
        """Line-by-line translation of removeReactions(BitSet) (lines 675-692)."""
        # Line 676: clone BitSet
        to_remove = reactions_to_remove.clone()
        
        # Line 678: iterate through set bits
        reac = to_remove.next_set_bit(0)
        while reac >= 0:
            # Line 679: remove the reaction
            self.remove_reaction(reac)
            
            # Lines 680-690: handle swapping logic
            if reac != self.size.reacs and to_remove.get(self.size.reacs):
                # Line 682: clear the swapped position
                to_remove.clear(self.size.reacs)
                # Continue with same reac (now has different reaction)
            else:
                # Lines 687-689: move to next reaction
                to_remove.clear(reac)
                reac = to_remove.next_set_bit(reac + 1)
    
    def remove_metabolite(self, meta: int, set_stoich_to_zero: bool = True):
        """Line-by-line translation of removeMetabolite (lines 698-710)."""
        # Lines 699-703: optionally zero out metabolite row
        if set_stoich_to_zero:
            for reac in range(self.size.reacs):
                self.cmp.set_value_at(meta, reac, ZERO)
        
        # Line 704: decrement size
        self.size.metas -= 1
        
        # Lines 705-709: swap to end if not already there
        if meta != self.size.metas:
            # Line 706: pre.swapRows(meta, size.metas)
            self.pre.swap_rows(meta, self.size.metas)
            # Line 707: cmp.swapRows(meta, size.metas) 
            self.cmp.swap_rows(meta, self.size.metas)
            # Line 708: Arrays.swap(metaNames, meta, size.metas)
            swap_array_elements(self.meta_names, meta, self.size.metas)
    
    def remove_unused_metabolites(self) -> bool:
        """Line-by-line translation of removeUnusedMetabolites (lines 718-739)."""
        # Line 719: store original count
        orig_cnt = self.size.metas
        
        # Line 722: iterate through metabolites
        meta = 0
        while meta < self.size.metas:
            # Lines 723-726: check if metabolite is used
            any_nonzero = False
            for reac in range(self.size.reacs):
                if not is_zero(self.cmp.get_numerator_at(meta, reac)):
                    any_nonzero = True
                    break
            
            # Lines 727-737: remove if unused
            if not any_nonzero:
                self.remove_metabolite(meta, False)  # don't set to zero, already zero
                self.stats.inc_unused_metabolite()
                # Don't increment meta - new metabolite was swapped here
            else:
                meta += 1
        
        # Return true if any metabolites were removed
        return self.size.metas < orig_cnt


class IntArray:
    """Python equivalent of Java IntArray (from util package)."""
    
    def __init__(self):
        self._values = []
    
    def add(self, value: int):
        self._values.append(value)
    
    def get(self, index: int) -> int:
        return self._values[index]
    
    def first(self) -> int:
        return self._values[0] if self._values else -1
    
    def length(self) -> int:
        return len(self._values)


def negate_column(mx: RationalMatrix, col: int):
    """Line-by-line translation of negateColumn."""
    rows = mx.get_row_count()
    for row in range(rows):
        val = mx.get_value_at(row, col)
        mx.set_value_at(row, col, -val)


def add_column_multiple_to_with_separate_multipliers(mx: RationalMatrix, dst_col: int, 
                                                   dst_mul: Fraction, src_col: int, src_mul: Fraction):
    """Line-by-line translation of addColumnMultipleTo with separate multipliers (lines 848-858)."""
    rows = mx.get_row_count()
    for row in range(rows):
        if (mx.get_signum_at(row, src_col) != 0 or mx.get_signum_at(row, dst_col) != 0):
            # Line 852: add = mx.getBigFractionValueAt(row, srcCol).multiply(srcMul)
            add = mx.get_value_at(row, src_col) * src_mul
            # Lines 853-854: multiply dst by dst_mul, then add
            dst_val = mx.get_value_at(row, dst_col) * dst_mul
            mx.set_value_at(row, dst_col, dst_val + add)


def add_column_multiple_to(mx: RationalMatrix, dst_col: int, src_col: int, dst_to_src_ratio: Fraction):
    """Line-by-line translation of addColumnMultipleTo with ratio (lines 859-868)."""
    rows = mx.get_row_count()
    for row in range(rows):
        if mx.get_signum_at(row, src_col) != 0:
            # Line 863: add = mx.getBigFractionValueAt(row, srcCol).divide(dstToSrcRatio)
            add = mx.get_value_at(row, src_col) / dst_to_src_ratio
            # Line 864: mx.add(row, dstCol, add.getNumerator(), add.getDenominator())
            current_val = mx.get_value_at(row, dst_col)
            mx.set_value_at(row, dst_col, current_val + add)


class NullspaceRecord(WorkRecord):
    """
    Line-by-line translation of Java NullspaceRecord (inner class in StoichMatrixCompressor).
    """
    
    def __init__(self, work_record: WorkRecord):
        """Clone constructor matching Java pattern."""
        # Call the copy constructor pattern
        super().__init__(work_record.cmp, work_record.reversible, 
                        work_record.meta_names, work_record.reac_names)
        
        # Copy all fields from work_record
        self.pre = work_record.pre
        self.cmp = work_record.cmp
        self.post = work_record.post
        self.reversible = work_record.reversible
        self.stats = work_record.stats
        self.size = work_record.size
        self.meta_names = work_record.meta_names
        self.reac_names = work_record.reac_names
        
        # Compute kernel - this is added in the Java NullspaceRecord subclass
        self.kernel = self._compute_kernel()
    
    def _compute_kernel(self) -> RationalMatrix:
        """Compute the nullspace kernel of the stoichiometric matrix."""
        # GaussElimination should already be in globals from the module loading
        
        # Get the active stoichiometric matrix
        active_stoich = create_sub_stoich(self.pre, self.cmp, self.post, self.reversible, self.size)
        
        # Compute nullspace of S (not S^T)
        # This gives vectors in reaction space (reactions × null_space_dimension)
        return GaussElimination.nullspace(active_stoich)
    
    def remove_reaction(self, reac: int):
        """Override to also handle kernel matrix."""
        # Call parent method
        super().remove_reaction(reac)
        
        # Also swap kernel row (line 828 in Java: kernel.swapRows(reac, size.reacs))
        if self.kernel and reac != self.size.reacs:
            self.kernel.swap_rows(reac, self.size.reacs)
    
    def remove_reactions(self, reactions_to_remove: BitSet):
        """Remove reactions from nullspace record."""
        super().remove_reactions(reactions_to_remove)


# Compression method functions (line-by-line translation of Java methods)

def nullspace_zero_flux_reactions(nullspace_record: NullspaceRecord) -> bool:
    """Line-by-line translation of nullspaceZeroFluxReactions (lines 218-246)."""
    # Lines 220-221: aliasing
    size = nullspace_record.size
    kernel = nullspace_record.kernel
    
    # Line 223: boolean anyZeroFlux = false
    any_zero_flux = False
    
    # Line 224: final int cols = kernel.getColumnCount()
    cols = kernel.get_column_count()
    
    # Line 225: int reac = 0
    reac = 0
    
    # Line 226: while (reac < size.reacs)
    while reac < size.reacs:
        # Line 227: boolean allZero = true
        all_zero = True
        
        # Line 228: for (int col = 0; col < cols; col++)
        for col in range(cols):
            # Line 229: if (!isZero(kernel.getBigIntegerNumeratorAt(reac, col)))
            if not is_zero(kernel.get_numerator_at(reac, col)):
                # Line 230: allZero = false; break
                all_zero = False
                break
        
        # Line 233: if (allZero)
        if all_zero:
            # Lines 238-241: remove reaction and increment stats
            nullspace_record.remove_reaction(reac)
            nullspace_record.stats.inc_zero_flux_reactions()
            # Line 242: anyZeroFlux = true
            any_zero_flux = True
            # Don't increment reac - new reaction was swapped here
        else:
            # Line 244: reac++
            reac += 1
    
    # Line 245: return anyZeroFlux
    return any_zero_flux


def nullspace_coupled_reactions(nullspace_record: NullspaceRecord, incl_compression: bool,
                               do_contradicting: bool = True, do_combine: bool = True) -> bool:
    """Line-by-line translation of nullspaceCoupledReactions (lines 247-395)."""
    # Lines 252-257: aliasing
    kernel = nullspace_record.kernel
    stoich = nullspace_record.cmp
    post = nullspace_record.post
    reversible = nullspace_record.reversible
    size = nullspace_record.size
    
    # Lines 260-261: start
    cols = kernel.get_column_count()
    reacs = size.reacs
    
    # Line 262: List<IntArray> groups = new ArrayList<IntArray>()
    groups = []  # type: List[IntArray]
    
    # Line 263: BigFraction[] ratios = new BigFraction[reacs]
    ratios = [None] * reacs  # type: List[Optional[Fraction]]
    
    # Line 264: for (int reacA = 0; reacA < reacs; reacA++)
    for reac_a in range(reacs):
        # Line 265: if (ratios[reacA] == null)
        if ratios[reac_a] is None:
            # Line 266: IntArray group = null
            group = None
            
            # Line 267: for (int reacB = reacA + 1; reacB < reacs; reacB++)
            for reac_b in range(reac_a + 1, reacs):
                # Line 268: BigFraction ratio = null
                ratio = None
                
                # Line 269: for (int col = 0; col < cols; col++)
                for col in range(cols):
                    # Lines 270-271: check if zero
                    is_zero_a = is_zero(kernel.get_numerator_at(reac_a, col))
                    is_zero_b = is_zero(kernel.get_numerator_at(reac_b, col))
                    
                    # Line 272: if (isZeroA != isZeroB)
                    if is_zero_a != is_zero_b:
                        # Line 273-274: ratio = BigFraction.ZERO; break
                        ratio = ZERO
                        break
                    # Line 276: else if (!isZeroA)
                    elif not is_zero_a:
                        # Lines 277-278: get values
                        val_a = kernel.get_value_at(reac_a, col)
                        val_b = kernel.get_value_at(reac_b, col)
                        
                        # Line 279: BigFraction curRatio = valA.divide(valB).reduce()
                        cur_ratio = val_a / val_b
                        
                        # Line 280: if (ratio == null)
                        if ratio is None:
                            # Line 281: ratio = curRatio
                            ratio = cur_ratio
                        # Line 283: else if (ratio.compareTo(curRatio) != 0)
                        elif ratio != cur_ratio:
                            # Line 284-285: ratio = BigFraction.ZERO; break
                            ratio = ZERO
                            break
                
                # Line 289: if (ratio == null)
                if ratio is None:
                    # Line 290: throw new RuntimeException("no zero rows expected here")
                    raise RuntimeError("no zero rows expected here")
                # Line 292: else if (!isZero(ratio.getNumerator()))
                elif not RationalMath.is_zero(ratio):
                    # Line 294: ratios[reacB] = ratio
                    ratios[reac_b] = ratio
                    
                    # Line 295: if (group == null)
                    if group is None:
                        # Lines 296-297: group = new IntArray(); group.add(reacA)
                        group = IntArray()
                        group.add(reac_a)
                    
                    # Line 299: group.add(reacB)
                    group.add(reac_b)
            
            # Line 302: if (group != null)
            if group is not None:
                # Line 303: groups.add(group)
                groups.append(group)
    
    # Line 307: BitSet toRemove = new BitSet()
    to_remove = BitSet()
    
    # Line 308: for (IntArray grp : groups)
    for grp in groups:
        # Lines 312-321: check consistency
        forward = False
        while True:
            # Line 314: forward = !forward
            forward = not forward
            
            # Line 315: allOk = forward || reversible[grp.first()]
            all_ok = forward or reversible[grp.first()]
            
            # Line 316: for (int i = 1; i < grp.length() && allOk; i++)
            i = 1
            while i < grp.length() and all_ok:
                # Line 317: int reac = grp.get(i)
                reac = grp.get(i)
                
                # Line 318: allOk &= (forward == ratios[reac].signum() > 0) || reversible[reac]
                ratio_positive = RationalMath.signum(ratios[reac]) > 0
                all_ok = all_ok and ((forward == ratio_positive) or reversible[reac])
                i += 1
            
            # Line 321: while (forward && !allOk)
            if not forward or all_ok:
                break
        
        # Line 322: if (!allOk)
        if not all_ok:
            # Line 323: if (doCon)
            if do_contradicting:
                # Lines 327-336: remove inconsistently coupled reactions
                for i in range(grp.length()):
                    reac = grp.get(i)
                    to_remove.set(reac)
                    nullspace_record.stats.inc_contradicting_reactions()
        else:
            # Line 343: if (doCom && inclCompression)
            if do_combine and incl_compression:
                # Line 350: int masterReac = grp.first()
                master_reac = grp.first()
                
                # Line 355: if (!forward)
                if not forward:
                    # Lines 357-358: negate the column
                    negate_column(stoich, master_reac)
                    negate_column(post, master_reac)
                
                # Line 360: for (int i = 1; i < grp.length(); i++)
                for i in range(1, grp.length()):
                    # Line 361: int reac = grp.get(i)
                    reac = grp.get(i)
                    
                    # Line 362: BigFraction ratio = forward ? ratios[reac] : ratios[reac].negate()
                    ratio = ratios[reac] if forward else -ratios[reac]
                    
                    # Lines 367-368: add column multiple
                    add_column_multiple_to(stoich, master_reac, reac, ratio)
                    add_column_multiple_to(post, master_reac, reac, ratio)
                    
                    # Line 369: reversible[masterReac] &= reversible[reac]
                    reversible[master_reac] = reversible[master_reac] and reversible[reac]
                    
                    # Line 370: toRemove.set(reac)
                    to_remove.set(reac)
                    
                    # Line 371: nullspaceRecord.stats.incCoupledReactions()
                    nullspace_record.stats.inc_coupled_reactions()
                
                # Lines 377-383: check for all-zero column (Java bug - condition wrong)
                all_zero = True
                for meta in range(size.metas):
                    if not is_zero(stoich.get_numerator_at(meta, master_reac)):
                        all_zero = False
                        break
                
                # Line 381: if (allZero) 
                if all_zero:
                    # Line 382: throw new RuntimeException
                    # Java throws exception, but we could handle this better
                    # For now, match Java behavior exactly
                    raise RuntimeError(f"all entries found 0 for a reaction after merging: {master_reac}")
    
    # Line 393: nullspaceRecord.removeReactions(toRemove)
    nullspace_record.remove_reactions(to_remove)
    
    # Line 394: return !toRemove.isEmpty()
    return not to_remove.is_empty()