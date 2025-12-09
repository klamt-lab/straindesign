"""
EFMTool Work Record

Python port of ch.javasoft.metabolic.compress.StoichMatrixCompressor.WorkRecord
Mutable record extending CompressionRecord with working data for compression algorithms.

From Java source: efmtool_source/ch/javasoft/metabolic/compress/StoichMatrixCompressor.java (inner class)
Ported line-by-line for exact compatibility
"""

from typing import List, Optional, Set
from .compression_record import CompressionRecord
from .compression_statistics import CompressionStatistics
from ..math.readable_bigint_rational_matrix import ReadableBigIntegerRationalMatrix
from ..math.bigint_rational_matrix import BigIntegerRationalMatrix
from ..math.default_bigint_rational_matrix import DefaultBigIntegerRationalMatrix
from ..math.big_fraction import BigFraction


class Size:
    """
    Mutable counter for metabolite/reaction size tracking during compression.
    
    This class tracks the current "active" size of the matrices during compression,
    which may be smaller than the physical matrix dimensions as reactions/metabolites
    are removed and moved to the end.
    """
    
    def __init__(self, metas: int, reacs: int):
        """
        Initialize size tracking.
        
        Args:
            metas: Number of active metabolites
            reacs: Number of active reactions
        """
        self.metas = metas
        self.reacs = reacs
    
    def clone(self) -> 'Size':
        """
        Create a copy of this size object.
        
        Returns:
            New Size instance with same values
        """
        return Size(self.metas, self.reacs)
    
    def __str__(self) -> str:
        """String representation matching Java format"""
        return f"[metas={self.metas}, reacs={self.reacs}]"
    
    def __repr__(self) -> str:
        """Detailed representation"""
        return f"Size(metas={self.metas}, reacs={self.reacs})"
    
    def __eq__(self, other) -> bool:
        """Check equality of size objects"""
        if not isinstance(other, Size):
            return False
        return self.metas == other.metas and self.reacs == other.reacs
    
    def __hash__(self) -> int:
        """Hash for Size objects"""
        return hash((self.metas, self.reacs))


class WorkRecord(CompressionRecord):
    """
    Mutable record extending CompressionRecord with working data for compression algorithms.
    
    This class contains all the data needed during compression processing:
    - All matrices and reversibility from CompressionRecord
    - Statistics tracking for compression operations
    - Metabolite and reaction names for debugging/logging
    - Size tracking for active matrix dimensions
    
    The matrices in this record are mutable and modified during compression.
    """
    
    def __init__(self, rd_stoich: ReadableBigIntegerRationalMatrix, 
                 reversible: List[bool], 
                 meta_names: List[str], 
                 reac_names: List[str]):
        """
        Constructor for initial work record.
        
        Args:
            rd_stoich: Original stoichiometric matrix (read-only)
            reversible: Reversibility flags for original reactions
            meta_names: Names of metabolites
            reac_names: Names of reactions
        """
        # Create identity matrices and cancel (reduce) the stoichiometric matrix
        rows = rd_stoich.get_row_count()
        cols = rd_stoich.get_column_count()
        
        pre_identity = self._create_identity_matrix(rows)
        post_identity = self._create_identity_matrix(cols)
        cmp_reduced = self._cancel_matrix(rd_stoich)
        
        # Initialize parent with identity matrices and reduced stoichiometric matrix
        super().__init__(pre_identity, cmp_reduced, post_identity, reversible.copy())
        
        # Initialize working data
        self.stats = CompressionStatistics()
        self.stats.inc_compression_iteration()  # Initialize to iteration 0
        self.size = Size(self.cmp.get_row_count(), self.cmp.get_column_count())
        self.meta_names = meta_names.copy() if meta_names else []
        self.reac_names = reac_names.copy() if reac_names else []
        self.groups = []  # For tracking duplicate gene groups
    
    def __init_from_work_record(self, work_record: 'WorkRecord'):
        """
        Constructor cloning an existing work record, used by subclasses.
        
        Args:
            work_record: Existing WorkRecord to clone
        """
        # Initialize parent with same matrices
        super().__init__(work_record.pre, work_record.cmp, work_record.post, work_record.reversible)
        
        # Share references to working data (shallow copy, matches Java)
        self.stats = work_record.stats
        self.size = work_record.size
        self.meta_names = work_record.meta_names
        self.reac_names = work_record.reac_names
        self.groups = work_record.groups
    
    @classmethod
    def from_work_record(cls, work_record: 'WorkRecord') -> 'WorkRecord':
        """
        Create a WorkRecord by cloning an existing work record.
        
        Args:
            work_record: Existing WorkRecord to clone
            
        Returns:
            New WorkRecord sharing references with original
        """
        new_record = cls.__new__(cls)  # Create without calling __init__
        new_record.__init_from_work_record(work_record)
        return new_record
    
    def _create_identity_matrix(self, size: int) -> BigIntegerRationalMatrix:
        """
        Create an identity matrix of given size.
        
        Args:
            size: Size of identity matrix (size x size)
            
        Returns:
            Identity matrix with 1's on diagonal
        """
        identity = DefaultBigIntegerRationalMatrix(size, size)
        for i in range(size):
            identity.set_value_at(i, i, BigFraction.ONE)
        return identity
    
    def _cancel_matrix(self, matrix: ReadableBigIntegerRationalMatrix) -> BigIntegerRationalMatrix:
        """
        Cancel (reduce) a matrix by creating a mutable copy and reducing it.
        
        Args:
            matrix: Matrix to cancel/reduce
            
        Returns:
            Reduced mutable matrix
        """
        # Create mutable copy
        mutable_matrix = matrix.to_big_integer_rational_matrix(True)  # New instance
        # Reduce the matrix (GCD reduction)
        mutable_matrix.reduce()
        return mutable_matrix
    
    def get_truncated(self) -> CompressionRecord:
        """
        Create a truncated compression record containing only the active part of matrices.
        
        This method creates submatrices containing only the "active" portions as
        defined by self.size, excluding the inactive parts that have been moved to
        the end during compression.
        
        Returns:
            CompressionRecord with truncated matrices
        """
        # Create truncated compressed matrix using current size
        cmp_trunc = self._create_sub_stoich()
        
        # Get matrix dimensions
        m = self.cmp.get_row_count()        # Original metabolites
        r = self.cmp.get_column_count()     # Original reactions  
        mc = cmp_trunc.get_row_count()      # Compressed metabolites
        rc = cmp_trunc.get_column_count()   # Compressed reactions
        
        # Create truncated reversible array
        rev_trunc = self.reversible[:rc]
        
        # Create truncated matrices
        pre_trunc = self.pre.sub_big_integer_rational_matrix(0, mc, 0, m)
        post_trunc = self.post.sub_big_integer_rational_matrix(0, r, 0, rc)
        
        return CompressionRecord(pre_trunc, cmp_trunc, post_trunc, rev_trunc, self.stats)
    
    def _create_sub_stoich(self) -> BigIntegerRationalMatrix:
        """
        Create submatrix of compressed matrix using current size.
        
        Returns:
            Submatrix containing only active metabolites and reactions
        """
        return self.cmp.sub_big_integer_rational_matrix(0, self.size.metas, 0, self.size.reacs)
    
    def remove_reaction(self, reac: int) -> None:
        """
        Remove the specified reaction. The concerned reaction is put to the end
        of stoich/post/reversible (objects are modified) and decrements size.reacs.
        
        Args:
            reac: Index of reaction to remove
        """
        # Decrement active reaction count first
        self.size.reacs -= 1
        
        # If not already at the end, swap with last active reaction
        if reac != self.size.reacs:
            # Swap columns in matrices
            self.post.swap_columns(reac, self.size.reacs)
            self.cmp.swap_columns(reac, self.size.reacs)
            
            # Swap arrays elements
            self._swap_list_elements(self.reversible, reac, self.size.reacs)
            self._swap_list_elements(self.reac_names, reac, self.size.reacs)
        
        # Zero out the reaction column in compressed matrix (now at correct position)
        for meta in range(self.size.metas):
            self.cmp.set_value_at(meta, self.size.reacs, BigFraction.ZERO)
    
    def remove_reactions(self, suppressed_reactions: Optional[Set[str]]) -> bool:
        """
        Remove the specified reactions by name. The concerned reactions are put to the end
        of stoich/post/reversible (objects are modified) and decrements size.reacs.
        
        Args:
            suppressed_reactions: Set of reaction names to remove
            
        Returns:
            True if any reactions were removed, False otherwise
        """
        if not suppressed_reactions:
            return False
        
        # Find indices of reactions to remove
        indices_to_remove = []
        for reac_name in suppressed_reactions:
            try:
                index = self.reac_names.index(reac_name)
                if index < self.size.reacs:  # Only remove active reactions
                    indices_to_remove.append(index)
            except ValueError:
                # Reaction not found - skip
                continue
        
        if not indices_to_remove:
            return False
        
        # Sort indices in descending order to remove from end first
        indices_to_remove.sort(reverse=True)
        
        # Remove reactions
        for index in indices_to_remove:
            self.remove_reaction(index)
        
        return True
    
    def remove_reactions_by_indices(self, indices_to_remove: Set[int]) -> bool:
        """
        Remove reactions by their indices.
        
        Args:
            indices_to_remove: Set of reaction indices to remove
            
        Returns:
            True if any reactions were removed
        """
        if not indices_to_remove:
            return False
        
        # Sort indices in descending order to remove from end first
        sorted_indices = sorted(indices_to_remove, reverse=True)
        
        # Remove each reaction
        for index in sorted_indices:
            if index < self.size.reacs:  # Only remove active reactions
                self.remove_reaction(index)
        
        return True
    
    def remove_unused_metabolites(self) -> bool:
        """
        Remove unused metabolites from the working record.
        
        This method scans for metabolites (rows) that have all zero coefficients
        in the active reaction columns and removes them.
        
        Returns:
            True if any metabolites were removed
        """
        removed_any = False
        meta = 0
        while meta < self.size.metas:
            # Check if metabolite has any non-zero coefficients
            has_nonzero = False
            for reac in range(self.size.reacs):
                if not self.cmp.get_big_fraction_value_at(meta, reac).is_zero():
                    has_nonzero = True
                    break
            
            if not has_nonzero:
                # Remove unused metabolite
                self.remove_metabolite(meta)
                # Statistics tracking
                self.stats.inc_unused_metabolite()
                removed_any = True
            else:
                meta += 1
        
        return removed_any
    
    def remove_metabolite(self, meta: int) -> None:
        """
        Remove the specified metabolite. The concerned metabolite is put to the end
        of the matrices and decrements size.metas.
        
        Args:
            meta: Index of metabolite to remove
        """
        # Decrement active metabolite count first
        self.size.metas -= 1
        
        # If not already at the end, swap with last active metabolite
        if meta != self.size.metas:
            # Swap rows in matrices
            self.pre.swap_rows(meta, self.size.metas)
            self.cmp.swap_rows(meta, self.size.metas)
            
            # Swap metabolite names
            self._swap_list_elements(self.meta_names, meta, self.size.metas)
        
        # Zero out the metabolite row in compressed matrix (now at correct position)
        for reac in range(self.size.reacs):
            self.cmp.set_value_at(self.size.metas, reac, BigFraction.ZERO)
    
    def _swap_list_elements(self, lst: List, i: int, j: int) -> None:
        """
        Swap elements in a list.
        
        Args:
            lst: List to modify
            i: First index
            j: Second index
        """
        if 0 <= i < len(lst) and 0 <= j < len(lst):
            lst[i], lst[j] = lst[j], lst[i]
    
    def get_reaction_names(self, indices: List[int]) -> List[str]:
        """
        Get reaction names for a list of reaction indices.
        
        Args:
            indices: List of reaction indices
            
        Returns:
            List of reaction names
        """
        return [self.reac_names[i] for i in indices if 0 <= i < len(self.reac_names)]
    
    def log_reaction_details(self, level: int, prefix: str, reac: int) -> None:
        """
        Log detailed string representation of a reaction.
        
        Args:
            level: Logging level
            prefix: String prefix for log message
            reac: Reaction index
        """
        import logging
        logger = logging.getLogger(__name__)
        if logger.isEnabledFor(level):
            reaction_details = self.get_reaction_details(reac)
            logger.log(level, f"{prefix}{reaction_details}")
    
    def get_reaction_details(self, reac: int) -> str:
        """
        Get detailed string representation of a reaction for debugging.
        
        Args:
            reac: Reaction index
            
        Returns:
            Detailed reaction string with stoichiometric coefficients
        """
        if reac < 0 or reac >= len(self.reac_names):
            return f"INVALID_REACTION[{reac}]"
        
        reaction_name = self.reac_names[reac]
        reversible_flag = "<=>" if (reac < len(self.reversible) and self.reversible[reac]) else "=>"
        
        # Build stoichiometric equation
        educts = []
        products = []
        
        for meta in range(min(self.size.metas, len(self.meta_names))):
            coeff = self.cmp.get_big_fraction_value_at(meta, reac)
            if not coeff.is_zero():
                meta_name = self.meta_names[meta]
                if coeff.is_negative():
                    # Educt (reactant) - negative coefficient
                    educts.append(f"{coeff.negate()} {meta_name}")
                else:
                    # Product - positive coefficient
                    products.append(f"{coeff} {meta_name}")
        
        educt_str = " + ".join(educts) if educts else "∅"
        product_str = " + ".join(products) if products else "∅"
        
        return f"{reaction_name}: {educt_str} {reversible_flag} {product_str}"
    
    def get_working_info(self) -> dict:
        """
        Get information about the current working state.
        
        Returns:
            Dictionary with working record statistics
        """
        info = super().get_compression_info()
        info.update({
            "active_size": {
                "metabolites": self.size.metas,
                "reactions": self.size.reacs
            },
            "total_size": {
                "metabolites": len(self.meta_names),
                "reactions": len(self.reac_names)
            },
            "compression_iteration": self.stats.get_compression_iteration()
        })
        return info
    
    def __str__(self) -> str:
        """String representation focusing on working data"""
        return f"WorkRecord(active={self.size}, iteration={self.stats.get_compression_iteration()})"
    
    def __repr__(self) -> str:
        """Detailed representation for debugging"""
        return (f"WorkRecord(size={self.size}, "
                f"meta_names={len(self.meta_names)}, "
                f"reac_names={len(self.reac_names)}, "
                f"iteration={self.stats.get_compression_iteration()})")
    
    def validate_working_state(self) -> bool:
        """
        Validate the working record state for consistency.
        
        Returns:
            True if working state is valid, False otherwise
        """
        try:
            # Check base record validity
            if not self.validate_dimensions():
                return False
            
            # Check size consistency
            if (self.size.metas > self.get_compressed_metabolite_count() or
                self.size.reacs > self.get_compressed_reaction_count()):
                return False
            
            # Check name array consistency
            if (len(self.meta_names) != self.get_original_metabolite_count() or
                len(self.reac_names) != self.get_original_reaction_count()):
                return False
            
            return True
            
        except (AttributeError, TypeError, IndexError):
            return False