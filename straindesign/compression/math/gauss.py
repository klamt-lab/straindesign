"""
Gauss operations - Python port of ch.javasoft.smx.ops.Gauss

This module provides Gaussian elimination operations for exact rational arithmetic matrices.
Key operations include computing rank, nullity, nullspace, and matrix inversion using
exact BigFraction arithmetic to avoid floating-point precision issues.

When python-flint is available, the FlintGauss implementation is used automatically
for much faster O(n³) operations in C.
"""

from typing import List, Optional, Tuple
from .readable_bigint_rational_matrix import ReadableBigIntegerRationalMatrix
from .bigint_rational_matrix import BigIntegerRationalMatrix
from .default_bigint_rational_matrix import DefaultBigIntegerRationalMatrix
from .big_fraction import BigFraction

# Try to import FLINT-accelerated Gauss
try:
    from .flint_gauss import FlintGauss, FLINT_AVAILABLE
    _FLINT_GAUSS_AVAILABLE = FLINT_AVAILABLE
except ImportError:
    _FLINT_GAUSS_AVAILABLE = False
    FlintGauss = None


class Gauss:
    """
    Gauss operations - Python port of ch.javasoft.smx.ops.Gauss
    
    Matrix operations based on Gaussian elimination for exact rational arithmetic.
    This implementation focuses on the operations needed by compression algorithms,
    particularly nullspace computation.
    """
    
    def __init__(self, tolerance: float = 0.0):
        """
        Initialize Gauss operations with tolerance.
        
        Args:
            tolerance: Tolerance for zero detection (0.0 for exact arithmetic)
        """
        self.tolerance = tolerance
        self._tolerance_fraction = BigFraction(tolerance) if tolerance > 0 else None
    
    # Singleton instances (matching Java pattern)
    _double_instance = None
    _rational_instance = None
    
    @classmethod
    def get_double_instance(cls) -> 'Gauss':
        """Get singleton instance for double operations"""
        if cls._double_instance is None:
            cls._double_instance = cls(1e-10)  # Small tolerance for doubles
        return cls._double_instance
    
    @classmethod
    def get_rational_instance(cls) -> 'Gauss':
        """
        Get singleton instance for exact rational operations.

        When python-flint is available, returns FlintGauss for much faster
        O(n³) operations. Falls back to pure Python implementation otherwise.
        """
        # Use FLINT-accelerated Gauss if available
        if _FLINT_GAUSS_AVAILABLE:
            return FlintGauss.get_rational_instance()

        # Fallback to pure Python implementation
        if cls._rational_instance is None:
            cls._rational_instance = cls(0.0)  # Exact arithmetic, no tolerance
        return cls._rational_instance
    
    # Core operations
    def rank(self, matrix: ReadableBigIntegerRationalMatrix) -> int:
        """
        Compute the rank of the given matrix using Gaussian elimination.
        
        Args:
            matrix: Input matrix
            
        Returns:
            The rank of the matrix
        """
        # Create a copy to avoid modifying the original
        working_matrix = matrix.to_big_integer_rational_matrix(True)  # force new instance
        # Use full pivoting for accurate rank computation
        return self._row_echelon(working_matrix, reduced=False, use_full_pivoting=True)[0]
    
    def nullity(self, matrix: ReadableBigIntegerRationalMatrix) -> int:
        """
        Compute the nullity of the given matrix (dimension of nullspace).
        By the rank-nullity theorem: rank + nullity = number of columns.
        
        Args:
            matrix: Input matrix
            
        Returns:
            The nullity of the matrix (columns - rank)
        """
        return matrix.get_column_count() - self.rank(matrix)
    
    def nullspace(self, matrix: ReadableBigIntegerRationalMatrix) -> BigIntegerRationalMatrix:
        """
        Compute a basis for the nullspace using Gaussian elimination.
        
        The algorithm:
        1. Compute reduced row echelon form with column permutations tracked
        2. RREF has structure [I, M; 0] after permutations
        3. Nullspace basis is [-M; I] with proper permutation applied
        
        Args:
            matrix: Input matrix
            
        Returns:
            Matrix whose columns form a basis for the nullspace
        """
        cols = matrix.get_column_count()
        
        # For consistency with rank computation, first get the rank using full pivoting
        true_rank = self.rank(matrix)
        nullspace_dim = cols - true_rank
        
        if nullspace_dim == 0:
            # No nullspace - return empty matrix
            return DefaultBigIntegerRationalMatrix(cols, 0)
        
        # Create identity column map to track column permutations
        colmap = list(range(cols))
        
        # Create working copy and compute RREF using full pivoting for consistency
        rref = matrix.to_big_integer_rational_matrix(True)  # force new instance
        rank, final_colmap = self._row_echelon(rref, reduced=True, colmap=colmap, use_full_pivoting=True)
        
        # Should match our pre-computed rank
        assert rank == true_rank, f"Rank mismatch: {rank} vs {true_rank}"
        
        # Create nullspace matrix
        kernel = DefaultBigIntegerRationalMatrix(cols, nullspace_dim)
        
        # Fill the nullspace basis vectors
        # For RREF structure [I, M; 0], nullspace is [-M; I]
        for row in range(rank):
            for col in range(nullspace_dim):
                # Get the value from the M part of [I, M; 0]
                value = rref.get_big_fraction_value_at(row, col + rank)
                # Set -M in the nullspace
                kernel.set_value_at(final_colmap[row], col, -value)
        
        # Set the I part (identity for free variables)
        for row in range(nullspace_dim):
            kernel.set_value_at(final_colmap[row + rank], row, BigFraction(1))
        
        return kernel
    
    def invert(self, matrix: ReadableBigIntegerRationalMatrix) -> BigIntegerRationalMatrix:
        """
        Compute the inverse of a square matrix using Gaussian elimination.
        
        The method computes the reduced row-echelon form of [A | I] to get [I | A^-1].
        
        Args:
            matrix: Square matrix to invert
            
        Returns:
            The inverse matrix
            
        Raises:
            ValueError: If matrix is not square
            ArithmeticError: If matrix is singular (not invertible)
        """
        if matrix.get_row_count() != matrix.get_column_count():
            raise ValueError(f"Matrix must be square for inversion: {matrix.get_row_count()}x{matrix.get_column_count()}")
        
        n = matrix.get_row_count()
        
        # Create augmented matrix [A | I]
        augmented = DefaultBigIntegerRationalMatrix(n, 2 * n)
        
        # Fill with original matrix
        for row in range(n):
            for col in range(n):
                augmented.set_value_at(row, col, matrix.get_number_value_at(row, col))
        
        # Fill identity part
        for i in range(n):
            augmented.set_value_at(i, i + n, BigFraction(1))
        
        # Perform Gaussian elimination to get [I | A^-1] using partial pivoting
        rank, _ = self._row_echelon(augmented, reduced=True, use_full_pivoting=False)
        
        if rank < n:
            raise ArithmeticError(f"Matrix is singular (rank {rank} < {n})")
        
        # Extract the inverse from the right half
        inverse = DefaultBigIntegerRationalMatrix(n, n)
        for row in range(n):
            for col in range(n):
                inverse.set_value_at(row, col, augmented.get_big_fraction_value_at(row, col + n))
        
        return inverse
    
    def _is_zero(self, matrix: BigIntegerRationalMatrix, row: int, col: int) -> bool:
        """Check if matrix element is effectively zero (considering tolerance)"""
        value = matrix.get_big_fraction_value_at(row, col)
        if self._tolerance_fraction is None:
            return value.signum() == 0
        else:
            return abs(value) < self._tolerance_fraction
    
    def _row_echelon(self, matrix: BigIntegerRationalMatrix, reduced: bool = False, 
                    rowmap: Optional[List[int]] = None, colmap: Optional[List[int]] = None, 
                    use_full_pivoting: bool = True) -> Tuple[int, List[int]]:
        """
        Core Gaussian elimination implementation with optional full pivoting.
        
        Args:
            matrix: Matrix to reduce (modified in-place)
            reduced: Whether to compute reduced row echelon form (RREF)
            rowmap: Optional row permutation tracking
            colmap: Optional column permutation tracking
            use_full_pivoting: Whether to use full pivoting (with column swaps) or partial pivoting
            
        Returns:
            Tuple of (rank, final_colmap)
        """
        rows = matrix.get_row_count()
        cols = matrix.get_column_count()
        
        # Initialize column map if not provided
        if colmap is None:
            colmap = list(range(cols))
        
        if use_full_pivoting:
            return self._row_echelon_full_pivoting(matrix, reduced, rowmap, colmap)
        else:
            return self._row_echelon_partial_pivoting(matrix, reduced, rowmap, colmap)
    
    def _row_echelon_full_pivoting(self, matrix: BigIntegerRationalMatrix, reduced: bool, 
                                  rowmap: Optional[List[int]], colmap: List[int]) -> Tuple[int, List[int]]:
        """Full pivoting implementation for accurate rank computation."""
        rows = matrix.get_row_count()
        cols = matrix.get_column_count()
        max_pivots = min(rows, cols)
        current_rank = 0
        
        # Full pivoting: for each pivot position, search entire submatrix
        for pivot_idx in range(max_pivots):
            # Find the best pivot in the remaining submatrix
            best_pivot = self._find_best_pivot_full(matrix, pivot_idx, pivot_idx)
            
            if best_pivot is None:
                # No non-zero pivot found, matrix rank is current_rank
                break
            
            pivot_row, pivot_col = best_pivot
            
            # Swap rows if needed
            if pivot_row != pivot_idx:
                matrix.swap_rows(pivot_row, pivot_idx)
                if rowmap is not None and pivot_idx < len(rowmap) and pivot_row < len(rowmap):
                    rowmap[pivot_idx], rowmap[pivot_row] = rowmap[pivot_row], rowmap[pivot_idx]
            
            # Swap columns if needed
            if pivot_col != pivot_idx:
                matrix.swap_columns(pivot_col, pivot_idx)
                if colmap is not None:
                    colmap[pivot_idx], colmap[pivot_col] = colmap[pivot_col], colmap[pivot_idx]
            
            # Get pivot value (now at position [pivot_idx, pivot_idx])
            pivot_value = matrix.get_big_fraction_value_at(pivot_idx, pivot_idx)
            
            # Normalize pivot row (make pivot = 1)
            if not pivot_value.is_one():
                for j in range(cols):
                    if not self._is_zero(matrix, pivot_idx, j):
                        current_val = matrix.get_big_fraction_value_at(pivot_idx, j)
                        matrix.set_value_at(pivot_idx, j, current_val / pivot_value)
            
            # Eliminate column below pivot (and above if reduced)
            self._eliminate_column(matrix, pivot_idx, pivot_idx, reduced)
            
            current_rank += 1
        
        return current_rank, colmap
    
    def _row_echelon_partial_pivoting(self, matrix: BigIntegerRationalMatrix, reduced: bool,
                                     rowmap: Optional[List[int]], colmap: List[int]) -> Tuple[int, List[int]]:
        """Partial pivoting implementation for structure-preserving operations."""
        rows = matrix.get_row_count()
        cols = matrix.get_column_count()
        current_row = 0
        
        # Process each column looking for pivots
        for col in range(min(rows, cols)):
            # Find the best pivot in current column (starting from current_row)
            pivot_row = self._find_pivot_row_enhanced(matrix, current_row, col)
            
            if pivot_row == -1:
                # No non-zero pivot found in this column
                continue
            
            # Swap rows if needed
            if pivot_row != current_row:
                matrix.swap_rows(pivot_row, current_row)
                if rowmap is not None and current_row < len(rowmap) and pivot_row < len(rowmap):
                    rowmap[current_row], rowmap[pivot_row] = rowmap[pivot_row], rowmap[current_row]
            
            # Get pivot value
            pivot_value = matrix.get_big_fraction_value_at(current_row, col)
            
            # Normalize pivot row (make pivot = 1)
            if not pivot_value.is_one():
                for j in range(cols):
                    if not self._is_zero(matrix, current_row, j):
                        current_val = matrix.get_big_fraction_value_at(current_row, j)
                        matrix.set_value_at(current_row, j, current_val / pivot_value)
            
            # Eliminate column below pivot (and above if reduced)
            self._eliminate_column(matrix, current_row, col, reduced)
            
            current_row += 1
            
            # If we've processed all rows, we're done
            if current_row >= rows:
                break
        
        return current_row, colmap
    
    def _find_best_pivot_full(self, matrix: BigIntegerRationalMatrix, start_row: int, start_col: int) -> Optional[Tuple[int, int]]:
        """
        Find the best pivot in the remaining submatrix using full pivoting with bit length heuristic.
        
        This implementation follows the Java BiLenProductL strategy:
        - Prefers pivots with smaller bit length products (numerator_bits * denominator_bits)
        - Breaks ties by preferring elements that create fewer non-zeros in elimination
        
        Args:
            matrix: The matrix
            start_row: First row to consider
            start_col: First column to consider
            
        Returns:
            Tuple of (pivot_row, pivot_col) or None if no suitable pivot found
        """
        best_row = -1
        best_col = -1
        best_score = float('inf')
        
        rows = matrix.get_row_count()
        cols = matrix.get_column_count()
        
        # Search the entire remaining submatrix
        for row in range(start_row, rows):
            for col in range(start_col, cols):
                if not self._is_zero(matrix, row, col):
                    # Get the element value
                    value = matrix.get_big_fraction_value_at(row, col)
                    
                    # Compute pivot quality score (lower is better)
                    score = self._compute_pivot_score(value, matrix, row, col, start_row)
                    
                    if score < best_score:
                        best_score = score
                        best_row = row
                        best_col = col
        
        if best_row == -1:
            return None
        else:
            return (best_row, best_col)
    
    def _compute_pivot_score(self, value: BigFraction, matrix: BigIntegerRationalMatrix, 
                           row: int, col: int, pivot_start: int) -> float:
        """
        Compute a pivot quality score (lower is better).
        
        Optimized version that avoids expensive sparsity computation for performance.
        Uses primarily bit length as the scoring criterion.
        
        Args:
            value: The pivot candidate value
            matrix: The matrix
            row: Row index
            col: Column index  
            pivot_start: Starting index for pivot search
            
        Returns:
            Score for this pivot (lower is better)
        """
        # Primary score: bit length product (favors smaller numbers)
        num_bits = value.numerator.bit_length() if value.numerator != 0 else 1
        den_bits = value.denominator.bit_length() if value.denominator != 0 else 1
        bit_length_product = num_bits * den_bits
        
        # Simple secondary score based on position (prefer upper-left)
        # This avoids the expensive sparsity computation that was causing hangs
        position_score = row + col
        
        # Return weighted score (bit length dominates, position as tiebreaker)
        return bit_length_product * 1000 + position_score
    
    def _find_pivot_row_enhanced(self, matrix: BigIntegerRationalMatrix, start_row: int, col: int) -> int:
        """
        Find the best pivot row in the given column using bit length heuristic.
        
        Args:
            matrix: The matrix
            start_row: First row to consider
            col: Column to search in
            
        Returns:
            Row index of best pivot, or -1 if no suitable pivot found
        """
        best_row = -1
        best_score = float('inf')
        
        # Enhanced pivot selection: use bit length heuristic
        for row in range(start_row, matrix.get_row_count()):
            if not self._is_zero(matrix, row, col):
                value = matrix.get_big_fraction_value_at(row, col)
                score = self._compute_pivot_score(value, matrix, row, col, start_row)
                
                if score < best_score:
                    best_score = score
                    best_row = row
        
        return best_row
    
    def _find_pivot_row(self, matrix: BigIntegerRationalMatrix, start_row: int, col: int) -> int:
        """
        Find the best pivot row in the given column (legacy method for compatibility).
        
        Args:
            matrix: The matrix
            start_row: First row to consider
            col: Column to search in
            
        Returns:
            Row index of best pivot, or -1 if no suitable pivot found
        """
        best_row = -1
        
        # Simple pivot selection: first non-zero element
        for row in range(start_row, matrix.get_row_count()):
            if not self._is_zero(matrix, row, col):
                best_row = row
                break
        
        return best_row
    
    def _eliminate_column(self, matrix: BigIntegerRationalMatrix, pivot_row: int, col: int, reduced: bool):
        """
        Eliminate the column using the pivot row.
        
        Args:
            matrix: The matrix
            pivot_row: Row containing the pivot
            col: Column to eliminate
            reduced: If True, eliminate above and below; if False, only below
        """
        rows = matrix.get_row_count()
        cols = matrix.get_column_count()
        
        # Eliminate below (always)
        for row in range(pivot_row + 1, rows):
            if not self._is_zero(matrix, row, col):
                multiplier = matrix.get_big_fraction_value_at(row, col)
                
                # Subtract multiplier * pivot_row from row
                for j in range(cols):
                    if not self._is_zero(matrix, pivot_row, j):
                        pivot_val = matrix.get_big_fraction_value_at(pivot_row, j)
                        current_val = matrix.get_big_fraction_value_at(row, j)
                        new_val = current_val - multiplier * pivot_val
                        matrix.set_value_at(row, j, new_val)
        
        # Eliminate above (only if reduced=True)
        if reduced:
            for row in range(pivot_row):
                if not self._is_zero(matrix, row, col):
                    multiplier = matrix.get_big_fraction_value_at(row, col)
                    
                    # Subtract multiplier * pivot_row from row
                    for j in range(cols):
                        if not self._is_zero(matrix, pivot_row, j):
                            pivot_val = matrix.get_big_fraction_value_at(pivot_row, j)
                            current_val = matrix.get_big_fraction_value_at(row, j)
                            new_val = current_val - multiplier * pivot_val
                            matrix.set_value_at(row, j, new_val)