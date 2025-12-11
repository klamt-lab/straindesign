"""
FlintGauss - FLINT-accelerated Gaussian elimination for exact rational arithmetic.

This module provides fast Gaussian elimination operations using python-flint's fmpq_mat.
It is a drop-in replacement for the pure Python Gauss class when FLINT is available.
"""

from typing import List, Optional, Tuple

try:
    from flint import fmpq_mat, fmpq
    FLINT_AVAILABLE = True
except ImportError:
    FLINT_AVAILABLE = False
    fmpq_mat = None
    fmpq = None

from .readable_bigint_rational_matrix import ReadableBigIntegerRationalMatrix
from .bigint_rational_matrix import BigIntegerRationalMatrix
from .flint_rational_matrix import FlintBigIntegerRationalMatrix


class FlintGauss:
    """
    FLINT-accelerated Gaussian elimination for exact rational arithmetic.

    This class provides the same interface as Gauss but uses FLINT's native
    C implementation for O(n³) operations, providing massive speedups for
    large matrices.
    """

    _rational_instance = None

    def __init__(self, tolerance: float = 0.0):
        """
        Initialize FlintGauss with tolerance.

        Args:
            tolerance: Tolerance for zero detection (0.0 for exact arithmetic)
                      Note: FLINT always uses exact arithmetic, tolerance is ignored
        """
        self.tolerance = tolerance
        # FLINT uses exact arithmetic, so tolerance is effectively 0

    @classmethod
    def get_rational_instance(cls) -> 'FlintGauss':
        """Get singleton instance for exact rational operations."""
        if cls._rational_instance is None:
            cls._rational_instance = cls(0.0)
        return cls._rational_instance

    def _to_flint(self, matrix: ReadableBigIntegerRationalMatrix) -> 'fmpq_mat':
        """
        Convert any readable matrix to FLINT fmpq_mat.

        Args:
            matrix: Input matrix (any type implementing ReadableBigIntegerRationalMatrix)

        Returns:
            FLINT fmpq_mat with same values
        """
        if isinstance(matrix, FlintBigIntegerRationalMatrix):
            # Already FLINT - make a copy to avoid modifying original
            src = matrix.get_flint_matrix()
            rows, cols = src.nrows(), src.ncols()
            result = fmpq_mat(rows, cols)
            for r in range(rows):
                for c in range(cols):
                    result[r, c] = src[r, c]
            return result

        # Convert from other matrix type
        rows = matrix.get_row_count()
        cols = matrix.get_column_count()
        result = fmpq_mat(rows, cols)

        for r in range(rows):
            for c in range(cols):
                num = matrix.get_big_integer_numerator_at(r, c)
                if num != 0:
                    den = matrix.get_big_integer_denominator_at(r, c)
                    result[r, c] = fmpq(num, den)

        return result

    def rank(self, matrix: ReadableBigIntegerRationalMatrix) -> int:
        """
        Compute the rank of the given matrix using FLINT's native rank().

        Args:
            matrix: Input matrix

        Returns:
            The rank of the matrix
        """
        mat = self._to_flint(matrix)
        return mat.rank()

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
        Compute a basis for the nullspace using FLINT's native RREF.

        The algorithm:
        1. Use FLINT's rref() to compute reduced row echelon form (very fast O(n³) in C)
        2. Identify pivot columns (columns with leading 1s) vs free columns
        3. Build nullspace basis: for each free column, create a basis vector

        Args:
            matrix: Input matrix

        Returns:
            Matrix whose columns form a basis for the nullspace
        """
        mat = self._to_flint(matrix)
        rows, cols = mat.nrows(), mat.ncols()

        # Use FLINT's native RREF - O(n³) implemented in C, extremely fast
        rref, rank = mat.rref()

        # Find pivot columns (columns with leading 1s in RREF)
        # For each row, find the first non-zero column - that's a pivot column
        pivot_cols = []
        for r in range(min(rows, rank)):
            for c in range(cols):
                if rref[r, c] != fmpq(0):
                    pivot_cols.append(c)
                    break

        # Free columns = all columns not in pivot_cols
        pivot_set = set(pivot_cols)
        free_cols = [c for c in range(cols) if c not in pivot_set]
        nullspace_dim = len(free_cols)

        if nullspace_dim == 0:
            # Trivial nullspace - return empty matrix
            return FlintBigIntegerRationalMatrix(cols, 0)

        # Build nullspace basis
        # For each free column, create a basis vector where:
        # - The free variable position gets 1
        # - Each pivot variable position gets -RREF[pivot_row, free_col]
        kernel = FlintBigIntegerRationalMatrix(cols, nullspace_dim)

        for k, free_col in enumerate(free_cols):
            # Set 1 at the free variable position
            kernel.set_flint_value_at(free_col, k, fmpq(1))

            # Set -RREF values at pivot variable positions
            for i, pivot_col in enumerate(pivot_cols):
                val = rref[i, free_col]
                if val != fmpq(0):
                    kernel.set_flint_value_at(pivot_col, k, -val)

        return kernel

    def basic_columns(self, matrix: ReadableBigIntegerRationalMatrix) -> List[int]:
        """
        Find the indices of basic (pivot) columns using RREF.

        The basic columns are those that correspond to pivot positions in the RREF.
        These form a basis for the column space.

        Args:
            matrix: Input matrix

        Returns:
            List of column indices that are basic (pivot) columns
        """
        mat = self._to_flint(matrix)
        rows, cols = mat.nrows(), mat.ncols()

        # Use FLINT's native RREF
        rref, rank = mat.rref()

        # Find pivot columns (columns with leading 1s in RREF)
        pivot_cols = []
        for r in range(min(rows, rank)):
            for c in range(cols):
                if rref[r, c] != fmpq(0):
                    pivot_cols.append(c)
                    break

        return pivot_cols

    def invert(self, matrix: ReadableBigIntegerRationalMatrix) -> BigIntegerRationalMatrix:
        """
        Compute the inverse of a square matrix using FLINT's native inv().

        Args:
            matrix: Square matrix to invert

        Returns:
            The inverse matrix

        Raises:
            ValueError: If matrix is not square
            ArithmeticError: If matrix is singular (not invertible)
        """
        if matrix.get_row_count() != matrix.get_column_count():
            raise ValueError(
                f"Matrix must be square for inversion: "
                f"{matrix.get_row_count()}x{matrix.get_column_count()}"
            )

        mat = self._to_flint(matrix)

        try:
            inv = mat.inv()
            return FlintBigIntegerRationalMatrix(inv)
        except ZeroDivisionError:
            # FLINT raises ZeroDivisionError for singular matrices
            raise ArithmeticError(f"Matrix is singular (not invertible)")
