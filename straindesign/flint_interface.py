"""
Rational arithmetic interface for straindesign.

Backend: FLINT (fast) or sympy (fallback).
All FLINT-specific code is contained in this module.

Uses sympy.Rational throughout (COBRA already depends on sympy).

Example usage:
    >>> from straindesign.flint_interface import RationalMatrix, nullspace
    >>> mat = RationalMatrix.from_numpy(stoich_array)
    >>> kernel = nullspace(mat)
"""

from typing import List
import numpy as np

from sympy import Rational, Matrix as SympyMatrix, nsimplify

# Backend detection - ONLY place flint is imported in entire straindesign package
try:
    from flint import fmpq, fmpq_mat
    FLINT_AVAILABLE = True
except ImportError:
    FLINT_AVAILABLE = False
    fmpq = None
    fmpq_mat = None


def float_to_rational(val: float) -> Rational:
    """Convert float to sympy.Rational using nsimplify for nice fractions."""
    if val == 0:
        return Rational(0)
    if val == int(val):
        return Rational(int(val))
    return nsimplify(val, rational=True)


# =============================================================================
# Rational Matrix
# =============================================================================

class RationalMatrix:
    """
    Matrix of exact rationals.
    Uses FLINT fmpq_mat when available, sympy.Rational otherwise.
    """

    def __init__(self, rows: int, cols: int):
        """Create zero matrix with given dimensions."""
        self._rows = rows
        self._cols = cols

        if FLINT_AVAILABLE:
            self._mat = fmpq_mat(rows, cols)
            self._is_flint = True
        else:
            self._data = [Rational(0)] * (rows * cols)
            self._is_flint = False

    @classmethod
    def _from_flint(cls, mat: 'fmpq_mat') -> 'RationalMatrix':
        obj = object.__new__(cls)
        obj._rows = mat.nrows()
        obj._cols = mat.ncols()
        obj._mat = mat
        obj._is_flint = True
        return obj

    @classmethod
    def _from_sympy_data(cls, data: List, rows: int, cols: int) -> 'RationalMatrix':
        obj = object.__new__(cls)
        obj._rows = rows
        obj._cols = cols
        obj._data = data
        obj._is_flint = False
        return obj

    @classmethod
    def identity(cls, size: int) -> 'RationalMatrix':
        result = cls(size, size)
        for i in range(size):
            result.set_value(i, i, Rational(1))
        return result

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> 'RationalMatrix':
        """Create RationalMatrix from numpy array."""
        rows, cols = arr.shape
        result = cls(rows, cols)

        for r in range(rows):
            for c in range(cols):
                val = arr[r, c]
                if val != 0:
                    result.set_value(r, c, float_to_rational(val))

        return result

    # --- Dimensions ---

    def get_row_count(self) -> int:
        return self._rows

    def get_column_count(self) -> int:
        return self._cols

    # --- Element access ---

    def get_value(self, row: int, col: int) -> Rational:
        """Get sympy.Rational value at position."""
        if self._is_flint:
            v = self._mat[row, col]
            return Rational(int(v.p), int(v.q))
        else:
            return self._data[row * self._cols + col]

    def get_numerator(self, row: int, col: int) -> int:
        if self._is_flint:
            return int(self._mat[row, col].p)
        else:
            return int(self._data[row * self._cols + col].p)

    def get_denominator(self, row: int, col: int) -> int:
        if self._is_flint:
            return int(self._mat[row, col].q)
        else:
            return int(self._data[row * self._cols + col].q)

    def get_signum(self, row: int, col: int) -> int:
        num = self.get_numerator(row, col)
        if num == 0:
            return 0
        den = self.get_denominator(row, col)
        return (1 if num > 0 else -1) * (1 if den > 0 else -1)

    def set_value(self, row: int, col: int, value: Rational) -> None:
        """Set value from sympy.Rational."""
        if self._is_flint:
            self._mat[row, col] = fmpq(int(value.p), int(value.q))
        else:
            self._data[row * self._cols + col] = value

    def set_rational(self, row: int, col: int, num: int, den: int) -> None:
        """Set value from numerator/denominator."""
        if self._is_flint:
            self._mat[row, col] = fmpq(num, den)
        else:
            self._data[row * self._cols + col] = Rational(num, den)

    # --- Row/column operations ---

    def swap_rows(self, i: int, j: int) -> None:
        if i == j:
            return
        if self._is_flint:
            for c in range(self._cols):
                tmp = self._mat[i, c]
                self._mat[i, c] = self._mat[j, c]
                self._mat[j, c] = tmp
        else:
            for c in range(self._cols):
                idx_i, idx_j = i * self._cols + c, j * self._cols + c
                self._data[idx_i], self._data[idx_j] = self._data[idx_j], self._data[idx_i]

    def swap_columns(self, i: int, j: int) -> None:
        if i == j:
            return
        if self._is_flint:
            for r in range(self._rows):
                tmp = self._mat[r, i]
                self._mat[r, i] = self._mat[r, j]
                self._mat[r, j] = tmp
        else:
            for r in range(self._rows):
                idx_i, idx_j = r * self._cols + i, r * self._cols + j
                self._data[idx_i], self._data[idx_j] = self._data[idx_j], self._data[idx_i]

    # --- Matrix operations ---

    def clone(self) -> 'RationalMatrix':
        if self._is_flint:
            new_mat = fmpq_mat(self._rows, self._cols)
            for r in range(self._rows):
                for c in range(self._cols):
                    new_mat[r, c] = self._mat[r, c]
            return RationalMatrix._from_flint(new_mat)
        else:
            return RationalMatrix._from_sympy_data(self._data[:], self._rows, self._cols)

    # --- Conversion ---

    def to_numpy(self) -> np.ndarray:
        result = np.zeros((self._rows, self._cols), dtype=float)
        for r in range(self._rows):
            for c in range(self._cols):
                result[r, c] = float(self.get_numerator(r, c)) / float(self.get_denominator(r, c))
        return result

    def __repr__(self) -> str:
        return f"RationalMatrix({self._rows}x{self._cols})"


# =============================================================================
# Linear Algebra Functions
# =============================================================================

def nullspace(matrix: RationalMatrix) -> RationalMatrix:
    """Compute right nullspace (kernel). Returns K where matrix @ K = 0."""
    if FLINT_AVAILABLE:
        return _nullspace_flint(matrix)
    return _nullspace_sympy(matrix)


def rank(matrix: RationalMatrix) -> int:
    """Compute matrix rank."""
    if FLINT_AVAILABLE:
        return matrix._mat.rank()
    return _rank_sympy(matrix)


def basic_columns(matrix: RationalMatrix) -> List[int]:
    """Find pivot column indices from RREF."""
    if FLINT_AVAILABLE:
        return _basic_columns_flint(matrix)
    return _basic_columns_sympy(matrix)


def basic_columns_from_numpy(mx: np.ndarray) -> List[int]:
    """Find basic columns of numpy array using exact arithmetic."""
    return basic_columns(RationalMatrix.from_numpy(mx))


# =============================================================================
# Private FLINT implementations
# =============================================================================

def _nullspace_flint(matrix: RationalMatrix) -> RationalMatrix:
    mat = matrix._mat
    rows, cols = mat.nrows(), mat.ncols()

    work = fmpq_mat(rows, cols)
    for r in range(rows):
        for c in range(cols):
            work[r, c] = mat[r, c]

    rref, rk = work.rref()

    pivot_cols = []
    for r in range(min(rows, rk)):
        for c in range(cols):
            if rref[r, c] != fmpq(0):
                pivot_cols.append(c)
                break

    pivot_set = set(pivot_cols)
    free_cols = [c for c in range(cols) if c not in pivot_set]

    if not free_cols:
        return RationalMatrix(cols, 0)

    kernel = RationalMatrix(cols, len(free_cols))
    for k, free_col in enumerate(free_cols):
        kernel._mat[free_col, k] = fmpq(1)
        for i, pivot_col in enumerate(pivot_cols):
            val = rref[i, free_col]
            if val != fmpq(0):
                kernel._mat[pivot_col, k] = -val

    return kernel


def _basic_columns_flint(matrix: RationalMatrix) -> List[int]:
    mat = matrix._mat
    rows, cols = mat.nrows(), mat.ncols()

    work = fmpq_mat(rows, cols)
    for r in range(rows):
        for c in range(cols):
            work[r, c] = mat[r, c]

    rref, rk = work.rref()

    pivot_cols = []
    for r in range(min(rows, rk)):
        for c in range(cols):
            if rref[r, c] != fmpq(0):
                pivot_cols.append(c)
                break

    return pivot_cols


# =============================================================================
# Private sympy implementations
# =============================================================================

def _nullspace_sympy(matrix: RationalMatrix) -> RationalMatrix:
    rows, cols = matrix.get_row_count(), matrix.get_column_count()
    sympy_data = [[matrix.get_value(r, c) for c in range(cols)] for r in range(rows)]
    sympy_mat = SympyMatrix(sympy_data)

    null_vecs = sympy_mat.nullspace()
    if not null_vecs:
        return RationalMatrix(cols, 0)

    result = RationalMatrix(cols, len(null_vecs))
    for k, vec in enumerate(null_vecs):
        for r in range(cols):
            if vec[r] != 0:
                result.set_value(r, k, vec[r])

    return result


def _rank_sympy(matrix: RationalMatrix) -> int:
    rows, cols = matrix.get_row_count(), matrix.get_column_count()
    sympy_data = [[matrix.get_value(r, c) for c in range(cols)] for r in range(rows)]
    return SympyMatrix(sympy_data).rank()


def _basic_columns_sympy(matrix: RationalMatrix) -> List[int]:
    rows, cols = matrix.get_row_count(), matrix.get_column_count()
    sympy_data = [[matrix.get_value(r, c) for c in range(cols)] for r in range(rows)]
    _, pivot_cols = SympyMatrix(sympy_data).rref()
    return list(pivot_cols)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'FLINT_AVAILABLE',
    'Rational',  # Re-export sympy.Rational
    'RationalMatrix',
    'float_to_rational',
    'nullspace',
    'rank',
    'basic_columns',
    'basic_columns_from_numpy',
]
