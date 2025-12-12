"""
Rational arithmetic interface for straindesign.

Backend: FLINT (fast) or sympy (fallback).
All FLINT-specific code is contained in this module.

This module provides:
- Rational: Exact rational number (fmpq or sympy.Rational)
- RationalMatrix: Matrix of exact rationals (fmpq_mat or sympy)
- nullspace, rank, basic_columns: Linear algebra operations

Example usage:
    >>> from straindesign.flint_interface import RationalMatrix, nullspace
    >>> mat = RationalMatrix.from_numpy(stoich_array)
    >>> kernel = nullspace(mat)
"""

from typing import List, Optional, Union
import numpy as np
from fractions import Fraction

# Use sympy.Rational as fallback - COBRA already depends on sympy
from sympy import Rational as SympyRational, Matrix as SympyMatrix

# Backend detection - ONLY place flint is imported in entire straindesign package
try:
    from flint import fmpq, fmpq_mat
    FLINT_AVAILABLE = True
except ImportError:
    FLINT_AVAILABLE = False
    fmpq = None
    fmpq_mat = None


# =============================================================================
# Rational Number
# =============================================================================

class Rational:
    """
    Exact rational number.

    Uses FLINT fmpq when available, sympy.Rational otherwise.
    Provides a unified interface for exact rational arithmetic.
    """
    __slots__ = ('_value', '_is_flint')

    def __init__(self, numerator: int, denominator: int = 1):
        """Create rational from numerator and denominator."""
        if FLINT_AVAILABLE:
            self._value = fmpq(numerator, denominator)
            self._is_flint = True
        else:
            self._value = SympyRational(numerator, denominator)
            self._is_flint = False

    @classmethod
    def _from_raw(cls, value, is_flint: bool) -> 'Rational':
        """Create Rational from raw fmpq or sympy.Rational (internal use)."""
        obj = object.__new__(cls)
        obj._value = value
        obj._is_flint = is_flint
        return obj

    @classmethod
    def from_sympy(cls, val: SympyRational) -> 'Rational':
        """Create Rational from sympy.Rational."""
        if FLINT_AVAILABLE:
            return cls(int(val.p), int(val.q))
        return cls._from_raw(val, False)

    @classmethod
    def from_float(cls, val: float, limit_denominator: bool = True) -> 'Rational':
        """Create Rational from float, optionally limiting denominator."""
        frac = Fraction(val)
        if limit_denominator:
            frac = frac.limit_denominator()
        return cls(frac.numerator, frac.denominator)

    @property
    def numerator(self) -> int:
        """Return numerator."""
        if self._is_flint:
            return int(self._value.p)
        return int(self._value.p)

    @property
    def denominator(self) -> int:
        """Return denominator."""
        if self._is_flint:
            return int(self._value.q)
        return int(self._value.q)

    def __add__(self, other: 'Rational') -> 'Rational':
        if not isinstance(other, Rational):
            return NotImplemented
        return Rational._from_raw(self._value + other._value, self._is_flint)

    def __sub__(self, other: 'Rational') -> 'Rational':
        if not isinstance(other, Rational):
            return NotImplemented
        return Rational._from_raw(self._value - other._value, self._is_flint)

    def __mul__(self, other: 'Rational') -> 'Rational':
        if not isinstance(other, Rational):
            return NotImplemented
        return Rational._from_raw(self._value * other._value, self._is_flint)

    def __truediv__(self, other: 'Rational') -> 'Rational':
        if not isinstance(other, Rational):
            return NotImplemented
        return Rational._from_raw(self._value / other._value, self._is_flint)

    def __neg__(self) -> 'Rational':
        return Rational._from_raw(-self._value, self._is_flint)

    def __eq__(self, other) -> bool:
        if isinstance(other, Rational):
            return self._value == other._value
        if isinstance(other, int):
            return self._value == other
        return False

    def __hash__(self) -> int:
        return hash((self.numerator, self.denominator))

    def __repr__(self) -> str:
        if self.denominator == 1:
            return str(self.numerator)
        return f"{self.numerator}/{self.denominator}"

    def is_zero(self) -> bool:
        """Check if value is zero."""
        return self.numerator == 0

    def is_positive(self) -> bool:
        """Check if value is positive."""
        return self.signum() > 0

    def is_negative(self) -> bool:
        """Check if value is negative."""
        return self.signum() < 0

    def signum(self) -> int:
        """Return sign: -1, 0, or 1."""
        n, d = self.numerator, self.denominator
        if n == 0:
            return 0
        sign_n = 1 if n > 0 else -1
        sign_d = 1 if d > 0 else -1
        return sign_n * sign_d

    def abs(self) -> 'Rational':
        """Return absolute value."""
        if self.is_negative():
            return -self
        return self

    def invert(self) -> 'Rational':
        """Return multiplicative inverse (1/self)."""
        if self.is_zero():
            raise ArithmeticError("Cannot invert zero")
        return Rational(self.denominator, self.numerator)

    def to_sympy(self) -> SympyRational:
        """Convert to sympy.Rational."""
        if self._is_flint:
            return SympyRational(self.numerator, self.denominator)
        return self._value

    def to_float(self) -> float:
        """Convert to float."""
        return float(self.numerator) / float(self.denominator)


# Class constants (initialized after class definition)
Rational.ZERO = Rational(0, 1)
Rational.ONE = Rational(1, 1)


# =============================================================================
# Rational Matrix
# =============================================================================

class RationalMatrix:
    """
    Matrix of exact rationals.

    Uses FLINT fmpq_mat when available, list of sympy.Rational otherwise.
    Provides a unified interface for exact rational matrix operations.
    """

    def __init__(self, rows: int, cols: int):
        """Create zero matrix with given dimensions."""
        self._rows = rows
        self._cols = cols

        if FLINT_AVAILABLE:
            self._mat = fmpq_mat(rows, cols)
            self._is_flint = True
        else:
            # Store as flat list of sympy.Rational (row-major)
            self._data = [SympyRational(0)] * (rows * cols)
            self._is_flint = False

    @classmethod
    def _from_flint(cls, mat: 'fmpq_mat') -> 'RationalMatrix':
        """Create RationalMatrix wrapping existing FLINT matrix."""
        obj = object.__new__(cls)
        obj._rows = mat.nrows()
        obj._cols = mat.ncols()
        obj._mat = mat
        obj._is_flint = True
        return obj

    @classmethod
    def _from_sympy_data(cls, data: List, rows: int, cols: int) -> 'RationalMatrix':
        """Create RationalMatrix from sympy data (internal use)."""
        obj = object.__new__(cls)
        obj._rows = rows
        obj._cols = cols
        obj._data = data
        obj._is_flint = False
        return obj

    @classmethod
    def identity(cls, size: int) -> 'RationalMatrix':
        """Create identity matrix."""
        result = cls(size, size)
        for i in range(size):
            result.set_rational(i, i, 1, 1)
        return result

    @classmethod
    def from_numpy(cls, arr: np.ndarray, limit_denominator: bool = True) -> 'RationalMatrix':
        """
        Create RationalMatrix from numpy array.

        Args:
            arr: 2D numpy array
            limit_denominator: If True, use Fraction.limit_denominator() for floats
        """
        rows, cols = arr.shape
        result = cls(rows, cols)

        for r in range(rows):
            for c in range(cols):
                val = arr[r, c]
                if val != 0:
                    frac = Fraction(val)
                    if limit_denominator:
                        frac = frac.limit_denominator()
                    result.set_rational(r, c, frac.numerator, frac.denominator)

        return result

    # --- Dimensions ---

    def get_row_count(self) -> int:
        """Get number of rows."""
        return self._rows

    def get_column_count(self) -> int:
        """Get number of columns."""
        return self._cols

    # --- Element access ---

    def get(self, row: int, col: int) -> Rational:
        """Get Rational value at position."""
        if self._is_flint:
            val = self._mat[row, col]
            return Rational._from_raw(val, True)
        else:
            idx = row * self._cols + col
            return Rational._from_raw(self._data[idx], False)

    def get_numerator(self, row: int, col: int) -> int:
        """Get numerator at position."""
        if self._is_flint:
            return int(self._mat[row, col].p)
        else:
            idx = row * self._cols + col
            return int(self._data[idx].p)

    def get_denominator(self, row: int, col: int) -> int:
        """Get denominator at position."""
        if self._is_flint:
            return int(self._mat[row, col].q)
        else:
            idx = row * self._cols + col
            return int(self._data[idx].q)

    def get_signum(self, row: int, col: int) -> int:
        """Get sign (-1, 0, 1) at position."""
        num = self.get_numerator(row, col)
        if num == 0:
            return 0
        den = self.get_denominator(row, col)
        sign_n = 1 if num > 0 else -1
        sign_d = 1 if den > 0 else -1
        return sign_n * sign_d

    def set(self, row: int, col: int, value: Rational) -> None:
        """Set Rational value at position."""
        self.set_rational(row, col, value.numerator, value.denominator)

    def set_rational(self, row: int, col: int, num: int, den: int) -> None:
        """Set rational value using numerator/denominator."""
        if self._is_flint:
            self._mat[row, col] = fmpq(num, den)
        else:
            idx = row * self._cols + col
            self._data[idx] = SympyRational(num, den)

    # --- Row/column operations ---

    def swap_rows(self, i: int, j: int) -> None:
        """Swap two rows."""
        if i == j:
            return
        if self._is_flint:
            for c in range(self._cols):
                tmp = self._mat[i, c]
                self._mat[i, c] = self._mat[j, c]
                self._mat[j, c] = tmp
        else:
            for c in range(self._cols):
                idx_i = i * self._cols + c
                idx_j = j * self._cols + c
                self._data[idx_i], self._data[idx_j] = self._data[idx_j], self._data[idx_i]

    def swap_columns(self, i: int, j: int) -> None:
        """Swap two columns."""
        if i == j:
            return
        if self._is_flint:
            for r in range(self._rows):
                tmp = self._mat[r, i]
                self._mat[r, i] = self._mat[r, j]
                self._mat[r, j] = tmp
        else:
            for r in range(self._rows):
                idx_i = r * self._cols + i
                idx_j = r * self._cols + j
                self._data[idx_i], self._data[idx_j] = self._data[idx_j], self._data[idx_i]

    def multiply_row(self, row: int, num: int, den: int) -> None:
        """Multiply entire row by rational factor."""
        if self._is_flint:
            factor = fmpq(num, den)
            for c in range(self._cols):
                self._mat[row, c] = self._mat[row, c] * factor
        else:
            factor = SympyRational(num, den)
            for c in range(self._cols):
                idx = row * self._cols + c
                self._data[idx] = self._data[idx] * factor

    def add_row_scaled(self, src: int, dst: int, num: int, den: int) -> None:
        """Add src row * (num/den) to dst row."""
        if self._is_flint:
            factor = fmpq(num, den)
            for c in range(self._cols):
                self._mat[dst, c] = self._mat[dst, c] + self._mat[src, c] * factor
        else:
            factor = SympyRational(num, den)
            for c in range(self._cols):
                src_idx = src * self._cols + c
                dst_idx = dst * self._cols + c
                self._data[dst_idx] = self._data[dst_idx] + self._data[src_idx] * factor

    # --- Matrix operations ---

    def clone(self) -> 'RationalMatrix':
        """Create deep copy."""
        if self._is_flint:
            new_mat = fmpq_mat(self._rows, self._cols)
            for r in range(self._rows):
                for c in range(self._cols):
                    new_mat[r, c] = self._mat[r, c]
            return RationalMatrix._from_flint(new_mat)
        else:
            return RationalMatrix._from_sympy_data(self._data[:], self._rows, self._cols)

    def transpose(self) -> 'RationalMatrix':
        """Return transposed matrix."""
        if self._is_flint:
            return RationalMatrix._from_flint(self._mat.transpose())
        else:
            new_data = [SympyRational(0)] * (self._rows * self._cols)
            for r in range(self._rows):
                for c in range(self._cols):
                    old_idx = r * self._cols + c
                    new_idx = c * self._rows + r
                    new_data[new_idx] = self._data[old_idx]
            return RationalMatrix._from_sympy_data(new_data, self._cols, self._rows)

    # --- Conversion ---

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy float array."""
        result = np.zeros((self._rows, self._cols), dtype=float)
        for r in range(self._rows):
            for c in range(self._cols):
                num = self.get_numerator(r, c)
                den = self.get_denominator(r, c)
                result[r, c] = float(num) / float(den)
        return result

    def get_sympy_value(self, row: int, col: int) -> SympyRational:
        """Get value as sympy.Rational."""
        if self._is_flint:
            return SympyRational(self.get_numerator(row, col), self.get_denominator(row, col))
        else:
            idx = row * self._cols + col
            return self._data[idx]

    def __repr__(self) -> str:
        return f"RationalMatrix({self._rows}x{self._cols})"


# =============================================================================
# Linear Algebra Functions
# =============================================================================

def nullspace(matrix: RationalMatrix) -> RationalMatrix:
    """
    Compute right nullspace (kernel) of matrix.

    Returns matrix K where matrix @ K = 0.
    Columns of K form a basis for the nullspace.
    """
    if FLINT_AVAILABLE:
        return _nullspace_flint(matrix)
    return _nullspace_sympy(matrix)


def rank(matrix: RationalMatrix) -> int:
    """Compute matrix rank."""
    if FLINT_AVAILABLE:
        return _rank_flint(matrix)
    return _rank_sympy(matrix)


def basic_columns(matrix: RationalMatrix) -> List[int]:
    """Find pivot column indices from RREF."""
    if FLINT_AVAILABLE:
        return _basic_columns_flint(matrix)
    return _basic_columns_sympy(matrix)


def basic_columns_from_numpy(mx: np.ndarray) -> List[int]:
    """Find basic columns of numpy array using exact arithmetic."""
    mat = RationalMatrix.from_numpy(mx)
    return basic_columns(mat)


# =============================================================================
# Private FLINT implementations
# =============================================================================

def _nullspace_flint(matrix: RationalMatrix) -> RationalMatrix:
    """Compute nullspace using FLINT's native RREF."""
    mat = matrix._mat
    rows, cols = mat.nrows(), mat.ncols()

    # Copy matrix to avoid modifying original
    work = fmpq_mat(rows, cols)
    for r in range(rows):
        for c in range(cols):
            work[r, c] = mat[r, c]

    # Use FLINT's native RREF - O(n³) in C, very fast
    rref, rk = work.rref()

    # Find pivot columns (columns with leading 1s in RREF)
    pivot_cols = []
    for r in range(min(rows, rk)):
        for c in range(cols):
            if rref[r, c] != fmpq(0):
                pivot_cols.append(c)
                break

    # Free columns = all columns not in pivot_cols
    pivot_set = set(pivot_cols)
    free_cols = [c for c in range(cols) if c not in pivot_set]
    nullspace_dim = len(free_cols)

    if nullspace_dim == 0:
        # Trivial nullspace
        return RationalMatrix(cols, 0)

    # Build nullspace basis
    kernel = RationalMatrix(cols, nullspace_dim)
    for k, free_col in enumerate(free_cols):
        # Set 1 at free variable position
        kernel._mat[free_col, k] = fmpq(1)

        # Set -RREF values at pivot variable positions
        for i, pivot_col in enumerate(pivot_cols):
            val = rref[i, free_col]
            if val != fmpq(0):
                kernel._mat[pivot_col, k] = -val

    return kernel


def _rank_flint(matrix: RationalMatrix) -> int:
    """Compute rank using FLINT."""
    return matrix._mat.rank()


def _basic_columns_flint(matrix: RationalMatrix) -> List[int]:
    """Find basic columns using FLINT's RREF."""
    mat = matrix._mat
    rows, cols = mat.nrows(), mat.ncols()

    # Copy and compute RREF
    work = fmpq_mat(rows, cols)
    for r in range(rows):
        for c in range(cols):
            work[r, c] = mat[r, c]

    rref, rk = work.rref()

    # Find pivot columns
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
    """Compute nullspace using sympy."""
    rows, cols = matrix.get_row_count(), matrix.get_column_count()

    # Convert to sympy Matrix
    sympy_data = []
    for r in range(rows):
        row_data = []
        for c in range(cols):
            row_data.append(matrix.get_sympy_value(r, c))
        sympy_data.append(row_data)

    sympy_mat = SympyMatrix(sympy_data)

    # Compute nullspace
    null_vecs = sympy_mat.nullspace()

    if not null_vecs:
        return RationalMatrix(cols, 0)

    # Convert to RationalMatrix
    nullspace_dim = len(null_vecs)
    result = RationalMatrix(cols, nullspace_dim)

    for k, vec in enumerate(null_vecs):
        for r in range(cols):
            val = vec[r]
            if val != 0:
                result.set_rational(r, k, int(val.p), int(val.q))

    return result


def _rank_sympy(matrix: RationalMatrix) -> int:
    """Compute rank using sympy."""
    rows, cols = matrix.get_row_count(), matrix.get_column_count()

    # Convert to sympy Matrix
    sympy_data = []
    for r in range(rows):
        row_data = []
        for c in range(cols):
            row_data.append(matrix.get_sympy_value(r, c))
        sympy_data.append(row_data)

    sympy_mat = SympyMatrix(sympy_data)
    return sympy_mat.rank()


def _basic_columns_sympy(matrix: RationalMatrix) -> List[int]:
    """Find basic columns using sympy's RREF."""
    rows, cols = matrix.get_row_count(), matrix.get_column_count()

    # Convert to sympy Matrix
    sympy_data = []
    for r in range(rows):
        row_data = []
        for c in range(cols):
            row_data.append(matrix.get_sympy_value(r, c))
        sympy_data.append(row_data)

    sympy_mat = SympyMatrix(sympy_data)

    # rref() returns (reduced_matrix, pivot_columns_tuple)
    _, pivot_cols = sympy_mat.rref()

    return list(pivot_cols)


# =============================================================================
# Public exports
# =============================================================================

__all__ = [
    'FLINT_AVAILABLE',
    'Rational',
    'RationalMatrix',
    'nullspace',
    'rank',
    'basic_columns',
    'basic_columns_from_numpy',
]
