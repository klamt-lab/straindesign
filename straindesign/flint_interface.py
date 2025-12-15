"""
Rational arithmetic interface for straindesign.

Backend: FLINT (fast) or sympy (fallback).
All FLINT-specific code is contained in this module.

When FLINT is available, uses fractions.Fraction + fmpq for fast exact arithmetic.
Falls back to sympy.Rational when FLINT is not available.

Example usage:
    >>> from straindesign.flint_interface import RationalMatrix, nullspace
    >>> mat = RationalMatrix.from_numpy(stoich_array)
    >>> kernel = nullspace(mat)
"""

from typing import List, Tuple
import numpy as np
from fractions import Fraction
from math import gcd

# Backend detection - ONLY place flint is imported in entire straindesign package
try:
    from flint import fmpq, fmpq_mat
    FLINT_AVAILABLE = True
except ImportError:
    FLINT_AVAILABLE = False
    fmpq = None
    fmpq_mat = None

# Sympy fallback (only imported when needed)
_sympy_loaded = False
Rational = None
SympyMatrix = None


def _load_sympy():
    """Lazy load sympy only when needed (FLINT not available)."""
    global _sympy_loaded, Rational, SympyMatrix
    if not _sympy_loaded:
        from sympy import Rational as _Rational, Matrix as _Matrix
        Rational = _Rational
        SympyMatrix = _Matrix
        _sympy_loaded = True


def float_to_rational(val, max_precision: int = 6, max_denom: int = 100):
    """Convert numeric value to Fraction with bounded denominators.

    Strategy:
    1. Try limit_denominator(max_denom) for small fractions like 1/3, 5/11
    2. Check if it reconstructs correctly at given precision
    3. If not, use power of 10 (auto-reduces to only 2,5 factors)

    This ensures all denominators have only factors 2, 3, 5 (for denominators > max_denom),
    which keeps LCM bounded when clearing denominators row-wise.

    Args:
        val: Numeric value to convert
        max_precision: Maximum decimal precision to preserve (default 6)
        max_denom: Maximum denominator for "nice" fractions (default 100)

    Returns:
        Fraction with bounded denominator
    """
    if isinstance(val, Fraction):
        return val
    if val == 0:
        return Fraction(0)
    if val == int(val):
        return Fraction(int(val))

    # Try small denominator first
    small_frac = Fraction(val).limit_denominator(max_denom)
    if round(float(small_frac), max_precision) == round(val, max_precision):
        return small_frac

    # Fallback: power of 10 (auto-reduces to only 2,5 factors)
    denom = 10 ** max_precision
    numer = round(val * denom)
    return Fraction(numer, denom)


def float_to_fmpq(val: float, max_precision: int = 6, max_denom: int = 100) -> 'fmpq':
    """Convert float to FLINT fmpq using bounded rational approximation."""
    f = float_to_rational(val, max_precision, max_denom)
    return fmpq(f.numerator, f.denominator)


def float_to_rational_sympy(val: float, max_precision: int = 6, max_denom: int = 100) -> 'Rational':
    """Convert float to sympy.Rational using bounded rational approximation."""
    _load_sympy()
    f = float_to_rational(val, max_precision, max_denom)
    return Rational(f.numerator, f.denominator)


def fmpq_to_float(val: 'fmpq') -> float:
    """Convert FLINT fmpq to Python float."""
    return float(val.p) / float(val.q)


def detect_max_precision(arr: np.ndarray) -> int:
    """Detect maximum decimal precision in array coefficients.

    Examines non-zero values to find the maximum number of decimal places.
    Returns the precision as an integer (e.g., 6 for 6 decimal places).

    The precision is clamped to [3, 12] range.
    """
    max_decimals = 0
    for val in arr.flat:
        if val != 0 and val != int(val):
            # Count decimal places in string representation
            s = f"{abs(val):.15g}"  # 15 significant digits
            if '.' in s:
                # Remove trailing zeros and count decimals
                decimals = len(s.split('.')[1].rstrip('0'))
                max_decimals = max(max_decimals, decimals)

    # Clamp to reasonable range: minimum 3, maximum 12
    return min(12, max(3, max_decimals))


def _lcm(a: int, b: int) -> int:
    """Compute least common multiple of two integers."""
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(a, b)


def _lcm_list(numbers: List[int]) -> int:
    """Compute LCM of a list of integers."""
    if not numbers:
        return 1
    result = numbers[0]
    for n in numbers[1:]:
        result = _lcm(result, n)
    return result


def _gcd_list(numbers: List[int]) -> int:
    """Compute GCD of a list of integers."""
    if not numbers:
        return 1
    result = numbers[0]
    for n in numbers[1:]:
        result = gcd(result, n)
        if result == 1:
            return 1
    return result


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
            _load_sympy()
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
            result.set_rational(i, i, 1, 1)
        return result

    @classmethod
    def from_numpy(cls, arr: np.ndarray, max_precision: int = None,
                   max_denom: int = 100) -> 'RationalMatrix':
        """Create RationalMatrix from numpy array.

        Args:
            arr: Input numpy array
            max_precision: Maximum decimal precision. If None, auto-detected.
            max_denom: Maximum denominator for "nice" fractions (default 100)
        """
        rows, cols = arr.shape

        if max_precision is None:
            max_precision = detect_max_precision(arr)

        if FLINT_AVAILABLE:
            # Fast path: directly create fmpq_mat
            mat = fmpq_mat(rows, cols)
            for r in range(rows):
                for c in range(cols):
                    val = arr[r, c]
                    if val != 0:
                        mat[r, c] = float_to_fmpq(val, max_precision, max_denom)
            return cls._from_flint(mat)
        else:
            # Sympy fallback
            result = cls(rows, cols)
            for r in range(rows):
                for c in range(cols):
                    val = arr[r, c]
                    if val != 0:
                        result.set_value(r, c, float_to_rational_sympy(val, max_precision, max_denom))
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

    def submatrix(self, rows: int, cols: int) -> 'RationalMatrix':
        """Create a new matrix containing the top-left submatrix.

        Args:
            rows: Number of rows in submatrix (must be <= self._rows)
            cols: Number of columns in submatrix (must be <= self._cols)

        Returns:
            New RationalMatrix with dimensions rows x cols
        """
        if rows > self._rows or cols > self._cols:
            raise ValueError(f"Submatrix ({rows}x{cols}) exceeds matrix size ({self._rows}x{self._cols})")

        if self._is_flint:
            new_mat = fmpq_mat(rows, cols)
            for r in range(rows):
                for c in range(cols):
                    new_mat[r, c] = self._mat[r, c]
            return RationalMatrix._from_flint(new_mat)
        else:
            data = []
            for r in range(rows):
                for c in range(cols):
                    data.append(self._data[r * self._cols + c])
            return RationalMatrix._from_sympy_data(data, rows, cols)

    # --- Conversion ---

    def to_numpy(self) -> np.ndarray:
        result = np.zeros((self._rows, self._cols), dtype=float)
        for r in range(self._rows):
            for c in range(self._cols):
                result[r, c] = float(self.get_numerator(r, c)) / float(self.get_denominator(r, c))
        return result

    def to_numpy_int(self) -> np.ndarray:
        """Convert to numpy array, assuming all entries are integers.

        Raises ValueError if any entry has denominator != 1.
        """
        result = np.zeros((self._rows, self._cols), dtype=np.int64)
        for r in range(self._rows):
            for c in range(self._cols):
                den = self.get_denominator(r, c)
                if den != 1:
                    raise ValueError(f"Entry [{r},{c}] has denominator {den} != 1")
                result[r, c] = self.get_numerator(r, c)
        return result

    def clear_denominators_rowwise(self) -> List[int]:
        """Clear denominators row-wise by multiplying each row by LCM of its denominators.

        Modifies the matrix in-place. After this operation, all entries are integers
        (denominator = 1). Also reduces each row by GCD to get smallest integers.

        Returns:
            List of net row scaling factors (LCM / GCD for each row).
        """
        row_scalers = []
        for r in range(self._rows):
            # Collect all denominators in this row
            denominators = []
            for c in range(self._cols):
                den = self.get_denominator(r, c)
                if den != 1:
                    denominators.append(den)

            # Compute LCM of denominators
            if not denominators:
                # All denominators are 1, but still reduce by GCD of numerators
                numerators = [abs(self.get_numerator(r, c)) for c in range(self._cols)
                              if self.get_numerator(r, c) != 0]
                if numerators:
                    row_gcd = _gcd_list(numerators)
                    if row_gcd > 1:
                        for c in range(self._cols):
                            num = self.get_numerator(r, c)
                            self.set_rational(r, c, num // row_gcd, 1)
                row_scalers.append(1)
                continue

            lcm = _lcm_list(denominators)

            # Multiply each entry in row by lcm to clear denominators
            for c in range(self._cols):
                num = self.get_numerator(r, c)
                den = self.get_denominator(r, c)
                # new_num = num * (lcm / den), new_den = 1
                new_num = num * (lcm // den)
                self.set_rational(r, c, new_num, 1)

            # Now reduce by GCD of the row to get smallest integers
            numerators = [abs(self.get_numerator(r, c)) for c in range(self._cols)
                          if self.get_numerator(r, c) != 0]
            if numerators:
                row_gcd = _gcd_list(numerators)
                if row_gcd > 1:
                    for c in range(self._cols):
                        num = self.get_numerator(r, c)
                        self.set_rational(r, c, num // row_gcd, 1)
                    row_scalers.append(lcm // row_gcd)
                else:
                    row_scalers.append(lcm)
            else:
                row_scalers.append(lcm)

        return row_scalers

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
    _load_sympy()
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
    _load_sympy()
    rows, cols = matrix.get_row_count(), matrix.get_column_count()
    sympy_data = [[matrix.get_value(r, c) for c in range(cols)] for r in range(rows)]
    return SympyMatrix(sympy_data).rank()


def _basic_columns_sympy(matrix: RationalMatrix) -> List[int]:
    _load_sympy()
    rows, cols = matrix.get_row_count(), matrix.get_column_count()
    sympy_data = [[matrix.get_value(r, c) for c in range(cols)] for r in range(rows)]
    _, pivot_cols = SympyMatrix(sympy_data).rref()
    return list(pivot_cols)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'FLINT_AVAILABLE',
    'RationalMatrix',
    'float_to_rational',
    'float_to_fmpq',
    'float_to_rational_sympy',
    'detect_max_precision',
    'nullspace',
    'rank',
    'basic_columns',
    'basic_columns_from_numpy',
]
