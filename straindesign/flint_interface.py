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

from typing import List, Tuple, Optional, Iterator
import numpy as np
from fractions import Fraction
from math import gcd
from scipy.sparse import csr_matrix, csc_matrix

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
# Rational Matrix (Sparse backbone with dual representation)
# =============================================================================

class RationalMatrix:
    """
    Matrix of exact rationals with sparse backbone.

    When FLINT is available, stores two representations:
    1. Scaled form: _scaled / _common_denom (for slicing, pattern building, rref conversion)
    2. Per-element: _numerators / _denominators (for fast Fraction creation)

    fmpq_mat is only used for rref() computation. All other operations use sparse matrices.
    Falls back to sympy.Rational when FLINT is not available.
    """

    def __init__(self, rows: int, cols: int):
        """Create zero matrix with given dimensions."""
        self._rows = rows
        self._cols = cols

        if FLINT_AVAILABLE:
            self._is_flint = True
            # Dual sparse representation
            self._scaled: Optional[csr_matrix] = csr_matrix((rows, cols), dtype=np.int64)
            self._common_denom: int = 1
            self._numerators: csr_matrix = csr_matrix((rows, cols), dtype=np.int64)
            self._denominators: csr_matrix = csr_matrix((rows, cols), dtype=np.int64)
            # CSC cache for column iteration
            self._csc_num: Optional[csc_matrix] = None
            self._csc_den: Optional[csc_matrix] = None
        else:
            _load_sympy()
            self._data = [Rational(0)] * (rows * cols)
            self._is_flint = False

    def _invalidate_cache(self):
        """Invalidate CSC cache after modification."""
        if self._is_flint:
            self._csc_num = None
            self._csc_den = None

    def begin_batch_edit(self) -> None:
        """Enable batch modification mode for efficient column operations.

        Call this before making many add_scaled_column calls,
        then call end_batch_edit() when done. This caches CSC format for reads
        and accumulates changes in a dictionary to avoid repeated conversions.
        """
        if self._is_flint and not hasattr(self, '_in_batch_edit'):
            self._in_batch_edit = False

        if self._is_flint and not self._in_batch_edit:
            # Cache CSC format for fast column reads
            self._batch_csc_num = self._numerators.tocsc()
            self._batch_csc_den = self._denominators.tocsc()
            # Track column modifications: {col: {row: Fraction}}
            self._batch_changes = {}
            self._in_batch_edit = True
            self._invalidate_cache()

    def end_batch_edit(self) -> None:
        """Apply accumulated changes and switch back to CSR format."""
        if self._is_flint and getattr(self, '_in_batch_edit', False):
            if self._batch_changes:
                # Apply changes to DOK format for efficient updates
                dok_num = self._numerators.todok()
                dok_den = self._denominators.todok()

                # col_changes contains the COMPLETE new state of each column
                # First clear original entries in modified columns
                csc_num = self._batch_csc_num  # Original data
                for col in self._batch_changes.keys():
                    # Remove all original entries in this column
                    start = csc_num.indptr[col]
                    end = csc_num.indptr[col + 1]
                    for idx in range(start, end):
                        row = csc_num.indices[idx]
                        if (row, col) in dok_num:
                            del dok_num[row, col]
                            del dok_den[row, col]

                # Now set the new values
                for col, col_changes in self._batch_changes.items():
                    for row, frac in col_changes.items():
                        if frac != 0:
                            dok_num[row, col] = frac.numerator
                            dok_den[row, col] = frac.denominator

                self._numerators = dok_num.tocsr()
                self._denominators = dok_den.tocsr()
                self._scaled = None
                self._common_denom = None

            del self._batch_csc_num
            del self._batch_csc_den
            del self._batch_changes
            self._in_batch_edit = False
            self._invalidate_cache()

    @classmethod
    def _from_sparse(cls, numerators: csr_matrix, denominators: csr_matrix,
                     scaled: Optional[csr_matrix], common_denom: Optional[int]) -> 'RationalMatrix':
        """Create RationalMatrix from sparse components."""
        obj = object.__new__(cls)
        obj._rows = numerators.shape[0]
        obj._cols = numerators.shape[1]
        obj._is_flint = True
        obj._numerators = numerators
        obj._denominators = denominators
        obj._scaled = scaled
        obj._common_denom = common_denom
        obj._csc_num = None
        obj._csc_den = None
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
        """Create identity matrix."""
        if FLINT_AVAILABLE:
            diag_indices = np.arange(size)
            ones = np.ones(size, dtype=np.int64)

            numerators = csr_matrix(
                (ones, (diag_indices, diag_indices)),
                shape=(size, size), dtype=np.int64
            )
            denominators = csr_matrix(
                (ones, (diag_indices, diag_indices)),
                shape=(size, size), dtype=np.int64
            )
            scaled = csr_matrix(
                (ones, (diag_indices, diag_indices)),
                shape=(size, size), dtype=np.int64
            )
            return cls._from_sparse(numerators, denominators, scaled, 1)
        else:
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
            # Find nonzeros
            nz_rows, nz_cols = np.nonzero(arr)
            nz_vals = arr[nz_rows, nz_cols]

            # Convert each value to rational
            numerators = []
            denominators = []
            for val in nz_vals:
                frac = float_to_rational(val, max_precision, max_denom)
                numerators.append(frac.numerator)
                denominators.append(frac.denominator)

            return cls._build_from_sparse_data(
                list(nz_rows), list(nz_cols), numerators, denominators,
                rows, cols
            )
        else:
            # Sympy fallback
            result = cls(rows, cols)
            for r in range(rows):
                for c in range(cols):
                    val = arr[r, c]
                    if val != 0:
                        result.set_value(r, c, float_to_rational_sympy(val, max_precision, max_denom))
            return result

    @classmethod
    def from_cobra_model(cls, model, max_precision: int = 6, max_denom: int = 100) -> 'RationalMatrix':
        """Create RationalMatrix from COBRA model stoichiometry.

        This is optimized to:
        1. Iterate nonzeros directly (no dense array)
        2. Fast-path integers (most coefficients are ±1, ±2, etc.)
        3. Build sparse matrices in one shot

        Args:
            model: COBRA model
            max_precision: Maximum decimal precision for float conversion
            max_denom: Maximum denominator for "nice" fractions

        Returns:
            RationalMatrix with exact stoichiometric coefficients
        """
        num_mets = len(model.metabolites)
        num_rxns = len(model.reactions)

        if not FLINT_AVAILABLE:
            _load_sympy()
            result = cls(num_mets, num_rxns)
            for j, rxn in enumerate(model.reactions):
                for met, coeff in rxn.metabolites.items():
                    i = model.metabolites.index(met.id)
                    if hasattr(coeff, 'p'):  # sympy.Rational
                        result.set_value(i, j, Rational(int(coeff.p), int(coeff.q)))
                    elif hasattr(coeff, 'numerator'):  # fractions.Fraction
                        result.set_value(i, j, Rational(int(coeff.numerator), int(coeff.denominator)))
                    else:
                        result.set_value(i, j, float_to_rational_sympy(coeff, max_precision, max_denom))
            return result

        # Collect all nonzeros in one pass
        rows = []
        cols = []
        numerators = []
        denominators = []

        for j, rxn in enumerate(model.reactions):
            for met, coeff in rxn.metabolites.items():
                i = model.metabolites.index(met.id)
                rows.append(i)
                cols.append(j)

                # Fast path for integers (most common case)
                if isinstance(coeff, (int, np.integer)):
                    numerators.append(int(coeff))
                    denominators.append(1)
                elif hasattr(coeff, 'p'):  # sympy.Rational
                    numerators.append(int(coeff.p))
                    denominators.append(int(coeff.q))
                elif hasattr(coeff, 'numerator'):  # fractions.Fraction
                    numerators.append(int(coeff.numerator))
                    denominators.append(int(coeff.denominator))
                elif float(coeff) == int(coeff):  # Float that's actually integer
                    numerators.append(int(coeff))
                    denominators.append(1)
                else:
                    # Need float_to_rational conversion
                    frac = float_to_rational(coeff, max_precision, max_denom)
                    numerators.append(frac.numerator)
                    denominators.append(frac.denominator)

        return cls._build_from_sparse_data(rows, cols, numerators, denominators, num_mets, num_rxns)

    @classmethod
    def _build_from_sparse_data(cls, rows: list, cols: list,
                                 numerators: list, denominators: list,
                                 num_rows: int, num_cols: int) -> 'RationalMatrix':
        """Build RationalMatrix from sparse coordinate data in one shot.

        This avoids the O(nnz * n) cost of element-by-element set_rational calls.
        """
        if not numerators:
            return cls(num_rows, num_cols)

        # Build per-element representation
        num_sparse = csr_matrix(
            (numerators, (rows, cols)),
            shape=(num_rows, num_cols), dtype=np.int64
        )
        den_sparse = csr_matrix(
            (denominators, (rows, cols)),
            shape=(num_rows, num_cols), dtype=np.int64
        )

        # Build denominator-cleared representation
        common_denom = _lcm_list(denominators)
        if common_denom > 2**62:  # Leave headroom for arithmetic
            # LCM overflow - skip scaled representation
            scaled = None
            common_denom = None
        else:
            scaled_vals = [n * (common_denom // d) for n, d in zip(numerators, denominators)]
            scaled = csr_matrix(
                (scaled_vals, (rows, cols)),
                shape=(num_rows, num_cols), dtype=np.int64
            )
            common_denom = int(common_denom)

        return cls._from_sparse(num_sparse, den_sparse, scaled, common_denom)

    # --- Dimensions ---

    def get_row_count(self) -> int:
        return self._rows

    def get_column_count(self) -> int:
        return self._cols

    # --- Element access ---

    def get_value(self, row: int, col: int):
        """Get sympy.Rational value at position."""
        if self._is_flint:
            num = int(self._numerators[row, col])
            if num == 0:
                _load_sympy()
                return Rational(0)
            den = int(self._denominators[row, col])
            _load_sympy()
            return Rational(num, den)
        else:
            return self._data[row * self._cols + col]

    def get_numerator(self, row: int, col: int) -> int:
        if self._is_flint:
            return int(self._numerators[row, col])
        else:
            return int(self._data[row * self._cols + col].p)

    def get_denominator(self, row: int, col: int) -> int:
        if self._is_flint:
            den = int(self._denominators[row, col])
            return den if den != 0 else 1  # Zero entries have implicit denominator 1
        else:
            return int(self._data[row * self._cols + col].q)

    def get_fraction(self, row: int, col: int) -> Fraction:
        """Get Fraction for COBRA - uses per-element representation."""
        if self._is_flint:
            num = int(self._numerators[row, col])
            if num == 0:
                return Fraction(0)
            den = int(self._denominators[row, col])
            return Fraction(num, den)
        else:
            val = self._data[row * self._cols + col]
            return Fraction(int(val.p), int(val.q))

    def get_signum(self, row: int, col: int) -> int:
        num = self.get_numerator(row, col)
        if num == 0:
            return 0
        return 1 if num > 0 else -1

    def set_value(self, row: int, col: int, value) -> None:
        """Set value from sympy.Rational."""
        if self._is_flint:
            self.set_rational(row, col, int(value.p), int(value.q))
        else:
            self._data[row * self._cols + col] = value

    def set_rational(self, row: int, col: int, num: int, den: int) -> None:
        """Set value from numerator/denominator.

        Note: This invalidates the scaled representation if denominators change.
        For bulk modifications, consider rebuilding the matrix.
        """
        if self._is_flint:
            # Convert to lil for efficient update
            num_lil = self._numerators.tolil()
            den_lil = self._denominators.tolil()
            num_lil[row, col] = num
            den_lil[row, col] = den if num != 0 else 0
            self._numerators = num_lil.tocsr()
            self._denominators = den_lil.tocsr()

            # Invalidate scaled representation
            self._scaled = None
            self._common_denom = None
            self._invalidate_cache()
        else:
            self._data[row * self._cols + col] = Rational(num, den)

    def get_row_zero_pattern(self, row: int) -> tuple:
        """Get zero pattern for a row as tuple of booleans.

        Returns tuple where True means the element is zero.
        """
        if self._is_flint:
            nonzero_set = set(self.get_row_nonzero_indices(row))
            return tuple(c not in nonzero_set for c in range(self._cols))
        else:
            return tuple(self._data[row * self._cols + c] == 0 for c in range(self._cols))

    def get_row_nonzero_indices(self, row: int) -> List[int]:
        """Get indices of non-zero columns in a row - O(nnz_row)."""
        if self._is_flint:
            start = self._numerators.indptr[row]
            end = self._numerators.indptr[row + 1]
            return list(self._numerators.indices[start:end])
        else:
            return [c for c in range(self._cols) if self._data[row * self._cols + c] != 0]

    # --- Iteration methods for efficient COBRA writing ---

    def iter_row_fractions(self, row: int) -> Iterator[Tuple[int, Fraction]]:
        """Iterate (col, Fraction) pairs for nonzeros in a row."""
        if self._is_flint:
            start = self._numerators.indptr[row]
            end = self._numerators.indptr[row + 1]
            for idx in range(start, end):
                col = self._numerators.indices[idx]
                num = self._numerators.data[idx]
                den = self._denominators.data[idx]
                yield col, Fraction(int(num), int(den))
        else:
            for c in range(self._cols):
                val = self._data[row * self._cols + c]
                if val != 0:
                    yield c, Fraction(int(val.p), int(val.q))

    def iter_column_fractions(self, col: int) -> Iterator[Tuple[int, Fraction]]:
        """Iterate (row, Fraction) pairs for nonzeros in a column."""
        if self._is_flint:
            # Build CSC cache if needed
            if self._csc_num is None:
                self._csc_num = self._numerators.tocsc()
                self._csc_den = self._denominators.tocsc()

            start = self._csc_num.indptr[col]
            end = self._csc_num.indptr[col + 1]
            for idx in range(start, end):
                row = self._csc_num.indices[idx]
                num = self._csc_num.data[idx]
                den = self._csc_den.data[idx]
                yield row, Fraction(int(num), int(den))
        else:
            for r in range(self._rows):
                val = self._data[r * self._cols + col]
                if val != 0:
                    yield r, Fraction(int(val.p), int(val.q))

    # --- Row/column operations ---

    def swap_rows(self, i: int, j: int) -> None:
        if i == j:
            return
        if self._is_flint:
            # Efficient row swap using row permutation
            perm = np.arange(self._rows)
            perm[i], perm[j] = j, i

            self._numerators = self._numerators[perm, :]
            self._denominators = self._denominators[perm, :]
            if self._scaled is not None:
                self._scaled = self._scaled[perm, :]
            self._invalidate_cache()
        else:
            for c in range(self._cols):
                idx_i, idx_j = i * self._cols + c, j * self._cols + c
                self._data[idx_i], self._data[idx_j] = self._data[idx_j], self._data[idx_i]

    def swap_columns(self, i: int, j: int) -> None:
        if i == j:
            return
        if self._is_flint:
            # Efficient column swap using column permutation
            # Build permutation: all columns same except i<->j
            perm = np.arange(self._cols)
            perm[i], perm[j] = j, i

            # Apply permutation to columns via indexing
            self._numerators = self._numerators[:, perm]
            self._denominators = self._denominators[:, perm]
            if self._scaled is not None:
                self._scaled = self._scaled[:, perm]
            self._invalidate_cache()
        else:
            for r in range(self._rows):
                idx_i, idx_j = r * self._cols + i, r * self._cols + j
                self._data[idx_i], self._data[idx_j] = self._data[idx_j], self._data[idx_i]

    def add_scaled_column(self, dst_col: int, src_col: int, scalar_num: int, scalar_den: int) -> None:
        """In-place: column[dst] += column[src] * (scalar_num/scalar_den)."""
        if self._is_flint:
            scalar = Fraction(scalar_num, scalar_den)

            # Check if we're in batch edit mode
            in_batch = getattr(self, '_in_batch_edit', False)

            if in_batch:
                # Use cached CSC format for column reads
                csc_num = self._batch_csc_num
                csc_den = self._batch_csc_den

                # Get source column data (check pending changes first)
                if src_col in self._batch_changes:
                    src_data = {r: v * scalar for r, v in self._batch_changes[src_col].items()}
                else:
                    src_start = csc_num.indptr[src_col]
                    src_end = csc_num.indptr[src_col + 1]

                    if src_start == src_end:
                        return  # Empty source column

                    src_data = {}
                    for idx in range(src_start, src_end):
                        row = csc_num.indices[idx]
                        num = int(csc_num.data[idx])
                        den = int(csc_den.data[idx])
                        src_data[row] = Fraction(num, den) * scalar

                if not src_data:
                    return

                # Get current destination state (from changes or original)
                dst_data = {}
                if dst_col in self._batch_changes:
                    # Start with accumulated changes
                    dst_data = dict(self._batch_changes[dst_col])
                else:
                    # Start with original data
                    dst_start = csc_num.indptr[dst_col]
                    dst_end = csc_num.indptr[dst_col + 1]
                    for idx in range(dst_start, dst_end):
                        row = csc_num.indices[idx]
                        num = int(csc_num.data[idx])
                        den = int(csc_den.data[idx])
                        dst_data[row] = Fraction(num, den)

                # Compute result
                for row, val in src_data.items():
                    dst_data[row] = dst_data.get(row, Fraction(0)) + val

                # Remove zeros
                dst_data = {r: v for r, v in dst_data.items() if v != 0}

                # Store in changes dict
                self._batch_changes[dst_col] = dst_data
            else:
                # Slow path for single operations (original implementation)
                csc_num = self._numerators.tocsc()
                csc_den = self._denominators.tocsc()

                src_start = csc_num.indptr[src_col]
                src_end = csc_num.indptr[src_col + 1]

                if src_start == src_end:
                    return

                src_data = {}
                for idx in range(src_start, src_end):
                    row = csc_num.indices[idx]
                    num = int(csc_num.data[idx])
                    den = int(csc_den.data[idx])
                    src_data[row] = Fraction(num, den) * scalar

                dst_start = csc_num.indptr[dst_col]
                dst_end = csc_num.indptr[dst_col + 1]
                dst_data = {}
                for idx in range(dst_start, dst_end):
                    row = csc_num.indices[idx]
                    num = int(csc_num.data[idx])
                    den = int(csc_den.data[idx])
                    dst_data[row] = Fraction(num, den)

                all_rows = set(src_data.keys()) | set(dst_data.keys())
                result_data = {}
                for row in all_rows:
                    val = dst_data.get(row, Fraction(0)) + src_data.get(row, Fraction(0))
                    if val != 0:
                        result_data[row] = val

                # Convert to LIL and update
                num_lil = self._numerators.tolil()
                den_lil = self._denominators.tolil()

                for idx in range(dst_start, dst_end):
                    row = csc_num.indices[idx]
                    num_lil[row, dst_col] = 0
                    den_lil[row, dst_col] = 0

                for row, frac in result_data.items():
                    num_lil[row, dst_col] = frac.numerator
                    den_lil[row, dst_col] = frac.denominator

                self._numerators = num_lil.tocsr()
                self._denominators = den_lil.tocsr()
                self._scaled = None
                self._common_denom = None
                self._invalidate_cache()
        else:
            scalar = Rational(scalar_num, scalar_den)
            for r in range(self._rows):
                idx_dst = r * self._cols + dst_col
                idx_src = r * self._cols + src_col
                self._data[idx_dst] += self._data[idx_src] * scalar

    def add_scaled_row(self, dst_row: int, src_row: int, scalar_num: int, scalar_den: int) -> None:
        """In-place: row[dst] += row[src] * (scalar_num/scalar_den)."""
        if self._is_flint:
            scalar = Fraction(scalar_num, scalar_den)
            num_lil = self._numerators.tolil()
            den_lil = self._denominators.tolil()

            for c in range(self._cols):
                src_num = int(self._numerators[src_row, c])
                if src_num == 0:
                    continue
                src_den = int(self._denominators[src_row, c])
                src_frac = Fraction(src_num, src_den) * scalar

                dst_num = int(self._numerators[dst_row, c])
                if dst_num == 0:
                    dst_frac = Fraction(0)
                else:
                    dst_den = int(self._denominators[dst_row, c])
                    dst_frac = Fraction(dst_num, dst_den)

                result = dst_frac + src_frac
                if result == 0:
                    num_lil[dst_row, c] = 0
                    den_lil[dst_row, c] = 0
                else:
                    num_lil[dst_row, c] = result.numerator
                    den_lil[dst_row, c] = result.denominator

            self._numerators = num_lil.tocsr()
            self._denominators = den_lil.tocsr()
            self._scaled = None
            self._common_denom = None
            self._invalidate_cache()
        else:
            scalar = Rational(scalar_num, scalar_den)
            for c in range(self._cols):
                idx_dst = dst_row * self._cols + c
                idx_src = src_row * self._cols + c
                self._data[idx_dst] += self._data[idx_src] * scalar

    # --- Matrix operations ---

    def clone(self) -> 'RationalMatrix':
        """Create a deep copy."""
        if self._is_flint:
            scaled = self._scaled.copy() if self._scaled is not None else None
            return RationalMatrix._from_sparse(
                self._numerators.copy(),
                self._denominators.copy(),
                scaled,
                self._common_denom
            )
        else:
            return RationalMatrix._from_sympy_data(self._data[:], self._rows, self._cols)

    def submatrix(self, rows: int, cols: int) -> 'RationalMatrix':
        """Create a new matrix containing the top-left submatrix - O(nnz) slicing.

        Args:
            rows: Number of rows in submatrix (must be <= self._rows)
            cols: Number of columns in submatrix (must be <= self._cols)

        Returns:
            New RationalMatrix with dimensions rows x cols
        """
        if rows > self._rows or cols > self._cols:
            raise ValueError(f"Submatrix ({rows}x{cols}) exceeds matrix size ({self._rows}x{self._cols})")

        if self._is_flint:
            num_sub = self._numerators[:rows, :cols].copy()
            den_sub = self._denominators[:rows, :cols].copy()

            if self._scaled is not None:
                scaled_sub = self._scaled[:rows, :cols].copy()
                common_denom = self._common_denom
            else:
                scaled_sub = None
                common_denom = None

            return RationalMatrix._from_sparse(num_sub, den_sub, scaled_sub, common_denom)
        else:
            data = []
            for r in range(rows):
                for c in range(cols):
                    data.append(self._data[r * self._cols + c])
            return RationalMatrix._from_sympy_data(data, rows, cols)

    def remove_columns(self, keep_indices: List[int]) -> None:
        """Batch column removal - O(nnz)."""
        if self._is_flint:
            keep = np.array(keep_indices)
            self._numerators = self._numerators[:, keep]
            self._denominators = self._denominators[:, keep]
            if self._scaled is not None:
                self._scaled = self._scaled[:, keep]
            self._cols = len(keep_indices)
            self._invalidate_cache()
        else:
            new_cols = len(keep_indices)
            new_data = []
            for r in range(self._rows):
                for c in keep_indices:
                    new_data.append(self._data[r * self._cols + c])
            self._data = new_data
            self._cols = new_cols

    def remove_rows(self, keep_indices: List[int]) -> None:
        """Batch row removal - O(nnz)."""
        if self._is_flint:
            keep = np.array(keep_indices)
            self._numerators = self._numerators[keep, :]
            self._denominators = self._denominators[keep, :]
            if self._scaled is not None:
                self._scaled = self._scaled[keep, :]
            self._rows = len(keep_indices)
            self._invalidate_cache()
        else:
            new_rows = len(keep_indices)
            new_data = []
            for r in keep_indices:
                for c in range(self._cols):
                    new_data.append(self._data[r * self._cols + c])
            self._data = new_data
            self._rows = new_rows

    def to_sparse_csr(self) -> Tuple[csr_matrix, int]:
        """Convert to scipy.sparse CSR with common denominator.

        Returns scaled form if available, otherwise computes it.

        Returns:
            Tuple of (sparse_matrix, common_denominator) where values are integers.
            Original rational = sparse_matrix[i,j] / common_denominator
        """
        if not self._is_flint:
            raise NotImplementedError("to_sparse_csr only supported for FLINT matrices")

        if self._scaled is not None:
            return self._scaled.copy(), self._common_denom

        # Compute on-the-fly
        den_coo = self._denominators.tocoo()
        if len(den_coo.data) == 0:
            return csr_matrix((self._rows, self._cols), dtype=np.int64), 1

        denoms = [int(d) for d in den_coo.data]
        common_denom = _lcm_list(denoms)

        num_coo = self._numerators.tocoo()
        if common_denom <= 2**62:
            scaled = [int(n) * (common_denom // int(d))
                     for n, d in zip(num_coo.data, den_coo.data)]
            sparse = csr_matrix(
                (scaled, (num_coo.row, num_coo.col)),
                shape=(self._rows, self._cols), dtype=np.int64
            )
            return sparse, common_denom
        else:
            raise ValueError("Common denominator overflow - cannot create scaled form")

    def to_fmpq_mat(self) -> 'fmpq_mat':
        """Convert to fmpq_mat for rref() computation."""
        if not FLINT_AVAILABLE:
            raise RuntimeError("FLINT not available")

        mat = fmpq_mat(self._rows, self._cols)

        if self._scaled is not None:
            # Fast path: use scaled representation
            coo = self._scaled.tocoo()
            for r, c, v in zip(coo.row, coo.col, coo.data):
                mat[r, c] = fmpq(int(v), self._common_denom)
        else:
            # Try to compute common denom on-the-fly
            den_coo = self._denominators.tocoo()
            if len(den_coo.data) == 0:
                return mat  # All zeros

            denoms = [int(d) for d in den_coo.data]
            common_denom = _lcm_list(denoms)

            num_coo = self._numerators.tocoo()
            if common_denom <= 2**62:
                # Can use common denominator
                for r, c, n, d in zip(num_coo.row, num_coo.col, num_coo.data, den_coo.data):
                    scaled_num = int(n) * (common_denom // int(d))
                    mat[r, c] = fmpq(scaled_num, common_denom)
            else:
                # Per-element (slowest)
                for r, c, n, d in zip(num_coo.row, num_coo.col, num_coo.data, den_coo.data):
                    mat[r, c] = fmpq(int(n), int(d))

        return mat

    @classmethod
    def from_fmpq_mat(cls, mat: 'fmpq_mat') -> 'RationalMatrix':
        """Convert from fmpq_mat after rref()."""
        if not FLINT_AVAILABLE:
            raise RuntimeError("FLINT not available")

        # Use numer_denom for efficient extraction
        int_mat, denom = mat.numer_denom()
        rows, cols = int_mat.nrows(), int_mat.ncols()

        # Find nonzeros and build sparse
        nz_rows, nz_cols, scaled_vals = [], [], []
        numerators, denominators = [], []

        common_denom = int(denom)

        for r in range(rows):
            for c in range(cols):
                val = int(int_mat[r, c])
                if val != 0:
                    nz_rows.append(r)
                    nz_cols.append(c)
                    scaled_vals.append(val)

                    # Reduce to lowest terms for per-element representation
                    g = gcd(abs(val), common_denom)
                    numerators.append(val // g)
                    denominators.append(common_denom // g)

        if nz_rows:
            scaled = csr_matrix(
                (scaled_vals, (nz_rows, nz_cols)),
                shape=(rows, cols), dtype=np.int64
            )
            num_sparse = csr_matrix(
                (numerators, (nz_rows, nz_cols)),
                shape=(rows, cols), dtype=np.int64
            )
            den_sparse = csr_matrix(
                (denominators, (nz_rows, nz_cols)),
                shape=(rows, cols), dtype=np.int64
            )
        else:
            scaled = csr_matrix((rows, cols), dtype=np.int64)
            num_sparse = csr_matrix((rows, cols), dtype=np.int64)
            den_sparse = csr_matrix((rows, cols), dtype=np.int64)
            common_denom = 1

        return cls._from_sparse(num_sparse, den_sparse, scaled, common_denom)

    # --- Conversion ---

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy float array."""
        result = np.zeros((self._rows, self._cols), dtype=float)
        if self._is_flint:
            coo = self._numerators.tocoo()
            for r, c, n in zip(coo.row, coo.col, coo.data):
                d = self._denominators[r, c]
                result[r, c] = float(n) / float(d)
        else:
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
        mat = matrix.to_fmpq_mat()
        return mat.rank()
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
    """Compute nullspace using sparse backbone.

    1. Convert sparse RationalMatrix to fmpq_mat
    2. Compute rref() using FLINT
    3. Build kernel directly in sparse format
    """
    # Convert sparse matrix to fmpq_mat for rref
    mat = matrix.to_fmpq_mat()
    rows, cols = mat.nrows(), mat.ncols()

    # rref() returns a new matrix, no need to copy first
    rref_mat, rk = mat.rref()

    # Reuse zero object for faster comparison
    zero = fmpq(0)

    # Find pivot columns
    pivot_cols = []
    for r in range(min(rows, rk)):
        for c in range(cols):
            if rref_mat[r, c] != zero:
                pivot_cols.append(c)
                break

    pivot_set = set(pivot_cols)
    free_cols = [c for c in range(cols) if c not in pivot_set]

    if not free_cols:
        return RationalMatrix(cols, 0)

    # Build kernel directly in sparse format
    # Use numer_denom to get common denominator efficiently
    rref_int, rref_denom = rref_mat.numer_denom()
    common_denom = int(rref_denom)

    # Collect sparse entries for kernel
    nz_rows, nz_cols, numerators, denominators = [], [], [], []
    scaled_vals = []

    for k, free_col in enumerate(free_cols):
        # Free column gets 1 on diagonal
        nz_rows.append(free_col)
        nz_cols.append(k)
        numerators.append(1)
        denominators.append(1)
        scaled_vals.append(common_denom)

        # Pivot columns get -rref[i, free_col]
        for i, pivot_col in enumerate(pivot_cols):
            val = int(rref_int[i, free_col])
            if val != 0:
                nz_rows.append(pivot_col)
                nz_cols.append(k)
                scaled_vals.append(-val)

                # Reduce to lowest terms
                g = gcd(abs(val), common_denom)
                numerators.append(-val // g)
                denominators.append(common_denom // g)

    kernel_rows = cols
    kernel_cols = len(free_cols)

    if nz_rows:
        num_sparse = csr_matrix(
            (numerators, (nz_rows, nz_cols)),
            shape=(kernel_rows, kernel_cols), dtype=np.int64
        )
        den_sparse = csr_matrix(
            (denominators, (nz_rows, nz_cols)),
            shape=(kernel_rows, kernel_cols), dtype=np.int64
        )
        scaled_sparse = csr_matrix(
            (scaled_vals, (nz_rows, nz_cols)),
            shape=(kernel_rows, kernel_cols), dtype=np.int64
        )
    else:
        num_sparse = csr_matrix((kernel_rows, kernel_cols), dtype=np.int64)
        den_sparse = csr_matrix((kernel_rows, kernel_cols), dtype=np.int64)
        scaled_sparse = csr_matrix((kernel_rows, kernel_cols), dtype=np.int64)
        common_denom = 1

    return RationalMatrix._from_sparse(num_sparse, den_sparse, scaled_sparse, common_denom)


def _basic_columns_flint(matrix: RationalMatrix) -> List[int]:
    """Find pivot column indices using sparse backbone."""
    mat = matrix.to_fmpq_mat()
    rows, cols = mat.nrows(), mat.ncols()

    rref_mat, rk = mat.rref()

    zero = fmpq(0)
    pivot_cols = []
    for r in range(min(rows, rk)):
        for c in range(cols):
            if rref_mat[r, c] != zero:
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
