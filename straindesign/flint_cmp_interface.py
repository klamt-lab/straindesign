"""
Rational arithmetic interface for straindesign using FLINT.

This module provides exact rational arithmetic for metabolic network compression.
FLINT (python-flint) is required for fast exact computation.
"""

from typing import List, Tuple, Optional, Iterator
from functools import reduce
import numpy as np
from fractions import Fraction
from math import gcd, lcm
from scipy.sparse import csr_matrix, csc_matrix

# FLINT is required
from flint import fmpq, fmpq_mat
FLINT_AVAILABLE = True


# =============================================================================
# Utility Functions
# =============================================================================

def float_to_rational(val, max_precision: int = 6, max_denom: int = 100) -> Fraction:
    """Convert float to Fraction with bounded denominators."""
    if isinstance(val, Fraction):
        return val
    if val == 0:
        return Fraction(0)
    if val == int(val):
        return Fraction(int(val))

    small_frac = Fraction(val).limit_denominator(max_denom)
    if round(float(small_frac), max_precision) == round(val, max_precision):
        return small_frac

    denom = 10 ** max_precision
    numer = round(val * denom)
    return Fraction(numer, denom)


def detect_max_precision(model) -> int:
    """Detect maximum decimal precision needed for model coefficients."""
    max_decimals = 0
    for rxn in model.reactions:
        for met, coeff in rxn.metabolites.items():
            if isinstance(coeff, float) and coeff != 0:
                s = f"{abs(coeff):.15g}"
                if '.' in s:
                    decimals = len(s.split('.')[1].rstrip('0'))
                    max_decimals = max(max_decimals, decimals)
    return min(12, max(3, max_decimals))


def _lcm_list(numbers: List[int]) -> int:
    """Compute LCM of a list of integers."""
    return reduce(lcm, numbers, 1) if numbers else 1


# =============================================================================
# Rational Matrix with Sparse Storage
# =============================================================================

class RationalMatrix:
    """Sparse rational matrix using dual int sparse storage (numerators + denominators)."""

    def __init__(self, rows: int, cols: int):
        self._rows = rows
        self._cols = cols
        self._num_sparse: Optional[csr_matrix] = None  # Numerators
        self._den_sparse: Optional[csr_matrix] = None  # Denominators
        self._csc_cache: Optional[csc_matrix] = None
        self._batch_mode = False

    def _invalidate_cache(self):
        if not self._batch_mode:
            self._csc_cache = None

    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------

    @classmethod
    def _from_sparse(cls, num_sparse: csr_matrix, den_sparse: csr_matrix,
                     scaled: Optional[csr_matrix] = None,
                     common_denom: Optional[int] = None) -> 'RationalMatrix':
        """Create RationalMatrix from sparse components."""
        rows, cols = num_sparse.shape
        result = cls(rows, cols)
        result._num_sparse = num_sparse.copy()
        result._den_sparse = den_sparse.copy()
        return result

    @classmethod
    def identity(cls, size: int) -> 'RationalMatrix':
        """Create identity matrix."""
        row_idx = list(range(size))
        col_idx = list(range(size))
        num_data = [1] * size
        den_data = [1] * size
        num_sparse = csr_matrix((num_data, (row_idx, col_idx)), shape=(size, size), dtype=np.int64)
        den_sparse = csr_matrix((den_data, (row_idx, col_idx)), shape=(size, size), dtype=np.int64)
        return cls._from_sparse(num_sparse, den_sparse)

    @classmethod
    def from_numpy(cls, arr: np.ndarray, max_precision: int = 6,
                   max_denom: int = 100) -> 'RationalMatrix':
        """Create RationalMatrix from numpy array."""
        rows, cols = arr.shape
        row_idx, col_idx, num_data, den_data = [], [], [], []

        for r in range(rows):
            for c in range(cols):
                val = arr[r, c]
                if val != 0:
                    frac = float_to_rational(val, max_precision, max_denom)
                    row_idx.append(r)
                    col_idx.append(c)
                    num_data.append(frac.numerator)
                    den_data.append(frac.denominator)

        num_sparse = csr_matrix((num_data, (row_idx, col_idx)), shape=(rows, cols), dtype=np.int64)
        den_sparse = csr_matrix((den_data, (row_idx, col_idx)), shape=(rows, cols), dtype=np.int64)
        return cls._from_sparse(num_sparse, den_sparse)

    @classmethod
    def from_cobra_model(cls, model, max_precision: int = 6,
                         max_denom: int = 100) -> 'RationalMatrix':
        """Create RationalMatrix from COBRA model stoichiometry."""
        num_mets = len(model.metabolites)
        num_rxns = len(model.reactions)
        met_index = {m.id: i for i, m in enumerate(model.metabolites)}

        row_idx, col_idx, num_data, den_data = [], [], [], []

        for j, rxn in enumerate(model.reactions):
            for met, coeff in rxn.metabolites.items():
                if coeff != 0:
                    i = met_index[met.id]
                    if isinstance(coeff, Fraction):
                        frac = coeff
                    elif hasattr(coeff, 'p') and hasattr(coeff, 'q'):
                        frac = Fraction(int(coeff.p), int(coeff.q))
                    elif hasattr(coeff, 'numerator'):
                        frac = Fraction(coeff.numerator, coeff.denominator)
                    else:
                        frac = float_to_rational(coeff, max_precision, max_denom)

                    row_idx.append(i)
                    col_idx.append(j)
                    num_data.append(frac.numerator)
                    den_data.append(frac.denominator)

        num_sparse = csr_matrix((num_data, (row_idx, col_idx)),
                                shape=(num_mets, num_rxns), dtype=np.int64)
        den_sparse = csr_matrix((den_data, (row_idx, col_idx)),
                                shape=(num_mets, num_rxns), dtype=np.int64)
        return cls._from_sparse(num_sparse, den_sparse)

    @classmethod
    def _build_from_sparse_data(cls, row_indices: List[int], col_indices: List[int],
                                numerators: List[int], denominators: List[int],
                                num_rows: int, num_cols: int) -> 'RationalMatrix':
        """Build RationalMatrix from sparse coordinate data."""
        num_sparse = csr_matrix((numerators, (row_indices, col_indices)),
                                shape=(num_rows, num_cols), dtype=np.int64)
        den_sparse = csr_matrix((denominators, (row_indices, col_indices)),
                                shape=(num_rows, num_cols), dtype=np.int64)
        return cls._from_sparse(num_sparse, den_sparse)

    # -------------------------------------------------------------------------
    # Size queries
    # -------------------------------------------------------------------------

    def get_row_count(self) -> int:
        return self._rows

    def get_column_count(self) -> int:
        return self._cols

    # -------------------------------------------------------------------------
    # Iteration
    # -------------------------------------------------------------------------

    def iter_column_fractions(self, col: int) -> Iterator[Tuple[int, Fraction]]:
        """Iterate over non-zero entries in column as (row, Fraction) pairs."""
        if self._csc_cache is None:
            self._csc_cache = self._num_sparse.tocsc()
            self._den_csc_cache = self._den_sparse.tocsc()

        start = self._csc_cache.indptr[col]
        end = self._csc_cache.indptr[col + 1]

        for idx in range(start, end):
            row = self._csc_cache.indices[idx]
            num = int(self._csc_cache.data[idx])
            den = int(self._den_csc_cache.data[idx])
            if num != 0:
                yield row, Fraction(num, den)

    def get_signum(self, row: int, col: int) -> int:
        """Return sign of element: -1, 0, or 1."""
        num = self._num_sparse[row, col]
        if num > 0:
            return 1
        elif num < 0:
            return -1
        return 0

    # -------------------------------------------------------------------------
    # Batch edit mode
    # -------------------------------------------------------------------------

    def begin_batch_edit(self):
        """Enter batch edit mode - delays cache invalidation."""
        self._batch_mode = True
        self._num_sparse = self._num_sparse.tolil()
        self._den_sparse = self._den_sparse.tolil()

    def end_batch_edit(self):
        """Exit batch edit mode - converts back to CSR and invalidates cache."""
        self._batch_mode = False
        self._num_sparse = self._num_sparse.tocsr()
        self._den_sparse = self._den_sparse.tocsr()
        self._csc_cache = None

    # -------------------------------------------------------------------------
    # Matrix operations
    # -------------------------------------------------------------------------

    def clone(self) -> 'RationalMatrix':
        """Create a deep copy."""
        return RationalMatrix._from_sparse(
            self._num_sparse.copy(),
            self._den_sparse.copy()
        )

    def submatrix(self, rows: int, cols: int) -> 'RationalMatrix':
        """Extract top-left submatrix of given dimensions."""
        num_sub = self._num_sparse[:rows, :cols].copy()
        den_sub = self._den_sparse[:rows, :cols].copy()
        return RationalMatrix._from_sparse(num_sub, den_sub)

    def remove_rows(self, keep_indices: List[int]) -> None:
        """Keep only the specified rows."""
        keep = np.array(keep_indices, dtype=np.intp)
        self._num_sparse = self._num_sparse[keep, :]
        self._den_sparse = self._den_sparse[keep, :]
        self._rows = len(keep_indices)
        self._invalidate_cache()

    def remove_columns(self, keep_indices: List[int]) -> None:
        """Keep only the specified columns."""
        keep = np.array(keep_indices, dtype=np.intp)
        self._num_sparse = self._num_sparse[:, keep]
        self._den_sparse = self._den_sparse[:, keep]
        self._cols = len(keep_indices)
        self._invalidate_cache()

    def add_scaled_column(self, dst_col: int, src_col: int,
                          scalar_num: int, scalar_den: int) -> None:
        """Add scalar * column[src] to column[dst]. dst[i] += (num/den) * src[i]"""
        if scalar_num == 0:
            return

        num_lil = self._num_sparse
        den_lil = self._den_sparse

        # Get source column entries
        if hasattr(num_lil, 'rows'):  # LIL format
            src_rows = num_lil.rows[src_col] if num_lil.format == 'csc' else None

        # Convert to CSC for efficient column access
        num_csc = num_lil.tocsc() if num_lil.format != 'csc' else num_lil
        den_csc = den_lil.tocsc() if den_lil.format != 'csc' else den_lil

        start = num_csc.indptr[src_col]
        end = num_csc.indptr[src_col + 1]

        for idx in range(start, end):
            row = num_csc.indices[idx]
            src_num = int(num_csc.data[idx])
            src_den = int(den_csc.data[idx])

            if src_num == 0:
                continue

            # Get current dst value
            dst_num = int(num_lil[row, dst_col])
            dst_den = int(den_lil[row, dst_col]) if dst_num != 0 else 1

            # Compute: dst + (scalar_num/scalar_den) * (src_num/src_den)
            # = dst_num/dst_den + (scalar_num * src_num)/(scalar_den * src_den)
            add_num = scalar_num * src_num
            add_den = scalar_den * src_den

            if dst_num == 0:
                new_num, new_den = add_num, add_den
            else:
                # Common denominator addition
                new_num = dst_num * add_den + add_num * dst_den
                new_den = dst_den * add_den

            # Reduce
            if new_num != 0:
                g = gcd(abs(new_num), new_den)
                new_num //= g
                new_den //= g

            num_lil[row, dst_col] = new_num
            den_lil[row, dst_col] = new_den if new_num != 0 else 0

        self._invalidate_cache()

    # -------------------------------------------------------------------------
    # Conversion
    # -------------------------------------------------------------------------

    def to_fmpq_mat(self) -> fmpq_mat:
        """Convert to FLINT fmpq_mat for linear algebra operations."""
        mat = fmpq_mat(self._rows, self._cols)
        coo_num = self._num_sparse.tocoo()
        coo_den = self._den_sparse.tocoo()

        for r, c, num, den in zip(coo_num.row, coo_num.col, coo_num.data, coo_den.data):
            if num != 0:
                mat[int(r), int(c)] = fmpq(int(num), int(den))
        return mat

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy float array."""
        result = np.zeros((self._rows, self._cols), dtype=np.float64)
        coo_num = self._num_sparse.tocoo()
        coo_den = self._den_sparse.tocoo()

        for r, c, num, den in zip(coo_num.row, coo_num.col, coo_num.data, coo_den.data):
            if num != 0:
                result[r, c] = float(num) / float(den)
        return result

    def to_sparse_csr(self) -> Tuple[csr_matrix, int]:
        """Return sparse representation and common denominator.

        Returns numerator matrix scaled by LCM of denominators, plus the LCM.
        """
        # Compute LCM of all denominators
        dens = self._den_sparse.data
        if len(dens) == 0:
            return csr_matrix((self._rows, self._cols), dtype=np.int64), 1

        common_denom = _lcm_list([int(d) for d in dens if d != 0])

        # Scale numerators
        coo_num = self._num_sparse.tocoo()
        coo_den = self._den_sparse.tocoo()

        scaled_data = []
        for num, den in zip(coo_num.data, coo_den.data):
            if num != 0:
                scaled_data.append(int(num) * (common_denom // int(den)))
            else:
                scaled_data.append(0)

        scaled = csr_matrix((scaled_data, (coo_num.row, coo_num.col)),
                            shape=(self._rows, self._cols), dtype=np.int64)
        return scaled, common_denom

    def __repr__(self) -> str:
        return f"RationalMatrix({self._rows}x{self._cols})"


# =============================================================================
# Linear Algebra Functions
# =============================================================================

def nullspace(matrix: RationalMatrix) -> RationalMatrix:
    """Compute right nullspace (kernel). Returns K where matrix @ K = 0."""
    return _nullspace_flint(matrix)


def basic_columns(matrix: RationalMatrix) -> List[int]:
    """Find indices of basic (pivot) columns."""
    return _basic_columns_flint(matrix)


def basic_columns_from_numpy(mx: np.ndarray) -> List[int]:
    """Find basic columns from numpy array."""
    return basic_columns(RationalMatrix.from_numpy(mx))


def _nullspace_flint(matrix: RationalMatrix) -> RationalMatrix:
    """Compute nullspace using FLINT's RREF."""
    rows = matrix.get_row_count()
    cols = matrix.get_column_count()

    mat = matrix.to_fmpq_mat()
    rref_mat, rk = mat.rref()

    if rk == cols:
        return RationalMatrix(cols, 0)

    # Find pivot columns
    zero = fmpq(0)
    pivot_cols = []
    for r in range(min(rows, rk)):
        for c in range(cols):
            if rref_mat[r, c] != zero:
                pivot_cols.append(c)
                break

    pivot_set = set(pivot_cols)
    free_cols = [c for c in range(cols) if c not in pivot_set]
    nullity = len(free_cols)

    if nullity == 0:
        return RationalMatrix(cols, 0)

    # Build kernel matrix
    row_indices, col_indices = [], []
    numerators, denominators = [], []

    for k, free_col in enumerate(free_cols):
        # Identity entry for free variable
        row_indices.append(free_col)
        col_indices.append(k)
        numerators.append(1)
        denominators.append(1)

        # Entries from RREF for pivot variables
        for i, pivot_col in enumerate(pivot_cols):
            v = rref_mat[i, free_col]
            if v != zero:
                row_indices.append(pivot_col)
                col_indices.append(k)
                numerators.append(-int(v.p))
                denominators.append(int(v.q))

    return RationalMatrix._build_from_sparse_data(
        row_indices, col_indices, numerators, denominators, cols, nullity
    )


def _basic_columns_flint(matrix: RationalMatrix) -> List[int]:
    """Find pivot columns using FLINT's RREF."""
    rows = matrix.get_row_count()
    cols = matrix.get_column_count()

    mat = matrix.to_fmpq_mat()
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
# Exports
# =============================================================================

__all__ = [
    'FLINT_AVAILABLE',
    'RationalMatrix',
    'float_to_rational',
    'detect_max_precision',
    'nullspace',
    'basic_columns',
    'basic_columns_from_numpy',
]
