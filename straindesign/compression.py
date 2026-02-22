"""
Metabolic network compression using nullspace-based coupling detection.

This module provides network compression for COBRA models. The main function
is compress_cobra_model() which compresses a model and returns transformation
matrices for converting between original and compressed flux spaces.

API:
    >>> from straindesign.compression import compress_cobra_model
    >>> result = compress_cobra_model(model)
    >>> compressed_model = result.compressed_model

Note: The model should be preprocessed first (rational coefficients,
conservation relations removed). Use networktools.compress_model() for
automatic preprocessing.
"""

import copy
import logging
import numpy as np
from enum import Enum
from functools import reduce
from math import gcd, lcm, isinf
from typing import Dict, Iterator, List, Optional, Set, Tuple, Union, Any

from fractions import Fraction
from scipy import sparse
from scipy.sparse import csr_matrix, csc_matrix
from sympy import Rational
from cobra import Configuration
from cobra.util.array import create_stoichiometric_matrix

LOG = logging.getLogger(__name__)


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

    def to_sparse_pattern(self) -> Tuple[csr_matrix, Dict[int, Dict[int, Fraction]]]:
        """Return sparsity pattern as CSR and row-wise Fraction data.

        For pattern analysis (coupled reaction detection) without integer overflow.
        Returns:
            pattern: CSR matrix with 1s at non-zero positions (for indptr/indices)
            row_data: {row: {col: Fraction}} for actual values
        """
        # Build pattern matrix (just 1s for structure)
        coo_num = self._num_sparse.tocoo()
        pattern_data = [1] * len(coo_num.data)
        pattern = csr_matrix((pattern_data, (coo_num.row, coo_num.col)),
                             shape=(self._rows, self._cols), dtype=np.int8)

        # Build row-wise Fraction data
        coo_den = self._den_sparse.tocoo()
        row_data: Dict[int, Dict[int, Fraction]] = {}
        for r, c, num, den in zip(coo_num.row, coo_num.col, coo_num.data, coo_den.data):
            num_int = int(num)
            den_int = int(den) if den != 0 else 1
            if num_int != 0:
                if r not in row_data:
                    row_data[r] = {}
                row_data[r][c] = Fraction(num_int, den_int)

        return pattern, row_data

    def __repr__(self) -> str:
        return f"RationalMatrix({self._rows}x{self._cols})"


# =============================================================================
# Sparse Integer RREF for Nullspace Computation
# =============================================================================

def _rref_integer_sparse(rm: RationalMatrix) -> Tuple[Dict[int, Dict[int, int]], int, List[int]]:
    """Compute integer RREF using dict-of-dicts for sparse row operations.

    Columns are pre-sorted by nnz ascending so that sparse (likely pivot) columns
    are processed first, reducing pivot-search time and fill-in during elimination.
    Rows are similarly pre-sorted by nnz ascending. Results are translated back to
    the original column ordering before return.

    Uses GCD pre-scaling before row operations and post-reduction to control
    coefficient growth. No denominator tracking needed — rows are arbitrarily
    scalable for nullspace computation.

    Args:
        rm: Input RationalMatrix

    Returns:
        (rref_data, rank, pivot_columns)
        rref_data: {rref_row: {orig_col: value}} in original column space
        pivot_columns: pivot column indices in original column space
    """
    rows = rm.get_row_count()
    cols = rm.get_column_count()

    # --- Column sorting: sparse columns first ---
    # col_order[sorted_pos] = original_col
    nnz_per_col = np.diff(rm._num_sparse.tocsc().indptr)
    col_order = np.argsort(nnz_per_col, kind='stable').tolist()
    col_inverse = [0] * cols
    for sorted_pos, orig_col in enumerate(col_order):
        col_inverse[orig_col] = sorted_pos

    # Convert to integer matrix (scale each row by LCM of denominators)
    # Store column indices in sorted space
    num_csr = rm._num_sparse.tocsr()
    den_csr = rm._den_sparse.tocsr()

    data: Dict[int, Dict[int, int]] = {}
    for r in range(rows):
        start, end = num_csr.indptr[r], num_csr.indptr[r + 1]
        if start == end:
            continue

        # Compute row LCM of denominators
        row_dens = [int(den_csr.data[i]) for i in range(start, end) if den_csr.data[i] != 0]
        row_lcm = reduce(lcm, row_dens, 1) if row_dens else 1

        # Scale numerators; store using sorted column index
        row_data = {}
        for i in range(start, end):
            num = int(num_csr.data[i])
            den = int(den_csr.data[i]) if den_csr.data[i] != 0 else 1
            orig_col = int(num_csr.indices[i])
            scaled = num * (row_lcm // den)
            if scaled != 0:
                row_data[col_inverse[orig_col]] = scaled
        if row_data:
            data[r] = row_data

    # --- Row sorting: sparse rows first (better initial pivot candidates) ---
    if data:
        sorted_row_keys = sorted(data.keys(), key=lambda r: len(data[r]))
        data = {new_r: data[old_r] for new_r, old_r in enumerate(sorted_row_keys)}

    # Gaussian elimination with partial pivoting (all indices in sorted column space)
    pivot_cols_sorted = []
    pivot_row = 0

    for pivot_col in range(cols):
        if pivot_row >= rows:
            break

        # Find pivot with smallest absolute value (minimizes coefficient growth)
        best_row, best_val, best_abs = -1, 0, float('inf')
        for r, row_data in data.items():
            if r < pivot_row:
                continue
            if pivot_col in row_data:
                v = row_data[pivot_col]
                if abs(v) < best_abs:
                    best_row, best_val, best_abs = r, v, abs(v)

        if best_row < 0:
            continue

        # Swap rows if needed
        if best_row != pivot_row:
            if pivot_row in data:
                if best_row in data:
                    data[pivot_row], data[best_row] = data[best_row], data[pivot_row]
                else:
                    data[best_row] = data.pop(pivot_row)
            elif best_row in data:
                data[pivot_row] = data.pop(best_row)

        pivot_row_data = data.get(pivot_row)
        if pivot_row_data is None:
            continue
        pivot_val = pivot_row_data.get(pivot_col, 0)
        if pivot_val == 0:
            continue

        pivot_cols_sorted.append(pivot_col)

        # Collect rows to eliminate (snapshot keys before modification)
        elim_targets = [(r, row_data[pivot_col])
                        for r, row_data in data.items()
                        if r != pivot_row and pivot_col in row_data]

        # Eliminate pivot column from all other rows
        for elim_row, elim_val in elim_targets:
            elim_row_data = data[elim_row]

            # Pre-scale by GCD to minimize coefficient growth.
            # math.gcd handles negatives and returns a positive value (Python 3.5+).
            g = gcd(pivot_val, elim_val)
            pv_scaled = pivot_val // g
            ev_scaled = elim_val // g

            # new[c] = elim[c] * pv_scaled - ev_scaled * pivot[c]
            new_row = {}
            for c, p_val in pivot_row_data.items():
                new_val = elim_row_data.get(c, 0) * pv_scaled - ev_scaled * p_val
                if new_val != 0:
                    new_row[c] = new_val
            for c, e_val in elim_row_data.items():
                if c not in pivot_row_data:
                    new_val = e_val * pv_scaled
                    if new_val != 0:
                        new_row[c] = new_val

            # Update row and reduce by GCD.
            # gcd(*values) uses one C-level call instead of len(row) Python calls.
            if new_row:
                data[elim_row] = new_row
                row_gcd = gcd(*new_row.values())
                if row_gcd > 1:
                    for c in new_row:
                        new_row[c] //= row_gcd
            else:
                del data[elim_row]

        pivot_row += 1

    # Final GCD reduction of all rows
    for row_data in data.values():
        row_gcd = gcd(*row_data.values())
        if row_gcd > 1:
            for c in row_data:
                row_data[c] //= row_gcd

    # Translate results back to original column space
    original_data = {r: {col_order[sc]: v for sc, v in rd.items()} for r, rd in data.items()}
    pivot_cols_original = [col_order[p] for p in pivot_cols_sorted]

    return original_data, len(pivot_cols_original), pivot_cols_original


def _nullspace_sparse(matrix: RationalMatrix) -> RationalMatrix:
    """Compute nullspace using integer RREF with row scaling.

    No denominators needed - each row in RREF is scaled so pivot divides all elements.
    The nullspace is extracted by reading off the relationships between pivot and free columns.
    """
    cols = matrix.get_column_count()

    # Compute integer RREF
    rref_data, rank, pivot_cols = _rref_integer_sparse(matrix)

    if rank == cols:
        return RationalMatrix(cols, 0)

    # Free columns
    pivot_set = set(pivot_cols)
    free_cols = [c for c in range(cols) if c not in pivot_set]
    nullity = len(free_cols)

    if nullity == 0:
        return RationalMatrix(cols, 0)

    # Build kernel matrix
    # For each free column f, the nullspace vector has:
    # - Entry 1 at position f (the free variable)
    # - Entry -rref[i,f]/rref[i,pivot_i] at each pivot position
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
            row_data = rref_data.get(i, {})
            val_at_free = row_data.get(free_col, 0)
            if val_at_free != 0:
                pivot_val = row_data.get(pivot_col, 1)  # Should always exist
                # Nullspace entry: -val_at_free / pivot_val
                g = gcd(abs(val_at_free), abs(pivot_val))
                num = -val_at_free // g
                den = pivot_val // g
                # Ensure positive denominator
                if den < 0:
                    num, den = -num, -den
                row_indices.append(pivot_col)
                col_indices.append(k)
                numerators.append(num)
                denominators.append(den)

    return RationalMatrix._build_from_sparse_data(
        row_indices, col_indices, numerators, denominators, cols, nullity
    )


# =============================================================================
# Linear Algebra Functions
# =============================================================================

def nullspace(matrix: RationalMatrix) -> RationalMatrix:
    """Compute right nullspace (kernel). Returns K where matrix @ K = 0.

    Uses integer RREF with column/row pre-sorting and GCD reduction.
    All arithmetic is Python arbitrary-precision integers — no overflow possible.

    Args:
        matrix: Input RationalMatrix

    Returns:
        Kernel matrix K where matrix @ K = 0
    """
    return _nullspace_sparse(matrix)


def basic_columns(matrix: RationalMatrix) -> List[int]:
    """Find indices of basic (pivot) columns using integer RREF."""
    _, _, pivot_cols = _rref_integer_sparse(matrix)
    return pivot_cols


def basic_columns_from_numpy(mx: np.ndarray) -> List[int]:
    """Find basic columns from a numpy array."""
    return basic_columns(RationalMatrix.from_numpy(mx))


# =============================================================================
# Configuration
# =============================================================================

class CompressionMethod(Enum):
    """Compression methods for metabolic network compression."""
    NULLSPACE = "Nullspace"  # Nullspace-based compression
    RECURSIVE = "Recursive"  # Iterate until no more compression possible

    @classmethod
    def all(cls) -> List['CompressionMethod']:
        return list(cls)

    @classmethod
    def none(cls) -> List['CompressionMethod']:
        return []

    @classmethod
    def standard(cls) -> List['CompressionMethod']:
        """Standard compression methods (recommended)."""
        return [cls.NULLSPACE, cls.RECURSIVE]


# =============================================================================
# Statistics
# =============================================================================

class CompressionStatistics:
    """Tracks compression statistics for logging."""

    def __init__(self):
        self.iteration_count = 0
        self.zero_flux_count = 0
        self.contradicting_count = 0
        self.coupled_count = 0
        self.unused_metabolite_count = 0

    def inc_compression_iteration(self) -> int:
        self.iteration_count += 1
        return self.iteration_count

    def get_compression_iteration(self) -> int:
        return self.iteration_count

    def inc_zero_flux_reactions(self) -> None:
        self.zero_flux_count += 1

    def inc_contradicting_reactions(self) -> None:
        self.contradicting_count += 1

    def inc_coupled_reactions_count(self, count: int) -> None:
        self.coupled_count += count

    def inc_unused_metabolite(self) -> None:
        self.unused_metabolite_count += 1

    def write_to_log(self) -> None:
        LOG.info(f"Compression complete: {self.iteration_count} iterations, "
                 f"{self.zero_flux_count} zero-flux, {self.contradicting_count} contradicting, "
                 f"{self.coupled_count} coupled, {self.unused_metabolite_count} unused metabolites")

    def __repr__(self):
        return (f"CompressionStatistics(iterations={self.iteration_count}, "
                f"zero_flux={self.zero_flux_count}, contradicting={self.contradicting_count}, "
                f"coupled={self.coupled_count})")


# =============================================================================
# Compression Record
# =============================================================================

class CompressionRecord:
    """
    Compression result with transformation matrices.

    Contains: pre @ stoich @ post == cmp
    For EFM expansion: efm_original = post @ efm_compressed
    """

    def __init__(self, pre: RationalMatrix, cmp: RationalMatrix,
                 post: RationalMatrix, reversible: List[bool],
                 meta_names: List[str],
                 stats: Optional[CompressionStatistics] = None):
        self.pre = pre    # metabolite transformation
        self.cmp = cmp    # compressed stoich
        self.post = post  # reaction transformation
        self.reversible = list(reversible)
        self.meta_names = list(meta_names)  # compressed metabolite names (row order in cmp)
        self.stats = stats


# =============================================================================
# Working State (Internal)
# =============================================================================

class _Size:
    """Mutable counter for active matrix dimensions during compression."""
    def __init__(self, metas: int, reacs: int):
        self.metas = metas
        self.reacs = reacs


class _WorkRecord:
    """Mutable state during compression algorithm."""

    def __init__(self, stoich: RationalMatrix, reversible: List[bool],
                 meta_names: List[str], reac_names: List[str]):
        rows, cols = stoich.get_row_count(), stoich.get_column_count()
        self.pre = RationalMatrix.identity(rows)
        self.cmp = stoich.clone()
        self.post = RationalMatrix.identity(cols)
        self.reversible = list(reversible)
        self.meta_names = list(meta_names)
        self.reac_names = list(reac_names)
        self.size = _Size(rows, cols)
        # Store original dimensions for get_truncated()
        self._orig_metas = rows
        self._orig_reacs = cols
        self.stats = CompressionStatistics()
        self.stats.inc_compression_iteration()

    def remove_reactions(self, suppressed: Optional[Set[str]]) -> bool:
        """Remove reactions by name - uses batch removal."""
        if not suppressed:
            return False
        indices = set()
        for name in suppressed:
            try:
                idx = self.reac_names.index(name)
                if idx < self.size.reacs:
                    indices.add(idx)
            except ValueError:
                continue
        if not indices:
            return False
        return self.remove_reactions_by_indices(indices)

    def remove_reactions_by_indices(self, indices: Set[int]) -> bool:
        """Remove reactions by index - uses batch removal for efficiency."""
        if not indices:
            return False

        # Filter to only valid indices
        valid_indices = {idx for idx in indices if idx < self.size.reacs}
        if not valid_indices:
            return False

        # Compute indices to keep
        keep_indices = [i for i in range(self.size.reacs) if i not in valid_indices]

        if not keep_indices:
            self.size.reacs = 0
            return True

        # Use batch column removal
        self._batch_remove_columns(keep_indices)
        return True

    def _batch_remove_columns(self, keep_indices: list) -> None:
        """Batch remove columns by keeping only specified indices."""
        LOG.debug(f"_batch_remove_columns: keeping {len(keep_indices)} of {self.size.reacs}")
        LOG.debug(f"  Before: cmp={self.cmp.get_column_count()}, post={self.post.get_column_count()}")

        # Remove columns from post and cmp matrices
        self.post.remove_columns(keep_indices)
        self.cmp.remove_columns(keep_indices)

        LOG.debug(f"  After: cmp={self.cmp.get_column_count()}, post={self.post.get_column_count()}")

        # Reindex names and reversibility
        new_names = [self.reac_names[i] for i in keep_indices]
        new_rev = [self.reversible[i] for i in keep_indices]

        # Update arrays (preserving total length for consistency)
        orig_len = len(self.reac_names)
        for i, (name, rev) in enumerate(zip(new_names, new_rev)):
            self.reac_names[i] = name
            self.reversible[i] = rev

        self.size.reacs = len(keep_indices)
        LOG.debug(f"  size.reacs={self.size.reacs}")

    def _batch_remove_rows(self, keep_indices: list) -> None:
        """Batch remove rows (metabolites) by keeping only specified indices."""
        LOG.debug(f"_batch_remove_rows: keeping {len(keep_indices)} of {self.size.metas}")

        # Remove rows from pre and cmp matrices
        self.pre.remove_rows(keep_indices)
        self.cmp.remove_rows(keep_indices)

        # Reindex metabolite names
        new_names = [self.meta_names[i] for i in keep_indices]
        for i, name in enumerate(new_names):
            self.meta_names[i] = name

        self.size.metas = len(keep_indices)
        LOG.debug(f"  size.metas={self.size.metas}")

    def remove_metabolites_by_indices(self, indices: Set[int]) -> bool:
        """Remove metabolites by index - uses batch removal for efficiency."""
        if not indices:
            return False

        # Filter to only valid indices
        valid_indices = {idx for idx in indices if idx < self.size.metas}
        if not valid_indices:
            return False

        # Compute indices to keep
        keep_indices = [i for i in range(self.size.metas) if i not in valid_indices]

        if not keep_indices:
            self.size.metas = 0
            return True

        # Use batch row removal
        self._batch_remove_rows(keep_indices)
        return True

    def remove_unused_metabolites(self) -> bool:
        """Remove metabolites with all-zero rows - uses batch removal.

        Uses CSR indptr for O(m) zero-row detection (avoids element-by-element
        sparse indexing which is O(m*r) with large per-call Python overhead).
        """
        mc, rc = self.size.metas, self.size.reacs
        # Slice active submatrix and convert to CSR for indptr access.
        # tocsr() is a no-op if already CSR (returns self), so this is safe
        # regardless of current format.
        num_csr = self.cmp._num_sparse[:mc, :rc].tocsr()
        zero_rows = np.where(np.diff(num_csr.indptr) == 0)[0]
        unused_indices = set(zero_rows.tolist())
        for _ in unused_indices:
            self.stats.inc_unused_metabolite()

        if unused_indices:
            self.remove_metabolites_by_indices(unused_indices)
            return True
        return False

    def get_truncated(self) -> CompressionRecord:
        """Create final CompressionRecord with truncated matrices."""
        mc, rc = self.size.metas, self.size.reacs
        # Use stored original dimensions (batch removal changes matrix sizes)
        m = self._orig_metas
        r = self._orig_reacs

        # Create truncated matrices using efficient submatrix extraction
        # pre: mc compressed metabolites × m original metabolites
        # cmp: mc compressed metabolites × rc compressed reactions
        # post: r original reactions × rc compressed reactions
        pre_trunc = self.pre.submatrix(mc, m)
        cmp_trunc = self.cmp.submatrix(mc, rc)
        post_trunc = self.post.submatrix(r, rc)

        rev_trunc = self.reversible[:rc]
        meta_names_trunc = self.meta_names[:mc]
        return CompressionRecord(pre_trunc, cmp_trunc, post_trunc, rev_trunc, meta_names_trunc, self.stats)


# =============================================================================
# Core Algorithm
# =============================================================================

class StoichMatrixCompressor:
    """Nullspace-based metabolic network compression."""

    def __init__(self, *methods: CompressionMethod):
        self._methods = list(methods) if methods else CompressionMethod.standard()

    def compress(self, stoich: RationalMatrix, reversible: List[bool],
                 meta_names: List[str], reac_names: List[str],
                 suppressed: Optional[Set[str]] = None) -> CompressionRecord:
        """Compress network, return transformation matrices.

        Uses Java efmtool's two-phase approach:
        1. Phase 1: Remove zero-flux and contradicting reactions (iteratively)
        2. Phase 2: Combine coupled reactions (iteratively)

        This separation prevents cascading effects where combining reactions
        could create new coupling patterns that weren't present in the original.
        """
        work = _WorkRecord(stoich, reversible, meta_names, reac_names)

        do_nullspace = CompressionMethod.NULLSPACE in self._methods
        do_recursive = CompressionMethod.RECURSIVE in self._methods

        work.remove_reactions(suppressed)

        if do_nullspace:
            # Phase 1: Remove zero-flux and contradicting reactions only
            # (matches Java's inclCompression=false phase)
            # Optimization: Continue only if nullspace compression found reactions.
            # Removing unused metabolites (all-zero rows) doesn't change the nullspace,
            # so if _nullspace_compress returns False, the next iteration would too.
            while True:
                work.stats.inc_compression_iteration()
                work.remove_unused_metabolites()
                found = self._nullspace_compress(work, incl_compression=False)
                if not (found and do_recursive):
                    break

            # Phase 2: Combine coupled reactions
            # (matches Java's inclCompression=true phase)
            while True:
                work.stats.inc_compression_iteration()
                work.remove_unused_metabolites()
                found = self._nullspace_compress(work, incl_compression=True)
                if not (found and do_recursive):
                    break

        work.remove_unused_metabolites()
        work.stats.write_to_log()
        return work.get_truncated()

    def _nullspace_compress(self, work: _WorkRecord, incl_compression: bool = True) -> bool:
        """One pass of nullspace compression. Returns True if reactions removed.

        Args:
            incl_compression: If False, only remove zero-flux and contradicting.
                              If True, only combine coupled reactions.
        """
        # Build active submatrix for nullspace computation
        active = work.cmp.submatrix(work.size.metas, work.size.reacs)
        kernel = nullspace(active)
        LOG.debug(f"Nullspace: {active.get_row_count()}x{active.get_column_count()} -> kernel {kernel.get_row_count()}x{kernel.get_column_count()}")

        # Get kernel pattern (CSR for structure) and values (Fractions for exact ratios)
        kernel_pattern, kernel_values = kernel.to_sparse_pattern()

        changed = False
        if not incl_compression:
            # Phase 1: Remove zero-flux and contradicting
            # IMPORTANT: Find both sets BEFORE any removal to avoid index mismatch!
            zero_flux = self._find_zero_flux(work, kernel_pattern)
            contradicting = self._find_contradicting(work, kernel_pattern, kernel_values)

            # Remove all at once
            all_to_remove = zero_flux | contradicting
            LOG.debug(f"Phase 1: {len(zero_flux)} zero-flux + {len(contradicting)} contradicting = {len(all_to_remove)} to remove")
            if all_to_remove:
                work.remove_reactions_by_indices(all_to_remove)
                changed = True
        else:
            # Phase 2: Combine coupled reactions
            changed |= self._handle_coupled_combine(work, kernel_pattern, kernel_values)

        if changed:
            work.remove_unused_metabolites()

        return changed

    def _find_zero_flux(self, work: _WorkRecord, kernel_sparse) -> set:
        """Find reactions with all-zero kernel rows (indices in current work)."""
        zero_flux = set()
        for reac in range(work.size.reacs):
            if kernel_sparse.indptr[reac] == kernel_sparse.indptr[reac + 1]:
                zero_flux.add(reac)
                work.stats.inc_zero_flux_reactions()
        return zero_flux

    def _find_coupled_groups(self, kernel_pattern, kernel_values, num_reacs):
        """Find groups of coupled reactions from kernel sparsity pattern.

        Args:
            kernel_pattern: CSR matrix with 1s at non-zero positions (for indptr/indices)
            kernel_values: {row: {col: Fraction}} for actual values

        Returns (groups, ratios) where:
        - groups: list of lists, each containing indices of coupled reactions
        - ratios: list where ratios[i] is the coupling ratio for reaction i
        """
        # Group reactions by zero-pattern in kernel
        patterns = {}
        for reac in range(num_reacs):
            start = kernel_pattern.indptr[reac]
            end = kernel_pattern.indptr[reac + 1]
            pattern = tuple(kernel_pattern.indices[start:end])
            patterns.setdefault(pattern, []).append(reac)

        potential_groups = [idxs for idxs in patterns.values() if len(idxs) > 1]

        groups = []
        ratios = [None] * num_reacs

        for potential in potential_groups:
            ref_reac = potential[0]
            ref_start = kernel_pattern.indptr[ref_reac]
            ref_end = kernel_pattern.indptr[ref_reac + 1]
            nonzero_cols = list(kernel_pattern.indices[ref_start:ref_end])
            if not nonzero_cols:
                continue

            for i, reac_a in enumerate(potential):
                if ratios[reac_a] is not None:
                    continue
                group = None

                # Get values for reaction a
                a_row = kernel_values.get(reac_a, {})

                for j in range(i + 1, len(potential)):
                    reac_b = potential[j]
                    if ratios[reac_b] is not None:
                        continue

                    # Get values for reaction b
                    b_row = kernel_values.get(reac_b, {})

                    # Compute ratio from first non-zero column
                    first_col = nonzero_cols[0]
                    a_val = a_row.get(first_col, Fraction(0))
                    b_val = b_row.get(first_col, Fraction(0))

                    if b_val == 0:
                        continue

                    # ratio = a_val / b_val
                    ratio = a_val / b_val

                    # Check if ratio is consistent across all columns
                    is_consistent = True
                    for col in nonzero_cols[1:]:
                        a_v = a_row.get(col, Fraction(0))
                        b_v = b_row.get(col, Fraction(0))
                        if b_v == 0 or a_v / b_v != ratio:
                            is_consistent = False
                            break

                    if is_consistent:
                        ratios[reac_b] = ratio
                        if group is None:
                            group = [reac_a]
                        group.append(reac_b)

                if group is not None:
                    groups.append(group)

        return groups, ratios

    def _find_contradicting(self, work: _WorkRecord, kernel_pattern, kernel_values) -> set:
        """Find reactions in contradicting coupled groups (indices in current work)."""
        groups, ratios = self._find_coupled_groups(kernel_pattern, kernel_values, work.size.reacs)

        # Find contradicting reactions
        contradicting = set()
        for group in groups:
            is_consistent = self._check_coupling_consistency(group, ratios, work.reversible)
            if not is_consistent:
                for idx in group:
                    contradicting.add(idx)
                    work.stats.inc_contradicting_reactions()

        return contradicting

    def _handle_coupled_combine(self, work: _WorkRecord, kernel_pattern, kernel_values) -> bool:
        """Phase 2: Combine consistent coupled reactions, remove contradicting.

        Returns True if any contradicting reactions were removed (flux space changed),
        which means new couplings might be revealed. Returns False if only merging
        happened, since merging doesn't change the flux space.
        """
        groups, ratios = self._find_coupled_groups(kernel_pattern, kernel_values, work.size.reacs)

        # Combine consistent groups, remove contradicting groups
        reactions_to_remove = set()
        contradicting_removed = False

        # Enter batch edit mode for cmp and post matrices to avoid repeated LIL conversions
        work.cmp.begin_batch_edit()
        work.post.begin_batch_edit()

        for group in groups:
            is_consistent = self._check_coupling_consistency(group, ratios, work.reversible)
            if is_consistent:
                self._combine_coupled(work, group, ratios)
                for idx in group[1:]:
                    reactions_to_remove.add(idx)
            else:
                # Contradicting coupled group: reactions cannot carry flux together
                # due to irreversibility constraints. Remove them to prune blocked paths,
                # which may reveal new couplings in subsequent iterations.
                LOG.debug(f"Removing contradicting coupled group: {[work.reac_names[r] for r in group]}")
                for idx in group:
                    reactions_to_remove.add(idx)
                    work.stats.inc_contradicting_reactions()
                contradicting_removed = True

        # End batch edit mode
        work.cmp.end_batch_edit()
        work.post.end_batch_edit()

        LOG.debug(f"Phase 2: {len(groups)} groups, removing {len(reactions_to_remove)} reactions, contradicting={contradicting_removed}")
        work.remove_reactions_by_indices(reactions_to_remove)

        # Only return True if contradicting reactions were removed (flux space changed).
        # Merging alone doesn't change the flux space, so no new couplings can emerge.
        return contradicting_removed

    def _check_coupling_consistency(self, group: List[int], ratios: List[Optional[Fraction]],
                                    reversible: List[bool]) -> bool:
        """Check if coupled group is consistent with reversibility."""
        for forward in [True, False]:
            all_consistent = forward or reversible[group[0]]
            for idx in group[1:]:
                ratio = ratios[idx]
                ratio_positive = ratio.numerator * ratio.denominator > 0
                if not ((forward == ratio_positive) or reversible[idx]):
                    all_consistent = False
                    break
            if all_consistent:
                return True
        return False

    def _combine_coupled(self, work: _WorkRecord, group: List[int],
                         ratios: List[Optional[Fraction]]) -> None:
        """Combine coupled reactions into master reaction.

        Uses batch column operations with native fmpq arithmetic for performance.
        """
        master = group[0]
        LOG.debug(f"Combining coupled: {[work.reac_names[r] for r in group]}")

        for slave in group[1:]:
            ratio = ratios[slave]
            # multiplier = 1/ratio = ratio.denominator / ratio.numerator
            mult_num, mult_den = ratio.denominator, ratio.numerator

            # Update stoichiometric matrix: cmp[:, master] += cmp[:, slave] * mult
            work.cmp.add_scaled_column(master, slave, mult_num, mult_den)

            # Update post matrix: post[:, master] += post[:, slave] * mult
            work.post.add_scaled_column(master, slave, mult_num, mult_den)

        work.stats.inc_coupled_reactions_count(len(group))


# =============================================================================
# COBRA Interface
# =============================================================================

class CompressionResult:
    """Result of COBRA model compression."""

    def __init__(self, compressed_model, compression_converter, pre_matrix, post_matrix,
                 reaction_map, metabolite_map, statistics, methods_used,
                 original_reaction_names, original_metabolite_names, flipped_reactions):
        self.compressed_model = compressed_model
        self.compression_converter = compression_converter
        self.pre_matrix = pre_matrix
        self.post_matrix = post_matrix
        self.reaction_map = reaction_map
        self.metabolite_map = metabolite_map
        self.statistics = statistics
        self.methods_used = methods_used
        self.original_reaction_names = original_reaction_names
        self.original_metabolite_names = original_metabolite_names
        self.flipped_reactions = flipped_reactions

    @property
    def compression_ratio(self) -> float:
        return len(self.compressed_model.reactions) / len(self.original_reaction_names)

    @property
    def reactions_removed(self) -> int:
        return len(self.original_reaction_names) - len(self.compressed_model.reactions)

    def summary(self) -> str:
        return f"Compressed {len(self.original_reaction_names)} -> {len(self.compressed_model.reactions)} reactions ({self.compression_ratio:.1%})"


class CompressionConverter:
    """Bidirectional transformer for expressions between original and compressed spaces."""

    def __init__(self, reaction_map: Dict[str, Dict[str, Union[float, Fraction]]],
                 metabolite_map: Dict[str, Dict[str, Union[float, Fraction]]],
                 flipped_reactions: List[str]):
        self.reaction_map = reaction_map
        self.metabolite_map = metabolite_map
        self.flipped_reactions = set(flipped_reactions)

    def expand_expression(self, expression: Dict[str, float],
                          remove_missing: bool = False) -> Dict[str, float]:
        """Transform expression from compressed back to original space."""
        expanded = {}
        for comp_rxn, comp_coeff in expression.items():
            if comp_rxn in self.reaction_map:
                for orig_rxn, scale in self.reaction_map[comp_rxn].items():
                    coeff = comp_coeff * float(scale)
                    if orig_rxn in self.flipped_reactions:
                        coeff = -coeff
                    expanded[orig_rxn] = expanded.get(orig_rxn, 0) + coeff
            elif not remove_missing:
                LOG.warning(f"Compressed reaction {comp_rxn} not found")
        return expanded


def remove_conservation_relations(model) -> None:
    """Remove conservation relations (dependent metabolites) from a model.

    This reduces the number of metabolites while maintaining the original flux space.
    Uses exact rational arithmetic to find linearly independent rows.

    Args:
        model: COBRA model to modify in-place
    """
    # Build the transposed stoichiometric matrix (reactions × metabolites) directly
    # from cobra reaction coefficients — avoids a sparse→dense→sparse round-trip that
    # would iterate over all m×r elements including zeros.
    num_rxns = len(model.reactions)
    num_mets = len(model.metabolites)
    met_index = {m.id: i for i, m in enumerate(model.metabolites)}

    row_idx, col_idx, num_data, den_data = [], [], [], []
    for j, rxn in enumerate(model.reactions):
        for met, coeff in rxn._metabolites.items():
            if coeff == 0:
                continue
            i = met_index[met.id]
            if isinstance(coeff, Fraction):
                frac = coeff
            elif hasattr(coeff, 'numerator'):
                frac = Fraction(coeff.numerator, coeff.denominator)
            else:
                frac = float_to_rational(float(coeff))
            row_idx.append(j)   # reaction → row (transposed layout)
            col_idx.append(i)   # metabolite → column
            num_data.append(frac.numerator)
            den_data.append(frac.denominator)

    from scipy.sparse import csr_matrix as _csr
    num_sparse = _csr((num_data, (row_idx, col_idx)),
                      shape=(num_rxns, num_mets), dtype=np.int64)
    den_sparse = _csr((den_data, (row_idx, col_idx)),
                      shape=(num_rxns, num_mets), dtype=np.int64)
    rm = RationalMatrix._from_sparse(num_sparse, den_sparse)
    basic_mets = basic_columns(rm)

    dependent_mets = [model.metabolites[i].id
                      for i in set(range(num_mets)) - set(basic_mets)]
    for m_id in dependent_mets:
        model.metabolites.get_by_id(m_id).remove_from_model()


def compress_cobra_model(
    model,
    methods: Optional[List[Union[str, CompressionMethod]]] = None,
    in_place: bool = True,
    suppressed_reactions: Optional[Set[str]] = None
) -> CompressionResult:
    """
    Compress a COBRA model using nullspace-based coupling detection.

    Note: Model should be preprocessed first (rational coefficients,
    conservation relations removed). Use networktools.compress_model()
    for automatic preprocessing.

    Args:
        model: COBRA model to compress (should be preprocessed)
        methods: Compression methods. Default: CompressionMethod.standard()
        in_place: Modify original model (True) or work on copy (False)
        suppressed_reactions: Reaction names to exclude from compression

    Returns:
        CompressionResult with compressed model and transformation data
    """
    if not in_place:
        model = copy.deepcopy(model)

    original_reaction_names = [r.id for r in model.reactions]
    original_metabolite_names = [m.id for m in model.metabolites]

    # Parse methods
    if methods is None:
        compression_methods = CompressionMethod.standard()
    else:
        compression_methods = []
        for method in methods:
            if isinstance(method, CompressionMethod):
                compression_methods.append(method)
            elif isinstance(method, str):
                compression_methods.append(CompressionMethod[method.upper()])

    # Flip reactions that can only run backwards
    flipped_reactions = []
    for rxn in model.reactions:
        if rxn.upper_bound <= 0:
            rxn *= -1
            flipped_reactions.append(rxn.id)

    # Build stoichiometric matrix with exact arithmetic
    stoich_matrix = RationalMatrix.from_cobra_model(model)
    reversible = [rxn.reversibility for rxn in model.reactions]
    metabolite_names = [m.id for m in model.metabolites]
    reaction_names = [r.id for r in model.reactions]

    # Run compression
    compressor = StoichMatrixCompressor(*compression_methods)
    compression_record = compressor.compress(
        stoich_matrix, reversible, metabolite_names, reaction_names, suppressed_reactions
    )

    # Apply to model (uses direct manipulation, bypasses solver)
    reaction_map, objective_updates = _apply_compression_to_model(
        model, compression_record, reaction_names
    )

    # Rebuild solver after all direct modifications
    _rebuild_solver(model, objective_updates)

    pre_matrix = compression_record.pre.to_numpy()
    post_matrix = compression_record.post.to_numpy()

    converter = CompressionConverter(reaction_map, {}, flipped_reactions)

    return CompressionResult(
        compressed_model=model,
        compression_converter=converter,
        pre_matrix=pre_matrix,
        post_matrix=post_matrix,
        reaction_map=reaction_map,
        metabolite_map={},
        statistics=compression_record.stats,
        methods_used=compression_methods,
        original_reaction_names=original_reaction_names,
        original_metabolite_names=original_metabolite_names,
        flipped_reactions=flipped_reactions
    )


def _rebuild_solver(model, objective_updates: dict) -> None:
    """Rebuild solver after direct model modifications.

    This creates a fresh solver and populates it with the current model state.
    Must be called after using direct attribute manipulation that bypasses
    solver updates (e.g., _metabolites, _lower_bound, _upper_bound).

    Args:
        model: COBRA model with stale solver state
        objective_updates: dict mapping reactions to their objective coefficients
    """
    # Get solver interface type
    solver_interface = model.solver.interface

    # Create fresh solver
    new_solver = solver_interface.Model()
    model._solver = new_solver

    # Populate solver with current model state
    model._populate_solver(model.reactions, model.metabolites)

    # Set objective coefficients (must be done after solver is populated)
    for rxn, obj_coeff in objective_updates.items():
        if rxn in model.reactions:
            rxn.objective_coefficient = obj_coeff


def _apply_compression_to_model(model, compression_record, original_reaction_names):
    """Apply compression results to COBRA model using exact Fraction arithmetic.

    Sets stoichiometric coefficients and bounds as Fractions directly from
    the compressed matrices, avoiding floating-point arithmetic.

    Uses direct attribute manipulation to bypass solver updates during modification.
    Solver must be rebuilt after calling this function.
    """
    post = compression_record.post
    cmp = compression_record.cmp
    meta_names = compression_record.meta_names
    num_original = post.get_row_count()
    num_compressed = post.get_column_count()
    num_metas = cmp.get_row_count()

    # Track which original reactions to keep (main reactions)
    keep_rxns = [False] * num_original
    reaction_map = {}

    # Build metabolite lookup for compressed metabolites
    met_lookup = {name: model.metabolites.get_by_id(name) for name in meta_names}

    # Track objective coefficients to set at end (bypass solver updates)
    objective_updates = {}

    for j in range(num_compressed):
        # Find contributing original reactions from POST matrix (sparse iteration)
        contributing = list(post.iter_column_fractions(j))

        if not contributing:
            continue

        # Select "main" reaction: prefer one with non-zero objective coefficient
        main_idx = contributing[0][0]
        for idx, _ in contributing:
            if model.reactions[idx].objective_coefficient != 0:
                main_idx = idx
                break
        main_rxn = model.reactions[main_idx]
        keep_rxns[main_idx] = True

        # Scale objective coefficient by POST factors (store for later)
        combined_obj = Fraction(0)
        for idx, coeff in contributing:
            obj_coeff = model.reactions[idx].objective_coefficient
            if obj_coeff != 0:
                combined_obj += Fraction(obj_coeff).limit_denominator(10**12) * coeff
        objective_updates[main_rxn] = float(combined_obj)

        # Store subset info
        main_rxn.subset_rxns = [idx for idx, _ in contributing]
        main_rxn.subset_stoich = [coeff for _, coeff in contributing]

        # Build combined reaction name from contributing reactions
        for idx, _ in contributing[1:]:
            rxn = model.reactions[idx]
            if len(main_rxn.id) + len(rxn.id) < 220 and not main_rxn.id.endswith('...'):
                main_rxn.id += '*' + rxn.id
            elif not main_rxn.id.endswith('...'):
                main_rxn.id += '...'

        # Build new metabolites dict from cmp matrix (sparse iteration)
        new_metabolites = {}
        for m_idx, frac in cmp.iter_column_fractions(j):
            met = met_lookup[meta_names[m_idx]]
            new_metabolites[met] = frac

        # OPTIMIZATION: Direct _metabolites assignment (bypasses solver updates)
        # Clear old metabolite back-references
        for met in list(main_rxn._metabolites.keys()):
            if main_rxn in met._reaction:
                met._reaction.discard(main_rxn)
        # Set new metabolites directly
        main_rxn._metabolites = new_metabolites
        # Update new metabolite back-references
        for met in new_metabolites:
            met._reaction.add(main_rxn)

        # Compute bounds as Fractions from original reactions scaled by POST factors
        lb_candidates = []
        ub_candidates = []
        for idx, coeff in contributing:
            rxn = model.reactions[idx]
            lb, ub = rxn.lower_bound, rxn.upper_bound

            # v_original = coeff * v_compressed (from POST interpretation)
            # constraint: lb <= v_original <= ub
            # so: lb <= coeff * v_compressed <= ub
            # if coeff > 0: lb/coeff <= v_compressed <= ub/coeff
            # if coeff < 0: ub/coeff <= v_compressed <= lb/coeff (inequality flips)

            if coeff > 0:
                if lb != -float('inf'):
                    lb_candidates.append(Fraction(lb) / coeff if lb != 0 else Fraction(0))
                if ub != float('inf'):
                    ub_candidates.append(Fraction(ub) / coeff if ub != 0 else Fraction(0))
            else:  # coeff < 0
                if ub != float('inf'):
                    lb_candidates.append(Fraction(ub) / coeff if ub != 0 else Fraction(0))
                if lb != -float('inf'):
                    ub_candidates.append(Fraction(lb) / coeff if lb != 0 else Fraction(0))

        # OPTIMIZATION: Direct bounds assignment (bypasses solver updates)
        main_rxn._lower_bound = float(max(lb_candidates)) if lb_candidates else -float('inf')
        main_rxn._upper_bound = float(min(ub_candidates)) if ub_candidates else float('inf')

        # Build reaction map
        reaction_map[main_rxn.id] = {
            original_reaction_names[idx]: coeff for idx, coeff in contributing
        }

    # OPTIMIZATION: Batch reaction and metabolite removal
    # Use reaction objects directly (not IDs, since IDs are modified during compression)
    from cobra import DictList

    # Build set of reaction objects to keep
    keep_rxn_objs = {model.reactions[i] for i in range(num_original) if keep_rxns[i]}

    # Find metabolites that will be in kept reactions
    mets_in_kept_rxns = set()
    for rxn in keep_rxn_objs:
        mets_in_kept_rxns.update(rxn._metabolites.keys())

    # Build new reactions list (only kept reactions)
    new_reactions = DictList()
    for rxn in model.reactions:
        if rxn in keep_rxn_objs:
            new_reactions.append(rxn)
        else:
            # Clear metabolite back-references for removed reaction
            for met in list(rxn._metabolites.keys()):
                if rxn in met._reaction:
                    met._reaction.discard(rxn)
            rxn._model = None

    # Build new metabolites list (only those in kept reactions)
    new_metabolites = DictList()
    for met in model.metabolites:
        if met in mets_in_kept_rxns:
            new_metabolites.append(met)
        else:
            met._model = None

    # Replace model lists (using __dict__ to bypass any property setters)
    model.__dict__['reactions'] = new_reactions
    model.__dict__['metabolites'] = new_metabolites

    # Update model references
    for rxn in model.reactions:
        rxn._model = model
    for met in model.metabolites:
        met._model = model

    return reaction_map, objective_updates


# =============================================================================
# Preprocessing Functions
# =============================================================================

def remove_blocked_reactions(model) -> List:
    """Remove blocked reactions (bounds == (0, 0)) from a network."""
    blocked_reactions = [reac for reac in model.reactions if reac.bounds == (0, 0)]
    model.remove_reactions(blocked_reactions)
    return blocked_reactions


def remove_ext_mets(model) -> None:
    """Remove external metabolites from 'External_Species' compartment."""
    external_mets = [m for m in model.metabolites if m.compartment == 'External_Species']
    model.remove_metabolites(external_mets)
    stoich_mat = create_stoichiometric_matrix(model)
    obsolete_reacs = [r for r, has_nonzero in zip(model.reactions, np.any(stoich_mat, 0)) if not has_nonzero]
    model.remove_reactions(obsolete_reacs)


def remove_dummy_bounds(model) -> None:
    """Replace COBRA standard bounds with +/-inf."""
    cobra_conf = Configuration()
    bound_thres = max(abs(cobra_conf.lower_bound), abs(cobra_conf.upper_bound))
    if any(any(abs(b) >= bound_thres for b in r.bounds) for r in model.reactions):
        LOG.warning(f'Removing reaction bounds >= {round(bound_thres)}.')
        for rxn in model.reactions:
            if rxn.lower_bound <= -bound_thres:
                rxn.lower_bound = -np.inf
            if rxn.upper_bound >= bound_thres:
                rxn.upper_bound = np.inf


def stoichmat_coeff2rational(model) -> None:
    """Convert stoichiometric coefficients to rational numbers."""
    for rxn in model.reactions:
        for met, coeff in rxn._metabolites.items():
            if isinstance(coeff, (float, int)):
                rxn._metabolites[met] = float_to_rational(coeff)
            elif not hasattr(coeff, 'p'):  # Not sympy.Rational
                if hasattr(coeff, 'numerator'):  # fractions.Fraction
                    rxn._metabolites[met] = Rational(coeff.numerator, coeff.denominator)
                else:
                    raise TypeError(f"Unsupported coefficient type: {type(coeff)}")


def stoichmat_coeff2float(model) -> None:
    """Convert stoichiometric coefficients to floats."""
    for rxn in model.reactions:
        for met, coeff in rxn._metabolites.items():
            rxn._metabolites[met] = float(coeff)


# =============================================================================
# High-Level Compression API
# =============================================================================

def compress_model(model, no_par_compress_reacs=set(), backend='sparse'):
    """Compress a metabolic model using multiple techniques.

    Performs blocked reaction removal, conservation relation removal, and
    alternating dependent/parallel reaction lumping until no further
    compression is possible.

    Args:
        model: COBRA model to compress in-place
        no_par_compress_reacs: Reactions exempt from parallel compression
        backend: Compression backend to use:
            - 'sparse' (default): Pure Python integer RREF with column/row
              pre-sorting. No external dependencies beyond NumPy/SciPy.
            - 'efmtool': Java-based EFMTool via JPype. Requires a JVM and
              the jpype1 package.

    Returns:
        list of dict: Compression maps for reversing each compression step
    """
    # Suppress optlang LP coefficient updates during compression.
    #
    # compress_model_efmtool calls _rebuild_solver after each iteration,
    # which calls _populate_solver -> constraint.set_linear_coefficients
    # for every metabolite.  On Gurobi this involves _get_expression (a
    # full LP-row read) and dominates runtime for large/GPR-extended models.
    #
    # Fix: patch set_linear_coefficients to a no-op for the duration of
    # compression so that intermediate rebuilds only maintain LP structure
    # (variables, constraints, bounds) without re-writing coefficients.
    # One final _populate_solver call at the end sets the correct
    # stoichiometric coefficients from the compressed model state.
    #
    # Guards: all accesses are hasattr-checked so that cobra/optlang API
    # changes simply disable the optimisation without breaking compression.
    _slc_cls = None   # the Constraint class we patched
    _slc_orig = None  # the original set_linear_coefficients method
    try:
        if hasattr(model, 'problem') and hasattr(model.problem, 'Constraint'):
            _slc_cls = model.problem.Constraint
            if hasattr(_slc_cls, 'set_linear_coefficients'):
                _slc_orig = _slc_cls.set_linear_coefficients
                _slc_cls.set_linear_coefficients = lambda self, *a, **kw: None
    except Exception:
        _slc_cls = _slc_orig = None

    cmp_mapReac = []
    try:
        use_java = (backend == 'efmtool')
        LOG.info('  Removing blocked reactions.')
        remove_blocked_reactions(model)
        LOG.info('  Converting coefficients to rationals.')
        stoichmat_coeff2rational(model)
        LOG.info('  Removing conservation relations.')
        if use_java:
            _remove_conservation_relations_java(model)
        else:
            remove_conservation_relations(model)

        parallel = False
        run = 1
        numr = len(model.reactions)

        while True:
            if not parallel:
                LOG.info(f'  Compression {run}: Applying efmtool compression.')
                reac_map_exp = compress_model_efmtool(model, backend)
                for new_reac, old_reac_val in reac_map_exp.items():
                    old_reacs_no_compress = [r for r in no_par_compress_reacs if r in old_reac_val]
                    if old_reacs_no_compress:
                        for r in old_reacs_no_compress:
                            no_par_compress_reacs.remove(r)
                        no_par_compress_reacs.add(new_reac)
            else:
                LOG.info(f'  Compression {run}: Lumping parallel reactions.')
                reac_map_exp = compress_model_parallel(model, no_par_compress_reacs)

            if use_java:
                _remove_conservation_relations_java(model)
            else:
                remove_conservation_relations(model)

            if numr > len(reac_map_exp):
                LOG.info(f'  Reduced to {len(reac_map_exp)} reactions.')
                cmp_mapReac.append({
                    "reac_map_exp": reac_map_exp,
                    "parallel": parallel,
                })
                parallel = not parallel
                run += 1
                numr = len(reac_map_exp)
            else:
                LOG.info(f'  No further reduction ({numr} reactions).')
                LOG.info(f'  Compression complete ({run - 1} iterations).')
                break

        stoichmat_coeff2float(model)
    finally:
        # Step 1: Always restore set_linear_coefficients, even if compression raises.
        if _slc_orig is not None and _slc_cls is not None:
            _slc_cls.set_linear_coefficients = _slc_orig

        # Step 2: Single final LP rebuild with correct stoichiometric coefficients.
        # _populate_solver cannot be called on the existing solver (it would try to
        # re-add constraints already in the pending queue -> ContainerAlreadyContains).
        # _rebuild_solver creates a fresh optlang model first, then populates it.
        # We must collect the objective from the current (still valid) LP beforehand.
        if _slc_orig is not None:
            try:
                obj_updates = {}
                for rxn in model.reactions:
                    try:
                        c = rxn.objective_coefficient
                        if c != 0:
                            obj_updates[rxn] = c
                    except Exception:
                        pass
                _rebuild_solver(model, obj_updates)
            except Exception:
                pass  # Structural compression succeeded; LP may need manual resync.

    return cmp_mapReac


def _remove_conservation_relations_java(model) -> None:
    """Remove conservation relations using Java efmtool."""
    from . import efmtool_cmp_interface as efm
    stoich_mat = create_stoichiometric_matrix(model, array_type='lil')
    basic_mets = efm.basic_columns_rat_java(stoich_mat.transpose().toarray(), tolerance=0)
    dependent_mets = [model.metabolites[i].id
                      for i in set(range(len(model.metabolites))) - set(basic_mets)]
    for m_id in dependent_mets:
        model.metabolites.get_by_id(m_id).remove_from_model()


def compress_model_efmtool(model, backend='sparse'):
    """Compress by lumping dependent reactions (efmtool approach).

    Args:
        model: COBRA model to compress in-place
        backend: 'sparse' (default, Python) or 'efmtool' (Java legacy)

    Returns:
        dict: Mapping {compressed_id: {orig_id: factor, ...}}
    """
    if backend == 'efmtool':
        from .efmtool_cmp_interface import compress_model_java
        return compress_model_java(model)

    # Clear gene rules to match Java behavior
    for r in model.reactions:
        r.gene_reaction_rule = ''

    result = compress_cobra_model(
        model,
        methods=CompressionMethod.standard(),
        in_place=True
    )

    # Account for flipped reactions
    flipped = set(result.flipped_reactions)
    return {
        cmp_id: {orig_id: c * (-1 if orig_id in flipped else 1)
                 for orig_id, c in orig_map.items()}
        for cmp_id, orig_map in result.reaction_map.items()
    }


def compress_model_parallel(model, protected_rxns=set()):
    """Compress by lumping parallel reactions.

    Args:
        model: COBRA model to compress in-place
        protected_rxns: Reactions exempt from parallel compression

    Returns:
        dict: Mapping {compressed_id: {orig_id: factor, ...}}
    """
    old_num_reac = len(model.reactions)
    old_objective = [r.objective_coefficient for r in model.reactions]
    old_reac_ids = [r.id for r in model.reactions]

    stoichmat_T = create_stoichiometric_matrix(model, 'lil').transpose()
    factor = [d[0] if d else 1.0 for d in stoichmat_T.data]
    A = sparse.diags(factor) @ stoichmat_T

    lb = [float(r.lower_bound) for r in model.reactions]
    ub = [float(r.upper_bound) for r in model.reactions]

    fwd = sparse.lil_matrix([1. if (isinf(u) and f > 0 or isinf(l) and f < 0) else 0.
                             for f, l, u in zip(factor, lb, ub)]).transpose()
    rev = sparse.lil_matrix([1. if (isinf(l) and f > 0 or isinf(u) and f < 0) else 0.
                             for f, l, u in zip(factor, lb, ub)]).transpose()
    inh = sparse.lil_matrix([i + 1 if not ((isinf(ub[i]) or ub[i] == 0) and
                                           (isinf(lb[i]) or lb[i] == 0))
                             else 0 for i in range(len(model.reactions))]).transpose()
    A = sparse.hstack((A, fwd, rev, inh), 'csr')

    # Find parallel reactions via hash comparison
    subset_list = []
    prev_found = []
    protected = [r.id in protected_rxns for r in model.reactions]
    hashes = [hash((tuple(A[i].indices), tuple(A[i].data))) for i in range(A.shape[0])]

    for i in range(A.shape[0]):
        if i in prev_found:
            continue
        if protected[i]:
            subset_list.append([i])
            continue
        subset_i = [i]
        for j in range(i + 1, A.shape[0]):
            if not protected[j] and j not in prev_found and hashes[i] == hashes[j]:
                subset_i.append(j)
                prev_found.append(j)
        subset_list.append(subset_i)

    # Lump parallel reactions
    del_rxns = [False] * len(model.reactions)
    for rxn_idx in subset_list:
        for i in range(1, len(rxn_idx)):
            main_rxn = model.reactions[rxn_idx[0]]
            other_rxn = model.reactions[rxn_idx[i]]
            if len(main_rxn.id) + len(other_rxn.id) < 220 and main_rxn.id[-3:] != '...':
                main_rxn.id += '*' + other_rxn.id
            elif main_rxn.id[-3:] != '...':
                main_rxn.id += '...'
            del_rxns[rxn_idx[i]] = True

    del_indices = np.where(del_rxns)[0]
    for i in reversed(del_indices):
        model.reactions[i].remove_from_model(remove_orphans=True)

    # Build compression map
    rational_map = {}
    subT = np.zeros((old_num_reac, len(model.reactions)))
    for i in range(subT.shape[1]):
        for j in subset_list[i]:
            subT[j, i] = 1
        rational_map[model.reactions[i].id] = {old_reac_ids[j]: Rational(1) for j in subset_list[i]}

    # Update objective
    new_objective = old_objective @ subT
    for r, c in zip(model.reactions, new_objective):
        r.objective_coefficient = c

    return rational_map


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Rational matrix and utilities
    'RationalMatrix',
    'float_to_rational',
    'detect_max_precision',
    'nullspace',
    'basic_columns',
    'basic_columns_from_numpy',
    # Core compression
    'compress_cobra_model',
    'CompressionResult',
    'CompressionRecord',
    'CompressionStatistics',
    'CompressionMethod',
    'CompressionConverter',
    'StoichMatrixCompressor',
    # High-level API
    'compress_model',
    'compress_model_efmtool',
    'compress_model_parallel',
    # Preprocessing
    'remove_blocked_reactions',
    'remove_ext_mets',
    'remove_conservation_relations',
    'remove_dummy_bounds',
    'stoichmat_coeff2rational',
    'stoichmat_coeff2float',
]
