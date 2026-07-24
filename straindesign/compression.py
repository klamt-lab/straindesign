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

import ast
import copy
import logging
import re
import numpy as np
from enum import Enum
from functools import reduce
from math import gcd, lcm, isinf
from typing import Dict, Iterator, List, Optional, Set, Tuple, Union, Any

from collections import namedtuple
from fractions import Fraction
from scipy import sparse
from scipy.sparse import csr_matrix, csc_matrix
from cobra import Configuration
from cobra.util.array import create_stoichiometric_matrix

LOG = logging.getLogger(__name__)

# LP suppression utilities live in networktools.  Imports are deferred to
# function bodies to avoid the circular dependency (networktools re-exports
# compression symbols).

# =============================================================================
# Utility Functions
# =============================================================================


def float_to_fraction(val, max_precision: int = 6, max_denom: int = 100) -> Fraction:
    """Convert float to Fraction with bounded denominators."""
    if isinstance(val, Fraction):
        return val
    import numbers
    if isinstance(val, numbers.Rational):
        return Fraction(val.numerator, val.denominator)
    # Handle sympy.Float and other numeric types that Fraction() doesn't accept directly
    if not isinstance(val, (int, float)):
        val = float(val)
    if val == 0:
        return Fraction(0)
    if val == int(val):
        return Fraction(int(val))

    small_frac = Fraction(val).limit_denominator(max_denom)
    if round(float(small_frac), max_precision) == round(val, max_precision):
        return small_frac

    denom = 10**max_precision
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


_INT64_MAX = (1 << 63) - 1


def _fits_int64(values) -> bool:
    """True if every value is within the signed-int64 range (scipy's largest integer dtype)."""
    return all(-_INT64_MAX <= int(v) <= _INT64_MAX for v in values)


# Big-integer-safe sparse export (scipy sparse cannot hold >int64). Every basis entry equals
# data[k] / denom exactly, at position (rows[k], cols[k]); data are arbitrary-precision Python ints.
ExactCOO = namedtuple("ExactCOO", ["rows", "cols", "data", "shape", "denom"])


class RationalMatrix:
    """Sparse rational matrix using dual int sparse storage (numerators + denominators)."""

    def __init__(self, rows: int, cols: int):
        self._rows = rows
        self._cols = cols
        self._num_sparse: Optional[csr_matrix] = None  # Numerators
        self._den_sparse: Optional[csr_matrix] = None  # Denominators
        self._csc_cache: Optional[csc_matrix] = None
        self._batch_mode = False
        # Big-integer fallback: scipy.sparse only holds <=64-bit ints, so exact results whose
        # coefficients exceed int64 (e.g. yeast-GEM's nullspace) are stored here as {row:{col:Fraction}}
        # and _num_sparse/_den_sparse stay None. See is_bigint() / to_coo_exact().
        self._dict_frac: Optional[Dict[int, Dict[int, Fraction]]] = None

    def _invalidate_cache(self):
        if not self._batch_mode:
            self._csc_cache = None

    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------

    @classmethod
    def _from_sparse(cls,
                     num_sparse: csr_matrix,
                     den_sparse: csr_matrix,
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
    def from_numpy(cls, arr: np.ndarray, max_precision: int = 6, max_denom: int = 100) -> 'RationalMatrix':
        """Create RationalMatrix from numpy array."""
        rows, cols = arr.shape
        row_idx, col_idx, num_data, den_data = [], [], [], []

        for r in range(rows):
            for c in range(cols):
                val = arr[r, c]
                if val != 0:
                    frac = float_to_fraction(val, max_precision, max_denom)
                    row_idx.append(r)
                    col_idx.append(c)
                    num_data.append(frac.numerator)
                    den_data.append(frac.denominator)

        num_sparse = csr_matrix((num_data, (row_idx, col_idx)), shape=(rows, cols), dtype=np.int64)
        den_sparse = csr_matrix((den_data, (row_idx, col_idx)), shape=(rows, cols), dtype=np.int64)
        return cls._from_sparse(num_sparse, den_sparse)

    @classmethod
    def from_cobra_model(cls, model, max_precision: int = 6, max_denom: int = 100) -> 'RationalMatrix':
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
                        frac = float_to_fraction(coeff, max_precision, max_denom)

                    row_idx.append(i)
                    col_idx.append(j)
                    num_data.append(frac.numerator)
                    den_data.append(frac.denominator)

        num_sparse = csr_matrix((num_data, (row_idx, col_idx)), shape=(num_mets, num_rxns), dtype=np.int64)
        den_sparse = csr_matrix((den_data, (row_idx, col_idx)), shape=(num_mets, num_rxns), dtype=np.int64)
        return cls._from_sparse(num_sparse, den_sparse)

    @classmethod
    def _build_from_sparse_data(cls, row_indices: List[int], col_indices: List[int], numerators: List[int], denominators: List[int],
                                num_rows: int, num_cols: int) -> 'RationalMatrix':
        """Build RationalMatrix from sparse coordinate data.

        These numerators/denominators are exact nullspace values (subdeterminants) that can exceed
        int64 on large/dense networks (e.g. yeast-GEM). scipy.sparse cannot hold >64-bit integers,
        so in that case we fall back to a dict-of-Fractions store; otherwise use the fast int64 CSR.
        """
        if _fits_int64(numerators) and _fits_int64(denominators):
            num_sparse = csr_matrix((numerators, (row_indices, col_indices)), shape=(num_rows, num_cols), dtype=np.int64)
            den_sparse = csr_matrix((denominators, (row_indices, col_indices)), shape=(num_rows, num_cols), dtype=np.int64)
            return cls._from_sparse(num_sparse, den_sparse)
        # Big-integer mode: store exact Fractions in a dict-of-dicts (scipy-free).
        result = cls(num_rows, num_cols)
        dic: Dict[int, Dict[int, Fraction]] = {}
        for r, c, n, d in zip(row_indices, col_indices, numerators, denominators):
            if n != 0:
                dic.setdefault(int(r), {})[int(c)] = Fraction(int(n), int(d))
        result._dict_frac = dic
        return result

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
        return RationalMatrix._from_sparse(self._num_sparse.copy(), self._den_sparse.copy())

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

    def add_scaled_column(self, dst_col: int, src_col: int, scalar_num: int, scalar_den: int) -> None:
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

    def scale_column(self, col: int, scalar_num: int, scalar_den: int) -> None:
        """Multiply a column by a scalar: col[i] *= scalar_num/scalar_den."""
        if scalar_num == 0 or scalar_num == scalar_den:
            return
        num_lil, den_lil = self._num_sparse, self._den_sparse
        num_csc = num_lil.tocsc() if num_lil.format != 'csc' else num_lil
        den_csc = den_lil.tocsc() if den_lil.format != 'csc' else den_lil
        entries = [(num_csc.indices[i], int(num_csc.data[i]), int(den_csc.data[i]))
                   for i in range(num_csc.indptr[col], num_csc.indptr[col + 1])]
        for row, cur_num, cur_den in entries:
            if cur_num == 0:
                continue
            new_num, new_den = cur_num * scalar_num, cur_den * scalar_den
            if new_den < 0:
                new_num, new_den = -new_num, -new_den
            g = gcd(abs(new_num), new_den)
            if g:
                new_num //= g
                new_den //= g
            num_lil[row, col] = new_num
            den_lil[row, col] = new_den if new_num != 0 else 0
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
        if self.is_bigint():
            raise OverflowError(
                "coefficients exceed int64 and cannot be stored in a scipy sparse matrix; "
                "use to_coo_exact() / to_sparse_pattern(), or the public sparse_nullspace() helper "
                "which returns an ExactCOO in that case.")
        # Empty matrix (e.g. a 0-dimensional nullspace)
        if self._den_sparse is None:
            return csr_matrix((self._rows, self._cols), dtype=np.int64), 1
        dens = self._den_sparse.data
        if len(dens) == 0:
            return csr_matrix((self._rows, self._cols), dtype=np.int64), 1

        common_denom = _lcm_list([int(d) for d in dens if d != 0])
        coo_num = self._num_sparse.tocoo()
        coo_den = self._den_sparse.tocoo()
        scaled_data = [int(num) * (common_denom // int(den)) if num != 0 else 0
                       for num, den in zip(coo_num.data, coo_den.data)]
        scaled = csr_matrix((scaled_data, (coo_num.row, coo_num.col)), shape=(self._rows, self._cols), dtype=np.int64)
        return scaled, common_denom

    def is_bigint(self) -> bool:
        """True if this matrix is stored in big-integer mode (coefficients exceed int64, so it is
        kept as a dict of exact Fractions rather than scipy int64 sparse)."""
        return self._dict_frac is not None

    def to_coo_exact(self) -> 'ExactCOO':
        """Big-integer-safe exact sparse export, valid in both storage modes.

        Returns ExactCOO(rows, cols, data, shape, denom) where entry (rows[k], cols[k]) equals
        data[k] / denom exactly. data are arbitrary-precision Python ints (no int64 ceiling).
        """
        rows: List[int] = []
        cols: List[int] = []
        fracs: List[Fraction] = []
        if self._dict_frac is not None:
            for r, cd in self._dict_frac.items():
                for c, f in cd.items():
                    rows.append(int(r)); cols.append(int(c)); fracs.append(f)
        elif self._num_sparse is not None:
            coo_num = self._num_sparse.tocoo()
            coo_den = self._den_sparse.tocoo()
            for r, c, n, d in zip(coo_num.row, coo_num.col, coo_num.data, coo_den.data):
                if n != 0:
                    rows.append(int(r)); cols.append(int(c)); fracs.append(Fraction(int(n), int(d)))
        denom = _lcm_list([f.denominator for f in fracs]) if fracs else 1
        data = [int(f.numerator) * (denom // f.denominator) for f in fracs]
        return ExactCOO(rows, cols, data, (self._rows, self._cols), denom)

    def to_sparse_pattern(self) -> Tuple[csr_matrix, Dict[int, Dict[int, Fraction]]]:
        """Return sparsity pattern as CSR and row-wise Fraction data.

        For pattern analysis (coupled reaction detection) without integer overflow. Works in both the
        int64 and big-integer (dict-of-Fractions) storage modes.
        Returns:
            pattern: CSR matrix with 1s at non-zero positions (for indptr/indices)
            row_data: {row: {col: Fraction}} for actual values
        """
        if self._dict_frac is not None:
            ai, aj = [], []
            row_data: Dict[int, Dict[int, Fraction]] = {}
            for r, cd in self._dict_frac.items():
                rd = {}
                for c, f in cd.items():
                    if f != 0:
                        ai.append(int(r)); aj.append(int(c)); rd[int(c)] = f
                if rd:
                    row_data[int(r)] = rd
            pattern = csr_matrix(([1] * len(ai), (ai, aj)), shape=(self._rows, self._cols), dtype=np.int8)
            return pattern, row_data

        # Build pattern matrix (just 1s for structure)
        coo_num = self._num_sparse.tocoo()
        pattern_data = [1] * len(coo_num.data)
        pattern = csr_matrix((pattern_data, (coo_num.row, coo_num.col)), shape=(self._rows, self._cols), dtype=np.int8)

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
    Rows are pre-sorted by nnz ascending, and pivot rows are selected by Markowitz
    criterion (sparsest row first, tie-break by smallest absolute value).
    Results are translated back to the original column ordering before return.

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

    # Active (not-yet-pivoted) rows, and a column -> {rows containing it} index over the active rows.
    # After the echelon change made elimination cheap, the Markowitz pivot search dominates (~89% on
    # iML1515). The index turns the search from "scan every active row, test membership" (~99.9% of
    # tests miss) into "visit only the rows that actually contain the pivot column". Its maintenance
    # cost is proportional to the (now small) elimination fill, so it is a net win here — whereas
    # pre-echelon, when elimination dominated, it was a wash.
    active = set(data.keys())
    col_rows: Dict[int, set] = {}
    for r, rd in data.items():
        for c in rd:
            s = col_rows.get(c)
            if s is None:
                col_rows[c] = {r}
            else:
                s.add(r)

    def _eliminate(prd, pivot_val, pivot_col, targets, index):
        """Fraction-free elimination of ``pivot_col`` from each (row_key, entry) in ``targets`` using
        pivot row ``prd``. Mutates ``data`` (updates/deletes each target row). If ``index`` is True,
        keeps ``active``/``col_rows`` in sync (forward phase); back-substitution passes False since the
        index is no longer needed once the forward phase is done.

        new[c] = elim[c]*pv_scaled - ev_scaled*pivot[c], with pv_scaled/ev_scaled pre-divided by
        gcd(pivot_val, elim_val) to keep the products small, then the whole row reduced by its content
        GCD. The content reduction keeps coefficient growth polynomial — without it, cross-
        multiplication grows exponentially.
        """
        for elim_row, elim_val in targets:
            elim_row_data = data[elim_row]
            old_cols = set(elim_row_data) if index else None
            g = gcd(pivot_val, elim_val)
            pv_scaled = pivot_val // g
            ev_scaled = elim_val // g
            # Scale the whole elim row by pv_scaled in one comprehension (every product is nonzero),
            # then correct the few columns it shares with the (sparse) pivot row.
            new_row = {c: v * pv_scaled for c, v in elim_row_data.items()}
            for c, p_val in prd.items():
                new_val = new_row.get(c, 0) - ev_scaled * p_val
                if new_val != 0:
                    new_row[c] = new_val
                elif c in new_row:
                    del new_row[c]
            if new_row:
                data[elim_row] = new_row
                row_gcd = gcd(*new_row.values())
                if row_gcd > 1:
                    for c in new_row:
                        new_row[c] //= row_gcd
                if index:
                    new_cols = new_row.keys()
                    for c in new_cols - old_cols:
                        s = col_rows.get(c)
                        if s is None:
                            col_rows[c] = {elim_row}
                        else:
                            s.add(elim_row)
                    for c in old_cols - new_cols:
                        col_rows[c].discard(elim_row)
            else:
                del data[elim_row]
                if index:
                    active.discard(elim_row)
                    for c in old_cols:
                        col_rows[c].discard(elim_row)

    # ---- Phase 1: forward elimination to row-echelon form ----
    # Eliminate each pivot only from rows BELOW its pivot row, so already-processed pivot rows stay
    # sparse. Full Gauss-Jordan (eliminating upward too) re-reduces those filled rows with every later
    # pivot — ~99% of the total work on iML1515. The reduced form is recovered in phase 2. Rows are not
    # renumbered/swapped to pivot positions; instead pivot_keys[i] records the data-key of pivot i.
    pivot_cols_sorted = []
    pivot_keys = []
    for pivot_col in range(cols):
        if len(pivot_keys) >= rows:
            break

        holders = col_rows.get(pivot_col)
        if not holders:
            continue

        # Markowitz among the rows that actually contain this column: sparsest, then smallest abs.
        best_row, best_val = -1, 0
        best_nnz, best_abs = float('inf'), float('inf')
        for r in holders:
            row_data = data[r]
            v = row_data[pivot_col]
            rnnz = len(row_data)
            if rnnz < best_nnz or (rnnz == best_nnz and abs(v) < best_abs):
                best_row, best_val = r, v
                best_nnz, best_abs = rnnz, abs(v)

        pivot_row_data = data[best_row]
        pivot_cols_sorted.append(pivot_col)
        pivot_keys.append(best_row)

        # Retire the pivot row from the active index (remove it from every column's holder set).
        active.discard(best_row)
        for c in pivot_row_data:
            col_rows[c].discard(best_row)

        # Eliminate the pivot column from the remaining active rows that contain it (= col_rows now).
        targets = [(r, data[r][pivot_col]) for r in list(col_rows.get(pivot_col, ()))]
        _eliminate(pivot_row_data, best_val, pivot_col, targets, True)

    # ---- Phase 2: back-substitution to reduced row-echelon form ----
    # Process pivots last-to-first, clearing each pivot column from the pivot rows ABOVE it. In this
    # order each pivot row's later-pivot-column entries are already cleared, so back-substitution only
    # introduces free-column fill — far less than Gauss-Jordan (iML1515: ~0.8M ops vs ~9.4M).
    #
    # `pivcol_holders[c]` = pivot indices j whose row contains pivot column c (built once over the
    # still-sparse post-forward pivot rows). It replaces an O(rank^2) "which rows above contain pcol"
    # scan. Crucially, back-substitution only ever *removes* pivot-column entries (it adds free-column
    # fill, and the pivot row being applied has had its own later-pivot columns already cleared), so
    # the index needs no additions during phase 2 — only the discards after each step.
    rank = len(pivot_cols_sorted)
    pivot_col_set = set(pivot_cols_sorted)
    pivcol_holders: Dict[int, set] = {}
    for j, key in enumerate(pivot_keys):
        for c in data[key]:
            if c in pivot_col_set:
                pivcol_holders.setdefault(c, set()).add(j)
    for i in range(rank - 1, -1, -1):
        pcol = pivot_cols_sorted[i]
        prd = data[pivot_keys[i]]
        pval = prd[pcol]
        above = [j for j in pivcol_holders.get(pcol, ()) if j < i]
        targets = [(pivot_keys[j], data[pivot_keys[j]][pcol]) for j in above]
        _eliminate(prd, pval, pcol, targets, False)
        holders = pivcol_holders.get(pcol)
        if holders:
            holders.difference_update(above)   # pcol is now cleared from those rows

    # Final GCD reduction of the pivot rows (insurance; rows are already reduced per step).
    for key in pivot_keys:
        row_data = data[key]
        row_gcd = gcd(*row_data.values())
        if row_gcd > 1:
            for c in row_data:
                row_data[c] //= row_gcd

    # Translate results back to original column space, keyed by pivot index (rref_data[i] = pivot i).
    original_data = {i: {col_order[sc]: v for sc, v in data[key].items()}
                     for i, key in enumerate(pivot_keys)}
    pivot_cols_original = [col_order[p] for p in pivot_cols_sorted]

    return original_data, rank, pivot_cols_original



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

    return RationalMatrix._build_from_sparse_data(row_indices, col_indices, numerators, denominators, cols, nullity)


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


def sparse_nullspace(matrix):
    """Exact rational nullspace of a sparse integer/rational matrix, returned as a sparse basis.

    Computes an exact basis of the right nullspace (kernel) ``K`` with ``matrix @ K == 0`` exactly
    over the rationals -- no floating-point error. The basis is sparse (one column per free
    variable), the property that makes it useful for network reduction and pathway analysis.

    Parameters
    ----------
    matrix : scipy.sparse matrix, numpy.ndarray, or RationalMatrix
        Input with integer or rational entries. Floats are converted to nearby rationals.

    Returns
    -------
    scipy.sparse.csr_matrix
        Integer nullspace basis (columns are basis vectors), scaled to a common integer
        denominator; returned when all coefficients fit in signed 64-bit integers.
    ExactCOO
        Namedtuple ``(rows, cols, data, shape, denom)`` returned instead when the exact
        coefficients exceed 64 bits (scipy sparse cannot store them). Entry ``(rows[k], cols[k])``
        equals ``data[k] / denom`` exactly, with arbitrary-precision Python-int ``data``.
    """
    if isinstance(matrix, RationalMatrix):
        rm = matrix
    elif sparse.issparse(matrix):
        A = matrix.tocsr()
        if np.issubdtype(A.dtype, np.integer):
            num = csr_matrix((A.data.astype(np.int64), A.indices.copy(), A.indptr.copy()), shape=A.shape)
            den = csr_matrix((np.ones(A.nnz, dtype=np.int64), A.indices.copy(), A.indptr.copy()), shape=A.shape)
            rm = RationalMatrix._from_sparse(num, den)
        else:
            rm = RationalMatrix.from_numpy(A.toarray())
    else:
        rm = RationalMatrix.from_numpy(np.asarray(matrix))
    K = nullspace(rm)
    if K.is_bigint():
        return K.to_coo_exact()
    csr, _ = K.to_sparse_csr()
    return csr


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
        self.coupled_count = 0
        self.unused_metabolite_count = 0

    def inc_compression_iteration(self) -> int:
        self.iteration_count += 1
        return self.iteration_count

    def get_compression_iteration(self) -> int:
        return self.iteration_count

    def inc_zero_flux_reactions(self) -> None:
        self.zero_flux_count += 1

    def inc_coupled_reactions_count(self, count: int) -> None:
        self.coupled_count += count

    def inc_unused_metabolite(self) -> None:
        self.unused_metabolite_count += 1

    def write_to_log(self) -> None:
        LOG.info(f"Compression complete: {self.iteration_count} iterations, "
                 f"{self.zero_flux_count} zero-flux, "
                 f"{self.coupled_count} coupled, {self.unused_metabolite_count} unused metabolites")

    def __repr__(self):
        return (f"CompressionStatistics(iterations={self.iteration_count}, "
                f"zero_flux={self.zero_flux_count}, "
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

    def __init__(self,
                 pre: RationalMatrix,
                 cmp: RationalMatrix,
                 post: RationalMatrix,
                 meta_names: List[str],
                 stats: Optional[CompressionStatistics] = None):
        self.pre = pre  # metabolite transformation
        self.cmp = cmp  # compressed stoich
        self.post = post  # reaction transformation
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

    def __init__(self, stoich: RationalMatrix, meta_names: List[str], reac_names: List[str],
                 bounds: Optional[List[Tuple[float, float]]] = None):
        rows, cols = stoich.get_row_count(), stoich.get_column_count()
        self.pre = RationalMatrix.identity(rows)
        self.cmp = stoich.clone()
        self.post = RationalMatrix.identity(cols)
        self.meta_names = list(meta_names)
        self.reac_names = list(reac_names)
        self.bounds = list(bounds) if bounds else [(-float('inf'), float('inf'))] * cols
        self.size = _Size(rows, cols)
        # Store original dimensions for get_truncated()
        self._orig_metas = rows
        self._orig_reacs = cols
        self.stats = CompressionStatistics()
        self.stats.inc_compression_iteration()

    def remove_reactions(self, suppressed: Set[str]) -> bool:
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

        # Reindex names and bounds
        new_names = [self.reac_names[i] for i in keep_indices]
        new_bounds = [self.bounds[i] for i in keep_indices]
        for i, name in enumerate(new_names):
            self.reac_names[i] = name
        self.bounds = new_bounds

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

        meta_names_trunc = self.meta_names[:mc]
        return CompressionRecord(pre_trunc, cmp_trunc, post_trunc, meta_names_trunc, self.stats)


# =============================================================================
# Core Algorithm
# =============================================================================


class StoichMatrixCompressor:
    """Nullspace-based metabolic network compression."""

    def __init__(self, *methods: CompressionMethod):
        self._methods = list(methods) if methods else CompressionMethod.standard()

    def compress(self,
                 stoich: RationalMatrix,
                 meta_names: List[str],
                 reac_names: List[str],
                 suppressed: Set[str] = set(),
                 bounds: Optional[List[Tuple[float, float]]] = None,
                 protected: Set[str] = set()) -> CompressionRecord:
        """Compress network, return transformation matrices.

        Single-pass approach: each iteration computes the nullspace once,
        then removes zero-flux reactions AND merges coupled groups from
        the same kernel.  Re-iterates only when contradicting groups were
        removed (which may expose new couplings).

        `protected` is an optional set of reaction names that must NOT be merged
        into coupled groups (kept intact). The rest of their coupled group still
        merges. Unlike `suppressed`, protected reactions are NOT removed.
        """
        work = _WorkRecord(stoich, meta_names, reac_names, bounds)
        work.protected_names = set(protected)

        do_nullspace = CompressionMethod.NULLSPACE in self._methods
        do_recursive = CompressionMethod.RECURSIVE in self._methods

        work.remove_reactions(suppressed)

        if do_nullspace:
            while True:
                work.stats.inc_compression_iteration()
                work.remove_unused_metabolites()
                found = self._nullspace_compress(work)
                if not (found and do_recursive):
                    break

        work.remove_unused_metabolites()
        work.stats.write_to_log()
        return work.get_truncated()

    def _nullspace_compress(self, work: _WorkRecord) -> bool:
        """One pass of nullspace compression. Returns True if contradicting
        groups were removed (triggering re-iteration).

        Computes the nullspace once, then from the same kernel:
        - Identifies zero-flux reactions (absent from kernel)
        - Identifies and merges coupled reaction groups (proportional kernel rows)
        - Removes zero-flux + coupled slaves + contradicting groups in one batch
        """
        # Build active submatrix for nullspace computation
        active = work.cmp.submatrix(work.size.metas, work.size.reacs)
        kernel = nullspace(active)
        LOG.debug(
            f"Nullspace: {active.get_row_count()}x{active.get_column_count()} -> kernel {kernel.get_row_count()}x{kernel.get_column_count()}"
        )

        # Get kernel pattern (CSR for structure) and values (Fractions for exact ratios)
        kernel_pattern, kernel_values = kernel.to_sparse_pattern()

        # Find zero-flux reactions and merge coupled groups in one pass
        return self._handle_compress(work, kernel_pattern, kernel_values)

    def _find_zero_flux(self, work: _WorkRecord, kernel_sparse) -> set:
        """Find reactions with all-zero kernel rows (indices in current work)."""
        zero_flux = set()
        for reac in range(work.size.reacs):
            if kernel_sparse.indptr[reac] == kernel_sparse.indptr[reac + 1]:
                zero_flux.add(reac)
                work.stats.inc_zero_flux_reactions()
        return zero_flux

    def _find_coupled_groups(self, kernel_pattern, kernel_values, num_reacs, protected_indices=set()):
        """Find groups of coupled reactions from kernel sparsity pattern.

        Args:
            kernel_pattern: CSR matrix with 1s at non-zero positions (for indptr/indices)
            kernel_values: {row: {col: Fraction}} for actual values
            protected_indices: optional set of reaction indices that must NOT be merged
                into any coupled group (kept as their own reactions). The rest of a
                coupled group is still merged. Used e.g. to keep gene-controlled
                reactions intact through compression before GPR extension so that the
                gene multiplicity is preserved (regulatory/knock-in correctness).

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
                if ratios[reac_a] is not None or reac_a in protected_indices:
                    continue
                group = None

                # Get values for reaction a
                a_row = kernel_values.get(reac_a, {})

                for j in range(i + 1, len(potential)):
                    reac_b = potential[j]
                    if ratios[reac_b] is not None or reac_b in protected_indices:
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

    def _handle_compress(self, work: _WorkRecord, kernel_pattern, kernel_values) -> bool:
        """Remove zero-flux and merge coupled reactions from a single kernel.

        From one nullspace computation:
        1. Identifies zero-flux reactions (empty kernel rows)
        2. Finds and merges coupled groups (proportional kernel rows)
        3. Detects contradicting groups via bounds intersection
        4. Removes everything in one batch

        Returns True if contradicting groups were removed (flux space
        changed), triggering re-iteration to find new couplings.
        """
        # Collect zero-flux reactions (empty kernel rows)
        zero_flux = self._find_zero_flux(work, kernel_pattern)

        # Also catch reactions blocked by bounds (lb=ub=0) that still have
        # non-zero kernel rows.  These are structurally present but carry
        # no flux.  Removing them here avoids needing a separate FVA pass.
        for reac in range(work.size.reacs):
            if reac not in zero_flux:
                lb, ub = work.bounds[reac]
                if lb == 0 and ub == 0:
                    zero_flux.add(reac)
                    work.stats.inc_zero_flux_reactions()

        # Find and merge coupled groups (skipping protected reactions, which must
        # stay intact through compression — mapped from names to current indices)
        prot_names = getattr(work, 'protected_names', set())
        protected_idx = {i for i, n in enumerate(work.reac_names) if n in prot_names} if prot_names else set()
        groups, ratios = self._find_coupled_groups(kernel_pattern, kernel_values, work.size.reacs, protected_idx)

        reactions_to_remove = set(zero_flux)
        contradicting_removed = False

        # Enter batch edit mode for cmp and post matrices to avoid repeated LIL conversions
        work.cmp.begin_batch_edit()
        work.post.begin_batch_edit()

        for group in groups:
            # Count nonzeros per member here; used later to pin the lump's scale to the member
            # with the most coefficients.
            nnz = {r: sum(1 for _ in work.cmp.iter_column_fractions(r)) for r in group}
            self._combine_coupled(work, group, ratios)

            # Check bounds intersection to detect contradicting groups.
            # v_master = ratios[slave] * v_slave  =>  v_slave = v_master / ratios[slave]
            # Slave constraint: lb_s <= v_slave <= ub_s
            # Translates to bounds on v_master depending on sign of ratios[slave].
            master = group[0]
            lb_m, ub_m = work.bounds[master]
            intersected_lb = lb_m
            intersected_ub = ub_m

            for slave in group[1:]:
                ratio = ratios[slave]  # v_master / v_slave
                lb_s, ub_s = work.bounds[slave]

                if ratio > 0:
                    # lb_s * ratio <= v_master <= ub_s * ratio
                    s_lb = -float('inf') if lb_s == -float('inf') else lb_s * float(ratio)
                    s_ub = float('inf') if ub_s == float('inf') else ub_s * float(ratio)
                else:  # ratio < 0
                    # ub_s * ratio <= v_master <= lb_s * ratio
                    s_lb = -float('inf') if ub_s == float('inf') else ub_s * float(ratio)
                    s_ub = float('inf') if lb_s == -float('inf') else lb_s * float(ratio)

                intersected_lb = max(intersected_lb, s_lb)
                intersected_ub = min(intersected_ub, s_ub)

            # Update master bounds to intersection
            work.bounds[master] = (intersected_lb, intersected_ub)

            if intersected_lb > intersected_ub or (intersected_lb == 0 and intersected_ub == 0):
                # Contradicting: only zero flux feasible, or infeasible.
                # Remove master and all slaves.
                LOG.debug(f"Contradicting coupled group: {[work.reac_names[r] for r in group]}")
                for idx in group:
                    reactions_to_remove.add(idx)
                contradicting_removed = True
            else:
                # Consistent: only remove slaves (merged into master)
                for idx in group[1:]:
                    reactions_to_remove.add(idx)
                # Pick the member whose units the lump keeps. nnz was counted pre-merge above;
                # co-locate the small-bound test with it here so the whole decision reads in one
                # place. Prefer a reaction with a small finite bound (e.g. Biomass, ATP
                # maintenance) whose own ratio is near 1; else the member with the most
                # coefficients. Master bounds are already intersected at this point.
                def _small_bound(r):
                    fin = [abs(x) for x in work.bounds[r] if not isinf(x) and x != 0 and abs(x) < 100]
                    return min(fin) if fin else None

                def _lam(r):
                    return 1.0 if ratios[r] is None else float(abs(ratios[r]))   # master's own ratio is 1

                bounded = [(b, r) for r in group for b in [_small_bound(r)]
                           if b is not None and 0.1 <= _lam(r) <= 10]
                keep = min(bounded)[1] if bounded else max(group, key=lambda r: nnz[r])
                self._restore_group_scale(work, group, ratios, keep)

        # End batch edit mode
        work.cmp.end_batch_edit()
        work.post.end_batch_edit()

        LOG.debug(f"Compression: {len(zero_flux)} zero-flux, {len(groups)} coupled groups, "
                  f"removing {len(reactions_to_remove)} reactions, contradicting={contradicting_removed}")
        work.remove_reactions_by_indices(reactions_to_remove)

        return contradicting_removed

    def _restore_group_scale(self, work: _WorkRecord, group: List[int],
                             ratios: List[Optional[Fraction]], keep: int) -> None:
        """Express a merged group in the units of one of its members.

        A lump's ratios are fixed but its overall scale is free, and merging into ``group[0]`` can
        yield an extreme scale (the iML1515 biomass lump comes out 4484x, pushing ``biomass >= 0.001``
        below LP feasibility tolerance). ``keep`` names the member whose units to re-express in
        (chosen by the caller from the nnz / small-bound criteria); ``cmp``, ``post`` and the bounds
        are scaled together, so the change of units is exact.
        """
        master = group[0]
        if keep == master:
            return
        lam = abs(ratios[keep])                     # |.| so the reaction keeps its orientation
        if lam == 0 or lam == 1:
            return
        # v_master = ratios[keep] * v_keep, so re-expressing the lump in v_keep multiplies the
        # column by that ratio and divides the bounds by it.
        work.cmp.scale_column(master, lam.numerator, lam.denominator)
        work.post.scale_column(master, lam.numerator, lam.denominator)
        lb, ub = work.bounds[master]
        f = float(lam)
        work.bounds[master] = (lb if isinf(lb) else lb / f, ub if isinf(ub) else ub / f)

    def _combine_coupled(self, work: _WorkRecord, group: List[int], ratios: List[Optional[Fraction]]) -> None:
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

    def __init__(self, compressed_model, compression_converter, pre_matrix, post_matrix, reaction_map, metabolite_map, statistics,
                 methods_used, original_reaction_names, original_metabolite_names, flipped_reactions):
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
                 metabolite_map: Dict[str, Dict[str, Union[float, Fraction]]], flipped_reactions: List[str]):
        self.reaction_map = reaction_map
        self.metabolite_map = metabolite_map
        self.flipped_reactions = set(flipped_reactions)

    def expand_expression(self, expression: Dict[str, float], remove_missing: bool = False) -> Dict[str, float]:
        """Transform expression from compressed back to original space."""
        expanded = {}
        for comp_rxn, comp_coeff in expression.items():
            if comp_rxn in self.reaction_map:
                for orig_rxn, scale in self.reaction_map[comp_rxn].items():
                    coeff = comp_coeff * scale
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
                frac = float_to_fraction(float(coeff))
            row_idx.append(j)  # reaction → row (transposed layout)
            col_idx.append(i)  # metabolite → column
            num_data.append(frac.numerator)
            den_data.append(frac.denominator)

    from scipy.sparse import csr_matrix as _csr
    num_sparse = _csr((num_data, (row_idx, col_idx)), shape=(num_rxns, num_mets), dtype=np.int64)
    den_sparse = _csr((den_data, (row_idx, col_idx)), shape=(num_rxns, num_mets), dtype=np.int64)
    rm = RationalMatrix._from_sparse(num_sparse, den_sparse)
    basic_mets = basic_columns(rm)

    dependent_mets = [model.metabolites[i] for i in set(range(num_mets)) - set(basic_mets)]
    if dependent_mets:
        model.remove_metabolites(dependent_mets)


def compress_cobra_model(model,
                         methods: Optional[List[Union[str, CompressionMethod]]] = None,
                         in_place: bool = True,
                         suppressed_reactions: Set[str] = set(),
                         protected_reactions: Set[str] = set()) -> CompressionResult:
    """
    Compress a COBRA model using nullspace-based coupling detection.

    Note: Model should be preprocessed first (rational coefficients,
    conservation relations removed). Use networktools.compress_model()
    for automatic preprocessing.

    Args:
        model: COBRA model to compress (should be preprocessed)
        methods: Compression methods. Default: CompressionMethod.standard()
        in_place: Modify original model (True) or work on copy (False)
        suppressed_reactions: Reaction names removed from the network before
            compression (destructive; the reactions are deleted from the
            stoichiometric matrix). Note this changes the nullspace, so it is
            not the right tool for merely keeping reactions intact.
        protected_reactions: Reaction names kept in the network but exempted
            from being merged into a coupled (lumped) group. Non-destructive:
            the reactions stay, only their lumping is prevented.

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

    # Build stoichiometric matrix with exact arithmetic
    stoich_matrix = RationalMatrix.from_cobra_model(model)
    metabolite_names = [m.id for m in model.metabolites]
    reaction_names = [r.id for r in model.reactions]

    # Run compression
    compressor = StoichMatrixCompressor(*compression_methods)
    bounds = [(float(r.lower_bound), float(r.upper_bound)) for r in model.reactions]
    compression_record = compressor.compress(stoich_matrix, metabolite_names, reaction_names, suppressed_reactions, bounds, protected_reactions)

    # Apply to model (uses direct manipulation, bypasses solver)
    reaction_map = _apply_compression_to_model(model, compression_record, reaction_names)

    pre_matrix = compression_record.pre.to_numpy()
    post_matrix = compression_record.post.to_numpy()

    converter = CompressionConverter(reaction_map, {}, [])

    return CompressionResult(compressed_model=model,
                             compression_converter=converter,
                             pre_matrix=pre_matrix,
                             post_matrix=post_matrix,
                             reaction_map=reaction_map,
                             metabolite_map={},
                             statistics=compression_record.stats,
                             methods_used=compression_methods,
                             original_reaction_names=original_reaction_names,
                             original_metabolite_names=original_metabolite_names,
                             flipped_reactions=[])


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

    for j in range(num_compressed):
        # Find contributing original reactions from POST matrix (sparse iteration)
        contributing = list(post.iter_column_fractions(j))

        if not contributing:
            continue

        # Select "main" reaction: most metabolites (nonzeros), ties broken alphabetically
        main_idx = min(
            (idx for idx, _ in contributing),
            key=lambda i: (-len(model.reactions[i]._metabolites), model.reactions[i].id),
        )
        main_rxn = model.reactions[main_idx]
        keep_rxns[main_idx] = True

        if len(contributing) == 1:
            # Standalone reaction — not merged with anything.
            reaction_map[main_rxn.id] = {original_reaction_names[main_idx]: Fraction(1)}
            continue

        # --- Merged group (2+ contributing reactions) ---

        # Store subset info
        main_rxn.subset_rxns = [idx for idx, _ in contributing]
        main_rxn.subset_stoich = [coeff for _, coeff in contributing]

        # Build combined reaction name from contributing reactions
        for idx, _ in contributing:
            if idx == main_idx:
                continue
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
        main_rxn._lower_bound = max(lb_candidates) if lb_candidates else -float('inf')
        main_rxn._upper_bound = min(ub_candidates) if ub_candidates else float('inf')

        # Update stored objective through compression factors
        obj_dict = getattr(model, '_suppressed_obj', None)
        if obj_dict is not None:
            merged_obj = sum(
                obj_dict.pop(original_reaction_names[idx], 0.0) * float(coeff)
                for idx, coeff in contributing
            )
            if merged_obj != 0:
                obj_dict[main_rxn.id] = merged_obj

        # Build reaction map
        reaction_map[main_rxn.id] = {original_reaction_names[idx]: coeff for idx, coeff in contributing}

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

    return reaction_map


# =============================================================================
# Preprocessing Functions
# =============================================================================


def remove_blocked_reactions(model) -> List:
    """Remove blocked reactions (bounds == (0, 0)) from a network."""
    blocked_reactions = [reac for reac in model.reactions if reac.bounds == (0, 0)]
    if blocked_reactions:
        model.remove_reactions(blocked_reactions, remove_orphans=True)
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


def stoichmat_coeff_to_fraction(model) -> None:
    """Convert stoichiometric coefficients to exact fractions.Fraction."""
    for rxn in model.reactions:
        for met, coeff in rxn._metabolites.items():
            if isinstance(coeff, Fraction):
                continue                                        # already exact
            elif isinstance(coeff, (float, int)):
                rxn._metabolites[met] = float_to_fraction(coeff)          # -> Fraction
            elif hasattr(coeff, 'p'):                           # sympy.Rational -> Fraction
                rxn._metabolites[met] = Fraction(int(coeff.p), int(coeff.q))
            elif hasattr(coeff, 'numerator'):                   # other Rational -> Fraction
                rxn._metabolites[met] = Fraction(coeff.numerator, coeff.denominator)
            else:
                raise TypeError(f"Unsupported coefficient type: {type(coeff)}")


def stoichmat_coeff2float(model) -> None:
    """Convert stoichiometric coefficients to floats."""
    for rxn in model.reactions:
        for met, coeff in rxn._metabolites.items():
            rxn._metabolites[met] = float(coeff)


# =============================================================================
# GPR Propagation Helpers
# =============================================================================


def _gpr_ast_to_expr(node, op=None):
    """Convert a cobra GPR AST node to a nested expression, or join expressions under ``op``.

    A gene is its name (str) and a boolean node is ``(op, (children...))``; None means no gene
    requirement (always active). Passing ``op`` joins the given expressions instead of
    converting a node, applying only associativity (same-op children are flattened) and
    idempotence (duplicates dropped) -- real simplification is done by simplify_model_gprs, which
    compress_model runs at the end of a propagate_gpr pass.
    """
    if op is None:
        if isinstance(node, ast.BoolOp):
            op = 'and' if isinstance(node.op, ast.And) else 'or'
            node = [_gpr_ast_to_expr(v) for v in node.values]
        elif isinstance(node, ast.Name):
            return node.id
        else:
            return None
    flat = []
    for e in node:
        if e is None:
            continue
        flat.extend(e[1] if isinstance(e, tuple) and e[0] == op else [e])
    uniq = list(dict.fromkeys(flat))
    return (op, tuple(uniq)) if len(uniq) > 1 else (uniq[0] if uniq else None)


def _expr_to_gpr_string(expr):
    """Render an expression as a GPR rule string, '' for None.

    ``expr`` is one of the nested-expression forms produced by ``_gpr_ast_to_expr``: ``None`` (no
    gene requirement, renders to ''), a gene-id string (e.g. ``'g1'``), or an ``(op, args)`` tuple
    such as ``('and', ('g1', 'g2'))`` -> ``'g1 and g2'`` or, more nested,
    ``('and', ('g1', ('or', ('g2', 'g3'))))``.

    Operands are sorted so equivalent inputs give identical rules, and a nested clause of the
    opposite operator is parenthesised.
    """
    if expr is None:
        return ''
    if isinstance(expr, str):
        return expr
    op, args = expr
    other = 'or' if op == 'and' else 'and'
    parts = [f'({s})' if isinstance(a, tuple) and a[0] == other else s
             for a, s in sorted(((a, _expr_to_gpr_string(a)) for a in args), key=lambda p: p[1])]
    return f' {op} '.join(parts)


def _combine_gprs(gpr_bodies, op):
    """Combine GPR AST bodies (reaction.gpr.body, may include None) under ``op``, as a rule string.

    An empty GPR is True: under 'and' it is dropped, under 'or' it makes the whole rule
    unrestricted (''). Used to merge the GPRs of reactions that compression lumps together --
    'and' for coupled/serial merges, 'or' for parallel ones.
    """
    exprs = [_gpr_ast_to_expr(b) for b in gpr_bodies]
    if op == 'or' and (not exprs or any(e is None for e in exprs)):
        return ''
    return _expr_to_gpr_string(_gpr_ast_to_expr(exprs, op))


# =============================================================================
# High-Level Compression API
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# Monotone (positive-unate) GPR-rule simplification
#
# Pipeline:  parse -> minimal SOP (DNF + absorption) -> algebraic factoring.
# Cubes are int bitmasks (bit i == variable i): subset = (a & b) == a, union = a | b.
# Output is inverter-free by construction and boolean-EQUIVALENT to the input, so replacing a
# reaction's GPR with its factored form leaves flux/knockout semantics -- and strain designs --
# unchanged, while shrinking the GPR gadget built by extend_model_gpr. `factor_auto(node, budget)`
# guards the only source of DNF blow-up (an AND of large ORs) by AND-splitting over-budget
# conjuncts (exact, near-optimal since complexes sit on ~disjoint genes). `simplify_model_gprs(model)`
# is the entry point; compress_model calls it when propagate_gpr is set so standalone compression
# emits already-simplified rules.
# ─────────────────────────────────────────────────────────────────────────────

# popcount: C-level int.bit_count() on Python 3.10+, else the bin().count fallback
_popcount = getattr(int, 'bit_count', None) or (lambda c: bin(c).count('1'))


def _gpr_tokenize(s):
    for m in re.finditer(r'\(|\)|\*|\+|[^\s()*+]+', s):
        yield m.group()


def _gpr_parse(s):
    """Parse a GPR string into an AST.

    Accepts both ``and``/``or`` and ``*``/``+`` operators, and is robust to any gene id,
    including digit-leading or dotted names.
    """
    toks = list(_gpr_tokenize(s)); pos = 0
    def peek(): return toks[pos] if pos < len(toks) else None
    def eat():
        nonlocal pos; t = toks[pos]; pos += 1; return t
    def p_or():
        n = [p_and()]
        while peek() in ('or', '+'): eat(); n.append(p_and())
        return ('OR', n) if len(n) > 1 else n[0]
    def p_and():
        n = [p_atom()]
        while peek() in ('and', '*'): eat(); n.append(p_atom())
        return ('AND', n) if len(n) > 1 else n[0]
    def p_atom():
        if peek() == '(': eat(); e = p_or(); eat(); return e
        return ('VAR', eat())
    return p_or()


# ---- variable <-> bit mapping (reset per rule via simplify_gpr_string) ----
_GPR_VMAP = {}; _GPR_VINV = []
def _gpr_bit(v):
    i = _GPR_VMAP.get(v)
    if i is None:
        i = len(_GPR_VINV); _GPR_VMAP[v] = i; _GPR_VINV.append(v)
    return 1 << i
def _gpr_lits_of(mask):
    out = []
    while mask:
        l = mask & -mask; out.append(('VAR', _GPR_VINV[l.bit_length() - 1])); mask ^= l
    return out


# ---- cover algebra (cubes = ints) ----
def _gpr_absorb(cubes):
    uniq = set(cubes)
    buckets = {}
    for c in uniq:
        buckets.setdefault(_popcount(c), []).append(c)
    keep = []
    for pc in sorted(buckets):
        smaller = keep[:]
        for c in buckets[pc]:
            if not any((k & c) == k for k in smaller):
                keep.append(c)
    return keep


def _gpr_to_dnf(node):
    t = node[0]
    if t == 'VAR':   return [_gpr_bit(node[1])]
    if t == 'CONST': return [] if not node[1] else [0]
    if t == 'OR':
        cov = []
        for ch in node[1]: cov += _gpr_to_dnf(ch)
        return _gpr_absorb(cov)
    if t == 'AND':
        cov = [0]
        for ch in node[1]:
            sub = _gpr_to_dnf(ch)
            cov = _gpr_absorb([a | b for a in cov for b in sub])
        return cov
    raise ValueError(t)


def _gpr_common(cubes):
    it = iter(cubes); c = next(it)
    for x in it: c &= x
    return c


def _gpr_lit_counts(F):
    cnt = {}
    for c in F:
        m = c
        while m:
            l = m & -m; cnt[l] = cnt.get(l, 0) + 1; m ^= l
    return cnt


def _gpr_one_kernel(F, l):
    Q = [c & ~l for c in F if c & l]
    cc = _gpr_common(Q)
    if cc: Q = [c & ~cc for c in Q]
    Q = _gpr_absorb(Q)
    cnt = _gpr_lit_counts(Q)
    reps = [x for x, n in cnt.items() if n >= 2]
    if not reps: return Q
    return _gpr_one_kernel(Q, max(reps, key=lambda x: cnt[x]))


def _gpr_candidate_divisors(F):
    F = _gpr_absorb(F)
    if len(F) < 2: return []
    cnt = _gpr_lit_counts(F)
    reps = sorted((x for x, n in cnt.items() if n >= 2), key=lambda x: -cnt[x])
    seen = set(); out = []
    for l in reps:
        K = tuple(sorted(_gpr_one_kernel(F, l)))
        if len(K) >= 2 and K not in seen:
            seen.add(K); out.append(list(K))
    return out


def _gpr_divide(F, D):
    """Exact algebraic division: (Q, R) with D*Q disjoint-union R == F (correctness guaranteed
    regardless of divisor quality -- a quotient cube is accepted only if D*Q stays inside F)."""
    Fs = set(F); quo = None
    for d in D:
        vd = {c & ~d for c in F if (c & d) == d}
        quo = vd if quo is None else (quo & vd)
        if not quo: return [], list(F)
    Q = list(quo)
    DQ = {dc | qc for dc in D for qc in Q}
    if not DQ <= Fs: return [], list(F)
    return Q, list(Fs - DQ)


def _gpr_factor(F):
    F = _gpr_absorb(F)
    if not F:     return ('CONST', False)
    if F == [0]:  return ('CONST', True)
    if len(F) == 1:
        lits = _gpr_lits_of(F[0])
        return lits[0] if len(lits) == 1 else ('AND', lits)
    cc = _gpr_common(F)
    if cc:
        rem = [c & ~cc for c in F]
        return ('AND', _gpr_lits_of(cc) + [_gpr_factor(rem)])
    best = None
    for D in _gpr_candidate_divisors(F):
        Q, R = _gpr_divide(F, D)
        if not Q or len(D) >= len(F) or len(Q) >= len(F):
            continue
        clean = 1 if not R else 0
        pulled = sum(_popcount(c) for c in D)
        cand = (clean, pulled, D, Q, R)
        if best is None or cand[:2] > best[:2]:
            best = cand
    if best is None:
        return ('OR', [_gpr_factor([c]) for c in F])
    _, _, D, Q, R = best
    dq = ('AND', [_gpr_factor(D), _gpr_factor(Q)])
    return dq if not R else ('OR', [dq, _gpr_factor(R)])


def _gpr_est_cubes(node):
    """Upper bound on DNF cube count (product across ANDs, sum across ORs); cheap, no expansion."""
    t = node[0]
    if t == 'VAR':   return 1
    if t == 'CONST': return 1
    if t == 'OR':    return sum(_gpr_est_cubes(c) for c in node[1])
    if t == 'AND':
        p = 1
        for c in node[1]:
            p *= _gpr_est_cubes(c)
            if p > 1 << 62: return p
        return p


_GPR_WARN = []
def _gpr_factor_auto(node, budget=50000):
    """Global factoring within budget; AND-split above it. Never splits an OR unless one single
    OR-block alone exceeds budget (logged as a last resort -- raise the budget to avoid)."""
    if node[0] == 'VAR':
        return node
    if _gpr_est_cubes(node) <= budget:
        return _gpr_factor(_gpr_to_dnf(node))
    if node[0] == 'AND':
        return ('AND', [_gpr_factor_auto(c, budget) for c in node[1]])
    _GPR_WARN.append("OR-block of ~%d cubes exceeds budget %d; split anyway." % (_gpr_est_cubes(node), budget))
    return ('OR', [_gpr_factor_auto(c, budget) for c in node[1]])


def _gpr_to_string(n):
    if n[0] == 'VAR':
        return n[1]
    if n[0] == 'CONST':
        return ''  # tautology -> no gene requirement
    if n[0] == 'AND':
        return ' and '.join(('(%s)' % _gpr_to_string(c)) if c[0] == 'OR' else _gpr_to_string(c) for c in n[1])
    return ' or '.join(('(%s)' % _gpr_to_string(c)) if c[0] == 'AND' else _gpr_to_string(c) for c in n[1])


def simplify_gpr_string(rule, budget=50000):
    """Return a leaf-minimized, boolean-equivalent monotone GPR string ('' passes through)."""
    if not rule or not rule.strip():
        return rule
    _GPR_VMAP.clear(); _GPR_VINV.clear(); _GPR_WARN.clear()
    return _gpr_to_string(_gpr_factor_auto(_gpr_parse(rule), budget))


def simplify_model_gprs(model, budget=50000):
    """In place: replace each reaction's gene_reaction_rule with a leaf-minimized equivalent.

    Monotone AND/OR boolean-equivalence => flux/knockout semantics (and strain designs) unchanged;
    only the GPR gadget built by extend_model_gpr shrinks. Any per-rule failure keeps the original.
    """
    n = nchg = 0
    for r in model.reactions:
        s = r.gene_reaction_rule
        if not s:
            continue
        n += 1
        try:
            new = simplify_gpr_string(s, budget)
            if new and new != s:
                r.gene_reaction_rule = new; nchg += 1
        except Exception as e:
            logging.warning('gpr_simplify: kept original GPR for %s (%s)' % (r.id, type(e).__name__))
    logging.info('  GPR rule simplification: %d rules, %d rewritten.' % (n, nchg))


def compress_model(model, no_par_compress_reacs=set(), compression_backend='sparse_rref', propagate_gpr=False,
                   no_coupled_compress_reacs=set()):
    """Compress a metabolic model using multiple techniques.

    Performs blocked reaction removal, conservation relation removal, and
    alternating dependent/parallel reaction lumping until no further
    compression is possible.

    Args:
        model: COBRA model to compress in-place
        no_par_compress_reacs: Reactions exempt from parallel compression
        no_coupled_compress_reacs: Reactions exempt from coupled compression. The rest of
            their coupled group still merges. Used to keep gene-controlled reactions
            un-merged through COMPRESS#1 so that gene multiplicity is preserved exactly once
            GPR rules are integrated (correct gene-regulatory semantics under compression).
            To also exempt them from parallel merging, include them in no_par_compress_reacs.
        compression_backend: Compression backend to use:
            - 'sparse_rref' (default): Pure Python sparse integer RREF.
              No external dependencies beyond NumPy/SciPy.
            - 'efmtool_rref' (legacy): Java-based EFMTool via JPype.
              Requires a JVM and the jpype1 package.
        propagate_gpr: If True, propagate and simplify GPR rules through
            compression (AND for coupled, OR for parallel merges).
            Empty GPR rules are correctly handled: skipped in AND (always
            active), and absorb in OR (result is always active).
            Uses sympy for boolean simplification. Default False.

    Returns:
        list of dict: Compression maps for reversing each compression step
    """
    from straindesign.networktools import suppress_lp_context, _is_lp_suppressed
    no_coupled_compress_reacs = set(no_coupled_compress_reacs)
    with suppress_lp_context(model):
        cmp_mapReac = []
        use_java = (compression_backend == 'efmtool_rref')
        LOG.info('  Removing blocked reactions.')
        remove_blocked_reactions(model)
        LOG.info('  Converting coefficients to rationals.')
        stoichmat_coeff_to_fraction(model)
        coupled_changed = None  # None = not yet computed
        run = 1
        while True:
            numr = len(model.reactions)

            # 1. Parallel (cheap — hash-based, no RREF)
            LOG.info(f'  Compression {run}: Lumping parallel reactions.')
            reac_map_exp = compress_model_parallel(model, no_par_compress_reacs,
                                                    propagate_gpr=propagate_gpr)
            parallel_changed = numr > len(reac_map_exp)
            if parallel_changed:
                LOG.info(f'  Reduced to {len(reac_map_exp)} reactions.')
                cmp_mapReac.append({"reac_map_exp": reac_map_exp, "parallel": True})

            # 2. Conservation relation removal (reduces S rows for RREF)
            if use_java:
                _remove_conservation_relations_java(model)
            else:
                remove_conservation_relations(model)

            # 3. Exit if either parallel or coupled found nothing (after
            #    at least one full cycle).  If one step found nothing,
            #    re-running it won't help — the model hasn't changed in
            #    a way that creates new opportunities for that step.
            if coupled_changed is not None and (not parallel_changed or not coupled_changed):
                LOG.info(f'  Compression complete ({run - 1} cycles).')
                break

            # 4. Coupled (expensive — nullspace/RREF)
            numr_pre = len(model.reactions)
            LOG.info(f'  Compression {run}: Lumping coupled reactions.')
            reac_map_exp = compress_model_coupled(model, compression_backend,
                                                  propagate_gpr=propagate_gpr,
                                                  protected_reactions=no_coupled_compress_reacs)
            for new_reac, old_reac_val in reac_map_exp.items():
                old_reacs = [r for r in no_par_compress_reacs if r in old_reac_val]
                if old_reacs:
                    for r in old_reacs:
                        no_par_compress_reacs.remove(r)
                    no_par_compress_reacs.add(new_reac)
            coupled_changed = numr_pre > len(reac_map_exp)
            if coupled_changed:
                LOG.info(f'  Reduced to {len(reac_map_exp)} reactions.')
                cmp_mapReac.append({"reac_map_exp": reac_map_exp, "parallel": False})

            run += 1

    # suppress_lp_context handles solver rebuild, objective restoration and stale-group pruning
    # on exit
    if propagate_gpr:
        # Leaf-minimize the propagated rules so any caller (incl. standalone compression, not just
        # the SD pipeline) gets simplified GPRs. Monotone/boolean-equivalent -> designs unchanged;
        # only the extend_model_gpr gadget shrinks. In the pipeline this runs again after reduce, but
        # simplification is cheap and idempotent.
        simplify_model_gprs(model)
    return cmp_mapReac


def _remove_conservation_relations_java(model) -> None:
    """Remove conservation relations using Java efmtool."""
    from . import efmtool_cmp_interface as efm
    stoich_mat = create_stoichiometric_matrix(model, array_type='lil')
    basic_mets = efm.basic_columns_rat_java(stoich_mat.transpose().toarray(), tolerance=0)
    dependent = [model.metabolites[i] for i in set(range(len(model.metabolites))) - set(basic_mets)]
    if dependent:
        model.remove_metabolites(dependent)


def compress_model_coupled(model, compression_backend='sparse_rref', propagate_gpr=False,
                           suppressed_reactions=set(), protected_reactions=set()):
    """Compress by lumping stoichiometrically coupled (dependent) reactions.

    Identifies groups of reactions whose flux vectors are proportional in every
    steady state (i.e. they share a common nullspace direction) and merges each
    group into a single lumped reaction.  Both the pure-Python and legacy Java
    backends perform this operation; the compression_backend controls the nullspace algorithm.

    Args:
        model: COBRA model to compress in-place
        compression_backend: 'sparse_rref' (default, Python) or 'efmtool_rref' (Java legacy)
        propagate_gpr: If True, AND-combine GPR rules of merged reactions
            (with sympy simplification). Empty GPRs are skipped. Default False.
        suppressed_reactions: Set of reaction IDs to exclude from compression
            (Java backend only). Used to protect reactions referenced in strain
            design constraints from being deleted by the Java compressor's
            CoupledContradicting logic. Ignored for the Python backend (which
            handles contradicting groups correctly via bounds intersection).
        protected_reactions: Set of reaction IDs to exempt from coupled merging
            (kept as their own reactions; the rest of their coupled group still
            merges). Python (sparse_rref) backend only. Used to keep gene-controlled
            reactions intact through compression before GPR integration so that the
            gene multiplicity is preserved (correct gene-regulatory semantics).

    Returns:
        dict: Mapping {compressed_id: {orig_id: factor, ...}}
    """
    # Compression is pure linear algebra; keep it off the optlang solver.
    from straindesign.networktools import suppress_lp_context
    with suppress_lp_context(model):
        # Save GPR AST bodies before either backend clears them
        if propagate_gpr:
            saved_gpr_bodies = {r.id: r.gpr.body for r in model.reactions}

        if compression_backend == 'efmtool_rref':
            from .efmtool_cmp_interface import compress_model_java
            reaction_map = compress_model_java(model, suppressed_reactions=suppressed_reactions)
            # Clean up any remaining zero-flux reactions that the Java compressor created.
            zero_flux = {r for r in model.reactions if r.lower_bound == 0 and r.upper_bound == 0}
            for r in zero_flux:
                reaction_map.pop(r.id, None)
            if zero_flux:
                model.remove_reactions(list(zero_flux), remove_orphans=True)
        else:
            # Clear gene rules to match Java behavior
            for r in model.reactions:
                r.gene_reaction_rule = ''

            result = compress_cobra_model(model, methods=CompressionMethod.standard(), in_place=True,
                                          protected_reactions=protected_reactions)
            reaction_map = result.reaction_map

        # Propagate GPR rules: AND-combine contributing reactions' GPR ASTs
        if propagate_gpr:
            for cmp_id, orig_map in reaction_map.items():
                try:
                    rxn = model.reactions.get_by_id(cmp_id)
                except KeyError:
                    continue
                gpr_bodies = [saved_gpr_bodies.get(orig_id) for orig_id in orig_map]
                rxn.gene_reaction_rule = _combine_gprs(gpr_bodies, 'and')

    return reaction_map


# Backward-compatibility alias (old name referenced efmtool, but the function
# is backend-agnostic — the new name compress_model_coupled is preferred).
compress_model_efmtool = compress_model_coupled


def compress_model_parallel(model, protected_rxns=set(), propagate_gpr=False):
    """Compress by lumping parallel reactions.

    Args:
        model: COBRA model to compress in-place
        protected_rxns: Reactions exempt from parallel compression
        propagate_gpr: If True, OR-combine GPR rules of lumped reactions
            (with sympy simplification). Default False.

    Returns:
        dict: Mapping {compressed_id: {orig_id: factor, ...}}
    """
    old_num_reac = len(model.reactions)
    old_reac_ids = [r.id for r in model.reactions]

    if propagate_gpr:
        old_gpr_bodies = {r.id: r.gpr.body for r in model.reactions}

    stoichmat_T = create_stoichiometric_matrix(model, 'lil').transpose()
    factor = [d[0] if d else 1.0 for d in stoichmat_T.data]

    lb = [float(r.lower_bound) for r in model.reactions]
    ub = [float(r.upper_bound) for r in model.reactions]
    fwd = [1 if (isinf(u) and f > 0 or isinf(l) and f < 0) else 0 for f, l, u in zip(factor, lb, ub)]
    rev = [1 if (isinf(l) and f > 0 or isinf(u) and f < 0) else 0 for f, l, u in zip(factor, lb, ub)]
    inh = [i + 1 if not ((isinf(ub[i]) or ub[i] == 0) and (isinf(lb[i]) or lb[i] == 0)) else 0
           for i in range(len(model.reactions))]

    # Canonical scale-invariant key per reaction: normalize the stoichiometry row by its first
    # nonzero coefficient in exact rational arithmetic, so reactions parallel up to any rational
    # scale factor share a key (e.g. -1 A -> 2 B and -3 A -> 6 B both give ((A, 1), (B, -2))). The
    # reversibility (fwd/rev) and inhomogeneous-bound (inh) flags are part of the key so only
    # reactions with matching bound structure are lumped.
    def _parallel_key(i):
        cols, vals = stoichmat_T.rows[i], stoichmat_T.data[i]
        if not vals:
            return ((), fwd[i], rev[i], inh[i])
        f0 = float_to_fraction(vals[0])
        stoich = tuple((int(c), float_to_fraction(v) / f0) for c, v in zip(cols, vals))
        return (stoich, fwd[i], rev[i], inh[i])

    # Find parallel reactions by exact key comparison (hash pre-filter, then full compare)
    protected = [r.id in protected_rxns for r in model.reactions]
    keys = [_parallel_key(i) for i in range(len(model.reactions))]

    # Group reactions that share an exact key in a single O(n) pass. dict preserves first-occurrence
    # order, so each group's representative is its smallest index and subset_list stays ordered by
    # ascending representative -- matching the surviving-reaction order after remove_reactions below.
    # Protected reactions get a unique 2-tuple key (real keys are 4-tuples) so they never merge.
    groups = {}
    for i, key in enumerate(keys):
        groups.setdefault(('\0protected', i) if protected[i] else key, []).append(i)
    subset_list = list(groups.values())

    # Lump parallel reactions
    del_rxns = [False] * len(model.reactions)
    obj_dict = getattr(model, '_suppressed_obj', None)
    for rxn_idx in subset_list:
        for i in range(1, len(rxn_idx)):
            main_rxn = model.reactions[rxn_idx[0]]
            other_rxn = model.reactions[rxn_idx[i]]
            if len(main_rxn.id) + len(other_rxn.id) < 220 and not main_rxn.id.endswith('...'):
                main_rxn.id += '*' + other_rxn.id
            elif not main_rxn.id.endswith('...'):
                main_rxn.id += '...'
            del_rxns[rxn_idx[i]] = True
        # Update stored objective (parallel factor = 1.0, sum contributions)
        if len(rxn_idx) > 1 and obj_dict is not None:
            merged_obj = sum(obj_dict.pop(old_reac_ids[j], 0.0) for j in rxn_idx)
            if merged_obj != 0:
                obj_dict[model.reactions[rxn_idx[0]].id] = merged_obj

    # Pre-compute combined GPR for each group (OR logic) before deletions.
    # Save (surviving_rxn_object, combined_gpr_string) pairs.
    if propagate_gpr:
        group_gpr = []
        for rxn_idx_group in subset_list:
            main_rxn = model.reactions[rxn_idx_group[0]]
            gpr_bodies = [old_gpr_bodies.get(old_reac_ids[j]) for j in rxn_idx_group]
            group_gpr.append((main_rxn, _combine_gprs(gpr_bodies, 'or')))

    remove_list = [model.reactions[i] for i in np.where(del_rxns)[0]]
    if remove_list:
        model.remove_reactions(remove_list, remove_orphans=True)

    # Set combined GPR rules on surviving reactions
    if propagate_gpr:
        for rxn, combined_gpr in group_gpr:
            rxn.gene_reaction_rule = combined_gpr

    # Build compression map with flux-split fractions.
    # For parallel reactions, the compressed flux is the total through all
    # members.  Each member's fraction is proportional to |first_coeff|
    # (its stoichiometric scale relative to the representative).
    rational_map = {}
    subT = np.zeros((old_num_reac, len(model.reactions)))
    for i in range(subT.shape[1]):
        group = subset_list[i]
        for j in group:
            subT[j, i] = 1
        if len(group) == 1:
            rational_map[model.reactions[i].id] = {old_reac_ids[group[0]]: Fraction(1)}
        else:
            scales = [abs(factor[j]) for j in group]
            total = sum(Fraction(s).limit_denominator(1000) for s in scales)
            rational_map[model.reactions[i].id] = {
                old_reac_ids[j]: Fraction(abs(factor[j])).limit_denominator(1000) / total
                for j in group
            }

    return rational_map


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Rational matrix and utilities
    'RationalMatrix',
    'float_to_fraction',
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
    'compress_model_coupled',
    'compress_model_efmtool',  # backward-compat alias
    'compress_model_parallel',
    # GPR propagation helpers
    '_gpr_ast_to_expr',
    '_expr_to_gpr_string',
    '_combine_gprs',
    # GPR rule simplification
    'simplify_model_gprs',
    'simplify_gpr_string',
    # Preprocessing
    'remove_blocked_reactions',
    'remove_ext_mets',
    'remove_conservation_relations',
    'remove_dummy_bounds',
    'stoichmat_coeff_to_fraction',
    'stoichmat_coeff2float',
]
