"""
FLINT linear algebra functions for exact rational arithmetic.

This module provides FLINT-based implementations for nullspace and
basic column computation. It requires python-flint to be installed.
"""

from typing import List

from flint import fmpq, fmpq_mat

# Import RationalMatrix from compression module
from .compression import RationalMatrix

FLINT_AVAILABLE = True


# =============================================================================
# Conversion Functions
# =============================================================================

def rational_matrix_to_fmpq_mat(matrix: RationalMatrix) -> fmpq_mat:
    """Convert RationalMatrix to FLINT fmpq_mat for linear algebra operations."""
    rows = matrix.get_row_count()
    cols = matrix.get_column_count()
    mat = fmpq_mat(rows, cols)

    coo_num = matrix._num_sparse.tocoo()
    coo_den = matrix._den_sparse.tocoo()

    for r, c, num, den in zip(coo_num.row, coo_num.col, coo_num.data, coo_den.data):
        if num != 0:
            mat[int(r), int(c)] = fmpq(int(num), int(den))
    return mat


# =============================================================================
# Linear Algebra Functions
# =============================================================================

def nullspace_flint(matrix: RationalMatrix) -> RationalMatrix:
    """Compute nullspace using FLINT's RREF."""
    rows = matrix.get_row_count()
    cols = matrix.get_column_count()

    mat = rational_matrix_to_fmpq_mat(matrix)
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


def basic_columns_flint(matrix: RationalMatrix) -> List[int]:
    """Find pivot columns using FLINT's RREF."""
    rows = matrix.get_row_count()
    cols = matrix.get_column_count()

    mat = rational_matrix_to_fmpq_mat(matrix)
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
    'rational_matrix_to_fmpq_mat',
    'nullspace_flint',
    'basic_columns_flint',
]
