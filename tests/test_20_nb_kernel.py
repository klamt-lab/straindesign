"""Kernel basis variants for the nullspace (NB) dual (``SD_NB_KERNEL``).

Every variant is a change of basis of ``null(A_eq_p)``: the dual's feasible set is defined by the
SUBSPACE, so a variant is only admissible if it keeps the rank and the residual exactly. These
tests pin that, plus the sparsity claim that motivates the variants.
"""
import numpy as np
import pytest
from scipy import sparse

import straindesign as sd
from straindesign.nb_kernel import (_columns_from_kernel, _max_abs, _nnz, kernel_to_float_csr,
                                    transform_columns)

VARIANTS = ['none', 'gcd', 'greedy', 'greedycap', 'norm', 'greedynorm']


def _random_sparse_int_matrix(rng, m=18, n=40):
    A = np.zeros((m, n), dtype=np.int64)
    for j in range(n):
        rows = rng.choice(m, size=int(rng.integers(2, 5)), replace=False)
        A[rows, j] = rng.integers(-4, 5, size=len(rows))
    return A


@pytest.mark.parametrize('seed', [0, 1, 2, 3, 4])
@pytest.mark.parametrize('variant', VARIANTS)
def test_variant_preserves_the_subspace(seed, variant):
    rng = np.random.default_rng(seed)
    A = _random_sparse_int_matrix(rng)
    columns, n_rows = _columns_from_kernel(sd.sparse_nullspace(sparse.csr_matrix(A)))
    out = transform_columns([dict(c) for c in columns], n_rows, variant)
    K = kernel_to_float_csr(out, n_rows).toarray()
    assert len(out) == len(columns)
    assert np.linalg.matrix_rank(K) == len(columns)
    assert np.abs(A @ K).max() == 0.0


@pytest.mark.parametrize('seed', [0, 1, 2, 3, 4])
def test_elimination_variants_are_sparser(seed):
    rng = np.random.default_rng(seed)
    A = _random_sparse_int_matrix(rng)
    columns, n_rows = _columns_from_kernel(sd.sparse_nullspace(sparse.csr_matrix(A)))
    base = _nnz(columns)
    assert _nnz(transform_columns([dict(c) for c in columns], n_rows, 'gcd')) == base
    for variant in ('greedy', 'norm'):
        assert _nnz(transform_columns([dict(c) for c in columns], n_rows, variant)) < base


def test_gcd_only_shrinks_coefficients():
    rng = np.random.default_rng(7)
    A = _random_sparse_int_matrix(rng)
    columns, n_rows = _columns_from_kernel(sd.sparse_nullspace(sparse.csr_matrix(A)))
    out = transform_columns([dict(c) for c in columns], n_rows, 'gcd')
    assert _max_abs(out) <= _max_abs(columns)


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError):
        transform_columns([{0: 1}], 1, 'no_such_variant')


def test_exact_fraction_keeps_small_coefficients():
    from fractions import Fraction
    from straindesign.strainDesignProblem import _exact_fraction
    for v in (1.234567e-7, 3.3e-6, 0.1, 1.0 / 3.0, -2.5e-9, 7.0):
        f = _exact_fraction(v)
        assert abs(float(f) - v) <= 8.0 * np.spacing(abs(v))
    assert _exact_fraction(1.0 / 3.0) == Fraction(1, 3)


@pytest.mark.parametrize('kernel', ['none', 'markowitz', 'greedy'])
def test_kernel_of_small_coefficient_matrix_is_exact(kernel, monkeypatch):
    from straindesign.strainDesignProblem import _exact_fraction, _markowitz_kernel, _nullspace_float
    monkeypatch.setenv('SD_NB_KERNEL', kernel)
    rng = np.random.default_rng(11)
    A = _random_sparse_int_matrix(rng).astype(float)
    A[0, np.nonzero(A[0])[0][0]] = 1.234567e-7
    K = _markowitz_kernel(A) if kernel == 'markowitz' else _nullspace_float(A)
    assert K.shape[1] == A.shape[1] - np.linalg.matrix_rank(A)
    Kd = K.toarray()
    Af = np.vectorize(_exact_fraction, otypes=[object])(A)
    for j in range(Kd.shape[1]):
        col = np.array([_exact_fraction(x) for x in Kd[:, j]], dtype=object)
        res = Af.dot(col)
        scale = max(abs(float(x)) for x in col)
        assert max(abs(float(r)) for r in res) <= 1e-12 * scale
