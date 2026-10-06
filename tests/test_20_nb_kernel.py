"""Exact kernel input for the nullspace (NB) dual (``milp_nullspace``).

The NB dual constrains its certificate against the subspace null(A_eq_p), so the kernel must keep
the rank and annihilate A_eq_p exactly, including coefficients far below 1e-6.
"""
import numpy as np


def _random_sparse_int_matrix(rng, m=18, n=40):
    A = np.zeros((m, n), dtype=np.int64)
    for j in range(n):
        rows = rng.choice(m, size=int(rng.integers(2, 5)), replace=False)
        A[rows, j] = rng.integers(-4, 5, size=len(rows))
    return A


def test_exact_fraction_keeps_small_coefficients():
    from fractions import Fraction
    from straindesign.strainDesignProblem import _exact_fraction
    for v in (1.234567e-7, 3.3e-6, 0.1, 1.0 / 3.0, -2.5e-9, 7.0):
        f = _exact_fraction(v)
        assert abs(float(f) - v) <= 8.0 * np.spacing(abs(v))
    assert _exact_fraction(1.0 / 3.0) == Fraction(1, 3)


def test_kernel_of_small_coefficient_matrix_is_exact():
    from straindesign.strainDesignProblem import _exact_fraction, _nullspace_float
    rng = np.random.default_rng(11)
    A = _random_sparse_int_matrix(rng).astype(float)
    A[0, np.nonzero(A[0])[0][0]] = 1.234567e-7
    K = _nullspace_float(A)
    assert K.shape[1] == A.shape[1] - np.linalg.matrix_rank(A)
    Kd = K.toarray()
    Af = np.vectorize(_exact_fraction, otypes=[object])(A)
    for j in range(Kd.shape[1]):
        col = np.array([_exact_fraction(x) for x in Kd[:, j]], dtype=object)
        res = Af.dot(col)
        scale = max(abs(float(x)) for x in col)
        assert max(abs(float(r)) for r in res) <= 1e-12 * scale
