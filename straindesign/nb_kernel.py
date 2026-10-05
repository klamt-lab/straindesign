"""Basis changes for the nullspace (NB) dual's kernel, selected by ``SD_NB_KERNEL``.

The NB dual's projected equality block is ``K^T v - (Abar K)^T w = 0``, where ``K``'s columns
span ``null(A_eq_p)``. Its feasible set is ``v`` orthogonal to that SUBSPACE, so replacing
``K`` by ``K M`` for any invertible ``M`` leaves the dual's feasible set -- and therefore every
design -- unchanged: the block becomes ``M^T`` times itself, a row operation on an ``= 0`` block.

That freedom is the whole content of this module. Each kernel COLUMN is one dual ROW, so the
kernel's nnz is the block's nnz one for one, and a gated dual variable appears exactly in the
rows where its kernel entry is nonzero. All transforms here are exact integer column operations
(the project forbids floats in the kernel); they preserve rank by construction.

Variants (``SD_NB_KERNEL``):
  ``none``     the kernel as ``sparse_nullspace`` builds it (default).
  ``markowitz`` not a basis change but a different construction: Gauss-Jordan elimination with
               minimum-fill pivoting, built by ``strainDesignProblem._markowitz_kernel`` and never
               routed through this module.
  ``gcd``      divide every column by the gcd of its entries and fix its sign. Exact and free.
               Predicted to be a no-op on the MILP: ``nullspace_dualize`` row-scales the block by
               each row's max |coef| afterwards, which undoes any per-column scaling.
  ``greedy``   ``gcd``, then greedy pairwise column elimination: repeatedly replace column j by
               ``c_i[r] c_j - c_j[r] c_i`` (invertible, since ``c_i[r] != 0``) for the pair and
               row that removes the most nonzeros, re-reducing by the gcd each time.
  ``norm``     ``gcd``, then integer size reduction (the pairwise step of lattice reduction):
               ``c_j -> c_j - round(<c_j,c_i>/<c_i,c_i>) c_i``, unimodular, minimising the norm.
               The conditioning arm, where ``greedy`` is the sparsity arm.
  ``greedynorm`` ``greedy`` followed by ``norm``.
  ``greedycap`` ``greedy``, refusing any step that pushes |coefficient| past ``SD_NB_KERNEL_CAP``
               (default 1e9), which bounds the float cast's relative error.

``SD_NB_KERNEL_MODULE=/path/to/x.py`` instead loads an external variant: the file must define
``transform(columns, n_rows, name) -> columns``, with columns a list of ``{row: int}`` dicts.
``SD_NB_KERNEL_STATS=<path>`` appends one JSON line of kernel statistics per build.
"""
import json
import os
import time
from math import gcd

import numpy as np
from scipy import sparse


def _columns_from_kernel(K):
    """Exact integer columns of the kernel, as a list of ``{row: int}`` dicts.

    Accepts both shapes ``sparse_nullspace`` returns: an int64 CSR (integer numerators, common
    denominator already divided out) and an ``ExactCOO`` (arbitrary-precision numerators over a
    single denominator). A single global denominator scales every column alike, which is itself a
    free change of basis, so it is dropped.
    """
    if sparse.issparse(K):
        coo = sparse.coo_matrix(K)
        shape = coo.shape
        rows, cols, data = coo.row, coo.col, coo.data
        data = [int(v) for v in data]
    else:
        shape = K.shape
        rows, cols, data = K.rows, K.cols, [int(v) for v in K.data]
    columns = [dict() for _ in range(shape[1])]
    for r, c, v in zip(rows, cols, data):
        if v:
            columns[int(c)][int(r)] = int(v)
    return columns, shape[0]


def _gcd_reduce(col):
    """Divide a column by the gcd of its entries and make its first nonzero positive."""
    if not col:
        return col
    g = 0
    for v in col.values():
        g = gcd(g, abs(v))
    if g > 1:
        col = {r: v // g for r, v in col.items()}
    if col[min(col)] < 0:
        col = {r: -v for r, v in col.items()}
    return col


def _nnz(columns):
    return sum(len(c) for c in columns)


def _max_abs(columns):
    return max([0] + [abs(v) for c in columns for v in c.values()])


def _combine(cj, ci, r):
    """``c_i[r] c_j - c_j[r] c_i``: cancels row r, invertible because ``c_i[r] != 0``."""
    a, b = ci[r], cj[r]
    out = {}
    for row, v in cj.items():
        out[row] = a * v
    for row, v in ci.items():
        nv = out.get(row, 0) - b * v
        if nv:
            out[row] = nv
        else:
            out.pop(row, None)
    return _gcd_reduce({row: v for row, v in out.items() if v})


def _best_pivot(cj, ci):
    """Best row to cancel between two columns, and the resulting nnz.

    Combining on row r also cancels every shared row whose ratio ``c_j[s]/c_i[s]`` equals
    ``c_j[r]/c_i[r]``, so the most frequent ratio over the shared support is the optimal pivot and
    its multiplicity is the number of nonzeros removed. Counting ratios is linear in the overlap.
    """
    if len(ci) > len(cj):
        shared = [r for r in ci if r in cj]
    else:
        shared = [r for r in cj if r in ci]
    if not shared:
        return None, len(cj)
    counts = {}
    for r in shared:
        # exact rational ratio as a normalized integer pair
        a, b = cj[r], ci[r]
        g = gcd(abs(a), abs(b))
        key = (a // g, b // g) if b > 0 else (-(a // g), -(b // g))
        e = counts.get(key)
        if e is None:
            counts[key] = [1, r]
        else:
            e[0] += 1
    best_key = max(counts, key=lambda k: counts[k][0])
    mult, row = counts[best_key]
    return row, len(cj) + len(ci) - len(shared) - mult


def greedy_eliminate(columns, cap=None, max_rounds=64, budget_s=None):
    """Greedy pairwise column elimination; exact, rank-preserving, nnz-monotone.

    Each round scans every column j for the partner i and pivot row that removes the most
    nonzeros from j, applies every improving move, and stops when a round improves nothing.
    ``cap`` refuses a move whose result exceeds that |coefficient|.
    """
    t0 = time.time()
    columns = [_gcd_reduce(dict(c)) for c in columns]
    q = len(columns)
    rounds = 0
    for _ in range(max_rounds):
        rounds += 1
        improved = False
        # row -> columns touching it, to enumerate only pairs with a shared support
        occ = {}
        for j, c in enumerate(columns):
            for r in c:
                occ.setdefault(r, []).append(j)
        order = sorted(range(q), key=lambda j: -len(columns[j]))
        for j in order:
            cj = columns[j]
            if len(cj) < 2:
                continue
            seen = {}
            for r in cj:
                for i in occ.get(r, ()):
                    if i != j:
                        seen[i] = seen.get(i, 0) + 1
            best = (0, None, None)  # (gain, i, row)
            for i, overlap in seen.items():
                ci = columns[i]
                if 2 * overlap - len(ci) <= 0:      # gain <= 2*overlap - len(ci)
                    continue
                row, new_nnz = _best_pivot(cj, ci)
                if row is None:
                    continue
                gain = len(cj) - new_nnz
                if gain > best[0]:
                    best = (gain, i, row)
            if best[1] is None:
                continue
            cand = _combine(cj, columns[best[1]], best[2])
            if not cand or len(cand) >= len(cj):
                continue
            if cap is not None and max(abs(v) for v in cand.values()) > cap:
                continue
            for r in cj:
                if r not in cand:
                    lst = occ.get(r)
                    if lst and j in lst:
                        lst.remove(j)
            for r in cand:
                if r not in cj:
                    occ.setdefault(r, []).append(j)
            columns[j] = cand
            improved = True
        if not improved:
            break
        if budget_s is not None and time.time() - t0 > budget_s:
            break
    return columns, rounds


def size_reduce(columns, max_rounds=16, budget_s=None):
    """Integer size reduction (the pairwise step of lattice basis reduction).

    Replaces column j by ``c_j - mu c_i`` with the integer ``mu = round(<c_j,c_i>/<c_i,c_i>)``,
    which is the multiple that minimises the Euclidean norm of the result. The step is unimodular,
    so the basis -- and the subspace -- is preserved exactly. Accepted only when it removes
    nonzeros, or keeps the count and shrinks the norm, which makes the pass monotone and
    terminating. This is the conditioning arm: it targets the size of the coefficients, where
    :func:`greedy_eliminate` targets only their number.
    """
    t0 = time.time()
    columns = [_gcd_reduce(dict(c)) for c in columns]
    q = len(columns)
    sq = [sum(v * v for v in c.values()) for c in columns]
    for _ in range(max_rounds):
        improved = False
        occ = {}
        for j, c in enumerate(columns):
            for r in c:
                occ.setdefault(r, []).append(j)
        for j in sorted(range(q), key=lambda j: -sq[j]):
            cj = columns[j]
            if not cj:
                continue
            partners = set()
            for r in cj:
                partners.update(occ.get(r, ()))
            partners.discard(j)
            for i in partners:
                ci, si = columns[i], sq[i]
                if not si:
                    continue
                dot = 0
                if len(ci) < len(cj):
                    for r, v in ci.items():
                        w = cj.get(r)
                        if w:
                            dot += v * w
                else:
                    for r, v in cj.items():
                        w = ci.get(r)
                        if w:
                            dot += v * w
                if not dot:
                    continue
                mu = (2 * dot + si) // (2 * si) if dot > 0 else -((-2 * dot + si) // (2 * si))
                if not mu:
                    continue
                cand = dict(cj)
                for r, v in ci.items():
                    nv = cand.get(r, 0) - mu * v
                    if nv:
                        cand[r] = nv
                    else:
                        cand.pop(r, None)
                if not cand:
                    continue
                cand = _gcd_reduce(cand)
                sc = sum(v * v for v in cand.values())
                if len(cand) < len(cj) or (len(cand) == len(cj) and sc < sq[j]):
                    for r in cj:
                        if r not in cand:
                            lst = occ.get(r)
                            if lst and j in lst:
                                lst.remove(j)
                    for r in cand:
                        if r not in cj:
                            occ.setdefault(r, []).append(j)
                    columns[j], sq[j], cj = cand, sc, cand
                    improved = True
        if not improved:
            break
        if budget_s is not None and time.time() - t0 > budget_s:
            break
    return columns


def transform_columns(columns, n_rows, name):
    """Apply the named variant to exact integer kernel columns."""
    if name in ('', 'none'):
        return columns
    _b = os.environ.get('SD_NB_KERNEL_BUDGET')
    budget = float(_b) if _b else None
    if name == 'gcd':
        return [_gcd_reduce(dict(c)) for c in columns]
    if name == 'norm':
        return size_reduce(columns, budget_s=budget)
    if name == 'greedynorm':
        cols, _ = greedy_eliminate(columns, cap=None, budget_s=budget)
        return size_reduce(cols, budget_s=budget)
    if name in ('greedy', 'greedycap'):
        cap = float(os.environ.get('SD_NB_KERNEL_CAP', '1e9')) if name == 'greedycap' else None
        cols, _ = greedy_eliminate(columns, cap=cap, budget_s=budget)
        return cols
    raise ValueError(f'unknown SD_NB_KERNEL variant: {name}')


def kernel_to_float_csr(columns, n_rows):
    rows, cols, data = [], [], []
    for j, c in enumerate(columns):
        for r, v in c.items():
            rows.append(r)
            cols.append(j)
            data.append(float(v))
    return sparse.csr_matrix((np.asarray(data, dtype=float),
                              (np.asarray(rows, dtype=int), np.asarray(cols, dtype=int))),
                             shape=(n_rows, len(columns)))


def apply_variant(K, A_eq_p=None):
    """Entry point used by ``nullspace_dualize``: exact kernel in, float CSR out.

    An external module named by ``SD_NB_KERNEL_MODULE`` may either post-process this kernel
    (``transform(columns, n_rows, name)``) or build its own from the primal equality block
    (``build(A_eq_p, name) -> (columns, n_rows)``); ``build`` wins when both are defined.
    """
    name = os.environ.get('SD_NB_KERNEL', 'none').strip()
    stats_path = os.environ.get('SD_NB_KERNEL_STATS')
    columns, n_rows = _columns_from_kernel(K)
    before = (_nnz(columns), _max_abs(columns))
    t0 = time.time()
    mod_path = os.environ.get('SD_NB_KERNEL_MODULE')
    if mod_path:
        import importlib.util
        spec = importlib.util.spec_from_file_location('sd_nb_kernel_ext', mod_path)
        ext = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ext)
        if hasattr(ext, 'build') and A_eq_p is not None:
            columns, n_rows = ext.build(A_eq_p, name)
        else:
            columns = ext.transform(columns, n_rows, name)
    else:
        columns = transform_columns(columns, n_rows, name)
    dt = time.time() - t0
    after = (_nnz(columns), _max_abs(columns))
    if stats_path:
        with open(stats_path, 'a') as fh:
            fh.write(json.dumps(dict(variant=name, shape=[n_rows, len(columns)],
                                     nnz_before=before[0], nnz_after=after[0],
                                     max_abs_before=str(before[1]), max_abs_after=str(after[1]),
                                     transform_s=round(dt, 3))) + '\n')
    return kernel_to_float_csr(columns, n_rows)
