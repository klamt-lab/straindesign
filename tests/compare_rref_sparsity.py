"""A/B comparison: old static-sort RREF vs new dynamic-pivot RREF kernel sparsity."""
import numpy as np
import cobra
from scipy import sparse
from math import gcd, lcm
from functools import reduce
from typing import Dict, List, Tuple
from straindesign.compression import (
    RationalMatrix, create_stoichiometric_matrix, _nullspace_sparse
)
import time


def _rref_static_sort(rm: RationalMatrix) -> Tuple[Dict, int, List[int]]:
    """OLD implementation: static column pre-sort by initial nnz."""
    rows = rm.get_row_count()
    cols = rm.get_column_count()

    nnz_per_col = np.diff(rm._num_sparse.tocsc().indptr)
    col_order = np.argsort(nnz_per_col, kind='stable').tolist()
    col_inverse = [0] * cols
    for sorted_pos, orig_col in enumerate(col_order):
        col_inverse[orig_col] = sorted_pos

    num_csr = rm._num_sparse.tocsr()
    den_csr = rm._den_sparse.tocsr()

    data = {}
    for r in range(rows):
        start, end = num_csr.indptr[r], num_csr.indptr[r + 1]
        if start == end:
            continue
        row_dens = [int(den_csr.data[i]) for i in range(start, end) if den_csr.data[i] != 0]
        row_lcm = reduce(lcm, row_dens, 1) if row_dens else 1
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

    if data:
        sorted_row_keys = sorted(data.keys(), key=lambda r: len(data[r]))
        data = {new_r: data[old_r] for new_r, old_r in enumerate(sorted_row_keys)}

    pivot_cols_sorted = []
    pivot_row = 0

    for pivot_col in range(cols):
        if pivot_row >= rows:
            break
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
        elim_targets = [(r, rd[pivot_col]) for r, rd in data.items() if r != pivot_row and pivot_col in rd]

        for elim_row, elim_val in elim_targets:
            elim_row_data = data[elim_row]
            g = gcd(pivot_val, elim_val)
            pv_scaled = pivot_val // g
            ev_scaled = elim_val // g
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
            if new_row:
                data[elim_row] = new_row
                row_gcd = gcd(*new_row.values())
                if row_gcd > 1:
                    for c in new_row:
                        new_row[c] //= row_gcd
            else:
                del data[elim_row]

        pivot_row += 1

    for row_data in data.values():
        row_gcd = gcd(*row_data.values())
        if row_gcd > 1:
            for c in row_data:
                row_data[c] //= row_gcd

    original_data = {r: {col_order[sc]: v for sc, v in rd.items()} for r, rd in data.items()}
    pivot_cols_original = [col_order[p] for p in pivot_cols_sorted]
    return original_data, len(pivot_cols_original), pivot_cols_original


def _nullspace_from_rref(cols, rref_data, rank, pivot_cols):
    """Extract nullspace from RREF data (same as _nullspace_sparse internals)."""
    if rank == cols:
        return RationalMatrix(cols, 0)
    pivot_set = set(pivot_cols)
    free_cols = [c for c in range(cols) if c not in pivot_set]
    nullity = len(free_cols)
    if nullity == 0:
        return RationalMatrix(cols, 0)

    row_indices, col_indices = [], []
    numerators, denominators = [], []
    for k, free_col in enumerate(free_cols):
        row_indices.append(free_col)
        col_indices.append(k)
        numerators.append(1)
        denominators.append(1)
        for i, pivot_col in enumerate(pivot_cols):
            row_data = rref_data.get(i, {})
            val_at_free = row_data.get(free_col, 0)
            if val_at_free != 0:
                pivot_val = row_data.get(pivot_col, 1)
                g = gcd(abs(val_at_free), abs(pivot_val))
                num = -val_at_free // g
                den = pivot_val // g
                if den < 0:
                    num, den = -num, -den
                row_indices.append(pivot_col)
                col_indices.append(k)
                numerators.append(num)
                denominators.append(den)

    return RationalMatrix._build_from_sparse_data(
        row_indices, col_indices, numerators, denominators, cols, nullity)


def analyze_kernel(K_rm, label):
    K_np = K_rm.to_numpy()
    K_sp = sparse.csr_matrix(K_np)
    nullity = K_sp.shape[1]
    if nullity == 0:
        print(f"  [{label}] Kernel is empty (full rank)")
        return
    col_nnz = [K_sp.getcol(j).nnz for j in range(nullity)]
    print(f"  [{label}] K: {K_sp.shape[0]}x{nullity}, nnz={K_sp.nnz}, "
          f"overhead={K_sp.nnz/nullity:.1f}x, "
          f"col_nnz: min={min(col_nnz)} med={sorted(col_nnz)[len(col_nnz)//2]} max={max(col_nnz)}")
    return K_sp.nnz


def _rref_static_markowitz_row(rm: RationalMatrix) -> Tuple[Dict, int, List[int]]:
    """HYBRID: static column sort + Markowitz row selection (sparsest row)."""
    rows = rm.get_row_count()
    cols = rm.get_column_count()

    nnz_per_col = np.diff(rm._num_sparse.tocsc().indptr)
    col_order = np.argsort(nnz_per_col, kind='stable').tolist()
    col_inverse = [0] * cols
    for sorted_pos, orig_col in enumerate(col_order):
        col_inverse[orig_col] = sorted_pos

    num_csr = rm._num_sparse.tocsr()
    den_csr = rm._den_sparse.tocsr()

    data = {}
    for r in range(rows):
        start, end = num_csr.indptr[r], num_csr.indptr[r + 1]
        if start == end:
            continue
        row_dens = [int(den_csr.data[i]) for i in range(start, end) if den_csr.data[i] != 0]
        row_lcm = reduce(lcm, row_dens, 1) if row_dens else 1
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

    if data:
        sorted_row_keys = sorted(data.keys(), key=lambda r: len(data[r]))
        data = {new_r: data[old_r] for new_r, old_r in enumerate(sorted_row_keys)}

    pivot_cols_sorted = []
    pivot_row = 0

    for pivot_col in range(cols):
        if pivot_row >= rows:
            break
        # Markowitz row selection: sparsest row with nonzero in pivot column
        best_row, best_val = -1, 0
        best_row_nnz, best_abs = float('inf'), float('inf')
        for r, row_data in data.items():
            if r < pivot_row:
                continue
            if pivot_col in row_data:
                v = row_data[pivot_col]
                rnnz = len(row_data)
                if rnnz < best_row_nnz or (rnnz == best_row_nnz and abs(v) < best_abs):
                    best_row, best_val = r, v
                    best_row_nnz, best_abs = rnnz, abs(v)

        if best_row < 0:
            continue

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
        elim_targets = [(r, rd[pivot_col]) for r, rd in data.items() if r != pivot_row and pivot_col in rd]

        for elim_row, elim_val in elim_targets:
            elim_row_data = data[elim_row]
            g = gcd(pivot_val, elim_val)
            pv_scaled = pivot_val // g
            ev_scaled = elim_val // g
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
            if new_row:
                data[elim_row] = new_row
                row_gcd = gcd(*new_row.values())
                if row_gcd > 1:
                    for c in new_row:
                        new_row[c] //= row_gcd
            else:
                del data[elim_row]

        pivot_row += 1

    for row_data in data.values():
        row_gcd = gcd(*row_data.values())
        if row_gcd > 1:
            for c in row_data:
                row_data[c] //= row_gcd

    original_data = {r: {col_order[sc]: v for sc, v in rd.items()} for r, rd in data.items()}
    pivot_cols_original = [col_order[p] for p in pivot_cols_sorted]
    return original_data, len(pivot_cols_original), pivot_cols_original


def _rref_markowitz_product(rm: RationalMatrix) -> Tuple[Dict, int, List[int]]:
    """Full Markowitz: pick (col, row) pair minimizing (col_nnz-1)*(row_nnz-1)."""
    rows = rm.get_row_count()
    cols = rm.get_column_count()

    num_csr = rm._num_sparse.tocsr()
    den_csr = rm._den_sparse.tocsr()

    data = {}
    for r in range(rows):
        start, end = num_csr.indptr[r], num_csr.indptr[r + 1]
        if start == end:
            continue
        row_dens = [int(den_csr.data[i]) for i in range(start, end) if den_csr.data[i] != 0]
        row_lcm = reduce(lcm, row_dens, 1) if row_dens else 1
        row_data = {}
        for i in range(start, end):
            num = int(num_csr.data[i])
            den = int(den_csr.data[i]) if den_csr.data[i] != 0 else 1
            orig_col = int(num_csr.indices[i])
            scaled = num * (row_lcm // den)
            if scaled != 0:
                row_data[orig_col] = scaled
        if row_data:
            data[r] = row_data

    if data:
        sorted_row_keys = sorted(data.keys(), key=lambda r: len(data[r]))
        data = {new_r: data[old_r] for new_r, old_r in enumerate(sorted_row_keys)}

    pivot_cols = []
    pivot_row = 0
    used_cols = set()

    while pivot_row < rows:
        # Full Markowitz: find (col, row) minimizing (col_nnz-1) * (row_nnz-1)
        # First, count col nnz among active rows
        col_nnz = {}
        for r, rd in data.items():
            if r < pivot_row:
                continue
            for c in rd:
                if c not in used_cols:
                    col_nnz[c] = col_nnz.get(c, 0) + 1

        if not col_nnz:
            break

        # Find best (col, row) by Markowitz product
        best_col, best_row, best_val = -1, -1, 0
        best_score = float('inf')
        for r, rd in data.items():
            if r < pivot_row:
                continue
            rnnz = len(rd)
            for c, v in rd.items():
                if c in used_cols:
                    continue
                cnnz = col_nnz.get(c, 0)
                score = (cnnz - 1) * (rnnz - 1)
                if score < best_score or (score == best_score and abs(v) < abs(best_val)):
                    best_col, best_row, best_val = c, r, v
                    best_score = score
                    if score == 0:
                        break  # Can't beat 0
            if best_score == 0:
                break

        if best_row < 0:
            break

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
            break
        pivot_val = pivot_row_data.get(best_col, 0)
        if pivot_val == 0:
            break

        used_cols.add(best_col)
        pivot_cols.append(best_col)

        elim_targets = [(r, rd[best_col]) for r, rd in data.items() if r != pivot_row and best_col in rd]

        for elim_row, elim_val in elim_targets:
            elim_row_data = data[elim_row]
            g = gcd(pivot_val, elim_val)
            pv_scaled = pivot_val // g
            ev_scaled = elim_val // g
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
            if new_row:
                data[elim_row] = new_row
                row_gcd = gcd(*new_row.values())
                if row_gcd > 1:
                    for c in new_row:
                        new_row[c] //= row_gcd
            else:
                del data[elim_row]

        pivot_row += 1

    for row_data in data.values():
        row_gcd = gcd(*row_data.values())
        if row_gcd > 1:
            for c in row_data:
                row_data[c] //= row_gcd

    return data, len(pivot_cols), pivot_cols


for model_name in ["e_coli_core", "iML1515"]:
    print(f"\n{'='*70}")
    print(f"Model: {model_name}")
    print(f"{'='*70}")
    model = cobra.io.load_model(model_name)
    S = create_stoichiometric_matrix(model)
    if not sparse.issparse(S):
        S = sparse.csr_matrix(S)
    nmet, nrxn = S.shape
    print(f"  S: {nmet} x {nrxn}, nnz = {S.nnz}")

    rm = RationalMatrix.from_numpy(S.toarray())
    results = {}

    strategies = [
        ("static-minabs", _rref_static_sort),
        ("static-markowitz-row", _rref_static_markowitz_row),
        ("dynamic-sparsest-col", None),  # current code in compression.py
        ("full-markowitz", _rref_markowitz_product),
    ]

    for name, rref_func in strategies:
        t0 = time.perf_counter()
        if rref_func is not None:
            rref_data, rank, pivots = rref_func(rm)
            K = _nullspace_from_rref(nrxn, rref_data, rank, pivots)
        else:
            from straindesign.compression import _rref_integer_sparse
            rref_data, rank, pivots = _rref_integer_sparse(rm)
            K = _nullspace_from_rref(nrxn, rref_data, rank, pivots)
        elapsed = time.perf_counter() - t0

        rref_nnz = sum(len(rd) for rd in rref_data.values())
        K_np = K.to_numpy()
        K_sp = sparse.csr_matrix(K_np)
        nullity = K_sp.shape[1]
        kernel_nnz = K_sp.nnz
        results[name] = kernel_nnz

        col_nnzs = [K_sp.getcol(j).nnz for j in range(nullity)] if nullity > 0 else [0]
        print(f"  {name:25s}  rank={rank}  rref_nnz={rref_nnz:6d}  "
              f"K_nnz={kernel_nnz:6d}  overhead={kernel_nnz/max(nullity,1):.1f}x  "
              f"max_col={max(col_nnzs):3d}  time={elapsed:.2f}s")

    # Summary
    best = min(results, key=results.get)
    baseline = results["static-minabs"]
    print(f"\n  Best: {best} ({results[best]} nnz)")
    for name, nnz in results.items():
        ratio = baseline / nnz if nnz > 0 else 0
        print(f"    {name}: {ratio:.2f}x vs baseline")
