"""Directional flux coupling in the flux cone.

Reaction i is directionally coupled to reaction j (i -> j) when every steady-state flux with r_j = 0
also has r_i = 0, i.e. knocking out j blocks i. The cone is {N r = 0, r_k >= 0 for irreversible k};
finite bounds are ignored, so every coupling found here also holds in any bounded model whose flux
space lies inside the cone.

Per target j:
  * irreversible sources: maximise sum min(r_l, 1) over the irreversible candidates with r_j = 0.
    At an optimum every candidate that can carry flux reaches 1 (the cone lets all of them do so at
    once), so a candidate is alive iff its slack exceeds 0.5, and an optimum that adds nothing proves
    the remaining candidates blocked.
  * reversible sources: with A the irreversible reactions alive, the span of the face r_j = 0 is
    {N r = 0, r_j = 0, r_k = 0 for irreversible k not in A}. A reversible reaction can carry flux iff
    its row of the exact kernel K of N lies outside the span of the rows K_Z, Z = {j} + blocked
    irreversibles. Decided by exact rational row reduction, no LP.
"""
import logging
import time
from fractions import Fraction

import numpy as np
from scipy import sparse


def _kernel_rows(S):
    from straindesign import sparse_nullspace
    K = sparse_nullspace(sparse.csr_matrix(S))
    n = S.shape[1]
    if sparse.issparse(K):
        K = K.tocsr()
        return [{int(c): Fraction(int(v)) for c, v in zip(K[i].indices, K[i].data)} for i in range(n)]
    rows = [dict() for _ in range(n)]
    for r, c, v in zip(K.rows, K.cols, K.data):
        rows[r][c] = Fraction(int(v), int(K.denom))
    return rows


def _reduce(row, basis):
    row = dict(row)
    for p, b in basis:
        f = row.get(p)
        if f:
            for c, v in b.items():
                nv = row.get(c, 0) - f * v
                if nv:
                    row[c] = nv
                else:
                    row.pop(c, None)
    return row


def directional_couplings(S, irreversible, targets=None):
    """Return the set of pairs (i, j) with i -> j, for j in targets (default: all reactions).

    S: dense or sparse stoichiometric matrix, columns are reactions. irreversible: list of bool,
    True where the reaction is constrained to r >= 0. Reactions constrained to r <= 0 must be
    flipped by the caller.
    """
    import cplex
    S = np.asarray(S.todense() if sparse.issparse(S) else S, dtype=float)
    S = S[np.any(S != 0, axis=1)]
    m, n = S.shape
    irr = list(irreversible)
    I = [l for l in range(n) if irr[l]]
    tpos = {l: a for a, l in enumerate(I)}
    t0 = time.time()
    Krow = _kernel_rows(S)

    cp = cplex.Cplex()
    cp.set_log_stream(None)
    cp.set_results_stream(None)
    cp.set_warning_stream(None)
    cp.parameters.threads.set(1)
    cp.parameters.lpmethod.set(cp.parameters.lpmethod.values.dual)
    cp.objective.set_sense(cp.objective.sense.maximize)
    cp.variables.add(lb=[0.0 if irr[k] else -cplex.infinity for k in range(n)], ub=[cplex.infinity] * n)
    cp.variables.add(obj=[1.0] * len(I), lb=[0.0] * len(I), ub=[1.0] * len(I))
    Ssp = sparse.csr_matrix(S)
    cp.linear_constraints.add(lin_expr=[cplex.SparsePair(Ssp[i].indices.tolist(), Ssp[i].data.tolist()) for i in range(m)],
                              senses='E' * m, rhs=[0.0] * m)
    cp.linear_constraints.add(lin_expr=[cplex.SparsePair([n + a, l], [1.0, -1.0]) for a, l in enumerate(I)],
                              senses='L' * len(I), rhs=[0.0] * len(I))

    def solve():
        cp.solve()
        if cp.solution.get_status() != 1:
            # a stale basis can end a warm start badly; never read that as a stall
            cp.parameters.advance.set(0)
            cp.solve()
            cp.parameters.advance.set(1)
            if cp.solution.get_status() != 1:
                raise RuntimeError('flux coupling LP: ' + cp.solution.get_status_string())

    pairs, n_lp = set(), 0
    for j in (range(n) if targets is None else targets):
        cand = {l for l in I if l != j}
        A = set()
        cp.variables.set_lower_bounds(j, 0.0)
        cp.variables.set_upper_bounds(j, 0.0)
        while cand:
            cp.variables.set_upper_bounds([(n + tpos[l], 1.0 if l in cand else 0.0) for l in I])
            solve()
            n_lp += 1
            vals = cp.solution.get_values()
            x, tv = vals[:n], vals[n:]
            got = {l for l in cand if tv[tpos[l]] > 0.5}
            for l in [l for l in cand - got if tv[tpos[l]] > 1e-9 or x[l] > 1e-9]:
                # between noise and 1: settle on its own, cold
                cp.variables.set_upper_bounds([(n + tpos[q], 1.0 if q == l else 0.0) for q in I])
                cp.parameters.advance.set(0)
                solve()
                cp.parameters.advance.set(1)
                n_lp += 1
                if cp.solution.get_values(n + tpos[l]) > 0.5:
                    got.add(l)
                cp.variables.set_upper_bounds([(n + tpos[q], 1.0 if q in cand else 0.0) for q in I])
            if not got:
                break
            A |= got
            cand -= got
        cp.variables.set_lower_bounds(j, 0.0 if irr[j] else -cplex.infinity)
        cp.variables.set_upper_bounds(j, cplex.infinity)
        basis = []
        for z in [j] + [k for k in I if k != j and k not in A]:
            r = _reduce(Krow[z], basis)
            if r:
                p = min(r)
                inv = 1 / r[p]
                r = {c: v * inv for c, v in r.items()}
                basis = [(q, _reduce(b, [(p, r)])) for q, b in basis] + [(p, r)]
        for i in range(n):
            if i == j:
                continue
            if irr[i]:
                if i not in A:
                    pairs.add((i, j))
            elif not _reduce(Krow[i], basis):
                pairs.add((i, j))
    logging.warning('  Flux coupling: %d directional pairs, %d LPs, %.1fs' % (len(pairs), n_lp, time.time() - t0))
    return pairs
