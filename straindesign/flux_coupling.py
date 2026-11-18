#!/usr/bin/env python3
#
# Copyright 2022-2026 Max Planck Institute Magdeburg
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
#
"""Flux coupling analysis (FCA) in the flux cone of a metabolic network

Reaction i is directionally coupled to reaction j (i -> j) when every steady-state flux with
r_j = 0 also has r_i = 0, i.e. knocking out j blocks i. The flux cone is {N r = 0, r_k >= 0 for
irreversible k}: finite bounds are ignored (as in F2C2 and QFCA), reactions that can only run
backwards are flipped, and reactions fixed to zero are blocked.

Per target j, one LP maximises sum t_l with 0 <= t_l <= 1 and t_l <= r_l over the irreversible
reactions l, with r_j = 0. The flux vectors of all reactions that can carry flux add up to one flux
vector in the cone, so at an optimum every such reaction reaches t_l = 1 and every other one stays
at 0: a reaction is alive iff t_l > 0.5. A reversible reaction i can carry flux with r_j = 0 iff
its row of the exact kernel K of N lies outside the span of the rows K_z, z in {j} and the
irreversible reactions the LP found blocked. This is decided by exact rational row reduction, not
by an LP. Fully coupled pairs have proportional rows in the exact kernel of the unblocked network;
mutually coupled pairs whose rows are not proportional are partially coupled.
"""

from collections import namedtuple
from fractions import Fraction
import logging
import time

import numpy as np
from numpy import inf
from scipy import sparse

from straindesign import MILP_LP
from straindesign.compression import RationalMatrix, StoichMatrixCompressor, CompressionMethod, basic_columns, nullspace
from straindesign.lptools import select_solver
from straindesign.names import *

# t_l above _ALIVE marks a reaction that carries flux. Values between _NOISE and _ALIVE cannot occur
# at an exact optimum of the capped LP, so each one is settled by its own LP solved from scratch.
_ALIVE = 0.5
_NOISE = 1e-9

# Exact sparse matrix: parallel lists of row and column indices and Fraction values
_Exact = namedtuple('_Exact', ['rows', 'cols', 'vals', 'shape'])


class FluxCouplingResult:
    """Result of a flux coupling analysis, in the convention of F2C2

    Attributes:
        reactions (list of str):
            The unblocked reactions, in model order. They index the rows and columns of table.

        table (numpy.ndarray of int8):
            Coupling of reaction i (row) with reaction j (column): 0 uncoupled, 1 fully coupled,
            2 partially coupled, 3 i -> j (i is directionally coupled to j: r_j = 0 forces
            r_i = 0), 4 j -> i. The diagonal is 1.

        blocked (list of str):
            The reactions that carry no flux in any steady state of the flux cone. As in F2C2,
            they are left out of the table: a blocked reaction is directionally coupled to every
            reaction, and every reaction to a blocked one.
    """
    UNCOUPLED = 0
    FULLY = 1
    PARTIALLY = 2
    DIRECTIONAL = 3
    DIRECTIONAL_REVERSE = 4

    def __init__(self, reactions, table, blocked):
        self.reactions = reactions
        self.table = table
        self.blocked = blocked

    def __repr__(self):
        return f'FluxCouplingResult({len(self.reactions)} unblocked reactions, {len(self.blocked)} blocked)'


def flux_coupling_analysis(model, compress=True, solver=None, reactions=None) -> FluxCouplingResult:
    """Flux Coupling Analysis (FCA)

    Determines for every pair of reactions whether they are fully, partially or directionally
    coupled or uncoupled, and which reactions are blocked, in the flux cone of the model
    {S r = 0, r_k >= 0 for irreversible k}. Finite flux bounds are not considered: a reaction
    with lb >= 0 is irreversible, a reaction with lb < 0 and ub <= 0 is irreversible in the
    backward direction, and a reaction with lb = ub = 0 is blocked. The LPs only decide which
    irreversible reactions can carry flux; everything else is decided in exact rational
    arithmetic on the stoichiometric matrix.

    Example:
        fc = flux_coupling_analysis(model, solver='cplex')
        fc.table[fc.reactions.index('PGI'), fc.reactions.index('PFK')]

    Args:
        model (cobra.Model):
            A metabolic model that is an instance of the cobra.Model class.

        compress (optional (bool)): (Default: True)
            Lump reactions with proportional fluxes before the analysis (exact coupled
            compression, without parallel lumping). The result is the same either way, but the
            analysis is much faster on genome-scale models with compression.

        solver (optional (str)):
            The LP solver: 'cplex', 'gurobi', 'scip' or 'glpk'.

        reactions (optional (list of str)):
            Restrict the table to these reactions. Only their couplings are computed, so a
            subset takes fewer LPs. By default, all reactions of the model are analysed.

    Returns:
        (FluxCouplingResult):
            The coupling table over the unblocked reactions and the list of blocked reactions.
    """
    solver = select_solver(solver, model)
    ids = [r.id for r in model.reactions]
    if reactions is None:
        req = list(range(len(ids)))
    else:
        pos = {r: k for k, r in enumerate(ids)}
        wanted = {getattr(r, 'id', r) for r in reactions}
        missing = wanted - set(pos)
        if missing:
            raise KeyError('Reactions not in the model: ' + ', '.join(sorted(missing)))
        req = sorted(pos[r] for r in wanted)

    t0 = time.time()
    S, irr, post = _cone(model, compress)
    n = S.shape[1]
    t_cmp = time.time() - t0
    # group[k]: the column of S that carries reaction k, or -1 if the reaction was removed as blocked
    group = np.full(len(ids), -1)
    for k, c in zip(post.rows, post.cols):
        group[k] = c
    targets = sorted({int(group[k]) for k in req if group[k] >= 0})
    blocked_c, D = _directional_couplings(S, irr, targets, solver)

    # Full coupling: proportional rows in the kernel of the unblocked network, whose null space is
    # the linear hull of the cone
    unblocked_c = [c for c in range(n) if c not in blocked_c]
    kernel = _kernel_rows(_columns(S, unblocked_c))
    tc = [c for c in targets if c not in blocked_c]
    upos = {c: a for a, c in enumerate(unblocked_c)}
    keys = {}
    cls = np.empty(len(tc), dtype=int)
    for a, c in enumerate(tc):
        row = kernel[upos[c]]
        f = row[min(row)]
        cls[a] = keys.setdefault(tuple(sorted((q, v / f) for q, v in row.items())), len(keys))
    full = cls[:, None] == cls[None, :]
    Dt = D[np.ix_(tc, tc)]
    np.fill_diagonal(Dt, True)
    mutual = Dt & Dt.T
    if np.any(full & ~mutual):
        raise RuntimeError('Flux coupling: an LP found a fully coupled pair uncoupled. The LP solver ran into '
                           'numerical trouble; try another solver.')
    code = np.where(mutual, np.where(full, 1, 2), np.where(Dt, 3, np.where(Dt.T, 4, 0))).astype(np.int8)
    np.fill_diagonal(code, 1)

    # Reactions lumped into one column are fully coupled with each other, and share its couplings
    tpos = {c: a for a, c in enumerate(tc)}
    blocked = [ids[k] for k in req if group[k] < 0 or group[k] in blocked_c]
    kept = [k for k in req if group[k] >= 0 and group[k] not in blocked_c]
    idx = np.array([tpos[group[k]] for k in kept], dtype=int)
    table = code[np.ix_(idx, idx)]
    logging.info(f'  Flux coupling: {len(kept)} unblocked and {len(blocked)} blocked reactions, {n} reactions after '
                 f'compression ({t_cmp:.1f}s), {time.time() - t0:.1f}s in total.')
    return FluxCouplingResult([ids[k] for k in kept], table, blocked)


def _cone(model, compress):
    """Flux cone of the model: (S, irr, post)

    S is the (compressed) stoichiometric matrix with exact entries. Backward-irreversible columns
    are flipped, so irr[c] means r_c >= 0. post has an entry (k, c) for every reaction k that is
    carried by column c of S; reactions without an entry are blocked."""
    sgn, fixed = [], set()
    for k, r in enumerate(model.reactions):
        lb, ub = float(r.lower_bound), float(r.upper_bound)
        sgn.append(1 if lb >= 0 else (-1 if ub <= 0 else 0))  # direction of an irreversible reaction
        if lb == 0 and ub == 0:
            fixed.add(k)
    stoich = RationalMatrix.from_cobra_model(model)
    nr = len(sgn)
    if compress:
        # Parallel lumping stays off: parallel reactions (e.g. isozymes) are not coupled with each
        # other, and the couplings of their lump are not those of its members. Coupled lumping only
        # merges reactions with proportional fluxes, which are fully coupled.
        names = [str(k) for k in range(nr)]
        bounds = [(0.0 if s > 0 else -inf, 0.0 if s < 0 else inf) for s in sgn]
        record = StoichMatrixCompressor(*CompressionMethod.standard()).compress(
            stoich, [str(i) for i in range(stoich.get_row_count())], names, {names[k] for k in fixed}, bounds)
        S, post = _exact(record.cmp), _exact(record.post)
    else:
        keep = [k for k in range(nr) if k not in fixed]
        S = _columns(_exact(stoich), keep)
        post = _Exact(keep, list(range(len(keep))), [Fraction(1)] * len(keep), (nr, len(keep)))
    # r_k = post[k, c] * r_c, so an irreversible reaction fixes the direction of its column. Members
    # that demand opposite directions leave the column blocked (the compression normally removes
    # such groups itself).
    direction = [0] * S.shape[1]
    conflict = set()
    for k, c, f in zip(post.rows, post.cols, post.vals):
        if sgn[k]:
            d = sgn[k] if f > 0 else -sgn[k]
            if direction[c] == 0:
                direction[c] = d
            elif direction[c] != d:
                conflict.add(c)
    S = _Exact(S.rows, S.cols, [-v if direction[c] < 0 else v for c, v in zip(S.cols, S.vals)], S.shape)
    keep = [c for c in range(S.shape[1]) if c not in conflict]
    irr = [direction[c] != 0 for c in keep]
    S = _columns(S, keep)
    # Keep a basis of the rows: dependent rows (conservation relations) add nothing, and with them
    # CPLEX can end the LP with unscaled infeasibilities at the 1e-9 tolerance.
    rows = basic_columns(RationalMatrix.from_fractions(zip(S.cols, S.rows, S.vals), (S.shape[1], S.shape[0])))
    new = {r: a for a, r in enumerate(rows)}
    ent = [(new[r], c, v) for r, c, v in zip(S.rows, S.cols, S.vals) if r in new]
    S = _Exact([e[0] for e in ent], [e[1] for e in ent], [e[2] for e in ent], (len(rows), S.shape[1]))
    return S, irr, _columns(post, keep)


def _exact(rm):
    """Exact entries of a RationalMatrix."""
    e = rm.to_coo_exact()
    return _Exact(list(e.rows), list(e.cols), [Fraction(v, e.denom) for v in e.data], tuple(e.shape))


def _columns(A, cols):
    """The listed columns of A, renumbered in the order given."""
    new = {c: a for a, c in enumerate(cols)}
    ent = [(r, new[c], v) for r, c, v in zip(A.rows, A.cols, A.vals) if c in new]
    return _Exact([e[0] for e in ent], [e[1] for e in ent], [e[2] for e in ent], (A.shape[0], len(cols)))


def _kernel_rows(A):
    """Rows of an exact kernel basis of A, as {column: Fraction}; an empty row is a blocked reaction."""
    K = nullspace(RationalMatrix.from_fractions(zip(A.rows, A.cols, A.vals), A.shape))
    _, rows = K.to_sparse_pattern()
    return [{int(c): v for c, v in rows.get(i, {}).items()} for i in range(A.shape[1])]


def _reduce(row, basis):
    """Remainder of row after elimination with a reduced echelon basis [(pivot, row), ...]."""
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


def _extend(basis, rows):
    """Reduced echelon basis of the span of basis and rows. The input basis is not modified."""
    basis = list(basis)
    for row in rows:
        r = _reduce(row, basis)
        if r:
            p = min(r)
            inv = 1 / r[p]
            r = {c: v * inv for c, v in r.items()}
            basis = [(q, _reduce(b, [(p, r)])) for q, b in basis] + [(p, r)]
    return basis


class _CappedLP:
    """max sum t_l s.t. S r = 0, r_l >= 0 for irreversible l, 0 <= t_l <= min(r_l, 1)

    One LP object serves all targets: a target only changes a few bounds, so each solve restarts
    from the previous optimal basis."""

    def __init__(self, S, irr, solver):
        m, n = S.shape
        self.n, self.solver = n, solver
        self.I = [l for l in range(n) if irr[l]]
        self.tpos = {l: n + a for a, l in enumerate(self.I)}
        nt = len(self.I)
        A = sparse.csr_matrix(([float(v) for v in S.vals], (S.rows, S.cols)), shape=(m, n))
        self.A_eq = sparse.hstack((A, sparse.csr_matrix((m, nt)))).tocsr()
        # t_l - r_l <= 0
        self.A_ineq = sparse.csr_matrix((np.r_[np.ones(nt), -np.ones(nt)], (np.r_[np.arange(nt), np.arange(nt)],
                                                                             np.r_[n + np.arange(nt), self.I])),
                                        shape=(nt, n + nt))
        self.c = [0.0] * n + [-1.0] * nt
        self.lb = [0.0 if irr[k] else -inf for k in range(n)] + [0.0] * nt
        self.ub = [inf] * n + [1.0] * nt
        # dual simplex: a target only changes bounds, so the previous basis stays dual feasible
        self.lp = self._build(self.ub, LP_METHOD_DUAL)
        self.num_lp = 0

    def _build(self, ub, method):
        lp = MILP_LP(c=self.c, A_ineq=self.A_ineq, b_ineq=[0.0] * self.A_ineq.shape[0], A_eq=self.A_eq,
                     b_eq=[0.0] * self.A_eq.shape[0], lb=list(self.lb), ub=list(ub), solver=self.solver)
        lp.set_lp_method(method)
        return lp

    def set_bounds(self, idx, lb, ub):
        for i, l, u in zip(idx, lb, ub):
            self.lb[i], self.ub[i] = l, u
        self.lp.set_lb([[i, l] for i, l in zip(idx, lb)])
        self.lp.set_ub([[i, u] for i, u in zip(idx, ub)])

    def _solve_cold(self, ub):
        self.num_lp += 1
        # primal simplex: x = 0 is feasible, so a fresh solve never needs a feasibility phase
        x, _, status = self._build(ub, LP_METHOD_PRIMAL).solve()
        return x, status

    def alive(self):
        """The irreversible reactions that carry flux under the current bounds."""
        self.num_lp += 1
        x, _, status = self.lp.solve()
        if status != OPTIMAL:
            # a warm start from a stale basis can fail where a fresh solve does not; a failed LP must
            # never be read as "everything blocked"
            x, status = self._solve_cold(self.ub)
            if status != OPTIMAL:
                raise RuntimeError(f'Flux coupling: LP not solved to optimality (status {status}).')
        cand = [l for l in self.I if self.ub[self.tpos[l]] > 0]
        alive = {l for l in cand if x[self.tpos[l]] > _ALIVE}
        for l in cand:
            if l not in alive and (x[self.tpos[l]] > _NOISE or x[l] > _NOISE):
                ub = list(self.ub)
                for q in cand:
                    ub[self.tpos[q]] = 1.0 if q == l else 0.0
                y, status = self._solve_cold(ub)
                if status != OPTIMAL:
                    raise RuntimeError(f'Flux coupling: LP not solved to optimality (status {status}).')
                if y[self.tpos[l]] > _ALIVE:
                    alive.add(l)
        return alive


def _directional_couplings(S, irr, targets, solver):
    """Blocked reactions and directional couplings of the network S (irr[c]: r_c >= 0)

    Returns (blocked, D): blocked is the set of blocked columns, and D[i, j] is True iff i -> j,
    for every unblocked target j."""
    n = S.shape[1]
    K = _kernel_rows(S)
    lp = _CappedLP(S, irr, solver)
    I = set(lp.I)
    # Blocked reactions: the same LP without a target
    dead = I - lp.alive()
    basis0 = _extend([], [K[z] for z in sorted(dead)])
    blocked = dead | {i for i in range(n) if not irr[i] and not _reduce(K[i], basis0)}
    blk = sorted(blocked)
    lp.set_bounds(blk, [0.0] * len(blk), [0.0] * len(blk))
    tb = [lp.tpos[l] for l in blk if l in I]
    lp.set_bounds(tb, [0.0] * len(tb), [0.0] * len(tb))
    reversible = [i for i in range(n) if not irr[i] and i not in blocked]
    D = np.zeros((n, n), dtype=bool)
    for j in targets:
        if j in blocked:
            continue
        idx, lb, ub = [j], [lp.lb[j]], [lp.ub[j]]
        if irr[j]:
            idx.append(lp.tpos[j])
            lb.append(0.0)
            ub.append(1.0)
        lp.set_bounds(idx, [0.0] * len(idx), [0.0] * len(idx))
        dead = I - blocked - {j} - lp.alive()
        lp.set_bounds(idx, lb, ub)
        basis = _extend(basis0, [K[j]] + [K[z] for z in sorted(dead)])
        for i in dead:
            D[i, j] = True
        # a row with a nonzero outside the columns the basis rows use cannot lie in their span
        support = set().union(*(b.keys() for _, b in basis))
        for i in reversible:
            if i != j and K[i].keys() <= support and not _reduce(K[i], basis):
                D[i, j] = True
    logging.info(f'  Flux coupling: {len(targets)} targets, {lp.num_lp} LPs.')
    return blocked, D
