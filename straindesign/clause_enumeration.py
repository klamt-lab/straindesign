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
"""Enumerate one pinned cost level of a strain design MILP with linear programs only.

The strain design MILP couples its binaries z to the continuous variables through indicator
constraints ("gates") and through rows that hold a binary next to continuous columns; the latter
are turned into gates too (see :class:`GateOracle`). For a fixed z the continuous part is therefore
an LP, and a z is a design exactly when that LP is feasible. When it is infeasible, a Farkas proof
names the gates that cannot all stay active, which is a clause over z that every design has to
satisfy:

    sum(z_b for gates active at z=0) + sum(1 - z_b for gates active at z=1) >= 1

:class:`GateOracle` answers "design or clause" for one z with a single warm-started LP, the
alternative system of the continuous part with the gate use minimised. :class:`ClauseEnumerator`
walks the sets of interventions of one cost depth-first, branching over the literals of a violated
clause, and asks the oracle only for sets that satisfy every clause found so far. No integer
program is solved, so the method needs an LP solver and nothing else.

Requirements, checked by :func:`split_rows` and :meth:`ClauseEnumerator.load_z_rows` (a
:class:`NotApplicable` is raised otherwise and the caller keeps its own enumeration):

- all integer variables are binary, and no indicator row holds a binary;
- the rows over the binaries are the two cost rows, pinning a positive cost (``cost.z <= k`` and
  ``-cost.z <= -k``), and clauses (exclusion rows).
"""

import logging
import queue
import time
import numpy as np
from scipy import sparse
from straindesign.names import OPTIMAL, INFEASIBLE

# seconds of oracle LPs within one level before worker processes are started; starting them costs
# a few seconds, which a short level never earns back
_PARALLEL_AFTER = 3.0

# multipliers smaller than this, relative to the largest gate multiplier, are solver noise; the
# measured gap between real multipliers and noise spans 1e-8 to 1e-15
_REL_ZERO = 1e-12
# a proof is normalised to multipliers . rhs = -1; its combination of the rows has to vanish to this
_PROOF_TOL = 1e-6

# oracle answer for an assignment that has neither a proof of infeasibility nor a feasible point
UNDECIDED = 'undecided'
FEASIBLE = 'feasible'
# a system whose rows can all be met to within this is not called infeasible
_FEAS_TOL = 1e-9
# proofs supported by at most this many rows are re-derived and checked in rational arithmetic
_EXACT_MAX_ROWS = 500


class NotApplicable(Exception):
    """The MILP does not have the structure the clause enumeration needs."""


class NumericalTrouble(Exception):
    """An oracle LP ended undecided, or its proof did not hold when re-checked."""


def split_rows(rows, is_z):
    """Sort rows into those over the binaries, those over the continuous columns and mixed ones.

    Args:
        rows (list): (indices, values, rhs, is_equality) per row, or None for a row to skip
        is_z (numpy bool array): True for the binary columns

    Returns:
        (Tuple[List, List, List]): the rows over z, over the continuous columns, and over both
    """
    z_rows, c_rows, m_rows = [], [], []
    for row in rows:
        if row is None:
            continue
        idx, val, rhs, is_eq = row
        if not is_eq and np.isinf(rhs) and rhs > 0:  # a row switched off
            continue
        on_z = is_z[idx]
        if len(idx) and on_z.all():
            z_rows.append(row)
        elif not on_z.any():
            c_rows.append(row)
        else:
            m_rows.append(row)
    return z_rows, c_rows, m_rows


def _exact_margin(A, b, y, lb, ub):
    """Confirm in rational arithmetic that rows ``A x <= b`` have no solution in the box [lb, ub].

    The floating-point multipliers y only say which rows carry the proof and roughly how. The
    multipliers are re-derived exactly: wherever a column is unbounded on a side on which the
    combination y^T A must not be negative (or positive), and the floating-point combination is
    zero there, it is required to be exactly zero; y is projected onto the rational null space of
    those conditions. Signs and the margin are then evaluated with fractions. The input numbers
    are taken as the exact rationals the floats represent.

    Args:
        A (sparse matrix), b (array): the supporting rows
        y (array): their approximate multipliers, positive
        lb, ub (arrays): bounds of all columns

    Returns:
        True if the proof holds exactly.
    """
    from fractions import Fraction
    A = sparse.csc_matrix(A)
    k = A.shape[0]
    d = A.T @ y
    scale = np.abs(d).max(initial=0.0) + np.abs(A).multiply(y[:, None]).max()
    cols = np.flatnonzero(np.diff(A.indptr))
    column = {j: [(int(i), Fraction(float(v))) for i, v in zip(A.indices[A.indptr[j]:A.indptr[j + 1]], A.data[A.indptr[j]:A.indptr[j + 1]])]
              for j in cols}
    zero = []  # columns on which the combination has to vanish exactly
    for j in cols:
        tiny = abs(d[j]) <= 1e-9 * scale
        if (np.isinf(ub[j]) and (tiny or d[j] < 0)) or (np.isinf(lb[j]) and (tiny or d[j] > 0)):
            if not tiny:
                return False
            zero.append(j)
    # null space of the conditions sum_i y_i A[i, j] = 0, j in zero, by fraction Gauss-Jordan
    M = [[Fraction(0)] * k for _ in zero]
    for r, j in enumerate(zero):
        for i, v in column[j]:
            M[r][i] = v
    pivots, r = [], 0
    for c in range(k):
        p = next((i for i in range(r, len(M)) if M[i][c] != 0), None)
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        inv = 1 / M[r][c]
        M[r] = [v * inv for v in M[r]]
        for i in range(len(M)):
            if i != r and M[i][c] != 0:
                f = M[i][c]
                M[i] = [a - f * e for a, e in zip(M[i], M[r])]
        pivots.append(c)
        r += 1
    free = [c for c in range(k) if c not in pivots]
    if not free:
        return False
    # free multipliers keep their value (as a nearby rational), the pivot ones follow exactly
    y_exact = [Fraction(0)] * k
    for c in free:
        y_exact[c] = Fraction(float(y[c])).limit_denominator(10**12)
    for row, c in zip(M, pivots):
        y_exact[c] = -sum(row[f] * y_exact[f] for f in free)
    if any(v < 0 for v in y_exact) or not any(y_exact):
        return False
    margin = -sum(v * Fraction(float(bi)) for v, bi in zip(y_exact, b))
    for j in cols:
        dj = sum(v * y_exact[i] for i, v in column[j])
        if dj > 0:
            if np.isinf(lb[j]):
                return False
            margin += dj * Fraction(float(lb[j]))
        elif dj < 0:
            if np.isinf(ub[j]):
                return False
            margin += dj * Fraction(float(ub[j]))
    return margin > 0


class GateOracle:
    """Decide for a 0/1 assignment of the binaries whether the continuous system is feasible.

    A binary that appears in a row next to continuous columns is given a continuous copy in [0, 1],
    tied to the binary by two gates of its own (z = 0 -> copy <= 0, z = 1 -> copy >= 1). After that
    every dependence on z is a gate, and the LP is the alternative (Farkas) system of

        A x <= b,   Aeq x = beq,   lb <= x <= ub,   and the gates active at the assignment,

    normalised to ``multipliers . rhs = -1``, minimising the multipliers of the gates. Between two
    queries only bounds of gate multipliers change, so the dual simplex restarts from the last
    basis.

    A proof stays valid for every z that keeps its gates active. Hence the clause: a gate active
    at z_b = 0 contributes the literal z_b, one active at z_b = 1 the literal (not z_b).

    Args:
        c_rows (list): rows over the continuous columns, as (indices, values, rhs, is_equality)
        m_rows (list): rows over binaries and continuous columns, same format
        lb, ub (list of float): bounds of all columns of the MILP
        cont (numpy int array): indices of the continuous columns
        indic_constr: the MILP's indicator constraints (A, b, sense, binv, indicval), or None
        lp_class: LP backend, constructed as lp_class(c, A_ineq, b_ineq, A_eq, b_eq, lb, ub), with
            the methods chgBound, chgSide, getDual and infinity of pyscipopt's LP and a
            solve_verdict() that returns (solution, OPTIMAL | INFEASIBLE | ERROR)
    """

    def __init__(self, c_rows, m_rows, lb, ub, cont, indic_constr, lp_class):
        self.init_args = (c_rows, m_rows, lb, ub, cont, indic_constr, lp_class)
        n = len(lb)
        lb, ub = np.array(lb, float), np.array(ub, float)
        is_cont = np.zeros(n, bool)
        is_cont[cont] = True
        copied = sorted({int(j) for idx, _, _, _ in m_rows for j in np.asarray(idx)[~is_cont[idx]]})
        lb[copied], ub[copied] = 0.0, 1.0
        cols = np.concatenate((np.asarray(cont, int), np.array(copied, int)))  # the LP's variables
        pos = np.full(n, -1)
        pos[cols] = np.arange(len(cols))
        # gates as (variables, coefficients, sense, rhs), with the binary and the value that switch them on
        gates, gate_z, gate_val = [], [], []
        if indic_constr is not None:
            G = sparse.csr_matrix(indic_constr.A)
            for g in range(G.shape[0]):
                row = G.getrow(g)
                if (pos[row.indices] < 0).any() or not is_cont[row.indices].all():
                    raise NotApplicable('an indicator row holds a binary column')
                gates.append((row.indices, row.data, indic_constr.sense[g], indic_constr.b[g]))
                gate_z.append(indic_constr.binv[g])
                gate_val.append(indic_constr.indicval[g])
        for j in copied:
            gates += [(np.array([j]), np.array([1.0]), 'L', 0.0), (np.array([j]), np.array([1.0]), 'G', 1.0)]
            gate_z += [j, j]
            gate_val += [0, 1]
        self.gates = gates
        self.gate_z, self.gate_val = np.array(gate_z, int), np.array(gate_val, int)
        self.rows = list(c_rows) + list(m_rows)
        self.cols, self.pos, self.lb, self.ub = cols, pos, lb, ub

        ri, ci, vv, rhs, lo = [], [], [], [], []

        def add(idx, val, b, free=False, sign=1.0):
            k = len(rhs)
            ri.extend(pos[idx])
            ci.extend([k] * len(idx))
            vv.extend(sign * np.asarray(val, float))
            rhs.append(sign * b)
            lo.append(-np.inf if free else 0.0)
            return k

        for idx, val, b, is_eq in self.rows:
            add(idx, val, b, free=is_eq)
        for j in cols:
            if not np.isinf(ub[j]):
                add([j], [1.0], ub[j])
            if not np.isinf(lb[j]):
                add([j], [1.0], lb[j], sign=-1.0)
        self.gate_cols = []
        for idx, val, sense, b in gates:
            mult = []
            if sense in 'LE':
                mult.append(add(idx, val, b))
            if sense in 'GE':
                mult.append(add(idx, val, b, sign=-1.0))
            self.gate_cols.append(mult)
        m = len(rhs)
        self.R = sparse.csr_matrix((vv, (ri, ci)), shape=(len(cols), m))
        self.rhs = np.array(rhs, float)
        obj = np.zeros(m)
        self.col_gate = np.full(m, -1)
        for g, mult in enumerate(self.gate_cols):
            obj[mult] = 1.0
            self.col_gate[mult] = g
        self.is_gate_col = self.col_gate >= 0
        A_eq = sparse.vstack((self.R, sparse.csr_matrix(self.rhs)), format='csr')
        b_eq = [0.0] * len(cols) + [-1.0]
        self.lp = lp_class(list(obj), sparse.csr_matrix((0, m)), [], A_eq, b_eq, lo, [np.inf] * m)
        self.active = np.ones(len(self.gate_cols), bool)
        self.n_calls = 0
        self.n_undecided = 0
        self.n_by_violation = 0  # proofs taken from the minimal-violation LP ...
        self.n_exact = 0  # ... of which confirmed in rational arithmetic
        self.n_within_tolerance = 0  # sets feasible only to within the tolerance, by that LP
        self.seconds = 0.0
        self.system = None

    def _relaxed_system(self, active, relax):
        """The LP "minimise t subject to every row and active gate relaxed by t, bounds kept",
        with t fixed to 0 unless `relax`. Built on first use; a call sets the sides of all gate rows.

        Returns:
            (solution, status) of solve_verdict
        """
        lp_class = self.init_args[-1]
        n = len(self.cols)
        if self.system is None:
            ri, ci, vv, rhs, row_gate = [], [], [], [], []

            def add(idx, val, b, gate=-1):
                ri.extend([len(rhs)] * len(idx))
                ci.extend(self.pos[idx])
                vv.extend(val)
                rhs.append(b)
                row_gate.append(gate)

            for idx, val, b, is_eq in self.rows:
                add(idx, np.asarray(val, float), b)
                if is_eq:
                    add(idx, -np.asarray(val, float), -b)
            for g, (idx, val, sense, b) in enumerate(self.gates):
                if sense in 'LE':
                    add(idx, np.asarray(val, float), b, g)
                if sense in 'GE':
                    add(idx, -np.asarray(val, float), -b, g)
            m = len(rhs)
            self.sys_A = sparse.csr_matrix((vv, (ri, ci)), shape=(m, n))
            self.sys_b = np.array(rhs, float)
            self.sys_gate = np.array(row_gate)
            self.sys_gate_rows = np.flatnonzero(self.sys_gate >= 0)
            A = sparse.hstack((self.sys_A, -np.ones((m, 1))), format='csr')
            self.system = lp_class([0.0] * n + [1.0], A, list(rhs), sparse.csr_matrix((0, n + 1)), [],
                                   list(self.lb[self.cols]) + [0.0], list(self.ub[self.cols]) + [0.0])
            self.sys_on = np.ones(m, bool)
        lp = self.system
        inf = lp.infinity()
        on = self.sys_on.copy()
        on[self.sys_gate_rows] = active[self.sys_gate[self.sys_gate_rows]]
        for i in np.flatnonzero(on != self.sys_on):
            lp.chgSide(i, -inf, self.sys_b[i] if on[i] else inf)
        self.sys_on = on
        lp.chgBound(n, 0.0, inf if relax else 0.0)
        return lp.solve_verdict()

    def _proof_from_violation(self, active):
        """Decide an assignment that the proof LP could not: minimise the uniform violation t of
        its rows. A positive minimum t* comes with dual multipliers y >= 0, sum(y) = 1, and these
        are an infeasibility proof on a scale the simplex handles even where the proof LP's
        normalisation needs multipliers of 1e6 and more (a biomass coefficient of 1e-7 does that).
        The proof is checked without any solver tolerance:

            margin = sum_j min over [lb_j, ub_j] of d_j x_j  -  y.b  > 0,     d = y^T A

        and then, where the supporting rows are few, once more in exact rational arithmetic with
        the multipliers re-derived from their support (:func:`_exact_margin`).

        A minimum of at most 1e-9 means the rows can all be met to within the feasibility
        tolerance every LP verdict here rests on, so the assignment counts as feasible, unless
        the rational check proves the opposite.

        Returns:
            the gates of the proof; FEASIBLE; or None if there is neither a usable proof nor a
            point within tolerance
        """
        x, status = self._relaxed_system(active, relax=True)
        if status != OPTIMAL:
            raise NumericalTrouble('the minimal-violation LP of an assignment did not solve')
        self.min_violation = x[len(self.cols)]
        within_tolerance = self.min_violation <= _FEAS_TOL
        y = np.abs(np.array(self.system.getDual()))
        y[~self.sys_on] = 0.0
        y[y < _REL_ZERO * y.max(initial=0.0)] = 0.0
        if not y.any():
            return FEASIBLE if within_tolerance else None
        y /= y.sum()
        rows = np.flatnonzero(y)
        lb, ub = self.lb[self.cols], self.ub[self.cols]
        d = self.sys_A.T @ y
        d[np.abs(d) < 1e-14 * np.abs(d).max(initial=0.0)] = 0.0  # cancellation noise
        with np.errstate(invalid='ignore'):
            low = np.where(d > 0, d * lb, np.where(d < 0, d * ub, 0.0))
        margin = low.sum() - y @ self.sys_b
        exact = len(rows) <= _EXACT_MAX_ROWS and _exact_margin(self.sys_A[rows], self.sys_b[rows], y[rows], lb, ub)
        if within_tolerance and not exact:
            self.n_within_tolerance += 1
            return FEASIBLE
        if not exact and not (np.isfinite(margin) and margin > 0.5 * self.min_violation):
            return None
        self.n_by_violation += 1
        self.n_exact += bool(exact)
        return np.unique(self.sys_gate[rows][self.sys_gate[rows] >= 0])

    def query(self, z_one):
        """Ask whether the assignment admits a feasible continuous system.

        Args:
            z_one (numpy bool array over all MILP columns): True where a binary is 1

        Returns:
            None if the system is feasible (the assignment is a design), the clause of the
            infeasibility proof as (positive columns, negative columns) of the MILP if it is not,
            or UNDECIDED when there is neither a feasible point nor a proof that passes its
            checks (see :meth:`_proof_from_violation`). An undecided assignment is not a design.
        """
        t0 = time.time()
        lp = self.lp
        active = z_one[self.gate_z] == (self.gate_val == 1)
        inf = lp.infinity()
        for g in np.flatnonzero(active != self.active):
            for col in self.gate_cols[g]:
                lp.chgBound(col, 0.0, inf if active[g] else 0.0)
        self.active = active
        y, status = lp.solve_verdict()
        self.n_calls += 1
        try:
            if status == OPTIMAL:
                y = np.array(y)
                size = np.abs(y[self.is_gate_col])
                keep = size > _REL_ZERO * max(size.max(initial=0.0), 1e-300)
                y[np.flatnonzero(self.is_gate_col)[~keep]] = 0.0
                if np.abs(self.R @ y).max(initial=0.0) > _PROOF_TOL or abs(self.rhs @ y + 1.0) > _PROOF_TOL:
                    raise NumericalTrouble('the infeasibility proof of an oracle LP does not hold')
                gates = np.unique(self.col_gate[np.flatnonzero(self.is_gate_col)[keep]])
                if not self.active[gates].all():
                    raise NumericalTrouble('an oracle LP used a gate that is switched off')
                return np.unique(self.gate_z[gates[self.gate_val[gates] == 0]]), np.unique(self.gate_z[gates[self.gate_val[gates] == 1]])
            # no proof: none exists, or the proof LP failed numerically
            point = self._relaxed_system(active, relax=False)[1]
            if point == OPTIMAL:
                return None
            # no point either (or that LP failed as well): the minimal violation decides
            gates = self._proof_from_violation(active)
            if gates is FEASIBLE:
                return None
            if gates is not None:
                return np.unique(self.gate_z[gates[self.gate_val[gates] == 0]]), np.unique(self.gate_z[gates[self.gate_val[gates] == 1]])
            self.n_undecided += 1
            return UNDECIDED
        finally:
            self.seconds += time.time() - t0


_worker = {}


def _worker_init(oracle_args, z_cols, n_cols):
    """Build this worker process's own oracle LP."""
    _worker['oracle'] = GateOracle(*oracle_args)
    _worker['z_cols'] = np.array(z_cols, int)
    _worker['pos'] = np.full(n_cols, -1)
    _worker['pos'][_worker['z_cols']] = np.arange(len(z_cols))
    _worker['n_cols'] = n_cols


def _worker_leaf(S_idx, candidates):
    """Ask the oracle for S plus each candidate in turn, skipping candidates that a clause found
    earlier in this call already rules out.

    Returns:
        (S_idx, candidates that complete a design, clauses as (positive, negative) z positions,
        candidates the oracle could not decide, number of LPs)
    """
    o, z_cols, pos = _worker['oracle'], _worker['z_cols'], _worker['pos']
    in_S = set(int(i) for i in S_idx)
    designs, clauses, undecided, n0 = [], [], [], o.n_calls
    for e in candidates:
        e = int(e)
        if any(e not in pc and nc <= in_S | {e} for pc, nc in clauses):
            continue
        z_one = np.zeros(_worker['n_cols'], bool)
        z_one[z_cols[S_idx]] = True
        z_one[z_cols[e]] = True
        clause = o.query(z_one)
        if clause is None:
            designs.append(e)
        elif clause is UNDECIDED:
            undecided.append(e)
        else:
            clauses.append((set(int(i) for i in pos[clause[0]]), set(int(i) for i in pos[clause[1]])))
    return S_idx, designs, [(sorted(pc), sorted(nc)) for pc, nc in clauses], undecided, o.n_calls - n0


def _pack(mask):
    """Boolean array -> its bits in 64-bit words (little end first, zero padded)."""
    bits = np.zeros(-(-len(mask) // 64) * 64, bool)
    bits[:len(mask)] = mask
    return np.packbits(bits, bitorder='little').view(np.uint64)


def _unpack(words, m):
    """64-bit words -> boolean array of length m."""
    return np.unpackbits(words.view(np.uint8), bitorder='little')[:m].astype(bool)


def _has_bit(rows, e):
    """For each row of bit words: is bit e set."""
    return (rows[:, e >> 6] >> np.uint64(e & 63)) & np.uint64(1) == 1


def _with_bit(words, e):
    out = words.copy()
    out[e >> 6] |= np.uint64(1) << np.uint64(e & 63)
    return out


_BITS_OF_BYTE = np.array([bin(i).count('1') for i in range(256)])


def _count_bits(rows):
    """Number of set bits per row."""
    return _BITS_OF_BYTE[np.ascontiguousarray(rows).view(np.uint8)].sum(1)


class _Rows:
    """Growing bit matrix, one row per clause part, 64 columns to a word."""

    def __init__(self, width):
        self.width = width
        self.a = np.zeros((256, -(-width // 64)), np.uint64)
        self.n = 0

    def add(self, cols):
        if self.n == self.a.shape[0]:
            self.a = np.vstack((self.a, np.zeros_like(self.a)))
        mask = np.zeros(self.width, bool)
        mask[cols] = True
        self.a[self.n] = _pack(mask)
        self.n += 1


class _Timeout(Exception):
    pass


class ClauseEnumerator:
    """Depth-first enumeration of all designs of one cost over clauses supplied by an oracle.

    A clause is (positive columns, negative columns): at least one positive binary is 1 or one
    negative binary is 0. Clauses without negative part are kept apart, because a set that hits
    such a clause still hits it after growing, and each node only has to look at the clauses its
    parent left unhit.

    With workers > 1, the sets that differ in their last intervention only are handed to worker
    processes, each with an oracle LP of its own (call :meth:`close` after the last level). The
    clauses they find reach the search when a worker reports back; until then the search may ask for sets such a clause would have ruled
    out, which costs LPs and never a design.

    Args:
        oracle (GateOracle)
        z_cols (list of int): the binary columns of the MILP
        n_cols (int): number of MILP columns
        workers (int): number of worker processes for the oracle LPs (0 or 1: none)
    """

    def __init__(self, oracle, z_cols, n_cols, workers=0):
        self.workers = workers if workers and workers > 1 else 0
        self.pool = None
        self.n_worker_lps = 0
        self.n_undecided = 0
        self.oracle = oracle
        self.z_cols = np.array(z_cols, int)
        self.n_cols = n_cols
        self.m = len(z_cols)
        self.pos = np.full(n_cols, -1)
        self.pos[self.z_cols] = np.arange(self.m)
        self.P = _Rows(self.m)  # clauses with positive literals only
        self.Mp, self.Mn = _Rows(self.m), _Rows(self.m)  # clauses with a negative part
        self.n_static_mixed = 0
        self.n_nodes = 0

    def load_z_rows(self, z_rows, ub_z, cost_upper, cost_lower):
        """Set the level from the two cost rows and read the exclusion rows over the binaries.

        Args:
            z_rows (list): the rows over the binaries other than the two cost rows, as
                (indices, values, rhs, is_equality)
            ub_z (numpy array): upper bounds of the binaries, in the order of z_cols
            cost_upper, cost_lower: the rows ``cost.z <= k`` and ``-cost.z <= -k``
        """
        self.Mp.n = self.Mn.n = self.n_static_mixed
        for idx, val, rhs, is_eq in z_rows:
            val = np.asarray(val, float)
            if not is_eq and val.clip(min=0).sum() <= rhs:  # cannot be violated
                continue
            if not is_eq and (np.abs(val) == 1).all() and rhs == (val > 0).sum() - 1:
                self._add_mixed(self.pos[np.asarray(idx)[val < 0]], self.pos[np.asarray(idx)[val > 0]])
            else:
                raise NotApplicable('a row over the binaries is neither a cost bound nor a clause')
        if cost_upper is None or cost_lower is None or cost_upper[3] or cost_lower[3]:
            raise NotApplicable('the cost rows are not known')
        cost = np.zeros(self.m)
        cost[self.pos[cost_upper[0]]] = cost_upper[1]
        low = np.zeros(self.m)
        low[self.pos[cost_lower[0]]] = -np.asarray(cost_lower[1])
        if not np.array_equal(cost, low) or cost_upper[2] != -cost_lower[2]:
            raise NotApplicable('the cost is not pinned to one level')
        self.allowed = np.asarray(ub_z) > 0.5
        if (cost[self.allowed] <= 0).any():
            raise NotApplicable('an intervention without positive cost')
        self.cost = cost
        self.level = cost_upper[2]

    def _add_mixed(self, pos_cols, neg_cols):
        self.Mp.add(pos_cols)
        self.Mn.add(neg_cols)

    def _ask(self, S):
        """Query the oracle for the set S (bit words); a clause it returns is stored. True if S
        is a design."""
        z_one = np.zeros(self.n_cols, bool)
        z_one[self.z_cols[_unpack(S, self.m)]] = True
        clause = self.oracle.query(z_one)
        if clause is None:
            return True
        if clause is UNDECIDED:
            self._rule_out(S)
        else:
            self._store(self.pos[clause[0]], self.pos[clause[1]])
        return False

    def _rule_out(self, S):
        """Exclude exactly the set S (the oracle could not decide it): any other set differs from
        it by an intervention it lacks or one it does not share."""
        self.n_undecided += 1
        mask = _unpack(S, self.m)
        self._store(np.flatnonzero(~mask), np.flatnonzero(mask))

    def _store(self, pos_cols, neg_cols):
        """Keep a clause the oracle returned."""
        if len(neg_cols):
            # found during the search, valid beyond this level: keep it ahead of the level's rows
            self._add_mixed(pos_cols, neg_cols)
            k, last = self.n_static_mixed, self.Mp.n - 1
            if k != last:
                self.Mp.a[[k, last]] = self.Mp.a[[last, k]]
                self.Mn.a[[k, last]] = self.Mn.a[[last, k]]
            self.n_static_mixed += 1
        else:
            self.P.add(pos_cols)

    def _submit(self, S, candidates):
        """Hand S and its last-intervention candidates to a worker; collect what has come back."""
        if self.pool is None:
            from straindesign.pool import SDPool
            self.pool = SDPool(self.workers, initializer=_worker_init, initargs=(self.oracle.init_args, self.z_cols, self.n_cols))
            self.done = queue.Queue()
            self.n_pending = 0
        self.pool.apply_async(_worker_leaf, (np.flatnonzero(_unpack(S, self.m)), candidates), callback=self.done.put, error_callback=self.done.put)
        self.n_pending += 1
        self._collect(wait=self.n_pending >= 2 * self.workers)

    def _collect(self, wait=False, everything=False):
        """Take finished worker tasks into the clause store. wait: block for one result;
        everything: block until no task is pending."""
        while self.pool is not None and self.n_pending:
            try:
                res = self.done.get(block=wait or everything)
            except queue.Empty:
                return
            self.n_pending -= 1
            wait = False
            if isinstance(res, BaseException):
                raise res
            S_idx, designs, clauses, undecided, n_lps = res
            self.n_worker_lps += n_lps
            mask = np.zeros(self.m, bool)
            mask[S_idx] = True
            S = _pack(mask)
            for e in undecided:
                self._rule_out(_with_bit(S, e))
            for pos_cols, neg_cols in clauses:
                self._store(np.array(pos_cols, int), np.array(neg_cols, int))
            for e in designs:
                self._record(_with_bit(S, e))

    def close(self):
        """Stop the worker processes. They are started within a level and kept for the next one, so
        the caller closes the enumerator after its last level."""
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
            self.pool = None

    def __del__(self):
        self.close()

    def _violated_mixed(self, S):
        n = self.Mp.n
        return np.flatnonzero(~(self.Mp.a[:n] & S).any(1) & ~(self.Mn.a[:n] & ~S).any(1))

    def _breaking(self, S):
        """Bits of the interventions whose addition to S violates a clause with a negative part:
        the clause has no positive literal in S and exactly that one negative literal outside S."""
        n = self.Mp.n
        missing = self.Mn.a[:n] & ~S
        rows = ~(self.Mp.a[:n] & S).any(1) & (_count_bits(missing) == 1) & ~(self.Mp.a[:n] & missing).any(1)
        return np.bitwise_or.reduce(missing[rows], axis=0) if rows.any() else np.zeros_like(S)

    def enumerate(self, time_limit=np.inf):
        """Enumerate every set of cost `level` that satisfies all clauses and is a design.

        Returns:
            (Tuple[List, bool]): the designs as boolean arrays over z_cols, and whether the level
            was enumerated completely (False: time limit)
        """
        self.found = []
        self.deadline = time.time() + time_limit
        self.parallel_from = self.oracle.seconds + _PARALLEL_AFTER
        # per remaining budget: the interventions that still fit, and those that use it up
        self.fits, self.exact = {}, {}
        self.cost_masks = [(c, _pack(self.cost == c)) for c in np.unique(self.cost[self.allowed])]
        try:
            S = _pack(np.zeros(self.m, bool))
            self._descend(S, _pack(~self.allowed), 0.0, np.arange(self.P.n), self.P.n)
            self._collect(everything=True)
        except _Timeout:
            self.close()
            return self.found, False
        except BaseException:
            self.close()
            raise
        return self.found, True

    def _record(self, S):
        mask = _unpack(S, self.m)
        self.found.append(mask)
        self.Mp.add([])
        self.Mn.add(np.flatnonzero(mask))

    def _descend(self, S, barred, cost_S, unhit, n_seen):
        """S: chosen interventions; barred: interventions this subtree must not use (both as bit
        words); unhit: positive clauses among the first n_seen that S does not hit."""
        self.n_nodes += 1
        if time.time() > self.deadline:
            raise _Timeout()
        self._collect()
        P = self.P
        while True:
            if P.n > n_seen:
                new = np.arange(n_seen, P.n)
                unhit = np.concatenate((unhit, new[~(P.a[new] & S).any(1)]))
                n_seen = P.n
            mixed = self._violated_mixed(S)
            if len(unhit) or len(mixed):
                break
            if self._ask(S):
                if cost_S != self.level:
                    raise NumericalTrouble('a design below the pinned cost level was not excluded')
                self._record(S)
                return
        left = self.level - cost_S
        if left not in self.fits:
            self.fits[left] = _pack(self.allowed & (self.cost <= left))
            self.exact[left] = _pack(self.allowed & (self.cost == left))
        free = ~barred & ~S & self.fits[left]
        if not free.any():
            return
        cheapest = next(c for c, mask in self.cost_masks if (free & mask).any())
        cand = np.vstack((P.a[unhit] & free, self.Mp.a[mixed] & free))
        if left < 2 * cheapest:
            # one more intervention fits: it has to repair every violated clause on its own
            last = np.bitwise_and.reduce(cand, axis=0) & self.exact[left] & ~self._breaking(S)
            last = np.flatnonzero(_unpack(last, self.m))
            if self.workers and len(last) and self.oracle.seconds > self.parallel_from:
                self._submit(S, [int(e) for e in last])
                return
            n_mixed = self.Mp.n
            for e in last:
                if P.n > n_seen and not _has_bit(P.a[n_seen:P.n], e).all():
                    continue
                T = _with_bit(S, e)
                if self.Mp.n > n_mixed and len(self._violated_mixed(T)):
                    continue
                if self._ask(T):
                    self._record(T)
            return
        count = _count_bits(cand)
        i = int(count.argmin())
        if count[i] == 0:
            return
        for e in np.flatnonzero(_unpack(cand[i], self.m)):
            self._descend(_with_bit(S, e), barred, cost_S + self.cost[e], unhit[~_has_bit(P.a[unhit], e)], n_seen)
            barred = _with_bit(barred, e)
