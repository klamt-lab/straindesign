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
constraints ("gates") alone. For a fixed z the continuous part is therefore an LP, and a z is a
design exactly when that LP is feasible. When it is infeasible, a Farkas proof names the gates
that cannot all stay active, which is a clause over z that every design has to satisfy:

    sum(z_b for gates active at z=0) + sum(1 - z_b for gates active at z=1) >= 1

:class:`GateOracle` answers "design or clause" for one z with a single warm-started LP, the
alternative system of the continuous part with the gate use minimised. :class:`ClauseEnumerator`
walks the sets of interventions of one cost depth-first, branching over the literals of a violated
clause, and asks the oracle only for sets that satisfy every clause found so far. No integer
program is solved, so the method needs an LP solver and nothing else.

Requirements, checked by :func:`split_rows` and :meth:`ClauseEnumerator.load_z_rows` (a
:class:`NotApplicable` is raised otherwise and the caller keeps its own enumeration):

- every row holds binaries only or continuous variables only, and all integer variables are binary;
- the rows over the binaries are a pinned positive cost (``cost.z <= k`` and ``-cost.z <= -k``) and
  clauses (exclusion rows).
"""

import logging
import time
import numpy as np
from scipy import sparse

# multipliers smaller than this, relative to the largest gate multiplier, are solver noise; the
# measured gap between real multipliers and noise spans 1e-8 to 1e-15
_REL_ZERO = 1e-12
_PROOF_TOL = 1e-6


class NotApplicable(Exception):
    """The MILP does not have the structure the clause enumeration needs."""


class NumericalTrouble(Exception):
    """An oracle LP ended undecided, or its proof did not hold when re-checked."""


def split_rows(rows, is_z):
    """Sort rows into those over the binaries and those over the continuous columns.

    Args:
        rows (list): (indices, values, rhs, is_equality) per row, or None for a row to skip
        is_z (numpy bool array): True for the binary columns

    Returns:
        (Tuple[List, List]): the rows over z and the rows over the continuous columns
    """
    z_rows, c_rows = [], []
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
            raise NotApplicable('a row mixes binary and continuous columns')
    return z_rows, c_rows


class GateOracle:
    """Decide for a 0/1 assignment of the gate binaries whether the continuous system is feasible.

    The LP is the alternative (Farkas) system of

        A x <= b,   Aeq x = beq,   lb <= x <= ub,   and the gates active at the assignment,

    normalised to ``multipliers . rhs = -1``, minimising the multipliers of the gates. Between two
    queries only the bounds of gate multipliers change, so the dual simplex restarts from the last
    basis.

    Args:
        c_rows (list): rows over the continuous columns, as (indices, values, rhs, is_equality)
        lb, ub (list of float): bounds of all columns of the MILP
        cont (numpy int array): indices of the continuous columns
        indic_constr: the MILP's indicator constraints (A, b, sense, binv, indicval)
        lp_class: LP backend, constructed as lp_class(c, A_ineq, b_ineq, A_eq, b_eq, lb, ub), with
            the pyscipopt LP methods chgBound, solve, getPrimal, isOptimal, isPrimalFeasible,
            getObjVal, isInfinity and infinity
    """

    def __init__(self, c_rows, lb, ub, cont, indic_constr, lp_class):
        n = len(lb)
        pos = np.full(n, -1)
        pos[cont] = np.arange(len(cont))
        ri, ci, vv, rhs, lo = [], [], [], [], []

        def add(idx, val, b, free=False, sign=1.0):
            k = len(rhs)
            ri.extend(pos[idx])
            ci.extend([k] * len(idx))
            vv.extend(sign * np.asarray(val, float))
            rhs.append(sign * b)
            lo.append(-np.inf if free else 0.0)
            return k

        for idx, val, b, is_eq in c_rows:
            add(idx, val, b, free=is_eq)
        for j in cont:
            if not np.isinf(ub[j]):
                add([j], [1.0], ub[j])
            if not np.isinf(lb[j]):
                add([j], [1.0], lb[j], sign=-1.0)
        G = sparse.csr_matrix(indic_constr.A)
        self.gate_z = np.array(indic_constr.binv, int)
        self.gate_val = np.array(indic_constr.indicval, int)
        self.gate_cols = []
        for g in range(G.shape[0]):
            row = G.getrow(g)
            if (pos[row.indices] < 0).any():
                raise NotApplicable('an indicator row holds a binary column')
            cols = []
            if indic_constr.sense[g] in 'LE':
                cols.append(add(row.indices, row.data, indic_constr.b[g]))
            if indic_constr.sense[g] in 'GE':
                cols.append(add(row.indices, row.data, indic_constr.b[g], sign=-1.0))
            self.gate_cols.append(cols)
        m = len(rhs)
        self.R = sparse.csr_matrix((vv, (ri, ci)), shape=(len(cont), m))
        self.rhs = np.array(rhs)
        obj = np.zeros(m)
        self.is_gate_col = np.zeros(m, bool)
        self.col_gate = np.full(m, -1)
        for g, cols in enumerate(self.gate_cols):
            obj[cols] = 1.0
            self.is_gate_col[cols] = True
            self.col_gate[cols] = g
        A_eq = sparse.vstack((self.R, sparse.csr_matrix(self.rhs)), format='csr')
        b_eq = [0.0] * len(cont) + [-1.0]
        self.lp = lp_class(list(obj), sparse.csr_matrix((0, m)), [], A_eq, b_eq, lo, [np.inf] * m)
        self.active = np.ones(len(self.gate_cols), bool)
        self.n_calls = 0
        self.seconds = 0.0

    def query(self, z_one):
        """Ask whether the assignment admits a feasible continuous system.

        Args:
            z_one (numpy bool array over all MILP columns): True where a binary is 1

        Returns:
            None if the system is feasible (the assignment is a design), otherwise the indices of
            the gates in the infeasibility proof.
        """
        t0 = time.time()
        lp = self.lp
        active = z_one[self.gate_z] == (self.gate_val == 1)
        inf = lp.infinity()
        for g in np.flatnonzero(active != self.active):
            for col in self.gate_cols[g]:
                lp.chgBound(col, 0.0, inf if active[g] else 0.0)
        self.active = active
        lp.solve()
        self.n_calls += 1
        try:
            if lp.isOptimal():
                y = np.array(lp.getPrimal())
                size = np.abs(y[self.is_gate_col])
                keep = size > _REL_ZERO * max(size.max(), 1e-300) if len(size) else size > 0
                y[np.flatnonzero(self.is_gate_col)[~keep]] = 0.0
                scale = max(1.0, np.abs(y).max())
                if np.abs(self.R @ y).max(initial=0.0) > _PROOF_TOL * scale or abs(self.rhs @ y + 1.0) > _PROOF_TOL * scale:
                    raise NumericalTrouble('the infeasibility proof of an oracle LP does not hold')
                gates = np.unique(self.col_gate[np.flatnonzero(self.is_gate_col)[keep]])
                if not self.active[gates].all():
                    raise NumericalTrouble('an oracle LP used a gate that is switched off')
                return gates
            if lp.isInfinity(lp.getObjVal()) and not lp.isPrimalFeasible():
                return None
            raise NumericalTrouble('an oracle LP ended without a verdict')
        finally:
            self.seconds += time.time() - t0


class _Rows:
    """Growing boolean matrix, one row per clause part."""

    def __init__(self, width):
        self.a = np.zeros((256, width), bool)
        self.n = 0

    def add(self, cols):
        if self.n == self.a.shape[0]:
            self.a = np.vstack((self.a, np.zeros_like(self.a)))
        self.a[self.n, cols] = True
        self.n += 1


class _Timeout(Exception):
    pass


class ClauseEnumerator:
    """Depth-first enumeration of all designs of one cost over clauses supplied by an oracle.

    A clause is (positive columns, negative columns): at least one positive binary is 1 or one
    negative binary is 0. Clauses without negative part are kept apart, because a set that hits
    such a clause still hits it after growing, and each node only has to look at the clauses its
    parent left unhit.

    Args:
        oracle (GateOracle)
        z_cols (list of int): the binary columns of the MILP
        n_cols (int): number of MILP columns
    """

    def __init__(self, oracle, z_cols, n_cols):
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

    def load_z_rows(self, z_rows, ub_z):
        """Read the pinned cost and the exclusion rows from the MILP's rows over the binaries.

        Args:
            z_rows (list): (indices, values, rhs, is_equality) per row
            ub_z (numpy array): upper bounds of the binaries, in the order of z_cols
        """
        self.Mp.n = self.Mn.n = self.n_static_mixed
        upper, lower = [], []
        for idx, val, rhs, is_eq in z_rows:
            val = np.asarray(val, float)
            if not is_eq and val.clip(min=0).sum() <= rhs:  # cannot be violated
                continue
            if is_eq:
                raise NotApplicable('an equality row over the binaries')
            if (np.abs(val) == 1).all() and rhs == (val > 0).sum() - 1:
                self._add_mixed(self.pos[np.asarray(idx)[val < 0]], self.pos[np.asarray(idx)[val > 0]])
            elif (val > 0).all() and rhs >= 0:
                upper.append((idx, val, rhs))
            elif (val < 0).all() and rhs <= 0:
                lower.append((idx, -val, -rhs))
            else:
                raise NotApplicable('a row over the binaries is neither a cost bound nor a clause')
        if len(upper) != 1 or len(lower) != 1:
            raise NotApplicable('the cost is not pinned by one upper and one lower row')
        cost = np.zeros(self.m)
        cost[self.pos[upper[0][0]]] = upper[0][1]
        low = np.zeros(self.m)
        low[self.pos[lower[0][0]]] = lower[0][1]
        if not np.array_equal(cost, low) or upper[0][2] != lower[0][2]:
            raise NotApplicable('the cost is not pinned to one level')
        self.allowed = np.asarray(ub_z) > 0.5
        if (cost[self.allowed] <= 0).any():
            raise NotApplicable('an intervention without positive cost')
        self.cost = cost
        self.level = upper[0][2]

    def _add_mixed(self, pos_cols, neg_cols):
        self.Mp.add(pos_cols)
        self.Mn.add(neg_cols)

    def _ask(self, S):
        """Query the oracle for the set S; a clause it returns is stored. True if S is a design."""
        z_one = np.zeros(self.n_cols, bool)
        z_one[self.z_cols[S]] = True
        gates = self.oracle.query(z_one)
        if gates is None:
            return True
        o = self.oracle
        pos_cols = self.pos[o.gate_z[gates[o.gate_val[gates] == 0]]]
        neg_cols = self.pos[o.gate_z[gates[o.gate_val[gates] == 1]]]
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
        return False

    def _violated_mixed(self, S):
        n = self.Mp.n
        return np.flatnonzero(~(self.Mp.a[:n] & S).any(1) & ~(self.Mn.a[:n] & ~S).any(1))

    def enumerate(self, time_limit=np.inf):
        """Enumerate every set of cost `level` that satisfies all clauses and is a design.

        Returns:
            (Tuple[List, bool]): the designs as boolean arrays over z_cols, and whether the level
            was enumerated completely (False: time limit)
        """
        self.found = []
        self.deadline = time.time() + time_limit
        try:
            self._descend(np.zeros(self.m, bool), ~self.allowed, 0.0, np.arange(self.P.n), self.P.n)
        except _Timeout:
            return self.found, False
        return self.found, True

    def _record(self, S):
        self.found.append(S.copy())
        self.Mp.add([])
        self.Mn.add(np.flatnonzero(S))

    def _descend(self, S, barred, cost_S, unhit, n_seen):
        """S: chosen interventions; barred: interventions this subtree must not use; unhit: positive
        clauses among the first n_seen that S does not hit."""
        self.n_nodes += 1
        if time.time() > self.deadline:
            raise _Timeout()
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
        free = ~barred & ~S & (self.cost <= left)
        if not free.any():
            return
        cand = np.vstack((P.a[unhit] & free, self.Mp.a[mixed] & free))
        if left < 2 * self.cost[free].min():
            # one more intervention fits: it has to repair every violated clause on its own
            for e in np.flatnonzero(cand.all(0) & (self.cost == left)):
                if P.n > n_seen and not P.a[n_seen:P.n, e].all():
                    continue
                T = S.copy()
                T[e] = True
                if len(self._violated_mixed(T)):
                    continue
                if self._ask(T):
                    self._record(T)
            return
        count = cand.sum(1)
        i = int(count.argmin())
        if count[i] == 0:
            return
        barred = barred.copy()
        for e in np.flatnonzero(cand[i]):
            T = S.copy()
            T[e] = True
            self._descend(T, barred, cost_S + self.cost[e], unhit[~P.a[unhit, e]], n_seen)
            barred[e] = True
