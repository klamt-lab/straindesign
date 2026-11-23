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
"""Enumerate strain designs with linear programs over the metabolic model (PROTOTYPE).

Three layers, each kept out of the others:

- **Interventions** z (gene or reaction knockouts) with costs. They are the search variables.
- **Boolean layer**: every knockable reaction of the (compressed) model, an *atom*, is blocked when
  a monotone Boolean formula over the interventions holds (its GPR rule, dualised). A set of
  interventions z thereby blocks a set of atoms X(z). No linear program sees a gene.
- **LP oracle** in flux space: for a set X of blocked atoms it decides, per module, whether the
  module's flux region is empty. Every answer comes with a witness that carries it over to many
  other sets without a further LP:

  - a flux vector v of the region with ``supp(v) & X' == 0`` shows the region non-empty for X'
    (a *witness*, valid for every set that spares its support),
  - an infeasibility proof that uses the zero bounds of the atoms ``C <= X`` only shows the
    region empty for every ``X' >= C`` (a *core*).

  For a SUPPRESS module the witnesses are clauses ("block one of these atoms"), for a PROTECT
  module the cores are ("do not block all of these").

The search walks the intervention sets of one cost depth-first over the clauses, as
``clause_enumeration`` does over the MILP's binaries, and asks an LP only for a set that neither a
stored witness nor a stored core decides.
"""

import logging
import os
import time
import numpy as np
from scipy import sparse
from straindesign.names import *

# largest right-hand side of the module constraints of a flux cone after scaling: the region is
# scale-free, and with fluxes of this order a required flux of 1e-7 times the growth rate stands
# clear of the LP tolerance
_CONE_SCALE = 100.0
# a flux counts as part of a witness's support above this (too low only weakens a clause; too high
# would drop an atom the flux needs and make the clause wrong)
_SUPP_TOL = 1e-9
# multipliers smaller than this, relative to the largest, are not part of a core (too high would
# drop an atom the proof needs and make the clause wrong)
_CORE_REL_TOL = 1e-11
# objective weight of the atoms a targeted query should keep out of its witness (the others weigh 1)
_AVOID_WEIGHT = 1000.0
# a reduced cost above this makes the zero bound of a blocked atom part of an optimality proof
_CORE_ABS_TOL = 1e-9
# slack, relative, by which a flux may miss the inner optimum and still count as optimal
_OPT_TOL = 1e-7
# weight of flux through reactions that are no atoms: keeps witnesses free of large cycles
_BACKGROUND_WEIGHT = 1e-3

FEASIBLE, INFEAS = 'feasible', 'infeasible'


class LPError(Exception):
    """An oracle LP ended without a usable verdict."""


# ------------------------------------------------------------------------------------------------
# LP backends: one object per LP, bound and objective changes between warm-started solves
# ------------------------------------------------------------------------------------------------
class _LPBase:
    """min c.x  s.t.  A_ineq x <= b_ineq, A_eq x = b_eq, lb <= x <= ub."""
    n_solves = 0
    seconds = 0.0

    n_restarts = 0

    def solve(self):
        """OPTIMAL, INFEASIBLE or ERROR. The LPs posed here are bounded below, so any other status
        of a warm-started solve (Gurobi's dual simplex can report UNBOUNDED after 0 iterations from
        an old basis) is settled by one solve from scratch."""
        t0 = time.perf_counter()
        try:
            status = self._solve()
            if status not in (OPTIMAL, INFEASIBLE):
                _LPBase.n_restarts += 1
                self._from_scratch()
                status = self._solve()
            return status if status in (OPTIMAL, INFEASIBLE) else ERROR
        finally:
            _LPBase.n_solves += 1
            _LPBase.seconds += time.perf_counter() - t0

    def solve_from_scratch(self):
        """Solve without the stored basis (second opinion on a verdict)."""
        self._from_scratch()
        return self.solve()

    def _from_scratch(self):
        pass


class _GurobiLP(_LPBase):

    def __init__(self, c, A_ineq, b_ineq, A_eq, b_eq, lb, ub):
        from straindesign.gurobi_interface import Gurobi_MILP_LP
        self.m = Gurobi_MILP_LP(list(c), A_ineq, list(b_ineq), A_eq, list(b_eq), list(lb), list(ub), 'C' * len(c))
        self.m.params.Method = 1  # dual simplex: a query changes bounds
        self.m.params.Presolve = 0
        self.m.params.Threads = 1
        self.m.params.DualReductions = 0
        self.m.update()
        self.v = self.m.getVars()
        self.n = len(self.v)
        self.n_ineq = A_ineq.shape[0]

    def set_bounds(self, idx, lo, hi):
        vs = [self.v[i] for i in idx]
        self.m.setAttr('LB', vs, [max(float(v), -1e100) for v in lo])
        self.m.setAttr('UB', vs, [min(float(v), 1e100) for v in hi])

    def set_obj(self, idx, val):
        self.m.setAttr('Obj', [self.v[i] for i in idx], list(val))

    def _solve(self):
        self.m.optimize()
        s = self.m.Status
        if s == 2:
            return OPTIMAL
        if s == 3:
            return INFEASIBLE
        if s == 5:
            return UNBOUNDED
        if s == 13 and self.m.SolCount:
            return OPTIMAL
        return ERROR

    def _from_scratch(self):
        self.m.reset()

    def primal(self):
        return np.array(self.m.getAttr('X', self.v))

    def objval(self):
        return self.m.ObjVal

    def redcost(self):
        return np.array(self.m.getAttr('RC', self.v))

    def rowdual(self):
        return np.array(self.m.getAttr('Pi', self.m.getConstrs()))

    def iterations(self):
        return self.m.IterCount

    def chg_coef(self, eq_row, col, val):
        if not hasattr(self, 'rows'):
            self.rows = self.m.getConstrs()
        self.m.chgCoeff(self.rows[self.n_ineq + eq_row], self.v[col], float(val))


class _CplexLP(_LPBase):

    def __init__(self, c, A_ineq, b_ineq, A_eq, b_eq, lb, ub):
        from straindesign.cplex_interface import Cplex_MILP_LP
        import cplex
        inf = cplex.infinity
        lb = [max(float(v), -inf) for v in lb]
        ub = [min(float(v), inf) for v in ub]
        self.m = Cplex_MILP_LP(list(c), A_ineq, list(b_ineq), A_eq, list(b_eq), lb, ub, 'C' * len(c))
        self.m.parameters.threads.set(1)
        self.m.parameters.lpmethod.set(2)  # dual simplex
        self.m.parameters.preprocessing.presolve.set(0)
        self.n = len(c)
        self.inf = inf
        self.n_ineq = A_ineq.shape[0]

    def chg_coef(self, eq_row, col, val):
        self.m.linear_constraints.set_coefficients(int(self.n_ineq + eq_row), int(col), float(val))

    def set_bounds(self, idx, lo, hi):
        inf = self.inf
        self.m.variables.set_lower_bounds([(int(i), max(float(v), -inf)) for i, v in zip(idx, lo)])
        self.m.variables.set_upper_bounds([(int(i), min(float(v), inf)) for i, v in zip(idx, hi)])

    def set_obj(self, idx, val):
        self.m.objective.set_linear([(int(i), float(v)) for i, v in zip(idx, val)])

    def _solve(self):
        self.m.solve()
        s = self.m.solution.get_status()
        if s in (1, 5):
            return OPTIMAL
        if s == 3:
            return INFEASIBLE
        if s == 2:
            return UNBOUNDED
        if s == 4:  # infeasible or unbounded: settle it without the objective
            return self._settle()
        return ERROR

    def _from_scratch(self):
        # an all-slack start basis discards the stored one
        self.m.start.set_start(col_status=[], row_status=[], col_primal=[], row_primal=[], col_dual=[], row_dual=[])

    def _settle(self):
        p = self.m
        obj = p.objective.get_linear()
        p.objective.set_linear([(i, 0.0) for i in range(self.n)])
        try:
            p.solve()
            s = p.solution.get_status()
        finally:
            p.objective.set_linear(list(enumerate(obj)))
        return INFEASIBLE if s == 3 else (UNBOUNDED if s in (1, 5) else ERROR)

    def primal(self):
        return np.array(self.m.solution.get_values())

    def objval(self):
        return self.m.solution.get_objective_value()

    def redcost(self):
        return np.array(self.m.solution.get_reduced_costs())

    def rowdual(self):
        return np.array(self.m.solution.get_dual_values())

    def iterations(self):
        return self.m.solution.progress.get_num_iterations()


class _WrappedLP(_LPBase):
    """Any StrainDesign backend through its common interface (SCIP's SoPlex, GLPK)."""

    def __init__(self, c, A_ineq, b_ineq, A_eq, b_eq, lb, ub, solver):
        from straindesign.solver_interface import MILP_LP
        self.m = MILP_LP(c=list(c), A_ineq=A_ineq, b_ineq=list(b_ineq), A_eq=A_eq, b_eq=list(b_eq), lb=list(lb), ub=list(ub), solver=solver)
        self.n = len(c)
        self.x = None
        self.n_ineq = A_ineq.shape[0]
        self.solver = solver

    def chg_coef(self, eq_row, col, val):
        row = int(self.n_ineq + eq_row)
        if self.solver == SCIP:
            self.m.backend.chgCoef(int(row), int(col), float(val))
            return
        import swiglpk as g
        P = self.m.backend.glpk
        n = g.glp_get_num_cols(P)
        ind, v = g.intArray(n + 1), g.doubleArray(n + 1)
        k = g.glp_get_mat_row(P, row + 1, ind, v)
        entries = {int(ind[i]): float(v[i]) for i in range(1, k + 1)}
        entries[int(col) + 1] = float(val)
        entries = {j: a for j, a in entries.items() if a != 0.0}
        for i, (j, a) in enumerate(entries.items()):
            ind[i + 1], v[i + 1] = int(j), float(a)
        g.glp_set_mat_row(P, row + 1, len(entries), ind, v)

    def redcost(self):
        if self.solver == SCIP:
            return np.array(self.m.backend.getRedcost())
        import swiglpk as g
        P = self.m.backend.glpk
        return np.array([g.glp_get_col_dual(P, j + 1) for j in range(self.n)])

    def set_bounds(self, idx, lo, hi):
        self.m.set_lb([[int(i), float(v)] for i, v in zip(idx, lo)])
        self.m.set_ub([[int(i), float(v)] for i, v in zip(idx, hi)])

    def set_obj(self, idx, val):
        self.m.set_objective_idx([[int(i), float(v)] for i, v in zip(idx, val)])

    def _solve(self):
        self.x, self.opt, status = self.m.solve()
        if status in (OPTIMAL, INFEASIBLE, UNBOUNDED):
            return status
        return ERROR

    def primal(self):
        return np.array(self.x)

    def objval(self):
        return self.opt

    def iterations(self):
        return np.nan


class _GlpkExactLP(_WrappedLP):
    """GLPK, each solve finished in rational arithmetic: the floating-point simplex finds a basis,
    and glp_exact carries on from it without tolerances. A status is then a proven one, and a
    solution entry is zero exactly when the true vertex has a zero there. (GLPK takes an integer
    as it is and replaces any other double by a nearby fraction with a small denominator, so
    rows have to arrive scaled to integers; a threshold that is no integer is met to about ten
    digits.)"""
    exact_seconds = 0.0
    n_changed = 0  # solves in which the exact verdict differed from the floating-point one
    n_restarted = 0  # solves in which the exact simplex could not use the floating-point basis

    def __init__(self, c, A_ineq, b_ineq, A_eq, b_eq, lb, ub):
        super().__init__(c, A_ineq, b_ineq, A_eq, b_eq, lb, ub, GLPK)
        import swiglpk as g
        self.g, self.P = g, self.m.backend.glpk
        self.parm = g.glp_smcp()
        g.glp_init_smcp(self.parm)
        self.parm.msg_lev = g.GLP_MSG_OFF
        # the floating-point simplex only proposes a basis: GLPK's own tolerances, and a limit,
        # since at the tolerances of the wrapper it can stall on a degenerate LP
        self.fparm = g.glp_smcp()
        g.glp_init_smcp(self.fparm)
        self.fparm.msg_lev = g.GLP_MSG_OFF
        self.fparm.tm_lim = 2000

    def _solve(self):
        g, P = self.g, self.P
        ret = g.glp_simplex(P, self.fparm)
        st = g.glp_get_status(P)
        float_status = OPTIMAL if (ret == 0 and st == g.GLP_OPT) else INFEASIBLE if (ret == 0 and st == g.GLP_NOFEAS) else ERROR
        t0 = time.perf_counter()
        ret = g.glp_exact(P, self.parm)
        if ret != 0:
            # a basis the floating-point simplex ends on can be singular in exact arithmetic: try
            # once more from a triangular crash basis, and from the slack basis if that fails too
            _GlpkExactLP.n_restarted += 1
            out = g.glp_term_out(g.GLP_OFF)
            g.glp_adv_basis(P, 0)
            g.glp_term_out(out)
            g.glp_simplex(P, self.fparm)
            ret = g.glp_exact(P, self.parm)
            if ret != 0:
                g.glp_std_basis(P)
                ret = g.glp_exact(P, self.parm)
        _GlpkExactLP.exact_seconds += time.perf_counter() - t0
        st = g.glp_get_status(P)
        status = OPTIMAL if (ret == 0 and st == g.GLP_OPT) else INFEASIBLE if (ret == 0 and st == g.GLP_NOFEAS) else \
            UNBOUNDED if (ret == 0 and st == g.GLP_UNBND) else ERROR
        if status != float_status:
            _GlpkExactLP.n_changed += 1
        if status == OPTIMAL:
            self.x = [g.glp_get_col_prim(P, j + 1) for j in range(self.n)]
            self.opt = g.glp_get_obj_val(P)
        return status

    def _from_scratch(self):
        self.g.glp_std_basis(self.P)

    def redcost(self):
        return np.array([self.g.glp_get_col_dual(self.P, j + 1) for j in range(self.n)])


def make_lp(solver, c, A_ineq, b_ineq, A_eq, b_eq, lb, ub):
    A_ineq = sparse.csr_matrix(A_ineq)
    A_eq = sparse.csr_matrix(A_eq)
    if solver == 'glpk_exact':
        return _GlpkExactLP(c, A_ineq, b_ineq, A_eq, b_eq, lb, ub)
    if solver == GUROBI:
        return _GurobiLP(c, A_ineq, b_ineq, A_eq, b_eq, lb, ub)
    if solver == CPLEX:
        return _CplexLP(c, A_ineq, b_ineq, A_eq, b_eq, lb, ub)
    return _WrappedLP(c, A_ineq, b_ineq, A_eq, b_eq, lb, ub, solver)


# ------------------------------------------------------------------------------------------------
# bit sets as Python integers
# ------------------------------------------------------------------------------------------------
def bits(x):
    """Indices of the set bits of x, ascending."""
    while x:
        low = x & -x
        yield low.bit_length() - 1
        x ^= low


def mask(indices):
    m = 0
    for i in indices:
        m |= 1 << int(i)
    return m


class BitRows:
    """A growing list of bit sets held as 64-bit words, one array per word position, for tests
    against one set X that run over all sets at once."""

    def __init__(self, n_bits):
        self.nw = max(1, -(-n_bits // 64))
        self.a = np.zeros((self.nw, 64), np.uint64)
        self.n = 0

    def words(self, x):
        return np.frombuffer(int(x).to_bytes(self.nw * 8, 'little'), np.uint64)

    def add(self, x):
        if self.n == self.a.shape[1]:
            self.a = np.hstack((self.a, np.zeros_like(self.a)))
        self.a[:, self.n] = self.words(x)
        self.n += 1

    def _hits(self, xw):
        n = self.n
        acc = self.a[0, :n] & xw[0]
        for k in range(1, self.nw):
            acc |= self.a[k, :n] & xw[k]
        return acc

    def disjoint(self, xw):
        """Per set: no bit in common with X."""
        return self._hits(xw) == 0

    def within(self, xw):
        """Per set: a subset of X."""
        return self._hits(~xw) == 0


class BoundFacts:
    """What LPs have shown about an optimum max c over P(X): a flux of value f through the atoms
    W shows max >= f for every X that spares W, and a proof of max <= f that uses the zero bounds
    of C holds for every X that contains C. Per set only the best value is kept."""

    def __init__(self, n_atoms):
        self.n_atoms = n_atoms
        self.version = 0  # changes with every change of the facts
        self.clear_lower()
        self.clear_upper()

    def clear_lower(self):
        self.W, self.Wi, self.lo, self.row_lo = BitRows(self.n_atoms), [], np.empty(64), {}
        self.version += 1

    def clear_upper(self):
        self.C, self.Ci, self.up, self.row_up = BitRows(self.n_atoms), [], np.empty(64), {}
        self.version += 1

    def add_lower(self, W, val):
        self.version += 1
        i = self.row_lo.get(W)
        if i is not None:
            self.lo[i] = max(self.lo[i], val)
            return
        i = self.row_lo[W] = len(self.Wi)
        if i == len(self.lo):
            self.lo = np.concatenate((self.lo, np.empty_like(self.lo)))
        self.lo[i] = val
        self.W.add(W)
        self.Wi.append(W)

    def add_upper(self, C, val):
        self.version += 1
        i = self.row_up.get(C)
        if i is not None:
            self.up[i] = min(self.up[i], val)
            return
        i = self.row_up[C] = len(self.Ci)
        if i == len(self.up):
            self.up = np.concatenate((self.up, np.empty_like(self.up)))
        self.up[i] = val
        self.C.add(C)
        self.Ci.append(C)

    def bounds(self, X):
        """(best lower bound, its W, best upper bound, its C) for the blocked set X."""
        lo, W, up, C = -np.inf, 0, np.inf, 0
        xw = self.W.words(X)
        if self.Wi:
            vals = np.where(self.W.disjoint(xw), self.lo[:len(self.Wi)], -np.inf)
            i = int(vals.argmax())
            if vals[i] > -np.inf:
                lo, W = float(vals[i]), self.Wi[i]
        if self.Ci:
            vals = np.where(self.C.within(xw), self.up[:len(self.Ci)], np.inf)
            i = int(vals.argmin())
            if vals[i] < np.inf:
                up, C = float(vals[i]), self.Ci[i]
        return lo, W, up, C


class Memo:
    """Pairs (P, N): the module is fulfilled for every X with X & P == 0 and N <= X."""

    def __init__(self, n_atoms):
        self.n_atoms = n_atoms
        self.clear()

    def clear(self):
        self.P, self.N = BitRows(self.n_atoms), BitRows(self.n_atoms)

    def add(self, P, N):
        self.P.add(P)
        self.N.add(N)

    def covers(self, X):
        if not self.P.n:
            return False
        xw = self.P.words(X)
        return bool((self.P.disjoint(xw) & self.N.within(xw)).any())


# ------------------------------------------------------------------------------------------------
# LP oracle in flux space
# ------------------------------------------------------------------------------------------------
class _FluxLP:
    """An LP over the fluxes of a region, with what is set in it at the moment."""

    def __init__(self, lp):
        self.lp, self.X, self.avoid, self.thetas = lp, 0, 0, None


class Region:
    """A flux region  {v : S v = 0, lb <= v <= ub, V_ineq v <= v_ineq, V_eq v = v_eq,
    rows[k] . v >= thetas[k]}  with atoms, reactions that can be blocked (v = 0 on top of the
    bounds). The thresholds of the rows are set per query (None: no threshold).

    The LPs, all warm-started from query to query, between which only bounds, objective
    coefficients and the thresholds change:

    - the *flux LP*: a point of the region for the blocked set X that minimises the weighted flux
      through the atoms. Reversible atoms are split into a forward and a backward column.
    - the *proof LP*: the alternative (Farkas) system of the region, normalised to
      ``multipliers . rhs = -1``, minimising the multipliers of the atoms' zero bounds.
    - per row, on demand, a *value LP* that maximises the row over the region.

    Args:
        S (sparse matrix): stoichiometric matrix
        lb, ub (arrays): flux bounds
        V_ineq, v_ineq, V_eq, v_eq: the further fixed rows of the region
        atom_rxn (array of int): reaction index of each atom
        solver (str): LP backend
        rows (list of arrays): expressions over the fluxes that take a threshold per query
    """

    def __init__(self, S, lb, ub, V_ineq, v_ineq, V_eq, v_eq, atom_rxn, solver, rows=()):
        self.exact = solver == 'glpk_exact'
        # in exact arithmetic a value is part of a support or a core exactly when it is not zero
        self.supp_tol = 0.0 if self.exact else _SUPP_TOL
        self.core_tol = 0.0 if self.exact else _CORE_REL_TOL
        S = sparse.csc_matrix(S, dtype=float)
        m, n = S.shape
        lb, ub = np.array(lb, float), np.array(ub, float)
        V_ineq, V_eq = sparse.csc_matrix(V_ineq, dtype=float), sparse.csc_matrix(V_eq, dtype=float)
        v_ineq, v_eq = np.array(v_ineq, float), np.array(v_eq, float)
        self._S, self._lb, self._ub, self._V = S, lb.copy(), ub.copy(), (V_ineq, v_ineq.copy(), V_eq, v_eq.copy())
        E = sparse.csc_matrix(np.array(rows, float).reshape(len(rows), n))
        K = E.shape[0]
        self.n, self.K, self.solver = n, K, solver
        self.atom_rxn = np.array(atom_rxn, int)
        nA = len(self.atom_rxn)
        # a flux cone is scale-free: bring its right-hand sides to a size the LP tolerance does not touch
        self.is_cone = bool(np.all((lb == 0) | np.isinf(lb)) and np.all((ub == 0) | np.isinf(ub)) and not v_eq.any())
        self.scale = 1.0
        if self.is_cone and v_ineq.any() and not K and not self.exact:
            self.scale = _CONE_SCALE / np.abs(v_ineq).max()
            v_ineq = v_ineq * self.scale
        # atoms whose bounds exclude zero flux: blocking one empties the region outright
        self.forced = mask(a for a, j in enumerate(self.atom_rxn) if lb[j] > 0 or ub[j] < 0)
        # ---- flux LP: columns v, backward parts of the reversible atoms, one value column per row ----
        rev = np.array([lb[j] < 0 < ub[j] for j in self.atom_rxn], bool)
        self.col_fwd = self.atom_rxn.copy()  # column of an atom (its forward part if split)
        self.col_bwd = np.full(nA, -1)
        self.col_bwd[rev] = n + np.arange(rev.sum())
        split = self.atom_rxn[rev]
        ns = len(split)
        self.col_s = n + ns + np.arange(K)
        A_eq = sparse.vstack((S, V_eq, E), format='csc')
        tail = sparse.vstack((sparse.csc_matrix((m + V_eq.shape[0], K)), -sparse.identity(K, format='csc')), format='csc')
        A_eq = sparse.hstack((A_eq, -A_eq[:, split], tail), format='csr')
        A_ineq = sparse.hstack((V_ineq, -V_ineq[:, split], sparse.csc_matrix((V_ineq.shape[0], K))), format='csr')
        lo = np.concatenate((lb, np.zeros(ns), np.full(K, -np.inf)))
        hi = np.concatenate((ub, -lb[split], np.full(K, np.inf)))
        lo[split] = 0.0
        w = np.zeros(n + ns + K)
        # flux through a reaction that is no atom costs a little, so that witnesses carry no cycles
        w[:n] = np.where(lb >= 0, _BACKGROUND_WEIGHT, np.where(ub <= 0, -_BACKGROUND_WEIGHT, 0.0))
        self.sign = np.where(ub[self.atom_rxn] <= 0, -1.0, 1.0)  # an atom that only runs backwards
        w[self.col_fwd] = self.sign
        w[self.col_bwd[rev]] = 1.0
        self.flux_lo, self.flux_hi, self.flux_w = lo, hi, w
        self._flux_args = (A_ineq, v_ineq, A_eq, np.concatenate((np.zeros(m), v_eq, np.zeros(K))), lo, hi)
        self.flux = _FluxLP(make_lp(solver, w, *self._flux_args))
        self.value = {}  # row -> _FluxLP that maximises it
        # ---- proof LP ----
        # multipliers: y (S rows, free), y_eq (free), lam (>= 0), alpha (finite ub), beta (finite lb),
        # mu_fwd (atom: v <= 0), mu_bwd (atom: -v <= 0), beta_row (row k: -rows[k].v <= -theta_k)
        fin_ub, fin_lb = np.flatnonzero(np.isfinite(ub)), np.flatnonzero(np.isfinite(lb))
        need_f = np.flatnonzero(ub[self.atom_rxn] > 0)  # atoms whose forward flux a block removes
        need_b = np.flatnonzero(lb[self.atom_rxn] < 0)
        k_eq, k_in = V_eq.shape[0], V_ineq.shape[0]
        blocks = [
            S.T, V_eq.T, V_ineq.T,
            sparse.csr_matrix((np.ones(len(fin_ub)), (fin_ub, np.arange(len(fin_ub)))), shape=(n, len(fin_ub))),
            sparse.csr_matrix((-np.ones(len(fin_lb)), (fin_lb, np.arange(len(fin_lb)))), shape=(n, len(fin_lb))),
            sparse.csr_matrix((np.ones(len(need_f)), (self.atom_rxn[need_f], np.arange(len(need_f)))), shape=(n, len(need_f))),
            sparse.csr_matrix((-np.ones(len(need_b)), (self.atom_rxn[need_b], np.arange(len(need_b)))), shape=(n, len(need_b))),
            -E.T,
        ]
        R = sparse.hstack(blocks, format='csr')
        rhs = np.concatenate((np.zeros(m), v_eq, v_ineq, ub[fin_ub], -lb[fin_lb], np.zeros(len(need_f) + len(need_b) + K)))
        ncol = R.shape[1]
        off = m + k_eq + k_in + len(fin_ub) + len(fin_lb)
        self.mu_f = np.full(nA, -1)
        self.mu_f[need_f] = off + np.arange(len(need_f))
        self.mu_b = np.full(nA, -1)
        self.mu_b[need_b] = off + len(need_f) + np.arange(len(need_b))
        self.col_beta = off + len(need_f) + len(need_b) + np.arange(K)
        plo = np.zeros(ncol)
        plo[:m + k_eq] = -np.inf
        phi = np.full(ncol, np.inf)
        phi[off:] = 0.0  # no atom blocked, no threshold
        obj = np.zeros(ncol)
        obj[off:off + len(need_f) + len(need_b)] = 1.0
        self.proof_R, self.proof_rhs, self.proof_off = R.tolil(), rhs, off
        self.proof_R = R
        self.norm_row = n  # equality row that holds the right-hand sides
        self.proof_lp = make_lp(solver, obj, sparse.csr_matrix((0, ncol)), [], sparse.vstack((R, sparse.csr_matrix(rhs)), format='csr'),
                                np.concatenate((np.zeros(n), [-1.0])), plo, phi)
        self.proof_X = 0
        self.proof_thetas = (None,) * K
        self.n_flux = self.n_proof = self.n_value = 0

    # -- flux LPs --------------------------------------------------------------------------------
    def _atom_cols(self, atoms):
        cols = []
        for a in atoms:
            cols.append(self.col_fwd[a])
            if self.col_bwd[a] >= 0:
                cols.append(self.col_bwd[a])
        return cols

    def _set(self, f, X, avoid, thetas, weighted=True):
        """Bring a flux LP to the blocked set X, the avoid set and the thresholds."""
        lp = f.lp
        changed = X ^ f.X
        if changed:
            cols = self._atom_cols(bits(changed))
            blocked = [(X >> a) & 1 for a in bits(changed) for _ in range(1 + (self.col_bwd[a] >= 0))]
            lp.set_bounds(cols, [0.0 if b else self.flux_lo[c] for c, b in zip(cols, blocked)],
                          [0.0 if b else self.flux_hi[c] for c, b in zip(cols, blocked)])
            f.X = X
        if weighted:
            changed = avoid ^ f.avoid
            if changed:
                cols = self._atom_cols(bits(changed))
                heavy = [(avoid >> a) & 1 for a in bits(changed) for _ in range(1 + (self.col_bwd[a] >= 0))]
                lp.set_obj(cols, [self.flux_w[c] * (_AVOID_WEIGHT if h else 1.0) for c, h in zip(cols, heavy)])
                f.avoid = avoid
        thetas = tuple(thetas) if thetas is not None else (None,) * self.K
        if thetas != f.thetas and self.K:
            lp.set_bounds(self.col_s, [-np.inf if t is None else t * self.scale for t in thetas], [np.inf] * self.K)
            f.thetas = thetas

    def _support(self, x):
        flux = np.abs(x[self.col_fwd])
        has_b = self.col_bwd >= 0
        flux[has_b] += np.abs(x[self.col_bwd[has_b]])
        return mask(np.flatnonzero(flux > self.supp_tol))

    arbiter_factory = None  # builds the same region for the exact backend
    n_arbitrated = 0  # questions the floating-point LPs left open and the exact twin settled

    def _twin(self):
        if getattr(self, '_arbiter', None) is None:
            if self.exact or self.arbiter_factory is None:
                return None
            self._arbiter = self.arbiter_factory()
        return self._arbiter

    def decide(self, X, avoid=0, thetas=None, expect=FEASIBLE):
        """(FEASIBLE, atoms a point of the region uses) or (INFEAS, atoms an infeasibility proof
        uses). `expect` says which of the two LPs to try first. Where the floating-point LPs fail
        or contradict each other (a region with neither a point nor a proof), the exact twin of
        the region decides."""
        try:
            if expect == INFEAS:
                C = self.core(X, thetas)
                if C is not None:
                    return INFEAS, C
            W = self.witness(X, avoid, thetas)
            if W is not None:
                return FEASIBLE, W
            if expect != INFEAS:
                C = self.core(X, thetas)
                if C is not None:
                    return INFEAS, C
        except LPError:
            pass
        twin = self._twin()
        if twin is None:
            raise LPError('a region has neither a point nor an infeasibility proof')
        Region.n_arbitrated += 1
        W = twin.witness(X, avoid, thetas)
        if W is not None:
            return FEASIBLE, W
        C = twin.core(X, thetas)
        if C is None:
            raise LPError('a region has neither a point nor an infeasibility proof in exact arithmetic')
        self.rows_used = twin.rows_used
        return INFEAS, C

    def witness(self, X, avoid=0, thetas=None):
        """A point of the region with the atoms X blocked, as the set of atoms it uses; None if the
        region is empty. The point keeps away from the atoms `avoid` where it can."""
        if X & self.forced:
            return None
        self._set(self.flux, X, avoid, thetas)
        self.n_flux += 1
        status = self.flux.lp.solve()
        if status == INFEASIBLE:
            return None
        if status != OPTIMAL:
            raise LPError('flux LP ended with status %s' % status)
        self.last_flux = x = self.flux.lp.primal()
        return self._support(x)

    def maximize(self, k, X, thetas=None):
        """Largest value of row k over the region with X blocked.

        Returns:
            (value, atoms the maximising flux uses, blocked atoms with a non-zero reduced cost);
            (None, None, None) if the region is empty.
        """
        if X & self.forced:
            return None, None, None
        if k not in self.value:
            w = np.zeros(len(self.flux_w))
            w[self.col_s[k]] = -1.0
            self.value[k] = _FluxLP(make_lp(self.solver, w, *self._flux_args))
        f = self.value[k]
        self._set(f, X, 0, thetas, weighted=False)
        self.n_value += 1
        status = f.lp.solve()
        if status == INFEASIBLE:
            return None, None, None
        if status != OPTIMAL:
            raise LPError('value LP ended with status %s (an unbounded inner objective?)' % status)
        x = f.lp.primal()
        # the zero bound of a blocked atom carries the dual bound only if flux through the atom
        # would raise the value: a negative reduced cost of a column that can only grow (the LP
        # minimises minus the row)
        rc = f.lp.redcost()
        used = np.maximum(-self.sign * rc[self.col_fwd], 0.0)
        has_b = self.col_bwd >= 0
        used[has_b] += np.maximum(-rc[self.col_bwd[has_b]], 0.0)
        return x[self.col_s[k]] / self.scale, self._support(x), mask(np.flatnonzero(used > (0.0 if self.exact else _CORE_ABS_TOL))) & X

    # -- proof LP --------------------------------------------------------------------------------
    def core(self, X, thetas=None):
        """Atoms among X whose zero bounds an infeasibility proof of the region uses; None if there
        is no proof (the region is not empty). `self.rows_used` then says which thresholds the
        proof relies on."""
        if X & self.forced:
            low = X & self.forced
            self.rows_used = (False,) * self.K
            return low & -low
        lp = self.proof_lp
        changed = X ^ self.proof_X
        if changed:
            cols, on = [], []
            for a in bits(changed):
                for c in (self.mu_f[a], self.mu_b[a]):
                    if c >= 0:
                        cols.append(c)
                        on.append((X >> a) & 1)
            lp.set_bounds(cols, [0.0] * len(cols), [np.inf if o else 0.0 for o in on])
            self.proof_X = X
        thetas = tuple(thetas) if thetas is not None else (None,) * self.K
        if thetas != self.proof_thetas:
            for k, (t, t_old) in enumerate(zip(thetas, self.proof_thetas)):
                if t != t_old:
                    lp.chg_coef(self.norm_row, self.col_beta[k], 0.0 if t is None else -t * self.scale)
                    self.proof_rhs[self.col_beta[k]] = 0.0 if t is None else -t * self.scale
            lp.set_bounds(self.col_beta, [0.0] * self.K, [0.0 if t is None else np.inf for t in thetas])
            self.proof_thetas = thetas
        self.n_proof += 1
        status = lp.solve()
        if status == INFEASIBLE:
            return None
        if status != OPTIMAL:
            raise LPError('proof LP ended with status %s' % status)
        y = lp.primal()
        mu = np.zeros(len(self.atom_rxn))
        for cols in (self.mu_f, self.mu_b):
            ok = cols >= 0
            mu[ok] += np.abs(y[cols[ok]])
        top = max(mu.max(initial=0.0), np.abs(y[self.col_beta]).max(initial=0.0))
        # the proof has to hold up as it is: the rows combine to zero and the right-hand sides to
        # a negative number. The LP asks for -1, which a floating-point solution has to meet. The
        # exact backend meets it for the fractions GLPK reads from the doubles (a double that is
        # no integer becomes a nearby fraction with a small denominator), so for the doubles
        # themselves the sum is off by about 1e-10 times the multipliers; there the sign decides.
        res = np.abs(self.proof_R @ y).max(initial=0.0)
        if res > 1e-6 * max(1.0, np.abs(y).max()) or abs(self.proof_rhs @ y + 1.0) > (0.5 if self.exact else 1e-6):
            raise LPError('proof LP solution does not hold up (residual %.2g)' % res)
        cut = self.core_tol * max(top, 1e-300)
        self.rows_used = tuple(bool(v > cut) for v in np.abs(y[self.col_beta]))
        return mask(np.flatnonzero(mu > cut)) & X


# ------------------------------------------------------------------------------------------------
# modules: what a blocked set X has to fulfil
# ------------------------------------------------------------------------------------------------
class ClassicModule:
    """SUPPRESS (the region has to be empty) or PROTECT (it must not be)."""

    def __init__(self, kind, region):
        self.kind, self.region = kind, region
        self.memo = Memo(len(region.atom_rxn))
        self.cores = []  # SUPPRESS: the cores that empty the region

    def decided(self, X):
        return self.memo.covers(X)

    def check(self, X, avoid=0):
        """(True, None, None), or (False, P, N) with a clause: every X' that fulfils the module
        blocks an atom of P or leaves one of N unblocked."""
        verdict, A = self.region.decide(X, avoid if self.kind == SUPPRESS else 0)
        if verdict == FEASIBLE:
            if self.kind == SUPPRESS:
                return False, A, 0
            self.memo.add(A, 0)
            return True, None, None
        if self.kind == SUPPRESS:
            self.memo.add(0, A)
            self.cores.append(A)
            return True, None, None
        return False, 0, A


class NestedModule:
    """SUPPRESS or PROTECT of the part of a region that is optimal for an inner objective:

        { v in P(X) :  constraints,  c.v >= bound(g(X)) },      g(X) = max c over P(X)

    with bound(g) = tol * g - offset, less a tolerance. P(X) is the model with X blocked, without
    the module's constraints, as in the MILP formulation.

    The region has a point exactly if  b(X) >= bound(g(X)),  where b(X) is the largest c.v over
    P(X) *with* the constraints (and with the thresholds of the outer region's further rows). The
    module therefore compares two LP optima, and each of them can only fall when more is blocked.
    What an LP says about one optimum at one X carries over to other sets: a flux of value f
    through the atoms W shows optimum >= f for every X' that spares W, and a dual bound f that
    rests on the zero bounds of C <= X shows optimum <= f for every X' >= C. The facts on g and
    those on b are kept apart, and any lower bound of one combines with any upper bound of the
    other, whichever sets the LPs were solved for:

        b(X) <= u < bound(l) <= bound(g(X))    no point:  X >= C_b and X spares W_g
        b(X) >= l >= bound(u) >= bound(g(X))   a point:   X spares W_b and X >= C_g

    A set that the facts decide costs no LP. Otherwise the inner optimum is solved first, then b;
    for SUPPRESS a flux LP comes between the two: its point, kept away from the atoms the search
    wants to block, uses few atoms and makes the stronger clause.

    Args:
        kind: SUPPRESS or PROTECT
        inner (Region): P with the inner objective as row 0
        outer (Region): P with the module's constraints, inner objective as row 0, further rows after
        tol, offset (float): see above
        extra (tuple): thresholds of the outer region's further rows
    """

    def __init__(self, kind, inner, outer, tol=1.0, offset=0.0, extra=()):
        self.kind, self.inner, self.outer = kind, inner, outer
        self.tol, self.offset, self.extra = tol, offset, tuple(extra)
        self.facts = BoundFacts(len(inner.atom_rxn))  # on g
        self.bfacts = BoundFacts(len(inner.atom_rxn))  # on b, for the thresholds in force
        self.n_opt_memo = self.n_from_facts = 0
        self._seen = (None, None)
        self.enabled = True  # False: the module asks for nothing at the moment

    def bound(self, g):
        theta = self.tol * g - self.offset
        return theta - _OPT_TOL * max(1.0, abs(theta))

    def verdict(self, X):
        """What the stored facts say about the region at X: (FEASIBLE | INFEAS | None, P, N). The
        verdict holds for every X' that spares P and contains N."""
        key = (X, self.facts.version, self.bfacts.version, self.offset)
        if key != self._seen[0]:
            self._seen = (key, self._verdict(X))
        return self._seen[1]

    def _verdict(self, X):
        lo_g, W_g, up_g, C_g = self.facts.bounds(X)
        if up_g == -np.inf:  # no flux at all
            return INFEAS, 0, C_g
        lo_b, W_b, up_b, C_b = self.bfacts.bounds(X)
        if up_b == -np.inf:
            return INFEAS, 0, C_b
        if lo_g > -np.inf and up_b < self.bound(lo_g):
            return INFEAS, W_g, C_b
        if up_g < np.inf and lo_b > -np.inf and lo_b >= self.bound(up_g):
            return FEASIBLE, W_b, C_g
        return None, 0, 0

    def decided(self, X):
        if not self.enabled:
            return True
        v = self.verdict(X)[0]
        return v == (INFEAS if self.kind == SUPPRESS else FEASIBLE)

    @staticmethod
    def _value(R, X, thetas=None):
        try:
            return R.maximize(0, X, thetas)
        except LPError:
            twin = R._twin()
            if twin is None:
                raise
            Region.n_arbitrated += 1
            return twin.maximize(0, X, thetas)

    def _optimum(self, R, facts, X, thetas=None, sharp=False):
        """(value, W, C) of max c over the region R at X, from the facts where they pin it down
        (within the optimality tolerance; sharp: only where lower and upper bound meet);
        value None: the region is empty at X, and C a core of that."""
        lo, W, up, C = facts.bounds(X)
        if up == -np.inf:
            return None, None, C
        if np.isfinite(lo) and np.isfinite(up) and up - lo <= (0.0 if sharp else 2 * _OPT_TOL * max(1.0, abs(lo))):
            self.n_opt_memo += 1
            return lo, W, C
        val, W, C = self._value(R, X, thetas)
        if val is None:
            verdict, C = R.decide(X, 0, thetas, expect=INFEAS)
            if verdict == FEASIBLE:
                raise LPError('a region is infeasible for the value LP and has a point')
            facts.add_upper(C, -np.inf)
            return None, None, C
        facts.add_lower(W, val)
        facts.add_upper(C, val)
        return val, W, C

    def optimum(self, X, sharp=False):
        """(value, W, C) of the inner problem at X. value None: P(X) is empty, C its core."""
        return self._optimum(self.inner, self.facts, X, None, sharp)

    def constrained_optimum(self, X, sharp=False):
        """(value, W, C) of b at X."""
        return self._optimum(self.outer, self.bfacts, X, (None,) + self.extra, sharp)

    def _point(self, X, avoid):
        """A point of the region at X for the threshold bound(g(X)), g(X) known: recorded as a
        lower bound on b."""
        val = self.optimum(X)[0]
        if val is None:
            return
        theta = self.bound(val)
        R = self.outer
        try:
            W = R.witness(X, avoid, (theta,) + self.extra)
        except LPError:
            return  # the value LP of b decides
        if W is not None:
            # the LP holds c.v >= theta within its tolerance
            self.bfacts.add_lower(W, max(theta, R.last_flux[R.col_s[0]] / R.scale))

    def check(self, X, avoid=0):
        """(True, None, None), or (False, P, N) with a clause: every X' that fulfils the module
        blocks an atom of P or leaves one of N unblocked."""
        v, P, N = self.verdict(X)
        if v is not None:
            self.n_from_facts += 1
        elif self.kind == SUPPRESS:
            steps = (lambda: self.optimum(X), lambda: self._point(X, avoid), lambda: self.constrained_optimum(X))
        else:
            steps = (lambda: self.optimum(X), lambda: self.constrained_optimum(X))
        if v is None:
            # bounds that agree only within the tolerance can leave the comparison open: then
            # both optima are solved for X itself, which always decides it
            for step in steps + (lambda: self.optimum(X, True), lambda: self.constrained_optimum(X, True)):
                step()
                v, P, N = self.verdict(X)
                if v is not None:
                    break
            else:
                raise LPError('both optima of a nested module are known and do not decide it')
        if (v == INFEAS) == (self.kind == SUPPRESS):
            return True, None, None
        return False, P, N


def _minimal(sets):
    """The inclusion-minimal members of a collection of bit sets."""
    keep = []
    for J in sorted(set(sets), key=lambda J: (J.bit_count(), J)):
        if not any(k & J == k for k in keep):
            keep.append(J)
    return keep


def _blocking_sets(node, cost, budget):
    """The minimal sets of interventions, of cost <= budget, that make a blocked-formula true.

    node: int (an intervention) | ('or', children) | ('and', children) | False (never blocked).
    Unlike a clause form or a full list of blocking sets, this stays small whatever the shape of
    the rule, because the budget cuts every product short.
    """
    if node is False:
        return []
    if isinstance(node, (int, np.integer)):
        return [1 << int(node)] if cost[int(node)] <= budget else []
    op, children = node
    if op == 'or':  # blocked as soon as one part is
        out = set()
        for ch in children:
            out.update(_blocking_sets(ch, cost, budget))
        return _minimal(out)
    out = {0}  # and: every part has to be blocked
    for ch in children:
        part = _blocking_sets(ch, cost, budget)
        out = {a | b for a in out for b in part}
        out = _minimal(J for J in out if sum(cost[i] for i in bits(J)) <= budget)
        if not out:
            return []
    return out


class BoolLayer:
    """Interventions with costs, and for every atom the sets of interventions that block it.

    Args:
        costs (list of float): cost of each intervention
        formulas (list): per atom its blocked-formula over intervention indices, as
            int | ('or', [..]) | ('and', [..]) | False
        budget (float): the largest total cost a set of interventions may have
    """

    def __init__(self, costs, formulas, budget):
        self.cost = [float(c) for c in costs]
        self.nI, self.nA = len(costs), len(formulas)
        self.block = [_blocking_sets(f, self.cost, budget) for f in formulas]
        self.atoms_of = [0] * self.nI  # atoms that the intervention helps to block
        self.sets_with = [[] for _ in range(self.nI)]  # intervention -> (atom, blocking set) pairs
        self.single0 = [0] * self.nA  # interventions that block the atom alone
        for a, sets in enumerate(self.block):
            for B in sets:
                if not B & (B - 1):
                    self.single0[a] |= B
                for i in bits(B):
                    self.atoms_of[i] |= 1 << a
                    self.sets_with[i].append((a, B))
        self.cost_mask = {}  # cost -> interventions of that cost
        for i, c in enumerate(self.cost):
            self.cost_mask[c] = self.cost_mask.get(c, 0) | (1 << i)

    def set_cost(self, J):
        return sum(self.cost[i] for i in bits(J))

    def X(self, z):
        return mask(a for a in range(self.nA) if any(B & z == B for B in self.block[a]))

    def extend(self, z, X, J):
        """Blocked atoms of z | J, given those of z."""
        z2 = z | J
        for i in bits(J):
            for a, B in self.sets_with[i]:
                if B & z2 == B:
                    X |= 1 << a
        return X

    def singles(self, a, z):
        """Interventions that block atom a on their own, given z (bit set)."""
        out = 0
        for B in self.block[a]:
            rest = B & ~z
            if rest and not rest & (rest - 1):
                out |= rest
        return out

    def completions(self, a, z, budget):
        """Minimal sets J (disjoint from z, cost <= budget) such that z | J blocks atom a."""
        cost = self.cost
        return _minimal(J for J in (B & ~z for B in self.block[a]) if sum(cost[i] for i in bits(J)) <= budget)


# ------------------------------------------------------------------------------------------------
# the search
# ------------------------------------------------------------------------------------------------
class _Stop(Exception):
    pass


class LPEnumerator:
    """Enumerate the minimal intervention sets, cost level by cost level, that fulfil every module.

    A clause is a pair (P, N) of atom sets: an acceptable blocked set X has an atom of P or lacks
    one of N. X *violates* it if ``X & P == 0`` and ``N <= X``; a set above X can repair that only
    by blocking an atom of P. Clauses with empty N are the witnesses of SUPPRESS regions, clauses
    with empty P the cores of PROTECT regions; modules with an inner objective give both parts.

    Args:
        layer (BoolLayer)
        modules (list): ClassicModule / NestedModule, all over the same atoms
        max_cost (float)
        targeted (bool): at a node that takes one more intervention, first ask for a witness that
            keeps away from the atoms the candidates would block
        max_solutions (float): stop after this many designs
    """

    def __init__(self, layer, modules, max_cost, targeted=True, time_limit=np.inf, max_solutions=np.inf, probe_rounds=1):
        self.layer = layer
        # modules whose violation says what to block come first: their clauses steer the search
        self.modules = sorted(modules, key=lambda mod: mod.kind != SUPPRESS)
        self.max_cost, self.targeted, self.max_solutions = max_cost, targeted, max_solutions
        self.probe_rounds = int(os.environ.get('SD_LP_PROBES', probe_rounds))
        self.steer = os.environ.get('SD_LP_STEER', 'later')
        self.hot = 0
        self.deadline = time.time() + time_limit
        self.P, self.Nn = [], []  # clauses with a positive part (Nn: its negative part, 0 if none)
        self.G = []  # per clause: the interventions that block an atom of P on their own, from scratch
        self.N = []  # clauses without positive part: do not block all of these atoms
        self.has_mixed = False
        self.designs = []
        self.with_member = {}  # intervention -> designs of lower cost levels that contain it
        self.banned0 = 0
        self.min_cost = min(layer.cost) if layer.cost else np.inf
        self._affordable = {}
        self.stat = dict(nodes=0, queries=0, memo=0, targeted=0, levels=[])

    # -- oracle with memory ----------------------------------------------------------------------
    def _add_clause(self, P, N):
        if not P:
            if not N:
                stop = _Stop('a module cannot be fulfilled by any set of interventions')
                stop.complete = True  # there is no design, and that is the whole answer
                raise stop
            self.N.append(N)
            return
        self.P.append(P)
        self.Nn.append(N)
        self.has_mixed |= bool(N)
        g = 0
        single0 = self.layer.single0
        for a in bits(P):
            g |= single0[a]
        self.G.append(g)

    def _is_design(self, X, avoid=0):
        """Decide the blocked set X. False comes with a new clause that X violates."""
        for mod in self.modules:
            if mod.decided(X):
                self.stat['memo'] += 1
                continue
            self.stat['queries'] += 1
            ok, P, N = mod.check(X, avoid)
            if not ok:
                self._add_clause(P, N)
                return False
        return True

    def _ban(self, z2, J, banned):
        """For z2 = z | J: None if z2 contains a design of a lower level, else the interventions
        whose addition to z2 would complete one (those of z, `banned`, and the new ones)."""
        for i in bits(J):
            for d in self.with_member.get(i, ()):
                rest = d & ~z2
                if not rest:
                    return None
                if not rest & (rest - 1):
                    banned |= rest
        return banned

    def _record(self, z):
        self.designs.append(z)
        if len(self.designs) >= self.max_solutions:
            raise _Stop('solution limit')

    def _index_designs(self, start):
        """Make the designs found so far (from position start) count for the levels to come. A
        design of the level in work cannot lie inside another set of that level."""
        for d in self.designs[start:]:
            if not d & (d - 1):
                self.banned0 |= d  # a design of one intervention bans it everywhere
            for i in bits(d):
                self.with_member.setdefault(i, []).append(d)

    # -- depth-first walk of one cost level ------------------------------------------------------
    def _afford(self, left):
        """Interventions that cost at most `left`."""
        m = self._affordable.get(left)
        if m is None:
            m = self._affordable[left] = mask(i for i, c in enumerate(self.layer.cost) if c <= left)
        return m

    def _descend(self, z, X, cost, unhit, n_seen, barred, forbid, touched=0, banned=0):
        """z: chosen interventions, X: atoms they block, unhit: clauses among the first n_seen whose
        positive part X misses, barred: interventions an earlier branch covers, forbid: sets of
        several interventions an earlier branch covers, touched: atoms that an intervention of z
        helps to block, banned: interventions that would complete a known design."""
        self.stat['nodes'] += 1
        P, Nn, layer = self.P, self.Nn, self.layer
        while True:
            if len(P) > n_seen:
                unhit = unhit + [i for i in range(n_seen, len(P)) if not P[i] & X]
                n_seen = len(P)
            violated = [i for i in unhit if Nn[i] & X == Nn[i]] if self.has_mixed else unhit
            if violated:
                break
            n_neg = len(self.N)
            if self._is_design(X):
                if cost != self.level:
                    raise LPError('a design below the cost level in work was not found at its own level')
                self._record(z)
                return
            if len(self.N) > n_neg:  # X breaks a clause that no set above it can repair
                return
        left = self.level - cost
        if left <= 0:
            return
        if time.time() > self.deadline:
            raise _Stop('time limit')
        if left < 2 * self.min_cost:
            return self._last(z, X, left, unhit, violated, barred, forbid, touched, banned)
        # branch over the ways to repair the violated clause with the fewest atoms: single
        # interventions as one bit set, sets of several (an atom that needs more than one) as a list
        c = min((P[i] for i in violated), key=int.bit_count)
        singles, several = 0, set()
        for a in bits(c):
            for B in layer.block[a]:
                J = B & ~z
                if not J & (J - 1):
                    singles |= J
                else:
                    several.add(J)
        singles &= self._afford(left)
        N, cost_of = self.N, layer.cost
        for g in bits(singles & ~barred & ~banned):
            J = 1 << g
            z2 = z | J
            b2 = self._ban(z2, J, banned)
            if b2 is not None and not any(F & z2 == F for F in forbid):
                X2 = layer.extend(z, X, J)
                if not N or not any(C & X2 == C for C in N):
                    self._descend(z2, X2, cost + cost_of[g], [i for i in unhit if not P[i] & X2], n_seen, barred, forbid,
                                  touched | layer.atoms_of[g], b2)
            barred |= J
        barred |= singles
        if several:
            for J in _minimal(J for J in several if not J & (barred | banned) and layer.set_cost(J) <= left):
                z2 = z | J
                b2 = self._ban(z2, J, banned)
                if b2 is not None and not any(F & z2 == F for F in forbid):
                    X2 = layer.extend(z, X, J)
                    if not N or not any(C & X2 == C for C in N):
                        t2 = touched
                        for i in bits(J):
                            t2 |= layer.atoms_of[i]
                        self._descend(z2, X2, cost + layer.set_cost(J), [i for i in unhit if not P[i] & X2], n_seen, barred, forbid, t2, b2)
                forbid = forbid + [J]

    def _last(self, z, X, left, unhit, violated, barred, forbid, touched, banned):
        """One more intervention fits: it has to repair every violated clause on its own."""
        P, Nn, G, N, layer = self.P, self.Nn, self.G, self.N, self.layer
        for F in forbid:
            rest = F & ~z
            if not rest & (rest - 1):
                banned |= rest
        cand = layer.cost_mask.get(left, 0) & ~z & ~banned & ~barred
        open_atoms = touched & ~X  # atoms that z has brought closer to being blocked
        for i in violated:
            g = G[i]
            near = P[i] & open_atoms
            if near:
                for a in bits(near):
                    g |= layer.singles(a, z)
            cand &= g
            if not cand:
                return
        pending = [i for i in unhit if Nn[i] & X != Nn[i]] if self.has_mixed else ()
        kept = []
        for g in bits(cand):
            X2 = layer.extend(z, X, 1 << g)
            if N and any(C & X2 == C for C in N):
                continue
            if pending and any(not P[i] & X2 and Nn[i] & X2 == Nn[i] for i in pending):
                continue  # the intervention completes the negative part of a clause it does not repair
            kept.append((g, X2))
        if not kept:
            return
        n0, n1 = len(P), len(N)
        rounds = self.probe_rounds if self.targeted else 0
        steer = self.steer
        first = kept
        first_pos = {g: k for k, (g, _) in enumerate(first)}
        delta_all = 0  # what any candidate of this node would block
        for g, X2 in kept:
            delta_all |= X2 & ~X
        self.hot |= delta_all
        while rounds and len(kept) > 1:
            # a query on X itself, its witness steered away from what the candidates would block:
            # the clause it returns rules out every candidate that does not repair it. Repeated
            # while it rules candidates out, each time steered away from those that are left.
            rounds -= 1
            avoid = 0
            for g, X2 in kept:
                avoid |= X2 & ~X
            if steer == 'hot':
                avoid = self.hot & ~X
            n_before, n_clauses = len(kept), len(P)
            for mod in self.modules:
                if mod.kind != SUPPRESS or mod.decided(X):
                    continue
                self.stat['queries'] += 1
                self.stat['targeted'] += 1
                ok, Pc, Nc = mod.check(X, avoid)
                if not ok:
                    self._add_clause(Pc, Nc)
                    break
            kept = [(g, X2) for g, X2 in kept if not any(not P[i] & X2 and Nn[i] & X2 == Nn[i] for i in range(n_clauses, len(P)))]
            if len(kept) == n_before:
                break
        for k, (g, X2) in enumerate(kept):
            if any(not P[i] & X2 and Nn[i] & X2 == Nn[i] for i in range(n0, len(P))) or any(C & X2 == C for C in N[n1:]):
                continue
            avoid = 0
            if self.targeted:
                if steer == 'later':  # the candidates that follow in the list, ruled out or not
                    for g3, X3 in first[first_pos[g] + 1:]:
                        avoid |= X3 & ~X2
                elif steer == 'rest':  # the candidates still to come
                    for g3, X3 in kept[k + 1:]:
                        avoid |= X3 & ~X2
                elif steer == 'node':  # every candidate of this node, ruled out or not
                    avoid = delta_all & ~X2
                elif steer == 'hot':  # every atom a candidate of any node so far would have blocked
                    avoid = self.hot & ~X2
            if self._is_design(X2, avoid):
                self._record(z | (1 << g))

    def run(self):
        """Returns (designs as bit sets over the interventions, complete). A further call starts
        over with the clauses kept: what a module's violation taught stays true, the designs
        (which depend on what the modules ask for at the time) are searched afresh."""
        t0 = time.time()
        complete = True
        self.designs, self.with_member, self.banned0 = [], {}, 0
        try:
            if self._is_design(0):
                return [0], True
            n_d0 = 0
            for level in range(1, int(np.floor(self.max_cost)) + 1):
                self.level = float(level)
                n_d, n_q, t1 = len(self.designs), self.stat['queries'], time.time()
                self._index_designs(n_d0)
                n_d0 = len(self.designs)
                self._descend(0, 0, 0.0, [i for i in range(len(self.P))], len(self.P), 0, [], 0, self.banned0)
                self.stat['levels'].append((level, len(self.designs) - n_d, self.stat['queries'] - n_q, time.time() - t1))
                logging.info('  LP enumeration, cost %g: %d designs, %d queries, %.1f s, %d + %d clauses' %
                             (level, len(self.designs) - n_d, self.stat['queries'] - n_q, time.time() - t1, len(self.P), len(self.N)))
        except _Stop as e:
            logging.info('  LP enumeration stopped: %s' % e)
            complete = getattr(e, 'complete', False)
        self.stat['seconds'] = self.stat.get('seconds', 0.0) + time.time() - t0
        return self.designs, complete


# ------------------------------------------------------------------------------------------------
# set-up from a model
# ------------------------------------------------------------------------------------------------
def gpr_blocked_formula(rule, index):
    """Blocked-formula of a reaction from its GPR rule.

    Args:
        rule (str): gene-protein-reaction rule
        index (dict): gene identifier -> intervention index, for the genes that can be knocked out

    Returns:
        int | ('or', [..]) | ('and', [..]) | False (the reaction cannot be blocked)
    """
    from straindesign.compression import _gpr_parse

    def conv(node):
        if node[0] == 'VAR':
            return index.get(node[1], False)
        ch = [conv(c) for c in node[1]]
        if node[0] == 'AND':  # the reaction needs every part: it is blocked as soon as one is
            ch = [c for c in ch if c is not False]
            if not ch:
                return False
            return ch[0] if len(ch) == 1 else ('or', ch)
        if any(c is False for c in ch):  # alternatives: blocked only if all of them are
            return False
        return ch[0] if len(ch) == 1 else ('and', ch)

    return conv(_gpr_parse(rule))


def _atoms_and_layer(model, ko_cost, gko_cost, max_cost, gene_key):
    ko_cost, gko_cost = dict(ko_cost or {}), dict(gko_cost or {})
    names = list(gko_cost) + list(ko_cost)
    costs = [gko_cost[g] for g in gko_cost] + [ko_cost[r] for r in ko_cost]
    index = {g: i for i, g in enumerate(gko_cost)}
    r_index = {r: len(gko_cost) + i for i, r in enumerate(ko_cost)}
    if gene_key != 'id':
        index = {g.id: index[getattr(g, gene_key)] for g in model.genes if getattr(g, gene_key) in index}
    formulas, atom_rxn = [], []
    for j, r in enumerate(model.reactions):
        # a reaction is blocked by its own knockout or by the genes of its rule, whichever applies
        # (both for a reaction that is a target itself and carries a rule, as a target without
        # rule does once the compression has lumped it with a reaction that has one)
        f = r_index.get(r.id, False)
        if gko_cost and r.gene_reaction_rule:
            by_genes = gpr_blocked_formula(r.gene_reaction_rule, index)
            if f is False:
                f = by_genes
            elif by_genes is not False:
                f = ('or', [f, by_genes])
        if f is not False:
            formulas.append(f)
            atom_rxn.append(j)
    layer = BoolLayer(costs, formulas, max_cost)
    # a reaction that no affordable set of interventions blocks is no atom
    ok = [a for a in range(layer.nA) if layer.block[a]]
    if len(ok) < layer.nA:
        formulas, atom_rxn = [formulas[a] for a in ok], [atom_rxn[a] for a in ok]
        layer = BoolLayer(costs, formulas, max_cost)
    return names, layer, atom_rxn


def _objective(module, key, sense_key, rxn_ids):
    """Objective row of a module as a dense array, oriented for maximisation."""
    from straindesign.parse_constr import linexprdict2mat
    c = np.asarray(linexprdict2mat(module[key], rxn_ids).todense(), float).ravel()
    if module[sense_key] == MINIMIZE:
        c = -c
    return c


def _integer_stoichiometry(model):
    """The stoichiometric matrix with every row scaled to integers, from the rational
    coefficients the compression leaves in the model (floats are read as the short fractions they
    were written as). Scaling a row of S v = 0 changes nothing, and integers below 2**53 are
    doubles without error, which is what an exact LP on double input needs."""
    from fractions import Fraction
    from math import lcm, gcd
    from straindesign.compression import float_to_fraction
    rows = {}
    for j, r in enumerate(model.reactions):
        for met, coeff in r._metabolites.items():
            f = coeff if isinstance(coeff, Fraction) else float_to_fraction(float(coeff))
            rows.setdefault(met, []).append((j, f))
    ri, ci, vv = [], [], []
    for i, met in enumerate(model.metabolites):
        entries = rows.get(met, [])
        if not entries:
            continue
        L = lcm(*(f.denominator for _, f in entries))
        ints = [int(f * L) for _, f in entries]
        d = gcd(*ints) or 1
        for (j, _), a in zip(entries, ints):
            if abs(a // d) >= 2**53:
                raise NotImplementedError('a stoichiometric row does not fit into exact doubles')
            ri.append(i)
            ci.append(j)
            vv.append(float(a // d))
    return sparse.csr_matrix((vv, (ri, ci)), shape=(len(model.metabolites), len(model.reactions)))


def _integer_rows(V_ineq, v_ineq, V_eq, v_eq):
    """Module constraints with every row, right-hand side included, scaled to integers."""
    from fractions import Fraction
    from math import lcm
    out = []
    for V, v in ((V_ineq, v_ineq), (V_eq, v_eq)):
        V = sparse.csr_matrix(V, dtype=float)
        rows, rhs = [], []
        for i in range(V.shape[0]):
            row = V.getrow(i)
            fr = [Fraction(float(a)).limit_denominator(10**12) for a in row.data]
            b = Fraction(float(v[i])).limit_denominator(10**12)
            L = lcm(*(f.denominator for f in fr), b.denominator)
            rows.append(sparse.csr_matrix(([float(f * L) for f in fr], (np.zeros(len(fr), int), row.indices)), shape=(1, V.shape[1])))
            rhs.append(float(b * L))
        out += [sparse.vstack(rows, format='csr') if rows else sparse.csr_matrix((0, V.shape[1])), rhs]
    return tuple(out)


def _with_twin(ctx, region, V, rows=()):
    if ctx.get('twin'):
        region.arbiter_factory = ctx['twin'](V, rows)


def build_problem(model, sd_modules, ko_cost, gko_cost, max_cost, solver, gene_key='id', exact=False):
    """Set up the three layers for a model: interventions and their Boolean map onto the knockable
    reactions, and one module oracle per SUPPRESS / PROTECT module.

    Args:
        model (cobra.Model): the model the LPs run on (compressed or not); the GPR rules of its
            reactions say how genes block them
        sd_modules (list of SDModule): SUPPRESS / PROTECT modules, with or without inner objective
        ko_cost (dict): reaction id -> knockout cost
        gko_cost (dict): gene identifier -> knockout cost
        max_cost (float)
        solver (str)
        gene_key (str): 'id' or 'name', what gko_cost calls a gene

    Returns:
        (names of the interventions, BoolLayer, module oracles, atom_rxn, context for further regions)
    """
    from cobra.util.array import create_stoichiometric_matrix
    from straindesign.parse_constr import lineqlist2mat
    rxn_ids = model.reactions.list_attr('id')
    n = len(rxn_ids)
    if exact:
        solver = 'glpk_exact'
        S = _integer_stoichiometry(model)
    else:
        S = sparse.csr_matrix(np.array(create_stoichiometric_matrix(model), dtype=float))
    lb = np.array([float(r.lower_bound) for r in model.reactions])
    ub = np.array([float(r.upper_bound) for r in model.reactions])
    names, layer, atom_rxn = _atoms_and_layer(model, ko_cost, gko_cost, max_cost, gene_key)
    none = sparse.csr_matrix((0, n))
    if exact:
        lineqlist2mat = lambda constr, ids, _f=lineqlist2mat: _integer_rows(*_f(constr, ids))
    ctx = dict(S=S, lb=lb, ub=ub, rxn_ids=rxn_ids, atom_rxn=atom_rxn, solver=solver, none=none, exact=exact)
    if not exact:
        # every region can be rebuilt for the exact backend, on integer rows, should its LPs fail
        try:
            S_int = _integer_stoichiometry(model)
            ctx['twin'] = lambda V, rows=(): (lambda: Region(S_int, lb, ub, *_integer_rows(*V), atom_rxn, 'glpk_exact', rows=rows))
        except NotImplementedError:
            ctx['twin'] = None
    modules = []
    for mod in sd_modules:
        if mod[MODULE_TYPE] not in (SUPPRESS, PROTECT):
            continue
        if mod[OUTER_OBJECTIVE] is not None:
            raise NotImplementedError('the LP engine does not handle SUPPRESS / PROTECT modules with an outer objective yet')
        V_ineq, v_ineq, V_eq, v_eq = lineqlist2mat(mod[CONSTRAINTS], rxn_ids)
        V = (V_ineq, v_ineq, V_eq, v_eq)
        if mod[INNER_OBJECTIVE] is None:
            R = Region(S, lb, ub, *V, atom_rxn, solver)
            _with_twin(ctx, R, V)
            modules.append(ClassicModule(mod[MODULE_TYPE], R))
            continue
        c = _objective(mod, INNER_OBJECTIVE, INNER_OPT_SENSE, rxn_ids)
        inner = Region(S, lb, ub, none, [], none, [], atom_rxn, solver, rows=[c])
        outer = Region(S, lb, ub, *V, atom_rxn, solver, rows=[c])
        _with_twin(ctx, inner, (none, [], none, []), [c])
        _with_twin(ctx, outer, V, [c])
        tol = mod[INNER_OPT_TOL] if mod[INNER_OPT_TOL] is not None else 1.0
        modules.append(NestedModule(mod[MODULE_TYPE], inner, outer, tol=tol))
    return names, layer, modules, atom_rxn, ctx


# ------------------------------------------------------------------------------------------------
# optimisation problems: a threshold on the objective turns each into modules, and the threshold rises
# ------------------------------------------------------------------------------------------------
# a design has to beat the best value so far by this (relative) to count as better
_IMPROVE = 1e-6


class _Objective:
    """An objective over blocked sets, expressed through modules that ask for "value above a
    threshold". `modules` are appended to the problem's own; set_threshold moves the threshold
    and drops the facts that do not hold at the new one. The threshold only rises, and a clause
    found at a lower threshold holds at every higher one."""


class OptKnockObjective(_Objective):
    """max over X of  max { o.v : v in P(X), constraints, c.v >= tol * max c over P(X) }."""

    def __init__(self, ctx, mod, V):
        c = _objective(mod, INNER_OBJECTIVE, INNER_OPT_SENSE, ctx['rxn_ids'])
        o = _objective(mod, OUTER_OBJECTIVE, OUTER_OPT_SENSE, ctx['rxn_ids'])
        S, lb, ub, at, sv, none = (ctx[k] for k in ('S', 'lb', 'ub', 'atom_rxn', 'solver', 'none'))
        self.inner = Region(S, lb, ub, none, [], none, [], at, sv, rows=[c])
        self.outer = Region(S, lb, ub, *V, at, sv, rows=[c, o])
        tol = mod[INNER_OPT_TOL] if mod[INNER_OPT_TOL] is not None else 1.0
        self.mod = NestedModule(PROTECT, self.inner, self.outer, tol=tol, extra=(None,))
        self.modules = [self.mod]

    def set_threshold(self, t):
        self.mod.extra = (t,)
        # a higher threshold on the outer objective can only lower b: its upper bounds stand
        self.mod.bfacts.clear_lower()

    def value(self, X):
        val, _, _ = self.mod.optimum(X)
        if val is None:
            return None
        theta = self.mod.tol * val
        f, _, _ = self.outer.maximize(1, X, (theta - _OPT_TOL * max(1.0, abs(theta)), None))
        return f


class RobustKnockObjective(_Objective):
    """max over X of  min { o.v : v in P(X), c.v >= max c over P(X) },  with a flux of P(X) that
    meets the module's constraints."""

    def __init__(self, ctx, mod, V):
        c = _objective(mod, INNER_OBJECTIVE, INNER_OPT_SENSE, ctx['rxn_ids'])
        o = _objective(mod, OUTER_OBJECTIVE, OUTER_OPT_SENSE, ctx['rxn_ids'])
        S, lb, ub, at, sv, none = (ctx[k] for k in ('S', 'lb', 'ub', 'atom_rxn', 'solver', 'none'))
        self.inner = Region(S, lb, ub, none, [], none, [], at, sv, rows=[c])
        self.outer = Region(S, lb, ub, none, [], none, [], at, sv, rows=[c, -o])
        self.mod = NestedModule(SUPPRESS, self.inner, self.outer, extra=(None,))
        self.feasible = ClassicModule(PROTECT, Region(S, lb, ub, *V, at, sv))
        self.modules = [self.mod, self.feasible]
        self.t = None

    def set_threshold(self, t):
        self.t = t
        # the worst growth-optimal flux has to lie above t: no optimal flux with o.v <= t
        self.mod.enabled = t is not None
        self.mod.extra = (None if t is None else -t,)
        # a higher t admits more optimal fluxes and can only raise b: its lower bounds stand
        self.mod.bfacts.clear_upper()

    def value(self, X):
        val, _, _ = self.mod.optimum(X)
        if val is None:
            return None
        f, _, _ = self.outer.maximize(1, X, (val - _OPT_TOL * max(1.0, abs(val)), None))
        return None if f is None else -f


class OptCoupleObjective(_Objective):
    """max over X of the growth-coupling potential: max c over P(X) minus max c over P(X) without
    production, with a growth-optimal flux that meets the module's constraints."""

    def __init__(self, ctx, mod, V):
        from straindesign.parse_constr import linexprdict2mat
        c = _objective(mod, INNER_OBJECTIVE, INNER_OPT_SENSE, ctx['rxn_ids'])
        prod = sparse.csr_matrix(linexprdict2mat(mod[PROD_ID], ctx['rxn_ids']))
        S, lb, ub, at, sv, none = (ctx[k] for k in ('S', 'lb', 'ub', 'atom_rxn', 'solver', 'none'))
        self.inner = Region(S, lb, ub, none, [], none, [], at, sv, rows=[c])
        self.noprod = Region(S, lb, ub, none, [], prod, [0.0], at, sv, rows=[c])
        self.mod = NestedModule(SUPPRESS, self.inner, self.noprod)
        self.with_c = NestedModule(PROTECT, self.inner, Region(S, lb, ub, *V, at, sv, rows=[c]))
        self.with_c.facts = self.mod.facts  # one inner problem, one set of facts
        self.has_noprod = ClassicModule(PROTECT, self.noprod_plain(ctx, prod))
        self.modules = [self.mod, self.with_c, self.has_noprod]
        self.min_gcp = mod[MIN_GCP] if MIN_GCP in mod and mod[MIN_GCP] is not None else None

    @staticmethod
    def noprod_plain(ctx, prod):
        S, lb, ub, at, sv, none = (ctx[k] for k in ('S', 'lb', 'ub', 'atom_rxn', 'solver', 'none'))
        return Region(S, lb, ub, none, [], prod, [0.0], at, sv)

    def set_threshold(self, t):
        # growth without production has to stay below (optimum - t): no such flux at or above it
        self.mod.enabled = t is not None
        self.mod.offset = 0.0 if t is None else t

    def value(self, X):
        val, _, _ = self.mod.optimum(X)
        if val is None:
            return None
        g0, _, _ = self.noprod.maximize(0, X)
        return None if g0 is None else val - g0


_OBJECTIVES = {OPTKNOCK: OptKnockObjective, ROBUSTKNOCK: RobustKnockObjective, OPTCOUPLE: OptCoupleObjective}


def lp_enumerate_model(model, sd_modules, ko_cost=None, gko_cost=None, max_cost=3, solver=None, gene_key='id', **kwargs):
    """Enumerate knockout designs for MCS modules on the model as it is (no preprocessing).

    Returns:
        (designs as list of dict {identifier: -1.0}, complete, LPEnumerator)
    """
    names, layer, modules, atom_rxn, _ = build_problem(model, sd_modules, ko_cost, gko_cost, max_cost, solver, gene_key)
    enum = LPEnumerator(layer, modules, max_cost, **kwargs)
    found, complete = enum.run()
    enum.names, enum.atom_rxn = names, atom_rxn
    return [{names[i]: -1.0 for i in bits(z)} for z in found], complete, enum


class CompressedDesigns:
    """What compute_strain_designs' decompression expects of a solver result."""

    def __init__(self, sd, status, setup):
        self.sd, self.status, self.sd_setup = sd, status, setup

    def get_reaction_sd_mark_no_ki(self):
        return self.sd


def lp_counts(modules):
    """(flux LPs, proof LPs, value LPs) solved by the regions of the modules so far."""
    regions = {id(R): R for mod in modules for R in (getattr(mod, 'region', None), getattr(mod, 'inner', None), getattr(mod, 'outer', None)) if R}
    return tuple(sum(getattr(R, k) for R in regions.values()) for k in ('n_flux', 'n_proof', 'n_value'))


def lp_engine(cmp_model, sd_modules, ko_cost, ki_cost, gko_cost, gki_cost, has_gene_names, max_cost, solver, time_limit=np.inf,
              max_solutions=np.inf, exact=False, **kwargs):
    """Entry point for compute_strain_designs: enumerate on the compressed model, genes kept out
    of it.

    Returns:
        (CompressedDesigns, LPEnumerator)
    """
    if ki_cost or gki_cost:
        raise NotImplementedError('the LP engine does not handle knock-ins yet')
    costs = list(ko_cost.values()) + list((gko_cost or {}).values())
    if not np.isfinite(max_cost) or any(c <= 0 or abs(c - round(c)) > 1e-9 for c in costs):
        raise NotImplementedError('the LP engine needs a finite budget and positive integer intervention costs')
    opt_mods = [mod for mod in sd_modules if mod[MODULE_TYPE] in _OBJECTIVES]
    if any(mod[MODULE_TYPE] not in (SUPPRESS, PROTECT) and mod[MODULE_TYPE] not in _OBJECTIVES for mod in sd_modules) or len(opt_mods) > 1:
        raise NotImplementedError('the LP engine handles SUPPRESS, PROTECT, OptKnock, RobustKnock and OptCouple modules')
    from straindesign.parse_constr import lineqlist2mat
    t0 = time.time()
    names, layer, modules, atom_rxn, ctx = build_problem(cmp_model, sd_modules, ko_cost, gko_cost, max_cost, solver,
                                                         'name' if has_gene_names else 'id', exact=exact)
    objective = None
    if opt_mods:
        mod = opt_mods[0]
        V = lineqlist2mat(mod[CONSTRAINTS], ctx['rxn_ids'])
        objective = _OBJECTIVES[mod[MODULE_TYPE]](ctx, mod, _integer_rows(*V) if exact else V)
        modules = modules + objective.modules
    logging.info('  LP engine: %d interventions over %d knockable reactions of %d; %d modules (set up in %.1f s).' %
                 (layer.nI, layer.nA, len(cmp_model.reactions), len(modules), time.time() - t0))
    if objective is None:
        enum = LPEnumerator(layer, modules, max_cost, time_limit=time_limit, max_solutions=max_solutions, **kwargs)
        found, complete = enum.run()
    else:
        # the best design: ask for any design, then for one that beats it, until there is none.
        # Each pass walks the cost levels upwards and stops at its first design, so the last one
        # found is also the cheapest of the best.
        enum = LPEnumerator(layer, modules, max_cost, time_limit=time_limit, max_solutions=1, **kwargs)
        best, trace = None, []
        objective.set_threshold(getattr(objective, 'min_gcp', None))
        while True:
            found, complete = enum.run()
            if not found:
                break
            z = found[0]
            val = objective.value(layer.X(z))
            trace.append((layer.set_cost(z), val))
            logging.info('  LP engine: design of cost %g with objective value %.6g' % trace[-1])
            if val is None or (best is not None and val <= best[1]):
                raise LPError('a design that should beat the best one so far does not')
            best = (z, val)
            objective.set_threshold(val + _IMPROVE * max(1.0, abs(val)))
        found = [best[0]] if best is not None else []
        enum.objective_value, enum.trace = (best[1] if best else None), trace
    enum.names, enum.atom_rxn = names, atom_rxn
    sd = [{names[i]: -1.0 for i in bits(z)} for z in found]
    if complete:
        status = OPTIMAL if sd else INFEASIBLE
    else:
        status = TIME_LIMIT_W_SOL if sd else TIME_LIMIT
    setup = {MODEL_ID: cmp_model.id, MAX_COST: max_cost, SOLVER: solver, SOLUTION_APPROACH: POPULATE, KOCOST: dict(ko_cost), KICOST: {},
             MODULES: sd_modules, TIME_LIMIT: time_limit, MAX_SOLUTIONS: np.inf}
    enum.stat['lp_flux'], enum.stat['lp_proof'], enum.stat['lp_value'] = lp_counts(modules)
    logging.info('  LP engine: %d designs; LPs: %d flux, %d proof, %d value; %d module checks answered from memory; %.1f s (%.1f s in LPs).' %
                 (len(sd), enum.stat['lp_flux'], enum.stat['lp_proof'], enum.stat['lp_value'], enum.stat['memo'],
                  enum.stat.get('seconds', np.nan), _LPBase.seconds))
    return CompressedDesigns(sd, status, setup), enum
