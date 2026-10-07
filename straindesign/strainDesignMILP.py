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
#
"""Classes and function for the solution of strain design MILPs"""

import numpy as np
from scipy import sparse
from math import isinf
import time
from typing import Dict, List, Tuple
from straindesign import SDProblem, SDSolutions, MILP_LP, SDModule, Model
from straindesign.names import *
from straindesign.solver_interface import _CNA_GATE_BINARIES
import logging

# Farkas anchor the cost-level sweep starts from on CPLEX and Gurobi (see enumerate_ksweep).
_ANCHOR_START = 1e-5
# Solvers on which the dual tilt is divided by the anchor, so its weight in the objective stays
# the size it has at an anchor of 1 while the certificates shrink with the anchor.
_TILT_SCALES_WITH_ANCHOR = {GUROBI}
# Smallest work cap per populate in the cost-level sweep, in the solver's deterministic work unit
# (CPLEX ticks, Gurobi work units); both are about 10 s of an 8-thread solve.
_RESTART_MIN_WORK = {CPLEX: 7000.0, GUROBI: 15.0}


def _drop_non_minimal(sols):
    """Remove every design that strictly contains another design in the same result.

    An ascending enumeration emits designs of cost k only after excluding every superset of the
    designs it found at lower cost, so a design containing a smaller one can appear only when
    that smaller design was NOT found: the solver certified a level exhausted with a design
    missing. The supersets are valid cut sets but not minimal ones, and their presence is the
    proof that the result is incomplete. Returns the filtered rows and how many were dropped.
    """
    sets = [frozenset(sols[i].indices.tolist()) for i in range(sols.shape[0])]
    keep = [i for i, a in enumerate(sets) if not any(b < a for b in sets)]
    return sols[keep], sols.shape[0] - len(keep)


def _solve_verification_lp(lp, values=False):
    """Solve a verification LP: whether it is feasible, or with values=True its solution (None if
    it is not shown feasible).

    The certificate systems are scale-free, and on compressed networks their feasible points have
    entries of 1e5 and more, so at the 1e-9 tolerance the MILP runs at, round-off alone can make a
    feasible system read infeasible. CPLEX and Gurobi therefore verify at the solver default of
    1e-6 and confirm any other verdict with barrier before it counts. Only a definite status shows
    feasibility: CPLEX 1/5/6 (optimal, also with unscaled infeasibilities), Gurobi 2/13 (optimal,
    or suboptimal with a point); INF_OR_UNBD, NUMERIC or an aborted solve show nothing.
    """
    if lp.solver not in (CPLEX, GUROBI):
        if not values:
            return not np.isnan(lp.slim_solve())
        x, _, status = lp.solve()
        return x if status == OPTIMAL else None
    b = lp.backend
    if lp.solver == GUROBI:
        b.params.FeasibilityTol = 1e-6
        b.params.OptimalityTol = 1e-6
    else:
        b.parameters.simplex.tolerances.feasibility.set(1e-6)
        b.parameters.simplex.tolerances.optimality.set(1e-6)
    for confirm in (False, True):
        if confirm:
            lp.set_lp_method(LP_METHOD_BARRIER)
            lp.set_time_limit(120)  # a confirmation that does not finish confirms nothing
            if lp.solver == GUROBI:
                b.params.DualReductions = 0  # INFEASIBLE or OPTIMAL rather than INF_OR_UNBD
        try:
            lp.slim_solve()
        except Exception:
            pass  # a status the backend does not map (barrier, abort); the raw status decides
        if lp.solver == GUROBI:
            feasible = b.Status in (2, 13) and b.SolCount > 0
        else:
            feasible = b.solution.get_status() in (1, 5, 6)
        if feasible:
            if not values:
                return True
            return [v.X for v in b.getVars()] if lp.solver == GUROBI else b.solution.get_values()
    return None if values else False


class SDMILP(SDProblem, MILP_LP):
    """Class that contains functions for the solution of the strain design MILP
     
    This class is a wrapper and inherited from the casses SDProblem, MILP_LP.
    The constructor of SDProblem (see strainDesignProblem.py) translates a given
    problem into a MILP. The constructor of MILP_LP (see solver_interface.py) then
    sets up the solver interface for the selected solver. In addition to the functions
    from SDProblem and MILP_LP, SDMILP provides functions for the solution of the 
    strain design MILP, such as verification of strain design solutions or introduction 
    of exclusion constraints for computing multiple solutions.
    
    Args:
        model (cobra.Model):
            A metabolic model that is an instance of the cobra.Model class.
        
        sd_modules ((list of) straindesign.SDModule):
            Modules that specify the strain design problem, e.g., protected or suppressed flux states for 
            MCS strain design or inner and outer objective functions for OptKnock. See description
            of SDModule for more information on how to set up modules.
            
        ko_cost (optional (dict)): (Default: None)
            A dictionary of reaction identifiers and their associated knockout costs. If not specified, all reactions
            are treated as knockout candidates, equivalent to ko_cost = {'r1':1, 'r2':1, ...}. If a subset of reactions
            is listed in the dict, all other are not considered as knockout candidates.
            
        ki_cost (optional (dict)): (Default: None)
            A dictionary of reaction identifiers and their associated costs for addition. If not specified, all reactions
            are treated as knockout candidates. Reaction addition candidates must be present in the original model with
            the intended flux boundaries **after** insertion. Additions are treated adversely to knockouts, meaning that
            their exclusion from the network is not associated with any cost while their presence entails intervention costs.
            
        max_cost (optional (int)): (Default: inf): 
            The maximum cost threshold for interventions. Every possible intervention is associated with a
            cost value (1, by default). Strain designs cannot exceed the max_cost threshold. Individual
            intervention cost factors may be defined through ki_cost and ko_cost.
        
        solver (optional (str)): (Default: same as defined in model / COBRApy)
            The solver that should be used for preparing and carrying out the strain design computation.
            Allowed values are 'cplex', 'gurobi', 'scip' and 'glpk'.
            
        M (optional (int)): (Default: None)
            If this value is specified (and non-zero, not None), the computation uses the big-M 
            method instead of indicator constraints. Since GLPK does not support indicator constraints it uses
            the big-M method by default (with COBRA standard M=1000). M should be chosen 'sufficiently large' 
            to avoid computational artifacts and 'sufficiently small' to avoid numerical issues.
            
        essential_kis (optional (set)):
            A set of reactions that are marked as addable and that are essential for at least one of the
            strain design modules. Providing such "essential knock-ins" may speed up the strain design computation.
            
    Returns:
        (SDMILP):
            An instance of SDProblem containing the strain design MILP and providing several functions for its solution
    """

    def __init__(self, model: Model, sd_modules: List[SDModule], **kwargs):
        # Construct problem
        SDProblem.__init__(self, model, sd_modules, **kwargs)
        # A knock-in only enlarges the flux space and a knock-out only shrinks it. So when every
        # module is of the kind that survives that direction, a rewarding intervention can never
        # invalidate a design, and every design is beaten by the one that also takes it. Fixing
        # it here is exact and spares the enumeration from walking the designs it dominates.
        _types = {m[MODULE_TYPE] for m in sd_modules}
        _forced = []
        for i in range(self.num_z):
            if self.z_non_targetable[i] or self.cost[i] >= 0.0:
                continue
            if (self.z_inverted[i] and _types == {PROTECT}) or \
               (not self.z_inverted[i] and _types == {SUPPRESS}):
                _forced.append(i)
                logging.info('  Rewarding intervention %s cannot invalidate a design here and is '
                             'taken in every one of them.' % model.reactions[i].id)
        if _forced:
            # as a row rather than by fixing the bound: with the bound form SCIP's populate
            # re-reports the same design instead of reporting infeasible once the designs are
            # exhausted, which never terminates (measured; the other solvers accept either form)
            rows = sparse.lil_matrix((len(_forced), self.A_ineq.shape[1]))
            for k, i in enumerate(_forced):
                rows[k, i] = -1.0
            self.A_ineq = sparse.vstack((self.A_ineq, rows.tocsr()), format='csr')
            self.b_ineq = list(self.b_ineq) + [-1.0] * len(_forced)
        # Module type of each indicator gate (its continuous columns belong to one module), for the
        # solvers that branch on CellNetAnalyzer's gate binaries. Read before trimming moves columns.
        gate_modules = None
        if self.is_mcs_computation and self.solver in _CNA_GATE_BINARIES \
                and getattr(self.indic_constr, 'A', None) is not None and self.indic_constr.A.shape[0]:
            column_module = {}
            for module_type, start, end in self._module_cols:
                column_module.update((j, module_type) for j in range(start, end))
            gates = sparse.csr_matrix(self.indic_constr.A)
            gate_modules = []
            for k in range(gates.shape[0]):
                types = {column_module.get(int(j)) for j in gates.indices[gates.indptr[k]:gates.indptr[k + 1]]} - {None}
                gate_modules.append(types.pop() if len(types) == 1 else 'mixed')
        # Remove non-knockable z-variables before solver sees them
        self._trim_z_variables()
        # Interventions that do not cost anything to take. Adding one to a design can only keep
        # its cost equal or lower, so such a design is not dominated by the smaller one and the
        # exclusion constraints must leave it reachable. Empty for the usual all-positive setup,
        # where dominance and set inclusion agree and the cuts stay exactly as they were.
        self._free_z = [i for i in range(self.num_z) if not self.z_non_targetable[i] and self.cost[i] <= 0.0]
        # A rewarding intervention strictly lowers the cost of any design that can absorb it,
        # so a design is only worth reporting once none of them can be added while staying valid.
        self._rewarding_z = [i for i in self._free_z if self.cost[i] < 0.0]
        # Build MILP object from constructed problem
        MILP_LP.__init__(self,
                         gate_modules=gate_modules,
                         sos1_gates=self.is_mcs_computation,
                         c=self.c,
                         A_ineq=self.A_ineq,
                         b_ineq=self.b_ineq,
                         A_eq=self.A_eq,
                         b_eq=self.b_eq,
                         lb=self.lb,
                         ub=self.ub,
                         vtype=self.vtype,
                         indic_constr=self.indic_constr,
                         M=self.M,
                         solver=self.solver,
                         seed=self.seed,
                         milp_threads=self.milp_threads)

    def is_dominated(self, z):
        """True if a rewarding intervention can join this design without invalidating it.

        Such a design is strictly cheaper and contains this one, so this one is not worth
        reporting. Only reachable when some intervention carries a negative cost.
        """
        for j in self._rewarding_z:
            if z[0, j]:
                continue
            z_more = z.tolil()
            z_more[0, j] = 1.0
            if all(self.verify_sd(z_more.tocsr())):
                return True
        return False

    def _trim_z_variables(self):
        """Remove non-knockable (ub=0) z-variables from MILP matrices.

        Non-knockable reactions have ub=0 and cost=0. With Presolve=0 the
        solver carries these dead variables. This trims them at construction
        time. Stores _z_orig_indices for expanding solutions back.
        """
        keep_z = [i for i in range(self.num_z) if self.ub[i] > 0]
        if len(keep_z) == self.num_z:
            self._z_orig_indices = None  # no trimming needed
            return

        self._z_orig_indices = keep_z  # trimmed_idx -> orig_idx
        self._orig_num_z = self.num_z

        n_cont = len(self.c) - self.num_z
        keep_cols = keep_z + list(range(self.num_z, self.num_z + n_cont))

        # Trim constraint matrices (column selection)
        self.A_ineq = self.A_ineq[:, keep_cols]
        self.A_eq = self.A_eq[:, keep_cols]
        self.c = [self.c[i] for i in keep_cols]
        self.lb = [self.lb[i] for i in keep_cols]
        self.ub = [self.ub[i] for i in keep_cols]

        # Trim indicator constraints
        if hasattr(self.indic_constr, 'A') and self.indic_constr.A is not None:
            # over all kept columns, not just the z-block: an indicator may key on a binary
            # that sits past the interventions, and it still has to follow the renumbering
            old_to_new = {old: new for new, old in enumerate(keep_cols)}
            self.indic_constr.A = self.indic_constr.A[:, keep_cols]
            self.indic_constr.binv = [old_to_new[b] for b in self.indic_constr.binv]

        # Update z-related arrays (trim to knockable-only)
        new_num_z = len(keep_z)
        self.cost = [self.cost[i] for i in keep_z]
        self.z_inverted = [self.z_inverted[i] for i in keep_z]
        self.z_non_targetable = [self.z_non_targetable[i] for i in keep_z]
        self.idx_z = list(range(new_num_z))
        self.num_z = new_num_z
        # carry each kept column's own type across rather than rebuilding from the z-count,
        # which silently retypes any binary past the interventions as continuous
        self.vtype = ''.join(self.vtype[i] for i in keep_cols)
        self.c_bu = [float(i) for i in self.c]

        logging.info(f'  Trimmed z-variables: {self._orig_num_z} -> {new_num_z} '
                     f'({self._orig_num_z - new_num_z} non-knockable removed)')

    def _expand_z_to_orig(self, z_trimmed):
        """Expand trimmed z-solution back to original z-space."""
        if self._z_orig_indices is None:
            return z_trimmed
        n_rows = z_trimmed.shape[0]
        expanded = sparse.lil_matrix((n_rows, self._orig_num_z))
        z_coo = z_trimmed.tocoo()
        for row, col, val in zip(z_coo.row, z_coo.col, z_coo.data):
            expanded[row, self._z_orig_indices[col]] = val
        return expanded.tocsr()

    def add_exclusion_constraints(self, z):
        """Exclude a design and every superset of it that cannot be cheaper.

        With only positive intervention costs a superset always costs more, so this excludes
        all supersets and the rule is plain set inclusion. When some interventions are free or
        rewarding (cost <= 0), a superset taking one of those is not dominated by the design
        found here, so those literals enter the row negated and such supersets stay reachable.
        """
        for i in range(z.shape[0]):
            free = [j for j in self._free_z if not z[i, j]]
            # introduce constraint to make MILP infeasible. Some solvers cannot handle empty rows
            if z[i].nnz == 0 and not free:
                A_ineq = sparse.csr_matrix([1.0] * z[i].shape[1])
                A_ineq.resize((1, self.A_ineq.shape[1]))
                b_ineq = -1
                self.add_ineq_constraints(A_ineq, [b_ineq])
            # a single intervention can be switched off outright, but only while no free
            # intervention could join it to form a design that is not dominated by this one
            elif z[i].nnz == 1 and not free:
                interv_idx = int(z[i].indices[0])
                self.z_non_targetable[interv_idx] = True
                self.set_ub([[interv_idx, 0.0]])
            # otherwise, introduce integer cut constraint
            else:
                A_ineq = z[i].copy()
                A_ineq.resize((1, self.A_ineq.shape[1]))
                b_ineq = np.sum(z[i]) - 1
                if free:
                    A_ineq = A_ineq.tolil()
                    for j in free:
                        A_ineq[0, j] = -1.0
                    A_ineq = A_ineq.tocsr()
                self.add_ineq_constraints(A_ineq, [b_ineq])

    def _set_anchor(self, new, why, log=True):
        """Move the Farkas anchor of the MILP's anchor rows to -new (CPLEX, Gurobi); exact for any
        new > 0. Only the MILP moves; verify_sd reads the anchor rows through _verify_rhs."""
        new = float(min(1.0, max(1e-8, new)))
        if new == self._anchor_c:
            return
        for i in self._anchor_rows:
            self.b_ineq[i] = -new
            b = self.backend
            if hasattr(b, 'linear_constraints'):
                b.linear_constraints.set_rhs(int(i), -new)
            elif hasattr(b, 'getConstrs'):
                b.getConstrs()[int(i)].RHS = -new
                b.update()
        if log:
            logging.warning('Farkas anchor %g -> %g (%s)' % (self._anchor_c, new, why))
        self._anchor_c = new
        if self.solver in _TILT_SCALES_WITH_ANCHOR and getattr(self, '_tilt_base', None):
            self.set_objective([c0 + (t - c0) / new for c0, t in zip(*self._tilt_base)])

    def _steer_anchor(self, diags):
        """Between levels: keep the smallest gate slack of the accepted designs a thousand times
        above the MILP tolerance (c * margin >= 1e-6) and the largest certificate entry clear of
        round-off (c * umax <= 1e3). When both cannot hold, stay on the small side: a too-small
        anchor shows up as rejected designs and is corrected, a too-large one loses designs
        silently."""
        m = [d['margin'] for d in diags if d.get('valid') and d.get('margin')]
        u = [d['umax'] for d in diags if d.get('valid') and d.get('umax')]
        if not m:
            return
        lo = 1e-6 / min(m)
        hi = 1e3 / max(u) if u else np.inf
        c = self._anchor_c
        target = min(max(c, lo), hi) if lo <= hi else hi
        if target > 3 * c or target < c / 3:
            self._set_anchor(target, 'margin steering: min slack %.3g, max entry %.3g at anchor 1' % (min(m), max(u) if u else 0))

    def _minimise_design(self, z):
        """Drop gates whose slack is zero in the slack verification until none is, returning the
        cut set that remains (the same object when z is already minimal as far as the LP shows)."""
        cur = z
        inv = {o: t for t, o in enumerate(self._z_orig_indices)} if self._z_orig_indices is not None else None
        for _ in range(int(z.getnnz()) if hasattr(z, 'getnnz') else z.shape[1]):
            d = getattr(self, '_verify_diag', [None])[-1]
            if not d or not d.get('valid') or not d.get('zero_gates'):
                return cur
            j = d['zero_gates'][0]
            t = inv.get(j, None) if inv is not None else j
            if t is None or cur[0, t] == 0:
                return cur
            nxt = cur.tolil(copy=True); nxt[0, t] = 0; nxt = nxt.tocsr()
            if not all(self.verify_sd(nxt)):
                return cur
            cur = nxt
        return cur

    def add_exclusion_constraints_ineq(self, z):
        """Exclude exact binary solution in z (but not its supersets) from MILP.

        For each solution row in z, adds: sum(z_active) - sum(z_inactive) <= |active| - 1
        This excludes the exact pattern without blocking supersets or subsets.
        """
        z_dense = z.toarray()
        for j in range(z.shape[0]):
            n_active = int(np.sum(z_dense[j] != 0))
            if n_active == 0:
                continue
            coeffs = [1.0 if z_dense[j, i] else -1.0 for i in range(z.shape[1])]
            A_row = sparse.csr_matrix([coeffs])
            A_row.resize((1, self.A_ineq.shape[1]))
            b_ineq = n_active - 1
            self.add_ineq_constraints(A_row, [b_ineq])

    def sd2dict(self, sol, *args) -> Dict:
        """Translate binary solution vector to dictionary for human-readable output"""
        output = {}
        reacID = self.model.reactions.list_attr("id")
        for i in self.idx_z:
            orig_i = self._z_orig_indices[i] if self._z_orig_indices is not None else i
            if sol[0, i] != 0 and not np.isnan(sol[0, i]):
                if self.z_inverted[i]:
                    output[reacID[orig_i]] = sol[0, i]
                else:
                    output[reacID[orig_i]] = -sol[0, i]
            elif args and args[0] and (sol[0, i] == 0) and self.z_inverted[i]:
                output[reacID[orig_i]] = 0.0
        return output

    def solveZ(self) -> Tuple[List, int]:
        """Solve MILP, and return only binary variables rounded to 5 decimals (should return ints)"""
        x, opt, status = self.solve()
        z = sparse.csr_matrix([round(x[i], 5) for i in self.idx_z])
        return z, x, opt, status

    def populateZ(self, n) -> Tuple[List, int]:
        """Populate MILP, and return only binary variables rounded to 5 decimals (should return ints)"""
        x, _, status = self.populate(n)
        self.pool_exhausted = getattr(self.backend, '_pool_exhausted', False)
        if status in [OPTIMAL, TIME_LIMIT_W_SOL]:
            z = sparse.csr_matrix([[round(x[j][i], 5) for i in self.idx_z] for j in range(len(x))])
            z.resize((len(x), self.num_z))
            # remove duplicates
            unique_row_indices, unique_columns = [], []
            for row_idx, row in enumerate(z):
                indices = row.indices.tolist()
                if indices not in unique_columns:
                    unique_columns.append(indices)
                    unique_row_indices.append(row_idx)
            z = z[unique_row_indices]
        else:
            z = sparse.csr_matrix((0, self.num_z))
        return z, status

    def fixObjective(self, c, cx):
        """Enforce a certain objective function and value (or any other constraint of the form c*x <= cx)"""
        n = self.A_ineq.shape[1]
        if len(c) < n:  # columns appended after construction (gate slacks) carry no objective
            c = list(c) + [0.0] * (n - len(c))
        self.set_ineq_constraint(self.idx_row_obj, c, cx)

    def resetObjective(self):
        """Reset objective to the one set upon MILP construction"""
        self.set_objective_idx([[i, v] for i, v in enumerate(self.c_bu)])

    def setMinIntvCostObjective(self):
        """Reset minimization of intervention costs as global objective"""
        self.clear_objective()
        self.set_objective_idx([[i, self.cost[i]] for i in self.idx_z if i not in self.z_non_targetable])

    def resetTargetableZ(self):
        """Reset targetable/switchable intervention indicators / allow all intervention candidates"""
        self.set_ub([[i, 1.0] for i in self.idx_z if not self.z_non_targetable[i]])

    def setTargetableZ(self, sol):
        """Only allow a subset of intervention candidates"""
        self.set_ub([[i, 0.0] for i in self.idx_z if not sol[0, i]])

    def _verify_rhs(self, r):
        """Right-hand side of continuous row r as verify_sd reads it. A Farkas anchor row is read as
        b'y <= -max|b|: certificates form a cone, so this is the same set of designs whatever
        anchor constant and target threshold the MILP was built with, and it keeps the anchor at
        the scale of its own coefficients, far from the verification tolerance. Otherwise u = 0
        can pass within tolerance and the wild type reads as a cut set."""
        b = self.cont_MILP.b_ineq[r]
        if r in self._farkas_anchor_rows:
            row = self.cont_MILP.A_ineq[r, :]
            return -float(np.max(np.abs(row.data))) if row.nnz else b
        return b

    def _verify_slack(self, sol_row, inactive_vars, active_vars, inactive_ineqs, active_ineqs,
                      inactive_eqs, active_eqs) -> bool:
        """Verification that keeps the design's gate rows, each with a non-negative slack, and
        minimises the sum of slacks (equalities carry a slack in each direction).

        Infeasible: not a cut set. Feasible: a cut set, and a gate whose slack is zero at the
        optimum is not needed, so the design minus that gate is already a cut set (a proof of
        non-minimality; the converse does not hold). The smallest gate slack is the design's
        margin against the tolerance, and the largest certificate entry its exposure to round-off;
        both are recorded in _verify_diag at the fixed anchor of _verify_rhs, so that they scale
        with the MILP's anchor c.
        """
        cm = self.cont_MILP
        if not hasattr(self, '_gate_of_row'):
            self._gate_of_row = ({int(c): int(z) for z, c in zip(cm.z_map_constr_ineq.row, cm.z_map_constr_ineq.col)},
                                 {int(c): int(z) for z, c in zip(cm.z_map_constr_eq.row, cm.z_map_constr_eq.col)})
        gi, ge = self._gate_of_row
        nv = len(active_vars)
        ni, ne = len(inactive_ineqs), len(inactive_eqs)
        ns = ni + 2 * ne
        A_act = cm.A_ineq[active_ineqs, :][:, active_vars]
        A_rel = cm.A_ineq[inactive_ineqs, :][:, active_vars]
        E_act = cm.A_eq[active_eqs, :][:, active_vars]
        E_rel = cm.A_eq[inactive_eqs, :][:, active_vars]
        blocks_i = [sparse.hstack((A_act, sparse.csr_matrix((A_act.shape[0], ns))))]
        if ni:
            blocks_i.append(sparse.hstack((A_rel, -sparse.eye(ni, ns, format='csr'))))
        A_ineq = sparse.vstack(blocks_i, format='csr')
        b_ineq = [self._verify_rhs(r) for r in active_ineqs] + [self._verify_rhs(r) for r in inactive_ineqs]
        blocks_e = [sparse.hstack((E_act, sparse.csr_matrix((E_act.shape[0], ns))))]
        if ne:
            sp = sparse.lil_matrix((ne, ns))
            for k in range(ne):
                sp[k, ni + 2 * k] = -1.0
                sp[k, ni + 2 * k + 1] = 1.0
            blocks_e.append(sparse.hstack((E_rel, sp.tocsr())))
        A_eq = sparse.vstack(blocks_e, format='csr')
        b_eq = [cm.b_eq[r] for r in active_eqs] + [cm.b_eq[r] for r in inactive_eqs]
        lp = MILP_LP(c=[0.0] * nv + [1.0] * ns, A_ineq=A_ineq, b_ineq=b_ineq, A_eq=A_eq, b_eq=b_eq,
                     lb=[cm.lb[r] for r in active_vars] + [0.0] * ns,
                     ub=[cm.ub[r] for r in active_vars] + [np.inf] * ns, solver=self.solver, seed=self.seed)
        x = _solve_verification_lp(lp, values=True)
        diag = {'valid': x is not None, 'margin': None, 'umax': None, 'zero_gates': []}
        if x is not None:
            x = np.asarray(x, dtype=float)
            u, sl = x[:nv], x[nv:]
            per_gate = {}
            for k, r in enumerate(inactive_ineqs):
                per_gate[gi.get(int(r))] = per_gate.get(gi.get(int(r)), 0.0) + sl[k]
            for k, r in enumerate(inactive_eqs):
                per_gate[ge.get(int(r))] = per_gate.get(ge.get(int(r)), 0.0) + sl[ni + 2 * k] + sl[ni + 2 * k + 1]
            per_gate.pop(None, None)
            if per_gate:
                diag['margin'] = float(min(per_gate.values()))
                diag['zero_gates'] = [z for z, v in per_gate.items() if v <= 1e-7]
            diag['umax'] = float(np.max(np.abs(u))) if nv else 0.0
        self._verify_diag = getattr(self, '_verify_diag', [])
        self._verify_diag.append(diag)
        return diag['valid']

    def verify_sd(self, sols) -> List:
        """Verify computed strain design"""
        sols_orig = self._expand_z_to_orig(sols)
        valid = [False] * sols_orig.shape[0]
        def _split(z_map, sol_row):
            """Columns of z_map the solution switches off, and the complement."""
            inactive = [col for z_i, col, sense in zip(z_map.row, z_map.col, z_map.data)
                        if np.logical_xor(sol_row[z_i], sense == -1)]
            keep = np.ones(z_map.shape[1], dtype=bool)
            keep[inactive] = False
            return inactive, np.nonzero(keep)[0].tolist()

        for i, sol in zip(range(sols_orig.shape[0]), sols_orig):
            sol_row = np.ravel(sol.toarray() if hasattr(sol, 'toarray') else np.asarray(sol))
            inactive_vars, active_vars = _split(self.cont_MILP.z_map_vars, sol_row)
            inactive_ineqs, active_ineqs = _split(self.cont_MILP.z_map_constr_ineq, sol_row)
            inactive_eqs, active_eqs = _split(self.cont_MILP.z_map_constr_eq, sol_row)

            # Zeroing a variable whose bounds exclude zero contradicts that bound, so the
            # region is empty whatever the remaining system does. This has to be tested
            # here rather than left to the LP: prevent_boundary_knockouts keeps such a
            # bound (e.g. ATPM >= 3.15) as a row with no z-mapping so that a knockout
            # contradicts it, but reassign_lb_ub_from_ineq later folds single-variable
            # rows back into variable bounds, and those vanish together with the column.
            if any(self.cont_MILP.lb[j] > 0.0 or self.cont_MILP.ub[j] < 0.0 for j in inactive_vars):
                valid[i] = False
                continue
            if getattr(self, '_slack_verification', False):
                valid[i] = self._verify_slack(sol_row, inactive_vars, active_vars, inactive_ineqs,
                                              active_ineqs, inactive_eqs, active_eqs)
                continue
            # Otherwise drop the columns outright. Absence is a stronger statement than an
            # interval of [0, 0], since it owes nothing to feasibility tolerances.
            lp = MILP_LP(A_ineq=self.cont_MILP.A_ineq[active_ineqs, :][:, active_vars],
                         b_ineq=[self._verify_rhs(i) for i in active_ineqs],
                         A_eq=self.cont_MILP.A_eq[active_eqs, :][:, active_vars],
                         b_eq=[self.cont_MILP.b_eq[i] for i in active_eqs],
                         lb=[self.cont_MILP.lb[i] for i in active_vars],
                         ub=[self.cont_MILP.ub[i] for i in active_vars],
                         solver=self.solver,
                         seed=self.seed)
            valid[i] = _solve_verification_lp(lp)
        return valid

    def compute_optimal(self, **kwargs):
        """Compute the global optimum of the strain design MILP and iteratively find the next best solution
        
        Args:
            max_solutions (optional (int)): (Default: inf)
                The maximum number of MILP solutions that are generated for a strain design problem.
                
            time_limit (optional (int)): (Default: inf)
                The time limit in seconds for the MILP-solver.
                
            show_no_ki (optional (bool)): (Default: True)
                Indicate non-added addition candidates in a solution specifically with a value of 0
                
        Returns:
            (SDSolutions):
            Strain design solutions provided as an SDSolutions object
        """
        keys = {MAX_SOLUTIONS, T_LIMIT, 'show_no_ki'}
        # set keys passed in kwargs
        for key, value in dict(kwargs).items():
            if key in keys:
                setattr(self, key, value)
        # set all remaining keys to None
        for key in keys:
            if key not in dict(kwargs).keys():
                setattr(self, key, None)
        if self.max_solutions is None:
            self.max_solutions = np.inf
        if self.time_limit is None:
            self.time_limit = np.inf
        if self.show_no_ki is None:
            self.show_no_ki = True
        # first check if strain doesn't already fulfill the strain design setup
        if self.is_mcs_computation and self.verify_sd(sparse.csr_matrix((1, self.num_z)))[0]:
            logging.warning('The strain already meets the requirements defined in the strain design setup. ' \
                  'No interventions are needed.')
            return self.build_sd_solution([{}], OPTIMAL, BEST)
        # otherwise continue
        endtime = time.time() + self.time_limit
        status = OPTIMAL
        sols = sparse.csr_matrix((0, self.num_z))
        logging.info('Finding optimal strain designs ...')
        while sols.shape[0] < self.max_solutions and \
          status == OPTIMAL and \
          endtime-time.time() > 0:
            self.set_time_limit(endtime - time.time())
            self.resetTargetableZ()
            self.resetObjective()
            self.fixObjective(self.c_bu, np.inf)
            z, _, opt, status = self.solveZ()
            if 0 in z.shape or np.isnan(z[0, 0]):  # no (further) solution -> stop cleanly
                break
            output = self.sd2dict(z)
            if self.is_mcs_computation:
                if status in [OPTIMAL, TIME_LIMIT_W_SOL] and all(self.verify_sd(z)):
                    logging.info('Strain design with cost ' + str(round((z * self.cost)[0], 6)) + ': ' + str(output))
                    self.add_exclusion_constraints(z)
                    sols = sparse.vstack((sols, z))
                elif status in [OPTIMAL, TIME_LIMIT_W_SOL]:
                    logging.info('Invalid (minimal) solution found: ' + str(output))
                    self.add_exclusion_constraints(z)
                if status != OPTIMAL:
                    break
            else:
                # Verify solution and explore subspace to get minimal intervention sets
                logging.info('Found solution with objective value ' + str(-opt))
                logging.info('Minimizing number of interventions in subspace with ' + str(sum(z.toarray()[0])) + ' possible targets.')
                self.fixObjective(self.c_bu, opt)
                self.setMinIntvCostObjective()
                self.setTargetableZ(z)
                while sols.shape[0] < self.max_solutions and \
                        status == OPTIMAL and \
                        endtime-time.time() > 0:
                    self.set_time_limit(endtime - time.time())
                    z1, _, _, status1 = self.solveZ()
                    output = self.sd2dict(z1)
                    if status1 in [OPTIMAL, TIME_LIMIT_W_SOL] and all(self.verify_sd(z1)):
                        logging.info('Strain design with cost ' + str(round((z1 * self.cost)[0], 6)) + ': ' + str(output))
                        self.add_exclusion_constraints(z1)
                        sols = sparse.vstack((sols, z1))
                    elif status1 in [OPTIMAL, TIME_LIMIT_W_SOL]:
                        logging.warning('Invalid minimal solution found: ' + str(output))
                        self.add_exclusion_constraints_ineq(z1)
                    else:  # return to outside loop
                        break
        if status == INFEASIBLE and sols.shape[0] > 0:  # all solutions found
            status = OPTIMAL
        if status == TIME_LIMIT and sols.shape[0] > 0:  # some solutions found, timelimit reached
            status = TIME_LIMIT_W_SOL
        if endtime - time.time() > 0 and sols.shape[0] > 0:
            logging.info('Finished solving strain design MILP. ')
            if 'strainDesignMILP' in self.__module__:
                logging.info(str(sols.shape[0]) + ' solutions to MILP found.')
        elif endtime - time.time() > 0:
            logging.info('Finished solving strain design MILP.')
            if 'strainDesignMILP' in self.__module__:
                logging.info(' No solutions exist.')
        else:
            logging.info('Time limit reached.')
        # Translate solutions into dict
        sd_dict = []
        for sol in sols:
            sd_dict += [self.sd2dict(sol, self.show_no_ki)]
        return self.build_sd_solution(sd_dict, status, BEST)

    # Find iteratively intervention sets of arbitrary size or quality
    # output format: list of 'dict' (default) or 'sparse'
    def compute(self, **kwargs):
        """Compute arbitrary solutions of the strain design MILP and iteratively find further solutions
        
        Args:
            max_solutions (optional (int)): (Default: inf)
                The maximum number of MILP solutions that are generated for a strain design problem.
                
            time_limit (optional (int)): (Default: inf)
                The time limit in seconds for the MILP-solver.
                
            show_no_ki (optional (bool)): (Default: True)
                Indicate non-added addition candidates in a solution specifically with a value of 0
                
        Returns:
            (SDSolutions):
            Strain design solutions provided as an SDSolutions object
        """
        keys = {MAX_SOLUTIONS, T_LIMIT, 'show_no_ki'}
        # set keys passed in kwargs
        for key, value in kwargs.items():
            if key in keys:
                setattr(self, key, value)
        # set all remaining keys to None
        for key in keys:
            if key not in kwargs.keys():
                setattr(self, key, None)
        if self.max_solutions is None:
            self.max_solutions = np.inf
        if self.time_limit is None:
            self.time_limit = np.inf
        if self.show_no_ki is None:
            self.show_no_ki = True
        # first check if strain doesn't already fulfill the strain design setup
        if self.verify_sd(sparse.csr_matrix((1, self.num_z)))[0]:
            logging.warning('The strain already meets the requirements defined in the strain design setup. ' \
                  'No interventions are needed.')
            return self.build_sd_solution([{}], OPTIMAL, ANY)
        # otherwise continue
        endtime = time.time() + self.time_limit
        status = OPTIMAL
        sols = sparse.csr_matrix((0, self.num_z))
        logging.info('Finding (also non-optimal) strain designs ...')
        while sols.shape[0] < self.max_solutions and \
          status == OPTIMAL and \
          endtime-time.time() > 0:
            logging.info('Searching in full search space.')
            self.set_time_limit(endtime - time.time())
            self.resetTargetableZ()
            self.clear_objective()
            self.fixObjective(self.c_bu, np.inf)  # keep objective open
            z, x, _, status = self.solveZ()
            if status not in [OPTIMAL, TIME_LIMIT_W_SOL]:
                break
            if not all(self.verify_sd(z)):
                self.set_time_limit(endtime - time.time())
                self.resetObjective()
                self.setTargetableZ(z)
                self.fixObjective(self.c_bu, np.sum([c * x for c, x in zip(self.c_bu, x)]))
                z1, _, _, status1 = self.solveZ()
                if status1 == OPTIMAL and not self.verify_sd(z1):
                    self.add_exclusion_constraints(z1)
                    output = self.sd2dict(z1)
                    logging.warning('Invalid minimal solution found: ' + str(output))
                    continue
                if status1 != OPTIMAL and not self.verify_sd(z1):
                    self.add_exclusion_constraints_ineq(z1)
                    output = self.sd2dict(z1)
                    logging.warning('Invalid minimal solution found: ' + str(output))
                    continue
                else:
                    output = self.sd2dict(z)
                    logging.warning('Warning: Solver first found the infeasible solution: ' + str(output))
                    output = self.sd2dict(z1)
                    logging.warning('But a subset of this solution seems to be valid: ' + str(output))
            # Verify solution and explore subspace to get strain designs
            cx = np.sum([c * x for c, x in zip(self.c_bu, x)])
            if not self.is_mcs_computation:
                logging.info('Found preliminary solution.')
            logging.info('Minimizing number of interventions in subspace with ' + str(sum(z.toarray()[0])) + ' possible targets.')
            self.setMinIntvCostObjective()
            self.setTargetableZ(z)
            self.fixObjective(self.c_bu, cx)
            while sols.shape[0] < self.max_solutions and \
                    status == OPTIMAL and \
                    endtime-time.time() > 0:
                self.set_time_limit(endtime - time.time())
                z1, _, _, status1 = self.solveZ()
                output = self.sd2dict(z1)
                if status1 in [OPTIMAL, TIME_LIMIT_W_SOL] and all(self.verify_sd(z1)):
                    logging.info('Strain design with cost ' + str(round((z1 * self.cost)[0], 6)) + ': ' + str(output))
                    self.add_exclusion_constraints(z1)
                    sols = sparse.vstack((sols, z1))
                elif status1 in [OPTIMAL, TIME_LIMIT_W_SOL]:
                    logging.warning('Invalid minimal solution found: ' + str(output))
                    self.add_exclusion_constraints_ineq(z1)
                else:  # return to outside loop
                    break
        if status == INFEASIBLE and sols.shape[0] > 0:  # all solutions found
            status = OPTIMAL
        if status == TIME_LIMIT and sols.shape[0] > 0:  # some solutions found, timelimit reached
            status = TIME_LIMIT_W_SOL
        if endtime - time.time() > 0 and sols.shape[0] > 0:
            logging.info('Finished solving strain design MILP. ')
            if 'strainDesignMILP' in self.__module__:
                logging.info(str(sols.shape[0]) + ' solutions to MILP found.')
        elif endtime - time.time() > 0:
            logging.info('Finished solving strain design MILP.')
            if 'strainDesignMILP' in self.__module__:
                logging.info(' No solutions exist.')
        else:
            logging.info('Time limit reached.')
        # Translate solutions into dict if not stated otherwise
        sd_dict = []
        for sol in sols:
            sd_dict += [self.sd2dict(sol, self.show_no_ki)]
        return self.build_sd_solution(sd_dict, status, ANY)

    # Enumerate iteratively optimal strain designs using the populate function
    # output format: list of 'dict' (default) or 'sparse'
    def enumerate(self, **kwargs):
        """Find all globally optimal solutions to the strain design MILP and iteratively construct pools for the suboptimal values
            
        Args:
            max_solutions (optional (int)): (Default: inf)
                The maximum number of MILP solutions that are generated for a strain design problem.
                
            time_limit (optional (int)): (Default: inf)
                The time limit in seconds for the MILP-solver.
                
            show_no_ki (optional (bool)): (Default: True)
                Indicate non-added addition candidates in a solution specifically with a value of 0
                
        Returns:
            (SDSolutions):
            Strain design solutions provided as an SDSolutions object
        """
        keys = {MAX_SOLUTIONS, T_LIMIT, 'show_no_ki'}
        # set keys passed in kwargs
        for key, value in dict(kwargs).items():
            if key in keys:
                setattr(self, key, value)
        # set all remaining keys to None
        for key in keys:
            if key not in dict(kwargs).keys():
                setattr(self, key, None)
        if self.max_solutions is None:
            self.max_solutions = np.inf
        if self.time_limit is None:
            self.time_limit = np.inf
        if self.show_no_ki is None:
            self.show_no_ki = True
        # first check if strain doesn't already fulfill the strain design setup
        wt_is_design = self.is_mcs_computation and self.verify_sd(sparse.csr_matrix((1, self.num_z)))[0]
        if wt_is_design:
            logging.warning('The strain already meets the requirements defined in the strain design setup. ' \
                  'No interventions are needed.')
            # Free interventions can still yield designs that cost no more than doing nothing, so
            # keep enumerating and let the exclusion constraints carry the empty design forward.
            if not self._free_z:
                return self.build_sd_solution([{}], OPTIMAL, POPULATE)
        # otherwise continue
        if self.solver == 'scip':
            logging.warning("SCIP does not natively support solution pool generation. "+ \
                "An high-level implementation of populate is used. " + \
                "Consider using compute_optimal instead of enumerate, as " + \
                "it returns the same results but faster.")
        if self.solver == 'glpk':
            logging.warning("GLPK does not natively support solution pool generation. "+ \
                "An instable high-level implementation of populate is used. "
                "Consider using compute_optimal instead of enumerate, as " + \
                "it returns the same results but faster." )
        endtime = time.time() + self.time_limit
        status = OPTIMAL
        sols = sparse.csr_matrix((0, self.num_z))
        if wt_is_design:
            empty = sparse.csr_matrix((1, self.num_z))
            if not (self._rewarding_z and self.is_dominated(empty)):
                sols = sparse.vstack((sols, empty))
            self.add_exclusion_constraints(empty)
        logging.info('Enumerating strain designs ...')
        while sols.shape[0] < self.max_solutions and \
          status == OPTIMAL and \
          endtime-time.time() > 0:
            self.set_time_limit(endtime - time.time())
            if not self.is_mcs_computation:
                self.resetTargetableZ()
                self.resetObjective()
                self.fixObjective(self.c_bu, np.inf)
                z, _, opt, status = self.solveZ()
                if status not in [OPTIMAL, TIME_LIMIT_W_SOL]:
                    break
                logging.info('Enumerating all solutions with the objective value: ' + str(-opt))
                self.fixObjective(self.c_bu, opt)
                self.setMinIntvCostObjective()
            z, status = self.populateZ(self.max_solutions - sols.shape[0])
            if status in [OPTIMAL, TIME_LIMIT_W_SOL]:
                for i in range(z.shape[0]):
                    output = [self.sd2dict(z[i])]
                    if all(self.verify_sd(z[i])):
                        if self._rewarding_z and self.is_dominated(z[i]):
                            # a cheaper design contains this one; cut only the exact pattern so
                            # that the design dominating it stays reachable
                            logging.info('Dominated by a cheaper superset, skipping: ' + str(output))
                            self.add_exclusion_constraints_ineq(z[i])
                            continue
                        logging.info('Strain designs with cost ' + str(round((z[i] * self.cost)[0], 6)) + ': ' + str(output))
                        self.add_exclusion_constraints(z[i])
                        sols = sparse.vstack((sols, z[i]))
                    else:
                        logging.warning('Invalid (minimal) solution found: ' + str(output))
                        self.add_exclusion_constraints(z[i])
            if (status != OPTIMAL):  # or (z[i]*self.cost == self.max_cost):
                break
        if sols.shape[0] > 1:
            sols, n_super = _drop_non_minimal(sols)
            if n_super:
                logging.error('%d designs contain a smaller design: the enumeration missed one and '
                              'the result is INCOMPLETE.' % n_super)
                status = ERROR
        if status == INFEASIBLE and sols.shape[0] > 0:  # all solutions found or solution limit reached
            status = OPTIMAL
        if status == TIME_LIMIT and sols.shape[0] > 0:  # some solutions found, timelimit reached
            status = TIME_LIMIT_W_SOL
        if endtime - time.time() > 0 and sols.shape[0] > 0:
            logging.info('Finished solving strain design MILP. ')
            if 'strainDesignMILP' in self.__module__:
                logging.info(str(sols.shape[0]) + ' solutions to MILP found.')
        elif endtime - time.time() > 0:
            logging.info('Finished solving strain design MILP.')
            if 'strainDesignMILP' in self.__module__:
                logging.info(' No solutions exist.')
        else:
            logging.info('Time limit reached.')
        # Translate solutions into dict if not stated otherwise
        sd_dict = []
        for sol in sols:
            sd_dict += [self.sd2dict(sol, self.show_no_ki)]
        sd_solution = self.build_sd_solution(sd_dict, status, POPULATE)
        return sd_solution

    def enumerate_ksweep(self, **kwargs):
        """Enumerate minimal cut sets one cost level at a time.

        ``enumerate`` runs populate over the whole budget ``sum(cost*z) <= max_cost``. This sweep
        instead pins the design cost to each level ``k = 1 .. max_cost`` in turn and exhausts that
        level before moving on::

            for k in 1 .. max_cost:
                set  sum(cost*z) == k          (both budget rows -> k)
                while populate returns solutions:
                    verify every pool solution
                    add exclusion  sum_{j in K} z_j <= |K|-1  (and its supersets)

        It returns the same designs as ``enumerate``. With the cost pinned, the pool gap can be
        opened and the dual tilt applied, and on CPLEX and Gurobi the level loop also runs the
        Farkas anchor controller and the restarts described below.

        The sweep needs an MCS computation with a finite ``max_cost`` and positive, integer
        intervention costs: other costs reach totals between or below the levels it visits. In
        every other case it calls ``enumerate``.
        """
        keys = {MAX_SOLUTIONS, T_LIMIT, 'show_no_ki'}
        # set keys passed in kwargs
        for key, value in dict(kwargs).items():
            if key in keys:
                setattr(self, key, value)
        # set all remaining keys to None
        for key in keys:
            if key not in dict(kwargs).keys():
                setattr(self, key, None)
        if self.max_solutions is None:
            self.max_solutions = np.inf
        if self.time_limit is None:
            self.time_limit = np.inf
        if self.show_no_ki is None:
            self.show_no_ki = True
        # The levels are the integers 1, 2, ...: a non-integer total cost falls between two of
        # them and a total of zero or less below all of them, so either would be skipped.
        max_cost_finite = self.max_cost is not None and np.isfinite(self.max_cost)
        finite_costs = [c for c in self.cost if np.isfinite(c)]
        costs_integer = all(abs(c - round(c)) < 1e-9 for c in finite_costs)
        costs_positive = all(c > 0 for c in finite_costs)
        if (not self.is_mcs_computation) or (not max_cost_finite) or (not costs_integer) \
                or (not costs_positive):
            logging.info('Enumerating over the whole budget: the cost-level sweep needs an MCS '
                         'computation with a finite budget and positive, integer intervention costs.')
            return self.enumerate(**kwargs)
        # first check if strain doesn't already fulfill the strain design setup
        if self.verify_sd(sparse.csr_matrix((1, self.num_z)))[0]:
            logging.warning('The strain already meets the requirements defined in the strain design setup. ' \
                  'No interventions are needed.')
            return self.build_sd_solution([{}], OPTIMAL, POPULATE)
        # otherwise continue
        if self.solver == 'scip':
            logging.warning("SCIP does not natively support solution pool generation. "+ \
                "An high-level implementation of populate is used. " + \
                "Consider using compute_optimal instead of enumerate, as " + \
                "it returns the same results but faster.")
        if self.solver == 'glpk':
            logging.warning("GLPK does not natively support solution pool generation. "+ \
                "An instable high-level implementation of populate is used. "
                "Consider using compute_optimal instead of enumerate, as " + \
                "it returns the same results but faster." )
        # Full-width cost vector for the two budget-bracket rows (z-cols carry cost,
        # continuous cols carry 0). Rows: idx_row_mincost:  cost.z <= k  ;
        #                                 idx_row_maxcost: -cost.z <= -k  (-> cost.z >= k).
        # Together they pin sum(cost*z) == k for the current level.
        n_cont = len(self.c) - self.num_z
        cost_full = [float(c) for c in self.cost] + [0.0] * n_cont
        neg_cost_full = [-c for c in cost_full]
        k_max = int(np.floor(self.max_cost))  # a cost-k solution is within budget only if k <= max_cost
        endtime = time.time() + self.time_limit
        status = OPTIMAL
        hit_timelimit = False
        errored = False
        sols = sparse.csr_matrix((0, self.num_z))
        logging.info('Enumerating strain designs (k-sweep) ...')
        native = self.solver in (CPLEX, GUROBI)
        # Farkas anchor controller (CPLEX, Gurobi). The certificates form a cone, so the anchor c in
        # b'y <= -c is exact at any positive value; it sets their scale against the solver's absolute
        # tolerances, and a small one is markedly faster. The sweep starts at _ANCHOR_START. A design
        # that fails verification means a certificate passed only within tolerance, so c is raised
        # 100-fold (up to 1) and the level re-populated; between levels c is steered from the gate
        # slacks of the accepted designs, which the slack verification reports. A design that turns
        # out not to be minimal means a lower level lost one: c is lowered and those levels redone.
        self._anchor_rows = list(getattr(self, '_milp_anchor_rows', [])) if native else []
        # an anchor row always has the build value -1; anything else means the numbering is off
        self._anchor_rows = [i for i in self._anchor_rows if i < len(self.b_ineq) and self.b_ineq[i] == -1.0]
        self._anchor_adaptive = bool(self._anchor_rows)
        self._slack_verification = self._anchor_adaptive
        self._anchor_c = 1.0
        if self._anchor_adaptive:
            self._set_anchor(_ANCHOR_START, 'start', log=False)
            logging.info('  Farkas anchor %g on %d row(s), adaptive' % (self._anchor_c, len(self._anchor_rows)))
        rejected_here = False
        lowered_here = False
        # The solver's certificate that a pinned level is exhausted (CPLEX 129/130, Gurobi an optimal
        # pool search with room left) ends the level without a confirmatory populate. It is withdrawn
        # for the rest of the run once the lower levels had to be redone more than three times.
        trust_certificate = True
        # Only here is the design cost pinned to a single value, so only here can the pool's
        # optimality gap be opened without losing the ascending-cost order that makes an emitted
        # design minimal. Everything above this line -- including every fallback to enumerate() --
        # runs with the gap closed.
        self.set_pool_gap(True)
        tilt = getattr(self, 'dual_tilt', None)
        if tilt:
            # The pinned level makes the objective constant on z and zero on the dual variables, so
            # the node LP is a pure feasibility problem and the simplex wanders over a degenerate
            # optimal face. A small cost gives it a direction. It is not a bound: no certificate is
            # cut, and with the pool gap open every feasible design is still collected. Where gates
            # are SOS1 sets the tilt goes on their sign-restricted members, otherwise on every
            # sign-restricted continuous column (lb finite, ub infinite), so the LP stays bounded.
            tilted = list(self.c)
            cols = range(self.num_z, len(tilted))
            if self.sos1:
                cols = sorted({int(j) for members in self.sos1 for j in members if j >= self.num_z})
            n_tilted = 0
            for j in cols:
                if self.lb[j] is not None and not isinf(self.lb[j]) and self.lb[j] >= 0 and isinf(self.ub[j]):
                    tilted[j] = float(tilt)
                    n_tilted += 1
            self._tilt_base = (list(self.c), list(tilted))
            if self.solver in _TILT_SCALES_WITH_ANCHOR and self._anchor_adaptive:
                tilted = [c0 + (t - c0) / self._anchor_c for c0, t in zip(*self._tilt_base)]
            self.set_objective(tilted)
            logging.info('  dual tilt %g on %d of %d continuous columns' % (float(tilt), n_tilted, len(tilted) - self.num_z))
        # Restarts (CPLEX, Gurobi). Search time at a cost level is heavy-tailed in the solver's seed:
        # one seed can spend minutes on a level that others finish in seconds. Each populate is capped
        # in deterministic work (CPLEX ticks, Gurobi work units, so a run with a fixed seed repeats on
        # the same machine) at max(_RESTART_MIN_WORK, 4 x the previous level's work). A call that is
        # still finding new designs at the cap is resumed with the cap doubled: the model is left as
        # it is, so the solver continues the same tree with its pool. A call that found nothing new
        # by the cap has its designs recorded and is restarted from the next seed derived from the
        # run's seed, the cap doubling on. Every level ends with a populate that ran to completion,
        # so the result is the same as without restarts.
        prev_level_work, n_reseeds = 0.0, 0
        k, n_redos, redo_lower_levels = 0, 0, False
        while k < k_max:
            k += 1
            if sols.shape[0] >= self.max_solutions:
                break
            if endtime - time.time() <= 0:
                hit_timelimit = True
                break
            if self._anchor_adaptive:
                self._steer_anchor(getattr(self, '_verify_diag', []))
                self._verify_diag = []
            # pin the intervention cost to k for this level
            self.set_ineq_constraint(self.idx_row_mincost, cost_full, float(k))
            self.set_ineq_constraint(self.idx_row_maxcost, neg_cost_full, -float(k))
            logging.info('  Enumerating minimal cut sets of cost ' + str(k))
            work_level0 = self.get_work()
            budget = max(_RESTART_MIN_WORK[self.solver], 4.0 * prev_level_work) if native else None
            n_tree, reseed_after_batch = 0, False
            while sols.shape[0] < self.max_solutions and \
                    endtime - time.time() > 0:
                if reseed_after_batch:
                    n_reseeds += 1
                    self.set_seed(self._restart_seed(n_reseeds))
                    reseed_after_batch = False
                self.set_time_limit(endtime - time.time())
                self.set_work_limit(budget)
                work_call0 = self.get_work()
                z, status = self.populateZ(self.max_solutions - sols.shape[0])
                # stopped by the work cap: time is left and the call spent (about) its budget
                capped = budget is not None and endtime - time.time() > 0 and \
                    self.get_work() - work_call0 >= 0.5 * budget
                if capped and status == TIME_LIMIT:
                    n_reseeds += 1
                    self.set_seed(self._restart_seed(n_reseeds))
                    logging.info('  no design at cost %s within the work cap: restarting from another seed' % k)
                    budget *= 2
                    continue
                if capped and status == TIME_LIMIT_W_SOL:
                    if z.shape[0] > n_tree:
                        n_tree = z.shape[0]
                        budget *= 2
                        continue
                    logging.info('  no new design at cost %s within the work cap: restarting from another seed' % k)
                    status = OPTIMAL
                    budget *= 2
                    reseed_after_batch = True
                n_tree = 0
                if status in [OPTIMAL, TIME_LIMIT_W_SOL]:
                    if z.shape[0] == 0:  # level exhausted
                        break
                    for i in range(z.shape[0]):
                        output = [self.sd2dict(z[i])]
                        if all(self.verify_sd(z[i])):
                            zc = z[i]
                            zi = self._minimise_design(zc) if self._slack_verification else zc
                            if zi.getnnz() != zc.getnnz():
                                logging.warning('Non-minimal design %s contains the cut set %s, which a lower '
                                                'level missed; recovered' % (output, [self.sd2dict(zi)]))
                                lowered_here = True
                            logging.info('Strain designs with cost ' + str(round((zi * self.cost)[0], 6)) + ': ' + str([self.sd2dict(zi)]))
                            self.add_exclusion_constraints(zi)
                            sols = sparse.vstack((sols, zi))
                            if lowered_here and self._anchor_adaptive:
                                break  # the rest of the batch is re-found after the lower levels are redone
                        else:
                            logging.warning('Invalid (minimal) solution found: ' + str(output))
                            if self._anchor_adaptive:
                                # under a small anchor a rejection is a spurious certificate, not a
                                # lost design: cut only this point, and raise the anchor below
                                self.add_exclusion_constraints_ineq(z[i])
                                rejected_here = True
                                break  # the rest of the batch is re-found at the larger anchor
                            else:
                                self.add_exclusion_constraints(z[i])
                    if self._anchor_adaptive and rejected_here:
                        self._set_anchor(self._anchor_c * 100.0, 'a design failed verification: certificate inside tolerance')
                        rejected_here = False
                        continue  # re-populate this level at the larger anchor
                    if self._anchor_adaptive and lowered_here:
                        # a design below this level was lost: lower the anchor and redo the levels
                        # below (everything found stays excluded, so those passes only find losses)
                        self._set_anchor(self._anchor_c / 100.0, 'a lower level lost a design: certificate round-off')
                        lowered_here = False
                        redo_lower_levels = True
                        break
                    if status == TIME_LIMIT_W_SOL:
                        hit_timelimit = True
                        break
                    if trust_certificate and self.pool_exhausted:
                        break
                elif status == ERROR:
                    # A solver failure is not an empty level. Treating it as one silently drops
                    # every design at this cardinality and every level above it.
                    logging.error('Solver returned ERROR at cost %s; enumeration is INCOMPLETE '
                                  'from this level up.' % k)
                    errored = True
                    break
                elif status == TIME_LIMIT:
                    # Stopped without a proof either way: the level is not exhausted.
                    hit_timelimit = True
                    break
                else:  # INFEASIBLE at this cardinality -> level exhausted, next k
                    break
            prev_level_work = self.get_work() - work_level0
            if redo_lower_levels:
                redo_lower_levels = False
                n_redos += 1
                if n_redos > 3 and trust_certificate:
                    trust_certificate = False
                    logging.warning('Three redos of the lower levels: the exhaustion certificate is no '
                                    'longer trusted, every level gets its confirmatory pass from here on')
                logging.warning('Redoing cost levels 1..%d after a lost design' % (k - 1))
                k = 0
                continue
            if errored:
                break
            if hit_timelimit or endtime - time.time() <= 0:
                if endtime - time.time() <= 0:
                    hit_timelimit = True
                break
        # The level rows keep their last value on the object, so leave the pool and the work limit as
        # every other caller expects to find them.
        self.set_pool_gap(False)
        self.set_work_limit(None)
        # The status follows from how the sweep ended, not from the last populate.
        if sols.shape[0] > 1:
            sols, n_super = _drop_non_minimal(sols)
            if n_super:
                logging.error('%d designs contain a smaller design: a cost level was enumerated '
                              'incompletely and the result is INCOMPLETE.' % n_super)
                errored = True
        # A solver failure or a lost design makes the result incomplete: ERROR, never OPTIMAL.
        if errored:
            status = ERROR
        elif hit_timelimit and sols.shape[0] > 0:
            status = TIME_LIMIT_W_SOL
        elif hit_timelimit:
            status = TIME_LIMIT
        else:
            status = OPTIMAL
        if not hit_timelimit and sols.shape[0] > 0:
            logging.info('Finished solving strain design MILP. ')
            if 'strainDesignMILP' in self.__module__:
                logging.info(str(sols.shape[0]) + ' solutions to MILP found.')
        elif not hit_timelimit:
            logging.info('Finished solving strain design MILP.')
            if 'strainDesignMILP' in self.__module__:
                logging.info(' No solutions exist.')
        else:
            logging.info('Time limit reached.')
        # Translate solutions into dict
        sd_dict = []
        for sol in sols:
            sd_dict += [self.sd2dict(sol, self.show_no_ki)]
        return self.build_sd_solution(sd_dict, status, POPULATE)

    def _restart_seed(self, n):
        """Seed of the n-th restart, derived from the run's seed so a seeded run repeats."""
        return ((self.seed if self.seed is not None else 0) + 7919 * n) % 2**30

    def build_sd_solution(self, sd_dict, status, solution_approach):
        """Build the strain design solution object"""
        sd_setup = {}
        sd_setup[MODEL_ID] = self.model.id
        sd_setup[MAX_SOLUTIONS] = self.max_solutions
        sd_setup[MAX_COST] = self.max_cost
        sd_setup[TIME_LIMIT] = self.time_limit
        sd_setup[SOLVER] = self.solver
        sd_setup[SOLUTION_APPROACH] = solution_approach
        sd_setup[KOCOST] = {k: float(v) for k,v in \
            zip(self.model.reactions.list_attr('id'),self.ko_cost) if not np.isnan(v)}
        sd_setup[KICOST] = {k: float(v) for k,v in \
            zip(self.model.reactions.list_attr('id'),self.ki_cost) if not np.isnan(v)}
        sd_setup[MODULES] = self.sd_modules
        return SDSolutions(self.model, sd_dict, status, sd_setup)
