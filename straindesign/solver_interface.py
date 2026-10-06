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
"""Unified solver interface for LPs and MILPs (MILP_LP)"""

import math
from numpy import inf, isinf, isnan, unique
from scipy import sparse
from typing import List, Tuple
from straindesign import avail_solvers, GLPK
from straindesign.indicatorConstraints import IndicatorConstraints
from straindesign.names import *
import logging
import os


# CellNetAnalyzer's binary structure for the target-region gates, per solver: (own gate binaries,
# direction binaries for reversible reactions). See MILP_LP._cna_binaries.
_GATE_BINARIES = {CPLEX: (True, True)}


class MILP_LP(object):
    """Unified MILP and LP interface
    
    This class is a wrapper for several solver interfaces to offer unique and 
    consistent bindings for the construction and manipulation of MILPs and LPs 
    in an vector-matrix-based manner and their solution.
    
    Accepts a (mixed integer) linear problem in the form:
        minimize(c),
        subject to: 
        A_ineq * x <= b_ineq,
        A_eq * x  = b_eq,
        lb <= x <= ub,
        forall(i) type(x_i) = vtype(i) (continous, binary, integer),
        indicator constraints:
        x(j) = [0|1] -> a_indic * x [<=|=|>=] b_indic
                    
    Please ensure that the number of variables and (in)equalities is consistent
        
    Example: 
        milp = MILP_LP(c, A_ineq, b_ineq, A_eq, b_eq, lb, ub, vtype, indic_constr)
                
    Args:
        c (list of float): (Default: None)
            The objective vector (Objective sense: minimization).
            
        A_ineq (sparse.csr_matrix): (Default: None)
            A coefficient matrix of the static inequalities.   
            
        b_ineq (list of float): (Default: None)
            The right hand side of the static inequalities.
            
        A_eq (sparse.csr_matrix): (Default: None)
            A coefficient matrix of the static equalities.   
            
        b_eq (list of float): (Default: None)
            The right hand side of the static equalities.
            
        lb (list of float): (Default: None)
            The lower variable bounds.
            
        ub (list of float): (Default: None)
            The upper variable bounds.
            
        vtype (str): (Default: None)
            A character string that specifies the type of each variable:
            'c'ontinous, 'b'inary or 'i'nteger
            
        indic_constr (IndicatorConstraints): (Default: None)
            A set of indicator constraints stored in an object of IndicatorConstraints
            (see reference manual or docstring).

        M (int): (Default: None)
            A large value that is used in the translation of indicator constraints to
            bigM-constraints for solvers that do not natively support them. If no value 
            is provided, 1000 is used.
            
        solver (str): (Default: taken from avail_solvers)
            Solver backend that should be used: 'cplex', 'gurobi', 'glpk' or 'scip'

        skip_checks (bool): (Default: False)
            Upon MILP construction, the dimensions of all provided vectors and matrices
            are checked to verify their consistency. If skip_checks=True is set, these
            checks are skipped.
        
        tlim (float):
            Solution time limit in seconds.
            
        Returns:
            (MILP_LP):
            
            A MILP/LP solver interface class.
    """

    def __init__(self, **kwargs):
        allowed_keys = {
            'c', 'A_ineq', 'b_ineq', 'A_eq', 'b_eq', 'lb', 'ub', 'vtype', 'indic_constr', 'M', SOLVER, 'skip_checks', 'tlim', SEED, 'sos1_gates',
            MILP_THREADS, 'gate_modules'
        }
        # set all keys passed in kwargs
        for key, value in kwargs.items():
            if key in allowed_keys:
                setattr(self, key, value)
            else:
                raise Exception("Key " + key + " is not supported.")
        # set all remaining keys to None
        for key in allowed_keys:
            if key not in kwargs.keys():
                setattr(self, key, None)
        # Select solver (either by choice or automatically by priority cplex > gurobi > scip > glpk)
        if getattr(self, SOLVER) is None:
            # avail_solvers is an unordered set, so pick by an explicit deterministic priority
            for _solver in [CPLEX, GUROBI, SCIP, GLPK]:
                if _solver in avail_solvers:
                    setattr(self, SOLVER, _solver)
                    break
            if getattr(self, SOLVER) is None:
                raise Exception('No solver available. Please ensure that one of the following '\
                    'solvers is avaialable in your Python environment: CPLEX, Gurobi, SCIP, GLPK')
        elif getattr(self, SOLVER) not in avail_solvers:
            raise Exception("Selected solver '" + getattr(self, SOLVER) + "' is not installed / set up correctly.")
        # Copy parameters to object
        if self.A_ineq is not None:
            numvars = self.A_ineq.shape[1]
        elif self.A_eq is not None:
            numvars = self.A_eq.shape[1]
        else:
            logging.warning('Problem has no variables.')
            numvars = 0
        if self.c is None:
            self.c = [0.0] * numvars
        if self.A_ineq is None:
            self.A_ineq = sparse.csr_matrix((0, numvars))
        if self.b_ineq is None:
            self.b_ineq = []
        # Remove unbounded constraints
        if self.A_eq == None:
            self.A_eq = sparse.csr_matrix((0, numvars))
        if self.b_eq == None:
            self.b_eq = []
        if self.lb == None:
            self.lb = [-inf] * numvars
        if self.ub == None:
            self.ub = [inf] * numvars
        if self.vtype == None:
            self.vtype = 'C' * numvars
        # check dimensions
        if not self.skip_checks == True:
            if not (self.A_ineq.shape[0] == len(self.b_ineq)):
                raise Exception("A_ineq and b_ineq must have the same number of rows/elements")
            if not (self.A_eq.shape[0] == len(self.b_eq)):
                raise Exception("A_eq and b_eq must have the same number of rows/elements")
            if not (self.A_ineq.shape[1]==numvars and self.A_eq.shape[1]==numvars and len(self.c)==numvars and \
                    len(self.lb)==numvars and len(self.ub)==numvars and len(self.vtype)==numvars):
                raise Exception("A_eq, A_ineq, c, lb, ub, vtype must have the same number of columns/elements")
            # if (not self.indic_constr==None) and (not self.solver in [CPLEX, GUROBI, SCIP]):
            #     raise Exception("In order to use indicator constraints, you need to set up CPLEX, Gurobi or SCIP.")
            elif (not self.indic_constr == None):  # check dimensions of indicator constraints
                num_ic = self.indic_constr.A.shape[0]
                if not (self.indic_constr.A.shape[1] == numvars and \
                        len(self.indic_constr.b)==num_ic and len(self.indic_constr.binv)==num_ic and \
                        len(self.indic_constr.sense)==num_ic and len(self.indic_constr.indicval)==num_ic):
                    raise Exception("Check dimensions of indicator constraints.")
        # Cast variables as float
        self.A_ineq = self.A_ineq.astype(float)
        self.A_eq = self.A_eq.astype(float)
        self.c = [float(v) for v in self.c]
        self.b_ineq = [float(v) for v in self.b_ineq]
        self.b_eq = [float(v) for v in self.b_eq]
        self.lb = [float(v) for v in self.lb]
        self.ub = [float(v) for v in self.ub]
        if self.indic_constr:
            self.indic_constr.A = self.indic_constr.A.astype(float)
            self.indic_constr.b = [float(v) for v in self.indic_constr.b]
        if not self.solver == GLPK and self.M and not (isnan(self.M) or isinf(self.M)) and \
           self.indic_constr and self.indic_constr.A.shape[0]:
            logging.warning('Provided big M value is ignored unless glpk is used.')
        if self.solver == GLPK and self.milp_threads is not None:
            raise ValueError("milp_threads is not supported for GLPK, which is single-threaded.")
        # On CPLEX and Gurobi the knockout gates of an MCS problem are realised as SOS1 sets instead
        # of indicator constraints: a gate "z = indicval -> a*x sense b" becomes an always-present row
        # that a slack can absorb, plus an SOS1 set pairing that slack with a variable which is
        # nonzero exactly when the gate is on. No big-M is introduced and z keeps its meaning, so
        # costs, the budget row, exclusion rows, verify_sd and sd2dict are untouched. Where the
        # solver branches better on CellNetAnalyzer's binary structure for the target-region gates
        # (_GATE_BINARIES), that structure is built first and the SOS1 sets gate on its binaries.
        self._direct_gates = {}
        self.sos1 = []
        if getattr(self, 'sos1_gates', None) and self.solver in [CPLEX, GUROBI] \
           and self.indic_constr is not None and self.indic_constr.A.shape[0]:
            own, split = _GATE_BINARIES.get(self.solver, (False, False))
            if (own or split) and getattr(self, 'gate_modules', None) is not None:
                self._cna_binaries(own, split)
            self._gates_as_sos1()

        # Create backend
        if self.solver == CPLEX:
            from straindesign.cplex_interface import Cplex_MILP_LP
            self.backend = Cplex_MILP_LP(self.c, self.A_ineq, self.b_ineq, self.A_eq, self.b_eq, self.lb, self.ub, self.vtype,
                                         self.indic_constr, self.seed, self.milp_threads)
        elif self.solver == GUROBI:
            from straindesign.gurobi_interface import Gurobi_MILP_LP
            self.backend = Gurobi_MILP_LP(self.c, self.A_ineq, self.b_ineq, self.A_eq, self.b_eq, self.lb, self.ub, self.vtype,
                                          self.indic_constr, self.seed, self.milp_threads)
        elif self.solver == SCIP:
            from straindesign.scip_interface import SCIP_MILP, SCIP_LP
            self.isLP = all(v == 'C' for v in self.vtype) and self.indic_constr is None
            if self.isLP:
                self.backend = SCIP_LP(self.c, self.A_ineq, self.b_ineq, self.A_eq, self.b_eq, self.lb, self.ub)
                return
            else:
                self.backend = SCIP_MILP(self.c, self.A_ineq, self.b_ineq, self.A_eq, self.b_eq, self.lb, self.ub, self.vtype,
                                         self.indic_constr, self.seed, self.milp_threads)
        elif self.solver == GLPK:
            from straindesign.glpk_interface import GLPK_MILP_LP
            self.backend = GLPK_MILP_LP(self.c, self.A_ineq, self.b_ineq, self.A_eq, self.b_eq, self.lb, self.ub, self.vtype,
                                        self.indic_constr, self.M)
        if self.sos1:
            self.backend.add_sos1(self.sos1)
            # SOS1 gates and the pool intensity are not separable: at CPLEX's default 4 an
            # SOS1-gated e_coli_core does not finish a problem indicators solve in 0.8 s, while at
            # 3 it matches them. 3 is not safe on its own either -- it breaks OPTCOUPLE -- so it is
            # applied here, where the gates are, and nowhere else.
            if hasattr(self.backend, 'set_pool_intensity'):
                self.backend.set_pool_intensity(3)
            if hasattr(self.backend, 'relax_integrality_for_sos1'):
                self.backend.relax_integrality_for_sos1()
        if self.tlim is None:
            self.set_time_limit(inf)
        else:
            self.set_time_limit(self.tlim)


    def _cna_binaries(self, own, split):
        """CellNetAnalyzer's binary structure for the target-region (SUPPRESS) gates.

        Both parts leave the set of feasible z unchanged, so the designs are the same; what changes
        is what the solver can branch on.

        split: a reversible reaction's target-region gate is an equality on its dual row (a*y = b
        while the reaction is present). With it knocked out, any one certificate leaves that row on
        one side of b, so a direction suffices. The row becomes a*y - sp + sn = b with sp, sn >= 0,
        two binaries zp + zn = [reaction knocked out] (CNA's ZP, ZN), and two gates on one
        sign-constrained column each: zp = 0 -> sp <= 0, zn = 0 -> sn <= 0. Only binaries with
        exactly one target-region gate are split, since one direction is chosen per reaction. Cost,
        decoding and the desired-region gates stay on z.

        own: every target-region gate is keyed on its own binary i ("gate enforced") with
        i >= [the intervention requires the gate], i.e. i = 1 whenever the reaction is present and
        free once it is knocked out (CNA's x = 1 -> i = 1). Enforcing a knocked-out reaction's gate
        only removes certificates, so for a fixed z the block is feasible exactly when it is with i
        at its lower bound, which is the shared-z model. Applied after the direction split, so a
        split reaction gets one such binary per direction.

        Gate modules come from the caller (``gate_modules``, one entry per indicator row).
        """
        ic = self.indic_constr
        A = sparse.csr_matrix(ic.A)
        m, n0 = A.shape
        mods = [str(x).lower() for x in self.gate_modules]
        if len(mods) != m:
            logging.warning('gate_modules has %d entries for %d gates; CNA binaries not applied.' % (len(mods), m))
            return
        newcols = []  # (lb, ub, vtype)
        ineqrows, ineqrhs, eqrows, eqrhs = [], [], [], []

        def add_col(lb_, ub_, vt):
            newcols.append((lb_, ub_, vt))
            return n0 + len(newcols) - 1

        n_supp = {}
        for k in range(m):
            if mods[k] == SUPPRESS:
                n_supp[int(ic.binv[k])] = n_supp.get(int(ic.binv[k]), 0) + 1
        gates = []
        n_split = 0
        for k in range(m):
            row = A.getrow(k)
            idx, dat = [int(j) for j in row.indices], [float(v) for v in row.data]
            z, val, sense, b = int(ic.binv[k]), int(ic.indicval[k]), str(ic.sense[k]), float(ic.b[k])
            if split and mods[k] == SUPPRESS and sense == 'E' and n_supp.get(z) == 1:
                sp, sn = add_col(0.0, inf, 'C'), add_col(0.0, inf, 'C')
                zp, zn = add_col(0.0, 1.0, 'B'), add_col(0.0, 1.0, 'B')
                eqrows.append((idx + [sp, sn], dat + [-1.0, 1.0])); eqrhs.append(b)
                # the gate is off at z = 1 - val; exactly then one direction is chosen
                if val == 0:
                    eqrows.append(([z, zp, zn], [1.0, -1.0, -1.0])); eqrhs.append(0.0)
                else:
                    eqrows.append(([z, zp, zn], [1.0, 1.0, 1.0])); eqrhs.append(1.0)
                for d, part in ((zp, sp), (zn, sn)):
                    gates.append([d, 0, 'L', 0.0, [part], [1.0], SUPPRESS, idx])
                n_split += 1
            else:
                gates.append([z, val, sense, b, idx, dat, mods[k], None])
        n_own = 0
        if own:
            owner = {}
            for g in gates:
                if g[6] != SUPPRESS:
                    continue
                key = (g[0], g[1])
                if key not in owner:
                    i = add_col(0.0, 1.0, 'B')
                    owner[key] = i
                    if key[1] == 0:  # gate required while the binary is 0:  i + z >= 1
                        ineqrows.append(([i, key[0]], [-1.0, -1.0])); ineqrhs.append(-1.0)
                    else:            # gate required while the binary is 1:  i >= z
                        ineqrows.append(([key[0], i], [1.0, -1.0])); ineqrhs.append(0.0)
                g[0], g[1] = owner[key], 1
            n_own = len(owner)

        k_new = len(newcols)
        ncol = n0 + k_new
        self.c = list(self.c) + [0.0] * k_new
        self.lb = list(self.lb) + [c_[0] for c_ in newcols]
        self.ub = list(self.ub) + [c_[1] for c_ in newcols]
        self.vtype = self.vtype + ''.join(c_[2] for c_ in newcols)

        def stack(base, b_base, specs, b_specs):
            base = sparse.hstack((base, sparse.csr_matrix((base.shape[0], k_new))), format='csr')
            if not specs:
                return base, list(b_base)
            M_ = sparse.lil_matrix((len(specs), ncol))
            for r, (cols, vals) in enumerate(specs):
                for cc, vv in zip(cols, vals):
                    M_[r, cc] = vv
            return sparse.vstack((base, M_.tocsr()), format='csr'), list(b_base) + list(b_specs)

        self.A_ineq, self.b_ineq = stack(self.A_ineq, self.b_ineq, ineqrows, ineqrhs)
        self.A_eq, self.b_eq = stack(self.A_eq, self.b_eq, eqrows, eqrhs)
        gA = sparse.lil_matrix((len(gates), ncol))
        for r, g in enumerate(gates):
            for cc, vv in zip(g[4], g[5]):
                gA[r, cc] = vv
        self.indic_constr = IndicatorConstraints([g[0] for g in gates], gA.tocsr(), [g[3] for g in gates],
                                                 ''.join(g[2] for g in gates), [g[1] for g in gates])
        self.gate_modules = [g[6] for g in gates]
        # gates on a single sign-constrained column: the SOS1 rewrite pairs that column directly
        self._direct_gates = {r: g[7] for r, g in enumerate(gates) if g[7] is not None}
        logging.info('  CNA binaries: %d reversible target-region gates split by direction, %d own '
                     'gate binaries; %d new columns (%d binary), %d gates.'
                     % (n_split, n_own, k_new, sum(1 for c_ in newcols if c_[2] == 'B'), len(gates)))

    def _gates_as_sos1(self):
        """Rewrite indicator gates as slack rows plus SOS1 sets.

        For `z = indicval -> a*x sense b` add a slack the row can lean on and force that slack to
        zero exactly when the gate is active:

            sense L :  a*x + vp - vn = b ,  vp, vn >= 0 ,  SOS1(g, vn)
            sense G :  a*x + vp - vn = b ,  vp, vn >= 0 ,  SOS1(g, vp)
            sense E :  a*x - s       = b ,  s free     ,  SOS1(g, s)

        `SOS1(g, ...)` makes the slack vanish whenever g is nonzero, so g must be nonzero exactly
        when the gate is on. For indicval = 1 that is z itself; for indicval = 0 it is a continuous
        complement w with z + w = 1, created once per binary rather than once per gate. An
        inequality's slack is defined by an equality in two non-negative parts: a slack that is
        only bounded by its row has a continuum of values for every z pattern, which populate
        would enumerate without end.
        """
        ic = self.indic_constr
        A = sparse.csr_matrix(ic.A)
        n0 = A.shape[1]
        newcols_lb, newcols_ub, newcols_vt = [], [], []
        eqrows, eqrhs = [], []
        comp = {}
        # gate slack column -> columns of the gate row it was split from
        self._gate_slack_src = {}

        def add_col(lb, ub, vt):
            newcols_lb.append(lb); newcols_ub.append(ub); newcols_vt.append(vt)
            return n0 + len(newcols_lb) - 1

        for k in range(A.shape[0]):
            z, val, sense, b = int(ic.binv[k]), int(ic.indicval[k]), str(ic.sense[k]), float(ic.b[k])
            if val == 1:
                g = z
            else:
                if z not in comp:
                    # pinned to 1 - z by its row, so integral without being declared integer
                    w = add_col(0.0, 1.0, 'C')
                    eqrows.append(([z, w], [1.0, 1.0])); eqrhs.append(1.0)
                    comp[z] = w
                g = comp[z]
            row = A.getrow(k)
            idx, dat = list(row.indices), list(row.data)
            if k in self._direct_gates:
                # "x <= 0" on a single column with lb = 0: x is already the slack the gate kills
                x = int(idx[0])
                self.sos1.append([g, x])
                self._gate_slack_src[x] = self._direct_gates[k]
                continue
            if sense == 'E':
                sf = add_col(-inf, inf, 'C')
                eqrows.append((idx + [sf], dat + [-1.0])); eqrhs.append(b)
                self.sos1.append([g, sf])
            else:
                vp, vn = add_col(0.0, inf, 'C'), add_col(0.0, inf, 'C')
                sgn = -1.0 if sense == 'L' else 1.0
                eqrows.append((idx + [vp, vn], dat + [-sgn, sgn])); eqrhs.append(b)
                # a*y = vn - vp, so vn is the positive part: an 'L' gate (a*y <= 0 when on) kills
                # vn, a 'G' gate kills vp
                killed = vn if sense == 'L' else vp
                self.sos1.append([g, killed])
                self._gate_slack_src[killed] = idx

        k_new = len(newcols_lb)
        self.c += [0.0] * k_new
        self.lb += newcols_lb
        self.ub += newcols_ub
        self.vtype += ''.join(newcols_vt)
        ncol = n0 + k_new
        self.A_ineq = sparse.hstack((self.A_ineq, sparse.csr_matrix((self.A_ineq.shape[0], k_new))), format='csr')
        self.A_eq = sparse.hstack((self.A_eq, sparse.csr_matrix((self.A_eq.shape[0], k_new))), format='csr')
        if eqrows:
            m = sparse.lil_matrix((len(eqrows), ncol))
            for r, (cols, vals) in enumerate(eqrows):
                for cc, vv in zip(cols, vals):
                    m[r, cc] = vv
            self.A_eq = sparse.vstack((self.A_eq, m.tocsr()), format='csr')
            self.b_eq = list(self.b_eq) + list(eqrhs)
        logging.info('  Gates as SOS1: %d indicator constraints -> %d SOS1 sets, %d new columns '
                     '(%d complement columns).' % (A.shape[0], len(self.sos1), k_new, len(comp)))
        self.indic_constr = None

    def solve(self) -> Tuple[List, float, float]:
        """Solve the MILP or LP
        
        Example:
            sol_x, optim, status = milp.solve()
        
        Returns:
            (Tuple[List, float, float])
            
            solution_vector, optimal_value, optimization_status
        """
        x, min_cx, status = self.backend.solve()
        if status not in [INFEASIBLE, UNBOUNDED, TIME_LIMIT]:  # if solution exists (is not nan), round integers
            if 'B' in self.vtype or 'I' in self.vtype:
                x = [x[i] if self.vtype[i] == 'C' else int(round(x[i])) for i in range(len(x)) if not isnan(x[i])]
            else:
                # math.isnan: numpy's scalar isnan is slow enough to dominate LPs solved in a loop
                x = [v for v in x if not math.isnan(v)]
        return x, min_cx, status

    def slim_solve(self) -> float:
        """Solve the MILP or LP, but return only the optimal value
                
        Example:
            optim = cplex.slim_solve()
        
        Returns:
            (float)
            
            Optimum value of the objective function.
        """
        a = self.backend.slim_solve()
        return a

    def populate(self, n) -> Tuple[List, float, float]:
        """Generate a solution pool for MILPs
                
        Example:
            sols_x, optim, status = cplex.populate()
        
        Returns:
            (Tuple[List of lists, float, float])
            
            solution_vectors, optimal_value, optimization_status
        """
        return self.backend.populate(n)

    def set_objective(self, c):
        """Set the objective function with a vector"""
        self.c = c
        self.backend.set_objective(c)

    def set_objective_idx(self, C):
        """Set the objective function with index-value pairs
        
        e.g.: C=[[1, 1.0], [4,-0.2]]"""
        # when indices occur multiple times, take first one
        C_idx = [C[i][0] for i in range(len(C))]
        C_idx = unique([C_idx.index(C_idx[i]) for i in range(len(C_idx))])
        C = [C[i] for i in C_idx]
        for i in range(len(C)):
            self.c[C[i][0]] = C[i][1]
        self.backend.set_objective_idx(C)

    def set_ub(self, ub):
        """Set the upper bounds to a given vector"""
        self.ub = ub
        self.backend.set_ub(ub)

    def set_lb(self, lb):
        """Set lower bounds with index-value pairs

        e.g.: lb=[[1, 0.0], [4, -inf]]. Variables not listed keep their bounds."""
        for i, v in lb:
            self.lb[i] = float(v)
        self.backend.set_lb(lb)

    def set_pool_gap(self, open_gap):
        """Open or close the solution pool's optimality gap, where the backend has one.

        A closed gap keeps only pool members at the current optimum. ``enumerate`` depends on that:
        it is what makes designs arrive in ascending intervention cost, and the exclusion of a
        design together with all of its supersets is only a minimality argument under that order.
        Callers may open it where the design cost is pinned by a constraint and the objective can
        therefore no longer separate the pool members that are wanted from those that are not.
        Backends without a solution pool (glpk, scip) ignore this.
        """
        setter = getattr(self.backend, 'set_pool_gap', None)
        if setter is not None:
            setter(bool(open_gap))

    def set_work_limit(self, w):
        """Cap the next solves in deterministic work (CPLEX ticks, Gurobi work units); None lifts it.
        Backends without a deterministic work measure ignore this."""
        setter = getattr(self.backend, 'set_work_limit', None)
        if setter is not None:
            setter(w)

    def get_work(self):
        """Deterministic work counter of the backend, 0 where it has none"""
        getter = getattr(self.backend, 'get_work', None)
        return getter() if getter is not None else 0.0

    def set_seed(self, seed):
        """Set the random seed for subsequent solves, where the backend takes one."""
        setter = getattr(self.backend, 'set_seed', None)
        if setter is not None:
            setter(seed)

    def set_time_limit(self, t):
        """Set the computation time limit (in seconds)"""
        # Floor at 1 ms before dispatching to any backend. The remaining-time passed by the
        # strain-design loop is computed as endtime - time.time() right after a > 0 guard, so a
        # tiny scheduling delay can make it zero or slightly negative; a 1 ms floor keeps the
        # limit valid (Gurobi rejects negative TimeLimit) and, crucially, avoids GLPK treating
        # tm_lim == 0 as "no limit". inf passes through unchanged for the backends' own clamps.
        t = max(t, 1e-3)
        self.tlim = t
        self.backend.set_time_limit(t)

    def add_ineq_constraints(self, A_ineq, b_ineq):
        """Add inequality constraints to the model
        
        Additional inequality constraints have the form A_ineq * x <= b_ineq.
        The number of columns in A_ineq must match with the number of variables x
        in the problem.
        
        Args:
            A_ineq (sparse.csr_matrix):
                The coefficient matrix
                
            b_ineq (list of float):
                The right hand side vector
        """
        A_ineq = sparse.csr_matrix(A_ineq)
        A_ineq.eliminate_zeros()
        b_ineq = [float(b) for b in b_ineq]
        self.A_ineq = sparse.vstack((self.A_ineq, A_ineq))
        self.b_ineq += b_ineq
        self.backend.add_ineq_constraints(A_ineq, b_ineq)

    def add_eq_constraints(self, A_eq, b_eq):
        """Add equality constraints to the model
        
        Additional equality constraints have the form A_eq * x = b_eq.
        The number of columns in A_eq must match with the number of variables x
        in the problem.
        
        Args:
            A_eq (sparse.csr_matrix):
                The coefficient matrix
                
            b_eq (list of float):
                The right hand side vector
        """
        A_eq = sparse.csr_matrix(A_eq)
        A_eq.eliminate_zeros()
        b_eq = [float(b) for b in b_eq]
        self.A_eq = sparse.vstack((self.A_eq, A_eq))
        self.b_eq += b_eq
        self.backend.add_eq_constraints(A_eq, b_eq)

    def set_ineq_constraint(self, idx, a_ineq, b_ineq):
        """Replace a specific inequality constraint
        
        Replace the constraint with the index idx with the constraint a_ineq*x ~ b_ineq
        
        Args:
            idx (int):
                Index of the constraint
                
            a_ineq (list of float):
                The coefficient vector
                
            b_ineq (float):
                The right hand side value
        """
        self.A_ineq = self.A_ineq.tolil()
        self.A_ineq[idx] = sparse.lil_matrix(a_ineq)
        self.A_ineq = self.A_ineq.tocsr()
        self.b_ineq[idx] = b_ineq
        self.backend.set_ineq_constraint(idx, a_ineq, b_ineq)

    def set_lp_method(self, method):
        """Set the LP solving method.

        Uses solver-neutral constants from straindesign.names:
            LP_METHOD_AUTO    — solver default
            LP_METHOD_PRIMAL  — primal simplex
            LP_METHOD_DUAL    — dual simplex
            LP_METHOD_BARRIER — barrier / interior point

        Note: GLPK and SCIP_LP do not support barrier; it falls back to
        dual simplex (GLPK) or is ignored (SCIP_LP).
        """
        self.backend.set_lp_method(method)

    def get_lp_method(self):
        """Return the current LP method as a solver-neutral string."""
        return self.backend.get_lp_method()

    def get_basis(self):
        """Return the current LP basis for warm-starting.

        Returns:
            dict with solver-specific basis data, or None if not supported.
            Pass the returned dict to set_basis() on the same solver type.
        """
        return self.backend.get_basis()

    def set_basis(self, basis):
        """Load a previously saved basis for warm-starting.

        Args:
            basis: dict from get_basis() (same solver type required).
        """
        self.backend.set_basis(basis)

    def clear_objective(self):
        """Clear objective

        Set all coefficients in the objective vector to 0."""
        self.set_objective([0.0] * len(self.c))
