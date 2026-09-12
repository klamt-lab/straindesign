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

from numpy import inf, isinf, isnan, unique
from scipy import sparse
from typing import List, Tuple
from straindesign import avail_solvers, GLPK
from straindesign.indicatorConstraints import IndicatorConstraints
from straindesign.names import *
import logging
import os


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
            MILP_THREADS
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
        # Optional: realise the knockout gates as SOS1 sets instead of indicator constraints.
        # A gate "z = indicval -> a*x sense b" becomes an always-present row that a slack can
        # absorb, plus an SOS1 pairing that slack with a binary which is nonzero exactly when the
        # gate is off. No big-M is introduced and z keeps its meaning, so everything upstream --
        # costs, the budget row, exclusion rows, dominance, verify_sd, sd2dict -- is untouched.
        # Optional: name each gated row activity so every gate is a single-variable condition.
        # Applied before the SOS1 rewrite so the two compose.
        if getattr(self, 'sos1_gates', None) and os.environ.get('SD_GATE_NAMED') \
           and self.indic_constr is not None and self.indic_constr.A.shape[0] \
           and self.solver in [CPLEX, GUROBI]:
            self._gates_as_named_rows()

        self.sos1 = []
        if getattr(self, 'sos1_gates', None) and os.environ.get('SD_SOS1_GATES') \
           and self.indic_constr is not None and self.indic_constr.A.shape[0] \
           and self.solver in [CPLEX, GUROBI]:
            self._gates_as_sos1()

        # Optional: let CPLEX certify a pinned cost level. populate reports 129/130 ("all reachable
        # solutions enumerated") only when the objective varies over the integer variables; with the
        # level pinned the design cost is constant and it reports a bare 101, so the k-sweep has to
        # pay a confirmatory populate per level. One unconstrained binary with a negligible cost
        # restores the certificate (measured: same pool, status 130) and is invisible to every
        # reader of the solution, which index the z block only. Appended last so it sits after
        # any gate-transform columns.
        # SD_POOL_CERT=skip trusts the certificate without adding the column (an open pool gap can
        # produce it on its own); any other value adds the dummy binary as well.
        if getattr(self, 'sos1_gates', None) and os.environ.get('SD_POOL_CERT') not in (None, '', 'skip') \
           and self.solver == CPLEX:
            self.c = list(self.c) + [1e-6]
            self.lb = list(self.lb) + [0.0]
            self.ub = list(self.ub) + [1.0]
            self.vtype = self.vtype + 'B'
            for attr in ('A_ineq', 'A_eq'):
                A = getattr(self, attr)
                setattr(self, attr, sparse.hstack((A, sparse.csr_matrix((A.shape[0], 1))), format='csr'))
            if self.indic_constr is not None and self.indic_constr.A.shape[0]:
                Ai = sparse.csr_matrix(self.indic_constr.A)
                self.indic_constr.A = sparse.hstack((Ai, sparse.csr_matrix((Ai.shape[0], 1))), format='csr')

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
        _intensity = os.environ.get('SD_SOS1_INTENSITY')
        if _intensity and not self.sos1 and hasattr(self.backend, 'set_pool_intensity'):
            # the pool intensity governs how populate's second phase enumerates, which is worth
            # measuring on the indicator formulation too, not only where the SOS1 sets are
            self.backend.set_pool_intensity(int(_intensity))
        if self.sos1:
            self.backend.add_sos1(self.sos1)
            # SOS1 gates and the pool intensity are not separable: at CPLEX's default 4 an
            # SOS1-gated e_coli_core does not finish a problem indicators solve in 0.8 s, while at
            # 3 it matches them. 3 is not safe on its own either -- it breaks OPTCOUPLE -- so it is
            # applied here, where the gates are, and nowhere else.
            # SD_SOS1_INTENSITY overrides the 3: the 129/130 exhaustion certificate needs 4, and
            # the pathology that motivated 3 predates the integrality-tolerance fix.
            if hasattr(self.backend, 'set_pool_intensity'):
                self.backend.set_pool_intensity(int(os.environ.get('SD_SOS1_INTENSITY', 3)))
            if hasattr(self.backend, 'relax_integrality_for_sos1'):
                self.backend.relax_integrality_for_sos1()
        if self.tlim is None:
            self.set_time_limit(inf)
        else:
            self.set_time_limit(self.tlim)


    def _gates_as_named_rows(self):
        """Give every gated row activity a column of its own so each gate spans one variable.

        `z = indicval -> a*x sense b` becomes

            a*x - d = 0                    an ordinary equality row
            z = indicval -> d sense b      the same gate, now one term wide

        The gate keeps its direction, its indicator variable and its meaning; only its width
        changes. SD_GATE_NAMED=split additionally defines d as the difference of two non-negative
        parts, which makes the gated quantity sign-constrained.
        """
        ic = self.indic_constr
        A = sparse.csr_matrix(ic.A)
        m, n0 = A.shape
        split = os.environ.get('SD_GATE_NAMED') == 'split'
        per = 3 if split else 1
        eqrows, eqrhs = [], []
        newcols_lb, newcols_ub = [], []
        gate_col = []
        for k in range(m):
            row = A.getrow(k)
            idx, dat = list(row.indices), list(row.data)
            d = n0 + len(newcols_lb)
            newcols_lb.append(-inf); newcols_ub.append(inf)
            eqrows.append((idx + [d], dat + [-1.0])); eqrhs.append(0.0)
            if split:
                dp, dn = d + 1, d + 2
                newcols_lb += [0.0, 0.0]; newcols_ub += [inf, inf]
                eqrows.append(([d, dp, dn], [1.0, -1.0, 1.0])); eqrhs.append(0.0)
            gate_col.append(d)

        k_new = len(newcols_lb)
        assert k_new == m * per
        self.c += [0.0] * k_new
        self.lb += newcols_lb
        self.ub += newcols_ub
        self.vtype += 'C' * k_new
        ncol = n0 + k_new
        self.A_ineq = sparse.hstack((self.A_ineq, sparse.csr_matrix((self.A_ineq.shape[0], k_new))), format='csr')
        self.A_eq = sparse.hstack((self.A_eq, sparse.csr_matrix((self.A_eq.shape[0], k_new))), format='csr')
        eq = sparse.lil_matrix((len(eqrows), ncol))
        for r, (cols, vals) in enumerate(eqrows):
            for cc, vv in zip(cols, vals):
                eq[r, cc] = vv
        self.A_eq = sparse.vstack((self.A_eq, eq.tocsr()), format='csr')
        self.b_eq = list(self.b_eq) + eqrhs

        gA = sparse.lil_matrix((m, ncol))
        for k, d in enumerate(gate_col):
            gA[k, d] = 1.0
        self.indic_constr = IndicatorConstraints(list(ic.binv), gA.tocsr(), list(ic.b),
                                                 list(ic.sense), list(ic.indicval))
        logging.info('  Named gate rows: %d indicator constraints reduced to one term each, '
                     '%d new columns.' % (m, k_new))

    def _gates_as_sos1(self):
        """Rewrite indicator gates as slack rows plus SOS1 sets.

        For `z = indicval -> a*x sense b` add a slack the row can lean on and force that slack to
        zero exactly when the gate is active:

            sense L :  a*x - s  <= b ,  s >= 0
            sense G :  a*x + s  >= b ,  s >= 0
            sense E :  a*x - sp + sn = b ,  sp, sn >= 0

        `SOS1(g, s...)` then makes s vanish whenever g is nonzero, so g must be nonzero exactly when
        the gate is on. For indicval = 1 that is z itself; for indicval = 0 it is a complement
        binary w with z + w = 1, created once per binary rather than once per gate. Splitting the
        equality slack into non-negative parts keeps every SOS1 member sign-constrained, which is
        the setting the branching rule is meant for.
        """
        ic = self.indic_constr
        A = sparse.csr_matrix(ic.A)
        n0 = A.shape[1]
        # SD_GATE_SLACK_IND keeps the slack rewrite but gates the slack with an indicator on that
        # single variable instead of an SOS1 set -- the shape CellNetAnalyzer's dual has. A gate on
        # one sign-constrained variable is a bound change for the node LP, where a gate on a
        # multi-variable row activates a row.
        slack_ind = bool(os.environ.get('SD_GATE_SLACK_IND'))
        split_dir = slack_ind and bool(os.environ.get('SD_GATE_SPLIT_DIR'))
        ind_rows, ind_binv, ind_sense, ind_b, ind_val = [], [], [], [], []
        newcols_lb, newcols_ub, newcols_vt = [], [], []
        rows, rhs, senses = [], [], []
        eqrows, eqrhs = [], []
        comp = {}

        def add_col(lb, ub, vt):
            newcols_lb.append(lb); newcols_ub.append(ub); newcols_vt.append(vt)
            return n0 + len(newcols_lb) - 1

        for k in range(A.shape[0]):
            z, val, sense, b = int(ic.binv[k]), int(ic.indicval[k]), str(ic.sense[k]), float(ic.b[k])
            if val == 1 or slack_ind:
                # an indicator carries its own trigger value, so the complement column only exists
                # for the SOS1 form, whose sets must vanish on the gate-on side
                g = z
            else:
                if z not in comp:
                    # w is pinned to 1 - z by the row below, so it is already binary-valued and
                    # declaring it integer would only add a branching object the solver does not
                    # need. SD_SOS1_BINCOMP exists to measure that claim rather than assume it.
                    vt = 'B' if os.environ.get('SD_SOS1_BINCOMP') else 'C'
                    w = add_col(0.0, 1.0, vt)
                    eqrows.append(([z, w], [1.0, 1.0])); eqrhs.append(1.0)
                    comp[z] = w
                g = comp[z]
            row = A.getrow(k)
            idx, dat = list(row.indices), list(row.data)
            if sense == 'E':
                # An equality row defines its slack uniquely, so there is no continuum here to
                # pin and the signed split the inequality gates need is pure overhead. One free
                # slack in a two-member set is what gurobi's own presolve builds.
                if os.environ.get('SD_SOS1_SPLIT_EQ'):
                    sp, sn = add_col(0.0, inf, 'C'), add_col(0.0, inf, 'C')
                    eqrows.append((idx + [sp, sn], dat + [-1.0, 1.0])); eqrhs.append(b)
                    self.sos1.append([sp, sn])      # complementarity pins them to the two parts
                    self.sos1.append([g, sp, sn])
                elif slack_ind:
                    # A free slack gated as an equality is the one gate a solver cannot settle by
                    # bound propagation. Splitting it into non-negative parts makes both gated
                    # variables sign-constrained, so the gate collapses to two fixings -- the
                    # reversible-variable split that Klamt et al. 2020 measure at 2.9-5.6x.
                    sp, sn = add_col(0.0, inf, 'C'), add_col(0.0, inf, 'C')
                    eqrows.append((idx + [sp, sn], dat + [-1.0, 1.0])); eqrhs.append(b)
                    if not os.environ.get('SD_SOS1_NOPAIR'):
                        self.sos1.append([sp, sn])
                    for part in (sp, sn):
                        ind_rows.append([part]); ind_binv.append(z); ind_sense.append('L')
                        ind_b.append(0.0); ind_val.append(val)
                    if split_dir:
                        # With the gate off the row is free in both directions, but any one
                        # certificate uses only one of them. A cost-free direction binary lets the
                        # solver fix the unused part by branching instead of by an LP re-solve;
                        # the union over both settings is the same relaxation, so designs are
                        # unchanged. This is CellNetAnalyzer's two-binary split without a second
                        # intervention variable.
                        d = add_col(0.0, 1.0, 'B')
                        ind_rows.append([sn]); ind_binv.append(d); ind_sense.append('L')
                        ind_b.append(0.0); ind_val.append(0)
                        ind_rows.append([sp]); ind_binv.append(d); ind_sense.append('L')
                        ind_b.append(0.0); ind_val.append(1)
                        # Only a knocked-out reaction has a direction to choose. Left free on the
                        # gate-on side, both settings satisfy the model and the pool enumerates
                        # every combination of them.
                        rows.append(([d, z], [1.0, -1.0] if val == 0 else [1.0, 1.0]))
                        rhs.append(0.0 if val == 0 else 1.0); senses.append('L')
                else:
                    sf = add_col(-inf, inf, 'C')
                    eqrows.append((idx + [sf], dat + [-1.0])); eqrhs.append(b)
                    self.sos1.append([g, sf])
            else:
                # A one-sided gate needs slack, but a slack defined by an INEQUALITY is free above
                # the row activity: it sits in one row with no objective, so every gate-off
                # solution has a continuum of equally-optimal values and `populate` -- which is
                # asked for up to 2.1e9 solutions at zero gap -- never terminates. Defining both
                # parts by an EQUALITY and making them complementary pins them to the positive and
                # negative parts of the activity, so each z pattern has exactly one representative.
                vp, vn = add_col(0.0, inf, 'C'), add_col(0.0, inf, 'C')
                sgn = -1.0 if sense == 'L' else 1.0
                eqrows.append((idx + [vp, vn], dat + [-sgn, sgn])); eqrhs.append(b)
                # SD_SOS1_NOPAIR drops this complementarity set. gurobi's presolve removes it
                # anyway (1140 sets -> 597); cplex's keeps 1134, so it is measured separately.
                if not os.environ.get('SD_SOS1_NOPAIR'):
                    self.sos1.append([vp, vn])
                # a*y = vn - vp after complementarity, so vn is the positive part: an 'L' gate
                # (a*y <= 0 when on) kills vn, a 'G' gate kills vp
                killed = vn if sense == 'L' else vp
                if slack_ind:
                    ind_rows.append([killed]); ind_binv.append(z); ind_sense.append('L')
                    ind_b.append(0.0); ind_val.append(val)
                else:
                    self.sos1.append([g, killed])

        k_new = len(newcols_lb)
        # SD_SOS1_SLACK_EPS puts a negligible cost on every gate slack so the LP prefers one
        # representative among equally-feasible certificates. Only sound with the pool gap opened
        # (the cplex backend does that when this is set): with a zero gap the perturbed objective
        # would filter designs by their certificate's slack size, which is a silent truncation.
        _eps = float(os.environ.get('SD_SOS1_SLACK_EPS', 0) or 0)
        # Measured scope: a large win on SUPPRESS-only problems on cplex under the k-sweep (the
        # open gap it needs is cheap there), a 3-6x loss with a PROTECT module, and on gurobi the
        # open gap turns one populate into ~85 (10x slower). Refuse it where it can only hurt.
        if _eps and self.solver != CPLEX:
            logging.warning('SD_SOS1_SLACK_EPS is cplex-only (measured 10x slower on gurobi); ignored.')
            _eps = 0.0
        if _eps and not os.environ.get('SD_ENUM_KSWEEP'):
            logging.warning('SD_SOS1_SLACK_EPS without SD_ENUM_KSWEEP opens the pool gap on an unpinned '
                            'budget: measured 17k pool entries per populate. Use the k-sweep with it.')
        # only sign-constrained slacks: a cost on the FREE equality slack makes the LP unbounded
        # (measured: iJO1366 p1 returned 54 of 249 designs with populate status 118/119)
        self.c += [_eps if (vt == 'C' and lb_ >= 0.0) else 0.0 for vt, lb_ in zip(newcols_vt, newcols_lb)]
        self.lb += newcols_lb
        self.ub += newcols_ub
        self.vtype += ''.join(newcols_vt)
        self.A_ineq = sparse.hstack((self.A_ineq, sparse.csr_matrix((self.A_ineq.shape[0], k_new))), format='csr')
        self.A_eq = sparse.hstack((self.A_eq, sparse.csr_matrix((self.A_eq.shape[0], k_new))), format='csr')
        ncol = n0 + k_new

        def stack(base, b_base, specs, b_specs, into_eq):
            if not specs:
                return base, b_base
            m = sparse.lil_matrix((len(specs), ncol))
            for r, (cols, vals) in enumerate(specs):
                for cc, vv in zip(cols, vals):
                    m[r, cc] = vv
            return sparse.vstack((base, m.tocsr()), format='csr'), list(b_base) + list(b_specs)

        # a 'G' gate row is stored as its negation so everything stays in A_ineq's <= form
        le = [(c_, v_) for (c_, v_), sn in zip(rows, senses) if sn == 'L']
        le_b = [x for x, sn in zip(rhs, senses) if sn == 'L']
        ge = [(c_, [-v for v in v_]) for (c_, v_), sn in zip(rows, senses) if sn == 'G']
        ge_b = [-x for x, sn in zip(rhs, senses) if sn == 'G']
        self.A_ineq, self.b_ineq = stack(self.A_ineq, self.b_ineq, le + ge, le_b + ge_b, False)
        self.A_eq, self.b_eq = stack(self.A_eq, self.b_eq, eqrows, eqrhs, True)
        if slack_ind:
            m = sparse.lil_matrix((len(ind_rows), ncol))
            for r, cols in enumerate(ind_rows):
                m[r, cols[0]] = 1.0
            self.indic_constr = IndicatorConstraints(ind_binv, m.tocsr(), ind_b, ''.join(ind_sense), ind_val)
            logging.info('  Gates as slack indicators: %d row gates -> %d single-variable gates, '
                         '%d SOS1 sets kept, %d new columns.' % (A.shape[0], len(ind_rows), len(self.sos1), k_new))
        else:
            logging.info('  Gates as SOS1: %d indicator constraints -> %d SOS1 sets, %d new columns '
                         '(%d complement binaries).' % (A.shape[0], len(self.sos1), k_new, len(comp)))
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
            x = [x[i] if self.vtype[i] == 'C' else int(round(x[i])) for i in range(len(x)) if not isnan(x[i])]
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
