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
"""GLPK solver interface for LP and MILP"""

from scipy import sparse
from numpy import nan, isnan, inf, isinf, sum
from straindesign.names import *
from typing import Tuple, List
from swiglpk import *
import logging


class GLPK_MILP_LP():
    """GLPK interface for MILP and LP
    
    This class is a wrapper for the GLPK-Python API to offer bindings and namings
    for functions for the construction and manipulation of MILPs and LPs in an
    vector-matrix-based manner that are consistent with those of the other solver 
    interfaces in the StrainDesign package. The purpose is to unify the instructions 
    for operating with MILPs and LPs throughout StrainDesign.
    
    The GLPK interface does not natively support indicator constraints. They are
    hence translated to bigM-constraints when passed to the GLPK constructor
    (see docstring of IndicatorConstraints). The GLPK interface does not natively
    support the populate function. A high level implementation emulates the behavior
    of populate.
    
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
        glpk = GLPK_MILP_LP(c, A_ineq, b_ineq, A_eq, b_eq, lb, ub, vtype, indic_constr, M)
                
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
            A set of indicator constraints stored in an object of IndicatorConstraints.
            To make GLPK compatible with indicator constraints, they are translated into
            bigM-constraints (see reference manual or docstring of IndicatorConstraints).
            
        M (int): (Default: None)
            A large value that is used in the translation of indicator constraints to
            bigM-constraints. If no value is provided, 1000 is used.
            
        Returns:
            (GLPK_MILP_LP):
            
            A GLPK MILP/LP interface class.
    """

    def __init__(self, c=None, A_ineq=None, b_ineq=None, A_eq=None, b_eq=None, lb=None, ub=None, vtype=None, indic_constr=None, M=None):
        self.glpk = glp_create_prob()
        # Careful with indexing! GLPK indexing starts with 1 and not with 0
        try:
            numvars = A_ineq.shape[1]
        except:
            numvars = A_eq.shape[1]
        # prepare coefficient matrix
        if isinstance(A_eq, list):
            if not A_eq:
                A_eq = sparse.csr_matrix((0, numvars))
        if isinstance(A_ineq, list):
            if not A_ineq:
                A_ineq = sparse.csr_matrix((0, numvars))

        if all([v == 'C' for v in vtype]):
            self.ismilp = False
        else:
            self.ismilp = True

        # add and set variables, types and bounds
        if numvars > 0:
            glp_add_cols(self.glpk, numvars)
        for i, v in enumerate(vtype):
            if v == 'C':
                glp_set_col_kind(self.glpk, i + 1, GLP_CV)
            if v == 'I':
                glp_set_col_kind(self.glpk, i + 1, GLP_IV)
            if v == 'B':
                glp_set_col_kind(self.glpk, i + 1, GLP_BV)
        # set bounds
        lb = [float(l) for l in lb]
        ub = [float(u) for u in ub]
        for i in range(numvars):
            if isinf(lb[i]) and isinf(ub[i]):
                glp_set_col_bnds(self.glpk, i + 1, GLP_FR, lb[i], ub[i])
            elif not isinf(lb[i]) and isinf(ub[i]):
                glp_set_col_bnds(self.glpk, i + 1, GLP_LO, lb[i], ub[i])
            elif isinf(lb[i]) and not isinf(ub[i]):
                glp_set_col_bnds(self.glpk, i + 1, GLP_UP, lb[i], ub[i])
            elif not isinf(lb[i]) and not isinf(ub[i]) and lb[i] < ub[i]:
                glp_set_col_bnds(self.glpk, i + 1, GLP_DB, lb[i], ub[i])
            elif not isinf(lb[i]) and not isinf(ub[i]) and lb[i] == ub[i]:
                glp_set_col_bnds(self.glpk, i + 1, GLP_FX, lb[i], ub[i])

        # set objective
        glp_set_obj_dir(self.glpk, GLP_MIN)
        for i, c_i in enumerate(c):
            glp_set_obj_coef(self.glpk, i + 1, float(c_i))

        # add indicator constraints
        A_indic = sparse.lil_matrix((0, numvars))
        b_indic = []
        if not indic_constr == None:
            if not M:
                M = 1e3
            logging.warning('There is no native support of indicator constraints with GLPK.')
            logging.warning('Indicator constraints are translated to big-M constraints with M=' + str(M) + '.')
            num_ic = len(indic_constr.binv)

            eq_type_indic = []  # [GLP_UP]*len(b_ineq)+[GLP_FX]*len(b_eq)
            for i in range(num_ic):
                binv = indic_constr.binv[i]
                indicval = indic_constr.indicval[i]
                A = indic_constr.A[i]
                b = float(indic_constr.b[i])
                sense = indic_constr.sense[i]
                if sense == 'E':
                    A = sparse.vstack((A, -A)).tolil()
                    b = [b, -b]
                else:
                    A = A.tolil()
                    b = [b]
                if indicval:
                    A[:, binv] = M
                    b = [v + M for v in b]
                else:
                    A[:, binv] = -M
                A_indic = sparse.vstack((A_indic, A))
                b_indic = b_indic + b

        # stack all problem rows and add constraints
        if A_ineq.shape[0] + A_eq.shape[0] + A_indic.shape[0] > 0:
            glp_add_rows(self.glpk, A_ineq.shape[0] + A_eq.shape[0] + A_indic.shape[0])
            eq_type = [GLP_UP] * len(b_ineq) + [GLP_FX] * len(b_eq) + [GLP_UP] * len(b_indic)
            for i, t, b in zip(range(len(b_ineq + b_eq + b_indic)), eq_type, b_ineq + b_eq + b_indic):
                glp_set_row_bnds(self.glpk, i + 1, t, float(b), float(b))

            A = sparse.vstack((A_ineq, A_eq, A_indic), 'coo')
            ia = intArray(A.nnz + 1)
            ja = intArray(A.nnz + 1)
            ar = doubleArray(A.nnz + 1)
            for i, row, col, data in zip(range(A.nnz), A.row, A.col, A.data):
                ia[i + 1] = int(row) + 1
                ja[i + 1] = int(col) + 1
                ar[i + 1] = float(data)
            if A.nnz:
                glp_load_matrix(self.glpk, A.nnz, ia, ja, ar)

        # not sure if the parameter setup is okay
        # LP simplex parameters
        self.lp_params = glp_smcp()
        glp_init_smcp(self.lp_params)
        self.max_tlim = self.lp_params.tm_lim
        self.lp_params.tol_bnd = 1e-9
        self.lp_params.msg_lev = 0
        # MILP parameters
        if self.ismilp:
            self.milp_params = glp_iocp()
            glp_init_iocp(self.milp_params)
            self.milp_params.presolve = 1
            self.milp_params.tol_int = 1e-12
            self.milp_params.tol_obj = 1e-9
            self.milp_params.msg_lev = 0

        # ideally, one would generate random seeds here, but glpk does not seem to
        # offer this function

    def solve(self) -> Tuple[List, float, float]:
        """Solve the MILP or LP
        
        Example:
            sol_x, optim, status = glpk.solve()
        
        Returns:
            (Tuple[List, float, float])
            
            solution_vector, optimal_value, optimization_status
        """
        try:
            min_cx, status = self.solve_MILP_LP()
            if status in [OPTIMAL, TIME_LIMIT_W_SOL]:
                x = self.getSolution(status)
                x = [round(y, 12) for y in x]  # workaround, round to 12 decimals
                return x, round(min_cx, 12), status
            x = [nan] * glp_get_num_cols(self.glpk)
            min_cx = -inf if status == UNBOUNDED else nan
            return x, min_cx, status
        except:
            logging.error('Error while running GLPK.')
            min_cx = nan
            x = [nan] * glp_get_num_cols(self.glpk)
            return x, min_cx, ERROR

    def slim_solve(self) -> float:
        """Solve the MILP or LP, but return only the optimal value
                
        Example:
            optim = glpk.slim_solve()
        
        Returns:
            (float)
            
            Optimum value of the objective function.
        """
        try:
            opt, status = self.solve_MILP_LP()
            if status == OPTIMAL:
                return round(opt, 12)  # workaround, round to 12 decimals
            return -inf if status == UNBOUNDED else nan
        except:
            logging.error('Error while running GLPK.')
            return nan

    def populate(self, pool_limit) -> Tuple[List, float, float]:
        """Generate a solution pool for MILPs
        
        This is only a high-level implementation of the populate function.
        There is no native support in GLPK.
                
        Example:
            sols_x, optim, status = glpk.populate()
        
        Returns:
            (Tuple[List of lists, float, float])
            
            solution_vectors, optimal_value, optimization_status
        """
        numvars = glp_get_num_cols(self.glpk)
        numrows = glp_get_num_rows(self.glpk)
        try:
            if pool_limit > 0:
                sols = []
                stoptime = glp_time() + self.milp_params.tm_lim * 1000
                # 1. find optimal solution
                self.set_time_limit(glp_difftime(stoptime, glp_time()))
                x, min_cx, status = self.solve()
                if status not in [OPTIMAL, UNBOUNDED]:
                    return sols, min_cx, status
                sols = [x]
                # 2. constrain problem to optimality
                c = [glp_get_obj_coef(self.glpk, i + 1) for i in range(numvars)]
                self.add_ineq_constraints(sparse.csr_matrix(c), [min_cx])
                # 3. exclude first solution pool
                self.addExclusionConstraintsIneq(x)
                # 4. loop solve and exclude until problem becomes infeasible
                while status in [OPTIMAL,UNBOUNDED] and not isnan(x[0]) \
                  and glp_difftime(stoptime,glp_time()) > 0 and pool_limit > len(sols):
                    self.set_time_limit(glp_difftime(stoptime, glp_time()))
                    x, _, status = self.solve()
                    if status in [OPTIMAL, UNBOUNDED]:
                        self.addExclusionConstraintsIneq(x)
                        sols += [x]
                if glp_difftime(stoptime, glp_time()) < 0:
                    status = TIME_LIMIT_W_SOL
                elif status == INFEASIBLE:
                    status = OPTIMAL
                # 5. remove auxiliary constraints
                # Here, we only free the upper bound of the constraints
                totrows = glp_get_num_rows(self.glpk)
                for j in range(numrows, totrows):
                    self.set_ineq_constraint(j, [0] * numvars, inf)
                # Alternatively rows may be deleted, but this seems to be very unstable
                # delrows = intArray(totrows-numrows)
                # for i,j in range(numrows,totrows):
                # delrows[i+1] = j+1
                # glp_del_rows(self.glpk,totrows-numrows,delrows)
                return sols, min_cx, status
        except:
            logging.error('Error while running GLPK.')
            x = []
            min_cx = nan
            return x, min_cx, ERROR

    def set_objective(self, c):
        """Set the objective function with a vector"""
        for i, c_i in enumerate(c):
            glp_set_obj_coef(self.glpk, i + 1, float(c_i))

    def set_objective_idx(self, C):
        """Set the objective function with index-value pairs
        
        e.g.: C=[[1, 1.0], [4,-0.2]]"""
        for c in C:
            glp_set_obj_coef(self.glpk, c[0] + 1, float(c[1]))

    def set_ub(self, ub):
        """Set the upper bounds with index-value pairs, e.g.: ub=[[1, 0.0], [4, inf]]"""
        for i, u in ub:
            self._set_col_bounds(i, self._col_bounds(i)[0], float(u))

    def set_lb(self, lb):
        """Set the lower bounds with index-value pairs, e.g.: lb=[[1, 0.0], [4, -inf]]"""
        for i, l in lb:
            self._set_col_bounds(i, float(l), self._col_bounds(i)[1])

    def _col_bounds(self, i):
        # GLPK reports a missing bound as 0 or +-DBL_MAX, so read the bounds through the column type
        t = glp_get_col_type(self.glpk, i + 1)
        lb = -inf if t in [GLP_FR, GLP_UP] else glp_get_col_lb(self.glpk, i + 1)
        ub = inf if t in [GLP_FR, GLP_LO] else glp_get_col_ub(self.glpk, i + 1)
        return lb, ub

    def _set_col_bounds(self, i, lb, ub):
        # the column type follows from which bounds are finite; GLPK keeps the basis across the change
        if isinf(lb) and isinf(ub):
            t = GLP_FR
        elif isinf(ub):
            t = GLP_LO
        elif isinf(lb):
            t = GLP_UP
        elif lb == ub:
            t = GLP_FX
        else:
            t = GLP_DB
        glp_set_col_bnds(self.glpk, i + 1, t, 0.0 if isinf(lb) else lb, 0.0 if isinf(ub) else ub)

    def set_lp_method(self, method):
        """Set the LP solving method.

        Args:
            method: LP_METHOD_AUTO, LP_METHOD_PRIMAL, LP_METHOD_DUAL, or LP_METHOD_BARRIER

        Note: GLPK does not support barrier. LP_METHOD_BARRIER falls back to dual.
        """
        # GLPK meth: 1=primal, 2=dual, 3=dual+pricing
        if method == LP_METHOD_BARRIER:
            logging.warning('GLPK does not support barrier method, falling back to dual simplex.')
        _map = {LP_METHOD_AUTO: 1, LP_METHOD_PRIMAL: 1, LP_METHOD_DUAL: 2, LP_METHOD_BARRIER: 2}
        self.lp_params.meth = _map.get(method, 1)

    def get_lp_method(self):
        """Return the current LP method as a solver-neutral string."""
        _rmap = {1: LP_METHOD_PRIMAL, 2: LP_METHOD_DUAL, 3: LP_METHOD_DUAL}
        return _rmap.get(self.lp_params.meth, LP_METHOD_AUTO)

    def get_basis(self):
        """Return the current LP basis (column and row statuses).

        Returns:
            dict with 'vbasis' (list of int) and 'cbasis' (list of int).
            GLPK codes: 1=basic, 2=at lb, 3=at ub, 4=free, 5=fixed.
        """
        ncols = glp_get_num_cols(self.glpk)
        nrows = glp_get_num_rows(self.glpk)
        vbasis = [glp_get_col_stat(self.glpk, j + 1) for j in range(ncols)]
        cbasis = [glp_get_row_stat(self.glpk, i + 1) for i in range(nrows)]
        return {'vbasis': vbasis, 'cbasis': cbasis}

    def set_basis(self, basis):
        """Load a previously saved basis for warm-starting.

        Args:
            basis: dict from get_basis() with 'vbasis' and 'cbasis'.
        """
        for j, s in enumerate(basis['vbasis']):
            glp_set_col_stat(self.glpk, j + 1, s)
        for i, s in enumerate(basis['cbasis']):
            glp_set_row_stat(self.glpk, i + 1, s)

    def set_time_limit(self, t):
        """Set the computation time limit (in seconds)"""
        if t * 1000 > self.max_tlim:
            if self.ismilp:
                self.milp_params.tm_lim = self.max_tlim
            self.lp_params.tm_lim = self.max_tlim
        else:
            if self.ismilp:
                self.milp_params.tm_lim = int(t * 1000)
            self.lp_params.tm_lim = int(t * 1000)

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
        numvars = glp_get_num_cols(self.glpk)
        numrows = glp_get_num_rows(self.glpk)
        num_newrows = A_ineq.shape[0]
        col = intArray(numvars + 1)
        val = doubleArray(numvars + 1)
        glp_add_rows(self.glpk, num_newrows)
        for j in range(num_newrows):
            for i, v in enumerate(A_ineq[j].toarray()[0]):
                col[i + 1] = i + 1
                val[i + 1] = float(v)
            glp_set_mat_row(self.glpk, numrows + j + 1, numvars, col, val)
            if isinf(b_ineq[j]):
                glp_set_row_bnds(self.glpk, numrows + j + 1, GLP_FR, -inf, float(b_ineq[j]))
            else:
                glp_set_row_bnds(self.glpk, numrows + j + 1, GLP_UP, -inf, float(b_ineq[j]))

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
        numvars = glp_get_num_cols(self.glpk)
        numrows = glp_get_num_rows(self.glpk)
        num_newrows = A_eq.shape[0]
        col = intArray(numvars + 1)
        val = doubleArray(numvars + 1)
        glp_add_rows(self.glpk, num_newrows)
        for j in range(num_newrows):
            for i, v in enumerate(A_eq[j].toarray()[0]):
                col[i + 1] = i + 1
                val[i + 1] = float(v)
            glp_set_mat_row(self.glpk, numrows + j + 1, numvars, col, val)
            glp_set_row_bnds(self.glpk, numrows + j + 1, GLP_FX, float(b_eq[j]), float(b_eq[j]))

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
        numvars = glp_get_num_cols(self.glpk)
        col = intArray(numvars + 1)
        val = doubleArray(numvars + 1)
        for i, v in enumerate(a_ineq):
            col[i + 1] = i + 1
            val[i + 1] = float(v)
        glp_set_mat_row(self.glpk, idx + 1, numvars, col, val)
        if isinf(b_ineq):
            glp_set_row_bnds(self.glpk, idx + 1, GLP_FR, -inf, float(b_ineq))
        else:
            glp_set_row_bnds(self.glpk, idx + 1, GLP_UP, -inf, float(b_ineq))

    def getSolution(self, status) -> list:
        """Retrieve solution from GLPK backend"""
        if self.ismilp and status in [OPTIMAL, UNBOUNDED, TIME_LIMIT_W_SOL]:
            x = [glp_mip_col_val(self.glpk, i + 1) for i in range(glp_get_num_cols(self.glpk))]
        else:
            x = [glp_get_col_prim(self.glpk, i + 1) for i in range(glp_get_num_cols(self.glpk))]
        return x

    def solve_MILP_LP(self) -> Tuple[float, str]:
        """Trigger GLPK solution through backend

        Returns:
            (Tuple[float, str])

            objective value (of the incumbent for a MILP), solver-neutral status
        """
        # MILP solving needs prior solution of the LP-relaxed problem, because occasionally
        # the MILP solver interface crashes when a problem is infesible, which, in turn,
        # crashes the python program. This connection-loss to the solver can not be captured.
        status = self._solve_lp()
        if not self.ismilp:
            if status == ERROR:
                logging.error('GLPK LP: no proof of optimality or infeasibility, also from a fresh basis and presolved.')
            return glp_get_obj_val(self.glpk), status
        if status in [INFEASIBLE, TIME_LIMIT, TIME_LIMIT_W_SOL]:
            # an infeasible relaxation proves the MILP infeasible; at the time limit, no integer
            # solution exists yet. On any other outcome, glp_intopt presolves and solves the
            # relaxation itself.
            return nan, INFEASIBLE if status == INFEASIBLE else TIME_LIMIT
        ret = glp_intopt(self.glpk, self.milp_params)
        mip_status = glp_mip_status(self.glpk)
        has_sol = mip_status in [GLP_OPT, GLP_FEAS]
        if ret in [0, GLP_EMIPGAP] and has_sol:
            status = OPTIMAL
        elif ret == 0 and mip_status == GLP_NOFEAS or ret == GLP_ENOPFS:
            status = INFEASIBLE
        elif ret == GLP_ENODFS:  # relaxation infeasible or unbounded
            status = UNBOUNDED
        elif ret in [GLP_ETMLIM, GLP_ESTOP]:
            status = TIME_LIMIT_W_SOL if has_sol else TIME_LIMIT
        elif has_sol:
            # the search failed, but the incumbent is integer feasible
            logging.warning('GLPK MILP search ended with return code ' + str(ret) + '; returning the incumbent.')
            status = TIME_LIMIT_W_SOL
        else:
            logging.error('GLPK MILP search ended with return code ' + str(ret) + ' and no solution.')
            status = ERROR
        return glp_mip_obj_val(self.glpk), status

    def _lp_status(self, ret):
        """Solver-neutral status of the last glp_simplex call with return code ret.

        glp_get_status reports GLP_INFEAS or GLP_UNDEF for a basis that is primal infeasible or
        missing, e.g. after a numerical failure or when the dual simplex proves the dual
        infeasible (the primal is then infeasible or unbounded). Neither is a proof of
        infeasibility, so both are ERROR here; only GLP_NOFEAS and the presolver's GLP_ENOPFS
        are INFEASIBLE."""
        if ret == 0:
            status = glp_get_status(self.glpk)
            if status == GLP_OPT:
                return OPTIMAL
            if status == GLP_NOFEAS:
                return INFEASIBLE
            if status == GLP_UNBND:
                return UNBOUNDED
        elif ret == GLP_ETMLIM:
            return TIME_LIMIT_W_SOL if glp_get_prim_stat(self.glpk) == GLP_FEAS else TIME_LIMIT
        elif ret == GLP_ENOPFS:
            return INFEASIBLE
        elif ret == GLP_ENODFS:  # infeasible or unbounded
            return UNBOUNDED
        return ERROR

    def _solve_lp(self):
        """Solve the LP (relaxation) and return a solver-neutral status.

        A solve starts from the basis of the previous one. When that ends without a proof (a
        singular or stalled basis, GLP_EFAIL, or a basis that is only known to be primal
        infeasible), the LP is solved again with the primal simplex from a fresh advanced basis,
        and, if that fails too, with the presolver, which builds its own basis. GLPK has feasible
        LPs that fail initially but complete when presolved.

        GLP_NOFEAS from the dual simplex goes through the same retry: warm-started from a
        neighbouring optimal basis, it can end with GLP_NOFEAS on a feasible LP (iML1515 flux
        coupling: 6 of about 1150 LPs), so only the primal simplex's verdict counts as proof."""
        meth, presolve = self.lp_params.meth, self.lp_params.presolve
        status = self._lp_status(glp_simplex(self.glpk, self.lp_params))
        if status != ERROR and not (status == INFEASIBLE and meth != GLP_PRIMAL and presolve == GLP_OFF):
            return status
        try:
            logging.debug('GLPK LP: ' + status + ' from ' + ('primal' if meth == GLP_PRIMAL else 'dual') +
                          ' simplex, re-solving with the primal simplex from a fresh basis.')
            term_out = glp_term_out(GLP_OFF)  # glp_adv_basis reports to the terminal regardless of msg_lev
            glp_adv_basis(self.glpk, 0)
            glp_term_out(term_out)
            self.lp_params.meth, self.lp_params.presolve = GLP_PRIMAL, GLP_OFF
            status = self._lp_status(glp_simplex(self.glpk, self.lp_params))
            if status == ERROR:
                logging.debug('GLPK LP: fresh basis failed, re-solving with presolve.')
                self.lp_params.meth, self.lp_params.presolve = GLP_DUALP, GLP_ON
                status = self._lp_status(glp_simplex(self.glpk, self.lp_params))
        finally:
            self.lp_params.meth, self.lp_params.presolve = meth, presolve
        return status

    def addExclusionConstraintsIneq(self, x):
        """Function to add exclusion constraint (GLPK compatibility function)"""
        numvars = glp_get_num_cols(self.glpk)
        # Here, we also need to take integer variables into account, because GLPK changes
        # variable type to integer when you lock a binary variable to zero
        binvars = [i for i in range(numvars) if glp_get_col_kind(self.glpk, i + 1) in [GLP_BV, GLP_IV]]
        data = [1.0 if x[i] else -1.0 for i in binvars]
        row = [0] * len(binvars)
        A_ineq = sparse.csr_matrix((data, (row, binvars)), (1, numvars))
        b_ineq = sum([x[i] for i in binvars]) - 1
        self.add_ineq_constraints(A_ineq, [b_ineq])
