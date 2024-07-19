#!/usr/bin/env python3
#
# Copyright 2022 Max Planck Insitute Magdeburg
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
"""CPLEX solver interface for LP and MILP"""

from scipy import sparse
from numpy import nan, inf, isinf, random
from cplex import Cplex, infinity, _const
from cplex.exceptions import CplexError
from typing import Tuple, List
import logging
import io
from psutil import virtual_memory
from straindesign.names import *


class Cplex_MILP_LP(Cplex):
    """CPLEX interface for MILP and LP
    
    This class is a wrapper for the CPLEX-Python API to offer bindings and namings
    for functions for the construction and manipulation of MILPs and LPs in an
    vector-matrix-based manner that are consistent with those of the other solver 
    interfaces in the StrainDesign package. The purpose is to unify the instructions 
    for operating with MILPs and LPs throughout StrainDesign.
    
    The CPLEX interface provides support for indicator constraints as well as for
    the populate function.
    
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
        cplex = Cplex_MILP_LP(c, A_ineq, b_ineq, A_eq, b_eq, lb, ub, vtype, indic_constr)
                
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
        
        seed (int16): (Default: None)
            An integer value serving as a seed to make MILP solving reproducible.
        
        Returns:
            (Cplex_MILP_LP):
            
            A CPLEX MILP/LP interface class.
    """

    def __init__(self, c=None, A_ineq=None, b_ineq=None, A_eq=None, b_eq=None, lb=None, ub=None, vtype=None, indic_constr=None, seed=None):
        super().__init__()
        self.objective.set_sense(self.objective.sense.minimize)
        try:
            numvars = A_ineq.shape[1]
        except:
            numvars = A_eq.shape[1]
        # replace numpy inf with cplex infinity
        for i, v in enumerate(b_ineq):
            if isinf(v):
                b_ineq[i] = infinity
        # concatenate right hand sides
        b = b_ineq + b_eq
        # prepare coefficient matrix
        if isinstance(A_eq, list):
            if not A_eq:
                A_eq = sparse.csr_matrix((0, numvars))
        if isinstance(A_ineq, list):
            if not A_ineq:
                A_ineq = sparse.csr_matrix((0, numvars))
        A = sparse.vstack((A_ineq, A_eq), format='coo')  # concatenate coefficient matrices
        sense = len(b_ineq) * 'L' + len(b_eq) * 'E'

        # construct CPLEX problem. Add variables and linear constraints
        self.variables.add(obj=c, lb=lb, ub=ub, types=vtype)
        self.linear_constraints.add(rhs=b, senses=sense)
        if A.nnz:
            self.linear_constraints.set_coefficients(zip(A.row.tolist(), A.col.tolist(), A.data.tolist()))

        # add indicator constraints
        if not indic_constr == None:
            # cast variables and translate coefficient matrix A to right input format for CPLEX
            A = [[[int(i) for i in list(a.indices)], [float(i) for i in list(a.data)]] for a in sparse.csr_matrix(indic_constr.A)]
            b = [float(i) for i in indic_constr.b]
            sense = [str(i) for i in indic_constr.sense]
            indvar = [int(i) for i in indic_constr.binv]
            complem = [1 - int(i) for i in indic_constr.indicval]
            # call CPLEX function to add indicators
            self.indicator_constraints.add_batch(lin_expr=A, sense=sense, rhs=b, indvar=indvar, complemented=complem)
        # set parameters
        self.set_log_stream(io.StringIO())  # don't show output stream
        self.set_error_stream(io.StringIO())
        self.set_warning_stream(io.StringIO())
        self.set_results_stream(io.StringIO())
        self.parameters.simplex.tolerances.optimality.set(1e-9)
        self.parameters.simplex.tolerances.feasibility.set(1e-9)

        if 'B' in vtype or 'I' in vtype:
            # set usable working memory to 3/4 of the total available memory
            self.parameters.workmem.set(round(virtual_memory().total / 1024 / 1024 * 0.75))
            #self.parameters.threads.set(16)
            # yield only optimal solutions in pool
            if seed is None:
                # seed = random.randint(0, _const.CPX_BIGINT)
                seed = int(random.randint(0, 2**16 - 1))
            self.parameters.randomseed.set(seed)
            self.parameters.mip.pool.absgap.set(0.0)
            self.parameters.mip.pool.relgap.set(0.0)
            self.parameters.mip.pool.intensity.set(4)
            # no integrality tolerance
            self.parameters.mip.tolerances.integrality.set(0.0)

    def solve(self) -> Tuple[List, float, float]:
        """Solve the MILP or LP
        
        Example:
            sol_x, optim, status = cplex.solve()
        
        Returns:
            (Tuple[List, float, float])
            
            solution_vector, optimal_value, optimization_status
        """
        try:
            super().solve()  # call parent solve function (that was overwritten in this class)
            status = self.solution.get_status()
            if status in [1, 101, 102, 115, 128, 129, 130]:  # solution integer optimal
                min_cx = self.solution.get_objective_value()
                status = OPTIMAL
            elif status == 108:  # timeout without solution
                x = [nan] * self.variables.get_num()
                min_cx = nan
                status = TIME_LIMIT
                return x, min_cx, status
            elif status in [3, 103]:  # infeasible
                x = [nan] * self.variables.get_num()
                min_cx = nan
                status = INFEASIBLE
                return x, min_cx, status
            elif status in [11, 107]:  # timeout with solution
                min_cx = self.solution.get_objective_value()
                status = TIME_LIMIT_W_SOL
            elif status in [2, 4, 118, 119]:  # solution unbounded
                x = [nan] * self.variables.get_num()
                min_cx = -inf
                status = UNBOUNDED
                return x, min_cx, status
            else:
                logging.exception(status)
                logging.exception(self.solution.get_status_string())
                raise Exception("Case not yet handeld")
            x = self.solution.get_values()
            return x, min_cx, status

        except CplexError as exc:
            if not exc.args[2] == 1217:
                logging.error(exc)
            min_cx = nan
            x = [nan] * self.variables.get_num()
            return x, min_cx, ERROR

    def slim_solve(self) -> float:
        """Solve the MILP or LP, but return only the optimal value
                
        Example:
            optim = cplex.slim_solve()
        
        Returns:
            (float)
            
            Optimum value of the objective function.
        """
        try:
            super().solve()  # call parent solve function (that was overwritten in this class)
            status = self.solution.get_status()
            if status in [1, 101, 102, 107, 115, 128, 129, 130]:  # solution integer optimal (tolerance)
                opt = self.solution.get_objective_value()
            elif status in [118, 119]:  # solution unbounded (or inf or unbdd)
                opt = -inf
            elif status in [103, 108]:  # infeasible
                opt = nan
            else:
                logging.exception(status)
                logging.exception(self.solution.get_status_string())
                raise Exception("Case not yet handeld")
            return opt
        except CplexError as exc:
            return nan

    def populate(self, n) -> Tuple[List, float, float]:
        """Generate a solution pool for MILPs
                
        Example:
            sols_x, optim, status = cplex.populate()
        
        Returns:
            (Tuple[List of lists, float, float])
            
            solution_vectors, optimal_value, optimization_status
        """
        try:
            if isinf(n):
                self.parameters.mip.pool.capacity.set(self.parameters.mip.pool.capacity.max())
            else:
                self.parameters.mip.pool.capacity.set(n)
            self.populate_solution_pool()  # call parent solve function (that was overwritten in this class)
            status = self.solution.get_status()
            if status in [101, 102, 115, 128, 129, 130]:  # solution integer optimal
                min_cx = self.solution.get_objective_value()
                status = OPTIMAL
            elif status == 108:  # timeout without solution
                x = []
                min_cx = nan
                status = TIME_LIMIT
                return x, min_cx, status
            elif status == 103:  # infeasible
                x = []
                min_cx = nan
                status = INFEASIBLE
                return x, min_cx, status
            elif status == 107:  # timeout with solution
                min_cx = self.solution.get_objective_value()
                status = TIME_LIMIT_W_SOL
            elif status in [118, 119]:  # solution unbounded
                min_cx = -inf
                status = UNBOUNDED
            else:
                logging.exception(status)
                logging.exception(self.solution.get_status_string())
                raise Exception("Case not yet handeld")
            x = [self.solution.pool.get_values(i) for i in range(self.solution.pool.get_num())]
            return x, min_cx, status

        except CplexError as exc:
            if not exc.args[2] == 1217:
                logging.error(exc)
            min_cx = nan
            x = []
            return x, min_cx, ERROR

    def set_objective(self, c):
        """Set the objective function with a vector"""
        self.objective.set_linear([[i, c[i]] for i in range(len(c))])

    def set_objective_idx(self, C):
        """Set the objective function with index-value pairs
        
        e.g.: C=[[1, 1.0], [4,-0.2]]"""
        self.objective.set_linear(C)

    def set_ub(self, ub):
        """Set the upper bounds to a given vector"""
        self.variables.set_upper_bounds(ub)

    def set_time_limit(self, t):
        """Set the computation time limit (in seconds)"""
        if isinf(t):
            self.parameters.timelimit.set(self.parameters.timelimit.max())
        else:
            self.parameters.timelimit.set(t)

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
        numconst = self.linear_constraints.get_num()
        numnewconst = A_ineq.shape[0]
        newconst_idx = [numconst + i for i in range(numnewconst)]
        for i, v in enumerate(b_ineq):
            if isinf(v):
                b_ineq[i] = infinity
        self.linear_constraints.add(rhs=b_ineq, senses='L' * numnewconst)
        # retrieve row and column indices from sparse matrix and convert them to int
        A_ineq = A_ineq.tocoo()
        rows_A = [int(a) + numconst for a in A_ineq.row]
        cols_A = [int(a) for a in A_ineq.col]
        # convert matrix coefficients to float
        data_A = [float(a) for a in A_ineq.data]
        self.linear_constraints.set_coefficients(zip(rows_A, cols_A, data_A))

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
        numconst = self.linear_constraints.get_num()
        numnewconst = A_eq.shape[0]
        self.linear_constraints.add(rhs=b_eq, senses='E' * numnewconst)
        # retrieve row and column indices from sparse matrix and convert them to int
        A_eq = A_eq.tocoo()
        rows_A = [int(a) + numconst for a in A_eq.row]
        cols_A = [int(a) for a in A_eq.col]
        # convert matrix coefficients to float
        data_A = [float(a) for a in A_eq.data]
        self.linear_constraints.set_coefficients(zip(rows_A, cols_A, data_A))

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
        if isinf(b_ineq):
            b_ineq = infinity
        self.linear_constraints.set_coefficients(zip([idx] * len(a_ineq), range(len(a_ineq)), a_ineq))
        self.linear_constraints.set_rhs([[idx, b_ineq]])
