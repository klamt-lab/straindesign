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
"""Unified solver interface for LPs and MILPs (MILP_LP)"""

from numpy import inf, isinf, isnan, unique
from scipy import sparse
from typing import List, Tuple
from straindesign import avail_solvers, GLPK
from straindesign.names import *
import logging


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
            'c', 'A_ineq', 'b_ineq', 'A_eq', 'b_eq', 'lb', 'ub', 'vtype', 'indic_constr', 'M', 'solver', 'skip_checks', 'tlim', SEED
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
        # Select solver (either by choice or automatically cplex > gurobi > glpk)
        if self.solver is None:
            if len(avail_solvers) > 0:
                self.solver = list(avail_solvers)[0]
            else:
                raise Exception('No solver available. Please ensure that one of the following '\
                    'solvers is avaialable in your Python environment: CPLEX, Gurobi, SCIP, GLPK')
        elif self.solver not in avail_solvers:
            raise Exception("Selected solver '" + self.solver + "' is not installed / set up correctly.")
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
        # Create backend
        if self.solver == CPLEX:
            from straindesign.cplex_interface import Cplex_MILP_LP
            self.backend = Cplex_MILP_LP(self.c, self.A_ineq, self.b_ineq, self.A_eq, self.b_eq, self.lb, self.ub, self.vtype,
                                         self.indic_constr, self.seed)
        elif self.solver == GUROBI:
            from straindesign.gurobi_interface import Gurobi_MILP_LP
            self.backend = Gurobi_MILP_LP(self.c, self.A_ineq, self.b_ineq, self.A_eq, self.b_eq, self.lb, self.ub, self.vtype,
                                          self.indic_constr, self.seed)
        elif self.solver == SCIP:
            from straindesign.scip_interface import SCIP_MILP, SCIP_LP
            self.isLP = all(v == 'C' for v in self.vtype)
            if self.isLP:
                self.backend = SCIP_LP(self.c, self.A_ineq, self.b_ineq, self.A_eq, self.b_eq, self.lb, self.ub)
                return
            else:
                self.backend = SCIP_MILP(self.c, self.A_ineq, self.b_ineq, self.A_eq, self.b_eq, self.lb, self.ub, self.vtype,
                                         self.indic_constr, self.seed)
        elif self.solver == GLPK:
            from straindesign.glpk_interface import GLPK_MILP_LP
            self.backend = GLPK_MILP_LP(self.c, self.A_ineq, self.b_ineq, self.A_eq, self.b_eq, self.lb, self.ub, self.vtype,
                                        self.indic_constr, self.M)
        if self.tlim is None:
            self.set_time_limit(inf)
        else:
            self.set_time_limit(self.tlim)

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
            x = [x[i] if self.vtype[i] == 'C' else int(round(x[i])) for i in range(len(x))]
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

    def set_time_limit(self, t):
        """Set the computation time limit (in seconds)"""
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

    def clear_objective(self):
        """Clear objective
        
        Set all coefficients in the objective vector to 0."""
        self.set_objective([0.0] * len(self.c))
