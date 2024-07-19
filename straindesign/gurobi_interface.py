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
"""Gurobi solver interface for LP and MILP"""

from scipy import sparse
from numpy import nan, inf, isinf, sum, array, random
import gurobipy as gp
from gurobipy import GRB as grb
from straindesign.names import *
from typing import Tuple, List
import logging

gstatus = gp.StatusConstClass


class Gurobi_MILP_LP(gp.Model):
    """Gurobi interface for MILP and LP
    
    This class is a wrapper for the Gurobi-Python API to offer bindings and namings
    for functions for the construction and manipulation of MILPs and LPs in an
    vector-matrix-based manner that are consistent with those of the other solver 
    interfaces in the StrainDesign package. The purpose is to unify the instructions 
    for operating with MILPs and LPs throughout StrainDesign.
    
    The Gurobi interface provides support for indicator constraints as well as for
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
        gurobi = Gurobi_MILP_LP(c, A_ineq, b_ineq, A_eq, b_eq, lb, ub, vtype, indic_constr)
                
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
            (Gurobi_MILP_LP):
            
            A Gurobi MILP/LP interface class.
    """

    def __init__(self, c=None, A_ineq=None, b_ineq=None, A_eq=None, b_eq=None, lb=None, ub=None, vtype=None, indic_constr=None, seed=None):
        super().__init__()
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

        for i, v in enumerate(b_ineq):
            if isinf(v):
                b_ineq[i] = grb.INFINITY
        # concatenate right hand sides
        # construct Gurobi problem. Add variables and linear constraints
        x = self.addMVar(len(c), obj=c, lb=lb, ub=ub, vtype=[k for k in vtype])
        self.setObjective(array(c) @ x, grb.MINIMIZE)
        if A_ineq.shape[0]:
            self.addMConstr(A_ineq, x, grb.LESS_EQUAL, array(b_ineq))
        if A_eq.shape[0]:
            self.addMConstr(A_eq, x, grb.EQUAL, array(b_eq))

        # add indicator constraints
        if not indic_constr == None:
            for i in range(len(indic_constr.sense)):
                self.addGenConstrIndicator(x[indic_constr.binv[i]],
                                        bool(indic_constr.indicval[i]),
                                        sum([indic_constr.A[i,j] * x[j] \
                                            for j in range(len(c)) if not indic_constr.A[i,j] == 0.0]),
                                        '=' if indic_constr.sense[i] =='E' else '<',
                                        indic_constr.b[i])

        # set parameters
        self.params.OutputFlag = 0
        self.params.OptimalityTol = 1e-9
        self.params.FeasibilityTol = 1e-9
        if 'B' in vtype or 'I' in vtype:
            if seed is None:
                # seed = random(0, grb.MAXINT)
                seed = int(random.randint(0, 2**16 - 1))
                logging.info('  MILP Seed: ' + str(seed))
            self.params.Seed = seed
            self.params.IntFeasTol = 1e-9  # (0 is not allowed by Gurobi)
            # yield only optimal solutions in pool
            self.params.PoolGap = 1e-9
            self.params.PoolGapAbs = 1e-9
            self.params.MIPFocus = 0
        self.update()

    def solve(self) -> Tuple[List, float, float]:
        """Solve the MILP or LP
        
        Example:
            sol_x, optim, status = gurobi.solve()
        
        Returns:
            (Tuple[List, float, float])
            
            solution_vector, optimal_value, optimization_status
        """
        try:
            self.optimize()  # call parent solve function (that was overwritten in this class)
            status = self.Status
            if status in [gstatus.OPTIMAL, gstatus.SOLUTION_LIMIT, gstatus.SUBOPTIMAL, gstatus.USER_OBJ_LIMIT]:  # solution
                min_cx = self.ObjVal
                status = OPTIMAL
            elif status == gstatus.TIME_LIMIT and not hasattr(self._Model__vars[0], 'X'):  # timeout without solution
                x = [nan] * self.NumVars
                min_cx = nan
                status = TIME_LIMIT
                return x, min_cx, status
            elif status == gstatus.TIME_LIMIT and hasattr(self._Model__vars[0], 'X'):
                min_cx = self.ObjVal
                status = TIME_LIMIT_W_SOL
            elif status in [gstatus.INF_OR_UNBD, gstatus.UNBOUNDED, gstatus.INFEASIBLE]:
                # solve problem again without objective to verify that problem is feasible
                self.params.DualReductions = 0
                self.optimize()
                self.params.DualReductions = 1
                if self.Status == gstatus.INFEASIBLE:
                    x = [nan] * self.NumVars
                    min_cx = nan
                    status = INFEASIBLE
                    return x, min_cx, status
                else:
                    x = [nan] * self.NumVars
                    min_cx = -inf
                    status = UNBOUNDED
                    return x, min_cx, status
            else:
                raise Exception('Status code ' + str(status) + " not yet handeld.")
            x = self.getSolution()
            return x, min_cx, status

        except gp.GurobiError as e:
            logging.error('Error code ' + str(e.errno) + ": " + str(e))
            min_cx = nan
            x = [nan] * self.NumVars
            return x, min_cx, ERROR

    def slim_solve(self) -> float:
        """Solve the MILP or LP, but return only the optimal value
                
        Example:
            optim = gurobi.slim_solve()
        
        Returns:
            (float)
            
            Optimum value of the objective function.
        """
        try:
            self.optimize()  # call parent solve function (that was overwritten in this class)
            status = self.Status
            if status in [gstatus.OPTIMAL, gstatus.SOLUTION_LIMIT, gstatus.SUBOPTIMAL,
                          gstatus.USER_OBJ_LIMIT]:  # solution integer optimal (tolerance)
                opt = self.ObjVal
            elif status in [gstatus.INF_OR_UNBD, gstatus.UNBOUNDED]:
                opt = -inf
            elif status in [gstatus.INFEASIBLE, gstatus.TIME_LIMIT]:
                opt = nan
            else:
                raise Exception('Status code ' + str(status) + " not yet handeld.")
            return opt
        except gp.GurobiError as e:
            logging.error('Error code ' + str(e.errno) + ": " + str(e))
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
                self.params.PoolSolutions = grb.MAXINT
            else:
                self.params.PoolSolutions = n
            self.params.PoolSearchMode = 2
            self.params.NumericFocus = 2
            self.optimize()  # call parent solve function (that was overwritten in this class)
            self.params.PoolSearchMode = 0
            self.params.NumericFocus = 0
            status = self.Status
            if status in [2, 10, 13, 15]:  # solution integer optimal
                min_cx = self.ObjVal
                status = OPTIMAL
            elif status == 9 and not hasattr(self._Model__vars[0], 'X'):  # timeout without solution
                x = [nan] * len(self._Model__vars)
                min_cx = nan
                status = TIME_LIMIT
                return x, min_cx, status
            elif status == 3:  # infeasible
                x = [nan] * len(self._Model__vars)
                min_cx = nan
                status = INFEASIBLE
                return x, min_cx, status
            elif status == 9 and hasattr(self._Model__vars[0], 'X'):  # timeout with solution
                min_cx = self.ObjVal
                status = TIME_LIMIT_W_SOL
            elif status in [4, 5]:  # solution unbounded
                min_cx = -inf
                x = [nan] * len(self._Model__vars)
                status = UNBOUNDED
                return x, min_cx, status
            else:
                raise Exception('Status code ' + str(status) + " not yet handeld.")
            x = self.getSolutions()
            return x, min_cx, status

        except gp.GurobiError as e:
            self.params.PoolSearchMode = 0
            logging.error('Error code ' + str(e.errno) + ": " + str(e))
            min_cx = nan
            x = [nan] * len(self._Model__vars)
            return x, min_cx, ERROR

    def set_objective(self, c):
        """Set the objective function with a vector"""
        for i in range(len(self._Model__vars)):
            self._Model__vars[i].Obj = c[i]
        self.update()
        if any([self._Model__vars[i].Obj for i in range(len(self._Model__vars))]):
            self.params.MIPFocus = 0
        else:
            self.params.MIPFocus = 1

    def set_objective_idx(self, C):
        """Set the objective function with index-value pairs
        
        e.g.: C=[[1, 1.0], [4,-0.2]]"""
        for c in C:
            self._Model__vars[c[0]].Obj = c[1]
        self.update()
        if any([self._Model__vars[i].Obj for i in range(len(self._Model__vars))]):
            self.params.MIPFocus = 0
        else:
            self.params.MIPFocus = 1

    def set_ub(self, ub):
        """Set the upper bounds to a given vector"""
        for i in range(len(ub)):
            self._Model__vars[ub[i][0]].ub = ub[i][1]
        self.update()

    def set_time_limit(self, t):
        """Set the computation time limit (in seconds)"""
        self.params.TimeLimit = t
        self.update()

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
        vars = self._Model__vars
        for i in range(A_ineq.shape[0]):
            self.addLConstr(sum([A_ineq[i, j] * vars[j] for j in range(len(vars)) if not A_ineq[i, j] == 0.0]), grb.LESS_EQUAL, b_ineq[i])
        self.update()

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
        vars = self._Model__vars
        for i in range(A_eq.shape[0]):
            self.addLConstr(sum([A_eq[i, j] * vars[j] for j in range(len(vars)) if not A_eq[i, j] == 0.0]), grb.EQUAL, b_eq[i])
        self.update()

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
        constr = self._Model__constrs[idx]
        [self.chgCoeff(constr, x, val) for x, val in zip(self._Model__vars, a_ineq)]
        if isinf(b_ineq):
            constr.rhs = grb.INFINITY
        else:
            constr.rhs = b_ineq
        self.update()

    def getSolution(self) -> list:
        """Retrieve solution from Gurobi backend"""
        return [x.X for x in self._Model__vars]

    def getSolutions(self) -> list:
        """Retrieve solution pool from Gurobi backend"""
        nSols = self.SolCount
        x = []
        for i in range(nSols):
            self.setParam(grb.Param.SolutionNumber, i)
            if self.PoolObjVal == self.ObjVal:
                x += [[x.Xn for x in self._Model__vars]]
        return x
