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
"""SCIP and SoPlex solver interface for LP and MILP"""

from scipy import sparse
from numpy import isnan, nan, inf, isinf, sum, nonzero, random
import pyscipopt as pso
from straindesign.names import *
from typing import Tuple, List
import time as t
import logging


class SCIP_MILP(pso.Model):
    """SCIP interface for MILP
    
    This class is a wrapper for the SCIP-Python API to offer bindings and namings
    for functions for the construction and manipulation of MILPs in an
    vector-matrix-based manner that are consistent with those of the other solver 
    interfaces in the StrainDesign package. The purpose is to unify the instructions 
    for operating with MILPs and LPs throughout StrainDesign.
    
    The SCIP interface provides support for indicator constraints as well as for
    the populate function. The SCIP interface does not natively support the populate 
    function. A high level implementation emulates the behavior of populate.
    
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
        scip = SCIP_MILP(c, A_ineq, b_ineq, A_eq, b_eq, lb, ub, vtype, indic_constr)
                
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
            (SCIP_MILP):
            
            A SCIP MILP interface class.
    """

    def __init__(self, c=None, A_ineq=None, b_ineq=None, A_eq=None, b_eq=None, lb=None, ub=None, vtype=None, indic_constr=None, seed=None):
        super().__init__()
        # uncomment to forward SCIP output to python terminal
        # self.redirectOutput()
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

        ub = [u if not isinf(u) else None for u in ub]
        lb = [l if not isinf(l) else None for l in lb]
        # add variables and constraints
        x = [self.addVar(lb=l, ub=u, obj=o, vtype=v) for l, u, o, v in zip(lb, ub, c, vtype)]
        self.vars = x
        self.binvars = [i for i in range(len(x)) if vtype[i] == 'B']
        # generate "Terms" and "Expressions" for faster problem construction
        self.trms = [list(x[i].terms.items())[0][0] for i in range(numvars)]

        self.constr = []
        # add inequality constraints
        ineqs = [self.addCons(pso.Expr() <= b_i, modifiable=True) for b_i in b_ineq]
        for row, a_ineq in zip(ineqs, A_ineq):
            X = [x[i] for i in a_ineq.indices]
            for col, coeff in zip(X, a_ineq.data):
                self.addConsCoeff(row, col, coeff)
        self.constr += ineqs
        # add equality constraints
        eqs = [self.addCons(pso.Expr() == b_i, modifiable=True) for b_i in b_eq]
        for row, a_eq in zip(eqs, A_eq):
            X = [x[i] for i in a_eq.indices]
            for col, coeff in zip(X, a_eq.data):
                self.addConsCoeff(row, col, coeff)
        self.constr += eqs

        self.setMinimize()
        # add indicator constraints
        if indic_constr is not None:
            for i in range(len(indic_constr.sense)):
                if indic_constr.indicval[i] == 0:  # if the constraints activity is indicated by 0, an auxiliary variable needs to be added
                    z = self.addVar(lb=0, ub=1, obj=0, vtype='B')
                    xor = self.addCons(pso.Expr() == 1)
                    self.addConsCoeff(xor, x[indic_constr.binv[i]], 1)
                    self.addConsCoeff(xor, z, 1)
                else:
                    z = x[indic_constr.binv[i]]
                if indic_constr.sense[i] == 'E':
                    A = sparse.vstack((indic_constr.A[i], -indic_constr.A[i]))
                    b = [indic_constr.b[i], -indic_constr.b[i]]
                else:
                    A = indic_constr.A[i]
                    b = [indic_constr.b[i]]
                for k in range(A.shape[0]):
                    for a in A[k]:
                        e = pso.scip.Expr({self.trms[j]: d for j, d in zip(a.indices, a.data)})
                        f = pso.scip.ExprCons(e, lhs=None, rhs=b[k])
                        self.constr += [self.addConsIndicator(f, binvar=z, initial=False)]

        # set parameters
        self.max_tlim = self.getParam('limits/time')
        if 'B' in vtype or 'I' in vtype:
            if seed is None:
                # seed = randint(0, 2**31 - 1)
                seed = int(random.randint(2**16 - 1))
                logging.info('  MILP Seed: ' + str(seed))
            self.setParam('randomization/randomseedshift', seed)
            self.setEmphasis(0)
            # self.setParam('constraints/indicator/forcerestart',True)
            # Probably all seeds are set by the randomseedshift??
            # self.setParam('branching/random/seed', seed)
            # self.setParam('branching/relpscost/startrandseed', seed)
            # self.setParam('heuristics/alns/seed', seed)
            # self.setParam('heuristics/scheduler/seed', seed)
            # self.setParam('separating/zerohalf/initseed', seed)
        # self.enableReoptimization()
        # self.setParam('display/lpinfo',False)
        # self.setParam('reoptimization/enable',True)
        self.setParam('display/verblevel', 0)

        # SCIP_PARAMEMPHASIS_DEFAULT     = 0,        /**< use default values */
        # SCIP_PARAMEMPHASIS_CPSOLVER    = 1,        /**< get CP like search (e.g. no LP relaxation) */
        # SCIP_PARAMEMPHASIS_EASYCIP     = 2,        /**< solve easy problems fast */
        # SCIP_PARAMEMPHASIS_FEASIBILITY = 3,        /**< detect feasibility fast */
        # SCIP_PARAMEMPHASIS_HARDLP      = 4,        /**< be capable to handle hard LPs */
        # SCIP_PARAMEMPHASIS_OPTIMALITY  = 5,        /**< prove optimality fast */
        # SCIP_PARAMEMPHASIS_COUNTER     = 6,        /**< get a feasible and "fast" counting process */
        # SCIP_PARAMEMPHASIS_PHASEFEAS   = 7,        /**< feasibility phase settings during 3-phase solving approach */
        # SCIP_PARAMEMPHASIS_PHASEIMPROVE= 8,        /**< improvement phase settings during 3-phase solving approach */
        # SCIP_PARAMEMPHASIS_PHASEPROOF  = 9         /**< proof phase settings during 3-phase solving approach */

    def solve(self) -> Tuple[List, float, float]:
        """Solve the MILP
        
        Example:
            sol_x, optim, status = scip.solve()
        
        Returns:
            (Tuple[List, float, float])
            
            solution_vector, optimal_value, optimization_status
        """
        try:
            self.optimize()
            status = self.getStatus()
            if status in ['optimal']:  # solution
                min_cx = self.getObjVal()
                status = OPTIMAL
            elif status == 'timelimit' and self.getSols() == []:  # timeout without solution
                x = [nan] * len(self.vars)
                min_cx = nan
                status = TIME_LIMIT
                return x, min_cx, status
            elif status == 'infeasible':  # infeasible
                x = [nan] * len(self.vars)
                min_cx = nan
                status = INFEASIBLE
                return x, min_cx, status
            elif status == 'timelimit' and not self.getSols() == []:  # timeout with solution
                min_cx = self.getObjVal()
                status = TIME_LIMIT_W_SOL
            elif status in ['inforunbd', 'unbounded']:  # solution unbounded
                x = [nan] * len(self.vars)
                min_cx = -inf
                status = UNBOUNDED
                return x, min_cx, status
            else:
                raise Exception('Status code ' + str(status) + " not yet handeld.")
            x = self.getSolution()
            return x, min_cx, status

        except:
            logging.error('Error while running SCIP.')
            min_cx = nan
            x = [nan] * len(self.vars)
            return x, min_cx, ERROR

    def slim_solve(self) -> float:
        """Solve the MILP, but return only the optimal value
                
        Example:
            optim = scip.slim_solve()
        
        Returns:
            (float)
            
            Optimum value of the objective function.
        """
        try:
            self.optimize()
            status = self.getStatus()
            if status == 'optimal':  # solution
                opt = self.getObjVal()
            elif status in ['infeasible', 'timelimit']:
                opt = nan
            elif status in ['inforunbd', 'unbounded']:
                opt = -inf
            else:
                raise Exception('Status code ' + str(status) + " not yet handeld.")
            return opt

        except:
            logging.error('Error while running SCIP.')
            return nan

    def populate(self, pool_limit) -> Tuple[List, float, float]:
        numrows = len(self.constr)
        """Generate a solution pool for MILPs
        
        This is only a high-level implementation of the populate function.
        There is no native support in SCIP.
                
        Example:
            sols_x, optim, status = scip.populate()
        
        Returns:
            (Tuple[List of lists, float, float])
            
            solution_vectors, optimal_value, optimization_status
        """
        try:
            if pool_limit > 0:
                sols = []
                stoptime = t.time() + self.getParam('limits/time')
                # 1. find optimal solution
                self.set_time_limit(stoptime - t.time())
                x, min_cx, status = self.solve()
                if status not in [OPTIMAL, UNBOUNDED]:
                    return sols, min_cx, status
                sols = [x]
                # 2. constrain problem to optimality
                objTerms = self.getObjective().terms
                c = [objTerms[x] if x in objTerms.keys() else 0.0 for x in self.trms]
                self.add_ineq_constraints(sparse.csr_matrix(c), [min_cx])
                # 3. exclude first solution pool
                self.addExclusionConstraintIneq(x)
                # 4. loop solve and exclude until problem becomes infeasible
                while status in [OPTIMAL,UNBOUNDED] and not isnan(x[0]) \
                    and stoptime-t.time() > 0 and pool_limit > len(sols):
                    self.set_time_limit(stoptime - t.time())
                    x, _, status = self.solve()
                    if status in [OPTIMAL, UNBOUNDED]:
                        self.addExclusionConstraintIneq(x)
                        sols += [x]
                if stoptime - t.time() < 0:
                    status = TIME_LIMIT_W_SOL
                elif status == INFEASIBLE:
                    status = OPTIMAL
                # 5. remove auxiliary constraints
                # Here, we only free the upper bound of the constraints
                totrows = len(self.constr)
                self.freeTransform()
                for j in range(numrows, totrows):
                    self.chgRhs(self.constr[j], None)
                return sols, min_cx, status
        except:
            logging.error('Error while running SCIP.')
            min_cx = nan
            x = []
            return x, min_cx, ERROR

    def set_objective(self, c):
        """Set the objective function with a vector"""
        if self.getParam('reoptimization/enable'):
            self.freeReoptSolve()
            self.chgReoptObjective(pso.Expr({self.trms[i]: c[i] for i in nonzero(c)[0]}))
        else:
            self.freeTransform()
            self.setObjective(pso.Expr({self.trms[i]: c[i] for i in nonzero(c)[0]}))
        if any(self.getObjective()):
            self.setEmphasis(0)
            self.setParam('display/verblevel', 0)
        else:
            self.setEmphasis(1)

    def set_objective_idx(self, C):
        """Set the objective function with index-value pairs
        
        e.g.: C=[[1, 1.0], [4,-0.2]]"""
        if self.getParam('reoptimization/enable'):
            self.freeReoptSolve()
            self.chgReoptObjective(pso.Expr({self.trms[c[0]]: c[1] for c in C}))
        else:
            self.freeTransform()
            self.setObjective(pso.Expr({self.trms[c[0]]: c[1] for c in C}))
        if any(self.getObjective()):
            self.setEmphasis(0)
            self.setParam('display/verblevel', 0)
        else:
            self.setEmphasis(1)

    def set_ub(self, ub):
        """Set the upper bounds to a given vector"""
        self.freeTransform()
        for i in range(len(ub)):
            if not isinf(ub[i][1]):
                self.chgVarUb(self.vars[ub[i][0]], float(ub[i][1]))
            else:
                self.chgVarUb(self.vars[ub[i][0]], None)

    def set_time_limit(self, t):
        """Set the computation time limit (in seconds)"""
        if t >= self.max_tlim:
            self.setParam('limits/time', self.max_tlim)
        else:
            self.setParam('limits/time', t)

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
        self.freeTransform()
        ineqs = [self.addCons(pso.Expr() <= b_i, modifiable=True) for b_i in b_ineq]
        for row, a_ineq in zip(ineqs, A_ineq):
            X = [self.vars[i] for i in a_ineq.indices]
            for col, coeff in zip(X, a_ineq.data):
                self.addConsCoeff(row, col, float(coeff))
        self.constr += ineqs

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
        self.freeTransform()
        eqs = [self.addCons(pso.Expr() == b_i, modifiable=True) for b_i in b_eq]
        for row, a_eq in zip(eqs, A_eq):
            X = [self.vars[i] for i in a_eq.indices]
            for col, coeff in zip(X, a_eq.data):
                self.addConsCoeff(row, col, float(coeff))
        self.constr += eqs

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
        self.freeTransform()
        # Make previous constraint non binding. removing or
        # changing old constraints would be better but doesn't work
        self.chgRhs(self.constr[idx], None)
        # add new constraint and replace constraint pointer in list
        self.constr[idx] = self.addCons(pso.Expr() <= 0, modifiable=True)
        for i, a in enumerate(a_ineq):
            self.addConsCoeff(self.constr[idx], self.vars[i], a)
        if isinf(b_ineq):
            self.chgRhs(self.constr[idx], None)
        else:
            self.chgRhs(self.constr[idx], b_ineq)
        pass

    def getSolution(self) -> list:
        """Retrieve solution from SCIP backend"""
        return [self.getVal(x) for x in self.vars]

    def addExclusionConstraintIneq(self, x):
        """Function to add exclusion constraint (SCIP compatibility function)"""
        data = [1.0 if x[i] else -1.0 for i in self.binvars]
        row = [0] * len(self.binvars)
        A_ineq = sparse.csr_matrix((data, (row, self.binvars)), (1, len(self.vars)))
        b_ineq = sum([x[i] for i in self.binvars]) - 1
        self.add_ineq_constraints(A_ineq, [b_ineq])


class SCIP_LP(pso.LP):
    """SoPlex interface for LP
    
    This class is a wrapper for the SoPlex-Python API to offer bindings and namings
    for functions for the construction and manipulation of LPs in an
    vector-matrix-based manner that are consistent with those of the other solver 
    interfaces in the StrainDesign package. The purpose is to unify the instructions 
    for operating with MILPs and LPs throughout StrainDesign.
    """

    def __init__(self, c, A_ineq, b_ineq, A_eq, b_eq, lb, ub):
        """Constructor of the SCIP (SoPlex) LP interface class
        
        Accepts a (mixed integer) linear problem in the form:
            minimize(c)
            subject to: A_ineq * x <= b_ineq
                        A_eq   * x  = b_eq
                        lb <= x <= ub
                        forall(i) type(x_i) = vtype(i) (continous, binary, integer)
                        indicator constraints:
                        x(j) = [0|1] -> a_indic * x [<=|=|>=] b_indic
                        
        Please ensure that the number of variables and (in)equalities is consistent
            
        Example: 
            scip = SCIP_LP(c, A_ineq, b_ineq, A_eq, b_eq, lb, ub)
                    
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
                
            Returns:
                (SCIP_LP):
                
                    A SCIP LP interface class.
        """
        super().__init__(sense='minimize')
        # uncomment to forward SCIP output to python terminal
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

        ub = [u if not isinf(u) else self.infinity() for u in ub]
        lb = [l if not isinf(l) else -self.infinity() for l in lb]
        # add variables and constraints
        self.addCols([()] * len(c), objs=c, lbs=lb, ubs=ub)
        # add inequality constraints
        self.addRows([[(i,v) for i,v in zip(rows.indices,rows.data)] for rows in A_ineq], \
                     lhss = [-self.infinity()]*A_ineq.shape[0],\
                     rhss = b_ineq)
        # add equality constraints
        self.addRows([[(i,v) for i,v in zip(rows.indices,rows.data)] for rows in A_eq], \
                     lhss = b_eq,\
                     rhss = b_eq)
        self.optimize = super().solve

    def solve(self) -> Tuple[List, float, float]:
        """Solve the LP
        
        Example:
            sol_x, optim, status = scip.solve()
        
        Returns:
            (Tuple[List, float, float])
            
            solution_vector, optimal_value, optimization_status
        """
        try:
            min_cx = self.optimize()  # this function was inherited from super().solve() during initialization
            if self.isInfinity(-min_cx):  # solution
                min_cx = -inf
                status = UNBOUNDED
            elif self.isInfinity(min_cx):
                min_cx = nan
                status = INFEASIBLE
            if not isnan(min_cx) and not isinf(min_cx):
                x = self.getPrimal()
                status = OPTIMAL
            else:
                x = [nan] * len(self.getPrimal())
            return x, min_cx, status
        except:
            logging.error('Error while running SCIP.')
            min_cx = nan
            x = [nan] * len(self.getPrimal())
            return x, min_cx, ERROR

    def slim_solve(self) -> float:
        """Solve the LP, but return only the optimal value
                
        Example:
            optim = scip.slim_solve()
        
        Returns:
            (float)
            
            Optimum value of the objective function.
        """
        try:
            opt = self.optimize()  # this function was inherited from super().solve() during initialization
            if self.isInfinity(-opt):  # solution
                opt = -inf
            elif self.isInfinity(opt):
                opt = nan
            return opt
        except:
            logging.error('Error while running SCIP.')
            return nan

    def set_objective(self, c):
        """Set the objective function with a vector"""
        for i in range(len(c)):
            self.chgObj(i, c[i])

    def set_objective_idx(self, C):
        """Set the objective function with index-value pairs
        
        e.g.: C=[[1, 1.0], [4,-0.2]]"""
        for i_v in C:
            self.chgObj(i_v[0], i_v[1])

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
        self.addRows([[(i,v) for i,v in zip(rows.indices,rows.data)] for rows in A_ineq], \
                        lhss = [-self.infinity()]*A_ineq.shape[0],\
                        rhss = b_ineq)

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
        self.addRows([[(i,v) for i,v in zip(rows.indices,rows.data)] for rows in A_eq], \
                        lhss = b_eq,\
                        rhss = b_eq)
