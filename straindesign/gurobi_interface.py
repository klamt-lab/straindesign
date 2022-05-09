from random import randint
from scipy import sparse
from numpy import nan, inf, isinf, sum, array
import gurobipy as gp
from gurobipy import GRB as grb
from straindesign.names import *
from typing import Tuple, List

# Collection of Gurobi-related functions that facilitate the creation
# of Gurobi-object and the solutions of LPs/MILPs with Gurobi from
# vector-matrix-based problem setups.
#

# Create a Gurobi-object from a matrix-based problem setup
class Gurobi_MILP_LP(gp.Model):
    def __init__(self,c,A_ineq,b_ineq,A_eq,b_eq,lb,ub,vtype,indic_constr):
        super().__init__()
        try:
            numvars = A_ineq.shape[1]
        except:
            numvars = A_eq.shape[1]
        # prepare coefficient matrix
        if isinstance(A_eq,list):
            if not A_eq:
                A_eq = sparse.csr_matrix((0,numvars))
        if isinstance(A_ineq,list):
            if not A_ineq:
                A_ineq = sparse.csr_matrix((0,numvars))

        for i,v in enumerate(b_ineq):
            if isinf(v):
                b_ineq[i] = grb.INFINITY
        # concatenate right hand sides
        # construct Gurobi problem. Add variables and linear constraints
        x = self.addMVar(len(c),obj=c, lb=lb, ub=ub, vtype=[k for k in vtype])
        self.setObjective(array(c) @ x, grb.MINIMIZE)
        self.addConstr(A_ineq @ x <= array(b_ineq))
        self.addConstr(A_eq   @ x == array(b_eq))

        # add indicator constraints
        if not indic_constr==None:
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
            seed = randint(0,grb.MAXINT)
            print('  MILP Seed: '+str(seed))
            self.params.Seed = seed
            self.params.IntFeasTol = 1e-9 # (0 is not allowed by Gurobi)
            # yield only optimal solutions in pool
            self.params.PoolGap = 0.0
            self.params.PoolGapAbs = 0.0

    def solve(self) -> Tuple[List,float,float]:
        try:
            self.optimize() # call parent solve function (that was overwritten in this class)
            status = self.Status
            if status in [2,10,13,15]: # solution
                min_cx = self.ObjVal
                status = OPTIMAL
            elif status == 9 and not hasattr(self._Model__vars[0],'X'): # timeout without solution
                x = [nan]*self.NumVars
                min_cx = nan
                status = TIME_LIMIT
                return x, min_cx, status
            elif status in [3,4] and not hasattr(self._Model__vars[0],'X'): # infeasible
                x = [nan]*self.NumVars
                min_cx = nan
                status = INFEASIBLE
                return x, min_cx, status
            elif status == 9 and hasattr(self._Model__vars[0],'X'): # timeout with solution
                min_cx = self.ObjVal
                status = TIME_LIMIT_W_SOL
            elif status in [4,5] and hasattr(self._Model__vars[0],'X'): # solution unbounded
                min_cx = -inf
                status = 4
            else:
                raise Exception('Status code '+str(status)+" not yet handeld.")
            x = self.getSolution()
            return x, min_cx, status

        except gp.GurobiError as e:
            print('Error code ' + str(e.errno) + ": " + str(e))
            min_cx = nan
            x = [nan] * self.NumVars
            return x, min_cx, ERROR

    def slim_solve(self) -> float:
        try:
            self.optimize() # call parent solve function (that was overwritten in this class)
            status = self.Status
            if status in [2,10,13,15]: # solution integer optimal (tolerance)
                opt = self.ObjVal
            elif status in [4,5]: # solution unbounded (or inf or unbdd)
                opt = -inf
            elif status == 3 or status == 9: # infeasible or timeout
                opt = nan
            else:
                raise Exception('Status code '+str(status)+" not yet handeld.")
            return opt
        except gp.GurobiError as e:
            print('Error code ' + str(e.errno) + ": " + str(e))
            return nan

    def populate(self,n) -> Tuple[List,float,float]:
        try:
            if isinf(n):
                self.params.PoolSolutions = grb.MAXINT
            else:
                self.params.PoolSolutions = n
            self.params.PoolSearchMode = 2
            self.optimize() # call parent solve function (that was overwritten in this class)
            self.params.PoolSearchMode = 0
            status = self.Status
            if status in [2,10,13,15]: # solution integer optimal
                min_cx = self.ObjVal
                status = OPTIMAL
            elif status == 9 and not hasattr(self._Model__vars[0],'X'): # timeout without solution
                x = []
                min_cx = nan
                status = TIME_LIMIT
                return x, min_cx, status
            elif status == 3: # infeasible
                x = []
                min_cx = nan
                status = INFEASIBLE
                return x, min_cx, status
            elif status == 9 and hasattr(self._Model__vars[0],'X'): # timeout with solution
                min_cx = self.ObjVal
                status = TIME_LIMIT_W_SOL
            elif status in [4,5]: # solution unbounded
                min_cx = -inf
                status = UNBOUNDED
            else:
                raise Exception('Status code '+str(status)+" not yet handeld.")
            nSols = self.SolCount
            x = []
            for i in range(nSols):
                self.setParam(grb.Param.SolutionNumber, i)
                x += [self.getSolutionN()]
            return x, min_cx, status

        except gp.GurobiError as e:
            self.params.PoolSearchMode = 0
            print('Error code ' + str(e.errno) + ": " + str(e))
            min_cx = nan
            x = []
            return x, min_cx, ERROR

    def set_objective(self,c):
        for i in range(len(self._Model__vars)):
            self._Model__vars[i].Obj = c[i]

    def set_objective_idx(self,C):
        for c in C:
            self._Model__vars[c[0]].Obj = c[1]

    def set_ub(self,ub):
        for i in range(len(ub)):
            self._Model__vars[ub[i][0]].ub = ub[i][1]

    def set_time_limit(self,t):
        self.params.TimeLimit = t

    def add_ineq_constraints(self,A_ineq,b_ineq):
        vars = self._Model__vars
        for i in range(A_ineq.shape[0]):
            self.addConstr(sum([A_ineq[i,j] * vars[j] for j in range(len(vars)) if not A_ineq[i,j] == 0.0]) <= b_ineq[i])

    def add_eq_constraints(self,A_eq,b_eq):
        vars = self._Model__vars
        for i in range(A_eq.shape[0]):
            self.addConstr(sum([A_eq[i,j] * vars[j] for j in range(len(vars)) if not A_eq[i,j] == 0.0]) <= b_eq[i])

    def set_ineq_constraint(self,idx,a_ineq,b_ineq):
        constr = self._Model__constrs[idx]
        [self.chgCoeff(constr,x,val) for x,val in zip(self._Model__vars,a_ineq)]
        if isinf(b_ineq):
            constr.rhs = grb.INFINITY
        else:
            constr.rhs = b_ineq

    def getSolution(self) -> list:
        return [x.X for x in self._Model__vars]

    def getSolutionN(self) -> list:
        return [x.Xn for x in self._Model__vars]