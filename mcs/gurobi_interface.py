from scipy import sparse
from numpy import nan, inf, isinf, sum, array
import gurobipy as gp
from gurobipy import GRB as grb
import cobra
from mcs import indicator_constraints
from typing import Tuple, List

# Collection of Gurobi-related functions that facilitate the creation
# of Gurobi-object and the solutions of LPs/MILPs with Gurobi from
# vector-matrix-based problem setups.
#

# Create a Gurobi-object from a matrix-based problem setup
class Gurobi_MILP_LP(gp.Model):
    def __init__(self,c,A_ineq,b_ineq,A_eq,b_eq,lb,ub,vtype,indic_constr,x0,options):
        super().__init__()
        try:
            numvars = A_ineq.shape[1]
        except:
            numvars = A_eq.shape[1]
        # concatenate right hand sides
        b = b_ineq + b_eq
        # prepare coefficient matrix
        if isinstance(A_eq,list):
            if not A_eq:
                A_eq = sparse.csr_matrix((0,numvars))
        if isinstance(A_ineq,list):
            if not A_ineq:
                A_ineq = sparse.csr_matrix((0,numvars))

        # construct Gurobi problem. Add variables and linear constraints
        x = self.addMVar(len(c),obj=c, lb=lb, ub=ub, vtype=[k for k in vtype])
        self.setObjective(array(c) @ x, grb.MINIMIZE)
        self.addConstr(A_eq   @ x == array(b_eq))
        self.addConstr(A_ineq @ x <= array(b_ineq))

        # add indicator constraints
        if not indic_constr==None:
            for i in range(len(indic_constr.sense)):
                self.addGenConstrIndicator(x[indic_constr.binv[i]], 
                                        bool(indic_constr.indicval[i]), 
                                        sum([indic_constr.A[i,j] * x[j] \
                                            for j in range(len(c)) if not indic_constr.A[i,j] == 0.0]), 
                                        '=' if indic_constr.sense[1] =='E' else '<', 
                                        indic_constr.b[i])

        # set parameters
        self.params.OutputFlag = 0
        self.params.OptimalityTol = 1e-9
        self.params.FeasibilityTol = 1e-9
        self.params.IntFeasTol = 1e-9 # (0 is not allowed by Gurobi)
        # yield only optimal solutions in pool
        self.params.PoolGap = 0.0
        self.params.PoolGapAbs = 0.0
        self.params.PoolSearchMode = 2

    def solve(self) -> Tuple[List,float,float]:
        try:
            super().optimize() # call parent solve function (that was overwritten in this class)
            status = self.Status
            if status in [2,10,13,15]: # solution
                min_cx = self.ObjVal
                status = 0
            elif status == 9 and not hasattr(self._Model__vars[0],'X'): # timeout without solution
                x = [nan]*self.NumVars
                min_cx = nan
                status = 1
                return x, min_cx, status
            elif status == 3: # infeasible
                x = [nan]*self.NumVars
                min_cx = nan
                status = 2
                return x, min_cx, status
            elif status == 9 and hasattr(self._Model__vars[0],'X'): # timeout with solution
                min_cx = self.ObjVal
                status = 3
            elif status in [4,5]: # solution unbounded
                min_cx = -inf
                status = 4
            else:
                raise Exception('Status code '+str(status)+" not yet handeld.")
            x = self.getSolutions()
            return x, min_cx, status

        except gp.GurobiError as e:
            print('Error code ' + str(e.errno) + ": " + str(e))
            min_cx = nan
            x = [nan] * self.NumVars
            return x, min_cx, -1

    def slim_solve(self) -> float:
        try:
            super().optimize() # call parent solve function (that was overwritten in this class)
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
            self.optimize() # call parent solve function (that was overwritten in this class)
            status = self.Status
            if status in [2,10,13,15]: # solution integer optimal
                min_cx = self.ObjVal
                status = 0
            elif status == 9: # timeout without solution
                x = []
                min_cx = nan
                status = 1
                return x, min_cx, status
            elif status == 103: # infeasible
                x = []
                min_cx = nan
                status = 2
                return x, min_cx, status
            elif status == 107: # timeout with solution
                min_cx = self.ObjVal
                status = 3
            elif status in [118,119]: # solution unbounded
                min_cx = -inf
                status = 4
            else:
                raise Exception('Status code '+str(status)+" not yet handeld.")
            x = [self.solution.pool.get_values(i) for i in range(self.solution.pool.get_num())]
            return x, min_cx, status

        except gp.GurobiError as e:
            print('Error code ' + str(e.errno) + ": " + str(e))
            min_cx = nan
            x = [nan] * self.NumVars
            return x, min_cx, -1

    def set_objective(self,c):
        for i in range(len(self._Model__vars)):
            self._Model__vars[i].Obj = c[i]

    def set_objective_idx(self,C):
        for c in C:
            self._Model__vars[c[0]].Obj = c[1]

    def set_ub(self,ub):
        for i in range(len(self._Model__vars)):
            self._Model__vars[i].ub = ub[i]

    def set_time_limit(self,t):
        self.params.TimeLimit = t

    def add_ineq_constraint(self,A_ineq,b_ineq):
        self.addConstr(A_ineq @ self._Model__vars <= array(b_ineq))

    def add_eq_constraint(self,A_eq,b_eq):
        self.addConstr(A_eq   @ self._Model__vars == array(b_eq))

    def getSolutions(self) -> list:
        return [x.X for x in self._Model__vars]