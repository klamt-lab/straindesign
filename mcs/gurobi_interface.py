from scipy import sparse
import numpy as np
import gurobipy as gp
import cobra
from mcs import indicator_constraints
from typing import Tuple, List

# Collection of Gurobi-related functions that facilitate the creation
# of Gurobi-object and the solutions of LPs/MILPs with Gurobi from
# vector-matrix-based problem setups.
#

# Create a CPLEX-object from a matrix-based problem setup
class Gurobi_MILP_LP(gp.Model):
    def __init__(self,c,A_ineq,b_ineq,A_eq,b_eq,lb,ub,vtype,indic_constr,x0,options):
        super().__init__()
        self.objective.set_sense(self.objective.sense.minimize)
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
        A = sparse.vstack((A_ineq,A_eq),format='coo') # concatenate coefficient matrices
        sense = len(b_ineq)*'L' + len(b_eq)*'E'

        # construct CPLEX problem. Add variables and linear constraints
        if not vtype: # when undefined, all variables are continous
            vtype = 'C'*numvars
        self.variables.add(obj=c, lb=lb, ub=ub, types=vtype)
        self.linear_constraints.add(rhs=b, senses=sense)
        self.linear_constraints.set_coefficients(zip(A.row.tolist(), A.col.tolist(), A.data.tolist()))

        # add indicator constraints
        if not indic_constr==None:
            # cast variables and translate coefficient matrix A to right input format for CPLEX
            A = [[[int(i) for i in list(a.indices)], [float(i) for i in list(a.data)]] for a in sparse.csr_matrix(indic_constr.A)]
            b = [float(i) for i in indic_constr.b]
            sense = [str(i) for i in indic_constr.sense]
            indvar = [int(i) for i in indic_constr.binv]
            complem = [1-int(i) for i in indic_constr.indicval]
            # call CPLEX function to add indicators
            self.indicator_constraints.add_batch(lin_expr=A, sense=sense,rhs=b,
                                                indvar=indvar,complemented=complem)
        # set parameters
        self.set_log_stream(None)
        self.set_error_stream(None)
        self.set_warning_stream(None)
        self.set_results_stream(None)
        self.parameters.simplex.tolerances.optimality.set(1e-9)
        self.parameters.simplex.tolerances.feasibility.set(1e-9)

    def solve(self) -> Tuple[List,float,float]:
        try:
            super().solve() # call parent solve function (that was overwritten in this class)
            status = self.solution.get_status()
            if status in [101,102,115,128,129,130]: # solution integer optimal
                min_cx = self.solution.get_objective_value()
                status = 0
            elif status == 108: # timeout without solution
                x = [nan]*self.variables.get_num()
                min_cx = nan
                status = 1
                return x, min_cx, status
            elif status == 103: # infeasible
                x = [nan]*self.variables.get_num()
                min_cx = nan
                status = 2
                return x, min_cx, status
            elif status == 107: # timeout with solution
                min_cx = self.solution.get_objective_value()
                status = 3
            elif status == [118,119]: # solution unbounded
                min_cx = -inf
                status = 4
            else:
                print(self.solution.get_status_string())
                raise Exception("Case not yet handeld")

            x = self.solution.get_values()
            return x, min_cx, status

        except CplexError as exc:
            if not exc.args[2]==1217: 
                print(exc)
            min_cx = nan
            x = [nan] * self.variables.get_num()
            return x, min_cx, -1

    def slim_solve(self) -> float:
        try:
            super().solve() # call parent solve function (that was overwritten in this class)
            status = self.solution.get_status()
            if status in [101,102,107,115,128,129,130]: # solution integer optimal (tolerance)
                opt = self.solution.get_objective_value()
            elif status in [118,119]: # solution unbounded (or inf or unbdd)
                opt = -inf
            elif status == [103,108]: # infeasible
                opt = nan
            else:
                print(self.solution.get_status_string())
                raise Exception("Case not yet handeld")
            return opt
        except CplexError as exc:
            return nan

    def populate(self) -> Tuple[List,float,float]:
        super().populate_solution_pool()
        pass

    def set_objective(self,c):
        self.objective.set_linear([[i,c[i]] for i in range(len(c))])

    def set_objective_idx(self,C):
        self.objective.set_linear(C)

    def set_time_limit(self,t):
        if isinf(t):
            self.parameters.timelimit.set(self.parameters.timelimit.max())
        else:
            self.parameters.timelimit.set(t)

    def add_ineq_constraint(self,A_ineq,b_ineq):
        numconst = self.linear_constraints.get_num()
        numnewconst = A_ineq.shape[0]
        newconst_idx = [numconst+i for i in range(numnewconst)]
        self.linear_constraints.add(rhs=b_ineq, senses='L'*numnewconst)
        # retrieve row and column indices from sparse matrix and convert them to int
        A_ineq = A_ineq.tocoo()
        rows_A = [int(a)+numconst for a in A_ineq.row]
        cols_A = [int(a) for a in A_ineq.col]
        data_A = [float(a) for a in A_ineq.data]
        # convert matrix coefficients to float
        data_A = [float(a) for a in A_ineq.data]
        self.linear_constraints.set_coefficients(zip(rows_A, cols_A, data_A))

    def add_eq_constraint(self,A_eq,b_eq):
        numconst = self.linear_constraints.get_num()
        numnewconst = A_eq.shape[0]
        newconst_idx = [numconst+i for i in range(numnewconst)]
        self.linear_constraints.add(rhs=b_eq, senses='E'*numnewconst)
        # retrieve row and column indices from sparse matrix and convert them to int
        A_eq = A_eq.tocoo()
        rows_A = [int(a)+numconst for a in A_eq.row]
        cols_A = [int(a) for a in A_eq.col]
        data_A = [float(a) for a in A_eq.data]
        # convert matrix coefficients to float
        data_A = [float(a) for a in A_eq.data]
        self.linear_constraints.set_coefficients(zip(rows_A, cols_A, data_A))