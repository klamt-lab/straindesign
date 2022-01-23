from re import X
import cobra
import numpy as np
from scipy import sparse
from typing import Dict, List, Tuple
from mcs import cplex_interface,indicator_constraints
from cplex.exceptions import CplexError

class MILP_LP:
    def __init__(self, *args, **kwargs):
        allowed_keys = {'c', 'A_ineq','b_ineq','A_eq','b_eq','lb','ub','vtype',
                        'indic_constr','x0','options','solver','skip_checks'}
        # set all keys passed in kwargs
        for key,value in kwargs.items():
            if key in allowed_keys:
                setattr(self,key,value)
            else:
                raise Exception("Key "+key+" is not supported.")
        # set all remaining keys to None
        for key in allowed_keys:
            if key not in kwargs.keys():
                setattr(self,key,None)
        # Select solver (either by choice or automatically cplex > gurobi > glpk)
        avail_solvers = list(cobra.util.solvers.keys())
        try:
            import pyscipopt
            avail_solvers += ['scip']
        except Exception:
            False
        if self.solver == None:
            if 'cplex' in avail_solvers:
                self.solver = 'cplex'
            elif 'gurobi' in avail_solvers:
                self.solver = 'gurobi'
            elif 'scip' in avail_solvers:
                self.solver = 'scip'
            else:
                self.solver = 'glpk'
        elif not self.solver in avail_solvers:
            raise Exception("Selected solver is not installed / set up correctly.")
        # Copy parameters to object
        try:
            numvars = self.A_ineq.shape[1]
        except:
            numvars = self.A_eq.shape[1]
        if self.c == None:
            self.c = [0]*numvars
        numineq = self.A_ineq.shape[0]
        self.A_ineq = self.A_ineq[[True if not np.isinf(self.b_ineq[i]) else False for i in range(0,numineq)],:]
        self.b_ineq = [self.b_ineq[i] for i in range(0,numineq) if not np.isinf(self.b_ineq[i])]
        if self.A_eq == None:
            self.A_eq = sparse.csr_matrix((0,numvars))
        if self.b_eq == None:
            self.b_eq = []
        if self.lb == None:
            self.lb = [-np.inf()]*numvars
        if self.ub == None:
            self.ub = [ np.inf()]*numvars
        if self.vtype == None:
            self.vtype = 'C'*numvars
        # check dimensions
        if not self.skip_checks == True:
            if not (self.A_ineq.shape[0] == len(self.b_ineq)):
                raise Exception("A_ineq and b_ineq must have the same number of rows/elements")
            if not (self.A_eq.shape[0] == len(self.b_eq)):
                raise Exception("A_eq and b_eq must have the same number of rows/elements")
            if not (self.A_ineq.shape[1]==numvars and self.A_eq.shape[1]==numvars and len(self.c)==numvars and \
                    len(self.lb)==numvars and len(self.ub)==numvars and len(self.vtype)==numvars):
                raise Exception("A_eq, A_ineq, c, lb, ub, vtype must have the same number of columns/elements")
            if (not self.indic_constr==None) and (not self.solver in ['cplex', 'gurobi', 'scip']):
                raise Exception("In order to use indicator constraints, you need to set up CPLEX, Gurobi or SCIP.")
            elif (not self.indic_constr==None): # check dimensions of indicator constraints
                num_ic = self.indic_constr.A.shape[0]
                if not (self.indic_constr.A.shape[1] == numvars and \
                        len(self.indic_constr.b)==num_ic and len(self.indic_constr.binv)==num_ic and \
                        len(self.indic_constr.sense)==num_ic and len(self.indic_constr.indicval)==num_ic):
                    raise Exception("Check dimensions of indicator constraints.")
        if self.solver == 'cplex':
            self.cpx = cplex_interface.init_cpx_milp(self.c,self.A_ineq,self.b_ineq,self.A_eq,self.b_eq,self.lb,self.ub,self.vtype,
                                                     self.indic_constr,self.x0)

    def solve(self) -> Tuple[List,float,float]:
        if self.solver == 'cplex':
            try:
                self.cpx.solve()
                x = self.cpx.solution.get_values()
                min_cx = self.cpx.solution.get_objective_value()
                return x, min_cx, 0
            except CplexError as exc:
                if not exc.args[2]==1217: 
                    print(exc)
                min_cx = np.nan
                x = [np.nan] * self.cpx.variables.get_num()
                return x, min_cx, -1

    def set_objective(self,c):
        self.c = c
        if self.solver == 'cplex':
            self.cpx.objective.set_linear([[i,c[i]] for i in range(len(c))])