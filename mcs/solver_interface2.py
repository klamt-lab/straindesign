from cobra.util import solvers
from numpy import inf, isinf, sign, nan, unique
from scipy import sparse
from typing import List, Tuple
from mcs import cplex_interface2,indicator_constraints
from cplex.exceptions import CplexError

class MILP_LP2:
    def __init__(self, *args, **kwargs):
        allowed_keys = {'c', 'A_ineq','b_ineq','A_eq','b_eq','lb','ub','vtype',
                        'indic_constr','x0','options','solver','skip_checks','tlim'}
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
        avail_solvers = list(solvers.keys())
        try:
            import pyscipopt
            avail_solvers += ['scip']
        except Exception:
            False
        if self.solver is None:
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
        if self.A_ineq is not None:
            numvars = self.A_ineq.shape[1]
        elif self.A_eq is not None:
            numvars = self.A_eq.shape[1]
        elif self.ub is not None:
            numvars = len(self.ub)
        else:
            raise Exception("Number of variables could not be determined.")
        if self.c is None:
            self.c = [0]*numvars
        if self.A_eq is None:
            self.A_eq = sparse.csr_matrix((0,numvars))
        if self.b_eq is None:
            self.b_eq = []
        numineq = self.A_ineq.shape[0]
        # Remove unbounded constraints
        if any(isinf(self.b_ineq)) and self.solver == 'cplex':
            print("CPLEX does not support unbounded inequalities. Inf bound is replaced by +/-1e9")
            self.b_ineq = [sign(self.b_ineq[i])*1e9 if isinf(self.b_ineq[i]) else self.b_ineq[i] for i in range(len(self.b_ineq))]
        if self.A_eq == None:
            self.A_eq = sparse.csr_matrix((0,numvars))
        if self.b_eq == None:
            self.b_eq = []
        if self.lb == None:
            self.lb = [-inf()]*numvars
        if self.ub == None:
            self.ub = [ inf()]*numvars
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
        # Create backend
        if self.solver == 'cplex':
            self.backend = cplex_interface2(self.c,self.A_ineq,self.b_ineq,self.A_eq,self.b_eq,self.lb,self.ub,self.vtype,
                                            self.indic_constr,self.x0)
        elif self.solver == 'gurobi':
            self.solver = None
        elif self.solver == 'scip':
            self.solver = None
        elif self.solver == 'glpk':
            self.solver = None
        
        if self.tlim is None:
            self.set_time_limit(1e9)
        else:
            self.set_time_limit(self.tlim)

    def solve(self) -> Tuple[List,float,float]:
        return self.backend.solve()

    def slim_solve(self) -> float:
        return self.backend.slim_solve()

    def set_objective(self,c):
        self.c = c
        self.backend.set_objective(c)
        if self.solver == 'cplex':
            self.cpx.objective.set_linear([[i,c[i]] for i in range(len(c))])

    def set_objective_idx(self,C):
        # when indices occur multiple times, take first one
        C_idx = [C[i][0] for i in range (len(C))]
        C_idx = unique([C_idx.index(C_idx[i]) for i in range(len(C_idx))])
        C = [C[i] for i in C_idx]
        for i in range(len(C)):
            self.c[C[i][0]] = C[i][1]
        if self.solver == 'cplex':
            self.cpx.objective.set_linear(C)

    def add_eq_constraint(self,A_ineq,b_ineq):
        if self.solver == 'cplex':
            pass
        pass

    def add_ineq_constraint(self,A_ineq,b_ineq):
        A_ineq = sparse.csr_matrix(A_ineq)
        A_ineq.eliminate_zeros()
        b_ineq = [float(b) for b in b_ineq]
        self.A_ineq = sparse.vstack((self.A_ineq,A_ineq))
        self.b_ineq += b_ineq
        if self.solver == 'cplex':
            numconst = self.cpx.linear_constraints.get_num()
            numnewconst = A_ineq.shape[0]
            newconst_idx = [numconst+i for i in range(numnewconst)]
            self.cpx.linear_constraints.add(rhs=b_ineq, senses='L'*numnewconst)
            # retrieve row and column indices from sparse matrix and convert them to int
            A_ineq = A_ineq.tocoo()
            rows_A = [int(a)+numconst for a in A_ineq.row]
            cols_A = [int(a) for a in A_ineq.col]
            data_A = [float(a) for a in A_ineq.data]
            # convert matrix coefficients to float
            data_A = [float(a) for a in A_ineq.data]
            self.cpx.linear_constraints.set_coefficients(zip(rows_A, cols_A, data_A))

    # ONLY DUMMIES SO FAR
    def add_indic_constraint(self,A_ineq,b_ineq):
        if self.solver == 'cplex':
            pass
        pass

    def reset_objective(self):
        self.c = [0]*len(self.c)
        if self.solver == 'cplex':
            self.cpx.objective.set_linear([[i,0] for i in range(self.cpx.variables.get_num())])

    def set_time_limit(self,t):
        self.tlim = t
        if self.solver == 'cplex':
            self.cpx.parameters.timelimit.set(t)

    def populate(self):
        pass

    def set_targetable_z(self):
        pass

    def reset_targetable_z(self):
        pass

    def reset_objective(self):
        pass

    def clear_objective(self):
        pass