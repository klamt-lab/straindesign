from scipy import sparse
from numpy import nan, inf, isinf, sum, array
import cobra
from mcs import indicator_constraints
from typing import Tuple, List
from swiglpk import *

# Collection of Gurobi-related functions that facilitate the creation
# of Gurobi-object and the solutions of LPs/MILPs with Gurobi from
# vector-matrix-based problem setups.
#

# Create a Gurobi-object from a matrix-based problem setup
class GLPK_MILP_LP():
    def __init__(self,c,A_ineq,b_ineq,A_eq,b_eq,lb,ub,vtype,indic_constr,x0,options):
        self.glpk = glp_create_prob()
        # Careful with indexing! GLPK indexing starts with 1 and not with 0 
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

        if all([v =='C' for v in vtype]):
            self.ismilp = False
        else:
            self.ismilp = True

        # add and set variables, types and bounds
        glp_add_cols(self.glpk, numvars)
        for i,v in enumerate(vtype):
            if v=='C':
                glp_set_col_kind(self.glpk,i+1 ,GLP_CV)
            if v=='I':
                glp_set_col_kind(self.glpk,i+1 ,GLP_IV)
            if v=='B':
                glp_set_col_kind(self.glpk,i+1 ,GLP_BV)
        # set bounds
        lb = [float(l) for l in lb]
        ub = [float(u) for u in ub]
        for i in range(numvars):
            if       isinf(lb[i]) and     isinf(ub[i]):
                glp_set_col_bnds(self.glpk,i+1,GLP_FR,lb[i],ub[i])
            elif not isinf(lb[i]) and     isinf(ub[i]):
                glp_set_col_bnds(self.glpk,i+1,GLP_LO,lb[i],ub[i])
            elif     isinf(lb[i]) and not isinf(ub[i]):
                glp_set_col_bnds(self.glpk,i+1,GLP_UP,lb[i],ub[i])
            elif not isinf(lb[i]) and not isinf(ub[i]) and lb[i] < ub[i]:
                glp_set_col_bnds(self.glpk,i+1,GLP_DB,lb[i],ub[i])
            elif not isinf(lb[i]) and not isinf(ub[i]) and lb[i] == ub[i]:
                glp_set_col_bnds(self.glpk,i+1,GLP_FX,lb[i],ub[i])
            
        # set objective
        glp_set_obj_dir(self.glpk, GLP_MIN)
        for i,c_i in enumerate(c):
            glp_set_obj_coef(self.glpk,i+1,float(c_i))

        # add indicator constraints
        A_indic = sparse.lil_matrix((0,numvars))
        b_indic = []
        if not indic_constr==None:
            if options is not None and hasattr(options,'M'):
                M = options.M
            else:
                M = 1e3
            print('There is no native support of indicator constraints with GLPK.')
            print('Indicator constraints are translated to big-M constraints using M='+str(M)+'.')
            num_ic = len(indic_constr.binv)

            eq_type_indic = [] # [GLP_UP]*len(b_ineq)+[GLP_FX]*len(b_eq)
            for i in range(num_ic):
                binv = indic_constr.binv[i]
                indicval = indic_constr.indicval[i]
                A =  indic_constr.A[i]
                b =  float(indic_constr.b[i])
                sense = indic_constr.sense[i]
                if sense == 'E':
                    A = sparse.vstack((A,-A)).tolil()
                    b = [b, -b]
                else:
                    A = A.tolil()
                    b = [b]
                if indicval:
                    A[:,binv] =  M
                    b = [v+M for v in b]
                else:
                    A[:,binv] = -M
                A_indic = sparse.vstack((A_indic,A))
                b_indic = b_indic+b

        # stack all problem rows and add constraints
        glp_add_rows(self.glpk, A_ineq.shape[0]+A_eq.shape[0]+A_indic.shape[0])
        b_ineq = [float(b) for b in b_ineq]
        b_eq   = [float(b) for b in b_eq]
        eq_type = [GLP_UP]*len(b_ineq)+[GLP_FX]*len(b_eq)+[GLP_UP]*len(b_indic)
        for i,t,b in zip(range(len(b_ineq+b_eq+b_indic)),eq_type,b_ineq+b_eq+b_indic):
            glp_set_row_bnds(self.glpk,i+1,t,b,b)

        A = sparse.vstack((A_ineq,A_eq,A_indic),'coo')
        ia = intArray(A.nnz+1)
        ja = intArray(A.nnz+1)
        ar = doubleArray(A.nnz+1)
        for i,row,col,data in zip(range(A.nnz),A.row,A.col,A.data):
            ia[i+1] = int(row)+1
            ja[i+1] = int(col)+1
            ar[i+1] = float(data)
        glp_load_matrix(self.glpk, A.nnz, ia, ja, ar)

        # not sure if the parameter setup is okay
        self.milp_params = glp_iocp()
        self.lp_params   = glp_smcp()
        glp_init_iocp(self.milp_params)
        glp_init_smcp(self.lp_params)
        self.max_tlim = self.lp_params.tm_lim
    
        self.milp_params.presolve = 1
        self.milp_params.tol_int = 1e-12
        self.milp_params.tol_obj = 1e-9
        self.lp_params.tol_bnd = 1e-9
        

    def __del__(self):
        glp_delete_prob(self.glpk)

    def solve(self) -> Tuple[List,float,float]:
        try:
            min_cx, status, bool_tlim = self.solve_MILP_LP()
            if status in [GLP_OPT,GLP_FEAS]: # solution
                status = 0
            elif bool_tlim and status == GLP_UNDEF: # timeout without solution
                x = [nan]*glp_get_num_cols(self.glpk)
                min_cx = nan
                status = 1
                return x, min_cx, status
            elif status in [GLP_INFEAS,GLP_NOFEAS]: # infeasible
                x = [nan]*glp_get_num_cols(self.glpk)
                min_cx = nan
                status = 2
                return x, min_cx, status
            elif bool_tlim and status == GLP_FEAS: # timeout with solution
                min_cx = self.ObjVal
                status = 3
            elif status in [GLP_UNBND,GLP_UNDEF]: # solution unbounded
                min_cx = -inf
                status = 4
            else:
                raise Exception('Status code '+str(status)+" not yet handeld.")
            x = self.getSolution()
            return x, min_cx, status

        except:
            print('Error while running GLPK.')
            min_cx = nan
            x = [nan] * glp_get_num_cols(self.glpk)
            return x, min_cx, -1

    def slim_solve(self) -> float:
        try:
            opt, status, bool_tlim = self.solve_MILP_LP()
            if status in [GLP_OPT,GLP_FEAS]: # solution integer optimal (tolerance)
                pass
            elif status in [GLP_UNBND,GLP_UNDEF]: # solution unbounded (or inf or unbdd)
                opt = -inf
            elif bool_tlim or status in [GLP_INFEAS,GLP_NOFEAS]: # infeasible or timeout
                opt = nan
            else:
                raise Exception('Status code '+str(status)+" not yet handeld.")
            return opt
        except:
            print('Error while running GLPK.')
            return nan

    def populate(self,n) -> Tuple[List,float,float]:
        try:
            print('Gurobi does not support populate. Optimal solutions are generated iteratively instead.')
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
                status = 0
            elif status == 9 and not hasattr(self._Model__vars[0],'X'): # timeout without solution
                x = []
                min_cx = nan
                status = 1
                return x, min_cx, status
            elif status == 3: # infeasible
                x = []
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
            x = [nan] * self.NumVars
            return x, min_cx, -1

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
        if isinf(t):
            self.milp_params.tm_lim = self.max_tlim
            self.lp_params.tm_lim = self.max_tlim
        else:
            self.milp_params.tm_lim = t
            self.lp_params.tm_lim = t

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
        if self.ismilp:
            x = [glp_mip_col_val(self.glpk,i+1) for i in range(glp_get_num_cols(self.glpk))]
        else:
            x = [glp_get_col_prim(self.glpk,i+1) for i in range(glp_get_num_cols(self.glpk))]
        return x

    def solve_MILP_LP(self) -> Tuple[float,int,bool]:
        starttime = glp_time()
        if self.ismilp:
            glp_intopt(self.glpk,self.milp_params)
            status = glp_mip_status(self.glpk)
            opt = glp_mip_obj_val(self.glpk)
            timelim_reached = glp_difftime(glp_time(),starttime) >= self.milp_params.tm_lim
        else:
            glp_simplex(self.glpk,self.lp_params)
            status = glp_get_status(self.glpk)
            opt = glp_get_obj_val(self.glpk)
            timelim_reached = glp_difftime(glp_time(),starttime) >= self.lp_params.tm_lim
        return opt, status, timelim_reached