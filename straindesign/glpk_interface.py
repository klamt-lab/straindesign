from scipy import sparse
from numpy import nan, isnan, inf, isinf, sum, random
from straindesign.names import *
from typing import Tuple, List
from swiglpk import *
import logging

# Collection of GLPK-related functions that facilitate the creation
# of GLPK-object and the solutions of LPs/MILPs with GLPK from
# vector-matrix-based problem setups.


# Create a GLPK-object from a matrix-based problem setup
class GLPK_MILP_LP():

    def __init__(self,
                 c,
                 A_ineq,
                 b_ineq,
                 A_eq,
                 b_eq,
                 lb,
                 ub,
                 vtype,
                 indic_constr,
                 M=None):
        self.glpk = glp_create_prob()
        # Careful with indexing! GLPK indexing starts with 1 and not with 0
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

        if all([v == 'C' for v in vtype]):
            self.ismilp = False
        else:
            self.ismilp = True

        # add and set variables, types and bounds
        if numvars > 0:
            glp_add_cols(self.glpk, numvars)
        for i, v in enumerate(vtype):
            if v == 'C':
                glp_set_col_kind(self.glpk, i + 1, GLP_CV)
            if v == 'I':
                glp_set_col_kind(self.glpk, i + 1, GLP_IV)
            if v == 'B':
                glp_set_col_kind(self.glpk, i + 1, GLP_BV)
        # set bounds
        lb = [float(l) for l in lb]
        ub = [float(u) for u in ub]
        for i in range(numvars):
            if isinf(lb[i]) and isinf(ub[i]):
                glp_set_col_bnds(self.glpk, i + 1, GLP_FR, lb[i], ub[i])
            elif not isinf(lb[i]) and isinf(ub[i]):
                glp_set_col_bnds(self.glpk, i + 1, GLP_LO, lb[i], ub[i])
            elif isinf(lb[i]) and not isinf(ub[i]):
                glp_set_col_bnds(self.glpk, i + 1, GLP_UP, lb[i], ub[i])
            elif not isinf(lb[i]) and not isinf(ub[i]) and lb[i] < ub[i]:
                glp_set_col_bnds(self.glpk, i + 1, GLP_DB, lb[i], ub[i])
            elif not isinf(lb[i]) and not isinf(ub[i]) and lb[i] == ub[i]:
                glp_set_col_bnds(self.glpk, i + 1, GLP_FX, lb[i], ub[i])

        # set objective
        glp_set_obj_dir(self.glpk, GLP_MIN)
        for i, c_i in enumerate(c):
            glp_set_obj_coef(self.glpk, i + 1, float(c_i))

        # add indicator constraints
        A_indic = sparse.lil_matrix((0, numvars))
        b_indic = []
        if not indic_constr == None:
            if not M:
                M = 1e3
            logging.warning(
                'There is no native support of indicator constraints with GLPK.'
            )
            logging.warning(
                'Indicator constraints are translated to big-M constraints with M='
                + str(M) + '.')
            num_ic = len(indic_constr.binv)

            eq_type_indic = []  # [GLP_UP]*len(b_ineq)+[GLP_FX]*len(b_eq)
            for i in range(num_ic):
                binv = indic_constr.binv[i]
                indicval = indic_constr.indicval[i]
                A = indic_constr.A[i]
                b = float(indic_constr.b[i])
                sense = indic_constr.sense[i]
                if sense == 'E':
                    A = sparse.vstack((A, -A)).tolil()
                    b = [b, -b]
                else:
                    A = A.tolil()
                    b = [b]
                if indicval:
                    A[:, binv] = M
                    b = [v + M for v in b]
                else:
                    A[:, binv] = -M
                A_indic = sparse.vstack((A_indic, A))
                b_indic = b_indic + b

        # stack all problem rows and add constraints
        if A_ineq.shape[0] + A_eq.shape[0] + A_indic.shape[0] > 0:
            glp_add_rows(self.glpk,
                         A_ineq.shape[0] + A_eq.shape[0] + A_indic.shape[0])
            b_ineq = [float(b) for b in b_ineq]
            b_eq = [float(b) for b in b_eq]
            eq_type = [GLP_UP] * len(b_ineq) + [GLP_FX] * len(
                b_eq) + [GLP_UP] * len(b_indic)
            for i, t, b in zip(range(len(b_ineq + b_eq + b_indic)), eq_type,
                               b_ineq + b_eq + b_indic):
                glp_set_row_bnds(self.glpk, i + 1, t, b, b)

            A = sparse.vstack((A_ineq, A_eq, A_indic), 'coo')
            ia = intArray(A.nnz + 1)
            ja = intArray(A.nnz + 1)
            ar = doubleArray(A.nnz + 1)
            for i, row, col, data in zip(range(A.nnz), A.row, A.col, A.data):
                ia[i + 1] = int(row) + 1
                ja[i + 1] = int(col) + 1
                ar[i + 1] = float(data)
            if A.nnz:
                glp_load_matrix(self.glpk, A.nnz, ia, ja, ar)

        # not sure if the parameter setup is okay
        self.milp_params = glp_iocp()
        self.lp_params = glp_smcp()
        glp_init_iocp(self.milp_params)
        glp_init_smcp(self.lp_params)
        self.max_tlim = self.lp_params.tm_lim

        self.milp_params.presolve = 1
        self.milp_params.tol_int = 1e-12
        self.milp_params.tol_obj = 1e-9
        self.lp_params.tol_bnd = 1e-9
        self.lp_params.msg_lev = 0
        self.milp_params.msg_lev = 0
        # ideally, one would generate random seeds here, but glpk does not seem to
        # offer this function

    def solve(self) -> Tuple[List, float, float]:
        try:
            min_cx, status, bool_tlim = self.solve_MILP_LP()
            if status in [GLP_OPT, GLP_FEAS]:  # solution
                status = OPTIMAL
            elif bool_tlim and status == GLP_UNDEF:  # timeout without solution
                x = [nan] * glp_get_num_cols(self.glpk)
                min_cx = nan
                status = TIME_LIMIT
                return x, min_cx, status
            elif status in [GLP_INFEAS, GLP_NOFEAS]:  # infeasible
                x = [nan] * glp_get_num_cols(self.glpk)
                min_cx = nan
                status = INFEASIBLE
                return x, min_cx, status
            elif bool_tlim and status == GLP_FEAS:  # timeout with solution
                min_cx = self.ObjVal
                status = TIME_LIMIT_W_SOL
            elif status in [GLP_UNBND, GLP_UNDEF]:  # solution unbounded
                x = [nan] * glp_get_num_cols(self.glpk)
                min_cx = -inf
                status = UNBOUNDED
                return x, min_cx, status
            else:
                raise Exception('Status code ' + str(status) +
                                " not yet handeld.")
            x = self.getSolution(status)
            x = [round(y, 12) for y in x]  # workaround, round to 12 decimals
            min_cx = round(min_cx, 12)
            return x, min_cx, status

        except:
            logging.error('Error while running GLPK.')
            min_cx = nan
            x = [nan] * glp_get_num_cols(self.glpk)
            return x, min_cx, -1

    def slim_solve(self) -> float:
        try:
            opt, status, bool_tlim = self.solve_MILP_LP()
            if status in [GLP_OPT,
                          GLP_FEAS]:  # solution integer optimal (tolerance)
                pass
            elif status in [GLP_UNBND,
                            GLP_UNDEF]:  # solution unbounded (or inf or unbdd)
                opt = -inf
            elif bool_tlim or status in [GLP_INFEAS, GLP_NOFEAS
                                        ]:  # infeasible or timeout
                opt = nan
            else:
                raise Exception('Status code ' + str(status) +
                                " not yet handeld.")
            opt = round(opt, 12)  # workaround, round to 12 decimals
            return opt
        except:
            logging.error('Error while running GLPK.')
            return nan

    def populate(self, pool_limit) -> Tuple[List, float, float]:
        numvars = glp_get_num_cols(self.glpk)
        numrows = glp_get_num_rows(self.glpk)
        try:
            if pool_limit > 0:
                sols = []
                stoptime = glp_time() + self.milp_params.tm_lim * 1000
                # 1. find optimal solution
                self.set_time_limit(glp_difftime(stoptime, glp_time()))
                x, min_cx, status = self.solve()
                if status not in [OPTIMAL, UNBOUNDED]:
                    return sols, min_cx, status
                sols = [x]
                # 2. constrain problem to optimality
                c = [glp_get_obj_coef(self.glpk, i + 1) for i in range(numvars)]
                self.add_ineq_constraints(sparse.csr_matrix(c), [min_cx])
                # 3. exclude first solution pool
                self.addExclusionConstraintsIneq(x)
                # 4. loop solve and exclude until problem becomes infeasible
                while status in [OPTIMAL,UNBOUNDED] and not isnan(x[0]) \
                  and glp_difftime(stoptime,glp_time()) > 0 and pool_limit > len(sols):
                    self.set_time_limit(glp_difftime(stoptime, glp_time()))
                    x, _, status = self.solve()
                    if status in [OPTIMAL, UNBOUNDED]:
                        self.addExclusionConstraintsIneq(x)
                        sols += [x]
                if glp_difftime(stoptime, glp_time()) < 0:
                    status = TIME_LIMIT_W_SOL
                elif status == INFEASIBLE:
                    status = OPTIMAL
                # 5. remove auxiliary constraints
                # Here, we only free the upper bound of the constraints
                totrows = glp_get_num_rows(self.glpk)
                for j in range(numrows, totrows):
                    self.set_ineq_constraint(j, [0] * numvars, inf)
                # Alternatively rows may be deleted, but this seems to be very unstable
                # delrows = intArray(totrows-numrows)
                # for i,j in range(numrows,totrows):
                # delrows[i+1] = j+1
                # glp_del_rows(self.glpk,totrows-numrows,delrows)
                return sols, min_cx, status
        except:
            logging.error('Error while running GLPK.')
            x = []
            min_cx = nan
            return x, min_cx, ERROR

    def set_objective(self, c):
        for i, c_i in enumerate(c):
            glp_set_obj_coef(self.glpk, i + 1, float(c_i))

    def set_objective_idx(self, C):
        for c in C:
            glp_set_obj_coef(self.glpk, c[0] + 1, float(c[1]))

    def set_ub(self, ub):
        setvars = [ub[i][0] for i in range(len(ub))]
        lb = [glp_get_col_lb(self.glpk, i + 1) for i in setvars]
        ub = [ub[i][1] for i in range(len(ub))]
        type = [glp_get_col_type(self.glpk, i + 1) for i in setvars]
        for i, l, u, t in zip(setvars, lb, ub, type):
            if t in [GLP_FR, GLP_LO] and isinf(u):
                glp_set_col_bnds(self.glpk, i + 1, t, l, u)
            elif t == GLP_UP and isinf(u):
                glp_set_col_bnds(self.glpk, i + 1, GLP_FR, l, u)
            elif t in [GLP_LO, GLP_DB, GLP_FX] and not isinf(u) and l < u:
                glp_set_col_bnds(self.glpk, i + 1, GLP_DB, l, u)
            elif t in [GLP_LO, GLP_DB, GLP_FX] and not isinf(u) and l == u:
                glp_set_col_bnds(self.glpk, i + 1, GLP_FX, l, u)

    def set_time_limit(self, t):
        if t * 1000 > self.max_tlim:
            self.milp_params.tm_lim = self.max_tlim
            self.lp_params.tm_lim = self.max_tlim
        else:
            self.milp_params.tm_lim = int(t * 1000)
            self.lp_params.tm_lim = int(t * 1000)

    def add_ineq_constraints(self, A_ineq, b_ineq):
        numvars = glp_get_num_cols(self.glpk)
        numrows = glp_get_num_rows(self.glpk)
        num_newrows = A_ineq.shape[0]
        col = intArray(numvars + 1)
        val = doubleArray(numvars + 1)
        glp_add_rows(self.glpk, num_newrows)
        for j in range(num_newrows):
            for i, v in enumerate(A_ineq[j].toarray()[0]):
                col[i + 1] = i + 1
                val[i + 1] = float(v)
            glp_set_mat_row(self.glpk, numrows + j + 1, numvars, col, val)
            if isinf(b_ineq[j]):
                glp_set_row_bnds(self.glpk, numrows + j + 1, GLP_FR, -inf,
                                 b_ineq[j])
            else:
                glp_set_row_bnds(self.glpk, numrows + j + 1, GLP_UP, -inf,
                                 b_ineq[j])

    def add_eq_constraints(self, A_eq, b_eq):
        numvars = glp_get_num_cols(self.glpk)
        numrows = glp_get_num_rows(self.glpk)
        num_newrows = A_eq.shape[0]
        col = intArray(numvars + 1)
        val = doubleArray(numvars + 1)
        glp_add_rows(self.glpk, num_newrows)
        for j in range(num_newrows):
            for i, v in enumerate(A_eq[j].toarray()[0]):
                col[i + 1] = i + 1
                val[i + 1] = float(v)
            glp_set_mat_row(self.glpk, numrows + j + 1, numvars, col, val)
            glp_set_row_bnds(self.glpk, numrows + j + 1, GLP_FX, b_eq[j],
                             b_eq[j])

    def set_ineq_constraint(self, idx, a_ineq, b_ineq):
        numvars = glp_get_num_cols(self.glpk)
        col = intArray(numvars + 1)
        val = doubleArray(numvars + 1)
        for i, v in enumerate(a_ineq):
            col[i + 1] = i + 1
            val[i + 1] = float(v)
        glp_set_mat_row(self.glpk, idx + 1, numvars, col, val)
        if isinf(b_ineq):
            glp_set_row_bnds(self.glpk, idx + 1, GLP_FR, -inf, b_ineq)
        else:
            glp_set_row_bnds(self.glpk, idx + 1, GLP_UP, -inf, b_ineq)

    def getSolution(self, status) -> list:
        if self.ismilp and status in [OPTIMAL, UNBOUNDED, TIME_LIMIT_W_SOL]:
            x = [
                glp_mip_col_val(self.glpk, i + 1)
                for i in range(glp_get_num_cols(self.glpk))
            ]
        else:
            x = [
                glp_get_col_prim(self.glpk, i + 1)
                for i in range(glp_get_num_cols(self.glpk))
            ]
        return x

    def solve_MILP_LP(self) -> Tuple[float, int, bool]:
        starttime = glp_time()
        # MILP solving needs prior solution of the LP-relaxed problem, because occasionally
        # the MILP solver interface crashes when a problem is infesible, which, in turn,
        # crashes the python program. This connection-loss to the solver can not be captured.
        glp_simplex(self.glpk, self.lp_params)
        status = glp_get_status(self.glpk)
        if self.ismilp and status not in [GLP_INFEAS, GLP_NOFEAS]:
            glp_intopt(self.glpk, self.milp_params)
            status = glp_mip_status(self.glpk)
            opt = glp_mip_obj_val(self.glpk)
        else:
            opt = glp_get_obj_val(self.glpk)
        timelim_reached = glp_difftime(glp_time(),
                                       starttime) >= self.lp_params.tm_lim
        return opt, status, timelim_reached

    def addExclusionConstraintsIneq(self, x):
        numvars = glp_get_num_cols(self.glpk)
        # Here, we also need to take integer variables into account, because GLPK changes
        # variable type to integer when you lock a binary variable to zero
        binvars = [
            i for i in range(numvars)
            if glp_get_col_kind(self.glpk, i + 1) in [GLP_BV, GLP_IV]
        ]
        data = [1.0 if x[i] else -1.0 for i in binvars]
        row = [0] * len(binvars)
        A_ineq = sparse.csr_matrix((data, (row, binvars)), (1, numvars))
        b_ineq = sum([x[i] for i in binvars]) - 1
        self.add_ineq_constraints(A_ineq, [b_ineq])
