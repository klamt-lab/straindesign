from random import randint
from scipy import sparse
from numpy import isnan, nan, inf, isinf, sum, nonzero
import pyscipopt as pso
from straindesign.names import *
from typing import Tuple, List
import time as t
import logging

# Collection of SCIP-related functions that facilitate the creation
# of SCIP-object and the solutions of LPs/MILPs with SCIP from
# vector-matrix-based problem setups.
#


# Create a SCIP-object from a matrix-based problem setup
class SCIP_MILP(pso.Model):

    def __init__(self, c, A_ineq, b_ineq, A_eq, b_eq, lb, ub, vtype,
                 indic_constr):
        super().__init__()
        # uncomment to forward SCIP output to python terminal
        self.redirectOutput()
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
        x = [
            self.addVar(lb=l, ub=u, obj=o, vtype=v)
            for l, u, o, v in zip(lb, ub, c, vtype)
        ]
        self.vars = x
        self.binvars = [i for i in range(len(x)) if vtype[i] == 'B']
        # generate "Terms" and "Expressions" for faster problem construction
        self.trms = [list(x[i].terms.items())[0][0] for i in range(numvars)]

        self.constr = []
        # add inequality constraints
        ineqs = [self.addCons(pso.Expr() <= b_i) for b_i in b_ineq]
        for row, a_ineq in zip(ineqs, A_ineq):
            X = [x[i] for i in a_ineq.indices]
            for col, coeff in zip(X, a_ineq.data):
                self.addConsCoeff(row, col, coeff)
        self.constr += ineqs
        # add equality constraints
        eqs = [self.addCons(pso.Expr() == b_i) for b_i in b_eq]
        for row, a_eq in zip(eqs, A_eq):
            X = [x[i] for i in a_eq.indices]
            for col, coeff in zip(X, a_eq.data):
                self.addConsCoeff(row, col, coeff)
        self.constr += eqs

        self.setMinimize()
        # add indicator constraints
        if indic_constr is not None:
            for i in range(len(indic_constr.sense)):
                if indic_constr.indicval[
                        i] == 0:  # if the constraints activity is indicated by 0, an auxiliary variable needs to be added
                    z = self.addVar(lb=0, ub=1, obj=0, vtype='B')
                    xor_constr = self.addConsXor([x[indic_constr.binv[i]], z],
                                                 True)
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
                        pass
                        e = pso.scip.Expr({
                            self.trms[j]: d for j, d in zip(a.indices, a.data)
                        })
                        f = pso.scip.ExprCons(e, lhs=None, rhs=b[k])
                        self.constr += [
                            self.addConsIndicator(f, binvar=z, initial=False)
                        ]

        # set parameters
        self.max_tlim = self.getParam('limits/time')
        self.setParam('display/verblevel', 0)
        if 'B' in vtype or 'I' in vtype:
            seed = seed = randint(0, 2**31 - 1)
            # logging.info('  MILP Seed: '+str(seed))
            self.setParam('randomization/randomseedshift', seed)
        # self.enableReoptimization()
        # self.setParam('display/lpinfo',False)
        # self.setParam('reoptimization/enable',True)

    def solve(self) -> Tuple[List, float, float]:
        try:
            self.optimize()
            status = self.getStatus()
            if status in ['optimal']:  # solution
                min_cx = self.getObjVal()
                status = OPTIMAL
            elif status == 'timelimit' and self.getSols(
            ) == []:  # timeout without solution
                x = [nan] * len(self.vars)
                min_cx = nan
                status = TIME_LIMIT
                return x, min_cx, status
            elif status == 'infeasible':  # infeasible
                x = [nan] * len(self.vars)
                min_cx = nan
                status = INFEASIBLE
                return x, min_cx, status
            elif status == 'timelimit' and not self.getSols(
            ) == []:  # timeout with solution
                min_cx = self.getObjVal()
                status = TIME_LIMIT_W_SOL
            elif status in ['inforunbd', 'unbounded']:  # solution unbounded
                x = [nan] * len(self.vars)
                min_cx = -inf
                status = UNBOUNDED
                return x, min_cx, status
            else:
                raise Exception('Status code ' + str(status) +
                                " not yet handeld.")
            x = self.getSolution()
            return x, min_cx, status

        except:
            logging.error('Error while running SCIP.')
            min_cx = nan
            x = [nan] * len(self.vars)
            return x, min_cx, ERROR

    def slim_solve(self) -> float:
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
                raise Exception('Status code ' + str(status) +
                                " not yet handeld.")
            return opt

        except:
            logging.error('Error while running SCIP.')
            return nan

    def populate(self, pool_limit) -> Tuple[List, float, float]:
        numrows = len(self.constr)

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
                c = [
                    objTerms[x] if x in objTerms.keys() else 0.0
                    for x in self.trms
                ]
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
        if self.getParam('reoptimization/enable'):
            self.freeReoptSolve()
            self.chgReoptObjective(
                pso.Expr({self.trms[i]: c[i] for i in nonzero(c)[0]}))
        else:
            self.freeTransform()
            self.setObjective(
                pso.Expr({self.trms[i]: c[i] for i in nonzero(c)[0]}))

    def set_objective_idx(self, C):
        if self.getParam('reoptimization/enable'):
            self.freeReoptSolve()
            self.chgReoptObjective(pso.Expr({self.trms[c[0]]: c[1] for c in C}))
        else:
            self.freeTransform()
            self.setObjective(pso.Expr({self.trms[c[0]]: c[1] for c in C}))

    def set_ub(self, ub):
        self.freeTransform()
        for i in range(len(ub)):
            if not isinf(ub[i][1]):
                self.chgVarUb(self.vars[ub[i][0]], float(ub[i][1]))
            else:
                self.chgVarUb(self.vars[ub[i][0]], None)

    def set_time_limit(self, t):
        if t >= self.max_tlim:
            self.setParam('limits/time', self.max_tlim)
        else:
            self.setParam('limits/time', t)

    def add_ineq_constraints(self, A_ineq, b_ineq):
        self.freeTransform()
        ineqs = [self.addCons(pso.Expr() <= b_i) for b_i in b_ineq]
        for row, a_ineq in zip(ineqs, A_ineq):
            X = [self.vars[i] for i in a_ineq.indices]
            for col, coeff in zip(X, a_ineq.data):
                self.addConsCoeff(row, col, float(coeff))
        self.constr += ineqs

    def add_eq_constraints(self, A_eq, b_eq):
        self.freeTransform()
        eqs = [self.addCons(pso.Expr() == b_i) for b_i in b_eq]
        for row, a_eq in zip(eqs, A_eq):
            X = [self.vars[i] for i in a_eq.indices]
            for col, coeff in zip(X, a_eq.data):
                self.addConsCoeff(row, col, float(coeff))
        self.constr += eqs

    def set_ineq_constraint(self, idx, a_ineq, b_ineq):
        self.freeTransform()
        # Make previous constraint non binding. removing or
        # changing old constraints would be better but doesn't work
        self.chgRhs(self.constr[idx], None)
        # add new constraint and replace constraint pointer in list
        self.constr[idx] = self.addCons(pso.Expr() <= 0)
        for i, a in enumerate(a_ineq):
            self.addConsCoeff(self.constr[idx], self.vars[i], a)
        if isinf(b_ineq):
            self.chgRhs(self.constr[idx], None)
        else:
            self.chgRhs(self.constr[idx], b_ineq)
        pass

    def getSolution(self) -> list:
        return [self.getVal(x) for x in self.vars]

    def addExclusionConstraintIneq(self, x):
        data = [1.0 if x[i] else -1.0 for i in self.binvars]
        row = [0] * len(self.binvars)
        A_ineq = sparse.csr_matrix((data, (row, self.binvars)),
                                   (1, len(self.vars)))
        b_ineq = sum([x[i] for i in self.binvars]) - 1
        self.add_ineq_constraints(A_ineq, [b_ineq])


# Create a SCIP-object from a matrix-based problem setup
class SCIP_LP(pso.LP):

    def __init__(self, c, A_ineq, b_ineq, A_eq, b_eq, lb, ub):
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
        try:
            min_cx = self.optimize(
            )  # this function was inherited from super().solve() during initialization
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
        try:
            opt = self.optimize(
            )  # this function was inherited from super().solve() during initialization
            if self.isInfinity(-opt):  # solution
                opt = -inf
            elif self.isInfinity(opt):
                opt = nan
            return opt
        except:
            logging.error('Error while running SCIP.')
            return nan

    def set_objective(self, c):
        for i in range(len(c)):
            self.chgObj(i, c[i])

    def set_objective_idx(self, C):
        for i_v in C:
            self.chgObj(i_v[0], i_v[1])

    def add_ineq_constraints(self, A_ineq, b_ineq):
        self.addRows([[(i,v) for i,v in zip(rows.indices,rows.data)] for rows in A_ineq], \
                        lhss = [-self.infinity()]*A_ineq.shape[0],\
                        rhss = b_ineq)

    def add_eq_constraints(self, A_eq, b_eq):
        self.addRows([[(i,v) for i,v in zip(rows.indices,rows.data)] for rows in A_eq], \
                        lhss = b_eq,\
                        rhss = b_eq)
