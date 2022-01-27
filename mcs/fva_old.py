import cobra
from optlang.interface import OPTIMAL
from scipy import sparse
import mcs
from typing import Tuple
from pandas import DataFrame
from sympy import E
import numpy as np

# FBA for cobra model with CPLEX
# the user may provide the optional arguments
#   constr:         Additional constraints in text form (list of lists)
#   A_ineq, b_ineq: Additional constraints in matrix form
#   obj:            Alternative objective in text form
#   c:              Alternative objective in vector form

def idx2c(i,n) -> int:
    col = np.floor(i/2)
    sign = np.sign(np.mod(i,2)-0.5)
    c = [0 if not j == col else sign for j in range(n)]
    return c

def fva(model,*kwargs):
    try:
        import ray
    except:
        pass

    # Check type and size of A_ineq and b_ineq if they exist
    reaction_ids = model.reactions.list_attr("id")
    if ('A_ineq' in locals() or 'A_eq' in locals()) and 'const' in locals():
        raise Exception('Define either A_ineq, b_ineq or const, but not both.')
    if 'const' in locals():
        A_ineq, b_ineq, A_eq, b_eq = mcs.lineq2mat(const, reaction_ids)
    
    numr = len(model.reactions)
    # prepare vectors and matrices
    A_eq_base = cobra.util.create_stoichiometric_matrix(model)
    A_eq_base = sparse.csr_matrix(A_eq_base)
    b_eq_base = [0]*len(model.metabolites)
    if 'A_eq' in locals():
        A_eq  = sparse.vstack((A_eq_base, A_eq_supp))
        b_eq  = b_eq_base+b_eq
    else:
        A_eq = A_eq_base
        b_eq = b_eq_base
    if 'A_ineq' not in locals():
        A_ineq = sparse.csr_matrix((0,numr))
        b_ineq = []
    lb = [v.lower_bound for v in model.reactions]
    ub = [v.upper_bound for v in model.reactions]

    # build LP
    lp = mcs.MILP_LP(   A_ineq=A_ineq,
                    b_ineq=b_ineq,
                    A_eq=A_eq,
                    b_eq=b_eq,
                    lb=lb,
                    ub=ub,
                    solver='cplex')
    _, _, status = lp.solve()
    if status is not 0:
        raise Exception('FVA problem not feasible.')
    x = [np.nan]*2*numr
    if 'ray' in locals() and ray.is_initialized():
        # Build an Actor - a stateful worker based on a class
        @ray.remote  # class is decorated with ray.remote to for parallel use
        class M_optimizer(object):
            def __init__(self):  # The LP object is only constructed once upon Actor creation.
                self.lp = mcs.MILP_LP(A_ineq=A_ineq, b_ineq=b_ineq, A_eq=A_eq, b_eq=b_eq,
                                                lb=lb, ub=ub)
                self.numr = len(lb)
                self.lp.cpx.parameters.threads.set(1)
            def compute(self, idx_set):  # With each function call only the objective function is changed
                for i in range(len(idx_set)):
                    c = idx2c(idx_set[i],self.numr)
                    self.lp.set_objective(c)
                    _, x[i], _ = self.lp.solve()
                return x
        # b) Create pool of Actors on which the computations should be executed. Number of Actors = number of CPUs
        numcpus = int(ray.available_resources()['CPU'])
        parpool = ray.util.ActorPool([M_optimizer.remote() for _ in range(numcpus)])
        # c) Run M computations on actor pool. lambda is an inline function
        C = np.array_split(range(2*numr),numcpus)
        x = list(parpool.map(lambda a, x: a.compute.remote(x), C))
    else:
        for i in range(2*numr):
            lp.set_objective(idx2c(i,numr))
            _, x[i], _ = lp.solve()
    
    fva_result = DataFrame(
        {
            "minimum": [ x[i] for i in range(0,2*numr,2)],
            "maximum": [-x[i] for i in range(1,2*numr,2)],
        },
        index=reaction_ids,
    )

    return fva_result

