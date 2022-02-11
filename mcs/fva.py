from optlang.interface import OPTIMAL
from scipy import sparse
from mcs import MILP_LP, lineq2mat
from typing import Tuple
from pandas import DataFrame
from numpy import floor, sign, mod, nan, unique
from cobra.core import Configuration
from cobra.util import ProcessPool, create_stoichiometric_matrix, solvers
import cplex

# FBA for cobra model with CPLEX
# the user may provide the optional arguments
#   constr:         Additional constraints in text form (list of lists)
#   A_ineq, b_ineq: Additional constraints in matrix form
#   obj:            Alternative objective in text form
#   c:              Alternative objective in vector form
def idx2c(i,prev):
    col = int(floor(i/2))
    sig = sign(mod(i,2)-0.5)
    C = [[col,sig],[prev,0.0]]
    C_idx = [C[i][0] for i in range (len(C))]
    C_idx = unique([C_idx.index(C_idx[i]) for i in range(len(C_idx))])
    C = [C[i] for i in C_idx]
    return C

def worker_init(A_ineq,b_ineq,A_eq,b_eq,lb,ub,x0,solver):
    global lp_glob
    lp_glob = MILP_LP(A_ineq=A_ineq, b_ineq=b_ineq, A_eq=A_eq, b_eq=b_eq,
                                    lb=lb, ub=ub, x0=x0,solver=solver)
    if lp_glob.solver == 'cplex':
        lp_glob.backend.parameters.threads.set(2)
        lp_glob.backend.parameters.lpmethod.set(1)
    elif lp_glob.solver == 'gurobi':
        lp_glob.backend.params
    # elif 'scip' in avail_solvers:
    # else:
    lp_glob.prev = 0

def worker_compute(i) -> Tuple[int,float]:
    global lp_glob
    C = idx2c(i,lp_glob.prev)
    if lp_glob.solver in ['cplex','gurobi']:
        lp_glob.backend.set_objective_idx(C)
        min_cx = lp_glob.backend.slim_solve()
    else:
        lp_glob.set_objective_idx(C)
        min_cx = lp_glob.slim_solve()

    lp_glob.prev = C[0][0]
    return i, min_cx

def fva(model,**kwargs):
    reaction_ids = model.reactions.list_attr("id")
    numr = len(model.reactions)
    if ('A_ineq' in kwargs or 'A_ineq' in kwargs) and 'constr' in kwargs:
        raise Exception('Define either A_ineq, b_ineq or constr, but not both.')
        
    if 'constr' in kwargs:
        A_ineq, b_ineq, A_eq, b_eq = lineq2mat(kwargs['constr'], reaction_ids)
    else:
        if 'A_ineq' in kwargs and 'b_ineq' in kwargs:
            A_ineq = kwargs['A_ineq']
            b_ineq = kwargs['b_ineq']
        if 'A_eq' in kwargs and 'b_eq' in kwargs:
            A_eq = kwargs['A_eq']
            b_eq = kwargs['b_eq']
        else:
            A_eq = sparse.csr_matrix((0,numr))
            b_eq = []
    if 'solver' in kwargs:
        solver = kwargs['solver']
    else:
        solver = None
    
    
    # prepare vectors and matrices
    A_eq_base = create_stoichiometric_matrix(model)
    A_eq_base = sparse.csr_matrix(A_eq_base)
    b_eq_base = [0]*len(model.metabolites)
    if 'A_eq' in locals():
        A_eq  = sparse.vstack((A_eq_base, A_eq))
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
    lp = MILP_LP(   A_ineq=A_ineq,
                    b_ineq=b_ineq,
                    A_eq=A_eq,
                    b_eq=b_eq,
                    lb=lb,
                    ub=ub,
                    solver=solver)
    x0, _, status = lp.solve()
    if status is not 0:
        raise Exception('FVA problem not feasible.')

    processes = Configuration().processes
    num_reactions = len(reaction_ids)
    processes = min(processes, num_reactions)

    x = [nan]*2*numr

    # Dummy to check if optimization runs
    worker_init(A_ineq,b_ineq,A_eq,b_eq,lb,ub,x0,solver)
    worker_compute(1)

    if processes > 1:
        with ProcessPool(processes,initializer=worker_init,initargs=(A_ineq,b_ineq,A_eq,b_eq,lb,ub,x0,solver)) as pool:
            chunk_size = len(reaction_ids) // processes
            # x = pool.imap_unordered(worker_compute, range(2*numr), chunksize=chunk_size)
            for i, value in pool.imap_unordered( worker_compute, range(2*numr), chunksize=chunk_size):
                x[i] = value
    else:
        worker_init(A_ineq,b_ineq,A_eq,b_eq,lb,ub,x0,solver)
        for i in range(2*numr):
            lp.set_objective_idx(idx2c(i))
            _, x[i] = worker_compute(i)
    
    fva_result = DataFrame(
        {
            "minimum": [ x[i] for i in range(1,2*numr,2)],
            "maximum": [-x[i] for i in range(0,2*numr,2)],
        },
        index=reaction_ids,
    )

    return fva_result

