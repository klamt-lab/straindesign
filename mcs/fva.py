from optlang.interface import OPTIMAL
from scipy import sparse
import mcs
from typing import Tuple
from pandas import DataFrame
import numpy as np
from cobra.core import Configuration
from cobra.util import ProcessPool, create_stoichiometric_matrix

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

def worker_init(A_ineq,b_ineq,A_eq,b_eq,lb,ub,x0):
    global lp_glob
    lp_glob = mcs.MILP_LP(A_ineq=A_ineq, b_ineq=b_ineq, A_eq=A_eq, b_eq=b_eq,
                                    lb=lb, ub=ub, x0=x0)
    global prev
    prev=0

def worker_compute(i):
    global lp_glob
    global prev
    print(i)
    c = idx2c(i,len(lp_glob.ub))
    lp_glob.set_objective(c)
    _, min_cx, _ = lp_glob.solve()
    prev=i
    return (i,min_cx)

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
    A_eq_base = create_stoichiometric_matrix(model)
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
    x0, _, status = lp.solve()
    if status is not 0:
        raise Exception('FVA problem not feasible.')

    processes = Configuration().processes
    num_reactions = len(reaction_ids)
    processes = min(processes, num_reactions)

    x = [np.nan]*2*numr

    if processes > 1:
        with ProcessPool(processes,initializer=worker_init,initargs=(A_ineq,b_ineq,A_eq,b_eq,lb,ub,x0),
                        ) as pool:
            print('initialized')
            chunk_size = len(reaction_ids) // processes
            # x = pool.imap_unordered(worker_compute, range(2*numr), chunksize=chunk_size)
            for i, value in pool.imap_unordered( worker_compute, range(2*numr), chunksize=chunk_size):
                x[i] = value

        # C = np.array_split(range(2*numr),20)
        # x = pool.map(worker_compute, C)
        # with mp.Pool(processes=mp.cpu_count(), initializer=worker_init, initargs=(A_ineq,b_ineq,A_eq,b_eq,lb,ub)) as pool:

        # x = np.concatenate(x)

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

