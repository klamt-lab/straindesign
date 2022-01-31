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

def worker_init(A_ineq,b_ineq,A_eq,b_eq,lb,ub,x0,solver):
    global lp_glob
    lp_glob = MILP_LP(A_ineq=A_ineq, b_ineq=b_ineq, A_eq=A_eq, b_eq=b_eq,
                                    lb=lb, ub=ub, x0=x0,solver=solver)
    avail_solvers = list(solvers.keys())
    if 'cplex' in avail_solvers:
        lp_glob = lp_glob.cpx
        lp_glob.parameters.threads.set(2)
        lp_glob.parameters.lpmethod.set(1)
    # elif 'gurobi' in avail_solvers:
    # elif 'scip' in avail_solvers:
    # else:
    lp_glob.solver = solver
    lp_glob.prev = 0

def worker_compute(i) -> Tuple[int,float]:
    global lp_glob
    col = int(floor(i/2))
    sig = sign(mod(i,2)-0.5)
    C = [[col,sig],[lp_glob.prev,0.0]]
    C_idx = [C[i][0] for i in range (len(C))]
    C_idx = unique([C_idx.index(C_idx[i]) for i in range(len(C_idx))])
    C = [C[i] for i in C_idx]

    if lp_glob.solver == 'cplex':
        lp_glob.objective.set_linear(C)
        lp_glob.solve()
        min_cx = lp_glob.solution.get_objective_value()
    else:
        lp_glob.set_objective_idx(C)
        min_cx = lp_glob.slim_solve()

    lp_glob.prev = col
    return i, min_cx

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
        A_ineq, b_ineq, A_eq, b_eq = lineq2mat(const, reaction_ids)
    
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
    lp = MILP_LP(   A_ineq=A_ineq,
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

    x = [nan]*2*numr

    # Dummy to check if optimization runs
    worker_init(A_ineq,b_ineq,A_eq,b_eq,lb,ub,x0,list(solvers.keys())[0])
    worker_compute(1)

    if processes > 1:
        with ProcessPool(processes,initializer=worker_init,initargs=(A_ineq,b_ineq,A_eq,b_eq,lb,ub,
                        x0,list(solvers.keys())[0])) as pool:
            print('initialized')
            chunk_size = len(reaction_ids) // processes
            # x = pool.imap_unordered(worker_compute, range(2*numr), chunksize=chunk_size)
            for i, value in pool.imap_unordered( worker_compute, range(2*numr), chunksize=chunk_size):
                x[i] = value
    else:
        for i in range(2*numr):
            lp.set_objective(idx2c(i,numr))
            _, x[i], _ = lp.solve()
    
    fva_result = DataFrame(
        {
            "minimum": [ x[i] for i in range(1,2*numr,2)],
            "maximum": [-x[i] for i in range(0,2*numr,2)],
        },
        index=reaction_ids,
    )

    return fva_result

