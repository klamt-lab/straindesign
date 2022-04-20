from scipy import sparse
from straindesigner import MILP_LP, parse_constraints, lineqlist2mat, Pool
from straindesigner.names import *
from typing import Tuple
from pandas import DataFrame
from numpy import floor, sign, mod, nan, unique
from os import cpu_count
from cobra.util import create_stoichiometric_matrix
# from cobra.util import ProcessPool, create_stoichiometric_matrix

# FBA for cobra model with CPLEX
# the user may provide the optional arguments
#   constraints:    Additional constraints in text form (list of lists)
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

def worker_init(A_ineq,b_ineq,A_eq,b_eq,lb,ub,solver):
    global lp_glob
    lp_glob = MILP_LP(A_ineq=A_ineq, b_ineq=b_ineq, A_eq=A_eq, b_eq=b_eq,
                                    lb=lb, ub=ub, solver=solver)
    if lp_glob.solver == 'cplex':
        lp_glob.backend.parameters.threads.set(1)
        #lp_glob.backend.parameters.lpmethod.set(1)
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
        
    if CONSTRAINTS in kwargs: 
        kwargs[CONSTRAINTS] = parse_constraints(kwargs[CONSTRAINTS],reaction_ids)
        A_ineq, b_ineq, A_eq, b_eq = lineqlist2mat(kwargs[CONSTRAINTS], reaction_ids) 

    if SOLVER in kwargs:
        solver = kwargs[SOLVER]
    else:
        solver = None
    
    # prepare vectors and matrices
    A_eq_base = sparse.csr_matrix(create_stoichiometric_matrix(model))
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
    _, _, status = lp.solve()
    if status not in [OPTIMAL,UNBOUNDED]: # if problem not feasible or unbounded
        raise Exception('FVA problem not feasible.')

    processes = cpu_count()-1
    if not processes:
        print("The number of cores could not be detected - assuming one.")
        processes = 1
    num_reactions = len(reaction_ids)
    processes = min(processes, num_reactions)

    x = [nan]*2*numr

    # Dummy to check if optimization runs
    # worker_init(A_ineq,b_ineq,A_eq,b_eq,lb,ub,solver)
    # worker_compute(1)
    if processes > 1 & numr > 300:
        pool = Pool(processes,initializer=worker_init,initargs=(A_ineq,b_ineq,A_eq,b_eq,lb,ub,solver))
        chunk_size = len(reaction_ids) // processes
        # x = pool.imap_unordered(worker_compute, range(2*numr), chunksize=chunk_size)
        for i, value in pool.imap_unordered( worker_compute, range(2*numr), chunksize=chunk_size):
            x[i] = value
    else:
        worker_init(A_ineq,b_ineq,A_eq,b_eq,lb,ub,solver)
        for i in range(2*numr):
            _, x[i] = worker_compute(i)
    
    fva_result = DataFrame(
        {
            "minimum": [ x[i] for i in range(1,2*numr,2)],
            "maximum": [-x[i] for i in range(0,2*numr,2)],
        },
        index=reaction_ids,
    )

    return fva_result

