from scipy import sparse
import numpy as np
import cplex as cp
import ray
import cobra
from cplex.exceptions import CplexError

# Collection of CPLEX-related functions that facilitate the creation
# of CPLEX-object and the solutions of LPs/MILPs with CPLEX from
# vector-matrix-based problem setups.
#
# functions: 
# 1. init_cpx_milp: Creates a CPLEX-object from a matrix-based problem setup.
#                   Supports LPs, MILPs (including indicator constraints)
#
# 2. cplex_fba: Perform FBA for cobra model with CPLEX
# 3. cplex_fva: Perform FVA for cobra model with CPLEX
#
#   Philipp Schneider (schneiderp@mpi-magdeburg.mpg.de)
# - December 2021
#  

# Create a CPLEX-object from a matrix-based problem setup
def init_cpx_milp(c,A_ineq,b_ineq,A_eq,b_eq,lb,ub,vtype=None,ic_binv=None,ic_A_ineq=None,ic_b_ineq=None,ic_sense=None,ic_indicval=None,x0=None,options=None):
    
    prob = cp.Cplex()
    prob.objective.set_sense(prob.objective.sense.minimize)

    numvars = len(c)
    # concatenate right hand sides
    b = b_ineq + b_eq
    # prepare coefficient matrix
    if isinstance(A_eq,list):
        if not A_eq:
            A_eq = sparse.csr_matrix((0,numvars))
    if isinstance(A_ineq,list):
        if not A_ineq:
            A_ineq = sparse.csr_matrix((0,numvars))
    A = sparse.vstack((A_ineq,A_eq),format='csr') # concatenate coefficient matrices
    sense = len(b_ineq)*'L' + len(b_eq)*'E'

    # construct CPLEX problem. Add variables and linear constraints
    if not vtype: # when undefined, all variables are continous
        vtype = 'C'*numvars
    prob.variables.add(obj=c, lb=lb, ub=ub, types=vtype)
    prob.linear_constraints.add(rhs=b, senses=sense)

    # retrieve row and column indices from sparse matrix and convert them to int
    sparse_indices_A = [[int(j) for j in i] for i in A.nonzero()]
    # convert matrix coefficients to float
    data_A = [float(i) for i in A.data]
    prob.linear_constraints.set_coefficients(zip(sparse_indices_A[0], sparse_indices_A[1], data_A))

    # add indicator constraints
    if ic_binv:
        # translate coefficient matrix A to right input format for CPLEX
        ic_A_ineq = sparse.csr_matrix(ic_A_ineq)
        sparse_indices_ic_A_ineq = [[int(j) for j in i] for i in ic_A_ineq.nonzero()]
        lin_expr = []
        for i in np.unique(sparse_indices_ic_A_ineq[0]):
            for row in ic_A_ineq.getrow(i):
                lin_expr+=[[[int(j) for j in row.indices],[float(i) for i in row.data]]]
        # call CPLEX function to add indicators
        prob.indicator_constraints.add_batch(lin_expr=lin_expr,
            sense=ic_sense,rhs=ic_b_ineq,indvar=ic_binv,complemented=ic_indicval)

    # set parameters
    prob.set_log_stream(None)
    prob.set_error_stream(None)
    prob.set_warning_stream(None)
    prob.set_results_stream(None)
    return prob

# FBA for cobra model with CPLEX
def cplex_fba(model):
    # prepare vectors and matrices
    S = cobra.util.create_stoichiometric_matrix(model)
    S = sparse.csr_matrix(S)
    lb = [v.lower_bound for v in model.reactions]
    ub = [v.upper_bound for v in model.reactions]
    c  = [i.objective_coefficient for i in model.reactions]
    numreac = len(model.reactions)
    if model.objective_direction == 'max':
        c = [ -i for i in c]
    # build CPLEX object
    cpx = init_cpx_milp(c,[],[],S,[0]*numreac,lb,ub)
    try:
        cpx.solve()
        fv = cpx.solution.get_values()
        max_v = cpx.solution.get_objective_value()
        if model.objective_direction == 'max':
            max_v = -max_v
        return fv, max_v
    except CplexError as exc:
        print(exc)
        max_v = np.nan
        fv = [np.nan] * numreac
        return fv, max_v

# FVA for cobra model with CPLEX
def cplex_fva(model,reacs=None):
    # prepare vectors and matrices
    S = cobra.util.create_stoichiometric_matrix(model)
    S = sparse.csr_matrix(S)
    lb = [v.lower_bound for v in model.reactions]
    ub = [v.upper_bound for v in model.reactions]
    numreac = len(model.reactions)
    c = [0]*numreac
    # build CPLEX object
    cpx = init_cpx_milp(c,[],[],S,[0]*numreac,lb,ub)
    # TODO: specify range of reactions for which FVA should be done
    if reacs:
        reacRange = []
    try:
        # maximize and minimize successively flux for all reactions
        for i in range(0,numreac-1): # TODO: Use user-specified range only
            if i >= 1:
                cpx.objective.set_linear(i-1,0)
            cpx.objective.set_linear(i,1)
            cpx.solve()
            lb[i] = cpx.solution.get_objective_value()
            cpx.objective.set_linear(i,-1)
            cpx.solve()
            ub[i] = -cpx.solution.get_objective_value()
        return lb, ub
    except CplexError as exc:
        print(exc)
        lb = [np.nan] * numreac
        ub = [np.nan] * numreac
        return lb,ub

# FVA for cobra model with CPLEX
@ray.remote
def cplex_fva_ray(model,reacs=None):
    ray.init(num_cpus=16, ignore_reinit_error=True)
    # prepare vectors and matrices
    S = cobra.util.create_stoichiometric_matrix(model)
    S = sparse.csr_matrix(S)
    lb = [v.lower_bound for v in model.reactions]
    ub = [v.upper_bound for v in model.reactions]
    numreac = len(model.reactions)
    c = [0]*numreac
    # build CPLEX object
    cpx = init_cpx_milp(c,[],[],S,[0]*numreac,lb,ub)
    # TODO: specify range of reactions for which FVA should be done
    if reacs:
        reacRange = []
    try:
        # maximize and minimize successively flux for all reactions
        for i in range(0,numreac-1): # TODO: Use user-specified range only
            if i >= 1:
                cpx.objective.set_linear(i-1,0)
            cpx.objective.set_linear(i,1)
            cpx.solve()
            lb[i] = cpx.solution.get_objective_value()
            cpx.objective.set_linear(i,-1)
            cpx.solve()
            ub[i] = -cpx.solution.get_objective_value()
        return lb, ub
    except CplexError as exc:
        print(exc)
        lb = [np.nan] * numreac
        ub = [np.nan] * numreac
        return lb,ub
