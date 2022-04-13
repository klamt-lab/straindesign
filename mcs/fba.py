import cobra
from scipy import sparse
from mcs import MILP_LP, parse_constraints, lineqlist2mat, linexpr2dict, linexprdict2mat
from mcs.names import *
from typing import Dict
# FBA for cobra model with CPLEX
# the user may provide the optional arguments
#   constraints:         Additional constraints in text form (list of lists)
#   A_ineq, b_ineq: Additional constraints in matrix form
#   obj:            Alternative objective in text form
def fba(model,**kwargs):
    # allowed_keys = {'obj','constraints','solver'}
    # # set all keys passed in kwargs
    # for key,value in kwargs.items():
    #     if key in allowed_keys:
    #         locals()[key] = value
    #     else:
    #         raise Exception("Key "+key+" is not supported.")
    # # set all remaining keys to None
    # for key in allowed_keys:
    #     if key not in kwargs.keys():
    #         locals()[key] = None
    # Check type and size of A_ineq and b_ineq if they exist
    reaction_ids = model.reactions.list_attr("id")

    if CONSTRAINTS in kwargs: 
        kwargs[CONSTRAINTS] = parse_constraints(kwargs[CONSTRAINTS],reaction_ids)
        A_ineq, b_ineq, A_eq, b_eq = lineqlist2mat(kwargs[CONSTRAINTS], reaction_ids)        

    if 'obj' in kwargs:
        if kwargs['obj'] is not None:
            if type(kwargs['obj']) is str:
                kwargs['obj'] = linexpr2dict(kwargs['obj'],reaction_ids)
            if type(kwargs['obj']) is dict:
                c = linexprdict2mat(kwargs['obj'],reaction_ids).toarray()[0].tolist()

    if 'solver' in kwargs:
        solver = kwargs['solver']
    else:
        solver = None
    
    # prepare vectors and matrices
    A_eq_base = cobra.util.create_stoichiometric_matrix(model)
    A_eq_base = sparse.csr_matrix(A_eq_base)
    b_eq_base = [0]*len(model.metabolites)
    if 'A_eq' in locals():
        A_eq  = sparse.vstack((A_eq_base, A_eq))
        b_eq  = b_eq_base+b_eq
    else:
        A_eq = A_eq_base
        b_eq = b_eq_base
    if 'A_ineq' not in locals():
        A_ineq = sparse.csr_matrix((0,len(model.reactions)))
        b_ineq = []
    lb = [v.lower_bound for v in model.reactions]
    ub = [v.upper_bound for v in model.reactions]
    if 'c' not in locals():
        c  = [i.objective_coefficient for i in model.reactions]
        if model.objective_direction == 'max':
            c = [ -i for i in c]
    else:
        c = [ -i for i in c]
    # build LP
    my_prob = MILP_LP(  c=c,
                        A_ineq=A_ineq,
                        b_ineq=b_ineq,
                        A_eq=A_eq,
                        b_eq=b_eq,
                        lb=lb,
                        ub=ub,
                        solver=solver)

    x, opt_cx, status = my_prob.solve()
    if status not in [OPTIMAL, UNBOUNDED]:
        status = INFEASIBLE
    fluxes = {reaction_ids[i] : x[i] for i in range(len(x))}
    sol = cobra.core.Solution(objective_value=-opt_cx,status=status,fluxes=fluxes)
    return sol