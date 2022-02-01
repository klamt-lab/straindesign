import cobra
from optlang.interface import OPTIMAL
from scipy import sparse
from mcs import MILP_LP
from mcs.constr2mat import *
from typing import Dict
# FBA for cobra model with CPLEX
# the user may provide the optional arguments
#   constr:         Additional constraints in text form (list of lists)
#   A_ineq, b_ineq: Additional constraints in matrix form
#   obj:            Alternative objective in text form
#   c:              Alternative objective in vector form
def fba(model,**kwargs):
    allowed_keys = {'obj', 'A_ineq','b_ineq','A_eq','b_eq','constr','c','obj'}
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
    numr = len(model.reactions)
    if ('A_ineq' in kwargs or 'A_ineq' in kwargs) and 'constr' in kwargs:
        raise Exception('Define either A_ineq, b_ineq or constr, but not both.')
    if 'obj' in kwargs and 'c' in kwargs:
        raise Exception('Define either obj or c, but not both.')
        
    if 'constr' in kwargs:
        A_ineq, b_ineq, A_eq, b_eq = lineq2mat(kwargs['constr'], reaction_ids)
    if 'A_ineq' in kwargs and 'b_ineq' in kwargs:
        A_ineq = kwargs['A_ineq']
        b_ineq = kwargs['b_ineq']
    if 'A_eq' in kwargs and 'b_eq' in kwargs:
        A_eq = kwargs['A_eq']
        b_eq = kwargs['b_eq']
    else:
        A_eq = sparse.csr_matrix((0,numr))
        b_eq = []
    if 'obj' in kwargs:
        c = linexpr2mat(kwargs['obj'], reaction_ids)
    elif 'c' in kwargs:
        c = kwargs['c']
    
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
    # build LP
    my_prob = MILP_LP(  c=c,
                        A_ineq=A_ineq,
                        b_ineq=b_ineq,
                        A_eq=A_eq,
                        b_eq=b_eq,
                        lb=lb,
                        ub=ub,
                        solver='cplex')

    x, opt_cx, status = my_prob.solve()
    if status == 0:
        status = OPTIMAL
    fluxes = {reaction_ids[i] : x[i] for i in range(len(x))}
    sol = cobra.core.Solution(objective_value=-opt_cx,status=status,fluxes=fluxes)
    return sol