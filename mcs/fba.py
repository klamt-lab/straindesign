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
def fba(model,*kwargs):
    # Check type and size of A_ineq and b_ineq if they exist
    reaction_ids = model.reactions.list_attr("id")
    if ('A_ineq' in locals() or 'A_eq' in locals()) and 'const' in locals():
        raise Exception('Define either A_ineq, b_ineq or const, but not both.')
    if 'obj' in locals() and 'c' in locals():
        raise Exception('Define either obj or c, but not both.')
    if 'const' in locals():
        A_ineq, b_ineq, A_eq, b_eq = lineq2mat(const, reaction_ids)
    if 'obj' in locals():
        c = linexpr2mat(obj, reaction_ids)
    
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