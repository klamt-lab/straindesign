from typing import Dict, List, Tuple
from scipy import sparse
import re

def lineq2mat(equations, reaction_ids) -> Tuple[sparse.csr_matrix, Tuple, sparse.csr_matrix, Tuple]:
    numr = len(reaction_ids)
    A_ineq = sparse.csr_matrix((0, numr))
    b_ineq = []
    A_eq = sparse.csr_matrix((0, numr))
    b_eq = []
    for equation in equations:
        try:
            lhs, rhs = re.split('<=|=|>=', equation)
            eq_sign = re.search('<=|>=|=', equation)[0]
            rhs = float(rhs)
        except:
            raise Exception("Equations must contain exactly one (in)equality sign: <=,=,>=. Right hand side must be a float number.")
        A = linexpr2mat(lhs,reaction_ids)
        if eq_sign == '=':
            A_eq = sparse.vstack((A_eq, A))
            b_eq += [rhs]
        elif eq_sign == '<=':
            A_ineq = sparse.vstack((A_ineq, A))
            b_ineq += [rhs]
        elif eq_sign == '>=':
            A_ineq = sparse.vstack((A_ineq, -A))
            b_ineq += [-rhs]
    return A_ineq, b_ineq, A_eq, b_eq

def linexpr2mat(expr, reaction_ids) -> sparse.csr_matrix:
    # linexpr2mat translates the left hand side of a linear expression into a matrix
    #
    # e.g.: Model with reactions R1, R2, R3, R4
    #       Expression: '2 R3 - R1'
    #     translates into sparse matrix:
    #       A = [-1 0 2 0]
    # 
    A = sparse.lil_matrix((1, len(reaction_ids)))
    # split expression into parts and strip away special characters
    ridx = [re.sub(r'^(\s|-|\+|\()*|(\s|-|\+|\))*$', '', part) for part in expr.split()]
    # identify reaction identifiers by comparing with models reaction list
    ridx = [r for r in ridx if r in reaction_ids]
    if not len(ridx) == len(set(ridx)):  # check for duplicates
        raise Exception("Reaction identifiers may only occur once in each linear expression.")
    # iterate through reaction identifiers and retrieve coefficients from linear expression
    for rid in ridx:
        coeff = re.search('(\s|^)(\s|\d|-|\+|\.)*?(?=' + rid + '(\s|$))', expr)[0]
        coeff = re.sub('\s', '', coeff)
        if coeff in ['', '+']:
            coeff = 1
        if coeff == '-':
            coeff = -1
        else:
            coeff = float(coeff)
        A[0, reaction_ids.index(rid)] = coeff
    return A.tocsr()