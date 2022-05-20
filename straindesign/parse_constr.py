# -*- coding: utf-8 -*-
"""This module contains functions for parsing and converting constraints and linear expressions."""
from typing import Dict, List, Tuple
from scipy import sparse
import re


def parse_constraints(constr, reaction_ids) -> List:
    """Parses constraints written as strings
    
    This is a longer description of the parse constraints function.
    
    Args:
        constr (:class:str or :class:List): (List of) constraints in string form.
        reaction_ids (:class:List): List of reaction identifiers.

    Returns:
        parsed_constr: List of constraints. Each constraint is a list of three elements. [[dict_v,'=',0.3],[dict_w,'<=',0.5],...]
    """
    if not constr:
        return []
    if type(constr) is str:
        if "\n" in constr or "," in constr:
            constr = re.split(r"\n|,", constr)
    if bool(constr) and (type(constr) is not list or type(constr[0]) is dict):
        constr = [constr]
    for i, c in enumerate(constr):
        if type(c) is tuple:
            constr[i] = list(c)
    if type(constr[0]) is not list:
        constr = lineq2list(constr, reaction_ids)
    return constr


def parse_linexpr(expr, reaction_ids) -> List:
    if not expr:
        return []
    if type(expr) is str:
        if "\n" in expr or "," in expr:
            expr = re.split(r"\n|,", expr)
    if bool(expr) and (type(expr) is not list):
        expr = [expr]
    return [
        linexpr2dict(e, reaction_ids) if type(e) is str else e for e in expr
    ]


def lineq2mat(
        equations, reaction_ids
) -> Tuple[sparse.csr_matrix, Tuple, sparse.csr_matrix, Tuple]:
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
            raise Exception(
                "Equations must contain exactly one (in)equality sign: <=,=,>=. Right hand side must be a float number."
            )
        A = linexpr2mat(lhs, reaction_ids)
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


def lineq2list(equations, reaction_ids) -> List:
    D = []
    for equation in equations:
        if not equation:
            continue
        try:
            lhs, rhs = re.split('<=|=|>=', equation)
            eq_sign = re.search('<=|>=|=', equation)[0]
            rhs = float(rhs)
        except:
            raise Exception(
                "Equations must contain exactly one (in)equality sign: <=,=,>=. Right hand side must be a float number."
            )
        D.append((linexpr2dict(lhs, reaction_ids), eq_sign, rhs))
    return D


def lineqlist2str(D):
    if D[0]:
        return linexprdict2str(D[0]) + " " + D[1] + " " + str(D[2])
    elif D[1] and D[2]:
        return D[1] + " " + str(D[2])
    else:
        return ""


def lineqlist2mat(
        D, reaction_ids
) -> Tuple[sparse.csr_matrix, Tuple, sparse.csr_matrix, Tuple]:
    numr = len(reaction_ids)
    A_ineq = sparse.csr_matrix((0, numr))
    b_ineq = []
    A_eq = sparse.csr_matrix((0, numr))
    b_eq = []
    for d in D:
        d_expr = linexprdict2mat(d[0], reaction_ids)
        eq_sign = d[1]
        rhs = d[2]
        if eq_sign == '=':
            A_eq = sparse.vstack((A_eq, d_expr))
            b_eq += [rhs]
        elif eq_sign == '<=':
            A_ineq = sparse.vstack((A_ineq, d_expr))
            b_ineq += [rhs]
        elif eq_sign == '>=':
            A_ineq = sparse.vstack((A_ineq, -d_expr))
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
    expr_parts = [
        re.sub(r'^(\s|-|\+|\()*|(\s|-|\+|\))*$', '', part)
        for part in expr.split()
    ]
    # identify reaction identifiers by comparing with models reaction list
    ridx = [r for r in expr_parts if r in reaction_ids]
    # verify syntax of expression
    # 1. there must not be two numbers in a row
    # 2. there must not remain words that are not reaction identifiers
    # 3. there must be no reaction id duplicates
    last_was_number = False
    for part in expr_parts:
        if part in ridx:
            last_was_number = False
            continue
        if re.match('^\d*\.{0,1}\d*$', part) is not None:
            if last_was_number:
                raise Exception(
                    "Expression invalid. The expression contains at least two numbers in a row."
                )
            last_was_number = True
            continue
        raise Exception("Expression invalid. Unknown identifier " + part + ".")
    if not len(ridx) == len(set(ridx)):
        raise Exception(
            "Reaction identifiers may only occur once in each linear expression."
        )
    # iterate through reaction identifiers and retrieve coefficients from linear expression
    for rid in ridx:
        coeff = re.search(
            '(\s|^)(\s|\d|-|\+|\.)*?(?=' + re.escape(rid) + '(\s|$))', expr)[0]
        coeff = re.sub('\s', '', coeff)
        if coeff in ['', '+']:
            coeff = 1.0
        if coeff == '-':
            coeff = -1.0
        else:
            coeff = float(coeff)
        A[0, reaction_ids.index(rid)] = coeff
    return A.tocsr()


def linexpr2dict(expr, reaction_ids) -> dict:
    # linexpr2dict translates the left hand side of a linear expression into a dict
    #
    # e.g.: Model with reactions R1, R2, R3, R4
    #       Expression: '2 R3 - R1'
    #     translates to dict:
    #       D = {R1:-1, R3: 2}
    #
    # split expression into parts and strip away special characters
    expr_parts = [
        re.sub(r'^(\s|-|\+|\()*|(\s|-|\+|\))*$', '', part)
        for part in expr.split()
    ]
    expr_parts = [e for e in expr_parts if e != '']  # remove 'empty' entries
    # identify reaction identifiers by comparing with models reaction list
    ridx = [r for r in expr_parts if r in reaction_ids]
    # verify syntax of expression
    # 1. there must not be two numbers in a row
    # 2. there must not remain words that are not reaction identifiers
    # 3. there must be no reaction id duplicates
    last_was_number = False
    for part in expr_parts:
        if part in ridx:
            last_was_number = False
            continue
        if re.match('^\d*\.{0,1}\d*$', part) is not None:
            if last_was_number:
                raise Exception(
                    "Expression invalid. The expression contains at least two numbers in a row."
                )
            last_was_number = True
            continue
        raise Exception("Expression invalid. Unknown identifier " + part + ".")
    if not len(ridx) == len(set(ridx)):
        raise Exception(
            "Reaction identifiers may only occur once in each linear expression."
        )
    D = {}
    # iterate through reaction identifiers and retrieve coefficients from linear expression
    for rid in ridx:
        coeff = re.search(
            '(\s|^)(\s|\d|-|\+|\.)*?(?=' + re.escape(rid) + '(\s|$))', expr)[0]
        coeff = re.sub('\s', '', coeff)
        if coeff in ['', '+']:
            coeff = 1.0
        elif coeff == '-':
            coeff = -1.0
        else:
            coeff = float(coeff)
        D.update({rid: coeff})
    return D


def linexprdict2mat(D, reaction_ids) -> sparse.csr_matrix:
    # linedict2mat a dict that describes a linear expresson into a matrix
    #
    # e.g.: Model with reactions R1, R2, R3, R4
    #       Dict: {R3: 2, R1:-1}
    #     translates into sparse matrix:
    #       A = [-1 0 2 0]
    #
    # iterate through reaction identifiers and retrieve coefficients from linear expression
    A = sparse.lil_matrix((1, len(reaction_ids)))
    for k, v in D.items():
        A[0, reaction_ids.index(k)] = v
    return A.tocsr()


def linexprdict2str(D):
    if D:
        expr_parts = [str(v) + " " + k for k, v in D.items()]
        expr = expr_parts[0]
        for ep in expr_parts[1:]:
            if ep[0] == '-':
                expr += ' - ' + ep[1:]
            else:
                expr += ' + ' + ep
        return expr
    else:
        return ""


def get_rids(expr, reaction_ids):
    expr_parts = [
        re.sub(r'^(\s|-|\+|\()*|(\s|-|\+|\|<|\=|>)*$', '', part)
        for part in expr.split()
    ]
    reacIDs = []
    for part in expr_parts:
        if part in reaction_ids:
            reacIDs += [part]
            continue
        if re.match('^\d*\.{0,1}\d*$', part) is not None:
            continue
        raise Exception("Expression invalid. Unknown identifier " + part + ".")
    return reacIDs
