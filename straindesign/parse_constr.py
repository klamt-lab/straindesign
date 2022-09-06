#!/usr/bin/env python3
#
# Copyright 2022 Max Planck Insitute Magdeburg
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
#
"""Functions for parsing and converting constraints and linear expressions"""

from typing import Dict, List, Tuple
from scipy import sparse
import re


def parse_constraints(constr, reaction_ids) -> list:
    """Parses linear constraints written as strings
    
    Parses one or more *linear* constraints written as strings.
    
    Args:
        constr (str or list of str): 
            (List of) constraints in string form.
            E.g.: ['r1 + 3*r2 = 0.3', '-5*r3 -r4 <= -0.5'] or
            '1.0 r1 + 3.0*r2 =0.3,-r4-5*r3<=-0.5' or ...
            
        reaction_ids (list of str): 
            List of reaction identifiers.

    Returns:
        (List of dicts): 
        List of constraints. Each constraint is a list of three elements.
        E.g.: [[{'r1':1.0,'r2':3.0},'=',0.3],[{'r3':-5.0,'r4':-1.0},'<=',-0.5],...]
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
    """Parses linear expressions written as strings
    
    Parses one or more *linear* expressions written as strings.
    
    Args:
        expr (str or list of str): 
            (List of) expressions in string form.
            E.g.: ['r1 + 3*r2', '-5*r3 -r4'] or
            '1.0 r1 + 3.0*r2,-r4-5*r3' or ...
            
        reaction_ids (list of str): 
            List of reaction identifiers.

    Returns:
        (List of dicts): 
        List of expressions. Each expression is a dictionary.
        E.g.: [{'r1':1.0,'r2':3.0},{'r3':-5.0,'r4':-1.0},...]
    """
    if not expr:
        return []
    if type(expr) is str:
        if "\n" in expr or "," in expr:
            expr = re.split(r"\n|,", expr)
    if bool(expr) and (type(expr) is not list):
        expr = [expr]
    return [linexpr2dict(e, reaction_ids) if type(e) is str else e for e in expr]


def lineq2mat(equations, reaction_ids) -> Tuple[sparse.csr_matrix, Tuple, sparse.csr_matrix, Tuple]:
    """Translates *linear* (in)equalities to matrices
    
    Input inequalities in the form of strings is translated into matrices and vectors. The reaction
    list defines the order of variables and thus the columns of the resulting matrices, the order
    of (in)equalities will be preserved in the output matrices. As an example, take the input:
    
    equations = ['2*c - b +3*a <= 2','c - b = 0','2*b -a >=-2'], reaction_ids = ['a','b','c']
    
    This will be translated to the form A_ineq * x <= b_ineq, A_eq * x = b_eq and hence to
    
    A_ineq = sparse.csr_matrix([[3,-1,2],[1,-2,0]]), b_ineq = [2,2],
    A_eq = sparse.csr_matrix([[1,-2,0]]), b_eq = [0]
    
    Args:
        equations (list of str): 
            (List of) (in)equalities in string form equations=['r1 + 3*r2 = 0.3', '-5*r3 -r4 <= -0.5']
            
        reaction_ids (list of str): 
            List of reaction identifiers or variable names that are used to recognize variables in
            the provided (in)equalities

    Returns:
        (Tuple): 
        A_ineq, b_ineq, A_eq, b_eq. Coefficient matrices and right hand sides that represent the input
        (in)equalities as matrix-vector multiplications
    """
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
    """Translates *linear* (in)equalities to list format: [lhs,sign,rhs]
    
    Input inequalities in the form of strings are translated into a specific list format that
    facilitates the readout of left-hand-side, equality sign and right-hand-side of the inequality.
    
    equations = ['2*c - b +3*a <= 2','c - b = 0','2*b -5...], reaction_ids = ['a','b','c']
    
    This will be translated to the [[{'a':3.0,'b':-1.0,'c':2.0},'<=',2.0],[{'b':-1.0,'c':1.0},'=',0.0], ...]
    
    Args:
        equations (list of str): 
            (List of) (in)equalities in string form equations=['r1 + 3*r2 = 0.3', '-5*r3 -r4 <= -0.5']
            
        reaction_ids (list of str): 
            List of reaction identifiers or variable names that are used to recognize variables in
            the provided (in)equalities

    Returns:
        (list of lists): 
        (In)equalities presented in the form:
        [[{'a':3.0,'b':-1.0,'c':2.0},'<=',2.0], # e1
         [{'b':-1.0,'c':1.0},'=',0.0],          # e2
         ...]                                   # ...
    """
    D = []
    for equation in equations:
        if not equation:
            continue
        try:
            lhs, rhs = re.split('<=|=|>=', equation)
            eq_sign = re.search('<=|>=|=', equation)[0]
            rhs = float(rhs)
        except:
            raise Exception("Equations must contain exactly one (in)equality sign: <=,=,>=. Right hand side must be a float number.")
        D.append((linexpr2dict(lhs, reaction_ids), eq_sign, rhs))
    return D


def lineqlist2str(D):
    """Translates a *linear* (in)equality from the list format [lhs,sign,rhs] to a string
    
    E.g. input: D=[{'a':3.0,'b':-1.0,'c':2.0},'<=',2.0]] is translated to: out='3.0 a - 1.0 b + 2.0 c <= 2'
    
    Args:
        D (list): 
            (In)equality in list form, e.g.: D=[{'a':3.0,'b':-1.0,'c':2.0},'<=',2.0]]

    Returns:
        (str): 
        A list of (in)equalities in string form

    """
    if D[0]:
        return linexprdict2str(D[0]) + " " + D[1] + " " + str(D[2])
    elif D[1] and D[2]:
        return D[1] + " " + str(D[2])
    else:
        return ""


def lineqlist2mat(D, reaction_ids) -> Tuple[sparse.csr_matrix, Tuple, sparse.csr_matrix, Tuple]:
    """Translates *linear* (in)equalities presented in the list of lists format to matrices
    
    Input inequalities in the list of lists form is translated into matrices and vectors. The reaction
    list defines the order of variables and thus the columns of the resulting matrices, the order
    of (in)equalities will be preserved in the output matrices. As an example, take the input:
    
    D = [[{'a':3.0,'b':-1.0,'c':2.0},'<=',2.0],[{'b':-1.0,'c':1.0},'=',0.0], [{'a':-1,'b':2.0},'>=',-2.0]]
    
    This will be translated to the form A_ineq * x <= b_ineq, A_eq * x = b_eq and hence to
    
    A_ineq = sparse.csr_matrix([[3,-1,2],[1,-2,0]]), b_ineq = [2,2],
    A_eq = sparse.csr_matrix([[1,-2,0]]), b_eq = [0]
    
    Args:
        D (list of dict): 
            (List of) (in)equalities in the list of list form: 
            [[{'a':3.0,'b':-1.0,'c':2.0},'<=',2.0],[{'b':-1.0,'c':1.0},'=',0.0], ...]
            
        reaction_ids (list of str): 
            List of reaction identifiers or variable names that are used to recognize variables in
            the provided (in)equalities

    Returns:
        (Tuple): 
        A_ineq, b_ineq, A_eq, b_eq. Coefficient matrices and right hand sides that represent the input
        (in)equalities as matrix-vector multiplications
    """
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
    """Translates a linear expression into a vector
    
    E.g.: input: expr='2 R3 - R1', reaction_ids=['R1', 'R2', 'R3', 'R4'] translates into sparse matrix:  A = [-1 0 2 0]
    
    Args:
        expr (str): 
            (In)equality as a character string: e.g., expr='2 R3 - R1'
                        
        reaction_ids (list of str): 
            List of reaction identifiers or variable names that are used to recognize variables in the input

    Returns:
        (sparse.csr_matrix): 
        A single-row coefficient matrix that represents the input expression when multiplied with the variable vector
    """
    #
    A = sparse.lil_matrix((1, len(reaction_ids)))
    # split expression into parts and strip away special characters
    expr_parts = [re.sub(r'^(\s|-|\+|\()*|(\s|-|\+|\))*$', '', part) for part in expr.split()]
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
                raise Exception("Expression invalid. The expression contains at least two numbers in a row.")
            last_was_number = True
            continue
        raise Exception("Expression invalid. Unknown identifier " + part + ".")
    if not len(ridx) == len(set(ridx)):
        raise Exception("Reaction identifiers may only occur once in each linear expression.")
    # iterate through reaction identifiers and retrieve coefficients from linear expression
    for rid in ridx:
        coeff = re.search('(\s|^)(\s|\d|-|\+|\.)*?(?=' + re.escape(rid) + '(\s|$))', expr)[0]
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
    """Translates a linear expression into a dictionary
    
    E.g.: input: expr='2 R3 - R1', reaction_ids=['R1', 'R2', 'R3', 'R4'] translates to a dict D={'R1':-1.0, 'R3': 2.0}
    
    Args:
        expr (str): 
            (In)equalities as a character string, e.g.: expr='2 R3 - R1'
            
        reaction_ids (list of str): 
            List of reaction identifiers or variable names that are used to recognize variables in the input

    Returns:
        (dict): 
        A dictionary that contains the variable names and the variable coefficients in the linear expression
    """
    # split expression into parts and strip away special characters
    expr_parts = [re.sub(r'^(\s|-|\+|\()*|(\s|-|\+|\))*$', '', part) for part in expr.split()]
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
                raise Exception("Expression invalid. The expression contains at least two numbers in a row.")
            last_was_number = True
            continue
        raise Exception("Expression invalid. Unknown identifier " + part + ".")
    if not len(ridx) == len(set(ridx)):
        raise Exception("Reaction identifiers may only occur once in each linear expression.")
    D = {}
    # iterate through reaction identifiers and retrieve coefficients from linear expression
    for rid in ridx:
        coeff = re.search('(\s|^)(\s|\d|-|\+|\.)*?(?=' + re.escape(rid) + '(\s|$))', expr)[0]
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
    """Translates a linear expression from dict into a matrix
    
    E.g.: input: D={'R1':-1.0, 'R3': 2.0}, reaction_ids=['R1', 'R2', 'R3', 'R4'] translates into sparse matrix:  A = [-1 0 2 0]
    
    Args:
        D (dict): 
            Linear expression as a dictionary
                        
        reaction_ids (list of str): 
            List of reaction identifiers or variable names

    Returns:
        (sparse.csr_matrix): 
        A single-row coefficient matrix that represents the input expression when multiplied with the variable vector
    """
    # iterate through reaction identifiers and retrieve coefficients from linear expression
    A = sparse.lil_matrix((1, len(reaction_ids)))
    for k, v in D.items():
        A[0, reaction_ids.index(k)] = v
    return A.tocsr()


def linexprdict2str(D):
    """Translates a linear expression from dict into a caracter string
    
    E.g.: input: D={'R1':-1.0, 'R3': 2.0}, translates to the string:  '- 1.0 R1 + 2.0 R3'
    
    Args:
        D (dict): 
            Linear expression as a dictionary

    Returns:
        (str): 
        The input linear expression as a character string
    """
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
    """Get reaction identifiers that are present in string
    
    E.g.: input: D={'R1':-1.0, 'R3': 2.0}, translates to the string:  '- 1.0 R1 + 2.0 R3'
    
    Args:
        expr (str): 
            A character string
                        
        reaction_ids (list of str): 
            List of reaction identifiers or variable names

    Returns:
        (list of str): 
        A list of strings containing the reaction/variable strings present in the input string
    """
    expr_parts = [re.sub(r'^(\s|-|\+|\()*|(\s|-|\+|\|<|\=|>)*$', '', part) for part in expr.split()]
    reacIDs = []
    for part in expr_parts:
        if part in reaction_ids:
            reacIDs += [part]
            continue
        if re.match('^\d*\.{0,1}\d*$', part) is not None:
            continue
        raise Exception("Expression invalid. Unknown identifier " + part + ".")
    return reacIDs
