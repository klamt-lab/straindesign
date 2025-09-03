from functools import reduce
from math import gcd
from sympy import *


def get_denominators(smatrix):
    """
    Gets greatest denominator of each row in stoichiometric matrix
    :param smatrix: list of lists of stoichiometric matrix
    :return: list of greatest denominators for each row
    """
    row_denominators = [list(set([val.denominator for val in row])) for row in smatrix]
    greatest_denominators = [reduce(lambda x, y: x * y, sublist) for sublist in row_denominators]
    return greatest_denominators


def get_int_smatrix(smatrix, greatest_denominators):
    """
    Converts stoichiometric matrix from fractions to integer by multiplying each row with greatest denominator.
    :param smatrix: list of lists of stoichiometric matrix
    :param greatest_denominators: list of greatest denominators for each row
    :return: list of lists of converted stoichiometric matrix
    """
    int_smatrix = []
    i = 0
    for row in smatrix:
        new_row = []
        for val in row:
            new_val = val * greatest_denominators[i]
            new_row.append(int(new_val))
        int_smatrix.append(new_row)
        i += 1
    return int_smatrix


def get_min_smatrix(int_smatrix):
    """
    Gets greatest divisor for each row and divides each row through greatest divisor. Converts matrix from list of
    lists to sympy matrix.
    :param int_smatrix: list of lists of stoichiometric matrix with integers as coefficients
    :return: sympy matrix of stoichiometric matrix with smallest coefficients
    """
    min_smatrix = []
    for row in int_smatrix:
        greates_divisor = reduce(gcd, row)
        if greates_divisor > 1:
            new_row = [int(val / greates_divisor) for val in row]
            min_smatrix.append(new_row)
        else:
            min_smatrix.append(row)
    min_sympy_smatrix = Matrix(min_smatrix)
    return min_sympy_smatrix


def run(smatrix):
    """
    Converts stoichiometric matrix from fractions to integer by multiplying each row with greatest denominator. For
    each row in stoichiometric matrix reduces coefficients to smallest number by dividing each row through greatest
    divisor. Converts matrix from list of lists to sympy matrix.
    :param smatrix: list of lists of stoichiometric matrix
    :return: sympy matrix of stoichiometric matrix with smallest coefficients
    """
    greatest_denominators = get_denominators(smatrix)
    int_smatrix = get_int_smatrix(smatrix, greatest_denominators)
    min_smatrix = get_min_smatrix(int_smatrix)
    return min_smatrix
