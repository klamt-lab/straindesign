from efmlrs.util.log import *
from efmlrs.util.data import *


def find_redundant_metabolites(smatrix, inner_counter):
    """
    Calculates the reduced row echelon form of the stoichiometric matrix. Finds metabolites that are redundant due to
    conservation relations.

    :param smatrix: sympy matrix that contains the stoichiometric matrix
    :param int inner_counter: integer that counts iterative steps of nullspace compression
    :return: list redundant_metas: list of metabolites to be removed
    """
    print("Start reduced row echelon form calculations", str(inner_counter), ". This may take a while")
    echelon = smatrix.T.rref(simplify=True, pivots=False)
    columns = echelon.shape[1]
    rows = echelon.shape[0]
    redundant_metas = []
    j = 0
    for i in range(0, rows):
        row = echelon.row(i)
        while j < columns and row[j] != 1:
            redundant_metas.append(j)
            j += 1
        j += 1
    return redundant_metas


def remove_redundant_metabolites(smatrix, metabolites, redundant_metas):
    """
    Removes redundant metabolites from stoichiometric matrix and from list of metabolite names. Writes compression
    information to log file. (https://academic.oup.com/bioinformatics/article/24/19/2229/246674)

    :param smatrix: sympy matrix that contains the stoichiometric matrix
    :param list metabolites: list of metabolite names
    :param list redundant_metas: list of metabolites to be removed
    :return:
        - smatrix - sympy matrix reduced stoichiometric matrix
        - metabolites - list of reduced metabolite names
    """
    for i in reversed(redundant_metas):
        log_delete_meta(metabolites[i])
        del (metabolites[i])
        smatrix.row_del(i)
    return smatrix, metabolites


def run(smatrix, metabolites):
    """
    Entry point for echelon compression. Iteratively finds inconsistencies due to conservation relations of metabolites
    in the reduced row echelon form of the stoichiometric matrix and removes them.

    :param smatrix: sympy matrix that contains the stoichiometric matrix
    :param list metabolites: list of metabolite names
    :return:
        - smatrix - sympy matrix reduced stoichiometric matrix
        - metabolites - list of reduced metabolite names
    """
    log_module()
    inner_counter = 1

    while 1:
        redundant_metas = find_redundant_metabolites(smatrix, inner_counter)
        if len(redundant_metas) == 0:
            break
        smatrix, metabolites = remove_redundant_metabolites(smatrix, metabolites, redundant_metas)
        inner_counter += 1

    return smatrix, metabolites
