from efmlrs.util.log import *


def check_row(row):
    """
    Checks if rows qualifies as deadend metabolite. If row consists of only zeros and either positive
    or negative entries, it is a deadend metabolite and functions returns true.

    :param row: row of stoichiometric matrix
    :return: bool
    """
    state = 0
    for val in row:
        if state == 0:
            state = val
            continue
        if state < 0:
            if val <= 0:
                continue
            else:
                return False
        if state > 0:
            if val >= 0:
                continue
            else:
                return False
    return True


def check_reactions(smatrix, i, reversibilities):
    """
    Checks if corresponding reactions to a deadend metabolite are irreversible.
    Returns list of corresponding irreversible reactions to be removed.

    :param smatrix: sympy matrix that contains the stoichiometric matrix
    :param int i: index of deadend metabolite
    :param list reversibilities: list of reaction reversibilities
    :return: list of reaction names to be removed from stoichiometric matrix
    """
    rm_reactions = []
    index = 0
    for val in smatrix.row(i):
        if val != 0:
            if reversibilities[index] == 0:
                rm_reactions.append(index)
            else:
                return []
        index += 1
    return rm_reactions


def find_deadends(smatrix, reversibilities):
    """
    Checks all rows of stoichiometric matrix if they are deadend metabolites.
    Checks if corresponding reactions are irreversible.

    :param smatrix: sympy matrix that contains the stoichiometric matrix
    :param list reversibilities: list of reaction reversibilities
    :return:
        - remove_reactions - list of reaction names that will be removed
        - remove_metabolites - list of metabolite names that will be removed
    """
    remove_reactions = []
    remove_metabolites = []

    for i in range(0, smatrix.shape[0]):
        tmp = check_row(smatrix.row(i))
        if tmp == False:
            continue
        else:
            rm_reactions = check_reactions(smatrix, i, reversibilities)
            if len(rm_reactions) != 0:
                remove_metabolites.append(i)
                for item in rm_reactions:
                    remove_reactions.append(item)

    remove_reactions = sorted(set(remove_reactions))
    remove_metabolites = sorted(set(remove_metabolites))
    return remove_reactions, remove_metabolites


def write_deadend_info(core_name, outer_counter, removedReactions):
    """
    Writes compression information to *.info file for decompression during post-processing.

    :param str core_name: string that consists of path to and name of the input file excluding file extension
    :param int outer_counter: int that counts how many iterative steps with all compressions have been performed
    :param list removedReactions: list of removed reaction indices
    :return: None
    """
    info_file_name = core_name + ".info"
    file = open(info_file_name, "a")
    file.write("deadend_" + str(outer_counter) + "\n")
    for reactions in removedReactions:
        file.write("remove")
        for rea in reactions:
            file.write(" " + str(rea))
        file.write("\n")
    file.write("end\n")
    file.close()


def run(smatrix, reactions, reversibilities, metabolites, core_name, outer_counter):
    """
    Entry point for deadend compression. Finds and removes deadend metabolites and corresponding reaction that only
    contain zeros after a deadend metabolite is removed. This is done iteratively as long as deadend metabolites or
    corresponding reactions are found in the stoichiometric matrix. Writes information on removed reactions and
    metabolites to compression log.

    :param smatrix: sympy matrix that contains the stoichiometric matrix
    :param list reactions: list of reaction names
    :param list reversibilities: list of reaction reversibilities
    :param list metabolites: list of metabolite names
    :param str core_name: string that consists of path to and name of the input file excluding file extension
    :param int outer_counter: int that counts how many iterative steps with all compressions have been performed
    :return:
        - smatrix - sympy matrix reduced stoichiometric matrix
        - reactions - list of reduced reactions names
        - reversibilities - list of reduced reaction reversibilities
        - metabolites - list of reduced metabolite names
    """

    log_module()
    inner_counter = 1
    removed_reactions = []
    while 1:
        remove_reactions, remove_metabolites = find_deadends(smatrix, reversibilities)
        if len(remove_reactions) == 0 and len(remove_metabolites) == 0:
            break

        for index in reversed(remove_reactions):
            log_delete_rea(reactions[index])
            smatrix.col_del(index)
            del reactions[index]
            del reversibilities[index]
        removed_reactions.append(remove_reactions)

        for index in reversed(remove_metabolites):
            log_delete_meta(metabolites[index])
            smatrix.row_del(index)
            del metabolites[index]

        inner_counter += 1

    write_deadend_info(core_name, outer_counter, removed_reactions)

    return smatrix, reactions, reversibilities, metabolites
