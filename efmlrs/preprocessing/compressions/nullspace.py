from efmlrs.util.data import *
from efmlrs.util.log import *


def enumerate_rows(kernel):
    """
    Builds index for rows in kernel. Rows in kernel correspond to the reactions of the stoichiometric matrix, as does
    the index built.

    :param list kernel: kernel of stoichiometric matrix
    :return: list of tuples with list of reaction values from kernel matrix and corresponding reaction index
    """
    l = len(kernel[0])
    rows = []
    rows_index = []
    tuples = []
    for j in range(0, l):
        row = [item[j] for item in kernel]
        rows.append(row)
        rows_index.append(j)
        tuples.append(row)
    return list(zip(tuples, rows_index))


def check_multiplier(row1, row2):
    """
    Checks if rows are multiples of each other and calculates multiplication factor.

    :param list row1: first row
    :param list row2: second row
    :return: bool or multiplication factor as fraction
    """
    factor = 0
    for i, j in zip(row1, row2):
        if i == 0 and j == 0:
            continue
        if i == 0 or j == 0:
            return False
        if factor == 0:
            factor = i / j
        else:
            if i / j != factor:
                return False
    return factor


def search_multiples(kernel):
    """
    Builds index for reactions in kernel matrix and searches for linear depended reactions and their multiplication
    factor.

    :param list kernel: kernel of stoichiometric matrix
    :return: list of tuples with linear depend reactions indices and their multiplication factor
    """
    K = enumerate_rows(kernel)
    results = []
    i = 0
    for row1, index1 in K:
        j = 0
        for row2, index2 in K:
            if j <= i:
                j += 1
                continue
            factor = check_multiplier(row1, row2)
            if factor != False:
                results.append((index1, index2, factor))
            j += 1
        i += 1
    return results


def test_results(column_index, index_multiples, smatrix, reversibilities):
    """
    Checks if current reaction is marked for merge, if yes merges reactions and builds new merged reaction. Checks
    reversibilities of reactions to merge. Returns information if new reaction was build, new merged reaction or empty
    matrix, information on new merged reaction reversibility, index of reaction that can be removed from stoichiometric
    matrix after merge, multiplication factor, reversibility_flag (bool variable from or link of merged reaction
    reversibilities)

    :param int column_index: reaction index of current reaction
    :param list index_multiples: list of tuples with indices and multiplication factor of linear depend reactions
    :param smatrix: sympy matrix that contains the stoichiometric matrix
    :param list reversibilities: list of reaction reversibilities
    :return:
        - bool - True if no reactions were merged, False if reactions were merged
        - Matrix()/merged_rea - empty matrix or sympy matrix containing new reaction after merge
        - new_rev_info - reversibility of new merged reaction
        - index2  - int with index of second reaction
        - factor - multiplication factor (type: sympy.core.number)
        - reversibility_flag - bool from or link of reversibilities of involved reactions
    """

    for index1, index2, factor in index_multiples:
        reversibility_flag = reversibilities[index1] or reversibilities[index2]
        if index1 == column_index:
            rea1 = smatrix.col(index1)
            rea2 = smatrix.col(index2)
            tmp = rea2 / factor
            merged_rea = rea1 + tmp
            if reversibilities[index1] and reversibilities[index2]:
                new_rev_info = True
            else:
                new_rev_info = False
            return False, merged_rea, new_rev_info, index2, factor, reversibility_flag

        if index2 == column_index:
            return False, Matrix(), -1, -1, factor, reversibility_flag

    reversibility_flag = reversibilities[column_index]
    return True, Matrix(), reversibilities[column_index], -1, -1, reversibility_flag


def merge_multiples(smatrix, reversibilities, reactions, index_multiples):
    """
    Merges reactions, builds new merged reaction, removes old reactions and builds new stoichiometric matrix, new
    reversibilities and new reaction name list. Writes merge information to log file.

    :param smatrix: sympy matrix that contains the stoichiometric matrix
    :param list reversibilities: list of reaction reversibilities
    :param list reactions: list of reaction names
    :param list index_multiples: list of tuples with reaction indices
    :return:
        - smatrix_reduced - reduced stoichiometric matrix
        - reversibilities_reduced - reduced list of reaction's reversibilities
        - reactions_reduced - reduced list of merged reaction names
        - merge_info - list containing reaction indices_names and their multiplication factor
    """
    smatrix_reduced = Matrix()
    reversibilities_reduced = []
    reactions_reduced = []
    merge_info = []
    j = 0

    for column_index in range(0, smatrix.shape[1]):
        use_ori, merged_rea, new_rev_info, dropped_rea, factor, reversibility_flag = test_results(column_index,
                                                                                                  index_multiples,
                                                                                                  smatrix,
                                                                                                  reversibilities)
        if use_ori is True:
            smatrix_reduced = smatrix_reduced.col_insert(j, smatrix.col(column_index))
            reversibilities_reduced.append(new_rev_info)
            reactions_reduced.append(reactions[column_index])
            merge_info.append("R" + str(column_index))
            j += 1
        else:
            if reversibility_flag is False and factor < 0:
                log_delete(reactions[column_index], reactions[dropped_rea])
            else:
                if len(merged_rea) != 0:
                    smatrix_reduced = smatrix_reduced.col_insert(j, merged_rea)
                    reversibilities_reduced.append(new_rev_info)
                    result, newnames = log_merge(reactions[dropped_rea], reactions[column_index], factor)
                    if result is False:
                        if log_merge(reactions[column_index], reactions[dropped_rea], factor) is False:
                            log_merge(reactions[dropped_rea], reactions[column_index], factor, True)
                        else:
                            reactions[dropped_rea], reactions[column_index] = reactions[column_index], reactions[
                                dropped_rea]
                    if len(newnames) > 0:
                        reactions[column_index] = newnames[0]

                    reactions_reduced.append(reactions[column_index])
                    merge_info.append("R" + str(column_index) + "," + str() + str(dropped_rea) + ":" + str(factor))
                    j += 1

            for index in reversed(range(0, len(index_multiples))):
                i1, i2, f = index_multiples[index]
                if i1 == column_index and i2 == dropped_rea:
                    continue
                if i1 == column_index or i2 == column_index or i1 == dropped_rea or i2 == dropped_rea:
                    del index_multiples[index]
    return smatrix_reduced, reversibilities_reduced, reactions_reduced, merge_info


def get_index_zero_rows(smatrix):
    """
    Checks for rows that only contain zero values and return indices of these rows.

    :param smatrix: sympy matrix that contains the stoichiometric matrix
    :return: list of indices of zero rows
    """
    index_zero_rows = []
    for index in range(0, smatrix.shape[0]):
        row = [metabolite for metabolite in smatrix.row(index) if metabolite != 0]
        if bool(row) is False:
            index_zero_rows.append(index)
    return index_zero_rows


def iterate(smatrix, metabolites, reactions, reversibilities, nullspace_infos, inner_counter):
    """
    Main function of nullspace compression. Calculates kernel matrix, searches for linear depended reactions,
    merges linear dependend reactions, removes original reactions and metabolites that contain only zeros after merging
    reactions, builds new stoichiometric matrix, new list of reaction's reversibilities, new list of reaction names
    and new list of metabolite names. (https://academic.oup.com/bioinformatics/article/24/19/2229/246674)

    :param smatrix: sympy matrix that contains the stoichiometric matrix
    :param ist metabolites: list of metabolite names
    :param list reactions: list of reaction names
    :param list reversibilities: list of reaction reversibilities
    :param list nullspace_infos: list containing reaction names and their multiplication factor
    :param int inner_counter: integer that counts iterative steps of nullspace compression
    :return:
        - new_smatrix - reduced stoichiometric matrix
        - metabolites - reduced list of metabolite names
        - new_reactions - reduced list of merged reaction names
        - new_reversibilities - reduced list of reaction's reversibilities
        - nullspace_infos - list containing reaction indices_names and their multiplication factor
        - bool: indicating if nullspace compression loop is done (True) or not (False)
    """
    print("Start null space compressions round", str(inner_counter), ". This may take a while")
    K = smatrix.nullspace()
    if len(K) == 0:
        print("KERNELMATRIX is EMPTY")
        return smatrix, metabolites, reactions, reversibilities, nullspace_infos, True

    index_multiples = search_multiples(K)
    if len(index_multiples) == 0:
        return smatrix, metabolites, reactions, reversibilities, nullspace_infos, True

    new_smatrix, new_reversibilities, new_reactions, merge_info = merge_multiples(smatrix, reversibilities, reactions,
                                                                                  index_multiples)
    index_zero_rows = get_index_zero_rows(new_smatrix)

    for i in reversed(index_zero_rows):
        log_delete_meta(metabolites[i])
        del (metabolites[i])
        new_smatrix.row_del(i)

    nullspace_infos.append(merge_info)
    return new_smatrix, metabolites, new_reactions, new_reversibilities, nullspace_infos, False


def write_info(core_name, nullspace_infos, outer_counter):
    """
    Writes compression information to *.info file for decompression during post-processing.

    :param str core_name: string that consists of path to and name of the input file excluding file extension
    :param list nullspace_infos: list containing reaction names and their multiplication factor
    :param int outer_counter: int that counts how many iterative steps with all compressions have been performed
    :return: None
    """
    info_file_name = core_name + ".info"
    file = open(info_file_name, "a")
    file.write("nullspace_" + str(outer_counter) + "\n")
    reversed_info = nullspace_infos[::-1]

    for info in reversed_info:
        d = len(info)
        for i in range(0, d):
            file.write(str(info[i]) + " ")
        file.write("\n")
    file.write("end\n")
    file.close()


def run(smatrix, reactions, reversibilities, metabolites, core_name, outer_counter):
    """
    Entry point for nullspace compression. Iteratively finds and merges linear dependend reactions in the kernel matrix.
    Linear depend reactions are multiples of each others in the kernel matrix. When a linear dependencies between two
    reactions is found, their multiplication factor is calculated and these reactions get merged. Thereby a new
    reaction is build and replaces the old reactions. Metabolites that become zero through merges are removed.

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
    nullspace_infos = []
    inner_counter = 1

    while 1:
        smatrix, metabolites, reactions, reversibilities, infos, last = iterate(smatrix, metabolites, reactions,
                                                                                reversibilities, nullspace_infos,
                                                                                inner_counter)
        if last is False:
            inner_counter += 1
        else:
            break

    write_info(core_name, nullspace_infos, outer_counter)
    return smatrix, reactions, reversibilities, metabolites
