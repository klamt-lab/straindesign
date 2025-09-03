from efmlrs.util.data import *
from efmlrs.util.log import *


def rv_check(row, reversibilities):
    """
    Checks if involved reactions are reversible.

    :param row: row of stoichiometric matrix from panda dataframe
    :param list reversibilities: list of reaction reversibilities
    :return: bool
    """
    skip = False
    for i, rev in zip(range(0, len(reversibilities)), reversibilities):
        if rev == 1:
            if row[i] != 0:
                skip = True
                break
        else:
            continue
    return skip


def count_check(row):
    """
    Checks of metabolite carries only unique fluxes. This is indicated by having only positive and one negative
    coefficient respectively only negative and one positive coefficient in the row of the stoichiometric matrix.

    :param row: row of stoichiometric matrix from panda dataframe
    :return: bool
    """
    count_pos = 0
    count_neg = 0
    for reaction in row:
        if reaction == 0:
            continue
        elif reaction < 0:
            count_neg += 1
        else:
            count_pos += 1
        if count_pos > 1 and count_neg > 1:
            break
    if count_pos == 0 or count_neg == 0:
        return True
    if count_pos > 1 and count_neg > 1 or count_pos + count_neg < 2:
        return True
    return False


def get_index(row, index):
    """
    Gets indices and multiplication factors for reactions to be removed or kept.

    :param row:
    :param list index: list of reaction indices from panda data frame header
    :return: lists of tuples with reactions indices and multiplication factors
    """

    pos = [(i, abs(j)) for i, j in zip(index, row) if j > 0]
    neg = [(i, abs(j)) for i, j in zip(index, row) if j < 0]
    if len(pos) == 1:
        return pos, neg
    else:
        return neg, pos


def is_zero(row):
    """
    Checks if row contains only zeros.

    :param row: row of stoichiometric matrix from panda dataframe
    :return: bool
    """
    return all(i == 0 for i in row)


def remove_zero_rows(smatrix, metabolites):
    """
    Finds and removes rows in stoichiometric matrix that only contain zeros. Writes information on removed metabolites
    to compression log.

    :param smatrix: smatrix: panda dataframe that contains the stoichiometric matrix
    :param list metabolites: list of metabolite names
    :return: None
    """
    drops = []
    metadrops = []
    count = 0
    for index, row in smatrix.iterrows():
        if is_zero(row) is True:
            drops.append(index)
            metadrops.append(count)
        count += 1
    smatrix.drop(drops, axis=0, inplace=True)
    for i in reversed(metadrops):
        log_delete_meta(metabolites[i])
        del metabolites[i]


def merge_compress(smatrix, reversibilities, metabolites):
    """
    Iteratively searches, merges and removes irreversible reactions with unique fluxes from the stoichiometric matrix.
    Returns list of tuples of lists with merged reaction index and the corresponding multiplication factor.

    :param smatrix: panda dataframe that contains the stoichiometric matrix
    :param list reversibilities: list of reaction reversibilities
    :param list metabolites: list of metabolite names
    :return: list compressions: list of tuples with merged reaction indices and corresponding multiplication factors
    """

    inner_counter = 1
    compress = []
    while 1:
        remove = []
        for index, row in smatrix.iterrows():
            if rv_check(row, reversibilities) is True:
                continue
            if count_check(row) is True:
                continue
            remove, keep = get_index(row, smatrix.columns.values)
            compress.append((remove, keep))
            break
        if len(remove) == 0:
            break

        remove_index, remove_factor = remove[0]
        smatrix.loc[:, remove_index] /= remove_factor

        for column_index, factor in keep:
            smatrix.loc[:, column_index] /= factor
            smatrix.loc[:, column_index] += smatrix.loc[:, remove_index]

        smatrix.drop(remove_index, axis=1, inplace=True)
        remove_zero_rows(smatrix, metabolites)
        inner_counter += 1
    return compress


def cut_reactions(compressions, reversibilities, reactions):
    """
    Removes entries of removed reactions from reaction and reversibilities lists.
    Builds new merged reaction names. Writes merge information to compression log file.

    :param list compressions: list of tuples with merged reaction indices and corresponding multiplication factors
    :param list reversibilities: list of reaction reversibilities
    :param list reactions: list of reaction names
    :return:
        - reversibilities - reduced reversibilities
        - reactions - reduced list of reactions with merged reaction names
    """
    remove_entry = []
    for remove, keep in compressions:
        index = remove[0][0]
        remove_entry.append(index)
        remove_factor = remove[0][1]
        rm_rea_index = reactions[index]
        rea_names, factors = [reactions[names[0]] for names in keep], [factor[1] for factor in keep]
        result, new_names = log_merge_many(rm_rea_index, rea_names, remove_factor, factors)

        if result is False:
            if len(rea_names) == 1:
                reactions[index], reactions[keep[0][0]] = reactions[keep[0][0]], reactions[index]
                log_merge_many(rea_names[0], [rm_rea_index], remove_factor, factors, force=True)
            else:
                log_merge_many(rm_rea_index, rea_names, remove_factor, factors, force=True)

        if len(new_names) > 0:
            for i in range(0, len(keep)):
                index, factor = keep[i]
                reactions[index] = new_names[i]

    for i in reversed(sorted(remove_entry)):
        del reversibilities[i]
        del reactions[i]
    return reversibilities, reactions


def write_o2m_info(core_name, compressions, rea_pre_merge, rea_post_merge, outer_counter):
    """
    Writes compression information to *.info file for decompression during post-processing.

    :param str core_name: string that consists of path to and name of the input file excluding file extension
    :param list compressions: list of tuples with merged reaction indices and corresponding multiplication factors
    :param int rea_pre_merge: amount of reactions before many2one compression
    :param int rea_post_merge: amount of reactions after many2one compression
    :param int outer_counter: integer that counts how many iterative steps of all compressions have been performed
    :return: None
    """
    info_file_name = core_name + ".info"
    file = open(info_file_name, "a")
    file.write("many2one_" + str(outer_counter) + "\n")
    file.write("merged " + str(rea_pre_merge) + ":" + str(rea_post_merge) + " reactions\n")

    for remove, keep in compressions:
        for rea_index, factor in remove:
            file.write("remove R" + str(rea_index) + ":" + str(factor) + " ")
        for rea_index, factor in keep:
            file.write("keep R" + str(rea_index) + ":" + str(factor) + " ")
        file.write("\n")
    file.write("end\n")
    file.close()


def run(smatrix, reactions, reversibilities, metabolites, core_name, outer_counter):
    """
    Entry point for many2one compression. Iteratively searches for irreversible reactions with
    unique fluxes. The multiplication factors for these reactions is calculated and the reactions are merged. After merging
    reactions a new merged reaction is build and the original reactions are removed from the stoichiometric matrix.
    Metabolites that become zero through merges are removed.

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
    smatrix = convert_matrix2df(smatrix)
    rea_pre_merge = len(smatrix.columns.values)
    compressions = merge_compress(smatrix, reversibilities, metabolites)
    rea_post_merge = len(smatrix.columns.values)
    reversibilities, reactions = cut_reactions(compressions, reversibilities, reactions)
    write_o2m_info(core_name, compressions, rea_pre_merge, rea_post_merge, outer_counter)
    smatrix = convert_df2matrix(smatrix)

    return smatrix, reactions, reversibilities, metabolites
