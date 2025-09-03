from efmlrs.util.data import *


def search_bounds(model):
    """
    Searches cobrapy model for additional reaction bounds.

    :param model: cobrapy model
    :return: dictionary containing information on reactions with additional bounds (reaction id, lower and upper bound)
        or returns None if no additional reaction bounds were found in the model.
    """
    bounds = {}
    for r in model.reactions:
        tmp = []
        if r.reversibility is True:
            if r.lower_bound == -1000.0 and r.upper_bound == 1000.0:
                continue
            elif str(r.lower_bound) == "-inf" and str(r.upper_bound) == "inf":
                continue
            else:
                tmp.append(r.lower_bound)
                tmp.append(r.upper_bound)
                bounds[r.id] = tmp
        else:
            if r.lower_bound == 0 and r.upper_bound == 1000.0:
                continue
            elif r.lower_bound == 0 and str(r.upper_bound) == "inf":
                continue
            else:
                tmp.append(r.lower_bound)
                tmp.append(r.upper_bound)
                bounds[r.id] = tmp

    if bool(bounds) is False:
        return False
    else:
        return bounds


def add_boundary_names(bounds_info):
    """
    Extends reaction name for reactions with additional bounds by adding "min" or "max" to lower respectively upper
    bounds.

    :param bounds_info: dictionary containing information on reaction and associated bounds
    :return: dictionary containing information on reaction bounds with extended reaction name
    """
    bounds = {}
    for rea, bound in bounds_info.items():
        if bound[0] != 0 and bound[0] != -1000:
            rea_min = rea + "_min"
            bounds[rea_min] = bound[0]
        if bound[1] != 1000:
            rea_max = rea + "_max"
            bounds[rea_max] = bound[1]
    return bounds


def format_bounds(bounds_info):
    """
    Changes bound information for additional reaction bounds stored in bounds_info from str (cobrapy format) to int.

    :param bounds_info: dictionary containing information on reaction and associated bounds
    :return: dictionary containing information on reaction bounds with extended reaction name and int as bounds
    """
    for rea, bounds in bounds_info.items():
        if str(bounds[0]) == "-inf":
            del bounds[0]
            bounds.insert(0, -1000)
        elif str(bounds[1]) == "inf":
            del bounds[1]
            bounds.insert(1, 1000)
    bounds_info = add_boundary_names(bounds_info)
    return bounds_info


def extend_smatrix_1(matrix, bounds):
    """
    Extends columns of stoichiometric matrix by number of bounds plus 1 with arrays containing only zeros.

    :param matrix: stoichiometric matrix
    :param bounds: dictionary with bound information
    :return: extended stoichiometric matrix
    """
    slack_reas_array = [0] * len(matrix)
    t_matrix = matrix.transpose()
    for i in range(len(bounds) + 1):
        t_matrix = np.vstack([t_matrix, slack_reas_array])
    matrix = t_matrix.transpose()
    return matrix


def get_bounds_index(reas, bounds):
    """
    Gets index for addtional bounds according to list of reaction names.

    :param list reas: list of reaction names
    :param bounds: dictionary with bound information
    :return: list of indicis of reaction bounds
    """
    index_bounds = []
    for rea, bound in bounds.items():
        i = 0
        for reaction in reas:
            name = rea[:-4]
            if reaction == name:
                index_bounds.append(i)
            i += 1
    return index_bounds


def build_slack_metas_01(bounds, reactions):
    """
    Builds first part of slack metabolites. Length equals length of original stoichiometric matrix. Adds 1 or -1 as
    coefficient at index of slack metabolite for upper respectively lower bound.

    :param bounds: dictionary with bound information
    :param reactions: list of reaction names
    :return: list of lists with new build slack reactions
    """
    index_bounds = get_bounds_index(reactions, bounds)
    slack_metas_1 = []
    j = 0
    for rea, bounds in bounds.items():
        tmp = []
        for i in range(len(reactions)):
            if i == index_bounds[j]:
                if rea[-3:] == "max":
                    tmp.append(1)
                else:
                    tmp.append(-1)
            else:
                tmp.append(0)
        j += 1
        slack_metas_1.append(tmp)
    return slack_metas_1


def build_slack_metas_02(bounds):
    """
    Builds entries for entries for slack reactions length equals number of additional bounds.

    :param bounds: dictionary with bound information
    :return: list slack_reas_2 of lists with entries for slack reactions
    """
    slack_metas_2 = []
    j = 0
    for rea, bound in bounds.items():
        tmp = []
        for i in range(len(bounds)):
            if i == j:
                tmp.append(1)
            else:
                tmp.append(0)
        j += 1
        slack_metas_2.append(tmp)
    return slack_metas_2


def build_slack_metas_03(bounds):
    """
    Builds last part of slack metabolite (= lambda column) with actual reaction bound as coefficient.

    :param bounds: dictionary with bound information
    :return: list slack_metas_3 of list with actual reaction bound as entry
    """
    slack_metas_3 = []
    for rea, bound in bounds.items():
        tmp = []
        if rea[-3:] == "min":
            tmp.append(bound)
        else:
            new_bound = bound * -1
            tmp.append(new_bound)
        slack_metas_3.append(tmp)
    return slack_metas_3


def building_slack_metabolites(bounds, reactions):
    """
    Builds slack metabolites. Consists of three parts. First part builds list with length of original matrix and
    coefficients according to lower or upper bound. Second part build entries for slack reactions length equals number
    of additional bounds. Third part builds entry for lambda reaction, length always 1, entry is the actual reaction
    bound.

    :param bounds: dictionary with bound information
    :param reactions: list of reaction names
    :return: list slack_metas of lists with new build slack metabolites
    """
    slack_metas_1 = build_slack_metas_01(bounds, reactions)
    slack_metas_2 = build_slack_metas_02(bounds)
    slack_metas_3 = build_slack_metas_03(bounds)
    slack_metas = []
    for i in range(len(bounds)):
        tmp = slack_metas_1[i] + slack_metas_2[i] + slack_metas_3[i]
        slack_metas.append(tmp)
    return slack_metas


def extend_smatrix_2(slack_metas, smatrix_extended):
    """
    Extends stoichiometric matrix by new build slack metabolites.

    :param list slack_metas: list of lists with new build slack metabolites
    :param smatrix_extended_1: stoichiometric matrix that has been extended by columns containing only zero
    :return: new build smatrix with slack reactions and slack metabolites
    """
    for i in range(len(slack_metas)):
        smatrix_extended = np.vstack([smatrix_extended, slack_metas[i]])
    return smatrix_extended


def build_new_smatrix(smatrix, bounds, reactions):
    """
    Extends stoichiometric matrix according to additional bounds with slack reactions and slack metabolites. Builds
    and adds slack reactions and metabolites to stoichiometric matrix. Converts matrix coefficients to fractions.

    :param smatrix: stoichiometric matrix
    :param bounds: dictionary with bound information
    :param reactions: list of reaction names
    :return: extended stoichiometric matrix
    """
    extended_smatrix_1 = extend_smatrix_1(smatrix, bounds)
    slack_metas = building_slack_metabolites(bounds, reactions)
    extended_smatrix_2 = extend_smatrix_2(slack_metas, extended_smatrix_1)
    return extended_smatrix_2


def build_new_metas(bounds, meta_ori):
    """
    Builds new list of metabolite names including slack metabolites.

    :param bounds: dictionary with bound information
    :param meta_ori: list of metabolite names
    :return: new_metas list of metabolite names including added slack metabolites
    """
    slack_metas = [("MS_" + rea) for rea, bound in bounds.items()]
    new_metas = meta_ori + slack_metas
    return new_metas


def build_new_reas(bounds, reas_ori):
    """
    Builds new list of reaction names including slack metabolites.

    :param bounds: dictionary with bound information
    :param reas_ori: list of reactions names
    :return: new_reas list reaction names including added slack reactions
    """
    slack_reas = [("RS_" + rea) for rea, bound in bounds.items()]
    rea_lambda = ["RS_lambda"]
    new_reas = reas_ori + slack_reas + rea_lambda
    return new_reas


def build_new_reversibilities(bounds, revs_ori):
    """
    Builds new list of reaction reversibilities. All added reaction are irreversible.

    :param bounds: dictionary with bound information
    :param revs_ori: list of reaction reversibilities
    :return: new_revs list of reaction reversibilities including added slack reactions
    """
    slack_revs = [(False) for i in range(len(bounds) + 1)]
    new_revs = revs_ori + slack_revs
    return new_revs


def run(model, smatrix, reactions, reversibilities, metabolites):
    """
    Entry point for boundaries. Searched for additional reaction bounds. Adds slack reaction and metabolites to the
    stoichiometric matrix, the lists of reaction names and metabolie names and the list of reaction reversibilities.

    :param model: cobrapy model
    :param smatrix: stoichiometric matrix
    :param reactions: list of reaction names
    :param reversibilities: list of reaction reversibilities
    :param metabolites: list of metabolite names
    :return:
        - smatrix - sympy matrix reduced stoichiometric matrix
        - reactions - list of reactions names
        - reversibilities - list of reaction reversibilities
        - metabolites - list of metabolite names
        - bound_counter - integer number of added bounds
    """
    print("Checking for additional bounds")
    bound_info = search_bounds(model)
    if bool(bound_info) is True:
        bounds = format_bounds(bound_info)
        bound_counter = len(bounds)
        print("Found", bound_counter, "additional bounds:")
        bounds_print(bounds)

        new_smatrix = build_new_smatrix(smatrix, bounds, reactions)
        new_metabolites = build_new_metas(bounds, metabolites)
        new_reactions = build_new_reas(bounds, reactions)
        new_reversibilities = build_new_reversibilities(bounds, reversibilities)
        return new_smatrix, new_reactions, new_reversibilities, new_metabolites, bound_counter

    else:
        print("ERROR: ADDITIONAL BOUNDS SET IN PROGRAM CALL BUT NO ADDITIONAL BOUNDS FOUND IN SBML FILE")
        print("EXITING PROGRAM")
        exit(1)
