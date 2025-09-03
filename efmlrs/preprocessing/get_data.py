import cobra
import io
from contextlib import redirect_stderr
import efmlrs.preprocessing.boundaries as boundaries
from efmlrs.util.data import *
import optlang

def read_model(input_filename):
    """
    Reads metabolic model from sbml file using the 'cobra.io.read_sbml_model' functions. Reads io string during reading
    model and catches exchange reactions added by cobrapy. These reactions are going to be removed from the model again.
    :param input_filename: sbml file with cobrapy compatible metabolic model
    :return:
        - model - cobrapy model
        - reas_added - list of reactions added by cobrapy
    """
    
    optlang.glpk_interface.Configuration()
    stream = io.StringIO()
    with redirect_stderr(stream):
        model = cobra.io.read_sbml_model(input_filename)
    console_output = stream.getvalue()

    reas_added = []
    for text in console_output.split('\n'):
        if text.startswith("Adding exchange reaction"):
            tmp = text.split()
            reas_added.append(tmp[3])

    return model, reas_added


def rm_reactions(model, rea_list):
    """
    Removes exchange reaction that were added by cobrapy.

    :param model: cobrapy model
    :param list rea_list: list of reaction names that will be removed
    :return:
        - model - altered cobrapy model
    """
    model.remove_reactions(rea_list)
    return model


def rm_metas_in_specified_compartment(comp, model):
    """
    Removes metabolites in specified compartment.

    :param str comp: comparment to ignore
    :param model: cobrapy model
    :return: model: altered cobrapy model
    """
    metas_in_compartment2remove = []
    for meta in model.metabolites:
        if meta.compartment == comp:
            metas_in_compartment2remove.append(meta)
    model.remove_metabolites(metas_in_compartment2remove)
    return model


def compartments_2_rm(model, comp_list: list):
    """
    Checks if compartment were specified to ignore and removes metabolites that belong to specified compartments.

    :param model: cobrapy model
    :param list comp_list: list of compartments that will be removed
    :return: model: altered cobrapy model
    """

    if comp_list is None or len(comp_list) == 0:
        print("Ignoring compartments: None")
        return model
    else:
        print("Ignoring compartments:", comp_list)
        for comp in comp_list:
            rm_metas_in_specified_compartment(comp, model)
        return model


def orphaned_metas_rm(model):
    """
    Searches and removes orphaned metabolites from cobrapy model. Orphaned metabolites are not involved in any reaction.

    :param model: cobrapy model
    :return: model: altered cobrapy model
    """
    metabolites = [meta.id for meta in model.metabolites]
    orphaned_metas = []
    for meta in metabolites:
        meta_id = model.metabolites.get_by_id(meta)
        involved_reas = len(meta_id.reactions)
        if involved_reas == 0:
            orphaned_metas.append(meta_id)

    orphaned_metas4print = [meta.id for meta in orphaned_metas]
    if len(orphaned_metas4print) != 0:
        print("The following metabolites are orphans (not involved in any reaction) and will be removed from the model")
        print(orphaned_metas4print)
    model.remove_metabolites(orphaned_metas)
    return model


def rm_empty_reas(model):
    reactions = [rea.id for rea in model.reactions]
    i = 0
    empty_rea = []
    for rea in reactions:
        rea = model.reactions.get_by_id(rea)
        reactants = rea.reactants
        products = rea.products
        if len(reactants) == 0 and len(products) == 0:
            i += 1
            empty_rea.append(rea)
    if len(empty_rea) != 0:
        rm_reactions(model, empty_rea)
        print(len(empty_rea), 'empty reactions removed from the model')
    else:
        print('no empty reactions')
    return model


def get_smatrix(model):
    """
    Gets stoichiometric matrix from cobrapy model using the cobrapy function
    "cobra.util.array.create_stoichiometric_matrix".

    :param model: cobrapy model
    :return: matrix: stoichiometric matrix
    """
    matrix = cobra.util.array.create_stoichiometric_matrix(model)
    return matrix


def check_bounds(model, smatrix, reactions, reversibilities, metabolites):
    """
    Calls boundaries script.

    :param model: cobrapy model
    :param smatrix: stoichiometric matrix
    :param reactions: list of reaction names
    :param reversibilities: list of reaction reversibilities
    :param metabolites: list of metabolite names
    :return:
        - smatrix - matrix stoichiometric matrix
        - reactions - list of reactions names
        - reversibilities - list of reaction reversibilities
        - metabolites - list of metabolite names
        - bound_counter - integer number of added bounds
    """
    smatrix, reactions, reversibilities, metabolites, bound_counter = boundaries.run(model, smatrix, reactions,
                                                                                     reversibilities, metabolites)
    return smatrix, reactions, reversibilities, metabolites, bound_counter


def write_bound_info(core_name, bound_counter):
    """
    Writes boundary information to info file.

    :param core_name: string that consists of path to and name of the input file excluding file extension
    :param int bound_counter: integer number of added bounds
    :return: None
    """
    info_file_name = core_name + ".info"
    file = open(info_file_name, "w")
    file.write("bounds " + str(bound_counter) + "\n")
    file.close()


def run(inputfile, ignore_compartments, boundflag):
    """
    Entry point for get_data. Takes sbml file as input. Removes exchange reactions that were added by cobrapy during
    reading. Using cobrapy the model name and properties are extracted. Removes orphaned metabolites. As specified by
    user input ignores compartments a and creates boundary reactions and metabolites. Converts stoichiometric matrix
    coefficients into fractions for precise arithmetic in later calculations and converts stoichiometric matrix from
    list of lists to sympy matrix. Creates the following files: sfile (smatrix), mfile (metabolite names),
    rfile (reaction names), rvfile (reaction reversibilities) and info file.

    :param inputfile: sbml file with cobrapy compatible metabolic model
    :param str ignore_compartments: (optional) user input as string with compartment name that will be ignored
    :param bool boundflag: (optional) user input as bool flag if boundaries from sbml file will be taken into account
    :return:
        - smatrix - sympy matrix of stoichiometric matrix
        - reactions - list of reactions names
        - reversibilities - list of reaction reversibilities
        - metabolites - list of metabolite names
        - model - cobrapy model
        - core_name - path to input file without extensions
    """

    model, reas_added = read_model(inputfile)
    model = rm_reactions(model, reas_added)
    model = compartments_2_rm(model, ignore_compartments)
    model = orphaned_metas_rm(model)
    model = rm_empty_reas(model)
    model.reactions.sort()
    model.metabolites.sort()
    smatrix = get_smatrix(model)
    reactions = [rea.id for rea in model.reactions]
    reversibilities = [rea.reversibility for rea in model.reactions]
    metabolites = [meta.id for meta in model.metabolites]
    core_name = inputfile[:-4]

    if boundflag is True:
        smatrix, reactions, reversibilities, metabolites, bound_counter = check_bounds(model, smatrix, reactions,
                                                                                       reversibilities, metabolites)
        write_bound_info(core_name, bound_counter)

        smatrix_list = list(smatrix)
        write_sfile_float(core_name, smatrix_list)
        write_initial_files_with_bounds(core_name, reactions, reversibilities, metabolites)

        smatrix_fraction = convert_float2fraction(smatrix)
        smatrix = Matrix(smatrix_fraction)

    else:
        bound_counter = 0
        write_bound_info(core_name, bound_counter)

        smatrix_list = list(smatrix)
        write_sfile_float(core_name, smatrix_list)

        write_initial_files_no_bounds(core_name, reactions, reversibilities, metabolites)

        smatrix_fraction = convert_float2fraction(smatrix)
        smatrix = Matrix(smatrix_fraction)

    return smatrix, reactions, reversibilities, metabolites, model, core_name
