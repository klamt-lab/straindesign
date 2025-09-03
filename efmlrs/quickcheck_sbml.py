from argparse import ArgumentParser
import io
from contextlib import redirect_stderr
import cobra
from cobra.flux_analysis import flux_variability_analysis
import sys

def read_model(input_filename):
    """
    Reads metabolic model from sbml file using the 'cobra.io.read_sbml_model' functions. Reads io string during reading
    model and catches exchange reactions added by cobrapy. These reactions are going to be removed from the model again.
    :param input_filename: sbml file with cobrapy compatible metabolic model
    :return:
        - model - cobrapy model
        - reas_added - list of reactions added by cobrapy
    """

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


def get_rev_amount(model):
    """
    Counts reversible reactions in model
    :param model: model from sbml
    :return: rev_cnt (float equal to number of reversible reactions)
    """
    rev_cnt = 0
    for rea in model.reactions:
        if model.reactions.get_by_id(rea.id).reversibility is True:
            rev_cnt += 1
    return rev_cnt


def model_summary(model):
    """
    Prints out important model parameters: amount of reactions, reversible reactions, metabolites and genes. Calculates
    growth rate as well using standard FBA.
    :param model: model from sbml file
    :return: None
    """
    print('------------------------------------------------')
    print(model.name)
    print('reactions:', len(model.reactions), '(including reactions added by cobrapy)')
    reversibilities = get_rev_amount(model)
    print('reversibilities: ', reversibilities)
    print('metabolites:', len(model.metabolites))
    print('genes:', len(model.genes))
    solution_fba = model.optimize()
    print('growth rate:', solution_fba.objective_value)
    print('------------------------------------------------')


def get_reactions_only_sbml(model, reas_added):
    """
    Gets reactions that are in sbml and excludes reactions added by cobrapy.
    :param model: model from sbml file
    :param reas_added: list of reactions added by cobrapy
    :return: reactions: list of reactions that are in sbml file
    """
    reactions = []
    for rea in model.reactions:
        reactions.append(rea.id)
    for rea in reas_added:
        reactions.remove(rea)
    return reactions


def get_values(rea, fva_results):
    """
    Gets important values for evaluating reactions.
    :param rea: reaction from sbml file
    :param fva_results: FVA results from cobrapy fva function
    :return:
        - min_round - FVA minimum rounded to 7th decimal
        - min_exact - FVA minimum exact result
        - max_round - FVA maximum rounded to 7th decimal
        - max_exact - FVA maximum exact result
        - diff_round - difference between max_round - min_round
        - diff_exact - difference between max_exact - min_exact
        - lb - reaction lower bound
        - ub - reaction upper bound
    """
    rea2check_fva = fva_results.loc[rea.id, :]
    min_exact = rea2check_fva.minimum
    max_exact = rea2check_fva.maximum
    min_round = round(min_exact, 10)
    min_round = min_round + 0
    max_round = round(max_exact, 10)
    max_round = max_round + 0
    diff_exact = max_exact - min_exact
    diff_round = max_round - min_round
    lb = rea.lower_bound
    ub = rea.upper_bound
    lb, ub = convert_inf(lb, ub)
    return min_round, min_exact, max_round, max_exact, diff_round, diff_exact, lb, ub


def convert_inf(lb, ub):
    """
    Converts lower and upper bounds (-inf, inf) added by cobrapy to standard float (-1000, 1000)
    :param lb: reaction lower bound
    :param ub: reaction upper bound
    :return:
        - lb - float lower bound
        - ub - float upper bound
    """
    lb_str = str(lb)
    ub_str = str(ub)
    if lb_str == '-inf':
        lb = -1000
    if ub_str == 'inf':
        ub = 1000
    return lb, ub


def check_reactions_01(reactions, fva_results, model):
    """
    First step of checking reactions incl. getting important values for evaluation
    :param reactions: list of reactions from sbml file
    :param fva_results: FVA results from cobrapy fva function
    :param model: model from sbml file
    :return: check_reas - list of reactions to check
    """
    check_reas = {}
    for rea in reactions:
        rea = model.reactions.get_by_id(rea)
        min_round, min_exact, max_round, max_exact, diff_round, diff_exact, lb, ub = get_values(rea, fva_results)
        check_reas = check_reactions_02(lb, ub, min_round, max_round, rea, diff_round, check_reas)
    return check_reas


def check_reactions_02(lb, ub, min, max, rea, diff, check_reas):
    """
    Checks if reaction bounds for reversible/irreversible reactions are correct and calls functions for further checks.
    :param lb: reaction lower bound
    :param ub: reaction upper bound
    :param min: FVA minimum rounded to 10th decimal
    :param max: FVA maximum rounded to 10th decimal
    :param rea: current reaction
    :param diff: difference between max_round - min_round
    :param check_reas: check_reas - list of reactions to check
    :return: check_reas - list of reactions to check
    """
    values = []
    if rea.reversibility:
        if lb < 0 and ub > 0:
            check_reas = check_rev(lb, ub, min, max, rea, diff, check_reas)
        else:
            values.extend([lb, ub, min, max, rea, diff])
            check_reas[rea.id] = values
    else:
        if lb >= 0 and ub >= 0:
            check_reas = check_irrev(lb, ub, min, max, rea, diff, check_reas)
        else:
            values.extend([lb, ub, min, max, rea, diff])
            check_reas[rea.id] = values
    return check_reas


def check_irrev(lb, ub, min, max, rea, diff, check_reas):
    """
    Checks if fluxes of current irreversible reaction are truly irreversible and if reaction direction is correct.
    :param lb: reaction lower bound
    :param ub: reaction upper bound
    :param min: FVA minimum rounded to 10th decimal
    :param max: FVA maximum rounded to 10th decimal
    :param rea: current reaction
    :param diff: difference between max_round - min_round
    :param check_reas: check_reas - list of reactions to check
    :return: check_reas - list of reactions to check
    """
    values = []
    diff_min_lb = round(min - lb, 6)

    if lb <= min and ub >= max:
        if diff > 0:
            pass
        elif min == max:
            values.extend([lb, ub, min, max, rea, diff])
            check_reas[rea.id] = values
        else:
            values.extend([lb, ub, min, max, rea, diff])
            check_reas[rea.id] = values
    else:
        if diff_min_lb >= 0:
            pass
        else:
            values.extend([lb, ub, min, max, rea, diff])
            check_reas[rea.id] = values
    return check_reas


def check_rev(lb, ub, min, max, rea, diff, check_reas):
    """
    Checks if fluxes of current reversible reaction are truly irreversible and if reaction direction is correct.
    :param lb: reaction lower bound
    :param ub: reaction upper bound
    :param min: FVA minimum rounded to 10th decimal
    :param max: FVA maximum rounded to 10th decimal
    :param rea: current reaction
    :param diff: difference between max_round - min_round
    :param check_reas: check_reas - list of reactions to check
    :return: check_reas - list of reactions to check
    """
    values = []
    if lb <= min < 0 and ub >= max and max > 0:
        if diff >= 0:
            pass
        else:
            values.extend([lb, ub, min, max, rea, diff])
            check_reas[rea.id] = values
    else:
        if min == 0 and max == 0:
            values.extend([lb, ub, min, max, rea, diff])
            check_reas[rea.id] = values
        elif min <= 0 and max <= 0:
            values.extend([lb, ub, min, max, rea, diff])
            check_reas[rea.id] = values
        else:
            values.extend([lb, ub, min, max, rea, diff])
            check_reas[rea.id] = values
    return check_reas


def rea_write(outputpath, reactions, fva_results, model):
    """
    Writes reaction information to file including lower bound (lb), upper bound (ub), FVA minimum and maximum and
    reaction reversibility
    :param outputpath: same as input path for sbml model with '_reactions.txt' as extention
    :param reactions: list of reactions that are originally in the sbml file
    :param fva_results: results of cobrapy FVA
    :param model: model from sbml file
    :return: None
    """
    file = outputpath[:-4] + '_reactions.txt'
    print('Writing results to file:',file)
    f = open(file, 'w')
    for rea in reactions:
        rea = model.reactions.get_by_id(rea)
        min_round, min_exact, max_round, max_exact, diff_round, diff_exact, lb, ub = get_values(rea, fva_results)
        f.write(str(rea) + '\n')
        f.write('lb: ' + str(lb) + ' ub: ' + str(ub) + '\n')
        f.write('min: ' + str(min_exact) + ' max: ' + str(max_exact) + ' diff: ' + str(diff_exact) + '\n')
        f.write('reversibility: ' + str(rea.reversibility) + '\n')
        f.write('------------------------------------------------\n')
    f.close()


def main(inputsbml, fraction_optimum):
    """
    Main function that parses sbml file, performs FVA, check reaction bounds and flux directions and prints out list of
    reactions that may need correction
    :param inputsbml: sbml file of metabolic model
    :param fraction_optimum: float that defines the fraction of optimum used for cobrapy FVA
    :return: None
    """
    cobra.io.read_sbml_model(inputsbml)
    model, reas_added = read_model(inputsbml)


    print('------------------------------------------------')
    print('COBRAPY added',len(reas_added),'exchange reaction')
    model_summary(model)
    print('fraction of optimum:', fraction_optimum)
    print('------------------------------------------------')

    print('Calculating FVA...')
    fva_results = flux_variability_analysis(model, model.reactions, fraction_of_optimum=fraction_optimum)
    reactions = get_reactions_only_sbml(model, reas_added)
    print('Checking reactions...')
    check_reas = check_reactions_01(reactions, fva_results, model)
    rea_write(inputsbml, reactions, fva_results, model)

    if bool(check_reas):
        print('------------------------------------------------')
        print('Please check the following reactions:')
        print('NOTE: FVA results are rounded to 10th decimal, exact values can be found in *_reactions.txt')
        print('------------------------------------------------')
        for key, vals in check_reas.items():
            tmp = vals
            print(tmp[4])
            print('lower bound:', tmp[0], 'upper bound:', tmp[1])
            print('FVA minimum:', tmp[2], 'FVA maximum:', tmp[3], 'difference:', tmp[5])
            print('reversibility:', tmp[4].reversibility)
            print('------------------------------------------------')
        print('FVA results are rounded to 10th decimal, exact values can be found in *_reactions.txt')
    else:
        print('------------------------------------------------')
        print('All reactions seem to be fine')
        print('Exact FVA results can be found in *_reactions.txt')


def start_from_command_line():
    DEFFILE = '../tests/example_models/ecoli5010.xml'
    DEFFRO = 0.1

    parser = ArgumentParser(description='Quickcheck for sbml file before compressions with EFMlrs and later EFM/V calculations with efmtool or mplrs')
    parser.add_argument('-s', '--sbmlinput', help='input is name of the sbml model')
    parser.add_argument('-f', '--fraction_optimum', help='float (optional, default=0.1) - fraction of optimum parameter for cobrapy function: flux_variability_analysis(): Must be <= 1.0. Requires that the objective value is at least the fraction times maximum objective value. A value of 0.85 for instance means that the objective has to be at least at 85 percent of its maximum https://cobrapy.readthedocs.io/en/latest/_modules/cobra/flux_analysis/variability.html', default=DEFFRO)

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    try:
        main(args.sbmlinput, args.fraction_optimum)
    except Exception:
        print('crashed...')
        raise

if __name__ == "__main__":
    start_from_command_line()
