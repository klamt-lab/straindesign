from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter
import sys
from efmlrs.util.log import *
from efmlrs.util.data import *
import efmlrs.preprocessing.get_data as get_data
import efmlrs.preprocessing.compressions.deadend as deadend
import efmlrs.preprocessing.compressions.many2one as many2one
import efmlrs.preprocessing.compressions.nullspace as nullspace
import efmlrs.preprocessing.compressions.echelon as echelon
import efmlrs.preprocessing.mplrs_output as mplrs_output


def main(inputsbml, ignore_compartments, boundflag):
    """
    Main script of preprocessing part of EFMlrs. Calls all other scripts. Compressions are being done iteratively as
    long as compressions can be found. Loop breaks after one loop with no changes in length of metabolites or reactions.
    Writes uncompressed and compressed output files including additional versions of the sfile with integers
    (instead of fractions) to be compatible with efmtool. Writes compressed input file for mplrs algorithm. Writes info
    file that is needed for decompressions and log file that contains all information on the applied compressions.

    :param inputsbml: path to sbml input file that contains the metabolic model
    :param ignore_compartments: list of compartments to ignore
    :param boundflag: bool flag when additional bounds should be taken into account
    :return: None
    """
    efmlrs_start_compressions()
    smatrix, reactions, reversibilities, metabolites, model, core_name = get_data.run(inputsbml, ignore_compartments,
                                                                                      boundflag)
    mplrs_output.run_uncmp(core_name)
    rev_count = reversibilities4printing(reversibilities)
    print("Uncompressed network size:", smatrix.shape[1], "reactions (", rev_count, "reversible ) and", smatrix.shape[0],
          "metabolites.")

    outer_counter = 1
    print("========================================================================")
    print("START COMPRESSIONS")

    while 1:
        start_metabolites = metabolites
        start_reactions = reactions

        print("*** Compression round:", outer_counter, "***")
        print("Start deadend compression...")
        smatrix, reactions, reversibilities, metabolites = deadend.run(smatrix, reactions, reversibilities, metabolites,
                                                                       core_name, outer_counter)
        print("Done deadend compression. Network size:", smatrix.shape[0], "metabolites and", smatrix.shape[1], "reactions (", rev_count, "reversible )")

        print("Start many2one compression...")
        smatrix, reactions, reversibilities, metabolites = many2one.run(smatrix, reactions, reversibilities,
                                                                        metabolites, core_name, outer_counter)
        print("Done many2one compression. Network size:", smatrix.shape[0], "metabolites and", smatrix.shape[1], "reactions (", rev_count, "reversible )")

        print("Start nullspace compression...")
        smatrix, reactions, reversibilities, metabolites = nullspace.run(smatrix, reactions, reversibilities,
                                                                         metabolites, core_name, outer_counter)
        print("Done nullspace compression. Network size:", smatrix.shape[0], "metabolites and", smatrix.shape[1], "reactions (", rev_count, "reversible )")

        print("Start echelon compressions...")
        smatrix, metabolites = echelon.run(smatrix, metabolites)
        print("Done echelon compression. Network size:", smatrix.shape[0], "metabolites and", smatrix.shape[1], "reactions (", rev_count, "reversible )")

        end_metabolites = metabolites
        end_reactions = reactions

        if len(start_metabolites) != len(end_metabolites) or len(start_reactions) != len(end_reactions):
            outer_counter += 1
            continue
        else:
            print("*** COMPRESSIONS DONE after:", outer_counter, "rounds ***")
            print("========================================================================")
            rev_count = reversibilities4printing(reversibilities)
            print("Compressed network size:", smatrix.shape[1], "reactions (", rev_count, "reversible ) and", smatrix.shape[0], "metabolites.")
            break

    print("Writing files")
    write_info(core_name, reversibilities, outer_counter)
    write_all(smatrix, reactions, reversibilities, metabolites, core_name + "_cmp")
    mplrs_output.run_cmp(core_name)
    write_cmp_int_matrix(core_name)
    efmlrs_finish_compressions()


def start(inputsbml, ignore_compartments, bounds):
    """
    Takes all parameters from command line, checks if parameters are okay and if everything is fine initialises log file
    and calls main function if an error occurs an exception is raised and ends the program.
    :param inputsbml: path to sbml input file that contains the metabolic model
    :param ignore_compartments: list of compartments to ignore
    :param bounds: bool flag when additional bounds should be taken into account
    :return: None
    """
    try:
        log_init(inputsbml[:-4])
        main(inputsbml, ignore_compartments, bounds)
        log_close()
    except Exception:
        print("crashed...")
        log_close()
        raise


def start_from_command_line():
    """
    Entry point for preprocessing EFMlrs. Contains all information and arguments for command line call.
    Calls start function.
    :return: None
    """
    usage = '''usage: efmlrs_pre -i <metabolic_model>.xml [--ignore_compartments <compartment name>] [--bounds]'''
    parser = ArgumentParser(prog='EFMlrs', description='Process information on metabolic model from sbml,\n'
                                                       'compress stoichiometric matrix and create all necessary files\n'
                                                       'for calculating EFMs with mplrs', epilog=usage,
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--inputsbml", help="path to input sbml file ")
    parser.add_argument('--ignore_compartments', nargs='*',
                        help="name or names of compartments that will be ignored e.g. C_e,C_b")
    parser.add_argument("--bounds", action='store_true',help="if flag --bounds is set, bounds from sbml will be taken into account")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    start(args.inputsbml, args.ignore_compartments, args.bounds)


if __name__ == "__main__":
    start_from_command_line()
