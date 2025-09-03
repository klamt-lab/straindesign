from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter
import sys
import efmlrs.postprocessing.get_data as get_data
import efmlrs.postprocessing.decompressing as decompressing
from efmlrs.util.data import *


def main(inputfile, outputfile, infofile, efmtool):
    """
    Main script of prostprocessing part of EFMlrs. Calls all other scripts. First output files are read and compressed
    efms are stored in a list. Then the decompressing process is started iteratively in reversed order compared to the
    formerly applied compressions and each compressed efm is being decompressed one after another. The deompressed efms
    are written to the user specified output file.
    :param inputfile: output containing compressed efms either form mplrs or efmtool
    :param outputfile: file in which uncompressed efms are written to
    :param infofile: file automatically created during compressions, containing all information for decompressions
    :param efmtool: paramter that indicates whether decompressions for mplrs or efmtool is being executed
    :return: None
    """
    efmlrs_start_decompressions()
    if efmtool == True:
        print("Decompressing EFMs from EFMTOOL")
        compressed_efms = get_data.get_efmtool_efms(inputfile)
        decompressing.run(compressed_efms, infofile, outputfile)

    else:
        print("Decompressing EFMs from MPLRS")
        compressed_efms = get_data.get_mplrs_efms(inputfile, infofile)
        decompressing.run(compressed_efms, infofile, outputfile)
    efmlrs_finish_decompressions()


def start(inputfile, outputfile, infofile, efmtool):
    """
    Takes all parameters form commandline, checks if parameters are okay and calls main function if an error occurs an
    exception is raised and ends the program.
    :param inputfile: output containing compressed efms either form mplrs or efmtool
    :param outputfile: file in which uncompressed efms are written to
    :param infofile: file automatically created during compressions, containing all information for decompressions
    :param efmtool: paramter that indicates whether decompressions for mplrs or efmtool is being executed
    :return: None
    """
    try:
        main(inputfile, outputfile, infofile, efmtool)
    except Exception:
        print("crashed...")
        raise

def start_from_command_line():
    """
    Entry point for prostprocessing EFMlrs. Contains all information and arguments for command line call.
    Calls start function.
    :return: None
    """
    usage = '''usage: efmlrs_post -i <input file with compressed efms> -o <output file> -info <efmlrs info file> [--efmtool]'''
    parser = ArgumentParser(prog='EFMlrs', description='Extracts EFMs from result file of mplrs,\n'
                                                       'and decompresses EFMs from efmtool and mplrs results\n'
                                                       'that have been compressed with EFMlrs', epilog=usage,
                            formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument("-i", "--inputfile", help="name of input file that contains compressed EFMs")
    parser.add_argument("-o", "--outputfile", help="name of output file")
    parser.add_argument("-info", "--infofile", help="name of info file that was created during compressions")
    parser.add_argument("--efmtool", action='store_true', help="if parameter --efmtool is given, compressed efmtool result is expected as input")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    start(args.inputfile, args.outputfile, args.infofile, args.efmtool)

if __name__ == "__main__":
    start_from_command_line()
