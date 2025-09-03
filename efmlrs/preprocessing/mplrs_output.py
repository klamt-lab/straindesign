from efmlrs.util.data import *


def split_reversible_reas(smatrix, reversibilities):
    """
    Splits all reversible reactions into two single reactions. One for forward and one for backward.

    :param smatrix: list of lists of stoichiometric matrix
    :param reversibilities: list of reaction reversibilities
    :return: list of lists of reconfigured stoichiometric matrix
    """
    reconfigured_smatrix = []

    for line in smatrix:
        vtmp = []
        for i in range(0, len(line)):
            if reversibilities[i] == 1:
                vtmp.append(line[i])
                vtmp.append(-line[i])
            else:
                vtmp.append(line[i])
        reconfigured_smatrix.append(vtmp)
    return reconfigured_smatrix


def write_header(mplrs_file, core_name):
    """
    Writes header for mplrs input file.

    :param mplrs_file: path to mplrs input file
    :param core_name: path to input file without extensions
    :return: None
    """
    mplrs_file.write("* " + core_name + "\n")
    mplrs_file.write("H-representation" + "\n")


def write_smatrix(mplrs_file, reconf_smatrix):
    """
    Writes reconfigured stoichiometric matrix, unity matrix and ending to mplrs input file.

    :param mplrs_file: path to mplrs input file
    :param reconf_smatrix: list of lists of reconfigured stoichiometric matrix
    :return: None
    """
    d = len(reconf_smatrix[0])
    s = len(reconf_smatrix)
    m = s + d

    mplrs_file.write("linearity " + str(s))
    for i in range(1, s + 1):
        mplrs_file.write(" " + str(i))
    mplrs_file.write("\n")
    mplrs_file.write("begin" + "\n")
    mplrs_file.write(str(m) + " " + str((d + 1)) + " rational \n")

    for line in reconf_smatrix:
        mplrs_file.write(format(0) + " ")
        for val in line:
            mplrs_file.write(format(val) + " ")
        mplrs_file.write("\n")

    for i in range(0, d):
        mplrs_file.write(format(0) + " ")
        for j in range(0, d):
            if i == j:
                mplrs_file.write(format(1) + " ")
            else:
                mplrs_file.write(format(0) + " ")
        mplrs_file.write("\n")
    mplrs_file.write("end" + "\n")
    

def write_lrs(core_name, reconf_smatrix):
    """
    Write input file for mplrs algorithm.

    :param core_name: path to input file without extensions
    :param reconf_smatrix: list of lists of reconfigured stoichiometric matrix
    :return: None
    """
    core_name += ".ine"
    mplrs_file = open(core_name, "w")
    write_header(mplrs_file, core_name)
    write_smatrix(mplrs_file, reconf_smatrix)
    mplrs_file.close()


def write_lrs_cmp(core_name, reconf_smatrix):
    """
    Write input file for mplrs algorithm.

    :param core_name: path to input file without extensions
    :param reconf_smatrix: list of lists of reconfigured stoichiometric matrix
    :return: None
    """
    core_name += "_cmp.ine"
    mplrs_file = open(core_name, "w")
    write_header(mplrs_file, core_name)
    write_smatrix(mplrs_file, reconf_smatrix)
    mplrs_file.close()


def run_cmp(core_name):
    """
    Entry point for mplrs_output. Creates input file from sfile and rvfile suitable for mplrs algorithm.

    :param core_name: path to input file without extensions
    :return: None
    """
    smatrix = read_sfile(core_name + "_cmp")
    reversibilities = read_rvfile(core_name + "_cmp")
    reactions = read_rfile(core_name + "_cmp")

    if len(smatrix) == 0:
        print("*** NO INE FILE CREATED! ***")
        if len(reactions) > 0:
            print("*** solved due to compression ***")
            print("EFMs = ", len(reactions))
        else:
            print("*** SMATRIX EMPTY! ***")
        return
    reconfigured_smatrix = split_reversible_reas(smatrix, reversibilities)
    write_lrs_cmp(core_name, reconfigured_smatrix)

def run_uncmp(core_name):
    """
    Entry point for mplrs_output. Creates input file from sfile and rvfile suitable for mplrs algorithm.

    :param core_name: path to input file without extensions
    :return: None
    """
    smatrix = read_sfile(core_name)
    reversibilities = read_rvfile(core_name)
    reactions = read_rfile(core_name)

    if len(smatrix) == 0:
        print("*** NO INE FILE CREATED! ***")
        if len(reactions) > 0:
            print("*** solved due to compression ***")
            print("EFMs = ", len(reactions))
        else:
            print("*** SMATRIX EMPTY! ***")
        return
    reconfigured_smatrix = split_reversible_reas(smatrix, reversibilities)
    write_lrs(core_name, reconfigured_smatrix)
