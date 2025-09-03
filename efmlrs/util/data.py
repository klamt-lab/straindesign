"""
Collection of small helper scripts.
"""
from fractions import Fraction
import numpy as np
import pandas as pd
from sympy import *
import efmlrs.util.convert_matrix as conversion


def write_all(smatrix, reactions, reversibilities, metabolites, name):
    write_sfile(name, smatrix)
    write_rvfile(name, reversibilities)
    write_init_rfile(name, reactions)
    write_init_mfile(name, metabolites)


def write_initial_files_with_bounds(core_name, reactions, reversibilities, metabolites):
    write_init_rfile(core_name, reactions)
    write_rvfile(core_name, reversibilities)
    write_init_mfile(core_name, metabolites)


def write_initial_files_no_bounds(core_name, reactions, reversibilities, metabolites):
    write_init_rfile(core_name, reactions)
    write_rvfile(core_name, reversibilities)
    write_init_mfile(core_name, metabolites)


def read_sfile_int(name):
    name += ".sfile"
    sfile = open(name, "r")
    smatrix = []
    for line in sfile:
        if line == "":
            continue
        vtmp = []
        for val in line.split():
            if val == "":
                continue
            vtmp.append(int(val))
        smatrix.append(vtmp)
    sfile.close()
    return smatrix


def read_sfile(name):
    name += ".sfile"
    sfile = open(name, "r")
    smatrix = []
    for line in sfile:
        if line == "":
            continue
        vtmp = []
        for val in line.split():
            if val == "":
                continue
            vtmp.append(Fraction(val))
        smatrix.append(vtmp)
    sfile.close()
    return smatrix


def write_sfile(name, smatrix):
    name += ".sfile"
    file = open(name, "w")
    for i in range(0, smatrix.shape[0]):
        row = smatrix.row(i)
        for j in range(0, smatrix.shape[1]):
            val = row[j]
            file.write(str(val))
            file.write(" ")
        file.write("\n")
    file.close()


def write_sfile_float(name, smatrix):
    name += ".sfile"
    file = open(name, "w")
    for row in smatrix:
        cnt = 0
        for val in row:
            cnt += 1
            if cnt < len(smatrix[0]):
                check = (val).is_integer()
                if check is True:
                    file.write(str(int(val)))
                    file.write(" ")
                if check is False:
                    file.write(str(float(val)))
                    file.write(" ")
            else:
                check = (val).is_integer()
                if check is True:
                    file.write(str(int(val)))
                if check is False:
                    file.write(str(float(val)))
        file.write("\n")
    file.close()


def convert_float2fraction(matrix):
    fr_matrix = [[Fraction(str(val)) for val in line] for line in matrix]
    return fr_matrix


def convert_fraction2float(matrix):
    float_matrix = [[float(val) for val in line] for line in matrix]
    return float_matrix


def convert_matrix2df(smatrix):
    smatrix = np.array(smatrix)
    smatrix = pd.DataFrame(smatrix)
    return smatrix


def convert_df2matrix(smatrix_df):
    smatrix = Matrix(smatrix_df)
    return smatrix


def read_mfile(name):
    name += ".mfile"
    file = open(name, "r")
    metabolism = file.read().split()
    file.close()
    return metabolism


def write_mfile(name, metabolism):
    name += ".mfile"
    file = open(name, "w")
    for i in metabolism:
        file.write(i + " ")
    file.close()


def write_init_mfile(name, metabolism):
    name += ".mfile"
    file = open(name, "w")
    for i in metabolism:
        file.write('"')
        file.write(i)
        file.write('" ')
    file.close()


def read_rfile(name):
    name += ".rfile"
    file = open(name, "r")
    reactions = file.read().split()
    file.close()
    return reactions


def write_rfile(name, reactions):
    name += ".rfile"
    file = open(name, "w")
    for i in reactions:
        file.write(i + " ")
    file.close()


def write_init_rfile(name, reactions):
    name += ".rfile"
    file = open(name, "w")
    for i in reactions:
        file.write('"')
        file.write(i)
        file.write('" ')
    file.close()


def read_rvfile(name):
    name += ".rvfile"
    file = open(name, "r")
    reversibles = file.read().split()
    file.close()
    return [bool(int(i)) for i in reversibles]


def write_rvfile(name, reversibles):
    name += ".rvfile"
    file = open(name, "w")
    for i in [str(int(j)) for j in reversibles]:
        file.write(i + " ")
    file.close()


def write_efms(name, efms):
    efm_matrix = Matrix(efms)
    file = open(name, "w")
    for i in range(0, efm_matrix.shape[0]):
        row = efm_matrix.row(i)
        for j in range(0, efm_matrix.shape[1]):
            val = row[j]
            file.write(str(val))
            file.write(" ")
        file.write("\n")
    file.close()


def write_info(core_name, reversibilities, outer_counter):
    info_file_name = core_name + ".info"
    file = open(info_file_name, "a")
    file.write("rv: ")
    for i in [str(int(j)) for j in reversibilities]:
        file.write(i + " ")
    file.write("\n")
    file.write("counter " + str(outer_counter))
    file.close()


def bounds_print(bounds):
    for name, val in bounds.items():
        if name.endswith("_max"):
            print("Upper bound:", val, "for reaction", name[:-4])
        else:
            print("Lower bound:", val, "for reaction", name[:-4])


def reversibilities4printing(reversibilities):
    i = 0
    for val in reversibilities:
        if val is True:
            i += 1
    return i


def efmlrs_start_compressions():
    print('                          ')
    print(r'           EFMlrs     __ ')
    print(r'    (\   .-.   .-.   /_")')
    print(r'     \\_//^\\_//^\\_//   ')
    print(r'      `"´   `"´   `"´    ')
    print('     start compressions   ')
    print('                          ')


def efmlrs_finish_compressions():
    print('                          ')
    print(r'                    .-.  ')
    print(r'                  /  oo  ')
    print(r'   EFMlrs         \ -,_) ')
    print(r'             _..._| \ `-<')
    print(r'        {} ." .__.\' |   ')
    print(r'       {} (         /`\  ')
    print(r'       {}(`´------´   /  ')
    print(r'          `----------´   ')
    print(r'   finished compressions ')
    print(r'                          ')


def efmlrs_start_decompressions():
    print(r'                          ')
    print(r'                    .-.  ')
    print(r'                  /  oo  ')
    print(r'   EFMlrs         \ -,_) ')
    print(r'             _..._| \ `-<')
    print(r'        {} ." .__.\' |   ')
    print(r'       {} (         /`\  ')
    print(r'       {}(`´------´   /  ')
    print(r'          `----------´   ')
    print(r'      start decompressions   ')
    print(r'                             ')


def efmlrs_finish_decompressions():
    print('                          ')
    print(r'           EFMlrs     __ ')
    print(r'    (\   .-.   .-.   /_")')
    print(r'     \\_//^\\_//^\\_//   ')
    print(r'      `"´   `"´   `"´    ')
    print('   finished decompressions')
    print('                          ')


def write_uncmp_int_matrix(core_name):
    smatrix = read_sfile(core_name)
    smatrix = Matrix(smatrix)
    write_sfile(core_name, smatrix)
    smatrix = Matrix(smatrix)
    write_sfile(core_name + "_fractions", smatrix)


def write_cmp_int_matrix(core_name):
    smatrix = read_sfile(core_name + "_cmp")
    int_smatrix = conversion.run(smatrix)
    write_sfile(core_name + "_cmp", int_smatrix)
