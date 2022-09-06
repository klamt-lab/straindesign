from os.path import dirname, abspath
from cobra.io import read_sbml_model, load_model
import straindesign as sd
from straindesign.names import *
from numpy import inf
import logging

logging.basicConfig(level="INFO")

curr_solver = 'gurobi'
comp_approach = ANY
model_gpr = read_sbml_model(dirname(abspath(__file__)) + "/model_gpr.xml")
modules = [sd.SDModule(model_gpr, SUPPRESS, constraints=["1.0 rd_ex >= 1.0 "])]
modules += [sd.SDModule(model_gpr, PROTECT, constraints=[[{'r_bm': 1.0}, '>=', 1.0]])]
kocost = {'rs_up': 1.0, 'rd_ex': 1.0, 'rp_ex': 1.1, 'r_bm': 0.75}
gkocost = {
    'g1': 1.0,
    'g2': 1.0,
    'g4': 3.0,
    'g5': 2.0,
    'g6': 1.0,
    'g7': 1.0,
    'g8': 1.0,
    'g9': 1.0,
}
gkicost = {
    'g3': 1.0,
}
regcost = {'g4 <= 0.4': 1.2}
sd_setup = {
    MODULES: modules,
    MAX_COST: 2,
    MAX_SOLUTIONS: inf,
    SOLUTION_APPROACH: comp_approach,
    KOCOST: kocost,
    GKOCOST: gkocost,
    GKICOST: gkicost,
    REGCOST: regcost,
    SOLVER: curr_solver
}
solution = sd.compute_strain_designs(model_gpr, sd_setup=sd_setup)
assert (len(solution.gene_sd) == 4)
