from os.path import dirname, abspath
from cobra.io import read_sbml_model
import straindesign as sd
from straindesign.names import *
from numpy import inf, isinf
from cobra import Configuration
import logging

logging.basicConfig(level="INFO")

curr_solver = 'cplex'
comp_approach = POPULATE
model_weak_coupling = read_sbml_model(dirname(abspath(__file__)) + "/model_weak_coupling.xml")
modules = [sd.SDModule(model_weak_coupling, SUPPRESS, constraints=["r_P - 0.4 r_S <= 0", "r_S >= 0.1"])]
modules += [sd.SDModule(model_weak_coupling, PROTECT, constraints=["r_BM >= 0.5"])]
kocost = {
    'r1':1.0,
    'r2':1.0,
    'r4':1.1,
    'r5':0.75,
    'r7':0.8,
    'r8':1.0,
    'r9':1.0,
    'r_S':1.0,
    'r_P':1.0,
    'r_BM':1.0,
    'r_Q':1.5
}
kicost = {
    'r3':0.6,
    'r6':1.0,
}
regcost = {
    'r6 >= 4.5': 1.2
}
sd_setup = {MODULES:modules, 
            MAX_COST:4, 
            MAX_SOLUTIONS:inf, 
            SOLUTION_APPROACH:comp_approach,
            KOCOST:kocost,
            KICOST:kicost,
            REGCOST:regcost,
            SOLVER:curr_solver}
solution = sd.compute_strain_designs(model_weak_coupling, sd_setup=sd_setup)
pass