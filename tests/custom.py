from os.path import dirname, abspath
from cobra.io import read_sbml_model
import straindesign as sd
from straindesign.names import *
from numpy import inf, isinf
from cobra import Configuration

curr_solver = 'gurobi'
model_small_example = read_sbml_model(dirname(abspath(__file__)) + "/model_small_example.xml")

solver1 = sd.select_solver()
assert(solver1 in [CPLEX,GUROBI,GLPK,SCIP])

solver2 = sd.select_solver('gurobi')
assert(solver2 == GUROBI)

model_small_example.solver = 'glpk'
solver3 = sd.select_solver(None, model_small_example)
assert(solver3 == GLPK)

conf = Configuration()
conf.solver = 'cplex'
solver4 = sd.select_solver()
assert(solver4 == CPLEX)


sd.select_solver(solver=None, model=None)
sd.select_solver(solver=None, model=None)
sd.select_solver(solver=None, model=None)
sd.select_solver(solver=None, model=None)
sd.select_solver(solver=None, model=None)
pass