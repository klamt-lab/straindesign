from os.path import dirname, abspath
from cobra.io import read_sbml_model
import straindesign as sd
from straindesign.names import *
from numpy import inf, isinf
from cobra import Configuration

milp = sd.MILP_LP(solver='cplex')
milp.solve()

curr_solver = 'gurobi'
model_small_example = read_sbml_model(dirname(abspath(__file__)) + "/model_small_example.xml")

# with no solver specified
solver1 = sd.select_solver()
assert(solver1 in [CPLEX,GUROBI,GLPK,SCIP])

# with solver specified
solver2 = sd.select_solver('scip')
assert(solver2 == SCIP)

# with model-specified solver
model_small_example.solver = 'glpk'
solver3 = sd.select_solver(None, model_small_example)
assert(solver3 == GLPK)

# with cobrapy-specified solver
conf = Configuration()
conf.solver = 'cplex'
solver4 = sd.select_solver()
assert(solver4 == CPLEX)

# with solver in model that overwrites the global specification
model_small_example.solver = 'gurobi'
solver5 = sd.select_solver(None, model_small_example)
assert(solver5 == GUROBI)

# load solvers
from straindesign.cplex_interface import Cplex_MILP_LP
from straindesign.gurobi_interface import Gurobi_MILP_LP
from straindesign.glpk_interface import GLPK_MILP_LP
from straindesign.scip_interface import SCIP_MILP, SCIP_LP
pass