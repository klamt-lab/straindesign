import cobra
from mcs.names import *
import numpy as np
from importlib import reload
import mcs
import os
from scipy import sparse
#import traceback
#import warnings
import sys

# load network
network = cobra.io.read_sbml_model(os.path.dirname(os.path.abspath(__file__))+"/ECC2_23BDO.sbml")

# specify modules
modules  = [mcs.SD_Module(network,"mcs_lin",module_sense="desired",constraints=["BIOMASS_Ec_iJO1366_core_53p95M >= 0.05"])]
modules += [mcs.SD_Module(network,"mcs_lin",module_sense="undesired",constraints=["EX_23bdo_e + 0.3 EX_glc__D_e <= 0"])]

# specify MCS setup
maxSolutions = np.inf
maxCost = 5
solver = 'cplex'

# ko_cost = {'EX_o2_e'	: 0.4}
# ki_cost = None
gko_cost = None

M=None
# construct MCS MILP
mcsEnum = mcs.StrainDesigner(network,modules, gko_cost = gko_cost, max_cost=maxCost, solver=solver,M=M)
# mcsEnum = mcs.StrainDesignMILP(network,modules, max_cost=maxCost, solver=solver,M=M)

# solve MILP
rmcs = mcsEnum.enumerate(max_solutions=maxSolutions)
# mcsEnum.compute_optimal(max_solutions=maxSolutions)
# rmcs = mcsEnum.compute(max_solutions=maxSolutions,time_limit=15)
print(len(rmcs.get_sd()))
pass