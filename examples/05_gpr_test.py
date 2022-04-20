import cobra
import numpy as np
from importlib import reload
import straindesigner
import os
from scipy import sparse
#import traceback
#import warnings
import sys

# load network
network = cobra.io.read_sbml_model(os.path.dirname(os.path.abspath(__file__))+"/gpr_model.sbml")

# specify modules
modules  = [straindesigner.SD_Module(network,"mcs_lin",module_sense="undesired",constraints=["rd_ex >= 1"])]
modules += [straindesigner.SD_Module(network,"mcs_lin",module_sense="desired",constraints=["r_bm >= 1"])]

# specify MCS setup
maxSolutions = np.inf
maxCost = 7
solver = 'cplex'

ko_cost = None
ki_cost = None
gko_cost = None
gki_cost = None #{'g2' : 1}

M=None
# construct MCS MILP
mcsEnum = straindesigner.StrainDesigner(network,modules, max_cost=maxCost,\
    ko_cost=ko_cost, ki_cost=ki_cost, gko_cost=gko_cost, gki_cost=gki_cost, solver=solver,M=M)
# mcsEnum = straindesigner.StrainDesigner(network,modules, max_cost=maxCost,\
#     ko_cost=ko_cost, ki_cost=ki_cost, solver=solver,M=M)
# mcsEnum = straindesigner.StrainDesignMILP(network,modules,ko_cost=ko_cost, ki_cost=ki_cost, max_cost=maxCost,solver=solver,M=None)

# solve MILP
# mcsEnum.enumerate(max_solutions=maxSolutions)
gmcs = mcsEnum.compute_optimal(max_solutions=maxSolutions)
# rmcs = mcsEnum.compute(max_solutions=maxSolutions,time_limit=15)
print(gmcs)