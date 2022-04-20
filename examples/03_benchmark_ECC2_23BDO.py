import cobra
from straindesigner.names import *
import numpy as np
from importlib import reload
import mcs
import os
from scipy import sparse, io
#import traceback
#import warnings
import sys

# load network
network = cobra.io.read_sbml_model(os.path.dirname(os.path.abspath(__file__))+"/ECC2_23BDO.sbml")

# specify modules
modules  = [straindesigner.SD_Module(network,"mcs_lin",module_sense="desired",constraints=["BIOMASS_Ec_iJO1366_core_53p95M >= 0.05"])]
modules += [straindesigner.SD_Module(network,"mcs_lin",module_sense="undesired",constraints=["EX_23bdo_e + 0.3 EX_glc__D_e <= 0"])]

# specify MCS setup
maxSolutions = np.inf
maxCost = 9
solver = 'cplex'

ko_cost = {'EX_o2_e'	: 1.0}
# ki_cost = None
gko_cost = {k:1.0 for k in network.genes.list_attr('id') if k != 'spontanous'}

M=None
# construct MCS MILP
mcsEnum = straindesigner.StrainDesigner(network,modules,compress=True, gko_cost = gko_cost, ko_cost=ko_cost, max_cost=maxCost, solver=solver,M=M)
# mcsEnum = straindesigner.StrainDesignMILP(network,modules, max_cost=maxCost, solver=solver,M=M)

# solve MILP
solutions = mcsEnum.enumerate(max_solutions=maxSolutions)
# solutions = mcsEnum.compute_optimal(max_solutions=maxSolutions)
# solutions = mcsEnum.compute(max_solutions=3)
print(len(solutions.get_strain_designs()))
io.savemat('mcs_python.mat', mdict={'whatever_data': \
    [','.join(a) for a in [[k for k,v in m.items()] for m in solutions.get_strain_designs()]]})
pass