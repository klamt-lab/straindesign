import cobra
from straindesign.names import *
import numpy as np
import os
import straindesign
from scipy import sparse, io
#import traceback
#import warnings

# load network
network = cobra.io.read_sbml_model(os.path.dirname(os.path.abspath(__file__))+"/ECC2_23BDO.sbml")

# specify modules
modules  = [straindesign.SDModule(network,PROTECT,constraints=["BIOMASS_Ec_iJO1366_core_53p95M >= 0.05"])]
modules += [straindesign.SDModule(network,SUPPRESS,constraints=["EX_23bdo_e + 0.3 EX_glc__D_e <= 0"])]

# specify MCS setup
maxSolutions = np.inf
maxCost = 9
solver = 'cplex'

ko_cost = {'EX_o2_e'	: 1.0}
# ki_cost = None
gko_cost = {k:1.0 for k in network.genes.list_attr('id') if k != 'spontanous'}

M=None
# construct MCS MILP
mcsEnum = straindesign.StrainDesigner(network,modules,compress=True, gko_cost = gko_cost, ko_cost=ko_cost, max_cost=maxCost, solver=solver,M=M)
# mcsEnum = straindesign.StrainDesignMILP(network,modules, max_cost=maxCost, solver=solver,M=M)

# solve MILP
# solutions = mcsEnum.enumerate(max_solutions=maxSolutions)
# solutions = mcsEnum.compute_optimal(max_solutions=maxSolutions)
solutions = mcsEnum.compute(max_solutions=3)
print(len(solutions.get_strain_designs()))
io.savemat('mcs_python.mat', mdict={'whatever_data': \
    [','.join(a) for a in [[k for k,v in m.items()] for m in solutions.get_strain_designs()]]})
pass