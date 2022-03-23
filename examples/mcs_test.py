import cobra
import numpy as np
from importlib import reload
import mcs
import os
from scipy import sparse
#import traceback
#import warnings
import sys

# load network
network = cobra.io.read_sbml_model(os.path.dirname(os.path.abspath(__file__))+"/iML1515core.sbml")
# cobra.flux_analysis.flux_variability_analysis(network)
# network = cobra.io.read_sbml_model("iML1515core.sbml")
# results = mcs.fva(network,solver='cplex')
# remove external metabolites and obsolety
# e reactions
external_mets = [i for i,cpts in zip(network.metabolites,network.metabolites.list_attr("compartment")) if cpts == 'External_Species']
network.remove_metabolites(external_mets)
S = cobra.util.create_stoichiometric_matrix(network)
obsolete_reacs = [reac for reac,b_rempty in zip(network.reactions,np.any(S,0)) if not b_rempty]
network.remove_reactions(obsolete_reacs)
# close some uptakes
# network.reactions.AcUp.upper_bound = 0
# network.reactions.GlycUp.upper_bound = 0

sol = mcs.fba(network,constraints=["EX_o2_e=0"])
# specify modules
modules = mcs.SD_Module(network,"mcs_bilvl",module_sense="desired",constraints=["2 BIOMASS_Ec_iML1515_core_75p37M >= 0.1","EX_etoh_e >= 1"],inner_objective="BIOMASS_Ec_iML1515_core_75p37M")

network.remove_metabolites
network.metabolites.list_attr("compartment")
# specify MCS setup
maxSolutions = np.inf
maxCost = 3
solver = 'cplex'
# ko_cost = {'EDD':2, 'EDA':3,'FBA':1.5,'ENO':1.2,'ATPS4rpp':1,'PPS':1,'PGI':1.1,'GND':1,'h_pEx':1,'PGM':1.0,'AKGDH':1,'TALA':1,'TKT1':6,'RPE':1}
# ki_cost = {'ACKr':2.1, 'ICL':0.9,'MALS':1.5,'MDH':3, 'EDD':2, 'ENO':0.7}
ko_cost = None
ki_cost = None

# construct MCS MILP
mcsEnum = mcs.StrainDesigner(network,modules, max_cost=maxCost,ko_cost=ko_cost, ki_cost=ki_cost, solver=solver,M=None)
# mcsEnum = mcs.StrainDesignMILP(network,modules,ko_cost=ko_cost, ki_cost=ki_cost, max_cost=maxCost,solver=solver,M=None)

# solve MILP
mcsEnum.enumerate(max_solutions=maxSolutions)
# mcsEnum.compute_optimal(max_solutions=maxSolutions)
# rmcs = mcsEnum.compute(max_solutions=maxSolutions,time_limit=15)
pass