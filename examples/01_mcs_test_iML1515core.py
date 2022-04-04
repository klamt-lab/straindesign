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

network.metabolites.list_attr("compartment")
# specify MCS setup
maxSolutions = np.inf
maxCost = 7
solver = 'cplex'
ko_cost = { 'PTAr' 		: 1,
            'ACKr'		: 2,
            'FORtppi'	: 1.4,	
            'SUCDi'	    : 1.2,
            'H2Otpp'	: 2.6,	
            'ATPS4rpp'	: 1.3,
            'PFL'		: 0.7,
            'GLYCtex'	: 0.9,	
            'EX_ac_e'	: 0.4,	
            'ACtex'	    : 1.2,
            'CYTBO3_4pp': 2.1,	
            'O2tpp'		: 0.24,
            'O2tex'		: 0.77,
            'EX_o2_e'	: 0.4,	
            'POR5'		: 1.5,
            'CO2tpp'	: 3,	
            'ALCD19'	: 2,	
            'ASPtpp' 	: 0.25}
ki_cost = { 'EDD' 	 	: 0.5,
            'GAPP' 	 	: 0.4,
            'ENO'	 	: 0.6,
            'EDA' 	 	: 1.3,
            'AKGDH'  	: 0.7,
            'MALD' 	 	: 1.8,
            'NADTRHD'  	: 0.12,
            'LCARS'	 	: 0.13,
            'SUCCt1pp'	: 0.2,
            'SUCFUMtpp'	: 0.1}
# ko_cost = {'EX_o2_e'	: 0.4}
# ki_cost = None
# gko_cost = None

M=None
# construct MCS MILP
# mcsEnum = mcs.StrainDesigner(network,modules, max_cost=maxCost,ko_cost=ko_cost, ki_cost=ki_cost, gko_cost=gko_cost, solver=solver,M=M)
mcsEnum = mcs.StrainDesigner(network,modules, max_cost=maxCost,ko_cost=ko_cost, ki_cost=ki_cost, solver=solver,M=M)
# mcsEnum = mcs.StrainDesignMILP(network,modules,ko_cost=ko_cost, ki_cost=ki_cost, max_cost=maxCost,solver=solver,M=None)

# solve MILP
mcsEnum.enumerate(max_solutions=maxSolutions)
# mcsEnum.compute_optimal(max_solutions=maxSolutions)
# rmcs = mcsEnum.compute(max_solutions=maxSolutions,time_limit=15)
pass