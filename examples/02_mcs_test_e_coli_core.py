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
network = cobra.io.read_sbml_model(os.path.dirname(os.path.abspath(__file__))+"/e_coli_core.sbml")

# specify modules
modules = mcs.SD_Module(network,"mcs_bilvl",module_sense="desired",constraints=["2 BIOMASS_Ecoli_core_w_GAM >= 0.1","EX_etoh_e >= 1"],inner_objective="BIOMASS_Ecoli_core_w_GAM")

mcs.fba(network,obj=modules['inner_objective'])
sol = mcs.fba(network,constraints=["EX_o2_e=0"])

# specify MCS setup
maxSolutions = np.inf
maxCost = 25
solver = 'cplex'
# ko_cost = { 'PFK'       : 1,
#             'PFL'       : 2,
#             'PGI'       : 1.4,
#             'PGK'       : 1.2,
#             'PGL'       : 2.6,
#             'AKGDH'     : 1.3,
#             'ATPS4r'    : 0.7,
#             'PTAr'      : 0.9,
#             'PYK'       : 0.4,
#             'SUCCt3'    : 1.2,
#             'ETOHt2r'   : 2.1,
#             'SUCDi'     : 0.24,
#             'SUCOAS'    : 0.77,
#             'TALA'      : 1.5,
#             'EX_o2_e'	: 0.9}
# ki_cost = { 'ACALD'     : 0.5,
#             'AKGt2r'    : 0.4,
#             'PGM'       : 0.6,
#             'PIt2r'     : 1.3,
#             'ALCD2x'    : 0.7,
#             'ACALDt'    : 1.8,
#             'ACKr'      : 0.12,
#             'PPC'       : 0.13,
#             'CS'        : 0.2,
#             'RPI'       : 0.1,
#             'SUCCt2_2'  : 0.6,        
#             'CYTBD'     : 1.3,        
#             'D_LACt2'   : 0.7,        
#             'ENO'       : 1.8,    
#             'THD2'      : 0.12,    
#             'TKT1'      : 0.13,    
#             'O2t'       : 0.2,    
#             'PDH'       : 0.1}
gko_cost = {'pfkA'      : 0.7,
            'pfkB'      : 1,
            'grcA'      : 2,
            'tdcE'      : 2.2,
            'pflC'      : 1.8,
            'pflA'      : 2,
            'pflB'      : 2,
            'pgi'       : 1.4,
            'pgk'       : 1.2,
            'pgl'       : 2.6,
            'sucA'      : 1.3,
            'sucB'      : 1.38,
            'lpd'       : 1.3,
            'atpA'      : 0.7,
            'pta'       : 0.9,
            'eutD'      : 0.9,
            'pykA'      : 0.4,
            'pykF'      : 0.5,
            'sdhA'      : 0.24,
            'sucD'      : 0.77,
            'talA'      : 1.5}
gki_cost = {'adhE'      : 0.5,
            'kgtP'      : 0.4,
            'gpmA'      : 0.6,
            'gpmM'      : 0.7,
            'pitA'      : 1.3,
            'pitB'      : 1.3,
            'adhP'      : 0.7,
            'mhpF'      : 1.8,
            'tdcD'      : 0.12,
            'purT'      : 0.14,
            'ackA'      : 0.17,
            'ppc'       : 0.13,
            'gltA'      : 0.2,
            'rpiB'      : 0.1,
            'rpiA'      : 0.1,
            'eno'       : 1.8,
            'tktA'      : 0.13,
            'tktB'      : 0.13,
            'lpd'       : 0.13,
            'aceE'      : 0.13,
            'aceF'      : 0.1}
ko_cost = { 'SUCCt2_2'  : 0.6,
            'ETOHt2r'   : 2.1,
            'CYTBD'     : 1.3,
            'THD2'      : 0.12,
            'D_LACt2'   : 0.7}
ki_cost = { 'O2t'       : 0.2,
            'SUCCt3'    : 1.2}
# ko_cost = None
# gko_cost = None
M=None
output_format = 'auto'
sd, status = mcs.compute_strain_designs(network,modules, max_cost=maxCost,ko_cost=ko_cost, \
                                        ki_cost=ki_cost, gko_cost=gko_cost, gki_cost=gki_cost, \
                                        solver=solver,M=None ,output_format=output_format, \
                                        compress=True)
# sd, status = mcs.compute_strain_designs(network,modules, max_cost=maxCost,ko_cost=ko_cost, \
#                                         ki_cost=ki_cost, solver=solver,M=None ,output_format=output_format)
[print(r) for r in sd]
exit()
# conrtruct MCS MILP
# mcsEnum = mcs.StrainDesigner(network,modules, max_cost=maxCost,gko_cost=gko_cost,ko_cost=ko_cost, ki_cost=ki_cost, solver=solver,M=M)
mcsEnum = mcs.StrainDesigner(network,modules, max_cost=maxCost,ko_cost=ko_cost, ki_cost=ki_cost, solver=solver,M=M)
# mcsEnum = mcs.StrainDesignMILP(network,modules,ko_cost=ko_cost, ki_cost=ki_cost, max_cost=maxCost,solver=solver,M=None)

# solve MILP
rmcs,status = mcsEnum.enumerate(max_solutions=maxSolutions,show_no_ki=True)
# rmcs,status = mcsEnum.compute_optimal(max_solutions=maxSolutions)
# rmcs = mcsEnum.compute(max_solutions=maxSolutions)
[print(r) for r in rmcs]
pass