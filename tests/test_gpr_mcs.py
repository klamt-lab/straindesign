import cobra
import straindesign as sd
from straindesign.names import *
import numpy as np
from re import search
import logging
from cobra import Metabolite, Reaction

iMLcore = cobra.io.read_sbml_model('tests/iMLcore.xml')

module_optknock = sd.SDModule(iMLcore,sd.OPTKNOCK,
                              inner_objective='BIOMASS_Ec_iML1515_core_75p37M',
                              outer_objective='EX_etoh_e',
                              constraints=['BIOMASS_Ec_iML1515_core_75p37M >= 0.1','EX_etoh_e >= 1'])

gko_cost = {g.name:1 for g in iMLcore.genes}
gko_cost.pop('s0001')
# possible knockout of O2
ko_cost = {'EX_o2_e': 1}

sd_setup = {
    MODULES: module_optknock,
    MAX_COST: 7,
    MAX_SOLUTIONS: 3,
    SOLUTION_APPROACH: ANY,
    SOLVER: 'cplex',
    GKOCOST: gko_cost,
    KOCOST: ko_cost,
}

solution = sd.compute_strain_designs(iMLcore, sd_setup=sd_setup)
num_solutions = len(solution.reaction_sd)
print(f"✓ Found {num_solutions} strain design solutions")
for i, sol in enumerate(solution.get_reaction_sd()[:20]):
    print(f"  Solution {i+1}: {dict(list(sol.items())[:20])}{'...' if len(sol) > 20 else ''}")
if num_solutions > 20:
    print(f"  ... and {num_solutions - 20} more solutions")