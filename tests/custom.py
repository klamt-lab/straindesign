from os.path import dirname, abspath
from cobra.io import read_sbml_model, load_model
import straindesign as sd
from straindesign.names import *
from numpy import inf
import logging

logging.basicConfig(level="INFO")

curr_solver = 'gurobi'

iMLcore = read_sbml_model(dirname(abspath(__file__)) + "/iMLcore.xml")

iMLcore = read_sbml_model("C:/Users/phili/OneDrive/Dokumente/Python/straindesign/tests/iMLcore.xml")

modules = [sd.SDModule(iMLcore, SUPPRESS, constraints=['EX_etoh_e <= 1.0', 'BIOMASS_Ec_iML1515_core_75p37M >= 0.14'])]
modules += [sd.SDModule(iMLcore, PROTECT, constraints=['BIOMASS_Ec_iML1515_core_75p37M >= 0.15'])]

sol = sd.fba(iMLcore, solver=curr_solver,constraints=[[{'EX_lac__D_e': 1.0}, '<=', 1.0], [{'BIOMASS_Ec_iML1515_core_75p37M': 1.0}, '>=', 0.2]])

sd.plot_flux_space(iMLcore, axes=['BIOMASS_Ec_iML1515_core_75p37M','EX_etoh_e'], solver=curr_solver,constraints='EX_o2_e = 0.0')

# , constraints=[[{'EX_lac__D_e': 1.0}, '<=', 1.0], [{'BIOMASS_Ec_iML1515_core_75p37M': 1.0}, '>=', 0.2]]

sd_setup = {
    MODULES: modules,
    MAX_COST: 2,
    MAX_SOLUTIONS: inf,
    SOLUTION_APPROACH: POPULATE,
    'gene_kos': True,
    SOLVER: curr_solver
}
solution = sd.compute_strain_designs(iMLcore, sd_setup=sd_setup)
assert (len(solution.gene_sd) == 4)
