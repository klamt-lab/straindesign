"""Test if all strain design functions run correctly."""
from .test_01_load_models_and_solvers import *
import straindesign as sd
import logging


@pytest.mark.timeout(600)
def test_mcs_larger_model(curr_solver, model_core):
    logging.basicConfig(level=logging.INFO)
    modules = [sd.SDModule(model_core, SUPPRESS, constraints=["EX_etoh_e <= 0.5"])]
    modules += [sd.SDModule(model_core, SUPPRESS, constraints=["EX_etoh_e <= 0.5"])]
    modules += [sd.SDModule(model_core, SUPPRESS, constraints=["EX_etoh_e <= 0.5"])]
    # modules += [sd.SDModule(model_core, SUPPRESS, constraints=["EX_etoh_e <= 0.5"])]
    modules += [sd.SDModule(model_core, PROTECT, constraints=["BIOMASS_Ec_iML1515_core_75p37M >= 0.1"])]
    sd_setup = {MODULES: modules, MAX_COST: 4, MAX_SOLUTIONS: 0, SOLUTION_APPROACH: ANY, SOLVER: curr_solver, 'compress': False}
    solution = sd.compute_strain_designs(model_core, sd_setup=sd_setup)
    assert (len(solution.get_reaction_sd()) == 0)
