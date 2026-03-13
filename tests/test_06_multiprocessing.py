"""Test if all strain design functions run correctly."""
from .test_01_load_models_and_solvers import *
import platform
import straindesign as sd
import logging


@pytest.mark.timeout(600)
def test_mcs_larger_model(curr_solver, model_core):
    logging.basicConfig(level=logging.INFO)
    if curr_solver == GLPK and platform.system() == 'Darwin':
        pytest.skip("GLPK has numerical precision issues on macOS ARM64 with larger models")
    # Set model solver so SDModule prechecks use the correct solver.
    # SCIP has no optlang interface, so skip_checks and let compute_strain_designs validate.
    if curr_solver != SCIP:
        model_core.solver = curr_solver
    skip = (curr_solver == SCIP)
    modules = [sd.SDModule(model_core, SUPPRESS, constraints=["EX_etoh_e <= 0.5"], skip_checks=skip)]
    modules += [sd.SDModule(model_core, SUPPRESS, constraints=["EX_etoh_e <= 0.5"], skip_checks=skip)]
    modules += [sd.SDModule(model_core, SUPPRESS, constraints=["EX_etoh_e <= 0.5"], skip_checks=skip)]
    modules += [sd.SDModule(model_core, PROTECT, constraints=["BIOMASS_Ec_iML1515_core_75p37M >= 0.1"], skip_checks=skip)]
    sd_setup = {MODULES: modules, MAX_COST: 4, MAX_SOLUTIONS: 0, SOLUTION_APPROACH: ANY, SOLVER: curr_solver, 'compress': False}
    if curr_solver in (GUROBI, CPLEX):
        logging.info(f"Enabling compression for {curr_solver} (uncompressed model exceeds community license limit).")
        sd_setup['compress'] = True
    solution = sd.compute_strain_designs(model_core, sd_setup=sd_setup)
    assert (len(solution.get_reaction_sd()) == 0)
