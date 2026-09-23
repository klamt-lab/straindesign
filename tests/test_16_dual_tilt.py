import logging
import pytest
from .test_01_load_models_and_solvers import *
import straindesign as sd
from numpy import inf
from straindesign.names import *


def _designs(model, solver, **extra):
    biomass = next(r.id for r in model.reactions if r.objective_coefficient)
    modules = [sd.SDModule(model, SUPPRESS, constraints=[f"{biomass} >= 0.1"])]
    sd_setup = {MODULES: modules, MAX_COST: 2, MAX_SOLUTIONS: inf, SOLUTION_APPROACH: POPULATE,
                SOLVER: solver, 'compress': True, SEED: 1, MILP_THREADS: 2}
    sd_setup.update(extra)
    solution = sd.compute_strain_designs(model, sd_setup=sd_setup)
    assert solution.status == OPTIMAL
    return {frozenset(d.items()) for d in solution.get_reaction_sd()}


@pytest.mark.timeout(240)
def test_dual_tilt_is_design_neutral_and_engages(curr_solver, model_core, monkeypatch, caplog):
    """The tilt is a search direction, not a bound: it must return exactly the designs the
    untilted enumeration returns, and it must actually be applied when requested."""
    if curr_solver not in (CPLEX, GUROBI):
        pytest.skip("the tilt acts inside the pinned-level populate loop, which needs a pool solver")
    monkeypatch.setenv('SD_ENUM_KSWEEP', '1')
    monkeypatch.setenv('SD_SOS1_GATES', '1')
    monkeypatch.setenv('SD_SOS1_NOPAIR', '1')
    monkeypatch.setenv('SD_POOL_OPEN', '1')
    monkeypatch.delenv('SD_DUAL_TILT', raising=False)
    monkeypatch.delenv('SD_DUAL_TILT_SOS', raising=False)

    reference = _designs(model_core, curr_solver, dual_tilt=None)   # explicit: the default is on
    assert len(reference) > 0

    assert _designs(model_core, curr_solver, dual_tilt=0) == reference

    with caplog.at_level(logging.INFO):
        tilted = _designs(model_core, curr_solver, dual_tilt=1e-6)
    assert tilted == reference
    engaged = [r.getMessage() for r in caplog.records if 'dual tilt' in r.getMessage()]
    assert engaged, "dual_tilt=1e-6 was accepted but never applied"
    assert ' on 0 of ' not in engaged[0], engaged[0]
