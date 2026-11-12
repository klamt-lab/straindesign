"""Restarts in the cost-level sweep must not change the designs.

A tiny work cap makes nearly every populate stop early, so the sweep resumes and restarts throughout;
the result has to be the same design set the uncapped run returns, and the run has to be repeatable.
"""
import logging
from .test_01_load_models_and_solvers import *
import straindesign as sd
import straindesign.strainDesignMILP as sdm
from ._models import load_test_model


def _designs(model, solver):
    sol = sd.compute_strain_designs(model,
                                    sd_modules=[sd.SDModule(model, SUPPRESS, constraints=['BIOMASS_Ecoli_core_w_GAM >= 0.1'])],
                                    max_cost=3,
                                    solution_approach=POPULATE,
                                    solver=solver,
                                    seed=3)
    assert sol.status == OPTIMAL
    return {frozenset(d) for d in sol.get_reaction_sd()}


@pytest.mark.parametrize('solver', [s for s in (CPLEX, GUROBI) if s in sd.avail_solvers])
def test_restarts_keep_the_design_set(solver, monkeypatch, caplog):
    model = load_test_model('e_coli_core')
    reference = _designs(model, solver)
    assert reference
    monkeypatch.setattr(sdm, '_RESTART_MIN_WORK', {CPLEX: 1.0, GUROBI: 1e-4})
    with caplog.at_level(logging.INFO):
        first = _designs(model, solver)
    n_first = caplog.text.count('restarting from another seed')
    caplog.clear()
    with caplog.at_level(logging.INFO):
        second = _designs(model, solver)
    n_second = caplog.text.count('restarting from another seed')
    assert first == reference and second == reference
    assert n_first > 0 and n_first == n_second
