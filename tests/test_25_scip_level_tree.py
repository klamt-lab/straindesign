"""SCIP enumerates a cost level in one tree through a constraint handler."""
from .test_01_load_models_and_solvers import model_small_example
import pytest
import straindesign as sd
from straindesign.names import *


def _mcs(model, solver, **kw):
    modules = [sd.SDModule(model, SUPPRESS, constraints=["R4 >= 1"])]
    return sd.compute_strain_designs(model, sd_modules=modules, solution_approach=POPULATE, max_cost=3, solver=solver, **kw)


def test_scip_populate_matches_glpk(model_small_example):
    if SCIP not in sd.avail_solvers:
        pytest.skip('SCIP not available')
    scip = _mcs(model_small_example, SCIP)
    ref = _mcs(model_small_example, GLPK)
    assert scip.status == OPTIMAL
    assert len(scip.reaction_sd) > 1
    assert sorted(sorted(d) for d in scip.reaction_sd) == sorted(sorted(d) for d in ref.reaction_sd)


def test_scip_populate_respects_max_solutions(model_small_example):
    if SCIP not in sd.avail_solvers:
        pytest.skip('SCIP not available')
    every = _mcs(model_small_example, SCIP)
    some = _mcs(model_small_example, SCIP, max_solutions=1)
    ref = _mcs(model_small_example, GLPK, max_solutions=1)
    # designs of size 1 come from preprocessing and do not count towards the limit
    assert len(some.reaction_sd) == len(ref.reaction_sd) < len(every.reaction_sd)
    assert all(d in every.reaction_sd for d in some.reaction_sd)
