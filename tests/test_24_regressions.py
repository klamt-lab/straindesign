"""Regression tests for defects found in the review before the release after 1.19.2."""
from .test_01_load_models_and_solvers import model_weak_coupling, model_gpr
import pytest
import straindesign as sd
from straindesign.names import *


def _sets(solution):
    return sorted(sorted(d) for d in solution.reaction_sd)


def test_optknock_populate_returns_the_designs_of_best(curr_solver, model_weak_coupling):
    """In a bilevel problem a smaller design with a worse objective value is a result of its own.
    The check for non-minimal cut sets must not apply: it turned this run into status 'error'
    without designs."""
    modules = [sd.SDModule(model_weak_coupling, OPTKNOCK, inner_objective='r_BM', outer_objective='r_P', constraints=['r_BM >= 0.2'])]
    kw = dict(sd_modules=modules, max_cost=2, max_solutions=10, solver=curr_solver)
    best = sd.compute_strain_designs(model_weak_coupling, solution_approach=BEST, **kw)
    pop = sd.compute_strain_designs(model_weak_coupling, solution_approach=POPULATE, **kw)
    assert best.status == OPTIMAL and pop.status == OPTIMAL
    assert _sets(pop) == _sets(best)
    assert len(pop.reaction_sd) > 1


def test_reaction_bounds_of_selected_designs(model_weak_coupling):
    designs = [{'r1': -1.0}, {'r2': -1.0}, {'r4': -1.0, 'r5': -1.0}]
    sol = sd.SDSolutions(model_weak_coupling, designs, OPTIMAL, {MODEL_ID: model_weak_coupling.id, MAX_COST: 3})
    every = sol.get_reaction_sd_bnds()
    assert len(every) == 3
    assert sol.get_reaction_sd_bnds(1) == [every[1]]
    assert sol.get_reaction_sd_bnds([0, 2]) == [every[0], every[2]]


def test_cost_dicts_and_setup_of_the_caller_are_left_alone(model_gpr):
    modules = [sd.SDModule(model_gpr, SUPPRESS, constraints=["1.0 rd_ex >= 1.0 "])]
    modules += [sd.SDModule(model_gpr, PROTECT, constraints=[[{'r_bm': 1.0}, '>=', 1.0]])]
    ko_cost = {'rs_up': 1.0, 'rd_ex': 1.0, 'rp_ex': 1.1, 'r_bm': 0.75}
    gko_cost = {'g1': 1.0, 'g2': 1.0, 'g4': 3.0, 'g5': 2.0, 'g6': 1.0, 'g7': 1.0, 'g8': 1.0, 'g9': 1.0}
    gki_cost = {'g3': 1.0}
    setup = {MODULES: modules, MAX_COST: 2, MAX_SOLUTIONS: 1, SOLUTION_APPROACH: ANY, KOCOST: ko_cost, GKOCOST: gko_cost, GKICOST: gki_cost}
    before = (dict(ko_cost), dict(gko_cost), dict(gki_cost), set(setup))
    sd.compute_strain_designs(model_gpr, sd_setup=setup)
    assert (ko_cost, gko_cost, gki_cost, set(setup)) == before


@pytest.mark.parametrize('approach', [ANY, BEST])
def test_run_that_ends_on_the_clock_is_not_optimal(curr_solver, model_weak_coupling, approach):
    """A time limit that leaves no time for a single solve must not report a finished computation."""
    modules = [sd.SDModule(model_weak_coupling, SUPPRESS, constraints=['r_P <= 0.5', 'r_BM >= 0.1'])]
    sol = sd.compute_strain_designs(model_weak_coupling,
                                    sd_modules=modules,
                                    max_cost=3,
                                    solution_approach=approach,
                                    solver=curr_solver,
                                    time_limit=1e-9)
    assert sol.status in (TIME_LIMIT, TIME_LIMIT_W_SOL)
