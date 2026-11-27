"""Regression tests for inner_opt_sense and outer_opt_sense lookup in SDProblem."""

import cobra
import pytest
import straindesign as sd
from straindesign.names import (
    DOUBLEOPT,
    MAXIMIZE,
    MINIMIZE,
    OPTIMAL,
    OPTKNOCK,
    PROTECT,
    ROBUSTKNOCK,
    SUPPRESS,
)


def _make_chain_model():
    model = cobra.Model("chain")
    met_a = cobra.Metabolite("A")
    r_in = cobra.Reaction("r_in", lower_bound=2.0, upper_bound=10.0)
    r_in.add_metabolites({met_a: 1.0})
    r_p = cobra.Reaction("r_p", lower_bound=0.0, upper_bound=10.0)
    r_p.add_metabolites({met_a: -1.0})
    model.add_reactions([r_in, r_p])
    return model


def _make_branch_model(in_bounds=(1.0, 10.0), b_bounds=(0.0, 5.0), c_bounds=(0.0, 5.0)):
    model = cobra.Model("branch")
    met_a = cobra.Metabolite("A")
    r_in = cobra.Reaction("r_in", lower_bound=in_bounds[0], upper_bound=in_bounds[1])
    r_in.add_metabolites({met_a: 1.0})
    r_b = cobra.Reaction("r_b", lower_bound=b_bounds[0], upper_bound=b_bounds[1])
    r_b.add_metabolites({met_a: -1.0})
    r_c = cobra.Reaction("r_c", lower_bound=c_bounds[0], upper_bound=c_bounds[1])
    r_c.add_metabolites({met_a: -1.0})
    model.add_reactions([r_in, r_b, r_c])
    return model


def test_protect_inner_minimize(curr_solver):
    model = _make_chain_model()
    mod = sd.SDModule(
        model,
        PROTECT,
        inner_objective="r_in",
        inner_opt_sense=MINIMIZE,
        constraints="r_p <= 3",
    )
    sol = sd.compute_strain_designs(model, sd_modules=[mod], ko_cost={}, solver=curr_solver, compress=False)
    assert sol.status == OPTIMAL
    assert sol.reaction_sd == [{}]


def test_suppress_inner_minimize(curr_solver):
    model = _make_chain_model()
    mod = sd.SDModule(
        model,
        SUPPRESS,
        inner_objective="r_in",
        inner_opt_sense=MINIMIZE,
        constraints="r_p >= 5",
    )
    sol = sd.compute_strain_designs(model, sd_modules=[mod], ko_cost={}, solver=curr_solver, compress=False)
    assert sol.status == OPTIMAL
    assert sol.reaction_sd == [{}]


def test_optknock_inner_minimize(curr_solver):
    model = _make_branch_model()
    mod = sd.SDModule(
        model,
        OPTKNOCK,
        inner_objective="r_in",
        inner_opt_sense=MINIMIZE,
        outer_objective="r_b",
        outer_opt_sense=MAXIMIZE,
    )
    milp = sd.SDMILP(model, [mod], ko_cost={}, solver=curr_solver)
    _, obj, status = milp.solve()
    assert status == OPTIMAL
    assert obj == pytest.approx(-1.0, abs=1e-6)


def test_robustknock_inner_minimize(curr_solver):
    model = _make_branch_model(in_bounds=(1.0, 10.0), b_bounds=(0.0, 10.0), c_bounds=(0.0, 10.0))
    mod = sd.SDModule(
        model,
        ROBUSTKNOCK,
        inner_objective="r_in",
        inner_opt_sense=MINIMIZE,
        outer_objective="r_b",
        outer_opt_sense=MAXIMIZE,
    )
    milp = sd.SDMILP(model, [mod], ko_cost={"r_c": 1}, solver=curr_solver)
    _, obj, status = milp.solve()
    assert status == OPTIMAL
    assert obj == pytest.approx(-1.0, abs=1e-6)


def test_robustknock_outer_minimize(curr_solver):
    model = _make_branch_model(in_bounds=(1.0, 10.0), b_bounds=(2.0, 10.0), c_bounds=(0.0, 8.0))
    mod = sd.SDModule(
        model,
        ROBUSTKNOCK,
        inner_objective="r_in",
        inner_opt_sense=MAXIMIZE,
        outer_objective="r_b",
        outer_opt_sense=MINIMIZE,
    )
    milp = sd.SDMILP(model, [mod], ko_cost={}, solver=curr_solver)
    _, obj, status = milp.solve()
    assert status == OPTIMAL
    assert obj == pytest.approx(10.0, abs=1e-6)


def test_doubleopt_minimize(curr_solver):
    model = _make_chain_model()
    mod = sd.SDModule(
        model,
        DOUBLEOPT,
        inner_objective="r_in",
        inner_opt_sense=MINIMIZE,
        outer_objective="r_p",
        outer_opt_sense=MINIMIZE,
        constraints="r_p <= 3",
    )
    sol = sd.compute_strain_designs(model, sd_modules=[mod], ko_cost={}, solver=curr_solver, compress=False)
    assert sol.status == OPTIMAL
    assert sol.reaction_sd == [{}]


def test_protect_outer_minimize(curr_solver):
    model = _make_branch_model(in_bounds=(1.0, 6.0), b_bounds=(0.0, 5.0), c_bounds=(0.0, 5.0))
    mod = sd.SDModule(
        model,
        PROTECT,
        inner_objective="r_in",
        inner_opt_sense=MAXIMIZE,
        outer_objective="r_b",
        outer_opt_sense=MINIMIZE,
    )
    milp = sd.SDMILP(model, [mod], ko_cost={}, solver=curr_solver)
    x, _, status = milp.solve()
    assert status == OPTIMAL
    assert x[1] == pytest.approx(1.0, abs=1e-6)
