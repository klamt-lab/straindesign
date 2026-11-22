"""The LP engine (engine='lp'): designs enumerated with LPs over the model, genes kept out of it.

Every problem is solved by the MILP path and by the LP engine, and the two have to agree: on the
set of designs for the MCS problems, on the optimal value for the optimisation problems.
"""
from .test_01_load_models_and_solvers import model_small_example, model_weak_coupling, model_gpr
import numpy as np
import pytest
import straindesign as sd
from straindesign.names import *
from straindesign.lp_enumeration import BoolLayer, _blocking_sets, gpr_blocked_formula, bits, mask


def _sets(sol, genes=False):
    if sol.is_lazy:
        sol.expand_all()
    sds = sol.gene_sd if genes else sol.reaction_sd
    return sorted(sorted(k for k, v in d.items() if v != 0) for d in sds)


def _both(model, modules, solver, engine='lp', genes=False, **kw):
    kw = dict(sd_modules=modules, solution_approach=POPULATE, solver=solver, gene_kos=genes, **kw)
    return sd.compute_strain_designs(model, **kw), sd.compute_strain_designs(model, engine=engine, **kw)


def test_blocking_sets_follow_the_rule():
    index = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    cost = [1.0] * 4
    # isozymes: both have to go; a complex: any subunit does
    assert _blocking_sets(gpr_blocked_formula('a or b', index), cost, 3) == [mask([0, 1])]
    assert sorted(_blocking_sets(gpr_blocked_formula('a and b', index), cost, 3)) == [1, 2]
    # (a and b) or (c and d): one subunit of each complex
    sets = _blocking_sets(gpr_blocked_formula('(a and b) or (c and d)', index), cost, 3)
    assert sorted(sorted(bits(J)) for J in sets) == [[0, 2], [0, 3], [1, 2], [1, 3]]
    # the budget cuts the list short, and a gene that cannot be knocked out protects its isozyme partner
    assert _blocking_sets(gpr_blocked_formula('a or b or c', index), cost, 2) == []
    assert gpr_blocked_formula('a or x', index) is False
    assert gpr_blocked_formula('a and x', index) == 0


def test_layer_extends_blocked_atoms():
    layer = BoolLayer([1.0, 1.0, 1.0], [0, ('and', [0, 1]), ('or', [1, 2])], 3)
    assert layer.X(mask([0])) == mask([0])
    assert layer.extend(mask([0]), mask([0]), mask([1])) == mask([0, 1, 2])
    assert layer.singles(1, mask([0])) == mask([1])
    assert layer.completions(1, 0, 3) == [mask([0, 1])]


def test_mcs_suppress(curr_solver, model_small_example):
    modules = [sd.SDModule(model_small_example, SUPPRESS, constraints=["R3 - 0.5 R1 <= 0.0", "R2 <= 0", "R1 >= 0.1"])]
    ref, lp = _both(model_small_example, modules, curr_solver, max_cost=4)
    assert lp.status == OPTIMAL
    assert _sets(lp) == _sets(ref) and len(_sets(lp)) > 1


def test_mcs_suppress_protect(curr_solver, model_small_example):
    modules = [
        sd.SDModule(model_small_example, SUPPRESS, constraints=["1.0 R3 - 0.5 R1 - 0.5 R2 <= 0.0 ", "1.0 R2 >= 0.0 ", "1.0 R1 >= 0.1 "]),
        sd.SDModule(model_small_example, PROTECT, constraints=["1.0 R3 >= 1.0 "])
    ]
    ref, lp = _both(model_small_example, modules, curr_solver, max_cost=4)
    assert _sets(lp) == _sets(ref) and len(_sets(lp)) > 1


def test_mcs_genes(curr_solver, model_gpr):
    modules = [sd.SDModule(model_gpr, SUPPRESS, constraints=["1.0 rd_ex >= 1.0 "]), sd.SDModule(model_gpr, PROTECT, constraints=[[{'r_bm': 1.0}, '>=', 1.0]])]
    ref, lp = _both(model_gpr, modules, curr_solver, genes=True, max_cost=4)
    assert _sets(lp, True) == _sets(ref, True) and len(_sets(lp, True)) > 1


def test_mcs_inner_objective(curr_solver, model_weak_coupling):
    modules = [
        sd.SDModule(model_weak_coupling, SUPPRESS, inner_objective="r_BM", constraints=["r_P - 0.4 r_S <= 0", "r_S >= 0.1"]),
        sd.SDModule(model_weak_coupling, PROTECT, constraints=["r_BM >= 0.2"])
    ]
    ref, lp = _both(model_weak_coupling, modules, curr_solver, max_cost=4)
    assert _sets(lp) == _sets(ref) and len(_sets(lp)) > 1


def test_exact_engine_matches(model_small_example, model_weak_coupling):
    modules = [sd.SDModule(model_small_example, SUPPRESS, constraints=["R3 - 0.5 R1 <= 0.0", "R2 <= 0", "R1 >= 0.1"])]
    ref, lp = _both(model_small_example, modules, GLPK, engine='lp_exact', max_cost=4)
    assert _sets(lp) == _sets(ref)
    modules = [
        sd.SDModule(model_weak_coupling, SUPPRESS, inner_objective="r_BM", constraints=["r_P - 0.4 r_S <= 0", "r_S >= 0.1"]),
        sd.SDModule(model_weak_coupling, PROTECT, constraints=["r_BM >= 0.2"])
    ]
    ref, lp = _both(model_weak_coupling, modules, GLPK, engine='lp_exact', max_cost=4)
    assert _sets(lp) == _sets(ref)


@pytest.mark.parametrize('module, value', [
    (dict(module_type=OPTKNOCK, outer_objective='r_P', inner_objective='r_BM', constraints='r_BM >= 1'), 5.0),
    (dict(module_type=ROBUSTKNOCK, outer_objective='r_P', inner_objective='r_BM', constraints='r_BM >= 1'), 5.0),
    (dict(module_type=OPTCOUPLE, prod_id='r_P', inner_objective='r_BM', min_gcp=1.0), 5.0),
])
def test_optimisation_modules(curr_solver, model_weak_coupling, module, value):
    modules = [sd.SDModule(model_weak_coupling, **module)]
    lp = sd.compute_strain_designs(model_weak_coupling, sd_modules=modules, solution_approach=BEST, max_cost=4, solver=curr_solver, engine='lp')
    assert lp.status == OPTIMAL and len(lp.reaction_sd) == 1
    assert abs(lp.objective_value - value) < 1e-5


def test_unsupported_setups_say_so(model_small_example):
    modules = [sd.SDModule(model_small_example, SUPPRESS, constraints=["R4 >= 1"])]
    with pytest.raises(NotImplementedError):
        sd.compute_strain_designs(model_small_example, sd_modules=modules, solution_approach=POPULATE, max_cost=3, solver=GLPK, engine='lp',
                                  ko_cost={'R1': 1.0, 'R2': 1.0}, ki_cost={'R3': 1.0})
