"""The LP engine (engine='lp'): designs enumerated with LPs over the model, genes kept out of it.

Every problem is solved by the MILP path and by the LP engine, and the two have to agree: on the
set of designs for the MCS problems, on the optimal value for the optimisation problems.
"""
from .test_01_load_models_and_solvers import model_small_example, model_weak_coupling, model_gpr
import numpy as np
import pytest
import straindesign as sd
from straindesign.names import *
from straindesign.lp_enumeration import BoolLayer, BoundFacts, _blocking_sets, build_problem, gpr_blocked_formula, bits, mask


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


def test_reaction_target_with_rule_is_blocked_either_way():
    """A reaction that can be knocked out itself and has a rule over knockable genes is blocked
    by its own knockout and by the genes, as is a target without rule that the compression lumps
    with a reaction that has one."""
    import cobra
    from straindesign.lp_enumeration import _atoms_and_layer
    model = cobra.Model('either_way')
    a, b = cobra.Metabolite('A', compartment='c'), cobra.Metabolite('B', compartment='c')
    up, conv, ex = cobra.Reaction('up', upper_bound=10), cobra.Reaction('conv', lower_bound=2, upper_bound=10), cobra.Reaction('ex', upper_bound=10)
    up.add_metabolites({a: 1})
    conv.add_metabolites({a: -1, b: 1})
    ex.add_metabolites({b: -1})
    model.add_reactions([up, conv, ex])
    conv.gene_reaction_rule = 'g1 and g2'
    names, layer, atom_rxn = _atoms_and_layer(model, {'conv': 2.0, 'ex': 1.0}, {'g1': 1.0, 'g2': 1.0}, 3, 'id')
    assert names == ['g1', 'g2', 'conv', 'ex'] and atom_rxn == [1, 2]
    assert sorted(layer.block[0]) == [mask([0]), mask([1]), mask([2])] and layer.block[1] == [mask([3])]
    # 'up' has no rule and is a target; the compression lumps it with 'conv', whose genes then block it too
    modules = [sd.SDModule(model, SUPPRESS, constraints=['ex >= 1'])]
    for compress in (True, False):
        sol = sd.compute_strain_designs(model, sd_modules=modules, solution_approach=POPULATE, max_cost=2, solver=GLPK, engine='lp', gene_kos=True,
                                        ko_cost={'up': 2.0}, compress=compress)
        assert _sets(sol, True) == [['g1'], ['g2'], ['up']]


def test_bound_facts_keep_the_best_per_set():
    facts = BoundFacts(8)
    facts.add_lower(mask([0, 1]), 2.0)
    facts.add_lower(mask([0, 1]), 3.0)
    facts.add_lower(mask([2]), 5.0)
    facts.add_upper(mask([4]), 4.0)
    facts.add_upper(0, 9.0)
    assert facts.bounds(0) == (5.0, mask([2]), 9.0, 0)
    assert facts.bounds(mask([2, 4])) == (3.0, mask([0, 1]), 4.0, mask([4]))
    assert facts.bounds(mask([0, 2]))[0] == -np.inf
    facts.clear_lower()
    assert facts.bounds(mask([4]))[::2] == (-np.inf, 4.0)


@pytest.mark.parametrize('exact', [False, True])
def test_an_optimum_carries_over_to_other_blocked_sets(curr_solver, model_weak_coupling, exact):
    """The flux of an optimum is a lower bound wherever its atoms are spared, and the blocked atoms
    with a reduced cost of the right sign carry the upper bound on their own."""
    if exact and curr_solver != GLPK:
        pytest.skip('the exact backend is one')
    modules = [sd.SDModule(model_weak_coupling, SUPPRESS, inner_objective="r_BM", constraints=["r_P - 0.4 r_S <= 0", "r_S >= 0.1"])]
    ko_cost = {r.id: 1.0 for r in model_weak_coupling.reactions}
    _, layer, mods, _, _ = build_problem(model_weak_coupling, modules, ko_cost, None, 4, curr_solver, exact=exact)
    R, everything = mods[0].inner, (1 << layer.nA) - 1
    rng, n_cores = np.random.default_rng(1), 0
    for _ in range(40):
        X = mask(a for a in range(layer.nA) if rng.random() < 0.3)
        val, W, C = R.maximize(0, X)
        if val is None:
            continue
        assert not W & X and C & X == C
        assert R.maximize(0, C)[0] <= val + 1e-7  # the core alone holds the optimum down
        assert R.maximize(0, everything & ~W)[0] >= val - 1e-7  # the flux survives whatever else is blocked
        n_cores += C != X
    assert n_cores  # the cores are smaller than the blocked sets they come from


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


def test_exact_engine_proves_that_nothing_beats_the_best(model_weak_coupling):
    """The last pass of an optimisation problem ends with proofs that no design beats the best one
    by more than a millionth: proofs with large multipliers, on thresholds that are no integers
    and that GLPK's exact simplex therefore rounds. They have to be accepted as the proofs they are."""
    modules = [sd.SDModule(model_weak_coupling, OPTKNOCK, outer_objective='r_P', inner_objective='r_BM', constraints='r_BM >= 1')]
    lp = sd.compute_strain_designs(model_weak_coupling, sd_modules=modules, solution_approach=BEST, max_cost=4, solver=GLPK, engine='lp_exact')
    assert lp.status == OPTIMAL and len(lp.reaction_sd) == 1
    assert abs(lp.objective_value - 5.0) < 1e-5


def test_robustknock_constraint_holds_at_the_guaranteed_value(curr_solver, model_weak_coupling):
    """A constraint on the product is met only by the designs that guarantee it, as in the MILP
    formulation: the best guarantee on this network is 5."""

    def run(constraints):
        modules = [sd.SDModule(model_weak_coupling, ROBUSTKNOCK, outer_objective='r_P', inner_objective='r_BM', constraints=constraints)]
        return sd.compute_strain_designs(model_weak_coupling, sd_modules=modules, solution_approach=BEST, max_cost=4, solver=curr_solver, engine='lp')

    met = run(['r_P >= 4', 'r_BM >= 0.5'])
    assert met.status == OPTIMAL and abs(met.objective_value - 5.0) < 1e-5
    asked_too_much = run(['r_P >= 6'])
    assert asked_too_much.status == INFEASIBLE and not asked_too_much.reaction_sd
    # a cap on the product caps the value
    assert abs(run(['r_P <= 3']).objective_value - 3.0) < 1e-5


def _signed(sol):
    return sorted(sorted((k, v) for k, v in d.items() if v != 0) for d in sol.reaction_sd)


def test_knock_ins(curr_solver, model_small_example):
    """A reaction with a knock-in cost is absent unless it is knocked in; designs mix both kinds."""
    modules = [
        sd.SDModule(model_small_example, SUPPRESS, constraints=["1.0 R3 - 0.5 R1 - 0.5 R2 <= 0.0 ", "1.0 R2 >= 0.0 ", "1.0 R1 >= 0.1 "]),
        sd.SDModule(model_small_example, PROTECT, constraints=["1.0 R3 >= 1.0 "])
    ]
    ko_cost = {r.id: 1.0 for r in model_small_example.reactions if r.id != 'R9'}
    for compress in (True, False):
        ref, lp = _both(model_small_example, modules, curr_solver, max_cost=4, ko_cost=ko_cost, ki_cost={'R9': 1.0}, compress=compress)
        assert _signed(lp) == _signed(ref)
        assert any(d.get('R9') == 1 for d in lp.reaction_sd) and len(lp.reaction_sd) > 1


def test_knock_in_sets_of_the_layer():
    layer = BoolLayer([1.0, 2.0, 1.0], [0, False, False], 3, needs=[-1, 1, 2])
    assert layer.ki == mask([1, 2])
    assert layer.X(0) == mask([1, 2]) and layer.X(mask([0, 2])) == mask([0, 1])
    assert layer.knock_in_sets(3) == [(0.0, 0), (1.0, mask([2])), (2.0, mask([1])), (3.0, mask([1, 2]))]
    assert layer.knock_in_sets(1) == [(0.0, 0), (1.0, mask([2]))]


def test_unsupported_setups_say_so(model_gpr):
    modules = [sd.SDModule(model_gpr, SUPPRESS, constraints=["1.0 rd_ex >= 1.0 "])]
    genes = [g.name or g.id for g in model_gpr.genes]
    with pytest.raises(NotImplementedError):
        sd.compute_strain_designs(model_gpr, sd_modules=modules, solution_approach=POPULATE, max_cost=3, solver=GLPK, engine='lp', gene_kos=True,
                                  gko_cost={g: 1.0 for g in genes[1:]}, gki_cost={genes[0]: 1.0})
