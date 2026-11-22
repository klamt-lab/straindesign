"""The LP engine's 'any' mode (engine='lp', solution_approach='any'): single designs by dives, each
polished to a local optimum.

Every returned design is checked with plain LPs on the original model, written without the
engine: the interventions are knocked out in the cobra model and each module's region is tested.
"""
from .test_01_load_models_and_solvers import model_small_example, model_weak_coupling, model_gpr, model_core
import itertools
import numpy as np
import pytest
import straindesign as sd
from straindesign.names import *

TOL = 1e-7


def _expr(model, d):
    return sum(c * model.reactions.get_by_id(r).flux_expression for r, c in d.items())


def _max(model, obj, cons, ko, genes):
    """max obj . v over the model with the interventions ko knocked out and the rows cons; None if empty."""
    with model:
        names = {g.name: g for g in model.genes}
        for k in ko:
            (names[k] if genes else model.reactions.get_by_id(k)).knock_out()
        for d, sense, rhs in cons:
            e = _expr(model, d)
            lo, hi = {'<=': (None, rhs), '>=': (rhs, None), '=': (rhs, rhs)}[sense]
            model.add_cons_vars(model.problem.Constraint(e, lb=lo, ub=hi))
        model.objective = model.problem.Objective(_expr(model, obj) if obj else 0, direction='max')
        val = model.slim_optimize(error_value=np.nan)
    return None if np.isnan(val) else val


def _region_empty(model, cons, ko, genes, inner=None):
    if inner is None:
        return _max(model, {}, cons, ko, genes) is None
    opt = _max(model, inner, [], ko, genes)
    if opt is None:
        return True
    return _max(model, {}, cons + [(inner, '>=', opt - TOL * max(1.0, abs(opt)))], ko, genes) is None


def _fulfils(model, spec, ko, genes):
    """spec: list of (kind, constraints, inner objective or None)."""
    return all(_region_empty(model, cons, ko, genes, inner) == (kind == SUPPRESS) for kind, cons, inner in spec)


def _modules(model, spec):
    return [sd.SDModule(model, kind, constraints=cons, inner_objective=inner) for kind, cons, inner in spec]


def _designs(sol, genes):
    sds = sol.gene_sd if genes else sol.reaction_sd
    return [frozenset(k for k, v in d.items() if v != 0) for d in sds]


def _run(model, modules, solver, genes=False, **kw):
    return sd.compute_strain_designs(model, sd_modules=modules, solution_approach=ANY, solver=solver, gene_kos=genes, engine='lp', seed=3, **kw)


def _check_valid_minimal(model, spec, designs, genes):
    for d in designs:
        assert _fulfils(model, spec, d, genes), sorted(d)
        for k in d:
            assert not _fulfils(model, spec, d - {k}, genes), (sorted(d), k)


SPEC_S = [(SUPPRESS, [[{'R3': 1, 'R1': -0.5}, '<=', 0.0], [{'R2': 1}, '<=', 0], [{'R1': 1}, '>=', 0.1]], None)]
SPEC_SP = [(SUPPRESS, [[{'R3': 1, 'R1': -0.5, 'R2': -0.5}, '<=', 0.0], [{'R2': 1}, '>=', 0.0], [{'R1': 1}, '>=', 0.1]], None),
           (PROTECT, [[{'R3': 1}, '>=', 1.0]], None)]


def test_suppress(curr_solver, model_small_example):
    sol = _run(model_small_example, _modules(model_small_example, SPEC_S), curr_solver, max_cost=4, max_solutions=4)
    designs = _designs(sol, False)
    assert sol.status == OPTIMAL and len(set(designs)) >= 4
    _check_valid_minimal(model_small_example, SPEC_S, designs, False)
    # local optimum: no two knockouts of a design can be replaced by one, unless that leads into
    # (a superset of) another returned design
    rxns = [r.id for r in model_small_example.reactions]
    for d in designs:
        for pair in itertools.combinations(sorted(d), 2):
            for r in rxns:
                nb = (d - set(pair)) | {r}
                if r in d or any(o <= nb for o in designs if o != d):
                    continue
                assert not _fulfils(model_small_example, SPEC_S, nb, False), (sorted(d), pair, r)


def test_suppress_protect(curr_solver, model_small_example):
    sol = _run(model_small_example, _modules(model_small_example, SPEC_SP), curr_solver, max_cost=4, max_solutions=3)
    designs = _designs(sol, False)
    assert designs and sol.status in (OPTIMAL, TIME_LIMIT_W_SOL)
    _check_valid_minimal(model_small_example, SPEC_SP, designs, False)


def test_unbounded_budget(curr_solver, model_small_example):
    """No max_cost: the dive is not bounded by a budget, the polishing still gives minimal designs."""
    sol = _run(model_small_example, _modules(model_small_example, SPEC_S), curr_solver, max_solutions=2)
    designs = _designs(sol, False)
    assert len(designs) >= 2
    _check_valid_minimal(model_small_example, SPEC_S, designs, False)


def test_genes(curr_solver, model_gpr):
    spec = [(SUPPRESS, [[{'rd_ex': 1}, '>=', 1.0]], None), (PROTECT, [[{'r_bm': 1}, '>=', 1.0]], None)]
    sol = _run(model_gpr, _modules(model_gpr, spec), curr_solver, genes=True, max_cost=4, max_solutions=3)
    designs = _designs(sol, True)
    assert designs
    _check_valid_minimal(model_gpr, spec, designs, True)


def test_inner_objective(curr_solver, model_weak_coupling):
    spec = [(SUPPRESS, [[{'r_P': 1, 'r_S': -0.4}, '<=', 0], [{'r_S': 1}, '>=', 0.1]], {'r_BM': 1}),
            (PROTECT, [[{'r_BM': 1}, '>=', 0.2]], None)]
    sol = _run(model_weak_coupling, _modules(model_weak_coupling, spec), curr_solver, max_cost=4, max_solutions=3)
    designs = _designs(sol, False)
    assert designs
    _check_valid_minimal(model_weak_coupling, spec, designs, False)


def _optknock(m, ko):
    opt = _max(m, {'r_BM': 1}, [], ko, False)
    if opt is None:
        return None
    return _max(m, {'r_P': 1}, [[{'r_BM': 1}, '>=', 1.0], [{'r_BM': 1}, '>=', opt - TOL * max(1, opt)]], ko, False)


def _robustknock(m, ko):
    if _max(m, {}, [[{'r_BM': 1}, '>=', 1.0]], ko, False) is None:
        return None
    opt = _max(m, {'r_BM': 1}, [], ko, False)
    v = _max(m, {'r_P': -1}, [[{'r_BM': 1}, '>=', opt - TOL * max(1, opt)]], ko, False)
    return None if v is None else -v


def _optcouple(m, ko):
    opt = _max(m, {'r_BM': 1}, [], ko, False)
    if opt is None:
        return None
    g0 = _max(m, {'r_BM': 1}, [[{'r_P': 1}, '=', 0.0]], ko, False)
    if g0 is None:
        return None
    return opt - g0 if opt - g0 >= 1.0 - 1e-6 else None  # min_gcp 1


@pytest.mark.parametrize('module, value_of', [
    (dict(module_type=OPTKNOCK, outer_objective='r_P', inner_objective='r_BM', constraints='r_BM >= 1'), _optknock),
    (dict(module_type=ROBUSTKNOCK, outer_objective='r_P', inner_objective='r_BM', constraints='r_BM >= 1'), _robustknock),
    (dict(module_type=OPTCOUPLE, prod_id='r_P', inner_objective='r_BM', min_gcp=1.0), _optcouple),
])
def test_optimisation_local_optimum(curr_solver, model_weak_coupling, module, value_of):
    m = model_weak_coupling
    max_cost = 3
    sol = _run(m, [sd.SDModule(m, **module)], curr_solver, max_cost=max_cost, max_solutions=1)
    assert sol.status == OPTIMAL and len(sol.reaction_sd) >= 1
    d = _designs(sol, False)[0]
    val = value_of(m, d)
    assert val is not None and abs(val - sol.objective_value) <= 1e-5 * max(1.0, abs(val))
    # no neighbour (one knockout more, one less, one exchanged) within the budget does better
    rxns = [r.id for r in m.reactions]
    nbs = [d | {r} for r in rxns if r not in d and len(d) < max_cost] + [d - {k} for k in d]
    nbs += [(d - {k}) | {r} for k in d for r in rxns if r not in d]
    for nb in nbs:
        v = value_of(m, nb)
        assert v is None or v <= val + 1e-6 * max(1.0, abs(val)), (sorted(d), val, sorted(nb), v)


def test_imlcore_growth_coupling(model_core):
    """A design of several knockouts on a genome-scale core model, found without a
    budget: ethanol yield coupled to growth."""
    m = model_core
    bio = [r.id for r in m.reactions if r.objective_coefficient][0]
    spec = [(SUPPRESS, [[{'EX_etoh_e': 1, 'EX_glc__D_e': 0.5}, '<=', 0.0], [{bio: 1}, '>=', 0.05]], None),
            (PROTECT, [[{bio: 1}, '>=', 0.1]], None)]
    solver = GLPK if GLPK in sd.avail_solvers else sd.select_solver()
    sol = _run(m, _modules(m, spec), solver, max_solutions=1, time_limit=300)
    designs = _designs(sol, False)
    # one design of three knockouts, which the decompression spreads over the reactions lumped with them
    assert designs and all(len(d) >= 3 for d in designs)
    _check_valid_minimal(m, spec, designs, False)
