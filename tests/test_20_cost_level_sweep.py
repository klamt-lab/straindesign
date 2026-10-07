"""The cost-level sweep: when it runs, when it falls back, and what it leaves behind."""
import pytest
import straindesign as sd
from straindesign.names import *
from numpy import inf
from .test_01_load_models_and_solvers import model_small_example, model_weak_coupling
from .test_05_straindesign import _two_route_network, _designs


def _pool_gap(milp):
    """The backend's absolute solution-pool optimality gap, whatever the solver calls it."""
    backend = milp.backend
    if hasattr(backend, 'parameters'):  # cplex
        return backend.parameters.mip.pool.absgap.get()
    return backend.params.PoolGapAbs  # gurobi


@pytest.mark.parametrize('ko_cost,pinned', [
    ({'R1': 1.0, 'R2': 1.0, 'R3': 1.0, 'R4': 1.0}, True),
    ({'R1': 1.0, 'R2': 1.1, 'R3': 1.0, 'R4': 1.0}, False),
])
@pytest.mark.timeout(60)
def test_pool_gap_is_open_only_while_a_cost_level_is_pinned(monkeypatch, ko_cost, pinned):
    """``enumerate`` has no cost level pinned, so it needs the solution pool's optimality gap
    CLOSED: a closed gap is what makes populate hand back only the cheapest designs, which is in
    turn the only reason a design emitted before its own subset cannot happen -- validity is all
    ``verify_sd`` checks, minimality comes from the ascending-cost order plus the exclusion of
    every superset. ``enumerate_ksweep`` pins the cost to one value per level, and only there does
    the gap stop filtering anything that is wanted.

    The gap used to be opened in the backend constructor, so it was open for the plain
    ``enumerate`` path as well -- including the k-sweep's own fallback, which any non-integer
    intervention cost triggers.
    """
    solver = next((s for s in [CPLEX, GUROBI] if s in sd.avail_solvers), None)
    if solver is None:
        pytest.skip('a solution pool is implemented for cplex and gurobi only')

    from straindesign.strainDesignMILP import SDMILP
    seen = []
    original = SDMILP.populateZ

    def recording_populateZ(self, n):
        seen.append(_pool_gap(self))
        return original(self, n)

    monkeypatch.setattr(SDMILP, 'populateZ', recording_populateZ)

    model = _two_route_network()
    sd.compute_strain_designs(model,
                              sd_modules=[sd.SDModule(model, PROTECT, constraints=['R4 >= 1'])],
                              max_cost=3,
                              ki_cost=ko_cost,
                              solution_approach='populate',
                              solver=solver,
                              milp_threads=1,
                              compress=False)
    assert seen, 'populate was never called, so the test asserts nothing'
    if pinned:
        assert all(g > 0 for g in seen), \
            'the k-sweep pins the level, so it should have opened the gap there: ' + str(seen)
    else:
        assert all(g == 0 for g in seen), \
            'a non-integer cost falls back to plain enumerate, whose ascending-cost order needs ' \
            'the pool gap closed; it was open: ' + str(seen)


@pytest.mark.timeout(120)
def test_designs_stay_minimal_when_the_k_sweep_falls_back_under_an_open_pool(monkeypatch, model_weak_coupling):
    """Minimal cut sets form an antichain: no design may contain another.

    This is the behaviour the gap bug broke -- a superset emitted before the subset that excludes
    it is valid, so ``verify_sd`` passes it and it is reported as a design. The costs here are the
    ones test_mcs_opt uses, and they are non-integer, so the k-sweep falls back to ``enumerate``.
    """
    solver = next((s for s in [CPLEX, GUROBI] if s in sd.avail_solvers), None)
    if solver is None:
        pytest.skip('a solution pool is implemented for cplex and gurobi only')
    modules = [sd.SDModule(model_weak_coupling, SUPPRESS, inner_objective="r_BM", constraints=["r_P - 0.4 r_S <= 0", "r_S >= 0.1"])]
    modules += [sd.SDModule(model_weak_coupling, PROTECT, constraints=["r_BM >= 0.2"])]
    kocost = {'r1': 1, 'r2': 1, 'r4': 1.1, 'r5': 0.75, 'r7': 0.8, 'r8': 1, 'r9': 1, 'r_S': 1.0, 'r_P': 1, 'r_BM': 1, 'r_Q': 1.5}
    solution = sd.compute_strain_designs(model_weak_coupling,
                                         sd_modules=modules,
                                         max_cost=4,
                                         max_solutions=inf,
                                         solution_approach='populate',
                                         ko_cost=kocost,
                                         ki_cost={'r3': 0.6, 'r6': 1.0},
                                         reg_cost={'r6 >= 4.5': 1.2},
                                         solver=solver,
                                         milp_threads=1,
                                         compress=False)
    designs = [frozenset(k for k, v in d.items() if v != 0) for d in solution.get_reaction_sd()]
    for a in designs:
        for b in designs:
            assert not (a < b), 'design %s contains %s, so it is not minimal' % (sorted(b), sorted(a))


@pytest.mark.parametrize('approach', ['any', 'best'])
@pytest.mark.timeout(120)
def test_sos1_gate_rewrite_keeps_the_objective_row_full_width(monkeypatch, model_small_example, approach):
    """The SOS1 gate rewrite appends slack columns to the matrix after ``c_bu`` was stored, so an
    objective vector of the original width no longer spans a row. ``any`` and ``best`` re-write the
    objective row through fixObjective and raised ``ValueError: shape mismatch in assignment``
    before the solver was ever called; ``populate`` never takes that path, which is why only these
    two approaches broke. The designs asserted here are the same ones test_mcs expects."""
    solver = next((s for s in [CPLEX, GUROBI] if s in sd.avail_solvers), None)
    if solver is None:
        pytest.skip('the SOS1 gate rewrite is implemented for cplex and gurobi only')
    modules = [sd.SDModule(model_small_example, SUPPRESS, constraints=["R3 - 0.5 R1 <= 0.0", "R2 <= 0", "R1 >= 0.1"])]
    modules += [
        sd.SDModule(model_small_example, SUPPRESS, constraints=["1.0 R3 - 0.5 R1 - 0.5 R2 <= 0.0 ", "1.0 R2 >= 0.0 ", "1.0 R1 >= 0.1 "])
    ]
    modules += [sd.SDModule(model_small_example, PROTECT, constraints=["1.0 R3 >= 1.0 "])]
    sols = sd.compute_strain_designs(model_small_example,
                                     sd_modules=modules,
                                     max_cost=inf,
                                     max_solutions=inf,
                                     solution_approach=approach,
                                     ki_cost={'R2': 1},
                                     solver=solver,
                                     milp_threads=1,
                                     compress=False).get_reaction_sd()
    assert ({'R1': -1.0, 'R2': 1.0} in sols)
    assert ({'R6': -1.0, 'R8': -1.0} in sols)
    assert ({'R4': -1.0} in sols)
    assert ({'R7': -1.0} in sols)
    assert ({'R10': -1.0} in sols)


@pytest.mark.parametrize('ki_cost,max_cost,expected', [
    ({'R1': 1.0, 'R2': -5.0, 'R3': 1.0, 'R4': 1.0}, 3, ['R1', 'R2', 'R4']),
    ({'R1': 1.0, 'R2': -100.0, 'R3': 1.0, 'R4': 1.0}, -97, ['R1', 'R2', 'R4']),
])
@pytest.mark.timeout(60)
def test_ksweep_falls_back_when_an_intervention_is_not_positively_priced(monkeypatch, curr_solver, ki_cost, max_cost,
                                                                        expected):
    """The k-sweep walks cost levels 1, 2, ... upward and asks for each level in turn. A design
    whose total cost is zero or negative -- which one rewarding intervention is enough to produce
    -- lies below every level the sweep visits, so the sweep returns none of them and the
    enumeration silently comes back short. Only a budget whose costs are all positive can be
    swept, so anything else has to fall back to plain populate.

    Every benchmark prices interventions at 1, which is why this never showed up there."""
    model = _two_route_network()
    sol = sd.compute_strain_designs(model,
                                    sd_modules=[sd.SDModule(model, PROTECT, constraints=['R4 >= 1'])],
                                    max_cost=max_cost,
                                    ki_cost=ki_cost,
                                    solution_approach='populate',
                                    solver=curr_solver,
                                    compress=False)
    designs = _designs(sol)
    assert designs, 'the k-sweep returned no design at all; it swept levels the design sits below'
    assert expected in designs, designs
