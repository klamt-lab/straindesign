"""Elementary flux vectors as minimal knock-in sets (see the EFV chapter of the documentation)."""
from .test_01_load_models_and_solvers import *
import straindesign as sd

# All support-minimal flux vectors of model_small_example with R3 >= 1, obtained independently by
# enumerating reaction subsets with LPs; the same six are the EFMs of the network's flux cone with R3 > 0.
EFV_SUPPORTS_R3 = {
    frozenset({'R2', 'R3', 'R9'}),
    frozenset({'R1', 'R3', 'R5', 'R9'}),
    frozenset({'R1', 'R3', 'R6', 'R8', 'R9'}),
    frozenset({'R1', 'R3', 'R4', 'R6', 'R7', 'R10'}),
    frozenset({'R1', 'R2', 'R3', 'R4', 'R7', 'R8', 'R10'}),
    frozenset({'R1', 'R3', 'R4', 'R5', 'R7', 'R8', 'R10'}),
}


def _efv_designs(model, solver, approach, compress, **kwargs):
    return sd.compute_strain_designs(model,
                                     sd_modules=[sd.SDModule(model, PROTECT, constraints='R3 >= 1')],
                                     ki_cost={r.id: 1 for r in model.reactions},
                                     solution_approach=approach,
                                     solver=solver,
                                     compress=compress,
                                     **kwargs)


@pytest.mark.timeout(15)
@pytest.mark.parametrize("compress", [True, False])
def test_efv_shortest(curr_solver, model_small_example, compress):
    """BEST with one solution returns a single EFV of minimal support size."""
    sols = _efv_designs(model_small_example, curr_solver, BEST, compress, max_solutions=1)
    assert sols.status == OPTIMAL
    assert [frozenset(s) for s in sols.get_reaction_sd()] == [frozenset({'R2', 'R3', 'R9'})]
    assert sols.sd_cost == [3.0]


@pytest.mark.timeout(15)
@pytest.mark.parametrize("compress", [True, False])
def test_efv_enumeration(curr_solver, model_small_example, comp_approach_best_populate, compress):
    """Enumerating all minimal knock-in sets yields exactly the EFV supports, each with a flux vector."""
    sols = _efv_designs(model_small_example, curr_solver, comp_approach_best_populate, compress)
    assert sols.status == OPTIMAL
    supports = {frozenset(s) for s in sols.get_reaction_sd()}
    assert supports == EFV_SUPPORTS_R3
    assert all(c == len(s) for c, s in zip(sols.sd_cost, sols.get_reaction_sd()))
    for support in supports:
        with model_small_example as m:
            for r in m.reactions:
                if r.id not in support:
                    r.bounds = (0.0, 0.0)
            flux = sd.fba(m, constraints='R3 = 1', solver=curr_solver).fluxes
        assert {r for r, v in flux.items() if abs(v) > 1e-9} == support


@pytest.mark.timeout(15)
def test_efv_max_cost(curr_solver, model_small_example):
    """max_cost caps the support size."""
    sols = _efv_designs(model_small_example, curr_solver, BEST, True, max_cost=5)
    assert {frozenset(s) for s in sols.get_reaction_sd()} == {s for s in EFV_SUPPORTS_R3 if len(s) <= 5}
