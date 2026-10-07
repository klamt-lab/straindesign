"""An enumeration reports OPTIMAL only when it is proven complete."""
import numpy as np
from scipy import sparse
import pytest
import straindesign as sd
from straindesign.names import *
from straindesign import strainDesignMILP as sdm
from straindesign.strainDesignMILP import _drop_non_minimal
from ._models import load_test_model


def test_final_status_does_not_promote_a_failed_enumeration():
    """A solver that failed mid-enumeration leaves a truncated pool, which must not be reported as
    OPTIMAL. Only exhaustion and the time limit may be rewritten."""
    from straindesign.compute_strain_designs import _final_status

    assert _final_status(INFEASIBLE, True) == OPTIMAL
    assert _final_status(TIME_LIMIT, True) == TIME_LIMIT_W_SOL
    assert _final_status(ERROR, True) == ERROR
    assert _final_status(ERROR, False) == ERROR
    assert _final_status(INFEASIBLE, False) == INFEASIBLE
    assert _final_status(OPTIMAL, True) == OPTIMAL


def test_supersets_are_dropped_and_counted():
    rows = [[1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 0], [0, 0, 0, 1], [1, 1, 0, 1]]
    z = sparse.csr_matrix(np.array(rows))
    kept, dropped = _drop_non_minimal(z)
    assert dropped == 2                                     # {0,1,2} > {0,1}; {0,1,3} > {0,1} and > {3}
    assert sorted(frozenset(kept[i].indices.tolist()) for i in range(kept.shape[0])) == \
        sorted([frozenset({0, 1}), frozenset({0, 2}), frozenset({3})])


def test_minimal_set_is_untouched():
    z = sparse.csr_matrix(np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]]))
    kept, dropped = _drop_non_minimal(z)
    assert dropped == 0 and kept.shape[0] == 3


@pytest.mark.parametrize('solver', [s for s in (CPLEX, GUROBI) if s in sd.avail_solvers])
def test_time_limit_at_a_level_is_not_exhaustion(solver, monkeypatch):
    """A populate that stopped without a proof either way must not end the enumeration as OPTIMAL."""
    model = load_test_model('e_coli_core')
    original = sdm.SDMILP.populateZ
    calls = {'n': 0}

    def populate_then_stall(self, n):
        calls['n'] += 1
        if calls['n'] == 2:
            # what a populate that stopped without a proof returns: no designs, TIME_LIMIT
            return sparse.csr_matrix((0, self.num_z)), TIME_LIMIT
        return original(self, n)

    monkeypatch.setattr(sdm.SDMILP, 'populateZ', populate_then_stall)
    modules = [sd.SDModule(model, SUPPRESS, constraints=["BIOMASS_Ecoli_core_w_GAM >= 0.1"])]
    sol = sd.compute_strain_designs(model, sd_modules=modules, max_cost=3, solver=solver,
                                    solution_approach=POPULATE)
    assert calls['n'] >= 2
    assert sol.status != OPTIMAL
