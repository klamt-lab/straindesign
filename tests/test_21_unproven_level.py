"""A cost level the solver neither exhausted nor proved empty must not count as exhausted."""
from .test_01_load_models_and_solvers import *
import straindesign as sd
from straindesign import strainDesignMILP as sdm
from ._models import load_test_model


@pytest.mark.parametrize('solver', [s for s in (CPLEX, GUROBI) if s in sd.avail_solvers])
def test_time_limit_at_a_level_is_not_exhaustion(solver, monkeypatch):
    model = load_test_model('e_coli_core')
    original = sdm.SDMILP.populateZ
    calls = {'n': 0}

    def populate_then_stall(self, n):
        calls['n'] += 1
        if calls['n'] == 2:
            # what a populate that stopped without a proof returns: no designs, TIME_LIMIT
            return sdm.sparse.csr_matrix((0, self.num_z)), TIME_LIMIT
        return original(self, n)

    monkeypatch.setattr(sdm.SDMILP, 'populateZ', populate_then_stall)
    modules = [sd.SDModule(model, SUPPRESS, constraints=["BIOMASS_Ecoli_core_w_GAM >= 0.1"])]
    sol = sd.compute_strain_designs(model, sd_modules=modules, max_cost=3, solver=solver,
                                    solution_approach=POPULATE)
    assert calls['n'] >= 2
    assert sol.status != OPTIMAL
