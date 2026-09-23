import pytest
import straindesign as sd
from straindesign.names import *
from numpy import inf
from scipy import sparse


@pytest.mark.timeout(60)
def test_open_pool_gap_returns_every_pool_entry(curr_solver):
    """A pinned cost level with a tie-breaking objective has one solution per pool entry, all
    of them wanted. The read-out must not re-close an open gap by keeping only the entries whose
    objective equals the incumbent's exactly."""
    if curr_solver != GUROBI:
        pytest.skip("the exact-equality read-out was specific to the Gurobi backend")
    # three binaries, exactly one on; distinct tiny costs stand in for the tilt
    milp = sd.MILP_LP(c=[1e-6, 2e-6, 3e-6],
                      A_ineq=sparse.csr_matrix((0, 3)), b_ineq=[],
                      A_eq=sparse.csr_matrix([[1.0, 1.0, 1.0]]), b_eq=[1.0],
                      lb=[0.0, 0.0, 0.0], ub=[1.0, 1.0, 1.0], vtype='BBB', solver=curr_solver)
    milp.backend.set_pool_gap(True)
    x, _, status = milp.populate(inf)
    assert status == OPTIMAL
    assert len(x) == 3, "open gap: every solution at the pinned level is wanted"

    milp.backend.set_pool_gap(False)
    x, _, status = milp.populate(inf)
    assert status == OPTIMAL
    assert len(x) == 1, "closed gap: only the optimum"
