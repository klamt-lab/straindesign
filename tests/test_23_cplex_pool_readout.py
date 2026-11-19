"""CPLEX solution pool: read on chosen columns, one entry per distinct assignment, in bounded chunks."""
import pytest
from math import inf
from scipy import sparse
import straindesign as sd
from straindesign import MILP_LP
from straindesign.names import CPLEX, OPTIMAL

pytestmark = pytest.mark.skipif(CPLEX not in sd.avail_solvers, reason='needs CPLEX')


def _three_free_binaries():
    return MILP_LP(c=[0.0, 0.0, 0.0], A_ineq=sparse.csr_matrix([[1.0, 1.0, 1.0]]), b_ineq=[3.0], lb=[0.0] * 3, ub=[1.0] * 3, vtype='BBB', solver=CPLEX)


def test_pool_is_read_in_full_by_default():
    milp = _three_free_binaries()
    x, _, status = milp.populate(inf)
    assert status == OPTIMAL
    assert sorted(tuple(round(v) for v in row) for row in x) == sorted((a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1))


def test_pool_columns_give_one_entry_per_assignment():
    """Eight pool entries differ in the two columns that are not asked for; two remain."""
    milp = _three_free_binaries()
    milp.backend._pool_columns = [0]
    x, _, status = milp.populate(inf)
    assert status == OPTIMAL
    assert sorted(round(row[0]) for row in x) == [0, 1]
    assert all(len(row) == 1 for row in x)


def test_a_full_chunk_is_not_an_exhausted_pool(monkeypatch):
    import straindesign.cplex_interface as ci
    monkeypatch.setattr(ci, '_POOL_CHUNK', 3)
    milp = _three_free_binaries()
    x, _, status = milp.populate(inf)
    assert status == OPTIMAL
    assert len(x) == 3
    assert not milp.backend._pool_exhausted
