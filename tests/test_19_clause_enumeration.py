"""SCIP's populate inside the k-sweep: the clause enumeration against SCIP's own tree."""
import logging
from importlib.util import find_spec
import pytest
from .test_01_load_models_and_solvers import *
import straindesign as sd
import straindesign.clause_enumeration as clause_enumeration
import numpy as np
from numpy import inf
from straindesign.names import *

pytestmark = pytest.mark.skipif(find_spec("pyscipopt") is None, reason="needs SCIP")

KO_COST = {'r1': 1, 'r2': 1, 'r4': 1, 'r5': 1, 'r7': 1, 'r8': 1, 'r9': 1, 'r_S': 1, 'r_P': 1, 'r_BM': 1, 'r_Q': 2}


def _weak_coupling(model):
    """Suppress low product yield at growth, protect growth: PROTECT and finite flux bounds put
    rows over z and fluxes into the MILP, next to the indicator constraints."""
    return [sd.SDModule(model, SUPPRESS, constraints=["r_P - 0.4 r_S <= 0", "r_BM >= 0.1"]),
            sd.SDModule(model, PROTECT, constraints=["r_BM >= 0.2"])]


def _nested(model):
    """The same with an inner objective in the SUPPRESS module."""
    return [sd.SDModule(model, SUPPRESS, inner_objective="r_BM", constraints=["r_P - 0.4 r_S <= 0", "r_S >= 0.1"]),
            sd.SDModule(model, PROTECT, constraints=["r_BM >= 0.2"])]


def _designs(model, modules, caplog, expect, **extra):
    """Design set of one enumeration; `expect` is the log line that proves which method ran."""
    sd_setup = {MODULES: modules, MAX_COST: 4, MAX_SOLUTIONS: inf, SOLUTION_APPROACH: POPULATE,
                SOLVER: SCIP, KOCOST: KO_COST, 'compress': False, SEED: 1}
    sd_setup.update(extra)
    caplog.clear()
    with caplog.at_level(logging.INFO):
        solution = sd.compute_strain_designs(model, sd_setup=sd_setup)
    assert solution.status == OPTIMAL
    messages = [r.getMessage() for r in caplog.records]
    assert any(expect in m for m in messages), "expected '%s' in the log" % expect
    return {frozenset(d.items()) for d in solution.get_reaction_sd()}


@pytest.mark.parametrize("modules", [_weak_coupling, _nested])
@pytest.mark.parametrize("compress", [False, True])
def test_clause_enumeration_returns_the_designs_of_the_tree(model_weak_coupling, modules, compress, caplog, monkeypatch):
    """Same problem, both methods of SCIP's populate: identical, non-empty design sets."""
    monkeypatch.delenv('SD_SCIP_CLAUSES', raising=False)
    by_clauses = _designs(model_weak_coupling, modules(model_weak_coupling), caplog, 'clause enumeration, cost', compress=compress)
    monkeypatch.setenv('SD_SCIP_CLAUSES', '0')
    by_tree = _designs(model_weak_coupling, modules(model_weak_coupling), caplog, 'cb-reject', compress=compress)
    assert len(by_clauses) > 0
    assert by_clauses == by_tree


def test_clause_enumeration_with_worker_processes(model_weak_coupling, caplog, monkeypatch):
    """Oracle LPs in worker processes return the designs of the sequential search."""
    monkeypatch.delenv('SD_SCIP_CLAUSES', raising=False)
    modules = _weak_coupling(model_weak_coupling)
    sequential = _designs(model_weak_coupling, modules, caplog, 'clause enumeration, cost')
    monkeypatch.setattr(clause_enumeration, '_PARALLEL_AFTER', -1.0)
    parallel = _designs(model_weak_coupling, modules, caplog, 'clause enumeration, cost', **{MILP_THREADS: 2})
    assert len(sequential) > 0
    assert parallel == sequential


def test_knock_ins_keep_the_solve_and_exclude_loop(model_weak_coupling, caplog, monkeypatch):
    """With a knock-in column neither of populate's level methods applies; the enumeration has to
    end all the same, without announcing one of them."""
    monkeypatch.delenv('SD_SCIP_CLAUSES', raising=False)
    kos = {k: v for k, v in KO_COST.items()}
    with caplog.at_level(logging.INFO):
        solution = sd.compute_strain_designs(model_weak_coupling, sd_modules=_weak_coupling(model_weak_coupling),
                                             max_cost=3, max_solutions=inf, solution_approach=POPULATE, solver=SCIP,
                                             ko_cost=kos, ki_cost={'r3': 1.0, 'r6': 1.0}, compress=False)
    # no design fits the budget, which a proven enumeration reports as INFEASIBLE
    assert solution.status == INFEASIBLE and not solution.reaction_sd
    messages = [r.getMessage() for r in caplog.records]
    assert not any('clause enumeration, cost' in m or 'cb-reject' in m for m in messages)


def _tiny_coefficient_system(threshold):
    """a = 1e-7 * g (a precursor of g with a tiny coefficient), a <= 0 (knocked out), g >= threshold,
    as rows A x <= b over x = (a, g), both columns unbounded above."""
    from scipy import sparse
    A = sparse.csr_matrix(np.array([[-1.0, 1e-7], [1.0, 0.0], [0.0, -1.0]]))
    b = np.array([0.0, 0.0, -threshold])
    return A, b, np.array([0.0, 0.0]), np.array([np.inf, np.inf])


def test_exact_margin_confirms_a_proof_through_a_tiny_coefficient():
    """Infeasible by 0.05 * 1e-7 in the rows: the multipliers, given only to float accuracy and
    slightly off, are re-derived exactly and the proof holds."""
    A, b, lb, ub = _tiny_coefficient_system(0.05)
    y = np.array([1.0, 1.0, 1e-7 * (1 + 3e-12)])
    assert clause_enumeration._exact_margin(A, b, y / y.sum(), lb, ub)


def test_exact_margin_rejects_a_feasible_system():
    """With the threshold at 0 the same rows have the solution a = g = 0, so no multipliers prove
    anything."""
    A, b, lb, ub = _tiny_coefficient_system(0.0)
    y = np.array([1.0, 1.0, 1e-7])
    assert not clause_enumeration._exact_margin(A, b, y / y.sum(), lb, ub)


def test_oracle_decides_a_set_the_proof_lp_cannot(monkeypatch):
    """One gate (z = 0 -> a <= 0) on the tiny-coefficient system. Whatever the proof LP does with
    multipliers of 1e7, the answer for z = 0 has to be a clause on that gate, and z = 1 a design."""
    from types import SimpleNamespace
    from scipy import sparse
    from straindesign.scip_interface import SCIP_LP
    # columns: z, a, g
    c_rows = [(np.array([1, 2]), np.array([-1.0, 1e-7]), 0.0, False), (np.array([2]), np.array([-1.0]), -0.05, False)]
    gate = SimpleNamespace(A=sparse.csr_matrix(np.array([[0.0, 1.0, 0.0]])), b=[0.0], sense='L', binv=[0], indicval=[0])
    oracle = clause_enumeration.GateOracle(c_rows, [], [0.0, 0.0, 0.0], [1.0, np.inf, np.inf], np.array([1, 2]), gate, SCIP_LP)
    pos, neg = oracle.query(np.array([False, False, False]))
    assert list(pos) == [0] and len(neg) == 0
    assert oracle.query(np.array([True, False, False])) is None
    # the same with the proof LP switched off: the minimal-violation LP alone has to decide
    monkeypatch.setattr(oracle.lp, 'solve_verdict', lambda: (None, ERROR))
    pos, neg = oracle.query(np.array([False, False, False]))
    assert list(pos) == [0] and len(neg) == 0 and oracle.n_by_violation == 1 and oracle.n_exact == 1
    assert oracle.query(np.array([True, False, False])) is None
