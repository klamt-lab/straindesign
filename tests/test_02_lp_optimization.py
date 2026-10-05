"""Test if basic lp-functions finish correctly (FBA, FVA, yield optimization)."""
from .test_01_load_models_and_solvers import *
import straindesign as sd
from numpy import inf, isinf, isnan
import numpy as np


def test_fba(curr_solver, model_gpr):
    """Test FBA with constraints."""
    sol = sd.fba(model_gpr, solver=curr_solver, constraints=['r3 <= 3', 'r5 = 1.5'])
    assert (round(sol.objective_value, 9) == 4.5)
    assert (sol.status == sd.OPTIMAL)


def test_fba_unbounded(curr_solver, model_gpr):
    """Test unbounded FBA."""
    for r in model_gpr.reactions:
        if r._lower_bound < 0:
            r._lower_bound = -inf
        if r._upper_bound > 0:
            r._upper_bound = inf
    sol = sd.fba(model_gpr, solver=curr_solver)
    assert (sol.status == sd.UNBOUNDED)


def test_fba_infeasible(curr_solver, model_gpr):
    """Test infeasible FBA."""
    sol = sd.fba(model_gpr, solver=curr_solver, constraints='r3 <= -2')
    assert (sol.status == sd.INFEASIBLE)


def test_fva(curr_solver, model_gpr):
    """Test FVA with constraints."""
    sol = sd.fva(model_gpr, solver=curr_solver, constraints=['r3 <= 3', 'r5 = 1.5'])
    assert (sol.shape == (11, 2))


def test_fva_infeasible(curr_solver, model_gpr):
    """Test infeasible FVA with constraints."""
    sol = sd.fva(model_gpr, solver=curr_solver, constraints=['r3 <= -3', 'r5 = 1.5'])
    assert (sol.shape == (11, 2))
    assert (isnan(sol.values[1, 1]))


def test_fva_unbounded(curr_solver, model_small_example):
    """Test FVA that is partially unbounded."""
    for r in model_small_example.reactions:
        if r.name in ['R2', 'R3', 'R9']:
            if r._lower_bound < 0:
                r._lower_bound = -inf
            if r._upper_bound > 0:
                r._upper_bound = inf
    sol = sd.fva(model_small_example, solver=curr_solver, constraints=['R5 = 1.5'])
    assert (sol.shape == (10, 2))
    assert (isinf(sol.values[1, 1]))


def test_yield_opt(curr_solver, model_weak_coupling):
    """Test yield optimization."""
    constr = ['r4 = 0', 'r7 = 0', 'r9 = 0', 'r_BM >= 4']
    num = 'r_P'
    den = 'r_S'
    sol = sd.yopt(model_weak_coupling, obj_num=num, obj_den=den, solver=curr_solver, constraints=constr)
    assert (round(sol.objective_value, 9) == 0.6)
    sol = sd.yopt(model_weak_coupling, obj_sense='min', obj_num=num, obj_den=den, solver=curr_solver, constraints=constr)
    assert (round(sol.objective_value, 9) == 0.2)


def test_yield_opt_unbounded(curr_solver, model_small_example):
    """Test yield optimization that is unbounded."""
    num = 'R4'
    den = 'R2'
    sol = sd.yopt(model_small_example, obj_num=num, obj_den=den, solver=curr_solver)
    assert (sol.status == sd.UNBOUNDED)


def test_yield_opt_infeasible(curr_solver, model_small_example):
    """Test yield optimization that is infeasible."""
    constr = ['R1 = -1']
    num = 'R4'
    den = 'R1'
    sol = sd.yopt(model_small_example, obj_num=num, obj_den=den, constraints=constr, solver=curr_solver)
    assert (sol.status == sd.INFEASIBLE)


# ── FVA implementation correctness (e_coli_core) ─────────────────────
# Compares speedy_fva (compressed + uncompressed), fva_legacy, and cobra FVA.

from ._models import load_test_model
from cobra.flux_analysis import flux_variability_analysis as cobra_fva
from straindesign.lptools import fva, fva_legacy, select_solver
from straindesign.speedy_fva import speedy_fva

FVA_TOL = 1e-6


@pytest.fixture(scope="module")
def ecoli_core():
    return load_test_model("e_coli_core")


@pytest.fixture(scope="module")
def fva_solver(ecoli_core):
    return select_solver(None, ecoli_core)


@pytest.fixture(scope="module")
def ref_cobra(ecoli_core):
    """Cobra's own FVA as ground truth (unconstrained)."""
    return cobra_fva(ecoli_core.copy(), fraction_of_optimum=0.0)


@pytest.fixture(scope="module")
def ref_legacy(ecoli_core, fva_solver):
    """StrainDesign legacy (brute-force) FVA."""
    return fva_legacy(ecoli_core.copy(), solver=fva_solver)


@pytest.fixture(scope="module")
def res_compressed(ecoli_core, fva_solver):
    """speedy_fva with compression."""
    return speedy_fva(ecoli_core.copy(), solver=fva_solver, compress=True)


@pytest.fixture(scope="module")
def res_uncompressed(ecoli_core, fva_solver):
    """speedy_fva without compression."""
    return speedy_fva(ecoli_core.copy(), solver=fva_solver, compress=False)


def _max_err(a, b):
    """Max absolute error across min and max columns, aligning by index."""
    common = a.index.intersection(b.index)
    err_min = np.abs(a.loc[common, "minimum"].values - b.loc[common, "minimum"].values).max()
    err_max = np.abs(a.loc[common, "maximum"].values - b.loc[common, "maximum"].values).max()
    return max(err_min, err_max)


def test_fva_compressed_vs_legacy(res_compressed, ref_legacy):
    assert _max_err(res_compressed, ref_legacy) < FVA_TOL


def test_fva_uncompressed_vs_legacy(res_uncompressed, ref_legacy):
    assert _max_err(res_uncompressed, ref_legacy) < FVA_TOL


def test_fva_compressed_vs_cobra(res_compressed, ref_cobra):
    assert _max_err(res_compressed, ref_cobra) < FVA_TOL


def test_fva_uncompressed_vs_cobra(res_uncompressed, ref_cobra):
    assert _max_err(res_uncompressed, ref_cobra) < FVA_TOL


def test_fva_wrapper_matches_speedy(ecoli_core, fva_solver):
    """fva() wrapper should produce identical results to speedy_fva()."""
    res_wrapper = fva(ecoli_core.copy(), solver=fva_solver)
    res_direct = speedy_fva(ecoli_core.copy(), solver=fva_solver)
    assert _max_err(res_wrapper, res_direct) < FVA_TOL


def test_fva_all_reactions_present(ecoli_core, res_compressed):
    """FVA result must contain every reaction in the model."""
    assert set(res_compressed.index) == {r.id for r in ecoli_core.reactions}


def test_fva_small_model_stays_sequential(ecoli_core, fva_solver, ref_legacy, monkeypatch):
    """A network this small solves faster in-process than a worker pool can start."""
    import straindesign.speedy_fva as sf
    monkeypatch.setattr(sf, "_PARALLEL_PHASE2_MIN", 1)
    res = speedy_fva(ecoli_core.copy(), solver=fva_solver, compress=False, precheck=False, threads=2)
    assert res.attrs["pooled"] is False
    assert _max_err(res, ref_legacy) < FVA_TOL


def test_fva_hands_off_to_pool(ecoli_core, fva_solver, ref_legacy, monkeypatch):
    """Once the probe projects more work than the pool's start-up, the rest goes to the pool."""
    import straindesign.speedy_fva as sf
    monkeypatch.setattr(sf, "_PARALLEL_PHASE2_MIN", 1)
    monkeypatch.setattr(sf, "_PROBE_LPS", 5)
    monkeypatch.setattr(sf, "_POOL_START_SECONDS", 0.0)
    monkeypatch.setattr(sf, "_POOL_START_SECONDS_PER_WORKER", 0.0)
    res = speedy_fva(ecoli_core.copy(), solver=fva_solver, compress=False, precheck=False, threads=2)
    assert res.attrs["pooled"] is True
    assert _max_err(res, ref_legacy) < FVA_TOL


def test_indicator_is_one_directional(curr_solver):
    """An indicator gates its row in one direction only.

    z = indicval enforces the row; the other value leaves it out entirely, and in particular
    does NOT force the row to be violated. Minimal cut sets rely on exactly this: a knockout
    must be able to drop a constraint, but a constraint that happens to hold must not force
    the knockout binary back to zero. All backends have to agree on it.
    """
    from scipy import sparse

    def max_v(indicval, fix_z):
        ic = sd.IndicatorConstraints([1], sparse.csr_matrix([[1.0, 0.0]]), [0.0], 'L', [indicval])
        milp = sd.MILP_LP(c=[-1.0, 0.0],
                          A_ineq=sparse.csr_matrix((0, 2)), b_ineq=[],
                          A_eq=sparse.csr_matrix((0, 2)), b_eq=[],
                          lb=[0.0, float(fix_z)], ub=[10.0, float(fix_z)],
                          vtype='CB', indic_constr=ic, solver=curr_solver)
        x, _, status = milp.solve()
        assert status == sd.names.OPTIMAL
        return x[0]

    for indicval in (0, 1):
        assert max_v(indicval, indicval) == pytest.approx(0.0)       # row enforced: v <= 0
        assert max_v(indicval, 1 - indicval) == pytest.approx(10.0)  # row absent: v free


# =============================================================================
# Solver status mapping: OPTIMAL, INFEASIBLE and UNBOUNDED only with a proof
# =============================================================================


def _small_lp(A_ineq, b_ineq, solver, vtype=None, lp_method=None):
    """min -x1 - x2 s.t. A_ineq x <= b_ineq, x >= 0"""
    from scipy import sparse
    lp = sd.MILP_LP(c=[-1.0, -1.0],
                    A_ineq=sparse.csr_matrix(A_ineq), b_ineq=b_ineq,
                    A_eq=sparse.csr_matrix((0, 2)), b_eq=[],
                    lb=[0.0, 0.0], ub=[inf, inf], vtype=vtype, solver=solver)
    if lp_method is not None:
        lp.set_lp_method(lp_method)
    return lp


@pytest.mark.parametrize("vtype", [None, 'CI'])
@pytest.mark.parametrize("lp_method", [None, sd.names.LP_METHOD_PRIMAL, sd.names.LP_METHOD_DUAL])
def test_status_mapping(curr_solver, vtype, lp_method):
    """Feasible, infeasible and unbounded (MI)LPs get the matching status, with every LP method."""
    x, opt, status = _small_lp([[1.0, 1.0]], [4.0], curr_solver, vtype, lp_method).solve()
    assert status == sd.names.OPTIMAL
    assert opt == pytest.approx(-4.0)
    assert sum(x) == pytest.approx(4.0)
    x, opt, status = _small_lp([[1.0, 1.0]], [-1.0], curr_solver, vtype, lp_method).solve()
    assert status == sd.names.INFEASIBLE
    assert isnan(opt)
    x, opt, status = _small_lp([[1.0, -1.0]], [4.0], curr_solver, vtype, lp_method).solve()
    assert status == sd.names.UNBOUNDED
    assert opt == -inf


@pytest.mark.parametrize("lp_method", [None, sd.names.LP_METHOD_PRIMAL, sd.names.LP_METHOD_DUAL])
def test_warm_bound_changes_match_fresh_solve(curr_solver, lp_method):
    """A sequence of in-place bound changes, each solved warm, gives the optimum of a fresh LP.

    The sequence passes through infeasible bounds, so a warm solve after an infeasible one is
    covered too."""
    from scipy import sparse
    rng = np.random.default_rng(0)
    m, n = 12, 30
    A = sparse.random(m, n, density=0.3, random_state=1, data_rvs=lambda k: rng.integers(-3, 4, k)).tocsr()
    c = list(rng.integers(-5, 6, n).astype(float))
    lb0, ub0 = [-10.0] * n, [10.0] * n

    def build(lb, ub):
        lp = sd.MILP_LP(c=c, A_ineq=sparse.csr_matrix((0, n)), b_ineq=[], A_eq=A, b_eq=[0.0] * m,
                        lb=list(lb), ub=list(ub), solver=curr_solver)
        if lp_method is not None:
            lp.set_lp_method(lp_method)
        return lp

    warm = build(lb0, ub0)
    lb, ub = list(lb0), list(ub0)
    seen = set()
    for step in range(40):
        idx = [int(i) for i in rng.choice(n, 3, replace=False)]
        for i in idx:
            kind = rng.integers(4)
            if kind == 0:
                lb[i], ub[i] = 0.0, 10.0
            elif kind == 1:
                lb[i], ub[i] = -10.0, 0.0
            elif kind == 2:
                lb[i], ub[i] = -10.0, 10.0
            else:  # r_i >= 1 is infeasible whenever the network forces r_i <= 0
                lb[i], ub[i] = 1.0, 10.0
        warm.set_lb([[i, lb[i]] for i in idx])
        warm.set_ub([[i, ub[i]] for i in idx])
        _, opt_w, status_w = warm.solve()
        _, opt_f, status_f = build(lb, ub).solve()
        assert status_w == status_f
        assert status_f in [sd.names.OPTIMAL, sd.names.INFEASIBLE]
        seen.add(status_f)
        if status_f == sd.names.OPTIMAL:
            assert opt_w == pytest.approx(opt_f, abs=1e-7)
    assert seen == {sd.names.OPTIMAL, sd.names.INFEASIBLE}


@pytest.mark.skipif(sd.names.SCIP not in sd.avail_solvers, reason="SCIP not installed")
def test_scip_lp_recovers_from_lp_error(caplog, monkeypatch):
    """An LP error from a warm start is re-solved from scratch; a persistent one is an error."""
    lp = _small_lp([[1.0, 1.0]], [4.0], sd.names.SCIP)
    lp.solve()
    lp.set_ub([[0, 3.0]])
    backend = lp.backend
    optimize = backend.optimize
    calls = []

    def fail_once(dual=True):
        calls.append(dual)
        if len(calls) == 1:
            raise Exception('SCIP: error in LP solver!')
        return optimize(dual=dual)

    monkeypatch.setattr(backend, 'optimize', fail_once)
    with caplog.at_level('DEBUG'):
        x, opt, status = lp.solve()
    assert len(calls) == 2
    assert status == sd.names.OPTIMAL
    assert opt == pytest.approx(-4.0)
    assert not [r for r in caplog.records if r.levelno >= 40]
    assert backend.getIntParam(__import__('pyscipopt').SCIP_LPPARAM.FROMSCRATCH) == 0

    def fail(dual=True):
        raise Exception('SCIP: error in LP solver!')

    monkeypatch.setattr(backend, 'optimize', fail)
    x, opt, status = lp.solve()
    assert status == sd.names.ERROR
    assert isnan(opt) and all(isnan(x))
    assert isnan(lp.slim_solve())


@pytest.mark.skipif(sd.names.SCIP not in sd.avail_solvers, reason="SCIP not installed")
def test_scip_lp_stopped_early_is_not_optimal():
    """A SoPlex solve stopped by its iteration limit has a finite objective but is not optimal."""
    from pyscipopt import SCIP_LPPARAM
    from scipy import sparse
    lp = sd.MILP_LP(c=[1.0, 1.0, 1.0], A_ineq=sparse.csr_matrix(-np.eye(3)), b_ineq=[-1.0] * 3,
                    A_eq=sparse.csr_matrix((0, 3)), b_eq=[], lb=[0.0] * 3, ub=[10.0] * 3, solver=sd.names.SCIP)
    lp.backend.setIntParam(SCIP_LPPARAM.LPITLIM, 1)
    _, opt, status = lp.solve()
    assert status == sd.names.ERROR
    assert isnan(opt)
    lp.backend.setIntParam(SCIP_LPPARAM.LPITLIM, 2**31 - 1)
    _, opt, status = lp.solve()
    assert status == sd.names.OPTIMAL
    assert opt == pytest.approx(3.0)


def test_glpk_unproven_infeasibility_is_resolved(monkeypatch):
    """GLPK's GLP_INFEAS (current basis infeasible) and a dual-simplex GLP_NOFEAS are no proof.

    The first simplex call is cut short: with an iteration limit of one, the slack basis of
    x_i >= 1 (as rows) is still primal infeasible. Next, the dual simplex is made to report
    GLP_NOFEAS on the feasible LP. Both must end OPTIMAL, not INFEASIBLE."""
    import swiglpk
    import straindesign.glpk_interface as glpk_interface
    from scipy import sparse
    simplex = glpk_interface.glp_simplex
    get_status = glpk_interface.glp_get_status

    def build():
        return sd.MILP_LP(c=[1.0, 1.0, 1.0], A_ineq=sparse.csr_matrix(-np.eye(3)), b_ineq=[-1.0] * 3,
                          A_eq=sparse.csr_matrix((0, 3)), b_eq=[], lb=[0.0] * 3, ub=[10.0] * 3,
                          solver=sd.names.GLPK)

    calls = []

    def simplex_itlim_once(prob, params):
        calls.append(1)
        if len(calls) > 1:
            return simplex(prob, params)
        it_lim, params.it_lim = params.it_lim, 1
        try:
            return simplex(prob, params)
        finally:
            params.it_lim = it_lim

    monkeypatch.setattr(glpk_interface, 'glp_simplex', simplex_itlim_once)
    _, opt, status = build().solve()
    assert status == sd.names.OPTIMAL
    assert opt == pytest.approx(3.0)
    monkeypatch.setattr(glpk_interface, 'glp_simplex', simplex)

    lp = build()
    lp.set_lp_method(sd.names.LP_METHOD_DUAL)
    reported = []

    def get_status_nofeas_once(prob):
        if not reported:
            reported.append(1)
            return swiglpk.GLP_NOFEAS
        return get_status(prob)

    monkeypatch.setattr(glpk_interface, 'glp_get_status', get_status_nofeas_once)
    _, opt, status = lp.solve()
    assert status == sd.names.OPTIMAL
    assert opt == pytest.approx(3.0)
    assert lp.get_lp_method() == sd.names.LP_METHOD_DUAL
