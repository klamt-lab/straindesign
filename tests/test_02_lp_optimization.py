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

from cobra.io import load_model
from cobra.flux_analysis import flux_variability_analysis as cobra_fva
from straindesign.lptools import fva, fva_legacy, select_solver
from straindesign.speedy_fva import speedy_fva

FVA_TOL = 1e-6


@pytest.fixture(scope="module")
def ecoli_core():
    return load_model("e_coli_core")


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
