"""Test if basic lp-functions finish correctly (FBA, FVA, yield optimization)."""
from .test_01_load_models_and_solvers import *
import straindesign as sd
from numpy import inf, isinf, isnan


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
