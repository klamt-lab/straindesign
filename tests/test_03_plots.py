"""Test if basic plotting functions finish correctly (flux space, yield space, mixed 3d-space)."""
from .test_01_load_models_and_solvers import *
import straindesign as sd


@pytest.mark.timeout(15)
def test_plot_2d_flux_space(curr_solver, model_weak_coupling):
    """Test plot with constraints."""
    constr = ['r4 = 0', 'r7 = 0', 'r9 = 0', 'r_BM >= 4']
    sd.plot_flux_space(model_weak_coupling, ('r_P', 'r_S'), constraints=constr, plt_backend='template')


@pytest.mark.timeout(15)
def test_plot_2d_yield_space(curr_solver, model_weak_coupling):
    """Test plot with constraints."""
    constr = ['r4 = 0', 'r7 = 0', 'r9 = 0', 'r_BM >= 4']
    sd.plot_flux_space(model_weak_coupling, (('r_P', 'r_S'), ('r_BM', 'r_S')), constraints=constr, plt_backend='template')


@pytest.mark.timeout(15)
def test_plot_3d_space(curr_solver, model_weak_coupling):
    """Test plot with constraints."""
    constr = ['r4 = 0', 'r7 = 0', 'r9 = 0', 'r_BM >= 4']
    sd.plot_flux_space(model_weak_coupling, (('r_P', 'r_S'), 'r_BM', 'r_Q'), constraints=constr, solver=curr_solver, plt_backend='template')
