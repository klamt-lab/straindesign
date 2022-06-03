"""Test if models load."""
from os.path import dirname, abspath
from cobra.io import read_sbml_model
from cobra import Configuration
import straindesign as sd
from straindesign.names import *
import pytest


@pytest.fixture
def model_gpr():
    """Load model with complex GPR rules."""
    return read_sbml_model(dirname(abspath(__file__)) + "/model_gpr.xml")


@pytest.fixture
def model_weak_coupling():
    """Load model with potential for designing weakly growth coupled production."""
    return read_sbml_model(
        dirname(abspath(__file__)) + "/model_weak_coupling.xml")


@pytest.fixture
def model_small_example():
    """Load example model with two substrates."""
    return read_sbml_model(
        dirname(abspath(__file__)) + "/model_small_example.xml")


def test_import_sd():
    import straindesign


def test_solver_availability(curr_solver):
    """Test solver availability."""
    assert (curr_solver in sd.avail_solvers)


def test_solver_loading(curr_solver):
    """Test that solvers interfaces can be loaded."""
    milp = sd.MILP_LP(solver=curr_solver)
    assert (milp.solve() == ([], 0.0, OPTIMAL))


def test_load_solvers(model_small_example):
    """Test solver choice."""

    # solver selection with no solver specified
    solver1 = sd.select_solver()
    assert (solver1 in [CPLEX, GUROBI, GLPK, SCIP])

    # with solver specified
    solver2 = sd.select_solver('scip')
    assert (solver2 == SCIP)

    # with model-specified solver
    model_small_example.solver = 'glpk'
    solver3 = sd.select_solver(None, model_small_example)
    assert (solver3 == GLPK)

    # with cobrapy-specified solver
    conf = Configuration()
    conf.solver = 'cplex'
    solver4 = sd.select_solver()
    assert (solver4 == CPLEX)

    # with solver in model that overwrites the global specification
    model_small_example.solver = 'gurobi'
    solver5 = sd.select_solver(None, model_small_example)
    assert (solver5 == GUROBI)
