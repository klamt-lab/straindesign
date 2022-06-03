import pytest
from cobra import Configuration
from straindesign.names import *

cobra_conf = Configuration()
bound_thres = max((abs(cobra_conf.lower_bound), abs(cobra_conf.upper_bound)))


@pytest.fixture(params=[CPLEX, GUROBI, SCIP, GLPK], scope="session")
def curr_solver(request: pytest.FixtureRequest) -> str:
    """Provide session-level fixture for parametrized solver names."""
    return request.param


@pytest.fixture(params=[ANY, BEST, POPULATE], scope="session")
def comp_approach(request: pytest.FixtureRequest) -> str:
    """Provide session-level fixture for computation modes."""
    return request.param


@pytest.fixture(params=[BEST, POPULATE], scope="session")
def comp_approach_best_populate(request: pytest.FixtureRequest) -> str:
    """Provide session-level fixture for nested-opt computation modes."""
    return request.param


@pytest.fixture(params=[None, bound_thres], scope="session")
def bigM(request: pytest.FixtureRequest) -> str:
    """Provide session-level fixture for nested-opt computation modes."""
    return request.param


@pytest.fixture(scope="session")
def compression(request: pytest.FixtureRequest) -> str:
    """Provide session-level fixture for nested-opt computation modes."""
    return False
