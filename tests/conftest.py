import pytest
from straindesign.names import *

@pytest.fixture(params=[CPLEX,GUROBI,SCIP,GLPK], scope="session")
def curr_solver(request: pytest.FixtureRequest) -> str:
    """Provide session-level fixture for parametrized solver names."""
    return request.param

@pytest.fixture(params=[ANY,BEST,POPULATE], scope="session")
def comp_approach(request: pytest.FixtureRequest) -> str:
    """Provide session-level fixture for parametrized solver names."""
    return request.param