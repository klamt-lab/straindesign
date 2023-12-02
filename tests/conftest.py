import pytest
from cobra import Configuration
from straindesign.names import *

cobra_conf = Configuration()
bound_thres = max((abs(cobra_conf.lower_bound), abs(cobra_conf.upper_bound)))

# Initialize an empty list for solvers
solvers = [GLPK]

# Add GRUOBI to the list if the cplex package is installed
try:
    import gurobipy
    solvers.append(GUROBI)
except ImportError:
    pass  # GUROBI is not installed

# Add CPLEX to the list if the cplex package is installed
try:
    import cplex
    solvers.append(CPLEX)
except ImportError:
    pass  # CPLEX is not installed

# Add SCIP to the list if the pyscipopt package is installed
try:
    import pyscipopt
    solvers.append(SCIP)
except ImportError:
    pass  # SCIP is not installed


@pytest.fixture(params=solvers, scope="session")
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
