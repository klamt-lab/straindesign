import pytest
from importlib.util import find_spec
from cobra import Configuration
from straindesign.names import *


# ---------------------------------------------------------------------------
# Custom CLI flags for test_performance.py tiered benchmarks
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    for name, help_text in [
        ("--medium", "Run iMLcore genome-scale benchmarks (~4 min total)."),
        ("--large",  "Run iML1515 large-model benchmarks (several min/solver)."),
    ]:
        try:
            parser.addoption(name, action="store_true", default=False, help=help_text)
        except ValueError:
            pass  # already registered


def pytest_configure(config):
    for marker, desc in [
        ("medium", "genome-scale benchmark; enable with --medium"),
        ("large",  "large-model benchmark; enable with --large"),
    ]:
        config.addinivalue_line("markers", f"{marker}: {desc}")


def pytest_collection_modifyitems(config, items):
    for flag, marker in [("--medium", "medium"), ("--large", "large")]:
        if not config.getoption(flag, default=False):
            skip = pytest.mark.skip(reason=f"pass {flag} to enable this benchmark")
            for item in items:
                if marker in item.keywords:
                    item.add_marker(skip)

cobra_conf = Configuration()
bound_thres = max((abs(cobra_conf.lower_bound), abs(cobra_conf.upper_bound)))

# Detect available solvers via find_spec (no eager import of native libraries)
solvers = [GLPK]
if find_spec("gurobipy"):
    solvers.append(GUROBI)
if find_spec("cplex"):
    solvers.append(CPLEX)
if find_spec("pyscipopt"):
    solvers.append(SCIP)


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
