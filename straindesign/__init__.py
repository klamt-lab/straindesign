from importlib.util import find_spec as module_exists
from .names import *
import logging


class DisableLogger():

    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)


avail_solvers = set()
if module_exists("swiglpk"):
    avail_solvers.add(GLPK)
if module_exists("cplex"):
    avail_solvers.add(CPLEX)
if module_exists("gurobipy"):
    avail_solvers.add(GUROBI)
if module_exists("pyscipopt"):
    avail_solvers.add(SCIP)

from .solver_interface import *
from .indicatorConstraints import *
from .pool import *
from .efmtool import *
from .parse_constr import *
from .lptools import *
from .networktools import *
from .strainDesignModule import *
from .strainDesignSolutions import *
from .strainDesignProblem import *
from .strainDesignMILP import *
from .compute_strain_designs import *
