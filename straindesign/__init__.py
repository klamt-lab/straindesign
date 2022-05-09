from importlib.util import find_spec as module_exists

avail_solvers = []
if module_exists('swiglpk'):
    avail_solvers += ['glpk']
if module_exists('cplex'):
    avail_solvers += ['cplex']
if module_exists('gurobipy'):
    avail_solvers += ['gurobi']
if module_exists('pyscipopt'):
    avail_solvers += ['scip']
    
from .names import *
from .solver_interface import *
from .indicatorConstraints import *
from .pool import *
from .efmtool import *
from .parse_constr import *
from .lptools import *
from .strainDesignModule import *
from .strainDesignSolution import *
from .strainDesignMILPBuilder import *
from .strainDesignMILP import *
from .strainDesigner import *
from .compute_strain_designs import *
