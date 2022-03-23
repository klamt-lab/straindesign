# from mcs.StrainDesigner import StrainDesigner
# from mcs.cplex_interface import cplex_interface
# from mcs.mcs_computation import mcs_computation
# from mcs.mcs_module import mcs_module
# from mcs.solver_interface import solver_interface
from inspect import ismodule
from .strainDesignModule import *
from .indicator_constraints import *
from .solver_interface import *
from .pool import Pool
# if ismodule('cplex'):
#     from .cplex_interface import *
# elif ismodule('gurobipy'):
#     from .gurobi_interface import *
# elif ismodule('pyscipopt'):
#     from .scip_interface import *
from .constr2mat import *
from .strainDesignMILPBuilder import *
from .strainDesignMILP import *
from .strainDesigner import *
from .fba import *
from .fva import *