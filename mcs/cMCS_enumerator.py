import numpy as np
import scipy
import cobra
# import optlang.glpk_interface
# from optlang.symbolics import add
# from optlang.exceptions import IndicatorConstraintsNotSupported
# from swiglpk import glp_write_lp
# try:
#     import optlang.cplex_interface
#     import cplex
#     from cplex.exceptions import CplexSolverError
#     from cplex._internal._subinterfaces import SolutionStatus # can be also accessed by a CPLEX object under .solution.status
# except:
#     optlang.cplex_interface = None # make sure this symbol is defined for type() comparisons
# try:
#     import optlang.gurobi_interface
#     from gurobipy import GRB, LinExpr
# except:
#     optlang.gurobi_interface = None # make sure this symbol is defined for type() comparisons
# try:
#     import optlang.coinor_cbc_interface
# except:
#     optlang.coinor_cbc_interface = None # make sure this symbol is defined for type() comparisons
from typing import Dict, List, Tuple, Union, FrozenSet
import time
from mcs import mcs_computation,mcs_module

class MinimalCutSetsEnumerator:
    def __init__(self, model: cobra.Model, mcs_modules: mcs_module.MCS_Module, 
        koCost: Dict[str, float]={}, kiCost: Dict[str, float]={}, *args, **kwargs):
        # the matrices in mcs_modules, koCost and kiCost should be numpy.array or scipy.sparse (csr, csc, lil) format
        self.model  = model
        reac_ids = model.reactions.list_attr("id")
        numr = len(model.reactions)
        # Create vectors for koCost, kiCost, inverted and non-targetable
        self.koCost = [koCost.get(key) if (key in koCost.keys()) else np.nan for key in reac_ids]
        self.kiCost = [kiCost.get(key) if (key in kiCost.keys()) else np.nan for key in reac_ids]
        self.num_z  = numr
        self.cost = [i for i in self.koCost]
        for i in [i for i, x in enumerate(self.kiCost) if not np.isnan(x)]:
            self.cost[i] = self.kiCost[i]
        self.z_inverted = [not np.isnan(x) for x in self.kiCost]
        self.z_non_targetable = [np.isnan(x) for x in self.cost]
        for i in [i for i, x in enumerate(self.cost) if np.isnan(x)]:
            self.cost[i] = 0
        # Prepare top line of MILP (sum of KOs below threshold)
        print("done")


    def compute_mcs(self,maxSolutions=numpy.inf, maxCost=numpy.inf) -> List:
        mcs = []
        return mcs