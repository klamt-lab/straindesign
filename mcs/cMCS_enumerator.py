import numpy as np
from scipy import sparse
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
    def __init__(self, model: cobra.Model, mcs_modules: List[mcs_module.MCS_Module], 
        koCost: Dict[str, float]={}, kiCost: Dict[str, float]={}, *args, **kwargs):
        # the matrices in mcs_modules, koCost and kiCost should be np.array or scipy.sparse (csr, csc, lil) format
        self.model  = model
        reac_ids = model.reactions.list_attr("id")
        numr = len(model.reactions)
        # Create vectors for koCost, kiCost, inverted bool-vars and non-targetable bools
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
        # Prepare top lines of MILP (sum of KOs below threshold) and objective function
        self.A_ineq = sparse.csr_matrix([self.cost, [-i for i in self.cost]])
        self.b_ineq = [0, np.inf]
        self.z_map_constr_ineq = sparse.csr_matrix((numr,2))
        self.rownames_ineq = ['sum_z_min', 'sum_z_max']
        self.colnames = ['z' + str(i) for i in range(0,numr)]
        self.c = self.cost.copy
        self.lb = [0]*numr
        self.ub = [1-float(i) for i in self.z_non_targetable]
        self.idx_z = [i for i in range(0,numr)]
        # Initialize also empty equality matrix
        self.A_eq = sparse.csr_matrix((0,numr))
        self.b_eq = []
        self.z_map_constr_eq = sparse.csr_matrix((numr,0))
        self.rownames_eq = []
        self.num_modules = 0
        self.indicators = [] # Add instances of the class 'Indicator_constraint' later
        # Initialize association between z and variables and variables
        self.z_map_vars = sparse.csr_matrix((numr,numr))
        for i in range(0,len(mcs_modules)):
            self.addModule(mcs_modules[i])

        print("done")

    def addModule(self,mcs_module):
        # Generating LP and z-linking-matrix for each module
        #
        # Modules to describe desired or undesired flux states for MCS strain design.
        # mcs_module needs to be an Instance of the class 'MCS_Module'
        #
        # There are three kinds of flux states that can be described
        # 1. The wildtype model, constrainted with additional inequalities:
        #    e.g.: T v <= t.
        # 2. The wildtype model at a specific optimum, constrained with additional inequalities
        #    e.g.: objective*v = optimal, T v <= t
        # 3. A yield range:
        #    e.g.: numerator*v/denominator*v <= t (Definition of A and b is not required)
        #
        # Attributes
        # ----------
        #     module_sense: 'desired' or 'target'
        #     module_type: 'lin_constraints', 'bilev_w_constr', 'yield_w_constr'
        #     equation: String to specify linear constraints: A v <= b, A v >= b, A v = b
        #         (e.g. T v <= t with 'target' or D v <= d with 'desired')
        # ----------
        # Module type specific attributes:
        #     lin_constraints: <none>
        #     bilev_w_constr: inner_objective: Inner optimization vector
        #     yield_w_constr: numerator: numerator of yield function,
        #                     denominator: denominator of yield function
        #
        self.num_modules += 1
        z_map_constr_ineq_i = []
        z_map_constr_eq_i = []
        z_map_vars_i = []
        # Translate (in)equalities into matrix form
        self.const2mat(mcs_module.equations)

        # 2. Construct LP for module
        # if mcs_module.module_type == 'lin_constraints':
        #     # Classical MCS
        #     [A_ineq_p, b_ineq_p, A_eq_p, b_eq_p, ~, lb_p, ub_p, z_map_constr_ineq_p, z_map_constr_eq_p, z_map_vars_p] = build_primal(obj,module.V,module.v,[],module.lb,module.ub);
        #     # 3. Prepare module as target or desired
        # if mcs_module.module_sense == 'desired':
        #     [A_ineq_i, b_ineq_i, A_eq_i, b_eq_i, lb_i, ub_i, z_map_constr_ineq_i, z_map_constr_eq_i] = reassign_lb_ub_from_ineq(obj,A_ineq_p, b_ineq_p, A_eq_p, b_eq_p, lb_p, ub_p, z_map_constr_ineq_p, z_map_constr_eq_p, z_map_vars_p);
        #     z_map_vars_i        = z_map_vars_p;
        # elif mcs_module.module_sense == 'target':
        #     c_p = zeros(1,size(A_eq_p,2));
        #     [A_ineq_d, b_ineq_d, A_eq_d, b_eq_d, c_d, lb_i, ub_i, z_map_constr_ineq_d, z_map_constr_eq_d, z_map_vars_i] = dualize(obj,A_ineq_p, b_ineq_p, A_eq_p, b_eq_p, c_p, lb_p, ub_p, z_map_constr_ineq_p, z_map_constr_eq_p, z_map_vars_p);
        #     [A_ineq_i, b_ineq_i, A_eq_i, b_eq_i, z_map_constr_ineq_i, z_map_constr_eq_i] = dual_2_farkas(obj,A_ineq_d, b_ineq_d,A_eq_d, b_eq_d, c_d, z_map_constr_ineq_d,z_map_constr_eq_d);
        
        # 3. Add module to global MILP
        # obj.z_map_constr_ineq   = [obj.z_map_constr_ineq,   z_map_constr_ineq_i];
        # obj.z_map_constr_eq     = [obj.z_map_constr_eq,     z_map_constr_eq_i];
        # obj.z_map_vars          = [obj.z_map_vars,          z_map_vars_i];
        # obj.A_ineq  = [obj.A_ineq,                                  zeros(size(obj.A_ineq,1), size(A_ineq_i,2)) ; ...
        #     zeros(size(A_ineq_i,1), size(obj.A_ineq,2)), A_ineq_i];
        # obj.b_ineq  = [obj.b_ineq;  b_ineq_i];
        # obj.A_eq    = [obj.A_eq,                                zeros(size(obj.A_eq,1),size(A_eq_i,2)) ; ...
        #     zeros(size(A_eq_i,1), size(obj.A_eq,2)), A_eq_i];
        # obj.b_eq    = [obj.b_eq;  b_eq_i];
        # obj.c  = [obj.c;  zeros(size(A_ineq_i,2),1)];
        # obj.lb = [obj.lb; lb_i];
        # obj.ub = [obj.ub; ub_i];
        # obj.rownames_ineq = [ obj.rownames_ineq ; cellfun(@(x) [num2str(obj.num_modules) '_mod_ineq_' num2str(x)], num2cell(1:size(A_ineq_i,1)), 'UniformOutput', false)'];
        # obj.rownames_eq = [ obj.rownames_eq ; cellfun(@(x) [num2str(obj.num_modules) '_mod_eq_' num2str(x)], num2cell(1:size(A_eq_i,1)), 'UniformOutput', false)'];
        # obj.colnames = [ obj.colnames , cellfun(@(x) [num2str(obj.num_modules) '_mod_var_' num2str(x)], num2cell(1:size(A_eq_i,2)), 'UniformOutput', false)];
        print('hello world')

    def compute_mcs(self,maxSolutions=np.inf, maxCost=np.inf) -> List:
        mcs = []
        return mcs

    def const2mat(self,equations):
        for equation in equations:
            semantics = []
            reaction_ids = []
            last_part = ""
            counter = 1
            for char in equation+" ":
                if (char == " ") or (char in ("*", "/", "+", "-")) or (counter == len(equation+" ")):
                    if last_part != "":
                        try:
                            float(last_part)
                        except ValueError:
                            reaction_ids.append(last_part)
                            semantics.append("reaction")
                        else:
                            semantics.append("number")
                        last_part = ""

                    if counter == len(equation+" "):
                        break

                if char in "*":
                    semantics.append("multiplication")
                elif char in "/":
                    semantics.append("division")
                elif char in ("+", "-"):
                    semantics.append("dash")
                elif char not in " ":
                    last_part += char
                counter += 1
        print(char)

    class Indicator_constraint:
        def __init__(self, ic_binv, ic_A_ineq, ic_b_ineq, ic_sense, ic_indicval):
            self.ic_binv     = ic_binv # index of binary variable
            self.ic_A_ineq   = ic_A_ineq # CPLEX: lin_expr,   left hand side coefficient row for indicator constraint
            self.ic_b_ineq   = ic_b_ineq # right hand side for indicator constraint
            self.ic_sense    = ic_sense # sense of the indicator constraint can be 'L', 'E', 'G' (lower-equal, equal, greater-equal)
            self.ic_indicval = ic_indicval # value the binary variable takes when constraint is fulfilled

