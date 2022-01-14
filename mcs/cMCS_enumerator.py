from xml.etree.ElementTree import QName
import numpy as np
from scipy import sparse
import cobra
import re
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
        # get lower and upper bound from module if available, otherwise from model
        if not mcs_module.lb == []: 
            lb = mcs_module.lb
        else:
            lb = [r.lower_bound for r in self.model.reactions]
        if not mcs_module.ub == []: 
            ub = mcs_module.ub
        else:
            ub = [r.upper_bound for r in self.model.reactions]
        # 1. Translate (in)equalities into matrix form
        V_ineq, v_ineq, V_eq, v_eq = self.lineq2mat(mcs_module.equations)

        # 2. Construct LP for module
        # if mcs_module.module_type == 'lin_constraints':
        #     # Classical MCS
        A_ineq_p, b_ineq_p, A_eq_p, b_eq_p, c_p, lb_p, ub_p, z_map_constr_ineq_p, z_map_constr_eq_p, z_map_vars_p = self.build_primal(V_ineq,v_ineq,V_eq,v_eq,[],lb,ub)
        # 3. Prepare module as target or desired
        if mcs_module.module_sense == 'desired':
            A_ineq_i, b_ineq_i, A_eq_i, b_eq_i, lb_i, ub_i, z_map_constr_ineq_i, z_map_constr_eq_i = self.reassign_lb_ub_from_ineq(A_ineq_p, b_ineq_p, A_eq_p, b_eq_p, lb_p, ub_p, z_map_constr_ineq_p, z_map_constr_eq_p, z_map_vars_p)
            z_map_vars_i        = z_map_vars_p
        elif mcs_module.module_sense == 'target':
            c_p = [0]*len(c_p)
            A_ineq_d, b_ineq_d, A_eq_d, b_eq_d, c_d, lb_i, ub_i, z_map_constr_ineq_d, z_map_constr_eq_d, z_map_vars_i = self.dualize(A_ineq_p, b_ineq_p, A_eq_p, b_eq_p, c_p, lb_p, ub_p, z_map_constr_ineq_p, z_map_constr_eq_p, z_map_vars_p)
        print(ub_i)
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

    # Builds primal LP problem for a module in the
    # standard form: A_ineq x <= b_ineq, A_eq x = b_eq, lb <= x <= ub, min{c'x}.
    # returns A_ineq, b_ineq, A_eq, b_eq, c, lb, ub, z_map_constr_ineq, z_map_constr_eq, z_map_vars
    def build_primal(self, V_ineq, v_ineq, V_eq, v_eq, c, lb, ub) -> \
            Tuple[sparse.csr_matrix, Tuple, sparse.csr_matrix, Tuple, Tuple, Tuple, sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
        numr = len(self.model.reactions)
        # initialize matices (if not provided in function call)
        if V_ineq==[]: V_ineq = sparse.csr_matrix((0,numr))
        if V_eq==[]:   V_eq   = sparse.csr_matrix((0,numr))
        if lb==[]:     lb = [v.lower_bound for v in self.model.reactions]
        if ub==[]:     ub = [v.upper_bound for v in self.model.reactions]
        if c ==[]:     c  = [i.objective_coefficient for i in self.model.reactions]
        S = sparse.csr_matrix(cobra.util.create_stoichiometric_matrix(self.model))
        # fill matrices
        A_eq = sparse.vstack((S,V_eq))
        b_eq = [0]*S.shape[0]+v_eq
        A_ineq = V_ineq
        b_ineq = v_ineq
        z_map_vars        = sparse.identity(numr,'d',format="csr")
        z_map_constr_eq   = sparse.csr_matrix((self.num_z,A_eq.shape[0]))
        z_map_constr_ineq = sparse.csr_matrix((self.num_z,A_ineq.shape[0]))
        A_ineq, b_ineq, lb, ub, z_map_constr_ineq = self.prevent_boundary_knockouts(A_ineq, b_ineq, lb, ub, z_map_constr_ineq, z_map_vars)
        return A_ineq, b_ineq, A_eq, b_eq, c, lb, ub, z_map_constr_ineq, z_map_constr_eq, z_map_vars

    def dualize(self,A_ineq_p, b_ineq_p, A_eq_p, b_eq_p, c_p, lb_p, ub_p, z_map_constr_ineq_p, z_map_constr_eq_p, z_map_vars_p) -> \
            Tuple[sparse.csr_matrix, Tuple, sparse.csr_matrix, Tuple, Tuple, Tuple, sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
            # Translates a primal system to a dual system. The primal system must
            # be given in the standard form: A_ineq x <= b_ineq, A_eq x = b_eq, lb <= x < ub, min{c'x}.
            #
            # Variables translate to constraints:
            # x={R} ->   =
            # x>=0  ->  >= (new constraint is multiplied with -1 to translate to <=
            #               e.g. -A_i' y <= -c_i)
            # x<=0  ->  <=
            # Constraints translate to variables:
            # =     ->   y={R}
            # <=    ->   y>=0
            #
            #
            
            #
            # Consider that the following is not implemented:
            # In the case of (1) A x = b, (2) x={R}, (3) b~=0, Farkas' lemma is special,
            # because b'y ~= 0 is required to make the primal infeasible instead of b'y < 0.
            # 1. This does not occur very often.
            # 2. Splitting the equality into two inequalities that translate to y>=0
            #    would be posible, and yield b'y < 0 in the farkas' lemma.
            # Maybe splitting is required, but I actually don't think so. Using the
            # special case of b'y < 0 for b'y ~= 0 should be enough.

            if z_map_vars_p == []:
                z_map_vars_p = sparse.csr_matrix((self.num_z,A_ineq_p.shape[1]))
            if z_map_constr_eq_p == []:
                z_map_constr_eq_p = sparse.csr_matrix((self.num_z,A_eq_p.shape[0]))
            if z_map_constr_ineq_p == []:
                z_map_constr_ineq_p = sparse.csr_matrix((self.num_z,A_ineq_p.shape[0]))
            
            # knockouts of variables and constraints must not overlap in the problem matrix
            if not len(A_eq_p[[True if i in z_map_constr_eq_p.nonzero()[1] else False for i in range(0,A_eq_p.shape[0])],:]\
                            [:,[True if i in z_map_vars_p.nonzero()[1] else False for i in range(0,A_eq_p.shape[1])]].nonzero()[0]) == 0 \
                or not len(A_ineq_p[[True if i in z_map_constr_ineq_p.nonzero()[1] else False for i in range(0,A_ineq_p.shape[0])],:]\
                            [:,[True if i in z_map_vars_p.nonzero()[1] else False for i in range(0,A_ineq_p.shape[1])]].nonzero()[0]) == 0:
                raise Exception("knockouts of variables and constraints must not overlap in the problem matrix. Something went wrong during the construction of the primal problem.")
            
            numr = len(self.model.reactions)
            if c_p == []:
                c_p = [0]*numr

            print('debug')
            
            # # Translate inhomogenous bounds into inequality constraints
            # lb_inh_bounds = lb_p~=0 & ~isinf(lb_p);
            # ub_inh_bounds = ub_p~=0 & ~isinf(ub_p);
            # x_geq0 = lb_p>=0 & ub_p >0;
            # x_eR =   lb_p< 0 & ub_p >0;
            # x_leq0 = lb_p< 0 & ub_p<=0;
            
            # LB = full(sparse(1:sum(lb_inh_bounds),find(lb_inh_bounds),-1,sum(lb_inh_bounds),size(A_ineq_p,2)));
            # UB = full(sparse(1:sum(ub_inh_bounds),find(ub_inh_bounds), 1,sum(ub_inh_bounds),size(A_ineq_p,2)));
            # A_ineq_p = [ A_ineq_p; ...
            #     LB; ...
            #     UB ];
            # b_ineq_p = [ b_ineq_p; ...
            #     -lb_p(lb_inh_bounds); ...
            #     ub_p(ub_inh_bounds)];
            
            # # Translate into dual system
            # A_ineq = [-A_eq_p(:,x_geq0)', -A_ineq_p(:,x_geq0)'; A_eq_p(:,x_leq0)', A_ineq_p(:,x_leq0)'];
            # b_ineq = [c_p(x_geq0)' ; -c_p(x_leq0)'];
            # A_eq = [A_eq_p(:,x_eR)', A_ineq_p(:,x_eR)'];
            # b_eq = c_p(x_eR)';
            # lb = [-inf(size(A_eq_p,1),1); zeros(size(A_ineq_p,1),1)];
            # ub =   inf(size(A_eq_p,1)+size(A_ineq_p,1),1);
            # c  = [b_eq_p; b_ineq_p]';
            
            # # translate mapping of z-variables to rows instead of columns
            # z_map_constr_ineq = [z_map_vars_p(:, x_geq0), z_map_vars_p(:, x_leq0)];
            # z_map_constr_eq   = z_map_vars_p(:,x_eR);
            # z_map_vars        = [z_map_constr_eq_p, z_map_constr_ineq_p, zeros(obj.num_z, size([LB;UB],1))];
            
            A_ineq, b_ineq, A_eq, b_eq, lb, ub, z_map_constr_ineq, z_map_constr_eq = self.reassign_lb_ub_from_ineq(A_ineq, b_ineq, A_eq, b_eq, lb, ub, z_map_constr_ineq, z_map_constr_eq, z_map_vars);

    def prevent_boundary_knockouts(self, A_ineq, b_ineq, lb, ub, z_map_constr_ineq, z_map_vars) -> \
            Tuple[sparse.csr_matrix, Tuple, Tuple, Tuple, sparse.csr_matrix]:
        numr = A_ineq.shape[0]
        for i in range(0,A_ineq.shape[0]):
            if any(z_map_vars[:,0]) and lb[i]>0:
                A_ineq = sparse.vstack((A_ineq,sparse.csr_matrix(([-1],([0],[i])),shape=(1,numr))))
                b_ineq += [-lb[i]]
                z_map_constr_ineq = sparse.hstack((z_map_constr_ineq,sparse.csr_matrix((self.num_z,1))))
                lb[i] = 0
            if any(z_map_vars[:,0]) and ub[i]<0:
                A_ineq = sparse.vstack((A_ineq,sparse.csr_matrix(([1],([0],[i])),shape=(1,numr))))
                b_ineq += [ub[i]]
                z_map_constr_ineq = sparse.hstack((z_map_constr_ineq,sparse.csr_matrix((self.num_z,1))))
                ub[i] = 0
        return A_ineq, b_ineq, lb, ub, z_map_constr_ineq

    def reassign_lb_ub_from_ineq(self,A_ineq, b_ineq, A_eq, b_eq, lb, ub, z_map_constr_ineq, z_map_constr_eq, z_map_vars) -> \
            Tuple[sparse.csr_matrix, Tuple, sparse.csr_matrix, Tuple, Tuple, Tuple, sparse.csr_matrix, sparse.csr_matrix]:
            # This function searches for bounds on variables (single entries in A_ineq or A_eq). 
            # Returns: A_ineq, b_ineq, A_eq, b_eq, lb, ub, z_map_constr_ineq, z_map_constr_eq
            # 
            # Those constraints are removed and instead put into lb and ub. This
            # is useful to avoid unnecessary bigM constraints on variable
            # bounds when true upper and lower bounds exist instead.
            # To avoid interference with the knock-out logic, negative ub 
            # and positive ub are not translated.
            lb = [[l] for l in lb]
            ub = [[u] for u in ub]

            # translate entries to lb or ub
            # find all entries in A_ineq
            numr = A_ineq.shape[1]
            row_ineq = A_ineq.nonzero()[0]
            # filter for rows with only one entry 
            var_bound_constraint_ineq = [i for i in row_ineq if list(row_ineq).count(i)==1]
            # exclude knockable constraints
            var_bound_constraint_ineq = [i for i in var_bound_constraint_ineq if i not in z_map_constr_ineq.nonzero()[1]]
            # retrieve all bounds from inequality constraints
            for i in var_bound_constraint_ineq:
                idx_r = A_ineq[i,:].nonzero()[1] # get reaction from constraint (column of entry)
                if A_ineq[i,idx_r]>0: # upper bound constraint
                    ub[i] += [b_ineq[i]/A_ineq[i,idx_r].toarray()[0][0]]
                else: # lower bound constraint
                    lb[i] += [b_ineq[i]/A_ineq[i,idx_r].toarray()[0][0]]

            # find all entries in A_eq
            row_eq = A_eq.nonzero()[0]
            # filter for rows with only one entry 
            var_bound_constraint_eq = [i for i in row_eq if list(row_eq).count(i)==1]
            # exclude knockable constraints
            var_bound_constraint_eq = [i for i in var_bound_constraint_eq if i not in z_map_constr_eq.nonzero()[1]]
            # retrieve all bounds from equality constraints
            # and partly set lb or ub derived from equality constraints, for instance:
            # If x =  5, set ub = 5 and keep the inequality constraint -x <= -5.
            # If x = -5, set lb =-5 and keep the inequality constraint  x <=  5.
            A_ineq_new = sparse.csr_matrix((0,numr))
            b_ineq_new = []
            for i in var_bound_constraint_eq:
                idx_r = A_eq[i,:].nonzero()[1] # get reaction from constraint (column of entry)
                if any(z_map_vars[:,idx_r]): # if reaction is knockable
                    if A_eq[i,idx_r]*b_eq[i]>0: # upper bound constraint
                        ub[i] += [b_eq[i]/A_eq[i,idx_r].toarray()[0][0]]
                        A_ineq_new = sparse.vstack((A_ineq_new,-A_eq[i,:]))
                        b_ineq_new += [-b_eq[i]]
                    elif A_eq[i,idx_r]*b_eq[i]<0: # lower bound constraint
                        lb[i] += [b_eq[i]/A_eq[i,idx_r].toarray()[0][0]]
                        A_ineq_new = sparse.vstack((A_ineq_new,A_eq[i,:]))
                        b_ineq_new += [b_eq[i]]
                    else:
                        ub[i] += [0]
                        lb[i] += [0]
                else:
                    lb[i] += [b_eq[i]/A_eq[i,idx_r].toarray()[0][0]]
                    ub[i] += [b_eq[i]/A_eq[i,idx_r].toarray()[0][0]]
            # set tightest bounds (avoid inf)
            lb = [max([i for i in l if not np.isinf(i)]+[np.nan]) for l in lb]
            ub = [min([i for i in u if not np.isinf(i)]+[np.nan]) for u in ub]
            # set if only if no other bound remains
            lb = [-np.inf if np.isnan(l) else l for l in lb]
            ub = [ np.inf if np.isnan(u) else u for u in ub]

            # check if bounds are consistent
            if any(np.greater(lb,ub)):
                raise Exception("There is a lower bound that is greater than its upper bound counterpart.")

            # remove constraints that became redundant
            A_ineq = A_ineq[[False if i in var_bound_constraint_ineq else True for i in range(0,A_ineq.shape[0])]]
            b_ineq = [b_ineq[i] for i in var_bound_constraint_ineq]
            z_map_constr_ineq[:,[False if i in var_bound_constraint_ineq else True for i in range(0,A_ineq.shape[0])]]
            A_eq = A_eq[[False if i in var_bound_constraint_eq else True for i in range(0,A_eq.shape[0])]]
            b_eq = [b_eq[i] for i in var_bound_constraint_eq]
            z_map_constr_eq[:,[False if i in var_bound_constraint_eq else True for i in range(0,A_eq.shape[0])]]
            # add equality constraints that transformed to inequality constraints
            A_ineq = sparse.vstack((A_ineq,A_ineq_new))
            b_ineq += b_ineq_new
            z_map_constr_ineq = sparse.hstack((z_map_constr_ineq,sparse.csr_matrix((self.num_z,A_ineq_new.shape[0]))))
            return A_ineq, b_ineq, A_eq, b_eq, lb, ub, z_map_constr_ineq, z_map_constr_eq

    def lineq2mat(self,equations) -> Tuple[sparse.csr_matrix, Tuple, sparse.csr_matrix, Tuple]:
        numr = len(self.model.reactions)
        A_ineq = sparse.csr_matrix((0,numr))
        b_ineq = []
        A_eq   = sparse.csr_matrix((0,numr))
        b_eq   = []
        for equation in equations:
            try:
                lhs, rhs = re.split('<=|=|>=',equation)
                eq_sign = re.search('<=|>=|=',equation)[0]
                rhs = float(rhs)
            except:
                raise Exception("Equations must contain exactly one (in)equality sign: <=,=,>=. Right hand side must be a float number.")
            A = self.linexpr2mat(lhs)
            if eq_sign == '=':
                A_eq = sparse.vstack((A_eq,A))
                b_eq+=[rhs]
            elif eq_sign == '<=':
                A_ineq = sparse.vstack((A_ineq, A))
                b_ineq+=[rhs]
            elif eq_sign == '>=':
                A_ineq = sparse.vstack((A_ineq,-A))
                b_ineq+=[-rhs]
        return A_ineq, b_ineq, A_eq, b_eq
    
    def linexpr2mat(self,lhs) -> sparse.csr_matrix:
        # linexpr2mat translates the left hand side of a linear expression into a matrix
        #
        # e.g.: Model with reactions R1, R2, R3, R4
        #       Expression: '2 R3 - R1'
        #     translates into list:
        #       A = [-1 0 2 0]
        # 
        A = sparse.csr_matrix((1,len(self.model.reactions)))
        # split expression into parts and strip away special characters
        ridx = [re.sub(r'^(\s|-|\+|\.|\()*|(\s|-|\+|\.|\))*$','',part) for part in lhs.split()]
        # identify reaction identifiers by comparing with models reaction list
        ridx = [r for r in ridx if r in self.model.reactions.list_attr('id')]
        if not len(ridx) == len(set(ridx)): # check for duplicates
            raise Exception("Reaction identifiers may only occur once in each linear expression.")
        # iterate through reaction identifiers and retrieve coefficients from linear expression
        for rid in ridx:
            coeff = re.search('(\s|^)(\s|\d|-|\+|\.)*?(?=' +rid+ '(\s|$))',lhs)[0]
            coeff = re.sub('\s','',coeff)
            if coeff in ['','+']:
                coeff = 1
            if coeff == '-':
                coeff = -1
            else:
                coeff = float(coeff)
            A[0,self.model.reactions.list_attr('id').index(rid)] = coeff
        return A

    class Indicator_constraint:
        def __init__(self, ic_binv, ic_A_ineq, ic_b_ineq, ic_sense, ic_indicval):
            self.ic_binv     = ic_binv # index of binary variable
            self.ic_A_ineq   = ic_A_ineq # CPLEX: lin_expr,   left hand side coefficient row for indicator constraint
            self.ic_b_ineq   = ic_b_ineq # right hand side for indicator constraint
            self.ic_sense    = ic_sense # sense of the indicator constraint can be 'L', 'E', 'G' (lower-equal, equal, greater-equal)
            self.ic_indicval = ic_indicval # value the binary variable takes when constraint is fulfilled

