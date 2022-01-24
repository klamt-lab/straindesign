import numpy as np
from scipy import sparse
import cobra
import re
from typing import Dict, List, Tuple, Union, FrozenSet
from mcs import mcs_module, solver_interface, indicator_constraints
import ray


class StrainDesignMILPBuilder:
    """Class for computing Minimal Cut Sets (MCS)"""

    def __init__(self, model: cobra.Model, mcs_modules: List[mcs_module.MCS_Module], *args, **kwargs):
        allowed_keys = {'ko_cost', 'ki_cost', 'solver', 'max_cost', 'M'}
        # set all keys passed in kwargs
        for key, value in dict(kwargs).items():
            if key in allowed_keys:
                setattr(self, key, value)
            else:
                raise Exception("Key " + key + " is not supported.")
        # set all remaining keys to None
        for key in allowed_keys:
            if key not in dict(kwargs).keys():
                setattr(self, key, None)

        #     print ("%s == %s" %(key, value))
        # ko_cost: , ki_cost: Dict[str, float]={}, M=None, max_cost=None, solver=None,
        # select solver
        avail_solvers = list(cobra.util.solvers.keys())
        try:
            import pyscipopt
            avail_solvers += ['scip']
        except:
            pass
        if self.solver is None:
            if 'cplex' in avail_solvers:
                self.solver = 'cplex'
            elif 'gurobi' in avail_solvers:
                self.solver = 'gurobi'
            elif 'scip' in avail_solvers:
                self.solver = 'scip'
            else:
                self.solver = 'glpk'
        elif self.solver not in avail_solvers:
            raise Exception("Selected solver is not installed / set up correctly.")
        else:
            self.solver = self.solver
        if self.solver in ['scip', 'gurobi', 'glpk']:
            raise Exception(self.solver + ' not yet supported')
        elif self.M is None and self.solver == 'glpk':
            print('GLPK only supports MCS computation with the bigM method. Default: M=1000')
            self.M = 1000.0
        elif self.M is None:
            self.M = np.inf
        # the matrices in mcs_modules, ko_cost and ki_cost should be np.array or scipy.sparse (csr, csc, lil) format
        self.model = model
        reac_ids = model.reactions.list_attr("id")
        numr = len(model.reactions)
        # Create vectors for ko_cost, ki_cost, inverted bool-vars and non-targetable bools
        # If no knockable reactions are assigned, assume all are KO-able.
        # Generally, KIs overwrite KOs
        if self.ko_cost is None:
            self.ko_cost = {rid: 1 for rid in reac_ids}
        if self.ki_cost is None:
            self.ki_cost = {}
        self.ko_cost = [self.ko_cost.get(key) if (key in self.ko_cost.keys()) else np.nan for key in reac_ids]
        self.ki_cost = [self.ki_cost.get(key) if (key in self.ki_cost.keys()) else np.nan for key in reac_ids]
        self.ko_cost = [self.ko_cost[i] if np.isnan(self.ki_cost[i]) else np.nan for i in range(numr)]
        self.num_z = numr
        self.cost = [i for i in self.ko_cost]
        for i in [i for i, x in enumerate(self.ki_cost) if not np.isnan(x)]:
            self.cost[i] = self.ki_cost[i]
        self.z_inverted = [not np.isnan(x) for x in self.ki_cost]
        self.z_non_targetable = [np.isnan(x) for x in self.cost]
        for i in [i for i, x in enumerate(self.cost) if np.isnan(x)]:
            self.cost[i] = 0
        # Prepare top lines of MILP (sum of KOs below and above threshold) and objective function
        self.A_ineq = sparse.csr_matrix([[-i for i in self.cost], self.cost])
        if self.max_cost is None:
            self.b_ineq = [0, np.sum(np.abs(self.cost))]
        else:
            self.b_ineq = [0, self.max_cost]
        self.z_map_constr_ineq = sparse.csc_matrix((numr, 2))
        self.rownames_ineq = ['sum_z_min', 'sum_z_max']
        self.colnames = ['z' + str(i) for i in range(0, numr)]
        self.c = self.cost.copy()
        self.lb = [0] * numr
        self.ub = [1 - float(i) for i in self.z_non_targetable]
        self.idx_z = [i for i in range(0, numr)]
        # Initialize also empty equality matrix
        self.A_eq = sparse.csc_matrix((0, numr))
        self.b_eq = []
        self.z_map_constr_eq = sparse.csc_matrix((numr, 0))
        self.rownames_eq = []
        self.num_modules = 0
        self.indicators = []  # Add instances of the class 'Indicator_constraint' later
        # Initialize association between z and variables and variables
        self.z_map_vars = sparse.csc_matrix((numr, numr))
        for i in range(0, len(mcs_modules)):
            self.addModule(mcs_modules[i])

        # Assign knock-ins/outs correctly by taking into account z_inverted
        # invert *(-1) rows in z_map_constr_eq, z_map_constr_ineq, z_map_vars
        # where there are "knock-ins" / additions
        # make knock-in/out matrix
        z_kos_kis = [
            1 if (not self.z_non_targetable[i]) and (not self.z_inverted[i]) else -1 if self.z_inverted[i] else 0 for i
            in range(0, self.num_z)]
        z_kos_kis = sparse.diags(z_kos_kis)
        self.z_map_constr_ineq = z_kos_kis * self.z_map_constr_ineq
        self.z_map_constr_eq = z_kos_kis * self.z_map_constr_eq
        self.z_map_vars = z_kos_kis * self.z_map_vars

        # Save continous part of MILP for easy MCS validation
        cont_vars = [i for i in range(0, self.A_ineq.shape[1]) if not i in self.idx_z]
        self.cont_MILP = ContMILP(self.A_ineq[:, cont_vars],
                                  self.b_ineq.copy(),
                                  self.A_eq[:, cont_vars],
                                  self.b_eq.copy(),
                                  [self.lb[i] for i in cont_vars],
                                  [self.ub[i] for i in cont_vars],
                                  [self.c[i] for i in cont_vars],
                                  self.z_map_constr_ineq,
                                  self.z_map_constr_eq,
                                  self.z_map_vars[:, cont_vars])

        # 4. Link LP module to z-variables
        self.link_z()

        self.vtype = 'B' * self.num_z + 'C' * (self.z_map_vars.shape[1] - self.num_z)

    def addModule(self, mcs_module):
        # Generating LP and z-linking-matrix for each module
        #
        # Modules to describe desired or undesired flux states for MCS strain design.
        # mcs_module needs to be an Instance of the class 'MCS_Module'
        #
        # There are three kinds of flux states that can be described
        # 1. The wild type model, constrained with additional inequalities:
        #    e.g.: T v <= t.
        # 2. The wild type model at a specific optimum, constrained with additional inequalities
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
        A_ineq_p, b_ineq_p, A_eq_p, b_eq_p, c_p, lb_p, ub_p, z_map_constr_ineq_p, z_map_constr_eq_p, z_map_vars_p = self.build_primal(
            V_ineq, v_ineq, V_eq, v_eq, [], lb, ub)
        # 3. Prepare module as target or desired
        if mcs_module.module_sense == 'desired':
            A_ineq_i, b_ineq_i, A_eq_i, b_eq_i, lb_i, ub_i, z_map_constr_ineq_i, z_map_constr_eq_i = self.reassign_lb_ub_from_ineq(
                A_ineq_p, b_ineq_p, A_eq_p, b_eq_p, lb_p, ub_p, z_map_constr_ineq_p, z_map_constr_eq_p, z_map_vars_p)
            z_map_vars_i = z_map_vars_p
        elif mcs_module.module_sense == 'target':
            c_p = [0] * len(c_p)
            A_ineq_d, b_ineq_d, A_eq_d, b_eq_d, c_d, lb_i, ub_i, z_map_constr_ineq_d, z_map_constr_eq_d, z_map_vars_i = self.dualize(
                A_ineq_p, b_ineq_p, A_eq_p, b_eq_p, c_p, lb_p, ub_p, z_map_constr_ineq_p, z_map_constr_eq_p,
                z_map_vars_p)
            A_ineq_i, b_ineq_i, A_eq_i, b_eq_i, z_map_constr_ineq_i, z_map_constr_eq_i = self.dual_2_farkas(A_ineq_d,
                                                                                                            b_ineq_d,
                                                                                                            A_eq_d,
                                                                                                            b_eq_d, c_d,
                                                                                                            z_map_constr_ineq_d,
                                                                                                            z_map_constr_eq_d)

        # 3. Add module to global MILP
        self.z_map_constr_ineq = sparse.hstack((self.z_map_constr_ineq, z_map_constr_ineq_i)).tocsc()
        self.z_map_constr_eq = sparse.hstack((self.z_map_constr_eq, z_map_constr_eq_i)).tocsc()
        self.z_map_vars = sparse.hstack((self.z_map_vars, z_map_vars_i)).tocsc()
        self.A_ineq = sparse.bmat([[self.A_ineq, None], [None, A_ineq_i]]).tocsr()
        self.b_ineq += b_ineq_i
        self.A_eq = sparse.bmat([[self.A_eq, None], [None, A_eq_i]]).tocsr()
        self.b_eq += b_eq_i
        self.c += [0] * A_ineq_i.shape[1]
        self.lb += lb_i
        self.ub += ub_i
        self.rownames_ineq += [str(self.num_modules) + '_mod_ineq_' + str(i) for i in range(0, A_ineq_i.shape[0])]
        self.rownames_eq += [str(self.num_modules) + '_mod_eq_' + str(i) for i in range(0, A_eq_i.shape[0])]
        self.colnames += [str(self.num_modules) + '_mod_var_' + str(i) for i in range(0, A_ineq_i.shape[1])]

    # Builds primal LP problem for a module in the
    # standard form: A_ineq x <= b_ineq, A_eq x = b_eq, lb <= x <= ub, min{c'x}.
    # returns A_ineq, b_ineq, A_eq, b_eq, c, lb, ub, z_map_constr_ineq, z_map_constr_eq, z_map_vars
    def build_primal(self, V_ineq, v_ineq, V_eq, v_eq, c, lb, ub) -> \
            Tuple[
                sparse.csr_matrix, Tuple, sparse.csr_matrix, Tuple, Tuple, Tuple, sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
        numr = len(self.model.reactions)
        # initialize matices (if not provided in function call)
        if V_ineq == []: V_ineq = sparse.csr_matrix((0, numr))
        if V_eq == []:   V_eq = sparse.csr_matrix((0, numr))
        if lb == []:     lb = [v.lower_bound for v in self.model.reactions]
        if ub == []:     ub = [v.upper_bound for v in self.model.reactions]
        if c == []:     c = [i.objective_coefficient for i in self.model.reactions]
        S = sparse.csr_matrix(cobra.util.create_stoichiometric_matrix(self.model))
        # fill matrices
        A_eq = sparse.vstack((S, V_eq))
        b_eq = [0] * S.shape[0] + v_eq
        A_ineq = V_ineq
        b_ineq = v_ineq
        z_map_vars = sparse.identity(numr, 'd', format="csc")
        z_map_constr_eq = sparse.csc_matrix((self.num_z, A_eq.shape[0]))
        z_map_constr_ineq = sparse.csc_matrix((self.num_z, A_ineq.shape[0]))
        A_ineq, b_ineq, lb, ub, z_map_constr_ineq = self.prevent_boundary_knockouts(A_ineq, b_ineq, lb, ub,
                                                                                    z_map_constr_ineq, z_map_vars)
        return A_ineq, b_ineq, A_eq, b_eq, c, lb, ub, z_map_constr_ineq, z_map_constr_eq, z_map_vars

    def dualize(self, A_ineq_p, b_ineq_p, A_eq_p, b_eq_p, c_p, lb_p, ub_p, z_map_constr_ineq_p, z_map_constr_eq_p,
                z_map_vars_p) -> \
            Tuple[
                sparse.csr_matrix, Tuple, sparse.csr_matrix, Tuple, Tuple, Tuple, sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
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
        numr = A_ineq_p.shape[0]

        if z_map_vars_p == []:
            z_map_vars_p = sparse.csc_matrix((self.num_z, A_ineq_p.shape[1]))
        if z_map_constr_eq_p == []:
            z_map_constr_eq_p = sparse.csc_matrix((self.num_z, A_eq_p.shape[0]))
        if z_map_constr_ineq_p == []:
            z_map_constr_ineq_p = sparse.csc_matrix((self.num_z, A_ineq_p.shape[0]))

        # knockouts of variables and constraints must not overlap in the problem matrix
        if not len(
                A_eq_p[[True if i in z_map_constr_eq_p.nonzero()[1] else False for i in range(0, A_eq_p.shape[0])], :] \
                        [:,
                [True if i in z_map_vars_p.nonzero()[1] else False for i in range(0, A_eq_p.shape[1])]].nonzero()[
                    0]) == 0 \
                or not len(
            A_ineq_p[[True if i in z_map_constr_ineq_p.nonzero()[1] else False for i in range(0, A_ineq_p.shape[0])], :] \
                    [:,
            [True if i in z_map_vars_p.nonzero()[1] else False for i in range(0, A_ineq_p.shape[1])]].nonzero()[
                0]) == 0:
            raise Exception(
                "knockouts of variables and constraints must not overlap in the problem matrix. Something went wrong during the construction of the primal problem.")

        numr = len(self.model.reactions)
        if c_p == []:
            c_p = [0] * numr

        # Translate inhomogenous bounds into inequality constraints
        lb_inh_bounds = [i for i in np.nonzero(lb_p)[0] if not np.isinf(i)]
        ub_inh_bounds = [i for i in np.nonzero(ub_p)[0] if not np.isinf(i)]
        x_geq0 = np.nonzero(np.greater_equal(lb_p, 0) & np.greater(ub_p, 0))[0]
        x_eR = np.nonzero(np.greater(0, lb_p) & np.greater(ub_p, 0))[0]
        x_leq0 = np.nonzero(np.greater(0, lb_p) & np.greater_equal(0, ub_p))[0]

        LB = sparse.csr_matrix((len(lb_inh_bounds) * [-1], (range(0, len(lb_inh_bounds)), lb_inh_bounds)),
                               shape=(len(lb_inh_bounds), numr))
        UB = sparse.csr_matrix((len(ub_inh_bounds) * [1], (range(0, len(ub_inh_bounds)), ub_inh_bounds)),
                               shape=(len(ub_inh_bounds), numr))
        A_ineq_p = sparse.vstack((A_ineq_p, LB, UB))
        b_ineq_p = b_ineq_p + [-lb_p[i] for i in lb_inh_bounds] + [ub_p[i] for i in ub_inh_bounds]

        # Translate into dual system
        # Transpose parts of matrices with variables >=0,<=0 to put them into A_ineq
        A_ineq = sparse.vstack((sparse.hstack((np.transpose(-A_eq_p[:, x_geq0]), np.transpose(-A_ineq_p[:, x_geq0]))), \
                                sparse.hstack(
                                    (np.transpose(A_eq_p[:, x_leq0]), np.transpose(A_ineq_p[:, x_leq0]))))).tocsr()
        b_ineq = [c_p[i] for i in x_geq0] + [-c_p[i] for i in x_leq0]
        A_eq = sparse.hstack((np.transpose(A_eq_p[:, x_eR]), np.transpose(A_ineq_p[:, x_eR]))).tocsr()
        b_eq = [c_p[i] for i in x_eR]
        lb = [-np.inf] * A_eq_p.shape[0] + [0] * A_ineq_p.shape[0]
        ub = [np.inf] * (A_eq_p.shape[0] + A_ineq_p.shape[0])
        c = b_eq_p + b_ineq_p

        # translate mapping of z-variables to rows instead of columns
        z_map_constr_ineq = sparse.hstack((z_map_vars_p[:, x_geq0], z_map_vars_p[:, x_leq0])).tocsc()
        z_map_constr_eq = z_map_vars_p[:, x_eR]
        z_map_vars = sparse.hstack((z_map_constr_eq_p, z_map_constr_ineq_p,
                                    sparse.csc_matrix((self.num_z, len(lb_inh_bounds) + len(ub_inh_bounds))))).tocsc()

        A_ineq, b_ineq, A_eq, b_eq, lb, ub, z_map_constr_ineq, z_map_constr_eq = self.reassign_lb_ub_from_ineq(A_ineq,
                                                                                                               b_ineq,
                                                                                                               A_eq,
                                                                                                               b_eq, lb,
                                                                                                               ub,
                                                                                                               z_map_constr_ineq,
                                                                                                               z_map_constr_eq,
                                                                                                               z_map_vars)
        return A_ineq, b_ineq, A_eq, b_eq, c, lb, ub, z_map_constr_ineq, z_map_constr_eq, z_map_vars

    def dual_2_farkas(self, A_ineq, b_ineq, A_eq, b_eq, c_dual, z_map_constr_ineq, z_map_constr_eq) -> \
            Tuple[sparse.csr_matrix, Tuple, sparse.csr_matrix, Tuple, sparse.csr_matrix, sparse.csr_matrix]:
        # add constraint b_prim'y or (c_dual'*y) <= -1;
        A_ineq = sparse.vstack((A_ineq, c_dual)).tocsr()
        b_ineq += [-1]
        z_map_constr_ineq = sparse.hstack((z_map_constr_ineq, sparse.csr_matrix((self.num_z, 1)))).tocsc()
        # it would also be possible (but ofc not necessary) to force (c_dual*y) == -1; instead
        # A_eq = sparse.vstack((A_eq,c_dual)).tocsr()
        # b_eq += [-1]
        # z_map_constr_eq = sparse.hstack((z_map_constr_eq,sparse.csr_matrix((self.num_z,1)))).tocsc()
        return A_ineq, b_ineq, A_eq, b_eq, z_map_constr_ineq, z_map_constr_eq

    def prevent_boundary_knockouts(self, A_ineq, b_ineq, lb, ub, z_map_constr_ineq, z_map_vars) -> \
            Tuple[sparse.csr_matrix, Tuple, Tuple, Tuple, sparse.csr_matrix]:
        numr = A_ineq.shape[0]
        for i in range(0, A_ineq.shape[0]):
            if any(z_map_vars[:, 0]) and lb[i] > 0:
                A_ineq = sparse.vstack((A_ineq, sparse.csr_matrix(([-1], ([0], [i])), shape=(1, numr))))
                b_ineq += [-lb[i]]
                z_map_constr_ineq = sparse.hstack((z_map_constr_ineq, sparse.csc_matrix((self.num_z, 1))))
                lb[i] = 0
            if any(z_map_vars[:, 0]) and ub[i] < 0:
                A_ineq = sparse.vstack((A_ineq, sparse.csr_matrix(([1], ([0], [i])), shape=(1, numr))))
                b_ineq += [ub[i]]
                z_map_constr_ineq = sparse.hstack((z_map_constr_ineq, sparse.csc_matrix((self.num_z, 1))))
                ub[i] = 0
        return A_ineq, b_ineq, lb, ub, z_map_constr_ineq

    def link_z(self):
        # (1) Assign knock-ins/outs correctly
        # (2) Translate equality-KOs to inequality-KOs 
        # (3) Translate variable-KOs to inequality-KOs 
        # (4) Link z-variables with bigM constraints, where available
        # (5) Translate remaining inequalities back to equalities if applicable
        # (6) Add indicator constraints
        # (7) Remove knockable equalities from static problem
        #

        # 1. Split knockable equality constraints into foward and reverse direction
        knockable_constr_eq = self.z_map_constr_eq.nonzero()  # first array: z, second array: eq constr
        eq_constr_A = sparse.vstack((self.A_eq[knockable_constr_eq[1], :], -self.A_eq[knockable_constr_eq[1], :]))
        eq_constr_b = [self.b_eq[i] for i in knockable_constr_eq[1]] + [-self.b_eq[i] for i in knockable_constr_eq[1]]
        z_eq = self.z_map_constr_eq[:, tuple(knockable_constr_eq[1]) * 2]
        # Add knockable inequalities to global A_ineq matrix
        self.A_ineq = sparse.vstack((self.A_ineq, eq_constr_A)).tocsr()
        self.b_ineq += eq_constr_b
        self.z_map_constr_ineq = sparse.hstack((self.z_map_constr_ineq, z_eq)).tocsc()
        # Remove knockable equalities from A_eq
        n_rows_eq = self.A_eq.shape[0]
        self.A_eq = self.A_eq[[False if i in knockable_constr_eq[1] else True for i in range(0, n_rows_eq)]]
        self.b_eq = [self.b_eq[i] for i in range(0, len(self.b_eq)) if i not in knockable_constr_eq[1]]
        self.z_map_constr_eq = self.z_map_constr_eq[:,
                               [False if i in knockable_constr_eq[1] else True for i in range(0, n_rows_eq)]]

        # 2. Translate all variable knockouts to inequality knockouts
        numvars = self.A_ineq.shape[1]
        knockable_vars = self.z_map_vars.nonzero()  # first array: z, second array: x
        knockable_vars_geq0 = [i for i in knockable_vars[1] if self.ub[i] > 0]
        knockable_vars_leq0 = [i for i in knockable_vars[1] if self.lb[i] < 0]
        ub_constr_A = sparse.csr_matrix(
            ([1] * len(knockable_vars_geq0), (range(0, len(knockable_vars_geq0)), knockable_vars_geq0)),
            [len(knockable_vars_geq0), numvars])
        ub_constr_b = [0] * len(knockable_vars_geq0)
        lb_constr_A = sparse.csr_matrix(
            ([-1] * len(knockable_vars_leq0), (range(0, len(knockable_vars_leq0)), knockable_vars_leq0)),
            [len(knockable_vars_leq0), numvars])
        lb_constr_b = [0] * len(knockable_vars_leq0)
        bnd_constr_A = sparse.vstack((ub_constr_A, lb_constr_A)).tocsr()
        bnd_constr_b = ub_constr_b + lb_constr_b
        var_kos = [knockable_vars[0][(knockable_vars[1] == i).nonzero()[0][0]] for i in
                   knockable_vars_geq0 + knockable_vars_leq0]
        z_lb_ub = -self.z_map_vars[:, knockable_vars_geq0 + knockable_vars_leq0]
        # add constraints to main problem
        self.A_ineq = sparse.vstack((self.A_ineq, bnd_constr_A)).tocsr()
        self.b_ineq += bnd_constr_b
        self.z_map_constr_ineq = sparse.hstack((self.z_map_constr_ineq, z_lb_ub)).tocsc()

        # 3. Use LP to identify M-values for knockable constraints
        #    For this purpose, first construct a most relaxed LP-model (use all possible constraint-KOs, no possible var-KOs)
        knockable_constr_ineq = set(self.z_map_constr_ineq.nonzero()[1])

        cont_vars = [False if i in self.idx_z else True for i in range(0, numvars)]
        M_A_ineq = self.A_ineq[[False if i in knockable_constr_ineq else True for i in range(0, self.A_ineq.shape[0])],
                   :][:, cont_vars]
        M_b_ineq = [self.b_ineq[i] for i in range(0, self.A_ineq.shape[0]) if i not in knockable_constr_ineq]
        M_A_eq = self.A_eq[:, cont_vars]
        M_b_eq = self.b_eq.copy()
        M_lb = [self.lb[i] for i in np.nonzero(cont_vars)[0]]
        M_ub = [self.ub[i] for i in np.nonzero(cont_vars)[0]]
        # M_A contains a list of all knockable constraints. We need to maximize their value (M_A(i)*x) to get a good M
        # M_A(i)*x - z*M <= b
        # M*x <= b+z*M
        #     or
        # M_A(i)*x + z*M <= b + M
        # b is the right hand side value
        M_A = self.A_ineq[[True if i in knockable_constr_ineq else False for i in range(0, self.A_ineq.shape[0])], :][:,
              cont_vars]
        M_A = [(-M_A[i, :])[0].toarray()[0] for i in range(len(knockable_constr_ineq))]
        M_b = [self.b_ineq[i] for i in range(0, self.A_ineq.shape[0]) if i in knockable_constr_ineq]

        # Run LPs to determine maximal values knockable constraints can take.
        # This task is supported by the parallelization module 'Ray', if Ray is initialized
        if ray.is_initialized():
            # a) Build an Actor - a stateful worker based on a class
            @ray.remote  # class is decorated with ray.remote to for parallel use
            class M_optimizer(object):
                def __init__(self):  # The LP object is only constructed once upon Actor creation.
                    self.lp = solver_interface.MILP_LP(A_ineq=M_A_ineq, b_ineq=M_b_ineq, A_eq=M_A_eq, b_eq=M_b_eq,
                                                       lb=M_lb, ub=M_ub)

                def compute(self, c):  # With each function call only the objective function is changed
                    self.lp.set_objective(c)
                    x, min, status = self.lp.solve()
                    return -min

            # b) Create pool of Actors on which the computations should be executed. Number of Actors = number of CPUs
            parpool = ray.util.ActorPool([M_optimizer.remote() for _ in range(int(ray.available_resources()['CPU']))])
            # c) Run M computations on actor pool. lambda is an inline function 
            max_Ax = list(parpool.map(lambda a, x: a.compute.remote(x), M_A))
        # If Ray is not available, use regular loop
        else:
            max_Ax = [np.nan] * len(M_A)
            lp = solver_interface.MILP_LP(A_ineq=M_A_ineq, b_ineq=M_b_ineq, A_eq=M_A_eq, b_eq=M_b_eq, lb=M_lb, ub=M_ub)
            for i in range(len(M_A)):
                lp.set_objective(M_A[i])
                x, min, status = lp.solve()
                max_Ax[i] = -min

        Ms = [M - b if not np.isnan(M) else self.M for M, b in zip(max_Ax, M_b)]
        # fill up M-vector also for notknockable reactions
        Ms = [Ms[np.array([i == j for j in knockable_constr_ineq]).nonzero()[0][
            0]] if i in knockable_constr_ineq else np.nan for i in range(self.A_ineq.shape[0])]

        # 4. Link constraints to z-variables for available (and also arbitrary high) Ms
        self.z_map_constr_ineq = self.z_map_constr_ineq.tocsc()
        self.A_ineq = self.A_ineq.todok()
        # iterate through knockable constraints
        for row in range(self.A_ineq.shape[0]):
            if not np.isinf(Ms[row]) and not np.isnan(Ms[row]):  # if there is a real number for M, use this for KO
                z_i = self.z_map_constr_ineq[:, row].nonzero()[0][0]
                sense = self.z_map_constr_ineq[z_i, row]
                if sense > 0:  # This means z_i = 1 knocks out ineq:
                    #     a_ineq*x - M*z <= b
                    self.A_ineq[row, z_i] = -Ms[row]
                else:  # This means z_i = 0 knocks out ineq:
                    #     a_ineq*x + M*z <= b + M
                    self.A_ineq[row, z_i] = Ms[row]
                    self.b_ineq[row] = self.b_ineq[row] + Ms[row]
        self.z_map_constr_ineq = self.z_map_constr_ineq.tocsc()

        # 5. Translate back remaining inequalities to equations if applicable
        knockable_constr_ineq = tuple(knockable_constr_ineq)
        knockable_constr_ineq_ic = [i for i in range(self.A_ineq.shape[0]) if np.isinf(Ms[i])]
        self.A_ineq = self.A_ineq.tocsr()

        # approach to find inequalities that can be lumped:
        # - construct a matrix from A_ineq, b_ineq, z_ineq for knockable constraints 
        #   where every first entry of a row is positive
        # - search for row duplicates
        # - delete one if their first row entry had the same sign, lump to equality if they had opposite signs 
        first_entry_A_ineq_sign = [np.sign(a.data[0]) for a in self.A_ineq]
        Ab_find_dupl = sparse.hstack((sparse.diags(first_entry_A_ineq_sign) * self.A_ineq, \
                                      sparse.coo_matrix(
                                          sparse.diags(first_entry_A_ineq_sign) * self.b_ineq).transpose(), \
                                      self.z_map_constr_ineq.transpose())) \
            .tocsr()
        # find rows that are identical
        ident_rows = []  # stores duplicate rows as Tuple (i,j,k): first row, second row, positive or negative duplicate
        for i, a1 in enumerate(Ab_find_dupl[range(Ab_find_dupl.shape[0] - 1)]):
            if i in knockable_constr_ineq_ic:  # only compare knockable ineqs
                for j, a2 in enumerate(Ab_find_dupl):
                    if j in knockable_constr_ineq_ic and j > i:  # only compare knockable ineqs
                        if a1.nnz == a2.nnz:
                            if all(a1.indices == a2.indices) & all(a1.data == a2.data):
                                ident_rows += [(i, j, first_entry_A_ineq_sign[i] * first_entry_A_ineq_sign[j])]
        # replace two ineqs by one eq
        self.z_map_constr_ineq = self.z_map_constr_ineq.tocsc()
        A_eq = sparse.csr_matrix((0, self.A_ineq.shape[1]))
        z_eq = sparse.csc_matrix((self.num_z, 0))
        b_eq = []
        for j in [i for i in range(len(ident_rows)) if ident_rows[i][2] == -1]:
            A_eq = sparse.vstack((A_eq, self.A_ineq[ident_rows[j][0]]))
            b_eq += [self.b_ineq[ident_rows[j][0]]]
            z_eq = sparse.hstack((z_eq, self.z_map_constr_ineq[:, ident_rows[j][0]]))
        # Add to global equality problem part
        knockable_constr_eq_ic = [i + len(self.b_eq) for i in range(len(b_eq))]
        self.A_eq = sparse.vstack((self.A_eq, A_eq))
        self.b_eq += b_eq
        self.z_map_constr_eq = sparse.hstack((self.z_map_constr_eq, z_eq)).tocsc()

        # Remove all duplicates from ineq
        if ident_rows == []:
            remove_ineq = np.array([], 'int')
        else:
            remove_ineq = np.unique(np.hstack([[ir[0], ir[1]] if ir[2] == -1 else [ir[1]] for ir in ident_rows]))
        keep_ineq = [i for i in range(self.A_ineq.shape[0]) if i not in remove_ineq]
        knockable_constr_ineq_ic = [True if i in knockable_constr_ineq_ic else False for i in
                                    range(self.A_ineq.shape[0])]
        knockable_constr_ineq_ic = np.nonzero([knockable_constr_ineq_ic[i] for i in keep_ineq])[0]
        self.A_ineq = self.A_ineq[keep_ineq, :]
        self.b_ineq = [self.b_ineq[i] for i in keep_ineq]
        self.z_map_constr_ineq = self.z_map_constr_ineq[:, keep_ineq]

        # 6. Link remaining (in)equalities to z via indicator constraints
        #    and remove them from the static problem
        #    - first ineqs then eqs
        ic_binv = np.append(self.z_map_constr_ineq[:, knockable_constr_ineq_ic].indices,
                            self.z_map_constr_eq[:, knockable_constr_eq_ic].indices)
        ic_A = sparse.vstack((self.A_ineq[knockable_constr_ineq_ic, :], self.A_eq[knockable_constr_eq_ic, :]))
        ic_b = [self.b_ineq[i] for i in knockable_constr_ineq_ic] + [self.b_eq[i] for i in knockable_constr_eq_ic]
        ic_sense = 'L' * len(knockable_constr_ineq_ic) + 'E' * len(knockable_constr_eq_ic)
        ic_indicval = np.append(self.z_map_constr_ineq[:, knockable_constr_ineq_ic].data,
                                self.z_map_constr_eq[:, knockable_constr_eq_ic].data)
        ic_indicval = [0 if i == 1 else 1 for i in ic_indicval]
        # in z-maps: -1 => z=1 -> A_ineq*x <= b_ineq
        #             1 => z=0 -> A_ineq*x <= b_ineq
        self.indic_constr = indicator_constraints.Indicator_constraints(ic_binv, ic_A, ic_b, ic_sense, ic_indicval)

        # 7. Remove knockable (in)equalities from static problem, as they are now indicator constraints
        keep_ineq = [False if i in knockable_constr_ineq_ic else True for i in range(self.A_ineq.shape[0])]
        self.A_ineq = self.A_ineq[keep_ineq, :]
        self.b_ineq = [self.b_ineq[i] for i in range(len(keep_ineq)) if keep_ineq[i]]
        keep_eq = [False if i in knockable_constr_eq_ic else True for i in range(self.A_eq.shape[0])]
        self.A_eq = self.A_eq[keep_eq, :]
        self.b_eq = [self.b_eq[i] for i in range(len(keep_eq)) if keep_eq[i]]

    def reassign_lb_ub_from_ineq(self, A_ineq, b_ineq, A_eq, b_eq, lb, ub, z_map_constr_ineq, z_map_constr_eq,
                                 z_map_vars) -> \
            Tuple[
                sparse.csr_matrix, Tuple, sparse.csr_matrix, Tuple, Tuple, Tuple, sparse.csr_matrix, sparse.csr_matrix]:
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
        var_bound_constraint_ineq = [i for i in row_ineq if list(row_ineq).count(i) == 1]
        # exclude knockable constraints
        var_bound_constraint_ineq = [i for i in var_bound_constraint_ineq if i not in z_map_constr_ineq.nonzero()[1]]
        # retrieve all bounds from inequality constraints
        for i in var_bound_constraint_ineq:
            idx_r = A_ineq[i, :].nonzero()[1]  # get reaction from constraint (column of entry)
            if A_ineq[i, idx_r] > 0:  # upper bound constraint
                ub[i] += [b_ineq[i] / A_ineq[i, idx_r].toarray()[0][0]]
            else:  # lower bound constraint
                lb[i] += [b_ineq[i] / A_ineq[i, idx_r].toarray()[0][0]]

        # find all entries in A_eq
        row_eq = A_eq.nonzero()[0]
        # filter for rows with only one entry
        var_bound_constraint_eq = [i for i in row_eq if list(row_eq).count(i) == 1]
        # exclude knockable constraints
        var_bound_constraint_eq = [i for i in var_bound_constraint_eq if i not in z_map_constr_eq.nonzero()[1]]
        # retrieve all bounds from equality constraints
        # and partly set lb or ub derived from equality constraints, for instance:
        # If x =  5, set ub = 5 and keep the inequality constraint -x <= -5.
        # If x = -5, set lb =-5 and keep the inequality constraint  x <=  5.
        A_ineq_new = sparse.csr_matrix((0, numr))
        b_ineq_new = []
        for i in var_bound_constraint_eq:
            idx_r = A_eq[i, :].nonzero()[1]  # get reaction from constraint (column of entry)
            if any(z_map_vars[:, idx_r]):  # if reaction is knockable
                if A_eq[i, idx_r] * b_eq[i] > 0:  # upper bound constraint
                    ub[i] += [b_eq[i] / A_eq[i, idx_r].toarray()[0][0]]
                    A_ineq_new = sparse.vstack((A_ineq_new, -A_eq[i, :]))
                    b_ineq_new += [-b_eq[i]]
                elif A_eq[i, idx_r] * b_eq[i] < 0:  # lower bound constraint
                    lb[i] += [b_eq[i] / A_eq[i, idx_r].toarray()[0][0]]
                    A_ineq_new = sparse.vstack((A_ineq_new, A_eq[i, :]))
                    b_ineq_new += [b_eq[i]]
                else:
                    ub[i] += [0]
                    lb[i] += [0]
            else:
                lb[i] += [b_eq[i] / A_eq[i, idx_r].toarray()[0][0]]
                ub[i] += [b_eq[i] / A_eq[i, idx_r].toarray()[0][0]]
        # set tightest bounds (avoid inf)
        lb = [max([i for i in l if not np.isinf(i)] + [np.nan]) for l in lb]
        ub = [min([i for i in u if not np.isinf(i)] + [np.nan]) for u in ub]
        # set if only if no other bound remains
        lb = [-np.inf if np.isnan(l) else l for l in lb]
        ub = [np.inf if np.isnan(u) else u for u in ub]

        # check if bounds are consistent
        if any(np.greater(lb, ub)):
            raise Exception("There is a lower bound that is greater than its upper bound counterpart.")

        # remove constraints that became redundant
        A_ineq = A_ineq[[False if i in var_bound_constraint_ineq else True for i in range(0, A_ineq.shape[0])]]
        b_ineq = [b_ineq[i] for i in range(0, len(b_ineq)) if i not in var_bound_constraint_ineq]
        z_map_constr_ineq = z_map_constr_ineq[:,
                            [False if i in var_bound_constraint_ineq else True for i in range(0, A_ineq.shape[0])]]
        A_eq = A_eq[[False if i in var_bound_constraint_eq else True for i in range(0, A_eq.shape[0])]]
        b_eq = [b_eq[i] for i in range(0, len(b_eq)) if i not in var_bound_constraint_eq]
        z_map_constr_eq = z_map_constr_eq[:,
                          [False if i in var_bound_constraint_eq else True for i in range(0, A_eq.shape[0])]]
        # add equality constraints that transformed to inequality constraints
        A_ineq = sparse.vstack((A_ineq, A_ineq_new))
        b_ineq += b_ineq_new
        z_map_constr_ineq = sparse.hstack((z_map_constr_ineq, sparse.csc_matrix((self.num_z, A_ineq_new.shape[0]))))
        return A_ineq, b_ineq, A_eq, b_eq, lb, ub, z_map_constr_ineq, z_map_constr_eq

    def lineq2mat(self, equations) -> Tuple[sparse.csr_matrix, Tuple, sparse.csr_matrix, Tuple]:
        numr = len(self.model.reactions)
        A_ineq = sparse.csr_matrix((0, numr))
        b_ineq = []
        A_eq = sparse.csr_matrix((0, numr))
        b_eq = []
        for equation in equations:
            try:
                lhs, rhs = re.split('<=|=|>=', equation)
                eq_sign = re.search('<=|>=|=', equation)[0]
                rhs = float(rhs)
            except:
                raise Exception(
                    "Equations must contain exactly one (in)equality sign: <=,=,>=. Right hand side must be a float number.")
            A = self.linexpr2mat(lhs)
            if eq_sign == '=':
                A_eq = sparse.vstack((A_eq, A))
                b_eq += [rhs]
            elif eq_sign == '<=':
                A_ineq = sparse.vstack((A_ineq, A))
                b_ineq += [rhs]
            elif eq_sign == '>=':
                A_ineq = sparse.vstack((A_ineq, -A))
                b_ineq += [-rhs]
        return A_ineq, b_ineq, A_eq, b_eq

    def linexpr2mat(self, lhs) -> sparse.csr_matrix:
        # linexpr2mat translates the left hand side of a linear expression into a matrix
        #
        # e.g.: Model with reactions R1, R2, R3, R4
        #       Expression: '2 R3 - R1'
        #     translates into list:
        #       A = [-1 0 2 0]
        # 
        A = sparse.lil_matrix((1, len(self.model.reactions)))
        # split expression into parts and strip away special characters
        ridx = [re.sub(r'^(\s|-|\+|\()*|(\s|-|\+|\))*$', '', part) for part in lhs.split()]
        # identify reaction identifiers by comparing with models reaction list
        ridx = [r for r in ridx if r in self.model.reactions.list_attr('id')]
        if not len(ridx) == len(set(ridx)):  # check for duplicates
            raise Exception("Reaction identifiers may only occur once in each linear expression.")
        # iterate through reaction identifiers and retrieve coefficients from linear expression
        for rid in ridx:
            coeff = re.search('(\s|^)(\s|\d|-|\+|\.)*?(?=' + rid + '(\s|$))', lhs)[0]
            coeff = re.sub('\s', '', coeff)
            if coeff in ['', '+']:
                coeff = 1
            if coeff == '-':
                coeff = -1
            else:
                coeff = float(coeff)
            A[0, self.model.reactions.list_attr('id').index(rid)] = coeff
        return A.tocsr()


class ContMILP:
    def __init__(self, A_ineq, b_ineq, A_eq, b_eq, lb, ub, c, z_map_constr_ineq, z_map_constr_eq, z_map_vars):
        self.A_ineq = A_ineq
        self.b_ineq = b_ineq
        self.A_eq = A_eq
        self.b_eq = b_eq
        self.lb = lb
        self.ub = ub
        self.c = c
        self.z_map_constr_ineq = z_map_constr_ineq
        self.z_map_constr_eq = z_map_constr_eq
        self.z_map_vars = z_map_vars
