#!/usr/bin/env python3
#
# Copyright 2022 Max Planck Insitute Magdeburg
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
#
#
"""Classes and functions for the construction of strain design MILPs

This module contains functions that help construct mixed-integer linear problems,
mainly functions that facilitate the construction of LP and Farkas dual
problems from linear problems of the type A_ineq*x<=b_ineq, A_eq*x<=b_eq, lb<=x<=ub.
The functions also help keeping track of the relationship of constraints and variables
and their individual counterparts in dual problems, which is essential when simulating 
knockouts in dual problems. Most of the time, the sparse datatype is used to store and
edit matrices for improved speed and memory."""

import numpy as np
from scipy import sparse
from cobra.util import create_stoichiometric_matrix
from cobra import Model, Configuration
from typing import List, Tuple
from straindesign import SDModule, IndicatorConstraints, lineqlist2mat, linexprdict2mat, MILP_LP, SDPool, \
                         avail_solvers, select_solver, remove_dummy_bounds, SDModule
from straindesign.names import *
import logging


class SDProblem:
    """Strain design MILP
    
    The construcor of this class translates a model and strain design modules into a mixed integer linear
    problem. This class, however, is the backbone of the strain design computation. Preprocessing steps 
    that enable gene, reaction and regulatory interventions, or network compression usually preceed the 
    construction of an SDProblem-object and are integrated in the function *compute_strain_designs*. 

    Args:
        model (cobra.Model):
            A metabolic model that is an instance of the cobra.Model class.
        
        sd_modules ((list of) straindesign.SDModule):
            Modules that specify the strain design problem, e.g., protected or suppressed flux states for 
            MCS strain design or inner and outer objective functions for OptKnock. See description
            of SDModule for more information on how to set up modules.
            
        ko_cost (optional (dict)): (Default: None)
            A dictionary of reaction identifiers and their associated knockout costs. If not specified, all reactions
            are treated as knockout candidates, equivalent to ko_cost = {'r1':1, 'r2':1, ...}. If a subset of reactions
            is listed in the dict, all other are not considered as knockout candidates.
            
        ki_cost (optional (dict)): (Default: None)
            A dictionary of reaction identifiers and their associated costs for addition. If not specified, all reactions
            are treated as knockout candidates. Reaction addition candidates must be present in the original model with
            the intended flux boundaries **after** insertion. Additions are treated adversely to knockouts, meaning that
            their exclusion from the network is not associated with any cost while their presence entails intervention costs.
            
        max_cost (optional (int)): (Default: inf): 
            The maximum cost threshold for interventions. Every possible intervention is associated with a
            cost value (1, by default). Strain designs cannot exceed the max_cost threshold. Individual
            intervention cost factors may be defined through ki_cost and ko_cost.
            
        solver (optional (str)): (Default: same as defined in model / COBRApy)
            The solver that should be used for preparing and carrying out the strain design computation.
            Allowed values are 'cplex', 'gurobi', 'scip' and 'glpk'.
            
        M (optional (int)): (Default: None)
            If this value is specified (and non-zero, not None), the computation uses the big-M 
            method instead of indicator constraints. Since GLPK does not support indicator constraints it uses
            the big-M method by default (with COBRA standard M=1000). M should be chosen 'sufficiently large' 
            to avoid computational artifacts and 'sufficiently small' to avoid numerical issues.
            
        essential_kis (optional (set)):
            A set of reactions that are marked as addable and that are essential for at least one of the
            strain design modules. Providing such "essential knock-ins" may speed up the strain design computation.
            
    Returns:
        (SDProblem):
            An instance of SDProblem containing the strain design MILP
    """

    def __init__(self, model: Model, sd_modules: List[SDModule], *args, **kwargs):
        allowed_keys = {KOCOST, KICOST, SOLVER, MAX_COST, 'M', 'essential_kis', SEED}
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

        if "SD_Module" in str(type(sd_modules)):
            sd_modules = [sd_modules]

        if self.solver is None:
            if len(avail_solvers) > 0:
                self.solver = list(avail_solvers)[0]
            else:
                raise Exception('No solver available. Please ensure that one of the following '\
                    'solvers is avaialable in your Python environment: CPLEX, Gurobi, SCIP, GLPK')
        self.solver = select_solver(self.solver, model)
        cobra_conf = Configuration()
        bound_thres = max((abs(cobra_conf.lower_bound), abs(cobra_conf.upper_bound)))
        if self.M is None and self.solver == 'glpk':
            logging.warning(
                'GLPK only supports strain design computation with the bigM method. Using cobra bound: '+str(bound_thres)+\
                ' as M.')
            self.M = bound_thres
        elif self.M is None:
            self.M = np.inf
        # the matrices in sd_modules, ko_cost and ki_cost should be numpy.array or scipy.sparse (csr, csc, lil) format
        self.model = model
        self.sd_modules = sd_modules
        reac_ids = model.reactions.list_attr("id")
        numr = len(model.reactions)
        # Create vectors for ko_cost, ki_cost, inverted bool-vars and non-targetable bools
        # If no knockable reactions are assigned, assume all are KO-able.
        # Generally, KIs overwrite KOs
        if self.ko_cost is None:
            self.ko_cost = {rid: 1.0 for rid in reac_ids}
        if self.ki_cost is None:
            self.ki_cost = {}
        if self.essential_kis is None:
            self.essential_kis = set()
        self.ko_cost = [float(self.ko_cost.get(key)) if (key in self.ko_cost.keys()) else np.nan for key in reac_ids]
        self.ki_cost = [float(self.ki_cost.get(key)) if (key in self.ki_cost.keys()) else np.nan for key in reac_ids]
        self.ko_cost = [self.ko_cost[i] if np.isnan(self.ki_cost[i]) else np.nan for i in range(numr)]
        self.num_z = numr
        self.cost = [i for i in self.ko_cost]
        for i in [i for i, x in enumerate(self.ki_cost) if not np.isnan(x)]:
            self.cost[i] = self.ki_cost[i]
        self.z_inverted = [not np.isnan(x) for x in self.ki_cost]
        self.z_non_targetable = [np.isnan(x) for x in self.cost]
        for i in [i for i, x in enumerate(self.cost) if np.isnan(x)]:
            self.cost[i] = 0.0
        # Prepare top 3 lines of MILP (sum of weighted interventions below (0) and above (1) threshold) and objective function (2)
        self.idx_row_maxcost = 0
        self.idx_row_mincost = 1
        self.idx_row_obj = 2
        self.A_ineq = sparse.csr_matrix([[-i for i in self.cost], self.cost, [0 for _ in range(self.num_z)]])
        if self.max_cost is None:
            self.b_ineq = [0.0, float(np.sum(np.abs(self.cost))), np.inf]
        else:
            self.b_ineq = [0.0, float(self.max_cost), np.inf]
        self.z_map_constr_ineq = sparse.csc_matrix((numr, 3))
        self.lb = [1.0 if r in self.essential_kis else 0.0 for r in model.reactions]
        self.ub = [1.0 - float(i) for i in self.z_non_targetable]
        self.idx_z = [i for i in range(0, numr)]
        self.c = [0.0] * numr
        # Initialize also empty equality matrix
        self.A_eq = sparse.csc_matrix((0, numr))
        self.b_eq = []
        self.z_map_constr_eq = sparse.csc_matrix((numr, 0))
        self.num_modules = 0
        self.indic_constr = []  # Add instances of the class 'Indicator_constraint' later
        # Initialize association between z and variables and variables
        self.z_map_vars = sparse.csc_matrix((numr, numr))
        # replace bounds with inf if above a cobra bound threshold
        remove_dummy_bounds(self.model)
        logging.info('Constructing strain design MILP for solver: ' + self.solver + '.')
        for i in range(len(sd_modules)):
            self.addModule(sd_modules[i])

        # Assign knock-ins/outs correctly by taking into account z_inverted
        # invert *(-1) rows in z_map_constr_eq, z_map_constr_ineq, z_map_vars
        # where there are "knock-ins" / additions
        # make knock-in/out matrix
        z_kos_kis = [
            1 if (not self.z_non_targetable[i]) and (not self.z_inverted[i]) else -1 if self.z_inverted[i] else 0
            for i in range(0, self.num_z)
        ]
        z_kos_kis = sparse.diags(z_kos_kis)
        self.z_map_constr_ineq = z_kos_kis * self.z_map_constr_ineq
        self.z_map_constr_eq = z_kos_kis * self.z_map_constr_eq
        self.z_map_vars = z_kos_kis * self.z_map_vars

        # Save continous part of MILP for easy strain design validation
        cont_vars = [i for i in range(0, self.A_ineq.shape[1]) if not i in self.idx_z]
        self.cont_MILP = ContMILP(self.A_ineq[:, cont_vars], self.b_ineq.copy(), self.A_eq[:, cont_vars], self.b_eq.copy(),
                                  [self.lb[i] for i in cont_vars], [self.ub[i] for i in cont_vars], [self.c[i] for i in cont_vars],
                                  self.z_map_constr_ineq.tocoo(), self.z_map_constr_eq.tocoo(), self.z_map_vars[:, cont_vars].tocoo())

        # 4. Link LP module to z-variables
        self.link_z()

        # if there are only mcs modules, minimize the knockout costs,
        # otherwise use objective function(s) from modules
        if all([mod[MODULE_TYPE] in [PROTECT, SUPPRESS] for mod in sd_modules]):
            for i in self.idx_z:
                self.c[i] = self.cost[i]
            self.is_mcs_computation = True
        else:
            self.is_mcs_computation = False
            for i in self.idx_z:
                self.c[i] = 0.0
            self.A_ineq = self.A_ineq.tolil()
            self.A_ineq[2] = sparse.lil_matrix(self.c)  # set objective
            self.A_ineq = self.A_ineq.tocsr()

        # backup objective function
        self.c_bu = [float(i) for i in self.c.copy()]

        # # for debugging
        # A = sparse.vstack(( self.A_ineq,sparse.csr_matrix([np.nan]*len(self.c)),\
        #                     self.A_eq,sparse.csr_matrix([np.nan]*len(self.c)),\
        #                     self.indic_constr.A,sparse.csr_matrix([np.nan]*len(self.c)),\
        #                     sparse.csr_matrix(self.ub),sparse.csr_matrix(self.lb)))
        # b = self.b_ineq + [np.nan] + self.b_eq + [np.nan] + self.indic_constr.b + [np.nan, np.nan, np.nan]
        # Ab = sparse.hstack((A,sparse.csr_matrix([np.nan]*len(b)).transpose(),sparse.csr_matrix(b).transpose()))
        # np.savetxt("Ab_py.tsv", Ab.todense(), delimiter='\t')
        self.vtype = 'B' * self.num_z + 'C' * (self.z_map_vars.shape[1] - self.num_z)

    def addModule(self, sd_module):
        """Generate module LP and z-linking-matrix for each module and add them to the strain design MILP
        
        Args:
            sd_module (straindesign.SDModule):
                Modules to describe strain design problems like protected or suppressed flux states for 
                MCS strain design or inner and outer objective functions for OptKnock. See description
                of SDModule for more information on how to set up modules.
        """
        self.num_modules += 1
        z_map_constr_ineq_i = []
        z_map_constr_eq_i = []
        z_map_vars_i = []
        # 1. Translate (in)equalities into matrix form
        V_ineq, v_ineq, V_eq, v_eq = lineqlist2mat(sd_module[CONSTRAINTS], self.model.reactions.list_attr('id'))
        # 2. Construct LP for module
        if sd_module[MODULE_TYPE] in [PROTECT, SUPPRESS] and sd_module[INNER_OBJECTIVE] is None:
            # Classical MCS
            A_ineq_p, b_ineq_p, A_eq_p, b_eq_p, lb_p, ub_p, c_p, z_map_constr_ineq_p, z_map_constr_eq_p, z_map_vars_p \
                = build_primal_from_cbm(self.model, V_ineq, v_ineq, V_eq, v_eq)
        elif sd_module[MODULE_TYPE] in [PROTECT, SUPPRESS, OPTKNOCK, OPTCOUPLE]:
            c_in = linexprdict2mat(sd_module[INNER_OBJECTIVE], self.model.reactions.list_attr('id'))
            # by default, assume maximization of the inner objective
            if not hasattr(sd_module,INNER_OPT_SENSE) or sd_module[INNER_OPT_SENSE] is None or \
                sd_module[INNER_OPT_SENSE] not in [MINIMIZE, MAXIMIZE] or sd_module[INNER_OPT_SENSE] == MAXIMIZE:
                c_in = -c_in
            c_in = c_in.toarray()[0].tolist()
            # 1. build primal w/ desired constraint (build_primal_from_cbm) - also store variable c
            A_ineq_v, b_ineq_v, A_eq_v, b_eq_v, lb_v, ub_v, c_v, z_map_constr_ineq_v, z_map_constr_eq_v, z_map_vars_v \
                = build_primal_from_cbm(self.model, V_ineq, v_ineq, V_eq, v_eq, c_in)
            # 2. build primal w/o desired constraint (build_primal_from_cbm) - store c_inner
            A_ineq_inner, b_ineq_inner, A_eq_inner, b_eq_inner, lb_inner, ub_inner, c_inner, z_map_constr_ineq_inner, z_map_constr_eq_inner, z_map_vars_inner \
                = build_primal_from_cbm(self.model, V_ineq=None, v_ineq=None, V_eq=None, v_eq=None, c=c_in)
            # 3. build dual from primal w/o desired constraint (build_dual w/o the farkas-option) - store c_inner_dual
            A_ineq_dual, b_ineq_dual, A_eq_dual, b_eq_dual, lb_dual, ub_dual, c_inner_dual, z_map_constr_ineq_dual, z_map_constr_eq_dual, z_map_vars_dual \
                = LP_dualize(A_ineq_inner, b_ineq_inner, A_eq_inner, b_eq_inner, lb_inner, ub_inner, c_inner, z_map_constr_ineq_inner, z_map_constr_eq_inner, z_map_vars_inner)
            # 4. connect primal w/ undesired region and dual w/o undesired region (i.e. biomass) via c = c_inner.
            A_ineq_p = sparse.block_diag((A_ineq_v, A_ineq_dual)).tocsr()
            b_ineq_p = b_ineq_v + b_ineq_dual
            A_eq_p = sparse.vstack((sparse.block_diag(
                (A_eq_v, A_eq_dual)), sparse.hstack((sparse.csr_matrix(c_v), sparse.csr_matrix(c_inner_dual))))).tocsr()
            b_eq_p = b_eq_v + b_eq_dual + [0.0]
            lb_p = lb_v + lb_dual
            ub_p = ub_v + ub_dual
            # 5. Update z-associations
            z_map_vars_p = sparse.hstack((z_map_vars_v, z_map_vars_dual))
            z_map_constr_ineq_p = sparse.hstack((z_map_constr_ineq_v, z_map_constr_ineq_dual))
            z_map_constr_eq_p = sparse.hstack((z_map_constr_eq_v, z_map_constr_eq_dual, sparse.csc_matrix((self.num_z, 1))))
        elif sd_module[MODULE_TYPE] == ROBUSTKNOCK:
            # RobustKnock has three layers, inner maximization and an outer min-max problem
            c_in = linexprdict2mat(sd_module[INNER_OBJECTIVE], self.model.reactions.list_attr('id'))
            # by default, assume maximization of the inner objective
            if not hasattr(sd_module,INNER_OPT_SENSE) or sd_module[INNER_OPT_SENSE] is None or \
                sd_module[INNER_OPT_SENSE] not in [MINIMIZE, MAXIMIZE] or sd_module[INNER_OPT_SENSE] == MAXIMIZE:
                c_in = -c_in
            c_in = c_in.toarray()[0].tolist()
            # 1. build primal of inner problem (build_primal_from_cbm) - also store variable c
            A_ineq_v, b_ineq_v, A_eq_v, b_eq_v, lb_v, ub_v, c_v, z_map_constr_ineq_v, z_map_constr_eq_v, z_map_vars_v \
                = build_primal_from_cbm(self.model, V_ineq=None, v_ineq=None, V_eq=None, v_eq=None, c=c_in)
            # 2. build primal of outer problem (base of inner outer problem) (build_primal_from_cbm) - with inner objective
            A_ineq_inner, b_ineq_inner, A_eq_inner, b_eq_inner, lb_inner, ub_inner, c_inner, z_map_constr_ineq_inner, z_map_constr_eq_inner, z_map_vars_inner \
                = build_primal_from_cbm(self.model,V_ineq=None, v_ineq=None, V_eq=None, v_eq=None, c=c_in)
            # 3. build primal of outer problem (outer outer problem) (build_primal_from_cbm) - store c_inner
            c_out = linexprdict2mat(sd_module[OUTER_OBJECTIVE], self.model.reactions.list_attr('id'))  # get outer objective
            if not hasattr(sd_module,OUTER_OPT_SENSE) or sd_module[OUTER_OPT_SENSE] is None or \
                sd_module[OUTER_OPT_SENSE] not in [MINIMIZE, MAXIMIZE] or sd_module[OUTER_OPT_SENSE] == MAXIMIZE:
                c_out = -c_out
            c_out = c_out.toarray()[0].tolist()
            A_ineq_r, b_ineq_r, A_eq_r, b_eq_r, lb_r, ub_r, c_r, z_map_constr_ineq_r, z_map_constr_eq_r, z_map_vars_r \
                = build_primal_from_cbm(self.model,V_ineq, v_ineq, V_eq, v_eq, [-c for c in c_out])
            # 4. build dual of innerst problem - store c_inner_dual
            A_ineq_dual, b_ineq_dual, A_eq_dual, b_eq_dual, lb_dual, ub_dual, c_inner_dual, z_map_constr_ineq_dual, z_map_constr_eq_dual, z_map_vars_dual \
                = LP_dualize(A_ineq_inner, b_ineq_inner, A_eq_inner, b_eq_inner, lb_inner, ub_inner, c_inner, z_map_constr_ineq_inner, z_map_constr_eq_inner, z_map_vars_inner)
            # 5. connect primal w/ undesired region and dual w/o undesired region (i.e. biomass) via c = c_inner.
            A_ineq_p = sparse.block_diag((A_ineq_v, A_ineq_dual)).tocsr()
            b_ineq_p = b_ineq_v + b_ineq_dual
            A_eq_p = sparse.vstack((sparse.block_diag(
                (A_eq_v, A_eq_dual)), sparse.hstack((sparse.csr_matrix(c_v), sparse.csr_matrix(c_inner_dual))))).tocsr()
            b_eq_p = b_eq_v + b_eq_dual + [0.0]
            lb_p = lb_v + lb_dual
            ub_p = ub_v + ub_dual
            c_out_in_p = -sparse.csr_matrix(c_out)
            c_out_in_p.resize((1, A_ineq_p.shape[1]))
            c_out_in_p = c_out_in_p.toarray()[0].tolist()
            # 6. Update z-associations
            z_map_vars_p = sparse.hstack((z_map_vars_v, z_map_vars_dual))
            z_map_constr_ineq_p = sparse.hstack((z_map_constr_ineq_v, z_map_constr_ineq_dual))
            z_map_constr_eq_p = sparse.hstack((z_map_constr_eq_v, z_map_constr_eq_dual, sparse.csc_matrix((self.num_z, 1))))
            # 7. Dualize the joint inner problem
            A_ineq_dl_mmx, b_ineq_dl_mmx, A_eq_dl_mmx, b_eq_dl_mmx, lb_dl_mmx, ub_dl_mmx, c_dl_mmx, z_map_constr_ineq_dl_mmx, z_map_constr_eq_dl_mmx, z_map_vars_dl_mmx \
                = LP_dualize(A_ineq_p, b_ineq_p, A_eq_p, b_eq_p, lb_p, ub_p, c_out_in_p, z_map_constr_ineq_p, z_map_constr_eq_p, z_map_vars_p)
            # 8. Connect outer problem to the dualized combined inner problem to construct min-max problem.
            A_ineq_q = sparse.block_diag((A_ineq_r, A_ineq_dl_mmx)).tocsr()
            b_ineq_q = b_ineq_r + b_ineq_dl_mmx
            A_eq_q = sparse.vstack((sparse.block_diag(
                (A_eq_r, A_eq_dl_mmx)), sparse.hstack((sparse.csr_matrix(c_r), sparse.csr_matrix(c_dl_mmx))))).tocsr()
            b_eq_q = b_eq_r + b_eq_dl_mmx + [0.0]
            lb_q = lb_r + lb_dl_mmx
            ub_q = ub_r + ub_dl_mmx
            # 9. Update z-associations
            z_map_vars_q = sparse.hstack((z_map_vars_r, z_map_vars_dl_mmx))
            z_map_constr_ineq_q = sparse.hstack((z_map_constr_ineq_r, z_map_constr_ineq_dl_mmx))
            z_map_constr_eq_q = sparse.hstack((z_map_constr_eq_r, z_map_constr_eq_dl_mmx, sparse.csc_matrix((self.num_z, 1))))
            # 10. reassign lb, ub where possible
            A_ineq_i, b_ineq_i, A_eq_i, b_eq_i, lb_i, ub_i, z_map_constr_ineq_i, z_map_constr_eq_i = reassign_lb_ub_from_ineq(
                A_ineq_q, b_ineq_q, A_eq_q, b_eq_q, lb_q, ub_q, z_map_constr_ineq_q, z_map_constr_eq_q, z_map_vars_q)
            z_map_vars_i = z_map_vars_q
            c_i = sparse.csr_matrix(c_out)
            c_i.resize((1, A_ineq_i.shape[1]))
            c_i = c_i.toarray()[0].tolist()
        if sd_module[MODULE_TYPE] == OPTCOUPLE:
            c_p = c_v + [0 for _ in c_inner_dual]
            # (continued from optknock)
            prod_eq = linexprdict2mat(sd_module[PROD_ID], self.model.reactions.list_attr('id'))
            # 6.  build primal system with no production - also store variable c
            A_ineq_r, b_ineq_r, A_eq_r, b_eq_r, lb_r, ub_r, c_r, z_map_constr_ineq_r, z_map_constr_eq_r, z_map_vars_r \
                = build_primal_from_cbm(self.model,V_ineq=None, v_ineq=None, V_eq=prod_eq, v_eq=[0], c=c_in)
            # 7. Dualize no-production system.
            A_ineq_r_dl, b_ineq_dl_r_dl, A_eq_dl_r_dl, b_eq_r_dl, lb_r_dl, ub_r_dl, c_r_dl, z_map_constr_ineq_r_dl, z_map_constr_eq_r_dl, z_map_vars_r_dl \
                = LP_dualize(A_ineq_r, b_ineq_r, A_eq_r, b_eq_r, lb_r, ub_r, c_r, z_map_constr_ineq_r, z_map_constr_eq_r, z_map_vars_r)
            # 8. Create no-production bi-level system.
            A_ineq_b = sparse.block_diag((A_ineq_r, A_ineq_r_dl), format='csr')
            b_ineq_b = b_ineq_r + b_ineq_dl_r_dl
            A_eq_b = sparse.vstack((sparse.block_diag(
                (A_eq_r, A_eq_dl_r_dl)), sparse.hstack((sparse.csr_matrix(c_r), sparse.csr_matrix(c_r_dl)))),
                                   format='csr')
            b_eq_b = b_eq_r + b_eq_r_dl + [0.0]
            lb_b = lb_r + lb_r_dl
            ub_b = ub_r + ub_r_dl
            c_b = c_r + [0.0] * len(c_r_dl)
            z_map_vars_b = sparse.hstack((z_map_vars_r, z_map_vars_r_dl))
            z_map_constr_ineq_b = sparse.hstack((z_map_constr_ineq_r, z_map_constr_ineq_r_dl))
            z_map_constr_eq_b = sparse.hstack((z_map_constr_eq_r, z_map_constr_eq_r_dl, sparse.csc_matrix((self.num_z, 1))))
            # 9. Connect optknock-problem to the no-production-bilevel system to construct the optcouple problem.
            A_ineq_q = sparse.block_diag((A_ineq_p, A_ineq_b), format='csr')
            b_ineq_q = b_ineq_p + b_ineq_b
            A_eq_q = sparse.block_diag((A_eq_p, A_eq_b), format='csr')
            b_eq_q = b_eq_p + b_eq_b
            lb_q = lb_p + lb_b
            ub_q = ub_p + ub_b
            z_map_vars_q = sparse.hstack((z_map_vars_p, z_map_vars_b))
            z_map_constr_ineq_q = sparse.hstack((z_map_constr_ineq_p, z_map_constr_ineq_b))
            z_map_constr_eq_q = sparse.hstack((z_map_constr_eq_p, z_map_constr_eq_b))
            # if minimum growth-coupling potential is specified, enforce it through inequality
            if MIN_GCP in sd_module:
                A_ineq_q = sparse.vstack((A_ineq_q, sparse.lil_matrix(c_p + [-c for c in c_b])), format='csr')
                b_ineq_q = b_ineq_q + [-sd_module[MIN_GCP]]
                z_map_constr_ineq_q = sparse.hstack((z_map_constr_ineq_q, sparse.csc_matrix((self.num_z, 1))))
            # 10. reassign lb, ub where possible
            A_ineq_i, b_ineq_i, A_eq_i, b_eq_i, lb_i, ub_i, z_map_constr_ineq_i, z_map_constr_eq_i = reassign_lb_ub_from_ineq(
                A_ineq_q, b_ineq_q, A_eq_q, b_eq_q, lb_q, ub_q, z_map_constr_ineq_q, z_map_constr_eq_q, z_map_vars_q)
            z_map_vars_i = z_map_vars_q
            # 11. objective: maximize distance between growth rate at production and no production
            c_i = c_p + [-c for c in c_b]

        # 3. Prepare module as undesired, desired or other
        if sd_module[MODULE_TYPE] == PROTECT:
            A_ineq_i, b_ineq_i, A_eq_i, b_eq_i, lb_i, ub_i, z_map_constr_ineq_i, z_map_constr_eq_i = reassign_lb_ub_from_ineq(
                A_ineq_p, b_ineq_p, A_eq_p, b_eq_p, lb_p, ub_p, z_map_constr_ineq_p, z_map_constr_eq_p, z_map_vars_p)
            z_map_vars_i = z_map_vars_p
            c_i = [0 for _ in range(A_ineq_i.shape[1])]
        elif sd_module[MODULE_TYPE] == SUPPRESS:
            A_ineq_i, b_ineq_i, A_eq_i, b_eq_i, lb_i, ub_i, z_map_constr_ineq_i, z_map_constr_eq_i, z_map_vars_i = farkas_dualize(
                A_ineq_p, b_ineq_p, A_eq_p, b_eq_p, lb_p, ub_p, z_map_constr_ineq_p, z_map_constr_eq_p, z_map_vars_p)
            c_i = [0 for _ in range(A_ineq_i.shape[1])]
        elif sd_module[MODULE_TYPE] == OPTKNOCK:
            A_ineq_i, b_ineq_i, A_eq_i, b_eq_i, lb_i, ub_i, z_map_constr_ineq_i, z_map_constr_eq_i = reassign_lb_ub_from_ineq(
                A_ineq_p, b_ineq_p, A_eq_p, b_eq_p, lb_p, ub_p, z_map_constr_ineq_p, z_map_constr_eq_p, z_map_vars_p)
            z_map_vars_i = z_map_vars_p
            # prepare outer objective
            c_i = linexprdict2mat(sd_module[OUTER_OBJECTIVE], self.model.reactions.list_attr('id'))
            c_i.resize((1, A_ineq_i.shape[1]))
            if OUTER_OPT_SENSE not in sd_module or sd_module[OUTER_OPT_SENSE] is None or \
                sd_module[OUTER_OPT_SENSE] not in [MINIMIZE, MAXIMIZE] or sd_module[OUTER_OPT_SENSE] == MAXIMIZE:
                c_i = -c_i
            c_i = c_i.toarray()[0].tolist()

        # 3. Add module to global MILP
        self.z_map_constr_ineq = sparse.hstack((self.z_map_constr_ineq, z_map_constr_ineq_i)).tocsc()
        self.z_map_constr_eq = sparse.hstack((self.z_map_constr_eq, z_map_constr_eq_i)).tocsc()
        self.z_map_vars = sparse.hstack((self.z_map_vars, z_map_vars_i)).tocsc()
        self.A_ineq = sparse.bmat([[self.A_ineq, None], [None, A_ineq_i]]).tocsr()
        self.b_ineq += b_ineq_i
        self.A_eq = sparse.bmat([[self.A_eq, None], [None, A_eq_i]]).tocsr()
        self.b_eq += b_eq_i
        self.c += c_i
        self.lb += lb_i
        self.ub += ub_i

    def link_z(self):
        """Connect binary intervention variables to variables and constraints of the strain design problem
        
        Function that uses the maps between intervention indicators z and variables and constraints
        of the linear strain design (in)equality system (self.z_map_constr_ineq, self.z_map_constr_eq and 
        self.z_map_vars) to set up the strain design MILP.
        
        MILP construction uses the following steps:
        
        (1) Translate equality-KOs/KIs to two inequality-KOs/KIs
        (2) Translate variable-KOs/KIs to inequality-KIs/KOs
        (3) Try to bound the problem with LPs
        (4) Use LP-determined bounds to link z-variables, where such bounds were found
        (5) Translate remaining inequalities back to equalities when possible and link z via indicator constraints.
        If necessary, the solver interface will translate them to big-M constraints.
        (6) Remove redundant equalities from static problem
        """

        # 1. Split knockable equality constraints into foward and reverse direction
        knockable_constr_eq = self.z_map_constr_eq.nonzero()[1]  # first array: z, second array: eq constr
        eq_constr_A = sparse.vstack((self.A_eq[knockable_constr_eq, :], -self.A_eq[knockable_constr_eq, :]))
        eq_constr_b = [self.b_eq[i] for i in knockable_constr_eq] + [-self.b_eq[i] for i in knockable_constr_eq]
        z_eq = self.z_map_constr_eq[:, tuple(knockable_constr_eq) * 2]
        # Add knockable inequalities to global A_ineq matrix
        self.A_ineq = sparse.vstack((self.A_ineq, eq_constr_A)).tocsr()
        self.b_ineq += eq_constr_b
        self.z_map_constr_ineq = sparse.hstack((self.z_map_constr_ineq, z_eq)).tocsc()
        # Remove knockable equalities from A_eq
        n_rows_eq = self.A_eq.shape[0]
        self.A_eq = self.A_eq[[False if i in knockable_constr_eq else True for i in range(0, n_rows_eq)]]
        self.b_eq = [self.b_eq[i] for i in range(0, len(self.b_eq)) if i not in knockable_constr_eq]
        self.z_map_constr_eq = self.z_map_constr_eq[:, [False if i in knockable_constr_eq else True for i in range(0, n_rows_eq)]]

        # 2. Translate all variable knockouts to inequality knockouts
        numvars = self.A_ineq.shape[1]
        knockable_vars = self.z_map_vars.nonzero()  # first array: z, second array: x
        knockable_vars_geq0 = [i for i in knockable_vars[1] if self.ub[i] > 0]
        knockable_vars_leq0 = [i for i in knockable_vars[1] if self.lb[i] < 0]
        ub_constr_A = sparse.csr_matrix(([1] * len(knockable_vars_geq0), (range(0, len(knockable_vars_geq0)), knockable_vars_geq0)),
                                        [len(knockable_vars_geq0), numvars])
        ub_constr_b = [0 for _ in knockable_vars_geq0]
        lb_constr_A = sparse.csr_matrix(([-1] * len(knockable_vars_leq0), (range(0, len(knockable_vars_leq0)), knockable_vars_leq0)),
                                        [len(knockable_vars_leq0), numvars])
        lb_constr_b = [0 for _ in knockable_vars_leq0]
        bnd_constr_A = sparse.vstack((ub_constr_A, lb_constr_A)).tocsr()
        bnd_constr_b = ub_constr_b + lb_constr_b
        var_kos = [knockable_vars[0][(knockable_vars[1] == i).nonzero()[0][0]] for i in knockable_vars_geq0 + knockable_vars_leq0]
        z_lb_ub = -self.z_map_vars[:, knockable_vars_geq0 + knockable_vars_leq0]
        # add constraints to main problem
        self.A_ineq = sparse.vstack((self.A_ineq, bnd_constr_A)).tocsr()
        self.b_ineq += bnd_constr_b
        self.z_map_constr_ineq = sparse.hstack((self.z_map_constr_ineq, z_lb_ub)).tocsc()

        # 3. Use LP to identify M-values for knockable constraints
        #    For this purpose, first construct a most relaxed LP-model (use all possible constraint-KOs, no possible var-KOs)
        knockable_constr_ineq = np.sort(self.z_map_constr_ineq.nonzero()[1])

        cont_vars = [False if i in self.idx_z else True for i in range(0, numvars)]
        M_A_ineq = self.A_ineq[[False if i in knockable_constr_ineq else True for i in range(0, self.A_ineq.shape[0])], :][:, cont_vars]
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
        M_A = self.A_ineq[[True if i in knockable_constr_ineq else False for i in range(0, self.A_ineq.shape[0])], :][:, cont_vars]
        M_A = [(M_A[i, :])[0].toarray()[0] for i in range(M_A.shape[0])]
        M_b = [self.b_ineq[i] for i in range(0, self.A_ineq.shape[0]) if i in knockable_constr_ineq]

        processes = Configuration().processes
        num_Ms = len(M_A)
        processes = min(processes, num_Ms)

        max_Ax = [np.nan] * num_Ms

        # Dummy to check if optimization runs
        # worker_init(M_A,M_A_ineq,M_b_ineq,M_A_eq,M_b_eq,M_lb,M_ub,list(solvers.keys())[0])
        # worker_compute(1)

        logging.info('  Bounding MILP.')
        if processes > 1 and num_Ms > 1000:
            with SDPool(processes, initializer=worker_init,
                        initargs=(M_A, M_A_ineq, M_b_ineq, M_A_eq, M_b_eq, M_lb, M_ub, self.solver)) as pool:
                chunk_size = num_Ms // processes
                for i, value in pool.imap_unordered(worker_compute, range(num_Ms), chunksize=chunk_size):
                    max_Ax[i] = value
        else:
            worker_init(M_A, M_A_ineq, M_b_ineq, M_A_eq, M_b_eq, M_lb, M_ub, self.solver, self.seed)
            for i in range(num_Ms):
                _, max_Ax[i] = worker_compute(i)

        # round Ms up to 5 digits
        Ms = [np.ceil(M * 1e5) / 1e5 if not np.isinf(M) else self.M for M in max_Ax]
        # fill up M-vector also for notknockable reactions
        Ms = [
            Ms[np.array([i == j
                         for j in knockable_constr_ineq]).nonzero()[0][0]] if i in knockable_constr_ineq else np.nan
            for i in range(self.A_ineq.shape[0])
        ]

        # 4. Link constraints to z-variables for available upper bounds
        self.z_map_constr_ineq = self.z_map_constr_ineq.tocsc()
        self.A_ineq = self.A_ineq.todok()
        # iterate through knockable constraints
        for row in range(self.A_ineq.shape[0]):
            if not np.isinf(Ms[row]) and not np.isnan(Ms[row]):  # if there is a real number for M, use this for KO
                z_i = self.z_map_constr_ineq[:, row].nonzero()[0][0]
                sense = self.z_map_constr_ineq[z_i, row]
                if sense > 0:  # This means z_i = 1 knocks out ineq:
                    #     a_ineq*x - M*z <= b
                    self.A_ineq[row, z_i] = -Ms[row] + self.b_ineq[row]
                else:  # This means z_i = 0 knocks out ineq:
                    #     a_ineq*x + (M-b)*z <= M  (from a_ineq*x + M*z <= M + b)
                    self.A_ineq[row, z_i] = Ms[row] - self.b_ineq[row]
                    self.b_ineq[row] = Ms[row]
        self.z_map_constr_ineq = self.z_map_constr_ineq.tocsc()

        # 5. Translate back remaining inequalities to equations if applicable and link via indicator constraints
        knockable_constr_ineq = tuple(knockable_constr_ineq)
        knockable_constr_ineq_ic = [i for i in range(self.A_ineq.shape[0]) if np.isinf(Ms[i])]
        self.A_ineq = self.A_ineq.tocsr()

        # approach to find inequalities that can be lumped:
        # - construct a matrix from A_ineq, b_ineq, z_ineq for knockable constraints
        #   where every first entry of a row is positive
        # - search for row duplicates
        # - delete one if their first row entry had the same sign, lump to equality if they had opposite signs
        first_entry_A_ineq_sign = [np.sign(a.data[0]) if a.nnz > 0 else 0 for a in self.A_ineq]
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
        self.A_eq = sparse.vstack((self.A_eq, A_eq), 'csr')
        self.b_eq += b_eq
        self.z_map_constr_eq = sparse.hstack((self.z_map_constr_eq, z_eq)).tocsc()

        # Remove all duplicates from ineq
        if ident_rows == []:
            remove_ineq = np.array([], 'int')
        else:
            remove_ineq = np.unique(np.hstack([[ir[0], ir[1]] if ir[2] == -1 else [ir[1]] for ir in ident_rows]))
        keep_ineq = [i for i in range(self.A_ineq.shape[0]) if i not in remove_ineq]
        knockable_constr_ineq_ic = [True if i in knockable_constr_ineq_ic else False for i in range(self.A_ineq.shape[0])]
        knockable_constr_ineq_ic = np.nonzero([knockable_constr_ineq_ic[i] for i in keep_ineq])[0]
        self.A_ineq = self.A_ineq[keep_ineq, :]
        self.b_ineq = [self.b_ineq[i] for i in keep_ineq]
        self.z_map_constr_ineq = self.z_map_constr_ineq[:, keep_ineq]

        # b5. Link remaining (in)equalities to z via indicator constraints
        #    and remove them from the static problem
        #    - first ineqs then eqs
        ic_binv = np.append(self.z_map_constr_ineq[:, knockable_constr_ineq_ic].indices,
                            self.z_map_constr_eq[:, knockable_constr_eq_ic].indices)
        ic_A = sparse.vstack((self.A_ineq[knockable_constr_ineq_ic, :], self.A_eq[knockable_constr_eq_ic, :]))
        ic_b = [self.b_ineq[i] for i in knockable_constr_ineq_ic] + [self.b_eq[i] for i in knockable_constr_eq_ic]
        ic_sense = 'L' * len(knockable_constr_ineq_ic) + 'E' * len(knockable_constr_eq_ic)
        ic_indicval = np.append(self.z_map_constr_ineq[:, knockable_constr_ineq_ic].data, self.z_map_constr_eq[:,
                                                                                                               knockable_constr_eq_ic].data)
        ic_indicval = [0 if i == 1 else 1 for i in ic_indicval]
        # in z-maps: -1 => z=1 -> A_ineq*x <= b_ineq
        #             1 => z=0 -> A_ineq*x <= b_ineq
        self.indic_constr = IndicatorConstraints(ic_binv, ic_A, ic_b, ic_sense, ic_indicval)

        # b6. Remove knockable (in)equalities from static problem, as they are now indicator constraints
        keep_ineq = [False if i in knockable_constr_ineq_ic else True for i in range(self.A_ineq.shape[0])]
        self.A_ineq = self.A_ineq[keep_ineq, :]
        self.b_ineq = [self.b_ineq[i] for i in range(len(keep_ineq)) if keep_ineq[i]]
        keep_eq = [False if i in knockable_constr_eq_ic else True for i in range(self.A_eq.shape[0])]
        self.A_eq = self.A_eq[keep_eq, :]
        self.b_eq = [self.b_eq[i] for i in range(len(keep_eq)) if keep_eq[i]]


class ContMILP:
    """Continuous representation of the strain design MILP.
    
    This MILP can be used to verify computation results. Since this class also stores the relationship
    between intervention variables z and corresponding (in)equality constraints and variables in the 
    problem, it can be used to verify computed designs quickly and in a numerically stable manner."""

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

def build_primal_from_cbm(model, V_ineq=None, v_ineq=None, V_eq=None, v_eq=None, c=None) -> \
        Tuple[sparse.csr_matrix, Tuple, sparse.csr_matrix, Tuple, Tuple, Tuple, sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    """Builds primal LP from constraint-based model and (optionally) additional constraints.
    
    standard form: A_ineq x <= b_ineq, A_eq x = b_eq, lb <= x <= ub, min{c'x}. Additionally,
    this function also returns a set of matrices that associate each variable (and constraint)
    with reactions. In the primal problems all variables correspond to reactions (z), therefore,
    the z_map_vars matrix is an identity matrix. The constraints correspond to metabolites, thus
    z_map_constr_ineq, z_map_constr_eq are all-zero.
    
    Args:
        model (cobra.Model):
            A metabolic model that is an instance of the cobra.Model class
            
        V_ineq, v_ineq (sparse.csr_matrix, list of float):
            Linear inequality constraints of the form V_ineq*x <= v_ineq. Ensure that the number
            of columns in V_ineq is identical to the number of reactions in the model.
            
        V_eq, v_eq (sparse.csr_matrix, list of float):
            Linear equality constraints of the form V_eq*x = v_eq. Ensure that the number
            of columns in V_eq is identical to the number of reactions in the model.
            
        c (list of float):
            Object coefficient vector (same lenght as variable vector).
    
    Returns:
        (Tuple):
        A_ineq, b_ineq, A_eq, b_eq, lb, ub, c, z_map_constr_ineq, z_map_constr_eq, z_map_vars.
        A constraint-based steady-state model in the form of a linear (in)equality system. The matrices
        z_map_constr_ineq, z_map_constr_eq, z_map_vars contain the association between reactions and
        different parts of the LP, such as reactions, metabolites or other (in)equalities.
    """
    numr = len(model.reactions)
    # initialize matices (if not provided in function call)
    if V_ineq is None or v_ineq is None:
        V_ineq = sparse.csr_matrix((0, numr))
        v_ineq = []
    if V_eq is None or v_eq is None:
        V_eq = sparse.csr_matrix((0, numr))
        v_eq = []
    if c is None:
        c = [i.objective_coefficient for i in model.reactions]
    S = sparse.csr_matrix(create_stoichiometric_matrix(model))
    # fill matrices
    A_eq = sparse.vstack((S, V_eq))
    b_eq = [0 for _ in range(S.shape[0])] + v_eq
    A_ineq = V_ineq.copy()
    b_ineq = v_ineq.copy()
    lb = [v.lower_bound for v in model.reactions]
    ub = [v.upper_bound for v in model.reactions]
    z_map_vars = sparse.identity(numr, 'd', format="csc")
    z_map_constr_eq = sparse.csc_matrix((numr, A_eq.shape[0]))
    z_map_constr_ineq = sparse.csc_matrix((numr, A_ineq.shape[0]))
    A_ineq, b_ineq, lb, ub, z_map_constr_ineq = prevent_boundary_knockouts(A_ineq, b_ineq, lb.copy(), ub.copy(), z_map_constr_ineq,
                                                                           z_map_vars)
    return A_ineq, b_ineq, A_eq, b_eq, lb, ub, c, z_map_constr_ineq, z_map_constr_eq, z_map_vars

def LP_dualize(A_ineq_p, b_ineq_p, A_eq_p, b_eq_p, lb_p, ub_p, c_p,
            z_map_constr_ineq_p=None, z_map_constr_eq_p=None, z_map_vars_p=None) -> \
        Tuple[sparse.csr_matrix, Tuple, sparse.csr_matrix, Tuple, Tuple, Tuple, sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    """Translate a primal system to its LP dual system
    
    The primal system must be given in the standard form: A_ineq x <= b_ineq, A_eq x = b_eq, lb <= x < ub, min{c'x}. The LP duality
    theorem defines a set of two problems. If one of the LPs is a maximization and and optimum exists, the optimal value of this LP
    is identical to the minimal optimum of its LP dual problem. LP duality can be used for nested optimization, since solving the 
    primal and the LP dual problem, while enfocing equality of the objective value, guarantees optimality.
    
    Construction of the LP dual:    
        Variables translate to constraints:
            x={R} ->   =
            x>=0  ->  >= (new constraint is multiplied with -1 to translate to <= e.g. -A_i' y <= -c_i)
            x<=0  ->  <=
        Constraints translate to variables:
            =     ->   y={R}
            <=    ->   y>=0
            
    Args:
        A_ineq_p, b_ineq_p (sparse.csr_matrix and list of float): 
            A coefficient matrix and a vector that describe the linear inequalities of the primal LP A_ineq_p*x <= b_ineq_p
            
        A_eq_p, b_eq_p (sparse.csr_matrix and list of float): 
            A coefficient matrix and a vector that describe the linear equalities of the primal LP A_eq_p*x <= b_eq_p
            
        lb_p, ub_p (list of float):
            Upper and lower variable bounds in vector form.
            
        c_p (list of float):
            The objective coefficient vector of the primal minimization-LP.
            z_map_constr_ineq_p, z_map_constr_eq_p, z_map_vars_p
                    
        z_map_constr_ineq, z_map_constr_eq, z_map_vars (optional (sparse.csr_matrix)):
            Matrices that contain the relationship between metabolic reactions and different parts of the LP, such as reactions, 
            metabolites or other (in)equalities. These matrices help keeping track of the parts of the LP that are affected by
            reaction knockouts and additions. When a reaction (i) knockout removes the variable or constraint (j), the respective
            matrix contains a coefficient 1 at this position. -1 marks additions. E.g.: If the knockout of reaction i corresponds
            to the removal of inequality constraint j, there is a matrix entry z_map_constr_ineq_(i,j) = 1. If these matrices are
            provided, they are updated for the dualized LP, if not, the dual problem is constructed without returning information 
            about these relationships.
            
    Returns
        (Tuple):
        The LP dual of the problem in the format: A_ineq, b_ineq, A_eq, b_eq, c, lb, ub
        and optionally also z_map_constr_ineq, z_map_constr_eq, z_map_vars
    """
    numr = A_ineq_p.shape[1]
    numz = max([z.shape[0] for z in [z_map_vars_p, z_map_constr_eq_p, z_map_constr_ineq_p] if z is not None])

    if numz:
        if z_map_vars_p is None:
            z_map_vars_p = sparse.csc_matrix((numz, A_ineq_p.shape[1]))
        if z_map_constr_eq_p is None:
            z_map_constr_eq_p = sparse.csc_matrix((numz, A_eq_p.shape[0]))
        if z_map_constr_ineq_p is None:
            z_map_constr_ineq_p = sparse.csc_matrix((numz, A_ineq_p.shape[0]))

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
            "knockouts of variables and constraints must not overlap in the problem matrix. Something went wrong during the construction of the primal problem."
        )

    if c_p == []:
        c_p = [0.0] * numr

    # Translate inhomogenous bounds into inequality constraints
    lb_inh_bounds = [i for i in np.nonzero(lb_p)[0] if not np.isinf(lb_p[i])]
    ub_inh_bounds = [i for i in np.nonzero(ub_p)[0] if not np.isinf(ub_p[i])]
    x_geq0 = np.nonzero(np.greater_equal(lb_p, 0) & np.greater(ub_p, 0))[0]
    x_eR = np.nonzero(np.greater(0, lb_p) & np.greater(ub_p, 0))[0]
    x_leq0 = np.nonzero(np.greater(0, lb_p) & np.greater_equal(0, ub_p))[0]

    LB = sparse.csr_matrix((len(lb_inh_bounds) * [-1], (range(0, len(lb_inh_bounds)), lb_inh_bounds)), shape=(len(lb_inh_bounds), numr))
    UB = sparse.csr_matrix((len(ub_inh_bounds) * [1], (range(0, len(ub_inh_bounds)), ub_inh_bounds)), shape=(len(ub_inh_bounds), numr))
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
    lb = [-np.inf] * A_eq_p.shape[0] + [0.0] * A_ineq_p.shape[0]
    ub = [np.inf] * (A_eq_p.shape[0] + A_ineq_p.shape[0])
    c = b_eq_p + b_ineq_p

    if numz:
        # translate mapping of z-variables to rows instead of columns
        z_map_constr_ineq = sparse.hstack((z_map_vars_p[:, x_geq0], z_map_vars_p[:, x_leq0])).tocsc()
        z_map_constr_eq = z_map_vars_p[:, x_eR]
        z_map_vars = sparse.hstack(
            (z_map_constr_eq_p, z_map_constr_ineq_p, sparse.csc_matrix((numz, len(lb_inh_bounds) + len(ub_inh_bounds))))).tocsc()
        A_ineq, b_ineq, A_eq, b_eq, lb, ub, z_map_constr_ineq, z_map_constr_eq = \
            reassign_lb_ub_from_ineq(A_ineq, b_ineq, A_eq, b_eq, lb, ub,  z_map_constr_ineq, z_map_constr_eq, z_map_vars)
        return A_ineq, b_ineq, A_eq, b_eq, lb, ub, c, z_map_constr_ineq, z_map_constr_eq, z_map_vars
    else:
        A_ineq, b_ineq, A_eq, b_eq, lb, ub = reassign_lb_ub_from_ineq(A_ineq, b_ineq, A_eq, b_eq, lb, ub)
        return A_ineq, b_ineq, A_eq, b_eq, lb, ub, c

def farkas_dualize(A_ineq_p, b_ineq_p, A_eq_p, b_eq_p, lb_p, ub_p,
                  z_map_constr_ineq_p=None, z_map_constr_eq_p=None, z_map_vars_p=None) -> \
        Tuple[sparse.csr_matrix, Tuple, sparse.csr_matrix, Tuple, Tuple, sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    """Translate a primal system of linear (in)equality to its Farkas dual
    
    The primal system must be given in the standard form: A_ineq x <= b_ineq, A_eq x = b_eq, lb <= x < ub. Farkas' lemma
    defines a set of two systems of linear (in)equalities of which exactly one is feasible. Since the feasibility of one is a
    certificate for the infeasibility of the other one, this theorem can be used to set up problems that imply the infeasibility
    and thus exclusion of a certain subspace. This priciple is used for MCS calculation (the SUPPRESS module).
    
    Consider that the following is not implemented:
    In the case of (1) A x = b, (2) x={R}, (3) b~=0, Farkas' lemma is special,
    because b'y ~= 0 is required to make the primal infeasible instead of b'y < 0.
    1. This does not occur very often.
    2. Splitting the equality into two inequalities that translate to y>=0
    would be posible, and yield b'y < 0 in the farkas' lemma.
    Maybe splitting is required, but I actually don't think so. Using the
    special case of b'y < 0 for b'y ~= 0 should be enough.
    
    Args:
        A_ineq_p, b_ineq_p (sparse.csr_matrix and list of float): 
            A coefficient matrix and a vector that describe the linear inequalities of the primal LP A_ineq_p*x <= b_ineq_p
            
        A_eq_p, b_eq_p (sparse.csr_matrix and list of float): 
            A coefficient matrix and a vector that describe the linear equalities of the primal LP A_eq_p*x <= b_eq_p
            
        lb_p, ub_p (list of float):
            Upper and lower variable bounds in vector form.
                    
        z_map_constr_ineq, z_map_constr_eq, z_map_vars (optional (sparse.csr_matrix)):
            Matrices that contain the relationship between metabolic reactions and different parts of the LP, such as reactions, 
            metabolites or other (in)equalities. These matrices help keeping track of the parts of the LP that are affected by
            reaction knockouts and additions. When a reaction (i) knockout removes the variable or constraint (j), the respective
            matrix contains a coefficient 1 at this position. -1 marks additions. E.g.: If the knockout of reaction i corresponds
            to the removal of inequality constraint j, there is a matrix entry z_map_constr_ineq_(i,j) = 1. If these matrices are
            provided, they are updated for the dualized LP, if not, the dual problem is constructed without returning information 
            about these relationships.
            
    Returns
        (Tuple):
        The Farkas dual of the linear (in)equality system in the format: A_ineq, b_ineq, A_eq, b_eq, lb, ub
        and optionally also z_map_constr_ineq, z_map_constr_eq, z_map_vars
    """
    numz = max([0] + [z.shape[0] for z in [z_map_vars_p, z_map_constr_eq_p, z_map_constr_ineq_p] if z is not None])
    c_p = [0 for _ in range(A_ineq_p.shape[1])]

    if numz:
        A_ineq_d, b_ineq_d, A_eq_d, b_eq_d, lb_f, ub_f, c_d, z_map_constr_ineq_d, z_map_constr_eq_f, z_map_vars_f = LP_dualize(
            A_ineq_p, b_ineq_p, A_eq_p, b_eq_p, lb_p, ub_p, c_p, z_map_constr_ineq_p, z_map_constr_eq_p, z_map_vars_p)
    else:
        A_ineq_d, b_ineq_d, A_eq_d, b_eq_d, lb_f, ub_f, c_d = LP_dualize(A_ineq_p, b_ineq_p, A_eq_p, b_eq_p, lb_p, ub_p, c_p)
    # add constraint b_prim'y or (c_dual'*y) <= -1;
    A_ineq_f = sparse.vstack((A_ineq_d, sparse.csr_matrix(c_d))).tocsr()
    b_ineq_f = b_ineq_d + [-1]
    A_eq_f = A_eq_d
    b_eq_f = b_eq_d
    # it would also be possible (but ofc not necessary) to force (c_dual*y) == -1; instead
    # A_eq = sparse.vstack((A_eq,c_dual)).tocsr()
    # b_eq += [-1]
    if numz:
        z_map_constr_ineq_f = sparse.hstack((z_map_constr_ineq_d, sparse.csr_matrix((numz, 1)))).tocsc()
        # z_map_constr_eq_f = sparse.hstack((z_map_constr_eq_d,sparse.csr_matrix((numz,1)))).tocsc()
        return A_ineq_f, b_ineq_f, A_eq_f, b_eq_f, lb_f, ub_f, z_map_constr_ineq_f, z_map_constr_eq_f, z_map_vars_f
    else:
        return A_ineq_f, b_ineq_f, A_eq_f, b_eq_f, lb_f, ub_f, z_map_constr_ineq_f, z_map_constr_eq_f, z_map_vars_f

def reassign_lb_ub_from_ineq(A_ineq, b_ineq, A_eq, b_eq, lb, ub,
                             z_map_constr_ineq=None, z_map_constr_eq=None, z_map_vars=None) -> \
        Tuple[sparse.csr_matrix, Tuple, sparse.csr_matrix, Tuple, Tuple, Tuple, sparse.csr_matrix, sparse.csr_matrix]:
    """Remove single constraints in A_ineq or A_eq in favor of lower and upper bounds on variables
    
    Constraints on single variables instead translated into lower and upper bounds (lb, ub). This
    is useful to filter out redundant bounds on variables and keep the (in)equality system concise. 
    To avoid interference with the knock-out logic, negative upper bounds and positive lower bounds 
    are not put into lb and ub, when reactions are flagged knockable with z_map_vars.
    
    Args:
        A_ineq_p, b_ineq_p (sparse.csr_matrix and list of float): 
            A coefficient matrix and a vector that describe the linear inequalities of the primal LP A_ineq_p*x <= b_ineq_p
            
        A_eq_p, b_eq_p (sparse.csr_matrix and list of float): 
            A coefficient matrix and a vector that describe the linear equalities of the primal LP A_eq_p*x <= b_eq_p
            
        lb_p, ub_p (list of float):
            Upper and lower variable bounds in vector form.
            
        z_map_constr_ineq, z_map_constr_eq, z_map_vars (optional (sparse.csr_matrix)):
            Matrices that contain the relationship between metabolic reactions and different parts of the LP, such as reactions, 
            metabolites or other (in)equalities. These matrices help keeping track of the parts of the LP that are affected by
            reaction knockouts and additions. When a reaction (i) knockout removes the variable or constraint (j), the respective
            matrix contains a coefficient 1 at this position. -1 marks additions. E.g.: If the knockout of reaction i corresponds
            to the removal of inequality constraint j, there is a matrix entry z_map_constr_ineq_(i,j) = 1. If these matrices are
            provided, they are updated. Otherwise, all reactions are assumed to be notknockable and thus all constraints on 
            single variables put into lb and ub.
    
    Returns
        (Tuple):
        A linear (in)equality system in the format: A_ineq, b_ineq, A_eq, b_eq, lb, ub
        and optionally also updated z_map_constr_ineq, z_map_constr_eq
    """
    lb = [[l] for l in lb]
    ub = [[u] for u in ub]
    numz = max([0] + [z.shape[0] for z in [z_map_vars, z_map_constr_eq, z_map_constr_ineq] if z is not None])
    numr = A_ineq.shape[1]

    if z_map_vars is None:
        z_map_vars = sparse.csc_matrix((numz, numr))

    # translate entries to lb or ub
    # find all entries in A_ineq
    row_ineq = A_ineq.nonzero()[0]
    # filter for rows with only one entry
    var_bound_constraint_ineq = [i for i in row_ineq if list(row_ineq).count(i) == 1]
    # exclude knockable constraints
    var_bound_constraint_ineq = [i for i in var_bound_constraint_ineq if i not in z_map_constr_ineq.nonzero()[1]]
    # retrieve all bounds from inequality constraints
    for i in var_bound_constraint_ineq:
        idx_r = A_ineq[i, :].nonzero()[1][0]  # get reaction from constraint (column of entry)
        if A_ineq[i, idx_r] > 0:  # upper bound constraint
            ub[idx_r] += [b_ineq[i] / A_ineq[i, idx_r]]
        else:  # lower bound constraint
            lb[idx_r] += [b_ineq[i] / A_ineq[i, idx_r]]

    # find all entries in A_eq
    row_eq = A_eq.nonzero()[0]
    # filter for rows with only one entry
    var_bound_constraint_eq = [i for i in row_eq if list(row_eq).count(i) == 1]
    # exclude knockable constraints
    var_bound_constraint_eq = [i for i in var_bound_constraint_eq if i not in z_map_constr_eq.nonzero()[1]]
    # retrieve all bounds from equality constraints
    # and partly set lb or ub derived from equality constraints, for instance:
    # If x =  5, set ub = 5 and keep the inequality constraint -x <= -5.
    # If x = -5, set lb =-5 and keep the inequality constraint  x <= -5.
    A_ineq_new = sparse.csr_matrix((0, numr))
    b_ineq_new = []
    for i in var_bound_constraint_eq:
        idx_r = A_eq[i, :].nonzero()[1][0]  # get reaction from constraint (column of entry)
        if any(z_map_vars[:, idx_r]):  # if reaction is knockable
            if A_eq[i, idx_r] * b_eq[i] > 0:  # upper bound constraint
                ub[idx_r] += [b_eq[i] / A_eq[i, idx_r]]
                A_ineq_new = sparse.vstack((A_ineq_new, -A_eq[i, :]))
                b_ineq_new += [-b_eq[i]]
            elif A_eq[i, idx_r] * b_eq[i] < 0:  # lower bound constraint
                lb[idx_r] += [b_eq[i] / A_eq[i, idx_r]]
                A_ineq_new = sparse.vstack((A_ineq_new, A_eq[i, :]))
                b_ineq_new += [b_eq[i]]
            else:
                ub[idx_r] += [0.0]
                lb[idx_r] += [0.0]
        else:
            lb[idx_r] += [b_eq[i] / A_eq[i, idx_r]]
            ub[idx_r] += [b_eq[i] / A_eq[i, idx_r]]
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
    numineq = A_ineq.shape[0]
    A_ineq = A_ineq[[False if i in var_bound_constraint_ineq else True for i in range(0, numineq)]]
    b_ineq = [b_ineq[i] for i in range(0, len(b_ineq)) if i not in var_bound_constraint_ineq]
    z_map_constr_ineq = z_map_constr_ineq[:, [False if i in var_bound_constraint_ineq else True for i in range(0, numineq)]]
    numeq = A_eq.shape[0]
    A_eq = A_eq[[False if i in var_bound_constraint_eq else True for i in range(0, numeq)]]
    b_eq = [b_eq[i] for i in range(0, len(b_eq)) if i not in var_bound_constraint_eq]
    # add equality constraints that transformed to inequality constraints
    A_ineq = sparse.vstack((A_ineq, A_ineq_new))
    b_ineq += b_ineq_new
    if numz:
        z_map_constr_eq = z_map_constr_eq[:, [False if i in var_bound_constraint_eq else True for i in range(0, numeq)]]
        z_map_constr_ineq = sparse.hstack((z_map_constr_ineq, sparse.csc_matrix((numz, A_ineq_new.shape[0]))))
        return A_ineq, b_ineq, A_eq, b_eq, lb, ub, z_map_constr_ineq, z_map_constr_eq
    else:
        return A_ineq, b_ineq, A_eq, b_eq, lb, ub

def prevent_boundary_knockouts(A_ineq, b_ineq, lb, ub, z_map_constr_ineq, z_map_vars) -> \
        Tuple[sparse.csr_matrix, Tuple, Tuple, Tuple, sparse.csr_matrix]:
    """Put negative lower bounds and positive upper bounds into (notknockable) inequalities
    
    This is a helper function that puts negative lower bounds and positive upper bounds into (not-knockable) 
    inequalities. Later on, one may simulate the knockouts by multiplying the upper and lower bounds with a
    binary variable z. This functions prevents that 
    
    Args:
        A_ineq_p, b_ineq_p (sparse.csr_matrix and list of float): 
            A coefficient matrix and a vector that describe the linear inequalities of the primal LP A_ineq_p*x <= b_ineq_p
            
        lb_p, ub_p (list of float):
            Upper and lower variable bounds in vector form.
            
        z_map_constr_ineq, z_map_vars (optional (sparse.csr_matrix)):
            Matrices that contain the relationship between metabolic reactions and different parts of the LP, such as reactions, 
            metabolites or other (in)equalities. These matrices help keeping track of the parts of the LP that are affected by
            reaction knockouts and additions. When a reaction (i) knockout removes the variable or constraint (j), the respective
            matrix contains a coefficient 1 at this position. -1 marks additions. E.g.: If the knockout of reaction i corresponds
            to the removal of inequality constraint j, there is a matrix entry z_map_constr_ineq_(i,j) = 1. If these matrices are
            provided, they are updated. Otherwise, all reactions are assumed to be knockable and thus all negative upper and 
            positive lower bounds are translated into constraints.           
    
    Returns
        (Tuple):
        A linear (in)equality system in the format: A_ineq, b_ineq, A_eq, b_eq, lb, ub
        and optionally also updated z_map_constr_ineq, z_map_constr_eq
    """
    numr = A_ineq.shape[1]
    numz = max([0] + [z.shape[0] for z in [z_map_vars, z_map_constr_ineq] if z is not None])

    if z_map_vars is None:
        z_map_vars = sparse.csc_matrix((numz, numr))

    for i in range(0, numr):
        if any(z_map_vars[:, i]) and lb[i] > 0:
            A_ineq = sparse.vstack((A_ineq, sparse.csr_matrix(([-1], ([0], [i])), shape=(1, numr))))
            b_ineq += [-lb[i]]
            z_map_constr_ineq = sparse.hstack((z_map_constr_ineq, sparse.csc_matrix((numz, 1))))
            lb[i] = 0.0
        if any(z_map_vars[:, i]) and ub[i] < 0:
            A_ineq = sparse.vstack((A_ineq, sparse.csr_matrix(([1], ([0], [i])), shape=(1, numr))))
            b_ineq += [ub[i]]
            z_map_constr_ineq = sparse.hstack((z_map_constr_ineq, sparse.csc_matrix((numz, 1))))
            ub[i] = 0.0

    return A_ineq, b_ineq, lb, ub, z_map_constr_ineq


def worker_init(A, A_ineq, b_ineq, A_eq, b_eq, lb, ub, solver, seed):
    """Helper function for determining bounds on linear expressions"""
    global lp_glob
    lp_glob = MILP_LP(A_ineq=A_ineq, b_ineq=b_ineq, A_eq=A_eq, b_eq=b_eq, lb=lb, ub=ub, solver=solver, seed=seed)
    if lp_glob == CPLEX:
        lp_glob.backend.parameters.lpmethod.set(1)
        if Configuration().processes > 1:
            lp_glob.backend.parameters.threads.set(2)
    lp_glob.solver = solver
    lp_glob.A = A


def worker_compute(i) -> Tuple[int, float]:
    """Helper function for determining bounds on linear expressions"""
    global lp_glob
    # maximize by minimizing negative objective and negating result
    lp_glob.set_objective(-lp_glob.A[i])
    min_cx = -lp_glob.slim_solve()
    return i, min_cx
