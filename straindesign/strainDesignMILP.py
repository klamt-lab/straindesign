#!/usr/bin/env python3
#
# Copyright 2022-2026 Max Planck Institute Magdeburg
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
"""Classes and function for the solution of strain design MILPs"""

import numpy as np
from scipy import sparse
import time
from typing import Dict, List, Tuple
from straindesign import SDProblem, SDSolutions, MILP_LP, SDModule, Model
from straindesign.names import *
import logging


class SDMILP(SDProblem, MILP_LP):
    """Class that contains functions for the solution of the strain design MILP
     
    This class is a wrapper and inherited from the casses SDProblem, MILP_LP.
    The constructor of SDProblem (see strainDesignProblem.py) translates a given
    problem into a MILP. The constructor of MILP_LP (see solver_interface.py) then
    sets up the solver interface for the selected solver. In addition to the functions
    from SDProblem and MILP_LP, SDMILP provides functions for the solution of the 
    strain design MILP, such as verification of strain design solutions or introduction 
    of exclusion constraints for computing multiple solutions.
    
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
        (SDMILP):
            An instance of SDProblem containing the strain design MILP and providing several functions for its solution
    """

    def __init__(self, model: Model, sd_modules: List[SDModule], **kwargs):
        # Construct problem
        SDProblem.__init__(self, model, sd_modules, **kwargs)
        # A knock-in only enlarges the flux space and a knock-out only shrinks it. So when every
        # module is of the kind that survives that direction, a rewarding intervention can never
        # invalidate a design, and every design is beaten by the one that also takes it. Fixing
        # it here is exact and spares the enumeration from walking the designs it dominates.
        _types = {m[MODULE_TYPE] for m in sd_modules}
        _forced = []
        for i in range(self.num_z):
            if self.z_non_targetable[i] or np.isnan(self.cost[i]) or self.cost[i] >= 0.0:
                continue
            if (self.z_inverted[i] and _types == {PROTECT}) or \
               (not self.z_inverted[i] and _types == {SUPPRESS}):
                _forced.append(i)
                logging.info('  Rewarding intervention %s cannot invalidate a design here and is '
                             'taken in every one of them.' % model.reactions[i].id)
        if _forced:
            # as a row rather than by fixing the bound: a binary pinned by lb == ub sends SCIP's
            # populate into a loop where it re-reports the same design indefinitely
            rows = sparse.lil_matrix((len(_forced), self.A_ineq.shape[1]))
            for k, i in enumerate(_forced):
                rows[k, i] = -1.0
            self.A_ineq = sparse.vstack((self.A_ineq, rows.tocsr()), format='csr')
            self.b_ineq = list(self.b_ineq) + [-1.0] * len(_forced)
        # Remove non-knockable z-variables before solver sees them
        self._trim_z_variables()
        # Interventions that do not cost anything to take. Adding one to a design can only keep
        # its cost equal or lower, so such a design is not dominated by the smaller one and the
        # exclusion constraints must leave it reachable. Empty for the usual all-positive setup,
        # where dominance and set inclusion agree and the cuts stay exactly as they were.
        self._free_z = [i for i in range(self.num_z) if not self.z_non_targetable[i] and self.cost[i] <= 0.0]
        # A rewarding intervention strictly lowers the cost of any design that can absorb it,
        # so a design is only worth reporting once none of them can be added while staying valid.
        self._rewarding_z = [i for i in self._free_z if self.cost[i] < 0.0]
        # Build MILP object from constructed problem
        MILP_LP.__init__(self,
                         c=self.c,
                         A_ineq=self.A_ineq,
                         b_ineq=self.b_ineq,
                         A_eq=self.A_eq,
                         b_eq=self.b_eq,
                         lb=self.lb,
                         ub=self.ub,
                         vtype=self.vtype,
                         indic_constr=self.indic_constr,
                         M=self.M,
                         solver=self.solver,
                         seed=self.seed,
                         milp_threads=self.milp_threads)

    def is_dominated(self, z):
        """True if a rewarding intervention can join this design without invalidating it.

        Such a design is strictly cheaper and contains this one, so this one is not worth
        reporting. Only reachable when some intervention carries a negative cost.
        """
        for j in self._rewarding_z:
            if z[0, j]:
                continue
            z_more = z.tolil()
            z_more[0, j] = 1.0
            if all(self.verify_sd(z_more.tocsr())):
                return True
        return False

    def _trim_z_variables(self):
        """Remove non-knockable (ub=0) z-variables from MILP matrices.

        Non-knockable reactions have ub=0 and cost=0. With Presolve=0 the
        solver carries these dead variables. This trims them at construction
        time. Stores _z_orig_indices for expanding solutions back.
        """
        keep_z = [i for i in range(self.num_z) if self.ub[i] > 0]
        if len(keep_z) == self.num_z:
            self._z_orig_indices = None  # no trimming needed
            return

        self._z_orig_indices = keep_z  # trimmed_idx -> orig_idx
        self._orig_num_z = self.num_z

        n_cont = len(self.c) - self.num_z
        keep_cols = keep_z + list(range(self.num_z, self.num_z + n_cont))

        # Trim constraint matrices (column selection)
        self.A_ineq = self.A_ineq[:, keep_cols]
        self.A_eq = self.A_eq[:, keep_cols]
        self.c = [self.c[i] for i in keep_cols]
        self.lb = [self.lb[i] for i in keep_cols]
        self.ub = [self.ub[i] for i in keep_cols]

        # Trim indicator constraints
        if hasattr(self.indic_constr, 'A') and self.indic_constr.A is not None:
            old_to_new = {old: new for new, old in enumerate(keep_z)}
            self.indic_constr.A = self.indic_constr.A[:, keep_cols]
            self.indic_constr.binv = [old_to_new[b] for b in self.indic_constr.binv]

        # Update z-related arrays (trim to knockable-only)
        new_num_z = len(keep_z)
        self.cost = [self.cost[i] for i in keep_z]
        self.z_inverted = [self.z_inverted[i] for i in keep_z]
        self.z_non_targetable = [self.z_non_targetable[i] for i in keep_z]
        self.idx_z = list(range(new_num_z))
        self.num_z = new_num_z
        self.vtype = 'B' * new_num_z + 'C' * n_cont
        self.c_bu = [float(i) for i in self.c]

        logging.info(f'  Trimmed z-variables: {self._orig_num_z} -> {new_num_z} '
                     f'({self._orig_num_z - new_num_z} non-knockable removed)')

    def _expand_z_to_orig(self, z_trimmed):
        """Expand trimmed z-solution back to original z-space."""
        if self._z_orig_indices is None:
            return z_trimmed
        n_rows = z_trimmed.shape[0]
        expanded = sparse.lil_matrix((n_rows, self._orig_num_z))
        z_coo = z_trimmed.tocoo()
        for row, col, val in zip(z_coo.row, z_coo.col, z_coo.data):
            expanded[row, self._z_orig_indices[col]] = val
        return expanded.tocsr()

    def add_exclusion_constraints(self, z):
        """Exclude a design and every superset of it that cannot be cheaper.

        With only positive intervention costs a superset always costs more, so this excludes
        all supersets and the rule is plain set inclusion. When some interventions are free or
        rewarding (cost <= 0), a superset taking one of those is not dominated by the design
        found here, so those literals enter the row negated and such supersets stay reachable.
        """
        for i in range(z.shape[0]):
            free = [j for j in self._free_z if not z[i, j]]
            # introduce constraint to make MILP infeasible. Some solvers cannot handle empty rows
            if z[i].nnz == 0 and not free:
                A_ineq = sparse.csr_matrix([1.0] * z[i].shape[1])
                A_ineq.resize((1, self.A_ineq.shape[1]))
                b_ineq = -1
                self.add_ineq_constraints(A_ineq, [b_ineq])
            # a single intervention can be switched off outright, but only while no free
            # intervention could join it to form a design that is not dominated by this one
            elif z[i].nnz == 1 and not free:
                interv_idx = int(z[i].indices[0])
                self.z_non_targetable[interv_idx] = True
                self.set_ub([[interv_idx, 0.0]])
            # otherwise, introduce integer cut constraint
            else:
                A_ineq = z[i].copy()
                A_ineq.resize((1, self.A_ineq.shape[1]))
                b_ineq = np.sum(z[i]) - 1
                if free:
                    A_ineq = A_ineq.tolil()
                    for j in free:
                        A_ineq[0, j] = -1.0
                    A_ineq = A_ineq.tocsr()
                self.add_ineq_constraints(A_ineq, [b_ineq])

    def add_exclusion_constraints_ineq(self, z):
        """Exclude exact binary solution in z (but not its supersets) from MILP.

        For each solution row in z, adds: sum(z_active) - sum(z_inactive) <= |active| - 1
        This excludes the exact pattern without blocking supersets or subsets.
        """
        z_dense = z.toarray()
        for j in range(z.shape[0]):
            n_active = int(np.sum(z_dense[j] != 0))
            if n_active == 0:
                continue
            coeffs = [1.0 if z_dense[j, i] else -1.0 for i in range(z.shape[1])]
            A_row = sparse.csr_matrix([coeffs])
            A_row.resize((1, self.A_ineq.shape[1]))
            b_ineq = n_active - 1
            self.add_ineq_constraints(A_row, [b_ineq])

    def sd2dict(self, sol, *args) -> Dict:
        """Translate binary solution vector to dictionary for human-readable output"""
        output = {}
        reacID = self.model.reactions.list_attr("id")
        for i in self.idx_z:
            orig_i = self._z_orig_indices[i] if self._z_orig_indices is not None else i
            if sol[0, i] != 0 and not np.isnan(sol[0, i]):
                if self.z_inverted[i]:
                    output[reacID[orig_i]] = sol[0, i]
                else:
                    output[reacID[orig_i]] = -sol[0, i]
            elif args and args[0] and (sol[0, i] == 0) and self.z_inverted[i]:
                output[reacID[orig_i]] = 0.0
        return output

    def solveZ(self) -> Tuple[List, int]:
        """Solve MILP, and return only binary variables rounded to 5 decimals (should return ints)"""
        x, opt, status = self.solve()
        z = sparse.csr_matrix([round(x[i], 5) for i in self.idx_z])
        return z, x, opt, status

    def populateZ(self, n) -> Tuple[List, int]:
        """Populate MILP, and return only binary variables rounded to 5 decimals (should return ints)"""
        x, _, status = self.populate(n)
        if status in [OPTIMAL, TIME_LIMIT_W_SOL]:
            z = sparse.csr_matrix([[round(x[j][i], 5) for i in self.idx_z] for j in range(len(x))])
            z.resize((len(x), self.num_z))
            # remove duplicates
            unique_row_indices, unique_columns = [], []
            for row_idx, row in enumerate(z):
                indices = row.indices.tolist()
                if indices not in unique_columns:
                    unique_columns.append(indices)
                    unique_row_indices.append(row_idx)
            z = z[unique_row_indices]
        else:
            z = sparse.csr_matrix((0, self.num_z))
        return z, status

    def fixObjective(self, c, cx):
        """Enforce a certain objective function and value (or any other constraint of the form c*x <= cx)"""
        self.set_ineq_constraint(self.idx_row_obj, c, cx)

    def resetObjective(self):
        """Reset objective to the one set upon MILP construction"""
        self.set_objective_idx([[i, v] for i, v in enumerate(self.c_bu)])

    def setMinIntvCostObjective(self):
        """Reset minimization of intervention costs as global objective"""
        self.clear_objective()
        self.set_objective_idx([[i, self.cost[i]] for i in self.idx_z if i not in self.z_non_targetable])

    def resetTargetableZ(self):
        """Reset targetable/switchable intervention indicators / allow all intervention candidates"""
        self.set_ub([[i, 1.0] for i in self.idx_z if not self.z_non_targetable[i]])

    def setTargetableZ(self, sol):
        """Only allow a subset of intervention candidates"""
        self.set_ub([[i, 0.0] for i in self.idx_z if not sol[0, i]])

    def verify_sd(self, sols) -> List:
        """Verify computed strain design"""
        sols_orig = self._expand_z_to_orig(sols)
        valid = [False] * sols_orig.shape[0]
        for i, sol in zip(range(sols_orig.shape[0]), sols_orig):
            inactive_vars = [var for z_i,var,sense in \
                            zip(self.cont_MILP.z_map_vars.row,self.cont_MILP.z_map_vars.col,self.cont_MILP.z_map_vars.data)\
                            if np.logical_xor(sol[0,z_i],sense==-1)]
            active_vars = [i for i in range(self.cont_MILP.z_map_vars.shape[1]) if i not in inactive_vars]
            inactive_ineqs = [ineq for z_i,ineq,sense in \
                            zip(self.cont_MILP.z_map_constr_ineq.row,self.cont_MILP.z_map_constr_ineq.col,self.cont_MILP.z_map_constr_ineq.data)\
                            if np.logical_xor(sol[0,z_i],sense==-1) ]
            active_ineqs = [i for i in range(self.cont_MILP.z_map_constr_ineq.shape[1]) if i not in inactive_ineqs]
            inactive_eqs = [eq for z_i,eq,sense in \
                            zip(self.cont_MILP.z_map_constr_eq.row,self.cont_MILP.z_map_constr_eq.col,self.cont_MILP.z_map_constr_eq.data)\
                            if np.logical_xor(sol[0,z_i],sense==-1) ]
            active_eqs = [i for i in range(self.cont_MILP.z_map_constr_eq.shape[1]) if i not in inactive_eqs]

            # Zeroing a variable whose bounds exclude zero contradicts that bound, so the
            # region is empty whatever the remaining system does. This has to be tested
            # here rather than left to the LP: prevent_boundary_knockouts keeps such a
            # bound (e.g. ATPM >= 3.15) as a row with no z-mapping so that a knockout
            # contradicts it, but reassign_lb_ub_from_ineq later folds single-variable
            # rows back into variable bounds, and those vanish together with the column.
            if any(self.cont_MILP.lb[j] > 0.0 or self.cont_MILP.ub[j] < 0.0 for j in inactive_vars):
                valid[i] = False
                continue
            # Otherwise drop the columns outright. Absence is a stronger statement than an
            # interval of [0, 0], since it owes nothing to feasibility tolerances.
            lp = MILP_LP(A_ineq=self.cont_MILP.A_ineq[active_ineqs, :][:, active_vars],
                         b_ineq=[self.cont_MILP.b_ineq[i] for i in active_ineqs],
                         A_eq=self.cont_MILP.A_eq[active_eqs, :][:, active_vars],
                         b_eq=[self.cont_MILP.b_eq[i] for i in active_eqs],
                         lb=[self.cont_MILP.lb[i] for i in active_vars],
                         ub=[self.cont_MILP.ub[i] for i in active_vars],
                         solver=self.solver,
                         seed=self.seed)
            valid[i] = not np.isnan(lp.slim_solve())
        return valid

    def compute_optimal(self, **kwargs):
        """Compute the global optimum of the strain design MILP and iteratively find the next best solution
        
        Args:
            max_solutions (optional (int)): (Default: inf)
                The maximum number of MILP solutions that are generated for a strain design problem.
                
            time_limit (optional (int)): (Default: inf)
                The time limit in seconds for the MILP-solver.
                
            show_no_ki (optional (bool)): (Default: True)
                Indicate non-added addition candidates in a solution specifically with a value of 0
                
        Returns:
            (SDSolutions):
            Strain design solutions provided as an SDSolutions object
        """
        keys = {MAX_SOLUTIONS, T_LIMIT, 'show_no_ki'}
        # set keys passed in kwargs
        for key, value in dict(kwargs).items():
            if key in keys:
                setattr(self, key, value)
        # set all remaining keys to None
        for key in keys:
            if key not in dict(kwargs).keys():
                setattr(self, key, None)
        if self.max_solutions is None:
            self.max_solutions = np.inf
        if self.time_limit is None:
            self.time_limit = np.inf
        if self.show_no_ki is None:
            self.show_no_ki = True
        # first check if strain doesn't already fulfill the strain design setup
        if self.is_mcs_computation and self.verify_sd(sparse.csr_matrix((1, self.num_z)))[0]:
            logging.warning('The strain already meets the requirements defined in the strain design setup. ' \
                  'No interventions are needed.')
            return self.build_sd_solution([{}], OPTIMAL, BEST)
        # otherwise continue
        endtime = time.time() + self.time_limit
        status = OPTIMAL
        sols = sparse.csr_matrix((0, self.num_z))
        logging.info('Finding optimal strain designs ...')
        while sols.shape[0] < self.max_solutions and \
          status == OPTIMAL and \
          endtime-time.time() > 0:
            self.set_time_limit(endtime - time.time())
            self.resetTargetableZ()
            self.resetObjective()
            self.fixObjective(self.c_bu, np.inf)
            z, _, opt, status = self.solveZ()
            if 0 in z.shape or np.isnan(z[0, 0]):  # no (further) solution -> stop cleanly
                break
            output = self.sd2dict(z)
            if self.is_mcs_computation:
                if status in [OPTIMAL, TIME_LIMIT_W_SOL] and all(self.verify_sd(z)):
                    logging.info('Strain design with cost ' + str(round((z * self.cost)[0], 6)) + ': ' + str(output))
                    self.add_exclusion_constraints(z)
                    sols = sparse.vstack((sols, z))
                elif status in [OPTIMAL, TIME_LIMIT_W_SOL]:
                    logging.info('Invalid (minimal) solution found: ' + str(output))
                    self.add_exclusion_constraints(z)
                if status != OPTIMAL:
                    break
            else:
                # Verify solution and explore subspace to get minimal intervention sets
                logging.info('Found solution with objective value ' + str(-opt))
                logging.info('Minimizing number of interventions in subspace with ' + str(sum(z.toarray()[0])) + ' possible targets.')
                self.fixObjective(self.c_bu, opt)
                self.setMinIntvCostObjective()
                self.setTargetableZ(z)
                while sols.shape[0] < self.max_solutions and \
                        status == OPTIMAL and \
                        endtime-time.time() > 0:
                    self.set_time_limit(endtime - time.time())
                    z1, _, _, status1 = self.solveZ()
                    output = self.sd2dict(z1)
                    if status1 in [OPTIMAL, TIME_LIMIT_W_SOL] and all(self.verify_sd(z1)):
                        logging.info('Strain design with cost ' + str(round((z1 * self.cost)[0], 6)) + ': ' + str(output))
                        self.add_exclusion_constraints(z1)
                        sols = sparse.vstack((sols, z1))
                    elif status1 in [OPTIMAL, TIME_LIMIT_W_SOL]:
                        logging.warning('Invalid minimal solution found: ' + str(output))
                        self.add_exclusion_constraints_ineq(z1)
                    else:  # return to outside loop
                        break
        if status == INFEASIBLE and sols.shape[0] > 0:  # all solutions found
            status = OPTIMAL
        if status == TIME_LIMIT and sols.shape[0] > 0:  # some solutions found, timelimit reached
            status = TIME_LIMIT_W_SOL
        if endtime - time.time() > 0 and sols.shape[0] > 0:
            logging.info('Finished solving strain design MILP. ')
            if 'strainDesignMILP' in self.__module__:
                logging.info(str(sols.shape[0]) + ' solutions to MILP found.')
        elif endtime - time.time() > 0:
            logging.info('Finished solving strain design MILP.')
            if 'strainDesignMILP' in self.__module__:
                logging.info(' No solutions exist.')
        else:
            logging.info('Time limit reached.')
        # Translate solutions into dict
        sd_dict = []
        for sol in sols:
            sd_dict += [self.sd2dict(sol, self.show_no_ki)]
        return self.build_sd_solution(sd_dict, status, BEST)

    # Find iteratively intervention sets of arbitrary size or quality
    # output format: list of 'dict' (default) or 'sparse'
    def compute(self, **kwargs):
        """Compute arbitrary solutions of the strain design MILP and iteratively find further solutions
        
        Args:
            max_solutions (optional (int)): (Default: inf)
                The maximum number of MILP solutions that are generated for a strain design problem.
                
            time_limit (optional (int)): (Default: inf)
                The time limit in seconds for the MILP-solver.
                
            show_no_ki (optional (bool)): (Default: True)
                Indicate non-added addition candidates in a solution specifically with a value of 0
                
        Returns:
            (SDSolutions):
            Strain design solutions provided as an SDSolutions object
        """
        keys = {MAX_SOLUTIONS, T_LIMIT, 'show_no_ki'}
        # set keys passed in kwargs
        for key, value in kwargs.items():
            if key in keys:
                setattr(self, key, value)
        # set all remaining keys to None
        for key in keys:
            if key not in kwargs.keys():
                setattr(self, key, None)
        if self.max_solutions is None:
            self.max_solutions = np.inf
        if self.time_limit is None:
            self.time_limit = np.inf
        if self.show_no_ki is None:
            self.show_no_ki = True
        # first check if strain doesn't already fulfill the strain design setup
        if self.verify_sd(sparse.csr_matrix((1, self.num_z)))[0]:
            logging.warning('The strain already meets the requirements defined in the strain design setup. ' \
                  'No interventions are needed.')
            return self.build_sd_solution([{}], OPTIMAL, ANY)
        # otherwise continue
        endtime = time.time() + self.time_limit
        status = OPTIMAL
        sols = sparse.csr_matrix((0, self.num_z))
        logging.info('Finding (also non-optimal) strain designs ...')
        while sols.shape[0] < self.max_solutions and \
          status == OPTIMAL and \
          endtime-time.time() > 0:
            logging.info('Searching in full search space.')
            self.set_time_limit(endtime - time.time())
            self.resetTargetableZ()
            self.clear_objective()
            self.fixObjective(self.c_bu, np.inf)  # keep objective open
            z, x, _, status = self.solveZ()
            if status not in [OPTIMAL, TIME_LIMIT_W_SOL]:
                break
            if not all(self.verify_sd(z)):
                self.set_time_limit(endtime - time.time())
                self.resetObjective()
                self.setTargetableZ(z)
                self.fixObjective(self.c_bu, np.sum([c * x for c, x in zip(self.c_bu, x)]))
                z1, _, _, status1 = self.solveZ()
                if status1 == OPTIMAL and not self.verify_sd(z1):
                    self.add_exclusion_constraints(z1)
                    output = self.sd2dict(z1)
                    logging.warning('Invalid minimal solution found: ' + str(output))
                    continue
                if status1 != OPTIMAL and not self.verify_sd(z1):
                    self.add_exclusion_constraints_ineq(z1)
                    output = self.sd2dict(z1)
                    logging.warning('Invalid minimal solution found: ' + str(output))
                    continue
                else:
                    output = self.sd2dict(z)
                    logging.warning('Warning: Solver first found the infeasible solution: ' + str(output))
                    output = self.sd2dict(z1)
                    logging.warning('But a subset of this solution seems to be valid: ' + str(output))
            # Verify solution and explore subspace to get strain designs
            cx = np.sum([c * x for c, x in zip(self.c_bu, x)])
            if not self.is_mcs_computation:
                logging.info('Found preliminary solution.')
            logging.info('Minimizing number of interventions in subspace with ' + str(sum(z.toarray()[0])) + ' possible targets.')
            self.setMinIntvCostObjective()
            self.setTargetableZ(z)
            self.fixObjective(self.c_bu, cx)
            while sols.shape[0] < self.max_solutions and \
                    status == OPTIMAL and \
                    endtime-time.time() > 0:
                self.set_time_limit(endtime - time.time())
                z1, _, _, status1 = self.solveZ()
                output = self.sd2dict(z1)
                if status1 in [OPTIMAL, TIME_LIMIT_W_SOL] and all(self.verify_sd(z1)):
                    logging.info('Strain design with cost ' + str(round((z1 * self.cost)[0], 6)) + ': ' + str(output))
                    self.add_exclusion_constraints(z1)
                    sols = sparse.vstack((sols, z1))
                elif status1 in [OPTIMAL, TIME_LIMIT_W_SOL]:
                    logging.warning('Invalid minimal solution found: ' + str(output))
                    self.add_exclusion_constraints_ineq(z1)
                else:  # return to outside loop
                    break
        if status == INFEASIBLE and sols.shape[0] > 0:  # all solutions found
            status = OPTIMAL
        if status == TIME_LIMIT and sols.shape[0] > 0:  # some solutions found, timelimit reached
            status = TIME_LIMIT_W_SOL
        if endtime - time.time() > 0 and sols.shape[0] > 0:
            logging.info('Finished solving strain design MILP. ')
            if 'strainDesignMILP' in self.__module__:
                logging.info(str(sols.shape[0]) + ' solutions to MILP found.')
        elif endtime - time.time() > 0:
            logging.info('Finished solving strain design MILP.')
            if 'strainDesignMILP' in self.__module__:
                logging.info(' No solutions exist.')
        else:
            logging.info('Time limit reached.')
        # Translate solutions into dict if not stated otherwise
        sd_dict = []
        for sol in sols:
            sd_dict += [self.sd2dict(sol, self.show_no_ki)]
        return self.build_sd_solution(sd_dict, status, ANY)

    # Enumerate iteratively optimal strain designs using the populate function
    # output format: list of 'dict' (default) or 'sparse'
    def enumerate(self, **kwargs):
        """Find all globally optimal solutions to the strain design MILP and iteratively construct pools for the suboptimal values
            
        Args:
            max_solutions (optional (int)): (Default: inf)
                The maximum number of MILP solutions that are generated for a strain design problem.
                
            time_limit (optional (int)): (Default: inf)
                The time limit in seconds for the MILP-solver.
                
            show_no_ki (optional (bool)): (Default: True)
                Indicate non-added addition candidates in a solution specifically with a value of 0
                
        Returns:
            (SDSolutions):
            Strain design solutions provided as an SDSolutions object
        """
        keys = {MAX_SOLUTIONS, T_LIMIT, 'show_no_ki'}
        # set keys passed in kwargs
        for key, value in dict(kwargs).items():
            if key in keys:
                setattr(self, key, value)
        # set all remaining keys to None
        for key in keys:
            if key not in dict(kwargs).keys():
                setattr(self, key, None)
        if self.max_solutions is None:
            self.max_solutions = np.inf
        if self.time_limit is None:
            self.time_limit = np.inf
        if self.show_no_ki is None:
            self.show_no_ki = True
        # first check if strain doesn't already fulfill the strain design setup
        wt_is_design = self.is_mcs_computation and self.verify_sd(sparse.csr_matrix((1, self.num_z)))[0]
        if wt_is_design:
            logging.warning('The strain already meets the requirements defined in the strain design setup. ' \
                  'No interventions are needed.')
            # Free interventions can still yield designs that cost no more than doing nothing, so
            # keep enumerating and let the exclusion constraints carry the empty design forward.
            if not self._free_z:
                return self.build_sd_solution([{}], OPTIMAL, POPULATE)
        # otherwise continue
        if self.solver == 'scip':
            logging.warning("SCIP does not natively support solution pool generation. "+ \
                "An high-level implementation of populate is used. " + \
                "Consider using compute_optimal instead of enumerate, as " + \
                "it returns the same results but faster.")
        if self.solver == 'glpk':
            logging.warning("GLPK does not natively support solution pool generation. "+ \
                "An instable high-level implementation of populate is used. "
                "Consider using compute_optimal instead of enumerate, as " + \
                "it returns the same results but faster." )
        endtime = time.time() + self.time_limit
        status = OPTIMAL
        sols = sparse.csr_matrix((0, self.num_z))
        if wt_is_design:
            empty = sparse.csr_matrix((1, self.num_z))
            if not (self._rewarding_z and self.is_dominated(empty)):
                sols = sparse.vstack((sols, empty))
            self.add_exclusion_constraints(empty)
        logging.info('Enumerating strain designs ...')
        while sols.shape[0] < self.max_solutions and \
          status == OPTIMAL and \
          endtime-time.time() > 0:
            self.set_time_limit(endtime - time.time())
            if not self.is_mcs_computation:
                self.resetTargetableZ()
                self.resetObjective()
                self.fixObjective(self.c_bu, np.inf)
                z, _, opt, status = self.solveZ()
                if status not in [OPTIMAL, TIME_LIMIT_W_SOL]:
                    break
                logging.info('Enumerating all solutions with the objective value: ' + str(-opt))
                self.fixObjective(self.c_bu, opt)
                self.setMinIntvCostObjective()
            z, status = self.populateZ(self.max_solutions - sols.shape[0])
            if status in [OPTIMAL, TIME_LIMIT_W_SOL]:
                for i in range(z.shape[0]):
                    output = [self.sd2dict(z[i])]
                    if all(self.verify_sd(z[i])):
                        if self._rewarding_z and self.is_dominated(z[i]):
                            # a cheaper design contains this one; cut only the exact pattern so
                            # that the design dominating it stays reachable
                            logging.info('Dominated by a cheaper superset, skipping: ' + str(output))
                            self.add_exclusion_constraints_ineq(z[i])
                            continue
                        logging.info('Strain designs with cost ' + str(round((z[i] * self.cost)[0], 6)) + ': ' + str(output))
                        self.add_exclusion_constraints(z[i])
                        sols = sparse.vstack((sols, z[i]))
                    else:
                        logging.warning('Invalid (minimal) solution found: ' + str(output))
                        self.add_exclusion_constraints(z[i])
            if (status != OPTIMAL):  # or (z[i]*self.cost == self.max_cost):
                break
        if status == INFEASIBLE and sols.shape[0] > 0:  # all solutions found or solution limit reached
            status = OPTIMAL
        if status == TIME_LIMIT and sols.shape[0] > 0:  # some solutions found, timelimit reached
            status = TIME_LIMIT_W_SOL
        if endtime - time.time() > 0 and sols.shape[0] > 0:
            logging.info('Finished solving strain design MILP. ')
            if 'strainDesignMILP' in self.__module__:
                logging.info(str(sols.shape[0]) + ' solutions to MILP found.')
        elif endtime - time.time() > 0:
            logging.info('Finished solving strain design MILP.')
            if 'strainDesignMILP' in self.__module__:
                logging.info(' No solutions exist.')
        else:
            logging.info('Time limit reached.')
        # Translate solutions into dict if not stated otherwise
        sd_dict = []
        for sol in sols:
            sd_dict += [self.sd2dict(sol, self.show_no_ki)]
        sd_solution = self.build_sd_solution(sd_dict, status, POPULATE)
        return sd_solution

    def build_sd_solution(self, sd_dict, status, solution_approach):
        """Build the strain design solution object"""
        sd_setup = {}
        sd_setup[MODEL_ID] = self.model.id
        sd_setup[MAX_SOLUTIONS] = self.max_solutions
        sd_setup[MAX_COST] = self.max_cost
        sd_setup[TIME_LIMIT] = self.time_limit
        sd_setup[SOLVER] = self.solver
        sd_setup[SOLUTION_APPROACH] = solution_approach
        sd_setup[KOCOST] = {k: float(v) for k,v in \
            zip(self.model.reactions.list_attr('id'),self.ko_cost) if not np.isnan(v)}
        sd_setup[KICOST] = {k: float(v) for k,v in \
            zip(self.model.reactions.list_attr('id'),self.ki_cost) if not np.isnan(v)}
        sd_setup[MODULES] = self.sd_modules
        return SDSolutions(self.model, sd_dict, status, sd_setup)
