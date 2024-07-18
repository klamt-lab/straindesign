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
                         seed=self.seed)

    def add_exclusion_constraints(self, z):
        """Exclude binary solution in z and all supersets from MILP"""
        for i in range(z.shape[0]):
            # introduce constraint to make MILP infeasible. Some solvers cannot handle empty rows
            if z[i].nnz == 0:
                A_ineq = sparse.csr_matrix([1.0] * z[i].shape[1])
                A_ineq.resize((1, self.A_ineq.shape[1]))
                b_ineq = -1
                self.add_ineq_constraints(A_ineq, [b_ineq])
            # otherwise, introduce integer cut constraint
            elif z[i].nnz == 1:
                interv_idx = int(z[i].indices[0])
                self.z_non_targetable[interv_idx] = True
                self.set_ub([[interv_idx, 0.0]])
            # otherwise, introduce integer cut constraint
            else:
                A_ineq = z[i].copy()
                A_ineq.resize((1, self.A_ineq.shape[1]))
                b_ineq = np.sum(z[i]) - 1
                self.add_ineq_constraints(A_ineq, [b_ineq])

    def add_exclusion_constraints_ineq(self, z):
        """Exclude binary solution in z (but not its supersets) from MILP"""
        for j in range(z.shape[0]):
            A_ineq = [1.0 if z[j, i] else -1.0 for i in self.idx_z]
            A_ineq.resize((1, self.A_ineq.shape[1]))
            b_ineq = np.sum(z[j]) - 1
            self.add_ineq_constraints(A_ineq, [b_ineq])

    def sd2dict(self, sol, *args) -> Dict:
        """Translate binary solution vector to dictionary for human-readable output"""
        output = {}
        reacID = self.model.reactions.list_attr("id")
        for i in self.idx_z:
            if sol[0, i] != 0 and not np.isnan(sol[0, i]):
                if self.z_inverted[i]:
                    output[reacID[i]] = sol[0, i]
                else:
                    output[reacID[i]] = -sol[0, i]
            elif args and args[0] and (sol[0, i] == 0) and self.z_inverted[i]:
                output[reacID[i]] = 0.0
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
        valid = [False] * sols.shape[0]
        for i, sol in zip(range(sols.shape[0]), sols):
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
            if np.isnan(z[0, 0]):
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
                        self.add_exclusion_constraints(z)
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
                    self.add_exclusion_constraints_ineq(z)
                    output = self.sd2dict(z)
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
                    self.add_exclusion_constraints(z)
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
        if self.is_mcs_computation and self.verify_sd(sparse.csr_matrix((1, self.num_z)))[0]:
            logging.warning('The strain already meets the requirements defined in the strain design setup. ' \
                  'No interventions are needed.')
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
