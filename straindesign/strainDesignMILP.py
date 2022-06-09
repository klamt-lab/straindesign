import numpy as np
from scipy import sparse
import time
from typing import Dict, List, Tuple
from straindesign import SDProblem, SDSolutions, MILP_LP, SDModule
from straindesign.names import *
from warnings import warn
import logging


class SDMILP(MILP_LP):

    def __init__(self, sd_problem: SDProblem):
        # Build MILP object from constructed problem
        super().__init__(c=sd_problem.c,
                         A_ineq=sd_problem.A_ineq,
                         b_ineq=sd_problem.b_ineq,
                         A_eq=sd_problem.A_eq,
                         b_eq=sd_problem.b_eq,
                         lb=sd_problem.lb,
                         ub=sd_problem.ub,
                         vtype=sd_problem.vtype,
                         indic_constr=sd_problem.indic_constr,
                         M=sd_problem.M,
                         solver=sd_problem.solver)
        # Copy some parameters
        self.cont_MILP = sd_problem.cont_MILP
        self.model = sd_problem.model
        self.sd_modules = sd_problem.sd_modules
        self.is_mcs_computation = sd_problem.is_mcs_computation
        self.max_cost = sd_problem.max_cost
        self.cost = sd_problem.cost
        self.c_bu = sd_problem.c
        self.idx_z = sd_problem.idx_z
        self.z_inverted = sd_problem.z_inverted
        self.z_non_targetable = sd_problem.z_non_targetable
        self.num_z = sd_problem.num_z
        self.ko_cost = sd_problem.ko_cost
        self.ki_cost = sd_problem.ki_cost
        self.idx_row_maxcost = sd_problem.idx_row_maxcost
        self.idx_row_mincost = sd_problem.idx_row_mincost
        self.idx_row_obj = sd_problem.idx_row_obj

    def add_exclusion_constraints(self, z):
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
        for j in range(z.shape[0]):
            A_ineq = [1.0 if z[j, i] else -1.0 for i in self.idx_z]
            A_ineq.resize((1, self.A_ineq.shape[1]))
            b_ineq = np.sum(z[j]) - 1
            self.add_ineq_constraints(A_ineq, [b_ineq])

    def sd2dict(self, sol, *args) -> Dict:
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
        x, opt, status = self.solve()
        z = sparse.csr_matrix([round(x[i], 5) for i in self.idx_z])
        return z, x, opt, status

    def populateZ(self, n) -> Tuple[List, int]:
        x, _, status = self.populate(n)
        if status in [OPTIMAL, TIME_LIMIT_W_SOL]:
            z = sparse.csr_matrix([
                [round(x[j][i], 5) for i in self.idx_z] for j in range(len(x))
            ])
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
        self.set_ineq_constraint(self.idx_row_obj, c, cx)

    def resetObjective(self):
        self.set_objective_idx([[i, v] for i, v in enumerate(self.c_bu)])

    def setMinIntvCostObjective(self):
        self.clear_objective()
        self.set_objective_idx([[i, self.cost[i]]
                                for i in self.idx_z
                                if i not in self.z_non_targetable])

    def resetTargetableZ(self):
        self.set_ub(
            [[i, 1.0] for i in self.idx_z if not self.z_non_targetable[i]])

    def setTargetableZ(self, sol):
        self.set_ub([[i, 0.0] for i in self.idx_z if not sol[0, i]])

    def verify_sd(self, sols) -> List:
        valid = [False] * sols.shape[0]
        for i, sol in zip(range(sols.shape[0]), sols):
            inactive_vars = [var for z_i,var,sense in \
                            zip(self.cont_MILP.z_map_vars.row,self.cont_MILP.z_map_vars.col,self.cont_MILP.z_map_vars.data)\
                            if np.logical_xor(sol[0,z_i],sense==-1)]
            active_vars = [
                i for i in range(self.cont_MILP.z_map_vars.shape[1])
                if i not in inactive_vars
            ]
            inactive_ineqs = [ineq for z_i,ineq,sense in \
                            zip(self.cont_MILP.z_map_constr_ineq.row,self.cont_MILP.z_map_constr_ineq.col,self.cont_MILP.z_map_constr_ineq.data)\
                            if np.logical_xor(sol[0,z_i],sense==-1) ]
            active_ineqs = [
                i for i in range(self.cont_MILP.z_map_constr_ineq.shape[1])
                if i not in inactive_ineqs
            ]
            inactive_eqs = [eq for z_i,eq,sense in \
                            zip(self.cont_MILP.z_map_constr_eq.row,self.cont_MILP.z_map_constr_eq.col,self.cont_MILP.z_map_constr_eq.data)\
                            if np.logical_xor(sol[0,z_i],sense==-1) ]
            active_eqs = [
                i for i in range(self.cont_MILP.z_map_constr_eq.shape[1])
                if i not in inactive_eqs
            ]

            lp = MILP_LP(
                A_ineq=self.cont_MILP.A_ineq[active_ineqs, :][:, active_vars],
                b_ineq=[self.cont_MILP.b_ineq[i] for i in active_ineqs],
                A_eq=self.cont_MILP.A_eq[active_eqs, :][:, active_vars],
                b_eq=[self.cont_MILP.b_eq[i] for i in active_eqs],
                lb=[self.cont_MILP.lb[i] for i in active_vars],
                ub=[self.cont_MILP.ub[i] for i in active_vars],
                solver=self.solver)
            valid[i] = not np.isnan(lp.slim_solve())
        return valid

    # Find iteratively smallest solutions
    def compute_optimal(self, **kwargs):
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
        # first check if strain doesn't already fulfill the strain design setup
        if self.is_mcs_computation and self.verify_sd(
                sparse.csr_matrix((1, self.num_z)))[0]:
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
                if status in [OPTIMAL, TIME_LIMIT_W_SOL] and all(
                        self.verify_sd(z)):
                    logging.info('Strain design with cost ' +
                                 str(round((z * self.cost)[0], 6)) + ': ' +
                                 str(output))
                    self.add_exclusion_constraints(z)
                    sols = sparse.vstack((sols, z))
                elif status in [OPTIMAL, TIME_LIMIT_W_SOL]:
                    logging.info('Invalid (minimal) solution found: ' +
                                 str(output))
                    self.add_exclusion_constraints(z)
                if status != OPTIMAL:
                    break
            else:
                # Verify solution and explore subspace to get minimal intervention sets
                logging.info('Found solution with objective value ' + str(-opt))
                logging.info(
                    'Minimizing number of interventions in subspace with ' +
                    str(sum(z.toarray()[0])) + ' possible targets.')
                self.fixObjective(self.c_bu, opt)
                self.setMinIntvCostObjective()
                self.setTargetableZ(z)
                while sols.shape[0] < self.max_solutions and \
                        status == OPTIMAL and \
                        endtime-time.time() > 0:
                    self.set_time_limit(endtime - time.time())
                    z1, _, _, status1 = self.solveZ()
                    output = self.sd2dict(z1)
                    if status1 in [OPTIMAL, TIME_LIMIT_W_SOL] and all(
                            self.verify_sd(z1)):
                        logging.info('Strain design with cost ' +
                                     str(round((z1 * self.cost)[0], 6)) + ': ' +
                                     str(output))
                        self.add_exclusion_constraints(z1)
                        sols = sparse.vstack((sols, z1))
                    elif status1 in [OPTIMAL, TIME_LIMIT_W_SOL]:
                        logging.warning('Invalid minimal solution found: ' +
                                        str(output))
                        self.add_exclusion_constraints(z)
                    else:  # return to outside loop
                        break
        if status == INFEASIBLE and sols.shape[0] > 0:  # all solutions found
            status = OPTIMAL
        if status == TIME_LIMIT and sols.shape[
                0] > 0:  # some solutions found, timelimit reached
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
        m = sd_dict = []
        for sol in sols:
            sd_dict += [self.sd2dict(sol, self.show_no_ki)]
        return self.build_sd_solution(sd_dict, status, BEST)

    # Find iteratively intervention sets of arbitrary size or quality
    # output format: list of 'dict' (default) or 'sparse'
    def compute(self, **kwargs):
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
                self.fixObjective(self.c_bu,
                                  np.sum([c * x for c, x in zip(self.c_bu, x)]))
                z1, _, _, status1 = self.solveZ()
                if status1 == OPTIMAL and not self.verify_sd(z1):
                    self.add_exclusion_constraints(z1)
                    output = self.sd2dict(z1)
                    logging.warning('Invalid minimal solution found: ' +
                                    str(output))
                    continue
                if status1 != OPTIMAL and not self.verify_sd(z1):
                    self.add_exclusion_constraints_ineq(z)
                    output = self.sd2dict(z)
                    logging.warning('Invalid minimal solution found: ' +
                                    str(output))
                    continue
                else:
                    output = self.sd2dict(z)
                    logging.warning(
                        'Warning: Solver first found the infeasible solution: '
                        + str(output))
                    output = self.sd2dict(z1)
                    logging.warning(
                        'But a subset of this solution seems to be valid: ' +
                        str(output))
            # Verify solution and explore subspace to get strain designs
            cx = np.sum([c * x for c, x in zip(self.c_bu, x)])
            if not self.is_mcs_computation:
                logging.info('Found preliminary solution.')
            logging.info(
                'Minimizing number of interventions in subspace with ' +
                str(sum(z.toarray()[0])) + ' possible targets.')
            self.setMinIntvCostObjective()
            self.setTargetableZ(z)
            self.fixObjective(self.c_bu, cx)
            while sols.shape[0] < self.max_solutions and \
                    status == OPTIMAL and \
                    endtime-time.time() > 0:
                self.set_time_limit(endtime - time.time())
                z1, _, _, status1 = self.solveZ()
                output = self.sd2dict(z1)
                if status1 in [OPTIMAL, TIME_LIMIT_W_SOL] and all(
                        self.verify_sd(z1)):
                    logging.info('Strain design with cost ' +
                                 str(round((z1 * self.cost)[0], 6)) + ': ' +
                                 str(output))
                    self.add_exclusion_constraints(z1)
                    sols = sparse.vstack((sols, z1))
                elif status1 in [OPTIMAL, TIME_LIMIT_W_SOL]:
                    logging.warning('Invalid minimal solution found: ' +
                                    str(output))
                    self.add_exclusion_constraints(z)
                else:  # return to outside loop
                    break
        if status == INFEASIBLE and sols.shape[0] > 0:  # all solutions found
            status = OPTIMAL
        if status == TIME_LIMIT and sols.shape[
                0] > 0:  # some solutions found, timelimit reached
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
        # first check if strain doesn't already fulfill the strain design setup
        if self.is_mcs_computation and self.verify_sd(
                sparse.csr_matrix((1, self.num_z)))[0]:
            logging.warning('The strain already meets the requirements defined in the strain design setup. ' \
                  'No interventions are needed.')
            return self.build_sd_solution([{}], OPTIMAL, POPULATE)
        # otherwise continue
        if self.solver == 'scip':
            warn("SCIP does not natively support solution pool generation. "+ \
                "An high-level implementation of populate is used. " + \
                "Consider using compute_optimal instead of enumerate, as " + \
                "it returns the same results but faster.")
        if self.solver == 'glpk':
            warn("GLPK does not natively support solution pool generation. "+ \
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
                logging.info(
                    'Enumerating all solutions with the objective value: ' +
                    str(-opt))
                self.fixObjective(self.c_bu, opt)
                self.setMinIntvCostObjective()
            z, status = self.populateZ(self.max_solutions - sols.shape[0])
            if status in [OPTIMAL, TIME_LIMIT_W_SOL]:
                for i in range(z.shape[0]):
                    output = [self.sd2dict(z[i])]
                    if all(self.verify_sd(z[i])):
                        logging.info('Strain designs with cost ' +
                                     str(round((z[i] * self.cost)[0], 6)) +
                                     ': ' + str(output))
                        self.add_exclusion_constraints(z[i])
                        sols = sparse.vstack((sols, z[i]))
                    else:
                        logging.warning('Invalid (minimal) solution found: ' +
                                        str(output))
                        self.add_exclusion_constraints(z[i])
            if (status != OPTIMAL):  # or (z[i]*self.cost == self.max_cost):
                break
        if status == INFEASIBLE and sols.shape[
                0] > 0:  # all solutions found or solution limit reached
            status = OPTIMAL
        if status == TIME_LIMIT and sols.shape[
                0] > 0:  # some solutions found, timelimit reached
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
        sd_setup = {}
        sd_setup[MODEL_ID] = self.model.id
        sd_setup[MAX_SOLUTIONS] = self.max_solutions
        sd_setup[MAX_COST] = self.max_cost
        sd_setup[TIME_LIMIT] = self.time_limit
        sd_setup[SOLVER] = self.solver
        sd_setup[SOLUTION_APPROACH] = solution_approach
        sd_setup[KOCOST] = {k:float(v) for k,v in \
            zip(self.model.reactions.list_attr('id'),self.ko_cost) if not np.isnan(v)}
        sd_setup[KICOST] = {k:float(v) for k,v in \
            zip(self.model.reactions.list_attr('id'),self.ki_cost) if not np.isnan(v)}
        sd_setup[MODULES] = self.sd_modules
        return SDSolutions(self.model, sd_dict, status, sd_setup)
