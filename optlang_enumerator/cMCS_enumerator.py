import numpy
import scipy
import optlang.glpk_interface
from optlang.symbolics import add
from optlang.exceptions import IndicatorConstraintsNotSupported
from swiglpk import glp_write_lp
try:
    import optlang.cplex_interface
    import cplex
    from cplex.exceptions import CplexSolverError
    from cplex._internal._subinterfaces import SolutionStatus # can be also accessed by a CPLEX object under .solution.status
except:
    optlang.cplex_interface = None # make sure this symbol is defined for type() comparisons
try:
    import optlang.coinor_cbc_interface
except:
    optlang.coinor_cbc_interface = None # make sure this symbol is defined for type() comparisons
from typing import List, Tuple, Union, FrozenSet
import time
import optlang_enumerator.mcs_computation as mcs_computation

class ConstrainedMinimalCutSetsEnumerator:
    def __init__(self, optlang_interface, st, reversible, targets, kn=None, cuts=None,
        desired=None, knock_in=None, bigM=0, threshold=1, split_reversible_v=True,
        irrev_geq=False, ref_set= None): # reduce_constraints=True, combined_z=True
        # the matrices in st, targets and desired should be numpy.array or scipy.sparse (csr, csc, lil) format
        # targets is a list of (T,t) pairs that represent T <= t
        # implements only combined_z which implies reduce_constraints=True
        # knock_in not yet implemented
        self.ref_set = ref_set # optional set of reference MCS for debugging
        self._optlang_interface = optlang_interface
        self.model = optlang_interface.Model()
        self.model.configuration.presolve = True # presolve on
        # without presolve CPLEX sometimes gives false results when using indicators ?!?
        self.model.configuration.lp_method = 'auto'
        self.Constraint = optlang_interface.Constraint
        if bigM <= 0 and self.Constraint._INDICATOR_CONSTRAINT_SUPPORT is False:
            raise IndicatorConstraintsNotSupported("This solver does not support indicators. Please choose a differen solver or use a big M formulation.")
        self.optlang_variable_class = optlang_interface.Variable
        irr = [not r for r in reversible]
        self.num_reac = len(reversible)
        if cuts is None:
            cuts = numpy.full(self.num_reac, True, dtype=bool)
            irrepressible = []
        else:
            irrepressible = numpy.where(cuts == False)[0]
            #iv_cost(irrepressible)= 0;
        if desired is None:
            desired = []
        num_targets = len(targets)
        use_kn_in_dual = kn is not None
        if use_kn_in_dual:
            if irrev_geq:
                raise ValueError('Use of irrev_geq together with kn parameter is not possible.')
            if type(kn) is numpy.ndarray:
                kn = scipy.sparse.csc_matrix(kn) # otherwise stacking for dual does not work

        if split_reversible_v:
            split_v_idx = [i for i, x in enumerate(reversible) if x]
            dual_rev_neg_idx = [i for i in range(self.num_reac, self.num_reac + len(split_v_idx))]
            dual_rev_neg_idx_map = [None] * self.num_reac
            for i in range(len(split_v_idx)):
                dual_rev_neg_idx_map[split_v_idx[i]]= dual_rev_neg_idx[i];
        else:
            split_v_idx = []

        self.zero_objective= optlang_interface.Objective(0, direction='min', name='zero_objective')
        self.model.objective= self.zero_objective
        self.z_vars = [self.optlang_variable_class("Z"+str(i), type="binary", problem=self.model.problem) for i in range(self.num_reac)]
        self.model.add(self.z_vars)
        self.model.update() # cannot change bound below without this
        for i in irrepressible:
            self.z_vars[i].ub = 0 # only if it is not a knock-in (not yet supported)
        self.minimize_sum_over_z= optlang_interface.Objective(add(self.z_vars), direction='min', name='minimize_sum_over_z')
        z_local = [None] * num_targets
        if num_targets == 1:
            z_local[0] = self.z_vars # global and local Z are the same if there is only one target
        else:
            # with reduce constraints is should not be necessary to have local Z, they can be the same as the global Z
            for k in range(num_targets):
                z_local[k] = self.z_vars
            # for k in range(num_targets):
            #     z_local[k] = [self.optlang_variable_class("Z"+str(k)+"_"+str(i), type="binary", problem=self.model.problem) for i in range(self.num_reac)]
            #     self.model.add(z_local[k])
            # for i in range(self.num_reac):
            #     if cuts[i]: # && ~knock_in(i) % knock-ins only use global Z, do not need local ones
            #         self.model.add(self.Constraint(
            #             (1/num_targets - 1e-9)*add([z_local[k][i] for k in range(num_targets)]) - self.z_vars[i], ub=0,
            #             name= "ZL"+str(i)))

        dual_vars = [None] * num_targets
        # num_dual_cols = [0] * num_targets
        for k in range(num_targets):
            # !! unboundedness is only properly represented by None with optlang; using inifinity may cause trouble !!
            dual_lb = [None] * self.num_reac # optlang interprets None as Inf
            #dual_lb = numpy.full(self.num_reac, numpy.NINF)
            dual_ub = [None] * self.num_reac
            #dual_ub = numpy.full(self.num_reac, numpy.inf) # can lead to GLPK crash when trying to otimize an infeasible MILP
            # GLPK treats infinity different than declaring unboundedness explicitly by glp_set_col_bnds ?!?
            # could use numpy arrays and convert them to lists where None replaces inf before calling optlang
            if split_reversible_v:
                    for i in range(self.num_reac):
                        if irrev_geq or reversible[i]:
                            dual_lb[i] = 0
            else:
                if irrev_geq:
                    for i in range(self.num_reac):
                        if irr[i]:
                            dual_lb[i] = 0
                for i in irrepressible:
                    if reversible[i]:
                        dual_lb[i] = 0
            for i in irrepressible:
                dual_ub[i] = 0
            if split_reversible_v:
                dual_vars[k] = [self.optlang_variable_class("DP"+str(k)+"_"+str(i), lb=dual_lb[i], ub=dual_ub[i]) for i in range(self.num_reac)] + \
                    [self.optlang_variable_class("DN"+str(k)+"_"+str(i), ub=0) for i in split_v_idx]
                for i in irrepressible:
                    if reversible[i]: # fixes DN of irrepressible reversible reactions to 0
                         dual_vars[k][dual_rev_neg_idx_map[i]].lb = 0
            else:
                dual_vars[k] = [self.optlang_variable_class("DR"+str(k)+"_"+str(i), lb=dual_lb[i], ub=dual_ub[i]) for i in range(self.num_reac)]
            first_w= len(dual_vars[k]) # + 1;
            if use_kn_in_dual is False:
                dual = scipy.sparse.eye(self.num_reac, format='csr')
                if split_reversible_v:
                    dual = scipy.sparse.hstack((dual, dual[:, split_v_idx]), format='csr')
                dual = scipy.sparse.hstack((dual, st.transpose(), targets[k][0].transpose()), format='csr')
                dual_vars[k] += [self.optlang_variable_class("DS"+str(k)+"_"+str(i)) for i in range(st.shape[0])]
                first_w += st.shape[0]
            else:
                if split_reversible_v:
                    dual = scipy.sparse.hstack((kn.transpose(), kn[reversible, :].transpose(), kn.transpose() @ targets[k][0].transpose()), format='csr')
                else:
                    dual = scipy.sparse.hstack((kn.transpose(), kn.transpose() @ targets[k][0].transpose()), format='csr')
                #       switch split_level
                #         case 1 % split dual vars associated with reversible reactions
                #           dual= [kn', kn(~irr, :)', kn'*T{k}'];
                #         case 2 % split all dual vars which are associated with reactions into DN <= 0, DP >= 0
                #           dual= [kn', kn', kn'*T{k}'];
                #         otherwise % no splitting
                #           dual= [kn', kn'*T{k}'];
            dual_vars[k] += [self.optlang_variable_class("DT"+str(k)+"_"+str(i), lb=0) for i in range(targets[k][0].shape[0])]
            self.model.add(dual_vars[k])
            # num_dual_cols[k]= dual.shape[1]
            constr= [None] * (dual.shape[0]+1)
            # print(dual_vars[k][first_w:])
            expr = mcs_computation.matrix_row_expressions(dual, dual_vars[k])
            # print(expr)
            for i in range(dual.shape[0]):
                if irrev_geq and irr[i]:
                    ub = None
                else:
                    ub = 0
                #expr = add([cf * var for cf, var in zip(dual[i, :], dual_vars[k]) if cf != 0])
                #expr = add([dual[i, k] *  dual_vars[k] for k in dual[i, :].nonzero()[1]])
                constr[i] = self.Constraint(expr[i], lb=0, ub=ub, name="D"+str(k)+"_"+str(i), sloppy=True)
            expr = add([cf * var for cf, var in zip(targets[k][1], dual_vars[k][first_w:]) if cf != 0])
            constr[-1] = self.Constraint(expr, ub=-threshold, name="DW"+str(k), sloppy=True)
            self.model.add(constr)

            # constraints for the target(s) (cuts and knock-ins)
            if bigM > 0:
                for i in range(self.num_reac):
                    if cuts[i]:
                        self.model.add(self.Constraint(dual_vars[k][i] - bigM*z_local[k][i],
                                       ub=0, name=z_local[k][i].name+dual_vars[k][i].name))
                        if reversible[i]:
                            if split_reversible_v:
                                dn = dual_vars[k][dual_rev_neg_idx_map[i]]
                            else:
                                dn = dual_vars[k][i]
                            self.model.add(self.Constraint(dn + bigM*z_local[k][i],
                                           lb=0, name=z_local[k][i].name+dn.name+"r"))
                #         if knock_in(i)
                #           lpfw.write_z_flux_link(obj.z_var_names{i}, dual_var_names{k}{i}, bigM, '<=');
                #           if ~irr(i)
                #             switch split_level
                #               case 1
                #                 dn= dual_var_names{k}{dual_rev_neg_idx_map(i)};
                #               case 2
                #                 dn= dual_var_names{k}{obj.num_reac+i};
                #               otherwise
                #                 dn= dual_var_names{k}{i};
                #             end
                #             lpfw.write_z_flux_link(obj.z_var_names{i}, dn, -bigM, '>=');
                #           end
                #         end
                #       end
            else: # indicators
                for i in range(self.num_reac):
                    if cuts[i]:
                        if split_reversible_v:
                            self.model.add(self.Constraint(dual_vars[k][i], ub=0,
                                           indicator_variable=z_local[k][i], active_when=0,
                                           name=z_local[k][i].name+dual_vars[k][i].name))
                            if reversible[i]:
                                dn = dual_vars[k][dual_rev_neg_idx_map[i]]
                                self.model.add(self.Constraint(dn, lb=0,
                                               indicator_variable=z_local[k][i], active_when=0,
                                               name=z_local[k][i].name+dn.name))
                        else:
                            if irr[i]:
                                lb = None
                            else:
                                lb = 0
                            self.model.add(self.Constraint(dual_vars[k][i], lb=lb, ub=0,
                                           indicator_variable=z_local[k][i], active_when=0,
                                           name=z_local[k][i].name+dual_vars[k][i].name))
                #         if knock_in(i)
                #           fprintf(lpfw_fid, '%s = 1 -> %s <= 0\n', obj.z_var_names{i}, dual_var_names{k}{i});
                #           if ~irr(i)
                #             switch split_level
                #               case 1
                #                 dn= dual_var_names{k}{dual_rev_neg_idx_map(i)};
                #               case 2
                #                 dn= dual_var_names{k}{obj.num_reac+i};
                #               otherwise
                #                 dn= dual_var_names{k}{i};
                #             end
                #             fprintf(lpfw_fid, '%s = 1 -> %s >= 0\n', obj.z_var_names{i}, dn);
                #           end
                #         end
                #       end

        self.flux_vars= [None]*len(desired)
        for l in range(len(desired)):
            # desired[l][0]: D, desired[l][1]: d 
            flux_lb = desired[l][2]
            flux_ub = desired[l][3]
            self.flux_vars[l]= [self.optlang_variable_class("R"+str(l)+"_"+str(i),
                                lb=flux_lb[i], ub=flux_ub[i],
                                problem=self.model.problem) for i in range(self.num_reac)]
            self.model.add(self.flux_vars[l])
            expr = mcs_computation.matrix_row_expressions(st, self.flux_vars[l])
            constr= [None]*st.shape[0]
            for i in range(st.shape[0]):
                constr[i] = self.Constraint(expr[i], lb=0, ub=0, name="M"+str(l)+"_"+str(i), sloppy=True)
            self.model.add(constr)
            expr = mcs_computation.matrix_row_expressions(desired[l][0], self.flux_vars[l])
            constr= [None]*desired[l][0].shape[0]
            for i in range(desired[l][0].shape[0]):
                constr[i] = self.Constraint(expr[i], ub=desired[l][1][i], name="DES"+str(l)+"_"+str(i), sloppy=True)
            self.model.add(constr)

            for i in numpy.where(cuts)[0]:
                if flux_ub[i] != 0: # a repressible (non_essential) reacion must have flux_ub >= 0
                    # self.flux_vars[l][i] <= (1 - self.z_vars[i]) * flux_ub[i]
                    self.model.add(self.Constraint(
                      self.flux_vars[l][i] + flux_ub[i]*self.z_vars[i], ub=flux_ub[i],
                      name= self.flux_vars[l][i].name+self.z_vars[i].name+"UB"))
                if flux_lb[i] != 0: # a repressible (non_essential) reacion must have flux_ub >= 0
                    # self.flux_vars[l][i] >= (1 - self.z_vars[i]) * flux_lb[i]
                    self.model.add(self.Constraint(
                      self.flux_vars[l][i] + flux_lb[i]*self.z_vars[i], lb=flux_lb[i],
                      name= self.flux_vars[l][i].name+self.z_vars[i].name+"LB"))
            #         for i= knock_in_idx
            #           if flux_ub{l}(i) ~= 0
            #             lpfw.write_direct_z_flux_link(obj.z_var_names{i}, obj.flux_var_names{i, l}, flux_ub{l}(i), '<=');
            #           end
            #           if flux_lb{l}(i) ~= 0
            #             lpfw.write_direct_z_flux_link(obj.z_var_names{i}, obj.flux_var_names{i, l}, flux_lb{l}(i), '>=');
            #           end
            #         end
        
        self.evs_sz_lb = 0
        self.evs_sz = self.Constraint(add(self.z_vars), lb=self.evs_sz_lb, name='evs_sz')
        self.model.add(self.evs_sz)
        self.model.update() # transfer the model to the solver

    def single_solve(self):
        status = self.model._optimize() # raw solve without any retries
        self.model._status = status # needs to be set when using _optimize
        #self.model.problem.parameters.reset() # CPLEX raw
        #self.model.problem.solve() # CPLEX raw
        #cplex_status = self.model.problem.solution.get_status() # CPLEX raw
        #self.model._status = optlang.cplex_interface._CPLEX_STATUS_TO_STATUS[cplex_status] # CPLEX raw
        #status = self.model.status
        #if self.model.problem.solution.get_status_string() == 'integer optimal solution': # CPLEX raw
        if status is optlang.interface.OPTIMAL or status is optlang.interface.FEASIBLE:
            print("Found solution with objective value", self.model.objective.value)
            z_idx= tuple(i for zv, i in zip(self.z_vars, range(len(self.z_vars))) if round(zv.primal))
            if self.ref_set is not None and z_idx not in self.ref_set:
                print("Incorrect result")
                print([zv.primal for zv in self.z_vars])
                print([(n,v) for n,v in zip(self.model.problem.variables.get_names(), self.model.problem.solution.get_values()) if v != 0])
                self.write_lp_file('failed')
            return z_idx
        else:
            return None

    def add_exclusion_constraint(self, mcs):
        expression = add([self.z_vars[i] for i in mcs])
        ub = len(mcs) - 1
        self.model.add(self.Constraint(expression, ub=ub, sloppy=True))

    def enumerate_mcs(self, max_mcs_size=None, max_mcs_num=float('inf'), enum_method=1, timeout=None,
                        model=None, targets=None, desired=None, info=None) -> Tuple[List[Union[Tuple[int], FrozenSet[int]]], int]:
        # model is the metabolic network, not the MILP
        # returns a list of sorted tuples (enum_method 1-3) or a list of frozensets (enum_method 4)
        # if a dictionary is passed as info some status/runtime information is stored in there
        all_mcs = []
        err_val = 0
        if enum_method == 2 or enum_method == 4:
            if self._optlang_interface is not optlang.cplex_interface:
                raise TypeError('enum_methods 2/4 is not available for this solver.')
            if max_mcs_size is None:
                max_mcs_size = len(self.z_vars)
            self.model.problem.parameters.mip.pool.intensity.set(4)
            # self.model.problem.parameters.mip.pool.absgap.set(0)
            self.model.problem.parameters.mip.pool.relgap.set(self.model.configuration.tolerances.optimality)
            if max_mcs_num == float('inf'):
                self.model.problem.parameters.mip.limits.populate.set(self.model.problem.parameters.mip.pool.capacity.get())
            z_idx = self.model.problem.variables.get_indices([z.name for z in self.z_vars]) # for solution pool/callback
            if enum_method == 2:
                print("Populate by cardinality up tp MCS size ", max_mcs_size)
                self.model.problem.parameters.emphasis.mip.set(1) # integer feasibility
                self.evs_sz.ub = self.evs_sz_lb # make sure self.evs_sz.ub is not None
            else:
                print("Continuous search up to MCS size", max_mcs_size)
                self.evs_sz.ub = max_mcs_size
                self.evs_sz.lb = self.evs_sz_lb
                if self.model.objective is self.zero_objective:
                    self.model.objective = self.minimize_sum_over_z
                    print('Objective function is empty; set objective to self.minimize_sum_over_z')
                cut_set_cb = CPLEXmakeMCSCallback(z_idx, model, targets, desired=desired, max_mcs_num=max_mcs_num)
                self.model.problem.set_callback(cut_set_cb, cplex.callbacks.Context.id.candidate)
        elif enum_method == 1 or enum_method == 3:
            if self.model.objective is self.zero_objective:
                self.model.objective = self.minimize_sum_over_z
                print('Objective function is empty; set objective to self.minimize_sum_over_z')
            if enum_method == 3:
                target_constraints= mcs_computation.get_leq_constraints(model, targets)
            if max_mcs_size is not None:
                self.evs_sz.ub = max_mcs_size
        else:
            raise ValueError('Unknown enumeration method.')
        continue_loop = True
        start_time = time.monotonic()
        while continue_loop and (max_mcs_size is None or self.evs_sz_lb <= max_mcs_size) and len(all_mcs) < max_mcs_num:
            if timeout is not None:
                remaining_time = round(timeout - (time.monotonic() - start_time)) # integer in optlang
                if remaining_time <= 0:
                    print('Time limit exceeded, stopping enumeration.')
                    break
                else:
                    self.model.configuration.timeout = remaining_time
            if enum_method == 1 or enum_method == 3:
                mcs = self.single_solve()
                if self.model.status == 'optimal' or (enum_method == 3 and self.model.status == 'feasible'):
                    # if self.model.status == 'optimal': # cannot use this because e.g. CPLEX 'integer optimal, tolerance' is also optlang 'optimal'
                    # GLPK appears to not have functions for accesing the MIP gap or best bound
                    global_optimum = enum_method == 1 or \
                                     (self._optlang_interface is optlang.cplex_interface and 
                                      self.model.problem.solution.MIP.get_mip_relative_gap() < self.model.configuration.tolerances.optimality) or \
                                     (self._optlang_interface is optlang.glpk_interface and self.model.status == 'optimal') # geht das sicher?
                    if global_optimum: #enum_method == 1: # only for this method optlang 'optimal' is a guaranteed global optimum
                        ov = round(self.model.objective.value)
                        if ov >  self.evs_sz_lb:
                            self.evs_sz_lb = ov
                        #if ov > e.evs_sz.lb: # increase lower bound of evs_sz constraint, but is this really always helpful?
                        #    e.evs_sz.lb = ov
                        #    print(ov)
                        
                        # not necessary when self.evs_sz.ub = max_mcs_size
                        # if round(self.model.objective.value) > max_mcs_size:
                        #     print('MCS size limit exceeded, stopping enumeration.')
                        #     break
                    else: # enum_method == 3: # and self.model.status == 'feasible':
                        print("CS", mcs, end="")
                        mcs = mcs_computation.make_minimal_cut_set(model, mcs, target_constraints)
                        print(" -> MCS", mcs)
                    self.add_exclusion_constraint(mcs)
                    self.model.update() # needs to be done explicitly when using _optimize
                    all_mcs.append(mcs)
                else:
                    print('Stopping enumeration with status', self.model.status)
                    if self.model.status != 'infeasible':
                        err_val = 1
                    continue_loop = False
            elif enum_method == 2: # populate with CPLEX
                if max_mcs_num != float('inf'):
                    self.model.problem.parameters.mip.limits.populate.set(max_mcs_num - len(all_mcs))
                if self.evs_sz_lb != self.evs_sz.lb: # only touch the bounds if necessary to preserve the search tree
                    self.evs_sz.ub = self.evs_sz_lb
                    self.evs_sz.lb = self.evs_sz_lb
                print("Enumerating MCS of size", self.evs_sz_lb)
                try:
                    self.model.problem.populate_solution_pool()
                except CplexSolverError:
                    print("Exception raised during populate")
                    continue_loop = False
                    err_val = 1
                    break
                print("Found", self.model.problem.solution.pool.get_num(), "MCS.")
                print("Solver status is:", self.model.problem.solution.get_status_string())
                cplex_status = self.model.problem.solution.get_status()
                if type(info) is dict:
                    info['cplex_status'] = cplex_status
                    info['cplex_status_string'] = self.model.problem.solution.get_status_string()
                if cplex_status is SolutionStatus.MIP_optimal or cplex_status is SolutionStatus.MIP_time_limit_feasible \
                        or cplex_status is SolutionStatus.optimal_populated_tolerance: # may occur when a non-zero onjective function is set
                    if cplex_status is SolutionStatus.MIP_optimal or cplex_status is SolutionStatus.optimal_populated_tolerance:
                        self.evs_sz_lb += 1
                        print("Increased MCS size to:", self.evs_sz_lb)
                    for i in range(self.model.problem.solution.pool.get_num()):
                        mcs = tuple(numpy.where(numpy.round(
                                    self.model.problem.solution.pool.get_values(i, z_idx)))[0])
                        self.add_exclusion_constraint(mcs)
                        all_mcs.append(mcs)
                    self.model.update() # needs to be done explicitly when not using optlang optimize
                elif cplex_status is SolutionStatus.MIP_infeasible:
                    print('No MCS of size ', self.evs_sz_lb)
                    self.evs_sz_lb += 1
                elif cplex_status is SolutionStatus.MIP_time_limit_infeasible:
                    print('No further MCS of size', self.evs_sz_lb, 'found, time limit reached.')
                else:
                    print('Unexpected CPLEX status', self.model.problem.solution.get_status_string())
                    err_val = 1
                    continue_loop = False
                # reset parameters here?
            elif enum_method == 4: # continuous solve with CPLEX
                try:
                    self.model.problem.populate_solution_pool()
                except CplexSolverError:
                    print("Exception raised during populate")
                    err_val = 1
                    continue_loop = False
                    break
                print("Found", len(cut_set_cb.minimal_cut_sets), "MCS.")
                print("Solver status is: ", self.model.problem.solution.get_status_string(),
                      ", best bound is ", self.model.problem.solution.MIP.get_best_objective(), sep="")
                all_mcs = cut_set_cb.minimal_cut_sets
                cplex_status = self.model.problem.solution.get_status()
                if type(info) is dict:
                    info['cplex_status'] = cplex_status
                    info['cplex_status_string'] = self.model.problem.solution.get_status_string()
                if cplex_status is SolutionStatus.MIP_infeasible:
                    print("Enumerated all MCS up to size", max_mcs_size)
                    self.evs_sz_lb = max_mcs_size + 1
                elif cplex_status is SolutionStatus.MIP_time_limit_feasible \
                        or cplex_status is SolutionStatus.MIP_time_limit_infeasible:
                    print("Stopped enumeration due to time limit.")
                elif cplex_status is SolutionStatus.MIP_abort_feasible or cplex_status is SolutionStatus.MIP_abort_infeasible:
                    if cut_set_cb.abort_status == 1:
                        print("Stopped enumeration because number of MCS has reached limit.")
                    elif cut_set_cb.abort_status == -1:
                        print("Aborted enumeration due to excessive generation of candidates that are not cut sets.")
                        err_val = -1
                else:
                    print('Unexpected CPLEX status', self.model.problem.solution.get_status_string())
                    err_val = 1
                continue_loop = False
        if type(info) is dict:
            info['optlang_status'] = self.model.status
            info['time'] = time.monotonic() - start_time
        return all_mcs, err_val

    def write_lp_file(self, fname):
        fname = fname + r".lp"
        if isinstance(self.model, optlang.cplex_interface.Model):
            self.model.problem.write(fname)
        elif isinstance(self.model, optlang.glpk_interface.Model):
            glp_write_lp(self.model.problem, None, fname)
        else:
            raise NotImplementedError("Writing LP files not yet implemented for this solver.")

class CPLEXmakeMCSCallback():
    def __init__(self, z_vars_idx, model, targets, desired=None, max_mcs_num=float('inf'), redundant_constraints=True):
        # needs max_mcs_num parameter
        self.z_vars_idx = z_vars_idx
        self.candidate_count = 0
        self.minimal_cut_sets = []
        self.model = model
        self.target_constraints= mcs_computation.get_leq_constraints(model, targets)
        if desired is None:
            self.desired_constraints = None
        else:
            self.desired_constraints = mcs_computation.get_leq_constraints(model, desired)
        self.redundant_constraints = redundant_constraints
        self.non_cut_set_candidates = 0
        self.abort_status = 0 # 1: stop because max_mcs_num is reached; -1: aborted due to excessive generation of candidates that are not cut sets
        self.max_mcs_num = max_mcs_num
    
    def invoke(self, context):
        if context.in_candidate() and context.is_candidate_point(): # there are also candidate rays but these should not occur here
            self.candidate_count += 1
            cut_set = numpy.nonzero(numpy.round(context.get_candidate_point(self.z_vars_idx)))[0]
            print("CS", cut_set, end="")
            if self.desired_constraints is not None:
                for des in self.desired_constraints:
                    if not mcs_computation.check_mcs(self.model, des, [cut_set], optlang.interface.OPTIMAL)[0]:
                        print(": Rejecting candidate that does not fulfill a desired behaviour.")
                        context.reject_candidate(constraints=[cplex.SparsePair([self.z_vars_idx[c] for c in cut_set], [1.0]*len(cut_set))],
                                                 senses="L", rhs=[len(cut_set)-1.0])
                        return
            for targ in self.target_constraints:
                if not mcs_computation.check_mcs(self.model, targ, [cut_set], optlang.interface.INFEASIBLE)[0]:
                    # cut_set cannot be a superset of an already identified MCS here
                    print(": Rejecting candidate that does not inhibit a target.")
                    self.non_cut_set_candidates += 1
                    if self.non_cut_set_candidates < max(100, len(self.minimal_cut_sets)):
                        context.reject_candidate()
                    else:
                        # there are no exclusion constraints for this case, therefore abort if it occurs repeatedly
                        print("\nAborting due to excessive generation of candidates that are not cut sets.")
                        self.abort_status = -1
                        context.abort()
                    return
            # print(len(cut_set), ", best bound:", context.get_double_info(cplex.callbacks.Context.info.best_bound)) # lags behind
            not_superset = True # not a superset of an already found MCS
            cut_set_s = set(cut_set) # cut_set is an array, need set for >= comparison
            for mcs in self.minimal_cut_sets:
                if cut_set_s >= mcs:
                    print(" already contained as", mcs)
                    not_superset = False
                    cut_set = mcs # for the lazy constraint
                    break
            if not_superset:
                if len(cut_set) > context.get_double_info(cplex.callbacks.Context.info.best_bound):
                    # could use ceiling of best bound unless there are non-integer intervention costs
                    cut_set = mcs_computation.make_minimal_cut_set(self.model, cut_set, self.target_constraints)
                    print(" -> MCS", cut_set, end="")
                else:
                    print(" is MCS", end="")
                self.minimal_cut_sets.append(frozenset(cut_set))
                print(";", len(self.minimal_cut_sets), "MCS found so far.")
                if len(self.minimal_cut_sets) >= self.max_mcs_num:
                    print("Reached maximum number of MCS.")
                    self.abort_status = 1
                    context.abort()
            if not_superset or self.redundant_constraints:
                # !! from the reference manual:
                # !! There is however no guarantee that CPLEX will actually use those additional constraints.
                context.reject_candidate(constraints=[cplex.SparsePair([self.z_vars_idx[c] for c in cut_set], [1.0]*len(cut_set))],
                                         senses="L", rhs=[len(cut_set)-1.0])
            else:
                context.reject_candidate()
 