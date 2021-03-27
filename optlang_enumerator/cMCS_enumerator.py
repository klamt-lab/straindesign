import numpy
import scipy
import cobra
import optlang.glpk_interface
from optlang.symbolics import add, mul
from optlang.exceptions import IndicatorConstraintsNotSupported
from swiglpk import glp_write_lp, glp_ios_mip_gap, GLP_DUAL
try:
    import optlang.cplex_interface
    from cplex.exceptions import CplexSolverError
    from cplex._internal._subinterfaces import SolutionStatus # can be also accessed by a CPLEX object under .solution.status
except:
    optlang.cplex_interface = None # make sure this symbol is defined for type() comparisons
import itertools
from typing import List, Tuple
import time
import sys
import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from cobra.core.configuration import Configuration
from os.path import join, dirname
import efmtool_link.efmtool4cobra as efmtool4cobra

class ConstrainedMinimalCutSetsEnumerator:
    def __init__(self, optlang_interface, st, reversible, targets, kn=None, cuts=None,
        desired=[], knock_in=[], bigM=0, threshold=1, split_reversible_v=True,
        irrev_geq=False, ref_set= None): # reduce_constraints=True, combined_z=True
        # the matrices in st, targets and desired should be numpy.array or scipy.sparse (csr, csc, lil) format
        # targets is a list of (T,t) pairs that represent T <= t
        # implements only combined_z which implies reduce_constraints=True
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
        num_targets = len(targets)
        use_kn_in_dual = kn is not None
        if use_kn_in_dual:
            if irrev_geq:
                raise
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
            self.z_vars[i].ub = 0 # nur wenn es keine KI sind
        self.minimize_sum_over_z= optlang_interface.Objective(add(self.z_vars), direction='min', name='minimize_sum_over_z')
        z_local = [None] * num_targets
        if num_targets == 1:
            z_local[0] = self.z_vars # global and local Z are the same if there is only one target
        else:
            # with reduce constraints is should not be necessary to have local Z, they can be the same as the global Z
            for k in range(num_targets):
                z_local[k] = [self.optlang_variable_class("Z"+str(k)+"_"+str(i), type="binary", problem=self.model.problem) for i in range(self.num_reac)]
                self.model.add(z_local[k])
            for i in range(self.num_reac):
                if cuts[i]: # && ~knock_in(i) % knock-ins only use global Z, do not need local ones
                    self.model.add(self.Constraint(
                        (1/num_targets - 1e-9)*add([z_local[k][i] for k in range(num_targets)]) - self.z_vars[i], ub=0,
                        name= "ZL"+str(i)))

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
            expr = matrix_row_expressions(dual, dual_vars[k])
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
            expr = matrix_row_expressions(st, self.flux_vars[l])
            constr= [None]*st.shape[0]
            for i in range(st.shape[0]):
                constr[i] = self.Constraint(expr[i], lb=0, ub=0, name="M"+str(l)+"_"+str(i), sloppy=True)
            self.model.add(constr)
            expr = matrix_row_expressions(desired[l][0], self.flux_vars[l])
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
            print(self.model.objective.value)
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
                        model=None, targets=None, info=None): #, larger_mcs=None):
        # if a dictionary is passed as info some status/runtime information is stored in there
        all_mcs = []
        if enum_method == 2:
            if self._optlang_interface is not optlang.cplex_interface:
                raise TypeError('enum_method 2 is not available for this solver.')
            if max_mcs_size is None:
                max_mcs_size = len(self.z_vars)
            print("Populate up tp MCS size ", max_mcs_size)
            self.model.problem.parameters.mip.pool.intensity.set(4)
            # self.model.problem.parameters.mip.pool.absgap.set(0)
            self.model.problem.parameters.mip.pool.relgap.set(self.model.configuration.tolerances.optimality)
            self.model.problem.parameters.mip.strategy.search.set(1) # traditional branch-and-cut search
            self.model.problem.parameters.emphasis.mip.set(1) # integer feasibility
            # also set model.problem.parameters.parallel to deterministic?
            if max_mcs_num == float('inf'):
                self.model.problem.parameters.mip.limits.populate.set(self.model.problem.parameters.mip.pool.capacity.get())
            z_idx = self.model.problem.variables.get_indices([z.name for z in self.z_vars]) # for querying the solution pool
            self.evs_sz.ub = self.evs_sz_lb # make sure self.evs_sz.ub is not None
        elif enum_method == 1 or enum_method == 3:
            if self.model.objective is self.zero_objective:
                self.model.objective = self.minimize_sum_over_z
                print('Objective function is empty; set objective to self.minimize_sum_over_z')
            if enum_method == 3:
                target_constraints= get_leq_constraints(model, targets)
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
                        print("CS", mcs)
                        mcs = make_minimal_cut_set(model, mcs, target_constraints)    
                        print("MCS", mcs)
                    self.add_exclusion_constraint(mcs)
                    self.model.update() # needs to be done explicitly when using _optimize
                    all_mcs.append(mcs)
                else:
                    print('Stopping enumeration with status', self.model.status)
                    continue_loop = False
            elif enum_method == 2: # populate with CPLEX
                if max_mcs_num != float('inf'):
                    self.model.problem.parameters.mip.limits.populate.set(max_mcs_num - len(all_mcs))
                if self.evs_sz_lb != self.evs_sz.lb: # only touch the bounds if necessary to preserve the search tree
                    self.evs_sz.ub = self.evs_sz_lb
                    self.evs_sz.lb = self.evs_sz_lb
                try:
                    self.model.problem.populate_solution_pool()
                except CplexSolverError:
                    print("Exception raised during populate")
                    continue_loop = False
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
                    print('Unexpected CPLEX status ', self.model.problem.solution.get_status_string())
                    continue_loop = False
                    break # provisional break
                # reset parameters here?
        if type(info) is dict:
            info['optlang_status'] = self.model.status
            info['time'] = time.monotonic() - start_time
        return all_mcs

    def write_lp_file(self, fname):
        fname = fname + r".lp"
        if isinstance(self.model, optlang.cplex_interface.Model):
            self.model.problem.write(fname)
        elif isinstance(self.model, optlang.glpk_interface.Model):
            glp_write_lp(self.model.problem, None, fname)
        else:
            raise # add a proper exception here

def equations_to_matrix(model, equations):
    # add option to use names instead of ids
    # allow equations to be a list of lists
    dual = cobra.Model()
    reaction_ids = [r.id for r in model.reactions]
    dual.add_metabolites([cobra.Metabolite(r) for r in reaction_ids])
    for i in range(len(equations)):
        r = cobra.Reaction("R"+str(i)) 
        dual.add_reaction(r)
        r.build_reaction_from_string('=> '+equations[i])
    dual = cobra.util.array.create_stoichiometric_matrix(dual, array_type='DataFrame')
    if numpy.all(dual.index.values == reaction_ids):
        return dual.values.transpose()
    else:
        raise RuntimeError("Index order was not preserved.")

def expand_mcs(mcs: List[Tuple], subT):
    mcs = [[list(m)] for m in mcs] # list of lists; mcs[i] will contain a list of MCS expanded from it
    rxn_in_sub = [numpy.where(subT[:, i])[0] for i in range(subT.shape[1])]
    for i in range(len(mcs)):
        num_iv = len(mcs[i][0]) # number of interventions in this MCS
        for s_idx in range(num_iv): # subset index
            for j in range(len(mcs[i])):
                rxns = rxn_in_sub[mcs[i][j][s_idx]]
                mcs[i][j][s_idx] = rxns[0]
                for k in range(1, len(rxns)):
                    mcs[i].append(mcs[i][j].copy())
                    mcs[i][-1][s_idx] = rxns[k]
    mcs = list(itertools.chain(*mcs))
    return list(map(tuple, map(numpy.sort, mcs)))

def matrix_row_expressions(mat, vars):
    # mat can be a numpy matrix or scipy sparse matrix (csc, csr, lil formats work; COO/DOK formats do not work)
    # expr = [None] * mat.shape[0]
    # for i in range(mat.shape[0]):
    #     idx = numpy.nonzero(mat)
    ridx, cidx = mat.nonzero() # !! assumes that the indices in ridx are grouped together, not fulfilled for DOK !! 
    if len(ridx) == 0:
        return []
    # expr = []
    expr = [None] * mat.shape[0]
    first = 0
    current_row = ridx[0]
    i = 1
    while True:
        at_end = i == len(ridx)
        if at_end or ridx[i] != current_row:
            # expr.append(add([mat[current_row, c] * vars[c] for c in cidx[first:i]]))
            expr[current_row] = add([mat[current_row, c] * vars[c] for c in cidx[first:i]])
            if at_end:
                break
            first = i
            current_row = ridx[i]
        i = i + 1
    return expr

def leq_constraints(optlang_constraint_class, row_expressions, rhs):
    return [optlang_constraint_class(expr, ub=ub) for expr, ub in zip(row_expressions, rhs)]

def check_mcs(model, constr, mcs, expected_status, flux_expr=None):
    # if flux_expr is None:
    #     flux_expr = [r.flux_expression for r in model.reactions]
    check_ok= numpy.zeros(len(mcs), dtype=numpy.bool)
    with model as constr_model:
        constr_model.problem.Objective(0)
        if flux_expr is None:
            flux_expr = [r.flux_expression for r in constr_model.reactions]
        rexpr = matrix_row_expressions(constr[0], flux_expr)
        constr_model.add_cons_vars(leq_constraints(constr_model.problem.Constraint, rexpr, constr[1]))
        # constr_model.add_cons_vars(get_leq_constraints(model, [constr], flux_expr=flux_expr))
        for m in range(len(mcs)):
            with constr_model as KO_model:
                for r in mcs[m]:
                    if type(r) is str:
                        KO_model.reactions.get_by_id(r).knock_out()
                    else: # assume r is an index if it is not a string
                        KO_model.reactions[r].knock_out()
                # for r in KO_model.reactions.get_by_any(mcs[m]): # get_by_any() does not accept tuple
                #     r.knock_out()
                KO_model.slim_optimize()
                check_ok[m] = KO_model.solver.status == expected_status
    return check_ok

from swiglpk import * # for direkt use of glü_exact, tentative solution only
def make_minimal_cut_set(model, cut_set, target_constraints):
    original_bounds = [model.reactions[r].bounds for r in cut_set]
    keep_ko = [True] * len(cut_set)
    # with model as KO_model:
    #     for r in cut_set:
    #         KO_model.reactions[r].knock_out()
    try:
        for r in cut_set:
            model.reactions[r].knock_out()
        for i in range(len(cut_set)):
            r = cut_set[i]
            model.reactions[r].bounds = original_bounds[i]
            still_infeasible = True
            for target in target_constraints:
                with model as target_model:
                    target_model.problem.Objective(0)
                    target_model.add_cons_vars(target)
                    if type(target_model.solver) is optlang.glpk_exact_interface.Model:
                        target_model.solver.update() # need manual update because GLPK is called through private function
                        status = target_model.solver._run_glp_exact() # optimize would run GLPK first
                        if status == 'undefined':
                            # print('Making fresh model')
                            # target_model_copy = target_model.copy() # kludge to lose the old basis
                            # status = target_model_copy.solver._run_glp_exact()
                            print("Make new basis")
                            glp_adv_basis(target_model.solver.problem, 0) # probably not with rational arithmetric?
                            status = target_model.solver._run_glp_exact() # optimize would run GLPK first
                        print(status)
                    else:
                        target_model.slim_optimize()
                        status = target_model.solver.status
                    still_infeasible = still_infeasible and status == optlang.interface.INFEASIBLE
                    if still_infeasible is False:
                        break
            if still_infeasible:
                keep_ko[i] = False # this KO is redundant
            else:
                model.reactions[r].knock_out() # reactivate
        mcs = tuple(ko for(ko, keep) in zip(cut_set, keep_ko) if keep)
    # don't handle the exception, just make sure the model is restored
    finally:
        for i in range(len(cut_set)):
            r = cut_set[i]
            model.reactions[r].bounds = original_bounds[i]
        model.solver.update() # just in case...
    return mcs

def parse_relation(lhs : str, rhs : float, reac_id_symbols=None):
    transformations = (standard_transformations + (implicit_multiplication_application,))
    slash = lhs.find('/')
    if slash >= 0:
        denominator = lhs[slash+1:]
        numerator = lhs[0:slash]
        denominator = parse_expr(denominator, transformations=transformations, evaluate=False, local_dict=reac_id_symbols)
        denominator = sympy.collect(denominator, denominator.free_symbols)
        numerator = parse_expr(numerator, transformations=transformations, evaluate=False, local_dict=reac_id_symbols)
        numerator = sympy.collect(numerator, numerator.free_symbols)
        lhs = numerator - rhs*denominator
        rhs = 0
    else:
        lhs = parse_expr(lhs, transformations=transformations, evaluate=False, local_dict=reac_id_symbols)
    lhs = sympy.collect(lhs, lhs.free_symbols, evaluate=False)
    
    return lhs, rhs

def parse_relations(relations : List, reac_id_symbols=None):
    for r in range(len(relations)):
        lhs, rhs = parse_relation(relations[r][0], relations[r][2], reac_id_symbols=reac_id_symbols)
        relations[r] = (lhs, relations[r][1], rhs)
    return relations

# def get_reac_id_symbols(model) -> dict:
#     return {id: sympy.symbols(id) for id in model.reactions.list_attr("id")}

def get_reac_id_symbols(reac_id) -> dict:
    return {rxn: sympy.symbols(rxn) for rxn in reac_id}

def relations2leq_matrix(relations : List, variables):
    matrix = numpy.zeros((len(relations), len(variables)))
    rhs = numpy.zeros(len(relations))
    for i in range(len(relations)):
        if relations[i][1] == ">=":
            f = -1.0
        else:
            f = 1.0
        for r in relations[i][0].keys(): # the keys are symbols
            matrix[i][variables.index(str(r))] = f*relations[i][0][r]
        rhs[i] = f*relations[i][2]
    return matrix, rhs # matrix <= rhs

def get_leq_constraints(model, leq_mat : List[Tuple], flux_expr=None):
    # leq_mat can be either targets or desired (as matrices)
    # returns contstraints that can be added to model 
    if flux_expr is None:
        flux_expr = [r.flux_expression for r in model.reactions]
    return [leq_constraints(model.problem.Constraint, matrix_row_expressions(lqm[0], flux_expr), lqm[1]) for lqm in leq_mat]

# !! in cobrapy a reaction that runs backward only also counts as irreversible
def reaction_bounds_to_leq_matrix(model):
    config = Configuration()
    lb_idx = []
    ub_idx = []
    for i in range(len(model.reactions)):
        if model.reactions[i].lower_bound not in (0, config.lower_bound, -float('inf')):
            lb_idx.append(i)
            # print(model.reactions[i].id, model.reactions[i].lower_bound)
        if model.reactions[i].upper_bound not in (0, config.upper_bound, float('inf')):
            ub_idx.append(i)
            # print(model.reactions[i].id, model.reactions[i].upper_bound)
    num_bounds = len(lb_idx) + len(ub_idx)
    leq_mat = scipy.sparse.lil_matrix((num_bounds, len(model.reactions)))
    rhs = numpy.zeros(num_bounds)
    count = 0
    for r in lb_idx:
        leq_mat[count, r] = -1.0
        rhs[count] = -model.reactions[r].lower_bound
        count += 1
    for r in ub_idx:
        leq_mat[count, r] = 1.0
        rhs[count] = model.reactions[r].upper_bound
        count += 1
    return leq_mat, rhs

def integrate_model_bounds(model, targets, desired=None):
    bounds_mat, bounds_rhs = reaction_bounds_to_leq_matrix(model)
    for i in range(len(targets)):
        targets[i] = (scipy.sparse.vstack((targets[i][0], bounds_mat), format='lil'), numpy.hstack((targets[i][1], bounds_rhs)))
    if desired is not None:
        for i in range(len(desired)):
            desired[i] = (scipy.sparse.vstack((desired[i][0], bounds_mat), format='lil'), numpy.hstack((desired[i][1], bounds_rhs)))


# convenience function
def compute_mcs(model, targets, desired=None, cuts=None, enum_method=1, max_mcs_size=2, max_mcs_num=1000, timeout=600,
                exclude_boundary_reactions_as_cuts=False, network_compression=True, fva_tolerance=1e-9,
                include_model_bounds=True):
    # if include_model_bounds=True this function integrates non-default reaction bounds of the model into the
    # target and desired regions and directly modifies(!) these parameters

    # make fva_res and compressed model optional parameters
    if desired is None:
        desired = []
    else:
        config = Configuration()

    target_constraints= get_leq_constraints(model, targets)
    desired_constraints= get_leq_constraints(model, desired)

    # check whether all target/desired regions are feasible
    for i in range(len(targets)):
        with model as feas:
            feas.objective = model.problem.Objective(0.0)
            feas.add_cons_vars(target_constraints[i])
            feas.slim_optimize()
            if feas.solver.status != 'optimal':
                raise Exception('Target region', i, 'is not feasible; solver status is', feas.solver.status)
    for i in range(len(desired)):
        with model as feas:
            feas.objective = model.problem.Objective(0.0)
            feas.add_cons_vars(desired_constraints[i])
            feas.slim_optimize()
            if feas.solver.status != 'optimal':
                raise Exception('Desired region', i, 'is not feasible; solver status is', feas.solver.status)

    if include_model_bounds:
        integrate_model_bounds(model, targets, desired)

    if cuts is None:
        cuts= numpy.full(len(model.reactions), True, dtype=bool)
    if exclude_boundary_reactions_as_cuts:
        for r in range(len(model.reactions)):
            if model.reactions[r].boundary:
                cuts[r] = False

    # get blocked reactions with glpk_exact FVA (includes those that are blocked through (0,0) bounds)
    print("Running FVA to find blocked reactions...")
    start_time = time.monotonic()
    with model as fva:
        # fva.solver = 'glpk_exact' # too slow for large models
        fva.tolerance = fva_tolerance
        fva.objective = model.problem.Objective(0.0)
        if fva.problem.__name__ == 'optlang.glpk_interface':
            # should emulate setting an optimality tolerance (which GLPK simplex does not have)
            fva.solver.configuration._smcp.meth = GLP_DUAL
            fva.solver.configuration._smcp.tol_dj = fva_tolerance
        elif fva.problem.__name__ == 'optlang.coinor_cbc_interface':
            fva.solver.problem.opt_tol = fva_tolerance
        # currently unsing just 1 process is much faster than 2 or 4 ?!? not only with glpk_exact, also with CPLEX
        # is this a Windows problem? yes, multiprocessing performance under Windows is fundamemtally poor
        fva_res = cobra.flux_analysis.flux_variability_analysis(fva, fraction_of_optimum=0.0, processes=1)
    print(time.monotonic() - start_time)
    # integrate FVA bounds into model? might be helpful for compression because some reversible reactions may have become irreversible
    if network_compression:
        compr_model = model.copy() # preserve the original model
        # integrate FVA bounds and flip reactions where necessary
        flipped = []
        for i in range(fva_res.values.shape[0]): # assumes the FVA results are ordered same as the model reactions
            if abs(fva_res.values[i, 0]) > fva_tolerance: # resolve with glpk_exact?
                compr_model.reactions[i].lower_bound = fva_res.values[i, 0]
            else:
                # print('LB', fva_res.index[i], fva_res.values[i, :])
                compr_model.reactions[i].lower_bound = 0
            if abs(fva_res.values[i, 1]) > fva_tolerance: # resolve with glpk_exact?
                compr_model.reactions[i].upper_bound = fva_res.values[i, 1]
            else:
                # print('UB', fva_res.index[i], fva_res.values[i, :])
                compr_model.reactions[i].upper_bound = 0
 
        subT = efmtool4cobra.compress_model_sympy(compr_model)
        model = compr_model
        reduced = cobra.util.array.create_stoichiometric_matrix(model, array_type='dok', dtype=numpy.object)
        stoich_mat = efmtool4cobra.dokRatMat2lilFloatMat(reduced) # DOK does not (always?) work
        targets = [(T@subT, t) for T, t in targets]
        desired = [(D@subT, d) for D, d in desired]
        full_cuts = cuts # needed for MCS expansion
        cuts = numpy.any(subT[cuts, :], axis=0)
    else:
        stoich_mat = cobra.util.array.create_stoichiometric_matrix(model, array_type='lil')
        blocked_rxns = []
        for i in range(fva_res.values.shape[0]):
            # if res.values[i, 0] == 0 and res.values[i, 1] == 0:
            if fva_res.values[i, 0] >= -fva_tolerance and fva_res.values[i, 1] <= fva_tolerance:
                blocked_rxns.append(fva_res.index[i])
                cuts[i] = False
        print("Found", len(blocked_rxns), "blocked reactions:\n", blocked_rxns) # FVA may not be worth it without compression

    rev = [r.lower_bound < 0 for r in model.reactions] # use this as long as there might be irreversible backwards only reactions
    # add FVA bounds for desired
    desired_constraints= get_leq_constraints(model, desired)
    print("Running FVA for desired regions...")
    for i in range(len(desired)):
        with model as fva_desired:
            fva_desired.tolerance = fva_tolerance
            fva_desired.objective = model.problem.Objective(0.0)
            if fva_desired.problem.__name__ == 'optlang.glpk_interface':
                # should emulate setting an optimality tolerance (which GLPK simplex does not have)
                fva_desired.solver.configuration._smcp.meth = GLP_DUAL
                fva_desired.solver.configuration._smcp.tol_dj = fva_tolerance
            elif fva_desired.problem.__name__ == 'optlang.coinor_cbc_interface':
                fva_desired.solver.problem.opt_tol = fva_tolerance
            fva_desired.add_cons_vars(desired_constraints[i])
            fva_res = cobra.flux_analysis.flux_variability_analysis(fva_desired, fraction_of_optimum=0.0, processes=1)
            # make tiny FVA values zero
            fva_res.values[numpy.abs(fva_res.values) < fva_tolerance] = 0
            essential = numpy.where(numpy.logical_or(fva_res.values[:, 0] > fva_tolerance, fva_res.values[:, 1] < -fva_tolerance))[0]
            print(len(essential), "essential reactions in desired region", i)
            cuts[essential] = False
            # fva_res.values[fva_res.values[:, 0] == -numpy.inf, 0] = config.lower_bound # cannot happen because cobrapy FVA does not do unbounded
            # fva_res.values[fva_res.values[:, 1] == numpy.inf, 1] = config.upper_bound
            desired[i] = (desired[i][0], desired[i][1], fva_res.values[:, 0], fva_res.values[:, 1])
            
    optlang_interface = model.problem
    if optlang_interface.Constraint._INDICATOR_CONSTRAINT_SUPPORT:
        bigM = 0.0
        print("Using indicators.")
    else:
        bigM = 1000.0
        print("Using big M.")

    e = ConstrainedMinimalCutSetsEnumerator(optlang_interface, stoich_mat, rev, targets, desired=desired,
                                    bigM=bigM, threshold=0.1, cuts=cuts, split_reversible_v=True, irrev_geq=True)
    if enum_method == 3:
        if optlang_interface.__name__ == 'optlang.cplex_interface':
            e.model.problem.parameters.mip.tolerances.mipgap.set(0.98)
        elif optlang_interface.__name__ == 'optlang.glpk_interface':
            e.model.configuration._iocp.mip_gap = 0.98
        elif optlang_interface.__name__ == 'optlang.coinor_cbc_interface':
            e.model.problem.max_solutions = 1 # stop with first feasible solutions
        else:
            print('No method implemented for this solver to stop with a suboptimal incumbent, will behave like enum_method 1.')
   if optlang_interface.__name__ == 'optlang.coinor_cbc_interface':
       e.model.problem.threads = -1 # activate multithreading
    
    e.evs_sz_lb = 1 # feasibility of all targets has been checked
    mcs = e.enumerate_mcs(max_mcs_size=max_mcs_size, max_mcs_num=max_mcs_num, enum_method=enum_method,
                            model=model, targets=targets)
    if network_compression:
        xsubT= subT.copy()
        xsubT[numpy.logical_not(full_cuts), :] = 0 # only expand to reactions that are repressible within a given subset
        mcs = expand_mcs(mcs, xsubT)
    # print(mcs)
    return mcs
