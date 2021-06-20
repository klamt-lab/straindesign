import numpy
import scipy
import cobra
import optlang.glpk_interface
from swiglpk import GLP_DUAL
try:
    import optlang.cplex_interface
except:
    optlang.cplex_interface = None # make sure this symbol is defined for type() comparisons
try:
    import optlang.gurobi_interface
except:
    optlang.gurobi_interface = None # make sure this symbol is defined for type() comparisons
try:
    import optlang.coinor_cbc_interface
except:
    optlang.coinor_cbc_interface = None # make sure this symbol is defined for type() comparisons
import itertools
from typing import List, Tuple, Union, Set, FrozenSet
import time
import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from cobra.core.configuration import Configuration
import efmtool_link.efmtool4cobra as efmtool4cobra
import optlang_enumerator.cMCS_enumerator as cMCS_enumerator

def expand_mcs(mcs: List[Union[Tuple[int], Set[int], FrozenSet[int]]], subT) -> List[Tuple[int]]:
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
            # expr[current_row] = sympy.simplify(add([mat[current_row, c] * vars[c] for c in cidx[first:i]])) # simplify to flatten the sum, slow/hangs
            expr[current_row] = sympy.Add(*[mat[current_row, c] * vars[c] for c in cidx[first:i]]) # gives flat sum
            # expr[current_row] = sum([mat[current_row, c] * vars[c] for c in cidx[first:i]]) # gives flat sum, slow/hangs
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
        if isinstance(constr[0], optlang.interface.Constraint):
            constr_model.add_cons_vars(constr)
        else:
            if flux_expr is None:
                flux_expr = [r.flux_expression for r in constr_model.reactions]
            rexpr = matrix_row_expressions(constr[0], flux_expr)
            constr_model.add_cons_vars(leq_constraints(constr_model.problem.Constraint, rexpr, constr[1]))
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

from swiglpk import glp_adv_basis # for direkt use of glp_exact, experimental only
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

class InfeasibleRegion(Exception):
    pass

# convenience function
def compute_mcs(model, targets, desired=None, cuts=None, enum_method=1, max_mcs_size=2, max_mcs_num=1000, timeout=600,
                exclude_boundary_reactions_as_cuts=False, network_compression=True, fva_tolerance=1e-9,
                include_model_bounds=True, bigM=0, mip_opt_tol=1e-6, mip_feas_tol=1e-6, mip_int_tol=1e-6,
                set_mip_parameters_callback=None) -> List[Tuple[int]]:
    # if include_model_bounds=True this function integrates non-default reaction bounds of the model into the
    # target and desired regions and directly modifies(!) these parameters

    # make fva_res and compressed model optional parameters
    if desired is None:
        desired = []

    target_constraints= get_leq_constraints(model, targets)
    desired_constraints= get_leq_constraints(model, desired)

    # check whether all target/desired regions are feasible
    for i in range(len(targets)):
        with model as feas:
            feas.objective = model.problem.Objective(0.0)
            feas.add_cons_vars(target_constraints[i])
            feas.slim_optimize()
            if feas.solver.status != 'optimal':
                raise InfeasibleRegion('Target region '+str(i)+' is not feasible; solver status is: '+feas.solver.status)
    for i in range(len(desired)):
        with model as feas:
            feas.objective = model.problem.Objective(0.0)
            feas.add_cons_vars(desired_constraints[i])
            feas.slim_optimize()
            if feas.solver.status != 'optimal':
                raise InfeasibleRegion('Desired region'+str(i)+' is not feasible; solver status is: '+feas.solver.status)

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
        # when include_model_bounds=False modify bounds so that only reversibilites are used?
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
        targets = [[T@subT, t] for T, t in targets]
        # as a result of compression empty constraints can occur (e.g. limits on reactions that turn out to be blocked)
        for i in range(len(targets)): # remove empty target constraints
            keep = numpy.any(targets[i][0], axis=1)
            targets[i][0] = targets[i][0][keep, :]
            targets[i][1] = targets[i][1][keep]
        desired = [[D@subT, d] for D, d in desired]
        for i in range(len(desired)): # remove empty desired constraints
            keep = numpy.any(desired[i][0], axis=1)
            desired[i][0] = desired[i][0][keep, :]
            desired[i][1] = desired[i][1][keep]
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
    if optlang_interface.Constraint._INDICATOR_CONSTRAINT_SUPPORT and bigM == 0:
        bigM = 0.0
        print("Using indicators.")
    else:
        bigM = 1000.0
        print("Using big M.")

    e = cMCS_enumerator.ConstrainedMinimalCutSetsEnumerator(optlang_interface, stoich_mat, rev, targets, desired=desired,
                                    bigM=bigM, threshold=0.1, cuts=cuts, split_reversible_v=True, irrev_geq=True)
    if enum_method == 3:
        if optlang_interface.__name__ == 'optlang.cplex_interface':
            e.model.problem.parameters.mip.tolerances.mipgap.set(0.98)
        elif optlang_interface.__name__ == 'optlang.gurobi_interface':
            e.model.problem.Params.MipGap = 0.98
        elif optlang_interface.__name__ == 'optlang.glpk_interface':
            e.model.configuration._iocp.mip_gap = 0.98
        elif optlang_interface.__name__ == 'optlang.coinor_cbc_interface':
            e.model.problem.max_solutions = 1 # stop with first feasible solutions
        else:
            print('No method implemented for this solver to stop with a suboptimal incumbent, will behave like enum_method 1.')
    elif enum_method == 4:
        e.model.configuration.verbosity = 3
    # if optlang_interface.__name__ == 'optlang.coinor_cbc_interface':
    #    e.model.problem.threads = -1 # activate multithreading
    
    e.evs_sz_lb = 1 # feasibility of all targets has been checked
    e.model.configuration.tolerances.optimality = mip_opt_tol
    e.model.configuration.tolerances.feasibility = mip_feas_tol
    e.model.configuration.tolerances.integrality = mip_int_tol
    if set_mip_parameters_callback != None:
        set_mip_parameters_callback(e.model.problem)
    mcs, err_val = e.enumerate_mcs(max_mcs_size=max_mcs_size, max_mcs_num=max_mcs_num, enum_method=enum_method,
                            model=model, targets=targets, desired=desired, timeout=timeout)
    if network_compression:
        xsubT= subT.copy()
        xsubT[numpy.logical_not(full_cuts), :] = 0 # only expand to reactions that are repressible within a given subset
        mcs = expand_mcs(mcs, xsubT)
    elif enum_method == 4:
        mcs = [tuple(sorted(m)) for m in mcs]
    return mcs, err_val

def stoich_mat2cobra(stoich_mat, irrev_reac):
    model = cobra.Model('stoich_mat')
    model.add_metabolites([cobra.Metabolite('M'+str(i)) for i in range(stoich_mat.shape[0])])
    model.add_reactions([cobra.Reaction('R'+str(i)) for i in range(stoich_mat.shape[1])])
    for r in range(stoich_mat.shape[1]):
        if irrev_reac[r] == 0:
            model.reactions[r].lower_bound = cobra.Configuration().lower_bound
        model.reactions[r].add_metabolites({model.metabolites[m]: stoich_mat[m, r] for m in numpy.nonzero(stoich_mat[:, r])[0]})
    return model

def equations_to_matrix(model, equations):
    # deprecated
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

