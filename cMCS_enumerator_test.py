# exec(open('optlang_mcs_test5.py').read())
#%%
import cobra
import cobra.util.array
from optlang_enumerator.cMCS_enumerator import *
from optlang_enumerator.mcs_computation import *
import time
import numpy
import os
import sys
import efmtool_link.efmtool_intern as efmtool_intern
import efmtool_link.efmtool4cobra as efmtool4cobra
import pickle
import optlang_enumerator
import mcs_computation
from importlib import reload
#%%
reload(optlang_enumerator.cMCS_enumerator)
from optlang_enumerator.cMCS_enumerator import *
reload(optlang_enumerator.mcs_computation)
from optlang_enumerator.mcs_computation import *

#%%
ex = cobra.io.read_sbml_model(r"metatool_example_no_ext.xml")
ex.solver = 'coinor_cbc'
stdf = cobra.util.array.create_stoichiometric_matrix(ex, array_type='DataFrame')
rev = [r.reversibility for r in ex.reactions]
reac_id = stdf.columns.tolist()
reac_id_symbols = get_reac_id_symbols(reac_id)
# target = [(equations_to_matrix(ex, ["-1 Pyk", "-1 Pck"]), [-1, -1])] # -Pyk does not work
target = [[("Pyk", ">=", 1), ("Pck", ">=", 1)]]
target = [relations2leq_matrix(parse_relations(t, reac_id_symbols=reac_id_symbols), reac_id) for t in target]
#target.append(target[0]) # duplicate target
flux_expr= [r.flux_expression for r in ex.reactions] # !! lose validity when the solver is changed !!
target_constraints= get_leq_constraints(ex, target)
kn = efmtool_intern.null_rat_efmtool(stdf.values)
#res = cobra.flux_analysis.single_reaction_deletion(ex, processes=1) # no interactive multiprocessing on Windows
    
# %%
info = dict()
e = ConstrainedMinimalCutSetsEnumerator(optlang.glpk_interface, stdf.values, rev, target, 
                                        bigM= 100, threshold=0.1, split_reversible_v=True, irrev_geq=True)
# e.model.objective = e.minimize_sum_over_z
# e.write_lp_file('testM')
mcs,_ = e.enumerate_mcs(max_mcs_size=5, info=info)
print(info)

e = ConstrainedMinimalCutSetsEnumerator(optlang.cplex_interface, stdf.values, rev, target,
                                        threshold=0.1, split_reversible_v=False, irrev_geq=False, kn=kn) #, ref_set=mcs)
# e.model.objective = e.minimize_sum_over_z
# e.write_lp_file('testI')
#e.model.configuration.verbosity = 3
#e.model.configuration.presolve = 'auto' # presolve remains off on the CPLEX side
#e.model.configuration.presolve = True # presolve on, this is CPLEX default
#e.model.configuration.lp_method = 'auto' # sets lpmethod back to automatic on the CPLEX side
#e.model.problem.parameters.reset() # works (CPLEX specific)
# without reset() optlang switches presolve off and fixes lpmethod
mcs2,_ = e.enumerate_mcs(max_mcs_size=5, info=info)
print(info)

print(len(set(mcs).intersection(set(mcs2))))
print(set(mcs) == set(mcs2))
all(check_mcs(ex, target[0], mcs, optlang.interface.INFEASIBLE))

# %%
# e.model.problem.solution.MIP.get_best_objective()
# e.model.problem.solution.MIP.get_mip_relative_gap()
e = ConstrainedMinimalCutSetsEnumerator(optlang.cplex_interface, stdf.values, rev, target, 
                                        bigM= 100, threshold=0.1, split_reversible_v=True, irrev_geq=True)
e.model.problem.parameters.mip.tolerances.mipgap.set(0.99)
info = dict()
mcs3,_ = e.enumerate_mcs(max_mcs_size=5, enum_method=3, model=ex, targets=target, info=info)
print(info)
print(e.evs_sz_lb)
print(set(mcs) == set(mcs3))

#%%
e = ConstrainedMinimalCutSetsEnumerator(optlang.glpk_interface, stdf.values, rev, target, 
                                        bigM= 100, threshold=0.1, split_reversible_v=True, irrev_geq=True)
info = dict()
e.model.configuration._iocp.mip_gap = 0.99
mcs3,_ = e.enumerate_mcs(max_mcs_size=5, enum_method=3, model=ex, targets=target, info=info)
print(info)
print(e.evs_sz_lb)
print(set(mcs) == set(mcs3))

#%%
# flux_lb= -1000*numpy.ones(stdf.values.shape[1])
# flux_lb[numpy.logical_not(rev)] = 0
# flux_ub= 1000*numpy.ones(stdf.values.shape[1])
flux_lb = numpy.array([ex.reactions[i].lower_bound for i in range(len(ex.reactions))])
flux_ub = numpy.array([ex.reactions[i].upper_bound for i in range(len(ex.reactions))])
cobra_config = cobra.Configuration()
flux_lb[numpy.isinf(flux_lb)] = cobra_config.lower_bound
flux_ub[numpy.isinf(flux_ub)] = cobra_config.upper_bound
desired = [(equations_to_matrix(ex, ["-1 AspCon"]), [-1], flux_lb, flux_ub)]
# desired[0] = list(desired[0])
# desired[0][2] = numpy.array([   0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.
# ,0.,0.,0.,1.,0., -999., -999., -999., -999.,1.,1.,0.])
# desired[0][3] = numpy.array([1000.,1000., 999., 999., 999., 500., 500.,1000., 999.,1000.,
#  1000.,1000., 499.5,999., 999.,1000., 999., 999., 999.,1000.,
#  1000.,1000.,1000., 999. ])
# cuts = numpy.array([ True,True,True,True,True,True,True, False,True,True,True,True
# ,True,True,True, False,True,True,True,True,True, False, False,True])
# e = ConstrainedMinimalCutSetsEnumerator(optlang.cplex_interface, stdf.values, rev, target,
#                                         threshold=0.1, split_reversible_v=False, irrev_geq=False,
#                                         desired=desired)
e = ConstrainedMinimalCutSetsEnumerator(optlang.cplex_interface, stdf.values, rev, target,
                                        threshold=0.1, split_reversible_v=True, irrev_geq=True,
                                        desired=desired, SOS_constraints=True)
# e.model.objective = e.minimize_sum_over_z
# e.write_lp_file('testA')
mcs3,_ = e.enumerate_mcs() #max_mcs_size=5)
print(len(mcs3))
print(all(check_mcs(ex, target[0], mcs3, optlang.interface.INFEASIBLE)))
print(all(check_mcs(ex, desired[0], mcs3, optlang.interface.OPTIMAL)))
#%%
with ex as exb: # cobrapy FVA raises error with unbounded reactions
    exb.solver = 'cplex'
    for r in exb.reactions:
        if r.lower_bound == -numpy.inf:
            r.lower_bound = -1000
        if r.upper_bound == numpy.inf:
            r.upper_bound = 1000
    # mcs4,_ = compute_mcs(exb, target, desired=desired, enum_method=1, max_mcs_size=20)
    mcs4,_ = compute_mcs(exb, target, enum_method=2, max_mcs_size=20, desired=desired, network_compression=True)
print(len(mcs4))
set(mcs3) == set(mcs4)
#%%
# subset_compression = efmtool_intern.CompressionMethod[:]([efmtool_intern.CompressionMethod.CoupledZero, efmtool_intern.CompressionMethod.CoupledCombine, efmtool_intern.CompressionMethod.CoupledContradicting])
rd, subT, comprec = efmtool_intern.compress_rat_efmtool(stdf.values, rev, remove_cr=True,
            compression_method= efmtool_intern.subset_compression) #[0:2]
# rd = stdf.values
# subT = numpy.eye(rd.shape[1])
rev_rd = numpy.logical_not(numpy.any(subT[numpy.logical_not(rev), :], axis=0))
#%%
# exc, subT = efmtool4cobra.compress_model(ex)
exc = ex.copy()
subT = efmtool4cobra.compress_model_sympy(exc)
rd = cobra.util.array.create_stoichiometric_matrix(exc, array_type='dok') #, dtype=numpy.object) # does actually work with rationals
rev_rd = [r.reversibility for r in exc.reactions]

#%%
target_rd = [(T@subT, t) for T, t in target]
# e = ConstrainedMinimalCutSetsEnumerator(optlang.glpk_interface, rd, rev_rd, target_rd, 
#                                         bigM= 100, threshold=0.1, split_reversible_v=True, irrev_geq=True)
# e = ConstrainedMinimalCutSetsEnumerator(optlang.glpk_interface, rd, rev_rd, target_rd, 
#                                         bigM= 100, threshold=0.1, split_reversible_v=True, kn=efmtool_intern.null_rat_efmtool(rd))
e = ConstrainedMinimalCutSetsEnumerator(optlang.coinor_cbc_interface, rd, rev_rd, target_rd, 
                                        bigM= 100, threshold=0.1, split_reversible_v=True, kn=efmtool_intern.null_rat_efmtool(rd))
# e.model.objective = e.minimize_sum_over_z
# e.model.configuration._iocp.mip_gap = 0.99
rd_mcs,_ = e.enumerate_mcs(max_mcs_size=5, enum_method=1, model=exc, targets=target_rd)
# rd_mcs,_ = e.enumerate_mcs(max_mcs_size=5)
print(len(rd_mcs))
#set(expand_mcs(rd_mcs, subT)) == set(map(lambda x: tuple(numpy.where(x)[0]), mcs2))
set(expand_mcs(rd_mcs, subT)) == set(mcs2)

#%%
e = ConstrainedMinimalCutSetsEnumerator(optlang.cplex_interface, stdf.values, rev, target,
                                        threshold=0.1, split_reversible_v=False, irrev_geq=False) #, ref_set=mcs)
# e.model.configuration.verbosity = 3
# e.model.objective = e.minimize_sum_over_z
mcs2p,_ = e.enumerate_mcs(max_mcs_size=5, enum_method=2) #, max_mcs_num=1)
print(set(mcs) == set(mcs2p))
# e.model.problem.solution.get_objective_value() == 0

#%%
cuts = numpy.full(24, True, dtype=bool)
cuts[0] = False
cuts[23] = False
e = ConstrainedMinimalCutSetsEnumerator(optlang.cplex_interface, stdf.values, rev, target, 
                                        cuts=cuts, threshold=0.1, split_reversible_v=True, irrev_geq=True)
e.model.objective = e.minimize_sum_over_z
# e.write_lp_file('testI2')
t = time.time()
mcs3,_ = e.enumerate_mcs(max_mcs_size=5)
print(time.time() - t)

e = ConstrainedMinimalCutSetsEnumerator(optlang.glpk_interface, stdf.values, rev, target, 
                                        cuts=cuts, bigM=100, threshold=0.1, split_reversible_v=True, irrev_geq=False)
e.model.objective = e.minimize_sum_over_z
# e.write_lp_file('testI2')
t = time.time()
mcs4,_ = e.enumerate_mcs(max_mcs_size=5)
print(time.time() - t)

print(set(mcs3) == set([m for m in mcs if 0 not in m and 23 not in m]))
print(set(mcs4) == set(mcs3))

#%%
ecc2 = cobra.io.read_sbml_model(r"../cnapy-projects/ECC2comp/model.sbml")
# ecc2.solver = 'glpk'
ecc2_stdf = cobra.util.array.create_stoichiometric_matrix(ecc2, array_type='DataFrame')
cuts= numpy.full(ecc2_stdf.shape[1], True, dtype=bool) # results do not agree when exchange reactions can be cut, problem with tiny fluxes (and M too small)
# for r in ecc2.boundary:
#     cuts[ecc2_stdf.columns.get_loc(r.id)] = False
# for r in range(len(ecc2.reactions)):
#     if ecc2.reactions[r].boundary:
#         cuts[r] = False
cuts = numpy.array([not r.boundary for r in ecc2.reactions])
reac_id = ecc2_stdf.columns.tolist()
reac_id_symbols = get_reac_id_symbols(reac_id)
# ecc2_mue_target = [[("Growth", ">=", 0.01), ("GlcUp", "<=", 10), ("AcUp", "<=", 10), ("GlycUp", "<=", 10), ("SuccUp", "<=", 10)]]
ecc2_mue_target = [[("Growth", ">=", 0.01)]]
ecc2_mue_target = [relations2leq_matrix(parse_relations(t, reac_id_symbols=reac_id_symbols), reac_id) for t in ecc2_mue_target]
bounds_mat, bounds_rhs = reaction_bounds_to_leq_matrix(ecc2)
ecc2_mue_target = [(scipy.sparse.vstack((t[0], bounds_mat), format='csr'), numpy.hstack((t[1], bounds_rhs))) for t in ecc2_mue_target]
ecc2_mue_target_constraints= get_leq_constraints(ecc2, ecc2_mue_target)
# %%
e = ConstrainedMinimalCutSetsEnumerator(optlang.cplex_interface, ecc2_stdf.values, [r.reversibility for r in ecc2.reactions], ecc2_mue_target,
                                        cuts=cuts, threshold=0.1, split_reversible_v=True, irrev_geq=True)#, SOS_constraints=True)
# e.model.objective = e.minimize_sum_over_z
# e.model.configuration.tolerances.feasibility = 1e-9
# e.model.configuration.tolerances.optimality = 1e-9
# e.model.configuration.tolerances.integrality = 1e-10
#e.write_lp_file('testI')
#e.model.configuration.verbosity = 3
e.evs_sz_lb = 1 
info = dict()
ecc2_mcs,_ = e.enumerate_mcs(max_mcs_size=3, enum_method=3, model=ecc2, targets=ecc2_mue_target, info=info)
# ecc2_mcs,_ = e.enumerate_mcs(max_mcs_size=3, enum_method=1, info=info)
print(info) # in this example SOS are somewhat slower than indicators
# %%
e = ConstrainedMinimalCutSetsEnumerator(optlang.cplex_interface, ecc2_stdf.values, [r.reversibility for r in ecc2.reactions], ecc2_mue_target,
                                        cuts=cuts, threshold=0.1, split_reversible_v=True, irrev_geq=True)#, SOS_constraints=True)
e.model.problem.parameters.mip.pool.intensity.set(4)
e.model.problem.parameters.mip.strategy.search.set(1) # traditional branch-and-cut search
z_idx = e.model.problem.variables.get_indices([z.name for z in e.z_vars])

cut_set_cb = MakeMinimalCutSetCallback(z_idx, ecc2, ecc2_mue_target) #, redundant_constraints=False)
e.model.problem.set_callback(cut_set_cb, cplex.callbacks.Context.id.candidate)
e.model.problem.parameters.mip.limits.populate.set(e.model.problem.parameters.mip.pool.capacity.get())
e.model.configuration.verbosity = 3
e.model.problem.parameters.emphasis.mip.set(1) # integer feasibility
e.model.objective = e.minimize_sum_over_z
e.evs_sz.lb = 1
e.evs_sz.ub = 3
e.model.problem.populate_solution_pool()
print(e.model.problem.solution.get_status_string(), cut_set_cb.candidate_count,
      len(set(cut_set_cb.minimal_cut_sets)))

# %%
# ecc2_mcs,_ = compute_mcs(ecc2, ecc2_mue_target, [], cuts, 2, 3, 1000, 100)
ecc2_mcs,_ = compute_mcs(ecc2, ecc2_mue_target, cuts=cuts, enum_method=3, max_mcs_size=3, network_compression=True)
print(len(ecc2_mcs))
ecc2_mcs_rxns= [tuple(ecc2_stdf.columns[r] for r in mcs) for mcs in ecc2_mcs]
print(ecc2_mcs_rxns)
ecc2_mcsF,_ = compute_mcs(ecc2, ecc2_mue_target, cuts=cuts, enum_method=2, max_mcs_size=3, network_compression=False)
print(all(check_mcs(ecc2, ecc2_mue_target[0], ecc2_mcs, optlang.interface.INFEASIBLE)))
print(set(ecc2_mcs) == set(ecc2_mcsF))

# %%
with open("ecc2_mcs.pkl","wb") as f:
    pickle.dump(set(ecc2_mcs), f)

# %%
# ecc2.solver = 'glpk_exact'
# ecc2.solver = 'glpk'
# ecc2.solver = 'cplex'
with ecc2 as model:
    model.objective = model.problem.Objective(0)
    model.solver = 'glpk_exact'
    model_target_constraints= get_leq_constraints(model, ecc2_mue_target)
    model.add_cons_vars(model_target_constraints[0])
    res = cobra.flux_analysis.single_reaction_deletion(model, reaction_list=ecc2_stdf.columns[cuts], processes=1) # no interactive multiprocessing on Windows
del model
single_cuts= list(map(tuple, res.index[res.values[:, 1] == optlang.interface.INFEASIBLE]))
print(set(single_cuts) == set(mcs for mcs in ecc2_mcs_rxns if len(mcs) == 1))
print(set(single_cuts) - set(mcs for mcs in ecc2_mcs_rxns if len(mcs) == 1))
check_mcs(ecc2, ecc2_mue_target[0], single_cuts, optlang.interface.INFEASIBLE)

# %% FVA with copied model
model = ecc2.copy() 
model.objective = model.problem.Objective(0)
fva_tol = 1e-9 # with CPLEX 1e-8 leads to removal of EX_adp_c, 1e-9 keeps EX_adp_c
model.tolerance = fva_tol # prevent essential EX_meoh_ex from being blocked, sets solver feasibility/optimality tolerances
# model.solver.configuration.tolerances.feasibility = 1e-9
# model.solver = 'glpk' #'glpk_exact' # appears to make problems for context management
# model_mue_target_constraints= get_leq_constraints(model, ecc2_mue_target)
# model.add_cons_vars(model_mue_target_constraints[0])
fva_res = cobra.flux_analysis.flux_variability_analysis(model, fraction_of_optimum=0, processes=1) # no interactive multiprocessing on Windows
print(fva_res.loc['EX_adp_c',:])
print(fva_res.loc['EX_AMOB_ex'])
blocked = []
blocked_rxns = []
for i in range(fva_res.values.shape[0]):
    # if fva_res.values[i, 0] == 0 and fva_res.values[i, 1] == 0:
    if fva_res.values[i, 0] >= -fva_tol and fva_res.values[i, 1] <= fva_tol:
        blocked.append(i)
        blocked_rxns.append(fva_res.index[i])
print(blocked_rxns)

# %% FVA in context
#model = ecc2.copy() # copy model because switching solver in context sometimes gives an error (?!?)
ecc2.solver = 'coinor_cbc'
ecc2.solver.problem.opt_tol = 1e-9
with ecc2 as model:
    model.objective = model.problem.Objective(0)
    fva_tol = 1e-9 # with CPLEX 1e-8 leads to removal of EX_adp_c, 1e-9 keeps EX_adp_c
    model.tolerance = fva_tol # prevent essential EX_meoh_ex from being blocked, sets solver feasibility/optimality tolerances
    # model.solver.configuration.tolerances.feasibility = 1e-9
    # model.solver = 'glpk' #'glpk_exact' # appears to make problems for context management
    # model_mue_target_constraints= get_leq_constraints(model, ecc2_mue_target)
    # model.add_cons_vars(model_mue_target_constraints[0])
    fva_res = cobra.flux_analysis.flux_variability_analysis(model, fraction_of_optimum=0, processes=1) # no interactive multiprocessing on Windows
print(fva_res.loc['EX_adp_c',:])
print(fva_res.loc['EX_AMOB_ex']) # !!! with GLPK significantly different compared to the copied model, OK with CPLEX
print(fva_res.loc['EX_meoh_ex',:])
blocked = []
blocked_rxns = []
for i in range(fva_res.values.shape[0]):
    # if fva_res.values[i, 0] == 0 and fva_res.values[i, 1] == 0:
    if fva_res.values[i, 0] >= -fva_tol and fva_res.values[i, 1] <= fva_tol:
        blocked.append(i)
        blocked_rxns.append(fva_res.index[i])
print(blocked_rxns)

# %% FVA with try/finally model restoration
model = ecc2
previous_tolerance = model.tolerance
previous_objective = model.objective
try:
    model.objective = model.problem.Objective(0)
    fva_tol = 1e-9 # with CPLEX 1e-8 leads to removal of EX_adp_c, 1e-9 keeps EX_adp_c
    model.tolerance = fva_tol # prevent essential EX_meoh_ex from being blocked, sets solver feasibility/optimality tolerances
    fva_res = cobra.flux_analysis.flux_variability_analysis(model, fraction_of_optimum=0, processes=1) # no interactive multiprocessing on Windows
finally:
    model.tolerance = previous_tolerance
    model.objective = previous_objective
print(fva_res.loc['EX_adp_c',:])
print(fva_res.loc['EX_AMOB_ex']) # !!! also incorrect ?!?
blocked = []
blocked_rxns = []
for i in range(fva_res.values.shape[0]):
    # if fva_res.values[i, 0] == 0 and fva_res.values[i, 1] == 0:
    if fva_res.values[i, 0] >= -fva_tol and fva_res.values[i, 1] <= fva_tol:
        blocked.append(i)
        blocked_rxns.append(fva_res.index[i])
print(blocked_rxns)

#%% only glpk_exact recognizes EX_adp_c as essential, but knocking it out is
# not a problem with CPLEX as its flux is below the minimal tolerance
# however compression with rationals will not work properly when EX_adp_c is removed
# because then growth will appear to be blocked
print(ecc2.slim_optimize())
for i in blocked:
    with ecc2 as model:
        model_mue_target_constraints= get_leq_constraints(model, ecc2_mue_target)
        model.add_cons_vars(model_mue_target_constraints[0])
        model.reactions[i].knock_out()
        print(model.reactions[i].id, model.slim_optimize())

# %% try FASTCC
model = ecc2.copy() # copy model because switching solver in context gives an error (?!?)
model.tolerance= 1e-9
model.objective = model.problem.Objective(0)
model.solver = 'glpk_exact' # appears to use CPLEX anyway?
# modelcc = cobra.flux_analysis.fastcc(ecc2, zero_cutoff=1e-9) # gives some falsely blocked reactions
modelcc = cobra.flux_analysis.fastcc(model, flux_threshold=10, zero_cutoff=1e-9) # still identifies EX_adp_c as blocked
del model
set(ecc2.reactions.list_attr("id")) - set(modelcc.reactions.list_attr("id"))

#%% FVA mit verschiedenen solvern machen um zu sehen ob {'EX_adp_c'} ein cut set ist (boundary müssen cuts sein)
ecc2.solver = 'glpk_exact'
with ecc2 as model:
    model.objective = model.problem.Objective(0)
    #print(model.objective.expression)
    flux_expr= [r.flux_expression for r in model.reactions] # !! lose validity when the solver is changed !!
    rexpr = matrix_row_expressions(ecc2_mue_target[0][0], flux_expr)
    model.add_cons_vars(leq_constraints(model.problem.Constraint, rexpr, ecc2_mue_target[0][1]))
    res = cobra.flux_analysis.flux_variability_analysis(model, [ecc2.reactions.get_by_id(r[0]) for r in single_cuts], fraction_of_optimum=0, processes=1) # no interactive multiprocessing on Windows
print(res.loc['EX_adp_c',:])
# CPLEX: 
# minimum    2.042810e-13
# maximum    9.059420e-14
# Name: EX_adp_c, dtype: float64
# glpk_exact
# minimum   -1.117471e-09
# maximum   -4.705152e-12
# Name: EX_adp_c, dtype: float64
numpy.linalg.matrix_rank(ecc2_stdf.values) # no conservation relations

#%%
e = ConstrainedMinimalCutSetsEnumerator(optlang.glpk_interface, ecc2_stdf.values, [r.reversibility for r in ecc2.reactions], ecc2_mue_target,
                                        cuts=cuts, bigM=1000, threshold=0.1, split_reversible_v=True, irrev_geq=True)
# e.model.objective = e.minimize_sum_over_z
#e.write_lp_file('testM')
#e.model.configuration.verbosity = 3
e.model.configuration._iocp.mip_gap = 0.99
ecc2_mcsB,_ = e.enumerate_mcs(max_mcs_size=3, enum_method=3, model=ecc2, targets=ecc2_mue_target)
print(set(ecc2_mcs) == set(ecc2_mcsB), len(ecc2_mcsB))

# %% sympy and efmtool conversions to rational are not necessarily the same, nsimplify appears to match
# v = 1/3
# v = 0.1
v = 0.2
r1 = sympy.Rational(v)
r2 = sympy.nsimplify(v, rational=True) # same as sympy.Rational with rational_conversion='exact'
print(r1, r2, abs(float(r1-r2)))
efmtool4cobra.jRatMat2sympyRatMat(efmtool_intern.numpy_mat2jBigIntegerRationalMatrix(numpy.array([[v]]))).values()

#%%
# not for enum_method 3 because this requires as compressed model for the minimality checks
rev = [r.reversibility for r in ecc2.reactions]
# rd, subT = efmtool_intern.compress_rat_efmtool(ecc2_stdf.values, rev, remove_cr=True, # if CR are not removed problem with enumeration
#             compression_method=efmtool_intern.subset_compression)[0:2]
rd, subT = efmtool_intern.compress_rat_efmtool(ecc2_stdf.values, rev, remove_cr=True, # if CR are not removed problem with enumeration
            compression_method=efmtool_intern.subset_compression, remove_rxns=blocked)[0:2] # OK when EX_adp_c is removed
rev_rd = numpy.logical_not(numpy.any(subT[numpy.logical_not(rev), :], axis=0))
kn = efmtool_intern.null_rat_efmtool(rd)
print(rd.shape)
print(numpy.linalg.matrix_rank(rd))
print(kn.shape)
print(numpy.max(abs(rd@kn)))

#%%
# compression_tolerance = numpy.finfo(float).eps # does not work
compression_tolerance = 1e-10 # OK
# compression_tolerance = 1e-9 # misses one CR
# ecc2c, subT = efmtool4cobra.compress_model(ecc2)
ecc2c, subT = efmtool4cobra.compress_model(ecc2, remove_rxns=blocked_rxns, tolerance=compression_tolerance) # tolerance=0 here also OK
rev_rd = [r.reversibility for r in ecc2c.reactions]
# rd = cobra.util.array.create_stoichiometric_matrix(ecc2c, array_type='dok')
# bc = efmtool_intern.basic_columns_rat(rd.transpose().toarray(), tolerance=1e-12) # needs non-zero tolerance or...
# rd = rd[numpy.sort(bc), :] # ...it misses one CR
efmtool4cobra.remove_conservation_relations(ecc2c, tolerance=compression_tolerance)
rd = cobra.util.array.create_stoichiometric_matrix(ecc2c, array_type='dok')
print(numpy.sort(list(map(abs, rd.values())))[[0, -1]])
kn = efmtool_intern.null_rat_efmtool(rd)
print(rd.shape)
print(numpy.linalg.matrix_rank(rd.toarray()))
print(kn.shape)
print(numpy.max(abs(rd@kn)))

#%%
ecc2c = ecc2.copy()
# subT = efmtool4cobra.compress_model_sympy(ecc2)
subT = efmtool4cobra.compress_model_sympy(ecc2c, remove_rxns=blocked_rxns)
print(len(ecc2c.metabolites), len(ecc2c.reactions))
rev_rd = [r.reversibility for r in ecc2c.reactions]
efmtool4cobra.remove_conservation_relations_sympy(ecc2c)
reduced = cobra.util.array.create_stoichiometric_matrix(ecc2c, array_type='dok', dtype=numpy.object)
# reduced, bc = efmtool4cobra.remove_conservation_relations_sympy(ecc2c, return_reduced_only=True)[0:2]
# rdb = scipy.sparse.dok_matrix((reduced.shape[0], reduced.shape[1]))
# for (r, c), v in reduced.items():
#     rdb[r, c] = float(v)
# rd = rdb
rd = efmtool4cobra.dokRatMat2lilFloatMat(reduced)
# for m in [ecc2c.metabolites[i].id for i in set(range(len(ecc2c.metabolites))) - set(bc)]:
#     ecc2c.metabolites.get_by_id(m).remove_from_model()
#%%
# kn = efmtool_intern.null_rat_efmtool(rd)
import ch.javasoft.smx.ops.Gauss as Gauss
jkn = Gauss.getRationalInstance().nullspace(efmtool4cobra.sympyRatMat2jRatMat(reduced))
kn = efmtool_intern.jpypeArrayOfArrays2numpy_mat(jkn.getDoubleRows())
print(rd.shape)
print(numpy.linalg.matrix_rank(rd.toarray())) # toarray() makes it a full matrix, otherwise weird result with scipy sparse
print(kn.shape) # !! is incomplete when tolerance=0
print(numpy.max(abs(rd@kn)))
#%% cannot get scipy sparse matrix multiplication to work, probably because the matrices have object dtype
import sympy.matrices.sparse
srd = sympy.matrices.sparse.MutableSparseMatrix(reduced.shape[0], reduced.shape[1], reduced.items())
# srd = sympy.matrices.SparseMatrix(reduced.shape[0], reduced.shape[1], dict(reduced.items()))
# skn = srd.nullspace() # extremely slow
skn = efmtool4cobra.jRatMat2sympyRatMat(jkn)
sympy.matrices.sparse.MutableSparseMatrix(skn.shape[0], skn.shape[1], skn.items())
max(abs(srd@skn))
#%%
#ecc2.solver = 'glpk_exact'
#ecc2c.solver = 'glpk_exact'
sol = ecc2.optimize()
growth_idx = ecc2.reactions.index('Growth')
growth_idx_c = numpy.where(subT[growth_idx, :])[0][0]
ecc2c.objective = ecc2c.reactions[growth_idx_c]
sol_c = ecc2c.optimize()
print(sol, sol_c, abs(sol.objective_value - sol_c.objective_value))
# # %%
# for carbon_uptake in ['AcUp', 'GlycUp', 'SuccUp', 'GlcUp']:
#     idx = ecc2.reactions.index(carbon_uptake)
#     idx_c = numpy.where(subT[idx, :])[0][0]
#     print(sol.fluxes[carbon_uptake], sol_c.fluxes[ecc2c.reactions[idx_c].id], subT[idx, idx_c])
# # at least the bound on GlcUp is lost

# #%%
# print(ecc2_stdf.values.shape) # has 40 extra exchange reactions for the 40 external metabolites in ECC2comp
# print(ecc2_stdf.columns[0])
# ecc2_stdf.columns[numpy.where(subT[:, 0])]

#%%
target_rd = [(T@subT, t) for T, t in ecc2_mue_target]

# e = ConstrainedMinimalCutSetsEnumerator(optlang.cplex_interface, rd, rev_rd, target_rd, 
#                                         threshold=0.1, split_reversible_v=True, irrev_geq=True,
#                                         cuts=numpy.any(subT[cuts, :], axis=0))
# e.model.problem.parameters.mip.tolerances.mipgap.set(0.99)

e = ConstrainedMinimalCutSetsEnumerator(optlang.coinor_cbc_interface, rd, rev_rd, target_rd, 
                                        threshold=0.1, bigM=1000, split_reversible_v=True, irrev_geq=True,
                                        cuts=numpy.any(subT[cuts, :], axis=0))#, kn=kn) #efmtool_intern.null_rat_efmtool(rd))
#e.model.problem.max_mip_gap = 0.99 # does not appear to make the solver stop early
e.model.problem.max_solutions = 1 # stops with feasible solutions
e.model.problem.threads = -1 # default does not appear to use multi-threading
#e = ConstrainedMinimalCutSetsEnumerator(optlang.glpk_interface, rd, rev_rd, target_rd, 
#                                        threshold=0.1, bigM=1000, split_reversible_v=True, #irrev_geq=True,
#                                        cuts=numpy.any(subT[cuts, :], axis=0), kn=kn) #efmtool_intern.null_rat_efmtool(rd))
#e.model.configuration._iocp.mip_gap = 0.99

# here a subset is repressible when one ot its reactions is repressible
# e.model.objective = e.minimize_sum_over_z
info = dict()
# rd_mcs,_ = e.enumerate_mcs(max_mcs_size=3, info=info)
# rd_mcs,_ = e.enumerate_mcs(max_mcs_size=3, enum_method=3, model=ecc2c, targets=target_rd, info=info)
# with ecc2c as tm:
    # tm.solver = 'glpk' #'glpk_exact' # actually optlang runs regular GLPK first and only if the results is optimal glpk_exact
    # tm.solver.configuration.verbosity = 3
rd_mcs,_ = e.enumerate_mcs(max_mcs_size=3, enum_method=3, model=ecc2c, targets=target_rd, info=info)
print(info)
print(len(rd_mcs))
xsubT= subT.copy()
xsubT[numpy.logical_not(cuts), :] = 0 # only expand to reactions that are repressible 
xmcs = expand_mcs(rd_mcs, xsubT)
set(xmcs) == set(ecc2_mcs)

#%%
from swiglpk import *
test_model = ecc2c.copy()
test_model.solver = 'glpk_exact'
target_constraints= get_leq_constraints(test_model, target_rd)
print(make_minimal_cut_set(test_model, (9,38,64), target_constraints)) # should give (38, 64), looks like GLPK has warm start issues
test_model.add_cons_vars(target_constraints[0])
# test_model.reactions[9].knock_out()
test_model.reactions[38].knock_out()
test_model.reactions[64].knock_out()
return_value = glp_exact(test_model.solver.problem, test_model.solver.configuration._smcp)
glpk_status = glp_get_status(test_model.solver.problem)
print(return_value, glpk_status)
print(test_model.solver._run_glp_exact())
test_model2 = test_model.copy() # with copy
return_value = glp_exact(test_model2.solver.problem, test_model2.solver.configuration._smcp)
glpk_status = glp_get_status(test_model2.solver.problem)
print(return_value, glpk_status)
print(test_model2.solver._run_glp_exact())
test_model3 = glp_create_prob()
glp_copy_prob(test_model3, test_model.solver.problem, GLP_OFF)
# glp_std_basis(test_model3) # without this the old basis is still used
glp_adv_basis(test_model3, 0) # without this the old basis is still used
return_value = glp_exact(test_model3, test_model.solver.configuration._smcp)
glpk_status = glp_get_status(test_model3)
print(return_value, glpk_status)

# %%
from swiglpk import *
test_model = ecc2c.copy()
# test_model = ecc2.copy()
test_model.solver = 'glpk_exact' # actually optlang runs regular GLPK first and only if the results is optimal glpk_exact
test_model.objective = test_model.reactions.EX_Biomass
print(test_model.objective)
print(test_model.solver._run_glp_exact())
ov = test_model.optimize()
print(ov)
# print(test_model.solver._run_glp_exact())
return_value = glp_exact(test_model.solver.problem, test_model.solver.configuration._smcp)
glpk_status = glp_get_status(test_model.solver.problem)
print(return_value, glpk_status)
target_constraints= get_leq_constraints(test_model, target_rd)
print(make_minimal_cut_set(test_model, (9,14,49), target_constraints)) # looks like GLPK has warm start issues
test_model.add_cons_vars(target_constraints[0])
# print(test_model.solver._run_glp_exact()) # -> undefined
return_value = glp_exact(test_model.solver.problem, test_model.solver.configuration._smcp)
glpk_status = glp_get_status(test_model.solver.problem)
print(return_value, glpk_status)
print(test_model.solver._run_glp_exact())
test_model.reactions[14].knock_out()
test_model.reactions[49].knock_out()
print(test_model.slim_optimize(), test_model.solver.status)

#%% FVA in reduced system

# %%
num_reac = len(ecc2c.reactions)
j_max = numpy.full((num_reac, num_reac), numpy.nan)
j_min = numpy.full((num_reac, num_reac), numpy.nan)
j_min_max = numpy.full((num_reac, num_reac), numpy.nan)
for i in range(num_reac):
    with ecc2c as mKOi:
        mKOi.reactions[i].knock_out()
        for j in range(i+1, num_reac): # hier doch über alle Reaktionen (ausser i)
            mKOi.objective = mKOi.reactions[j]
            mKOi.objective.direction = "max"
            j_max[i, j] = mKOi.slim_optimize()
            j_min_max[i, j] = mKOi.slim_optimize()
            mKOi.objective.direction = "min"
            j_min[i, j] = mKOi.slim_optimize()
            j_min_max[j, i] = mKOi.slim_optimize()
            if j_min_max[i, j] == j_min_max[j, i] and j_min_max[j, i] == 0:
                print(i, j)

#%%
iJO1366 = cobra.io.read_sbml_model(r"..\cnapy-projects\iJO1366\model.sbml")
# iJO1366.solver = 'glpk_exact'
# hash(str(cobra.io.model_to_dict(iJO1366))) # could this be used as a kind of model ID so that a compressed model can know if its parent model changed?
# %% full FVA
fva_tol = 1e-9
model = iJO1366.copy()
# model.solver = 'cplex' # 14.1 s, 878 blocked
# model.solver = 'coinor_cbc' # 128.3 s, 880 blocked -> probably does not warm start
model.solver = 'glpk' # 26.6 seconds, 879 blocked,  would be extremely slow with glpk_exact 
# model.objective = model.problem.Objective(0)
model.tolerance = fva_tol
start_time = time.monotonic()
fva_res = cobra.flux_analysis.flux_variability_analysis(model, fraction_of_optimum=0, processes=1) # no interactive multiprocessing on Windows
# fva_res = cobra.flux_analysis.flux_variability_analysis(model, fraction_of_optimum=0.99, processes=1) # fraction 1.0 may lead to too tight optimum constraint
print(time.monotonic() - start_time) # 20 seconds in iJO1366 without remove_rxns
del model
blocked_rxns = []
for i in range(fva_res.values.shape[0]):
    # if fva_res.values[i, 0] == 0 and fva_res.values[i, 1] == 0:
    if fva_res.values[i, 0] >= -fva_tol and fva_res.values[i, 1] <= fva_tol:
        blocked_rxns.append(fva_res.index[i])
print(len(blocked_rxns))
# %% problematic result close to zero near the solver tolerance
zero_threshold = fva_tol/100
nonzero_threshold = fva_tol*100
abs_fva_res = numpy.abs(fva_res.values)
idx = numpy.where(numpy.logical_or(numpy.logical_and(abs_fva_res[:, 0] < nonzero_threshold, abs_fva_res[:, 0] >= zero_threshold),
        numpy.logical_and(abs_fva_res[:, 1] < nonzero_threshold, abs_fva_res[:, 1] >= zero_threshold)))[0]
print(len(idx))
# %%
model = iJO1366.copy()
model.solver = 'glpk_exact'
# model.objective = model.problem.Objective(0)
model.tolerance = fva_tol
fva_res_gx = cobra.flux_analysis.flux_variability_analysis(model, reaction_list=[model.reactions[i] for i in idx], fraction_of_optimum=0.99, processes=1)
numpy.max(numpy.abs(fva_res_gx.values - fva_res.values[idx, :])) # no meaningful accuracy gain here

#%%
iJO1366c = iJO1366.copy()
# subT = efmtool4cobra.compress_model_sympy(iJO1366c)
subT = efmtool4cobra.compress_model_sympy(iJO1366c, blocked_rxns)
# %% save with rational stoichiometric coefficients
f = open("iJO1366c_subT.pkl","wb")
pickle.dump((cobra.io.model_to_dict(iJO1366c), subT),f)
f.close()
#%% loading back appears to work correctly
with open("iJO1366c_subT.pkl", "rb") as f:
    m, subT = pickle.load(f)
    iJO1366c = cobra.io.model_from_dict(m)
iJO1366c.solver = "glpk_exact" # solver type appears not to be restored

#%%
sol = iJO1366.optimize()
sol_c = iJO1366c.optimize()
print(sol, sol_c, abs(sol.objective_value - sol_c.objective_value))

#%%
rev_rd = [r.reversibility for r in iJO1366c.reactions]
efmtool4cobra.remove_conservation_relations_sympy(iJO1366c)
reduced = cobra.util.array.create_stoichiometric_matrix(iJO1366c, array_type='dok', dtype=numpy.object)
rd = efmtool4cobra.dokRatMat2lilFloatMat(reduced)
# reduced, bc = efmtool4cobra.remove_conservation_relations_sympy(iJO1366c)[0:2]
# rdb = scipy.sparse.dok_matrix((reduced.shape[0], reduced.shape[1]))
# for (r, c), v in reduced.items():
#     rdb[r, c] = float(v)
# rd = rdb
# for m in [iJO1366c.metabolites[i].id for i in set(range(len(iJO1366c.metabolites))) - set(bc)]:
#     iJO1366c.metabolites.get_by_id(m).remove_from_model()
sol_c = iJO1366c.optimize()
print(sol, sol_c, abs(sol.objective_value - sol_c.objective_value))

#%%
# kn = efmtool_intern.null_rat_efmtool(rd)
import ch.javasoft.smx.ops.Gauss as Gauss
jkn = Gauss.getRationalInstance().nullspace(efmtool4cobra.sympyRatMat2jRatMat(reduced))
kn = efmtool_intern.jpypeArrayOfArrays2numpy_mat(jkn.getDoubleRows())
print(rd.shape)
print(numpy.linalg.matrix_rank(rd.toarray())) # toarray() makes it a full matrix, otherwise weird result with scipy sparse
print(kn.shape) # !! is incomplete when tolerance=0
print(numpy.max(abs(rd@kn)))

#%%
iJO1366_stdf = cobra.util.array.create_stoichiometric_matrix(iJO1366, array_type='DataFrame')
cuts= numpy.full(iJO1366_stdf.shape[1], True, dtype=bool)
for r in iJO1366.boundary:
    cuts[iJO1366_stdf.columns.get_loc(r.id)] = False
#cuts = None
# iJO1366_mue_target = [(equations_to_matrix(iJO1366, 
#                     ["-1 BIOMASS_Ec_iJO1366_core_53p95M", "-1 EX_glc__D_e"]), [-0.01, 10])]
# iJO1366_mue_target_constraints= get_leq_constraints(iJO1366, iJO1366_mue_target)
reac_id = iJO1366.reactions.list_attr("id")
reac_id_symbols = get_reac_id_symbols(reac_id)
iJO1366_mue_target = [[("BIOMASS_Ec_iJO1366_core_53p95M", ">=", 0.01)]]
iJO1366_mue_target = [relations2leq_matrix(parse_relations(t, reac_id_symbols=reac_id_symbols), reac_id) for t in iJO1366_mue_target]
bounds_mat, bounds_rhs = reaction_bounds_to_leq_matrix(iJO1366)
iJO1366_mue_targetB = [(scipy.sparse.vstack((t[0], bounds_mat), format='csr'), numpy.hstack((t[1], bounds_rhs))) for t in iJO1366_mue_target]
iJO1366_mue_target_constraintsB= get_leq_constraints(iJO1366, iJO1366_mue_targetB)
# %% 
iJO1366_mcs,_ = compute_mcs(iJO1366, iJO1366_mue_target, cuts=cuts, enum_method=2, max_mcs_size=1, network_compression=True)
print(len(iJO1366_mcs))
# %% 
iJO1366_mcs,_ = compute_mcs(iJO1366, iJO1366_mue_target, cuts=cuts, enum_method=3, max_mcs_size=1, network_compression=True)
print(len(iJO1366_mcs))
# %%
e = ConstrainedMinimalCutSetsEnumerator(optlang.cplex_interface, iJO1366_stdf.values, [r.reversibility for r in iJO1366.reactions], iJO1366_mue_targetB,
                                        cuts=cuts, threshold=0.1, split_reversible_v=False, irrev_geq=False)
# e.model.objective = e.minimize_sum_over_z
#e.write_lp_file('testI')
# e.model.configuration.verbosity = 3
iJO1366_mcs,_ = e.enumerate_mcs(max_mcs_size=1, enum_method=2)
iJO1366_mcs_rxns = [tuple(reac_id[r] for r in mcs) for mcs in iJO1366_mcs]
print(len(iJO1366_mcs)) # 271
print(all(check_mcs(iJO1366, iJO1366_mue_targetB[0], iJO1366_mcs, optlang.interface.INFEASIBLE)))
# !!!!!!                                      ^^^ really need the implicit bounds in the target
# otherwise ATPM as KO would be missed in this example because its >= 3.15 requirement would vanisch through the KO in the check
#%% misses some cuts with CPLEX, OK with glpk_exact
candidates = [r for r, c in zip(iJO1366.reactions.list_attr("id"), cuts) if c]
with iJO1366 as model:
    model.objective = model.problem.Objective(0)
    # model.solver = 'glpk_exact' # again problems with context
    model_target_constraints= get_leq_constraints(model, iJO1366_mue_targetB)
    model.add_cons_vars(model_target_constraints[0])
    resD = cobra.flux_analysis.single_reaction_deletion(model, reaction_list=candidates,
                                                         processes=1) # no interactive multiprocessing on Windows
    single_cuts= list(map(tuple, resD.index[resD.values[:, 1] == optlang.interface.INFEASIBLE]))
    print(len(single_cuts)) # 255 ??
    print(set(iJO1366_mcs_rxns) - set(single_cuts))
# print(set(single_cuts) == set(mcs for mcs in iJO1366_mcs_rxns if len(mcs) == 1))
#%% wird auch infeasible wenn die nächste cell vorher ausgeführt wird?!?!??!
# muss was mit dem internen solver status zu tun haben da das Modell sich nicht ändert (str(cobra.io.model_to_dict(iJO1366) bleibt identisch)
# auch infeasible wenn der solver nach dem Laden auf 'glpk_exact' umgestellt wird
with iJO1366 as model:
    model.objective = model.problem.Objective(0)
    model_target_constraints= get_leq_constraints(model, iJO1366_mue_targetB)
    model.add_cons_vars(model_target_constraints[0])
    model.reactions.DBTS.knock_out()
    res = model.optimize()
    print(res, res.status)
#%% 
check_mcs(iJO1366, iJO1366_mue_targetB[0], [(iJO1366.reactions.index('DBTS'),)], optlang.interface.INFEASIBLE)

#%% 
model = iJO1366
model.objective = model.problem.Objective(0)
model_target_constraints= get_leq_constraints(model, iJO1366_mue_targetB)
model.add_cons_vars(model_target_constraints[0])
model.reactions.DBTS.knock_out()
res = model.optimize()
print(res, res.status)
#%% misses some cuts
candidates = [r for r, c in zip(iJO1366.reactions.list_attr("id"), cuts) if c]
model = iJO1366
model.objective = model.problem.Objective(0)
# model.solver = 'glpk_exact' # again problems with context
model_target_constraints= get_leq_constraints(model, iJO1366_mue_targetB)
model.add_cons_vars(model_target_constraints[0])
resD = cobra.flux_analysis.single_reaction_deletion(model, reaction_list=candidates,
                                                        processes=1) # no interactive multiprocessing on Windows
single_cuts= list(map(tuple, resD.index[resD.values[:, 1] == optlang.interface.INFEASIBLE]))
print(len(single_cuts)) # 255 ??
print(set(iJO1366_mcs_rxns) - set(single_cuts))
# print(set(single_cuts) == set(mcs for mcs in iJO1366_mcs_rxns if len(mcs) == 1))

#%% OK
candidates = [(c,) for c in numpy.where(cuts)[0]]
res = check_mcs(iJO1366, iJO1366_mue_targetB[0], candidates, optlang.interface.INFEASIBLE)
print(sum(res == True))
m = set(iJO1366_mcs) - set([c for c, b in zip(candidates, res) if b])
print(len(m))

# %%
e = ConstrainedMinimalCutSetsEnumerator(optlang.cplex_interface, iJO1366_stdf.values, [r.reversibility for r in iJO1366.reactions], iJO1366_mue_target,
                                        cuts=cuts, bigM=10000, threshold=0.0001, split_reversible_v=False, irrev_geq=False)
# e.model.objective = e.minimize_sum_over_z
#e.write_lp_file('testM')
# e.model.configuration.verbosity = 3
e.model.configuration.tolerances.feasibility = 1e-9
e.model.configuration.tolerances.optimality = 1e-9
e.model.configuration.tolerances.integrality = 1e-10
iJO1366_mcsB,_ = e.enumerate_mcs(max_mcs_size=1)
print(set(iJO1366_mcs) == set(iJO1366_mcsB), len(iJO1366_mcsB))

# %%
target_rd = [(T@subT, t) for T, t in iJO1366_mue_targetB]
e = ConstrainedMinimalCutSetsEnumerator(optlang.cplex_interface, rd, rev_rd, target_rd, 
                                        threshold=0.1, split_reversible_v=True, irrev_geq=True,
                                        cuts=numpy.any(subT[cuts, :], axis=0))
info = dict()
rd_mcs,_ = e.enumerate_mcs(max_mcs_size=1, enum_method=2, info=info)
print(info)
print(len(rd_mcs))
xsubT= subT.copy()
xsubT[numpy.logical_not(cuts), :] = 0 # only expand to reactions that are repressible 
xmcs = expand_mcs(rd_mcs, xsubT)
set(xmcs) == set(iJO1366_mcs)

#%% OK, a bit slower than CPLEX populate
start_time = time.monotonic()
candidates = [(c,) for c in numpy.where(numpy.any(subT[cuts, :], axis=0))[0]]
res = check_mcs(iJO1366c, target_rd[0], candidates, optlang.interface.INFEASIBLE)
print(time.monotonic() - start_time)
print(sum(res == True))
m = set(rd_mcs) - set([c for c, b in zip(candidates, res) if b])
print(len(m))

#
# %% does not work with bigM out of the box
# e = ConstrainedMinimalCutSetsEnumerator(optlang.glpk_interface, rd, rev_rd, target_rd, 
e = ConstrainedMinimalCutSetsEnumerator(optlang.cplex_interface, rd, rev_rd, target_rd, 
                                        threshold=0.1, bigM=10000, split_reversible_v=True, irrev_geq=True,
                                        cuts=numpy.any(subT[cuts, :], axis=0))
rd_mcs,_ = e.enumerate_mcs(max_mcs_size=1, enum_method=1)

#%% GLPK struggles with numerical instability 
e = ConstrainedMinimalCutSetsEnumerator(optlang.glpk_interface, iJO1366_stdf.values, [r.reversibility for r in iJO1366.reactions], iJO1366_mue_target,
                                        cuts=cuts, bigM=1000, threshold=0.001, split_reversible_v=False, irrev_geq=False)
e.model.objective = e.minimize_sum_over_z
#e.write_lp_file('testM')
e.model.configuration.verbosity = 3
e.model.configuration.tolerances.feasibility = 1e-8
e.model.configuration._iocp.tol_obj= 1e-8
e.model.configuration._iocp.tol_int= 1e-10
iJO1366_mcsB,_ = e.enumerate_mcs(max_mcs_size=1)
print(set(iJO1366_mcs) == set(iJO1366_mcsB), len(iJO1366_mcsB))


# %%
print(len(iJO1366.genes), sum(iJO1366.genes.list_attr('functional')))
# %%
ast = cobra.core.gene.parse_gpr(iJO1366.reactions[436].gene_reaction_rule)
cobra.core.gene.eval_gpr(ast[0], {'b0351'})
cobra.core.gene.eval_gpr(ast[0], ast[1])
#sympy.sympify('b0351 | b1241')
#sympy.logic.boolalg.to_dnf(sympy.sympify('(b4213) & (b0351 | b1241)'))
# %%

# %%
iJO1366 = cobra.io.read_sbml_model(r"..\cnapy-projects\iJO1366\model.sbml")

# %% see how much memory resources GLPK needs...
from swiglpk import *
test_model = iJO1366.copy()
# test_model = ecc2.copy()
test_model.solver = 'glpk_exact'
test_model.objective = test_model.problem.Objective(0.0)
print(test_model.objective)
start_time = time.monotonic()
#print(test_model.solver._run_glp_exact()) # to make the basis
print(test_model.slim_optimize()) # to make the basis
print("Initial simplex", time.monotonic() - start_time) # iJO1366 glp_exact: 56 s, slim_optimize with glp_simplex: 0.1 s
num_models = 1000 # about a 1.1 GB python process (10000 for ECC2); 2.3 GB for 1000 iJO1366
model_collection = [None] * num_models
return_value = [None] * num_models
glpk_status = [None] * num_models
start_time = time.monotonic()
for i in range(num_models):
    model_collection[i] = glp_create_prob()
    glp_copy_prob(model_collection[i], test_model.solver.problem, GLP_OFF)
print("Copied models in", time.monotonic() - start_time)
start_time = time.monotonic()
for i in range(num_models):
    return_value[i] = glp_simplex(model_collection[i], test_model.solver.configuration._smcp)
    return_value[i] = glp_exact(model_collection[i], test_model.solver.configuration._smcp)
    glpk_status[i] = glp_get_status(model_collection[i])
print(time.monotonic() - start_time) # 9 seconds with ECC2, 20 with iJO1366


# %%
