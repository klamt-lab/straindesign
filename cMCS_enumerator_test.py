# exec(open('optlang_mcs_test5.py').read())
#%%
import cobra
import cobra.util.array
from cMCS_enumerator import *
import time
import numpy
import os
import sys
sys.path.append(os.path.join('..', 'efmtool_link'))
import efmtool_link
import efmtool4cobra

#%%
ex = cobra.io.read_sbml_model(r"metatool_example_no_ext.xml")
ex.solver = 'glpk_exact'
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
kn = efmtool_link.null_rat_efmtool(stdf.values)
#res = cobra.flux_analysis.single_reaction_deletion(ex, processes=1) # no interactive multiprocessing on Windows
    
# %%
info = dict()
e = ConstrainedMinimalCutSetsEnumerator(optlang.glpk_interface, stdf.values, rev, target, 
                                        bigM= 100, threshold=0.1, split_reversible_v=True, irrev_geq=True)
# e.model.objective = e.minimize_sum_over_z
# e.write_lp_file('testM')
mcs = e.enumerate_mcs(max_mcs_size=5, info=info)
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
mcs2 = e.enumerate_mcs(max_mcs_size=5, info=info)
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
mcs3 = e.enumerate_mcs(max_mcs_size=5, enum_method=3, model=ex, targets=target, info=info)
print(info)
print(e.evs_sz_lb)
print(set(mcs) == set(mcs3))

#%%
e = ConstrainedMinimalCutSetsEnumerator(optlang.glpk_interface, stdf.values, rev, target, 
                                        bigM= 100, threshold=0.1, split_reversible_v=True, irrev_geq=True)
info = dict()
e.model.configuration._iocp.mip_gap = 0.99
mcs3 = e.enumerate_mcs(max_mcs_size=5, enum_method=3, model=ex, targets=target, info=info)
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
e = ConstrainedMinimalCutSetsEnumerator(optlang.cplex_interface, stdf.values, rev, target,
                                        threshold=0.1, split_reversible_v=False, irrev_geq=False,
                                        desired=desired)
# e.model.objective = e.minimize_sum_over_z
# e.write_lp_file('test')
mcs3 = e.enumerate_mcs() #max_mcs_size=5)
print(len(mcs3))
print(all(check_mcs(ex, target[0], mcs3, optlang.interface.INFEASIBLE)))
print(all(check_mcs(ex, desired[0], mcs3, optlang.interface.OPTIMAL)))

#%%
# subset_compression = efmtool_link.CompressionMethod[:]([efmtool_link.CompressionMethod.CoupledZero, efmtool_link.CompressionMethod.CoupledCombine, efmtool_link.CompressionMethod.CoupledContradicting])
rd, subT, comprec = efmtool_link.compress_rat_efmtool(stdf.values, rev, remove_cr=True,
            compression_method= efmtool_link.subset_compression) #[0:2]
# rd = stdf.values
# subT = numpy.eye(rd.shape[1])
rev_rd = numpy.logical_not(numpy.any(subT[numpy.logical_not(rev), :], axis=0))
#%%
# exc, subT = efmtool4cobra.compress_model(ex)
exc, subT = efmtool4cobra.compress_model_sympy(ex)
rd = cobra.util.array.create_stoichiometric_matrix(exc, array_type='dok') #, dtype=numpy.object) # does actually work with rationals
rev_rd = [r.reversibility for r in exc.reactions]

#%%
target_rd = [(T@subT, t) for T, t in target]
# e = ConstrainedMinimalCutSetsEnumerator(optlang.glpk_interface, rd, rev_rd, target_rd, 
#                                         bigM= 100, threshold=0.1, split_reversible_v=True, irrev_geq=True)
e = ConstrainedMinimalCutSetsEnumerator(optlang.glpk_interface, rd, rev_rd, target_rd, 
                                        bigM= 100, threshold=0.1, split_reversible_v=True, kn=efmtool_link.null_rat_efmtool(rd))
# e.model.objective = e.minimize_sum_over_z
e.model.configuration._iocp.mip_gap = 0.99
rd_mcs = e.enumerate_mcs(max_mcs_size=5, enum_method=3, model=exc, targets=target_rd)
# rd_mcs = e.enumerate_mcs(max_mcs_size=5)
print(len(rd_mcs))
#expand_mcs(rd_mcs, subT) == set(map(lambda x: tuple(numpy.where(x)[0]), mcs2))
expand_mcs(rd_mcs, subT) == set(mcs2)

#%%
e = ConstrainedMinimalCutSetsEnumerator(optlang.cplex_interface, stdf.values, rev, target,
                                        threshold=0.1, split_reversible_v=False, irrev_geq=False) #, ref_set=mcs)
# e.model.configuration.verbosity = 3
e.evs_sz_lb = 1
mcs2p = e.enumerate_mcs(max_mcs_size=5, enum_method=2)
print(set(mcs) == set(mcs2p))

#%%
cuts = numpy.full(24, True, dtype=bool)
cuts[0] = False
cuts[23] = False
e = ConstrainedMinimalCutSetsEnumerator(optlang.cplex_interface, stdf.values, rev, target, 
                                        cuts=cuts, threshold=0.1, split_reversible_v=True, irrev_geq=True)
e.model.objective = e.minimize_sum_over_z
# e.write_lp_file('testI2')
t = time.time()
mcs3 = e.enumerate_mcs(max_mcs_size=5)
print(time.time() - t)

e = ConstrainedMinimalCutSetsEnumerator(optlang.glpk_interface, stdf.values, rev, target, 
                                        cuts=cuts, bigM=100, threshold=0.1, split_reversible_v=True, irrev_geq=False)
e.model.objective = e.minimize_sum_over_z
# e.write_lp_file('testI2')
t = time.time()
mcs4 = e.enumerate_mcs(max_mcs_size=5)
print(time.time() - t)

print(set(mcs3) == set([m for m in mcs if 0 not in m and 23 not in m]))
print(set(mcs4) == set(mcs3))

#%%
ecc2 = cobra.io.read_sbml_model(r"..\CNApy\projects\ECC2comp\ECC2comp.xml")
ecc2_stdf = cobra.util.array.create_stoichiometric_matrix(ecc2, array_type='DataFrame')
cuts= numpy.full(ecc2_stdf.shape[1], True, dtype=bool) # results do not agree when exchange reactions can be cut, problem with tiny fluxes (and M too small)
for r in ecc2.boundary:
    cuts[ecc2_stdf.columns.get_loc(r.id)] = False
reac_id = ecc2_stdf.columns.tolist()
reac_id_symbols = get_reac_id_symbols(reac_id)
ecc2_mue_target = [[("Growth", ">=", 0.01), ("GlcUp", "<=", 10), ("AcUp", "<=", 10), ("GlycUp", "<=", 10), ("SuccUp", "<=", 10)]]
ecc2_mue_target = [relations2leq_matrix(parse_relations(t, reac_id_symbols=reac_id_symbols), reac_id) for t in ecc2_mue_target]
# ecc2_mue_target = [(equations_to_matrix(ecc2, 
#                     ["-1 Growth", "1 GlcUp", "1 AcUp", "1 GlycUp", "1 SuccUp"]), [-0.01, 10, 10, 10, 10])]
e = ConstrainedMinimalCutSetsEnumerator(optlang.cplex_interface, ecc2_stdf.values, [r.reversibility for r in ecc2.reactions], ecc2_mue_target,
                                        cuts=cuts, threshold=1, split_reversible_v=True, irrev_geq=True)
# e.model.objective = e.minimize_sum_over_z
e.model.configuration.tolerances.feasibility = 1e-9
e.model.configuration.tolerances.optimality = 1e-9
e.model.configuration.tolerances.integrality = 1e-10
#e.write_lp_file('testI')
#e.model.configuration.verbosity = 3
e.evs_sz_lb = 1 
ecc2_mcs = e.enumerate_mcs(max_mcs_size=3, enum_method=2)
print(len(ecc2_mcs))
ecc2_mcs_rxns= [tuple(ecc2_stdf.columns[r] for r in mcs) for mcs in ecc2_mcs]
print(ecc2_mcs_rxns)
all(check_mcs(ecc2, ecc2_mue_target[0], ecc2_mcs, optlang.interface.INFEASIBLE))

# %% single cuts via LP
ecc2.solver = 'glpk_exact'
# ecc2.solver = 'glpk'
# ecc2.solver = 'cplex'
with ecc2 as model:
    model.objective = model.problem.Objective(0)
    #print(model.objective.expression)
    flux_expr= [r.flux_expression for r in model.reactions] # !! lose validity when the solver is changed !!
    rexpr = matrix_row_expressions(ecc2_mue_target[0][0], flux_expr)
    model.add_cons_vars(leq_constraints(model.problem.Constraint, rexpr, ecc2_mue_target[0][1]))
    res = cobra.flux_analysis.single_reaction_deletion(model, reaction_list=ecc2_stdf.columns[cuts], processes=1) # no interactive multiprocessing on Windows
single_cuts= list(map(tuple, res.index[res.values[:, 1] == optlang.interface.INFEASIBLE]))
print(set(single_cuts) == set(mcs for mcs in ecc2_mcs_rxns if len(mcs) == 1))
print(set(single_cuts) - set(mcs for mcs in ecc2_mcs_rxns if len(mcs) == 1))
check_mcs(ecc2, ecc2_mue_target[0], single_cuts, optlang.interface.INFEASIBLE)

#%% FVA mit verschiedenen solvern machen um zu sehen ob {'EX_adp_c'} ein cut set ist
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
ecc2_mcsB = e.enumerate_mcs(max_mcs_size=3, enum_method=3, model=ecc2, targets=ecc2_mue_target)
print(set(ecc2_mcs) == set(ecc2_mcsB), len(ecc2_mcsB))

#%%
# not for enum_method 3 because this requires as compressed model for the minimality checks
rev = [r.reversibility for r in ecc2.reactions]
rd, subT = efmtool_link.compress_rat_efmtool(ecc2_stdf.values, rev, remove_cr=True, # if CR are not removed problem with enumeration
            compression_method=efmtool_link.subset_compression)[0:2]
rev_rd = numpy.logical_not(numpy.any(subT[numpy.logical_not(rev), :], axis=0))
kn = efmtool_link.null_rat_efmtool(rd)
print(rd.shape)
print(numpy.linalg.matrix_rank(rd))
print(kn.shape)
print(numpy.max(abs(rd@kn)))

#%%
ecc2c, subT = efmtool4cobra.compress_model(ecc2)
rd = cobra.util.array.create_stoichiometric_matrix(ecc2c, array_type='dok')
bc = efmtool_link.basic_columns_rat(rd.transpose().toarray(), tolerance=1e-10) # needs non-zero tolerance
rd = rd[numpy.sort(bc), :] # misses one CR
kn = efmtool_link.null_rat_efmtool(rd)
print(rd.shape)
print(numpy.linalg.matrix_rank(rd.toarray()))
print(kn.shape)
print(numpy.max(abs(rd@kn)))

#%%
import scipy.sparse
ecc2c, subT = efmtool4cobra.compress_model_sympy(ecc2)
rev_rd = [r.reversibility for r in ecc2c.reactions]
reduced, bc = efmtool4cobra.remove_conservation_relations_sympy(ecc2c)[0:2]
rdb = scipy.sparse.dok_matrix((reduced.shape[0], reduced.shape[1]))
for (r, c), v in reduced.items():
    rdb[r, c] = float(v)
rd = rdb
for m in [ecc2c.metabolites[i].id for i in set(range(len(ecc2c.metabolites))) - set(bc)]:
    ecc2c.metabolites.get_by_id(m).remove_from_model()
#%%
# kn = efmtool_link.null_rat_efmtool(rd)
import ch.javasoft.smx.ops.Gauss as Gauss
jkn = Gauss.getRationalInstance().nullspace(efmtool4cobra.sympyRatMat2jRatMat(reduced))
kn = efmtool_link.jpypeArrayOfArrays2numpy_mat(jkn.getDoubleRows())
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

e = ConstrainedMinimalCutSetsEnumerator(optlang.glpk_interface, rd, rev_rd, target_rd, 
                                        threshold=0.1, bigM=1000, split_reversible_v=True, #irrev_geq=True,
                                        cuts=numpy.any(subT[cuts, :], axis=0), kn=kn) #efmtool_link.null_rat_efmtool(rd))
e.model.configuration._iocp.mip_gap = 0.99 # kann nicht benutzt werden solange efmtool4cobra.compress_model nicht läuft

# here a subset is repressible when one ot its reactions is repressible
# e.model.objective = e.minimize_sum_over_z
info = dict()
# rd_mcs = e.enumerate_mcs(max_mcs_size=3, info=info)
# rd_mcs = e.enumerate_mcs(max_mcs_size=3, enum_method=3, model=ecc2c, targets=target_rd, info=info)
with ecc2c as tm:
    tm.solver = 'glpk' #'glpk_exact' # actually optlang runs regular GLPK first and only if the results is optimal glpk_exact
    # tm.solver.configuration.verbosity = 3
    rd_mcs = e.enumerate_mcs(max_mcs_size=3, enum_method=3, model=ecc2c, targets=target_rd, info=info)
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
iJO1366 = cobra.io.read_sbml_model(r"..\CNApy\projects\iJO1366\iJO1366.xml")
iJO1366_stdf = cobra.util.array.create_stoichiometric_matrix(iJO1366, array_type='DataFrame')
cuts= numpy.full(iJO1366_stdf.shape[1], True, dtype=bool)
for r in iJO1366.boundary:
    cuts[iJO1366_stdf.columns.get_loc(r.id)] = False
#cuts = None
iJO1366_mue_target = [(equations_to_matrix(iJO1366, 
                    ["-1 BIOMASS_Ec_iJO1366_core_53p95M", "-1 EX_glc__D_e"]), [-0.01, 10])]
e = ConstrainedMinimalCutSetsEnumerator(optlang.cplex_interface, iJO1366_stdf.values, [r.reversibility for r in iJO1366.reactions], iJO1366_mue_target,
                                        cuts=cuts, threshold=1, split_reversible_v=False, irrev_geq=False)
e.model.objective = e.minimize_sum_over_z
#e.write_lp_file('testI')
# e.model.configuration.verbosity = 3
iJO1366_mcs = e.enumerate_mcs(max_mcs_size=1)
print(len(iJO1366_mcs))
print([[r for r, c in zip(iJO1366_stdf.columns, mcs) if c == 1] for mcs in iJO1366_mcs])


e = ConstrainedMinimalCutSetsEnumerator(optlang.cplex_interface, iJO1366_stdf.values, [r.reversibility for r in iJO1366.reactions], iJO1366_mue_target,
                                        cuts=cuts, bigM=10000, threshold=0.0001, split_reversible_v=False, irrev_geq=False)
e.model.objective = e.minimize_sum_over_z
#e.write_lp_file('testM')
# e.model.configuration.verbosity = 3
e.model.configuration.tolerances.feasibility = 1e-9
e.model.configuration.tolerances.optimality = 1e-9
e.model.configuration.tolerances.integrality = 1e-10
iJO1366_mcsB = e.enumerate_mcs(max_mcs_size=1)
print(set(iJO1366_mcs) == set(iJO1366_mcsB), len(iJO1366_mcsB))

#%% GLPK struggles with numerical instability 
e = ConstrainedMinimalCutSetsEnumerator(optlang.glpk_interface, iJO1366_stdf.values, [r.reversibility for r in iJO1366.reactions], iJO1366_mue_target,
                                        cuts=cuts, bigM=1000, threshold=0.001, split_reversible_v=False, irrev_geq=False)
e.model.objective = e.minimize_sum_over_z
#e.write_lp_file('testM')
e.model.configuration.verbosity = 3
e.model.configuration.tolerances.feasibility = 1e-8
e.model.configuration._iocp.tol_obj= 1e-8
e.model.configuration._iocp.tol_int= 1e-10
iJO1366_mcsB = e.enumerate_mcs(max_mcs_size=1)
print(set(iJO1366_mcs) == set(iJO1366_mcsB), len(iJO1366_mcsB))

# %%
import scipy
input_keys = ['rd_rat', 'irrev_rd_rat', 'flux_lb', 'flux_ub', 'cuts', 'kn', 'idx', 'inh', 'ub', 'des', 'db']
conf = scipy.io.loadmat(os.path.join('..', 'FLB_NB_benchmarks', 'iJM658_mcs_input'), variable_names=input_keys, simplify_cells=True)
for k in input_keys: # put as variables into the workspace for simple access
    exec(k+" = conf['"+k+"']")

# %%
i = idx[37]-1
rd = rd_rat[i]
irrev_rd = irrev_rd_rat[i]
target = [[inh[i], ub[i]]]
desired= [[des[i], db[i], flux_lb[i], flux_ub[i]]]
#%%
# e = ConstrainedMinimalCutSetsEnumerator(optlang.glpk_interface, rd, numpy.logical_not(irrev_rd), target,
#      cuts=cuts[i], desired=desired, split_reversible_v=True, irrev_geq=False, kn=kn[i], threshold=0.1, bigM=1000)
e = ConstrainedMinimalCutSetsEnumerator(optlang.cplex_interface, rd, numpy.logical_not(irrev_rd), target,
     cuts=cuts[i], desired=desired, split_reversible_v=True, irrev_geq=False, kn=kn[i]) #, threshold=0.1, bigM=1000)
#     cuts=cuts[i], desired=desired, split_reversible_v=True, irrev_geq=True) #, threshold=0.1, bigM=1000)
e.model.problem.parameters.emphasis.numerical.set(1) # is not listed as changed parameter by CPLEX
e.model.problem.parameters.parallel.set(1) # set to deterministic for time comparison
e.model.problem.parameters.randomseed.set(5)
e.model.configuration.tolerances.optimality = 1e-6
e.model.configuration.tolerances.feasibility = 1e-6
e.model.configuration.tolerances.integrality = 1e-7
e.model.configuration.verbosity = 3
e.evs_sz_lb = 1 
#e.write_lp_file('testOL')
#e.model.problem.parameters.get_changed()
#%%
info = dict()
mcs = e.enumerate_mcs(max_mcs_size=8, enum_method=2, info=info) #, timeout=30)
print(info)

# %%
res = scipy.io.loadmat(os.path.join('..', 'FLB_NB_benchmarks', 'iJM658_mcs_input_s5_255_255'), simplify_cells=True)
set(mcs) == set([tuple(numpy.nonzero(res['mcs'][i][:, j])[0]) for j in range(res['mcs'][i].shape[1])])

# %%
