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

#%%
ex = cobra.io.read_sbml_model(r"metatool_example_no_ext.xml")
ex.solver = 'glpk_exact'
stdf = cobra.util.array.create_stoichiometric_matrix(ex, array_type='DataFrame')
rev = [r.reversibility for r in ex.reactions]
#target = [(-equations_to_matrix(ex, ["Pyk", "Pck"]), [-1, -1])]
target = [(equations_to_matrix(ex, ["-1 Pyk", "-1 Pck"]), [-1, -1])] # -Pyk does not work
#target.append(target[0]) # duplicate target
flux_expr= [r.flux_expression for r in ex.reactions] # !! lose validity when the solver is changed !!

# %%
sol = ex.slim_optimize()
ex.solver.status == optlang.interface.OPTIMAL
# sol = ex.optimize()
#res = cobra.flux_analysis.single_reaction_deletion(ex, processes=1) # no interactive multiprocessing on Windows

# %%
# bei multiplem target falsche Ergebnisse mit enthalten?!?

e = ConstrainedMinimalCutSetsEnumerator(optlang.glpk_interface, stdf.values, rev, target, 
                                        bigM= 100, threshold=0.1, split_reversible_v=True, irrev_geq=True)
e.model.objective = e.minimize_sum_over_z
# e.write_lp_file('testM')
t = time.time()
mcs = e.enumerate_mcs(max_mcs_size=5)
print(time.time() - t)

e = ConstrainedMinimalCutSetsEnumerator(optlang.cplex_interface, stdf.values, rev, target,
                                        threshold=0.1, split_reversible_v=False, irrev_geq=False) #, ref_set=mcs)
e.model.objective = e.minimize_sum_over_z
# e.write_lp_file('testI')
#e.model.configuration.verbosity = 3
#e.model.configuration.presolve = 'auto' # presolve remains off on the CPLEX side
#e.model.configuration.presolve = True # presolve on, this is CPLEX default
#e.model.configuration.lp_method = 'auto' # sets lpmethod back to automatic on the CPLEX side
#e.model.problem.parameters.reset() # works (CPLEX specific)
# without reset() optlang switches presolve off and fixes lpmethod
t = time.time()
mcs2 = e.enumerate_mcs(max_mcs_size=5)
print(time.time() - t)

print(len(set(mcs).intersection(set(mcs2))))
print(set(mcs) == set(mcs2))
all(check_mcs(ex, target[0], mcs, optlang.interface.INFEASIBLE))

# # %% test MCS
# is_cut_set= numpy.zeros(len(mcs), dtype=numpy.bool)
# with ex as model:
#     # tc1 = model.problem.Constraint(model.reactions.get_by_id('Pyk').flux_expression, lb = 1)
#     # tc2 = model.problem.Constraint(model.reactions.get_by_id('Pck').flux_expression, lb = 1)
#     # model.add_cons_vars([tc1, tc2])
#     rexpr = matrix_row_expressions(target[0][0], flux_expr)
#     model.add_cons_vars(leq_constraints(model.problem.Constraint, rexpr, target[0][1]))
#     # model.add_cons_vars([model.problem.Constraint(expr, ub=rhs) for expr, rhs in zip(rexpr, target[0][1])])
#     # tc = [None] * target[0][0].shape[0]
#     # for i in range(target[0][0].shape[0]):
#     #     tc[i] = model.problem.Constraint(rexpr[i], ub=target[0][1][i])
#     # model.add_cons_vars(tc)
#     # #res = cobra.flux_analysis.deletion._multi_deletion(model, 'reaction', [tuple(stdf.columns[r] for r in mcs) for mcs in mcs], processes=1) # geht nicht
#     # res = cobra.flux_analysis.deletion._multi_deletion(model, 'reaction',
#     #  [['Pyk', 'AceEF', 'GltA', 'Icd', 'SucAB'],['Fum', 'Mdh', 'AspC', 'Gdh', 'IlvEAvtA']], processes=1) # macht alle Paare durch...
#     for m in range(len(mcs)):
#         with model as KO_model:
#             for r in mcs[m]:
#                 KO_model.reactions[r].knock_out()
#             KO_model.slim_optimize()
#             is_cut_set[m] = KO_model.solver.status == optlang.interface.INFEASIBLE
# numpy.all(is_cut_set)

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
e.model.objective = e.minimize_sum_over_z
e.write_lp_file('test')
mcs3 = e.enumerate_mcs() #max_mcs_size=5)
print(len(mcs3))
print(all(check_mcs(ex, target[0], mcs3, optlang.interface.INFEASIBLE)))
print(all(check_mcs(ex, desired[0], mcs3, optlang.interface.OPTIMAL)))

#%%
subset_compression = efmtool_link.CompressionMethod[:]([efmtool_link.CompressionMethod.CoupledZero, efmtool_link.CompressionMethod.CoupledCombine, efmtool_link.CompressionMethod.CoupledContradicting])
rd, subT = efmtool_link.compress_rat_efmtool(stdf.values, rev, remove_cr=True,
            compression_method=subset_compression)[0:2]
# rd = stdf.values
# subT = numpy.eye(rd.shape[1])
rev_rd = numpy.logical_not(numpy.any(subT[numpy.logical_not(rev), :], axis=0))
target_rd = [(T@subT, t) for T, t in target]
e = ConstrainedMinimalCutSetsEnumerator(optlang.glpk_interface, rd, rev_rd, target_rd, 
                                        bigM= 100, threshold=0.1, split_reversible_v=True, irrev_geq=True)
e.model.objective = e.minimize_sum_over_z
rd_mcs = e.enumerate_mcs(max_mcs_size=5)
print(len(rd_mcs))
#expand_mcs(rd_mcs, subT) == set(map(lambda x: tuple(numpy.where(x)[0]), mcs2))
expand_mcs(rd_mcs, subT) == set(mcs2)

#%%
e = ConstrainedMinimalCutSetsEnumerator(optlang.cplex_interface, stdf.values, rev, target,
                                        threshold=0.1, split_reversible_v=False, irrev_geq=False) #, ref_set=mcs)
e.model.configuration.verbosity = 3
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
# for r in ecc2.boundary:
#     cuts[ecc2_stdf.columns.get_loc(r.id)] = False
ecc2_mue_target = [(equations_to_matrix(ecc2, 
                    ["-1 Growth", "1 GlcUp", "1 AcUp", "1 GlycUp", "1 SuccUp"]), [-0.01, 10, 10, 10, 10])]
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
e.model.objective = e.minimize_sum_over_z
#e.write_lp_file('testM')
#e.model.configuration.verbosity = 3
ecc2_mcsB = e.enumerate_mcs(max_mcs_size=3)
print(set(ecc2_mcs) == set(ecc2_mcsB), len(ecc2_mcsB))

#%%
subset_compression = efmtool_link.CompressionMethod[:]([efmtool_link.CompressionMethod.CoupledZero, efmtool_link.CompressionMethod.CoupledCombine, efmtool_link.CompressionMethod.CoupledContradicting])
rev = [r.reversibility for r in ecc2.reactions]
rd, subT = efmtool_link.compress_rat_efmtool(ecc2_stdf.values, rev, remove_cr=True,
            compression_method=subset_compression)[0:2]
#%%
print(ecc2_stdf.values.shape) # has 40 extra exchange reactions for the 40 external metabolites in ECC2comp
print(ecc2_stdf.columns[0])
ecc2_stdf.columns[numpy.where(subT[:, 0])]

#%%
rev_rd = numpy.logical_not(numpy.any(subT[numpy.logical_not(rev), :], axis=0))
target_rd = [(T@subT, t) for T, t in ecc2_mue_target]
# e = ConstrainedMinimalCutSetsEnumerator(optlang.cplex_interface, rd, rev_rd, target_rd, 
#                                         threshold=0.1, split_reversible_v=True, irrev_geq=True,
#                                         cuts=numpy.any(subT[cuts, :], axis=0))
e = ConstrainedMinimalCutSetsEnumerator(optlang.glpk_interface, rd, rev_rd, target_rd, 
                                        threshold=0.1, bigM=1000, split_reversible_v=True, irrev_geq=True,
                                        cuts=numpy.any(subT[cuts, :], axis=0))
# here a subset is repressible when one ot its reactions is repressible
e.model.objective = e.minimize_sum_over_z
rd_mcs = e.enumerate_mcs(max_mcs_size=3)
print(len(rd_mcs))
xsubT= subT.copy()
xsubT[numpy.logical_not(cuts), :] = 0 # only expand to reactions that are repressible 
xmcs = expand_mcs(rd_mcs, xsubT)
set(xmcs) == set(ecc2_mcs)

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
conf = scipy.io.loadmat('..\FLB_NB_benchmarks\iJM658_mcs_input', variable_names=input_keys, simplify_cells=True)
for k in input_keys: # put as variables into the workspace for simple access
    exec(k+" = conf['"+k+"']")

# %%
i = idx[37]-1
rd = rd_rat[i]
irrev_rd = irrev_rd_rat[i]
target = [[inh[i], ub[i]]]
desired= [[des[i], db[i], flux_lb[i], flux_ub[i]]]
#%%
e = ConstrainedMinimalCutSetsEnumerator(optlang.cplex_interface, rd, numpy.logical_not(irrev_rd), target,
     cuts=cuts[i], desired=desired, split_reversible_v=True, irrev_geq=True)
e.model.configuration.tolerances.optimality = 1e-6
e.model.configuration.tolerances.feasibility = 1e-6
e.model.configuration.tolerances.integrality = 1e-7
e.model.configuration.verbosity = 3
e.model.problem.parameters.parallel = 1 # set to deterministic for time comparison
e.model.problem.parameters.randomseed = 5
e.evs_sz_lb = 1 
#%%
mcs = e.enumerate_mcs(max_mcs_size=8, enum_method=2)

# %%
