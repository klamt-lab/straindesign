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
stdf = cobra.util.array.create_stoichiometric_matrix(ex, array_type='DataFrame')
rev = [r.reversibility for r in ex.reactions]
#target = [(-equations_to_matrix(ex, ["Pyk", "Pck"]), [-1, -1])]
target = [(equations_to_matrix(ex, ["-1 Pyk", "-1 Pck"]), [-1, -1])] # -Pyk does not work
#target.append(target[0]) # duplicate target

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

#%%
flux_lb= -1000*numpy.ones(stdf.values.shape[1])
flux_lb[numpy.logical_not(rev)] = 0
flux_ub= 1000*numpy.ones(stdf.values.shape[1])
desired = [(equations_to_matrix(ex, ["-1 AspCon"]), [-1], flux_lb, flux_ub)]
e = ConstrainedMinimalCutSetsEnumerator(optlang.cplex_interface, stdf.values, rev, target,
                                        threshold=0.1, split_reversible_v=False, irrev_geq=False,
                                        desired=desired)
e.model.objective = e.minimize_sum_over_z
e.write_lp_file('test')
mcs3 = e.enumerate_mcs() #max_mcs_size=5)

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
e.write_lp_file('testI2')
t = time.time()
mcs3 = e.enumerate_mcs(max_mcs_size=5)
print(time.time() - t)

e = ConstrainedMinimalCutSetsEnumerator(optlang.glpk_interface, stdf.values, rev, target, 
                                        cuts=cuts, bigM=100, threshold=0.1, split_reversible_v=True, irrev_geq=False)
e.model.objective = e.minimize_sum_over_z
e.write_lp_file('testI2')
t = time.time()
mcs4 = e.enumerate_mcs(max_mcs_size=5)
print(time.time() - t)

print(set(mcs3) == set([m for m in mcs if 0 not in m and 23 not in m]))
print(set(mcs4) == set(mcs3))

#%%
ecc2 = cobra.io.read_sbml_model(r"..\CNApy\projects\ECC2comp\ECC2comp.xml")
ecc2_stdf = cobra.util.array.create_stoichiometric_matrix(ecc2, array_type='DataFrame')
cuts= numpy.full(ecc2_stdf.shape[1], True, dtype=bool)
for r in ecc2.boundary:
    cuts[ecc2_stdf.columns.get_loc(r.id)] = False
#cuts = None # results do not agree when exchange reactions can be cut, problem with tiny fluxes and M too small
ecc2_mue_target = [(equations_to_matrix(ecc2, 
                    ["-1 Growth", "1 GlcUp", "1 AcUp", "1 GlycUp", "1 SuccUp"]), [-0.01, 10, 10, 10, 10])]
e = ConstrainedMinimalCutSetsEnumerator(optlang.cplex_interface, ecc2_stdf.values, [r.reversibility for r in ecc2.reactions], ecc2_mue_target,
                                        cuts=cuts, threshold=1, split_reversible_v=True, irrev_geq=True)
e.model.objective = e.minimize_sum_over_z
#e.write_lp_file('testI')
#e.model.configuration.verbosity = 3
e.evs_sz_lb = 1 
ecc2_mcs = e.enumerate_mcs(max_mcs_size=3, enum_method=2)
print(len(ecc2_mcs))
print([tuple(ecc2_stdf.columns[r] for r in mcs) for mcs in ecc2_mcs])

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
