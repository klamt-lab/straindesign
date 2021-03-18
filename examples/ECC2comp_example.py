#%%
import cobra
import cobra.util.array
from optlang_enumerator.cMCS_enumerator import *
import time
import numpy
import os
import sys
import efmtool_link.efmtool_intern as efmtool_intern
import efmtool_link.efmtool4cobra as efmtool4cobra
import pickle

#%%
ecc2 = cobra.io.read_sbml_model(r"..\..\cnapy-projects\ECC2comp\model.sbml")
# ecc2_stdf = cobra.util.array.create_stoichiometric_matrix(ecc2, array_type='DataFrame')
cuts = numpy.array([not r.boundary for r in ecc2.reactions])
reac_id = ecc2.reactions.list_attr('id')
ecc2_mue_target = [[("Growth", ">=", 0.01)]]
ecc2_mue_target = [relations2leq_matrix(parse_relations(t, reac_id_symbols=get_reac_id_symbols(reac_id)), reac_id) for t in ecc2_mue_target]
bounds_mat, bounds_rhs = reaction_bounds_to_leq_matrix(ecc2)
ecc2_mue_target = [(scipy.sparse.vstack((t[0], bounds_mat), format='csr'), numpy.hstack((t[1], bounds_rhs))) for t in ecc2_mue_target]
ecc2_mue_target_constraints= get_leq_constraints(ecc2, ecc2_mue_target)
ecc2_mcs = compute_mcs(ecc2, ecc2_mue_target, cuts=cuts, enum_method=2, max_mcs_size=3, network_compression=True)
print(len(ecc2_mcs))
ecc2_mcs_rxns= [tuple(ecc2_stdf.columns[r] for r in mcs) for mcs in ecc2_mcs]
print(ecc2_mcs_rxns)
ecc2_mcsF = compute_mcs(ecc2, ecc2_mue_target, cuts=cuts, enum_method=2, max_mcs_size=3, network_compression=False)
print(all(check_mcs(ecc2, ecc2_mue_target[0], ecc2_mcs, optlang.interface.INFEASIBLE)))
print(set(ecc2_mcs) == set(ecc2_mcsF))

# %% full FVA
# with ecc2 as model:
model = ecc2.copy() # copy model because switching solver in context sometimes gives an error (?!?)
model.objective = model.problem.Objective(0)
fva_tol = 1e-8 # with CPLEX 1e-8 leads to removal of EX_adp_c, 1e-9 keeps EX_adp_c
model.tolerance = fva_tol # prevent essential EX_meoh_ex from being blocked, sets solver feasibility/optimality tolerances
# model.solver.configuration.tolerances.feasibility = 1e-9
model.solver = 'glpk_exact' # appears to make problems for context management
# model_mue_target_constraints= get_leq_constraints(model, ecc2_mue_target)
# model.add_cons_vars(model_mue_target_constraints[0])
fva_res = cobra.flux_analysis.flux_variability_analysis(model, fraction_of_optimum=0, processes=1) # no interactive multiprocessing on Windows
print(fva_res.loc['EX_adp_c',:])
blocked = []
blocked_rxns = []
for i in range(fva_res.values.shape[0]):
    if fva_res.values[i, 0] >= -fva_tol and fva_res.values[i, 1] <= fva_tol:
        blocked.append(i)
        blocked_rxns.append(fva_res.index[i])
print(blocked_rxns)

#%%
ecc2c = ecc2.copy()
subT = efmtool4cobra.compress_model_sympy(ecc2c, remove_rxns=blocked_rxns)
print(len(ecc2c.metabolites), len(ecc2c.reactions))
rev_rd = [r.reversibility for r in ecc2c.reactions]
efmtool4cobra.remove_conservation_relations_sympy(ecc2c)
reduced = cobra.util.array.create_stoichiometric_matrix(ecc2c, array_type='dok', dtype=numpy.object)
rd = efmtool4cobra.dokRatMat2lilFloatMat(reduced)


# %%
