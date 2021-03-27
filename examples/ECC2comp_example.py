#%%
import cobra
import optlang
import optlang_enumerator.cMCS_enumerator as cMCS_enumerator
import numpy
import scipy

#%% 
from importlib import reload
import optlang_enumerator
reload(optlang_enumerator)
import optlang_enumerator.cMCS_enumerator as cMCS_enumerator

#%%
ecc2 = cobra.io.read_sbml_model("ECC2comp.sbml")
ecc2.solver = 'coinor_cbc' # incorrect result with compression and enum_method 3, gets stuck with enum_method 1 ?!?
# allow all reactions that are not boundary reactions as cuts (same as exclude_boundary_reactions_as_cuts option of compute_mcs)
cuts = numpy.array([not r.boundary for r in ecc2.reactions])
reac_id = ecc2.reactions.list_attr('id') # list of reaction IDs in the model
# define target (multiple targets are possible; each target can have multiple linear inequality constraints)
ecc2_mue_target = [[("Growth", ">=", 0.01)]] # one target with one constraint, a.k.a. syntehtic lethals
# this constraint alone would not be sufficient, but there are uptake limits defined in the reaction bounds
# of the model that are integerated automatically by the compute_mcs function into all target and desired regions
# convert into matrix/vector relation format
ecc2_mue_target = [cMCS_enumerator.relations2leq_matrix(
                   cMCS_enumerator.parse_relations(t, reac_id_symbols=cMCS_enumerator.get_reac_id_symbols(reac_id)), reac_id)
                   for t in ecc2_mue_target]
# convert non-default bounds of the newtork model into matrix/vector relation format
# in this network these are substrate uptake bounds
# bounds_mat, bounds_rhs = cMCS_enumerator.reaction_bounds_to_leq_matrix(ecc2)
# integrate the relations defined through the network bounds into every target (still matrix/vector relation format)
# ecc2_mue_target = [(scipy.sparse.vstack((t[0], bounds_mat), format='csr'), numpy.hstack((t[1], bounds_rhs))) for t in ecc2_mue_target]
cMCS_enumerator.integrate_model_bounds(ecc2, ecc2_mue_target)
# convert into constraints that can be added to the COBRApy model (e.g. in context)
ecc2_mue_target_constraints= cMCS_enumerator.get_leq_constraints(ecc2, ecc2_mue_target)
for c in ecc2_mue_target_constraints[0]: # print constraints that make up the first target
    print(c)
#%%
ecc2_mcs = cMCS_enumerator.compute_mcs(ecc2, ecc2_mue_target, cuts=cuts, enum_method=3, max_mcs_size=3, network_compression=True, include_model_bounds=False)
print(len(ecc2_mcs))
# show MCS as n-tuples of reaction IDs
ecc2_mcs_rxns= [tuple(reac_id[r] for r in mcs) for mcs in ecc2_mcs]
print(ecc2_mcs_rxns)
# check that all MCS disable the target
print(all(cMCS_enumerator.check_mcs(ecc2, ecc2_mue_target[0], ecc2_mcs, optlang.interface.INFEASIBLE)))

# %% same calculation without network compression
# works OK with COINOR
ecc2_mcsF = cMCS_enumerator.compute_mcs(ecc2, ecc2_mue_target, cuts=cuts, enum_method=3, max_mcs_size=3, network_compression=False, include_model_bounds=False)
print(set(ecc2_mcs) == set(ecc2_mcsF))

# %%
import pickle
with open("ecc2_mcs.pkl","rb") as f:
    ref = pickle.load(f)
set(ecc2_mcs) - ref

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
import efmtool_link.efmtool4cobra as efmtool4cobra
ecc2c = ecc2.copy()
subT = efmtool4cobra.compress_model_sympy(ecc2c, remove_rxns=blocked_rxns)
print(len(ecc2c.metabolites), len(ecc2c.reactions))
rev_rd = [r.reversibility for r in ecc2c.reactions]
efmtool4cobra.remove_conservation_relations_sympy(ecc2c)
reduced = cobra.util.array.create_stoichiometric_matrix(ecc2c, array_type='dok', dtype=numpy.object)
rd = efmtool4cobra.dokRatMat2lilFloatMat(reduced)


# %%
