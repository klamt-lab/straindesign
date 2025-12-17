"""Test MCS computation."""
from cobra.io import load_model
from straindesign.compression import compress_cobra_model
from straindesign import compute_strain_designs, SDModule, SUPPRESS, POPULATE
import time

# Test e_coli_core (expects 455 MCS)
print("=== e_coli_core ===")
e_coli_core = load_model('e_coli_core')
modules = [SDModule(e_coli_core, SUPPRESS, constraints='BIOMASS_Ecoli_core_w_GAM >= 0.001')]

start = time.time()
sols = compute_strain_designs(e_coli_core, sd_modules=modules, solution_approach=POPULATE, max_cost=3, gene_kos=True)
print(f'Num MCS: {len(sols.reaction_sd)} (expected 455)')
print(f'Time: {time.time() - start:.2f}s')

# Test iML1515 (expects 393 MCS)
print("\n=== iML1515 ===")
iML1515 = load_model('iML1515')
modules = [SDModule(iML1515, SUPPRESS, constraints='BIOMASS_Ec_iML1515_core_75p37M >= 0.001')]

start = time.time()
sols = compute_strain_designs(iML1515, sd_modules=modules, solution_approach=POPULATE, max_cost=3, gene_kos=True)
print(f'Num MCS: {len(sols.reaction_sd)} (expected 393)')
print(f'Time: {time.time() - start:.2f}s')