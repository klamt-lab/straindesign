"""Quick standalone timing test - no pytest timeout."""
import time
import warnings
warnings.filterwarnings("ignore")

from cobra.io import load_model
import straindesign as sd
from straindesign.names import *

print("Loading e_coli_core...", flush=True)
t0 = time.perf_counter()
m = load_model("e_coli_core")
print(f"  loaded in {time.perf_counter()-t0:.2f}s", flush=True)

for solver in [s for s in [CPLEX, GUROBI] if s in sd.avail_solvers]:
    print(f"\n[{solver}] Computing 455 MCS...", flush=True)
    mc = m.copy()
    t0 = time.perf_counter()
    sol = sd.compute_strain_designs(
        mc,
        sd_modules=[sd.SDModule(mc, SUPPRESS,
                                constraints="BIOMASS_Ecoli_core_w_GAM >= 0.001")],
        solution_approach=POPULATE,
        max_cost=3,
        gene_kos=True,
        solver=solver,
    )
    elapsed = time.perf_counter() - t0
    print(f"  done: {len(sol.reaction_sd)} solutions in {elapsed:.2f}s  status={sol.status}", flush=True)
