"""Test BEST vs POPULATE for e_coli_core 455 MCS."""
import time, warnings
warnings.filterwarnings("ignore")
from cobra.io import load_model
import straindesign as sd
from straindesign.names import *

m = load_model("e_coli_core")

for approach, label in [(POPULATE, "POPULATE"), (BEST, "BEST   ")]:
    for solver in [s for s in [CPLEX, GUROBI] if s in sd.avail_solvers]:
        mc = m.copy()
        t0 = time.perf_counter()
        sol = sd.compute_strain_designs(
            mc,
            sd_modules=[sd.SDModule(mc, SUPPRESS,
                                    constraints="BIOMASS_Ecoli_core_w_GAM >= 0.001")],
            solution_approach=approach,
            max_cost=3,
            gene_kos=True,
            solver=solver,
        )
        elapsed = time.perf_counter() - t0
        print(f"{label} [{solver:6s}]: {elapsed:.2f}s  {len(sol.reaction_sd)} sol  {sol.status}", flush=True)
