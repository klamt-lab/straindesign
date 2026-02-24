"""Debug CPLEX timing."""
import sys, time, warnings
warnings.filterwarnings("ignore")
from cobra.io import load_model
import straindesign as sd
from straindesign.names import *

print(f"avail_solvers: {sd.avail_solvers}", flush=True)
solvers = [s for s in [CPLEX, GUROBI] if s in sd.avail_solvers]
print(f"solvers list: {solvers}", flush=True)

m = load_model("e_coli_core")
print(f"model loaded", flush=True)

for solver in solvers:
    print(f"[{solver}] starting...", flush=True)
    mc = m.copy()
    t0 = time.perf_counter()
    sol = sd.compute_strain_designs(
        mc,
        sd_modules=[sd.SDModule(mc, SUPPRESS,
                                constraints="BIOMASS_Ecoli_core_w_GAM >= 0.001")],
        solution_approach=POPULATE,
        max_cost=3,
        max_solutions=10,
        gene_kos=True,
        solver=solver,
    )
    elapsed = time.perf_counter() - t0
    print(f"[{solver}] done: {elapsed:.2f}s  got={len(sol.reaction_sd)}  status={sol.status}", flush=True)
    sys.stdout.flush()
