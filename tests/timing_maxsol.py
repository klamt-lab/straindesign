"""Test CPLEX with limited max_solutions to see where time is spent."""
import time, warnings
warnings.filterwarnings("ignore")
from cobra.io import load_model
import straindesign as sd
from straindesign.names import *

m = load_model("e_coli_core")

for solver in [s for s in [CPLEX, GUROBI] if s in sd.avail_solvers]:
    for n in [1, 10, 50, 100, 455]:
        mc = m.copy()
        t0 = time.perf_counter()
        sol = sd.compute_strain_designs(
            mc,
            sd_modules=[sd.SDModule(mc, SUPPRESS,
                                    constraints="BIOMASS_Ecoli_core_w_GAM >= 0.001")],
            solution_approach=POPULATE,
            max_cost=3,
            max_solutions=n,
            gene_kos=True,
            solver=solver,
        )
        elapsed = time.perf_counter() - t0
        print(f"[{solver:6s}] max_sol={n:4d}: {elapsed:.2f}s  got={len(sol.reaction_sd)}  status={sol.status}", flush=True)
