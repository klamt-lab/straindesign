"""Find a fast CPLEX seed for e_coli_core 455 MCS."""
import time, warnings
warnings.filterwarnings("ignore")
from cobra.io import load_model
import straindesign as sd
from straindesign.names import *

m = load_model("e_coli_core")

results = {}
for seed in [0, 1, 2, 7, 13, 42, 100, 1234, 9999, 12345]:
    mc = m.copy()
    t0 = time.perf_counter()
    sol = sd.compute_strain_designs(
        mc,
        sd_modules=[sd.SDModule(mc, SUPPRESS,
                                constraints="BIOMASS_Ecoli_core_w_GAM >= 0.001")],
        solution_approach=POPULATE,
        max_cost=3,
        gene_kos=True,
        solver=CPLEX,
        seed=seed,
    )
    elapsed = time.perf_counter() - t0
    results[seed] = elapsed
    print(f"  seed={seed:6d}: {elapsed:.2f}s  {len(sol.reaction_sd)} sol", flush=True)

best = min(results, key=results.get)
print(f"\nBest seed: {best} ({results[best]:.2f}s)")
