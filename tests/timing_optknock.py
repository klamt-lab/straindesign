"""Time optknock on weak coupling model."""
import time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from cobra.io import read_sbml_model
import straindesign as sd
from straindesign.names import *

TESTS_DIR = Path(__file__).parent
m = read_sbml_model(str(TESTS_DIR / "model_weak_coupling.xml"))

_WEAK_KO = {'r1':1,'r2':1,'r4':1.1,'r5':0.75,'r7':0.8,'r8':1,'r9':1,'r_S':1.0,'r_P':1,'r_BM':1,'r_Q':1.5}
_WEAK_KI = {'r3':0.6,'r6':1.0}
_WEAK_REG = {'r6 >= 4.5':1.2}

for solver in [s for s in [CPLEX, GUROBI] if s in sd.avail_solvers]:
    mc = m.copy()
    modules = [sd.SDModule(mc, OPTKNOCK, outer_objective='r_P', inner_objective='r_BM', constraints='r_BM>=1')]
    t0 = time.perf_counter()
    sol = sd.compute_strain_designs(
        mc, sd_modules=modules, solution_approach=BEST,
        max_cost=4, max_solutions=3,
        ko_cost=_WEAK_KO, ki_cost=_WEAK_KI, reg_cost=dict(_WEAK_REG),
        solver=solver,
    )
    elapsed = time.perf_counter() - t0
    print(f"optknock [{solver:6s}]: {elapsed:.2f}s  {len(sol.reaction_sd)} sol  {sol.status}", flush=True)
