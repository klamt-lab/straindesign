"""Time all nested-method weak coupling tests without pytest overhead."""
import time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from cobra.io import read_sbml_model
import straindesign as sd
from straindesign.names import *

TESTS_DIR = Path(__file__).parent
m = read_sbml_model(str(TESTS_DIR / "model_weak_coupling.xml"))

_KO = {'r1':1,'r2':1,'r4':1.1,'r5':0.75,'r7':0.8,'r8':1,'r9':1,'r_S':1.0,'r_P':1,'r_BM':1,'r_Q':1.5}
_KI = {'r3':0.6,'r6':1.0}
_REG = {'r6 >= 4.5':1.2}

for solver in [s for s in [CPLEX, GUROBI] if s in sd.avail_solvers]:
    for name, approach, mk, extra in [
        ("mcs_wgcp",    POPULATE,
         lambda mc: [sd.SDModule(mc, SUPPRESS, inner_objective="r_BM",
                                 constraints=["r_P - 0.4 r_S <= 0", "r_S >= 0.1"]),
                     sd.SDModule(mc, PROTECT, constraints=["r_BM >= 0.2"])],
         dict(max_cost=4, max_solutions=float('inf'))),
        ("optknock",    BEST,
         lambda mc: [sd.SDModule(mc, OPTKNOCK, outer_objective='r_P', inner_objective='r_BM', constraints='r_BM>=1')],
         dict(max_cost=4, max_solutions=3)),
        ("robustknock", BEST,
         lambda mc: [sd.SDModule(mc, ROBUSTKNOCK, outer_objective='r_P', inner_objective='r_BM',
                                 constraints=[[{'r_BM':1.0},'>=',1.0]])],
         dict(max_cost=4, max_solutions=3)),
        ("optcouple",   BEST,
         lambda mc: [sd.SDModule(mc, OPTCOUPLE, prod_id='r_P', inner_objective='r_BM', min_gcp=1.0)],
         dict(max_cost=6, max_solutions=3)),
    ]:
        mc = m.copy()
        modules = mk(mc)
        t0 = time.perf_counter()
        sol = sd.compute_strain_designs(
            mc, sd_modules=modules, solution_approach=approach,
            ko_cost=_KO, ki_cost=_KI, reg_cost=dict(_REG),
            solver=solver, **extra,
        )
        elapsed = time.perf_counter() - t0
        print(f"{name:<15} [{solver:6s}]: {elapsed:.2f}s  {len(sol.reaction_sd)} sol  {sol.status}", flush=True)
