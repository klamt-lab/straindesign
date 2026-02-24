"""Time all weak-coupling tests with both solvers."""
import time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from cobra.io import read_sbml_model
import straindesign as sd
from straindesign.names import *
from numpy import inf

TESTS_DIR = Path(__file__).parent
m = read_sbml_model(str(TESTS_DIR / "model_weak_coupling.xml"))

_WEAK_KO = {'r1':1,'r2':1,'r4':1.1,'r5':0.75,'r7':0.8,'r8':1,'r9':1,'r_S':1.0,'r_P':1,'r_BM':1,'r_Q':1.5}
_WEAK_KI = {'r3':0.6,'r6':1.0}
_WEAK_REG = {'r6 >= 4.5':1.2}

tests = [
    ("mcs_wgcp", POPULATE, [
        sd.SDModule(m, SUPPRESS, inner_objective="r_BM", constraints=["r_P - 0.4 r_S <= 0", "r_S >= 0.1"]),
        sd.SDModule(m, PROTECT, constraints=["r_BM >= 0.2"]),
    ], dict(max_cost=4, max_solutions=inf)),
    ("optknock", BEST, [
        sd.SDModule(m, OPTKNOCK, outer_objective="r_P", inner_objective="r_BM", constraints="r_BM >= 1"),
    ], dict(max_cost=4, max_solutions=3)),
    ("robustknock", BEST, [
        sd.SDModule(m, ROBUSTKNOCK, outer_objective="r_P", inner_objective="r_BM", constraints=[{"r_BM":1.0},">=",1.0]),
    ], dict(max_cost=4, max_solutions=3)),
    ("optcouple", BEST, [
        sd.SDModule(m, OPTCOUPLE, prod_id="r_P", inner_objective="r_BM", min_gcp=1.0),
    ], dict(max_cost=6, max_solutions=3)),
]

for solver in [s for s in [CPLEX, GUROBI] if s in sd.avail_solvers]:
    for name, approach, modules, extra in tests:
        mc = m.copy()
        t0 = time.perf_counter()
        sol = sd.compute_strain_designs(
            mc, sd_modules=modules, solution_approach=approach,
            ko_cost=_WEAK_KO, ki_cost=_WEAK_KI, reg_cost=_WEAK_REG,
            solver=solver, **extra,
        )
        elapsed = time.perf_counter() - t0
        print(f"{name:<15} [{solver:6s}]: {elapsed:.2f}s  {len(sol.reaction_sd)} sol  {sol.status}", flush=True)
