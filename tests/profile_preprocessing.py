"""
Micro-benchmarks for model.copy() and create_stoichiometric_matrix(),
plus a cProfile of the full preprocessing pipeline.

Run:  conda run -n straindesign python tests/profile_preprocessing.py
"""
import sys, time, cProfile, pstats, io as _io
import cobra, cobra.io
from cobra.util import create_stoichiometric_matrix
from scipy import sparse

SEP = '=' * 62


def bench(label, fn, reps=3):
    times = [None] * reps
    for i in range(reps):
        t0 = time.perf_counter()
        result = fn()
        times[i] = time.perf_counter() - t0
    avg = sum(times) / reps
    mn  = min(times)
    tag = f"(n={reps})" if reps > 1 else ""
    print(f"  {label:<52s} avg={avg*1e3:7.1f} ms  min={mn*1e3:7.1f} ms  {tag}")
    return result


# ── load models ──────────────────────────────────────────────────────────────
print(SEP)
print("  Loading models")
print(SEP)

models = {}
for name, loader in [
    ('iMLcore',  lambda: cobra.io.read_sbml_model('tests/iMLcore.xml')),
    ('iJO1366',  lambda: cobra.io.load_model('iJO1366')),
    ('iML1515',  lambda: cobra.io.load_model('iML1515')),
]:
    try:
        t0 = time.perf_counter()
        m = loader()
        dt = time.perf_counter() - t0
        models[name] = m
        print(f"  {name:<10s} {len(m.reactions):4d} rxns  {len(m.metabolites):4d} mets"
              f"  (loaded in {dt*1e3:.0f} ms)")
    except Exception as e:
        print(f"  {name:<10s} unavailable: {e}")

if not models:
    print("No models loaded — exiting.")
    sys.exit(1)


# ── micro-benchmarks per model ────────────────────────────────────────────────
for name, model in models.items():
    print()
    print(SEP)
    print(f"  {name}  —  {len(model.reactions)} rxns, {len(model.metabolites)} mets")
    print(SEP)

    # 1. Standard model.copy()
    bench("model.copy()", lambda: model.copy(), reps=3)

    # 2. Empty-solver copy
    def copy_no_solver(m=model):
        orig = None
        try:
            orig = m._solver
            m._solver = m._solver.interface.Model()
        except Exception:
            pass
        c = m.copy()
        if orig is not None:
            m._solver = orig
        return c

    bench("copy_no_solver()", copy_no_solver, reps=3)

    # 3. create_stoichiometric_matrix  (dense → csr)
    bench("create_stoichiometric_matrix() → csr",
          lambda: sparse.csr_matrix(create_stoichiometric_matrix(model)), reps=5)

    # 4. bounds extraction (what fva() also does)
    bench("read lb/ub into lists",
          lambda: ([r.lower_bound for r in model.reactions],
                   [r.upper_bound for r in model.reactions]), reps=5)

    # 5. straindesign FVA
    try:
        from straindesign import fva
        bench("straindesign fva() — gurobi",
              lambda: fva(model, solver='gurobi'), reps=1)
    except Exception as e:
        print(f"  fva(gurobi) failed: {e}")

    try:
        from straindesign import fva
        bench("straindesign fva() — glpk",
              lambda: fva(model, solver='glpk'), reps=1)
    except Exception as e:
        print(f"  fva(glpk) failed: {e}")

    # 6. bound_blocked_or_irrevers_fva  (on a fresh copy each call)
    try:
        from straindesign.networktools import (bound_blocked_or_irrevers_fva,
                                               remove_dummy_bounds)
        mc = model.copy()
        remove_dummy_bounds(mc)
        bench("bound_blocked_or_irrevers_fva() — gurobi",
              lambda: bound_blocked_or_irrevers_fva(mc.copy(), solver='gurobi'), reps=1)
    except Exception as e:
        print(f"  bound_blocked_or_irrevers_fva failed: {e}")

    # 7. compress_model  (sparse_rref)
    try:
        from straindesign.networktools import (compress_model, remove_ext_mets,
                                               remove_dummy_bounds)
        mc_base = model.copy()
        remove_ext_mets(mc_base)
        remove_dummy_bounds(mc_base)
        bench("compress_model() — sparse_rref",
              lambda: compress_model(mc_base.copy(), compression_backend='sparse_rref'), reps=1)
    except Exception as e:
        print(f"  compress_model failed: {e}")


# ── cProfile of compute_strain_designs ───────────────────────────────────────
prof_model_name = next(n for n in ['iML1515', 'iJO1366', 'iMLcore'] if n in models)
prof_model = models[prof_model_name]

# Find objective reaction
obj_rxns = [r.id for r in prof_model.reactions if abs(r.objective_coefficient) > 0]
if not obj_rxns:
    obj_rxns = [prof_model.reactions[0].id]
obj_id = obj_rxns[0]

print()
print(SEP)
print(f"  cProfile — compute_strain_designs")
print(f"  Model: {prof_model_name}  |  module: suppress  {obj_id} >= 0.1")
print(f"  max_cost=1, solver=gurobi, compress=True")
print(SEP)

from straindesign import compute_strain_designs, SDModule
mod = SDModule(prof_model, 'suppress', constraints=f'{obj_id} >= 0.1')

pr = cProfile.Profile()
pr.enable()
try:
    sols = compute_strain_designs(
        prof_model, sd_modules=[mod], max_cost=1,
        solver='gurobi', compress=True)
    print(f"  Result: {len(sols.reaction_sd)} solution(s),  status={sols.status}")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()
pr.disable()

out = _io.StringIO()
ps = pstats.Stats(pr, stream=out).sort_stats('cumulative')
ps.print_stats(40)
print(out.getvalue())
