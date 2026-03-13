"""Performance & correctness benchmarks for StrainDesign.

Three tiers, controlled by pytest flags:

  Quick suite (default)
  ─────────────────────
  Tests all major code paths on fast models.  CPLEX and Gurobi compared.

  Observed runtimes (Windows, Python 3.12):
    Gurobi: ~18 s (e_coli_core) +  ~4 s (5 × weak model) ≈  22 s total
    CPLEX:  ~80 s (e_coli_core) + ~180 s (5 × weak model) ≈ 260 s total

  NOTE: On this machine CPLEX processes each LP call with ~2 s overhead
  (vs. <0.01 s for Gurobi).  This accumulates in the link_z preprocessing
  (one LP per knockable reaction) and in CPLEX's POPULATE search, making
  CPLEX ~50-100x slower than Gurobi for these small models.

  e_coli_core tests (large-model correctness / MCS speed):
  1. mcs_455   — gene-level SUPPRESS, POPULATE, max_cost=3.
                 Known answer: 455 solutions.  Primary correctness gate.
                 Timeout: 180 s (CPLEX ~100 s, Gurobi ~18 s).

  model_weak_coupling tests (all nested-opt methods):
  2. mcs_wgcp  — SUPPRESS with inner_objective + PROTECT.
                 Tests Farkas-dual + inner-optimization (POPULATE).
                 Timeout: 180 s (CPLEX ~84 s, Gurobi ~1 s).
  3. optknock  — OPTKNOCK bilevel (BEST, 3 solutions).
                 Tests LP-duality MILP construction.
                 Timeout: 60 s (CPLEX ~26 s, Gurobi ~0.5 s).
  4. robustknock — ROBUSTKNOCK three-level (BEST, 3 solutions).
                 Tests double LP-duality.
                 Timeout: 60 s (CPLEX ~24 s, Gurobi ~0.5 s).
  5. optcouple — OPTCOUPLE two-subproblem (BEST, 2 solutions).
                 Tests GCP objective construction.
                 Timeout: 60 s (CPLEX ~25 s, Gurobi ~0.6 s).

  NOTE: OptKnock / RobustKnock / OptCouple on e_coli_core take >2 minutes
  per solver (the bilinear strong-duality constraint creates hard MILPs; the
  example notebooks use time_limit=300).  Using model_weak_coupling keeps
  the quick suite inside 5 minutes while exercising every MILP path.

  Standard suite (--medium flag, ~4 min)
  ───────────────────────────────────────
  iMLcore genome-scale MCS benchmarks (custom.py scenarios, gene_kos=True).

  Large suite (--large flag, several minutes per solver)
  ───────────────────────────────────────────────────────
  iML1515 gene-level MCS, max_cost=3.  Known answer: 393 solutions.

Usage:
    pytest tests/test_performance.py -v -s --tb=short
    pytest tests/test_performance.py -v -s --tb=short --medium
    pytest tests/test_performance.py -v -s --tb=short --large
    pytest tests/test_performance.py -v -s --tb=short --medium --large

Results written to tests/perf_results/<timestamp>.json.
"""
import json
import time
import pytest
import platform
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path
from os.path import dirname, abspath
from numpy import inf

warnings.filterwarnings("ignore")

from cobra.io import read_sbml_model, load_model
import straindesign as sd
from straindesign.names import *

# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

TESTS_DIR = Path(dirname(abspath(__file__)))
RESULTS_DIR = TESTS_DIR / "perf_results"
RESULTS_DIR.mkdir(exist_ok=True)

_RESULTS: list[dict] = []
_SESSION_TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=TESTS_DIR, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _solver_ver(solver: str) -> str:
    try:
        if solver == CPLEX:
            import cplex
            return cplex.__version__
        if solver == GUROBI:
            import gurobipy as gp
            return ".".join(str(v) for v in gp.gurobi.version())
    except Exception:
        return "n/a"
    return "n/a"


def record(name: str, solver: str, model_id: str,
           elapsed: float, n_sol: int, status: str) -> None:
    entry = {
        "test": name,
        "solver": solver,
        "model": model_id,
        "elapsed_s": round(elapsed, 3),
        "n_solutions": n_sol,
        "status": status,
    }
    _RESULTS.append(entry)
    print(f"\n  [{solver:6s}|{model_id}] {name}: "
          f"{elapsed:.2f} s  →  {n_sol} solution(s)  ({status})")


# ---------------------------------------------------------------------------
# Solver parametrization
# ---------------------------------------------------------------------------

STRONG_SOLVERS = [s for s in [CPLEX, GUROBI] if s in sd.avail_solvers]


# ---------------------------------------------------------------------------
# Model fixtures  (session-scoped → loaded once per run)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def model_core():
    return load_model("e_coli_core")


@pytest.fixture(scope="session")
def model_weak():
    return read_sbml_model(str(TESTS_DIR / "model_weak_coupling.xml"))


@pytest.fixture(scope="session")
def model_imlcore():
    return read_sbml_model(str(TESTS_DIR / "iMLcore.xml"))


# ===========================================================================
# Quick suite — e_coli_core  (MCS correctness + speed, ~8 s / solver)
# ===========================================================================

@pytest.mark.parametrize("solver", STRONG_SOLVERS)
@pytest.mark.timeout(180)
def test_ecoli_core_mcs_455(solver, model_core):
    """Enumerate all gene-level MCS on e_coli_core (max_cost=3).

    Known correct answer: 455 solutions.
    Fails on timeout (30 s) or wrong count.
    """
    m = model_core.copy()
    t0 = time.perf_counter()
    sol = sd.compute_strain_designs(
        m,
        sd_modules=[sd.SDModule(m, SUPPRESS,
                                constraints="BIOMASS_Ecoli_core_w_GAM >= 0.001")],
        solution_approach=POPULATE,
        max_cost=3,
        gene_kos=True,
        solver=solver,
    )
    elapsed = time.perf_counter() - t0
    record("mcs_455", solver, "e_coli_core", elapsed,
           len(sol.reaction_sd), sol.status)
    assert len(sol.reaction_sd) == 455, (
        f"[{solver}] Expected 455 MCS, got {len(sol.reaction_sd)}")


# ===========================================================================
# Quick suite — model_weak_coupling  (nested methods, ~1-5 s / solver each)
#
# model_weak_coupling is a curated toy model (~10 reactions) designed
# to exercise all MILP construction paths quickly and with known results.
# The existing test_05_straindesign.py verifies exact correctness; here we
# measure solver speed and verify ≥ expected_n solutions.
# ===========================================================================

# Shared KO / KI costs for model_weak_coupling  (from test_05_straindesign.py)
_WEAK_KO = {
    'r1': 1, 'r2': 1, 'r4': 1.1, 'r5': 0.75, 'r7': 0.8, 'r8': 1,
    'r9': 1, 'r_S': 1.0, 'r_P': 1, 'r_BM': 1, 'r_Q': 1.5,
}
_WEAK_KI = {'r3': 0.6, 'r6': 1.0}
_WEAK_REG = {'r6 >= 4.5': 1.2}


@pytest.mark.parametrize("solver", STRONG_SOLVERS)
@pytest.mark.timeout(120)
def test_weak_mcs_wgcp(solver, model_weak):
    """MCS with inner optimization (wGCP) on model_weak_coupling.

    SUPPRESS(inner_objective=r_BM): at max growth, low production is infeasible.
    PROTECT: growth ≥ 0.2.
    Tests the Farkas-dual + nested-optimization MILP path.
    Expected: ≥3 solutions, POPULATE, max_cost=4.
    """
    m = model_weak.copy()
    modules = [
        sd.SDModule(m, SUPPRESS,
                    inner_objective="r_BM",
                    constraints=["r_P - 0.4 r_S <= 0", "r_S >= 0.1"]),
        sd.SDModule(m, PROTECT, constraints=["r_BM >= 0.2"]),
    ]
    t0 = time.perf_counter()
    sol = sd.compute_strain_designs(
        m,
        sd_modules=modules,
        solution_approach=POPULATE,
        max_cost=4,
        max_solutions=inf,
        ko_cost=_WEAK_KO,
        ki_cost=_WEAK_KI,
        reg_cost=dict(_WEAK_REG),   # fresh copy: extend_model_regulatory mutates its arg
        solver=solver,
    )
    elapsed = time.perf_counter() - t0
    record("mcs_wgcp", solver, "weak_coupling", elapsed,
           len(sol.reaction_sd), sol.status)
    assert len(sol.reaction_sd) == 3, (
        f"[{solver}] Expected 3 wGCP MCS solutions, got {len(sol.reaction_sd)}")


@pytest.mark.parametrize("solver", STRONG_SOLVERS)
@pytest.mark.timeout(60)
def test_weak_optknock(solver, model_weak):
    """OptKnock on model_weak_coupling: maximize r_P at growth optimum.

    Tests LP-duality bilevel MILP construction.
    Expected: 3 solutions (BEST + iterative, max_solutions=3, max_cost=4).
    """
    m = model_weak.copy()
    modules = [
        sd.SDModule(m, OPTKNOCK,
                    outer_objective="r_P",
                    inner_objective="r_BM",
                    constraints="r_BM >= 1"),
    ]
    t0 = time.perf_counter()
    sol = sd.compute_strain_designs(
        m,
        sd_modules=modules,
        solution_approach=BEST,
        max_cost=4,
        max_solutions=3,
        ko_cost=_WEAK_KO,
        ki_cost=_WEAK_KI,
        reg_cost=dict(_WEAK_REG),   # fresh copy: extend_model_regulatory mutates its arg
        solver=solver,
    )
    elapsed = time.perf_counter() - t0
    record("optknock", solver, "weak_coupling", elapsed,
           len(sol.reaction_sd), sol.status)
    assert len(sol.reaction_sd) == 3, (
        f"[{solver}] Expected 3 OptKnock solutions, got {len(sol.reaction_sd)}")


@pytest.mark.parametrize("solver", STRONG_SOLVERS)
@pytest.mark.timeout(60)
def test_weak_robustknock(solver, model_weak):
    """RobustKnock on model_weak_coupling: guarantee r_P at growth optimum.

    Tests double LP-duality (three-level) MILP construction.
    Expected: ≥2 solutions (BEST, max_solutions=3, max_cost=4).
    """
    m = model_weak.copy()
    modules = [
        sd.SDModule(m, ROBUSTKNOCK,
                    outer_objective="r_P",
                    inner_objective="r_BM",
                    constraints=[[{"r_BM": 1.0}, ">=", 1.0]]),
    ]
    t0 = time.perf_counter()
    sol = sd.compute_strain_designs(
        m,
        sd_modules=modules,
        solution_approach=BEST,
        max_cost=4,
        max_solutions=3,
        ko_cost=_WEAK_KO,
        ki_cost=_WEAK_KI,
        reg_cost=dict(_WEAK_REG),   # fresh copy: extend_model_regulatory mutates its arg
        solver=solver,
    )
    elapsed = time.perf_counter() - t0
    record("robustknock", solver, "weak_coupling", elapsed,
           len(sol.reaction_sd), sol.status)
    assert len(sol.reaction_sd) >= 2, (
        f"[{solver}] Expected ≥2 RobustKnock solutions, got {len(sol.reaction_sd)}")


@pytest.mark.parametrize("solver", STRONG_SOLVERS)
@pytest.mark.timeout(60)
def test_weak_optcouple(solver, model_weak):
    """OptCouple on model_weak_coupling: maximize GCP for r_P.

    Tests two-subproblem GCP MILP construction.
    Expected: 2 solutions (BEST, max_solutions=3, max_cost=6).
    """
    m = model_weak.copy()
    modules = [
        sd.SDModule(m, OPTCOUPLE,
                    prod_id="r_P",
                    inner_objective="r_BM",
                    min_gcp=1.0),
    ]
    t0 = time.perf_counter()
    sol = sd.compute_strain_designs(
        m,
        sd_modules=modules,
        solution_approach=BEST,
        max_cost=6,
        max_solutions=3,
        ko_cost=_WEAK_KO,
        ki_cost=_WEAK_KI,
        reg_cost=dict(_WEAK_REG),   # fresh copy: extend_model_regulatory mutates its arg
        solver=solver,
    )
    elapsed = time.perf_counter() - t0
    record("optcouple", solver, "weak_coupling", elapsed,
           len(sol.reaction_sd), sol.status)
    assert len(sol.reaction_sd) == 2, (
        f"[{solver}] Expected 2 OptCouple solutions, got {len(sol.reaction_sd)}")


# ===========================================================================
# Standard suite (--medium): iMLcore gene-level MCS  (~47 s / solver each)
# ===========================================================================

@pytest.mark.medium
@pytest.mark.parametrize("solver", STRONG_SOLVERS)
@pytest.mark.timeout(90)
def test_imlcore_mcs_ethanol(solver, model_imlcore):
    """Gene-level MCS on iMLcore: suppress ethanol overproduction (custom.py).

    SUPPRESS: EX_etoh_e ≤ 1 AND growth ≥ 0.14
    PROTECT:  growth ≥ 0.15
    max_cost=2, gene_kos=True, POPULATE.  Expected: 4 solutions (~47 s CPLEX).
    """
    m = model_imlcore.copy()
    modules = [
        sd.SDModule(m, SUPPRESS,
                    constraints=["EX_etoh_e <= 1.0",
                                 "BIOMASS_Ec_iML1515_core_75p37M >= 0.14"]),
        sd.SDModule(m, PROTECT,
                    constraints=["BIOMASS_Ec_iML1515_core_75p37M >= 0.15"]),
    ]
    t0 = time.perf_counter()
    sol = sd.compute_strain_designs(
        m,
        sd_modules=modules,
        solution_approach=POPULATE,
        max_cost=2,
        gene_kos=True,
        solver=solver,
    )
    elapsed = time.perf_counter() - t0
    record("imlcore_ethanol", solver, "iMLcore", elapsed,
           len(sol.reaction_sd), sol.status)
    assert len(sol.reaction_sd) > 0, (
        f"[{solver}] Expected ≥1 MCS for iMLcore ethanol scenario, got 0")


@pytest.mark.medium
@pytest.mark.parametrize("solver", STRONG_SOLVERS)
@pytest.mark.timeout(120)
def test_imlcore_mcs_growth(solver, model_imlcore):
    """Gene-level MCS on iMLcore: suppress growth (no protect module).

    SUPPRESS: growth ≥ 0.001
    max_cost=2, gene_kos=True, POPULATE.  Expected: 310 solutions (~70 s CPLEX).
    """
    m = model_imlcore.copy()
    modules = [
        sd.SDModule(m, SUPPRESS,
                    constraints="BIOMASS_Ec_iML1515_core_75p37M >= 0.001"),
    ]
    t0 = time.perf_counter()
    sol = sd.compute_strain_designs(
        m,
        sd_modules=modules,
        solution_approach=POPULATE,
        max_cost=2,
        gene_kos=True,
        solver=solver,
    )
    elapsed = time.perf_counter() - t0
    record("imlcore_growth", solver, "iMLcore", elapsed,
           len(sol.reaction_sd), sol.status)
    assert len(sol.reaction_sd) > 0, (
        f"[{solver}] Expected ≥1 MCS for iMLcore growth scenario, got 0")


# ===========================================================================
# Large suite (--large): iML1515 — known answer 393
# ===========================================================================

@pytest.mark.large
@pytest.mark.parametrize("solver", [GUROBI])  # CPLEX is very slow on this one; skip to save time
def test_iml1515_mcs_393(solver):
    """Enumerate all gene-level MCS on iML1515 (max_cost=3).

    Known correct answer: 393 solutions.
    Only run with --large (takes several minutes per solver).
    """
    try:
        m = load_model("iML1515")
    except Exception:
        pytest.skip("iML1515 not available in this COBRApy installation")
    t0 = time.perf_counter()
    sol = sd.compute_strain_designs(
        m,
        sd_modules=[sd.SDModule(m, SUPPRESS,
                                constraints="BIOMASS_Ec_iML1515_core_75p37M >= 0.001")],
        solution_approach=POPULATE,
        max_cost=3,
        gene_kos=True,
        solver=solver,
    )
    elapsed = time.perf_counter() - t0
    record("iml1515_393", solver, "iML1515", elapsed,
           len(sol.reaction_sd), sol.status)
    assert len(sol.reaction_sd) == 393, (
        f"[{solver}] Expected 393 MCS for iML1515, got {len(sol.reaction_sd)}")


# ===========================================================================
# Large suite (--large): iML1515 + 14BDO — SUPPRESS + PROTECT
# ===========================================================================

def _add_14bdo_pathway(model):
    """Add heterologous 1,4-butanediol pathway to a model (in-place).

    Adds 9 reactions and 7 metabolites (some may already exist in the model).
    Returns the modified model.
    """
    from cobra import Reaction, Metabolite

    # Metabolites (only add if not already present)
    met_ids = {
        'sucsal_c': ('Succinic semialdehyde', 'c'),
        '4hb_c': ('4-Hydroxybutanoate', 'c'),
        '4hbcoa_c': ('4-Hydroxybutyryl-CoA', 'c'),
        '4hbal_c': ('4-Hydroxybutanal', 'c'),
        '14bdo_c': ('1,4-Butanediol', 'c'),
        '14bdo_p': ('1,4-Butanediol', 'p'),
        '14bdo_e': ('1,4-Butanediol', 'e'),
    }
    existing = {m.id for m in model.metabolites}
    for mid, (name, comp) in met_ids.items():
        if mid not in existing:
            model.add_metabolites([Metabolite(mid, name=name, compartment=comp)])

    def _met(mid):
        return model.metabolites.get_by_id(mid)

    # Reactions
    rxns = []

    r = Reaction('SSCOARx'); r.name = 'Succinyl-CoA reductase'
    r.add_metabolites({_met('h_c'): -1, _met('nadph_c'): -1, _met('succoa_c'): -1,
                       _met('coa_c'): 1, _met('nadp_c'): 1, _met('sucsal_c'): 1})
    r.bounds = (0, 1000); r.gene_reaction_rule = 'gsscoar'; rxns.append(r)

    r = Reaction('AKGDC'); r.name = 'Alpha-ketoglutarate decarboxylase'
    r.add_metabolites({_met('akg_c'): -1, _met('h_c'): -1,
                       _met('co2_c'): 1, _met('sucsal_c'): 1})
    r.bounds = (0, 1000); r.gene_reaction_rule = 'gakgdc'; rxns.append(r)

    r = Reaction('4HBD'); r.name = '4-Hydroxybutyrate dehydrogenase'
    r.add_metabolites({_met('h_c'): -1, _met('nadh_c'): -1, _met('sucsal_c'): -1,
                       _met('4hb_c'): 1, _met('nad_c'): 1})
    r.bounds = (0, 1000); rxns.append(r)

    r = Reaction('4HBCT'); r.name = '4-Hydroxybutyrate CoA-transferase'
    r.add_metabolites({_met('4hb_c'): -1, _met('accoa_c'): -1,
                       _met('4hbcoa_c'): 1, _met('ac_c'): 1})
    r.bounds = (0, 1000); rxns.append(r)

    r = Reaction('4HBDH'); r.name = '4-Hydroxybutyryl-CoA dehydrogenase'
    r.add_metabolites({_met('4hbcoa_c'): -1, _met('h_c'): -1, _met('nadh_c'): -1,
                       _met('4hbal_c'): 1, _met('coa_c'): 1, _met('nad_c'): 1})
    r.bounds = (0, 1000); rxns.append(r)

    r = Reaction('4HBDx'); r.name = '4-Hydroxybutanal reductase'
    r.add_metabolites({_met('4hbal_c'): -1, _met('h_c'): -1, _met('nadh_c'): -1,
                       _met('14bdo_c'): 1, _met('nad_c'): 1})
    r.bounds = (0, 1000); rxns.append(r)

    r = Reaction('14BDOtpp'); r.name = '1,4-Butanediol transport (c->p)'
    r.add_metabolites({_met('14bdo_c'): -1, _met('14bdo_p'): 1})
    r.bounds = (0, 1000); rxns.append(r)

    r = Reaction('14BDOtex'); r.name = '1,4-Butanediol transport (p->e)'
    r.add_metabolites({_met('14bdo_p'): -1, _met('14bdo_e'): 1})
    r.bounds = (-1000, 1000); rxns.append(r)

    r = Reaction('EX_14bdo_e'); r.name = '1,4-Butanediol exchange'
    r.add_metabolites({_met('14bdo_e'): -1})
    r.bounds = (0, 1000); rxns.append(r)

    model.add_reactions(rxns)

    # Restrict exchanges: shut all, then open fermentation products
    for rx in model.reactions:
        if all(s < 0 for s in rx.metabolites.values()) and rx.id != 'EX_14bdo_e':
            rx.upper_bound = 0.0
    # Shut CO2 uptake
    model.reactions.get_by_id('EX_co2_e').lower_bound = 0.0
    # Open main fermentation product and utility exchanges
    open_exchanges = [
        'EX_14bdo_e', 'EX_ac_e', 'EX_co2_e', 'EX_etoh_e', 'EX_for_e',
        'EX_h2_e', 'EX_h2o2_e', 'EX_h2o_e', 'EX_h_e', 'EX_lac__D_e',
        'EX_meoh_e', 'EX_o2_e', 'EX_succ_e', 'EX_tungs_e',
        'DM_4crsol_c', 'DM_5drib_c', 'DM_aacald_c', 'DM_amob_c',
        'DM_mththf_c', 'DM_oxam_c',
    ]
    for rid in open_exchanges:
        try:
            model.reactions.get_by_id(rid).upper_bound = 1000.0
        except KeyError:
            pass
    return model


@pytest.mark.large
@pytest.mark.parametrize("solver", [GUROBI])
@pytest.mark.timeout(1200)
def test_iml1515_14bdo_mcs(solver, tmp_path):
    """MCS on iML1515 + 14BDO: SUPPRESS + PROTECT with gene KOs/KIs.

    Primary benchmark for preprocessing + MILP performance experiments.
    Exercises both constraint types in BoundM (Type A from PROTECT, Type B from SUPPRESS).
    """
    try:
        m = load_model("iML1515")
    except Exception:
        pytest.skip("iML1515 not available in this COBRApy installation")

    _add_14bdo_pathway(m)

    # Cost configuration: gene KOs + pathway KIs + reaction KO
    ko_cost = {r.id: 1.0 for r in m.reactions
               if r.genes and m.genes.get_by_id('s0001') not in r.genes}
    ko_cost['EX_o2_e'] = 1.0
    ko_cost.pop('AKGDC', None)
    ko_cost.pop('SSCOARx', None)
    ki_cost = {'AKGDC': 1.0, 'SSCOARx': 1.0}

    module_suppress = sd.SDModule(m, SUPPRESS,
        constraints='EX_14bdo_e + 0.25 EX_glc__D_e <= 0')
    module_protect = sd.SDModule(m, PROTECT,
        constraints='BIOMASS_Ec_iML1515_core_75p37M >= 0.1')

    dump_path = str(tmp_path / 'iml1515_14bdo_preprocessed.pkl')

    t0 = time.perf_counter()
    sol = sd.compute_strain_designs(
        m,
        sd_modules=[module_suppress, module_protect],
        solution_approach=ANY,
        max_cost=40,
        max_solutions=1,
        gene_kos=True,
        ko_cost=ko_cost,
        ki_cost=ki_cost,
        solver=solver,
        dump_preprocessed=dump_path,
    )
    elapsed = time.perf_counter() - t0
    record("iml1515_14bdo", solver, "iML1515+14BDO", elapsed,
           len(sol.reaction_sd), sol.status)
    assert len(sol.reaction_sd) >= 1, (
        f"[{solver}] Expected ≥1 MCS for iML1515+14BDO, got {len(sol.reaction_sd)}")


# ===========================================================================
# Session teardown: write JSON + print comparison table
# ===========================================================================

@pytest.fixture(scope="session", autouse=True)
def _write_results():
    yield
    if not _RESULTS:
        return
    out = {
        "timestamp": _SESSION_TS,
        "git_sha": _git_sha(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "solver_versions": {s: _solver_ver(s) for s in STRONG_SOLVERS},
        "results": _RESULTS,
    }
    out_file = RESULTS_DIR / f"{_SESSION_TS}.json"
    out_file.write_text(json.dumps(out, indent=2))
    print(f"\n\nPerformance results → {out_file}\n")

    solvers_ran = sorted({r["solver"] for r in _RESULTS})
    tests_ran = sorted({(r["test"], r["model"]) for r in _RESULTS})
    col = 16

    header = (f"{'Test':<22} {'Model':<16}"
              + "".join(f"  {s:>{col}}" for s in solvers_ran))
    print("=== Solver Performance Summary ===")
    print(header)
    print("-" * len(header))
    for (test, model) in tests_ran:
        row = {r["solver"]: r for r in _RESULTS
               if r["test"] == test and r["model"] == model}
        line = f"{test:<22} {model:<16}"
        for s in solvers_ran:
            if s in row:
                r = row[s]
                cell = f"{r['elapsed_s']:.2f}s/{r['n_solutions']}sol"
                line += f"  {cell:>{col}}"
            else:
                line += f"  {'—':>{col}}"
        print(line)
