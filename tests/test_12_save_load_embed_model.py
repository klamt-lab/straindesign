"""Save/load of SDSolutions: no forced expansion, embedded portable model,
opt-in model restore, lazy round-trip (issue #47)."""
import os
import sys
import pickle
from math import inf
import pytest
from cobra.io import load_model, read_sbml_model
import straindesign as sd
import straindesign.compute_strain_designs  # noqa: F401  (ensure submodule imported)
# `straindesign.compute_strain_designs` the attribute is the *function* (star-
# imported into the package), so reach the module via sys.modules.
csd = sys.modules["straindesign.compute_strain_designs"]
from straindesign import SUPPRESS, PROTECT
from straindesign.names import (MODULES, MAX_COST, MAX_SOLUTIONS,
                                SOLUTION_APPROACH, KOCOST, GKOCOST, SOLVER, SEED)

GPR = os.path.join(os.path.dirname(__file__), "model_gpr.xml")
TOL = 1e-6


@pytest.fixture(scope="module")
def model():
    return load_model("textbook")


def _solver():
    for s in ("gurobi", "cplex", "scip", "glpk"):
        if s in sd.avail_solvers:
            return s
    pytest.skip("no MILP solver available")


def _compute(model, threshold=None):
    """Small gene-based MCS: knock out growth (>=0.1) on the textbook model."""
    if threshold is not None:
        old, csd.LAZY_EXPANSION_THRESHOLD = csd.LAZY_EXPANSION_THRESHOLD, threshold
    try:
        return sd.compute_strain_designs(
            model,
            sd_modules=[sd.SDModule(model, SUPPRESS,
                                    constraints="Biomass_Ecoli_core >= 0.1")],
            gene_kos=True, max_cost=3, max_solutions=8,
            solution_approach="any", solver=_solver(), seed=1, compress=True)
    finally:
        if threshold is not None:
            csd.LAZY_EXPANSION_THRESHOLD = old


# ── non-lazy: portable embedded model + opt-in restore ───────────────────
def test_embed_and_restore_roundtrip(model, tmp_path):
    sols = _compute(model)
    ref_rsd = sols.get_reaction_sd()
    ref_gsd = sols.get_gene_sd()
    f = str(tmp_path / "sd.pkl")
    sols.save(f)                                   # embed_model=True by default

    # default load: model NOT rebuilt, but solutions are there
    loaded = sd.SDSolutions.load(f)
    assert loaded._model is None
    assert loaded.get_reaction_sd() == ref_rsd
    assert loaded.get_gene_sd() == ref_gsd

    # opt-in restore: embedded model rebuilt, structure + GPR preserved
    restored = sd.SDSolutions.load(f, model=True, cmp_model=True)
    assert restored._model is not None
    assert len(restored._model.reactions) == len(model.reactions)
    assert len(restored._model.genes) == len(model.genes)
    r = "AKGDH"
    assert (restored._model.reactions.get_by_id(r).gene_reaction_rule
            == model.reactions.get_by_id(r).gene_reaction_rule)


def test_explicit_model_takes_precedence(model, tmp_path):
    sols = _compute(model)
    f = str(tmp_path / "sd.pkl")
    sols.save(f, embed_model=False)                # leaner file, no snapshot
    loaded = sd.SDSolutions.load(f)
    assert loaded._embedded_model_dict is None
    # nothing to restore from, stays model-less
    assert sd.SDSolutions.load(f, model=True, cmp_model=True)._model is None
    # explicit model attaches
    assert sd.SDSolutions.load(f, model=model)._model is model


def test_live_model_and_solver_never_pickled(model, tmp_path):
    sols = _compute(model)
    f = str(tmp_path / "sd.pkl")
    sols.save(f)
    with open(f, "rb") as fh:
        raw = pickle.load(fh)               # must not raise (no live solver in pickle)
    assert raw._model is None
    assert raw._embedded_model_dict is not None


# ── lazy round-trip: save must not force-expand; restore then expand ─────
def test_lazy_save_no_expand_then_restore_and_expand(model, tmp_path):
    sols = _compute(model, threshold=1)            # force lazy expansion
    assert sols.is_lazy
    materialized = sols.get_num_materialized()
    f = str(tmp_path / "lazy.pkl")
    sols.save(f)                                   # must NOT expand_all (no hang)
    assert sols.is_lazy                            # still lazy after save

    # default load: lazy, materialized reps available, expand errors clearly
    loaded = sd.SDSolutions.load(f)
    assert loaded.is_lazy and loaded._model is None
    assert loaded.get_num_materialized() == materialized
    with pytest.raises(RuntimeError, match="model=True"):
        loaded.expand_all()

    # restore model → expansion works and materialises >= the representatives
    restored = sd.SDSolutions.load(f, model=True, cmp_model=True)
    restored.expand_all()
    assert not restored.is_lazy
    assert restored.get_num_materialized() >= materialized
    for s in restored.get_reaction_sd():
        assert isinstance(s, dict) and s


# ── super-extensive reproducibility round-trip on the GPR toy model ──────
#    combined gene + reaction interventions, SUPPRESS/PROTECT validated by FBA
#    on compressed representatives AND after expansion, then the whole
#    computation is REPRODUCED from the artifact alone (embedded model + sd_setup).

def _max_flux(model, rid):
    with model:
        model.objective = rid
        model.objective_direction = "maximize"
        s = model.optimize()
        return s.objective_value if s.status == "optimal" else None


def _apply_design(model, design):
    for k, v in design.items():
        if any(op in str(k) for op in ("<=", ">=", "<", ">", "=")):
            continue  # regulatory intervention, not exercised here
        if v in (-1, -1.0, False):
            if model.genes.has_id(k):
                model.genes.get_by_id(k).knock_out()
            elif model.reactions.has_id(k):
                model.reactions.get_by_id(k).bounds = (0.0, 0.0)


def _validate_designs(reaction_designs):
    """Every design, re-applied to a fresh model, must satisfy PROTECT (r_bm>=1)
    and enforce SUPPRESS (max rd_ex < 1)."""
    assert reaction_designs, "expected at least one design"
    for i, d in enumerate(reaction_designs):
        m = read_sbml_model(GPR)
        _apply_design(m, d)
        bm = _max_flux(m, "r_bm")
        assert bm is not None and bm >= 1.0 - TOL, f"design {i} violates PROTECT: {d}"
        rd = _max_flux(m, "rd_ex")
        assert rd is None or rd < 1.0 + TOL, f"design {i} fails SUPPRESS: {d}"


def _keyset(reaction_designs):
    """Canonical set of designs = set of frozensets of knocked-out reaction ids."""
    return {frozenset(k for k, v in d.items() if v in (-1, -1.0, False))
            for d in reaction_designs}


def _combined_setup(model, solver, approach):
    modules = [sd.SDModule(model, SUPPRESS, constraints=["1.0 rd_ex >= 1.0 "]),
               sd.SDModule(model, PROTECT, constraints=[[{"r_bm": 1.0}, ">=", 1.0]])]
    return {MODULES: modules, MAX_COST: 3, MAX_SOLUTIONS: inf,
            SOLUTION_APPROACH: approach,
            KOCOST: {"rs_up": 1.0, "rd_ex": 1.0, "rp_ex": 1.1},         # reaction KOs
            GKOCOST: {g.id: 1.0 for g in model.genes},                 # gene KOs
            SOLVER: solver, SEED: 7}


@pytest.mark.parametrize("approach", ["any", "best", "populate"])
@pytest.mark.parametrize("force_lazy", [False, True])
def test_full_reproducibility_roundtrip(tmp_path, approach, force_lazy):
    solver = _solver()
    if solver == "glpk" and approach == "populate":
        pytest.skip("GLPK cannot reliably populate")

    model = read_sbml_model(GPR)
    setup = _combined_setup(model, solver, approach)

    old = csd.LAZY_EXPANSION_THRESHOLD
    if force_lazy:
        csd.LAZY_EXPANSION_THRESHOLD = 1
    try:
        sol = sd.compute_strain_designs(model, sd_setup=setup)
    finally:
        csd.LAZY_EXPANSION_THRESHOLD = old

    ref_gene_sd = sol.get_gene_sd()
    ref_keys = _keyset(sol.get_reaction_sd())
    assert ref_gene_sd
    _validate_designs(sol.get_reaction_sd())            # sanity on the fresh result

    # save a self-contained artifact (embed_model=True by default)
    f = str(tmp_path / "repro.pkl")
    sol.save(f)
    assert sol.is_lazy == force_lazy                    # save never expanded a lazy result

    # (A) reload WITHOUT a model: designs preserved; validate the COMPRESSED
    #     representatives directly via SUPPRESS/PROTECT FBA
    loaded = sd.SDSolutions.load(f)
    assert loaded._model is None
    assert loaded.get_gene_sd() == ref_gene_sd
    _validate_designs(loaded.get_reaction_sd())

    # (B) reload WITH the embedded model restored, then EXPAND and validate the
    #     expanded design set via FBA
    restored = sd.SDSolutions.load(f, model=True, cmp_model=True)
    assert restored._model is not None
    restored.expand_all()                               # no-op if already non-lazy
    assert not restored.is_lazy
    _validate_designs(restored.get_reaction_sd())
    expanded_keys = _keyset(restored.get_reaction_sd())   # full design set
    assert expanded_keys >= ref_keys                      # reps ⊆ full set

    # (C) REPRODUCE the computation from the artifact ALONE: the embedded
    #     (uncompressed) model + the stored sd_setup must regenerate the exact
    #     same full design set (also confirms lazy expansion == eager compute).
    reran = sd.compute_strain_designs(restored._model, sd_setup=restored.sd_setup)
    assert _keyset(reran.get_reaction_sd()) == expanded_keys
    _validate_designs(reran.get_reaction_sd())


# ── compressed-model embedding: rational-safe serialisation + restore ────
from fractions import Fraction


def _combined_gpr(model, solver, approach="any"):
    modules = [sd.SDModule(model, SUPPRESS, constraints=["1.0 rd_ex >= 1.0 "]),
               sd.SDModule(model, PROTECT, constraints=[[{"r_bm": 1.0}, ">=", 1.0]])]
    return {MODULES: modules, MAX_COST: 3, SOLUTION_APPROACH: approach,
            KOCOST: {"rs_up": 1.0, "rd_ex": 1.0, "rp_ex": 1.1},
            GKOCOST: {g.id: 1.0 for g in model.genes}, SOLVER: solver, SEED: 7}


def test_networktools_model_dict_rational_exact():
    """model_to_dict/from_dict preserve exact rationals (no float rounding) and
    round-trip +/-inf; a Fraction never crosses through float."""
    from cobra import Model, Metabolite, Reaction
    from straindesign.networktools import model_to_dict, model_from_dict
    import json, math
    m = Model("rat")
    A = Metabolite("A_c", compartment="c"); B = Metabolite("B_c", compartment="c")
    r = Reaction("r1"); m.add_reactions([r])
    r.add_metabolites({A: Fraction(1, 3), B: Fraction(-7, 3)})
    r.lower_bound = Fraction(1, 3); r.upper_bound = float("inf")
    d = model_to_dict(m)
    rt = model_from_dict(json.loads(json.dumps(d)))     # force through JSON primitives
    lb = rt.reactions.r1.lower_bound
    coeff = rt.reactions.r1.metabolites[rt.metabolites.A_c]
    assert isinstance(lb, Fraction) and lb == Fraction(1, 3)          # exact, still rational
    assert isinstance(coeff, Fraction) and coeff == Fraction(1, 3)
    assert float(Fraction(1, 3)) != Fraction(1, 3)                    # sanity: float would differ
    assert math.isinf(rt.reactions.r1.upper_bound)                   # inf round-trips


def test_networktools_model_dict_float_matches_cobra(model):
    """For an ordinary float model the dict is JSON-serialisable and reloads to
    an equivalent working model (schema-compatible with cobra)."""
    from straindesign.networktools import model_to_dict, model_from_dict
    import json
    d = model_to_dict(model)
    json.dumps(d)                                        # must be JSON-clean
    rt = model_from_dict(d)
    assert len(rt.reactions) == len(model.reactions)
    assert len(rt.genes) == len(model.genes)
    assert rt.reactions[0].bounds == model.reactions[0].bounds
    assert abs(rt.slim_optimize() - model.slim_optimize()) < 1e-6


def test_compressed_model_embedded_by_default_restore_optin(tmp_path):
    solver = _solver()
    m = read_sbml_model(GPR)
    sol = sd.compute_strain_designs(m, sd_setup=_combined_gpr(m, solver))
    assert sol._cmp_model is not None and len(sol._cmp_model.reactions) < len(m.reactions)
    cm0 = sol._cmp_model
    f = str(tmp_path / "sd.pkl")
    sol.save(f)                                          # embeds BOTH by default

    # default load restores neither model
    o0 = sd.SDSolutions.load(f)
    assert o0.get_model() is None and o0.get_compressed_model() is None
    assert o0._embedded_cmp_model_dict is not None       # ...but it IS embedded

    # model=True, cmp_model=True rebuild both; compressed model is exact + usable
    o = sd.SDSolutions.load(f, model=True, cmp_model=True)
    assert o.get_model() is not None
    cm = o.get_compressed_model()
    assert cm is not None and len(cm.reactions) == len(cm0.reactions)
    # exactness: every bound/coeff value-equal; every rational stays a Fraction
    for i in range(len(cm0.reactions)):
        r0, r1 = cm0.reactions[i], cm.reactions[i]
        assert r0.lower_bound == r1.lower_bound and r0.upper_bound == r1.upper_bound
        for a in ("lower_bound", "upper_bound"):
            if isinstance(getattr(r0, a), Fraction):
                assert isinstance(getattr(r1, a), Fraction)   # rational not float-ified
        for mt, c0 in r0.metabolites.items():
            c1 = r1.metabolites[cm.metabolites.get_by_id(mt.id)]
            assert c0 == c1 and (not isinstance(c0, Fraction) or isinstance(c1, Fraction))
    assert any(isinstance(b, Fraction) for r in cm.reactions for b in (r.lower_bound, r.upper_bound))
    cm.objective = cm.reactions[0].id
    assert cm.optimize().status == "optimal"             # a fully working model

    # explicit overrides take precedence
    assert sd.SDSolutions.load(f, cmp_model=cm0).get_compressed_model() is cm0
    # embed_model=False -> no compressed snapshot
    f2 = str(tmp_path / "lean.pkl"); sol.save(f2, embed_model=False)
    assert sd.SDSolutions.load(f2, model=True, cmp_model=True).get_compressed_model() is None


def test_compressed_solutions_analyzable_in_restored_cmp_model(tmp_path):
    """The compressed solutions reference reactions of the (restored) compressed
    model, so they can be analysed there directly (the fast path)."""
    solver = _solver()
    m = read_sbml_model(GPR)
    sol = sd.compute_strain_designs(m, sd_setup=_combined_gpr(m, solver))
    f = str(tmp_path / "sd.pkl"); sol.save(f)
    o = sd.SDSolutions.load(f, model=True, cmp_model=True)
    cm = o.get_compressed_model()
    cm_rxn_ids = {r.id for r in cm.reactions}
    non_empty = [cs for cs in o.compressed_sd if cs]
    assert non_empty, "expected at least one non-empty compressed solution"
    for cs in non_empty:
        assert set(cs).issubset(cm_rxn_ids)              # analysable in the compressed model
    # apply one compressed KO set and confirm FBA still runs in the small model
    cs = non_empty[0]
    with cm:
        for rid, v in cs.items():
            if v in (-1, -1.0, False):
                cm.reactions.get_by_id(rid).bounds = (0, 0)
        cm.objective = cm.reactions[0].id
        assert cm.optimize().status in ("optimal", "infeasible")


def test_networktools_preserves_objective_direction():
    """networktools serializers keep the objective SENSE, which cobra drops."""
    from cobra import Model, Metabolite, Reaction
    from straindesign.networktools import model_to_dict, model_from_dict
    m = Model("obj"); A = Metabolite("A_c", compartment="c")
    r = Reaction("r1", lower_bound=0, upper_bound=10); m.add_reactions([r])
    r.add_metabolites({A: 1.0}); m.objective = "r1"; m.objective.direction = "min"
    rt = model_from_dict(model_to_dict(m))
    assert rt.objective.direction == "min"
