"""Compression tests: unit tests, backend parity, FVA equivalence, and MCS validation."""
import sys
import pytest
import numpy as np
import warnings
from fractions import Fraction
from os.path import dirname, abspath
from cobra.io import load_model, read_sbml_model
from cobra.flux_analysis import flux_variability_analysis
from sympy import Rational as SympyRational
import straindesign as sd
import straindesign.networktools as nt

warnings.filterwarnings('ignore')

# =============================================================================
# Helpers / fixtures
# =============================================================================


def is_rational_type(value):
    return isinstance(value, (Fraction, SympyRational))


@pytest.fixture
def model_gpr():
    return read_sbml_model(dirname(abspath(__file__)) + r"/model_gpr.xml")


@pytest.fixture
def model_small_example():
    return read_sbml_model(dirname(abspath(__file__)) + r"/model_small_example.xml")


# =============================================================================
# Unit tests (Python-only, no Java required)
# =============================================================================


def test_no_jpype_loaded():
    """Verify that jpype is not loaded when importing straindesign."""
    jpype_before = [m for m in sys.modules if 'jpype' in m.lower()]
    modules_to_remove = [m for m in sys.modules if m.startswith('straindesign')]
    for m in modules_to_remove:
        del sys.modules[m]
    import straindesign
    jpype_after = [m for m in sys.modules if 'jpype' in m.lower()]
    new_jpype = set(jpype_after) - set(jpype_before)
    assert len(new_jpype) == 0, f"straindesign loaded jpype modules: {new_jpype}"


def test_python_compression_basic(model_gpr):
    """Compression reduces reaction count and returns a non-empty map."""
    sd.extend_model_gpr(model_gpr, use_names=False)
    extended_reactions = len(model_gpr.reactions)
    cmp_map = sd.compress_model(model_gpr)
    assert len(model_gpr.reactions) < extended_reactions
    assert len(cmp_map) > 0


def test_python_compression_coupled_function(model_small_example):
    """compress_model_coupled with backend='sparse_rref' returns a dict."""
    nt.stoichmat_coeff2rational(model_small_example)
    nt.remove_conservation_relations(model_small_example)
    reac_map = nt.compress_model_coupled(model_small_example, backend='sparse_rref')
    assert isinstance(reac_map, dict)


def test_compression_coefficient_type(model_small_example):
    """Compression coefficients are exact rational number types."""
    nt.stoichmat_coeff2rational(model_small_example)
    nt.remove_conservation_relations(model_small_example)
    reac_map = nt.compress_model_coupled(model_small_example, backend='sparse_rref')
    for new_reac, old_reacs in reac_map.items():
        for old_reac, coeff in old_reacs.items():
            assert is_rational_type(coeff), (f"Coefficient for {old_reac} in {new_reac}: expected rational, got {type(coeff)}")


def test_stoichmat_coeff2rational_uses_rational_type(model_small_example):
    """stoichmat_coeff2rational converts all coefficients to rational types."""
    nt.stoichmat_coeff2rational(model_small_example)
    for reaction in model_small_example.reactions:
        for metabolite, coeff in reaction._metabolites.items():
            assert is_rational_type(coeff), (f"Coefficient for {metabolite.id} in {reaction.id}: expected rational, got {type(coeff)}")


def test_basic_columns_rat_python():
    """basic_columns_rat returns correct pivot count for a rank-2 matrix."""
    import straindesign.efmtool_cmp_interface as efm
    mx = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 2.0]])
    basic_cols = efm.basic_columns_rat(mx)
    assert len(basic_cols) == 2, f"Expected 2 basic columns, got {len(basic_cols)}"


def test_compression_preserves_flux_space(model_small_example):
    """FBA objective value is unchanged after compression."""
    from straindesign.names import MAXIMIZE
    obj = {r.id: 1 for r in model_small_example.reactions if 'biomass' in r.id.lower() or r.id == 'r_bm'}
    if not obj:
        obj = {model_small_example.reactions[0].id: 1}
    original_value = sd.fba(model_small_example, obj=obj, obj_sense=MAXIMIZE).objective_value
    cmp_map = sd.compress_model(model_small_example)
    for cmp_step in cmp_map:
        for new_reac, old_reacs in cmp_step['reac_map_exp'].items():
            for old_reac in list(obj.keys()):
                if old_reac in old_reacs:
                    obj[new_reac] = obj.pop(old_reac) * float(old_reacs[old_reac])
    compressed_value = sd.fba(model_small_example, obj=obj, obj_sense=MAXIMIZE).objective_value
    assert abs(original_value - compressed_value) < 1e-6, (f"FBA values differ: original={original_value}, compressed={compressed_value}")


@pytest.mark.timeout(30)
def test_full_strain_design_without_java(model_gpr):
    """A full strain design computation completes using only the sparse backend."""
    from straindesign.names import SUPPRESS, ANY
    sd.extend_model_gpr(model_gpr, use_names=False)
    module = sd.SDModule(model_gpr, module_type=SUPPRESS, constraints='r_bm >= 0.1')
    try:
        sd.compute_strain_designs(
            model_gpr,
            sd_modules=[module],
            max_solutions=1,
            max_cost=2,
            compress=True,
            solution_approach=ANY,
        )
    except ImportError as e:
        if 'jpype' in str(e).lower():
            pytest.fail(f"Java/jpype was required but should not be: {e}")
        raise


# =============================================================================
# Backend parity tests (Java required; skipped if jpype unavailable)
# =============================================================================


@pytest.fixture
def jpype_available():
    jpype = pytest.importorskip("jpype", reason="jpype not installed; skipping Java parity tests")
    return jpype


def test_compression_parity_reaction_count(jpype_available):
    """Both backends compress e_coli_core to the same number of reactions."""
    model_py = load_model("e_coli_core")
    nt.compress_model(model_py, backend='sparse_rref')
    model_java = load_model("e_coli_core")
    nt.compress_model(model_java, backend='efmtool_rref')
    assert len(model_py.reactions) == len(
        model_java.reactions), (f"Reaction count mismatch: sparse_rref={len(model_py.reactions)}, efmtool_rref={len(model_java.reactions)}")


def test_fba_equivalence(jpype_available):
    """Both backends produce compressed models with the same optimal FBA value."""
    model_py = load_model("e_coli_core")
    nt.compress_model(model_py, backend='sparse_rref')
    model_java = load_model("e_coli_core")
    nt.compress_model(model_java, backend='efmtool_rref')

    biomass_py = next((r for r in model_py.reactions if 'biomass' in r.id.lower()), None)
    biomass_java = next((r for r in model_java.reactions if 'biomass' in r.id.lower()), None)
    assert biomass_py and biomass_java, "Could not find biomass reaction"

    model_py.objective = biomass_py
    model_java.objective = biomass_java
    val_py = model_py.optimize().objective_value
    val_java = model_java.optimize().objective_value
    assert abs(val_py - val_java) < 1e-6, (f"FBA objective mismatch: sparse_rref={val_py}, efmtool_rref={val_java}")


def test_fva_equivalence(jpype_available):
    """Both backends produce flux spaces with no true FVA mismatches.

    Sign-convention differences (Python = -Java for some lumped reactions) are
    mathematically equivalent and are not counted as mismatches.
    """
    model_py = load_model("e_coli_core")
    nt.compress_model(model_py, backend='sparse_rref')
    model_java = load_model("e_coli_core")
    nt.compress_model(model_java, backend='efmtool_rref')

    fva_py = flux_variability_analysis(model_py, fraction_of_optimum=0.0, processes=1)
    fva_java = flux_variability_analysis(model_java, fraction_of_optimum=0.0, processes=1)

    common = set(fva_py.index) & set(fva_java.index)
    true_mismatches = []
    for r_id in common:
        py_min, py_max = fva_py.loc[r_id, 'minimum'], fva_py.loc[r_id, 'maximum']
        java_min, java_max = fva_java.loc[r_id, 'minimum'], fva_java.loc[r_id, 'maximum']
        direct = abs(py_min - java_min) < 1e-6 and abs(py_max - java_max) < 1e-6
        flipped = abs(py_min - (-java_max)) < 1e-6 and abs(py_max - (-java_min)) < 1e-6
        if not direct and not flipped:
            true_mismatches.append(r_id)

    assert len(true_mismatches) == 0, (f"True FVA mismatches between sparse_rref and efmtool_rref backends: {true_mismatches}")


# =============================================================================
# FVA back-mapping test (sparse only)
# =============================================================================


def test_fva_expansion():
    """Compression map correctly back-maps FVA results to the original reaction space."""
    model_orig = load_model("e_coli_core")
    original_ids = [r.id for r in model_orig.reactions]
    fva_orig = flux_variability_analysis(model_orig, fraction_of_optimum=0.0, processes=1)

    model_cmp = load_model("e_coli_core")
    cmp_map = nt.compress_model(model_cmp, backend='sparse_rref')
    fva_cmp = flux_variability_analysis(model_cmp, fraction_of_optimum=0.0, processes=1)

    # Build inverse map: orig_id -> (compressed_id, coefficient)
    orig_to_cmp = {}
    for step in cmp_map:
        for new_reac, old_reacs in step.get('reac_map_exp', {}).items():
            for old_reac, coeff in old_reacs.items():
                orig_to_cmp[old_reac] = (new_reac, float(coeff))

    true_mismatches = []
    for orig_id in original_ids:
        if orig_id not in orig_to_cmp:
            continue  # zero-flux reaction removed during compression
        comp_id, coeff = orig_to_cmp[orig_id]
        if comp_id not in fva_cmp.index:
            continue
        comp_min = fva_cmp.loc[comp_id, 'minimum']
        comp_max = fva_cmp.loc[comp_id, 'maximum']
        exp_min = coeff * comp_min if coeff >= 0 else coeff * comp_max
        exp_max = coeff * comp_max if coeff >= 0 else coeff * comp_min
        orig_min = fva_orig.loc[orig_id, 'minimum']
        orig_max = fva_orig.loc[orig_id, 'maximum']
        if abs(exp_min - orig_min) > 1e-5 or abs(exp_max - orig_max) > 1e-5:
            true_mismatches.append(orig_id)

    assert len(true_mismatches) == 0, (f"FVA expansion mismatches for reactions: {true_mismatches}")


# =============================================================================
# MCS validation
# =============================================================================


@pytest.mark.parametrize("backend", ["sparse_rref", "efmtool_rref"])
def test_mcs_e_coli_core(backend):
    """MCS computation on e_coli_core returns the expected 455 solutions.

    Parametrized over both compression backends so regressions in either
    are caught. The efmtool_rref variant is skipped when jpype is not installed.

    Requires a strong MILP solver (Gurobi, CPLEX, or SCIP). GLPK cannot
    reliably enumerate all solutions via POPULATE and is excluded.
    """
    if backend == "efmtool_rref":
        pytest.importorskip("jpype", reason="jpype not installed; skipping efmtool backend")
    from straindesign.names import SUPPRESS, POPULATE, GLPK, SCIP, GUROBI, CPLEX
    # Solver priority: SCIP (no size limit) > CPLEX > GUROBI (both have free-tier limits)
    strong_solvers = sd.avail_solvers - {GLPK}
    if not strong_solvers:
        pytest.skip("test_mcs_e_coli_core requires Gurobi, CPLEX, or SCIP (GLPK gives incorrect results)")
    solver = SCIP if SCIP in strong_solvers else next(iter(strong_solvers))
    model = load_model('e_coli_core')
    modules = [sd.SDModule(model, SUPPRESS, constraints='BIOMASS_Ecoli_core_w_GAM >= 0.001')]
    sols = sd.compute_strain_designs(model,
                                     sd_modules=modules,
                                     solution_approach=POPULATE,
                                     max_cost=3,
                                     gene_kos=True,
                                     solver=solver,
                                     backend=backend)
    assert len(sols.reaction_sd) == 455, (f"Expected 455 MCS for e_coli_core (backend={backend}), got {len(sols.reaction_sd)}")


# iML1515 is excluded from CI: free solver tiers reject models of this size.
# Run manually to validate the larger model:
#
# def test_mcs_iml1515():
#     """MCS computation on iML1515 returns the expected 393 solutions."""
#     from straindesign.names import SUPPRESS, POPULATE
#     model = load_model('iML1515')
#     modules = [sd.SDModule(model, SUPPRESS,
#                            constraints='BIOMASS_Ec_iML1515_core_75p37M >= 0.001')]
#     sols = sd.compute_strain_designs(model, sd_modules=modules, solution_approach=POPULATE,
#                                      max_cost=3, gene_kos=True)
#     assert len(sols.reaction_sd) == 393, (
#         f"Expected 393 MCS for iML1515, got {len(sols.reaction_sd)}"
#     )
