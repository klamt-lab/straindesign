"""Compression tests: unit tests, map correctness, FVA equivalence, and MCS validation."""
import sys
import pytest
import numpy as np
import warnings
from fractions import Fraction
from os.path import dirname, abspath
from cobra.io import read_sbml_model
from ._models import load_test_model
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
# Unit tests
# =============================================================================


def test_python_compression_basic(model_gpr):
    """Compression reduces reaction count and returns a non-empty map."""
    sd.extend_model_gpr(model_gpr, use_names=False)
    extended_reactions = len(model_gpr.reactions)
    cmp_map = sd.compress_model(model_gpr)
    assert len(model_gpr.reactions) < extended_reactions
    assert len(cmp_map) > 0


def test_python_compression_coupled_function(model_small_example):
    """compress_model_coupled returns a dict."""
    nt.stoichmat_coeff_to_fraction(model_small_example)
    nt.remove_conservation_relations(model_small_example)
    reac_map = nt.compress_model_coupled(model_small_example)
    assert isinstance(reac_map, dict)


def test_compression_coefficient_type(model_small_example):
    """Compression coefficients are exact rational number types."""
    nt.stoichmat_coeff_to_fraction(model_small_example)
    nt.remove_conservation_relations(model_small_example)
    reac_map = nt.compress_model_coupled(model_small_example)
    for new_reac, old_reacs in reac_map.items():
        for old_reac, coeff in old_reacs.items():
            assert is_rational_type(coeff), (f"Coefficient for {old_reac} in {new_reac}: expected rational, got {type(coeff)}")


def test_stoichmat_coeff_to_fraction_uses_rational_type(model_small_example):
    """stoichmat_coeff_to_fraction converts all coefficients to rational types."""
    nt.stoichmat_coeff_to_fraction(model_small_example)
    for reaction in model_small_example.reactions:
        for metabolite, coeff in reaction._metabolites.items():
            assert is_rational_type(coeff), (f"Coefficient for {metabolite.id} in {reaction.id}: expected rational, got {type(coeff)}")


def test_basic_columns_from_numpy():
    """basic_columns_from_numpy returns correct pivot count for a rank-2 matrix."""
    from straindesign.compression import basic_columns_from_numpy
    mx = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 2.0]])
    basic_cols = basic_columns_from_numpy(mx)
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
def test_full_strain_design_compressed(model_gpr):
    """A full strain design computation completes with compression enabled."""
    from straindesign.names import SUPPRESS, ANY
    sd.extend_model_gpr(model_gpr, use_names=False)
    module = sd.SDModule(model_gpr, module_type=SUPPRESS, constraints='r_bm >= 0.1')
    sd.compute_strain_designs(
        model_gpr,
        sd_modules=[module],
        max_solutions=1,
        max_cost=2,
        compress=True,
        solution_approach=ANY,
    )


# =============================================================================
# Compression correctness through the reaction map
# =============================================================================


def _trace_lump(cmp_maps, orig_id):
    """Follow an original reaction through the compression rounds.

    Returns (compressed_id, factor) with orig_flux == factor * compressed_flux.
    """
    cur, factor = orig_id, 1.0
    for rnd in cmp_maps:
        for new_id, members in rnd["reac_map_exp"].items():
            if cur in members:
                factor *= float(members[cur])
                cur = new_id
                break
    return cur, factor


def test_fba_optimum_recovered_through_map():
    """Compression preserves the uncompressed optimum once the lump factor is applied.

    A lump's overall scale is free: only its ratios are fixed, so the raw objective value of a
    lumped reaction is not meaningful on its own. What a caller relies on is the flux recovered
    through the compression map, which must reproduce the uncompressed optimum exactly. This also
    exercises the map itself -- it would catch factors drifting out of step with the column
    scaling applied when a lump is re-expressed in one member's units.
    """
    base = load_test_model("e_coli_core")
    biomass = next((r.id for r in base.reactions if 'biomass' in r.id.lower()), None)
    assert biomass, "Could not find biomass reaction"
    ref = sd.fba(base, obj={biomass: 1}, obj_sense='maximize').objective_value

    model = load_test_model("e_coli_core")
    cmp_maps = nt.compress_model(model)
    cmp_id, factor = _trace_lump(cmp_maps, biomass)
    assert cmp_id in [r.id for r in model.reactions], (f"compression map names {cmp_id}, which is not in the compressed model")
    val = sd.fba(model, obj={cmp_id: 1}, obj_sense='maximize').objective_value
    assert abs(factor * val - ref) < 1e-6, (f"recovered optimum {factor * val} != uncompressed {ref}")


def test_cobra_optimize_after_compression():
    """Cobra's model.optimize() works correctly after compress_model (standalone use).

    This ensures compress_model rebuilds the solver when called outside of
    compute_strain_designs, so cobra's LP interface is not left in a broken state.
    Uses the compression map to back-transform the compressed biomass flux and
    verify it matches the original uncompressed value.
    """
    model_orig = load_test_model("e_coli_core")
    biomass_id = next((r.id for r in model_orig.reactions if 'biomass' in r.id.lower()), None)
    assert biomass_id is not None, "Could not find biomass reaction"
    model_orig.objective = biomass_id
    val_orig = model_orig.optimize().objective_value

    model_cmp = load_test_model("e_coli_core")
    cmp_map = nt.compress_model(model_cmp)

    # Find biomass in compressed model via compression map
    biomass_cmp_id = None
    biomass_coeff = 1.0
    for step in cmp_map:
        for new_reac, old_reacs in step.get('reac_map_exp', {}).items():
            if biomass_id in old_reacs:
                biomass_cmp_id = new_reac
                biomass_coeff = float(old_reacs[biomass_id])
    assert biomass_cmp_id is not None, "Biomass reaction not found in compression map"

    model_cmp.objective = biomass_cmp_id
    sol = model_cmp.optimize()
    assert sol.status == 'optimal', f"Expected optimal solution, got {sol.status}"
    val_expanded = sol.objective_value * biomass_coeff
    assert abs(val_orig - val_expanded) < 1e-6, (f"Expanded objective mismatch: original={val_orig}, expanded={val_expanded}")


# =============================================================================
# FVA back-mapping test (sparse only)
# =============================================================================


def test_fva_expansion():
    """Compression map correctly back-maps FVA results to the original reaction space."""
    model_orig = load_test_model("e_coli_core")
    original_ids = [r.id for r in model_orig.reactions]
    fva_orig = flux_variability_analysis(model_orig, fraction_of_optimum=0.0, processes=1)

    model_cmp = load_test_model("e_coli_core")
    cmp_map = nt.compress_model(model_cmp)
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


def test_mcs_e_coli_core():
    """MCS computation on e_coli_core returns the expected 455 solutions.

    Requires a strong MILP solver (Gurobi, CPLEX, or SCIP). GLPK cannot
    reliably enumerate all solutions via POPULATE and is excluded.
    """
    from straindesign.names import SUPPRESS, POPULATE, GLPK, SCIP, GUROBI, CPLEX
    # Solver priority: SCIP (no size limit) > CPLEX > GUROBI (both have free-tier limits)
    strong_solvers = sd.avail_solvers - {GLPK}
    if not strong_solvers:
        pytest.skip("test_mcs_e_coli_core requires Gurobi, CPLEX, or SCIP (GLPK gives incorrect results)")
    solver = SCIP if SCIP in strong_solvers else next(iter(strong_solvers))
    model = load_test_model('e_coli_core')
    modules = [sd.SDModule(model, SUPPRESS, constraints='BIOMASS_Ecoli_core_w_GAM >= 0.001')]
    sols = sd.compute_strain_designs(
        model,
        sd_modules=modules,
        solution_approach=POPULATE,
        max_cost=3,
        gene_kos=True,
        solver=solver,
    )
    assert len(sols.reaction_sd) == 455, (f"Expected 455 MCS for e_coli_core, got {len(sols.reaction_sd)}")


# =============================================================================
# Exact nullspace: arbitrary-precision input
# =============================================================================


def _kernel_columns(K):
    """Kernel columns as {col: {row: Fraction}}, for either return type."""
    from collections import defaultdict
    cols = defaultdict(dict)
    if isinstance(K, sd.ExactCOO):
        for r, c, v in zip(K.rows, K.cols, K.data):
            cols[int(c)][int(r)] = Fraction(int(v), int(K.denom))
        return cols, K.shape
    A = K.tocoo()
    for r, c, v in zip(A.row, A.col, A.data):
        cols[int(c)][int(r)] = Fraction(int(v))
    return cols, A.shape


def _assert_kernel_exact(entries, shape, K):
    cols, kshape = _kernel_columns(K)
    assert kshape[0] == shape[1]
    rows = {}
    for r, c, v in entries:
        rows.setdefault(r, {})[c] = v
    for j in range(kshape[1]):
        kj = cols[j]
        assert kj, f"kernel column {j} is empty"
        for row in rows.values():
            assert sum(v * kj.get(c, 0) for c, v in row.items()) == 0


def test_rational_matrix_input_stays_int64_when_it_fits():
    entries = [(0, 0, Fraction(1)), (0, 1, Fraction(-2)), (1, 1, Fraction(3)), (1, 2, Fraction(-1))]
    rm = sd.RationalMatrix.from_fractions(entries, (2, 3))
    assert not rm.is_bigint()
    K = sd.sparse_nullspace(rm)
    assert not isinstance(K, sd.ExactCOO)
    _assert_kernel_exact(entries, (2, 3), K)


def test_nullspace_accepts_coefficients_beyond_int64():
    """A coefficient whose exact form needs more than 64 bits must not be rejected.

    Genome-scale models carry 15-significant-digit decimals such as -7.73333333333333e-08,
    whose exact rational has a denominator of 10**22. Clearing denominators across such a row
    exceeds int64, which scipy sparse cannot store.
    """
    tiny = Fraction(-773333333333333, 10**22)
    entries = [(0, 0, tiny), (0, 1, Fraction(-1)), (0, 2, Fraction(1)),
               (1, 1, Fraction(1)), (1, 3, Fraction(-1))]
    rm = sd.RationalMatrix.from_fractions(entries, (2, 4))
    assert rm.is_bigint()
    _assert_kernel_exact(entries, (2, 4), sd.sparse_nullspace(rm))


def test_nullspace_round_trips_its_own_exact_output():
    tiny = Fraction(-773333333333333, 10**22)
    entries = [(0, 0, tiny), (0, 1, Fraction(-1)), (1, 1, Fraction(1)), (1, 2, Fraction(-1))]
    K = sd.sparse_nullspace(sd.RationalMatrix.from_fractions(entries, (2, 3)))
    assert isinstance(K, sd.ExactCOO)
    sd.sparse_nullspace(K)


def test_nullspace_keeps_exact_values_from_object_arrays():
    tiny = Fraction(-773333333333333, 10**22)
    A = np.empty((2, 3), dtype=object)
    A[:] = Fraction(0)
    A[0, 0], A[0, 1] = tiny, Fraction(-1)
    A[1, 1], A[1, 2] = Fraction(1), Fraction(-1)
    entries = [(0, 0, tiny), (0, 1, Fraction(-1)), (1, 1, Fraction(1)), (1, 2, Fraction(-1))]
    _assert_kernel_exact(entries, (2, 3), sd.sparse_nullspace(A))


def test_from_fractions_rejects_out_of_range_entries():
    with pytest.raises(IndexError):
        sd.RationalMatrix.from_fractions([(0, 5, Fraction(1))], (2, 3))


def test_bigint_matrix_reports_unsupported_operations():
    tiny = Fraction(-773333333333333, 10**22)
    rm = sd.RationalMatrix.from_fractions([(0, 0, tiny), (0, 1, Fraction(-1))], (1, 2))
    assert rm.is_bigint()
    with pytest.raises(NotImplementedError):
        rm.clone()


def test_nullspace_falls_back_when_the_scaled_basis_exceeds_int64():
    """The return type must follow what can actually be stored, not just the input's store.

    to_sparse_csr scales the basis by a common denominator, and that product can exceed int64
    even when every individual numerator fits.
    """
    from scipy import sparse as sp
    rng = np.random.default_rng(0)
    A = sp.random(300, 500, density=0.01, random_state=1, format="csr")
    A.data = np.round(A.data, 3)
    K = sd.sparse_nullspace(A)
    assert isinstance(K, sd.ExactCOO)
    assert K.shape[0] == 500


def test_nullspace_of_float_sparse_matches_the_dense_route():
    from scipy import sparse as sp
    from straindesign.compression import RationalMatrix
    A = sp.random(40, 60, density=0.08, random_state=3, format="csr")
    A.data = np.round(A.data, 2)
    viaseq = sd.sparse_nullspace(A)
    viadense = sd.sparse_nullspace(RationalMatrix.from_numpy(A.toarray()))
    to_set = lambda K: ({(int(r), int(c), Fraction(int(v), int(K.denom)))
                         for r, c, v in zip(K.rows, K.cols, K.data)} if isinstance(K, sd.ExactCOO)
                        else {(int(r), int(c), Fraction(int(v)))
                              for r, c, v in zip(*[K.tocoo().row, K.tocoo().col, K.tocoo().data])})
    assert to_set(viaseq) == to_set(viadense)
