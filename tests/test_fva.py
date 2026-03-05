"""FVA correctness tests on e_coli_core.

Compares four FVA implementations against each other:
  - speedy_fva (compressed)
  - speedy_fva (uncompressed)
  - fva_legacy (brute-force 2*n LPs)
  - cobra flux_variability_analysis

All must agree within tolerance on every reaction bound.
"""
import numpy as np
import pytest
from cobra.io import load_model
from cobra.flux_analysis import flux_variability_analysis as cobra_fva

from straindesign.lptools import fva, fva_legacy, select_solver
from straindesign.speedy_fva import speedy_fva

TOL = 1e-6


@pytest.fixture(scope="module")
def model():
    return load_model("e_coli_core")


@pytest.fixture(scope="module")
def solver(model):
    return select_solver(None, model)


@pytest.fixture(scope="module")
def ref_cobra(model):
    """Cobra's own FVA as ground truth (unconstrained)."""
    df = cobra_fva(model.copy(), fraction_of_optimum=0.0)
    return df


@pytest.fixture(scope="module")
def ref_legacy(model, solver):
    """StrainDesign legacy (brute-force) FVA."""
    return fva_legacy(model.copy(), solver=solver)


@pytest.fixture(scope="module")
def res_compressed(model, solver):
    """speedy_fva with compression."""
    return speedy_fva(model.copy(), solver=solver, compress=True)


@pytest.fixture(scope="module")
def res_uncompressed(model, solver):
    """speedy_fva without compression."""
    return speedy_fva(model.copy(), solver=solver, compress=False)


def _max_err(a, b):
    """Max absolute error across min and max columns, aligning by index."""
    common = a.index.intersection(b.index)
    err_min = np.abs(a.loc[common, "minimum"].values - b.loc[common, "minimum"].values).max()
    err_max = np.abs(a.loc[common, "maximum"].values - b.loc[common, "maximum"].values).max()
    return max(err_min, err_max)


def test_compressed_vs_legacy(res_compressed, ref_legacy):
    assert _max_err(res_compressed, ref_legacy) < TOL


def test_uncompressed_vs_legacy(res_uncompressed, ref_legacy):
    assert _max_err(res_uncompressed, ref_legacy) < TOL


def test_compressed_vs_cobra(res_compressed, ref_cobra):
    assert _max_err(res_compressed, ref_cobra) < TOL


def test_uncompressed_vs_cobra(res_uncompressed, ref_cobra):
    assert _max_err(res_uncompressed, ref_cobra) < TOL


def test_fva_wrapper_matches_speedy(model, solver):
    """fva() wrapper should produce identical results to speedy_fva()."""
    res_wrapper = fva(model.copy(), solver=solver)
    res_direct = speedy_fva(model.copy(), solver=solver)
    assert _max_err(res_wrapper, res_direct) < TOL


def test_all_reactions_present(model, res_compressed):
    """FVA result must contain every reaction in the model."""
    assert set(res_compressed.index) == {r.id for r in model.reactions}
