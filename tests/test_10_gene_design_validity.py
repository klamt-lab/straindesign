"""Regression tests for gene-level strain design validity (issues #43, #44).

These guard the invariant that every returned gene-based strain design, when re-applied to the
original model, actually satisfies the PROTECT modules and renders the SUPPRESS modules infeasible —
the class of check that was missing when #44 (PROTECT violated by gene_kos) slipped through. Also
guards #43 (neutral gene KOs / gene-name vs gene-id handling).
"""
import pytest
from math import inf
from cobra.io import read_sbml_model
import straindesign as sd
from straindesign.names import (MODULES, MAX_COST, MAX_SOLUTIONS, SOLUTION_APPROACH,
                                KOCOST, GKOCOST, SOLVER)
from straindesign import SUPPRESS, PROTECT

TOL = 1e-6


def _max_flux(model, rid):
    with model:
        model.objective = rid
        model.objective_direction = "maximize"
        s = model.optimize()
        return s.objective_value if s.status == "optimal" else None


def _apply_gene_design(model, design):
    """Apply a (stripped) gene/reaction design to `model` in-place via cobra GPR knockout."""
    for k, v in design.items():
        if any(op in str(k) for op in ["<=", ">=", "<", ">", "="]):
            continue  # regulatory intervention: not exercised in these pure-KO tests
        if v in (-1, False):
            if k in model.genes:
                model.genes.get_by_id(k).knock_out()
            elif model.reactions.has_id(k):
                model.reactions.get_by_id(k).bounds = (0.0, 0.0)


def _gpr_mcs_setup(model, gko_cost, solver, approach):
    modules = [sd.SDModule(model, SUPPRESS, constraints=["1.0 rd_ex >= 1.0 "]),
               sd.SDModule(model, PROTECT, constraints=[[{'r_bm': 1.0}, '>=', 1.0]])]
    return {MODULES: modules, MAX_COST: 3, MAX_SOLUTIONS: inf, SOLUTION_APPROACH: approach,
            KOCOST: {'rs_up': 1.0, 'rd_ex': 1.0, 'rp_ex': 1.1},
            GKOCOST: gko_cost, SOLVER: solver}


@pytest.fixture
def gpr_path():
    import os
    return os.path.join(os.path.dirname(__file__), "model_gpr.xml")


def test_gene_kos_designs_satisfy_protect_and_suppress(gpr_path, curr_solver, comp_approach):
    """#44 regression: every returned gene design must satisfy PROTECT (r_bm>=1) and enforce
    SUPPRESS (max rd_ex < 1) when re-evaluated on a fresh model."""
    if curr_solver == "glpk" and comp_approach == "populate":
        pytest.skip("GLPK cannot reliably populate")
    model = read_sbml_model(gpr_path)
    gko = {g.id: 1.0 for g in model.genes}
    sol = sd.compute_strain_designs(model, sd_setup=_gpr_mcs_setup(model, gko, curr_solver, comp_approach))
    designs = sol.get_gene_sd()
    assert designs, "expected at least one strain design"
    for i, d in enumerate(designs):
        m = read_sbml_model(gpr_path)
        _apply_gene_design(m, d)
        max_bm = _max_flux(m, "r_bm")
        assert max_bm is not None and max_bm >= 1.0 - TOL, \
            f"design {i} VIOLATES PROTECT (max r_bm={max_bm}): {d}"
        max_rd = _max_flux(m, "rd_ex")
        assert max_rd is None or max_rd < 1.0 + TOL, \
            f"design {i} fails to SUPPRESS rd_ex (max rd_ex={max_rd}): {d}"


def test_gene_names_equivalent_to_ids_no_neutral_kos(gpr_path):
    """#43 regression: gko_cost keyed by gene NAMES must give the same valid designs as by gene IDs,
    and must not contain neutral gene KOs (a gene KO whose associated reactions are not knocked)."""
    preferred = [s for s in ["cplex", "scip", "gurobi"] if s in sd.avail_solvers]
    if not preferred:
        pytest.skip("requires cplex/scip/gurobi")
    solver = preferred[0]
    model = read_sbml_model(gpr_path)
    if not all(g.name for g in model.genes):
        pytest.skip("model has no gene names")

    def run(by_names):
        m = read_sbml_model(gpr_path)
        gko = {(g.name if by_names else g.id): 1.0 for g in m.genes}
        sol = sd.compute_strain_designs(m, sd_setup=_gpr_mcs_setup(m, gko, solver, "any"))
        return sol

    def neutral_count(sol, m):
        n = 0
        for gdes, rdes in zip(sol.gene_sd, sol.reaction_sd):  # unstripped attrs
            ko_reacs = {k for k, v in rdes.items() if v in (-1, 0.0, False) and m.reactions.has_id(k)}
            for k, v in gdes.items():
                if v != -1:
                    continue
                g = m.genes.get_by_id(k) if m.genes.has_id(k) else \
                    next((gg for gg in m.genes if gg.name == k), None)
                if g is None:
                    continue  # reaction-level KO, not a gene
                assoc = {r.id for r in g.reactions}
                if assoc and not (assoc & ko_reacs):
                    n += 1
        return n

    m = read_sbml_model(gpr_path)
    sol_ids = run(False)
    sol_names = run(True)
    assert neutral_count(sol_ids, m) == 0, "neutral gene KOs found with gene IDs"
    assert neutral_count(sol_names, m) == 0, "neutral gene KOs found with gene NAMES (#43)"
    # same number of (valid) gene designs regardless of id/name keying
    assert len(sol_ids.get_gene_sd()) == len(sol_names.get_gene_sd())
