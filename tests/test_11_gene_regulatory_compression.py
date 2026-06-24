"""Regression tests for gene-regulatory correctness under compression.

A gene controlling several reactions that get merged BEFORE GPR integration would otherwise be
hooked to the merged reaction with the wrong (collapsed) stoichiometry, so a gene-regulatory bound
(g <= X / g >= X) is mis-scaled vs the uncompressed model. The fix exempts gene-controlled reactions
from coupled merging in COMPRESS#1 (`no_coupled_compress_reacs`). These tests pin both the exemption
mechanism and the corrected gene-regulatory multiplicity.
"""
import pytest
from cobra import Model, Reaction, Metabolite
import straindesign as sd
from straindesign.networktools import extend_model_gpr
from straindesign.compression import compress_model
from straindesign.names import MAXIMIZE

TOL = 1e-6


def _fba(model, obj):
    return sd.fba(model, obj=obj, obj_sense=MAXIMIZE).objective_value


def _remap_obj(obj, cmap):
    obj = dict(obj)
    for step in cmap:
        for new_r, old in step["reac_map_exp"].items():
            for o in list(obj.keys()):
                if o in old:
                    obj[new_r] = obj.pop(o) * float(old[o])
    return obj


def _coupled_chain():
    m = Model("chain")
    mets = {x: Metabolite(x) for x in "ABCD"}
    m.add_metabolites(list(mets.values()))
    vin = Reaction("vin"); vin.add_metabolites({mets["A"]: 1}); vin.bounds = (0, 10)
    r1 = Reaction("r1"); r1.add_metabolites({mets["A"]: -1, mets["B"]: 1}); r1.bounds = (0, 1000)
    r2 = Reaction("r2"); r2.add_metabolites({mets["B"]: -1, mets["C"]: 1}); r2.bounds = (0, 1000)
    r3 = Reaction("r3"); r3.add_metabolites({mets["C"]: -1, mets["D"]: 1}); r3.bounds = (0, 1000)
    vout = Reaction("vout"); vout.add_metabolites({mets["D"]: -1}); vout.bounds = (0, 1000)
    m.add_reactions([vin, r1, r2, r3, vout]); m.objective = "vout"
    return m


def test_coupled_exemption_keeps_reaction_and_merges_rest():
    """Protecting one reaction in a coupled group keeps it intact while the rest still merge,
    and flux space is preserved."""
    base = _fba(_coupled_chain(), {"vout": 1})

    m_full = _coupled_chain(); compress_model(m_full)
    assert len(m_full.reactions) == 1, "unprotected coupled chain should fully collapse"

    m_prot = _coupled_chain()
    cmap = compress_model(m_prot, no_coupled_compress_reacs={"r2"})
    ids = [r.id for r in m_prot.reactions]
    assert any(r.id == "r2" for r in m_prot.reactions), "protected reaction r2 must be kept intact"
    assert any(("r1" in i and "r3" in i) for i in ids), "r1 and r3 should still merge together"
    obj = _remap_obj({"vout": 1}, cmap)
    assert abs(base - _fba(m_prot, obj)) < TOL, "compression must preserve flux space"


def _gene_model():
    """g1 controls r1 (A->2B) and r2 (B->C), which are coupled 1:2."""
    m = Model("g"); A, B, C = Metabolite("A"), Metabolite("B"), Metabolite("C")
    m.add_metabolites([A, B, C])
    vS = Reaction("vS"); vS.add_metabolites({A: 1}); vS.bounds = (0, 10)
    r1 = Reaction("r1"); r1.add_metabolites({A: -1, B: 2}); r1.bounds = (0, 1000); r1.gene_reaction_rule = "g1"
    r2 = Reaction("r2"); r2.add_metabolites({B: -1, C: 1}); r2.bounds = (0, 1000); r2.gene_reaction_rule = "g1"
    vC = Reaction("vC"); vC.add_metabolites({C: -1}); vC.bounds = (0, 1000)
    m.add_reactions([vS, r1, r2, vC]); m.objective = "vC"
    return m


def _max_C_under_g1_le_1(protect):
    m = _gene_model()
    cmap = compress_model(m, no_coupled_compress_reacs=({"r1", "r2"} if protect else set()),
                          propagate_gpr=True)
    extend_model_gpr(m, use_names=False)
    m.reactions.g1.upper_bound = 1.0
    return _fba(m, _remap_obj({"vC": 1}, cmap))


def test_gene_regulatory_multiplicity_preserved_under_compression():
    """With g1 controlling coupled r1,r2, the regulatory bound g1<=1 must give the same max product
    compressed (with protection) as uncompressed; without protection it is wrong (3x)."""
    mu = _gene_model(); extend_model_gpr(mu, use_names=False); mu.reactions.g1.upper_bound = 1.0
    ref = _fba(mu, {"vC": 1})
    assert abs(ref - 2.0 / 3.0) < 1e-4, "reference should be 2/3"

    assert abs(_max_C_under_g1_le_1(protect=True) - ref) < 1e-4, \
        "protected compression must match the uncompressed gene-regulatory result"
    # sanity: the bug (no protection) would give 2.0, i.e. 3x off
    assert abs(_max_C_under_g1_le_1(protect=False) - 2.0) < 1e-4, \
        "unprotected compression collapses the gene multiplicity (documents the bug)"
