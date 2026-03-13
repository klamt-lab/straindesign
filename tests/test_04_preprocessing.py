"""Test preprocessing: GPR extension, compression, and GPR propagation."""
from .test_01_load_models_and_solvers import *
import straindesign as sd
from numpy import inf
import copy
import ast
from cobra.io import read_sbml_model
from straindesign.compression import (
    compress_model,
    compress_model_coupled,
    compress_model_parallel,
    remove_blocked_reactions,
    stoichmat_coeff2rational,
    remove_conservation_relations,
    stoichmat_coeff2float,
    _combine_gpr_and,
    _combine_gpr_or,
    _gpr_ast_to_sympy,
    _sympy_to_gpr_string,
)
from sympy import simplify_logic, And as SA, Or as SO, Symbol as SS


# ── GPR extension + compression ──────────────────────────────────────

@pytest.mark.timeout(15)
def test_gpr_extension_compression1(model_gpr):
    gkocost = {
        'g1': 1.0,
        'g2': 1.0,
        'g4': 3.0,
        'g5': 2.0,
        'g8': 1.0,
        'g9': 1.0,
    }
    gkicost = {'g3': 1.0, 'g6': 1.0, 'g7': 1.0}
    sd.extend_model_gpr(model_gpr, use_names=False)
    cmp_map = sd.compress_model(model_gpr)
    assert (len(model_gpr.reactions) == 16)
    gkocost, gkicost, cmp_map = sd.compress_ki_ko_cost(gkocost, gkicost, cmp_map)
    assert (len(gkocost) == 4)
    assert (len(gkicost) == 3)


@pytest.mark.timeout(15)
def test_gpr_extension_compression2(model_gpr):
    gkocost = {
        'g1': 1.0,
        'g2': 1.0,
        'g4': 3.0,
        'g5': 2.0,
        'g8': 1.0,
        'g9': 1.0,
    }
    gkicost = {'g3': 1.0, 'g6': 1.0, 'g7': 1.0}
    sd.extend_model_gpr(model_gpr, use_names=False)
    assert (len(model_gpr.reactions) == 40)
    sd.compress_model(model_gpr, set(('g5', 'g9')))
    assert (len(model_gpr.reactions) == 18)


# ── GPR propagation helper unit tests ────────────────────────────────

class TestGprAstToSympy:
    def test_none_returns_none(self):
        assert _gpr_ast_to_sympy(None) is None

    def test_single_gene(self):
        node = ast.Name(id='g1')
        result = _gpr_ast_to_sympy(node)
        assert result == SS('g1')

    def test_and_expression(self):
        node = ast.BoolOp(op=ast.And(), values=[ast.Name(id='g1'), ast.Name(id='g2')])
        result = _gpr_ast_to_sympy(node)
        assert result == SA(SS('g1'), SS('g2'))

    def test_or_expression(self):
        node = ast.BoolOp(op=ast.Or(), values=[ast.Name(id='g1'), ast.Name(id='g2')])
        result = _gpr_ast_to_sympy(node)
        assert result == SO(SS('g1'), SS('g2'))

    def test_nested(self):
        # (g1 and g2) or g3
        node = ast.BoolOp(op=ast.Or(), values=[
            ast.BoolOp(op=ast.And(), values=[ast.Name(id='g1'), ast.Name(id='g2')]),
            ast.Name(id='g3')
        ])
        result = _gpr_ast_to_sympy(node)
        expected = SO(SA(SS('g1'), SS('g2')), SS('g3'))
        assert result == expected


class TestSympyToGprString:
    def test_none_returns_empty(self):
        assert _sympy_to_gpr_string(None) == ''

    def test_single_symbol(self):
        assert _sympy_to_gpr_string(SS('g1')) == 'g1'

    def test_and(self):
        result = _sympy_to_gpr_string(SA(SS('g1'), SS('g2')))
        assert result == 'g1 and g2'

    def test_or(self):
        result = _sympy_to_gpr_string(SO(SS('g1'), SS('g2')))
        assert result == 'g1 or g2'

    def test_nested_and_in_or(self):
        # g3 or (g1 and g2)
        expr = SO(SA(SS('g1'), SS('g2')), SS('g3'))
        result = _sympy_to_gpr_string(expr)
        assert result == '(g1 and g2) or g3'

    def test_nested_or_in_and(self):
        # g3 and (g1 or g2)
        expr = SA(SO(SS('g1'), SS('g2')), SS('g3'))
        result = _sympy_to_gpr_string(expr)
        assert result == '(g1 or g2) and g3'

    def test_deterministic_sorting(self):
        # Should always produce same order
        result1 = _sympy_to_gpr_string(SA(SS('g2'), SS('g1'), SS('g3')))
        result2 = _sympy_to_gpr_string(SA(SS('g3'), SS('g1'), SS('g2')))
        assert result1 == result2 == 'g1 and g2 and g3'


class TestCombineGprAnd:
    def test_all_empty(self):
        assert _combine_gpr_and([None, None]) == ''

    def test_single_non_empty(self):
        node = ast.BoolOp(op=ast.And(), values=[ast.Name(id='g1'), ast.Name(id='g2')])
        result = _combine_gpr_and([node])
        assert result == 'g1 and g2'

    def test_skip_empty(self):
        """Empty GPR (None) should be skipped in AND combination."""
        node = ast.BoolOp(op=ast.And(), values=[ast.Name(id='g1'), ast.Name(id='g2')])
        result = _combine_gpr_and([node, None, None])
        assert result == 'g1 and g2'

    def test_two_non_empty(self):
        node1 = ast.BoolOp(op=ast.And(), values=[ast.Name(id='g1'), ast.Name(id='g2')])
        node2 = ast.Name(id='g3')
        result = _combine_gpr_and([node1, node2])
        assert result == 'g1 and g2 and g3'

    def test_simplification(self):
        """AND of overlapping expressions should simplify."""
        # (g1 and g2) AND (g1 and g3) -> g1 and g2 and g3
        node1 = ast.BoolOp(op=ast.And(), values=[ast.Name(id='g1'), ast.Name(id='g2')])
        node2 = ast.BoolOp(op=ast.And(), values=[ast.Name(id='g1'), ast.Name(id='g3')])
        result = _combine_gpr_and([node1, node2])
        assert result == 'g1 and g2 and g3'

    def test_empty_list(self):
        assert _combine_gpr_and([]) == ''


class TestCombineGprOr:
    def test_any_empty_returns_empty(self):
        """If any reaction has empty GPR (always active), result is empty."""
        node = ast.Name(id='g1')
        result = _combine_gpr_or([node, None])
        assert result == ''

    def test_all_empty(self):
        assert _combine_gpr_or([None, None]) == ''

    def test_two_non_empty(self):
        node1 = ast.Name(id='g1')
        node2 = ast.Name(id='g2')
        result = _combine_gpr_or([node1, node2])
        assert result == 'g1 or g2'

    def test_deduplication(self):
        """OR with duplicate terms should deduplicate (sympy constructor)."""
        # g1 OR g1 -> g1
        node1 = ast.Name(id='g1')
        node2 = ast.Name(id='g1')
        result = _combine_gpr_or([node1, node2])
        assert result == 'g1'

    def test_no_absorption(self):
        """OR does raw merge — absorption is deferred to reduce_gpr."""
        # (g1 and g2) OR g1 -> kept as-is (not simplified to g1)
        node1 = ast.BoolOp(op=ast.And(), values=[ast.Name(id='g1'), ast.Name(id='g2')])
        node2 = ast.Name(id='g1')
        result = _combine_gpr_or([node1, node2])
        assert 'g1 and g2' in result and 'or' in result

    def test_empty_list(self):
        assert _combine_gpr_or([]) == ''


# ── GPR propagation integration tests (model_gpr.xml) ────────────────

@pytest.fixture
def gpr_model():
    return read_sbml_model('tests/model_gpr.xml')


class TestModelGprCompression:
    def test_coupled_compression_propagates_gpr(self, gpr_model):
        """Coupled compression should AND-combine GPR rules, skipping empty ones."""
        remove_blocked_reactions(gpr_model)
        stoichmat_coeff2rational(gpr_model)
        remove_conservation_relations(gpr_model)

        orig_gprs = {r.id: r.gene_reaction_rule for r in gpr_model.reactions}
        reac_map = compress_model_coupled(gpr_model, propagate_gpr=True)

        for rxn in gpr_model.reactions:
            rid = rxn.id
            if rid in reac_map and len(reac_map[rid]) > 1:
                orig_ids = list(reac_map[rid].keys())
                non_empty_inputs = [orig_gprs[oid] for oid in orig_ids if orig_gprs.get(oid, '')]
                if non_empty_inputs:
                    assert rxn.gene_reaction_rule != '', \
                        f"Merged reaction {rid} lost GPR rules from {orig_ids}"

    def test_full_compression_propagates_gpr(self, gpr_model):
        """Full compress_model with propagate_gpr should preserve gene information."""
        orig_genes = set()
        for r in gpr_model.reactions:
            orig_genes.update(g.id for g in r.genes)

        compress_model(gpr_model, propagate_gpr=True)

        compressed_genes = set()
        for r in gpr_model.reactions:
            compressed_genes.update(g.id for g in r.genes)

        assert compressed_genes <= orig_genes, \
            f"Compression introduced unknown genes: {compressed_genes - orig_genes}"
        assert len(compressed_genes) > 0, "All gene information was lost during compression"

    def test_full_compression_without_propagation_clears_gpr(self, gpr_model):
        """Without propagate_gpr, compression should clear GPR rules (legacy behavior)."""
        compress_model(gpr_model, propagate_gpr=False)

        for r in gpr_model.reactions:
            assert r.gene_reaction_rule == '', \
                f"Reaction {r.id} has GPR '{r.gene_reaction_rule}' but propagate_gpr=False"

    def test_coupled_group_r4_r5_r6_rdex(self, gpr_model):
        """Verify the specific coupled group {r4,r5,r6,rd_ex} gets correct GPR.

        r4: (g8 and g1) or (g8 and g4)  = g8 and (g1 or g4)
        r5: (g7 and g1) or (g9 and g5 and g1)  = g1 and (g7 or (g5 and g9))
        r6: g1 and g4
        rd_ex: '' (empty, skip)

        AND combine (skip empty): g1 & g4 & g8 & (g7 | (g5 & g9))
        DNF: (g1 & g4 & g7 & g8) | (g1 & g4 & g5 & g8 & g9)
        """
        remove_blocked_reactions(gpr_model)
        stoichmat_coeff2rational(gpr_model)
        remove_conservation_relations(gpr_model)

        reac_map = compress_model_coupled(gpr_model, propagate_gpr=True)

        target_rxn = None
        for rid, orig_map in reac_map.items():
            if 'r4' in orig_map and 'r5' in orig_map and 'r6' in orig_map:
                target_rxn = gpr_model.reactions.get_by_id(rid)
                break

        assert target_rxn is not None, \
            f"Expected r4, r5, r6 to be merged. Reaction map: {reac_map}"

        from cobra.core.gene import GPR
        parsed = GPR.from_string(target_rxn.gene_reaction_rule)
        result_sympy = _gpr_ast_to_sympy(parsed.body)

        g1, g4, g5, g7, g8, g9 = [SS(f'g{i}') for i in [1, 4, 5, 7, 8, 9]]
        expected = SO(SA(g1, g4, g7, g8), SA(g1, g4, g5, g8, g9))

        assert simplify_logic(result_sympy ^ expected) == False, \
            f"GPR mismatch. Got: {result_sympy}, expected: {expected}"

    def test_coupled_group_r3_rpex(self, gpr_model):
        """Verify the coupled group {r3, rp_ex} gets correct GPR.

        r3: g8 or (g6 and g3)
        rp_ex: '' (empty, skip)

        AND combine (skip empty): just r3's GPR = g8 or (g3 and g6)
        """
        remove_blocked_reactions(gpr_model)
        stoichmat_coeff2rational(gpr_model)
        remove_conservation_relations(gpr_model)

        reac_map = compress_model_coupled(gpr_model, propagate_gpr=True)

        target_rxn = None
        for rid, orig_map in reac_map.items():
            if 'r3' in orig_map:
                target_rxn = gpr_model.reactions.get_by_id(rid)
                break

        assert target_rxn is not None
        from cobra.core.gene import GPR
        parsed = GPR.from_string(target_rxn.gene_reaction_rule)
        result_sympy = _gpr_ast_to_sympy(parsed.body)

        g3, g6, g8 = SS('g3'), SS('g6'), SS('g8')
        expected = SO(g8, SA(g3, g6))

        assert simplify_logic(result_sympy ^ expected) == False, \
            f"GPR mismatch. Got: {result_sympy}, expected: {expected}"


class TestEfmtoolBackendGpr:
    """Test GPR propagation with efmtool_rref backend (if Java available)."""

    @pytest.fixture
    def java_available(self):
        try:
            from straindesign.efmtool_cmp_interface import _check_jpype_available
            if not _check_jpype_available():
                pytest.skip("jpype not installed")
            from straindesign.efmtool_cmp_interface import _init_java
            _init_java()
        except Exception as e:
            pytest.skip(f"Java/jpype not available: {e}")

    def test_efmtool_coupled_gpr_matches_sparse_rref(self, gpr_model, java_available):
        """Both backends should produce semantically equivalent GPR rules."""
        model_java = copy.deepcopy(gpr_model)

        # Sparse RREF path
        remove_blocked_reactions(gpr_model)
        stoichmat_coeff2rational(gpr_model)
        remove_conservation_relations(gpr_model)
        rref_map = compress_model_coupled(gpr_model, compression_backend='sparse_rref', propagate_gpr=True)

        # Efmtool path
        remove_blocked_reactions(model_java)
        stoichmat_coeff2rational(model_java)
        remove_conservation_relations(model_java)
        java_map = compress_model_coupled(model_java, compression_backend='efmtool_rref', propagate_gpr=True)

        def gpr_by_group(model, reac_map):
            result = {}
            for rid, orig_map in reac_map.items():
                key = frozenset(orig_map.keys())
                rxn = model.reactions.get_by_id(rid)
                result[key] = rxn.gene_reaction_rule
            return result

        rref_gprs = gpr_by_group(gpr_model, rref_map)
        java_gprs = gpr_by_group(model_java, java_map)

        assert rref_gprs.keys() == java_gprs.keys(), \
            f"Different groups: rref={rref_gprs.keys()}, java={java_gprs.keys()}"

        for group_key in rref_gprs:
            gpr_rref = rref_gprs[group_key]
            gpr_java = java_gprs[group_key]

            if not gpr_rref and not gpr_java:
                continue

            from cobra.core.gene import GPR
            sym_rref = _gpr_ast_to_sympy(GPR.from_string(gpr_rref).body)
            sym_java = _gpr_ast_to_sympy(GPR.from_string(gpr_java).body)
            assert simplify_logic(sym_rref ^ sym_java) == False, \
                f"GPR mismatch for group {sorted(group_key)}: rref='{gpr_rref}', java='{gpr_java}'"

    def test_efmtool_full_compression_gpr(self, gpr_model, java_available):
        """Full compression with efmtool backend should preserve gene info."""
        orig_genes = set()
        for r in gpr_model.reactions:
            orig_genes.update(g.id for g in r.genes)

        compress_model(gpr_model, compression_backend='efmtool_rref', propagate_gpr=True)

        compressed_genes = set()
        for r in gpr_model.reactions:
            compressed_genes.update(g.id for g in r.genes)

        assert compressed_genes <= orig_genes
        assert len(compressed_genes) > 0, "All gene information was lost"
