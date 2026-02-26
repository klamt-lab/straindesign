"""Test GPR propagation through compression."""
import copy
import pytest
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
import ast


# =====================================================================
# Unit tests for helper functions
# =====================================================================

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


# =====================================================================
# Integration tests with model_gpr.xml
# =====================================================================

@pytest.fixture
def model():
    return read_sbml_model('tests/model_gpr.xml')


class TestModelGprCompression:
    def test_coupled_compression_propagates_gpr(self, model):
        """Coupled compression should AND-combine GPR rules, skipping empty ones."""
        # Pre-process like compress_model does
        remove_blocked_reactions(model)  # removes r7 (F has no consumer)
        stoichmat_coeff2rational(model)
        remove_conservation_relations(model)

        # Save original GPRs for reference
        orig_gprs = {r.id: r.gene_reaction_rule for r in model.reactions}
        print(f"\nAfter blocked removal, reactions: {[r.id for r in model.reactions]}")
        print(f"GPRs: {orig_gprs}")

        reac_map = compress_model_coupled(model, propagate_gpr=True)

        print(f"\nAfter coupled compression:")
        print(f"Reactions: {[r.id for r in model.reactions]}")
        print(f"Reaction map: {reac_map}")
        for r in model.reactions:
            print(f"  {r.id}: GPR='{r.gene_reaction_rule}', genes={sorted(g.id for g in r.genes)}")

        # Check that compressed reactions have correct GPR rules
        for rxn in model.reactions:
            rid = rxn.id
            if rid in reac_map and len(reac_map[rid]) > 1:
                # This is a merged reaction
                orig_ids = list(reac_map[rid].keys())
                print(f"\n  Merged {orig_ids} -> {rid}")
                print(f"  GPR: {rxn.gene_reaction_rule}")

                # Verify: merged GPR should not be empty unless all inputs were empty
                non_empty_inputs = [orig_gprs[oid] for oid in orig_ids if orig_gprs.get(oid, '')]
                if non_empty_inputs:
                    assert rxn.gene_reaction_rule != '', \
                        f"Merged reaction {rid} lost GPR rules from {orig_ids}"

    def test_full_compression_propagates_gpr(self, model):
        """Full compress_model with propagate_gpr should preserve gene information."""
        orig_genes = set()
        for r in model.reactions:
            orig_genes.update(g.id for g in r.genes)
        print(f"\nOriginal genes: {sorted(orig_genes)}")

        cmp_maps = compress_model(model, propagate_gpr=True)

        compressed_genes = set()
        for r in model.reactions:
            compressed_genes.update(g.id for g in r.genes)
            if r.gene_reaction_rule:
                print(f"  {r.id}: {r.gene_reaction_rule}")

        print(f"\nCompressed genes: {sorted(compressed_genes)}")
        print(f"Original genes: {sorted(orig_genes)}")

        # Genes referenced after compression should be a subset of original genes
        assert compressed_genes <= orig_genes, \
            f"Compression introduced unknown genes: {compressed_genes - orig_genes}"

        # The compression should NOT have lost all gene information
        assert len(compressed_genes) > 0, "All gene information was lost during compression"

    def test_full_compression_without_propagation_clears_gpr(self, model):
        """Without propagate_gpr, compression should clear GPR rules (legacy behavior)."""
        compress_model(model, propagate_gpr=False)

        for r in model.reactions:
            assert r.gene_reaction_rule == '', \
                f"Reaction {r.id} has GPR '{r.gene_reaction_rule}' but propagate_gpr=False"

    def test_coupled_group_r4_r5_r6_rdex(self, model):
        """Verify the specific coupled group {r4,r5,r6,rd_ex} gets correct GPR.

        r4: (g8 and g1) or (g8 and g4)  = g8 and (g1 or g4)
        r5: (g7 and g1) or (g9 and g5 and g1)  = g1 and (g7 or (g5 and g9))
        r6: g1 and g4
        rd_ex: '' (empty, skip)

        AND combine (skip empty): g1 & g4 & g8 & (g7 | (g5 & g9))
        DNF: (g1 & g4 & g7 & g8) | (g1 & g4 & g5 & g8 & g9)
        """
        remove_blocked_reactions(model)
        stoichmat_coeff2rational(model)
        remove_conservation_relations(model)

        reac_map = compress_model_coupled(model, propagate_gpr=True)

        # Find the merged reaction that contains r4, r5, r6
        target_rxn = None
        for rid, orig_map in reac_map.items():
            if 'r4' in orig_map and 'r5' in orig_map and 'r6' in orig_map:
                target_rxn = model.reactions.get_by_id(rid)
                break

        assert target_rxn is not None, \
            f"Expected r4, r5, r6 to be merged. Reaction map: {reac_map}"

        # Check the GPR is semantically correct using sympy
        gpr = target_rxn.gene_reaction_rule
        print(f"\nMerged {list(reac_map[target_rxn.id].keys())} -> GPR: {gpr}")

        # Parse back and verify semantic equivalence
        from cobra.core.gene import GPR
        parsed = GPR.from_string(gpr)
        result_sympy = _gpr_ast_to_sympy(parsed.body)

        g1, g4, g5, g7, g8, g9 = [SS(f'g{i}') for i in [1, 4, 5, 7, 8, 9]]
        expected = SO(SA(g1, g4, g7, g8), SA(g1, g4, g5, g8, g9))

        # Simplify both and compare
        assert simplify_logic(result_sympy ^ expected) == False, \
            f"GPR mismatch. Got: {result_sympy}, expected: {expected}"

    def test_coupled_group_r3_rpex(self, model):
        """Verify the coupled group {r3, rp_ex} gets correct GPR.

        r3: g8 or (g6 and g3)
        rp_ex: '' (empty, skip)

        AND combine (skip empty): just r3's GPR = g8 or (g3 and g6)
        """
        remove_blocked_reactions(model)
        stoichmat_coeff2rational(model)
        remove_conservation_relations(model)

        reac_map = compress_model_coupled(model, propagate_gpr=True)

        # Find the merged reaction containing r3
        target_rxn = None
        for rid, orig_map in reac_map.items():
            if 'r3' in orig_map:
                target_rxn = model.reactions.get_by_id(rid)
                break

        assert target_rxn is not None
        gpr = target_rxn.gene_reaction_rule
        print(f"\nMerged {list(reac_map[target_rxn.id].keys())} -> GPR: {gpr}")

        # Parse and check semantic equivalence
        from cobra.core.gene import GPR
        parsed = GPR.from_string(gpr)
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

    def test_efmtool_coupled_gpr_matches_sparse_rref(self, model, java_available):
        """Both backends should produce semantically equivalent GPR rules."""
        model_java = copy.deepcopy(model)

        # Sparse RREF path
        remove_blocked_reactions(model)
        stoichmat_coeff2rational(model)
        remove_conservation_relations(model)
        rref_map = compress_model_coupled(model, compression_backend='sparse_rref', propagate_gpr=True)

        # Efmtool path
        remove_blocked_reactions(model_java)
        stoichmat_coeff2rational(model_java)
        remove_conservation_relations(model_java)
        java_map = compress_model_coupled(model_java, compression_backend='efmtool_rref', propagate_gpr=True)

        # Collect GPR rules by sorted original reaction sets
        def gpr_by_group(model, reac_map):
            result = {}
            for rid, orig_map in reac_map.items():
                key = frozenset(orig_map.keys())
                rxn = model.reactions.get_by_id(rid)
                result[key] = rxn.gene_reaction_rule
            return result

        rref_gprs = gpr_by_group(model, rref_map)
        java_gprs = gpr_by_group(model_java, java_map)

        print(f"\nSparse RREF groups: {len(rref_gprs)}")
        print(f"Efmtool groups: {len(java_gprs)}")

        # Same number of groups
        assert rref_gprs.keys() == java_gprs.keys(), \
            f"Different groups: rref={rref_gprs.keys()}, java={java_gprs.keys()}"

        # Each group should have semantically equivalent GPR
        for group_key in rref_gprs:
            gpr_rref = rref_gprs[group_key]
            gpr_java = java_gprs[group_key]
            print(f"  {sorted(group_key)}: rref='{gpr_rref}', java='{gpr_java}'")

            if not gpr_rref and not gpr_java:
                continue  # both empty, OK

            from cobra.core.gene import GPR
            sym_rref = _gpr_ast_to_sympy(GPR.from_string(gpr_rref).body)
            sym_java = _gpr_ast_to_sympy(GPR.from_string(gpr_java).body)
            assert simplify_logic(sym_rref ^ sym_java) == False, \
                f"GPR mismatch for group {sorted(group_key)}: rref='{gpr_rref}', java='{gpr_java}'"

    def test_efmtool_full_compression_gpr(self, model, java_available):
        """Full compression with efmtool backend should preserve gene info."""
        orig_genes = set()
        for r in model.reactions:
            orig_genes.update(g.id for g in r.genes)

        cmp_maps = compress_model(model, compression_backend='efmtool_rref', propagate_gpr=True)

        compressed_genes = set()
        for r in model.reactions:
            compressed_genes.update(g.id for g in r.genes)
            if r.gene_reaction_rule:
                print(f"  {r.id}: {r.gene_reaction_rule}")

        assert compressed_genes <= orig_genes
        assert len(compressed_genes) > 0, "All gene information was lost"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
