"""Test gene knockout simulation via gene_kos_to_constraints and resolve_gene_constraints."""
import pytest
from cobra.io import load_model
import straindesign as sd
from straindesign.networktools import gene_kos_to_constraints, resolve_gene_constraints


@pytest.fixture(scope="module")
def ecoli_core():
    return load_model('textbook')


@pytest.fixture(scope="module")
def wt_growth(ecoli_core):
    return sd.fba(ecoli_core).objective_value


# ── gene_kos_to_constraints unit tests ───────────────────────────────

class TestGeneKosToConstraints:
    def test_single_gene_by_id(self, ecoli_core):
        """b0727 (sucB) is in AND rule for AKGDH → single KO kills AKGDH."""
        c = gene_kos_to_constraints(ecoli_core, ['b0727'])
        knocked = {list(x[0].keys())[0] for x in c}
        assert 'AKGDH' in knocked

    def test_single_gene_by_name(self, ecoli_core):
        """Gene name 'sucB' should resolve to b0727 and produce same result."""
        c_id = gene_kos_to_constraints(ecoli_core, ['b0727'])
        c_name = gene_kos_to_constraints(ecoli_core, ['sucB'])
        assert c_id == c_name

    def test_or_rule_not_knocked_out(self, ecoli_core):
        """b1241 (adhE) in OR rules for ACALD, ALCD2x → single KO insufficient."""
        c = gene_kos_to_constraints(ecoli_core, ['b1241'])
        knocked = {list(x[0].keys())[0] for x in c}
        assert 'ACALD' not in knocked
        assert 'ALCD2x' not in knocked

    def test_and_rule_any_member_sufficient(self, ecoli_core):
        """AKGDH = b0726 and b0116 and b0727 → any single member KOs AKGDH."""
        for g_id in ['b0726', 'b0116', 'b0727']:
            c = gene_kos_to_constraints(ecoli_core, [g_id])
            knocked = {list(x[0].keys())[0] for x in c}
            assert 'AKGDH' in knocked, f'{g_id} alone should knock out AKGDH'

    def test_multi_reaction_knockout(self, ecoli_core):
        """b0116 (lpd) participates in both AKGDH and PDH AND rules."""
        c = gene_kos_to_constraints(ecoli_core, ['b0116'])
        knocked = {list(x[0].keys())[0] for x in c}
        assert knocked == {'AKGDH', 'PDH'}

    def test_unknown_gene_ignored(self, ecoli_core):
        """Unknown gene identifiers are silently skipped."""
        c = gene_kos_to_constraints(ecoli_core, ['NONEXISTENT_GENE'])
        assert c == []

    def test_empty_input(self, ecoli_core):
        c = gene_kos_to_constraints(ecoli_core, [])
        assert c == []


# ── resolve_gene_constraints unit tests ──────────────────────────────

class TestResolveGeneConstraints:
    def test_string_format(self, ecoli_core):
        """Gene KO as string 'b0727 = 0' should produce reaction constraints."""
        c = resolve_gene_constraints(ecoli_core, 'b0727 = 0')
        knocked = {list(x[0].keys())[0] for x in c}
        assert 'AKGDH' in knocked

    def test_list_of_strings(self, ecoli_core):
        """Multiple gene KOs as list of strings."""
        c = resolve_gene_constraints(ecoli_core, ['b0727 = 0', 'b0116 = 0'])
        knocked = {list(x[0].keys())[0] for x in c}
        assert 'AKGDH' in knocked
        assert 'PDH' in knocked

    def test_mixed_gene_and_reaction(self, ecoli_core):
        """Gene constraints mixed with reaction constraints."""
        c = resolve_gene_constraints(ecoli_core, ['b0727 = 0', 'EX_o2_e = 0'])
        knocked = {list(x[0].keys())[0] for x in c}
        assert 'AKGDH' in knocked
        assert 'EX_o2_e' in knocked

    def test_pure_reaction_constraints_unchanged(self, ecoli_core):
        """Pure reaction constraints pass through without modification."""
        c = resolve_gene_constraints(ecoli_core, ['EX_o2_e = 0', 'PFK <= 5'])
        # Should be exactly the parsed reaction constraints, no gene processing
        keys = {list(x[0].keys())[0] for x in c}
        assert keys == {'EX_o2_e', 'PFK'}

    def test_empty_constraints(self, ecoli_core):
        assert resolve_gene_constraints(ecoli_core, []) == []
        assert resolve_gene_constraints(ecoli_core, '') == []


# ── FBA integration tests ────────────────────────────────────────────

class TestFbaWithGeneKO:
    def test_nonessential_single_ko(self, ecoli_core, wt_growth):
        """b0727 (sucB) KO reduces growth slightly but is not lethal."""
        sol = sd.fba(ecoli_core, constraints='b0727 = 0')
        assert sol.status == sd.OPTIMAL
        assert sol.objective_value > 0.9 * wt_growth  # ~0.858 vs 0.874

    def test_essential_single_ko(self, ecoli_core):
        """b0720 (gltA, citrate synthase) is essential — KO kills growth."""
        sol = sd.fba(ecoli_core, constraints='b0720 = 0')
        assert sol.objective_value < 1e-6

    def test_synthetic_lethal_pair(self, ecoli_core, wt_growth):
        """b1761 (gdhA) and b3213 (gltD) are individually viable but
        synthetically lethal: both glutamate synthesis routes blocked."""
        sol1 = sd.fba(ecoli_core, constraints='b1761 = 0')
        sol2 = sd.fba(ecoli_core, constraints='b3213 = 0')
        assert sol1.objective_value > 0.5 * wt_growth
        assert sol2.objective_value > 0.5 * wt_growth

        sol_both = sd.fba(ecoli_core, constraints=['b1761 = 0', 'b3213 = 0'])
        assert sol_both.objective_value < 1e-6

    def test_gene_ko_with_name(self, ecoli_core, wt_growth):
        """Gene KO by name should produce same FBA result as by ID."""
        sol_id = sd.fba(ecoli_core, constraints='b0727 = 0')
        sol_name = sd.fba(ecoli_core, constraints='sucB = 0')
        assert abs(sol_id.objective_value - sol_name.objective_value) < 1e-9

    def test_gene_ko_mixed_with_reaction_constraint(self, ecoli_core, wt_growth):
        """Gene KO combined with reaction constraint."""
        sol = sd.fba(ecoli_core, constraints=['b0727 = 0', 'EX_glc__D_e >= -5'])
        assert sol.status == sd.OPTIMAL
        assert sol.objective_value < wt_growth  # more constrained


# ── FVA integration tests ────────────────────────────────────────────

class TestFvaWithGeneKO:
    def test_knocked_reaction_fixed_at_zero(self, ecoli_core):
        """FVA after b0727 KO: AKGDH must have min=max=0."""
        fva_r = sd.fva(ecoli_core, constraints='b0727 = 0')
        assert abs(fva_r.loc['AKGDH', 'minimum']) < 1e-9
        assert abs(fva_r.loc['AKGDH', 'maximum']) < 1e-9

    def test_essential_gene_ko_infeasible(self, ecoli_core):
        """FVA after essential gene KO: biomass range is [0, 0]."""
        fva_r = sd.fva(ecoli_core, constraints='b0720 = 0')
        assert abs(fva_r.loc['Biomass_Ecoli_core', 'maximum']) < 1e-6

    def test_fva_gene_plus_reaction_constraint(self, ecoli_core):
        """FVA with gene KO (s0001) + reaction constraint."""
        fva_r = sd.fva(ecoli_core, constraints=['s0001 = 0', 'EX_glc__D_e >= -5'])
        # s0001 KOs ACALDt, CO2t, O2t
        for rxn in ['ACALDt', 'CO2t', 'O2t']:
            assert abs(fva_r.loc[rxn, 'minimum']) < 1e-9
            assert abs(fva_r.loc[rxn, 'maximum']) < 1e-9
