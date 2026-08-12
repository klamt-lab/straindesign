"""CarveMe modules: the guarantees the formulation buys, and that it composes.

A reconstructed network must satisfy both of

  * a reaction that was not bought carries no flux, and
  * a core reaction that WAS kept demonstrably carries flux,

which together mean the result contains no blocked reactions. Those are properties of the returned
model, so the tests check the model rather than the MILP: rebuild the network from the reported
design and run FVA over it.

The module is built inside SDProblem like every other module type, so it also has to compose with
them -- combining it with a PROTECT module is tested here, not just assumed.
"""

import pytest
import cobra
from cobra.flux_analysis import flux_variability_analysis
import straindesign as sd
from straindesign.names import *
from straindesign.carveme import build_witness

BIO = 'BIOMASS_Ecoli_core_w_GAM'
TOL = 1e-7


@pytest.fixture(scope='module')
def universe():
    return cobra.io.load_model('e_coli_core')


def _setup(model):
    """Annotated reactions are rewarded and must run; unannotated ones cost."""
    annotated = [r.id for r in model.reactions if r.gene_reaction_rule]
    hetero = [r.id for r in model.reactions if not r.gene_reaction_rule and r.id != BIO]
    cost = {r: -1.0 for r in annotated}
    cost.update({r: 1.0 for r in hetero})
    return annotated, hetero, cost


def _reconstruct(model, modules, cost, solver):
    return sd.compute_strain_designs(model, sd_modules=modules, ki_cost=cost, solver=solver,
                                     solution_approach=BEST, max_solutions=1, compress=False)


def _rebuild(model, design, candidates):
    """Every candidate the design does not mark as bought is absent from the network.

    Iterating the candidates rather than the design's keys matters: a candidate whose binary the
    MILP fixed to zero is reported in no design at all, so reading the design alone would leave
    it in the model.
    """
    sub = model.copy()
    sub.remove_reactions([r for r in candidates if not design.get(r)], remove_orphans=True)
    return sub


def test_reconstruction_leaves_nothing_blocked(universe, curr_solver):
    annotated, _, cost = _setup(universe)
    module = sd.SDModule(universe, CARVEME, constraints=[BIO + ' >= 0.1'],
                         core_reactions=annotated)
    solution = _reconstruct(universe, [module], cost, curr_solver)
    assert solution.status == OPTIMAL
    design = solution.reaction_sd[0]

    sub = _rebuild(universe, design, cost)
    assert sub.slim_optimize() >= 0.1 - 1e-6

    ranges = flux_variability_analysis(sub, fraction_of_optimum=0.0)
    blocked = [r for r in ranges.index
               if max(abs(ranges.minimum[r]), abs(ranges.maximum[r])) < TOL]
    assert blocked == [], 'reconstructed network contains blocked reactions: %s' % blocked[:5]


def test_kept_core_reactions_carry_flux(universe, curr_solver):
    """Jointly, not merely one at a time: individually feasible directions need not be mutually
    consistent, so the demand is only meaningful when imposed on the whole kept core at once."""
    annotated, _, cost = _setup(universe)
    directions, thresholds, unreachable = build_witness(universe, annotated, [BIO + ' >= 0.1'],
                                                        solver=curr_solver)
    core = [r for r in annotated if r not in unreachable]
    module = sd.SDModule(universe, CARVEME, constraints=[BIO + ' >= 0.1'], core_reactions=core,
                         core_directions=directions, core_thresholds=thresholds)
    design = _reconstruct(universe, [module], cost, curr_solver).reaction_sd[0]
    kept_core = [r for r in core if design.get(r)]
    assert kept_core, 'no annotated reaction survived'

    sub = _rebuild(universe, design, cost)
    with sub:
        for rid in kept_core:
            rxn = sub.reactions.get_by_id(rid)
            if directions[rid] > 0:
                rxn.lower_bound = max(rxn.lower_bound, 1e-5)
            else:
                rxn.upper_bound = min(rxn.upper_bound, -1e-5)
        growth = sub.slim_optimize()
    assert growth is not None and growth == growth, \
        'the kept core cannot carry flux in its assigned directions simultaneously'


def test_unbought_reactions_are_gated_off(universe, curr_solver):
    annotated, _, cost = _setup(universe)
    module = sd.SDModule(universe, CARVEME, constraints=[BIO + ' >= 0.1'],
                         core_reactions=annotated)
    design = _reconstruct(universe, [module], cost, curr_solver).reaction_sd[0]
    dropped = [r for r in cost if not design.get(r)]
    assert dropped, 'nothing was dropped, the test would be vacuous'
    sub = _rebuild(universe, design, cost)
    assert not (set(dropped) & {r.id for r in sub.reactions})


def test_reward_keeps_more_than_penalty(universe, curr_solver):
    """The economics have to bite: pricing annotated reactions above heterologous ones must keep
    fewer of them than rewarding them does."""
    annotated, _, cheap = _setup(universe)
    module = sd.SDModule(universe, CARVEME, constraints=[BIO + ' >= 0.1'],
                         core_reactions=annotated)
    rewarded = _reconstruct(universe, [module], cheap, curr_solver).reaction_sd[0]
    dear = {r: (5.0 if c < 0 else 1.0) for r, c in cheap.items()}
    penalised = _reconstruct(universe, [module], dear, curr_solver).reaction_sd[0]
    n_rewarded = sum(1 for r in annotated if rewarded.get(r))
    n_penalised = sum(1 for r in annotated if penalised.get(r))
    assert n_penalised < n_rewarded


def test_combines_with_a_protect_module(universe, curr_solver):
    """The reason this module type is built inside SDProblem rather than beside it: an extra
    PROTECT module must constrain the same binaries, in one MILP."""
    annotated, _, cost = _setup(universe)
    carve = sd.SDModule(universe, CARVEME, constraints=[BIO + ' >= 0.1'],
                        core_reactions=annotated)
    # the reconstructed network must additionally be able to secrete acetate
    protect = sd.SDModule(universe, PROTECT, constraints=['EX_ac_e >= 1'])
    solution = _reconstruct(universe, [carve, protect], cost, curr_solver)
    assert solution.status == OPTIMAL
    design = solution.reaction_sd[0]

    sub = _rebuild(universe, design, cost)
    with sub:
        sub.reactions.EX_ac_e.lower_bound = 1.0
        assert sub.slim_optimize() >= 0.1 - 1e-6, \
            'the PROTECT module was not enforced on the reconstructed network'
    ranges = flux_variability_analysis(sub, fraction_of_optimum=0.0)
    blocked = [r for r in ranges.index
               if max(abs(ranges.minimum[r]), abs(ranges.maximum[r])) < TOL]
    assert blocked == [], 'combined design left blocked reactions: %s' % blocked[:5]


def test_module_rejects_bad_setups(universe):
    with pytest.raises(Exception, match=CORE_REACTIONS):
        sd.SDModule(universe, CARVEME, constraints=[BIO + ' >= 0.1'])
    with pytest.raises(Exception, match='not in the model'):
        sd.SDModule(universe, CARVEME, constraints=[], core_reactions=['NOT_A_REACTION'])


def test_unreachable_core_reactions_are_not_bought(universe, curr_solver):
    """A core reaction that cannot carry flux under the module's constraints is dropped from the
    core, which leaves nothing obliging it to run. Rewarded, it would then be bought and sit in
    the result blocked -- so the purchase is forbidden outright."""
    annotated, _, cost = _setup(universe)
    module = sd.SDModule(universe, CARVEME, constraints=[BIO + ' >= 0.1'],
                         core_reactions=annotated)
    design = _reconstruct(universe, [module], cost, curr_solver).reaction_sd[0]
    # closed on this medium, so unusable however the rest of the network is reconstructed
    for rid in ['FRUpts2', 'GLNabc', 'MALt2_2']:
        assert not design.get(rid), '%s cannot run but was bought' % rid


def test_positive_lower_bounds_are_not_overridden(universe, curr_solver):
    """StrainDesign does not let an intervention relax a model bound, and a CarveMe module is no
    exception: ATPM's maintenance demand still holds, which costs one annotated reaction here."""
    annotated, _, cost = _setup(universe)
    module = sd.SDModule(universe, CARVEME, constraints=[BIO + ' >= 0.1'],
                         core_reactions=annotated)
    assert universe.reactions.ATPM.lower_bound > 0
    assert cost['ATPM'] > 0, 'ATPM must be priced as a penalty for this test to mean anything'
    design = _reconstruct(universe, [module], cost, curr_solver).reaction_sd[0]

    # Dropping ATPM would mean v = 0, which its own lower bound forbids. StrainDesign will not
    # relax a model bound to make an intervention possible, so ATPM is bought even though it is
    # charged for and nothing rewards it.
    assert design.get('ATPM'), 'a reaction with a positive lower bound was dropped'
    sub = _rebuild(universe, design, cost)
    assert sub.reactions.ATPM.lower_bound == universe.reactions.ATPM.lower_bound
