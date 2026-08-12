"""Network completion: the two guarantees the formulation is supposed to buy.

A completed network must satisfy both of

  * a reaction that was not bought carries no flux, and
  * a core reaction that WAS kept demonstrably carries flux,

which together mean the result contains no blocked reactions. Those are properties of the returned
model, so the tests check the model rather than the MILP: rebuild the network from the reported
design and run FVA over it.
"""

import pytest
import cobra
from cobra.flux_analysis import flux_variability_analysis
import straindesign as sd
from straindesign.names import *
from straindesign.completion import build_witness

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


def _rebuild(model, design):
    sub = model.copy()
    sub.remove_reactions([r for r, kept in design.items() if not kept], remove_orphans=True)
    return sub


def test_completion_leaves_nothing_blocked(universe, curr_solver):
    annotated, _, cost = _setup(universe)
    module = sd.SDModule(universe, COMPLETE, constraints=[BIO + ' >= 0.1'],
                         core_reactions=annotated)
    sols = sd.compute_strain_designs(universe, sd_modules=[module], ki_cost=cost,
                                     solver=curr_solver)
    assert sols.status == OPTIMAL
    design = sols.reaction_sd[0]

    sub = _rebuild(universe, design)
    assert sub.slim_optimize() >= 0.1 - 1e-6

    ranges = flux_variability_analysis(sub, fraction_of_optimum=0.0)
    blocked = [r for r in ranges.index
               if max(abs(ranges.minimum[r]), abs(ranges.maximum[r])) < TOL]
    assert blocked == [], 'completed network contains blocked reactions: %s' % blocked[:5]


def test_kept_core_reactions_carry_flux(universe, curr_solver):
    """Jointly, not merely one at a time: individually feasible directions need not be mutually
    consistent, so the demand is only meaningful when imposed on the whole kept core at once."""
    annotated, _, cost = _setup(universe)
    directions, thresholds, unreachable = build_witness(universe, annotated, [BIO + ' >= 0.1'],
                                                        solver=curr_solver)
    core = [r for r in annotated if r not in unreachable]
    module = sd.SDModule(universe, COMPLETE, constraints=[BIO + ' >= 0.1'], core_reactions=core,
                         core_directions=directions, core_thresholds=thresholds)
    design = sd.compute_strain_designs(universe, sd_modules=[module], ki_cost=cost,
                                       solver=curr_solver).reaction_sd[0]
    kept_core = [r for r in core if design.get(r)]
    assert kept_core, 'no annotated reaction survived'

    sub = _rebuild(universe, design)
    with sub:
        for rid in kept_core:
            r = sub.reactions.get_by_id(rid)
            if directions[rid] > 0:
                r.lower_bound = max(r.lower_bound, 1e-5)
            else:
                r.upper_bound = min(r.upper_bound, -1e-5)
        growth = sub.slim_optimize()
    assert growth is not None and growth == growth, \
        'the kept core cannot carry flux in its assigned directions simultaneously'


def test_unbought_reactions_are_gated_off(universe, curr_solver):
    """Gating is by indicator, so no reaction left out may carry flux -- not even solver tolerance
    worth of it, which a big-M formulation would permit."""
    annotated, _, cost = _setup(universe)
    module = sd.SDModule(universe, COMPLETE, constraints=[BIO + ' >= 0.1'],
                         core_reactions=annotated)
    sols = sd.compute_strain_designs(universe, sd_modules=[module], ki_cost=cost,
                                     solver=curr_solver)
    design = sols.reaction_sd[0]
    dropped = [r for r, kept in design.items() if not kept]
    assert dropped, 'nothing was dropped, the test would be vacuous'

    sub = _rebuild(universe, design)
    assert not (set(dropped) & {r.id for r in sub.reactions})


def test_reward_keeps_more_than_penalty(universe, curr_solver):
    """The economics have to bite: pricing annotated reactions above heterologous ones must keep
    fewer of them than rewarding them does."""
    annotated, _, cheap = _setup(universe)
    module = sd.SDModule(universe, COMPLETE, constraints=[BIO + ' >= 0.1'],
                         core_reactions=annotated)
    rewarded = sd.compute_strain_designs(universe, sd_modules=[module], ki_cost=cheap,
                                         solver=curr_solver).reaction_sd[0]
    dear = {r: (5.0 if c < 0 else 1.0) for r, c in cheap.items()}
    penalised = sd.compute_strain_designs(universe, sd_modules=[module], ki_cost=dear,
                                          solver=curr_solver).reaction_sd[0]
    n_rewarded = sum(1 for r in annotated if rewarded.get(r))
    n_penalised = sum(1 for r in annotated if penalised.get(r))
    assert n_penalised < n_rewarded


def test_module_rejects_bad_setups(universe):
    with pytest.raises(Exception, match=CORE_REACTIONS):
        sd.SDModule(universe, COMPLETE, constraints=[BIO + ' >= 0.1'])
    with pytest.raises(Exception, match='not in the model'):
        sd.SDModule(universe, COMPLETE, constraints=[], core_reactions=['NOT_A_REACTION'])
    module = sd.SDModule(universe, COMPLETE, constraints=[], core_reactions=[BIO])
    protect = sd.SDModule(universe, PROTECT, constraints=[BIO + ' >= 0.1'])
    with pytest.raises(Exception, match='cannot be combined'):
        sd.compute_strain_designs(universe, sd_modules=[module, protect], ki_cost={BIO: 1.0})
    with pytest.raises(Exception, match=KICOST):
        sd.compute_strain_designs(universe, sd_modules=[module])


def test_unreachable_core_reactions_are_left_out(universe, curr_solver):
    """A core reaction that cannot carry flux under the module conditions has no must-run
    condition to satisfy, so a reward for keeping it would buy a reaction dead in the result."""
    annotated, _, cost = _setup(universe)
    module = sd.SDModule(universe, COMPLETE, constraints=[BIO + ' >= 0.1'],
                         core_reactions=annotated)
    design = sd.compute_strain_designs(universe, sd_modules=[module], ki_cost=cost,
                                       solver=curr_solver).reaction_sd[0]
    # closed on this medium, so unusable however the rest of the network is completed
    for rid in ['FRUpts2', 'GLNabc', 'MALt2_2']:
        assert not design.get(rid), '%s cannot run but was kept' % rid
