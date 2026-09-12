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

BIO = 'BIOMASS_Ecoli_core_w_GAM'
TOL = 1e-7


@pytest.fixture(scope='module')
def universe():
    return cobra.io.load_model('e_coli_core')


@pytest.fixture(scope='module')
def capable_solver():
    """A deterministic, capable backend for the tests that are not about solver behaviour.

    Not the ambient default: that is whatever earlier tests happened to leave configured, so the
    same test can land on GLPK in a full-suite run and on CPLEX when the file is run alone. GLPK
    substitutes every indicator constraint with a big-M and does not finish two combined CarveMe
    modules at all, which is how this surfaced.
    """
    from straindesign import avail_solvers
    for candidate in (CPLEX, GUROBI, SCIP, GLPK):
        if candidate in avail_solvers:
            return candidate
    return GLPK


def _setup(model):
    """Annotated reactions are rewarded and must run; unannotated ones cost."""
    annotated = [r.id for r in model.reactions if r.gene_reaction_rule]
    hetero = [r.id for r in model.reactions if not r.gene_reaction_rule and r.id != BIO]
    cost = {r: -1.0 for r in annotated}
    cost.update({r: 1.0 for r in hetero})
    return annotated, hetero, cost


def _reconstruct(model, modules, cost, solver):
    kwargs = {} if solver is None else {'solver': solver}
    return sd.compute_strain_designs(model, sd_modules=modules, ki_cost=cost,
                                     solution_approach=BEST, max_solutions=1, compress=False,
                                     **kwargs)


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
    """Jointly, not merely one at a time.

    The check must not read directions off per-reaction FVA ranges: individually feasible
    directions need not be jointly consistent, which is the whole reason the module lets the MILP
    choose them. Instead it constructs one flux state that carries the entire kept core --
    maximise each reaction in turn, normalise, and sum. The feasible set is convex so the sum is
    feasible, and it is non-zero wherever any summand was.
    """
    import numpy as np
    annotated, _, cost = _setup(universe)
    module = sd.SDModule(universe, CARVEME, constraints=[BIO + ' >= 0.1'],
                         core_reactions=annotated)
    design = _reconstruct(universe, [module], cost, curr_solver).reaction_sd[0]
    kept_core = [r for r in annotated if design.get(r)]
    assert kept_core, 'no annotated reaction survived'

    sub = _rebuild(universe, design, cost)
    state = np.zeros(len(sub.reactions))
    order = {r.id: i for i, r in enumerate(sub.reactions)}
    for rid in kept_core:
        for sense in ('max', 'min'):
            with sub:
                sub.objective = sub.reactions.get_by_id(rid)
                sub.objective_direction = sense
                solution = sub.optimize()
            if solution.status != 'optimal':
                continue
            values = solution.fluxes.values
            if abs(values[order[rid]]) > 1e-9:
                state += values / (np.abs(values).max() + 1e-12)
                break

    dead = [r for r in kept_core if abs(state[order[r]]) < TOL]
    assert dead == [], 'kept core reactions carrying no flux in a common state: %s' % dead[:5]


def test_unbought_reactions_are_gated_off(universe, curr_solver):
    annotated, _, cost = _setup(universe)
    module = sd.SDModule(universe, CARVEME, constraints=[BIO + ' >= 0.1'],
                         core_reactions=annotated)
    design = _reconstruct(universe, [module], cost, curr_solver).reaction_sd[0]
    dropped = [r for r in cost if not design.get(r)]
    assert dropped, 'nothing was dropped, the test would be vacuous'
    sub = _rebuild(universe, design, cost)
    assert not (set(dropped) & {r.id for r in sub.reactions})


def test_reward_keeps_more_than_penalty(universe, capable_solver):
    """The economics have to bite: pricing annotated reactions above heterologous ones must keep
    fewer of them than rewarding them does.

    Not run per solver: this is a property of the cost function, not of any backend, and the
    all-positive-cost variant is pathologically slow on GLPK (150 s against 5 s elsewhere) for no
    added coverage.
    """
    annotated, _, cheap = _setup(universe)
    module = sd.SDModule(universe, CARVEME, constraints=[BIO + ' >= 0.1'],
                         core_reactions=annotated)
    rewarded = _reconstruct(universe, [module], cheap, capable_solver).reaction_sd[0]
    dear = {r: (5.0 if c < 0 else 1.0) for r, c in cheap.items()}
    penalised = _reconstruct(universe, [module], dear, capable_solver).reaction_sd[0]
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
    """A core reaction that cannot carry flux has no satisfiable must-run row, so z = 0 is the
    only feasible choice for it. No detection pass is involved -- the formulation does it."""
    annotated, _, cost = _setup(universe)
    module = sd.SDModule(universe, CARVEME, constraints=[BIO + ' >= 0.1'],
                         core_reactions=annotated)
    design = _reconstruct(universe, [module], cost, curr_solver).reaction_sd[0]
    # closed on this medium, so unusable however the rest of the network is reconstructed
    for rid in ['FRUpts2', 'GLNabc', 'MALt2_2']:
        assert not design.get(rid), '%s cannot run but was bought' % rid


def test_positive_lower_bounds_are_not_overridden(universe, capable_solver):
    """StrainDesign does not let an intervention relax a model bound, and a CarveMe module is no
    exception: ATPM's maintenance demand still holds, which costs one annotated reaction here."""
    annotated, _, cost = _setup(universe)
    module = sd.SDModule(universe, CARVEME, constraints=[BIO + ' >= 0.1'],
                         core_reactions=annotated)
    assert universe.reactions.ATPM.lower_bound > 0
    assert cost['ATPM'] > 0, 'ATPM must be priced as a penalty for this test to mean anything'
    design = _reconstruct(universe, [module], cost, capable_solver).reaction_sd[0]

    # Dropping ATPM would mean v = 0, which its own lower bound forbids. StrainDesign will not
    # relax a model bound to make an intervention possible, so ATPM is bought even though it is
    # charged for and nothing rewards it.
    assert design.get('ATPM'), 'a reaction with a positive lower bound was dropped'
    sub = _rebuild(universe, design, cost)
    assert sub.reactions.ATPM.lower_bound == universe.reactions.ATPM.lower_bound


@pytest.mark.parametrize('compress', [False, COUPLED, True])
@pytest.mark.parametrize('skip_fvas', [False, True])
def test_pipeline_options_do_not_change_the_answer(universe, compress, skip_fvas):
    """Compression and the preprocessing FVAs are ordinary pipeline options here, not something
    this module type bypasses -- so every combination has to reach the same objective.

    The FVA setting is not merely cosmetic for this module type: bound_blocked_or_irrevers_fva
    widens a bound to infinity once it proves the bound never binds, and the must-run rows then
    take their finite relaxation value from the FVA range instead. Leaving both settings under
    test keeps that interaction honest.
    """
    annotated, _, cost = _setup(universe)
    module = sd.SDModule(universe, CARVEME, constraints=[BIO + ' >= 0.1'],
                         core_reactions=annotated)
    solution = sd.compute_strain_designs(universe, sd_modules=[module], ki_cost=cost,
                                         solution_approach=BEST, max_solutions=1,
                                         compress=compress, skip_preprocessing_fvas=skip_fvas)
    assert solution.status == OPTIMAL
    design = solution.reaction_sd[0]
    kept = {r for r in cost if design.get(r)}
    assert sum(cost[r] for r in kept) == pytest.approx(-47.0)

    sub = _rebuild(universe, design, cost)
    ranges = flux_variability_analysis(sub, fraction_of_optimum=0.0)
    blocked = [r for r in ranges.index
               if max(abs(ranges.minimum[r]), abs(ranges.maximum[r])) < TOL]
    assert blocked == []


def test_thermodynamic_none_is_weaker_but_still_unblocked(universe, curr_solver):
    """Dropping the loopless block buys back reward -- a core reaction may then satisfy its
    must-run condition inside a cycle -- but must not leave the network blocked."""
    annotated, _, cost = _setup(universe)
    loose = sd.SDModule(universe, CARVEME, constraints=[BIO + ' >= 0.1'],
                        core_reactions=annotated, thermodynamic=None)
    design = _reconstruct(universe, [loose], cost, curr_solver).reaction_sd[0]
    kept = {r for r in cost if design.get(r)}
    assert sum(cost[r] for r in kept) <= -47.0, 'the looser problem cannot be worse'

    sub = _rebuild(universe, design, cost)
    ranges = flux_variability_analysis(sub, fraction_of_optimum=0.0)
    assert [r for r in ranges.index
            if max(abs(ranges.minimum[r]), abs(ranges.maximum[r])) < TOL] == []


def test_core_reaction_without_a_ki_cost_is_always_present(universe, capable_solver):
    """A core reaction the caller did not price is not a candidate: it is simply never absent,
    and its must-run condition is unconditional. It must not become a knockout candidate by way
    of the default ko_cost, which would contradict the must-run condition outright."""
    annotated, _, cost = _setup(universe)
    cost.pop('PGI')
    module = sd.SDModule(universe, CARVEME, constraints=[BIO + ' >= 0.1'],
                         core_reactions=annotated)
    solution = sd.compute_strain_designs(universe, sd_modules=[module], ki_cost=cost,
                                         solver=capable_solver, solution_approach=BEST,
                                         max_solutions=1, compress=False)
    assert solution.status == OPTIMAL
    design = solution.reaction_sd[0]
    assert 'PGI' not in design, 'an unpriced core reaction is not an intervention'
    sub = _rebuild(universe, design, cost)
    assert 'PGI' in {r.id for r in sub.reactions}


def test_core_reaction_may_not_be_a_knockout_candidate(universe, capable_solver):
    annotated, _, cost = _setup(universe)
    module = sd.SDModule(universe, CARVEME, constraints=[BIO + ' >= 0.1'],
                         core_reactions=annotated)
    with pytest.raises(Exception, match='knockout candidate'):
        sd.compute_strain_designs(universe, sd_modules=[module], ki_cost={},
                                  ko_cost={r: 1.0 for r in annotated}, solver=capable_solver,
                                  solution_approach=BEST, max_solutions=1, compress=False)


def test_rewarding_a_non_core_reaction_is_reported(universe, caplog, capable_solver):
    """Nothing requires a non-core reaction to carry flux, so a reward buys it whether or not it
    can run -- the very defect this module type removes for the core. It cannot be refused, since
    costs are the caller's to set, but it must not pass silently."""
    annotated, _, cost = _setup(universe)
    reversible = [r for r in annotated
                  if universe.reactions.get_by_id(r).lower_bound < 0 <
                  universe.reactions.get_by_id(r).upper_bound]
    assert reversible, 'need a rewarded reaction to leave out of the core'
    module = sd.SDModule(universe, CARVEME, constraints=[BIO + ' >= 0.1'],
                         core_reactions=[r for r in annotated if r not in reversible])
    import logging as _logging
    with caplog.at_level(_logging.WARNING):
        sd.compute_strain_designs(universe, sd_modules=[module], ki_cost=cost,
                                  solver=capable_solver, solution_approach=BEST,
                                  max_solutions=1, compress=False)
    assert any('not core reactions' in rec.message for rec in caplog.records)


def test_two_carveme_modules_share_the_binaries(universe, capable_solver):
    """Two reconstruction conditions, one set of interventions: the network must satisfy both."""
    annotated, _, cost = _setup(universe)
    modules = [sd.SDModule(universe, CARVEME, constraints=[BIO + ' >= 0.1'],
                           core_reactions=annotated),
               sd.SDModule(universe, CARVEME, constraints=['EX_ac_e >= 1'],
                           core_reactions=annotated)]
    solution = sd.compute_strain_designs(universe, sd_modules=modules, ki_cost=cost,
                                         solver=capable_solver, solution_approach=BEST,
                                         max_solutions=1, compress=False)
    assert solution.status == OPTIMAL
    design = solution.reaction_sd[0]
    sub = _rebuild(universe, design, cost)
    assert sub.slim_optimize() >= 0.1 - 1e-6
    with sub:
        sub.reactions.EX_ac_e.lower_bound = 1.0
        assert sub.slim_optimize() is not None


def test_cost_dictionaries_are_not_edited(universe, capable_solver):
    """The caller keeps its own dictionaries. Compression relabels reactions and gene handling
    renames genes, both of which used to reach into whatever was passed in -- so a second
    computation would run against costs the first one had rewritten."""
    annotated, _, cost = _setup(universe)
    module = sd.SDModule(universe, CARVEME, constraints=[BIO + ' >= 0.1'],
                         core_reactions=annotated)
    ki_cost = dict(cost)
    ko_cost = {r.id: 1.0 for r in universe.reactions
               if r.id not in cost and r.id != BIO}
    before_ki, before_ko = dict(ki_cost), dict(ko_cost)
    sd.compute_strain_designs(universe, sd_modules=[module], ki_cost=ki_cost, ko_cost=ko_cost,
                              solver=capable_solver, solution_approach=BEST, max_solutions=1,
                              compress=True)
    assert ki_cost == before_ki
    assert ko_cost == before_ko
