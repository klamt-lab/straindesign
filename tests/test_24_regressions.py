"""Regression tests for defects found in the review before the release after 1.19.2."""
from .test_01_load_models_and_solvers import model_weak_coupling, model_gpr
import os
import subprocess
import sys
from numpy import inf
from scipy import sparse
from cobra import Model, Metabolite, Reaction
import pytest
import straindesign as sd
from straindesign.names import *
from straindesign.speedy_fva import fast_reversibility, speedy_fva


def _sets(solution):
    return sorted(sorted(d) for d in solution.reaction_sd)


def test_optknock_populate_returns_the_designs_of_best(curr_solver, model_weak_coupling):
    """In a bilevel problem a smaller design with a worse objective value is a result of its own.
    The check for non-minimal cut sets must not apply: it turned this run into status 'error'
    without designs."""
    modules = [sd.SDModule(model_weak_coupling, OPTKNOCK, inner_objective='r_BM', outer_objective='r_P', constraints=['r_BM >= 0.2'])]
    kw = dict(sd_modules=modules, max_cost=2, max_solutions=10, solver=curr_solver)
    best = sd.compute_strain_designs(model_weak_coupling, solution_approach=BEST, **kw)
    pop = sd.compute_strain_designs(model_weak_coupling, solution_approach=POPULATE, **kw)
    assert best.status == OPTIMAL and pop.status == OPTIMAL
    assert _sets(pop) == _sets(best)
    assert len(pop.reaction_sd) > 1


def test_reaction_bounds_of_selected_designs(model_weak_coupling):
    designs = [{'r1': -1.0}, {'r2': -1.0}, {'r4': -1.0, 'r5': -1.0}]
    sol = sd.SDSolutions(model_weak_coupling, designs, OPTIMAL, {MODEL_ID: model_weak_coupling.id, MAX_COST: 3})
    every = sol.get_reaction_sd_bnds()
    assert len(every) == 3
    assert sol.get_reaction_sd_bnds(1) == [every[1]]
    assert sol.get_reaction_sd_bnds([0, 2]) == [every[0], every[2]]


def test_cost_dicts_and_setup_of_the_caller_are_left_alone(model_gpr):
    modules = [sd.SDModule(model_gpr, SUPPRESS, constraints=["1.0 rd_ex >= 1.0 "])]
    modules += [sd.SDModule(model_gpr, PROTECT, constraints=[[{'r_bm': 1.0}, '>=', 1.0]])]
    ko_cost = {'rs_up': 1.0, 'rd_ex': 1.0, 'rp_ex': 1.1, 'r_bm': 0.75}
    gko_cost = {'g1': 1.0, 'g2': 1.0, 'g4': 3.0, 'g5': 2.0, 'g6': 1.0, 'g7': 1.0, 'g8': 1.0, 'g9': 1.0}
    gki_cost = {'g3': 1.0}
    setup = {MODULES: modules, MAX_COST: 2, MAX_SOLUTIONS: 1, SOLUTION_APPROACH: ANY, KOCOST: ko_cost, GKOCOST: gko_cost, GKICOST: gki_cost}
    before = (dict(ko_cost), dict(gko_cost), dict(gki_cost), set(setup))
    sd.compute_strain_designs(model_gpr, sd_setup=setup)
    assert (ko_cost, gko_cost, gki_cost, set(setup)) == before


@pytest.mark.parametrize('approach', [ANY, BEST])
def test_run_that_ends_on_the_clock_is_not_optimal(curr_solver, model_weak_coupling, approach):
    """A time limit that leaves no time for a single solve must not report a finished computation."""
    modules = [sd.SDModule(model_weak_coupling, SUPPRESS, constraints=['r_P <= 0.5', 'r_BM >= 0.1'])]
    sol = sd.compute_strain_designs(model_weak_coupling,
                                    sd_modules=modules,
                                    max_cost=3,
                                    solution_approach=approach,
                                    solver=curr_solver,
                                    time_limit=1e-9)
    assert sol.status in (TIME_LIMIT, TIME_LIMIT_W_SOL)


def _model_small_flux_ratio():
    """Flux cone in which N is a by-product of r_p at 1e-6 per unit, and r_p needs 59.91 S.

    N leaves through the export r_n (gene gE) or is recycled by r_d (gene gD), both at
    1e-6 / 59.91 = 1.7e-8 of the uptake flux. Growth (r_bm) therefore stops without the uptake
    (gU) or without both ways of removing N (gD and gE)."""
    model = Model('small_flux_ratio')
    mets = {m: Metabolite(m, compartment='c') for m in 'NSB'}
    for rid, stoich, lb, ub, gpr in [('r_n', {'N': 1}, -inf, 0, 'gE'), ('r_d', {'N': -1, 'S': 1}, 0, inf, 'gD'),
                                     ('r_p', {'N': 1e-6, 'S': -59.91, 'B': 1}, 0, inf, ''), ('r_up', {'S': 1}, 0, inf, 'gU'),
                                     ('r_bm', {'B': -1}, 0, inf, ''), ('r_b2', {'B': -1, 'S': 3}, 0, inf, 'gB')]:
        reaction = Reaction(rid, lower_bound=lb, upper_bound=ub)
        model.add_reactions([reaction])
        reaction.add_metabolites({mets[m]: v for m, v in stoich.items()})
        reaction.gene_reaction_rule = gpr
    return model


def test_lp_with_a_small_reduced_cost_is_unbounded(curr_solver):
    """min x0 with x0 + 1e-6 x1 = 0 and 59.91 x1 = x2 falls without limit along x2, by 1.7e-8 per
    unit. In the basis (x0, x1) that is the reduced cost of x2; a dual tolerance above it takes
    the basis for optimal and reports the value 0."""
    A_eq = sparse.csr_matrix([[1.0, 1e-6, 0.0], [0.0, 59.91, -1.0]])
    lp = sd.MILP_LP(c=[1.0, 0.0, 0.0], A_eq=A_eq, b_eq=[0.0, 0.0], lb=[-inf, 0.0, 0.0], ub=[0.0, inf, inf], solver=curr_solver)
    _, value, status = lp.solve()
    assert (status, value) == (UNBOUNDED, -inf)


def test_reversibility_keeps_a_direction_with_a_small_flux_ratio(curr_solver):
    """The two ways of removing N carry 1.7e-8 of the uptake flux. They are feasible directions
    and must not be reported as blocked."""
    reversibility = fast_reversibility(_model_small_flux_ratio(), solver=curr_solver)
    assert reversibility == {
        'r_n': (False, True),
        'r_d': (True, False),
        'r_p': (True, False),
        'r_up': (True, False),
        'r_bm': (True, False),
        'r_b2': (True, False)
    }


def test_gene_mcs_with_a_small_flux_ratio(curr_solver):
    """Blocking the removal of N in the reversibility pre-tightening takes growth out of the model
    before the cut sets are computed."""
    model = _model_small_flux_ratio()
    # GLPK has no indicator constraints, and the certificate of {gD, gE} needs multipliers of 1e6
    kw = {'M': 1e8} if curr_solver == GLPK else {}
    sol = sd.compute_strain_designs(model,
                                    sd_modules=[sd.SDModule(model, SUPPRESS, constraints='r_bm >= 1')],
                                    gene_kos=True,
                                    max_cost=3,
                                    solution_approach=BEST,
                                    solver=curr_solver,
                                    **kw)
    assert sol.status == OPTIMAL
    assert sorted(sorted(d) for d in sol.gene_sd) == [['gD', 'gE'], ['gU']]


def test_glpk_keeps_the_optimum_when_its_confirmation_stalls(monkeypatch):
    """GLPK re-solves an optimal LP at a tight dual tolerance, under an iteration limit. When that
    ends at the limit, the optimum of the default tolerance stands."""
    import swiglpk
    import straindesign.glpk_interface as glpk_interface
    simplex = glpk_interface.glp_simplex
    stalled = []

    def simplex_stalls_at_tight_tolerance(prob, params):
        if params.tol_dj < 1e-8:
            stalled.append(1)
            return swiglpk.GLP_EITLIM
        return simplex(prob, params)

    monkeypatch.setattr(glpk_interface, 'glp_simplex', simplex_stalls_at_tight_tolerance)
    lp = sd.MILP_LP(c=[1.0, 1.0, 1.0],
                    A_ineq=sparse.csr_matrix([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]),
                    b_ineq=[-1.0] * 3,
                    A_eq=sparse.csr_matrix((0, 3)),
                    b_eq=[],
                    lb=[0.0] * 3,
                    ub=[10.0] * 3,
                    solver=GLPK)
    x, value, status = lp.solve()
    assert stalled
    assert status == OPTIMAL
    assert value == pytest.approx(3.0)
    assert x == pytest.approx([1.0, 1.0, 1.0])


def _model_two_routes(genes=('x1', 'x2', 'x3'), names=('', '', '')):
    """A is made into B by r1 (first gene) or by r2 (second and third gene); r_bm drains B.

    r_bm stops without the first gene and one of the other two."""
    model = Model('two_routes')
    A, B = Metabolite('A', compartment='c'), Metabolite('B', compartment='c')
    for rid, stoich, ub, gpr in [('r_s', {A: 1}, 10, ''), ('r1', {A: -1, B: 1}, 1000, genes[0]),
                                 ('r2', {A: -1, B: 1}, 1000, genes[1] + ' and ' + genes[2]), ('r_bm', {B: -1}, 1000, '')]:
        reaction = Reaction(rid, lower_bound=0, upper_bound=ub)
        model.add_reactions([reaction])
        reaction.add_metabolites(stoich)
        reaction.gene_reaction_rule = gpr
    for gene, name in zip(genes, names):
        model.genes.get_by_id(gene).name = name
    return model


def _gene_mcs(model, **kwargs):
    modules = [sd.SDModule(model, SUPPRESS, constraints='r_bm >= 1')]
    sol = sd.compute_strain_designs(model, sd_modules=modules, max_cost=3, solution_approach=POPULATE, solver=kwargs.pop('solver', None), **kwargs)
    assert sol.status == OPTIMAL
    return sol


def _gene_sets(solution):
    return sorted(sorted(d) for d in solution.gene_sd)


def test_usable_gene_names_identify_the_genes():
    sol = _gene_mcs(_model_two_routes(names=('N1', 'N2', 'N3')), gene_kos=True)
    assert _gene_sets(sol) == [['N1', 'N2'], ['N1', 'N3']]


@pytest.mark.parametrize('names, reason', [(('N1', 'N', 'N'), 'shared by several genes'), (('r_bm', 'N2', 'N3'), 'also reaction IDs'),
                                           (('x2', 'N2', 'N3'), 'also IDs of other genes')])
def test_gene_names_that_cannot_be_identifiers_give_way_to_ids(curr_solver, caplog, names, reason):
    """Without a reference of the caller to a gene name, the genes are called by their IDs when the
    names are shared between genes or taken by a reaction or another gene."""
    with caplog.at_level('WARNING'):
        sol = _gene_mcs(_model_two_routes(names=names), gene_kos=True, solver=curr_solver)
    assert reason in caplog.text
    assert _gene_sets(sol) == [['x1', 'x2'], ['x1', 'x3']]
    assert sol.reaction_sd == [{'r1': -1.0, 'r2': -1.0}] * 2
    assert sol.sd_cost == [2.0, 2.0]


@pytest.mark.parametrize('names, reason', [(('N1', 'N', 'N'), 'shared by several genes'), (('r_bm', 'N', 'N3'), 'also reaction IDs')])
def test_reference_to_a_gene_name_that_cannot_be_an_identifier_fails_early(names, reason):
    model = _model_two_routes(names=names)
    with pytest.raises(Exception, match=reason):
        _gene_mcs(model, gko_cost={names[0]: 1.0, 'N': 1.0})
    # the same genes by ID
    assert _gene_sets(_gene_mcs(model, gko_cost={'x1': 1.0, 'x2': 1.0})) == [['x1', 'x2']]


def test_genes_referred_to_by_id_and_by_name_are_all_candidates():
    """With one gene named by its ID and the others by name, each of them is an intervention
    candidate. The pseudoreactions carry the names, also for the gene given by ID."""
    sol = _gene_mcs(_model_two_routes(names=('N1', 'N2', 'N3')), gko_cost={'x1': 1.0, 'N2': 2.0, 'N3': 1.0})
    assert _gene_sets(sol) == [['N1', 'N2'], ['N1', 'N3']]
    assert sorted(sol.sd_cost) == [2.0, 3.0]
    # a gene without a name is called by its ID next to named ones
    sol = _gene_mcs(_model_two_routes(names=('N1', 'N2', '')), gko_cost={'N1': 1.0, 'N2': 1.0, 'x3': 1.0})
    assert _gene_sets(sol) == [['N1', 'N2'], ['N1', 'x3']]


def test_regulatory_intervention_on_a_gene_referred_to_by_name():
    """A gene name in reg_cost selects the names as identifiers, like one in gko_cost."""
    sol = _gene_mcs(_model_two_routes(names=('N1', 'N2', 'N3')), gene_kos=True, reg_cost={'N1 <= 0.5': 1.0})
    assert sorted(sorted(k for k, v in d.items() if v) for d in sol.gene_sd) == [['N1', 'N2'], ['N1', 'N3'], ['N1 <= 0.5', 'N2'],
                                                                              ['N1 <= 0.5', 'N3']]


def test_gpr_extension_refuses_names_shared_between_genes():
    from straindesign.networktools import extend_model_gpr
    with pytest.raises(Exception, match="stand for several genes"):
        extend_model_gpr(_model_two_routes(names=('N1', 'N', 'N')), use_names=True)


@pytest.mark.parametrize('by_name', [False, True])
def test_costs_keyed_by_genes_that_start_with_a_digit(curr_solver, by_name):
    """Gene IDs and names that start with a digit get the prefix 'g'; the caller's cost keys and
    regulatory constraints follow, so the genes stay intervention candidates."""
    kw = dict(genes=('1x', '2x', '3x')) if not by_name else dict(names=('1n', '2n', '3n'))
    keys = ('1n', '2n', '3n') if by_name else ('1x', '2x', '3x')
    default = _gene_mcs(_model_two_routes(**kw), gene_kos=True, solver=curr_solver)
    keyed = _gene_mcs(_model_two_routes(**kw), gko_cost={k: 1.0 for k in keys}, solver=curr_solver)
    expected = [['g' + keys[0], 'g' + keys[1]], ['g' + keys[0], 'g' + keys[2]]]
    assert _gene_sets(default) == expected and _gene_sets(keyed) == expected
    # limiting the first gene does what knocking it out does
    limited = _gene_mcs(_model_two_routes(**kw), gko_cost={k: 1.0 for k in keys}, reg_cost={keys[0] + ' <= 0.5': 1.0}, solver=curr_solver)
    limit = 'g' + keys[0] + ' <= 0.5'
    assert sorted(sorted(k for k, v in d.items() if v) for d in limited.gene_sd) == sorted(expected + [sorted([limit, 'g' + keys[1]]),
                                                                                              sorted([limit, 'g' + keys[2]])])


def _model_dead_end():
    """r_s supplies A, r_a and r_b drain it; no two of these are coupled. r_dead turns A into D,
    which nothing consumes, so it is blocked."""
    model = Model('dead_end')
    A, D = Metabolite('A', compartment='c'), Metabolite('D', compartment='c')
    for rid, stoich, ub in [('r_s', {A: 1}, 10), ('r_a', {A: -1}, inf), ('r_b', {A: -1}, inf), ('r_dead', {A: -1, D: 1}, inf)]:
        reaction = Reaction(rid, lower_bound=0, upper_bound=ub)
        model.add_reactions([reaction])
        reaction.add_metabolites(stoich)
    return model


def test_reversibility_reports_a_blocked_reaction(curr_solver):
    """The compression inside fast_reversibility removes blocked reactions. They are part of the
    result also when the compression lumps no reactions."""
    assert fast_reversibility(_model_dead_end(), solver=curr_solver) == {
        'r_s': (True, False),
        'r_a': (True, False),
        'r_b': (True, False),
        'r_dead': (False, False)
    }


@pytest.mark.parametrize('constraints', [None, 'r_dead <= 5'])
def test_compressed_fva_reports_a_blocked_reaction(curr_solver, constraints):
    """As above for speedy_fva, also with a constraint on the reaction that is removed."""
    fva = speedy_fva(_model_dead_end(), solver=curr_solver, compress=True, constraints=constraints)
    assert list(fva.index) == ['r_s', 'r_a', 'r_b', 'r_dead']
    assert fva['minimum'].tolist() == pytest.approx([0.0] * 4)
    assert fva['maximum'].tolist() == pytest.approx([10.0, 10.0, 10.0, 0.0])


def test_cut_sets_of_a_network_with_a_blocked_reaction(curr_solver):
    model = _model_dead_end()
    sol = sd.compute_strain_designs(model,
                                    sd_modules=[sd.SDModule(model, SUPPRESS, constraints='r_a >= 1')],
                                    max_cost=2,
                                    solution_approach=POPULATE,
                                    solver=curr_solver)
    assert sol.status == OPTIMAL
    assert _sets(sol) == [['r_a'], ['r_s']]


def _model_supply_and_drains(drains=('r_a', 'r_b'), supply=inf):
    """r_s supplies A (at most `supply`), which the drains take out of the network."""
    model = Model('supply_and_drains')
    A = Metabolite('A', compartment='c')
    for rid, stoich, ub in [('r_s', {A: 1}, supply)] + [(d, {A: -1}, inf) for d in drains]:
        reaction = Reaction(rid, lower_bound=0, upper_bound=ub)
        model.add_reactions([reaction])
        reaction.add_metabolites(stoich)
    return model


@pytest.mark.parametrize('compress', [True, False])
def test_single_knockout_over_budget_is_no_design(curr_solver, compress):
    """Flux through r_a stops without r_s or without r_a. The knockout of r_a costs more than the
    budget; no reactions are lumped in this network."""
    model = _model_supply_and_drains()
    sol = sd.compute_strain_designs(model,
                                    sd_modules=[sd.SDModule(model, SUPPRESS, constraints='r_a >= 0.2')],
                                    ko_cost={
                                        'r_s': 1,
                                        'r_a': 3,
                                        'r_b': 1
                                    },
                                    max_cost=1,
                                    solution_approach=POPULATE,
                                    compress=compress,
                                    solver=curr_solver)
    assert _sets(sol) == [['r_s']]
    assert sol.sd_cost == [1.0]


def test_pool_of_a_problem_without_integer_variables(curr_solver):
    """The pool of an LP holds its optimum; the pool of an infeasible LP is empty."""
    kw = dict(c=[1.0, 1.0], A_ineq=sparse.csr_matrix([[-1.0, -1.0]]), lb=[0.0, 0.0], ub=[10.0, 10.0], solver=curr_solver)
    pool, value, status = sd.MILP_LP(b_ineq=[-1.0], **kw).populate(10)
    assert status == OPTIMAL
    assert value == pytest.approx(1.0)
    assert len(pool) == 1 and sum(pool[0]) == pytest.approx(1.0)
    pool, _, status = sd.MILP_LP(b_ineq=[-30.0], **kw).populate(10)
    assert (pool, status) == ([], INFEASIBLE)


@pytest.mark.parametrize('approach', [ANY, BEST, POPULATE])
def test_single_knockouts_that_leave_no_candidate_for_the_milp(curr_solver, approach):
    """r_s supplies A and r_a drains it. Each knockout alone stops r_a, both are found before the
    MILP is built, and the MILP is left without an intervention variable."""
    model = _model_supply_and_drains(drains=('r_a',))
    sol = sd.compute_strain_designs(model,
                                    sd_modules=[sd.SDModule(model, SUPPRESS, constraints='r_a >= 0.1')],
                                    max_cost=1,
                                    solution_approach=approach,
                                    solver=curr_solver)
    assert sol.status == OPTIMAL
    assert _sets(sol) == [['r_a'], ['r_s']]


def _model_product_with_and_without_growth():
    """r_s supplies A (at most 10). r1 turns A into B and the product P, r2 into B alone and r3
    into P alone; r_bm drains B (growth) and r_px drains P."""
    model = Model('product_with_and_without_growth')
    A, B, P = (Metabolite(m, compartment='c') for m in 'ABP')
    for rid, stoich, ub in [('r_s', {A: 1}, 10), ('r1', {A: -1, B: 1, P: 1}, inf), ('r2', {A: -1, B: 1}, inf), ('r3', {A: -1, P: 1}, inf),
                            ('r_bm', {B: -1}, inf), ('r_px', {P: -1}, inf)]:
        reaction = Reaction(rid, lower_bound=0, upper_bound=ub)
        model.add_reactions([reaction])
        reaction.add_metabolites(stoich)
    return model


@pytest.mark.parametrize('approach', [BEST, POPULATE])
def test_single_knockout_has_to_leave_an_optimal_flux_in_the_protected_region(curr_solver, approach):
    """PROTECT: r_px >= 1 is attainable at maximal growth. SUPPRESS: r1 >= 1.

    Without r1 no flux meets r1 >= 1, and r_px >= 1 stays attainable through r3, but not at
    maximal growth, which then runs through r2 alone: {r1} is no design. Without r1 and r2
    nothing grows, every flux is growth-optimal and r3 makes the product."""
    model = _model_product_with_and_without_growth()
    modules = [
        sd.SDModule(model, PROTECT, inner_objective='r_bm', constraints='r_px >= 1'),
        sd.SDModule(model, SUPPRESS, constraints='r1 >= 1')
    ]
    sol = sd.compute_strain_designs(model,
                                    sd_modules=modules,
                                    ko_cost={
                                        'r1': 1,
                                        'r2': 1,
                                        'r3': 1
                                    },
                                    max_cost=2,
                                    solution_approach=approach,
                                    solver=curr_solver)
    assert sol.status == OPTIMAL
    assert _sets(sol) == [['r1', 'r2']]


def test_no_single_knockout_next_to_the_empty_design(curr_solver):
    """SUPPRESS r_b >= 1 among the fluxes that maximise r_a: these have r_b = 0, so no intervention
    is needed. Without r_s or r_b no flux at all meets r_b >= 1; these knockouts are not minimal."""
    model = _model_supply_and_drains(supply=10)
    sol = sd.compute_strain_designs(model,
                                    sd_modules=[sd.SDModule(model, SUPPRESS, inner_objective='r_a', constraints='r_b >= 1')],
                                    max_cost=1,
                                    solution_approach=POPULATE,
                                    solver=curr_solver)
    assert sol.status == OPTIMAL
    assert _sets(sol) == [[]]


@pytest.mark.parametrize('forced', ['r_p', 'r_s'])
@pytest.mark.parametrize('compress', [False, True])
def test_optcouple_without_a_flux_that_makes_no_product(curr_solver, forced, compress):
    """OptCouple compares growth with and without production. r_s supplies A and the product
    reaction r_p is its only drain; r_t and r_bm (growth) are separate. A lower bound of 2 on r_p
    or on r_s leaves no flux without production, so no design qualifies."""
    model = Model('forced_product')
    A, B = Metabolite('A', compartment='c'), Metabolite('B', compartment='c')
    for rid, stoich in [('r_s', {A: 1}), ('r_p', {A: -1}), ('r_t', {B: 1}), ('r_bm', {B: -1})]:
        reaction = Reaction(rid, lower_bound=2 if rid == forced else 0, upper_bound=10)
        model.add_reactions([reaction])
        reaction.add_metabolites(stoich)
    sol = sd.compute_strain_designs(model,
                                    sd_modules=[sd.SDModule(model, OPTCOUPLE, inner_objective='r_bm', prod_id='r_p')],
                                    max_cost=1,
                                    solution_approach=BEST,
                                    compress=compress,
                                    solver=curr_solver)
    assert sol.status == INFEASIBLE
    assert sol.reaction_sd == []


def _network(name, reactions):
    """Model from (id, {metabolite: coefficient}, lower bound, upper bound) tuples."""
    model = Model(name)
    metabolites = {}
    for rid, stoich, lb, ub in reactions:
        reaction = Reaction(rid, lower_bound=lb, upper_bound=ub)
        model.add_reactions([reaction])
        reaction.add_metabolites({metabolites.setdefault(m, Metabolite(m, compartment='c')): v for m, v in stoich.items()})
    return model


def _problem_without_candidates(case, min_gcp=0.0):
    """Optimisation problems in which the wild type qualifies and no intervention variable is left."""
    if case == 'every reaction essential':
        # r_s makes A and B, growth drains A and the product reaction B: the three are coupled
        model = _network(case, [('r_s', {'A': 1, 'B': 1}, 0, 10), ('r_bm', {'A': -1}, 0, inf), ('r_p', {'B': -1}, 0, inf)])
        modules = [sd.SDModule(model, OPTCOUPLE, inner_objective='r_bm', prod_id='r_p', min_gcp=min_gcp)]
        return model, modules + [sd.SDModule(model, PROTECT, constraints='r_p >= 0.2')], {}
    if case == 'the candidate left is blocked':
        # nothing makes B, which growth needs
        model = _network(case, [('r_s', {'A': 1}, 0, inf), ('r_p', {'A': -1}, 0, 1000), ('r_bm', {'A': -1, 'B': -1}, 0, 1000)])
        modules = [sd.SDModule(model, ROBUSTKNOCK, inner_objective='r_bm', outer_objective='r_p')]
        return model, modules + [sd.SDModule(model, PROTECT, constraints='r_p >= 1')], {}
    model = _network(case, [('r_s', {'A': 1}, 0, 10), ('r_p', {'A': -1}, 0, inf), ('r_bm', {'A': -1}, 0, inf)])
    return model, [sd.SDModule(model, OPTKNOCK, inner_objective='r_bm', outer_objective='r_p')], {'ko_cost': {}}


@pytest.mark.parametrize('approach', [ANY, BEST, POPULATE])
@pytest.mark.parametrize('case', ['every reaction essential', 'the candidate left is blocked', 'no candidate given'])
def test_optimisation_without_candidates_returns_the_wild_type(curr_solver, case, approach):
    model, modules, kw = _problem_without_candidates(case)
    sol = sd.compute_strain_designs(model,
                                    sd_modules=modules,
                                    max_cost=1,
                                    solution_approach=approach,
                                    compress=False,
                                    solver=curr_solver,
                                    **kw)
    assert sol.status == OPTIMAL
    assert sol.reaction_sd == [{}]


@pytest.mark.parametrize('approach', [ANY, BEST, POPULATE])
def test_optimisation_without_candidates_and_without_a_design(curr_solver, approach):
    """Growth is 10 with production and 0 without it; a coupling potential of 20 is out of reach."""
    model, modules, kw = _problem_without_candidates('every reaction essential', min_gcp=20.0)
    sol = sd.compute_strain_designs(model,
                                    sd_modules=modules,
                                    max_cost=1,
                                    solution_approach=approach,
                                    compress=False,
                                    solver=curr_solver,
                                    **kw)
    assert sol.status == INFEASIBLE
    assert sol.reaction_sd == []


def _model_reversible_reaction_with_a_rule():
    """r_sa supplies A and r_sb supplies B (at most 10 each), r_a and r_b drain them. r_x (gene gX)
    converts A into B and back."""
    model = _network('reversible_with_rule', [('r_sa', {'A': 1}, 0, 10), ('r_sb', {'B': 1}, 0, 10), ('r_x', {'A': -1, 'B': 1}, -inf, inf),
                                              ('r_a', {'A': -1}, 0, inf), ('r_b', {'B': -1}, 0, inf)])
    model.reactions.r_x.gene_reaction_rule = 'gX'
    return model


@pytest.mark.parametrize('constraint', ['r_a - r_sa >= 1', 'r_b - r_sb >= 1'], ids=['backward', 'forward'])
@pytest.mark.parametrize('compress', [False, True])
def test_knockout_of_a_reversible_reaction_that_has_a_gene_rule(curr_solver, constraint, compress):
    """The reaction r_x and its gene are both knockout candidates. More drain of A than supply needs
    r_x backward, more drain of B than supply needs it forward: either knockout stops both."""
    model = _model_reversible_reaction_with_a_rule()
    sol = sd.compute_strain_designs(model,
                                    sd_modules=[sd.SDModule(model, SUPPRESS, constraints=constraint)],
                                    gene_kos=True,
                                    ko_cost={'r_x': 1},
                                    max_cost=1,
                                    solution_approach=POPULATE,
                                    compress=compress,
                                    solver=curr_solver)
    assert sol.status == OPTIMAL
    assert _gene_sets(sol) == [['gX'], ['r_x']]
    assert sol.reaction_sd == [{'r_x': -1.0}] * 2


@pytest.mark.parametrize('constraint', ['r_a - r_sa >= 1', 'r_b - r_sb >= 1'], ids=['backward', 'forward'])
@pytest.mark.parametrize('compress', [False, True])
def test_addition_of_a_reversible_reaction_that_has_a_gene_rule(curr_solver, constraint, compress):
    """r_x is an addition candidate. Neither direction is available without the addition."""
    model = _model_reversible_reaction_with_a_rule()
    sol = sd.compute_strain_designs(model,
                                    sd_modules=[sd.SDModule(model, PROTECT, constraints=constraint)],
                                    gene_kos=True,
                                    ki_cost={'r_x': 1},
                                    max_cost=1,
                                    solution_approach=POPULATE,
                                    compress=compress,
                                    solver=curr_solver)
    assert sol.status == OPTIMAL
    assert sol.reaction_sd == [{'r_x': 1.0}]


@pytest.mark.parametrize('module', [
    dict(constraints='r_a + r_dead >= 1'),
    dict(constraints=['r_a >= 1', 'r_dead <= 5']),
    dict(constraints='r_a >= 1', inner_objective='r_dead'),
],
                         ids=['next to another reaction', 'alone in a constraint', 'as inner objective'])
def test_module_that_names_a_blocked_reaction(curr_solver, module):
    """The compression removes the blocked r_dead. Its terms in the module are zero, and the cut
    sets are those of r_a >= 1."""
    model = _model_dead_end()
    sol = sd.compute_strain_designs(model,
                                    sd_modules=[sd.SDModule(model, SUPPRESS, **module)],
                                    max_cost=2,
                                    solution_approach=POPULATE,
                                    solver=curr_solver)
    assert sol.status == OPTIMAL
    assert _sets(sol) == [['r_a'], ['r_s']]


@pytest.mark.parametrize('blocked_in', [OUTER_OBJECTIVE, INNER_OBJECTIVE, CONSTRAINTS])
def test_optknock_module_that_names_a_blocked_reaction(curr_solver, model_weak_coupling, blocked_in):
    """r_dead turns A into a metabolite that nothing consumes. Adding it to an objective or to
    the constraint of the module does not change the designs."""
    dead_end = Metabolite('X', compartment='c')
    r_dead = Reaction('r_dead', lower_bound=0, upper_bound=100)
    model_weak_coupling.add_reactions([r_dead])
    r_dead.add_metabolites({model_weak_coupling.metabolites.A: -1, dead_end: 1})
    module = {INNER_OBJECTIVE: 'r_BM', OUTER_OBJECTIVE: 'r_P', CONSTRAINTS: 'r_BM >= 0.2'}
    ko_cost = {r.id: 1 for r in model_weak_coupling.reactions if r.id != 'r_dead'}
    kw = dict(ko_cost=ko_cost, max_cost=2, max_solutions=10, solution_approach=BEST, solver=curr_solver)
    plain = sd.compute_strain_designs(model_weak_coupling, sd_modules=[sd.SDModule(model_weak_coupling, OPTKNOCK, **module)], **kw)
    module[blocked_in] = module[blocked_in].replace('r_BM', 'r_BM + r_dead').replace('r_P', 'r_P + r_dead')
    named = sd.compute_strain_designs(model_weak_coupling, sd_modules=[sd.SDModule(model_weak_coupling, OPTKNOCK, **module)], **kw)
    assert plain.status == OPTIMAL and named.status == OPTIMAL
    assert len(plain.reaction_sd) > 0
    assert _sets(named) == _sets(plain)


def _model_two_supplies():
    """r_s1 (at most 10) and r_s2 (at most 5) supply A. Growth (r_bm) turns A into the product P,
    which r_p exports; r_q drains A unused."""
    return _network('two_supplies', [('r_s1', {'A': 1}, 0, 10), ('r_s2', {'A': 1}, 0, 5), ('r_bm', {'A': -1, 'P': 1}, 0, inf),
                                     ('r_p', {'P': -1}, 0, inf), ('r_q', {'A': -1}, 0, inf)])


@pytest.mark.parametrize('compress', [False, True])
def test_populate_moves_on_from_a_wild_type_that_a_rewarded_knockout_dominates(curr_solver, compress):
    """The wild type exports the most product (15). The knockout of r_s2 is rewarded and lowers
    the export to 10, so the pool at 15 holds the wild type alone, and the enumeration sets it
    aside for the cheaper design that contains it. The wild type then has to leave the MILP, or
    the same pool comes back until the time limit."""
    model = _model_two_supplies()
    kw = dict(sd_modules=[sd.SDModule(model, OPTKNOCK, inner_objective='r_bm', outer_objective='r_p')],
              ko_cost={'r_s2': -1, 'r_q': 1},
              max_cost=1,
              compress=compress,
              solver=curr_solver,
              time_limit=10)
    best = sd.compute_strain_designs(model, solution_approach=BEST, **kw)
    pop = sd.compute_strain_designs(model, solution_approach=POPULATE, **kw)
    assert best.status == OPTIMAL and pop.status == OPTIMAL
    assert len(pop.reaction_sd) > 0
    assert _sets(pop) == _sets(best)


@pytest.mark.parametrize('costs, designs', [(dict(ko_cost={'r_s2': 1, 'r_q': 1}), [{'r_q': -1}, {'r_s2': -1}]),
                                            (dict(ko_cost={'r_q': 1}, ki_cost={'r_s2': 1}), [{'r_q': -1, 'r_s2': 0}, {'r_s2': 1}]),
                                            (dict(ko_cost={}), [])],
                         ids=['knockouts', 'knockout and addition', 'no candidate'])
@pytest.mark.parametrize('approach', [ANY, BEST])
def test_wild_type_that_fails_verification_is_left_behind(curr_solver, monkeypatch, approach, costs, designs):
    """A design that the solver returns and the verification rejects is taken out of the MILP.
    Here the verification rejects the strain without interventions, as it does for a point
    that the solver accepts within its tolerances only. The search has to go on to the designs
    with one intervention, or end where there is no candidate, instead of asking for the same
    point until the time limit."""
    if curr_solver == SCIP and designs:
        pytest.skip('SCIP drops a row whose variables are all fixed, here the row that takes the wild type out')
    from straindesign.strainDesignMILP import SDMILP
    verify = SDMILP.verify_sd
    monkeypatch.setattr(SDMILP, 'verify_sd', lambda self, sols: [ok and sols[i].nnz > 0 for i, ok in enumerate(verify(self, sols))])
    model = _model_two_supplies()
    sol = sd.compute_strain_designs(model,
                                    sd_modules=[sd.SDModule(model, OPTKNOCK, inner_objective='r_bm', outer_objective='r_p')],
                                    max_cost=1,
                                    solution_approach=approach,
                                    compress=False,
                                    solver=curr_solver,
                                    time_limit=10,
                                    **costs)
    assert sol.status == (OPTIMAL if designs else INFEASIBLE)
    assert sorted(sol.reaction_sd, key=lambda design: sorted(design.items())) == designs


def _model_coefficients_of_two():
    """Twelve reactions with stoichiometric coefficients of 1 and 2; R6 is blocked (nothing consumes M3).

    Without R2 or without UP0 no flux state with an optimal BM has UP0 >= 0.1 and EX_P <= UP0 / 2,
    and BM >= 0.2 stays within reach."""
    model = Model('coefficients_of_two')
    mets = {m: Metabolite(m, compartment='c') for m in ['M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6']}
    # the metabolites go in first: their order is the row order of the MILP, and the rounding
    # error that the test below is about appears with this order
    model.add_metabolites(list(mets.values()))
    for rid, stoich, lb, ub in [('UP0', {'M0': 1}, 0, 10), ('R0', {'M0': -1, 'M4': 2}, -inf, inf), ('R1', {'M0': 1, 'M4': -2, 'M5': 1}, -inf, inf),
                                ('R2', {'M2': 1, 'M4': -2, 'M5': -1}, 0, inf), ('R3', {'M1': 1, 'M2': -1, 'M6': 2}, 0, 1000),
                                ('R4', {'M1': -2, 'M5': 2, 'M6': -1}, 0, inf), ('R5', {'M0': -1, 'M1': 2, 'M4': 1}, 0, inf),
                                ('R6', {'M2': 2, 'M3': 2, 'M5': -2}, 0, 1000), ('R7', {'M2': -1, 'M4': -2, 'M6': 1}, -inf, inf),
                                ('EX_P', {'M0': -1}, 0, 10), ('EX1', {'M6': -1}, 0, inf), ('BM', {'M2': -2, 'M4': -1, 'M6': -2}, 0, inf)]:
        reaction = Reaction(rid, lower_bound=lb, upper_bound=ub)
        model.add_reactions([reaction])
        reaction.add_metabolites({mets[m]: v for m, v in stoich.items()})
    return model


@pytest.mark.parametrize('compress', [True, False])
def test_cut_sets_behind_a_binary_that_is_integral_up_to_rounding(curr_solver, compress):
    """Both minimal cut sets cost 2. In the MILP of that cost level, GLPK meets a binary whose LP
    value is an integer up to the rounding error of the simplex (about 1e-12). An integrality
    tolerance below that error makes GLPK branch on it and close the branch that holds the cut
    sets, so that only supersets of them are returned."""
    model = _model_coefficients_of_two()
    modules = [
        sd.SDModule(model, SUPPRESS, constraints=['EX_P - 0.5 UP0 <= 0', 'UP0 >= 0.1'], inner_objective='BM'),
        sd.SDModule(model, PROTECT, constraints='BM >= 0.2', inner_objective='BM', inner_opt_tol=0.9)
    ]
    ko_cost = {'UP0': 2, 'R0': 1, 'R1': 1, 'R2': 2, 'R3': 1, 'R4': 3, 'R5': 2, 'R6': 1, 'R7': 3, 'EX_P': 2, 'EX1': 3, 'BM': 1}
    sol = sd.compute_strain_designs(model,
                                    sd_modules=modules,
                                    ko_cost=ko_cost,
                                    max_cost=5,
                                    solution_approach=POPULATE,
                                    compress=compress,
                                    solver=curr_solver)
    assert sol.status == OPTIMAL
    assert _sets(sol) == [['R2'], ['UP0']]


@pytest.mark.parametrize('compress', [True, False])
def test_reaction_candidate_parallel_to_a_reaction_with_a_gene_rule(curr_solver, compress):
    """R1, EX_P and BM drain the same metabolite. R1 and EX_P are reaction candidates (cost 2 each),
    BM is switched by its gene (cost 1). The uptake stops only when all three are off: cost 5."""
    model = _network('parallel_drains', [('UP0', {'M0': 1}, 0, inf), ('R1', {'M0': -1}, 0, inf), ('EX_P', {'M0': -1}, 0, inf),
                                         ('BM', {'M0': -1}, 0, inf)])
    model.reactions.BM.gene_reaction_rule = 'g0'
    kw = dict(sd_modules=[sd.SDModule(model, SUPPRESS, constraints=['EX_P - UP0 <= 0', 'UP0 >= 0.5'])],
              gene_kos=True,
              ko_cost={
                  'R1': 2,
                  'EX_P': 2
              },
              solution_approach=POPULATE,
              compress=compress,
              solver=curr_solver)
    sol = sd.compute_strain_designs(model, max_cost=4, **kw)
    assert sol.status == INFEASIBLE and _gene_sets(sol) == []
    sol = sd.compute_strain_designs(model, max_cost=5, **kw)
    assert sol.status == OPTIMAL and _gene_sets(sol) == [['EX_P', 'R1', 'g0']]


@pytest.mark.parametrize('compress', [True, False])
def test_network_in_which_no_reaction_can_carry_flux(curr_solver, compress):
    """Nothing produces M0, so every reaction is blocked and the strain without interventions is
    the only candidate: it is the best design of an optimisation problem, and it cannot empty a
    region that holds the zero flux."""
    reactions = [('EX_P', {'M0': -1}, 0, 1000), ('BM', {'M0': -1}, 0, 1000), ('R1', {'M0': -1, 'M1': 1}, -1000, 1000), ('EX1', {'M1': -1}, 0, 1000)]
    model = _network('no_flux', reactions)
    sol = sd.compute_strain_designs(model,
                                    sd_modules=[sd.SDModule(model, OPTKNOCK, inner_objective='BM', outer_objective='EX_P')],
                                    max_cost=2,
                                    solution_approach=BEST,
                                    compress=compress,
                                    solver=curr_solver)
    assert sol.status == OPTIMAL and sol.reaction_sd == [{}]
    model = _network('no_flux', reactions)
    sol = sd.compute_strain_designs(model,
                                    sd_modules=[sd.SDModule(model, SUPPRESS, constraints='BM <= 1')],
                                    max_cost=2,
                                    solution_approach=POPULATE,
                                    compress=compress,
                                    solver=curr_solver)
    assert sol.status == INFEASIBLE and sol.reaction_sd == []


def test_glpk_milp_whose_bounds_grow_without_limit_in_the_presolver():
    """The first row forces z = 1, and then x >= 5 y + 1 and y >= x have no solution. GLPK's MIP
    presolver raises the lower bounds of x and y in turns; neither has an upper bound, so nothing
    stops it until a bound overflows and GLPK ends the process. The solve runs in a process of
    its own for that reason."""
    code = '\n'.join([
        "from scipy import sparse", "from straindesign import MILP_LP", "from straindesign.names import *",
        "A = sparse.csr_matrix([[-2, 0, 0], [0, -1, 5], [1000, 1, -1]])",
        "milp = MILP_LP(c=[1, 0, 0], A_ineq=A, b_ineq=[-1, -1, 1000], lb=[0, 0, 0], ub=[1, float('inf'), float('inf')], vtype='BCC', solver=GLPK)",
        "print('status', milp.solve()[2])"
    ])
    run = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=120, cwd=os.path.dirname(os.path.dirname(sd.__file__)))
    assert run.returncode == 0, run.stderr[-300:]
    assert run.stdout.split()[-1] == INFEASIBLE


def _make_chain_model():
    model = Model("chain")
    met_a = Metabolite("A")
    r_in = Reaction("r_in", lower_bound=2.0, upper_bound=10.0)
    r_in.add_metabolites({met_a: 1.0})
    r_p = Reaction("r_p", lower_bound=0.0, upper_bound=10.0)
    r_p.add_metabolites({met_a: -1.0})
    model.add_reactions([r_in, r_p])
    return model


def _make_branch_model(in_bounds=(1.0, 10.0), b_bounds=(0.0, 5.0), c_bounds=(0.0, 5.0)):
    model = Model("branch")
    met_a = Metabolite("A")
    r_in = Reaction("r_in", lower_bound=in_bounds[0], upper_bound=in_bounds[1])
    r_in.add_metabolites({met_a: 1.0})
    r_b = Reaction("r_b", lower_bound=b_bounds[0], upper_bound=b_bounds[1])
    r_b.add_metabolites({met_a: -1.0})
    r_c = Reaction("r_c", lower_bound=c_bounds[0], upper_bound=c_bounds[1])
    r_c.add_metabolites({met_a: -1.0})
    model.add_reactions([r_in, r_b, r_c])
    return model


def test_protect_inner_minimize(curr_solver):
    model = _make_chain_model()
    mod = sd.SDModule(
        model,
        PROTECT,
        inner_objective="r_in",
        inner_opt_sense=MINIMIZE,
        constraints="r_p <= 3",
    )
    sol = sd.compute_strain_designs(model, sd_modules=[mod], ko_cost={}, solver=curr_solver, compress=False)
    assert sol.status == OPTIMAL
    assert sol.reaction_sd == [{}]


def test_suppress_inner_minimize(curr_solver):
    model = _make_chain_model()
    mod = sd.SDModule(
        model,
        SUPPRESS,
        inner_objective="r_in",
        inner_opt_sense=MINIMIZE,
        constraints="r_p >= 5",
    )
    sol = sd.compute_strain_designs(model, sd_modules=[mod], ko_cost={}, solver=curr_solver, compress=False)
    assert sol.status == OPTIMAL
    assert sol.reaction_sd == [{}]


def test_optknock_inner_minimize(curr_solver):
    model = _make_branch_model()
    mod = sd.SDModule(
        model,
        OPTKNOCK,
        inner_objective="r_in",
        inner_opt_sense=MINIMIZE,
        outer_objective="r_b",
        outer_opt_sense=MAXIMIZE,
    )
    milp = sd.SDMILP(model, [mod], ko_cost={}, solver=curr_solver)
    _, obj, status = milp.solve()
    assert status == OPTIMAL
    assert obj == pytest.approx(-1.0, abs=1e-6)


def test_robustknock_inner_minimize(curr_solver):
    model = _make_branch_model(in_bounds=(1.0, 10.0), b_bounds=(0.0, 10.0), c_bounds=(0.0, 10.0))
    mod = sd.SDModule(
        model,
        ROBUSTKNOCK,
        inner_objective="r_in",
        inner_opt_sense=MINIMIZE,
        outer_objective="r_b",
        outer_opt_sense=MAXIMIZE,
    )
    milp = sd.SDMILP(model, [mod], ko_cost={"r_c": 1}, solver=curr_solver)
    _, obj, status = milp.solve()
    assert status == OPTIMAL
    assert obj == pytest.approx(-1.0, abs=1e-6)


def test_robustknock_outer_minimize(curr_solver):
    model = _make_branch_model(in_bounds=(1.0, 10.0), b_bounds=(2.0, 10.0), c_bounds=(0.0, 8.0))
    mod = sd.SDModule(
        model,
        ROBUSTKNOCK,
        inner_objective="r_in",
        inner_opt_sense=MAXIMIZE,
        outer_objective="r_b",
        outer_opt_sense=MINIMIZE,
    )
    milp = sd.SDMILP(model, [mod], ko_cost={}, solver=curr_solver)
    _, obj, status = milp.solve()
    assert status == OPTIMAL
    assert obj == pytest.approx(10.0, abs=1e-6)


def test_doubleopt_minimize(curr_solver):
    model = _make_chain_model()
    mod = sd.SDModule(
        model,
        DOUBLEOPT,
        inner_objective="r_in",
        inner_opt_sense=MINIMIZE,
        outer_objective="r_p",
        outer_opt_sense=MINIMIZE,
        constraints="r_p <= 3",
    )
    sol = sd.compute_strain_designs(model, sd_modules=[mod], ko_cost={}, solver=curr_solver, compress=False)
    assert sol.status == OPTIMAL
    assert sol.reaction_sd == [{}]


def test_protect_outer_minimize(curr_solver):
    model = _make_branch_model(in_bounds=(1.0, 6.0), b_bounds=(0.0, 5.0), c_bounds=(0.0, 5.0))
    mod = sd.SDModule(
        model,
        PROTECT,
        inner_objective="r_in",
        inner_opt_sense=MAXIMIZE,
        outer_objective="r_b",
        outer_opt_sense=MINIMIZE,
    )
    milp = sd.SDMILP(model, [mod], ko_cost={}, solver=curr_solver)
    x, _, status = milp.solve()
    assert status == OPTIMAL
    assert x[1] == pytest.approx(1.0, abs=1e-6)
