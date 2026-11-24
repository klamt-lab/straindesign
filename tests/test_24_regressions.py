"""Regression tests for defects found in the review before the release after 1.19.2."""
from .test_01_load_models_and_solvers import model_weak_coupling, model_gpr
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
    modules = [sd.SDModule(model, PROTECT, inner_objective='r_bm', constraints='r_px >= 1'), sd.SDModule(model, SUPPRESS, constraints='r1 >= 1')]
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
