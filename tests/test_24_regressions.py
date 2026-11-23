"""Regression tests for defects found in the review before the release after 1.19.2."""
from .test_01_load_models_and_solvers import model_weak_coupling, model_gpr
from numpy import inf
from scipy import sparse
from cobra import Model, Metabolite, Reaction
import pytest
import straindesign as sd
from straindesign.names import *
from straindesign.speedy_fva import fast_reversibility


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
