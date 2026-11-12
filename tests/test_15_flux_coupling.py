"""Flux coupling analysis: coupling tables against a pairwise LP reference."""
import numpy as np
import pytest
from cobra import Metabolite, Model, Reaction
from scipy import sparse
from scipy.optimize import linprog
from ._models import load_test_model
import straindesign as sd

# =============================================================================
# Helpers / fixtures
# =============================================================================


def reference_table(model):
    """Coupling table by definition, with one LP per ordered pair (scipy HiGHS).

    i -> j iff no flux in the cone has r_j = 0 and r_i != 0. A mutually coupled pair is fully
    coupled iff r_j takes a single value when r_i is fixed to a feasible +-1."""
    ids = [r.id for r in model.reactions]
    n = len(ids)
    mets = {m.id: a for a, m in enumerate(model.metabolites)}
    S = sparse.lil_matrix((len(mets), n))
    for k, r in enumerate(model.reactions):
        for m, v in r.metabolites.items():
            S[mets[m.id], k] = float(v)
    S = S.tocsr()
    bnd = []
    for r in model.reactions:
        lb, ub = r.lower_bound, r.upper_bound
        bnd.append((0, 0) if lb == 0 and ub == 0 else (0, None) if lb >= 0 else (None, 0) if ub <= 0 else (None, None))

    def lp(fix, obj=None):
        b = list(bnd)
        for k, v in fix.items():
            b[k] = (v, v)
        c = np.zeros(n)
        if obj is not None:
            c[obj[0]] = obj[1]
        res = linprog(c, A_eq=S, b_eq=np.zeros(S.shape[0]), bounds=b, method='highs')
        assert res.status in (0, 2, 3), res.message
        return res

    signs = [[s for s in (1, -1) if (s > 0 and b[1] is None) or (s < 0 and b[0] is None)] for b in bnd]
    feasible_sign = [next((s for s in signs[i] if lp({i: s}).status == 0), None) for i in range(n)]
    U = [i for i in range(n) if feasible_sign[i] is not None]
    D = np.zeros((len(U), len(U)), bool)
    for a, i in enumerate(U):
        for b, j in enumerate(U):
            D[a, b] = a == b or all(lp({i: s, j: 0}).status == 2 for s in signs[i])
    T = np.where(D & D.T, 2, np.where(D, 3, np.where(D.T, 4, 0)))
    for a, b in zip(*np.nonzero(D & D.T)):
        i, j = U[a], U[b]
        lo, hi = lp({i: feasible_sign[i]}, (j, 1)), lp({i: feasible_sign[i]}, (j, -1))
        if lo.status == 0 and hi.status == 0 and abs(lo.fun + hi.fun) <= 1e-9 * max(1, abs(lo.fun)):
            T[a, b] = 1
    return [ids[i] for i in U], T, [ids[i] for i in range(n) if feasible_sign[i] is None]


def toy_model():
    """A -> A2 splits into two routes to B (partial coupling), a reversible reaction that can only
    run forwards (reversible source), a backward-only reaction, a dead end and a knocked-out
    reaction (blocked)."""
    model = Model('fca_toy')
    A, A2, B, C, D, E = (Metabolite(x) for x in 'A A2 B C D E'.split())
    reactions = [('EX_A', {A: 1}, 0, 1000), ('R_split', {A: -1, A2: 1}, 0, 1000), ('R_1', {A2: -1, B: 1}, 0, 1000),
                 ('R_2', {A2: -1, B: 2}, 0, 1000), ('EX_B', {B: -1}, 0, 1000), ('R_rev', {A: -1, C: 1}, -1000, 1000),
                 ('EX_C', {C: -1}, 0, 1000), ('R_back', {E: -1, A: 1}, -1000, 0), ('EX_E', {E: -1}, 0, 1000),
                 ('R_dead', {B: -1, D: 1}, 0, 1000), ('R_off', {A: -1, B: 1}, 0, 0)]
    for rid, stoich, lb, ub in reactions:
        r = Reaction(rid, lower_bound=lb, upper_bound=ub)
        model.add_reactions([r])
        r.add_metabolites(stoich)
    return model


@pytest.fixture(scope='module')
def ecc_reference():
    model = load_test_model('e_coli_core')
    return model, reference_table(model)


def assert_consistent(fc):
    T = fc.table
    assert np.all(np.diag(T) == 1)
    assert np.all((T == 3) == (T.T == 4))
    assert np.all((T == 1) == (T.T == 1)) and np.all((T == 2) == (T.T == 2))


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.parametrize('compress', [True, False])
def test_fca_e_coli_core_matches_reference(curr_solver, compress, ecc_reference):
    """The full table and blocked set equal the pairwise LP reference."""
    model, (reactions, table, blocked) = ecc_reference
    fc = sd.flux_coupling_analysis(model, compress=compress, solver=curr_solver)
    assert fc.reactions == reactions
    assert fc.blocked == blocked
    assert np.array_equal(fc.table, table)
    assert_consistent(fc)


@pytest.mark.parametrize('compress', [True, False])
def test_fca_toy_model(curr_solver, compress):
    """Hand-checked couplings, including a reversible source decided by the kernel test."""
    model = toy_model()
    fc = sd.flux_coupling_analysis(model, compress=compress, solver=curr_solver)
    code = lambda i, j: fc.table[fc.reactions.index(i), fc.reactions.index(j)]
    assert set(fc.blocked) == {'R_dead', 'R_off'}
    assert code('R_split', 'EX_B') == 2  # r_B / r_split ranges over [1, 2]
    assert code('R_1', 'R_split') == 3 and code('R_split', 'R_1') == 4
    assert code('R_1', 'R_2') == 0
    assert code('R_rev', 'EX_C') == 1  # reversible, but fully coupled with an irreversible reaction
    assert code('R_rev', 'EX_A') == 3 and code('EX_A', 'R_rev') == 4
    assert code('R_back', 'EX_E') == 1  # backward-only reaction, flipped
    assert code('EX_B', 'EX_A') == 3 and code('EX_A', 'EX_B') == 4
    reactions, table, blocked = reference_table(model)
    assert fc.reactions == reactions and fc.blocked == blocked
    assert np.array_equal(fc.table, table)
    assert_consistent(fc)


def test_fca_reaction_subset(curr_solver, ecc_reference):
    """A subset of reactions gives the corresponding part of the full table."""
    model, (reactions, table, blocked) = ecc_reference
    subset = [r.id for r in model.reactions][::3]
    fc = sd.flux_coupling_analysis(model, solver=curr_solver, reactions=subset)
    assert set(fc.reactions) | set(fc.blocked) == set(subset)
    assert fc.blocked == [r for r in blocked if r in subset]
    idx = [reactions.index(r) for r in fc.reactions]
    assert np.array_equal(fc.table, table[np.ix_(idx, idx)])


def test_fca_unknown_reaction():
    with pytest.raises(KeyError):
        sd.flux_coupling_analysis(toy_model(), reactions=['EX_A', 'nonexistent'])
