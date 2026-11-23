"""Random small problems, each solved by brute force, by the MILP path and by the LP engines.

The brute force of tests/lp_fuzz.py decides every intervention set within the budget by plain
LPs (HiGHS) and applies the definition of the result. The MILP path (CPLEX, where installed), the
LP engine inside compute_strain_designs (exact, and on the installed floating-point solvers) and
the LP engine on the network as it is have to return that result. The sample is fixed, four
problems per module type, picked for what there is to find in them; each goes to one of the
floating-point solvers, in turn, so that every solver sees a quarter of the sample. A long
campaign runs from the command line::

    python tests/lp_fuzz.py --seed 0 --n 2000 --out DIR
"""
import pytest
from . import lp_fuzz as F

# intervention sets per problem at most: keeps the brute force of the whole sample within seconds
MAX_SETS = 80
SEEDS = {
    'S': (140, 170, 180, 210),
    'SP': (101, 121, 161, 181),
    'SS': (102, 142, 182, 192),
    'nS': (113, 133, 163, 193),
    'nSP': (154, 174, 194, 204),
    'nP': (115, 145, 185, 205),
    'nSnP': (106, 126, 146, 166),
    'optknock': (157, 177, 187, 217),
    'robustknock': (108, 148, 158, 198),
    'optcouple': (119, 159, 199, 219),
}
SAMPLE = [(kind, seed) for kind in F.KINDS for seed in SEEDS[kind]]
_FLOAT = ('lp-glpk', 'lp-cplex', 'lp-gurobi', 'lp-scip')

# Open defects the sample runs into, (kind, seed) -> {backend: what is wrong}. A listed backend has
# to miss the reference (an entry that passes is to be removed), every other one has to meet it.
KNOWN = {
    ('S', 180): {b: "preprocessing of the compression: fast_reversibility raises KeyError: 'R3'" for b in ('milp-cplex',) + _FLOAT},
    ('SP', 121): {'milp-cplex': 'CPLEX error 3003 (not a mixed-integer problem) once the size-1 shortcut has taken every design: status error, designs right'},
    ('nP', 205): {'milp-cplex': "the size-1 shortcut returns ['UP0'], which breaks the PROTECT module with inner objective"},
}


@pytest.fixture(scope='module')
def backends():
    return F.installed_backends()


@pytest.mark.parametrize('kind, seed', SAMPLE, ids=['%s-%d' % ks for ks in SAMPLE])
def test_engines_return_the_brute_force_result(kind, seed, backends):
    problem = F.random_problem(seed, kind, max_sets=MAX_SETS)
    turn = _FLOAT[SEEDS[kind].index(seed)]
    out = F.compare(problem, [b for b in backends if not b.startswith('lp-') or b == turn])
    missed = {b: v for b, v in out['verdict'].items() if v not in ('ok', 'tie')}
    known = {b: why for b, why in KNOWN.get((kind, seed), {}).items() if b in out['verdict']}
    ref = out['reference']
    assert set(missed) == set(known), '%s seed %d, reference %s: %s' % (kind, seed, ref.get('designs', ref.get('value')), missed or
                                                                        'the open defect did not show: %s' % known)


def test_brute_force_on_a_problem_done_by_hand():
    """Two routes from substrate to product: the cut sets of the product are the pairs across."""
    problem = dict(rxns=['up', 'a1', 'a2', 'b', 'ex'], mets=['S', 'X', 'P'],
                   S=[[1, -1, 0, -1, 0], [0, 1, -1, 0, 0], [0, 0, 1, 1, -1]], lb=[0.0] * 5, ub=[10.0] * 5, gpr=[''] * 5, genes=[],
                   modules=[dict(module_type='suppress', constraints=[[{'ex': 1.0}, '>=', 1.0]])], max_cost=2)
    designs = F.BruteForce(problem).enumerate()
    assert sorted(sorted(z) for z in designs) == [['a1', 'b'], ['a2', 'b'], ['ex'], ['up']]
    # the optimum of the inner objective is taken without the module's constraints: with 'b' at
    # most half as fast as 'a1', growth-optimal fluxes (all through 'a') do not reach 'b >= 1'
    nested = dict(problem, modules=[dict(module_type='protect', inner_objective={'a2': 1.0}, constraints=[[{'b': 1.0}, '>=', 1.0]])], max_cost=1)
    bf = F.BruteForce(nested)
    assert not bf.mcs_ok(frozenset()) and bf.mcs_ok(frozenset(['a1'])) and bf.mcs_ok(frozenset(['a2']))
