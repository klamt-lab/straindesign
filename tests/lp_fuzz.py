"""Brute-force reference and random problems for the strain design engines (helper, no test).

A *problem* is a plain dict (see :func:`random_problem`): a small network given by its
stoichiometric matrix and bounds, GPR rules, strain design modules, intervention costs and a
budget. Three things can be done with it:

- :class:`BruteForce` decides every intervention set within the budget by plain LPs on the
  problem's own matrices and applies the definition of the result. It shares no code with the
  package: the rules are evaluated with Python's ``and`` / ``or``, the LPs go to HiGHS (through
  scipy) or to gurobipy directly.
- :func:`run_package` hands the same problem to ``compute_strain_designs`` (MILP path or an LP
  engine) and brings the result into the same form; :func:`run_engine_alone` gives it to the LP
  engine directly, on the network as it is (the backends ``raw-...``).
- :func:`compare` runs both and names every backend whose result is not the reference, telling
  *ties* (the verdict of an intervention set changes when a threshold moves by ``DELTA``) from
  mismatches.

What the brute force takes as the definition of a result:

- ``P(z)``: the fluxes with ``S v = 0`` and ``lb <= v <= ub`` whose knocked-out reactions carry
  no flux. A gene knockout switches off the reactions whose rule becomes false. Knocking out a
  reaction whose bounds exclude zero leaves ``P(z)`` empty.
- A SUPPRESS module asks for an empty region, a PROTECT module for a non-empty one. Without inner
  objective the region is ``P(z)`` with the module's constraints. With inner objective ``c`` it
  is ``{v in P(z) : constraints, c.v >= tol * max c over P(z)}`` (the maximum without the
  constraints; ``c.v <= tol * min`` for a minimised objective).
- Enumeration: the acceptable sets in order of cost, each one excluding its supersets; the empty
  set alone if it is acceptable.
- OptKnock: the largest ``max {o.v : v in P(z), constraints, c.v >= tol * max c over P(z)}``.
  RobustKnock: the largest ``min {o.v : v in P(z), c.v >= max c over P(z)}`` over the z for which
  some flux of ``P(z)`` meets the constraints. OptCouple: the largest difference between the
  maximum of ``c`` over ``P(z)`` (which a flux meeting the constraints has to reach) and its
  maximum without production, at least ``min_gcp``. Further SUPPRESS / PROTECT modules restrict
  the sets that count.

Command line (long campaigns)::

    python tests/lp_fuzz.py --seed 1 --n 2000 --out DIR [--kinds S,SP,...] [--backends ...]
                            [--bf gurobi] [--cross 0.1] [--deep] [--rich 0.8] [--exotic 0.3]

writes one line per problem to DIR/results_*.jsonl and the problem itself, as json, wherever a
backend's verdict is not 'ok'.
"""
import itertools
import json
import logging
import os
import sys
import time
import numpy as np

INF = float('inf')
# a verdict that changes when a threshold moves by this much is a tie, not a statement
DELTA = 1e-6
# an inner optimum counts as reached within this (relative): the LPs are solved in floating point
OPT_SLACK = 1e-8
# optimal values of two backends agree within this (relative to max(1, |value|))
VALUE_TOL = 1e-5
# seconds a backend gets for one problem
TIME_LIMIT = 120

SUPPRESS, PROTECT, OPTKNOCK, ROBUSTKNOCK, OPTCOUPLE = 'suppress', 'protect', 'optknock', 'robustknock', 'optcouple'
OPTIMISATION = (OPTKNOCK, ROBUSTKNOCK, OPTCOUPLE)


class ProblemError(Exception):
    """The problem is outside what the definitions cover (an unbounded objective, for instance)."""


# ------------------------------------------------------------------------------------------------
# LPs of the brute force:  max obj.v  s.t.  S v = 0, bounds, fixed rows, one row with a rhs per call
# ------------------------------------------------------------------------------------------------
class _HighsLP:
    """Every solve builds the LP anew and hands it to HiGHS through scipy."""

    def __init__(self, S, lb, ub, A_ub, b_ub, A_eq, b_eq, row=None):
        n = S.shape[1]
        self.lb, self.ub = lb, ub
        self.A_eq = np.vstack((S, np.reshape(A_eq, (-1, n))))
        self.b_eq = np.concatenate((np.zeros(S.shape[0]), b_eq))
        self.A_ub, self.b_ub = np.reshape(A_ub, (-1, n)), np.array(b_ub, float)
        self.row = row

    def solve(self, obj, blocked, row_min=None):
        """('optimal', value) | ('infeasible', None) | ('unbounded', None) for max obj.v with the
        reactions `blocked` at zero and, if given, ``row . v >= row_min``."""
        from scipy.optimize import linprog
        lb, ub = self.lb.copy(), self.ub.copy()
        forced = [j for j in blocked if lb[j] > 0 or ub[j] < 0]
        if forced:
            return 'infeasible', None
        lb[list(blocked)] = 0.0
        ub[list(blocked)] = 0.0
        A_ub, b_ub = self.A_ub, self.b_ub
        if row_min is not None:
            A_ub, b_ub = np.vstack((A_ub, -self.row)), np.append(b_ub, -row_min)
        opts = dict(primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9)
        kw = dict(A_ub=A_ub if len(b_ub) else None, b_ub=b_ub if len(b_ub) else None, A_eq=self.A_eq, b_eq=self.b_eq,
                  bounds=np.column_stack((lb, ub)), method='highs', options=opts)
        res = linprog(-np.asarray(obj, float), **kw)
        if res.status == 0:
            return 'optimal', -res.fun
        if res.status == 2:
            return 'infeasible', None
        # unbounded, or not told apart from infeasible: the LP without objective settles it
        res0 = linprog(np.zeros(len(lb)), **kw)
        if res0.status == 2:
            return 'infeasible', None
        if res0.status == 0 and res.status == 3:
            return 'unbounded', None
        raise ProblemError('HiGHS ended with status %d: %s' % (res.status, res.message))


class _GurobiLP:
    """One gurobipy model per LP; bounds, objective and the row's right-hand side change per solve."""
    _env = None

    def __init__(self, S, lb, ub, A_ub, b_ub, A_eq, b_eq, row=None):
        import gurobipy as gp
        if _GurobiLP._env is None:
            _GurobiLP._env = gp.Env(params={'OutputFlag': 0})
        n = S.shape[1]
        self.lb, self.ub = lb, ub
        m = self.m = gp.Model(env=_GurobiLP._env)
        m.Params.Threads = 1
        m.Params.FeasibilityTol = 1e-9
        m.Params.OptimalityTol = 1e-9
        m.Params.DualReductions = 0  # infeasible and unbounded are told apart
        self.x = m.addMVar(n, lb=np.maximum(lb, -gp.GRB.INFINITY), ub=np.minimum(ub, gp.GRB.INFINITY))
        m.addConstr(S @ self.x == 0)
        if len(b_eq):
            m.addConstr(np.reshape(A_eq, (-1, n)) @ self.x == np.array(b_eq, float))
        if len(b_ub):
            m.addConstr(np.reshape(A_ub, (-1, n)) @ self.x <= np.array(b_ub, float))
        self.row = None
        if row is not None:
            self.row = m.addConstr(np.asarray(row, float) @ self.x >= -gp.GRB.INFINITY)
        m.ModelSense = gp.GRB.MAXIMIZE
        m.update()
        self.blocked = frozenset()

    def solve(self, obj, blocked, row_min=None):
        import gurobipy as gp
        if any(self.lb[j] > 0 or self.ub[j] < 0 for j in blocked):
            return 'infeasible', None
        for j in self.blocked - blocked:
            self.x[j].LB, self.x[j].UB = max(self.lb[j], -gp.GRB.INFINITY), min(self.ub[j], gp.GRB.INFINITY)
        for j in blocked - self.blocked:
            self.x[j].LB, self.x[j].UB = 0.0, 0.0
        self.blocked = frozenset(blocked)
        if self.row is not None:
            self.row.RHS = -gp.GRB.INFINITY if row_min is None else float(row_min)
        self.x.Obj = np.asarray(obj, float)
        self.m.optimize()
        s = self.m.Status
        if s == gp.GRB.OPTIMAL:
            return 'optimal', self.m.ObjVal
        if s == gp.GRB.INFEASIBLE:
            return 'infeasible', None
        if s == gp.GRB.UNBOUNDED:
            return 'unbounded', None
        raise ProblemError('Gurobi ended with status %d' % s)


_LP_CLASSES = {'highs': _HighsLP, 'gurobi': _GurobiLP}


# ------------------------------------------------------------------------------------------------
# the problem as data
# ------------------------------------------------------------------------------------------------
def interventions(problem):
    """[(name, cost, is_gene)] of a problem, genes first."""
    out = []
    if problem.get('gene_kos'):
        genes = problem['gko_cost'] if problem.get('gko_cost') else {g: 1 for g in problem['genes']}
        out += [(g, c, True) for g, c in genes.items()]
        out += [(r, c, False) for r, c in (problem.get('ko_cost') or {}).items()]
    else:
        costs = problem['ko_cost'] if problem.get('ko_cost') else {r: 1 for r in problem['rxns']}
        out += [(r, c, False) for r, c in costs.items()]
    return out


def module_reactions(problem):
    """Reactions that a module names (in a constraint, an objective or as product)."""
    used = set()
    for mod in problem['modules']:
        for expr, _, _ in mod.get('constraints') or []:
            used.update(expr)
        for key in ('inner_objective', 'outer_objective', 'prod_id'):
            used.update(mod.get(key) or {})
    return used


def expr_str(expr):
    """A linear expression {reaction: coefficient} the way a module is written by hand."""
    return ' + '.join('%r %s' % (float(v), k) for k, v in expr.items()).replace('+ -', '- ')


def constraint_str(constraint):
    expr, op, rhs = constraint
    return '%s %s %r' % (expr_str(expr), op, float(rhs))


def problem_from_model(model, modules, **settings):
    """The problem dict of a cobra model. modules: list of dicts in the form of random_problem."""
    from cobra.util.array import create_stoichiometric_matrix
    return dict(rxns=[r.id for r in model.reactions], mets=[m.id for m in model.metabolites],
                S=np.array(create_stoichiometric_matrix(model), float).tolist(), lb=[float(r.lower_bound) for r in model.reactions],
                ub=[float(r.upper_bound) for r in model.reactions], gpr=[r.gene_reaction_rule for r in model.reactions],
                genes=[g.id for g in model.genes], modules=modules, **settings)


# ------------------------------------------------------------------------------------------------
# the brute force
# ------------------------------------------------------------------------------------------------
class BruteForce:
    """Every intervention set of a problem within the budget, decided by plain LPs.

    Args:
        problem (dict)
        lp (str): 'highs' or 'gurobi'
        shift (float): moves the threshold of every inequality a module states (and min_gcp):
            positive values enlarge the modules' regions, negative ones shrink them. With a
            positive shift an inner optimum also counts as reached within DELTA; with a negative
            one a fraction of the inner optimum (inner_opt_tol < 1) has to be exceeded by DELTA.
    """

    def __init__(self, problem, lp='highs', shift=0.0):
        p = self.p = problem
        self.S = np.array(p['S'], float).reshape(len(p['mets']), len(p['rxns']))
        self.lb, self.ub = np.array(p['lb'], float), np.array(p['ub'], float)
        self.n = len(p['rxns'])
        self.col = {r: j for j, r in enumerate(p['rxns'])}
        self.itv = interventions(p)
        self.names = [name for name, _, _ in self.itv]
        self.cost = {name: cost for name, cost, _ in self.itv}
        self.is_gene = {name: g for name, _, g in self.itv}
        self.max_cost = p['max_cost']
        self.shift = shift
        self.opt_slack = DELTA if shift > 0 else OPT_SLACK
        # a fraction of the inner optimum (inner_opt_tol < 1) is a threshold like any other
        self.frac_slack = -DELTA if shift < 0 else self.opt_slack
        self._lp_class = _LP_CLASSES[lp]
        self._lps, self._cache, self._blocked = {}, {}, {}
        self.n_lps = 0
        self.mcs_modules = [(i, m) for i, m in enumerate(p['modules']) if m['module_type'] in (SUPPRESS, PROTECT)]
        opt = [(i, m) for i, m in enumerate(p['modules']) if m['module_type'] in OPTIMISATION]
        if len(opt) > 1 or len(opt) + len(self.mcs_modules) != len(p['modules']):
            raise ProblemError('one optimisation module at most, next to SUPPRESS and PROTECT modules')
        self.opt_module = opt[0] if opt else None

    # -- interventions ---------------------------------------------------------------------------
    def sets(self):
        """All intervention sets within the budget as frozensets of names, cheapest first."""
        out = []

        def grow(start, chosen, cost):
            out.append((cost, len(chosen), tuple(chosen)))
            for k in range(start, len(self.itv)):
                name, c, _ = self.itv[k]
                if cost + c <= self.max_cost:
                    grow(k + 1, chosen + [name], cost + c)

        grow(0, [], 0)
        out.sort(key=lambda t: (t[0], t[1], [self.names.index(x) for x in t[2]]))
        return [frozenset(t[2]) for t in out]

    def set_cost(self, z):
        return sum(self.cost[x] for x in z)

    def blocked(self, z):
        """Reactions (column indices) that the interventions z switch off."""
        z = frozenset(z)
        if z not in self._blocked:
            off = {self.col[x] for x in z if not self.is_gene[x]}
            knocked = {x for x in z if self.is_gene[x]}
            if knocked:
                present = {g: g not in knocked for g in self.p['genes']}
                for j, rule in enumerate(self.p['gpr']):
                    if rule and not eval(rule, {'__builtins__': {}}, present):
                        off.add(j)
            self._blocked[z] = frozenset(off)
        return self._blocked[z]

    # -- LPs -------------------------------------------------------------------------------------
    def _vec(self, expr, sign=1.0):
        v = np.zeros(self.n)
        for r, a in (expr or {}).items():
            v[self.col[r]] += sign * a
        return v

    def _rows(self, constraints):
        """(A_ub, b_ub, A_eq, b_eq) of a list of constraints, inequality thresholds shifted."""
        A_ub, b_ub, A_eq, b_eq = [], [], [], []
        for expr, op, rhs in constraints or []:
            if op == '=':
                A_eq.append(self._vec(expr))
                b_eq.append(rhs)
            elif op == '<=':
                A_ub.append(self._vec(expr))
                b_ub.append(rhs + self.shift)
            elif op == '>=':
                A_ub.append(self._vec(expr, -1.0))
                b_ub.append(-rhs + self.shift)
            else:
                raise ProblemError('unknown relation %r' % op)
        return A_ub, b_ub, A_eq, b_eq

    def _solve(self, key, rows, obj_key, obj, X, row=None, row_min=None):
        """Maximum of obj over P(X) with the given rows; (status, value), cached."""
        ck = (key, obj_key, X, row_min)
        if ck not in self._cache:
            if key not in self._lps:
                self._lps[key] = self._lp_class(self.S, self.lb, self.ub, *rows, row=row)
            self.n_lps += 1
            self._cache[ck] = self._lps[key].solve(obj, X, row_min)
        return self._cache[ck]

    def _inner(self, i, mod):
        """Inner objective of module i, oriented for maximisation."""
        return self._vec(mod['inner_objective'], -1.0 if mod.get('inner_opt_sense') == 'minimize' else 1.0)

    def _inner_max(self, i, mod, X):
        """max c over P(X); None if P(X) is empty."""
        status, val = self._solve('P', ([], [], [], []), ('inner', i), self._inner(i, mod), X)
        if status == 'unbounded':
            raise ProblemError('the inner objective of module %d is unbounded' % i)
        return val

    def _floor(self, target, tol=1.0):
        """What a flux has to reach to count as reaching `target` = tol * inner optimum."""
        return target - (self.opt_slack if tol >= 1.0 else self.frac_slack) * max(1.0, abs(target))

    def _reached(self, value, target, tol=1.0):
        return value >= self._floor(target, tol)

    # -- SUPPRESS / PROTECT ----------------------------------------------------------------------
    def region_nonempty(self, i, mod, X):
        rows = self._rows(mod.get('constraints'))
        if not mod.get('inner_objective'):
            return self._solve(('mod', i), rows, 'zero', np.zeros(self.n), X)[0] != 'infeasible'
        g1 = self._inner_max(i, mod, X)
        if g1 is None:
            return False
        status, g2 = self._solve(('mod', i), rows, ('inner', i), self._inner(i, mod), X)
        if status != 'optimal':
            return False
        tol = mod.get('inner_opt_tol') or 1.0
        return self._reached(g2, tol * g1, tol)

    def mcs_ok(self, z):
        """Every SUPPRESS / PROTECT module is fulfilled."""
        X = self.blocked(z)
        return all(self.region_nonempty(i, mod, X) == (mod['module_type'] == PROTECT) for i, mod in self.mcs_modules)

    # -- optimisation modules --------------------------------------------------------------------
    def score(self, z):
        """Objective value of the intervention set z for the optimisation module; None if z does
        not qualify. Values are in the module's own orientation (the larger the better unless the
        outer objective is minimised, then the value is that of the minimised expression)."""
        if not self.mcs_ok(z):
            return None
        X = self.blocked(z)
        i, mod = self.opt_module
        kind = mod['module_type']
        rows = self._rows(mod.get('constraints'))
        c = self._inner(i, mod)
        g1 = self._inner_max(i, mod, X)
        if g1 is None:
            return None
        tol = mod.get('inner_opt_tol') or 1.0
        if kind == OPTCOUPLE:
            prod = self._vec(mod['prod_id'])
            status, g0 = self._solve(('noprod', i), ([], [], [prod], [0.0]), ('inner', i), c, X)
            status, g1c = self._solve(('mod', i), rows, ('inner', i), c, X)
            if g0 is None or g1c is None or not self._reached(g1c, tol * g1, tol):
                return None
            gcp = g1c - g0
            return gcp if gcp >= (mod.get('min_gcp') or 0.0) - self.shift - OPT_SLACK else None
        sign = -1.0 if mod.get('outer_opt_sense') == 'minimize' else 1.0
        o = self._vec(mod['outer_objective'], sign)
        if kind == OPTKNOCK:
            target = tol * g1
            status, val = self._solve(('mod+', i), rows, ('outer', i), o, X, row=c, row_min=self._floor(target, tol))
        else:  # the worst growth-optimal flux, for sets that keep the constraints attainable
            if self._solve(('mod', i), rows, 'zero', np.zeros(self.n), X)[0] == 'infeasible':
                return None
            status, val = self._solve(('P+', i), ([], [], [], []), ('-outer', i), -o, X, row=c, row_min=self._floor(g1))
            val = None if val is None else -val
        if status == 'unbounded':
            raise ProblemError('the outer objective is unbounded')
        return None if val is None else sign * val

    def better(self, a, b):
        """Value a is better than value b for the optimisation module."""
        return a < b if self.opt_module[1].get('outer_opt_sense') == 'minimize' else a > b

    # -- results ---------------------------------------------------------------------------------
    def enumerate(self):
        """The acceptable sets in order of cost, each excluding its supersets."""
        if self.mcs_ok(frozenset()):
            return [frozenset()]
        found = []
        for z in self.sets():
            if not any(d <= z for d in found) and self.mcs_ok(z):
                found.append(z)
        return found

    def scores(self):
        """{z: value} for every set that qualifies for the optimisation module."""
        out = {}
        for z in self.sets():
            val = self.score(z)
            if val is not None:
                out[z] = val
        return out

    def optimum(self):
        """(best value, cheapest cost at which it is reached within VALUE_TOL); (None, None) if no
        set qualifies."""
        sc = self.scores()
        if not sc:
            return None, None
        best = None
        for val in sc.values():
            if best is None or self.better(val, best):
                best = val
        cost = min(self.set_cost(z) for z, val in sc.items() if abs(val - best) <= VALUE_TOL * max(1.0, abs(best)))
        return best, cost


def reference(problem, lp='highs'):
    """Result of the brute force: dict(designs=[frozenset]) for an enumeration problem,
    dict(value=, cost=) for an optimisation problem."""
    bf = BruteForce(problem, lp)
    if bf.opt_module is None:
        return dict(designs=bf.enumerate(), n_lps=bf.n_lps, n_sets=len(bf.sets()))
    value, cost = bf.optimum()
    return dict(value=value, cost=cost, n_lps=bf.n_lps, n_sets=len(bf.sets()))


# ------------------------------------------------------------------------------------------------
# the same problem through the package
# ------------------------------------------------------------------------------------------------
def build_model(problem):
    """cobra model of a problem."""
    import cobra
    model = cobra.Model(problem.get('name', 'fuzz'))
    mets = [cobra.Metabolite(m, compartment='c') for m in problem['mets']]
    S = np.array(problem['S'], float).reshape(len(mets), len(problem['rxns']))
    rxns = []
    for j, rid in enumerate(problem['rxns']):
        r = cobra.Reaction(rid, lower_bound=problem['lb'][j], upper_bound=problem['ub'][j])
        r.add_metabolites({mets[i]: S[i, j] for i in np.flatnonzero(S[:, j])})
        rxns.append(r)
    model.add_metabolites(mets)
    model.add_reactions(rxns)
    for r, rule in zip(rxns, problem['gpr']):
        if rule:
            r.gene_reaction_rule = rule
    return model


def build_modules(problem, model):
    """SDModule objects of a problem; raises what the package raises for an invalid module."""
    import straindesign as sd
    out = []
    for mod in problem['modules']:
        kw = {k: v for k, v in mod.items() if k not in ('module_type', 'constraints', 'inner_objective', 'outer_objective', 'prod_id')}
        if mod.get('constraints'):
            kw['constraints'] = [constraint_str(c) for c in mod['constraints']]
        for key in ('inner_objective', 'outer_objective', 'prod_id'):
            if mod.get(key):
                kw[key] = expr_str(mod[key])
        out.append(sd.SDModule(model, mod['module_type'], **kw))
    return out


def package_arguments(problem, solver, engine=None):
    """Keyword arguments of compute_strain_designs for a problem (without the modules)."""
    optimisation = any(m['module_type'] in OPTIMISATION for m in problem['modules'])
    kw = dict(max_cost=problem['max_cost'], solver=solver, compress=problem.get('compress', True), seed=1,
              solution_approach='best' if optimisation else 'populate', time_limit=TIME_LIMIT)
    if optimisation:
        kw['max_solutions'] = 1
    if problem.get('gene_kos'):
        kw['gene_kos'] = True
        if problem.get('gko_cost'):
            kw['gko_cost'] = dict(problem['gko_cost'])
    if problem.get('ko_cost'):
        kw['ko_cost'] = dict(problem['ko_cost'])
    if engine:
        kw['engine'] = engine
    elif solver != 'glpk':
        kw['milp_threads'] = 1
    return kw


def run_engine_alone(problem, solver, exact=False):
    """The LP engine on the problem's network as it is: no compression, no FVA that takes
    essential or blocked reactions out of the search first."""
    try:
        from straindesign.lp_enumeration import lp_engine
        model = build_model(problem)
        modules = build_modules(problem, model)
        itv = interventions(problem)
        res, enum = lp_engine(model, modules, {r: float(c) for r, c, gene in itv if not gene}, {}, {g: float(c) for g, c, gene in itv if gene}, {},
                              False, problem['max_cost'], solver, time_limit=TIME_LIMIT, exact=exact)
    except Exception as e:
        return dict(error='%s: %s' % (type(e).__name__, e))
    out = dict(designs=[frozenset(d) for d in res.sd], status=res.status)
    if any(m['module_type'] in OPTIMISATION for m in problem['modules']):
        out['claimed'] = getattr(enum, 'objective_value', None)
    return out


def run_package(problem, solver, engine=None):
    """Result of compute_strain_designs in the form of :func:`reference`, plus status; an
    exception comes back as dict(error=...)."""
    import straindesign as sd
    if engine in ('raw', 'raw_exact'):
        return run_engine_alone(problem, solver, engine == 'raw_exact')
    try:
        model = build_model(problem)
        modules = build_modules(problem, model)
        sol = sd.compute_strain_designs(model, sd_modules=modules, **package_arguments(problem, solver, engine))
        if sol.is_lazy:
            sol.expand_all()
    except Exception as e:  # reported, not raised: a failing backend is a finding
        return dict(error='%s: %s' % (type(e).__name__, e))
    sds = sol.gene_sd if problem.get('gene_kos') else sol.reaction_sd
    designs = [frozenset(k for k, v in d.items() if v != 0) for d in sds]
    out = dict(designs=designs, status=sol.status)
    if any(m['module_type'] in OPTIMISATION for m in problem['modules']):
        out['claimed'] = getattr(sol, 'objective_value', None)
    return out


# ------------------------------------------------------------------------------------------------
# comparison
# ------------------------------------------------------------------------------------------------
BACKENDS = {
    'milp-cplex': ('cplex', None),
    'lp-glpk': ('glpk', 'lp'),
    'lp-cplex': ('cplex', 'lp'),
    'lp-gurobi': ('gurobi', 'lp'),
    'lp-scip': ('scip', 'lp'),
    'lp_exact': ('glpk', 'lp_exact'),
    # the engine without compute_strain_designs around it
    'raw-glpk': ('glpk', 'raw'),
    'raw-exact': ('glpk', 'raw_exact'),
}


def installed_backends():
    """The names in BACKENDS whose solver can be imported."""
    from importlib.util import find_spec
    have = {'glpk': True, 'cplex': bool(find_spec('cplex')), 'gurobi': bool(find_spec('gurobipy')), 'scip': bool(find_spec('pyscipopt'))}
    return [name for name, (solver, _) in BACKENDS.items() if have[solver]]


class _Verdicts:
    """Verdicts of the brute force at the stated thresholds and with every threshold moved by
    DELTA either way; a set whose verdict depends on that is *sensitive*."""

    def __init__(self, problem, lp):
        self.bf = {s: BruteForce(problem, lp, shift=s) for s in (0.0, DELTA, -DELTA)}
        last = _LAST_BRUTE_FORCE.get('last')
        if last is not None and last[0] is problem and last[1] == lp:
            self.bf[0.0] = last[2]
        self.nominal = self.bf[0.0]

    def ok(self, z):
        """(acceptable at the stated thresholds, sensitive)"""
        v = [bf.mcs_ok(z) for bf in self.bf.values()]
        return v[0], len(set(v)) > 1

    def score(self, z):
        """(value at the stated thresholds or None, sensitive)"""
        v = [bf.score(z) for bf in self.bf.values()]
        return v[0], len({x is None for x in v}) > 1

    def score_range(self, z):
        """(smallest, largest) value of z over the three settings of the thresholds; None if z
        qualifies under none of them."""
        vals = [v for v in (bf.score(z) for bf in self.bf.values()) if v is not None]
        return (min(vals), max(vals)) if vals else None

    def optimum_range(self):
        """(smallest, largest) optimum over the three settings; None if there is none in any."""
        vals = [v for v in (bf.optimum()[0] for bf in self.bf.values()) if v is not None]
        return (min(vals), max(vals)) if vals else None


# the brute force that random_problem ran last, for compare() to take over: (problem, lp, BruteForce)
_LAST_BRUTE_FORCE = {}


def _judge_enumeration(designs, verdicts):
    """'ok', 'tie' or a text that says what is wrong with a list of designs."""
    bf = verdicts.nominal
    ref = bf.enumerate()
    if len(set(designs)) != len(designs):
        return 'a design is returned twice'
    if set(designs) == set(ref):
        return 'ok'
    unknown = [sorted(z) for z in designs if not set(z) <= set(bf.names)]
    if unknown:
        return 'designs with unknown interventions: %s' % unknown[:3]
    # with the sensitive sets free to fall either way: is this a valid result?
    found, wrong = set(designs), []
    if frozenset() in found and len(found) > 1:
        wrong.append('the empty design next to others')
    for z in bf.sets():
        ok, sensitive = verdicts.ok(z)
        below = any(d < z for d in found)
        if z in found:
            if below:
                wrong.append('%s contains another design' % sorted(z))
            elif not ok and not sensitive:
                wrong.append('%s is no design' % sorted(z))
        elif ok and not sensitive and not below:
            wrong.append('%s is missing' % sorted(z))
    over = [sorted(z) for z in found if bf.set_cost(z) > bf.max_cost]
    if over:
        wrong.append('over budget: %s' % over[:3])
    if not wrong:
        return 'tie'
    return '; '.join(wrong[:6]) + (' ...' if len(wrong) > 6 else '') + ' (reference: %d designs, returned: %d)' % (len(ref), len(designs))


def _close(a, b):
    return abs(a - b) <= VALUE_TOL * max(1.0, abs(a), abs(b))


def _within(x, lo, hi):
    return lo - VALUE_TOL * max(1.0, abs(lo)) <= x <= hi + VALUE_TOL * max(1.0, abs(hi))


def _judge_optimisation(res, verdicts):
    """'ok', 'tie' or a text that says what is wrong with the result of an optimisation problem.
    A tie: the result is right for thresholds moved by DELTA, not for the stated ones."""
    bf = verdicts.nominal
    best, _ = bf.optimum()
    designs = res['designs']
    if not designs:
        if best is None:
            return 'ok'
        # no design although sets qualify: a tie if every qualifying set is sensitive
        if all(verdicts.score(z)[1] for z in bf.scores()):
            return 'tie'
        return 'no design returned, the reference optimum is %.9g' % best
    if len(designs) > 1:
        # one design of the compressed network can stand for several of the original one (lumped
        # reactions, each of which may be the one knocked out): every one of them has to hold up
        each = [_judge_optimisation(dict(res, designs=[z]), verdicts) for z in designs]
        wrong = [v for v in each if v not in ('ok', 'tie')]
        return wrong[0] if wrong else ('tie' if 'tie' in each else 'ok')
    z = designs[0]
    if not set(z) <= set(bf.names):
        return 'design with unknown interventions: %s' % sorted(z)
    if bf.set_cost(z) > bf.max_cost:
        return 'design over budget: %s' % sorted(z)
    val, sensitive = verdicts.score(z)
    span = verdicts.score_range(z)
    if span is None:
        return 'the returned design %s does not qualify (reference optimum %s)' % (sorted(z), best)
    tie = sensitive
    claimed = res.get('claimed')
    if claimed is not None and (val is None or not _close(claimed, val)):
        if not _within(claimed, *span):
            return 'the returned design %s has value %s (%.9g to %.9g with the thresholds moved), not the claimed %.9g' % (
                sorted(z), val, span[0], span[1], claimed)
        tie = True
    if val is not None and best is not None and _close(val, best):
        return 'tie' if tie else 'ok'
    # better or worse than the reference optimum: a tie if moved thresholds bring the two together
    top = verdicts.optimum_range()
    if top is not None and span[0] <= top[1] + VALUE_TOL * max(1.0, abs(top[1])) and top[0] <= span[1] + VALUE_TOL * max(1.0, abs(span[1])):
        return 'tie'
    return 'the returned design %s has value %s, the reference optimum is %s' % (sorted(z), val, best)


def compare(problem, backends=None, lp='highs'):
    """Run the brute force and the backends on a problem.

    Returns:
        dict(reference=..., results={backend: result}, verdict={backend: 'ok' | 'tie' | text})
    """
    backends = list(backends or installed_backends())
    verdicts = _Verdicts(problem, lp)
    bf = verdicts.nominal
    optimisation = bf.opt_module is not None
    if optimisation:
        value, cost = bf.optimum()
        ref = dict(value=value, cost=cost)
    else:
        ref = dict(designs=bf.enumerate())
    results, verdict = {}, {}
    for name in backends:
        solver, engine = BACKENDS[name]
        res = results[name] = run_package(problem, solver, engine)
        if 'error' in res:
            verdict[name] = 'error: ' + res['error']
        elif res['status'] not in ('optimal', 'infeasible'):
            verdict[name] = 'status %s' % res['status']
        elif optimisation:
            verdict[name] = _judge_optimisation(res, verdicts)
        else:
            verdict[name] = _judge_enumeration(res['designs'], verdicts)
    ref['n_sets'] = len(bf.sets())
    return dict(reference=ref, results=results, verdict=verdict)


# ------------------------------------------------------------------------------------------------
# random problems
# ------------------------------------------------------------------------------------------------
KINDS = ('S', 'SP', 'SS', 'nS', 'nSP', 'nP', 'nSnP', 'optknock', 'robustknock', 'optcouple')
_NEEDS_BOUNDS = ('nS', 'nSP', 'nP', 'nSnP', 'optknock', 'robustknock', 'optcouple')  # an inner optimum has to exist


def _random_network(rng, bounds):
    """A small network with uptake, excretion, internal reactions and a biomass reaction.

    bounds: 'cone' (0 / inf), 'bounded' (finite capacities, some reactions without), 'forced'
    (bounded, and one reaction has to carry flux). Returns the network part of a problem dict plus
    the roles of some reactions.
    """
    m = int(rng.integers(4, 9))
    n = int(rng.integers(6, 15))
    n_up = int(rng.integers(1, 3))
    n_ex = int(rng.integers(2, min(m, 3) + 1))
    n_int = max(1, n - n_up - n_ex - 1)
    cols, ids, rev = [], [], []
    reachable = []
    for k in range(n_up):  # -> M
        col = np.zeros(m)
        col[k] = 1
        cols.append(col)
        ids.append('UP%d' % k)
        rev.append(False)
        reachable.append(k)
    for k in range(n_int):
        col = np.zeros(m)
        subs = rng.choice(reachable, size=min(len(reachable), int(rng.integers(1, 3))), replace=False)
        rest = [i for i in range(m) if i not in subs]
        prods = rng.choice(rest, size=min(len(rest), int(rng.integers(1, 3))), replace=False)
        for i in subs:
            col[i] = -int(rng.integers(1, 3))
        for i in prods:
            col[i] = int(rng.integers(1, 3))
            if i not in reachable:
                reachable.append(int(i))
        cols.append(col)
        ids.append('R%d' % k)
        rev.append(bool(rng.random() < 0.25))
    pool = list(reachable)
    rng.shuffle(pool)
    out = pool[:n_ex] if len(pool) >= n_ex else pool
    for k, i in enumerate(out):  # M ->
        col = np.zeros(m)
        col[i] = -1
        cols.append(col)
        ids.append('EX_P' if k == 0 else 'EX%d' % k)
        rev.append(False)
    col = np.zeros(m)  # biomass: several metabolites at once
    for i in rng.choice(reachable, size=min(len(reachable), int(rng.integers(2, 4))), replace=False):
        col[i] = -int(rng.integers(1, 3))
    cols.append(col)
    ids.append('BM')
    rev.append(False)
    S = np.array(cols).T
    keep = np.flatnonzero(np.abs(S).sum(1))
    S = S[keep]
    n = len(ids)
    lb, ub = np.zeros(n), np.full(n, INF)
    if bounds == 'cone':
        lb[np.array(rev)] = -INF
    else:
        open_ended = rng.random() < 0.4  # some reactions without capacity
        for j, rid in enumerate(ids):
            if rid.startswith('UP'):
                ub[j] = float(rng.choice([10, 20]))
            elif not (open_ended and rng.random() < 0.5):
                ub[j] = float(rng.choice([10, 50, 100, 1000]))
            if rev[j]:
                lb[j] = -ub[j]
    net = dict(rxns=ids, mets=['M%d' % i for i in keep], S=S.tolist(), lb=lb.tolist(), ub=ub.tolist())
    return net


def _random_rule(rng, genes, depth=0):
    """A random and/or tree over a few genes as a rule string; a gene may come up twice."""
    r = rng.random()
    if depth == 2 or r < (0.35 if depth == 0 else 0.6):
        return str(rng.choice(genes))
    op = ' and ' if rng.random() < 0.5 else ' or '
    parts = [_random_rule(rng, genes, depth + 1) for _ in range(int(rng.integers(2, 4)))]
    return '(' + op.join(parts) + ')' if depth else op.join(parts)


def _wild_type_range(problem, expr, constraints=()):
    """(min, max) of an expression over the wild type's fluxes that meet the constraints; None
    where there is no such flux, +-inf where it is unbounded."""
    bf = BruteForce(dict(problem, modules=[], max_cost=0))
    v = bf._vec(expr)
    out = []
    for sign in (-1.0, 1.0):
        status, val = bf._lp_class(bf.S, bf.lb, bf.ub, *bf._rows(constraints)).solve(sign * v, frozenset())
        if status == 'infeasible':
            return None
        out.append(sign * (INF if status == 'unbounded' else val))
    return tuple(out)


def _nice(rng, top):
    """A short decimal in (0, top]."""
    grid = [x for x in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100) if x <= top + 1e-9]
    return float(rng.choice(grid)) if grid else round(top / 2, 3)


def _random_modules(rng, net, kind, cone, exotic=0.0):
    """Modules of one kind over the roles of the network; None if the network does not allow them.
    exotic: probability, per place, of an equality among the constraints, an objective over two
    reactions, a SUPPRESS module next to an optimisation module."""
    bm, prod = {'BM': 1.0}, {'EX_P': 1.0}
    ups = [r for r in net['rxns'] if r.startswith('UP')]
    others = [r for r in net['rxns'] if r.startswith('EX') and r != 'EX_P']
    rng_bm = _wild_type_range(net, bm)
    rng_p = _wild_type_range(net, prod)
    if rng_bm is None or rng_bm[1] <= 1e-6 or rng_p[1] <= 1e-6:
        return None
    top_bm = 2.0 if cone else rng_bm[1]
    top_p = 2.0 if cone else rng_p[1]
    growth = lambda frac=1.0: [dict(bm), '>=', _nice(rng, top_bm * frac)]

    def low_product():  # little product (absolute, or per substrate), with growth or uptake so that zero flux is excluded
        if rng.random() < 0.5 or cone:
            y = float(rng.choice([0.2, 0.5, 1.0, 2.0]))
            c = [[{'EX_P': 1.0, ups[0]: -y}, '<=', 0.0]]
        else:
            c = [[dict(prod), '<=', _nice(rng, top_p * 0.6)]]
        c.append(growth(0.5) if rng.random() < 0.6 else [{ups[0]: 1.0}, '>=', _nice(rng, 1.0)])
        return c

    def odd(mod):  # an equality: a fixed uptake, or a by-product that is not excreted
        if exotic and rng.random() < exotic:
            if others and rng.random() < 0.5:
                mod['constraints'] = mod['constraints'] + [[{str(rng.choice(others)): 1.0}, '=', 0.0]]
            else:
                mod['constraints'] = mod['constraints'] + [[{ups[0]: 1.0}, '=', _nice(rng, 5.0)]]
        return mod

    def objective(expr, other):  # at times a second reaction with a small weight
        expr = dict(expr)
        if exotic and other and rng.random() < exotic:
            expr[other] = float(rng.choice([-0.5, -0.1, 0.1, 0.5]))
        return expr

    def suppress():
        r = rng.random()
        if r < 0.3:
            return odd(dict(module_type=SUPPRESS, constraints=[growth()]))
        if r < 0.75 or not others:
            return odd(dict(module_type=SUPPRESS, constraints=low_product()))
        return odd(dict(module_type=SUPPRESS, constraints=[[{str(rng.choice(others)): 1.0}, '>=', _nice(rng, 2.0)]]))

    def protect():
        r = rng.random()
        c = [growth(0.5)]
        if r < 0.3:
            c = [[dict(prod), '>=', _nice(rng, top_p * 0.5)]]
        elif r < 0.5:
            c.append([dict(prod), '>=', _nice(rng, top_p * 0.3)])
        return odd(dict(module_type=PROTECT, constraints=c))

    def inner(mod):
        mod['inner_objective'] = objective(bm, ups[0])
        if rng.random() < 0.3:
            mod['inner_opt_tol'] = float(rng.choice([0.5, 0.8, 0.9, 0.99]))
        return odd(mod)

    if kind == 'S':
        return [suppress()]
    if kind == 'SP':
        return [suppress(), protect()]
    if kind == 'SS':
        return [suppress(), suppress()] + ([protect()] if rng.random() < 0.4 else [])
    if kind == 'nS':
        return [inner(dict(module_type=SUPPRESS, constraints=low_product()))]
    if kind == 'nSP':
        return [inner(dict(module_type=SUPPRESS, constraints=low_product())), protect()]
    if kind == 'nP':
        mods = [inner(dict(module_type=PROTECT, constraints=[[dict(prod), '>=', _nice(rng, top_p * 0.5)]]))]
        return mods + [suppress()]
    if kind == 'nSnP':
        return [inner(dict(module_type=SUPPRESS, constraints=low_product())), inner(dict(module_type=PROTECT, constraints=[growth(0.5)]))]
    constraints = [growth(0.5)] if rng.random() < 0.6 else []
    extra = [protect()] if rng.random() < 0.15 else []
    if exotic and rng.random() < exotic:
        extra.append(suppress())
    if kind == 'optknock':
        mod = dict(module_type=OPTKNOCK, inner_objective=objective(bm, ups[0]), outer_objective=objective(prod, others[0] if others else None),
                   constraints=constraints)
        if rng.random() < 0.2:
            mod['inner_opt_tol'] = float(rng.choice([0.5, 0.9]))
        return [mod] + extra
    if kind == 'robustknock':
        return [dict(module_type=ROBUSTKNOCK, inner_objective=objective(bm, ups[0]), outer_objective=objective(prod, others[0] if others else None),
                     constraints=constraints)] + extra
    if kind == 'optcouple':
        mod = dict(module_type=OPTCOUPLE, inner_objective=objective(bm, ups[0]), prod_id=dict(prod), constraints=constraints)
        if rng.random() < 0.5:
            mod['min_gcp'] = _nice(rng, top_bm * 0.5)
        return [mod] + extra
    raise ValueError('unknown kind %r' % kind)


def _well_posed(problem):
    """The problem is one the package accepts and the definitions cover: every module's
    constraints can be met by the wild type, a module with inner objective excludes the zero
    flux, and every objective is bounded."""
    zero = [[{r: 1.0}, '=', 0.0] for r in problem['rxns']]
    for mod in problem['modules']:
        constraints = mod.get('constraints') or []
        if _wild_type_range(problem, {}, constraints) is None:
            return False
        if mod['module_type'] in (SUPPRESS, PROTECT) and mod.get('inner_objective') and _wild_type_range(problem, {}, constraints + zero):
            return False
        for key in ('inner_objective', 'outer_objective'):
            if mod.get(key):
                lo, hi = _wild_type_range(problem, mod[key])
                if not np.isfinite(lo) or not np.isfinite(hi):
                    return False
    return True


def random_problem(seed, kind=None, max_sets=500, trivial=0.1, deep=False, lp='highs', rich=0.0, exotic=0.0):
    """A random problem, a function of its arguments alone (as far as the LP solver's answers on
    the candidate networks do not depend on the solver).

    Args:
        seed (int)
        kind (str): one of KINDS; random if None
        max_sets (int): the budget is chosen so that at most this many intervention sets lie within it
        trivial (float): probability of keeping a problem whose wild type is acceptable already,
            or that has no design at all
        deep (bool): prefer large budgets (up to 5)
        rich (float): probability of passing over a problem with little to find (fewer than two
            designs or none that costs more than 1; an optimum that the wild type reaches)
        exotic (float): probability, per place, of the less usual: an equality among a module's
            constraints, an objective over two reactions, a SUPPRESS module next to an
            optimisation module, a reaction written backwards (upper bound 0)
        lp (str): LP solver of the brute force that looks at the candidates
    """
    rng = np.random.default_rng(seed)
    kind = kind or str(rng.choice(KINDS))
    for attempt in range(200):
        bounds = str(rng.choice(['bounded', 'forced'] if kind in _NEEDS_BOUNDS else ['cone', 'bounded', 'forced']))
        net = _random_network(rng, bounds)
        net['gpr'], net['genes'] = [''] * len(net['rxns']), []
        modules = _random_modules(rng, net, kind, bounds == 'cone', exotic)
        if modules is None:
            continue
        problem = dict(name='fuzz_%s_%d' % (kind, seed), kind=kind, bounds=bounds, seed=seed, **net, modules=modules)
        if exotic:  # internal reactions written backwards: the same network, fluxes of the other sign
            named = module_reactions(problem)
            for j, rid in enumerate(problem['rxns']):
                if rid.startswith('R') and rid not in named and rng.random() < exotic:
                    for row in problem['S']:
                        row[j] = -row[j]
                    problem['lb'][j], problem['ub'][j] = -problem['ub'][j], -problem['lb'][j]
        if bounds == 'forced':  # a reaction that has to run, at a rate the wild type can afford
            cand = [r for r in problem['rxns'] if r != 'BM' and not r.startswith('UP')]
            r = str(rng.choice(cand))
            lo, hi = _wild_type_range(problem, {r: 1.0})
            if hi < 1.0:
                continue
            problem['lb'][problem['rxns'].index(r)] = float(rng.choice([x for x in (1, 2, 5) if x <= hi]))
        gene_mode = rng.random() < 0.5
        if gene_mode:
            genes = ['g%d' % i for i in range(int(rng.integers(4, 11)))]
            for j in range(len(problem['rxns'])):
                if rng.random() < 0.75:
                    pool = list(rng.choice(genes, size=int(rng.integers(1, 5)), replace=False))
                    problem['gpr'][j] = _random_rule(rng, pool)
            used = [g for g in genes if any(g in rule.replace('(', ' ').replace(')', ' ').split() for rule in problem['gpr'])]
            if len(used) < 2:
                continue
            problem['genes'], problem['gene_kos'] = used, True
            r = rng.random()
            if r < 0.3:  # integer costs
                problem['gko_cost'] = {g: int(rng.integers(1, 4)) for g in used}
            elif r < 0.5:  # not every gene can be knocked out
                problem['gko_cost'] = {g: 1 for g in used if rng.random() < 0.8} or {used[0]: 1}
            if rng.random() < 0.3:  # reactions as further targets: those without rule, at times any
                anyone = rng.random() < 0.3
                free = [r for r, rule in zip(problem['rxns'], problem['gpr']) if anyone or not rule]
                if free:
                    problem['ko_cost'] = {r: int(rng.integers(1, 3)) for r in free if rng.random() < 0.7}
        else:
            r = rng.random()
            if r < 0.3:
                problem['ko_cost'] = {r: int(rng.integers(1, 4)) for r in problem['rxns']}
            elif r < 0.5:
                problem['ko_cost'] = {r: 1 for r in problem['rxns'] if rng.random() < 0.7} or {problem['rxns'][0]: 1}
        problem['compress'] = bool(rng.random() < 0.6)
        try:
            if not _well_posed(problem):
                continue
            n_sets = {}
            for budget in (1, 2, 3, 4, 5) if deep else (1, 2, 3, 4):
                n_sets[budget] = len(BruteForce(dict(problem, max_cost=budget)).sets())
            budgets = [b for b in n_sets if n_sets[b] <= max_sets]
            weights = np.array(([1, 2, 6, 8, 6] if deep else [1, 3, 4, 2])[:len(budgets)], float)
            problem['max_cost'] = int(rng.choice(budgets, p=weights / weights.sum()))
            bf = BruteForce(problem, lp)
            if bf.opt_module is None:
                designs = bf.enumerate()
                dull = not designs or designs == [frozenset()]
                little = len(designs) < 2 or max(bf.set_cost(z) for z in designs) < 2
            else:
                dull = len(bf.scores()) <= 1
                little = dull or bf.optimum()[1] == 0
        except ProblemError:
            continue
        if dull and rng.random() >= trivial:
            continue
        if rich and little and not dull and rng.random() < rich:
            continue
        _LAST_BRUTE_FORCE['last'] = (problem, lp, bf)
        return problem
    raise ProblemError('no well-posed problem of kind %s for seed %d' % (kind, seed))


# ------------------------------------------------------------------------------------------------
# shrinking a failing problem
# ------------------------------------------------------------------------------------------------
def _without_reaction(problem, rid):
    j = problem['rxns'].index(rid)
    p = dict(problem)
    for key in ('rxns', 'lb', 'ub', 'gpr'):
        p[key] = [x for k, x in enumerate(problem[key]) if k != j]
    p['S'] = [[x for k, x in enumerate(row) if k != j] for row in problem['S']]
    for key in ('ko_cost',):
        if p.get(key):
            p[key] = {r: c for r, c in p[key].items() if r != rid}
    return _used_genes_only(p)


def _used_genes_only(p):
    if p.get('gene_kos'):
        words = set(' '.join(p['gpr']).replace('(', ' ').replace(')', ' ').split())
        p['genes'] = [g for g in p['genes'] if g in words]
        if p.get('gko_cost'):
            p['gko_cost'] = {g: c for g, c in p['gko_cost'].items() if g in words}
    return p


def _smaller(problem):
    """Problems one step simpler than the given one."""
    p = problem
    if p['max_cost'] > 1:
        yield dict(p, max_cost=p['max_cost'] - 1)
    for k in range(len(p['modules'])):
        if len(p['modules']) > 1:
            yield dict(p, modules=p['modules'][:k] + p['modules'][k + 1:])
    named = module_reactions(p)
    for rid in list(p['rxns']):
        if rid not in named and len(p['rxns']) > 2:
            yield _without_reaction(p, rid)
    for i in range(len(p['mets'])):
        if len(p['mets']) > 1:
            yield dict(p, mets=p['mets'][:i] + p['mets'][i + 1:], S=p['S'][:i] + p['S'][i + 1:])
    for k, mod in enumerate(p['modules']):
        for c in range(len(mod.get('constraints') or [])):
            m2 = dict(mod, constraints=mod['constraints'][:c] + mod['constraints'][c + 1:])
            yield dict(p, modules=p['modules'][:k] + [m2] + p['modules'][k + 1:])
        for key in ('inner_opt_tol', 'min_gcp'):
            if key in mod:
                yield dict(p, modules=p['modules'][:k] + [{a: b for a, b in mod.items() if a != key}] + p['modules'][k + 1:])
    if p.get('gene_kos'):
        for j, rule in enumerate(p['gpr']):
            if rule:
                words = rule.replace('(', ' ').replace(')', ' ').split()
                for simpler in [''] + sorted({w for w in words if w not in ('and', 'or')}):
                    if simpler != rule:
                        yield _used_genes_only(dict(p, gpr=p['gpr'][:j] + [simpler] + p['gpr'][j + 1:]))
        if p.get('ko_cost'):
            yield {k: v for k, v in p.items() if k != 'ko_cost'}
        if p.get('gko_cost') and (set(p['gko_cost'].values()) != {1} or set(p['gko_cost']) != set(p['genes'])):
            yield {k: v for k, v in p.items() if k != 'gko_cost'}
    elif p.get('ko_cost'):
        yield {k: v for k, v in p.items() if k != 'ko_cost'}
        if set(p['ko_cost'].values()) != {1}:
            yield dict(p, ko_cost={r: 1 for r in p['ko_cost']})
    if p.get('compress', True):
        yield dict(p, compress=False)
    for j in range(len(p['rxns'])):
        if p['lb'][j] < 0:
            yield dict(p, lb=p['lb'][:j] + [0.0] + p['lb'][j + 1:])
        if p['lb'][j] > 0:
            yield dict(p, lb=p['lb'][:j] + [0.0] + p['lb'][j + 1:])
        if np.isfinite(p['ub'][j]) and p['ub'][j] != 1000:
            yield dict(p, ub=p['ub'][:j] + [INF] + p['ub'][j + 1:])
            yield dict(p, ub=p['ub'][:j] + [1000.0] + p['ub'][j + 1:])
    for i, row in enumerate(p['S']):
        for j, a in enumerate(row):
            if abs(a) > 1:
                S = [list(r) for r in p['S']]
                S[i][j] = a / abs(a)
                yield dict(p, S=S)


def shrink(problem, fails, max_tests=400):
    """Simplify a problem step by step while `fails(problem)` stays true.

    fails: callable(problem) -> bool; has to return False for a problem that is not well-posed.
    Returns (smallest failing problem found, number of candidates tried).
    """
    tried = 0
    progress = True
    while progress and tried < max_tests:
        progress = False
        for cand in _smaller(problem):
            tried += 1
            try:
                ok = _well_posed(cand) and fails(cand)
            except Exception:
                ok = False
            if ok:
                problem, progress = cand, True
                break
            if tried >= max_tests:
                break
    return problem, tried


def backend_fails(backend, lp='highs', like=None):
    """Predicate for :func:`shrink`: the backend's result is neither the reference nor a tie.
    like: 'error' or 'result' restricts it to failures of the same sort as the original one."""

    def fails(problem):
        verdict = compare(problem, [backend], lp)['verdict'][backend]
        if verdict in ('ok', 'tie'):
            return False
        is_error = verdict.startswith('error') or verdict.startswith('status')
        if 'There is no feasible solution' in verdict or 'zero vector' in verdict:
            return False  # the package rejects the problem as ill-posed: nothing to compare
        return like is None or (like == 'error') == is_error

    return fails


# ------------------------------------------------------------------------------------------------
# reproducers: a script that carries the problem and everything needed to judge it
# ------------------------------------------------------------------------------------------------
_REPRODUCER = '''#!/usr/bin/env python3
"""{title}

{note}

Run:  SD_WT=<straindesign checkout> python {file}
Needs straindesign (with the LP engine), cobra and scipy; nothing else. The reference is computed
here by brute force: every intervention set within the budget is decided by plain LPs (HiGHS).
"""
import itertools, logging, os, sys, warnings
warnings.filterwarnings('ignore')
if os.environ.get('SD_WT'):
    sys.path.insert(0, os.environ['SD_WT'])
import numpy as np
logging.disable(logging.WARNING)
inf = INF = float('inf')
DELTA, OPT_SLACK, VALUE_TOL, TIME_LIMIT = {DELTA!r}, {OPT_SLACK!r}, {VALUE_TOL!r}, {TIME_LIMIT!r}
SUPPRESS, PROTECT, OPTKNOCK, ROBUSTKNOCK, OPTCOUPLE = 'suppress', 'protect', 'optknock', 'robustknock', 'optcouple'
OPTIMISATION = (OPTKNOCK, ROBUSTKNOCK, OPTCOUPLE)

PROBLEM = {problem}

BACKENDS = {backends!r}  # name -> (solver, engine)


{source}


if __name__ == '__main__':
    import straindesign
    print('straindesign from', os.path.dirname(straindesign.__file__))
    out = compare(PROBLEM, list(BACKENDS))
    ref = out['reference']
    if 'designs' in ref:
        print('brute force: %d designs  %s' % (len(ref['designs']), sorted(sorted(z) for z in ref['designs'])))
    else:
        print('brute force: optimum %s at cost %s' % (ref['value'], ref['cost']))
    bad = 0
    for name, verdict in out['verdict'].items():
        res = out['results'][name]
        shown = res.get('error') or sorted(sorted(z) for z in res['designs'])
        claimed = '' if res.get('claimed') is None else '  claimed value %.9g' % res['claimed']
        print('%-11s %s%s\\n            -> %s' % (name, shown, claimed, verdict))
        bad += verdict not in ('ok', 'tie')
    sys.exit(1 if bad else 0)
'''


def write_reproducer(problem, path, backends, title, note=''):
    """Write a script that states the problem, runs the named backends and the brute force, and
    exits with 1 while a backend's result is neither the reference nor a tie."""
    import inspect
    import pprint
    parts = [ProblemError, _HighsLP, interventions, expr_str, constraint_str, BruteForce, build_model, build_modules, package_arguments,
             run_engine_alone, run_package, _Verdicts, _judge_enumeration, _close, _within, _judge_optimisation, compare]
    source = '\n\n'.join(inspect.getsource(obj) for obj in parts)
    source = '_LP_CLASSES, _LAST_BRUTE_FORCE = {}, {}\n\n\n' + source + "\n\n_LP_CLASSES['highs'] = _HighsLP\n"
    text = _REPRODUCER.format(title=title, note=note, file=os.path.basename(path), DELTA=DELTA, OPT_SLACK=OPT_SLACK, VALUE_TOL=VALUE_TOL,
                              TIME_LIMIT=TIME_LIMIT,
                              problem=pprint.pformat(problem, width=150, sort_dicts=False), backends={b: BACKENDS[b] for b in backends},
                              source=source)
    with open(path, 'w') as f:
        f.write(text)
    return path


# ------------------------------------------------------------------------------------------------
# command line: long campaigns
# ------------------------------------------------------------------------------------------------
def _flags(problem):
    """Short description of what a problem exercises."""
    out = [problem.get('bounds', '?'), 'genes' if problem.get('gene_kos') else 'reactions', 'compress' if problem.get('compress', True) else 'plain']
    costs = [c for _, c, _ in interventions(problem)]
    if any(c != 1 for c in costs):
        out.append('costs')
    if problem.get('gene_kos') and problem.get('ko_cost'):
        out.append('mixed')
    if any('inner_opt_tol' in m for m in problem['modules']):
        out.append('tol')
    if problem.get('retry'):
        out.append('retry')
    if any(op == '=' for m in problem['modules'] for _, op, _ in m.get('constraints') or []):
        out.append('equality')
    if any(len(m.get(k) or {}) > 1 for m in problem['modules'] for k in ('inner_objective', 'outer_objective')):
        out.append('objective2')
    if any(u <= 0 for u in problem['ub']):
        out.append('backwards')
    return out


def _use_checkout():
    """Put the checkout this file lies in (or SD_WT) ahead of an installed straindesign."""
    root = os.environ.get('SD_WT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path[:1]:
        sys.path.insert(0, root)


def _record(problem, out, t_gen, t_run):
    """One line of results.jsonl."""
    ref = out['reference']
    got = {}
    for b, res in out['results'].items():
        got[b] = 'error' if 'error' in res else [len(res['designs']), res.get('claimed')]
    rec = dict(seed=problem.get('seed'), kind=problem.get('kind'), flags=_flags(problem), n_rxns=len(problem['rxns']),
               n_itv=len(interventions(problem)), max_cost=problem['max_cost'], n_sets=ref['n_sets'], verdict=out['verdict'], got=got,
               t_gen=round(t_gen, 3), t=round(t_run, 3))
    if 'designs' in ref:
        rec['n_designs'] = len(ref['designs'])
    else:
        rec['value'], rec['cost'] = ref['value'], ref['cost']
    return rec


def main(argv=None):
    import argparse
    import resource
    _use_checkout()
    ap = argparse.ArgumentParser(description='Fuzz the strain design engines against a brute-force reference.')
    ap.add_argument('--seed', type=int, default=0, help='first seed')
    ap.add_argument('--n', type=int, default=100, help='number of problems')
    ap.add_argument('--step', type=int, default=1, help='seed increment (workers: --seed offset by the worker number, --step = number of workers)')
    ap.add_argument('--kinds', default=','.join(KINDS), help='module types, comma separated; a worker goes through them in turn')
    ap.add_argument('--deep', action='store_true', help='prefer large budgets (up to 5)')
    ap.add_argument('--rich', type=float, default=0.0, help='probability of passing over a problem with little to find')
    ap.add_argument('--exotic', type=float, default=0.0, help='probability, per place, of equalities, objectives over two reactions, '
                    'a SUPPRESS module next to an optimisation module, reactions written backwards')
    ap.add_argument('--backends', default=None, help='comma separated; default: all installed (%s)' % ', '.join(BACKENDS))
    ap.add_argument('--bf', default='highs', choices=sorted(_LP_CLASSES), help='LP solver of the brute force')
    ap.add_argument('--cross', type=float, default=0.0, help='fraction of problems on which the brute force is repeated with the other LP solver')
    ap.add_argument('--out', required=True, help='directory for results_*.jsonl and the problems with a finding')
    ap.add_argument('--seconds', type=float, default=INF, help='stop after this much time')
    ap.add_argument('--max-sets', type=int, default=500, help='largest number of intervention sets within the budget')
    args = ap.parse_args(argv)
    logging.disable(logging.WARNING)
    import warnings
    warnings.filterwarnings('ignore')
    os.makedirs(args.out, exist_ok=True)
    kinds = args.kinds.split(',')
    backends = args.backends.split(',') if args.backends else installed_backends()
    tag = 'w%d_of_%d' % (args.seed % args.step, args.step)
    import straindesign
    code = os.path.dirname(os.path.dirname(os.path.abspath(straindesign.__file__)))
    rev = os.environ.get('SD_REV') or os.path.basename(code)
    t_end = time.time() + args.seconds
    n_bad = n_done = 0
    with open(os.path.join(args.out, 'results_%s.jsonl' % tag), 'a') as log:

        def write(rec):
            rec['rss_mb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
            rec['rev'] = rev
            log.write(json.dumps(rec) + '\n')
            log.flush()

        for k in range(args.n):
            if time.time() > t_end:
                break
            seed = args.seed + k * args.step
            kind = kinds[(seed // args.step) % len(kinds)]
            t0 = time.time()
            try:
                problem = random_problem(seed, kind, max_sets=args.max_sets, deep=args.deep, lp=args.bf, rich=args.rich, exotic=args.exotic)
            except ProblemError as e:
                write(dict(seed=seed, kind=kind, skipped=str(e)))
                continue
            todo = [problem]
            while todo:
                problem = todo.pop()
                t1 = time.time()
                try:
                    out = compare(problem, backends, args.bf)
                except ProblemError as e:
                    write(dict(seed=seed, kind=kind, skipped=str(e)))
                    continue
                rec = _record(problem, out, t1 - t0, time.time() - t1)
                if args.cross and np.random.default_rng(seed).random() < args.cross:
                    ref = out['reference']
                    other = reference(problem, 'gurobi' if args.bf == 'highs' else 'highs')
                    same = (set(other['designs']) == set(ref['designs'])) if 'designs' in ref else \
                        (other['value'] is None) == (ref['value'] is None) and (ref['value'] is None or _close(other['value'], ref['value']))
                    rec['cross'] = 'ok' if same else 'the brute force differs between its LP solvers'
                bad = [b for b, v in out['verdict'].items() if v != 'ok'] + (['cross'] if rec.get('cross', 'ok') != 'ok' else [])
                n_done += 1
                if bad:
                    n_bad += 1
                    name = 'problem_%d%s.json' % (seed, '_plain' if problem.get('retry') else '')
                    with open(os.path.join(args.out, name), 'w') as f:
                        json.dump(problem, f)
                write(rec)
                if bad:
                    print('seed %d %s %s: %s' % (seed, kind, ' '.join(rec['flags']),
                                                 '; '.join('%s: %s' % (b, str(out['verdict'].get(b, rec.get('cross')))[:90]) for b in bad)), flush=True)
                # where the preprocessing of the compression fails for the MILP path, the engines
                # are still compared on the problem without compression
                first = out['verdict'].get('milp-cplex', '')
                if first.startswith('error') and problem.get('compress', True) and not problem.get('retry'):
                    todo.append(dict(problem, compress=False, retry=True))
    print('done: %d problems, %d with a finding' % (n_done, n_bad), flush=True)


if __name__ == '__main__':
    main()
