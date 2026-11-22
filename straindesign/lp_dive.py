#!/usr/bin/env python3
#
# Copyright 2022-2026 Max Planck Institute Magdeburg
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
#
"""Find single strain designs with the LP oracle, without enumerating, and polish each one (DRAFT).

This is solution_approach='any' of the LP engine (see lp_enumeration). It uses the same three
layers (interventions, Boolean layer, LP oracle per module) and the same kind of clauses, but
instead of visiting the intervention sets cost level by cost level it

1. **dives**: starting from no intervention, it asks the modules about the blocked set X; a module
   that is not fulfilled returns a clause (P, N), and the dive adds the interventions that block
   one atom of P, chosen by a score (clauses repaired per cost, whether the stored flux witness of
   every PROTECT-type module survives), with random tie-breaking. A dead end (empty P, budget
   used up) undoes the last choice; a dive gives up after a bounded number of nodes, and the next
   one starts with every clause learned so far.
2. **polishes** the design it reaches to a local optimum (see `DiveSearch.polish`).

Optimisation modules (OptKnock, RobustKnock, OptCouple) are handled by local search on the
objective value (see `OptSearch`).

Neither part proves anything about the designs it does not return: the designs are valid and
locally optimal, not the cheapest or the best.
"""

import logging
import random
import time
import numpy as np
from straindesign.names import *
from straindesign.lp_enumeration import (BitRows, ClassicModule, FEASIBLE, LPError, _IMPROVE, _OBJECTIVES, _LPBase, _integer_rows, _minimal,
                                         bits, build_problem, lp_counts, CompressedDesigns)


def _row_int(br, i):
    return int.from_bytes(br.a[i].tobytes(), 'little')


class _Stop(Exception):
    pass


class ClauseStore:
    """Clauses (P, N) over atoms: every design blocks an atom of P or leaves an atom of N
    unblocked. A blocked set X violates (P, N) if X & P == 0 and N <= X. Clauses learned under a
    weaker requirement stay valid under every stronger one."""

    def __init__(self, n_atoms):
        self.Pb, self.Nb = BitRows(n_atoms), BitRows(n_atoms)
        self.P, self.N = [], []
        self.size = []

    def add(self, P, N):
        self.Pb.add(P)
        self.Nb.add(N)
        self.P.append(P)
        self.N.append(N)
        self.size.append(P.bit_count() if P else 0)

    def violated(self, X):
        """Indices of the clauses X violates."""
        if not self.P:
            return np.zeros(0, int)
        xw = self.Pb.words(X)
        return np.flatnonzero(self.Pb.disjoint(xw) & self.Nb.within(xw))

    def hit(self, idx, D):
        """For the clauses idx: does the atom set D contain an atom of P?"""
        if not len(idx):
            return np.zeros(0, bool)
        return ~self.Pb.disjoint(self.Pb.words(D))[idx]


class _Oracle:
    """The modules, asked through stored clauses first and LPs second."""

    def __init__(self, modules, clauses, stat):
        self.modules, self.clauses, self.stat = list(modules), clauses, stat
        # a module whose violation cannot be repaired by more interventions is asked first in a
        # dive: there is no point in a clause of a SUPPRESS module at a dead end
        self.order = sorted(self.modules, key=lambda m: m.kind == SUPPRESS)

    def protect_atoms(self, X):
        """Atoms the stored witnesses of the PROTECT-type modules that hold at X run through: a
        SUPPRESS witness that keeps away from them gives a clause whose atoms can be blocked
        without losing those witnesses."""
        out = 0
        for mod in self.modules:
            if mod.kind == SUPPRESS or not mod.memo.P.n:
                continue
            xw = mod.memo.P.words(X)
            ok = np.flatnonzero(mod.memo.P.disjoint(xw) & mod.memo.N.within(xw))
            if len(ok):
                out |= _row_int(mod.memo.P, ok[-1])
        return out & ~X

    def check(self, X, avoid_s=0, avoid_p=0):
        """(True, None, None) if X fulfils every module, else (False, P, N) with a clause X
        violates. avoid_s steers SUPPRESS witnesses, avoid_p those of classic PROTECT modules."""
        v = self.clauses.violated(X)
        if len(v):
            self.stat['from_clauses'] += 1
            size = [self.clauses.size[i] for i in v]
            i = v[int(np.argmin(size))]  # a dead clause (P empty) first, then the smallest
            return False, self.clauses.P[i], self.clauses.N[i]
        for mod in self.order:
            if mod.decided(X):
                self.stat['memo'] += 1
                continue
            self.stat['queries'] += 1
            if isinstance(mod, ClassicModule):
                verdict, A = mod.region.decide(X, avoid_s if mod.kind == SUPPRESS else avoid_p)
                if verdict == FEASIBLE:
                    if mod.kind != SUPPRESS:
                        mod.memo.add(A, 0)
                        continue
                    P, N = A, 0
                else:
                    if mod.kind == SUPPRESS:
                        mod.memo.add(0, A)
                        mod.cores.append(A)
                        continue
                    P, N = 0, A
            else:
                ok, P, N = mod.check(X, avoid_s)
                if ok:
                    continue
            if not P and not N:
                stop = _Stop('a module cannot be fulfilled by any set of interventions')
                stop.proven = True
                raise stop
            self.clauses.add(P, N)
            return False, P, N
        return True, None, None


class DiveSearch:
    """Designs for SUPPRESS / PROTECT modules (with or without inner objective) by randomised
    dives over the clauses, each design polished to a local optimum.

    Args:
        layer (BoolLayer)
        modules (list): module oracles over the atoms of the layer
        max_cost (float): budget, may be infinite
        seed (int): makes the random choices repeatable
        time_limit (float): seconds
        dive_nodes (int): nodes a dive may visit before it gives up
        width (int): alternatives tried per node of a dive before it backs up further
        noise (float): relative random spread of the candidate scores (diversity between dives)
        break_penalty (float): score factor for a choice that blocks an atom of a PROTECT witness
        stall (int): consecutive dives without a new design after which the search ends
        swap_trials (float): LP checks the polishing may spend on the swap neighbourhood per design
    """

    def __init__(self, layer, modules, max_cost, seed=None, time_limit=np.inf, dive_nodes=400, width=3, noise=0.3, break_penalty=0.2, stall=8,
                 swap_trials=np.inf):
        self.layer, self.max_cost = layer, max_cost
        self.rng = random.Random(seed)
        self.deadline = time.time() + time_limit
        self.dive_nodes, self.width, self.noise, self.break_penalty = dive_nodes, width, noise, break_penalty
        self.stall, self.swap_trials = stall, swap_trials
        self.stat = dict(queries=0, memo=0, from_clauses=0, dives=0, failed_dives=0, nodes=0, backtracks=0, swaps=0, swap_checks=0,
                         swap_capped=0)
        self.clauses = ClauseStore(layer.nA)
        self.oracle = _Oracle(modules, self.clauses, self.stat)
        self.protect_mods = [m for m in modules if m.kind != SUPPRESS]
        self.found = []  # designs returned so far (intervention bit sets)
        self.log = []  # per design: (seconds since start, cost after the dive, cost after polishing, LP checks so far)

    # -- helpers ---------------------------------------------------------------------------------
    def _time_up(self):
        return time.time() > self.deadline

    def _excluded(self, z):
        """z contains a design returned before."""
        return any(d & z == d for d in self.found)

    def _is_design(self, z, X=None):
        X = self.layer.X(z) if X is None else X
        return self.oracle.check(X)[0]

    def _wanted(self, X):
        """Atoms of the positive parts of the clauses X violates: what a PROTECT witness should
        keep away from."""
        out = 0
        for i in self.clauses.violated(X):
            out |= self.clauses.P[i]
        return out & ~X

    def _candidates(self, z, X, cost, P):
        """Ways to block one atom of P, best first: (score, z2, X2, cost2)."""
        layer = self.layer
        left = self.max_cost - cost
        Js = set()
        for a in bits(P):
            Js.update(layer.completions(a, z, left))
        viol = self.clauses.violated(X)
        viol = viol[[bool(self.clauses.P[i]) for i in viol]] if len(viol) else viol
        out = []
        for J in _minimal(Js):
            z2 = z | J
            if self.found and self._excluded(z2):
                continue
            X2 = layer.extend(z, X, J)
            dead = [i for i in self.clauses.violated(X2) if not self.clauses.P[i]]
            if dead:
                continue
            gain = int(self.clauses.hit(viol, X2 & ~X).sum())
            cJ = layer.set_cost(J)
            score = (1.0 + gain) / cJ
            if not all(m.decided(X2) for m in self.protect_mods):
                score *= self.break_penalty
            score *= 1.0 + self.noise * self.rng.random()
            out.append((score, z2, X2, cost + cJ))
        out.sort(key=lambda t: -t[0])
        return out

    # -- dive ------------------------------------------------------------------------------------
    def dive(self):
        """One dive from the empty set; the design it reaches, or None."""
        self.stat['dives'] += 1
        layer = self.layer
        z, X, cost = 0, 0, 0.0
        frames = []  # [candidates, position]
        nodes = 0
        while True:
            if self._time_up():
                raise _Stop('time limit')
            nodes += 1
            self.stat['nodes'] += 1
            ok, P, N = self.oracle.check(X, self.oracle.protect_atoms(X), self._wanted(X))
            if ok:
                return z
            cands = self._candidates(z, X, cost, P) if P else []
            if cands and nodes < self.dive_nodes:
                frames.append([cands, 0])
                _, z, X, cost = cands[0]
                continue
            if nodes >= self.dive_nodes:
                return None
            # dead end: back up to the latest node with an untried alternative that no clause
            # learned since rules out
            self.stat['backtracks'] += 1
            while frames:
                f = frames[-1]
                f[1] += 1
                while f[1] < min(len(f[0]), self.width):
                    _, z2, X2, c2 = f[0][f[1]]
                    if not any(not self.clauses.P[i] for i in self.clauses.violated(X2)) and not (self.found and self._excluded(z2)):
                        break
                    f[1] += 1
                if f[1] < min(len(f[0]), self.width):
                    _, z, X, cost = f[0][f[1]]
                    break
                frames.pop()
            else:
                return None

    # -- polishing -------------------------------------------------------------------------------
    def minimise(self, z):
        """Drop interventions, the most expensive first, while z stays a design, until no single
        one can go."""
        cost = self.layer.cost
        changed = True
        while changed:
            changed = False
            for i in sorted(bits(z), key=lambda i: (-cost[i], self.rng.random())):
                z2 = z & ~(1 << i)
                if self._is_design(z2):
                    z, changed = z2, True
            if all(m.__class__ is ClassicModule for m in self.oracle.modules):
                break  # one pass suffices: a set that fails SUPPRESS fails it below as well
        return z

    def _swap_round(self, z):
        """A design obtained by replacing one or two interventions of z with one intervention of
        lower total cost, or None if there is none. Exhaustive unless swap_trials runs out: the
        clause of z minus the removed ones names every intervention that could repair it."""
        layer = self.layer
        cost = layer.cost
        members = list(bits(z))
        self.rng.shuffle(members)
        pairs = [(i,) for i in members] + [(i, j) for k, i in enumerate(members) for j in members[k + 1:]]
        for R in pairs:
            cR = sum(cost[i] for i in R)
            if cR <= min(cost):
                continue
            z1 = z
            for i in R:
                z1 &= ~(1 << i)
            X1 = layer.X(z1)
            ok, P, N = self.oracle.check(X1)
            if ok:
                return z1
            if not P:
                continue
            G = 0
            for a in bits(P):
                G |= layer.singles(a, z1)
            G &= ~z
            for g in sorted(bits(G), key=lambda g: (cost[g], self.rng.random())):
                if cost[g] >= cR or layer.set_cost(z1) + cost[g] > self.max_cost:
                    continue
                z2 = z1 | (1 << g)
                if self.found and self._excluded(z2):
                    continue
                X2 = layer.extend(z1, X1, 1 << g)
                if len(self.clauses.violated(X2)):
                    continue
                if self.stat['swap_checks'] >= self._swap_budget:
                    self.stat['swap_capped'] += 1
                    return None
                self.stat['swap_checks'] += 1
                if self.oracle.check(X2)[0]:
                    return z2
        return None

    def polish(self, z):
        """Bring a design to a local optimum: no single intervention can be removed, and no one or
        two interventions can be replaced by a single intervention of lower total cost, without
        losing the design (among the sets that contain no design returned earlier). Both
        neighbourhoods are searched completely unless swap_trials runs out, which
        `stat['swap_capped']` records."""
        self._swap_budget = self.stat['swap_checks'] + self.swap_trials
        z = self.minimise(z)
        while not self._time_up():
            z2 = self._swap_round(z)
            if z2 is None:
                break
            self.stat['swaps'] += 1
            z = self.minimise(z2)
        return z

    def run(self, max_solutions=1):
        """Returns (designs as bit sets, proven: True if a module was shown to be unfulfillable)."""
        t0 = time.time()
        fails = 0
        try:
            if self._is_design(0):
                self.found, self.trivial = [0], True
                return self.found, False
            while len(self.found) < max_solutions and fails < self.stall:
                z = self.dive()
                if z is None:
                    fails += 1
                    self.stat['failed_dives'] += 1
                    continue
                c0 = self.layer.set_cost(z)
                z = self.polish(z)
                if z in self.found or self._excluded(z):
                    fails += 1
                    continue
                fails = 0
                self.found.append(z)
                self.log.append((time.time() - t0, c0, self.layer.set_cost(z), self.stat['queries']))
                logging.info('  LP dive: design %d, cost %g after the dive, %g after polishing (%.1f s)' %
                             (len(self.found), c0, self.layer.set_cost(z), time.time() - t0))
        except _Stop as e:
            logging.info('  LP dive stopped: %s' % e)
            if getattr(e, 'proven', False):
                return [], True
        return self.found, False


class OptSearch:
    """Local search for an optimisation module (OptKnock, RobustKnock, OptCouple) under the cost
    budget, with the feasibility modules respected.

    A move goes from z to a neighbour z' of cost <= max_cost that fulfils every module and has a
    higher objective value; neighbours are z plus one intervention, z minus one, and z with one
    intervention exchanged for another. The search asks whether any neighbour reaches the value
    threshold t (the objective's modules at t), so every clause learned at a lower threshold
    prunes the candidates of a higher one. A design is returned at a *local optimum*: no neighbour
    improves its value, and no intervention can be removed without lowering the value (beyond a
    relative 1e-6). Restarts rebuild from a random part of the best design so far.
    """

    def __init__(self, layer, modules, objective, max_cost, seed=None, time_limit=np.inf, restarts=4, **dive_kw):
        self.layer, self.objective, self.max_cost = layer, objective, max_cost
        self.rng = random.Random(seed)
        self.base = getattr(objective, 'min_gcp', None)
        self.ds = DiveSearch(layer, list(modules) + objective.modules, max_cost, seed=seed, time_limit=time_limit, **dive_kw)
        self.restarts = restarts
        self.trace = []  # (seconds, value, cost) after each accepted move

    def value(self, z):
        return self.objective.value(self.layer.X(z))

    def _set(self, t):
        self.objective.set_threshold(t)

    def _check(self, z, X=None):
        return self.ds.oracle.check(self.layer.X(z) if X is None else X)

    def _improving(self, z):
        """A neighbour of z that fulfils every module at the current threshold, or None."""
        layer, cost = self.layer, self.layer.cost
        cz = layer.set_cost(z)
        X = layer.X(z)
        cl = self.ds.clauses

        def repairs(z1, X1, P, exclude):
            G = 0
            for a in bits(P):
                G |= layer.singles(a, z1)
            G &= ~z1 & ~exclude
            c1 = layer.set_cost(z1)
            for g in sorted(bits(G), key=lambda g: (cost[g], self.rng.random())):
                if c1 + cost[g] > self.max_cost:
                    continue
                X2 = layer.extend(z1, X1, 1 << g)
                if len(cl.violated(X2)):
                    continue
                if self.ds.oracle.check(X2)[0]:
                    return z1 | (1 << g)
            return None

        ok, P, N = self.ds.oracle.check(X)
        if ok:
            return z
        if P and cz + min(cost) <= self.max_cost:
            z2 = repairs(z, X, P, 0)
            if z2 is not None:
                return z2
        members = list(bits(z))
        self.rng.shuffle(members)
        clauses1 = []
        for i in members:
            z1 = z & ~(1 << i)
            X1 = layer.X(z1)
            ok, P1, _ = self.ds.oracle.check(X1)
            if ok:
                return z1
            clauses1.append((i, z1, X1, P1))
        for i, z1, X1, P1 in clauses1:
            if P1:
                z2 = repairs(z1, X1, P1, 1 << i)
                if z2 is not None:
                    return z2
        return None

    def local_search(self, z, val, t0):
        while not self.ds._time_up():
            self._set(val + _IMPROVE * max(1.0, abs(val)))
            z2 = self._improving(z)
            if z2 is None:
                break
            v2 = self.value(z2)
            if v2 is None or v2 <= val:
                raise LPError('a neighbour that should improve the objective does not')
            z, val = z2, v2
            self.trace.append((time.time() - t0, val, self.layer.set_cost(z)))
        # interventions that do not contribute to the value go
        self._set(val - _IMPROVE * max(1.0, abs(val)))
        z = self.ds.minimise(z)
        self._set(self.base)
        return z, self.value(z)

    def run(self, max_solutions=1):
        """Returns (list of (design, value), best first; proven infeasible)."""
        t0 = time.time()
        ds = self.ds
        self._set(self.base)
        optima = {}
        try:
            z = ds.dive() if not ds._is_design(0) else 0
            tries = 0
            while z is None and tries < ds.stall:
                z, tries = ds.dive(), tries + 1
            if z is None:
                return [], False
            v = self.value(z)
            z, v = self.local_search(z, v, t0)
            optima[z] = v
            logging.info('  LP local search: value %.6g at cost %g' % (v, self.layer.set_cost(z)))
            for _ in range(self.restarts):
                if ds._time_up():
                    break
                best = max(optima, key=optima.get)
                members = list(bits(best))
                keep = [i for i in members if self.rng.random() < 0.5]
                z1 = 0
                for i in keep:
                    z1 |= 1 << i
                self._set(self.base)
                if not ds._is_design(z1):
                    continue
                z1, v1 = self.local_search(z1, self.value(z1), t0)
                optima[z1] = max(v1, optima.get(z1, -np.inf))
                logging.info('  LP local search, restart: value %.6g at cost %g' % (v1, self.layer.set_cost(z1)))
        except _Stop as e:
            logging.info('  LP local search stopped: %s' % e)
            if getattr(e, 'proven', False):
                return [], True
            self._set(self.base)
        out = sorted(optima.items(), key=lambda t: -t[1])
        return out[:int(max_solutions) if np.isfinite(max_solutions) else None], False


def lp_dive_engine(cmp_model, sd_modules, ko_cost, ki_cost, gko_cost, gki_cost, has_gene_names, max_cost, solver, time_limit=np.inf,
                   max_solutions=np.inf, seed=None, exact=False, **kwargs):
    """Entry point for compute_strain_designs with solution_approach='any'.

    Status: OPTIMAL if max_solutions designs were found (for an optimisation module: when the
    local search ended at a local optimum before the time limit), TIME_LIMIT_W_SOL if the search
    stopped earlier (time limit, or several dives in a row without a new design) with designs,
    TIME_LIMIT without, and INFEASIBLE only when a module was shown to be unfulfillable by any set
    of interventions.

    Returns:
        (CompressedDesigns, search object)
    """
    if ki_cost or gki_cost:
        raise NotImplementedError('the LP engine does not handle knock-ins yet')
    costs = list(ko_cost.values()) + list((gko_cost or {}).values())
    if any(c <= 0 for c in costs):
        raise NotImplementedError('the LP engine needs positive intervention costs')
    opt_mods = [mod for mod in sd_modules if mod[MODULE_TYPE] in _OBJECTIVES]
    if any(mod[MODULE_TYPE] not in (SUPPRESS, PROTECT) and mod[MODULE_TYPE] not in _OBJECTIVES for mod in sd_modules) or len(opt_mods) > 1:
        raise NotImplementedError('the LP engine handles SUPPRESS, PROTECT, OptKnock, RobustKnock and OptCouple modules')
    from straindesign.parse_constr import lineqlist2mat
    t0 = time.time()
    n_lp0 = _LPBase.n_solves
    names, layer, modules, atom_rxn, ctx = build_problem(cmp_model, sd_modules, ko_cost, gko_cost, max_cost, solver,
                                                         'name' if has_gene_names else 'id', exact=exact)
    objective = None
    if opt_mods:
        mod = opt_mods[0]
        V = lineqlist2mat(mod[CONSTRAINTS], ctx['rxn_ids'])
        objective = _OBJECTIVES[mod[MODULE_TYPE]](ctx, mod, _integer_rows(*V) if exact else V)
    logging.info('  LP dive: %d interventions over %d knockable reactions of %d (set up in %.1f s).' %
                 (layer.nI, layer.nA, len(cmp_model.reactions), time.time() - t0))
    if objective is None:
        search = DiveSearch(layer, modules, max_cost, seed=seed, time_limit=time_limit, **kwargs)
        found, proven = search.run(max_solutions)
        values = None
        done = len(found) >= max_solutions or getattr(search, 'trivial', False)
        all_modules = modules
    else:
        search = OptSearch(layer, modules, objective, max_cost, seed=seed, time_limit=time_limit, **kwargs)
        res, proven = search.run(max_solutions)
        found, values = [z for z, _ in res], [v for _, v in res]
        done = bool(found) and not search.ds._time_up()
        search.objective_value = values[0] if values else None
        all_modules = modules + objective.modules
    stat = search.stat if objective is None else search.ds.stat
    stat['seconds'] = time.time() - t0
    stat['lp_flux'], stat['lp_proof'], stat['lp_value'] = lp_counts(all_modules)
    stat['lp_solves'] = _LPBase.n_solves - n_lp0
    search.names, search.atom_rxn, search.values = names, atom_rxn, values
    sd = [{names[i]: -1.0 for i in bits(z)} for z in found]
    if proven:
        status = INFEASIBLE
    elif sd:
        status = OPTIMAL if done else TIME_LIMIT_W_SOL
    else:
        status = TIME_LIMIT
    setup = {MODEL_ID: cmp_model.id, MAX_COST: max_cost, SOLVER: solver, SOLUTION_APPROACH: ANY, KOCOST: dict(ko_cost), KICOST: {}, MODULES: sd_modules,
             TIME_LIMIT: time_limit, MAX_SOLUTIONS: max_solutions}
    logging.info('  LP dive: %d designs, status %s; %d LPs; %.1f s.' % (len(sd), status, stat['lp_solves'], stat['seconds']))
    return CompressedDesigns(sd, status, setup), search
