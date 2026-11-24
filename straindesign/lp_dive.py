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

0. looks for **cheap designs first**: the complete search of lp_enumeration walks the lowest cost
   levels for a short while (`first_levels`, `first_seconds` of `lp_dive_engine`). Where designs
   of one to three interventions exist it returns them within seconds, and they are the cheapest
   there are. Dives follow for what is still asked for.
1. **dives**. With classic SUPPRESS / PROTECT modules a *guided dive* asks for a whole cut set at
   once: a flux witness of each PROTECT module that also violates every SUPPRESS module is kept
   alive, and one proof LP per SUPPRESS region, with every other affordable atom blocked, names
   cheap atoms whose blocking empties the region (see `DiveSearch.guided_dive`). Otherwise, and
   when guided dives fail, a *clause dive* asks the modules about the blocked set X; a module
   that is not fulfilled returns a clause (P, N), and the dive adds the interventions that block
   one atom of P, chosen by a score (clauses repaired per cost, whether the stored flux witness of
   every PROTECT-type module survives), with random tie-breaking. A dead end (empty P, budget
   used up) undoes the last choice; a dive gives up after a bounded number of nodes, and the next
   one starts with every clause learned so far.
2. **polishes** the design it reaches to a local optimum (see `DiveSearch.polish`).
3. looks for **cheaper designs among the interventions the dives touched**: the complete search
   again, over that small set of interventions (`DiveSearch.cheaper_among_seen`).

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
from straindesign.lp_enumeration import (BitRows, ClassicModule, FEASIBLE, LPEnumerator, LPError, Region, _IMPROVE, _OBJECTIVES, _LPBase,
                                         _integer_rows, _minimal, bits, build_problem, lp_counts, CompressedDesigns)


def _row_int(br, i):
    return int.from_bytes(br.a[:, i].tobytes(), 'little')


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
            if mod.kind == SUPPRESS or mod.__class__ is not ClassicModule or not mod.memo.P.n:
                continue
            xw = mod.memo.P.words(X)
            ok = np.flatnonzero(mod.memo.P.disjoint(xw) & mod.memo.N.within(xw))
            if len(ok):
                out |= _row_int(mod.memo.P, ok[-1])
        return out & ~X

    def check(self, X, avoid_s=0, avoid_p=0, clean=False):
        """(True, None, None) if X fulfils every module, else (False, P, N) with a clause X
        violates. avoid_s steers SUPPRESS witnesses, avoid_p those of classic PROTECT modules.
        clean: the caller has just seen that X violates no stored clause."""
        v = [] if clean else self.clauses.violated(X)
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


class Target:
    """Where a flux that keeps a PROTECT module fulfilled has to lie once every SUPPRESS module is
    fulfilled: in the PROTECT region and outside each SUPPRESS region, so it violates one row of
    every SUPPRESS module. The region holds the PROTECT module's rows plus every row of the
    SUPPRESS modules turned around (a.v >= rhs for a.v <= rhs; both directions for an equality),
    each with a threshold that is set per query.

    Args:
        region (Region): PROTECT rows fixed, the turned-around rows as threshold rows
        rows (list): per threshold row (index of the SUPPRESS module, right-hand side)
        n_sup (int): number of SUPPRESS modules
    """

    def __init__(self, region, rows, n_sup):
        self.region, self.rows, self.n_sup = region, rows, n_sup

    def thetas(self, rng, stat):
        """Thresholds that ask for one violated row per SUPPRESS module, each violated by a random
        fraction (between a quarter and three quarters) of the largest violation the region allows
        together with the rows chosen before; None if some SUPPRESS module has no row the region
        can violate."""
        R = self.region
        th = [None] * len(self.rows)
        for j in range(self.n_sup):
            ks = [k for k, (jj, _) in enumerate(self.rows) if jj == j]
            rng.shuffle(ks)
            for k in ks:
                rhs = self.rows[k][1]
                stat['guide_lps'] += 1
                try:
                    val, _, _ = R.maximize(k, 0, th)
                except LPError:
                    continue
                if val is None:
                    return None  # the PROTECT region is empty with the rows chosen so far
                gap = min(val, rhs + 1e3 * max(1.0, abs(rhs))) - rhs  # an unbounded violation counts as a large one
                if gap > 1e-6 * max(1.0, abs(rhs)):
                    th[k] = rhs + (0.25 + 0.5 * rng.random()) * gap
                    break
            else:
                return None
        return th


def target_regions(ctx, sd_modules, modules):
    """A Target per classic PROTECT module, when every SUPPRESS module is classic; else None."""
    from straindesign.parse_constr import lineqlist2mat
    pairs = [(sm, m) for sm, m in zip([sm for sm in sd_modules if sm[MODULE_TYPE] in (SUPPRESS, PROTECT)], modules)]
    sup = [sm for sm, m in pairs if m.kind == SUPPRESS]
    if not sup or any(m.__class__ is not ClassicModule for _, m in pairs):
        return None
    if ctx.get('exact'):
        from straindesign.lp_enumeration import _integer_rows
        mat = lambda c: _integer_rows(*lineqlist2mat(c, ctx['rxn_ids']))
    else:
        mat = lambda c: lineqlist2mat(c, ctx['rxn_ids'])
    rows, info = [], []
    for j, sm in enumerate(sup):
        A, b, Ae, be = mat(sm[CONSTRAINTS])
        A, Ae = A.toarray(), Ae.toarray()
        for i in range(A.shape[0]):
            rows.append(A[i])
            info.append((j, float(b[i])))
        for i in range(Ae.shape[0]):
            rows += [Ae[i], -Ae[i]]
            info += [(j, float(be[i])), (j, -float(be[i]))]
    out = {}
    for sm, m in pairs:
        if m.kind == SUPPRESS:
            continue
        R = Region(ctx['S'], ctx['lb'], ctx['ub'], *mat(sm[CONSTRAINTS]), ctx['atom_rxn'], ctx['solver'], rows=rows)
        out[id(m)] = Target(R, info, len(sup))
    return out


class DiveSearch:
    """Designs for SUPPRESS / PROTECT modules (with or without inner objective) by randomised
    dives, each design polished to a local optimum. Guided dives (see `guided_dive`) need classic
    modules; clause dives (see `dive`) work for every module and take over when guided dives
    give nothing.

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
        guided (bool): try guided dives (classic SUPPRESS / PROTECT modules only) before clause dives
        guide_rounds (int): rounds of one guided dive
        keep_price (float): factor on the blocking cost of an atom a PROTECT witness uses, in the
            proof LP that tells which kept atoms are expensive to keep
        probes (int): guided dives whose designs are pooled; the cheapest is polished first
        redives (int): re-dives from a design with a fifth of it dropped, tried by the polishing
        targets (dict): id of a PROTECT module -> Target (see `target_regions`)
    """

    def __init__(self, layer, modules, max_cost, seed=None, time_limit=np.inf, dive_nodes=400, width=3, noise=0.3, break_penalty=0.2, stall=8,
                 swap_trials=500, guided=True, guide_rounds=30, keep_price=4.0, probes=8, redives=20, targets=None):
        self.layer, self.max_cost = layer, max_cost
        self.rng = random.Random(seed)
        self.deadline = time.time() + time_limit
        self.dive_nodes, self.width, self.noise, self.break_penalty = dive_nodes, width, noise, break_penalty
        self.stall, self.swap_trials = stall, swap_trials
        self.guided, self.guide_rounds, self.targets = guided, guide_rounds, targets or {}
        self.probes, self.pool, self.keep_price, self.redives = probes, [], keep_price, redives
        # guided dives may overshoot the budget (polishing often brings the design back into it)
        self.guide_budget = np.inf
        self._open0 = {}  # forbidden set -> per atom the (cost, blocked atoms) of its completions from the empty set
        self.stat = dict(queries=0, memo=0, from_clauses=0, dives=0, failed_dives=0, nodes=0, backtracks=0, swaps=0, swap_checks=0,
                         swap_capped=0, guided=0, guide_lps=0, over_budget=0, redives=0, redive_wins=0)
        self.clauses = ClauseStore(layer.nA)
        self.oracle = _Oracle(modules, self.clauses, self.stat)
        self.protect_mods = [m for m in modules if m.kind != SUPPRESS]
        self.found = []  # designs returned so far (intervention bit sets)
        self.seen = 0  # every intervention of a design a dive reached or the polishing produced
        self.log = []  # per design: (seconds since start, cost after the dive, cost after polishing, LP checks so far)

    # -- helpers ---------------------------------------------------------------------------------
    def take_over(self, enum, designs, seconds):
        """Start from what a complete search of the lowest cost levels found (designs, to be
        returned first and never again) and learned (its clauses)."""
        for P, N in zip(enum.P, enum.Nn):
            self.clauses.add(P, N)
        for N in enum.N:
            self.clauses.add(0, N)
        self.stat['level_queries'] = enum.stat['queries']
        for z in designs:
            self.found.append(z)
            c = self.layer.set_cost(z)
            self.log.append((seconds, c, c, self.stat['queries']))
        if designs == [0] or (designs and designs[0] == 0):
            self.trivial = True

    def cheaper_among_seen(self, seconds, max_solutions=1):
        """Look for designs cheaper than the cheapest one found, among the interventions that
        occur in any design the dives reached: the complete search of lp_enumeration over that
        small set of interventions, with every clause learned so far.

        Returns (designs, complete). complete without designs is a certificate: no set of these
        interventions that costs less than the cheapest design found is a design. The designs it
        does find are the cheapest there are among these interventions, cheapest first."""
        dived = [z for z in self.found if z]
        costs = self.layer.cost
        if not dived or not self.seen or any(abs(c - round(c)) > 1e-9 for c in costs):
            return [], False
        below = min(self.layer.set_cost(z) for z in dived) - min(costs[i] for i in bits(self.seen))
        if below < 1:
            return [], True
        sub = self.layer.restricted(self.seen, below)
        enum = LPEnumerator(sub, self.oracle.modules, below, time_limit=min(seconds, max(0.0, self.deadline - time.time())),
                            max_solutions=max_solutions)
        for P, N in zip(self.clauses.P, self.clauses.N):
            if P or N:
                enum._add_clause(P, N)
        designs, complete = enum.run()
        self.stat['union_size'], self.stat['union_queries'], self.stat['union_complete'] = self.seen.bit_count(), enum.stat['queries'], complete
        return list(designs), complete

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

    # -- proof-guided dive ------------------------------------------------------------------------
    def _open_atoms(self, z, X, keep, forbid):
        """Atoms that may still be blocked: not blocked, not kept, and with an affordable
        completion that uses no forbidden intervention and blocks no kept atom. Returns the atoms
        and the cost of each one's cheapest such completion."""
        layer = self.layer
        U, w = 0, {}
        if not z and not X:
            if forbid not in self._open0:
                opts = {}
                for a in range(layer.nA):
                    o = sorted((layer.set_cost(J), layer.extend(0, 0, J)) for J in layer.completions(a, 0, self.guide_budget) if not J & forbid)
                    if o:
                        opts[a] = o
                self._open0 = {forbid: opts}
            for a, o in self._open0[forbid].items():
                if keep >> a & 1:
                    continue
                c = next((c for c, XJ in o if not XJ & keep), None)
                if c is not None:
                    U |= 1 << a
                    w[a] = c
            return U, w
        left = self.guide_budget - layer.set_cost(z)
        for a in range(layer.nA):
            if (X | keep) >> a & 1:
                continue
            best = np.inf
            for J in layer.completions(a, z, left):
                if J & forbid:
                    continue
                c = layer.set_cost(J)
                if c < best and not layer.extend(z, X, J) & keep:
                    best = c
            if np.isfinite(best):
                U |= 1 << a
                w[a] = best
        return U, w

    def _weighted_core(self, region, X, weights):
        """An infeasibility proof of the region with the atoms X blocked whose zero bounds cost
        little: the proof LP's multipliers of the atoms weighted by what blocking them costs (with
        random spread). None if the region has a point even then."""
        lp = region.proof_lp
        cols, vals = [], []
        for a, c in weights.items():
            f = c * (1.0 + self.noise * self.rng.random())
            for col in (region.mu_f[a], region.mu_b[a]):
                if col >= 0:
                    cols.append(col)
                    vals.append(f)
        lp.set_obj(cols, vals)
        try:
            return region.core(X)
        except LPError:
            return None
        finally:
            lp.set_obj(cols, [1.0] * len(cols))

    def _irreducible(self, region, X, C, w):
        """The core C with its atoms beyond X thinned out (the most expensive first) while the
        region stays empty: blocking fewer of them on top of X would leave a point. Returns the
        core the last proof rests on, the atoms of X it uses included."""
        need = C & ~X
        for a in sorted(bits(need), key=lambda a: (-w.get(a, 0.0), self.rng.random())):
            if not need >> a & 1:
                continue
            self.stat['guide_lps'] += 1
            try:
                C2 = region.core(X | (need & ~(1 << a)))
            except LPError:
                continue
            if C2 is not None:
                C, need = C2, C2 & ~X
        return C

    def _block_all(self, z, X, atoms, keep, forbid):
        """z extended by interventions that block every atom of `atoms` without blocking a kept
        atom, cheapest completion first (random spread); None if the budget does not suffice."""
        layer = self.layer
        todo = list(bits(atoms & ~X))
        self.rng.shuffle(todo)
        for a in todo:
            if X >> a & 1:
                continue
            left = self.guide_budget - layer.set_cost(z)
            opts = []
            for J in layer.completions(a, z, left):
                if J & forbid:
                    continue
                X2 = layer.extend(z, X, J)
                if not X2 & keep:
                    opts.append((layer.set_cost(J) * (1.0 + self.noise * self.rng.random()), J, X2))
            if not opts:
                return None, None
            _, J, X = min(opts, key=lambda t: t[0])
            z |= J
        return z, X

    def guided_dive(self, z0=0, rounds=None, forbid=0):
        """A dive for classic SUPPRESS / PROTECT modules that asks for whole cut sets instead of
        single clauses, alternating between the two kinds of module. One round:

        1. every PROTECT module gives a flux witness that keeps away from the atoms H (from its
           Target region when there is one: a point that also violates a row of every SUPPRESS
           module, which a flux surviving a design has to do); the atoms the witnesses use are
           *kept*, so the PROTECT modules stay fulfilled without any further LP;
        2. per SUPPRESS module, a proof LP with every affordable atom blocked returns an
           infeasibility proof whose multipliers are weighted by the cost of the cheapest
           completion, kept atoms at `keep_price` times that cost. If the proof needs no kept
           atom, it is thinned out to an irreducible set and blocked, and the round ends with a
           design. Otherwise the kept atoms it uses join H (they are expensive to keep) and the
           same proof without kept atoms gives the round's fallback design.
        3. A SUPPRESS region that keeps a point even with every unkept atom blocked runs through
           kept atoms only: those join H.

        H only grows; the dive ends with the first design whose proofs need no kept atom, or after
        `guide_rounds` rounds or a round that adds nothing to H, with the cheapest fallback design.

        Args:
            z0 (int): interventions the dive starts from (the polishing re-dives from part of a design)
            rounds (int): rounds, default guide_rounds
            forbid (int): interventions the dive must not use

        Returns the design, or None (also when the modules are of another kind)."""
        mods = self.oracle.modules
        sup = [m for m in mods if m.kind == SUPPRESS]
        pro = [m for m in mods if m.kind != SUPPRESS]
        if not sup or any(m.__class__ is not ClassicModule for m in mods):
            return None
        self.stat['guided'] += 1
        layer = self.layer
        for d in self.found:  # one intervention of each design returned so far: the dive cannot contain one again
            forbid |= 1 << self.rng.choice(list(bits(d & ~z0 or d)))
        X0 = layer.X(z0)
        thetas = {}
        for p in pro:
            T = self.targets.get(id(p))
            thetas[id(p)] = T.thetas(self.rng, self.stat) if T is not None else None
        H, best = 0, None
        for _ in range(self.guide_rounds if rounds is None else rounds):
            if self._time_up():
                raise _Stop('time limit')
            keep = 0
            for p in pro:
                T, th = self.targets.get(id(p)), thetas[id(p)]
                W = T.region.witness(X0, avoid=H, thetas=th) if th is not None else p.region.witness(X0, avoid=H)
                self.stat['guide_lps'] += 1
                if W is None:
                    return best
                p.memo.add(W, 0)  # a point of the target region is one of the PROTECT region
                keep |= W
            H0 = H
            z, X, clean = z0, X0, True
            for s in sup:
                U, w = self._open_atoms(z, X, keep, forbid)
                if keep and clean:
                    K, wk = self._open_atoms(z, X, 0, forbid)
                    wk = {a: c * (self.keep_price if keep >> a & 1 else 1.0) for a, c in wk.items()}
                    C = self._weighted_core(s.region, X | K, wk)
                    self.stat['guide_lps'] += 1
                    if C is not None and C & keep:
                        H |= C & keep
                        clean = False
                C = self._weighted_core(s.region, X | U, w)
                self.stat['guide_lps'] += 1
                logging.debug('guided: |H| %d, |keep| %d, |U| %d, core %s' % (H.bit_count(), keep.bit_count(), U.bit_count(), None if C is None else C.bit_count()))
                if C is None:
                    S = s.region.witness(X | U)
                    self.stat['guide_lps'] += 1
                    if S is not None:
                        self.clauses.add(S, 0)  # every design blocks an atom of this point
                        H |= S & keep
                    z = None
                    break
                C = self._irreducible(s.region, X, C, w)
                s.memo.add(0, C)
                s.cores.append(C)
                z, X = self._block_all(z, X, C, keep, forbid)
                if z is None:
                    break
            if z is not None and (best is None or layer.set_cost(z) < layer.set_cost(best)):
                if self.oracle.check(X)[0]:
                    best = z
            logging.debug('guided: round cost %s, clean %s, best %s' % (None if z is None else layer.set_cost(z), clean, None if best is None else layer.set_cost(best)))
            if (z is not None and clean) or H == H0:
                break
        return best

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
            if self._time_up():
                return None
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
                if cost[g] >= cR:  # a swap lowers the cost, so it stays within the budget
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
                if self.oracle.check(X2, clean=True)[0]:
                    return z2
        return None

    def _redive(self, z):
        """Large-neighbourhood moves before the swaps: drop a random fifth of the design (at
        least two interventions), forbid them, complete the rest with a short guided dive and keep
        the result if its minimal form is cheaper. `redives` trials in all, guided modules only."""
        layer = self.layer
        mods = self.oracle.modules
        if not any(m.kind == SUPPRESS for m in mods) or any(m.__class__ is not ClassicModule for m in mods):
            return z
        for _ in range(self.redives):
            if self._time_up() or not z:
                break
            members = list(bits(z))
            drop = self.rng.sample(members, min(len(members), max(2, len(members) // 5)))
            keep = z
            for i in drop:
                keep &= ~(1 << i)
            self.stat['redives'] += 1
            z2 = self.guided_dive(keep, rounds=3, forbid=z & ~keep)
            if z2 is None:
                continue
            z2 = self.minimise(z2)
            if layer.set_cost(z2) < layer.set_cost(z) and not (self.found and self._excluded(z2)):
                self.stat['redive_wins'] += 1
                z = z2
        return z

    def polish(self, z):
        """Bring a design to a local optimum: no single intervention can be removed, and no one or
        two interventions can be replaced by a single intervention of lower total cost, without
        losing the design (among the sets that contain no design returned earlier). The first
        neighbourhood is always searched completely, so the result is a minimal design. The swap
        neighbourhood is searched completely unless swap_trials LP checks or the time limit run
        out first; `stat['swap_capped']` counts the designs for which the budget ran out, and for
        those the swap part of the local optimum is not guaranteed."""
        self._swap_budget = self.stat['swap_checks'] + self.swap_trials
        z = self.minimise(z)
        z = self._redive(z)
        while not self._time_up():
            z2 = self._swap_round(z)
            if z2 is None:
                break
            self.stat['swaps'] += 1
            z = self.minimise(z2)
        return z

    def _next_dive(self):
        """A design to polish: the cheapest one in a pool that guided dives fill up to `probes`
        members (a guided dive costs a few dozen LPs, polishing far more), else one clause dive."""
        if self.guided:
            misses = 0
            while len(self.pool) < self.probes and misses < 2:
                z = self.guided_dive()
                if z is None or z in self.pool:
                    misses += 1
                    continue
                self.pool.append(z)
                self.seen |= z
            self.pool = [z for z in self.pool if not self._excluded(z)]
            if self.pool:
                z = min(self.pool, key=lambda z: (self.layer.set_cost(z), self.rng.random()))
                self.pool.remove(z)
                return z
        return self.dive()

    def run(self, max_solutions=1):
        """Returns (designs as bit sets, proven: True if a module was shown to be unfulfillable)."""
        t0 = time.time()
        fails = 0
        try:
            if getattr(self, 'trivial', False):
                return self.found, False
            if not self.found and self._is_design(0):
                self.found, self.trivial = [0], True
                return self.found, False
            while len(self.found) < max_solutions and fails < self.stall:
                z = self._next_dive()
                if z is None:
                    fails += 1
                    self.stat['failed_dives'] += 1
                    continue
                c0 = self.layer.set_cost(z)
                self.seen |= z
                z = self.polish(z)
                self.seen |= z
                if self.layer.set_cost(z) > self.max_cost:
                    self.stat['over_budget'] += 1
                    fails += 1
                    continue
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
        self.base = getattr(objective, 'start', None)
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
                if self.ds.oracle.check(X2, clean=True)[0]:
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
                   max_solutions=np.inf, seed=None, exact=False, first_levels=3, first_seconds=20.0, union_seconds=120.0, **kwargs):
    """Entry point for compute_strain_designs with solution_approach='any'.

    For SUPPRESS / PROTECT modules the complete search first walks the cost levels up to
    `first_levels` for at most `first_seconds` (and at most a third of the time limit); what it
    finds are the cheapest designs. Dives deliver the rest. After the dives the complete search
    runs once more, for at most `union_seconds`, over the interventions that occur in the designs
    the dives reached: it returns the cheaper designs that these interventions hold, or shows that
    they hold none (`stat['union_complete']`). With reaction knock-ins only the complete search
    runs, over the whole (finite) budget.

    Status: OPTIMAL if max_solutions designs were found, or every design within the budget (for an
    optimisation module: when the local search ended at a local optimum before the time limit),
    TIME_LIMIT_W_SOL if the search stopped earlier (time limit, or several dives in a row without
    a new design) with designs, TIME_LIMIT without, and INFEASIBLE only when it was shown that no
    set of interventions within the budget fulfils the modules.

    Returns:
        (CompressedDesigns, search object)
    """
    if gki_cost:
        raise NotImplementedError('the LP engine does not handle gene knock-ins yet')
    costs = list(ko_cost.values()) + list((gko_cost or {}).values()) + list((ki_cost or {}).values())
    if any(c <= 0 for c in costs):
        raise NotImplementedError('the LP engine needs positive intervention costs')
    opt_mods = [mod for mod in sd_modules if mod[MODULE_TYPE] in _OBJECTIVES]
    if any(mod[MODULE_TYPE] not in (SUPPRESS, PROTECT) and mod[MODULE_TYPE] not in _OBJECTIVES for mod in sd_modules) or len(opt_mods) > 1:
        raise NotImplementedError('the LP engine handles SUPPRESS, PROTECT, OptKnock, RobustKnock and OptCouple modules')
    integer_costs = all(abs(c - round(c)) <= 1e-9 for c in costs)
    if ki_cost and (opt_mods or not np.isfinite(max_cost) or not integer_costs):
        raise NotImplementedError("solution_approach 'any' of the LP engine handles knock-ins for SUPPRESS / PROTECT modules with a finite "
                                  "budget and integer costs")
    from straindesign.parse_constr import lineqlist2mat
    t0 = time.time()
    n_lp0 = _LPBase.n_solves
    names, layer, modules, atom_rxn, ctx = build_problem(cmp_model, sd_modules, ko_cost, gko_cost, max_cost, solver,
                                                         'name' if has_gene_names else 'id', exact=exact, ki_cost=ki_cost)
    objective = None
    if opt_mods:
        mod = opt_mods[0]
        V = lineqlist2mat(mod[CONSTRAINTS], ctx['rxn_ids'])
        objective = _OBJECTIVES[mod[MODULE_TYPE]](ctx, mod, _integer_rows(*V) if exact else V)
    logging.info('  LP dive: %d interventions over %d knockable reactions of %d (set up in %.1f s).' %
                 (layer.nI, layer.nA, len(cmp_model.reactions), time.time() - t0))
    if objective is None:
        kwargs.setdefault('targets', target_regions(ctx, sd_modules, modules))
        search = DiveSearch(layer, modules, max_cost, seed=seed, time_limit=time_limit, **kwargs)
        proven = exhaustive = False
        levels = max_cost if ki_cost else min(max_cost, first_levels)
        if levels >= 1 and integer_costs:
            # the cheapest designs, where the lowest cost levels hold any
            enum = LPEnumerator(layer, modules, levels, max_solutions=max_solutions,
                                time_limit=time_limit if ki_cost else min(first_seconds, time_limit / 3))
            cheap, complete = enum.run()
            exhaustive = complete and levels >= max_cost and len(cheap) < max_solutions
            proven = exhaustive and not cheap
            search.take_over(enum, cheap, time.time() - t0)
        if not (proven or exhaustive or ki_cost or len(search.found) >= max_solutions):
            n_cheap = len(search.found)
            found, proven = search.run(max_solutions)
            if search.found and not n_cheap and not getattr(search, 'trivial', False) and union_seconds > 0:
                cheaper, _ = search.cheaper_among_seen(union_seconds, max_solutions)
                if cheaper:
                    logging.info('  LP dive: %d cheaper designs among the %d interventions of the designs found, cheapest of cost %g' %
                                 (len(cheaper), search.seen.bit_count(), layer.set_cost(cheaper[0])))
                    for z in cheaper:
                        search.log.append((time.time() - t0, layer.set_cost(z), layer.set_cost(z), search.stat['queries']))
                    keep = len(search.found) if not np.isfinite(max_solutions) else int(max_solutions)
                    search.found = (cheaper + search.found)[:max(keep, 1)]
        found = search.found
        values = None
        done = len(found) >= max_solutions or exhaustive or getattr(search, 'trivial', False)
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
    # a knockout made: -1, a knock-in made: 1, a knock-in not made: 0
    sd = [{**{names[i]: 0.0 for i in bits(layer.ki & ~z)}, **{names[i]: (1.0 if (layer.ki >> i) & 1 else -1.0) for i in bits(z)}} for z in found]
    if proven:
        status = INFEASIBLE
    elif sd:
        status = OPTIMAL if done else TIME_LIMIT_W_SOL
    else:
        status = TIME_LIMIT
    setup = {MODEL_ID: cmp_model.id, MAX_COST: max_cost, SOLVER: solver, SOLUTION_APPROACH: ANY, KOCOST: dict(ko_cost), KICOST: dict(ki_cost or {}), MODULES: sd_modules,
             TIME_LIMIT: time_limit, MAX_SOLUTIONS: max_solutions}
    logging.info('  LP dive: %d designs, status %s; %d LPs; %.1f s.' % (len(sd), status, stat['lp_solves'], stat['seconds']))
    return CompressedDesigns(sd, status, setup), search
