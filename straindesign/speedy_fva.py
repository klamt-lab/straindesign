#!/usr/bin/env python3
#
# Copyright 2026 Max Planck Institute Magdeburg
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
"""Accelerated FVA using scan LPs and bound scanning.

Standard FVA solves 2*n independent LPs (max and min for each reaction).
This implementation reduces LP count via a two-phase approach:

Phase 1 — Scan LPs (cheap, resolve ~50-70% of bounds):
  a. trivial flux vector v=0 test: if feasible, all exisiting lb=0/ub=0 bounds
     are immediately confirmed as flux limits.
  b. min(sum(|x|)): pushes reactions toward zero, resolves lb=0/ub=0 bounds.
     All fluxes that take the value 0 also immediately confirm the bounds 
     as true flux limits.
  c. Push-to-bounds: objectives with random positive weights push all unresolved
     reactions toward their variable bounds at once, re-solved by the primal
     simplex from the previous vertex (which stays feasible when only the
     objective changes)
  Each scan LP solution is processed by bound scanning (vectorized at-bound
  check that marks reactions whose flux equals their variable bound).

Phase 2 — Individual LPs for remaining unresolved objectives:
  Sequential (warm-started) or parallel (SDPool) dispatch depending on
  problem size and thread count.  Sequential mode includes co-optimization
  scanning: each LP vertex is checked for other reactions at their bounds.
"""

import logging
import time as _time
import numpy as np
from scipy import sparse
from math import nan
from cobra.util.array import create_stoichiometric_matrix
from pandas import DataFrame

from cobra import Configuration

from straindesign.lptools import rev_worker_init, rev_worker_compute
from straindesign.lptools import (
    select_solver,
    idx2c,
    fva_worker_init,
    fva_worker_compute,
)
from straindesign.solver_interface import MILP_LP
from straindesign.pool import SDPool
from straindesign.parse_constr import parse_constraints, lineqlist2mat
from straindesign.names import CONSTRAINTS, SOLVER, OPTIMAL, UNBOUNDED, INFEASIBLE, GLPK, CPLEX, GUROBI, LP_METHOD_PRIMAL
from straindesign.networktools import suppress_lp_context
from straindesign.compression import (
    compress_cobra_model,
    CompressionMethod,
    remove_conservation_relations,
    stoichmat_coeff_to_fraction,
    stoichmat_coeff2float,
    remove_blocked_reactions,
    _detach_gprs,
)

# Tunable threshold at which parallel FVA kicks in. Is compared against the number of
# LPs that are left to be solved by Phase 2, and covers the pool's start-up: a worker pool
# reaches its first result in about 3 s on a genome-scale LP and then solves roughly 6x
# faster than the sequential loop, so it pays from a few hundred LPs upwards.
_PARALLEL_PHASE2_MIN = 500
# Above that count, the first LPs are solved sequentially as a probe (until _PROBE_LPS of them or
# _PROBE_SECONDS have passed); the rest goes to the pool only if the time the pool saves at the
# probe's rate (all but one worker's share of the sequential time) exceeds what the pool takes to
# start, measured at about 2 s plus 0.07 s per worker.
_PROBE_LPS = 50
_PROBE_SECONDS = 0.5
_POOL_START_SECONDS = 2.0
_POOL_START_SECONDS_PER_WORKER = 0.07
# Sequential Phase 2 replaces its LP by a freshly built one after this many solves. CPLEX is exempt: its
# results are the same without the rebuild, and each rebuild costs a cold solve.
_PHASE2_REBUILD_EVERY = 200
# Objectives are handed to the pool in this many chunks per worker. One chunk each leaves workers idle
# while the slowest chunk finishes; the LPs of a genome-scale model differ in cost by orders of magnitude.
_POOL_CHUNKS_PER_WORKER = 4
# The push LPs are solved in batches of _PUSH_BATCH until a batch resolves fewer than _PUSH_MIN_YIELD
# directions per LP. A push LP takes as long as two to ten single-objective LPs of the sequential
# Phase 2, which resolve about one direction each.
_PUSH_BATCH = 4
_PUSH_MIN_YIELD = 5
# Infinite bounds are replaced by +-_PUSH_BOX in the push LPs, which keeps their objectives bounded.
_PUSH_BOX = 1000.0

# ---------------------------------------------------------------------------
# Compression helpers
# ---------------------------------------------------------------------------


def _pin_structurally_blocked(model) -> int:
    """Fix reactions the structural sweep proves blocked to (0,0), in place.

    `remove_blocked_reactions` filters on bounds and detects nothing itself, so without this it
    has nothing to remove on a normally-bounded model.
    """
    af, ar = _rev_structural_sweep(model)
    n = 0
    for r in model.reactions:
        if not af[r.id] and not ar[r.id] and (r.lower_bound, r.upper_bound) != (0.0, 0.0):
            r.lower_bound = 0.0
            r.upper_bound = 0.0
            n += 1
    return n


def _compress_for_fva(model):
    """Copy and compress model for FVA (single-pass coupled compression + conservation removal).

    Suppresses optlang LP updates and skips solver deepcopy (not needed —
    speedy_fva builds its own MILP_LP objects).  Uses a single nullspace
    compression pass (no recursive iteration) since FVA only needs the
    first-order couplings.

    Returns (compressed_model, cmp_maps) where cmp_maps is a list of
    {compressed_id: {orig_id: Fraction_factor}} dicts, one per compression round
    that actually reduced the reaction count.
    """
    cmp_maps = []
    with suppress_lp_context(model):
        # Fast copy: the suppressed Model.copy skips the solver deepcopy (~0.3s vs ~3.3s on
        # iML1515) and attaches a backend-free carrier.  Safe because speedy_fva builds its own
        # MILP_LP and the compression pipeline is solver-independent.
        cmp_model = model.copy()
        _pin_structurally_blocked(cmp_model)
        remove_blocked_reactions(cmp_model)
        stoichmat_coeff_to_fraction(cmp_model)
        n_before = len(cmp_model.reactions)
        # Single-pass coupled compression (NULLSPACE only, no RECURSIVE iteration)
        _detach_gprs(cmp_model)
        result = compress_cobra_model(cmp_model, methods=[CompressionMethod.NULLSPACE], in_place=True)
        rmap = result.reaction_map
        if len(rmap) < n_before:
            cmp_maps.append(rmap)
        # Coupling can create new dependent rows — remove for clean LU factorization
        remove_conservation_relations(cmp_model)
    return cmp_model, cmp_maps


def _map_constraints(parsed_constraints, cmp_maps, cmp_reaction_ids):
    """Map parsed constraint dicts through compression mappings.

    Each constraint is [coeff_dict, operator, rhs].  For each compression step,
    original reaction IDs in coeff_dict are replaced by their compressed ID with
    the coefficient scaled by the coupling factor (v_orig = factor * v_compressed).
    """
    cmp_ids = set(cmp_reaction_ids)
    for rmap in cmp_maps:
        for cmp_id, orig_map in rmap.items():
            for constraint in parsed_constraints:
                coeff_dict = constraint[0]
                lumped = [k for k in coeff_dict if k in orig_map]
                if lumped:
                    coeff_dict[cmp_id] = sum(coeff_dict.pop(k) * float(orig_map[k]) for k in lumped)
    # Remove references to reactions not in compressed model (e.g., blocked)
    for constraint in parsed_constraints:
        for k in list(constraint[0].keys()):
            if k not in cmp_ids:
                del constraint[0][k]
    return parsed_constraints


def _expand_fva(fva_cmp, cmp_maps, orig_reaction_ids):
    """Expand compressed FVA results back to original reaction space.

    Reverses through compression steps, expanding lumped reactions using
    v_orig = factor * v_compressed (with min/max swap when factor < 0).
    """
    result_min = {}
    result_max = {}

    for rxn_id in fva_cmp.index:
        result_min[rxn_id] = fva_cmp.loc[rxn_id, 'minimum']
        result_max[rxn_id] = fva_cmp.loc[rxn_id, 'maximum']

    for rmap in reversed(cmp_maps):
        for cmp_id, orig_map in rmap.items():
            if len(orig_map) <= 1:
                continue  # singleton — ID unchanged
            cmp_min = result_min.pop(cmp_id, 0.0)
            cmp_max = result_max.pop(cmp_id, 0.0)
            for orig_id, factor in orig_map.items():
                f = float(factor)
                if f > 0:
                    result_min[orig_id] = f * cmp_min
                    result_max[orig_id] = f * cmp_max
                else:
                    result_min[orig_id] = f * cmp_max
                    result_max[orig_id] = f * cmp_min

    # Blocked reactions (removed by remove_blocked_reactions): min=max=0
    for rxn_id in orig_reaction_ids:
        if rxn_id not in result_min:
            result_min[rxn_id] = 0.0
            result_max[rxn_id] = 0.0

    return DataFrame(
        {
            "minimum": [result_min[r] for r in orig_reaction_ids],
            "maximum": [result_max[r] for r in orig_reaction_ids]
        },
        index=orig_reaction_ids,
    )


# ---------------------------------------------------------------------------
# LP helpers
# ---------------------------------------------------------------------------


def _single_thread(lp):
    """Restrict a warm-started LP to one solver thread and return it.

    The objective loops re-solve one LP thousands of times from the previous basis. CPLEX with
    its default thread count takes several times longer per such re-solve on genome-scale LPs
    (Recon3D: 27 ms against 11 ms, Human-GEM: 116 ms against 17 ms). With Gurobi the difference
    is small (Human-GEM: 111 s against 103 s for the whole analysis); the pool workers run both
    solvers on one thread as well.
    """
    if lp.solver == CPLEX:
        lp.backend.parameters.threads.set(1)
    elif lp.solver == GUROBI:
        lp.backend.params.Threads = 1
    return lp


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def speedy_fva(model, **kwargs):
    """Accelerated FVA using global scan LPs and KKT-based optimality propagation.

    Returns the same DataFrame as straindesign.lptools.fva.

    Parameters
    ----------
    model : cobra.Model
    solver : str, optional
    constraints : str or list, optional
    compress : bool or None, optional (default None)
        Compress the model (coupled compression + conservation removal) before
        running FVA, then expand results back.  None = auto (True if n >= 200).
    precheck : bool or None, optional (default None)
        Run the push LPs, which pre-resolve many objectives without individual
        LPs.  None = auto: on when Phase 2 runs its objectives sequentially, off
        when they go to the worker pool (whose LPs are cheaper per objective than
        a push LP).  True runs them before Phase 2 in either case.
    threads : int or None, optional (default None)
        Number of parallel workers for Phase 2 dispatch.  None = auto
        (Configuration().processes if n >= 1000, else 1).
    reaction_list : list of str, optional (default None)
        Only compute FVA for these reaction IDs.  Non-listed reactions get
        NaN in the returned DataFrame.  None = compute for all reactions.
    verbose : bool, optional (default False)

    Returns
    -------
    pandas.DataFrame with columns 'minimum' and 'maximum', indexed by reaction ID.
    """
    compress = kwargs.pop('compress', None)
    precheck = kwargs.pop('precheck', None)
    threads = kwargs.pop('threads', None)
    reaction_list = kwargs.pop('reaction_list', None)
    orig_reaction_ids = model.reactions.list_attr("id")
    n_original = len(orig_reaction_ids)
    cmp_maps = []

    # Auto-tuning
    if compress is None:
        compress = n_original >= 200
    if threads is None:
        # Worker cap only; whether Phase 2 actually goes parallel is decided
        # below from n_remaining (post-scan) against _PARALLEL_PHASE2_MIN.
        threads = Configuration().processes
    threads = max(1, int(threads))

    t_phase = {}
    _tp0 = _time.perf_counter()

    orig_model = model  # uncompressed model: carries the user-facing reaction and gene IDs
    if compress:
        model, cmp_maps = _compress_for_fva(model)
    t_phase['compress'] = _time.perf_counter() - _tp0

    reaction_ids = model.reactions.list_attr("id")
    n_orig = len(reaction_ids)

    has_constraints = CONSTRAINTS in kwargs and kwargs[CONSTRAINTS]
    if has_constraints:
        from straindesign.networktools import resolve_gene_constraints
        # Resolve gene/reaction constraints in the original ID space, then map onto the compressed model
        kwargs[CONSTRAINTS] = resolve_gene_constraints(orig_model, kwargs[CONSTRAINTS])
        kwargs[CONSTRAINTS] = parse_constraints(kwargs[CONSTRAINTS], orig_reaction_ids)
        if cmp_maps:
            kwargs[CONSTRAINTS] = _map_constraints(kwargs[CONSTRAINTS], cmp_maps, reaction_ids)
        A_ineq_extra, b_ineq_extra, A_eq_extra, b_eq_extra = lineqlist2mat(kwargs[CONSTRAINTS], reaction_ids)

    if SOLVER not in kwargs:
        kwargs[SOLVER] = None
    solver = select_solver(kwargs[SOLVER], model)
    verbose = kwargs.get('verbose', False)

    # ------------------------------------------------------------------
    # Phase 0: Setup
    # ------------------------------------------------------------------
    S = sparse.csr_matrix(create_stoichiometric_matrix(model))
    m_S = S.shape[0]
    b_eq_base = [0.0] * m_S

    if has_constraints:
        A_eq = sparse.vstack((S, A_eq_extra))
        b_eq = b_eq_base + b_eq_extra
        A_ineq = A_ineq_extra
        b_ineq = b_ineq_extra
    else:
        A_eq = S
        b_eq = b_eq_base
        A_ineq = sparse.csr_matrix((0, n_orig))
        b_ineq = []

    lb = np.array([v.lower_bound for v in model.reactions], dtype=np.float64)
    ub = np.array([v.upper_bound for v in model.reactions], dtype=np.float64)

    lp = MILP_LP(A_ineq=A_ineq, b_ineq=b_ineq, A_eq=A_eq, b_eq=b_eq, lb=lb.tolist(), ub=ub.tolist(), solver=solver)

    _, _, status = lp.solve()
    if status not in [OPTIMAL, UNBOUNDED]:
        logging.info('FVA problem not feasible.')
        return DataFrame(
            {
                "minimum": [nan] * n_orig,
                "maximum": [nan] * n_orig
            },
            index=reaction_ids,
        )

    # Initialize tracking
    incumbent_max = np.full(n_orig, -np.inf)
    incumbent_min = np.full(n_orig, np.inf)
    res_max = np.zeros(n_orig, dtype=bool)
    res_min = np.zeros(n_orig, dtype=bool)

    # Pre-resolve fixed reactions
    fixed = np.abs(ub - lb) < 1e-12
    res_max[fixed] = True
    res_min[fixed] = True
    incumbent_max[fixed] = ub[fixed]
    incumbent_min[fixed] = lb[fixed]

    # Scope FVA to reaction_list: mark non-listed reactions as resolved with NaN
    if reaction_list is not None:
        scope_set = set(reaction_list)
        for idx, rid in enumerate(reaction_ids):
            if rid not in scope_set:
                res_max[idx] = True
                res_min[idx] = True
                incumbent_max[idx] = nan
                incumbent_min[idx] = nan

    # Stats
    lps_solved = 0
    unresolved_bounds = []
    total_bound_resolved = 0
    t_solve = 0.0
    tol_bound = 1e-9

    def _bound_scan(x_local):
        nonlocal total_bound_resolved
        at_ub = (~res_max) & (np.abs(x_local - ub) < tol_bound)
        n_ub = int(at_ub.sum())
        if n_ub:
            res_max[at_ub] = True
            incumbent_max[at_ub] = ub[at_ub]
            total_bound_resolved += n_ub
        at_lb = (~res_min) & (np.abs(x_local - lb) < tol_bound)
        n_lb = int(at_lb.sum())
        if n_lb:
            res_min[at_lb] = True
            incumbent_min[at_lb] = lb[at_lb]
            total_bound_resolved += n_lb

    def _witness(x_local):
        """Update the incumbents from a feasible flux vector.

        Open directions follow every vector. A resolved bound is the variable bound or the optimum
        of its own LP, and the fluxes of another objective's vertex carry solver noise of about
        1e-9, so a resolved bound moves only for a flux that beats it by more than _WITNESS_TOL.
        """
        with np.errstate(invalid='ignore'):
            lift_max = ~res_max | (x_local > incumbent_max + _WITNESS_TOL * (1.0 + np.abs(incumbent_max)))
            lift_min = ~res_min | (x_local < incumbent_min - _WITNESS_TOL * (1.0 + np.abs(incumbent_min)))
        np.maximum(incumbent_max, x_local, out=incumbent_max, where=lift_max)
        np.minimum(incumbent_min, x_local, out=incumbent_min, where=lift_min)

    t_phase['setup'] = _time.perf_counter() - _tp0 - t_phase['compress']

    # ------------------------------------------------------------------
    # Phase 1: Global scan LPs + v=0 feasibility
    # ------------------------------------------------------------------
    _tp1 = _time.perf_counter()
    # 1a: v=0 feasibility check — free resolutions, no LP needed
    v0_feasible = (not np.any(lb > tol_bound) and not np.any(ub < -tol_bound) and not has_constraints)
    if v0_feasible:
        zero_lb = np.abs(lb) < tol_bound
        newly_min = zero_lb & (~res_min)
        res_min[newly_min] = True
        incumbent_min[newly_min] = 0.0
        total_bound_resolved += int(newly_min.sum())

        zero_ub = np.abs(ub) < tol_bound
        newly_max = zero_ub & (~res_max)
        res_max[newly_max] = True
        incumbent_max[newly_max] = 0.0
        total_bound_resolved += int(newly_max.sum())

    # Whether Phase 2 is expected to hand its objectives to the worker pool; its probe decides.
    n_unresolved = 2 * n_orig - int(res_max.sum() + res_min.sum())
    pool_candidate = threads > 1 and n_unresolved >= _PARALLEL_PHASE2_MIN and solver != GLPK

    def _push_phase():
        """Resolve directions by witnessing reactions at their variable bounds, without proving optimality.

        A push LP maximizes a randomly weighted sum over all unresolved directions. Its optimum is
        a vertex with many reactions at a variable bound, and each of them has that direction
        resolved. The weights change from LP to LP, so reactions that cannot sit at their bounds
        together are witnessed by different LPs.
        """
        nonlocal t_solve, lps_solved
        # Only the objective changes between solves: the previous vertex stays feasible and the
        # primal simplex continues from it.
        scan_lp = _single_thread(
            MILP_LP(A_ineq=A_ineq,
                    b_ineq=b_ineq,
                    A_eq=A_eq,
                    b_eq=b_eq,
                    lb=np.where(np.isinf(lb), -_PUSH_BOX, lb).tolist(),
                    ub=np.where(np.isinf(ub), _PUSH_BOX, ub).tolist(),
                    solver=solver))
        scan_lp.set_lp_method(LP_METHOD_PRIMAL)
        rng = np.random.default_rng(0)
        push_iter = 0
        while True:
            before = int(res_max.sum() + res_min.sum())
            for _ in range(_PUSH_BATCH):
                open_max = ~res_max
                open_min = ~res_min
                if not (open_max.any() or open_min.any()):
                    break
                c_push = np.zeros(n_orig)
                c_push[open_max] -= rng.exponential(1.0, int(open_max.sum()))  # push toward ub
                c_push[open_min] += rng.exponential(1.0, int(open_min.sum()))  # push toward lb
                scan_lp.set_objective(c_push.tolist())
                t0 = _time.perf_counter()
                x_list_scan, _, scan_status = scan_lp.solve()
                t_solve += _time.perf_counter() - t0
                lps_solved += 1
                if scan_status == OPTIMAL:
                    x_scan = np.array(x_list_scan[:n_orig], dtype=np.float64)
                    _bound_scan(x_scan)
                    _witness(x_scan)
            n_done_iter = int(res_max.sum() + res_min.sum())
            if verbose:
                logging.debug(f"  push batch {push_iter}: +{n_done_iter - before} "
                              f"({n_done_iter}/{2*n_orig} resolved)")
            push_iter += 1
            if n_done_iter - before < _PUSH_MIN_YIELD * _PUSH_BATCH or n_done_iter == 2 * n_orig:
                break

    # A pooled Phase-2 LP costs a worker's share of a sequential one, less than a push LP costs per
    # direction it resolves. With precheck on auto, a pool candidate therefore pushes only once the
    # probe of Phase 2 has decided against the pool.
    push_deferred = precheck is None and pool_candidate
    if precheck or (precheck is None and not pool_candidate):
        _push_phase()

    t_phase['phase1'] = _time.perf_counter() - _tp1

    # ------------------------------------------------------------------
    # Phase 2: Dispatch remaining objectives — sequential or parallel
    # ------------------------------------------------------------------
    _tp2 = _time.perf_counter()
    n_done = int(res_max.sum() + res_min.sum())
    n_remaining = 2 * n_orig - n_done
    phase2_entry_count = n_remaining  # for stats

    # The pool's start-up is a fixed cost of seconds, while an LP of a small network solves in well
    # under a millisecond, so the objective count alone cannot tell which dispatch is faster. The
    # sequential loop runs first as a probe and hands the remaining objectives to the pool only when
    # the pool would save more of their projected sequential time than it takes to start.
    handoff = False
    # GLPK stays sequential: its pool worker rebuilds the LP for every objective, which took
    # minutes on iMLcore where the warm-started loop takes two seconds.
    probe = n_remaining >= _PARALLEL_PHASE2_MIN and threads > 1 and solver != GLPK
    if n_remaining > 0:
        # Sequential dispatch: one warm-started LP, objectives in the order _next_objective picks them
        prev_col = -1

        def _rebuild_lp():
            nonlocal lp, prev_col
            lp = _single_thread(MILP_LP(A_ineq=A_ineq, b_ineq=b_ineq, A_eq=A_eq, b_eq=b_eq, lb=lb.tolist(), ub=ub.tolist(), solver=solver))
            prev_col = -1

        _rebuild_lp()
        seq_count = 0
        t_seq0 = _time.perf_counter()

        x_last = None  # flux vector of the latest optimal solve: the vertex the next solve starts from

        def _next_objective():
            """Pick the next unresolved (reaction index, direction), or None when all are resolved.

            The warm-started simplex needs the fewest pivots for a direction whose best known value
            is already attained at the current vertex, so those go first. Otherwise the direction
            whose current flux is closest to its best known value is taken.
            """
            open_max = ~res_max
            open_min = ~res_min
            if x_last is None:
                cand = np.flatnonzero(open_max | open_min)
                if not len(cand):
                    return None
                return int(cand[0]), (1 if open_max[cand[0]] else -1)
            with np.errstate(invalid='ignore'):  # resolved directions may hold inf or nan
                gap_max = np.where(open_max, (incumbent_max - x_last) / (1.0 + np.abs(incumbent_max)), np.inf)
                gap_min = np.where(open_min, (x_last - incumbent_min) / (1.0 + np.abs(incumbent_min)), np.inf)
            j_max = int(np.argmin(gap_max))
            j_min = int(np.argmin(gap_min))
            if not open_max[j_max] and not open_min[j_min]:
                return None
            if open_max[j_max] and (not open_min[j_min] or gap_max[j_max] <= max(gap_min[j_min], 1e-12)):
                return j_max, 1
            return j_min, -1

        while not handoff:
            nxt = _next_objective()
            if nxt is None:
                break
            j, direction = nxt
            if probe:
                elapsed = _time.perf_counter() - t_seq0
                if seq_count >= _PROBE_LPS or elapsed >= _PROBE_SECONDS:
                    probe = False
                    n_left = 2 * n_orig - int(res_max.sum() + res_min.sum())
                    seconds_left = elapsed / seq_count * n_left if seq_count > 0 else 0.0
                    handoff = seconds_left * (1.0 - 1.0 / threads) > _POOL_START_SECONDS + _POOL_START_SECONDS_PER_WORKER * threads
                    if handoff:
                        break
                    if push_deferred:
                        push_deferred = False
                        _push_phase()
                        if (res_max if direction == 1 else res_min)[j]:
                            continue

            # Periodic rebuild to limit warm-start degeneration
            if solver != CPLEX and seq_count > 0 and seq_count % _PHASE2_REBUILD_EVERY == 0:
                _rebuild_lp()

            sig = -direction
            for attempt in range(_MAX_STATUS_RETRIES + 1):
                if prev_col < 0 or prev_col == j:
                    C = [[j, float(sig)]]
                else:
                    C = [[j, float(sig)], [prev_col, 0.0]]
                if solver in ('cplex', 'gurobi'):
                    lp.backend.set_objective_idx(C)
                else:
                    lp.set_objective_idx(C)
                prev_col = j

                t0 = _time.perf_counter()
                x_list, obj_val, status = lp.solve()
                t_solve += _time.perf_counter() - t0
                lps_solved += 1
                seq_count += 1

                # Feasibility was established once at setup and only the objective changes
                # here, so a non-optimal status is a solver artifact, not an answer.
                if status in (OPTIMAL, UNBOUNDED) or attempt == _MAX_STATUS_RETRIES:
                    break
                _rebuild_lp()

            if status == UNBOUNDED:
                if direction == 1:
                    res_max[j] = True
                    incumbent_max[j] = np.inf
                else:
                    res_min[j] = True
                    incumbent_min[j] = -np.inf
                continue
            elif status != OPTIMAL:
                unresolved_bounds.append((reaction_ids[j], direction, status))
                if direction == 1:
                    res_max[j] = True
                    incumbent_max[j] = nan
                else:
                    res_min[j] = True
                    incumbent_min[j] = nan
                continue

            # Guard: LP optimum must not be worse than incumbent
            if direction == 1:
                val, inc = -obj_val, incumbent_max[j]
                bad = np.isfinite(inc) and val < inc - _DEGEN_TOL * (1 + abs(inc))
            else:
                val, inc = obj_val, incumbent_min[j]
                bad = np.isfinite(inc) and val > inc + _DEGEN_TOL * (1 + abs(inc))
            if bad:
                _rebuild_lp()
                C = [[j, float(sig)]]
                if solver in ('cplex', 'gurobi'):
                    lp.backend.set_objective_idx(C)
                else:
                    lp.set_objective_idx(C)
                prev_col = j
                t0 = _time.perf_counter()
                x_list, obj_val, status = lp.solve()
                t_solve += _time.perf_counter() - t0
                lps_solved += 1
                seq_count += 1

            if status == UNBOUNDED:
                if direction == 1:
                    res_max[j] = True
                    incumbent_max[j] = np.inf
                else:
                    res_min[j] = True
                    incumbent_min[j] = -np.inf
            elif status == OPTIMAL:
                if direction == 1:
                    res_max[j] = True
                    incumbent_max[j] = -obj_val
                else:
                    res_min[j] = True
                    incumbent_min[j] = obj_val
                # Co-optimization scan: check if this vertex also
                # resolves other unresolved directions (at-bound check
                # on non-zero values that improve the incumbent).
                x_arr = np.array(x_list[:n_orig], dtype=np.float64)
                _witness(x_arr)
                _bound_scan(x_arr)
                x_last = x_arr
            else:
                unresolved_bounds.append((reaction_ids[j], direction, status))
                if direction == 1:
                    res_max[j] = True
                    incumbent_max[j] = nan
                else:
                    res_min[j] = True
                    incumbent_min[j] = nan

    if handoff:
        # Parallel dispatch via SDPool
        # Build list of unresolved objective indices (even=max, odd=min)
        unresolved = []
        for j in range(n_orig):
            if not res_max[j]:
                unresolved.append(2 * j)  # even = max
            if not res_min[j]:
                unresolved.append(2 * j + 1)  # odd = min

        x_par = [nan] * (2 * n_orig)
        t0 = _time.perf_counter()
        with SDPool(threads, initializer=fva_worker_init, initargs=(A_ineq, b_ineq, A_eq, b_eq, lb.tolist(), ub.tolist(), solver)) as pool:
            chunk_size = max(1, len(unresolved) // (threads * _POOL_CHUNKS_PER_WORKER))
            for i, value in pool.imap_unordered(fva_worker_compute, unresolved, chunksize=chunk_size):
                x_par[i] = value
        t_solve += _time.perf_counter() - t0
        lps_solved += len(unresolved)

        # NaN retry with fresh LPs
        nan_idx = [i for i in unresolved if np.isnan(x_par[i])]
        if nan_idx:
            _BATCH = 50
            while nan_idx:
                lp_retry = MILP_LP(A_ineq=A_ineq, b_ineq=b_ineq, A_eq=A_eq, b_eq=b_eq, lb=lb.tolist(), ub=ub.tolist(), solver=solver)
                prev_retry = 0
                for i in nan_idx[:_BATCH]:
                    C = idx2c(i, prev_retry)
                    if solver in ('cplex', 'gurobi'):
                        lp_retry.backend.set_objective_idx(C)
                        x_par[i] = lp_retry.backend.slim_solve()
                    else:
                        lp_retry.set_objective_idx(C)
                        x_par[i] = lp_retry.slim_solve()
                    prev_retry = C[0][0]
                old_count = len(nan_idx)
                nan_idx = [i for i in nan_idx if np.isnan(x_par[i])]
                if len(nan_idx) == old_count:
                    break
            lps_solved += len(unresolved) - len(nan_idx) if nan_idx else len(unresolved)

        # Collect parallel results
        for j in range(n_orig):
            i_max = 2 * j
            if not res_max[j]:
                if np.isnan(x_par[i_max]):
                    incumbent_max[j] = nan
                    unresolved_bounds.append((reaction_ids[j], 1, 'nan'))
                else:
                    res_max[j] = True
                    incumbent_max[j] = -x_par[i_max]
            i_min = 2 * j + 1
            if not res_min[j]:
                if np.isnan(x_par[i_min]):
                    incumbent_min[j] = nan
                    unresolved_bounds.append((reaction_ids[j], -1, 'nan'))
                else:
                    res_min[j] = True
                    incumbent_min[j] = x_par[i_min]

    if unresolved_bounds:
        logging.warning(f"speedy_fva: {len(unresolved_bounds)} of {2*n_orig} bounds left unresolved after "
                        f"{_MAX_STATUS_RETRIES} retries (solver '{solver}' reported "
                        f"{sorted({st for _, _, st in unresolved_bounds})}); reported as NaN. "
                        f"First: {unresolved_bounds[0][0]}")

    # ------------------------------------------------------------------
    # Assemble results
    # ------------------------------------------------------------------
    incumbent_max[np.abs(incumbent_max) < 1e-11] = 0.0
    incumbent_min[np.abs(incumbent_min) < 1e-11] = 0.0

    fva_result = DataFrame(
        {
            "minimum": incumbent_min,
            "maximum": incumbent_max
        },
        index=reaction_ids,
    )

    if verbose:
        cmp_msg = ""
        if cmp_maps:
            cmp_msg = f" (compressed {n_original}→{n_orig} rxns)"
        logging.debug(f"  speedy_fva done{cmp_msg}: {lps_solved} LPs, "
                      f"{total_bound_resolved} bound-resolved, "
                      f"{2*n_orig} total objectives")
        logging.debug(f"  timing: solve={t_solve:.2f}s, threads={threads}")

    t_phase['phase2'] = _time.perf_counter() - _tp2
    t_phase['total'] = _time.perf_counter() - _tp0

    fva_result.attrs['t_phase'] = t_phase
    fva_result.attrs['lps_solved'] = lps_solved
    fva_result.attrs['bound_resolved'] = total_bound_resolved
    fva_result.attrs['phase2_remaining'] = phase2_entry_count
    fva_result.attrs['pooled'] = handoff
    if cmp_maps:
        fva_result.attrs['n_original'] = n_original
        fva_result.attrs['n_compressed'] = n_orig

    # Expand compressed results back to original reaction space
    if cmp_maps:
        expanded = _expand_fva(fva_result, cmp_maps, orig_reaction_ids)
        expanded.attrs = fva_result.attrs
        fva_result = expanded

    return fva_result


# ---------------------------------------------------------------------------
# Fast exact reversibility (sign-only FVA) for pre-compression tightening
# ---------------------------------------------------------------------------

_REV_SCAN_TOL = 1e-3  # co-option certifies only on flux comfortably above solver noise
_REV_REBUILD_EVERY = 200
_REV_PARALLEL_MIN = 1000  # reactions in the LP from which the scan is dispatched to a worker pool
_REV_BATCH_PER_WORKER = 32  # objectives handed out per worker and round; a round's witnesses prune the next
_REV_CHUNK = 4
_ZERO_SNAP = 1e-11  # |flux| below this is solver noise, not a direction; 0 disables snapping
_MAX_STATUS_RETRIES = 2  # fresh-LP retries before a bound is reported unresolved
_DEGEN_TOL = 1e-6  # warm-start guard: fresh optimum must not fall below a known-achievable incumbent
_WITNESS_TOL = 1e-8  # a flux vector lifts an already resolved bound only beyond this (relative to 1 + |bound|)


def _rev_structural_sweep(model):
    """Sound over-approximation of achievable directions via single-producer/consumer +
    dead-end propagation (0 LP, sign-only). Returns (af, ar): af[id]=fwd not proven blocked,
    ar[id]=rev not proven blocked. Every direction it kills is infeasible in ANY steady state,
    hence also in the flux polytope, so tightening on it is lossless."""
    af = {r.id: r.upper_bound > 0 for r in model.reactions}
    ar = {r.id: r.lower_bound < 0 for r in model.reactions}
    met_rx = {mm.id: [(r.id, r.metabolites[mm]) for r in mm.reactions] for mm in model.metabolites}
    changed = True
    while changed:
        changed = False
        for e in met_rx.values():
            prod = [(i, 'f') for i, c in e if c > 0 and af[i]] + [(i, 'r') for i, c in e if c < 0 and ar[i]]
            cons = [(i, 'f') for i, c in e if c < 0 and af[i]] + [(i, 'r') for i, c in e if c > 0 and ar[i]]
            prx = {i for i, _ in prod}
            crx = {i for i, _ in cons}

            def kill(i, d):
                nonlocal changed
                if d == 'f' and af[i]:
                    af[i] = False
                    changed = True
                if d == 'r' and ar[i]:
                    ar[i] = False
                    changed = True

            if not prod:
                for i, d in cons:
                    kill(i, d)
            elif len(prx) == 1:
                s = next(iter(prx))
                for i, d in cons:
                    if i == s: kill(i, d)
            if not cons:
                for i, d in prod:
                    kill(i, d)
            elif len(crx) == 1:
                s = next(iter(crx))
                for i, d in prod:
                    if i == s: kill(i, d)
    return af, ar


def fast_reversibility(model, solver=None, compress=True, threads=None):
    """Exact per-reaction reversibility on the ORIGINAL flux polytope, faster than full FVA.

    Returns {reaction_id: (can_fwd, can_rev)} for every reaction of ``model`` -- identical
    directionality to FVA (a blocked reaction is (False, False)), used to fix lb/ub to 0
    BEFORE compression so genuinely one-directional reactions fuse (and avoid the GPR
    fwd/rev split, which fires on lb<0).

    Method: (1) structural single-producer/consumer sweep tightens a copy's bounds (0 LP,
    sound); (2) a single coupled compression of the tightened copy shrinks the LP (default
    on -- the uncompressed matrix is per-LP pathological on some genome-scale models, e.g.
    yeast-GEM: ~68 vs ~4 ms/LP compressed); (3) warm-started per-reaction max/min on the
    compressed model (objective-only change) with a co-option scan that certifies other
    reactions carrying flux; (4) map compressed min/max back to the original reactions.
    Sign of the achieved min/max gives reversibility.

    The scan starts sequentially. When the remaining directions project to more sequential time
    than a worker pool takes to start, ``threads`` workers (default ``Configuration().processes``
    once the LP has at least ``_REV_PARALLEL_MIN`` reactions, else 1) take over, each holding one
    warm-started LP; the objectives are dispatched in rounds and every round's flux vectors certify
    directions before the next round is formed, so the scan stays the same as the sequential one up
    to which witness certifies a direction first."""
    solver = select_solver(solver, model)
    orig_rid = [r.id for r in model.reactions]

    # (1) structural sweep on the ORIGINAL model + tighten a copy's bounds (lossless)
    af, ar = _rev_structural_sweep(model)
    m = model.copy()
    for r in m.reactions:
        if not af[r.id]:
            r.upper_bound = min(float(r.upper_bound), 0.0)
        if not ar[r.id]:
            r.lower_bound = max(float(r.lower_bound), 0.0)

    # (2) single coupled compress of the tightened model (default on)
    if compress:
        m, cmp_maps = _compress_for_fva(m)
        stoichmat_coeff2float(m)
    else:
        cmp_maps = []

    # (3) LP phase on the (compressed) model
    cmp_rid = [r.id for r in m.reactions]
    n = len(cmp_rid)
    lb = np.array([float(r.lower_bound) for r in m.reactions])
    ub = np.array([float(r.upper_bound) for r in m.reactions])
    S = sparse.csr_matrix(create_stoichiometric_matrix(m))
    A_ineq = sparse.csr_matrix((0, n))
    b_ineq = []
    A_eq = S
    b_eq = [0.0] * S.shape[0]

    def build():
        return _single_thread(MILP_LP(A_ineq=A_ineq, b_ineq=b_ineq, A_eq=A_eq, b_eq=b_eq, lb=lb.tolist(), ub=ub.tolist(), solver=solver))

    lp = build()

    incumbent_max = np.full(n, -np.inf)
    incumbent_min = np.full(n, np.inf)
    # Bound signs are exact model data, not solver output, so they decide on the exact sign: a
    # direction counts as blocked only if the bound itself forbids it. A tolerance here would
    # discard a direction whose achievable flux is merely small (a max of 1e-8 is still forward).
    res_max = ub <= 0.0  # fwd already blocked by bounds (sweep/original)
    res_min = lb >= 0.0
    incumbent_max[res_max] = ub[res_max]
    incumbent_min[res_min] = lb[res_min]
    fixed = ub == lb
    res_max[fixed] = True
    res_min[fixed] = True
    incumbent_max[fixed] = ub[fixed]
    incumbent_min[fixed] = lb[fixed]

    def scan(x):
        nm = (~res_max) & (x > _REV_SCAN_TOL)
        res_max[nm] = True
        np.maximum(incumbent_max, x, out=incumbent_max)
        nn = (~res_min) & (x < -_REV_SCAN_TOL)
        res_min[nn] = True
        np.minimum(incumbent_min, x, out=incumbent_min)

    n_lp = 0
    prev_col = -1
    seq = 0

    def solve_dir(j, direction):
        nonlocal prev_col, seq, n_lp
        sig = -float(direction)
        C = [[j, sig]] if (prev_col < 0 or prev_col == j) else [[j, sig], [prev_col, 0.0]]
        if solver in ('cplex', 'gurobi'):
            lp.backend.set_objective_idx(C)
        else:
            lp.set_objective_idx(C)
        prev_col = j
        r = lp.solve()
        n_lp += 1
        seq += 1
        return r

    # Feasibility preflight: one zero-objective solve proves the polytope is non-empty (so a later
    # infeasible status can only come from the objective change, not the model) and its flux vector
    # seeds every incumbent, so no subsequent warm-start optimum can silently contradict a flux
    # already witnessed as achievable.
    x_feas, _, status_feas = lp.solve()
    n_lp += 1
    if status_feas == INFEASIBLE:
        raise ValueError('fast_reversibility: the model has no steady-state flux distribution.')
    if status_feas == OPTIMAL and x_feas:
        scan(np.array(x_feas[:n], dtype=np.float64))

    def unknown(j, direction):
        """Record 'direction not determined': the incumbent goes to +/-inf so the direction is
        reported as achievable. Tightening on an unproven bound could delete a feasible pathway;
        reporting a spurious direction only forgoes tightening."""
        if direction == 1:
            res_max[j] = True
            incumbent_max[j] = np.inf
        else:
            res_min[j] = True
            incumbent_min[j] = -np.inf

    def resolved(j, direction):
        return res_max[j] if direction == 1 else res_min[j]

    def accept(j, direction, x_arr, val):
        if direction == 1:
            res_max[j] = True
            incumbent_max[j] = max(incumbent_max[j], val)
        else:
            res_min[j] = True
            incumbent_min[j] = min(incumbent_min[j], val)
        scan(x_arr)

    def degenerate(j, direction, val):
        inc = incumbent_max[j] if direction == 1 else incumbent_min[j]
        return (direction == 1 and np.isfinite(inc) and val < inc - _DEGEN_TOL * (1 + abs(inc))) or \
               (direction == -1 and np.isfinite(inc) and val > inc + _DEGEN_TOL * (1 + abs(inc)))

    def fresh_resolve(j, direction):
        """Re-solve one direction on a freshly built LP (no warm start)."""
        nonlocal lp, prev_col
        lp = build()
        prev_col = -1
        x_list, obj_val, status = solve_dir(j, direction)
        if status != OPTIMAL:
            unknown(j, direction)
            return
        accept(j, direction, np.array(x_list[:n], dtype=np.float64), -obj_val if direction == 1 else obj_val)

    if threads is None:
        threads = Configuration().processes if n >= _REV_PARALLEL_MIN else 1
    threads = max(1, int(threads))

    def pending_directions():
        for j in range(n):
            for direction in (1, -1):
                if not resolved(j, direction):
                    yield j, direction

    # As in speedy_fva's phase 2: the pool's start-up is a fixed cost of seconds, so the sequential
    # loop runs first as a probe and hands the remaining directions to the pool only when their
    # projected sequential time exceeds that start-up.
    probe = threads > 1 and solver in ('cplex', 'gurobi')
    handoff = False
    n_seq, t_seq0 = 0, _time.perf_counter()
    for j, direction in pending_directions():
        if probe:
            elapsed = _time.perf_counter() - t_seq0
            if n_seq >= _PROBE_LPS or elapsed >= _PROBE_SECONDS:
                probe = False
                n_left = sum(1 for _ in pending_directions())
                if n_seq > 0 and elapsed / n_seq * n_left > \
                        _POOL_START_SECONDS + _POOL_START_SECONDS_PER_WORKER * threads:
                    handoff = True
                    break
        if seq > 0 and seq % _REV_REBUILD_EVERY == 0:
            lp = build()
            prev_col = -1
        x_list, obj_val, status = solve_dir(j, direction)
        n_seq += 1
        if status != OPTIMAL:
            # UNBOUNDED is a proven infinite direction, every other nonoptimal status (time
            # limit, numerical trouble) is simply unknown; both must not tighten.
            unknown(j, direction)
            continue
        val = -obj_val if direction == 1 else obj_val
        if degenerate(j, direction, val):
            fresh_resolve(j, direction)
            continue
        accept(j, direction, np.array(x_list[:n], dtype=np.float64), val)

    if handoff:
        with SDPool(threads,
                    initializer=rev_worker_init,
                    initargs=(A_ineq, b_ineq, A_eq, b_eq, lb.tolist(), ub.tolist(), solver, _REV_REBUILD_EVERY)) as pool:
            while True:
                pending = [2 * j + (0 if direction == 1 else 1) for j, direction in pending_directions()]
                if not pending:
                    break
                batch = pending[:threads * _REV_BATCH_PER_WORKER]
                for i, obj_val, status, x_bytes in pool.imap_unordered(rev_worker_compute, batch, chunksize=_REV_CHUNK):
                    n_lp += 1
                    j, direction = i // 2, (1 if i % 2 == 0 else -1)
                    if status != OPTIMAL:
                        # UNBOUNDED is a proven infinite direction, every other nonoptimal status
                        # (time limit, numerical trouble) is simply unknown; both must not tighten.
                        if not resolved(j, direction):
                            unknown(j, direction)
                        continue
                    val = -obj_val if direction == 1 else obj_val
                    if degenerate(j, direction, val):
                        fresh_resolve(j, direction)
                        continue
                    # A witness for a direction certified meanwhile is still an exact feasible
                    # point: its value and flux vector are used like any other.
                    accept(j, direction, np.frombuffer(x_bytes, dtype=np.float64)[:n], val)
    logging.debug('fast_reversibility: %d LPs on %d reactions, %d workers.', n_lp, n, threads if handoff else 1)

    # (4) expand compressed min/max back to original reactions
    # Snapping only removes flux the solver cannot distinguish from zero; the direction decision
    # itself is then the exact sign, so a small-but-real flux (1e-8) keeps its direction.
    incumbent_max[np.abs(incumbent_max) < _ZERO_SNAP] = 0.0
    incumbent_min[np.abs(incumbent_min) < _ZERO_SNAP] = 0.0
    df = DataFrame({"minimum": incumbent_min, "maximum": incumbent_max}, index=cmp_rid)
    if cmp_maps:
        df = _expand_fva(df, cmp_maps, orig_rid)
    return {r: (float(df.at[r, 'maximum']) > 0.0, float(df.at[r, 'minimum']) < 0.0) for r in orig_rid}
