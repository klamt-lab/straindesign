"""Accelerated FVA using scan LPs, bound scanning, and KKT certification.

Standard FVA solves 2*n independent LPs (max and min for each reaction).
This implementation reduces LP count via a two-phase approach:

Phase 1 — Scan LPs (cheap, resolve ~50% of bounds):
  a. v=0 feasibility: free resolutions when zero flux is feasible
  b. min(sum(|x|)): pushes reactions toward zero, resolves lb=0/ub=0 bounds
  c. Push-to-bounds: directed objectives push unresolved reactions toward
     their variable bounds, with dual simplex warm-start for fast re-solves
  d. Inf-bound scan: individual LPs for reactions with infinite bounds
  Each scan LP solution is processed by bound scanning (vectorized at-bound
  check) and KKT dual certification (sparse LU + null-space augmentation).

Phase 2 — Individual LPs for remaining unresolved objectives:
  Sequential (warm-started) or parallel (SDPool) dispatch depending on
  problem size and thread count.
"""

import logging
import time as _time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.linalg import null_space as _null_space
from scipy.optimize import linprog as _linprog
from math import nan
from cobra.util.array import create_stoichiometric_matrix
from pandas import DataFrame

from cobra import Configuration

from straindesign.lptools import (
    select_solver, idx2c, fva_worker_init, fva_worker_compute,
    fva_worker_init_glpk, fva_worker_compute_glpk,
)
from straindesign.solver_interface import MILP_LP
from straindesign.pool import SDPool
from straindesign.parse_constr import parse_constraints, lineqlist2mat
from straindesign.names import CONSTRAINTS, SOLVER, OPTIMAL, UNBOUNDED, GLPK, LP_METHOD_DUAL
from straindesign.networktools import suppress_lp_context
from straindesign.compression import (
    compress_cobra_model, CompressionMethod, remove_conservation_relations,
    stoichmat_coeff2rational, remove_blocked_reactions,
)


# ---------------------------------------------------------------------------
# DualChecker — null-space KKT optimality certification
# ---------------------------------------------------------------------------

class DualChecker:
    """Certify optimality via KKT for unresolved interior reactions.

    Given LP solution v* for  min c^T v  s.t. Sv = 0, lb <= v <= ub:

    Two-stage approach:
    1. Fast sparse LU check (always): M = S_I S_I^T, vectorized sign check
    2. Null-space augmentation (when d = m - |I| is small): for reactions that
       failed stage 1, search ALL dual solutions via tiny feasibility LP
       with incumbent and unfixable pre-filters
    """

    _D_MAX = 30  # max null-space dimension for stage 2

    def __init__(self, S_sparse, x, lb, ub, res_min, res_max,
                 incumbent_min, incumbent_max, tol=1e-7):
        x = np.asarray(x, dtype=np.float64)
        lb = np.asarray(lb, dtype=np.float64)
        ub = np.asarray(ub, dtype=np.float64)
        n = S_sparse.shape[1]
        m = S_sparse.shape[0]

        self.confirmed_min = np.zeros(n, dtype=bool)
        self.confirmed_max = np.zeros(n, dtype=bool)
        self.n_checks = 0
        self.n_ns_lps = 0

        # Partition I (interior), L (at lower), U (at upper)
        I_set = np.where((x - lb > tol) & (ub - x > tol))[0]
        L_set = np.where(x - lb <= tol)[0]
        U_set = np.where(ub - x <= tol)[0]
        nI, nL, nU = len(I_set), len(L_set), len(U_set)
        if nI == 0:
            return

        # Unresolved candidates
        unresolved = ~res_min[I_set] | ~res_max[I_set]
        I_check = I_set[unresolved]
        nC = len(I_check)
        if nC == 0:
            return

        S_csc = S_sparse.tocsc()

        # Stage 1: fast sparse LU check (always)
        self.n_checks = 2 * nC
        self._fast_sparse(S_csc, x, lb, ub, I_set, L_set, U_set,
                          I_check, res_min, res_max, m, nL, nU, tol)

        # Stage 2: null-space augmentation (only when d is small)
        d_est = m - nI
        if d_est > self._D_MAX:
            return

        # Check if any reactions still need resolving after stage 1
        still_min = (~res_min[I_set]) & (~self.confirmed_min[I_set])
        still_max = (~res_max[I_set]) & (~self.confirmed_max[I_set])
        # Incumbent filter: only check if v* improves best known
        still_min = still_min & (x[I_set] < incumbent_min[I_set] - tol)
        still_max = still_max & (x[I_set] > incumbent_max[I_set] + tol)
        n_remaining = int(still_min.sum() + still_max.sum())
        if n_remaining == 0:
            return

        self._nullspace_augment(
            S_csc, x, lb, ub, I_set, L_set, U_set,
            m, nI, nL, nU, still_min, still_max, tol)

    def _fast_sparse(self, S_csc, x, lb, ub, I_set, L_set, U_set,
                     I_check, res_min, res_max, m, nL, nU, tol):
        """Stage 1: sparse LU of M = S_I S_I^T, vectorized sign check."""
        S_I = S_csc[:, I_set]
        M = (S_I @ S_I.T).tocsc()
        M = M + sparse.eye(m, format='csc') * 1e-14
        try:
            M_lu = splu(M)
        except Exception:
            return

        S_check = S_csc[:, I_check].toarray()
        Lambda = M_lu.solve(S_check)  # m x nC
        nC = len(I_check)

        # Verify LU solution: diagonal of S_check^T @ Lambda should be ≈ 1.0.
        # When M is ill-conditioned, Lambda collapses toward zero and all sign
        # checks pass vacuously.  The diagonal check catches this reliably.
        diag = np.sum(S_check * Lambda, axis=0)  # shape (nC,)
        bad_col = np.abs(diag - 1.0) > 0.2
        if np.any(bad_col):
            good = ~bad_col
            I_check = I_check[good]
            Lambda = Lambda[:, good]
            nC = len(I_check)
            if nC == 0:
                return

        if nL > 0:
            D_L = S_csc[:, L_set].T @ Lambda
            lb_L = lb[L_set]
        if nU > 0:
            D_U = S_csc[:, U_set].T @ Lambda
            ub_U = ub[U_set]

        # Min direction
        sL = np.all(D_L <= tol, axis=0) if nL > 0 else np.ones(nC, dtype=bool)
        sU = np.all(D_U >= -tol, axis=0) if nU > 0 else np.ones(nC, dtype=bool)
        dual = np.zeros(nC)
        if nL > 0:
            dual -= lb_L @ D_L
        if nU > 0:
            dual -= ub_U @ D_U
        sd_ok = np.abs(dual - x[I_check]) < tol * (1 + np.abs(x[I_check]))
        conf_min = sL & sU & sd_ok & (~res_min[I_check])

        # Max direction
        sL = np.all(D_L >= -tol, axis=0) if nL > 0 else np.ones(nC, dtype=bool)
        sU = np.all(D_U <= tol, axis=0) if nU > 0 else np.ones(nC, dtype=bool)
        dual_max = np.zeros(nC)
        if nL > 0:
            dual_max += lb_L @ D_L
        if nU > 0:
            dual_max += ub_U @ D_U
        sd_ok = np.abs(dual_max + x[I_check]) < tol * (1 + np.abs(x[I_check]))
        conf_max = sL & sU & sd_ok & (~res_max[I_check])

        self.confirmed_min[I_check] = conf_min
        self.confirmed_max[I_check] = conf_max

    def _nullspace_augment(self, S_csc, x, lb, ub, I_set, L_set, U_set,
                           m, nI, nL, nU, cand_min, cand_max, tol):
        """Stage 2: null-space search for reactions that failed stage 1."""
        S_I = S_csc[:, I_set].toarray()

        N = _null_space(S_I.T)  # m x d
        d = N.shape[1]
        if d == 0:
            return  # stage 1 was already exact

        P = np.linalg.pinv(S_I.T)  # m x nI
        SL_P = S_csc[:, L_set].T.dot(P) if nL > 0 else None
        SU_P = S_csc[:, U_set].T.dot(P) if nU > 0 else None
        lb_L = lb[L_set] if nL > 0 else None
        ub_U = ub[U_set] if nU > 0 else None

        self.n_checks += int(cand_min.sum() + cand_max.sum())

        SL_N = S_csc[:, L_set].T.dot(N) if nL > 0 else np.zeros((0, d))
        SU_N = S_csc[:, U_set].T.dot(N) if nU > 0 else np.zeros((0, d))

        a_sd = np.zeros(d)
        if nL > 0:
            a_sd -= lb_L @ SL_N
        if nU > 0:
            a_sd -= ub_U @ SU_N

        parts = []
        if nL > 0:
            parts.append(SL_N)
        if nU > 0:
            parts.append(-SU_N)
        A_sign = np.vstack(parts) if parts else np.zeros((0, d))
        ns_norms = np.linalg.norm(A_sign, axis=1)

        self._run_filtered(d, SL_P, SU_P, A_sign, a_sd, ns_norms,
                           lb_L, ub_U, x, I_set, nL, nU,
                           cand_min, cand_max, tol)

    def _run_filtered(self, d, SL_P, SU_P, A_sign, a_sd, ns_norms,
                      lb_L, ub_U, x, I_set, nL, nU,
                      cand_min, cand_max, tol):
        """Three-filter pipeline per candidate (d > 0)."""
        c_obj = np.zeros(d)
        A_eq_sd = a_sd.reshape(1, -1)
        z_bounds = [(-1e12, 1e12)] * d

        for idx in range(len(I_set)):
            j = I_set[idx]
            for sign, cand, arr in [
                (1.0, cand_min, self.confirmed_min),
                (-1.0, cand_max, self.confirmed_max),
            ]:
                if not cand[idx]:
                    continue

                sl_p = sign * SL_P[:, idx] if nL > 0 else np.array([])
                su_p = sign * SU_P[:, idx] if nU > 0 else np.array([])

                vL = sl_p > tol if nL > 0 else np.zeros(0, dtype=bool)
                vU = su_p < -tol if nU > 0 else np.zeros(0, dtype=bool)

                dual_p = 0.0
                if nL > 0:
                    dual_p -= lb_L @ sl_p
                if nU > 0:
                    dual_p -= ub_U @ su_p
                target = sign * x[j]

                # Filter 2a: pseudoinverse passes
                if not np.any(vL) and not np.any(vU):
                    if abs(dual_p - target) < tol * (1 + abs(target)):
                        arr[j] = True
                        continue

                # Filter 2b: unfixable violations
                viol = np.concatenate([vL, vU])
                if np.any(viol & (ns_norms < 1e-12)):
                    continue

                # Filter 3: null-space feasibility LP
                rhs_L = -sl_p if nL > 0 else np.array([])
                rhs_U = su_p if nU > 0 else np.array([])
                b_sign = np.concatenate([rhs_L, rhs_U])
                b_sd = target - dual_p

                try:
                    res = _linprog(c_obj, A_ub=A_sign, b_ub=b_sign,
                                   A_eq=A_eq_sd, b_eq=[b_sd],
                                   bounds=z_bounds, method='highs')
                    if res.success and res.status == 0:
                        arr[j] = True
                        self.n_ns_lps += 1
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Compression helpers
# ---------------------------------------------------------------------------

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
        # Fast copy: swap solver with empty stub so deepcopy(solver) is cheap,
        # then populate stub with variable/constraint names (bounds and
        # coefficients are skipped by suppress_lp_context).
        saved_solver = model._solver
        model._solver = model.problem.Model()
        try:
            cmp_model = model.copy()
        finally:
            model._solver = saved_solver
        cmp_model._populate_solver(cmp_model.reactions, cmp_model.metabolites)
        remove_blocked_reactions(cmp_model)
        stoichmat_coeff2rational(cmp_model)
        n_before = len(cmp_model.reactions)
        # Single-pass coupled compression (NULLSPACE only, no RECURSIVE iteration)
        for r in cmp_model.reactions:
            r.gene_reaction_rule = ''
        result = compress_cobra_model(cmp_model,
                                      methods=[CompressionMethod.NULLSPACE],
                                      in_place=True)
        rmap = result.reaction_map
        if len(rmap) < n_before:
            cmp_maps.append(rmap)
        # Coupling can create new dependent rows — remove for DualChecker LU
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
                    coeff_dict[cmp_id] = sum(
                        coeff_dict.pop(k) * float(orig_map[k]) for k in lumped
                    )
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
        {"minimum": [result_min[r] for r in orig_reaction_ids],
         "maximum": [result_max[r] for r in orig_reaction_ids]},
        index=orig_reaction_ids,
    )


# ---------------------------------------------------------------------------
# Global scan LP helper
# ---------------------------------------------------------------------------

def _build_abssum_lp(S_eq, b_eq, A_ineq, b_ineq, lb, ub, solver, BIG=1000.0):
    """Build LP for min sum(|x|) via variable splitting.

    For each reaction j:
      - Forward only (lb >= 0): |x_j| = x_j, obj coeff = +1
      - Backward only (ub <= 0): |x_j| = -x_j, obj coeff = -1
      - Reversible (lb < 0 < ub): split x_j = p_j - n_j (p_j, n_j >= 0)
        with auxiliary equality x_j - p_j + n_j = 0, obj = +1 on both

    Returns (lp, n_orig) where:
      - lp: MILP_LP ready to solve (x[:n_orig] gives the reaction fluxes)
      - n_orig: number of original reaction variables
    """
    n = len(lb)
    tol = 1e-9

    # Classify reactions
    fwd = lb >= -tol          # forward-only or fixed
    bwd = ub <= tol           # backward-only or fixed
    rev = (~fwd) & (~bwd)     # truly reversible
    n_rev = int(rev.sum())
    rev_idx = np.where(rev)[0]

    # Extended variable vector: [x_0..x_{n-1}, p_0..p_{k-1}, n_0..n_{k-1}]
    n_ext = n + 2 * n_rev

    # Objective: min sum(|x|)
    c = np.zeros(n_ext)
    for j in range(n):
        if rev[j]:
            pass  # handled via p/n below
        elif bwd[j]:
            c[j] = -1.0  # |x_j| = -x_j
        else:
            c[j] = 1.0   # |x_j| = x_j (fwd or fixed)
    for k, j in enumerate(rev_idx):
        c[n + k] = 1.0       # p_k
        c[n + n_rev + k] = 1.0  # n_k

    # Equality constraints: original S*x = 0 (+ extras) + splitting equalities
    # Splitting: x_j - p_k + n_k = 0  for each reversible reaction k
    if n_rev > 0:
        # Build splitting equality block: n_rev rows x n_ext cols
        rows_split = np.arange(n_rev)
        # x_j coefficient = 1
        cols_x = rev_idx
        data_x = np.ones(n_rev)
        # p_k coefficient = -1
        cols_p = np.arange(n, n + n_rev)
        data_p = -np.ones(n_rev)
        # n_k coefficient = +1
        cols_n = np.arange(n + n_rev, n + 2 * n_rev)
        data_n = np.ones(n_rev)

        rows_all = np.concatenate([rows_split, rows_split, rows_split])
        cols_all = np.concatenate([cols_x, cols_p, cols_n])
        data_all = np.concatenate([data_x, data_p, data_n])
        A_split = sparse.csr_matrix((data_all, (rows_all, cols_all)),
                                    shape=(n_rev, n_ext))

        # Extend original equalities to n_ext columns
        S_ext = sparse.hstack([S_eq, sparse.csr_matrix((S_eq.shape[0], 2 * n_rev))])
        A_eq_full = sparse.vstack([S_ext, A_split])
        b_eq_full = list(b_eq) + [0.0] * n_rev

        if A_ineq.shape[0] > 0:
            A_ineq_ext = sparse.hstack([A_ineq, sparse.csr_matrix((A_ineq.shape[0], 2 * n_rev))])
        else:
            A_ineq_ext = sparse.csr_matrix((0, n_ext))
    else:
        A_eq_full = S_eq
        b_eq_full = list(b_eq)
        A_ineq_ext = A_ineq

    # Bounds
    lb_ext = np.zeros(n_ext)
    ub_ext = np.zeros(n_ext)
    lb_ext[:n] = lb
    ub_ext[:n] = ub
    # Clamp inf bounds for objective push (doesn't affect feasibility)
    for j in range(n):
        if np.isinf(lb_ext[j]):
            lb_ext[j] = -BIG
        if np.isinf(ub_ext[j]):
            ub_ext[j] = BIG
    # p_k bounds: [0, ub_j] (clamped)
    for k, j in enumerate(rev_idx):
        lb_ext[n + k] = 0.0
        ub_ext[n + k] = min(ub[j], BIG)
    # n_k bounds: [0, -lb_j] (clamped)
    for k, j in enumerate(rev_idx):
        lb_ext[n + n_rev + k] = 0.0
        ub_ext[n + n_rev + k] = min(-lb[j], BIG)

    lp = MILP_LP(c=c.tolist(), A_ineq=A_ineq_ext, b_ineq=list(b_ineq),
                 A_eq=A_eq_full, b_eq=b_eq_full,
                 lb=lb_ext.tolist(), ub=ub_ext.tolist(), solver=solver)
    return lp, n


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
        Run global scan LPs (min/max sum(|x|)) in Phase 1 to pre-resolve many
        objectives without individual LPs.  None = auto (always True).
    threads : int or None, optional (default None)
        Number of parallel workers for Phase 2 dispatch.  None = auto
        (Configuration().processes if n >= 1000, else 1).
    verbose : bool, optional (default False)

    Returns
    -------
    pandas.DataFrame with columns 'minimum' and 'maximum', indexed by reaction ID.
    """
    compress = kwargs.pop('compress', None)
    precheck = kwargs.pop('precheck', None)
    threads = kwargs.pop('threads', None)
    orig_reaction_ids = model.reactions.list_attr("id")
    n_original = len(orig_reaction_ids)
    cmp_maps = []

    # Auto-tuning
    if compress is None:
        compress = n_original >= 200
    if precheck is None:
        precheck = True
    if threads is None:
        threads = Configuration().processes if n_original >= 1000 else 1
    threads = max(1, int(threads))

    if compress:
        model, cmp_maps = _compress_for_fva(model)

    reaction_ids = model.reactions.list_attr("id")
    n_orig = len(reaction_ids)

    has_constraints = CONSTRAINTS in kwargs and kwargs[CONSTRAINTS]
    if has_constraints:
        kwargs[CONSTRAINTS] = parse_constraints(kwargs[CONSTRAINTS], orig_reaction_ids)
        if cmp_maps:
            kwargs[CONSTRAINTS] = _map_constraints(
                kwargs[CONSTRAINTS], cmp_maps, reaction_ids)
        A_ineq_extra, b_ineq_extra, A_eq_extra, b_eq_extra = lineqlist2mat(
            kwargs[CONSTRAINTS], reaction_ids)

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

    lp = MILP_LP(A_ineq=A_ineq, b_ineq=b_ineq, A_eq=A_eq, b_eq=b_eq,
                 lb=lb.tolist(), ub=ub.tolist(), solver=solver)

    _, _, status = lp.solve()
    if status not in [OPTIMAL, UNBOUNDED]:
        logging.error('FVA problem not feasible.')
        return DataFrame(
            {"minimum": [nan] * n_orig, "maximum": [nan] * n_orig},
            index=reaction_ids,
        )

    S_csc = S.tocsc()

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

    # Stats
    lps_solved = 0
    total_bound_resolved = 0
    total_dual_resolved = 0
    total_dual_checks = 0
    total_ns_lps = 0
    t_solve = 0.0
    t_dual = 0.0
    tol_bound = 1e-9
    has_extra_constraints = has_constraints

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

    def _apply_dual(x_local):
        """Build DualChecker, apply confirmed results."""
        nonlocal total_dual_resolved, total_dual_checks, total_ns_lps, t_dual
        if has_extra_constraints:
            return
        t0 = _time.perf_counter()
        checker = DualChecker(S, x_local, lb, ub, res_min, res_max,
                              incumbent_min, incumbent_max)
        newly_min = checker.confirmed_min & (~res_min)
        n_min = int(newly_min.sum())
        if n_min:
            res_min[newly_min] = True
            incumbent_min[newly_min] = x_local[newly_min]
            total_dual_resolved += n_min
        newly_max = checker.confirmed_max & (~res_max)
        n_max = int(newly_max.sum())
        if n_max:
            res_max[newly_max] = True
            incumbent_max[newly_max] = x_local[newly_max]
            total_dual_resolved += n_max
        total_dual_checks += checker.n_checks
        total_ns_lps += checker.n_ns_lps
        t_dual += _time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Phase 1: Global scan LPs + v=0 feasibility + DualChecker
    # ------------------------------------------------------------------
    # 1a: v=0 feasibility check — free resolutions, no LP needed
    v0_feasible = (not np.any(lb > tol_bound) and not np.any(ub < -tol_bound)
                   and not has_extra_constraints)
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

    if precheck:
        # 1b: min(sum(|x|)) scan — pushes reactions toward zero
        # Effective for resolving lb=0 / ub=0 bounds in one shot.
        scan_lp, n_scan = _build_abssum_lp(
            A_eq, b_eq, A_ineq, b_ineq, lb, ub, solver)
        scan_lp.set_lp_method(LP_METHOD_DUAL)
        n_ext = len(scan_lp.c)

        t0 = _time.perf_counter()
        x_list_scan, _, scan_status = scan_lp.solve()
        t_solve += _time.perf_counter() - t0
        lps_solved += 1
        resolved_absmin = 0

        if scan_status == OPTIMAL:
            x_scan = np.array(x_list_scan[:n_scan], dtype=np.float64)
            before = int(res_max.sum() + res_min.sum())
            _bound_scan(x_scan)
            _apply_dual(x_scan)
            np.maximum(incumbent_max, x_scan, out=incumbent_max)
            np.minimum(incumbent_min, x_scan, out=incumbent_min)
            resolved_absmin = int(res_max.sum() + res_min.sum()) - before

        if verbose:
            n_done_iter = int(res_max.sum() + res_min.sum())
            logging.info(
                f"  Phase 1 min|x|: +{resolved_absmin} "
                f"({n_done_iter}/{2*n_orig} resolved)")

        # 1c: Iterative push-to-bounds — directed per-reaction objectives
        # Push unresolved-max reactions toward ub, unresolved-min toward lb.
        # Dual simplex warm-start makes re-optimization nearly free.
        push_iter = 0
        while True:
            resolved_this_round = 0
            resolved_push_ub = 0
            resolved_push_lb = 0

            # Push toward ub (resolve max bounds): maximize x_j → c[j] = -1
            c_push = np.zeros(n_ext)
            any_unresolved = False
            for j in range(n_scan):
                if not res_max[j]:
                    c_push[j] = -1.0
                    any_unresolved = True

            if any_unresolved:
                scan_lp.set_objective(c_push.tolist())
                t0 = _time.perf_counter()
                x_list_scan, _, scan_status = scan_lp.solve()
                t_solve += _time.perf_counter() - t0
                lps_solved += 1

                if scan_status == OPTIMAL:
                    x_scan = np.array(x_list_scan[:n_scan], dtype=np.float64)
                    before = int(res_max.sum() + res_min.sum())
                    _bound_scan(x_scan)
                    _apply_dual(x_scan)
                    np.maximum(incumbent_max, x_scan, out=incumbent_max)
                    np.minimum(incumbent_min, x_scan, out=incumbent_min)
                    resolved_push_ub = int(res_max.sum() + res_min.sum()) - before
                    resolved_this_round += resolved_push_ub

            # Push toward lb (resolve min bounds): minimize x_j → c[j] = +1
            c_push = np.zeros(n_ext)
            any_unresolved = False
            for j in range(n_scan):
                if not res_min[j]:
                    c_push[j] = 1.0
                    any_unresolved = True

            if any_unresolved:
                scan_lp.set_objective(c_push.tolist())
                t0 = _time.perf_counter()
                x_list_scan, _, scan_status = scan_lp.solve()
                t_solve += _time.perf_counter() - t0
                lps_solved += 1

                if scan_status == OPTIMAL:
                    x_scan = np.array(x_list_scan[:n_scan], dtype=np.float64)
                    before = int(res_max.sum() + res_min.sum())
                    _bound_scan(x_scan)
                    _apply_dual(x_scan)
                    np.maximum(incumbent_max, x_scan, out=incumbent_max)
                    np.minimum(incumbent_min, x_scan, out=incumbent_min)
                    resolved_push_lb = int(res_max.sum() + res_min.sum()) - before
                    resolved_this_round += resolved_push_lb

            if verbose:
                n_done_iter = int(res_max.sum() + res_min.sum())
                logging.info(
                    f"  Phase 1 push {push_iter}: "
                    f"ub +{resolved_push_ub}, lb +{resolved_push_lb} "
                    f"({n_done_iter}/{2*n_orig} resolved)")

            push_iter += 1
            if resolved_this_round < 5:
                break

        del scan_lp

    # ------------------------------------------------------------------
    # Phase 1d: Inf-bound scan — resolve truly unbounded reactions
    # ------------------------------------------------------------------
    # With inf bounds, scan LPs (clamped to BIG) can't resolve reactions
    # at infinity. Check unresolved inf-bound directions individually.
    has_inf_ub = np.isinf(ub) & (~res_max)
    has_inf_lb = np.isinf(lb) & (~res_min)
    n_inf_check = int(has_inf_ub.sum() + has_inf_lb.sum())
    if n_inf_check > 0:
        lp_inf = MILP_LP(A_ineq=A_ineq, b_ineq=b_ineq, A_eq=A_eq, b_eq=b_eq,
                         lb=lb.tolist(), ub=ub.tolist(), solver=solver)
        prev_col = -1
        for j in np.where(has_inf_ub)[0]:
            C = [[int(j), -1.0]] if prev_col < 0 or prev_col == j \
                else [[int(j), -1.0], [prev_col, 0.0]]
            if solver in ('cplex', 'gurobi'):
                lp_inf.backend.set_objective_idx(C)
            else:
                lp_inf.set_objective_idx(C)
            prev_col = int(j)
            t0 = _time.perf_counter()
            _, obj_val, status = lp_inf.solve()
            t_solve += _time.perf_counter() - t0
            lps_solved += 1
            if status == UNBOUNDED:
                res_max[j] = True
                incumbent_max[j] = np.inf
                total_bound_resolved += 1
            elif status == OPTIMAL:
                res_max[j] = True
                incumbent_max[j] = -obj_val
                total_bound_resolved += 1
        for j in np.where(has_inf_lb)[0]:
            C = [[int(j), 1.0]] if prev_col < 0 or prev_col == j \
                else [[int(j), 1.0], [prev_col, 0.0]]
            if solver in ('cplex', 'gurobi'):
                lp_inf.backend.set_objective_idx(C)
            else:
                lp_inf.set_objective_idx(C)
            prev_col = int(j)
            t0 = _time.perf_counter()
            _, obj_val, status = lp_inf.solve()
            t_solve += _time.perf_counter() - t0
            lps_solved += 1
            if status == UNBOUNDED:
                res_min[j] = True
                incumbent_min[j] = -np.inf
                total_bound_resolved += 1
            elif status == OPTIMAL:
                res_min[j] = True
                incumbent_min[j] = obj_val
                total_bound_resolved += 1
        del lp_inf

    # ------------------------------------------------------------------
    # Phase 2: Dispatch remaining objectives — sequential or parallel
    # ------------------------------------------------------------------
    n_done = int(res_max.sum() + res_min.sum())
    n_remaining = 2 * n_orig - n_done
    phase2_entry_count = n_remaining  # for stats

    if n_remaining > 0 and n_remaining >= 100 and threads > 1:
        # Parallel dispatch via SDPool
        # Build list of unresolved objective indices (even=max, odd=min)
        unresolved = []
        for j in range(n_orig):
            if not res_max[j]:
                unresolved.append(2 * j)      # even = max
            if not res_min[j]:
                unresolved.append(2 * j + 1)  # odd = min

        x_par = [nan] * (2 * n_orig)
        t0 = _time.perf_counter()
        if solver == GLPK:
            with SDPool(threads, initializer=fva_worker_init_glpk,
                        initargs=(A_ineq, b_ineq, A_eq, b_eq,
                                  lb.tolist(), ub.tolist())) as pool:
                chunk_size = max(1, len(unresolved) // threads)
                for i, value in pool.imap_unordered(
                        fva_worker_compute_glpk, unresolved,
                        chunksize=chunk_size):
                    x_par[i] = value
        else:
            with SDPool(threads, initializer=fva_worker_init,
                        initargs=(A_ineq, b_ineq, A_eq, b_eq,
                                  lb.tolist(), ub.tolist(), solver)) as pool:
                chunk_size = max(1, len(unresolved) // threads)
                for i, value in pool.imap_unordered(
                        fva_worker_compute, unresolved,
                        chunksize=chunk_size):
                    x_par[i] = value
        t_solve += _time.perf_counter() - t0
        lps_solved += len(unresolved)

        # NaN retry with fresh LPs
        nan_idx = [i for i in unresolved if np.isnan(x_par[i])]
        if nan_idx:
            _BATCH = 50
            while nan_idx:
                lp_retry = MILP_LP(A_ineq=A_ineq, b_ineq=b_ineq,
                                   A_eq=A_eq, b_eq=b_eq,
                                   lb=lb.tolist(), ub=ub.tolist(), solver=solver)
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
            if not res_max[j] and not np.isnan(x_par[i_max]):
                res_max[j] = True
                incumbent_max[j] = -x_par[i_max]
            i_min = 2 * j + 1
            if not res_min[j] and not np.isnan(x_par[i_min]):
                res_min[j] = True
                incumbent_min[j] = x_par[i_min]

    elif n_remaining > 0:
        # Sequential dispatch — simple loop, no hub-first, no dual check
        prev_col = -1

        def _rebuild_lp():
            nonlocal lp, prev_col
            lp = MILP_LP(A_ineq=A_ineq, b_ineq=b_ineq, A_eq=A_eq, b_eq=b_eq,
                         lb=lb.tolist(), ub=ub.tolist(), solver=solver)
            prev_col = -1

        _rebuild_lp()
        seq_count = 0

        for j in range(n_orig):
            for direction in (1, -1):
                if direction == 1 and res_max[j]:
                    continue
                if direction == -1 and res_min[j]:
                    continue

                # Periodic rebuild to limit warm-start degeneration
                if seq_count > 0 and seq_count % 200 == 0:
                    _rebuild_lp()

                sig = -direction
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

                if status == UNBOUNDED:
                    if direction == 1:
                        res_max[j] = True
                        incumbent_max[j] = np.inf
                    else:
                        res_min[j] = True
                        incumbent_min[j] = -np.inf
                    continue
                elif status != OPTIMAL:
                    if direction == 1:
                        res_max[j] = True
                    else:
                        res_min[j] = True
                    continue

                # Guard: LP optimum must not be worse than incumbent
                if direction == 1:
                    val, inc = -obj_val, incumbent_max[j]
                    bad = np.isfinite(inc) and val < inc - 1e-6 * (1 + abs(inc))
                else:
                    val, inc = obj_val, incumbent_min[j]
                    bad = np.isfinite(inc) and val > inc + 1e-6 * (1 + abs(inc))
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

    # ------------------------------------------------------------------
    # Assemble results
    # ------------------------------------------------------------------
    incumbent_max[np.abs(incumbent_max) < 1e-11] = 0.0
    incumbent_min[np.abs(incumbent_min) < 1e-11] = 0.0

    fva_result = DataFrame(
        {"minimum": incumbent_min, "maximum": incumbent_max},
        index=reaction_ids,
    )

    if verbose:
        cmp_msg = ""
        if cmp_maps:
            cmp_msg = f" (compressed {n_original}→{n_orig} rxns)"
        logging.info(
            f"  speedy_fva done{cmp_msg}: {lps_solved} LPs, "
            f"{total_bound_resolved} bound-resolved, "
            f"{total_dual_resolved} dual-resolved "
            f"({total_dual_checks} checks, {total_ns_lps} NS-LPs), "
            f"{2*n_orig} total objectives")
        logging.info(
            f"  timing: solve={t_solve:.2f}s, dual={t_dual:.3f}s, "
            f"threads={threads}")

    fva_result.attrs['lps_solved'] = lps_solved
    fva_result.attrs['trivial_resolved'] = total_bound_resolved
    fva_result.attrs['duality_resolved'] = total_dual_resolved
    fva_result.attrs['duality_checks'] = total_dual_checks
    fva_result.attrs['ns_lps'] = total_ns_lps
    fva_result.attrs['phase2_remaining'] = phase2_entry_count
    if cmp_maps:
        fva_result.attrs['n_original'] = n_original
        fva_result.attrs['n_compressed'] = n_orig

    # Expand compressed results back to original reaction space
    if cmp_maps:
        expanded = _expand_fva(fva_result, cmp_maps, orig_reaction_ids)
        expanded.attrs = fva_result.attrs
        fva_result = expanded

    return fva_result
