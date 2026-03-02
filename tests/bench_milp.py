"""Benchmark: MCS MILP formulation variants.

Compares StrainDesign's current compact-FLB formulation against the paper's
FLB and nullspace-based (NB) formulations from Klamt et al. 2020 ("Speeding up
the core algorithm for the dual calculation of minimal cut sets").

Tests four dimensions:
  1. Dual formulation: SD-current / FLB-explicit / NB (nullspace)
  2. Reaction splitting: condensed / split (fwd+rev)
  3. Indicator strategy: indic-eq / indic-ineq / aux-var
  4. Bounds: orig / unbind / fva / cap

Models:
  - e_coli_core: COBRApy built-in, reaction-level KOs, 353 expected MCS
  - iMLcore: pre-compressed snapshot (gene-level KOs already extended),
             462 rxns, 205 knockable, 108 expected MCS

Usage:
    conda run -n straindesign python tests/bench_milp.py
"""

import json
import logging
import pickle
import sys
import time
from copy import deepcopy
from math import isinf
from pathlib import Path

import numpy as np
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cobra.io import load_model
from cobra.util import create_stoichiometric_matrix

from straindesign import (
    SDMILP, SDModule, IndicatorConstraints, MILP_LP,
    remove_dummy_bounds,
)
from straindesign.compression import (
    nullspace, RationalMatrix,
)
from straindesign.names import SUPPRESS
from straindesign.parse_constr import lineqlist2mat
from straindesign.strainDesignProblem import build_primal_from_cbm

logging.basicConfig(level=logging.WARNING, format="%(message)s")


# ── Test matrix ─────────────────────────────────────────────────────────

# (formulation, split, indicator, bounds)
TESTS = [
    # === Baselines with all bound variants ===
    ("SD",  "condensed", "indic-eq",   "orig"),
    ("SD",  "condensed", "indic-eq",   "unbind"),
    ("SD",  "condensed", "indic-eq",   "fva"),
    ("SD",  "condensed", "indic-eq",   "cap"),
    ("FLB", "condensed", "indic-eq",   "orig"),
    ("FLB", "condensed", "indic-eq",   "unbind"),
    ("FLB", "condensed", "indic-eq",   "fva"),
    ("FLB", "condensed", "indic-eq",   "cap"),
    ("NB",  "condensed", "indic-eq",   "orig"),
    ("NB",  "condensed", "indic-eq",   "unbind"),
    ("NB",  "condensed", "indic-eq",   "fva"),
    ("NB",  "condensed", "indic-eq",   "cap"),
    # === Remaining formulation variants (orig bounds only) ===
    ("SD",  "split",     "indic-eq",   "orig"),
    ("FLB", "split",     "indic-eq",   "orig"),
    ("NB",  "split",     "indic-eq",   "orig"),
    ("FLB", "condensed", "indic-ineq", "orig"),
    ("NB",  "condensed", "indic-ineq", "orig"),
    ("FLB", "split",     "indic-ineq", "orig"),
    ("NB",  "split",     "indic-ineq", "orig"),
    # === Auxiliary-variable indicators (SD only, FLB/NB already simple) ===
    ("SD",  "condensed", "aux-var",    "orig"),
    ("SD",  "split",     "aux-var",    "orig"),
]


# ═══════════════════════════════════════════════════════════════════════
# Model setup
# ═══════════════════════════════════════════════════════════════════════

BIOMASS_CONSTRAINT = "BIOMASS_Ecoli_core_w_GAM >= 0.001"
MAX_COST = 3


def load_ecoli_core():
    """Load e_coli_core, remove dummy bounds, build sd_modules + kwargs."""
    model = load_model("e_coli_core")
    remove_dummy_bounds(model)

    constraints = [[{"BIOMASS_Ecoli_core_w_GAM": 1.0}, ">=", 0.001]]
    sd_modules = [{"constraints": constraints}]

    ko_cost = {r.id: 1.0 for r in model.reactions}
    kwargs_milp = {"ko_cost": ko_cost, "max_cost": MAX_COST}

    return model, sd_modules, kwargs_milp


SNAPSHOT_DIR = Path(__file__).resolve().parent / "milp_snapshot"


def load_imlcore_snapshot():
    """Load pre-compressed iMLcore model from snapshot.

    The snapshot contains a compressed model (462 rxns, 205 knockable)
    with gene-level KOs already extended via GPR propagation.
    Expected: 108 MCS with max_cost=3.
    """
    with open(SNAPSHOT_DIR / "cmp_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(SNAPSHOT_DIR / "milp_args.pkl", "rb") as f:
        args = pickle.load(f)
    with open(SNAPSHOT_DIR / "metadata.json", "r") as f:
        metadata = json.load(f)

    # SDModule (dict subclass) → plain dict for our builders
    mod = args["sd_modules"][0]
    sd_modules = [{"constraints": mod["constraints"]}]

    kwargs_milp = {
        "ko_cost": args["kwargs_milp"]["ko_cost"],
        "max_cost": args["kwargs_milp"]["max_cost"],
    }

    expected = metadata["expected_milp_count"]
    return model, sd_modules, kwargs_milp, expected


# ═══════════════════════════════════════════════════════════════════════
# Bound variants
# ═══════════════════════════════════════════════════════════════════════

def fva_tighten_bounds(model):
    """Tighten model bounds to FVA range.

    Replaces each bound with the actual achievable range from FVA.
    Same number of finite bounds, but tighter values.
    """
    from straindesign import fva
    fva_result = fva(model)
    for i, rxn in enumerate(model.reactions):
        rxn.lower_bound = max(rxn.lower_bound, fva_result.iloc[i]["minimum"])
        rxn.upper_bound = min(rxn.upper_bound, fva_result.iloc[i]["maximum"])
    return model


def fva_unbind(model, tol=1e-6):
    """Remove non-binding bounds via FVA.

    If FVA shows a reaction never reaches its bound, set that bound to
    +/-inf.  This removes dummy bounds (e.g. the default +-1000 in cobra)
    that don't actually constrain the model, reducing the number of
    bound-derived constraints in the MILP dual.
    """
    from straindesign import fva
    fva_result = fva(model)
    for i, rxn in enumerate(model.reactions):
        fva_min = fva_result.iloc[i]["minimum"]
        fva_max = fva_result.iloc[i]["maximum"]
        # Lower bound: non-binding if FVA minimum is strictly above it
        if not isinf(rxn.lower_bound) and fva_min > rxn.lower_bound + tol:
            rxn.lower_bound = -np.inf
        # Upper bound: non-binding if FVA maximum is strictly below it
        if not isinf(rxn.upper_bound) and fva_max < rxn.upper_bound - tol:
            rxn.upper_bound = np.inf
    return model


def cap_unbounded(model, M=1000.0):
    """Add finite bounds where currently unbounded.

    Keeps all existing finite bounds as-is. Only adds +-M for reactions
    that are currently at +-inf.  This enables big-M linking in the SD
    formulation (instead of indicator constraints) and adds bound
    constraints in the FLB/NB dual.
    """
    for rxn in model.reactions:
        if isinf(rxn.lower_bound) and rxn.lower_bound < 0:
            rxn.lower_bound = -M
        if isinf(rxn.upper_bound) and rxn.upper_bound > 0:
            rxn.upper_bound = M
    return model


# ═══════════════════════════════════════════════════════════════════════
# Reaction splitting
# ═══════════════════════════════════════════════════════════════════════

def split_reversible_model(model):
    """Split reversible reactions into fwd + rev in the cobra model.

    Returns (new_model, split_map) where:
      split_map[orig_id] = (fwd_id, rev_id) or (orig_id,) if irreversible
    After split: all lb >= 0.
    """
    import cobra
    new_model = model.copy()
    split_map = {}
    rxns_to_add = []
    for rxn in list(new_model.reactions):
        if rxn.lower_bound < 0:
            rev_id = rxn.id + "_rev"
            split_map[rxn.id] = (rxn.id, rev_id)
            rev_rxn = cobra.Reaction(rev_id)
            rev_rxn.name = rxn.name + " (reverse)"
            rev_rxn.lower_bound = 0.0
            rev_rxn.upper_bound = -rxn.lower_bound if not isinf(rxn.lower_bound) else np.inf
            rev_rxn.add_metabolites({m: -c for m, c in rxn.metabolites.items()})
            rxns_to_add.append(rev_rxn)
            # Set fwd bounds atomically (avoids cobra validation if ub < 0)
            fwd_ub = max(rxn.upper_bound, 0.0)
            rxn.bounds = (0.0, fwd_ub)
        else:
            split_map[rxn.id] = (rxn.id,)
    new_model.add_reactions(rxns_to_add)
    return new_model, split_map


# ═══════════════════════════════════════════════════════════════════════
# Formulation builders
# ═══════════════════════════════════════════════════════════════════════

def _parse_constraint(sd_modules, model):
    """Extract V_ineq, v_ineq, V_eq, v_eq from sd_modules for model."""
    reac_ids = [r.id for r in model.reactions]
    module = sd_modules[0]
    V_ineq, v_ineq, V_eq, v_eq = lineqlist2mat(module["constraints"], reac_ids)
    return V_ineq, v_ineq, V_eq, v_eq


def _build_ko_maps(reac_ids, kwargs_milp, split_map=None):
    """Build knockable reaction data structures.

    Returns (knockable_orig, knockable_cost, orig_to_split_idx,
             knockable, ko_cost_vec).
    """
    orig_ko = kwargs_milp.get("ko_cost", None)
    if orig_ko is None:
        orig_ko = {}

    ko_cost_vec = []
    knockable = []
    for rid in reac_ids:
        orig_id = rid[:-4] if rid.endswith("_rev") else rid
        if orig_id in orig_ko:
            ko_cost_vec.append(orig_ko[orig_id])
            knockable.append(True)
        elif not orig_ko:
            ko_cost_vec.append(1.0)
            knockable.append(True)
        else:
            ko_cost_vec.append(np.nan)
            knockable.append(False)

    # Map original reaction -> split indices
    orig_to_split_idx = {}
    orig_ids_ordered = []
    for i, rid in enumerate(reac_ids):
        orig_id = rid[:-4] if rid.endswith("_rev") else rid
        if orig_id not in orig_to_split_idx:
            orig_to_split_idx[orig_id] = []
            orig_ids_ordered.append(orig_id)
        orig_to_split_idx[orig_id].append(i)

    knockable_orig = []
    knockable_cost = []
    for orig_id in orig_ids_ordered:
        idx0 = orig_to_split_idx[orig_id][0]
        if knockable[idx0]:
            knockable_orig.append(orig_id)
            knockable_cost.append(ko_cost_vec[idx0])

    return knockable_orig, knockable_cost, orig_to_split_idx, knockable, ko_cost_vec


# ── SD-current (existing SDMILP pipeline) ────────────────────────────

def build_sd_current(model, sd_modules, kwargs_milp, do_split, solver,
                     seed=None, trim=False):
    """Wrap existing SDMILP path.

    If trim=True, removes fixed (ub=0) z-variables and returns a plain MILP_LP.
    """
    m = model.copy()
    kw = deepcopy(kwargs_milp)
    kw["solver"] = solver
    if seed is not None:
        kw["seed"] = seed

    if do_split:
        m, split_map = split_reversible_model(m)
        old_ko = kw.get("ko_cost", None)
        if old_ko:
            new_ko = {}
            for orig_id, parts in split_map.items():
                if orig_id in old_ko:
                    for pid in parts:
                        new_ko[pid] = old_ko[orig_id]
            kw["ko_cost"] = new_ko

    # Always create fresh SDModule from constraints
    mods = [SDModule(m, SUPPRESS,
                     constraints=sd_modules[0]["constraints"],
                     skip_checks=True)]
    sd_milp = SDMILP(m, mods, **kw)

    if not trim:
        return sd_milp, m

    return _trim_fixed_binaries(sd_milp, solver, seed), m


def _trim_fixed_binaries(sd_milp, solver, seed=None):
    """Remove fixed (ub=0) z-variables from an SDMILP, return a plain MILP_LP.

    The SD pipeline creates one z per reaction (num_z = numr).
    Non-knockable reactions have ub=0 (fixed to 0).  With Presolve=0
    Gurobi still carries these dead variables.  This function removes them.
    """
    numr = sd_milp.num_z
    # Identify effective binaries
    keep_z = [i for i in range(numr) if sd_milp.ub[i] > 0]
    drop_z = set(range(numr)) - set(keep_z)
    n_cont = sd_milp.A_ineq.shape[1] - numr
    keep_cols = keep_z + list(range(numr, numr + n_cont))

    # Trim matrices
    A_ineq = sd_milp.A_ineq[:, keep_cols]
    A_eq = sd_milp.A_eq[:, keep_cols]
    lb = [sd_milp.lb[i] for i in keep_cols]
    ub = [sd_milp.ub[i] for i in keep_cols]
    c = [sd_milp.c[i] for i in keep_cols]
    vtype = "".join(sd_milp.vtype[i] for i in keep_cols)
    b_ineq = list(sd_milp.b_ineq)
    b_eq = list(sd_milp.b_eq)

    # Remap indicator constraint indices
    old_to_new = {old: new for new, old in enumerate(keep_z)}
    ic = sd_milp.indic_constr
    if ic and hasattr(ic, 'A') and ic.A is not None and ic.A.shape[0] > 0:
        indic_A = ic.A[:, keep_cols]
        indic_binv = [old_to_new[b] for b in ic.binv]
        indic_constr = IndicatorConstraints(
            indic_binv, indic_A, list(ic.b), ic.sense, list(ic.indicval))
    else:
        indic_constr = None

    num_z = len(keep_z)
    milp = MILP_LP(c=c, A_ineq=A_ineq, b_ineq=b_ineq,
                   A_eq=A_eq, b_eq=b_eq,
                   lb=lb, ub=ub, vtype=vtype,
                   indic_constr=indic_constr, solver=solver, seed=seed)
    milp._num_z = num_z
    milp._idx_z = list(range(num_z))
    # Map back from trimmed z-index to original reaction id
    reac_ids = sd_milp.model.reactions.list_attr("id")
    milp._knockable_orig = [reac_ids[i] for i in keep_z]
    milp._cost_vec = [sd_milp.cost[i] for i in keep_z]
    return milp


# ── FLB-explicit (paper's Eq. 14 / 14a) ──────────────────────────────

def build_flb_explicit(model, sd_modules, kwargs_milp, indicator_mode,
                       do_split, solver, seed=None):
    """FLB formulation with explicit v-variables and simple indicators.

    S^T u + A_ineq_full^T w + v = 0   (n equalities)
    b_ineq_full^T w <= -1              (Farkas certificate)
    Indicators link z to v.
    """
    m = model.copy()
    if do_split:
        m, split_map = split_reversible_model(m)
    else:
        split_map = {r.id: (r.id,) for r in m.reactions}

    reac_ids = [r.id for r in m.reactions]
    numr = len(reac_ids)
    max_cost = kwargs_milp.get("max_cost", np.inf)

    knockable_orig, knockable_cost, orig_to_split_idx, knockable, ko_cost_vec = \
        _build_ko_maps(reac_ids, kwargs_milp, split_map)
    num_z = len(knockable_orig)

    V_ineq, v_ineq, V_eq, v_eq = _parse_constraint(sd_modules, m)

    # Get standardized primal
    (A_ineq_p, b_ineq_p, A_eq_p, b_eq_p, lb_p, ub_p, c_p,
     z_map_ci, z_map_ce, z_map_v) = build_primal_from_cbm(
        m, V_ineq, v_ineq, V_eq, v_eq)

    n_eq = A_eq_p.shape[0]

    # Convert inhomogeneous bounds to inequality constraints
    lb_inh = [i for i in np.nonzero(lb_p)[0] if not isinf(lb_p[i])]
    ub_inh = [i for i in np.nonzero(ub_p)[0] if not isinf(ub_p[i])]
    LB_rows = sparse.csr_matrix(
        ([-1.0] * len(lb_inh), (range(len(lb_inh)), lb_inh)),
        shape=(len(lb_inh), numr))
    UB_rows = sparse.csr_matrix(
        ([1.0] * len(ub_inh), (range(len(ub_inh)), ub_inh)),
        shape=(len(ub_inh), numr))
    A_ineq_full = sparse.vstack([A_ineq_p, LB_rows, UB_rows]).tocsr()
    b_ineq_full = b_ineq_p + [-lb_p[i] for i in lb_inh] + [ub_p[i] for i in ub_inh]
    n_ineq_full = A_ineq_full.shape[0]

    # Classify primal variables by sign
    x_geq0 = [i for i in range(numr) if lb_p[i] >= 0 and ub_p[i] > 0]
    x_leq0 = [i for i in range(numr) if lb_p[i] < 0 and ub_p[i] <= 0]
    x_free = [i for i in range(numr) if lb_p[i] < 0 and ub_p[i] > 0]

    # Variable layout: z (num_z) | u (n_eq) | w (n_ineq_full) | v (numr)
    n_u = n_eq
    n_w = n_ineq_full
    n_total = num_z + n_u + n_w + numr

    # Objective: minimize sum(cost * z)
    c_obj = knockable_cost + [0.0] * (n_u + n_w + numr)

    # Equality: A_eq^T u + A_ineq_full^T w + v = 0
    eq_block = sparse.hstack([
        sparse.csr_matrix((numr, num_z)),
        A_eq_p.T,
        A_ineq_full.T,
        sparse.eye(numr),
    ]).tocsr()
    A_eq_milp = eq_block
    b_eq_milp = [0.0] * numr

    # Inequality: Farkas certificate + cost constraint
    farkas_row = sparse.hstack([
        sparse.csr_matrix((1, num_z)),
        sparse.csr_matrix([b_eq_p]),
        sparse.csr_matrix([b_ineq_full]),
        sparse.csr_matrix((1, numr)),
    ]).tocsr()
    cost_row = sparse.lil_matrix((1, n_total))
    for i in range(num_z):
        cost_row[0, i] = knockable_cost[i]

    A_ineq_milp = sparse.vstack([farkas_row, cost_row.tocsr()]).tocsr()
    b_ineq_milp = [-1.0,
                   float(max_cost) if max_cost and not isinf(max_cost)
                   else float(np.sum(np.abs(knockable_cost)))]

    # Bounds: z binary, u free, w >= 0, v free
    lb_milp = [0.0] * num_z + [-np.inf] * n_u + [0.0] * n_w + [-np.inf] * numr
    ub_milp = [1.0] * num_z + [np.inf] * n_u + [np.inf] * n_w + [np.inf] * numr

    # Static sign bounds for NON-KNOCKABLE reactions
    knockable_set = set()
    for orig_id in knockable_orig:
        for idx in orig_to_split_idx[orig_id]:
            knockable_set.add(idx)
    v_base = num_z + n_u + n_w
    for i in x_geq0:
        if i not in knockable_set:
            ub_milp[v_base + i] = 0.0
    for i in x_leq0:
        if i not in knockable_set:
            lb_milp[v_base + i] = 0.0
    for i in x_free:
        if i not in knockable_set:
            lb_milp[v_base + i] = 0.0
            ub_milp[v_base + i] = 0.0

    vtype = "B" * num_z + "C" * (n_u + n_w + numr)

    # Indicator constraints
    z_orig_idx_map = {orig_id: i for i, orig_id in enumerate(knockable_orig)}
    indic_constr = _make_indicators(
        num_z, n_total, numr, knockable_orig, orig_to_split_idx,
        z_orig_idx_map, v_base, x_free, x_geq0, x_leq0,
        indicator_mode, do_split)

    milp = MILP_LP(c=c_obj, A_ineq=A_ineq_milp, b_ineq=b_ineq_milp,
                   A_eq=A_eq_milp, b_eq=b_eq_milp,
                   lb=lb_milp, ub=ub_milp, vtype=vtype,
                   indic_constr=indic_constr, solver=solver, seed=seed)
    milp._num_z = num_z
    milp._idx_z = list(range(num_z))
    milp._knockable_orig = knockable_orig
    milp._cost_vec = knockable_cost
    return milp, m


# ── NB (nullspace-based, paper's Eq. 16 / 16a) ───────────────────────

def build_nb(model, sd_modules, kwargs_milp, indicator_mode,
             do_split, solver, seed=None):
    """Nullspace-based dual formulation.

    K^T v + K^T A_ineq_full^T w = 0   (k equalities, k = dim(null(S)))
    b_ineq_full^T w <= -1              (Farkas certificate)
    Indicators link z to v.
    No u-variables at all.
    """
    m = model.copy()
    if do_split:
        m_orig = m.copy()
        m, split_map = split_reversible_model(m)
    else:
        m_orig = m
        split_map = {r.id: (r.id,) for r in m.reactions}

    reac_ids = [r.id for r in m.reactions]
    numr = len(reac_ids)
    orig_reac_ids = [r.id for r in m_orig.reactions]
    max_cost = kwargs_milp.get("max_cost", np.inf)

    knockable_orig, knockable_cost, orig_to_split_idx, knockable, ko_cost_vec = \
        _build_ko_maps(reac_ids, kwargs_milp, split_map)
    num_z = len(knockable_orig)

    # Compute nullspace K of S (on ORIGINAL unsplit model)
    S_rat = RationalMatrix.from_cobra_model(m_orig)
    K_rat = nullspace(S_rat)
    n_kernel = K_rat.get_column_count()
    K_np = K_rat.to_numpy()  # (n_orig x n_kernel)

    # If split: expand K for split reactions.
    # IMPORTANT: cobra appends _rev reactions at the END of the model.
    # K rows must match the model's reaction order.
    if do_split:
        n_orig = len(orig_reac_ids)
        # First n_orig rows: original reactions in order
        K_fwd = list(K_np)
        # Then: _rev reactions appended at end (cobra order)
        K_rev = []
        rev_fwd_indices = []
        for i, rid in enumerate(orig_reac_ids):
            if len(split_map.get(rid, (rid,))) > 1:
                rev_row = n_orig + len(K_rev)
                rev_fwd_indices.append((i, rev_row))
                K_rev.append(np.zeros(n_kernel))
        K_expanded = np.vstack(K_fwd + K_rev)
        # New basis columns: [1_fwd, 1_rev] for each split reaction
        n_new = len(rev_fwd_indices)
        if n_new > 0:
            new_cols = np.zeros((K_expanded.shape[0], n_new))
            for col_idx, (fwd_row, rev_row) in enumerate(rev_fwd_indices):
                new_cols[fwd_row, col_idx] = 1.0
                new_cols[rev_row, col_idx] = 1.0
            K_np = np.hstack([K_expanded, new_cols])
        else:
            K_np = K_expanded
        n_kernel = K_np.shape[1]

        # Verify S_split * K_split = 0
        S_split = sparse.csr_matrix(create_stoichiometric_matrix(m))
        SK = S_split @ K_np
        residual = np.max(np.abs(SK))
        if residual > 1e-8:
            raise RuntimeError(f"K expansion invalid: S*K residual = {residual}")

    # Parse constraint
    V_ineq, v_ineq, V_eq, v_eq = _parse_constraint(sd_modules, m)

    # Get standardized primal for bound info
    (A_ineq_p, b_ineq_p, A_eq_p, b_eq_p, lb_p, ub_p, c_p,
     z_map_ci, z_map_ce, z_map_v) = build_primal_from_cbm(
        m, V_ineq, v_ineq, V_eq, v_eq)

    # Convert inhomogeneous bounds to inequality constraints
    lb_inh = [i for i in np.nonzero(lb_p)[0] if not isinf(lb_p[i])]
    ub_inh = [i for i in np.nonzero(ub_p)[0] if not isinf(ub_p[i])]
    LB_rows = sparse.csr_matrix(
        ([-1.0] * len(lb_inh), (range(len(lb_inh)), lb_inh)),
        shape=(len(lb_inh), numr))
    UB_rows = sparse.csr_matrix(
        ([1.0] * len(ub_inh), (range(len(ub_inh)), ub_inh)),
        shape=(len(ub_inh), numr))
    A_ineq_full = sparse.vstack([A_ineq_p, LB_rows, UB_rows]).tocsr()
    b_ineq_full = b_ineq_p + [-lb_p[i] for i in lb_inh] + [ub_p[i] for i in ub_inh]
    n_ineq_full = A_ineq_full.shape[0]

    # Classify primal variables by sign
    x_geq0 = [i for i in range(numr) if lb_p[i] >= 0 and ub_p[i] > 0]
    x_leq0 = [i for i in range(numr) if lb_p[i] < 0 and ub_p[i] <= 0]
    x_free = [i for i in range(numr) if lb_p[i] < 0 and ub_p[i] > 0]

    # Variables: z (num_z binary) | v (numr cont) | w (n_ineq_full cont >= 0)
    n_w = n_ineq_full
    n_total = num_z + numr + n_w
    v_offset = num_z
    w_offset = num_z + numr

    # Objective
    c_obj = knockable_cost + [0.0] * (numr + n_w)

    # Equality: K^T v + K^T A_ineq_full^T w = 0
    K_sparse = sparse.csr_matrix(K_np.T)
    KtAt = K_sparse @ A_ineq_full.T
    A_eq_milp = sparse.hstack([
        sparse.csr_matrix((n_kernel, num_z)),
        K_sparse,
        KtAt,
    ]).tocsr()
    b_eq_milp = [0.0] * n_kernel

    # Inequality: Farkas certificate + cost constraint
    farkas_row = sparse.hstack([
        sparse.csr_matrix((1, num_z)),
        sparse.csr_matrix((1, numr)),
        sparse.csr_matrix([b_ineq_full]),
    ]).tocsr()
    cost_row = sparse.lil_matrix((1, n_total))
    for i in range(num_z):
        cost_row[0, i] = knockable_cost[i]

    A_ineq_milp = sparse.vstack([farkas_row, cost_row.tocsr()]).tocsr()
    b_ineq_milp = [-1.0,
                   float(max_cost) if max_cost and not isinf(max_cost)
                   else float(np.sum(np.abs(knockable_cost)))]

    # Bounds: z binary, v free, w >= 0
    lb_milp = [0.0] * num_z + [-np.inf] * numr + [0.0] * n_w
    ub_milp = [1.0] * num_z + [np.inf] * numr + [np.inf] * n_w

    # Static sign bounds for NON-KNOCKABLE reactions
    knockable_set = set()
    for orig_id in knockable_orig:
        for idx in orig_to_split_idx[orig_id]:
            knockable_set.add(idx)
    for i in x_geq0:
        if i not in knockable_set:
            ub_milp[v_offset + i] = 0.0
    for i in x_leq0:
        if i not in knockable_set:
            lb_milp[v_offset + i] = 0.0
    for i in x_free:
        if i not in knockable_set:
            lb_milp[v_offset + i] = 0.0
            ub_milp[v_offset + i] = 0.0

    vtype = "B" * num_z + "C" * (numr + n_w)

    # Indicator constraints
    z_orig_idx_map = {orig_id: i for i, orig_id in enumerate(knockable_orig)}
    indic_constr = _make_indicators(
        num_z, n_total, numr, knockable_orig, orig_to_split_idx,
        z_orig_idx_map, v_offset, x_free, x_geq0, x_leq0,
        indicator_mode, do_split)

    milp = MILP_LP(c=c_obj, A_ineq=A_ineq_milp, b_ineq=b_ineq_milp,
                   A_eq=A_eq_milp, b_eq=b_eq_milp,
                   lb=lb_milp, ub=ub_milp, vtype=vtype,
                   indic_constr=indic_constr, solver=solver, seed=seed)
    milp._num_z = num_z
    milp._idx_z = list(range(num_z))
    milp._knockable_orig = knockable_orig
    milp._cost_vec = knockable_cost
    return milp, m


# ═══════════════════════════════════════════════════════════════════════
# Indicator construction
# ═══════════════════════════════════════════════════════════════════════

def _make_indicators(num_z, n_total, numr, knockable_orig, orig_to_split_idx,
                     z_orig_idx_map, v_offset,
                     x_free, x_geq0, x_leq0,
                     indicator_mode, do_split):
    """Create indicator constraints.

    Sign constraints on v are INDICATOR constraints (not static):
      z=0 (present), irrev (x>=0):  v_i <= 0   (sense 'L')
      z=0 (present), rev (x free):  v_i = 0    (sense 'E' for indic-eq)
                                 or v_i<=0 + -v_i<=0 (for indic-ineq)
      z=1 (KO): v_i free (no constraint)
    """
    x_free_set = set(x_free)
    x_geq0_set = set(x_geq0)
    x_leq0_set = set(x_leq0)

    binv_list = []
    indic_A_rows = []
    indic_b = []
    indic_sense = ""
    indic_val = []

    for orig_id in knockable_orig:
        z_idx = z_orig_idx_map[orig_id]
        split_idxs = orig_to_split_idx[orig_id]

        for rxn_idx in split_idxs:
            v_idx = v_offset + rxn_idx

            if do_split:
                # After split, all vars >= 0: z=0 -> v_i <= 0
                row = sparse.lil_matrix((1, n_total))
                row[0, v_idx] = 1.0
                indic_A_rows.append(row.tocsr())
                indic_b.append(0.0)
                indic_sense += "L"
                binv_list.append(z_idx)
                indic_val.append(0)

            elif indicator_mode == "indic-eq":
                row = sparse.lil_matrix((1, n_total))
                if rxn_idx in x_free_set:
                    row[0, v_idx] = 1.0
                    indic_sense += "E"  # rev: z=0 -> v=0
                elif rxn_idx in x_geq0_set:
                    row[0, v_idx] = 1.0
                    indic_sense += "L"  # irrev (x>=0): z=0 -> v<=0
                elif rxn_idx in x_leq0_set:
                    row[0, v_idx] = -1.0  # x<=0: z=0 -> v>=0 i.e. -v<=0
                    indic_sense += "L"
                else:
                    continue  # blocked (lb=0, ub=0): skip
                indic_A_rows.append(row.tocsr())
                indic_b.append(0.0)
                binv_list.append(z_idx)
                indic_val.append(0)

            elif indicator_mode == "indic-ineq":
                if rxn_idx in x_free_set:
                    # rev: z=0 -> v<=0 AND z=0 -> -v<=0 (equiv to v=0)
                    row1 = sparse.lil_matrix((1, n_total))
                    row1[0, v_idx] = 1.0
                    indic_A_rows.append(row1.tocsr())
                    indic_b.append(0.0)
                    indic_sense += "L"
                    binv_list.append(z_idx)
                    indic_val.append(0)

                    row2 = sparse.lil_matrix((1, n_total))
                    row2[0, v_idx] = -1.0
                    indic_A_rows.append(row2.tocsr())
                    indic_b.append(0.0)
                    indic_sense += "L"
                    binv_list.append(z_idx)
                    indic_val.append(0)
                elif rxn_idx in x_geq0_set:
                    # irrev (x>=0): z=0 -> v<=0
                    row = sparse.lil_matrix((1, n_total))
                    row[0, v_idx] = 1.0
                    indic_A_rows.append(row.tocsr())
                    indic_b.append(0.0)
                    indic_sense += "L"
                    binv_list.append(z_idx)
                    indic_val.append(0)
                elif rxn_idx in x_leq0_set:
                    # x<=0: z=0 -> -v<=0
                    row = sparse.lil_matrix((1, n_total))
                    row[0, v_idx] = -1.0
                    indic_A_rows.append(row.tocsr())
                    indic_b.append(0.0)
                    indic_sense += "L"
                    binv_list.append(z_idx)
                    indic_val.append(0)

    if not indic_A_rows:
        return None

    indic_A = sparse.vstack(indic_A_rows).tocsr()
    return IndicatorConstraints(binv_list, indic_A, indic_b, indic_sense, indic_val)


# ═══════════════════════════════════════════════════════════════════════
# Auxiliary-variable indicator transformation
# ═══════════════════════════════════════════════════════════════════════

def _auxvar_indicators(milp, solver, seed=None):
    """Replace complex indicator constraints with auxiliary-variable indicators.

    For each existing indicator  z = indicval  →  A·x <sense> b:
      - Add static equality:  A·x + s = b  (always active, s free)
      - Replace indicator with a simple single-variable constraint:
          'E':  z = indicval  →  s = 0
          'L':  z = indicval  →  -s ≤ 0   (i.e. s ≥ 0, so A·x ≤ b)

    This keeps SD's compact constraint structure but makes every indicator
    trivial (one z per one auxiliary variable), analogous to the FLB v-variable
    binding.  The complex constraints are always in the LP relaxation.
    """
    ic = milp.indic_constr
    if ic is None or not hasattr(ic, 'A') or ic.A is None or ic.A.shape[0] == 0:
        return milp

    n_ic = ic.A.shape[0]
    n_vars_old = milp.A_ineq.shape[1]
    n_vars_new = n_vars_old + n_ic

    # 1. Expand existing constraint matrices with zero columns for aux vars
    zero_ineq = sparse.csr_matrix((milp.A_ineq.shape[0], n_ic))
    A_ineq_new = sparse.hstack([milp.A_ineq, zero_ineq]).tocsr()

    zero_eq_old = sparse.csr_matrix((milp.A_eq.shape[0], n_ic))
    A_eq_old = sparse.hstack([milp.A_eq, zero_eq_old]).tocsr()

    # 2. Build new static equality rows:  A_row · x + s_i = b_i
    #    (for 'G' sense we'd use -s_i, but SD only produces 'L' and 'E')
    ic_A_csr = sparse.csr_matrix(ic.A)
    new_eq_rows = []
    for i in range(n_ic):
        # Start with the original indicator A row, expanded to new width
        row = sparse.lil_matrix((1, n_vars_new))
        start, end = ic_A_csr.indptr[i], ic_A_csr.indptr[i + 1]
        for idx in range(start, end):
            row[0, ic_A_csr.indices[idx]] = ic_A_csr.data[idx]
        # Add +1 for auxiliary variable s_i
        row[0, n_vars_old + i] = 1.0
        new_eq_rows.append(row.tocsr())

    new_eq_A = sparse.vstack(new_eq_rows).tocsr()
    A_eq_new = sparse.vstack([A_eq_old, new_eq_A]).tocsr()
    b_eq_new = list(milp.b_eq) + list(ic.b)

    # 3. Build simple indicator constraints on auxiliary variables
    new_ic_rows = []
    new_ic_sense = ""
    for i in range(n_ic):
        row = sparse.lil_matrix((1, n_vars_new))
        s_col = n_vars_old + i
        if ic.sense[i] == 'E':
            # z = indicval → s = 0
            row[0, s_col] = 1.0
            new_ic_sense += 'E'
        else:
            # 'L': z = indicval → s ≥ 0, encoded as -s ≤ 0
            row[0, s_col] = -1.0
            new_ic_sense += 'L'
        new_ic_rows.append(row.tocsr())

    new_ic_A = sparse.vstack(new_ic_rows).tocsr()
    new_ic = IndicatorConstraints(
        list(ic.binv), new_ic_A, [0.0] * n_ic, new_ic_sense, list(ic.indicval))

    # 4. Bounds for aux vars: free (constrained only by indicators)
    lb_new = list(milp.lb) + [-np.inf] * n_ic
    ub_new = list(milp.ub) + [np.inf] * n_ic
    c_new = list(milp.c) + [0.0] * n_ic
    vtype_new = milp.vtype + 'C' * n_ic

    # 5. Build new MILP
    new_milp = MILP_LP(c=c_new, A_ineq=A_ineq_new, b_ineq=list(milp.b_ineq),
                       A_eq=A_eq_new, b_eq=b_eq_new,
                       lb=lb_new, ub=ub_new, vtype=vtype_new,
                       indic_constr=new_ic, solver=solver, seed=seed)

    # Copy metadata
    new_milp._num_z = milp._num_z
    new_milp._idx_z = milp._idx_z
    new_milp._knockable_orig = milp._knockable_orig
    new_milp._cost_vec = milp._cost_vec
    return new_milp


# ═══════════════════════════════════════════════════════════════════════
# Enumeration loop
# ═══════════════════════════════════════════════════════════════════════

def enumerate_milp(milp, max_solutions=2000):
    """Enumerate via Gurobi populate + integer cuts."""
    if isinstance(milp, SDMILP):
        result = milp.enumerate(show_no_ki=True)
        return len(result.reaction_sd), result

    num_z = milp._num_z
    idx_z = milp._idx_z
    solutions = []
    n_total_vars = milp.A_ineq.shape[1]

    # Improve numerical stability
    if hasattr(milp, "backend") and hasattr(milp.backend, "setParam"):
        try:
            milp.backend.setParam("NumericFocus", 2)
        except Exception:
            pass

    while len(solutions) < max_solutions:
        try:
            xs, _, status = milp.populate(max_solutions - len(solutions))
        except Exception:
            try:
                x, _, st = milp.solve()
                if x is not None and not all(v != v for v in x[:3]):
                    xs = [x]
                    status = st
                else:
                    break
            except Exception:
                break

        if isinstance(status, str):
            if status not in ("optimal", "feasible"):
                break
        elif isinstance(status, int):
            if status not in (1, 11):
                break
        if not xs:
            break
        if isinstance(xs[0], (int, float)):
            xs = [xs]

        found_new = False
        for x in xs:
            z = tuple(int(round(x[i])) for i in idx_z)
            if z not in solutions:
                solutions.append(z)
                found_new = True
                nz_indices = [idx_z[i] for i in range(num_z) if z[i] == 1]
                if not nz_indices:
                    row = sparse.csr_matrix(
                        ([1.0] * num_z, ([0] * num_z, idx_z)),
                        shape=(1, n_total_vars))
                    milp.add_ineq_constraints(row, [-1.0])
                elif len(nz_indices) == 1:
                    milp.set_ub([[nz_indices[0], 0.0]])
                else:
                    row = sparse.csr_matrix(
                        ([1.0] * len(nz_indices),
                         ([0] * len(nz_indices), nz_indices)),
                        shape=(1, n_total_vars))
                    milp.add_ineq_constraints(row, [len(nz_indices) - 1.0])

        if not found_new:
            break

    return len(solutions), solutions


# ═══════════════════════════════════════════════════════════════════════
# Metrics extraction
# ═══════════════════════════════════════════════════════════════════════

def extract_metrics(milp):
    """Extract structure metrics from a MILP."""
    n_vars = milp.A_ineq.shape[1] if milp.A_ineq is not None else 0
    n_ineq = milp.A_ineq.shape[0] if milp.A_ineq is not None else 0
    n_eq = milp.A_eq.shape[0] if milp.A_eq is not None else 0
    n_bin = sum(1 for v in milp.vtype if v == "B")
    n_cont = n_vars - n_bin

    nnz = 0
    if milp.A_ineq is not None:
        nnz += milp.A_ineq.nnz
    if milp.A_eq is not None:
        nnz += milp.A_eq.nnz

    n_indic = 0
    if hasattr(milp, "indic_constr") and milp.indic_constr is not None:
        if hasattr(milp.indic_constr, "A"):
            n_indic = milp.indic_constr.A.shape[0]

    return {
        "vars": n_vars, "bin": n_bin, "cont": n_cont,
        "ineqs": n_ineq, "eqs": n_eq, "indic": n_indic, "nnz": nnz,
    }


# ═══════════════════════════════════════════════════════════════════════
# Runner + main
# ═══════════════════════════════════════════════════════════════════════

def run_one(test_spec, model_variants, sd_modules, kwargs_milp, solver="gurobi",
            seed=None):
    """Build formulation, extract metrics, solve, return results."""
    formulation, split, indicator, bounds = test_spec
    do_split = (split == "split")
    model = model_variants[bounds]
    label = f"{formulation:>3s} {split:>9s} {indicator:>9s} {bounds:>5s}"

    t0 = time.perf_counter()
    try:
        if formulation == "SD":
            milp, m_used = build_sd_current(model, sd_modules, kwargs_milp,
                                             do_split, solver, seed=seed,
                                             trim=True)
            if indicator == "aux-var":
                milp = _auxvar_indicators(milp, solver, seed=seed)
        elif formulation == "FLB":
            # aux-var is equivalent to indic-eq for FLB (already simple indicators)
            eff_indicator = "indic-eq" if indicator == "aux-var" else indicator
            milp, m_used = build_flb_explicit(model, sd_modules, kwargs_milp,
                                               eff_indicator, do_split, solver,
                                               seed=seed)
        elif formulation == "NB":
            eff_indicator = "indic-eq" if indicator == "aux-var" else indicator
            milp, m_used = build_nb(model, sd_modules, kwargs_milp,
                                     eff_indicator, do_split, solver, seed=seed)
        else:
            raise ValueError(f"Unknown formulation: {formulation}")
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"label": label, "error": str(e)}

    t_build = time.perf_counter() - t0
    metrics = extract_metrics(milp)

    t1 = time.perf_counter()
    try:
        n_sol, _ = enumerate_milp(milp)
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"label": label, **metrics,
                "t_build": t_build, "t_solve": 0, "n_solutions": -1,
                "error": str(e)}

    t_solve = time.perf_counter() - t1

    return {"label": label, **metrics,
            "t_build": t_build, "t_solve": t_solve,
            "n_solutions": n_sol, "error": None}


SEEDS = [42, 123, 7]
N_REPS = len(SEEDS)


def run_suite(label, model, sd_modules, kwargs_milp, expected, tests,
              solver="gurobi"):
    """Run a suite of formulation tests on one model, N_REPS times.

    Reports median solve time across seeds.
    Returns (n_pass, n_fail, n_err) based on correctness across ALL runs.
    """
    print(f"\n{'=' * 80}")
    print(f"  {label}")
    print(f"  {len(model.reactions)} rxns, {len(model.metabolites)} mets, "
          f"expected {expected} MCS, {N_REPS} seeds {SEEDS}")
    print(f"{'=' * 80}")

    # Pre-compute model variants for all bounds levels used in tests
    bounds_levels = sorted(set(t[3] for t in tests))
    print(f"  Preparing model variants: {bounds_levels} ...")
    model_variants = {}
    for bl in bounds_levels:
        if bl == "orig":
            model_variants["orig"] = model.copy()
        elif bl == "unbind":
            model_variants["unbind"] = fva_unbind(model.copy())
        elif bl == "fva":
            model_variants["fva"] = fva_tighten_bounds(model.copy())
        elif bl == "cap":
            model_variants["cap"] = cap_unbounded(model.copy())
    print()

    cols = ["Form", "Split", "Indicator", "Bnds",
            "Vars", "Bin", "Cont", "Ineqs", "Eqs", "Indic", "NNZ",
            "Build", "Solve (median/all)", "Sols", "Status"]
    widths = [4, 9, 9, 5,
              5, 4, 5, 5, 5, 5, 6,
              6, 24, 5, 6]

    def fmt_row(vals):
        parts = [f"{v:>{w}}" for v, w in zip(vals, widths)]
        return " | ".join([
            " ".join(parts[:4]),
            " ".join(parts[4:11]),
            " ".join(parts[11:]),
        ])

    header = fmt_row(cols)
    sep = "\u2500" * len(header)
    print(header)
    print(sep)

    all_ok = True
    n_pass = n_fail = n_err = 0

    for test_spec in tests:
        solve_times = []
        build_times = []
        last_metrics = None
        last_nsol = None
        error_msg = None

        for seed in SEEDS:
            r = run_one(test_spec, model_variants, sd_modules, kwargs_milp,
                        solver, seed=seed)
            if r.get("error"):
                error_msg = r["error"]
                n_err += 1
                break
            last_metrics = r
            last_nsol = r["n_solutions"]
            solve_times.append(r["t_solve"])
            build_times.append(r["t_build"])
            if last_nsol != expected:
                n_fail += 1
                break
        else:
            # All seeds passed
            n_pass += 1

        if error_msg:
            vals = list(test_spec) + ["?"] * 7 + ["?", "?", "?", "ERROR"]
            print(fmt_row([str(v) for v in vals]))
            print(f"    ERROR: {error_msg}")
        else:
            r = last_metrics
            med_solve = sorted(solve_times)[len(solve_times) // 2]
            med_build = sorted(build_times)[len(build_times) // 2]
            passed = (last_nsol == expected)
            status = "PASS" if passed else f"FAIL({expected})"
            solve_str = (f"{med_solve:.1f}" if len(solve_times) == 1
                         else f"{med_solve:.1f}({'/'.join(f'{t:.1f}' for t in solve_times)})")
            vals = list(test_spec) + [
                r["vars"], r["bin"], r["cont"],
                r["ineqs"], r["eqs"], r["indic"], r["nnz"],
                f"{med_build:.1f}", solve_str,
                last_nsol, status,
            ]
            print(fmt_row([str(v) for v in vals]))

    print(sep)
    total = n_pass + n_fail + n_err
    print(f"{n_pass} PASS, {n_fail} FAIL, {n_err} ERROR"
          f" out of {total} tests")
    return n_pass, n_fail, n_err


def main():
    solver = "gurobi"
    try:
        import gurobipy  # noqa: F401
    except ImportError:
        print("ERROR: Gurobi not available. This benchmark requires Gurobi.")
        sys.exit(1)

    print("\nMILP Formulation Variant Benchmark")
    print(f"  Solver: {solver}")

    total_pass, total_fail, total_err = 0, 0, 0

    # ── e_coli_core ──────────────────────────────────────────────────
    print("\nLoading e_coli_core ...")
    model_ec, sd_mod_ec, kw_ec = load_ecoli_core()
    # SD-current establishes expected count (353 for reaction-level KOs)
    p, f, e = run_suite(
        "e_coli_core (reaction-level KOs, max_cost=3)",
        model_ec, sd_mod_ec, kw_ec, expected=353, tests=TESTS, solver=solver)
    total_pass += p; total_fail += f; total_err += e

    # ── iMLcore (compressed snapshot) ────────────────────────────────
    print("\nLoading iMLcore snapshot ...")
    model_im, sd_mod_im, kw_im, expected_im = load_imlcore_snapshot()
    p, f, e = run_suite(
        "iMLcore compressed (gene-level KOs extended, max_cost=3)",
        model_im, sd_mod_im, kw_im, expected=expected_im, tests=TESTS,
        solver=solver)
    total_pass += p; total_fail += f; total_err += e

    # ── Summary ──────────────────────────────────────────────────────
    total = total_pass + total_fail + total_err
    print(f"\n{'=' * 80}")
    print(f"TOTAL: {total_pass} PASS, {total_fail} FAIL, {total_err} ERROR"
          f" out of {total} tests")
    print(f"{'=' * 80}")

    if total_fail > 0 or total_err > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
