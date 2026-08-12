#!/usr/bin/env python3
#
# Copyright 2024 Max Planck Insitute Magdeburg
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
"""Network completion: keep an annotated core, buy the cheapest additions that let it run.

Genome-scale reconstruction is usually posed the other way round -- take a universal network and
carve it down by annotation score -- which leaves the tool free to discard a reaction the genome
does support, and free to keep one that can never carry flux. Posed as a completion instead, the
two guarantees come from the formulation:

    z_r = 0                  =>  v_r = 0          a reaction that was not bought carries nothing
    z_r = 1, r in the core   =>  d_r * v_r >= t_r a reaction that WAS kept demonstrably runs
    minimise  sum_r cost_r * z_r                  annotated reactions priced below heterologous

Inclusion stays soft. Making the core mandatory instead removes the choice to omit a reaction
whose supporting chemistry is too expensive, and at genome scale that is simply infeasible.

The gating and the must-run condition are **indicator constraints**, never big-M. A binary that
rests fractionally between 0 and 1 -- which solvers permit within their integrality tolerance --
would otherwise licence M * tol units of flux through a reaction that reads as switched off.
StrainDesign's solver interfaces pin that tolerance (CPLEX to 0, Gurobi to 1e-9), so the leak
cannot occur here; it is a real defect in tools that use big-M with default tolerances.

Directions and thresholds
-------------------------
"The reaction runs" means ``|v_r| >= t_r``, which is not linear. Two ways out, and only one works:

* Split the reaction and demand ``v_fwd + v_rev >= t``. **Vacuous** -- the halves cycle against
  each other at zero net flux and satisfy it while the reaction does nothing. The split even
  creates the cycle.
* Fix a direction per reaction and demand ``d_r * v_r >= t_r``. Linear, and correct.

A direction cannot be read off each reaction's own FVA range: individually feasible directions
need not be jointly consistent, and demanding them together is then infeasible. Pass
``core_directions`` built from a single flux state in which the whole core carries flux (see
:func:`build_witness`), or omit it and let the MILP choose per reaction with a binary pair.
"""

import logging
import numpy as np
from scipy import sparse
from typing import Dict, List, Tuple
from cobra.util import create_stoichiometric_matrix
from straindesign import MILP_LP, IndicatorConstraints
from straindesign.names import *

_INF = 1e4


def build_witness(model, core, constraints=None, solver=None, threads=None):
    """A single flux state in which every core reaction carries flux, and its directions.

    For each core reaction, maximise its flux in one direction and then the other; normalise each
    solution and sum them. The feasible set is convex, so the sum is feasible too, and a generic
    combination is non-zero wherever any summand was -- so one state carries the whole core.

    Args:
        model (cobra.Model): the model to search in (typically the full universe).
        core (list of str): reactions that must carry flux.
        constraints (list): additional constraints, e.g. a growth requirement.
        solver (str): 'gurobi', 'cplex', 'scip' or 'glpk'.
        threads (int): solver threads.

    Returns:
        (Tuple[Dict, Dict, List]):

        Directions (+1/-1) and thresholds keyed by reaction id, and the core reactions for which
        no flux was achievable at all -- those can never satisfy a must-run condition and are
        reported rather than demanded.
    """
    from straindesign.parse_constr import lineqlist2mat, parse_constraints
    reac_ids = model.reactions.list_attr('id')
    constraints = parse_constraints(constraints, reac_ids) if constraints else None
    idx = {r: i for i, r in enumerate(reac_ids)}
    S = sparse.csr_matrix(create_stoichiometric_matrix(model))
    lb = np.clip(np.array(model.reactions.list_attr('lower_bound'), float), -_INF, None)
    ub = np.clip(np.array(model.reactions.list_attr('upper_bound'), float), None, _INF)
    if constraints:
        A_ineq, b_ineq, A_eq, b_eq = lineqlist2mat(constraints, reac_ids)
        A_eq = sparse.vstack((S, A_eq)).tocsr()
        b_eq = [0.0] * S.shape[0] + list(b_eq)
    else:
        A_ineq = sparse.csr_matrix((0, len(reac_ids)))
        b_ineq, A_eq, b_eq = [], S, [0.0] * S.shape[0]

    acc = np.zeros(len(reac_ids))
    unreachable = []
    for rid in core:
        got = None
        for sgn in (1.0, -1.0):
            c = np.zeros(len(reac_ids))
            c[idx[rid]] = -sgn                      # MILP_LP minimises
            lp = MILP_LP(c=c.tolist(), A_ineq=A_ineq, b_ineq=list(b_ineq), A_eq=A_eq,
                         b_eq=list(b_eq), lb=lb.tolist(), ub=ub.tolist(), solver=solver,
                         milp_threads=threads)
            x, _, status = lp.solve()
            if status == OPTIMAL and x is not None and sgn * x[idx[rid]] > 1e-9:
                got = np.array(x)
                break
        if got is None:
            unreachable.append(rid)
        else:
            acc += got / (np.abs(got).max() + 1e-12)

    directions, thresholds = {}, {}
    for rid in core:
        a = acc[idx[rid]]
        if abs(a) > 1e-12:
            directions[rid] = 1 if a > 0 else -1
            thresholds[rid] = abs(a)
    return directions, thresholds, unreachable


def compute_completion(model, sd_module, cost=None, solver=None, threads=None, time_limit=None,
                       extra_blocks=None, drop=None):
    """Solve one completion module and report which reactions to keep.

    Args:
        model (cobra.Model): the universe to draw from.
        sd_module (SDModule): a module of type 'complete'.
        cost (dict): cost per candidate reaction. Negative rewards keeping it. Reactions absent
            from this dict are not candidates and stay in the model unconditionally.
        solver (str), threads (int), time_limit (float): passed to the MILP.
        extra_blocks (list): additional constraint lists, each of which gets its own steady-state
            flux system sharing the same binaries -- one per growth medium, say. Gap-filling for
            several media therefore happens inside the same MILP rather than as a second pass.
        drop (set): candidates forced out. A reaction that cannot carry flux under the module's
            conditions belongs here: it has no must-run condition to satisfy, so a reward for
            keeping it would buy a reaction that is dead in the result.

    Returns:
        (Tuple[set, str, float]):

        The reactions to keep, the solver status, and the objective value.
    """
    from straindesign.parse_constr import lineqlist2mat, parse_constraints
    reac_ids = model.reactions.list_attr('id')
    idx = {r: i for i, r in enumerate(reac_ids)}
    nR = len(reac_ids)
    S = sparse.csr_matrix(create_stoichiometric_matrix(model))
    nM = S.shape[0]
    lb = np.clip(np.array(model.reactions.list_attr('lower_bound'), float), -_INF, None)
    ub = np.clip(np.array(model.reactions.list_attr('upper_bound'), float), None, _INF)

    core = [r for r in sd_module[CORE_REACTIONS] if r in idx]
    directions = dict(sd_module[CORE_DIRECTIONS] or {})
    thresholds = dict(sd_module[CORE_THRESHOLDS] or {})
    min_flux = sd_module[MIN_FLUX]
    free_dir = [r for r in core if r not in directions]
    cost = dict(cost or {})
    cands = [r for r in reac_ids if r in cost]

    blocks = [parse_constraints(b, reac_ids) if b else b
              for b in [sd_module[CONSTRAINTS]] + list(extra_blocks or [])]
    off = {i: i * nR for i in range(len(blocks))}
    base = len(blocks) * nR
    z_at = {r: base + i for i, r in enumerate(cands)}
    base += len(cands)
    zf_at = {r: base + i for i, r in enumerate(free_dir)}
    base += len(free_dir)
    zr_at = {r: base + i for i, r in enumerate(free_dir)}
    base += len(free_dir)
    mu_at, nVar = base, base + (nM if sd_module[LOOPLESS] else 0)

    vlb, vub = np.zeros(nVar), np.zeros(nVar)
    eq_rows, eq_b, ineq_rows, ineq_b = [], [], [], []
    for i, cons in enumerate(blocks):
        blo, bhi = lb.copy(), ub.copy()
        for r in cands:                       # a candidate's bounds must admit zero flux
            j = idx[r]
            blo[j], bhi[j] = min(blo[j], 0.0), max(bhi[j], 0.0)
        vlb[off[i]:off[i] + nR], vub[off[i]:off[i] + nR] = blo, bhi
        pad_l = sparse.csr_matrix((nM, off[i]))
        pad_r = sparse.csr_matrix((nM, nVar - off[i] - nR))
        eq_rows.append(sparse.hstack((pad_l, S, pad_r)).tocsr())
        eq_b += [0.0] * nM
        if cons:
            Ai, bi, Ae, be = lineqlist2mat(cons, reac_ids)
            for A_, b_, rows_, bs_ in ((Ai, bi, ineq_rows, ineq_b), (Ae, be, eq_rows, eq_b)):
                if A_.shape[0]:
                    rows_.append(sparse.hstack((sparse.csr_matrix((A_.shape[0], off[i])), A_,
                                                sparse.csr_matrix((A_.shape[0],
                                                                   nVar - off[i] - nR)))).tocsr())
                    bs_ += list(b_)
    vlb[len(blocks) * nR:mu_at], vub[len(blocks) * nR:mu_at] = 0.0, 1.0
    for r in (drop or ()):
        if r in z_at:
            vub[z_at[r]] = 0.0
    if sd_module[LOOPLESS]:
        vlb[mu_at:], vub[mu_at:] = -1e3, 1e3
    vtype = ''.join('B' if len(blocks) * nR <= i < mu_at else 'C' for i in range(nVar))

    if free_dir:                              # a kept reaction runs in exactly one direction
        R_ = sparse.lil_matrix((len(free_dir), nVar))
        for k, r in enumerate(free_dir):
            R_[k, zf_at[r]] = R_[k, zr_at[r]] = 1.0
            R_[k, z_at[r]] = -1.0
        eq_rows.append(R_.tocsr())
        eq_b += [0.0] * len(free_dir)

    Scsc = S.tocsc()
    ic_rows, binv, ic_b, sense, indval = [], [], [], [], []

    def _ind(bvar, val, coeffs, rhs, sns):
        row = sparse.lil_matrix((1, nVar))
        for j, c in coeffs:
            row[0, j] = c
        ic_rows.append(row.tocsr())
        binv.append(bvar)
        indval.append(val)
        ic_b.append(rhs)
        sense.append(sns)

    coreset = set(core)
    for r in cands:
        for i in range(len(blocks)):
            _ind(z_at[r], 0, [(off[i] + idx[r], 1.0)], 0.0, 'E')
        if r not in coreset:
            continue
        t = thresholds.get(r, 1.0) * min_flux
        col = Scsc[:, idx[r]]
        gcoef = [(mu_at + int(i), float(c)) for i, c in zip(col.indices, col.data)]
        if r in zf_at:                        # direction chosen by the MILP
            _ind(zf_at[r], 1, [(idx[r], -1.0)], -t, 'L')
            _ind(zr_at[r], 1, [(idx[r], 1.0)], -t, 'L')
            if sd_module[LOOPLESS] and gcoef:
                _ind(zf_at[r], 1, gcoef, -1.0, 'L')
                _ind(zr_at[r], 1, [(j, -c) for j, c in gcoef], -1.0, 'L')
        else:
            d = float(directions[r])
            _ind(z_at[r], 1, [(idx[r], -d)], -t, 'L')
            if sd_module[LOOPLESS] and gcoef:
                _ind(z_at[r], 1, [(j, d * c) for j, c in gcoef], -1.0, 'L')

    c_obj = np.zeros(nVar)
    for r in cands:
        c_obj[z_at[r]] = cost[r]
    logging.info('  Completion MILP: %d variables (%d binary), %d indicator constraints.' %
                 (nVar, len(cands) + 2 * len(free_dir), len(binv)))

    milp = MILP_LP(c=c_obj.tolist(),
                   A_ineq=sparse.vstack(ineq_rows).tocsr() if ineq_rows
                   else sparse.csr_matrix((0, nVar)),
                   b_ineq=ineq_b,
                   A_eq=sparse.vstack(eq_rows).tocsr(), b_eq=eq_b,
                   lb=vlb.tolist(), ub=vub.tolist(), vtype=vtype,
                   indic_constr=IndicatorConstraints(np.array(binv),
                                                     sparse.vstack(ic_rows).tocsr(), ic_b,
                                                     ''.join(sense), np.array(indval)),
                   solver=solver, milp_threads=threads)
    if time_limit is not None:
        milp.set_time_limit(time_limit)
    x, obj, status = milp.solve()
    if x is None:
        return set(), status, float('nan')
    x = np.array(x)
    keep = {r for r in cands if x[z_at[r]] > 0.5}
    return keep, status, obj
