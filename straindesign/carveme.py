#!/usr/bin/env python3
#
# Copyright 2024 Max Planck Insitute Magdeburg
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
"""Preprocessing for CarveMe modules: a flux state in which the whole annotated core runs.

A 'carveme' module demands that a reaction it keeps demonstrably carries flux. Stated as
``|v_r| >= t_r`` that is not linear, so the module fixes a direction per reaction and demands
``d_r * v_r >= t_r`` instead. Choosing those directions is the job of this module, and it is not
as simple as reading each reaction's own FVA range: individually feasible directions need not be
jointly consistent, and demanding all of them at once is then infeasible.

:func:`build_witness` sidesteps that by producing an actual flux state that carries the whole
core. It maximises each core reaction's flux in turn and sums the normalised solutions. The
feasible set is convex, so the sum is feasible; and it is non-zero wherever any summand was, so
one state carries every reaction that can carry anything, with signs that agree by construction.
"""

import logging
import numpy as np
from scipy import sparse
from typing import Dict, List, Tuple
from cobra.util import create_stoichiometric_matrix
from straindesign import MILP_LP
from straindesign.names import *

_INF = 1e4


def build_witness(model, core, constraints=None, solver=None, threads=None) -> Tuple[Dict, Dict, List]:
    """A single flux state in which every core reaction carries flux, and its directions.

    Args:
        model (cobra.Model):
            The model to search in, typically the universe a CarveMe module draws from.

        core (list of str):
            Reactions that have to carry flux.

        constraints (optional (list)):
            Additional constraints the state must satisfy, e.g. a growth requirement. Directions
            are only meaningful together with the conditions they were derived under.

        solver (optional (str)), threads (optional (int)):
            Solver name and thread count for the LPs.

    Returns:
        (Tuple[Dict, Dict, List]):

        Directions (+1/-1) and thresholds keyed by reaction id, and the core reactions for which
        no flux was achievable at all. Those cannot satisfy a must-run condition however the rest
        of the network is completed, so a module that demanded them would be infeasible.
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

    accumulated = np.zeros(len(reac_ids))
    unreachable = []
    for rid in core:
        found = None
        for sign in (1.0, -1.0):
            c = np.zeros(len(reac_ids))
            c[idx[rid]] = -sign  # MILP_LP minimises
            lp = MILP_LP(c=c.tolist(), A_ineq=A_ineq, b_ineq=list(b_ineq), A_eq=A_eq,
                         b_eq=list(b_eq), lb=lb.tolist(), ub=ub.tolist(), solver=solver,
                         milp_threads=threads)
            x, _, status = lp.solve()
            if status == OPTIMAL and x is not None and sign * x[idx[rid]] > 1e-9:
                found = np.array(x)
                break
        if found is None:
            unreachable.append(rid)
        else:
            accumulated += found / (np.abs(found).max() + 1e-12)

    directions, thresholds = {}, {}
    for rid in core:
        value = accumulated[idx[rid]]
        if abs(value) > 1e-12:
            directions[rid] = 1 if value > 0 else -1
            thresholds[rid] = abs(value)
    return directions, thresholds, unreachable
