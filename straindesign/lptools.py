#!/usr/bin/env python3
#
# Copyright 2022 Max Planck Insitute Magdeburg
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
"""A collection of functions for the LP-based analysis of metabolic networks"""

from cobra.core import Solution
from cobra.util import create_stoichiometric_matrix
from cobra import Configuration
from scipy import sparse
from scipy.spatial import ConvexHull
from math import log2
from straindesign import MILP_LP, parse_constraints, parse_linexpr, lineqlist2mat, linexpr2dict, \
                         linexprdict2mat, SDPool, IndicatorConstraints, avail_solvers
from re import search
from straindesign.names import *
from typing import Dict, Tuple
from pandas import DataFrame
from numpy import floor, sign, mod, nan, isnan, unique, inf, isinf, full, linspace, \
                  prod, array, mean, flip, ceil, floor, arctan2
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import matplotlib.pyplot as plt
from matplotlib import use as set_matplotlib_backend
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import logging

from straindesign.parse_constr import linexpr2mat, linexprdict2str


def select_solver(solver=None, model=None) -> str:
    """Select a solver for subsequent MILP/LP computations
    
    This function will determine the solver to be used for subsequend MILP/LP computations. If no
    argument is provided, this function will try to determine the currently selected solver from the
    COBRA configuration. If unavailable, the solver will be inferred from the packages available at
    package initialization and one of the solvers will be picked and retured in the prioritized 
    order: 'glpk', 'cplex', 'gurobi', 'scip'
    One may provide a solver or a model manually. This function then checks if the selected solver 
    is available, or else, if the solver indicated in the model is available. If yes, this function 
    returns the name of the solver as a str. If both arguments are specified, the function prefers
    'solver' over 'model'.
    
    Example:
        solver = select_solver('cplex')
    
    Args:
        solver (optional (str)):
        
            A user preferred solver, that should be checked for availability: 'glpk', 'cplex',
            'gurobi' or 'scip'.
            
        model (optional (cobra.Model)):
        
            A metabolic model that is an instance of the cobra.Model class. The function will try to
            dertermine the selected solver by accessing the field model.solver.
            
    Returns:
        (str):
        
            The selected solver name as a str (one of the following: 'glpk', 'cplex', 'gurobi', 'scip').
            
    """
    # first try to use selected solver
    if solver:
        if solver in avail_solvers:
            return solver
        else:
            logging.warning('Selected solver ' + solver + ' not available. Using ' + list(avail_solvers)[0] + " instead.")
    try:
        # if no solver was defined, use solver specified in model
        if hasattr(model, 'solver') and hasattr(model.solver, 'interface'):
            solver = search('(' + '|'.join(avail_solvers) + ')', model.solver.interface.__name__)
            if solver is not None:
                return solver[0]
            else:
                logging.warning('Solver specified in model (' + model.solver.interface.__name__ + ') unavailable')
        # if no solver specified in model, use solver from cobra configuration
        cobra_conf = Configuration()
        if hasattr(cobra_conf, 'solver'):
            solver = search('(' + '|'.join(avail_solvers) + ')', cobra_conf.solver.__name__)
            if solver is not None:
                return solver[0]
            else:
                logging.warning('Solver specified in cobra config (' + cobra_conf.solver.__name__ + ') unavailable')
    except:
        pass
    # if no solver is specified in cobra, fall back to list of available solvers and return the
    # first one available.
    return list(avail_solvers)[0]


def idx2c(i, prev) -> 'list':
    """Helper function for parallel FVA
    
    Builds the objective function for minimizing or maximizing the flux through the reaction
    with the index floor(i / 2). If i is even, there is a maximization.
    
    Args:
        i (float):
            An index between 0 and 2*num_reacs.
        prev (optional (str)):
            Index of the previously optimized reaction.
    Returns:
        (list):
            An optimization vector.
    """
    col = int(floor(i / 2))
    sig = sign(mod(i, 2) - 0.5)
    C = [[col, sig], [prev, 0.0]]
    C_idx = [C[i][0] for i in range(len(C))]
    C_idx = unique([C_idx.index(C_idx[i]) for i in range(len(C_idx))])
    C = [C[i] for i in C_idx]
    return C


def _fva_worker_cleanup():
    """Dispose the global LP and solver environment on worker exit."""
    global lp_glob
    try:
        if lp_glob is not None and hasattr(lp_glob, 'solver'):
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                if lp_glob.solver == 'gurobi':
                    lp_glob.backend.dispose()
                    import gurobipy as gp
                    gp.disposeDefaultEnv()
                elif lp_glob.solver == 'cplex':
                    lp_glob.backend.end()
        lp_glob = None
    except Exception:
        pass


def fva_worker_init(A_ineq, b_ineq, A_eq, b_eq, lb, ub, solver):
    """Helper function for parallel FVA

    Initialize the LP that will be solved iteratively. Is executed on workers, not on main thread.

    Args:
        A_ineq, b_ineq, A_eq, b_eq, lb, ub:
            The LP.
        solver (str):
            Solver to be used.
    """
    global lp_glob
    # redirect output to empty stream. Perhaps avoids some multithreading issues
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        lp_glob = MILP_LP(A_ineq=A_ineq, b_ineq=b_ineq, A_eq=A_eq, b_eq=b_eq, lb=lb, ub=ub, solver=solver)
        if lp_glob.solver == 'cplex':
            lp_glob.backend.parameters.threads.set(1)
            #lp_glob.backend.parameters.lpmethod.set(1)
        elif lp_glob.solver == 'gurobi':
            lp_glob.backend.params.Threads = 1
        lp_glob.prev = 0
    # Register cleanup to properly release solver resources on worker exit
    if solver in ('gurobi', 'cplex'):
        import atexit
        atexit.register(_fva_worker_cleanup)


def fva_worker_compute(i) -> Tuple[int, float]:
    """Helper function for parallel FVA
    
    Run a single LP as a step of FVA. Is executed on workers, not on main thread.
    
    Args:
        i (int):
            Index of the computation step.
    """
    global lp_glob
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        C = idx2c(i, lp_glob.prev)
        if lp_glob.solver in ['cplex', 'gurobi']:
            lp_glob.backend.set_objective_idx(C)
            min_cx = lp_glob.backend.slim_solve()
        else:
            lp_glob.set_objective_idx(C)
            min_cx = lp_glob.slim_solve()
        lp_glob.prev = C[0][0]
        return i, min_cx


# GLPK needs a workaround, because problems cannot be solved in a different thread
# which apparently happens with the multiprocess


def fva_worker_init_glpk(A_ineq, b_ineq, A_eq, b_eq, lb, ub):
    """Helper function for parallel FVA
    
    Initialize the LP for GLPK that will be solved iteratively. Is executed on workers, not on main thread.
    
    Args:
        A_ineq, b_ineq, A_eq, b_eq, lb, ub:
            The LP.
    """
    global lp_glob
    lp_glob = {}
    lp_glob['A_ineq'] = A_ineq
    lp_glob['b_ineq'] = b_ineq
    lp_glob['A_eq'] = A_eq
    lp_glob['b_eq'] = b_eq
    lp_glob['lb'] = lb
    lp_glob['ub'] = ub


def fva_worker_compute_glpk(i) -> Tuple[int, float]:
    """Helper function for parallel FVA
    
    Run a single LP for GLPK as a step of FVA. Is executed on workers, not on main thread.
    
    Args:
        i (int):
            Index of the computation step.
    """
    global lp_glob
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        lp_i = MILP_LP(A_ineq=lp_glob['A_ineq'],
                       b_ineq=lp_glob['b_ineq'],
                       A_eq=lp_glob['A_eq'],
                       b_eq=lp_glob['b_eq'],
                       lb=lp_glob['lb'],
                       ub=lp_glob['ub'],
                       solver=GLPK)
        col = int(floor(i / 2))
        sig = sign(mod(i, 2) - 0.5)
        lp_i.set_objective_idx([[col, sig]])
        min_cx = lp_i.slim_solve()
    return i, min_cx


def fva(model, **kwargs) -> DataFrame:
    """Flux Variability Analysis (FVA)

    Flux Variability Analysis determines the global flux ranges of reactions by minimizing and
    maximizing the flux through all reactions of a given metabolic network. This FVA function
    additionally allows the user to narrow down the flux states with additional constraints.

    Uses an accelerated two-phase approach: global scan LPs with dual simplex
    warm-start resolve ~50% of bounds cheaply, then individual LPs for the rest.
    Large models (>= 200 reactions) are automatically compressed via coupled
    reaction lumping.  Multiprocessing is used when >= 1000 reactions and
    cobra.Configuration().processes > 1.

    Example:
        flux_ranges = fva(model, constraints='EX_o2_e=0', solver='gurobi')

    Args:
        model (cobra.Model):
            A metabolic model that is an instance of the cobra.Model class.

        solver (optional (str)):
            The solver that should be used for FVA.

        constraints (optional (str) or (list of str) or (list of [dict,str,float])): (Default: '')
            List of *linear* constraints to be applied on top of the model: signs + or -, scalar
            factors for reaction rates, inclusive (in)equalities and a float value on the right hand
            side. The parsing of the constraints input allows for some flexibility. Correct (and
            identical) inputs are, for instance:
            constraints='-EX_o2_e <= 5, ATPM = 20' or
            constraints=['-EX_o2_e <= 5', 'ATPM = 20'] or
            constraints=[[{'EX_o2_e':-1},'<=',5], [{'ATPM':1},'=',20]]

    Returns:
        (pandas.DataFrame):
            A data frame containing the minimum and maximum attainable flux rates for all reactions.
    """
    from straindesign.speedy_fva import speedy_fva
    return speedy_fva(model, **kwargs)


def fva_legacy(model, **kwargs) -> DataFrame:
    """Legacy FVA implementation (brute-force 2*n LPs, no scan/compression).

    Kept as fallback for debugging. Use fva() for production.
    """
    reaction_ids = model.reactions.list_attr("id")
    numr = len(model.reactions)

    if CONSTRAINTS in kwargs and kwargs[CONSTRAINTS]:
        from straindesign.networktools import resolve_gene_constraints
        kwargs[CONSTRAINTS] = resolve_gene_constraints(model, kwargs[CONSTRAINTS])
        kwargs[CONSTRAINTS] = parse_constraints(kwargs[CONSTRAINTS], reaction_ids)
        A_ineq, b_ineq, A_eq, b_eq = lineqlist2mat(kwargs[CONSTRAINTS], reaction_ids)

    if SOLVER not in kwargs:
        kwargs[SOLVER] = None
    solver = select_solver(kwargs[SOLVER], model)

    A_eq_base = sparse.csr_matrix(create_stoichiometric_matrix(model))
    b_eq_base = [0] * len(model.metabolites)
    if 'A_eq' in locals():
        A_eq = sparse.vstack((A_eq_base, A_eq))
        b_eq = b_eq_base + b_eq
    else:
        A_eq = A_eq_base
        b_eq = b_eq_base
    if 'A_ineq' not in locals():
        A_ineq = sparse.csr_matrix((0, numr))
        b_ineq = []
    lb = [v.lower_bound for v in model.reactions]
    ub = [v.upper_bound for v in model.reactions]

    lp = MILP_LP(A_ineq=A_ineq, b_ineq=b_ineq, A_eq=A_eq, b_eq=b_eq, lb=lb, ub=ub, solver=solver)
    _, _, status = lp.solve()
    if status not in [OPTIMAL, UNBOUNDED]:
        logging.error('FVA problem not feasible.')
        return DataFrame(
            {"minimum": [nan] * numr, "maximum": [nan] * numr},
            index=reaction_ids,
        )

    processes = Configuration().processes
    processes = min(processes, numr)
    x = [nan] * 2 * numr

    if processes > 1 and numr > 300:
        with SDPool(processes, initializer=fva_worker_init,
                    initargs=(A_ineq, b_ineq, A_eq, b_eq, lb, ub, solver)) as pool:
            chunk_size = len(reaction_ids) // processes
            for i, value in pool.imap_unordered(fva_worker_compute, range(2 * numr),
                                                chunksize=chunk_size):
                x[i] = value
    elif processes > 1 and numr > 500 and solver == GLPK:
        with SDPool(processes, initializer=fva_worker_init_glpk,
                    initargs=(A_ineq, b_ineq, A_eq, b_eq, lb, ub)) as pool:
            chunk_size = len(reaction_ids) // processes
            for i, value in pool.imap_unordered(fva_worker_compute_glpk, range(2 * numr),
                                                chunksize=chunk_size):
                x[i] = value
    else:
        fva_worker_init(A_ineq, b_ineq, A_eq, b_eq, lb, ub, solver)
        for i in range(2 * numr):
            _, x[i] = fva_worker_compute(i)

    # NaN retry
    nan_remaining = [i for i in range(2 * numr) if isnan(x[i])]
    if nan_remaining:
        logging.warning(f'FVA: {len(nan_remaining)}/{2*numr} LP solves returned NaN, re-solving.')
        _BATCH = 50
        while nan_remaining:
            lp_retry = MILP_LP(A_ineq=A_ineq, b_ineq=b_ineq, A_eq=A_eq, b_eq=b_eq,
                               lb=lb, ub=ub, solver=solver)
            prev_retry = 0
            for i in nan_remaining[:_BATCH]:
                C = idx2c(i, prev_retry)
                if solver in ('cplex', 'gurobi'):
                    lp_retry.backend.set_objective_idx(C)
                    x[i] = lp_retry.backend.slim_solve()
                else:
                    lp_retry.set_objective_idx(C)
                    x[i] = lp_retry.slim_solve()
                prev_retry = C[0][0]
            old_count = len(nan_remaining)
            nan_remaining = [i for i in nan_remaining if isnan(x[i])]
            if len(nan_remaining) == old_count:
                break
        if nan_remaining:
            for i in list(nan_remaining):
                col = int(floor(i / 2))
                sig = sign(mod(i, 2) - 0.5)
                c_vec = [0.0] * numr
                c_vec[col] = sig
                lp_last = MILP_LP(c=c_vec, A_ineq=A_ineq, b_ineq=b_ineq, A_eq=A_eq, b_eq=b_eq,
                                  lb=lb, ub=ub, solver=solver)
                x[i] = lp_last.slim_solve()
            nan_remaining = [i for i in nan_remaining if isnan(x[i])]
        if nan_remaining:
            logging.warning(f'FVA: {len(nan_remaining)} LP solves still NaN after all retries.')

    x = [v if abs(v) >= 1e-11 else 0.0 for v in x]
    return DataFrame(
        {"minimum": [x[i] for i in range(1, 2 * numr, 2)],
         "maximum": [-x[i] for i in range(0, 2 * numr, 2)]},
        index=reaction_ids,
    )


def remove_redundant_bounds(model, **kwargs) -> DataFrame:
    """Remove non-binding bounds from a model using FVA.

    Runs FVA and relaxes bounds that never bind at steady state:
    - If fva_min > lb + tol: set lb = -inf  (lower bound is not binding)
    - If fva_max < ub - tol: set ub = +inf  (upper bound is not binding)

    Modifies the model IN-PLACE. Returns the FVA DataFrame.

    Args:
        model (cobra.Model):
            A metabolic model. Modified in-place.

        solver (optional (str)):
            Solver for FVA.

        constraints (optional):
            Constraints passed through to fva().

        compress (optional (bool)):
            Compress before FVA (passed through).

        threads (optional (int)):
            Parallel threads for FVA (passed through).

        tol (optional (float)): (Default: 1e-6)
            Tolerance for considering a bound as binding.

    Returns:
        (pandas.DataFrame):
            FVA results with 'minimum' and 'maximum' columns.
    """
    tol = kwargs.pop('tol', 1e-6)
    fva_result = fva(model, **kwargs)

    for rxn in model.reactions:
        fva_min = fva_result.loc[rxn.id, 'minimum']
        fva_max = fva_result.loc[rxn.id, 'maximum']
        if fva_min > rxn.lower_bound + tol:
            rxn.lower_bound = -float('inf')
        if fva_max < rxn.upper_bound - tol:
            rxn.upper_bound = float('inf')

    return fva_result


def fba(model, **kwargs) -> Solution:
    """Flux Balance Analysis (FBA), parsimonius Flux Balance Analysis (pFBA),
    
    Flux Balance Analysis optimizes a *linear objective function* in a space of steady-state
    flux vectors given by a constraint-based metabolic model. FVA is often used to determine
    the (stoichiometrically) maximal possible growth rate, or flux rate towards a particular
    product. This FBA function allows to us a custom objective function and sense and 
    allows the user to narrow down the flux states with additional constraints. In addition, 
    one may use different types of parsimonious FBAs to either reduce the total sum of fluxes
    or the total number of active reactions after the primary objective is optimized.    
    
    Example:
        optim = fba(model, constraints='EX_o2_e=0', solver='gurobi', pfba=1)
    
    Args:
        model (cobra.Model):
            A metabolic model that is an instance of the cobra.Model class. If no custom objective
            function is provided, the model's objective function is retrieved from the fields
            model.reactions[i].objective_coefficient.
            
        solver (optional (str)):
            The solver that should be used for FBA.
            
        constraints (optional (str) or (list of str) or (list of [dict,str,float])): (Default: '')
            List of *linear* constraints to be applied on top of the model: signs + or -, scalar 
            factors for reaction rates, inclusive (in)equalities and a float value on the right hand 
            side. The parsing of the constraints input allows for some flexibility. Correct (and 
            identical) inputs are, for instance: 
            constraints='-EX_o2_e <= 5, ATPM = 20' or
            constraints=['-EX_o2_e <= 5', 'ATPM = 20'] or
            constraints=[[{'EX_o2_e':-1},'<=',5], [{'ATPM':1},'=',20]]
            
        obj (optional (str) or (dict)):
            As a custom objective function, any linear expression can be used, either provided as a 
            single string or as a dict. Correct (and identical) inputs are, for instance:
            inner_objective='BIOMASS_Ecoli_core_w_GAM'
            inner_objective={'BIOMASS_Ecoli_core_w_GAM': 1}
            
        obj_sense (optional (str)): (Default: 'maximize')
            The optimization direction can be set either to 'maximize' (or 'max') or 'minimize' (or 'min').
            
        pfba (optional (int)): (Default: 0)
            The level of parsimonious FBA that should be applied. 0: no pFBA, only optmize the primary 
            objective, 1: minimize sum of fluxes after the primary objective is optimized, 2: minimize the
            number of active reactions after the primary objective is optimized.
            
    Returns:
        (cobra.core.Solution):
            A solution object that contains the objective value, an optimal flux vector and the optmization
            status.
    """
    from straindesign.networktools import resolve_gene_constraints
    reaction_ids = model.reactions.list_attr("id")

    if CONSTRAINTS in kwargs:
        kwargs[CONSTRAINTS] = resolve_gene_constraints(model, kwargs[CONSTRAINTS])
        kwargs[CONSTRAINTS] = parse_constraints(kwargs[CONSTRAINTS], reaction_ids)
        A_ineq, b_ineq, A_eq, b_eq = lineqlist2mat(kwargs[CONSTRAINTS], reaction_ids)
    else:
        kwargs[CONSTRAINTS] = []

    if 'obj' in kwargs and kwargs['obj'] is not None:
        if type(kwargs['obj']) is str:
            kwargs['obj'] = linexpr2dict(kwargs['obj'], reaction_ids)
        c = linexprdict2mat(kwargs['obj'], reaction_ids).toarray()[0].tolist()
    else:
        c = [i.objective_coefficient for i in model.reactions]

    if ('obj_sense' not in kwargs and model.objective_direction == 'max') or \
       ('obj_sense' in kwargs and kwargs['obj_sense'] not in ['min','minimize']):
        obj_sense = 'maximize'
        c = [-i for i in c]
    else:
        obj_sense = 'minimize'

    if 'pfba' in kwargs:
        pfba = kwargs['pfba']
    else:
        pfba = False

    if SOLVER not in kwargs:
        kwargs[SOLVER] = None
    solver = select_solver(kwargs[SOLVER], model)

    # prepare vectors and matrices
    A_eq_base = create_stoichiometric_matrix(model)
    A_eq_base = sparse.csr_matrix(A_eq_base)
    b_eq_base = [0.0] * len(model.metabolites)
    if 'A_eq' in locals():
        A_eq = sparse.vstack((A_eq_base, A_eq))
        b_eq = b_eq_base + b_eq
    else:
        A_eq = A_eq_base
        b_eq = b_eq_base
    if 'A_ineq' not in locals():
        A_ineq = sparse.csr_matrix((0, len(model.reactions)))
        b_ineq = []
    lb = [v.lower_bound for v in model.reactions]
    ub = [v.upper_bound for v in model.reactions]

    # build LP
    fba_prob = MILP_LP(c=c, A_ineq=A_ineq, b_ineq=b_ineq, A_eq=A_eq, b_eq=b_eq, lb=lb, ub=ub, solver=solver)

    x, opt_cx, status = fba_prob.solve()
    if obj_sense == 'minimize':
        opt_cx = -opt_cx
    if status == UNBOUNDED:
        num_prob = MILP_LP(c=[-v for v in c], A_ineq=A_ineq, b_ineq=b_ineq, A_eq=A_eq, b_eq=b_eq, lb=lb, ub=ub, solver=solver)
        min_cx = num_prob.slim_solve()
        if min_cx <= 0 or isnan(min_cx):
            num_prob.add_eq_constraints(c, [-1.0])
        else:
            num_prob.add_eq_constraints(c, min_cx)
        x, _, _ = num_prob.solve()
    elif status not in [OPTIMAL, UNBOUNDED]:
        status = INFEASIBLE
    if pfba and status == OPTIMAL:  # for pfba, split all reversible reactions and minimize the total flux
        numr = len(c)
        if pfba == 2:  # pfba mode 2 minimizes the number of active reactions
            # fix optimal flux and do fva to get essential reactions and speed up minimization
            kwargs_fva = kwargs.copy()
            kwargs_fva[CONSTRAINTS].append([{reaction_ids[i]: c[i] for i in range(numr) if c[i] != 0}, '=', opt_cx])
            fva_sol = fva(model, **kwargs_fva)

            A_ineq_pfba2 = A_ineq.copy()
            A_ineq_pfba2.resize(A_ineq.shape[0], 2 * numr)
            A_eq_pfba2 = A_eq.copy()
            A_eq_pfba2.resize(A_eq.shape[0], 2 * numr)
            lb_pfba2 = lb + [0.0] * numr
            # only set non-essential reactions as knockable
            ub_pfba2 = ub + [0.0 if prod(sign(lim)) > 0 else 1.0 for _, lim in fva_sol.iterrows()]
            c_pfba2 = [0.0] * numr + [-1.0] * numr
            A_ic = sparse.csr_matrix(([1] * numr, ([j for j in range(numr)], [j for j in range(numr)])), [numr, 2 * numr])
            ic = IndicatorConstraints([numr + j for j in range(numr)], A_ic, [0] * numr, 'E' * numr, [1.0] * numr)
            pfba2_prob = MILP_LP(c=c_pfba2,
                                 A_ineq=A_ineq_pfba2,
                                 b_ineq=b_ineq,
                                 A_eq=A_eq_pfba2,
                                 b_eq=b_eq,
                                 lb=lb_pfba2,
                                 ub=ub_pfba2,
                                 indic_constr=ic,
                                 vtype='C' * numr + 'B' * numr,
                                 solver=solver)
            pfba2_prob.add_eq_constraints(c + [0] * numr, [opt_cx])
            y, _, _ = pfba2_prob.solve()
            zero_flux = [i for i, j in enumerate(range(numr, 2 * numr)) if y[j]]
            lb = [l if i not in zero_flux else 0.0 for i, l in enumerate(lb)]
            ub = [u if i not in zero_flux else 0.0 for i, u in enumerate(ub)]
        A_ineq_pfba = sparse.hstack((A_ineq, -A_ineq))
        A_eq_pfba = sparse.hstack((A_eq, -A_eq))
        lb_pfba = [max((0, l)) for l in lb] + [max((0, -u)) for u in ub]
        ub_pfba = [max((0, u)) for u in ub] + [max((0, -l)) for l in lb]
        c_pfba = c + [-v for v in c]
        pfba_prob = MILP_LP(c=[1.0] * 2 * numr,
                            A_ineq=A_ineq_pfba,
                            b_ineq=b_ineq,
                            A_eq=A_eq_pfba,
                            b_eq=b_eq,
                            lb=lb_pfba,
                            ub=ub_pfba,
                            solver=solver)
        pfba_prob.add_eq_constraints(c_pfba, [opt_cx])
        x, _, _ = pfba_prob.solve()
        x = [x[i] - x[j] for i, j in enumerate(range(numr, 2 * numr))]

    x = [v if abs(v) >= 1e-11 else 0.0 for v in x]  # cut off for very small absolute values
    fluxes = {reaction_ids[i]: x[i] for i in range(len(x))}
    sol = Solution(objective_value=-opt_cx, status=status, fluxes=fluxes)
    return sol


def yopt(model, **kwargs) -> Solution:
    """Yield optmization (YOpt)
    
    Yield optimization optimizes a *fractional objective function* in a space of steady-state
    flux vectors given by a constraint-based metabolic model. Yield optimization employs linear
    fractional programming, and is often utilized to determine the (stoichiometrically) maximal 
    possible product yield, that is, the fraction between the product exchange and the substrate
    uptake flux. This function requires a custom fractional objective function specified by a 
    *linear* numerator and denominator terms. Coefficients in the linear numerator or denominator
    expression can be used to optimize for carbon recovery, for instance: 
    objective: (3*pyruvate_ex)/(2*ac_up+6*glc_up)
    The user may also specify the optimization sense. In addition, additional constraints can be 
    specified to narrow down the flux space.
    
    Yield optimization can fail because of several reasons. Here is how the function reacts:
    
    1. The model is infeasible:
        The function returns infeasible with no flux vector
    2. The denominator is fixed to zero:
        The function returns infeasible with no flux vector
    3. The numerator is unbounded:
        The function returns unbounded with no flux vector
    4. The denominator can become zero:
        The function returns unbounded, and a flux vector is computed by fixing the
        the denominator
        
    Example:
        optim = yopt(model, obj_num='2 EX_etoh_e', obj_den='-6 EX_glc__D_e', constraints='EX_o2_e=0')
    
    Args:
        model (cobra.Model):
            A metabolic model that is an instance of the cobra.Model class.
            
        obj_num ((str) or (dict)):
            The numerator of the fractional objective function, provided as a linear expression,
            either as a character string or as a dict. E.g.: obj_num='EX_prod_e' or 
            obj_num={'EX_prod_e': 1}
            
        obj_den ((str) or (dict)):
            The denominator of the fractional objective function, provided as a linear expression,
            either as a character string or as a dict. E.g.: obj_num='1.0 EX_subst_e' or 
            obj_num={'EX_subst_e': 1}
            
        obj_sense (optional (str)): (Default: 'maximize')
            The optimization direction can be set either to 'maximize' (or 'max') or 'minimize' (or 'min').
            
        solver (optional (str)):
            The solver that should be used for YOpt.
            
        constraints (optional (str) or (list of str) or (list of [dict,str,float])): (Default: '')
            List of *linear* constraints to be applied on top of the model: signs + or -, scalar 
            factors for reaction rates, inclusive (in)equalities and a float value on the right hand 
            side. The parsing of the constraints input allows for some flexibility. Correct (and 
            identical) inputs are, for instance: 
            constraints='-EX_o2_e <= 5, ATPM = 20' or
            constraints=['-EX_o2_e <= 5', 'ATPM = 20'] or
            constraints=[[{'EX_o2_e':-1},'<=',5], [{'ATPM':1},'=',20]]

    Returns:
        (cobra.core.Solution):
            A solution object that contains the objective value, an optimal flux vector and the optmization
            status.
    """
    reaction_ids = model.reactions.list_attr("id")
    if 'obj_num' not in kwargs:
        raise Exception('For a yield optimization, the numerator expression must be provided under the keyword "obj_num".')
    else:
        if type(kwargs['obj_num']) is not dict:
            obj_num = linexpr2mat(kwargs['obj_num'], reaction_ids)
        else:
            obj_num = linexprdict2mat(kwargs['obj_num'], reaction_ids)

    if 'obj_den' not in kwargs:
        raise Exception('For a yield optimization, the denominator expression must be provided under the keyword "obj_den".')
    else:
        if type(kwargs['obj_den']) is not dict:
            obj_den = linexpr2mat(kwargs['obj_den'], reaction_ids)
        else:
            obj_den = linexprdict2mat(kwargs['obj_den'], reaction_ids)

    if 'obj_sense' not in kwargs or kwargs['obj_sense'] not in ['min', 'minimize']:
        obj_sense = 'maximize'
    else:
        obj_sense = 'minimize'
        obj_num = -obj_num

    if CONSTRAINTS in kwargs:
        kwargs[CONSTRAINTS] = parse_constraints(kwargs[CONSTRAINTS], reaction_ids)
        A_ineq, b_ineq, A_eq, b_eq = lineqlist2mat(kwargs[CONSTRAINTS], reaction_ids)

    if SOLVER not in kwargs:
        kwargs[SOLVER] = None
    solver = select_solver(kwargs[SOLVER], model)

    # prepare vectors and matrices for base problem
    A_eq_base = create_stoichiometric_matrix(model)
    A_eq_base = sparse.csr_matrix(A_eq_base)
    b_eq_base = [0] * len(model.metabolites)
    if 'A_eq' in locals():
        A_eq = sparse.vstack((A_eq_base, A_eq), 'csr')
        b_eq = b_eq_base + b_eq
    else:
        A_eq = A_eq_base
        b_eq = b_eq_base
    if 'A_ineq' not in locals():
        A_ineq = sparse.csr_matrix((0, len(model.reactions)))
        b_ineq = []
    # Integrate upper and lower bounds into A_ineq and b_ineq
    real_lb = [i for i, v in enumerate(model.reactions.list_attr('lower_bound')) if not isinf(v)]
    real_ub = [i for i, v in enumerate(model.reactions.list_attr('upper_bound')) if not isinf(v)]
    sparse_lb = sparse.coo_matrix(([-1] * len(real_lb), (range(len(real_lb)), real_lb)), (len(real_lb), A_ineq.shape[1]))
    sparse_ub = sparse.coo_matrix(([1] * len(real_ub), (range(len(real_ub)), real_ub)), (len(real_ub), A_ineq.shape[1]))
    A_ineq = sparse.vstack((A_ineq, sparse_lb, sparse_ub), 'csr')
    b_ineq = b_ineq + [-model.reactions[i].lower_bound for i in real_lb] + \
                      [ model.reactions[i].upper_bound for i in real_ub]
    # Analyze maximum and minimum value of denominator function to decide whether to fix it to +1 or -1 or abort computation
    den_sign = []
    den_prob = MILP_LP(c=obj_den.todense().tolist()[0], A_ineq=A_ineq, b_ineq=b_ineq, A_eq=A_eq, b_eq=b_eq, solver=solver)
    _, min_denx, status_i = den_prob.solve()
    # check model feasibility
    if status_i not in [OPTIMAL, UNBOUNDED]:
        fluxes = {reaction_ids[i]: nan for i in range(len(reaction_ids))}
        return Solution(objective_value=nan, status=INFEASIBLE, fluxes=fluxes)
    # is minimum of denominator term negative?
    if min_denx < 0:
        den_sign += [-1]
    # is maximum of denominator term positive?
    den_prob.set_objective((-obj_den).todense().tolist()[0])
    _, max_denx, _ = den_prob.solve()
    if max_denx < 0:
        den_sign += [1]
    # is denominator fixed to zero
    if not den_sign:
        logging.error('Denominator term can only take the value 0. Yield computation impossible.')
        fluxes = {reaction_ids[i]: nan for i in range(len(reaction_ids))}
        return Solution(objective_value=nan, status=INFEASIBLE, fluxes=fluxes)

    # Create linear fractional problem (LFP)
    # A variable is added here to scale the right hand side of the original problem
    A_ineq_lfp = sparse.hstack((A_ineq, sparse.csr_matrix([-b for b in b_ineq]).transpose()), 'csr')
    b_ineq_lfp = [0 for _ in b_ineq]
    A_eq_lfp = sparse.vstack((  sparse.hstack((A_eq,sparse.csr_matrix([-b for b in b_eq]).transpose())),\
                            sparse.hstack((obj_den,sparse.csr_matrix((1, 1))))),'csr')
    opt_cx = inf
    for d in den_sign:
        b_eq_lfp = [0 for _ in b_eq] + [d]
        # build LP
        yopt_prob = MILP_LP(c=(-d * obj_num).todense().tolist()[0] + [0],
                            A_ineq=A_ineq_lfp,
                            b_ineq=b_ineq_lfp,
                            A_eq=A_eq_lfp,
                            b_eq=b_eq_lfp,
                            solver=solver)
        x_i, opt_i, status_i = yopt_prob.solve()
        if opt_i < opt_cx:
            x = x_i
            opt_cx = opt_i
            status = status_i
    if status is OPTIMAL:
        factor = x[-1]  # get factor from LFP
        if factor == 0:
            factor = 1
        fluxes = {r: x[i] / factor for i, r in enumerate(reaction_ids)}
        if obj_sense == 'maximize':
            opt_cx = -opt_cx  # correct sign (maximization)
        sol = Solution(objective_value=opt_cx, status=status, fluxes=fluxes)
        if x[-1] == 0:
            sol.scalable = True
            logging.info('Solution flux vector may be scaled with an arbitrary factor.')
        else:
            sol.scalable = False
        return sol
    elif status is UNBOUNDED:
        opt_cx = nan
        # check if numerator can be nonzero when denominator is zero
        fct = 1 - 2 * (obj_sense == 'maximize')
        den_prob.set_objective((fct * obj_num).todense().tolist()[0])
        den_prob.add_eq_constraints(obj_den, [0])
        x, max_num, status_i = den_prob.solve()
        if isinf(max_num) or status_i == INFEASIBLE:  # if numerator is still unbounded, generate a fixed solution
            num_prob = MILP_LP(c=(-fct * obj_num).todense().tolist()[0], A_ineq=A_ineq, b_ineq=b_ineq, A_eq=A_eq, b_eq=b_eq, solver=solver)
            min_num = num_prob.slim_solve()
            if min_num <= 0:
                num_prob.add_eq_constraints((-fct * obj_num).todense().tolist()[0], [1])
            else:
                num_prob.add_eq_constraints((-fct * obj_num).todense().tolist()[0], min_num)
            x, opt_i, status_i = num_prob.solve()
        if not isnan(max_num):
            if max_num == 0:
                num_prob = MILP_LP(c=(-fct * obj_num).todense().tolist()[0],
                                   A_ineq=A_ineq,
                                   b_ineq=b_ineq,
                                   A_eq=A_eq,
                                   b_eq=b_eq,
                                   solver=solver)
                num_prob.add_eq_constraints((obj_den).todense().tolist()[0], [0])
                x, opt_i, status_i = num_prob.solve()
                fluxes = {r: x[i] for i, r in enumerate(reaction_ids)}
                logging.warning('Yield is undefined because denominator can become zero. Solution '\
                    'flux vector maximizes the numerator.')
                sol = Solution(objective_value=opt_cx, status=status, fluxes=fluxes)
                sol.scalable = False
            else:
                fluxes = {r: x[i] for i, r in enumerate(reaction_ids)}
                logging.warning('Yield is undefined because denominator can become zero. Solution '\
                    'flux vector maximizes the numerator.')
                sol = Solution(objective_value=opt_cx, status=status, fluxes=fluxes)
                sol.scalable = False
            return sol
        else:
            if obj_sense == 'maximize':
                opt_cx = inf
            else:
                opt_cx = -inf
            fluxes = {r: x[i] for i, r in enumerate(reaction_ids)}
            logging.warning('Yield is infinite because the numerator is unbounded.')
            sol = Solution(objective_value=opt_cx, status=status, fluxes=fluxes)
            sol.scalable = True
            return sol
    else:
        status = INFEASIBLE


def _make_fix_constraint(axes, ax_idx, ax_type, value):
    """Create an equality constraint fixing axis ax_idx to value."""
    if ax_type == 'rate':
        return [axes[ax_idx][0], '=', value]
    else:  # yield
        merged = dict(axes[ax_idx][0])
        for k, v in axes[ax_idx][1].items():
            merged[k] = merged.get(k, 0) - v * value
        return [merged, '=', 0]


def _optimize_axis(model, ax_idx, axes, ax_type, constraints, solver, sense):
    """Optimize one axis subject to constraints. Returns (value, status)."""
    if ax_type == 'rate':
        sol = fba(model, obj=axes[ax_idx][0], constraints=constraints, solver=solver, obj_sense=sense)
    else:  # yield
        sol = yopt(model, obj_num=axes[ax_idx][0], obj_den=axes[ax_idx][1],
                   constraints=constraints, solver=solver, obj_sense=sense)
    return sol


def _detect_degeneracy(val_limits, num_axes):
    """Classify solution space dimensionality based on axis ranges."""
    degenerate = []
    for vmin, vmax in val_limits:
        scale = max(1.0, abs(vmin), abs(vmax))
        degenerate.append(abs(vmax - vmin) < 1e-8 * scale)
    n_degen = sum(degenerate)
    if n_degen == num_axes:
        return 'point', degenerate
    elif n_degen == num_axes - 1:
        return 'line', degenerate
    elif n_degen == num_axes - 2:
        return 'plane', degenerate
    return 'full', degenerate


def _trace_polygon_rate_rate(model, axes, constraints, solver):
    """Trace the exact convex polygon boundary for rate-rate 2D plots.

    Uses recursive normal-bisection: finds 4 initial extremes, then refines
    each edge by optimizing along the outward normal. O(V) LPs for V vertices.
    """
    ax0_coeff = axes[0][0]
    ax1_coeff = axes[1][0]

    def _fba_project(obj_dict):
        sol = fba(model, obj=obj_dict, constraints=constraints, solver=solver, obj_sense='maximize')
        if sol.status not in [OPTIMAL]:
            return None
        x0 = sum(c * sol.fluxes.get(r, 0) for r, c in ax0_coeff.items())
        x1 = sum(c * sol.fluxes.get(r, 0) for r, c in ax1_coeff.items())
        return (ceil_dec(x0, 9), ceil_dec(x1, 9))

    # Step 1: Find 4 extremes
    extremes = []
    for coeff, sense in [(ax0_coeff, 'maximize'), (ax0_coeff, 'minimize'),
                         (ax1_coeff, 'maximize'), (ax1_coeff, 'minimize')]:
        sol = fba(model, obj=coeff, constraints=constraints, solver=solver, obj_sense=sense)
        if sol.status == OPTIMAL:
            x0 = sum(c * sol.fluxes.get(r, 0) for r, c in ax0_coeff.items())
            x1 = sum(c * sol.fluxes.get(r, 0) for r, c in ax1_coeff.items())
            extremes.append((ceil_dec(x0, 9), ceil_dec(x1, 9)))

    if len(extremes) < 2:
        return extremes if extremes else [(0, 0)]

    # Step 2: Deduplicate
    tol = 1e-8
    unique_pts = []
    for p in extremes:
        if not any(abs(p[0] - q[0]) < tol and abs(p[1] - q[1]) < tol for q in unique_pts):
            unique_pts.append(p)

    if len(unique_pts) == 1:
        return unique_pts

    # Step 3: Order counterclockwise via atan2
    cx = sum(p[0] for p in unique_pts) / len(unique_pts)
    cy = sum(p[1] for p in unique_pts) / len(unique_pts)
    unique_pts.sort(key=lambda p: arctan2(p[1] - cy, p[0] - cx))

    # Step 4: Recursive edge refinement
    diameter = max(
        ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5
        for a in unique_pts for b in unique_pts
    )
    min_edge = 1e-10 * diameter if diameter > 0 else 1e-10

    def _refine(vi, vj, depth):
        if depth > 50:
            return [vi]
        edge_len = ((vi[0]-vj[0])**2 + (vi[1]-vj[1])**2)**0.5
        if edge_len < min_edge:
            return [vi]
        # Outward normal (perpendicular to edge, pointing outward from centroid)
        dx, dy = vj[0] - vi[0], vj[1] - vi[1]
        nx, ny = dy, -dx  # rotate 90 degrees
        # Ensure normal points outward (away from centroid)
        mid_x, mid_y = (vi[0] + vj[0]) / 2, (vi[1] + vj[1]) / 2
        if nx * (mid_x - cx) + ny * (mid_y - cy) < 0:
            nx, ny = -nx, -ny
        # Combine into single LP objective
        obj = {}
        for r, c in ax0_coeff.items():
            obj[r] = obj.get(r, 0) + nx * c
        for r, c in ax1_coeff.items():
            obj[r] = obj.get(r, 0) + ny * c
        p_new = _fba_project(obj)
        if p_new is None:
            return [vi]
        # Check if p_new is outside the edge (beyond tolerance)
        # Project p_new onto edge normal direction
        dist = nx * (p_new[0] - vi[0]) + ny * (p_new[1] - vi[1])
        norm_len = (nx**2 + ny**2)**0.5
        if norm_len > 0:
            dist /= norm_len
        if dist > tol and edge_len > min_edge:
            # Check p_new is not a duplicate of vi or vj
            if ((abs(p_new[0]-vi[0]) < tol and abs(p_new[1]-vi[1]) < tol) or
                (abs(p_new[0]-vj[0]) < tol and abs(p_new[1]-vj[1]) < tol)):
                return [vi]
            left = _refine(vi, p_new, depth + 1)
            right = _refine(p_new, vj, depth + 1)
            return left + right
        return [vi]

    # Refine all edges of the polygon
    refined = []
    n = len(unique_pts)
    for i in range(n):
        vi = unique_pts[i]
        vj = unique_pts[(i + 1) % n]
        refined.extend(_refine(vi, vj, 0))

    # Close polygon
    if refined and refined[0] != refined[-1]:
        refined.append(refined[0])
    return refined


def _trace_boundary_adaptive(model, axes, ax_types, constraints, solver, max_depth=15):
    """Trace upper and lower boundaries adaptively for yield or mixed axes.

    Returns (upper_boundary, lower_boundary) as sorted lists of (x, y) tuples.
    Uses recursive midpoint refinement where linear interpolation error exceeds tolerance.
    """
    def _fix_and_opt(x_val, sense):
        constr = constraints.copy()
        constr.append(_make_fix_constraint(axes, 0, ax_types[0], x_val))
        sol = _optimize_axis(model, 1, axes, ax_types[1], constr, solver, sense)
        if sol.status in [OPTIMAL]:
            return ceil_dec(sol.objective_value, 9) if sense == 'minimize' else floor_dec(sol.objective_value, 9)
        return nan

    # Step 1: axis-0 range endpoints (already known from val_limits, but we need y values)
    x_min, x_max = ceil_dec(
        _optimize_axis(model, 0, axes, ax_types[0], constraints, solver, 'minimize').objective_value, 8
    ), floor_dec(
        _optimize_axis(model, 0, axes, ax_types[0], constraints, solver, 'maximize').objective_value, 8
    )

    # Step 2: y values at endpoints
    y_min_at_xmin = _fix_and_opt(x_min, 'minimize')
    y_max_at_xmin = _fix_and_opt(x_min, 'maximize')
    y_min_at_xmax = _fix_and_opt(x_max, 'minimize')
    y_max_at_xmax = _fix_and_opt(x_max, 'maximize')

    # Upper boundary
    upper = [(x_min, y_max_at_xmin), (x_max, y_max_at_xmax)]
    upper = [(x, y) for x, y in upper if not isnan(y)]
    # Lower boundary
    lower = [(x_min, y_min_at_xmin), (x_max, y_min_at_xmax)]
    lower = [(x, y) for x, y in lower if not isnan(y)]

    if not upper and not lower:
        return [], []

    # Compute y_range for tolerance
    all_y = [y for _, y in upper + lower]
    y_range = max(all_y) - min(all_y) if all_y else 0
    abs_tol = max(1e-6, 1e-3 * abs(y_range))

    def _refine_boundary(boundary, sense, depth):
        if depth > max_depth or len(boundary) < 2:
            return boundary
        new_boundary = [boundary[0]]
        changed = False
        for i in range(len(boundary) - 1):
            x_left, y_left = boundary[i]
            x_right, y_right = boundary[i + 1]
            if abs(x_right - x_left) < 1e-10:
                new_boundary.append(boundary[i + 1])
                continue
            x_mid = (x_left + x_right) / 2
            y_actual = _fix_and_opt(x_mid, sense)
            if isnan(y_actual):
                new_boundary.append(boundary[i + 1])
                continue
            y_interp = (y_left + y_right) / 2
            if abs(y_actual - y_interp) > abs_tol:
                new_boundary.append((x_mid, y_actual))
                changed = True
            new_boundary.append(boundary[i + 1])
        if changed:
            return _refine_boundary(new_boundary, sense, depth + 1)
        return new_boundary

    upper = _refine_boundary(upper, 'maximize', 0)
    lower = _refine_boundary(lower, 'minimize', 0)
    return upper, lower


def _trace_polytope_3d_rate(model, axes, constraints, solver):
    """Trace exact 3D convex polytope for all-rate axes via face-normal refinement.

    Finds 6 initial extremes, builds convex hull, then iteratively refines
    by optimizing along each face's outward normal. Converges when no face
    produces a new vertex.
    """
    coeffs = [axes[i][0] for i in range(3)]
    tol = 1e-8

    def _project(sol):
        return tuple(
            ceil_dec(sum(c * sol.fluxes.get(r, 0) for r, c in coeffs[j].items()), 9)
            for j in range(3)
        )

    def _is_dup(pt, pts):
        return any(all(abs(pt[k] - q[k]) < tol for k in range(3)) for q in pts)

    # Find 6 extremes (max/min of each axis)
    vertices = []
    for i in range(3):
        for sense in ['maximize', 'minimize']:
            sol = fba(model, obj=coeffs[i], constraints=constraints, solver=solver, obj_sense=sense)
            if sol.status == OPTIMAL:
                pt = _project(sol)
                if not _is_dup(pt, vertices):
                    vertices.append(pt)

    if len(vertices) < 4:
        return [array(v) for v in vertices], []

    # Iterative face-normal refinement
    for _ in range(20):
        try:
            hull = ConvexHull(array(vertices))
        except Exception:
            break
        new_found = False
        for eq in hull.equations:
            normal = eq[:3]  # outward-pointing face normal from ConvexHull
            obj = {}
            for k in range(3):
                for r, c in coeffs[k].items():
                    obj[r] = obj.get(r, 0) + normal[k] * c
            sol = fba(model, obj=obj, constraints=constraints, solver=solver, obj_sense='maximize')
            if sol.status != OPTIMAL:
                continue
            p_new = _project(sol)
            if not _is_dup(p_new, vertices):
                vertices.append(p_new)
                new_found = True
        if not new_found:
            break

    # Final hull — extract triangles and merged polygon faces
    try:
        hull = ConvexHull(array(vertices))
        triang = hull.simplices.tolist()
        face_polys = _hull_face_polygons(hull)
    except Exception:
        triang = []
        face_polys = None
    return [array(v) for v in vertices], triang, face_polys


def _hull_face_polygons(hull):
    """Merge coplanar ConvexHull simplices into polygon faces.

    Returns a list of faces, each face being a list of vertex indices
    ordered counterclockwise (viewed from outside).
    """
    from collections import defaultdict
    # Group simplices by face equation (rounded for coplanarity check)
    face_groups = defaultdict(set)
    for i, eq in enumerate(hull.equations):
        key = tuple(round(v, 5) for v in eq)
        for vi in hull.simplices[i]:
            face_groups[key].add(vi)

    faces = []
    for eq_key, vertex_indices in face_groups.items():
        verts = list(vertex_indices)
        if len(verts) < 3:
            continue
        normal = array(eq_key[:3])
        pts = hull.points[verts]
        centroid = pts.mean(axis=0)
        # Build orthonormal basis in the face plane
        ref = array([1, 0, 0]) if abs(normal[0]) < 0.9 else array([0, 1, 0])
        u = ref - normal * normal.dot(ref)
        u = u / (u.dot(u) ** 0.5)
        v = array([normal[1]*u[2] - normal[2]*u[1],
                    normal[2]*u[0] - normal[0]*u[2],
                    normal[0]*u[1] - normal[1]*u[0]])
        # Sort by angle in face plane
        angles = [arctan2((pt - centroid).dot(v), (pt - centroid).dot(u)) for pt in pts]
        order = sorted(range(len(verts)), key=lambda i: angles[i])
        faces.append([verts[i] for i in order])
    return faces


def _trace_3d_slice_polygon(model, axes, ax_type, val_limits, constraints, solver, points):
    """3D surface for 1-yield + 2-rate: slice along yield axis, trace rate-rate polygon per slice."""
    yield_idx = next(i for i, t in enumerate(ax_type) if t == 'yield')
    rate_indices = [i for i, t in enumerate(ax_type) if t == 'rate']

    y_space = linspace(val_limits[yield_idx][0], val_limits[yield_idx][1], num=points).tolist()

    datapoints = []
    slice_idx_lists = []

    for y_val in y_space:
        constr = constraints.copy()
        constr.append(_make_fix_constraint(axes, yield_idx, ax_type[yield_idx], y_val))
        sub_axes = [axes[rate_indices[0]], axes[rate_indices[1]]]
        polygon_2d = _trace_polygon_rate_rate(model, sub_axes, constr, solver)

        if not polygon_2d or len(polygon_2d) < 2:
            continue

        # Remove closing duplicate if present
        if len(polygon_2d) > 1 and polygon_2d[0] == polygon_2d[-1]:
            polygon_2d = polygon_2d[:-1]
        if not polygon_2d:
            continue

        # Align polygons: rotate to start at max rate_indices[0] value
        max_i = max(range(len(polygon_2d)), key=lambda i: (polygon_2d[i][0], polygon_2d[i][1]))
        polygon_2d = polygon_2d[max_i:] + polygon_2d[:max_i]

        # Convert 2D → 3D
        idx_list = []
        for pt in polygon_2d:
            p3d = [0.0, 0.0, 0.0]
            p3d[rate_indices[0]] = pt[0]
            p3d[rate_indices[1]] = pt[1]
            p3d[yield_idx] = y_val
            idx_list.append(len(datapoints))
            datapoints.append(array(p3d))
        slice_idx_lists.append(idx_list)

    if not slice_idx_lists:
        return [], [], []

    # Stitch adjacent slice polygons (close the ring by appending first index)
    triang = []
    for s in range(len(slice_idx_lists) - 1):
        strip_a = slice_idx_lists[s] + [slice_idx_lists[s][0]]
        strip_b = slice_idx_lists[s + 1] + [slice_idx_lists[s + 1][0]]
        _triangulate_strips(strip_a, strip_b, datapoints, triang)

    # Cap first and last slices (fan triangulation)
    first = slice_idx_lists[0]
    if len(first) >= 3:
        for i in range(1, len(first) - 1):
            triang.append([first[0], first[i + 1], first[i]])
    last = slice_idx_lists[-1]
    if len(last) >= 3:
        for i in range(1, len(last) - 1):
            triang.append([last[0], last[i], last[i + 1]])

    return datapoints, triang, slice_idx_lists


def _triangulate_strips(strip_a, strip_b, datapoints, triang, flip_winding=False):
    """Connect two ordered index strips into triangles (for 3D mesh construction).

    Walks both strips simultaneously, advancing whichever side has the shorter
    diagonal, creating a triangle strip between two contours.
    """
    if len(strip_a) < 1 or len(strip_b) < 1:
        return
    i, j = 0, 0
    while i < len(strip_a) - 1 or j < len(strip_b) - 1:
        if i >= len(strip_a) - 1:
            tri = [strip_a[i], strip_b[j], strip_b[j + 1]]
            j += 1
        elif j >= len(strip_b) - 1:
            tri = [strip_a[i], strip_b[j], strip_a[i + 1]]
            i += 1
        else:
            # Choose shorter diagonal
            pa_next = datapoints[strip_a[i + 1]]
            pb_next = datapoints[strip_b[j + 1]]
            pa_curr = datapoints[strip_a[i]]
            pb_curr = datapoints[strip_b[j]]
            d1 = sum((pa_next[k] - pb_curr[k])**2 for k in range(len(pa_next)))
            d2 = sum((pa_curr[k] - pb_next[k])**2 for k in range(len(pa_curr)))
            if d1 < d2:
                tri = [strip_a[i], strip_b[j], strip_a[i + 1]]
                i += 1
            else:
                tri = [strip_a[i], strip_b[j], strip_b[j + 1]]
                j += 1
        if flip_winding:
            tri = [tri[0], tri[2], tri[1]]
        triang.append(tri)


def plot_flux_space(model, axes, **kwargs) -> Tuple[list, list, list]:
    """Plot projections of the space of steady-state flux vectors onto two or three dimensions.
    
    This function uses LP and matplotlib to generate lower dimensional representations of the 
    flux space. Custom *linear* or *fractional-linear* expressions can be used for the plot
    axis. The most commonly used flux space reprentations are the *production envelope* that
    plots the growth rate (x) vs the product synthesis rate (y) and the *yield space plot*
    that plots the biomass yield (x) vs the product yiel (y). One may specify additional
    constraints to investigate subspaces of the metabolic steady-state flux space.
    
    Example:
        plot_flux_space(model,('BIOMASS_Ecoli_core_w_GAM','EX_etoh_e'))
    
    Args:
        model (cobra.Model):
            A metabolic model that is an instance of the cobra.Model class.
            
        axes ((list of lists) or (list of str)):
            A set of linear expressions that specify which reactions/expressions/dimensions should be
            used on the axis. Examples: axes=['BIOMASS_Ecoli_core_w_GAM','EX_etoh_e'] or 
            axes=[['BIOMASS_Ecoli_core_w_GAM','-EX_glc_e'],['EX_etoh_e','-EX_glc_e']] or
            axes=[['BIOMASS_Ecoli_core_w_GAM'],['EX_etoh_e','-EX_glc_e']]
            
        solver (optional (str)):
            The solver that should be used for scanning the flux space.
            
        constraints (optional (str) or (list of str) or (list of [dict,str,float])): (Default: '')
            List of *linear* constraints to be applied on top of the model: signs + or -, scalar 
            factors for reaction rates, inclusive (in)equalities and a float value on the right hand 
            side. The parsing of the constraints input allows for some flexibility. Correct (and 
            identical) inputs are, for instance: 
            constraints='-EX_o2_e <= 5, ATPM = 20' or
            constraints=['-EX_o2_e <= 5', 'ATPM = 20'] or
            constraints=[[{'EX_o2_e':-1},'<=',5], [{'ATPM':1},'=',20]]
            
        plt_backend (optional (str)):
            The matplotlib backend that should be used for plotting:
            interactive backends: GTK3Agg, GTK3Cairo, GTK4Agg, GTK4Cairo, MacOSX, nbAgg, QtAgg, QtCairo,
                                  TkAgg, TkCairo, WebAgg, WX, WXAgg, WXCairo, Qt5Agg, Qt5Cairo
            non-interactive backends: agg, cairo, pdf, pgf, ps, svg, template
            
        show (optional (bool)): (Default: True)
            Should matplotlib show the plot or should it stop after plot generation. show=False can
            be useful if flux spaces should be plotted and saved in a non-interactive environment, 
            multiple flux spaces should be plotted at once or the plot should be modified before been shown.
        
        cmap (optional (str)): (Default: 'managua')
            The matplotlib colormap used for 3D face coloring.

        points (optional (int)): (Default: 25 (3D) or 40 (2D))
            Controls resolution of non-exact (approximate) plot regions. For rate-only axes, boundary
            tracing finds exact vertices and this parameter is ignored. For yield axes, this controls:
            - 2D yield plots: max refinement depth = max(5, log2(points))
            - 3D with 1 yield axis: number of slices along the yield axis
            - 3D with 2+ yield axes: number of grid intervals per axis

    Returns:
        (Tuple):
            (datapoints, triang, plot1). The array of datapoints from which the plot was generated. These
            datapoints are optimal values for different optimizations within the flux space. The triang
            variable contains information about which datapoints need to be connected in triangles to
            render a closed surface. The last variable contains the matplotlib object.
    """
    reaction_ids = model.reactions.list_attr("id")

    if CONSTRAINTS in kwargs:
        kwargs[CONSTRAINTS] = parse_constraints(kwargs[CONSTRAINTS], reaction_ids)
    else:
        kwargs[CONSTRAINTS] = []

    if SOLVER not in kwargs:
        kwargs[SOLVER] = None
    solver = select_solver(kwargs[SOLVER], model)

    if 'plt_backend' in kwargs:
        set_matplotlib_backend(kwargs['plt_backend'])

    if 'show' not in kwargs:
        show = True
    else:
        show = kwargs['show']

    cmap = kwargs.get('cmap', 'managua')

    axes = [list(ax) if not isinstance(ax, str) else [ax] for ax in axes]  # cast to list of lists
    num_axes = len(axes)
    if num_axes not in [2, 3]:
        raise Exception('Please define 2 or 3 axes as a list of tuples [ax1, ax2, (optional) ax3] with ax1 = (den,num).\n'+\
                        '"den" and "num" being linear expressions.')

    if 'points' in kwargs:
        points = kwargs['points']
    else:
        if num_axes == 2:
            points = 40
        else:
            points = 25

    ax_name = ["" for _ in range(num_axes)]
    ax_limits = [(nan, nan) for _ in range(num_axes)]
    val_limits = [(nan, nan) for _ in range(num_axes)]
    ax_type = ["" for _ in range(num_axes)]
    for i, ax in enumerate(axes):
        if len(ax) == 1:
            ax_type[i] = 'rate'
        elif len(ax) == 2:
            ax_type[i] = 'yield'
        if ax_type[i] == 'rate':
            ax[0] = parse_linexpr(ax[0], reaction_ids)[0]
            ax_name[i] = linexprdict2str(ax[0])
            sol_min = fba(model, obj=ax[0], constraints=kwargs[CONSTRAINTS], solver=solver, obj_sense='minimize')
            sol_max = fba(model, obj=ax[0], constraints=kwargs[CONSTRAINTS], solver=solver, obj_sense='maximize')
            # abort if any of the fluxes are unbounded or undefined
            inval = [i + 1 for i, v in enumerate([sol_min, sol_max]) if v.status == UNBOUNDED or v.status == INFEASIBLE]
            if any(inval):
                raise Exception('One of the specified reactions is unbounded or problem is infeasible. Plot cannot be generated.')
        elif ax_type[i] == 'yield':
            ax[0] = parse_linexpr(ax[0], reaction_ids)[0]
            ax[1] = parse_linexpr(ax[1], reaction_ids)[0]
            ax_name[i] = '(' + linexprdict2str(ax[0]) + ') / (' + linexprdict2str(ax[1]) + ')'
            sol_min = yopt(model, obj_num=ax[0], obj_den=ax[1], constraints=kwargs[CONSTRAINTS], solver=solver, obj_sense='minimize')
            sol_max = yopt(model, obj_num=ax[0], obj_den=ax[1], constraints=kwargs[CONSTRAINTS], solver=solver, obj_sense='maximize')
            # abort if any of the yields are unbounded or undefined
            inval = [i + 1 for i, v in enumerate([sol_min, sol_max]) if v.status == UNBOUNDED or v.status == INFEASIBLE]
            if any(inval):
                raise Exception('One of the specified yields is unbounded or undefined or problem is infeasible. Plot cannot be generated.')
        val_limits[i] = [ceil_dec(sol_min.objective_value, 8), floor_dec(sol_max.objective_value, 8)]
        ax_limits[i] = [min((0, val_limits[i][0])), max((0, val_limits[i][1]))]
        # Ensure non-singular axis limits (avoid identical low/high)
        if ax_limits[i][0] == ax_limits[i][1]:
            pad = max(0.5, abs(ax_limits[i][0]) * 0.1)
            ax_limits[i] = [ax_limits[i][0] - pad, ax_limits[i][1] + pad]

    # Detect degeneracy
    degen, degen_axes = _detect_degeneracy(val_limits, num_axes)

    if num_axes == 2:
        # === 2D dispatch ===
        if degen == 'point':
            plot1 = plt.plot([val_limits[0][0]], [val_limits[1][0]], 'o')[0]
            plot1.axes.set_xlabel(ax_name[0])
            plot1.axes.set_ylabel(ax_name[1])
            plot1.axes.set_xlim(ax_limits[0][0] * 1.05, ax_limits[0][1] * 1.05)
            plot1.axes.set_ylim(ax_limits[1][0] * 1.05, ax_limits[1][1] * 1.05)
            if show:
                try:
                    plt.show()
                except UserWarning as e:
                    if 'FigureCanvasTemplate is non-interactive' in str(e):
                        logging.warning('warning: Interactive plot not supported in current execution environment.')
            return [[val_limits[0][0], val_limits[1][0]]], [], plot1

        if degen == 'line':
            # Determine which axis is degenerate and trace the other
            if degen_axes[0]:
                # x is fixed, trace y range
                x_pts = [val_limits[0][0], val_limits[0][0]]
                y_pts = [val_limits[1][0], val_limits[1][1]]
            else:
                # y is fixed, trace x range
                x_pts = [val_limits[0][0], val_limits[0][1]]
                y_pts = [val_limits[1][0], val_limits[1][0]]
            plot1 = plt.plot(x_pts, y_pts, linewidth=1.5)[0]
            plot1.axes.set_xlabel(ax_name[0])
            plot1.axes.set_ylabel(ax_name[1])
            plot1.axes.set_xlim(ax_limits[0][0] * 1.05, ax_limits[0][1] * 1.05)
            plot1.axes.set_ylim(ax_limits[1][0] * 1.05, ax_limits[1][1] * 1.05)
            if show:
                try:
                    plt.show()
                except UserWarning as e:
                    if 'FigureCanvasTemplate is non-interactive' in str(e):
                        logging.warning('warning: Interactive plot not supported in current execution environment.')
            datapoints = [[x, y] for x, y in zip(x_pts, y_pts)]
            return datapoints, [], plot1

        # Full 2D: choose algorithm based on axis types
        if all(t == 'rate' for t in ax_type):
            vertices = _trace_polygon_rate_rate(model, axes, kwargs[CONSTRAINTS], solver)
        else:
            adapt_depth = max(5, int(log2(max(points, 2))))
            upper, lower = _trace_boundary_adaptive(
                model, axes, ax_type, kwargs[CONSTRAINTS], solver, max_depth=adapt_depth)
            # Build polygon from upper (left-to-right) + reversed lower (right-to-left)
            if upper and lower:
                vertices = upper + list(reversed(lower))
            elif upper:
                vertices = upper
            elif lower:
                vertices = lower
            else:
                vertices = []
            # Close polygon if endpoints don't match
            if len(vertices) > 1 and vertices[0] != vertices[-1]:
                vertices.append(vertices[0])

        if not vertices:
            raise Exception('Could not trace any boundary. Problem may be infeasible.')

        # Check for collinear degeneracy (polygon traced to a line or point)
        unique_verts = []
        for v in vertices:
            if not any(abs(v[0] - u[0]) < 1e-10 and abs(v[1] - u[1]) < 1e-10 for u in unique_verts):
                unique_verts.append(v)
        is_collinear = False
        if len(unique_verts) <= 2:
            is_collinear = True
        elif len(unique_verts) >= 3:
            # Check if all points lie on a single line
            x0, y0 = unique_verts[0]
            dx, dy = unique_verts[1][0] - x0, unique_verts[1][1] - y0
            span = max(abs(dx), abs(dy), 1e-12)
            is_collinear = all(
                abs((v[0] - x0) * dy - (v[1] - y0) * dx) / span < 1e-6
                for v in unique_verts[2:])

        if is_collinear and len(unique_verts) <= 1:
            # Collapsed to a point
            plot1 = plt.plot([unique_verts[0][0]], [unique_verts[0][1]], 'o')[0]
            plot1.axes.set_xlabel(ax_name[0])
            plot1.axes.set_ylabel(ax_name[1])
            plot1.axes.set_xlim(ax_limits[0][0] * 1.05, ax_limits[0][1] * 1.05)
            plot1.axes.set_ylim(ax_limits[1][0] * 1.05, ax_limits[1][1] * 1.05)
            if show:
                try:
                    plt.show()
                except UserWarning as e:
                    if 'FigureCanvasTemplate is non-interactive' in str(e):
                        logging.warning('warning: Interactive plot not supported in current execution environment.')
            return [list(unique_verts[0])], [], plot1

        if is_collinear:
            # Collapsed to a line — sort along the line direction and draw
            x0, y0 = unique_verts[0]
            dx, dy = unique_verts[-1][0] - x0, unique_verts[-1][1] - y0
            unique_verts.sort(key=lambda v: (v[0] - x0) * dx + (v[1] - y0) * dy)
            x_pts = [v[0] for v in unique_verts]
            y_pts = [v[1] for v in unique_verts]
            plot1 = plt.plot(x_pts, y_pts, linewidth=1.5)[0]
            plot1.axes.set_xlabel(ax_name[0])
            plot1.axes.set_ylabel(ax_name[1])
            plot1.axes.set_xlim(ax_limits[0][0] * 1.05, ax_limits[0][1] * 1.05)
            plot1.axes.set_ylim(ax_limits[1][0] * 1.05, ax_limits[1][1] * 1.05)
            if show:
                try:
                    plt.show()
                except UserWarning as e:
                    if 'FigureCanvasTemplate is non-interactive' in str(e):
                        logging.warning('warning: Interactive plot not supported in current execution environment.')
            datapoints = [[x, y] for x, y in zip(x_pts, y_pts)]
            return datapoints, [], plot1

        # Build datapoints and triangulation for return value
        datapoints = [[v[0], v[1]] for v in vertices]
        # Fan triangulation (for return compatibility)
        n_v = len(vertices)
        if n_v > 2 and vertices[0] == vertices[-1]:
            n_v -= 1  # don't count the closing duplicate
        triang = [[0, i, i + 1] for i in range(1, n_v - 1)] if n_v >= 3 else []

        # Plot
        x = [v[0] for v in vertices]
        y = [v[1] for v in vertices]
        plot1 = plt.fill(x, y, linewidth=0.5)[0]

        plot1.axes.set_xlabel(ax_name[0])
        plot1.axes.set_ylabel(ax_name[1])
        plot1.axes.set_xlim(ax_limits[0][0] * 1.05, ax_limits[0][1] * 1.05)
        plot1.axes.set_ylim(ax_limits[1][0] * 1.05, ax_limits[1][1] * 1.05)
        if any(t == 'yield' for t in ax_type):
            plot1.axes.set_title('approximate', fontsize=8, color='gray')
        if show:
            try:
                plt.show()
            except UserWarning as e:
                if 'FigureCanvasTemplate is non-interactive' in str(e):
                    logging.warning('warning: Interactive plot not supported in current execution environment.')
        return datapoints, triang, plot1

    elif num_axes == 3:
        # === 3D dispatch ===
        if degen == 'point':
            ax3 = plt.figure().add_subplot(projection='3d')
            ax3.scatter([val_limits[0][0]], [val_limits[1][0]], [val_limits[2][0]], s=50)
            ax3.set_xlabel(ax_name[0])
            ax3.set_ylabel(ax_name[1])
            ax3.set_zlabel(ax_name[2])
            ax3.set_xlim(ax_limits[0])
            ax3.set_ylim(ax_limits[1])
            ax3.set_zlim(ax_limits[2])
            if show:
                try:
                    plt.show()
                except UserWarning as e:
                    if 'FigureCanvasTemplate is non-interactive' in str(e):
                        logging.warning('warning: Interactive plot not supported in current execution environment.')
            return [[val_limits[0][0], val_limits[1][0], val_limits[2][0]]], [], ax3

        if degen == 'line':
            # Find the non-degenerate axis and trace it
            free_ax = [i for i, d in enumerate(degen_axes) if not d][0]
            pts_3d = []
            for val in [val_limits[free_ax][0], val_limits[free_ax][1]]:
                pt = [val_limits[i][0] for i in range(3)]
                pt[free_ax] = val
                pts_3d.append(pt)
            ax3 = plt.figure().add_subplot(projection='3d')
            ax3.plot3D([p[0] for p in pts_3d], [p[1] for p in pts_3d], [p[2] for p in pts_3d], linewidth=1.5)
            ax3.set_xlabel(ax_name[0])
            ax3.set_ylabel(ax_name[1])
            ax3.set_zlabel(ax_name[2])
            ax3.set_xlim(ax_limits[0])
            ax3.set_ylim(ax_limits[1])
            ax3.set_zlim(ax_limits[2])
            if show:
                try:
                    plt.show()
                except UserWarning as e:
                    if 'FigureCanvasTemplate is non-interactive' in str(e):
                        logging.warning('warning: Interactive plot not supported in current execution environment.')
            return pts_3d, [], ax3

        if degen == 'plane':
            # One axis is degenerate — trace 2D boundary in the free plane
            free_axes = [i for i, d in enumerate(degen_axes) if not d]
            fixed_axis = [i for i, d in enumerate(degen_axes) if d][0]
            fixed_val = val_limits[fixed_axis][0]
            constr_plane = kwargs[CONSTRAINTS].copy()
            constr_plane.append(_make_fix_constraint(axes, fixed_axis, ax_type[fixed_axis], fixed_val))
            sub_axes = [axes[free_axes[0]], axes[free_axes[1]]]
            sub_types = [ax_type[free_axes[0]], ax_type[free_axes[1]]]
            if all(t == 'rate' for t in sub_types):
                polygon_2d = _trace_polygon_rate_rate(model, sub_axes, constr_plane, solver)
            else:
                upper, lower = _trace_boundary_adaptive(model, sub_axes, sub_types, constr_plane, solver)
                polygon_2d = (upper + list(reversed(lower))) if upper and lower else (upper or lower)
                if len(polygon_2d) > 1 and polygon_2d[0] != polygon_2d[-1]:
                    polygon_2d.append(polygon_2d[0])
            datapoints = []
            for pt in polygon_2d:
                p3d = [0.0, 0.0, 0.0]
                p3d[free_axes[0]] = pt[0]
                p3d[free_axes[1]] = pt[1]
                p3d[fixed_axis] = fixed_val
                datapoints.append(array(p3d))
            n_pts = len(datapoints)
            if n_pts > 1 and all(abs(datapoints[0][k] - datapoints[-1][k]) < 1e-8 for k in range(3)):
                n_pts -= 1
            triang = [[0, i, i + 1] for i in range(1, n_pts - 1)] if n_pts >= 3 else []
            face_polys = None

        else:
            # 'full' — dispatch based on number of yield axes
            n_yields = sum(1 for t in ax_type if t == 'yield')

            face_polys = None  # set by polytope path only
            slice_outlines = None  # set by slice path only

            if n_yields == 0:
                # All rate: exact 3D polytope via face-normal refinement
                datapoints, triang, face_polys = _trace_polytope_3d_rate(model, axes, kwargs[CONSTRAINTS], solver)

            elif n_yields == 1:
                # 1 yield + 2 rate: slice along yield, trace rate-rate polygon per slice
                datapoints, triang, slice_outlines = _trace_3d_slice_polygon(
                    model, axes, ax_type, val_limits, kwargs[CONSTRAINTS], solver, points)

            else:
                # 2+ yields: grid-based scanning (fallback)
                x_space = linspace(val_limits[0][0], val_limits[0][1], num=points).tolist()
                slices = []
                for x_val in x_space:
                    constr_x = kwargs[CONSTRAINTS].copy()
                    constr_x.append(_make_fix_constraint(axes, 0, ax_type[0], x_val))
                    sol_y_min = _optimize_axis(model, 1, axes, ax_type[1], constr_x, solver, 'minimize')
                    sol_y_max = _optimize_axis(model, 1, axes, ax_type[1], constr_x, solver, 'maximize')
                    if sol_y_min.status not in [OPTIMAL] or sol_y_max.status not in [OPTIMAL]:
                        continue
                    y_lo = ceil_dec(sol_y_min.objective_value, 9)
                    y_hi = floor_dec(sol_y_max.objective_value, 9)
                    if isnan(y_lo) or isnan(y_hi):
                        continue
                    if abs(y_hi - y_lo) < 1e-10:
                        y_space = [y_lo]
                    else:
                        n_pts = max(3, int(points * abs(y_hi - y_lo) / max(1e-10,
                            max(abs(val_limits[1][1] - val_limits[1][0]), 1e-10))))
                        n_pts = min(n_pts, points)
                        y_space = linspace(y_lo, y_hi, n_pts).tolist()
                    upper_slice = []
                    lower_slice = []
                    for y_val in y_space:
                        constr_xy = constr_x.copy()
                        constr_xy.append(_make_fix_constraint(axes, 1, ax_type[1], y_val))
                        sol_z_min = _optimize_axis(model, 2, axes, ax_type[2], constr_xy, solver, 'minimize')
                        sol_z_max = _optimize_axis(model, 2, axes, ax_type[2], constr_xy, solver, 'maximize')
                        z_lo = ceil_dec(sol_z_min.objective_value, 9) if sol_z_min.status in [OPTIMAL] else nan
                        z_hi = floor_dec(sol_z_max.objective_value, 9) if sol_z_max.status in [OPTIMAL] else nan
                        if not isnan(z_lo) and not isnan(z_hi):
                            upper_slice.append((y_val, z_hi))
                            lower_slice.append((y_val, z_lo))
                    if upper_slice:
                        slices.append((x_val, upper_slice, lower_slice))
                if not slices:
                    raise Exception('No feasible slices found. Problem may be infeasible.')
                datapoints = []
                datapoints_top = []
                datapoints_bottom = []
                for x_val, upper_slice, lower_slice in slices:
                    top_ids = []
                    bot_ids = []
                    for (y_val, z_hi), (_, z_lo) in zip(upper_slice, lower_slice):
                        top_ids.append(len(datapoints))
                        datapoints.append(array([x_val, y_val, z_hi]))
                        bot_ids.append(len(datapoints))
                        datapoints.append(array([x_val, y_val, z_lo]))
                    datapoints_top.append(top_ids)
                    datapoints_bottom.append(bot_ids)
                triang = []
                for s in range(len(slices) - 1):
                    _triangulate_strips(datapoints_top[s], datapoints_top[s + 1], datapoints, triang)
                    _triangulate_strips(datapoints_bottom[s], datapoints_bottom[s + 1], datapoints, triang,
                                        flip_winding=True)
                front_top = [t[0] for t in datapoints_top]
                front_bot = [b[0] for b in datapoints_bottom]
                _triangulate_strips(front_top, front_bot, datapoints, triang, flip_winding=True)
                back_top = [t[-1] for t in datapoints_top]
                back_bot = [b[-1] for b in datapoints_bottom]
                _triangulate_strips(back_top, back_bot, datapoints, triang)
                _triangulate_strips(datapoints_top[0], datapoints_bottom[0], datapoints, triang)
                _triangulate_strips(datapoints_top[-1], datapoints_bottom[-1], datapoints, triang,
                                    flip_winding=True)

        if not datapoints:
            raise Exception('No feasible points found. Problem may be infeasible.')

        # 3D plot
        x = [d[0] for d in datapoints]
        y = [d[1] for d in datapoints]
        z = [d[2] for d in datapoints]
        ax3 = plt.figure().add_subplot(projection='3d')
        ax3.dist = 10
        ax3.azim = 30
        ax3.elev = 10
        ax3.set_xlim(ax_limits[0])
        ax3.set_ylim(ax_limits[1])
        ax3.set_zlim(ax_limits[2])
        ax3.set_xlabel(ax_name[0])
        ax3.set_ylabel(ax_name[1])
        ax3.set_zlabel(ax_name[2])
        if any(t == 'yield' for t in ax_type):
            ax3.set_title('approximate', fontsize=8, color='gray')

        def _normal_color(face_pts):
            """Compute color value from face normal direction."""
            pts = array(face_pts)
            if len(pts) < 3:
                return 0.5
            e1 = pts[1] - pts[0]
            e2 = pts[2] - pts[0]
            normal = array([e1[1]*e2[2] - e1[2]*e2[1],
                            e1[2]*e2[0] - e1[0]*e2[2],
                            e1[0]*e2[1] - e1[1]*e2[0]])
            length = (normal.dot(normal)) ** 0.5
            if length > 0:
                normal = normal / length
            # Map normal direction to scalar: use spherical angles
            return arctan2(normal[1], normal[0]) + 1.5 * normal[2]

        if face_polys is not None and face_polys:
            # Polytope: render merged polygon faces with Poly3DCollection
            poly_verts = [[datapoints[i].tolist() for i in face] for face in face_polys]
            color_vals = [_normal_color(verts) for verts in poly_verts]
            mn, mx = min(color_vals), max(color_vals)
            rng = mx - mn if mx > mn else 1
            face_colors = plt.get_cmap(cmap)([(c - mn) / rng for c in color_vals])
            face_colors[:, 3] = 1.0
            collection = Poly3DCollection(poly_verts, facecolors=face_colors,
                                          edgecolors='black', linewidths=1.0)
            ax3.add_collection3d(collection)
            plot1 = collection
        elif triang:
            # Triangle mesh: render edgeless faces, color by normal
            tri_verts = [[datapoints[i].tolist() for i in t] for t in triang]
            color_vals = [_normal_color(verts) for verts in tri_verts]
            mn, mx = min(color_vals), max(color_vals)
            rng = mx - mn if mx > mn else 1
            face_colors = plt.get_cmap(cmap)([(c - mn) / rng for c in color_vals])
            face_colors[:, 3] = 1.0
            collection = Poly3DCollection(tri_verts, facecolors=face_colors,
                                          edgecolors='none', linewidths=0)
            ax3.add_collection3d(collection)
            # Draw slice polygon outlines (exact contour at each yield level)
            if slice_outlines:
                for idx_list in slice_outlines:
                    pts = array([datapoints[i] for i in idx_list])
                    pts = array(list(pts) + [pts[0]])  # close the loop
                    ax3.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                             color='gray', linewidth=0.4)
            plot1 = collection
        else:
            plot1 = ax3.scatter(x, y, z, s=20)
        if show:
            try:
                plt.show()
            except UserWarning as e:
                if 'FigureCanvasTemplate is non-interactive' in str(e):
                    logging.warning('warning: Interactive plot not supported in current execution environment.')
        return datapoints, triang, plot1


def ceil_dec(v, n):
    """Round up v to n decimals"""
    return ceil(v * (10**n)) / (10**n)


def floor_dec(v, n):
    """Round down v to n decimals"""
    return floor(v * (10**n)) / (10**n)
