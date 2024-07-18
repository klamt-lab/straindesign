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
from scipy.spatial import Delaunay  #, ConvexHull
from straindesign import MILP_LP, parse_constraints, parse_linexpr, lineqlist2mat, linexpr2dict, \
                         linexprdict2mat, SDPool, IndicatorConstraints, avail_solvers
from re import search
from straindesign.names import *
from typing import Dict, Tuple
from pandas import DataFrame
from numpy import floor, sign, mod, nan, isnan, unique, inf, isinf, full, linspace, \
                  prod, array, mean, flip, ceil, floor
from numpy.linalg import matrix_rank
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import matplotlib.pyplot as plt
from matplotlib import use as set_matplotlib_backend
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
        lp_glob.prev = 0


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
    reaction_ids = model.reactions.list_attr("id")
    numr = len(model.reactions)

    if CONSTRAINTS in kwargs and kwargs[CONSTRAINTS]:
        kwargs[CONSTRAINTS] = parse_constraints(kwargs[CONSTRAINTS], reaction_ids)
        A_ineq, b_ineq, A_eq, b_eq = lineqlist2mat(kwargs[CONSTRAINTS], reaction_ids)

    if SOLVER not in kwargs:
        kwargs[SOLVER] = None
    solver = select_solver(kwargs[SOLVER], model)

    # prepare vectors and matrices
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

    # build LP
    lp = MILP_LP(A_ineq=A_ineq, b_ineq=b_ineq, A_eq=A_eq, b_eq=b_eq, lb=lb, ub=ub, solver=solver)
    _, _, status = lp.solve()
    if status not in [OPTIMAL, UNBOUNDED]:  # if problem not feasible or unbounded
        logging.error('FVA problem not feasible.')
        return DataFrame(
            {
                "minimum": [nan for i in range(1, 2 * numr, 2)],
                "maximum": [nan for i in range(0, 2 * numr, 2)],
            },
            index=reaction_ids,
        )

    processes = Configuration().processes
    num_reactions = len(reaction_ids)
    processes = min(processes, num_reactions)

    x = [nan] * 2 * numr

    # Dummy to check if optimization runs
    # worker_init(A_ineq,b_ineq,A_eq,b_eq,lb,ub,solver)
    # worker_compute(1)
    if processes > 1 and numr > 300:  #and solver != 'GLPK': # activate alternative routine with GLPK, if issues arise
        # with Pool(processes,initializer=worker_init,initargs=(A_ineq,b_ineq,A_eq,b_eq,lb,ub,solver)) as pool:
        with SDPool(processes, initializer=fva_worker_init, initargs=(A_ineq, b_ineq, A_eq, b_eq, lb, ub, solver)) as pool:
            chunk_size = len(reaction_ids) // processes
            # x = pool.imap_unordered(worker_compute, range(2*numr), chunksize=chunk_size)
            for i, value in pool.imap_unordered(fva_worker_compute, range(2 * numr), chunksize=chunk_size):
                x[i] = value
    # GLPK works better when reinitializing the LP in every iteration. Unfortunately, this is slow
    # but for now by far the most stable solution.
    elif processes > 1 and numr > 500 and solver == GLPK:
        with SDPool(processes, initializer=fva_worker_init_glpk, initargs=(A_ineq, b_ineq, A_eq, b_eq, lb, ub)) as pool:
            chunk_size = len(reaction_ids) // processes
            # # x = pool.imap_unordered(worker_compute, range(2*numr), chunksize=chunk_size)
            for i, value in pool.imap_unordered(fva_worker_compute_glpk, range(2 * numr), chunksize=chunk_size):
                x[i] = value
    else:
        fva_worker_init(A_ineq, b_ineq, A_eq, b_eq, lb, ub, solver)
        for i in range(2 * numr):
            _, x[i] = fva_worker_compute(i)

    x = [v if abs(v) >= 1e-11 else 0.0 for v in x]  # cut off for very small absolute values
    fva_result = DataFrame(
        {
            "minimum": [x[i] for i in range(1, 2 * numr, 2)],
            "maximum": [-x[i] for i in range(0, 2 * numr, 2)],
        },
        index=reaction_ids,
    )

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
    reaction_ids = model.reactions.list_attr("id")

    if CONSTRAINTS in kwargs:
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
        
        points (optional (int)): (Default: 25 (3D) or 40 (2D))
            The number of intervals in which the flux space should be sampled along each axis. A higher
            number will increase resoltion but also computation time.

    Returns:
        (Tuple):
            (datapoints, triang, plot1). The array of datapoints from which the plot was generated. These
            datapoints are optimal values for different optimizations within the flux space. The triang
            variable contains information about which datapoints need to be connected in triangles to
            render a consistend 3-D surface with Delaunay triangles (Delaunay triangulation). The last
            variable contains the matplotlib object.
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

    # compute points
    x_space = linspace(val_limits[0][0], val_limits[0][1], num=points).tolist()
    lb = full(points, nan)
    ub = full(points, nan)
    for i, x in enumerate(x_space):
        constr = kwargs[CONSTRAINTS].copy()
        if ax_type[0] == 'rate':
            constr += [[axes[0][0], '=', x]]
        elif ax_type[0] == 'yield':
            constr += [[{**axes[0][0], **{k: -v * x for k, v in axes[0][1].items()}}, '=', 0]]
        if ax_type[1] == 'rate':
            sol_vmin = fba(model, constraints=constr, obj=axes[1][0], solver=solver, obj_sense='minimize')
            sol_vmax = fba(model, constraints=constr, obj=axes[1][0], solver=solver, obj_sense='maximize')
        elif ax_type[1] == 'yield':
            sol_vmin = yopt(model, constraints=constr, obj_num=axes[1][0], obj_den=axes[1][1], solver=solver, obj_sense='minimize')
            sol_vmax = yopt(model, constraints=constr, obj_num=axes[1][0], obj_den=axes[1][1], solver=solver, obj_sense='maximize')
        lb[i] = ceil_dec(sol_vmin.objective_value, 9)
        ub[i] = floor_dec(sol_vmax.objective_value, 9)

    if num_axes == 2:
        datapoints = [[x, l] for x, l in zip(x_space, lb)] + [[x, u] for x, u in zip(x_space, ub)]
        datapoints_bottom = [i for i, _ in enumerate(x_space)]
        datapoints_top = [len(x_space) + i for i, _ in enumerate(x_space)]
        # triangles 2D (only for return)
        triang = []
        for i in range(len(x_space) - 1):
            temp_points = [datapoints_bottom[i], datapoints_top[i], datapoints_bottom[i + 1], datapoints_top[i + 1]]
            pts = [array(datapoints[idx_p]) for idx_p in temp_points]
            try:
                if matrix_rank(pts - pts[0]) > 1:
                    triang_temp = Delaunay(pts).simplices
                    triang += [[temp_points[idx] for idx in p] for p in triang_temp]
            except:
                logging.warning('Computing matrix rank or Delaunay simplices failed.')
        # Plot
        x = [v for v in x_space] + [v for v in reversed(x_space)]
        y = [v for v in lb] + [v for v in reversed(ub)]
        if lb[0] != ub[0]:
            x.extend([x_space[0], x_space[0]])
            y.extend([lb[0], ub[0]])
        plot1 = plt.fill(x, y, linewidth=0.5)  # edgecolor='mediumblue',
        plot1 = plot1[0]
        # plot1 = plt.plot(x, y)

        # Alternatively, this can be plotted as triangles
        # One may use this code-snippet to plot the return value of this function
        # x = [d[0] for d in datapoints]
        # y = [d[1] for d in datapoints]
        # trg = tri.Triangulation(x,y,triangles=triang)
        # colors = [1.0 for _ in trg.triangles]
        # plot1 = plt.tripcolor(trg,colors,antialiased=True,cmap='Blues_r',shading='flat')

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
        return datapoints, triang, plot1

    elif num_axes == 3:
        max_diff_y = max([abs(l - u) for l, u in zip(lb, ub)])
        datapoints = []
        datapoints_top = []
        datapoints_bottom = []
        for i, (x, l, u) in enumerate(zip(x_space.copy(), lb, ub)):
            if l != u:
                y_space = linspace(l, u, int(-(-points // (max_diff_y / abs(l - u)))))
            else:
                y_space = [l]
            datapoints_top += [[]]
            datapoints_bottom += [[]]
            for j, y in enumerate(y_space):
                constr = kwargs[CONSTRAINTS].copy()
                if ax_type[0] == 'rate':
                    constr += [[axes[0][0], '=', x]]
                elif ax_type[0] == 'yield':
                    constr += [[{**axes[0][0], **{k: -v * x for k, v in axes[0][1].items()}}, '=', 0]]
                if ax_type[1] == 'rate':
                    constr += [[axes[1][0], '=', y]]
                elif ax_type[1] == 'yield':
                    constr += [[{**axes[1][0], **{k: -v * y for k, v in axes[1][1].items()}}, '=', 0]]
                if ax_type[2] == 'rate':
                    sol_vmin = fba(model, constraints=constr, obj=axes[2][0], obj_sense='minimize')
                    sol_vmax = fba(model, constraints=constr, obj=axes[2][0], obj_sense='maximize')
                elif ax_type[2] == 'yield':
                    sol_vmin = yopt(model, constraints=constr, obj_num=axes[2][0], obj_den=axes[2][1], obj_sense='minimize')
                    sol_vmax = yopt(model, constraints=constr, obj_num=axes[2][0], obj_den=axes[2][1], obj_sense='maximize')
                datapoints_top[-1] += [len(datapoints)]
                datapoints += [array([x, y, floor_dec(sol_vmax.objective_value, 9)])]
                datapoints_bottom[-1] += [len(datapoints)]
                datapoints += [array([x, y, ceil_dec(sol_vmin.objective_value, 9)])]
            if any([isnan(datapoints[d][2]) for d in datapoints_top[-1] + datapoints_bottom[-1]]):
                logging.warning('warning: An optimization finished infeasible. Some sample points are missing.')
                datapoints_top = datapoints_top[:-1]
                datapoints_bottom = datapoints_bottom[:-1]
                x_space.remove(x)

        # Construct Denaunay triangles for plotting from all 6 perspectives
        triang = []
        # triangles top
        for i in range(len(datapoints_top) - 1):
            temp_points = datapoints_top[i] + datapoints_top[i + 1]
            pts = [array([datapoints[idx_p][0], datapoints[idx_p][1]]) for idx_p in temp_points]
            if matrix_rank(pts - pts[0]) > 1:
                triang_temp = Delaunay(pts).simplices
                triang += [[temp_points[idx] for idx in p] for p in triang_temp]
        # triangles bottom
        for i in range(len(datapoints_bottom) - 1):
            temp_points = datapoints_bottom[i] + datapoints_bottom[i + 1]
            pts = [array([datapoints[idx_p][0], datapoints[idx_p][1]]) for idx_p in temp_points]
            if matrix_rank(pts - pts[0]) > 1:
                triang_temp = Delaunay(pts).simplices
                triang += [[temp_points[idx] for idx in flip(p)] for p in triang_temp]
        # triangles front
        for i in range(len(x_space) - 1):
            temp_points = [datapoints_top[i][0], datapoints_top[i + 1][0], datapoints_bottom[i][0], datapoints_bottom[i + 1][0]]
            pts = [array([datapoints[idx_p][0], datapoints[idx_p][2]]) for idx_p in temp_points]
            if matrix_rank(pts - pts[0]) > 1:
                triang_temp = Delaunay(pts).simplices
                triang += [[temp_points[idx] for idx in flip(p)] for p in triang_temp]
        # triangles back
        for i in range(len(x_space) - 1):
            temp_points = [datapoints_top[i][-1], datapoints_top[i + 1][-1], datapoints_bottom[i][-1], datapoints_bottom[i + 1][-1]]
            pts = [array([datapoints[idx_p][0], datapoints[idx_p][2]]) for idx_p in temp_points]
            if matrix_rank(pts - pts[0]) > 1:
                triang_temp = Delaunay(pts).simplices
                triang += [[temp_points[idx] for idx in flip(p)] for p in triang_temp]
        # triangles left
        for i in range(len(datapoints_top[0]) - 1):
            temp_points = [datapoints_top[0][i], datapoints_top[0][i + 1], datapoints_bottom[0][i], datapoints_bottom[0][i + 1]]
            pts = [array([datapoints[idx_p][1], datapoints[idx_p][2]]) for idx_p in temp_points]
            if matrix_rank(pts - pts[0]) > 1:
                triang_temp = Delaunay(pts).simplices
                triang += [[temp_points[idx] for idx in p] for p in triang_temp]
        # triangles right
        for i in range(len(datapoints_top[-1]) - 1):
            temp_points = [datapoints_top[-1][i], datapoints_top[-1][i + 1], datapoints_bottom[-1][i], datapoints_bottom[-1][i + 1]]
            pts = [array([datapoints[idx_p][1], datapoints[idx_p][2]]) for idx_p in temp_points]
            if matrix_rank(pts - pts[0]) > 1:
                triang_temp = Delaunay(pts).simplices
                triang += [[temp_points[idx] for idx in p] for p in triang_temp]

        # hull = ConvexHull(datapoints)
        x = [d[0] for d in datapoints]
        y = [d[1] for d in datapoints]
        z = [d[2] for d in datapoints]
        # ax = a3.Axes3D(plt.figure())
        ax = plt.figure().add_subplot(projection='3d')
        ax.dist = 10
        ax.azim = 30
        ax.elev = 10
        ax.set_xlim(ax_limits[0])
        ax.set_ylim(ax_limits[1])
        ax.set_zlim(ax_limits[2])
        ax.set_xlabel(ax_name[0])
        ax.set_ylabel(ax_name[1])
        ax.set_zlabel(ax_name[2])
        # set higher color value for 'larger' values in all dimensions
        # use middle value of triangles instead of means and then norm
        colors = [nan for _ in triang]
        for i, t in enumerate(triang):
            p = [datapoints[i] for i in t]
            interior_point = [0.0, 0.0, 0.0]
            for d in range(3):
                v = [q[d] for q in p]
                interior_point[d] = mean([max(v), min(v)]) / max(abs(array(ax_limits[d])))
            colors[i] = sum(interior_point)  # norm(interior_point)
        lw = min([1, 6.0 / len(triang)])
        plot1 = ax.plot_trisurf(x, y, z, triangles=triang, linewidth=lw, edgecolors='black', antialiased=True,
                                alpha=0.90)  #  array=colors, cmap=plt.cm.winter
        colors = colors / max(colors)
        colors = plt.get_cmap("Spectral")(colors)
        plot1.set_fc(colors)
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
