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
#
"""Function: computing metabolic strain designs (compute_strain_designs)"""

from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, List, Tuple
import numpy as np
import logging
import json
import io
from copy import deepcopy
import logging
from cobra import Model
from cobra.manipulation import rename_genes
from straindesign import SDModule, SDSolutions, select_solver, fva, DisableLogger, SDProblem, SDMILP
from straindesign.names import *
from straindesign.networktools import   remove_ext_mets, remove_dummy_bounds, bound_blocked_or_irrevers_fva, \
                                        remove_irrelevant_genes, extend_model_gpr, extend_model_regulatory, \
                                        compress_model, compress_modules, compress_ki_ko_cost, expand_sd, filter_sd_maxcost


def compute_strain_designs(model: Model, **kwargs: dict) -> SDSolutions:
    """Computes strain designs for a user-defined strain design problem

    A number of arguments can be specified to detail the problem and influence the solution process.
    This function supports the computation of Minimal Cut Sets (MCS), OptKock, RobustKnock and OptCouple
    strain designs. It is possible to combine any of the latter ones with the MCS approach, e.g., to
    engineer growth coupled production, but also suppress the production of an undesired by-product.
    The computation can be started in two different ways. Either by specifying the computation parameters
    indivdually or reuse a parameters dictionary from a previous computation. CNApy stores strain design
    setup dics as JSON ".sd"-files that can be loaded in python and used as an input for this function.

    Example:
        sols = compute_strain_designs(model, sd_modules=[sd_module1, sd_module2], solution_approach = 'any')

    Args:
        model (cobra.Model):
            A metabolic model that is an instance of the cobra.Model class. The model may or may not
            contain genes/GPR-rules.

        sd_setup (dict):
            sd_setup should be a dictionary containing a set of parameters for strain design computation.
            The allowed keywords are the same listed hereafter. Therefore, *sd_setup and other arguments
            (except for model) must not be used together*.

        sd_modules ([straindesign.SDModule]):
            List of strain design modules that describe the sub-problems, such as the MCS-like protection
            or suppression of flux subspaces or the OptKnock, RobustKnock or OptCouple objective and
            constraints. The list of modules determines the global objective function of the strain design
            computation. If only SUPPRESS and PROTECT modules are used, the strain design computation is
            MCS-like, such that the number of interventions is minimized. If a module for one of the nested
            optimization approaches is used, the global objective function is retrieved from this module.
            The number of SUPPRESS and PROTECT modules is unrestricted and can be combined with the other
            modules, however only one of the modules OPTKNOCK, ROBUSKNOCK and OPTCOUPLE may be used at a time.
            For details, see SDModule.

        solver (optional (str)): (Default: same as defined in model / COBRApy)
            The solver that should be used for preparing and carrying out the strain design computation.
            Allowed values are 'cplex', 'gurobi', 'scip' and 'glpk'.

        max_cost (optional (int)): (Default: inf):
            The maximum cost threshold for interventions. Every possible intervention is associated with a
            cost value (1, by default). Strain designs cannot exceed the max_cost threshold. Individual
            intervention cost factors may be defined through ki_cost, ko_cost, gki_cost, gko_cost and reg_cost.

        max_solutions (optional (int)): (Default: inf)
            The maximum number of MILP solutions that are generated for a strain design problem. The number of returned
            strain designs is usually larger than the number of max_solutions, since a MILP solution is decompressed
            to multiple strain designs. When the compress-flag is set to 'False' the number of returned solutions is
            equal to max_solutions.

        M (optional (int)): (Default: None)
            If this value is specified (and non-zero, not None), the computation uses the big-M
            method instead of indicator constraints. Since GLPK does not support indicator constraints it uses
            the big-M method by default (with M=1000). M should be chosen 'sufficiently large' to avoid computational
            artifacts and 'sufficiently small' to avoid numerical issues.

        compress (optional (bool)): (Default: True)
            If 'True', the interative network compressor is used.

        gene_kos (optional (bool)): (Default: False)
            If 'True', strain designs are computed based on gene-knockouts instead of reaction knockouts. This
            parameter needs not be defined if any of ki_cost, ko_cost, gki_cost, gko_cost and reg_cost is used.
            By default, reactions are considered as knockout targets.

        ko_cost (optional (dict)): (Default: None)
            A dictionary of reaction identifiers and their associated knockout costs. If not specified, all reactions
            are treated as knockout candidates, equivalent to ko_cost = {'r1':1, 'r2':1, ...}. If a subset of reactions
            is listed in the dict, all other are not considered as knockout candidates.

        ki_cost (optional (dict)): (Default: None)
            A dictionary of reaction identifiers and their associated costs for addition. If not specified, all reactions
            are treated as knockout candidates. Reaction addition candidates must be present in the original model with
            the intended flux boundaries **after** insertion. Additions are treated adversely to knockouts, meaning that
            their exclusion from the network is not associated with any cost while their presence entails intervention costs.

        gko_cost (optional (dict)): (Default: None)
            A dictionary of gene identifiers and their associated knockout costs. To reference genes, gene IDs can be
            used,as well as gene names. If not specified, genes are not treated as knockout candidates. An exception is
            the 'gene_kos' argument. If 'gene_kos' is used, all genes are treated as knockout candidates with intervention
            costs of 1. This is equivalent to gko_cost = {'g1':1, 'g2':1, ...}.

        gki_cost (optional (dict)): (Default: None)
            A dictionary of gene identifiers and their associated addition costs. To reference genes, gene IDs can be
            used, as well as gene names. If not specified, none of the genes are treated as addition candidates.

        reg_cost (optional [dict]): ( Default: None)
            Regulatory interventions candidates can be optionally specified as a list. Thereby, the constraint marking the
            regulatory intervention is put as key and the associated intervention cost is used as the corresponding value.
            E.g., reg_cost = {'1 EX_o2_e = -1': 1, ... <other regulatory interventions>}. Instead of strings, constraints
            can also be passed as lists. reg_cost = {[{'EX_o2_e':1}, '=', -1]: 1, ...}

        solution_approach (optional (str)): ( Default: 'best')
            The approach used to find strain designs. Possible values are 'any', 'best' or 'populate'. 'any' is usually the
            fastest option, since optimality is not enforced. Hereby computed MCS are still irreducible intervention sets,
            however, not MCS with the fewest possible number of interventions. 'best' computes globally optimal strain designs,
            that is, MCS with the fewest number of interventions, OptKnock strain designs with the highest possible production
            rate, OptCouple strain designs with the hightest growth coupling potential etc.. 'populate' does the same as 'best',
            but makes use of CPLEX' and Gurobi's populate function to generate multiple strain designs. It is identical to 'best'
            when used with SCIP or GLPK.
            Attention:
            If 'any' used with OptKnock, for instance, the MILP may return the wild type as a possible immediately. Technically,
            the wiltype fulfills the criterion of maximal growth (inner objective) and maximality of the global objective is
            omitted by using 'any', so that carrying no product synthesis is permitted. Additional constraints can be used
            in the OptKnock problem to circumvent this. However, Optknock should generally be used with the 'best' option.

        time_limit (optional (int)): (Default: inf)
            The time limit in seconds for the MILP-solver.

        advanced, use_scenario (optional (bool)):
            Dummy parameters used for the CNApy interface.

    Returns:
        (SDSolutions):

            An object that contains all computed strain designs. If strain designs were computed
            as gene-interventions, the solution object contains a set of corresponding reaction-interventions
            that facilitate the analysis of the computed strain designs with COBRA methods.
    """
    allowed_keys = {
        MODULES, SETUP, SOLVER, MAX_COST, MAX_SOLUTIONS, 'M', 'compress', 'gene_kos', KOCOST, KICOST, GKOCOST, GKICOST, REGCOST,
        SOLUTION_APPROACH, 'advanced', 'use_scenario', T_LIMIT, SEED
    }
    logging.info('Preparing strain design computation.')
    if SETUP in kwargs:
        if type(kwargs[SETUP]) is str:
            with open(kwargs[SETUP], 'r') as fs:
                kwargs = json.load(fs)
        else:
            kwargs = kwargs[SETUP]

    if MODULES in kwargs:
        sd_modules = kwargs.pop(MODULES)
        if isinstance(sd_modules, SDModule):
            sd_modules = [sd_modules]
        orig_sd_modules = [m.copy() for m in sd_modules]
        sd_modules = [m.copy() for m in sd_modules]

    if SOLVER in kwargs:
        kwargs[SOLVER] = select_solver(kwargs[SOLVER])
    else:
        kwargs[SOLVER] = select_solver(None, model)

    if MAX_COST in kwargs:
        kwargs.update({MAX_COST: float(kwargs.pop(MAX_COST))})
    else:
        kwargs.update({MAX_COST: np.inf})

    if MODEL_ID in kwargs:
        model_id = kwargs.pop(MODEL_ID)
        if model_id != model.id:
            logging.warning( "Model IDs of provided model and setup not matching. Apparently, ",\
                          "the strain design setup was specified for a different model. "+\
                          "Errors might occur due to non-matching reaction or gene-identifiers.")

    if 'advanced' in kwargs:
        model_id = kwargs.pop('advanced')
    if 'use_scenario' in kwargs:
        model_id = kwargs.pop('use_scenario')

    if SEED not in kwargs:
        kwargs[SEED] = int(np.random.default_rng().integers(1, 2**16 - 1))
        logging.info("  Using random seed " + str(kwargs[SEED]))
    else:
        logging.info("  Using seed " + str(kwargs[SEED]))

    # check all keys passed in kwargs
    for key, value in dict(kwargs).items():
        if key not in allowed_keys:
            raise Exception("Key " + key + " is not supported.")
        if key == KOCOST:
            uncmp_ko_cost = value
        if key == KICOST:
            uncmp_ki_cost = value
        if key == GKOCOST:
            uncmp_gko_cost = value
        if key == GKICOST:
            uncmp_gki_cost = value
        if key == REGCOST:
            uncmp_reg_cost = value
    if (GKOCOST in kwargs or GKICOST in kwargs or
        ('gene_kos' in kwargs and kwargs['gene_kos'])) and hasattr(model, 'genes') and model.genes:
        kwargs['gene_kos'] = True
        used_g_ids = set(kwargs[GKOCOST] if GKOCOST in kwargs and kwargs[GKOCOST] else set())
        used_g_ids.update(set(kwargs[GKICOST] if GKICOST in kwargs and kwargs[GKICOST] else set()))
        used_g_ids.update(set(kwargs[REGCOST] if REGCOST in kwargs and kwargs[REGCOST] else set()))
        # genes must not begin with number, put a 'g' in front of genes that start with a number
        if any([True for g in model.genes if g.id[0].isdigit()]):
            logging.warning("Gene IDs must not start with a digit. Inserting prefix 'g' where necessary.")
            rename_genes(model, {g.id: 'g' + g.id for g in model.genes if g.id[0].isdigit()})
        if np.all([len(g.name) for g in model.genes]) and (np.any([g.name in used_g_ids for g in model.genes]) or not used_g_ids):
            has_gene_names = True
        else:
            has_gene_names = False
        if has_gene_names and any([True for g in model.genes if g.name[0].isdigit()]):
            logging.warning("Gene names must not start with a digit. Inserting prefix 'g' where necessary.")
            for g, v in {g.id: 'g' + g.name for g in model.genes if g.name[0].isdigit()}.items():
                model.genes.get_by_id(g).name = v
        if GKOCOST not in kwargs or not kwargs[GKOCOST]:
            if has_gene_names:  # if gene names are defined, use them instead of ids
                uncmp_gko_cost = {k: 1.0 for k in model.genes.list_attr('name')}
            else:
                uncmp_gko_cost = {k: 1.0 for k in model.genes.list_attr('id')}
        if GKICOST not in kwargs or not kwargs[GKICOST]:
            uncmp_gki_cost = {}
    else:
        kwargs['gene_kos'] = False
        has_gene_names = False
    if KOCOST not in kwargs and not kwargs['gene_kos']:
        uncmp_ko_cost = {k: 1.0 for k in model.reactions.list_attr('id')}
    elif KOCOST not in kwargs or not kwargs[KOCOST]:
        uncmp_ko_cost = {}
    if KICOST not in kwargs or not kwargs[KICOST]:
        uncmp_ki_cost = {}
    if REGCOST not in kwargs or not kwargs[REGCOST]:
        uncmp_reg_cost = {}
    # check that at most one bilevel module is provided
    bilvl_modules = [i for i,m in enumerate(sd_modules) \
                if m[MODULE_TYPE] in [OPTKNOCK,ROBUSTKNOCK,OPTCOUPLE]]
    if len(bilvl_modules) > 1:
        raise Exception("Only one of the module types 'OptKnock', 'RobustKnock' and 'OptCouple' can be defined per "\
                            "strain design setup.")
    logging.info('  Using ' + kwargs[SOLVER] + ' for solving LPs during preprocessing.')
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), DisableLogger():  # suppress standard output from copying model
        orig_model = model
        model = model.copy()
        uncmp_model = model.copy()
    orig_ko_cost = deepcopy(uncmp_ko_cost)
    orig_ki_cost = deepcopy(uncmp_ki_cost)
    orig_reg_cost = deepcopy(uncmp_reg_cost)
    if 'compress' not in kwargs:
        kwargs['compress'] = True
    if kwargs['gene_kos']:
        orig_gko_cost = uncmp_gko_cost
        orig_gki_cost = uncmp_gki_cost
        # ensure that gene and reaction kos/kis do not overlap
        g_itv = {g for g in list(uncmp_gko_cost.keys()) + list(uncmp_gki_cost.keys())}
        r_itv = {r for r in list(uncmp_ko_cost.keys()) + list(uncmp_ki_cost.keys())}
        if np.any([np.any([True for g in uncmp_model.reactions.get_by_id(r).genes if g in g_itv]) for r in r_itv]) or \
            np.any(set(uncmp_gko_cost.keys()).intersection(set(uncmp_gki_cost.keys()))) or \
            np.any(set(uncmp_ko_cost.keys()).intersection(set(uncmp_ki_cost.keys()))):
            raise Exception('Specified gene and reaction knock-out/-in costs contain overlap. '\
                            'Make sure that metabolic interventions are enabled either through reaction or '\
                            'through gene interventions and are defined either as knock-ins or as knock-outs.')
    # 1) Preprocess Model
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), DisableLogger():  # suppress standard output from copying model
        cmp_model = uncmp_model.copy()
    # remove external metabolites
    remove_ext_mets(cmp_model)
    # replace model bounds with +/- inf if above a certain threshold
    remove_dummy_bounds(model)
    # FVAs to identify blocked, irreversible and essential reactions, as well as non-bounding bounds
    logging.info('  FVA to identify blocked reactions and irreversibilities.')
    bound_blocked_or_irrevers_fva(model, solver=kwargs[SOLVER])
    logging.info('  FVA(s) to identify essential reactions.')
    essential_reacs = set()
    for m in sd_modules:
        if m[MODULE_TYPE] != SUPPRESS:  # Essential reactions can only be determined from desired
            # or opt-/robustknock modules
            flux_limits = fva(cmp_model, solver=kwargs[SOLVER], constraints=m[CONSTRAINTS])
            for (reac_id, limits) in flux_limits.iterrows():
                if np.min(abs(limits)) > 1e-10 and np.prod(np.sign(limits)) > 0:  # find essential
                    essential_reacs.add(reac_id)
    # remove ko-costs (and thus knockability) of essential reactions
    [uncmp_ko_cost.pop(er) for er in essential_reacs if er in uncmp_ko_cost]
    # If computation of gene-gased intervention strategies, (optionally) compress gpr are rules and extend stoichimetric network with genes
    if kwargs['gene_kos']:
        if kwargs['compress'] is True or kwargs['compress'] is None:
            num_genes = len(cmp_model.genes)
            num_gpr = len([True for r in model.reactions if r.gene_reaction_rule])
            logging.info('Preprocessing GPR rules (' + str(num_genes) + ' genes, ' + str(num_gpr) + ' gpr rules).')
            # removing irrelevant genes will also remove essential reactions from the list of knockable genes
            uncmp_gko_cost = remove_irrelevant_genes(cmp_model, essential_reacs, uncmp_gki_cost, uncmp_gko_cost)
            if len(cmp_model.genes) < num_genes or len([True for r in model.reactions if r.gene_reaction_rule]) < num_gpr:
                num_genes = len(cmp_model.genes)
                num_gpr = len([True for r in cmp_model.reactions if r.gene_reaction_rule])
                logging.info('  Simplifyied to '+str(num_genes)+' genes and '+\
                    str(num_gpr)+' gpr rules.')
        logging.info('  Extending metabolic network with gpr associations.')
        reac_map = extend_model_gpr(cmp_model, has_gene_names)
        for i, m in enumerate(sd_modules):
            for p in [CONSTRAINTS, INNER_OBJECTIVE, OUTER_OBJECTIVE, PROD_ID]:
                if p in m and m[p] is not None:
                    if p == CONSTRAINTS:
                        for c in m[p]:
                            for k in list(c[0].keys()):
                                v = c[0].pop(k)
                                for n, w in reac_map[k].items():
                                    c[0][n] = v * w
                    if p in [INNER_OBJECTIVE, OUTER_OBJECTIVE, PROD_ID]:
                        for k in list(m[p].keys()):
                            v = m[p].pop(k)
                            for n, w in reac_map[k].items():
                                m[p][n] = v * w
        uncmp_ko_cost.update(uncmp_gko_cost)
        uncmp_ki_cost.update(uncmp_gki_cost)
    uncmp_ko_cost.update(extend_model_regulatory(cmp_model, uncmp_reg_cost))
    cmp_ko_cost = uncmp_ko_cost
    cmp_ki_cost = uncmp_ki_cost
    # Compress model
    if kwargs['compress'] is True or kwargs['compress'] is None:  # If compression is activated (or not defined)
        logging.info('Compressing Network (' + str(len(cmp_model.reactions)) + ' reactions).')
        # compress network by lumping sequential and parallel reactions alternatingly.
        # Exclude reactions named in strain design modules from parallel compression
        no_par_compress_reacs = set()
        for m in sd_modules:
            for p in [CONSTRAINTS, INNER_OBJECTIVE, OUTER_OBJECTIVE, PROD_ID]:
                if p in m and m[p] is not None:
                    param = m[p]
                    if p == CONSTRAINTS:
                        for c in param:
                            for k in c[0].keys():
                                no_par_compress_reacs.add(k)
                    if p in [INNER_OBJECTIVE, OUTER_OBJECTIVE, PROD_ID]:
                        for k in param.keys():
                            no_par_compress_reacs.add(k)
        cmp_mapReac = compress_model(cmp_model, no_par_compress_reacs)
        # compress information in strain design modules
        sd_modules = compress_modules(sd_modules, cmp_mapReac)
        # compress ko_cost and ki_cost
        cmp_ko_cost, cmp_ki_cost, cmp_mapReac = compress_ki_ko_cost(cmp_ko_cost, cmp_ki_cost, cmp_mapReac)
    else:
        cmp_mapReac = []

    # An FVA to identify essentials before building and launching MILP (not sure if this has an effect)
    logging.info('  FVA(s) in compressed model to identify essential reactions.')
    essential_reacs = set()
    for m in sd_modules:
        if m[MODULE_TYPE] != SUPPRESS:  # Essential reactions can only be determined from desired
            # or opt-/robustknock modules
            flux_limits = fva(cmp_model, solver=kwargs[SOLVER], constraints=m[CONSTRAINTS])
            for (reac_id, limits) in flux_limits.iterrows():
                if np.min(abs(limits)) > 1e-10 and np.prod(np.sign(limits)) > 0:  # find essential
                    essential_reacs.add(reac_id)
    # remove ko-costs (and thus knockability) of essential reactions
    [cmp_ko_cost.pop(er) for er in essential_reacs if er in cmp_ko_cost]
    essential_kis = set(cmp_ki_cost[er] for er in essential_reacs if er in cmp_ki_cost)
    # Build MILP
    kwargs1 = kwargs
    kwargs1[KOCOST] = cmp_ko_cost
    kwargs1[KICOST] = cmp_ki_cost
    kwargs1['essential_kis'] = essential_kis
    kwargs1.pop('compress')
    if GKOCOST in kwargs1:
        kwargs1.pop(GKOCOST)
    if GKICOST in kwargs1:
        kwargs1.pop(GKICOST)
    if REGCOST in kwargs1:
        kwargs1.pop(REGCOST)

    kwargs_milp = {k: v for k, v in kwargs.items() if k in [SOLVER, MAX_COST, 'M', SEED]}
    kwargs_milp.update({KOCOST: cmp_ko_cost})
    kwargs_milp.update({KICOST: cmp_ki_cost})
    kwargs_milp.update({'essential_kis': essential_kis})
    logging.info("Finished preprocessing:")
    logging.info("  Model size: " + str(len(cmp_model.reactions)) + " reactions, " + str(len(cmp_model.metabolites)) + " metabolites")
    logging.info("  " + str(len(cmp_ko_cost) + len(cmp_ki_cost) - len(essential_kis)) + " targetable reactions")

    sd_milp = SDMILP(cmp_model, sd_modules, **kwargs_milp)

    kwargs_computation = {}
    if MAX_SOLUTIONS in kwargs:
        kwargs_computation.update({MAX_SOLUTIONS: float(kwargs.pop(MAX_SOLUTIONS))})
    if T_LIMIT in kwargs:
        kwargs_computation.update({T_LIMIT: float(kwargs.pop(T_LIMIT))})
    kwargs_computation.update({'show_no_ki': True})

    # solution approach
    if SOLUTION_APPROACH in kwargs:
        solution_approach = kwargs.pop(SOLUTION_APPROACH)
    else:
        solution_approach = BEST

    # solve MILP
    if solution_approach == ANY:
        cmp_sd_solution = sd_milp.compute(**kwargs_computation)
    elif solution_approach == BEST:
        cmp_sd_solution = sd_milp.compute_optimal(**kwargs_computation)
    elif solution_approach == POPULATE:
        cmp_sd_solution = sd_milp.enumerate(**kwargs_computation)

    logging.info('  Decompressing.')
    if cmp_sd_solution.status in [OPTIMAL, TIME_LIMIT_W_SOL]:
        sd = expand_sd(cmp_sd_solution.get_reaction_sd_mark_no_ki(), cmp_mapReac)
        sd = filter_sd_maxcost(sd, kwargs[MAX_COST], uncmp_ko_cost, uncmp_ki_cost)
        sd = postprocess_reg_sd(uncmp_reg_cost, sd)
    else:
        sd = []

    setup = deepcopy(cmp_sd_solution.sd_setup)
    setup.update({MODULES: orig_sd_modules, KOCOST: orig_ko_cost, KICOST: orig_ki_cost, REGCOST: orig_reg_cost})
    if kwargs['gene_kos']:
        setup.update({GKOCOST: orig_gko_cost, GKICOST: orig_gki_cost})
    sd_solutions = SDSolutions(orig_model, sd, cmp_sd_solution.status, setup)
    logging.info(str(len(sd)) + ' solutions found.')

    return sd_solutions


def postprocess_reg_sd(reg_cost, sd):
    """Postprocess regulatory interventions

    Mark regulatory interventions with true or false"""
    for s in sd:
        for k, v in reg_cost.items():
            if k in s:
                s.pop(k)
                s.update({v['str']: True})
            else:
                s.update({v['str']: False})
    return sd
