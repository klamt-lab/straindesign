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

from typing import Dict, List, Tuple
import numpy as np
import logging
import json
import time
from copy import deepcopy
from cobra import Model
from cobra.manipulation import rename_genes
from straindesign import SDModule, SDSolutions, select_solver, fva, DisableLogger, SDProblem, SDMILP
from straindesign.names import *
from straindesign.networktools import   remove_ext_mets, bound_blocked_or_irrevers_fva, \
                                        reduce_gpr, extend_model_gpr, extend_model_regulatory, \
                                        compress_model, compress_modules, compress_ki_ko_cost, expand_sd, filter_sd_maxcost, \
                                        estimate_expansion_size, with_suppressed_lp, _silent_io, copy_model_suppressed
from straindesign.gpr_bitmask import simplify_model_gprs


def _restore_module_coeff_scaling(cmp_model, sd_modules, cmp_mapReac, orig_sd_modules):
    """Revert compression-induced coefficient shifts in strain-design module constraints.

    Lumping folds several reactions into one and scales the module-constraint coefficient of the
    lumped reaction by the lumping ratio (see compress_modules). When a target reaction (e.g.
    biomass) is lumped, that ratio can be large (4484x on iML1515), so ``biomass >= 0.001`` becomes
    ``4484*v_lumped >= 0.001`` -- an effective per-variable threshold of ~2e-7, below LP feasibility
    tolerance. That sub-tolerance value is a COMPRESSION ARTIFACT: it does not exist in the original
    model, where the coefficient was ~1 and the small numbers lived in the biomass stoichiometry.

    This function undoes that shift. For each compressed reaction appearing in a module constraint it
    rescales the variable ``v' = s * v`` with ``s = |compressed coef| / max|original coef|`` so the
    constraint coefficient returns to the largest coefficient the reaction had in the ORIGINAL
    constraint. The transform is exact (a change of variable units): the reaction's stoichiometry
    column is divided by ``s``, its finite bounds multiplied by ``s``, and every module coefficient on
    it divided by ``s`` -- so the feasible set and every strain design are unchanged. The small
    numbers move back into the stoichiometry (Sv=0), exactly where the original model carried them.

    The rescaling is recorded as a 1:1 pseudo-step appended to ``cmp_mapReac`` for completeness;
    ``expand_sd`` treats a single-reaction map as knockout-identity, so decompression is unaffected.

    NB operates on ``cmp_model._metabolites`` / ``_lower_bound`` / ``_upper_bound`` directly: the
    compressed model's optlang solver is stale, so ``add_metabolites`` would raise. Kept exact-rational
    (Fraction) throughout, consistent with exact-nullspace compression.

    Called once, right after ``cmp_mapReac = cmp_mapReac_1 + cmp_mapReac_2``.
    """
    from fractions import Fraction

    def _frac(x):
        if isinstance(x, Fraction):
            return x
        try:
            return Fraction(x)
        except Exception:
            return Fraction(float(x)).limit_denominator(10**12)

    def _expand(reac):
        cur = {reac}
        for exp in cmp_mapReac[::-1]:
            rme = exp["reac_map_exp"]
            cur = set().union(*[set(rme[r].keys()) if r in rme else {r} for r in cur])
        return cur

    # largest |coef| each reaction carried in any ORIGINAL module constraint
    orig_coef = {}
    for m in orig_sd_modules:
        for c in (m[CONSTRAINTS] or []):
            for k, v in c[0].items():
                orig_coef[k] = max(orig_coef.get(k, Fraction(0)), abs(_frac(v)))

    # scale per compressed reaction (consistent across constraints -- same lumping)
    scales = {}
    for m in sd_modules:
        for c in (m[CONSTRAINTS] or []):
            for R, C in c[0].items():
                targets = [orig_coef[o] for o in _expand(R) if orig_coef.get(o, 0) != 0]
                if not targets:
                    continue
                s = abs(_frac(C)) / max(targets)
                if s != 1:
                    scales[R] = s
    if not scales:
        return sd_modules, cmp_mapReac

    for R, s in scales.items():
        r = cmp_model.reactions.get_by_id(R)
        inv = Fraction(1) / s
        for met in list(r._metabolites.keys()):
            r._metabolites[met] = r._metabolites[met] * inv          # column /= s
        if r._lower_bound not in (float('inf'), float('-inf')):
            r._lower_bound = r._lower_bound * s                      # finite bounds *= s
        if r._upper_bound not in (float('inf'), float('-inf')):
            r._upper_bound = r._upper_bound * s
    for m in sd_modules:
        for c in (m[CONSTRAINTS] or []):
            for R, s in scales.items():
                if R in c[0]:
                    c[0][R] = _frac(c[0][R]) / s
        for p in [INNER_OBJECTIVE, OUTER_OBJECTIVE, PROD_ID]:
            if m.get(p):
                for R, s in scales.items():
                    if R in m[p]:
                        m[p][R] = _frac(m[p][R]) / s
    cmp_mapReac.append({"reac_map_exp": {R: {R: s} for R, s in scales.items()},
                        "parallel": False, KOCOST: {}, KICOST: {}})
    logging.info('  Reverted compression coeff shift on %d module reaction(s) '
                 '(largest scale %.4g).' % (len(scales), float(max(scales.values()))))
    return sd_modules, cmp_mapReac


def _collect_no_par_compress_reacs(sd_modules):
    """Collect reaction IDs referenced in SD modules that must not be parallel-compressed."""
    reacs = set()
    for m in sd_modules:
        for p in [CONSTRAINTS, INNER_OBJECTIVE, OUTER_OBJECTIVE, PROD_ID]:
            if p in m and m[p] is not None:
                param = m[p]
                if p == CONSTRAINTS:
                    for c in param:
                        for k in c[0].keys():
                            reacs.add(k)
                if p in [INNER_OBJECTIVE, OUTER_OBJECTIVE, PROD_ID]:
                    for k in param.keys():
                        reacs.add(k)
    return reacs


@with_suppressed_lp
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
        SOLUTION_APPROACH, 'advanced', 'use_scenario', T_LIMIT, SEED, MILP_THREADS, 'compression_backend', 'dump_preprocessed',
        'enum_method'
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
                if m[MODULE_TYPE] in [OPTKNOCK,ROBUSTKNOCK,OPTCOUPLE,DOUBLEOPT]]
    if len(bilvl_modules) > 1:
        raise Exception("Only one of the module types 'OptKnock', 'RobustKnock', 'OptCouple' and 'DoubleOpt' can be defined per "\
                            "strain design setup.")
    # Validate module constraints with the selected solver (the SDModule
    # constructor validates with the model's default solver, which may differ).
    from straindesign import fba as _fba
    for m in sd_modules:
        if m[CONSTRAINTS]:
            if _fba(model, constraints=m[CONSTRAINTS], solver=kwargs[SOLVER]).status == INFEASIBLE:
                raise Exception("There is no feasible solution of the model under the given constraints.")
    logging.info('  Using ' + kwargs[SOLVER] + ' for solving LPs during preprocessing.')
    with _silent_io():
        orig_model = model
        model = copy_model_suppressed(model)
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
        if np.any([np.any([True for g in model.reactions.get_by_id(r).genes if g in g_itv]) for r in r_itv]) or \
            np.any(set(uncmp_gko_cost.keys()).intersection(set(uncmp_gki_cost.keys()))) or \
            np.any(set(uncmp_ko_cost.keys()).intersection(set(uncmp_ki_cost.keys()))):
            raise Exception('Specified gene and reaction knock-out/-in costs contain overlap. '\
                            'Make sure that metabolic interventions are enabled either through reaction or '\
                            'through gene interventions and are defined either as knock-ins or as knock-outs.')
    # 1) Preprocess Model
    # Copy model for compression/processing
    with _silent_io():
        cmp_model = copy_model_suppressed(model)
    # remove external metabolites
    remove_ext_mets(cmp_model)
    # Extend with regulatory constraints: reaction-based can be applied now,
    # gene-based must be deferred until after GPR extension.
    # extend_model_regulatory mutates its dict arg in-place (replacing original
    # keys with generated names), so we must update uncmp_reg_cost accordingly.
    _deferred_reg = {}
    if uncmp_reg_cost:
        from straindesign.parse_constr import parse_constraints as _parse_constr
        _rxn_ids = set(cmp_model.reactions.list_attr('id'))
        _immediate_reg = {}
        for k, v in uncmp_reg_cost.items():
            try:
                _parse_constr(k, _rxn_ids)
                _immediate_reg[k] = v
            except Exception:
                _deferred_reg[k] = v
        if _immediate_reg:
            uncmp_ko_cost.update(extend_model_regulatory(cmp_model, _immediate_reg))
        # Rebuild uncmp_reg_cost: immediate entries are now mutated, deferred are unchanged
        uncmp_reg_cost.clear()
        uncmp_reg_cost.update(_immediate_reg)
    # --- COMPRESS #1: on model WITHOUT gene pseudoreactions ---
    if kwargs['compress'] is True or kwargs['compress'] is None:
        no_par_compress_reacs = _collect_no_par_compress_reacs(sd_modules)
        # Keep reactions controlled by a gene that carries a REGULATORY intervention intact
        # through COMPRESS#1 (exempt from merging). Otherwise, if a gene controls several
        # reactions that get merged before GPR integration, the merged reaction is hooked to
        # the gene with the wrong (collapsed) stoichiometry, so a gene-regulatory bound
        # (g <= X / g >= X) is mis-scaled vs the uncompressed model. They are merged correctly
        # in COMPRESS#2 once the g_gene metabolite exists. Gene KOs (=0) and gene KIs
        # (unbounded when added) are unaffected, so only regulatory genes need protecting.
        no_coupled_compress_reacs = set()
        if _deferred_reg:
            import re as _re
            _gene_by_id = {g.id: g for g in cmp_model.genes}
            _gene_by_name = {g.name: g for g in cmp_model.genes if g.name}
            for _constr in _deferred_reg:
                for _tok in _re.findall(r'[A-Za-z_][\w]*', _constr):
                    _g = _gene_by_id.get(_tok) or _gene_by_name.get(_tok)
                    if _g is not None:
                        no_coupled_compress_reacs.update(r.id for r in _g.reactions)
            # also exempt them from parallel merging so their names stay stable across the
            # compression passes (keeps the coupled-exemption matching them by name)
            no_par_compress_reacs.update(no_coupled_compress_reacs)
        compression_backend = kwargs.get('compression_backend', 'sparse_rref')
        # --- Reversibility pre-tightening (BEFORE compress #1) ---
        # Exact per-reaction reversibility (sign-only FVA, faster than full FVA): fix lb/ub to 0
        # for directions that carry no flux in the base polytope. Design-neutral (a base-infeasible
        # direction stays infeasible under any added module constraint -- same tightening SD already
        # applies after compress #2 at bound_blocked_or_irrevers_fva, just moved up). Doing it here
        # lets compress #1 fuse the now-one-directional reactions and, since the GPR fwd/rev split
        # fires on lb<0, avoids splitting genuinely irreversible reactions.
        from straindesign.speedy_fva import fast_reversibility
        t0 = time.time()
        _rev = fast_reversibility(cmp_model, solver=kwargs[SOLVER])
        _n_tight = 0
        for r in cmp_model.reactions:
            can_fwd, can_rev = _rev[r.id]
            if not can_fwd and float(r._upper_bound) > 0.0:
                r._upper_bound = min(0.0, float(r._upper_bound)); _n_tight += 1
            if not can_rev and float(r._lower_bound) < 0.0:
                r._lower_bound = max(0.0, float(r._lower_bound)); _n_tight += 1
        logging.info('  Reversibility pre-tightening fixed %d reaction directions (%.1fs).'
                     % (_n_tight, time.time() - t0))
        logging.info('Compressing Network (' + str(len(cmp_model.reactions)) + ' reactions).')
        t0 = time.time()
        cmp_mapReac_1 = compress_model(cmp_model, no_par_compress_reacs,
                                        compression_backend=compression_backend,
                                        propagate_gpr=True,
                                        no_coupled_compress_reacs=no_coupled_compress_reacs)
        sd_modules = compress_modules(sd_modules, cmp_mapReac_1)
        # Compress reaction + regulatory costs only (gene costs not yet added)
        cmp_ko_cost, cmp_ki_cost, cmp_mapReac_1 = compress_ki_ko_cost(
            uncmp_ko_cost, uncmp_ki_cost, cmp_mapReac_1)
        logging.info('  Compressed to ' + str(len(cmp_model.reactions)) + ' reactions (%.1fs).' % (time.time() - t0))
    else:
        cmp_mapReac_1 = []
        cmp_ko_cost = uncmp_ko_cost
        cmp_ki_cost = uncmp_ki_cost
    # --- FVAs on (possibly compressed) model ---
    logging.info('  FVA(s) to identify essential reactions.')
    essential_reacs = set()
    for m in sd_modules:
        if m[MODULE_TYPE] != SUPPRESS:  # Essential reactions can only be determined from desired
            # or opt-/robustknock modules
            flux_limits = fva(cmp_model, solver=kwargs[SOLVER], constraints=m[CONSTRAINTS], compress=False)
            for (reac_id, limits) in flux_limits.iterrows():
                if np.min(abs(limits)) > 1e-10 and np.prod(np.sign(limits)) > 0:  # find essential
                    essential_reacs.add(reac_id)
    # remove ko-costs (and thus knockability) of essential reactions
    [cmp_ko_cost.pop(er) for er in essential_reacs if er in cmp_ko_cost]
    # --- GPR extension on (possibly compressed) model ---
    if kwargs['gene_kos']:
        if kwargs['compress'] is True or kwargs['compress'] is None:
            num_genes = len(cmp_model.genes)
            num_gpr = len([True for r in cmp_model.reactions if r.gene_reaction_rule])
            logging.info('Preprocessing GPR rules (' + str(num_genes) + ' genes, ' + str(num_gpr) + ' gpr rules).')
            # removing irrelevant genes will also remove essential reactions from the list of knockable genes
            uncmp_gko_cost = reduce_gpr(cmp_model, essential_reacs, uncmp_gki_cost, uncmp_gko_cost)
            if len(cmp_model.genes) < num_genes or len([True for r in cmp_model.reactions if r.gene_reaction_rule]) < num_gpr:
                num_genes = len(cmp_model.genes)
                num_gpr = len([True for r in cmp_model.reactions if r.gene_reaction_rule])
                logging.info('  Simplified to ' + str(num_genes) + ' genes and ' +
                    str(num_gpr) + ' gpr rules.')
        # Leaf-minimize the GPR rules before building the pseudo-reaction gadget. Monotone
        # boolean-equivalent rewrite (designs unchanged), so it always runs for gene-based
        # (gMCS) computations to shrink the gadget extend_model_gpr generates.
        simplify_model_gprs(cmp_model)
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
        # Apply deferred regulatory constraints (gene-based, need GPR extension first)
        if _deferred_reg:
            reg_costs = extend_model_regulatory(cmp_model, _deferred_reg)
            cmp_ko_cost.update(reg_costs)
            uncmp_ko_cost.update(reg_costs)
            uncmp_reg_cost.update(_deferred_reg)  # now mutated by extend_model_regulatory
            _deferred_reg = {}  # prevent double application
        # Merge gene costs into compressed costs (and uncompressed for filter_sd_maxcost)
        cmp_ko_cost.update(uncmp_gko_cost)
        cmp_ki_cost.update(uncmp_gki_cost)
        uncmp_ko_cost.update(uncmp_gko_cost)
        uncmp_ki_cost.update(uncmp_gki_cost)
    # Apply any deferred regulatory constraints that weren't handled in the gene_kos block
    if _deferred_reg:
        reg_costs = extend_model_regulatory(cmp_model, _deferred_reg)
        cmp_ko_cost.update(reg_costs)
        uncmp_ko_cost.update(reg_costs)
        uncmp_reg_cost.update(_deferred_reg)  # now mutated by extend_model_regulatory
    # --- COMPRESS #2: after GPR extension ---
    if kwargs['compress'] is True or kwargs['compress'] is None:
        logging.info('Compressing after GPR extension (' + str(len(cmp_model.reactions)) + ' reactions).')
        t0 = time.time()
        no_par_compress_reacs = _collect_no_par_compress_reacs(sd_modules)
        cmp_mapReac_2 = compress_model(cmp_model, no_par_compress_reacs,
                                        compression_backend=compression_backend)
        sd_modules = compress_modules(sd_modules, cmp_mapReac_2)
        cmp_ko_cost, cmp_ki_cost, cmp_mapReac_2 = compress_ki_ko_cost(
            cmp_ko_cost, cmp_ki_cost, cmp_mapReac_2)
        cmp_mapReac = cmp_mapReac_1 + cmp_mapReac_2
        # Undo compression-induced coefficient shifts in module constraints. Exact change of
        # variable units; design set unchanged, small numbers returned to the stoichiometry.
        sd_modules, cmp_mapReac = _restore_module_coeff_scaling(cmp_model, sd_modules, cmp_mapReac, orig_sd_modules)
        logging.info('  Compressed to ' + str(len(cmp_model.reactions)) + ' reactions (%.1fs).' % (time.time() - t0))
    else:
        cmp_mapReac = []

    # --- FVA to set irreversibility and remove non-binding bounds ---
    # Runs after compress #2 so ALL reactions (including GPR-added) are processed.
    logging.info('  FVA to identify blocked reactions and irreversibilities.')
    t0 = time.time()
    # Save pre-FVA bounds for dump_preprocessed (bound config experiments)
    pre_fva_bounds = {r.id: (r.lower_bound, r.upper_bound) for r in cmp_model.reactions}
    # Subset-scope the bound-stripping FVA off reactions already at (0,+inf). 
    # FVA on such a reaction can only ever tighten a genuinely blocked one 
    # to (0,0); leaving it at (0,+inf) changes no feasible flux (the stoichiometry 
    # already forces it to 0) and a blocked knockable can never belong to a minimal cut set.
    _fva_scope = [r.id for r in cmp_model.reactions
                  if not (float(r.lower_bound) == 0.0
                          and np.isinf(float(r.upper_bound))
                          and float(r.upper_bound) > 0)]
    bound_blocked_or_irrevers_fva(cmp_model, solver=kwargs[SOLVER], compress=False,
                                  reaction_list=_fva_scope)
    logging.info('  FVA done (%.1fs).' % (time.time() - t0))

    # FVA to identify essential reactions and size-1 MCS before building MILP
    logging.info('  FVA(s) in compressed model to identify essential reactions.')
    essential_reacs = set()
    suppress_essential = set()
    cmp_size1_mcs = []
    # FVA over each module's region, scoped to knockable reactions. The ranges serve two purposes:
    # (1) essentiality for size-1 MCS detection, and (2) region-FVA subproblem tightening, read back
    # in SDMILP -- so the separate region FVA in strainDesignProblem is no longer needed. flux_limits
    # is stored on the module and flows to SDMILP via sd_modules. Scoping to knockable reactions keeps
    # the LP count down (and only knockable reactions carry z-links to tighten anyway).
    knockable_ids = list(set(cmp_ko_cost.keys()) | set(cmp_ki_cost.keys()))
    for m in sd_modules:
        flux_limits = fva(cmp_model, solver=kwargs[SOLVER], constraints=m[CONSTRAINTS],
                          compress=False, reaction_list=knockable_ids)
        m['fva_bounds'] = flux_limits
        essentials_in_module = set()
        for (reac_id, limits) in flux_limits.iterrows():
            if np.min(abs(limits)) > 1e-10 and np.prod(np.sign(limits)) > 0:
                essentials_in_module.add(reac_id)
        if m[MODULE_TYPE] != SUPPRESS:
            essential_reacs.update(essentials_in_module)
        else:
            suppress_essential.update(essentials_in_module)

    # Size-1 MCS detection: only for classical MCS problems (one SUPPRESS + any PROTECT)
    is_classical_mcs = (len([m for m in sd_modules if m[MODULE_TYPE] == SUPPRESS]) == 1 and
                        all(m[MODULE_TYPE] == PROTECT for m in [m for m in sd_modules if m[MODULE_TYPE] != SUPPRESS]))
    if is_classical_mcs and suppress_essential:
        # Essential for SUPPRESS but NOT for PROTECT → size-1 MCS
        size1_mcs = suppress_essential - essential_reacs
        # Filter to only knockable reactions (in ko_cost, not ki_cost or regulatory)
        size1_mcs_knockable = {r for r in size1_mcs if r in cmp_ko_cost}
        if size1_mcs_knockable:
            cmp_size1_mcs = [{r: -1} for r in size1_mcs_knockable]
            logging.info('  Found ' + str(len(cmp_size1_mcs)) + ' size-1 MCS via SUPPRESS FVA.')
        # Reactions essential for BOTH SUPPRESS and PROTECT are non-knockable
        both_essential = suppress_essential & essential_reacs
        essential_reacs.update(both_essential)
        # Size-1 MCS reactions: remove from ko_cost only (not from ki/reg costs)
        # They are already found; any larger MCS containing them is non-minimal.
        # But we only remove pure KO candidates — reactions with regulatory or KI
        # interventions may still participate in non-KO solutions.
        for r in size1_mcs_knockable:
            cmp_ko_cost.pop(r, None)

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

    kwargs_milp = {k: v for k, v in kwargs.items() if k in [SOLVER, MAX_COST, 'M', SEED, MILP_THREADS]}
    kwargs_milp.update({KOCOST: cmp_ko_cost})
    kwargs_milp.update({KICOST: cmp_ki_cost})
    kwargs_milp.update({'essential_kis': essential_kis})
    logging.info("Finished preprocessing:")
    logging.info("  Model size: " + str(len(cmp_model.reactions)) + " reactions, " + str(len(cmp_model.metabolites)) + " metabolites")
    logging.info("  " + str(len(cmp_ko_cost) + len(cmp_ki_cost) - len(essential_kis)) + " targetable reactions")

    t0 = time.time()
    sd_milp = SDMILP(cmp_model, sd_modules, **kwargs_milp)
    logging.info('  MILP constructed (%.1fs).' % (time.time() - t0))

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

    # enumeration loop variant (only affects the POPULATE approach):
    #   'populate' -> single full-budget populate loop (SDMILP.enumerate)
    #   'ksweep'   -> ascending-cardinality sweep      (SDMILP.enumerate_ksweep)
    # Default is solver-conditional (benchmarked on iML1515-cone gene-MCS, design-identical 393):
    # k-sweep gives CPLEX ~1.8-2.1x and near-parity with gMCSpy, but is SLOWER on gurobi (0.58-0.98x),
    # where the native populate loop already beats gMCSpy. So default ksweep for cplex, populate else.
    enum_method = kwargs.pop('enum_method', 'ksweep' if kwargs.get(SOLVER) == CPLEX else 'populate')

    dump_preprocessed = kwargs.pop('dump_preprocessed', None)

    if dump_preprocessed:
        import os, pickle as _pickle
        dump_path = dump_preprocessed
        with open(dump_path, 'wb') as f:
            _pickle.dump({
                'cmp_model': cmp_model,
                'sd_modules': sd_modules,
                'kwargs_milp': kwargs_milp,
                'kwargs_computation': kwargs_computation,
                'solution_approach': solution_approach,
                'enum_method': enum_method,
                'cmp_mapReac': cmp_mapReac,
                # Expansion/filtering data
                'uncmp_ko_cost': uncmp_ko_cost,
                'uncmp_ki_cost': uncmp_ki_cost,
                'uncmp_reg_cost': uncmp_reg_cost,
                'orig_model': orig_model,
                'orig_sd_modules': orig_sd_modules,
                'orig_ko_cost': orig_ko_cost,
                'orig_ki_cost': orig_ki_cost,
                'orig_reg_cost': orig_reg_cost,
                'gene_kos': kwargs['gene_kos'],
                'orig_gko_cost': locals().get('orig_gko_cost'),
                'orig_gki_cost': locals().get('orig_gki_cost'),
                'max_cost': kwargs[MAX_COST],
                'cmp_size1_mcs': cmp_size1_mcs,
                'pre_fva_bounds': pre_fva_bounds,
            }, f)
        logging.info('Preprocessed data saved to ' + dump_path)
        logging.info('  Resume with:')
        logging.info('    from straindesign import compute_strain_designs_from_preprocessed')
        logging.info("    sol = compute_strain_designs_from_preprocessed('%s', seed=42)" %
                     dump_path.replace('\\', '\\\\'))

        # Return early with size-1 MCS only (or empty)
        setup = deepcopy(cmp_sd_solution.sd_setup) if 'cmp_sd_solution' in dir() else {MODEL_ID: orig_model.id}
        setup.update({MODULES: orig_sd_modules, KOCOST: orig_ko_cost, KICOST: orig_ki_cost, REGCOST: orig_reg_cost})
        if kwargs['gene_kos']:
            setup.update({GKOCOST: orig_gko_cost, GKICOST: orig_gki_cost})
        sd = []
        group_map = []
        compressed_sd = []
        if cmp_size1_mcs:
            for grp_idx, cmp_s in enumerate(cmp_size1_mcs):
                expanded = expand_sd([cmp_s.copy()], cmp_mapReac)
                expanded = filter_sd_maxcost(expanded, kwargs[MAX_COST], uncmp_ko_cost, uncmp_ki_cost)
                expanded = postprocess_reg_sd(uncmp_reg_cost, expanded)
                for s in expanded:
                    sd.append(s)
                    group_map.append(grp_idx)
                compressed_sd.append(cmp_s)
        sd_solutions = SDSolutions(orig_model, sd, OPTIMAL if sd else INFEASIBLE, setup)
        sd_solutions.compressed_sd = compressed_sd
        sd_solutions.compression_map = cmp_mapReac
        sd_solutions.group_map = group_map
        sd_solutions._cmp_model = cmp_model
        logging.info('Returned %d size-1 MCS. MILP solve skipped (dump_preprocessed mode).' % len(sd))
        return sd_solutions

    # solve MILP
    logging.info('  Solving MILP (%s)...' % solution_approach)
    t0 = time.time()
    if solution_approach == ANY:
        cmp_sd_solution = sd_milp.compute(**kwargs_computation)
    elif solution_approach == BEST:
        cmp_sd_solution = sd_milp.compute_optimal(**kwargs_computation)
    elif solution_approach == POPULATE:
        if enum_method == 'ksweep':
            cmp_sd_solution = sd_milp.enumerate_ksweep(**kwargs_computation)
        else:
            cmp_sd_solution = sd_milp.enumerate(**kwargs_computation)
    logging.info('  MILP solved (%.1fs).' % (time.time() - t0))

    # Decompress solutions
    setup = deepcopy(cmp_sd_solution.sd_setup)
    setup.update({MODULES: orig_sd_modules, KOCOST: orig_ko_cost, KICOST: orig_ki_cost, REGCOST: orig_reg_cost})
    if kwargs['gene_kos']:
        setup.update({GKOCOST: orig_gko_cost, GKICOST: orig_gki_cost})

    sd_solutions = _decompress_solutions(
        cmp_sd_solution, cmp_mapReac, cmp_size1_mcs,
        kwargs[MAX_COST], uncmp_ko_cost, uncmp_ki_cost, uncmp_reg_cost,
        orig_model, setup, kwargs['gene_kos'],
        locals().get('orig_gko_cost'), locals().get('orig_gki_cost'))
    sd_solutions._cmp_model = cmp_model
    logging.info(str(sd_solutions.get_num_materialized()) + ' solutions found'
                 + (' (lazy, estimated %d total).' % sd_solutions.get_num_sols()
                    if sd_solutions.is_lazy else '.'))

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


LAZY_EXPANSION_THRESHOLD = 100_000


def _decompress_solutions(cmp_sd_solution, cmp_mapReac, cmp_size1_mcs,
                          max_cost, uncmp_ko_cost, uncmp_ki_cost, uncmp_reg_cost,
                          orig_model, setup, gene_kos, orig_gko_cost, orig_gki_cost):
    """Decompress MILP solutions, using lazy expansion if estimated count exceeds threshold."""
    logging.info('  Decompressing.')

    compressed_sd = []
    cmp_sds = []
    if cmp_sd_solution.status in [OPTIMAL, TIME_LIMIT_W_SOL]:
        cmp_sds = cmp_sd_solution.get_reaction_sd_mark_no_ki()
        compressed_sd = [s.copy() for s in cmp_sds]

    # Estimate expansion size
    estimated = estimate_expansion_size(cmp_sds, cmp_mapReac)
    estimated += estimate_expansion_size(cmp_size1_mcs, cmp_mapReac)

    if estimated > LAZY_EXPANSION_THRESHOLD:
        logging.info('  Estimated %d expanded solutions - using lazy expansion.' % estimated)
        sd, group_map, compressed_sd = _build_lazy_representatives(
            cmp_sds, cmp_size1_mcs, cmp_mapReac, max_cost,
            uncmp_ko_cost, uncmp_ki_cost, uncmp_reg_cost)

        status = cmp_sd_solution.status
        if status not in [OPTIMAL, TIME_LIMIT_W_SOL] and sd:
            status = OPTIMAL

        lazy_meta = {
            'compressed_sd': compressed_sd,
            'compression_map': cmp_mapReac,
            'uncmp_ko_cost': uncmp_ko_cost,
            'uncmp_ki_cost': uncmp_ki_cost,
            'uncmp_reg_cost': uncmp_reg_cost,
            'max_cost': max_cost,
            'model': orig_model,
            'estimated_total': estimated,
        }
        sd_solutions = SDSolutions(orig_model, sd, status, setup, _lazy_init=lazy_meta)
        sd_solutions.compressed_sd = compressed_sd
        sd_solutions.compression_map = cmp_mapReac
        sd_solutions.group_map = group_map
        return sd_solutions

    # Eager expansion (original path)
    group_map = []
    if cmp_sd_solution.status in [OPTIMAL, TIME_LIMIT_W_SOL]:
        sd = []
        for grp_idx, cmp_s in enumerate(cmp_sds):
            expanded = expand_sd([cmp_s], cmp_mapReac)
            expanded = filter_sd_maxcost(expanded, max_cost, uncmp_ko_cost, uncmp_ki_cost)
            expanded = postprocess_reg_sd(uncmp_reg_cost, expanded)
            for s in expanded:
                sd.append(s)
                group_map.append(grp_idx)
    else:
        sd = []

    # Add size-1 MCS found via SUPPRESS FVA
    next_grp = len(compressed_sd)
    if cmp_size1_mcs:
        existing = [frozenset(s.items()) for s in sd]
        for grp_idx, cmp_s in enumerate(cmp_size1_mcs):
            expanded = expand_sd([cmp_s], cmp_mapReac)
            expanded = filter_sd_maxcost(expanded, max_cost, uncmp_ko_cost, uncmp_ki_cost)
            expanded = postprocess_reg_sd(uncmp_reg_cost, expanded)
            for s in expanded:
                if frozenset(s.items()) not in existing:
                    sd.append(s)
                    group_map.append(next_grp + grp_idx)
                    existing.append(frozenset(s.items()))
            compressed_sd.append(cmp_s)
        if cmp_sd_solution.status not in [OPTIMAL, TIME_LIMIT_W_SOL] and sd:
            cmp_sd_solution.status = OPTIMAL

    sd_solutions = SDSolutions(orig_model, sd, cmp_sd_solution.status, setup)
    sd_solutions.compressed_sd = compressed_sd
    sd_solutions.compression_map = cmp_mapReac
    sd_solutions.group_map = group_map
    return sd_solutions


def _build_lazy_representatives(cmp_sds, cmp_size1_mcs, cmp_mapReac, max_cost,
                                uncmp_ko_cost, uncmp_ki_cost, uncmp_reg_cost):
    """Build one representative expanded solution per compressed group.

    Returns (sd, group_map, compressed_sd).
    """
    sd = []
    group_map = []
    compressed_sd = []

    # MILP solutions
    for grp_idx, cmp_s in enumerate(cmp_sds):
        expanded = expand_sd([cmp_s.copy()], cmp_mapReac)
        expanded = filter_sd_maxcost(expanded, max_cost, uncmp_ko_cost, uncmp_ki_cost)
        expanded = postprocess_reg_sd(uncmp_reg_cost, expanded)
        if expanded:
            sd.append(expanded[0])  # cheapest representative
            group_map.append(grp_idx)
        compressed_sd.append(cmp_s.copy())

    # Size-1 MCS
    next_grp = len(compressed_sd)
    existing = {frozenset(s.items()) for s in sd}
    for grp_idx, cmp_s in enumerate(cmp_size1_mcs):
        expanded = expand_sd([cmp_s.copy()], cmp_mapReac)
        expanded = filter_sd_maxcost(expanded, max_cost, uncmp_ko_cost, uncmp_ki_cost)
        expanded = postprocess_reg_sd(uncmp_reg_cost, expanded)
        for s in expanded:
            if frozenset(s.items()) not in existing:
                sd.append(s)
                group_map.append(next_grp + grp_idx)
                existing.add(frozenset(s.items()))
                break  # only first representative
        compressed_sd.append(cmp_s.copy())

    return sd, group_map, compressed_sd


def compute_strain_designs_from_preprocessed(dump, seed=None, solver=None,
                                             solution_approach=None, max_solutions=None,
                                             time_limit=None, enum_method=None):
    """Load preprocessed model and run MILP solve with optional overrides.

    Args:
        dump (str or dict): Path to pickle file created by dump_preprocessed,
            or the dict itself (e.g. after unpickling and modifying).
        seed (int, optional): Override MILP solver seed.
        solver (str, optional): Override solver.
        solution_approach (str, optional): Override solution approach ('any', 'best', 'populate').
        max_solutions (int or float, optional): Override max_solutions.
        time_limit (int or float, optional): Override time limit.

    Returns:
        (SDSolutions): Strain design solutions.
    """
    if isinstance(dump, dict):
        d = dump
    else:
        import pickle as _pickle
        with open(dump, 'rb') as f:
            d = _pickle.load(f)

    cmp_model = d['cmp_model']
    sd_modules = d['sd_modules']
    kwargs_milp = d['kwargs_milp']
    kwargs_computation = d['kwargs_computation']
    sol_approach = d['solution_approach']
    cmp_mapReac = d['cmp_mapReac']
    uncmp_ko_cost = d['uncmp_ko_cost']
    uncmp_ki_cost = d['uncmp_ki_cost']
    uncmp_reg_cost = d['uncmp_reg_cost']
    orig_model = d['orig_model']
    orig_sd_modules = d['orig_sd_modules']
    orig_ko_cost = d['orig_ko_cost']
    orig_ki_cost = d['orig_ki_cost']
    orig_reg_cost = d['orig_reg_cost']
    gene_kos = d['gene_kos']
    orig_gko_cost = d.get('orig_gko_cost')
    orig_gki_cost = d.get('orig_gki_cost')
    max_cost = d['max_cost']
    cmp_size1_mcs = d['cmp_size1_mcs']
    enum_meth = d.get('enum_method', 'populate')

    # Apply overrides
    if enum_method is not None:
        enum_meth = enum_method
    if seed is not None:
        kwargs_milp[SEED] = seed
    if solver is not None:
        kwargs_milp[SOLVER] = select_solver(solver)
    if max_solutions is not None:
        kwargs_computation[MAX_SOLUTIONS] = float(max_solutions)
    if time_limit is not None:
        kwargs_computation[T_LIMIT] = float(time_limit)
    if solution_approach is not None:
        sol_approach = solution_approach

    # The cmp_model was pickled under LP suppression (solver is _SolverStub).
    # Re-apply suppression so SDMILP construction can access variables safely.
    from straindesign.networktools import suppress_lp_context
    with suppress_lp_context(cmp_model):
        logging.info('Loading preprocessed data from ' + (dump if isinstance(dump, str) else 'dict input.'))
        logging.info('  Seed: %s, Solver: %s, Approach: %s' % (
            kwargs_milp.get(SEED), kwargs_milp.get(SOLVER), sol_approach))

        t0 = time.time()
        sd_milp = SDMILP(cmp_model, sd_modules, **kwargs_milp)
        logging.info('  MILP constructed (%.1fs).' % (time.time() - t0))

    logging.info('  Solving MILP (%s)...' % sol_approach)
    t0 = time.time()
    if sol_approach == ANY:
        cmp_sd_solution = sd_milp.compute(**kwargs_computation)
    elif sol_approach == BEST:
        cmp_sd_solution = sd_milp.compute_optimal(**kwargs_computation)
    elif sol_approach == POPULATE:
        if enum_meth == 'ksweep':
            cmp_sd_solution = sd_milp.enumerate_ksweep(**kwargs_computation)
        else:
            cmp_sd_solution = sd_milp.enumerate(**kwargs_computation)
    logging.info('  MILP solved (%.1fs).' % (time.time() - t0))

    setup = deepcopy(cmp_sd_solution.sd_setup)
    setup.update({MODULES: orig_sd_modules, KOCOST: orig_ko_cost, KICOST: orig_ki_cost, REGCOST: orig_reg_cost})
    if gene_kos:
        setup.update({GKOCOST: orig_gko_cost, GKICOST: orig_gki_cost})

    sd_solutions = _decompress_solutions(
        cmp_sd_solution, cmp_mapReac, cmp_size1_mcs,
        max_cost, uncmp_ko_cost, uncmp_ki_cost, uncmp_reg_cost,
        orig_model, setup, gene_kos, orig_gko_cost, orig_gki_cost)
    sd_solutions._cmp_model = cmp_model
    logging.info(str(sd_solutions.get_num_materialized()) + ' solutions found'
                 + (' (lazy, estimated %d total).' % sd_solutions.get_num_sols()
                    if sd_solutions.is_lazy else '.'))

    return sd_solutions
