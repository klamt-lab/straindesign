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
"""Functions for metabolic network compression and extension with GPR rules"""

import numpy as np
from scipy import sparse
from sympy.core.numbers import One
from sympy import Rational, nsimplify, parse_expr, to_dnf
from typing import Dict, List
from re import search
import jpype
from cobra import Model, Metabolite, Reaction, Configuration
from cobra.util.array import create_stoichiometric_matrix
import straindesign.efmtool as efm
from straindesign import fva, select_solver, parse_constraints, avail_solvers
from straindesign.names import *
import logging


def remove_irrelevant_genes(model, essential_reacs, gkis, gkos):
    """Remove genes whose that do not affect the flux space of the model
    
    This function is used in preprocessing of computational strain design computations. Often,
    certain reactions, for instance, reactions essential for microbial growth can/must not be
    targeted by interventions. That can be exploited to reduce the set of genes in which 
    interventions need to be considered.
    
    Given a set of essential reactions that is to be maintained operational, some genes can be 
    removed from a metabolic model, either because they only affect only blocked reactions or 
    essential reactions, or because they are essential reactions and must not be removed. As a
    consequence, the GPR rules of a model can be simplified. 
    
        
    Example:
        remove_irrelevant_genes(model, essential_reacs, gkis, gkos):
    
    Args:
        model (cobra.Model):
            A metabolic model that is an instance of the cobra.Model class containing GPR rules
            
        essential_reacs (list of str):
            A list of identifiers of essential reactions.
            
        gkis, gkos (dict):
            Dictionaries that contain the costs for gene knockouts and additions. E.g.,
            gkos={'adhE': 1.0, 'ldhA' : 1.0 ...}
            
    Returns:
        (dict):
        An updated dictionary of the knockout costs in which irrelevant genes are removed.
    """
    # 1) Remove gpr rules from blocked reactions
    blocked_reactions = [reac.id for reac in model.reactions if reac.bounds == (0, 0)]
    for rid in blocked_reactions:
        model.reactions.get_by_id(rid).gene_reaction_rule = ''
    for g in model.genes[::-1]:  # iterate in reverse order to avoid mixing up the order of the list when removing genes
        if not g.reactions:
            model.genes.remove(g)
    protected_genes = set()
    # 1. Protect genes that only occur in essential reactions
    for g in model.genes:
        if not g.reactions or {r.id for r in g.reactions}.issubset(essential_reacs):
            protected_genes.add(g)
    # 2. Protect genes that are essential to essential reactions
    for r in [model.reactions.get_by_id(s) for s in essential_reacs]:
        for g in r.genes:
            exp = r.gene_reaction_rule.replace(' or ', ' | ').replace(' and ', ' & ')
            exp = to_dnf(parse_expr(exp, {g.id: False}), force=True)
            if exp == False:
                protected_genes.add(g)
    # 3. Remove essential genes, and knockouts without impact from gko_costs
    [gkos.pop(pg.id) for pg in protected_genes if pg.id in gkos]
    # 4. Add all notknockable genes to the protected list
    [protected_genes.add(g) for g in model.genes if (g.id not in gkos) and (g.name not in gkos)]  # support names or ids in gkos
    # genes with kiCosts are kept
    gki_ids = [g.id for g in model.genes if (g.id in gkis) or (g.name in gkis)]  # support names or ids in gkis
    protected_genes = protected_genes.difference({model.genes.get_by_id(g) for g in gki_ids})
    protected_genes_dict = {pg.id: True for pg in protected_genes}
    # 5. Simplify gpr rules and discard rules that cannot be knocked out
    for r in model.reactions:
        if r.gene_reaction_rule:
            exp = r.gene_reaction_rule.replace(' or ', ' | ').replace(' and ', ' & ')
            exp = to_dnf(parse_expr(exp, protected_genes_dict), simplify=True, force=True)
            if exp == True:
                model.reactions.get_by_id(r.id).gene_reaction_rule = ''
            elif exp == False:
                logging.error('Something went wrong during gpr rule simplification.')
            else:
                model.reactions.get_by_id(r.id).gene_reaction_rule = str(exp).replace(' & ', ' and ').replace(' | ', ' or ')
    # 6. Remove obsolete genes and protected genes
    for g in model.genes[::-1]:
        if not g.reactions or g in protected_genes:
            model.genes.remove(g)
    return gkos


def extend_model_gpr(model, use_names=False):
    """Integrate GPR-rules into a metabolic model as pseudo metabolites and reactions
    
    COBRA modules often have gene-protein-reaction (GPR) rules associated with each reaction. 
    These can be integrated into the metabolic network structure through pseudo reactions
    and variables. As GPR rules are integrated into the metabolic network, the metabolic flux 
    space does not change. After integration, the gene-pseudoreactions can be fixed to a flux of 
    zero to simulate gene knockouts. Gene pseudoreactions are referenced either by the gene name 
    or the gene identifier (user selected).
    
    GPR-rule integration enables the computation of strain designs based on genetic interventions.
    
    This function requires all GPR-rules to be provided in DNF (disjunctive normal form).
    (e.g. g1 and g2 or g1 and g3, NOT g1 and (g2 or g3)). Brackets are allowed but not required.
    If reversible reactions are associated with GPR-rules, these reactions are spit during
    GPR-integration. The function returns a mapping of old and new reaction identifiers.
        
    Example:
        reac_map = extend_model_gpr(model):
    
    Args:
        model (cobra.Model):
            A metabolic model that is an instance of the cobra.Model class containing GPR rules 
            in DNF
                      
        use_names (bool): (Default: False)
            If set to True, the gene pseudoreactions will carry the gene name as reaction
            identifier. If False, the gene identifier will be used. By default this option is
            turned off because many models do not provide gene names.
            
    Returns:
        (dict):
        A dictionary to reference old and new reaction identifiers, for reversible reactions
        that were split (when they are associated with GPR rules). Entries have the form:
        {'Reaction1' : {'Reaction1' : 1, 'Reaction1_reverse_a59c' : -1}}
            
    """
    # Check if reaction names and gene names/IDs overlap. If yes, throw Error
    reac_ids = {r.id for r in model.reactions}
    if (not use_names) and any([g.id in reac_ids for g in model.genes]):
        raise Exception("GPR rule integration requires distinct identifiers for reactions and "+\
                        "genes.\nThe following identifiers seem to be both, reaction IDs and gene "+\
                        "IDs:\n"+str([g.id for g in model.genes if g.id in reac_ids])+\
                        "\nTo prevent this problem, call extend_model_gpr with the use_names=False "+\
                        "option or refer\nto gene names instead of IDs in your strain design "+\
                        "computation if they are available.\nYou may also rename the conflicting "+\
                        "genes with cobra.manipulation.modify.rename_genes(model, {'g_old': 'g_new'})")
    elif use_names and any([g.name in reac_ids for g in model.genes]):
        raise Exception("GPR rule integration requires distinct identifiers for reactions and "+\
                        "genes.\nThe following identifiers seem to be both, reaction IDs and gene "+\
                        "names:\n"+str([g.name for g in model.genes if g.name in reac_ids])+\
                        "\nTo prevent this problem, call extend_model_gpr with the use_names=False "+\
                        "option or refer \nto gene IDs instead of names in your strain design "+\
                        "computation. You may also rename\nthe conflicting genes, e.g., with "+\
                        "model.genes.get_by_id('ADK1').name = 'adk1'")

    MAX_NAME_LEN = 230

    def warning_name_too_long(id, p=""):
        logging.warning(" GPR rule integration automatically generates new reactions and metabolites."+\
                        "\nOne of the generated reaction names is beyond or close to the limit of 255 "+\
                        "characters\npermitted by GLPK and Gurobi. The name of the newly generated "+\
                        "reaction or metabolite: \n "+id+",\ngenerated from reaction or metabolite:\n "+\
                        p+"\n"+"was therefore trimmed to:\n "+id[0:MAX_NAME_LEN]+".\nThis trimming is "+\
                        "usually safe, no guarantee is given. To avoid this message,\nuse the CPLEX "+\
                        "solver or consider simplifying GPR rules or gene names in your model.")

    def truncate(id):
        return id[0:MAX_NAME_LEN - 16] + hex(abs(hash(id)))[2:]

    solver = search('(' + '|'.join(avail_solvers) + ')', model.solver.interface.__name__)[0]

    # Split reactions when necessary
    reac_map = {}
    rev_reac = set()
    del_reac = set()
    for r in model.reactions:
        reac_map.update({r.id: {}})
        if not r.gene_reaction_rule:
            reac_map[r.id].update({r.id: 1.0})
            continue
        if r.gene_reaction_rule and r.bounds[0] < 0:
            r_rev = (r * -1)
            if r.gene_reaction_rule and r.bounds[1] > 0:
                r_rev.id = r.id + '_reverse_' + hex(hash(r))[8:]
            r_rev.lower_bound = np.max([0, r_rev.lower_bound])
            reac_map[r.id].update({r_rev.id: -1.0})
            if len(r_rev.id) > MAX_NAME_LEN and solver in {GUROBI, GLPK}:
                warning_name_too_long(r_rev.id, r.id)
                r_rev.id = truncate(r_rev.id)
            rev_reac.add(r_rev)
        if r.gene_reaction_rule and r.bounds[1] > 0:
            reac_map[r.id].update({r.id: 1.0})
            r._lower_bound = np.max([0, r._lower_bound])
        else:
            del_reac.add(r)
    model.remove_reactions(del_reac)
    model.add_reactions(rev_reac)

    # All reaction rules are provided in dnf.
    for r in model.reactions:
        if r.gene_reaction_rule:  # if reaction has a gpr rule
            dt = [s.strip() for s in r.gene_reaction_rule.split(' or ')]
            for i, p in enumerate(dt.copy()):
                ct = [s.strip() for s in p.replace('(', '').replace(')', '').split(' and ')]
                for j, g in enumerate(ct.copy()):
                    gene_met_id = 'g_' + g
                    # Check name length of new metabolite
                    if len(gene_met_id) > MAX_NAME_LEN and solver in {GUROBI, GLPK}:
                        if truncate(gene_met_id) not in [m.id for m in model.metabolites]:
                            warning_name_too_long(gene_met_id, r.id)
                        gene_met_id = truncate(gene_met_id)
                    # if gene is not in model, add gene pseudoreaction and metabolite
                    if gene_met_id not in model.metabolites.list_attr('id'):
                        model.add_metabolites(Metabolite(gene_met_id))
                        gene = model.genes.get_by_id(g)
                        w = Reaction(gene.id)
                        if use_names:  # if gene name is available and used in gki_cost and gko_cost
                            w.id = gene.name
                        # Check name length of new reaction
                        if len(w.id) > MAX_NAME_LEN and solver in {GUROBI, GLPK}:
                            warning_name_too_long(w.id, r.id)
                            w.id = truncate(w.id)
                        model.add_reactions([w])
                        w.reaction = '--> ' + gene_met_id
                        w._upper_bound = np.inf
                    ct[j] = gene_met_id
                if len(ct) > 1:
                    ct_met_id = "_and_".join(ct)
                    # Check name length of new metabolite
                    if len(ct_met_id) > MAX_NAME_LEN and solver in {GUROBI, GLPK}:
                        if truncate(ct_met_id) not in [m.id for m in model.metabolites]:
                            warning_name_too_long(ct_met_id, r.id)
                        ct_met_id = truncate(ct_met_id)
                    if ct_met_id not in model.metabolites.list_attr('id'):
                        # if conjunct term is not in model, add pseudoreaction and metabolite
                        model.add_metabolites(Metabolite(ct_met_id))
                        w = Reaction("R_" + ct_met_id)
                        # Check name length of new reaction
                        if len(w.id) > MAX_NAME_LEN and solver in {GUROBI, GLPK}:
                            warning_name_too_long(w.id, r.id)
                            w.id = truncate(w.id)
                        model.add_reactions([w])
                        w.reaction = ' + '.join(ct) + '--> ' + ct_met_id
                        w._upper_bound = np.inf
                    dt[i] = ct_met_id
                else:
                    dt[i] = gene_met_id
            if len(dt) > 1:
                dt_met_id = "_or_".join(dt)
                # Check name length of new metabolite
                if len(dt_met_id) > MAX_NAME_LEN and solver in {GUROBI, GLPK}:
                    if truncate(dt_met_id) not in [m.id for m in model.metabolites]:
                        warning_name_too_long(dt_met_id, r.id)
                    dt_met_id = truncate(dt_met_id)
                if dt_met_id not in model.metabolites.list_attr('id'):
                    model.add_metabolites(Metabolite(dt_met_id))
                    for k, d in enumerate(dt):
                        w = Reaction("R" + str(k) + "_" + dt_met_id)
                        # Check name length of new reaction
                        if len(w.id) > MAX_NAME_LEN and solver in {GUROBI, GLPK}:
                            warning_name_too_long(w.id, r.id)
                            w.id = truncate(w.id)
                        model.add_reactions([w])
                        w.reaction = d + ' --> ' + dt_met_id
                        w._upper_bound = np.inf
            else:
                dt_met_id = dt[0]
            r.add_metabolites({model.metabolites.get_by_id(dt_met_id): -1.0})
    return reac_map


def extend_model_regulatory(model, reg_itv):
    """Extend a metabolic model to account for regulatory constraints
    
    This function emulates regulatory interventions in a network. These can either be added
    permanently or linked to a pseudoreation whose boundaries can be fixed to zero used 
    to activate the regulatory constraint.
    
    Accounting for regulatory interventions, such as applying an upper or lower bound
    to a reaction or gene pseudoreaction, can be achieved by combining different
    pseudometabolites and reactions. For instance, to introduce the regulatory constraint:
    
    2*r_1 + 3*r_2 <= 4
    
    and make it 'toggleable', one adds 1 metabolite 'm' and 2 reactions, 'r_bnd' to account for
    the bound/rhs and r_ctl to control whether the regulatory intervention is active or not:
    
    dm/dt = 2*r_1 + 3*r_2 - r_bnd + r_ctl = 0, -inf <= r_bnd <= 4, -inf <= r_ctl <= inf
    
    When r_ctl is fixed to zero, the constraint 2*r_1 + 3*r_2 <= 4 is enforced, otherwise,
    the constraint is non binding, thus virtually non-existant. To use this mechanism for 
    strain design, we add the metabolite and reactions as described above and tag r_ctl as 
    knockout candidate. If the algorithm decides to knockout r_ctl, this means, it choses to 
    add the regulatory intervention 2*r_1 + 3*r_2 <= 4.
    
    If the constraint is be added permanently, this function completely omits the r_ctl reaction.
    
    Example:
        reg_itv_costs = extend_model_regulatory(model, {'1 PDH + 1 PFL <= 5' : 1, '-EX_o2_e <= 2' : 1.5})
    
    Args:
        model (cobra.Model):
            A metabolic model that is an instance of the cobra.Model class
            
        reg_itv (dict or list of str or str):
            A set of regulatory constraints that should be added to the model. If reg_itv is a
            string or a list of strings, regulatory constraints are added permanently. If reg_itv
            is a dict, regulatory interventions are added in a controllable manner. The id of the
            reaction that controls the constraint is contained in the return variable. The constraint
            to be added will be parsed from strings, so ensure that you use the correct reaction
            identifiers. Valid inputs are:
            reg_itv = '-EX_o2_e <= 2' # A single permanent regulatory constraint
            reg_itv = ['1 PDH + 1 PFL <= 5', '-EX_o2_e <= 2'] # Two permanent constraints
            reg_itv = {'1 PDH + 1 PFL <= 5' : 1, '-EX_o2_e <= 2' : 1.5} # Two controllable constraints
            # one costs '1' and the other one '1.5' to be added. The function returns a dict with 
            # {'p1_PDH_p1_PFK_le_5' : 1 'nEX_o2_e_le_2' : 1.5}. Fixing the reaction 
            # p1_PDH_p1_PFK_le_5 to zero will activate the constraint in the model.
            
    Returns:
        (dict):
        A dictionary that contains the cost of adding each constraint; e.g.,
        {'p1_PDH_p1_PFK_le_5' : 1 'n1EX_o2_e_le_2' : 1.5}
    """
    keywords = set(model.reactions.list_attr('id'))
    if '' in keywords:
        keywords.remove('')
    if not isinstance(reg_itv, dict):
        if not isinstance(reg_itv, str) and not isinstance(reg_itv, list):
            raise Exception('reg_itv must be a string, a list or a dictionary.'\
                            'See function description for details.')
        if isinstance(reg_itv, str):
            reg_itv = {reg_itv: np.nan}
        elif isinstance(reg_itv, list):
            reg_itv = {r: np.nan for r in reg_itv}
    regcost = {}
    for k, v in reg_itv.copy().items():
        # generate name for regulatory pseudoreaction
        try:
            constr = parse_constraints(k, keywords)[0]
        except:
            raise Exception('Regulatory constraints could not be parsed. Please revise.')
        reacs_dict = constr[0]
        eqsign = constr[1]
        rhs = constr[2]
        reg_name = ''
        for l, w in reacs_dict.items():
            if w < 0:
                reg_name += 'n' + str(w) + '_' + l
            else:
                reg_name += 'p' + str(w) + '_' + l
            reg_name += '_'
        if eqsign == '=':
            reg_name += 'eq_'
        elif eqsign == '<=':
            reg_name += 'le_'
        elif eqsign == '>=':
            reg_name += 'ge_'
        reg_name += str(rhs).replace('-', 'n').replace('.', 'p')
        reg_itv.pop(k)
        reg_itv.update({reg_name: {'str': k, 'cost': v}})
        reg_pseudomet_name = 'met_' + reg_name
        # add pseudometabolite
        m = Metabolite(reg_pseudomet_name)
        model.add_metabolites(m)
        # add pseudometabolite to stoichiometries
        for l, w in reacs_dict.items():
            r = model.reactions.get_by_id(l)
            r.add_metabolites({m: w})
        # add pseudoreaction that defines the bound
        r_bnd = Reaction("bnd_" + reg_name)
        model.add_reactions([r_bnd])
        r_bnd.reaction = reg_pseudomet_name + ' --> '
        if eqsign == '=':
            r_bnd._lower_bound = -np.inf
            r_bnd._upper_bound = rhs
            r_bnd._lower_bound = rhs
        elif eqsign == '<=':
            r_bnd._lower_bound = -np.inf
            r_bnd._upper_bound = rhs
        elif eqsign == '>=':
            r_bnd._upper_bound = np.inf
            r_bnd._lower_bound = rhs
        # add knockable pseudoreaction and add it to the kocost list
        if not np.isnan(v):
            r_ctl = Reaction(reg_name)
            model.add_reactions([r_ctl])
            r_ctl.reaction = '--> ' + reg_pseudomet_name
            r_ctl._upper_bound = np.inf
            r_ctl._lower_bound = -np.inf
            regcost.update({reg_name: v})
    return regcost


def compress_model(model, no_par_compress_reacs=set()):
    """Compress a metabolic model with a number of different techniques
    
    The network compression routine removes blocked reactions, removes conservation
    relations and then performs alternatingly lumps dependent (compress_model_efmtool) 
    and parallel (compress_model_parallel) reactions. The compression returns a compressed 
    network and a list of compression maps. Each map consists of a dictionary that contains 
    complete information for reversing the compression steps successively and expand 
    information obtained from the compressed model to the full model. Each entry of each 
    map contains the id of a compressed reaction, associated with the original reaction 
    names and their factor (provided as a rational number) with which they were lumped.
    
    Furthermore, the user can select reactions that should be exempt from the parallel 
    compression. This is a critical feature for strain design computations. There is
    currently no way to exempt reactions from the efmtool/dependency compression.
    
    Example:
        comression_map = compress_model(model,set('EX_etoh_e','PFL'))
    
    Args:
        model (cobra.Model):
            A metabolic model that is an instance of the cobra.Model class
        
        no_par_compress_reacs (set or list of str): (Default: set())
            A set of reaction identifiers whose reactions should not be lumped with other
            parallel reactions.
        
    Returns:
        (list of dict):
        A list of compression maps. Each map is a dict that contains information for reversing 
        the compression steps successively and expand information obtained from the compressed 
        model to the full model. Each entry of each map contains the id of a compressed reaction, 
        associated with the original reaction identifiers and their factor with which they are
        represented in the lumped reaction (provided as a rational number) with which they were 
        lumped.
    """
    # Remove conservation relations.
    logging.info('  Removing blocked reactions.')
    remove_blocked_reactions(model)
    logging.info('  Translating stoichiometric coefficients to rationals.')
    stoichmat_coeff2rational(model)
    logging.info('  Removing conservation relations.')
    remove_conservation_relations(model)
    parallel = False
    run = 1
    cmp_mapReac = []
    numr = len(model.reactions)
    while True:
        if not parallel:
            logging.info('  Compression ' + str(run) + ': Applying compression from EFM-tool module.')
            reac_map_exp = compress_model_efmtool(model)
            for new_reac, old_reac_val in reac_map_exp.items():
                old_reacs_no_compress = [r for r in no_par_compress_reacs if r in old_reac_val]
                if old_reacs_no_compress:
                    [no_par_compress_reacs.remove(r) for r in old_reacs_no_compress]
                    no_par_compress_reacs.add(new_reac)
        else:
            logging.info('  Compression ' + str(run) + ': Lumping parallel reactions.')
            reac_map_exp = compress_model_parallel(model, no_par_compress_reacs)
        remove_conservation_relations(model)
        if numr > len(reac_map_exp):
            logging.info('  Reduced to ' + str(len(reac_map_exp)) + ' reactions.')
            # store the information for decompression in a Tuple
            # (0) compression matrix, (1) reac_id dictornary {cmp_rid: {orig_rid1: factor1, orig_rid2: factor2}},
            # (2) linear (True) or parallel (False) compression (3,4) ko and ki costs of expanded network
            cmp_mapReac += [{
                "reac_map_exp": reac_map_exp,
                "parallel": parallel,
            }]
            if parallel:
                parallel = False
            else:
                parallel = True
            run += 1
            numr = len(reac_map_exp)
        else:
            logging.info('  Last step could not reduce size further (' + str(numr) + ' reactions).')
            logging.info('  Network compression completed. (' + str(run - 1) + ' compression iterations)')
            logging.info('  Translating stoichiometric coefficients back to float.')
            break
    stoichmat_coeff2float(model)
    return cmp_mapReac


def compress_modules(sd_modules, cmp_mapReac):
    """Compress strain design modules to match with a compressed model
    
    When a strain design task has been specified with modules and the original metabolic model 
    was compressed, one needs to refit the strain design modules (objects of the SDModule class)
    to the new compressed model. This function takes a list of modules and a compression map
    and returns the strain design modules for a compressed network.
    
    Example:
        comression_map = compress_modules(sd_modules, cmp_mapReac)
    
    Args:
        model (list of SDModule):
            A list of strain design modules
        
        cmp_mapReac (list of dicts):
            Compression map obtained from cmp_mapReac = compress_model(model)
        
    Returns:
        (list of SDModule):
        A list of strain design modules for the compressed network
    """
    sd_modules = modules_coeff2rational(sd_modules)
    for cmp in cmp_mapReac:
        reac_map_exp = cmp["reac_map_exp"]
        parallel = cmp["parallel"]
        if not parallel:
            for new_reac, old_reac_val in reac_map_exp.items():
                for i, m in enumerate(sd_modules):
                    for p in [CONSTRAINTS, INNER_OBJECTIVE, OUTER_OBJECTIVE, PROD_ID, MIN_GCP]:
                        if p in m and m[p] is not None:
                            param = m[p]
                            if p == CONSTRAINTS:
                                for j, c in enumerate(m[p]):
                                    if np.any([k in old_reac_val for k in c[0].keys()]):
                                        lumped_reacs = [k for k in c[0].keys() if k in old_reac_val]
                                        c[0][new_reac] = np.sum([c[0].pop(k) * old_reac_val[k] for k in lumped_reacs])
                            if p in [INNER_OBJECTIVE, OUTER_OBJECTIVE, PROD_ID]:
                                if np.any([k in old_reac_val for k in param.keys()]):
                                    lumped_reacs = [k for k in param.keys() if k in old_reac_val]
                                    m[p][new_reac] = np.sum([param.pop(k) * old_reac_val[k] for k in lumped_reacs if k in old_reac_val])
    sd_modules = modules_coeff2float(sd_modules)
    return sd_modules


def compress_ki_ko_cost(kocost, kicost, cmp_mapReac):
    """Compress knockout/addition cost vectors to match with a compressed model
    
    When knockout/addition cost vectors have been specified (as dicts) and the original
    metabolic model was compressed, one needs to update the knockout/addition cost vectors.
    This function takes care of this. In particular it makes sure that the resulting costs
    are calculated correctly. 
    
    E.g.: r_ko_a (cost 1) and r_ko_b (cost 2) are lumped parallel: The resulting cost of r_ko_ab is 3
    If they are lumped as dependent reactions the resulting cost is 1. If one of the two reactions
    is an addition candidate, the resulting reaction will be an addition candidate when lumped as 
    dependent reactions and a knockout candidate when lumped in parallel. There are various possible
    cases that are treated by this function.
    
    Example:
        kocost, kicost, cmp_mapReac = compress_ki_ko_cost(kocost, kicost, cmp_mapReac)
    
    Args:
        kocost, kicost (dict):
            Knockout and addition cost vectors
        
        cmp_mapReac (list of dicts):
            Compression map obtained from cmp_mapReac = compress_model(model)
        
    Returns:
        (Tuple):
        Updated vectors of KO costs and KI costs and an updated compression map that contains information
        on how to expand strain designs and correctly distinguish between knockouts and additions.
    """
    # kocost of lumped reactions: when reacs sequential: lowest of ko costs, when parallel: sum of ko costs
    # kicost of lumped reactions: when reacs sequential: sum of ki costs, when parallel: lowest of ki costs
    for cmp in cmp_mapReac:
        reac_map_exp = cmp["reac_map_exp"]
        parallel = cmp["parallel"]
        cmp.update({KOCOST: kocost, KICOST: kicost})
        if kocost:
            ko_cost_new = {}
            for r in reac_map_exp:
                if np.any([s in kocost for s in reac_map_exp[r]]):
                    if not parallel and not np.any([s in kicost for s in reac_map_exp[r]]):
                        ko_cost_new[r] = np.min([kocost[s] for s in reac_map_exp[r] if s in kocost])
                    elif parallel:
                        ko_cost_new[r] = np.sum([kocost[s] for s in reac_map_exp[r] if s in kocost])
            kocost = ko_cost_new
        if kicost:
            ki_cost_new = {}
            for r in reac_map_exp:
                if np.any([s in kicost for s in reac_map_exp[r]]):
                    if not parallel:
                        ki_cost_new[r] = np.sum([kicost[s] for s in reac_map_exp[r] if s in kicost])
                    elif parallel and not np.any([s in kocost for s in reac_map_exp[r]]):
                        ki_cost_new[r] = np.min([kicost[s] for s in reac_map_exp[r] if s in kicost])
            kicost = ki_cost_new
    return kocost, kicost, cmp_mapReac


def expand_sd(sd, cmp_mapReac):
    """Expand computed strain designs from a compressed to a full model
    
    Needed after computing strain designs in a compressed model
    
    Example:
        expanded_sd = expand_sd(compressed_sds, cmp_mapReac)
    
    Args:
        sd (SDSolutions):
            Solutions of a strain design computation that refer to a compressed model
        
        cmp_mapReac (list of dicts):
            Compression map obtained from cmp_mapReac = compress_model(model) and updated with
            kocost, kicost, cmp_mapReac = compress_ki_ko_cost(kocost, kicost, cmp_mapReac)
        
    Returns:
        (SDSolutions):
        Strain design solutions that refer to the uncompressed model
    """
    # expand mcs by applying the compression steps in the reverse order
    cmp_map = cmp_mapReac[::-1]
    for exp in cmp_map:
        reac_map_exp = exp["reac_map_exp"]  # expansion map
        ko_cost = exp[KOCOST]
        ki_cost = exp[KICOST]
        par_reac_cmp = exp["parallel"]  # if parallel or sequential reactions were lumped
        for r_cmp, r_orig in reac_map_exp.items():
            if len(r_orig) > 1:
                for m in sd.copy():
                    if r_cmp in m:
                        val = m[r_cmp]
                        del m[r_cmp]
                        if val < 0:  # case: KO
                            if par_reac_cmp:
                                new_m = m.copy()
                                for d in r_orig:
                                    if d in ko_cost:
                                        new_m[d] = val
                                sd += [new_m]
                            else:
                                for d in r_orig:
                                    if d in ko_cost:
                                        new_m = m.copy()
                                        new_m[d] = val
                                        sd += [new_m]
                        elif val > 0:  # case: KI
                            if par_reac_cmp:
                                for d in r_orig:
                                    if d in ki_cost:
                                        new_m = m.copy()
                                        new_m[d] = val
                                        # other reactions do not need to be knocked in
                                        for f in [e for e in r_orig if (e in ki_cost) and e != d]:
                                            new_m[f] = 0.0
                                        sd += [new_m]
                            else:
                                new_m = m.copy()
                                for d in r_orig:
                                    if d in ki_cost:
                                        new_m[d] = val
                                sd += [new_m]
                        elif val == 0:  # case: KI that was not introduced
                            new_m = m.copy()  # assume that none of the expanded
                            for d in r_orig:  # reactions are inserted, neither
                                if d in ki_cost:  # parallel, nor sequential
                                    new_m[d] = val
                            sd += [new_m]
                        sd.remove(m)
    return sd


def filter_sd_maxcost(sd, max_cost, kocost, kicost):
    """Filter out strain designs that exceed the maximum allowed intervention costs
    
    Returns:
        (SDSolutions):
        Strain design solutions complying with the intervention costs limit
    """
    # eliminate mcs that are too expensive
    if max_cost:
        costs = [np.sum([kocost[k] if v < 0 else (kicost[k] if v > 0 else 0) for k, v in m.items()]) for m in sd]
        sd = [sd[i] for i in range(len(sd)) if costs[i] <= max_cost + 1e-8]
        # sort strain designs by intervention costs
        [s.update({'**cost**': c}) for s, c in zip(sd, costs)]
        sd.sort(key=lambda x: x.pop('**cost**'))
    return sd


# compression function (mostly copied from efmtool)
def compress_model_efmtool(model):
    """Compress model by lumping dependent reactions using the efmtool compression approach
    
    Example:
        cmp_mapReac = compress_model_efmtool(model)
    
    Args:
        model (cobra.Model):
            A metabolic model that is an instance of the cobra.Model class
            
    Returns:
        (dict):
        A dict that contains information about the lumping done in the compression process.
        process. E.g.: {'reaction_lumped1' : {'reaction_orig1' : 2/3 'reaction_orig2' : 1/2}, ...}
    """
    for r in model.reactions:
        r.gene_reaction_rule = ''
    num_met = len(model.metabolites)
    num_reac = len(model.reactions)
    old_reac_ids = [r.id for r in model.reactions]
    stoich_mat = efm.DefaultBigIntegerRationalMatrix(num_met, num_reac)
    reversible = jpype.JBoolean[:]([r.reversibility for r in model.reactions])
    # start_time = time.monotonic()
    flipped = []
    for i in range(num_reac):
        if model.reactions[i].upper_bound <= 0:  # can run in backwards direction only (is and stays classified as irreversible)
            model.reactions[i] *= -1
            flipped.append(i)
            logging.debug("Flipped " + model.reactions[i].id)
        # have to use _metabolites because metabolites gives only a copy
        for k, v in model.reactions[i]._metabolites.items():
            n, d = efm.sympyRat2jBigIntegerPair(v)
            stoich_mat.setValueAt(model.metabolites.index(k.id), i, efm.BigFraction(n, d))
    # compress
    smc = efm.StoichMatrixCompressor(efm.subset_compression)
    reacNames = jpype.JString[:](model.reactions.list_attr('id'))
    comprec = smc.compress(stoich_mat, reversible, jpype.JString[num_met], reacNames, None)
    subset_matrix = efm.jpypeArrayOfArrays2numpy_mat(comprec.post.getDoubleRows())
    del_rxns = np.logical_not(np.any(subset_matrix, axis=1))  # blocked reactions
    for j in range(subset_matrix.shape[1]):
        rxn_idx = subset_matrix[:, j].nonzero()[0]
        r0 = rxn_idx[0]
        model.reactions[r0].subset_rxns = []
        model.reactions[r0].subset_stoich = []
        for r in rxn_idx:  # rescale all reactions in this subset
            # !! swaps lb, ub when the scaling factor is < 0, but does not change the magnitudes
            factor = efm.jBigFraction2sympyRat(comprec.post.getBigFractionValueAt(r, j))
            # factor = jBigFraction2intORsympyRat(comprec.post.getBigFractionValueAt(r, j)) # does not appear to make a speed difference
            model.reactions[r] *= factor  #subset_matrix[r, j]
            # factor = abs(float(factor)) # context manager has trouble with non-float bounds
            if model.reactions[r].lower_bound not in (0, -float('inf')):
                model.reactions[r].lower_bound /= abs(subset_matrix[r, j])  #factor
            if model.reactions[r].upper_bound not in (0, float('inf')):
                model.reactions[r].upper_bound /= abs(subset_matrix[r, j])  #factor
            model.reactions[r0].subset_rxns.append(r)
            if r in flipped:
                model.reactions[r0].subset_stoich.append(-factor)
            else:
                model.reactions[r0].subset_stoich.append(factor)
        for r in rxn_idx[1:]:  # merge reactions
            # rename main reaction
            if len(model.reactions[r0].id) + len(model.reactions[r].id) < 220 and model.reactions[r0].id[-3:] != '...':
                model.reactions[r0].id += '*' + model.reactions[r].id  # combine names
            elif not model.reactions[r0].id[-3:] == '...':
                model.reactions[r0].id += '...'
            # !! keeps bounds of reactions[rxn_idx[0]]
            model.reactions[r0] += model.reactions[r]
            if model.reactions[r].lower_bound > model.reactions[r0].lower_bound:
                model.reactions[r0].lower_bound = model.reactions[r].lower_bound
            if model.reactions[r].upper_bound < model.reactions[r0].upper_bound:
                model.reactions[r0].upper_bound = model.reactions[r].upper_bound
            del_rxns[r] = True
    del_rxns = np.where(del_rxns)[0]
    for i in range(len(del_rxns) - 1, -1, -1):  # delete in reversed index order to keep indices valid
        model.reactions[del_rxns[i]].remove_from_model(remove_orphans=True)
    subT = np.zeros((num_reac, len(model.reactions)))
    rational_map = {}
    for j in range(subT.shape[1]):
        subT[model.reactions[j].subset_rxns, j] = [float(v) for v in model.reactions[j].subset_stoich]
        # rational_map is a dictionary that associates the new reaction with a dict of its original reactions and its scaling factors
        rational_map.update(
            {model.reactions[j].id: {
                 old_reac_ids[i]: v for i, v in zip(model.reactions[j].subset_rxns, model.reactions[j].subset_stoich)
             }})
    # for i in flipped: # adapt so that it matches the reaction direction before flipping
    #         subT[i, :] *= -1
    return rational_map


def compress_model_parallel(model, protected_rxns=set()):
    """Compress model by lumping parallel reactions
    
    Example:
        cmp_mapReac = compress_model_parallel(model)
    
    Args:
        model (cobra.Model):
            A metabolic model that is an instance of the cobra.Model class
            
    Returns:
        (dict):
        A dict that contains information about the lumping done in the compression process.
        E.g.: {'reaction_lumped1' : {'reaction_orig1' : 1 'reaction_orig2' : 1}, ...}
    """
    #
    # - exclude lumping of reactions with inhomogenous bounds
    # - exclude protected reactions
    old_num_reac = len(model.reactions)
    old_objective = [r.objective_coefficient for r in model.reactions]
    old_reac_ids = [r.id for r in model.reactions]
    stoichmat_T = create_stoichiometric_matrix(model, 'lil').transpose()
    factor = [d[0] if d else 1.0 for d in stoichmat_T.data]
    A = (sparse.diags(factor) @ stoichmat_T)
    lb = [r.lower_bound for r in model.reactions]
    ub = [r.upper_bound for r in model.reactions]
    fwd = sparse.lil_matrix([1. if (np.isinf(u) and f > 0 or np.isinf(l) and f < 0) else 0. for f, l, u in zip(factor, lb, ub)]).transpose()
    rev = sparse.lil_matrix([1. if (np.isinf(l) and f > 0 or np.isinf(u) and f < 0) else 0. for f, l, u in zip(factor, lb, ub)]).transpose()
    inh = sparse.lil_matrix([i+1 if not ((np.isinf(ub[i]) or ub[i] == 0) and (np.isinf(lb[i]) or lb[i] == 0)) \
                                 else 0 for i in range(len(model.reactions))]).transpose()
    A = sparse.hstack((A, fwd, rev, inh), 'csr')
    # find equivalent/parallel reactions
    subset_list = []
    prev_found = []
    protected = [True if r.id in protected_rxns else False for r in model.reactions]
    hashes = [hash((tuple(A[i].indices), tuple(A[i].data))) for i in range(A.shape[0])]
    for i in range(A.shape[0]):
        if i in prev_found:  # if reaction was already found to be identical to another one, skip.
            continue
        if protected[i]:  # if protected, add 1:1 relationship, skip.
            subset_list += [[i]]
            continue
        subset_i = [i]
        for j in range(i + 1, A.shape[0]):  # otherwise, identify parallel reactions
            if not protected[j] and j not in prev_found:
                # if np.all(A[i].indices == A[j].indices) and np.all(A[i].data == A[j].data):
                if hashes[i] == hashes[j]:
                    subset_i += [j]
                    prev_found += [j]
        if subset_i:
            subset_list += [subset_i]
    # lump parallel reactions (delete redundant)
    del_rxns = [False] * len(model.reactions)
    for rxn_idx in subset_list:
        for i in range(1, len(rxn_idx)):
            if len(model.reactions[rxn_idx[0]].id) + len(
                    model.reactions[rxn_idx[i]].id) < 220 and model.reactions[rxn_idx[0]].id[-3:] != '...':
                model.reactions[rxn_idx[0]].id += '*' + model.reactions[rxn_idx[i]].id  # combine names
            elif not model.reactions[rxn_idx[0]].id[-3:] == '...':
                model.reactions[rxn_idx[0]].id += '...'
            del_rxns[rxn_idx[i]] = True
    del_rxns = np.where(del_rxns)[0]
    for i in range(len(del_rxns) - 1, -1, -1):  # delete in reversed index order to keep indices valid
        model.reactions[del_rxns[i]].remove_from_model(remove_orphans=True)
    # create compression map
    rational_map = {}
    subT = np.zeros((old_num_reac, len(model.reactions)))
    for i in range(subT.shape[1]):
        for j in subset_list[i]:
            subT[j, i] = 1
        # rational_map is a dictionary that associates the new reaction with a dict of its original reactions and its scaling factors
        rational_map.update({model.reactions[i].id: {old_reac_ids[j]: One() for j in subset_list[i]}})

    new_objective = old_objective @ subT
    for r, c in zip(model.reactions, new_objective):
        r.objective_coefficient = c
    return rational_map


def remove_blocked_reactions(model) -> List:
    """Remove blocked reactions from a network"""
    blocked_reactions = [reac for reac in model.reactions if reac.bounds == (0, 0)]
    model.remove_reactions(blocked_reactions)
    return blocked_reactions


def remove_ext_mets(model):
    """Remove (unbalanced) external metabolites from the compartment External_Species"""
    external_mets = [i for i, cpts in zip(model.metabolites, model.metabolites.list_attr("compartment")) if cpts == 'External_Species']
    model.remove_metabolites(external_mets)
    stoich_mat = create_stoichiometric_matrix(model)
    obsolete_reacs = [reac for reac, b_rempty in zip(model.reactions, np.any(stoich_mat, 0)) if not b_rempty]
    model.remove_reactions(obsolete_reacs)


def remove_conservation_relations(model):
    """Remove conservation relations in a model
    
    This reduces the number of metabolites in a model while maintaining the 
    original flux space. This is a compression technique."""
    stoich_mat = create_stoichiometric_matrix(model, array_type='lil')
    basic_metabolites = efm.basic_columns_rat(stoich_mat.transpose().toarray(), tolerance=0)
    dependent_metabolites = [model.metabolites[i].id for i in set(range(len(model.metabolites))) - set(basic_metabolites)]
    for m in dependent_metabolites:
        model.metabolites.get_by_id(m).remove_from_model()


# replace all stoichiometric coefficients with rationals.
def stoichmat_coeff2rational(model):
    """Convert coefficients from to stoichiometric matrix to rational numbers"""
    num_reac = len(model.reactions)
    for i in range(num_reac):
        for k, v in model.reactions[i]._metabolites.items():
            if isinstance(v, float) or isinstance(v, int):
                if isinstance(v, int):
                    # v = int(v)
                    # n = int2jBigInteger(v)
                    # d = BigInteger.ONE
                    v = Rational(v)  # for simplicity and actually slighlty faster (?)
                else:
                    rational_conversion = 'base10'  # This was formerly in the function parameters
                    v = nsimplify(v, rational=True, rational_conversion=rational_conversion)
                    # v = sympy.Rational(v)
                model.reactions[i]._metabolites[k] = v  # only changes coefficient in the model, not in the solver
            elif not isinstance(v, Rational):
                raise TypeError


# replace all stoichiometric coefficients with ints and floats
def stoichmat_coeff2float(model):
    """Convert coefficients from to stoichiometric matrix to floats"""
    num_reac = len(model.reactions)
    for i in range(num_reac):
        for k, v in model.reactions[i]._metabolites.items():
            if isinstance(v, float) or isinstance(v, int) or isinstance(v, Rational):
                model.reactions[i]._metabolites[k] = float(v)
            else:
                raise Exception('unknown data type')


def modules_coeff2rational(sd_modules):
    """Convert coefficients occurring in SDModule objects to rational numbers"""
    for i, module in enumerate(sd_modules):
        for param in [CONSTRAINTS, INNER_OBJECTIVE, OUTER_OBJECTIVE, PROD_ID]:
            if param in module and module[param] is not None:
                if param == CONSTRAINTS:
                    for constr in module[CONSTRAINTS]:
                        for reac in constr[0].keys():
                            constr[0][reac] = nsimplify(constr[0][reac])
                if param in [INNER_OBJECTIVE, OUTER_OBJECTIVE, PROD_ID]:
                    for reac in module[param].keys():
                        module[param][reac] = nsimplify(module[param][reac])
    return sd_modules


def modules_coeff2float(sd_modules):
    """Convert coefficients occurring in SDModule objects to floats"""
    for i, module in enumerate(sd_modules):
        for param in [CONSTRAINTS, INNER_OBJECTIVE, OUTER_OBJECTIVE, PROD_ID]:
            if param in module and module[param] is not None:
                if param == CONSTRAINTS:
                    for constr in module[CONSTRAINTS]:
                        for reac in constr[0].keys():
                            constr[0][reac] = float(constr[0][reac])
                if param in [INNER_OBJECTIVE, OUTER_OBJECTIVE, PROD_ID]:
                    for reac in module[param].keys():
                        module[param][reac] = float(module[param][reac])
    return sd_modules


def remove_dummy_bounds(model):
    """Replace COBRA standard bounds with +/-inf
    
    Retrieve the standard bounds from the COBRApy Configuration and replace model bounds
    of the same value with +/-inf.
    """
    cobra_conf = Configuration()
    bound_thres = max((abs(cobra_conf.lower_bound), abs(cobra_conf.upper_bound)))
    if any([any([abs(b) >= bound_thres for b in r.bounds]) for r in model.reactions]):
        logging.warning('  Removing reaction bounds when larger than the cobra-threshold of ' + str(round(bound_thres)) + '.')
        for i in range(len(model.reactions)):
            if model.reactions[i].lower_bound <= -bound_thres:
                model.reactions[i].lower_bound = -np.inf
            if model.reactions[i].upper_bound >= bound_thres:
                model.reactions[i].upper_bound = np.inf


def bound_blocked_or_irrevers_fva(model, solver=None):
    """Use FVA to determine the flux ranges. Use this information to update the model bounds
    
    If flux ranges for a reaction are narrower than its bounds in the mode, these bounds can be omitted, 
    since other reactions must constrain the reaction flux. If (upper or lower) flux bounds are found to 
    be zero, the model bounds are updated to reduce the model complexity.
    """
    # FVAs to identify blocked and irreversible reactions, as well as non-bounding bounds
    flux_limits = fva(model)
    if select_solver(solver) in [SCIP, GLPK]:
        tol = 1e-10  # use tolerance for tightening problem bounds
    else:
        tol = 0.0
    for (reac_id, limits) in flux_limits.iterrows():
        r = model.reactions.get_by_id(reac_id)
        # modify _lower_bound and _upper_bound to make changes permanent
        if r.lower_bound < 0.0 and limits.minimum - tol > r.lower_bound:
            r._lower_bound = -np.inf
        if limits.minimum >= tol:
            r._lower_bound = max([0.0, r._lower_bound])
        if r.upper_bound > 0.0 and limits.maximum + tol < r.upper_bound:
            r._upper_bound = np.inf
        if limits.maximum <= -tol:
            r._upper_bound = min([0.0, r._upper_bound])
