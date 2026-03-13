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
"""Container for strain design solutions (SDSolutions)"""

from numpy import all, any, nan, isnan, sign
from typing import List, Dict, Tuple, Union, Set, FrozenSet
from straindesign.parse_constr import *
from straindesign.names import *
import re
import json
import pickle
import logging


class SDSolutions(object):
    """Container for strain design solutions

    Objects of this class are returned by strain design computations. This class
    contains the metabolic interventions on the gene, reaction or regulation level
    alongside with information about the strain design setup, including the model
    used and the strain design modules. Strain design solutions can be accessed
    either through the fields or through specific functions that preprocess or
    reformat strain designs for different purposes.

    Instances of this class are not meant to be created by StrainDesign users.

    Args:
        model (cobra.Model):
            A metabolic model that is an instance of the cobra.Model class.

        sd (list of dict):
            A list of dicts every dict represents an intervention set. Keys in
            each dict are reaction/gene identifiers and the associated value
            determines if it is added (1), not added (0) or knocked out (-1).
            For regulatory interventions, (1) means active regulation and
            (0) means regulatory intervention not added. These will be translated
            to True and False.

        status (str):
            Status string of the computation (e.g.: 'optimal')

        sd_setup (dict):
            A dictionary containing information about the problem setup. This dict can/should contain
            the keys MODEL_ID, MODULES, MAX_SOLUTIONS, MAX_COST, TIME_LIMIT, SOLVER, KOCOST, KICOST,
            REGCOST, GKICOST, GKOCOST

            These entries can be set like this:
            sd_setup[straindesign.MODEL_ID] = model.id

    Returns
        (SDSolutions):
        Strain design solutions

    """

    def __init__(self, model, sd, status, sd_setup, *, _lazy_init=None):
        self.status = status
        self.sd_setup = sd_setup
        self._lazy = _lazy_init is not None
        self._expanded_groups = set()
        self._expansion_meta = _lazy_init if _lazy_init else {}

        if GKOCOST in sd_setup or GKICOST in sd_setup:
            logging.info('  Preparing (reaction-)phenotype prediction of gene intervention strategies.')
            self.reaction_sd, self.gene_sd = self._translate_genes_to_reactions(sd, model)
            self.is_gene_sd = True
            cost_sd = [s.copy() for s in self.gene_sd]
        else:
            self.reaction_sd = sd
            self.is_gene_sd = False
            cost_sd = sd

        self.sd_cost, self.itv_bounds, self.has_complex_regul_itv = \
            self._compute_costs_and_bounds(cost_sd, self.reaction_sd, model, sd_setup)

        if self._lazy:
            self._estimated_total = _lazy_init.get('estimated_total', len(self.reaction_sd))

    @staticmethod
    def _translate_genes_to_reactions(sd_list, model):
        """Translate gene-level solution dicts to reaction-level.

        Takes raw solution dicts (with gene IDs) and returns
        (reaction_sd_list, gene_sd_list) where gene_sd_list preserves
        original gene names/IDs before any name-to-ID replacement.
        """
        gene_sd = [s.copy() for s in sd_list]
        working_sd = [s.copy() for s in sd_list]
        interventions = set()
        [[interventions.add(k) for k in s.keys()] for s in working_sd]
        # replace gene names with identifiers if necessary
        gene_name_id_dict = {}
        [gene_name_id_dict.update({g.name: g.id}) for g in model.genes if interventions.intersection([g.name])]
        for g_name, g_id in gene_name_id_dict.items():
            for s in working_sd:
                if g_name in s:
                    s.update({g_id: s.pop(g_name)})
            interventions.remove(g_name)
            interventions.add(g_id)
        # get potential gene- and reaction interventions and potentially affected reactions
        reac_itv = set(v for v in interventions if model.reactions.has_id(v))
        gene_itv = set(v for v in interventions if v in model.genes.list_attr('name') + model.genes.list_attr('id'))
        regl_itv = set(v for v in interventions if v not in reac_itv and v not in gene_itv)
        affected_reacs = set()
        [[affected_reacs.add(r.id) for r in g.reactions] for g in model.genes]
        gpr = {}
        for r in affected_reacs:
            gpr_i = model.reactions.get_by_id(r).gene_reaction_rule
            cj_terms = gpr_i.split(' or ')
            for i, c in enumerate(cj_terms):
                cj_terms[i] = c.split(' and ')
                for j, g in enumerate(cj_terms[i]):
                    cj_terms[i][j] = re.sub(r'^(\s|-|\+|\()*|(\s|-|\+|\))*$', '', g)
            gpr.update({r: cj_terms})

        # translate every cut set to reaction intervention sets
        reaction_sd = [{} for _ in range(len(working_sd))]
        for i, s in enumerate(working_sd):
            reac_ko = set(k for k, v in s.items() if v < 0 and (k in reac_itv))
            reac_ki = set(k for k, v in s.items() if v > 0 and (k in reac_itv))
            reac_no_ki = set(k for k, v in s.items() if v == 0 and (k in reac_itv))
            reg_itv = set(k for k, v in s.items() if v and (k in regl_itv))
            reg_no_itv = set(k for k, v in s.items() if not v and (k in regl_itv))
            gene_ko = {k: False for k, v in s.items() if v < 0 and (k in gene_itv)}
            gene_ki = {k: True for k, v in s.items() if v > 0 and (k in gene_itv)}
            gene_no_ki = {k: False for k, v in s.items() if v == 0 and (k in gene_itv)}
            gene_ki_inv = {k: False for k in gene_ki.keys()}
            for r in affected_reacs:
                # is_possible = gpr[r].subs({**gene_ko,**gene_ki,**gene_no_ki})
                is_possible = gpr_eval(gpr[r], {**gene_ko, **gene_ki, **gene_no_ki})
                if is_possible != False:  # reaction is only feasible because of KIs
                    # is_possible_wo_ki = gpr[r].subs({**gene_ko,**gene_ki_inv,**gene_no_ki})
                    is_possible_wo_ki = gpr_eval(gpr[r], {**gene_ko, **gene_ki_inv, **gene_no_ki})
                    if is_possible_wo_ki == False:
                        reac_ki.add(r)
                elif is_possible == False:
                    # is_possible_wo_ko = gpr[r].subs({**gene_ki,**gene_no_ki})
                    is_possible_wo_ko = gpr_eval(gpr[r], {**gene_ki, **gene_no_ki})
                    if is_possible_wo_ko != False:
                        reac_ko.add(r)
                    else:
                        reac_no_ki.add(r)
                else:  # reaction is not (sufficiently) affected by interventions
                    pass
            reaction_sd[i].update({k: -1.0 for k in reac_ko})
            reaction_sd[i].update({k: 1.0 for k in reac_ki})
            reaction_sd[i].update({k: 0.0 for k in reac_no_ki})
            reaction_sd[i].update({k: True for k in reg_itv})
            reaction_sd[i].update({k: False for k in reg_no_itv})
        return reaction_sd, gene_sd

    @staticmethod
    def _compute_costs_and_bounds(cost_sd, reaction_sd, model, sd_setup):
        """Compute intervention costs and reaction bounds for solutions.

        Args:
            cost_sd: list of dicts for cost lookup (gene_sd copies for gene mode,
                     original sd for reaction mode)
            reaction_sd: list of reaction-level solution dicts
            model: cobra Model (for bounds lookup)
            sd_setup: setup dict containing cost dictionaries

        Returns:
            (sd_cost, itv_bounds, has_complex_regul_itv)
        """
        # compute intervention costs
        sd_cost = [0 for _ in range(len(cost_sd))]
        if KOCOST in sd_setup:
            for k, v in sd_setup[KOCOST].items():
                for i, s in enumerate(cost_sd):
                    if k in s and s[k] != 0:
                        sd_cost[i] += float(v)
        if KICOST in sd_setup:
            for k, v in sd_setup[KICOST].items():
                for i, s in enumerate(cost_sd):
                    if k in s and s[k] != 0:
                        sd_cost[i] += float(v)
        if GKOCOST in sd_setup:
            for k, v in sd_setup[GKOCOST].items():
                for i, s in enumerate(cost_sd):
                    if k in s and s[k] != 0:
                        sd_cost[i] += float(v)
        if GKICOST in sd_setup:
            for k, v in sd_setup[GKICOST].items():
                for i, s in enumerate(cost_sd):
                    if k in s and s[k] != 0:
                        sd_cost[i] += float(v)
        if REGCOST in sd_setup:
            for k, v in sd_setup[REGCOST].items():
                for i, s in enumerate(cost_sd):
                    if k in s and s[k] != 0:
                        sd_cost[i] += float(v)

        has_complex_regul_itv = False
        itv_bounds = [{} for _ in range(len(reaction_sd))]
        for i, s in enumerate(reaction_sd):
            for k, v in s.items():
                if type(v) is not bool:
                    if v == -1:  # reaction was knocked out
                        itv_bounds[i].update({k: (0.0, 0.0)})
                    elif v == 1:  # reaction was added
                        itv_bounds[i].update({k: model.reactions.get_by_id(k).bounds})
                    elif v == 0:  # reaction was not added
                        itv_bounds[i].update({k: (nan, nan)})
            for k, v in s.items():
                if type(v) == bool and v:
                    try:
                        lineq = lineq2list([k], model.reactions.list_attr('id'))[0]
                    except:
                        has_complex_regul_itv = True
                        continue
                    lhs = lineq[0]
                    if len(lhs) != 1:
                        has_complex_regul_itv = True
                    else:
                        eqsign = lineq[1]
                        rhs = lineq[2]
                        reac = list(lhs.keys())[0]
                        coeff = list(lhs.values())[0]
                        if reac in itv_bounds[i]:
                            bnds = itv_bounds[i].pop(reac)
                        else:
                            bnds = model.reactions.get_by_id(reac).bounds
                        if eqsign == '=':
                            bnds = (rhs / coeff, rhs / coeff)
                        elif (eqsign == '<=') == (sign(coeff) > 0):
                            bnds = (bnds[0], rhs / coeff)
                        else:
                            bnds = (rhs / coeff, bnds[1])
                        itv_bounds[i].update({reac: bnds})
        return sd_cost, itv_bounds, has_complex_regul_itv

    def get_num_sols(self):
        """Get number of solutions"""
        if self._lazy:
            return self._estimated_total
        return len(self.reaction_sd)

    def get_strain_design_costs(self, i=None):
        """Get costs of i-th strain design or of all in a list"""
        if i is None:
            return self.sd_cost
        else:
            return get_subset(self.sd_cost, i)

    def get_strain_designs(self, i=None):
        """Get i-th strain design (intervention set) or all in original format"""
        if self.is_gene_sd:
            return self.get_gene_sd(i)
        else:
            return self.get_reaction_sd(i)

    def get_reaction_sd(self, i=None):
        """Get reaction-based strain design solutions

        Gene-based intervention sets are translated to the reaction level. This can
        be helpful to understand the impact of gene interventions. GPR-rules are
        accounted for automatically.
        """
        if i is None:
            return [strip_non_ki(s) for s in self.reaction_sd]
        else:
            if type(i) == int:
                i = [i]
            return [strip_non_ki(s) for j, s in enumerate(self.reaction_sd) if j in i]

    def get_reaction_sd_bnds(self, i=None):
        """Get reaction-based strain design solutions represented by upper and lower bounds

        Knocked-out reactions will show as upper and lower bounds of zero.
        """
        if i is None:
            return self.itv_bounds
        else:
            if type(i) == int:
                i = [i]
            return [self.itv_bounds for j, s in enumerate(self.itv_bounds) if j in i]

    def get_gene_sd(self, i=None):
        """Get gene-based strain design solutions"""
        if not self.is_gene_sd:
            raise Exception('The solutions are based on reaction interventions only.')
        if i is None:
            return [strip_non_ki(s) for s in self.gene_sd]
        else:
            if type(i) == int:
                i = [i]
            return [strip_non_ki(s) for j, s in enumerate(self.gene_sd) if j in i]

    def get_reaction_sd_mark_no_ki(self, i=None):
        """Get reaction-based strain design solutions,
        but also tag knock-ins that were not made with a 0

        This can be helpful to analyze gene intervention sets in original metabolic models.
        GPR-rules are accounted for automatically."""
        if i is None:
            return self.reaction_sd
        else:
            if type(i) == int:
                i = [i]
            return get_subset(self.reaction_sd, i)

    def get_gene_sd_mark_no_ki(self, i=None):
        """Get gene-based strain design solutions,
        but also tag knock-ins that were not made with a 0"""
        if not self.is_gene_sd:
            raise Exception('The solutions are based on reaction interventions only.')
        if i is None:
            return self.gene_sd
        else:
            if type(i) == int:
                i = [i]
            return get_subset(self.gene_sd, i)

    def get_gene_reac_sd_assoc(self, i=None):
        """Get reaction and gene-based strain design solutions,
        and show which reaction-based solution corresponds to which gene-based.

        Often the association is not 1:1 but n:1."""
        if not self.is_gene_sd:
            raise Exception('The solutions are based on reaction interventions only.')
        if i is None:
            i = [j for j in range(len(self.gene_sd))]
        else:
            if type(i) == int:
                i = [i]
        reacs_sd_hash = []
        reacs_sd = []
        assoc = []
        gene_sd = [strip_non_ki(s) for j, s in enumerate(self.gene_sd) if j in i]
        for s in [strip_non_ki(t) for j, t in enumerate(self.reaction_sd) if j in i]:
            hs = hash(json.dumps(s, sort_keys=True))
            if hs not in reacs_sd_hash:
                reacs_sd_hash.append(hs)
                reacs_sd.append(s)
            assoc.append(reacs_sd_hash.index(hs))
        return reacs_sd, assoc, gene_sd

    def get_gene_reac_sd_assoc_mark_no_ki(self, i=None):
        """Get reaction and gene-based strain design solutions,
        but also tag knock-ins that were not made with a 0

        Often the association is not 1:1 but n:1."""
        if not self.is_gene_sd:
            raise Exception('The solutions are based on reaction interventions only.')
        if i is None:
            i = [j for j in range(len(self.gene_sd))]
        else:
            if type(i) == int:
                i = [i]
        reacs_sd_hash = []
        reacs_sd = []
        assoc = []
        gene_sd = [s for j, s in enumerate(self.gene_sd) if j in i]
        for s in [t for j, t in enumerate(self.reaction_sd) if j in i]:
            hs = hash(json.dumps(s, sort_keys=True))
            if hs not in reacs_sd_hash:
                reacs_sd_hash.append(hs)
                reacs_sd.append(s)
            assoc.append(reacs_sd_hash.index(hs))
        return reacs_sd, assoc, gene_sd

    def get_group(self, i):
        """Get all expanded solution indices that belong to the same compressed group as solution i.

        Returns a list of indices into reaction_sd that share the same compressed solution origin.
        Requires that compute_strain_designs was called with compression enabled.
        """
        if not hasattr(self, 'group_map') or not self.group_map:
            raise AttributeError('No group information available. Run compute_strain_designs with compression enabled.')
        grp = self.group_map[i]
        return [j for j, g in enumerate(self.group_map) if g == grp]

    def get_num_groups(self):
        """Get the number of distinct compressed solution groups."""
        if not hasattr(self, 'group_map') or not self.group_map:
            return len(self.reaction_sd)
        return len(set(self.group_map))

    def get_representative_sd(self):
        """Get one representative expanded solution per compressed group.

        Returns a list of dicts, one per unique compressed solution.
        """
        if not hasattr(self, 'group_map') or not self.group_map:
            return self.get_reaction_sd()
        seen = set()
        reps = []
        for i, grp in enumerate(self.group_map):
            if grp not in seen:
                seen.add(grp)
                reps.append(strip_non_ki(self.reaction_sd[i]))
        return reps

    def expand_group(self, grp_idx):
        """Expand one compressed group on demand.

        Returns list of expanded solution dicts. Also updates reaction_sd,
        sd_cost, itv_bounds, and group_map in place.
        """
        if not self._lazy:
            raise RuntimeError('expand_group requires lazy mode')
        if grp_idx in self._expanded_groups:
            return [self.reaction_sd[j] for j, g in enumerate(self.group_map) if g == grp_idx]

        from straindesign.networktools import expand_sd, filter_sd_maxcost
        meta = self._expansion_meta
        compressed_sd = meta['compressed_sd']
        cmp_mapReac = meta['compression_map']

        # Expand
        expanded = expand_sd([compressed_sd[grp_idx].copy()], cmp_mapReac)
        expanded = filter_sd_maxcost(expanded, meta['max_cost'],
                                     meta['uncmp_ko_cost'], meta['uncmp_ki_cost'])
        # Postprocess regulatory interventions (inline to avoid circular import)
        reg_cost = meta.get('uncmp_reg_cost', {})
        for s in expanded:
            for k, v in reg_cost.items():
                if k in s:
                    s.pop(k)
                    s.update({v['str']: True})
                else:
                    s.update({v['str']: False})

        # GPR translation + costs/bounds
        model = meta['model']
        if self.is_gene_sd:
            reaction_sd_exp, gene_sd_exp = self._translate_genes_to_reactions(expanded, model)
            cost_sd = [s.copy() for s in gene_sd_exp]
        else:
            reaction_sd_exp = expanded
            gene_sd_exp = None
            cost_sd = expanded

        sd_cost_exp, itv_bounds_exp, has_complex = self._compute_costs_and_bounds(
            cost_sd, reaction_sd_exp, model, self.sd_setup)
        if has_complex:
            self.has_complex_regul_itv = True

        # Remove the single representative for this group
        rep_indices = [j for j, g in enumerate(self.group_map) if g == grp_idx]
        for j in sorted(rep_indices, reverse=True):
            self.reaction_sd.pop(j)
            self.sd_cost.pop(j)
            self.itv_bounds.pop(j)
            self.group_map.pop(j)
            if self.is_gene_sd and hasattr(self, 'gene_sd'):
                self.gene_sd.pop(j)

        # Append expanded solutions
        for idx in range(len(reaction_sd_exp)):
            self.reaction_sd.append(reaction_sd_exp[idx])
            self.sd_cost.append(sd_cost_exp[idx])
            self.itv_bounds.append(itv_bounds_exp[idx])
            self.group_map.append(grp_idx)
            if self.is_gene_sd and gene_sd_exp is not None:
                self.gene_sd.append(gene_sd_exp[idx])

        self._expanded_groups.add(grp_idx)
        return reaction_sd_exp

    def expand_all(self, n_per_group=None):
        """Expand all compressed groups.

        Args:
            n_per_group: None means all, int means keep up to n per group.
        """
        if not self._lazy:
            return
        compressed_sd = self._expansion_meta['compressed_sd']
        for grp_idx in range(len(compressed_sd)):
            if grp_idx not in self._expanded_groups:
                expanded = self.expand_group(grp_idx)
                if n_per_group is not None and len(expanded) > n_per_group:
                    # Keep only first n_per_group from this group
                    indices = [j for j, g in enumerate(self.group_map) if g == grp_idx]
                    for j in sorted(indices[n_per_group:], reverse=True):
                        self.reaction_sd.pop(j)
                        self.sd_cost.pop(j)
                        self.itv_bounds.pop(j)
                        self.group_map.pop(j)
                        if self.is_gene_sd and hasattr(self, 'gene_sd'):
                            self.gene_sd.pop(j)
        self._lazy = False

    @property
    def is_lazy(self):
        """True if lazy expansion is active (some groups unexpanded)."""
        return self._lazy

    def get_num_materialized(self):
        """Number of currently materialized solutions in reaction_sd."""
        return len(self.reaction_sd)

    def save(self, filename):
        """Save strain design solutions to a file."""
        if self._lazy:
            self.expand_all()
        # Don't persist model reference from expansion metadata
        meta_backup = self._expansion_meta
        self._expansion_meta = {}
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
        finally:
            self._expansion_meta = meta_backup

    @classmethod
    def load(cls, filename):
        """Load strain design solutions from a file."""
        with open(filename, 'rb') as f:
            cls = pickle.load(f)
        return cls

    def _check_merge_compatible(self, other):
        """Raises ValueError if SDSolutions objects cannot be merged."""
        if self.sd_setup.get(MODEL_ID) != other.sd_setup.get(MODEL_ID):
            raise ValueError('Cannot merge SDSolutions with different model IDs')
        if self.is_gene_sd != other.is_gene_sd:
            raise ValueError('Cannot merge gene-level and reaction-level SDSolutions')
        if hasattr(self, 'compression_map') and hasattr(other, 'compression_map'):
            if len(self.compression_map) != len(other.compression_map):
                raise ValueError('Cannot merge SDSolutions with different compression depths')
            for s_step, o_step in zip(self.compression_map, other.compression_map):
                if set(s_step['reac_map_exp'].keys()) != set(o_step['reac_map_exp'].keys()):
                    raise ValueError('Cannot merge SDSolutions with different compression maps')
                if s_step['parallel'] != o_step['parallel']:
                    raise ValueError('Cannot merge SDSolutions with different compression types')

    def __iadd__(self, other):
        """In-place merge of two SDSolutions objects (deduplicates at compressed level)."""
        self._check_merge_compatible(other)

        if not hasattr(self, 'compressed_sd') or not hasattr(other, 'compressed_sd'):
            # No compression info — merge at expanded level with deduplication
            existing = {frozenset(s.items()) for s in self.reaction_sd}
            for j in range(len(other.reaction_sd)):
                key = frozenset(other.reaction_sd[j].items())
                if key not in existing:
                    self.reaction_sd.append(other.reaction_sd[j])
                    self.sd_cost.append(other.sd_cost[j])
                    self.itv_bounds.append(other.itv_bounds[j])
                    if self.is_gene_sd:
                        self.gene_sd.append(other.gene_sd[j])
                    existing.add(key)
            if self.status not in [OPTIMAL, TIME_LIMIT_W_SOL]:
                if other.status in [OPTIMAL, TIME_LIMIT_W_SOL]:
                    self.status = other.status
            return self

        # Deduplicate at compressed solution level
        existing = {frozenset(s.items()) for s in self.compressed_sd}
        grp_offset = (max(self.group_map) + 1) if self.group_map else 0

        for other_grp_idx in range(len(other.compressed_sd)):
            cmp_s = other.compressed_sd[other_grp_idx]
            if frozenset(cmp_s.items()) in existing:
                continue  # skip duplicate
            self.compressed_sd.append(cmp_s)
            new_grp = grp_offset
            grp_offset += 1

            # Copy expanded solutions for this group
            other_indices = [j for j, g in enumerate(other.group_map) if g == other_grp_idx]
            for j in other_indices:
                self.reaction_sd.append(other.reaction_sd[j])
                self.sd_cost.append(other.sd_cost[j])
                self.itv_bounds.append(other.itv_bounds[j])
                self.group_map.append(new_grp)
                if self.is_gene_sd:
                    self.gene_sd.append(other.gene_sd[j])

            # Track lazy expansion state
            if self._lazy:
                if other._lazy and other_grp_idx in other._expanded_groups:
                    self._expanded_groups.add(new_grp)
                elif not other._lazy:
                    self._expanded_groups.add(new_grp)

        # Status: OPTIMAL wins
        if self.status not in [OPTIMAL, TIME_LIMIT_W_SOL]:
            if other.status in [OPTIMAL, TIME_LIMIT_W_SOL]:
                self.status = other.status
        return self

    def __add__(self, other):
        """Merge two SDSolutions objects, returning a new object."""
        from copy import deepcopy
        result = deepcopy(self)
        result += other
        return result


def strip_non_ki(sd):
    """SDSolutions internal function: removing non-added reactions or genes"""
    return {k: v for k, v in sd.items() if v not in (0.0, False)}


def get_subset(sd, i):
    """SDSolutions internal function: getting a subset of solutions"""
    return [s for j, s in enumerate(sd) if j in i]


def gpr_eval(cj_terms, interv):
    """SDSolutions internal function: evaluate a GPR term"""
    gpr_ev = [0.0 for _ in range(len(cj_terms))]
    for i, c in enumerate(cj_terms):
        cj_term = [interv[k] if k in interv else nan for k in c]
        if any([v == False for v in cj_term]):
            gpr_ev[i] = False
        elif any(isnan(cj_term)):
            gpr_ev[i] = nan
        elif all(cj_term):
            gpr_ev[i] = True

    if any([v == True for v in gpr_ev]):
        return True
    elif all([v == False for v in gpr_ev]):
        return False
    elif any(isnan(gpr_ev)):
        return nan
    else:
        raise Exception('Shoud not happen')
