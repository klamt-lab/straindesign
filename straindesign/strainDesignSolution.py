# 2022 Max Planck institute for dynamics of complex technical systems.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from numpy import all, any, nan, isnan, inf
from typing import List, Dict, Tuple, Union, Set, FrozenSet
from straindesign.parse_constr import *
from straindesign.names import *
import re
import json
import pickle
import logging


class SDSolution(object):

    def __init__(self, model, sd, status, sd_setup):
        self.status = status
        self.sd_setup = sd_setup
        if GKOCOST in sd_setup or GKICOST in sd_setup:
            self.gene_sd = [s.copy() for s in sd]
            self.is_gene_sd = True
            logging.info(
                '  Preparing (reaction-)phenotype prediction of gene intervention strategies.'
            )
            interventions = set()
            [[interventions.add(k) for k in s.keys()] for s in sd]
            # replace gene names with identifiers if necessary
            gene_name_id_dict = {}
            [
                gene_name_id_dict.update({g.name: g.id})
                for g in model.genes
                if interventions.intersection([g.name])
            ]
            for g_name, g_id in gene_name_id_dict.items():
                for s in sd:
                    if g_name in s:
                        s.update({g_id: s.pop(g_name)})
                interventions.remove(g_name)
                interventions.add(g_id)
            # get potential gene- and reaction interventions and potentially affected reactions
            reac_itv = set(
                v for v in interventions if model.reactions.has_id(v))
            gene_itv = set(v for v in interventions
                           if v in model.genes.list_attr('name') +
                           model.genes.list_attr('id'))
            regl_itv = set(v for v in interventions
                           if v not in reac_itv and v not in gene_itv)
            affected_reacs = set()
            [[affected_reacs.add(r.id)
              for r in g.reactions]
             for g in model.genes]
            gpr = {}
            for r in affected_reacs:
                gpr_i = model.reactions.get_by_id(r).gene_reaction_rule
                cj_terms = gpr_i.split(' or ')
                for i, c in enumerate(cj_terms):
                    cj_terms[i] = c.split(' and ')
                    for j, g in enumerate(cj_terms[i]):
                        cj_terms[i][j] = re.sub(
                            r'^(\s|-|\+|\()*|(\s|-|\+|\))*$', '', g)
                gpr.update({r: cj_terms})

            # # get gpr rules
            # gpr = {r.id:r.gene_reaction_rule for r in model.reactions if r.id in affected_reacs}
            # [gpr.update({k:parse_expr(v.replace(' or ',' | ').replace(' and ',' & '))}) for k,v in gpr.items()]
            # translate every cut set to reaction intervention sets
            self.reaction_sd = [{} for _ in range(len(sd))]
            for i, s in enumerate(sd):
                reac_ko = set(
                    k for k, v in s.items() if v < 0 and (k in reac_itv))
                reac_ki = set(
                    k for k, v in s.items() if v > 0 and (k in reac_itv))
                reac_no_ki = set(
                    k for k, v in s.items() if v == 0 and (k in reac_itv))
                reg_itv = set(k for k, v in s.items() if v and (k in regl_itv))
                reg_no_itv = set(
                    k for k, v in s.items() if not v and (k in regl_itv))
                gene_ko = {
                    k: False for k, v in s.items() if v < 0 and (k in gene_itv)
                }
                gene_ki = {
                    k: True for k, v in s.items() if v > 0 and (k in gene_itv)
                }
                gene_no_ki = {
                    k: False for k, v in s.items() if v == 0 and (k in gene_itv)
                }
                gene_ki_inv = {k: False for k in gene_ki.keys()}
                for r in affected_reacs:
                    # is_possible = gpr[r].subs({**gene_ko,**gene_ki,**gene_no_ki})
                    is_possible = gpr_eval(gpr[r], {
                        **gene_ko,
                        **gene_ki,
                        **gene_no_ki
                    })
                    if is_possible != False:  # reaction is only feasible because of KIs
                        # is_possible_wo_ki = gpr[r].subs({**gene_ko,**gene_ki_inv,**gene_no_ki})
                        is_possible_wo_ki = gpr_eval(gpr[r], {
                            **gene_ko,
                            **gene_ki_inv,
                            **gene_no_ki
                        })
                        if is_possible_wo_ki == False:
                            reac_ki.add(r)
                    elif is_possible == False:
                        # is_possible_wo_ko = gpr[r].subs({**gene_ki,**gene_no_ki})
                        is_possible_wo_ko = gpr_eval(gpr[r], {
                            **gene_ki,
                            **gene_no_ki
                        })
                        if is_possible_wo_ko != False:
                            reac_ko.add(r)
                        else:
                            reac_no_ki.add(r)
                    else:  # reaction is not (sufficiently) affected by interventions
                        pass
                self.reaction_sd[i].update({k: -1.0 for k in reac_ko})
                self.reaction_sd[i].update({k: 1.0 for k in reac_ki})
                self.reaction_sd[i].update({k: 0.0 for k in reac_no_ki})
                self.reaction_sd[i].update({k: True for k in reg_itv})
                self.reaction_sd[i].update({k: False for k in reg_no_itv})
        else:
            self.reaction_sd = sd
            self.is_gene_sd = False

        # compute intervention costs
        self.sd_cost = [0 for _ in range(len(sd))]
        if KOCOST in sd_setup:
            for k, v in sd_setup[KOCOST].items():
                for i, s in enumerate(sd):
                    if k in s and s[k] != 0:
                        self.sd_cost[i] += float(v)
        if KICOST in sd_setup:
            for k, v in sd_setup[KICOST].items():
                for i, s in enumerate(sd):
                    if k in s and s[k] != 0:
                        self.sd_cost[i] += float(v)
        if GKOCOST in sd_setup:
            for k, v in sd_setup[GKOCOST].items():
                for i, s in enumerate(sd):
                    if k in s and s[k] != 0:
                        self.sd_cost[i] += float(v)
        if GKICOST in sd_setup:
            for k, v in sd_setup[GKICOST].items():
                for i, s in enumerate(sd):
                    if k in s and s[k] != 0:
                        self.sd_cost[i] += float(v)
        if REGCOST in sd_setup:
            for k, v in sd_setup[REGCOST].items():
                for i, s in enumerate(sd):
                    if k in s and s[k] != 0:
                        self.sd_cost[i] += float(v)

        self.has_complex_regul_itv = False
        self.itv_bounds = [{} for _ in range(len(self.reaction_sd))]
        for i, s in enumerate(self.reaction_sd):
            for k, v in s.items():
                if type(v) is not bool:
                    if v == -1:  # reaction was knocked out
                        self.itv_bounds[i].update({k: (0.0, 0.0)})
                    elif v == 1:  # reaction was added
                        self.itv_bounds[i].update(
                            {k: model.reactions.get_by_id(k).bounds})
                    elif v == 0:  # reaction was not added
                        self.itv_bounds[i].update({k: (nan, nan)})
            for k, v in s.items():
                if type(v) == bool and v:
                    try:
                        lineq = lineq2list([k],
                                           model.reactions.list_attr('id'))[0]
                    except:
                        self.has_complex_regul_itv = True
                        continue
                    lhs = lineq[0]
                    if len(lhs) != 1:
                        self.has_complex_regul_itv = True
                    else:
                        eqsign = lineq[1]
                        rhs = lineq[2]
                        reac = list(lhs.keys())[0]
                        coeff = list(lhs.values())[0]
                        if reac in self.itv_bounds[i]:
                            bnds = self.itv_bounds[i].pop(reac)
                        else:
                            bnds = model.reactions.get_by_id(reac).bounds
                        if eqsign == '=':
                            bnds = (rhs / coeff, rhs / coeff)
                        elif eqsign == '<=':
                            bnds = (bnds[0], rhs / coeff)
                        elif eqsign == '>=':
                            bnds = (rhs / coeff, bnds[1])
                        self.itv_bounds[i].update({reac: bnds})

    def get_num_sols(self):
        return len(self.reaction_sd)

    def get_strain_design_costs(self, i=None):
        if i is None:
            return self.sd_cost
        else:
            return get_subset(self.sd_cost, i)

    def get_strain_designs(self, i=None):
        if self.is_gene_sd:
            return self.get_gene_sd(i)
        else:
            return self.get_reaction_sd(i)

    def get_reaction_sd(self, i=None):
        if i is None:
            return [strip_non_ki(s) for s in self.reaction_sd]
        else:
            if type(i) == int:
                i = [i]
            return [
                strip_non_ki(s)
                for j, s in enumerate(self.reaction_sd)
                if j in i
            ]

    def get_reaction_sd_bnds(self, i=None):
        if i is None:
            return self.itv_bounds
        else:
            if type(i) == int:
                i = [i]
            return [
                self.itv_bounds for j, s in enumerate(self.itv_bounds) if j in i
            ]

    def get_gene_sd(self, i=None):
        if not self.is_gene_sd:
            raise Exception(
                'The solutions are based on reaction interventions only.')
        if i is None:
            return [strip_non_ki(s) for s in self.gene_sd]
        else:
            if type(i) == int:
                i = [i]
            return [
                strip_non_ki(s) for j, s in enumerate(self.gene_sd) if j in i
            ]

    def get_reaction_sd_mark_no_ki(self, i=None):
        if i is None:
            return self.reaction_sd
        else:
            if type(i) == int:
                i = [i]
            return get_subset(self.reaction_sd, i)

    def get_gene_sd_mark_no_ki(self, i=None):
        if not self.is_gene_sd:
            raise Exception(
                'The solutions are based on reaction interventions only.')
        if i is None:
            return self.gene_sd
        else:
            if type(i) == int:
                i = [i]
            return get_subset(self.gene_sd, i)

    def get_gene_reac_sd_assoc(self, i=None):
        if not self.is_gene_sd:
            raise Exception(
                'The solutions are based on reaction interventions only.')
        if i is None:
            i = [j for j in range(len(self.gene_sd))]
        else:
            if type(i) == int:
                i = [i]
        reacs_sd_hash = []
        reacs_sd = []
        assoc = []
        gene_sd = [
            strip_non_ki(s) for j, s in enumerate(self.gene_sd) if j in i
        ]
        for s in [
                strip_non_ki(t)
                for j, t in enumerate(self.reaction_sd)
                if j in i
        ]:
            hs = hash(json.dumps(s, sort_keys=True))
            if hs not in reacs_sd_hash:
                reacs_sd_hash.append(hs)
                reacs_sd.append(s)
            assoc.append(reacs_sd_hash.index(hs))
        return reacs_sd, assoc, gene_sd

    def get_gene_reac_sd_assoc_mark_no_ki(self, i=None):
        if not self.is_gene_sd:
            raise Exception(
                'The solutions are based on reaction interventions only.')
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

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            cls = pickle.load(f)
        return cls


def strip_non_ki(sd):
    return {k: v for k, v in sd.items() if v not in (0.0, False)}


def get_subset(sd, i):
    return [s for j, s in enumerate(sd) if j in i]


def gpr_eval(cj_terms, interv):
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
