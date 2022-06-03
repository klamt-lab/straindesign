import numpy as np
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, List, Tuple
from cobra import Model, Metabolite, Reaction, Configuration
from straindesign import StrainDesignMILP, SDModule, SDSolution, avail_solvers, select_solver, fva, parse_constraints
from straindesign.names import *
from straindesign.networktools import *
import io
import logging


class StrainDesigner(StrainDesignMILP):

    def __init__(self, model: Model, sd_modules: List[SDModule], *args,
                 **kwargs):
        allowed_keys = {
            SOLVER, MAX_COST, 'M', 'compress', KOCOST, KICOST, GKOCOST, GKICOST,
            REGCOST
        }
        # set all keys that are not in kwargs to None
        for key in allowed_keys:
            if key not in dict(kwargs).keys() and key not in {GKOCOST, GKICOST}:
                kwargs[key] = None
        # check all keys passed in kwargs
        for key, value in dict(kwargs).items():
            if key in allowed_keys:
                setattr(self, key, value)
            else:
                raise Exception("Key " + key + " is not supported.")
            if key == KOCOST:
                self.uncmp_ko_cost = value
            if key == KICOST:
                self.uncmp_ki_cost = value
            if key == GKOCOST:
                self.uncmp_gko_cost = value
            if key == GKICOST:
                self.uncmp_gki_cost = value
            if key == REGCOST:
                self.uncmp_reg_cost = value
        if (GKOCOST in kwargs or GKICOST in kwargs) and hasattr(
                model, 'genes') and model.genes:
            self.gene_sd = True
            used_g_ids = set(self.uncmp_gko_cost if hasattr(
                self, 'uncmp_gko_cost') and self.uncmp_gko_cost else set())
            used_g_ids.update(
                set(self.uncmp_gki_cost if hasattr(self, 'uncmp_gki_cost') and
                    self.uncmp_gki_cost else set()))
            used_g_ids.update(
                set(self.uncmp_reg_cost if hasattr(self, 'uncmp_reg_cost') and
                    self.uncmp_reg_cost else set()))
            if np.any([len(g.name) for g in model.genes]) and (np.any(
                [g.name in used_g_ids for g in model.genes]) or not used_g_ids):
                self.has_gene_names = True
            else:
                self.has_gene_names = False
            if GKOCOST not in kwargs or not kwargs[GKOCOST]:
                if self.has_gene_names:  # if gene names are defined, use them instead of ids
                    self.uncmp_gko_cost = {
                        k: 1.0 for k in model.genes.list_attr('name')
                    }
                else:
                    self.uncmp_gko_cost = {
                        k: 1.0 for k in model.genes.list_attr('id')
                    }
            if GKICOST not in kwargs or not kwargs[GKICOST]:
                self.uncmp_gki_cost = {}
        else:
            self.gene_sd = False
            self.has_gene_names = False
        if not kwargs[KOCOST] and not self.gene_sd:
            self.uncmp_ko_cost = {
                k: 1.0 for k in model.reactions.list_attr('id')
            }
        elif not kwargs[KOCOST]:
            self.uncmp_ko_cost = {}
        if not kwargs[KICOST]:
            self.uncmp_ki_cost = {}
        if not kwargs[REGCOST]:
            self.uncmp_reg_cost = {}
        # put module in list if only one module was provided
        if "SDModule" in str(type(sd_modules)):
            sd_modules = [sd_modules]
        self.orig_sd_modules = sd_modules
        # check that at most one bilevel module is provided
        bilvl_modules = [i for i,m in enumerate(sd_modules) \
                    if m[MODULE_TYPE] in [OPTKNOCK,ROBUSTKNOCK,OPTCOUPLE]]
        if len(bilvl_modules) > 1:
            raise Exception("Only one of the module types 'OptKnock', 'RobustKnock' and 'OptCouple' can be defined per "\
                                "strain design setup.")
        with redirect_stdout(io.StringIO()), redirect_stderr(
                io.StringIO()):  # suppress standard output from copying model
            self.orig_model = model.copy()
            self.uncmp_model = model.copy()
        self.orig_ko_cost = self.uncmp_ko_cost
        self.orig_ki_cost = self.uncmp_ki_cost
        self.orig_reg_cost = self.uncmp_reg_cost
        self.compress = kwargs['compress']
        self.M = kwargs['M']
        if self.gene_sd:
            self.orig_gko_cost = self.uncmp_gko_cost
            self.orig_gki_cost = self.uncmp_gki_cost
            # ensure that gene and reaction kos/kis do not overlap
            g_itv = {
                g for g in list(self.uncmp_gko_cost.keys()) +
                list(self.uncmp_gki_cost.keys())
            }
            r_itv = {
                r for r in list(self.uncmp_ko_cost.keys()) +
                list(self.uncmp_ki_cost.keys())
            }
            if np.any([np.any([True for g in self.uncmp_model.reactions.get_by_id(r).genes if g in g_itv]) for r in r_itv]) or \
                np.any(set(self.uncmp_gko_cost.keys()).intersection(set(self.uncmp_gki_cost.keys()))) or \
                np.any(set(self.uncmp_ko_cost.keys()).intersection(set(self.uncmp_ki_cost.keys()))):
                raise Exception('Specified gene and reaction knock-out/-in costs contain overlap. '\
                                'Make sure that metabolic interventions are enabled either through reaction or '\
                                'through gene interventions and are defined either as knock-ins or as knock-outs.')
        # 1) Preprocess Model
        logging.info('Preparing strain design computation.')
        self.solver = select_solver(self.solver, self.uncmp_model)
        kwargs[SOLVER] = self.solver
        logging.info('  Using ' + self.solver +
                     ' for solving LPs during preprocessing.')
        with redirect_stdout(io.StringIO()), redirect_stderr(
                io.StringIO()):  # suppress standard output from copying model
            cmp_model = self.uncmp_model.copy()
        # remove external metabolites
        remove_ext_mets(cmp_model)
        # replace model bounds with +/- inf if above a certain threshold
        remove_dummy_bounds(model)
        # FVAs to identify blocked, irreversible and essential reactions, as well as non-bounding bounds
        logging.info(
            '  FVA to identify blocked reactions and irreversibilities.')
        bound_blocked_or_irrevers_fva(model, solver=kwargs[SOLVER])
        logging.info('  FVA(s) to identify essential reactions.')
        essential_reacs = set()
        for m in sd_modules:
            if m[MODULE_TYPE] != SUPPRESS:  # Essential reactions can only be determined from desired
                # or opt-/robustknock modules
                flux_limits = fva(cmp_model,
                                  solver=kwargs[SOLVER],
                                  constraints=m[CONSTRAINTS])
                for (reac_id, limits) in flux_limits.iterrows():
                    if np.min(abs(limits)) > 1e-10 and np.prod(
                            np.sign(limits)) > 0:  # find essential
                        essential_reacs.add(reac_id)
        # remove ko-costs (and thus knockability) of essential reactions
        [
            self.uncmp_ko_cost.pop(er)
            for er in essential_reacs
            if er in self.uncmp_ko_cost
        ]
        # If computation of gene-gased intervention strategies, (optionally) compress gpr are rules and extend stoichimetric network with genes
        if self.gene_sd:
            if kwargs['compress'] is True or kwargs['compress'] is None:
                num_genes = len(cmp_model.genes)
                num_gpr = len(
                    [True for r in model.reactions if r.gene_reaction_rule])
                logging.info('Preprocessing GPR rules (' + str(num_genes) +
                             ' genes, ' + str(num_gpr) + ' gpr rules).')
                # removing irrelevant genes will also remove essential reactions from the list of knockable genes
                self.uncmp_gko_cost = remove_irrelevant_genes(
                    cmp_model, essential_reacs, self.uncmp_gki_cost,
                    self.uncmp_gko_cost)
                if len(cmp_model.genes) < num_genes or len([
                        True for r in model.reactions if r.gene_reaction_rule
                ]) < num_gpr:
                    num_genes = len(cmp_model.genes)
                    num_gpr = len([
                        True for r in cmp_model.reactions
                        if r.gene_reaction_rule
                    ])
                    logging.info('  Simplifyied to '+str(num_genes)+' genes and '+\
                        str(num_gpr)+' gpr rules.')
            logging.info('  Extending metabolic network with gpr associations.')
            reac_map = extend_model_gpr(cmp_model, self.uncmp_gko_cost,
                                        self.uncmp_gki_cost)
            for i, m in enumerate(sd_modules):
                for p in [
                        CONSTRAINTS, INNER_OBJECTIVE, OUTER_OBJECTIVE, PROD_ID
                ]:
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
            self.uncmp_ko_cost.update(self.uncmp_gko_cost)
            self.uncmp_ki_cost.update(self.uncmp_gki_cost)
        self.uncmp_ko_cost = extend_model_regulatory(cmp_model,
                                                     self.uncmp_reg_cost,
                                                     self.uncmp_ko_cost)
        self.cmp_ko_cost = self.uncmp_ko_cost
        self.cmp_ki_cost = self.uncmp_ki_cost
        # Compress model
        if kwargs['compress'] is True or kwargs[
                'compress'] is None:  # If compression is activated (or not defined)
            logging.info('Compressing Network (' +
                         str(len(cmp_model.reactions)) + ' reactions).')
            # compress network by lumping sequential and parallel reactions alternatingly.
            # Exclude reactions named in strain design modules from parallel compression
            no_par_compress_reacs = set()
            for m in sd_modules:
                for p in [
                        CONSTRAINTS, INNER_OBJECTIVE, OUTER_OBJECTIVE, PROD_ID
                ]:
                    if p in m and m[p] is not None:
                        param = m[p]
                        if p == CONSTRAINTS:
                            for c in param:
                                for k in c[0].keys():
                                    no_par_compress_reacs.add(k)
                        if p in [INNER_OBJECTIVE, OUTER_OBJECTIVE, PROD_ID]:
                            for k in param.keys():
                                no_par_compress_reacs.add(k)
            self.cmp_mapReac = compress_model(cmp_model, no_par_compress_reacs)
            # compress information in strain design modules
            sd_modules = compress_modules(sd_modules, self.cmp_mapReac)
            # compress ko_cost and ki_cost
            self.cmp_ko_cost, self.cmp_ki_cost, self.cmp_mapReac = compress_ki_ko_cost(
                self.cmp_ko_cost, self.cmp_ki_cost, self.cmp_mapReac)
        else:
            self.cmp_mapReac = []

        # An FVA to identify essentials before building and launching MILP (not sure if this has an effect)
        logging.info(
            '  FVA(s) in compressed model to identify essential reactions.')
        essential_reacs = set()
        for m in sd_modules:
            if m[MODULE_TYPE] != SUPPRESS:  # Essential reactions can only be determined from desired
                # or opt-/robustknock modules
                flux_limits = fva(cmp_model,
                                  solver=kwargs[SOLVER],
                                  constraints=m[CONSTRAINTS])
                for (reac_id, limits) in flux_limits.iterrows():
                    if np.min(abs(limits)) > 1e-10 and np.prod(
                            np.sign(limits)) > 0:  # find essential
                        essential_reacs.add(reac_id)
        # remove ko-costs (and thus knockability) of essential reactions
        [
            self.cmp_ko_cost.pop(er)
            for er in essential_reacs
            if er in self.cmp_ko_cost
        ]
        essential_kis = set(self.cmp_ki_cost[er]
                            for er in essential_reacs
                            if er in self.cmp_ki_cost)
        # Build MILP
        kwargs1 = kwargs
        kwargs1[KOCOST] = self.cmp_ko_cost
        kwargs1[KICOST] = self.cmp_ki_cost
        kwargs1['essential_kis'] = essential_kis
        kwargs1.pop('compress')
        if GKOCOST in kwargs1:
            kwargs1.pop(GKOCOST)
        if GKICOST in kwargs1:
            kwargs1.pop(GKICOST)
        if REGCOST in kwargs1:
            kwargs1.pop(REGCOST)
        logging.info("Finished preprocessing:")
        logging.info("  Model size: " + str(len(cmp_model.reactions)) +
                     " reactions, " + str(len(cmp_model.metabolites)) +
                     " metabolites")
        logging.info("  " + str(
            len(self.cmp_ko_cost) + len(self.cmp_ki_cost) -
            len(essential_kis)) + " targetable reactions")
        super().__init__(cmp_model, sd_modules, *args, **kwargs1)

    def postprocess_reg_sd(self, sd):
        # mark regulatory interventions with true or false
        for s in sd:
            for k, v in self.uncmp_reg_cost.items():
                if k in s:
                    s.pop(k)
                    s.update({v['str']: True})
                else:
                    s.update({v['str']: False})
        return sd

    # function wrappers for compute, compute_optimal and enumerate
    def enumerate(self, *args, **kwargs):
        cmp_sd_solution = super().enumerate(*args, **kwargs)
        if cmp_sd_solution.status in [OPTIMAL, TIME_LIMIT_W_SOL]:
            sd = expand_sd(cmp_sd_solution.get_reaction_sd_mark_no_ki(),
                           self.cmp_mapReac)
            sd = filter_sd_maxcost(sd, self.max_cost, self.uncmp_ko_cost,
                                   self.uncmp_ki_cost)
            sd = self.postprocess_reg_sd(sd)
        else:
            sd = []
        solutions = self.build_full_sd_solution(sd, cmp_sd_solution)
        logging.info(str(len(sd)) + ' solutions found.')
        return solutions

    def compute_optimal(self, *args, **kwargs):
        cmp_sd_solution = super().compute_optimal(*args, **kwargs)
        if cmp_sd_solution.status in [OPTIMAL, TIME_LIMIT_W_SOL]:
            sd = expand_sd(cmp_sd_solution.get_reaction_sd_mark_no_ki(),
                           self.cmp_mapReac)
            sd = filter_sd_maxcost(sd, self.max_cost, self.uncmp_ko_cost,
                                   self.uncmp_ki_cost)
            sd = self.postprocess_reg_sd(sd)
        else:
            sd = []
        solutions = self.build_full_sd_solution(sd, cmp_sd_solution)
        logging.info(str(len(sd)) + ' solutions found.')
        return solutions

    def compute(self, *args, **kwargs):
        cmp_sd_solution = super().compute(*args, **kwargs)
        if cmp_sd_solution.status in [OPTIMAL, TIME_LIMIT_W_SOL]:
            sd = expand_sd(cmp_sd_solution.get_reaction_sd_mark_no_ki(),
                           self.cmp_mapReac)
            sd = filter_sd_maxcost(sd, self.max_cost, self.uncmp_ko_cost,
                                   self.uncmp_ki_cost)
            sd = self.postprocess_reg_sd(sd)
        else:
            sd = []
        solutions = self.build_full_sd_solution(sd, cmp_sd_solution)
        logging.info(str(len(sd)) + ' solutions found.')
        return solutions

    def build_full_sd_solution(self, sd, sd_solution_cmp):
        sd_setup = {}
        sd_setup[MODEL_ID] = sd_solution_cmp.sd_setup.pop(MODEL_ID)
        sd_setup[MAX_SOLUTIONS] = sd_solution_cmp.sd_setup.pop(MAX_SOLUTIONS)
        sd_setup[MAX_COST] = sd_solution_cmp.sd_setup.pop(MAX_COST)
        sd_setup[TIME_LIMIT] = sd_solution_cmp.sd_setup.pop(TIME_LIMIT)
        sd_setup[SOLVER] = sd_solution_cmp.sd_setup.pop(SOLVER)
        sd_setup[SOLUTION_APPROACH] = sd_solution_cmp.sd_setup.pop(
            SOLUTION_APPROACH)
        sd_setup[MODULES] = self.orig_sd_modules
        sd_setup[KOCOST] = self.orig_ko_cost
        sd_setup[KICOST] = self.orig_ki_cost
        sd_setup[REGCOST] = self.orig_reg_cost
        if self.gene_sd:
            sd_setup[GKOCOST] = self.orig_gko_cost
            sd_setup[GKICOST] = self.orig_gki_cost
        return SDSolution(self.orig_model, sd, sd_solution_cmp.status, sd_setup)
