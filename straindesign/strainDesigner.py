import numpy as np
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, List, Tuple
from cobra import Model, Metabolite, Reaction
from straindesign import StrainDesignMILP, SDModule, SDSolution, avail_solvers, select_solver, fva, parse_constraints
from straindesign.names import *
from straindesign.networktools import *
from sympy import Rational, nsimplify, parse_expr, to_dnf
import io

def remove_irrelevant_genes(model,essential_reacs,gkis,gkos):
    # 1) Remove gpr rules from blocked reactions
    blocked_reactions = [reac.id for reac in model.reactions if reac.bounds == (0, 0)]
    for rid in blocked_reactions:
        model.reactions.get_by_id(rid).gene_reaction_rule = ''
    for g in model.genes[::-1]: # iterate in reverse order to avoid mixing up the order of the list when removing genes
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
            exp = r.gene_reaction_rule.replace(' or ',' | ').replace(' and ',' & ')
            exp = to_dnf(parse_expr(exp,{g.id:False}),force=True)
            if exp == False:
                protected_genes.add(g)
    # 3. Remove essential genes, and knockouts without impact from gko_costs
    [gkos.pop(pg.id) for pg in protected_genes if pg.id in gkos]
    # 4. Add all notknockable genes to the protected list
    [protected_genes.add(g) for g in model.genes if (g.id not in gkos) and (g.name not in gkos)] # support names or ids in gkos
    # genes with kiCosts are kept
    gki_ids = [g.id for g in model.genes if (g.id in gkis) or (g.name in gkis)] # support names or ids in gkis
    protected_genes = protected_genes.difference({model.genes.get_by_id(g) for g in gki_ids})
    protected_genes_dict = {pg.id: True for pg in protected_genes}
    # 5. Simplify gpr rules and discard rules that cannot be knocked out
    for r in model.reactions:
        if r.gene_reaction_rule:
            exp = r.gene_reaction_rule.replace(' or ',' | ').replace(' and ',' & ')
            exp = to_dnf(parse_expr(exp,protected_genes_dict),simplify=True,force=True)
            if exp == True:
                model.reactions.get_by_id(r.id).gene_reaction_rule = ''
            elif exp == False:
                print('Something went wrong during gpr rule simplification.')
            else:
                model.reactions.get_by_id(r.id).gene_reaction_rule = str(exp).replace(' & ',' and ').replace(' | ',' or ')
    # 6. Remove obsolete genes and protected genes
    for g in model.genes[::-1]:
        if not g.reactions or g in protected_genes:
            model.genes.remove(g)
    return gkos

def extend_model_regulatory(model, regcost, kocost):
    for reg_name, vals in regcost.items():
        lhs = vals['lhs']
        eqsign = vals['eqsign'] 
        rhs = vals['rhs']
        cost = vals['cost']
        reg_pseudomet_name = 'met_'+reg_name
        # add pseudometabolite
        m = Metabolite(reg_pseudomet_name)
        model.add_metabolites(m)
        # add pseudometabolite to stoichiometries
        for l,w in lhs.items():
            r = model.reactions.get_by_id(l)
            r.add_metabolites({m: w})
        # add pseudoreaction that defines the bound
        s = Reaction("bnd_"+reg_name)
        model.add_reaction(s)
        s.reaction = reg_pseudomet_name + ' --> '
        if eqsign == '=':
            s._lower_bound = -np.inf
            s._upper_bound = rhs
            s._lower_bound = rhs
        elif eqsign == '<=':
            s._lower_bound = -np.inf
            s._upper_bound = rhs
        elif eqsign == '>=':
            s._upper_bound = np.inf
            s._lower_bound = rhs
        # add knockable pseudoreaction and add it to the kocost list
        t = Reaction(reg_name)
        model.add_reaction(t)
        t.reaction = '--> '+reg_pseudomet_name
        t._upper_bound =  np.inf
        t._lower_bound = -np.inf
        kocost.update({reg_name:cost})
    return kocost

def preprocess_regulatory(model,reg_cost,has_gene_names):
    keywords = set(model.reactions.list_attr('id')+model.genes.list_attr('name')+model.genes.list_attr('id'))
    if '' in keywords:
        keywords.remove('')
    if has_gene_names:
        g_id_name_dict = {k:v for k,v in zip(model.genes.list_attr('id'),model.genes.list_attr('name'))}
    for k,v in reg_cost.copy().items():
    # generate name for regulatory pseudoreaction
        try:
            constr = parse_constraints(k,keywords)[0]
        except:
            raise Exception('Regulatory constraints could not be parsed. Please revise.')
        reacs_dict = constr[0]
        if has_gene_names:
            [reacs_dict.update({g_id_name_dict[l]:reacs_dict.pop(l)}) \
                for l in reacs_dict.copy().keys() if l in g_id_name_dict]
        eqsign = constr[1]
        rhs = constr[2]
        reg_name = ''
        for l,w in reacs_dict.items():
            if w<0:
                reg_name += 'n'+str(w)+'_'+l
            else:
                reg_name += 'p'+str(w)+'_'+l
            reg_name += '_'
        if eqsign == '=':
            reg_name += 'eq_'
        elif eqsign == '<=':
            reg_name += 'le_'
        elif eqsign == '>=':
            reg_name += 'ge_'
        reg_name += str(rhs)
        reg_cost.pop(k)
        reg_cost.update({reg_name:{'str' : k,'lhs' : reacs_dict, 'eqsign' : eqsign, 'rhs' : rhs, 'cost': v}})
    return reg_cost
            
def modules_coeff2rational(sd_modules):
    for i,module in enumerate(sd_modules):
        for param in [CONSTRAINTS,INNER_OBJECTIVE, OUTER_OBJECTIVE, PROD_ID]:
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
    for i,module in enumerate(sd_modules):
        for param in [CONSTRAINTS,INNER_OBJECTIVE, OUTER_OBJECTIVE, PROD_ID]:
            if param in module and module[param] is not None:
                if param == CONSTRAINTS:
                    for constr in module[CONSTRAINTS]:
                        for reac in constr[0].keys():
                            constr[0][reac] = float(constr[0][reac])
                if param in [INNER_OBJECTIVE, OUTER_OBJECTIVE, PROD_ID]:
                    for reac in module[param].keys():
                        module[param][reac] = float(module[param][reac])
    return sd_modules

class StrainDesigner(StrainDesignMILP):
    def __init__(self, model: Model, sd_modules: List[SDModule], *args, **kwargs):
        allowed_keys = {SOLVER, MAX_COST, 'M', 'compress',KOCOST,KICOST,GKOCOST,GKICOST,REGCOST}
        # set all keys that are not in kwargs to None
        for key in allowed_keys:
            if key not in dict(kwargs).keys() and key not in {GKOCOST,GKICOST}:
                kwargs[key] = None
        # check all keys passed in kwargs
        for key, value in dict(kwargs).items():
            if key in allowed_keys:
                setattr(self,key,value)
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
        if (GKOCOST in kwargs or GKICOST in kwargs) and hasattr(model,'genes') and model.genes:
            self.gene_sd = True
            if np.any([len(g.name) for g in model.genes]):
                self.has_gene_names = True
            else:
                self.has_gene_names = False
            if GKOCOST not in kwargs or not kwargs[GKOCOST]:
                if self.has_gene_names: # if gene names are defined, use them instead of ids
                    self.uncmp_gko_cost = {k:1.0 for k in model.genes.list_attr('name')}
                else:
                    self.uncmp_gko_cost = {k:1.0 for k in model.genes.list_attr('id')}
            if GKICOST not in kwargs or not kwargs[GKICOST]:
                self.uncmp_gki_cost = {}
        else:
            self.gene_sd = False
            self.has_gene_names = False
        if not kwargs[KOCOST] and not self.gene_sd:
            self.uncmp_ko_cost = {k:1.0 for k in model.reactions.list_attr('id')}
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
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()): # suppress standard output from copying model
            self.orig_model = model.copy()
        self.orig_ko_cost   = self.uncmp_ko_cost
        self.orig_ki_cost   = self.uncmp_ki_cost
        self.orig_reg_cost  = self.uncmp_reg_cost
        self.compress        = kwargs['compress']
        self.M               = kwargs['M']
        self.uncmp_reg_cost = preprocess_regulatory(model,self.uncmp_reg_cost,self.has_gene_names)
        if self.gene_sd:
            self.orig_gko_cost   = self.uncmp_gko_cost
            self.orig_gki_cost   = self.uncmp_gki_cost
            # ensure that gene and reaction kos/kis do not overlap
            g_itv = {g for g in list(self.uncmp_gko_cost.keys())+list(self.uncmp_gki_cost.keys())}
            r_itv = {r for r in list(self.uncmp_ko_cost.keys())+list(self.uncmp_ki_cost.keys())}
            if np.any([np.any([True for g in model.reactions.get_by_id(r).genes if g in g_itv]) for r in r_itv]) or \
                np.any(set(self.uncmp_gko_cost.keys()).intersection(set(self.uncmp_gki_cost.keys()))) or \
                np.any(set(self.uncmp_ko_cost.keys()).intersection(set(self.uncmp_ki_cost.keys()))):
                raise Exception('Specified gene and reaction knock-out/-in costs contain overlap. '\
                                'Make sure that metabolic interventions are enabled either through reaction or '\
                                'through gene interventions and are defined either as knock-ins or as knock-outs.')
        # 1) Preprocess Model
        print('Preparing strain design computation.')
        self.solver = select_solver(self.solver,model)
        kwargs[SOLVER] = self.solver
        print('  Using '+self.solver+' for solving LPs during preprocessing.')
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()): # suppress standard output from copying model
            cmp_model = model.copy()
        # remove external metabolites
        remove_ext_mets(cmp_model)
        # replace model bounds with +/- inf if above a certain threshold
        bound_thres = 1000
        if any([any([abs(b)>=bound_thres for b in r.bounds]) for r in cmp_model.reactions]):
            print('  Removing reaction bounds when larger than the threshold of '+str(bound_thres)+'.')
            for i in range(len(cmp_model.reactions)):
                if cmp_model.reactions[i].lower_bound <= -bound_thres:
                    cmp_model.reactions[i].lower_bound = -np.inf
                if cmp_model.reactions[i].upper_bound >=  bound_thres:
                    cmp_model.reactions[i].upper_bound =  np.inf
        # FVAs to identify blocked, irreversible and essential reactions, as well as non-bounding bounds
        print('  FVA to identify blocked reactions and irreversibilities.')
        flux_limits = fva(cmp_model,solver=kwargs[SOLVER])
        if kwargs[SOLVER] in ['scip','glpk']:
            tol = 1e-10 # use tolerance for tightening problem bounds
        else:
            tol = 0.0
        for (reac_id, limits) in flux_limits.iterrows():
            r = cmp_model.reactions.get_by_id(reac_id)
            # modify _lower_bound and _upper_bound to make changes permanent
            if r.lower_bound < 0.0 and limits.minimum - tol > r.lower_bound:
                r._lower_bound = -np.inf
            if limits.minimum >= tol:
                r._lower_bound = np.max([0.0,r._lower_bound])
            if r.upper_bound > 0.0 and limits.maximum + tol < r.upper_bound:
                r._upper_bound = np.inf
            if limits.maximum <= -tol:
                r._upper_bound = np.min([0.0,r._upper_bound])
        print('  FVA(s) to identify essential reactions.')
        essential_reacs = set()
        for m in sd_modules:
            if m[MODULE_TYPE] != SUPPRESS:  # Essential reactions can only be determined from desired
                                            # or opt-/robustknock modules
                flux_limits = fva(cmp_model,solver=kwargs[SOLVER],constraints=m[CONSTRAINTS])
                for (reac_id, limits) in flux_limits.iterrows():
                    if np.min(abs(limits)) > tol and np.prod(np.sign(limits)) > 0: # find essential
                        essential_reacs.add(reac_id)
        # remove ko-costs (and thus knockability) of essential reactions
        [self.uncmp_ko_cost.pop(er) for er in essential_reacs if er in self.uncmp_ko_cost]
        # If computation of gene-gased intervention strategies, (optionally) compress gpr are rules and extend stoichimetric network with genes
        if self.gene_sd:
            if kwargs['compress'] is True or kwargs['compress'] is None:
                num_genes = len(cmp_model.genes)
                num_gpr   = len([True for r in model.reactions if r.gene_reaction_rule])
                print('Preprocessing GPR rules ('+str(num_genes)+' genes, '+str(num_gpr)+' gpr rules).')
                # removing irrelevant genes will also remove essential reactions from the list of knockable genes
                self.uncmp_gko_cost = remove_irrelevant_genes(cmp_model, essential_reacs, self.uncmp_gki_cost, self.uncmp_gko_cost)
                if len(cmp_model.genes) < num_genes or len([True for r in model.reactions if r.gene_reaction_rule]) < num_gpr:
                    num_genes = len(cmp_model.genes)
                    num_gpr   = len([True for r in cmp_model.reactions if r.gene_reaction_rule])
                    print('  Simplifyied to '+str(num_genes)+' genes and '+\
                        str(num_gpr)+' gpr rules.')
            print('  Extending metabolic network with gpr associations.')
            reac_map = extend_model_gpr(cmp_model,self.uncmp_gko_cost, self.uncmp_gki_cost)
            for i,m in enumerate(sd_modules):
                for p in [CONSTRAINTS, INNER_OBJECTIVE, OUTER_OBJECTIVE, PROD_ID]:
                    if p in m and m[p] is not None:
                        if p == CONSTRAINTS:
                            for c in m[p]:
                                for k in list(c[0].keys()):
                                    v = c[0].pop(k)
                                    for n,w in reac_map[k].items():
                                        c[0][n] = v*w
                        if p in [INNER_OBJECTIVE, OUTER_OBJECTIVE, PROD_ID]:
                            for k in list(m[p].keys()):
                                v = m[p].pop(k)
                                for n,w in reac_map[k].items():
                                    m[p][n] = v*w
            self.uncmp_ko_cost.update(self.uncmp_gko_cost)
            self.uncmp_ki_cost.update(self.uncmp_gki_cost)
        self.uncmp_ko_cost = extend_model_regulatory(cmp_model,self.uncmp_reg_cost,self.uncmp_ko_cost)
        self.cmp_ko_cost = self.uncmp_ko_cost
        self.cmp_ki_cost = self.uncmp_ki_cost
        # Compress model
        self.cmp_mapReac = []
        if  kwargs['compress'] is True or kwargs['compress'] is None: # If compression is activated (or not defined)
            print('Compressing Network ('+str(len(cmp_model.reactions))+' reactions).')
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
            # Remove conservation relations.
            print('  Removing blocked reactions.')
            blocked_reactions = remove_blocked_reactions(cmp_model)
            # remove blocked reactions from ko- and ki-costs
            [self.cmp_ko_cost.pop(br.id) for br in blocked_reactions if br.id in self.cmp_ko_cost]
            [self.cmp_ki_cost.pop(br.id) for br in blocked_reactions if br.id in self.cmp_ki_cost]
            print('  Translating stoichiometric coefficients to rationals.')
            stoichmat_coeff2rational(cmp_model)
            sd_modules = modules_coeff2rational(sd_modules)
            print('  Removing conservation relations.')
            remove_conservation_relations(cmp_model)
            odd = True
            run = 1
            while True:
                # np.savetxt('Table.csv',create_stoichiometric_matrix(cmp_model),'%i',',')
                if odd:
                    print('  Compression '+str(run)+': Applying compression from EFM-tool module.')
                    subT, reac_map_exp = compress_model(cmp_model)
                    for new_reac, old_reac_val in reac_map_exp.items():
                        old_reacs_no_compress = [r for r in no_par_compress_reacs if r in old_reac_val]
                        if old_reacs_no_compress:
                            [no_par_compress_reacs.remove(r) for r in old_reacs_no_compress]
                            no_par_compress_reacs.add(new_reac)                            
                else:
                    print('  Compression '+str(run)+': Lumping parallel reactions.')
                    subT, reac_map_exp = compress_model_parallel(cmp_model, no_par_compress_reacs)
                remove_conservation_relations(cmp_model)
                if subT.shape[0] > subT.shape[1]:
                    print('  Reduced to '+str(subT.shape[1])+' reactions.')
                    # store the information for decompression in a Tuple
                    # (0) compression matrix, (1) reac_id dictornary {cmp_rid: {orig_rid1: factor1, orig_rid2: factor2}}, 
                    # (2) linear (True) or parallel (False) compression (3,4) ko and ki costs of expanded network
                    self.cmp_mapReac += [{"reac_map_exp":reac_map_exp,
                                          "odd":odd,
                                          KOCOST:self.cmp_ko_cost,
                                          KICOST:self.cmp_ki_cost}]
                    # compress information in strain design modules
                    if odd:
                        for new_reac, old_reac_val in reac_map_exp.items():
                            for i,m in enumerate(sd_modules):
                                for p in [CONSTRAINTS, INNER_OBJECTIVE, OUTER_OBJECTIVE, PROD_ID,MIN_GCP]:
                                    if p in m and m[p] is not None:
                                        param = m[p]
                                        if p == CONSTRAINTS:
                                            for j,c in enumerate(m[p]):
                                                if np.any([k in old_reac_val for k in c[0].keys()]):
                                                    lumped_reacs = [k for k in c[0].keys() if k in old_reac_val]
                                                    c[0][new_reac] = np.sum([c[0].pop(k)*old_reac_val[k] for k in lumped_reacs])
                                        if p in [INNER_OBJECTIVE, OUTER_OBJECTIVE, PROD_ID]:
                                            if np.any([k in old_reac_val for k in param.keys()]):
                                                lumped_reacs = [k for k in param.keys() if k in old_reac_val]
                                                m[p][new_reac] = np.sum([param.pop(k)*old_reac_val[k] for k in lumped_reacs if k in old_reac_val])
                                        # if p == MIN_GCP:
                                        #     m[p] = m[p]
                                        #     print('lol')
                    # compress ko_cost and ki_cost
                    # ko_cost of lumped reactions: when reacs sequential: lowest of ko costs, when parallel: sum of ko costs
                    # ki_cost of lumped reactions: when reacs sequential: sum of ki costs, when parallel: lowest of ki costs
                    if self.cmp_ko_cost:
                        ko_cost_new = {}
                        for r in cmp_model.reactions.list_attr('id'):
                            if np.any([s in self.cmp_ko_cost for s in reac_map_exp[r]]):
                                if odd and not np.any([s in self.cmp_ki_cost for s in reac_map_exp[r]]):
                                    ko_cost_new[r] = np.min([self.cmp_ko_cost[s] for s in reac_map_exp[r] if s in self.cmp_ko_cost])
                                elif not odd:
                                    ko_cost_new[r] = np.sum([self.cmp_ko_cost[s] for s in reac_map_exp[r] if s in self.cmp_ko_cost])
                        self.cmp_ko_cost = ko_cost_new
                    if self.cmp_ki_cost:
                        ki_cost_new = {}
                        for r in cmp_model.reactions.list_attr('id'):
                            if np.any([s in self.cmp_ki_cost for s in reac_map_exp[r]]):
                                if odd:
                                    ki_cost_new[r] = np.sum([self.cmp_ki_cost[s] for s in reac_map_exp[r] if s in self.cmp_ki_cost])
                                elif not odd and not np.any([s in self.cmp_ko_cost for s in reac_map_exp[r]]):
                                    ki_cost_new[r] = np.min([self.cmp_ki_cost[s] for s in reac_map_exp[r] if s in self.cmp_ki_cost])
                        self.cmp_ki_cost = ki_cost_new
                    if odd:
                        odd = False
                    else:
                        odd = True
                    run += 1
                else:
                    print('  Last step could not reduce size further ('+str(subT.shape[0])+' reactions).')
                    print('  Network compression completed. ('+str(run-1)+' compression iterations)')
                    print('  Translating stoichiometric coefficients back to float.')
                    stoichmat_coeff2float(cmp_model)
                    sd_modules = modules_coeff2float(sd_modules)
                    break

        # An FVA to identify essentials before building and launching MILP (not sure if this has an effect)
        print('  FVA(s) in compressed model to identify essential reactions.')
        essential_reacs = set()
        for m in sd_modules:
            if m[MODULE_TYPE] != SUPPRESS:  # Essential reactions can only be determined from desired
                                            # or opt-/robustknock modules
                flux_limits = fva(cmp_model,solver=kwargs[SOLVER],constraints=m[CONSTRAINTS])
                for (reac_id, limits) in flux_limits.iterrows():
                    if np.min(abs(limits)) > tol and np.prod(np.sign(limits)) > 0: # find essential
                        essential_reacs.add(reac_id)
        # remove ko-costs (and thus knockability) of essential reactions
        [self.cmp_ko_cost.pop(er) for er in essential_reacs if er in self.cmp_ko_cost]
        essential_kis = set(self.cmp_ki_cost[er] for er in essential_reacs if er in self.cmp_ki_cost)
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
        print("Finished preprocessing:")
        print("  Model size: "+str(len(cmp_model.reactions))+" reactions, "+str(len(cmp_model.metabolites))+" metabolites")
        print("  "+str(len(self.cmp_ko_cost)+len(self.cmp_ki_cost)-len(essential_kis))+" targetable reactions")
        super().__init__(cmp_model,sd_modules, *args, **kwargs1)

    def expand_sd(self,sd):
        # expand mcs by applying the compression steps in the reverse order
        cmp_map = self.cmp_mapReac[::-1]
        for exp in cmp_map:
            reac_map_exp = exp["reac_map_exp"] # expansion map
            ko_cost      = exp[KOCOST]
            ki_cost      = exp[KICOST]
            par_reac_cmp = not exp["odd"] # if parallel or sequential reactions were lumped
            for r_cmp,r_orig in reac_map_exp.items():
                if len(r_orig) > 1:
                    for m in sd.copy():
                        if r_cmp in m:
                            val = m[r_cmp]
                            del m[r_cmp]
                            if val < 0: # case: KO
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
                            elif val > 0: # case: KI
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
                            elif val == 0: # case: KI that was not introduced
                                new_m = m.copy() # assume that none of the expanded
                                for d in r_orig: # reactions are inserted, neither
                                    if d in ki_cost: # parallel, nor sequential
                                        new_m[d] = val
                                sd += [new_m]
                            sd.remove(m)
        # eliminate mcs that are too expensive
        if self.max_cost:
            costs = [np.sum([self.uncmp_ko_cost[k] if v<0 else self.uncmp_ki_cost[k] for k,v in m.items()]) for m in sd]
            sd = [sd[i] for i in range(len(sd)) if costs[i] <= self.max_cost+1e-8]
        # mark regulatory interventions with true or false
        for s in sd:
            for k,v in self.uncmp_reg_cost.items():
                if k in s:
                    s.pop(k)
                    s.update({v['str']:True})
                else:
                    s.update({v['str']:False})
        return sd

    # function wrappers for compute, compute_optimal and enumerate
    def enumerate(self, *args, **kwargs):
        cmp_sd_solution = super().enumerate(*args, **kwargs)
        if cmp_sd_solution.status in [OPTIMAL,TIME_LIMIT_W_SOL]:
            sd = self.expand_sd(cmp_sd_solution.get_reaction_sd_mark_no_ki())
        else:
            sd = []
        solutions = self.build_full_sd_solution(sd, cmp_sd_solution)
        print(str(len(sd)) +' solutions found.')
        return solutions
    
    def compute_optimal(self, *args, **kwargs):
        cmp_sd_solution = super().compute_optimal(*args, **kwargs)
        if cmp_sd_solution.status in [OPTIMAL,TIME_LIMIT_W_SOL]:
            sd = self.expand_sd(cmp_sd_solution.get_reaction_sd_mark_no_ki())
        else:
            sd = []
        solutions = self.build_full_sd_solution(sd, cmp_sd_solution)
        print(str(len(sd)) +' solutions found.')
        return solutions
    
    def compute(self, *args, **kwargs):
        cmp_sd_solution = super().compute(*args, **kwargs)
        if cmp_sd_solution.status in [OPTIMAL,TIME_LIMIT_W_SOL]:
            sd = self.expand_sd(cmp_sd_solution.get_reaction_sd_mark_no_ki())
        else:
            sd = []
        solutions = self.build_full_sd_solution(sd, cmp_sd_solution)
        print(str(len(sd)) +' solutions found.')
        return solutions
    
    def build_full_sd_solution(self, sd, sd_solution_cmp):
        sd_setup = {}
        sd_setup[MODEL_ID]          = sd_solution_cmp.sd_setup.pop(MODEL_ID)
        sd_setup[MAX_SOLUTIONS]     = sd_solution_cmp.sd_setup.pop(MAX_SOLUTIONS)
        sd_setup[MAX_COST]          = sd_solution_cmp.sd_setup.pop(MAX_COST)
        sd_setup[TIME_LIMIT]        = sd_solution_cmp.sd_setup.pop(TIME_LIMIT)
        sd_setup[SOLVER]            = sd_solution_cmp.sd_setup.pop(SOLVER)
        sd_setup[SOLUTION_APPROACH] = sd_solution_cmp.sd_setup.pop(SOLUTION_APPROACH)
        sd_setup[MODULES] = self.orig_sd_modules
        sd_setup[KOCOST]  = self.orig_ko_cost
        sd_setup[KICOST]  = self.orig_ki_cost
        sd_setup[REGCOST] = self.orig_reg_cost
        if self.gene_sd:
            sd_setup[GKOCOST] = self.orig_gko_cost
            sd_setup[GKICOST] = self.orig_gki_cost
        return SDSolution(self.orig_model,sd,sd_solution_cmp.status,sd_setup)