from ipaddress import v4_int_to_packed, v6_int_to_packed
from re import sub
import numpy as np
from scipy import sparse
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, List, Tuple
from cobra import Model, Metabolite, Reaction
from cobra.util.array import create_stoichiometric_matrix
from mcs import StrainDesignMILP, StrainDesignMILPBuilder, MILP_LP, SD_Module, get_rids
from mcs.fva import *
from warnings import warn, catch_warnings
import jpype
from sympy import Rational, nsimplify, parse_expr, to_dnf
from sympy.core.numbers import One

import efmtool_link.efmtool4cobra as efm
import efmtool_link.efmtool_intern as efmi
from zmq import EVENT_CLOSE_FAILED
import java.util.HashSet

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
    for r in essential_reacs:
        gpr = model.reactions.get_by_id(r).gpr
        [protected_genes.add(model.genes.get_by_id(g)) for g in gpr.genes if not gpr.eval(g)]
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

def extend_model_gpr(model,gkos,gkis):
    protein_pool_pseudomets = {r.id: 'protpool_'+r.id for r in model.reactions if r.gpr.body}
    gene_pseudomets = {g.id: 'gene_'+g.id for g in model.genes}
    # All reaction rules are provided in dnf. Make dict of dicts to look up
    # (1) how many disjuct terms there are (2) how the conjuncted terms look like inside
    gpr_associations = {}
    for r in model.reactions:
        if r.gpr.body: # if reaction has a gpr rule
            for i,p in enumerate(r.gene_reaction_rule.split('or')):
                conj_genes = set()
                for g in p.replace('(','').replace(')','').split('and'):
                    conj_genes.add(gene_pseudomets[g.strip()])
                gpr_associations.update({r.id+'_gpr_'+str(i) : ' + '.join(conj_genes)+' --> '+protein_pool_pseudomets[r.id]})
    # Find reactions that need to be split
    reac_map = {}
    rev_reac = set()
    del_reac = set()
    for r in model.reactions:
        reac_map.update({r.id:{}})
        if not r.gpr.body:
            reac_map[r.id].update({r.id: 1.0})
            continue
        if r.gpr.body and r.bounds[0] < 0:
            r_rev = (r*-1)
            if r.gpr.body and r.bounds[1] > 0:
                r_rev.id = r.id+'_reverse_'+hex(hash(r))[8:]
            r_rev.lower_bound = np.max([0,r_rev.lower_bound])
            reac_map[r.id].update({r_rev.id: -1.0})
            rev_reac.add(r_rev)
        if r.gpr.body and r.bounds[1] > 0:
            reac_map[r.id].update({r.id: 1.0})
            r._lower_bound = np.max([0,r._lower_bound])
        else:
            del_reac.add(r)
    model.remove_reactions(del_reac)
    model.add_reactions(rev_reac)
    # add all pseudo metabolites
    [model.add_metabolites(Metabolite(m)) for m in gene_pseudomets.values()]
    [model.add_metabolites(Metabolite(m)) for m in protein_pool_pseudomets.values()]
    # add gene reactions and use gene names instead of ids if available
    gene_ids = {g.id for g in model.genes}
    gene_names = set(g.name for g in model.genes)
    gene_names_exist = np.all([len(g.name) for g in model.genes])
    use_name_not_id = gene_names_exist and (gene_names.intersection(gkos) or gene_names.intersection(gkis))
    for g in model.genes:
        r = Reaction(g.id)
        if use_name_not_id: # if gene name is available and used in gki_cost and gko_cost
            r.id = g.name
        model.add_reaction(r)
        r.reaction = '--> '+gene_pseudomets[g.id]
        r._upper_bound = np.inf
    # add gpr reactions
    for gpr in gpr_associations.keys():
        r = Reaction(gpr)
        model.add_reaction(r)
        r.reaction = gpr_associations[gpr]
        r._upper_bound = np.inf
    # add pseudometabolites to forward and reverse reactions
    for r in protein_pool_pseudomets.keys():
        for s in reac_map[r]:
            reac = model.reactions.get_by_id(s)
            reac.add_metabolites({model.metabolites.get_by_id(protein_pool_pseudomets[r]): -1.0})
    return gkos, gkis, reac_map

def remove_blocked_reactions(model) -> List:
    blocked_reactions = [reac for reac in model.reactions if reac.bounds == (0, 0)]
    model.remove_reactions(blocked_reactions)
    return blocked_reactions

def remove_ext_mets(model):
    external_mets = [i for i,cpts in zip(model.metabolites,model.metabolites.list_attr("compartment")) if cpts == 'External_Species']
    model.remove_metabolites(external_mets)
    stoich_mat = create_stoichiometric_matrix(model)
    obsolete_reacs = [reac for reac,b_rempty in zip(model.reactions,np.any(stoich_mat,0)) if not b_rempty]
    model.remove_reactions(obsolete_reacs)

def remove_conservation_relations(model):
    stoich_mat = create_stoichiometric_matrix(model, array_type='lil')
    basic_metabolites = efmi.basic_columns_rat(stoich_mat.transpose().toarray(), tolerance=0)
    dependent_metabolites = [model.metabolites[i].id for i in set(range(len(model.metabolites))) - set(basic_metabolites)]
    # print("The following metabolites have been removed from the model:")
    # print(dependent_metabolites)
    for m in dependent_metabolites:
        model.metabolites.get_by_id(m).remove_from_model()
    
# replace all stoichiometric coefficients with rationals.
def stoichmat_coeff2rational(model):
    num_reac = len(model.reactions)
    for i in range(num_reac):
        for k, v in model.reactions[i]._metabolites.items():
            if type(v) is float or type(v) is int:
                if type(v) is int or v.is_integer():
                    # v = int(v)
                    # n = int2jBigInteger(v)
                    # d = BigInteger.ONE
                    v = Rational(v) # for simplicity and actually slighlty faster (?)
                else:
                    rational_conversion='base10' # This was formerly in the function parameters 
                    v = nsimplify(v, rational=True, rational_conversion=rational_conversion)
                    # v = sympy.Rational(v)
                model.reactions[i]._metabolites[k] = v # only changes coefficient in the model, not in the solver
            elif not isinstance(v,Rational):
                raise TypeError

# replace all stoichiometric coefficients with ints and floats
def stoichmat_coeff2float(model):
    num_reac = len(model.reactions)
    for i in range(num_reac):
        for k, v in model.reactions[i]._metabolites.items():
            if v.is_Float or v.is_Rational or v.is_Integer:
                model.reactions[i]._metabolites[k] = float(v)
            else:
                raise Exception('unknown data type')
            
def modules_coeff2rational(sd_modules):
    for i,m in enumerate(sd_modules):
        for p in ['constraints','inner_objective','outer_objective','prod_id']:
            if hasattr(m,p) and getattr(m,p) is not None:
                param = getattr(m,p)
                if p == 'constraints':
                    for c in param:
                        for k in c[0].keys():
                            c[0][k] = nsimplify(c[0][k])
                if p in ['inner_objective','outer_objective','prod_id']:
                    for k in param.keys():
                        param[k] = nsimplify(param[k])
    return sd_modules

def modules_coeff2float(sd_modules):
    for i,m in enumerate(sd_modules):
        for p in ['constraints','inner_objective','outer_objective','prod_id']:
            if hasattr(m,p) and getattr(m,p) is not None:
                param = getattr(m,p)
                if p == 'constraints':
                    for c in param:
                        for k in c[0].keys():
                            c[0][k] = float(c[0][k])
                if p in ['inner_objective','outer_objective','prod_id']:
                    for k in param.keys():
                        param[k] = float(param[k])
    return sd_modules

# compression function (mostly copied from efmtool)
def compress_model(model):
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
        if model.reactions[i].upper_bound <= 0: # can run in backwards direction only (is and stays classified as irreversible)
            model.reactions[i] *= -1
            flipped.append(i)
            # print("Flipped", model.reactions[i].id)
        # have to use _metabolites because metabolites gives only a copy
        for k, v in model.reactions[i]._metabolites.items():
            n, d = efm.sympyRat2jBigIntegerPair(v)
            stoich_mat.setValueAt(model.metabolites.index(k.id), i, efm.BigFraction(n, d))
    # compress
    smc = efm.StoichMatrixCompressor(efmi.subset_compression)
    reacNames = jpype.JString[:](model.reactions.list_attr('id'))
    comprec = smc.compress(stoich_mat, reversible, jpype.JString[num_met], reacNames,None)
    subset_matrix= efmi.jpypeArrayOfArrays2numpy_mat(comprec.post.getDoubleRows())
    del_rxns = np.logical_not(np.any(subset_matrix, axis=1)) # blocked reactions
    for j in range(subset_matrix.shape[1]):
        rxn_idx = subset_matrix[:, j].nonzero()[0]
        r0 = rxn_idx[0]
        model.reactions[r0].subset_rxns = []
        model.reactions[r0].subset_stoich = []
        for r in rxn_idx: # rescale all reactions in this subset
            # !! swaps lb, ub when the scaling factor is < 0, but does not change the magnitudes
            factor = efm.jBigFraction2sympyRat(comprec.post.getBigFractionValueAt(r, j))
            # factor = jBigFraction2intORsympyRat(comprec.post.getBigFractionValueAt(r, j)) # does not appear to make a speed difference
            model.reactions[r] *=  factor #subset_matrix[r, j]
            # factor = abs(float(factor)) # context manager has trouble with non-float bounds
            if model.reactions[r].lower_bound not in (0, -float('inf')):
                model.reactions[r].lower_bound/= abs(subset_matrix[r, j]) #factor
            if model.reactions[r].upper_bound not in (0, float('inf')):
                model.reactions[r].upper_bound/= abs(subset_matrix[r, j]) #factor
            model.reactions[r0].subset_rxns.append(r)
            model.reactions[r0].subset_stoich.append(factor)
        for r in rxn_idx[1:]: # merge reactions
            # rename main reaction
            if len(model.reactions[r0].id)+len(model.reactions[r].id) < 220:
                model.reactions[r0].id += '*'+model.reactions[r].id # combine names
            elif not model.reactions[r0].id[-3:] == '...':
                model.reactions[r0].id += '...'
            # !! keeps bounds of reactions[rxn_idx[0]]
            model.reactions[r0] += model.reactions[r]
            if model.reactions[r].lower_bound > model.reactions[r0].lower_bound:
                model.reactions[r0].lower_bound = model.reactions[r].lower_bound
            if model.reactions[r].upper_bound < model.reactions[r0].upper_bound:
                model.reactions[r0].upper_bound = model.reactions[r].upper_bound
            del_rxns[r] = True
    # print(time.monotonic() - start_time) # 11 seconds in iJO1366
    del_rxns = np.where(del_rxns)[0]
    for i in range(len(del_rxns)-1, -1, -1): # delete in reversed index order to keep indices valid
        model.reactions[del_rxns[i]].remove_from_model(remove_orphans=True)
    subT = np.zeros((num_reac, len(model.reactions)))
    rational_map = {}
    for j in range(subT.shape[1]):
        subT[model.reactions[j].subset_rxns, j] = [float(v) for v in model.reactions[j].subset_stoich]
        # rational_map is a dictionary that associates the new reaction with a dict of its original reactions and its scaling factors
        rational_map.update({model.reactions[j].id: {old_reac_ids[i]: v for i,v in zip(model.reactions[j].subset_rxns,model.reactions[j].subset_stoich)}})
    for i in flipped: # adapt so that it matches the reaction direction before flipping
            subT[i, :] *= -1
    # adapt compressed_model name
    return sparse.csc_matrix(subT), rational_map

def compress_model_parallel(model, protected_rxns=[]):
    # lump parallel reactions
    # 
    # - exclude lumping of reactions with inhomogenous bounds
    # - exclude protected reactions
    old_num_reac = len(model.reactions)
    old_objective = [r.objective_coefficient for r in model.reactions]
    old_reac_ids = [r.id for r in model.reactions]
    stoichmat_T = create_stoichiometric_matrix(model,'lil').transpose()
    factor = [d[0] if d else 1.0 for d in stoichmat_T.data]
    A = (sparse.diags(factor)@stoichmat_T)
    lb = [r.lower_bound for r in model.reactions]
    ub = [r.upper_bound for r in model.reactions]
    fwd = sparse.lil_matrix([1. if (np.isinf(u) and f>0 or np.isinf(l) and f<0) else 0. for f,l,u in zip(factor,lb,ub)]).transpose()
    rev = sparse.lil_matrix([1. if (np.isinf(l) and f>0 or np.isinf(u) and f<0) else 0. for f,l,u in zip(factor,lb,ub)]).transpose()
    inh = sparse.lil_matrix([i+1 if not ((np.isinf(ub[i]) or ub[i] == 0) and (np.isinf(lb[i]) or lb[i] == 0)) \
                                 else 0 for i in range(len(model.reactions))]).transpose()
    A = sparse.hstack((A,fwd,rev,inh),'csr')
    # find equivalent/parallel reactions
    subset_list = []
    prev_found = []
    protected = [True if r.id in protected_rxns else False for r in model.reactions]
    hashes = [hash((tuple(A[i].indices),tuple(A[i].data))) for i in range(A.shape[0])]
    for i in range(A.shape[0]):
        if i in prev_found: # if reaction was already found to be identical to another one, skip.
            continue
        if protected[i]: # if protected, add 1:1 relationship, skip.
            subset_list += [[i]]
            continue
        subset_i = [i]
        for j in range(i+1,A.shape[0]): # otherwise, identify parallel reactions
            if not protected[j] and j not in prev_found:
                # if np.all(A[i].indices == A[j].indices) and np.all(A[i].data == A[j].data):
                if hashes[i] == hashes[j]:
                    subset_i += [j]
                    prev_found += [j]
        if subset_i:
            subset_list += [subset_i]
    # lump parallel reactions (delete redundant)
    del_rxns = [False]*len(model.reactions)
    for rxn_idx in subset_list:
        for i in range(1, len(rxn_idx)):
            if len(model.reactions[rxn_idx[0]].id)+len(model.reactions[rxn_idx[i]].id) < 220:
                model.reactions[rxn_idx[0]].id += '*'+model.reactions[rxn_idx[i]].id # combine names
            elif not model.reactions[rxn_idx[0]].id[-3:] == '...':
                model.reactions[rxn_idx[0]].id += '...'
            del_rxns[rxn_idx[i]] = True
    del_rxns = np.where(del_rxns)[0]
    for i in range(len(del_rxns)-1, -1, -1): # delete in reversed index order to keep indices valid
        model.reactions[del_rxns[i]].remove_from_model(remove_orphans=True)
    # create compression map
    rational_map = {}
    subT = np.zeros((old_num_reac, len(model.reactions)))
    for i in range(subT.shape[1]):
        for j in subset_list[i]:
            subT[j,i] = 1
        # rational_map is a dictionary that associates the new reaction with a dict of its original reactions and its scaling factors
        rational_map.update({model.reactions[i].id: {old_reac_ids[j]: One() for j in subset_list[i]}})
        
    new_objective = old_objective@subT
    for r,c in zip(model.reactions,new_objective):
        r.objective_coefficient = c
    return sparse.csc_matrix(subT), rational_map


class StrainDesigner(StrainDesignMILP):
    def __init__(self, model: Model, sd_modules: List[SD_Module], *args, **kwargs):
        allowed_keys = {'solver', 'max_cost', 'M','threads', 'mem', 'compress','ko_cost','ki_cost','gko_cost','gki_cost'}
        # set all keys that are not in kwargs to None
        for key in allowed_keys:
            if key not in dict(kwargs).keys() and key not in {'gko_cost','gki_cost'}:
                kwargs[key] = None
        # check all keys passed in kwargs
        for key, value in dict(kwargs).items():
            if key in allowed_keys:
                setattr(self,key,value)
            else:
                raise Exception("Key " + key + " is not supported.")
            if key == 'ko_cost':
                self.uncmp_ko_cost = value
            if key == 'ki_cost':
                self.uncmp_ki_cost = value
            if key == 'gko_cost':
                self.uncmp_gko_cost = value
            if key == 'gki_cost':
                self.uncmp_gki_cost = value
        if ('gko_cost' in kwargs or 'gki_cost' in kwargs) and hasattr(model,'genes') and model.genes:
            gene_sd = True
            if 'gko_cost' not in kwargs or not kwargs['gko_cost']:
                if np.any([len(g.name) for g in model.genes]): # if gene names are defined, use them instead of ids
                    self.uncmp_gko_cost = {k:1.0 for k in model.genes.list_attr('name')}
                else:
                    self.uncmp_gko_cost = {k:1.0 for k in model.genes.list_attr('id')}
            if 'gki_cost' not in kwargs or not kwargs['gki_cost']:
                self.uncmp_gki_cost = {}
        else:
            gene_sd = False
        if not kwargs['ko_cost'] and not gene_sd:
            self.uncmp_ko_cost = {k:1.0 for k in model.reactions.list_attr('id')}
        elif not kwargs['ko_cost']:
            self.uncmp_ko_cost = {}
        if not kwargs['ki_cost']:
            self.uncmp_ki_cost = {}
        # put module in list if only one module was provided
        if "SD_Module" in str(type(sd_modules)):
            sd_modules = [sd_modules]
        # 1) Preprocess Model
        print('Preparing strain design computation.')
        print('Using '+kwargs['solver']+' for solving LPs during preprocessing.')
        with redirect_stdout(None), redirect_stderr(None): # suppress standard output from copying model
            uncmp_model = model.copy()
        # remove external metabolites
        remove_ext_mets(uncmp_model)
        # replace model bounds with +/- inf if above a certain threshold
        bound_thres = 1000
        for i in range(len(uncmp_model.reactions)):
            if uncmp_model.reactions[i].lower_bound <= -bound_thres:
                uncmp_model.reactions[i].lower_bound = -np.inf
            if uncmp_model.reactions[i].upper_bound >=  bound_thres:
                uncmp_model.reactions[i].upper_bound =  np.inf
        # FVAs to identify blocked, irreversible and essential reactions, as well as non-bounding bounds
        print('FVA to identify blocked reactions and irreversibilities.')
        flux_limits = fva(uncmp_model,solver=kwargs['solver'])
        if kwargs['solver'] in ['scip','glpk']:
            tol = 1e-10 # use tolerance for tightening problem bounds
        else:
            tol = 0.0
        for (reac_id, limits) in flux_limits.iterrows():
            r = uncmp_model.reactions.get_by_id(reac_id)
            # modify _lower_bound and _upper_bound to make changes permanent
            if r.lower_bound < 0.0 and limits.minimum - tol > r.lower_bound:
                r._lower_bound = -np.inf
            if limits.minimum >= tol:
                r._lower_bound = np.max([0.0,r._lower_bound])
            if r.upper_bound > 0.0 and limits.maximum + tol < r.upper_bound:
                r._upper_bound = np.inf
            if limits.maximum <= -tol:
                r._upper_bound = np.min([0.0,r._upper_bound])
        print('FVA(s) to identify essential reactions.')
        essential_reacs = set()
        for m in sd_modules:
            if m.module_sense != 'undesired': # Essential reactions can only be determined from desired
                                            # or opt-/robustknock modules
                flux_limits = fva(uncmp_model,solver=kwargs['solver'],constraints=m.constraints)
                for (reac_id, limits) in flux_limits.iterrows():
                    if np.min(abs(limits)) > tol and np.prod(np.sign(limits)) > 0: # find essential
                        essential_reacs.add(reac_id)
        # remove ko-costs (and thus knockability) of essential reactions
        [self.uncmp_ko_cost.pop(er) for er in essential_reacs if er in self.uncmp_ko_cost]
        # If computation of gene-gased intervention strategies, (optionally) compress gpr are rules and extend stoichimetric network with genes
        if gene_sd:
            if kwargs['compress'] is True or kwargs['compress'] is None:
                num_genes = len(uncmp_model.genes)
                num_gpr   = len([True for r in model.reactions if r.gpr.body])
                print('Preprocessing GPR rules ('+str(num_genes)+' genes, '+str(num_gpr)+' gpr rules).')
                # removing irrelevant genes will also remove essential reactions from the list of knockable genes
                self.uncmp_gko_cost = remove_irrelevant_genes(uncmp_model, essential_reacs, self.uncmp_gki_cost, self.uncmp_gko_cost)
                if len(uncmp_model.genes) < num_genes or len([True for r in model.reactions if r.gpr.body]) < num_gpr:
                    num_genes = len(uncmp_model.genes)
                    num_gpr   = len([True for r in uncmp_model.reactions if r.gpr.body])
                    print('  Simplifyied to '+str(num_genes)+' genes and '+\
                        str(num_gpr)+' gpr rules.')
            print('  Extending metabolic network with gpr associations.')
            self.uncmp_gko_cost, self.uncmp_gki_cost, reac_map = extend_model_gpr(uncmp_model,self.uncmp_gko_cost, self.uncmp_gki_cost)
            for i,m in enumerate(sd_modules):
                for p in ['constraints','inner_objective','outer_objective','prod_id']:
                    if hasattr(m,p) and getattr(m,p) is not None:
                        param = getattr(m,p)
                        if p == 'constraints':
                            for c in param:
                                for k in list(c[0].keys()):
                                    v = c[0].pop(k)
                                    for n,w in reac_map[k].items():
                                        c[0][n] = v*w
                        if p in ['inner_objective','outer_objective','prod_id']:
                            for k in list(param.keys()):
                                v = param.pop(k)
                                for n,w in reac_map[k].items():
                                    param[n] = v*w
            self.uncmp_ko_cost.update(self.uncmp_gko_cost)
            self.uncmp_ki_cost.update(self.uncmp_gki_cost)
        with redirect_stdout(None), redirect_stderr(None): # suppress standard output from copying model
            cmp_model = uncmp_model.copy()
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
                for p in ['constraints','inner_objective','outer_objective','prod_id']:
                    if hasattr(m,p) and getattr(m,p) is not None:
                        param = getattr(m,p)
                        if p == 'constraints':
                            for c in param:
                                for k in c[0].keys():
                                    no_par_compress_reacs.add(k)
                        if p in ['inner_objective','outer_objective','prod_id']:
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
                    self.cmp_mapReac += [(reac_map_exp,odd,self.cmp_ko_cost,self.cmp_ki_cost)]
                    # compress information in strain design modules
                    if odd:
                        for new_reac, old_reac_val in reac_map_exp.items():
                            for i,m in enumerate(sd_modules):
                                for p in ['constraints','inner_objective','outer_objective','prod_id']:
                                    if hasattr(m,p) and getattr(m,p) is not None:
                                        param = getattr(m,p)
                                        if p == 'constraints':
                                            for c in param:
                                                if np.any([k in old_reac_val for k in c[0].keys()]):
                                                    lumped_reacs = [k for k in c[0].keys() if k in old_reac_val]
                                                    c[0][new_reac] = np.sum([c[0].pop(k)*old_reac_val[k] for k in lumped_reacs])
                                        if p in ['inner_objective','outer_objective','prod_id']:
                                            if np.any([k in old_reac_val for k in param.keys()]):
                                                lumped_reacs = [k for k in param.keys() if k in old_reac_val]
                                                param[new_reac] = np.sum([param.pop(k)*old_reac_val[k] for k in lumped_reacs if k in old_reac_val])
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
            for m in sd_modules:
                m.model = cmp_model
        # Build MILP
        kwargs1 = kwargs
        kwargs1['ko_cost'] = self.cmp_ko_cost
        kwargs1['ki_cost'] = self.cmp_ki_cost
        kwargs1.pop('compress')
        if 'gko_cost' in kwargs1:
            kwargs1.pop('gko_cost')
        if 'gki_cost' in kwargs1:
            kwargs1.pop('gki_cost')
        print("Finished preprocessing:")
        print("  Model size: "+str(len(cmp_model.reactions))+" reactions, "+str(len(cmp_model.metabolites))+" metabolites")
        print("  "+str(len(self.cmp_ko_cost)+len(self.cmp_ki_cost))+" targetable reactions")
        super().__init__(cmp_model,sd_modules, *args, **kwargs1)

    def expand_mcs(self):
        rmcs = self.cmp_rmcs.copy()
        # expand mcs by applying the compression steps in the reverse order
        cmp_map = self.cmp_mapReac[::-1]
        for exp in cmp_map:
            reac_map_exp = exp[0]
            par_reac_cmp = not exp[1] # if parallel or sequential reactions were lumped
            ko_cost = exp[2] 
            ki_cost = exp[3] 
            for r_cmp,r_orig in reac_map_exp.items():
                if len(r_orig) > 1:
                    for m in rmcs.copy():
                        if r_cmp in m:
                            val = m[r_cmp]
                            del m[r_cmp]
                            if val < 0: # case: KO
                                if par_reac_cmp:
                                    new_m = m.copy()
                                    for d in r_orig:
                                        if d in ko_cost:
                                            new_m[d] = val
                                    rmcs += [new_m]
                                else:
                                    for d in r_orig:
                                        if d in ko_cost:
                                            new_m = m.copy()
                                            new_m[d] = val
                                            rmcs += [new_m]
                            elif val > 0: # case: KI
                                if par_reac_cmp:
                                    for d in r_orig:
                                        if d in ko_cost:
                                            new_m = m.copy()
                                            new_m[d] = val
                                            rmcs += [new_m]
                                else:
                                    new_m = m.copy()
                                    for d in r_orig:
                                        if d in ki_cost:
                                            new_m[d] = val
                                    rmcs += [new_m]
                            rmcs.remove(m)
        # eliminate mcs that are too expensive
        if self.max_cost:
            costs = [np.sum([self.uncmp_ko_cost[k] if v<0 else self.uncmp_ki_cost[k] for k,v in m.items()]) for m in rmcs]
            self.rmcs = [rmcs[i] for i in range(len(rmcs)) if costs[i] <= self.max_cost]
        else:
            self.rmcs = rmcs

    # function wrappers for compute, compute_optimal and enumerate
    def enumerate(self, *args, **kwargs):
        self.cmp_rmcs, status = super().enumerate(*args, **kwargs)
        if status in [0,3]:
            self.expand_mcs()
        else:
            self.rmcs = []
        print(str(len(self.rmcs)) +' solutions found.')
        return self.rmcs, status
    
    def compute_optimal(self, *args, **kwargs):
        self.cmp_rmcs, status = super().compute_optimal(*args, **kwargs)
        if status in [0,3]:
            self.expand_mcs()
        else:
            self.rmcs = []
        print(str(len(self.rmcs)) +' solutions found.')
        return self.rmcs, status
    
    def compute(self, *args, **kwargs):
        self.cmp_rmcs, status = super().compute(*args, **kwargs)
        if status in [0,3]:
            self.expand_mcs()
        else:
            self.rmcs = []
        print(str(len(self.rmcs)) +' solutions found.')
        return self.rmcs, status