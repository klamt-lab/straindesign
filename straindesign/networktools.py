import numpy as np
from scipy import sparse
from sympy.core.numbers import One
from sympy import Rational, nsimplify
from typing import Dict, List
import jpype
from cobra import Model, Metabolite, Reaction
from cobra.util.array import create_stoichiometric_matrix
import straindesign.efmtool as efm

def extend_model_gpr(model,gkos,gkis):
    # Split reactions when necessary
    reac_map = {}
    rev_reac = set()
    del_reac = set()
    for r in model.reactions:
        reac_map.update({r.id:{}})
        if not r.gene_reaction_rule:
            reac_map[r.id].update({r.id: 1.0})
            continue
        if r.gene_reaction_rule and r.bounds[0] < 0:
            r_rev = (r*-1)
            if r.gene_reaction_rule and r.bounds[1] > 0:
                r_rev.id = r.id+'_reverse_'+hex(hash(r))[8:]
            r_rev.lower_bound = np.max([0,r_rev.lower_bound])
            reac_map[r.id].update({r_rev.id: -1.0})
            rev_reac.add(r_rev)
        if r.gene_reaction_rule and r.bounds[1] > 0:
            reac_map[r.id].update({r.id: 1.0})
            r._lower_bound = np.max([0,r._lower_bound])
        else:
            del_reac.add(r)
    model.remove_reactions(del_reac)
    model.add_reactions(rev_reac)
    
    gene_names = set(g.name for g in model.genes)
    gene_names_exist = np.all([len(g.name) for g in model.genes])
    use_name_not_id = gene_names_exist and (gene_names.intersection(gkos) or gene_names.intersection(gkis))
   
    # All reaction rules are provided in dnf.
    for r in model.reactions:
        if r.gene_reaction_rule: # if reaction has a gpr rule
            dt = [s.strip() for s in r.gene_reaction_rule.split(' or ')]
            for i,p in enumerate(dt.copy()):
                ct = [s.strip() for s in p.replace('(','').replace(')','').split(' and ')]
                for j,g in enumerate(ct.copy()):
                    gene_met_id = 'g_'+g
                    # if gene is not in model, add gene pseudoreaction and metabolite
                    if gene_met_id not in model.metabolites.list_attr('id'):
                        model.add_metabolites(Metabolite(gene_met_id))
                        gene = model.genes.get_by_id(g)
                        w = Reaction(gene.id)
                        if use_name_not_id: # if gene name is available and used in gki_cost and gko_cost
                            w.id = gene.name
                        model.add_reaction(w)
                        w.reaction = '--> '+gene_met_id
                        w._upper_bound = np.inf
                    ct[j] = gene_met_id
                if len(ct) > 1:
                    ct_met_id = "_and_".join(ct)
                    if ct_met_id not in model.metabolites.list_attr('id'):
                        # if conjunct term is not in model, add pseudoreaction and metabolite
                        model.add_metabolites(Metabolite(ct_met_id))
                        w = Reaction("R_"+ct_met_id)
                        model.add_reaction(w)
                        w.reaction = ' + '.join(ct)+'--> '+ct_met_id
                        w._upper_bound = np.inf
                    dt[i] = ct_met_id
                else:
                    dt[i] = gene_met_id
            if len(dt) > 1:
                dt_met_id = "_or_".join(dt)
                if dt_met_id not in model.metabolites.list_attr('id'):
                    model.add_metabolites(Metabolite(dt_met_id))
                    for k,d in enumerate(dt):
                        w = Reaction("R"+str(k)+"_"+dt_met_id)
                        model.add_reaction(w)
                        w.reaction = d+' --> '+dt_met_id
                        w._upper_bound = np.inf
            else:
                dt_met_id = dt[0]
            r.add_metabolites({model.metabolites.get_by_id(dt_met_id): -1.0})
    return reac_map

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
    smc = efm.StoichMatrixCompressor(efm.subset_compression)
    reacNames = jpype.JString[:](model.reactions.list_attr('id'))
    comprec = smc.compress(stoich_mat, reversible, jpype.JString[num_met], reacNames,None)
    subset_matrix= efm.jpypeArrayOfArrays2numpy_mat(comprec.post.getDoubleRows())
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
            if r in flipped:
                model.reactions[r0].subset_stoich.append(-factor)
            else:
                model.reactions[r0].subset_stoich.append(factor)
        for r in rxn_idx[1:]: # merge reactions
            # rename main reaction
            if len(model.reactions[r0].id)+len(model.reactions[r].id) < 220 and model.reactions[r0].id[-3:] != '...':
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
    # for i in flipped: # adapt so that it matches the reaction direction before flipping
    #         subT[i, :] *= -1
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
            if len(model.reactions[rxn_idx[0]].id)+len(model.reactions[rxn_idx[i]].id) < 220  and model.reactions[rxn_idx[0]].id[-3:] != '...':
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
    basic_metabolites = efm.basic_columns_rat(stoich_mat.transpose().toarray(), tolerance=0)
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