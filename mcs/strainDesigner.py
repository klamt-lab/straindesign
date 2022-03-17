import numpy as np
from scipy import sparse
from typing import Dict, List, Tuple
from scipy import sparse
from cobra import Model, Metabolite, Reaction
from cobra.util.array import create_stoichiometric_matrix
from mcs import StrainDesignMILP, StrainDesignMILPBuilder, MILP_LP, SD_Module, get_rids
from warnings import warn
import jpype
from sympy import Rational, nsimplify
import efmtool_link.efmtool4cobra as efm
import efmtool_link.efmtool_intern as efmi
import java.util.HashSet

def remove_blocked_reactions(model):
    blocked_reactions = [reac for reac in model.reactions if reac.bounds == (0, 0)]
    model.remove_reactions(blocked_reactions)

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

# compression function imported from efmtool
def compress_model(model, protected_rxns=[]):
    # modifies the model that is passed as first parameter; if you want to preserve the original model copy it first
    # 1) Remove GPR rules
    remove_gene_reaction_rules=True # This was formerly in the function parameters
    if remove_gene_reaction_rules:
        # remove all rules because processing them during combination of reactions into subsets
        # can sometimes raise MemoryErrors (probably when subsets get very large)
        for r in model.reactions:
            r.gene_reaction_rule = ''
    # 2) load configuration an initialize variables, transcribe stoichiometric coefficients as
    #    Rationals to java-matrix.
    old_num_reac = len(model.reactions)
    old_objective = [r.objective_coefficient for r in model.reactions]
    # duplicate protected reactions prevents them from being lumped.
    # as this function 
    no_lump_aux_reac = []
    for s in protected_rxns:
        reac = model.reactions.get_by_id(s).copy()
        reac.id = 'no_lump_'+s
        no_lump_aux_reac += [reac.id]
        model.add_reactions([reac])
    config = efm.Configuration()
    num_met = len(model.metabolites)
    num_reac = len(model.reactions)
    stoich_mat = efm.DefaultBigIntegerRationalMatrix(num_met, num_reac)
    # reversible = jpype.JBoolean[num_reac]
    reversible = jpype.JBoolean[:]([r.reversibility for r in model.reactions])
    # start_time = time.monotonic()
    flipped = []
    for i in range(num_reac):
        # 3.1) flip negative reactions (unless protected from compression)
        if model.reactions[i].upper_bound <= 0 and model.reactions[i].id not in protected_rxns: # can run in backwards direction only (is and stays classified as irreversible)
            model.reactions[i] *= -1
            flipped.append(i)
            # print("Flipped", model.reactions[i].id)
        # 3.2) replace all stoichiometric coefficients with rationals.
        # have to use _metabolites because metabolites gives only a copy
        for k, v in model.reactions[i]._metabolites.items():
            n, d = efm.sympyRat2jBigIntegerPair(v)
            stoich_mat.setValueAt(model.metabolites.index(k.id), i, efm.BigFraction(n, d))
    # initialize compressor
    smc = efm.StoichMatrixCompressor(efmi.subset_compression)
    reacNames = jpype.JString[len(model.reactions)]
    comprec = smc.compress(stoich_mat, reversible, jpype.JString[num_met], reacNames, None)
    # would be faster to do the computations with floats and afterwards substitute the coefficients
    # with rationals from efmtool
    subset_matrix= efmi.jpypeArrayOfArrays2numpy_mat(comprec.post.getDoubleRows())
    del_rxns = np.logical_not(np.any(subset_matrix, axis=1)) # blocked reactions
    # do not lump protected reactions
    subset_matrix = sparse.csr_matrix(subset_matrix)
    for j in range(subset_matrix.shape[1]):
        rxn_idx = subset_matrix[:, j].nonzero()[0]
        for i in range(len(rxn_idx)): # rescale reactions in this subset
            # !! swaps lb, ub when the scaling factor is < 0, but does not change the magnitudes
            factor = efm.jBigFraction2sympyRat(comprec.post.getBigFractionValueAt(rxn_idx[i], j))
            # factor = jBigFraction2intORsympyRat(comprec.post.getBigFractionValueAt(rxn_idx[i], j)) # does not appear to make a speed difference
            model.reactions[rxn_idx[i]] *=  factor #subset_matrix[rxn_idx[i], j]
            # factor = abs(float(factor)) # context manager has trouble with non-float bounds
            if model.reactions[rxn_idx[i]].lower_bound not in (0, config.lower_bound, -float('inf')):
                model.reactions[rxn_idx[i]].lower_bound/= abs(subset_matrix[rxn_idx[i], j]) #factor
            if model.reactions[rxn_idx[i]].upper_bound not in (0, config.upper_bound, float('inf')):
                model.reactions[rxn_idx[i]].upper_bound/= abs(subset_matrix[rxn_idx[i], j]) #factor
        model.reactions[rxn_idx[0]].subset_rxns = rxn_idx # reaction indices of the base model
        model.reactions[rxn_idx[0]].subset_stoich = [subset_matrix[rxn_idx, j].data[0]] # use rationals here?
        for i in range(1, len(rxn_idx)): # merge reactions
            # !! keeps bounds of reactions[rxn_idx[0]]
            model.reactions[rxn_idx[0]] += model.reactions[rxn_idx[i]] # combine stoichiometries
            model.reactions[rxn_idx[0]].subset_stoich += [subset_matrix[int(rxn_idx[i]), j]]
            if len(model.reactions[rxn_idx[0]].id)+len(model.reactions[rxn_idx[i]].id) < 220:
                model.reactions[rxn_idx[0]].id += '*'+model.reactions[rxn_idx[i]].id # combine names
            elif not model.reactions[rxn_idx[0]].id[-3:] == '...':
                model.reactions[rxn_idx[0]].id += '...'
            if model.reactions[rxn_idx[i]].lower_bound > model.reactions[rxn_idx[0]].lower_bound:
                model.reactions[rxn_idx[0]].lower_bound = model.reactions[rxn_idx[i]].lower_bound
            if model.reactions[rxn_idx[i]].upper_bound < model.reactions[rxn_idx[0]].upper_bound:
                model.reactions[rxn_idx[0]].upper_bound = model.reactions[rxn_idx[i]].upper_bound
            del_rxns[rxn_idx[i]] = True
    # delete auxiliary reactions
    for i,r in enumerate(model.reactions):
        if r.id in no_lump_aux_reac:
            del_rxns[i] = True
    del_rxns = np.where(del_rxns)[0]
    for i in range(len(del_rxns)-1, -1, -1): # delete in reversed index order to keep indices valid
        model.reactions[del_rxns[i]].remove_from_model(remove_orphans=True)
    subT = np.zeros((old_num_reac, len(model.reactions)))
    for j in range(subT.shape[1]):
        subT[model.reactions[j].subset_rxns, j] = model.reactions[j].subset_stoich
    for i in flipped: # adapt so that it matches the reaction direction before flipping
            subT[i, :] *= -1
    # map back objective coefficients
    new_objective = old_objective@subT
    for r,c in zip(model.reactions,new_objective):
        r.objective_coefficient = c
    return subT

def compress_model_parallel(model, protected_rxns=[]):
    # lump parallel reactions
    # 
    # - exclude lumping of reactions with inhomogenous bounds
    # - exclude protected reactions
    old_num_reac = len(model.reactions)
    old_objective = [r.objective_coefficient for r in model.reactions]
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
    subT = np.zeros((old_num_reac, len(model.reactions)))
    for i in range(subT.shape[1]):
        for j in subset_list[i]:
            subT[j,i] = 1

    new_objective = old_objective@subT
    for r,c in zip(model.reactions,new_objective):
        r.objective_coefficient = c
    return subT


class StrainDesigner(StrainDesignMILP):
    def __init__(self, model: Model, sd_modules: List[SD_Module], *args, **kwargs):
        allowed_keys = {'solver', 'max_cost', 'M','threads', 'mem', 'compress', 'options','ko_cost','ki_cost'}
        # set all keys that are not in kwargs to None
        for key in allowed_keys:
            if key not in dict(kwargs).keys():
                kwargs[key] = None
        # check all keys passed in kwargs
        for key, value in dict(kwargs).items():
            if key in allowed_keys:
                setattr(self,key,value)
            else:
                raise Exception("Key " + key + " is not supported.")
            if key == 'ko_cost':
                self.cmp_ko_cost = value
                self.orig_ko_cost = value
            if key == 'ki_cost':
                self.cmp_ki_cost = value
                self.orig_ki_cost = value
        if not kwargs['ko_cost']:
            self.cmp_ko_cost = {k:1 for k in model.reactions.list_attr('id')}
            self.orig_ko_cost = {k:1 for k in model.reactions.list_attr('id')}
        if not kwargs['ki_cost']:
            self.cmp_ki_cost = {}
            self.orig_ki_cost = {}
        # put module in list if only one module was provided
        if "SD_Module" in str(type(sd_modules)):
            sd_modules = [sd_modules]
        # 1) Preprocess Model
        print('Preparing strain design computation.')
        # remove external metabolites
        remove_ext_mets(model)
        # replace model bounds with +/- inf if above a certain threshold
        bound_thres = 1000
        for i in range(len(model.reactions)):
            if model.reactions[i].lower_bound <= -bound_thres:
                model.reactions[i].lower_bound = -np.inf
            if model.reactions[i].upper_bound >=  bound_thres:
                model.reactions[i].upper_bound =  np.inf
        
        # Compress model
        cmp_model = model.copy()
        self.cmp_mapReac = []
        if  kwargs['compress'] is True or kwargs['compress'] is None: # If compression is activated (or not defined)
            print('Compressing Network ('+str(len(cmp_model.reactions))+' reactions).')
            # exclude reactions named in strain design modules from compression
            no_compress_reacs = []
            for m in sd_modules:
                for p in ['constraints','inner_objective','outer_objective','prod_id','numerator','denomin']:
                    if hasattr(m,p) and getattr(m,p) is not None:
                        expr = getattr(m,p)
                        if isinstance(expr, str):
                            expr = [expr]
                        for s in expr:
                            rid = get_rids(s,model.reactions.list_attr('id'))
                            [no_compress_reacs.append(r) for r in rid if r not in no_compress_reacs]
            # compress network by lumping sequential and parallel reactions alternatingly. 
            # Remove conservation relations.
            print('  Removing blocked reactions.')
            remove_blocked_reactions(cmp_model)
            print('  Translating stoichiometric coefficients to rationals.')
            stoichmat_coeff2rational(cmp_model)
            print('  Removing conservation relations.')
            remove_conservation_relations(cmp_model)
            odd = True
            run = 1
            while True:
                prev_reac_names = cmp_model.reactions.list_attr('id')
                if odd:
                    print('  Compression '+str(run)+': Applying compression from EFM-tool module.')
                    subT = sparse.csc_matrix(compress_model(cmp_model, no_compress_reacs))
                else:
                    print('  Compression '+str(run)+': Lumping parallel reactions.')
                    subT = sparse.csc_matrix(compress_model_parallel(cmp_model, no_compress_reacs))
                remove_conservation_relations(cmp_model)
                if subT.shape[0] > subT.shape[1]:
                    print('  Reduced to '+str(subT.shape[1])+' reactions.')
                    reac_map_exp = {} # map compressed reactions to original reactions (needed for kiCost, koCost mapping and decompression)
                    for i in range(subT.shape[1]):
                        reac_map_exp[cmp_model.reactions[i].id] = [prev_reac_names[j] for j in subT[:,i].indices]
                    # store the information for decompression in a Tuple
                    # (0) compression matrix, (1) reac_id dictornary {cmp_rid: [orig_rids]}, (2) linear (True) or parallel (False) compression
                    # (3,4) ko and ki costs of expanded network
                    self.cmp_mapReac += [(subT,reac_map_exp,odd,self.cmp_ko_cost,self.cmp_ki_cost)]
                    # compress ko_cost and ki_cost
                    # ko_cost of lumped reactions: when reacs sequential: lowest of ko costs, when parallel: sum of ko costs
                    # ki_cost of lumped reactions: when reacs sequential: sum of ki costs, when parallel: lowest of ki costs
                    if self.cmp_ko_cost:
                        ko_cost_new = {}
                        for r in cmp_model.reactions.list_attr('id'):
                            if np.any([s in self.cmp_ko_cost for s in reac_map_exp[r]]):
                                if odd:
                                    ko_cost_new[r] = np.min([self.cmp_ko_cost[s] for s in reac_map_exp[r] if s in self.cmp_ko_cost])
                                else:
                                    ko_cost_new[r] = np.sum([self.cmp_ko_cost[s] for s in reac_map_exp[r] if s in self.cmp_ko_cost])
                        self.cmp_ko_cost = ko_cost_new
                    if self.cmp_ki_cost:
                        ki_cost_new = {}
                        for r in cmp_model.reactions.list_attr('id'):
                            if np.any([s in self.cmp_ki_cost for s in reac_map_exp[r]]):
                                if odd:
                                    ki_cost_new[r] = np.sum([self.cmp_ki_cost[s] for s in reac_map_exp[r] if s in self.cmp_ki_cost])
                                else:
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
                    print('  Translating stoichiometric coefficients back to real (float or int) numbers.')
                    stoichmat_coeff2float(cmp_model)
                    break
            for m in sd_modules:
                m.model = cmp_model

        # Build MILP
        kwargs1 = kwargs
        kwargs1['ko_cost'] = self.cmp_ko_cost
        kwargs1['ki_cost'] = self.cmp_ki_cost
        del kwargs1['compress']
        super().__init__(cmp_model,sd_modules, *args, **kwargs1)

    def expand_mcs(self):
        rmcs = self.cmp_rmcs.copy()
        # expand mcs by applying the compression steps in the reverse order
        cmp_map = self.cmp_mapReac[::-1]
        for exp in cmp_map:
            reac_map_exp = exp[1]
            par_reac_cmp = not exp[2] 
            ko_cost = exp[3] 
            ki_cost = exp[4] 
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
            costs = [np.sum([self.orig_ko_cost[k] if v<0 else self.orig_ki_cost[k] for k,v in m.items()]) for m in rmcs]
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