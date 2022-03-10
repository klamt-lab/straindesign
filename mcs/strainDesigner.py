from re import S
from sre_compile import isstring
import numpy as np
from scipy import sparse
from typing import Dict, List, Tuple
from scipy import sparse
from cobra import Model, Metabolite
from cobra.util.array import create_stoichiometric_matrix
from mcs import StrainDesignMILP, StrainDesignMILPBuilder, MILP_LP, SD_Module, get_rids
from warnings import warn
import jpype
from sympy import Rational, nsimplify
import efmtool_link.efmtool4cobra as efm
import efmtool_link.efmtool_intern as efmi
import java.util.HashSet

# compression function imported from efmtool
def compress_model(model, protected_rxns=[]):
    # modifies the model that is passed as first parameter; if you want to preserve the original model copy it first
    # all irreversible reactions in the compressed model will flow in the forward direction
    remove_gene_reaction_rules=True # This was formerly in the function parameters
    if remove_gene_reaction_rules:
        # remove all rules because processing them during combination of reactions into subsets
        # can sometimes raise MemoryErrors (probably when subsets get very large)
        for r in model.reactions:
            r.gene_reaction_rule = ''
    # remove conservation relations
    stoich_mat = create_stoichiometric_matrix(model, array_type='lil')
    basic_metabolites = efmi.basic_columns_rat(stoich_mat.transpose().toarray(), tolerance=0)
    dependent_metabolites = [model.metabolites[i].id for i in set(range(len(model.metabolites))) - set(basic_metabolites)]
    # print("The following metabolites have been removed from the model:")
    # print(dependent_metabolites)
    for m in dependent_metabolites:
        model.metabolites.get_by_id(m).remove_from_model()
    # start reaction compression
    remove_rxns = [] # This was formerly in the function parameters

    # add pseudo metabolites for reactions that should not be lumped with others.
    # aux_metab_no_lump = ['no_lump_'+kr for kr in protected_rxns]
    # [model.add_metabolites(Metabolite(s)) for s in aux_metab_no_lump]
    # [getattr(model.reactions,r).add_metabolites({m:1}) for r,m in zip(protected_rxns,aux_metab_no_lump)]

    config = efm.Configuration()
    num_met = len(model.metabolites)
    num_reac = len(model.reactions)
    stoich_mat = efm.DefaultBigIntegerRationalMatrix(num_met, num_reac)
    # reversible = jpype.JBoolean[num_reac]
    reversible = jpype.JBoolean[:]([r.reversibility for r in model.reactions])
    # start_time = time.monotonic()
    flipped = []
    for i in range(num_reac):
        if model.reactions[i].bounds == (0, 0): # blocked reaction
            remove_rxns.append(model.reactions[i].id)
        elif model.reactions[i].upper_bound <= 0: # can run in backwards direction only (is and stays classified as irreversible)
            model.reactions[i] *= -1
            flipped.append(i)
            # print("Flipped", model.reactions[i].id)
        # have to use _metabolites because metabolites gives only a copy
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
            elif type(v) is not Rational:
                raise TypeError
            n, d = efm.sympyRat2jBigIntegerPair(v)
            # does not work although there is a public void setValueAt(int row, int col, BigInteger numerator, BigInteger denominator) method
            # leads to kernel crash directly or later
            # stoic_mat.setValueAt(compr_model.metabolites.index(k.id), i, n, d)
            stoich_mat.setValueAt(model.metabolites.index(k.id), i, efm.BigFraction(n, d))
            # reversible[i] = compr_model.reactions[i].reversibility # somehow makes problems with the smc.compress call
    
    smc = efm.StoichMatrixCompressor(efmi.subset_compression)
    if len(remove_rxns) == 0:
        reacNames = jpype.JString[num_reac]
        remove_rxns = None
    else:
        reacNames = jpype.JString[:](model.reactions.list_attr('id'))
        remove_rxns = java.util.HashSet(remove_rxns) # works because of some jpype magic
        # print("Removing", remove_rxns.size(), "reactions:")
        # print(remove_rxns.toString())
    comprec = smc.compress(stoich_mat, reversible, jpype.JString[num_met], reacNames, remove_rxns)
    del remove_rxns
    # print(time.monotonic() - start_time) # 20 seconds in iJO1366 without remove_rxns
    # start_time = time.monotonic()

    # would be faster to do the computations with floats and afterwards substitute the coefficients
    # with rationals from efmtool
    subset_matrix= efmi.jpypeArrayOfArrays2numpy_mat(comprec.post.getDoubleRows())
    del_rxns = np.logical_not(np.any(subset_matrix, axis=1)) # blocked reactions
    # do not lump protected reactions
    subset_matrix = sparse.csr_matrix(subset_matrix)
    for i,s in enumerate(model.reactions.list_attr('id')):
        if s in protected_rxns:
            col_idx = subset_matrix[i].indices[0]
            col_old = subset_matrix[:,col_idx]
            # if reaction was marked to be lumped with another, separate them again.
            if col_old.nnz == 0:
                raise Exception('reaction '+s+' was deleted unexpectedly during network compression')
            if col_old.nnz > 1:
                subset_matrix[i,col_idx] = 0
                col_new = sparse.csc_matrix(([1.],([i],[0])),(subset_matrix.shape[0],1))
                subset_matrix = sparse.hstack((subset_matrix[:,range(col_idx)],col_new,subset_matrix[:,range(col_idx,subset_matrix.shape[1])]),format='csr')
    for j in range(subset_matrix.shape[1]):
        rxn_idx = subset_matrix[:, j].nonzero()[0]
        for i in range(len(rxn_idx)): # rescale reactions in this subset
            if model.reactions[rxn_idx[i]].id not in protected_rxns: # except for protected reactions
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
        model.reactions[rxn_idx[0]].subset_stoich = subset_matrix[rxn_idx, j].data[0] # use rationals here?
        for i in range(1, len(rxn_idx)): # merge reactions
            # !! keeps bounds of reactions[rxn_idx[0]]
            model.reactions[rxn_idx[0]] += model.reactions[rxn_idx[i]]
            if model.reactions[rxn_idx[i]].lower_bound > model.reactions[rxn_idx[0]].lower_bound:
                model.reactions[rxn_idx[0]].lower_bound = model.reactions[rxn_idx[i]].lower_bound
            if model.reactions[rxn_idx[i]].upper_bound < model.reactions[rxn_idx[0]].upper_bound:
                model.reactions[rxn_idx[0]].upper_bound = model.reactions[rxn_idx[i]].upper_bound
            del_rxns[rxn_idx[i]] = True
    # print(time.monotonic() - start_time) # 11 seconds in iJO1366
    del_rxns = np.where(del_rxns)[0]
    for i in range(len(del_rxns)-1, -1, -1): # delete in reversed index order to keep indices valid
        model.reactions[del_rxns[i]].remove_from_model(remove_orphans=True)
    subT = np.zeros((num_reac, len(model.reactions)))
    for j in range(subT.shape[1]):
        subT[model.reactions[j].subset_rxns, j] = model.reactions[j].subset_stoich
    for i in flipped: # adapt so that it matches the reaction direction before flipping
            subT[i, :] *= -1
    # maybe add adapt compressed_model name
    # cast coefficients back from rational to integer or float
    num_reac = len(model.reactions)
    for i in range(num_reac):
        for k, v in model.reactions[i]._metabolites.items():
            if v.is_Integer:
                model.reactions[i]._metabolites[k] = int(v)
            elif v.is_Float or v.is_Rational:
                model.reactions[i]._metabolites[k] = float(v)
            else:
                raise Exception('unknown data type')
    # again remove conservation relations if remaining
    stoich_mat = create_stoichiometric_matrix(model, array_type='lil')
    basic_metabolites = efmi.basic_columns_rat(stoich_mat.transpose().toarray(), tolerance=0)
    dependent_metabolites = [model.metabolites[i].id for i in set(range(len(model.metabolites))) - set(basic_metabolites)]
    # print("The following metabolites have been removed from the model:")
    # print(dependent_metabolites)
    for m in dependent_metabolites:
        model.metabolites.get_by_id(m).remove_from_model()
    return subT


class StrainDesigner(StrainDesignMILP):
    def __init__(self, model: Model, sd_modules: List[SD_Module], *args, **kwargs):
        allowed_keys = {'ko_cost', 'ki_cost', 'solver', 'max_cost', 'M','threads', 'mem', 'compress', 'options'}
        # set all keys passed in kwargs
        for key, value in dict(kwargs).items():
            if key in allowed_keys:
                locals()[key] = value
            else:
                raise Exception("Key " + key + " is not supported.")
        # set all remaining keys to None
        for key in allowed_keys:
            if key not in dict(kwargs).keys():
                locals()[key] = None
        # put module in list if only one module was provided
        if "SD_Module" in str(type(sd_modules)):
            sd_modules = [sd_modules]
        # Preprocess Model
        
        # Compress model
        # exclude reactions named in strain design modules from compression
        no_compress_reacs = []
        for m in sd_modules:
            for p in ['constraints','inner_objective','outer_objective','prod_id','numerator','denomin']:
                if hasattr(m,p) and getattr(m,p) is not None:
                    expr = getattr(m,p)
                    if isstring(expr):
                        expr = [expr]
                    for s in expr:
                        rid = get_rids(s,model.reactions.list_attr('id'))
                        [no_compress_reacs.append(r) for r in rid if r not in no_compress_reacs]
        
        orig_model = model.copy()
        print('Compress Network')
        cmp_mapReac = compress_model(model, no_compress_reacs)

        # Build MILP
        super().__init__(model,sd_modules, *args, **kwargs)

