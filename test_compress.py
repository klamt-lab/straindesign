import straindesign as sd
from straindesign.names import *
from typing import List
import cobra
from cobra.util.array import create_stoichiometric_matrix
import logging
import numpy
import jpype
import os
import sympy
import io
import sys  # added for relative path lookup
from contextlib import redirect_stdout, redirect_stderr
import numpy as np
from scipy import sparse
from sympy.core.numbers import One
from sympy import Rational, nsimplify
from typing import List
from cobra.util.array import create_stoichiometric_matrix
import ast


logging.basicConfig(level=logging.INFO)


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
    stoich_mat = DefaultBigIntegerRationalMatrix(num_met, num_reac)
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
            n, d = sympyRat2jBigIntegerPair(v)
            stoich_mat.setValueAt(model.metabolites.index(k.id), i, BigFraction(n, d))
    # compress
    smc = StoichMatrixCompressor(subset_compression)
    reacNames = jpype.JString[:](model.reactions.list_attr('id'))
    comprec = smc.compress(stoich_mat, reversible, jpype.JString[num_met], reacNames, None)
    subset_matrix = jpypeArrayOfArrays2numpy_mat(comprec.post.getDoubleRows())
    del_rxns = np.logical_not(np.any(subset_matrix, axis=1))  # blocked reactions
    for j in range(subset_matrix.shape[1]):
        rxn_idx = subset_matrix[:, j].nonzero()[0]
        r0 = rxn_idx[0]
        model.reactions[r0].subset_rxns = []
        model.reactions[r0].subset_stoich = []
        for r in rxn_idx:  # rescale all reactions in this subset
            # !! swaps lb, ub when the scaling factor is < 0, but does not change the magnitudes
            factor = jBigFraction2sympyRat(comprec.post.getBigFractionValueAt(r, j))
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

def search_for_jvm():
    common_java_paths = [
        "C:\\Program Files\\Java",  # Windows
        "/usr/lib/jvm",  # Linux
        "/Library/Java/JavaVirtualMachines",  # macOS
        os.path.dirname(sys.executable)
    ]
    for base in common_java_paths:
        if os.path.exists(base):
            for root, _dirs, files in os.walk(base):
                if any(lib in files for lib in ["jvm.dll", "libjvm.so", "libjvm.dylib"]):
                    return root
    return None

"""Initialization of the java machine, since efmtool compression is done in java."""
efmtool_jar = os.path.join(os.path.dirname(__file__), 'straindesign/efmtool.jar')
jpype.addClassPath(efmtool_jar)
if not jpype.isJVMStarted():
    # Look up JVM at different locations
    if not os.environ.get("JAVA_HOME"):
        candidate = search_for_jvm()
        if candidate:
            os.environ["JAVA_HOME"] = candidate
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):  # suppress console output
            jpype.startJVM()
    except Exception as e:
        extra_info = ""
        if not os.environ.get("JAVA_HOME"):
            extra_info = " JAVA_HOME is not defined."
        raise RuntimeError("Failed to start JVM. Please ensure that Java (OpenJDK) is installed." + extra_info +
                           " If using conda, install openjdk from conda-forge and set JAVA_HOME to the OpenJDK installation path.") from e
import jpype.imports

import ch.javasoft.smx.impl.DefaultBigIntegerRationalMatrix as DefaultBigIntegerRationalMatrix
import ch.javasoft.smx.ops.Gauss as Gauss
import ch.javasoft.metabolic.compress.CompressionMethod as CompressionMethod

subset_compression = CompressionMethod[:](
    [CompressionMethod.CoupledZero, CompressionMethod.CoupledCombine, CompressionMethod.CoupledContradicting])
import ch.javasoft.metabolic.compress.StoichMatrixCompressor as StoichMatrixCompressor
import ch.javasoft.math.BigFraction as BigFraction
import java.math.BigInteger as BigInteger

jTrue = jpype.JBoolean(True)
jSystem = jpype.JClass("java.lang.System")

# # try to find a working java executable
# _java_executable = 'java'
# try:
#     cp = subprocess.run([_java_executable, '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#     if cp.returncode != 0:
#         _java_executable = ''
# except:
#     _java_executable = ''

# if _java_executable == '':
#     _java_executable = os.path.join(os.environ.get('JAVA_HOME', ''), "bin", "java")
#     try:
#         cp = subprocess.run([_java_executable, '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#         if cp.returncode != 0:
#             _java_executable = ''
#     except:
#         _java_executable = ''
# if _java_executable == '':
#     _java_executable = os.path.join(str(jpype.jSystem.getProperty("java.home")), "bin", "java")
# if _java_executable == '':
#     import shutil
#     _java_executable = shutil.which("java")


def basic_columns_rat(mx, tolerance=0):  # mx is ch.javasoft.smx.impl.DefaultBigIntegerRationalMatrix
    """efmtool: Translate matrix coefficients to rational numbers"""
    if type(mx) is numpy.ndarray:
        mx = DefaultBigIntegerRationalMatrix(numpy_mat2jpypeArrayOfArrays(mx), jTrue, jTrue)
    row_map = jpype.JInt[mx.getRowCount()]  # just a placeholder because we don't care about the row permutation here
    col_map = jpype.JInt[:](range(mx.getColumnCount()))
    rank = Gauss.getRationalInstance().rowEchelon(mx, False, row_map, col_map)

    return col_map[0:rank]


def numpy_mat2jpypeArrayOfArrays(npmat):
    """efmtool: Translate matrix to array of arrays"""
    rows = npmat.shape[0]
    cols = npmat.shape[1]
    jmat = jpype.JDouble[rows, cols]
    # for sparse matrices can use nonzero() here instead of iterating through everything
    for r in range(rows):
        for c in range(cols):
            jmat[r][c] = npmat[r, c]
    return jmat


def jpypeArrayOfArrays2numpy_mat(jmat):
    """efmtool: Translate array of arrays to numpy matrix"""
    rows = len(jmat)
    cols = len(jmat[0])  # assumes all rows have the same number of columns
    npmat = numpy.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            npmat[r, c] = jmat[r][c]
    return npmat


def sympyRat2jBigIntegerPair(val):
    """efmtool: Translate rational numbers to big integer pair"""
    numer = val.p  # numerator
    if numer.bit_length() <= 63:
        numer = BigInteger.valueOf(numer)
    else:
        numer = BigInteger(str(numer))
    denom = val.q  # denominator
    if denom.bit_length() <= 63:
        denom = BigInteger.valueOf(denom)
    else:
        denom = BigInteger(str(denom))
    return (numer, denom)


def jBigFraction2sympyRat(val):
    """efmtool: Translate rational numbers to sympy rational numbers"""
    return jBigIntegerPair2sympyRat(val.getNumerator(), val.getDenominator())


def jBigIntegerPair2sympyRat(numer, denom):
    """efmtool: Translate big integer pair to sympy rational numbers"""
    if numer.bitLength() <= 63:
        numer = numer.longValue()
    else:
        numer = str(numer.toString())
    if denom.bitLength() <= 63:
        denom = denom.longValue()
    else:
        denom = str(denom.toString())
    return sympy.Rational(numer, denom)


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
    basic_metabolites = basic_columns_rat(stoich_mat.transpose().toarray(), tolerance=0)
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

print("def compression_in_python_made_by_claude() ...")

# Test both approaches
print("="*60)
print("TESTING COMPRESSION APPROACH")
print("="*60)

# Test EFMtool approach
print("\n1. EFMtool approach:")
# model = cobra.io.read_sbml_model("tests/iMLcore.xml")
model1 = cobra.io.load_model("e_coli_core")
sd.extend_model_gpr(model1)

compression_map_efmtool = compress_model(model1)

print(f"\nEFMtool results: {len(compression_map_efmtool)} compression steps")
print(f"Final model size: {len(model1.reactions)} reactions, {len(model1.metabolites)} metabolites")

# Test python implementation of EFMtool compression
print("1. Claude's Python EFMtool approach")