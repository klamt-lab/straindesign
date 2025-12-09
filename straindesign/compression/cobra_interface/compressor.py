"""
High-level COBRA model compression functionality.

Implements compress_cobra_model() function as specified in API_SPECIFICATION.md
"""

import copy
import numpy as np
from typing import List, Union, Optional, Set
from fractions import Fraction

from ..core import CompressionMethod, StoichMatrixCompressor
from ..math import DefaultBigIntegerRationalMatrix, BigFraction
from .result import CompressionResult
from .converter import CompressionConverter


def compress_cobra_model(
    model,  # cobra.Model - avoid import to prevent dependency issues
    methods: Optional[List[Union[str, CompressionMethod]]] = None,
    in_place: bool = True,
    preprocessing: bool = True,
    suppressed_reactions: Optional[Set[str]] = None
) -> CompressionResult:
    """
    Compress a COBRA model using specified compression methods.
    
    This function provides the main high-level interface for metabolic network compression
    as specified in API_SPECIFICATION.md.
    
    Args:
        model: COBRA model to compress
        methods: Compression methods to apply. Can be:
            - None: Use standard methods (CoupledZero, CoupledCombine, etc.)
            - List of strings: ["CoupledZero", "DeadEnd", "UniqueFlows"]  
            - List of CompressionMethod enums
        in_place: Whether to modify the original model (True) or work on copy (False)
        preprocessing: Whether to apply rational conversion and conservation removal
        suppressed_reactions: Set of reaction names to suppress during compression
        
    Returns:
        CompressionResult object with compressed model and transformation utilities
    """
    # Work on copy if not in-place
    if not in_place:
        model = copy.deepcopy(model)
    
    # Store original state
    original_reaction_names = [r.id for r in model.reactions]
    original_metabolite_names = [m.id for m in model.metabolites]
    
    # Apply preprocessing if requested
    if preprocessing:
        model = preprocess_cobra_model(model)
    
    # Convert methods to CompressionMethod enums
    if methods is None:
        compression_methods = CompressionMethod.standard()
    else:
        compression_methods = _parse_compression_methods(methods)
    
    # Flip reactions that can only run backwards 
    flipped_reactions = []
    for rxn in model.reactions:
        if rxn.upper_bound <= 0:  # Can only run backwards
            rxn *= -1
            flipped_reactions.append(rxn.id)
    
    # Build stoichiometric matrix with exact arithmetic
    stoich_matrix = _build_stoich_matrix(model)
    reversible = [rxn.reversibility for rxn in model.reactions]
    metabolite_names = [m.id for m in model.metabolites]
    reaction_names = [r.id for r in model.reactions]
    
    # Apply compression using FULL algorithm implementation
    compressor = StoichMatrixCompressor(*compression_methods)
    compression_record = compressor.compress(
        stoich_matrix, reversible, metabolite_names, reaction_names, suppressed_reactions
    )
    
    # Build reaction and metabolite maps from transformation matrices
    reaction_map, metabolite_map = _build_transformation_maps(
        compression_record, reaction_names, metabolite_names
    )
    
    # Modify the model in-place based on compression results
    _apply_compression_to_model(model, compression_record, reaction_map)
    
    # Create numpy arrays for easy access
    pre_matrix = _matrix_to_numpy(compression_record.pre)
    post_matrix = _matrix_to_numpy(compression_record.post)
    
    # Create compression converter
    converter = CompressionConverter(reaction_map, metabolite_map, flipped_reactions)
    
    # Return complete result
    return CompressionResult(
        compressed_model=model,
        compression_converter=converter,
        pre_matrix=pre_matrix,
        post_matrix=post_matrix, 
        reaction_map=reaction_map,
        metabolite_map=metabolite_map,
        statistics=compression_record.stats,
        methods_used=compression_methods,
        original_reaction_names=original_reaction_names,
        original_metabolite_names=original_metabolite_names,
        flipped_reactions=flipped_reactions
    )


def preprocess_cobra_model(model, rational_conversion: bool = True, 
                          remove_conservation: bool = True):
    """
    Apply preprocessing steps to COBRA model.
    
    Args:
        model: COBRA model to preprocess
        rational_conversion: Convert coefficients to exact rationals
        remove_conservation: Remove conservation relations
        
    Returns:
        Preprocessed model (modified in-place)
    """
    if rational_conversion:
        _convert_coefficients_to_rational(model)
    
    if remove_conservation:
        _remove_conservation_relations(model)
    
    return model


def _parse_compression_methods(methods: List[Union[str, CompressionMethod]]) -> List[CompressionMethod]:
    """Convert method specifications to CompressionMethod enums."""
    result = []
    
    for method in methods:
        if isinstance(method, str):
            # Handle string method names
            method_upper = method.upper().replace('-', '_').replace(' ', '_')
            
            # Handle common variations
            name_mappings = {
                'DEADEND': 'DEAD_END',
                'UNIQUEFLOWS': 'UNIQUE_FLOWS',
                'COUPLEDZERO': 'COUPLED_ZERO', 
                'COUPLEDCOMBINE': 'COUPLED_COMBINE',
                'COUPLEDCONTRADICTING': 'COUPLED_CONTRADICTING',
                'DUPLICATEGENE': 'DUPLICATE_GENE',
            }
            
            method_name = name_mappings.get(method_upper, method_upper)
            
            try:
                result.append(CompressionMethod[method_name])
            except KeyError:
                raise ValueError(f"Unknown compression method: {method}. "
                               f"Available: {[m.name for m in CompressionMethod]}")
        else:
            result.append(method)
    
    return result


def _build_stoich_matrix(model) -> DefaultBigIntegerRationalMatrix:
    """Build stoichiometric matrix with exact rational arithmetic."""
    n_metabolites = len(model.metabolites)
    n_reactions = len(model.reactions)
    
    # Create matrix with exact arithmetic
    stoich_matrix = DefaultBigIntegerRationalMatrix(n_metabolites, n_reactions)
    
    for j, reaction in enumerate(model.reactions):
        for metabolite, coefficient in reaction.metabolites.items():
            i = model.metabolites.index(metabolite)
            
            # Convert to exact rational
            if isinstance(coefficient, (int, float)):
                rational_coeff = BigFraction(Fraction(coefficient).limit_denominator())
            else:
                rational_coeff = BigFraction(Fraction(str(coefficient)))
            
            stoich_matrix.set_value_at(i, j, rational_coeff)
    
    return stoich_matrix


def _build_transformation_maps(compression_record, reaction_names, metabolite_names):
    """Build transformation maps from compression matrices."""
    # Build reaction map from post matrix
    reaction_map = {}
    post_matrix = compression_record.post
    n_original_reactions = post_matrix.get_row_count()
    n_compressed_reactions = post_matrix.get_column_count()
    
    for j in range(n_compressed_reactions):
        # Find contributing reactions
        contributing_reactions = {}
        reaction_name_parts = []
        
        for i in range(n_original_reactions):
            coeff = post_matrix.get_big_fraction_value_at(i, j)
            if not coeff.is_zero():
                orig_name = reaction_names[i]
                contributing_reactions[orig_name] = coeff
                reaction_name_parts.append(orig_name)
        
        # Create compressed reaction name (using same logic as _apply_compression_to_model)
        if len(reaction_name_parts) == 1:
            compressed_name = reaction_name_parts[0]
        else:
            # Build name by concatenating with '*' until too long, then add '...'
            compressed_name = reaction_name_parts[0]
            for part in reaction_name_parts[1:]:
                if len(compressed_name) + len(part) + 1 < 220 and not compressed_name.endswith('...'):
                    compressed_name += '*' + part
                elif not compressed_name.endswith('...'):
                    compressed_name += '...'
                    break
        
        reaction_map[compressed_name] = contributing_reactions
    
    # Build metabolite map from pre matrix
    metabolite_map = {}
    pre_matrix = compression_record.pre
    n_compressed_metabolites = pre_matrix.get_row_count()
    n_original_metabolites = pre_matrix.get_column_count()
    
    for i in range(n_compressed_metabolites):
        contributing_metabolites = {}
        metabolite_name_parts = []
        
        for j in range(n_original_metabolites):
            coeff = pre_matrix.get_big_fraction_value_at(i, j)
            if not coeff.is_zero():
                orig_name = metabolite_names[j]
                contributing_metabolites[orig_name] = coeff
                metabolite_name_parts.append(orig_name)
        
        # Create compressed metabolite name
        if len(metabolite_name_parts) == 1:
            compressed_name = metabolite_name_parts[0]
        elif metabolite_name_parts:
            compressed_name = f"M{i}_({'_'.join(metabolite_name_parts[:2])})"
        else:
            compressed_name = f"M{i}"
            
        metabolite_map[compressed_name] = contributing_metabolites
    
    return reaction_map, metabolite_map


def _apply_compression_to_model(model, compression_record, reaction_map):
    """Apply compression results to the COBRA model in-place."""
    import numpy as np
    from sympy import Rational
    
    # Convert post matrix to numpy for processing
    post_matrix = _matrix_to_numpy(compression_record.post)
    
    # Safety check: if post matrix has no columns (no compressed reactions), 
    # this means all reactions were eliminated - this is likely an error
    if post_matrix.shape[1] == 0:
        print(f"WARNING: All reactions eliminated during compression! Post matrix shape: {post_matrix.shape}")
        print("This suggests the model may be infeasible or the compression is too aggressive.")
        # Don't apply any changes - keep the original model
        for reaction in model.reactions:
            reaction.notes['compression_applied'] = True
            reaction.notes['compression_warning'] = 'All reactions eliminated - compression not applied'
        return
    
    # Find reactions to delete (those with no contribution to any compressed reaction)
    del_rxns = np.logical_not(np.any(post_matrix, axis=1))
    
    # Process each compressed reaction (column in post matrix)
    for j in range(post_matrix.shape[1]):
        rxn_indices = post_matrix[:, j].nonzero()[0]
        if len(rxn_indices) == 0:
            continue
            
        # Main reaction (first in the group)
        main_idx = rxn_indices[0]
        main_reaction = model.reactions[main_idx]
        
        # Initialize compression tracking
        main_reaction.subset_rxns = []
        main_reaction.subset_stoich = []
        
        # Process all reactions in this compressed group
        for r_idx in rxn_indices:
            reaction = model.reactions[r_idx]
            # Get scaling factor from post matrix
            scaling_factor = Rational(float(post_matrix[r_idx, j]))
            
            # Scale the reaction
            reaction *= scaling_factor
            
            # Update bounds
            if reaction.lower_bound not in (0, -float('inf')):
                reaction.lower_bound /= abs(float(scaling_factor))
            if reaction.upper_bound not in (0, float('inf')):
                reaction.upper_bound /= abs(float(scaling_factor))
                
            # Track compression info
            main_reaction.subset_rxns.append(r_idx)
            main_reaction.subset_stoich.append(scaling_factor)
        
        # Merge additional reactions into the main reaction
        for r_idx in rxn_indices[1:]:
            reaction = model.reactions[r_idx]
            
            # Update main reaction name
            if len(main_reaction.id) + len(reaction.id) < 220 and not main_reaction.id.endswith('...'):
                main_reaction.id += '*' + reaction.id
            elif not main_reaction.id.endswith('...'):
                main_reaction.id += '...'
            
            # Merge stoichiometry
            main_reaction += reaction
            
            # Update bounds (keep most restrictive)
            if reaction.lower_bound > main_reaction.lower_bound:
                main_reaction.lower_bound = reaction.lower_bound
            if reaction.upper_bound < main_reaction.upper_bound:
                main_reaction.upper_bound = reaction.upper_bound
                
            # Mark for deletion
            del_rxns[r_idx] = True
    
    # Remove reactions that were merged (in reverse order to maintain indices)
    del_indices = np.where(del_rxns)[0]
    for i in range(len(del_indices) - 1, -1, -1):
        model.reactions[del_indices[i]].remove_from_model(remove_orphans=True)
    
    # Add compression metadata to remaining reactions
    for reaction in model.reactions:
        reaction.notes['compression_applied'] = True
        stats = compression_record.stats
        reaction.notes['compression_stats'] = {
            'iterations': stats.get_iteration_count(),
            'total_reaction_compressions': stats.get_total_reaction_compressions(),
            'total_metabolite_compressions': stats.get_total_metabolite_compressions()
        }


def _matrix_to_numpy(matrix) -> np.ndarray:
    """Convert BigIntegerRationalMatrix to numpy array."""
    rows = matrix.get_row_count()
    cols = matrix.get_column_count()
    result = np.zeros((rows, cols))
    
    for i in range(rows):
        for j in range(cols):
            val = matrix.get_big_fraction_value_at(i, j)
            result[i, j] = float(val)
    
    return result


def _convert_coefficients_to_rational(model):
    """Convert stoichiometric coefficients to exact rationals."""
    for reaction in model.reactions:
        for metabolite, coeff in list(reaction.metabolites.items()):
            if isinstance(coeff, float):
                # Convert to rational
                rational_coeff = Fraction(coeff).limit_denominator()
                reaction.add_metabolites({metabolite: rational_coeff - coeff})


def _remove_conservation_relations(model):
    """Remove conservation relations from the model using exact rational arithmetic."""
    from cobra.util.array import create_stoichiometric_matrix
    from ..math.default_bigint_rational_matrix import DefaultBigIntegerRationalMatrix
    from ..math.gauss import Gauss
    from ..math.big_fraction import BigFraction
    import numpy as np
    from sympy import Rational, nsimplify
    
    # Get stoichiometric matrix as sparse matrix
    stoich_mat = create_stoichiometric_matrix(model, array_type='lil')
    
    # Find basic metabolites using transposed matrix (reactions × metabolites)
    basic_metabolites = _basic_columns_rat(stoich_mat.transpose().toarray(), tolerance=0)
    
    # Identify dependent metabolites (those not in basic set)
    all_metabolite_indices = set(range(len(model.metabolites)))
    basic_metabolite_indices = set(basic_metabolites)
    dependent_metabolite_indices = all_metabolite_indices - basic_metabolite_indices
    
    # Get dependent metabolite IDs
    dependent_metabolites = [model.metabolites[i].id for i in dependent_metabolite_indices]
    
    # Remove dependent metabolites from model
    for metabolite_id in dependent_metabolites:
        metabolite = model.metabolites.get_by_id(metabolite_id)
        metabolite.remove_from_model()


def _basic_columns_rat(matrix, tolerance=0):
    """Find basic columns using exact rational Gaussian elimination - ported from compression_python_port"""
    from ..math.default_bigint_rational_matrix import DefaultBigIntegerRationalMatrix
    from ..math.gauss import Gauss
    from ..math.big_fraction import BigFraction
    from sympy import Rational, nsimplify
    import numpy as np
    
    # Convert numpy array to rational matrix
    rows, cols = matrix.shape
    rational_matrix = DefaultBigIntegerRationalMatrix(rows, cols)
    
    # Convert each element to exact rational
    for i in range(rows):
        for j in range(cols):
            value = matrix[i, j]
            
            # Convert to exact rational
            if value == 0:
                continue  # Default is zero
            elif isinstance(value, (int, float)):
                if isinstance(value, int):
                    rational_value = Rational(value)
                else:
                    # Convert float to exact rational
                    rational_value = nsimplify(value, rational=True, rational_conversion='base10')
                
                # Convert to BigFraction
                big_fraction = BigFraction(rational_value.p, rational_value.q)
                rational_matrix.set_value_at(i, j, big_fraction)
            else:
                # Already rational, convert to BigFraction
                if hasattr(value, 'p') and hasattr(value, 'q'):
                    big_fraction = BigFraction(value.p, value.q)
                    rational_matrix.set_value_at(i, j, big_fraction)
                else:
                    # Fallback - convert via sympy
                    rational_value = nsimplify(float(value), rational=True, rational_conversion='base10')
                    big_fraction = BigFraction(rational_value.p, rational_value.q)
                    rational_matrix.set_value_at(i, j, big_fraction)
    
    # Prepare arrays for Gaussian elimination
    row_count = rational_matrix.get_row_count()
    col_count = rational_matrix.get_column_count()
    
    row_map = [0] * row_count  # Row permutation (we don't use this)
    col_map = list(range(col_count))  # Column permutation
    
    # Perform row echelon form using exact rational arithmetic
    gauss = Gauss.get_rational_instance()
    # Use the private method directly since we need the column permutation
    rank, updated_col_map = gauss._row_echelon(rational_matrix, False, row_map, col_map)
    
    # Return indices of basic columns (first 'rank' columns after permutation)
    return updated_col_map[:rank]