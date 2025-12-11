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
    
    # Modify the model in-place based on compression results
    # This returns the reaction_map with keys that match the actual model reaction IDs
    reaction_map = _apply_compression_to_model(model, compression_record, reaction_names)

    # Build metabolite map from transformation matrices
    _, metabolite_map = _build_transformation_maps(
        compression_record, reaction_names, metabolite_names
    )
    
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


def _apply_compression_to_model(model, compression_record, reaction_names):
    """Apply compression results to the COBRA model in-place.

    Optimized version that avoids O(n²) complexity when merging many reactions.
    Instead of incremental merging, computes final stoichiometry directly.

    Returns:
        reaction_map: Dictionary mapping compressed reaction IDs to original reactions with coefficients
    """
    import numpy as np
    from fractions import Fraction
    from ..math import BigFraction

    # Convert post matrix to numpy for processing
    post_matrix_np = _matrix_to_numpy(compression_record.post)
    post_matrix = compression_record.post  # Keep original for exact coefficients

    # Build reaction_map as we process
    reaction_map = {}

    # Safety check: if post matrix has no columns (no compressed reactions),
    # this means all reactions were eliminated - this is likely an error
    if post_matrix_np.shape[1] == 0:
        print(f"WARNING: All reactions eliminated during compression! Post matrix shape: {post_matrix_np.shape}")
        print("This suggests the model may be infeasible or the compression is too aggressive.")
        for reaction in model.reactions:
            reaction.notes['compression_applied'] = True
            reaction.notes['compression_warning'] = 'All reactions eliminated - compression not applied'
        return {}

    # Find reactions to delete (those with no contribution to any compressed reaction)
    del_rxns = np.logical_not(np.any(post_matrix_np, axis=1))

    # Build metabolite index map for fast lookups
    met_to_idx = {m: i for i, m in enumerate(model.metabolites)}

    # Process each compressed reaction (column in post matrix)
    for j in range(post_matrix_np.shape[1]):
        rxn_indices = post_matrix_np[:, j].nonzero()[0]
        if len(rxn_indices) == 0:
            continue

        # Main reaction (first in the group)
        main_idx = rxn_indices[0]
        main_reaction = model.reactions[main_idx]

        # Track compression info
        main_reaction.subset_rxns = list(rxn_indices)
        main_reaction.subset_stoich = [Fraction(float(post_matrix_np[r_idx, j])).limit_denominator()
                                        for r_idx in rxn_indices]

        if len(rxn_indices) == 1:
            # Single reaction - just scale if needed
            scaling = float(post_matrix_np[main_idx, j])
            if scaling != 1.0:
                for met, coeff in list(main_reaction.metabolites.items()):
                    main_reaction.add_metabolites({met: coeff * (scaling - 1)})
            # Build reaction_map entry for this single reaction
            orig_name = reaction_names[main_idx]
            coeff = post_matrix.get_big_fraction_value_at(main_idx, j)
            reaction_map[orig_name] = {orig_name: coeff}
            continue

        # Multiple reactions to merge - compute final stoichiometry directly (O(n))
        # This avoids COBRApy's slow incremental merge which is O(n²)
        merged_stoich = {}
        merged_lb = -float('inf')
        merged_ub = float('inf')
        name_parts = []
        contributing_reactions = {}  # For reaction_map

        for r_idx in rxn_indices:
            reaction = model.reactions[r_idx]
            scaling = float(post_matrix_np[r_idx, j])

            # Store exact coefficient for reaction_map
            orig_name = reaction_names[r_idx]
            exact_coeff = post_matrix.get_big_fraction_value_at(r_idx, j)
            contributing_reactions[orig_name] = exact_coeff

            # Accumulate scaled stoichiometry
            for met, met_coeff in reaction.metabolites.items():
                met_id = met.id
                if met_id in merged_stoich:
                    merged_stoich[met_id] += met_coeff * scaling
                else:
                    merged_stoich[met_id] = met_coeff * scaling

            # Track most restrictive bounds (scaled)
            # If v_merged = v_original / scaling, then:
            #   lb_original <= v_original <= ub_original
            #   lb_original / scaling <= v_merged <= ub_original / scaling (if scaling > 0)
            #   ub_original / scaling <= v_merged <= lb_original / scaling (if scaling < 0)
            if scaling > 0:
                # Positive scaling: divide bounds by scale
                lb_scaled = reaction.lower_bound / scaling
                ub_scaled = reaction.upper_bound / scaling
                merged_lb = max(merged_lb, lb_scaled)
                merged_ub = min(merged_ub, ub_scaled)
            elif scaling < 0:
                # Negative scaling: divide and swap bounds
                lb_scaled = reaction.upper_bound / scaling  # ub becomes lb
                ub_scaled = reaction.lower_bound / scaling  # lb becomes ub
                merged_lb = max(merged_lb, lb_scaled)
                merged_ub = min(merged_ub, ub_scaled)

            # Build name - use '::' separator to match efmtool convention
            name_parts.append(reaction.id)

            # Mark for deletion (except main)
            if r_idx != main_idx:
                del_rxns[r_idx] = True

        # Update main reaction with merged stoichiometry
        # First clear existing stoichiometry
        main_reaction.subtract_metabolites(main_reaction.metabolites.copy())

        # Add merged stoichiometry
        new_metabolites = {}
        for met_id, met_coeff in merged_stoich.items():
            if abs(met_coeff) > 1e-12:  # Skip near-zero coefficients
                met = model.metabolites.get_by_id(met_id)
                new_metabolites[met] = met_coeff
        main_reaction.add_metabolites(new_metabolites)

        # Update bounds
        if merged_lb > -float('inf'):
            main_reaction.lower_bound = merged_lb
        if merged_ub < float('inf'):
            main_reaction.upper_bound = merged_ub

        # Update name - use '::' separator to match efmtool convention
        # Truncate to 240 chars max (Gurobi limit is 255, COBRApy adds _reverse suffix)
        compressed_name = '::'.join(name_parts)
        if len(compressed_name) > 240:
            # Truncate and add ellipsis
            compressed_name = compressed_name[:237] + '...'
        main_reaction.id = compressed_name

        # Add to reaction_map with the actual name that will be used in the model
        reaction_map[compressed_name] = contributing_reactions

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

    return reaction_map


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
    """
    Find basic columns using exact rational Gaussian elimination.

    Uses FLINT when available for fast O(n³) operations, otherwise falls back
    to pure Python implementation.
    """
    from ..math.gauss import Gauss
    from ..math.big_fraction import BigFraction
    from fractions import Fraction

    # Check if FLINT is available for fast path
    try:
        from flint import fmpq_mat, fmpq
        FLINT_AVAILABLE = True
    except ImportError:
        FLINT_AVAILABLE = False

    rows, cols = matrix.shape

    if FLINT_AVAILABLE:
        # Fast path: use FLINT directly
        flint_mat = fmpq_mat(rows, cols)
        for i in range(rows):
            for j in range(cols):
                value = matrix[i, j]
                if value != 0:
                    # Convert to exact rational using Python's Fraction
                    if isinstance(value, float):
                        frac = Fraction(value).limit_denominator()
                        flint_mat[i, j] = fmpq(frac.numerator, frac.denominator)
                    else:
                        flint_mat[i, j] = fmpq(int(value))

        # Use FLINT's native RREF to find pivot columns
        rref, rank = flint_mat.rref()

        # Find pivot columns
        pivot_cols = []
        for r in range(min(rows, rank)):
            for c in range(cols):
                if rref[r, c] != fmpq(0):
                    pivot_cols.append(c)
                    break

        return pivot_cols

    else:
        # Slow path: use pure Python with BigFraction
        from ..math.default_bigint_rational_matrix import DefaultBigIntegerRationalMatrix

        rational_matrix = DefaultBigIntegerRationalMatrix(rows, cols)

        for i in range(rows):
            for j in range(cols):
                value = matrix[i, j]
                if value != 0:
                    if isinstance(value, float):
                        frac = Fraction(value).limit_denominator()
                        big_fraction = BigFraction(frac.numerator, frac.denominator)
                    else:
                        big_fraction = BigFraction(int(value))
                    rational_matrix.set_value_at(i, j, big_fraction)

        # Use basic_columns method if available, otherwise fallback
        gauss = Gauss.get_rational_instance()
        if hasattr(gauss, 'basic_columns'):
            return gauss.basic_columns(rational_matrix)
        else:
            # Fallback for pure Python Gauss
            row_map = [0] * rows
            col_map = list(range(cols))
            rank, updated_col_map = gauss._row_echelon(rational_matrix, False, row_map, col_map)
            return updated_col_map[:rank]