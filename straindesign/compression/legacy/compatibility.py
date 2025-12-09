"""
Legacy compatibility functions.

These functions provide backward compatibility with the old API patterns
while using the new implementation underneath.
"""

from typing import Dict, List, Union, Set, Optional
from fractions import Fraction

from ..cobra_interface import compress_cobra_model
from ..core import CompressionMethod


def compress_model_efmtool(model, methods: Optional[List[CompressionMethod]] = None) -> Dict[str, Dict[str, Union[float, Fraction]]]:
    """
    Legacy compression function that returns reaction mapping dictionary.
    
    This function provides backward compatibility with existing code that expects
    the reaction mapping dictionary format.
    
    Returns:
        Dictionary mapping compressed reaction names to original reaction coefficients
        Format: {'new_reaction': {'old_reaction1': coeff1, 'old_reaction2': coeff2}}
    """
    # Use the new API internally
    result = compress_cobra_model(model, methods=methods, in_place=True)
    
    # Return the reaction map in the old format
    return result.reaction_map


def compress_objective(objective: Dict[str, float], 
                      compression_map: Dict[str, Dict[str, Union[float, Fraction]]]) -> Dict[str, float]:
    """
    Legacy objective compression function.
    
    Transforms an objective function using a compression mapping.
    """
    compressed_objective = {}
    
    for new_reac, old_reac_map in compression_map.items():
        contributing_reactions = [k for k in objective.keys() if k in old_reac_map]
        
        if contributing_reactions:
            value = sum(objective[k] * float(old_reac_map[k]) for k in contributing_reactions)
            if value != 0:
                compressed_objective[new_reac] = value
    
    return compressed_objective


def compress_modules(modules: List[Dict], compression_maps: List[Dict]) -> List[Dict]:
    """
    Legacy module compression function for StrainDesign integration.
    
    Transforms StrainDesign modules using compression mappings.
    """
    try:
        from straindesign.names import CONSTRAINTS, INNER_OBJECTIVE, OUTER_OBJECTIVE, PROD_ID
    except ImportError:
        CONSTRAINTS = 'constraints'
        INNER_OBJECTIVE = 'inner_objective'
        OUTER_OBJECTIVE = 'outer_objective'
        PROD_ID = 'prod_id'
    
    compressed_modules = []
    
    for module in modules:
        compressed_module = module.copy()
        
        # Apply each compression map
        for compression_map in compression_maps:
            reac_map_exp = compression_map.get("reac_map_exp", compression_map)
            
            # Transform constraints
            if CONSTRAINTS in compressed_module and compressed_module[CONSTRAINTS] is not None:
                new_constraints = []
                for constraint in compressed_module[CONSTRAINTS]:
                    linear_expr, sense, rhs = constraint
                    compressed_expr = compress_objective(linear_expr, reac_map_exp)
                    new_constraints.append((compressed_expr, sense, rhs))
                compressed_module[CONSTRAINTS] = new_constraints
            
            # Transform objectives
            for obj_key in [INNER_OBJECTIVE, OUTER_OBJECTIVE, PROD_ID]:
                if obj_key in compressed_module and compressed_module[obj_key] is not None:
                    compressed_module[obj_key] = compress_objective(compressed_module[obj_key], reac_map_exp)
        
        compressed_modules.append(compressed_module)
    
    return compressed_modules


def stoichmat_coeff2rational(model):
    """
    Legacy function to convert stoichiometric coefficients to rationals.
    
    This is now handled automatically by the new API, but provided for compatibility.
    """
    for reaction in model.reactions:
        for metabolite, coeff in list(reaction.metabolites.items()):
            if isinstance(coeff, float):
                rational_coeff = Fraction(coeff).limit_denominator()
                reaction.add_metabolites({metabolite: rational_coeff - coeff})


def remove_conservation_relations(model):
    """
    Legacy function to remove conservation relations.
    
    This is now handled automatically by the new API, but provided for compatibility.
    """
    # Placeholder - full implementation would identify and remove conserved quantities
    pass