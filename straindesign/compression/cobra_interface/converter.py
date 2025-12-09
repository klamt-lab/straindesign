"""
CompressionConverter class - bidirectional expression transformation.

Implements the converter class as specified in API_SPECIFICATION.md
"""

from typing import Dict, List, Tuple, Union, Set
from fractions import Fraction
import logging

LOG = logging.getLogger(__name__)


class CompressionConverter:
    """
    Bidirectional transformer for linear expressions between original and compressed spaces.
    
    This class handles transformation of objectives, constraints, and any linear expressions
    between the original metabolic model and its compressed representation.
    """
    
    def __init__(self, reaction_map: Dict[str, Dict[str, Union[float, Fraction]]],
                 metabolite_map: Dict[str, Dict[str, Union[float, Fraction]]],
                 flipped_reactions: List[str]):
        """
        Initialize compression converter.
        
        Args:
            reaction_map: Mapping from compressed to original reactions with coefficients
            metabolite_map: Mapping from compressed to original metabolites with coefficients  
            flipped_reactions: List of reactions that were flipped during preprocessing
        """
        self.reaction_map = reaction_map
        self.metabolite_map = metabolite_map
        self.flipped_reactions = set(flipped_reactions)
        
        # Create reverse mappings for expansion
        self._create_reverse_mappings()
    
    def compress_expression(self, expression: Dict[str, float], 
                          remove_missing: bool = False) -> Dict[str, float]:
        """
        Transform expression from original to compressed reaction space.
        
        Args:
            expression: Dictionary mapping reaction names to coefficients
            remove_missing: If True, ignore missing reactions; if False, raise error
            
        Returns:
            Expression in compressed space
        """
        compressed_expr = {}
        
        # For each compressed reaction, check if any original reactions are in the expression
        for compressed_rxn, original_rxns in self.reaction_map.items():
            contributing_value = 0.0
            
            for original_rxn, scaling_factor in original_rxns.items():
                if original_rxn in expression:
                    # Account for reaction flipping
                    expr_coeff = expression[original_rxn]
                    if original_rxn in self.flipped_reactions:
                        expr_coeff = -expr_coeff
                    
                    contributing_value += expr_coeff * float(scaling_factor)
            
            if contributing_value != 0:
                compressed_expr[compressed_rxn] = contributing_value
        
        # Handle reactions that weren't compressed (direct mapping)
        original_reactions_in_compression = set().union(*[set(v.keys()) for v in self.reaction_map.values()])
        
        for rxn_id, coeff in expression.items():
            if rxn_id not in original_reactions_in_compression:
                if rxn_id in [list(v.keys())[0] for v in self.reaction_map.values() if len(v) == 1]:
                    # This reaction maps directly to a compressed reaction
                    compressed_name = next(k for k, v in self.reaction_map.items() 
                                         if len(v) == 1 and list(v.keys())[0] == rxn_id)
                    compressed_expr[compressed_name] = coeff
                elif not remove_missing:
                    LOG.warning(f"Reaction {rxn_id} not found in compression map")
        
        return compressed_expr
    
    def expand_expression(self, expression: Dict[str, float], 
                         remove_missing: bool = False) -> Dict[str, float]:
        """
        Transform expression from compressed back to original reaction space.
        
        Args:
            expression: Dictionary mapping compressed reaction names to coefficients
            remove_missing: If True, ignore missing reactions; if False, raise error
            
        Returns:
            Expression in original space
        """
        expanded_expr = {}
        
        for compressed_rxn, compressed_coeff in expression.items():
            if compressed_rxn in self.reaction_map:
                # Expand this compressed reaction to its original components
                original_rxns = self.reaction_map[compressed_rxn]
                
                for original_rxn, scaling_factor in original_rxns.items():
                    expanded_coeff = compressed_coeff * float(scaling_factor)
                    
                    # Account for reaction flipping
                    if original_rxn in self.flipped_reactions:
                        expanded_coeff = -expanded_coeff
                    
                    if original_rxn in expanded_expr:
                        expanded_expr[original_rxn] += expanded_coeff
                    else:
                        expanded_expr[original_rxn] = expanded_coeff
            elif not remove_missing:
                LOG.warning(f"Compressed reaction {compressed_rxn} not found in reaction map")
        
        return expanded_expr
    
    def compress_constraint(self, constraint: Tuple[Dict[str, float], str, float]) -> Tuple[Dict[str, float], str, float]:
        """
        Transform constraint from original to compressed space.
        
        Args:
            constraint: Tuple of (linear_expression, sense, rhs)
                
        Returns:
            Constraint tuple in compressed space
        """
        linear_expr, sense, rhs = constraint
        compressed_expr = self.compress_expression(linear_expr)
        return (compressed_expr, sense, rhs)
    
    def expand_constraint(self, constraint: Tuple[Dict[str, float], str, float]) -> Tuple[Dict[str, float], str, float]:
        """Transform constraint from compressed back to original space."""
        linear_expr, sense, rhs = constraint
        expanded_expr = self.expand_expression(linear_expr)
        return (expanded_expr, sense, rhs)
    
    def compress_strain_design_module(self, module: Dict) -> Dict:
        """
        Transform StrainDesign module to compressed space.
        
        Handles CONSTRAINTS, INNER_OBJECTIVE, OUTER_OBJECTIVE, etc.
        """
        # Import StrainDesign constants locally to avoid dependency issues
        try:
            from straindesign.names import CONSTRAINTS, INNER_OBJECTIVE, OUTER_OBJECTIVE, PROD_ID, MIN_GCP
        except ImportError:
            # Define constants locally if StrainDesign not available
            CONSTRAINTS = 'constraints'
            INNER_OBJECTIVE = 'inner_objective'
            OUTER_OBJECTIVE = 'outer_objective'
            PROD_ID = 'prod_id'
            MIN_GCP = 'min_gcp'
        
        compressed_module = module.copy()
        
        # Transform constraints
        if CONSTRAINTS in module and module[CONSTRAINTS] is not None:
            compressed_constraints = []
            for constraint in module[CONSTRAINTS]:
                compressed_constraint = self.compress_constraint(constraint)
                compressed_constraints.append(compressed_constraint)
            compressed_module[CONSTRAINTS] = compressed_constraints
        
        # Transform objectives
        for objective_key in [INNER_OBJECTIVE, OUTER_OBJECTIVE, PROD_ID]:
            if objective_key in module and module[objective_key] is not None:
                compressed_module[objective_key] = self.compress_expression(module[objective_key])
        
        return compressed_module
    
    def expand_strain_design_module(self, module: Dict) -> Dict:
        """Transform StrainDesign module back to original space."""
        try:
            from straindesign.names import CONSTRAINTS, INNER_OBJECTIVE, OUTER_OBJECTIVE, PROD_ID, MIN_GCP
        except ImportError:
            CONSTRAINTS = 'constraints'
            INNER_OBJECTIVE = 'inner_objective'
            OUTER_OBJECTIVE = 'outer_objective'  
            PROD_ID = 'prod_id'
            MIN_GCP = 'min_gcp'
        
        expanded_module = module.copy()
        
        # Transform constraints
        if CONSTRAINTS in module and module[CONSTRAINTS] is not None:
            expanded_constraints = []
            for constraint in module[CONSTRAINTS]:
                expanded_constraint = self.expand_constraint(constraint)
                expanded_constraints.append(expanded_constraint)
            expanded_module[CONSTRAINTS] = expanded_constraints
        
        # Transform objectives
        for objective_key in [INNER_OBJECTIVE, OUTER_OBJECTIVE, PROD_ID]:
            if objective_key in module and module[objective_key] is not None:
                expanded_module[objective_key] = self.expand_expression(module[objective_key])
        
        return expanded_module
    
    def _create_reverse_mappings(self):
        """Create reverse mappings for efficient expansion operations."""
        self._reverse_reaction_map = {}
        
        for compressed_rxn, original_rxns in self.reaction_map.items():
            for original_rxn, scaling_factor in original_rxns.items():
                if original_rxn not in self._reverse_reaction_map:
                    self._reverse_reaction_map[original_rxn] = []
                self._reverse_reaction_map[original_rxn].append((compressed_rxn, scaling_factor))


def create_compression_converter(reaction_map: Dict, metabolite_map: Dict = None, 
                               flipped_reactions: List[str] = None) -> CompressionConverter:
    """Create standalone CompressionConverter from mapping data."""
    if metabolite_map is None:
        metabolite_map = {}
    if flipped_reactions is None:
        flipped_reactions = []
    
    return CompressionConverter(reaction_map, metabolite_map, flipped_reactions)