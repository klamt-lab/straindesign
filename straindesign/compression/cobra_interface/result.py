"""
CompressionResult class - container for compression results.

Implements the result object as specified in API_SPECIFICATION.md
"""

import numpy as np
from typing import Dict, List, Union, Any
from fractions import Fraction


class CompressionResult:
    """
    Result of model compression containing the compressed model and transformation utilities.
    
    This class provides access to all compression results and utilities as specified
    in API_SPECIFICATION.md.
    """
    
    def __init__(self, compressed_model, compression_converter, pre_matrix, post_matrix,
                 reaction_map, metabolite_map, statistics, methods_used,
                 original_reaction_names, original_metabolite_names, flipped_reactions):
        """Initialize compression result."""
        # Core results
        self.compressed_model = compressed_model
        self.compression_converter = compression_converter
        
        # Transformation matrices (numpy arrays)
        self.pre_matrix = pre_matrix
        self.post_matrix = post_matrix
        
        # Detailed information
        self.reaction_map = reaction_map
        self.metabolite_map = metabolite_map
        self.statistics = statistics
        self.methods_used = methods_used
        
        # Original state
        self.original_reaction_names = original_reaction_names
        self.original_metabolite_names = original_metabolite_names
        self.flipped_reactions = flipped_reactions
    
    @property
    def compression_ratio(self) -> float:
        """Ratio of compressed to original reactions."""
        return len(self.compressed_model.reactions) / len(self.original_reaction_names)
    
    @property
    def reactions_removed(self) -> int:
        """Number of reactions removed."""
        return len(self.original_reaction_names) - len(self.compressed_model.reactions)
        
    @property
    def metabolites_removed(self) -> int:
        """Number of metabolites removed."""
        return len(self.original_metabolite_names) - len(self.compressed_model.metabolites)
    
    def summary(self) -> str:
        """Human-readable compression summary."""
        return f"Compressed {len(self.original_reaction_names)} → {len(self.compressed_model.reactions)} reactions ({self.compression_ratio:.1%})"
    
    def get_compression_info(self) -> Dict[str, Any]:
        """Get comprehensive compression information."""
        info = {
            'original_reactions': len(self.original_reaction_names),
            'compressed_reactions': len(self.compressed_model.reactions),
            'original_metabolites': len(self.original_metabolite_names),
            'compressed_metabolites': len(self.compressed_model.metabolites),
            'reactions_removed': self.reactions_removed,
            'metabolites_removed': self.metabolites_removed,
            'flipped_reactions': len(self.flipped_reactions),
            'compression_ratio': self.compression_ratio,
            'methods_used': [method.name for method in self.methods_used]
        }
        
        # Add statistics if available
        if self.statistics:
            if hasattr(self.statistics, 'get_iteration_count'):
                info['iterations'] = self.statistics.get_iteration_count()
            if hasattr(self.statistics, 'get_dead_end_metabolite_reactions_count'):
                info['dead_end_reactions'] = self.statistics.get_dead_end_metabolite_reactions_count()
            if hasattr(self.statistics, 'get_unique_flow_reactions_count'):
                info['unique_flow_reactions'] = self.statistics.get_unique_flow_reactions_count()
            if hasattr(self.statistics, 'get_coupled_reactions_count'):
                info['coupled_reactions'] = self.statistics.get_coupled_reactions_count()
            if hasattr(self.statistics, 'get_zero_flux_reactions_count'):
                info['zero_flux_reactions'] = self.statistics.get_zero_flux_reactions_count()
        
        return info