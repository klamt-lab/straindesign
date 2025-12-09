#!/usr/bin/env python3
"""
DefaultReactionConstraints implementation - Python port of Java ch.javasoft.metabolic.impl.DefaultReactionConstraints

Standard implementation of ReactionConstraints with upper and lower bounds.

Java source: efmtool_source/ch/javasoft/metabolic/impl/DefaultReactionConstraints.java
"""

import math
from typing import TYPE_CHECKING

from ..reaction_constraints import ReactionConstraints

if TYPE_CHECKING:
    from ..metabolic_network_visitor import MetabolicNetworkVisitor


class DefaultReactionConstraints(ReactionConstraints):
    """
    Standard implementation of ReactionConstraints with upper and lower bounds.
    """
    
    def __init__(self, lower_bound: float, upper_bound: float):
        """
        Create reaction constraints with bounds
        
        Args:
            lower_bound: Minimum flux value allowed
            upper_bound: Maximum flux value allowed
            
        Raises:
            ValueError: If lower_bound > upper_bound or for invalid bound combinations
        """
        if lower_bound > upper_bound:
            raise ValueError(f"lower bound > upper bound: {lower_bound} > {upper_bound}")
        
        # Check for reverse irreversible reactions (not supported)
        if ((lower_bound == float('-inf') or lower_bound < 0.0) and upper_bound == 0.0):
            raise ValueError(f"reverse irreversible reactions not supported: [{lower_bound}, {upper_bound}]")
        
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
    
    def is_reversible(self) -> bool:
        """Check if the reaction is reversible (can proceed in both directions)"""
        return ((self._lower_bound == float('-inf') or self._lower_bound < 0.0) and 
                (self._upper_bound == float('inf') or self._upper_bound > 0.0))
    
    def get_lower_bound(self) -> float:
        """Get the lower bound for the reaction flux"""
        return self._lower_bound
    
    def get_upper_bound(self) -> float:
        """Get the upper bound for the reaction flux"""
        return self._upper_bound
    
    def accept(self, visitor: 'MetabolicNetworkVisitor') -> None:
        """Accept visitor for constraint traversal"""
        visitor.visit_reaction_constraints(self)
    
    def __hash__(self) -> int:
        """Hash based on both bounds (matching Java implementation)"""
        # Mimic Java Double.hashCode() behavior
        lower_bits = hash(self._lower_bound)
        upper_bits = hash(self._upper_bound)
        return lower_bits ^ upper_bits
    
    def __eq__(self, other) -> bool:
        """Equality based on bounds"""
        if not isinstance(other, DefaultReactionConstraints):
            return False
        return (self._lower_bound == other._lower_bound and 
                self._upper_bound == other._upper_bound)
    
    def __str__(self) -> str:
        """String representation of constraints"""
        return f"[{self._lower_bound}, {self._upper_bound}]"


# Standard constraint constants (matching Java)
DEFAULT_REVERSIBLE = DefaultReactionConstraints(float('-inf'), float('inf'))
DEFAULT_IRREVERSIBLE = DefaultReactionConstraints(0.0, float('inf'))
DEFAULT_CONSTRAINTS_NAN = DefaultReactionConstraints(float('nan'), float('nan'))