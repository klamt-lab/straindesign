#!/usr/bin/env python3
"""
ReactionConstraints interface - Python port of Java ch.javasoft.metabolic.ReactionConstraints

The ReactionConstraints define the directionality of reactions
or more precise upper and lower bounds for reaction fluxes.

Java source: efmtool_source/ch/javasoft/metabolic/ReactionConstraints.java
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .annotateable import Annotateable

if TYPE_CHECKING:
    from .metabolic_network_visitor import MetabolicNetworkVisitor


class ReactionConstraints(Annotateable):
    """
    The ReactionConstraints define the directionality of reactions
    or more precise upper and lower bounds for reaction fluxes.
    """
    
    @abstractmethod
    def is_reversible(self) -> bool:
        """
        Check if the reaction is reversible
        
        Returns:
            bool: True if the reaction can proceed in both directions
        """
        pass
    
    @abstractmethod
    def get_upper_bound(self) -> float:
        """
        Get the upper bound for the reaction flux
        
        Returns:
            float: The maximum flux value allowed
        """
        pass
    
    @abstractmethod
    def get_lower_bound(self) -> float:
        """
        Get the lower bound for the reaction flux
        
        Returns:
            float: The minimum flux value allowed
        """
        pass
    
    @abstractmethod
    def accept(self, visitor: 'MetabolicNetworkVisitor') -> None:
        """
        Accept method for the visitor
        
        Args:
            visitor: The visitor to callback
        """
        pass