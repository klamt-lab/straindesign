#!/usr/bin/env python3
"""
Metabolite interface - Python port of Java ch.javasoft.metabolic.Metabolite

A metabolite in a metabolic network stands for a (bio)chemical species or 
substance. Also cofactors might be treated as metabolites. A metabolite is 
typically associated with a row of the stoichiometric matrix.

Java source: efmtool_source/ch/javasoft/metabolic/Metabolite.java
"""

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

from .annotateable import Annotateable

if TYPE_CHECKING:
    from .metabolic_network_visitor import MetabolicNetworkVisitor


class Metabolite(Annotateable):
    """
    A metabolite in a metabolic network stands for a (bio)chemical species or 
    substance. Also cofactors might be treated as metabolites. A metabolite is 
    typically associated with a row of the stoichiometric matrix.
    """
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Unique id or name, must be unique across compartments
        
        Returns:
            str: The unique name/id of this metabolite
        """
        pass
    
    @abstractmethod
    def get_description(self) -> Optional[str]:
        """
        The metabolite descriptive name, not necessarily unique. Might be None
        
        Returns:
            Optional[str]: The description or None if not available
        """
        pass
    
    @abstractmethod
    def accept(self, visitor: 'MetabolicNetworkVisitor') -> None:
        """
        Accept method for the visitor, implementations usually delegate back to 
        visitor.visit_metabolite(self)
        
        Args:
            visitor: The visitor to callback
        """
        pass