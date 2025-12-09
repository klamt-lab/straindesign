#!/usr/bin/env python3
"""
MetaboliteRatio interface - Python port of Java ch.javasoft.metabolic.MetaboliteRatio

A metabolite ratio is a stoichiometric coefficient, that is, it is associated
with a reaction and a metabolite. It represents a single cell value of the 
stoichiometric matrix.

Java source: efmtool_source/ch/javasoft/metabolic/MetaboliteRatio.java
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union

from .annotateable import Annotateable

if TYPE_CHECKING:
    from .metabolite import Metabolite
    from .metabolic_network_visitor import MetabolicNetworkVisitor


class MetaboliteRatio(Annotateable):
    """
    A metabolite ratio is a stoichiometric coefficient, that is, it is associated
    with a reaction and a metabolite. It represents a single cell value of the 
    stoichiometric matrix.
    """
    
    @abstractmethod
    def get_metabolite(self) -> 'Metabolite':
        """
        Returns the metabolite associated with this ratio
        
        Returns:
            Metabolite: The associated metabolite
        """
        pass
    
    @abstractmethod
    def get_ratio(self) -> float:
        """
        Returns the ratio as double value
        
        Returns:
            float: The stoichiometric coefficient as float
        """
        pass
    
    @abstractmethod
    def get_number_ratio(self) -> Union[int, float]:
        """
        Returns the ratio as number value
        
        Returns:
            Union[int, float]: The stoichiometric coefficient as number
        """
        pass
    
    @abstractmethod
    def is_educt(self) -> bool:
        """
        Returns true if the ratio value is negative, indicating that the 
        associated metabolite is consumed and therefore is an educt metabolite
        
        Returns:
            bool: True if this metabolite is consumed (negative coefficient)
        """
        pass
    
    @abstractmethod
    def is_integer_ratio(self) -> bool:
        """
        Returns true if the ratio value is an integer
        
        Returns:
            bool: True if the coefficient is an integer value
        """
        pass
    
    @abstractmethod
    def to_string_abs(self) -> str:
        """
        Return the string for the ratio's absolute value
        
        Returns:
            str: String representation of the absolute value
        """
        pass
    
    @abstractmethod
    def accept(self, visitor: 'MetabolicNetworkVisitor') -> None:
        """
        Accept method for the visitor, implementations usually delegate back to 
        visitor.visit_metabolite_ratio(self)
        
        Args:
            visitor: The visitor to callback
        """
        pass
    
    @abstractmethod
    def invert(self) -> 'MetaboliteRatio':
        """
        Returns the inverted ratio with opposite sign.
        
        Returns:
            MetaboliteRatio: New ratio with negated coefficient
        """
        pass