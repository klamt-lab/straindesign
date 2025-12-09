#!/usr/bin/env python3
"""
Reaction interface - Python port of Java ch.javasoft.metabolic.Reaction

A reaction in a metabolic network consumes educt metabolites and generates
product metabolites. A reaction is typically associated with a column of the
stoichiometric matrix.

Java source: efmtool_source/ch/javasoft/metabolic/Reaction.java
"""

from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

from .annotateable import Annotateable

if TYPE_CHECKING:
    from .metabolite import Metabolite
    from .metabolite_ratio import MetaboliteRatio
    from .reaction_constraints import ReactionConstraints
    from .metabolic_network_visitor import MetabolicNetworkVisitor


class Reaction(Annotateable):
    """
    A reaction in a metabolic network consumes educt metabolites and generates
    product metabolites. A reaction is typically associated with a column of the
    stoichiometric matrix.
    """
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Returns the name of the reaction, unique throughout the metabolic network
        
        Returns:
            str: The unique reaction name
        """
        pass
    
    @abstractmethod
    def get_full_name(self) -> str:
        """
        Returns the full name of the reaction, e.g. with additional compartment
        information
        
        Returns:
            str: The full reaction name
        """
        pass
    
    @abstractmethod
    def get_educt_ratios(self) -> List['MetaboliteRatio']:
        """
        Returns all ratios associated with educt metabolites, that is, 
        the ratio values are negative
        
        Returns:
            List[MetaboliteRatio]: List of educt (reactant) ratios
        """
        pass
    
    @abstractmethod
    def get_product_ratios(self) -> List['MetaboliteRatio']:
        """
        Returns all ratios associated with product metabolites, that is, 
        the ratio values are positive
        
        Returns:
            List[MetaboliteRatio]: List of product ratios
        """
        pass
    
    @abstractmethod
    def get_metabolite_ratios(self) -> List['MetaboliteRatio']:
        """
        Returns all ratios associated with educt and product metabolites;
        ratios are negative for educts and positive for products
        
        Returns:
            List[MetaboliteRatio]: List of all metabolite ratios (educts + products)
        """
        pass
    
    @abstractmethod
    def get_constraints(self) -> 'ReactionConstraints':
        """
        Returns the reaction constraints, which also defines the reversibility 
        of the reaction
        
        Returns:
            ReactionConstraints: The reaction constraints
        """
        pass
    
    @abstractmethod
    def get_ratio_value_for_metabolite(self, metabolite: 'Metabolite') -> float:
        """
        Get the stoichiometric coefficient for a specific metabolite
        
        Args:
            metabolite: The metabolite to query
            
        Returns:
            float: == 0 if the metabolite doesn't participate
                  > 0 if it participates and is produced
                  < 0 if it participates and is consumed
        """
        pass
    
    @abstractmethod
    def is_metabolite_participating(self, metabolite: 'Metabolite') -> bool:
        """
        Check if a metabolite participates in this reaction
        
        Args:
            metabolite: The metabolite to check
            
        Returns:
            bool: True if get_ratio_value_for_metabolite() returns != 0
        """
        pass
    
    @abstractmethod
    def is_metabolite_produced(self, metabolite: 'Metabolite') -> bool:
        """
        Check if a metabolite is produced by this reaction
        
        Args:
            metabolite: The metabolite to check
            
        Returns:
            bool: True if get_ratio_value_for_metabolite() returns > 0
                 (metabolite is a product). Reversibility is not considered.
        """
        pass
    
    @abstractmethod
    def is_metabolite_consumed(self, metabolite: 'Metabolite') -> bool:
        """
        Check if a metabolite is consumed by this reaction
        
        Args:
            metabolite: The metabolite to check
            
        Returns:
            bool: True if get_ratio_value_for_metabolite() returns < 0
                 (metabolite is an educt). Reversibility is not considered.
        """
        pass
    
    @abstractmethod
    def is_external(self) -> bool:
        """
        Check if this reaction is either an uptake or an extract reaction
        
        Returns:
            bool: True if this reaction has either no educts or no products
        """
        pass
    
    @abstractmethod
    def is_uptake(self) -> bool:
        """
        Check if this reaction has no educts (uptake reaction)
        
        Returns:
            bool: True if this reaction has no educts
        """
        pass
    
    @abstractmethod
    def is_extract(self) -> bool:
        """
        Check if this reaction has no products (extract reaction)
        
        Returns:
            bool: True if this reaction has no products
        """
        pass
    
    @abstractmethod
    def has_integer_ratios(self) -> bool:
        """
        Check if all stoichiometric coefficients are integers
        
        Returns:
            bool: True if all ratios are integers
        """
        pass
    
    @abstractmethod
    def accept(self, visitor: 'MetabolicNetworkVisitor') -> None:
        """
        Accept method for the visitor, implementations usually delegate back to 
        visitor.visit_reaction(self)
        
        Args:
            visitor: The visitor to callback
        """
        pass