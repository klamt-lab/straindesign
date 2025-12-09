#!/usr/bin/env python3
"""
MetabolicNetwork interface - Python port of Java ch.javasoft.metabolic.MetabolicNetwork

The MetabolicNetwork is a collection of reactions.
Some convenience methods allow better searching and iterating over 
metabolites (the educts and products of reactions) and reactions.

Java source: efmtool_source/ch/javasoft/metabolic/MetabolicNetwork.java
"""

from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

from .annotateable import Annotateable

if TYPE_CHECKING:
    from .metabolite import Metabolite
    from .reaction import Reaction
    from .metabolic_network_visitor import MetabolicNetworkVisitor


class MetabolicNetwork(Annotateable):
    """
    The MetabolicNetwork is a collection of reactions.
    Some convenience methods allow better searching and iterating over 
    metabolites (the educts and products of reactions) and reactions.
    """
    
    @abstractmethod
    def get_metabolites(self) -> List['Metabolite']:
        """
        Returns the list of all metabolites
        
        Returns:
            List[Metabolite]: All metabolites in the network
        """
        pass
    
    @abstractmethod
    def get_metabolite(self, name: str) -> 'Metabolite':
        """
        Returns the desired metabolite, or raises ValueError
        if no metabolite exists with the given name
        
        Args:
            name: The metabolite name to find
            
        Returns:
            Metabolite: The metabolite with the given name
            
        Raises:
            ValueError: If no metabolite exists with the given name
        """
        pass
    
    @abstractmethod
    def get_metabolite_index(self, name: str) -> int:
        """
        Returns the index of the given metabolite, or -1 if no metabolite exists 
        with the given name
        
        Args:
            name: The metabolite name to find
            
        Returns:
            int: The metabolite index, or -1 if not found
        """
        pass
    
    @abstractmethod
    def get_reactions(self, metabolite: 'Metabolite' = None) -> List['Reaction']:
        """
        Returns reactions that involve the given metabolite, or all reactions if None
        
        Args:
            metabolite: Optional metabolite to filter by
            
        Returns:
            List[Reaction]: Reactions involving the metabolite (or all reactions)
        """
        pass
    
    @abstractmethod
    def get_reaction(self, name: str) -> 'Reaction':
        """
        Returns the desired reaction, or raises ValueError
        if no reaction exists with the given name
        
        Args:
            name: The reaction name to find
            
        Returns:
            Reaction: The reaction with the given name
            
        Raises:
            ValueError: If no reaction exists with the given name
        """
        pass
    
    @abstractmethod
    def get_reaction_index(self, name: str) -> int:
        """
        Returns the index of the given reaction, or -1 if no reaction exists 
        with the given name
        
        Args:
            name: The reaction name to find
            
        Returns:
            int: The reaction index, or -1 if not found
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