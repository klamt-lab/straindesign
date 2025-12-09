#!/usr/bin/env python3
"""
MetabolicNetworkVisitor interface - Python port of Java ch.javasoft.metabolic.MetabolicNetworkVisitor

A visitor for the components of a MetabolicNetwork. For most 
implementors, it might be useful to inherit the default implementation 
and override the desired methods.

Java source: efmtool_source/ch/javasoft/metabolic/MetabolicNetworkVisitor.java
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

# Forward references to avoid circular imports
if TYPE_CHECKING:
    from .metabolite import Metabolite
    from .metabolic_network import MetabolicNetwork
    from .reaction import Reaction
    from .metabolite_ratio import MetaboliteRatio
    from .reaction_constraints import ReactionConstraints


class MetabolicNetworkVisitor(ABC):
    """
    A visitor for the components of a MetabolicNetwork.
    """
    
    @abstractmethod
    def visit_metabolic_network(self, net: 'MetabolicNetwork') -> None:
        """Visit a metabolic network"""
        pass
    
    @abstractmethod
    def visit_metabolite(self, metabolite: 'Metabolite') -> None:
        """Visit a metabolite"""
        pass
    
    @abstractmethod
    def visit_reaction(self, reaction: 'Reaction') -> None:
        """Visit a reaction"""
        pass
    
    @abstractmethod
    def visit_metabolite_ratio(self, ratio: 'MetaboliteRatio') -> None:
        """Visit a metabolite ratio"""
        pass
    
    @abstractmethod
    def visit_reaction_constraints(self, constraints: 'ReactionConstraints') -> None:
        """Visit reaction constraints"""
        pass