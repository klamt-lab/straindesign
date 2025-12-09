#!/usr/bin/env python3
"""
AbstractReaction base class - Python port of Java ch.javasoft.metabolic.impl.AbstractReaction

Base implementation of the Reaction interface providing common functionality.

Java source: efmtool_source/ch/javasoft/metabolic/impl/AbstractReaction.java
"""

from abc import abstractmethod
from typing import List, TYPE_CHECKING

from ..reaction import Reaction

if TYPE_CHECKING:
    from ..metabolite import Metabolite
    from ..metabolite_ratio import MetaboliteRatio
    from ..reaction_constraints import ReactionConstraints
    from ..metabolic_network_visitor import MetabolicNetworkVisitor


class AbstractReaction(Reaction):
    """
    Base implementation of the Reaction interface providing common functionality.
    
    This abstract class implements most of the Reaction methods by delegating
    to the abstract methods that subclasses must implement.
    """
    
    def __init__(self):
        """Create an AbstractReaction"""
        pass
    
    # Abstract methods that subclasses must implement
    @abstractmethod
    def get_name(self) -> str:
        """Get the unique reaction name"""
        pass
    
    @abstractmethod
    def get_metabolite_ratios(self) -> List['MetaboliteRatio']:
        """Get all metabolite ratios (educts + products)"""
        pass
    
    @abstractmethod
    def get_constraints(self) -> 'ReactionConstraints':
        """Get the reaction constraints"""
        pass
    
    # Implemented methods
    def get_full_name(self) -> str:
        """Returns get_name() by default"""
        return self.get_name()
    
    def get_educt_ratios(self) -> List['MetaboliteRatio']:
        """Get all educt (reactant) ratios with negative coefficients"""
        educts = []
        for ratio in self.get_metabolite_ratios():
            if ratio.get_ratio() < 0.0:
                educts.append(ratio)
        return educts
    
    def get_product_ratios(self) -> List['MetaboliteRatio']:
        """Get all product ratios with positive coefficients"""
        products = []
        for ratio in self.get_metabolite_ratios():
            if ratio.get_ratio() > 0.0:
                products.append(ratio)
        return products
    
    def is_metabolite_participating(self, metabolite: 'Metabolite') -> bool:
        """Check if metabolite participates (coefficient != 0)"""
        return self.get_ratio_value_for_metabolite(metabolite) != 0.0
    
    def is_metabolite_consumed(self, metabolite: 'Metabolite') -> bool:
        """Check if metabolite is consumed (coefficient < 0)"""
        return self.get_ratio_value_for_metabolite(metabolite) < 0.0
    
    def is_metabolite_produced(self, metabolite: 'Metabolite') -> bool:
        """Check if metabolite is produced (coefficient > 0)"""
        return self.get_ratio_value_for_metabolite(metabolite) > 0.0
    
    def get_ratio_value_for_metabolite(self, metabolite: 'Metabolite') -> float:
        """Get the stoichiometric coefficient for a specific metabolite"""
        for ratio in self.get_metabolite_ratios():
            if ratio.get_metabolite() == metabolite:
                return ratio.get_ratio()
        return 0.0
    
    def is_external(self) -> bool:
        """Check if reaction is external (uptake or extract)"""
        return self.is_uptake() or self.is_extract()
    
    def is_uptake(self) -> bool:
        """Check if reaction has no educts (uptake reaction)"""
        return len(self.get_educt_ratios()) == 0
    
    def is_extract(self) -> bool:
        """Check if reaction has no products (extract reaction)"""
        return len(self.get_product_ratios()) == 0
    
    def has_integer_ratios(self) -> bool:
        """Check if all stoichiometric coefficients are integers"""
        for ratio in self.get_metabolite_ratios():
            if not ratio.is_integer_ratio():
                return False
        return True
    
    def accept(self, visitor: 'MetabolicNetworkVisitor') -> None:
        """Accept visitor for reaction traversal"""
        visitor.visit_reaction(self)
    
    def __str__(self) -> str:
        """String representation of the reaction formula"""
        return self.to_string(self.get_metabolite_ratios(), self.get_constraints().is_reversible())
    
    @staticmethod
    def to_string(ratios: List['MetaboliteRatio'], reversible: bool) -> str:
        """
        Create a string representation of a reaction from ratios and reversibility.
        
        Args:
            ratios: List of metabolite ratios
            reversible: Whether the reaction is reversible
            
        Returns:
            str: Reaction formula string like "A + B --> C + D" or "A <--> B"
        """
        educts = []
        products = []
        
        for ratio in ratios:
            ratio_value = ratio.get_ratio()
            if ratio_value < 0.0:
                educts.append(ratio.to_string_abs())
            elif ratio_value > 0.0:
                products.append(ratio.to_string_abs())
            else:
                # Zero coefficient goes to products side
                products.append(ratio.to_string_abs())
        
        # Build educt and product strings
        educt_str = " + ".join(educts) if educts else "#"
        product_str = " + ".join(products) if products else "#"
        
        # Choose arrow based on reversibility
        arrow = " <--> " if reversible else " --> "
        
        return educt_str + arrow + product_str
    
    def __hash__(self) -> int:
        """Hash based on reaction name"""
        return hash(self.get_name())
    
    def __eq__(self, other) -> bool:
        """
        Equality based on name and metabolite ratios.
        
        Two reactions are equal if they have the same name, same class,
        and the same metabolite ratios (after sorting by metabolite name).
        """
        if self is other:
            return True
        if other is None:
            return False
        if type(self) != type(other):
            return False
        if not isinstance(other, Reaction):
            return False
            
        # Check names match
        if self.get_name() != other.get_name():
            return False
        
        # Get metabolite ratios for both reactions
        my_ratios = list(self.get_metabolite_ratios())
        other_ratios = list(other.get_metabolite_ratios())
        
        if len(my_ratios) != len(other_ratios):
            return False
        
        # Sort both lists by metabolite name, then by ratio value
        def ratio_sort_key(ratio):
            return (ratio.get_metabolite().get_name(), ratio.get_ratio())
        
        my_ratios.sort(key=ratio_sort_key)
        other_ratios.sort(key=ratio_sort_key)
        
        # Compare sorted ratios
        for my_ratio, other_ratio in zip(my_ratios, other_ratios):
            if my_ratio != other_ratio:
                return False
        
        return True
    
    def _obj_hash_code(self) -> int:
        """Get the object hash code (like Java's super.hashCode())"""
        return object.__hash__(self)