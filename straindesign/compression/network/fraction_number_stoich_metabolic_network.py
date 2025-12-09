#!/usr/bin/env python3
"""
FractionNumberStoichMetabolicNetwork implementation - Python port of Java ch.javasoft.metabolic.impl.FractionNumberStoichMetabolicNetwork

CRITICAL COMPONENT: Network representation with rational stoichiometry for compression algorithms.

This is the core network representation that the entire compression system operates on.
ANY ERRORS HERE WILL BE EXTREMELY HARD TO CATCH.

Java source: efmtool_source/ch/javasoft/metabolic/impl/FractionNumberStoichMetabolicNetwork.java
"""

from typing import List, Dict, Optional, Union, TYPE_CHECKING

from .metabolic_network import MetabolicNetwork
from .impl.abstract_reaction import AbstractReaction
from .impl.default_metabolite import DefaultMetabolite
from .impl.default_reaction_constraints import DEFAULT_REVERSIBLE, DEFAULT_IRREVERSIBLE
from .impl.default_metabolic_network import DefaultMetabolicNetwork

if TYPE_CHECKING:
    from ..math.bigint_rational_matrix import BigIntegerRationalMatrix
    from ..math.readable_bigint_rational_matrix import ReadableBigIntegerRationalMatrix
    from ..math.big_fraction import BigFraction
    from .metabolite import Metabolite
    from .reaction import Reaction
    from .metabolite_ratio import MetaboliteRatio
    from .reaction_constraints import ReactionConstraints
    from .metabolic_network_visitor import MetabolicNetworkVisitor


class FractionNumberStoichMetabolicNetwork(MetabolicNetwork):
    """
    CRITICAL: Network representation with rational stoichiometry for compression algorithms.
    
    This class represents a metabolic network using:
    - Exact rational arithmetic (BigFraction) for stoichiometric coefficients
    - String arrays for metabolite and reaction names
    - Boolean array for reaction reversibility
    - BigIntegerRationalMatrix for the stoichiometric matrix
    
    The matrix structure is: rows = metabolites, columns = reactions
    Matrix[i,j] = stoichiometric coefficient of metabolite i in reaction j
    """
    
    def __init__(self, stoich_or_metabolite_names, reaction_names_or_reversible=None, 
                 stoich=None, reversible=None):
        """
        Create FractionNumberStoichMetabolicNetwork
        
        Args:
            Two constructor patterns:
            1) stoich_only: (stoich: BigIntegerRationalMatrix, reversible: List[bool])
            2) full: (metabolite_names: List[str], reaction_names: List[str], 
                     stoich: BigIntegerRationalMatrix, reversible: List[bool])
        """
        if stoich is None:
            # Pattern 1: FractionNumberStoichMetabolicNetwork(stoich, reversible)
            stoich_matrix = stoich_or_metabolite_names
            reversible_array = reaction_names_or_reversible
            
            # Generate systematic names using DefaultMetabolicNetwork utility
            self._metabolite_names = DefaultMetabolicNetwork.metabolite_names(stoich_matrix.get_row_count())
            self._reaction_names = DefaultMetabolicNetwork.reaction_names(stoich_matrix.get_column_count())
        else:
            # Pattern 2: FractionNumberStoichMetabolicNetwork(metabolite_names, reaction_names, stoich, reversible)
            self._metabolite_names = list(stoich_or_metabolite_names)  # Defensive copy
            self._reaction_names = list(reaction_names_or_reversible)   # Defensive copy
            stoich_matrix = stoich
            reversible_array = reversible
        
        # Store core data - CRITICAL: These must be exact references for compression
        self._reversible = list(reversible_array)  # Defensive copy
        self._stoich = stoich_matrix  # Matrix reference (should not be copied)
        
        # Lazy initialization - populated when first accessed
        self._meta_indices_by_name: Optional[Dict[str, int]] = None
        self._reaction_indices_by_name: Optional[Dict[str, int]] = None
        self._metabolites: Optional[List['Metabolite']] = None
        self._reactions: Optional[List['Reaction']] = None
        
        # Validate dimensions for safety
        if len(self._metabolite_names) != self._stoich.get_row_count():
            raise ValueError(f"Metabolite names count ({len(self._metabolite_names)}) != matrix row count ({self._stoich.get_row_count()})")
        if len(self._reaction_names) != self._stoich.get_column_count():
            raise ValueError(f"Reaction names count ({len(self._reaction_names)}) != matrix column count ({self._stoich.get_column_count()})")
        if len(self._reversible) != self._stoich.get_column_count():
            raise ValueError(f"Reversible array length ({len(self._reversible)}) != matrix column count ({self._stoich.get_column_count()})")
    
    def _get_meta_indices_by_name(self) -> Dict[str, int]:
        """Get metabolite name -> index mapping (lazy initialization)"""
        if self._meta_indices_by_name is None:
            self._meta_indices_by_name = {
                name: i for i, name in enumerate(self._metabolite_names)
            }
        return self._meta_indices_by_name
    
    def _get_reaction_indices_by_name(self) -> Dict[str, int]:
        """Get reaction name -> index mapping (lazy initialization)"""
        if self._reaction_indices_by_name is None:
            self._reaction_indices_by_name = {
                name: i for i, name in enumerate(self._reaction_names)
            }
        return self._reaction_indices_by_name
    
    def get_metabolite(self, name: str) -> 'Metabolite':
        """Get metabolite by name or raise ValueError if not found"""
        index = self.get_metabolite_index(name)
        if index < 0:
            raise ValueError(f"no such metabolite: {name}")
        return self.get_metabolites()[index]
    
    def get_metabolite_index(self, name: str) -> int:
        """Get metabolite index by name, return -1 if not found"""
        return self._get_meta_indices_by_name().get(name, -1)
    
    def get_metabolites(self) -> List['Metabolite']:
        """Get all metabolites (lazy initialization)"""
        if self._metabolites is None:
            self._metabolites = [
                DefaultMetabolite(name) for name in self._metabolite_names
            ]
        return self._metabolites
    
    def get_reaction(self, name: str) -> 'Reaction':
        """Get reaction by name or raise ValueError if not found"""
        index = self.get_reaction_index(name)
        if index < 0:
            raise ValueError(f"no such reaction: {name}")
        return self.get_reactions()[index]
    
    def get_reaction_index(self, name: str) -> int:
        """Get reaction index by name, return -1 if not found"""
        return self._get_reaction_indices_by_name().get(name, -1)
    
    def get_metabolite_names(self) -> List[str]:
        """Get list of all metabolite names"""
        return self._metabolite_names.copy()
    
    def get_reaction_names(self) -> List[str]:
        """Get list of all reaction names"""
        return self._reaction_names.copy()
    
    def get_reaction_reversibilities(self) -> List[bool]:
        """Get list of reaction reversibility flags"""
        return self._reversible.copy()
    
    def get_reactions(self, metabolite: 'Metabolite' = None) -> List['Reaction']:
        """
        Get reactions (all reactions, or those involving a specific metabolite)
        
        Args:
            metabolite: Optional metabolite to filter by
            
        Returns:
            List[Reaction]: All reactions or filtered reactions
        """
        if self._reactions is None:
            # Lazy initialization of all reactions
            self._reactions = [
                BigIntegerReaction(self, i) for i in range(len(self._reaction_names))
            ]
        
        if metabolite is None:
            return self._reactions
        else:
            # Filter reactions that involve this metabolite
            return [reaction for reaction in self._reactions 
                   if reaction.is_metabolite_participating(metabolite)]
    
    def get_stoichiometric_matrix(self) -> 'ReadableBigIntegerRationalMatrix':
        """
        CRITICAL: Get the stoichiometric matrix for compression algorithms.
        
        Returns:
            ReadableBigIntegerRationalMatrix: The matrix with exact rational coefficients
        """
        return self._stoich
    
    def accept(self, visitor: 'MetabolicNetworkVisitor') -> None:
        """Accept visitor for network traversal"""
        visitor.visit_metabolic_network(self)
    
    @staticmethod
    def get_stoich(net: 'MetabolicNetwork') -> 'ReadableBigIntegerRationalMatrix':
        """
        CRITICAL: Extract stoichiometric matrix from any metabolic network.
        
        If the network is a FractionNumberStoichMetabolicNetwork, returns the exact matrix.
        Otherwise, creates an approximated matrix (this should be avoided for compression).
        
        Args:
            net: The metabolic network
            
        Returns:
            ReadableBigIntegerRationalMatrix: The stoichiometric matrix
        """
        if isinstance(net, FractionNumberStoichMetabolicNetwork):
            return net.get_stoichiometric_matrix()
        else:
            # This branch should not be used for compression - it's for compatibility only
            raise NotImplementedError("Conversion from other network types not yet implemented")


class BigIntegerReaction(AbstractReaction):
    """
    CRITICAL: Reaction implementation backed by rational stoichiometric matrix.
    
    This class represents a single reaction (column) in the stoichiometric matrix.
    """
    
    def __init__(self, network: FractionNumberStoichMetabolicNetwork, reaction_index: int):
        """
        Create reaction for a specific column in the stoichiometric matrix
        
        Args:
            network: The parent network
            reaction_index: The column index in the stoichiometric matrix
        """
        super().__init__()
        self._network = network
        self._reaction_index = reaction_index
        
        # CRITICAL: Pre-compute metabolite indices with non-zero coefficients
        # This optimization is essential for compression performance
        self._metabolite_indices = []
        stoich = network.get_stoichiometric_matrix()
        for metabolite_idx in range(len(network._metabolite_names)):
            if stoich.get_signum_at(metabolite_idx, reaction_index) != 0:
                self._metabolite_indices.append(metabolite_idx)
    
    def get_name(self) -> str:
        """Get the reaction name"""
        return self._network._reaction_names[self._reaction_index]
    
    def get_constraints(self) -> 'ReactionConstraints':
        """Get reaction constraints based on reversibility"""
        return (DEFAULT_REVERSIBLE if self._network._reversible[self._reaction_index] 
                else DEFAULT_IRREVERSIBLE)
    
    def get_metabolite_ratios(self) -> List['MetaboliteRatio']:
        """
        CRITICAL: Get all metabolite ratios (stoichiometric coefficients) for this reaction.
        
        Returns only ratios for metabolites with non-zero coefficients.
        """
        return [
            BigIntegerMetaboliteRatio(self._network, self._reaction_index, metabolite_idx)
            for metabolite_idx in self._metabolite_indices
        ]


class AbstractBigIntegerMetaboliteRatio:
    """
    CRITICAL: Base class for metabolite ratios backed by rational matrix.
    
    This represents a single stoichiometric coefficient matrix[metabolite_idx, reaction_idx].
    """
    
    def __init__(self, reaction_index: int, metabolite_index: int):
        """
        Create metabolite ratio for specific matrix position
        
        Args:
            reaction_index: Column index in stoichiometric matrix
            metabolite_index: Row index in stoichiometric matrix
        """
        self._reaction_index = reaction_index
        self._metabolite_index = metabolite_index
    
    def get_stoich(self) -> 'ReadableBigIntegerRationalMatrix':
        """Get the stoichiometric matrix (must be implemented by subclass)"""
        raise NotImplementedError("Subclasses must implement get_stoich()")
    
    def accept(self, visitor: 'MetabolicNetworkVisitor') -> None:
        """Accept visitor for ratio traversal"""
        visitor.visit_metabolite_ratio(self)
    
    def get_ratio(self) -> float:
        """Get ratio as float (may lose precision)"""
        return self.get_stoich().get_double_value_at(self._metabolite_index, self._reaction_index)
    
    def get_number_ratio(self) -> Union[int, 'BigFraction']:
        """CRITICAL: Get exact rational ratio"""
        return self.get_stoich().get_big_fraction_value_at(self._metabolite_index, self._reaction_index)
    
    def is_educt(self) -> bool:
        """Check if this is an educt (negative coefficient)"""
        return self.get_stoich().get_signum_at(self._metabolite_index, self._reaction_index) < 0
    
    def is_integer_ratio(self) -> bool:
        """Check if the ratio is an integer"""
        ratio = self.get_number_ratio()
        # For BigFraction, check if denominator is 1
        reduced = ratio.reduce() if hasattr(ratio, 'reduce') else ratio
        return reduced.denominator == 1 if hasattr(reduced, 'denominator') else ratio == int(ratio)
    
    def __str__(self) -> str:
        """String representation of the ratio"""
        return self._to_string(self.get_number_ratio(), self.get_metabolite())
    
    def to_string_abs(self) -> str:
        """String representation of the absolute value"""
        ratio = self.get_number_ratio()
        abs_ratio = abs(ratio) if hasattr(ratio, '__abs__') else ratio.abs()
        return self._to_string(abs_ratio, self.get_metabolite())
    
    def _to_string(self, ratio: Union[int, 'BigFraction'], metabolite: 'Metabolite') -> str:
        """Format ratio and metabolite as string"""
        if hasattr(ratio, 'is_one') and ratio.is_one():
            return str(metabolite)
        else:
            return f"{ratio} {metabolite}"
    
    def invert(self) -> 'MetaboliteRatio':
        """Return inverted ratio (negated coefficient)"""
        original = self
        
        class InvertedMetaboliteRatio:
            """Inverted metabolite ratio with negated coefficient"""
            
            def __init__(self):
                self._ratio = -original.get_number_ratio()
            
            def get_metabolite(self) -> 'Metabolite':
                return original.get_metabolite()
            
            def get_number_ratio(self) -> Union[int, 'BigFraction']:
                return self._ratio
            
            def get_ratio(self) -> float:
                return float(self._ratio)
            
            def invert(self) -> 'MetaboliteRatio':
                return original
            
            def is_educt(self) -> bool:
                return not original.is_educt()
            
            def is_integer_ratio(self) -> bool:
                return original.is_integer_ratio()
            
            def to_string_abs(self) -> str:
                return original.to_string_abs()
            
            def accept(self, visitor: 'MetabolicNetworkVisitor') -> None:
                visitor.visit_metabolite_ratio(self)
        
        return InvertedMetaboliteRatio()
    
    def __hash__(self) -> int:
        """Hash based on metabolite and ratio"""
        return hash(self.get_metabolite()) ^ hash(self.get_number_ratio())
    
    def __eq__(self, other) -> bool:
        """
        CRITICAL: Equality based on metabolite and numerical ratio value.
        
        Uses numerical equality for BigFraction comparison.
        """
        if self is other:
            return True
        if other is None or type(self) != type(other):
            return False
        
        # Check metabolite equality
        if self.get_metabolite() != other.get_metabolite():
            return False
        
        # Check numerical ratio equality
        my_ratio = self.get_number_ratio()
        other_ratio = other.get_number_ratio()
        
        # Use numerical comparison for BigFraction
        if hasattr(my_ratio, 'equals_numerically'):
            return my_ratio.equals_numerically(other_ratio)
        else:
            return my_ratio == other_ratio


class BigIntegerMetaboliteRatio(AbstractBigIntegerMetaboliteRatio):
    """
    CRITICAL: Concrete metabolite ratio implementation for FractionNumberStoichMetabolicNetwork.
    """
    
    def __init__(self, network: FractionNumberStoichMetabolicNetwork, 
                 reaction_index: int, metabolite_index: int):
        """
        Create metabolite ratio for specific network and matrix position
        
        Args:
            network: The parent network
            reaction_index: Column index in stoichiometric matrix
            metabolite_index: Row index in stoichiometric matrix
        """
        super().__init__(reaction_index, metabolite_index)
        self._network = network
    
    def get_metabolite(self) -> 'Metabolite':
        """Get the metabolite object"""
        return self._network.get_metabolites()[self._metabolite_index]
    
    def get_stoich(self) -> 'ReadableBigIntegerRationalMatrix':
        """Get the stoichiometric matrix from parent network"""
        return self._network._stoich