#!/usr/bin/env python3
"""
DefaultMetabolite implementation - Python port of Java ch.javasoft.metabolic.impl.DefaultMetabolite

Basic implementation of the Metabolite interface with name generation utilities.

Java source: efmtool_source/ch/javasoft/metabolic/impl/DefaultMetabolite.java
"""

from typing import Optional, TYPE_CHECKING

from ..metabolite import Metabolite

if TYPE_CHECKING:
    from ..metabolic_network_visitor import MetabolicNetworkVisitor


class DefaultMetabolite(Metabolite):
    """
    Basic implementation of the Metabolite interface.
    
    Provides metabolite name and description storage with utility methods
    for generating systematic names from indices.
    """
    
    def __init__(self, name_or_index, description: Optional[str] = None):
        """
        Create a DefaultMetabolite
        
        Args:
            name_or_index: Either a string name or int index (for auto-generated name)
            description: Optional description string
        """
        if isinstance(name_or_index, int):
            # Constructor from index - generate name
            name = DefaultMetabolite.name(name_or_index)
        else:
            # Constructor from name string
            name = name_or_index
        
        if name is None:
            raise ValueError("null name not allowed")
        
        # Note: Java version had whitespace check commented out, so we skip it too
        
        self._name = name
        self._description = description
    
    def get_name(self) -> str:
        """Get the unique metabolite name"""
        return self._name
    
    def get_description(self) -> Optional[str]:
        """Get the optional metabolite description"""
        return self._description
    
    def accept(self, visitor: 'MetabolicNetworkVisitor') -> None:
        """Accept visitor for metabolite traversal"""
        visitor.visit_metabolite(self)
    
    def __hash__(self) -> int:
        """Hash based on name (like Java version)"""
        return hash(self._name)
    
    def __eq__(self, other) -> bool:
        """
        Equality based on name comparison with any Metabolite
        
        Note: This matches the Java logic which compares with any Metabolite instance,
        not just DefaultMetabolite instances.
        """
        if self is other:
            return True
        if isinstance(other, Metabolite):
            return self._name == other.get_name()
        return False
    
    def __str__(self) -> str:
        """String representation is just the name"""
        return self.get_name()
    
    @staticmethod
    def name(index: int) -> str:
        """
        Generate a systematic name from an integer index.
        
        Uses the same algorithm as Java efmtool:
        0 -> A, 1 -> B, ..., 25 -> Z, 26 -> AA, 27 -> AB, etc.
        
        Args:
            index: Non-negative integer index
            
        Returns:
            str: Generated name like "A", "B", ..., "Z", "AA", "AB", etc.
        """
        if index < 0:
            raise ValueError("Index must be non-negative")
        
        char_range = ord('Z') - ord('A') + 1  # 26
        result = []
        
        # Exact translation of Java do-while loop
        while True:
            ch = chr(ord('A') + (index % char_range))
            index = index // char_range - 1
            result.insert(0, ch)
            if index < 0:
                break
        
        return ''.join(result)
    
    @staticmethod 
    def names(count: int, prefix: str = "") -> list[str]:
        """
        Generate multiple systematic names with optional prefix.
        
        Args:
            count: Number of names to generate
            prefix: Optional prefix for each name
            
        Returns:
            list[str]: List of generated names
        """
        return [prefix + DefaultMetabolite.name(i) for i in range(count)]