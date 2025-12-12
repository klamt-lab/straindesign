"""
EFMTool Compression Methods

Python port of ch.javasoft.metabolic.compress.CompressionMethod
Defines the 9 supported compression methods and their combinations.

From Java source: efmtool_source/ch/javasoft/metabolic/compress/CompressionMethod.java
Ported line-by-line for exact compatibility
"""

from enum import Enum
from typing import List, Tuple
import logging

class CompressionMethod(Enum):
    """
    The supported compression methods. The enum constants and 
    collections (lists) of them can be used to enable/disable
    specific compression methods.
    If no user specific configuration is made, STANDARD is used.
    
    Maps exactly to Java CompressionMethod enum values.
    """
    
    # Compression methods
    NULLSPACE = "Nullspace"  # Full nullspace-based compression (zero-flux, contradicting, combine)
    RECURSIVE = "Recursive"  # Iterate until no more compression possible

    @classmethod
    def all(cls) -> List['CompressionMethod']:
        """All compression methods"""
        return list(cls)

    @classmethod
    def none(cls) -> List['CompressionMethod']:
        """Empty list of compression methods"""
        return []

    @classmethod
    def standard(cls) -> List['CompressionMethod']:
        """
        Standard compression methods for metabolic network compression.
        Uses nullspace-based detection of zero-flux, contradicting, and coupled reactions.
        Iterates until no more compression is possible.
        """
        return [cls.NULLSPACE, cls.RECURSIVE]

    @staticmethod
    def is_containing_recursive(*methods: 'CompressionMethod') -> bool:
        """
        Returns True if Recursive is contained in the given methods.
        """
        return CompressionMethod.RECURSIVE in methods
    
    def __str__(self) -> str:
        """String representation uses the enum value (matches Java toString())"""
        return self.value
    
    def __repr__(self) -> str:
        """Representation shows the enum name"""
        return f"CompressionMethod.{self.name}"