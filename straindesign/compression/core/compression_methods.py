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
    
    # Primary compression methods
    COUPLED_ZERO = "CoupledZero"
    COUPLED_CONTRADICTING = "CoupledContradicting"  
    COUPLED_COMBINE = "CoupledCombine"
    UNIQUE_FLOWS = "UniqueFlows"
    DEAD_END = "DeadEnd"
    DUPLICATE_GENE = "DuplicateGene"
    DUPLICATE_GENE_EXTENDED = "DuplicateGeneExtended"
    INTERCHANGEABLE_METABOLITE = "InterchangeableMetabolite"
    RECURSIVE = "Recursive"

    # Static method collections (translated to class properties)
    
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
        Standard uses CoupledZero, CoupledContradicting, CoupledCombine, 
        UniqueFlows, DeadEnd, Recursive, DuplicateGene compression 
        (omitting InterchangeableMetabolite)
        """
        return [cls.COUPLED_ZERO, cls.COUPLED_CONTRADICTING, cls.COUPLED_COMBINE,
                cls.UNIQUE_FLOWS, cls.DEAD_END, cls.RECURSIVE, cls.DUPLICATE_GENE]
    
    @classmethod
    def standard_no_duplicate(cls) -> List['CompressionMethod']:
        """
        Like standard, but without duplicate gene removal, uses 
        CoupledZero, CoupledContradicting, CoupledCombine, UniqueFlows, DeadEnd, Recursive compression
        (omitting DuplicateGene, DuplicateGeneExtended and InterchangeableMetabolite)
        """
        return [cls.COUPLED_ZERO, cls.COUPLED_CONTRADICTING, cls.COUPLED_COMBINE,
                cls.UNIQUE_FLOWS, cls.DEAD_END, cls.RECURSIVE]
    
    @classmethod
    def standard_no_combine(cls) -> List['CompressionMethod']:
        """
        Like standard, but without compression, i.e. only removal of inconsistencies and duplicate genes. 
        Uses CoupledZero, CoupledContradicting, DeadEnd, DuplicateGene, Recursive compression
        """
        return [cls.COUPLED_ZERO, cls.COUPLED_CONTRADICTING, cls.DEAD_END, 
                cls.DUPLICATE_GENE, cls.RECURSIVE]
    
    @classmethod
    def standard_no_null(cls) -> List['CompressionMethod']:
        """
        Like standard, but without nullspace compression. 
        Uses CoupledZero, CoupledContradicting, UniqueFlows, DeadEnd, DuplicateGene, Recursive compression
        """
        return [cls.UNIQUE_FLOWS, cls.DEAD_END, cls.DUPLICATE_GENE, cls.RECURSIVE]
    
    @classmethod
    def standard_no_null_combine(cls) -> List['CompressionMethod']:
        """
        Like standard, but nullspace compression only for removal of zero fluxes. 
        Uses CoupledZero, CoupledContradicting, UniqueFlows, DeadEnd, DuplicateGene, Recursive compression
        """
        return [cls.COUPLED_ZERO, cls.COUPLED_CONTRADICTING, cls.UNIQUE_FLOWS,
                cls.DEAD_END, cls.DUPLICATE_GENE, cls.RECURSIVE]
    
    # Instance methods
    
    def contained_in(self, *methods: 'CompressionMethod') -> bool:
        """
        Check if this compression method is contained in the given methods.
        
        Args:
            *methods: Variable number of CompressionMethod instances to check
            
        Returns:
            bool: True if this method is in the given methods, False otherwise
        """
        return self in methods
    
    def is_duplicate_gene(self) -> bool:
        """
        Returns True if this method is DuplicateGene or DuplicateGeneExtended
        """
        return self in (CompressionMethod.DUPLICATE_GENE, CompressionMethod.DUPLICATE_GENE_EXTENDED)
    
    def is_coupled(self) -> bool:
        """
        Returns True if this method is CoupledZero, CoupledContradicting or CoupledCombine
        """
        return self in (CompressionMethod.COUPLED_ZERO, 
                       CompressionMethod.COUPLED_CONTRADICTING, 
                       CompressionMethod.COUPLED_COMBINE)
    
    # Static utility methods
    
    @staticmethod
    def methods(*methods: 'CompressionMethod') -> List['CompressionMethod']:
        """
        Utility method to create a list of compression methods.
        
        Args:
            *methods: Variable number of CompressionMethod instances
            
        Returns:
            List of CompressionMethod instances
        """
        return list(methods)
    
    @staticmethod
    def is_containing_coupled(*methods: 'CompressionMethod') -> bool:
        """
        Returns True if any of the methods is coupled (CoupledZero, CoupledContradicting, CoupledCombine)
        
        Args:
            *methods: Variable number of CompressionMethod instances to check
            
        Returns:
            bool: True if any method is coupled, False otherwise
        """
        for method in methods:
            if method.is_coupled():
                return True
        return False
    
    @staticmethod
    def is_containing_recursive(*methods: 'CompressionMethod') -> bool:
        """
        Returns True if Recursive is contained in the given methods.
        
        Args:
            *methods: Variable number of CompressionMethod instances to check
            
        Returns:
            bool: True if Recursive is in methods, False otherwise
        """
        return CompressionMethod.RECURSIVE.contained_in(*methods)
    
    @staticmethod
    def is_containing_duplicate_gene(*methods: 'CompressionMethod') -> bool:
        """
        Returns True if any of the methods is duplicate gene (DuplicateGene or DuplicateGeneExtended)
        
        Args:
            *methods: Variable number of CompressionMethod instances to check
            
        Returns:
            bool: True if any method is duplicate gene, False otherwise
        """
        for method in methods:
            if method.is_duplicate_gene():
                return True
        return False
    
    @staticmethod
    def log(log_level: int, *methods: 'CompressionMethod') -> None:
        """
        Logs the compression methods. The output string has the form:
        "network compression methods are [COUPLED_ZERO, RECURSIVE]"
        
        Args:
            log_level: The log level (e.g., logging.INFO)
            *methods: The compression methods to log
        """
        method_names = [method.value for method in methods]
        logging.log(log_level, f"network compression methods are {method_names}")
    
    @staticmethod
    def log_unsupported(log_level: int, methods: List['CompressionMethod'], 
                       *supported: 'CompressionMethod') -> None:
        """
        Logs that the given compression methods are unsupported. The log string
        looks like this:
        "NOTE: ignoring unsupported network compression methods: COUPLED_ZERO, RECURSIVE"
        
        Args:
            log_level: The log level on which to log (e.g., logging.INFO)
            methods: The compression methods which have been specified
            *supported: The compression methods which are supported
        """
        unsupported_methods = []
        for method in methods:
            if not method.contained_in(*supported):
                unsupported_methods.append(method.value)
        
        if unsupported_methods:
            plural = "s" if len(unsupported_methods) > 1 else ""
            unsupported_str = ", ".join(unsupported_methods)
            logging.log(log_level, 
                       f"NOTE: ignoring unsupported network compression method{plural}: {unsupported_str}")
    
    @staticmethod
    def remove_duplicate_gene_methods(*methods: 'CompressionMethod') -> List['CompressionMethod']:
        """
        Returns the methods after removing DuplicateGene and DuplicateGeneExtended if contained in methods
        
        Args:
            *methods: Variable number of CompressionMethod instances
            
        Returns:
            List of CompressionMethod instances with duplicate gene methods removed
        """
        return [method for method in methods if not method.is_duplicate_gene()]
    
    def __str__(self) -> str:
        """String representation uses the enum value (matches Java toString())"""
        return self.value
    
    def __repr__(self) -> str:
        """Representation shows the enum name"""
        return f"CompressionMethod.{self.name}"