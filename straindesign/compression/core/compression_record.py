"""
EFMTool Compression Record

Python port of ch.javasoft.metabolic.compress.StoichMatrixCompressor.CompressionRecord
Core data structure containing compression matrices and reversibility information.

From Java source: efmtool_source/ch/javasoft/metabolic/compress/StoichMatrixCompressor.java (inner class)
Ported line-by-line for exact compatibility
"""

from typing import List, Optional
from ..math.bigint_rational_matrix import BigIntegerRationalMatrix


class CompressionRecord:
    """
    Core data structure containing compression matrices and reversibility information.
    
    This class represents the result of a compression operation, containing:
    - pre: preprocessing matrix (m_compressed x m_original) 
    - cmp: compressed stoichiometric matrix (m_compressed x r_compressed)
    - post: postprocessing matrix (r_original x r_compressed)
    - reversible: reversibility flags for compressed reactions
    
    The compression relationship is: pre * stoich_original * post = cmp
    For EFM uncompression: efm_original = post * efm_compressed
    """
    
    def __init__(self, pre: BigIntegerRationalMatrix, 
                 cmp: BigIntegerRationalMatrix,
                 post: BigIntegerRationalMatrix, 
                 reversible: List[bool],
                 stats: Optional['CompressionStatistics'] = None):
        """
        Initialize compression record with matrices and reversibility information.
        
        Args:
            pre: Preprocessing matrix (m_compressed x m_original)
            cmp: Compressed stoichiometric matrix (m_compressed x r_compressed) 
            post: Postprocessing matrix (r_original x r_compressed)
            reversible: Reversibility flags for compressed reactions
            stats: Optional compression statistics
            
        Note:
            All matrices are stored as final references in Java - immutable after creation.
            The reversible array is also treated as immutable.
        """
        # Store matrix references (immutable in Java style)
        self.pre = pre
        self.cmp = cmp  
        self.post = post
        
        # Store reversibility array (defensive copy in Python style)
        self.reversible = reversible.copy() if reversible else []
        
        # Store statistics if provided
        self.stats = stats
    
    def get_pre_matrix(self) -> BigIntegerRationalMatrix:
        """
        Get the preprocessing matrix.
        
        Returns:
            Preprocessing matrix (m_compressed x m_original)
        """
        return self.pre
    
    def get_compressed_matrix(self) -> BigIntegerRationalMatrix:
        """
        Get the compressed stoichiometric matrix.
        
        Returns:
            Compressed stoichiometric matrix (m_compressed x r_compressed)
        """
        return self.cmp
    
    def get_post_matrix(self) -> BigIntegerRationalMatrix:
        """
        Get the postprocessing matrix.
        
        Returns:
            Postprocessing matrix (r_original x r_compressed)
        """
        return self.post
    
    def get_reversible(self) -> List[bool]:
        """
        Get the reversibility flags for compressed reactions.
        
        Returns:
            Copy of reversibility flags array
        """
        return self.reversible.copy()
    
    def get_original_metabolite_count(self) -> int:
        """
        Get the number of metabolites in the original network.
        
        Returns:
            Number of columns in pre matrix (original metabolite count)
        """
        return self.pre.get_column_count()
    
    def get_compressed_metabolite_count(self) -> int:
        """
        Get the number of metabolites in the compressed network.
        
        Returns:
            Number of rows in pre matrix (compressed metabolite count)
        """
        return self.pre.get_row_count()
    
    def get_original_reaction_count(self) -> int:
        """
        Get the number of reactions in the original network.
        
        Returns:
            Number of rows in post matrix (original reaction count)
        """
        return self.post.get_row_count()
    
    def get_compressed_reaction_count(self) -> int:
        """
        Get the number of reactions in the compressed network.
        
        Returns:
            Number of columns in post matrix (compressed reaction count)
        """
        return self.post.get_column_count()
    
    def validate_dimensions(self) -> bool:
        """
        Validate that all matrix dimensions are consistent.
        
        Returns:
            True if all dimensions are consistent, False otherwise
        """
        try:
            # Check pre matrix dimensions
            m_compressed = self.pre.get_row_count()
            m_original = self.pre.get_column_count()
            
            # Check cmp matrix dimensions
            if self.cmp.get_row_count() != m_compressed:
                return False
            r_compressed = self.cmp.get_column_count()
            
            # Check post matrix dimensions
            r_original = self.post.get_row_count()
            if self.post.get_column_count() != r_compressed:
                return False
            
            # Check reversible array length
            if len(self.reversible) != r_compressed:
                return False
            
            return True
            
        except (AttributeError, TypeError):
            return False
    
    def get_compression_info(self) -> dict:
        """
        Get summary information about the compression.
        
        Returns:
            Dictionary with compression statistics
        """
        if not self.validate_dimensions():
            return {"valid": False, "error": "Invalid matrix dimensions"}
        
        m_original = self.get_original_metabolite_count()
        m_compressed = self.get_compressed_metabolite_count()
        r_original = self.get_original_reaction_count()
        r_compressed = self.get_compressed_reaction_count()
        
        return {
            "valid": True,
            "metabolites": {
                "original": m_original,
                "compressed": m_compressed,
                "reduction": m_original - m_compressed,
                "reduction_percent": ((m_original - m_compressed) / m_original * 100) if m_original > 0 else 0
            },
            "reactions": {
                "original": r_original,
                "compressed": r_compressed,
                "reduction": r_original - r_compressed,
                "reduction_percent": ((r_original - r_compressed) / r_original * 100) if r_original > 0 else 0
            },
            "reversible_reactions": sum(self.reversible)
        }
    
    def __str__(self) -> str:
        """
        String representation of compression record.
        
        Returns:
            Human-readable summary of compression record
        """
        info = self.get_compression_info()
        if not info["valid"]:
            return f"CompressionRecord(INVALID: {info.get('error', 'Unknown error')})"
        
        return (f"CompressionRecord("
                f"metabolites: {info['metabolites']['original']} → {info['metabolites']['compressed']}, "
                f"reactions: {info['reactions']['original']} → {info['reactions']['compressed']}, "
                f"reversible: {info['reversible_reactions']}/{info['reactions']['compressed']})")
    
    def __repr__(self) -> str:
        """
        Detailed string representation for debugging.
        
        Returns:
            Detailed representation showing matrix dimensions
        """
        try:
            return (f"CompressionRecord("
                    f"pre={self.pre.get_row_count()}x{self.pre.get_column_count()}, "
                    f"cmp={self.cmp.get_row_count()}x{self.cmp.get_column_count()}, "
                    f"post={self.post.get_row_count()}x{self.post.get_column_count()}, "
                    f"reversible={len(self.reversible)})")
        except (AttributeError, TypeError):
            return "CompressionRecord(INVALID_MATRICES)"
    
    def __eq__(self, other) -> bool:
        """
        Check equality of compression records.
        
        Args:
            other: Other CompressionRecord to compare
            
        Returns:
            True if records are equal (same matrices and reversibility)
        """
        if not isinstance(other, CompressionRecord):
            return False
        
        try:
            # Compare matrix equality (assuming matrices implement __eq__)
            return (self.pre == other.pre and
                    self.cmp == other.cmp and
                    self.post == other.post and
                    self.reversible == other.reversible)
        except (AttributeError, TypeError):
            return False
    
    def __hash__(self) -> int:
        """
        Hash for CompressionRecord (based on dimensions and reversible count).
        
        Returns:
            Hash value for the record
        """
        try:
            return hash((
                self.pre.get_row_count(), self.pre.get_column_count(),
                self.cmp.get_row_count(), self.cmp.get_column_count(),
                self.post.get_row_count(), self.post.get_column_count(),
                tuple(self.reversible)
            ))
        except (AttributeError, TypeError):
            return hash(id(self))  # Fallback to object id