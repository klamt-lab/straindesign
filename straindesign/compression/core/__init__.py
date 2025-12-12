"""
Core Compression Module

This module contains the complete compression algorithm implementations:
- StoichMatrixCompressor with full nullspace-based coupling detection
- All compression methods (CoupledZero, CoupledCombine, DeadEnd, etc.)
- Compression data structures and statistics

All algorithms preserve exact functionality from the original implementation.
"""

from .compression_method import CompressionMethod
from .stoich_matrix_compressor import StoichMatrixCompressor  
from .compression_statistics import CompressionStatistics
from .compression_record import CompressionRecord
from .work_record import WorkRecord
from .duplicate_gene_compressor import DuplicateGeneCompressor
from .stoich_matrix_compressed_network import StoichMatrixCompressedMetabolicNetwork as StoichMatrixCompressedNetwork

# Utility functions
def get_standard_compression_methods():
    """Get the standard set of compression methods."""
    return CompressionMethod.standard()

def get_nullspace_compression_methods(): 
    """Get compression methods that require nullspace computation."""
    return CompressionMethod.nullspace()

def get_safe_compression_methods():
    """Get compression methods that are verified to work safely."""
    return CompressionMethod.safe()

def get_java_compatible_compression_methods():
    """Get compression methods that should match Java EFMTool behavior."""
    return CompressionMethod.java_compatible()

__all__ = [
    'CompressionMethod',
    'StoichMatrixCompressor',
    'CompressionStatistics', 
    'CompressionRecord',
    'WorkRecord',
    'DuplicateGeneCompressor',
    'StoichMatrixCompressedNetwork',
    'get_standard_compression_methods',
    'get_nullspace_compression_methods',
    'get_safe_compression_methods',
    'get_java_compatible_compression_methods',
]