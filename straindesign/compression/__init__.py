"""
Metabolic Network Compression Module (Pure Python EFMTool Port)

This package provides a clean, high-level API for metabolic network compression
using algorithms ported from the Java EFMTool implementation.

Key Features:
- Complete compression algorithm suite (CoupledZero, CoupledCombine, DeadEnd, etc.)
- Direct COBRA model integration
- Bidirectional expression transformation
- Exact rational arithmetic throughout (using fractions.Fraction)
- No Java or sympy dependencies

Primary API:
    >>> from straindesign.compression import compress_cobra_model
    >>>
    >>> result = compress_cobra_model(cobra_model, methods=["CoupledZero", "DeadEnd"])
    >>> compressed_model = result.compressed_model
    >>> converter = result.converter
    >>>
    >>> # Transform expressions
    >>> new_objective = converter.compress_expression(old_objective)
    >>> new_constraint = converter.compress_constraint(old_constraint)
"""

# Import with error handling to avoid dependency issues during development
try:
    # High-level API imports
    from .cobra_interface import (
        compress_cobra_model,
        CompressionResult,
        CompressionConverter,
        create_compression_converter,
        preprocess_cobra_model,
    )
    COBRA_INTERFACE_AVAILABLE = True
except ImportError:
    COBRA_INTERFACE_AVAILABLE = False

try:
    # Core compression functionality
    from .core import (
        CompressionMethod,
        CompressionStatistics,
        CompressionRecord,
        StoichMatrixCompressor,
        DuplicateGeneCompressor,
        get_standard_compression_methods,
        get_nullspace_compression_methods,
    )
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False

try:
    # Mathematical infrastructure  
    from .math import (
        BigFraction,
        DefaultBigIntegerRationalMatrix,
        GaussianElimination,
    )
    MATH_AVAILABLE = True
except ImportError:
    MATH_AVAILABLE = False

try:
    # Network infrastructure
    from .network import (
        MetabolicNetwork,
        FractionNumberStoichMetabolicNetwork,
        Metabolite,
        Reaction,
        FluxDistribution,
    )
    NETWORK_AVAILABLE = True
except ImportError:
    NETWORK_AVAILABLE = False

try:
    # Legacy compatibility
    from .legacy import (
        compress_model_efmtool,
        compress_objective,
        compress_modules,
    )
    LEGACY_AVAILABLE = True
except ImportError:
    LEGACY_AVAILABLE = False

# Version information
__version__ = "2.0.0"
__author__ = "EFMTool Python Port Contributors"
__description__ = "Complete EFMTool compression algorithms with clean Python API"

# Public API
__all__ = [
    # High-level API
    'compress_cobra_model',
    'CompressionResult', 
    'CompressionConverter',
    'create_compression_converter',
    'preprocess_cobra_model',
    
    # Core compression
    'CompressionMethod',
    'CompressionStatistics', 
    'CompressionRecord',
    'StoichMatrixCompressor',
    'DuplicateGeneCompressor',
    
    # Mathematical operations
    'BigFraction',
    'DefaultBigIntegerRationalMatrix',
    'GaussianElimination',
    
    # Network infrastructure
    'MetabolicNetwork',
    'FractionNumberStoichMetabolicNetwork', 
    'StoichMatrixCompressedNetwork',
    'Metabolite',
    'Reaction',
    'FluxDistribution',
    
    # Legacy compatibility
    'compress_model_efmtool',
    'compress_objective',
    'compress_modules',
    
    # Utility functions
    'get_standard_compression_methods',
    'get_nullspace_compression_methods',
]


def get_compression_info():
    """Get information about available compression methods and capabilities."""
    return {
        'version': __version__,
        'standard_methods': [method.name for method in get_standard_compression_methods()],
        'nullspace_methods': [method.name for method in get_nullspace_compression_methods()],
        'features': [
            'Exact rational arithmetic',
            'Complete nullspace-based coupling detection',
            'COBRA model integration',
            'Bidirectional expression transformation',
            'StrainDesign integration',
            'Backward compatibility'
        ]
    }