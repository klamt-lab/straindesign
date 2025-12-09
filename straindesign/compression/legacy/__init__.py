"""
Legacy Compatibility Module

This module provides backward compatibility with existing code that uses
the old API patterns from compression_python_port_test.
"""

from .compatibility import (
    compress_model_efmtool,
    compress_objective,
    compress_modules,
    stoichmat_coeff2rational,
    remove_conservation_relations,
)

__all__ = [
    'compress_model_efmtool',
    'compress_objective',
    'compress_modules', 
    'stoichmat_coeff2rational',
    'remove_conservation_relations',
]