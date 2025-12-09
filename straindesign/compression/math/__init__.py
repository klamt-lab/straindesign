"""
Mathematical Infrastructure Module

This module provides the mathematical foundation for the EFMTool compression algorithms:
- Exact rational arithmetic with BigFraction
- Matrix operations with exact precision  
- Gaussian elimination and nullspace computation

All operations maintain exact precision using rational arithmetic.
"""

from .big_fraction import BigFraction
from .bigint_rational_matrix import BigIntegerRationalMatrix
from .default_bigint_rational_matrix import DefaultBigIntegerRationalMatrix
from .gauss import Gauss as GaussianElimination
from .bigint_rational_matrix_operations import BigIntegerRationalMatrixOperations as MatrixOperations

__all__ = [
    'BigFraction',
    'BigIntegerRationalMatrix',
    'DefaultBigIntegerRationalMatrix', 
    'GaussianElimination',
    'MatrixOperations',
]