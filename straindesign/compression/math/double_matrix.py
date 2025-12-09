"""
DoubleMatrix interface - Python port of ch.javasoft.smx.iface.DoubleMatrix

This module provides the interface for matrices containing double-precision floating-point
numbers, combining readable and writable capabilities.
"""

from abc import abstractmethod
from typing import List
from .readable_bigint_rational_matrix import ReadableDoubleMatrix
from .bigint_rational_matrix import WritableBigIntegerRationalMatrix


class WritableDoubleMatrix(WritableBigIntegerRationalMatrix):
    """
    WritableDoubleMatrix interface - Python port of ch.javasoft.smx.iface.WritableDoubleMatrix<N>
    
    Provides write operations for double-precision matrices.
    Extends WritableBigIntegerRationalMatrix with double-specific operations.
    """
    
    @abstractmethod
    def set_value_at_double(self, row: int, col: int, value: float) -> None:
        """Set double value at specified position"""
        pass
    
    @abstractmethod
    def add_double(self, row: int, col: int, value: float) -> None:
        """Add double value to existing value at specified position"""
        pass
    
    @abstractmethod
    def multiply_double(self, row: int, col: int, factor: float) -> None:
        """Multiply existing value at specified position by double factor"""
        pass
    
    @abstractmethod
    def multiply_row_double(self, row: int, factor: float) -> None:
        """Multiply entire row by double factor"""
        pass
    
    @abstractmethod
    def add_row_to_other_row_double(self, src_row: int, src_factor: float, 
                                   dst_row: int, dst_factor: float) -> None:
        """Add source row (multiplied by src_factor) to destination row (multiplied by dst_factor)"""
        pass


class DoubleMatrix(ReadableDoubleMatrix, WritableDoubleMatrix):
    """
    DoubleMatrix interface - Python port of ch.javasoft.smx.iface.DoubleMatrix
    
    Main double matrix interface that combines readable and writable capabilities.
    This interface is used for approximate computations where double precision
    is sufficient (as opposed to exact rational arithmetic).
    """
    
    @abstractmethod
    def new_instance(self, rows: int, cols: int) -> 'DoubleMatrix':
        """Create new double matrix instance with given dimensions"""
        pass
    
    @abstractmethod
    def new_instance_from_data(self, data: List[List[float]], rows_in_dim1: bool = True) -> 'DoubleMatrix':
        """Create new double matrix instance from 2D double data"""
        pass
    
    @abstractmethod
    def clone(self) -> 'DoubleMatrix':
        """Create deep copy of this double matrix"""
        pass
    
    @abstractmethod
    def transpose(self) -> 'DoubleMatrix':
        """Return transposed version of this double matrix"""
        pass