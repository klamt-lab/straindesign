"""
ReadableBigIntegerRationalMatrix interface - Python port of ch.javasoft.smx.iface.ReadableBigIntegerRationalMatrix

This module provides the interface for readable matrices containing rational numbers
(BigFraction values), extending the basic readable matrix functionality.
"""

from abc import abstractmethod
from typing import List
from .readable_matrix import ReadableMatrix, N
from .big_fraction import BigFraction


class ReadableDoubleMatrix(ReadableMatrix[N]):
    """
    ReadableDoubleMatrix interface - Python port of ch.javasoft.smx.iface.ReadableDoubleMatrix<N>
    
    Extends ReadableMatrix with double-precision specific methods.
    """
    
    @abstractmethod
    def to_double_matrix(self, enforce_new_instance: bool = False) -> 'DoubleMatrix':
        """Convert to double matrix, optionally forcing new instance"""
        pass
    
    @abstractmethod
    def sub_double_matrix(self, row_start: int, row_end: int, col_start: int, col_end: int) -> 'DoubleMatrix':
        """Extract submatrix as double matrix"""
        pass
    
    @abstractmethod
    def get_double_value_at(self, row: int, col: int) -> float:
        """Get double value at specified position"""
        pass
    
    @abstractmethod
    def to_double_array(self) -> List[float]:
        """Convert matrix to 1D double array (row-major order)"""
        pass
    
    @abstractmethod
    def to_array(self, array: List[float]) -> None:
        """Fill provided array with matrix values (row-major order)"""
        pass
    
    @abstractmethod
    def get_double_row(self, row: int) -> List[float]:
        """Get specified row as double array"""
        pass
    
    @abstractmethod
    def get_double_column(self, col: int) -> List[float]:
        """Get specified column as double array"""
        pass
    
    @abstractmethod
    def get_double_rows(self) -> List[List[float]]:
        """Get all rows as 2D double array"""
        pass
    
    @abstractmethod
    def get_double_columns(self) -> List[List[float]]:
        """Get all columns as 2D double array"""
        pass


class ReadableBigIntegerRationalMatrix(ReadableDoubleMatrix[N]):
    """
    ReadableBigIntegerRationalMatrix interface - Python port of ch.javasoft.smx.iface.ReadableBigIntegerRationalMatrix<N>
    
    Extends ReadableDoubleMatrix with rational number (BigFraction) specific methods.
    This interface provides access to exact rational arithmetic operations on matrices.
    """
    
    @abstractmethod
    def to_big_integer_rational_matrix(self, enforce_new_instance: bool = False) -> 'BigIntegerRationalMatrix':
        """Convert to mutable rational matrix, optionally forcing new instance"""
        pass
    
    @abstractmethod
    def sub_big_integer_rational_matrix(self, row_start: int, row_end: int, col_start: int, col_end: int) -> 'BigIntegerRationalMatrix':
        """Extract submatrix as rational matrix"""
        pass
    
    @abstractmethod
    def get_big_fraction_value_at(self, row: int, col: int) -> BigFraction:
        """Get BigFraction value at specified position"""
        pass
    
    @abstractmethod
    def get_big_integer_numerator_at(self, row: int, col: int) -> int:
        """Get numerator (as Python int) at specified position"""
        pass
    
    @abstractmethod
    def get_big_integer_denominator_at(self, row: int, col: int) -> int:
        """Get denominator (as Python int) at specified position"""
        pass


# Forward declarations for matrix types (will be implemented in their respective modules)
class DoubleMatrix(ReadableDoubleMatrix[N]):
    """Forward declaration - actual implementation in double_matrix.py"""
    pass


class BigIntegerRationalMatrix(ReadableBigIntegerRationalMatrix[N]):
    """Forward declaration - actual implementation in bigint_rational_matrix.py"""
    pass