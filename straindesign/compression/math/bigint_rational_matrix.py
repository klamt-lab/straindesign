"""
BigIntegerRationalMatrix interface - Python port of ch.javasoft.smx.iface.BigIntegerRationalMatrix

This module provides the main rational matrix interface that combines readable and writable
capabilities for matrices containing exact rational numbers (BigFraction values).
"""

from abc import abstractmethod
from typing import List
from .readable_bigint_rational_matrix import ReadableBigIntegerRationalMatrix
from .big_fraction import BigFraction


class RationalMatrix:
    """
    RationalMatrix interface - Python port of ch.javasoft.smx.iface.RationalMatrix
    
    Provides reduction operations for rational matrices - operations that simplify
    fractions by dividing numerators and denominators by their GCD.
    """
    
    @abstractmethod
    def reduce(self) -> bool:
        """
        Reduce the whole matrix, dividing numerators/denominators by their GCD.
        
        Returns:
            True if any value has been changed in the matrix
        """
        pass
    
    @abstractmethod
    def reduce_row(self, row: int) -> bool:
        """
        Reduce the specified row, dividing numerators/denominators by their GCD.
        
        Args:
            row: Row index to reduce
            
        Returns:
            True if any value has been changed in the given row
        """
        pass
    
    @abstractmethod
    def reduce_value_at(self, row: int, col: int) -> bool:
        """
        Reduce the specified value, dividing numerator/denominator by their GCD.
        
        Args:
            row: Row index
            col: Column index
            
        Returns:
            True if the value has been changed
        """
        pass


class WritableBigIntegerRationalMatrix:
    """
    WritableBigIntegerRationalMatrix interface - Python port of ch.javasoft.smx.iface.WritableBigIntegerRationalMatrix<N>
    
    Provides write operations for rational matrices with BigInteger precision.
    Note: In Python, we use int (arbitrary precision) instead of BigInteger.
    """
    
    @abstractmethod
    def set_value_at(self, row: int, col: int, value: BigFraction) -> None:
        """Set BigFraction value at specified position"""
        pass
    
    @abstractmethod
    def set_value_at_rational(self, row: int, col: int, numerator: int, denominator: int) -> None:
        """Set rational value at specified position using numerator/denominator"""
        pass
    
    @abstractmethod
    def add(self, row: int, col: int, numerator: int, denominator: int) -> None:
        """Add rational value to existing value at specified position"""
        pass
    
    @abstractmethod
    def multiply(self, row: int, col: int, numerator: int, denominator: int) -> None:
        """Multiply existing value at specified position by rational value"""
        pass
    
    @abstractmethod
    def multiply_row(self, row: int, numerator: int, denominator: int) -> None:
        """Multiply entire row by rational value"""
        pass
    
    @abstractmethod
    def add_row_to_other_row(self, src_row: int, src_numerator: int, src_denominator: int, 
                            dst_row: int, dst_numerator: int, dst_denominator: int) -> None:
        """Add source row (multiplied by src ratio) to destination row (multiplied by dst ratio)"""
        pass


class BigIntegerRationalMatrix(RationalMatrix, ReadableBigIntegerRationalMatrix, WritableBigIntegerRationalMatrix):
    """
    BigIntegerRationalMatrix interface - Python port of ch.javasoft.smx.iface.BigIntegerRationalMatrix
    
    Main rational matrix interface that combines readable and writable capabilities.
    This is the primary interface for exact rational arithmetic matrices used in
    the compression algorithms.
    """
    
    @abstractmethod
    def new_instance(self, rows: int, cols: int) -> 'BigIntegerRationalMatrix':
        """Create new rational matrix instance with given dimensions"""
        pass
    
    @abstractmethod
    def new_instance_from_data(self, data: List[List[BigFraction]], rows_in_dim1: bool = True) -> 'BigIntegerRationalMatrix':
        """Create new rational matrix instance from 2D BigFraction data"""
        pass
    
    @abstractmethod
    def clone(self) -> 'BigIntegerRationalMatrix':
        """Create deep copy of this rational matrix"""
        pass
    
    @abstractmethod
    def transpose(self) -> 'BigIntegerRationalMatrix':
        """Return transposed version of this rational matrix"""
        pass