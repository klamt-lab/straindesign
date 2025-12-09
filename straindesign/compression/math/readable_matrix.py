"""
ReadableMatrix interface - Python port of ch.javasoft.smx.iface.ReadableMatrix

This module provides the base interface for readable matrices, corresponding to
the Java ReadableMatrix<N> interface. Matrices can be read but not modified
through this interface.
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional
import copy
import io

# Type variable for number types (BigFraction, float, etc.)
N = TypeVar('N')


class MatrixBase(ABC, Generic[N]):
    """
    Base interface for matrices corresponding to Java MatrixBase<N>.
    Contains basic methods applicable to both readable and writable matrices.
    """
    
    @abstractmethod
    def clone(self) -> 'MatrixBase[N]':
        """Create a deep copy of this matrix"""
        pass
    
    @abstractmethod
    def new_instance(self, rows: int, cols: int) -> 'MatrixBase[N]':
        """Create new matrix instance with given dimensions"""
        pass
    
    @abstractmethod
    def new_instance_from_data(self, data: List[List[N]], rows_in_dim1: bool = True) -> 'MatrixBase[N]':
        """Create new matrix instance from 2D data array"""
        pass
    
    @abstractmethod
    def get_row_count(self) -> int:
        """Get number of rows in the matrix"""
        pass
    
    @abstractmethod
    def get_column_count(self) -> int:
        """Get number of columns in the matrix"""
        pass
    
    @abstractmethod
    def transpose(self) -> 'MatrixBase[N]':
        """Return transposed version of this matrix"""
        pass
    
    @abstractmethod
    def get_number_operations(self):
        """Get the NumberOperations instance for this matrix's number type"""
        pass
    
    @abstractmethod
    def get_matrix_operations(self):
        """Get the MatrixOperations instance for this matrix type"""
        pass
    
    # Output methods
    @abstractmethod
    def __str__(self) -> str:
        """Single line string representation"""
        pass
    
    @abstractmethod
    def to_multiline_string(self) -> str:
        """Multi-line string representation"""
        pass
    
    @abstractmethod
    def write_to(self, writer: io.TextIOBase) -> None:
        """Write single line representation to text writer"""
        pass
    
    @abstractmethod
    def write_to_multiline(self, writer: io.TextIOBase) -> None:
        """Write multi-line representation to text writer"""
        pass
    
    @abstractmethod
    def write_to_stream(self, stream: io.BytesIO) -> None:
        """Write to binary output stream"""
        pass
    
    @abstractmethod
    def write_to_multiline_stream(self, stream: io.BytesIO) -> None:
        """Write multi-line representation to binary output stream"""
        pass


class ReadableMatrix(MatrixBase[N]):
    """
    ReadableMatrix interface - Python port of ch.javasoft.smx.iface.ReadableMatrix<N>
    
    Contains methods for readable matrices of any data type. From readable matrices,
    data can be read, but not written to. Some implementations might implement both
    readable and writable interfaces.
    """
    
    @abstractmethod
    def clone(self) -> 'ReadableMatrix[N]':
        """Create a deep copy of this readable matrix"""
        pass
    
    @abstractmethod
    def new_instance(self, rows: int, cols: int) -> 'WritableMatrix[N]':
        """Create new writable matrix instance with given dimensions"""
        pass
    
    @abstractmethod
    def new_instance_from_data(self, data: List[List[N]], rows_in_dim1: bool = True) -> 'WritableMatrix[N]':
        """Create new writable matrix instance from 2D data array"""
        pass
    
    @abstractmethod
    def to_writable_matrix(self, enforce_new_instance: bool = False) -> 'WritableMatrix[N]':
        """Convert to writable matrix, optionally forcing new instance"""
        pass
    
    @abstractmethod
    def get_number_value_at(self, row: int, col: int) -> N:
        """Get the number value at the specified position"""
        pass
    
    @abstractmethod
    def get_number_rows(self) -> List[List[N]]:
        """Get all rows as a 2D list of numbers"""
        pass
    
    @abstractmethod
    def get_signum_at(self, row: int, col: int) -> int:
        """Get the sign (-1, 0, 1) of the value at the specified position"""
        pass
    
    @abstractmethod
    def transpose(self) -> 'ReadableMatrix[N]':
        """Return transposed version of this readable matrix"""
        pass


# Forward declaration for WritableMatrix (will be implemented in writable_matrix.py)
class WritableMatrix(ReadableMatrix[N]):
    """Forward declaration - actual implementation in writable_matrix.py"""
    pass