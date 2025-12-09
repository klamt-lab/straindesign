"""
MatrixOperations interface - Python port of ch.javasoft.smx.ops.MatrixOperations

This module provides the interface for matrix operations, similar to NumberOperations
but for matrices. Implementations provide factory methods for creating matrices and
basic matrix arithmetic operations.
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List

# Type variable for number types (BigFraction, float, etc.)
N = TypeVar('N')


class MatrixOperations(ABC, Generic[N]):
    """
    MatrixOperations interface - Python port of ch.javasoft.smx.ops.MatrixOperations<N>
    
    The MatrixOperations is similar to NumberOperations, but for matrices.
    Provides factory methods for creating matrices and basic matrix operations.
    """
    
    # Matrix creation methods
    @abstractmethod
    def create_readable_matrix_from_data(self, values: List[List[N]], rows_in_first_dim: bool = True):
        """Create readable matrix from 2D data array"""
        pass
    
    @abstractmethod
    def create_writable_matrix_from_data(self, values: List[List[N]], rows_in_first_dim: bool = True):
        """Create writable matrix from 2D data array"""
        pass
    
    @abstractmethod
    def create_readable_matrix(self, rows: int, cols: int):
        """Create empty readable matrix with given dimensions"""
        pass
    
    @abstractmethod
    def create_writable_matrix(self, rows: int, cols: int):
        """Create empty writable matrix with given dimensions"""
        pass
    
    # Vector creation methods (for future use - not needed by compression algorithms yet)
    @abstractmethod
    def create_readable_vector(self, values: List[N], column_vector: bool = True):
        """Create readable vector from data array"""
        pass
    
    @abstractmethod
    def create_writable_vector(self, values: List[N], column_vector: bool = True):
        """Create writable vector from data array"""
        pass
    
    @abstractmethod
    def create_readable_vector_empty(self, size: int, column_vector: bool = True):
        """Create empty readable vector with given size"""
        pass
    
    @abstractmethod
    def create_writable_vector_empty(self, size: int, column_vector: bool = True):
        """Create empty writable vector with given size"""
        pass
    
    # Basic matrix operations
    @abstractmethod
    def transpose(self, matrix):
        """Return transposed matrix"""
        pass
    
    @abstractmethod
    def negate(self, matrix):
        """Return negated matrix"""
        pass
    
    @abstractmethod
    def add_scalar(self, matrix, value: N):
        """Add scalar value to all matrix elements"""
        pass
    
    @abstractmethod
    def add_matrix(self, matrix_a, matrix_b):
        """Add two matrices element-wise"""
        pass
    
    @abstractmethod
    def subtract_scalar(self, matrix, value: N):
        """Subtract scalar value from all matrix elements"""
        pass
    
    @abstractmethod
    def subtract_matrix(self, matrix_a, matrix_b):
        """Subtract matrix_b from matrix_a element-wise"""
        pass
    
    @abstractmethod
    def multiply_scalar(self, matrix, value: N):
        """Multiply all matrix elements by scalar value"""
        pass
    
    @abstractmethod
    def multiply_matrix(self, matrix_a, matrix_b):
        """Multiply two matrices (matrix multiplication)"""
        pass
    
    # NumberOperations access
    @abstractmethod
    def get_number_operations(self):
        """Get the NumberOperations instance for this matrix's number type"""
        pass