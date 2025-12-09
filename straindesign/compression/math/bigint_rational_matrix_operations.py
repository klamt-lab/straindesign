"""
BigIntegerRationalMatrixOperations - Python port of ch.javasoft.smx.ops.matrix.BigIntegerRationalMatrixOperations

This module provides concrete matrix operations for rational number matrices (BigFraction).
Implements both basic MatrixOperations and extended operations for compression algorithms.
"""

from typing import List
from .matrix_operations import MatrixOperations
from .readable_matrix import ReadableMatrix
from .default_bigint_rational_matrix import DefaultBigIntegerRationalMatrix
from .big_fraction import BigFraction
from .big_fraction_operations import BigFractionOperations


class BigIntegerRationalMatrixOperations(MatrixOperations[BigFraction]):
    """
    BigIntegerRationalMatrixOperations - Python port of ch.javasoft.smx.ops.matrix.BigIntegerRationalMatrixOperations
    
    Concrete implementation of MatrixOperations for rational number matrices.
    Uses singleton pattern like the Java version.
    """
    
    _instance = None
    
    def __init__(self):
        """Private constructor - use instance() method"""
        pass
    
    @classmethod
    def instance(cls) -> 'BigIntegerRationalMatrixOperations':
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    # Matrix creation methods
    def create_readable_matrix_from_data(self, values: List[List[BigFraction]], rows_in_first_dim: bool = True) -> DefaultBigIntegerRationalMatrix:
        """Create readable matrix from 2D BigFraction data"""
        return DefaultBigIntegerRationalMatrix(values, rows_in_dim1=rows_in_first_dim)
    
    def create_writable_matrix_from_data(self, values: List[List[BigFraction]], rows_in_first_dim: bool = True) -> DefaultBigIntegerRationalMatrix:
        """Create writable matrix from 2D BigFraction data"""
        return DefaultBigIntegerRationalMatrix(values, rows_in_dim1=rows_in_first_dim)
    
    def create_readable_matrix(self, rows: int, cols: int) -> DefaultBigIntegerRationalMatrix:
        """Create empty readable matrix with given dimensions"""
        return DefaultBigIntegerRationalMatrix(rows, cols)
    
    def create_writable_matrix(self, rows: int, cols: int) -> DefaultBigIntegerRationalMatrix:
        """Create empty writable matrix with given dimensions"""
        return DefaultBigIntegerRationalMatrix(rows, cols)
    
    # Vector creation methods (stubbed - not needed by compression algorithms yet)
    def create_readable_vector(self, values: List[BigFraction], column_vector: bool = True):
        """Create readable vector - TODO: implement when needed"""
        raise NotImplementedError("Vector operations not yet implemented - will add when compression algorithms require them")
    
    def create_writable_vector(self, values: List[BigFraction], column_vector: bool = True):
        """Create writable vector - TODO: implement when needed"""
        raise NotImplementedError("Vector operations not yet implemented - will add when compression algorithms require them")
    
    def create_readable_vector_empty(self, size: int, column_vector: bool = True):
        """Create empty readable vector - TODO: implement when needed"""
        raise NotImplementedError("Vector operations not yet implemented - will add when compression algorithms require them")
    
    def create_writable_vector_empty(self, size: int, column_vector: bool = True):
        """Create empty writable vector - TODO: implement when needed"""
        raise NotImplementedError("Vector operations not yet implemented - will add when compression algorithms require them")
    
    # Basic matrix operations
    def transpose(self, matrix: ReadableMatrix[BigFraction]) -> ReadableMatrix[BigFraction]:
        """Return transposed matrix"""
        return matrix.transpose()
    
    def negate(self, matrix: ReadableMatrix[BigFraction]) -> DefaultBigIntegerRationalMatrix:
        """Return negated matrix"""
        result = self.create_writable_matrix(matrix.get_row_count(), matrix.get_column_count())
        for row in range(matrix.get_row_count()):
            for col in range(matrix.get_column_count()):
                value = matrix.get_number_value_at(row, col)
                result.set_value_at(row, col, -value)
        return result
    
    def add_scalar(self, matrix: ReadableMatrix[BigFraction], value: BigFraction) -> DefaultBigIntegerRationalMatrix:
        """Add scalar value to all matrix elements"""
        result = self.create_writable_matrix(matrix.get_row_count(), matrix.get_column_count())
        for row in range(matrix.get_row_count()):
            for col in range(matrix.get_column_count()):
                existing_value = matrix.get_number_value_at(row, col)
                result.set_value_at(row, col, existing_value + value)
        return result
    
    def add_matrix(self, matrix_a: ReadableMatrix[BigFraction], matrix_b: ReadableMatrix[BigFraction]) -> DefaultBigIntegerRationalMatrix:
        """Add two matrices element-wise"""
        if matrix_a.get_row_count() != matrix_b.get_row_count() or matrix_a.get_column_count() != matrix_b.get_column_count():
            raise ValueError(f"Matrix dimensions don't match: {matrix_a.get_row_count()}x{matrix_a.get_column_count()} vs {matrix_b.get_row_count()}x{matrix_b.get_column_count()}")
        
        result = self.create_writable_matrix(matrix_a.get_row_count(), matrix_a.get_column_count())
        for row in range(matrix_a.get_row_count()):
            for col in range(matrix_a.get_column_count()):
                value_a = matrix_a.get_number_value_at(row, col)
                value_b = matrix_b.get_number_value_at(row, col)
                result.set_value_at(row, col, value_a + value_b)
        return result
    
    def subtract_scalar(self, matrix: ReadableMatrix[BigFraction], value: BigFraction) -> DefaultBigIntegerRationalMatrix:
        """Subtract scalar value from all matrix elements"""
        result = self.create_writable_matrix(matrix.get_row_count(), matrix.get_column_count())
        for row in range(matrix.get_row_count()):
            for col in range(matrix.get_column_count()):
                existing_value = matrix.get_number_value_at(row, col)
                result.set_value_at(row, col, existing_value - value)
        return result
    
    def subtract_matrix(self, matrix_a: ReadableMatrix[BigFraction], matrix_b: ReadableMatrix[BigFraction]) -> DefaultBigIntegerRationalMatrix:
        """Subtract matrix_b from matrix_a element-wise"""
        if matrix_a.get_row_count() != matrix_b.get_row_count() or matrix_a.get_column_count() != matrix_b.get_column_count():
            raise ValueError(f"Matrix dimensions don't match: {matrix_a.get_row_count()}x{matrix_a.get_column_count()} vs {matrix_b.get_row_count()}x{matrix_b.get_column_count()}")
        
        result = self.create_writable_matrix(matrix_a.get_row_count(), matrix_a.get_column_count())
        for row in range(matrix_a.get_row_count()):
            for col in range(matrix_a.get_column_count()):
                value_a = matrix_a.get_number_value_at(row, col)
                value_b = matrix_b.get_number_value_at(row, col)
                result.set_value_at(row, col, value_a - value_b)
        return result
    
    def multiply_scalar(self, matrix: ReadableMatrix[BigFraction], value: BigFraction) -> DefaultBigIntegerRationalMatrix:
        """Multiply all matrix elements by scalar value"""
        result = self.create_writable_matrix(matrix.get_row_count(), matrix.get_column_count())
        for row in range(matrix.get_row_count()):
            for col in range(matrix.get_column_count()):
                existing_value = matrix.get_number_value_at(row, col)
                result.set_value_at(row, col, existing_value * value)
        return result
    
    def multiply_matrix(self, matrix_a: ReadableMatrix[BigFraction], matrix_b: ReadableMatrix[BigFraction]) -> DefaultBigIntegerRationalMatrix:
        """Multiply two matrices (matrix multiplication)"""
        if matrix_a.get_column_count() != matrix_b.get_row_count():
            raise ValueError(f"Matrix dimensions incompatible for multiplication: {matrix_a.get_row_count()}x{matrix_a.get_column_count()} * {matrix_b.get_row_count()}x{matrix_b.get_column_count()}")
        
        result = self.create_writable_matrix(matrix_a.get_row_count(), matrix_b.get_column_count())
        
        # Standard matrix multiplication: C[i,j] = sum(A[i,k] * B[k,j])
        for row in range(matrix_a.get_row_count()):
            for col in range(matrix_b.get_column_count()):
                sum_value = BigFraction(0)
                for k in range(matrix_a.get_column_count()):
                    a_val = matrix_a.get_number_value_at(row, k)
                    b_val = matrix_b.get_number_value_at(k, col)
                    sum_value += a_val * b_val
                result.set_value_at(row, col, sum_value)
        
        return result
    
    # NumberOperations access
    def get_number_operations(self) -> BigFractionOperations:
        """Get BigFractionOperations instance"""
        return BigFractionOperations.instance()
    
    # Extended operations (for compression algorithms - will be implemented when we get to Gauss)
    def rank(self, matrix: ReadableMatrix[BigFraction]) -> int:
        """Compute matrix rank - TODO: implement when we add Gauss operations"""
        raise NotImplementedError("rank() requires Gauss operations - will implement in Level 4 Step 2")
    
    def nullity(self, matrix: ReadableMatrix[BigFraction]) -> int:
        """Compute matrix nullity - TODO: implement when we add Gauss operations"""
        raise NotImplementedError("nullity() requires Gauss operations - will implement in Level 4 Step 2")
    
    def invert(self, matrix: ReadableMatrix[BigFraction]) -> ReadableMatrix[BigFraction]:
        """Compute matrix inverse - TODO: implement when we add Gauss operations"""
        raise NotImplementedError("invert() requires Gauss operations - will implement in Level 4 Step 2")
    
    def nullspace(self, matrix: ReadableMatrix[BigFraction]) -> ReadableMatrix[BigFraction]:
        """Compute matrix nullspace - TODO: implement when we add Gauss operations"""
        raise NotImplementedError("nullspace() requires Gauss operations - will implement in Level 4 Step 2")