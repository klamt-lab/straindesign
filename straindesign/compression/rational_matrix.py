#!/usr/bin/env python3
"""
Rational matrix operations for exact arithmetic in network compression.

This module provides matrix operations using rational numbers (fractions),
replacing the Java BigIntegerRationalMatrix functionality from efmtool.
"""

from fractions import Fraction
from typing import Union, List, Tuple, Optional, Any
import numpy as np
from scipy import sparse
import copy

from .rational_math import RationalMath, ZERO, ONE


class RationalMatrix:
    """
    Matrix with rational number (Fraction) entries.
    
    This class implements the functionality of DefaultBigIntegerRationalMatrix
    from the Java efmtool library, providing exact arithmetic operations.
    """
    
    def __init__(self, rows: int, cols: int, sparse_mode: bool = True):
        """
        Initialize a rational matrix.
        
        Args:
            rows: Number of rows
            cols: Number of columns  
            sparse_mode: If True, use sparse storage for efficiency
        """
        self.rows = rows
        self.cols = cols
        self.sparse_mode = sparse_mode
        
        if sparse_mode:
            # Use dictionary for sparse storage
            self.data = {}  # (row, col) -> Fraction
        else:
            # Use numpy object array for dense storage
            self.data = np.zeros((rows, cols), dtype=object)
            for i in range(rows):
                for j in range(cols):
                    self.data[i, j] = ZERO
    
    @classmethod
    def from_numpy(cls, array: np.ndarray, sparse_mode: bool = True) -> 'RationalMatrix':
        """
        Create a RationalMatrix from a numpy array.
        
        Args:
            array: Numpy array to convert
            sparse_mode: If True, use sparse storage
            
        Returns:
            RationalMatrix with the same values
        """
        rows, cols = array.shape
        matrix = cls(rows, cols, sparse_mode)
        
        for i in range(rows):
            for j in range(cols):
                val = RationalMath.to_fraction(array[i, j])
                if not RationalMath.is_zero(val):
                    matrix.set_value_at(i, j, val)
        
        return matrix
    
    @classmethod
    def from_sparse(cls, sparse_matrix: sparse.spmatrix) -> 'RationalMatrix':
        """
        Create a RationalMatrix from a scipy sparse matrix.
        
        Args:
            sparse_matrix: Scipy sparse matrix
            
        Returns:
            RationalMatrix with the same values
        """
        rows, cols = sparse_matrix.shape
        matrix = cls(rows, cols, sparse_mode=True)
        
        # Convert to COO format for easy iteration
        coo = sparse_matrix.tocoo()
        for i, j, v in zip(coo.row, coo.col, coo.data):
            val = RationalMath.to_fraction(v)
            if not RationalMath.is_zero(val):
                matrix.set_value_at(i, j, val)
        
        return matrix
    
    def get_row_count(self) -> int:
        """Get number of rows."""
        return self.rows
    
    def get_column_count(self) -> int:
        """Get number of columns."""
        return self.cols
    
    def get_value_at(self, row: int, col: int) -> Fraction:
        """
        Get value at position (row, col).
        
        Args:
            row: Row index (0-based)
            col: Column index (0-based)
            
        Returns:
            Fraction value at the position
        """
        if self.sparse_mode:
            return self.data.get((row, col), ZERO)
        else:
            return self.data[row, col]
    
    def set_value_at(self, row: int, col: int, value: Union[Fraction, int, float]):
        """
        Set value at position (row, col).
        
        Args:
            row: Row index (0-based)
            col: Column index (0-based)
            value: Value to set (will be converted to Fraction)
        """
        val = RationalMath.to_fraction(value)
        
        if self.sparse_mode:
            if RationalMath.is_zero(val):
                # Remove zero entries from sparse storage
                self.data.pop((row, col), None)
            else:
                self.data[(row, col)] = val
        else:
            self.data[row, col] = val
    
    def get_signum_at(self, row: int, col: int) -> int:
        """
        Get sign of value at position.
        
        Args:
            row: Row index
            col: Column index
            
        Returns:
            -1, 0, or 1
        """
        val = self.get_value_at(row, col)
        return RationalMath.signum(val)
    
    def get_numerator_at(self, row: int, col: int) -> int:
        """Get numerator of value at position."""
        val = self.get_value_at(row, col)
        return val.numerator
    
    def get_denominator_at(self, row: int, col: int) -> int:
        """Get denominator of value at position."""
        val = self.get_value_at(row, col)
        return val.denominator
    
    def add_column_mult_to(self, source_col: int, source_mult: Fraction,
                          target_col: int, target_mult: Fraction):
        """
        Add multiple of source column to target column.
        
        Performs: target_col = target_mult * target_col + source_mult * source_col
        
        Args:
            source_col: Source column index
            source_mult: Multiplier for source column
            target_col: Target column index
            target_mult: Multiplier for target column
        """
        source_mult = RationalMath.to_fraction(source_mult)
        target_mult = RationalMath.to_fraction(target_mult)
        
        for row in range(self.rows):
            source_val = self.get_value_at(row, source_col)
            target_val = self.get_value_at(row, target_col)
            new_val = target_mult * target_val + source_mult * source_val
            self.set_value_at(row, target_col, new_val)
    
    def multiply_column(self, col: int, multiplier: Fraction):
        """
        Multiply a column by a scalar.
        
        Args:
            col: Column index
            multiplier: Scalar multiplier
        """
        mult = RationalMath.to_fraction(multiplier)
        
        for row in range(self.rows):
            val = self.get_value_at(row, col)
            self.set_value_at(row, col, mult * val)
    
    def swap_columns(self, col1: int, col2: int):
        """Swap two columns."""
        if col1 == col2:
            return
            
        for row in range(self.rows):
            val1 = self.get_value_at(row, col1)
            val2 = self.get_value_at(row, col2)
            self.set_value_at(row, col1, val2)
            self.set_value_at(row, col2, val1)
    
    def swap_rows(self, row1: int, row2: int):
        """Swap two rows."""
        if row1 == row2:
            return
            
        for col in range(self.cols):
            val1 = self.get_value_at(row1, col)
            val2 = self.get_value_at(row2, col)
            self.set_value_at(row1, col, val2)
            self.set_value_at(row2, col, val1)
    
    def remove_row(self, row: int):
        """
        Remove a row from the matrix.
        
        Args:
            row: Row index to remove
        """
        if self.sparse_mode:
            # Create new dictionary without the row
            new_data = {}
            for (r, c), val in self.data.items():
                if r < row:
                    new_data[(r, c)] = val
                elif r > row:
                    new_data[(r - 1, c)] = val
            self.data = new_data
        else:
            # Remove row from dense array
            self.data = np.delete(self.data, row, axis=0)
        
        self.rows -= 1
    
    def remove_column(self, col: int):
        """
        Remove a column from the matrix.
        
        Args:
            col: Column index to remove
        """
        if self.sparse_mode:
            # Create new dictionary without the column
            new_data = {}
            for (r, c), val in self.data.items():
                if c < col:
                    new_data[(r, c)] = val
                elif c > col:
                    new_data[(r, c - 1)] = val
            self.data = new_data
        else:
            # Remove column from dense array
            self.data = np.delete(self.data, col, axis=1)
        
        self.cols -= 1
    
    def get_row(self, row: int) -> List[Fraction]:
        """Get a row as a list of Fractions."""
        return [self.get_value_at(row, c) for c in range(self.cols)]
    
    def get_column(self, col: int) -> List[Fraction]:
        """Get a column as a list of Fractions."""
        return [self.get_value_at(r, col) for r in range(self.rows)]
    
    def to_numpy(self, as_float: bool = False) -> np.ndarray:
        """
        Convert to numpy array.
        
        Args:
            as_float: If True, convert to float array; if False, return object array with Fractions
            
        Returns:
            Numpy array representation
        """
        if as_float:
            result = np.zeros((self.rows, self.cols), dtype=float)
            for i in range(self.rows):
                for j in range(self.cols):
                    result[i, j] = float(self.get_value_at(i, j))
        else:
            result = np.zeros((self.rows, self.cols), dtype=object)
            for i in range(self.rows):
                for j in range(self.cols):
                    result[i, j] = self.get_value_at(i, j)
        
        return result
    
    def get_double_rows(self) -> List[List[float]]:
        """
        Get matrix as list of lists of floats.
        
        This matches the Java getDoubleRows() method.
        
        Returns:
            List of rows, each row as list of floats
        """
        result = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(float(self.get_value_at(i, j)))
            result.append(row)
        return result
    
    def transpose(self) -> 'RationalMatrix':
        """
        Return transposed matrix.
        
        Returns:
            New RationalMatrix that is the transpose
        """
        result = RationalMatrix(self.cols, self.rows, self.sparse_mode)
        
        if self.sparse_mode:
            for (r, c), val in self.data.items():
                result.set_value_at(c, r, val)
        else:
            for i in range(self.rows):
                for j in range(self.cols):
                    result.set_value_at(j, i, self.get_value_at(i, j))
        
        return result
    
    def copy(self) -> 'RationalMatrix':
        """Create a deep copy of the matrix."""
        result = RationalMatrix(self.rows, self.cols, self.sparse_mode)
        
        if self.sparse_mode:
            result.data = self.data.copy()
        else:
            result.data = copy.deepcopy(self.data)
        
        return result
    
    def multiply(self, other: 'RationalMatrix') -> 'RationalMatrix':
        """
        Multiply this matrix by another matrix: self * other
        
        Args:
            other: Matrix to multiply by (must have compatible dimensions)
            
        Returns:
            Result of matrix multiplication
        """
        if self.cols != other.rows:
            raise ValueError(f"Matrix dimensions incompatible: {self.rows}×{self.cols} * {other.rows}×{other.cols}")
        
        result = RationalMatrix(self.rows, other.cols, self.sparse_mode)
        
        # Perform matrix multiplication
        for i in range(self.rows):
            for j in range(other.cols):
                sum_val = ZERO
                for k in range(self.cols):
                    a_ik = self.get_value_at(i, k)
                    b_kj = other.get_value_at(k, j)
                    if not RationalMath.is_zero(a_ik) and not RationalMath.is_zero(b_kj):
                        sum_val += a_ik * b_kj
                
                if not RationalMath.is_zero(sum_val):
                    result.set_value_at(i, j, sum_val)
        
        return result
    
    def swap_columns(self, col1: int, col2: int):
        """Swap two columns in the matrix."""
        for row in range(self.rows):
            val1 = self.get_value_at(row, col1)
            val2 = self.get_value_at(row, col2)
            self.set_value_at(row, col1, val2)
            self.set_value_at(row, col2, val1)
    
    def swap_rows(self, row1: int, row2: int):
        """Swap two rows in the matrix."""
        for col in range(self.cols):
            val1 = self.get_value_at(row1, col)
            val2 = self.get_value_at(row2, col)
            self.set_value_at(row1, col, val2)
            self.set_value_at(row2, col, val1)
    
    def sub_matrix(self, start_row: int, end_row: int, start_col: int, end_col: int) -> 'RationalMatrix':
        """Extract submatrix from start_row:end_row, start_col:end_col."""
        rows = end_row - start_row
        cols = end_col - start_col
        result = RationalMatrix(rows, cols, self.sparse_mode)
        
        for i in range(rows):
            for j in range(cols):
                val = self.get_value_at(start_row + i, start_col + j)
                if not RationalMath.is_zero(val):
                    result.set_value_at(i, j, val)
        
        return result


class GaussElimination:
    """
    Gaussian elimination for rational matrices.
    
    This implements the functionality of ch.javasoft.smx.ops.Gauss
    for exact rational arithmetic.
    """
    
    @staticmethod
    def row_echelon(matrix: RationalMatrix, reduced: bool = False,
                    row_mapping: Optional[List[int]] = None,
                    col_mapping: Optional[List[int]] = None) -> int:
        """
        Perform Gaussian elimination to row echelon form.
        
        Args:
            matrix: RationalMatrix to reduce (modified in place)
            reduced: If True, create reduced row echelon form
            row_mapping: Optional list to store row permutations
            col_mapping: Optional list to store column permutations
            
        Returns:
            Rank of the matrix
        """
        rows = matrix.get_row_count()
        cols = matrix.get_column_count()
        
        # Initialize mappings if not provided
        if row_mapping is None:
            row_mapping = list(range(rows))
        if col_mapping is None:
            col_mapping = list(range(cols))
        
        pivot_row = 0
        pivot_col = 0
        
        while pivot_row < rows and pivot_col < cols:
            # Find pivot element (non-zero)
            found_pivot = False
            
            # Look for non-zero pivot in current column
            for r in range(pivot_row, rows):
                if not RationalMath.is_zero(matrix.get_value_at(r, pivot_col)):
                    if r != pivot_row:
                        # Swap rows
                        matrix.swap_rows(pivot_row, r)
                        row_mapping[pivot_row], row_mapping[r] = row_mapping[r], row_mapping[pivot_row]
                    found_pivot = True
                    break
            
            if not found_pivot:
                # No pivot in this column, move to next column
                pivot_col += 1
                continue
            
            # Get pivot value
            pivot_val = matrix.get_value_at(pivot_row, pivot_col)
            
            if reduced:
                # Scale pivot row to make pivot = 1
                if pivot_val != ONE:
                    for c in range(cols):
                        val = matrix.get_value_at(pivot_row, c)
                        matrix.set_value_at(pivot_row, c, val / pivot_val)
                    pivot_val = ONE
            
            # Eliminate column entries
            start_row = pivot_row + 1 if not reduced else 0
            for r in range(start_row, rows):
                if r == pivot_row:
                    continue
                
                val = matrix.get_value_at(r, pivot_col)
                if not RationalMath.is_zero(val):
                    # Eliminate this entry
                    multiplier = -val / pivot_val
                    for c in range(pivot_col, cols):
                        current = matrix.get_value_at(r, c)
                        pivot_row_val = matrix.get_value_at(pivot_row, c)
                        matrix.set_value_at(r, c, current + multiplier * pivot_row_val)
            
            pivot_row += 1
            pivot_col += 1
        
        return pivot_row  # This is the rank
    
    @staticmethod
    def nullspace(matrix: RationalMatrix) -> RationalMatrix:
        """
        Compute the nullspace (kernel) of a matrix.
        
        Args:
            matrix: Input matrix
            
        Returns:
            Matrix whose columns form a basis for the nullspace
        """
        rows = matrix.get_row_count()
        cols = matrix.get_column_count()
        
        # Create augmented matrix [A | I]
        augmented = RationalMatrix(rows, cols + rows, matrix.sparse_mode)
        
        # Copy original matrix
        for i in range(rows):
            for j in range(cols):
                augmented.set_value_at(i, j, matrix.get_value_at(i, j))
        
        # Add identity matrix
        for i in range(rows):
            augmented.set_value_at(i, cols + i, ONE)
        
        # Perform row reduction
        col_mapping = list(range(cols + rows))
        rank = GaussElimination.row_echelon(augmented, reduced=True, col_mapping=col_mapping)
        
        # Extract nullspace basis
        free_vars = []
        pivot_cols = []
        
        # Identify pivot and free columns
        row = 0
        for col in range(cols):
            if row < rank:
                # Check if this is a pivot column
                is_pivot = False
                for r in range(row, min(row + 1, rows)):
                    if not RationalMath.is_zero(augmented.get_value_at(r, col)):
                        is_pivot = True
                        pivot_cols.append(col)
                        row += 1
                        break
                
                if not is_pivot:
                    free_vars.append(col)
            else:
                free_vars.append(col)
        
        # Build nullspace matrix
        nullspace_dim = len(free_vars)
        if nullspace_dim == 0:
            # Empty nullspace
            return RationalMatrix(cols, 0, matrix.sparse_mode)
        
        kernel = RationalMatrix(cols, nullspace_dim, matrix.sparse_mode)
        
        for i, free_col in enumerate(free_vars):
            # Set free variable to 1
            kernel.set_value_at(free_col, i, ONE)
            
            # Set dependent variables based on row reduction
            for row in range(min(rank, rows)):
                for col in range(cols):
                    if not RationalMath.is_zero(augmented.get_value_at(row, col)):
                        # This is a pivot column
                        val = augmented.get_value_at(row, free_col)
                        kernel.set_value_at(col, i, -val)
                        break
        
        return kernel
    
    @staticmethod
    def basic_columns(matrix: RationalMatrix) -> List[int]:
        """
        Find basic (linearly independent) columns.
        
        Args:
            matrix: Input matrix
            
        Returns:
            List of column indices that form a basis
        """
        rows = matrix.get_row_count()
        cols = matrix.get_column_count()
        
        # Create a copy for row reduction
        work_matrix = matrix.copy()
        
        # Track column permutations
        col_map = list(range(cols))
        
        # Perform row reduction
        rank = GaussElimination.row_echelon(work_matrix, reduced=False, col_mapping=col_map)
        
        # Find pivot columns
        basic_cols = []
        row = 0
        
        for col in range(cols):
            if row < rank and row < rows:
                # Check if this column has a pivot
                if not RationalMath.is_zero(work_matrix.get_value_at(row, col)):
                    basic_cols.append(col_map[col])
                    row += 1
        
        return basic_cols[:rank]