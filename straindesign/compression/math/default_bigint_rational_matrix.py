"""
DefaultBigIntegerRationalMatrix - Python port of ch.javasoft.smx.impl.DefaultBigIntegerRationalMatrix

This module provides the main concrete implementation of rational matrices using exact
BigFraction arithmetic. The matrix is stored as two flat arrays (numerators, denominators)
in row-major order for efficient memory usage and operations.
"""

import copy
import io
from typing import List, Optional, Union
import math
from fractions import Fraction

from .bigint_rational_matrix import BigIntegerRationalMatrix
from .readable_bigint_rational_matrix import ReadableBigIntegerRationalMatrix
from .big_fraction import BigFraction


class DefaultBigIntegerRationalMatrix(BigIntegerRationalMatrix):
    """
    DefaultBigIntegerRationalMatrix - Python port of ch.javasoft.smx.impl.DefaultBigIntegerRationalMatrix
    
    Main concrete implementation of rational matrices. Stores matrix data as two flat arrays
    of numerators and denominators in row-major order, providing exact rational arithmetic.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize matrix with various input types.
        
        Supported signatures:
        - DefaultBigIntegerRationalMatrix(rows, cols) - zero matrix
        - DefaultBigIntegerRationalMatrix(readable_matrix) - copy constructor
        - DefaultBigIntegerRationalMatrix(data, rows_in_dim1=True) - from 2D data
        - DefaultBigIntegerRationalMatrix(numerators, denominators, rows, cols) - from arrays
        """
        if len(args) == 2 and isinstance(args[0], int) and isinstance(args[1], int):
            # Constructor: (rows, cols)
            self._init_zero_matrix(args[0], args[1])
        elif len(args) == 1 and hasattr(args[0], 'get_row_count'):
            # Constructor: (readable_matrix)
            self._init_from_readable_matrix(args[0])
        elif len(args) == 1 and isinstance(args[0], list) and len(args[0]) > 0 and isinstance(args[0][0], list):
            # Constructor: (data) - single 2D list argument
            rows_in_dim1 = kwargs.get('rows_in_dim1', True)
            self._init_from_2d_data(args[0], rows_in_dim1)
        elif len(args) == 2 and isinstance(args[0], list) and len(args[0]) > 0 and isinstance(args[0][0], list):
            # Constructor: (data, rows_in_dim1)
            rows_in_dim1 = kwargs.get('rows_in_dim1', True)
            self._init_from_2d_data(args[0], rows_in_dim1)
        elif len(args) == 4:
            # Constructor: (numerators, denominators, rows, cols)
            self._init_from_arrays(args[0], args[1], args[2], args[3])
        else:
            raise ValueError(f"Invalid constructor arguments: {args}")
    
    def _init_zero_matrix(self, row_count: int, col_count: int):
        """Initialize as zero matrix with given dimensions"""
        if row_count < 0:
            raise ValueError(f"negative row count: {row_count}")
        if col_count < 0:
            raise ValueError(f"negative column count: {col_count}")
            
        vals = row_count * col_count
        self._row_count = row_count
        self._column_count = col_count
        self._numerators = [0] * vals     # Python int (replaces BigInteger.ZERO)
        self._denominators = [1] * vals   # Python int (replaces BigInteger.ONE)
    
    def _init_from_readable_matrix(self, mx):
        """Initialize from readable matrix (copy constructor)"""
        self._init_zero_matrix(mx.get_row_count(), mx.get_column_count())
        for row in range(mx.get_row_count()):
            for col in range(mx.get_column_count()):
                if hasattr(mx, 'get_big_integer_numerator_at'):
                    # ReadableBigIntegerRationalMatrix
                    num = mx.get_big_integer_numerator_at(row, col)
                    den = mx.get_big_integer_denominator_at(row, col)
                    self.set_value_at_rational(row, col, num, den)
                else:
                    # ReadableMatrix<BigFraction>
                    value = mx.get_number_value_at(row, col)
                    self.set_value_at(row, col, value)
    
    def _init_from_2d_data(self, data: List[List], rows_in_dim1: bool = True):
        """Initialize from 2D data array"""
        if rows_in_dim1:
            rows = len(data)
            cols = len(data[0]) if rows > 0 else 0
        else:
            cols = len(data)
            rows = len(data[0]) if cols > 0 else 0
            
        self._init_zero_matrix(rows, cols)
        
        for row in range(rows):
            for col in range(cols):
                if rows_in_dim1:
                    value = data[row][col]
                else:
                    value = data[col][row]
                    
                if isinstance(value, BigFraction):
                    self.set_value_at(row, col, value)
                elif isinstance(value, (int, float)):
                    self.set_value_at(row, col, BigFraction(value))
                else:
                    self.set_value_at(row, col, BigFraction(str(value)))
    
    def _init_from_arrays(self, numerators: List[int], denominators: List[int], 
                         row_count: int, col_count: int):
        """Initialize from numerator/denominator arrays"""
        expected_len = row_count * col_count
        if len(numerators) != expected_len:
            raise ValueError(f"expected {expected_len} numerators, but found {len(numerators)}")
        if len(denominators) != expected_len:
            raise ValueError(f"expected {expected_len} denominators, but found {len(denominators)}")
            
        self._numerators = numerators[:]  # Copy the arrays
        self._denominators = denominators[:]
        self._row_count = row_count
        self._column_count = col_count
    
    # Core interface methods
    def get_number_operations(self):
        """Get BigFractionOperations instance"""
        return BigFractionOperations.instance()
    
    def get_matrix_operations(self):
        """Get matrix operations instance"""
        from .bigint_rational_matrix_operations import BigIntegerRationalMatrixOperations
        return BigIntegerRationalMatrixOperations.instance()
    
    def get_row_count(self) -> int:
        """Get number of rows"""
        return self._row_count
    
    def get_column_count(self) -> int:
        """Get number of columns"""
        return self._column_count
    
    def get_big_fraction_value_at(self, row: int, col: int) -> BigFraction:
        """Get BigFraction value at specified position"""
        return BigFraction(
            self.get_big_integer_numerator_at(row, col),
            self.get_big_integer_denominator_at(row, col)
        )
    
    def get_big_integer_numerator_at(self, row: int, col: int) -> int:
        """Get numerator at specified position"""
        return self._numerators[row * self._column_count + col]
    
    def get_big_integer_denominator_at(self, row: int, col: int) -> int:
        """Get denominator at specified position"""
        return self._denominators[row * self._column_count + col]
    
    def get_number_value_at(self, row: int, col: int) -> BigFraction:
        """Get number value (BigFraction) at specified position"""
        return self.get_big_fraction_value_at(row, col)
    
    def get_signum_at(self, row: int, col: int) -> int:
        """Get sign (-1, 0, 1) of value at specified position"""
        num = self.get_big_integer_numerator_at(row, col)
        den = self.get_big_integer_denominator_at(row, col)
        
        # Python int signum calculation
        def signum(x):
            return 0 if x == 0 else (1 if x > 0 else -1)
        
        return signum(num) * signum(den)
    
    # Write operations
    def set_value_at(self, row: int, col: int, value: BigFraction) -> None:
        """Set BigFraction value at specified position"""
        self.set_value_at_rational(row, col, value.numerator, value.denominator)
    
    def set_value_at_rational(self, row: int, col: int, numerator: int, denominator: int) -> None:
        """Set rational value using numerator/denominator"""
        index = row * self._column_count + col
        self._numerators[index] = numerator
        self._denominators[index] = denominator
    
    def add(self, row: int, col: int, numerator: int, denominator: int) -> None:
        """Add rational value to existing value at position"""
        current = self.get_big_fraction_value_at(row, col)
        add_val = BigFraction(numerator, denominator)
        result = current + add_val  # Auto-reduces due to fractions.Fraction
        self.set_value_at_rational(row, col, result.numerator, result.denominator)
    
    def multiply(self, row: int, col: int, numerator: int, denominator: int) -> None:
        """Multiply existing value at position by rational value"""
        current = self.get_big_fraction_value_at(row, col)
        mult_val = BigFraction(numerator, denominator)
        result = current * mult_val  # Auto-reduces
        self.set_value_at_rational(row, col, result.numerator, result.denominator)
    
    def multiply_row(self, row: int, numerator: int, denominator: int) -> None:
        """Multiply entire row by rational value"""
        mult_val = BigFraction(numerator, denominator)
        for col in range(self.get_column_count()):
            current = self.get_big_fraction_value_at(row, col)
            result = current * mult_val
            self.set_value_at_rational(row, col, result.numerator, result.denominator)
    
    def add_row_to_other_row(self, src_row: int, src_numerator: int, src_denominator: int,
                            dst_row: int, dst_numerator: int, dst_denominator: int) -> None:
        """Add source row (multiplied by src ratio) to destination row (multiplied by dst ratio)"""
        cols = self.get_column_count()
        for col in range(cols):
            src_val = self.get_big_fraction_value_at(src_row, col)
            dst_val = self.get_big_fraction_value_at(dst_row, col)
            
            src_mult = src_val * BigFraction(src_numerator, src_denominator)
            dst_mult = dst_val * BigFraction(dst_numerator, dst_denominator)
            result = dst_mult + src_mult  # Auto-reduces
            
            self.set_value_at_rational(dst_row, col, result.numerator, result.denominator)
    
    # Reduction operations (from RationalMatrix interface)
    def reduce(self) -> bool:
        """Reduce whole matrix by dividing by GCD - returns True if any change"""
        changed = False
        for row in range(self.get_row_count()):
            for col in range(self.get_column_count()):
                changed |= self.reduce_value_at(row, col)
        return changed
    
    def reduce_row(self, row: int) -> bool:
        """Reduce specified row - returns True if any change"""
        changed = False
        for col in range(self.get_column_count()):
            changed |= self.reduce_value_at(row, col)
        return changed
    
    def reduce_value_at(self, row: int, col: int) -> bool:
        """Reduce value at position - returns True if changed"""
        # Note: With auto-reducing fractions, this is mostly a no-op
        # But we implement the logic for compatibility
        num = self.get_big_integer_numerator_at(row, col)
        den = self.get_big_integer_denominator_at(row, col)
        
        if num == 0:
            if num != 0 or den != 1:
                self.set_value_at_rational(row, col, 0, 1)
                return True
            return False
        
        gcd = math.gcd(abs(num), abs(den))
        if gcd != 1:
            if den < 0:
                gcd = -gcd
            
            new_num = num // gcd
            new_den = den // gcd
            self.set_value_at_rational(row, col, new_num, new_den)
            return True
        
        return False
    
    # Matrix operations
    def clone(self) -> 'DefaultBigIntegerRationalMatrix':
        """Create deep copy of this matrix"""
        return DefaultBigIntegerRationalMatrix(self)
    
    def new_instance(self, rows: int, cols: int) -> 'DefaultBigIntegerRationalMatrix':
        """Create new matrix instance with given dimensions"""
        return DefaultBigIntegerRationalMatrix(rows, cols)
    
    def new_instance_from_data(self, data: List[List[BigFraction]], 
                              rows_in_dim1: bool = True) -> 'DefaultBigIntegerRationalMatrix':
        """Create new matrix from 2D BigFraction data"""
        return DefaultBigIntegerRationalMatrix(data, rows_in_dim1=rows_in_dim1)
    
    def transpose(self) -> 'DefaultBigIntegerRationalMatrix':
        """Return transposed matrix"""
        rows = self.get_row_count()
        cols = self.get_column_count()
        result = DefaultBigIntegerRationalMatrix(cols, rows)
        
        for row in range(rows):
            for col in range(cols):
                result.set_value_at_rational(
                    col, row,
                    self.get_big_integer_numerator_at(row, col),
                    self.get_big_integer_denominator_at(row, col)
                )
        
        return result
    
    # String representation
    def __str__(self) -> str:
        """Single line string representation"""
        return self._matrix_to_string("{", " }", " [", "]", "", "", "", ", ")
    
    def to_multiline_string(self) -> str:
        """Multi-line string representation"""
        return self._matrix_to_string("{\n", "}\n", " [", "]\n", "", " ", " ", ",")
    
    def _matrix_to_string(self, prefix: str, postfix: str, row_prefix: str, 
                         row_postfix: str, row_separator: str, col_prefix: str, 
                         col_postfix: str, col_separator: str) -> str:
        """Internal string formatting method"""
        result = [prefix]
        
        for row in range(self.get_row_count()):
            if row > 0:
                result.append(row_separator)
            result.append(row_prefix)
            
            for col in range(self.get_column_count()):
                if col > 0:
                    result.append(col_separator)
                result.append(col_prefix)
                result.append(str(self.get_big_fraction_value_at(row, col)))
                result.append(col_postfix)
                
            result.append(row_postfix)
            
        result.append(postfix)
        return ''.join(result)
    
    # Output methods (basic implementations)
    def write_to(self, writer: io.TextIOBase) -> None:
        """Write single line representation to text writer"""
        writer.write(str(self))
    
    def write_to_multiline(self, writer: io.TextIOBase) -> None:
        """Write multi-line representation to text writer"""
        size_str = f"{self.get_row_count()}x{self.get_column_count()}"
        writer.write(f"{size_str} {self.to_multiline_string()}")
    
    def write_to_stream(self, stream: io.BytesIO) -> None:
        """Write to binary output stream"""
        stream.write(str(self).encode('utf-8'))
    
    def write_to_multiline_stream(self, stream: io.BytesIO) -> None:
        """Write multi-line representation to binary output stream"""
        size_str = f"{self.get_row_count()}x{self.get_column_count()}"
        stream.write(f"{size_str} {self.to_multiline_string()}".encode('utf-8'))
    
    # Double conversion methods
    def get_double_value_at(self, row: int, col: int) -> float:
        """Get double precision value at specified position"""
        fraction = self.get_big_fraction_value_at(row, col)
        return fraction.double_value()
    
    def get_double_row(self, row: int) -> List[float]:
        """Get specified row as list of doubles"""
        return [self.get_double_value_at(row, col) for col in range(self.get_column_count())]
    
    def get_double_column(self, col: int) -> List[float]:
        """Get specified column as list of doubles"""
        return [self.get_double_value_at(row, col) for row in range(self.get_row_count())]
    
    def get_double_rows(self) -> List[List[float]]:
        """Get all rows as 2D list of doubles"""
        return [self.get_double_row(row) for row in range(self.get_row_count())]
    
    def get_double_columns(self) -> List[List[float]]:
        """Get all columns as 2D list of doubles"""
        return [self.get_double_column(col) for col in range(self.get_column_count())]
    
    def to_double_array(self) -> List[float]:
        """Convert matrix to 1D list of doubles (row-major order)"""
        result = []
        for row in range(self.get_row_count()):
            for col in range(self.get_column_count()):
                result.append(self.get_double_value_at(row, col))
        return result
    
    def to_array(self, array: List[float]) -> None:
        """Fill provided array with matrix values (row-major order)"""
        expected_len = self.get_row_count() * self.get_column_count()
        if len(array) != expected_len:
            raise ValueError(f"expected array length {expected_len} but found {len(array)}")
        
        idx = 0
        for row in range(self.get_row_count()):
            for col in range(self.get_column_count()):
                array[idx] = self.get_double_value_at(row, col)
                idx += 1
    
    def get_number_rows(self) -> List[List[BigFraction]]:
        """Get all rows as 2D list of BigFractions"""
        result = []
        for row in range(self.get_row_count()):
            row_data = []
            for col in range(self.get_column_count()):
                row_data.append(self.get_big_fraction_value_at(row, col))
            result.append(row_data)
        return result
    
    # Submatrix extraction
    def sub_big_integer_rational_matrix(self, row_start: int, row_end: int, 
                                       col_start: int, col_end: int) -> 'DefaultBigIntegerRationalMatrix':
        """Extract submatrix as rational matrix"""
        rows = self.get_row_count()
        cols = self.get_column_count()
        
        if row_end < row_start:
            raise ValueError("row_end < row_start")
        if col_end < col_start:
            raise ValueError("col_end < col_start")
        if row_start < 0:
            raise ValueError("row_start < 0")
        if col_start < 0:
            raise ValueError("col_start < 0")
        if row_end > rows:
            raise ValueError("row_end > get_row_count()")
        if col_end > cols:
            raise ValueError("col_end > get_column_count()")
        
        new_rows = row_end - row_start
        new_cols = col_end - col_start
        new_vals = new_rows * new_cols
        
        numerators = [0] * new_vals
        denominators = [1] * new_vals
        
        for row in range(new_rows):
            for col in range(new_cols):
                src_idx = (row_start + row) * cols + (col_start + col)
                dst_idx = row * new_cols + col
                numerators[dst_idx] = self._numerators[src_idx]
                denominators[dst_idx] = self._denominators[src_idx]
        
        return DefaultBigIntegerRationalMatrix(numerators, denominators, new_rows, new_cols)
    
    # Additional conversion methods
    def to_big_integer_rational_matrix(self, enforce_new_instance: bool = False) -> 'DefaultBigIntegerRationalMatrix':
        """Convert to rational matrix (self or copy)"""
        return self.clone() if enforce_new_instance else self
    
    def to_writable_matrix(self, enforce_new_instance: bool = False) -> 'DefaultBigIntegerRationalMatrix':
        """Convert to writable matrix"""
        return self.to_big_integer_rational_matrix(enforce_new_instance)
    
    # Row/column operations
    def swap_rows(self, row_a: int, row_b: int) -> None:
        """Swap two rows"""
        if row_a == row_b:
            return
        
        cols = self.get_column_count()
        
        # Swap numerators
        for col in range(cols):
            idx_a = row_a * cols + col
            idx_b = row_b * cols + col
            self._numerators[idx_a], self._numerators[idx_b] = self._numerators[idx_b], self._numerators[idx_a]
            self._denominators[idx_a], self._denominators[idx_b] = self._denominators[idx_b], self._denominators[idx_a]
    
    def swap_columns(self, col_a: int, col_b: int) -> None:
        """Swap two columns"""
        if col_a == col_b:
            return
        
        rows = self.get_row_count()
        for row in range(rows):
            num_a = self.get_big_integer_numerator_at(row, col_a)
            den_a = self.get_big_integer_denominator_at(row, col_a)
            num_b = self.get_big_integer_numerator_at(row, col_b)
            den_b = self.get_big_integer_denominator_at(row, col_b)
            
            self.set_value_at_rational(row, col_a, num_b, den_b)
            self.set_value_at_rational(row, col_b, num_a, den_a)
    
    def negate(self, row: int, col: int) -> None:
        """Negate value at specified position"""
        value = self.get_big_fraction_value_at(row, col)
        if value.signum() == 0:
            self.set_value_at(row, col, BigFraction(0))
        else:
            self.set_value_at(row, col, -value)
    
    # Convenience methods for different numeric types
    def set_value_at_int(self, row: int, col: int, value: int) -> None:
        """Set integer value (denominator = 1)"""
        self.set_value_at_rational(row, col, value, 1)
    
    def set_value_at_double(self, row: int, col: int, value: float) -> None:
        """Set double value"""
        fraction = BigFraction(value)
        self.set_value_at(row, col, fraction)
    
    def add_double(self, row: int, col: int, value: float) -> None:
        """Add double value to existing value"""
        current = self.get_big_fraction_value_at(row, col)
        add_val = BigFraction(value)
        result = current + add_val
        self.set_value_at(row, col, result)
    
    def multiply_double(self, row: int, col: int, factor: float) -> None:
        """Multiply existing value by double factor"""
        current = self.get_big_fraction_value_at(row, col)
        mult_val = BigFraction(factor)
        result = current * mult_val
        self.set_value_at(row, col, result)
    
    def multiply_row_double(self, row: int, factor: float) -> None:
        """Multiply entire row by double factor"""
        mult_val = BigFraction(factor)
        for col in range(self.get_column_count()):
            current = self.get_big_fraction_value_at(row, col)
            result = current * mult_val
            self.set_value_at(row, col, result)
    
    def add_row_to_other_row_double(self, src_row: int, src_factor: float,
                                   dst_row: int, dst_factor: float) -> None:
        """Add source row (multiplied by src_factor) to destination row (multiplied by dst_factor)"""
        src_mult = BigFraction(src_factor)
        dst_mult = BigFraction(dst_factor)
        
        for col in range(self.get_column_count()):
            src_val = self.get_big_fraction_value_at(src_row, col)
            dst_val = self.get_big_fraction_value_at(dst_row, col)
            
            result = (dst_val * dst_mult) + (src_val * src_mult)
            self.set_value_at(dst_row, col, result)
    
    # Missing abstract methods from interface hierarchy
    def to_double_matrix(self, enforce_new_instance: bool = False):
        """Convert to double matrix - TODO: implement when DoubleMatrix is needed"""
        # This will be implemented when we need DoubleMatrix functionality
        raise NotImplementedError("DoubleMatrix not yet implemented")
    
    def sub_double_matrix(self, row_start: int, row_end: int, col_start: int, col_end: int):
        """Extract submatrix as double matrix - TODO: implement when DoubleMatrix is needed"""
        # For now, extract as rational matrix then convert
        return self.sub_big_integer_rational_matrix(row_start, row_end, col_start, col_end).to_double_matrix(False)
    
    @staticmethod
    def from_fractions(data: List[List[Union[BigFraction, int, float]]], rows_in_dim1: bool = True) -> 'DefaultBigIntegerRationalMatrix':
        """
        Create matrix from 2D list of BigFraction values.
        
        Args:
            data: 2D list containing BigFraction or numeric values
            rows_in_dim1: If True, data[i][j] = row i, column j. If False, data[i][j] = column i, row j
        
        Returns:
            DefaultBigIntegerRationalMatrix initialized with the fraction data
        """
        if not data or not data[0]:
            raise ValueError("data must not be empty")
        
        # Convert all values to BigFraction
        fraction_data = []
        for row in data:
            fraction_row = []
            for val in row:
                if isinstance(val, BigFraction):
                    fraction_row.append(val)
                else:
                    fraction_row.append(BigFraction.value_of(val))
            fraction_data.append(fraction_row)
        
        return DefaultBigIntegerRationalMatrix(fraction_data, rows_in_dim1)
    
    @staticmethod
    def from_values(data: List[List[Union[int, float]]], rows_in_dim1: bool = True) -> 'DefaultBigIntegerRationalMatrix':
        """
        Create matrix from 2D list of numeric values.
        
        Args:
            data: 2D list containing int or float values
            rows_in_dim1: If True, data[i][j] = row i, column j. If False, data[i][j] = column i, row j
        
        Returns:
            DefaultBigIntegerRationalMatrix initialized with the numeric data
        """
        return DefaultBigIntegerRationalMatrix(data, rows_in_dim1)