"""
FlintBigIntegerRationalMatrix - FLINT-backed rational matrix implementation.

This module provides a fast implementation of BigIntegerRationalMatrix using
python-flint's fmpq_mat for internal storage. This eliminates the overhead of
Python fraction objects for element access and matrix operations.
"""

import io
from typing import List, Optional, Union

# Try to import FLINT - this module should only be used when FLINT is available
try:
    from flint import fmpq_mat, fmpq
    FLINT_AVAILABLE = True
except ImportError:
    FLINT_AVAILABLE = False
    fmpq_mat = None
    fmpq = None

from .bigint_rational_matrix import BigIntegerRationalMatrix
from .big_fraction import BigFraction


class FlintBigIntegerRationalMatrix(BigIntegerRationalMatrix):
    """
    FLINT-backed implementation of BigIntegerRationalMatrix.

    Uses python-flint's fmpq_mat for internal storage, providing much faster
    element access and matrix operations compared to the default Python implementation.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize matrix with various input types.

        Supported signatures:
        - FlintBigIntegerRationalMatrix(rows, cols) - zero matrix
        - FlintBigIntegerRationalMatrix(readable_matrix) - copy constructor
        - FlintBigIntegerRationalMatrix(fmpq_mat) - wrap existing FLINT matrix
        """
        if not FLINT_AVAILABLE:
            raise RuntimeError("python-flint is not available")

        if len(args) == 2 and isinstance(args[0], int) and isinstance(args[1], int):
            # Constructor: (rows, cols) - create zero matrix
            self._mat = fmpq_mat(args[0], args[1])
        elif len(args) == 1 and isinstance(args[0], fmpq_mat):
            # Constructor: wrap existing FLINT matrix (no copy for speed)
            self._mat = args[0]
        elif len(args) == 1 and hasattr(args[0], 'get_row_count'):
            # Constructor: (readable_matrix) - copy from another matrix
            self._init_from_readable_matrix(args[0])
        else:
            raise ValueError(f"Invalid constructor arguments: {args}")

    def _init_from_readable_matrix(self, mx):
        """Initialize from readable matrix (copy constructor)"""
        rows = mx.get_row_count()
        cols = mx.get_column_count()
        self._mat = fmpq_mat(rows, cols)

        for row in range(rows):
            for col in range(cols):
                if hasattr(mx, 'get_big_integer_numerator_at'):
                    num = mx.get_big_integer_numerator_at(row, col)
                    den = mx.get_big_integer_denominator_at(row, col)
                    if num != 0:
                        self._mat[row, col] = fmpq(num, den)
                else:
                    value = mx.get_number_value_at(row, col)
                    if value.signum() != 0:
                        self._mat[row, col] = fmpq(value.numerator, value.denominator)

    @classmethod
    def from_flint_matrix(cls, mat: 'fmpq_mat') -> 'FlintBigIntegerRationalMatrix':
        """Create wrapper around existing FLINT matrix (no copy)"""
        return cls(mat)

    # Core interface methods
    def get_row_count(self) -> int:
        """Get number of rows"""
        return self._mat.nrows()

    def get_column_count(self) -> int:
        """Get number of columns"""
        return self._mat.ncols()

    def get_big_fraction_value_at(self, row: int, col: int) -> BigFraction:
        """Get BigFraction value at specified position"""
        val = self._mat[row, col]
        return BigFraction(int(val.p), int(val.q))

    def get_big_integer_numerator_at(self, row: int, col: int) -> int:
        """Get numerator at specified position"""
        return int(self._mat[row, col].p)

    def get_big_integer_denominator_at(self, row: int, col: int) -> int:
        """Get denominator at specified position"""
        return int(self._mat[row, col].q)

    def get_number_value_at(self, row: int, col: int) -> BigFraction:
        """Get number value (BigFraction) at specified position"""
        return self.get_big_fraction_value_at(row, col)

    def get_signum_at(self, row: int, col: int) -> int:
        """Get sign (-1, 0, 1) of value at specified position"""
        val = self._mat[row, col]
        num = int(val.p)
        den = int(val.q)
        if num == 0:
            return 0
        # num and den signs
        num_sign = 1 if num > 0 else -1
        den_sign = 1 if den > 0 else -1
        return num_sign * den_sign

    # Direct access to FLINT value (for optimized operations)
    def get_flint_value_at(self, row: int, col: int) -> 'fmpq':
        """Get raw FLINT fmpq value at specified position (no conversion)"""
        return self._mat[row, col]

    def get_flint_matrix(self) -> 'fmpq_mat':
        """Get underlying FLINT matrix (for direct operations)"""
        return self._mat

    # Write operations
    def set_value_at(self, row: int, col: int, value: BigFraction) -> None:
        """Set BigFraction value at specified position"""
        self._mat[row, col] = fmpq(value.numerator, value.denominator)

    def set_value_at_rational(self, row: int, col: int, numerator: int, denominator: int) -> None:
        """Set rational value using numerator/denominator"""
        self._mat[row, col] = fmpq(numerator, denominator)

    def set_flint_value_at(self, row: int, col: int, value: 'fmpq') -> None:
        """Set raw FLINT fmpq value at specified position (no conversion)"""
        self._mat[row, col] = value

    def add(self, row: int, col: int, numerator: int, denominator: int) -> None:
        """Add rational value to existing value at position"""
        current = self._mat[row, col]
        add_val = fmpq(numerator, denominator)
        self._mat[row, col] = current + add_val

    def multiply(self, row: int, col: int, numerator: int, denominator: int) -> None:
        """Multiply existing value at position by rational value"""
        current = self._mat[row, col]
        mult_val = fmpq(numerator, denominator)
        self._mat[row, col] = current * mult_val

    def multiply_row(self, row: int, numerator: int, denominator: int) -> None:
        """Multiply entire row by rational value"""
        mult_val = fmpq(numerator, denominator)
        cols = self.get_column_count()
        for col in range(cols):
            self._mat[row, col] = self._mat[row, col] * mult_val

    def add_row_to_other_row(self, src_row: int, src_numerator: int, src_denominator: int,
                            dst_row: int, dst_numerator: int, dst_denominator: int) -> None:
        """Add source row (multiplied by src ratio) to destination row (multiplied by dst ratio)"""
        src_mult = fmpq(src_numerator, src_denominator)
        dst_mult = fmpq(dst_numerator, dst_denominator)
        cols = self.get_column_count()

        for col in range(cols):
            src_val = self._mat[src_row, col]
            dst_val = self._mat[dst_row, col]
            self._mat[dst_row, col] = dst_val * dst_mult + src_val * src_mult

    # Reduction operations
    def reduce(self) -> bool:
        """Reduce whole matrix - FLINT auto-reduces, so this is a no-op"""
        return False  # FLINT fmpq values are always reduced

    def reduce_row(self, row: int) -> bool:
        """Reduce specified row - FLINT auto-reduces"""
        return False

    def reduce_value_at(self, row: int, col: int) -> bool:
        """Reduce value at position - FLINT auto-reduces"""
        return False

    # Matrix operations
    def clone(self) -> 'FlintBigIntegerRationalMatrix':
        """Create deep copy of this matrix"""
        # Create new matrix and copy values
        rows, cols = self.get_row_count(), self.get_column_count()
        new_mat = fmpq_mat(rows, cols)
        for row in range(rows):
            for col in range(cols):
                new_mat[row, col] = self._mat[row, col]
        return FlintBigIntegerRationalMatrix(new_mat)

    def new_instance(self, rows: int, cols: int) -> 'FlintBigIntegerRationalMatrix':
        """Create new matrix instance with given dimensions"""
        return FlintBigIntegerRationalMatrix(rows, cols)

    def new_instance_from_data(self, data: List[List[BigFraction]],
                              rows_in_dim1: bool = True) -> 'FlintBigIntegerRationalMatrix':
        """Create new matrix from 2D BigFraction data"""
        if rows_in_dim1:
            rows = len(data)
            cols = len(data[0]) if rows > 0 else 0
        else:
            cols = len(data)
            rows = len(data[0]) if cols > 0 else 0

        result = FlintBigIntegerRationalMatrix(rows, cols)

        for row in range(rows):
            for col in range(cols):
                if rows_in_dim1:
                    value = data[row][col]
                else:
                    value = data[col][row]
                result.set_value_at(row, col, value)

        return result

    def transpose(self) -> 'FlintBigIntegerRationalMatrix':
        """Return transposed matrix"""
        return FlintBigIntegerRationalMatrix(self._mat.transpose())

    # Row/column operations
    def swap_rows(self, row_a: int, row_b: int) -> None:
        """Swap two rows"""
        if row_a == row_b:
            return
        cols = self.get_column_count()
        for col in range(cols):
            tmp = self._mat[row_a, col]
            self._mat[row_a, col] = self._mat[row_b, col]
            self._mat[row_b, col] = tmp

    def swap_columns(self, col_a: int, col_b: int) -> None:
        """Swap two columns"""
        if col_a == col_b:
            return
        rows = self.get_row_count()
        for row in range(rows):
            tmp = self._mat[row, col_a]
            self._mat[row, col_a] = self._mat[row, col_b]
            self._mat[row, col_b] = tmp

    def negate(self, row: int, col: int) -> None:
        """Negate value at specified position"""
        self._mat[row, col] = -self._mat[row, col]

    # String representation
    def __str__(self) -> str:
        """String representation"""
        return str(self._mat)

    def to_multiline_string(self) -> str:
        """Multi-line string representation"""
        return str(self._mat)

    # Output methods
    def write_to(self, writer: io.TextIOBase) -> None:
        """Write single line representation to text writer"""
        writer.write(str(self))

    def write_to_multiline(self, writer: io.TextIOBase) -> None:
        """Write multi-line representation to text writer"""
        writer.write(self.to_multiline_string())

    # Double conversion methods
    def get_double_value_at(self, row: int, col: int) -> float:
        """Get double precision value at specified position"""
        val = self._mat[row, col]
        return float(val.p) / float(val.q)

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

    # Submatrix extraction
    def sub_big_integer_rational_matrix(self, row_start: int, row_end: int,
                                       col_start: int, col_end: int) -> 'FlintBigIntegerRationalMatrix':
        """Extract submatrix as rational matrix"""
        new_rows = row_end - row_start
        new_cols = col_end - col_start
        result = FlintBigIntegerRationalMatrix(new_rows, new_cols)

        for row in range(new_rows):
            for col in range(new_cols):
                result._mat[row, col] = self._mat[row_start + row, col_start + col]

        return result

    # Conversion methods
    def to_big_integer_rational_matrix(self, enforce_new_instance: bool = False) -> 'FlintBigIntegerRationalMatrix':
        """Convert to rational matrix (self or copy)"""
        return self.clone() if enforce_new_instance else self

    def to_writable_matrix(self, enforce_new_instance: bool = False) -> 'FlintBigIntegerRationalMatrix':
        """Convert to writable matrix"""
        return self.to_big_integer_rational_matrix(enforce_new_instance)

    def to_default_matrix(self):
        """Convert to DefaultBigIntegerRationalMatrix"""
        from .default_bigint_rational_matrix import DefaultBigIntegerRationalMatrix
        return DefaultBigIntegerRationalMatrix(self)

    # Convenience methods
    def set_value_at_int(self, row: int, col: int, value: int) -> None:
        """Set integer value (denominator = 1)"""
        self._mat[row, col] = fmpq(value)

    def set_value_at_double(self, row: int, col: int, value: float) -> None:
        """Set double value"""
        from fractions import Fraction
        frac = Fraction(value).limit_denominator()
        self._mat[row, col] = fmpq(frac.numerator, frac.denominator)

    # Missing abstract method implementations
    def get_number_rows(self) -> List[List[BigFraction]]:
        """Get all rows as 2D list of BigFraction"""
        result = []
        for row in range(self.get_row_count()):
            row_data = []
            for col in range(self.get_column_count()):
                row_data.append(self.get_big_fraction_value_at(row, col))
            result.append(row_data)
        return result

    def to_double_matrix(self, enforce_new_instance: bool = False):
        """Convert to double matrix"""
        from .double_matrix import DefaultDoubleMatrix
        rows, cols = self.get_row_count(), self.get_column_count()
        result = DefaultDoubleMatrix(rows, cols)
        for row in range(rows):
            for col in range(cols):
                result.set_value_at(row, col, self.get_double_value_at(row, col))
        return result

    def sub_double_matrix(self, row_start: int, row_end: int, col_start: int, col_end: int):
        """Extract submatrix as double matrix"""
        from .double_matrix import DefaultDoubleMatrix
        new_rows = row_end - row_start
        new_cols = col_end - col_start
        result = DefaultDoubleMatrix(new_rows, new_cols)
        for row in range(new_rows):
            for col in range(new_cols):
                result.set_value_at(row, col, self.get_double_value_at(row_start + row, col_start + col))
        return result

    def to_double_array(self) -> List[float]:
        """Convert matrix to 1D double array (row-major order)"""
        result = []
        for row in range(self.get_row_count()):
            for col in range(self.get_column_count()):
                result.append(self.get_double_value_at(row, col))
        return result

    def to_array(self, array: List[float]) -> None:
        """Fill provided array with matrix values (row-major order)"""
        idx = 0
        for row in range(self.get_row_count()):
            for col in range(self.get_column_count()):
                array[idx] = self.get_double_value_at(row, col)
                idx += 1

    def to_writable_matrix(self, enforce_new_instance: bool = False) -> 'FlintBigIntegerRationalMatrix':
        """Convert to writable matrix"""
        return self.clone() if enforce_new_instance else self

    def get_number_operations(self):
        """Get the NumberOperations instance for this matrix's number type"""
        from .bigint_rational_matrix_operations import BigIntegerRationalNumberOperations
        return BigIntegerRationalNumberOperations()

    def get_matrix_operations(self):
        """Get the MatrixOperations instance for this matrix type"""
        from .bigint_rational_matrix_operations import BigIntegerRationalMatrixOperations
        return BigIntegerRationalMatrixOperations()

    def write_to_stream(self, stream: io.BytesIO) -> None:
        """Write to binary output stream"""
        stream.write(str(self).encode('utf-8'))

    def write_to_multiline_stream(self, stream: io.BytesIO) -> None:
        """Write multi-line representation to binary output stream"""
        stream.write(self.to_multiline_string().encode('utf-8'))
