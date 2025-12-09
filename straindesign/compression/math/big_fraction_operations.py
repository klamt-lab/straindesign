"""
Python port of ch.javasoft.math.ops.BigFractionOperations from EFMTool.

This module provides operations on BigFraction instances, implementing the
NumberOperations interface pattern used in Java EFMTool.

The class serves as a factory and operations provider for BigFraction instances,
following the singleton pattern from the Java implementation.
"""

import io
from typing import List, Union, Any
import math
from .big_fraction import BigFraction


class BigFractionOperations:
    """
    Python port of Java's BigFractionOperations class.
    
    Provides factory methods and operations for BigFraction instances.
    Implements singleton pattern like the Java version.
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern implementation"""
        if cls._instance is None:
            cls._instance = super(BigFractionOperations, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def instance(cls) -> 'BigFractionOperations':
        """
        Returns the singleton instance.
        Matches Java's instance() method.
        """
        return cls()
    
    def number_class(self) -> type:
        """Return the BigFraction class"""
        return BigFraction
    
    def new_array(self, size: int) -> List[BigFraction]:
        """Create new BigFraction array of given size"""
        return [None] * size
    
    def new_matrix(self, rows: int, cols: int) -> List[List[BigFraction]]:
        """Create new BigFraction matrix of given dimensions"""
        return [[None for _ in range(cols)] for _ in range(rows)]
    
    # Factory methods - delegate to BigFraction.value_of
    def value_of_string(self, s: str) -> BigFraction:
        """Create BigFraction from string"""
        return BigFraction.value_of(s)
    
    def value_of_number(self, number: Union[int, float]) -> BigFraction:
        """Create BigFraction from number"""
        return BigFraction.value_of(number)
    
    def value_of_long(self, value: int) -> BigFraction:
        """Create BigFraction from long/int"""
        return BigFraction.value_of(value)
    
    def value_of_double(self, value: float) -> BigFraction:
        """Create BigFraction from double/float"""
        return BigFraction.value_of(value)
    
    # Arithmetic operations - delegate to BigFraction methods
    def abs(self, number: BigFraction) -> BigFraction:
        """Return absolute value"""
        return number.abs()
    
    def add(self, num_a: BigFraction, num_b: BigFraction) -> BigFraction:
        """Add two BigFractions"""
        return num_a.add(num_b)
    
    def subtract(self, num_a: BigFraction, num_b: BigFraction) -> BigFraction:
        """Subtract two BigFractions"""
        return num_a.subtract(num_b)
    
    def multiply(self, num_a: BigFraction, num_b: BigFraction) -> BigFraction:
        """Multiply two BigFractions"""
        return num_a.multiply(num_b)
    
    def divide(self, num_a: BigFraction, num_b: BigFraction) -> BigFraction:
        """Divide two BigFractions"""
        return num_a.divide(num_b)
    
    def negate(self, number: BigFraction) -> BigFraction:
        """Return negation"""
        return number.negate()
    
    def invert(self, number: BigFraction) -> BigFraction:
        """Return multiplicative inverse"""
        return number.invert()
    
    def reduce(self, number: BigFraction) -> BigFraction:
        """Return reduced form"""
        return number.reduce()
    
    # Comparison operations
    def compare(self, o1: BigFraction, o2: BigFraction) -> int:
        """Compare two BigFractions"""
        return o1.compare_to(o2)
    
    def signum(self, number: BigFraction) -> int:
        """Return sign of number"""
        return number.signum()
    
    # Predicates
    def is_zero(self, number: BigFraction) -> bool:
        """Check if number is zero"""
        return number.is_zero()
    
    def is_one(self, number: BigFraction) -> bool:
        """Check if number is one"""
        return number.is_one()
    
    # Constants
    def zero(self) -> BigFraction:
        """Return zero"""
        return BigFraction.ZERO
    
    def one(self) -> BigFraction:
        """Return one"""
        return BigFraction.ONE
    
    # Power operation
    def pow(self, num_a: BigFraction, num_b: BigFraction) -> BigFraction:
        """
        Raise num_a to the power of num_b.
        Matches Java implementation - only integer exponents supported.
        """
        exp_sign = num_b.signum()
        if exp_sign == 0:
            return BigFraction.ONE
        
        if num_a.is_one():
            return BigFraction.ONE
        
        num_b = num_b.reduce()
        if not num_b.is_integer():
            raise ArithmeticError("non-integer exponent not supported")
        
        exponent = num_b.numerator
        if abs(exponent) > 2**31 - 1:  # Java Integer.MAX_VALUE
            raise ArithmeticError(f"exponent too large, only integer range supported: {exponent}")
        
        return num_a.pow(int(exponent))
    
    def reduce_vector(self, clone_on_change: bool, *vector: BigFraction) -> List[BigFraction]:
        """
        Reduce a vector of BigFractions by their GCD.
        Matches Java implementation behavior.
        
        Args:
            clone_on_change: If True, clone vector before modifying
            *vector: Variable arguments of BigFraction instances
        
        Returns:
            Reduced vector (possibly the same instance if no reduction needed)
        """
        if not vector:
            return list(vector)
        
        # Calculate GCD of all elements
        gcd = BigFraction.gcd(*vector).abs()
        
        if not (gcd.is_one() or gcd.is_zero()):
            # Need to reduce
            if clone_on_change:
                vector = list(vector)  # Clone the vector
            else:
                vector = list(vector)  # Convert to list for modification
            
            # Divide each element by GCD and reduce
            for i in range(len(vector)):
                if vector[i] is not None:
                    vector[i] = vector[i].divide(gcd).reduce()
        
        return list(vector)
    
    # Serialization methods (simplified versions)
    def to_byte_array(self, number: BigFraction) -> bytes:
        """
        Convert BigFraction to byte array.
        Simplified version - uses string representation for portability.
        """
        string_repr = str(number)
        return string_repr.encode('utf-8')
    
    def from_byte_array(self, bytes_data: bytes) -> BigFraction:
        """
        Create BigFraction from byte array.
        Simplified version - reads string representation.
        """
        string_repr = bytes_data.decode('utf-8')
        return BigFraction.value_of(string_repr)
    
    def write_to(self, number: BigFraction, output: io.BytesIO) -> None:
        """Write BigFraction to output stream"""
        byte_data = self.to_byte_array(number)
        output.write(len(byte_data).to_bytes(4, 'big'))  # Write length first
        output.write(byte_data)
    
    def read_from(self, input_stream: io.BytesIO) -> BigFraction:
        """Read BigFraction from input stream"""
        length_bytes = input_stream.read(4)
        if len(length_bytes) < 4:
            raise EOFError("Unexpected end of stream")
        length = int.from_bytes(length_bytes, 'big')
        data = input_stream.read(length)
        if len(data) < length:
            raise EOFError("Unexpected end of stream")
        return self.from_byte_array(data)
    
    def byte_length(self) -> int:
        """
        Return fixed byte length for this number type.
        Returns -1 as BigFraction has variable length.
        """
        return -1


# Create singleton instance (following Java pattern)
INSTANCE = BigFractionOperations.instance()