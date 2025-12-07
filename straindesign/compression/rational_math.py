#!/usr/bin/env python3
"""
Rational number mathematics for exact arithmetic in network compression.

This module provides utilities for working with rational numbers using Python's
fractions.Fraction class, replacing the Java BigFraction functionality from efmtool.
"""

from fractions import Fraction
from typing import Union, Tuple
import numpy as np
from sympy import Rational
import math

# Type alias for numeric types that can be converted to Fraction
Numeric = Union[int, float, Fraction, Rational]


class RationalMath:
    """Utility class for rational number operations."""
    
    # Commonly used constants
    ZERO = Fraction(0)
    ONE = Fraction(1)
    MINUS_ONE = Fraction(-1)
    
    @staticmethod
    def to_fraction(value: Numeric) -> Fraction:
        """
        Convert a numeric value to a Fraction.
        
        Args:
            value: An int, float, Fraction, or sympy.Rational
            
        Returns:
            Fraction representation of the value
        """
        if isinstance(value, Fraction):
            return value
        elif isinstance(value, Rational):
            return Fraction(int(value.p), int(value.q))
        elif isinstance(value, (int, float)):
            return Fraction(value).limit_denominator()
        else:
            raise TypeError(f"Cannot convert {type(value)} to Fraction")
    
    @staticmethod
    def to_sympy_rational(value: Union[Fraction, Numeric]) -> Rational:
        """
        Convert a Fraction to a sympy Rational.
        
        Args:
            value: A Fraction or numeric value
            
        Returns:
            sympy.Rational representation
        """
        if isinstance(value, Rational):
            return value
        elif isinstance(value, Fraction):
            return Rational(value.numerator, value.denominator)
        else:
            frac = RationalMath.to_fraction(value)
            return Rational(frac.numerator, frac.denominator)
    
    @staticmethod
    def is_zero(value: Union[Fraction, int]) -> bool:
        """
        Check if a value is zero.
        
        Args:
            value: Value to check
            
        Returns:
            True if value is zero
        """
        if isinstance(value, Fraction):
            return value.numerator == 0
        return value == 0
    
    @staticmethod
    def signum(value: Fraction) -> int:
        """
        Get the sign of a fraction.
        
        Args:
            value: Fraction to check
            
        Returns:
            -1 if negative, 0 if zero, 1 if positive
        """
        if value.numerator == 0:
            return 0
        elif value.numerator < 0:
            return -1 if value.denominator > 0 else 1
        else:
            return 1 if value.denominator > 0 else -1
    
    @staticmethod
    def abs(value: Fraction) -> Fraction:
        """
        Get absolute value of a fraction.
        
        Args:
            value: Fraction to process
            
        Returns:
            Absolute value as Fraction
        """
        return Fraction(abs(value.numerator), abs(value.denominator))
    
    @staticmethod
    def reduce(value: Fraction) -> Fraction:
        """
        Reduce a fraction to lowest terms.
        
        Args:
            value: Fraction to reduce
            
        Returns:
            Reduced fraction
        """
        # Fraction automatically reduces, but we ensure consistency
        return Fraction(value.numerator, value.denominator)
    
    @staticmethod
    def gcd(a: int, b: int) -> int:
        """
        Calculate greatest common divisor.
        
        Args:
            a: First integer
            b: Second integer
            
        Returns:
            GCD of a and b
        """
        return math.gcd(a, b)
    
    @staticmethod
    def array_to_fractions(arr: np.ndarray) -> np.ndarray:
        """
        Convert a numpy array to an array of Fractions.
        
        Args:
            arr: Numpy array of numeric values
            
        Returns:
            Object array containing Fraction objects
        """
        result = np.empty(arr.shape, dtype=object)
        flat_result = result.flat
        flat_arr = arr.flat
        
        for i, val in enumerate(flat_arr):
            flat_result[i] = RationalMath.to_fraction(val)
        
        return result
    
    @staticmethod
    def fractions_to_floats(arr: np.ndarray) -> np.ndarray:
        """
        Convert an array of Fractions to floats.
        
        Args:
            arr: Object array containing Fractions
            
        Returns:
            Float array
        """
        result = np.empty(arr.shape, dtype=float)
        flat_result = result.flat
        flat_arr = arr.flat
        
        for i, val in enumerate(flat_arr):
            if isinstance(val, Fraction):
                flat_result[i] = float(val)
            else:
                flat_result[i] = float(val)
        
        return result
    
    @staticmethod
    def fractions_to_sympy(arr: np.ndarray) -> np.ndarray:
        """
        Convert an array of Fractions to sympy Rationals.
        
        Args:
            arr: Object array containing Fractions
            
        Returns:
            Object array containing sympy Rationals
        """
        result = np.empty(arr.shape, dtype=object)
        flat_result = result.flat
        flat_arr = arr.flat
        
        for i, val in enumerate(flat_arr):
            if isinstance(val, Fraction):
                flat_result[i] = Rational(val.numerator, val.denominator)
            elif isinstance(val, Rational):
                flat_result[i] = val
            else:
                frac = RationalMath.to_fraction(val)
                flat_result[i] = Rational(frac.numerator, frac.denominator)
        
        return result


# Conversion functions matching the Java interface from test_compress.py
def sympyRat2Fraction(val: Rational) -> Tuple[int, int]:
    """
    Convert sympy Rational to numerator/denominator pair.
    
    This matches the sympyRat2jBigIntegerPair function interface.
    
    Args:
        val: sympy Rational
        
    Returns:
        Tuple of (numerator, denominator)
    """
    return (int(val.p), int(val.q))


def Fraction2sympyRat(numerator: int, denominator: int) -> Rational:
    """
    Convert numerator/denominator pair to sympy Rational.
    
    This matches the jBigIntegerPair2sympyRat function interface.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        
    Returns:
        sympy Rational
    """
    return Rational(numerator, denominator)


def value2Fraction(val: Union[Fraction, float, int, Rational]) -> Fraction:
    """
    Convert various numeric types to Fraction.
    
    This provides a unified interface for fraction conversion.
    
    Args:
        val: Numeric value to convert
        
    Returns:
        Fraction representation
    """
    if isinstance(val, Fraction):
        return val
    elif isinstance(val, Rational):
        return Fraction(int(val.p), int(val.q))
    elif isinstance(val, (int, float)):
        return Fraction(val).limit_denominator()
    else:
        raise TypeError(f"Cannot convert {type(val)} to Fraction")


# Module-level constants for compatibility
ZERO = RationalMath.ZERO
ONE = RationalMath.ONE
MINUS_ONE = RationalMath.MINUS_ONE