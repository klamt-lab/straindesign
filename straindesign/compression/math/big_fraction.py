"""
Python port of ch.javasoft.math.BigFraction from EFMTool.

This module provides exact rational arithmetic using Python's built-in fractions.Fraction
as the underlying implementation, with a wrapper to match the Java BigFraction API.

Key decision (USER APPROVED): We use Python's fractions.Fraction as the base implementation because:
1. It provides exact rational arithmetic with arbitrary precision
2. Auto-reduction is actually PREFERRED for numerical stability and performance
3. Python's int type has arbitrary precision (equivalent to Java's BigInteger)
4. Functional equivalence is more important than exact implementation details
5. User specifically approved this approach over manual unreduced arithmetic

The BigFraction class is a thin wrapper providing Java-compatible method names
while leveraging Python's superior fraction handling.
"""

from fractions import Fraction
from typing import Union, Optional
import math


class BigFraction:
    """
    Python port of Java's BigFraction class.
    
    Uses Python's fractions.Fraction internally for optimal performance and numerical stability.
    Auto-reduction is preferred over Java's unreduced arithmetic (USER APPROVED).
    """
    
    def __init__(self, numerator: Union[int, 'BigFraction', Fraction], denominator: Optional[int] = None):
        """
        Constructor matching Java BigFraction constructors.
        
        Args:
            numerator: Either an integer numerator, or another BigFraction/Fraction to copy
            denominator: Optional denominator (default 1 if not provided)
        """
        if denominator is None:
            if isinstance(numerator, BigFraction):
                # Copy constructor
                self._fraction = numerator._fraction
            elif isinstance(numerator, Fraction):
                self._fraction = numerator
            else:
                # Single argument - treat as whole number
                self._fraction = Fraction(numerator)
        else:
            # Two arguments - numerator and denominator
            if denominator == 0:
                if numerator == 0:
                    raise ArithmeticError("Division undefined")  # NaN equivalent
                raise ArithmeticError("Division by zero")
            self._fraction = Fraction(numerator, denominator)
    
    @property
    def numerator(self) -> int:
        """Returns the numerator (matches Java getNumerator())"""
        return self._fraction.numerator
    
    @property
    def denominator(self) -> int:
        """Returns the denominator (matches Java getDenominator())"""
        return self._fraction.denominator
    
    def get_numerator(self) -> int:
        """Java-style getter for numerator"""
        return self.numerator
    
    def get_denominator(self) -> int:
        """Java-style getter for denominator"""
        return self.denominator
    
    def double_value(self) -> float:
        """Convert to float (matches Java doubleValue())"""
        return float(self._fraction)
    
    def float_value(self) -> float:
        """Convert to float (matches Java floatValue())"""
        return float(self._fraction)
    
    def int_value(self) -> int:
        """Convert to int (matches Java intValue())"""
        return int(self._fraction)
    
    def long_value(self) -> int:
        """Convert to int (matches Java longValue())"""
        return int(self._fraction)
    
    def abs(self) -> 'BigFraction':
        """Return absolute value"""
        return BigFraction(abs(self._fraction))
    
    def negate(self) -> 'BigFraction':
        """Return negation"""
        return BigFraction(-self._fraction)
    
    def add(self, other: 'BigFraction') -> 'BigFraction':
        """Add two BigFractions"""
        if not isinstance(other, BigFraction):
            other = BigFraction(other)
        return BigFraction(self._fraction + other._fraction)
    
    def subtract(self, other: 'BigFraction') -> 'BigFraction':
        """Subtract two BigFractions"""
        if not isinstance(other, BigFraction):
            other = BigFraction(other)
        return BigFraction(self._fraction - other._fraction)
    
    def multiply(self, other: 'BigFraction') -> 'BigFraction':
        """Multiply two BigFractions"""
        if not isinstance(other, BigFraction):
            other = BigFraction(other)
        return BigFraction(self._fraction * other._fraction)
    
    def divide(self, other: 'BigFraction') -> 'BigFraction':
        """Divide two BigFractions"""
        if not isinstance(other, BigFraction):
            other = BigFraction(other)
        if other._fraction == 0:
            raise ArithmeticError("Division by zero")
        return BigFraction(self._fraction / other._fraction)
    
    def reduce(self) -> 'BigFraction':
        """
        Return reduced form (canonical form with gcd removed).
        Note: Python's Fraction is always in reduced form, so this returns self.
        """
        return self
    
    def signum(self) -> int:
        """Return sign: -1, 0, or 1"""
        if self._fraction < 0:
            return -1
        elif self._fraction > 0:
            return 1
        else:
            return 0
    
    def is_zero(self) -> bool:
        """Check if fraction is zero"""
        return self._fraction == 0
    
    def is_negative(self) -> bool:
        """Check if fraction is negative"""
        return self._fraction < 0
    
    def is_one(self) -> bool:
        """Check if fraction is one"""
        return self._fraction == 1
    
    def is_integer(self) -> bool:
        """Check if fraction represents an integer"""
        return self._fraction.denominator == 1
    
    def to_big_integer(self) -> int:
        """Convert to integer (truncating)"""
        return self._fraction.numerator // self._fraction.denominator
    
    def invert(self) -> 'BigFraction':
        """
        Return multiplicative inverse (1/this).
        Matches Java BigFraction.invert() method.
        """
        if self._fraction == 0:
            raise ArithmeticError("Division by zero")
        return BigFraction(1 / self._fraction)
    
    def pow(self, exponent: int) -> 'BigFraction':
        """
        Return this^exponent.
        Matches Java BigFraction.pow() method.
        """
        if not isinstance(exponent, int):
            raise TypeError("Exponent must be an integer")
        return BigFraction(self._fraction ** exponent)
    
    def compare_to(self, other: 'BigFraction') -> int:
        """Compare to another BigFraction: -1 if less, 0 if equal, 1 if greater"""
        if not isinstance(other, BigFraction):
            other = BigFraction(other)
        if self._fraction < other._fraction:
            return -1
        elif self._fraction > other._fraction:
            return 1
        else:
            return 0
    
    def __eq__(self, other) -> bool:
        """Equality comparison"""
        if isinstance(other, BigFraction):
            return self._fraction == other._fraction
        return False
    
    def __lt__(self, other) -> bool:
        """Less than comparison"""
        if isinstance(other, BigFraction):
            return self._fraction < other._fraction
        return NotImplemented
    
    def __le__(self, other) -> bool:
        """Less than or equal comparison"""
        if isinstance(other, BigFraction):
            return self._fraction <= other._fraction
        return NotImplemented
    
    def __gt__(self, other) -> bool:
        """Greater than comparison"""
        if isinstance(other, BigFraction):
            return self._fraction > other._fraction
        return NotImplemented
    
    def __ge__(self, other) -> bool:
        """Greater than or equal comparison"""
        if isinstance(other, BigFraction):
            return self._fraction >= other._fraction
        return NotImplemented
    
    def __float__(self) -> float:
        """Convert to Python float"""
        return float(self._fraction)
    
    def to_double(self) -> float:
        """Convert to double (Java compatibility method)"""
        return float(self._fraction)
    
    def get_double(self) -> float:
        """Get double value (Java compatibility method)"""
        return float(self._fraction)
    
    def __hash__(self) -> int:
        """Hash code"""
        return hash(self._fraction)
    
    def __str__(self) -> str:
        """String representation"""
        if self._fraction.denominator == 1:
            return str(self._fraction.numerator)
        return f"{self._fraction.numerator}/{self._fraction.denominator}"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"BigFraction({self._fraction.numerator}, {self._fraction.denominator})"
    
    # Python operator overloading for convenience
    def __add__(self, other):
        return self.add(other)
    
    def __sub__(self, other):
        return self.subtract(other)
    
    def __mul__(self, other):
        return self.multiply(other)
    
    def __truediv__(self, other):
        return self.divide(other)
    
    def __neg__(self):
        return self.negate()
    
    def __abs__(self):
        return self.abs()
    
    # Static methods and constants
    @staticmethod
    def value_of(value: Union[int, float, str, 'BigFraction']) -> 'BigFraction':
        """
        Factory method to create BigFraction from various types.
        Matches Java's valueOf methods.
        """
        if isinstance(value, BigFraction):
            return value
        elif isinstance(value, str):
            # Parse string - can be "num/den" or just "num"
            if '/' in value:
                parts = value.split('/')
                if len(parts) != 2:
                    raise ValueError(f"Invalid fraction format: {value}")
                return BigFraction(int(parts[0]), int(parts[1]))
            else:
                return BigFraction(Fraction(value))
        elif isinstance(value, (int, float)):
            return BigFraction(Fraction(value).limit_denominator())
        else:
            return BigFraction(value)
    
    @staticmethod
    def gcd(*values: 'BigFraction') -> 'BigFraction':
        """
        Calculate GCD of multiple BigFractions.
        For fractions, gcd(a/b, c/d) = gcd(a,c) / lcm(b,d)
        """
        if not values:
            return BigFraction.ZERO
        if len(values) == 1:
            return values[0].abs()
        
        # For multiple fractions, compute GCD
        numerators = [v.numerator for v in values]
        denominators = [v.denominator for v in values]
        
        # GCD of numerators
        gcd_num = numerators[0]
        for n in numerators[1:]:
            gcd_num = math.gcd(gcd_num, n)
        
        # LCM of denominators
        lcm_den = denominators[0]
        for d in denominators[1:]:
            lcm_den = (lcm_den * d) // math.gcd(lcm_den, d)
        
        return BigFraction(gcd_num, lcm_den)


# Constants matching Java's BigFraction
BigFraction.ZERO = BigFraction(0, 1)
BigFraction.ONE = BigFraction(1, 1)
BigFraction.TWO = BigFraction(2, 1)
BigFraction.TEN = BigFraction(10, 1)