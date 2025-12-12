#!/usr/bin/env python3
#
# Copyright 2022 Max Planck Insitute Magdeburg
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
#
"""Functions for the compression of metabolic networks.

This module provides compression utilities. By default, pure Python implementations
are used. Legacy Java compression is available via `legacy_java_compression=True`.

For the documentation of the efmtool compression provided by StrainDesign,
refer to the networktools module.
"""

import numpy
import os
import sys

# Try to import python-flint for fast exact rational operations
try:
    from flint import fmpq_mat as _FlintRationalMatrix, fmpq as _FlintRational
    _FLINT_AVAILABLE = True
except ImportError:
    _FLINT_AVAILABLE = False


# ============================================================================
# Lazy Java Initialization
# ============================================================================

_JAVA_INITIALIZED = False
_JPYPE_AVAILABLE = None

# Java classes (populated by _init_java)
DefaultBigIntegerRationalMatrix = None
Gauss = None
CompressionMethod = None
StoichMatrixCompressor = None
BigFraction = None
BigInteger = None
subset_compression = None
jTrue = None
jSystem = None


def _check_jpype_available():
    """Check if jpype is available without importing it."""
    global _JPYPE_AVAILABLE
    if _JPYPE_AVAILABLE is None:
        import importlib.util
        _JPYPE_AVAILABLE = importlib.util.find_spec("jpype") is not None
    return _JPYPE_AVAILABLE


def _check_sympy_available():
    """Check if sympy is available without importing it."""
    import importlib.util
    return importlib.util.find_spec("sympy") is not None


def _search_for_jvm():
    """Search for JVM in common locations."""
    common_java_paths = [
        "C:\\Program Files\\Java",  # Windows
        "/usr/lib/jvm",  # Linux
        "/Library/Java/JavaVirtualMachines",  # macOS
        os.path.dirname(sys.executable)
    ]
    for base in common_java_paths:
        if os.path.exists(base):
            for root, _dirs, files in os.walk(base):
                if any(lib in files for lib in ["jvm.dll", "libjvm.so", "libjvm.dylib"]):
                    return root
    return None


def _init_java():
    """
    Initialize JVM and Java classes.

    This function is called lazily only when legacy_java_compression=True.
    Raises ImportError if jpype is not installed.
    """
    global _JAVA_INITIALIZED
    global DefaultBigIntegerRationalMatrix, Gauss, CompressionMethod
    global StoichMatrixCompressor, BigFraction, BigInteger
    global subset_compression, jTrue, jSystem

    if _JAVA_INITIALIZED:
        return

    if not _check_jpype_available():
        raise ImportError(
            "jpype1 is not installed. Legacy Java compression requires jpype1.\n"
            "Install with: pip install jpype1\n"
            "Or use the default Python compression (legacy_java_compression=False)."
        )

    if not _check_sympy_available():
        raise ImportError(
            "sympy is not installed. Legacy Java compression requires sympy.\n"
            "Install with: pip install sympy\n"
            "Or use the default Python compression (legacy_java_compression=False)."
        )

    import jpype
    import io
    from contextlib import redirect_stdout, redirect_stderr

    # Add efmtool.jar to classpath
    efmtool_jar = os.path.join(os.path.dirname(__file__), 'efmtool.jar')
    if not os.path.exists(efmtool_jar):
        raise FileNotFoundError(
            f"efmtool.jar not found at {efmtool_jar}. "
            "Legacy Java compression requires the efmtool.jar file."
        )

    jpype.addClassPath(efmtool_jar)

    if not jpype.isJVMStarted():
        # Look up JVM at different locations
        if not os.environ.get("JAVA_HOME"):
            candidate = _search_for_jvm()
            if candidate:
                os.environ["JAVA_HOME"] = candidate
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                jpype.startJVM()
        except Exception as e:
            extra_info = ""
            if not os.environ.get("JAVA_HOME"):
                extra_info = " JAVA_HOME is not defined."
            raise RuntimeError(
                "Failed to start JVM. Please ensure that Java (OpenJDK) is installed." + extra_info +
                " If using conda, install openjdk from conda-forge and set JAVA_HOME to the OpenJDK installation path."
            ) from e

    import jpype.imports

    # Import Java classes
    import ch.javasoft.smx.impl.DefaultBigIntegerRationalMatrix as _DefaultBigIntegerRationalMatrix
    import ch.javasoft.smx.ops.Gauss as _Gauss
    import ch.javasoft.metabolic.compress.CompressionMethod as _CompressionMethod
    import ch.javasoft.metabolic.compress.StoichMatrixCompressor as _StoichMatrixCompressor
    import ch.javasoft.math.BigFraction as _BigFraction
    import java.math.BigInteger as _BigInteger

    # Assign to module-level globals
    DefaultBigIntegerRationalMatrix = _DefaultBigIntegerRationalMatrix
    Gauss = _Gauss
    CompressionMethod = _CompressionMethod
    StoichMatrixCompressor = _StoichMatrixCompressor
    BigFraction = _BigFraction
    BigInteger = _BigInteger

    subset_compression = CompressionMethod[:](
        [CompressionMethod.CoupledZero, CompressionMethod.CoupledCombine, CompressionMethod.CoupledContradicting]
    )
    jTrue = jpype.JBoolean(True)
    jSystem = jpype.JClass("java.lang.System")

    _JAVA_INITIALIZED = True


# ============================================================================
# Pure Python Implementation (Default)
# ============================================================================

def basic_columns_rat(mx, tolerance=0):
    """
    Find basic columns using rational Gaussian elimination.

    Uses FLINT for fast exact rational arithmetic when available,
    falls back to pure Python (sympy) implementation otherwise.

    Args:
        mx: Matrix (numpy array or compatible type)
        tolerance: Tolerance for zero detection (unused in exact arithmetic)

    Returns:
        Array of indices of basic columns
    """
    if not isinstance(mx, numpy.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(mx)}")

    # Use unified flint_interface which handles FLINT/sympy fallback
    from .flint_interface import basic_columns_from_numpy
    return basic_columns_from_numpy(mx)


# ============================================================================
# Legacy Java Implementation (requires jpype + sympy)
# ============================================================================

def basic_columns_rat_java(mx, tolerance=0):
    """
    Find basic columns using Java Gaussian elimination.

    Legacy implementation using jpype and Java efmtool.
    Requires jpype1 and sympy to be installed.

    Args:
        mx: Matrix (numpy array or Java matrix)
        tolerance: Tolerance (unused in exact arithmetic)

    Returns:
        Array of indices of basic columns
    """
    _init_java()
    import jpype

    if isinstance(mx, numpy.ndarray):
        mx = DefaultBigIntegerRationalMatrix(numpy_mat2jpypeArrayOfArrays(mx), jTrue, jTrue)

    row_map = jpype.JInt[mx.getRowCount()]
    col_map = jpype.JInt[:](range(mx.getColumnCount()))
    rank = Gauss.getRationalInstance().rowEchelon(mx, False, row_map, col_map)

    return col_map[0:rank]


def numpy_mat2jpypeArrayOfArrays(npmat):
    """Convert numpy matrix to jpype array of arrays (requires Java init)."""
    _init_java()
    import jpype

    rows = npmat.shape[0]
    cols = npmat.shape[1]
    jmat = jpype.JDouble[rows, cols]
    for r in range(rows):
        for c in range(cols):
            jmat[r][c] = npmat[r, c]
    return jmat


def jpypeArrayOfArrays2numpy_mat(jmat):
    """Convert jpype array of arrays to numpy matrix."""
    rows = len(jmat)
    cols = len(jmat[0])
    npmat = numpy.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            npmat[r, c] = jmat[r][c]
    return npmat


def sympyRat2jBigIntegerPair(val):
    """Convert Fraction or sympy Rational to Java BigInteger pair (requires Java init)."""
    _init_java()

    # Support both fractions.Fraction (.numerator/.denominator) and sympy.Rational (.p/.q)
    numer = val.numerator if hasattr(val, 'numerator') else val.p
    if numer.bit_length() <= 63:
        numer = BigInteger.valueOf(numer)
    else:
        numer = BigInteger(str(numer))

    denom = val.denominator if hasattr(val, 'denominator') else val.q
    if denom.bit_length() <= 63:
        denom = BigInteger.valueOf(denom)
    else:
        denom = BigInteger(str(denom))

    return (numer, denom)


def jBigFraction2sympyRat(val):
    """Convert Java BigFraction to sympy Rational (requires Java init)."""
    return jBigIntegerPair2sympyRat(val.getNumerator(), val.getDenominator())


def jBigIntegerPair2sympyRat(numer, denom):
    """Convert Java BigInteger pair to sympy Rational (requires sympy)."""
    import sympy

    if numer.bitLength() <= 63:
        numer = numer.longValue()
    else:
        numer = str(numer.toString())

    if denom.bitLength() <= 63:
        denom = denom.longValue()
    else:
        denom = str(denom.toString())

    return sympy.Rational(numer, denom)
