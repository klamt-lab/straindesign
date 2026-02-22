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
"""EFMtool compression interface for straindesign.

This module provides compression utilities for metabolic networks.
By default, the pure Python 'sparse_rref' backend is used.
The Java EFMTool backend is available via backend='efmtool_rref' (requires jpype1).

For the documentation of the compression API provided by StrainDesign,
refer to straindesign.compression.compress_model.
"""

import logging
import numpy as np
import os
import sys

# =============================================================================
# Pure Python Implementation (Default)
# =============================================================================


def basic_columns_rat(mx, tolerance=0):
    """Find basic columns using exact rational arithmetic (FLINT or sympy)."""
    from .compression import basic_columns_from_numpy
    return basic_columns_from_numpy(mx)


# =============================================================================
# Lazy Java Initialization
# =============================================================================

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

    This function is called lazily only when backend='efmtool_rref'.
    Raises ImportError if jpype is not installed.
    """
    global _JAVA_INITIALIZED
    global DefaultBigIntegerRationalMatrix, Gauss, CompressionMethod
    global StoichMatrixCompressor, BigFraction, BigInteger
    global subset_compression, jTrue, jSystem

    if _JAVA_INITIALIZED:
        return

    if not _check_jpype_available():
        raise ImportError("jpype1 is not installed. Legacy Java compression requires jpype1.\n"
                          "Install with: pip install jpype1\n"
                          "Or use the default Python compression (backend='sparse_rref').")

    if not _check_sympy_available():
        raise ImportError("sympy is not installed. Legacy Java compression requires sympy.\n"
                          "Install with: pip install sympy\n"
                          "Or use the default Python compression (backend='sparse_rref').")

    import jpype
    import io
    from contextlib import redirect_stdout, redirect_stderr

    # Add efmtool.jar to classpath
    efmtool_jar = os.path.join(os.path.dirname(__file__), 'efmtool.jar')
    if not os.path.exists(efmtool_jar):
        raise FileNotFoundError(f"efmtool.jar not found at {efmtool_jar}. "
                                "Legacy Java compression requires the efmtool.jar file.")

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
                " If using conda, install openjdk from conda-forge and set JAVA_HOME to the OpenJDK installation path.") from e

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
        [CompressionMethod.CoupledZero, CompressionMethod.CoupledCombine, CompressionMethod.CoupledContradicting])
    jTrue = jpype.JBoolean(True)
    jSystem = jpype.JClass("java.lang.System")

    _JAVA_INITIALIZED = True


# =============================================================================
# Java Conversion Utilities
# =============================================================================


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
    npmat = np.zeros((rows, cols))
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


# =============================================================================
# Legacy Java Compression Functions
# =============================================================================


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

    if isinstance(mx, np.ndarray):
        mx = DefaultBigIntegerRationalMatrix(numpy_mat2jpypeArrayOfArrays(mx), jTrue, jTrue)

    row_map = jpype.JInt[mx.getRowCount()]
    col_map = jpype.JInt[:](range(mx.getColumnCount()))
    rank = Gauss.getRationalInstance().rowEchelon(mx, False, row_map, col_map)

    return col_map[0:rank]


def compress_model_java(model):
    """Legacy Java compression using jpype (requires jpype and sympy).

    Args:
        model: COBRA model (will be modified in place)

    Returns:
        dict: Reaction map from compressed to original reactions with scaling factors
    """
    import jpype
    from .networktools import stoichmat_coeff2rational

    # Initialize Java if not already done
    _init_java()

    # Convert to rational coefficients for Java
    stoichmat_coeff2rational(model)

    for r in model.reactions:
        r.gene_reaction_rule = ''
    num_met = len(model.metabolites)
    num_reac = len(model.reactions)
    old_reac_ids = [r.id for r in model.reactions]
    stoich_mat = DefaultBigIntegerRationalMatrix(num_met, num_reac)
    reversible = jpype.JBoolean[:]([r.reversibility for r in model.reactions])
    flipped = []
    for i in range(num_reac):
        if model.reactions[i].upper_bound <= 0:
            model.reactions[i] *= -1
            flipped.append(i)
            logging.debug("Flipped " + model.reactions[i].id)
        for k, v in model.reactions[i]._metabolites.items():
            n, d = sympyRat2jBigIntegerPair(v)
            stoich_mat.setValueAt(model.metabolites.index(k.id), i, BigFraction(n, d))
    # compress
    smc = StoichMatrixCompressor(subset_compression)
    reacNames = jpype.JString[:](model.reactions.list_attr('id'))
    comprec = smc.compress(stoich_mat, reversible, jpype.JString[num_met], reacNames, None)
    subset_matrix = jpypeArrayOfArrays2numpy_mat(comprec.post.getDoubleRows())
    del_rxns = np.logical_not(np.any(subset_matrix, axis=1))
    for j in range(subset_matrix.shape[1]):
        rxn_idx = subset_matrix[:, j].nonzero()[0]
        r0 = rxn_idx[0]
        model.reactions[r0].subset_rxns = []
        model.reactions[r0].subset_stoich = []
        # Scale objective coefficient by POST factors
        combined_obj = 0.0
        for r in rxn_idx:
            factor = jBigFraction2sympyRat(comprec.post.getBigFractionValueAt(r, j))
            # Accumulate objective contribution before scaling
            combined_obj += model.reactions[r].objective_coefficient * float(factor)
            model.reactions[r] *= factor
            if model.reactions[r].lower_bound not in (0, -float('inf')):
                model.reactions[r].lower_bound /= abs(subset_matrix[r, j])
            if model.reactions[r].upper_bound not in (0, float('inf')):
                model.reactions[r].upper_bound /= abs(subset_matrix[r, j])
            model.reactions[r0].subset_rxns.append(r)
            if r in flipped:
                model.reactions[r0].subset_stoich.append(-factor)
            else:
                model.reactions[r0].subset_stoich.append(factor)
        model.reactions[r0].objective_coefficient = combined_obj
        for r in rxn_idx[1:]:
            if len(model.reactions[r0].id) + len(model.reactions[r].id) < 220 and model.reactions[r0].id[-3:] != '...':
                model.reactions[r0].id += '*' + model.reactions[r].id
            elif not model.reactions[r0].id[-3:] == '...':
                model.reactions[r0].id += '...'
            model.reactions[r0] += model.reactions[r]
            if model.reactions[r].lower_bound > model.reactions[r0].lower_bound:
                model.reactions[r0].lower_bound = model.reactions[r].lower_bound
            if model.reactions[r].upper_bound < model.reactions[r0].upper_bound:
                model.reactions[r0].upper_bound = model.reactions[r].upper_bound
            del_rxns[r] = True
    del_rxns = np.where(del_rxns)[0]
    for i in range(len(del_rxns) - 1, -1, -1):
        model.reactions[del_rxns[i]].remove_from_model(remove_orphans=True)
    subT = np.zeros((num_reac, len(model.reactions)))
    rational_map = {}
    for j in range(subT.shape[1]):
        subT[model.reactions[j].subset_rxns, j] = [float(v) for v in model.reactions[j].subset_stoich]
        rational_map.update(
            {model.reactions[j].id: {
                 old_reac_ids[i]: v for i, v in zip(model.reactions[j].subset_rxns, model.reactions[j].subset_stoich)
             }})
    return rational_map


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Pure Python
    'basic_columns_rat',
    # Java initialization
    '_init_java',
    '_check_jpype_available',
    # Java compression
    'basic_columns_rat_java',
    'compress_model_java',
    # Java conversion utilities
    'numpy_mat2jpypeArrayOfArrays',
    'jpypeArrayOfArrays2numpy_mat',
    'sympyRat2jBigIntegerPair',
    'jBigFraction2sympyRat',
    'jBigIntegerPair2sympyRat',
]
