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
By default, the pure Python 'sparse_rref' compression backend is used.
The Java EFMTool backend is available via compression_backend='efmtool_rref' (requires jpype1).

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

    This function is called lazily only when compression_backend='efmtool_rref'.
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
                          "Or use the default Python compression (compression_backend='sparse_rref').")

    if not _check_sympy_available():
        raise ImportError("sympy is not installed. Legacy Java compression requires sympy.\n"
                          "Install with: pip install sympy\n"
                          "Or use the default Python compression (compression_backend='sparse_rref').")

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
            # Suppress faulthandler during JVM startup to prevent ugly
            # "Windows fatal exception: access violation" messages.
            # On Windows, jpype.startJVM() can trigger an access violation
            # that is caught by Python's structured exception handler, but
            # faulthandler prints a stack trace before the exception is raised.
            import faulthandler as _fh
            _fh_was_enabled = _fh.is_enabled()
            if _fh_was_enabled:
                _fh.disable()
            try:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    jpype.startJVM()
            finally:
                if _fh_was_enabled:
                    _fh.enable()
        except Exception as e:
            extra_info = ""
            if not os.environ.get("JAVA_HOME"):
                extra_info = " JAVA_HOME is not defined."
            raise RuntimeError(
                "Failed to start JVM. Please ensure that Java (OpenJDK) is installed." + extra_info +
                " If using conda, install openjdk from conda-forge and set JAVA_HOME to the OpenJDK installation path.") from e

    # Load Java classes via JClass (not `import` statements) to avoid
    # jpype.imports.find_spec which can segfault on Windows.
    try:
        DefaultBigIntegerRationalMatrix = jpype.JClass('ch.javasoft.smx.impl.DefaultBigIntegerRationalMatrix')
        Gauss = jpype.JClass('ch.javasoft.smx.ops.Gauss')
        CompressionMethod = jpype.JClass('ch.javasoft.metabolic.compress.CompressionMethod')
        StoichMatrixCompressor = jpype.JClass('ch.javasoft.metabolic.compress.StoichMatrixCompressor')
        BigFraction = jpype.JClass('ch.javasoft.math.BigFraction')
        BigInteger = jpype.JClass('java.math.BigInteger')
    except Exception as e:
        raise RuntimeError(
            "Failed to load EFMTool Java classes. The JVM started but the efmtool.jar "
            "classes could not be loaded. Use compression_backend='sparse_rref' instead.") from e

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


def compress_model_java(model, suppressed_reactions=None):
    """Legacy Java compression using jpype (requires jpype and sympy).

    Args:
        model: COBRA model (will be modified in place)
        suppressed_reactions: Set of reaction IDs to exclude from compression.
            These reactions are kept as standalone entries with identity mapping.
            Used to protect reactions referenced in strain design constraints
            from being deleted by the Java compressor's CoupledContradicting logic.

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

    suppressed_set = set(suppressed_reactions) if suppressed_reactions else set()
    num_met = len(model.metabolites)
    num_reac = len(model.reactions)
    old_reac_ids = [r.id for r in model.reactions]

    # Build mapping between active (non-suppressed) indices and model indices
    active_to_model = [i for i in range(num_reac) if old_reac_ids[i] not in suppressed_set]
    num_active = len(active_to_model)

    stoich_mat = DefaultBigIntegerRationalMatrix(num_met, num_active)
    reversible = jpype.JBoolean[:]([model.reactions[active_to_model[ai]].reversibility for ai in range(num_active)])
    flipped = set()
    for ai in range(num_active):
        mi = active_to_model[ai]
        if model.reactions[mi].upper_bound <= 0:
            model.reactions[mi] *= -1
            flipped.add(ai)
            logging.debug("Flipped " + model.reactions[mi].id)
        for k, v in model.reactions[mi]._metabolites.items():
            n, d = sympyRat2jBigIntegerPair(v)
            stoich_mat.setValueAt(model.metabolites.index(k.id), ai, BigFraction(n, d))

    # Compress active reactions only
    smc = StoichMatrixCompressor(subset_compression)
    reacNames = jpype.JString[:]([old_reac_ids[active_to_model[ai]] for ai in range(num_active)])
    comprec = smc.compress(stoich_mat, reversible, jpype.JString[num_met], reacNames, None)
    subset_matrix = jpypeArrayOfArrays2numpy_mat(comprec.post.getDoubleRows())

    # subset_matrix shape: (num_active, num_compressed)
    del_model = np.zeros(num_reac, dtype=bool)

    # Mark zero-flux active reactions for deletion
    for ai in range(num_active):
        if not np.any(subset_matrix[ai, :]):
            del_model[active_to_model[ai]] = True

    for j in range(subset_matrix.shape[1]):
        rxn_ai = subset_matrix[:, j].nonzero()[0]
        if len(rxn_ai) == 0:
            continue
        r0_mi = active_to_model[rxn_ai[0]]
        model.reactions[r0_mi].subset_rxns = []
        model.reactions[r0_mi].subset_stoich = []
        # Scale objective coefficient by POST factors
        combined_obj = 0.0
        for ai in rxn_ai:
            mi = active_to_model[ai]
            factor = jBigFraction2sympyRat(comprec.post.getBigFractionValueAt(ai, j))
            # Accumulate objective contribution before scaling
            combined_obj += model.reactions[mi].objective_coefficient * float(factor)
            model.reactions[mi] *= factor
            if model.reactions[mi].lower_bound not in (0, -float('inf')):
                model.reactions[mi].lower_bound /= abs(subset_matrix[ai, j])
            if model.reactions[mi].upper_bound not in (0, float('inf')):
                model.reactions[mi].upper_bound /= abs(subset_matrix[ai, j])
            model.reactions[r0_mi].subset_rxns.append(mi)
            if ai in flipped:
                model.reactions[r0_mi].subset_stoich.append(-factor)
            else:
                model.reactions[r0_mi].subset_stoich.append(factor)
        model.reactions[r0_mi].objective_coefficient = combined_obj
        for ai in rxn_ai[1:]:
            mi = active_to_model[ai]
            if len(model.reactions[r0_mi].id) + len(model.reactions[mi].id) < 220 and model.reactions[r0_mi].id[-3:] != '...':
                model.reactions[r0_mi].id += '*' + model.reactions[mi].id
            elif not model.reactions[r0_mi].id[-3:] == '...':
                model.reactions[r0_mi].id += '...'
            model.reactions[r0_mi] += model.reactions[mi]
            if model.reactions[mi].lower_bound > model.reactions[r0_mi].lower_bound:
                model.reactions[r0_mi].lower_bound = model.reactions[mi].lower_bound
            if model.reactions[mi].upper_bound < model.reactions[r0_mi].upper_bound:
                model.reactions[r0_mi].upper_bound = model.reactions[mi].upper_bound
            del_model[mi] = True

    # Add suppressed reactions as standalone entries
    from fractions import Fraction
    for mi in range(num_reac):
        if old_reac_ids[mi] in suppressed_set:
            model.reactions[mi].subset_rxns = [mi]
            model.reactions[mi].subset_stoich = [Fraction(1)]

    # Delete reactions (reverse order to preserve indices)
    del_indices = np.where(del_model)[0]
    for i in range(len(del_indices) - 1, -1, -1):
        model.reactions[del_indices[i]].remove_from_model(remove_orphans=True)

    # Build rational_map
    rational_map = {}
    for j in range(len(model.reactions)):
        rational_map[model.reactions[j].id] = {
            old_reac_ids[mi]: v
            for mi, v in zip(model.reactions[j].subset_rxns, model.reactions[j].subset_stoich)
        }
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
