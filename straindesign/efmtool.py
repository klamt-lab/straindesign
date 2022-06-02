# This code was derived from the efmtool_link package
import numpy
import jpype
import os
import subprocess
import sympy
import io
from contextlib import redirect_stdout, redirect_stderr
# import psutil

efmtool_jar = os.path.join(os.path.dirname(__file__), 'efmtool.jar')
jpype.addClassPath(efmtool_jar)
if not jpype.isJVMStarted():
    with redirect_stdout(io.StringIO()), redirect_stderr(
            io.StringIO()):  # suppress console output
        # mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')  # e.g. 4015976448
        # mem_mb = round(mem_bytes/(1024.**2)*0.75) # allow 75% of total memory for heap space
        # mem_mb = round(psutil.virtual_memory()[0]/(1024.**2)*0.75)
        # jpype.startJVM( jpype.getDefaultJVMPath() , f"-Xmx{mem_mb}m" )
        jpype.startJVM()
import jpype.imports

import ch.javasoft.smx.impl.DefaultBigIntegerRationalMatrix as DefaultBigIntegerRationalMatrix
import ch.javasoft.smx.ops.Gauss as Gauss
import ch.javasoft.metabolic.compress.CompressionMethod as CompressionMethod

subset_compression = CompressionMethod[:]([
    CompressionMethod.CoupledZero, CompressionMethod.CoupledCombine,
    CompressionMethod.CoupledContradicting
])
import ch.javasoft.metabolic.compress.StoichMatrixCompressor as StoichMatrixCompressor
import ch.javasoft.math.BigFraction as BigFraction
import java.math.BigInteger as BigInteger

jTrue = jpype.JBoolean(True)
jSystem = jpype.JClass("java.lang.System")

# try to find a working java executable
_java_executable = 'java'
try:
    cp = subprocess.run([_java_executable, '-version'],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)
    if cp.returncode != 0:
        _java_executable = ''
except:
    _java_executable = ''
if _java_executable == '':
    _java_executable = os.path.join(os.environ.get('JAVA_HOME', ''), "bin",
                                    "java")
    try:
        cp = subprocess.run([_java_executable, '-version'],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
        if cp.returncode != 0:
            _java_executable = ''
    except:
        _java_executable = ''
if _java_executable == '':
    import efmtool_link.efmtool_intern  # just to find java executable via jpype
    _java_executable = os.path.join(
        str(efmtool_link.efmtool_intern.jSystem.getProperty("java.home")),
        "bin", "java")


def basic_columns_rat(
        mx,
        tolerance=0
):  # mx is ch.javasoft.smx.impl.DefaultBigIntegerRationalMatrix
    if type(mx) is numpy.ndarray:
        mx = DefaultBigIntegerRationalMatrix(numpy_mat2jpypeArrayOfArrays(mx),
                                             jTrue, jTrue)
    row_map = jpype.JInt[mx.getRowCount(
    )]  # just a placeholder because we don't care about the row permutation here
    col_map = jpype.JInt[:](range(mx.getColumnCount()))
    rank = Gauss.getRationalInstance().rowEchelon(mx, False, row_map, col_map)

    return col_map[0:rank]


def numpy_mat2jpypeArrayOfArrays(npmat):
    rows = npmat.shape[0]
    cols = npmat.shape[1]
    jmat = jpype.JDouble[rows, cols]
    # for sparse matrices can use nonzero() here instead of iterating through everything
    for r in range(rows):
        for c in range(cols):
            jmat[r][c] = npmat[r, c]
    return jmat


def jpypeArrayOfArrays2numpy_mat(jmat):
    rows = len(jmat)
    cols = len(jmat[0])  # assumes all rows have the same number of columns
    npmat = numpy.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            npmat[r, c] = jmat[r][c]
    return npmat


def sympyRat2jBigIntegerPair(val):
    numer = val.p  # numerator
    if numer.bit_length() <= 63:
        numer = BigInteger.valueOf(numer)
    else:
        numer = BigInteger(str(numer))
    denom = val.q  # denominator
    if denom.bit_length() <= 63:
        denom = BigInteger.valueOf(denom)
    else:
        denom = BigInteger(str(denom))
    return (numer, denom)


def jBigFraction2sympyRat(val):
    return jBigIntegerPair2sympyRat(val.getNumerator(), val.getDenominator())


def jBigIntegerPair2sympyRat(numer, denom):
    if numer.bitLength() <= 63:
        numer = numer.longValue()
    else:
        numer = str(numer.toString())
    if denom.bitLength() <= 63:
        denom = denom.longValue()
    else:
        denom = str(denom.toString())
    return sympy.Rational(numer, denom)
