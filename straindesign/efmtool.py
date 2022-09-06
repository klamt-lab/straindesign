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
"""Functions for the compression of metabolic networks, taken from the efmtool_link package

Functions in this module not meant to be used outside the compression of networks. For a
the documentation of the efmtool compression provided by StrainDesign, refer to the networktools
module."""

import numpy
import jpype
import os
import subprocess
import sympy
import io
from contextlib import redirect_stdout, redirect_stderr
"""Initialization of the java machine, since efmtool compression is done in java."""
efmtool_jar = os.path.join(os.path.dirname(__file__), 'efmtool.jar')
jpype.addClassPath(efmtool_jar)
if not jpype.isJVMStarted():
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):  # suppress console output
        # mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')  # e.g. 4015976448
        # mem_mb = round(mem_bytes/(1024.**2)*0.75) # allow 75% of total memory for heap space
        # mem_mb = round(psutil.virtual_memory()[0]/(1024.**2)*0.75)
        # jpype.startJVM( jpype.getDefaultJVMPath() , f"-Xmx{mem_mb}m" )
        jpype.startJVM()
import jpype.imports

import ch.javasoft.smx.impl.DefaultBigIntegerRationalMatrix as DefaultBigIntegerRationalMatrix
import ch.javasoft.smx.ops.Gauss as Gauss
import ch.javasoft.metabolic.compress.CompressionMethod as CompressionMethod

subset_compression = CompressionMethod[:](
    [CompressionMethod.CoupledZero, CompressionMethod.CoupledCombine, CompressionMethod.CoupledContradicting])
import ch.javasoft.metabolic.compress.StoichMatrixCompressor as StoichMatrixCompressor
import ch.javasoft.math.BigFraction as BigFraction
import java.math.BigInteger as BigInteger

jTrue = jpype.JBoolean(True)
jSystem = jpype.JClass("java.lang.System")

# try to find a working java executable
_java_executable = 'java'
try:
    cp = subprocess.run([_java_executable, '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if cp.returncode != 0:
        _java_executable = ''
except:
    _java_executable = ''
if _java_executable == '':
    _java_executable = os.path.join(os.environ.get('JAVA_HOME', ''), "bin", "java")
    try:
        cp = subprocess.run([_java_executable, '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if cp.returncode != 0:
            _java_executable = ''
    except:
        _java_executable = ''
if _java_executable == '':
    import efmtool_link.efmtool_intern  # just to find java executable via jpype
    _java_executable = os.path.join(str(efmtool_link.efmtool_intern.jSystem.getProperty("java.home")), "bin", "java")


def basic_columns_rat(mx, tolerance=0):  # mx is ch.javasoft.smx.impl.DefaultBigIntegerRationalMatrix
    """efmtool: Translate matrix coefficients to rational numbers"""
    if type(mx) is numpy.ndarray:
        mx = DefaultBigIntegerRationalMatrix(numpy_mat2jpypeArrayOfArrays(mx), jTrue, jTrue)
    row_map = jpype.JInt[mx.getRowCount()]  # just a placeholder because we don't care about the row permutation here
    col_map = jpype.JInt[:](range(mx.getColumnCount()))
    rank = Gauss.getRationalInstance().rowEchelon(mx, False, row_map, col_map)

    return col_map[0:rank]


def numpy_mat2jpypeArrayOfArrays(npmat):
    """efmtool: Translate matrix to array of arrays"""
    rows = npmat.shape[0]
    cols = npmat.shape[1]
    jmat = jpype.JDouble[rows, cols]
    # for sparse matrices can use nonzero() here instead of iterating through everything
    for r in range(rows):
        for c in range(cols):
            jmat[r][c] = npmat[r, c]
    return jmat


def jpypeArrayOfArrays2numpy_mat(jmat):
    """efmtool: Translate array of arrays to numpy matrix"""
    rows = len(jmat)
    cols = len(jmat[0])  # assumes all rows have the same number of columns
    npmat = numpy.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            npmat[r, c] = jmat[r][c]
    return npmat


def sympyRat2jBigIntegerPair(val):
    """efmtool: Translate rational numbers to big integer pair"""
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
    """efmtool: Translate rational numbers to sympy rational numbers"""
    return jBigIntegerPair2sympyRat(val.getNumerator(), val.getDenominator())


def jBigIntegerPair2sympyRat(numer, denom):
    """efmtool: Translate big integer pair to sympy rational numbers"""
    if numer.bitLength() <= 63:
        numer = numer.longValue()
    else:
        numer = str(numer.toString())
    if denom.bitLength() <= 63:
        denom = denom.longValue()
    else:
        denom = str(denom.toString())
    return sympy.Rational(numer, denom)
