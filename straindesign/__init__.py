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
"""StrainDesign package for computational metabolic engineering"""

from importlib.util import find_spec as module_exists
from .names import *
import logging


class DisableLogger():
    """Environment in which logging is disabled"""

    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)


avail_solvers = set()
if module_exists("swiglpk"):
    avail_solvers.add(GLPK)
if module_exists("cplex"):
    avail_solvers.add(CPLEX)
if module_exists("gurobipy"):
    avail_solvers.add(GUROBI)
if module_exists("pyscipopt"):
    avail_solvers.add(SCIP)

# Conditional eager JVM startup — required for stable JPype operation.
# The JVM must start before NumPy/OpenBLAS spawns worker threads, otherwise
# JNI calls crash with SIGBUS/SIGSEGV (jpype#808, jpype#934).
# No-op when jpype1 or Java is not installed (neither is a dependency).
# See developers_guide.md "efmtool_cmp_interface.py — JPype/JVM Initialization".
from .efmtool_cmp_interface import _start_jvm as _start_jvm
_start_jvm()
del _start_jvm

from .solver_interface import *
from .indicatorConstraints import *
from .pool import *

from .parse_constr import *
from .lptools import *
from .networktools import *
from .strainDesignModule import *
from .strainDesignSolutions import *
from .strainDesignProblem import *
from .strainDesignMILP import *
from .compute_strain_designs import *
