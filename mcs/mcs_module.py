# 2021 Max Planck institute for dynamics of complex technical systems.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy
import scipy
import cobra
import itertools
from typing import List, Tuple, Union, Set, FrozenSet
import time
import sympy

"""
MCS module (:class:`MCS_Module`)
MCS modules are used to describe desired or undesired flux states for MCS strain design.
"""

class MCS_Module:
    """Modules to describe desired or undesired flux states for MCS strain design.
    There are three kinds of flux states that can be described
          1. The wildtype model, constrainted with additional inequalities:
             e.g.: T v <= t.
           2. The wildtype model at a specific optimum, constrained with additional inequalities
              e.g.: c*v = optimal, T v <= t
           3. A yield range:
              e.g.: e*v/f*v <= t (Definition of A and b is not required)
     fields:

    Attributes
    ----------
        module_sense: 'desired' or 'target'
        module_type: 'lin_constraints', 'bilev_w_constr', 'yield_w_constr'
        A,b: Matrix/vector to specify additional linear constraints: A v <= b
            (e.g. T v <= t with 'target' or D v <= d with 'desired')
    ----------
    Module type specific attributes:
        lin_constraints: <none>
        bilev_w_constr: c: Inner optimization vector
        yield_w_constr: e: numerator of yield function,
                        f: denominator of yield function
    Examples
    --------
    >>> ADD EXAMPLE HERE
    """
    def __init__(self, module_sense, module_type, A=None, b=None, c=None, e=None, f=None, *args, **kwargs):
        self.module_sense = str(module_sense)
        self.module_type  = str(module_type)
        
        if self.module_sense not in ["desired", "target"]:
            raise ValueError('"module_sense" must be "target" or "desired".')

        self.A = A
        self.b = b

        if self.module_type not in  ["lin_constraints", "bilev_w_constr", "yield_w_constr"]:
            raise ValueError('"module_type" must be "lin_constraints", "bilev_w_constr" or "yield_w_constr".')
    
        self.c = c
        self.e = e
        self.f = f

        if self.module_type == "bilev_w_constr" & self.c == None:
            raise ValueError('When module type is "bilev_w_constr", an objective function c must be provided.')
        elif self.module_type == "yield_w_constr" & (self.e==None & self.f==None):
            raise ValueError('When module type is "bilev_w_constr", coefficients for numerator e and denominator f must be provided.')

#        self.__test_valid_lower_bound(type, self._lb, name)
