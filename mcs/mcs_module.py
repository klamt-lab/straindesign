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
import re
import sympy
from mcs.constr2mat import *

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
              e.g.: objective*v = optimal, T v <= t
           3. A yield range:
              e.g.: numerator*v/denominator*v <= t (Definition of A and b is not required)
     fields:

    Attributes
    ----------
        module_sense: 'desired' or 'target'
        module_type: 'lin_constraints', 'bilev_w_constr', 'yield_w_constr'
        equation: String to specify linear constraints: A v <= b, A v >= b, A v = b
            (e.g. T v <= t with 'target' or D v <= d with 'desired')
    ----------
    Module type specific attributes:
        lin_constraints: <none>
        bilev_w_constr: inner_objective: Inner optimization vector
        yield_w_constr: numerator: numerator of yield function,
                        denominator: denominator of yield function
    Examples
    --------
        modules = [         mcs_module.MCS_Module(network,"target","lin_constraints","R4 >= 1")]
        modules = [modules, mcs_module.MCS_Module(network,"desired","lin_constraints","R3 >= 1")]
        ...
    """
    def __init__(self, model, module_sense, module_type, equations, inner_objective=None, numerator=None, \
                 denomin=None, lb =[], ub = [], *args, **kwargs):
        self.module_sense = str(module_sense)
        self.module_type  = str(module_type)
        
        if self.module_sense not in ["desired", "target"]:
            raise ValueError('"module_sense" must be "target" or "desired".')

        reac_id = model.reactions.list_attr('id')

        if "\n" in equations:
            equations = re.split(r"\n",equations)
        if type(equations) is not list:
            equations = [equations]
        self.equations = equations
        # verify equations
        try:
            for eq in equations:
                re.search('<=|>=|=',eq)
                eq_sign = re.search('<=|>=|=',eq)[0]
                split_eq = re.split('<=|>=|=',eq)
                self.check_lhs(split_eq[0],reac_id)
        except:
            raise NameError('Equations must contain a sign (<=,=,>=)')

        if self.module_type not in  ["lin_constraints", "bilev_w_constr", "yield_w_constr"]:
            raise ValueError('"module_type" must be "lin_constraints", "bilev_w_constr" or "yield_w_constr".')
    
        self.inner_objective = inner_objective
        self.numerator = numerator
        self.denomin = denomin
        self.lb = lb
        self.ub = ub

        if (self.module_type == "bilev_w_constr") & (self.inner_objective == None):
            raise ValueError('When module type is "bilev_w_constr", an objective function must be provided.')
        elif (self.module_type == "yield_w_constr") & ((self.numerator==None) & (self.denomin==None)):
            raise ValueError('When module type is "bilev_w_constr", a numerator and denominator must be provided.')


    def check_lhs(self, lhs: str, model_reac_ids: List) -> str:
        ridx = [re.sub(r'^(\s|-|\+|\.|\()*|(\s|-|\+|\.|\))*$','',part) for part in lhs.split()]
        # identify reaction identifiers by comparing with models reaction list
        ridx = [r for r in ridx if r in model_reac_ids]
        if not len(ridx) == len(set(ridx)): # check for duplicates
            raise Exception("Reaction identifiers may only occur once in each linear expression.")
        # TODO: Add more checks (e.g.: For ratios etc.)

    def check_rhs(self, equation: str) -> str:
        try:
            float(equation)
        except ValueError:
            error = f"ERROR in {equation}:\nRight equation must be a number\n"
            return error
        else:
            return ""

    def set_model_lb_ub(self, lb, ub):
        self.lb = lb
        self.ub = ub

    def check_for_mcs_equation_errors(self) -> str:
        errors = ""
        rows = self.target_list.rowCount()
        for i in range(0, rows):
            target_left = self.target_list.cellWidget(i, 1).text()
            errors += self.check_left_mcs_equation(target_left)
            target_right = self.target_list.cellWidget(i, 3).text()
            errors += self.check_right_mcs_equation(target_right)

        rows = self.desired_list.rowCount()
        for i in range(0, rows):
            desired_left = self.desired_list.cellWidget(i, 1).text()
            if len(desired_left) > 0:
                errors += self.check_left_mcs_equation(desired_left)

            desired_right = self.desired_list.cellWidget(i, 3).text()
            if len(desired_right) > 0:
                errors += self.check_right_mcs_equation(desired_right)
        return errors