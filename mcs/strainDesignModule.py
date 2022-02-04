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
Strain design module (:class:`SD_Module`)
Strain design modules are used to describe strain design problems, 
e.g. desired or undesired flux states for MCS strain design.
"""

class SD_Module:
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
        module_type: 'mcs_lin', 'mcs_bilvl', 'mcs_yield'
        equation: String to specify linear constraints: A v <= b, A v >= b, A v = b
            (e.g. T v <= t with 'target' or D v <= d with 'desired')
    ----------
    Module type specific attributes:
        mcs_lin: <none>
        mcs_bilvl: inner_objective: Inner optimization expression
        mcs_yield: numerator: numerator of yield function,
                        denominator: denominator of yield function
    Examples
    --------
        modules = [         mcs_module.MCS_Module(network,"mcs_lin",module_sense="target",constraints="R4 >= 1")]
        modules = [modules, mcs_module.MCS_Module(network,"mcs_lin",module_sense="desired",constraints="R3 >= 1")]
        ...
    """
    def __init__(self, model, module_type, *args, **kwargs):
        self.model = model
        self.module_type = module_type
        allowed_keys = {'module_sense', 'constraints','inner_objective','numerator','denomin','lb','ub','skip_checks'}
        # set all keys passed in kwargs
        for key,value in kwargs.items():
            if key in allowed_keys:
                setattr(self,key,value)
            else:
                raise Exception("Key "+key+" is not supported.")
        # set all remaining keys to None
        for key in allowed_keys:
            if key not in kwargs.keys():
                setattr(self,key,None)

        if self.lb is None:
            self.lb = [r.lower_bound for r in model.reactions]
        if self.ub is None:
            self.ub = [r.upper_bound for r in model.reactions]        
        
        if 'mcs' in self.module_type and self.module_sense not in ["desired", "target"]:
            raise ValueError('"module_sense" must be "target" or "desired".')

        reac_id = model.reactions.list_attr('id')

        if self.constraints is not None:
            if "\n" in self.constraints:
                self.constraints = re.split(r"\n",self.constraints)
            if type(self.constraints) is not list:
                self.constraints = [self.constraints]
        else:
            self.constraints=[]

        # verify self.constraints
        if self.skip_checks is None or not self.skip_checks:
            try:
                for eq in self.constraints:
                    re.search('<=|>=|=',eq)
                    eq_sign = re.search('<=|>=|=',eq)[0]
                    split_eq = re.split('<=|>=|=',eq)
                    self.check_lhs(split_eq[0],reac_id)
            except:
                raise NameError('self.constraints must contain a sign (<=,=,>=)')

        if self.module_type not in  ["mcs_lin", "mcs_bilvl", "mcs_yield"]:
            raise ValueError('"module_type" must be "mcs_lin", "mcs_bilvl" or "mcs_yield".')

        if (self.module_type == "mcs_bilvl") & (self.inner_objective == None):
            raise ValueError('When module type is "mcs_bilvl", an objective function must be provided.')
        elif (self.module_type == "mcs_yield") & ((self.numerator==None) & (self.denomin==None)):
            raise ValueError('When module type is "mcs_yield", a numerator and denominator must be provided.')


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