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
    >>> ADD EXAMPLE HERE
    """
    def __init__(self, model, module_sense, module_type, equations, inner_objective=None, numerator=None, denomin=None, *args, **kwargs):
        self.module_sense = str(module_sense)
        self.module_type  = str(module_type)
        
        if self.module_sense not in ["desired", "target"]:
            raise ValueError('"module_sense" must be "target" or "desired".')

        reac_id = model.reactions.list_attr('id')

        self.equations = equations
        # verify equations
        try:
            eqs = re.split(r"\n",equations)
            for eq in eqs:
                re.search('<=|>=|=',eq)
                eq_sign = re.search('<=|>=|=',eq)
                eq_sign = eq_sign[0]
                split_eq = re.split('<=|>=|=',eq)
                self.check_lhs(split_eq[0])
        except:
            raise NameError('Equations must contain a sign (<=,=,>=)')

        if self.module_type not in  ["lin_constraints", "bilev_w_constr", "yield_w_constr"]:
            raise ValueError('"module_type" must be "lin_constraints", "bilev_w_constr" or "yield_w_constr".')
    
        self.inner_objective = inner_objective
        self.numerator = numerator
        self.denomin = denomin

        if (self.module_type == "bilev_w_constr") & (self.inner_objective == None):
            raise ValueError('When module type is "bilev_w_constr", an objective function must be provided.')
        elif (self.module_type == "yield_w_constr") & ((self.numerator==None) & (self.denomin==None)):
            raise ValueError('When module type is "bilev_w_constr", a numerator and denominator must be provided.')


    def check_lhs(self, equation: str) -> str:
        errors = ""

        semantics = []
        reaction_ids = []
        last_part = ""
        counter = 1
        for char in equation+" ":
            if (char == " ") or (char in ("*", "/", "+", "-")) or (counter == len(equation+" ")):
                if last_part != "":
                    try:
                        float(last_part)
                    except ValueError:
                        reaction_ids.append(last_part)
                        semantics.append("reaction")
                    else:
                        semantics.append("number")
                    last_part = ""

                if counter == len(equation+" "):
                    break

            if char in "*":
                semantics.append("multiplication")
            elif char in "/":
                semantics.append("division")
            elif char in ("+", "-"):
                semantics.append("dash")
            elif char not in " ":
                last_part += char
            counter += 1

        if len(reaction_ids) == 0:
            errors += f"EQUATION ERROR in {equation}:\nNo reaction ID is given in the equation\n"

        if semantics.count("division") > 1:
            errors += f"ERROR in {equation}:\nAn equation must not have more than one /"

        last_is_multiplication = False
        last_is_division = False
        last_is_dash = False
        last_is_reaction = False
        prelast_is_reaction = False
        prelast_is_dash = False
        last_is_number = False
        is_start = True
        for semantic in semantics:
            if is_start:
                if semantic in ("multiplication", "division"):
                    errors += f"ERROR in {equation}:\nAn equation must not start with * or /"
                is_start = False

            if (last_is_multiplication or last_is_division) and (semantic in ("multiplication", "division")):
                errors += f"ERROR in {equation}:\n* or / must not follow on * or /\n"
            if last_is_dash and (semantic in ("multiplication", "division")):
                errors += f"ERROR in {equation}:\n* or / must not follow on + or -\n"
            if last_is_number and (semantic == "reaction"):
                errors += f"ERROR in {equation}:\nA reaction must not directly follow on a number without a mathematical operation\n"
            if last_is_reaction and (semantic == "reaction"):
                errors += f"ERROR in {equation}:\nA reaction must not follow on a reaction ID\n"
            if last_is_number and (semantic == "number"):
                errors += f"ERROR in {equation}:\nA number must not follow on a number ID\n"

            if prelast_is_reaction and last_is_multiplication and (semantic == "reaction"):
                errors += f"ERROR in {equation}:\nTwo reactions must not be multiplied together\n"

            if last_is_reaction:
                prelast_is_reaction = True
            else:
                prelast_is_reaction = False

            if last_is_dash:
                prelast_is_dash = True
            else:
                prelast_is_dash = False

            last_is_multiplication = False
            last_is_division = False
            last_is_dash = False
            last_is_reaction = False
            last_is_number = False
            if semantic == "multiplication":
                last_is_multiplication = True
            elif semantic == "division":
                last_is_division = True
            elif semantic == "reaction":
                last_is_reaction = True
            elif semantic == "dash":
                last_is_dash = True
            elif semantic == "number":
                last_is_number = True

        if last_is_dash or last_is_multiplication or last_is_division:
            errors += (f"ERROR in {equation}:\nA reaction must not end "
                       f"with a +, -, * or /")

        if prelast_is_dash and last_is_number:
            errors += (f"ERROR in {equation}:\nA reaction must not end "
                       f"with a separated number term only")

        with self.appdata.project.cobra_py_model as model:
            model_reaction_ids = [x.id for x in model.reactions]
            for reaction_id in reaction_ids:
                if reaction_id not in model_reaction_ids:
                    errors += (f"ERROR in {equation}:\nA reaction with "
                               f"the ID {reaction_id} does not exist in the model\n")

        return errors

    def check_rhs(self, equation: str) -> str:
        try:
            float(equation)
        except ValueError:
            error = f"ERROR in {equation}:\nRight equation must be a number\n"
            return error
        else:
            return ""

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