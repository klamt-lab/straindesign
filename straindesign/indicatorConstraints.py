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
"""Class for indicator contraints (IndicatorConstraints)"""


class IndicatorConstraints:
    """A class for storing indicator contraints
    
    This class is a container for indicator constraints. Indicator constraints are used to link the fulfillment
    of a constraint to an indicating variable. For instance the indicator constraint:    
    z = 1 -> 2*d - 1*e <= 3
    can be described as: 
    If z=1, then 2*d - 1*e <= 3
    An alternative formulation of this association is possible with a bigM constraint:
    2*d - 1*e - z*M <= 3, with M = very large
    The constraint z = 0 -> 2*d - 1*e <= 3 would translate to 2*d - 1*e + z*M <= 3 + M
    Generally, indicator constraints are preferred over bigM, because they provide better numerical stability.
    However, not all solvers support indicator constraints

    Indicator constraints have the form:
    x_binv = indicval -> a * x <sense> b
    e.g.,: x_35 = 1 -> 2 * x_2 + 3 *x_3 'L' 6
                                        (<=)
    This class contains a set of indicator constraints:
    x_binv_1 = indicval_1 -> A_1 * x <sense_1> b_1
    x_binv_2 = indicval_2 -> A_2 * x <sense_2> b_2
    ...

    Example: 
        ic = IndicatorConstraints(binv, A, b, sense, indicval)
                    
    Args:
        binv (list of int): (e.g.: [25, 27, 30])
            The index of the binary, indicating variables (indicators) for all indicator constraints.
            Integers are allowed ot occur more than once.
        
        A (sparse.csr_matrix):
            Coefficient vectors for all indicator constraints, stored in one matrix, whereas
            each row is used in one indicator constraint. 
            (num_columns = number of variables, num_rows = number of indicator constraints)

        b (list of float):
            Right hand sides of all indicator constraints. (e.g.: [0.1, 2.0, 3])
            
        sense (str):
            (In)equality signs for all indicator constraints. 
            'L'ess or equal, 'E'qual or, 'G'reater or equal (e.g.: 'EEGLGLGE')
            
        indicval (list of int):
            Indicator values for all indicator constraints. Which value of the indicator 
            enforces the constraint, 0 or 1? (e.g., 0001010110)
            
    Returns:
        (IndicatorConstraints):
        An object of the IndicatorConstraints class to pass indicator constraints.
    """

    def __init__(self, binv, A, b, sense, indicval):
        self.binv = binv  # index of binary variable
        self.A = A  # CPLEX: lin_expr,   left hand side coefficient row for indicator constraint
        self.b = b  # right hand side for indicator constraint
        self.sense = sense  # sense of the indicator constraint can be 'L', 'E', 'G' (lower-equal, equal, greater-equal)
        self.indicval = indicval  # value the binary variable takes when constraint is fulfilled
