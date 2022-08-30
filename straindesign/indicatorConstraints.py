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

    def __init__(self, binv, A, b, sense, indicval):
        self.binv = binv  # index of binary variable
        self.A = A  # CPLEX: lin_expr,   left hand side coefficient row for indicator constraint
        self.b = b  # right hand side for indicator constraint
        self.sense = sense  # sense of the indicator constraint can be 'L', 'E', 'G' (lower-equal, equal, greater-equal)
        self.indicval = indicval  # value the binary variable takes when constraint is fulfilled
