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
from numpy import all
from typing import List, Tuple, Union, Set, FrozenSet
import re
from mcs.parse_constr import *
from optlang.interface import OPTIMAL, INFEASIBLE, UNBOUNDED

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
        module_sense: 'desired' or 'undesired'
        module_type: 'mcs_lin', 'mcs_bilvl', 'mcs_yield', 'optknock'
        constraints: Linear constraints: A v <= b, A v >= b, A v = b
                    (e.g. T v <= t with 'undesired' or D v <= d with 'desired')
    ----------
    Module type specific attributes:
        mcs_lin: <none>
        mcs_bilvl: inner_objective: Inner optimization expression
        mcs_yield: numerator: numerator of yield function,
                        denominator: denominator of yield function
        optknock: inner_objective: Inner optimization expression
                  outer_objective: Outer optimization expression
        robustknock:
        optcouple:
    Examples
    --------
        modules = [         mcs_module.MCS_Module(network,"mcs_lin",module_sense="undesired",constraints="R4 >= 1")]
        modules = [modules, mcs_module.MCS_Module(network,"mcs_lin",module_sense="desired",constraints="R3 >= 1")]
        ...
    """
    def __init__(self, model, module_type, *args, **kwargs):
        self.model = model
        self.module_type = module_type
        allowed_keys = {'module_sense', 'constraints','inner_objective','inner_opt_sense','outer_objective',
                        'outer_opt_sense','prod_id','skip_checks','min_gcp'}
        # set all keys passed in kwargs as properties of the SD_Module object
        for key,value in kwargs.items():
            if key in allowed_keys:
                setattr(self,key,value)
            else:
                raise Exception("Key "+key+" is not supported.")
        # set all undefined keys to None
        for key in allowed_keys:
            if key not in kwargs.keys():
                setattr(self,key,None)      
        
        # module sense must be desired or undesired when using mcs or else remain undefined/None.
        if 'mcs' in self.module_type and self.module_sense not in ["desired", "undesired"]:
            raise ValueError('"module_sense" must be "undesired" or "desired".')
        elif 'mcs' not in self.module_type:
            print('module_sense is ignored unless module_type is mcs_lin, mcs_bilvl or mcs_yield.')
            
        # check if there is sufficient information for each module type
        if self.module_type not in  ["mcs_lin", "mcs_bilvl", "mcs_yield", "optknock", "robustknock","optcouple"]:
            raise ValueError('"module_type" must be "mcs_lin", "mcs_bilvl", "mcs_yield", "optknock", "robustknock", "optcouple".')
        if (self.module_type == "mcs_bilvl") & (self.inner_objective == None):
            raise ValueError('When module type is "mcs_bilvl", an objective function must be provided.')
        elif (self.module_type == "mcs_yield") & (self.constraints==None):
            raise ValueError('When module type is "mcs_yield", a numerator and denominator must be provided.')
        elif (self.module_type in ["optknock","robustknock"]):
            if self.inner_opt_sense is None:
                self.inner_opt_sense = 'maximize'
            if self.outer_opt_sense is None:
                self.outer_opt_sense = 'maximize'
            elif self.inner_opt_sense not in ['minimize', 'maximize'] or self.outer_opt_sense not in ['minimize', 'maximize']:
                raise ValueError('Inner and outer optimization sense must be "minimize" or "maximize" (default).')
            if ((self.inner_objective == None) or (self.outer_objective == None)):
                raise ValueError('When module type is "optknock" or "robustknock", an inner and outer objective function must be provided.')
        elif (self.module_type == "optcouple"):
            if self.inner_opt_sense is None:
                self.inner_opt_sense = 'maximize'
            if self.inner_opt_sense not in ['minimize', 'maximize']:
                raise ValueError('Inner optimization sense must be "minimize" or "maximize" (default).')
            if self.inner_objective == None:
                raise ValueError('When module type is "optcouple", an inner objective function must be provided.')
            if self.prod_id == None:
                raise ValueError('When module type is "optcouple", the production reaction id must be provided.')

        reac_id = model.reactions.list_attr('id')

        # parse constraints and ensure they have the form:
        # [ [{'r1': -1, 'r3': 2}, '<=', 3],
        #   [{'r2': 1, 'r3': -1},  '=', 0]  ]
        if self.constraints is not None:
            self.constraints = parse_constraints(self.constraints,reac_id)
        else:
            self.constraints=[]
            
        # parse inner objective
        if self.inner_objective is not None:
            if type(self.inner_objective) is str:
                self.inner_objective = linexpr2dict(self.inner_objective,reac_id)

        # parse outer objective
        if self.outer_objective is not None:
            if type(self.outer_objective) is str:
                self.outer_objective = linexpr2dict(self.outer_objective,reac_id)

        # parse prod_id
        if self.prod_id is not None:
            if type(self.prod_id) is str:
                self.prod_id = linexpr2dict(self.prod_id,reac_id)

        # verify self.constraints
        if self.skip_checks is None or not self.skip_checks:
            from mcs import fba
            if fba(model,constraints=self.constraints).status == INFEASIBLE:
                raise Exception("There is no feasible solution of the model under the given constraints.")
            
            if (self.inner_objective is not None) and (not all([True if r in reac_id else False for r in self.inner_objective.keys()])):
                raise Exception("Inner objective invalid.")

            if (self.outer_objective is not None) and (not all([True if r in reac_id else False for r in self.outer_objective.keys()])):
                raise Exception("Outer objective invalid.")

            if (self.prod_id is not None) and (not all([True if r in reac_id else False for r in self.prod_id.keys()])):
                raise Exception("Production id (prod_id) invalid.")
            
            if (self.min_gcp is not None) and type(self.min_gcp) is not None:
                if type(self.min_gcp) is int:
                    self.min_gcp = float(self.min_gcp)
                else:
                    raise Exception("Minimum growth coupling potential (min_gcp).")
