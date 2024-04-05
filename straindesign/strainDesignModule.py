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
#
"""Class: strain design module (SDModule)"""

from numpy import all
from copy import deepcopy
from typing import List, Dict, Tuple, Union, Set, FrozenSet
from straindesign.parse_constr import *
from straindesign.names import *


class SDModule(Dict):
    """
    Strain design modules are used to specify the goal of a strain design computation
    
    (Lists of) SDModule objects are passed to compute_strain_design to specify the goal strain design computation.
    Strain design modules indicate the appraoch that should be used (OptKnock, RobustKnock, OptCouple or MCS) and
    the parameters for each approach. In each strain design computation, one of the following modules can be used
    at most once: OptKnock, RobustKnock and OptCouple. Additionally an arbitrary number of MCS modules (PROTECT or 
    SUPPRESS) may me used. The global objective of a strain design computation depends on the specified modules.
    If an OptKnock or RobustKnock module is used, the global objective function will be defined by 'outer_objective'
    and 'outer_opt_sense'. If OptCouple is used, the global objective is derived from by 'inner_objective' and 
    'inner_opt_sense'. If only MCS-like modules (suppress, protect) are used in a computation, the number of
    interventions is globally minimized.
    
    In the following, the modules and their mandatory/optional arguments are presented in detail.
    
    OptKnock:
        Globally maximize an *outer objective* subjected to the maximization/optimization of an *inner objective*. 
        For instance, maximize product synthesis, assuming that the strain maximizes its growth rate. When used 
        with the 'best' or 'populate' approach (see compute_strain_designs), this module will guarantee that the
        found intervention set allows for the highest possible outer objective (e.g., production) under the premise
        that the inner objective (growth) is forced to be maximal. This means that the *production potential* is
        maximal. However it does not necessarily mean that production is enforced at high growth rates. For enforced
        growth coupling, refer to the other module types. Additional constraints can be used to impose certain
        properties on the designed strains: constraints='growth >= 0.5' will guarantee that the designed strain
        can reach growth rates above 0.5. constraints=['growth >= 0.5', 'EX_byprod_e <= 2'] additionally guarantees
        that the synthesis rate of a by-product stays below 2 at growth maximal flux states. Alternative inner or
        outer objective functions (e.g., ATP maintenance) can be used for diffent types of strain design.
                
        mandatory arguments: model, module_type='optknock', inner_objective, outer_objective
        optional arguments: constraints, inner_opt_sense, outer_opt_sense, skip_checks
        (Detailed description of the arguments follow below)
    
    RobustKnock:
        Globally minimize the maximum of an *outer objective* subjected to the maximization/optimization of an *inner 
        objective*. For instance, maximize the minimal product synthesis rate, assuming that the strain maximizes its 
        growth rate. When used with the 'best' or 'populate' approach (see compute_strain_designs), this module will 
        guarantee that the found intervention set enforces that the minimum of the outer objective (e.g., maximal 
        production) is maximal (max-min), given that the inner objective (growth) is forced to be maximal. This means
        that the *minimal guaranteed production* is maximal and production is guaranteed at growth-maximal flux states.
        Additional constraints can be used to impose certain properties on the designed strains: 
        constraints='growth >= 0.5' will guarantee that the designed strain can reach growth rates above 0.5/h. 
        constraints=['growth >= 0.5', 'EX_byprod_e<=2'] additionally guarantees that the synthesis rate of a by-product 
        stays below 2 mmol/gCDW/h at growth maximal flux states.

        mandatory arguments: model, module_type='robustknock', inner_objective, outer_objective
        optional arguments: constraints, inner_opt_sense, outer_opt_sense, skip_checks, reac_ids
        (Detailed description of the arguments follow below)
        
    OptCouple:
        Globally maximize the *growth-coupling potential*, that is, the difference between the maximal growth rate 
        without product synthesis and the maximum overall growth rate (with product synthesis). This strain design
        approach often leads to directionally growth-coupled strain designs. Again, alternative definitions of this 
        objective are possible to, for instance, couple production to the synthesis of ATP. To specify the product
        for which coupling should be engineered, the reaction identifier of the product exchange (pseudo)reaction
        is passed through the *prod_id* parameter (see below). One may addidionally define a minimum growth-coupling
        potential through the parameter *min_gcp*.
                
        mandatory arguments: model, module_type='optcouple', inner_objective, prod_id
        optional arguments: constraints, inner_opt_sense, min_gcp, skip_checks, reac_ids
        (Detailed description of the arguments follow below)
        
    Suppress:
        MCS-like suppression of a subspace of all steady-state flux vectors. The 'constraints' parameter is used to
        descirbe the flux states that should not be eliminated from the flux space. Depending on the goal of the
        strain design computation, undesired flux states may be those with production of an undesired by-product,
        low product synthesis rates, low product yields or even flux states with microbial growth. In addition to
        constraints, an inner objective function can be defined that is enforced. In that case flux states are
        suppressed that are optimal with respect to the specified inner objective function and additionally fulfil
        the specified constraints. If the goal is to find minimal sets of knockouts that are lethal for an organism,
        one can use a suppress module with the constraint parameter: constraints='growth >= 0.01'. The algorithm
        will then find thes smallest sets of knockouts that render flux states with growth >= 0.01 infeasible. If we
        take the example of production strain design, one could use the suppress module to enforce a certain product 
        yield (prod/subst > min_yield) growth-maximal flux states. This can be done by defining an inner objective 
        function for optmizing growth and selecting a minimum production threshold to be attained at maximum growth: 
        inner_objective='1.0 growth', constraints='EX_prod_e - min_yield*UP_subst_e  <= 0'. If used without any
        constraints, the suppress module ensures that the model is infeasible.
        
        mandatory arguments: model, module_type='suppress'
        optional arguments: constraints, inner_objective, inner_opt_sense, skip_checks, reac_ids
        (Detailed description of the arguments follow below)
        
    Protect:
        MCS-like protection of a subspace of all steady-state flux vectors. The 'constraints' parameter is used to
        descirbe flux states that should become/stay feasible when/by introducing interventions. This can be used
        to maintain or protect certain metabolic functions, like microbial growth, despite engineering a strain for 
        bioroduction. When provided with an inner objective, the protect module will ensure that flux vectors
        optimal with respect to the inner objective function will be able to fulfil the constraints set in the
        'constraints' parameter. This can be used to design potentially growth-coupled strains with a minimum set
        of metabolic interventions. As an example, if one uses the protect module to ensure that growth with 
        rates above 0.1/h remains feasible, one sets constraints='growth >= -0.1'. If the goal is to ensure that
        product synthesis is possible at rates of more than 5 mmol/gCDW/h at maximal growth, one sets:
        inner_objective='1.0 growth' and constraints='EX_prod_e >= 5'. If used without any constraints, the protect
        module just ensures that the model is feasible.
        
        mandatory arguments: model, module_type='protect'
        optional arguments: constraints,inner_objective, inner_opt_sense, skip_checks, reac_ids
        (Detailed description of the arguments follow below)

    Example:
        m = SDModule(model,'optknock',outer_objective='growth', inner_objective='EX_etoh_e', constraints='growth >= 0.2')

    Args:
        model (cobra.Model):
        
            A metabolic model that is an instance of the cobra.Model class. Instead of a model, a dummy
            object can be used as long as it has the field 'id'. If a dummy is used, skip_checks=True
            must be used and a list of reaction ids must be provided through reac_ids=*list_of_strings*.
            
        module_type (str):
        
            A string that specifies the module type. Allowed values are 'optknock', 'robustknock', 
            'optcouple', 'protect', 'suppress'. Depending on the specified module type, other parameters
            must be set accordingly (see description above).
            
        constraints (optional (str) or (list of str) or (list of [dict,str,float])): (Default: '')
        
            List of *linear* constraints to be used in the module, e.g., to be enforced, suppressed or taken 
            into account: signs + or -, scalar factors for reaction rates, inclusive (in)equalities and a
            float value on the right hand side. The parsing of the constraints input allows for some 
            flexibility. Correct (and identical) inputs are, for instance: 
            constraints='-EX_o2_e <= 5, ATPM = 20' or
            constraints=['-EX_o2_e <= 5', 'ATPM = 20'] or
            constraints=[[{'EX_o2_e':-1},'<=',5], [{'ATPM':1},'=',20]]
            
        inner_objective (optional (str) or (dict)):

            The *linear* inner objective function for any module type. This parameter is mandatory for 
            OptKnock, RobustKnock and OptCouple modules and optional for suppress and protect modules. If 
            used, an optimization is nested into the global problem. It can be used to account for the 
            biological objective of growth (inner_objective='BIOMASS_Ecoli_core_w_GAM'). Any linear 
            expression can be used as input, either as a single string or as a dict. Correct (and identical) 
            inputs are, for instance:
            inner_objective='BIOMASS_Ecoli_core_w_GAM - 0.05 EX_etoh_e'
            inner_objective={'BIOMASS_Ecoli_core_w_GAM': 1, 'EX_etoh_e': -0.05}
            
        inner_opt_sense (optional (str)): (Default: 'maximize')
        
            Sense of the inner optimization (maximization or minimization). Allowed values are 'minimize'
            and 'maximize'.

        outer_objective (optional (str) or (dict)):

            The *linear* outer objective function for any module type. This parameter is mandatory for 
            OptKnock and RobustKnock and cannot be used with OptCouple, suppress and protect modules. If 
            applied, this objective function is used as the global objective function. In case of OptCouple,
            the global objective is the maximization of the growth-coupling potential and the outer objective
            is not specified manually. Typical outer objectives are the optimization of product synthesis
            (outer_objective='BIOMASS_Ecoli_core_w_GAM'), but also combinations of product synthesis and 
            growth are possible. Any linear expression can be used as input, either as a single string or as 
            a dict. Correct (and identical) inputs are, for instance:
            outer_objective='BIOMASS_Ecoli_core_w_GAM + 0.1 EX_prod_e'
            outer_objective={'BIOMASS_Ecoli_core_w_GAM': 1, 'EX_prod_e': 0.1}
            
        outer_opt_sense (optional (str)): (Default: 'maximize')
        
            Sense of the outer optimization (maximization or minimization). Allowed values are 'minimize'
            and 'maximize'.
            
        prod_id (optional (str) or (dict)):
        
            The reaction id of the product of interest. This parameter is *only* used in OptCouple strain
            design and will have no effect as part of any other module. Permitted is any linear expression
            either in the form of a string or as a dict:
            prod_id='EX_etoh'
            prod_id={'EX_etoh': 1}
            
        min_gcp (optional (float)): (Default: 0.0)
        
            Minimial growth-coupling potential (GCP). I.e., the minimum difference between maximum growth
            with and without production. In practice there are two nested optimizations, one that optimizes
            inner_objective and one that optimizes inner_objective and additionally demands that the
            constraint: prod_id=0 holds. Therefore, min_gcp presents the minimum enforced growth-coupling
            potential, so, a minimum objective value. This parameter is supposed to be used with the 'any'
            approach and has (virtually) no effect when used with 'best' or 'populate', since GCP is
            maximized anyway.
            
        skip_checks (optional (bool)): (Default: False)

            Skip the module verification. If checks are not skipped, the constructor will verify, if the
            module is feasible with the original model, that is, if all entered parameters parse correctly
            and if the given sets of constraints can be applied on the model without rendering it infeasible.
            Finally, it throws an error if the trivial 0-vector is feasible in the subspaces defined by a
            suppress or protect module, since the user should ensure that the 0-vector is excluded from these
            subspaces (see online documentation for detail).

    Returns:
        (SDModule):
            A strain design module object that can be used with the function compute_strain_design.
            Multiple modules can be used to specify a strain design problem.
    """

    def __init__(self, model, module_type, *args, **kwargs):
        self[MODEL_ID] = model.id
        self[MODULE_TYPE] = module_type
        allowed_keys = {
            CONSTRAINTS, INNER_OBJECTIVE, INNER_OPT_SENSE, OUTER_OBJECTIVE, OUTER_OPT_SENSE, PROD_ID, 'skip_checks', MIN_GCP, 'reac_ids'
        }
        # set all keys passed in kwargs as properties of the SD_Module object
        for key, value in kwargs.items():
            if key in allowed_keys:
                self[key] = value
            else:
                raise Exception("Key " + key + " is not supported.")
        # set all undefined keys to None
        for key in allowed_keys:
            if key not in kwargs.keys():
                self[key] = None

        if not self['reac_ids'] and not model.reactions:
            raise Exception('Strain design module cannot be constructed without information about '+\
                            'available reactions reactions. Make sure to provide a valid model or '+\
                            'reaction list.')

        # check if there is sufficient information for each module type
        if self[MODULE_TYPE] not in [PROTECT, SUPPRESS, OPTKNOCK, ROBUSTKNOCK, OPTCOUPLE]:
            raise Exception('"' + MODULE_TYPE + '" must be "' + PROTECT + '", "' + SUPPRESS + '", "' + OPTKNOCK + '", "' + ROBUSTKNOCK +
                            '", "' + OPTCOUPLE + '".')
        if (self[MODULE_TYPE] in [OPTKNOCK, ROBUSTKNOCK]):
            if self[INNER_OPT_SENSE] is None:
                self[INNER_OPT_SENSE] = MAXIMIZE
            if self[OUTER_OPT_SENSE] is None:
                self[OUTER_OPT_SENSE] = MAXIMIZE
            elif self[INNER_OPT_SENSE] not in [MINIMIZE, MAXIMIZE] or self[OUTER_OPT_SENSE] not in [MINIMIZE, MAXIMIZE]:
                raise Exception('Inner and outer optimization sense must be "' + MINIMIZE + '" or "' + MAXIMIZE + '" (default).')
            if ((self[INNER_OBJECTIVE] == None) or (self[OUTER_OBJECTIVE] == None)):
                raise Exception('When module type is "' + OPTKNOCK + '" or "' + ROBUSTKNOCK +
                                '", an inner and outer objective function must be provided.')
        elif (self[MODULE_TYPE] == OPTCOUPLE):
            if self[INNER_OPT_SENSE] is None:
                self[INNER_OPT_SENSE] = MAXIMIZE
            if self[INNER_OPT_SENSE] not in [MINIMIZE, MAXIMIZE]:
                raise Exception('Inner optimization sense must be "' + MINIMIZE + '" or "' + MAXIMIZE + '" (default).')
            if self[MIN_GCP] is None:
                self[MIN_GCP] = 0.0
            if self[INNER_OBJECTIVE] == None:
                raise Exception('When module type is "' + OPTCOUPLE + '", an inner objective function must be provided.')
            if self[PROD_ID] == None:
                raise Exception('When module type is "' + OPTCOUPLE + '", the production reaction id must be provided.')

        if not self['reac_ids']:
            self['reac_ids'] = model.reactions.list_attr('id')

        # parse constraints and ensure they have the form:
        # [ [{'r1': -1, 'r3': 2}, '<=', 3],
        #   [{'r2': 1, 'r3': -1},  '=', 0]  ]
        if self[CONSTRAINTS] is not None:
            self[CONSTRAINTS] = parse_constraints(self[CONSTRAINTS], self['reac_ids'])
        else:
            self[CONSTRAINTS] = []

        # parse inner objective
        if self[INNER_OBJECTIVE] is not None:
            if type(self[INNER_OBJECTIVE]) is str:
                self[INNER_OBJECTIVE] = linexpr2dict(self[INNER_OBJECTIVE], self['reac_ids'])

        # parse outer objective
        if self[OUTER_OBJECTIVE] is not None:
            if type(self[OUTER_OBJECTIVE]) is str:
                self[OUTER_OBJECTIVE] = linexpr2dict(self[OUTER_OBJECTIVE], self['reac_ids'])

        # parse prod_id
        if self[PROD_ID] is not None:
            if type(self[PROD_ID]) is str:
                self[PROD_ID] = linexpr2dict(self[PROD_ID], self['reac_ids'])

        # verify self[CONSTRAINTS]
        if not self['skip_checks']:
            from straindesign import fba
            if fba(model, constraints=self[CONSTRAINTS]).status == INFEASIBLE:
                raise Exception("There is no feasible solution of the model under the given constraints.")

            if self[MODULE_TYPE] in [SUPPRESS, PROTECT] and self[INNER_OBJECTIVE] is not None:
                constr = [[{k: 1}, '=', 0] for k in model.reactions.list_attr('id')]
                if fba(model, constraints=self[CONSTRAINTS] + constr).status != INFEASIBLE:
                    raise Exception('When '+MODULE_TYPE+' is "'+SUPPRESS+\
                        '", the zero vector must not be a contained in the described flux space.')

            if (self[INNER_OBJECTIVE]
                    is not None) and (not all([True if r in self['reac_ids'] else False for r in self[INNER_OBJECTIVE].keys()])):
                raise Exception("Inner objective invalid.")

            if (self[OUTER_OBJECTIVE]
                    is not None) and (not all([True if r in self['reac_ids'] else False for r in self[OUTER_OBJECTIVE].keys()])):
                raise Exception("Outer objective invalid.")

            if (self[PROD_ID] is not None) and (not all([True if r in self['reac_ids'] else False for r in self[PROD_ID].keys()])):
                raise Exception("Production id (prod_id) invalid.")

            if (self[MIN_GCP] is not None) and type(self[MIN_GCP]) is not None:
                if type(self[MIN_GCP]) is float:
                    pass
                elif type(self[MIN_GCP]) is int:
                    self[MIN_GCP] = float(self[MIN_GCP])
                else:
                    raise Exception("Minimum growth coupling potential (" + MIN_GCP + ").")

    def copy(self):
        """Create a deep copy of a strain design module."""

        class DummyModel:
            id = self[MODEL_ID]

        return SDModule(DummyModel(),
                        self[MODULE_TYPE],
                        constraints=deepcopy(self[CONSTRAINTS]),
                        inner_objective=deepcopy(self[INNER_OBJECTIVE]),
                        inner_opt_sense=deepcopy(self[INNER_OPT_SENSE]),
                        outer_objective=deepcopy(self[OUTER_OBJECTIVE]),
                        outer_opt_sense=deepcopy(self[OUTER_OPT_SENSE]),
                        prod_id=deepcopy(self[PROD_ID]),
                        min_gcp=self[MIN_GCP],
                        skip_checks=True,
                        reac_ids=deepcopy(self['reac_ids']))
