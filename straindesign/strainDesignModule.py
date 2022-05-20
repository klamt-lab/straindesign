"""
Strain design module (:class:`SDModule`)

Strain design modules are used to describe strain design problems, 
e.g. desired or undesired flux states for MCS strain design.
"""

from numpy import all
from typing import List, Dict, Tuple, Union, Set, FrozenSet
from straindesign.parse_constr import *
from straindesign.names import *


class SDModule(Dict):
    """
    Modules to describe desired or undesired flux states for MCS strain design.
    
    There are three kinds of flux states that can be described
        1. The wildtype model, constrainted with additional inequalities:
            e.g.: T v <= t.
        2. The wildtype model at a specific optimum, constrained with additional inequalities
            e.g.: objective*v = optimal, T v <= t
        3. A yield range:
            e.g.: numerator*v/denominator*v <= t (Definition of A and b is not required)

    Args:
        module_sense: 'desired' or 'undesired'
        module_type: 'mcs_lin', 'mcs_bilvl', 'optknock'
        constraints: Linear constraints: A v <= b, A v >= b, A v = b
                    (e.g. T v <= t with 'undesired' or D v <= d with 'desired')
        mcs_lin: <none>
        mcs_bilvl: inner_objective: Inner optimization expression
        mcs_yield: numerator: numerator of yield function,
                        denominator: denominator of yield function
        optknock: inner_objective: Inner optimization expression
                  outer_objective: Outer optimization expression
        robustknock:
        optcouple:

    Returns:
        modules = [         mcs_module.MCS_Module(network,"mcs_lin",module_sense="undesired",constraints="R4 >= 1")]
        modules = [modules, mcs_module.MCS_Module(network,"mcs_lin",module_sense="desired",constraints="R3 >= 1")]
        ...

    """

    def __init__(self, model, module_type, *args, **kwargs):
        self[MODEL_ID] = model.id
        self[MODULE_TYPE] = module_type
        allowed_keys = {
            CONSTRAINTS, INNER_OBJECTIVE, INNER_OPT_SENSE, OUTER_OBJECTIVE,
            OUTER_OPT_SENSE, PROD_ID, 'skip_checks', MIN_GCP
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

        if not model.reactions:
            raise Exception('Strain design module cannot be constructed for models without reactions. ' \
                             'Make sure to provide a valid module.')

        # check if there is sufficient information for each module type
        if self[MODULE_TYPE] not in [
                PROTECT, SUPPRESS, OPTKNOCK, ROBUSTKNOCK, OPTCOUPLE
        ]:
            raise Exception('"' + MODULE_TYPE + '" must be "' + PROTECT +
                            '", "' + SUPPRESS + '", "' + OPTKNOCK + '", "' +
                            ROBUSTKNOCK + '", "' + OPTCOUPLE + '".')
        if (self[MODULE_TYPE] in [OPTKNOCK, ROBUSTKNOCK]):
            if self[INNER_OPT_SENSE] is None:
                self[INNER_OPT_SENSE] = MAXIMIZE
            if self[OUTER_OPT_SENSE] is None:
                self[OUTER_OPT_SENSE] = MAXIMIZE
            elif self[INNER_OPT_SENSE] not in [
                    MINIMIZE, MAXIMIZE
            ] or self[OUTER_OPT_SENSE] not in [MINIMIZE, MAXIMIZE]:
                raise Exception('Inner and outer optimization sense must be "' +
                                MINIMIZE + '" or "' + MAXIMIZE + '" (default).')
            if ((self[INNER_OBJECTIVE] == None) or
                (self[OUTER_OBJECTIVE] == None)):
                raise Exception(
                    'When module type is "' + OPTKNOCK + '" or "' +
                    ROBUSTKNOCK +
                    '", an inner and outer objective function must be provided.'
                )
        elif (self[MODULE_TYPE] == OPTCOUPLE):
            if self[INNER_OPT_SENSE] is None:
                self[INNER_OPT_SENSE] = MAXIMIZE
            if self[INNER_OPT_SENSE] not in [MINIMIZE, MAXIMIZE]:
                raise Exception('Inner optimization sense must be "' +
                                MINIMIZE + '" or "' + MAXIMIZE + '" (default).')
            if self[INNER_OBJECTIVE] == None:
                raise Exception(
                    'When module type is "' + OPTCOUPLE +
                    '", an inner objective function must be provided.')
            if self[PROD_ID] == None:
                raise Exception(
                    'When module type is "' + OPTCOUPLE +
                    '", the production reaction id must be provided.')

        reac_id = model.reactions.list_attr('id')

        # parse constraints and ensure they have the form:
        # [ [{'r1': -1, 'r3': 2}, '<=', 3],
        #   [{'r2': 1, 'r3': -1},  '=', 0]  ]
        if self[CONSTRAINTS] is not None:
            self[CONSTRAINTS] = parse_constraints(self[CONSTRAINTS], reac_id)
        else:
            self[CONSTRAINTS] = []

        # parse inner objective
        if self[INNER_OBJECTIVE] is not None:
            if type(self[INNER_OBJECTIVE]) is str:
                self[INNER_OBJECTIVE] = linexpr2dict(self[INNER_OBJECTIVE],
                                                     reac_id)

        # parse outer objective
        if self[OUTER_OBJECTIVE] is not None:
            if type(self[OUTER_OBJECTIVE]) is str:
                self[OUTER_OBJECTIVE] = linexpr2dict(self[OUTER_OBJECTIVE],
                                                     reac_id)

        # parse prod_id
        if self[PROD_ID] is not None:
            if type(self[PROD_ID]) is str:
                self[PROD_ID] = linexpr2dict(self[PROD_ID], reac_id)

        # verify self[CONSTRAINTS]
        if self['skip_checks'] is None:
            from straindesign import fba
            if fba(model, constraints=self[CONSTRAINTS]).status == INFEASIBLE:
                raise Exception(
                    "There is no feasible solution of the model under the given constraints."
                )

            if self[MODULE_TYPE] == SUPPRESS and self[
                    INNER_OBJECTIVE] is not None:
                constr = [[{
                    k: 1
                }, '=', 0] for k in model.reactions.list_attr('id')]
                if fba(model, constraints=self[CONSTRAINTS] +
                       constr).status != INFEASIBLE:
                    raise Exception('When '+MODULE_TYPE+' is "'+SUPPRESS+\
                        '", the zero vector must not be a contained in the described flux space.')

            if (self[INNER_OBJECTIVE] is not None) and (not all([
                    True if r in reac_id else False
                    for r in self[INNER_OBJECTIVE].keys()
            ])):
                raise Exception("Inner objective invalid.")

            if (self[OUTER_OBJECTIVE] is not None) and (not all([
                    True if r in reac_id else False
                    for r in self[OUTER_OBJECTIVE].keys()
            ])):
                raise Exception("Outer objective invalid.")

            if (self[PROD_ID] is not None) and (not all(
                [True if r in reac_id else False
                 for r in self[PROD_ID].keys()])):
                raise Exception("Production id (prod_id) invalid.")

            if (self[MIN_GCP] is not None) and type(self[MIN_GCP]) is not None:
                if type(self[MIN_GCP]) is float:
                    pass
                elif type(self[MIN_GCP]) is int:
                    self[MIN_GCP] = float(self[MIN_GCP])
                else:
                    raise Exception("Minimum growth coupling potential (" +
                                    MIN_GCP + ").")
