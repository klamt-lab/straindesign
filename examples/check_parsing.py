from mcs import *
from cobra.io import read_sbml_model
from os import path

# load network
network = read_sbml_model(path.dirname(path.abspath(__file__))+"/gpr_model.sbml")

## module_type
# allowed values: 'mcs_lin', 'mcs_bilvl', 'mcs_yield', 'optknock', 'robustknock', 'optcouple'
module_type = 'mcs_lin'
# module_type = 'mcs_bilvl'
# module_type = 'mcs_yield'
# module_type = 'optknock'
# module_type = 'robustknock'
# module_type = 'optcouple'

## module_sense:
# allowed values: 'undesired', 'desired'
module_sense = 'undesired'
# module_sense = 'desired'

## constraints:
# linear constraints of the form T*r <= t
# allowed values: str, lists of str, str with newlines (\n), (lists) of Tuples of dict (expressions) and rhs (i.e. C = ({'R1':-1, 'R2':2},'=' ,3) )
# for module_type = 'mcs_yield', one of these terms may contain a division '/'
# constraints = 'r1 + 2 r3 <= 3'
# constraints = 'r1 + 2 r3 <= 3 \n r2 - r3 = 0'
# constraints = ['r1 + 2 r3 <= 3',' r2 - r3 = 0']
constraints = [{'r1':-1, 'r3':2} , '<=' ,3]
# constraints = [({'r1':-1, 'r3':2},'<=' ,3), ({'r2':1, 'r3':-1} ,'=' ,0)]

fba(network,constraints=constraints)

## inner_objective:
# allowed values: str, dict
inner_objective = '3 r_bm'
# inner_objective = {'r_bm':3}

## inner_opt_sense
# allowed values: 'minimize', 'maximize'
inner_opt_sense = 'maximize'

## outer_objective:
# allowed values: str, dict
outer_objective = '3 r_bm'
outer_objective = {'r_bm':3}

## outer_opt_sense
# allowed values: 'minimize', 'maximize'
outer_opt_sense = 'maximize'

## prod_id:
# allowed values: str, dict
prod_id = 'rp_ex'
# prod_id = {'r_p': 2}

## min_gcp:
# allowed values: float
min_gcp = 0.2

## skip_checks:
# allowed values: True, False (default)
skip_checks = False

mod = SD_Module(network,
                module_type,
                module_sense   = module_sense,
                constraints    = constraints,
                inner_objective= inner_objective,
                inner_opt_sense= inner_opt_sense,
                outer_objective= outer_objective,
                outer_opt_sense= outer_opt_sense,
                prod_id        = prod_id,
                min_gcp        = min_gcp,
                skip_checks    = skip_checks )
