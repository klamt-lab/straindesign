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
"""Static strings used in the StrainDesign package

    Model and module

        MODEL_ID = 'model_id'

        PROTECT = 'mcs_lin'

        SUPPRESS = 'mcs_bilvl'

        SUPPRESS = 'suppress'

        PROTECT = 'protect'

        OPTKNOCK = 'optknock'

        ROBUSTKNOCK = 'robustknock'

        OPTCOUPLE = 'optcouple'
        
        MODULE_TYPE = 'module_type'
        
        CONSTRAINTS = 'constraints'
        
        INNER_OBJECTIVE = 'inner_objective'
        
        INNER_OPT_SENSE = 'inner_opt_sense'
        
        OUTER_OBJECTIVE = 'outer_objective'
        
        OUTER_OPT_SENSE = 'outer_opt_sense'
        
        PROD_ID = 'prod_id'
        
        MIN_GCP = 'min_gcp'
        
    Solvers and status codes
    
        SOLVER = 'solver'
        
        CPLEX = 'cplex'
        
        GUROBI = 'gurobi'
        
        SCIP = 'scip'
        
        GLPK = 'glpk'

        OPTIMAL = 'optimal' # from optlang interface
        
        INFEASIBLE ='infeasible' # from optlang interface
        
        TIME_LIMIT = 'time_limit' # from optlang interface
        
        UNBOUNDED = 'unbounded' # from optlang interface
        
        TIME_LIMIT_W_SOL = 'time_limit_w_sols'
        
        ERROR = 'error'
        
    Strain design setup

        KOCOST = 'ko_cost'
        
        KICOST = 'ki_cost'
        
        GKOCOST = 'gko_cost'
        
        GKICOST = 'gki_cost'
        
        REGCOST = 'reg_cost'

        MODULES = 'sd_modules'
    
        SETUP = 'sd_setup'
        
        MAX_SOLUTIONS = 'max_solutions'
        
        MAX_COST = 'max_cost'
        
        T_LIMIT = 'time_limit'
        
        SOLUTION_APPROACH = 'solution_approach'
        
        ANY = 'any'
        
        BEST = 'best'
        
        POPULATE = 'populate'
        
        SEED = 'seed'
        
    Analysis
    
        MAXIMIZE = 'maximize'
        
        MINIMIZE = 'minimize'
"""

# Model and module
MODEL_ID = 'model_id'
PROTECT = 'mcs_lin'
SUPPRESS = 'mcs_bilvl'
SUPPRESS = 'suppress'
PROTECT = 'protect'
OPTKNOCK = 'optknock'
ROBUSTKNOCK = 'robustknock'
OPTCOUPLE = 'optcouple'
MODULE_TYPE = 'module_type'
CONSTRAINTS = 'constraints'
INNER_OBJECTIVE = 'inner_objective'
INNER_OPT_SENSE = 'inner_opt_sense'
OUTER_OBJECTIVE = 'outer_objective'
OUTER_OPT_SENSE = 'outer_opt_sense'
PROD_ID = 'prod_id'
MIN_GCP = 'min_gcp'

# Solvers and status codes
SOLVER = 'solver'
CPLEX = 'cplex'
GUROBI = 'gurobi'
SCIP = 'scip'
GLPK = 'glpk'
from optlang.interface import OPTIMAL,    \
                              INFEASIBLE, \
                              TIME_LIMIT, \
                              UNBOUNDED

TIME_LIMIT_W_SOL = 'time_limit_w_sols'
ERROR = 'error'

# Strain design setup
KOCOST = 'ko_cost'
KICOST = 'ki_cost'
GKOCOST = 'gko_cost'
GKICOST = 'gki_cost'
REGCOST = 'reg_cost'
MODULES = 'sd_modules'
SETUP = 'sd_setup'
MAX_SOLUTIONS = 'max_solutions'
MAX_COST = 'max_cost'
T_LIMIT = 'time_limit'
SOLUTION_APPROACH = 'solution_approach'
ANY = 'any'
BEST = 'best'
POPULATE = 'populate'
SEED = 'seed'

# Analysis
MAXIMIZE = 'maximize'
MINIMIZE = 'minimize'
