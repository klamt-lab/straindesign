from optlang.interface import OPTIMAL, INFEASIBLE, TIME_LIMIT, UNBOUNDED

MODEL_ID = 'model_id'
MCS = 'mcs'
MCS_LIN = 'mcs_lin'
MCS_BILVL = 'mcs_bilvl'
MCS_YIELD = 'mcs_yield'
OPTKNOCK = 'optknock'
ROBUSTKNOCK = 'robustknock'
OPTCOUPLE = 'optcouple'

TIME_LIMIT_W_SOL   = TIME_LIMIT+'_w_sols'
ERROR = 'error'

KOCOST = 'ko_cost'
KICOST = 'ki_cost'
GKOCOST = 'gko_cost'
GKICOST = 'gki_cost'

MODULES = 'sd_modules'
MODULE_TYPE = 'module_type'
MODULE_SENSE = 'module_sense'
CONSTRAINTS = 'constraints'
INNER_OBJECTIVE = 'inner_objective'
INNER_OPT_SENSE = 'inner_opt_sense'
OUTER_OBJECTIVE = 'outer_objective'
OUTER_OPT_SENSE = 'outer_opt_sense'
PROD_ID = 'prod_id'
MIN_GCP = 'min_gcp'
MAXIMIZE = 'maximize'
MINIMIZE = 'minimize'
DESIRED = 'desired'
UNDESIRED = 'undesired'
SOLVER = 'solver'
CPLEX = 'cplex'
GUROBI = 'gurobi'
SCIP = 'scip'
GLPK = 'glpk'

SETUP = 'sd_setup'
MAX_SOLUTIONS = 'max_solutions'
MAX_COST = 'max_cost'
TIME_LIMIT = 'time_limit'
SOLUTION_APPROACH = 'solution_approach'
ANY = 'any'
SMALLEST = 'smallest'
CARDINALITY = 'cardinality'