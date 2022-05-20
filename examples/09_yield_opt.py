from straindesign import yopt, fba
from straindesign.names import *
from cobra.io import read_sbml_model

se = read_sbml_model(
    'C:/Users/Philipp/Documents/Python/straindesign/examples/SmallExample.sbml')
# se = read_sbml_model('/scratch/Python/straindesign/examples/SmallExample.sbml')
results = yopt(se,
               obj_num='R3',
               obj_den='R1',
               constraints='R2=0',
               obj_sense='maximize')
print(results)

results_fba = fba(se, obj='R2', constraints='R9=0', obj_sense='maximize')
print(results_fba)
