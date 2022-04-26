from straindesign import compute_strain_designs
from straindesign.names import SETUP
from cobra.io import read_sbml_model
import json

se = read_sbml_model('/scratch/Python/straindesign/examples/SmallExample.sbml')
solution = compute_strain_designs(se,sd_setup='/home/schneiderp/CNApy-projects/small_example_ki.sdc')
print('stop')