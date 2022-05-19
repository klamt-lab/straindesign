from straindesign import compute_strain_designs
from straindesign.names import SETUP
from cobra.io import read_sbml_model
import json

ecc2 = read_sbml_model('/scratch/Python/straindesign/examples/ECC2.sbml')
solution = compute_strain_designs(ecc2,sd_setup='/home/schneiderp/CNApy-projects/ECC2_optknock.sdc')
print('stop')