from straindesign import compute_strain_designs
from straindesign.names import SETUP
from cobra.io import read_sbml_model
import json

ecc = read_sbml_model('/scratch/Python/straindesign/examples/e_coli_core.sbml')
solution = compute_strain_designs(ecc,sd_setup='/home/schneiderp/CNApy-projects/ECC_optcouple.sdc')
print('stop')