import cobra
import straindesign as sd
from straindesign.names import *
import logging

logging.basicConfig(level="INFO")

ecc = cobra.io.load_model('e_coli_core')

modules = [
    sd.SDModule(ecc,
                SUPPRESS,
                inner_objective='BIOMASS_Ecoli_core_w_GAM',
                constraints=["1.0 EX_etoh_e = 0.0 "])
]
modules += [
    sd.SDModule(ecc,
                OPTKNOCK,
                outer_objective='EX_etoh_e',
                inner_objective='BIOMASS_Ecoli_core_w_GAM',
                constraints=["1.0 BIOMASS_Ecoli_core_w_GAM >= 0.1 "])
]

sd_setup = {
    MODULES: modules,
    MAX_COST: 7,
    MAX_SOLUTIONS: 5,
    SOLUTION_APPROACH: BEST,
    GKOCOST: None
}

solution = sd.compute_strain_designs(ecc, sd_setup=sd_setup)
pass
