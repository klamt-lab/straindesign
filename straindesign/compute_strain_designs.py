from straindesign import StrainDesigner
from straindesign.names import *
from cobra import Model
from typing import Dict, List, Tuple
import json

def gpr_to_reac_sd(model,strain_designs):
    pass
    print('lol')
    return True

def compute_strain_designs(model: Model, **kwargs):
    ## Two computation modes:
    # 1. Provide model, strain design module and optional computation parameters
    # 2. Provide a full strain design setup in dict form (either as a dict from 
    #    previous MCS computations or a JSON ".sd"-file)
    if SETUP in kwargs:
        if type(kwargs[SETUP]) is str:
            with open(kwargs[SETUP],'r') as fs:
                kwargs = json.load(fs)
        else:
            kwargs = kwargs[SETUP]
            
    if MODULES in kwargs:
        sd_modules = kwargs.pop(MODULES)
        
    if MAX_COST in kwargs:
        kwargs.update({MAX_COST : float(kwargs.pop(MAX_COST))})
        
    if MODEL_ID in kwargs:
        model_id = kwargs.pop(MODEL_ID)
        if model_id != model.id:
            print(  "Model IDs of provided model and setup not matching. Apparently, ",\
                    "the strain design setup was specified for a different model. "+\
                    "Errors might occur due to non-matching reaction or gene-identifiers.")
    
    if 'gene_kos' in kwargs:
        model_id = kwargs.pop('gene_kos')
        if not GKOCOST in kwargs and not GKICOST in kwargs:
            kwargs[GKOCOST] = None
    if 'advanced' in kwargs:
        model_id = kwargs.pop('advanced')
    if 'use_scenario' in kwargs:
        model_id = kwargs.pop('use_scenario')
            
    # solution approach
    if SOLUTION_APPROACH in kwargs:
        solution_approach = kwargs.pop(SOLUTION_APPROACH)
    else:
        solution_approach = ANY

    kwargs_computation = {}
    if MAX_SOLUTIONS in kwargs:
        kwargs_computation.update({MAX_SOLUTIONS : float(kwargs.pop(MAX_SOLUTIONS))})
    if TIME_LIMIT in kwargs:
        kwargs_computation.update({TIME_LIMIT : float(kwargs.pop(TIME_LIMIT))})
    kwargs_computation.update({'show_no_ki' : True})
        
    # construct Strain Desing MILP
    strain_design_MILP = StrainDesigner( model, sd_modules, **kwargs)

    # solve MILP
    if solution_approach == ANY:
        sd_solutions = strain_design_MILP.compute(**kwargs_computation)
    elif solution_approach == BEST:
        sd_solutions = strain_design_MILP.compute_optimal(**kwargs_computation)
    elif solution_approach == POPULATE:
        sd_solutions = strain_design_MILP.enumerate(**kwargs_computation)

    return sd_solutions
