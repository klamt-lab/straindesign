from mcs import StrainDesigner, SD_Module
from mcs.names import *
from cobra import Model, Metabolite, Reaction
from typing import Dict, List, Tuple

def gpr_to_reac_sd(model,strain_designs):
    pass
    print('lol')
    return True

def compute_strain_designs(model: Model, sd_modules: List[SD_Module], *args, **kwargs):
    ## output format
    # auto: gene-intervention-based or gene-intervention-based strain designs
    # reac: for gene-based computations, all interventions translated into reaction interventions
    # reac_kos: same as 'reac', but non-introduced additions are flagged with 0 ({reac_id: 0})
    # auto_kos: same as 'auto' but non-introduced additions are flagged with 0 ({reac_id: 0})
    kwargs_computation = {}
    if 'output_format' in kwargs and kwargs['output_format'] != 'auto':
        output_format = kwargs.pop('output_format')
        kwargs_computation.update({'show_no_ki' : True})
    else:
        output_format = 'auto'
        kwargs_computation.update({'show_no_ki' : False})
        
    if MAX_COST in kwargs:
        kwargs.update({MAX_COST : float(kwargs.pop(MAX_COST))})
    
    # solution approach
    if SOLUTION_APPROACH in kwargs:
        solution_approach = kwargs.pop(SOLUTION_APPROACH)
    else:
        solution_approach = ANY

    if MAX_SOLUTIONS in kwargs:
        kwargs_computation.update({MAX_SOLUTIONS : float(kwargs.pop(MAX_SOLUTIONS))})
    if TIME_LIMIT in kwargs:
        kwargs_computation.update({TIME_LIMIT : float(kwargs.pop(TIME_LIMIT))})
        
    # construct Strain Desing MILP
    mcsEnum = StrainDesigner( model, sd_modules, **kwargs)

    # solve MILP
    if solution_approach == ANY:
        sd, status = mcsEnum.compute(**kwargs_computation)
    elif solution_approach == SMALLEST:
        sd, status = mcsEnum.compute_optimal(**kwargs_computation)
    elif solution_approach == CARDINALITY:
        sd, status = mcsEnum.enumerate(**kwargs_computation)

    if status in [0,3]:
        if output_format == 'auto':
            return sd, status
        elif output_format == 'auto_kos':
            for s in sd:
                [s.update({k:-1.0}) for k,v in s.items() if v==0.0]
            return sd, status
        else:
            if output_format == 'reac':
                return [], status
    else:
        return [], status