import numpy as np
from scipy import sparse
from typing import Dict, List, Tuple
from scipy import sparse
from cobra import Model
from mcs import StrainDesignMILP, StrainDesignMILPBuilder, MILP_LP, SD_Module
from warnings import warn
import efmtool_link.efmtool4cobra as efmtool4cobra

class StrainDesigner(StrainDesignMILP):
    def __init__(self, model: Model, sd_modules: List[SD_Module], *args, **kwargs):
        allowed_keys = {'ko_cost', 'ki_cost', 'solver', 'max_cost', 'M','threads', 'mem', 'compress', 'options'}
        # set all keys passed in kwargs
        for key, value in dict(kwargs).items():
            if key in allowed_keys:
                locals()[key] = value
            else:
                raise Exception("Key " + key + " is not supported.")
        # set all remaining keys to None
        for key in allowed_keys:
            if key not in dict(kwargs).keys():
                locals()[key] = None
        # Preprocess Model
        subT = efmtool4cobra.compress_model_sympy(model)
        # Build MILP
        super().__init__(model,sd_modules, *args, **kwargs)
