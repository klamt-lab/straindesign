from httpx import options
import numpy as np
from scipy import sparse
import cobra
import re
from typing import Dict, List, Tuple
import mcs
import ray

class StrainDesigner(mcs.StrainDesignMILPBuilder):
    def __init__(self, *args, **kwargs):
        mcs.StrainDesignMILPBuilder.__init__(self, *args, **kwargs)
        keys = {'threads', 'mem', 'options'}        
        # set keys passed in kwargs
        for key,value in dict(kwargs).items():
            if key in keys:
                setattr(self,key,value)
        # set all remaining keys to None
        for key in keys:
            if key not in dict(kwargs).keys():
                setattr(self,key,None)
        if self.mem == None:
            self.mem = 2048;
        self.milp = mcs.MILP_LP(c           =self.c,
                                A_ineq      =self.A_ineq,
                                b_ineq      =self.b_ineq,
                                A_eq        =self.A_eq,
                                b_eq        =self.b_eq,
                                lb          =self.lb,
                                ub          =self.ub,
                                vtype       =self.vtype,
                                indic_constr=self.indic_constr,
                                options     =self.options,
                                solver      =self.solver)
    def compute_mcs(self,max_solutions=np.inf, max_cost=np.inf, *kwargs) -> List:
        keys = {'max_solutions', 'max_cost'}   
        # set keys passed in kwargs
        for key,value in dict(kwargs).items():
            if key in keys:
                setattr(self,key,value)
        # set all remaining keys to None
        for key in keys:
            if key not in dict(kwargs).keys():
                setattr(self,key,None)
        x, min_cx, status = self.milp.solve()
        print({self.model.reactions[i].name : x[i] for i in range(len(self.model.reactions))})
        return mcs