from httpx import options
import numpy as np
from scipy import sparse
import cobra
import re
from typing import Dict, List, Tuple
import mcs

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
            self.mem = 2048
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

    def add_exclusion_constraint(self,z):
        A_ineq = z.copy()
        A_ineq.resize((1,self.milp.A_ineq.shape[1]))
        b_ineq = np.sum(z)-1
        self.A_ineq = sparse.vstack((self.A_ineq,A_ineq))
        self.b_ineq += b_ineq
        self.milp.add_ineq_constraint(A_ineq,[b_ineq])
        # TODO also add rownames

    def verify_mcs(self,sol):
        pass


    def compute_smallest_mcs(self, *kwargs) -> List:
        keys = {'max_solutions','time_limit'}
        # set keys passed in kwargs
        for key,value in dict(kwargs).items():
            if key in keys:
                setattr(self,key,value)
        # set all remaining keys to None
        for key in keys:
            if key not in dict(kwargs).keys():
                setattr(self,key,None)
        if self.max_solutions is None:
            self.max_solutions = np.inf
        if self.time_limit is None:
            self.time_limit = np.inf
        status = 0
        mcs_sols = sparse.csr_matrix((0,self.num_z))
        while status is 0:
            x, min_cx, status = self.milp.solve()
            if status is not 0:
                break
            z = sparse.csr_matrix([x[i] for i in self.idx_z])
            print({self.model.reactions[i].name : z[0,i] if self.z_inverted[i] else -z[0,i] \
                   for i in range(len(self.model.reactions))})
            self.add_exclusion_constraint(z)
            mcs_sols = sparse.vstack((mcs_sols,z))
        # maybe translate solutions into dict for returning
        return mcs_sols