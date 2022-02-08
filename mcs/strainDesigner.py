from httpx import options
import numpy as np
from scipy import sparse
import cobra
import re
import time
from typing import Dict, List, Tuple
import mcs

class StrainDesigner(mcs.StrainDesignMILPBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

    def add_exclusion_constraints(self,z):
        for i in range(z.shape[0]):
            A_ineq = z[i].copy()
            A_ineq.resize((1,self.milp.A_ineq.shape[1]))
            b_ineq = np.sum(z[i])-1
            self.A_ineq = sparse.vstack((self.A_ineq,A_ineq))
            self.b_ineq += b_ineq
            self.milp.add_ineq_constraint(A_ineq,[b_ineq])

    def addExclusionConstraintsIneq(self,z):
        for j in range(z.shape[0]):
            A_ineq = [1.0 if z[j,i] else -1.0 for i in self.idx_z]
            A_ineq.resize((1,self.milp.A_ineq.shape[1]))
            b_ineq = np.sum(z[j])-1
            self.A_ineq = sparse.vstack((self.A_ineq,A_ineq))
            self.b_ineq += b_ineq
            self.milp.add_ineq_constraint(A_ineq,[b_ineq])

    def mcs2dict(self,sol) -> Dict:
        output = {}
        for i in self.idx_z:
            if not sol[0,i] == 0 and not np.isnan(sol[0,i]):
                if self.z_inverted[i]:
                    output[self.model.reactions[i].name] =  sol[0,i]
                else:
                    output[self.model.reactions[i].name] = -sol[0,i]
        return output

    def solveZ(self) -> Tuple[List,int]:
        x, _ , status = self.milp.solve()
        z = sparse.csr_matrix([x[i] for i in self.idx_z])
        return z, status

    def populateZ(self,n) -> Tuple[List,int]:
        x, _ , status = self.milp.populate(n)
        z = sparse.csr_matrix([[x[j][i] for i in self.idx_z] for j in range(len(x))])
        z.resize((len(x),self.num_z))
        return z, status

    def clearObjective(self):
        self.milp.clear_objective()
        self.c = [0]*len(self.c)

    def resetObjective(self):
        for i in range(self.c_bu):
            self.c[i] = self.c_bu[i]
        self.milp.set_objective_idx([[i,self.c[i]] for i in range(len(self.c))])

    def resetTargetableZ(self):
        self.ub = [1 - float(i) for i in self.z_non_targetable]
        self.milp.set_ub([[i,1] for i in self.idx_z if not self.z_non_targetable[i]])

    def setTargetableZ(self,sol):
        self.ub = [1.0 if sol[0,i] else 0.0 for i in self.idx_z]
        self.milp.set_ub([[i,0] for i in self.idx_z if not sol[0,i]])

    def verify_mcs(self,sols) -> List:
        valid = [False]*sols.shape[0]
        for i,sol in zip(range(sols.shape[0]),sols):
            inactive_vars = [var for z_i,var,sense in \
                            zip(self.cont_MILP.z_map_vars.row,self.cont_MILP.z_map_vars.col,self.cont_MILP.z_map_vars.data)\
                            if np.logical_xor(sol[0,z_i],sense==-1)]
            active_vars = [i for i in range(self.cont_MILP.z_map_vars.shape[1]) if i not in inactive_vars]
            inactive_ineqs = [ineq for z_i,ineq,sense in \
                            zip(self.cont_MILP.z_map_constr_ineq.row,self.cont_MILP.z_map_constr_ineq.col,self.cont_MILP.z_map_constr_ineq.data)\
                            if np.logical_xor(sol[0,z_i],sense==-1) ]
            active_ineqs = [i for i in range(self.cont_MILP.z_map_constr_ineq.shape[1]) if i not in inactive_ineqs]
            inactive_eqs = [eq for z_i,eq,sense in \
                            zip(self.cont_MILP.z_map_constr_eq.row,self.cont_MILP.z_map_constr_eq.col,self.cont_MILP.z_map_constr_eq.data)\
                            if np.logical_xor(sol[0,z_i],sense==-1) ]
            active_eqs = [i for i in range(self.cont_MILP.z_map_constr_eq.shape[1]) if i not in inactive_eqs]

            lp = mcs.MILP_LP(A_ineq  = self.cont_MILP.A_ineq[active_ineqs,:][:,active_vars],
                             b_ineq  = [self.cont_MILP.b_ineq[i] for i in active_ineqs],
                             A_eq    = self.cont_MILP.A_eq[active_eqs,:][:,active_vars],
                             b_eq    = [self.cont_MILP.b_eq[i] for i in active_eqs],
                             lb      = [self.cont_MILP.lb[i] for i in active_vars],
                             ub      = [self.cont_MILP.ub[i] for i in active_vars],
                             options = self.options,
                             solver  = self.solver)
            valid[i] = not np.isnan(lp.slim_solve())
        return valid

    # Find iteratively smallest MCS
    def compute_smallest_mcs(self, **kwargs):
        keys = {'max_solutions','time_limit','output_format'}
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
        endtime = time.time() + self.time_limit
        status = 0
        mcs_sols = sparse.csr_matrix((0,self.num_z))
        print('Finding smallest MCS ...')
        while mcs_sols.shape[0] < self.max_solutions and \
          status is 0 and \
          endtime-time.time() > 0:
            self.milp.set_time_limit(endtime-time.time())
            z, status = self.solveZ()
            output = self.mcs2dict(z)
            if status in [0,3] and all(self.verify_mcs(z)):
                print('MCS with cost '+str((z*self.cost)[0])+': '+str(output))
                self.add_exclusion_constraints(z)
                mcs_sols = sparse.vstack((mcs_sols,z))
            elif status in [0,3]:
                print('Invalid (minimal) solution found: '+ str(output))
                self.add_exclusion_constraints(z)
            if status is not 0:
                break
        if status == 2 and mcs_sols.shape[0] > 0: # all solutions found
            status = 0
        if status == 1 and mcs_sols.shape[0] > 0: # some solutions found, timelimit reached
            status = 3
        if endtime-time.time() > 0 and mcs_sols.shape[0] > 0:
            print('Finished. '+ str(mcs_sols.shape[0]) +' solutions found.')
        elif endtime-time.time() > 0:
            print('Finished. No solutions exist.')
        else:
            print('Time limit reached.')
        # Translate solutions into dict if not stated otherwise
        if self.output_format is None or self.output_format=='dict':
            mcs_dict = []
            for mcs in mcs_sols:
                mcs_dict += [self.mcs2dict(mcs)]
            return mcs_dict, status
        else:
            return mcs_sols, status

    # Find iteratively MCS of arbitrary size
    # output format: list of 'dict' (default) or 'sparse'
    def compute_mcs(self, **kwargs):
        keys = {'max_solutions','time_limit','output_format'}
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
        endtime = time.time() + self.time_limit
        status = 0
        mcs_sols = sparse.csr_matrix((0,self.num_z))
        print('Finding MCS of arbitrary size ...')
        while mcs_sols.shape[0] < self.max_solutions and \
          status is 0 and \
          endtime-time.time() > 0:
            print('Searching in full search space.')
            self.milp.set_time_limit(endtime-time.time())
            self.resetTargetableZ()
            self.clearObjective()
            z, status = self.solveZ()
            if np.isnan(z[0,0]):
                break
            if not all(self.verify_mcs(z)):
                self.milp.set_time_limit(endtime-time.time())
                self.resetObjective()
                self.setTargetableZ(z);
                z1, status1 = self.solveZ()
                if status1 == 0 and not self.verify_mcs(z1):
                    self.add_exclusion_constraints(z1)
                    output = self.mcs2dict(z1)
                    print('Invalid minimal solution found: '+ str(output))
                    continue
                if not status1 == 0 and not self.verify_mcs(z1):
                    self.addExclusionConstraintsIneq(z);
                    output = self.mcs2dict(z)
                    print('Invalid minimal solution found: '+ str(output))
                    continue
                else:
                    output = self.mcs2dict(z)
                    print('Warning: Solver first found the infeasible solution: '+ str(output))
                    output = self.mcs2dict(z1)
                    print('but a subset of this solution seems to be a valid MCS: '+ str(output))
            # Verify solution and explore subspace to get MCS
            print('Searching in subspace with '+str(sum(z.toarray()[0]))+' possible targets.')
            while mcs_sols.shape[0] < self.max_solutions and \
                    status is 0 and \
                    endtime-time.time() > 0:    
                self.milp.set_time_limit(endtime-time.time())
                self.resetObjective()
                self.setTargetableZ(z)
                z1, status1 = self.solveZ()
                output = self.mcs2dict(z1)
                if status1 in [0,3] and all(self.verify_mcs(z1)):
                    print('MCS with cost '+str((z1*self.cost)[0])+': '+str(output))
                    self.add_exclusion_constraints(z1)
                    mcs_sols = sparse.vstack((mcs_sols,z1))
                elif status1 in [0,3]:
                    print('Invalid minimal solution found: '+ str(output))
                    self.add_exclusion_constraints(z)
                else: # return to outside loop
                    break
        if status == 2 and mcs_sols.shape[0] > 0: # all solutions found
            status = 0
        if status == 1 and mcs_sols.shape[0] > 0: # some solutions found, timelimit reached
            status = 3
        if endtime-time.time() > 0 and mcs_sols.shape[0] > 0:
            print('Finished. '+ str(mcs_sols.shape[0]) +' solutions found.')
        elif endtime-time.time() > 0:
            print('Finished. No solutions exist.')
        else:
            print('Time limit reached.')
        # Translate solutions into dict if not stated otherwise
        if self.output_format is None or self.output_format=='dict':
            mcs_dict = []
            for mcs in mcs_sols:
                mcs_dict += [self.mcs2dict(mcs)]
            return mcs_dict, status
        else:
            return mcs_sols, status

    # Find iteratively MCS of arbitrary size
    # output format: list of 'dict' (default) or 'sparse'
    def enumerate_mcs(self, **kwargs):
        keys = {'max_solutions','time_limit','output_format'}
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
        endtime = time.time() + self.time_limit
        status = 0
        mcs_sols = sparse.csr_matrix((0,self.num_z))
        print('Enumerating MCS ...')
        while mcs_sols.shape[0] < self.max_solutions and \
          status is 0 and \
          endtime-time.time() > 0:
            self.milp.set_time_limit(endtime-time.time())
            z, status = self.populateZ(self.max_solutions - mcs_sols.shape[0])
            if status in [0,3]:
                for i in range(z.shape[0]):
                    output = [self.mcs2dict(z[i])]
                    if all(self.verify_mcs(z[i])):
                        print('MCS with cost '+str((z*self.cost)[0])+': '+str(output))
                        self.add_exclusion_constraints(z[i])
                        mcs_sols = sparse.vstack((mcs_sols,z[i]))
                    else:
                        print('Invalid (minimal) solution found: '+ str(output))
                        self.add_exclusion_constraints(z)
            if status is not 0:
                break
        if status == 2 and mcs_sols.shape[0] > 0: # all solutions found or solution limit reached
            status = 0
        if status == 1 and mcs_sols.shape[0] > 0: # some solutions found, timelimit reached
            status = 3
        if endtime-time.time() > 0 and mcs_sols.shape[0] > 0:
            print('Finished. '+ str(mcs_sols.shape[0]) +' solutions found.')
        elif endtime-time.time() > 0:
            print('Finished. No solutions exist.')
        else:
            print('Time limit reached.')
        # Translate solutions into dict if not stated otherwise
        if self.output_format is None or self.output_format=='dict':
            mcs_dict = []
            for mcs in mcs_sols:
                mcs_dict += [self.mcs2dict(mcs)]
            return mcs_dict, status
        else:
            return mcs_sols, status
