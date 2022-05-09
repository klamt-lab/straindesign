from straindesign.strainDesignSolution import SDSolution
import numpy as np
from scipy import sparse
import time
from cobra import Model
from typing import Dict, List, Tuple
from straindesign import StrainDesignMILPBuilder, MILP_LP, SDModule
from straindesign.names import *
from warnings import warn

class StrainDesignMILP(StrainDesignMILPBuilder):
    def __init__(self, model: Model, sd_modules: List[SDModule], **kwargs):
        keys = {'options'}
        # remove keys that are irrelevant for MILP construction
        kwargs1 = kwargs.copy()
        for k in keys:
            if k in kwargs1:
                del kwargs1[k]
        super().__init__(model, sd_modules, **kwargs1)  
        # set keys passed in kwargs
        for key,value in dict(kwargs).items():
            if key in keys:
                setattr(self,key,value)
        # set all remaining keys to None
        for key in keys:
            if key not in dict(kwargs).keys():
                setattr(self,key,None)
        self.sd_modules = sd_modules
        self.milp = MILP_LP(c           =self.c,
                            A_ineq      =self.A_ineq,
                            b_ineq      =self.b_ineq,
                            A_eq        =self.A_eq,
                            b_eq        =self.b_eq,
                            lb          =self.lb,
                            ub          =self.ub,
                            vtype       =self.vtype,
                            indic_constr=self.indic_constr,
                            M           =self.M,
                            solver      =self.solver)

    def add_exclusion_constraints(self,z):
        for i in range(z.shape[0]):
            A_ineq = z[i].copy()
            A_ineq.resize((1,self.milp.A_ineq.shape[1]))
            b_ineq = np.sum(z[i])-1
            self.A_ineq = sparse.vstack((self.A_ineq,A_ineq))
            self.b_ineq += [b_ineq]
            self.milp.add_ineq_constraints(A_ineq,[b_ineq])

    def add_exclusion_constraints_ineq(self,z):
        for j in range(z.shape[0]):
            A_ineq = [1.0 if z[j,i] else -1.0 for i in self.idx_z]
            A_ineq.resize((1,self.milp.A_ineq.shape[1]))
            b_ineq = np.sum(z[j])-1
            self.A_ineq = sparse.vstack((self.A_ineq,A_ineq))
            self.b_ineq += [b_ineq]
            self.milp.add_ineq_constraints(A_ineq,[b_ineq])

    def sd2dict(self,sol,*args) -> Dict:
        output = {}
        reacID = self.model.reactions.list_attr("id")
        for i in self.idx_z:
            if sol[0,i] != 0 and not np.isnan(sol[0,i]):
                if self.z_inverted[i]:
                    output[reacID[i]] =  sol[0,i]
                else:
                    output[reacID[i]] = -sol[0,i]
            elif args and args[0] and (sol[0,i] == 0) and self.z_inverted[i]:
                output[reacID[i]] = 0.0
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

    def fixObjective(self,c,cx):
        self.milp.set_ineq_constraint(2,c,cx)
        self.A_ineq = self.A_ineq.tolil()
        self.A_ineq[2] = sparse.lil_matrix(c)
        self.A_ineq = self.A_ineq.tocsr()
        self.b_ineq[2] = cx

    def resetObjective(self):
        for i,v in enumerate(self.c_bu):
            self.c[i] = v
        self.milp.set_objective_idx([[i,v] for i,v in enumerate(self.c_bu)])

    def setMinIntvCostObjective(self):
        self.clearObjective()
        for i in self.idx_z:
            if i not in self.z_non_targetable:
                self.c[i] = self.cost[i]
        self.milp.set_objective_idx([[i,self.c[i]] for i in self.idx_z if i not in self.z_non_targetable])

    def resetTargetableZ(self):
        self.ub = [1 - float(i) for i in self.z_non_targetable]
        self.milp.set_ub([[i,1] for i in self.idx_z if not self.z_non_targetable[i]])

    def setTargetableZ(self,sol):
        self.ub = [1.0 if sol[0,i] else 0.0 for i in self.idx_z]
        self.milp.set_ub([[i,0] for i in self.idx_z if not sol[0,i]])

    def verify_sd(self,sols) -> List:
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

            lp = MILP_LP(   A_ineq  = self.cont_MILP.A_ineq[active_ineqs,:][:,active_vars],
                            b_ineq  = [self.cont_MILP.b_ineq[i] for i in active_ineqs],
                            A_eq    = self.cont_MILP.A_eq[active_eqs,:][:,active_vars],
                            b_eq    = [self.cont_MILP.b_eq[i] for i in active_eqs],
                            lb      = [self.cont_MILP.lb[i] for i in active_vars],
                            ub      = [self.cont_MILP.ub[i] for i in active_vars],
                            solver  = self.solver)
            valid[i] = not np.isnan(lp.slim_solve())
        return valid

    # Find iteratively smallest solutions
    def compute_optimal(self, **kwargs):
        keys = {'max_solutions','time_limit','show_no_ki'}
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
        # first check if strain doesn't already fulfill the strain design setup
        if self.is_mcs_computation and self.verify_sd(sparse.csr_matrix((1,self.num_z)))[0]:
            print('The strain already meets the requirements defined in the strain design setup. ' \
                  'No interventions are needed.')
            return self.build_sd_solution([{}], OPTIMAL, BEST)
        # otherwise continue
        endtime = time.time() + self.time_limit
        status = OPTIMAL
        sols = sparse.csr_matrix((0,self.num_z))
        print('Finding optimal strain designs ...')
        while sols.shape[0] < self.max_solutions and \
          status == OPTIMAL and \
          endtime-time.time() > 0:
            self.milp.set_time_limit(endtime-time.time())
            self.resetTargetableZ()
            self.resetObjective()
            self.fixObjective(self.c_bu,np.inf)
            x, min_cx , status = self.milp.solve()     
            z = sparse.csr_matrix([x[i] for i in self.idx_z])
            if np.isnan(z[0,0]):
                break
            output = self.sd2dict(z)
            if self.is_mcs_computation:
                if status in [OPTIMAL,TIME_LIMIT_W_SOL] and all(self.verify_sd(z)):
                    print('Strain design with cost '+str(round((z*self.cost)[0],6))+': '+str(output))
                    self.add_exclusion_constraints(z)
                    sols = sparse.vstack((sols,z))
                elif status in [OPTIMAL,TIME_LIMIT_W_SOL]:
                    print('Invalid (minimal) solution found: '+ str(output))
                    self.add_exclusion_constraints(z)
                if status != OPTIMAL:
                    break
            else:
                # Verify solution and explore subspace to get minimal intervention sets
                print('Found solution with objective value '+str(min_cx))
                print('Minimizing number of interventions in subspace with '+str(sum(z.toarray()[0]))+' possible targets.')
                self.fixObjective(self.c_bu,min_cx)
                self.setMinIntvCostObjective()
                self.setTargetableZ(z)
                while sols.shape[0] < self.max_solutions and \
                        status == OPTIMAL and \
                        endtime-time.time() > 0:    
                    self.milp.set_time_limit(endtime-time.time())
                    z1, status1 = self.solveZ()
                    output = self.sd2dict(z1)
                    if status1 in [OPTIMAL,TIME_LIMIT_W_SOL] and all(self.verify_sd(z1)):
                        print('Strain design with cost '+str(round((z1*self.cost)[0],6))+': '+str(output))
                        self.add_exclusion_constraints(z1)
                        sols = sparse.vstack((sols,z1))
                    elif status1 in [OPTIMAL,TIME_LIMIT_W_SOL]:
                        print('Invalid minimal solution found: '+ str(output))
                        self.add_exclusion_constraints(z)
                    else: # return to outside loop
                        break
        if status == INFEASIBLE and sols.shape[0] > 0: # all solutions found
            status = OPTIMAL
        if status == TIME_LIMIT and sols.shape[0] > 0: # some solutions found, timelimit reached
            status = TIME_LIMIT_W_SOL
        if endtime-time.time() > 0 and sols.shape[0] > 0:
            print('Finished solving strain design MILP. ')
            if 'strainDesignMILP' in self.__module__:
                print(str(sols.shape[0]) +' solutions found.')
        elif endtime-time.time() > 0:
            print('Finished solving strain design MILP.')
            if 'strainDesignMILP' in self.__module__:
                print(' No solutions exist.')
        else:
            print('Time limit reached.')
        # Translate solutions into dict
        m=sd_dict = []
        for sol in sols:
            sd_dict += [self.sd2dict(sol,self.show_no_ki)]
        return self.build_sd_solution(sd_dict, status, BEST)

    # Find iteratively intervention sets of arbitrary size or quality
    # output format: list of 'dict' (default) or 'sparse'
    def compute(self, **kwargs):
        keys = {'max_solutions','time_limit','show_no_ki'}
        # set keys passed in kwargs
        for key,value in kwargs.items():
            if key in keys:
                setattr(self,key,value)
        # set all remaining keys to None
        for key in keys:
            if key not in kwargs.keys():
                setattr(self,key,None)
        if self.max_solutions is None:
            self.max_solutions = np.inf
        if self.time_limit is None:
            self.time_limit = np.inf
        # first check if strain doesn't already fulfill the strain design setup
        if self.verify_sd(sparse.csr_matrix((1,self.num_z)))[0]:
            print('The strain already meets the requirements defined in the strain design setup. ' \
                  'No interventions are needed.')
            return self.build_sd_solution([{}], OPTIMAL, ANY)
        # otherwise continue
        endtime = time.time() + self.time_limit
        status = OPTIMAL
        sols = sparse.csr_matrix((0,self.num_z))
        print('Finding (also non-optimal) strain designs ...')
        while sols.shape[0] < self.max_solutions and \
          status == OPTIMAL and \
          endtime-time.time() > 0:
            print('Searching in full search space.')
            self.milp.set_time_limit(endtime-time.time())
            self.resetTargetableZ()
            self.clearObjective()
            self.fixObjective(self.c_bu,np.inf) # keep objective open
            x, min_cx , status = self.milp.solve()
            z = sparse.csr_matrix([x[i] for i in self.idx_z])
            if np.isnan(z[0,0]):
                break
            if not all(self.verify_sd(z)):
                self.milp.set_time_limit(endtime-time.time())
                self.resetObjective()
                self.setTargetableZ(z)
                self.fixObjective(self.c_bu,np.sum([c*x for c,x in zip(self.c_bu,x)]))
                z1, status1 = self.solveZ()
                if status1 == OPTIMAL and not self.verify_sd(z1):
                    self.add_exclusion_constraints(z1)
                    output = self.sd2dict(z1)
                    print('Invalid minimal solution found: '+ str(output))
                    continue
                if status1 != OPTIMAL and not self.verify_sd(z1):
                    self.add_exclusion_constraints_ineq(z);
                    output = self.sd2dict(z)
                    print('Invalid minimal solution found: '+ str(output))
                    continue
                else:
                    output = self.sd2dict(z)
                    print('Warning: Solver first found the infeasible solution: '+ str(output))
                    output = self.sd2dict(z1)
                    print('But a subset of this solution seems to be valid: '+ str(output))
            # Verify solution and explore subspace to get strain designs
            cx = np.sum([c*x for c,x in zip(self.c_bu,x)])
            if not self.is_mcs_computation:
                print('Found preliminary solution.')
            print('Minimizing number of interventions in subspace with '+str(sum(z.toarray()[0]))+' possible targets.')
            self.setMinIntvCostObjective()
            self.setTargetableZ(z)
            self.fixObjective(self.c_bu,cx)
            while sols.shape[0] < self.max_solutions and \
                    status == OPTIMAL and \
                    endtime-time.time() > 0:    
                self.milp.set_time_limit(endtime-time.time())
                x1, min_cx , status1 = self.milp.solve()
                z1 = sparse.csr_matrix([x1[i] for i in self.idx_z])
                output = self.sd2dict(z1)
                if status1 in [OPTIMAL,TIME_LIMIT_W_SOL] and all(self.verify_sd(z1)):
                    print('Strain design with cost '+str(round((z1*self.cost)[0],6))+': '+str(output))
                    self.add_exclusion_constraints(z1)
                    sols = sparse.vstack((sols,z1))
                elif status1 in [OPTIMAL,TIME_LIMIT_W_SOL]:
                    print('Invalid minimal solution found: '+ str(output))
                    self.add_exclusion_constraints(z)
                else: # return to outside loop
                    break
        if status == INFEASIBLE and sols.shape[0] > 0: # all solutions found
            status = OPTIMAL
        if status == TIME_LIMIT and sols.shape[0] > 0: # some solutions found, timelimit reached
            status = TIME_LIMIT_W_SOL
        if endtime-time.time() > 0 and sols.shape[0] > 0:
            print('Finished solving strain design MILP. ')
            if 'strainDesignMILP' in self.__module__:
                print(str(sols.shape[0]) +' solutions found.')
        elif endtime-time.time() > 0:
            print('Finished solving strain design MILP.')
            if 'strainDesignMILP' in self.__module__:
                print(' No solutions exist.')
        else:
            print('Time limit reached.')
        # Translate solutions into dict if not stated otherwise
        sd_dict = []
        for sol in sols:
            sd_dict += [self.sd2dict(sol,self.show_no_ki)]
        return self.build_sd_solution(sd_dict, status, ANY)

    # Enumerate iteratively optimal strain designs using the populate function
    # output format: list of 'dict' (default) or 'sparse'
    def enumerate(self, **kwargs):
        keys = {'max_solutions','time_limit','show_no_ki'}
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
        # first check if strain doesn't already fulfill the strain design setup
        if self.is_mcs_computation and self.verify_sd(sparse.csr_matrix((1,self.num_z)))[0]:
            print('The strain already meets the requirements defined in the strain design setup. ' \
                  'No interventions are needed.')
            return self.build_sd_solution([{}], OPTIMAL, POPULATE)
        # otherwise continue
        if self.solver == 'scip':
            warn("SCIP does not natively support solution pool generation. "+ \
                "An high-level implementation of populate is used. " + \
                "Consider using compute_optimal instead of enumerate, as " + \
                "it returns the same results but faster.")
        if self.solver == 'glpk':
            warn("GLPK does not natively support solution pool generation. "+ \
            "An instable high-level implementation of populate is used. "
            "Consider using compute_optimal instead of enumerate, as " + \
            "it returns the same results but faster." )
        endtime = time.time() + self.time_limit
        status = OPTIMAL
        sols = sparse.csr_matrix((0,self.num_z))
        print('Enumerating strain designs ...')
        while sols.shape[0] < self.max_solutions and \
          status == OPTIMAL and \
          endtime-time.time() > 0:
            self.milp.set_time_limit(endtime-time.time())
            if not self.is_mcs_computation:
                self.resetTargetableZ()
                self.resetObjective()
                self.fixObjective(self.c_bu,np.inf)
                x, min_cx , status = self.milp.solve()     
                z = sparse.csr_matrix([x[i] for i in self.idx_z])
                if np.isnan(z[0,0]):
                    break
                print('Enumerating all solutions with the objective value: '+str(min_cx))
                self.fixObjective(self.c_bu,min_cx)
                self.setMinIntvCostObjective()
            z, status = self.populateZ(self.max_solutions - sols.shape[0])
            if status in [OPTIMAL,TIME_LIMIT_W_SOL]:
                for i in range(z.shape[0]):
                    output = [self.sd2dict(z[i])]
                    if all(self.verify_sd(z[i])):
                        print('Strain designs with cost '+str(round((z[i]*self.cost)[0],6))+': '+str(output))
                        self.add_exclusion_constraints(z[i])
                        sols = sparse.vstack((sols,z[i]))
                    else:
                        print('Invalid (minimal) solution found: '+ str(output))
                        self.add_exclusion_constraints(z[i])
            if (status != OPTIMAL): # or (z[i]*self.cost == self.max_cost):
                break
        if status == INFEASIBLE and sols.shape[0] > 0: # all solutions found or solution limit reached
            status = OPTIMAL
        if status == TIME_LIMIT and sols.shape[0] > 0: # some solutions found, timelimit reached
            status = TIME_LIMIT_W_SOL
        if endtime-time.time() > 0 and sols.shape[0] > 0:
            print('Finished solving strain design MILP. ')
            if 'strainDesignMILP' in self.__module__:
                print(str(sols.shape[0]) +' solutions found.')
        elif endtime-time.time() > 0:
            print('Finished solving strain design MILP.')
            if 'strainDesignMILP' in self.__module__:
                print(' No solutions exist.')
        else:
            print('Time limit reached.')
        # Translate solutions into dict if not stated otherwise
        sd_dict = []
        for sol in sols:
            sd_dict += [self.sd2dict(sol,self.show_no_ki)]
        sd_solution = self.build_sd_solution(sd_dict, status, POPULATE)
        return sd_solution

    def build_sd_solution(self, sd_dict, status, solution_approach):
        sd_setup = {}
        sd_setup[MODEL_ID] = self.model.id
        sd_setup[MAX_SOLUTIONS] = self.max_solutions
        sd_setup[MAX_COST] = self.max_cost
        sd_setup[TIME_LIMIT] = self.time_limit
        sd_setup[SOLVER] = self.solver
        sd_setup[SOLUTION_APPROACH] = solution_approach
        sd_setup[KOCOST] = {k:float(v) for k,v in \
            zip(self.model.reactions.list_attr('id'),self.ko_cost) if not np.isnan(v)}
        sd_setup[KICOST] = {k:float(v) for k,v in \
            zip(self.model.reactions.list_attr('id'),self.ki_cost) if not np.isnan(v)}
        sd_setup[MODULES] = self.sd_modules
        return SDSolution(self.model,sd_dict,status,sd_setup)