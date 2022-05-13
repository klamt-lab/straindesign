from cobra.core import Solution
from cobra.util import create_stoichiometric_matrix
from cobra import Configuration 
from scipy import sparse
from straindesign import MILP_LP, parse_constraints, parse_linexpr, lineqlist2mat, linexpr2dict, \
                         linexprdict2mat, SDPool, IndicatorConstraints, avail_solvers
from re import search
from straindesign.names import *
from typing import Dict, Tuple
from pandas import DataFrame
from numpy import floor, sign, mod, nan, isnan, unique, inf, isinf, zeros, full, linspace, prod
from os import cpu_count
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull   
import mpl_toolkits.mplot3d as a3

from straindesign.parse_constr import linexpr2mat, linexprdict2str

# FBA, FVA and yield optimization for cobra using a unified 
# the user may provide the optional arguments
#   constraints:    Additional constraints in text form (list of lists)
#   A_ineq, b_ineq: Additional constraints in matrix form
#   obj:            Alternative objective in text form
#   c:              Alternative objective in vector form
def select_solver(solver=None,model=None):
    # first try to use selected solver
    if solver:
        if solver in avail_solvers:
            return solver
        else:
            print('Selected solver '+solver+' not available. Using '+avail_solvers[0]+" instead.")
    try:
        # if no solver was defined, use solver specified in model
        if hasattr(model,'solver') and hasattr(model.solver,'interface') :
            solver = search('('+'|'.join(avail_solvers)+')',model.solver.interface.__name__)
            if solver is not None:
                return solver[0]
            else:
                print('Solver specified in model ('+model.solver.interface.__name__+') unavailable') 
        # if no solver specified in model, use solver from cobra configuration
        cobra_conf = Configuration()
        if hasattr(cobra_conf,'solver'):
            solver = search('('+'|'.join(avail_solvers)+')',cobra_conf.solver.__name__)
            if solver is not None:
                return solver[0]
            else:
                print('Solver specified in cobra config ('+cobra_conf.solver.__name__+') unavailable') 
    except:
        pass
    return avail_solvers[0] # if no solver is specified in cobra, fall back to list of available solvers
            
    

def idx2c(i,prev):
    col = int(floor(i/2))
    sig = sign(mod(i,2)-0.5)
    C = [[col,sig],[prev,0.0]]
    C_idx = [C[i][0] for i in range(len(C))]
    C_idx = unique([C_idx.index(C_idx[i]) for i in range(len(C_idx))])
    C = [C[i] for i in C_idx]
    return C

def fva_worker_init(A_ineq,b_ineq,A_eq,b_eq,lb,ub,solver):
    global lp_glob
    # redirect output to empty stream. Perhaps avoids some multithreading issues
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        lp_glob = MILP_LP(A_ineq=A_ineq, b_ineq=b_ineq, A_eq=A_eq, b_eq=b_eq,
                                        lb=lb, ub=ub, solver=solver)
        if lp_glob.solver == 'cplex':
            lp_glob.backend.parameters.threads.set(1)
            #lp_glob.backend.parameters.lpmethod.set(1)
        lp_glob.prev = 0

def fva_worker_compute(i) -> Tuple[int,float]:
    global lp_glob
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        C = idx2c(i,lp_glob.prev)
        if lp_glob.solver in ['cplex','gurobi']:
            lp_glob.backend.set_objective_idx(C)
            min_cx = lp_glob.backend.slim_solve()
        else:
            lp_glob.set_objective_idx(C)
            min_cx = lp_glob.slim_solve()
        lp_glob.prev = C[0][0]
        return i, min_cx

# GLPK needs a workaround, because problems cannot be solved in a different thread
# which apparently happens with the multiprocess

def fva_worker_init_glpk(A_ineq,b_ineq,A_eq,b_eq,lb,ub):
    global lp_glob
    lp_glob = {}
    lp_glob['A_ineq'] = A_ineq
    lp_glob['b_ineq'] = b_ineq
    lp_glob['A_eq'] = A_eq
    lp_glob['b_eq'] = b_eq
    lp_glob['lb'] = lb
    lp_glob['ub'] = ub

def fva_worker_compute_glpk(i) -> Tuple[int,float]:
    global lp_glob
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        lp_i = MILP_LP(A_ineq=lp_glob['A_ineq'], b_ineq=lp_glob['b_ineq'], 
                    A_eq=lp_glob['A_eq'], b_eq=lp_glob['b_eq'], lb=lp_glob['lb'], 
                    ub=lp_glob['ub'], solver=GLPK)
        col = int(floor(i/2))
        sig = sign(mod(i,2)-0.5)
        lp_i.set_objective_idx([[col,sig]])
        min_cx = lp_i.slim_solve()
    return i, min_cx

def fva(model,**kwargs):
    reaction_ids = model.reactions.list_attr("id")
    numr = len(model.reactions)
        
    if CONSTRAINTS in kwargs and kwargs[CONSTRAINTS]: 
        kwargs[CONSTRAINTS] = parse_constraints(kwargs[CONSTRAINTS],reaction_ids)
        A_ineq, b_ineq, A_eq, b_eq = lineqlist2mat(kwargs[CONSTRAINTS], reaction_ids) 

    if SOLVER not in kwargs:
        kwargs[SOLVER] = None
    solver = select_solver(kwargs[SOLVER],model)
        
    
    # prepare vectors and matrices
    A_eq_base = sparse.csr_matrix(create_stoichiometric_matrix(model))
    b_eq_base = [0]*len(model.metabolites)
    if 'A_eq' in locals():
        A_eq  = sparse.vstack((A_eq_base, A_eq))
        b_eq  = b_eq_base+b_eq
    else:
        A_eq = A_eq_base
        b_eq = b_eq_base
    if 'A_ineq' not in locals():
        A_ineq = sparse.csr_matrix((0,numr))
        b_ineq = []
    lb = [v.lower_bound for v in model.reactions]
    ub = [v.upper_bound for v in model.reactions]

    # build LP
    lp = MILP_LP(   A_ineq=A_ineq,
                    b_ineq=b_ineq,
                    A_eq=A_eq,
                    b_eq=b_eq,
                    lb=lb,
                    ub=ub,
                    solver=solver)
    _, _, status = lp.solve()
    if status not in [OPTIMAL,UNBOUNDED]: # if problem not feasible or unbounded
        raise Exception('FVA problem not feasible.')

    processes = cpu_count()-1
    if not processes:
        print("The number of cores could not be detected - assuming one.")
        processes = 1
    num_reactions = len(reaction_ids)
    processes = min(processes, num_reactions)

    x = [nan]*2*numr

    # Dummy to check if optimization runs
    # worker_init(A_ineq,b_ineq,A_eq,b_eq,lb,ub,solver)
    # worker_compute(1)
    if processes > 1 and numr > 300 and solver != GLPK:
        # with Pool(processes,initializer=worker_init,initargs=(A_ineq,b_ineq,A_eq,b_eq,lb,ub,solver)) as pool:
        with SDPool(processes,initializer=fva_worker_init,initargs=(A_ineq,b_ineq,A_eq,b_eq,lb,ub,solver)) as pool:
            chunk_size = len(reaction_ids) // processes
            # x = pool.imap_unordered(worker_compute, range(2*numr), chunksize=chunk_size)
            for i, value in pool.imap_unordered( fva_worker_compute, range(2*numr), chunksize=chunk_size):
                x[i] = value
    # GLPK works better when reinitializing the LP in every iteration. Unfortunately, this is slow
    # but for now by far the most stable solution.
    elif processes > 1 and numr > 500 and solver == GLPK:
        with SDPool(processes,initializer=fva_worker_init_glpk,initargs=(A_ineq,b_ineq,A_eq,b_eq,lb,ub)) as pool:
            chunk_size = len(reaction_ids) // processes
            # # x = pool.imap_unordered(worker_compute, range(2*numr), chunksize=chunk_size)
            for i, value in pool.imap_unordered( fva_worker_compute_glpk, range(2*numr), chunksize=chunk_size):
                x[i] = value
    else:
        fva_worker_init(A_ineq,b_ineq,A_eq,b_eq,lb,ub,solver)
        for i in range(2*numr):
            _, x[i] = fva_worker_compute(i)
    
    x = [v if abs(v)>= 1e-11 else 0.0 for v in x] # cut off for very small absolute values
    fva_result = DataFrame(
        {
            "minimum": [ x[i] for i in range(1,2*numr,2)],
            "maximum": [-x[i] for i in range(0,2*numr,2)],
        },
        index=reaction_ids,
    )

    return fva_result


def fba(model,**kwargs):
    reaction_ids = model.reactions.list_attr("id")

    if CONSTRAINTS in kwargs: 
        kwargs[CONSTRAINTS] = parse_constraints(kwargs[CONSTRAINTS],reaction_ids)
        A_ineq, b_ineq, A_eq, b_eq = lineqlist2mat(kwargs[CONSTRAINTS], reaction_ids)
    else:
        kwargs[CONSTRAINTS] = []

    if 'obj' in kwargs and kwargs['obj'] is not None:
        if type(kwargs['obj']) is str:
            kwargs['obj'] = linexpr2dict(kwargs['obj'],reaction_ids)
        c = linexprdict2mat(kwargs['obj'],reaction_ids).toarray()[0].tolist()
    else:
        c  = [i.objective_coefficient for i in model.reactions]
        
    if ('obj_sense' not in kwargs and model.objective_direction == 'max') or \
       ('obj_sense' in kwargs and kwargs['obj_sense'] not in ['min','minimize']):
        obj_sense = 'maximize'
        c = [ -i for i in c]
    else:
        obj_sense = 'minimize'

    if 'pfba' in kwargs:
        pfba = kwargs['pfba']
    else:
        pfba = False

    if SOLVER not in kwargs:
        kwargs[SOLVER] = None
    solver = select_solver(kwargs[SOLVER],model)
    
    # prepare vectors and matrices
    A_eq_base = create_stoichiometric_matrix(model)
    A_eq_base = sparse.csr_matrix(A_eq_base)
    b_eq_base = [0]*len(model.metabolites)
    if 'A_eq' in locals():
        A_eq  = sparse.vstack((A_eq_base, A_eq))
        b_eq  = b_eq_base+b_eq
    else:
        A_eq = A_eq_base
        b_eq = b_eq_base
    if 'A_ineq' not in locals():
        A_ineq = sparse.csr_matrix((0,len(model.reactions)))
        b_ineq = []
    lb = [v.lower_bound for v in model.reactions]
    ub = [v.upper_bound for v in model.reactions]

    # build LP
    fba_prob = MILP_LP( c=c,
                        A_ineq=A_ineq,
                        b_ineq=b_ineq,
                        A_eq=A_eq,
                        b_eq=b_eq,
                        lb=lb,
                        ub=ub,
                        solver=solver)

    x, opt_cx, status = fba_prob.solve()
    if obj_sense == 'minimize':
        opt_cx = -opt_cx
    if status == UNBOUNDED:
        num_prob = MILP_LP(c=[-v for v in c],
                            A_ineq=A_ineq,
                            b_ineq=b_ineq,
                            A_eq=A_eq,
                            b_eq=b_eq,
                            lb=lb,
                            ub=ub,
                            solver=solver)
        min_cx = num_prob.slim_solve()
        if min_cx <= 0:
            num_prob.add_eq_constraints(c,[-1])
        else:
            num_prob.add_eq_constraints(c,min_cx)
        x, _, _ = num_prob.solve()
    elif status not in [OPTIMAL, UNBOUNDED]:
        status = INFEASIBLE
    if pfba and status == OPTIMAL: # for pfba, split all reversible reactions and minimize the total flux
        numr = len(c)
        if pfba == 2: # pfba mode 2 minimizes the number of active reactions
            # fix optimal flux and do fva to get essential reactions and speed up minimization
            kwargs_fva = kwargs.copy()
            kwargs_fva[CONSTRAINTS].append([{reaction_ids[i]:c[i] for i in range(numr) if c[i]!=0},'=',opt_cx])
            fva_sol = fva(model,**kwargs_fva)
   
            A_ineq_pfba2 = A_ineq.copy()
            A_ineq_pfba2.resize(A_ineq.shape[0],2*numr)
            A_eq_pfba2 = A_eq.copy()
            A_eq_pfba2.resize(A_eq.shape[0],2*numr)
            lb_pfba2 = lb+[0.0]*numr
            # only set non-essential reactions as knockable
            ub_pfba2 = ub+[0.0 if prod(sign(lim))>0 else 1.0 for _,lim in fva_sol.iterrows()]
            c_pfba2  = [0.0]*numr+[-1.0]*numr
            A_ic = sparse.csr_matrix(([1]*numr,([j for j in range(numr)],[j for j in range(numr)])),[numr,2*numr])
            ic = IndicatorConstraints([numr+j for j in range(numr)], A_ic, [0]*numr, 'E'*numr, [1.0]*numr)
            pfba2_prob = MILP_LP(   c=c_pfba2,
                                    A_ineq=A_ineq_pfba2,
                                    b_ineq=b_ineq,
                                    A_eq=A_eq_pfba2,
                                    b_eq=b_eq,
                                    lb=lb_pfba2,
                                    ub=ub_pfba2,
                                    indic_constr=ic,
                                    vtype='C'*numr+'B'*numr,
                                    solver=solver)
            pfba2_prob.add_eq_constraints(c+[0]*numr,[opt_cx])
            y,_,_ = pfba2_prob.solve()
            zero_flux = [i for i,j in enumerate(range(numr,2*numr)) if y[j]]
            lb = [l if i not in zero_flux else 0.0 for i,l in enumerate(lb)]
            ub = [u if i not in zero_flux else 0.0 for i,u in enumerate(ub)]
        A_ineq_pfba = sparse.hstack((A_ineq,-A_ineq))
        A_eq_pfba = sparse.hstack((A_eq,-A_eq))
        lb_pfba = [max((0,l)) for l in lb] + [max((0,-u)) for u in ub]
        ub_pfba = [max((0,u)) for u in ub] + [max((0,-l)) for l in lb]
        c_pfba  = c+[-v for v in c]
        pfba_prob = MILP_LP(c=[1.0]*2*numr,
                            A_ineq=A_ineq_pfba,
                            b_ineq=b_ineq,
                            A_eq=A_eq_pfba,
                            b_eq=b_eq,
                            lb=lb_pfba,
                            ub=ub_pfba,
                            solver=solver)
        pfba_prob.add_eq_constraints(c_pfba,[opt_cx])
        x,_,_ = pfba_prob.solve()
        x = [x[i]-x[j] for i,j in enumerate(range(numr,2*numr))]
        
    x = [v if abs(v)>= 1e-11 else 0.0 for v in x] # cut off for very small absolute values
    fluxes = {reaction_ids[i] : x[i] for i in range(len(x))}
    sol = Solution(objective_value=-opt_cx,status=status,fluxes=fluxes)
    return sol

def yopt(model,**kwargs):
    reaction_ids = model.reactions.list_attr("id")
    if 'obj_num' not in kwargs:
        raise Exception('For a yield optimization, the numerator expression must be provided under the keyword "obj_num".')
    else:
        if type(kwargs['obj_num']) is not dict:
            obj_num = linexpr2mat(kwargs['obj_num'],reaction_ids)
        else:
            obj_num = linexprdict2mat(kwargs['obj_num'],reaction_ids)
            
    if 'obj_den' not in kwargs:
        raise Exception('For a yield optimization, the denominator expression must be provided under the keyword "obj_den".')
    else:
        if type(kwargs['obj_den']) is not dict:
            obj_den = linexpr2mat(kwargs['obj_den'],reaction_ids)
        else:
            obj_den = linexprdict2mat(kwargs['obj_den'],reaction_ids)
            
    if 'obj_sense' not in kwargs or kwargs['obj_sense'] not in ['min','minimize']:
        obj_sense = 'maximize'
    else:
        obj_sense = 'minimize'
        obj_num = -obj_num

    if CONSTRAINTS in kwargs: 
        kwargs[CONSTRAINTS] = parse_constraints(kwargs[CONSTRAINTS],reaction_ids)
        A_ineq, b_ineq, A_eq, b_eq = lineqlist2mat(kwargs[CONSTRAINTS], reaction_ids)

    if SOLVER not in kwargs:
        kwargs[SOLVER] = None
    solver = select_solver(kwargs[SOLVER],model)
        
    # prepare vectors and matrices for base problem
    A_eq_base = create_stoichiometric_matrix(model)
    A_eq_base = sparse.csr_matrix(A_eq_base)
    b_eq_base = [0]*len(model.metabolites)
    if 'A_eq' in locals():
        A_eq  = sparse.vstack((A_eq_base, A_eq))
        b_eq  = b_eq_base+b_eq
    else:
        A_eq = A_eq_base
        b_eq = b_eq_base
    if 'A_ineq' not in locals():
        A_ineq = sparse.csr_matrix((0,len(model.reactions)))
        b_ineq = []
    # Integrate upper and lower bounds into A_ineq and b_ineq
    real_lb = [i for i,v in enumerate(model.reactions.list_attr('lower_bound')) if not isinf(v)]
    real_ub = [i for i,v in enumerate(model.reactions.list_attr('upper_bound')) if not isinf(v)]
    sparse_lb = sparse.coo_matrix(([-1]*len(real_lb),(range(len(real_lb)),real_lb)),(len(real_lb),A_ineq.shape[1]))
    sparse_ub = sparse.coo_matrix(([ 1]*len(real_ub),(range(len(real_ub)),real_ub)),(len(real_ub),A_ineq.shape[1]))
    A_ineq = sparse.vstack((A_ineq,sparse_lb,sparse_ub))
    b_ineq = b_ineq + [-model.reactions[i].lower_bound for i in real_lb] + \
                      [ model.reactions[i].upper_bound for i in real_ub]
    # Analyze maximum and minimum value of denominator function to decide whether to fix it to +1 or -1 or abort computation
    den_sign = []
    den_prob = MILP_LP( c=obj_den.todense().tolist()[0],
                        A_ineq=A_ineq,
                        b_ineq=b_ineq,
                        A_eq=A_eq,
                        b_eq=b_eq,
                        solver=solver)
    _, min_denx, status_i = den_prob.solve()
    # check model feasibility
    if status_i not in [OPTIMAL, UNBOUNDED]:
        fluxes = {reaction_ids[i] : nan for i in range(len(reaction_ids))}
        return Solution(objective_value=nan,status=INFEASIBLE,fluxes=fluxes)
    # is minimum of denominator term negative?
    if min_denx < 0:
        den_sign += [-1]
    # is maximum of denominator term positive?
    den_prob.set_objective((-obj_den).todense().tolist()[0])
    _, max_denx, _ = den_prob.solve()
    if max_denx < 0:
        den_sign += [1]
    # is denominator fixed to zero
    if not den_sign:
        raise Exception('Denominator term can only take the value 0. Yield computation impossible.')

    # Create linear fractional problem (LFP)
    # A variable is added here to scale the right hand side of the original problem
    A_ineq_lfp = sparse.hstack((A_ineq,sparse.csr_matrix([-b for b in b_ineq]).transpose()))
    b_ineq_lfp = [0 for _ in b_ineq]
    A_eq_lfp = sparse.vstack((  sparse.hstack((A_eq,sparse.csr_matrix([-b for b in b_eq]).transpose())),\
                            sparse.hstack((obj_den,0))))
    opt_cx = inf
    for d in den_sign:
        b_eq_lfp = [0 for _ in b_eq] + [d]
        # build LP
        yopt_prob = MILP_LP(c=(-d*obj_num).todense().tolist()[0]+[0],
                            A_ineq=A_ineq_lfp,
                            b_ineq=b_ineq_lfp,
                            A_eq=A_eq_lfp,
                            b_eq=b_eq_lfp,
                            solver=solver)
        x_i, opt_i, status_i = yopt_prob.solve()
        if opt_i < opt_cx:
            x = x_i
            opt_cx = opt_i
            status = status_i
    if status is OPTIMAL:
        factor = x[-1] # get factor from LFP
        if factor == 0:
            factor = 1
        fluxes = {r : x[i]/factor for i,r in enumerate(reaction_ids)}
        if obj_sense == 'maximize':
            opt_cx = -opt_cx  # correct sign (maximization)
        sol = Solution(objective_value=opt_cx,status=status,fluxes=fluxes)
        if x[-1] == 0:
            sol.scalable = True
            print('Solution flux vector may be scaled with an arbitrary factor.')
        else:
            sol.scalable = False
        return sol
    elif status is UNBOUNDED:
        opt_cx = nan
        # check if numerator can be nonzero when denominator is zero
        fct = 1-2*(obj_sense == 'maximize')
        den_prob.set_objective((fct*obj_num).todense().tolist()[0])
        den_prob.add_eq_constraints(obj_den,[0])
        x, max_num, status_i = den_prob.solve()
        if isinf(max_num) or status_i == INFEASIBLE: # if numerator is still unbounded, generate a fixed solution
            num_prob = MILP_LP(c=(-fct*obj_num).todense().tolist()[0],
                                A_ineq=A_ineq,
                                b_ineq=b_ineq,
                                A_eq=A_eq,
                                b_eq=b_eq,
                                solver=solver)
            min_num = num_prob.slim_solve()
            if min_num <= 0:
                num_prob.add_eq_constraints((-fct*obj_num).todense().tolist()[0],[1])
            else:
                num_prob.add_eq_constraints((-fct*obj_num).todense().tolist()[0],min_num)
            x, opt_i, status_i = num_prob.solve()
        if not isnan(max_num):
            if max_num == 0:
                num_prob = MILP_LP( c=(-fct*obj_num).todense().tolist()[0],
                                    A_ineq=A_ineq,
                                    b_ineq=b_ineq,
                                    A_eq=A_eq,
                                    b_eq=b_eq,
                                    solver=solver)
                num_prob.add_eq_constraints((obj_den).todense().tolist()[0],[0])
                x, opt_i, status_i = num_prob.solve()
                fluxes = {r : x[i] for i,r in enumerate(reaction_ids)}
                print('Yield is undefined because denominator can become zero. Solution '\
                    'flux vector maximizes the numerator.')
                sol = Solution(objective_value=opt_cx,status=status,fluxes=fluxes)
                sol.scalable = False
            else:
                fluxes = {r : x[i] for i,r in enumerate(reaction_ids)}
                print('Yield is undefined because denominator can become zero. Solution '\
                    'flux vector maximizes the numerator.')
                sol = Solution(objective_value=opt_cx,status=status,fluxes=fluxes)
                sol.scalable = False
            return sol
        else:
            if obj_sense == 'maximize':
                opt_cx = inf
            else:
                opt_cx = -inf
            fluxes = {r : x[i] for i,r in enumerate(reaction_ids)}
            print('Yield is infinite because the numerator is unbounded.')
            sol = Solution(objective_value=opt_cx,status=status,fluxes=fluxes)
            sol.scalable = True
            return sol
    else:
        status = INFEASIBLE
    
def phase_plane(model, axes, **kwargs):
    reaction_ids = model.reactions.list_attr("id")
    
    if CONSTRAINTS in kwargs: 
        kwargs[CONSTRAINTS] = parse_constraints(kwargs[CONSTRAINTS],reaction_ids)
    else:
        kwargs[CONSTRAINTS] = None

    if SOLVER not in kwargs:
        kwargs[SOLVER] = None
    solver = select_solver(kwargs[SOLVER],model)
        
    if 'show' in kwargs:
            show = kwargs['show']
    else:
        show = True    
    
    axes = list(axes) # cast to list of lists
    num_axes = len(axes)
    if num_axes not in [2,3]:
        raise Exception('Please define 2 or 3 axes as a list of tuples [ax1, ax2, (optional) ax3] with ax1 = (den,num).\n'+\
                        '"den" and "num" being linear expressions.')
            
    if 'points' in kwargs:
        points = kwargs['points']
    else:
        if num_axes == 2:
            points = 40
        else:
            points = 25
    
    ax_name = ["" for _ in range(num_axes)]
    ax_limits = [(nan,nan) for _ in range(num_axes)]
    for i,ax in enumerate(axes):
        ax = parse_linexpr(ax,reaction_ids)[0]
        ax_name[i] = linexprdict2str(ax)
        sol_min = fba(model,obj=ax,constraints=kwargs[CONSTRAINTS],solver=solver,obj_sense='minimize')
        sol_max = fba(model,obj=ax,constraints=kwargs[CONSTRAINTS],solver=solver,obj_sense='maximize')
        # abort if any of the fluxes are unbounded or undefined
        unbnd = [i+1 for i,v in enumerate([sol_min,sol_max]) if v.status == UNBOUNDED]
        if any(unbnd):
            raise Exception('One of the specified reactions is unbounded or undefined. Phase plane cannot be generated.')
        ax_limits[i] = [min((0,sol_min.objective_value)),max((0,sol_max.objective_value))]
        axes[i] = ax

    # compute points
    x_space = linspace(ax_limits[0][0], ax_limits[0][1], num=points)
    lb = full(points, nan)
    ub = full(points, nan)
    for i,x in enumerate(x_space):
        constr = [axes[0],'=',x]
        sol_vmin = fba(model,constraints=constr,obj=axes[1],obj_sense='minimize')
        lb[i] = sol_vmin.objective_value
        sol_vmax = fba(model,constraints=constr,obj=axes[1],obj_sense='maximize')
        ub[i] = sol_vmax.objective_value

    if num_axes == 2:
        x = [v for v in x_space] + [v for v in reversed(x_space)]
        y = [v for v in lb] + [v for v in reversed(ub)]
        if lb[0] != ub[0]:
            x.extend([x_space[0], x_space[0]])
            y.extend([lb[0],      ub[0]])
        plot1 = plt.fill(x, y)
        plot1[0].axes.set_xlabel(ax_name[0])
        plot1[0].axes.set_ylabel(ax_name[1])
        plot1[0].axes.set_xlim(ax_limits[0][0]*1.05,ax_limits[0][1]*1.05)
        plot1[0].axes.set_ylim(ax_limits[1][0]*1.05,ax_limits[1][1]*1.05)
        if show:
            plt.show()
        return plot1
    
    elif num_axes == 3:
        max_diff_y = max([abs(l-u) for l,u in zip(lb,ub)])
        datapoints = []
        for x,l,u in zip(x_space,lb,ub):
            if l-u != 0:
                y_space = linspace(l,u,int(-(-points // (max_diff_y/abs(l-u)))))
            else:
                y_space = [0.0]
            for y in y_space:
                constr = [[axes[0],'=',x], [axes[1],'=',y]]
                sol_vmin = fba(model,constraints=constr,obj=axes[2],obj_sense='minimize')
                datapoints += [[x,y,sol_vmin.objective_value]]
                sol_vmax = fba(model,constraints=constr,obj=axes[2],obj_sense='maximize')
                datapoints += [[x,y,sol_vmax.objective_value]]
        hull = ConvexHull(datapoints)
        x = [d[0] for d in datapoints]
        y = [d[1] for d in datapoints]
        z = [d[2] for d in datapoints]
        ax = a3.Axes3D(plt.figure())
        ax.dist=10
        ax.azim=30
        ax.elev=10
        ax.set_xlim(ax_limits[0])
        ax.set_ylim(ax_limits[1])
        ax.set_zlim(ax_limits[2])
        ax.set_xlabel(ax_name[0])
        ax.set_ylabel(ax_name[1])
        ax.set_zlabel(ax_name[2])
        plot1 = ax.plot_trisurf(x,y,z,triangles=hull.simplices,linewidth=0.2, antialiased=True, alpha=0.67)
        if show:
            plt.show()
        return plot1
    
def yield_space(model, axes, **kwargs):
    reaction_ids = model.reactions.list_attr("id")
    
    if CONSTRAINTS in kwargs: 
        kwargs[CONSTRAINTS] = parse_constraints(kwargs[CONSTRAINTS],reaction_ids)
    else:
        kwargs[CONSTRAINTS] = None

    if SOLVER not in kwargs:
        kwargs[SOLVER] = None
    solver = select_solver(kwargs[SOLVER],model)
        
    if 'show' in kwargs:
            show = kwargs['show']
    else:
        show = True    
    
    axes = [list(ax) for ax in axes] # cast to list of lists
    num_axes = len(axes)
    if num_axes not in [2,3]:
        raise Exception('Please define 2 or 3 axes as a list of tuples [ax1, ax2, (optional) ax3] with ax1 = (den,num).\n'+\
                        '"den" and "num" being linear expressions.')
            
    if 'points' in kwargs:
        points = kwargs['points']
    else:
        if num_axes == 2:
            points = 40
        else:
            points = 25
    
    ax_name = ["" for _ in range(num_axes)]
    ax_limits = [(nan,nan) for _ in range(num_axes)]
    for i,ax in enumerate(axes):
        ax[0] = parse_linexpr(ax[0],reaction_ids)[0]
        ax[1] = parse_linexpr(ax[1],reaction_ids)[0]
        ax_name[i] = '('+linexprdict2str(ax[0])+') / ('+linexprdict2str(ax[1])+')'
        sol_min = yopt(model,obj_num=ax[0],obj_den=ax[1],constraints=kwargs[CONSTRAINTS],solver=solver,obj_sense='minimize')
        sol_max = yopt(model,obj_num=ax[0],obj_den=ax[1],constraints=kwargs[CONSTRAINTS],solver=solver,obj_sense='maximize')
        # abort if any of the yields are unbounded or undefined
        unbnd = [i+1 for i,v in enumerate([sol_min,sol_max]) if v.status == UNBOUNDED]
        if any(unbnd):
            raise Exception('One of the specified yields is unbounded or undefined. Yield space cannot be generated.')
        ax_limits[i] = [min((0,sol_min.objective_value)),max((0,sol_max.objective_value))]

    # compute points
    x_space = linspace(ax_limits[0][0], ax_limits[0][1], num=points)
    lb = full(points, nan)
    ub = full(points, nan)
    for i,x in enumerate(x_space):
        constr = [{**axes[0][0], **{k:-v*x for k,v in axes[0][1].items()}},'=',0]
        sol_vmin = yopt(model,constraints=constr,obj_num=axes[1][0],obj_den=axes[1][1],obj_sense='minimize')
        lb[i] = sol_vmin.objective_value
        sol_vmax = yopt(model,constraints=constr,obj_num=axes[1][0],obj_den=axes[1][1],obj_sense='maximize')
        ub[i] = sol_vmax.objective_value

    if num_axes == 2:
        x = [v for v in x_space] + [v for v in reversed(x_space)]
        y = [v for v in lb] + [v for v in reversed(ub)]
        if lb[0] != ub[0]:
            x.extend([x_space[0], x_space[0]])
            y.extend([lb[0],      ub[0]])
        plot1 = plt.fill(x, y)
        plot1[0].axes.set_xlabel(ax_name[0])
        plot1[0].axes.set_ylabel(ax_name[1])
        plot1[0].axes.set_xlim(ax_limits[0][0]*1.05,ax_limits[0][1]*1.05)
        plot1[0].axes.set_ylim(ax_limits[1][0]*1.05,ax_limits[1][1]*1.05)
        if show:
            plt.show()
        return plot1
    
    elif num_axes == 3:
        max_diff_y = max([abs(l-u) for l,u in zip(lb,ub)])
        datapoints = []
        for x,l,u in zip(x_space,lb,ub):
            if l-u != 0:
                y_space = linspace(l,u,int(-(-points // (max_diff_y/abs(l-u)))))
            else:
                y_space = [0.0]
            for y in y_space:
                constr = [[{**axes[0][0], **{k:-v*x for k,v in axes[0][1].items()}},'=',0],\
                          [{**axes[1][0], **{k:-v*y for k,v in axes[1][1].items()}},'=',0]]
                sol_vmin = yopt(model,constraints=constr,obj_num=axes[2][0],obj_den=axes[2][1],obj_sense='minimize')
                datapoints += [[x,y,sol_vmin.objective_value]]
                sol_vmax = yopt(model,constraints=constr,obj_num=axes[2][0],obj_den=axes[2][1],obj_sense='maximize')
                datapoints += [[x,y,sol_vmax.objective_value]]
        hull = ConvexHull(datapoints)
        x = [d[0] for d in datapoints]
        y = [d[1] for d in datapoints]
        z = [d[2] for d in datapoints]
        ax = a3.Axes3D(plt.figure())
        ax.dist=10
        ax.azim=30
        ax.elev=10
        ax.set_xlim(ax_limits[0])
        ax.set_ylim(ax_limits[1])
        ax.set_zlim(ax_limits[2])
        ax.set_xlabel(ax_name[0])
        ax.set_ylabel(ax_name[1])
        ax.set_zlabel(ax_name[2])
        plot1 = ax.plot_trisurf(x,y,z,triangles=hull.simplices,linewidth=0.2, antialiased=True, alpha=0.67)
        if show:
            plt.show()
        return plot1