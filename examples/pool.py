#!/usr/bin/env python3.7

# Copyright 2021, Gurobi Optimization, LLC

# We find alternative epsilon-optimal solutions to a given knapsack
# problem by using PoolSearchMode

from __future__ import print_function
import gurobipy as gp
from gurobipy import GRB
import sys

try:
    # Sample data
    Groundset = range(10)
    objCoef = [32, 32, 15, 15, 6, 6, 1, 1, 1, 1]
    knapsackCoef = [16, 16,  8,  8, 4, 4, 2, 2, 1, 1]
    Budget = 33

    # Create initial model
    model = gp.Model("poolsearch")

    # Create dicts for tupledict.prod() function
    objCoefDict = dict(zip(Groundset, objCoef))
    knapsackCoefDict = dict(zip(Groundset, knapsackCoef))

    # Initialize decision variables for ground set:
    # x[e] == 1 if element e is chosen
    Elem = model.addVars(Groundset, vtype=GRB.BINARY, name='El')

    # Set objective function
    model.ModelSense = GRB.MAXIMIZE
    model.setObjective(Elem.prod(objCoefDict))

    # Constraint: limit total number of elements to be picked to be at most
    # Budget
    model.addConstr(Elem.prod(knapsackCoefDict) <= Budget, name='Budget')

    # Limit how many solutions to collect
    model.setParam(GRB.Param.PoolSolutions, 1024)
    # Limit the search space by setting a gap for the worst possible solution
    # that will be accepted
    model.setParam(GRB.Param.PoolGap, 0.10)
    # do a systematic search for the k-best solutions
    model.setParam(GRB.Param.PoolSearchMode, 2)

    # save problem
    model.write('poolsearch.lp')

    # Optimize
    model.optimize()

    model.setParam(GRB.Param.OutputFlag, 0)

    # Status checking
    status = model.Status
    if status in (GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED):
        print('The model cannot be solved because it is infeasible or '
              'unbounded')
        sys.exit(1)

    if status != GRB.OPTIMAL:
        print('Optimization was stopped with status ' + str(status))
        sys.exit(1)

    # Print best selected set
    print('Selected elements in best solution:')
    print('\t', end='')
    for e in Groundset:
        if Elem[e].X > .9:
            print(' El%d' % e, end='')
    print('')

    # Print number of solutions stored
    nSolutions = model.SolCount
    print('Number of solutions found: ' + str(nSolutions))

    # Print objective values of solutions
    for e in range(nSolutions):
        model.setParam(GRB.Param.SolutionNumber, e)
        print('%g ' % model.PoolObjVal, end='')
        if e % 15 == 14:
            print('')
    print('')

    # print fourth best set if available
    if (nSolutions >= 4):
        model.setParam(GRB.Param.SolutionNumber, 3)

        print('Selected elements in fourth best solution:')
        print('\t', end='')
        for e in Groundset:
            if Elem[e].Xn > .9:
                print(' El%d' % e, end='')
        print('')

except gp.GurobiError as e:
    print('Gurobi error ' + str(e.errno) + ": " + str(e.message))

except AttributeError as e:
    print('Encountered an attribute error: ' + str(e))