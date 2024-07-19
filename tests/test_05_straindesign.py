"""Test if all strain design functions run correctly."""
from .test_01_load_models_and_solvers import *
import straindesign as sd
from numpy import inf


@pytest.mark.timeout(15)
def test_mcs(curr_solver, model_small_example, comp_approach, bigM, compression):
    modules = [sd.SDModule(model_small_example, SUPPRESS, constraints=["R3 - 0.5 R1 <= 0.0", "R2 <= 0", "R1 >= 0.1"])]
    modules += [
        sd.SDModule(model_small_example, SUPPRESS, constraints=["1.0 R3 - 0.5 R1 - 0.5 R2 <= 0.0 ", "1.0 R2 >= 0.0 ", "1.0 R1 >= 0.1 "])
    ]
    modules += [sd.SDModule(model_small_example, PROTECT, constraints=["1.0 R3 >= 1.0 "])]
    kicost = {
        'R2': 1,
    }
    sd_setup = {
        MODULES: modules,
        MAX_COST: inf,
        MAX_SOLUTIONS: inf,
        SOLUTION_APPROACH: comp_approach,
        KICOST: kicost,
        SOLVER: curr_solver,
        'compress': compression,
        'M': bigM
    }
    solution = sd.compute_strain_designs(model_small_example, sd_setup=sd_setup)
    sols = solution.get_reaction_sd()
    assert ({'R1': -1.0, 'R2': 1.0} in sols)
    assert ({'R6': -1.0, 'R8': -1.0} in sols)
    assert ({'R4': -1.0} in sols)
    assert ({'R7': -1.0} in sols)
    assert ({'R10': -1.0} in sols)


@pytest.mark.timeout(15)
def test_mcs_opt(curr_solver, model_weak_coupling, comp_approach, bigM, compression):
    """Test MCS computation with nested optimization constraints."""
    modules = [sd.SDModule(model_weak_coupling, SUPPRESS, inner_objective="r_BM", constraints=["r_P - 0.4 r_S <= 0", "r_S >= 0.1"])]
    modules += [sd.SDModule(model_weak_coupling, PROTECT, constraints=["r_BM >= 0.2"])]
    kocost = {'r1': 1, 'r2': 1, 'r4': 1.1, 'r5': 0.75, 'r7': 0.8, 'r8': 1, 'r9': 1, 'r_S': 1.0, 'r_P': 1, 'r_BM': 1, 'r_Q': 1.5}
    kicost = {
        'r3': 0.6,
        'r6': 1.0,
    }
    regcost = {'r6 >= 4.5': 1.2}
    sd_setup = {
        MODULES: modules,
        MAX_COST: 4,
        MAX_SOLUTIONS: inf,
        SOLUTION_APPROACH: comp_approach,
        KOCOST: kocost,
        KICOST: kicost,
        REGCOST: regcost,
        SOLVER: curr_solver,
        'compress': compression,
        'M': bigM
    }
    solution = sd.compute_strain_designs(model_weak_coupling, sd_setup=sd_setup)
    assert (len(solution.reaction_sd) == 3)


@pytest.mark.timeout(15)
def test_mcs_gpr(model_gpr, comp_approach):
    """Test MCS computation with gpr rules."""
    modules = [sd.SDModule(model_gpr, SUPPRESS, constraints=["1.0 rd_ex >= 1.0 "])]
    modules += [sd.SDModule(model_gpr, PROTECT, constraints=[[{'r_bm': 1.0}, '>=', 1.0]])]
    kocost = {'rs_up': 1.0, 'rd_ex': 1.0, 'rp_ex': 1.1, 'r_bm': 0.75}
    gkocost = {
        'g1': 1.0,
        'g2': 1.0,
        'g4': 3.0,
        'g5': 2.0,
        'g6': 1.0,
        'g7': 1.0,
        'g8': 1.0,
        'g9': 1.0,
    }
    gkicost = {
        'g3': 1.0,
    }
    regcost = {'g4 <= 0.4': 1.2}
    sd_setup = {
        MODULES: modules,
        MAX_COST: 2,
        MAX_SOLUTIONS: inf,
        SOLUTION_APPROACH: comp_approach,
        KOCOST: kocost,
        GKOCOST: gkocost,
        GKICOST: gkicost,
        REGCOST: regcost,
        SOLVER: 'gurobi'
    }
    solution = sd.compute_strain_designs(model_gpr, sd_setup=sd_setup)
    assert (len(solution.gene_sd) == 4)


@pytest.mark.timeout(15)
def test_mcs_gpr2(model_gpr, comp_approach):
    """Test MCS computation gpr rule (compression)."""
    modules = [sd.SDModule(model_gpr, SUPPRESS, constraints=["1.0 rd_ex >= 1.0 "])]
    modules += [sd.SDModule(model_gpr, PROTECT, constraints=[[{'r_bm': 1.0}, '>=', 1.0]])]
    kocost = {'rs_up': 1.0, 'rd_ex': 1.0, 'rp_ex': 1.1, 'r_bm': 0.75}
    gkocost = {
        'G_g1': 1.0,
        'G_g2': 1.0,
        'G_g4': 3.0,
        'G_g5': 2.0,
        'G_g6': 1.0,
        'G_g7': 1.0,
        'G_g8': 1.0,
        'G_g9': 1.0,
    }
    gkicost = {
        'G_g3': 1.0,
    }
    regcost = {'G_g4 <= 0.4': 1.2}
    sd_setup = {
        MODULES: modules,
        MAX_COST: 2,
        MAX_SOLUTIONS: inf,
        SOLUTION_APPROACH: comp_approach,
        KOCOST: kocost,
        GKOCOST: gkocost,
        GKICOST: gkicost,
        REGCOST: regcost,
        SOLVER: 'cplex'
    }
    solution = sd.compute_strain_designs(model_gpr, sd_setup=sd_setup)
    assert (len(solution.gene_sd) == 4)
    assert (any(['G_g4 <= 0.4' in sol for sol in solution.get_gene_sd()]))


@pytest.mark.timeout(15)
def test_optknock(curr_solver, model_weak_coupling, comp_approach_best_populate, bigM, compression):
    """Test OptKnock computation."""
    modules = [sd.SDModule(model_weak_coupling, OPTKNOCK, outer_objective='r_P', inner_objective='r_BM',
                           constraints='r_BM>=1')]  # constraints=[[{'r_BM':1.0}, '>=' ,1.0 ]]
    kocost = {'r1': 1, 'r2': 1, 'r4': 1.1, 'r5': 0.75, 'r7': 0.8, 'r8': 1, 'r9': 1, 'r_S': 1.0, 'r_P': 1, 'r_BM': 1, 'r_Q': 1.5}
    kicost = {
        'r3': 0.6,
        'r6': 1.0,
    }
    regcost = {'r6 >= 4.5': 1.2}
    sd_setup = {
        MODULES: modules,
        MAX_COST: 4,
        MAX_SOLUTIONS: 3,
        SOLUTION_APPROACH: comp_approach_best_populate,
        KOCOST: kocost,
        KICOST: kicost,
        REGCOST: regcost,
        SOLVER: curr_solver,
        'compress': compression,
        'M': bigM
    }
    solution = sd.compute_strain_designs(model_weak_coupling, sd_setup=sd_setup)
    assert (len(solution.reaction_sd) == 3)


@pytest.mark.timeout(15)
def test_robustknock(curr_solver, model_weak_coupling, comp_approach_best_populate, bigM, compression):
    """Test RobustKnock computation."""
    modules = [
        sd.SDModule(model_weak_coupling,
                    ROBUSTKNOCK,
                    outer_objective='r_P',
                    inner_objective='r_BM',
                    constraints=[[{
                        'r_BM': 1.0
                    }, '>=', 1.0]])
    ]
    kocost = {'r1': 1, 'r2': 1, 'r4': 1.1, 'r5': 0.75, 'r7': 0.8, 'r8': 1, 'r9': 1, 'r_S': 1.0, 'r_P': 1, 'r_BM': 1, 'r_Q': 1.5}
    kicost = {
        'r3': 0.6,
        'r6': 1.0,
    }
    regcost = {'r6 >= 4.5': 1.2}
    sd_setup = {
        MODULES: modules,
        MAX_COST: 4,
        MAX_SOLUTIONS: 2,
        SOLUTION_APPROACH: comp_approach_best_populate,
        KOCOST: kocost,
        KICOST: kicost,
        REGCOST: regcost,
        SOLVER: curr_solver,
        'compress': compression,
        'M': bigM
    }
    solution = sd.compute_strain_designs(model_weak_coupling, sd_setup=sd_setup)
    for s in solution.get_reaction_sd():
        m = model_weak_coupling.copy()
        for itv, v in s.items():
            if v == -1.0:
                m.reactions.get_by_id(itv).bounds = (0, 0)
        sol_max_BM = sd.fba(m, obj='r_BM')
        sol_min_P = sd.fba(m, obj_sense='minimize', obj='r_P', constraints="r_BM >= " + str(sol_max_BM.objective_value))
        assert (sol_min_P.objective_value > 0)


@pytest.mark.timeout(15)
def test_optcouple(curr_solver, model_weak_coupling, comp_approach_best_populate, bigM, compression):
    """Test OptCouple computation."""
    modules = [sd.SDModule(model_weak_coupling, OPTCOUPLE, prod_id='r_P', inner_objective='r_BM', min_gcp=1.0)]
    kocost = {'r1': 1, 'r2': 1, 'r4': 1.1, 'r5': 0.75, 'r7': 0.8, 'r8': 1, 'r9': 1, 'r_S': 1.0, 'r_P': 1, 'r_BM': 1, 'r_Q': 1.5}
    kicost = {
        'r3': 0.6,
        'r6': 1.0,
    }
    regcost = {'r6 >= 4.5': 1.2}
    sd_setup = {
        MODULES: modules,
        MAX_COST: 6,
        MAX_SOLUTIONS: 3,
        SOLUTION_APPROACH: comp_approach_best_populate,
        KOCOST: kocost,
        KICOST: kicost,
        REGCOST: regcost,
        SOLVER: curr_solver,
        'compress': compression,
        'M': bigM
    }
    solution = sd.compute_strain_designs(model_weak_coupling, sd_setup=sd_setup)
    assert (len(solution.get_reaction_sd()) == 2)
    for s in solution.get_reaction_sd():
        constraints = []
        for itv, v in s.items():
            if v == -1.0:
                constraints += [[{itv: 1}, '=', 0]]
        constraints_no_P = constraints + [[modules[0][PROD_ID], '=', 0]]
        sol_max_BM_no_P = sd.fba(model_weak_coupling, obj='r_BM', constraints=constraints_no_P)
        max_BM_no_P = sol_max_BM_no_P.objective_value
        sol_max_BM = sd.fba(model_weak_coupling, obj='r_BM', constraints=constraints)
        max_BM = sol_max_BM.objective_value
        assert (max_BM - max_BM_no_P >= modules[0][MIN_GCP])
        constraints_max_BM = constraints + [[{'r_BM': 1}, '=', max_BM]]
        sol_min_P = sd.fba(model_weak_coupling, obj_sense='minimize', obj='r_P', constraints=constraints_max_BM)
        assert (sol_min_P.objective_value > 0)
    pass
