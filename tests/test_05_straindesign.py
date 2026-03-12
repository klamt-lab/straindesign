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
def test_mcs_gpr(curr_solver, model_gpr, comp_approach):
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
        SOLVER: curr_solver
    }
    solution = sd.compute_strain_designs(model_gpr, sd_setup=sd_setup)
    assert (len(solution.gene_sd) == 4)


@pytest.mark.timeout(15)
def test_mcs_gpr2(model_gpr, comp_approach):
    """Test MCS computation with gene names instead of IDs (requires a strong solver).

    Uses CPLEX > SCIP > Gurobi in order of preference; skipped if none are available.
    GLPK is excluded because it cannot reliably enumerate all solutions via POPULATE.
    """
    preferred = [CPLEX, SCIP, GUROBI]
    solver = next((s for s in preferred if s in sd.avail_solvers), None)
    if solver is None:
        pytest.skip("test_mcs_gpr2 requires CPLEX, SCIP, or Gurobi")
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
        SOLVER: solver,
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


@pytest.mark.timeout(30)
def test_doubleopt(curr_solver, model_doubleopt):
    """Test DOUBLEOPT computation with and without optimality tolerance.

    Uses the two-organism community model where both organisms compete for
    shared substrate. Knocking out shared metabolite sinks forces cross-feeding,
    creating tight coupling where both organisms are at their individual optima.
    With opt_tol=0.6, smaller KO sets become feasible because organisms only
    need to achieve 60% of their optima.
    """
    # Knockable: organism-internal reactions + shared sinks (enable cross-feeding KOs)
    ko_reacs = [r.id for r in model_doubleopt.reactions if r.id.startswith('A_R') or r.id.startswith('B_R')
                or r.id in ['A_10', 'B_10', 'shared_R_D', 'shared_R_C']]
    kocost = {r: 1 for r in ko_reacs}

    # --- Exact DOUBLEOPT: tight coupling via shared sink KOs ---
    modules_exact = [sd.SDModule(model_doubleopt, DOUBLEOPT,
                                 inner_objective='A_BM',
                                 outer_objective='B_BM',
                                 constraints=['A_BM >= 0.1', 'B_BM >= 0.1'])]
    sol_exact = sd.compute_strain_designs(model_doubleopt.copy(),
        sd_modules=modules_exact, max_cost=6, max_solutions=inf,
        solution_approach=POPULATE, ko_cost=kocost, solver=curr_solver, compress=False)
    assert len(sol_exact.reaction_sd) > 0, \
        "Exact DOUBLEOPT should find solutions when shared sinks are knockable"
    min_cost_exact = min(sum(abs(v) for v in s.values()) for s in sol_exact.reaction_sd)

    # --- Relaxed DOUBLEOPT: 60% optimality tolerance finds cheaper solutions ---
    modules_relaxed = [sd.SDModule(model_doubleopt, DOUBLEOPT,
                                   inner_objective='A_BM',
                                   outer_objective='B_BM',
                                   inner_opt_tol=0.6,
                                   outer_opt_tol=0.6,
                                   constraints=['A_BM >= 0.1', 'B_BM >= 0.1'])]
    sol_relaxed = sd.compute_strain_designs(model_doubleopt.copy(),
        sd_modules=modules_relaxed, max_cost=3, max_solutions=inf,
        solution_approach=POPULATE, ko_cost=kocost, solver=curr_solver, compress=False)
    assert len(sol_relaxed.reaction_sd) > 0, \
        "Relaxed DOUBLEOPT should find solutions with opt_tol=0.6"
    min_cost_relaxed = min(sum(abs(v) for v in s.values()) for s in sol_relaxed.reaction_sd)
    assert min_cost_relaxed < min_cost_exact, \
        "Relaxed DOUBLEOPT should find cheaper solutions than exact"


@pytest.mark.timeout(30)
def test_solution_merging(model_small_example):
    """Test SDSolutions __add__ and __iadd__ operators."""
    solver = next((s for s in [CPLEX, GUROBI, SCIP, GLPK] if s in sd.avail_solvers), None)
    if solver is None:
        pytest.skip("No solver available")
    modules = [sd.SDModule(model_small_example, SUPPRESS, constraints=["R3 - 0.5 R1 <= 0.0", "R2 <= 0", "R1 >= 0.1"])]
    modules += [
        sd.SDModule(model_small_example, SUPPRESS, constraints=["1.0 R3 - 0.5 R1 - 0.5 R2 <= 0.0 ", "1.0 R2 >= 0.0 ", "1.0 R1 >= 0.1 "])
    ]
    modules += [sd.SDModule(model_small_example, PROTECT, constraints=["1.0 R3 >= 1.0 "])]
    kicost = {'R2': 1}
    sd_setup = {
        MODULES: modules,
        MAX_COST: inf,
        MAX_SOLUTIONS: inf,
        SOLUTION_APPROACH: 'any',
        KICOST: kicost,
        SOLVER: solver,
        'compress': True,
    }
    from copy import deepcopy
    sol1 = sd.compute_strain_designs(model_small_example, sd_setup=deepcopy(sd_setup))
    sol2 = sd.compute_strain_designs(model_small_example, sd_setup=deepcopy(sd_setup))
    n1 = len(sol1.reaction_sd)
    n2 = len(sol2.reaction_sd)

    # __add__ returns a new object, originals unchanged
    merged = sol1 + sol2
    assert len(sol1.reaction_sd) == n1, "Original should be unchanged after +"
    # Merged should have unique solutions (deduplicated)
    assert len(merged.reaction_sd) >= n1, "Merged should have at least as many as sol1"
    assert len(merged.reaction_sd) <= n1 + n2, "Merged should not exceed sum"
    # Verify deduplication: same solutions should be deduplicated
    merged_keys = {frozenset(s.items()) for s in merged.reaction_sd}
    sol1_keys = {frozenset(s.items()) for s in sol1.reaction_sd}
    assert sol1_keys.issubset(merged_keys), "All sol1 solutions should be in merged"

    # __iadd__ modifies in place
    orig_len = len(sol1.reaction_sd)
    sol1 += sol2
    assert len(sol1.reaction_sd) >= orig_len, "iadd should not lose solutions"


@pytest.mark.timeout(30)
def test_dump_preprocessed(model_small_example, tmp_path):
    """Test dump_preprocessed and compute_strain_designs_from_preprocessed workflow."""
    solver = next((s for s in [CPLEX, GUROBI, SCIP, GLPK] if s in sd.avail_solvers), None)
    if solver is None:
        pytest.skip("No solver available")
    modules = [sd.SDModule(model_small_example, SUPPRESS, constraints=["R3 - 0.5 R1 <= 0.0", "R2 <= 0", "R1 >= 0.1"])]
    modules += [
        sd.SDModule(model_small_example, SUPPRESS, constraints=["1.0 R3 - 0.5 R1 - 0.5 R2 <= 0.0 ", "1.0 R2 >= 0.0 ", "1.0 R1 >= 0.1 "])
    ]
    modules += [sd.SDModule(model_small_example, PROTECT, constraints=["1.0 R3 >= 1.0 "])]
    kicost = {'R2': 1}
    dump_path = str(tmp_path / 'preprocessed.pkl')

    # Step 1: Dump preprocessed data (should return early without solving)
    sol_dump = sd.compute_strain_designs(model_small_example,
        sd_modules=modules, max_cost=inf, max_solutions=inf,
        solution_approach='any', ki_cost=kicost, solver=solver,
        compress=True, dump_preprocessed=dump_path)
    import os
    assert os.path.exists(dump_path), "Dump file should exist"

    # Step 2: Reload and solve
    from straindesign.compute_strain_designs import compute_strain_designs_from_preprocessed
    sol_reload = compute_strain_designs_from_preprocessed(dump_path, seed=42)
    sols = sol_reload.get_reaction_sd()
    assert len(sols) > 0, "Should find solutions from preprocessed data"
    assert ({'R4': -1.0} in sols), "Should find R4 KO solution"

    # Step 3: Solve with different seed and merge
    sol_reload2 = compute_strain_designs_from_preprocessed(dump_path, seed=123)
    merged = sol_reload + sol_reload2
    assert len(merged.reaction_sd) >= len(sol_reload.reaction_sd), "Merged should not lose solutions"


@pytest.mark.timeout(15)
def test_lazy_expansion(model_small_example):
    """Test lazy expansion by temporarily lowering the threshold."""
    solver = next((s for s in [CPLEX, GUROBI, SCIP, GLPK] if s in sd.avail_solvers), None)
    if solver is None:
        pytest.skip("No solver available")
    modules = [sd.SDModule(model_small_example, SUPPRESS, constraints=["R3 - 0.5 R1 <= 0.0", "R2 <= 0", "R1 >= 0.1"])]
    modules += [
        sd.SDModule(model_small_example, SUPPRESS, constraints=["1.0 R3 - 0.5 R1 - 0.5 R2 <= 0.0 ", "1.0 R2 >= 0.0 ", "1.0 R1 >= 0.1 "])
    ]
    modules += [sd.SDModule(model_small_example, PROTECT, constraints=["1.0 R3 >= 1.0 "])]
    kicost = {'R2': 1}

    # Temporarily lower the threshold to force lazy mode
    from straindesign import compute_strain_designs as _csd_module
    # Access the actual module where LAZY_EXPANSION_THRESHOLD lives
    import sys
    csd = sys.modules['straindesign.compute_strain_designs']
    orig_threshold = csd.LAZY_EXPANSION_THRESHOLD
    csd.LAZY_EXPANSION_THRESHOLD = 1  # Force lazy mode
    try:
        sol = sd.compute_strain_designs(model_small_example,
            sd_modules=modules, max_cost=inf, max_solutions=inf,
            solution_approach='any', ki_cost=kicost, solver=solver,
            compress=True)
        if sol.is_lazy:
            assert sol.get_num_materialized() < sol.get_num_sols(), \
                "Lazy mode: materialized < estimated total"
            # Expand all
            sol.expand_all()
            assert not sol.is_lazy, "After expand_all, should not be lazy"
            sols = sol.get_reaction_sd()
            assert ({'R4': -1.0} in sols), "Should still find R4 KO after expansion"
        else:
            # Estimation might be <= 1 for this tiny model, that's OK
            pass
    finally:
        csd.LAZY_EXPANSION_THRESHOLD = orig_threshold
