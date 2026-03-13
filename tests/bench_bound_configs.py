#!/usr/bin/env python3
"""Bound configuration experiments for MILP performance.

Standalone script (not pytest). Tests different bound configurations on
preprocessed models to determine which gives the best MILP performance.

Usage:
    # First, create a preprocessed dump (run from repo root):
    python -m pytest tests/test_09_performance.py::test_iml1515_14bdo_mcs -v -s --large

    # Or create dumps manually:
    python tests/bench_bound_configs.py --create-dump imlcore
    python tests/bench_bound_configs.py --create-dump iml1515

    # Run experiments:
    python tests/bench_bound_configs.py <dump_path> [--seeds N] [--configs PA,FB,...]

Bound configs apply independently to PROTECT (primal) and SUPPRESS (Farkas):

  Primal configs (PROTECT):
    P-A: Current (redundant bounds -> ±inf after FVA)
    P-B: FVA-tightened (redundant bounds -> FVA limits)
    P-C: Homogeneous (redundant -> ±inf, additionally homogenize where able)

  Farkas configs (SUPPRESS):
    F-A: Current behavior
    F-B: FVA-tightened (all bounds -> FVA limits)
    F-C: Homogeneous redundant (non-zero finite redundant -> ±inf)
    F-D: Knockable-only (keep bounds only for knockable reactions)
    F-E: Knockable FVA (FVA limits for knockable, remove redundant non-knockable)

Critical constraint: only REDUNDANT bounds are modified (FVA doesn't reach them).
Functional bounds (FVA reaches the bound) are always preserved.
"""
import argparse
import json
import logging
import math
import os
import pickle
import sys
import time
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Add parent to path for straindesign imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import straindesign as sd
from straindesign import fva, select_solver, compute_strain_designs_from_preprocessed
from straindesign.names import *


TESTS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = TESTS_DIR / "perf_results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Bound modification helpers
# ---------------------------------------------------------------------------

def classify_bounds(model, constraints=None, solver='gurobi'):
    """Run FVA and classify each reaction's bounds as functional or redundant.

    Returns dict: reaction_id -> {
        'fva_min': float, 'fva_max': float,
        'lb': float, 'ub': float,
        'lb_functional': bool,  # FVA min reaches lb
        'ub_functional': bool,  # FVA max reaches ub
    }
    """
    kwargs = {'solver': solver, 'compress': False}
    if constraints:
        kwargs['constraints'] = constraints
    flux_limits = fva(model, **kwargs)

    result = {}
    tol = 1e-9
    for r in model.reactions:
        limits = flux_limits.loc[r.id]
        fva_min = float(limits['minimum'])
        fva_max = float(limits['maximum'])
        lb = float(r.lower_bound)
        ub = float(r.upper_bound)
        # Functional: FVA reaches within tolerance of the bound
        lb_func = (abs(fva_min - lb) < tol) if not math.isinf(lb) else False
        ub_func = (abs(fva_max - ub) < tol) if not math.isinf(ub) else False
        result[r.id] = {
            'fva_min': fva_min, 'fva_max': fva_max,
            'lb': lb, 'ub': ub,
            'lb_functional': lb_func,
            'ub_functional': ub_func,
        }
    return result


def apply_primal_config(model, config, bound_info):
    """Apply a primal bound configuration to the model (modifies in-place)."""
    for r in model.reactions:
        info = bound_info.get(r.id)
        if info is None:
            continue

        if config == 'P-A':
            pass  # Current behavior (already applied by preprocessing)

        elif config == 'P-B':
            # FVA-tighten redundant bounds
            if not info['lb_functional'] and not math.isinf(info['lb']):
                r._lower_bound = info['fva_min']
            if not info['ub_functional'] and not math.isinf(info['ub']):
                r._upper_bound = info['fva_max']

        elif config == 'P-C':
            # Remove redundant bounds (make homogeneous where possible)
            if not info['lb_functional']:
                if r.lower_bound < 0:
                    r._lower_bound = -np.inf
                # lb=0 is already homogeneous, keep it
            if not info['ub_functional']:
                if r.upper_bound > 0:
                    r._upper_bound = np.inf


def apply_farkas_config(model, config, bound_info, knockable_ids=None):
    """Apply a Farkas bound configuration to the model (modifies in-place)."""
    if knockable_ids is None:
        knockable_ids = set()

    for r in model.reactions:
        info = bound_info.get(r.id)
        if info is None:
            continue

        if config == 'F-A':
            pass  # Current behavior

        elif config == 'F-B':
            # FVA-tighten all bounds
            if not info['lb_functional'] and not math.isinf(info['lb']):
                r._lower_bound = info['fva_min']
            if not info['ub_functional'] and not math.isinf(info['ub']):
                r._upper_bound = info['fva_max']

        elif config == 'F-C':
            # Homogeneous redundant: non-zero finite redundant -> ±inf
            # Preserve variable classification (sign of lb/ub)
            if not info['lb_functional'] and r.lower_bound < 0 and not math.isinf(float(r.lower_bound)):
                r._lower_bound = -np.inf
            if not info['ub_functional'] and r.upper_bound > 0 and not math.isinf(float(r.upper_bound)):
                r._upper_bound = np.inf

        elif config == 'F-D':
            # Knockable-only: keep bounds for knockable, remove redundant non-knockable
            if r.id not in knockable_ids:
                if not info['lb_functional'] and r.lower_bound < 0 and not math.isinf(float(r.lower_bound)):
                    r._lower_bound = -np.inf
                if not info['ub_functional'] and r.upper_bound > 0 and not math.isinf(float(r.upper_bound)):
                    r._upper_bound = np.inf

        elif config == 'F-E':
            # Knockable FVA: FVA limits for knockable, remove redundant non-knockable
            if r.id in knockable_ids:
                if not info['lb_functional'] and not math.isinf(info['lb']):
                    r._lower_bound = info['fva_min']
                if not info['ub_functional'] and not math.isinf(info['ub']):
                    r._upper_bound = info['fva_max']
            else:
                if not info['lb_functional'] and r.lower_bound < 0 and not math.isinf(float(r.lower_bound)):
                    r._lower_bound = -np.inf
                if not info['ub_functional'] and r.upper_bound > 0 and not math.isinf(float(r.upper_bound)):
                    r._upper_bound = np.inf


# ---------------------------------------------------------------------------
# Dump creation helpers
# ---------------------------------------------------------------------------

def create_imlcore_dump(dump_path):
    """Create preprocessed dump for iMLcore ethanol benchmark."""
    from cobra.io import read_sbml_model
    m = read_sbml_model(str(TESTS_DIR / "iMLcore.xml"))
    modules = [
        sd.SDModule(m, SUPPRESS,
                    constraints=["EX_etoh_e <= 1.0",
                                 "BIOMASS_Ec_iML1515_core_75p37M >= 0.14"]),
        sd.SDModule(m, PROTECT,
                    constraints=["BIOMASS_Ec_iML1515_core_75p37M >= 0.15"]),
    ]
    sol = sd.compute_strain_designs(
        m, sd_modules=modules,
        solution_approach=POPULATE, max_cost=2,
        gene_kos=True, solver='gurobi',
        dump_preprocessed=dump_path,
    )
    print(f"iMLcore dump saved to {dump_path}")
    return dump_path


def create_iml1515_dump(dump_path):
    """Create preprocessed dump for iML1515 + 14BDO benchmark."""
    from cobra.io import load_model
    # Import the 14BDO pathway setup from the performance test
    sys.path.insert(0, str(TESTS_DIR))
    from test_09_performance import _add_14bdo_pathway

    m = load_model("iML1515")
    _add_14bdo_pathway(m)

    ko_cost = {r.id: 1.0 for r in m.reactions
               if r.genes and m.genes.get_by_id('s0001') not in r.genes}
    ko_cost['EX_o2_e'] = 1.0
    ko_cost.pop('AKGDC', None)
    ko_cost.pop('SSCOARx', None)
    ki_cost = {'AKGDC': 1.0, 'SSCOARx': 1.0}

    module_suppress = sd.SDModule(m, SUPPRESS,
        constraints='EX_14bdo_e + 0.25 EX_glc__D_e <= 0')
    module_protect = sd.SDModule(m, PROTECT,
        constraints='BIOMASS_Ec_iML1515_core_75p37M >= 0.1')

    sol = sd.compute_strain_designs(
        m, sd_modules=[module_suppress, module_protect],
        solution_approach=ANY, max_cost=40, max_solutions=1,
        gene_kos=True, ko_cost=ko_cost, ki_cost=ki_cost,
        solver='gurobi', dump_preprocessed=dump_path,
    )
    print(f"iML1515+14BDO dump saved to {dump_path}")
    return dump_path


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(dump_path, p_config, f_config, seed=42, solver='gurobi'):
    """Run a single bound-configuration experiment.

    Returns dict with timing and M-value statistics.
    """
    with open(dump_path, 'rb') as f:
        d = pickle.load(f)

    cmp_model = d['cmp_model']
    sd_modules = d['sd_modules']
    kwargs_milp = d['kwargs_milp']

    # Identify knockable reactions
    ko_cost = kwargs_milp.get(KOCOST, {})
    ki_cost_dict = kwargs_milp.get(KICOST, {})
    knockable_ids = set(ko_cost.keys()) | set(ki_cost_dict.keys())

    # Run FVA on compressed model to classify bounds
    # We need constraints from each module to properly classify
    # For simplicity, run unconstrained FVA (conservative — may classify
    # some functional bounds as redundant, but that's safe)
    logging.info(f"  Classifying bounds on compressed model ({len(cmp_model.reactions)} rxns)...")
    from straindesign.networktools import suppress_lp_context
    with suppress_lp_context(cmp_model):
        bound_info = classify_bounds(cmp_model, solver=solver)

    # Count baseline statistics
    n_lb_finite = sum(1 for r in cmp_model.reactions if not math.isinf(float(r.lower_bound)) and r.lower_bound != 0)
    n_ub_finite = sum(1 for r in cmp_model.reactions if not math.isinf(float(r.upper_bound)) and r.upper_bound != 0)
    n_lb_redundant = sum(1 for info in bound_info.values() if not info['lb_functional'] and not math.isinf(info['lb']) and info['lb'] != 0)
    n_ub_redundant = sum(1 for info in bound_info.values() if not info['ub_functional'] and not math.isinf(info['ub']) and info['ub'] != 0)

    # Apply bound configurations to a copy of the model
    with suppress_lp_context(cmp_model):
        model_copy = cmp_model.copy()
        apply_primal_config(model_copy, p_config, bound_info)
        apply_farkas_config(model_copy, f_config, bound_info, knockable_ids)

    # Reload dump with modified model
    d_copy = dict(d)
    d_copy['cmp_model'] = model_copy

    # Save modified dump to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
        tmp_path = tmp.name
        pickle.dump(d_copy, tmp)

    try:
        # Run MILP solve
        t0 = time.perf_counter()
        sol = compute_strain_designs_from_preprocessed(
            tmp_path, seed=seed, solver=solver,
            solution_approach=ANY, max_solutions=1,
        )
        elapsed = time.perf_counter() - t0
    finally:
        os.unlink(tmp_path)

    result = {
        'p_config': p_config,
        'f_config': f_config,
        'seed': seed,
        'solver': solver,
        'total_time': round(elapsed, 3),
        'n_solutions': len(sol.reaction_sd),
        'status': sol.status,
        'model_rxns': len(cmp_model.reactions),
        'n_knockable': len(knockable_ids),
        'n_lb_finite': n_lb_finite,
        'n_ub_finite': n_ub_finite,
        'n_lb_redundant': n_lb_redundant,
        'n_ub_redundant': n_ub_redundant,
    }
    return result


def run_matrix(dump_path, configs, seeds, solver='gurobi'):
    """Run a matrix of bound configurations x seeds."""
    results = []
    for p_cfg, f_cfg in configs:
        for seed in seeds:
            label = f"{p_cfg} x {f_cfg} (seed={seed})"
            logging.info(f"\n{'='*60}")
            logging.info(f"Config: {label}")
            logging.info(f"{'='*60}")
            try:
                result = run_experiment(dump_path, p_cfg, f_cfg, seed=seed, solver=solver)
                result['label'] = label
                results.append(result)
                logging.info(f"  -> {result['total_time']:.1f}s, {result['n_solutions']} solutions")
            except Exception as e:
                logging.error(f"  FAILED: {e}")
                results.append({
                    'label': label, 'p_config': p_cfg, 'f_config': f_cfg,
                    'seed': seed, 'error': str(e),
                })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_CONFIGS = [
    ('P-A', 'F-A'),  # Baseline
    ('P-B', 'F-A'),  # Tighten primal only
    ('P-B', 'F-B'),  # Tighten both
    ('P-B', 'F-C'),  # Tight primal, lean Farkas
    ('P-B', 'F-E'),  # Tight primal, tight knockable Farkas
]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dump_path', nargs='?', help='Path to preprocessed pickle')
    parser.add_argument('--create-dump', choices=['imlcore', 'iml1515'],
                        help='Create a preprocessed dump instead of running experiments')
    parser.add_argument('--seeds', type=int, default=3, help='Number of seeds (default: 3)')
    parser.add_argument('--solver', default='gurobi', help='Solver (default: gurobi)')
    parser.add_argument('--configs', help='Comma-separated config pairs (e.g. PA-FA,PB-FA)')
    parser.add_argument('--output', help='Output JSON path')
    args = parser.parse_args()

    if args.create_dump:
        dump_dir = TESTS_DIR / "bench_dumps"
        dump_dir.mkdir(exist_ok=True)
        if args.create_dump == 'imlcore':
            create_imlcore_dump(str(dump_dir / 'imlcore_ethanol.pkl'))
        elif args.create_dump == 'iml1515':
            create_iml1515_dump(str(dump_dir / 'iml1515_14bdo.pkl'))
        return

    if not args.dump_path:
        parser.error("dump_path is required when not using --create-dump")

    # Parse configs
    if args.configs:
        configs = []
        for pair in args.configs.split(','):
            p, f = pair.strip().split('-', 1)
            configs.append((f'P-{p.upper()}', f'F-{f.upper()}'))
    else:
        configs = DEFAULT_CONFIGS

    seeds = list(range(42, 42 + args.seeds))

    logging.info(f"Dump: {args.dump_path}")
    logging.info(f"Configs: {configs}")
    logging.info(f"Seeds: {seeds}")
    logging.info(f"Solver: {args.solver}")

    results = run_matrix(args.dump_path, configs, seeds, solver=args.solver)

    # Print summary table
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<20} {'Seeds':<8} {'Median(s)':<12} {'Min(s)':<10} {'Max(s)':<10} {'Solutions'}")
    print("-" * 70)

    # Group by config
    from collections import defaultdict
    by_config = defaultdict(list)
    for r in results:
        if 'error' not in r:
            key = f"{r['p_config']} x {r['f_config']}"
            by_config[key].append(r)

    for key, runs in sorted(by_config.items()):
        times = [r['total_time'] for r in runs]
        sols = [r['n_solutions'] for r in runs]
        print(f"{key:<20} {len(runs):<8} {np.median(times):<12.1f} "
              f"{min(times):<10.1f} {max(times):<10.1f} {sols[0]}")

    # Save results
    out_path = args.output or str(RESULTS_DIR / f"bound_configs_{int(time.time())}.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
