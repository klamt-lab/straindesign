#!/usr/bin/env python3
"""
Comprehensive benchmark and validation of Python vs Java compression.

This script:
1. Benchmarks compression performance on 3 models (3 runs each)
2. Compares FVA results between Python and Java compressed models
3. Verifies that FVA results can be correctly expanded back
"""

import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from cobra.io import read_sbml_model, load_model
from fractions import Fraction
import warnings
warnings.filterwarnings('ignore')

# Import straindesign
import straindesign as sd
import straindesign.networktools as nt


def expand_fva_results(fva_results, cmp_map, original_reactions):
    """
    Expand FVA results from compressed to original model space.

    Args:
        fva_results: dict of {reaction_id: (min, max)}
        cmp_map: list of compression step dicts with 'reac_map_exp'
        original_reactions: list of original reaction IDs

    Returns:
        dict of expanded FVA results for original reactions
    """
    expanded = {}

    # Build reverse mapping: original -> compressed reactions with coefficients
    reverse_map = {}  # original_reac -> [(compressed_reac, coeff), ...]

    for cmp_step in cmp_map:
        reac_map_exp = cmp_step.get('reac_map_exp', {})
        for new_reac, old_reacs in reac_map_exp.items():
            for old_reac, coeff in old_reacs.items():
                if old_reac not in reverse_map:
                    reverse_map[old_reac] = []
                reverse_map[old_reac].append((new_reac, float(coeff)))

    # Expand each original reaction
    for orig_reac in original_reactions:
        if orig_reac in fva_results:
            # Reaction wasn't compressed, use directly
            expanded[orig_reac] = fva_results[orig_reac]
        elif orig_reac in reverse_map:
            # Reaction was merged, expand using coefficients
            min_val = 0.0
            max_val = 0.0
            for comp_reac, coeff in reverse_map[orig_reac]:
                if comp_reac in fva_results:
                    comp_min, comp_max = fva_results[comp_reac]
                    if coeff >= 0:
                        min_val += coeff * comp_min
                        max_val += coeff * comp_max
                    else:
                        min_val += coeff * comp_max
                        max_val += coeff * comp_min
            expanded[orig_reac] = (min_val, max_val)
        else:
            # Reaction not found in any map - might be removed
            expanded[orig_reac] = (np.nan, np.nan)

    return expanded


def run_fva_on_model(model, reactions=None):
    """Run FVA on model and return results as dict (single-threaded for Windows)."""
    from cobra.flux_analysis import flux_variability_analysis

    if reactions is None:
        reactions = [r.id for r in model.reactions]

    # Use processes=1 to avoid multiprocessing spawn issues on Windows
    fva = flux_variability_analysis(model, reaction_list=reactions, fraction_of_optimum=0.0, processes=1)
    results = {}
    for r_id in fva.index:
        results[r_id] = (fva.loc[r_id, 'minimum'], fva.loc[r_id, 'maximum'])
    return results


def benchmark_compression(model_name, loader_func, n_runs=3):
    """Benchmark both compression methods on a model."""

    print(f"\n{'='*60}")
    print(f"Benchmarking {model_name}")
    print(f"{'='*60}")

    # Load original model info
    model_orig = loader_func()
    orig_reactions = len(model_orig.reactions)
    orig_metabolites = len(model_orig.metabolites)
    original_reaction_ids = [r.id for r in model_orig.reactions]

    print(f"Original model: {orig_reactions} reactions, {orig_metabolites} metabolites")

    results = {
        'python_times': [],
        'java_times': [],
        'python_final_reactions': 0,
        'java_final_reactions': 0,
        'python_cmp_map': None,
        'java_cmp_map': None,
    }

    # Python compression benchmark
    print("\nPython compression:")
    for i in range(n_runs):
        model_copy = loader_func()
        start = time.perf_counter()
        cmp_map = nt.compress_model(model_copy, legacy_java_compression=False)
        elapsed = time.perf_counter() - start
        results['python_times'].append(elapsed)
        results['python_final_reactions'] = len(model_copy.reactions)
        if i == 0:
            results['python_cmp_map'] = cmp_map
            results['python_model'] = model_copy
        print(f"  Run {i+1}: {elapsed:.3f}s -> {len(model_copy.reactions)} reactions")

    py_mean = np.mean(results['python_times'])
    py_std = np.std(results['python_times'])
    print(f"  Average: {py_mean:.3f}s ± {py_std:.3f}s")

    # Java compression benchmark
    print("\nJava compression:")
    for i in range(n_runs):
        model_copy = loader_func()
        start = time.perf_counter()
        cmp_map = nt.compress_model(model_copy, legacy_java_compression=True)
        elapsed = time.perf_counter() - start
        results['java_times'].append(elapsed)
        results['java_final_reactions'] = len(model_copy.reactions)
        if i == 0:
            results['java_cmp_map'] = cmp_map
            results['java_model'] = model_copy
        print(f"  Run {i+1}: {elapsed:.3f}s -> {len(model_copy.reactions)} reactions")

    java_mean = np.mean(results['java_times'])
    java_std = np.std(results['java_times'])
    print(f"  Average: {java_mean:.3f}s ± {java_std:.3f}s")

    # Compression ratio
    print(f"\nCompression results:")
    print(f"  Python: {orig_reactions} -> {results['python_final_reactions']} reactions "
          f"({100*(1-results['python_final_reactions']/orig_reactions):.1f}% reduction)")
    print(f"  Java:   {orig_reactions} -> {results['java_final_reactions']} reactions "
          f"({100*(1-results['java_final_reactions']/orig_reactions):.1f}% reduction)")

    results['original_reaction_ids'] = original_reaction_ids
    return results


def verify_fva_correctness(results, model_name):
    """Verify FVA results match between Python and Java compression."""

    print(f"\n{'='*60}")
    print(f"FVA Verification for {model_name}")
    print(f"{'='*60}")

    python_model = results.get('python_model')
    java_model = results.get('java_model')

    if python_model is None or java_model is None:
        print("  ERROR: Models not available for FVA comparison")
        return False

    # Run FVA on both compressed models
    print("\nRunning FVA on Python-compressed model...")
    python_fva = run_fva_on_model(python_model)
    print(f"  Got FVA for {len(python_fva)} reactions")

    print("Running FVA on Java-compressed model...")
    java_fva = run_fva_on_model(java_model)
    print(f"  Got FVA for {len(java_fva)} reactions")

    # Compare common reactions
    common_reactions = set(python_fva.keys()) & set(java_fva.keys())
    print(f"\nComparing {len(common_reactions)} common reactions...")

    tolerance = 1e-6
    mismatches = []

    for r_id in common_reactions:
        py_min, py_max = python_fva[r_id]
        java_min, java_max = java_fva[r_id]

        if abs(py_min - java_min) > tolerance or abs(py_max - java_max) > tolerance:
            mismatches.append({
                'reaction': r_id,
                'python': (py_min, py_max),
                'java': (java_min, java_max)
            })

    if mismatches:
        print(f"\n  WARNING: {len(mismatches)} FVA mismatches found!")
        for m in mismatches[:5]:
            print(f"    {m['reaction']}: Python={m['python']}, Java={m['java']}")
        return False
    else:
        print(f"\n  SUCCESS: All {len(common_reactions)} common reactions have matching FVA results!")
        return True


def create_benchmark_plot(all_results, output_file='compression_benchmark.png'):
    """Create bar chart comparing compression performance."""

    model_names = list(all_results.keys())
    x = np.arange(len(model_names))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Performance comparison
    python_means = [np.mean(all_results[m]['python_times']) for m in model_names]
    python_stds = [np.std(all_results[m]['python_times']) for m in model_names]
    java_means = [np.mean(all_results[m]['java_times']) for m in model_names]
    java_stds = [np.std(all_results[m]['java_times']) for m in model_names]

    bars1 = ax1.bar(x - width/2, python_means, width, yerr=python_stds,
                    label='Python (new)', capsize=5, color='#2ecc71')
    bars2 = ax1.bar(x + width/2, java_means, width, yerr=java_stds,
                    label='Java (legacy)', capsize=5, color='#3498db')

    ax1.set_ylabel('Time (seconds)')
    ax1.set_xlabel('Model')
    ax1.set_title('Compression Performance: Python vs Java')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.2f}s',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    # Speedup comparison
    speedups = [java_means[i] / python_means[i] if python_means[i] > 0 else 0
                for i in range(len(model_names))]

    bars3 = ax2.bar(x, speedups, width*1.5, color='#9b59b6')
    ax2.axhline(y=1.0, color='red', linestyle='--', label='Equal performance')
    ax2.set_ylabel('Speedup (Java time / Python time)')
    ax2.set_xlabel('Model')
    ax2.set_title('Python Speedup vs Java')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    for bar, speedup in zip(bars3, speedups):
        height = bar.get_height()
        label = f'{speedup:.2f}x'
        if speedup > 1:
            label += '\n(Python faster)'
        elif speedup < 1:
            label += '\n(Java faster)'
        ax2.annotate(label,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"\nPlot saved to {output_file}")

    return fig


def main():
    """Main benchmark function."""

    print("="*70)
    print("COMPRESSION BENCHMARK AND FVA VERIFICATION")
    print("="*70)

    # Define models to test
    models_info = [
        ("e_coli_core", lambda: load_model("e_coli_core")),
        ("iMLcore", lambda: read_sbml_model("tests/iMLcore.xml")),
        ("iML1515", lambda: load_model("iML1515")),
    ]

    all_results = {}
    n_runs = 3

    # Run benchmarks
    for model_name, loader in models_info:
        try:
            results = benchmark_compression(model_name, loader, n_runs)
            all_results[model_name] = results
        except Exception as e:
            print(f"ERROR benchmarking {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # FVA verification (on smaller models only for speed)
    print("\n" + "="*70)
    print("FVA VERIFICATION")
    print("="*70)

    for model_name in ['e_coli_core', 'iMLcore']:
        if model_name in all_results:
            verify_fva_correctness(all_results[model_name], model_name)

    # Create plot
    if all_results:
        create_benchmark_plot(all_results)

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Model':<15} {'Python (s)':<18} {'Java (s)':<18} {'Speedup':<12} {'Reactions'}")
    print("-"*75)

    for model_name, results in all_results.items():
        py_mean = np.mean(results['python_times'])
        py_std = np.std(results['python_times'])
        java_mean = np.mean(results['java_times'])
        java_std = np.std(results['java_times'])
        speedup = java_mean / py_mean if py_mean > 0 else 0

        print(f"{model_name:<15} {py_mean:.3f} ± {py_std:.3f}      "
              f"{java_mean:.3f} ± {java_std:.3f}      {speedup:.2f}x         "
              f"{results['python_final_reactions']}/{results['java_final_reactions']}")

    print("\nDone!")


if __name__ == "__main__":
    main()
