#!/usr/bin/env python3
"""Simple compression benchmark - compression only, no FVA."""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from cobra.io import read_sbml_model, load_model
import straindesign.networktools as nt
import warnings
warnings.filterwarnings('ignore')

def log(msg):
    print(msg, flush=True)

if __name__ == "__main__":
    log("="*70)
    log("COMPRESSION BENCHMARK (Python vs Java)")
    log("="*70)

    # Models to test
    models_info = [
        ("e_coli_core", lambda: load_model("e_coli_core")),
        ("iMLcore", lambda: read_sbml_model("tests/iMLcore.xml")),
        ("iML1515", lambda: load_model("iML1515")),
    ]

    all_results = {}
    n_runs = 3

    for model_name, loader in models_info:
        log(f"\n{'='*60}")
        log(f"Benchmarking {model_name}")
        log(f"{'='*60}")

        model = loader()
        log(f"Original: {len(model.reactions)} reactions, {len(model.metabolites)} metabolites")

        results = {'python_times': [], 'java_times': [], 'python_final': 0, 'java_final': 0}

        # Python compression
        log("\nPython compression:")
        for i in range(n_runs):
            model_copy = loader()
            start = time.perf_counter()
            nt.compress_model(model_copy, legacy_java_compression=False)
            elapsed = time.perf_counter() - start
            results['python_times'].append(elapsed)
            results['python_final'] = len(model_copy.reactions)
            log(f"  Run {i+1}: {elapsed:.3f}s -> {len(model_copy.reactions)} reactions")

        py_mean = np.mean(results['python_times'])
        py_std = np.std(results['python_times'])
        log(f"  Average: {py_mean:.3f}s ± {py_std:.3f}s")

        # Java compression
        log("\nJava compression:")
        for i in range(n_runs):
            model_copy = loader()
            start = time.perf_counter()
            nt.compress_model(model_copy, legacy_java_compression=True)
            elapsed = time.perf_counter() - start
            results['java_times'].append(elapsed)
            results['java_final'] = len(model_copy.reactions)
            log(f"  Run {i+1}: {elapsed:.3f}s -> {len(model_copy.reactions)} reactions")

        java_mean = np.mean(results['java_times'])
        java_std = np.std(results['java_times'])
        log(f"  Average: {java_mean:.3f}s ± {java_std:.3f}s")

        all_results[model_name] = results

    # Create plot
    log("\n" + "="*70)
    log("Creating plot...")

    model_names = list(all_results.keys())
    x = np.arange(len(model_names))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

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
    plt.savefig('compression_benchmark.png', dpi=150)
    log(f"Plot saved to compression_benchmark.png")

    # Summary
    log("\n" + "="*70)
    log("SUMMARY")
    log("="*70)
    log(f"{'Model':<15} {'Python (s)':<18} {'Java (s)':<18} {'Speedup':<12} {'Reactions'}")
    log("-"*75)

    for model_name, results in all_results.items():
        py_mean = np.mean(results['python_times'])
        py_std = np.std(results['python_times'])
        java_mean = np.mean(results['java_times'])
        java_std = np.std(results['java_times'])
        speedup = java_mean / py_mean if py_mean > 0 else 0

        log(f"{model_name:<15} {py_mean:.3f} ± {py_std:.3f}      "
            f"{java_mean:.3f} ± {java_std:.3f}      {speedup:.2f}x         "
            f"{results['python_final']}/{results['java_final']}")

    log("\nDone!")
