"""One-time snapshot creation for MILP benchmarking.

Runs the full compute_strain_designs pipeline on iMLcore with gene_kos=True,
max_cost=3, SUPPRESS biomass >= 0.001, solution_approach=POPULATE.

Monkey-patches SDMILP.__init__ and SDMILP.enumerate to capture:
  - The compressed model and sd_modules passed to the MILP constructor
  - The kwargs_milp dict (without solver)
  - The MILP-level solution count (before decompression)

Saves everything to tests/milp_snapshot/ for repeatable benchmarking.

Usage:
    conda run -n straindesign python tests/create_milp_snapshot.py
"""

import os
import sys
import json
import pickle
import logging
from copy import deepcopy
from pathlib import Path

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cobra.io import read_sbml_model
from straindesign import compute_strain_designs, SDModule, SDMILP
from straindesign.names import SUPPRESS, MODULES, MAX_COST, SOLVER, SOLUTION_APPROACH, POPULATE

logging.basicConfig(level=logging.INFO, format="%(message)s")

SNAPSHOT_DIR = Path(__file__).resolve().parent / "milp_snapshot"
MODEL_PATH = Path(__file__).resolve().parent / "iMLcore.xml"

# Storage for captured state
_captured = {}


def _patch_sdmilp():
    """Monkey-patch SDMILP to capture pre-MILP state and solution count."""
    _orig_init = SDMILP.__init__
    _orig_enumerate = SDMILP.enumerate

    def _capturing_init(self, model, sd_modules, **kwargs):
        # Deep-copy everything before the original __init__ consumes it
        _captured["model"] = model.copy()
        _captured["sd_modules"] = deepcopy(sd_modules)
        # Save kwargs without 'solver' so snapshot is solver-independent
        kw = deepcopy(kwargs)
        kw.pop("solver", None)
        _captured["kwargs_milp"] = kw
        return _orig_init(self, model, sd_modules, **kwargs)

    def _capturing_enumerate(self, **kwargs):
        result = _orig_enumerate(self, **kwargs)
        _captured["milp_solution_count"] = len(result.reaction_sd)
        return result

    SDMILP.__init__ = _capturing_init
    SDMILP.enumerate = _capturing_enumerate


def _unpatch_sdmilp():
    """Restore original SDMILP methods (defensive)."""
    # The originals are closured in _patch; we just delete the overrides
    if hasattr(SDMILP.__init__, "__wrapped__"):
        SDMILP.__init__ = SDMILP.__init__.__wrapped__


def main():
    print(f"Loading model from {MODEL_PATH} ...")
    model = read_sbml_model(str(MODEL_PATH))
    print(f"  {len(model.reactions)} reactions, {len(model.metabolites)} metabolites, {len(model.genes)} genes")

    # Biomass reaction for iMLcore
    biomass_id = "BIOMASS_Ec_iML1515_core_75p37M"
    assert biomass_id in [r.id for r in model.reactions], \
        f"Biomass reaction '{biomass_id}' not found in model"

    module = SDModule(model, SUPPRESS, constraints=f"{biomass_id} >= 0.001")

    print("\nRunning compute_strain_designs (gurobi, gene_kos=True, max_cost=3, populate) ...")
    _patch_sdmilp()

    sol = compute_strain_designs(
        model,
        sd_modules=[module],
        max_cost=3,
        gene_kos=True,
        solver="gurobi",
        solution_approach=POPULATE,
    )

    decompressed_count = len(sol.reaction_sd)
    milp_count = _captured.get("milp_solution_count", -1)

    print(f"\nResults:")
    print(f"  MILP-level solutions:         {milp_count}")
    print(f"  Decompressed solutions:        {decompressed_count}")

    # Save snapshot
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    cmp_model_path = SNAPSHOT_DIR / "cmp_model.pkl"
    milp_args_path = SNAPSHOT_DIR / "milp_args.pkl"
    metadata_path = SNAPSHOT_DIR / "metadata.json"

    with open(cmp_model_path, "wb") as f:
        pickle.dump(_captured["model"], f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(milp_args_path, "wb") as f:
        pickle.dump(
            {"sd_modules": _captured["sd_modules"], "kwargs_milp": _captured["kwargs_milp"]},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    metadata = {
        "expected_milp_count": milp_count,
        "expected_decompressed_count": decompressed_count,
        "model_file": "iMLcore.xml",
        "biomass_constraint": f"{biomass_id} >= 0.001",
        "max_cost": 3,
        "gene_kos": True,
        "solution_approach": "populate",
        "reference_solver": "gurobi",
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSnapshot saved to {SNAPSHOT_DIR}/")
    print(f"  cmp_model.pkl   ({cmp_model_path.stat().st_size / 1024:.1f} KB)")
    print(f"  milp_args.pkl   ({milp_args_path.stat().st_size / 1024:.1f} KB)")
    print(f"  metadata.json   ({metadata_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
