# StrainDesign Developer's Guide

> **Reference paper:** Schneider P., Bekiaris P. S., von Kamp A., Klamt S. — *StrainDesign: a comprehensive Python package for computational design of metabolic networks.* Bioinformatics, btac632 (2022). DOI: 10.1093/bioinformatics/btac632

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Repository Structure](#2-repository-structure)
3. [Public API & Entry Points](#3-public-api--entry-points)
4. [Preprocessing Pipeline](#4-preprocessing-pipeline)
   - 4.1 [Model Cleaning](#41-model-cleaning)
   - 4.2 [Compression Pass 1](#42-compression-pass-1-before-gpr-extension)
   - 4.3 [FVA](#43-fva)
   - 4.4 [GPR Integration](#44-gpr-integration)
   - 4.5 [Compression Pass 2](#45-compression-pass-2-after-gpr-extension)
   - 4.6 [Network Compression — Algorithm Details](#46-network-compression--algorithm-details)
   - 4.7 [Regulatory Interventions](#47-regulatory-interventions)
5. [MILP Construction](#5-milp-construction)
   - 5.1 [Overview & z-map Matrices](#51-overview--z-map-matrices)
   - 5.2 [Primal LP from COBRApy Model](#52-primal-lp-from-cobrapy-model)
   - 5.3 [SUPPRESS Module — Farkas' Lemma Dual](#53-suppress-module--farkas-lemma-dual)
   - 5.4 [PROTECT Module](#54-protect-module)
   - 5.5 [OPTKNOCK — LP Duality for Bilevel Optimization](#55-optknock--lp-duality-for-bilevel-optimization)
   - 5.6 [ROBUSTKNOCK — Three-Level Duality](#56-robustknock--three-level-duality)
   - 5.7 [OPTCOUPLE — Growth Coupling Potential](#57-optcouple--growth-coupling-potential)
   - 5.8 [Linking Binary Variables: `link_z`](#58-linking-binary-variables-link_z)
   - 5.9 [Big-M Bounding](#59-big-m-bounding)
   - 5.10 [Indicator Constraints vs. Big-M](#510-indicator-constraints-vs-big-m)
6. [Solver Backends](#6-solver-backends)
   - 6.1 [MILP_LP Unified Interface](#61-milp_lp-unified-interface)
   - 6.2 [CPLEX](#62-cplex)
   - 6.3 [Gurobi](#63-gurobi)
   - 6.4 [SCIP](#64-scip)
   - 6.5 [GLPK](#65-glpk)
7. [Computation Approaches — SDMILP](#7-computation-approaches--sdmilp)
   - 7.1 [ANY](#71-any)
   - 7.2 [BEST](#72-best)
   - 7.3 [POPULATE](#73-populate)
   - 7.4 [Exclusion Constraints](#74-exclusion-constraints)
   - 7.5 [Solution Verification](#75-solution-verification)
8. [Post-processing & Result Container](#8-post-processing--result-container)
   - 8.1 [Solution Decompression](#81-solution-decompression)
   - 8.2 [GPR Translation](#82-gpr-translation)
   - 8.3 [SDSolutions](#83-sdsolutions)
9. [Constraint & Expression Parsing](#9-constraint--expression-parsing)
10. [Supporting Utilities](#10-supporting-utilities)
    - 10.1 [LP Suppression (`networktools.py`)](#networktoolspy--lp-suppression)
11. [Known Issues, Urgent Actions & Future Work](#11-known-issues-urgent-actions--future-work)
12. [MILP Performance Notes](#12-milp-performance-notes)
13. [Testing](#13-testing)

---

## 1. Architecture Overview

StrainDesign solves **metabolic strain design** problems: given a genome-scale metabolic model (COBRApy `Model`) and a set of biological objectives, find the minimal sets of gene/reaction knockouts (and additions) that force the network into a desired flux state.

All methods reduce to **Mixed-Integer Linear Programs (MILP)**. The key intellectual contribution is the translation of nested/bilevel biological objectives into a flat MILP via LP duality and Farkas' lemma, so a single MILP call suffices to find strain designs. The pipeline is:

```
User (model + SDModules)
        │
        ▼
   Preprocessing        networktools.py, compression.py, lptools.py
   ─ FVA cleanup
   ─ GPR extension
   ─ Network compression
        │
        ▼
   MILP Construction    strainDesignProblem.py  (SDProblem)
   ─ build_primal_from_cbm
   ─ farkas_dualize / LP_dualize
   ─ link_z (bounding + indicator constraints)
        │
        ▼
   Solver Execution     strainDesignMILP.py  (SDMILP)
   ─ ANY / BEST / POPULATE loops
   ─ verify_sd
   ─ exclusion constraints
        │
        ▼
   Post-processing      compute_strain_designs.py, strainDesignSolutions.py
   ─ expand_sd (decompress)
   ─ GPR translation
   ─ SDSolutions
```

All methods (**MCS/suppress/protect**, **OptKnock**, **RobustKnock**, **OptCouple**) use the same pipeline — only how `addModule` builds the LP blocks differs.

---

## 2. Repository Structure

```
straindesign/
├── __init__.py                     # Package exports & avail_solvers detection
├── names.py                        # All string constants (solver names, module types, etc.)
│
├── compute_strain_designs.py       # Main user-facing orchestration function
├── strainDesignModule.py           # SDModule: problem specification (a dict subclass)
├── strainDesignProblem.py          # SDProblem: translates model+modules → MILP matrices
├── strainDesignMILP.py             # SDMILP: solves the MILP, manages solution loop
├── strainDesignSolutions.py        # SDSolutions: result container, GPR translation
│
├── compression.py                  # Network compression (RREF, RationalMatrix, nullspace)
├── networktools.py                 # GPR extension, regulatory extension, compress wrappers, LP suppression
├── lptools.py                      # FVA, flux space plotting, solver selection
├── parse_constr.py                 # Constraint/expression string → matrix conversion
├── indicatorConstraints.py         # IndicatorConstraints data class
├── solver_interface.py             # MILP_LP factory: instantiates the right solver
├── cplex_interface.py              # CPLEX backend (Cplex_MILP_LP)
├── gurobi_interface.py             # Gurobi backend (Gurobi_MILP_LP)
├── scip_interface.py               # SCIP backend (SCIP_MILP_LP)
├── glpk_interface.py               # GLPK backend (GLPK_MILP_LP)
├── efmtool_cmp_interface.py        # EFMtool JAR interface (legacy compression backend)
├── pool.py                         # SDPool: Windows-compatible multiprocessing pool
└── efmtool.jar                     # Bundled EFMtool binary
```

Tests are in:
```
tests/
└── *.py   (pytest test files, one per feature area)
```

Configuration / packaging:
```
setup.py           # python_requires=">=3.7"; classifiers include 3.9–3.13
conda-recipe/meta.yaml
.github/workflows/CI-test.yml
```

---

## 3. Public API & Entry Points

### Primary function

```python
from straindesign import compute_strain_designs, SDModule

sols = compute_strain_designs(
    model,
    sd_modules   = [sd_module1, sd_module2],
    solver       = 'gurobi',      # 'cplex' | 'gurobi' | 'scip' | 'glpk'
    max_cost     = 3,
    max_solutions= 10,
    solution_approach = 'best',   # 'any' | 'best' | 'populate'
    ko_cost      = {'r1': 1, 'r2': 2},   # reaction knockouts
    ki_cost      = {'r3': 1},             # reaction additions
    gko_cost     = {'geneA': 1},          # gene knockouts
    gki_cost     = {'geneB': 1},          # gene additions
    time_limit   = 3600,
)
```

### SDModule specification

```python
# MCS-style: suppress undesired subspace, protect desired one
SDModule(model, 'suppress', constraints='EX_product_e >= 0.1')
SDModule(model, 'protect',  constraints='BIOMASS >= 0.05')

# With inner optimization (force to inner optimum first)
SDModule(model, 'suppress',
         constraints='EX_product_e >= 0.1',
         inner_objective='BIOMASS', inner_opt_sense='maximize')

# OptKnock
SDModule(model, 'optknock',
         inner_objective='BIOMASS',
         outer_objective='EX_product_e')

# RobustKnock
SDModule(model, 'robustknock',
         inner_objective='BIOMASS',
         outer_objective='EX_product_e')

# OptCouple
SDModule(model, 'optcouple',
         inner_objective='BIOMASS',
         prod_id='EX_product_e',
         min_gcp=0.1)
```

### Result access

```python
sols.status           # 'optimal', 'infeasible', 'time_limit', ...
sols.reaction_sd      # list of dicts: {reac_id: -1 (KO) | +1 (KI) | 0 (not added KI)}
sols.gene_sd          # list of dicts: gene-level interventions (if gene_kos=True)
sols.get_strain_designs()          # same as reaction_sd
sols.get_gene_reac_sd_assoc()      # (reac_sd, assoc, gene_sd) with GPR linkage
sols.get_strain_design_costs()     # list of floats
sols.get_reaction_sd_bnds()        # intervention expressed as flux bounds
```

### Lower-level access

```python
from straindesign import SDProblem, SDMILP, MILP_LP
from straindesign import fva, select_solver
from straindesign.compression import compress_cobra_model
```

---

## 4. Preprocessing Pipeline

`compute_strain_designs` (`compute_strain_designs.py`) orchestrates all preprocessing before constructing the MILP. The pipeline uses **two-pass compression**: the model is compressed once *before* GPR extension (so that FVA and GPR processing operate on a smaller model), then again *after* GPR extension (to compress the newly added gene pseudoreactions). This roughly halves model size at each downstream step and yields a ~4x end-to-end speedup on genome-scale models like iML1515.

### 4.1 Model Cleaning

**Functions:** `remove_ext_mets`, `remove_dummy_bounds`
(in `networktools.py`, re-exported from `compression.py`).

Steps:
1. **Remove dummy bounds** — COBRApy uses ±1000 as a proxy for ±∞. Anything above a `bound_thres` (from `cobra.Configuration`) is replaced with `±inf` so big-M bounding works correctly.
2. **Deep-copy** the model so the user's model is never mutated.
3. **Remove external metabolites** — metabolites with no stoichiometric coupling to the rest of the network (boundary metabolites already handled by COBRApy exchange reactions, but orphaned ones are pruned).

### 4.2 Compression Pass 1 (before GPR extension)

**File:** `compression.py`
**Orchestrated by:** `networktools.compress_model`

Before FVA or GPR processing, the model is compressed with `propagate_gpr=True`. This merges coupled and parallel reactions while propagating GPR rules through the compression map: when reactions are lumped, their GPR rules are AND-combined (using sympy `And`/`Or` constructors for flattening and deduplication). The result is a much smaller model that retains correct GPR annotations.

After compress #1:
- `compress_modules(sd_modules, cmp_mapReac_1)` — rewrites constraint/objective reaction IDs.
- `compress_ki_ko_cost(ko_cost, ki_cost, cmp_mapReac_1)` — maps reaction and regulatory costs (gene costs are not yet present and are handled separately after GPR extension).

### 4.3 FVA

**Function:** `bound_blocked_or_irrevers_fva`, `fva` (from `lptools.py`), `speedy_fva` (from `speedy_fva.py`)

Now running on the **compressed model** (roughly half the reactions of the original), FVA identifies:
- **Blocked reactions** (FVA min = FVA max = 0): forced to zero, removed from intervention candidates.
- **Irreversible reactions** (FVA min >= 0 or FVA max <= 0): bounds tightened.
- **Essential reactions** (cannot be zero under module constraints): excluded from knockout candidates.

`fva()` in `lptools.py` delegates to `speedy_fva()`, which replaces the legacy implementation with a two-phase approach that reduces LP count by 70-80% on genome-scale models.

#### speedy_fva algorithm

**Phase 1 — Scan LPs** resolve the majority of bounds cheaply:

1. **v=0 feasibility (Phase 1a):** If `lb[j] <= 0 <= ub[j]`, then zero flux is feasible, so `min x_j = 0` (or `max x_j = 0` when the bound is one-sided). No LP needed.

2. **min(sum|x|) scan LP (Phase 1b):** Builds an extended LP with splitting variables for reversible reactions to minimize total absolute flux. The optimal vertex pushes many reactions to zero or their bounds. A vectorized **bound scan** marks any reaction whose flux equals its variable bound (within tolerance).

3. **Push-to-bounds iteration (Phase 1c):** Using the same LP with dual simplex warm-start, iteratively sets directed objectives that push unresolved reactions toward their upper/lower bounds. Each round: set `c[j] = -1` for unresolved-max reactions, solve, scan; then `c[j] = +1` for unresolved-min reactions, solve, scan. Stops when a round resolves fewer than 5 new bounds. Warm-started dual simplex makes each re-solve near-instant (the basis stays primal feasible after objective changes).

**Phase 2 — Individual LPs** for remaining unresolved objectives:
- **Sequential mode** (default for small/medium models): warm-started dual simplex with periodic LP rebuild (every 200 solves) to limit degeneration. Each LP vertex is checked for **co-optimization**: a bound scan on the full solution vector that may resolve other unresolved directions for free.
- **Parallel mode** (for large models, configurable via `threads`): process-based `SDPool` with per-worker LP instances.

**Optional compression:** `speedy_fva` can compress the model internally (auto-enabled for n >= 200). This uses single-pass nullspace compression + conservation removal, with a fast model copy that avoids expensive `deepcopy(solver)`.

**Benchmarks** (Gurobi, iML1515, compressed, single-threaded):

| Configuration | Standard FVA | speedy_fva | LP count | Speedup |
|---|---|---|---|---|
| Unconstrained | 90s | 15s | 71% fewer | 6.0x |
| Biomass >= 0.1 | 17s | 13s | 41% fewer | 1.3x |

Phase 1 resolves 70-80% of objectives in the unconstrained case, but only ~40% when constraints limit the feasible space (fewer reactions can achieve their variable bounds).

#### Design decisions and rejected alternatives

**KKT dual certification (investigated, not pursued):**
We explored using KKT optimality conditions to certify co-optimal reactions without solving additional LPs. After solving `min x_j`, we obtain a vertex solution and its basis. For another reaction k, optimality at the same vertex requires dual variables satisfying `S^T lambda + mu_ub - mu_lb = e_k` with correct complementarity. This can be checked by:

1. Extracting the basis B from the solver
2. Computing reduced costs for objective `e_k`: `pi = B^{-T} e_{pos(k)}`, then `r_i = -A_i^T pi` for nonbasic variables
3. Verifying sign conditions: `r_i >= 0` for nonbasic-at-lb, `r_i <= 0` for nonbasic-at-ub

The problem is that **bound scanning already handles nonbasic variables** (they are at their bounds by definition in a simplex vertex). KKT certification only adds value for *basic* variables (at interior points), which requires a sparse LU factorization of B per LP solve (1-5ms for m=900) — comparable to or exceeding the cost of the warm-started dual simplex LP itself (~0.5ms). The overhead exceeds the savings.

An earlier implementation (`DualChecker`) used null-space projection to test dual feasibility via sparse LU. On **compressed models**, this produced zero certifications because compression creates linearly dependent column subsets (coupled reactions with diagonal values of 2/3 instead of 1), causing the factorization check to reject all candidates. On uncompressed models certifications were possible but the null-space dimension was too large (d=39-304 on iMLcore) for efficient LP-based certification. The approach was removed in favor of the simpler bound scan, which captures the same information for nonbasic variables at zero cost.

**Intelligent objective ordering (considered, marginal benefit):**
After each LP solve, the basis determines which variables are basic (interior) vs nonbasic (at bounds). Solving next for a basic variable should require fewer dual simplex pivots than jumping to a distant nonbasic variable. However, after bound scanning, the remaining unresolved reactions are precisely those that were basic (interior) in *every* solution seen — these "hard" reactions resist being pushed to bounds regardless of ordering. The objective change is always a rank-1 update, and pivot count depends more on problem structure than ordering. Warm-started dual simplex is already so fast per-LP (~0.5ms) that ordering heuristics cannot recover their own overhead.

### 4.4 GPR Integration

**File:** `networktools.py`
**Key functions:** `reduce_gpr`, `extend_model_gpr`

When gene-level interventions are requested (`gko_cost` or `gki_cost` provided), the model is **extended** with artificial reactions that represent gene-level knockouts. This allows the MILP's binary variables `z` to operate on genes rather than reactions. Because the model has already been compressed, there are fewer reactions to extend and fewer GPR rules to process.

#### Step 1 — Reduce GPR rules (`reduce_gpr`)

Uses **AST-based GPR evaluation** (`evaluate_gpr_ast`):
- Each reaction's GPR rule is parsed to an AST (`ast.parse`).
- Genes are removed if their knockout/addition cannot affect any non-blocked, non-essential reaction.
- This is determined by evaluating the DNF/CNF GPR expression with gene states `{True, False, None (undetermined)}`.
- Reduces the gene intervention space before extension.

#### Step 2 — Extend model with artificial gene reactions (`extend_model_gpr`)

For each gene/isoenzyme group that participates in the GPR rules, artificial reactions and metabolites are added to encode the boolean logic:

- **AND gate**: If a reaction requires `g1 AND g2`, an artificial metabolite and two artificial reactions are added. Knocking out either gene removes the pathway.
- **OR gate**: If `g1 OR g2`, parallel paths are created. Both must be knocked out to abolish the reaction.

The function returns a `reac_map` dict mapping original reactions to the new artificial ones in the extended model. This map is used later to:
1. Translate `gko_cost`/`gki_cost` into reaction costs on the extended model.
2. Expand computed solutions back to gene-level interventions.

**Important:** Gene names starting with digits are prefixed with `'g'` (`cobra.manipulation.rename_genes`) because Python AST cannot parse identifiers starting with digits.

#### Step 3 — GPR solution translation (`SDSolutions.__init__`)

After computation, the artificial reactions in solutions are translated back to gene-level interventions using `gpr_eval`. This evaluates each solution's intervention set against the DNF GPR rules to determine which reactions are truly knocked out (or added).

`gpr_eval(cj_terms, interv)`:
- `cj_terms`: the GPR rule as a list of "conjunctive clauses" (AND-groups within OR expression).
- `interv`: current gene intervention state.
- Returns `True/False/nan` (active/inactive/undetermined).

### 4.5 Compression Pass 2 (after GPR extension)

After GPR extension adds gene pseudoreactions, the model is compressed a second time. This pass merges newly coupled gene pseudoreactions and parallel reactions introduced by the GPR encoding. The `no_par_compress_reacs` set is rebuilt from the (now rewritten) `sd_modules` to protect constraint/objective reactions.

The final compression map is the concatenation of both passes: `cmp_mapReac = cmp_mapReac_1 + cmp_mapReac_2`. When decompressing solutions, `expand_sd` processes this list in reverse order (pass 2 first, then pass 1).

A final FVA pass identifies essential reactions in the fully compressed model and removes them from the knockout candidates.

### 4.6 Network Compression — Algorithm Details

**File:** `compression.py`
**Orchestrated by:** `networktools.compress_model`

Network compression reduces model size while preserving the strain design problem. It is **lossless** — all solutions in the compressed space correspond to solutions in the original, and vice versa (via the `cmp_mapReac` mapping).

#### RationalMatrix

The core data structure for numerically exact RREF computation. It uses **sparse dual-integer storage** (numerator matrix + denominator matrix, both `scipy.sparse.csr_matrix`). Operations are performed in rational arithmetic using `fractions.Fraction` and `sympy.Rational`. The model stays in exact rational arithmetic throughout both compression passes; no float conversion is performed between passes.

Key operations: row operations, scaling, finding pivots, extracting nullspace.

Floating-point coefficients are converted via `float_to_rational` with configurable precision (`detect_max_precision` scans the model's stoichiometry).

#### Compression Algorithm (StoichMatrixCompressor)

The main class `StoichMatrixCompressor` performs the following in each cycle, repeating until no further reduction is possible:

1. **Parallel reactions** (`compress_model_parallel`):
   Reactions with identical stoichiometry (same column in `S`) are merged via hash-based O(nnz) comparison. This is cheap and runs first to shrink the model before the expensive RREF step. The resulting "super-reaction" carries the combined cost, and the z-variable controls all of them simultaneously. Hash matches are verified with exact index/data comparison to prevent collisions.

2. **Conservation relation removal** (`remove_conservation_relations`):
   Compute the left nullspace of the stoichiometric matrix `S` over the rationals (homogeneous dependencies among metabolite rows). Linearly dependent rows are removed. This reduces the row count of `S` for the subsequent RREF computation.

3. **Exit check**: If the loop has completed at least one full cycle (including a coupled step) and neither parallel nor coupled compression reduced the reaction count, exit. This ensures the model is always fully cleaned (conservation relations removed) on exit.

4. **Coupled reactions** (`compress_model_coupled`):
   Two reactions `r_i` and `r_j` are **coupled** if there exists a scalar `alpha` such that in every feasible steady state, `v_i = alpha * v_j`. These can be lumped into one reaction. Coupling is detected by computing the right nullspace (RREF) and identifying proportional columns. Lumpable reactions are merged and their costs combined. When `propagate_gpr=True`, GPR rules are AND-combined during merging. **Contradicting groups** (where bounds intersection yields lb > ub or lb = ub = 0) are detected and removed.

Blocked reactions are removed once before the loop via `remove_blocked_reactions`.

#### Compression Result

`cmp_mapReac`: a list of dicts, one per compression pass. Each dict maps compressed reaction IDs to their original reactions with scaling factors.

This is used by:
- `compress_modules(sd_modules, cmp_mapReac)` — rewrites constraints/objectives in terms of compressed reaction IDs.
- `compress_ki_ko_cost(cmp_ko_cost, cmp_ki_cost, cmp_mapReac)` — maps intervention costs.
- `expand_sd(sd, cmp_mapReac)` — after solving, maps binary solutions back to original reactions (processes passes in reverse order).

#### Backend Selection

The compression backend is chosen by the `compression_backend` parameter:
- `'sparse_rref'` (default): Pure Python, uses `RationalMatrix` for exact arithmetic. No Java dependency.
- `'efmtool_rref'` (legacy): Uses the bundled `efmtool.jar` via JPype for RREF. Requires `pip install straindesign[java]`. Available via `compress_model_efmtool`.

### 4.7 Regulatory Interventions

**Function:** `extend_model_regulatory` (`networktools.py`)

Regulatory interventions represent changes that activate or silence a pathway without directly knocking out a gene. They are encoded similarly to gene interventions: artificial reactions are added to the model, and their "presence" or "absence" is controlled by binary z-variables.

Regulatory constraints are **split by parsability**: constraints that reference only existing reaction IDs (e.g., `r6 >= 4.5`) are applied before compression pass 1; constraints that reference gene IDs (e.g., `g4 <= 0.4`) are deferred until after `extend_model_gpr` creates the corresponding pseudoreactions.

After solving, `postprocess_reg_sd` in `compute_strain_designs.py` converts regulatory intervention values from numeric (`1`/`0`) to boolean (`True`/`False`) in the solution dict.

---

## 5. MILP Construction

**File:** `strainDesignProblem.py`
**Classes:** `SDProblem`, `ContMILP`
**Key functions:** `build_primal_from_cbm`, `LP_dualize`, `farkas_dualize`, `reassign_lb_ub_from_ineq`, `prevent_boundary_knockouts`, `link_z`

### 5.1 Overview & z-map Matrices

The MILP has the form:

```
minimize   c' * x
subject to:
  A_ineq * x  <=  b_ineq
  A_eq   * x   =  b_eq
  lb  <=  x  <=  ub
  x[0..num_z-1]  ∈ {0,1}         (binary z-variables: interventions)
  x[num_z..]     ∈ ℝ             (continuous: flux variables + dual variables)
  indicator constraints or big-M
```

The first `num_z` variables are binary: `z_i = 1` means reaction `i` is knocked out (or added, if `z_inverted[i]`).

Three **z-map matrices** track which z-variables affect which parts of the LP:
- `z_map_vars[i, j]`: z-variable `i` controls continuous variable `j` (knockout/addition of that reaction flux).
- `z_map_constr_ineq[i, j]`: z-variable `i` controls inequality constraint `j`.
- `z_map_constr_eq[i, j]`: z-variable `i` controls equality constraint `j`.

A value of `+1` means "`z=1` removes this constraint/variable" (knockout); `-1` means "`z=1` adds this" (knock-in).

These maps are propagated through all dualization steps (see below) and ultimately used in `link_z` to wire up big-M coefficients or indicator constraints.

**MILP header rows** (fixed positions):
- Row 0 (`idx_row_maxcost`): `−cost' z ≤ 0` (ensures non-negative cost)
- Row 1 (`idx_row_mincost`): `cost' z ≤ max_cost`
- Row 2 (`idx_row_obj`): objective constraint (used by `fixObjective`)

### 5.2 Primal LP from COBRApy Model

**Function:** `build_primal_from_cbm(model, V_ineq, v_ineq, V_eq, v_eq, c=None)`

Constructs the standard LP from a COBRApy model:

```
A_eq  * v  =  b_eq    (stoichiometric matrix S stacked with V_eq)
A_ineq* v  <= b_ineq  (V_ineq constraints, e.g. flux ratio constraints)
lb <= v <= ub         (reaction bounds from model)
```

- `A_eq = [S; V_eq]`, `b_eq = [0; v_eq]`
- `z_map_vars = I` (identity — each z directly maps to its reaction variable)
- `z_map_constr_ineq = 0`, `z_map_constr_eq = 0` (in the primal, knockouts affect variables, not constraints)

**`prevent_boundary_knockouts`**: If a variable has `lb < 0` or `ub > 0`, knocking it out to zero requires converting these non-zero bounds into inequality constraints. This prevents situations where a "zero flux" constraint on a reaction with `lb = −1000` would be applied incorrectly.

### 5.3 SUPPRESS Module — Farkas' Lemma Dual

**Function:** `farkas_dualize(A_ineq_p, b_ineq_p, A_eq_p, b_eq_p, lb_p, ub_p, ...)`

The **SUPPRESS** module wants to make a certain flux subspace infeasible. Farkas' lemma provides a certificate of infeasibility:

**Farkas' Lemma:** The system `{A_ineq x ≤ b_ineq, A_eq x = b_eq, lb ≤ x ≤ ub}` is **infeasible** if and only if there exist multipliers `y` such that:
```
A_ineq' y_ineq + A_eq' y_eq = 0   (dual feasibility)
b_ineq' y_ineq + b_eq' y_eq < 0   (certificate: b'y < 0)
y_ineq ≥ 0
```

The Farkas dual is constructed by calling `LP_dualize` with zero objective (`c = 0`), yielding dual variables `y`. Then the constraint `c_dual' y ≤ −1` is appended (equivalently: `b_ineq' y_ineq + b_eq' y_eq ≤ −1`), which enforces the Farkas infeasibility certificate.

When an intervention (z) removes a primal constraint or variable, the corresponding dual variable is also removed — this is the mechanism by which knockouts "activate" the SUPPRESS condition. When the dual system becomes feasible with the `≤ −1` constraint, the corresponding primal flux subspace is infeasible, i.e., the targeted flux state is suppressed.

**Known limitation:** For the special case `A x = b, x ∈ ℝ, b ≈ 0` (homogeneous equality with unrestricted variable), the Farkas certificate requires `b'y ≠ 0` rather than `< 0`. This edge case is **not currently implemented** (see section 11).

### 5.4 PROTECT Module

The **PROTECT** module keeps a flux subspace feasible. It uses the **primal** LP directly (`build_primal_from_cbm`) without dualization, then calls `reassign_lb_ub_from_ineq` to tighten bounds. As long as the primal system is feasible, the protected state is reachable. The objective is set to zero (feasibility only).

### 5.5 OPTKNOCK — LP Duality for Bilevel Optimization

**OptKnock** solves:
```
maximize  outer_objective(v)
s.t.      v ∈ argmax { inner_objective(v) : v feasible }
```

This bilevel program is converted to a single-level MILP using **LP duality**:

**LP Duality Theorem:** For a minimization LP `{min c'x : Ax ≤ b, x ≥ 0}`, the dual is `{max b'y : A'y ≤ c, y ≥ 0}`. At optimality, primal and dual objectives are equal: `c'x* = b'y*`.

**`LP_dualize(A_ineq_p, b_ineq_p, A_eq_p, b_eq_p, lb_p, ub_p, c_p, ...)`**

Constructs the dual LP. The primal constraints become dual variables and vice versa:

| Primal | → | Dual |
|--------|---|------|
| `x ≥ 0` | | `y ≤ c_i` (inequality constraint) |
| `x ∈ ℝ` | | `y = c_i` (equality constraint) |
| `x ≤ 0` | | `y ≥ c_i` (flipped inequality) |
| `A_ineq row` | | `y_ineq ≥ 0` |
| `A_eq row` | | `y_eq ∈ ℝ` |
| Objective `b'y` | | `max → min`, `b` becomes new objective |

The OptKnock MILP block is assembled as:

```python
# Build: [primal (with constraints) | dual (without constraints)]
# Connect: c_inner_primal == c_inner_dual (strong duality)
A_eq_combined = block_diag(A_eq_v, A_eq_dual) stacked with
                hstack(c_primal, c_dual_objective) == 0
```

The outer objective is set as the MILP objective. The strong duality equality guarantees that `v` is at the inner optimum.

### 5.6 ROBUSTKNOCK — Three-Level Duality

**RobustKnock** solves a **min-max** problem:
```
maximize  min_v { outer(v) : v ∈ argmax inner(v) }
```

This requires **two applications of LP duality**:

1. First, the inner bilevel problem `{inner(v) at max}` is dualized into primal+dual pair (like OptKnock).
2. The combined primal+dual system is treated as a new primal, and **dualized again** to convert the outer minimization to a maximization.

The result is a three-layer LP construction assembled into a single MILP block. The outer objective (minimized production, maximized over worst case) becomes the MILP objective.

This is the most complex module and the most sensitive to numerical issues.

### 5.7 OPTCOUPLE — Growth Coupling Potential

**OptCouple** maximizes the **Growth Coupling Potential (GCP)**:
```
GCP = max_v { inner(v) } − max_v { inner(v) | prod = 0 }
```

The two terms are the max growth with and without production. The objective is their difference.

Construction:
1. Build primal for the "with production" LP → `(A_ineq_p, ..., c_p)`.
2. Build primal for the "baseline" LP (no production constraint) → `(A_ineq_b, ..., c_b)`.
3. Dualize the baseline LP → `c_dual_b` (dual objective coefficients).
4. Assemble combined block with strong duality for both LPs.
5. Final objective: `c_p - c_b` (growth with production minus baseline).

`min_gcp` parameter sets a lower bound on GCP in the MILP.

### 5.8 Linking Binary Variables: `link_z`

**Function:** `link_z` in `SDProblem` (called at end of `__init__`).

This is the step that physically wires the binary variables `z` into the MILP. It proceeds in 5 steps:

**Step 1 — Split knockable equalities into pairs of inequalities:**
An equality `A_eq[j] x = b` that is controlled by z is split into `A_ineq x ≤ b` and `−A_ineq x ≤ −b`. Both inequalities are individually linked to z, so when z is activated, both are activated together (enforcing zero).

**Step 2 — Convert variable knockouts to inequality knockouts:**
A variable knockout (controlled by `z_map_vars`) is converted to: `x ≤ 0` and `−x ≤ 0`. These bound constraints are added to `A_ineq` and linked to z via `z_map_constr_ineq`.

**Step 3 — Compute big-M values via LP bounding (parallel):**
For each knockable constraint `A[i] x ≤ b[i]`, compute:
```
M_i = max { A[i] x : x in most-relaxed feasible LP }
```
The "most-relaxed LP" removes all knockable constraints (allowing all knockouts) to get an upper bound on `A[i] x`. This gives the tightest valid big-M for each constraint.

Parallelized via `SDPool` when there are > 1000 constraints. Each worker (`worker_init`/`worker_compute`) holds a private LP object.

**Step 4 — Link z via big-M for bounded constraints (finite M):**
For constraints where a finite `M_i` was found:
- z=1 knocks out `A[i] x ≤ b`: add coefficient `−M_i` in column z_i of row i:
  `A[i] x − M_i * z ≤ b` → when `z=1`: `A[i] x ≤ b + M_i` (always satisfied if `M_i` tight enough).
- z=0 knocks out: `A[i] x + (M_i − b) * z ≤ M_i`.

**Step 5 — Create indicator constraints for remaining knockables:**
Constraints where `M_i = ∞` (unbounded LP subspace) cannot use big-M. These become **indicator constraints** passed to the solver backend:
```
z_i = indicval  →  A[i] x ≤ b[i]   (sense 'L')
z_i = indicval  →  A[i] x  = b[i]  (sense 'E')
```

### 5.9 Big-M Bounding

The quality of big-M values is critical:
- Too small → valid solutions may be cut off (incorrect results).
- Too large → numerical instability, especially for GLPK.
- M = ∞ → the constraint must use an indicator constraint (solver-dependent, slower).

StrainDesign computes per-constraint tight bounds via LP. The LP maximizes `A[i] x` subject to the static (non-knockable) constraints, giving the tightest valid `M_i` for each knockable constraint.

For GLPK (which has no indicator constraint support), `M` defaults to the COBRApy bound threshold (default: 1000). This is conservative and may cause numerical issues on models with large flux ranges. Setting `M` manually via the `M` parameter allows tuning.

Round-up: computed Ms are ceiled to 5 significant decimal digits.

### 5.10 Indicator Constraints vs. Big-M

| Feature | Indicator Constraints | Big-M |
|---|---|---|
| Solver support | CPLEX, Gurobi, SCIP | All (required for GLPK) |
| Numerical stability | Excellent (no large coefficients) | Can be poor with large M |
| Solver speed | Typically faster (native branching) | Depends on M tightness |
| When used | Unbounded constraints (M = ∞) | Bounded constraints (finite M) |
| Forced by user | `M=some_value` forces big-M everywhere | Default for GLPK |

The `IndicatorConstraints` class (`indicatorConstraints.py`) stores:
- `binv`: indices of binary variables that trigger each constraint
- `A`: coefficient matrix for the triggered constraints
- `b`: RHS
- `sense`: `'L'` (≤), `'E'` (=), `'G'` (≥)
- `indicval`: which value of z (0 or 1) activates the constraint

Each solver backend translates `IndicatorConstraints` into its native format. For GLPK, the solver interface translates them to big-M constraints using the provided `M`.

**Sign convention in z-maps (subtle!):**
- `z_map_constr_ineq[i, j] = +1`: z_i = 1 → constraint j removed (knockout)
- `z_map_constr_ineq[i, j] = −1`: z_i = 1 → constraint j active (knock-in)

This sign is inverted when KIs vs. KOs are assigned (in `z_kos_kis` diagonal matrix at end of `SDProblem.__init__`), so the final indicator constraint `indicval` correctly maps to 0 (KO: constraint active when z=0) or 1 (KI: constraint active when z=1).

---

## 6. Solver Backends

### 6.1 MILP_LP Unified Interface

**File:** `solver_interface.py`
**Class:** `MILP_LP`

`MILP_LP` is a **factory/dispatcher**: its `__init__` detects the requested solver and instantiates the appropriate backend class (`Cplex_MILP_LP`, `Gurobi_MILP_LP`, etc.). All backends expose the same interface:

```python
class MILP_LP:
    solve()                       # → (x, opt, status)
    slim_solve()                  # → float (objective value only, fast)
    populate(n)                   # → (x_list, opt, status)  [CPLEX/Gurobi only]
    set_objective(c)
    set_ub(ub_list)               # [(idx, val), ...]
    set_time_limit(t)
    add_ineq_constraints(A, b)
    add_eq_constraints(A, b)
    set_ineq_constraint(row, c, b) # Overwrite a specific row
    clear_objective()
    set_objective_idx(idx_val_list)
```

`slim_solve()` is used heavily for verification — it runs the LP and returns just the objective value (or `nan` for infeasible). This avoids extracting the full solution vector.

### 6.2 CPLEX

**File:** `cplex_interface.py`
**Class:** `Cplex_MILP_LP`

- Native indicator constraint support (`cplex.indicator_constraints.add`)
- Solution pool (`populate_solution_pool`) for enumerating multiple solutions
- Thread control via `cplex.parameters.threads`
- Seed via `cplex.parameters.randomseed`
- Status mapped from CPLEX codes to StrainDesign names (`OPTIMAL`, `INFEASIBLE`, etc.)
- **Advantages:** Best solution pool support, fastest for large MILPs in practice, population-based enumeration.
- **Limitation:** Licensed (academic/commercial). No indicator constraints → automatic big-M fallback.

### 6.3 Gurobi

**File:** `gurobi_interface.py`
**Class:** `Gurobi_MILP_LP`

- Native indicator constraint support (`gp.Model.addGenConstrIndicator`)
- Solution pool via Gurobi `SolCount` / `PoolSolutions`
- Seed via `Seed` parameter
- Status mapped from Gurobi `Status` codes
- **Advantages:** Often fastest solver, excellent solution pool. MIPFocus parameter available for tuning.
- **Note:** Requires a Gurobi license (free for academic use).

### 6.4 SCIP

**File:** `scip_interface.py`
**Class:** `SCIP_MILP_LP`

- Indicator constraints via `pyscipopt` `addConsIndicator`
- No native solution pool → falls back to iterative `compute_optimal`
- Seed via `setSeed`
- **Advantages:** Open-source, no license needed, handles large models well.
- **Limitation:** No populate support; POPULATE mode falls back to BEST.

### 6.5 GLPK

**File:** `glpk_interface.py`
**Class:** `GLPK_MILP_LP`

- **No indicator constraint support** — all indicator constraints are translated to big-M at construction time.
- Uses `swiglpk` (bundled with COBRApy).
- No solution pool → iterative enumeration only.
- **Advantages:** Always available without additional installation.
- **Limitation:** Poorest performance on large MILPs. Sensitive to M value choice. Default `M = 1000` (COBRApy bound threshold).

---

## 7. Computation Approaches — SDMILP

**File:** `strainDesignMILP.py`
**Class:** `SDMILP` (inherits from `SDProblem` and `MILP_LP`)

`SDMILP.__init__` calls `SDProblem.__init__` to build the MILP matrices, then `MILP_LP.__init__` to set up the solver backend.

### 7.1 ANY

**Method:** `SDMILP.compute(**kwargs)`

Finds feasible solutions without enforcing global optimality:
1. Solve MILP as-is (no cost-minimization objective, just feasibility).
2. Verify each solution.
3. Add exclusion constraint.
4. Repeat until `max_solutions` or infeasibility.

For MCS problems, `z_sum ≤ max_cost` and `z_sum ≥ 0` are the only cost constraints. Results may have suboptimal (higher) cardinality compared to BEST.

**Warning:** For OptKnock/RobustKnock, ANY may return the wild-type (z=0) immediately if that satisfies the outer objective. Use BEST for these methods.

### 7.2 BEST

**Method:** `SDMILP.compute_optimal(**kwargs)`

Finds globally optimal solutions:
1. Solve MILP with full objective (minimize cost for MCS; maximize outer for OptKnock, etc.).
2. Verify solution.
3. Add exclusion constraint (excludes solution and supersets for MCS; only the exact solution for OptKnock via `add_exclusion_constraints_ineq`).
4. Fix objective: `fixObjective(c_bu, opt)` — constrain next solution to have at least the same objective value.
5. Among solutions with equal objective, minimize intervention cost: `setMinIntvCostObjective()`.
6. Repeat until `max_solutions`, infeasibility, or time limit.

For MCS computations, the MILP first checks if the wild-type already satisfies all modules (via `verify_sd(sparse.csr_matrix((1, num_z)))`).

### 7.3 POPULATE

**Method:** `SDMILP.enumerate(**kwargs)`

Uses the solver's native solution pool (CPLEX/Gurobi):
1. Call `populateZ(max_solutions)` to generate a pool.
2. Verify all solutions in the pool.
3. Add exclusion constraints for invalid solutions.
4. Repeat.

Falls back to `compute_optimal` for SCIP and GLPK (which lack native pool support).

**Gurobi pool notes:** Pool quality/diversity can be tuned via Gurobi's `PoolSearchMode` and `PoolGap` parameters, but this is not currently exposed in the interface.

### 7.4 Exclusion Constraints

**Methods:** `add_exclusion_constraints(z)`, `add_exclusion_constraints_ineq(z)`

For a solution with binary vector `z`:

**Case 1 — z is all zeros (wild-type):**
Add `sum(z_i) ≥ 1` → forces at least one intervention next time. Implemented as `1 ≥ −1` (infeasible if all z = 0).

**Case 2 — z has exactly one non-zero entry (single intervention):**
Set `ub[idx] = 0` → permanently disable that intervention.

**Case 3 — z has multiple non-zero entries:**
Integer cut: `sum(z_i : i active) ≤ sum(z_i_current) − 1`
This excludes the exact solution and all its **supersets** (for MCS: if removing r1 and r2 works, we don't want r1+r2+r3 as a "new" solution since it's a non-minimal superset).

`add_exclusion_constraints_ineq` uses a different cut that excludes **only the exact solution**, not its supersets (used for OptKnock where we want all optima).

### 7.5 Solution Verification

**Method:** `SDMILP.verify_sd(sols)`

After each MILP solution, the binary vector `z` is verified by building and solving a **continuous LP** with the corresponding constraint/variable removal applied:

1. From `cont_MILP` (saved before `link_z`): the original LP matrices with z-map information.
2. For each z_i in the solution: determine which variables (`z_map_vars`) and constraints (`z_map_constr_ineq`, `z_map_constr_eq`) are deactivated.
3. Build the restricted LP: remove deactivated constraints/variables.
4. Solve via `slim_solve()`.
5. If `slim_solve()` returns `nan` → infeasible → solution is valid (for SUPPRESS) or invalid (for PROTECT).

This catches numerical artifacts where the MILP returns a slightly non-integer solution that passes rounding but does not actually satisfy the biological constraint.

**`cont_MILP` is the continuous pre-`link_z` version of the problem.** It preserves the structural relationship between z-variables and constraints/variables before big-M coefficients obscure it. This is why the continuous validation is numerically robust even if M values were imprecise.

---

## 8. Post-processing & Result Container

### 8.1 Solution Decompression

**Function:** `expand_sd(sd, cmp_mapReac)` in `networktools.py`

After solving on the compressed model, solutions refer to compressed reaction IDs. `expand_sd` maps each compressed intervention back to the set of original reactions it represents (using `cmp_mapReac`).

- A single compressed KO may expand to multiple original reaction KOs (coupled/parallel reactions).
- The expanded solution preserves the cost structure.

**Function:** `filter_sd_maxcost(sd, max_cost, uncmp_ko_cost, uncmp_ki_cost)`
After expansion, the cost of a solution in original space may differ from compressed space (due to parallel reactions with non-unit costs). Solutions exceeding `max_cost` in original space are filtered out.

### 8.2 GPR Translation

**SDSolutions** translates gene interventions to reaction phenotypes using `gpr_eval`.

For each gene-level solution `gene_sd`:
1. Collect all affected reactions from the model's GPR rules.
2. Evaluate each reaction's GPR with the gene states using `gpr_eval`.
3. Reactions whose GPR evaluates to `False` (given knockouts) are collected as reaction knockouts.
4. Reactions requiring knocked-in genes that are `True` (given additions) are collected as reaction additions.

`get_gene_reac_sd_assoc()` returns three parallel lists:
- `reac_sd`: reaction-level intervention dicts
- `assoc`: association dict showing which genes caused which reaction change
- `gene_sd`: the original gene interventions

### 8.3 SDSolutions

**File:** `strainDesignSolutions.py`

`SDSolutions` stores:
- `reaction_sd`: list of dicts `{reac_id: -1 | 0 | +1}` for each solution
- `gene_sd`: list of gene-level dicts (if gene interventions were used)
- `status`: solver status string
- `sd_setup`: full setup dict (model_id, modules, costs, solver, etc.) — can be serialized to JSON and reloaded

Serialization: `sd_setup` is JSON-compatible, allowing CNApy to save `.sd` files that can be reloaded into `compute_strain_designs` via the `sd_setup` dict parameter.

---

## 9. Constraint & Expression Parsing

**File:** `parse_constr.py`

Handles user-supplied constraints and objectives in multiple formats:

**Input formats (all equivalent):**
```python
# String
"r1 + 3*r2 = 0.3, -5*r3 - r4 <= -0.5"

# List of strings
["r1 + 3*r2 = 0.3", "-5*r3 - r4 <= -0.5"]

# List of structured constraints
[[{'r1': 1.0, 'r2': 3.0}, '=', 0.3], [{'r3': -5.0, 'r4': -1.0}, '<=', -0.5]]
```

**Key functions:**

| Function | Input → Output |
|---|---|
| `parse_constraints(constr, reaction_ids)` | Any format → `[[dict, sense, rhs], ...]` |
| `parse_linexpr(expr, reaction_ids)` | Any format → `[dict, ...]` |
| `lineq2mat(equations, reaction_ids)` | String list → `(A_ineq, b_ineq, A_eq, b_eq)` sparse matrices |
| `lineqlist2mat(eqlist, reaction_ids)` | Structured list → sparse matrices |
| `linexpr2dict(expr, reaction_ids)` | String → `{reac_id: coeff}` |
| `linexprdict2mat(expr_dict, reaction_ids)` | Dict → sparse row vector |

The `parse_constraints` function uses regex splitting on `','` and `'\n'` to separate multiple constraints in a single string, then dispatches to `lineq2list` which uses further regex for coefficient/identifier/operator extraction.

---

## 10. Supporting Utilities

### `lptools.py`
- **`fva(model, variables, constraints, solver)`**: Flux variability analysis. Parallelized via `SDPool` using `fva_worker_init`/`fva_worker_compute`. Returns a `pandas.DataFrame` with min/max flux values.
- **`select_solver(solver, model)`**: Selects solver based on availability, preference, and model settings. Priority: CPLEX > Gurobi > SCIP > GLPK.
- **`plot_flux_space(model, ...)`**: 2D/3D convex hull visualization using `scipy.spatial.ConvexHull` and Matplotlib.
- **`DisableLogger`**: Context manager to suppress logging during FVA.

### `networktools.py` — LP Suppression

#### Problem

COBRApy's `Model` is tightly coupled to its solver backend (optlang). Every mutation — setting a reaction's bounds, renaming a reaction, adding/removing metabolites — immediately updates the solver's LP representation via `set_bounds`, `set_linear_coefficients`, `_populate_solver`, etc. During compression and preprocessing, hundreds of such mutations occur on a model copy. Synchronizing each one to the solver is pure overhead: the LP is rebuilt from scratch at the end anyway.

Early approaches patched optlang classes (`Constraint.set_linear_coefficients`, `Variable.set_bounds`, `Objective.set_linear_coefficients`) at the class level during suppression. This worked for optlang-level calls but missed cobra-level code paths: `Reaction.add_metabolites` accesses `self.forward_variable`, `self.reverse_variable`, and `model.constraints[met.id]` to build solver constraint updates. These paths reached the solver through cobra internals, not optlang's public API.

#### Architecture: class-level cobra patching

`suppress_lp_context(model)` is a context manager that patches **both optlang and cobra classes** at the class level for the duration of the block. All solver-bound operations become no-ops, and cobra methods that would access the solver are redirected to safe stubs.

**Patches applied by `_suppress_lp_updates`:**

| Target | Original | Suppressed replacement |
|---|---|---|
| `Reaction._set_id_with_model` | Renames solver variable | `_suppressed_set_id`: writes `_id` + updates `DictList._dict` index only |
| `Reaction.lower_bound` (setter) | Calls `update_variable_bounds` → solver | `_suppressed_set_lb`: writes `_lower_bound` only |
| `Reaction.upper_bound` (setter) | Calls `update_variable_bounds` → solver | `_suppressed_set_ub`: writes `_upper_bound` only |
| `Reaction.update_variable_bounds` | Accesses `forward_variable.set_bounds(...)` | No-op |
| `Model._populate_solver` | Full solver rebuild | No-op |
| `Model.remove_reactions` | Removes solver variables/constraints | `_suppressed_remove_reactions`: removes from model lists only |
| `Model.remove_metabolites` | Removes solver constraints | `_suppressed_remove_metabolites`: removes from model lists only |
| `Container.__getitem__` | Raises `KeyError` for missing keys | `_permissive_container_getitem`: returns `_SolverStub` for missing keys |
| optlang `Constraint.set_linear_coefficients` | Updates solver constraint | No-op |
| optlang `Variable.set_bounds` | Updates solver variable bounds | No-op |
| optlang `Objective.set_linear_coefficients` | Updates solver objective | No-op |

**`_SolverStub`** is a lightweight null object returned by the permissive `Container.__getitem__` when cobra code tries to access a solver variable/constraint that doesn't exist (because `_populate_solver` is suppressed). It is hashable (used as a dict key in `set_linear_coefficients` calls) and has no-op `set_bounds`/`set_linear_coefficients` methods. This single class eliminates the need to patch every individual cobra method that might touch the solver.

**`_suppressed_set_id` and DictList invariant:** cobra's `DictList._dict` maps `id → int index` (not `id → object`). The suppressed setter must maintain this invariant: it pops the old key and inserts the new key with the same integer index. Violating this causes `ValueError` in `DictList.__setitem__` due to its duplicate-check logic.

#### Objective preservation through compression

When `suppress_lp_context` enters, it captures the model's objective coefficients into `model._suppressed_obj` (a `dict` mapping `reaction_id → float coefficient`). This dict is updated by compression code as reactions are merged:

- **Coupled compression** (`compression.py`, `_apply_compression_to_model`): when reactions are merged, their objective contributions are weighted by compression factors and summed into the surviving reaction's entry.
- **Parallel compression** (`compression.py`, `compress_model_parallel`): when identical-stoichiometry reactions are merged, their objective contributions are summed directly.
- **Legacy Java compression** (`efmtool_cmp_interface.py`, `compress_model_java`): same logic using `jBigFraction2sympyRat` conversion factors.

On exit, if the model was modified (reaction IDs changed), the context manager rebuilds the solver from the model's current state and restores the objective from `_suppressed_obj`. If the model was not modified (e.g., `compute_strain_designs` only modifies a copy), no rebuild occurs — this is critical because an unconditional rebuild would produce a solver with the wrong objective direction for methods like RobustKnock that have inner/outer objectives.

```python
# Simplified exit logic
current_ids = {r.id for r in model.reactions}
if current_ids != pre_ids:
    model._solver = solver_interface.Model()
    model._populate_solver(model.reactions, model.metabolites)
    for rxn in model.reactions:
        c = final_obj.get(rxn.id, 0.0)
        if c != 0:
            rxn.objective_coefficient = c
```

#### Design principles

1. **All plumbing in the context manager.** Compression code, GPR extension, and other preprocessing functions are backend-agnostic — they manipulate the cobra model without knowing whether LP updates are suppressed. The context manager handles all solver synchronization.

2. **Class-level patching, not instance-level.** Patching at the class level (e.g., `Reaction.lower_bound = property(fget, suppressed_fset)`) ensures that ALL reactions are covered, including ones created during compression. Instance-level patches would miss newly created objects.

3. **Conditional rebuild.** The solver is only rebuilt when the model was actually modified. This prevents regressions in methods like RobustKnock/OptKnock/OptCouple that have complex objective structures.

#### Performance impact

On iML1515, LP suppression reduces preprocessing time from ~131s to ~70s (46% reduction). The bulk of the savings comes from avoiding ~2000 `set_linear_coefficients` calls during compression, each of which would otherwise rebuild solver constraint data structures.

### `pool.py` — SDPool
Windows-compatible `multiprocessing.Pool` subclass. On Windows, the initializer function is pickled to a temporary file and loaded by worker processes, avoiding a performance regression from passing the initializer directly (COBRApy issue #997). Uses `'spawn'` context universally to avoid `fork`-related instability.

### `names.py`
All string constants. Importing with `from straindesign.names import *` is used throughout the codebase. Constants include module type strings, solver names, status codes, and parameter keys. This single-file approach ensures consistent naming.

**Notable:** `PROTECT = 'mcs_lin'` and `SUPPRESS = 'mcs_bilvl'` are defined first (legacy names), then immediately overwritten with `'protect'` and `'suppress'`. This is vestigial code — the legacy names are not used and could be removed.

### `__init__.py`
Detects available solvers at import time:
```python
avail_solvers = set()
try: import cplex; avail_solvers.add('cplex')
try: import gurobipy; avail_solvers.add('gurobi')
try: import pyscipopt; avail_solvers.add('scip')
try: import swiglpk; avail_solvers.add('glpk')
```
Exports: all public classes, functions, and the `avail_solvers` set.

Also calls `_start_jvm()` from `efmtool_cmp_interface.py` — see next section.

### `efmtool_cmp_interface.py` — JPype/JVM Initialization

**JPype and Java are optional dependencies.** StrainDesign works without them; the
default compression backend is `sparse_rref` (pure Python). The efmtool Java backend
(`compression_backend='efmtool_rref'`) requires `jpype1` and a JVM.

#### Conditional eager JVM startup

`__init__.py` calls `_start_jvm()` at package import time. This function:

1. Checks if `jpype1` is installed (via `find_spec`). If not → immediate return, no-op.
2. Adds `efmtool.jar` to the classpath.
3. Starts the JVM with `jpype.startJVM("--enable-native-access=ALL-UNNAMED")`.
4. Loads all Java classes via `import jpype.imports` + Python import statements.
5. Registers `atexit` handler for clean JVM shutdown.

If any step fails (no Java installed, jar missing, etc.), the function returns silently.
Users who never use `efmtool_rref` are unaffected. When `_init_java()` is later called
(on first efmtool use), it checks `_JAVA_INITIALIZED` and falls back to `JClass()` loading.

#### Why eager, not lazy?

This design is critical for stability. In v1.14, `from .efmtool import *` in `__init__.py`
started the JVM at import time. When we switched to lazy initialization (JVM started on
first efmtool use), JPype began crashing with SIGBUS/SIGSEGV on Linux and macOS CI
runners when processing larger matrices (iMLcore, 586 reactions).

Root causes identified:
- **JVM + OpenBLAS pthread conflict** ([jpype#808](https://github.com/jpype-project/jpype/issues/808)):
  The JVM modifies pthread stack allocation. If started after NumPy/OpenBLAS has already
  spawned worker threads, subsequent JNI calls can trigger SIGSEGV.
- **GC finalization race** ([jpype#934](https://github.com/jpype-project/jpype/issues/934)):
  Python's garbage collector can attempt to finalize JPype proxy objects during a JNI
  call, causing Bus error. Mitigated with `gc.disable()`/`gc.enable()` around heavy
  Java calls in `basic_columns_rat_java` and `compress_model_java`.
- **`import jpype.imports` vs `JClass()`**: The Python import-style class loading
  (`import ch.javasoft...`) sets up JNI references differently from `JClass()`. The
  import style (used in v1.14) is more stable on CI runners.

#### CI considerations

Java 21 (Temurin) is used on all CI platforms. The `--enable-native-access=ALL-UNNAMED`
flag suppresses Java 17+ warnings about JPype's `System.load()` calls. Despite all
mitigations, JPype JNI crashes remain non-deterministic at ~1-in-20 frequency on some
Linux runners ([jpype#934](https://github.com/jpype-project/jpype/issues/934)). The
iMLcore efmtool parity tests are marked `--large` (skipped on CI) since the e_coli_core
tests exercise the same code path on a smaller, more reliable matrix size.

---

## 11. Known Issues, Urgent Actions & Future Work

### 11.1 Urgent Action Items

**⚠ Farkas special case (homogeneous equality):**
In `farkas_dualize`, there is a documented comment noting the unimplemented edge case:
> "In the case of (1) A x = b, (2) x ∈ ℝ, (3) b ≈ 0, Farkas' lemma is special, because `b'y ≠ 0` is required..."

This could cause silent incorrect results for SUPPRESS modules with equality constraints where `b ≈ 0` (e.g., free reactions with zero RHS). Impact: depends on whether the compressed, preprocessed models trigger this case. A defensive check + explicit splitting into two inequalities would fix it.

**⚠ Big-M for GLPK with large flux ranges:**
GLPK uses `M = cobra_bound_threshold = 1000` by default. For models with non-standard bounds (e.g., ATPM = 8.39, but oxygen intake up to 10,000), this can produce incorrect strain designs. Models should always have `remove_dummy_bounds` applied, but verifying this is enforced is important.

**⚠ `add_exclusion_constraints_ineq` has a bug:**
```python
A_ineq = [1.0 if z[j, i] else -1.0 for i in self.idx_z]
A_ineq.resize((1, self.A_ineq.shape[1]))  # ← list.resize doesn't exist
```
`list.resize` is a NumPy array method, not a list method. This would raise an `AttributeError` at runtime. This method is not used in the current main computation paths (OptKnock uses `add_exclusion_constraints`), but should be fixed. The fix: convert to `np.array` before calling `.resize`.

**⚠ Legacy name double-assignment in `names.py`:**
```python
PROTECT = 'mcs_lin'
SUPPRESS = 'mcs_bilvl'
PROTECT = 'protect'      # overwrites silently
SUPPRESS = 'suppress'    # overwrites silently
```
The first assignments are dead code and should be removed to avoid confusion.

### 11.2 Low-Hanging Fruits

**Solution space plotting issues (reported):**
`lptools.plot_flux_space` uses `scipy.spatial.ConvexHull` and `Matplotlib`. Known issues include:
- 3D plots with degenerate hulls (lower-dimensional polytopes) cause `QhullError`.
- Color scaling in `plot_flux_space_colored` may fail for single-point solutions.
- Fix: add degeneracy checks and fallback to scatter plots for ≤ 2 unique points.

**Populate mode for SCIP/GLPK:**
Currently silently falls back to BEST. Should log a clear warning and document that SCIP/GLPK do not support POPULATE.

**Gurobi pool tuning:**
The `populate` method in `gurobi_interface.py` could expose `PoolSearchMode=2` (focused enumeration) for better solution diversity. Currently uses default settings.

**RobustKnock + POPULATE:**
RobustKnock with POPULATE is not well-tested. The three-level dual structure is particularly sensitive to solver pool heuristics. Add explicit test cases.

**Parallel FVA compression for large models:**
For genome-scale models (> 2000 reactions), compression can be slow. The RREF step in `RationalMatrix` has `O(n²)` row operations. Potential improvement: use block-sparse RREF or exploit model structure (e.g., subsystems).

**FVA Phase 2 parallelization for non-Gurobi solvers:**
Gurobi releases the GIL during `model.optimize()`, enabling `ThreadPoolExecutor`-based parallelism with batched warm-start (4.3x speedup measured on iML1515). CPLEX, GLPK, and SCIP hold the GIL, so they must use process-based `SDPool` with its ~4-6s spawn overhead. Investigating ctypes/Cython wrappers that release the GIL for these solvers could enable thread-based parallelism across all backends.

**Essential KI detection:**
`essential_kis` is computed but may not always be passed correctly through the compression/GPR pipeline. Verify that the set persists through `compress_ki_ko_cost`.

**Opposite-direction parallel merging:**
Two irreversible reactions with opposite stoichiometry and complementary bounds (e.g., `A→B (0,inf)` and `B→A (0,inf)`) could be merged into one reversible reaction `(-inf,inf)`. This would help eliminate flux cycles. However, a computed strain design might need to knock out one direction specifically (the other being essential), which limits the merger to non-knockable reactions. Few reactions are essential in both directions, so the practical benefit is small.

### 11.3 Future Development Directions

**Nullspace-based primal/dual formalization:**
The referenced paper (DOI: 10.1093/bioinformatics/btz393, von Kamp & Klamt 2019) presents a nullspace-based approach to computing minimal cut sets. Integrating this formalization could:
- Avoid explicit dualization for certain model types.
- Give a more compact MILP representation.
- Improve numerical conditioning.

A key idea: the primal flux space has a right nullspace (internal cycles). The Farkas dual operates in the left nullspace. Exploiting this duality directly, rather than constructing full dual matrices, could reduce MILP size significantly.

**Variable splitting:**
Reversible reactions split into forward/reverse can sometimes improve MILP performance (tighter LP relaxation). This is a pre-processing step that should be optional and tested for impact on different model types.

**On the mathematical basis of big-M values (investigated, not pursued):**

This section documents a thorough investigation into computing theoretically optimal
big-M values for the Farkas dual MILP. The investigation confirmed the mathematical
foundations but revealed that the problem is inherently as hard as MCS enumeration
itself. No implementation was made; this section exists to prevent re-derivation.

*Where M comes from — the primal view:*

In the Farkas dual formulation (section 5.3), each knockable constraint `i` is linked
to a binary z_i via either big-M (`A[i]x - M_i*z_i ≤ b[i]`) or an indicator constraint
(`z_i = 0 → A[i]x ≤ b[i]`). The big-M value M_i must be large enough that when z_i = 1,
the constraint `A[i]x ≤ b[i] + M_i` is never binding — i.e., M_i ≥ max{A[i]x - b[i]}
over all feasible x in every MCS that knocks out constraint i.

For SUPPRESS modules, the Farkas dual constraints encode the *primal* flux space
through duality. Each knockable dual constraint corresponds to a primal reaction. The
M value for reaction j's dual constraint is determined by the *primal* flux behavior:

> **M_j = 1 / min|v_j|** taken over all Elementary Flux Vectors (EFVs) in which
> reaction j participates.

Here, an EFV is a support-minimal feasible flux distribution satisfying the target
constraint. MCS are the minimal hitting sets of EFV supports. For reaction j to appear
in any MCS of cardinality > 1, there must exist an EFV where j carries flux — and the
*smallest* such flux across all EFVs determines the tightest valid M.

*Why single-KO M is trivial:*

For reactions that are individually essential (size-1 MCS candidates), M_j = 1/FVA_min(|v_j|).
This is a single LP: minimize |v_j| subject to `Sv = 0`, bounds, and the target
constraint. This is already computed during preprocessing FVA and is the basis for
the current M-bounding in step 3 of `link_z`.

*Why multi-KO M is hard:*

For reactions that only appear in MCS of size ≥ 2, M_j depends on the *combinatorial
structure* of which other reactions are simultaneously knocked out. The minimum |v_j|
is not taken over the full flux space (where j can always carry large flux via
alternative pathways) but over the restricted space where a specific set of other
reactions are forced to zero — and j must be essential in that restricted space.

Formally, computing the optimal M_j requires solving:

```
M_j = max over all MCS K containing j of:
        1 / min{ |v_j| : Sv = 0, lb ≤ v ≤ ub, v_target ≥ t,
                          v_i = 0 for i in K\{j} }
```

This is a max-min over an exponential number of MCS. Even computing the inner
minimization for a *single* known MCS K is just an LP, but enumerating which K to
consider requires MCS enumeration — the very problem M values are meant to help solve.

*Combined primal/Farkas MILP — formulation and results:*

We attempted to bypass MCS enumeration by embedding the combinatorial search in a
single MILP. The formulation combines primal feasibility with a Farkas certificate:

- **Primal:** `S*v = 0`, `lb_i*(1-k_i) ≤ v_i ≤ ub_i*(1-k_i)` (big-M knockout),
  `v_target ≥ target_lb`. Binary k_i = 1 knocks out reaction i.
- **Essential participation:** Sign binary forces `v_j ≥ δ` or `v_j ≤ -δ`.
- **Farkas certificate:** Same k_i binaries control dual constraints via indicators.
  When k_i = 0 (active): `S[:,i]'u - r_i + s_i = δ_{i=target}`.
  When k_i = 1 (knocked): `r_i = s_i = 0`.
  Farkas sum: `Σ(ub_i*s_i - lb_i*r_i) ≤ target_lb - ε`.
- **Objective:** minimize |v_j| = t, giving M_j = 1/t.

The Farkas certificate proves that removing j from the active set (while keeping the
knocked reactions knocked) makes the system infeasible — i.e., j is essential in the
knockout context defined by the k_i values. The optimizer simultaneously searches for
the knockout set and the flux distribution where j's participation is minimized.

*Practical obstacle — primal leakage:*

The big-M knockout constraints `v_i ≤ ub_i*(1-k_i)` allow leakage when k_i is
within solver tolerance of 1 but not exactly 1. With Gurobi's default IntFeasTol = 10⁻⁵,
a variable with k_i = 0.9999963 (accepted as integer 1) gives leakage
v_i ≤ ub_i × 3.7×10⁻⁶. With default bounds ±1000, this is 0.0037 per reaction.
Across 15+ knocked reactions, the cumulative leakage invalidates the primal solution
entirely — the model appears feasible when it should be infeasible.

This was diagnosed by inspecting k/v values at full precision (`tests/check_k_precision.py`):
PIt2r showed k = 0.9999963, v = 0.003679, meaning the "knocked" reaction carried
measurable flux that kept the system alive.

*Mitigation — FVA-tight bounds + solver tolerances:*

Two measures reduce leakage to negligible levels:
1. FVA-derived bounds (replace ±1000 with actual flux ranges, typically O(1)–O(10))
2. Tight solver tolerances (IntFeasTol = 10⁻⁹, FeasibilityTol = 10⁻⁹, NumericFocus = 3)

With both, per-reaction leakage drops to O(bound × 10⁻⁹) ≈ 10⁻⁸.

*Validation on E. coli core (95 reactions, 353 MCS up to size 3):*

Ground truth M values were computed independently: enumerate all MCS via indicator-based
MILP (no big-M), then for each MCS, solve one LP per knocked reaction to find the
minimum |v_j| in the Farkas certificate. M_gt[j] = max over all MCS containing j.
(See `tests/ground_truth_M.py`.)

With tight tolerances and FVA bounds, the combined MILP matched ground truth exactly
for 6 of 15 tested reactions, spanning 4 orders of magnitude:

| Reaction   | M_gt        | M_milp      | Ratio  | #KO | Time  |
|------------|-------------|-------------|--------|-----|-------|
| TALA       | 413,793     | 413,793     | 1.0000 | 15  | ~120s |
| TKT1       | 413,793     | 413,818     | 1.0001 | 16  | ~120s |
| NADH16     | 68,966      | 68,966      | 1.0000 | 6   | ~35s  |
| EX_etoh_e  | 68,966      | 68,966      | 1.0000 | 23  | ~120s |
| ETOHt2r    | 68,966      | 68,966      | 1.0000 | 9   | ~120s |
| ATPM       | 0.12        | 0.12        | 1.0000 | 36  | 0.8s  |

Reactions with intermediate M values (PFL: 329, NADTRHD: 0.91) did not converge within
120s, returning inflated values. The combinatorial search space is too large for the
solver to explore efficiently.

*Why we did not pursue this further:*

The investigation confirmed that the formulation is mathematically correct — it produces
exact M values when the solver converges. However, it also confirmed the fundamental
insight: **computing the optimal M_j for multi-reaction MCS is a combinatorial problem
of the same complexity as MCS enumeration itself.** There is no shortcut:

1. For size-1 MCS: M = 1/FVA_min — already computed cheaply via LP.
2. For size-k MCS (k ≥ 2): the optimal M depends on which *other* reactions are
   simultaneously knocked out, requiring exploration of the space of knockout
   combinations. This is exactly what the MCS MILP does.
3. The "FVA over the full Farkas polyhedron" approach (computing bounds with all
   constraints active) gives a *lower bound* on M — it tells you the range of dual
   variables when no constraints are relaxed. But in an actual MCS, some constraints
   ARE relaxed (z = 1), and the remaining dual variables may need larger values to
   construct a valid Farkas certificate. The FVA approach is a natural progression
   from the single-KO case but does not capture the combinatorial interaction that
   determines the true M for multi-reaction MCS.

In summary: the relationship M_j = 1/min|v_j| over EFVs containing j is exact but
not efficiently computable. Any finite M chosen without enumerating MCS risks being
too small (cutting off valid MCS) or too large (degrading solver performance).
Indicator constraints avoid the M problem entirely at the cost of solver performance
(lazy enforcement, weaker LP relaxation). This is the fundamental trade-off; there
is no free lunch.

*Investigation scripts were removed after documenting findings. The formulations
above can be reconstructed from the mathematical descriptions if needed.*

**Reaction scaling for better-conditioned M values:**
Trace metabolites (e.g., MoYB at 7×10⁻⁷ mmol/gDW/h) create small inhomogeneous bounds in the primal, which after dualization produce small coefficients in the Farkas objective, forcing dual variables to be very large to satisfy `b^T u ≤ -1`. This leads to M values spanning many orders of magnitude (irreducible condition number ∝ κ(S)), harming LP relaxation quality.

Column scaling (per reaction) addresses this: multiply flux variable `v_j` by a power-of-10 factor `α_j` so that inhomogeneous bounds move into a reasonable range (e.g., [0.1, 10]):
- Stoichiometry: `S[:, j] /= α_j` (coefficients shrink)
- Bounds: `lb_j *= α_j`, `ub_j *= α_j` (bounds grow)
- Mass balance: `(S[:,j]/α_j) × (α_j × v_j) = S[:,j] × v_j` (unchanged)

Row scaling (per metabolite) rebalances after column scaling: multiply metabolite row `i` by `β_i` (power of 10) to bring the max coefficient back to O(1). This does not change the solution space (just scales the mass balance equation).

Implementation would go between `build_primal_from_cbm` and dualization in `addModule`. The scaling factors must be tracked for unscaling continuous solution values in non-MCS problem types (OptKnock, etc.). For MCS, only binary z-variables are returned, so unscaling is not needed.

The FVA bounds from `bound_blocked_or_irrevers_fva` (already computed in preprocessing) provide the tight flux ranges needed to determine `α_j`. A per-module FVA considering module constraints could provide even tighter ranges.

**Size-1 MCS via FVA on SUPPRESS modules:**
The current preprocessing (line 346–356 in `compute_strain_designs.py`) runs FVA on PROTECT (and OPTKNOCK etc.) modules to find essential reactions — reactions that must carry flux in the protected region — and marks them non-knockable. SUPPRESS modules are explicitly skipped (`if m[MODULE_TYPE] != SUPPRESS`).

The idea: also run FVA on SUPPRESS modules (without inner objective) to identify reactions essential for the target (undesired) region. Combining the two FVAs yields size-1 MCS cheaply and further reduces the knockable reaction set:

*Step 1 — FVA on PROTECT (already implemented):*
Reactions where `min(|flux|) > 0` under PROTECT constraints are essential for the protected region. These are already marked non-knockable.

*Step 2 — FVA on SUPPRESS (new):*
Reactions where `min(|flux|) > 0` under SUPPRESS constraints are essential for the target region. Knocking out any single one of these makes the target infeasible — each is a size-1 MCS candidate.

*Step 3 — Cross-check:*
Size-1 candidates from step 2 that are NOT in the essential set from step 1 are true size-1 MCS: they disrupt the target without being required for the protected region. Add them to the output.

*Step 4 — Mark non-knockable:*
All reactions identified in steps 1 and 2 can be removed from the knockable set for the main MILP:
- Step 1 essentials: knocking them out violates protection (already handled).
- Step 2 essentials not in step 1: they are size-1 MCS, already found. Since any larger MCS containing a size-1 MCS is not minimal, these reactions cannot participate in any new MCS. Safe to exclude.
- Step 2 essentials also in step 1: essential for both regions, obviously non-knockable.

*Benefits:*
- Size-1 MCS are found for free (just FVA, no MILP). This is the cheapest possible way to discover them.
- Fewer knockable reactions in the main MILP → fewer z-variables → smaller B&B tree.
- Reactions with tiny flux bounds (e.g., trace metabolite carriers at 1e-7) that are essential in the SUPPRESS region get excluded early, improving the conditioning of the main MILP's M-value computation.
- The model may simplify further (fewer knockables may enable additional compression passes or remove degenerate constraints), though the extent of this benefit is unclear.

*Caveats:*
- Only finds size-1 MCS. Size ≥ 2 MCS still require the full MILP.
- The inner objective (if present in the SUPPRESS module) is ignored for this FVA. This is correct for standard MCS but may miss reactions that are only essential under the bilevel structure.
- Adds O(2n) LP solves per SUPPRESS module (one FVA). For compressed models this is fast.

**MILP warm-starting:**
When BEST generates multiple solutions iteratively, each call starts cold. Warm-starting from the previous solution's LP relaxation (available in CPLEX/Gurobi) could speed up subsequent solves significantly.

**Improving RobustKnock numerics:**
The three-level dual construction in ROBUSTKNOCK creates very large matrices with potential for numerical blow-up. Applying scaling (row/column normalization) to the LP before and after dualization could help. Alternatively, use iterative refinement at the inner LP level.

**SOS1/SOS2 constraints for bilevel formulations:**
Gurobi, CPLEX, and SCIP support SOS1 (Special Ordered Sets type 1) and SOS2 constraints natively. These could be useful for connecting variables of primal and KKT parts of multi-level optimization problems, encoding complementarity conditions (primal-dual coupling) more efficiently, or as an alternative to big-M for disjunctive constraints in bilevel formulations. SOS1 constraints enforce that at most one variable in a set is nonzero, which is exactly the complementarity condition in KKT-based reformulations.

**Sparse matrix format conversions:**
The computation pipeline contains many sparse matrix format conversions (`tocsr`, `tocsc`, `todok`, `tocoo`) that may not all be necessary. A profiling pass could identify redundant conversions. Key areas: `SDProblem.__init__` performs multiple `tocsc`/`tocsr` conversions during `link_z`; `addModule` builds matrices in one format then converts for block assembly; `compression.py` converts between formats during RREF and coupled-reaction detection. Eliminating unnecessary conversions could reduce preprocessing time, especially for large models.

**Integration of regulatory networks beyond simple T/F interventions:**
The current regulatory intervention model (active/inactive) is binary. Future work could integrate thermodynamic constraints or kinetic feasibility checks.

**SCIP native solution enumeration:**
SCIP supports solution enumeration beyond the incidental pool (`limits/maxsol`, default 100).
Since version 2.0, the `cons_countsols` constraint handler can count and enumerate all
feasible solutions of a constraint integer program via the `count` command. Key details:

- `constraints/countsols/collect = TRUE` stores detected solutions (default FALSE = count only).
- To enumerate all *optimal* solutions: solve to optimality to get `c*`, add the objective
  as a constraint with both bounds equal to `c*`, then run `count`.
- Restarts must be turned off during counting (use `SCIPsetParamsCountsols()`).
- SCIP uses unrestricted subtree detection, which can detect several solutions at once,
  so a soft solution limit may be exceeded before SCIP stops.
- Collected solutions are stored with respect to active (non-presolved) variables only;
  they are lifted back into the original variable space when written to file.
- This is *not* equivalent to Gurobi's ranked near-optimal pool (`PoolSearchMode`).
  SCIP enumerates feasibility/optimality, not a ranked set of near-optimal solutions.

Currently, StrainDesign's SCIP `populate` method uses a custom workaround
(iterative solve with solution exclusion constraints). Replacing this with
native `cons_countsols` enumeration could improve performance and correctness.
However, pyscipopt does not yet expose the `count` command or `cons_countsols`
parameters — this would require upstream pyscipopt changes or direct C API calls.

**pyscipopt gaps for LP method and basis control:**
Several SoPlex and SCIP features that would benefit StrainDesign are not exposed
through the pyscipopt Python bindings:

- *LP basis get/set:* SCIP internally maintains LP bases via SoPlex, but
  pyscipopt does not expose `SCIPlpGetBasisInd`, `SCIPgetLPBasisInd`, or
  the SoPlex `getBasis`/`setBasis` methods. This prevents explicit basis
  warm-starting for LPs solved through the SCIP/SoPlex interface.
- *LP algorithm selection for SCIP_LP (pure LP via `pyscipopt.LP`):*
  The `pyscipopt.LP` class wraps SoPlex directly but does not expose
  SoPlex's `setIntParam(ALGORITHM, ...)` for selecting primal vs dual
  simplex. The SCIP_MILP path supports this via `lp/initalgorithm` and
  `lp/resolvealgorithm` parameters, but the direct SoPlex LP wrapper does not.
- *Solution enumeration:* The `count` command and `cons_countsols` constraint
  handler (see above) are not accessible from pyscipopt.

These gaps are candidates for upstream pyscipopt issues or contributions.

**Improved CNApy integration:**
The `sd_setup` dict format is designed for CNApy interoperability. Ensuring full round-trip serialization (JSON → `compute_strain_designs` → JSON) with all parameter types (modules, costs, constraints) is an ongoing compatibility concern.

---

## 12. MILP Performance Notes

The following design choices have a measurable impact on MILP enumeration
performance. They are listed roughly by impact, based on benchmarking with
e_coli_core (95 rxns, 353 MCS) and iML1515 (1920 compressed rxns, 393 MCS).

**Unbounded reactions save constraints and variables.**
Reactions with infinite bounds (`-inf` / `inf`) do not generate big-M or
indicator constraints because no finite bound needs enforcement. Calling
`remove_dummy_bounds` early (replacing ±1000 with ±inf) is therefore not
just cosmetic — it directly reduces MILP size. Tightening bounds to FVA
ranges has the opposite effect: more finite bounds means more constraints.

**Omit non-knockable z-variables.**
The SD formulation creates one binary z per reaction, but only knockable
reactions (non-zero cost, non-essential) actually need one. Non-knockable
z-variables have ub=0 and are fixed to zero; with solver presolve disabled
they become dead weight. `_trim_z_variables` removes them before the solver
sees the problem. On iML1515: 462 → 205 binaries after trimming.

**Aggressive compression.**
Two-pass network compression (before and after GPR extension) reduces the
stoichiometric matrix substantially. On iML1515: 2712 → 1920 reactions,
1728 → 974 metabolites. Every removed reaction is one fewer z-variable,
one fewer column in all constraint matrices, and fewer indicator constraints.

**GPR propagation through compression.**
Merging GPR rules during compression (via `propagate_gpr=True`) and only
running `extend_model_gpr` on the compressed model avoids creating gene
pseudoreactions for reactions that were already merged away. This reduces
both preprocessing time (FVA on ~190 vs ~2700 reactions) and final model
size.

**Condensed formulation (no reaction splitting).**
The SD "condensed" formulation handles reversible reactions without
splitting them into forward/reverse pairs. This produces fewer z-variables
and fewer indicator constraints than the "split" variant. Benchmarks show
condensed is consistently faster (4.0s vs 4.1s on iMLcore, but the gap
widens on larger models).

**Auxiliary-variable indicators do not help.**
Replacing SD's complex indicators (`z=0 → A·x = b`) with FLB-style
auxiliary variables (`A·x + s = b` always, `z=0 → s = 0`) was tested.
The extra continuous variables and equality constraints outweigh the
benefit of simpler indicator propagation (~15% slower on iMLcore).

**Solver settings matter.**
Gurobi's presolve is critical for indicator-constraint MILPs. With
`Presolve=0` (which was temporarily needed for a Gurobi 13 bug),
solve times were ~1.6× slower across all formulations. On iML1515 the
full pipeline went from ~12 minutes (presolve off) to ~7 minutes
(presolve on).

---

## 13. Testing

Tests are in `tests/`. They are run with:
```bash
pytest tests -v --log-cli-level=INFO --junit-xml=test-results.xml
```

**CI matrix** (`.github/workflows/CI-test.yml`):
- OS: `ubuntu-latest`, `windows-latest`
- Python: `3.10`, `3.11`, `3.12`, `3.13`
- Package managers: `pip`, `conda`
- CPLEX excluded for Python 3.13 (max supported: 3.12)
- Segfault on Ubuntu (JPype JVM shutdown) handled via JUnit XML exit-code check

**Key test areas to verify after changes:**
1. Constraint parsing — any change to `parse_constr.py` must be verified against all input formats.
2. MILP construction — after changes to `link_z`, `build_primal_from_cbm`, or dualization functions: run with a small toy model and verify solutions against known correct answers.
3. Compression — changes to `compression.py` or `networktools.py` must verify that `expand_sd` correctly reconstructs original-space solutions.
4. Solver backends — changes to any `*_interface.py` file require testing with the specific solver installed.
5. GPR translation — changes to `networktools.extend_model_gpr` or `SDSolutions.gpr_eval` must be tested with models that have AND/OR GPR logic (e.g., iJO1366 for *E. coli*).

**Adding new tests:**
- Use a small toy model (3-5 reactions) for unit tests of MILP construction.
- Use `e_coli_core` for integration tests of the full pipeline.
- Use `iJO1366` for performance regression tests (track solve time).
- Always test with at least GLPK (always available) and one commercial solver if possible.

**Performance profiling:**
```python
import cProfile
cProfile.run("compute_strain_designs(model, sd_modules=[...], solver='glpk')", 'profile_out')
import pstats
pstats.Stats('profile_out').sort_stats('cumulative').print_stats(30)
```

The hot spots are typically: `link_z` (LP bounding), `fva` (preprocessing), and the solver's `solve()` loop.

---

## Quick Reference: Data Flow Through Key Functions

```
compute_strain_designs(model, **kwargs)
    │
    │── remove_dummy_bounds(model)
    │── cmp_model = model.copy()
    │── remove_ext_mets(cmp_model)
    │── [if reg] split regulatory constraints:
    │       ├── reaction-based → extend_model_regulatory(cmp_model) now
    │       └── gene-based → deferred until after GPR extension
    │
    │── ─── COMPRESS #1 (before GPR extension) ───
    │── compress_model(cmp_model, propagate_gpr=True) → cmp_mapReac_1
    │── compress_modules(sd_modules, cmp_mapReac_1)
    │── compress_ki_ko_cost(rxn_ko, rxn_ki, cmp_mapReac_1)
    │
    │── ─── FVA (on compressed model) ───
    │── bound_blocked_or_irrevers_fva(cmp_model, ...)
    │── fva(cmp_model, ...) → essential_reacs
    │
    │── ─── GPR extension (on compressed model) ───
    │── [if gene_kos] rename_genes(cmp_model, ...)
    │── [if gene_kos] reduce_gpr(cmp_model, essential_reacs, gkis, gkos)
    │── [if gene_kos] extend_model_gpr(cmp_model, ...) → reac_map
    │── [if gene_kos] apply deferred gene-based regulatory constraints
    │── [if gene_kos] merge gene costs into cmp_ko_cost / cmp_ki_cost
    │
    │── ─── COMPRESS #2 (after GPR extension) ───
    │── compress_model(cmp_model) → cmp_mapReac_2
    │── compress_modules(sd_modules, cmp_mapReac_2)
    │── compress_ki_ko_cost(all_ko, all_ki, cmp_mapReac_2)
    │── cmp_mapReac = cmp_mapReac_1 + cmp_mapReac_2
    │
    │── fva(cmp_model, ...) → final essential_reacs
    │
    │── ─── MILP construction & solving ───
    │── SDMILP(cmp_model, sd_modules, ko_cost, ki_cost, solver, M, ...)
    │       ├── SDProblem.__init__(...)
    │       │       ├── [for each module] addModule(sd_module)
    │       │       │       ├── build_primal_from_cbm(model, V_ineq, v_ineq, ...)
    │       │       │       ├── [SUPPRESS] farkas_dualize(...)
    │       │       │       ├── [PROTECT]  reassign_lb_ub_from_ineq(...)
    │       │       │       ├── [OPTKNOCK] LP_dualize(...) + block_diag assembly
    │       │       │       ├── [ROBUSTKNOCK] LP_dualize x2
    │       │       │       └── [OPTCOUPLE] LP_dualize + GCP objective
    │       │       └── link_z()
    │       │               ├── split equalities → inequalities
    │       │               ├── variable KOs → inequality KOs
    │       │               ├── parallel LP bounding → big-M values
    │       │               ├── insert big-M into A_ineq
    │       │               ├── lump duplicate pairs → equalities
    │       │               └── remaining → IndicatorConstraints
    │       └── MILP_LP.__init__(c, A_ineq, ..., solver=solver)
    │               └── instantiate Cplex/Gurobi/SCIP/GLPK backend
    ├── [BEST]     sdmilp.compute_optimal(max_solutions, time_limit)
    ├── [ANY]      sdmilp.compute(max_solutions, time_limit)
    ├── [POPULATE] sdmilp.enumerate(max_solutions, time_limit)
    │       each iteration:
    │       ├── solveZ() / populateZ(n)
    │       ├── verify_sd(z) → valid[]
    │       └── add_exclusion_constraints(z)
    ├── expand_sd(sd, cmp_mapReac)          ← processes pass 2, then pass 1
    ├── filter_sd_maxcost(sd, max_cost, ...)
    ├── postprocess_reg_sd(sd, ...)
    └── SDSolutions(model, sd, status, sd_setup)
            └── [if gene_kos] translate gene_sd → reaction_sd via gpr_eval
```
