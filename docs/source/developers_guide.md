# StrainDesign Developer's Guide

This guide explains how StrainDesign works *internally* — the mechanics, the mathematics, and the
rationale behind each stage of a strain-design / Minimal Cut Set (MCS) computation. It is written for
developers and contributors who want to understand, extend, debug, or optimize the package, rather than
just use it. For usage, see the tutorial notebooks and the {doc}`API reference <api_reference>`.

For every processing stage it answers *what* the code does, *how* it works (including the underlying
linear-algebra and optimization theory), and *why* it is built that way.

**Audience.** A scientific programmer comfortable with linear and mixed-integer programming and
constraint-based metabolic modeling, but new to this codebase. The chapters are largely self-contained,
though the notation is established in Chapter 1 and the LP/duality groundwork in Chapters 2 and 6. Code
is cited as `file.py:line`; line numbers are anchors that drift with edits, so treat them as pointers,
not addresses.

## How to read this guide

- **New to strain design / this package:** Chapter 1 (problem + notation) → Chapter 2 (LP foundation) →
  then follow the pipeline order, Chapters 3–9.
- **Optimizing performance:** Chapter 11 (bottleneck profile + levers) first, then Chapter 3
  (compression), Chapter 5 (the preprocessing FVA), Chapter 7 (MILP conditioning), Chapter 8 (enumeration).
- **Debugging correctness:** Chapter 10 (failure modes), then the relevant mechanism chapter (Chapter 4
  GPR, Chapter 3 compression, Chapter 9 solution semantics).
- **For the mathematics:** Chapter 2 (polytope/LP) → Chapter 6 (duality, Farkas, strong-duality reuse) →
  Chapter 7 (big-M vs indicator linearization) → Chapter 8 (integer cuts).

## Chapters at a glance

1. **Orientation & the strain-design problem** — the MCS problem, SUPPRESS/PROTECT/bilevel semantics, interventions & cost, the binary `z` vector, invocation, and the master notation table.
2. **The constraint-based foundation** — `Sv=0`, the flux polytope/cone, FBA & FVA as LPs, the internal standard form, and the convex geometry needed for duality.
3. **Network compression** — why compress; the exact integer/rational nullspace (fraction-free RREF, big-int path); parallel, coupled (kernel-proportionality + bound intersection), conservation-relation, and blocked/zero-flux reductions; the alternating fixpoint; GPR AND/OR propagation; the compression map; and the legacy efmtool Java backend.
4. **GPR integration** — why gene KOs are encoded as flux structure; `extend_model_gpr` pseudo-metabolite construction (AND/OR), the flux-space-invariance argument, reversible split & `reac_map`; `reduce_gpr`; the two-pass boundary and the regulatory-gene exemption.
5. **FVA in preprocessing** — the three FVA uses and their rationale; `bound_blocked_or_irrevers_fva` bound relaxation and its MILP effect; size-1 MCS extraction; the `speedy_fva` acceleration algorithm.
6. **Dualization (the mathematical core)** — LP duality & complementary slackness; Farkas' lemma and the SUPPRESS infeasibility certificate (why the dual ray is unbounded); strong-duality encoding of bilevel problems and *why the one `LP_dualize` operation is reusable* across OptKnock/RobustKnock/OptCouple/DoubleOpt.
7. **MILP construction & the z-linking** — the seed cost rows, `num_z`, block-diagonal module assembly, `prevent_boundary_knockouts`; `link_z`: per-constraint big-M from a bounding LP vs native indicator constraints, the bound-driven fork, and why indicators give a tighter relaxation.
8. **Solving & enumeration** — ANY/BEST/POPULATE objective setups; the iterative loop and superset-excluding integer cuts; solver parameters; the CPLEX-vs-Gurobi gap.
9. **Decompression & solution semantics** — reverse-map expansion of compressed interventions; size-1 MCS re-injection; `filter_sd_maxcost`; the KI value-0/`(nan,nan)` & `strip_non_ki` encoding; gene↔reaction translation.
10. **Known issues, gotchas & failure modes** — neutral-gene-KO paths and superset artifacts with mechanism; the in-place dict-mutation footgun; name truncation; numeric-status robustness.
11. **Performance, benchmarking & roadmap** — the bottleneck profile; the lever groups; benchmarking discipline (multi-seed, known-answer gates, MCS2/gMCSpy).
12. **Model surgery & constraint parsing** — the utility layer: `remove_ext_mets`, regulatory-intervention encoding, `gene_kos_to_constraints`, module/cost remapping through compression, and `parse_constr` (strings → matrix rows).
13. **The object model & result API** — `SDModule` (types, validation), `SDSolutions` (result access, KO/KI encoding, lazy expansion, save/load), the `sd_setup` bundle, and the preprocessed-dump workflow.
14. **The solver-interface layer** — `MILP_LP` and the four backends: how indicators/big-M/populate/status/params map onto CPLEX, Gurobi, SCIP, GLPK.
15. **Analysis & exploration API** — the standalone tools (not part of the compute pipeline): `fba`/`fva`, `yopt` yield optimization, `plot_flux_space` (production envelopes, yield spaces), and the compressed-analysis tools.

## Repository structure

```
straindesign/
├── __init__.py                     # Package exports & avail_solvers detection
├── names.py                        # All string constants (solver names, module types, etc.)
│
├── compute_strain_designs.py       # Main user-facing orchestration function
├── strainDesignModule.py           # SDModule: problem specification (a dict subclass)
├── strainDesignProblem.py          # SDProblem: translates model+modules → MILP matrices
├── strainDesignMILP.py             # SDMILP: solves the MILP, manages the solution loop
├── strainDesignSolutions.py        # SDSolutions: result container, GPR translation
│
├── compression.py                  # Network compression (RREF, RationalMatrix, nullspace)
├── networktools.py                 # GPR extension, regulatory extension, compress wrappers, LP suppression
├── lptools.py                      # FVA, FBA, flux-space plotting, solver selection
├── speedy_fva.py                   # Accelerated FVA (scan-LP + push-to-bounds)
├── parse_constr.py                 # Constraint/expression string → matrix conversion
├── indicatorConstraints.py         # IndicatorConstraints data class
├── solver_interface.py             # MILP_LP factory: instantiates the right backend
├── cplex_interface.py              # CPLEX backend (Cplex_MILP_LP)
├── gurobi_interface.py             # Gurobi backend (Gurobi_MILP_LP)
├── scip_interface.py               # SCIP backend (SCIP_MILP_LP)
├── glpk_interface.py               # GLPK backend (GLPK_MILP_LP)
├── efmtool_cmp_interface.py        # EFMtool JAR interface (legacy compression backend)
├── pool.py                         # SDPool: cross-platform multiprocessing pool
└── efmtool.jar                     # Bundled EFMtool binary
```

Which chapter covers which module: compression → Ch 3; GPR / networktools → Ch 4, Ch 12; FVA / lptools /
speedy_fva → Ch 5, Ch 15; dualization & problem build (`strainDesignProblem.py`) → Ch 6, Ch 7; the solve
loop (`strainDesignMILP.py`) → Ch 8; results (`strainDesignSolutions.py`) → Ch 9, Ch 13; solver
interfaces → Ch 14; parsing → Ch 12.


## 1. Orientation & the strain-design problem

This chapter is the entry point for the whole reference. It states the strain-design /
Minimal Cut Set (MCS) problem in both plain and formal terms, fixes the notation that every
later chapter reuses, defines the two atomic building blocks (SUPPRESS and PROTECT) and the
bilevel variants that generalize them, explains what "an intervention" and "minimal" mean in
this codebase, sketches the end-to-end pipeline (with forward references to the chapters that
work out each stage), and shows exactly how the package is invoked. Nothing here is proved;
the mathematics that *is* worked out later (LP duality, Farkas certificates, big-M, integer
cuts, exact compression) is named and pointed at, not reproduced.

### 1.1 The metabolic model

A constraint-based metabolic model is, for our purposes, a linear description of the space of
steady-state flux distributions a cell can sustain. Three objects define it.

- **Stoichiometric matrix** `S ∈ ℝ^{m×n}`. Rows are the `m` internal metabolites, columns are
  the `n` reactions. Entry `S[i,j]` is the signed stoichiometric coefficient of metabolite `i`
  in reaction `j` (negative = consumed, positive = produced). In `straindesign` this is a
  scipy sparse matrix; it is never formed densely for genome-scale models (iML1515 has
  `m ≈ 1877`, `n ≈ 2712`).
- **Flux vector** `v ∈ ℝ^n`. Component `v_j` is the net rate (mmol · gDW⁻¹ · h⁻¹) of reaction
  `j`. Fluxes are the *decision variables* of every LP in the package; a "strain" is not
  identified with one flux vector but with the whole set of flux vectors its network admits.
- **Bounds** `lb, ub ∈ (ℝ ∪ {±∞})^n`, applied component-wise as `lb ≤ v ≤ ub`. A reaction is
  **irreversible** when `lb_j ≥ 0` (or `ub_j ≤ 0`), **reversible** when `lb_j < 0 < ub_j`.
  Exchange/boundary reactions carry the medium definition through their bounds (e.g.
  `EX_glc__D_e` with `lb = -10` fixes maximum glucose uptake).

The **steady-state assumption** — that internal metabolite pools neither accumulate nor
deplete — is written as the homogeneous balance

```
S v = 0.
```

Together with the bounds this carves out the **flux polytope**

```
P = { v ∈ ℝ^n : S v = 0,  lb ≤ v ≤ ub }.                                     (1.1)
```

`P` is a convex polyhedron (a pointed cone truncated by the finite bounds). Its full
linear-algebraic structure — why steady state is a null-space condition, why `P` is a cone
when bounds are 0/±∞, how FBA maximizes a linear objective over `P` and how FVA sweeps each
coordinate's range — is the subject of Ch 2. For this chapter, `P` is simply *the set of flux
behaviors the unmodified model permits*.

A **flux behavior** (used informally throughout) is any linearly-describable subset of `P`:
"growth ≥ 0.1", "ethanol export ≥ 5 while growth is maximal", "no net product at all". Every
strain-design module names such a subset and declares whether it must be destroyed or
preserved.

### 1.2 Desired vs. undesired flux regions

Strain design starts from a partition of flux space into behaviors we want and behaviors we do
not. Write a **linear flux region** as a system of linear (in)equalities on `v`:

```
D = { v ∈ P : A_ineq^{(D)} v ≤ b^{(D)},  A_eq^{(D)} v = b_eq^{(D)} }.
```

Two roles:

- an **undesired region** `D⁻` is a behavior we want the engineered strain to be *unable* to
  exhibit — e.g. "the cell grows but makes no target product", or simply "the cell grows at
  all" (for a lethal knockout set). The goal is to make `D⁻ ∩ P' = ∅` in the modified network
  `P'`.
- a **desired region** `D⁺` is a behavior we want the engineered strain to *retain* — e.g.
  "growth of at least 0.1 h⁻¹ is still achievable". The goal is to keep `D⁺ ∩ P' ≠ ∅`.

An **intervention** modifies the network — most commonly by forcing some reactions' fluxes to
zero (a knockout) — turning `P` into a smaller (or, for knock-ins, larger) polytope `P'`. The
strain-design problem is: **choose the cheapest set of interventions such that every undesired
region becomes infeasible while every desired region stays feasible.** SUPPRESS and PROTECT are
exactly the machine encodings of "make `D⁻` infeasible" and "keep `D⁺` feasible".

A subtle but load-bearing convention: the zero vector `v = 0` lies in `P` for essentially every
model (all reactions off is trivially steady-state and within bounds). A SUPPRESS/PROTECT
region must therefore be defined so it *excludes* `v = 0`; otherwise "make the region
infeasible" is impossible (you cannot knock out the do-nothing state) and "keep it feasible" is
vacuous. The `SDModule` constructor enforces this for modules that carry an inner objective:
`strainDesignModule.py:316-320` runs an FBA with every reaction pinned to 0 and rejects the
module if that trivial point satisfies the constraints. This is why a lethality SUPPRESS is
written `growth ≥ 0.001` and not `growth ≥ 0` — the strict-ish positive threshold pushes the
target region off the origin.

### 1.3 SUPPRESS and PROTECT: precise semantics

These are the two atomic module types. Both take a `constraints` list describing a flux region;
they differ only in what the solver is asked to guarantee about that region in the engineered
strain.

**SUPPRESS** — *make the region infeasible.* Given a region

```
D⁻ = { v : Sv = 0,  lb ≤ v ≤ ub,  T v ≤ t }        (T, t encode the module's constraints)
```

a SUPPRESS module demands that after intervention **no** flux vector satisfies all of these
simultaneously: `D⁻ ∩ P' = ∅`. "Growth without production is impossible", "growth is
impossible" (lethality), "the yield falls below the threshold at maximum growth" are all
SUPPRESS behaviors. Mechanically this is the hard case: to certify that a *linear system has no
solution* you cannot just exhibit a point, you must produce an infeasibility certificate. The
package builds one via **Farkas' lemma / LP duality** — the module is dualized so that a bounded
dual ray exists *iff* the primal region is empty, and the intervention variables `z` are wired
to force such a ray to exist. That dualization is the mathematical core of the package and is
worked out in full in Ch 6 (`farkas_dualize`, `strainDesignProblem.py`); z-linking of the dual
rows is Ch 7.

**PROTECT** — *keep the region feasible.* Given a region `D⁺` described the same way, a PROTECT
module demands that after intervention **at least one** flux vector still satisfies all
constraints: `D⁺ ∩ P' ≠ ∅`. "Growth of ≥ 0.1 remains possible", "the model stays feasible at
all" are PROTECT behaviors. This is the easy case: feasibility is certified by a *witness* flux
vector, so PROTECT contributes the region's constraints as **raw primal rows** to the MILP —
the same `Sv=0, lb≤v≤ub, Tv≤t` block, with the `z`-linking arranged so that a knocked-out
reaction drops out of that primal system. No dualization is needed for a bare PROTECT.

This primal/dual asymmetry is fossilized in the internal name constants
(`names.py:120-123`): PROTECT was historically `'mcs_lin'` (a *linear*/primal feasibility
block) and SUPPRESS was `'mcs_bilvl'` (a *bilevel*/dualized block), before both were renamed to
`'protect'` and `'suppress'`. The renaming is cosmetic; the linear-vs-dual split it encoded is
still exactly how the two module types are assembled.

**The classical MCS = one SUPPRESS + PROTECT.** A *Minimal Cut Set* in the original sense of
Klamt & Gilles is the smallest set of reaction deletions that blocks a specified undesired
behavior while (optionally) sparing desired ones. In this package that is written as **exactly
one SUPPRESS module** (the behavior to eliminate) together with **zero or more PROTECT modules**
(behaviors to preserve). If there are no PROTECT modules, an MCS just makes the SUPPRESS region
empty (classic lethality/blocking). The code recognizes this canonical shape explicitly:
`compute_strain_designs.py:473-474` sets `is_classical_mcs` true precisely when there is one
SUPPRESS and every other module is a PROTECT, and only then does it attempt the size-1 MCS
shortcut (§1.6, Ch 5). The number of SUPPRESS and PROTECT modules is otherwise unrestricted and
they can be freely combined; several SUPPRESS modules just mean several regions must all be
eliminated at once.

Optionally, either module may carry an **`inner_objective`**. Then the region is not "all `v`
satisfying the constraints" but "all `v` that are *optimal* for the inner objective and also
satisfy the constraints". SUPPRESS-with-inner-objective says "flux states that are optimal for
(say) growth and also over-produce a by-product must be impossible"; this couples an
optimization *inside* the feasibility question and therefore uses the same dualization
machinery as the bilevel modules below (Ch 6). An `inner_opt_tol < 1` relaxes "optimal" to
"within a fraction of optimal" (`strainDesignModule.py:277-282`,
`strainDesignProblem.py:264-274`).

### 1.4 The bilevel variants (conceptual only)

MCS reasons about whole flux regions. A second family of modules reasons about what a cell will
do *if it optimizes its own objective* — the biologically realistic assumption that a strain
grows as fast as its network allows. These are **bilevel** problems: an outer design objective
subject to an inner cellular optimization. `straindesign` supports four, of which at most one
may appear in a computation (`compute_strain_designs.py:271-276`); they may still be combined
with any number of SUPPRESS/PROTECT modules.

- **OptKnock** — maximize an *outer* objective (e.g. product export) over the flux state that
  *maximizes an inner* objective (e.g. growth). Answers "what knockouts give the highest
  possible product synthesis at the growth-optimal flux state?" It bounds the *production
  potential*, not guaranteed production.
- **RobustKnock** — max–min: maximize the *worst-case* outer objective over all growth-optimal
  flux states. Guards against the alternative-optima loophole of OptKnock (the cell could pick a
  growth-optimal state that makes nothing); it maximizes the *guaranteed* production.
- **OptCouple** — maximize the *growth-coupling potential*: the gap between max growth without
  production and max growth overall. Drives designs where growth forces production.
- **inner-objective SUPPRESS/PROTECT**, and **`DOUBLEOPT`** — as in §1.3, feasibility modules whose
  region is defined relative to an inner optimum. `DOUBLEOPT` is a distinct, fully validated module type
  (`names.py`), so the complete type set is **six**: PROTECT, SUPPRESS, OPTKNOCK, ROBUSTKNOCK, OPTCOUPLE,
  DOUBLEOPT (matching Ch 13's enumeration).

Conceptually, all four reduce to the same trick: replace "`v` optimizes the inner LP" with the
LP's **strong-duality** condition (primal feasible + dual feasible + zero duality gap), which is
a set of linear constraints the outer MILP can carry. That is why one dualization routine
(`LP_dualize`) serves every bilevel case. The exact primal/dual constructions, the max–min
handling, and the growth-coupling-potential formula are Ch 6. This chapter only needs the reader
to know that these modules exist, that they set the *global objective* of the computation (see
§1.5), and that mechanically they are "SUPPRESS/PROTECT with an optimization welded inside".

### 1.5 Interventions, costs, the binary vector `z`, and "minimal"

**Intervention kinds.**

- **Knockout (KO)** — force a reaction (or, via GPR, a gene) permanently off. In the MILP a KO
  is expressed by driving the reaction's flux to 0 when its intervention variable is active.
- **Knock-in (KI)** — *add* a reaction to the network; its cost is incurred by *keeping* it,
  and it is free to omit. KI is handled as an inverted KO: the same `z` machinery with the sense
  flipped. The reaction must already exist in the model with the bounds it would have *after*
  insertion (`compute_strain_designs.py`/docstring lines 125-129).
- **Regulatory** — impose (or remove) a linear flux constraint as an intervention, e.g.
  "`EX_o2_e = -1`" to model a forced aeration change. Reaction-based regulatory constraints are
  added during preprocessing via `extend_model_regulatory`; gene-based ones are deferred until
  after GPR integration (Ch 4). Costs live in `reg_cost`.

**Costs.** Every candidate intervention carries a positive cost; `max_cost` bounds the total.
Costs are supplied per-kind: `ko_cost`, `ki_cost` (reactions), `gko_cost`, `gki_cost` (genes),
`reg_cost` (regulatory). Defaults: with reaction interventions, every reaction is a KO
candidate at cost 1 (`compute_strain_designs.py:263-264`); with `gene_kos=True`, every gene is a
KO candidate at cost 1 (`:253-257`). Supplying a partial dict *restricts* candidacy to the
listed items — anything not listed is simply not knockable. Essential reactions/genes (those
whose removal would break a PROTECT or desired region) have their cost entries dropped during
preprocessing so they are never proposed (`:381`, `:494`; Ch 5).

**The binary vector `z`.** After preprocessing, the model has been compressed and GPR-extended;
`SDProblem.__init__` allocates **one binary variable per (compressed) reaction**: `num_z = numr`
(`strainDesignProblem.py:144`), `z ∈ {0,1}^{num_z}`. `z_j = 1` means "reaction `j` is
intervened" (knocked out, or — for a KI reaction, whose sense is inverted — kept in). The cost
data is compiled (`strainDesignProblem.py:141-151`) into three aligned per-reaction arrays:

- `cost[j]` — the intervention cost of reaction `j` (0 if `j` is not targetable);
- `z_inverted[j]` — true iff `j` is a KI (a `ki_cost` entry present), meaning `z_j`'s sense is
  flipped so cost is paid for *presence*;
- `z_non_targetable[j]` — true iff `j` has neither a KO nor KI cost, so `z_j` is fixed to 0
  (`ub[j] = 1 − z_non_targetable[j]`, `strainDesignProblem.py:163`).

KIs override KOs when both are given (`:143` blanks the KO cost wherever a KI cost exists). The
resulting cost vector feeds the two budget rows placed at the very top of the MILP
(`strainDesignProblem.py:152-160`): a row `Σ cost_j z_j ≤ max_cost` (the `idx_row_mincost`
row, `b_ineq[1] = max_cost`) and a companion `−Σ cost_j z_j ≤ 0` row (`idx_row_maxcost`), plus a
reserved objective row. The exact meaning of these two rows and their interaction with KI
inversion is Ch 7; here they matter only as the place where `max_cost` enters.

**What "minimal" means.** For an MCS-only computation (all modules SUPPRESS/PROTECT), the
*global objective* is to **minimize total intervention cost** `Σ cost_j z_j`
(`strainDesignProblem.py:202-205` sets `c ← cost` and flags `is_mcs_computation`). "Minimal" has
two precisions the reader must keep distinct, and they map onto the `solution_approach` kwarg
(§1.7, Ch 8):

- **irreducible** (the `'any'` approach): the intervention set contains no proper subset that is
  itself a valid design — you cannot drop any single intervention and still block every
  SUPPRESS region. This is what "Minimal Cut Set" strictly means.
- **cardinality/cost-minimal** (`'best'` / `'populate'`): among all valid designs, one of
  globally least total cost. Every cost-minimal design is irreducible, but not vice versa.

If a bilevel module is present, the global objective is *not* cost minimization — it is the
module's own objective (OptKnock/RobustKnock outer objective, OptCouple's growth-coupling
potential), and `max_cost` merely bounds how many interventions the design may spend
(`strainDesignProblem.py:206-212` installs the module objective into the objective row instead).
The `max_cost` bound is the same in both regimes: no design may exceed it, and it is the primary
lever that keeps the enumeration tractable (the canonical benchmarks all cap it at 2–6).

### 1.6 The end-to-end pipeline at a glance

`compute_strain_designs(model, **kwargs)` (`compute_strain_designs.py:56`) is the orchestrator.
Its stages, in order, with the chapter that details each:

1. **Parse & validate** (`:178-304`) — resolve `sd_setup` vs. explicit kwargs, select the
   solver, seed the RNG, normalize cost dicts, reject overlapping gene/reaction candidates,
   rename genes whose IDs start with a digit, and re-validate each module's constraints against
   the chosen solver. (This chapter, §1.7.)
2. **Preprocess** — the bulk of wall-time (measured ~117 s of blocked/irreversible FVA on the
   iML1515 gene-MCS benchmark). It interleaves several transformations:
   - `remove_ext_mets` and reaction-based regulatory constraints (`:310-330`).
   - **Compression pass #1** (`compress_model(..., propagate_gpr=True)`, `:357`): lossless,
     *exact integer/rational* network compression on the metabolic model *before* gene
     pseudo-reactions exist — Ch 3.
   - **FVA #1** (`:373-381`): flux-variability analysis on each desired/PROTECT module to find
     reactions essential to those behaviors, and drop them from the knockable set — Ch 5.
   - **GPR integration** (`:383-422`, only if `gene_kos`): `reduce_gpr` prunes irrelevant genes,
     then `extend_model_gpr` encodes the Boolean gene–protein–reaction rules as *flux structure*
     (gene pseudo-metabolites / pseudo-reactions) so that a gene knockout becomes an ordinary
     reaction-level constraint in the same MILP; module references are remapped through
     `reac_map` — Ch 4.
   - **Compression pass #2** (`compress_model(...)`, `propagate_gpr` default, `:434`): compress
     the now GPR-extended network — Ch 3/4.
   - **FVA #2** (`bound_blocked_or_irrevers_fva`, `:450`): relax non-binding bounds to ±∞ and pin
     blocked/irreversible reactions to 0, which tightens the downstream big-M/indicator
     linearization — Ch 5.
   - **FVA #3** (knockable-scoped, `:454-494`): find reactions essential to SUPPRESS vs. PROTECT
     and, for a classical MCS problem, extract **size-1 MCS** (single reactions whose removal
     alone blocks the SUPPRESS region) so they need not be re-discovered by the MILP — Ch 5.
3. **Build the MILP** (`SDMILP(cmp_model, sd_modules, **kwargs_milp)`, `:518`; Ch 7). Each
   module is appended by `addModule` as a block: **SUPPRESS → dualized Farkas infeasibility
   rows, PROTECT → raw primal feasibility rows**, bilevel → strong-duality rows (Ch 6). Then
   `link_z` wires the binary `z` to those continuous rows, as **native indicator constraints or
   big-M** depending on bound structure (Ch 7).
4. **Solve / enumerate** (Ch 8): `compute` (ANY), `compute_optimal` (BEST), or `enumerate`
   (POPULATE). Found designs are excluded by iterative **integer cuts** so the next solve returns
   a genuinely new design.
5. **Decompress** (`_decompress_solutions`, `:589`; Ch 9): `expand_sd` reverses the two
   compression maps to recover interventions on original reactions, re-injects the size-1 MCS,
   filters by `max_cost`, and translates reaction designs to gene designs via the cobra GPR AST.

Chapters 2–5 cover preprocessing, 6–7 the MILP construction, 8 the solve loop, 9 decompression,
10 known gotchas, and 11 performance and roadmap.

### 1.7 How the package is invoked

The whole computation is one function call. The canonical e_coli_core gene-MCS benchmark
(`tests/test_09_performance.py:165-173`) is:

```python
import straindesign as sd
from straindesign.names import SUPPRESS, POPULATE

sol = sd.compute_strain_designs(
    model,                                    # a cobra.Model
    sd_modules=[sd.SDModule(model, SUPPRESS,
                            constraints="BIOMASS_Ecoli_core_w_GAM >= 0.001")],
    solution_approach=POPULATE,               # enumerate all cost-minimal designs
    max_cost=3,                               # ≤ 3 interventions
    gene_kos=True,                            # knock out genes, not reactions
    solver=solver,                            # 'cplex' | 'gurobi' | 'glpk' | 'scip'
)
# sol.reaction_sd  -> list of reaction-level designs (455 for this problem)
# sol.gene_sd      -> the corresponding gene-level designs
```

This one SUPPRESS module says *"flux states with biomass ≥ 0.001 must become impossible"* — i.e.
find gene knockout sets that are lethal — and `POPULATE` asks for **all** minimal such sets
(455 of them; CPLEX ≈ 1.2 s). No PROTECT module is present, so nothing is preserved beyond
model feasibility.

**Constructing an `SDModule`** (`strainDesignModule.py:221`). Signature:
`SDModule(model, module_type, *args, **kwargs)`. `module_type` is one of `'suppress'`,
`'protect'`, `'optknock'`, `'robustknock'`, `'optcouple'`. The constructor:

- parses `constraints` into canonical `[{reac: coeff, …}, op, rhs]` triples via
  `parse_constraints` (`:290-291`); the string `"BIOMASS_Ecoli_core_w_GAM >= 0.001"` and the
  list forms `["-EX_o2_e <= 5", "ATPM = 20"]` and `[[{'EX_o2_e':-1},'<=',5], …]` are all
  accepted (`:144-152`);
- parses `inner_objective` / `outer_objective` / `prod_id` from string or dict into
  `{reac: coeff}` maps (`:296-308`);
- validates that the module type has the arguments it needs (OptKnock/RobustKnock require inner
  *and* outer objectives, `:248-257`; OptCouple requires an inner objective and `prod_id`,
  `:258-268`), and that senses/tolerances are legal (`:277-282`);
- unless `skip_checks=True`, runs an FBA to confirm the region is feasible in the original model
  and (for inner-objective modules) that `v = 0` is excluded (`:311-320`).

A `dummy` object with just an `id` may stand in for the model if `skip_checks=True` and
`reac_ids=[…]` are supplied (`:239-242, 284-285`).

**Key `compute_strain_designs` kwargs** (docstring `:70-166`, handling `:174-534`):

| kwarg | meaning | default |
|---|---|---|
| `sd_modules` | list of `SDModule` (or one module); at most one bilevel among them | required |
| `solution_approach` | `'any'` (irreducible), `'best'` (cost-minimal), `'populate'` (all cost-minimal, CPLEX/Gurobi only) | `'best'` |
| `solver` | `'cplex'`, `'gurobi'`, `'glpk'`, `'scip'`; resolved by `select_solver` | model default |
| `max_cost` | upper bound on total intervention cost | `inf` |
| `max_solutions` | cap on MILP solutions generated (designs returned may exceed this after decompression) | `inf` |
| `gene_kos` | knock out genes (triggers GPR integration) instead of reactions | `False` |
| `ko_cost` / `ki_cost` | per-reaction KO / KI cost dicts (partial dict restricts candidacy) | all-KO@1 / none |
| `gko_cost` / `gki_cost` | per-gene KO / KI cost dicts | all-gene@1 if `gene_kos` / none |
| `reg_cost` | regulatory-intervention constraints → cost | none |
| `compress` | run the iterative network compressor | `True` |
| `M` | if set (nonzero), use big-M instead of indicator constraints; GLPK forces `M=1000` | `None` (→ `inf` = indicators) |
| `seed` | MILP seed (feeds solver branch-and-bound) | random (`:215-217`) |
| `time_limit` | MILP solver time limit (s) | `inf` |

`M` deserves a note because it silently changes the MILP encoding. With the default `M = None`,
`SDProblem.__init__` sets `self.M = np.inf` (`strainDesignProblem.py:125-126`), and `link_z`
attaches each `z` to its continuous rows as a **native indicator constraint** — except GLPK,
which cannot express indicators and is forced to `M = 1000` (`:120-124`). Because SUPPRESS's
dualized rows are unbounded (the Farkas ray) while PROTECT's primal rows are finite-flux, the
*emergent* behavior under `M = inf` is that SUPPRESS rows become indicators and PROTECT rows
become big-M — but this is a consequence of bound structure inside `link_z`, not a hard-coded
per-module switch (Ch 7). No MIP optimality gap is set anywhere, so both CPLEX and Gurobi run at
their default 1e-4 relative gap (Ch 8, Ch 11).

The call returns an `SDSolutions` object exposing `reaction_sd` (reaction-level designs) and,
when `gene_kos`, `gene_sd` (gene-level designs) plus a `status` — the translation between the
two is Ch 9.

### 1.8 Notation reference

The symbols below are used consistently in later chapters; where a symbol names a concrete
attribute in the code the file/field is given.

| symbol | meaning | code |
|---|---|---|
| `m` | number of (internal) metabolites = rows of `S` | `len(model.metabolites)` |
| `n`, `numr` | number of reactions = columns of `S` = length of `v` | `len(model.reactions)` |
| `S ∈ ℝ^{m×n}` | stoichiometric matrix | sparse `S` |
| `v ∈ ℝ^n` | flux vector (LP decision variables) | — |
| `lb, ub ∈ (ℝ∪{±∞})^n` | lower / upper flux bounds | `SDProblem.lb`, `.ub` |
| `P` | flux polytope `{v : Sv=0, lb≤v≤ub}` (eq. 1.1) | — |
| `D⁻`, `D⁺` | undesired (SUPPRESS) / desired (PROTECT) flux region | module `constraints` |
| `z ∈ {0,1}^{num_z}` | binary intervention vector, one per compressed reaction | `SDProblem`, `num_z = numr` (`:144`) |
| `cost ∈ ℝ_{≥0}^{num_z}` | per-reaction intervention cost | `SDProblem.cost` (`:145-151`) |
| `z_inverted` | KI mask (cost paid for *presence*) | `.z_inverted` (`:148`) |
| `z_non_targetable` | non-knockable mask (`z_j` fixed 0) | `.z_non_targetable` (`:149`) |
| `max_cost` | budget: `Σ cost_j z_j ≤ max_cost` | `.max_cost`, `b_ineq[1]` (`:157-160`) |
| `A_ineq z ≤ b_ineq` | MILP inequality block (top rows: budget + objective) | `.A_ineq`, `.b_ineq` (`:156-160`) |
| `A_eq z = b_eq` | MILP equality block | `.A_eq`, `.b_eq` (`:167-168`) |
| `M` | big-M constant (∞ ⇒ indicator constraints) | `.M` (`:120-126`) |
| `T v ≤ t` | a module's linear region constraints (schematic) | `lineqlist2mat` (`addModule`) |
| `c` | MILP objective coefficients (cost vector for MCS; module objective for bilevel) | `.c` (`:202-212`) |
| `z_map_*` | maps linking `z` to constraint rows / variables | `.z_map_constr_ineq/_eq/_vars` |

Two matrix conventions recur. First, "primal" always refers to a flux-space LP over `v`
(`build_primal_from_cbm`), and "dual" to its LP dual over multipliers (`LP_dualize`,
`farkas_dualize`); a SUPPRESS block lives in dual space, a PROTECT block in primal space (§1.3,
Ch 6). Second, equality constraints (`Sv = 0`) contribute **free** dual variables while
inequality constraints contribute **sign-constrained** ones — a distinction the dualization code
handles carefully and which Ch 6 proves out. Keep the `z`-vs-`v` split in mind: `z` is binary
and indexes *interventions*; `v` (and the dual multipliers) are continuous and index *flux
behavior*. The entire MILP is the coupling of these two through the `z_map_*` matrices.


## 2. The constraint-based foundation

Everything `straindesign` does — compression, FVA-based preprocessing, dualization, the MILP itself — is built on one linear-algebraic object: the set of *steady-state flux distributions* of a metabolic network, carved out of ℝⁿ by a homogeneous equation `S·v = 0` and a box of bounds `lb ≤ v ≤ ub`. This chapter derives that object from first principles, establishes the polyhedral geometry the later chapters lean on (faces, vertices, rays, the recession cone — the machinery that makes a Farkas certificate exist and a dual go unbounded), and shows precisely how FBA and FVA are posed as linear programs in the code. It closes with the *standard form* `(A_ineq, b_ineq, A_eq, b_eq, lb, ub, c)` that is the lingua franca of the whole package, and how a cobra model is poured into it by `build_primal_from_cbm` (`strainDesignProblem.py:971`).

Notation follows Ch 1: `S ∈ ℝ^{m×n}` is the stoichiometric matrix, `v ∈ ℝⁿ` the flux vector, `m` metabolites, `n` reactions.

### 2.1 Mass balance and the steady-state assumption

#### 2.1.1 From dynamic mass balance to `S·v = 0`

Consider a well-mixed cell (or compartment) of constant volume containing `m` internal metabolites with concentration vector `x(t) ∈ ℝ^m` (units: mmol·gDW⁻¹, per gram dry weight), and `n` reactions with flux (rate) vector `v(t) ∈ ℝⁿ` (units: mmol·gDW⁻¹·h⁻¹). The stoichiometric matrix `S ∈ ℝ^{m×n}` has entry `S_{ij}` = the signed molar stoichiometric coefficient of metabolite `i` in reaction `j`: negative if `i` is consumed, positive if produced, zero if uninvolved. Column `j` of `S` is the net reaction vector of reaction `j`; row `i` lists every reaction touching metabolite `i`.

The instantaneous mass balance for each internal metabolite is a bookkeeping identity — rate of change = production − consumption, summed over all reactions weighted by their stoichiometry:

```
dx_i/dt = Σ_{j=1..n} S_{ij} · v_j          (i = 1..m)
```

or in matrix form

```
dx/dt = S · v(t).
```

A dilution/growth term is folded into the biomass reaction and exchange fluxes in genome-scale models, so the bare `S·v` form is the working equation. The **steady-state assumption** is that internal metabolite pools do not accumulate or deplete on the timescale of interest:

```
dx/dt = 0    ⟹    S · v = 0.                          (SS)
```

The biological justification is timescale separation: intracellular metabolite turnover is on the order of seconds to sub-second, while the phenotypes of interest (growth rate, product secretion) play out over hours. Over the slow timescale the fast internal pools are effectively at quasi-steady-state, so their net rate of change is negligible relative to the through-fluxes. Crucially, `(SS)` says nothing about the fluxes being small — it says they are *balanced*: whatever is made is immediately consumed. Metabolites we deliberately allow to accumulate or leave the system (biomass, secreted products, medium components) are handled not by relaxing `(SS)` but by giving them dedicated **exchange/boundary reactions** (Sec 2.3) that act as sources/sinks, so those degrees of freedom re-enter through `v`, never as a nonzero right-hand side.

#### 2.1.2 Dimensions, rank, and what the solution set *is*

`(SS)` is a homogeneous linear system: `m` equations, `n` unknowns. In genome-scale models `n > m` (typically by a factor of ~1.3–2; e.g. iML1515 has `n ≈ 2712` reactions, `m ≈ 1877` metabolites), so the system is underdetermined and has a nontrivial solution space. That solution space is exactly the **null space (kernel)** of `S`:

```
𝒩(S) = { v ∈ ℝⁿ : S·v = 0 }.
```

Let `r = rank(S) ≤ min(m, n)`. By the rank–nullity theorem,

```
dim 𝒩(S) = n − r.
```

Two facts about `r` matter downstream:

- **`r` is usually strictly less than `m`.** Rows of `S` are linearly dependent whenever there is a *conservation relation* — a left null vector `γ ∈ ℝ^m` with `γᵀ S = 0`, meaning the pool `γᵀx` is conserved by every reaction (e.g. total carbon in a closed sub-network, or a moiety like CoA/ACP that is never net-produced). Each independent conservation relation drops `rank(S)` below `m` by one, i.e. makes one metabolite balance row redundant given the others. Ch 3's `remove_conservation_relations` (`compression.py`) exploits exactly this: redundant rows can be deleted from `S` without changing `𝒩(S)`, shrinking the equality block of the LP.
- **`n − r`, the degrees of freedom, is the dimension of the flux cone before bounds.** These are the independent "flux modes" you may set freely; the rest are pinned by balance. Compression (Ch 3) works in `𝒩(S)` using an *exact integer/rational* nullspace basis (never floating point — see the engagement notes and Ch 3's RREF construction), because the geometry of the cone must be preserved bit-for-bit.

Without any bounds, `𝒩(S)` is a linear subspace: closed under addition and under multiplication by *any* real scalar (positive, negative, or zero). If `v` balances, so does `−v` (run every reaction backward) and so does `α·v` for any `α`. A pure subspace is not yet a useful model of a cell — it permits negative fluxes through irreversible reactions and unbounded fluxes through everything. Thermodynamics and capacity limits enter as bounds, and that is what turns the subspace into a *cone/polytope*.

### 2.2 Bounds, reversibility, and the flux polytope

#### 2.2.1 Bounds encode direction and capacity

Each reaction `j` carries a lower and upper flux bound, assembled into vectors `lb, ub ∈ (ℝ ∪ {±∞})ⁿ`:

```
lb_j ≤ v_j ≤ ub_j        (j = 1..n).
```

The **sign convention** is fixed by how the reaction is written: column `j` of `S` is oriented so that positive `v_j` means "forward" (left-to-right as written). Reversibility is then purely a statement about `lb_j`:

| Reaction type | Typical bounds | Meaning |
|---|---|---|
| Irreversible (forward) | `0 ≤ v_j ≤ ub_j` | may only run forward; `v_j ≥ 0` enforced by `lb_j = 0` |
| Reversible | `−∞ (or −c) ≤ v_j ≤ +∞ (or +c)` | may run either direction |
| Irreversible (reverse-only) | `lb_j ≤ v_j ≤ 0` | only the reverse net direction is thermodynamically allowed |
| Fixed / measured | `lb_j = ub_j = β` | flux pinned to a measured value |

There is nothing special about "reversible" beyond `lb_j < 0`: direction is *entirely* an artifact of the sign of the bounds, not a separate attribute the LP sees. This is why compression and GPR integration (Ch 3, Ch 4) freely **split** a reversible reaction into a forward part (`0 ≤ v_j⁺`) and a reverse part (`0 ≤ v_j⁻`) with `v_j = v_j⁺ − v_j⁻`: it is a lossless re-encoding of the same bound interval into two irreversible columns, needed because a knockout / gene rule must act on a nonnegative flux magnitude.

**A subtle but load-bearing point the code relies on** (recorded in the engagement notes and re-derived in Ch 7's `prevent_boundary_knockouts`): an LP *variable bound* `lb_j ≤ v_j` is not the same object as a *constraint row*. A constraint row `−v_j ≤ −lb_j` can be selectively switched off by a binary `z` (that is how a knockout is simulated: multiply the effective bound by `z`), whereas a hard variable bound cannot be overridden by any row you add — the variable box always wins. `prevent_boundary_knockouts` (`strainDesignProblem.py:1322`) therefore migrates the *knockable* side of a bound (a negative lower bound or a positive upper bound on a knockable reaction) out of the box and into `A_ineq` so that `z` can later clamp it; non-knockable bounds stay in the box. The mechanics belong to Ch 7, but the reason lives here: **direction and capacity are encoded in bounds, and only bounds that have become rows can be knocked out.**

#### 2.2.2 The steady-state flux set is a polyhedron (cone / polytope)

Intersect the kernel with the box:

```
P = { v ∈ ℝⁿ : S·v = 0,  lb ≤ v ≤ ub }.                (FLUX-POLYHEDRON)
```

`P` is the intersection of a linear subspace (`𝒩(S)`, cut out by the equalities `S·v=0`) with a box (finitely many inequalities `v_j ≤ ub_j`, `−v_j ≤ −lb_j`). A finite intersection of closed half-spaces and hyperplanes is by definition a **convex polyhedron**. Two special shapes matter:

- If all bounds are homogeneous — every finite bound is `0`, the rest `±∞` — then `P` is closed under nonnegative scaling: `v ∈ P, α ≥ 0 ⟹ α v ∈ P`. This is a **polyhedral cone**, the *flux cone* `C = { v : S v = 0, v_j ≥ 0 for irreversible j }`. It is the natural home of Elementary Flux Modes and, dually, of Minimal Cut Sets: an MCS is a minimal set of constraints whose removal empties a target sub-cone.
- With finite bounds present (`ub_j < ∞`, a fixed uptake `lb = ub`, etc.), `P` is a **bounded (or partially bounded) polytope** — the object FBA optimizes over.

Convexity is not a nicety; it is *the* enabling property. Because `P` is convex, a linear objective attains its optimum at an extreme point (a vertex), FVA's per-reaction min/max are well-defined and attained, and — most importantly for Ch 6 — infeasibility of a target region has a *certificate* (Farkas), and the dual of an LP over `P` behaves predictably (Sec 2.5).

##### A 3-reaction toy

Take metabolites A, B and reactions `v1: → A`, `v2: A → B`, `v3: B →` (an uptake, a conversion, a secretion). With metabolites {A, B} as rows and reactions {v1, v2, v3} as columns,

```
        v1  v2  v3
   A [   1  -1   0 ]
   B [   0   1  -1 ]
S =
```

`S·v = 0` gives `v1 = v2` (A balance) and `v2 = v3` (B balance), so `v1 = v2 = v3`. The kernel is one-dimensional: `𝒩(S) = span{(1,1,1)}`, consistent with `n − r = 3 − 2 = 1`. Add irreversibility `v ≥ 0` and a capacity `v1 ≤ 10`: `P = { (t,t,t) : 0 ≤ t ≤ 10 }` — a line segment, a 1-D polytope with two vertices `(0,0,0)` and `(10,10,10)`. Drop the upper bound and `P` becomes the ray `{ t(1,1,1) : t ≥ 0 }` — a 1-D cone with a single extreme ray. This tiny example already exhibits everything Sec 2.5 formalizes: a vertex, a recession ray, and (if we asked for `v1 ≥ 11` on the capped model) an infeasible region whose infeasibility is provable.

### 2.3 Exchange / boundary reactions

Internal metabolites must balance, but a cell is an open system: it takes up substrate and secretes product/biomass. These flows are modeled by **exchange (boundary) reactions** — columns of `S` with a *single* nonzero entry (they touch exactly one metabolite), representing a source or sink across the system boundary. An uptake reaction `EX_glc: glc_e ↔` has stoichiometry `−1` on external glucose; by convention the exchange flux is written so that **negative = uptake, positive = secretion**. The medium is then defined purely by bounds on exchanges: `lb(EX_glc) = −10` allows up to 10 mmol·gDW⁻¹·h⁻¹ glucose uptake, `lb(EX_o2) = 0` makes the environment anaerobic, `lb = ub = 0` deletes a metabolite from the medium.

Because exchanges are genuine columns with genuine bounds, `(SS)` still holds with a strict zero right-hand side: the "accumulation" of secreted product is carried by the exchange flux, not by a nonzero `dx/dt`. This is the mechanism promised in Sec 2.1: everything that would otherwise break steady state is re-expressed as a reaction. Practically, exchanges are the reactions the SUPPRESS/PROTECT modules point at (e.g. "product exchange must stay ≥ y" for PROTECT, "biomass with zero product exchange must be impossible" for SUPPRESS), and they are usually excluded from the knockable set — you knock out *internal* enzymatic steps, not the definition of the medium. Ch 5's essential-reaction FVA and Ch 7's `prevent_boundary_knockouts` both special-case them.

### 2.4 FBA and FVA as linear programs — and how the code builds them

#### 2.4.1 FBA: one LP

**Flux Balance Analysis** picks, among all steady-state flux distributions, one that maximizes a linear objective `cᵀv` (classically `c` = the biomass reaction indicator, so `cᵀv` = growth rate):

```
maximize   cᵀ v
subject to S v = 0
           lb ≤ v ≤ ub.                                 (FBA)
```

This is a linear program over the polytope `P`. Its optimum is attained at a vertex of `P` (Sec 2.5). The optimal *value* is unique; the optimal *v* need not be (the objective face can be higher-dimensional — this degeneracy is exactly why pFBA and FVA exist).

`straindesign` implements `(FBA)` in `fba` (`lptools.py:438`). The construction is worth tracing because it fixes the sign/standard-form conventions used everywhere:

1. **Objective.** `c` comes from `model.reactions[i].objective_coefficient`, or from a user `obj` dict parsed by `linexprdict2mat` (`lptools.py:499-504`).
2. **Sense flip to a minimizer.** The internal solver interface `MILP_LP` always *minimizes*. So a maximization is turned into a minimization by negating `c` (`lptools.py:506-509`):
   ```python
   obj_sense = 'maximize'
   c = [-i for i in c]      # min (−cᵀv) ≡ max (cᵀv)
   ```
   and the reported objective is negated back at the end (`Solution(objective_value=-opt_cx, ...)`, `lptools.py:613`). **Every LP in the package is a minimization internally**; keep this in mind when reading the dual (Ch 6).
3. **Equality block = stoichiometry (+ user equalities).** `A_eq` starts as `S` via cobra's `create_stoichiometric_matrix(model)` (`lptools.py:523-524`), with `b_eq = 0` (one zero per metabolite). Any user equality constraints (parsed to matrix form by `lineqlist2mat`) are stacked underneath:
   ```python
   A_eq = sparse.vstack((A_eq_base, A_eq));  b_eq = b_eq_base + b_eq
   ```
   So `A_eq·v = b_eq` is literally `[S; V_eq]·v = [0; v_eq]`.
4. **Inequality block = user inequalities only.** If the user supplied none, `A_ineq` is an empty `0×n` matrix (`lptools.py:532-533`). Reaction directionality and capacity are *not* put into `A_ineq` here; they ride in the variable box:
5. **Box.** `lb, ub` are read straight off the reactions (`lptools.py:535-536`).
6. **Solve.** `MILP_LP(c, A_ineq, b_ineq, A_eq, b_eq, lb, ub, solver).solve()`.

Two robustness wrinkles are handled explicitly and are worth flagging because they reflect the polyhedral theory of Sec 2.5:

- **Unbounded objective** (`status == UNBOUNDED`, `lptools.py:544`): the objective face is a recession ray — `cᵀv → ∞` along a ray in the recession cone. The code then re-solves for a *finite representative* point by fixing `cᵀv` to a computed value (`add_eq_constraints`), so the caller still gets a usable flux vector rather than "∞".
- **pFBA** (`pfba` ≥ 1, `lptools.py:554`): after the primary optimum `opt_cx` is found, a secondary LP minimizes total flux `Σ|v_j|` (mode 1) or the *number* of active reactions (mode 2) subject to `cᵀv = opt_cx`. Minimizing `Σ|v_j|` is linearized by the classic reversible split `v_j = v_j⁺ − v_j⁻`, `v_j⁺, v_j⁻ ≥ 0`, minimizing `Σ(v_j⁺ + v_j⁻)` (`lptools.py:587-602`) — the same split motivated in Sec 2.2.1. Mode 2 uses indicator constraints and binaries (`lptools.py:570-583`), a mini-MILP, foreshadowing the main event.

#### 2.4.2 FVA: 2n LPs

**Flux Variability Analysis** asks, for each reaction `i`, the full range of `v_i` consistent with steady state (optionally after fixing the objective, or under extra constraints):

```
for each i = 1..n:
    v_i^min = minimize v_i   s.t.  S v = 0,  lb ≤ v ≤ ub  (+ extra constr.)
    v_i^max = maximize v_i   s.t.  same feasible set.
```

That is `2n` LPs sharing one feasible polytope `P`; only the objective vector `e_i` (the `i`-th unit vector) changes between them. FVA is the workhorse of preprocessing: it detects **blocked** reactions (`v_i^min = v_i^max = 0` — the reaction can carry no steady-state flux at all, so it is deleted), **essential** reactions in a PROTECT/desired module (bounds forcing `|v_i| > 0`, hence not knockable — dropped from the knockable set), and reactions whose model bound never binds (relaxed to `±∞` by `bound_blocked_or_irrevers_fva`). The three distinct uses and their rationale are Ch 5's subject; here we fix only the LP form and the code's entry point.

In `straindesign` the public `fva` (`lptools.py:245`) is a thin wrapper that delegates to `speedy_fva` (`lptools.py:281-282`):
```python
def fva(model, **kwargs):
    from straindesign.speedy_fva import speedy_fva
    return speedy_fva(model, **kwargs)
```
`speedy_fva` does *not* solve `2n` independent LPs blindly; it uses a two-phase accelerated scheme (global scan LPs with dual-simplex warm-starts resolve ~half the bounds cheaply, then individual LPs for the rest, with optional coupled-reaction compression for large models). The mathematics of that acceleration is Ch 5's. The *reference* brute-force implementation — literally `2n` LPs — survives as `fva_legacy` (`lptools.py:285`) and makes the standard form explicit:

- Build `A_eq = [S; V_eq]`, `b_eq = [0…0, v_eq]`, `A_ineq/b_ineq` from user constraints, `lb/ub` from the reactions (`lptools.py:303-315`) — identical assembly to FBA.
- Instantiate one `MILP_LP` and iterate over `i ∈ {0, …, 2n−1}`. The helper `idx2c(i, prev)` (`lptools.py:107`) maps index `i` to an objective: `col = floor(i/2)` is the reaction, `sig = sign(mod(i,2) − 0.5)` is `−1` for even `i` (**maximize**, since the solver minimizes `−v_col`) and `+1` for odd `i` (**minimize** `v_col`). The warm-start trick is `prev`: consecutive LPs differ in one objective coefficient, so the simplex basis is reused.
- Results are unpacked with the sign undone: `maximum = −x[even]`, `minimum = x[odd]` (`lptools.py:386-388`), and anything with `|value| < 1e-11` is snapped to `0` to kill solver noise (`lptools.py:384`).

The takeaway for a developer: **FVA is `2n` LPs over the same polytope, distinguished only by the objective `±e_i`, and the internal minimize-only convention means "maximize `v_i`" is submitted as "minimize `−v_i`" and negated on return.** Every accelerated variant is an optimization of *how many* of those `2n` LPs you actually solve, not of *what* they compute.

### 2.5 Enough polyhedral theory for Ch 6

The dualization chapter (Ch 6) needs three geometric facts about `P` (and about the target regions the modules define). We state them here with just enough proof-sketch to make the later "why does a Farkas certificate exist" and "why is this dual unbounded" self-contained.

#### 2.5.1 Faces, vertices, rays, recession cone

Let `P = { v : A v ≤ b }` be any polyhedron (fold the equalities `S v = 0` into two inequalities `S v ≤ 0`, `−S v ≤ 0`, and the box into rows, to view `P` uniformly as `A v ≤ b` with `A ∈ ℝ^{p×n}`).

- A **face** of `P` is `P ∩ { v : wᵀv = δ }` for a valid inequality `wᵀv ≤ δ` (one satisfied by all of `P`). Faces of dimension 0 are **vertices**, dimension 1 are **edges**.
- A **vertex** (extreme point) is a `v ∈ P` at which `n` linearly independent constraint rows are active (tight). Equivalently, `v` is not the midpoint of any segment in `P`. A linear objective, if bounded on `P`, attains its optimum on a face, and if `P` has a vertex, at a vertex — this is *why* FBA/FVA optima are attained and why they occur at biologically "cornered" flux states.
- The **recession cone** (characteristic cone) of `P` is
  ```
  rec(P) = { d ∈ ℝⁿ : A d ≤ 0 } = { d : S d = 0,  d_j ≥ 0 (j irrev), d_j ≤ 0 where ub binds, … }.
  ```
  A direction `d ∈ rec(P)` is a **ray**: for any `v ∈ P` and `t ≥ 0`, `v + t d ∈ P`. Rays are the unbounded directions. **Minkowski–Weyl / decomposition theorem:** every polyhedron decomposes as `P = conv(vertices) + cone(extreme rays)` — a bounded convex hull plus a recession cone. When all bounds are finite, `rec(P) = {0}` and `P` is a polytope (bounded). When some bound is `±∞` (or the flux cone case, all homogeneous), `rec(P)` is nontrivial and carries the unbounded modes — precisely the extreme ray `(1,1,1)` we saw in the Sec 2.2.2 toy.

#### 2.5.2 Why an objective goes unbounded → the dual is infeasible

`(FBA)`'s objective `cᵀv` is **unbounded above on `P` iff there is a recession ray `d ∈ rec(P)` with `cᵀd > 0`** (walk along `d` forever, gaining `cᵀd` per unit). By LP duality (stated and proved in Ch 6), primal unboundedness is equivalent to **dual infeasibility**: no dual `y` satisfies the dual constraints. This is the geometric root of the `status == UNBOUNDED` branch in `fba` (Sec 2.4.1) — and, more importantly, of the bilevel dualization in Ch 6, where forcing primal–dual objective equality (strong duality) is used to encode inner-problem optimality. A recession ray of the *inner* polytope with positive inner objective would make that encoding vacuous, which is exactly the pathology the module builders guard against.

#### 2.5.3 Why an infeasible target region yields a Farkas certificate

The SUPPRESS module (Ch 1) demands that a *target* flux region become **empty** after knockouts. The target region is itself a polyhedron:

```
T = { v : S v = 0,  lb ≤ v ≤ ub,  T_ineq v ≤ t_ineq,  T_eq v = t_eq }
```

(the module's inequalities, e.g. "biomass ≥ 0.001 and product ≤ 0"). Write `T` uniformly as `{ v : A v ≤ b }`. **Farkas' lemma (affine form)** states the dichotomy:

> Exactly one of the following holds:
> (a) `∃ v : A v ≤ b`  (the region is nonempty), or
> (b) `∃ y ≥ 0 : Aᵀ y = 0  and  bᵀ y < 0`  (a certificate of emptiness).

The vector `y` in (b) is a **Farkas certificate**: a nonnegative combination of the constraint rows that derives the contradiction `0 = (Aᵀy)ᵀv ≤ bᵀ y < 0`. Geometrically, `y` is a recession ray of the *dual* polyhedron — an unbounded dual direction — which is why Ch 6's `farkas_dualize` deliberately builds a dual whose *ray* (not vertex) encodes infeasibility, and adds a normalization row (e.g. `bᵀy = −1`) to pin down the otherwise scale-free ray. The existence of `y` is guaranteed by Farkas' lemma *precisely because `T` is a polyhedron* — convexity is what makes emptiness certifiable by a single linear witness. The MILP then hunts for a knockout set `z` that *forces* case (b) to hold, i.e. makes such a `y` exist. That is the entire logical content of "SUPPRESS → Farkas infeasibility certificate," and it rests on the geometry of this section.

For completeness, **strong duality** (used for PROTECT-as-feasibility and for bilevel strong-duality encodings) and the full proofs of Farkas' lemma and the duality theorem are Ch 6's; here we have only established *why the objects exist* — `P`, `T`, their vertices, rays, and recession cones — and that emptiness/unboundedness are certifiable, which is all the later chapters assume.

### 2.6 The standard form and `build_primal_from_cbm`

Internally, `straindesign` never manipulates a cobra model directly during MILP assembly. Every constraint system — primal, dual, per-module block — is carried as a tuple in **one standard form**:

```
A_ineq · x ≤ b_ineq
A_eq   · x = b_eq
lb ≤ x ≤ ub
minimize  cᵀ x
```

with `A_ineq ∈ ℝ^{p×N}`, `A_eq ∈ ℝ^{q×N}`, `x ∈ ℝ^N`, `lb, ub ∈ (ℝ∪{±∞})^N`, `c ∈ ℝ^N`. This is the signature of `MILP_LP` and the contract every builder honors. It is deliberately minimal and symmetric: separate equality and inequality blocks (so dualization can treat them by their type — equalities → free dual vars, inequalities → sign-restricted dual vars; Ch 6), an explicit variable box (kept separate from `A_ineq` so bounds and rows are distinguishable, per Sec 2.2.1), and a single objective `c` in minimize sense (Sec 2.4.1).

#### 2.6.1 Mapping a cobra model into the standard form

`build_primal_from_cbm` (`strainDesignProblem.py:971`) is the canonical adapter from a cobra model (plus optional extra constraints `V_ineq·x ≤ v_ineq`, `V_eq·x = v_eq`) into this form. In the *primal* every variable is a reaction flux, so `N = numr = len(model.reactions)`. The construction (`strainDesignProblem.py:1003-1026`):

```python
numr = len(model.reactions)
S    = sparse.csr_matrix(create_stoichiometric_matrix(model))
A_eq   = sparse.vstack((S, V_eq))                 # [ S ; V_eq ]
b_eq   = [0]*S.shape[0] + v_eq                     # [ 0 ; v_eq ]
A_ineq = V_ineq.copy();  b_ineq = v_ineq.copy()    # only user inequalities
lb = [float(r.lower_bound) for r in model.reactions]
ub = [float(r.upper_bound) for r in model.reactions]
```

So the equality block is the stoichiometry `S` (the steady-state constraint `S·x = 0`) with the module's equalities stacked below; `b_eq` is zeros for the metabolite rows and `v_eq` for the extra rows. The inequality block starts as *just* the module's inequalities — reaction directionality/capacity live in the box `lb/ub`, exactly as in FBA (Sec 2.4.1). The objective `c` defaults to the model's `objective_coefficient` vector.

#### 2.6.2 The bookkeeping matrices (`z_map_*`)

Beyond the LP itself, `build_primal_from_cbm` returns three **association matrices** that thread reaction identity (and therefore knockout binary `z_j`) through the standard form (`strainDesignProblem.py:1021-1023`):

- `z_map_vars` — shape `numz × N`, relating intervention binaries to *variables*. In the primal it is the identity (`sparse.identity(numr)`): variable `x_j` *is* reaction `j`, so knocking out reaction `j` acts on variable `j`. An entry `+1` marks "this reaction's knockout removes this variable"; `−1` marks an addition (knock-in).
- `z_map_constr_ineq` — shape `numz × p`, relating binaries to *inequality rows*. Zero at construction, because in the raw primal no inequality row is tied to a specific reaction knockout (the model's own bounds are still in the box).
- `z_map_constr_eq` — shape `numz × q`, relating binaries to *equality rows*. Zero for the same reason (the metabolite balances belong to no single reaction's knockout).

These matrices are the mechanism by which dualization (Ch 6) and z-linking (Ch 7) know *which* rows and variables a given `z_j` must switch. When `LP_dualize` transposes the system, it also transposes/re-routes these maps so that a reaction still tracks the correct dual object — this is why `LP_dualize` (`strainDesignProblem.py:1028`) takes and returns the `z_map_*` triple, not just the LP.

The last step in the adapter is the bound migration already previewed in Sec 2.2.1:

```python
A_ineq, b_ineq, lb, ub, z_map_constr_ineq = prevent_boundary_knockouts(
        A_ineq, b_ineq, lb.copy(), ub.copy(), z_map_constr_ineq, z_map_vars)
```

`prevent_boundary_knockouts` (`strainDesignProblem.py:1322`) moves the knockable side of each nonzero bound (a negative lower bound / positive upper bound on a reaction that carries a nonzero `z`-mapping, `strainDesignProblem.py:1358`) out of the box and into a new `A_ineq` row, updating `z_map_constr_ineq` so the row is tagged with the owning reaction — but leaves non-knockable bounds untouched. The *why* (a hard variable box cannot be relaxed by a binary, only a constraint row can) is Sec 2.2.1; the *how* (which bounds move, and how the resulting big-M/indicator gets attached) is Ch 7. What matters for this chapter is that after `build_primal_from_cbm` returns, the tuple `(A_ineq, b_ineq, A_eq, b_eq, lb, ub, c, z_map_constr_ineq, z_map_constr_eq, z_map_vars)` is a faithful, dualization-ready standard-form encoding of `P` together with the intervention bookkeeping — the single object every subsequent chapter consumes.

With `S·v = 0` derived from mass balance, the flux set established as a convex polyhedron `P` (cone when homogeneous, polytope when bounded), FBA/FVA pinned down as LPs over `P` in the exact form the code builds them, and the standard-form tuple that `build_primal_from_cbm` produces, the linear-algebra/LP bedrock is in place. Ch 3 works *inside* `𝒩(S)` to shrink `P` losslessly; Ch 5 exploits FVA over `P`; Ch 6 dualizes the standard form and turns the emptiness/unboundedness facts of Sec 2.5 into Farkas and strong-duality certificates; Ch 7 attaches the `z` binaries via the `z_map_*` matrices assembled here.


## 3. Network compression

Compression is the single most consequential preprocessing stage for strain design. It runs twice in
the pipeline (`compress_model` at `compression.py:1853`, once before GPR integration and once after —
see the end-to-end flow in Ch 1 and the GPR boundary in Ch 4), and everything downstream — the FVA
passes, the dualization, the MILP, the enumeration loop — operates on the *compressed* network. This
chapter explains what compression removes, the exact integer/rational linear algebra that makes it
correct, and why the design is the way it is. The organising fact is that compression is **lossless
for the flux space**: it produces a smaller network whose steady-state flux cone is an exact linear
image of the original, so any minimal cut set found on the compressed network expands (Ch 9) to a
minimal cut set of the original. Nothing is approximated; that is the whole point, and it is why the
arithmetic must be exact.

### 3.1 Why compress at all

Strain design is solved as a mixed-integer linear program (MILP) with one binary intervention variable
`z_j ∈ {0,1}` per knockable reaction and a block of continuous constraint rows per module (Ch 6–7).
Two quantities dominate the difficulty of that MILP:

1. **The number of binaries.** Branch-and-bound explores a tree whose worst-case size is exponential
   in the count of binary variables. Each reaction that survives into the compressed model is a
   candidate `z`. Halving the reaction count is, very loosely, squaring the amount of pruning the
   solver must do to close the tree — the practical effect is far larger than the raw ratio suggests.
2. **The number of constraint rows.** Every module contributes a dualized or primal block whose height
   is proportional to the number of metabolites (dual variables, one per `S`-row) or reactions
   (primal fluxes). Fewer metabolites and fewer reactions means smaller, denser constraint blocks and
   fewer big-M / indicator couplings to relax.

Compression attacks both. It removes reactions (columns of `S`, hence binaries) by *merging* sets that
are forced to carry proportional flux, and it removes metabolites (rows of `S`, hence dual variables)
that are linearly dependent. On iML1515 the effect across passes is large and worth stating concretely:
the model enters compression with **2712 reactions**; the first compression pass (before GPR
integration) drives it down to roughly **1237 reactions**; after the GPR-extended network is built
(which re-introduces gene pseudoreactions) and the second compression pass runs, the working reaction
count settles around **2152**. The metabolite count falls in step through conservation-relation
removal. These are the numbers the MILP actually sees; the raw 2712 never reaches the solver.

There is a second, subtler payoff that the codebase treats as a working hypothesis. A stoichiometric
matrix `S ∈ ℝ^{m×n}` of a genome-scale model is far from full column rank: it has a large right
nullspace (the space of steady-state flux vectors, `dim = n − rank(S)`), and it has *conservation
relations* — left-null vectors `y` with `yᵀS = 0` — that make its rows linearly dependent. Coupled-
reaction merging collapses columns that are nullspace-proportional; conservation removal deletes
dependent rows. A well-compressed network has had both its column redundancy and its row redundancy
squeezed out, so it sits much closer to full rank than the original. The hypothesis noted in the code
is that as the compressed `S` approaches full rank, the strain-design MILP starts to resemble the
tighter, better-conditioned formulation used by MCS2 (Ch 11) — fewer degenerate LP relaxations, fewer
alternative optima to enumerate. This is a rationale for compressing *aggressively and exactly*, not
merely *enough*.

The correctness knife-edge that runs through the entire chapter: **compression decisions are made from
the kernel (nullspace) of `S`, and those decisions are combinatorial** — "is this kernel row exactly
zero?", "are these two kernel rows exactly proportional?". A floating-point kernel answers those
questions with a tolerance, and a tolerance is a guess. Guess wrong in either direction and you either
merge reactions that are not truly coupled (corrupting the flux space, producing *wrong* cut sets that
pass silently) or miss couplings that exist (leaving the MILP larger than necessary). The engine is
therefore built on exact integer/rational arithmetic end to end. §3.2 is that engine.

### 3.2 The exact integer/rational nullspace engine

#### 3.2.1 Why never float

Every compression primitive reduces to one linear-algebra question about `S`: *what is its kernel, and
which rows/columns are dependent?* Concretely:

- A reaction carries no steady-state flux ⇔ its **row in the kernel `K` is identically zero** (a
  structural blocked reaction).
- Two reactions are flux-coupled ⇔ their **kernel rows are scalar multiples of one another** with a
  *constant* ratio across every kernel column.
- A metabolite is redundant ⇔ its **row of `S` is a linear combination of other rows** (a conservation
  relation).

Each is an *exact* predicate: zero-vs-nonzero, equal-ratio-vs-not, dependent-vs-independent. Under
floating point every one of these becomes a threshold test `|x| < ε`. Choosing `ε` is choosing which
errors to make. Set it loose and two nearly-parallel-but-distinct reactions get merged — the merged
column is not a true consequence of `Sv=0`, so the compressed flux cone is *wrong*, and because the
MILP is built on the wrong cone, it can return "minimal cut sets" that do not cut anything in the real
model, with no error raised. Set `ε` tight and genuine couplings born from large stoichiometric
coefficients (see the 263-bit yeast-GEM case below) are missed. There is no safe `ε`, because the
coefficients that arise mid-elimination span many orders of magnitude. The project constraint is
therefore absolute: **the nullspace and rank computations are done in exact arithmetic — Python
arbitrary-precision integers and `fractions.Fraction` — and never in float.** `stoichmat_coeff2rational`
(`compression.py:1729`) converts every stoichiometric coefficient to an exact `Fraction`/`sympy.Rational`
before any compression math runs, and `float_to_rational` (`compression.py:46`) is the one controlled
place where a stray float coefficient is turned into a bounded-denominator rational (it first tries
`Fraction(val).limit_denominator(100)` and accepts it only if it round-trips to `max_precision`
decimals, else falls back to `round(val·10^p)/10^p`). Once inside the engine, no float ever appears.

#### 3.2.2 `RationalMatrix` and exact storage

The exact matrix type is `RationalMatrix` (`compression.py:106`). It stores a sparse rational matrix as
**two parallel integer sparse matrices** — a numerator CSR and a denominator CSR — so that entry
`(i,j)` is `num[i,j] / den[i,j]`. Keeping numerators and denominators as separate scipy `int64` CSR
matrices lets the common operations (column iteration, row/column deletion, submatrix extraction) stay
in fast compiled sparse code, while every value remains an exact rational. Construction paths:
`from_cobra_model` (`:175`) reads a model's coefficients straight into num/den arrays, preserving
`Fraction`/sympy-`Rational` exactly and only calling `float_to_rational` for genuine floats;
`identity` (`:144`), `from_numpy` (`:155`), and `_from_sparse` (`:130`) cover the rest.

Two features of `RationalMatrix` matter later:

- **`add_scaled_column`** (`:313`) performs `col[dst] += (num/den)·col[src]` in exact rational
  arithmetic with per-entry GCD reduction — this is the primitive that merges a coupled slave column
  into its master (§3.4).
- **Batch edit mode** (`begin_batch_edit`/`end_batch_edit`, `:270`/`:276`) switches the backing store
  to LIL for a burst of column mutations and back to CSR afterward, so a whole coupled-group merge
  does not pay repeated format-conversion costs.

#### 3.2.3 Fraction-free integer RREF: `_rref_integer_sparse`

The core is `_rref_integer_sparse` (`compression.py:484`), which computes a **reduced row-echelon
form over the integers** without ever introducing a denominator. The key idea is that for the purposes
we need (rank, pivot columns, and reading off a kernel), rows may be scaled by any nonzero integer:
scaling a row of `S` does not change its null vectors. So instead of dividing (which creates
fractions), the algorithm cross-multiplies and then *removes the common integer factor*.

**Setup — clear denominators once.** Each input row `r` has its rational entries `num/den` cleared to
integers by multiplying the whole row by the LCM of its denominators (`:527`–`:539`). After this every
working row is a pure integer row; there are no denominators to track for the rest of the routine —
this is the sense in which it is "fraction-free."

**Fraction-free elimination.** To eliminate pivot column `c` (pivot value `pv` in the pivot row) from a
target row with entry `ev` in column `c`, the update is

```
new_row[k] = ev_scaled · pivot[k] − pv_scaled · target[k]      (conceptually)
```

where the code (`_eliminate`, `:564`) first divides `pv, ev` by `g = gcd(pv, ev)` to get
`pv_scaled = pv/g`, `ev_scaled = ev/g`, then computes, for the sparse pivot row `prd`,
`new_row = {c: v·pv_scaled}` over the target row and subtracts `ev_scaled·prd[c]` on the shared
columns (`:583`–`:589`). This is the classical **fraction-free (Bareiss-style) update**: it keeps
everything integer and makes column `c` vanish in the target, because
`ev_scaled·pv − pv_scaled·ev = 0` after the GCD split.

**Content reduction (GCD) — why coefficients stay polynomial.** Cross-multiplying integer rows makes
entries grow. Without control, the bit-length of coefficients grows *exponentially* down the
elimination. The defence is to divide each freshly-computed row by the GCD of all its entries — its
"content" — right after forming it (`:592`–`:595`): `row_gcd = gcd(*new_row.values())` then
`row[c] //= row_gcd`. This is exactly the mechanism (Bareiss / fraction-free Gaussian elimination) that
bounds intermediate integers to the size of subdeterminants of the original matrix, i.e. keeps the
bit-length **polynomial** rather than exponential. A final content reduction of the pivot rows runs at
`:680`–`:686` as insurance.

**Markowitz pivoting — keep it sparse.** On a genome-scale `S` the elimination is dominated not by
arithmetic but by *fill-in* and *pivot search*. Two heuristics keep both small:

- Columns are pre-sorted by ascending nnz (`col_order`, `:510`–`:514`) so that sparse columns — the
  likely pivots — are visited first; rows are pre-sorted by ascending nnz (`:544`–`:546`). Results are
  translated back to the original column order at the end (`:688`–`:691`).
- At each step the pivot is chosen by the **Markowitz criterion** among the rows that actually contain
  the current pivot column: sparsest row first, ties broken by smallest absolute pivot value
  (`:628`–`:637`). A live `col_rows` index (`:554`–`:562`) maps each column to the set of active rows
  containing it, so pivot search visits only the handful of rows that hold the column instead of
  scanning all active rows (on iML1515 that scan was ~99.9% misses; the index removes it).

**Two-phase echelon, not full Gauss–Jordan.** Phase 1 (`:613`–`:650`) does forward elimination only —
each pivot is cleared from rows *below* it, leaving already-processed pivot rows sparse. Phase 2
(`:652`–`:679`) does back-substitution, processing pivots last-to-first and clearing each pivot column
from the pivot rows *above* it. Doing it in this order means that when a pivot row is applied during
back-substitution, its own later-pivot columns are already cleared, so back-substitution only ever
introduces *free-column* fill and only ever *removes* pivot-column entries — enabling the
`pivcol_holders` index (`:664`–`:668`) to be maintained with discards only. The commit comments record
the payoff on iML1515: ~0.8M back-substitution ops versus ~9.4M for naive Gauss–Jordan, because full
Gauss–Jordan re-reduces every filled row against every later pivot (~99% of the total work).

The routine returns `(rref_data, rank, pivot_columns)` where `rref_data[i]` is pivot row `i` as a
`{orig_col: integer_value}` dict, all in the original column space.

#### 3.2.4 Reading off the kernel: `_nullspace_sparse`

`_nullspace_sparse` (`compression.py:697`) turns the RREF into an explicit kernel basis. With `rank`
pivots and `cols` columns, the free columns are `free_cols = {0..cols−1} \ pivots` and the nullity is
`|free_cols|`. For each free column `f` the basis vector `k_f` is built by the standard RREF rule:

- entry `+1` at row `f` (the free variable is set to 1), `:726`–`:731`;
- at each pivot row `i` with pivot column `p_i`, entry `−rref[i,f] / rref[i,p_i]`, reduced by GCD to a
  clean rational and given a positive denominator (`:734`–`:749`).

So `k_f` has value `1` in its own free coordinate and `−(free entry)/(pivot value)` in each pivot
coordinate. By construction `S·k_f = 0` exactly. The set `{k_f}` is a sparse rational basis of the
right nullspace — one column per free variable — assembled by `_build_from_sparse_data` (`:206`). This
sparsity is exactly what makes coupling detection cheap in §3.3–§3.4: a coupled reaction shows up as a
kernel *row* with a distinctive zero pattern, and sparse kernel rows make that pattern comparison a
dictionary lookup.

`nullspace` (`:759`) is the public wrapper; `basic_columns` (`:774`) returns just the pivot columns
(used by conservation removal, §3.5); `sparse_nullspace` (`:785`) is the general-purpose exact-kernel
helper that accepts scipy/numpy/`RationalMatrix` input.

#### 3.2.5 The big-integer path — when subdeterminants exceed int64

Fraction-free RREF keeps coefficients polynomial, but "polynomial" is not "small". The exact kernel
entries are ratios of subdeterminants of `S`, and on dense, large models those subdeterminants can
exceed the 64-bit integers that scipy sparse matrices can hold. The verified extreme is **yeast-GEM,
whose exact nullspace needs coefficients up to ~263 bits** — far beyond int64.

The engine handles this transparently. `_INT64_MAX` (`:93`) and `_fits_int64` (`:96`) test whether all
numerators and denominators fit in signed int64. `_build_from_sparse_data` (`:206`) checks this: if
everything fits, it builds the fast dual-`int64`-CSR representation (`:214`–`:217`); if not, it falls
back to a **dict-of-`Fraction`s** store, `_dict_frac : {row: {col: Fraction}}` (`:218`–`:225`), which
uses Python arbitrary-precision integers and bypasses scipy entirely. `is_bigint` (`:407`) reports
which mode a matrix is in. The RREF itself never overflows — it works in Python `int` throughout; only
the *storage* of the finished kernel needs the fallback.

Because scipy sparse cannot hold >int64 values, the export helpers are mode-aware. `to_sparse_csr`
(`:382`) raises `OverflowError` in big-integer mode (with a message pointing at the exact exports).
`to_coo_exact` (`:412`) is the big-integer-safe export used in both modes: it returns an `ExactCOO`
namedtuple `(rows, cols, data, shape, denom)` (defined `:103`) in which entry `(rows[k], cols[k])`
equals `data[k]/denom` exactly, with `data` arbitrary-precision Python ints scaled to a common
denominator. `to_sparse_pattern` (`:435`) returns a pure-structure `int8` CSR (1s where nonzero) plus a
`{row: {col: Fraction}}` value map — this is the form coupling detection consumes, and it works
identically in int64 and big-integer mode, so the whole compression pipeline runs unchanged on
yeast-GEM. `sparse_nullspace` (`:785`) returns a scipy CSR in the common case and an `ExactCOO` when
`K.is_bigint()` (`:820`–`:823`).

### 3.3 The compression working state and the single-kernel pass

The nullspace-driven compressor is `StoichMatrixCompressor` (`compression.py:1089`), driven through a
mutable `_WorkRecord` (`:930`). The `_WorkRecord` carries three exact matrices that together record the
entire transformation and satisfy the invariant recorded on `CompressionRecord` (`:896`):

```
pre @ stoich @ post == cmp
```

with the flux-space consequence `v_original = post @ v_compressed`. Concretely `pre` is a
`RationalMatrix` starting as `identity(m)` (metabolite transformation, tracks row/metabolite
operations), `post` starts as `identity(n)` (reaction transformation, tracks column/reaction merges),
and `cmp` starts as a clone of `stoich` and is mutated in place as compression proceeds (`:930`–`:947`).
Every reaction merge is applied *identically to `cmp` and to `post`* so the invariant is preserved and
`post` can later expand a compressed flux vector back to the original reaction space (Ch 9).

The compress driver `StoichMatrixCompressor.compress` (`:1095`) runs a loop (`:1121`–`:1128`): remove
all-zero metabolite rows, then call `_nullspace_compress`, and re-iterate only while the previous pass
reported a *contradicting* removal (which changes the flux space and can expose new couplings). Note the
important design choice: **one nullspace computation drives both zero-flux detection and coupled-group
merging in the same pass.** `_nullspace_compress` (`:1133`) builds the active submatrix, computes
`kernel = nullspace(active)` once (`:1144`), extracts `(kernel_pattern, kernel_values)` via
`to_sparse_pattern` (`:1150`), and hands both to `_handle_compress` (`:1248`).

The single kernel yields three kinds of removals in one batch (`_handle_compress`, `:1248`–`:1337`):

1. **Structural zero-flux reactions** — reactions whose kernel *row is empty*. `_find_zero_flux`
   (`:1155`) reports reaction `reac` as zero-flux iff `kernel_pattern.indptr[reac] ==
   kernel_pattern.indptr[reac+1]`, i.e. the reaction appears in no null vector. Such a reaction cannot
   carry any steady-state flux (`Sv=0` forces `v_reac = 0`), so it can never be part of a working
   pathway and is deleted. This is the *structural* blocked-reaction test, and because it falls out of
   the kernel it needs no LP/FVA (contrast the bounds-based test in §3.6).
2. **Bounds-blocked reactions** — reactions with `lb = ub = 0` that nonetheless have a nonzero kernel
   row are added to the same removal set (`:1266`–`:1271`); they are structurally capable of flux but
   pinned to zero by bounds, so removing them here avoids a separate FVA pass.
3. **Coupled-group slaves (and contradicting groups)** — see §3.4.

Everything collected is removed in one `remove_reactions_by_indices` batch (`:1335`), which drops the
columns from `cmp` and `post` together and reindexes names/bounds (`:986`–`:1004`). `_handle_compress`
returns `True` only if a *contradicting* group was removed, which is the sole trigger for another
iteration.

### 3.4 Coupled / flux-coupled merge

#### 3.4.1 The math: why coupled reactions share a kernel pattern

Two reactions `a` and `b` are **fully (flux-)coupled** when, in *every* steady-state flux vector `v`
(every `v` with `Sv=0`), their fluxes are in a fixed ratio: `v_a = λ · v_b` for a constant `λ ≠ 0`
independent of `v`. Serial reactions in an unbranched pathway are the canonical example: if `A → B` and
`B → C` are the only producer and consumer of the intermediate `B`, then steady state on `B`
(`S`-row for `B` reads `v_{A→B} − v_{B→C} = 0`) forces `v_{A→B} = v_{B→C}` in *every* feasible flux
distribution.

The kernel expresses this directly. Let `K` be a basis of the right nullspace, so its columns span all
steady-state flux vectors and the *row* `K[a,:]` is the vector of coefficients of reaction `a` across
the basis directions. If `v_a = λ v_b` for every `v` in the span, then in particular it holds for each
basis column, so

```
K[a,:] = λ · K[b,:]        (row a is a constant multiple of row b, same λ in every column)
```

Two consequences, both used by the detector:

- **Same zero pattern.** `K[a,:] = λ K[b,:]` with `λ ≠ 0` implies `K[a,j] = 0 ⇔ K[b,j] = 0` — coupled
  reactions have *identical* kernel-row sparsity patterns. This is a cheap necessary condition: group
  candidate reactions by their kernel-row zero pattern (a hashable tuple of column indices).
- **Constant ratio.** Within a candidate group, the ratio `K[a,j]/K[b,j]` must be the *same* rational
  `λ` for every nonzero column `j`. If it drifts between columns, the rows are not proportional and the
  reactions are not fully coupled.

Both tests are exact equalities on rationals — which is precisely why §3.2's exactness is load-bearing.

#### 3.4.2 Detection: `_find_coupled_groups`

`_find_coupled_groups` (`compression.py:1164`) implements exactly that two-stage test. First it buckets
reactions by kernel-row zero pattern: `pattern = tuple(kernel_pattern.indices[start:end])` per reaction,
grouped into a dict, keeping only buckets of size > 1 (`:1181`–`:1188`). Then, within each candidate
bucket, it verifies the constant ratio (`:1201`–`:1244`): pick reaction `a`, take the first nonzero
column `first_col`, compute `ratio = a_val/b_val` there (exact `Fraction` division, `:1218`–`:1226`),
and confirm `a_v/b_v == ratio` for every remaining nonzero column (`:1230`–`:1235`). Reactions that
pass are collected into a group with `ratios[reac_b] = ratio` recorded per slave. The output is
`(groups, ratios)`: each group is `[master, slave1, slave2, …]` (master is the first member), and
`ratios[slave]` is the exact `Fraction` `v_master / v_slave`.

The `protected_indices` argument (`:1164`, applied at `:1202`/`:1211`) lets specific reactions be kept
out of any coupled group — the rest of the group still merges. This is how gene-controlled reactions
are held intact through COMPRESS #1 so that gene multiplicity survives into GPR integration
(cross-reference Ch 4); the mapping from protected *names* to current *indices* is done in
`_handle_compress` (`:1275`–`:1276`).

#### 3.4.3 The merge (COLUMN reduction): `_combine_coupled`

Merging is a column operation. `_combine_coupled` (`:1339`) folds each slave column into the master.
Given `ratios[slave] = v_master/v_slave = λ`, the master flux relates to the slave's own flux by
`v_slave = v_master/λ`, so the slave's stoichiometric contribution, expressed in units of the master
flux, is `col[slave] · (1/λ)`. The code computes the multiplier as `mult = 1/λ = λ.denominator /
λ.numerator` (`:1350`) and applies `cmp[:,master] += cmp[:,slave]·mult` and the *same* update to
`post[:,master]` (`:1353`–`:1356`), both via the exact `add_scaled_column`. Applying it to `post`
records that the compressed master reaction expands back to a specific exact linear combination of the
original columns — the master column of `cmp` becomes the exact stoichiometry of the lumped pathway,
and the master column of `post` becomes the exact expansion recipe. The slaves are then deleted
(`:1326`–`:1327`), so the group of `k` reactions becomes **one** reaction: `k−1` binaries eliminated per
group. This is a **column (reaction) reduction**.

**Worked micro-example.** Take the linear pathway `r1: A→B`, `r2: B→C`, `r3: C→D(ext)` with `A` supplied
and `D` drained, all irreversible. The only steady states have `v1=v2=v3`, so the kernel has one column
`(1,1,1)ᵀ` (up to scale) and all three kernel rows are identical → one coupled group `[r1, r2, r3]`
with `λ = 1` for both slaves. `_combine_coupled` adds `col(r2)` and `col(r3)` into `col(r1)`: the
intermediate metabolites `B` and `C` cancel (produced by one column, consumed by the next, in equal
units) and the merged column is the net reaction `A → D`. Three reactions, three binaries, collapse to
one. Now suppose instead `r2: 2 B → C` (two B per C). Steady state on `B` gives `v1 = 2 v2`, so
`ratios[r2] = v1/v2 = 2` and `mult = 1/2`: `col(r1)` gains `½·col(r2)`, again cancelling `B` exactly —
the constant ratio, carried as an exact `Fraction`, is what makes the cancellation exact.

#### 3.4.4 Bound intersection of a coupled group

Merging the columns is not the whole story: the slaves' flux *bounds* must be transferred to the
master, or the compressed model would silently drop feasibility restrictions. `_handle_compress`
(`:1289`–`:1327`) does this. Because `v_slave = v_master/λ` (with `λ = ratios[slave]`), the slave's
box `lb_s ≤ v_slave ≤ ub_s` becomes a constraint on `v_master`:

- if `λ > 0`: `lb_s·λ ≤ v_master ≤ ub_s·λ` (`:1302`–`:1305`);
- if `λ < 0`: the inequality flips, `ub_s·λ ≤ v_master ≤ lb_s·λ` (`:1306`–`:1309`).

with `±inf` propagated so that an unbounded slave contributes no restriction. The master's new box is
the **intersection** of its own box with all translated slave boxes: `intersected_lb = max(...)`,
`intersected_ub = min(...)` (`:1311`–`:1315`), written back to `work.bounds[master]` (`:1315`).

**Contradicting groups.** If the intersection is empty (`intersected_lb > intersected_ub`) or collapses
to a single point at zero (`intersected_lb == intersected_ub == 0`), the coupled group can carry no
nonzero flux in any steady state — a *contradicting* group. Then the master *and all slaves* are removed
(`:1317`–`:1323`) and `contradicting_removed` is set, which is the flag that triggers a re-iteration of
the whole pass (`:1337` → `:1126`): removing a contradicting group changes the flux space and may make
previously-uncoupled reactions coupled. A consistent (nonempty) group removes only the slaves
(`:1324`–`:1327`). This bound-intersection logic replaced a Java-era behaviour that could drop
reactions incorrectly; getting the translate-and-intersect direction right (especially the `λ<0` flip
and the `±inf` handling) is exactly the subject of the closed issue #44 cautionary tale in Ch 10.

### 3.5 Conservation-relation removal (ROW-rank reduction)

`remove_conservation_relations` (`compression.py:1419`) shrinks `S` by deleting **metabolite rows that
are linearly dependent** on the others — the *conservation relations* of the network. A conservation
relation is a left-null vector `y` with `yᵀS = 0`: a weighted sum of metabolite balances that is
identically zero (e.g. a moiety like total ATP+ADP, or a redundant compartment balance). If row `i` of
`S` is a linear combination of other rows, then the steady-state equation `S_i · v = 0` is *implied* by
the others and carries no information — dropping metabolite `i` leaves the flux space `{v : Sv=0}`
exactly unchanged. It is therefore lossless for fluxes, and it strictly reduces the row count.

The mechanics use the exact RREF as a rank/independence oracle. The function builds `Sᵀ` (reactions ×
metabolites) directly from the cobra coefficients as a `RationalMatrix` — deliberately transposed so
that *metabolites become columns* (`:1428`–`:1455`) — and calls `basic_columns` (`:1456`), which runs
`_rref_integer_sparse` and returns the pivot columns. The pivot columns of `Sᵀ` are a maximal set of
**linearly independent metabolite rows**; every non-pivot metabolite is a dependent row, i.e. a
conservation relation. Those dependent metabolites are removed from the model (`:1458`–`:1460`).

Two design points. First, this is a **row-rank reduction**, complementary to the column reduction of
§3.4 — together they push `S` toward full rank (the §3.1 hypothesis). Second, the *ordering* matters:
conservation removal runs *before* the expensive coupled step in each cycle (`compress_model`,
`:1906`–`:1910`). Fewer metabolite rows means the nullspace RREF that drives coupling detection operates
on a smaller matrix, so removing dependent rows first makes the costliest stage cheaper. (There is a
legacy Java oracle, `_remove_conservation_relations_java` at `:1943`, selectable via the
`efmtool_rref` backend; the default `sparse_rref` path uses the pure-Python exact RREF above.)

### 3.6 Blocked and zero-flux removal

There are two distinct notions of "carries no flux," removed at two points:

- **Bounds-blocked reactions** — `remove_blocked_reactions` (`compression.py:1699`) deletes reactions
  whose bounds are exactly `(0, 0)` (`:1701`) with `remove_orphans=True` so metabolites left dangling
  go too. This runs once at the very start of `compress_model` (`:1889`), before any rational
  conversion, as a cheap first cut.
- **Structural zero-flux reactions** — reactions whose *kernel row is empty* (§3.3, `_find_zero_flux`,
  `:1155`). These are reactions that `Sv=0` forces to zero regardless of bounds; they are found for
  free from the nullspace during each coupled pass and removed in the same batch. The additional check
  at `:1266`–`:1271` catches reactions pinned to `(0,0)` by bounds that still have a nonzero kernel row,
  folding the bounds-blocked case into the kernel pass as well.

`remove_unused_metabolites` (`_WorkRecord`, `:1044`) is the row-side companion: after columns are
dropped, any metabolite row that has become all-zero (detected in O(m) via CSR `indptr` diffs,
`:1054`–`:1055`) is removed. It runs at the top and bottom of the compress loop (`:1124`, `:1129`).

### 3.7 The alternating fixpoint

`compress_model` (`compression.py:1853`) orchestrates the three reducers into an **alternating
fixpoint** (`:1894`–`:1937`). The order within each cycle is deliberate:

1. **Parallel merge** (`compress_model_parallel`, §3.8) — cheapest: a hash of the (scale-normalized)
   stoichiometry row, no RREF (`:1899`).
2. **Conservation-relation removal** (§3.5) — shrinks `S`'s rows so the next step's RREF is smaller
   (`:1906`–`:1910`).
3. **Coupled merge** (`compress_model_coupled`, §3.4) — most expensive: a full exact nullspace/RREF
   (`:1920`–`:1935`).

The loop runs cheap-to-expensive so that each stage feeds the next a smaller network, and the expensive
kernel computation only ever runs on an already-thinned matrix.

**Why it alternates and why it terminates.** Each reducer can *expose* new opportunities for the
others: a coupled merge cancels intermediate metabolites, which can make two previously-different
columns become exactly parallel (new parallel merges); a parallel merge changes the column set, which
can change the kernel (new couplings); conservation removal changes the row set likewise. So a single
pass of each is not enough — the pipeline loops. Termination is guaranteed because **every reducer
only ever removes reactions or metabolites; none ever adds one.** The reaction count is a non-negative
integer that is non-increasing across the loop, so it cannot decrease forever. The explicit stop
condition (`:1916`–`:1918`) is: after at least one full cycle, if *either* the parallel step or the
coupled step found nothing, stop — because a step that changed nothing on the current network will
change nothing on re-run unless the *other* step alters the network, and the loop has just established
that it did not make progress. `run` counts cycles for the log. In practice on genome-scale models this
converges in a handful of cycles.

Each productive step appends a record to `cmp_mapReac` — `{"reac_map_exp": reac_map_exp, "parallel":
<bool>}` (`:1904`, `:1935`) — the compression map consumed by decompression (§3.10).

### 3.8 Parallel merge

`compress_model_parallel` (`compression.py:2025`) is the cheap reducer. It lumps **parallel reactions**:
reactions that are stoichiometric scalar multiples of one another (identical up to a rational scale
factor) *and* have compatible bound topology, e.g. two isozymic reactions with the same net conversion.
It never computes a kernel — it groups reactions by an exact hashable key.

**Scale-invariant, exact key.** The stoichiometry matrix is taken transposed (`stoichmat_T`, one row
per reaction) and each reaction's key (`_parallel_key`, `:2058`) is its stoichiometry row **normalized
by its first nonzero coefficient in exact rational arithmetic**: `f0 = float_to_rational(vals[0])`, then
`stoich = tuple((col, float_to_rational(v)/f0) …)` (`:2062`–`:2064`). Normalizing by the first
coefficient makes the key **scale-invariant**: `−1 A → 2 B` and `−3 A → 6 B` both reduce to the tuple
`((A,1),(B,−2))` and so share a key, but the division is exact (`Fraction`), so two rows that are only
*nearly* proportional get *different* keys — no reaction is ever merged on a rounding coincidence.

**Bound topology is part of the key.** The key also carries three bound-derived flags per reaction,
computed at `:2048`–`:2051`:

- `fwd`/`rev`: whether the reaction is unbounded in the forward / reverse chemical direction (an `inf`
  bound on the appropriate side given the sign of the first coefficient);
- `inh`: set to the *unique* value `i+1` if the reaction has any finite nonzero bound
  (`not ((ub inf or 0) and (lb inf or 0))`), else `0`.

Because `inh` is `i+1` (unique per reaction), **any reaction carrying a finite nonzero bound gets a key
component no other reaction can match, so it is never lumped in parallel.** Parallel merging is thereby
restricted to reactions whose bounds are homogeneous (each side `0` or `±inf`) and whose reversibility
matches — i.e. reactions that live in the same cone face. This is the correctness guard that keeps
parallel merging from combining reactions with incompatible feasibility. Grouping is a hash pre-filter
(`key_hashes`) followed by an exact full key comparison (`:2073`–`:2085`); `protected_rxns` are forced
into singleton groups (`:2076`–`:2078`).

**COLUMN reduction and the flux-split map.** Each group keeps one representative (its id is decorated
with `*`-joined member ids, truncated to `...` past ~220 chars, `:2094`–`:2097`) and the others are
removed (`:2114`–`:2116`) — again a **column reduction**, `k−1` binaries removed per group. The
compression map differs from the coupled case in a way that matters for cost accounting: for a parallel
group the *compressed* flux is the **total** flux through all members, and each member's share is
proportional to its stoichiometric scale `|factor[j]|` (its first-coefficient magnitude). The map is
built (`:2127`–`:2141`) as normalized flux-split fractions:

```
rational_map[cmp_id][orig_j] = |factor[j]| / Σ_k |factor[k]|      (fractions sum to 1)
```

So expanding a compressed flux of a parallel group distributes it across the originals in these exact
proportions, whereas expanding a coupled group scales by the `post`-column factors (`v_orig = coeff ·
v_cmp`). This is why cost propagation (Ch 9's `compress_ki_ko_cost`) treats the two directions oppositely
— KO cost of a parallel lump is the *sum* over members (you must knock out all parallel routes), while
KO cost of a serial/coupled lump is the *min* (knocking out any one link breaks the chain).

**Worked micro-example.** Two isozymes `r1: A→B` and `r2: A→B` (same stoichiometry, both irreversible
with `ub=inf`, `lb=0`): identical normalized key `((A,1),(B,−1))`, matching `fwd/rev/inh=0` flags → one
parallel group `[r1,r2]`. They lump into a single reaction `r1*r2: A→B` carrying `v = v1+v2`, with
flux-split map `{r1: ½, r2: ½}` (equal `|factor|`). A knockout of the lump means *both* isozymes are
knocked out, so its KO cost is the sum — correctly capturing that either isozyme alone still runs the
reaction.

### 3.9 GPR propagation through compression

When compression runs with `propagate_gpr=True` (COMPRESS #1, before gene pseudoreactions exist), each
merge must carry the Boolean gene–protein–reaction (GPR) rules of its members onto the surviving
reaction, so the compressed model still knows which genes control the lumped reaction. This chapter
covers *only the propagation through a merge*; the semantics of encoding GPR as flux structure belongs
to Ch 4 (`extend_model_gpr`), cross-referenced there.

The rule follows the flux logic of each merge type:

- **Serial / coupled merges → AND.** A coupled group is an unbranched chain that must run as a unit —
  every member's genes are required for the lumped reaction to carry flux — so their GPRs are combined
  with **AND**. `_combine_gpr_and` (`compression.py:1802`) is invoked from `compress_model_coupled`
  (`:2007`–`:2015`) over the saved GPR ASTs of the contributing reactions.
- **Parallel merges → OR.** Parallel members are alternative routes for the same conversion — *any* of
  them suffices — so their GPRs are combined with **OR**. `_combine_gpr_or` (`compression.py:1825`) is
  invoked from `compress_model_parallel` (`:2107`–`:2121`).

Both combiners lift the cobra GPR AST to sympy Boolean expressions (`_gpr_ast_to_sympy`, `:1754`),
combine with `sympy.And`/`sympy.Or` (which auto-flatten and dedupe), and render back to a rule string
(`_sympy_to_gpr_string`, `:1773`). The subtlety is the treatment of an **empty GPR** (a reaction with
no gene requirement, "always active", logically `True`): in an AND-combine an empty GPR is a no-op and
is skipped, and if *all* members are empty the result is empty (`:1815`–`:1822`); in an OR-combine a
single empty member makes the whole lump always-active, so the result is empty (`:1837`–`:1839`). Full
Boolean simplification is deferred to `reduce_gpr` downstream (Ch 4). Note also that the coupled Python
backend clears gene rules on the raw reactions before the merge (`compress_model_coupled`, `:1996`–
`:1998`) and reinstates the combined rule afterward from the *saved* ASTs (`:1982`–`:1983`,
`:2007`–`:2015`), so the propagation is driven off a clean snapshot rather than the mutated model.

### 3.10 The compression map `cmp_mapReac` and back-expansion

The output of `compress_model` is `cmp_mapReac`: an **ordered list of step records**, one per productive
merge, in the order the merges were applied. Each record is a dict

```
{"reac_map_exp": {compressed_id: {original_id: factor, …}, …}, "parallel": bool}
```

where `factor` is an exact `Fraction` — a normalized flux-split share for a parallel step, or a
`post`-column scaling coefficient for a coupled step (§3.4/§3.8). `parallel` records which merge
produced the step, because cost and constraint propagation treat the two directions oppositely (§3.8).

Because compression is iterative, a compressed id in step `t` may itself be an *original* id inside step
`t+1` — the maps **compose**. Back-expansion therefore walks the list and composes the per-step maps.
Forward composition to a single flat lookup is `_build_cmp_reverse_map` (`networktools.py:515`), which
threads original ids through intermediate compressed ids to a final `{original_id: final_compressed_id}`
table; full solution decompression walks `cmp_mapReac` **in reverse** (`estimate_expansion_size` reverses
it at `networktools.py:1430`, and `expand_sd` composes the reverse maps) to turn a compressed
intervention set back into original-reaction interventions, re-injecting the flux-split/scaling factors
at each step. The complete decompression semantics — expanding a knockout of a lumped reaction into the
correct combination of original knockouts, handling parallel-vs-serial multiplicity, size-1 MCS
re-injection, and gene translation — are owned by Ch 9; this section only fixes the structure of the
map that Ch 9 consumes.


### 3.11 The legacy efmtool (Java) backend

Everything in §3.2–§3.10 describes the **default** compression engine: the pure-Python, exact
integer/rational `sparse_rref` backend. That engine is a *reimplementation*. The original backend —
and the one every pre-1.15 release actually ran — was **efmtool**, Marco Terzer's Java tool for
elementary-flux-mode enumeration and network compression (the compression stage of efmtool is exactly
the coupled/zero/contradicting reduction that §3.4/§3.6 now do in Python). It is still shipped and
still reachable, selected with `compression_backend='efmtool_rref'`, and this section documents how the
bridge works and *why* it has been demoted to legacy. Reading it also explains the vocabulary the
Python code inherited: the Python `CompressionMethod` enum (`compression.py:831`), the Python class
name `StoichMatrixCompressor` (`compression.py:1089`), and the `CoupledZero`/`CoupledCombine`/
`CoupledContradicting` method names are all deliberate echoes of the efmtool Java API they replaced.

#### 3.11.1 What efmtool is and how straindesign reaches it

efmtool is a Java library (namespace `ch.javasoft.*`, packaged as `efmtool.jar` alongside the Python
sources at `straindesign/efmtool.jar`). straindesign uses only its *compression* half — not its EFM
enumeration — through the classes loaded in `efmtool_cmp_interface.py:167`–`:179`:
`ch.javasoft.smx.impl.DefaultBigIntegerRationalMatrix` (an arbitrary-precision rational matrix),
`ch.javasoft.smx.ops.Gauss` (rational Gaussian elimination), `ch.javasoft.metabolic.compress.
StoichMatrixCompressor` and `CompressionMethod`, and `ch.javasoft.math.BigFraction` /
`java.math.BigInteger`. The bridge is **JPype**: `_start_jvm` (`efmtool_cmp_interface.py:93`) starts an
in-process JVM, adds `efmtool.jar` to the classpath, and imports the Java classes via
`jpype.imports` so they become callable Python objects.

The routing has three layers.

1. **Import time.** `__init__.py:51`–`:53` calls `_start_jvm()` *eagerly* at `import straindesign`.
   This is a no-op when jpype1 or a JVM is absent (neither is a package dependency), so a normal install
   never touches Java. When Java *is* present the JVM must be started here — before NumPy/OpenBLAS spins
   up worker threads — or JNI calls later crash with SIGBUS/SIGSEGV (`__init__.py:46`–`:50`; the code is
   littered with such mitigations, see §3.11.4).
2. **Backend selection.** `compute_strain_designs` reads the kwarg
   `compression_backend = kwargs.get('compression_backend', 'sparse_rref')`
   (`compute_strain_designs.py:354`) and threads it into both `compress_model` calls
   (`:357`–`:360`, `:435`). `compress_model` sets `use_java = (compression_backend == 'efmtool_rref')`
   (`compression.py:1887`).
3. **Dispatch inside the fixpoint.** Crucially, `efmtool_rref` does **not** replace the whole
   compression pipeline — only two of its three reducers. Inside the alternating fixpoint (§3.7,
   `compression.py:1894`–`:1937`):
   - **Parallel merge** (step 1, §3.8) is **always** the Python hash-based `compress_model_parallel` —
     efmtool has no equivalent and it is never routed to Java.
   - **Conservation removal** (step 2, §3.5) forks on `use_java` (`:1907`–`:1910`): Java goes through
     `_remove_conservation_relations_java` (`:1943`), Python through `remove_conservation_relations`.
   - **Coupled merge** (step 3, §3.4) forks inside `compress_model_coupled` (`:1985`): Java calls
     `compress_model_java` (`efmtool_cmp_interface.py:367`), Python calls `compress_cobra_model`.

   So `efmtool_rref` is really a **hybrid**: Python parallel-merge + Java conservation-removal + Java
   coupled-merge, iterated by the same Python fixpoint driver. The two backends differ only in the
   *nullspace/rank algorithm* used for steps 2 and 3.

#### 3.11.2 Data marshalling: cobra model → Java → cobra model

The coupled step, `compress_model_java` (`efmtool_cmp_interface.py:367`), is where the interesting
marshalling lives. It mutates the cobra model in place and returns the same
`{compressed_id: {orig_id: factor}}` reaction map that the Python backend produces, so the rest of the
pipeline (module remapping, cost compression, decompression in Ch 9) is backend-agnostic.

**Into Java.**
- `stoichmat_coeff2rational(model)` (`:387`) first converts every stoichiometric coefficient to an
  exact `Fraction`/sympy-`Rational` — the same exactness discipline as §3.2.1, done *before* any Java
  call.
- All gene rules are cleared, `r.gene_reaction_rule = ''` (`:389`), matching the Python coupled path
  (§3.9); GPR is re-attached afterward (below).
- A `DefaultBigIntegerRationalMatrix(num_met, num_active)` is allocated (`:407`) and filled column by
  column. Reactions whose upper bound is `≤ 0` are **flipped** to the forward direction
  (`model.reactions[mi] *= -1`, `:412`–`:415`) and their index recorded in `flipped`; efmtool's
  compressor assumes a canonical orientation. Each coefficient `v` is converted by
  `sympyRat2jBigIntegerPair` (`:285`) into a Java `BigInteger` numerator/denominator pair — using
  `BigInteger.valueOf` for values that fit in 63 bits and `BigInteger(str(...))` otherwise — and set as
  a `BigFraction(n, d)` (`:416`–`:418`). This path is **exact**: efmtool's `DefaultBigIntegerRational
  Matrix` is arbitrary-precision, so the Java core does *not* overflow.
- A `StoichMatrixCompressor(subset_compression)` is built, where `subset_compression =
  [CoupledZero, CoupledCombine, CoupledContradicting]` (`:181`–`:183`): remove structurally
  zero-flux reactions, combine coupled groups, and drop contradicting groups — the Java analogues of
  §3.3's three removal kinds. `smc.compress(stoich_mat, reversible, …, reacNames, None)` (`:423`)
  returns a `comprec` whose `post` matrix is the reaction transformation (the Java counterpart of the
  Python `post` in §3.3, `v_original = post · v_compressed`).

**Back to Python.** Here is the seam that matters for correctness:

```python
subset_matrix = jpypeArrayOfArrays2numpy_mat(comprec.post.getDoubleRows())   # :424 — DOUBLES
```

The *structure* of the compression (which original reaction maps into which compressed column, and the
zero pattern) is read back as a **double-precision** numpy matrix via `getDoubleRows()`. The
per-reaction merge then:
- flags a reaction zero-flux iff its `subset_matrix` row is all-zero (`:432`–`:434`);
- for each compressed column `j`, gathers members from `subset_matrix[:,j].nonzero()` (`:437`), scales
  each member's stoichiometry by the **exact** factor `jBigFraction2sympyRat(comprec.post.
  getBigFractionValueAt(ai, j))` (`:445`–`:446`, exact `BigFraction → sympy.Rational`), and **rescales
  its bounds by `/= abs(subset_matrix[ai, j])`** (`:447`–`:450`, i.e. by a **double**);
- merges member reactions into the group representative, concatenating ids with `*` and truncating past
  ~220 chars to `...` (`:456`–`:467`) — the same naming convention as the parallel backend (§3.8);
- records `subset_rxns`/`subset_stoich` per representative (negating the stoich for `flipped`
  reactions, `:452`–`:455`) and finally assembles `rational_map` from them (`:493`–`:499`).

So the *factors* are exact rationals, but the *pattern detection and the bound rescaling* pass through
double precision. The `suppressed_reactions` argument (`:367`, `:392`) — reaction ids that must survive
because a strain-design module references them — are excluded from the active set entirely and re-added
as standalone identity entries (`:480`–`:485`), a workaround for efmtool's `CoupledContradicting` step,
which will otherwise delete reactions it deems inconsistent (contrast the Python backend, which keeps
them via the exact bounds-intersection of §3.4.4). Back in `compress_model_coupled` the Java branch
then sweeps up any leftover `(0,0)` reactions (`compression.py:1990`–`:1994`) and — identically to the
Python branch — re-attaches the **AND-combined GPR** from the pre-merge snapshot
(`compression.py:2007`–`:2015`). GPR propagation is therefore the *same* for both backends on the
coupled step.

**The conservation path.** `_remove_conservation_relations_java` (`compression.py:1943`) builds `S` as
a LIL matrix, **densifies its transpose** (`stoich_mat.transpose().toarray()`, `:1947`), and hands it
to `basic_columns_rat_java` (`efmtool_cmp_interface.py:332`). That function wraps the dense array into a
`DefaultBigIntegerRationalMatrix` via `numpy_mat2jpypeArrayOfArrays` — which builds a **`JDouble[rows,
cols]`** (`:267`) — then runs `Gauss.getRationalInstance().rowEchelon(...)` (`:360`) and returns the
pivot columns, i.e. the independent metabolite rows; the non-pivot metabolites are dependent
(conservation relations) and removed (`compression.py:1948`–`:1950`). This is the exact-RREF
independence oracle of §3.5, but computed in Java — and note it marshals the stoichiometry through a
**dense double** array, both memory-heavy on genome-scale models and lossy for large coefficients.

#### 3.11.3 Why it is legacy

The pure-Python `sparse_rref` engine (§3.2) was written to replace efmtool for four concrete reasons,
each a decisive advantage on a genome-scale correctness/performance workload:

1. **No JVM / JPype dependency.** efmtool needs a JVM, the `efmtool.jar`, `jpype1`, and `sympy` all
   present and version-compatible (`_init_java`, `efmtool_cmp_interface.py:192`, raises `ImportError`
   for any missing piece). The Python backend needs only NumPy/SciPy, which straindesign already
   depends on. A default that requires a working Java toolchain is a default that fails on many
   installs.
2. **Native-crash fragility.** The bridge is defensive to a degree that itself signals the risk:
   eager JVM startup ordered before OpenBLAS threads (§3.11.1); `gc.disable()` wrapped around *every*
   JNI block (`efmtool_cmp_interface.py:358`–`:363`, `:404`–`:426`) because Python's garbage collector
   finalizing a JPype proxy mid-call causes Bus error / SIGSEGV; an `atexit` JVM-shutdown hook to dodge
   a JPype teardown race (`:150`–`:158`). None of this can occur in a pure-Python engine.
3. **Big-integer safety at the interface.** efmtool's Java core is arbitrary-precision (`DefaultBig
   IntegerRationalMatrix`), so the *internal* arithmetic does not overflow. The hazard is at the
   **marshalling boundary**: the compression structure and bound rescaling are read back through
   `getDoubleRows()` and `abs(subset_matrix[...])` in double precision (§3.11.2), and conservation
   removal pushes `S` through a dense `JDouble` array. On models whose exact subdeterminants are huge —
   the verified extreme is **yeast-GEM, needing ~263-bit coefficients** (§3.2.5) — a double cannot
   represent those magnitudes, so bound rescaling and pattern detection silently lose precision. The
   Python engine keeps *everything* in Python big integers / `Fraction` end to end and switches to a
   dict-of-`Fraction` store above int64 (§3.2.5), so it is exact even on yeast-GEM. This is the single
   most important reason the Python path is the default.
4. **It is the default and the tested path.** The measured pipeline numbers (§3.1, and the iML1515
   timings in CONTEXT) are all on `sparse_rref`; that is the code that receives ongoing correctness
   work (e.g. the bounds-intersection fix of §3.4.4 / issue #44).

**The trade-off, honestly stated.** efmtool is not bad code — it is a mature, well-tested Java library
whose fraction-free rational Gauss elimination is fast compiled code, and for a decade it *was* the
compression engine for this and related tools. If you have a JVM handy and a model whose coefficients
stay comfortably inside double range, `efmtool_rref` will produce a correct compression at competitive
speed. Its costs are the heavy dependency stack, the native-crash surface, and the double-precision
marshalling seam. Given a pure-Python alternative that is exact to arbitrary precision, needs no JVM,
and is the maintained default, the Java backend earns its "legacy" label: **there is essentially no
production reason to select it.** The realistic remaining uses are (a) cross-validation — regression-
testing the Python engine's output against the historical efmtool result on a model both can handle —
and (b) a fallback if a bug were ever found in the Python RREF. For everyday strain design, leave
`compression_backend` at its default.

#### 3.11.4 Behavioral differences to be aware of

The two backends are *intended* to produce the same lossless flux-space compression, but they are not
byte-identical and a few divergences are worth knowing:

- **GPR propagation is identical on the coupled step.** Both backends clear gene rules before merging
  and re-attach the AND-combined GPR from the saved AST snapshot in `compress_model_coupled`
  (`compression.py:2007`–`:2015`), and the parallel OR-combine is always the Python
  `compress_model_parallel` (§3.9). So GPR handling does *not* diverge between backends.
- **Protected reactions are honored only by the Python backend.** `compress_model` passes gene-
  controlled reactions as `protected_reactions` (`no_coupled_compress_reacs`, `compression.py:1923`–
  `:1925`) so they survive COMPRESS #1 un-merged and gene multiplicity is preserved for GPR
  integration (§3.4.2, Ch 4). `compress_model_java` **ignores `protected_reactions`** — it reads only
  `suppressed_reactions`, which `compress_model` never populates on this path. On the Java backend those
  reactions can therefore be lumped in COMPRESS #1, a genuine semantic divergence in the gene-KO
  pipeline.
- **Contradicting groups are handled differently.** efmtool's `CoupledContradicting` deletes groups it
  finds inconsistent (the reason `suppressed_reactions` exists as a shield). The Python backend instead
  computes the exact **bounds intersection** of the coupled group and removes only genuinely
  empty/zero groups (§3.4.4). This is precisely the logic whose Java-era version "could drop reactions
  incorrectly" — the cautionary tale of closed issue #44 (Ch 10). The two backends can thus disagree on
  which reactions a contradicting group costs you.
- **Direction bookkeeping differs.** The Java path physically flips `ub ≤ 0` reactions (`*= -1`) and
  negates their recorded stoich (`efmtool_cmp_interface.py:412`–`:415`, `:452`–`:455`); the Python
  coupled backend carries sign inside the exact `ratios` (§3.4.3). Same flux space, different maps —
  which is fine because decompression (Ch 9) consumes whichever map its backend produced.
- **Bound rescaling precision.** Java rescales merged-reaction bounds by a **double**
  (`efmtool_cmp_interface.py:447`–`:450`); the Python backend intersects bounds using exact rationals
  (§3.4.4). On well-scaled models this is invisible; on large-coefficient models it is another place the
  Java path can drift.

The safe reading: `efmtool_rref` is preserved for provenance and cross-checking, exercises the same
fixpoint and produces the same *kind* of map, but the exact-arithmetic Python backend is the one whose
compression you should trust for correctness-sensitive strain design.


## 4. GPR integration

Strain design asks a question posed in **gene** space — "which genes do I delete so that the cell
can no longer do X but can still do Y?" — but the machinery that answers it, the MILP built in later
chapters, lives entirely in **flux/reaction** space. Its variables are fluxes `v ∈ ℝⁿ` constrained by
`S·v = 0` and bounds, and its binary intervention variables `z` toggle *reactions* on and off (Ch 6,
Ch 7). A gene is not a reaction. A gene influences a reaction only indirectly, through a Boolean
**Gene–Protein–Reaction (GPR) rule**: reaction `PFK` might carry the rule `pfkA or pfkB`, meaning the
reaction can run if *either* isozyme's gene is expressed. Knocking out one gene of an `or` does
nothing to the flux; knocking out one gene of an `and` (a required subunit of an enzyme complex)
kills the reaction. The relationship between "genes deleted" and "reactions disabled" is a Boolean
function, not a simple map.

This chapter explains how `straindesign` makes gene knockouts expressible inside the same flux MILP.
The central idea — `extend_model_gpr` (`networktools.py:946`) — is to **compile each Boolean GPR rule
into auxiliary flux structure**: extra pseudo-metabolites and pseudoreactions bolted onto the
stoichiometric matrix, arranged so that the linear steady-state constraints reproduce exactly the
Boolean logic. After extension, "gene *g* is knocked out" becomes the purely linear statement "fix the
flux of pseudoreaction *g* to zero," and the MILP's existing reaction-knockout machinery handles it
with no separate Boolean-logic layer. We then cover the reversible-reaction split that GPR extension
forces (`extend_model_gpr` + the `reac_map` remap in `compute_strain_designs.py:397–410`), the
pre-pruning pass `reduce_gpr` (`networktools.py:664`) that shrinks the work, the delicate ordering of
the two compression passes around extension (`compute_strain_designs.py:331–440`), and the sha256 name
truncation that only fires for Gurobi/GLPK.

### 4.1 Why encode gene logic as flux structure at all

There are two ways to let a reaction-space MILP reason about gene knockouts.

**Alternative A — post-hoc gene→reaction mapping.** Solve the strain-design problem in reaction space
as usual, producing reaction-knockout sets; then, *after the fact*, translate each reaction KO back
to the genes that could cause it via the GPR rules. This is what the *decompression*/solution-translation
step does in the reverse direction for reporting (Ch 9). But using it as the *only* gene mechanism is
wrong for optimization, for three reasons:

1. **The cost model is gene-based, not reaction-based.** A minimal *gene* cut set minimizes the number
   (or cost) of *genes* deleted. One gene can disable several reactions (pleiotropy); several genes may
   need deleting to disable one reaction (an `and` of subunits). A minimal *reaction* cut set optimizes
   the wrong objective and its cardinality does not correspond to any achievable set of gene deletions.
2. **`and`/`or` structure creates feasibility that reaction-space cannot see.** To disable a reaction
   guarded by `g1 and g2`, deleting *either* gene suffices — so the "cost" of killing that reaction is 1
   gene, and the choice of which gene is itself a decision the optimizer should make (it may reuse `g1`
   to also kill another reaction). To disable a reaction guarded by `g1 or g2`, you must delete *both*
   genes — cost 2, and only that exact pair works. A reaction-level KO variable cannot represent "this
   reaction dies iff this particular Boolean combination of shared gene variables is all-off."
3. **Genes are shared across reactions.** The same gene appears in many reactions' rules. A correct
   gene-MCS must count a shared gene once and account for *all* of its downstream reaction effects
   simultaneously. Post-hoc mapping, done per reaction, double-counts or misses these couplings.

**Alternative B — a separate Boolean-constraint layer.** Add binary gene variables `y_g` and, for each
reaction, a logical constraint `v_r = 0` implied by the Boolean rule over the `y_g`. This is *correct*
but forces the MILP to carry two coupled logic systems — the flux LP and a Boolean CNF/DNF layer with
its own indicator or big-M linearizations of every `and`/`or` — roughly doubling the modeling surface
and the constraint count, and requiring bespoke code to linearize arbitrary nested Boolean trees.

**The chosen design — encode the Boolean rule *as flux*.** `straindesign` instead embeds the Boolean
function directly into the stoichiometry `S`, so the LP's own `S·v = 0` mass-balance *is* the Boolean
logic. No gene binaries, no second logic layer: a gene knockout is literally a reaction (pseudoreaction)
knockout of the same kind the MILP already handles, so the entire dualization/`link_z` machinery (Ch 6,
Ch 7) applies unchanged. The price is a modest number of extra rows/columns in `S` (one pseudoreaction
per surviving gene, plus one pseudo-metabolite/pseudoreaction per Boolean operator), which the second
compression pass (§4.5) then partly reabsorbs. The correctness guarantee that makes this legal is that
**the extension does not change the reachable flux space of the original reactions** (§4.3): all the new
structure is "upstream plumbing" whose only effect, when a pseudoreaction is fixed to zero, is to force
the guarded reactions to zero exactly when the Boolean rule says the enzyme is absent.

### 4.2 `extend_model_gpr`: turning a rule into pseudo-metabolites and pseudoreactions

`extend_model_gpr(model, use_names=False)` (`networktools.py:946`) walks each reaction's GPR abstract
syntax tree (cobra parses the rule string into `reaction.gpr.body`, an `ast.BoolOp`/`ast.Name` tree)
and materializes it as network structure. The design has one **supply** primitive (a gene) and two
**combinator** primitives (`and`, `or`), each realized by a small stoichiometric gadget. Every gadget
obeys the same invariant: it *produces* a pseudo-metabolite that represents "this sub-expression is
TRUE (its enzyme/gene product is available)," and the guarded reaction is finally made to *consume*
one unit of the top-level pseudo-metabolite, so it can carry flux only if that metabolite can be
supplied.

Throughout, the pseudo-metabolites are abstract tokens — they have no physical units and appear in no
other balance except the gadget that defines them. The bounds on the gene pseudoreactions are `[0, ∞)`
(a one-directional source), so a gene "product" can be supplied in unlimited quantity but never
consumed negatively.

#### The gene gadget (leaf / `ast.Name`)

`create_gene_pseudoreaction(gene_id)` (`networktools.py:1021`) does, for gene `g`:

- create a pseudo-metabolite `g_{gene_id}` (e.g. `g_b0727`);
- create a pseudoreaction whose id is the gene id (or gene name if `use_names=True`) with reaction
  `--> g_{gene_id}` and `upper_bound = ∞`, `lower_bound = 0` (the default for a product-only reaction).

So the gene pseudoreaction is an unbounded *source* of the gene's token. Symbolically, if `w_g ≥ 0` is
the flux of gene *g*'s pseudoreaction, it contributes `+w_g` to the balance row of metabolite `g_{g}`.

**Knockout = fix `w_g = 0`.** To knock gene *g* out, the MILP's `z` variable pins `w_g = 0` (via
`link_z`, Ch 7, exactly as for any reaction KO). With the source shut, `g_{g}` can no longer be
produced, and — because in steady state it must also balance to zero — nothing downstream may consume
it. That "no consumption allowed" is precisely how "TRUE becomes FALSE" propagates through the gadgets.

#### The `and` gadget (`ast.And`)

`create_and_metabolite(child_metabolites)` (`networktools.py:1054`): given the child pseudo-metabolites
`c₁,…,c_k` of the children of an `and` node, create one pseudo-metabolite `A = c₁_and_…_and_c_k` (the id
is the sorted children joined by `_and_`) and **one** pseudoreaction

```
c₁ + c₂ + … + c_k  -->  A          (upper_bound = ∞)
```

This reaction consumes **one unit of every child** to produce one unit of `A`. Steady-state mass
balance then forces: to make `A` at rate `f`, each child metabolite `c_i` must be *supplied* at rate
`f`. If *any* child `c_i` cannot be supplied (its subtree is knocked out, so its production capacity is
0), then its balance row forces `f = 0`, hence `A` cannot be produced. **`A` is available iff *all*
children are available — exactly Boolean `and`.** A single shared consuming reaction is what couples
the children conjunctively.

#### The `or` gadget (`ast.Or`)

`create_or_metabolite(child_metabolites)` (`networktools.py:1083`): given children `c₁,…,c_k`, create
one pseudo-metabolite `O = c₁_or_…_or_c_k` and **k separate** pseudoreactions, one per child:

```
c₁ --> O          (upper_bound = ∞)
c₂ --> O
   ⋮
c_k --> O
```

Each reaction alone can produce `O` from a single child. Steady state: `O` can be produced (at any
positive rate) as long as *at least one* child can be supplied. If *all* children are knocked out,
every producing reaction is starved and `O` cannot be produced. **`O` is available iff *any* child is
available — exactly Boolean `or`.** Separate parallel producing reactions is what makes the children
disjunctive.

The recursion `process_ast_node` (`networktools.py:1114`) applies these three rules bottom-up: a
`ast.Name` returns its `g_{id}` metabolite; a `ast.BoolOp` recursively resolves its children to their
pseudo-metabolite ids, then calls the `and`- or `or`-combinator on them; the return value bubbles the
*top-level* pseudo-metabolite id up to the reaction.

#### Attaching the rule to the guarded reaction

After `process_ast_node(r.gpr.body)` returns the top-level metabolite id `M`, the loop at
`networktools.py:1157–1163` does:

```python
r.add_metabolites({model.metabolites.get_by_id(final_metabolite_id): -1.0})
```

i.e. it inserts a **−1** coefficient for `M` into reaction `r`'s column. Now `r` *consumes* one unit of
`M` per unit of its own flux. Since `M` can only balance if it is produced by its gadget, and that
gadget can produce `M` only when the Boolean rule is satisfiable given the surviving gene sources,
`r`'s flux is forced to zero exactly when the rule evaluates FALSE under the current knockouts. When
the rule is TRUE, `M` is available in unlimited supply (all gadget reactions are `[0,∞)`), so `r`'s
flux is unconstrained by the gadget and behaves as before.

Reactions with **no** GPR rule are skipped entirely (`if not r.gene_reaction_rule` at the split loop,
and the AST loop guards on `r.gpr and r.gpr.body`); they get no pseudo-metabolite and are untouched.

There is a string-parsing **fallback** (`networktools.py:1166–1182`) that fires only if AST processing
raises: it splits the rule on `' or '` / `' and '` textually and rebuilds the same gadgets. It exists
for malformed or non-standard rule strings cobra's AST cannot parse; the AST path is the norm.

#### Worked example: `(g1 and g2) or g3`

Take a reaction `R1: A --> B` (`lb=0`, so irreversible; the reversible case is §4.4) with GPR rule
`(g1 and g2) or g3`. Extension produces:

Gene sources (three pseudoreactions, each `[0,∞)`):

```
Wg1:  --> g_g1
Wg2:  --> g_g2
Wg3:  --> g_g3
```

The `and` node over `{g_g1, g_g2}` → metabolite `g_g1_and_g_g2` (sorted), one reaction:

```
R_g_g1_and_g_g2:   g_g1 + g_g2 --> g_g1_and_g_g2
```

The `or` node over `{g_g1_and_g_g2, g_g3}` → metabolite `O = g_g1_and_g_g2_or_g_g3` (children sorted),
two reactions:

```
R0_O:   g_g1_and_g_g2 --> O
R1_O:   g_g3          --> O
```

Finally `R1` is edited to consume `O`:

```
R1:   A + O --> B
```

Now check the Boolean truth table by asking, for each knockout pattern, whether `R1` can carry flux
`> 0` (equivalently whether `O` can be produced):

| deleted genes | `w_g1` | `w_g2` | `w_g3` | can make `g_g1_and_g_g2`? | can make `O`? | `R1` active? | rule value |
|---|---|---|---|---|---|---|---|
| none | free | free | free | yes | yes | yes | TRUE |
| g1 | 0 | free | free | no (and starved) | yes via `g_g3` | yes | TRUE |
| g3 | free | free | 0 | yes | yes via and-branch | yes | TRUE |
| g1,g2 | 0 | 0 | free | no | yes via `g_g3` | yes | TRUE |
| g1,g3 | 0 | free | 0 | no | no | **no** | FALSE |
| g2,g3 | free | 0 | 0 | no | no | **no** | FALSE |
| g1,g2,g3 | 0 | 0 | 0 | no | no | **no** | FALSE |

This matches `(g1 and g2) or g3` exactly. Note the minimal cut sets to disable `R1` are `{g1,g3}` and
`{g2,g3}` — size 2 — which the flux structure discovers without any Boolean-logic code: it is just the
set of pseudoreaction knockouts that renders metabolite `O` unproducible in the LP.

### 4.3 Why the original flux space is unchanged (the correctness invariant)

The docstring claims "the metabolic flux space does not change." Here is why that is exactly true (when
no gene is knocked out), which is the property that legitimizes the whole encoding.

Let the original model have stoichiometry `S ∈ ℝ^{m×n}` over metabolites `M` and reactions `R`, with a
feasible flux `v` satisfying `S·v = 0`, `lb ≤ v ≤ ub`. Extension adds:

- new metabolite rows (all the `g_*`, `*_and_*`, `*_or_*` pseudo-metabolites) — call them `P`;
- new reaction columns (gene sources `w_g`, `and`-reactions, `or`-reactions) — call their fluxes `u`,
  all with bounds `[0, ∞)`;
- for each GPR reaction `r`, a single `−1` entry in row `M_r` (its top pseudo-metabolite) of column `r`.

**Claim (no-KO case).** For every feasible original flux `v`, there exists a choice of pseudoreaction
fluxes `u ≥ 0` such that `(v, u)` is feasible in the extended model; and conversely every feasible
extended flux, projected onto the original reaction columns, is a feasible original flux. Hence the
projection of the extended polytope onto the original reactions equals the original polytope.

*Forward direction (construct `u`).* Take any feasible `v`. Each GPR reaction `r` now demands its top
pseudo-metabolite `M_r` at rate `v_r` (coefficient `−1`, so consumption `= v_r`; for an irreversible
GPR reaction `v_r ≥ 0` after the split of §4.4, so this is a nonnegative demand). Because every gadget
is a chain of `[0,∞)` reactions from the gene sources up to `M_r`, we can *route supply* to meet any
nonnegative demand: set each `or`-branch and `and`-reaction flux to carry exactly the demand upward,
and set each gene source `w_g` to the total demand routed through it (a nonnegative sum). Concretely,
push `v_r` up through *one* satisfying branch of each `or` and through the (unique) `and`-reaction,
accumulating at the gene sources. Every gadget reaction and gene source is `[0,∞)`, and all demands are
`≥ 0`, so all these `u` values are feasible, and every pseudo-metabolite row balances by construction
(produced = consumed = the routed rate). Thus `(v,u)` is feasible.

*Reverse direction (project).* Given feasible `(v,u)`, the original metabolite rows `S·v = 0` are a
*subset* of the extended balance equations (the pseudo-metabolite rows only involve `u` and the new
`−1` entries, never the original `S` entries), so `S·v = 0` still holds, and the original bounds on `v`
are unchanged. Hence `v` is feasible in the original model.

Because the gadget reactions are unbounded above and the pseudo-metabolites appear in no other balance,
they never *add* any constraint on `v` in the no-KO case: for any `v` you can always find `u`. The only
way the extension can bite is when a gene source is *fixed to zero* — then the routing argument fails
for exactly those `v_r` whose every satisfying branch passes through a zeroed source, i.e. exactly the
reactions whose Boolean rule is now FALSE, forcing `v_r = 0`. That is the intended and only effect.

Two implementation details protect this invariant. First, the pseudoreactions are created **once** and
memoized: `created_metabolites` (a set) and the `... not in model.metabolites` guards (e.g.
`networktools.py:1032, 1065, 1094`) ensure a gene shared by many reactions gets a *single* `g_{id}`
source and metabolite, so all its reactions draw from the same tap — this is what makes a shared gene
count once and couple all its reactions. Second, the `and`/`or` metabolite ids are built from the
**sorted** child ids (`"_and_".join(sorted(...))`, `"_or_".join(sorted(...))`), so identical
sub-expressions occurring in different rules collapse to the same pseudo-metabolite and are not
duplicated — canonicalization by sorted name.

### 4.4 Reversible-reaction split and the `reac_map` remap

A subtlety: the invariant argument above needed `v_r ≥ 0` so that a GPR reaction's *demand* for its
pseudo-metabolite is nonnegative. But a reversible reaction has `v_r` free (`lb < 0`). If a reversible
GPR reaction consumed `M_r` with coefficient `−1` and ran *backwards* (`v_r < 0`), it would *produce*
`M_r` — turning the guarded reaction itself into a spurious source of its own gene token, breaking the
logic (the reaction could "power its own enzyme"). Worse, the pseudo-metabolite balance would let a
reverse flux exist even with all genes deleted.

The fix (`networktools.py:1129–1154`): **split every GPR-associated reversible reaction into a forward
and a reverse leg, both irreversible.** For a reaction `r` with `lb < 0`:

- build `r_rev = r * -1` (all stoichiometric coefficients negated, so the reverse direction becomes a
  forward-running reaction);
- if `r` is *bidirectional* (`ub > 0` too), give the reverse leg a distinct id
  `r.id + '_reverse_' + hex(hash(r))[8:]` (`networktools.py:1141`), clamp `r_rev.lower_bound = max(0,
  …)`, and keep the forward leg `r` with `lower_bound = max(0, lb) = 0`;
- if `r` is *purely reverse* (`ub ≤ 0`), the forward leg is dropped (`del_reac`), only the reverse leg
  survives.

Both legs are now `[0, ∞)`-style irreversible, so each has a nonnegative demand for its own copy of the
GPR pseudo-metabolite, and the §4.3 argument holds for each leg independently. Removed/added reactions
are committed with `model.remove_reactions(del_reac)` / `model.add_reactions(rev_reac)`
(`networktools.py:1153–1154`) *before* the AST loop, so both legs get their own consumption edge to the
(shared) top pseudo-metabolite `M_r` — i.e. deleting the gene kills *both* directions at once, as it
must.

**The `reac_map` bookkeeping.** The function returns `reac_map`, a dict recording how each original
reaction id decomposes into the new columns and with what sign. For an un-split reaction:
`reac_map[r.id] = {r.id: 1.0}`. For a split bidirectional reaction:

```python
reac_map[r.id] = {r.id: 1.0, r.id + '_reverse_xxxx': -1.0}
```

The **signs encode the change of variables**: the original signed flux `v_r` equals `(+1)·v_fwd +
(−1)·v_rev`, because `v_rev` is the magnitude of flow in the reverse direction (its column is `r*-1`),
so a reverse flux of magnitude `f` corresponds to original `v_r = −f`. This is the standard
`v = v⁺ − v⁻`, `v⁺,v⁻ ≥ 0` reversible-flux splitting, restricted here to *GPR-associated* reactions
only (non-GPR reactions are never split — no need, since they carry no pseudo-metabolite that a reverse
flux could corrupt).

**Remapping the modules through `reac_map`** (`compute_strain_designs.py:397–410`). Every strain-design
module refers to reactions by id in its `CONSTRAINTS`, `INNER_OBJECTIVE`, `OUTER_OBJECTIVE`, and
`PROD_ID` fields (Ch 1). If a referenced reaction was split, those references must be rewritten in the
new variables, using the *same signed decomposition*. For a constraint's coefficient dict `c[0]`:

```python
for k in list(c[0].keys()):
    v = c[0].pop(k)
    for n, w in reac_map[k].items():
        c[0][n] = v * w
```

i.e. a term `v·(x_k)` becomes `Σ_n (v·w)·(x_n)` over the pieces `n` of `k`, with `w ∈ {+1, −1}`. For a
split reversible reaction this turns `v·v_r` into `v·v_fwd − v·v_rev`, faithfully preserving the signed
flux the module intended. Objectives (`INNER_OBJECTIVE`, `OUTER_OBJECTIVE`, `PROD_ID`) are single dicts
and remapped the same way (`compute_strain_designs.py:406–410`). Because `reac_map` contains an entry
for *every* reaction (`{r.id: 1.0}` for the untouched ones, `networktools.py:1134–1136, 1149`), the loop
can blindly remap every key without special-casing which reactions were split.

### 4.5 `reduce_gpr`: pruning before extension

Extension cost scales with the number of surviving genes and Boolean operators: each gene adds a
pseudoreaction + metabolite, each operator a gadget. Many genes can be proven irrelevant *before* any
of that structure is built, which both shrinks `S` and removes useless binary candidates from the MILP.
`reduce_gpr(model, essential_reacs, gkis, gkos)` (`networktools.py:664`) does this pruning, returning a
trimmed `gkos` (gene-KO-cost dict); it runs just before `extend_model_gpr`
(`compute_strain_designs.py:389`). Its steps:

1. **Blocked reactions lose their GPR** (`networktools.py:882–888`). Any reaction with bounds `(0,0)`
   is dead anyway; its rule is cleared and genes that end up in no reaction are dropped. No point
   encoding logic for a reaction that can never carry flux.

2. **Protect genes that touch only essential reactions** (`networktools.py:893–895`). A gene whose
   reaction set is a subset of `essential_reacs` (reactions that *must* stay operational — from the FVA
   over PROTECT/desired modules, Ch 5) can never be a useful KO: knocking it out could only threaten an
   essential reaction. It is added to `protected_genes`.

3. **Protect genes that are individually essential *to* an essential reaction** (`networktools.py:898–901`).
   Using `is_gene_essential_to_reaction_ast`, which evaluates the reaction's GPR AST with that one gene
   set to `False` and checks whether the whole rule collapses to `False`: if deleting the gene alone
   would kill an essential reaction, the gene must be protected. (A gene inside an `or` of an essential
   reaction is *not* caught here — deleting it leaves the reaction alive — so it stays knockable.)

4. **Drop protected genes from the KO-cost dict** (`networktools.py:904`): `[gkos.pop(pg.id) …]` — they
   are no longer intervention candidates.

5. **Everything the user did not list as knockable is also protected** (`networktools.py:907`): genes
   whose id *and* name are absent from `gkos` cannot be knocked out, so they are protected too.

6. **Genes with knock-in costs are un-protected** (`networktools.py:910–911`): a gene in `gkis` is a
   *target* (it can be added), so it is removed from the protected set even if the above rules caught it.

7. **Simplify each GPR rule with protected genes pinned TRUE** (`networktools.py:915–933`).
   `simplify_gpr_ast` walks the AST setting every protected gene to `True` and applies Boolean
   simplification (`apply_gene_protection_to_ast`, `networktools.py:719`): `True and X → X`,
   `True or X → True`, plus absorption (`A or (A and B) → A`, `networktools.py:765–822`). If the rule
   collapses to `True`, the reaction is no longer knockable-by-gene and its rule is cleared (so it gets
   no gadget at all); otherwise the simplified, *smaller* rule replaces the original — fewer operators,
   hence fewer gadgets at extension.

8. **Remove obsolete and protected genes** from the model (`networktools.py:935–937`), so
   `extend_model_gpr` never sees them.

The net effect: `extend_model_gpr` is handed a model whose GPR rules mention only genes that are (a)
user-declared knockable or knock-in-able and (b) capable of affecting a non-essential reaction, with
the rules already Boolean-minimized. On genome-scale models this removes a large fraction of genes and
operators before the expensive structure is built.

**The id-vs-name subtlety.** Genes can be referenced by *either* their id or their (human-readable)
name, and models are inconsistent about which the user supplies in `gkos`/`gkis`. `reduce_gpr` therefore
checks **both**: the protection rule at `networktools.py:907` protects a gene only if *neither*
`g.id in gkos` *nor* `g.name in gkos`, and the KI un-protection at `networktools.py:910` collects
`g.id for g in model.genes if (g.id in gkis) or (g.name in gkis)`. Note the asymmetry that this matching
introduces downstream: `extend_model_gpr` names each gene pseudoreaction by id *or* name depending on
the global `has_gene_names` flag (`use_names`, decided at `compute_strain_designs.py:396` and passed in),
so the id-vs-name choice must stay consistent between the cost dicts and the pseudoreaction ids or the
later cost lookup silently misses (see Ch 10 for the fragility this creates). `reduce_gpr` hedges by
accepting both spellings; the pseudoreaction naming commits to one.

### 4.6 The two-compression-pass boundary and why regulatory genes are exempt from pass #1

Network compression (Ch 3) is run **twice**, straddling GPR extension:

- **COMPRESS #1** (`compute_strain_designs.py:357`) runs on the pre-extension metabolic model with
  `propagate_gpr=True`.
- **COMPRESS #2** (`compute_strain_designs.py:434`) runs *after* `extend_model_gpr`, with
  `propagate_gpr` left at its default `False` (`compression.py:1853`).

**Why two passes.** The first pass compresses the genuine metabolic network while it is still small and
GPR-free, so the expensive lossless reductions operate on the original reactions. But it *cannot*
compress the gene pseudoreactions/pseudo-metabolites, because they do not exist yet. The second pass
runs on the *extended* network to reabsorb structure that extension introduced — chains of gene
sources, `and`/`or` gadgets, and split legs that turn out to be flux-coupled can be merged, shrinking
the MILP. Splitting the work this way keeps each pass cheap and lets the GPR structure benefit from
compression too.

**Why `propagate_gpr` differs.** In pass #1 the metabolic reactions still carry Boolean GPR *strings*.
When two reactions are merged, their rules must be combined correctly — an AND-merge for flux-coupled
reactions, an OR-merge for parallel ones (`compression.py:1982, 2040`, the `_combine_gpr_and/or` helpers,
Ch 3) — so that after extension the merged reaction's rule still reflects both originals. Hence
`propagate_gpr=True`. In pass #2 the rules have *already been consumed* by `extend_model_gpr` (converted
to flux structure) and the reactions' `gene_reaction_rule` strings are no longer the source of truth —
the gadgets are. Propagating GPR strings again would be meaningless and double-count, so pass #2 uses
`propagate_gpr=False` and merges purely on stoichiometry.

**Why regulatory-gene reactions are exempted from COMPRESS #1** (`compute_strain_designs.py:334–353`).
A *gene-based regulatory* intervention (a constraint like `g <= X` or `g >= X` on a gene, as opposed to
a plain KO `g = 0`) is applied by `extend_model_regulatory` *after* GPR extension, because it needs the
`g_gene` pseudo-metabolite to exist so the bound can be hung on the gene's pseudoreaction flux. The
problem: if COMPRESS #1 merges several reactions that a regulatory gene controls, the merged reaction is
attached to that gene with a **collapsed/rescaled stoichiometry** — parallel/coupled merging multiplies
reactions by rational scale factors (Ch 3) — so a later gene-regulatory bound `g <= X` would be applied
against a *mis-scaled* flux and would not mean the same thing as in the uncompressed model. The code
comment (`compute_strain_designs.py:335–340`) states this directly: a pre-GPR merge "hooks the gene to
the merged reaction with the wrong (collapsed) stoichiometry, so a gene-regulatory bound … is
mis-scaled." 

The remedy is to **exempt exactly the reactions controlled by a deferred-regulatory gene** from merging
in pass #1. The block scans each deferred regulatory constraint string for tokens matching a gene id or
name (`compute_strain_designs.py:344–350`), collects that gene's reactions into
`no_coupled_compress_reacs`, and passes them to `compress_model` so they are *not* coupled-merged; it
also adds them to `no_par_compress_reacs` (`:353`) so they are not parallel-merged and their **names stay
stable** across the two passes (the pass-#1 exemption matches them by name, so a rename would break the
matching). These same reactions *do* merge safely in **pass #2**, once `extend_model_gpr` has created the
`g_gene` metabolite and `extend_model_regulatory` has hung the bound on the gene pseudoreaction — at that
point the regulatory constraint lives on the *gene source flux*, not on the metabolic reaction's
possibly-rescaled flux, so merging the metabolic reactions no longer distorts it. The comment is explicit
that **plain gene KOs (`=0`) and KIs (unbounded when added) are unaffected** and need no exemption —
only *regulatory* genes, whose bound is a finite scaled quantity, are sensitive to the stoichiometric
rescaling. (This exemption logic is the fix for closed issue #44's class of bound-scaling bugs; see
Ch 3 for the compression bound-intersection mechanics and Ch 10 for the cautionary history.)

### 4.7 Name truncation (sha256), Gurobi/GLPK only

Extension generates pseudo-metabolite and pseudoreaction ids by *concatenating* child ids with `_and_`
/ `_or_` separators. Nested rules over long gene ids can produce names hundreds of characters long.
**Gurobi and GLPK impose a 255-character limit on variable/constraint names**; CPLEX and SCIP do not.
The code sets `MAX_NAME_LEN = 230` (`networktools.py:1001`) and, *only when the active solver is in
`{GUROBI, GLPK}`* (checked at every id-construction site, e.g. `networktools.py:1026, 1043, 1059, 1072,
1088, 1103, 1144`), truncates:

```python
def truncate(id):
    h = hashlib.sha256(id.encode()).hexdigest()[:20]
    return id[0:MAX_NAME_LEN - 21] + "_" + h
```

i.e. it keeps the first `209` characters and appends `_` + a 20-hex-char sha256 digest of the full id,
yielding a ≤230-char name. The digest suffix preserves uniqueness (two long ids sharing a 209-char
prefix still differ in hash) so distinct pseudo-metabolites do not accidentally collide after
truncation. A `warning_name_too_long` message (`networktools.py:1003`) is logged once per truncated
name, suggesting the user switch to CPLEX or simplify gene names to avoid it.

Two properties matter for a maintainer. First, **truncation is solver-conditional**: the *same model*
produces different pseudoreaction ids under Gurobi/GLPK than under CPLEX/SCIP. Any code that matches
these ids by string (cost-dict lookups, module remapping, decompression) must therefore see the *same*
truncated names — which is why the id is truncated at the single point of creation and reused, not
re-derived elsewhere. Second, the sha256 rewrite is a **known fragility, adjacent to open issue #43**:
because the truncated name is not human-meaningful and because the truncation depends on solver
identity, a mismatch between where a name is generated and where it is later looked up can silently drop
a gene knockout from the reported solution. The mechanism and the concrete failure are owned by **Ch 10**;
here we only flag that the `{GUROBI, GLPK}`-gated sha256 truncation is the code path involved.


## 5. FVA in preprocessing

Flux Variability Analysis (FVA) — the pair of LPs that, for every reaction *j*, compute
`min v_j` and `max v_j` over the steady-state polytope `{v : Sv = 0, lb ≤ v ≤ ub}` (see
Ch 2 for the LP formulation) — appears **three times** in `compute_strain_designs`'s
preprocessing, at three different points in the pipeline, on three different versions of the
model, each time answering a different question and feeding a different downstream consumer.
None of the three is "just diagnostics": each one *removes work from the MILP* that the solver
would otherwise have to do, and one of them (the second) is the single largest slice of
genome-scale wall-time. This chapter dissects all three, then the accelerated FVA engine
(`speedy_fva`) that all of them call, and closes by explaining why FVA #2 costs ~117 s.

The three uses, at a glance:

| # | Call site (`compute_strain_designs.py`) | Model state | Scope | Question answered | Consumer |
|---|------------------------------------------|-------------|-------|-------------------|----------|
| 1 | ~L373–381 | after COMPRESS #1, **pre-GPR** | whole model | Which reactions are *essential* for a PROTECT/desired behaviour? | drop from `ko_cost`; feed `reduce_gpr` |
| 2 | `bound_blocked_or_irrevers_fva`, ~L450 (→ `networktools.py:1589`) | after GPR extension + COMPRESS #2 | whole model | Which bounds never bind? Which reactions are blocked/irreversible? | rewrite model bounds → shrink/condition the MILP |
| 3 | ~L460–491 | after COMPRESS #2 | **knockable only** (`reaction_list`) | Which knockable reactions are essential per module? Which are size-1 cut sets? | drop essentials + size-1 MCS from `ko_cost`; re-inject MCS at decompression |

All three ultimately dispatch to `fva()` in `lptools.py:245`, which is a thin wrapper that
immediately calls `speedy_fva` (`lptools.py:281–282`). The legacy brute-force implementation
`fva_legacy` (`lptools.py:285`) is retained only as a debugging fallback.

### 5.1 The essentiality test — geometry of `min(abs(range)) > 1e-10 and prod(sign(range)) > 0`

Both FVA #1 and FVA #3 classify a reaction as *essential* (for a given module's constraint
set) using the identical predicate, at `compute_strain_designs.py:378` and again at `:465`:

```python
if np.min(abs(limits)) > 1e-10 and np.prod(np.sign(limits)) > 0:  # find essential
    essential_reacs.add(reac_id)
```

Here `limits` is the two-element vector `[v_min, v_max]` returned by FVA for reaction *j*,
i.e. the endpoints of the attainable flux interval `[v_min^j, v_max^j]` under that module's
constraints. Read the predicate geometrically:

- **`np.prod(np.sign(limits)) > 0`** — `sign(v_min)·sign(v_max) > 0` — is true iff `v_min`
  and `v_max` have the **same, nonzero sign**. That is exactly the statement *the interval
  `[v_min, v_max]` does not contain 0*. (If either endpoint were 0 the product would be 0;
  if the interval straddled 0 the signs would differ and the product would be negative.)
- **`np.min(abs(limits)) > 1e-10`** — `min(|v_min|, |v_max|) > 10⁻¹⁰` — is the *numerical
  guard* that the endpoint closest to zero is a strict, non-noise distance away from it, so
  the "does not contain 0" conclusion is not an artifact of solver tolerance.

Together they assert: **every feasible flux state that satisfies the module's constraints
routes a strictly nonzero, sign-definite flux through reaction *j*.** Geometrically, the flux
polytope of that module lies entirely on one side of the hyperplane `v_j = 0` and does not
touch it. Consequently, the constraint `v_j = 0` (which is precisely what a knockout imposes)
is *inconsistent* with the module: **knocking out *j* makes the module infeasible.**

Why that matters depends on the module type, and this is the whole point of running FVA #1/#3
separately per module (`for m in sd_modules:`):

- If the module is **PROTECT/desired** (a behaviour that must remain *possible*), a reaction
  essential to it can never appear in a valid design — knocking it out would violate the
  PROTECT requirement. Such a reaction is therefore useless as a knockout candidate and is
  stripped from `ko_cost` (removing its binary `z_j` from the MILP entirely).
- If the module is **SUPPRESS** (a behaviour that must be made *impossible*), a reaction
  essential to it is, by itself, a valid intervention: deleting it kills the behaviour. That
  is the size-1 MCS observation exploited by FVA #3 (§5.4).

A tiny worked example. Two reactions, `R1: A→B`, `R2: B→C`, sink `EX_C`, with a PROTECT
module requiring `EX_C ≥ 1`. FVA over `{Sv=0, v≥0, EX_C≥1}` yields `v_R1 ∈ [1, 1000]`,
`v_R2 ∈ [1, 1000]`: both intervals sit strictly above 0, `sign(1)·sign(1000)=+1`, and
`min(|1|,|1000|)=1 > 10⁻¹⁰`. Both are flagged essential — correctly, since either KO drops
`EX_C` to 0 and breaks the PROTECT.

### 5.2 FVA #1 — essential reactions in PROTECT/desired modules (pre-GPR)

FVA #1 runs immediately after COMPRESS #1 and *before* GPR integration
(`compute_strain_designs.py:371–381`), so it sees a purely metabolic, compressed network with
no gene pseudoreactions yet (see Ch 4 for the COMPRESS #1/GPR boundary). It iterates only over
non-SUPPRESS modules:

```python
for m in sd_modules:
    if m[MODULE_TYPE] != SUPPRESS:      # essentiality only meaningful for desired / opt-/robustknock
        flux_limits = fva(cmp_model, solver=..., constraints=m[CONSTRAINTS], compress=False)
        for (reac_id, limits) in flux_limits.iterrows():
            if np.min(abs(limits)) > 1e-10 and np.prod(np.sign(limits)) > 0:
                essential_reacs.add(reac_id)
[cmp_ko_cost.pop(er) for er in essential_reacs if er in cmp_ko_cost]
```

**Rationale (why drop from `ko_cost`).** As argued in §5.1, a reaction essential for a
required (PROTECT/desired) behaviour can *never* be part of any feasible design — its knockout
would violate a PROTECT constraint that the MILP is required to keep feasible. Every candidate
design that includes it is infeasible *a priori*. Popping it from `cmp_ko_cost` (line 381)
removes its binary variable `z_j` from the intervention set the MILP will branch over: the
solver never even considers it, and no infeasible node is generated to reject it. This is a
pure model-size reduction with zero effect on the solution set.

**Second consumer: `reduce_gpr`.** The `essential_reacs` set computed here is passed straight
into GPR reduction (`compute_strain_designs.py:389`):

```python
uncmp_gko_cost = reduce_gpr(cmp_model, essential_reacs, uncmp_gki_cost, uncmp_gko_cost)
```

`reduce_gpr` (`networktools.py:664`) simplifies the Boolean gene–protein–reaction rules before
they are compiled into flux structure (Ch 4). Knowing which reactions are essential lets it
also drop the *genes* that only ever control essential reactions from the knockable gene set:
if a reaction can never be knocked out, a gene whose only role is to (be required to) enable
that reaction is likewise non-knockable, and pruning it shrinks both the GPR encoding and the
gene KO cost dictionary. Thus one FVA pass feeds two reductions — reaction-level and, through
`reduce_gpr`, gene-level.

**Why `compress=False` here.** The model is *already* compressed (COMPRESS #1 just ran), so
`speedy_fva`'s own internal coupled-compression pass is switched off to avoid re-compressing an
already-compressed, rational-bound network. FVA #1 is comparatively cheap: it runs on the small
pre-GPR metabolic network and typically for a single PROTECT module.

### 5.3 FVA #2 — `bound_blocked_or_irrevers_fva`: relaxing non-binding bounds

FVA #2 runs *after* GPR extension and COMPRESS #2, so that **all** reactions — including the
gene pseudoreactions added by `extend_model_gpr` — are processed
(`compute_strain_designs.py:444–451`):

```python
bound_blocked_or_irrevers_fva(cmp_model, solver=kwargs[SOLVER], compress=False)
```

Its body (`networktools.py:1589–1625`) runs one whole-model FVA and then rewrites each
reaction's *stored* bounds (`r._lower_bound` / `r._upper_bound` directly, to make the change
permanent and bypass cobra's optlang synchronisation) according to **four independent
branches**. With CPLEX/Gurobi the tolerance `tol` is `0.0`; with SCIP/GLPK it is `1e-10`
(`networktools.py:1599–1602`). Let `[v_min, v_max]` be the FVA interval and `[lb, ub]` the
current bounds.

```python
if r.lower_bound < 0.0 and limits.minimum - tol > r.lower_bound:   # (A) redundant lb → −inf
    r._lower_bound = -np.inf ;                       n_lb_to_inf += 1
if limits.minimum >= tol:                                          # (B) min ≥ 0 → lb = 0
    r._lower_bound = max([0.0, r._lower_bound]) ;    n_tightened_zero += 1
if r.upper_bound > 0.0 and limits.maximum + tol < r.upper_bound:   # (C) redundant ub → +inf
    r._upper_bound = np.inf ;                        n_ub_to_inf += 1
if limits.maximum <= -tol:                                         # (D) max ≤ 0 → ub = 0
    r._upper_bound = min([0.0, r._upper_bound]) ;    n_tightened_zero += 1
```

Decoding the four branches:

- **(A) redundant lower bound → −∞.** The reaction *can* go negative (`lb < 0`), yet the
  achievable minimum flux `v_min` is strictly greater than `lb`. The lower bound therefore
  never binds — the network's stoichiometry constrains `v_j` more tightly than the box bound
  does. Relaxing `lb` to `−∞` discards a constraint that is provably slack everywhere.
- **(B) min ≥ 0 → lb = 0.** FVA proves `v_j` cannot be negative under steady state. The
  reaction is effectively **irreversible in the forward direction**, so its lower bound is
  pinned at 0 (`max(0, lb)`). Note the interaction with (A): a reaction with `lb = −1000` but
  `v_min = 2` first has `lb` set to `−∞` by (A), then *overwritten* to `0` by (B) because the
  branches are evaluated in sequence on the same reaction. The net effect is `lb = 0`
  (irreversible), not `−∞`. Detecting irreversibility this way lets the MILP omit the negative
  half-space entirely.
- **(C) redundant upper bound → +∞.** Symmetric to (A): `ub > 0` but the achievable maximum
  `v_max` is strictly below `ub`, so the upper box bound never binds and is relaxed to `+∞`.
- **(D) max ≤ 0 → ub = 0.** Symmetric to (B): the reaction cannot carry positive flux, so it
  is irreversible in the backward direction and `ub` is pinned at 0.

A reaction that is fully **blocked** (`v_min = v_max = 0`) triggers (B) *and* (D): `lb` and
`ub` are both pinned to 0, freezing it out of every flux state.

**Decoding the real log line.** `bound_blocked_or_irrevers_fva` emits, on iML1515 after GPR
extension (`networktools.py:1624`):

```
FVA bounds: 4 lb→inf, 1825 ub→inf, 2258 tightened to 0, 2150 stayed finite
```

- `4 lb→inf` = branch (A) fired 4 times: only 4 reactions had a genuinely reversible, slack
  lower bound. (Almost all reactions in a curated model are already forward-irreversible, so
  few have a slack negative lower bound to relax.)
- `1825 ub→inf` = branch (C) fired 1825 times: for 1825 reactions the upper bound was slack
  and is relaxed to `+∞`. This is the large one — most reactions' nominal upper bound (e.g.
  the default 1000) never binds; the true maximum is limited by network stoichiometry.
- `2258 tightened to 0` = **the combined count of branches (B) and (D)** — the same counter
  `n_tightened_zero` is incremented in both (`networktools.py:1615` and `:1621`). It therefore
  aggregates "lower bound pinned to 0 (forward-irreversible)" and "upper bound pinned to 0
  (backward-irreversible / blocked)". It is *not* a count of distinct reactions: a single
  reaction that triggers both (B) and (D) — i.e. a blocked reaction — is counted twice, and a
  reaction that triggers (A) then (B) contributes to both `n_lb_to_inf` and `n_tightened_zero`.
- `2150 stayed finite` is computed independently at `networktools.py:1622–1623` as the number
  of reactions with **at least one finite bound after all rewrites**:
  `sum(1 for r in model.reactions if not isinf(r.lower_bound) or not isinf(r.upper_bound))`.
  These are the reactions that were *not* fully relaxed to `(−∞, +∞)`.

Because the four counters overlap (a reaction can increment several), they do **not** sum to
the reaction count; only "stayed finite" is a clean per-reaction tally. This subtlety is easy
to misread as an inconsistency — it is intentional (each counter reports how often a *branch*
fired), not a bug.

#### Why relaxing a provably non-binding bound to ±∞ shrinks and conditions the MILP

This FVA is not cosmetic — it directly determines the size and numerical quality of the MILP
built next (`SDMILP`, Ch 6–7). The mechanism has two prongs.

**(1) Only genuinely finite (binding) bounds become knockable constraints.** In the MILP, a
reaction knockout is enforced by tying its binary `z_j` to the reaction's flux-bound rows so
that `z_j = 1 ⇒ v_j = 0`; and in the dualized SUPPRESS block every finite reaction bound
becomes a *dual variable* with its own row and its own coupling to `z` (see Ch 6 for the
Farkas dualization and Ch 7 for `link_z`). A bound relaxed to `±∞` is, by definition, *no
constraint at all*: it contributes no row to the primal, hence no dual variable to the
dualized problem, and nothing for `z` to switch on that side. So every `lb→−∞` (branch A) and
`ub→+∞` (branch C) *deletes* a constraint row and, in the dual, a variable. On the numbers
above that is `4 + 1825 = 1829` bound rows removed. Conversely, the 2150 reactions that
"stayed finite" are exactly the ones whose remaining binding bound *does* need an
indicator/big-M linkage in the MILP — the relaxation has narrowed the set of reactions that
require this machinery to the ones that genuinely constrain flux.

**(2) The remaining big-Ms get tighter.** Where a knockout linkage is realised as a **big-M**
constraint (PROTECT's finite-flux primal rows; the big-M vs indicator fork is emergent from
bound structure, Ch 7), the constant `M` must be a valid over-estimate of `|v_j|`. `link_z`
derives each `M` from a bounding LP over the reaction's flux range. By replacing the loose
nominal box bounds (e.g. `±1000`) with (a) the *tight, FVA-proved* range or (b) an honest
`±∞` where the bound is slack, FVA #2 feeds `link_z` sharper information: reactions with a
proved finite range get a smaller, tighter `M` (better LP relaxation, faster branch-and-bound),
and reactions whose bound is genuinely non-binding are steered toward the **indicator**
formulation (which has no `M` at all and yields a tighter relaxation) rather than a
meaningless huge `M`. Both outcomes improve the MILP: fewer rows, tighter continuous
relaxation, better conditioning. (See Ch 7 for the exact `self.M`/bounding-LP fork.)

The important invariant: because branches (A) and (C) only relax bounds that FVA has *proved*
never bind, and (B)/(D) only pin bounds the reaction can provably never cross, **the feasible
flux set is unchanged.** No design is added or lost; only the *description* of the polytope is
made leaner and better-conditioned.

### 5.4 FVA #3 — knockable-scoped essentials and size-1 MCS extraction

FVA #3 (`compute_strain_designs.py:453–494`) runs on the final, fully GPR-extended and
COMPRESS #2-compressed model, but — unlike #1 and #2 — it is **scoped to knockable reactions
only** via `speedy_fva`'s `reaction_list` kwarg:

```python
knockable_ids = list(set(cmp_ko_cost.keys()) | set(cmp_ki_cost.keys()))
for m in sd_modules:
    flux_limits = fva(cmp_model, solver=..., constraints=m[CONSTRAINTS],
                      compress=False, reaction_list=knockable_ids)
    ...
    if m[MODULE_TYPE] != SUPPRESS:
        essential_reacs.update(essentials_in_module)     # essential for a PROTECT/desired module
    else:
        suppress_essential.update(essentials_in_module)  # essential for the SUPPRESS module
```

Essentiality of a *non-knockable* reaction is irrelevant here — the MILP will never toggle its
`z` — so restricting FVA to `knockable_ids` avoids computing `2n` LPs and instead computes only
`2·|knockable|`. The same essentiality predicate from §5.1 is applied, but now the results are
**split by module type** into two sets: `essential_reacs` (essential for some PROTECT/desired
module) and `suppress_essential` (essential for the SUPPRESS module).

**Size-1 MCS: the core observation.** A Minimal Cut Set is a smallest set of knockouts that
makes the SUPPRESS behaviour infeasible while keeping PROTECT feasible (Ch 1). A reaction that
is **essential for the SUPPRESS behaviour but NOT essential for any PROTECT behaviour** is,
all by itself, a valid cut set of size one: deleting it makes SUPPRESS infeasible (essential ⇒
`v_j = 0` breaks it, §5.1), and — because it is *not* PROTECT-essential — deleting it leaves
PROTECT feasible. This is computed by a set difference
(`compute_strain_designs.py:472–491`):

```python
is_classical_mcs = (len([m for m in sd_modules if m[MODULE_TYPE] == SUPPRESS]) == 1 and
                    all(m[MODULE_TYPE] == PROTECT for m in [... non-SUPPRESS ...]))
if is_classical_mcs and suppress_essential:
    size1_mcs = suppress_essential - essential_reacs          # SUPPRESS-essential, not PROTECT-essential
    size1_mcs_knockable = {r for r in size1_mcs if r in cmp_ko_cost}
    if size1_mcs_knockable:
        cmp_size1_mcs = [{r: -1} for r in size1_mcs_knockable]
    both_essential = suppress_essential & essential_reacs      # essential for BOTH → non-knockable
    essential_reacs.update(both_essential)
    for r in size1_mcs_knockable:
        cmp_ko_cost.pop(r, None)                                # remove from KO candidates
```

**The `is_classical_mcs` guard.** The size-1-MCS shortcut is *only* valid for a classical MCS
problem: **exactly one SUPPRESS module and every remaining module a PROTECT**
(`compute_strain_designs.py:473–474`). The guard exists because the "essential-for-SUPPRESS ⇒
valid single cut" argument relies on there being a single, well-defined behaviour to suppress
and only feasibility-preservation (not optimization) requirements to respect. In bilevel
problems (OptKnock/RobustKnock/OptCouple, which carry inner/outer objectives) or multi-SUPPRESS
problems, a reaction that is SUPPRESS-essential is *not* guaranteed to be a self-contained
minimal intervention — the objective coupling or a second SUPPRESS can make the "singleton"
either non-minimal or insufficient — so the shortcut is disabled and those reactions flow into
the ordinary MILP.

**Why pull size-1 MCS out of `ko_cost`.** Once a reaction *r* is known to be a size-1 cut set,
any larger design that *contains* *r* is **non-minimal** — it is a superset of the already-known
minimal cut `{r}`. Leaving *r*'s binary `z_r` in the MILP would invite the solver to enumerate
exactly those non-minimal supersets, wasting branch-and-bound effort and (in POPULATE mode)
polluting the solution pool with dominated designs that would only be filtered out later. So
each such *r* is `pop`ped from `cmp_ko_cost` (line 490–491), deleting `z_r` from the MILP. The
size-1 cuts themselves are stashed in `cmp_size1_mcs` as `[{r: -1}]` entries (the `-1` encodes
"knock this reaction out") and are **re-injected as standalone solutions at decompression**
(`_decompress_solutions`, Ch 9), so they still appear in the final result set — they are simply
solved by inspection instead of by the MILP.

Two guard details worth noting:

- The filter `size1_mcs_knockable = {r for r in size1_mcs if r in cmp_ko_cost}` (line 479)
  restricts extraction to reactions that are *pure KO candidates*. Reactions carrying a KI or
  regulatory intervention are left in place (comment at `:486–489`), because they may still
  participate in non-KO solutions that the singleton-KO shortcut does not represent.
- `both_essential = suppress_essential & essential_reacs` (line 484): a reaction essential for
  BOTH the SUPPRESS and a PROTECT behaviour cannot be knocked out at all (it would break
  PROTECT), and is therefore folded into `essential_reacs` and removed from `ko_cost` by the
  final sweep at `compute_strain_designs.py:494`.

### 5.5 The `speedy_fva` acceleration engine

Every FVA above calls `fva()` → `speedy_fva` (`speedy_fva.py:263`). Understanding its algorithm
is essential because it is where the wall-time is spent, and its behaviour depends sharply on
the `reaction_list` scoping and `compress` flags the three call sites pass.

The naive FVA (`fva_legacy`, `lptools.py:285`) solves **`2n` independent LPs**: for each of the
`n` reactions it sets objective `+e_j` and `−e_j` and solves to get `v_min^j` and `v_max^j`.
`speedy_fva` produces the identical result but replaces most of those `2n` solves with a small
number of *global scan LPs* whose single optimal vertex simultaneously resolves the min or max
of many reactions at once. It is a **two-phase** algorithm.

#### Bookkeeping and the "resolved" mask

`speedy_fva` maintains, for the `n` reactions, boolean masks `res_max`, `res_min` and
incumbent vectors `incumbent_max`, `incumbent_min` (`speedy_fva.py:367–370`). A reaction's max
(resp. min) is "resolved" when its true `v_max` (resp. `v_min`) is known. Three cheap
pre-resolutions run before any LP:

- **Fixed reactions** (`|ub − lb| < 10⁻¹²`, line 373–377): `v_min = lb`, `v_max = ub` with no
  LP.
- **`reaction_list` scoping** (line 380–387): every reaction *not* in the requested list is
  marked resolved with `NaN` incumbents. This is how FVA #3's `reaction_list=knockable_ids`
  collapses the problem — non-knockable reactions are simply never scanned or solved, and come
  back as `NaN` in the returned DataFrame.
- **`v = 0` feasibility shortcut** (line 416–430): if `0` is a feasible flux vector — which
  holds when no lower bound is strictly positive, no upper bound strictly negative, and there
  are no extra constraints (`not np.any(lb > tol) and not np.any(ub < -tol) and not
  has_constraints`) — then for every reaction whose `lb = 0`, the minimum is provably `0`
  (it cannot go below `lb=0`, and `0` is attainable), and symmetrically every reaction with
  `ub = 0` has maximum `0`. These are resolved for free, no LP. This single check typically
  clears a large fraction of an irreversible-heavy genome-scale model's bounds.

#### Phase 1 — global scan LPs

**(1b) The `min Σ|x|` scan LP.** The first real LP minimizes the total absolute flux
`Σ_j |v_j|` subject to `Sv = 0`, the extra constraints, and the bounds (`_build_abssum_lp`,
`speedy_fva.py:159`). Absolute values are linearized by **variable splitting**: reactions are
classified as forward-only (`lb ≥ 0`, so `|v_j| = v_j`, objective coeff `+1`), backward-only
(`ub ≤ 0`, so `|v_j| = −v_j`, coeff `−1`), or truly reversible (`lb < 0 < ub`). For each
reversible reaction the variable is split `v_j = p_j − n_j` with `p_j, n_j ≥ 0` and an
auxiliary equality row `v_j − p_j + n_j = 0`, and both `p_j` and `n_j` carry objective coeff
`+1` so the objective equals `p_j + n_j = |v_j|` at optimum (`speedy_fva.py:186–223`).
Infinite bounds are clamped to `±BIG (=1000)` purely so the *push* objective is bounded; this
does not alter feasibility (line 238–251).

The optimal vertex of this LP is the flux state with the least total flux. Its virtue is that
it drives most reactions **to zero**: any reaction sitting exactly at a `lb = 0` or `ub = 0`
bound at this vertex is resolved by the vectorized *bound scan* `_bound_scan`
(`speedy_fva.py:395–408`), which marks `res_max`/`res_min` wherever `|x_j − ub_j| < 10⁻⁹` or
`|x_j − lb_j| < 10⁻⁹`. In one LP this resolves the min/max of every reaction that touches a
zero bound at the min-flux vertex. Simultaneously the vertex's flux values update the
incumbents (`np.maximum(incumbent_max, x_scan)`, `np.minimum(incumbent_min, x_scan)`,
line 451–452): even a reaction not *proved* extreme has its known range widened by this
witness — **co-optimization**, one LP contributing evidence about `n` reactions at once.

**(1c) Iterative push-to-bounds with warm-started dual simplex.** The remaining unresolved
maxima are attacked collectively: a single objective `c` puts `−1` on *every* reaction whose
max is still unresolved (`speedy_fva.py:470–478`) and the LP is re-solved — pushing all of
them toward their upper bounds at once. Whatever lands on its `ub` is resolved by `_bound_scan`;
incumbents update for the rest. The symmetric objective with `+1` on unresolved-min reactions
(line 495–502) pushes toward lower bounds. This alternation repeats
(`while True: ... if resolved_this_round < 5: break`, line 465–529) until a round resolves
fewer than 5 new bounds — i.e. until the cheap global pushes stop paying off.

The critical performance ingredient is that the scan LP object is **reused** across all these
re-solves — only the objective vector changes (`scan_lp.set_objective(...)`), never the
constraint matrix — and the solver is set to **dual simplex** (`set_lp_method(LP_METHOD_DUAL)`,
line 437). Changing only the objective keeps the previous basis *primal*-feasible but
dual-infeasible, which is exactly the situation dual simplex resumes from cheaply: each
re-optimization is a warm-started handful of pivots rather than a cold solve. Dozens of push
LPs therefore cost a small multiple of one LP.

#### Phase 2 — individual LPs for the residual

Whatever Phase 1 could not resolve (`n_remaining = 2n − n_done`, line 539–540) is finished with
individual per-objective LPs, dispatched one of two ways (`speedy_fva.py:543–708`):

- **Parallel** (`n_remaining ≥ 1000 and threads > 1`, line 543): the unresolved objective
  indices (even = max, odd = min, via `idx2c`) are farmed to an `SDPool` of workers, each
  holding its own persistent LP (`fva_worker_init`/`fva_worker_compute`), with a NaN-retry
  loop for any solve that returns NaN.
- **Sequential** (`0 < n_remaining < 1000`, or `threads == 1`, line 611): a single warm-started
  LP is stepped through the residual objectives with `set_objective_idx`, periodically rebuilt
  every 200 solves to limit warm-start basis degeneration (line 632–633). Each solved vertex is
  *also* run through `_bound_scan` and the incumbent update (line 705–708), so even in Phase 2
  one LP can opportunistically resolve *other* pending reactions — the same co-optimization
  trick. A correctness guard (line 667–686) detects when a warm-started optimum is *worse* than
  the incumbent (a sign of a degenerate/stale basis) and rebuilds the LP and re-solves from
  scratch for that objective.

`threads` auto-selects to `Configuration().processes` only when the model has `≥ 1000`
reactions, else `1` (line 304–305). Note the asymmetry that drives §5.6: the parallel path is
gated on **`n_remaining ≥ 1000`**, i.e. on how many objectives *survive Phase 1*, not on the
model size.

#### Internal compression (`compress`) and result expansion

When `compress` is `None`/`True` and the model has `≥ 200` reactions (line 300–301),
`speedy_fva` first lumps flux-coupled reactions and removes conservation rows
(`_compress_for_fva`, line 49) — a *single* nullspace pass (no recursive fixpoint), since FVA
needs only first-order couplings — runs FVA on the smaller compressed model, then expands the
results back via `_expand_fva` (line 114), scaling lumped reactions by their coupling factor
(with a min/max swap when the factor is negative, line 138–140) and filling blocked reactions
with `0/0`. **All three preprocessing call sites pass `compress=False`**, because the model is
already compressed by the pipeline's own COMPRESS passes; this is the key fact for §5.6.

#### Contrast with `fva_legacy`

`fva_legacy` (`lptools.py:285`) always solves the full `2n` LPs (parallel over an `SDPool` when
`processes > 1 and numr > 300`, else a serial warm-started loop), with no scan phase, no `v=0`
shortcut, no co-optimization, and no `reaction_list` scoping. On genome-scale models
`speedy_fva`'s Phase 1 typically resolves well over half of the `2n` objectives with a handful
of scan LPs, so the residual handed to Phase 2 is a fraction of `2n`. The two return identical
DataFrames (both post-process `|value| < 10⁻¹¹ → 0`); `fva_legacy` exists purely as a
debugging oracle.

### 5.6 Why FVA #2 is the ~117 s genome-scale bottleneck

On the canonical iML1515 gene-MCS run (SUPPRESS biomass ≥ 0.001, POPULATE, `max_cost=3`,
`gene_kos`), preprocessing's blocked/irreversible FVA — **FVA #2** — measures at **~117 s**,
the single largest preprocessing slice (Ch 11). Every structural reason for this is visible in
the three call sites and in `speedy_fva`'s control flow:

1. **It is whole-model — no `reaction_list`.** FVA #2 (`bound_blocked_or_irrevers_fva`,
   `networktools.py:1598`) forwards its kwargs to `fva` with *no* `reaction_list`, so
   `speedy_fva` must resolve **all `2n` objectives** — every bound of every reaction — because
   the bound-relaxation logic in §5.3 needs the true range of *every* reaction, not just
   knockable ones. FVA #1 is also whole-model but runs on the smaller pre-GPR network; FVA #3
   is scoped to `knockable_ids` and so solves only `2·|knockable|` objectives. FVA #2 is the
   only one paying the full `2n` on the *large* model.

2. **It runs on the GPR-extended model, which is much larger.** FVA #2 executes *after*
   `extend_model_gpr`, which injects a gene pseudoreaction per gene and additional
   pseudoreactions/pseudo-metabolites to encode the Boolean AND/OR structure (Ch 4). On
   iML1515 this roughly doubles the reaction count relative to the metabolic-only network FVA #1
   saw. The log line's totals (`1825` + `2258` + `2150` + …) reflect a network of several
   thousand reactions. More reactions ⇒ more objectives *and* larger per-LP factorizations.

3. **Internal compression is disabled (`compress=False`).** Because the model is already
   compressed by COMPRESS #2, FVA #2 passes `compress=False`, so `speedy_fva` does **not** run
   its own coupled-lumping pass — it solves LPs at the full GPR-extended dimension rather than a
   reduced one. This is correct (re-compressing the rational-bound model would be wasteful and
   the caller needs bounds on the *actual* reactions), but it means no dimension reduction
   cushions the LP cost.

4. **Phase 2 likely drops below the parallel threshold.** `speedy_fva` parallelizes Phase 2
   only when `n_remaining ≥ 1000` (line 543). Phase 1's scan LPs are very effective at resolving
   the many trivially-bounded reactions of a GPR-extended model (huge numbers of forward-only
   reactions with `lb=0`, resolved by the `v=0` shortcut and the `min Σ|x|` scan), so the
   *residual* handed to Phase 2 can fall **below 1000** — at which point Phase 2 runs the
   **sequential, single-threaded** path (line 611), grinding through the residual individual LPs
   one at a time. A residual of a few hundred genome-scale LPs solved serially, each on a
   several-thousand-variable model, accounts for the bulk of the 117 s. (Phase 1's own push LPs
   are cheap thanks to dual-simplex warm-starting; the cost concentrates in the serial Phase 2
   tail.)

This makes FVA #2 a concrete, high-value **performance lever** (Ch 11). Candidate mitigations
that follow directly from the analysis above: force Phase 2 onto the parallel path even for
`n_remaining < 1000` (or lower the threshold) so the residual LPs use all cores; or restrict
FVA #2's objectives to the reactions whose bounds can actually matter downstream — although,
unlike FVA #3, it genuinely needs *all* reactions' ranges to relax bounds correctly, so a
`reaction_list` restriction is not directly applicable and any scoping must be justified against
the bound-relaxation semantics of §5.3. The safe, immediately-available win is parallelism on
the Phase 2 tail.


## 6. Dualization (the mathematical core)

Everything the strain-design MILP does to a *behaviour* — forbid it (SUPPRESS), keep it
possible (PROTECT), or force an inner optimizer to reach its optimum (OptKnock, RobustKnock,
OptCouple, DoubleOpt) — is expressed through one of two linear-programming duality operations
applied to a standard-form linear system. This chapter derives those two operations, states the
theorems they instantiate, and reads the code that builds them:

- `LP_dualize` (`strainDesignProblem.py:1028`) — the LP dual of a maximization LP, used to
  certify *optimality* of an inner problem by strong duality.
- `farkas_dualize` (`strainDesignProblem.py:1141`) — the Farkas (alternative-system) dual, used
  to certify *infeasibility* of an undesired flux region.

Both are the same matrix transpose with different bookkeeping, and `farkas_dualize` literally
calls `LP_dualize` (`strainDesignProblem.py:1188`). Understanding the one construction, and the
two theorems it serves, explains the entire block-assembly logic of `addModule`
(`strainDesignProblem.py:227`).

This chapter produces the **continuous rows** — dual variables, dual-feasibility constraints,
strong-duality equalities, and the primal blocks they are paired with. It does *not* attach the
binary intervention variables `z` to those rows; that is `link_z` (`strainDesignProblem.py:699`)
and is owned by Ch 7. Where the dual bookkeeping matrices `z_map_vars`, `z_map_constr_ineq`,
`z_map_constr_eq` are updated here, we explain *what they now point at* so Ch 7 can wire them, but
the actual big-M / indicator machinery is deferred there.

Notation follows Ch 1: the metabolic model has stoichiometry `S ∈ ℝ^{m×n}` over `n` (compressed)
reactions, flux vector `v ∈ ℝ^n`, steady state `Sv = 0`, bounds `lb ≤ v ≤ ub`. A *module* adds
extra linear constraints `V_ineq v ≤ v_ineq`, `V_eq v = v_eq` describing a flux behaviour.

### 6.1 Why dualize at all

A strain-design constraint is a statement about the *solvability* of an inner LP, and such
statements cannot be written directly as linear constraints on the outer variables.

- **SUPPRESS** demands: *after the knockouts encoded by `z`, the undesired region
  `{v : Sv=0, V_ineq v ≤ v_ineq, lb ≤ v ≤ ub}` is empty.* "This polyhedron is empty" is not a
  linear constraint on `v` — indeed there is no `v` to constrain. Farkas' lemma converts it into
  "there exists a dual vector `y` satisfying a *feasible* linear system," which *is* linear and can
  live in the outer MILP.

- **OptKnock / inner-objective PROTECT** demands: *the flux `v` is optimal for the inner objective
  `max c_inner^T v` over the (knocked-out) network.* "Is optimal" is a quantifier over all other
  feasible fluxes. LP strong duality collapses it to three linear conditions — primal feasibility,
  dual feasibility, and equality of the two objective values — all linear once the dual variables
  are introduced.

In both cases dualization is the device that turns a *nested optimization / feasibility quantifier*
into a flat system of linear (in)equalities that a single-level MILP can hold. The binary `z` then
switch individual rows of that flat system on and off (Ch 7), which is why the dual must be built so
that each `z` still maps cleanly onto the object (a reaction) it knocks out — the role of the
`z_map_*` matrices threaded through every function below.

### 6.2 LP duality refresher, in the exact standard form the code uses

#### 6.2.1 Primal standard form

Every primal the code dualizes is produced by `build_primal_from_cbm`
(`strainDesignProblem.py:971`) and has the shape

```
(P)   max  c^T x
      s.t. A_ineq x ≤ b_ineq        (dual multipliers μ)
           A_eq   x =  b_eq         (dual multipliers λ)
           lb ≤ x ≤ ub
```

with `x ∈ ℝ^{n}`. For a bare metabolic primal, `A_eq = S` (so `b_eq = 0`, `Sv=0`), `A_ineq` holds
the module's `V_ineq` rows, and `lb, ub` are the flux bounds (`strainDesignProblem.py:1013-1023`).

The **sense is maximization**. This is important and easy to get wrong: `LP_dualize`'s docstring
writes the format as `min{c'x}`, but the transform it implements is the dual of the *maximization*
`max c^T x`. This was verified directly (see §6.2.4): dualizing the metabolic primal with the
biomass objective and solving the returned dual reproduces the FBA optimum only under the max
reading. Throughout `addModule`, a maximize-sense inner objective is stored *negated* precisely so
that the downstream strong-duality equality comes out as a clean sum-to-zero (§6.4).

Variables carry a **sign class**, and it is the class — not the numeric bound values — that decides
the dual constraint sense. The code computes the three classes at `strainDesignProblem.py:1107-1109`
from the *original* bounds, before any bound is rewritten:

```
x_geq0 = { j : lb_j ≥ 0 and ub_j > 0 }     # sign-nonnegative
x_eR   = { j : lb_j < 0 and ub_j > 0 }     # free (both signs reachable)
x_leq0 = { j : lb_j < 0 and ub_j ≤ 0 }     # sign-nonpositive
```

A reversible reaction (`lb<0<ub`) is *free*; an irreversible forward reaction (`lb=0`) is
*nonnegative*; a strictly-reverse reaction is *nonpositive*. The finite, nonzero magnitudes of the
bounds are handled separately (see §6.2.3): they are *not* what selects the dual sense.

#### 6.2.2 Weak duality, strong duality, complementary slackness

For the pair (P) above and its dual (D) (constructed in §6.2.3),

- **Weak duality.** For any primal-feasible `x` and dual-feasible `y = (λ, μ)`,
  `c^T x ≤ b^T y` where `b = (b_eq, b_ineq)`. The primal max is bounded above by every dual value.
- **Strong duality.** If (P) has a finite optimum, so does (D), and the optima coincide:
  `max c^T x = min b^T y`. This is the theorem the bilevel modules exploit.
- **Complementary slackness.** At optimality, for each inequality either the primal row is tight
  (`A_ineq[i,:] x = b_ineq[i]`) or its multiplier vanishes (`μ_i = 0`); symmetrically for a
  sign-constrained primal variable `x_j` and its dual reduced-cost row. The MILP never encodes
  complementary slackness explicitly — it uses the equivalent strong-duality equality `c^T x = b^T y`
  (§6.4), which is one linear row instead of a disjunction per constraint and needs no extra binary
  variables. This is the deliberate design choice over a KKT/complementarity encoding.

The value of dualization is exactly the strong-duality clause: *primal feasibility ∧ dual
feasibility ∧ (`c^T x = b^T y`)* is, by the theorem, equivalent to "`x` is optimal for (P)" — a
statement with a universal quantifier, now written as flat linear rows.

#### 6.2.3 `LP_dualize` line by line

`LP_dualize(A_ineq_p, b_ineq_p, A_eq_p, b_eq_p, lb_p, ub_p, c_p, z_maps…)` returns the dual system
in the *same* standard container `(A_ineq, b_ineq, A_eq, b_eq, lb, ub, c, z_maps…)`, so that dualized
systems can themselves be re-dualized (RobustKnock does this — §6.5.3).

**Step 1 — inhomogeneous bounds become inequality rows** (`strainDesignProblem.py:1104-1114`). A
finite nonzero lower/upper bound is not left on the variable; it is appended to `A_ineq_p` as an
explicit row so it acquires its own dual multiplier:

```
lb_j finite, ≠ 0:   −x_j ≤ −lb_j        (row in LB, line 1111)
ub_j finite, ≠ 0:    x_j ≤  ub_j        (row in UB, line 1112)
A_ineq_p ← [A_ineq_p ; LB ; UB]         (line 1113)
b_ineq_p ← b_ineq_p + [−lb_j…] + [ub_j…] (line 1114)
```

Zero bounds and `±∞` bounds are skipped (an `x_j ≥ 0` reaction contributes no LB row; its
nonnegativity is carried by the *sign class*, not a row). This is why the sign class and the bound
magnitude are decoupled: sign → dual constraint sense; finite magnitude → an extra `≥0` dual
variable.

**Step 2 — variable class ⇒ dual constraint sense** (`strainDesignProblem.py:1116-1123`). Writing
the stacked primal constraint columns as `A[:,j] = (A_eq[:,j] ; A_ineq[:,j])` and the stacked dual
vector as `y = (λ ; μ)`, the transpose is split by class:

| primal variable `x_j` | class | dual row built | code |
|---|---|---|---|
| free `x_j ∈ ℝ` | `x_eR` | `A_eq[:,j]^T λ + A_ineq[:,j]^T μ = c_j` (equality) | line 1122-1123 |
| `x_j ≥ 0` | `x_geq0` | `−(A_eq[:,j]^T λ + A_ineq[:,j]^T μ) ≤ c_j`  i.e. reduced-cost row into `A_ineq` | line 1118, 1121 |
| `x_j ≤ 0` | `x_leq0` | `(A_eq[:,j]^T λ + A_ineq[:,j]^T μ) ≤ −c_j` | line 1120, 1121 |

The free-variable rows land in the dual's `A_eq` (equality — a free primal variable forces
stationarity exactly), the sign-constrained rows land in the dual's `A_ineq` (a one-sided
reduced-cost / dual-feasibility condition). This is the textbook correspondence

```
primal variable  →  dual constraint
   x ∈ ℝ         →       =
   x ≥ 0         →       (one-sided inequality)
   x ≤ 0         →       (one-sided inequality, opposite)
```

read off the columns of `[A_eq ; A_ineq]`.

**Step 3 — constraint class ⇒ dual variable class** (`strainDesignProblem.py:1124-1125`). The dual
variables are ordered `[λ (one per A_eq row) ; μ (one per A_ineq row)]` with bounds

```
lb = [−∞]·(#A_eq rows) + [0]·(#A_ineq rows)     (line 1124)
ub = [+∞]·(#A_eq rows + #A_ineq rows)           (line 1125)
```

So an **equality primal constraint → free dual variable** (`λ_i ∈ ℝ`), an **inequality primal
constraint → sign-constrained dual variable** (`μ_i ≥ 0`). The steady-state rows `Sv=0` therefore
produce *free* dual variables (the classical metabolite "shadow prices," unbounded in sign), while
every bound/module inequality produces a *nonnegative* dual variable. This is the symmetric partner
of the table above:

```
primal constraint  →  dual variable
      =            →       y ∈ ℝ
      ≤            →       y ≥ 0
```

**Step 4 — dual objective** (`strainDesignProblem.py:1126`). `c = b_eq_p + b_ineq_p`: the dual's
objective coefficients are the primal's right-hand sides, in the same `[λ ; μ]` order. Because
`b_eq = 0` for the pure `Sv=0` rows, only the module RHS and the finite-bound rows contribute — the
dual objective is a weighted sum of flux bounds and module thresholds.

**Step 5 — the `z`-map transposition** (`strainDesignProblem.py:1128-1133`). This is the part that
makes dualization *reusable inside a knockout MILP*, and it is the reason these functions carry the
three bookkeeping matrices everywhere. A knockout removes a *reaction*; in the primal a reaction is
a *variable*, but after dualization the same knockout must remove the corresponding *dual object*.
The maps are transposed accordingly:

```
# a knockable primal VARIABLE (reaction flux) becomes a knockable dual CONSTRAINT
z_map_constr_ineq ← [ z_map_vars_p[:, x_geq0] , z_map_vars_p[:, x_leq0] ]   (line 1130)
z_map_constr_eq   ←   z_map_vars_p[:, x_eR]                                 (line 1131)

# a knockable primal CONSTRAINT becomes a knockable dual VARIABLE
z_map_vars ← [ z_map_constr_eq_p , z_map_constr_ineq_p , 0(for the new LB/UB rows) ]  (line 1132-1133)
```

Reading it in words: reaction `j`'s flux variable maps, after dualization, onto its *reduced-cost
row* (its dual constraint); a knockable primal constraint maps onto its *dual multiplier*. The
appended LB/UB bound rows get zero columns in `z_map_vars` — their multipliers are never knocked out
directly (their knockout is handled through the flux variable they bound). The overlap guard at
`strainDesignProblem.py:1087-1099` enforces the invariant that no single `z` simultaneously flags a
variable *and* a constraint in the same block, which would make the transpose ambiguous.

**Step 6 — `reassign_lb_ub_from_ineq`** (`strainDesignProblem.py:1134`, defined at
`:1207`). After transposing, many dual `A_ineq` rows are single-entry (a reduced-cost row on a dual
variable with no metabolic coupling). This helper folds single-variable inequality rows back into
`lb/ub` on the dual variables, *except* where the row is flagged knockable (`z_map_constr_ineq`
nonzero), because a knockable row must remain an explicit constraint for `z` to switch. This keeps
the dual compact and is also where the "negative-ub / positive-lb stay as rows" subtlety lives
(shared with `prevent_boundary_knockouts`, Ch 7).

#### 6.2.4 What `LP_dualize` does and does not guarantee

`LP_dualize` returns the constraints that define the **dual feasible set** plus the dual objective.
It does *not*, on its own, deliver strong duality as a solved number — and it is not meant to. Two
verified facts pin this down (e_coli_core, biomass objective):

1. Solving the returned dual as a standalone LP gives objective value `0`, not the FBA optimum
   `0.873922`. The dual feasible set contains `y = 0` (with `c_p` on a nonnegative variable and
   `b_eq = 0`), so minimizing `b^T y` alone certifies nothing.
2. Strong duality appears only after the **coupling row** is added (§6.4). Assembling primal ⊕ dual
   ⊕ the single equality `c_v^T x + c_dual^T y = 0` and then *both* maximizing and minimizing
   biomass over the joint system returns `max = min = 0.873922` exactly — biomass is pinned to its
   FBA optimum.

So the correct mental model is: **`LP_dualize` supplies dual feasibility; the caller supplies the
objective-equality row; the strong-duality theorem does the rest.** The sign conventions in the
table of §6.2.3 are exactly those under which that composite is correct — this was checked
end-to-end, not merely per-row.

### 6.3 Farkas' lemma and the SUPPRESS infeasibility certificate

#### 6.3.1 The lemma

Farkas' lemma is the theorem of the alternative for linear systems. One standard form: exactly one
of the following holds —

```
(I)   ∃ x ≥ 0 :  A x = b
(II)  ∃ y     :  A^T y ≥ 0  and  b^T y < 0
```

Geometrically, (I) says `b` lies in the finitely-generated cone `{A x : x ≥ 0}`; (II) says a
hyperplane through the origin (normal `y`) has the cone on one side (`A^T y ≥ 0`) and `b` strictly on
the other (`b^T y < 0`) — a *separating* hyperplane. For the mixed `≤ / =`, sign-constrained system
the code uses, the corresponding alternative is: the primal region

```
{ x : A_ineq x ≤ b_ineq, A_eq x = b_eq, lb ≤ x ≤ ub }
```

is **empty** if and only if there exists a dual vector `y` that is *feasible for the homogeneous
dual* (the dual constraints with objective `c = 0`) and additionally makes `b^T y < 0`. Such a `y`
is a **Farkas certificate** (a separating / infeasibility certificate).

#### 6.3.2 `farkas_dualize`

`farkas_dualize` (`strainDesignProblem.py:1141`) builds precisely system (II) for the undesired
region. Its steps:

1. **Zero objective** (`strainDesignProblem.py:1185`): `c_p = [0,…,0]`. The certificate is about
   *feasibility*, not optimization, so there is no objective. This also removes the entire
   reduced-cost RHS from the dual constraints of §6.2.3 (all right-hand sides `±c_j` become `0`),
   leaving the *homogeneous* dual `A^T y ≥ 0 / = 0`.

2. **Dualize** (`strainDesignProblem.py:1188-1191`): call `LP_dualize` with that zero objective.
   The returned `(A_ineq_d, b_ineq_d, A_eq_d, b_eq_d, lb_f, ub_f, c_d, …)` is the homogeneous dual;
   crucially `c_d = b_eq_p + b_ineq_p` is the *primal right-hand side vector* `b`.

3. **Normalization row** (`strainDesignProblem.py:1192-1194`): append one inequality

   ```
   A_ineq_f = [ A_ineq_d ; c_d ]          (line 1193)
   b_ineq_f =   b_ineq_d + [ −1 ]         (line 1194)
   ```

   i.e. `c_d^T y ≤ −1`, which is `b^T y ≤ −1`. This is the `b^T y < 0` clause of Farkas' lemma,
   with the strict inequality replaced by a fixed slack `≤ −1`. A knockable-column of zeros is added
   to `z_map_constr_ineq` for this new row (`strainDesignProblem.py:1201`) — the normalization row
   is structural and never itself knocked out.

The result is a *feasibility* system in `y`: it is solvable exactly when the undesired region is
infeasible. Making the undesired region infeasible *after knockouts* therefore reduces to keeping
this dual system feasible after the same knockouts — which is a set of ordinary linear rows the MILP
can hold, with `z` switching the rows that correspond to knocked reactions (via the transposed
`z_map` from §6.2.3). This is the `SUPPRESS` branch: `addModule` calls `farkas_dualize` at
`strainDesignProblem.py:668` and sets a zero module objective `c_i` at `:670`.

#### 6.3.3 Why the certificate is unbounded by nature, and the normalization row

A Farkas certificate is a **recession ray, not a point.** If `y*` satisfies `A^T y* ≥ 0` and
`b^T y* < 0`, then for any scalar `α > 0`, `α y*` satisfies `A^T(α y*) ≥ 0` and `b^T(α y*) < 0` as
well — the homogeneous constraints and the strict sign are both scale-invariant. The certificate
lives on an open ray through the origin; the dual variables are **intrinsically unbounded** (the
code sets `ub = +∞` and, for the `Sv=0`-derived duals, `lb = −∞`; §6.2.3, line 1124-1125).

The `c_d^T y ≤ −1` normalization does two jobs at once:

- **Pins the scale.** Without it, `y = 0` is feasible for the homogeneous system (`A^T·0 ≥ 0`), and
  `0` certifies nothing. Requiring `b^T y ≤ −1` forces `y` strictly off the origin and onto the
  ray, turning "`∃ y : b^T y < 0`" (an open condition, awkward for a solver) into the closed,
  numerically stable "`∃ y : b^T y ≤ −1`." Any true certificate can be rescaled to satisfy it, so no
  certificate is lost.
- **Fixes the orientation.** It selects the half-line with `b^T y < 0`, discarding the trivial
  `y = 0` and the wrong-sign ray.

A direct performance consequence follows from the unboundedness: **FVA-style bound tightening cannot
bound these dual variables.** The preprocessing FVA (Ch 5) tightens variable ranges by
maximizing/minimizing each variable over the polytope; for a Farkas dual variable that range is
`(−∞, +∞)` by construction (the feasible set is a cone, scale-free), so FVA returns `±∞` and buys
nothing. In `link_z` (Ch 7) this is exactly why the SUPPRESS dual rows end up as **indicator
constraints** rather than big-M: the per-constraint bounding LP that would supply a finite `M`
returns `±∞`, and the code's `self.M = inf` default routes an unbounded row to a native indicator.
This is emergent from the cone geometry, not a hard-coded "SUPPRESS ⇒ indicator" switch.

#### 6.3.4 The `b^T y ≠ 0` caveat

The docstring (`strainDesignProblem.py:1151-1158`) flags an unimplemented special case. When the
undesired region is described purely by equalities `A x = b` with **all variables free**
(`x ∈ ℝ^n`) and **`b ≠ 0`**, the correct Farkas alternative requires `b^T y ≠ 0` (not `b^T y < 0`):
an all-equality, all-free system `Ax=b` is infeasible iff there is a `y` in the left null space of
`A` (`A^T y = 0`) with `b^T y ≠ 0`, and both signs of `b^T y` are valid certificates because the
equality has no orientation. Forcing `b^T y ≤ −1` only captures the `b^T y < 0` half.

The code deliberately keeps the `≤ −1` form and notes the omission is benign in practice:

1. The case is rare. Metabolic primals mix `Sv=0` (equalities) with bound and module *inequalities*,
   so pure all-equality/all-free undesired regions essentially do not arise; and `Sv=0` itself is
   homogeneous (`b_eq = 0`), contributing nothing to `b^T y`.
2. Where it did matter, splitting each equality `A_i x = b_i` into `A_i x ≤ b_i` and `−A_i x ≤ −b_i`
   (two `≥0` dual variables) would recover a `b^T y < 0` certificate. The docstring judges the split
   unnecessary and keeps the single-sided normalization.

For the reader modifying this path: if you ever construct a SUPPRESS module whose region is
equality-only with a nonzero RHS and free variables, the `≤ −1` normalization can miss certificates
of the opposite sign — the split is the fix.

### 6.4 Strong-duality encoding of bilevel problems

#### 6.4.1 The coupling row

An inner optimizer `max c_inner^T v` over the network is encoded by pairing the inner **primal**
with its **dual** and forcing their objectives equal. By strong duality (§6.2.2), for
primal-feasible `v` and dual-feasible `y`,

```
c_inner^T v = b^T y     ⟺     v is optimal for the inner LP.
```

The `≤` direction is weak duality (always true); the `=` case is attained only at the common
optimum. So adding the single equality row `c_inner^T v − b^T y = 0` on top of primal feasibility
(`v` in the network) and dual feasibility (`y` in the `LP_dualize` output) is *exactly* the
statement "`v` maximizes the inner objective."

In the code this equality appears as a **sum**, not a difference, because the maximize-sense inner
objective is stored negated. Concretely, in the inner-objective branch
(`strainDesignProblem.py:247-299`):

- `c_in` is the inner objective, negated when the sense is `MAXIMIZE` (the default):
  `c_in = −c_inner` (`strainDesignProblem.py:250-253`).
- `build_primal_from_cbm` builds the region primal `_v` with objective `c_v = c_in`
  (`strainDesignProblem.py:255-256`).
- `LP_dualize` dualizes the *unconstrained* inner primal and returns `c_inner_dual = b` (the primal
  RHS) as the dual objective (`strainDesignProblem.py:258-262`).
- The exact-optimality coupling (`strainDesignProblem.py:288-294`) block-diagonalizes the region
  primal with the dual and appends

  ```
  A_eq row:  [ c_v | c_inner_dual ] · [v ; y] = 0        (line 293)
  ```

  i.e. `c_v^T v + c_inner_dual^T y = 0`. With `c_v = −c_inner` this reads
  `c_inner^T v = c_inner_dual^T y = b^T y` — the strong-duality equality. The negation is bookkeeping
  that turns "objectives equal" into a clean sum-to-zero row.

**Verification.** Reproducing this exact assembly on e_coli_core with the biomass inner objective,
then optimizing biomass in both directions over the joint system, yields `max = min = 0.873922` —
biomass is forced onto the inner-optimal face (the FBA value), confirming the sign convention and
the whole construction end to end. (Contrast: solving the dual alone gives `0`; §6.2.4.)

#### 6.4.2 Exact vs relaxed inner optimality

The inner problem need not be solved *to* the optimum — only *near* it. The module carries
`INNER_OPT_TOL ∈ (0, 1]` (default `1.0`, exact), handled at `strainDesignProblem.py:264-299`:

- **Exact (`inner_opt_tol = 1.0`, `strainDesignProblem.py:288-299`).** The single equality row of
  §6.4.1: `c_v^T v + c_inner_dual^T y = 0`. The optimizing flux must land on the inner-optimal face.

- **Relaxed (`inner_opt_tol < 1.0`, `strainDesignProblem.py:265-287`).** Being within a fraction
  `tol` of the optimum is not, by itself, a linear condition — you still need to *know* the optimum.
  The code introduces a second, **reference** copy of the inner primal (variables `x_ref`) whose only
  job is to attain the true optimum and anchor the dual. Two rows are added:

  ```
  equality  (anchor):  c_inner^T x_ref + c_inner_dual^T d = 0     (line 273, dual at optimum)
  inequality (relax):  c_v^T v + tol · c_inner_dual^T d ≤ 0       (line 271, actual ≥ tol·optimal)
  ```

  The anchor equality pins the dual `d` to the true inner optimum via the reference primal `x_ref`;
  the relaxed inequality then requires the *actual* flux `v`'s inner-objective value to be at least
  `tol` times that optimum. The three blocks — actual primal `_v`, reference primal `_inner`, dual
  `_dual` — are block-diagonalized at `strainDesignProblem.py:275-283`, and their `z`-maps
  concatenated (the anchor/relax rows get zero knockable columns). This reference-primal pattern
  recurs verbatim in the relaxed *outer* objective (`strainDesignProblem.py:604-631`) and in
  DoubleOpt (§6.5.5).

#### 6.4.3 Optional outer objective on PROTECT/SUPPRESS

A PROTECT or SUPPRESS module may itself carry an *outer* objective to be optimized over the
inner-optimal set (`strainDesignProblem.py:584-643`). The already-assembled bilevel `_p` (region
primal ⊕ inner dual) is dualized *again* by `LP_dualize` with the outer objective `c_out`
(`strainDesignProblem.py:598-601`), and coupled by the same strong-duality equality
(`strainDesignProblem.py:632-643` exact, `:604-631` relaxed with a reference copy of the whole `_p`).
Nesting `LP_dualize` on an already-dual system is possible precisely because it returns its output in
the same standard container it consumes (§6.2.3) — the transform is closed under composition.

### 6.5 One dualization, reused across every module type

The reason a single `LP_dualize` (plus its zero-objective specialization `farkas_dualize`) suffices
for all of MCS, OptKnock, RobustKnock, OptCouple, and DoubleOpt is a structural one:

> **Every strain-design assertion reduces to one of two primitives — "this LP attains its optimum"
> (primal + dual + strong-duality equality) or "this region is infeasible" (Farkas dual +
> normalization) — and both primitives are instances of dualizing a standard-form system.**

Because `LP_dualize` (a) consumes and produces the same container, (b) transposes the `z`-maps so
knockouts survive the transform, and (c) can be applied to its own output, the module builders in
`addModule` are just different *stackings* of the two primitives. The following subsections walk each.

#### 6.5.1 Inner-objective PROTECT / SUPPRESS (`strainDesignProblem.py:247-299`)

One strong-duality link. Build region primal `_v` (with the module's desired/undesired constraints),
dualize the unconstrained inner primal, couple with `c_v^T v + c_inner_dual^T y = 0` (§6.4.1). Then
dispatch by type (`strainDesignProblem.py:658-674`): **PROTECT** treats the coupled system as a raw
feasibility block (`reassign_lb_ub_from_ineq`, no MILP objective, `strainDesignProblem.py:658-666`);
**SUPPRESS** wraps the coupled system in `farkas_dualize` to demand its *infeasibility*
(`strainDesignProblem.py:667-674`). The same bilevel `_p` thus serves both "keep the inner-optimal
production reachable" and "make inner-optimal-with-target-production impossible," differing only in
which final primitive (feasibility vs Farkas) is applied.

#### 6.5.2 OptKnock — bilevel max-min (`strainDesignProblem.py:300-361`)

OptKnock maximizes an outer objective `c_out` over the *inner-optimal* flux set:
`max_z max_{v ∈ argmax c_inner^T v} c_out^T v`. Construction:

1. Region/inner primals and the inner dual are built as in §6.5.1, coupled by strong duality
   (`strainDesignProblem.py:308-339`), giving a system whose feasible set is exactly the
   inner-optimal face.
2. The **whole coupled inner** is then dualized once more with the outer objective `c_out_in_p`
   (`strainDesignProblem.py:340-342`), and the outer problem `_r` is joined to that second dual by a
   further strong-duality equality (`strainDesignProblem.py:343-354`). Bounds are reassigned
   (`strainDesignProblem.py:355-357`) and the outer objective set (`strainDesignProblem.py:358-361`,
   and the final MILP objective at `:675-685`).

The max-min is thus two `LP_dualize` calls: one to characterize the inner-optimal face, one to turn
the maximization *over* that face into flat rows.

#### 6.5.3 RobustKnock — three levels, two nested dualizations (`strainDesignProblem.py:300-361`)

RobustKnock is the *pessimistic* OptKnock: `max_z min_{v ∈ argmax c_inner^T v} c_out^T v` — it
guards against the worst production the cell might choose among its growth-optimal fluxes. The extra
`min` over the inner-optimal set is the third level. The code (same branch as OptKnock, distinguished
by `MODULE_TYPE == ROBUSTKNOCK`) dualizes the inner primal (`strainDesignProblem.py:322-324`), builds
the combined inner (region ⊕ inner-dual coupled, `strainDesignProblem.py:325-339`), then **dualizes
that combined system with the negated outer objective** (`strainDesignProblem.py:340-342`, the joint
min-max), and finally connects the outer primal `_r` to the dualized combined inner
(`strainDesignProblem.py:343-354`). Two nested `LP_dualize` calls convert the three-level
max-min-max into a single flat system; the inner `min` is expressed by dualizing it into a `max`
that can share the outer maximization's sense.

#### 6.5.4 OptCouple — growth-coupling distance (`strainDesignProblem.py:539-582`)

OptCouple maximizes the *gap* between the inner (growth) optimum *with* target production and the
inner optimum *without* it — a design where product synthesis is forced by growth. It builds two
bilevel systems: the production one (`_p`, inherited from the OptKnock-style block above) and a
**no-production** one (`_b`), the latter constructed by adding the production reaction fixed to zero
(`V_eq = prod_eq, v_eq = [0]`, `strainDesignProblem.py:544-545`), building its primal, and dualizing
it (`strainDesignProblem.py:546-548`) with its own strong-duality coupling
(`strainDesignProblem.py:549-561`). The two bilevel systems are block-diagonally joined
(`strainDesignProblem.py:562-571`), an optional minimum growth-coupling potential is enforced as an
inequality on the *difference* of the two inner objectives (`strainDesignProblem.py:572-576`), and the
MILP objective is set to **maximize that difference** `c_p − c_b` (`strainDesignProblem.py:581-582`).
Same primitive, instantiated twice and subtracted.

#### 6.5.5 DoubleOpt — two parallel strong-duality links (`strainDesignProblem.py:362-538`)

DoubleOpt enforces two optimality conditions on the *same* primal flux simultaneously (e.g. an inner
and an outer objective both attained). It builds the region primal `_v`
(`strainDesignProblem.py:377-378`) and dualizes two independent unconstrained inner primals — one per
objective `c_in` and `c_in2` (`strainDesignProblem.py:379-388`). Two strong-duality links are then
added over a shared column layout `[x_v | (x_ref1) | dual1 | (x_ref2) | dual2]`
(`strainDesignProblem.py:389-533`), each either exact (one equality row) or relaxed (anchor equality
+ relaxed inequality with a reference primal, §6.4.2), independently governed by `INNER_OPT_TOL` and
`OUTER_OPT_TOL`. DoubleOpt dispatches PROTECT-style as a feasibility block
(`strainDesignProblem.py:534-538`) with no MILP objective, and — note `strainDesignProblem.py:202` —
is grouped with PROTECT/SUPPRESS as an "MCS-type" module for objective selection (minimize
intervention cost). It is literally §6.5.1's link, laid down twice on one primal.

#### 6.5.6 The unifying picture

| module | primitive(s) | # `LP_dualize` | final assertion |
|---|---|---|---|
| PROTECT (plain) | region primal | 0 | region feasible |
| SUPPRESS (plain, MCS) | Farkas dual | 1 (via `farkas_dualize`) | region infeasible |
| PROTECT / SUPPRESS (inner obj) | 1 strong-duality link | 1 (+1 if outer obj) | inner-optimal flux feasible / infeasible |
| OptKnock | inner-optimal face + outer max | 2 | max over inner-optimal set |
| RobustKnock | inner-optimal face + inner min + outer | 2 (nested) | worst-case max-min-max |
| OptCouple | two bilevels, subtracted | 2 (one per bilevel) | max growth-coupling distance |
| DoubleOpt | two strong-duality links | 2 | two objectives jointly optimal |

Every row is a stacking of "assert an LP's optimum via primal + dual + strong-duality equality" or
"assert infeasibility via Farkas + normalization." Both are the single transpose-with-bookkeeping of
`LP_dualize`. That is what makes the dualization machinery reusable: the metabolic content changes,
the linear-algebra primitive does not.

### 6.6 Boundary with Chapter 7

Everything above produces **continuous rows only**: dual variables `y = (λ, μ)`, dual-feasibility
constraints, strong-duality equality rows, Farkas normalization rows, and the primal blocks they are
paired with — together with the `z_map_vars`, `z_map_constr_ineq`, `z_map_constr_eq` matrices that
record *which reaction's knockout removes which row or variable* after all the transposition. What is
**not** done here is attaching the binary intervention variables `z` to those rows. That is
`link_z` (`strainDesignProblem.py:699`), Ch 7: it reads the `z_map_*` matrices, splits knockable
equalities into directional inequalities, tries to bound each row with an LP to obtain a valid
big-M, and — where the bounding LP returns `±∞`, as it always does for the scale-free Farkas dual
rows (§6.3.3) — falls back to native indicator constraints. The emergent split noted throughout this
chapter (SUPPRESS's unbounded Farkas rows → indicators; PROTECT's finite-flux primal rows → big-M)
is a *consequence* of the bound structure this chapter's dualization produces, decided in Ch 7's
`self.M`/bounding-LP fork, not a per-type switch. Read this chapter for *what the rows mean*; read
Ch 7 for *how `z` turns them on and off*.


## 7. MILP construction & the z-linking

By the time this chapter's code runs, every strain-design *module* has been turned into a
self-contained linear (in)equality block — a Farkas infeasibility certificate for **SUPPRESS**, a raw
primal feasibility system for **PROTECT**, or a strong-duality sandwich for the bilevel types (Ch 6
owns that content). What remains is *assembly*: stacking those blocks into one matrix, attaching the
seed rows that account for intervention cost, and — the substance of this chapter — **wiring the binary
intervention variables `z` to the continuous rows** so that flipping `z_j` genuinely removes reaction
`j` from the flux system. That wiring is done two ways, native **indicator constraints** or **big-M**
linearization, and the choice between them is made per-constraint by a bound-computing LP. Getting it
right is what separates a correct, numerically well-behaved MILP from one that either admits phantom
solutions (M too small) or grinds through a useless LP relaxation (M too large).

All line references are to `strainDesignProblem.py` unless noted; the indicator container lives in
`indicatorConstraints.py`.

### 7.1 Notation and the shape of the master problem

The MILP variable vector is partitioned as

```
x = [ z ; y ]        z ∈ {0,1}^{num_z},   y ∈ ℝ^{n_cont}
```

with the `num_z` binaries occupying the *leading* columns (`self.idx_z = [0..numr-1]`,
`SDProblem.__init__`:164) and all continuous module variables `y` appended afterward. The final
`self.vtype = 'B'*num_z + 'C'*(z_map_vars.shape[1]-num_z)` (line 225) simply records that split.

`z_j = 1` means "intervention `j` is applied". For a **knockout** that is removal of reaction `j`; for
a **knock-in** the meaning is inverted (`z_inverted[j] = True`, set at line 148 from `ki_cost`), and
the sign machinery of §7.6 flips the coupling so that `z_j = 1` still reads as "the intervention is
made". One binary per *compressed* reaction: `self.num_z = numr` (line 144, `numr =
len(model.reactions)`), because at this point the model has already been through both compression
passes and GPR extension (Ch 3, Ch 4), so a "reaction" may be a lumped subnet or a gene
pseudoreaction. There is deliberately **no** separate binary per constraint or per variable — a single
`z_j` fans out to *all* rows and variables that reaction `j` controls, tracked by the three maps
introduced below.

Throughout, the master inequality system is `A_ineq · x ≤ b_ineq`, the equality system `A_eq · x =
b_eq`, with variable box `lb ≤ x ≤ ub`.

#### The three z-maps

Coupling bookkeeping is carried in three sparse matrices, each with `num_z` rows (one per binary) and
one column per constraint/variable of the system being tracked:

| map | shape | entry `(j, k)` meaning |
|---|---|---|
| `z_map_constr_ineq` | `num_z × #ineq` | `z_j` knocks inequality row `k` |
| `z_map_constr_eq`   | `num_z × #eq`   | `z_j` knocks equality row `k` |
| `z_map_vars`        | `num_z × #vars` | `z_j` knocks variable `k` (forces its flux to 0) |

The stored value encodes *both* which binary and the coupling polarity: **`+1` = knockout** (this row
disappears when `z_j = 1`), **`−1` = knock-in / addition** (the row is present only when `z_j = 1`),
`0` = no coupling. These are the maps `link_z` reads to decide, for every row, which `z` column to
write into and with which sense. They are the single source of truth linking the *combinatorial* layer
(`z`) to the *continuous* layer (fluxes, dual variables).

### 7.2 `SDProblem.__init__` — the seed rows, `num_z`, and the M switch

Before any module is added, `__init__` (line 95) lays down a 3-row skeleton over the `z` columns only.

#### The three fixed seed rows (line 156)

```python
self.A_ineq = sparse.csr_matrix([[-i for i in self.cost],   # row 0: idx_row_maxcost
                                  self.cost,                 # row 1: idx_row_mincost
                                  [0 for _ in range(num_z)]]) # row 2: idx_row_obj
self.b_ineq = [0.0, max_cost_or_sum, np.inf]
```

with `self.cost` the per-reaction intervention weight (KO cost, overwritten by KI cost where a KI is
defined; lines 145–151, `nan`→`0`). The three rows and their right-hand sides:

- **Row 0, `idx_row_maxcost`** (line 153): `−Σ_j cost_j · z_j ≤ 0`, i.e. `Σ_j cost_j z_j ≥ 0`. With
  non-negative costs this is slack at construction, but it is a *live lower bracket* on total
  intervention cost: the enumeration/optimization layer (Ch 8) raises its RHS to force the solver past
  cost levels already exhausted, turning it into `Σ cost_j z_j ≥ κ`. Keeping it as a permanent row
  means that lower bound can be tightened in place without restructuring the matrix.

- **Row 1, `idx_row_mincost`** (line 154): `Σ_j cost_j z_j ≤ b`, the **budget cap**. Its RHS is
  `self.max_cost` when the user supplied one, else `Σ_j |cost_j|` (line 158) — the latter is a
  vacuous cap (no design can cost more than the sum of all weights), present so the row always exists
  and can be tightened later. This is the constraint that makes "minimal" cut sets minimal-*enough*:
  no design exceeding the budget is admitted.

- **Row 2, `idx_row_obj`** (line 155): an all-zero placeholder with RHS `+∞`. For a pure MCS problem
  the objective is *minimize intervention cost* and lives in the objective vector `self.c` (lines
  202–205: `c[j] = cost[j]`), so this row stays inert. For **bilevel** problems (OptKnock, OptCouple,
  …) the outer objective is a flux expression, not a cost sum; the row is then overwritten with the
  objective coefficients (lines 210–212) and used by `fixObjective` (`strainDesignMILP.py`:239–241) to
  pin `c·x ≤ value` during the BEST search. Reserving row 2 up front lets that pin be a single
  `set_ineq_constraint` call rather than a matrix resize.

The naming (`maxcost` on the `≥ 0` row, `mincost` on the `≤ budget` row) reads backwards against the
RHS values and is best treated as an internal label; the *mathematics* is: row 0 lower-brackets and
row 1 upper-brackets the weighted intervention sum, and row 2 is the swappable objective slot.

The companion `z_map_constr_ineq` is initialised to `(numr × 3)` **zeros** (line 161): the seed rows
are *not knockable* — they constrain `z`, they are not part of any flux subsystem, so no `z` ever
"removes" them.

#### `self.M` — the master indicator/big-M switch (lines 118–126)

```python
bound_thres = max(|cobra_conf.lower_bound|, |cobra_conf.upper_bound|)
if self.M is None and solver == 'glpk':   self.M = bound_thres   # GLPK: no indicators
elif self.M is None:                       self.M = np.inf         # default
# else: user-supplied M kept as-is
```

`self.M` is the *fallback* big-M used only when the per-constraint bounding LP (§7.5) cannot produce a
finite bound. Its three regimes:

- **`inf` (default).** Rows with no finite bound get **no** big-M row; they fall through to native
  **indicator constraints** (§7.7). This is the preferred, numerically clean path.
- **cobra bound (GLPK).** GLPK has no indicator-constraint API, so `self.M` is forced finite (the
  cobra default bound, typically 1000) and *every* unbounded row becomes a big-M row with that
  constant. A warning is logged (line 121). This is the escape hatch that lets the open-source solver
  run at all, at the cost of a loose, uniform M.
- **user override.** Passing `M=<value>` in kwargs pins the fallback explicitly (for a solver that
  supports indicators, this forces big-M everywhere a bound is missing).

So `self.M` decides what happens to the rows the bounding LP *cannot* bound; the bounding LP decides
everything else. The emergent SUPPRESS→indicator / PROTECT→big-M split (§7.8) is a downstream
consequence of this, not a separate branch.

### 7.3 `addModule` — block-diagonal assembly (line 227)

Each module produces its own block `(A_ineq_i, b_ineq_i, A_eq_i, b_eq_i, lb_i, ub_i, c_i)` plus its
own three z-maps `z_map_*_i` (the Ch 6 dual/primal machinery; here we only care about *how* the block
joins the master). The join (lines 687–697) is:

```python
self.z_map_constr_ineq = hstack((self.z_map_constr_ineq, z_map_constr_ineq_i))  # 688
self.z_map_constr_eq   = hstack((self.z_map_constr_eq,   z_map_constr_eq_i))    # 689
self.z_map_vars        = hstack((self.z_map_vars,        z_map_vars_i))         # 690
self.A_ineq = sparse.bmat([[self.A_ineq, None],
                           [None,        A_ineq_i]]).tocsr()                     # 691
self.b_ineq += b_ineq_i
self.A_eq   = sparse.bmat([[self.A_eq, None], [None, A_eq_i]]).tocsr()          # 693
self.b_eq   += b_eq_i
self.c  += c_i;  self.lb += lb_i;  self.ub += ub_i
```

The constraint matrices grow **block-diagonally**: the new module's rows occupy new rows *and* new
columns, with explicit `None` (zero) off-diagonal blocks. The z-maps, in contrast, grow **only in
columns** (`hstack`) — they keep their `num_z` rows.

#### Why block-diagonal for the continuous part

Each module owns a **private set of continuous variables**. A SUPPRESS module's block is a Farkas dual
living in *dual* space (one dual variable per primal constraint of that module's flux system); a
PROTECT module's block is a *primal* flux vector `v`; a bilevel module carries primal flux *and* dual
variables. These variable sets are semantically disjoint — the flux that must stay feasible in a
PROTECT module has nothing to do with the dual ray that certifies infeasibility in a SUPPRESS module,
and two SUPPRESS modules certify infeasibility of two *different* behaviors, each needing its own ray.
Sharing continuous columns between them would impose spurious equalities (module A's flux = module B's
flux) that are simply wrong. Block-diagonal placement gives each module an independent copy of flux
space; the modules never see each other's continuous variables.

#### Why the z-columns are shared

The *only* thing all modules must agree on is **which reactions are cut** — that is the design, and it
is global. Those are the `z` columns, columns `0..num_z-1`, which are *not* re-created per module: the
seed skeleton put them there once, and every module's z-maps are `hstack`-ed onto the same `num_z`
rows. When `link_z` later writes a big-M coefficient into `A_ineq[row, z_j]`, it writes into that
shared leftmost block — filling the bottom-left "`None`" corner that `bmat` left as zeros. So the
architecture is: **block-diagonal in the continuous variables, dense-shared in the `z` variables**.
The design vector `z` is the coupling backbone; every module hangs off it. This is exactly the
structure that makes a *single* set of `num_z` binaries enforce *all* modules simultaneously — a
knockout that satisfies the SUPPRESS certificate is the *same* `z` that must leave the PROTECT flux
feasible.

`z_map_constr_ineq_i / z_map_constr_eq_i / z_map_vars_i` carried in with each module record precisely
which of that module's *new* rows/variables reaction `j` controls, so after the `hstack` the master
maps know, for every row in the assembled system, which `z` (if any) knocks it and with what polarity.

### 7.4 `prevent_boundary_knockouts` — why nonzero-sign bounds must be moved (line 1322)

This runs inside `build_primal_from_cbm` (line 1024), before dualization, on every primal flux
system. It repairs a specific incompatibility between the KO encoding and reactions whose flux is
*forced away from zero*.

#### The KO encoding and the failure

A knockout of reaction `j` is ultimately realized (link_z, §7.5–7.6) by driving its flux `v_j` to 0.
The mechanism *tightens the reaction's box toward 0*: for a variable with `ub_j > 0` it adds the row
`v_j ≤ 0` gated by `z`; for `lb_j < 0` it adds `−v_j ≤ 0`. This is valid **iff `0 ∈ [lb_j, ub_j]`** —
the KO row merely collapses the box onto a value the box already contains.

Now suppose the reaction has a **nonzero-sign bound**: `lb_j > 0` (obligatorily forward) or `ub_j < 0`
(obligatorily reverse). Then `0 ∉ [lb_j, ub_j]`. The variable's *own box bound* — which is a property
of the variable, not a constraint row, and is therefore **never multiplied by `z`** — keeps forcing
`v_j ≥ lb_j > 0` even when the KO row `v_j ≤ 0` is active. The two are contradictory: the "knockout"
does not remove the reaction, it renders the subsystem infeasible. Equivalently, in the
bound-multiplication view the docstring uses (multiply the bound by `z` to simulate the KO):
multiplying a bound that lies strictly on one side of 0 can never *reach* 0, so **the residual bound
still forces flux**.

#### The transformation (lines 1363–1377)

For each knockable column (`col_has_z`, from `z_map_vars`):

```
if lb_j > 0:   add row  -v_j ≤ -lb_j      (i.e.  v_j ≥ lb_j),   then set lb_j := 0
if ub_j < 0:   add row  +v_j ≤  ub_j      (i.e.  v_j ≤ ub_j),   then set ub_j := 0
```

The obligation is *moved out of the variable box and into an explicit inequality row*, and the box is
reset so that `0 ∈ [lb_j, ub_j]`. Concretely, `lb_j > 0` becomes box `[0, ub_j]` plus a standalone row
`v_j ≥ lb_j`. The new rows are appended with **zero z-columns** (line 1377: `hstack([z_map_constr_ineq,
zeros(numz, new_z_cols)])`) — they are **non-knockable**. That is the crucial point: the obligation is
now a fixed property of the flux system that survives into the dual as an ordinary constraint with an
unconditioned multiplier, rather than a variable bound that the z-machinery would try (and fail) to
multiply. The KO machinery can now cleanly collapse the (0-containing) box, and the moved row, carrying
no `z`, cannot be corrupted by the coupling.

(The docstring's summary says "negative lower / positive upper bounds"; the *code* moves `lb > 0` and
`ub < 0`. Trust the code: it is the nonzero-sign bounds — the ones that exclude 0 — that break the
encoding.)

In practice this fires rarely, because FVA preprocessing (Ch 5) has already relaxed non-binding bounds
to `±∞` and pinned irreversible/blocked reactions to 0; the survivors are the genuinely
obligatory-flux reactions, and this function is what keeps them knockable.

### 7.5 `link_z` — the heart of the chapter (line 699)

`link_z` transforms the assembled but *unlinked* system — where `z`-columns are still zero in every
module row — into a fully coupled MILP. Six steps.

#### Step 1: knockable equalities → ± inequality pairs (lines 718–734)

You cannot "relax an equality with a big-M" in one row: `a·x = b` gated off needs both `a·x ≤ b` and
`a·x ≥ b` to disappear. So each knockable equality (a nonzero column of `z_map_constr_eq`) is split:

```
a·x = b     →     a·x ≤ b   and   −a·x ≤ −b
```

Both new inequalities are gated by the *same* `z` (`z_eq = z_map_constr_eq[:, tuple(idx)*2]`,
line 723 — the column is duplicated). The originals are deleted from `A_eq` (lines 729–732). When the
gate is *inactive*, the pair re-imposes the equality exactly; when active, both directions relax. (If
this equality later lands on the indicator path with both directions unbounded, §7.7's lumping step
fuses the pair *back* into a single `'E'` indicator — the split is undone once it is no longer needed.)

#### Step 2: variable-KOs → inequality rows (lines 736–758)

A knockable *variable* (nonzero column of `z_map_vars`) is translated into an inequality that pins its
flux to 0 on the relevant side:

```
if ub_j > 0:   row  +1·v_j ≤ 0     (knock the positive side toward 0)
if lb_j < 0:   row  −1·v_j ≤ 0     (knock the negative side toward 0)
```

A reversible reaction (`lb_j<0<ub_j`) gets *both* rows. The gating column is `z_lb_ub =
−z_map_vars[:, cols]` (line 751) — **note the negation**. A plain KO reaction has `z_map_vars` entry
`+1`; negated to `−1`, which by the polarity convention (§7.6) means "enforce the row when `z=1`" —
exactly right: applying the KO (`z=1`) must *enforce* `v_j ≤ 0`. So the negation is what makes a
knockout gate correctly.

#### Step 3: the per-constraint bounding LP (lines 760–846)

For every knockable inequality row `a·x ≤ b`, we need a big-M large enough that when the gate relaxes
it, the row becomes non-binding. The valid/tight-M theory:

**A big-M is *valid* if `M ≥ max{ a·x : x ∈ P_relaxed }`** and **tight** if it equals that max, where
`P_relaxed` is the *most relaxed* polytope any knockout combination can produce. The code builds
`P_relaxed` by **dropping all knockable inequality rows** and keeping only the non-knockable
inequalities, all equalities, and the (continuous-variable) box:

```python
cont_vars = columns that are not z                                  # 765
M_A_ineq / M_b_ineq = A_ineq/b_ineq  with knockable rows removed     # 766-767 (the constraints of P)
M_A_eq  / M_b_eq    = all equalities                                 # 768-769
M_lb / M_ub         = box over continuous vars                       # 770-771
M_A_sparse / M_b    = the knockable rows themselves (the objectives) # 774-775
```

Dropping *every* knockable row is what makes `P_relaxed` a superset of every actually-reachable knocked
polytope (any real design drops only *some* rows), so `max a·x` over `P_relaxed` upper-bounds `a·x`
over any knocked subsystem — hence a **valid** M — and taking the exact max makes it **tight**.

Because solving one LP per knockable row is expensive, rows are triaged by sparsity (lines 787–802):

- **`nnz == 0`** (empty row): `max = 0`. (`n_zero`)
- **`nnz == 1`** (single variable `coeff·v_c`): `max = coeff·ub_c` if `coeff>0` else `coeff·lb_c`,
  read straight off the box; `∞` if that bound is infinite. (`n_single`)
- **`nnz ≥ 2`**: needs an actual LP, `max a·x = −min(−a·x)` over `P_relaxed`. (`n_lp`)

logged as `Bounding MILP: N constraints (X zero, Y single-var, Z need LP)` (line 805). Only the `n_lp`
rows hit the solver, optionally across a worker pool (`worker_compute` maximises `a·x` by minimising
`−a·x` and negating, line 1421–1422). Finite results are rounded *up* to 5 digits (`ceil(M·1e5)/1e5`,
line 839) to stay safely on the valid side; **infinite** results are replaced by `self.M` (line 839) —
the point where §7.2's switch takes effect.

#### Step 4: the fork at the M value (lines 847–864)

For each knockable inequality row, `Ms[row]` is now either a finite number or `self.M` (which may be
`inf`). The loop:

```python
for row in ...:
    if not isinf(Ms[row]) and not isnan(Ms[row]):     # finite M  → big-M row
        z_i   = z_map_constr_ineq[:, row].nonzero()[0][0]
        sense = z_map_constr_ineq[z_i, row]
        if sense > 0:   # z_i = 1 knocks out (KO)
            A_ineq[row, z_i] = -Ms[row] + b_ineq[row]
        else:           # z_i = 0 knocks out (KI convention)
            A_ineq[row, z_i] = Ms[row] - b_ineq[row]
            b_ineq[row]      = Ms[row]
```

Rows with `isinf(Ms[row])` are **skipped** here and picked up by the indicator path in step 5. The two
sense cases, written out (let `a·x ≤ b` be the row, `M = Ms[row]`):

- **`sense > 0` (KO, active when `z=1`)** — coefficient `b − M` in the z-column gives the row
  `a·x + (b−M)·z ≤ b`:
  - `z = 0`: `a·x ≤ b` — **enforced**.
  - `z = 1`: `a·x ≤ M` — relaxed to the tight maximum, hence **non-binding** (since `M = max a·x`).

  This is exactly tight: at the knocked state the bound equals the reachable maximum, not the looser
  `b + M` a naive formulation would use.

- **`sense < 0` (KI, active when `z=1`, absent when `z=0`)** — coefficient `M − b`, and `b` reset to
  `M`, giving `a·x + (M−b)·z ≤ M`:
  - `z = 1`: `a·x ≤ b` — **enforced** (reaction present).
  - `z = 0`: `a·x ≤ M` — relaxed, **non-binding** (reaction absent).

Both cases realize the same logic — "constraint holds in the active state, evaporates in the knocked
state" — with the polarity dictated by the `z_map` sign. The finite-M rows are now permanently part of
`A_ineq`; only their `z`-column entries changed.

#### Steps 5–6: indicators and cleanup (lines 866–948)

Every row still carrying `isinf(Ms[row])` (`knockable_constr_ineq_ic`, line 869) becomes a **native
indicator constraint**. First, a **lumping** pass (lines 874–925) undoes the step-1 split where it is
no longer useful: rows are canonicalised by the sign of their first nonzero entry (line 879), grouped
by an exact `(indices, data)` key (lines 891–899), and pairs found to be identical up to a global sign
flip (`ident_rows` product `−1`) — i.e. an `a·x ≤ b` and an `a·x ≥ b` on the same `z` — are fused into
a single equality indicator (lines 905–913); exact duplicates (product `+1`) drop one copy. The
survivors are packaged into an `IndicatorConstraints` object (lines 930–940) and *removed* from the
static `A_ineq`/`A_eq` (lines 943–948), because an indicator row is enforced by the solver's logic
engine, not by the LP matrix.

### 7.6 Indicator constraints (`indicatorConstraints.py`)

`IndicatorConstraints(binv, A, b, sense, indicval)` is a thin container (constructor at line 74) for
rows of the form

```
z_{binv[k]} = indicval[k]   ⇒   A[k]·x  <sense[k]>  b[k]
```

with `sense ∈ {'L','E','G'}` (≤, =, ≥). The container is populated in `link_z` (lines 930–940):

- **`binv`** — the `z` index gating each row, read from the nonzero of the row's `z_map` column.
- **`A, b`** — the surviving knockable inequality rows first (`'L'`), then the lumped equality rows
  (`'E'`): `sense = 'L'*n_ineq + 'E'*n_eq` (line 934).
- **`indicval`** — *which* value of the binary triggers enforcement, derived from the `z_map` polarity
  (line 937): `[0 if d == 1 else 1 for d in data]`. So a `z_map` entry of **`+1` (KO) → `indicval = 0`**
  (the constraint is enforced while the reaction is *present*, `z=0`, and released on knockout), and
  **`−1` (KI/addition) → `indicval = 1`** (enforced only when the reaction is *added*, `z=1`). The code
  comment (lines 938–939) states this mapping directly. This is the exact combinatorial analogue of the
  big-M sense cases in §7.5 step 4.

Semantically, `z = indicval ⇒ A·x <sense> b` and, when `z ≠ indicval`, the constraint is simply *not
present* — there is no slack variable, no large constant, nothing in the LP relaxation. The solver
enforces the implication by branching/logic.

### 7.7 Why indicators give a tighter LP relaxation than big-M

Take the KO row from §7.5, `a·x + (b−M)·z ≤ b`, and relax the binary to `z ∈ [0,1]` (what every LP
node in branch-and-bound actually sees). Rearranged:

```
a·x ≤ b + (M − b)·z
```

At a *fractional* `z` the right-hand side floats up proportionally to `z`: the relaxation lets `a·x`
exceed its true bound `b` by up to `(M−b)·z`. The feasible region of the relaxation is therefore
**enlarged**, and the enlargement grows *linearly with M*. A loose (large) M produces a weak
relaxation: the LP bound at each node is poor, branch-and-bound explores more nodes, and the wide
spread between M and the unit-scale flux coefficients degrades numerical conditioning (`FeasibilityTol`
/ `IntFeasTol` interactions, ill-scaled bases). This is the concrete cost of a bad M.

The indicator constraint has *no* continuous relaxation of the implication: at fractional `z` the
solver does not manufacture a proportional slack; it enforces `z=indicval ⇒ a·x ≤ b` combinatorially.
The relaxation it presents is at least as tight as the big-M one and usually strictly tighter, with no
M to condition on. That is why indicators are the default whenever the solver supports them, and why
the per-constraint tight M matters when it does *not*: the bounding LP of §7.5 exists precisely to
make each finite M as small as validly possible. This is also the payoff of Ch 5's FVA bound
relaxation — by pushing non-binding bounds to `±∞`, FVA makes the corresponding `max a·x` *infinite*,
which routes those rows to indicators (the tightest option, no M at all) instead of leaving them with a
finite-but-large M. Tight preprocessing and tight linearization are the same fight.

### 7.8 The emergent SUPPRESS→indicator / PROTECT→big-M split

A frequently observed pattern under the default `M = inf`: SUPPRESS modules end up almost entirely on
**indicator** constraints, PROTECT modules almost entirely on **big-M**. This is *emergent from bound
structure*, not a per-type branch anywhere in the code.

- A **SUPPRESS** module is a **Farkas dual** (`farkas_dualize`, Ch 6). Its variables are the components
  of an unbounded *dual ray*; the dual feasible set is a **homogeneous cone**, so the dual variables
  are unbounded above. The knockable rows are constraints on these unbounded dual variables, so their
  bounding LP returns `max a·x = +∞` → `Ms = self.M = inf` → **indicator**.

- A **PROTECT** module is a **raw primal** flux system (`reassign_lb_ub_from_ineq`, Ch 6). Its
  variables are fluxes with **finite FVA bounds**; the knockable rows are ordinary flux constraints,
  so their bounding LP returns a **finite** `max a·x` → **big-M** with that tight constant.

So the fork is decided entirely by whether `max a·x` over the relaxed polytope is finite — a property
of the *bounds*, funneled through the single `self.M`/bounding-LP mechanism in `link_z`. Change the
bound structure (e.g. cap the dual variables, or lose FVA relaxation on the primal) and the split
moves. On GLPK it collapses entirely: `self.M` is finite, so even the unbounded SUPPRESS rows get a
big-M, and there are no indicators at all. This is the mechanistic content behind the memory note that
SUPPRESS means *"cannot"* (make a behavior infeasible — certified by an unbounded dual ray, hence
indicators) and PROTECT means *"can"* (keep a behavior feasible — a bounded primal flux, hence big-M).

### 7.9 Final consolidation and the binary block

After `link_z`, the master problem is:

- **`A_ineq`** — seed rows 0–2, then the block-diagonal module rows, plus the eq→ineq rows (step 1)
  and var-KO rows (step 2), with finite-M `z`-column coefficients written in place; indicator rows have
  been *removed* (they live in `self.indic_constr`).
- **`A_eq`** — the non-knockable equalities (stoichiometry `S·v = 0`, fixed module equalities) plus any
  lumped equalities that stayed on the big-M path; indicator equalities removed.
- **`self.indic_constr`** — the `IndicatorConstraints` bundle.
- **`self.c`** — for a pure MCS problem, `c[j] = cost[j]` on the `z` block, 0 elsewhere (minimize
  intervention cost, `is_mcs_computation = True`, lines 202–205); for bilevel, `c` on `z` is 0 and the
  outer objective sits in seed row 2 (lines 206–212). `self.c_bu` backs it up (line 215).
- **`self.vtype = 'B'*num_z + 'C'*(z_map_vars.shape[1]-num_z)`** (line 225): the binary block is the
  leading `num_z` columns — the design variables `z`, which every module's coupling was wired into —
  and everything after is the continuous module variables (fluxes, dual rays) that hang off them
  block-diagonally.

The `ContMILP` snapshot (lines 191–195) stores the continuous projection (all columns except `idx_z`)
together with the three z-maps, so that a candidate design `z*` can be validated by substitution
without re-solving the full MILP (used by `verify_sd`, Ch 8). At this point the problem is a complete,
solver-ready MILP: binaries coupled to continuous rows through tight per-constraint big-Ms where
bounds are finite and native indicators where they are not.


## 8. Solving & enumeration

By the time this chapter's code runs, the strain-design problem is a fully assembled MILP: binary
intervention variables `z ∈ {0,1}^{n_z}`, continuous variables (fluxes, dual/Farkas variables, big-M
slacks), a stack of static inequality/equality rows, a set of indicator constraints or big-M rows
linking `z` to the continuous block (Ch 7), and an objective. What remains is the *search*: driving
the solver to produce not one design but a stream of **minimal, distinct, valid** designs, and doing so
with a strategy appropriate to the question being asked ("give me *any* design", "give me the *cheapest*
design", "give me *all* cheapest designs"). That orchestration lives almost entirely in
`strainDesignMILP.py`, in the three public entry points `compute` (ANY), `compute_optimal` (BEST) and
`enumerate` (POPULATE), plus the shared machinery `solveZ`/`solve`, `fixObjective`/`resetObjective`/
`setMinIntvCostObjective`, `add_exclusion_constraints`(`_ineq`) and `verify_sd`.

This chapter assumes the MILP already exists. How `z` attaches to the continuous rows (indicator vs
big-M, the bound-driven fork) is Ch 7; the dual/Farkas content of those rows is Ch 6. Here we take the
constraint matrix as given and study what happens *at solve time*.

### 8.1 The objective is both a vector and a constraint row

Everything in this chapter hinges on one structural decision made at MILP-construction time
(`strainDesignProblem.py:152-160`): the top three rows of `A_ineq` are reserved, and **row 2 is a copy
of the objective**.

```python
self.idx_row_maxcost = 0   # -cost·z ≤ 0            (unused lower guard)
self.idx_row_mincost = 1   #  cost·z ≤ max_cost     (the max_cost budget)
self.idx_row_obj     = 2   #  c·x    ≤ b_ineq[2]     (the objective, as a constraint)
self.A_ineq = sparse.csr_matrix([[-i for i in self.cost], self.cost,
                                 [0 for _ in range(self.num_z)]])
self.b_ineq = [0.0, max_cost, np.inf]              # row 2 rhs starts at +inf (inert)
```

So the objective exists in **two representations simultaneously**:

1. As the solver's objective vector `self.c` / `self.c_bu` (the backup copy, `strainDesignProblem.py:215`).
   This is what the branch-and-bound engine *minimizes*.
2. As inequality **row `idx_row_obj = 2`** of `A_ineq`, of the form `c·x ≤ β`. Initially `β = +∞`, so
   the row is inert (it constrains nothing).

`fixObjective` (`strainDesignMILP.py:239-241`) is nothing but a rewrite of that row:

```python
def fixObjective(self, c, cx):
    self.set_ineq_constraint(self.idx_row_obj, c, cx)   # row 2 := (c·x ≤ cx)
```

`resetObjective` (`:243-245`) restores the *vector* to `c_bu`; `setMinIntvCostObjective` (`:247-250`)
clears the vector and installs the intervention-cost objective `Σ cost_i z_i` over targetable `z`;
`clear_objective` (`solver_interface.py:385-389`) zeroes the vector.

**Why carry the objective as a row at all?** Because the algorithms below need to *decouple* two uses
of the same linear form:

- **as an optimization direction** — "minimize `c·x`" — which the solver's objective vector expresses;
- **as a feasibility cap** — "hold `c·x` at the value we just found and now search *within* that level
  set" — which only a constraint row can express.

You cannot express "fix the objective at its optimum and then optimize a *different* objective in the
resulting face" with a single objective vector. You need the optimum value pinned as a constraint while
the vector is repurposed. Row 2 is exactly that pin. Concretely, BEST does: solve with vector `= c_bu`
to get optimum `opt`; then `fixObjective(c_bu, opt)` pins `c_bu·x ≤ opt` (row 2) and
`setMinIntvCostObjective()` swaps the vector to `Σ cost·z`, so the next solve minimizes intervention
count *inside the optimal face*. Same linear form, two jobs, held apart by the vector/row duality.

A subtle consequence: because row 2 is a genuine `≤` inequality, "fixing" the objective at value `v`
really imposes `c·x ≤ v`, a *half-space*, not an equality. For a minimization that has already reached
its optimum `v`, the polytope is empty above `v`, so `≤ v` and `= v` coincide on the feasible set — the
inequality is enough and avoids the numerical fragility of an equality row.

### 8.2 `solveZ` / `solve`: what one solver call returns

`solveZ` (`strainDesignMILP.py:215-219`) is the workhorse wrapper:

```python
def solveZ(self):
    x, opt, status = self.solve()
    z = sparse.csr_matrix([round(x[i], 5) for i in self.idx_z])
    return z, x, opt, status
```

`self.solve()` (inherited from `MILP_LP`, `solver_interface.py:211-225`) dispatches to the backend,
then **rounds integer-typed variables** to the nearest integer (`int(round(x[i]))`) for all
`vtype=='B'/'I'` positions. `solveZ` additionally slices out just the binary block `idx_z`, rounds to 5
decimals, and returns it as a 1×`n_z` sparse row `z`, alongside the full primal `x`, the objective
value `opt`, and the solver status. The rounding matters: with `IntFeasTol`/`integrality` set to `1e-9`
(see §8.6) the solver's `z` are already all but exact, and rounding removes the last `1e-10`-scale dust
so that `z.indices` — the *support* — is exact set membership, which the exclusion cuts and
`verify_sd` rely on.

`status` is one of the solver-neutral constants `OPTIMAL`, `INFEASIBLE`, `UNBOUNDED`, `TIME_LIMIT`,
`TIME_LIMIT_W_SOL`, `ERROR` (mapped from raw CPLEX/Gurobi codes in the backends). The loops below treat
`OPTIMAL` and `TIME_LIMIT_W_SOL` as "a usable solution exists" and everything else as "stop".

### 8.3 The three approaches, their objective setups, and *why*

All three share the same skeleton: an outer `while` loop that repeatedly asks the solver for a design,
verifies it, records it, and adds an exclusion cut so the next iteration must produce something new. They
differ in **how the objective is set before each solve**, and that difference is the whole story of
ANY vs BEST vs POPULATE.

#### 8.3.1 ANY — `compute` (`strainDesignMILP.py:396-511`): feasibility-first, then subspace minimization

The user wants *some* valid design, not necessarily the smallest. Each outer iteration does two solves.

**Solve 1 — zero-objective feasibility (`:443-446`).**

```python
self.resetTargetableZ()          # all candidate z free again (ub=1)
self.clear_objective()           # objective vector := 0
self.fixObjective(self.c_bu, np.inf)   # row 2 := (c_bu·x ≤ +inf) → inert
z, x, _, status = self.solveZ()
```

The objective vector is **all zeros** and the objective row is inert. This is a **pure feasibility
problem**: "find any `(z, x)` satisfying all constraints". Why do this first?

- **It is cheap.** With a zero objective there is no optimality gap to close — the branch-and-bound tree
  terminates the instant it finds *one* integer-feasible leaf, because every feasible point has the same
  objective (0) and is therefore optimal. There is no lower-bound/upper-bound race, no need to prove
  optimality by exhausting the tree. For a genome-scale MCS MILP this is the difference between "descend
  to the first feasible leaf" and "search the whole tree to prove nothing cheaper exists". The first
  feasible `z` the solver stumbles onto is typically *far* from minimal (it may knock out dozens of
  reactions), but that is fine — we only wanted a foothold.

**Solve 2 — minimize intervention cost within the found subspace (`:470-492`).**

```python
cx = np.sum([c*x for c,x in zip(self.c_bu, x)])   # objective value at the found point
self.setMinIntvCostObjective()   # vector := Σ cost_i z_i over targetable z
self.setTargetableZ(z)           # forbid every z_i that was 0 in the found design
self.fixObjective(self.c_bu, cx) # row 2 := c_bu·x ≤ cx  (stay in this objective level)
while ...:
    z1, _, _, status1 = self.solveZ()
    ...
```

`setTargetableZ(z)` (`:256-258`) sets `ub=0` on every candidate `z_i` that the feasibility solve left
at 0. This **restricts the search to the subspace spanned by the reactions the first design already
touched** — the support of `z` and its subsets. Inside that tiny subspace the solver now *minimizes*
`Σ cost_i z_i`: it finds the cheapest sub-design that still satisfies all modules.

**What subspace minimization achieves.** The expensive part of an MCS MILP is choosing *which* reactions
to cut out of thousands of candidates — a huge combinatorial space. Once the feasibility solve has
handed us a concrete superset `K = supp(z)` of reactions that *demonstrably* suffice, the minimization
only has to decide which subset of `K` (at most `2^{|K|}` choices, and `|K|` is small — a handful to a
few dozen) is minimal. That is a dramatically smaller MILP: the binary variables outside `K` are pinned
to 0, so branch-and-bound never explores them. The two-phase structure trades one hard global
optimization for one cheap feasibility solve plus one small local optimization — and crucially it can
then *iterate*, peeling off multiple minimal designs from the same subspace in the inner `while` before
returning to the full space for a fresh foothold. This is why ANY is the fastest way to get a *stream*
of valid minimal designs when you do not care about global cost-optimality across designs.

Note the design is minimal *within the subspace* — ANY does **not** guarantee it is globally
cost-minimal (a cheaper design might live in a subspace the feasibility solve never visited). That is
precisely the guarantee BEST adds.

#### 8.3.2 BEST — `compute_optimal` (`strainDesignMILP.py:289-392`): global optimum, then fix and iterate

The user wants the **globally cheapest** design(s), in nondecreasing cost order. The first solve is *not*
a feasibility solve; it is a genuine global optimization (`:335-338`):

```python
self.resetTargetableZ()
self.resetObjective()            # vector := c_bu  (the real cost objective)
self.fixObjective(self.c_bu, np.inf)   # row 2 inert
z, _, opt, status = self.solveZ()
```

Here the objective **vector is the cost objective** `c_bu` and the solver must prove global optimality —
close the gap between the best incumbent and the lower bound. That is inherently more work than ANY's
feasibility solve (the whole tree may need pruning to certify no cheaper design exists), which is the
price of the stronger guarantee.

For a pure MCS problem (`is_mcs_computation`, `:342-351`) the objective *is* the intervention cost, so
the optimal `z` is already a minimal design; BEST verifies it, records it, adds the exclusion cut, and
loops — each iteration returns the next-cheapest design because the accumulated cuts push the solver to
progressively higher cost.

For a bilevel problem (OptKnock etc., `is_mcs_computation == False`, `:352-373`) the primary objective
is a *production* objective, not cost, so BEST does the same fix-and-reminimize trick as ANY but around
the **global** optimum: `fixObjective(c_bu, opt)` pins the optimal production value, `setMinIntvCostObjective`
switches to minimizing knockouts, `setTargetableZ(z)` restricts to the found subspace, and the inner
loop enumerates minimal-intervention designs that all achieve the optimal production value.

#### 8.3.3 POPULATE — `enumerate` (`strainDesignMILP.py:515-613`): native solution pool per cost level

The user wants **all equally-optimal designs at each cost level** — the exhaustive enumeration used for
the correctness gates (e_coli_core = 455 MCS, iML1515 393 gene-MCS). The objective setup is the same as
BEST (optimize, then fix the optimal value), but instead of extracting one solution per solve it calls
the solver's **native solution pool** via `populateZ` (`:221-237`) → `populate` (`solver_interface.py:241-252`).

For pure MCS (`:571`), the cost objective is already installed, so `enumerate` goes straight to
`populateZ(remaining)`. For bilevel (`:571-580`) it first optimizes the production objective, fixes it,
and swaps to the cost objective — then populates.

```python
z, status = self.populateZ(self.max_solutions - sols.shape[0])
for i in range(z.shape[0]):
    if all(self.verify_sd(z[i])):
        self.add_exclusion_constraints(z[i]); sols = vstack((sols, z[i]))
    else:
        self.add_exclusion_constraints(z[i])   # drop invalid, still exclude
```

`populateZ` (`:221-237`) collects the whole pool, rounds the binary blocks, and **deduplicates by
support** (two pool members with identical `z.indices` are the same design even if their continuous
tails differ — the same cut set can be certified by different Farkas rays / flux distributions). The
pool is configured to contain **only equally-optimal** members (pool gaps set to ~0, §8.6), so one
`populate` call returns every design at the current optimal cost. The outer loop then adds cuts for all
of them, re-optimizes to the *next* cost level, and populates again — walking up the cost ladder,
emitting a complete pool at each rung, until infeasible or `max_solutions` reached.

This is why POPULATE is the tool for the correctness gates: it is the only mode that provably returns
*all* minimal designs at each cost, so a count like "393" is meaningful. It is also the most expensive
mode, because filling a pool means the solver keeps searching after finding the optimum (see §8.7).

### 8.4 The iterative loop and integer cuts (the minimal-and-distinct guarantee)

All three modes are iterative: find a design, exclude it, repeat until infeasible. "Exclude it" is where
the **minimality** and **distinctness** guarantees are actually enforced, via two different exclusion
constraints chosen by whether the found design is valid.

#### 8.4.1 The superset-excluding cut — `add_exclusion_constraints` (`:162-181`)

Given a found binary design `z*` with support `K = {i : z*_i = 1}`, `|K| = k`, this routine handles
three cases:

**Case `k ≥ 2` (the classic no-good / integer cut, `:177-181`):**

$$\sum_{i \in K} z_i \le k - 1$$

Claim: a binary point `z'` violates this cut (is excluded) **iff `supp(z') ⊇ K`**, i.e. iff `z'` is
`z*` or any superset of it. Proof: `Σ_{i∈K} z'_i ≤ k-1` fails exactly when `Σ_{i∈K} z'_i = k`, and since
each `z'_i ∈ {0,1}`, that sum reaches `k` only if `z'_i = 1` for *every* `i ∈ K` — i.e. `K ⊆ supp(z')`.
Values of `z'` outside `K` are unconstrained by the cut. ∎

**Why exclude supersets, not just `z*`?** Because for MCS, *any superset of a valid cut set is itself a
valid cut set* — adding more knockouts cannot make a suppressed behavior feasible again (it can only
remove flux capability). A superset is therefore always **non-minimal** and must never be reported. The
single cut `Σ_{i∈K} z_i ≤ k-1` removes `z*` *and its entire up-set* in one row, guaranteeing that once a
minimal design is found, no bloated version of it can ever be returned. This is the mechanical heart of
the minimality guarantee. (The PROTECT constraints mean a superset is not *automatically* feasible in
the MILP, but excluding it is still correct and keeps the enumeration to minimal designs; the inner
subspace minimization is what ensures we found the *minimal* member of that up-set before cutting it.)

**Case `k = 1` (single-reaction cut, `:172-175`):**

```python
interv_idx = int(z[i].indices[0])
self.z_non_targetable[interv_idx] = True
self.set_ub([[interv_idx, 0.0]])
```

Instead of adding a row `z_{i*} ≤ 0`, it directly **pins `ub(z_{i*}) = 0`** and marks the reaction
non-targetable. This is strictly stronger and cheaper than a constraint row: setting the upper bound to
0 removes the variable from consideration entirely (presolve fixes it), and it excludes `z*` *and every
superset containing `i*`* — same up-set semantics as the `k≥2` cut, but implemented as a bound rather
than a row, so it does not grow the constraint matrix. A size-1 MCS means reaction `i*` alone suffices;
no design containing `i*` can ever be minimal-and-new, so banning `i*` outright is exactly right.

**Case `k = 0` (empty design, `:166-170`):** adds the row `Σ_i z_i ≤ -1`, which is **infeasible** for
any nonnegative `z`. This deliberately makes the MILP infeasible to force clean termination. It is only
reachable in degenerate setups (the "no interventions needed" case is caught earlier by the `verify_sd`
of the all-zero design at `:322`/`:429`/`:548`); the guard is defensive — some solvers reject genuinely
empty constraint rows, so a `-1` rhs is used rather than an empty row.

#### 8.4.2 The exact-pattern cut — `add_exclusion_constraints_ineq` (`:183-198`)

Sometimes we must exclude *exactly* `z*` but **not** its supersets:

$$\sum_{i \in K} z_i - \sum_{i \notin K} z_i \le k - 1$$

In code the row is built with coefficient `+1` on `i∈K` and `-1` on `i∉K`, rhs `k-1`. Claim: a binary
`z'` is excluded **iff `z' = z*` exactly**. Proof: the left side is `Σ_{i∈K} z'_i − Σ_{i∉K} z'_i`. It
reaches `k` (violating `≤ k-1`) only when `Σ_{i∈K} z'_i = k` **and** `Σ_{i∉K} z'_i = 0`, i.e.
`z'_i = 1 ∀ i∈K` and `z'_i = 0 ∀ i∉K` — the single point `z' = z*`. Any superset (which turns on some
`i∉K`, subtracting from the sum) survives; any subset (which turns off some `i∈K`) survives. ∎

**Why would we ever want to keep supersets?** When the found `z*` is **invalid** — the MILP produced a
`z` that its relaxation accepted but which `verify_sd` (§8.5) rejects. An invalid pattern's supersets
may well be *valid* designs, so we must not cut the whole up-set; we surgically remove only the exact
offending point and let the search revisit its supersets. This is the asymmetry that makes the loop
both **complete** (no valid design lost) and **minimal** (no non-minimal design kept):

| Found design | validity | exclusion used | removes |
|---|---|---|---|
| valid, minimal-in-subspace | `verify_sd` ✓ | `add_exclusion_constraints` | `z*` **and all supersets** |
| invalid (relaxation artifact) | `verify_sd` ✗ | `add_exclusion_constraints_ineq` | **exactly** `z*` |

You can see the branch explicitly in `compute` (`:484-490`) and `compute_optimal` (`:365-371`): valid →
superset cut + record; invalid → exact cut, no record.

### 8.5 `verify_sd`: re-checking validity in the true continuous subsystem

`verify_sd` (`strainDesignMILP.py:260-287`) is the referee. Given one or more binary designs, it
reconstructs — for each — the **continuous LP that the design actually induces** and checks whether the
strain-design intent is met, independently of the MILP's big-M/indicator machinery.

Mechanically, it uses the stored `cont_MILP` (the continuous-only slice of the MILP,
`strainDesignProblem.py:193-195`) together with three z→row/variable maps (`z_map_vars`,
`z_map_constr_ineq`, `z_map_constr_eq`). For a design `sol`:

```python
inactive_vars  = [ var for z_i,var,sense in zip(z_map_vars.row, .col, .data)
                        if np.logical_xor(sol[0, z_i], sense == -1) ]
active_vars    = [ i for i in range(...) if i not in inactive_vars ]
# same for ineqs and eqs
lp = MILP_LP(A_ineq=cont_MILP.A_ineq[active_ineqs][:, active_vars], ...,
             solver=self.solver, seed=self.seed)
valid[i] = not np.isnan(lp.slim_solve())
```

The `logical_xor(sol[z_i], sense == -1)` handles the **knock-in inversion**: for a normal knockout
(`sense == +1`) a `z_i = 1` deactivates the linked variable/row; for an addition/knock-in
(`sense == -1`, the sign flip installed at `strainDesignProblem.py:182-189`) the polarity is reversed.
The result is the set of variables and constraints that *remain live* after the design is applied.
`verify_sd` then builds the reduced LP over exactly those, and returns validity = "the reduced LP is
feasible" (`slim_solve()` not NaN).

**Why re-verify at all, when the MILP already enforces the modules?** Three reasons:

1. **Big-M / indicator slack.** With finite big-M (Ch 7) or indicator activation the MILP enforces the
   Farkas/primal conditions only to within `FeasibilityTol` (`1e-9` here, but scaled by M). A `z` can
   satisfy the *relaxed* certificate to tolerance yet not correspond to a genuinely infeasible/feasible
   continuous subsystem. `verify_sd` re-solves the **exact** continuous LP with no M and no tolerance
   fudge, catching these artifacts.
2. **Subspace minimization can overshoot.** The "minimize intervention cost in subspace" step can, in
   edge cases, drop a knockout the certificate needed, producing a `z` the MILP's relaxation still
   accepts but that is not truly valid. Re-verification is the guard that routes such a `z` to the
   exact-pattern cut.
3. **The all-zero pre-check.** At the top of each mode (`:322`, `:429`, `:548`) `verify_sd` is called on
   the empty design `csr_matrix((1, num_z))`; if the untouched strain already satisfies the modules, no
   interventions are needed and the mode returns `[{}]` immediately.

Because `verify_sd` builds a fresh `MILP_LP` per design per call, it is not free — for large models the
repeated LP feasibility solves are a measurable slice of enumeration time — but it is what lets the loop
trust the solver's output enough to add the *strong* (superset) cut, which is what keeps enumeration
tractable. Note it is passed `self.seed`, so even these auxiliary LPs are reproducible.

### 8.6 Solver parameters and determinism

#### 8.6.1 No MIP optimality gap is set — the 1e-4 consequence

Neither backend sets a MIP relative or absolute **optimality** gap. Grep the interfaces:
`cplex_interface.py:161-166` sets the *pool* gaps (`mip.pool.absgap`, `mip.pool.relgap`) and the
*integrality* tolerance, but **never** `mip.tolerances.mipgap`; `gurobi_interface.py:162-163` sets
`PoolGap`/`PoolGapAbs` but **never** `MIPGap`. Both solvers therefore run at their **default relative
MIP gap of `1e-4`**.

Consequence, stated plainly: the solver considers a MILP "solved to optimality" once the incumbent is
within `0.01 %` of the proven bound. For BEST and for each cost rung of POPULATE, the reported "optimal
cost" can in principle be off by up to `1e-4 · |opt|`. For integer intervention costs (the common case,
`cost_i = 1`) this is harmless — `1e-4` is far below the unit spacing between distinct cost levels, so
the *cost ordering* and the *set of designs at each level* are unaffected. But it is a latent hazard for
**non-integer or widely-scaled cost vectors**, where two genuinely different cost levels could fall
within `1e-4` relative of each other and be conflated. A developer tightening correctness for weighted
costs should set `MIPGap`/`mipgap` to 0 explicitly (accepting the extra time to close the last sliver
of gap). Do not confuse this with the `1e-9` values that *are* set: those are `OptimalityTol`/
`FeasibilityTol`/`IntFeasTol` (Gurobi) and `simplex.tolerances.optimality`/`feasibility` +
`mip.tolerances.integrality` (CPLEX) — LP-level and integrality tolerances, not the MIP optimality gap.

#### 8.6.2 The solution-pool parameters are inert for single `solve()`

The CPLEX pool parameters `mip.pool.intensity = 4`, `mip.pool.absgap = 0`, `mip.pool.relgap = 0`
(`cplex_interface.py:161-163`), and the Gurobi `PoolGap`/`PoolGapAbs = 1e-9` (`:162-163`), only take
effect during pool generation (`populate_solution_pool` / `PoolSearchMode = 2`). During an ordinary
`solve()` — which is all ANY and BEST ever call — the pool stays empty and these settings do nothing.
They matter **only for POPULATE**, where `intensity = 4` (CPLEX's most aggressive pool search) and
zero pool gaps mean "find *every* solution tied at the optimum". These are **verified inert for single
solve**; they are not a performance bug and predate the current code (2022). Gurobi's `populate`
additionally flips `PoolSearchMode = 2`, `NumericFocus = 2` on entry and resets them to `0` on exit
(`gurobi_interface.py:305-309`), so single solves see the defaults.

#### 8.6.3 Seed → branch-and-bound tree shape → why speed needs a distribution

The `seed` flows from the SD problem to the backend and lands on `randomseed` (CPLEX,
`cplex_interface.py:158`), `Params.Seed` (Gurobi, `:157`), and `randomization/randomseedshift` (SCIP,
`scip_interface.py:168`). If the user gives no seed, each backend draws one from `[0, 2^16)` and logs
it — so *even an unseeded run is reproducible after the fact*, given the logged seed.

The seed perturbs tie-breaking throughout branch-and-bound: which fractional variable to branch on when
several are equally attractive, which node to explore next, the order heuristics fire, how the simplex
breaks degenerate pivots. On a genome-scale MCS MILP the LP relaxation is **massively degenerate** (many
equally-good fractional `z`), so tie-breaking dominates the *path* the solver takes to the optimum. Two
seeds can produce wildly different tree sizes and hence wildly different wall-times for the *same*
instance and the *same* final design set.

The practical rule this forces: **any speed claim — ANY vs BEST, CPLEX vs Gurobi, and especially
POPULATE — must be measured over multiple seeds and reported as a distribution (median/IQR), not a
single number.** A single-seed "Gurobi is 4.4× faster" could be an artifact of one lucky/unlucky tree.
The numbers in §8.7 are useful as orders of magnitude but should be reproduced across seeds before any
optimization is judged to have helped. This is the single most important benchmarking discipline for
this code: never tune against one seed.

The `_trim_z_variables` step (`strainDesignMILP.py:107-149`) is a determinism-adjacent optimization
worth noting: it physically removes non-knockable (`ub=0`, `cost=0`) binary columns from the matrices
before the solver sees them, shrinking the binary count and keeping the B&B tree from carrying dead
variables. Solutions are expanded back to the original `z`-space afterward (`_expand_z_to_orig`,
`:151-160`).

### 8.7 Verified performance: the phase timeline and CPLEX vs Gurobi

For the canonical **iML1515 gene-MCS** problem (SUPPRESS biomass ≥ 0.001, POPULATE, `max_cost = 3`,
gene KOs) yielding **393 MCS** (package v1.18):

| Phase | Time | Notes |
|---|---|---|
| Preprocessing: blocked/irreversible FVA | **~117 s** | solver-agnostic, one-time |
| MILP build | **~4 s** | matrix assembly + `link_z` |
| Populate (enumeration) | **~1101 s** (CPLEX) | dominates |
| **Total** | **CPLEX 1241 s / Gurobi 280 s (≈4.4×)** | |

For **e_coli_core** (455 MCS) the whole thing is **~1.2 s** on CPLEX — small enough that phase structure
is irrelevant.

**Interpretation.** On iML1515, preprocessing FVA (~117 s) and build (~4 s) are essentially fixed costs
independent of the MILP solver; they are ~10 % of the CPLEX total. The remaining **~89 %** is the
**pool search** inside `populate`. So the thing that dominates genome-scale enumeration is *not* solving
a single MILP to optimality — a single feasibility or optimality solve is comparatively quick — it is
**exhaustively filling the solution pool at each cost level**: the solver must, after finding the optimal
cost, keep branching to enumerate *every* tied design and prove there are no more. That is intrinsically
harder than a single optimize, and it is where CPLEX and Gurobi diverge: Gurobi's pool search
(`PoolSearchMode = 2`) closes this instance ~4.4× faster than CPLEX's `populate_solution_pool` at
`intensity = 4`. The preprocessing FVA (Ch 5) is the second-largest lever and, being solver-agnostic, is
where portable speedups live; the pool search is a solver-quality question.

Because this 4.4× is a **single-seed** figure, per §8.6.3 it should be read as "Gurobi is materially
faster here", not as a precise constant — reproduce across seeds before quoting it as a benchmark.

**The discredited "big-M / indicators-catastrophic" dead-end.** An earlier performance hypothesis held
that native **indicator constraints** were catastrophically slow at genome scale and that forcing a
global **big-M** reformulation would fix it. This was investigated and **discredited** — do not repeat
it. Two reasons: (1) The dominant cost is pool enumeration (~89 % above), *not* the LP relaxation of the
z-linking, so swapping the linking mechanism cannot address the actual bottleneck. (2) Indicator
constraints give a **tighter** LP relaxation than big-M (Ch 7) — a valid big-M must be large enough to
never spuriously bind, which loosens the relaxation and generally *hurts* branch-and-bound, the opposite
of the hypothesis. Recall also (CONTEXT §, Ch 7) that under the default `M = inf`, SUPPRESS's unbounded
Farkas-dual rows *become* indicator constraints and PROTECT's finite-flux primal rows *become* big-M
**emergently** from the bound structure in `link_z` — there is no per-module type switch to "fix". The
lever that actually moves genome-scale time is faster pool search (solver choice) and cheaper
preprocessing FVA, not the linking encoding.

### 8.8 How SCIP and GLPK differ

Neither SCIP nor GLPK exposes a native optimal-solution pool, so `enumerate`/POPULATE is **emulated** at
the Python level, and both backends emit a warning steering the user toward `compute_optimal` instead
(`strainDesignMILP.py:553-562`).

**SCIP** (`scip_interface.py:282-336`) emulates `populate` by a solve-and-exclude loop: solve to the
optimum, add a constraint pinning `c·x ≤ min_cx` (optimality), exclude the found solution with an
exact-pattern inequality cut (`addExclusionConstraintIneq`), and repeat until infeasible — reconstructing
the pool one solution at a time. It supports indicator constraints natively (so no big-M is forced), and
its seed enters via `randomization/randomseedshift`. It is correct but slower than a native pool, hence
the "consider `compute_optimal`" advice.

**GLPK** (`glpk_interface.py:291-349`) emulates `populate` the same solve-and-exclude way, but with two
extra handicaps. First, GLPK has **no indicator constraints at all**, so z-linking is done entirely by
**big-M**, defaulting to the COBRA bound `M = bound_thres` (typically 1000, `strainDesignProblem.py:120-124`).
At genome scale this big-M is **too weak**: 1000 is simultaneously large enough to admit numerical slop
in the Farkas certificate (values that are "zero" only to `1e-3·M`) and small enough to occasionally
bind a flux it should not, so the relaxation is both loose and numerically fragile. Second, the emulated
pool is explicitly flagged **"instable"** in code and warnings. The practical upshot: GLPK is fine for
small models (e_coli_core-scale validation) but not a genome-scale enumeration engine — for iML1515-class
problems use CPLEX or Gurobi. The emulation cleans up after itself by freeing the auxiliary rows'
right-hand sides rather than deleting rows (`glpk_interface.py:334-343`), because GLPK row deletion
proved unstable.


## 9. Decompression & solution semantics

The MILP does not run on the model the user handed to `compute_strain_designs`. By the time
`SDMILP` is built (Ch 7), the network has passed through two lossless compression rounds (COMPRESS
#1 before GPR integration, COMPRESS #2 after — Ch 3), an optional GPR extension that turned genes
into pseudoreactions (Ch 4), and three FVA passes that pruned essential reactions and pulled out
size‑1 minimal cut sets (Ch 5). The binary intervention variables `z` therefore index **compressed
reactions of the GPR‑extended model**, not the original reactions or genes the user cares about.

Decompression is the inverse map. It takes each compressed intervention set the solver returned and
rewrites it in terms of the *original* model's reactions (and, for gene‑based problems, the original
genes), re‑injects the size‑1 MCS that never entered the MILP, re‑checks the max‑cost budget that
expansion can silently violate, and finally packages everything into an `SDSolutions` object whose
display methods hide the internal bookkeeping. This chapter covers the mechanics and the math of each
step, and the precise value‑encoding (`-1` / `+1` / `0` / `(nan,nan)`) that the downstream tooling —
and two open bugs, deferred to Ch 10 — depends on.

The entry point is `_decompress_solutions` (`compute_strain_designs.py:641`), called from the main
orchestrator at `compute_strain_designs.py:611` after the solve, and again from the
resume‑from‑pickle path at `compute_strain_designs.py:842`. The two workhorse routines it delegates
to live in `networktools.py`: `expand_sd` (`networktools.py:1465`) and `filter_sd_maxcost`
(`networktools.py:1537`).

### 9.1 Why decompression is needed, and the shape of the compression map

Compression merges reactions in two ways (Ch 3), and the merge is recorded step by step in a list
called `cmp_mapReac`. It is assembled by concatenating the two compression rounds' maps at
`compute_strain_designs.py:439`:

```python
cmp_mapReac = cmp_mapReac_1 + cmp_mapReac_2
```

Each element of `cmp_mapReac` is one *compression step* — a Python dict with the fields that
`compress_model` and `compress_ki_ko_cost` write:

| field           | meaning |
|-----------------|---------|
| `reac_map_exp`  | `{ cmp_id : { orig_id : factor, … } }` — for each reaction that exists *after* this step, the reactions *before* this step it stands for, each with a rational scaling `factor` |
| `parallel`      | `True` if this step lumped **parallel** reactions (same flux direction, scaled‑identical columns of `S`), `False` if it lumped **coupled** (flux‑coupled / sequential) reactions |
| `ko_cost`       | the knockout‑cost dict *in the pre‑step (finer) reaction space* |
| `ki_cost`       | the knock‑in‑cost dict *in the pre‑step reaction space* |

The `reac_map_exp` structure is produced by the compressors (`compress_model_coupled` documents the
return type as `{compressed_id: {orig_id: factor, …}}` at `compression.py:1979`; the parallel
compressor is analogous), and `parallel` is stamped on when the step is appended to the list
(`compression.py:1904` for parallel, `compression.py:1935` for coupled). The two cost dicts are
attached later by `compress_ki_ko_cost` (`networktools.py:1392`): `cmp.update({KOCOST: kocost,
KICOST: kicost})` records the cost vectors *as they stood entering that step*, before that step's
lumping rewrites them. This is the crucial invariant that makes reverse expansion self‑describing:
**step *k* carries exactly the cost dicts keyed by the reaction ids that step *k*'s expansion will
re‑introduce.**

The `factor` in `reac_map_exp` is the stoichiometric scaling that made the merge exact. It is used
when *constraints and objectives* are pushed into the compressed space (`compress_modules`,
`networktools.py:1350`: `c[0][new_reac] = Σ c[0][k]·old_reac_val[k]`) and when a compressed *flux
vector* is mapped back to originals. For **strain‑design expansion** the factor is not needed — a
knockout of a lumped reaction is a *set* decision (which originals to cut), not a numeric scaling — so
`expand_sd` iterates only the keys of `r_orig` and ignores the factors.

Why compress at all before solving, given that we must undo it here? Because the MILP's size — number
of binary `z`, number of continuous dual/primal rows, big‑M/indicator wiring — scales with the
compressed reaction count, and the branch‑and‑bound cost is super‑linear in that. Compression on
iML1515 removes the overwhelming majority of columns losslessly (the merged reactions are provably
flux‑coupled or scaled‑parallel, so no cut set is lost — see Ch 3). Solving small and expanding after
is strictly cheaper than solving large, *provided* the expansion is faithful. The rest of this
section is that faithfulness argument.

### 9.2 The math of reverse expansion (`expand_sd`)

A compressed solution is a dict `m = { cmp_id : val }` where `val ∈ {-1, +1, 0}` (the encoding is
§9.5). Expansion must invert the *composition* of the step maps. Compression built the compressed
reactions by applying step 1, then step 2, …, then step `L`; so a compressed id in the final space is
the image of step `L ∘ … ∘ step 1`. To recover originals we apply the inverse steps in the opposite
order — step `L`⁻¹ first, then step `L−1`⁻¹, …, ending at step 1⁻¹. `expand_sd` does exactly this by
reversing the list once at the top (`networktools.py:1486`):

```python
cmp_map = cmp_mapReac[::-1]
for exp in cmp_map:
    reac_map_exp = exp["reac_map_exp"]
    ko_cost      = exp[KOCOST]
    ki_cost      = exp[KICOST]
    par_reac_cmp = exp["parallel"]
    for r_cmp, r_orig in reac_map_exp.items():
        if len(r_orig) > 1:
            for m in sd.copy():
                if r_cmp in m:
                    ...
```

`sd` is a *list* of solution dicts that grows during the loop; one compressed design can fan out into
several expanded designs. Only genuinely lumped reactions (`len(r_orig) > 1`) need work — a
compressed id that stands for a single original is renamed implicitly because the same string id was
kept, so no rewrite is required. When a design `m` mentions a lumped id `r_cmp`, the value `val =
m[r_cmp]` is popped and the members of `r_orig` are re‑introduced. There are four cases, and they
split on the KO/KI sign of `val` **crossed with** whether the step was parallel or coupled — because
the biology of "what does cutting/adding the group mean for its members" differs between the two merge
types.

#### Case KO of a **parallel** group (`val < 0`, `par_reac_cmp` True)

Parallel reactions carry flux in fixed proportion because their `S`‑columns are scalar multiples of
one another; they are, metabolically, redundant routes for the same conversion. To *suppress* the
group you must remove **every** knockable member — leaving any one open leaves the conversion
possible. So expansion produces **one** design that knocks out all knockable members
(`networktools.py:1499‑1504`):

```python
if par_reac_cmp:
    new_m = m.copy()
    for d in r_orig:
        if d in ko_cost:
            new_m[d] = val         # -1 on every knockable member
    sd += [new_m]
```

The `if d in ko_cost` guard matters: a member that is not knockable (no KO cost — e.g. it was made
essential, or the user never offered it as a target) is simply not added, and the group is still
considered "knocked out" to the extent the modeller allowed. This is why the pre‑step `ko_cost`
dict is carried on the step.

#### Case KO of a **coupled** group (`val < 0`, `par_reac_cmp` False)

Coupled (flux‑coupled) reactions must all carry flux together in every steady state — `v_i = 0 ⇔ v_j
= 0` for members of the group. Therefore killing **any single** member forces the whole group to
zero. Cutting the group is not "cut them all"; it is "cut *one*, your choice." Each choice is a
distinct, minimal strain design, so expansion **branches** — it emits one new design per knockable
member (`networktools.py:1505‑1510`):

```python
else:  # coupled
    for d in r_orig:
        if d in ko_cost:
            new_m = m.copy()
            new_m[d] = val         # a separate design per member
            sd += [new_m]
```

This is the source of solution multiplicity: a single compressed KO of a coupled 4‑reaction group
becomes up to four original‑model MCS. All are correct and all are minimal — they are genuinely
different interventions with the same downstream effect. Keeping them distinct (rather than reporting
one representative) is what lets the user pick the intervention that is easiest to realise in the lab.

#### Case KI of a group (`val > 0`)

Knock‑ins mirror the KOs with parallel/coupled swapped, because "adding capability" is dual to
"removing it":

- **Parallel KI** (`networktools.py:1512‑1520`): the parallel members are interchangeable routes, so
  adding *any one* suffices. Expansion branches — one design per KI‑able member — and, in each branch,
  explicitly marks the *other* members as **not added** with value `0.0`:

  ```python
  if par_reac_cmp:
      for d in r_orig:
          if d in ki_cost:
              new_m = m.copy()
              new_m[d] = val
              for f in [e for e in r_orig if (e in ki_cost) and e != d]:
                  new_m[f] = 0.0        # the alternatives, explicitly un-added
              sd += [new_m]
  ```

  The `0.0` tags are not cosmetic — they carry the "this KI candidate existed and was deliberately
  left out" information that §9.5 and `strip_non_ki` depend on.

- **Coupled KI** (`networktools.py:1521‑1526`): coupled members only carry flux together, so a
  functional insertion must add **all** of them; expansion emits **one** design that knocks in every
  KI‑able member.

#### Case KI **not** introduced (`val == 0`)

A compressed id may appear in the design with value `0` — a KI candidate the solver decided *not* to
use (§9.5). Expansion propagates that "not added" verdict to every member of the group
(`networktools.py:1527‑1532`):

```python
elif val == 0:      # KI that was not introduced
    new_m = m.copy()
    for d in r_orig:
        if d in ki_cost:
            new_m[d] = val            # 0.0 on every member — none inserted
    sd += [new_m]
```

No branching here: "added nothing" has exactly one realisation.

#### A worked micro‑example

Take a two‑step map. COMPRESS #1 lumped coupled reactions `{R1, R2} → C` (`parallel: False`), and
COMPRESS #2 lumped parallel reactions `{C, R3} → P` (`parallel: True`), where after step 1 the space
is `{C, R3, …}`. All of `R1, R2, R3` are knockable. The solver returns the compressed KO `{P: -1}`.

Reverse order: apply step 2⁻¹ (parallel) first. `P` is a parallel lump of `{C, R3}`, `val < 0`, so
we emit **one** design knocking out all members: `{C: -1, R3: -1}`. Now apply step 1⁻¹ (coupled). `C`
is a coupled lump of `{R1, R2}`, `val < 0`, so we **branch** into one design per member, carrying `R3`
along: `{R1: -1, R3: -1}` and `{R2: -1, R3: -1}`. Final: two original‑model MCS. Both cut R3 (parallel
redundancy demanded it) and each cuts one of the coupled pair (either suffices). This is precisely the
set of minimal cut sets in the original network that the compressed `{P:-1}` stands for.

`estimate_expansion_size` (`networktools.py:1414`) computes the *count* of this fan‑out without doing
it, by walking the same reversed map and multiplying a `factor`: coupled‑KO and parallel‑KI multiply
by the number of eligible members (they branch), parallel‑KO and coupled‑KI multiply by 1 (they
don't). It returns an exact count for single‑step compression and an upper bound otherwise (because
across steps the same original could in principle be reached twice); it drives the lazy‑expansion
decision in §9.4.

### 9.3 Size‑1 MCS re‑injection

Recall from Ch 5 that FVA #3 (`compute_strain_designs.py:453‑491`) finds reactions that are
**essential for the SUPPRESS behaviour but not for any PROTECT behaviour** — i.e. reactions whose sole
knockout already makes the undesired flux infeasible while keeping the desired flux feasible. These
are size‑1 minimal cut sets. They are deliberately **removed from the knockable set before the MILP is
built** (`cmp_ko_cost.pop(r, None)` at `compute_strain_designs.py:491`) and stored separately:

```python
cmp_size1_mcs = [{r: -1} for r in size1_mcs_knockable]   # compute_strain_designs.py:481
```

The rationale (Ch 5) is twofold: they need no search, and — more importantly — leaving them in the
MILP would let the enumerator report every *superset* that contains a size‑1 MCS, which is
non‑minimal. Pulling them out keeps the MILP's minimal‑cut‑set guarantee clean. But they are still
real solutions, so decompression must add them back. Note this happens only for **classical MCS
problems** (exactly one SUPPRESS + only PROTECT modules — the `is_classical_mcs` gate at
`compute_strain_designs.py:472‑475`); bilevel problems (OptKnock etc.) never populate `cmp_size1_mcs`.

Re‑injection runs after the MILP designs have been expanded (`compute_strain_designs.py:697‑712`).
Each stored size‑1 MCS `{r:-1}` is itself a compressed design — `r` is a compressed reaction id — so
it goes through the *same* `expand_sd` + `filter_sd_maxcost` pipeline (one size‑1 compressed cut can
still fan out to several originals if `r` is a lumped reaction). It is then de‑duplicated against the
already‑expanded MILP designs before being appended:

```python
existing = [frozenset(s.items()) for s in sd]
for grp_idx, cmp_s in enumerate(cmp_size1_mcs):
    expanded = expand_sd([cmp_s], cmp_mapReac)
    expanded = filter_sd_maxcost(expanded, max_cost, uncmp_ko_cost, uncmp_ki_cost)
    expanded = postprocess_reg_sd(uncmp_reg_cost, expanded)
    for s in expanded:
        if frozenset(s.items()) not in existing:
            sd.append(s)
            group_map.append(next_grp + grp_idx)
            existing.append(frozenset(s.items()))
    compressed_sd.append(cmp_s)
```

The de‑dup guard exists because a size‑1 MCS *can* coincide with something the MILP also found through
a different route (e.g. via a coupled expansion), and we must not report it twice. Two further
details:

- **Group bookkeeping.** Each compressed design (MILP or size‑1) is a *group*; its expanded members
  share a `group_map` index (`compute_strain_designs.py:698`, `next_grp = len(compressed_sd)`). This
  is what powers `get_group` / `get_representative_sd` on the result object — the user can collapse the
  fan‑out back to "one decision per group" for display.
- **Status promotion.** If the MILP itself found nothing (INFEASIBLE) but size‑1 MCS exist, the status
  is lifted to OPTIMAL so the result is not reported as "no solution" (`compute_strain_designs.py:711`).
  The `dump_preprocessed` early‑return path (`compute_strain_designs.py:577‑592`) uses the same
  expand→filter→postprocess sequence to return size‑1 MCS even when the MILP solve is skipped entirely.

### 9.4 `filter_sd_maxcost`: why a post‑expansion cost re‑check is mandatory

`max_cost` bounds the total intervention cost of an acceptable design. The MILP already enforces it in
*compressed* space (the cost row over `z`, Ch 7). Why filter again after expansion? Because expansion
can change a design's effective cost, in both directions, so a compressed design that was within budget
can expand into original‑model designs that are not — and vice versa.

The reason is that `compress_ki_ko_cost` (`networktools.py:1387‑1410`) does not preserve cost
additively; it collapses a group's member costs to a single number using rules that are correct for
the *group* decision but lossy about the *members*:

- **coupled KO cost** = `min` of member KO costs (`networktools.py:1398`) — because cutting the group
  costs only as much as cutting its cheapest member (you only need one).
- **parallel KO cost** = `sum` of member KO costs (`networktools.py:1400`) — because you must cut them
  all.
- **coupled KI cost** = `sum`; **parallel KI cost** = `min` (`networktools.py:1407,1409`) — the duals.

Now cross this against §9.2's expansion. A **coupled KO** was compressed at cost `min`, but expansion
branches into one design *per member*, and each branch's true cost is *that member's* KO cost — which
for every member other than the minimum is **larger** than the compressed cost. The compressed design
passed the MILP budget at the cheap member's price; several of its expansions must be discarded because
their actual price exceeds `max_cost`. Concretely: a coupled pair with KO costs `{1, 5}` compresses to
cost `1`; under `max_cost = 3` the compressed design is feasible, but only the cost‑1 expansion
survives — the cost‑5 sibling is filtered out. Without the re‑check we would report an over‑budget
design.

`filter_sd_maxcost` recomputes the true cost in original space and keeps designs within a small
tolerance of the budget (`networktools.py:1548‑1554`):

```python
if max_cost:
    costs = [np.sum([(kocost[k] if k in kocost else kicost.get(k, 0)) if v != 0 else 0
                     for k, v in m.items()]) for m in sd]
    sd = [sd[i] for i in range(len(sd)) if costs[i] <= max_cost + 1e-8]
    [s.update({'**cost**': c}) for s, c in zip(sd, costs)]
    sd.sort(key=lambda x: x.pop('**cost**'))
```

Three things to read carefully here. First, the `if v != 0` clause: **only interventions actually made
count toward cost.** A KI candidate left un‑made carries value `0` and is free — this is exactly the
`(nan,nan)` / value‑0 encoding of §9.5, and it is why that encoding must survive expansion rather than
being stripped early. Second, it costs each *original* reaction independently with the *uncompressed*
cost dicts `uncmp_ko_cost` / `uncmp_ki_cost` (assembled in the orchestrator and, for gene problems,
merged with gene costs at `compute_strain_designs.py:421‑422`) — never the compressed dicts. Third,
the surviving designs are **sorted by ascending true cost** via a throwaway `'**cost**'` key, so the
cheapest realisations surface first; in the lazy path (below) this ordering is what makes
`expanded[0]` the "cheapest representative" of a group (`compute_strain_designs.py:737`).

#### The lazy‑expansion path (estimated count > 100 000)

For problems where the fan‑out is enormous — many deep coupled groups multiplying together —
materialising every expanded design would exhaust memory even though the search itself finished
(this is issue #47, noted in `SDSolutions.save`). `_decompress_solutions` guards against this
(`compute_strain_designs.py:638,654‑681`):

```python
LAZY_EXPANSION_THRESHOLD = 100_000
...
estimated  = estimate_expansion_size(cmp_sds, cmp_mapReac)
estimated += estimate_expansion_size(cmp_size1_mcs, cmp_mapReac)
if estimated > LAZY_EXPANSION_THRESHOLD:
    sd, group_map, compressed_sd = _build_lazy_representatives(...)
```

`_build_lazy_representatives` (`compute_strain_designs.py:721`) expands each compressed group *just
far enough* to keep **one** representative — the cheapest survivor of `expand_sd` + `filter_sd_maxcost`
— and records the machinery (the compressed designs, the map, the uncompressed cost dicts, the model)
in an `_expansion_meta` dict on the `SDSolutions` (`compute_strain_designs.py:667‑677`). The result
reports `get_num_sols()` as the *estimated total* while only a handful are materialised
(`get_num_materialized()`), and the user can force any group's full expansion on demand via
`expand_group` / `expand_all` (`strainDesignSolutions.py:446,520`), which run the identical
expand→filter→translate pipeline lazily. This is a pure space/time optimisation — the eager and lazy
paths compute the same designs; lazy just defers the combinatorial blow‑up until (if ever) the user
asks for it.

### 9.5 KI/KO value encoding and the strip semantics

Every solution dict — compressed or expanded, gene‑ or reaction‑level — encodes each intervention as a
numeric value with a fixed meaning:

| value        | meaning                                   | bounds it maps to (`itv_bounds`) |
|--------------|-------------------------------------------|----------------------------------|
| `-1.0`       | reaction/gene **knocked out**             | `(0.0, 0.0)` |
| `+1.0`       | knock‑in candidate **added**              | the reaction's original `.bounds` |
| `0.0`        | knock‑in candidate **offered but not added** | `(nan, nan)` |
| `True`       | regulatory intervention active            | derived from the constraint |
| `False`      | regulatory intervention not added         | — |
| *(absent)*   | reaction never a candidate                | — |

The value originates in `sd2dict` (`strainDesignMILP.py:200‑213`), which reads the solved binary
vector. A `z` variable is *inverted* iff it is a KI candidate — `z_inverted[i] = not isnan(ki_cost[i])`
(`strainDesignProblem.py:148`). For a non‑inverted (KO) variable, `z=1` means "apply the cut", written
as `-sol = -1`; for an inverted (KI) variable, `z=1` means "insert", written as `+sol = +1`. The
subtle line is the `0.0`:

```python
elif args and args[0] and (sol[0, i] == 0) and self.z_inverted[i]:
    output[reacID[orig_i]] = 0.0
```

Only when `show_no_ki` is on (it is, by default, set at `compute_strain_designs.py:526`) and the
variable is a KI candidate that came back at `z=0`, does the design record an explicit `0.0`.
`_decompress_solutions` reads these `0`‑tagged designs via `get_reaction_sd_mark_no_ki`
(`strainDesignSolutions.py:341`, `compute_strain_designs.py:650`).

**Why encode "not added" at all, instead of just omitting it?** Because a knock‑in candidate that the
solver *chose to leave out* is different information from a reaction that was *never a candidate*, and
several steps downstream need to tell them apart:

1. **Expansion correctness.** §9.2's `val == 0` branch must propagate "not added" to a lumped KI
   group's members, so that a compressed un‑made KI does not silently reappear as made after expansion.
2. **Cost correctness.** `filter_sd_maxcost` charges only `v != 0` interventions; an un‑made KI must be
   present‑but‑free, which requires it to be present with value `0`, not absent.
3. **Bounds semantics.** `_compute_costs_and_bounds` (`strainDesignSolutions.py:247‑255`) turns value
   `0` into bounds `(nan, nan)` — a deliberate "no bound change; this capability was considered and
   declined" marker, distinct from a KO's `(0,0)` and from an added KI's real bounds.

The flip side is that these `0`/`False` entries are noise in a human‑readable listing. `strip_non_ki`
(`strainDesignSolutions.py:768`) removes them:

```python
def strip_non_ki(sd):
    return {k: v for k, v in sd.items() if v not in (0.0, False)}
```

The public accessors `get_reaction_sd` and `get_gene_sd` (`strainDesignSolutions.py:304,330`) pass
every design through `strip_non_ki`, so the user sees only interventions that were *actually made*.
The un‑stripped forms remain available through `get_reaction_sd_mark_no_ki` /
`get_gene_sd_mark_no_ki` for callers that need the full picture. This "internal representation keeps
value‑0, display drops it" split is exactly the seam that issues #38 (superset reporting) and #43
(neutral gene KOs) turn on — the semantics are laid out here; Ch 10 owns the bugs. The one property to
carry into that chapter: **`strip_non_ki` uses membership in `(0.0, False)`, so a genuine `0` value —
whatever its origin — is indistinguishable at display time from a declined KI.**

### 9.6 Gene‑level vs reaction‑level translation (`_translate_genes_to_reactions`)

For gene‑based problems the MILP's `z` correspond to gene pseudoreactions (Ch 4), so after `expand_sd`
the design dicts are keyed by **gene** ids (and any surviving reaction/regulatory ids). The user
usually wants both views: the *gene* interventions (what to edit in the lab) and the *reaction*
phenotype (what those edits actually disable in the network). `SDSolutions.__init__` detects a gene
problem (presence of `GKOCOST`/`GKICOST` in the setup, `strainDesignSolutions.py:91`) and builds both
via `_translate_genes_to_reactions` (`strainDesignSolutions.py:134`).

The translation's job is: given a set of gene knockouts/knock‑ins, determine which *reactions* are
disabled. A reaction is governed by its **gene–protein–reaction (GPR) rule**, an arbitrary Boolean
expression over genes (e.g. `(b0001 and b0002) or b0003`). The previous implementation re‑parsed these
rules into disjunctive normal form and evaluated a hand‑rolled `gpr_eval`. The current code instead
reuses cobra's already‑parsed GPR abstract syntax tree and its evaluator
(`strainDesignSolutions.py:159‑161`):

```python
rxn_gpr = {r.id: r.gpr for g in model.genes for r in g.reactions}
```

`reaction.gpr` is a `cobra.core.gene.GPR` AST; `gpr.eval(knockouts)` walks that tree and returns
**True iff the reaction can still be active** when the genes in `knockouts` are removed. The
semantics that make this usable — and the subtlety to get right — is the **present‑genes‑default‑active
convention**: `eval` treats every gene *listed* in `knockouts` as off and **every gene not listed as
present/active**. So you drive it entirely through which genes you place in the knockout set.

The translation exploits this by evaluating each reaction's GPR under three different knockout sets, to
answer three distinct phenotype questions (`strainDesignSolutions.py:174‑195`):

```python
ko_off   = gene_ko | gene_no_ki    # KOs applied; un-made KIs off; made KIs on
all_off  = ko_off | gene_ki        # additionally undo the knock-ins
noki_off = set(gene_no_ki)         # only the un-made KIs are off; KOs undone
...
for r in candidate_reacs:
    gpr_r = rxn_gpr[r]
    if gpr_r.eval(ko_off):          # reaction still possible under the interventions
        if not gpr_r.eval(all_off): #   ... only because a knock-in kept it alive
            reac_ki.add(r)
    else:                           # reaction dead under the interventions
        if gpr_r.eval(noki_off):    #   ... the knock-out is what killed it
            reac_ko.add(r)
        else:                       #   ... dead regardless (e.g. an un-made knock-in)
            reac_no_ki.add(r)
```

Reading the three comparisons:

- **`eval(ko_off)`** — the actual post‑intervention world: knocked‑out genes off, un‑made KI genes off,
  made KI genes on (present by default). If the reaction survives this, it is not knocked out.
- **`eval(all_off)`** vs the above — additionally switch off the *made* knock‑ins. If the reaction was
  alive under `ko_off` but dies once the KIs are also removed, then it was alive *only because of a
  knock‑in* → it is a reaction‑level KI (`reac_ki`, value `+1`).
- If the reaction is dead under `ko_off`, ask **`eval(noki_off)`**: put back the knockouts (only the
  un‑made KIs stay off). If it *revives*, then the knockouts are what killed it → reaction‑level KO
  (`reac_ko`, value `-1`). If it stays dead even with the knockouts undone, it was doomed by something
  else (typically an un‑made KI it depended on) → reaction "not added" (`reac_no_ki`, value `0`).

Only reactions attached to an intervened gene are examined (`candidate_reacs` is built from the union
of the gene KO/KI/no‑KI sets, `strainDesignSolutions.py:182‑185`) — every other reaction is untouched
by definition, so evaluating it would waste time and could only return "unchanged."

The output preserves the §9.5 encoding on the reaction side: `-1.0` for `reac_ko`, `+1.0` for
`reac_ki`, `0.0` for `reac_no_ki`, plus `True`/`False` for regulatory interventions
(`strainDesignSolutions.py:196‑200`). The gene‑level view (`gene_sd`) is kept verbatim from the raw
solution dicts (`strainDesignSolutions.py:142`), including any gene‑name→gene‑id normalisation
(`strainDesignSolutions.py:147‑154`), so the two views stay linkable via `get_gene_reac_sd_assoc`
(the association is typically many gene sets → one reaction phenotype, since different gene KOs can
disable the same reactions).

Because this reaction phenotype is derived by *asking the GPR what dies*, a gene KO that turns out to
disable **no** reaction (its reaction is protected by an OR‑redundant gene, or the gene is an
isozyme partner) produces an *empty* reaction‑level effect while still showing up as a gene
intervention with a cost — the mechanism behind the "neutral gene KO" question of issue #43. The
translation is faithful to the GPR; whether such a design should have been enumerated at all is a
question for Ch 10.


## 10. Known issues, gotchas & failure modes

This chapter is a field guide to the ways `straindesign` can surprise you: two currently-open
correctness issues (#43, #38), one instructive closed one (#44), and a set of API/solver footguns that
have each cost real debugging time. For every item the goal is the same as the rest of this reference —
*what* goes wrong, the *mechanism* in the code that produces it, and *why* the design is shaped that
way. Line numbers were verified against the current source and, like the rest of this guide, may drift with later edits.

A recurring theme unifies most of this chapter: the package carries **two parallel identifier spaces for
genes (id vs. name)** and **two parallel encodings for interventions (a "real" nonzero value vs. a value-0
"not-added knock-in" marker)**. Almost every open gotcha is a place where those two representations are
not kept in lockstep. Keep that lens handy while reading.

### 10.1 Issue #43 (OPEN) — gene-level designs with no reaction-level effect

**Symptom (as reported).** A returned design lists one or more gene knockouts in `gene_sd`, but the
corresponding `reaction_sd` for that design contains no reaction the gene actually disables — the gene KO
is *neutral*. Critically, the reporter observed that the effect **appears when `gko_cost` is keyed by
gene names and disappears when it is keyed by gene ids**, together with a ">255 char reaction name"
trimming warning. That id-vs-name sensitivity is the fingerprint, and it points at two independent
mechanisms, either of which can leave a knockable-but-inert gene in the problem.

> **Status note.** On current `main` the investigation could not reproduce a genuine neutral gene KO from
> the reporter's setup (the setup as re-run only ever produced knock-in designs, so there were no gene KOs
> to be neutral). Both mechanisms below are therefore best understood as **latent, still-live code paths**
> that match the reported id/name signature and remain worth hardening — not as a bug with a known fixing
> commit. The issue stays open awaiting the reporter's exact failing `gene_sd`.

#### Mechanism 1 — `reduce_gpr` pops protected/essential genes by **id only**

`reduce_gpr` (`networktools.py:664`) is the pre-GPR-integration pass that removes genes which cannot
usefully be knocked out — genes that only touch essential reactions, or that are essential to an essential
reaction — so they never become MILP binary variables (see Ch 4 for the full GPR-reduction role). It builds
a `protected_genes` set (steps 2–3, lines 890–901), and then, in step 4:

```python
# line 904
[gkos.pop(pg.id) for pg in protected_genes if pg.id in gkos]
```

The removal key is `pg.id` **only**. If the caller passed `gko_cost` keyed by gene *name*
(`gkos = {'someGeneName': 1, ...}`), then `pg.id in gkos` is `False` for every protected gene whose id
differs from its name, so **nothing is popped** and the protected/essential gene stays in the knockable
cost dict.

The asymmetry is visible one line later. Step 5 (line 907) protects "all genes that are not knockable",
and *this* line is name-aware:

```python
# line 907 — note: id OR name
[protected_genes.add(g) for g in model.genes if (g.id not in gkos) and (g.name not in gkos)]
```

Likewise step 6 (lines 910–911) restores knock-in candidates by matching *either* `g.id in gkis` or
`g.name in gkis`. So `reduce_gpr` knows perfectly well that `gkos`/`gkis` may be name-keyed — every
membership *test* checks both id and name — but the one place it *mutates* `gkos`, the `.pop()` at line
904, uses `pg.id` alone. That is the fragility: a single un-mirrored key access in an otherwise
id-or-name-tolerant function.

The downstream effect compounds through the rest of `reduce_gpr`. `protected_genes_dict` is keyed by
`pg.id` (line 912) and fed to `simplify_gpr_ast`, which rewrites each reaction's GPR treating protected
genes as constant-`True` and **deletes them from the Boolean rule** (lines 914–932); then step 8 removes
protected genes from `model.genes` entirely (lines 934–937). So after `reduce_gpr` a name-keyed essential
gene can be in an inconsistent state: still present as a cost entry in `gkos` (because the pop missed it),
but scrubbed out of the GPRs and the gene list. When `extend_model_gpr` then builds gene pseudoreactions
from `model.genes` (Ch 4), that gene has no pseudoreaction to attach a `z` to — the intervention is
declared but wired to nothing, i.e. a neutral gene KO. **Fix direction:** pop by id *and* name at line 904,
mirroring the membership tests already used at 907/910.

#### Mechanism 2 — `_translate_genes_to_reactions` evaluates the GPR only over solution-present genes

Even with a clean knockable set, a gene KO can be genuinely chosen by the MILP and still map to *no*
reaction, because of how gene designs are translated back to reaction designs at decompression.
`_translate_genes_to_reactions` (`strainDesignSolutions.py:135`) takes a gene-level cut set and asks, for
each reaction the intervened genes touch, whether the reaction survives. It uses cobra's parsed Boolean GPR
and its `.eval()` (line 161, `rxn_gpr = {r.id: r.gpr ...}`; the AST evaluator replaced the old DNF-only
`gpr_eval`, per PR #51):

```python
# lines 187–195 (paraphrased structure)
if gpr_r.eval(ko_off):          # reaction still possible under the interventions
    if not gpr_r.eval(all_off): # ... only because of a knock-in → it's an effective KI
        reac_ki.add(r)
else:                           # reaction dead under the interventions
    if gpr_r.eval(noki_off):    # ... the KO is what killed it → real reaction KO
        reac_ko.add(r)
    else:
        reac_no_ki.add(r)       # dead regardless (un-made knock-in)
```

The mathematics of `cobra` `GPR.eval(knockouts)` is: **every gene named in `knockouts` is treated as
absent, every gene *not* named is treated as present.** The knockout set here is `ko_off = gene_ko |
gene_no_ki` — genes *in this solution*. Genes absent from the solution default to present/active.

That default is exactly what produces a neutral KO. Consider a reaction with GPR `a or b`, and a design
that knocks out only `b`. Then `ko_off = {b}`, and `gpr_r.eval({b})` evaluates `False or True = True`
(because `a`, not in the cut set, is treated as present) — the reaction is "still possible", so `b` is
**not** added to `reac_ko`. The gene `b` is faithfully recorded in `gene_sd` (which is just a copy of the
raw solution, line 142), but it contributes nothing to `reaction_sd`. An **OR-shadowed gene KO** — a gene
behind an `or` with a non-knocked partner — therefore always appears as a design that has a gene effect but
no reaction effect. This is not a bug in the translator per se; it is the correct GPR semantics. It becomes
a *reporting* surprise only because such a KO should arguably never have entered the design in the first
place (it is cost with no benefit), which loops back to Mechanism 1 and to the essentiality/knockability
pruning that is supposed to remove inert genes upstream.

#### The id-vs-name fragility, end to end

Beyond `reduce_gpr`, the id/name split threads through several stages and is the reason "names break, ids
work" is a plausible signature:

- **Pseudoreaction vs. pseudometabolite naming diverge.** In `extend_model_gpr`, when `use_names=True` the
  gene *pseudoreaction* is named from `gene.name`, while the gene *pseudometabolite* is always `g_{gene_id}`
  (see Ch 4). A downstream lookup that expects one convention but gets the other silently misses.
- **Name→id remap happens inside the translator, not before.** `_translate_genes_to_reactions` builds
  `gene_name_id_dict` and rewrites name keys to id keys on its *working copy* (lines 147–154), but
  `gene_sd` keeps the original (possibly name) keys. Two dicts, two key spaces, kept only loosely in sync.
- **Truncation is solver-dependent** (§10.5b): long lumped names are sha256-truncated for Gurobi/GLPK but
  not CPLEX, so a name that is a valid key on CPLEX can be a *different* (hashed) key on Gurobi — id-keyed
  runs sidestep this because ids are short.

The practical takeaway: whenever you touch gene-keyed logic, test with `gko_cost` keyed **both** ways and
assert the two runs produce identical designs. That equivalence is precisely the regression assertion the
investigation recommended and that no existing test yet enforces.

### 10.2 Issue #38 (OPEN) — superset/subset (non-minimal) solutions

**Symptom (as reported).** Pooling `sd.ANY` results across many random seeds yields designs that are
supersets of other designs in the pool — a 3-intervention cut set that strictly contains a valid
2-intervention one, i.e. apparently non-minimal MCS. The reporter saw "up to ~50%" of pooled solutions
implicated.

There are two genuinely distinct effects here, and separating them is the whole point of triaging #38.

#### The leading explanation: a reporting artifact from value-0 KI markers

`straindesign` encodes a **not-added knock-in** as the value `0.0` in the raw `reaction_sd`/`gene_sd`
dicts (and as `(nan, nan)` bounds in `itv_bounds`); a made KI is `+1`, a KO is `-1` (see Ch 9 for the full
value/`strip_non_ki` semantics). The user-facing accessors hide the value-0 entries:

```python
# strainDesignSolutions.py:768
def strip_non_ki(sd):
    return {k: v for k, v in sd.items() if v not in (0.0, False)}
```

`get_reaction_sd()`/`get_gene_sd()` apply `strip_non_ki` (lines 312, 316), so the *stripped* view shows
only real interventions. But the **raw `sols.reaction_sd` attribute is unstripped** — it still carries the
`(some_KI, 0.0)` markers for every knock-in candidate that a given design did *not* add. If a user dedups
or compares designs by `str(sols.reaction_sd)` (as the reporter's notebook did), two designs that make the
*same real interventions* but differ in *which value-0 KI markers happen to be present* stringify
differently, and one can string-contain the other. That manufactures spurious subset/superset pairs that
have **no** difference in actual interventions — a pure reporting artifact of comparing the unstripped
representation.

The corrective is mechanical: compare designs on the **stripped** view (`get_reaction_sd()`), i.e. on real
interventions only. Value-0 markers must never enter a minimality comparison. This alone accounts for the
bulk of the reported rate, and is consistent with the earlier compression-correctness fix (§10.3) having
already removed the *structural* half of the reporter's original ~50%.

#### The genuine residual: numerical-boundary non-minimality

A small residual (~2% on the re-run, each superset adding exactly one provably-redundant KO) is **real**
non-minimality, and it has a different root — numerical tolerance at a growth-coupling boundary, compounded
by the fact that `sd.ANY` gives no *cross-seed* global-minimality guarantee.

The mechanism: within a single seed, `compute()` already excludes supersets. After it accepts a design it
adds an exclusion (integer-cut) constraint `Σ z_active ≤ |active| − 1` (see Ch 8) that forbids that design
*and every superset of it* from reappearing in the same run. So true supersets cannot arise within one
seed. They only appear when **pooling independent seeds**: seed A finds an irreducible cut set `C`; seed B,
exploring a different branch-and-bound tree, finds `C ∪ {r}` and accepts it because, at that seed's
numerical tolerance, dropping `r` looked infeasible. Whether the extra intervention `r` is redundant is
decided at a SUPPRESS boundary that sits essentially at zero — the observed growth-coupling min is ≈ 4×10⁻⁷,
far below any biologically meaningful flux but far *above* the essentiality tolerance of `1e-10` (§10.6).
At that boundary the subspace cost-minimization in `compute()` cannot reliably tell the redundant
intervention from a needed one, so the non-minimal design is accepted as (locally) valid. Independent
validation confirms these are real: both the sub- and superset give identical max biomass and identical
suppress-boundary value, so the extra KO is provably inert.

**Distinguishing the two in practice.** Recompute each pooled design's interventions in the stripped view
and re-check pairwise containment. If a "superset" collapses to equality under stripping, it was the
value-0 artifact; if it survives (the larger design has a strictly larger *real*-intervention set), it is
the genuine numerical-boundary residual, and the fix is a **post-hoc cross-solution minimality/dedup pass**
over the pooled result plus better MILP conditioning at the coupling boundary — not a change to the
per-seed search, which is already superset-free.

### 10.3 Issue #44 (CLOSED 2026-06-23) — PROTECT violated under gene_kos, as a cautionary tale

**Symptom.** With `gene_kos` (and, in the reporter's case, mixed reaction KI/KO), some returned designs,
when re-applied to the *original* model, dropped biomass below the PROTECT threshold (e.g. `BIOMASS ≥
0.1`) — invalid designs presented as valid. Never seen with reaction-only interventions.

**Root cause — a compressed-model phantom flux.** The bug lived at the coupled-compression step. When
`compress_model_coupled` (Ch 3) merges a flux-coupled group of reactions into one master column, the
master's admissible flux must be the **intersection** of the members' bounds, translated through the
coupling ratios. The pre-fix code merged the group *without* intersecting bounds. A group whose members'
bounds actually intersect to `[0, 0]` — i.e. the coupling forces zero net flux — was nonetheless kept as a
flux-carrying master. That master could then carry a **phantom flux** that no combination of the original
reactions can realize. A strain design that relied on suppressing (or permitting) that phantom flux was
feasible in compressed space but meaningless on the original model: decompressed, PROTECT could fail
because the biomass route the compressed solver "used" cannot exist.

**The fix** (commit `d6f3d28`, post-v1.18; now the standing behavior documented in Ch 3): thread each
reaction's `bounds` through the coupled-merge work records, intersect the coupled-group bounds as
`(max lᵢ, min uᵢ)` after ratio/sign translation, and if the intersection is empty or forced to `(0,0)`,
declare the whole group *contradicting* and **remove master and slaves**, re-iterating the compression
fixpoint. Version-proofing confirmed the causal story: pre-fix commit `c851df2` produced 7/60 (~12%)
PROTECT-violating designs on the reporter's setup; current code produces 0 across 800+.

**Why this is a cautionary tale, not just history.** Two lessons carry forward:

1. **The class of bug — compressed-space validity ≠ original-model validity — is not gene-specific.** Any
   future change to compression (merging rules, new coupling detection, the exact-nullspace work) can
   reintroduce a compressed model that admits flux the original does not. The gene_kos path merely made it
   *visible*, because gene KOs exercise more of the coupled/GPR-extended structure.

2. **The blind spot: the existing tests were cardinality-only.** `test_05` (`mcs_gpr`) and `test_08`
   asserted the *number* of solutions, never that each returned design actually satisfies its PROTECT
   modules on the original model. A bug that returns the right *count* of *wrong* designs sails straight
   through. The guard that would have caught #44 — and must be added as a standing regression test — is:
   **re-evaluate every returned design against every PROTECT module on the ORIGINAL (uncompressed,
   un-extended) model**, by re-applying the gene/reaction interventions via cobra's own GPR knockout and
   solving, and assert feasibility. This is a different assertion class from cardinality, and it is the
   single test most likely to catch any regression of the whole "compressed phantom flux" family. Note the
   coupled-merge fix `d6f3d28` shipped without a *targeted* unit test for the bound-intersection /
   contradicting-group logic (its test additions were unrelated), so this coverage gap is still open at
   both the compression-unit level and the end-to-end validation level.

### 10.4 Gotcha (a) — `compute_strain_designs` mutates the caller's `reg_cost`/module dicts in place

`compute_strain_designs` is not free of side effects on its arguments. Two are worth internalizing.

**Modules are copied; cost dicts largely are not.** The `sd_modules` list is defensively copied
(`compute_strain_designs.py:190–191`, `[m.copy() for m in sd_modules]`), so the module objects the caller
passed are safe. The cost dicts are **not** copied — they are aliased:

```python
# lines 225–234
if key == KOCOST:  uncmp_ko_cost  = value
if key == KICOST:  uncmp_ki_cost  = value
if key == REGCOST: uncmp_reg_cost = value   # <-- the caller's dict, by reference
```

`uncmp_reg_cost` *is* the caller's `reg_cost` object. The orchestrator makes a `deepcopy` for its own
bookkeeping (`orig_reg_cost = deepcopy(uncmp_reg_cost)`, line 290), but it keeps operating on the aliased
original.

**`extend_model_regulatory` rewrites its dict's keys, and the orchestrator writes that back onto the
caller's object.** `extend_model_regulatory` (`networktools.py:1187`) turns each human-readable constraint
string (e.g. `'1 PDH + 1 PFL <= 5'`) into a generated pseudoreaction name (e.g. `p1_PDH_p1_PFK_le_5`) and
mutates its argument dict in place to use those generated names. The orchestrator then does, for the
immediate (reaction-based) regulatory constraints:

```python
# lines 329–330
uncmp_reg_cost.clear()
uncmp_reg_cost.update(_immediate_reg)
```

Because `uncmp_reg_cost` aliases the caller's `reg_cost`, this **empties the caller's dict and refills it
with the generated-name keys.** After one call, the caller's `reg_cost` no longer contains the original
constraint strings — it contains parsed pseudoreaction names. Reusing that same dict object in a second
`compute_strain_designs` call is corrupt input: the generated names are not parseable constraint strings,
so they misroute (deferred as if gene-regulatory) or raise. The same aliasing means the caller's
`ko_cost` is also *augmented* in place — the regulatory pseudoreactions are added to it via
`uncmp_ko_cost.update(...)` (lines 327, 414–415, 427).

**Consequence & workaround.** Never reuse a `reg_cost` (or `ko_cost`) dict across runs; pass a fresh
`dict(...)`/`deepcopy` each time, or reconstruct the setup per call. This is entirely internal to the API
surface — there is a code comment acknowledging the in-place mutation (lines 313–314), but the fix (copy
the caller's dict on entry, as is already done for modules) has not been applied.

### 10.5 Gotcha (b) — Gurobi/GLPK-only name truncation (sha256; CPLEX exempt)

`extend_model_gpr` can generate very long pseudo-metabolite/pseudoreaction names, especially after
compression lumps many reactions into one (Ch 3/Ch 4): the lumped id is a `*`-joined concatenation of the
member ids and gene tags, easily exceeding a few hundred characters. To stay within solver name-length
limits, names longer than `MAX_NAME_LEN = 230` are hashed:

```python
# networktools.py:1001,1012–1014
MAX_NAME_LEN = 230
def truncate(id):
    h = hashlib.sha256(id.encode()).hexdigest()[:20]
    return id[0:MAX_NAME_LEN - 21] + "_" + h
```

The crucial detail is the **guard**: every truncation site fires only for `solver in {GUROBI, GLPK}`
(lines 1026, 1043, 1059, 1072, 1088, 1103, 1144). **CPLEX is exempt.** The consequence is that the *same
input model* produces *different reaction/metabolite identifiers* depending on which solver is selected: a
long name is preserved verbatim under CPLEX but replaced by `<prefix>_<sha256[:20]>` under Gurobi/GLPK.
That changes reaction/metabolite identity in logs and in any downstream lookup keyed by name — which is why
it is #43-adjacent: a name-keyed gene/reaction lookup that works on CPLEX can miss on Gurobi because the
key was hashed out from under it, and the reporter of #43 saw exactly the truncation warning. It also means
solver-to-solver diffs of the extended model are not name-comparable without accounting for truncation.
Ids, being short, never hit `MAX_NAME_LEN`, so id-keyed workflows are immune — a second reason the #43
signature is "names break, ids work".

### 10.6 Gotcha (c) — solver numeric-status robustness (Gurobi 12 NUMERIC; CPLEX 5/6 unscaled-infeasibilities)

Genome-scale MCS MILPs are numerically nasty, and both solvers can return an "I finished but I'm not sure"
status that older `straindesign` treated as an unhandled case and crashed on. Both are now handled
gracefully.

**Why these MILPs hit the numeric statuses.** The SUPPRESS blocks are Farkas infeasibility certificates
(Ch 6) whose dual variables are unbounded by nature and are anchored only by a normalization row, and the
`z`-linking mixes big-M rows with indicator rows (Ch 7). Big-M constants derived from bounding LPs on an
ill-conditioned genome-scale network can span many orders of magnitude (the MILP-conditioning workstream
measured a ~9-order big-M range), giving the LP relaxation a badly scaled constraint matrix. Under such
scaling the simplex/barrier can reach a point it believes optimal or feasible but whose *unscaled*
residuals exceed tolerance — that is precisely CPLEX status 5/6 ("optimal/best with unscaled
infeasibilities") and Gurobi status 12 (`NUMERIC`). These are not logic bugs; they are the expected
failure surface of a large, poorly-conditioned MILP, and they show up specifically on the big reaction-KO
problems (e.g. `ko_cost` on ~1600 reactions) rather than on small models.

**How they are handled now.**

- *Gurobi* (`gurobi_interface.py:233–256`): on `gstatus.NUMERIC`, the solver **retries once with
  `NumericFocus = 3`**, restoring the previous value afterward. If the retry yields a solution it is
  accepted as `OPTIMAL`; if it yields an incumbent under time-limit-like status it is returned as
  `TIME_LIMIT_W_SOL`; otherwise it reports no solution (`TIME_LIMIT`) — never a crash.
- *CPLEX* (`cplex_interface.py:209–215`, and `slim_solve` at 250–255): status `5`/`6` is accepted with a
  warning and mapped to `TIME_LIMIT_W_SOL` (the solution is used but flagged), rather than raising.

The philosophy is *degrade, don't crash*: a numerically-imperfect incumbent is far more useful to an
enumeration loop than an exception that discards the whole run. Note two residual rough edges: the raise
for truly-unrecognized statuses still carries the original typo `"not yet handeld"` / `"Case not yet
handeld"` (e.g. `gurobi_interface.py:258`, `cplex_interface.py:219`), and the SCIP/GLPK interfaces were
flagged as likely to have analogous unhandled-status gaps that have not all been audited. Also relevant to
#38: an accepted "unscaled infeasibilities" solution *is* slightly imprecise, and that imprecision at the
growth-coupling boundary is part of why the genuine non-minimal residual exists (§10.2) — the robustness
fix trades a crash for occasionally accepting a marginally non-minimal design.

### 10.7 Other footguns

- **Hard-coded essentiality tolerance `1e-10`.** Both essential-reaction FVA passes classify a reaction as
  essential with `np.min(abs(limits)) > 1e-10 and np.prod(np.sign(limits)) > 0`
  (`compute_strain_designs.py:378` and `:465`) — the flux range must exclude zero by more than `1e-10`
  with a fixed sign. This absolute threshold has no relation to model scaling: a reaction that is
  biologically essential but whose minimal required flux is below `1e-10` will be missed (and remain
  wrongly knockable), while the ~`4e-7` growth-coupling boundary of §10.2 sits *above* the threshold and is
  thus treated as nonzero — feeding the numerical-boundary non-minimality. If you rescale a model or work
  with unusually small fluxes, this constant is a place to check first.

- **Size-1 MCS extraction is classical-MCS-only and knockable-scoped.** Reactions essential for a SUPPRESS
  module but not for any PROTECT module are pulled out as size-1 MCS and re-injected at decompression
  (`compute_strain_designs.py:475–494`), but only when `is_classical_mcs` holds and only for reactions
  still in `cmp_ko_cost`. This is correct, but it means the set of designs the MILP enumerates is *not* the
  full set — anything relying on inspecting the raw compressed solutions must account for the re-injected
  size-1 MCS (Ch 9), or it will under-count.

- **Licensing environment.** Gurobi's license model (node-locked vs. WLS/web) can differ across machines:
  a node-locked license may validate only on specific hosts, and WLS/web licenses carry overage risk under
  heavy parallelism (mitigated here by the shared module-level `gp.Env`, PR #52). A benchmark or CI job that
  assumes Gurobi is available everywhere, or that spins up many parallel Gurobi environments, can fail or
  silently fall back to another solver for reasons that have nothing to do with the algorithm. Because Gurobi
  is ≈4× faster than CPLEX on the canonical iML1515 run, "the algorithm got slow" and "it silently fell back
  to CPLEX because Gurobi wasn't licensed here" look identical from the outside — check which solver was
  actually selected before trusting a timing.

- **Comparing designs by the raw attribute vs. the accessor.** Reiterating the #38 lesson as a general
  rule: `sols.reaction_sd`/`sols.gene_sd` are *unstripped* (carry value-0 KI markers); `get_reaction_sd()`/
  `get_gene_sd()` are *stripped*. Any dedup, minimality check, or set-containment test must use the
  stripped accessor, never `str(sols.reaction_sd)`. This one habit prevents an entire class of phantom
  subset/superset reports.


## 11. Performance, benchmarking & roadmap

This chapter is forward-facing. The rest of *StrainDesign Internals* explains how the pipeline works;
this one is a map for the developer who wants to make it **faster** without making it **wrong**. It
does three things: (1) pins down where wall-time actually goes at genome scale, with numbers, so that
optimization effort lands on real bottlenecks and not folklore; (2) enumerates the performance levers,
each grounded in that profile and in the mathematics of the formulation (see Ch 6, Ch 7); and (3) lays
out the benchmarking discipline and the roadmap. Throughout, the governing constraint is
**completeness** — a Minimal Cut Set (MCS) computation must never silently drop a valid design (Ch 8,
Ch 9), so every speedup is a claim that has to be gated against a known answer.

Two numbers to keep in your head, both measured on the canonical iML1515 gene-MCS run
(SUPPRESS `BIOMASS_Ec_iML1515_core_75p37M ≥ 0.001`, POPULATE, `max_cost=3`, `gene_kos=True`):
**CPLEX 1241 s, Gurobi 280 s**, both returning the identical 393 MCS. That ≈4.4× solver gap, and the
internal split of those seconds, is the spine of everything below.

### 11.1 The verified bottleneck profile

All timings here were measured against the real solver APIs (package v1.18, CPLEX 22.1.2 / Gurobi 13.0.1)
on the canonical iML1515 393-MCS problem. State them as given;
re-measure before trusting anything not on this list.

#### 11.1.1 Where the seconds go (canonical iML1515, CPLEX)

| Phase | What it is | Time |
|---|---|---|
| Prepare/parse | modules, solver, costs, seed | ~7 s |
| COMPRESS #1 | 2712 → 1237 reactions (parallel + coupled, 5 iters) | 3.4 s |
| GPR preprocessing | 1516 genes → `extend_model_gpr` (model → 3448 reac) | ~1 s |
| COMPRESS #2 | after GPR extension, 3448 → 2152 reactions | 4.3 s |
| **`bound_blocked_or_irrevers_fva`** | whole-model bound-classifying FVA (the ~4300-LP sweep) | **117.4 s** |
| FVA essential + size-1 MCS | 88 size-1 MCS extracted via SUPPRESS-scoped FVA | 3.5 s |
| MILP build | Farkas dual assembly 2.7 s + `link_z` (536 indicators) 0.9 s | **3.7 s** |
| **Solve (POPULATE)** | pool search → 84 compressed solutions | **1101 s** |
| Decompress | `expand_sd` + maxcost filter + phenotype → 393 | ~1 s |
| **Total** | | **1241 s** |

Three facts fall straight out of this table, and each one redirects a class of optimization effort:

1. **The two costs that matter at genome scale are the preprocessing FVA (~117 s) and the solve/pool
   search (~1101 s).** Together they are 98% of wall-time. Everything else — parse, both compressions,
   GPR extension, size-1 MCS extraction, decompression — is single-digit seconds. Optimize the two big
   phases; leave the rest alone unless it becomes structurally coupled to them.

2. **MILP *construction* is now cheap (~4 s).** This was not always true: before PR #55 the build was
   ~70 s, dominated by a scalar-loop `prevent_boundary_knockouts` (~51 s) and a non-deduplicated
   `link_z` (~16 s). Vectorizing `prevent_boundary_knockouts` and hashing the `link_z` bounding-LP
   dedup collapsed it to ~7 s, byte-identical output, and the exact-nullspace/build refinements since
   have trimmed it further. **The lesson for the next optimizer:** the build phase has already been
   wrung out; do not spend effort shaving milliseconds off matrix assembly. The money is in FVA and the
   solve.

3. **The 117 s FVA is a genuinely preprocessing cost, not a solve cost** — it is the whole-model
   `bound_blocked_or_irrevers_fva` call (see Ch 5, §3.3), roughly `2n` single-reaction LPs with no
   `reaction_list` scoping and no extra constraints. That structure is what makes it CPLEX's per-LP
   overhead multiplied by ~4300, and it is why it is separately attackable from the pool search.

#### 11.1.2 The CPLEX-vs-Gurobi ≈4.4× gap and its *true* causes

The same 393-MCS problem runs in **CPLEX 1241 s vs Gurobi 279.8 s**. Decomposing both runs by phase
localizes the entire gap to exactly two places:

- **Preprocessing FVA: ~117 s on CPLEX.** This is CPLEX's per-LP construction/solve overhead paid ~4300
  times over. Gurobi's per-LP overhead on the same sweep is materially lower. This is a *fixed tax per
  LP*, so the fix is architectural (fewer LPs, parallelism, cheaper backend for the sweep — §11.2.5),
  not a solver-parameter tweak.
- **Pool search (POPULATE): ~1101 s on CPLEX vs a small fraction of that on Gurobi.** CPLEX's
  solution-pool enumeration runs ~4–7× slower than Gurobi's on this MILP. This is the dominant term and
  the dominant contribution to the 4.4×.

Everything else — the branch-and-bound on the incumbent-finding solves, the MILP build — is at rough
parity between the two solvers. So the correct one-sentence statement of the gap is: **the CPLEX
disadvantage is per-LP preprocessing overhead plus pool-search speed, and nothing else.**

Three things the gap is emphatically **NOT**, each of which cost prior investigation time and is now
closed:

- **NOT the indicator constraints.** Under the default `M = inf`, SUPPRESS's Farkas-dual rows become
  indicator constraints and PROTECT's finite-flux primal rows become big-M rows — but this is emergent
  from the bound structure via the `self.M`/bounding-LP fork in `link_z` (`strainDesignProblem.py`, the
  finite-vs-`inf` `max_Ax` test around line ~853), **not** a per-module-type switch (Ch 7, §3.2). Both
  solvers get the *same* formulation with the same indicators, and both handle those indicators fine.
  The indicators are not the gap.

- **NOT the pool parameters.** CPLEX sets `mip.pool.absgap=0`, `mip.pool.relgap=0`,
  `mip.pool.intensity=4` at solver construction (`cplex_interface.py:161-163`), and Gurobi sets
  `PoolGap=PoolGapAbs=1e-9` (`gurobi_interface.py:162-163`). These have been dated by `git blame` to
  2022 (CPLEX line `b87d49c1`, 2022-04-18 — not a recent regression) and, more importantly, **verified
  inert for single `solve()`**: after a feasibility solve at `intensity=4`, `pool.get_num()==0`,
  identical to `intensity=0`. CPLEX does not populate the pool during a plain `optimize()`; the pool
  params only bite inside `populate()` (POPULATE). They are architecturally misplaced (they belong
  inside `populate()`), but they are **not a performance bug for ANY/BEST**. Do not re-derive this — it
  was tested three ways.

- **NOT a big-M conditioning catastrophe.** A discredited earlier reading claimed "CPLEX 400 s /
  indicators catastrophic / use big-M." That number came from calling `backend.solve()` on the MILP's
  *construction* objective — a global optimization that no production path ever runs — on a self-made
  iML1515/1,4-BDO/`max_cost=40` dump with 2228 indicators and a loose cardinality bound. It is not
  representative of any real run and has been thrown out. **The dead-end to remember:** there is no
  9.4-order big-M range in the built MILP to fix. As the MILP roadmap verified (§0–§1), the shipped
  formulation carries only a few dozen big-M rows, all at the loose default ±1000 (e.g. iMLcore: 34
  big-M / 388 indicators), because the wide-flux-span reactions all relax to ±inf bounds and become
  *indicators*, not tiny big-M's. Equilibration of a big-M range that does not exist is moot.

The practical upshot: **do not chase the solver gap through solver knobs or the indicator/big-M
dichotomy.** The gap lives in the *number of LPs* in preprocessing and in *pool-search throughput*.
Fix those structurally.

### 11.2 The performance levers

The levers below are grouped and ordered to match the profile: compression (cuts the problem before it
is built), formulation/conditioning (shapes the MILP the solver sees), skipping hopeless work, the
Farkas-dual pre-bounding problem, the preprocessing FVA, and the enumeration strategy. This list
reflects informed intuition, not a ranked plan — argue with it, and measure before committing effort.
Phil's standing prior: the biggest *suspected* structural win is a better MILP formulation/conditioning
(group 2), solver parameters (group 4-adjacent) are a fragile secondary bet, and the "good compression
≈ MCS2" insight (group 1) is a **hypothesis to verify**, not a foundation to build on.

#### 11.2.1 Compression depth = rank / z-count reduction (the structural lever)

The binary variable count `num_z = numr` — one `z` per compressed reaction (`strainDesignProblem.py`
`__init__`, `num_z` set around line ~144) — is the dominant complexity driver of the MILP. Branch and
bound over `z` is combinatorial; halving `numr` is worth far more than any constant-factor solver tune.
Network compression (Ch 3) is the mechanism that reduces `numr` losslessly and exactly, and it is
therefore the single largest structural lever available.

The reasoning is that compression is a **rank/dimension reduction of the flux system done for free**:
parallel merge, coupled/flux-coupled merge, conservation-relation (row) removal, and blocked/zero-flux
removal each shrink `S` while preserving the exact set of steady-state flux distributions (the exact
integer/rational nullspace guarantees this — never float; see Ch 3 and the hard constraint). Every
reaction removed is a `z` never created, an LP row never linked, a branch never taken. On the canonical
run, COMPRESS #1 takes 2712 → 1237 and COMPRESS #2 takes 3448 → 2152 (after GPR extension inflates the
count); pushing either merge closer to a true fixpoint directly removes binaries.

Concrete sub-levers, in decreasing certainty:

- **Scaled-parallel merging** (shipped, PR #54): merge reactions whose stoichiometry is identical *up
  to any rational scalar* and that share reversibility/bound topology. This is strictly more merging
  than exact-equality parallel detection, and it is exact (the merge factor is a flux-split share).
- **Push the coupled+parallel alternation to a genuine fixpoint.** The compression loop alternates
  parallel-merge → conservation-removal → coupled-merge until a step stops reducing (Ch 3). Confirming
  we reach *maximal* exact reduction — that no additional pass would remove one more reaction — is the
  cleanest way to guarantee the `z`-count is minimal for a given model.
- **Order interactions** between blocked/dead-end removal, conservation-relation removal, and coupling:
  removing dead ends first can expose new couplings and vice versa; the order the fixpoint visits them
  affects how quickly it converges and, at the margin, what it finds.

The deeper claim attached to this lever is the **"good compression ≈ MCS2" hypothesis** (Phil).
MCS2 (doi:10.1093/bioinformatics/btz393) computes minimal coordinated supports over the nullspace;
its structural benefit is essentially working in a full-rank coordinate system. The hypothesis is that
*a sufficiently good compression already reduces the MILP to (near) full rank, producing a problem
almost identical to MCS2's* — so maximizing exact compression captures most of the MCS2 advantage
without importing MCS2's method. Two pieces of evidence bear on it: a standalone MCS2-style nullspace
approach was tried and gave **no speedup** (solid compression already captured the structural benefit),
and the exact-nullspace PR #60 lifted compression ~1.6× and made yeast-GEM compress at all. But this
remains a **hypothesis, not a fact**, and the way to settle it is stated in §11.3: complete-enumerate
(ALL, not BEST/ANY) reaction MCS up to ~6 KOs on a couple of genome-scale models and compare
head-to-head with MCS2. If the hypothesis holds, compression depth is the whole game for competitiveness
and the MILP-formulation work is secondary; if it fails, the reverse.

#### 11.2.2 MILP formulation & conditioning

Compression decides *how many* binaries; formulation decides *how hard the solver's job is per binary*.
The relevant machinery is `link_z` (Ch 7), which wires each binary `z` to the continuous rows either as
a native indicator constraint or as a big-M row, choosing per-row on the sign of a bounding-LP maximum
`max_Ax` (finite ⇒ big-M with that constant; `inf` ⇒ indicator). The levers:

- **Prefer native indicators; use big-M only where forced.** Gurobi, CPLEX, and SCIP all support native
  indicator constraints; only GLPK forces everything to big-M (its `self.M` is a finite cobra bound).
  A loose big-M gives a weak LP relaxation, and a weak relaxation hurts CPLEX more than Gurobi. The
  shipped formulation already leans indicator-heavy by construction (536 indicators on the canonical
  run), which is why the indicator/big-M split was ruled *out* as the cause of the solver gap
  (§11.1.2). But the audit is still worth doing on new model classes: verify we never hand CPLEX a
  structurally weaker formulation than Gurobi on the same problem.

- **Tighten every big-M to its smallest valid bound.** `link_z` already computes a per-row
  `max_Ax` = max of the constraint over the LP-relaxed feasible region, which is the tightest *valid*
  M given the bounds (an LP-tight, not MILP-tight, heuristic — the true MILP-tight max-min is as hard as
  SUPPRESS itself). The gap here: the few dozen *functional* big-M rows that survive are written at the
  loose default ±1000, not at their tighter FVA maxima (MILP roadmap §0: iMLcore = 34 big-M all ≈1000).
  Tightening those 34 from 1000 to their FVA-computed maxima strengthens the relaxation. The honest
  caveat is that 34 ≪ 388 indicators, so the impact is likely small and *must be measured across models*
  before it earns effort.

- **Cut the `z` count at the formulation boundary, not just in compression.** Beyond compression
  (§11.2.1), drop structurally-non-knockable reactions and essential reactions *before* they become
  `z` variables: FVA #1 removes reactions essential to a desired/PROTECT module from the knockable set,
  and FVA #3 pulls size-1 MCS out entirely (re-injected at decompression so the MILP never enumerates
  their supersets; Ch 5, Ch 9). Every reaction kept out of `cmp_ko_cost` is one fewer binary.

- **The trace-cofactor ill-conditioning and the 9.4-order big-M range — a note, now largely closed.**
  The MILP roadmap initially diagnosed a chain: stoichiometry spanning 7.6 orders of magnitude →
  FVA flux spans of 9.4 orders → tiny big-M's from trace-cofactor pathways (biotin flux ~1e-6, etc.).
  Following the actual pipeline showed **that chain does not exist in the built MILP**: the tiny-flux
  reactions relax to ±inf bounds and become *indicators*, never tiny big-M's, so there is no 9.4-order
  big-M range to condition (§11.1.2). Exact row+col equilibration of the stoichiometry (7.6 → ~3.8–4.0
  orders, exact via `D·N·v=0 ⟺ N·v=0`) remains a *possible* lever on the primal/dual matrix
  conditioning that the SUPPRESS-indicator path sees — but whether stoich conditioning of 4.0 vs 7.6
  orders changes the indicator solve at all is **unproven and is the correct experiment to run**, not an
  assumption. Combined stoich + big-M equilibration is a genuine conflict (`s_j·M_j` spans ~9.7 orders;
  one column scaling can fix stoich·α *or* big-M/α but not both when `s·M ≉ 1`), so it is off the table
  for the big-M range and only live for the (separate, unproven) stoich angle.

#### 11.2.3 Skip hopeless big-M / dual work

The cheapest work is work not done. When a knockable constraint's reaction is provably always-zero, or
its bound provably never binds, the entire big-M/indicator machinery for that row can be skipped rather
than computed and added. Two concrete pieces:

- **The `link_z` sparse short-circuit** (on `hpc_benchmark`): before running the bounding LP, inspect
  the row's nonzero count. `nnz==0` ⇒ `M=0` directly; `nnz==1` (a plain reaction KO) ⇒ M is just
  `coeff·bound` (∞ if that bound is ∞) — no LP needed, because a single-variable row's maximum over a
  box is read straight off the bound. Only `nnz≥2` rows (module/dual constraints) go to an actual LP
  (parallelized via `SDPool` above ~1000 rows). This is what makes the build cheap; promote it and keep
  it. The corollary lever, from MILP roadmap §0, is that `max_Ax` for single-var KOs is *redundant* — it
  reproduces the bound `bound_blocked_or_irrevers_fva` just set — so the LP pool can be restricted to
  multi-variable rows with no behavior change and a measurable preprocessing saving.

- **Substituting out or removing binaries after a target is found** is the uncertain end of this lever.
  Once a synthetic-lethal single (`DBTS`) or a specific double (`AOXSr2, DBTS`) is identified, it is
  unclear whether anything beyond removing the binary variable helps — branch-and-bound may already
  prune those paths. This is problem-structure-dependent and may require a MILP rebuild; treat wins here
  as speculative until measured.

#### 11.2.4 The Farkas-dual pre-bounding problem (the known hard lever)

This is the deepest formulation lever and the one with the most headroom, because it is the one the
current architecture *cannot* address with its existing tools.

The asymmetry: PROTECT modules embed the raw primal (the desired flux state must stay feasible), so
their reaction variables carry **finite flux bounds** that FVA can pre-bound and tighten. SUPPRESS
modules instead build a **Farkas infeasibility certificate**: `farkas_dualize` (`strainDesignProblem.py`
~1141) dualizes the primal with a zero objective and appends the normalization row `c_d·y ≤ −1`
(verified: `A_ineq_f = vstack(A_ineq_d, c_d)`, `b_ineq_f = b_ineq_d + [-1]`), which encodes "the
undesired flux state is infeasible after the knockouts" (Ch 6). The knockouts act on **dual variables**,
and those duals are **unbounded by nature** — one-sided `[0,∞)` for inequality duals or free for
equality duals — pinned only by the `≤ −1` anchor. There is no finite flux bound to read off, so
**FVA pre-bounding does not help the SUPPRESS rows at all.** This is *why* they fall to `inf` `max_Ax`
and become indicators (§11.1.2): not a design choice, a mathematical fact about Farkas rays.

Because SUPPRESS is the "cannot" half of every classical MCS problem, this is not a corner case — it is
the core. Three redesign options, in increasing ambition, each a *different exact encoding of the same
problem* (Ch 6 owns the dual math; these are pointers for the optimizer):

1. **Split the compressed network into forward/reverse before Farkas construction.** Constructing the
   certificate over a sign-definite (fwd/rev-split) network changes which dual components are free vs
   one-sided and can expose bounds that the un-split formulation hides. This is the lowest-risk of the
   three because it operates on the network before dualization.
2. **Slack variables tied to global binaries.** Replace the pure dual-ray encoding with slacks that are
   directly linked to the intervention binaries, so the "infeasibility after KO" condition is carried by
   bounded slacks rather than unbounded duals — giving FVA something finite to bound.
3. **Branch on the indicator constraints directly** rather than routing through the dual ray at all.

A related, concrete M-dimensioning idea for the Farkas certificate (MILP roadmap R2, untested): run FVA
at *all combinatorial cases of the few inhomogeneous bounds* (PROTECT biomass, glucose uptake, ATPM),
take the smallest nonzero flux a reaction can carry, and use `1/v_min` as that reaction's M in the
certificate (or 1000 if every case gives 0). This would give tight-but-valid Farkas M's for the trace
reactions without the exponential max-min — but it must be prototyped and checked for **completeness**
(no missed solutions) before it is trusted.

#### 11.2.5 The whole-model preprocessing FVA

`bound_blocked_or_irrevers_fva` (Ch 5, `networktools.py:1589`) is ~117 s and the entire preprocessing
bottleneck. It runs one whole-model FVA — passing *no* `reaction_list` and *no* extra constraints, so it
does the full `2n` objectives — and then classifies each reaction's bounds: redundant bound (FVA never
reaches it) → ±inf; `min≥0` → irreversible-forward (`lb=0`); `max≤0` → blocked/reverse (`ub=0`); and it
mutates `_lower_bound`/`_upper_bound` in place. It *needs* every bound to do the classification, so it
genuinely cannot be scoped to knockable reactions only. The levers are therefore about the *cost of the
sweep*, not its scope:

- **Parallelize the Phase-2 residual.** `speedy_fva` (Ch 5, `speedy_fva.py`) already avoids most of the
  `2n` LPs via a `v=0`-feasibility pass, a `min Σ|x|` scan, and iterative warm-started push-to-bounds,
  falling to individual LPs only for the residual reactions Phase-1 did not resolve. The likely win: on
  this whole-model call Phase-1 resolves so much that the Phase-2 residual drops *below* the ~1000-LP
  parallelization threshold and runs **serially** — so it pays CPLEX's per-LP tax one reaction at a time.
  Forcing the residual to parallelize (or lowering the threshold for this call) directly attacks the
  117 s.
- **A cheaper backend for the LP sweep.** The 117 s is dominated by CPLEX's ~2 s/LP construction
  overhead × ~4300 LPs. Nothing about a bound-classification FVA needs CPLEX specifically; running the
  sweep on a lighter LP backend (or `slim_fba`/`slim_solve`-style reduced solves) sidesteps the per-LP
  tax that is the whole cost.
- **Amortize across seeds.** `dump_preprocessed` + `compute_strain_designs_from_preprocessed` (shipped)
  lets one preprocessing run feed many seeded solves — essential for the multi-seed benchmarking below,
  since it turns a per-seed 117 s tax into a one-time cost.
- **FVA relocation** (on `hpc_benchmark`): moving/reordering the FVA relative to COMPRESS #2 and snapshotting
  `pre_fva_bounds` is prototyped; its real speedup must be measured rigorously head-to-head, not assumed.

#### 11.2.6 Enumeration & pooling strategy

The ~1101 s pool search is the largest single term, and it is the one place where the enumeration
*strategy* (as opposed to the formulation) is the lever. The solve loop rebuilds and re-solves,
excluding each found design with `add_exclusion_constraints` (integer cuts that exclude a design *and
its supersets*; Ch 8). Levers:

- **Integer cuts as lazy constraints.** Adding the exclusion constraints as solver-native *lazy*
  constraints, and reusing the branch-and-bound tree / basis across iterations, avoids rebuilding the
  model for every solution found. This is the natural fit for the iterative enumerate loop and is where
  a warm-started, incremental architecture would pay off most against the 1101 s.
- **Warm starts.** Reuse the previous solve's basis and incumbent when adding the next cut, rather than
  cold-starting each populate iteration.
- **A cross-solution minimality/dedup pass** on pooled `sd.ANY` results — removes the residual ~2%
  non-minimal supersets (issue #38) that arise from value-0 KI markers and from pooling many seeds, and
  is cheap relative to the search itself.

Solver-parameter tuning of the pool (CPLEX emphasis/numeric-emphasis, indicator-API usage) is a
**fragile bet** and belongs strictly *after* the formulation is confirmed identical across solvers:
leaning on parameter defaults makes the package vulnerable to solver-version updates that change those
defaults or add better internal routes. Confirm the formulation first, tune params only to *confirm* a
hypothesis, never to carry one.

### 11.3 Benchmarking discipline

Speed claims about a branch-and-bound MILP are worthless without discipline, because B&B is chaotic in
ways that a naive timing hides. Four rules.

**Multi-seed distributions — single-seed timing is meaningless.** The seed is fully plumbed
(`compute_strain_designs(seed=)` → `kwargs_milp[SEED]` → the backend constructor → CPLEX
`parameters.randomseed` / Gurobi `Params.Seed`). The B&B tree *shape* is seed-dependent: the order in
which the solver branches, and therefore how quickly it finds and proves solutions, changes with the
seed. A single-seed run is one sample from a wide distribution, and comparing two configurations on one
seed each can invert the true ordering. **Every speed comparison — ANY, BEST, and POPULATE alike — needs
≥5 seeds** and is reported as a distribution (median + spread), never a single point. This is why the
`dump_preprocessed` amortization (§11.2.5) matters operationally: it makes a 5-seed sweep affordable by
paying the 117 s preprocessing once.

**Known-answer gates — completeness is the gate, not a nicety.** Two canonical counts are the regression
oracle: **e_coli_core = 455 MCS** (CPLEX ~1.2 s) and **iML1515 = 393 gene-MCS** (the canonical run
above). No MIP optimality gap is ever set, so both solvers run at their default 1e-4 relative gap, which
for integer intervention-cost objectives is effectively exact. Any change to bounds, big-M values,
Farkas M-dimensioning, compression depth, or enumeration strategy **must reproduce these counts
exactly**. A speedup that returns 392 MCS is not a speedup; it is a correctness regression. The
non-negotiable phrasing from the MILP roadmap: any M/bound change must not drop a valid MCS, and every
experiment must re-verify the known-answer counts. The test class that enforces this — re-evaluating
*every* returned design against all PROTECT modules on the original model — is precisely the gate that
would catch a completeness regression (and would have caught the historical #44).

**Head-to-head against the real competitors, on both solvers.** The target is competitiveness with
**MCS2** (doi:10.1093/bioinformatics/btz393, code at `github.com/RezaMash/MCS`) and **gMCSpy**
(doi:10.1093/bioinformatics/btae318, code + benchmark at `github.com/PlanesLab/gMCSpy`), measured on
**both Gurobi and CPLEX** — because the whole point of the Direction-A work is that Gurobi is currently
much faster than CPLEX on the same straindesign problem, and a fair comparison must not hide behind one
solver. The benchmark set is iML1515 / Yeast-GEM 8.7 / Human-GEM 1.16. The harness lives locally on the
`hpc_benchmark` branch (gitignored), with `benchmarks/tools/MCS2/` reconstructed and its MEX
Octave-recompiled. A caution learned the hard way: prior bound-config experiments (the P-A/B/C, F-A–E
configs in `bench_bound_configs.py`) produced almost no actual MILP change and *insignificant* perf
differences — the amount of real headroom is unknown, so **measure before committing effort**, and do
not mine old JSON in place of a fresh, correctly-distinct experiment.

**Never drop a valid MCS.** Restated because it is the one rule that overrides all others: completeness
is not traded for speed. The complete-enumeration (ALL, not BEST/ANY) runs up to ~6 KOs that would
settle the "good compression ≈ MCS2" hypothesis (§11.2.1) are themselves the strongest completeness
test, because they force the machinery to produce *every* MCS in a size band and expose any silent drop.

### 11.4 Roadmap & directions

**Direction A — compute performance & MCS2/gMCSpy competitiveness (the live thrust).** This is the
active work. Shipped so far: MILP build cut ~70 s → ~7 s (PR #55) and the CPLEX-populate configuration
win. The measured gap stands at CPLEX 1241 s vs Gurobi 280 s ≈ 4.4× on the canonical
iML1515 393, split into preprocessing FVA ~117 s and pool search ~1101 s — so the two real levers are
the whole-model bound FVA (§11.2.5) and the pool-enumeration strategy (§11.2.6), **not** indicators and
**not** the pool params (both verified inert). The near-term milestones are: (1) MCS2/gMCSpy
head-to-heads on iML1515 / Yeast-GEM 8.7 / Human-GEM 1.16; (2) push compression depth to a true fixpoint
(§11.2.1) and settle the "good compression ≈ MCS2" hypothesis by complete enumeration; (3) redesign the
Farkas-dual pre-bounding (§11.2.4); (4) clean up the solver-agnostic `internal_other` remnant. Hexaly is
an optional extra backend target.

**The exact-nullspace compression thread.** The exactness constraint is upstream and settled: the
nullspace/compression stays integer/rational (never float — small numeric deviations introduce
irreparable compression errors), and PR #60 folded the exact integer/rational sparse nullspace into
`compression.py` as public `straindesign.nullspace`/`sparse_nullspace`, delivering ~1.6× compression on
iML1515/Human-GEM and making **yeast-GEM compress at all** (it previously crashed on scipy's int64
ceiling; the fix routes >64-bit coefficients through a dict-of-Fractions mode + `ExactCOO`). This is the
shared building block under the compression-depth lever: better exact compression is more `z`-count
reduction, which §11.2.1 argues is the largest structural win.

**Adjacent efforts (pointers only).** Two prototypes share the exact-nullspace core but are not part of
the straindesign performance work: **SENUS** (`VonAlphaBisZulu/SENUS`) is the standalone exact
integer/rational sparse nullspace lifted out of `compression.py` — a longer-shot Direction-B play whose
next speedup is a Bareiss fraction-free elimination to bound coefficient growth; and **Kimonu**
(`VonAlphaBisZulu/Kimonu.py`) is an *independent* kinetic-module (COCOA-style) analyzer that reuses the
same nullspace core but is not a straindesign component. Both are mentioned here only so a reader tracing
the nullspace code across repos knows where it went; neither is on the straindesign performance critical
path.


## 12. Model surgery & constraint parsing

Every module in this codebase eventually reduces the user's intent to rows of a matrix: a stoichiometry
`S`, a stack of inequality rows `A_ineq·x ≤ b_ineq`, a stack of equality rows `A_eq·x = b_eq`, and cost
dicts that say which columns are knockable. Between the user's Python call and that matrix sits a thin
**utility layer** — a handful of functions in `networktools.py`, one edit routine in `compression.py`,
and the whole of `parse_constr.py` — that *edits the model in place* and *turns human-written strings
into sparse rows*. This chapter documents that layer.

None of it is the mathematical heart of strain design (that is dualization, Ch 6, and MILP assembly,
Ch 7). It is the **glue**: the code that makes the model clean enough to compress (Ch 3), that encodes a
regulatory bound as extra stoichiometry, that translates a gene knockout into a flux constraint, that
keeps modules and cost vectors consistent with the ever-shifting compressed reaction index, and that
lets a user write `"2 r1 - r2 <= 5"` instead of hand-assembling a `scipy.sparse` row. Glue is where
off-by-one bugs, index drift, and in-place-mutation footguns live, so it is worth the same care as the
core.

The functions appear in `compute_strain_designs` (the orchestrator, `compute_strain_designs.py:56`) in a
specific order, and the ordering is load-bearing. The map for this chapter, keyed to the preprocessing
block (`compute_strain_designs.py:305–495`):

| step | function | file:line | when |
|------|----------|-----------|------|
| clean the model | `remove_ext_mets` | `compression.py:1707` | first, before any compression |
| parse a constraint string | `parse_constraints` / `lineq2mat` | `parse_constr.py:26 / 89` | wherever a string enters |
| encode a regulatory bound | `extend_model_regulatory` | `networktools.py:1187` | reaction-based now; gene-based after GPR |
| gene KO → flux constraint | `gene_kos_to_constraints` | `networktools.py:438` | in `fba`/`fva` helpers |
| remap modules to compressed space | `compress_modules` | `networktools.py:1314` | after each `compress_model` |
| remap costs to compressed space | `compress_ki_ko_cost` | `networktools.py:1358` | after each `compress_model` |

We take them roughly in pipeline order, but front-load the constraint parser because everything else
consumes its output.

### 12.1 `remove_ext_mets` — deleting the boundary layer before compression

`remove_ext_mets(model)` (`compression.py:1707`) is three statements:

```python
def remove_ext_mets(model) -> None:
    external_mets = [m for m in model.metabolites if m.compartment == 'External_Species']
    model.remove_metabolites(external_mets)
    stoich_mat = create_stoichiometric_matrix(model)
    obsolete_reacs = [r for r, has_nonzero in zip(model.reactions, np.any(stoich_mat, 0)) if not has_nonzero]
    model.remove_reactions(obsolete_reacs)
```

**What an "external metabolite" is.** In constraint-based models the network's interface with its
surroundings is drawn one of two ways. Either the boundary is an **exchange reaction** — a reaction with
a single metabolite and one open bound, e.g. `glc__D_e -->` with `-10 ≤ v ≤ 1000`, representing "glucose
may leave/enter the system" — or the boundary metabolite is placed in a dedicated **external
compartment** and given its own balance row in `S`. The two conventions are not interchangeable. In the
exchange-reaction convention, the extracellular species `glc__D_e` still has a steady-state balance
`Σ Sᵢⱼ vⱼ = 0` like any internal metabolite, and the exchange reaction is what closes that balance. In
the external-compartment convention, some model authors additionally add a *species* row for the
truly-external pool (compartment tag `External_Species` here) whose only purpose is bookkeeping — it is
not a mass-balanced internal pool, it is the "outside world."

**Why they must go before compression.** Compression (Ch 3) rests on two exact linear-algebra facts about
the stoichiometric matrix `S ∈ ℝ^{m×n}`:

1. **Conservation-relation (row) removal** deletes metabolite rows that are linearly dependent — a left
   nullspace vector `yᵀS = 0` means that combination of metabolites is conserved and its row is
   redundant.
2. **Coupled/parallel merging** looks at the right nullspace of `S` to find reactions whose fluxes are
   forced proportional in every steady state.

An `External_Species` row is a *fake* balance. It is not a real conservation law of the metabolic
network; it is an artifact of how the author drew the boundary. Left in place, it does two damaging
things. First, it manufactures **spurious conservation relations**: the external pool row plus the
internal pool rows of the same metabolite are linearly dependent by construction, so the row-removal step
either wastes work eliminating a redundancy that is not chemistry, or (worse) the extra row perturbs the
rank count that governs how many conservation relations exist. Second, it manufactures **spurious
exchange structure**: a species that participates in exactly one reaction (a dangling boundary node)
creates a degenerate column/row pattern that the coupling analysis can misread as a forced flux
relationship. Removing the `External_Species` rows first means the nullspace math sees only genuine
internal mass balances, so every conservation relation it removes and every coupling it finds is real.

**The obsolete-reaction sweep.** Deleting the external metabolites can strand reactions: an exchange
reaction whose *only* metabolite was the external species now has an all-zero column in `S`. Line 1712
recomputes `S` and drops any reaction whose column is entirely zero (`np.any(stoich_mat, 0)` is the
per-column "has a nonzero" test; `has_nonzero == False` marks a now-empty reaction). These are reactions
that produce/consume nothing after the boundary layer is gone; keeping them would leave free variables
with no stoichiometric effect — pure noise for both FVA and the MILP. The order matters: metabolites
first, then recompute `S`, then reactions — you cannot know a column is empty until the rows are gone.

This runs exactly once, at `compute_strain_designs.py:310`, on the working copy `cmp_model`, immediately
before regulatory extension and COMPRESS #1. It is deliberately the very first surgery: it is the only
step that changes what "a genuine conservation relation" means, so it must precede everything that
reasons about the nullspace.

### 12.2 `parse_constr.py` — strings into `A·x {≤,=,≥} b`

This module is the input→matrix surface every other part of the package sits on. A user (and several
internal callers, including `extend_model_regulatory` below) may express a linear constraint as an
ordinary string, `"2 r1 - r2 <= 5"`. `parse_constr.py` turns that into the sparse rows the LP/MILP layer
consumes. There are two output shapes and the module offers both:

- the **list format** `[{r1: 2.0, r2: -1.0}, "<=", 5.0]` — a dict of coefficients, a sign token, a float
  right-hand side — produced by `parse_constraints` / `lineq2list`. Modules store constraints in this
  format (it survives compression remapping, §12.5, cleanly because it is keyed by reaction id, not by
  column index).
- the **matrix format** `A_ineq, b_ineq, A_eq, b_eq` — the actual sparse rows — produced by `lineq2mat`
  (string → matrix directly) or `lineqlist2mat` / `linexprdict2mat` (list format → matrix).

#### 12.2.1 The scanner: `linexpr2dict` / `linexpr2mat`

The atom of parsing is a single **linear expression** (a left-hand side, no sign, no rhs), handled by
`linexpr2dict` (`parse_constr.py:304`) and its twin `linexpr2mat` (`parse_constr.py:251`). They differ
only in output — a dict vs. a one-row `csr_matrix` — and run the identical scan. Take
`expr = "2 R3 - R1"`, `reaction_ids = ["R1","R2","R3","R4"]`.

1. **Tokenize and strip.** Split on whitespace, then strip leading/trailing sign, space, and parenthesis
   characters from each token (`re.sub(r"^(\s|-|\+|\()*|(\s|-|\+|\))*$", "", part)`, line 321). `"2 R3 -
   R1"` → tokens `["2", "R3", "R1"]` (the lone `-` is stripped away; its sign information is recovered
   later from the raw string, not from this token list).
2. **Identify variables.** `ridx = [r for r in expr_parts if r in reaction_ids]` keeps only tokens that
   are known reaction ids → `["R3", "R1"]`. Membership is by exact string equality against the model's
   reaction id list, which is why reaction ids must be passed in and why digit-leading gene names are
   renamed upstream (Ch 1) — a bare number token would be misread as a coefficient.
3. **Validate syntax** (lines 329–341). Three rules, each raising a descriptive `Exception`:
   - no two numeric tokens in a row (`last_was_number` guard) — `"2 3 R1"` is rejected;
   - no leftover token that is neither a number nor a known reaction id — `"2 Rx"` with unknown `Rx`
     raises `Unknown identifier Rx`;
   - no reaction id may appear twice (`len(ridx) == len(set(ridx))`) — `"R1 + R1"` is rejected, because a
     single sparse cell cannot hold two independent coefficients.
4. **Extract each coefficient** (lines 344–353). For every reaction id `rid`, a regex captures the run of
   sign/digit/dot characters immediately *preceding* that id in the raw string:

   ```python
   coeff = re.search(r"(\s|^)(\s|\d|-|\+|\.)*?(?=" + re.escape(rid) + r"(\s|$))", expr)[0]
   coeff = re.sub(r"\s", "", coeff)          # drop spaces → "" or "+" or "-" or "2" or "-3.5"
   if coeff in ["", "+"]: coeff = 1.0
   elif coeff == "-":     coeff = -1.0
   else:                  coeff = float(coeff)
   ```

   The lookahead `(?=…rid(\s|$))` anchors the capture to the id as a *whole* token (followed by space or
   end), so `R1` does not accidentally match inside `R12`. An empty or `"+"` prefix means an implicit
   `+1`; a lone `"-"` means `-1`; anything else is parsed as a float. For our example: `R3` is preceded
   by `"2 "` → `2.0`; `R1` is preceded by `" - "` → `"-"` → `-1.0`. Result:
   `{"R1": -1.0, "R3": 2.0}` (dict form) or the row `[-1, 0, 2, 0]` (matrix form).

   A minor implementation wart: `linexpr2mat` writes the same logic with a plain `if … if … else` (lines
   294–299) rather than `linexpr2dict`'s `if … elif … else`. The `""`/`"+"` branch sets `coeff = 1.0`
   first, and the subsequent `if coeff == "-"` is then false so the `else` runs `float(1.0)`. It produces
   the identical result, but the two copies of the coefficient logic are a maintenance hazard — a future
   fix to one can silently miss the other.

#### 12.2.2 The (in)equality splitters: `lineq2mat`, `lineq2list`

A full **(in)equality** adds a sign and a right-hand side. `lineq2mat` (`parse_constr.py:89`) is the
one-shot string→matrix path:

```python
lhs, rhs = re.split(r"<=|=|>=", equation)
eq_sign  = re.search(r"<=|>=|=", equation)[0]
rhs      = float(rhs)
A        = linexpr2mat(lhs, reaction_ids)
```

The split isolates the left expression (scanned by `linexpr2mat`) from the rhs (which *must* parse as a
single float — the `except` clause at line 126 rejects anything else with "Right hand side must be a
float number"). Then the sign decides which matrix the row joins, and here is the one genuine piece of
math in the parser — **canonicalizing every inequality to `≤`**:

- `=`  → append `A` to `A_eq`, `rhs` to `b_eq`.
- `<=` → append `A` to `A_ineq`, `rhs` to `b_ineq`.
- `>=` → append **`-A`** to `A_ineq`, **`-rhs`** to `b_ineq`.

The `≥` case uses the elementary equivalence `a·x ≥ b  ⇔  -a·x ≤ -b`: negate both sides of the row.
Worked example, the docstring's own case (`lineq2mat` docstring, lines 96–101):

```
equations   = ["2*c - b +3*a <= 2", "c - b = 0", "2*b -a >=-2"]
reaction_ids = ["a","b","c"]
```

- `"2*c - b + 3*a <= 2"` → row `[3, -1, 2]` (ordered by `reaction_ids`), sign `<=`, rhs `2` → into
  `A_ineq` / `b_ineq`.
- `"c - b = 0"` → row `[0, -1, 1]`, sign `=`, rhs `0` → into `A_eq` / `b_eq`.
- `"2*b - a >= -2"` → row `[-1, 2, 0]`; because the sign is `>=`, it is stored negated as `[1, -2, 0]`
  with rhs `+2` → into `A_ineq` / `b_ineq`.

Final:

```
A_ineq = [[ 3, -1,  2],      b_ineq = [ 2,
          [ 1, -2,  0]]                 2]
A_eq   = [[ 0, -1,  1]]      b_eq   = [ 0]
```

Note the sign is `*` -optional: `"2*c"` and `"2 c"` both work because `linexpr2mat`'s tokenizer strips
`*` implicitly (it is neither a variable nor a number token and is not part of the coefficient run — the
regex character class does not include `*`, so `"2*c"` tokenizes with the `*` swallowed by the split on
whitespace only if written `2 * c`; written `2*c` the whole token is `"2*c"`, which is *not* a reaction
id and *not* a pure-number token, so it would raise `Unknown identifier`). In practice callers use a
space or the tokenizer path that tolerates it; the safe, always-correct spelling is spaces:
`"2 c - b + 3 a <= 2"`. This is an easy place to trip, so upstream code that generates constraint strings
(including `extend_model_regulatory`, §12.3) writes coefficients with explicit spaces.¹

`lineq2list` (`parse_constr.py:141`) is the same split but emits list format
`(linexpr2dict(lhs), eq_sign, rhs)` instead of matrix rows, and — unlike `lineq2mat` — it does **not**
negate `≥` rows (the sign token is preserved verbatim), because list format records the relation
symbolically for later. Skipping empty strings (line 168) lets it tolerate trailing commas/newlines in a
multi-constraint string.

#### 12.2.3 The dispatcher: `parse_constraints`

`parse_constraints` (`parse_constr.py:26`) is the public front door that normalizes the *many* shapes a
user might pass into one uniform list-of-lists. It handles: falsy input → `[]`; a single string
possibly holding several constraints separated by `,` or `\n` (split at line 49); a single constraint vs.
a list of constraints (the `type(constr[0]) is dict` test distinguishes a lone list-format constraint
from a list of them, line 50); tuples coerced to lists (line 53); and finally, if the entries are still
strings, delegating to `lineq2list` to scan them (line 56). The result is always the list format
`[[{…}, sign, rhs], …]`. This is what modules carry, and it is the input to the compression remapping in
§12.5. `parse_linexpr` (line 60) is the sign-less analogue for bare expressions (objectives, production
ids).

### 12.3 `extend_model_regulatory` — a bound as an intervention

A **regulatory constraint** is an inequality (or equality) on flux that is *not* part of the base
stoichiometric model — for example "the combined flux through PDH and PFL must not exceed 5,"
`1 PDH + 1 PFL <= 5`, or "oxygen uptake is limited," `-EX_o2_e <= 2`. Two distinct uses:

- **Permanent**: the constraint always holds. Just an extra row of the flux polytope.
- **Toggleable / a regulatory *intervention***: the constraint is itself something the algorithm may
  *choose to impose* at a cost. In strain design, this models a regulatory edit ("engineer the cell so
  that PDH+PFL ≤ 5") on the same footing as a reaction knockout: it has a binary decision and a cost, and
  the MILP decides whether to buy it.

`extend_model_regulatory(model, reg_itv)` (`networktools.py:1187`) encodes either kind as **extra
stoichiometry**, so that the downstream LP/MILP machinery — which only understands `Sv = 0` plus bounds —
enforces it without any new constraint type.

#### 12.3.1 The encoding math

Take the toggleable constraint `2 r1 + 3 r2 ≤ 4`. The routine adds one pseudometabolite `m` and up to two
pseudoreactions:

- For each reaction `rᵢ` in the constraint, give `rᵢ` a stoichiometric coefficient `wᵢ` for `m` (so `r1`
  now *produces* 2 `m`, `r2` produces 3 `m`) — `r.add_metabolites({m: w})`, line 1283.
- Add a **bound reaction** `r_bnd`: `m -->` (consumes `m`), with bounds chosen from the sign (lines
  1288–1297). For `≤ rhs`: `-inf ≤ v_bnd ≤ rhs`.
- For the toggleable case only, add a **control reaction** `r_ctl`: `--> m` (produces `m`), fully
  unbounded `-inf ≤ v_ctl ≤ inf` (lines 1299–1305).

The steady-state balance of the new pseudometabolite `m` is the whole trick. With all pieces present:

```
dm/dt = 2·v_r1 + 3·v_r2  −  v_bnd  +  v_ctl  =  0
   ⇒   v_bnd = 2·v_r1 + 3·v_r2 + v_ctl
```

Now read off the two regimes:

- **`r_ctl` active (free `v_ctl`).** `v_bnd = 2v_r1 + 3v_r2 + v_ctl` with `v_ctl` free means `v_bnd` can be
  slid to *any* value regardless of the flux sum, so the bound `v_bnd ≤ 4` never actually constrains
  `2v_r1 + 3v_r2`. The regulatory constraint is **off** (non-binding, virtually absent).
- **`r_ctl` knocked out (`v_ctl = 0`).** Then `v_bnd = 2v_r1 + 3v_r2`, and the bound `v_bnd ≤ 4` becomes
  exactly `2v_r1 + 3v_r2 ≤ 4`. The regulatory constraint is **on**.

So *knocking out `r_ctl` = imposing the regulatory intervention.* That inversion is deliberate: it lets
the identical KO machinery (binary `z`, integer cuts, cost accounting) drive regulatory edits with no
special case — the reaction `r_ctl` is simply added to `ko_cost` with the user's cost `v` (line 1305,
`regcost.update({reg_name: v}}`), and the orchestrator folds `regcost` into `cmp_ko_cost`
(`compute_strain_designs.py:327` for reaction-based). For a **permanent** constraint (cost `np.nan`),
`r_ctl` is simply omitted (the `if not np.isnan(v)` guard, line 1299): with no control reaction there is
no `+v_ctl` term, `v_bnd = 2v_r1 + 3v_r2` always, and the bound holds unconditionally.

The equality and `≥` cases set `r_bnd`'s bounds accordingly (lines 1288–1297): `=` pins
`v_bnd` to `rhs` (the code sets lower and upper to `rhs`; note lines 1289–1291 set upper then lower —
the intermediate `-inf` upper is immediately overwritten, so the net effect is `v_bnd = rhs`), and `≥`
uses `rhs ≤ v_bnd ≤ inf`. The `m -->` / `--> m` directions never change; only the `r_bnd` bounds carry
the relation.

This is the *same* "encode a linear relation as pseudometabolite balance" idea that GPR integration uses
to turn Boolean gene rules into flux structure (Ch 4) — here applied to a single user inequality rather
than an AND/OR tree.

#### 12.3.2 Name generation and the in-place dict mutation (footgun)

The generated reaction id for `r_ctl` (and the cost-dict key) is built from the parsed constraint (lines
1260–1273): each term contributes `p`/`n` (sign of coefficient) + the coefficient + `_` + reaction id +
`_`; then `le_`/`ge_`/`eq_` for the relation; then the rhs with `-`→`n` and `.`→`p`. So
`2 r1 + 3 r2 <= 4` becomes something like `p2.0_r1_p3.0_r2_le_4`. These names are what appear in the
returned `regcost` dict and, after decompression, in the reported solution.

The sharp edge is that `extend_model_regulatory` **mutates its `reg_itv` argument in place** (lines
1274–1275):

```python
reg_itv.pop(k)                                            # remove the original string key
reg_itv.update({reg_name: {'str': k, 'cost': v}})         # replace with the generated name
```

It walks a `.copy()` of the items (line 1251) but pops from and writes to the *original* dict. On return,
the caller's dict no longer has the human-readable keys the caller passed in — they have been rewritten
to generated names, with the original string demoted to a `'str'` field inside the value. Because the
orchestrator's `uncmp_reg_cost` **aliases the caller's `reg_cost`** (it is bound by reference, then
`.clear()`/`.update()`-ed at `compute_strain_designs.py:329–330`), a single `compute_strain_designs`
call silently empties and refills the caller's `reg_cost` dict. Re-running with the same dict object then
mis-parses (the keys are now generated names, not constraint strings). The fix is to never reuse a
`reg_cost` dict across calls — pass a fresh one. This is catalogued as a known footgun in Ch 10; the
mechanism is exactly the in-place `pop`/`update` above.

#### 12.3.3 Reaction-based (immediate) vs. gene-based (deferred)

A regulatory constraint may reference **reactions** (`1 PDH + 1 PFL <= 5`) or **genes**
(`b0351 <= 2`, limiting a gene's activity). The distinction controls *when* the encoding can run, and the
orchestrator splits them at `compute_strain_designs.py:315–330`:

- **Reaction-based** constraints parse successfully against the current reaction id set, so they are
  encoded **immediately** (line 327), *before* COMPRESS #1. `parse_constraints(k, _rxn_ids)` succeeds →
  the constraint goes into `_immediate_reg` and `extend_model_regulatory` runs at once.
- **Gene-based** constraints reference identifiers that are *not* reaction ids yet — the gene has no
  pseudoreaction until GPR integration builds one (Ch 4). Trying to parse them against reaction ids
  throws, so they are routed to `_deferred_reg` (line 325) and held. They are encoded only **after**
  `extend_model_gpr` has created the `g_<gene>` pseudoreactions (`compute_strain_designs.py:411–417`),
  at which point the gene name *is* a reaction id and the same `extend_model_regulatory` call works.

The ordering is not cosmetic. A gene-regulatory bound `g <= X` is a bound on the *gene pseudoreaction's*
flux, and that pseudoreaction does not exist before GPR extension; encoding it early would fail to find
the identifier. There is a second, subtler reason the orchestrator protects gene-controlled reactions
from COMPRESS #1: if a gene controls several reactions that get merged before GPR integration, the merged
reaction is hooked to the gene with a collapsed stoichiometry and the gene-regulatory bound would be
mis-scaled. The code therefore adds those reactions to `no_coupled_compress_reacs`/`no_par_compress_reacs`
so they survive to COMPRESS #2, where the `g_gene` metabolite already exists and the merge is correct
(`compute_strain_designs.py:341–353`). The gene-vs-reaction encoding split and its rationale belong to Ch
4; here the point is only that `extend_model_regulatory` is called *twice* in the pipeline, on two disjoint
sub-dicts, for exactly this reason.

### 12.4 `gene_kos_to_constraints` — a gene KO set as flux constraints

`gene_kos_to_constraints(model, gene_kos)` (`networktools.py:438`) answers a narrower question than the
MILP's GPR machinery: *given a concrete, fixed set of knocked-out genes, which reactions die, and what
constraints pin them off?* It is used by the `fba`/`fva` helpers (Ch 2, Ch 5) when a caller wants to
evaluate a *specific* gene-KO scenario directly, not to *search* for interventions. (The search-time
encoding of gene KOs as intervention structure is `extend_model_gpr`, Ch 4 — a different mechanism.)

Mechanics:

1. **Resolve identifiers** (lines 476–486). Each entry of `gene_kos` may be a gene id or a gene name;
   names are mapped to ids via `{g.name: g.id}`, ids checked directly, unknown identifiers silently
   dropped.
2. **Set gene states** (line 489): every knocked gene → `False`; every other gene is implicitly `None`
   (undetermined).
3. **Find candidate reactions** (lines 492–499): the union of reactions linked to any knocked gene
   (`gene_obj.reactions`). Only these can change; no need to evaluate the rest.
4. **Evaluate each GPR** with tri-state Boolean logic, `evaluate_gpr_ast` (`networktools.py:401`). This
   walks the cobra GPR AST (`ast.Name` leaves, `ast.BoolOp` AND/OR nodes) over `{gene: True/False/None}`:
   - **AND**: `False` if *any* child is `False`; `True` only if *all* children are `True`; else `None`.
   - **OR**: `True` if *any* child is `True`; `False` only if *all* children are `False`; else `None`.

   The tri-state (three-valued Kleene) logic is what makes partial knockouts correct: with only some genes
   fixed to `False` and the rest `None`, the evaluator returns `False` *only* when the knockouts alone
   force the rule false — an isozyme (`geneA or geneB`) with just `geneA` knocked evaluates to `None`
   (undetermined, because `geneB` could carry it), not `False`. A reaction is declared dead only on a hard
   `False`.
5. **Emit constraints** (line 513): for each reaction whose GPR evaluated to `False`,
   `[{r_id: 1}, '=', 0]` — the list format meaning `1·v = 0`, i.e. pin the reaction to zero flux. Sorted
   for determinism.

The docstring records the SD grammar these constraints interoperate with: in solution vectors `-1` = KO,
`+1` = KI, `0` = non-added KI (Ch 9). When feeding gene constraints to `fba`/`fva`, both `gene = 0.0` and
`gene = -1.0` are treated as knockouts and `gene = 1.0` (active) is ignored, so a raw SD solution vector
can be handed straight in. The output here, though, is *reaction* constraints in list format — the same
format `parse_constraints` and the modules use — so it slots directly into any `constraints=` argument.

### 12.5 Module & cost compression — keeping references consistent with a moving index

Compression (Ch 3) merges reactions, so after every `compress_model` call the reaction index space
changes: a constraint or objective that named reaction `r7` may now have to name the lumped reaction
`r7*r9`, and a cost that applied to `r7` and `r9` separately must be re-expressed for the merged column.
Two functions repair this, both called right after each of the two compression rounds
(`compute_strain_designs.py:361/363` for round 1, `436/437` for round 2).

Both operate on the **compression map** `cmp_mapReac`, a list of per-step dicts. The field that matters
here is `reac_map_exp = { new_reac : { old_reac : factor, … } }` — for each reaction produced by the
step, the pre-step reactions it stands for, each with a rational `factor` — plus a boolean `parallel`
flag (`True` = parallel merge, `False` = coupled/dependent merge). The map's construction and the
`factor` semantics are Ch 3's territory; Ch 9 documents the reverse walk. Here we need only the forward
relation the factor encodes.

#### 12.5.1 The remapping math

For **both** merge kinds the factor obeys the same linear relation between an original reaction's flux
and the merged reaction's flux:

- **Coupled** merge: the merged reactions are flux-coupled, `v_oldₖ = factorₖ · v_new`, where `factorₖ`
  is the proportionality constant from the shared nullspace direction.
- **Parallel** merge: `v_new` is the *total* flux `Σₖ v_oldₖ`, split by fractions with `Σₖ factorₖ = 1`,
  so again `v_oldₖ = factorₖ · v_new` (`compression.py:2127–2141` builds these fractions from the
  stoichiometric scales).

Given that relation, remap a linear constraint `Σₖ aₖ vₖ {≤,=} b` when a subset `L` of its reactions
merges into `new`:

```
Σ_{k∈L} aₖ vₖ  =  Σ_{k∈L} aₖ (factorₖ · v_new)  =  ( Σ_{k∈L} aₖ·factorₖ ) · v_new
```

So the new coefficient on `v_new` is `Σ_{k∈L} aₖ·factorₖ`, and the right-hand side `b` is unchanged (the
transformation is a change of variables on the left only). This is exactly `compress_modules`
(`networktools.py:1350`):

```python
lumped_reacs = [k for k in c[0].keys() if k in old_reac_val]
c[0][new_reac] = np.sum([c[0].pop(k) * old_reac_val[k] for k in lumped_reacs])
```

`c[0]` is the coefficient dict; `old_reac_val` is `{old: factor}`; each merged term is popped and its
coefficient times its factor is accumulated onto `new_reac`. Objectives (`INNER_OBJECTIVE`,
`OUTER_OBJECTIVE`, `PROD_ID`) are linear expressions and get the identical treatment (lines 1351–1354).
Coefficients are first converted to exact rationals (`modules_coeff2rational`, line 1336) so the
factor multiply-and-sum stays exact — the same integer/rational discipline compression itself insists on
(Ch 3): never let a merge introduce float drift into a constraint that the MILP will treat as hard.

**Worked micro-example.** Module constraint `2 r7 - r9 <= 5`, and a coupled step merges `r7, r9` into
`r7*r9` with `v_r7 = 1·v_new`, `v_r9 = ½·v_new` (i.e. `factor_{r7}=1`, `factor_{r9}=½`). New coefficient
`= 2·1 + (−1)·½ = 3/2`, so the compressed constraint is `1.5 (r7*r9) <= 5`.

**Why `compress_modules` skips parallel steps.** Line 1340 guards the whole remap with `if not parallel:`
— it rewrites constraints/objectives *only* for coupled steps. This is safe, and necessary, because
reactions referenced in any module are **protected from parallel merging** in the first place:
`_collect_no_par_compress_reacs` (`compute_strain_designs.py:38`) gathers every reaction id named in a
module's constraints/objectives and passes them as `no_par_compress_reacs` to `compress_model`
(`compute_strain_designs.py:333, 433`), which exempts them from the parallel compressor. A
module-referenced reaction therefore never appears on the `old` side of a parallel `reac_map_exp`, so
there is nothing to remap for those steps — and if the code *did* try, it would still be correct but
redundant. (Coupled merges are not exempted this way; a module reaction may be coupled-merged, which is
precisely why the coupled branch must run the remap.)

#### 12.5.2 Cost remapping and the parallel/coupled asymmetry

`compress_ki_ko_cost(kocost, kicost, cmp_mapReac)` (`networktools.py:1358`) does the analogous job for the
knockout- and knock-in-cost dicts, but here the merge kind genuinely changes the *arithmetic*, because a
cost is a property of "cutting/adding this reaction," and what a cut of the *merged* reaction physically
means differs between the two merge types.

First, for provenance, each step records the cost dicts *as they stood entering that step* (line 1392,
`cmp.update({KOCOST: kocost, KICOST: kicost})`) — this is the self-describing invariant the reverse
expansion in Ch 9 relies on. Then it rebuilds the dicts (lines 1393–1410):

- **KO cost of a merged reaction** (lines 1394–1401):
  - **coupled** (and none of the group is a KI candidate): `min` of the members' KO costs. A coupled group
    fires together — knocking any one that carries the group's flux kills the whole coupled flux — so the
    cheapest cut suffices, hence the minimum.
  - **parallel**: `sum` of the members' KO costs. Parallel reactions are *alternative* routes carrying the
    same conversion; to actually knock the lumped capacity out you must cut *all* of them, so the costs add.
- **KI cost of a merged reaction** (lines 1402–1410) — the mirror image:
  - **coupled**: `sum` of the members' KI costs (adding a coupled pathway means adding every reaction in
    the chain).
  - **parallel** (and none of the group is a KO candidate): `min` (adding *one* of several parallel routes
    restores the capacity, so the cheapest addition wins).

The guard conditions (`not np.any([s in kicost …])` on the coupled-KO branch, and its mirror on the
parallel-KI branch) prevent a reaction that is simultaneously a KO and KI candidate from being collapsed
into the wrong category; such mixed groups fall through and are handled by expansion (Ch 9). The function
returns the rebuilt `kocost, kicost` **and** the annotated `cmp_mapReac` (now carrying the per-step
`KOCOST`/`KICOST` snapshots) — the third return value is what makes decompression able to walk the merge
backward and re-split a merged intervention into the right originals.

The `min`/`sum` asymmetry is the crux and is worth stating plainly: **coupled ⇒ KO-min / KI-sum;
parallel ⇒ KO-sum / KI-min.** Getting it backwards would report designs whose true intervention cost
violates the user's `max_cost` budget, or would prune valid cheap designs.

### 12.6 Ordering, and why it is load-bearing

Reading the preprocessing block (`compute_strain_designs.py:305–495`) top to bottom, the utility calls
interleave with the heavy steps in a sequence that is not arbitrary:

1. `remove_ext_mets` (§12.1) **first** — it redefines what a genuine conservation relation is, so it must
   precede any nullspace reasoning (COMPRESS #1, all FVAs).
2. **Reaction-based** `extend_model_regulatory` (§12.3) next, still before COMPRESS #1, so its
   pseudometabolites/pseudoreactions are present when compression analyzes the network and its `r_ctl`
   reactions enter `ko_cost` before FVA prunes essentials.
3. **COMPRESS #1**, then immediately `compress_modules` + `compress_ki_ko_cost` (§12.5) so modules and
   costs track the new index *before* the next step reads them. Skipping the remap here would leave
   modules naming reactions that no longer exist.
4. **GPR extension** (Ch 4), then **gene-based** `extend_model_regulatory` (§12.3.3) — deferred to exactly
   this point because the gene pseudoreactions it references do not exist earlier.
5. **COMPRESS #2**, then `compress_modules` + `compress_ki_ko_cost` again on the round-2 map; the two maps
   are concatenated (`cmp_mapReac = cmp_mapReac_1 + cmp_mapReac_2`, line 439) into the single history that
   decompression (Ch 9) later replays in reverse.

`gene_kos_to_constraints` (§12.4) sits outside this sequence — it is invoked on demand by the
`fba`/`fva` helpers whenever a caller evaluates a fixed gene-KO scenario — but it emits the same list
format and so composes cleanly with everything above. `parse_constr.py` (§12.2) is the substrate under
all of it: every constraint string, whether from a module, a regulatory intervention, or a direct
`fva(constraints=…)` call, becomes rows through the same scanner, guaranteeing one consistent
`A·x {≤,=,≥} b` convention across the whole package.

¹ Footnote on the `*` tokenization: the scanner splits only on whitespace, so `"2*c"` written without
spaces around `*` becomes a single token `"2*c"` that is neither a known reaction id nor a pure-number
match and therefore raises `Unknown identifier 2*c`. The docstrings advertise `"r1 + 3*r2 = 0.3"`-style
input, which works only when the `*` is adjacent to a coefficient that the surrounding regex tolerates;
the reliably-correct spelling used internally is space-separated (`"3 r2"`). If you extend the parser,
either strip `*` in the tokenizer or document the space requirement — the current behavior is
inconsistent with the examples in its own docstrings.


## 13. The object model & result API

Two small Python classes bracket the entire computation and are the only StrainDesign
types most users ever hold in their hands. **`SDModule`** (`strainDesignModule.py`) is the
*input* object: it says *what strain-design goal you want*. **`SDSolutions`**
(`strainDesignSolutions.py`) is the *output* object: it holds *the intervention sets that were
found* and translates them between the internal, compressed representation and the
reaction/gene view a modeller reasons about. Between them sits a third, less obvious surface —
the **preprocessed dump** (`dump_preprocessed` / `compute_strain_designs_from_preprocessed`) —
which lets a developer freeze the expensive preprocessing once and replay the cheap MILP solve
many times. This chapter documents all three as an *API contract*: what each field means, what
each method returns, when to reach for which, and why the objects are shaped the way they are.

The mathematics of *how* a design is decompressed (composing the reverse compression maps),
how knock-ins are encoded as value-0/`(nan,nan)`, and how `strip_non_ki`/`expand_sd` work are
owned by **Ch 9**; this chapter references those rules and instead documents the *surface* a
user touches. The dualization/z-linking that turns a module into MILP rows is Ch 6/7; here a
module is just a validated specification.

### 13.1 `SDModule` — the problem-specification object

#### 13.1.1 What it is: a validated `dict` subclass

`SDModule` is declared as

```python
class SDModule(Dict):          # strainDesignModule.py:29
    def __init__(self, model, module_type, *args, **kwargs):
```

i.e. it **subclasses `dict`**. An `SDModule` *is* a dictionary; every field is a key. After
construction, `m[CONSTRAINTS]`, `m[MODULE_TYPE]`, `m[INNER_OBJECTIVE]` etc. are ordinary
dictionary lookups (the constants are string keys defined in `names.py`, e.g.
`MODULE_TYPE = 'module_type'`, `CONSTRAINTS = 'constraints'`). The rest of the pipeline never
uses attribute access — `strainDesignProblem.py`, `compress_modules`, the FVA scoping loop and
so on all read `m[CONSTRAINTS]`, `m[MODULE_TYPE]`. The class is, in effect, **a schema-checked
dict with a constructor that parses and validates**.

**Why a dict-subclass rather than a class with positional/attribute fields or a dataclass?**
Three concrete reasons, all visible in the code:

1. **The set of meaningful fields is type-dependent and sparse.** An `optknock` module needs
   `inner_objective` + `outer_objective`; a `suppress` module needs neither (both optional); an
   `optcouple` module needs `inner_objective` + `prod_id` but *forbids* `outer_objective`. A
   flat keyword bag with per-type validation expresses "these keys are relevant, those are not"
   far more naturally than a fixed positional signature would. Unused fields are simply set to
   `None` (`strainDesignModule.py:235-237`), so every module carries the same key set and
   downstream code can blindly read `m[OUTER_OBJECTIVE]` without `hasattr` guards.

2. **Modules must survive serialization and transformation as plain data.** They are embedded
   verbatim into the `sd_setup` dict stored on every `SDSolutions` (under the `MODULES` key), are
   `deepcopy`-ed repeatedly (setup construction, `SDModule.copy`), are JSON-dumpable when a setup
   is written to a `.json` file (`compute_strain_designs` accepts `SETUP` as a path and
   `json.load`s it, `compute_strain_designs.py:181`), and are **remapped through the compression
   map** by `compress_modules` (Ch 12), which walks the constraint dicts and rewrites reaction
   keys. A dict subclass is trivially all of these; a bespoke class would need custom
   `__getstate__`/`to_dict` glue.

3. **The constructor is the single validation gate.** Because everything is keyed, one loop
   (`strainDesignModule.py:229-233`) can enforce the whitelist:

   ```python
   allowed_keys = {CONSTRAINTS, INNER_OBJECTIVE, INNER_OPT_SENSE, OUTER_OBJECTIVE,
                   OUTER_OPT_SENSE, INNER_OPT_TOL, OUTER_OPT_TOL, PROD_ID,
                   'skip_checks', MIN_GCP, 'reac_ids'}
   for key, value in kwargs.items():
       if key in allowed_keys: self[key] = value
       else: raise Exception("Key " + key + " is not supported.")
   ```

   A typo like `inner_objectiv=...` raises immediately rather than being silently ignored (the
   failure mode of `**kwargs` bags and of `setattr`-based objects). This is the payoff of the
   design: **fail loud at specification time, in the user's own call, long before the MILP is
   built.**

Note `model` and `module_type` are the only *positional* arguments; the first two lines of the
constructor set `self[MODEL_ID] = model.id` and `self[MODULE_TYPE] = module_type`. Everything
else is keyword-only in practice (`*args` is accepted but ignored).

#### 13.1.2 The six module types

`module_type` must be one of six strings (`strainDesignModule.py:245`), all defined in
`names.py`:

| Type | `names.py` value | Global objective it implies | Mandatory fields (beyond model/type) |
|------|------------------|------------------------------|--------------------------------------|
| **PROTECT** | `'protect'` | (none — cost-minimizing MCS) | — (`constraints` optional) |
| **SUPPRESS** | `'suppress'` | (none — cost-minimizing MCS) | — (`constraints` optional) |
| **OPTKNOCK** | `'optknock'` | maximize `outer_objective` | `inner_objective`, `outer_objective` |
| **ROBUSTKNOCK** | `'robustknock'` | max–min of `outer_objective` | `inner_objective`, `outer_objective` |
| **OPTCOUPLE** | `'optcouple'` | maximize growth-coupling potential | `inner_objective`, `prod_id` |
| **DOUBLEOPT** | `'doubleopt'` | (bilevel, like OptKnock) | `inner_objective`, `outer_objective` |

A single computation may contain **at most one** of OPTKNOCK/ROBUSTKNOCK/OPTCOUPLE/DOUBLEOPT
(they define the *global* objective), plus **arbitrarily many** PROTECT and SUPPRESS modules.
When only PROTECT/SUPPRESS modules are present, the global objective is "minimize the number
(cost) of interventions" — the classical MCS problem. The **semantics** of each type (SUPPRESS
= make a flux region infeasible via a Farkas certificate; PROTECT = keep a region feasible;
the bilevel types = nest an inner LP via strong duality) are the subject of Ch 1 and Ch 6; here
they are just labels that select a validation branch.

Two documentation caveats worth flagging: `DOUBLEOPT` is a valid, accepted type in the code
(`names.py:127`, validated exactly like OptKnock/RobustKnock at
`strainDesignModule.py:248-257`) but is **not** described in the class docstring — the docstring
predates it. And `names.py:120-123` deliberately rebinds `PROTECT`/`SUPPRESS`: they are first
set to legacy internal strings `'mcs_lin'`/`'mcs_bilvl'` and then *immediately overwritten* with
`'protect'`/`'suppress'`, so only the latter two are live. The overwrite is intentional (the old
strings are kept in the module docstring for historical reference only).

#### 13.1.3 Per-type validation, step by step

The constructor's validation (`strainDesignModule.py:244-339`) runs in this order:

1. **Type whitelist** (`:245`). Unknown `module_type` → exception.

2. **Bilevel objective presence & senses** (`:248-268`).
   - For OPTKNOCK/ROBUSTKNOCK/DOUBLEOPT: default `inner_opt_sense`/`outer_opt_sense` to
     `MAXIMIZE` if unset; both must be `'minimize'` or `'maximize'`; **both** `inner_objective`
     and `outer_objective` must be non-`None`, else raise.
   - For OPTCOUPLE: default `inner_opt_sense` to `MAXIMIZE`; default `min_gcp` to `0.0`;
     require `inner_objective` **and** `prod_id`. (No `outer_objective` — the outer objective is
     implicitly the growth-coupling potential.)

3. **MCS-with-inner-objective wrinkle** (`:269-276`). PROTECT/SUPPRESS normally take no outer
   objective, but *if one is supplied*, an `inner_objective` becomes mandatory and `outer_opt_sense`
   is defaulted/validated. This supports the "optimal-yield-at-max-growth" pattern the docstring
   describes.

4. **Optimality tolerances** (`:277-282`). `inner_opt_tol`/`outer_opt_tol`, if given, must lie in
   `(0, 1]` — a fraction of the optimum (`1.0` = exact, `0.95` = "within 95 % of optimal"). These
   feed the inner/outer LP as an ε-optimality band.

5. **`reac_ids` fallback** (`:284-285`). If no explicit reaction-id list was passed, it is taken
   from `model.reactions.list_attr('id')`. This is why a *dummy* model works: pass
   `skip_checks=True` and `reac_ids=[...]` and the constructor never touches `model.reactions`
   (see the guard at `:239-242`, which errors only if *both* `reac_ids` and `model.reactions` are
   empty).

6. **Parsing to matrix/dict form** (`:290-308`). This is where free-form user input is normalized
   (all via `parse_constr.py`, Ch 12):
   - `constraints` → a list of `[coeff_dict, sign, rhs]` triples via `parse_constraints`. So
     `'growth >= 0.1'` becomes `[[{'growth': 1.0}, '>=', 0.1]]`. `None` becomes `[]`.
   - `inner_objective`, `outer_objective`, `prod_id`, if strings, → coefficient dicts via
     `linexpr2dict`. So `'BIOMASS - 0.05 EX_etoh_e'` becomes
     `{'BIOMASS': 1.0, 'EX_etoh_e': -0.05}`. Passing a dict directly skips parsing.

   **Both string and dict forms are accepted for every expression field** — a deliberate
   convenience so the same module can be written terse (strings) or programmatic (dicts).

7. **Feasibility checks** (`:311-339`, skipped when `skip_checks=True`):
   - The constraints alone must leave the *original* model feasible: `fba(model,
     constraints=self[CONSTRAINTS]).status != INFEASIBLE`. This catches contradictory or
     mistyped constraints at construction time.
   - **The zero-vector exclusion** for SUPPRESS/PROTECT-with-inner-objective (`:316-320`): the
     constructor pins *every* reaction to 0 (`[[{k:1},'=',0] for k in reactions]`) and checks
     that the constraint region is then infeasible. If the all-zero flux vector satisfies the
     module's constraints, the module is ill-posed (an MCS can never exclude the trivial
     resting state) and it raises. This is a genuinely subtle correctness guard — it is why a
     suppress constraint is written `'growth >= 0.01'` (excludes 0) rather than `'growth >= 0'`
     (includes 0).
   - Every reaction referenced in `inner_objective`/`outer_objective`/`prod_id` must exist in
     `reac_ids` (`:322-331`), and `min_gcp` must be numeric (int is coerced to float, `:333-339`).

`skip_checks=True` bypasses items 7 entirely — used internally when a module is reconstructed
from already-validated data (see `SDModule.copy`, `:341-359`, which rebuilds via a `DummyModel`
carrying only `.id` and passes `skip_checks=True`).

#### 13.1.4 Construction examples

**Classical gene/reaction MCS** — "find minimal knockout sets that make growth ≥ 0.01
impossible while keeping the model otherwise feasible". One SUPPRESS module suffices; PROTECT is
implicit (an empty-constraint PROTECT just keeps the model feasible and is usually unnecessary):

```python
from straindesign import SDModule
import cobra
model = cobra.io.load_model('e_coli_core')

# Undesired behaviour to eliminate: any growth at/above 0.01
suppress = SDModule(model, 'suppress',
                    constraints='BIOMASS_Ecoli_core_w_GAM >= 0.01')
```

Internally this becomes `suppress[CONSTRAINTS] == [[{'BIOMASS_Ecoli_core_w_GAM': 1.0}, '>=',
0.01]]`, `suppress[MODULE_TYPE] == 'suppress'`, all other fields `None`/`[]`. Passing it to
`compute_strain_designs(model, sd_modules=suppress, ...)` yields the classical MCS.

To *also protect* a minimum viable growth of a different, desired phenotype (the standard MCS
pair), add a PROTECT module and pass the list:

```python
protect = SDModule(model, 'protect',
                   constraints='BIOMASS_Ecoli_core_w_GAM >= 0.05')
compute_strain_designs(model, sd_modules=[suppress, protect], solution_approach='populate')
```

**Bilevel — OptKnock for ethanol** — "maximize ethanol export at the growth-optimal flux state,
guaranteeing growth ≥ 0.2":

```python
optknock = SDModule(model, 'optknock',
                    inner_objective='BIOMASS_Ecoli_core_w_GAM',   # cell optimizes growth
                    outer_objective='EX_etoh_e',                  # we optimize ethanol
                    constraints='BIOMASS_Ecoli_core_w_GAM >= 0.2')
```

Here `inner_objective`/`outer_objective` become coefficient dicts, `inner_opt_sense` and
`outer_opt_sense` default to `'maximize'` (`:250-252`), and the constructor verifies that both
objectives reference real reactions and that the growth-≥-0.2 constraint is satisfiable. For
OptCouple you would instead pass `inner_objective='BIOMASS...'` and `prod_id='EX_etoh_e'` (no
outer objective), optionally with `min_gcp=0.05`.

### 13.2 `SDSolutions` — the result object

`SDSolutions` (`strainDesignSolutions.py:31`) is the return value of `compute_strain_designs`,
`compute_strain_designs_from_preprocessed`, and the lower-level `SDMILP` compute methods. Its
docstring is blunt: *"Instances of this class are not meant to be created by StrainDesign
users."* The orchestrator builds it; the user reads it.

#### 13.2.1 What a "design" is: the intervention dict

The atomic unit is an **intervention set**: a plain `dict` mapping a reaction/gene/regulatory
identifier to an integer-valued marker. The constructor docstring (`:47-54`) defines the
encoding, and `_compute_costs_and_bounds` (`:246-281`) turns it into bounds:

| Value in dict | Meaning | Reaction bounds produced (`itv_bounds`) |
|---------------|---------|------------------------------------------|
| `-1` | **knock-out** (KO) — remove the reaction/gene | `(0.0, 0.0)` |
| `1` | **knock-in** (KI) that *was* added | the reaction's own `model` bounds |
| `0` | a candidate KI that was **not** added | `(nan, nan)` |
| `True` | regulatory intervention **active** | derived from the parsed reg. constraint |
| `False` | regulatory intervention **not** added | (omitted / no bound change) |

The `(nan, nan)` sentinel for "not-added KI" is the crux of the KI accounting (full derivation
in Ch 9): a value-0 entry is *carried through* the solution so that cost bookkeeping and
superset-comparison see the full candidate set, but it represents no actual edit — hence bounds
that are literally "not a number". The `-1`/`1`/`0` trichotomy exists precisely because a KI is
not simply the absence of a KO: the same reaction can be a KO candidate in one design and a
not-added KI candidate in another, and the object must distinguish them.

`itv_bounds` is computed once at construction (`:246-281`) and cached; `get_reaction_sd_bnds`
just returns it. For a KO you get `(0,0)`; for an added KI you get the reaction's real bounds
(so the caller can re-impose them on a model); regulatory `True` entries with a *simple*
single-reaction constraint are folded into a bound (`:256-281`), while complex multi-reaction
regulatory constraints set `has_complex_regul_itv = True` and are left as symbolic strings.

#### 13.2.2 Internal storage

The fields set by `__init__` (`:72-105`):

- **`reaction_sd`** — `list[dict]`, the designs at *reaction* level. Always present.
- **`gene_sd`** — `list[dict]`, the designs at *gene* level. Present **only** when the
  computation used gene knockouts/knock-ins (i.e. `GKOCOST` or `GKICOST` in `sd_setup`); the
  flag `is_gene_sd` records this (`:91-99`). In gene mode, the raw solution dicts are
  gene-keyed, so the constructor calls `_translate_genes_to_reactions` (`:134-201`) to derive
  `reaction_sd` from `gene_sd` via cobra's parsed GPR AST (`reaction.gpr.eval`, Ch 9 owns this
  translation). In reaction mode `reaction_sd` *is* the raw input and `gene_sd` does not exist.
- **`sd_cost`** — `list[float]`, one total cost per design, summed over the applicable cost
  dictionaries (`KOCOST`/`KICOST`/`GKOCOST`/`GKICOST`/`REGCOST`) in `_compute_costs_and_bounds`
  (`:217-243`). An entry contributes its cost only when present *and non-zero* in the design
  (`if k in s and s[k] != 0`), so a not-added KI (value 0) costs nothing — consistent with the
  bounds table above.
- **`itv_bounds`** — `list[dict]`, the per-design bound overrides described in 13.2.1.
- **`status`** — the solver/computation status string (`'optimal'`, `'infeasible'`,
  `'time_limit_w_sols'`, …; from `names.py`, ultimately optlang's `optlang.interface`
  constants).
- **`sd_setup`** — the self-describing setup dict (`MODEL_ID`, `MODULES`, the cost dictionaries,
  `MAX_COST`, `SOLVER`, …): the reproducibility record embedded in the object. It is a
  first-class object with its own dual input/output role — see §13.3.
- **Compression bookkeeping** (set *after* construction by the orchestrator, not in
  `__init__`): `compressed_sd` (the designs in the compressed model's reaction space),
  `compression_map` (`cmp_mapReac`, the list of reverse-compression steps, Ch 3/9), and
  `group_map` (a parallel list mapping each expanded design index → the index of the compressed
  design it came from). These enable the group/representative API in 13.2.4.

#### 13.2.3 The public accessor contract

The methods differ along two axes: **level** (reaction vs gene) and **whether not-added KIs are
shown**. The rule for the "clean" accessors is `strip_non_ki` (`:768-770`):

```python
def strip_non_ki(sd):
    return {k: v for k, v in sd.items() if v not in (0.0, False)}
```

— it drops the value-0 KI markers and the `False` regulatory markers, leaving only *actual*
interventions (KO `-1`, added KI `1`, active reg `True`).

| Method | Returns | Strips non-added KIs? | When to use |
|--------|---------|:---:|-------------|
| `get_reaction_sd(i=None)` | reaction-level design(s) | **yes** | The default "what do I actually change" view. Reaction KOs/KIs only. |
| `get_gene_sd(i=None)` | gene-level design(s) | **yes** | The genetic engineering deliverable (which genes to KO/KI). Raises if `is_gene_sd` is False. |
| `get_reaction_sd_bnds(i=None)` | list of `{reac: (lb,ub)}` | n/a | When you want to *apply* a design to a cobra model (set bounds) or inspect KO as `(0,0)`. |
| `get_strain_designs(i=None)` | gene_sd if gene mode else reaction_sd | yes | Level-agnostic: "give me the designs in their native level." |
| `get_strain_design_costs(i=None)` | cost float(s) | n/a | Rank/filter designs by intervention cost. |
| `get_reaction_sd_mark_no_ki(i=None)` | reaction-level, **raw** | **no** | Analysis that must see the *candidate* KIs that were declined (value 0). |
| `get_gene_sd_mark_no_ki(i=None)` | gene-level, **raw** | **no** | Same, at gene level. |
| `get_gene_reac_sd_assoc(i=None)` | `(reacs, assoc, gene_sd)` | yes | Map the (often n:1) many-genes-→-one-reaction-phenotype relationship. |
| `get_num_sols()` | int | — | Count; returns the *estimated* total in lazy mode (see 13.2.4). |

`i` may be `None` (all designs), a single `int`, or a list of indices; a bare `int` is wrapped
to `[i]` internally. Two contract subtleties to note:

- **`get_reaction_sd` vs `get_reaction_sd_mark_no_ki`.** They differ *only* by `strip_non_ki`.
  If you are presenting an engineering result, use the stripped `get_reaction_sd`. If you are
  reasoning about *why* a KI was or wasn't chosen (superset logic, cost accounting), use
  `..._mark_no_ki` so the value-0 entries remain visible.
- **The attributes are public too.** `sol.reaction_sd`, `sol.gene_sd`, `sol.itv_bounds`,
  `sol.sd_cost`, `sol.status` are documented fields, not just internals. Reading them directly
  gives you the *raw* (unstripped) lists; the `get_*` methods are the curated view. `itv_bounds`
  has no stripping variant — `get_reaction_sd_bnds` returns it as-is.

`get_gene_reac_sd_assoc` (`:366-388`) deserves a note: gene-level designs are frequently
degenerate — several distinct gene-knockout sets collapse to the *same* reaction-level
phenotype (because different genes gate the same reactions through the GPR). This method
deduplicates the reaction-level designs by hashing `json.dumps(s, sort_keys=True)` and returns
`(unique_reaction_designs, association_indices, gene_designs)` so a caller can display "these 4
gene strategies all realize reaction phenotype #2."

#### 13.2.4 Lazy expansion and representatives (PR #40)

Decompression can be combinatorially explosive: one compressed design, when its merged
reactions are expanded back to originals, can fan out into an enormous number of equivalent
full-model designs. Materializing all of them is often pointless (they are interchangeable) and
can exhaust memory or hang `save` (issue #47). PR #40 introduced **lazy expansion** to defer
that fan-out.

The mechanism lives across `_decompress_solutions` (`compute_strain_designs.py:641`) and
`SDSolutions`. When the orchestrator's `estimate_expansion_size` exceeds
`LAZY_EXPANSION_THRESHOLD` (`= 100_000`, `compute_strain_designs.py:638`), it builds **one
representative expanded design per compressed group** via `_build_lazy_representatives`
(`:721-756`, taking `expanded[0]`, the cheapest, per group) and constructs the solution with a
`_lazy_init` payload:

```python
sd_solutions = SDSolutions(orig_model, sd, status, setup, _lazy_init=lazy_meta)
```

`lazy_meta` (`:667-676`) carries everything needed to expand a group on demand later:
`compressed_sd`, `compression_map`, the uncompressed cost dicts, `max_cost`, the live `model`,
and `estimated_total`. In lazy mode (`self._lazy == True`, `:75`):

- **`get_num_sols()`** returns `self._estimated_total` (the *estimated* full count), not the
  number materialized (`:284-288`). `get_num_materialized()` returns the actual count in
  `reaction_sd`.
- **`get_representative_sd()`** (`:431-444`) returns one stripped design per compressed group —
  the cheap, canonical answer. If there is no `group_map` it falls back to `get_reaction_sd()`.
- **`get_group(i)`** / **`get_num_groups()`** (`:414-429`) expose the group structure: which
  materialized indices share a compressed origin, and how many distinct compressed designs
  exist.
- **`expand_group(grp_idx)`** (`:446-518`) does the on-demand work: it calls `expand_sd` +
  `filter_sd_maxcost` (Ch 9) for that one group, re-runs the regulatory post-processing and the
  GPR translation + cost/bounds computation, then **splices** the results into `reaction_sd`,
  `sd_cost`, `itv_bounds`, `group_map` (and `gene_sd`) in place, replacing the single
  representative. It requires a live `self._model` — if the object was loaded without one it
  raises with an actionable message pointing at `load(..., model=True)` or `attach_model`.
- **`expand_all(n_per_group=None)`** (`:520-542`) expands every not-yet-expanded group,
  optionally capping to `n_per_group` designs per group, then clears `self._lazy`.

The design contract for a developer: **treat a fresh `SDSolutions` as possibly lazy.** Call
`is_lazy` / `get_num_materialized` / `get_num_sols` to see the state; iterate representatives
for a summary; call `expand_group`/`expand_all` only when you truly need the full fan-out — and
only while a model is attached.

#### 13.2.5 Save / load and model embedding

`SDSolutions` is designed to be a **self-contained, portable record** of a computation
(`save`/`load`, `:553-687`). The pickled state already includes the full problem specification
via `sd_setup` (§13.3); embedding a model snapshot closes the remaining gap. The central
complication is that the live `cobra` model carries an
un-picklable solver interface (and would tie the pickle to specific cobra/optlang/solver
versions), so the model is never pickled live. Instead:

- `__getstate__` (`:107-120`) strips `_model`, `_cmp_model`, and the `model` entry inside the
  lazy `_expansion_meta` before pickling.
- `save(filename, embed_model=True)` (`:553-612`) embeds *portable, solver-less snapshots* of
  both the full model and the compressed (GPR-extended) model, produced by StrainDesign's
  **rational-safe** `networktools.model_to_dict`. Rational-safety matters: the compressed
  model's bounds/coefficients are exact rationals (Ch 3), and a naive float round-trip would
  corrupt them. The two snapshots (`_embedded_model_dict`, `_embedded_cmp_model_dict`) are
  written only for *this* pickle and then restored off the live object so a subsequent
  `embed_model=False` save stays lean (`:597-612`).
- `save` **does not force expansion** of lazy/compressed results (`:565-571`) — it pickles them
  as-is, precisely to avoid the memory blow-up of issue #47. To persist a fully-expanded set,
  call `expand_all()` first.
- `load(filename, model=None, cmp_model=None)` (`:638-687`) rebuilds models only on request:
  `None` attaches nothing, `True` rebuilds the embedded snapshot via `model_from_dict`, and a
  passed `cobra.Model` attaches that object directly. `_resolve` (`:678-683`) implements this
  three-way choice independently for the full and compressed model. `get_model` /
  `get_compressed_model` / `attach_model` (`:614-636`) are the retrieval/attachment accessors.
  The compressed model is offered separately because analysing `compressed_sd` in the *small*
  compressed model is far faster than in the full one.

Finally, `SDSolutions` supports **merging** (`__iadd__`/`__add__`, `:704-765`): two result sets
over the same model can be combined, deduplicating at the compressed-design level (via
`frozenset(s.items())`) when compression info is present, or at the expanded level otherwise,
with `OPTIMAL` status winning. `_check_merge_compatible` (`:689-702`) refuses to merge across
different models, across gene/reaction levels, or across incompatible compression maps. This is
what lets the benchmarking harness stitch together the outputs of several seed runs into one
solution set.

### 13.3 The `sd_setup` object — one bundle, two roles

`sd_setup` (the string key `SETUP = 'sd_setup'`, `names.py:160`) is the **single serializable
dictionary that fully describes a strain-design problem** — modules, cost model, solver, and
solve-control parameters, all in plain-data form. It is deliberately not a class: it is a bare
`dict` of string keys → JSON-friendly values, precisely so it can be written to disk, version
controlled, diffed, and handed between processes without any StrainDesign type machinery. (The
modules it contains are `SDModule`s, which — being dict-subclasses, §13.1.1 — are themselves
plain data.) The same object plays two roles at opposite ends of the pipeline: it is the *input*
that specifies a computation, and it is the *record* that travels out with the results.

#### 13.3.1 The key set

The keys are the `names.py` constants; the value types are:

| Key (`names.py`) | String | Value type | Meaning |
|------------------|--------|-----------|---------|
| `MODEL_ID` | `'model_id'` | str | `model.id` the problem was posed on |
| `MODULES` | `'sd_modules'` | `list[SDModule]` | the problem specification (§13.1) |
| `KOCOST` / `KICOST` | `'ko_cost'` / `'ki_cost'` | `dict[str,float]` | per-reaction KO / KI costs |
| `GKOCOST` / `GKICOST` | `'gko_cost'` / `'gki_cost'` | `dict[str,float]` | per-gene KO / KI costs (present ⇒ gene mode) |
| `REGCOST` | `'reg_cost'` | `dict[str,·]` | regulatory-intervention costs |
| `MAX_COST` | `'max_cost'` | float | cost cap for a design |
| `SOLVER` | `'solver'` | str | `'cplex'`/`'gurobi'`/`'scip'`/`'glpk'` |
| `MAX_SOLUTIONS` | `'max_solutions'` | int/float | MILP solution cap |
| `SOLUTION_APPROACH` | `'solution_approach'` | str | `'any'`/`'best'`/`'populate'` |
| `T_LIMIT` | `'time_limit'` | int/float | solver time limit (s) |
| `SEED` | `'seed'` | int | MILP seed (solver B&B) |
| `'M'` | — | int/None | big-M value (None ⇒ indicator constraints) |
| `'compress'`, `'gene_kos'`, `MILP_THREADS`, `'advanced'`, `'use_scenario'` | — | bool/int | compression toggle, gene-KO mode, threads, CNApy dummies |

These are exactly the `allowed_keys` that `compute_strain_designs` accepts as top-level kwargs
(`compute_strain_designs.py:174-177`) — which is the whole point: **`sd_setup` is a frozen copy
of the keyword arguments of a `compute_strain_designs` call.** The two views (a bag of kwargs, or
one `sd_setup` dict) are interchangeable descriptions of the same problem.

Note that the `sd_setup` *stored on a result object* is not byte-identical to the input one: the
orchestrator rebuilds it from the *original* (uncompressed) modules and cost dictionaries at
decompression time (`:606-609`, `:837-840`) so that the record refers to the user's model, not
the internal compressed one (see §13.3.3).

#### 13.3.2 Role 1 — `sd_setup` as INPUT

`compute_strain_designs(model, **kwargs)` lets a caller pass the **entire** configuration as one
`sd_setup=` argument instead of spelling out every parameter (docstring `:75-78`). The handling
is at `compute_strain_designs.py:179-184`:

```python
if SETUP in kwargs:
    if type(kwargs[SETUP]) is str:
        with open(kwargs[SETUP], 'r') as fs:
            kwargs = json.load(fs)          # a path to a JSON file
    else:
        kwargs = kwargs[SETUP]              # an in-memory dict
```

Two accepted forms: the value may be an **in-memory dict**, or a **path to a JSON file** — the
latter is how CNApy stores problems as `.sd` files (docstring `:63-65`), which are then loadable
and re-runnable from Python. Either way the setup becomes the working `kwargs` for the rest of
the function.

**Merge semantics — a correctness caveat.** The code does **not** merge `sd_setup` with the other
keyword arguments; the `else` branch **replaces `kwargs` wholesale** with the setup dict, so any
explicit kwargs passed alongside `sd_setup` (other than `model`, which is a separate positional)
are silently discarded. The docstring states this as a hard rule: *"sd_setup and other arguments
(except for model) must not be used together"* (`:77-78`). So the contract is "all-or-nothing,"
not "defaults-plus-overrides": use *either* individual kwargs *or* one `sd_setup`, never both.
(This is unlike `compute_strain_designs_from_preprocessed`, §13.4.2, whose keyword arguments
genuinely *override* the dumped configuration.)

**Why this exists.** A single `sd_setup` dict/JSON is a **portable, version-controllable,
reproducible problem specification.** It can be committed to a repository, attached to a paper,
diffed across experiments, generated programmatically by a GUI (CNApy), or shipped between
machines — and it re-poses the *exact* same computation with one call. It collapses a
ten-argument invocation into one auditable artifact.

#### 13.3.3 Role 2 — `sd_setup` as OUTPUT / reproducibility record

Every `SDSolutions` stores the setup it was produced under: `self.sd_setup = sd_setup`
(`strainDesignSolutions.py:74`). This is what makes a result **self-describing** — the object
carries not just the answers but the full question. The orchestrator builds this record from the
*original* model/modules/costs right before constructing the solution: it `deepcopy`s the setup
returned by the MILP layer and overwrites the module/cost keys with the uncompressed originals
(`compute_strain_designs.py:606-609` in the normal path, `:837-840` in the from-preprocessed
path, and `:570-573` in the dump early-return), adding `GKOCOST`/`GKICOST` when in gene mode. The
`deepcopy` is deliberate: the record must be an immutable snapshot, decoupled from any later
mutation of the live cost dictionaries.

#### 13.3.4 Downstream uses — why a self-contained setup pays off

Carrying the full setup on the result is what lets the object be re-processed with **no reference
to the original call site**:

- **Re-costing.** `_compute_costs_and_bounds` (`strainDesignSolutions.py:204-243`) reads
  `KOCOST`/`KICOST`/`GKOCOST`/`GKICOST`/`REGCOST` *straight out of `sd_setup`* to total each
  design's cost. Because the cost model lives in the record, `sd_cost` can be recomputed for any
  (e.g. lazily expanded, §13.2.4) design without the caller re-supplying the cost dictionaries —
  `expand_group` (`:493-494`) does exactly this, passing `self.sd_setup` back into
  `_compute_costs_and_bounds`.
- **Re-expansion.** The same setup drives on-demand decompression of compressed groups; the
  gene-vs-reaction branch and the cost lookups both key off it.
- **Re-running.** Because `sd_setup` *is* a valid `compute_strain_designs` kwarg bundle (§13.3.1),
  `compute_strain_designs(model, sd_setup=sols.sd_setup)` re-runs the identical problem. Combined
  with the embedded model snapshot (§13.2.5), a saved solution file is a complete, machine-portable
  capsule: model + problem + answers, re-runnable and re-analysable on another host without the
  original script.

This is the deeper reason the save/load machinery (§13.2.5) embeds a model snapshot: the setup
already pins *everything except the model object*, so embedding the model closes the last gap and
makes the pickle a fully self-contained, reproducible record.

### 13.4 The preprocessed-dump workflow

The single most expensive part of a strain-design run is **preprocessing**, not the MILP solve:
the compression passes and — dominantly — the blocked/irreversible FVA. On the canonical
iML1515 gene-MCS problem the preprocessing FVA alone is ~117 s, while MILP *construction* is
~4 s (Ch 11). If you want to sweep the MILP solve across many configurations — different random
seeds, different solvers, different solution approaches, different pre-FVA bound settings — you
should pay the ~117 s **once** and replay the cheap part. That is exactly what `dump_preprocessed`
+ `compute_strain_designs_from_preprocessed` provide. This is the workhorse of the benchmarking
harness.

#### 13.4.1 Dumping: `dump_preprocessed`

`dump_preprocessed` is a kwarg to `compute_strain_designs` (whitelisted at
`compute_strain_designs.py:176`); its value is a path. The orchestrator runs the *entire*
preprocessing pipeline normally — compression #1/#2, GPR integration, all three FVA phases,
size-1 MCS extraction, essential-reaction removal, and MILP-kwarg assembly — and then, just
before it would solve the MILP (`:534-592`), if `dump_preprocessed` is set it pickles a
dictionary and returns early (with any size-1 MCS already found, but *without* running the
MILP). The dumped dict (`:540-562`) contains:

| Key | What it is | Why it's needed on replay |
|-----|-----------|----------------------------|
| `cmp_model` | the **compressed, GPR-extended** cobra model (exact-rational bounds) | the model the MILP is built on — the expensive artifact |
| `sd_modules` | the modules **remapped to compressed reaction space** | `SDMILP` construction consumes these |
| `kwargs_milp` | solver, `max_cost`, `M`, `seed`, threads, **compressed** ko/ki costs, `essential_kis` | the exact MILP-build arguments |
| `kwargs_computation` | `max_solutions`, `time_limit`, `show_no_ki` | passed to `compute`/`compute_optimal`/`enumerate` |
| `solution_approach` | `'any'`/`'best'`/`'populate'` | which solve method to call |
| `cmp_mapReac` | the compression map | needed to decompress the eventual solutions |
| `uncmp_ko_cost`, `uncmp_ki_cost`, `uncmp_reg_cost` | uncompressed cost dicts | decompression + `filter_sd_maxcost` |
| `orig_model`, `orig_sd_modules`, `orig_*_cost`, `orig_g*_cost` | the pristine originals | building `sd_setup` and the returned `SDSolutions` |
| `gene_kos` | bool flag | selects gene vs reaction decompression |
| `max_cost`, `cmp_size1_mcs` | cost cap and the size-1 MCS found in preprocessing | decompression/filtering |
| `pre_fva_bounds` | `{reac_id: (lb, ub)}` **before** the blocked/irrevers FVA | lets you *re-run* the bound-relaxation with a different config, or study its effect, without recompressing |

`pre_fva_bounds` (captured at `:449`, immediately before `bound_blocked_or_irrevers_fva`) is the
key enabler of **bound-configuration experiments**: the compressed model is snapshotted with its
bounds *as they were before* the redundant-bound relaxation, so a downstream experiment can
apply a different bound policy to the already-compressed model rather than re-deriving the whole
compression. The dump thus amortizes not just the FVA but the entire compression + GPR chain.

On dump the function logs a copy-pasteable resume line and returns an `SDSolutions` holding only
the size-1 MCS (or infeasible/empty), with `compressed_sd`/`compression_map`/`group_map` and
`_cmp_model` populated (`:568-592`).

#### 13.4.2 Replaying: `compute_strain_designs_from_preprocessed`

`compute_strain_designs_from_preprocessed(dump, seed=None, solver=None, solution_approach=None,
max_solutions=None, time_limit=None)` (`:759-851`) is the cheap replay. Its signature *is* the
sweep interface: every keyword is an **override** applied on top of the dumped configuration.

- `dump` may be a **path** (unpickled) or the **dict itself** (`:776-781`) — the latter lets you
  unpickle once, mutate the dict in a loop (e.g. rewrite `cmp_model` bounds using
  `pre_fva_bounds`, or swap `sd_modules`), and feed each variant in without touching disk.
- Overrides (`:803-813`): `seed` → `kwargs_milp[SEED]`; `solver` →
  `kwargs_milp[SOLVER]` (via `select_solver`); `max_solutions`/`time_limit` →
  `kwargs_computation`; `solution_approach` replaces the dumped approach.
- The compressed model was pickled while its LP/solver was suppressed (its solver is a stub), so
  the replay re-enters `suppress_lp_context(cmp_model)` (`:817-818`) before building the
  `SDMILP`, so that `SDMILP` can safely touch variables without triggering a solver build.
- It then rebuilds the MILP (`SDMILP(cmp_model, sd_modules, **kwargs_milp)`, `:824`), solves via
  the chosen approach, and — crucially — runs the **identical** `_decompress_solutions` path
  (`:842-845`) as the normal orchestrator, so the returned `SDSolutions` (lazy expansion, costs,
  bounds, gene translation, `_cmp_model`) is indistinguishable from one produced end-to-end.

#### 13.4.3 The developer workflow

The typical benchmarking loop:

```python
from straindesign import (compute_strain_designs,
                          compute_strain_designs_from_preprocessed)

# 1. Pay preprocessing ONCE (~117 s on iML1515). Returns early; writes the dump.
compute_strain_designs(model, sd_modules=[suppress],
                       gene_kos=True, max_cost=3,
                       solution_approach='populate',
                       dump_preprocessed='iml1515_gmcs.pkl')

# 2. Sweep the cheap MILP solve — e.g. a seed sweep for solver-variance study:
results = []
for s in range(10):
    sol = compute_strain_designs_from_preprocessed('iml1515_gmcs.pkl', seed=s)
    results.append(sol)

# 3. Or a solver comparison (the CPLEX-vs-Gurobi story, Ch 11):
gu = compute_strain_designs_from_preprocessed('iml1515_gmcs.pkl', solver='gurobi')
cp = compute_strain_designs_from_preprocessed('iml1515_gmcs.pkl', solver='cplex')

# 4. Or a bound-config experiment using the in-memory dict form:
import pickle
d = pickle.load(open('iml1515_gmcs.pkl', 'rb'))
for cfg in bound_configs:
    apply_bounds(d['cmp_model'], d['pre_fva_bounds'], cfg)   # mutate compressed model
    results.append(compute_strain_designs_from_preprocessed(d))   # pass the dict
```

Because each replay reuses the same compressed model, module remapping and cost translation, the
*only* variable across runs is the MILP itself — which is precisely the isolation a benchmark
wants. And because the returned `SDSolutions` objects are merge-compatible (same model, same
compression map), a seed or solver sweep can be folded into a single deduplicated solution set
with `sum(results, results[0])`-style `__iadd__` (13.2.5). This is the object-level plumbing
that makes the benchmarking harness (Ch 11) fast and reproducible.


## 14. The solver-interface layer (`MILP_LP` + backends)

Every LP and MILP that `straindesign` ever solves — the three preprocessing FVA sweeps, the
size-1 MCS probes, the bounding LPs that compute big-M values, and the central strain-design
MILP with its integer-cut enumeration — passes through a single class, `MILP_LP` in
`solver_interface.py`. `MILP_LP` is a thin, uniform façade over four numerically and API-wise
very different solvers (CPLEX, Gurobi, SCIP/SoPlex, GLPK). This chapter is about the physical
handoff: how the abstract problem `(c, A_ineq, b_ineq, A_eq, b_eq, lb, ub, vtype, indic_constr, M)`
built upstream (Ch 7) becomes a live solver object, how `solve` / `slim_solve` / `populate` map onto
each backend's very different notion of "solve," how indicator constraints are handed over natively
or reduced to big-M, how each solver's status codes are collapsed into one canonical vocabulary,
and where — physically — the ~4.4× CPLEX-vs-Gurobi runtime gap on the canonical iML1515 gene-MCS
benchmark lives.

Boundaries: **Ch 7** owns the *decision* of which continuous rows get a big-M encoding versus a
native indicator constraint (the `link_z` fork) and the mathematics of a valid/tight `M`. **Ch 8**
owns the *solve loop* — the ANY / BEST / POPULATE objective setups and the integer-cut enumeration
that repeatedly calls the methods described here. This chapter owns only the layer in between: the
abstraction and the four backend translations.

### 14.1 Why an abstraction layer exists

The four solvers do not agree on almost anything at the API level:

- **Problem construction.** CPLEX wants triplets fed into `variables.add` / `linear_constraints.add` /
  `set_coefficients`; Gurobi wants an `MVar` and matrix constraints (`addMVar`, `addMConstr`);
  pyscipopt wants variables and `Expr` objects assembled term by term; GLPK (via `swiglpk`) wants
  raw C arrays with **1-based** indexing.
- **Infinity.** `numpy.inf` must be rewritten to `cplex.infinity`, `gurobipy.GRB.INFINITY`,
  `SCIP.infinity()`, or GLPK's free-bound sentinels — each different.
- **Indicator constraints.** CPLEX and Gurobi support them natively (with opposite conventions for
  how you say "active when the binary is 0"); SCIP supports only the *indicator = 1* case natively
  and needs an auxiliary variable for *indicator = 0*; GLPK has no concept of them at all.
- **Status codes.** CPLEX returns small integers whose meaning depends on whether the problem is an
  LP or a MIP (e.g. unbounded is `2`/`4` for LP but `118`/`119` for MIP); Gurobi returns its own
  enum; SCIP returns strings (`'optimal'`, `'timelimit'`, `'unknown'`, …); GLPK returns yet another
  integer set (`GLP_OPT`, `GLP_NOFEAS`, …).
- **The solution pool.** CPLEX and Gurobi have native pools; SCIP and GLPK have none, so `populate`
  must be *emulated* by an outer solve-and-exclude loop.

Rather than sprinkle solver-specific branches through `SDMILP`, the whole pipeline is written once
against the `MILP_LP` API — `solve`, `slim_solve`, `populate`, `set_objective(_idx)`, `set_ub`,
`set_time_limit`, `add_ineq_constraints`, `set_ineq_constraint`, `set_lp_method`, `get/set_basis` —
and each backend implements exactly that surface with identical semantics. The invariant that makes
this correct is that **every backend presents the same canonical minimization problem** and returns
the **same canonical statuses**, regardless of how its underlying solver phrases them. `SDMILP`
itself is defined as `class SDMILP(SDProblem, MILP_LP)` (`strainDesignMILP.py:31`) and simply calls
`MILP_LP.__init__` with the matrices `SDProblem` assembled (`strainDesignMILP.py:92`), so the strain-
design MILP *is* an `MILP_LP` — the abstraction is not a wrapper the caller holds, it is a base class
the problem inherits.

### 14.2 The canonical problem and the `MILP_LP` constructor

`MILP_LP` accepts one problem shape (`solver_interface.py:36`):

```
minimize   cᵀx
subject to A_ineq · x ≤ b_ineq
           A_eq   · x = b_eq
           lb ≤ x ≤ ub
           x_i ∈ {C, B, I}                       (continuous / binary / integer)
           indicator constraints:
           x_j = [0|1]  →  a · x  [≤|=|≥]  b
```

with `A_ineq, A_eq ∈ scipy.sparse` of width `n = #variables`, `c, lb, ub, vtype` of length `n`, and
`indic_constr` an `IndicatorConstraints` object (Section 14.4). The sense is **always
minimization**; the enumeration and dualization layers arrange their objectives accordingly (Ch 8).

The constructor (`solver_interface.py:103`) does four jobs before touching a solver:

1. **Keyword plumbing and defaults.** It accepts exactly the keys
   `{c, A_ineq, b_ineq, A_eq, b_eq, lb, ub, vtype, indic_constr, M, solver, skip_checks, tlim, seed,
   milp_threads}`; any other key raises. Missing pieces are defaulted so a caller may pass only a
   constraint matrix: `c → 0`, empty `A_eq`/`A_ineq` become `(0, n)` sparse matrices, `lb → -inf`,
   `ub → +inf`, `vtype → 'C'·n` (`solver_interface.py:134–150`). Infinities/NaNs flow through here as
   `numpy` values; each backend rewrites them to its own sentinel.

2. **Solver selection.** If no `solver` is given, the first entry of the module-level `avail_solvers`
   set is used (`solver_interface.py:118–120`). `avail_solvers` is populated at import in
   `__init__.py` in the order GLPK, CPLEX, Gurobi, SCIP as each import succeeds; because it is a
   `set`, "first" is not a guaranteed priority order — for reproducible backend choice the caller
   should pass `solver=` explicitly (the orchestrator does). An explicit solver that is not installed
   raises immediately (`solver_interface.py:124`).

3. **Dimension checks** (unless `skip_checks=True`). Row counts of `A_ineq`/`b_ineq` and
   `A_eq`/`b_eq` must match, all widths must equal `n`, and the indicator block's dimensions must be
   internally consistent (`solver_interface.py:152–167`). These checks are the single most useful
   guard against a malformed dualization silently producing a wrong-shaped MILP; `skip_checks=True`
   is used only on hot paths where the shape is known (e.g. the `verify_sd` sub-LPs).

4. **Type casting and the big-M warning.** All matrices are cast to `float`; then, if the solver is
   *not* GLPK but a finite `M` was supplied alongside indicator constraints, a warning fires that
   `M` will be ignored (`solver_interface.py:179–181`) — because only GLPK consumes `M`. GLPK with
   `milp_threads` set raises, since GLPK is single-threaded (`solver_interface.py:182–183`).

Then it instantiates the backend (`solver_interface.py:185–205`), passing the full tuple plus
`seed` and `milp_threads` (and, for GLPK only, `M`). SCIP is special-cased: if the problem is a pure
LP (`all vtype == 'C'` and no indicator constraints) it routes to the **SoPlex** LP object `SCIP_LP`
and returns early; otherwise to the MILP object `SCIP_MILP` (`solver_interface.py:193–201`). Finally
it applies the time limit (`inf` if none), floored at 1 ms (Section 14.6).

Note that `MILP_LP` keeps its *own* copies of `c, A_ineq, b_ineq, …` in sync with the backend: e.g.
`add_ineq_constraints` both `vstack`s onto `self.A_ineq` *and* forwards to `self.backend`
(`solver_interface.py:287–306`). This shadow copy is what lets the enumeration layer read back the
current constraint system (for integer cuts) without a solver round-trip.

### 14.3 The three solve entry points and result normalization

`MILP_LP` exposes three ways to solve, each forwarding to the backend and each with a distinct
contract used by different parts of Ch 8's loop:

- **`solve() → (x, opt, status)`** (`solver_interface.py:211`). The full solve: returns the solution
  vector, objective value, and canonical status. Crucially, `MILP_LP.solve` post-processes the
  backend's raw `x`: when a solution exists (status not in `{INFEASIBLE, UNBOUNDED, TIME_LIMIT}`) it
  **rounds integer/binary variables to the nearest integer** and casts, leaving continuous variables
  untouched (`solver_interface.py:223–224`). This is where solver integrality tolerances (a `z_j`
  coming back as `0.9999999997`) are cleaned into exact `0/1` so that the downstream integer-cut math
  and cost accounting are exact.

- **`slim_solve() → opt`** (`solver_interface.py:227`). Returns only the optimal objective value, no
  solution vector. This is the workhorse of preprocessing: FVA minimizes/maximizes each flux, and the
  MILP verification LPs (`strainDesignMILP.py:286`, `valid[i] = not isnan(lp.slim_solve())`) only ask
  "is this feasible / what is the bound," never needing `x`. Skipping solution-vector extraction
  matters because, across genome-scale FVA, that extraction cost is paid thousands of times.

- **`populate(n) → (X, opt, status)`** (`solver_interface.py:241`). Returns a *list* of solution
  vectors — the solution pool — used by POPULATE enumeration to harvest many equally-optimal
  designs in one solver invocation. Only CPLEX and Gurobi implement this natively; SCIP and GLPK
  emulate it (Section 14.7).

The remaining API is manipulation used between solves by the enumeration loop: `set_objective` /
`set_objective_idx` (swap the objective vector, e.g. ANY's zero objective vs BEST's cost objective),
`set_ub` (fix a `z_j` by dropping its upper bound to 0), `add_ineq_constraints` (append an integer
cut), `set_ineq_constraint` (rewrite a row in place), and `set_time_limit` (Section 14.6). Warm-start
support (`get_basis` / `set_basis`, `set_lp_method`) exists for LP-heavy phases; barrier is
unavailable on GLPK and SoPlex and degrades gracefully (Section 14.6).

### 14.4 Indicator constraints vs big-M, per backend

An `IndicatorConstraints` object (`indicatorConstraints.py:22`) stores a *batch* of implications
`x_{binv[i]} = indicval[i] → A[i]·x [sense[i]] b[i]`, with `binv` the indicator variable indices,
`A` a sparse matrix (one row per constraint), `b` the right-hand sides, `sense ∈ {'L','E','G'}`, and
`indicval ∈ {0,1}`. This is a solver-neutral container; each backend translates it.

Recall the Ch 7 result stated as given in CONTEXT: under the default `M = inf`, `link_z` emits the
**SUPPRESS Farkas-dual rows as indicator constraints** (their fluxes are unbounded, so no finite `M`
exists) and the **PROTECT finite-flux primal rows as big-M rows already baked into `A_ineq`**. This
split is emergent from bound structure, not a per-module switch. Consequently, by the time a problem
reaches this layer, the big-M rows are *ordinary inequality rows* — no backend does anything special
with them — and the `indic_constr` block carries only the genuinely indicator-encoded implications.
The one exception is GLPK, which cannot represent indicators and must convert that block to big-M
here, using the `M` value the abstraction passed it.

**CPLEX** (`cplex_interface.py:132–141`). The batch is reshaped to CPLEX's format — each row becomes
`[[col indices],[coeffs]]` — and handed to `self.indicator_constraints.add_batch` with
`indvar=binv`, `sense`, `rhs=b`, and `complemented = 1 - indicval`. CPLEX's native convention is
"constraint active when `indvar = 1`," so an `indicval = 0` implication is passed as
`complemented = 1`. This is a single native call; CPLEX handles the linkage internally with no big-M
and no auxiliary variables, giving a tighter LP relaxation.

**Gurobi** (`gurobi_interface.py:136–146`). Each row `i` is turned into a linear expression
`Σ vals·x[cols]` and registered with
`addGenConstrIndicator(x[binv[i]], bool(indicval[i]), lhs, sense, b[i])`, where the sense is `'='`
for `'E'` and `'<'` otherwise. Gurobi's API takes the *active value* directly as its second
argument, so no complementation arithmetic is needed. The constructor records
`self._has_indicator_constr` because Gurobi 13's presolve has a known bug with indicators that the
solve path must guard against (Section 14.8).

**SCIP** (`scip_interface.py:139–159`). SCIP supports indicator constraints (`addConsIndicator`) but
with two limitations the code works around:
- *Only `indicval = 1` is native.* For an `indicval = 0` implication the interface adds an auxiliary
  binary `z` and an XOR/complement equality `x_{binv} + z = 1`, then indicates on `z`
  (`scip_interface.py:142–146`). So "active when the original binary is 0" becomes "active when the
  fresh `z` is 1."
- *Only `≤` (and `=` split) senses.* An `'E'` row is expanded to two rows `A·x ≤ b` and `-A·x ≤ -b`
  (`scip_interface.py:149–154`); each resulting `≤` row is added as an `addConsIndicator` with the
  chosen binary. Thus SCIP's handling is *native indicator constraints*, just with more
  bookkeeping — it does **not** fall back to big-M.

**GLPK** (`glpk_interface.py:150–179`). GLPK has no indicator support, so this is the one backend
where the `IndicatorConstraints` block is reduced to **big-M rows** at construction, using the `M`
passed down (default `1000` if none — `glpk_interface.py:154–155`). The reduction implements exactly
the linearization documented in `indicatorConstraints.py:29–33`:
- an `'E'` row is first split into `A·x ≤ b` and `-A·x ≤ -b`;
- if the constraint is active when the binary is **1** (`indicval = 1`): set the binary's column
  coefficient to `+M` and add `M` to the RHS, giving `A·x + M·z ≤ b + M` — inactive (slack `M`) when
  `z = 0`, and the original `A·x ≤ b` when `z = 1`;
- if active when the binary is **0** (`indicval = 0`): set the column coefficient to `-M`, giving
  `A·x − M·z ≤ b` — the original constraint at `z = 0`, relaxed by `M` at `z = 1`
  (`glpk_interface.py:173–177`).

These synthesized rows are stacked below the ordinary `A_ineq`/`A_eq` rows and loaded as `GLP_UP`
(upper-bounded) rows (`glpk_interface.py:181–197`). Two warnings fire announcing the reduction and
the `M` used (`glpk_interface.py:156–157`). Because a loose `M` inflates the LP relaxation and
courts the numerical trouble genome-scale MCS problems are already prone to (Ch 11), GLPK is the
weakest backend for large strain-design MILPs and is best reserved for validation on small models.

The upshot connects directly to Ch 7's fork: on CPLEX / Gurobi / SCIP the unbounded SUPPRESS rows
stay as *native indicators* (tight relaxation, better numerics); on GLPK they, plus any other
indicator rows, are forced into big-M with all the relaxation-quality and conditioning costs that
entails.

### 14.5 Status-code translation to the canonical vocabulary

The canonical statuses (defined in `names.py`) are the strings `OPTIMAL='optimal'`,
`INFEASIBLE='infeasible'`, `UNBOUNDED='unbounded'`, `TIME_LIMIT='time_limit'`,
`TIME_LIMIT_W_SOL='time_limit_w_sols'`, and `ERROR='error'`. Every backend's `solve`/`slim_solve`/
`populate` maps its native status into these. The mapping is not mechanical — several native codes
carry information the abstraction must preserve or repair:

| Canonical | CPLEX (`solve`) | Gurobi (`solve`) | SCIP (`solve`) | GLPK (`solve`) |
|---|---|---|---|---|
| `OPTIMAL` | `1,101,102,115,128,129,130` | `OPTIMAL, SOLUTION_LIMIT, SUBOPTIMAL, USER_OBJ_LIMIT` | `'optimal'` | `GLP_OPT, GLP_FEAS` |
| `INFEASIBLE` | `3,103` | `INFEASIBLE` (after DualReductions re-solve) | `'infeasible'` | `GLP_INFEAS, GLP_NOFEAS` |
| `UNBOUNDED` | `2,4,118,119` | `UNBOUNDED`/`INF_OR_UNBD` (after re-solve) | `'inforunbd','unbounded'` | `GLP_UNBND, GLP_UNDEF` |
| `TIME_LIMIT` | `108,114` | `TIME_LIMIT/INTERRUPTED`, no incumbent | `'timelimit','userinterrupt'`, no sol | `bool_tlim & GLP_UNDEF` |
| `TIME_LIMIT_W_SOL` | `11,13,107,113` **and 5,6** | `TIME_LIMIT/INTERRUPTED` with incumbent; NUMERIC salvage | `'timelimit','userinterrupt'` with sol; `'unknown'` salvage | `bool_tlim & GLP_FEAS` |
| `ERROR` | `CplexError` (except code 1217) | `GurobiError` in `populate` | bare `except` | bare `except` |

References: CPLEX `cplex_interface.py:188–219`; Gurobi `gurobi_interface.py:205–258`; SCIP
`scip_interface.py:211–244`; GLPK `glpk_interface.py:231–256`.

Three subtleties are worth calling out because they are correctness-load-bearing:

1. **Gurobi's `INF_OR_UNBD` disambiguation.** Gurobi's presolve frequently returns the fused status
   "infeasible *or* unbounded." The interface cannot let that ambiguity leak into the canonical
   vocabulary, so on `{INF_OR_UNBD, UNBOUNDED, INFEASIBLE}` it re-solves once with
   `DualReductions = 0` (which disables the presolve reduction responsible for the fusion), then
   reads the now-unambiguous status (`gurobi_interface.py:218–232`). This costs an extra solve but
   guarantees the enumeration loop is told *exactly* whether the current knockout set made the
   SUPPRESS behavior infeasible (the whole point of an MCS) or accidentally left the problem
   unbounded.

2. **CPLEX LP vs MIP code overloading.** Because CPLEX reuses `2`/`4` for LP-unbounded and
   `118`/`119` for MIP-unbounded (and similarly for other states), the interface lists *all* the
   variants in each branch (`cplex_interface.py:204`, `246`), so the same `MILP_LP.slim_solve()`
   works whether the object was built as an LP (preprocessing) or a MIP (strain design).

3. **Rounding at the boundary.** GLPK's `solve` rounds both `x` and `opt` to 12 decimals
   (`glpk_interface.py:254–255`) as a workaround for GLPK returning values like `-1e-15`; combined
   with `MILP_LP.solve`'s integer rounding, binary variables come back exactly `0/1`.

Anything a backend cannot classify raises `"Case not yet handled"`, deliberately loud so that a new
solver-version status code is caught in testing rather than silently mismapped.

### 14.6 Parameters: seed, threads, tolerances, time limit, working memory

All four constructors set tolerances *tighter than solver defaults*, because MCS/Farkas MILPs are
sensitive to integrality slop (a spuriously "knocked-in" reaction from a `z_j = 1e-6`):

| Parameter | CPLEX | Gurobi | SCIP (MILP) | GLPK |
|---|---|---|---|---|
| optimality/dual tol | `simplex.tolerances.optimality = 1e-9` | `OptimalityTol = 1e-9` | (SoPlex LP: `DUALFEASTOL = 1e-9`) | — |
| feasibility tol | `simplex.tolerances.feasibility = 1e-9` | `FeasibilityTol = 1e-9` | (SoPlex LP: `FEASTOL = 1e-9`) | `tol_bnd = 1e-9` |
| integrality tol | `mip.tolerances.integrality = 0.0` | `IntFeasTol = 1e-9` (0 disallowed) | emphasis-default | `tol_int = 1e-12` |
| seed | `randomseed` | `Seed` | `randomization/randomseedshift` | *not supported* |
| threads | `threads` (if set) | `Threads` (if set) | `parallel/{max,min}nthreads` | single-threaded |
| output silenced | log/error/warn/results streams → `StringIO` | `OutputFlag = 0` | `display/verblevel = 0` | `msg_lev = 0` |

References: CPLEX `cplex_interface.py:147–166`; Gurobi `gurobi_interface.py:149–164`; SCIP
`scip_interface.py:161–185`, SoPlex `scip_interface.py:543–544`; GLPK `glpk_interface.py:204–213`.

- **Seed.** When the caller does not supply one, each MILP backend draws a fresh seed in `[0, 2¹⁶)`
  and sets it (`cplex_interface.py:155–158`, `gurobi_interface.py:153–157`,
  `scip_interface.py:164–168`). This makes a single run reproducible given a fixed seed but means
  two default runs explore the branch-and-bound tree differently — relevant when comparing wall-times
  (Ch 8/11 pin the seed for benchmarking). **GLPK cannot set a seed** — the code notes swiglpk
  exposes no such hook (`glpk_interface.py:215–216`) — so GLPK enumeration order is not seed-tunable.

- **Threads.** Only set if `milp_threads` is explicitly passed; otherwise each solver uses its own
  default (typically all cores). SCIP sets *both* min and max thread counts. GLPK rejects the option
  upstream in `MILP_LP.__init__`.

- **Working memory (CPLEX only).** For MIPs, CPLEX's `workmem` is set to **75 % of total physical
  RAM** (`cplex_interface.py:151–152`, via `psutil.virtual_memory()`), so that CPLEX keeps its
  branch-and-cut node file and cut pool in memory rather than spilling to disk on the large iML1515
  problems. No equivalent knob is set for the others.

- **Time limit.** `MILP_LP.set_time_limit(t)` floors `t` at `1e-3` s before dispatch
  (`solver_interface.py:283`). The reason is documented in-line: the enumeration loop computes the
  remaining budget as `endtime − time.time()` right after a `> 0` guard, and a scheduling hiccup can
  make it zero or slightly negative; a 1 ms floor keeps the value valid because **Gurobi rejects a
  negative `TimeLimit`** and, critically, **GLPK treats `tm_lim == 0` as "no limit"** (an unbounded
  run). Each backend then clamps against its own maximum: CPLEX maps `inf → timelimit.max()`
  (`cplex_interface.py:369–374`); GLPK stores milliseconds and caps at its initial `tm_lim`
  (`glpk_interface.py:423–432`); SCIP caps at its `limits/time` max (`scip_interface.py:394–399`);
  `SCIP_LP.set_time_limit` is a **no-op** — SoPlex LP solves are not time-limited
  (`scip_interface.py:738–740`).

- **LP method / warm start.** `set_lp_method` maps the neutral constants
  `LP_METHOD_{AUTO,PRIMAL,DUAL,BARRIER}` to each solver's code
  (`cplex_interface.py:327–336`, `gurobi_interface.py:376–384`, `scip_interface.py:367–378`,
  `glpk_interface.py:379–392`). Barrier is unavailable on GLPK (falls back to dual, with a warning)
  and on SoPlex (`SCIP_LP.set_lp_method` is a no-op). Basis extraction (`get_basis`/`set_basis`) is
  supported on CPLEX, Gurobi, and GLPK for warm-starting LP re-solves; `SCIP_MILP` refuses it (the
  LP basis is discarded after a MIP solve — `scip_interface.py:386–392`), and `SCIP_LP`
  *reconstructs* an approximate basis from `getBasisInds` plus solution values
  (`scip_interface.py:683–729`).

**No MIP optimality gap is set anywhere.** Neither `mip.tolerances.mipgap` (CPLEX) nor `MIPGap`
(Gurobi) appears in these constructors, so both solvers run at their **default relative gap of
`1e-4`**. For MCS enumeration this is benign for correctness — every design that survives an integer
cut is re-checked — but it does mean "optimal" is optimal to `1e-4`, which is why the integer
variables are hard-rounded on the way out (Section 14.3).

### 14.7 The solution pool: native (CPLEX, Gurobi) vs emulated (SCIP, GLPK)

`populate(n)` is where POPULATE enumeration harvests many optimal-cost designs at once instead of
re-solving after each integer cut. The two commercial solvers do it natively; the two open-source
ones fake it.

**CPLEX** (`cplex_interface.py:260–311`). `populate` sets the pool capacity and the populate limit
to `n` (or their maxima for `n = inf`), calls `populate_solution_pool()`, translates the status, and
harvests every pool member via `self.solution.pool.get_values(i)`. The pool *behaviour* is governed
by three parameters fixed in the constructor (`cplex_interface.py:161–163`):
`mip.pool.absgap = 0.0`, `mip.pool.relgap = 0.0`, and `mip.pool.intensity = 4`. Absgap/relgap `= 0`
mean **only solutions matching the optimal objective are retained** — exactly what MCS enumeration
wants (all minimum-cost designs, nothing worse); `intensity = 4` is CPLEX's most aggressive pool-
generation effort. **These three parameters are inert for a single `solve()`**: `solve` calls
`super().solve()`, not `populate_solution_pool()`, so during ANY/BEST the pool is never populated
and stays empty (verified; CONTEXT). They matter only inside `populate`. This is not a performance
bug — it is simply that pool configuration only takes effect on the pool-generating call.

**Gurobi** (`gurobi_interface.py:289–342`). `populate` sets `PoolSolutions = n` (or `MAXINT`),
`PoolSearchMode = 2` ("find the n best solutions, systematically"), and raises `NumericFocus = 2`
for the pool sweep, solves, then **restores** `PoolSearchMode = 0` and `NumericFocus = 0` so an
ensuing single `solve()` is not slowed by pool search. The equivalent of CPLEX's absgap/relgap `= 0`
is set in the *constructor*: `PoolGap = 1e-9` and `PoolGapAbs = 1e-9` (`gurobi_interface.py:162–163`),
i.e. keep essentially only optimal-objective solutions. Harvesting (`getSolutions`,
`gurobi_interface.py:488–496`) iterates the pool via `SolutionNumber` and — this is the important
filter — **keeps a pool member only if its `PoolObjVal == ObjVal`**, dropping any non-optimal-cost
solution Gurobi may have parked in the pool. Without this filter the pool could return designs above
the minimum cost.

**SCIP** (`scip_interface.py:282–336`) and **GLPK** (`glpk_interface.py:291–349`) have **no native
pool**, so `populate` is *emulated* by a high-level loop that reproduces the pool semantics:

1. solve to optimality; keep `x`, record `min_cx`;
2. **pin the objective to optimality** by adding the row `cᵀx ≤ min_cx` — so every further solution
   has the same (minimum) cost;
3. add an **exclusion (integer-cut) constraint** on the current solution's binaries so it cannot
   recur;
4. loop: solve, and if still optimal, exclude and append — until the problem becomes infeasible
   (no more min-cost designs) or the time budget runs out;
5. **tear down** the temporary rows by freeing their RHS to `+∞` (SCIP `chgRhs(..., None)`,
   `scip_interface.py:326–330`; GLPK `set_ineq_constraint(j, 0, inf)`, `glpk_interface.py:334–338`),
   rather than deleting them (row deletion is "very unstable" in GLPK, per the in-code comment).

Two backend-specific wrinkles: the exclusion constraint's binary set is recomputed each pass, and
**GLPK must treat integer variables as binaries too**, because GLPK silently promotes a binary whose
bounds are pinned to `0` into an integer variable (`glpk_interface.py:555–557`), which would
otherwise be missed by the exclusion. If the loop exits by infeasibility it is relabelled `OPTIMAL`
(the pool was exhausted, not a genuine failure) — `scip_interface.py:323–324`,
`glpk_interface.py:332–333`. SCIP's emulated exclusion uses `addExclusionConstraintIneq`
(`scip_interface.py:478–484`), the classic `Σ_{j∈S} z_j − Σ_{j∉S}(…) ≤ |S| − 1` cut restricted to
the binary variables.

The performance consequence: on SCIP/GLPK, POPULATE pays *one full MILP solve per design found*,
whereas CPLEX/Gurobi amortize many designs into a single branch-and-cut tree. For the 393-design
iML1515 gene-MCS benchmark this is the difference between a viable and an impractical run — another
reason the open-source backends are validation tools, not production engines, for large problems.

### 14.8 Numeric-status robustness

Genome-scale strain-design MILPs are numerically nasty for reasons detailed in Ch 11: the big-M rows
for PROTECT behaviors, the wide dynamic range of stoichiometric coefficients, and the Farkas-dual
normalization all inflate the condition number, so even a "correct" model can drive a solver into
scaled-versus-unscaled disagreement. Historically these states crashed the pipeline; the interface
now *degrades gracefully*, accepting a usable-but-caveated solution with a warning instead of
raising.

- **CPLEX status 5/6 — "optimal / best with unscaled infeasibilities."** These mean CPLEX found a
  solution that is optimal in the scaled problem but shows small infeasibilities when unscaled. Both
  `solve` and `slim_solve` now take the objective value, log a warning recommending tighter bounds,
  and return it as `TIME_LIMIT_W_SOL` (a "usable but not certified-exact" status) rather than
  hitting the `"Case not yet handled"` branch (`cplex_interface.py:209–215`, `250–251`). Downstream,
  `verify_sd` (Ch 8) will re-check the design, so accepting the caveated incumbent is safe.

- **Gurobi status 12 — `NUMERIC`.** On numerical failure, `solve` retries once with the strongest
  setting, `NumericFocus = 3`, restores the previous focus, and then: if now optimal, report
  `OPTIMAL`; else if any incumbent exists (`SolCount > 0`), accept it as `TIME_LIMIT_W_SOL` with a
  warning; else report `TIME_LIMIT` with no solution — never a crash
  (`gurobi_interface.py:233–256`). `slim_solve` mirrors this by returning the incumbent objective if
  one exists, else `nan` (`gurobi_interface.py:282–284`), and `populate` pre-emptively runs the pool
  sweep at `NumericFocus = 2`.

- **Gurobi 13 indicator-presolve bug (error 10005).** A separate robustness layer,
  `_safe_optimize` (`gurobi_interface.py:173–190`), wraps every `optimize()` call: Gurobi 13 can
  raise `GurobiError 10005` ("Unable to retrieve attribute 'ObjBound'") when indicator constraints
  meet presolve. Rather than disabling presolve globally (~1.6× slowdown), it catches *only* that
  error *only when indicators are present* and retries with `Presolve = 0, Crossover = 1`. This is
  why the constructor records `self._has_indicator_constr`.

- **SCIP `'unknown'` and GLPK undefined states** are handled in the same spirit: SCIP salvages an
  incumbent if `getSols()` is non-empty, else reports `TIME_LIMIT` (`scip_interface.py:232–240`,
  `270–271`); GLPK's bare-`except` paths return `ERROR`/`-1` cleanly rather than propagating a
  swiglpk crash (`glpk_interface.py:258–262`), and the LP pre-solve in `solve_MILP_LP` retries with
  presolve on if `glp_simplex` returns `GLP_EFAIL` on a feasible LP (`glpk_interface.py:536–541`).

The common design principle: a numerically caveated but present solution is returned as
`TIME_LIMIT_W_SOL` and left for the outer verification to accept or reject, never crashing the
enumeration mid-run.

### 14.9 Where the CPLEX-vs-Gurobi performance story physically lives

The interface choices in this chapter are the physical substrate of the headline benchmark
(CONTEXT): the canonical iML1515 gene-MCS run (SUPPRESS biomass ≥ 0.001, POPULATE, `max_cost = 3`,
gene KOs) finds **393 MCS** in **Gurobi 280 s vs CPLEX 1241 s (≈ 4.4×)**, with the split
preprocessing FVA ~117 s, MILP build ~4 s, populate ~1101 s. Reading that against the code:

1. **The gap is in `populate`, not construction.** Both backends receive the *same* abstract MILP
   with the *same* native indicator constraints and the *same* default `1e-4` MIP gap; construction
   is ~4 s either way. The ~1101 s populate phase is a single native pool search on each solver, and
   the 4.4× difference is the two solvers' pool-search engines exploring the design space at
   different rates — not a formulation asymmetry this layer introduces. This is why the CPLEX pool
   parameters, though set since 2022, are *not* the culprit: they are inert during `solve` and, in
   `populate`, they configure the pool identically in spirit to Gurobi's `PoolGap`/`PoolSearchMode`.

2. **Per-LP overhead in preprocessing goes through this layer.** The ~117 s of blocked/irreversible
   FVA is thousands of small LPs, each a `slim_solve` on a freshly constructed backend object.
   Gurobi mitigates the per-object cost by sharing **one quiet `Env`** across all models
   (`gurobi_interface.py:31–41`, `_get_quiet_env`) — creating a Gurobi environment per model would
   spin up a licence session each time, which on a node-locked HPC licence is expensive. CPLEX
   constructs a fresh `Cplex()` per object (and sizes `workmem` to 75 % RAM each time). For a run
   that instantiates the interface thousands of times, this fixed per-solve overhead — object
   creation, parameter setting, matrix load — is real and is paid inside `MILP_LP.__init__` and the
   backend constructors, which is exactly why `slim_solve` (no solution-vector extraction) and
   `skip_checks` exist as fast paths.

3. **The abstraction does not tax the hot path with translation.** Matrices are handed to each solver
   in its preferred bulk form (CPLEX `set_coefficients` on COO triplets, Gurobi `addMConstr` on the
   sparse matrix directly, GLPK a single `glp_load_matrix`), so the per-call cost is solver-native
   assembly, not a Python re-encoding loop — with the exception of SCIP, whose term-by-term `Expr`
   assembly (`scip_interface.py:124–136`) is inherently slower and compounds its lack of a native
   pool. This is the mechanical reason SCIP and GLPK, while correct, are validation backends rather
   than the engines behind the benchmark numbers.

For the enumeration-loop mechanics that drive these calls and the deeper benchmark analysis, see
Ch 8 and Ch 11; for the conditioning that provokes the Section 14.8 numeric states, see Ch 11.


## 15. Analysis & exploration API

Everything documented so far serves one endpoint — `compute_strain_designs`. But `straindesign`
also ships a second, smaller surface that has nothing to do with the MILP: a set of standalone
LP-based tools for **inspecting a model or a design after the fact**. You call these directly, on a
`cobra.Model`, to ask "what growth rate does this network support?", "what is the flux range of every
reaction?", "what does the growth-vs-product trade-off look like before and after I knock out these
reactions?", "what is the maximal product yield per mole of substrate?". None of them is invoked
inside the compute pipeline (the preprocessing FVA calls, Ch 5, reach a *different* internal
entry point in `speedy_fva`; here we document the *public* wrappers). They all live in
`lptools.py` — "a collection of functions for the LP-based analysis of metabolic networks"
(`lptools.py:19`) — and are re-exported at package top level (`__init__.py:60`,
`from .lptools import *`), so a user writes `from straindesign import fba, fva, plot_flux_space, yopt`.

This chapter covers six public entry points and their private helpers:

| Function | `lptools.py` line | Returns | Purpose |
|---|---|---|---|
| `fba` | 438 | `cobra.core.Solution` | optimize a linear objective (FBA / pFBA) |
| `fva` | 245 | `pandas.DataFrame` | per-reaction min/max flux ranges |
| `yopt` | 733 | `cobra.core.Solution` | optimize a **ratio** (yield) via linear-fractional programming |
| `plot_flux_space` | 1406 | `(datapoints, triang, plot)` | 2D/3D projection of the flux polytope (production envelope, yield space) |
| `slim_fba_via_cmp` | 617 | `float` | FBA on a compressed model, objective returned in original units |
| `expand_fluxes` | 955 | `dict` | lift a compressed flux vector back to the original reactions |

The mathematics of the underlying LPs (`Sv = 0`, bounds, the flux polytope, FBA/FVA standard forms,
LP duality) is Ch 2; the internal preprocessing use of FVA and the `speedy_fva` acceleration is Ch 5;
the compression map (`reac_map_exp`) and how interventions are decompressed is Ch 3/9. This chapter
does not re-derive those; it documents the **API contract** — signatures, options, return shapes —
and the *new* mathematics that only appears here: the production-envelope scan, the Charnes–Cooper
transform behind `yopt`, and the flux-vector expansion (as opposed to the intervention-set expansion
of Ch 9).

A note that pervades the whole module: the objective sign convention. Every one of these functions
builds its LP through `MILP_LP` (Ch 14), whose `solve()`/`slim_solve()` **minimize** `c·x`. A user
request to *maximize* is therefore serviced by negating `c` and negating the returned optimum. You
will see `c = [-i for i in c]` and `opt_cx = -opt_cx` repeatedly; that is this convention, not a bug.

### 15.1 `fba` — flux balance analysis and its parsimonious variants

#### Contract

```python
sol = fba(model, obj=..., obj_sense='maximize', constraints='...', pfba=0, solver=None)
```

`fba` (`lptools.py:438`) solves

```
 max / min   cᵀ v
 subject to  S v = 0            (steady state, S ∈ ℝ^{m×n})
             A_ineq v ≤ b_ineq  (user constraints)
             A_eq   v = b_eq    (user equality constraints)
             lb ≤ v ≤ ub        (model bounds)
```

and returns a `cobra.core.Solution` carrying `objective_value`, the flux vector `fluxes` (a
`{reaction_id: value}` dict), and a `status` string. The keyword options:

- **`obj`** (`lptools.py:499–504`) — the objective, as a reaction-ID string, a linear-expression
  string (`'2 EX_etoh_e - EX_ac_e'`), or a dict `{'EX_etoh_e': 2, 'EX_ac_e': -1}`. A string is parsed
  by `linexpr2dict` and then densified to a coefficient row by `linexprdict2mat`. If `obj` is omitted,
  the objective is read from the model itself: `c = [i.objective_coefficient for i in model.reactions]`.
- **`obj_sense`** (`lptools.py:506–511`) — `'maximize'`/`'max'` (default, inferred from
  `model.objective_direction` when `obj` is not given) or `'minimize'`/`'min'`. Maximization is
  realized by negating `c`.
- **`constraints`** (`lptools.py:492–497`) — extra linear constraints layered on top of the model, in
  any of the flexible input forms (string, list of strings, list of `[dict, sign, rhs]`). These pass
  through `resolve_gene_constraints` (so a constraint may name a *gene*: `'b0008 = -1'` becomes the
  reaction-level effect of knocking that gene out; see Ch 12) and then `parse_constraints` /
  `lineqlist2mat`, which turn them into `A_ineq, b_ineq, A_eq, b_eq` rows (Ch 12 owns this grammar).
  The stoichiometric block `S v = 0` is stacked on top of the user's equality rows
  (`lptools.py:523–531`).
- **`solver`** (`lptools.py:518–520`) — `'glpk' | 'cplex' | 'gurobi' | 'scip'`, resolved by
  `select_solver` (`lptools.py:45`): a supplied name wins if available; otherwise the solver named in
  the model, then the cobra configuration, then the first available in priority order
  `glpk, cplex, gurobi, scip`.
- **`pfba`** (`lptools.py:513–516`) — the parsimonious-FBA level, discussed next.

#### Unbounded-objective repair

If the primal comes back `UNBOUNDED` (`lptools.py:544–551`) the objective can grow without limit, so
there is no finite optimal `v` to report — but the user still wants a representative flux vector on the
ray. The code re-solves the *opposite* objective to find the extreme in the bounded direction
(`min_cx = num_prob.slim_solve()`) and then **pins the objective to a finite value** with an added
equality row (`add_eq_constraints(c, min_cx)`, or to `-1.0` when the reversed optimum is non-positive)
before solving once more. The returned `objective_value` stays the unbounded signal; the flux vector is
a concrete point on the unbounded face. This is a deliberate usability choice: return *something*
plottable rather than an empty solution.

#### Parsimonious FBA (`pfba`)

Plain FBA fixes the objective value but leaves the rest of `v` under-determined — many flux
distributions achieve the same growth. pFBA picks a biologically motivated representative among them by
adding a secondary objective *after* pinning the primary optimum `opt_cx` (`add_eq_constraints(c_pfba,
[opt_cx])`). Two levels:

- **`pfba=1` — minimize total flux** (`lptools.py:587–602`). Each reaction is split into a forward and
  reverse part `v = v⁺ − v⁻`, `v⁺, v⁻ ≥ 0`, by horizontally stacking `A` with `−A` and building the
  split bounds `lb_pfba = [max(0, l)] + [max(0, -u)]`, `ub_pfba = [max(0, u)] + [max(0, -l)]`. Minimizing
  `∑(v⁺ + v⁻) = ∑|vⱼ|` (objective `[1.0]*2n`) subject to the pinned primal gives the flux vector with the
  smallest 1-norm — the "minimal enzyme usage" distribution. The reported flux is recomposed
  `x = v⁺ − v⁻` (`lptools.py:602`).

- **`pfba=2` — minimize the number of active reactions** (`lptools.py:556–586`). This is a genuine MILP,
  not an LP. First an FVA (with the primal pinned, `kwargs_fva[...].append([{...}, '=', opt_cx])`) finds
  which reactions are *essential* under the optimum — a reaction whose min and max fluxes share a sign
  (`prod(sign(lim)) > 0`) cannot be switched off, so it is excluded from the knockable set
  (`ub_pfba2 = ... 0.0 if prod(sign(lim)) > 0 else 1.0`). Then a binary `y_j` per remaining reaction is
  wired by an **indicator constraint** `y_j = 1 ⇒ v_j = 0` (`IndicatorConstraints([...], A_ic, [0]*numr,
  'E'*numr, [1.0]*numr)`, `lptools.py:570–571`) and `∑ −y_j` is minimized, i.e. the count of forced-zero
  reactions is maximized. The reactions selected zero are fixed to `lb=ub=0` and level-1 pFBA is then run
  on the reduced network. This yields the sparsest-support flux distribution.

#### The compressed-flux hook

`fba` accepts two undocumented-in-signature kwargs, `cmp_map` and `orig_reaction_ids`
(`lptools.py:607–611`). If both are present the resulting flux dict — computed on whatever model was
passed — is run through `expand_fluxes` (§15.5) to yield fluxes keyed by the *original* reaction IDs.
This is the glue that lets you FBA a compressed model but read the answer in original terms.

### 15.2 `fva` — flux variability analysis as a public tool

```python
df = fva(model, constraints='EX_o2_e=0', solver='gurobi',
         compress=None, threads=None, reaction_list=None)
```

`fva` (`lptools.py:245`) determines, for every reaction, the full range `[min vⱼ, max vⱼ]` reachable at
steady state under the model bounds and any extra `constraints`. Mathematically it is `2n` linear
programs — for each reaction `j`, minimize and maximize `vⱼ` over the same polytope FBA uses (Ch 2 gives
the standard form; `fva_legacy`, `lptools.py:285`, is the literal brute-force `2n`-LP reference kept for
debugging). The return is a `pandas.DataFrame` indexed by reaction ID with two columns, `minimum` and
`maximum`.

The public `fva` is a **one-line delegator** (`lptools.py:281–282`):

```python
from straindesign.speedy_fva import speedy_fva
return speedy_fva(model, **kwargs)
```

so its real options are `speedy_fva`'s (`speedy_fva.py:263`), and the acceleration mathematics is Ch 5.
For the API contract, the options a user sets are:

- **`constraints`**, **`solver`** — as for `fba` (gene IDs are resolved, strings parsed to matrix rows).
- **`compress`** (`speedy_fva.py:291`, default `None`) — whether to lump flux-coupled reactions and drop
  conservation rows *before* the scan, then map results back. When `None` it **auto-enables for models
  with ≥ 200 reactions** (`compress = n_original >= 200`). Compression shrinks the LP and makes each of
  the `2n` solves cheaper; the ranges of lumped reactions are recovered from the representative's range.
  This is the same coupled compression as Ch 3 but applied transiently, purely to speed the scan.
- **`reaction_list`** (default `None`) — restrict the scan to a subset of reactions, so you pay for `2k`
  LPs instead of `2n`. Used heavily by the internal pipeline (Ch 5's knockable-scoped FVA) but available
  to users who only care about a handful of reactions.
- **`threads`** (default `None`) — parallel worker count; auto-set to `Configuration().processes` for
  models with ≥ 1000 reactions, else 1 (`speedy_fva.py:305–306`). The multiprocessing machinery
  (`SDPool`, the `fva_worker_*` init/compute helpers at `lptools.py:148–242`, with a GLPK-specific path
  because GLPK cannot solve in a spawned thread) is shared with the legacy implementation.

An infeasible base problem yields a DataFrame of `NaN`s rather than an exception (`fva_legacy`
demonstrates this at `lptools.py:319–324`). Fluxes with `|v| < 1e-11` are snapped to `0.0`
(`lptools.py:384`) to suppress solver noise.

A companion utility, **`remove_redundant_bounds`** (`lptools.py:392`), runs `fva` and then relaxes every
non-binding bound in place: if `fva_min > lb + tol` the lower bound never binds, so it is set to `−inf`;
symmetrically for the upper bound. It returns the FVA DataFrame and mutates the model. This is the
user-facing sibling of the internal `bound_blocked_or_irrevers_fva` (Ch 5) — the same idea (a bound the
network can never reach is redundant and only bloats big-M constants downstream) offered as a
standalone model-cleanup step.

### 15.3 `yopt` — yield optimization by linear-fractional programming

#### What and why

FBA maximizes a *linear* objective. But the quantity a metabolic engineer most wants to push is often a
**ratio**: product formed per substrate consumed,

```
        cᵀ v          (numerator: e.g.  2·EX_etoh_e)
  Y  =  ─────
        dᵀ v          (denominator: e.g. −6·EX_glc__D_e)
```

The coefficients let you express carbon recovery directly — `2 EX_etoh_e / -6 EX_glc__D_e` is
(2 C in ethanol) per (6 C in glucose). A ratio of two linear forms over a polytope is a
**linear-fractional program (LFP)**; it is *not* an LP, but a classical result — the **Charnes–Cooper
transformation** — converts it to one exactly.

```python
sol = yopt(model, obj_num='2 EX_etoh_e', obj_den='-6 EX_glc__D_e',
           obj_sense='maximize', constraints='EX_o2_e=0', solver=None)
```

`yopt` (`lptools.py:733`) requires `obj_num` and `obj_den` (each a string or dict; missing either raises,
`lptools.py:797–811`) and returns a `Solution`.

#### The transform

Charnes–Cooper: to maximize `(cᵀv)/(dᵀv)` over `{v : Av ≤ b}` with `dᵀv > 0`, substitute

```
  t = 1 / (dᵀv) > 0,     y = t · v.
```

Then `dᵀy = t·dᵀv = 1`, the fractional objective becomes the *linear* `cᵀy`, and each original
constraint `Aᵢᵀv ≤ bᵢ` homogenizes to `Aᵢᵀy ≤ bᵢ t`, i.e. `Aᵢᵀy − bᵢ t ≤ 0`. The LFP is thus the LP

```
  max  cᵀ y
  s.t. A_ineq y − b_ineq t ≤ 0
       A_eq   y − b_eq   t = 0
       dᵀ y = 1,   (t free ≥ 0)
```

and the original flux is recovered as `v = y / t`.

The code builds precisely this (`lptools.py:870–890`). It appends one extra column — the scale variable
`t` — to every matrix:

- inequalities: `A_ineq_lfp = [A_ineq | −b_ineq]`, `b_ineq_lfp = 0` (`lptools.py:872–873`);
- equalities: stack `[A_eq | −b_eq]` with the normalization row `[obj_den | 0]`, RHS `[0,…,0, d]`
  (`lptools.py:874–878`);
- objective `c = [−d·obj_num | 0]`, minimized (the `−d` and the outer sign flip restore the requested
  `max`/`min`, `lptools.py:813–817`, `896–897`).

Note the bounds are *first folded into `A_ineq`* as explicit rows (`lptools.py:841–847`) — because in the
homogenized problem a finite bound `vⱼ ≤ uⱼ` must also become `yⱼ − uⱼ t ≤ 0`, so it cannot stay a plain
variable bound. Only finite bounds are added (`isinf` filtered).

After solving, `factor = x[-1]` is `t`, and the reported flux is `x[i] / factor` (`lptools.py:892–895`);
if `t = 0` the flux vector is scalable by any positive factor and `sol.scalable = True` is set.

#### The sign of the denominator — `den_sign`

The transform assumes `dᵀv` keeps a *fixed sign* over the polytope; the normalization `dᵀy = 1` (or
`= −1`) implicitly chooses it. But a user's denominator could be positive on part of the polytope and
negative on another, or fixed at zero. `yopt` handles this robustly (`lptools.py:848–868`) by first
solving for the min and max of `dᵀv`:

- if `min dᵀv < 0`, `−1` is a viable normalization sign;
- if `max dᵀv > 0`, `+1` is viable;
- if neither (the denominator can only be `0`), the yield is undefined — return `INFEASIBLE`.

It then solves the LFP once per attainable sign in `den_sign` and keeps the better optimum
(`lptools.py:877–890`). The documented failure taxonomy (`lptools.py:747–757`) maps directly onto the
return values:

| Situation | Return |
|---|---|
| base model infeasible | `INFEASIBLE`, all-NaN fluxes |
| denominator fixed to 0 | `INFEASIBLE` |
| numerator unbounded while denominator can be 0 | `UNBOUNDED`, flux from fixing the numerator |
| denominator can reach 0 (yield undefined) | flux vector maximizing the numerator, warning logged |

The `UNBOUNDED` branch (`lptools.py:905–951`) is the subtle one: an infinite yield means the numerator
grows while the denominator stays fixed near zero, so the code separately checks whether the numerator is
bounded when `dᵀv = 0` is added as a constraint, and returns a representative flux accordingly.

### 15.4 `plot_flux_space` — production envelopes, yield space, and 3D projections

#### What it visualizes

`plot_flux_space` (`lptools.py:1406`) projects the (high-dimensional) steady-state flux polytope onto 2 or
3 user-chosen axes and draws the resulting shadow. The two canonical uses:

- **Production envelope** — x = growth rate, y = product exchange. The shape shows, for every attainable
  growth rate, the min and max product rate; a strain design "works" when the envelope's lower boundary
  is lifted off zero at high growth (product becomes *coupled* to growth). This is the single most
  common way to validate that a computed MCS actually forces production.
- **Yield-space plot** — x = biomass yield, y = product yield (each a ratio), giving the trade-off in
  per-substrate terms.

```python
plot_flux_space(model, ('BIOMASS_Ecoli_core_w_GAM', 'EX_etoh_e'))              # 2D rate–rate
plot_flux_space(model, [['BIOMASS','-EX_glc_e'], ['EX_etoh_e','-EX_glc_e']])   # 2D yield–yield
plot_flux_space(model, ('r1','r2','r3'))                                       # 3D
```

#### Axis grammar

Each axis in `axes` is either a single linear expression → a **`'rate'`** axis (`len(ax)==1`,
`lptools.py:1545–1546`), or a `[numerator, denominator]` pair → a **`'yield'`** axis (`len(ax)==2`,
`lptools.py:1547–1548`). A rate axis is scanned with `fba`; a yield axis with `yopt` (§15.3). Two or three
axes are allowed (`lptools.py:1528`, else raise). Options mirror the other tools plus:

- **`constraints`, `solver`** — as before, applied to every internal LP so you can plot the envelope of a
  *sub*-model (e.g. add the knockouts of a candidate design as `constraints` and see how the envelope
  changes).
- **`points`** (`lptools.py:1532–1538`, default 40 in 2D, 25 in 3D) — resolution of the *approximate*
  (yield-containing) regions only. For pure rate axes the boundary is traced *exactly* (see below) and
  `points` is ignored.
- **`show`** (default `True`), **`plt_backend`**, **`cmap`** (default `'managua'`, for 3D face colouring).

The return is always `(datapoints, triang, plot1)`: `datapoints` are the computed boundary points,
`triang` a list of index-triples describing how to connect them into a closed surface, and `plot1` the
matplotlib artist. A user who wants a custom figure sets `show=False` and rebuilds from `datapoints`.

#### The mathematics of a production envelope

For each axis the code first finds the overall range by optimizing that axis both ways
(`lptools.py:1552–1553` for rate, `1562–1563` for yield). `val_limits[i] = [min, max]` and the drawing
window `ax_limits` is padded to include the origin (`lptools.py:1568–1569`). An axis whose min ≈ max is
**degenerate**; `_detect_degeneracy` (`lptools.py:1014`) classifies the projection as `point`, `line`,
`plane`, or `full` by counting degenerate axes, and each class has its own cheap drawing path
(`lptools.py:1598–1634`, `1752–1820`) rather than a wasted full scan.

For the non-degenerate 2D case the boundary is traced by one of two algorithms:

- **Rate–rate → exact convex polygon** (`_trace_polygon_rate_rate`, `lptools.py:1030`). The projection of
  a polytope under a linear map is again a convex polytope, so the boundary is a polygon with finitely
  many vertices, each the maximizer of some direction. The algorithm finds the four axis extremes,
  orders them CCW by `atan2` about the centroid, then **recursively refines each edge**: for edge
  `(vᵢ, vⱼ)` it optimizes the outward normal direction `n = (dy, −dx)` (one FBA with objective
  `nₓ·ax₀ + n_y·ax₁`, `lptools.py:1095–1101`); if the maximizer lies beyond the edge (`dist > tol`), it is
  a new vertex and both sub-edges recurse. This is `O(V)` LPs for `V` vertices and returns the polygon
  *exactly* — no discretization error. This is why `points` is irrelevant for production envelopes.

- **Yield-containing → adaptive upper/lower boundary** (`_trace_boundary_adaptive`, `lptools.py:1134`).
  A yield axis makes the region non-polygonal, so the boundary is traced as two functions of the x-axis:
  scan x, and at each x fix axis-0 to that value (`_make_fix_constraint`, `lptools.py:993` — for a yield
  axis this fixes `num − value·den = 0`, i.e. the ratio, as a linear equality) and optimize axis-1 up and
  down. Midpoints are refined recursively wherever the true boundary deviates from the linear
  interpolation by more than `abs_tol` (`lptools.py:1176–1199`), to a depth `max(5, log2(points))`
  (`lptools.py:1640`). The polygon is `upper + reversed(lower)`. Such plots are labelled
  `'approximate'` on the axes (`lptools.py:1740–1741`).

The 3D paths generalize this: pure-rate axes get an **exact polytope** by `ConvexHull` + iterative
face-normal refinement (`_trace_polytope_3d_rate`, `lptools.py:1206` — optimize each hull face's outward
normal, add any new vertex, repeat until no face yields one), with coplanar simplices merged into polygon
faces (`_hull_face_polygons`, `lptools.py:1272`) for clean rendering. One yield axis triggers
**slicing**: scan the yield level, trace an exact rate–rate polygon per slice, and stitch adjacent slices
into a triangle mesh (`_trace_3d_slice_polygon`, `lptools.py:1308`; `_triangulate_strips`,
`lptools.py:1371`). Two or more yield axes fall back to a full grid scan (`lptools.py:1838–1902`).
Faces are coloured by normal direction through the chosen `cmap` (`_normal_color`, `lptools.py:1924`).

To *use* it for design validation: plot the envelope of the wild-type, then call again with the design's
knockouts injected as `constraints` (e.g. `constraints=['ACALD = 0', 'PFL = 0']`) and overlay
(`show=False`, reuse the axes). A successful growth-coupled design shows the lower boundary of the second
envelope rising above zero at the growth optimum.

### 15.5 Compressed-analysis tools (PR #56)

Genome-scale FBA/plots are cheap individually, but a production envelope or a 3D scan issues *hundreds*
of LPs, and a plot of an iML1515-sized model can be slow. PR #56 added a path to do the LP work in the
**compressed space** (Ch 3) — where the network is a fraction of the size — and lift the answers back to
original reactions. Three pieces cooperate.

#### `expand_fluxes` — lifting a flux vector

`expand_fluxes(fluxes_cmp, cmp_map, orig_reaction_ids)` (`lptools.py:955`) reverses the compression to
recover a flux for every original reaction. The compression map is a list of step-dicts; each step's
`reac_map_exp` is `{ cmp_id : { orig_id : factor, … } }` — the reactions *before* the step that were
lumped into each compressed reaction, with the rational scaling that made the merge exact (the same
structure Ch 9 uses, `networktools.py`). The algorithm walks the steps **in reverse**
(`lptools.py:983–987`):

```python
for step in reversed(cmp_map):
    for cmp_id, orig_map in step["reac_map_exp"].items():
        v_cmp = fluxes.pop(cmp_id, 0.0)
        for orig_id, factor in orig_map.items():
            fluxes[orig_id] = factor * v_cmp
```

so each original reaction's flux is `v_orig = factor · v_cmp`. The factor's meaning depends on the merge
type (Ch 3):

- **coupled reactions** — `factor` is the stoichiometric coupling coefficient, so the split is exact and
  deterministic;
- **parallel reactions** — the total compressed flux is distributed by the stored proportional factors;
- **removed reactions** (blocked / never appear in any step) — set to `0.0` (`lptools.py:988–990`).

This is the crucial contrast with Ch 9's `expand_sd`, which lifts an *intervention set* and therefore
**ignores** the factors (it only needs the set of original IDs behind a compressed one). Here we lift a
*quantity*, so the factors are load-bearing — dropping them would give wrong flux magnitudes.

#### `slim_fba_via_cmp` — objective-only FBA on the compressed model

`slim_fba_via_cmp(model, cmp_model, cmp_map, obj=..., constraints=..., ...)` (`lptools.py:617`) returns
just the **optimal objective value**, in original-model units, without ever materializing a full flux
vector — the cheapest possible compressed FBA. It:

1. resolves and compresses the constraints (`resolve_gene_constraints` then `compress_constraints`,
   which applies the same coefficient-scaling as the compression map, `lptools.py:656–660`);
2. **traces only the objective reactions** through the compression steps to accumulate their cumulative
   coupling factor and map them to compressed IDs (`lptools.py:678–689`) — this is the "slim" part:
   instead of expanding an `O(n)` vector it follows only the handful of reactions in `obj`. The
   compressed objective coefficient is `coeff · cum_factor`;
3. solves with `slim_solve` (objective only, no vector) and returns `−opt_cx`/`opt_cx` per `obj_sense`
   (`lptools.py:722–730`). Because `c` was pre-scaled by `cum_factor`, the returned number is already in
   original units.

Use it when you need to evaluate a design's objective (say, max growth) thousands of times — e.g. inside
an outer search or a batch scan over candidate constraint sets — on a large model.

#### Compressed `plot_flux_space`

`plot_flux_space` accepts optional `cmp_model` and `cmp_map` kwargs (`lptools.py:1470–1503`). When both
are supplied it:

- resolves gene constraints on the *original* model and compresses them (`compress_constraints`);
- maps each axis's reaction IDs to compressed IDs via `_build_cmp_reverse_map`
  (`networktools.py:515` — walks forward through steps building `{orig_id : final_cmp_id}`), while
  tracing the cumulative coupling factor per axis into `_ax_scale` (`lptools.py:1487–1499`);
- switches `model = cmp_model` and runs the entire tracing machinery in the small space;
- finally **rescales the traced coordinates back to original units** by multiplying by `_ax_scale`
  (`lptools.py:1589–1596` for limits, `1660–1668` for the polygon vertices) and restores the original
  reaction names as axis labels (`lptools.py:1575–1581`).

The user sees an envelope drawn in original-reaction coordinates, but every LP behind it ran on the
compressed network. All the coupling factors are applied at the end, so the picture is quantitatively
identical to the uncompressed one (up to the exact rational scalings of Ch 3).

### 15.6 Where these fit the developer's workflow

These functions form the **exploration and reporting** layer that brackets a `compute_strain_designs`
run:

- **Before** — characterize the model. `fba(model)` confirms the wild-type growth rate; `fva(model)`
  (or `remove_redundant_bounds`) finds blocked/essential reactions and cleans non-binding bounds;
  `plot_flux_space(model, (growth, product))` shows the baseline production envelope, revealing whether
  the target is even producible and how far it sits from growth coupling. `yopt` gives the theoretical
  maximum yield the design could ever reach.
- **After** — validate a returned design. Take an `SDSolutions` entry (Ch 13), feed its knockouts as
  `constraints` to `fba`/`yopt`/`plot_flux_space`, and confirm that the SUPPRESS behavior is now
  impossible (product-free growth infeasible / envelope lower bound lifted) while the PROTECT behavior
  survives (growth still ≥ threshold). The overlaid production envelope is the standard figure for
  "this design works".
- **At scale / batch** — when validating hundreds of candidate designs on a genome-scale model, the
  PR #56 compressed tools (`slim_fba_via_cmp` for objective sweeps, compressed `plot_flux_space` for
  envelopes, `expand_fluxes` to report a representative flux vector in original terms) let all this LP
  work happen in the compressed space the pipeline already built, so exploration costs a fraction of the
  naïve genome-scale price.

None of this touches the MILP or the compute pipeline; it is the read-only microscope you point at a
model or a solution.


## 16. Testing & contributing

Tests live in `tests/` (pytest, one file per feature area) and are run with:

```bash
pytest tests -v --log-cli-level=INFO --junit-xml=test-results.xml
```

**CI matrix** (`.github/workflows/CI-test.yml`): OS `ubuntu-latest` / `windows-latest`; Python
`3.10`–`3.13`; both `pip` and `conda`. CPLEX is excluded for Python 3.13 (max supported: 3.12). A
JPype/JVM-shutdown segfault on Ubuntu is tolerated via a JUnit-XML exit-code check rather than the raw
process exit code.

**Correctness gates.** The canonical known-answer tests are the ones to keep green after any change to
the pipeline: gene-level MCS on `e_coli_core` = **455** solutions, and on `iML1515` = **393**. These
exercise compression, GPR extension, MILP construction, and enumeration end-to-end.

**What to re-verify after changes:**

1. **Constraint parsing** — any change to `parse_constr.py` must be checked against all input formats
   (string, list of strings, list of structured constraints; see Ch 12).
2. **MILP construction** — after touching `link_z`, `build_primal_from_cbm`, or the dualization
   functions (Ch 6, Ch 7): run a small toy model and check solutions against known answers.
3. **Compression** — changes to `compression.py` / `networktools.py` must verify that `expand_sd`
   reconstructs original-space solutions correctly (Ch 9), and that every returned design still satisfies
   all PROTECT modules when re-evaluated on the *original* model (Ch 10).
4. **Solver backends** — changes to any `*_interface.py` require testing with that solver installed
   (Ch 14).
5. **GPR translation** — changes to `networktools.extend_model_gpr` or
   `SDSolutions._translate_genes_to_reactions` must be tested with models that have AND/OR GPR logic
   (e.g., `iJO1366`).

**Adding tests.** Use a small toy model (3–5 reactions) for unit tests of MILP construction;
`e_coli_core` for full-pipeline integration tests; `iJO1366` / `iML1515` for performance regressions.
Test with at least GLPK (always available) plus one commercial solver where possible. Because the
branch-and-bound tree is seed-dependent, any *timing* comparison should use several seeds and report a
distribution, not a single run (Ch 8, Ch 11).

**Profiling.**

```python
import cProfile, pstats
cProfile.run("compute_strain_designs(model, sd_modules=[...], solver='glpk')", 'profile_out')
pstats.Stats('profile_out').sort_stats('cumulative').print_stats(30)
```

The hot spots are typically the preprocessing FVA, `link_z` (its per-constraint LP bounding), and the
solver's enumeration loop (Ch 11).
