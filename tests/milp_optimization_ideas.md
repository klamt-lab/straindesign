# MCS MILP Performance: Comprehensive Analysis & Test Matrix

## The Core Question
StrainDesign produces the most **concise** MILP (via compression) yet is slower
than gMCSpy and CNA's MCSEnumerator. Why?

---

## 1. Key Findings from Comparing Implementations

### 1.1 StrainDesign vs CNA (MATLAB) — Same Mathematical Formulation

StrainDesign was ported from CNA's `MCS_MILP.m`. The formulation is identical:
`build_primal` → `dualize` → `dual_2_farkas` → `link_z`. Even the comments
match verbatim (e.g., the "Consider that the following is not implemented" note
about Farkas' lemma special case).

**Critical difference: CNA has TWO separate link_z paths:**

| | CNA `link_z_indicators` (M=0) | CNA `link_z` (M>0) | StrainDesign `link_z` |
|---|---|---|---|
| M-bounding LPs | **None** (skipped entirely) | Yes (via `computeM`) | **Always** (even if M=inf) |
| Indicator setup | Direct: move knockable rows to indicators | N/A (big-M only) | After M-values computed, leftover inf→indicators |
| Duplicate finding | None needed | None | O(n²) loop |
| Complexity | O(n) | O(n × LP_time) | O(n × LP_time + n²) |

**Hypothesis**: StrainDesign always runs M-bounding LPs even when the result is
M=inf for every constraint (which happens with infinite bounds). This is pure
waste — CNA's indicator path skips it entirely.

### 1.2 gMCSpy — Different Mathematical Formulation

gMCSpy uses the **Nullspace-Based (NB) dual** instead of Farkas-Lemma-Based (FLB):

**FLB (StrainDesign/CNA):**
```
S^T u + v = 0,  T^T w + v_T = 0    (n equality constraints)
b^T [u; y_ineq] <= -1               (Farkas certificate)
u ∈ R^m,  y_ineq >= 0               (m + k continuous vars)
z_i = 0 → v_i constraints active    (knockable via z)
```
Total continuous vars: m + k (metabolite duals + inequality duals)
Total equality constraints: n (one per reaction)

**NB (gMCSpy):**
```
K^T v = 0                           ((n-m) equality constraints)
t^T w <= -1                         (Farkas certificate, 1 row)
v ∈ R^n,  w >= 0                    (n + t continuous vars)
z_i = 0 → v_i = 0 (or <= 0)        (knockable via z)
```
Total continuous vars: n + t
Total equality constraints: (n-m) + 1

**For iML1515 (n≈2700, m≈1877):**
| | FLB | NB | Difference |
|---|---|---|---|
| Continuous vars | ~1877 + k | ~2700 + t | NB has more vars |
| Equality constraints | ~2700 | ~824 | **NB has 70% fewer** |

The NB approach trades more variables for far fewer constraints. Since LP
relaxation time at each B&B node scales primarily with constraint count,
this makes the NB formulation faster despite having more variables.

### 1.3 CNA's Kernel Approach (MCS2)

When `kn` (kernel/nullspace matrix) is passed to CNA:
- Replaces reaction-level dual with kernel-level dual
- Reduces both variables AND constraints proportionally to kernel dimension
- Same principle as gMCSpy but within CNA's framework

### 1.4 Solver Tuning

| Parameter | StrainDesign | gMCSpy | CNA |
|---|---|---|---|
| MIPFocus | 0 (balanced) | 3 (bound proof) | 1 (feasibility) |
| Presolve | **0** (disabled, Gurobi 13 bug) | 1 | auto |
| Cuts | auto | 2 (aggressive) | auto |
| PoolSolutions | auto | **10000** | **101** |
| PoolSearchMode | 0 | 1 | populate() with intensity=4 |
| IntFeasTol | 1e-9 | 1e-5 | 1e-10 |
| Cutoff | none | maxKOLength | none |
| PreSOS1Encoding | auto | 2 | auto |

---

## 2. Complete Test Matrix

### Category A: MILP Formulation

| # | Idea | Hypothesis | Effort | Impact |
|---|---|---|---|---|
| A1 | **Nullspace-based dual (NB)** | 70% fewer equality constraints → faster LP relaxation at each B&B node. Main reason gMCSpy is fast. | HIGH | HIGH |
| A2 | **Skip M-bounding when using indicators** | CNA's indicator path does zero LP solves. We always compute Ms then throw them away. Fast-path for M=inf. | LOW | MED |
| A3 | **RREF-reduced stoichiometry** | Replace S with its RREF in the primal. Reduces row count and may produce sparser dual. Compression's RREF already exists. | MED | MED |
| A4 | **Reaction splitting (fwd/rev)** | Split reversible reactions into two irreversible ones. All variables become non-negative → all dual constraints become inequalities (no free vars, no equalities split to two ineqs). gMCSpy does this. | MED | MED |
| A5 | **Split mass balances (S*v=0 → S*v<=0, -S*v<=0)** | Equalities become ineqs. Dual variables become non-negative. May interact better with solver presolve. | LOW | LOW |
| A6 | **Separate fwd/rev variables feeding into z** | Introduce v_fwd, v_rev ≥ 0 with v = v_fwd - v_rev, single z controls both. More variables but all non-negative → tighter LP relaxation. | MED | MED |
| A7 | **Dual normalization (alpha=1)** | gMCSpy uses z=1 → v≥1 (not just v≥0). Prevents trivial solutions and may tighten LP relaxation. | LOW | LOW-MED |
| A8 | **Direct indicator construction** | Instead of z_map→M-values→bigM/indicator decision, go straight from z_maps to indicator constraints. No bounding LPs needed. | LOW | MED |

### Category B: Preprocessing / Bound Tightening

| # | Idea | Hypothesis | Effort | Impact |
|---|---|---|---|---|
| B1 | **FVA to tighten dual bounds** | Solve max/min for each dual variable. Tighter bounds → better LP relaxation → faster branching. CNA does this in `bound_vars`. | MED | MED |
| B2 | **Remove zero-flux reactions before MILP** | Already done by compression. But verify no zero-flux reactions survive into the MILP. | LOW | LOW |
| B3 | **Remove redundant constraints** | After dualization, some constraints may be redundant (dominated by others). Detect and remove. | MED | LOW |
| B4 | **Reorder variables/constraints** | Sort by sparsity, degree, or other heuristic. May help solver's internal presolve. | LOW | LOW |
| B5 | **Preconditioning / scaling** | CNA has a `preconditioning()` function (log-based scaling). May reduce condition number and improve numerical stability. | MED | LOW-MED |
| B6 | **Use compression nullspace for MILP** | Reuse the kernel from `compress_model` instead of letting the solver discover it. Pass as `kn` parameter. | MED | HIGH |

### Category C: Solver Parameters

| # | Idea | Hypothesis | Effort | Impact |
|---|---|---|---|---|
| C1 | **Re-enable Presolve** | Presolve=0 on Gurobi 13+ kills performance. Either pin Gurobi 12, find workaround, or report bug. | LOW | HIGH |
| C2 | **MIPFocus=3 (bound proof)** | gMCSpy uses this for non-targeted enumeration. Helps prove optimality faster → prunes more branches. | LOW | MED |
| C3 | **Aggressive cuts (Cuts=2)** | More cutting planes → tighter LP relaxation → faster branching. | LOW | MED |
| C4 | **Solution pool (PoolSolutions=10000)** | Find many solutions per solve call instead of one-at-a-time. Massive speedup for enumeration. | MED | HIGH |
| C5 | **Cutoff = max_cost** | Hard prune branches above max_cost. gMCSpy does this. Avoids exploring useless subtrees. | LOW | MED |
| C6 | **PreSOS1Encoding=2** | gMCSpy sets this. May help with indicator constraint handling. | LOW | LOW |
| C7 | **Relax tolerances** | IntFeasTol=1e-6 instead of 1e-9. Less time spent on numerical precision. | LOW | LOW |
| C8 | **MIP start / warm start** | Pass previous solution as hint for next solve. Gurobi can use this to seed heuristics. | LOW | MED |

### Category D: Enumeration Strategy

| # | Idea | Hypothesis | Effort | Impact |
|---|---|---|---|---|
| D1 | **Solution pool exploitation** | Use Gurobi's solution pool (PoolSolutions + populate) to extract multiple MCS per solve. | MED | HIGH |
| D2 | **Force cardinality (k=1,2,3,...)** | Add constraint sum(z)=k, increment k when exhausted. Focuses solver on one cardinality at a time. CNA and gMCSpy (CPLEX) do this. | LOW | MED |
| D3 | **Batch verification** | Verify multiple solutions in parallel instead of one-at-a-time. | MED | MED |
| D4 | **Lazy integer cuts** | Add exclusion constraints lazily via callback instead of modifying the model between solves. | MED | MED |
| D5 | **Conflict analysis on cuts** | Detect when accumulated exclusion cuts make the problem infeasible early. | MED | LOW |

### Category E: Alternative Formulations

| # | Idea | Hypothesis | Effort | Impact |
|---|---|---|---|---|
| E1 | **MCS2 rowspace formulation** | Use K^T (nullspace of S) to eliminate metabolite duals entirely. Equivalent to gMCSpy's NB approach but within our framework. | HIGH | HIGH |
| E2 | **Indicator-only formulation (no big-M at all)** | Never compute M-values. Use pure indicator constraints like gMCSpy. Simpler, avoids bounding LP overhead. | MED | MED-HIGH |
| E3 | **Compact dual (eliminate free variables)** | Free dual variables (from equality constraints) can be eliminated by substitution. Reduces total variable count at cost of denser constraints. | HIGH | MED |
| E4 | **G-matrix approach for gene knockouts** | gMCSpy's G-matrix maps gene sets to reactions. Could replace our GPR extension + second compression pass. | HIGH | MED |

---

## 3. Priority Ranking

### Tier 1: Quick wins (< 1 day, high confidence)

1. **A2/A8: Skip M-bounding for indicator path** — CNA proves this works.
   When M=inf, go directly from z_maps to indicator constraints.
2. **C1: Re-enable Presolve** — Find Gurobi 13 workaround or pin version.
3. **C2+C3: MIPFocus=3, Cuts=2** — Copy gMCSpy's proven settings.
4. **C5: Cutoff=max_cost** — Trivial to add, prunes useless branches.
5. **C8: MIP start** — Pass previous solution as hint.

### Tier 2: Medium effort, high impact (1-3 days)

6. **C4/D1: Solution pool exploitation** — Multiple solutions per solve.
7. **E2: Indicator-only formulation** — Eliminate M-bounding entirely.
8. **D2: Force cardinality** — Focus solver per cardinality level.
9. **B1: FVA to tighten dual bounds** — Better LP relaxation.

### Tier 3: Research / high effort, potentially transformative

10. **A1/E1: Nullspace-based dual** — The gMCSpy/MCS2 approach. 70% fewer
    constraints. Potentially the single biggest speedup.
11. **B6: Reuse compression nullspace** — Pass kernel to MILP construction.
12. **A4: Reaction splitting** — All-nonneg variables, no free duals.
13. **E3: Compact dual** — Eliminate free variables by substitution.

---

## 4. Suggested Test Protocol

### Test model
Use `model_small_example` (10 rxns, 6 mets) for correctness + structure
comparison. Use `iML1515_core` (iMLcore) for timing benchmarks.

### For each formulation variant:
1. Write LP file (`backend.write(path)`) → inspect structure
2. Count: total vars, binary vars, continuous vars, ineq constraints,
   eq constraints, indicator constraints, nnz in constraint matrix
3. Measure: MILP construction time, time-to-first-solution, time-to-10-solutions
4. Compare: solution quality (same MCS found?), numerical stability

### Comparison baselines:
- Current StrainDesign (this branch)
- CNA MCSEnumerator via MATLAB (CPLEX, same model)
- gMCSpy (if installable)

---

## 5. Why Conciseness ≠ Speed

The paradox: compression gives us fewer reactions/metabolites, but:

1. **Fewer reactions → fewer z-variables → faster B&B** ✓ (this helps)
2. **But: FLB dual creates m + k continuous vars** — compression reduces n but
   the dual size is dominated by m (metabolites), which compression reduces less
3. **The constraint matrix density increases** after compression (merged reactions
   have denser stoichiometry) → slower LP relaxation per node
4. **Compression preprocessing takes 70+ seconds** — amortization only pays off
   if MILP solve takes much longer
5. **The M-bounding overhead scales with knockable constraints** — compression
   doesn't reduce this proportionally
6. **Solver presolve does similar reductions internally** — our manual compression
   may partially duplicate what Gurobi would discover on its own

The NB approach (gMCSpy) sidesteps issues 2-5 by using a fundamentally different
dual structure that produces fewer constraints regardless of preprocessing.

---

## 6. Hypothesis: Why gMCSpy is Fast

**Primary factor**: NB dual → ~70% fewer equality constraints → LP relaxation
at each B&B node is 3-5x faster.

**Secondary factors**:
- Pure indicators (no M-bounding overhead)
- Solution pool (10000 solutions per solve)
- Aggressive solver tuning (MIPFocus=3, Cuts=2, Cutoff)
- No preprocessing overhead (lets Gurobi presolve handle it)

**What gMCSpy does NOT have**: compression. So for very large models where
compression removes 40% of reactions (hence 40% of z-variables), StrainDesign's
approach of compress-then-solve may eventually win — but only if we also adopt
the NB dual and solver tuning improvements.

The ideal approach: **compress first, then use NB dual on compressed model**.
