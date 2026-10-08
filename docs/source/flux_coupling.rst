Flux coupling analysis
======================

Flux coupling analysis (FCA) classifies every pair of reactions by how their fluxes depend on each
other in steady state. Reaction *i* is *directionally coupled* to reaction *j* (*i* → *j*) when
every steady-state flux with *r*\ :sub:`j` = 0 also has *r*\ :sub:`i` = 0, i.e. knocking out *j*
blocks *i*. Two reactions are *partially coupled* when each is directionally coupled to the other,
and *fully coupled* when, in addition, their fluxes have a fixed ratio. A reaction is *blocked* when
it carries no flux in any steady state.

.. code-block:: python

   import straindesign as sd

   fc = sd.flux_coupling_analysis(model, solver='cplex')
   fc.reactions   # unblocked reactions, indexing the table
   fc.blocked     # blocked reactions
   fc.table[fc.reactions.index('PGI'), fc.reactions.index('PFK')]

The table follows the convention of F2C2: 0 uncoupled, 1 fully coupled, 2 partially coupled,
3 *i* → *j* (row *i*, column *j*) and 4 *j* → *i*; the diagonal is 1. As in F2C2, blocked
reactions are listed separately and left out of the table. The argument ``reactions`` restricts
the analysis to a subset of reactions, which needs fewer LPs.

The analysis runs in the flux cone {*S r* = 0, *r*\ :sub:`k` ≥ 0 for irreversible *k*}: finite
flux bounds are ignored, a reaction with lb ≥ 0 is irreversible, a reaction with lb < 0 and
ub ≤ 0 is irreversible in the backward direction, and a reaction with lb = ub = 0 is blocked.
Couplings found in the cone also hold in any model with the same stoichiometry and
irreversibilities whose bounds are finite.

**Method.** The network is first compressed by lumping reactions with proportional fluxes, using
the exact nullspace that StrainDesign also uses for strain design (``compress=False`` skips this;
the result is the same, only slower). Parallel reactions are not lumped, because they are not
coupled with each other. Reactions lumped together are fully coupled.

For each target *j*, a single LP maximizes the sum of *t*\ :sub:`l`, with
0 ≤ *t*\ :sub:`l` ≤ 1 and *t*\ :sub:`l` ≤ *r*\ :sub:`l` for every irreversible reaction *l*, under
*r*\ :sub:`j` = 0. Since the flux vectors of all reactions that can carry flux add up to one flux
vector in the cone, every such reaction reaches *t*\ :sub:`l` = 1 at the optimum, and the
irreversible reactions left at 0 are directionally coupled to *j*. The LP object is reused across
targets, so each LP restarts from the previous optimal basis. A reversible reaction *i* is
coupled to *j* iff its row of an exact kernel basis of *S* lies in the span of the rows of *j* and
of the irreversible reactions that the LP found blocked; this is decided by exact rational row
reduction. Blocked reactions are found the same way, with an LP without a target, and are not
used as targets. Fully coupled pairs are those with proportional rows in the exact kernel of the
network without its blocked reactions; mutually coupled pairs whose rows are not proportional
are partially coupled.

The LPs only decide which irreversible reactions can carry flux, with a threshold of 0.5 on a
quantity that is either 0 or 1 at an exact optimum. Values in between are settled by a separate
LP solved from scratch, and an LP that fails is solved again from scratch rather than read as a
result. All other decisions are made in exact arithmetic.
