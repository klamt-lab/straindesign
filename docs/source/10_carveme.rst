CarveMe modules: network reconstruction
=======================================

Every other module type intervenes in a network that already exists: it picks knockouts,
additions or regulatory changes so that some flux region becomes unreachable or stays reachable. A
``'carveme'`` module goes the other way. It starts from a universe of candidate reactions and asks
which of them to keep, so the answer is a network rather than an intervention set. That is the
reconstruction problem `CarveMe <https://github.com/cdanielmachado/carveme>`_ solves, and this
module type is the StrainDesign backend for it.

CarveMe reconstructs by *carving*: score every reaction in a universal network by how well the
genome supports it, then delete the low scorers until what remains still grows. Nothing in that
formulation ties a reaction's presence to its ability to carry flux, so the result can contain
reactions that can never run and can lose reactions the genome plainly supports. Posed the other
way round -- keep the annotated core, buy the cheapest additions that let it run -- both become
constraints:

.. math::

   \begin{align}
   z_r = 0 \quad &\Rightarrow \quad v_r = 0 \\
   z_r = 1,\ r \in \text{core} \quad &\Rightarrow \quad d_r\, v_r \ge t_r \\
   \min \quad &\sum_r c_r\, z_r
   \end{align}

The first line says a reaction that was not bought carries nothing. The second says a core
reaction that *was* kept demonstrably runs, in a direction :math:`d_r` and above a threshold
:math:`t_r`. Together they mean the reconstructed network has no blocked reactions at all -- not
as a post-processing step, but because no other solution is feasible.

A minimal example
-----------------

.. code-block:: python

    import cobra, straindesign as sd

    universe = cobra.io.load_model('e_coli_core')
    growth = 'BIOMASS_Ecoli_core_w_GAM'

    # reactions the genome supports; these must run if they are kept
    annotated = [r.id for r in universe.reactions if r.gene_reaction_rule]

    module = sd.SDModule(universe, sd.CARVEME,
                         constraints=[growth + ' >= 0.1'],
                         core_reactions=annotated)

    # ki_cost lists the candidates and prices them. A negative cost rewards keeping a
    # reaction; a reaction absent from ki_cost is not a candidate and stays unconditionally.
    cost = {r.id: (-1.0 if r.gene_reaction_rule else 1.0) for r in universe.reactions}

    solution = sd.compute_strain_designs(universe, sd_modules=[module], ki_cost=cost,
                                         solution_approach='best', max_solutions=1)
    keep = {r for r, bought in solution.reaction_sd[0].items() if bought}

The ratio between the reward for an annotated reaction and the price of an unannotated one is the
one knob that matters. It decides how much evidence-free chemistry the reconstruction is willing
to buy in order to let the annotated core run, and it is worth tuning against whatever the model
is for -- gene essentiality, growth phenotypes, or flux data.

How it is built
---------------

A CarveMe module is assembled in :class:`straindesign.SDProblem` like every other module type, so
everything else in the pipeline applies to it unchanged: network and GPR compression, the
preprocessing FVAs, gene-level design, the solution pool, and combination with other modules.

Its primal is a PROTECT primal -- the module's constraints must remain feasible -- which means the
**gating half comes for free from the knock-in machinery**. A candidate is a knock-in, so
``z_kos_kis`` inverts its column in the z-maps and :meth:`straindesign.SDProblem.link_z` emits
:math:`v_r = 0` for every candidate that was not bought. The module adds only the half no other
module type states: one row per core reaction,

.. math::

   -d_r v_r \le -t_r

mapped to that reaction's own binary, which ``link_z`` then gates exactly like any other knockable
row. Because that row is single-variable, ``link_z`` gives it a finite big-M read straight off the
variable's own bound; for an irreversible reaction that M is zero and the link is exact. This is
sound rather than merely conventional because StrainDesign pins the solvers' integrality tolerance
(CPLEX to 0, Gurobi to 1e-9), so a binary cannot rest fractionally far enough from 1 to buy back
:math:`M \cdot \text{tol}` units of slack. Tools that use big-M gating at default tolerances do
leak here, and the leak is large enough to carry a pathway.

Since the objective is exactly :math:`\min \sum_r c_r z_r`, a CarveMe module needs no objective of
its own: it minimises the intervention cost the same way an MCS computation does, with a negative
``ki_cost`` rewarding a reaction rather than charging for it.

Combining with other modules
----------------------------

Because it is an ordinary module, a reconstruction can be constrained by anything else
StrainDesign can express. Adding a PROTECT module makes the reconstructed network satisfy a
further phenotype, in the same MILP rather than as a second pass:

.. code-block:: python

    carve = sd.SDModule(universe, sd.CARVEME, constraints=[growth + ' >= 0.1'],
                        core_reactions=annotated)
    secrete = sd.SDModule(universe, sd.PROTECT, constraints=['EX_ac_e >= 1'])
    solution = sd.compute_strain_designs(universe, sd_modules=[carve, secrete], ki_cost=cost,
                                         solution_approach='best', max_solutions=1)

Inclusion stays soft
--------------------

The core is *rewarded*, not *required*. Making it mandatory is tempting -- it sounds like the
stronger guarantee -- but it removes the choice to omit a reaction whose supporting chemistry is
too expensive or missing entirely, and at genome scale the result is usually infeasible. Under the
soft formulation the cost ratio decides, and reactions that were dropped are reported rather than
silently absent.

One consequence is worth stating plainly: StrainDesign never lets an intervention relax a model
bound, and a CarveMe module is no exception. A candidate whose lower bound is positive cannot
carry zero flux, so it cannot be dropped at all -- it is mandatory by construction, and it is paid
for. A maintenance reaction such as ``ATPM`` is the usual case.

Directions and thresholds
-------------------------

"The reaction runs" means :math:`|v_r| \ge t_r`, which is not linear. There are two ways around
that and only one of them works:

* Split the reaction into forward and reverse halves and demand :math:`v^+ + v^- \ge t`. This is
  **vacuous**: the two halves cycle against each other at zero net flux and satisfy the
  constraint while the reaction does nothing. The split even creates the cycle.
* Fix a direction :math:`d_r` per reaction and demand :math:`d_r v_r \ge t_r`. Linear, and it
  means what it says.

A direction cannot be read off each reaction's own FVA range. Individually feasible directions
need not be jointly consistent, and demanding all of them together is then infeasible. Instead
:func:`straindesign.build_witness` maximises each core reaction's flux in turn and sums the
normalised solutions. The feasible set is convex, so the sum is feasible; and it is non-zero
wherever any summand was, so one flux state carries the whole core and its signs are consistent by
construction. ``compute_strain_designs`` runs this during preprocessing, on the model the MILP is
built from and under the module's own constraints; pass ``core_directions`` to supply your own.

Core reactions that cannot carry flux under the module's constraints at any price are reported as
a warning and dropped from the core -- there is no reachable must-run condition for them, so
demanding it would make the module infeasible. They are also excluded from purchase, since a
reward would otherwise buy a reaction that sits in the result blocked, which is the defect this
module type exists to rule out.

Loops
-----

With ``loopless=True`` (the default) the module also carries the Schellenberger thermodynamic
constraint: a free potential :math:`\mu_m` per metabolite, and :math:`d_r \sum_m S_{mr}\mu_m < 0`
whenever core reaction :math:`r` runs. This forbids internally cycling flux, so a core reaction
cannot satisfy its must-run condition by spinning in a thermodynamically impossible loop with its
neighbours -- the letter of "it carries flux" without its spirit. These rows are multi-variable,
so ``link_z`` realises them as genuine indicator constraints. The cost is one continuous variable
per metabolite and one indicator per core reaction.
