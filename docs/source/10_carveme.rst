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
module type states: the must-run condition.

For a core reaction whose direction the bounds already settle, that is a single row
:math:`-d_r v_r \le -t_r` mapped to the reaction's own binary, which ``link_z`` gates exactly like
any other knockable row -- with a finite big-M read straight off the variable's own bound, which
for an irreversible reaction is zero, so the link is exact. For a reversible one the module adds
the direction pair described below, and the rows tying :math:`z^f_r + z^r_r = z_r` are what carry
the z-mapping instead.

Where a row does use a big-M rather than an indicator, that is sound rather than merely
conventional, because StrainDesign pins the solvers' integrality tolerance (CPLEX to 0, Gurobi to
1e-9): a binary cannot rest fractionally far enough from 1 to buy back :math:`M \cdot \text{tol}`
units of slack. Tools that gate with big-M at default tolerances do leak here, and the leak is
large enough to carry a pathway.

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

Directions are chosen, not fixed
--------------------------------

"The reaction runs" means :math:`|v_r| \ge t_r`, which is not linear. There are two ways around
that and only one of them works:

* Split the reaction into forward and reverse halves and demand :math:`v^+ + v^- \ge t`. This is
  **vacuous**: the two halves cycle against each other at zero net flux and satisfy the
  constraint while the reaction does nothing. The split even creates the cycle.
* Fix a direction :math:`d_r` per reaction and demand :math:`d_r v_r \ge t_r`. Linear, and it
  means what it says.

So a direction has to be picked -- and **the MILP picks it**, rather than anything deciding in
advance. For a reaction whose bounds already admit one sign there is nothing to choose. For a
genuinely reversible one the module adds a pair of binaries tied to the reaction's own
:math:`z_r` by :math:`z^f_r + z^r_r = z_r`, so buying the reaction picks exactly one direction and
not buying it picks neither. Each direction's must-run row is relaxed by its own binary, with the
relaxation value read off that reaction's own bound, so the row says no more than the bound
already says when the binary is zero.

Deciding directions ahead of time -- by FVA, or from a precomputed witness flux state -- looks
cheaper and is wrong twice over. It removes solutions, because which direction a reaction must run
in depends on which *other* reactions were bought, and that is the very thing being decided. And a
direction read off each reaction's own range need not be jointly consistent with the others, so
demanding them together can be infeasible when the module is not. ``core_directions`` remains
available for a caller who genuinely wants to pin one.

One thing this buys for free: a core reaction that cannot carry flux under the module's
constraints is simply not bought. Its must-run row can never be satisfied, so :math:`z_r = 0` is
the only feasible choice. No detection pass, no special case -- and which annotated reactions
could not be connected is read off the result rather than predicted before it.

Where preprocessing has *widened* a bound to infinity -- which
``bound_blocked_or_irrevers_fva`` does deliberately, for bounds it proved never bind -- the
module's FVA range supplies the finite relaxation value instead. That range is still redundant at
the relaxed value, so the row stays exactly tight, and it is a reason to leave the preprocessing
FVAs on for this module type.

``core_thresholds`` scales ``min_flux`` per reaction if some core reactions should be required to
carry more flux than others.

Loops
-----

With ``loopless=True`` (the default) the module also carries the Schellenberger thermodynamic
constraint: a free potential :math:`\mu_m` per metabolite, and :math:`d_r \sum_m S_{mr}\mu_m < 0`
whenever core reaction :math:`r` runs. This forbids internally cycling flux, so a core reaction
cannot satisfy its must-run condition by spinning in a thermodynamically impossible loop with its
neighbours -- the letter of "it carries flux" without its spirit. These rows are multi-variable,
so ``link_z`` realises them as genuine indicator constraints. The cost is one continuous variable
per metabolite and one indicator per core reaction.

Pipeline options
----------------

Compression and the preprocessing FVAs are ordinary options, not something this module type opts
out of. ``compress=True`` (the default) alternates parallel and coupled lumping to a fixed point;
``compress='coupled'`` runs a single coupled pass and no parallel lumping, which is much cheaper
and leaves intervention costs alone -- lumping two parallel reactions produces a group whose
knockout cost is the sum of its members, whereas a coupled group is knocked out by knocking out
any one member. On a universe, ``compress=False`` is usually right: there is little to compress
and the attempt costs more than it saves.
