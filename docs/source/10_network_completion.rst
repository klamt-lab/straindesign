Network completion
==================

Every other module type in StrainDesign *intervenes* in a network that already exists: it picks
knockouts, additions or regulatory changes so that some flux region becomes unreachable or stays
reachable. A ``'complete'`` module goes the other way. It starts from a universe of candidate
reactions and asks which of them to keep, so the answer is a network rather than an intervention
set. That is the reconstruction problem, and it is what draft genome-scale models are built by.

Reconstruction is usually posed as *carving*: take the universal network, score each reaction by
how well the genome supports it, and delete the low scorers until the model is small enough to
grow. Nothing in that formulation stops the tool from deleting a reaction the genome clearly does
support, or from keeping one that can never carry flux in the result. Draft models routinely
contain both. Posed as a completion instead, both guarantees fall out of the constraints:

.. math::

   \begin{align}
   z_r = 0 \quad &\Rightarrow \quad v_r = 0 \\
   z_r = 1,\ r \in \text{core} \quad &\Rightarrow \quad d_r\, v_r \ge t_r \\
   \min \quad &\sum_r c_r\, z_r
   \end{align}

The first line says a reaction that was not bought carries nothing. The second says a core
reaction that *was* kept demonstrably runs, in a specific direction :math:`d_r` and above a
threshold :math:`t_r`. Together they mean the completed network has no blocked reactions at all --
not as a post-processing step, but because no other solution is feasible.

Both are enforced as **indicator constraints**, never big-M. A binary that rests fractionally
between 0 and 1, which solvers permit within their integrality tolerance, would otherwise licence
:math:`M \cdot \text{tol}` units of flux through a reaction that reads as switched off. At
:math:`M = 1000` and a default tolerance of :math:`10^{-5}` that is enough flux to carry a
pathway, and the reaction then appears in the model with a flux nobody bought.

A minimal example
-----------------

.. code-block:: python

    import cobra, straindesign as sd

    universe = cobra.io.load_model('e_coli_core')
    growth = 'BIOMASS_Ecoli_core_w_GAM'

    # reactions the genome supports; these must run if they are kept
    annotated = [r.id for r in universe.reactions if r.gene_reaction_rule]

    module = sd.SDModule(universe, sd.COMPLETE,
                         constraints=[growth + ' >= 0.1'],
                         core_reactions=annotated)

    # ki_cost lists the candidates and prices them. A negative cost rewards keeping a
    # reaction; a reaction absent from ki_cost is not a candidate and stays unconditionally.
    cost = {r.id: (-1.0 if r.gene_reaction_rule else 1.0) for r in universe.reactions}

    solution = sd.compute_strain_designs(universe, sd_modules=[module], ki_cost=cost)
    keep = {r for r, kept in solution.reaction_sd[0].items() if kept}

The ratio between the reward for an annotated reaction and the price of a heterologous one is the
one knob that matters. It decides how much evidence-free chemistry the completion is willing to
buy in order to let the annotated core run, and it is worth tuning against whatever the model is
for -- gene essentiality, growth phenotypes, or flux data.

Inclusion stays soft
--------------------

The core is *rewarded*, not *required*. Making it mandatory is tempting -- it sounds like the
stronger guarantee -- but it removes the choice to omit a reaction whose supporting chemistry is
too expensive or missing entirely, and at genome scale the result is usually infeasible. Under the
soft formulation the cost ratio decides, and reactions that were dropped are reported rather than
silently absent.

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
:func:`straindesign.completion.build_witness` maximises each core reaction's flux in turn and sums
the normalised solutions. The feasible set is convex, so the sum is feasible; and it is non-zero
wherever any summand was, so one flux state carries the whole core and its signs are consistent by
construction. Pass the result as ``core_directions``, or omit it and let
``compute_strain_designs`` build it for you.

Core reactions that cannot carry flux under the module's conditions at any price are reported as a
warning and left out. They have no must-run condition to satisfy, so a reward for keeping one
would buy a reaction that is dead in the result -- precisely the defect the formulation exists to
rule out.

Loops
-----

With ``loopless=True`` (the default) the completion also carries the Schellenberger
thermodynamic constraint: a free potential :math:`\mu_m` per metabolite, and
:math:`\sum_m S_{mr}\mu_m < 0` whenever reaction :math:`r` runs in its assigned direction. This
forbids internally cycling flux, so a core reaction cannot satisfy its must-run condition by
spinning in a thermodynamically impossible loop with its neighbours. It costs one continuous
variable per metabolite and one indicator per core reaction.

Several growth conditions at once
---------------------------------

``extra_blocks`` gives each additional constraint list its own steady-state flux system sharing
the same binaries, so a network can be required to grow on several media simultaneously. Because
the conditions sit in one MILP, the completion pays for a reaction once no matter how many media
need it -- which is not what iterated single-condition gap-filling produces.

.. code-block:: python

    from straindesign.completion import compute_completion

    keep, status, obj = compute_completion(universe, module, cost=cost,
                                           extra_blocks=[['EX_glc__D_e >= -10', growth + ' >= 0.1'],
                                                         ['EX_ac_e >= -10', growth + ' >= 0.05']])
