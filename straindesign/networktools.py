#!/usr/bin/env python3
#
# Copyright 2022 Max Planck Insitute Magdeburg
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
#
"""Functions for metabolic network extension with GPR rules and strain design module handling.

Compression functions have been moved to straindesign.compression.
This module re-exports them for backwards compatibility.
"""

import ast
import hashlib
import io
import logging
import numpy as np
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from functools import wraps
from re import search
from typing import List

from cobra import Model, Metabolite, Reaction
from straindesign import fva, select_solver, avail_solvers, DisableLogger
from straindesign.names import *
from straindesign.parse_constr import parse_constraints

# =============================================================================
# LP Update Suppression
# =============================================================================

# Sentinel no-op: identity check tells us whether the class is already patched.
_SLC_NOOP = lambda self, *a, **kw: None


def _sb_noop(self, lb, ub):
    """No-op set_bounds: updates Python-side state only, skips solver."""
    self._lb = lb
    self._ub = ub


# -- Cobra-level suppression replacements ------------------------------------

def _suppressed_set_id(self, value):
    """Bypass solver variable rename: write _id and update DictList index."""
    old_id = self._id
    self._id = value
    if self._model is not None:
        dl = self._model.reactions
        old_index = dl._dict.pop(old_id, None)
        if old_index is not None:
            dl._dict[value] = old_index


def _suppressed_set_lb(self, value):
    """Bypass solver bounds update: write _lower_bound only."""
    self._lower_bound = value


def _suppressed_set_ub(self, value):
    """Bypass solver bounds update: write _upper_bound only."""
    self._upper_bound = value


def _suppressed_populate_solver(self, reaction_list=None, metabolite_list=None):
    """No-op during suppression: solver will be rebuilt on context exit."""
    pass


def _suppressed_update_variable_bounds(self):
    """No-op during suppression: solver bounds synced on context exit."""
    pass


class _SolverStub:
    """Stand-in returned by solver containers for keys missing during suppression.

    Hashable (used as dict key in set_linear_coefficients calls) and has
    no-op set_bounds / set_linear_coefficients so any downstream call is safe.
    """
    __slots__ = ('_id',)

    def __init__(self, name=''):
        self._id = name

    def set_bounds(self, *a, **kw):
        pass

    def set_linear_coefficients(self, *a, **kw):
        pass


_SOLVER_STUB = _SolverStub('__stub__')

_ORIG_CONTAINER_GETITEM = None  # saved Container.__getitem__


def _permissive_container_getitem(self, item):
    """Container.__getitem__ that returns a stub instead of raising KeyError."""
    try:
        return self._object_list[item]  # int/slice
    except TypeError:
        return self._dict.get(item, _SOLVER_STUB)


def _remove_reactions_direct(model, remove_set: set) -> None:
    """Remove reactions from model via direct DictList manipulation.

    Bypasses the solver entirely — used as the implementation behind
    ``_suppressed_remove_reactions``.
    """
    from cobra import DictList
    new_reactions = DictList()
    for rxn in model.reactions:
        if rxn not in remove_set:
            new_reactions.append(rxn)
        else:
            for met in list(rxn._metabolites.keys()):
                if rxn in met._reaction:
                    met._reaction.discard(rxn)
            rxn._model = None
    model.__dict__['reactions'] = new_reactions
    # Remove orphan metabolites
    mets_in_use = set()
    for rxn in model.reactions:
        mets_in_use.update(rxn._metabolites.keys())
    new_metabolites = DictList()
    for met in model.metabolites:
        if met in mets_in_use:
            new_metabolites.append(met)
        else:
            met._model = None
    model.__dict__['metabolites'] = new_metabolites


def _remove_metabolites_direct(model, remove_set: set) -> None:
    """Remove metabolites from model via direct DictList manipulation.

    Bypasses the solver entirely — used as the implementation behind
    ``_suppressed_remove_metabolites``.
    """
    from cobra import DictList
    for met in remove_set:
        for rxn in list(met._reaction):
            rxn._metabolites.pop(met, None)
        met._reaction.clear()
        met._model = None
    new_metabolites = DictList()
    for met in model.metabolites:
        if met not in remove_set:
            new_metabolites.append(met)
    model.__dict__['metabolites'] = new_metabolites


def _suppressed_remove_reactions(self, reactions, remove_orphans=False):
    """Bypass solver variable removal: direct DictList manipulation."""
    if not hasattr(reactions, '__iter__'):
        reactions = [reactions]
    remove_set = set()
    for rxn in reactions:
        if isinstance(rxn, str):
            rxn = self.reactions.get_by_id(rxn)
        remove_set.add(rxn)
    if remove_set:
        _remove_reactions_direct(self, remove_set)


def _suppressed_remove_metabolites(self, metabolite_list, destructive=False):
    """Bypass solver constraint removal: direct DictList manipulation."""
    if not hasattr(metabolite_list, '__iter__'):
        metabolite_list = [metabolite_list]
    remove_set = {m for m in metabolite_list if m.id in self.metabolites}
    if remove_set:
        _remove_metabolites_direct(self, remove_set)


# -- Saved originals (None = not suppressed) ----------------------------------

_ORIG_SLC = None   # (cls, method) for Constraint.set_linear_coefficients
_ORIG_SB = None    # (cls, method) for Variable.set_bounds
_ORIG_OSLC = None  # (cls, method) for Objective.set_linear_coefficients
_ORIG_COBRA = []   # list of (cls, attr_name, original) for cobra-level patches


def _suppress_lp_updates(model):
    """Patch optlang and cobra to skip all solver-touching operations.

    **Optlang level** (coefficient/bounds/objective updates):
      - Constraint.set_linear_coefficients → no-op
      - Variable.set_bounds → Python-side only
      - Objective.set_linear_coefficients → no-op

    **Cobra level** (property setters and model mutations):
      - Reaction._set_id_with_model → direct _id + DictList update
      - Reaction.lower_bound → direct _lower_bound
      - Reaction.upper_bound → direct _upper_bound
      - Reaction.update_variable_bounds → no-op
      - Model._populate_solver → no-op (rebuild on context exit)
      - Model.remove_reactions → direct list manipulation
      - Model.remove_metabolites → direct list manipulation
      - Container.__getitem__ → return stub for missing keys

    Safe to call when already suppressed (idempotent).  The real methods
    are saved on first call and restored by :func:`_restore_lp_updates`.
    """
    global _ORIG_SLC, _ORIG_SB, _ORIG_OSLC, _ORIG_COBRA
    try:
        if hasattr(model, 'problem'):
            prob = model.problem
            if hasattr(prob, 'Constraint'):
                cls_c = prob.Constraint
                if hasattr(cls_c, 'set_linear_coefficients'):
                    if cls_c.set_linear_coefficients is not _SLC_NOOP:
                        _ORIG_SLC = (cls_c, cls_c.set_linear_coefficients)
                        cls_c.set_linear_coefficients = _SLC_NOOP
            if hasattr(prob, 'Variable'):
                cls_v = prob.Variable
                if hasattr(cls_v, 'set_bounds'):
                    if cls_v.set_bounds is not _sb_noop:
                        _ORIG_SB = (cls_v, cls_v.set_bounds)
                        cls_v.set_bounds = _sb_noop
            if hasattr(prob, 'Objective'):
                cls_o = prob.Objective
                if hasattr(cls_o, 'set_linear_coefficients'):
                    if cls_o.set_linear_coefficients is not _SLC_NOOP:
                        _ORIG_OSLC = (cls_o, cls_o.set_linear_coefficients)
                        cls_o.set_linear_coefficients = _SLC_NOOP
    except Exception:
        pass

    # Cobra-level patches
    from cobra.core.reaction import Reaction
    from cobra.core.model import Model
    _ORIG_COBRA = []
    if Reaction._set_id_with_model is not _suppressed_set_id:
        _ORIG_COBRA.append((Reaction, '_set_id_with_model', Reaction._set_id_with_model))
        Reaction._set_id_with_model = _suppressed_set_id
    orig_lb = Reaction.__dict__.get('lower_bound')
    if orig_lb is not None and isinstance(orig_lb, property) and orig_lb.fset is not _suppressed_set_lb:
        _ORIG_COBRA.append((Reaction, 'lower_bound', orig_lb))
        Reaction.lower_bound = property(fget=orig_lb.fget, fset=_suppressed_set_lb, fdel=orig_lb.fdel)
    orig_ub = Reaction.__dict__.get('upper_bound')
    if orig_ub is not None and isinstance(orig_ub, property) and orig_ub.fset is not _suppressed_set_ub:
        _ORIG_COBRA.append((Reaction, 'upper_bound', orig_ub))
        Reaction.upper_bound = property(fget=orig_ub.fget, fset=_suppressed_set_ub, fdel=orig_ub.fdel)
    if Reaction.update_variable_bounds is not _suppressed_update_variable_bounds:
        _ORIG_COBRA.append((Reaction, 'update_variable_bounds', Reaction.update_variable_bounds))
        Reaction.update_variable_bounds = _suppressed_update_variable_bounds
    if Model._populate_solver is not _suppressed_populate_solver:
        _ORIG_COBRA.append((Model, '_populate_solver', Model._populate_solver))
        Model._populate_solver = _suppressed_populate_solver
    if Model.remove_reactions is not _suppressed_remove_reactions:
        _ORIG_COBRA.append((Model, 'remove_reactions', Model.remove_reactions))
        Model.remove_reactions = _suppressed_remove_reactions
    if Model.remove_metabolites is not _suppressed_remove_metabolites:
        _ORIG_COBRA.append((Model, 'remove_metabolites', Model.remove_metabolites))
        Model.remove_metabolites = _suppressed_remove_metabolites

    # Permissive solver container: return stub for missing keys
    global _ORIG_CONTAINER_GETITEM
    from optlang.container import Container
    if Container.__getitem__ is not _permissive_container_getitem:
        _ORIG_CONTAINER_GETITEM = Container.__getitem__
        Container.__getitem__ = _permissive_container_getitem


def _restore_lp_updates():
    """Restore original optlang and cobra methods.

    Safe to call when not suppressed (idempotent).
    """
    global _ORIG_SLC, _ORIG_SB, _ORIG_OSLC, _ORIG_COBRA, _ORIG_CONTAINER_GETITEM
    if _ORIG_SLC is not None:
        cls, method = _ORIG_SLC
        cls.set_linear_coefficients = method
        _ORIG_SLC = None
    if _ORIG_SB is not None:
        cls, method = _ORIG_SB
        cls.set_bounds = method
        _ORIG_SB = None
    if _ORIG_OSLC is not None:
        cls, method = _ORIG_OSLC
        cls.set_linear_coefficients = method
        _ORIG_OSLC = None
    for cls, attr_name, original in _ORIG_COBRA:
        setattr(cls, attr_name, original)
    _ORIG_COBRA = []
    if _ORIG_CONTAINER_GETITEM is not None:
        from optlang.container import Container
        Container.__getitem__ = _ORIG_CONTAINER_GETITEM
        _ORIG_CONTAINER_GETITEM = None


def _is_lp_suppressed():
    """Return True if LP updates are currently suppressed."""
    return _ORIG_SLC is not None or _ORIG_SB is not None or _ORIG_OSLC is not None or len(_ORIG_COBRA) > 0


@contextmanager
def suppress_lp_context(model):
    """Context manager that suppresses all solver-touching operations.

    Patches both optlang (coefficient/bounds updates) and cobra (property
    setters, Model.remove_reactions, Model.remove_metabolites) at the class
    level.  On exit, restores originals and rebuilds the solver so it
    reflects the current model state.

    Nests safely: if already suppressed by an outer context, this is a no-op.
    """
    entered_here = not _is_lp_suppressed()
    if entered_here:
        # Capture objective and reaction IDs before suppression (solver still live)
        obj_dict = {}
        for rxn in model.reactions:
            try:
                c = rxn.objective_coefficient
                if c != 0:
                    obj_dict[rxn.id] = float(c)
            except Exception:
                pass
        _pre_ids = {r.id for r in model.reactions}
        # Store on model so compression code can update it through maps
        model._suppressed_obj = dict(obj_dict)
        _suppress_lp_updates(model)
    try:
        yield
    finally:
        if entered_here:
            _restore_lp_updates()
            # Rebuild solver only if the model was modified during suppression.
            current_ids = {r.id for r in model.reactions}
            # Use the stored objective (compression may have updated it)
            final_obj = getattr(model, '_suppressed_obj', obj_dict)
            if hasattr(model, '_suppressed_obj'):
                del model._suppressed_obj
            if current_ids != _pre_ids:
                try:
                    solver_interface = model.solver.interface
                    model._solver = solver_interface.Model()
                    model._populate_solver(model.reactions, model.metabolites)
                    for rxn in model.reactions:
                        c = final_obj.get(rxn.id, 0.0)
                        if c != 0:
                            rxn.objective_coefficient = c
                except Exception:
                    pass


def with_suppressed_lp(func):
    """Decorator that wraps a function in :func:`suppress_lp_context`.

    The first positional argument must be a cobra Model.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        model = args[0]
        with suppress_lp_context(model):
            return func(*args, **kwargs)
    return wrapper


# =============================================================================
# I/O Suppression
# =============================================================================

@contextmanager
def _silent_io():
    """Suppress stdout, stderr and logging."""
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), DisableLogger():
        yield


# Re-export compression functions for backwards compatibility
from straindesign.compression import (
    compress_model,
    compress_model_coupled,
    compress_model_efmtool,  # backward-compat alias
    compress_model_parallel,
    remove_blocked_reactions,
    remove_ext_mets,
    remove_conservation_relations,
    remove_dummy_bounds,
    stoichmat_coeff2rational,
    stoichmat_coeff2float,
)


def evaluate_gpr_ast(node, gene_states):
    """Evaluate a GPR AST node with given gene states.

    Supports arbitrary nesting of AND/OR operators (not limited to DNF/CNF).
    Used by both gene_kos_to_constraints and reduce_gpr.

    Args:
        node: An ast.Name or ast.BoolOp node from a parsed GPR rule
              (e.g. reaction.gpr.body).
        gene_states (dict): Maps gene IDs to True, False, or None.
            Genes not in the dict are treated as None (undetermined).

    Returns:
        True if the GPR is satisfied, False if knocked out, None if undetermined.
    """
    if isinstance(node, ast.Name):
        return gene_states.get(node.id, None)
    elif isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            results = [evaluate_gpr_ast(child, gene_states) for child in node.values]
            if any(r is False for r in results):
                return False
            elif all(r is True for r in results):
                return True
            else:
                return None
        elif isinstance(node.op, ast.Or):
            results = [evaluate_gpr_ast(child, gene_states) for child in node.values]
            if any(r is True for r in results):
                return True
            elif all(r is False for r in results):
                return False
            else:
                return None
    raise ValueError(f"Unsupported AST node type: {type(node)}")


def gene_kos_to_constraints(model, gene_kos):
    """Translate gene knockouts to reaction-level constraints.

    Given a cobra model and a set of knocked-out genes, evaluates GPR rules
    (via AST parsing) to determine which reactions become non-functional.
    Returns a list of constraints (in straindesign list format) that fix
    those reactions to zero flux.

    Supports arbitrary GPR nesting (not limited to DNF/CNF).

    Gene identifiers can be gene IDs or gene names (case-sensitive).  Unknown
    identifiers are silently ignored.

    Note on the SD solution grammar:
        SDSolutions uses the following convention for intervention values:
        -1 = knockout (KO), +1 = knock-in (KI), 0 = non-added KI.

        When passing gene constraints to ``fba()`` or ``fva()``, both
        ``gene = 0`` and ``gene = -1`` are treated as knockouts, and
        ``gene = 1`` is ignored (the gene is active).  This means SD
        solution values can be used directly as constraints.

        The returned *reaction* constraints use the linear constraint
        format ``[{reaction_id: 1}, '=', 0]``, meaning ``1 * v = 0``.

    Example:
        >>> constraints = gene_kos_to_constraints(model, ['b0727', 'b1241'])
        >>> fba(model, constraints=constraints)

    Args:
        model (cobra.Model): A metabolic model.
        gene_kos (list or set): Gene IDs or names to knock out.

    Returns:
        list: Constraints in list format, e.g.
              [[{'AKGDH': 1}, '=', 0], [{'PDH': 1}, '=', 0], ...]
    """
    # Map gene names to IDs where needed
    gene_ids = set()
    name_to_id = {g.name: g.id for g in model.genes if g.name}
    for g in gene_kos:
        if g in name_to_id:
            gene_ids.add(name_to_id[g])
        elif any(mg.id == g for mg in model.genes):
            gene_ids.add(g)
        # else: unknown gene, skip

    if not gene_ids:
        return []

    # Build gene state dict: knocked-out genes → False
    gene_states = {g: False for g in gene_ids}

    # Find affected reactions via gene→reaction links
    affected_reacs = set()
    for g_id in gene_ids:
        try:
            gene_obj = model.genes.get_by_id(g_id)
            for r in gene_obj.reactions:
                affected_reacs.add(r.id)
        except KeyError:
            pass

    # Evaluate GPR rules using AST
    knocked_out_reactions = []
    for r_id in affected_reacs:
        reaction = model.reactions.get_by_id(r_id)
        if not reaction.gene_reaction_rule or not reaction.gene_reaction_rule.strip():
            continue
        if not reaction.gpr or not reaction.gpr.body:
            continue
        result = evaluate_gpr_ast(reaction.gpr.body, gene_states)
        if result is False:
            knocked_out_reactions.append(r_id)

    return [[{r: 1}, '=', 0] for r in sorted(knocked_out_reactions)]


def resolve_gene_constraints(model, constraints):
    """Scan constraints for gene IDs/names and replace with reaction constraints.

    Constraints that reference gene identifiers (rather than reaction IDs) are
    interpreted using the strain design solution grammar:

    - ``gene = -1`` — knockout (KO)
    - ``gene = 0``  — knockout (non-added, absence of knock-in)
    - ``gene = 1``  — ignored (knock-in, gene is active)

    This is consistent with the SDSolutions convention, so values from
    solution dicts can be passed directly as constraints.

    Non-gene constraints are passed through unchanged.

    Accepted gene constraint formats::

        'b0727 = 0'                          # string — KO
        'b0727 = -1'                         # string — KO (SD grammar)
        'b1241 = 1'                          # string — ignored (KI)
        ['b0727 = 0', 'b1241 = 0']           # list of strings
        [[{'b0727': 1}, '=', 0]]             # list format

    Gene identifiers are matched case-sensitively against both gene IDs
    and gene names in the model.

    This function is called automatically by ``fba()``, ``fva()``, and
    ``fva_legacy()`` before constraint parsing, so users can pass gene
    knockouts directly::

        fba(model, constraints='b0727 = 0')
        fva(model, constraints=['b0727 = 0', 'EX_o2_e <= 5'])

    Args:
        model (cobra.Model): A metabolic model.
        constraints: Constraints in any format accepted by straindesign
                     (str, list of str, or list of [dict, str, float]).

    Returns:
        list: Constraints with gene entries replaced by reaction entries.
    """
    from straindesign.parse_constr import lineq2list
    # Normalise to list format
    reac_ids = set(r.id for r in model.reactions)
    gene_ids = set(g.id for g in model.genes)
    gene_names = set(g.name for g in model.genes if g.name)
    all_ids = reac_ids | gene_ids | gene_names

    if not constraints:
        return []
    if isinstance(constraints, str):
        constraints = lineq2list([constraints], list(all_ids))
    elif isinstance(constraints, list) and constraints and isinstance(constraints[0], str):
        constraints = lineq2list(constraints, list(all_ids))
    # constraints is now list of [dict, str, float]

    gene_kos = set()
    clean_constraints = []
    for c in constraints:
        lhs, sign, rhs = c[0], c[1], c[2]
        keys = set(lhs.keys())
        # Check if this constraint references only genes (not reactions)
        gene_keys = keys & (gene_ids | gene_names)
        if gene_keys and not (keys & reac_ids):
            if sign == '=' and rhs <= 0:
                # KO: gene = 0 or gene = -1 (SD grammar)
                gene_kos.update(gene_keys)
            elif sign == '=' and rhs > 0:
                # KI: gene = 1 — gene is active, nothing to do
                pass
            else:
                # Inequality on gene — not meaningful
                logging.warning(f'Gene constraint with inequality ignored: {c}')
                clean_constraints.append(c)
        else:
            clean_constraints.append(c)

    if gene_kos:
        reaction_constraints = gene_kos_to_constraints(model, gene_kos)
        clean_constraints.extend(reaction_constraints)

    return clean_constraints


def reduce_gpr(model, essential_reacs, gkis, gkos):
    """Simplify GPR rules by removing non-targetable genes and reducing boolean expressions

    This function is used in preprocessing of computational strain design computations. Often,
    certain reactions, for instance, reactions essential for microbial growth can/must not be
    targeted by interventions. That can be exploited to reduce the set of genes in which
    interventions need to be considered.

    Given a set of essential reactions that is to be maintained operational, some genes can be
    removed from a metabolic model, either because they only affect only blocked reactions or
    essential reactions, or because they are essential reactions and must not be removed. As a
    consequence, the GPR rules of a model can be simplified using AST parsing for both DNF and non-DNF rules.


    Example:
        reduce_gpr(model, essential_reacs, gkis, gkos):
    
    Args:
        model (cobra.Model):
            A metabolic model that is an instance of the cobra.Model class containing GPR rules
            
        essential_reacs (list of str):
            A list of identifiers of essential reactions.
            
        gkis, gkos (dict):
            Dictionaries that contain the costs for gene knockouts and additions. E.g.,
            gkos={'adhE': 1.0, 'ldhA' : 1.0 ...}
            
    Returns:
        (dict):
        An updated dictionary of the knockout costs in which irrelevant genes are removed.
    """

    def ast_to_gene_reaction_rule(node):
        """
        Convert an AST node back to gene reaction rule string format.
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.BoolOp):
            child_strings = [ast_to_gene_reaction_rule(child) for child in node.values]
            if isinstance(node.op, ast.And):
                return ' and '.join(f'({s})' if ' or ' in s else s for s in child_strings)
            elif isinstance(node.op, ast.Or):
                return ' or '.join(f'({s})' if ' and ' in s else s for s in child_strings)
        else:
            raise ValueError(f"Unsupported AST node type: {type(node)}")

    def simplify_gpr_ast(node, protected_genes_dict):
        """
        Simplify GPR AST by setting protected genes to True and applying boolean simplification.
        This is equivalent to the original string-based approach but operates purely on AST.
        """
        return apply_gene_protection_to_ast(node, protected_genes_dict)

    def apply_gene_protection_to_ast(node, protected_genes_dict):
        """
        Apply gene protection to AST by setting protected genes to True and simplifying boolean expressions.
        Returns a simplified AST node with redundant terms removed and consistent gene ordering.
        """
        if isinstance(node, ast.Name):
            if node.id in protected_genes_dict:
                return True
            else:
                return node
        elif isinstance(node, ast.BoolOp):
            # Recursively apply to children
            new_children = []
            for child in node.values:
                simplified_child = apply_gene_protection_to_ast(child, protected_genes_dict)

                if isinstance(node.op, ast.And):
                    if simplified_child is False:
                        return False
                    elif simplified_child is not True:
                        new_children.append(simplified_child)
                elif isinstance(node.op, ast.Or):
                    if simplified_child is True:
                        return True
                    elif simplified_child is not False:
                        new_children.append(simplified_child)

            # Handle results
            if not new_children:
                return True if isinstance(node.op, ast.And) else False
            elif len(new_children) == 1:
                return new_children[0]
            else:
                # Apply additional simplifications for OR nodes
                if isinstance(node.op, ast.Or):
                    new_children = remove_redundant_or_terms(new_children)
                    if len(new_children) == 1:
                        return new_children[0]

                # Sort children for consistent ordering (like string approach does)
                sorted_children = sort_ast_nodes(new_children)
                new_node = ast.BoolOp(op=node.op, values=sorted_children)
                return new_node
        else:
            raise ValueError(f"Unsupported AST node type: {type(node)}")

    def remove_redundant_or_terms(children):
        """
        Remove redundant terms from OR expressions using boolean logic simplification.
        Example: (a and b and c) or (a and b) simplifies to (a and b)
        since (a and b) is logically sufficient when both terms are present.
        """
        # Convert AST nodes to comparable forms
        simplified = []
        for child in children:
            # Check if this child makes any other child redundant
            is_redundant = False
            for other in children:
                if child is not other and is_subset_of(child, other):
                    # child is a subset of other, so other is redundant
                    is_redundant = False  # Keep child, remove other later
                elif child is not other and is_subset_of(other, child):
                    # other is a subset of child, so child is redundant
                    is_redundant = True
                    break
            if not is_redundant:
                simplified.append(child)

        # Remove duplicates
        unique = []
        for child in simplified:
            if not any(ast_nodes_equal(child, existing) for existing in unique):
                unique.append(child)

        return unique if unique else children

    def is_subset_of(node1, node2):
        """
        Check if node1 logically absorbs node2 in boolean algebra.
        
        In OR expressions: A or (A and B) = A
        This means A absorbs (A and B) because A is simpler/more general.
        
        For absorption to work: node1 must be "simpler" than node2,
        meaning node2 implies node1 (node2 is more restrictive).
        
        Examples:
        - mobA absorbs (mobA and mobB) 
        - (a and b) absorbs (a and b and c)
        """
        # Case 1: Single gene absorbs AND expression containing that gene
        if isinstance(node1, ast.Name) and isinstance(node2, ast.BoolOp) and isinstance(node2.op, ast.And):
            genes_in_and = get_genes_from_ast(node2)
            return node1.id in genes_in_and

        # Case 2: Shorter AND expression absorbs longer AND expression with same genes
        if (isinstance(node1, ast.BoolOp) and isinstance(node1.op, ast.And) and isinstance(node2, ast.BoolOp) and
                isinstance(node2.op, ast.And)):
            genes1 = get_genes_from_ast(node1)
            genes2 = get_genes_from_ast(node2)
            # node1 absorbs node2 if node1's genes are a proper subset of node2's genes
            return genes1.issubset(genes2) and len(genes1) < len(genes2)

        return False

    def get_genes_from_ast(node):
        """Extract set of genes from AST node"""
        if isinstance(node, ast.Name):
            return {node.id}
        elif isinstance(node, ast.BoolOp):
            genes = set()
            for child in node.values:
                genes.update(get_genes_from_ast(child))
            return genes
        return set()

    def ast_nodes_equal(node1, node2):
        """Check if two AST nodes are equivalent"""
        if type(node1) != type(node2):
            return False
        if isinstance(node1, ast.Name):
            return node1.id == node2.id
        elif isinstance(node1, ast.BoolOp):
            if type(node1.op) != type(node2.op):
                return False
            return (len(node1.values) == len(node2.values) and all(ast_nodes_equal(a, b) for a, b in zip(node1.values, node2.values)))
        return False

    def sort_ast_nodes(nodes):
        """Sort AST nodes for consistent ordering"""

        def node_sort_key(node):
            if isinstance(node, ast.Name):
                return (0, node.id)
            elif isinstance(node, ast.BoolOp):
                return (1, len(node.values), str(type(node.op)))
            return (2, str(node))

        return sorted(nodes, key=node_sort_key)

    def is_gene_essential_to_reaction_ast(reaction, gene_id):
        """
        Determine if a gene is essential for a reaction using AST-based GPR analysis.
        A gene is considered essential if removing it (setting it to False) makes 
        the entire GPR expression evaluate to False, rendering the reaction impossible.
        """
        if not reaction.gene_reaction_rule:
            return False

        # Skip reactions without gene associations
        if not reaction.gpr or not reaction.gpr.body:
            return False

        try:
            # Test what happens if we knock out this gene using AST
            gene_states = {gene_id: False}
            result = evaluate_gpr_ast(reaction.gpr.body, gene_states)
            return result is False
        except Exception as e:
            # Catch unsupported AST node types but don't fall back to string parsing
            logging.warning(f'Unsupported AST node type in reaction {reaction.id} for gene {gene_id}: {e}')
            return False

    # 1) Remove gpr rules from blocked reactions
    blocked_reactions = [reac.id for reac in model.reactions if reac.bounds == (0, 0)]
    for rid in blocked_reactions:
        model.reactions.get_by_id(rid).gene_reaction_rule = ''
    for g in model.genes[::-1]:  # iterate in reverse order to avoid mixing up the order of the list when removing genes
        if not g.reactions:
            model.genes.remove(g)

    protected_genes = set()

    # 2. Protect genes that only occur in essential reactions
    for g in model.genes:
        if not g.reactions or {r.id for r in g.reactions}.issubset(essential_reacs):
            protected_genes.add(g)

    # 3. Protect genes that are essential to essential reactions (AST-based analysis)
    for r in [model.reactions.get_by_id(s) for s in essential_reacs]:
        for g in r.genes:
            if is_gene_essential_to_reaction_ast(r, g.id):
                protected_genes.add(g)

    # 4. Remove essential genes, and knockouts without impact from gko_costs
    [gkos.pop(pg.id) for pg in protected_genes if pg.id in gkos]

    # 5. Add all not-knockable genes to the protected list
    [protected_genes.add(g) for g in model.genes if (g.id not in gkos) and (g.name not in gkos)]  # support names or ids in gkos

    # 6. genes with kiCosts are kept (remove from protected list so they can be targeted)
    gki_ids = [g.id for g in model.genes if (g.id in gkis) or (g.name in gkis)]  # support names or ids in gkis
    protected_genes = protected_genes.difference({model.genes.get_by_id(g) for g in gki_ids})
    protected_genes_dict = {pg.id: True for pg in protected_genes}

    # 7. Simplify GPR rules using AST-based boolean logic and remove non-targetable rules
    for r in model.reactions:
        if r.gene_reaction_rule and r.gpr and r.gpr.body:
            try:
                simplified = simplify_gpr_ast(r.gpr.body, protected_genes_dict)

                if simplified is True:
                    # Rule is always satisfied (cannot be knocked out)
                    model.reactions.get_by_id(r.id).gene_reaction_rule = ''
                elif simplified is False:
                    # Rule is impossible - should not happen with proper protection
                    logging.error(f'Something went wrong during gpr rule simplification for {r.id}.')
                elif isinstance(simplified, (ast.Name, ast.BoolOp)):
                    # Convert simplified AST back to string
                    new_rule = ast_to_gene_reaction_rule(simplified)
                    model.reactions.get_by_id(r.id).gene_reaction_rule = new_rule
                # If simplified is the original node, keep original rule
            except Exception as e:
                logging.warning(f'Failed to simplify GPR rule for reaction {r.id}: {e}')

    # 8. Remove obsolete genes and protected genes
    for g in model.genes[::-1]:
        if not g.reactions or g in protected_genes:
            model.genes.remove(g)

    return gkos


# backward-compat alias
remove_irrelevant_genes = reduce_gpr


def extend_model_gpr(model, use_names=False):
    """Integrate GPR-rules into a metabolic model as pseudo metabolites and reactions using AST parsing
    
    COBRA modules often have gene-protein-reaction (GPR) rules associated with each reaction. 
    These can be integrated into the metabolic network structure through pseudo reactions
    and variables. As GPR rules are integrated into the metabolic network, the metabolic flux 
    space does not change. After integration, the gene-pseudoreactions can be fixed to a flux of 
    zero to simulate gene knockouts. Gene pseudoreactions are referenced either by the gene name 
    or the gene identifier (user selected).
    
    GPR-rule integration enables the computation of strain designs based on genetic interventions.
    
    This function now uses AST parsing to handle both DNF (disjunctive normal form) and non-DNF
    GPR rules. It processes the reaction.gpr.body AST structure directly, enabling proper handling
    of complex nested boolean expressions.
        
    Example:
        reac_map = extend_model_gpr(model):
    
    Args:
        model (cobra.Model):
            A metabolic model that is an instance of the cobra.Model class containing GPR rules 
                      
        use_names (bool): (Default: False)
            If set to True, the gene pseudoreactions will carry the gene name as reaction
            identifier. If False, the gene identifier will be used. By default this option is
            turned off because many models do not provide gene names.
            
    Returns:
        (dict):
        A dictionary to reference old and new reaction identifiers, for reversible reactions
        that were split (when they are associated with GPR rules). Entries have the form:
        {'Reaction1' : {'Reaction1' : 1, 'Reaction1_reverse_a59c' : -1}}
            
    """

    # Check if reaction names and gene names/IDs overlap. If yes, throw Error
    reac_ids = {r.id for r in model.reactions}
    if (not use_names) and any([g.id in reac_ids for g in model.genes]):
        raise Exception("GPR rule integration requires distinct identifiers for reactions and "+\
                        "genes.\nThe following identifiers seem to be both, reaction IDs and gene "+\
                        "IDs:\n"+str([g.id for g in model.genes if g.id in reac_ids])+\
                        "\nTo prevent this problem, call extend_model_gpr with the use_names=False "+\
                        "option or refer\nto gene names instead of IDs in your strain design "+\
                        "computation if they are available.\nYou may also rename the conflicting "+\
                        "genes with cobra.manipulation.modify.rename_genes(model, {'g_old': 'g_new'})")
    elif use_names and any([g.name in reac_ids for g in model.genes]):
        raise Exception("GPR rule integration requires distinct identifiers for reactions and "+\
                        "genes.\nThe following identifiers seem to be both, reaction IDs and gene "+\
                        "names:\n"+str([g.name for g in model.genes if g.name in reac_ids])+\
                        "\nTo prevent this problem, call extend_model_gpr with the use_names=False "+\
                        "option or refer \nto gene IDs instead of names in your strain design "+\
                        "computation. You may also rename\nthe conflicting genes, e.g., with "+\
                        "model.genes.get_by_id('ADK1').name = 'adk1'")

    MAX_NAME_LEN = 230

    def warning_name_too_long(id, p=""):
        logging.warning(" GPR rule integration automatically generates new reactions and metabolites."+\
                        "\nOne of the generated reaction names is beyond or close to the limit of 255 "+\
                        "characters\npermitted by GLPK and Gurobi. The name of the newly generated "+\
                        "reaction or metabolite: \n "+id+",\ngenerated from reaction or metabolite:\n "+\
                        p+"\n"+"was therefore trimmed to:\n "+id[0:MAX_NAME_LEN]+".\nThis trimming is "+\
                        "usually safe, no guarantee is given. To avoid this message,\nuse the CPLEX "+\
                        "solver or consider simplifying GPR rules or gene names in your model.")

    def truncate(id):
        h = hashlib.sha256(id.encode()).hexdigest()[:20]
        return id[0:MAX_NAME_LEN - 21] + "_" + h

    solver = search('(' + '|'.join(avail_solvers) + ')', model.solver.interface.__name__)[0]

    # Track created metabolites to avoid duplicates
    created_metabolites = set()

    def create_gene_pseudoreaction(gene_id):
        """Create a gene pseudoreaction and return the corresponding metabolite ID."""
        gene_met_id = f'g_{gene_id}'

        # Check name length and truncate if necessary
        if len(gene_met_id) > MAX_NAME_LEN and solver in {GUROBI, GLPK}:
            if truncate(gene_met_id) not in [m.id for m in model.metabolites]:
                warning_name_too_long(gene_met_id, gene_id)
            gene_met_id = truncate(gene_met_id)

        # Create metabolite and reaction if they don't exist
        if gene_met_id not in created_metabolites and gene_met_id not in model.metabolites.list_attr('id'):
            model.add_metabolites(Metabolite(gene_met_id))
            created_metabolites.add(gene_met_id)

            gene = model.genes.get_by_id(gene_id)
            if use_names and gene.name:
                reaction_id = gene.name
            else:
                reaction_id = gene.id

            # Check name length and truncate if necessary
            if len(reaction_id) > MAX_NAME_LEN and solver in {GUROBI, GLPK}:
                warning_name_too_long(reaction_id, gene_id)
                reaction_id = truncate(reaction_id)

            w = Reaction(reaction_id)
            model.add_reactions([w])
            w.reaction = f'--> {gene_met_id}'
            w._upper_bound = np.inf

        return gene_met_id

    def create_and_metabolite(child_metabolites):
        """Create an AND pseudometabolite that combines multiple child metabolites."""
        and_met_id = "_and_".join(sorted(child_metabolites))

        # Check name length and truncate if necessary
        if len(and_met_id) > MAX_NAME_LEN and solver in {GUROBI, GLPK}:
            if truncate(and_met_id) not in [m.id for m in model.metabolites]:
                warning_name_too_long(and_met_id, "AND combination")
            and_met_id = truncate(and_met_id)

        # Create metabolite and reaction if they don't exist
        if and_met_id not in created_metabolites and and_met_id not in model.metabolites.list_attr('id'):
            model.add_metabolites(Metabolite(and_met_id))
            created_metabolites.add(and_met_id)

            reaction_id = f"R_{and_met_id}"

            # Check name length and truncate if necessary
            if len(reaction_id) > MAX_NAME_LEN and solver in {GUROBI, GLPK}:
                warning_name_too_long(reaction_id, "AND combination")
                reaction_id = truncate(reaction_id)

            w = Reaction(reaction_id)
            model.add_reactions([w])
            w.reaction = f'{" + ".join(child_metabolites)} --> {and_met_id}'
            w._upper_bound = np.inf

        return and_met_id

    def create_or_metabolite(child_metabolites):
        """Create an OR pseudometabolite with separate reactions for each child."""
        or_met_id = "_or_".join(sorted(child_metabolites))

        # Check name length and truncate if necessary
        if len(or_met_id) > MAX_NAME_LEN and solver in {GUROBI, GLPK}:
            if truncate(or_met_id) not in [m.id for m in model.metabolites]:
                warning_name_too_long(or_met_id, "OR combination")
            or_met_id = truncate(or_met_id)

        # Create metabolite and reactions if they don't exist
        if or_met_id not in created_metabolites and or_met_id not in model.metabolites.list_attr('id'):
            model.add_metabolites(Metabolite(or_met_id))
            created_metabolites.add(or_met_id)

            # Create separate reactions for each child metabolite
            for i, child_met in enumerate(child_metabolites):
                reaction_id = f"R{i}_{or_met_id}"

                # Check name length and truncate if necessary
                if len(reaction_id) > MAX_NAME_LEN and solver in {GUROBI, GLPK}:
                    warning_name_too_long(reaction_id, "OR combination")
                    reaction_id = truncate(reaction_id)

                w = Reaction(reaction_id)
                model.add_reactions([w])
                w.reaction = f'{child_met} --> {or_met_id}'
                w._upper_bound = np.inf

        return or_met_id

    def process_ast_node(node):
        """Recursively process AST nodes to build GPR network."""
        if isinstance(node, ast.Name):
            # Leaf node: create gene pseudoreaction
            return create_gene_pseudoreaction(node.id)
        elif isinstance(node, ast.BoolOp):
            # Branch node: process children and create appropriate metabolite
            child_metabolites = [process_ast_node(child) for child in node.values]
            if isinstance(node.op, ast.And):
                return create_and_metabolite(child_metabolites)
            elif isinstance(node.op, ast.Or):
                return create_or_metabolite(child_metabolites)
        else:
            raise ValueError(f"Unsupported AST node type: {type(node)}")

    # Split reactions when necessary
    reac_map = {}
    rev_reac = set()
    del_reac = set()
    for r in model.reactions:
        reac_map.update({r.id: {}})
        if not r.gene_reaction_rule:
            reac_map[r.id].update({r.id: 1.0})
            continue
        if r.gene_reaction_rule and r.bounds[0] < 0:
            r_rev = (r * -1)
            if r.gene_reaction_rule and r.bounds[1] > 0:
                r_rev.id = r.id + '_reverse_' + hex(hash(r))[8:]
            r_rev.lower_bound = np.max([0, r_rev.lower_bound])
            reac_map[r.id].update({r_rev.id: -1.0})
            if len(r_rev.id) > MAX_NAME_LEN and solver in {GUROBI, GLPK}:
                warning_name_too_long(r_rev.id, r.id)
                r_rev.id = truncate(r_rev.id)
            rev_reac.add(r_rev)
        if r.gene_reaction_rule and r.bounds[1] > 0:
            reac_map[r.id].update({r.id: 1.0})
            r._lower_bound = np.max([0, r._lower_bound])
        else:
            del_reac.add(r)
    model.remove_reactions(del_reac)
    model.add_reactions(rev_reac)

    # Process GPR rules using AST parsing
    for r in model.reactions:
        if r.gene_reaction_rule and r.gpr and r.gpr.body:
            try:
                # Process the GPR AST and get the final metabolite ID
                final_metabolite_id = process_ast_node(r.gpr.body)
                # Connect the final metabolite to the reaction
                r.add_metabolites({model.metabolites.get_by_id(final_metabolite_id): -1.0})
            except Exception as e:
                logging.warning(f"Failed to process GPR rule for reaction {r.id}: {e}")
                # Fallback to string parsing for compatibility
                dt = [s.strip() for s in r.gene_reaction_rule.split(' or ')]
                for i, p in enumerate(dt.copy()):
                    ct = [s.strip() for s in p.replace('(', '').replace(')', '').split(' and ')]
                    for j, g in enumerate(ct.copy()):
                        gene_met_id = create_gene_pseudoreaction(g)
                        ct[j] = gene_met_id
                    if len(ct) > 1:
                        ct_met_id = create_and_metabolite(ct)
                        dt[i] = ct_met_id
                    else:
                        dt[i] = gene_met_id
                if len(dt) > 1:
                    dt_met_id = create_or_metabolite(dt)
                else:
                    dt_met_id = dt[0]
                r.add_metabolites({model.metabolites.get_by_id(dt_met_id): -1.0})

    return reac_map


def extend_model_regulatory(model, reg_itv):
    """Extend a metabolic model to account for regulatory constraints
    
    This function emulates regulatory interventions in a network. These can either be added
    permanently or linked to a pseudoreation whose boundaries can be fixed to zero used 
    to activate the regulatory constraint.
    
    Accounting for regulatory interventions, such as applying an upper or lower bound
    to a reaction or gene pseudoreaction, can be achieved by combining different
    pseudometabolites and reactions. For instance, to introduce the regulatory constraint:
    
    2*r_1 + 3*r_2 <= 4
    
    and make it 'toggleable', one adds 1 metabolite 'm' and 2 reactions, 'r_bnd' to account for
    the bound/rhs and r_ctl to control whether the regulatory intervention is active or not:
    
    dm/dt = 2*r_1 + 3*r_2 - r_bnd + r_ctl = 0, -inf <= r_bnd <= 4, -inf <= r_ctl <= inf
    
    When r_ctl is fixed to zero, the constraint 2*r_1 + 3*r_2 <= 4 is enforced, otherwise,
    the constraint is non binding, thus virtually non-existant. To use this mechanism for 
    strain design, we add the metabolite and reactions as described above and tag r_ctl as 
    knockout candidate. If the algorithm decides to knockout r_ctl, this means, it choses to 
    add the regulatory intervention 2*r_1 + 3*r_2 <= 4.
    
    If the constraint is be added permanently, this function completely omits the r_ctl reaction.
    
    Example:
        reg_itv_costs = extend_model_regulatory(model, {'1 PDH + 1 PFL <= 5' : 1, '-EX_o2_e <= 2' : 1.5})
    
    Args:
        model (cobra.Model):
            A metabolic model that is an instance of the cobra.Model class
            
        reg_itv (dict or list of str or str):
            A set of regulatory constraints that should be added to the model. If reg_itv is a
            string or a list of strings, regulatory constraints are added permanently. If reg_itv
            is a dict, regulatory interventions are added in a controllable manner. The id of the
            reaction that controls the constraint is contained in the return variable. The constraint
            to be added will be parsed from strings, so ensure that you use the correct reaction
            identifiers. Valid inputs are:
            reg_itv = '-EX_o2_e <= 2' # A single permanent regulatory constraint
            reg_itv = ['1 PDH + 1 PFL <= 5', '-EX_o2_e <= 2'] # Two permanent constraints
            reg_itv = {'1 PDH + 1 PFL <= 5' : 1, '-EX_o2_e <= 2' : 1.5} # Two controllable constraints
            # one costs '1' and the other one '1.5' to be added. The function returns a dict with 
            # {'p1_PDH_p1_PFK_le_5' : 1 'nEX_o2_e_le_2' : 1.5}. Fixing the reaction 
            # p1_PDH_p1_PFK_le_5 to zero will activate the constraint in the model.
            
    Returns:
        (dict):
        A dictionary that contains the cost of adding each constraint; e.g.,
        {'p1_PDH_p1_PFK_le_5' : 1 'n1EX_o2_e_le_2' : 1.5}
    """
    keywords = set(model.reactions.list_attr('id'))
    if '' in keywords:
        keywords.remove('')
    if not isinstance(reg_itv, dict):
        if not isinstance(reg_itv, str) and not isinstance(reg_itv, list):
            raise Exception('reg_itv must be a string, a list or a dictionary.'\
                            'See function description for details.')
        if isinstance(reg_itv, str):
            reg_itv = {reg_itv: np.nan}
        elif isinstance(reg_itv, list):
            reg_itv = {r: np.nan for r in reg_itv}
    regcost = {}
    for k, v in reg_itv.copy().items():
        # generate name for regulatory pseudoreaction
        try:
            constr = parse_constraints(k, keywords)[0]
        except:
            raise Exception('Regulatory constraints could not be parsed. Please revise.')
        reacs_dict = constr[0]
        eqsign = constr[1]
        rhs = constr[2]
        reg_name = ''
        for l, w in reacs_dict.items():
            if w < 0:
                reg_name += 'n' + str(w) + '_' + l
            else:
                reg_name += 'p' + str(w) + '_' + l
            reg_name += '_'
        if eqsign == '=':
            reg_name += 'eq_'
        elif eqsign == '<=':
            reg_name += 'le_'
        elif eqsign == '>=':
            reg_name += 'ge_'
        reg_name += str(rhs).replace('-', 'n').replace('.', 'p')
        reg_itv.pop(k)
        reg_itv.update({reg_name: {'str': k, 'cost': v}})
        reg_pseudomet_name = 'met_' + reg_name
        # add pseudometabolite
        m = Metabolite(reg_pseudomet_name)
        model.add_metabolites(m)
        # add pseudometabolite to stoichiometries
        for l, w in reacs_dict.items():
            r = model.reactions.get_by_id(l)
            r.add_metabolites({m: w})
        # add pseudoreaction that defines the bound
        r_bnd = Reaction("bnd_" + reg_name)
        model.add_reactions([r_bnd])
        r_bnd.reaction = reg_pseudomet_name + ' --> '
        if eqsign == '=':
            r_bnd._lower_bound = -np.inf
            r_bnd._upper_bound = rhs
            r_bnd._lower_bound = rhs
        elif eqsign == '<=':
            r_bnd._lower_bound = -np.inf
            r_bnd._upper_bound = rhs
        elif eqsign == '>=':
            r_bnd._upper_bound = np.inf
            r_bnd._lower_bound = rhs
        # add knockable pseudoreaction and add it to the kocost list
        if not np.isnan(v):
            r_ctl = Reaction(reg_name)
            model.add_reactions([r_ctl])
            r_ctl.reaction = '--> ' + reg_pseudomet_name
            r_ctl._upper_bound = np.inf
            r_ctl._lower_bound = -np.inf
            regcost.update({reg_name: v})
    return regcost


# =============================================================================
# Strain Design Module Compression
# =============================================================================


def compress_modules(sd_modules, cmp_mapReac):
    """Compress strain design modules to match with a compressed model
    
    When a strain design task has been specified with modules and the original metabolic model 
    was compressed, one needs to refit the strain design modules (objects of the SDModule class)
    to the new compressed model. This function takes a list of modules and a compression map
    and returns the strain design modules for a compressed network.
    
    Example:
        comression_map = compress_modules(sd_modules, cmp_mapReac)
    
    Args:
        model (list of SDModule):
            A list of strain design modules
        
        cmp_mapReac (list of dicts):
            Compression map obtained from cmp_mapReac = compress_model(model)
        
    Returns:
        (list of SDModule):
        A list of strain design modules for the compressed network
    """
    sd_modules = modules_coeff2rational(sd_modules)
    for cmp in cmp_mapReac:
        reac_map_exp = cmp["reac_map_exp"]
        parallel = cmp["parallel"]
        if not parallel:
            for new_reac, old_reac_val in reac_map_exp.items():
                for i, m in enumerate(sd_modules):
                    for p in [CONSTRAINTS, INNER_OBJECTIVE, OUTER_OBJECTIVE, PROD_ID, MIN_GCP]:
                        if p in m and m[p] is not None:
                            param = m[p]
                            if p == CONSTRAINTS:
                                for j, c in enumerate(m[p]):
                                    if np.any([k in old_reac_val for k in c[0].keys()]):
                                        lumped_reacs = [k for k in c[0].keys() if k in old_reac_val]
                                        c[0][new_reac] = np.sum([c[0].pop(k) * old_reac_val[k] for k in lumped_reacs])
                            if p in [INNER_OBJECTIVE, OUTER_OBJECTIVE, PROD_ID]:
                                if np.any([k in old_reac_val for k in param.keys()]):
                                    lumped_reacs = [k for k in param.keys() if k in old_reac_val]
                                    m[p][new_reac] = np.sum([param.pop(k) * old_reac_val[k] for k in lumped_reacs if k in old_reac_val])
    return sd_modules


def compress_ki_ko_cost(kocost, kicost, cmp_mapReac):
    """Compress knockout/addition cost vectors to match with a compressed model
    
    When knockout/addition cost vectors have been specified (as dicts) and the original
    metabolic model was compressed, one needs to update the knockout/addition cost vectors.
    This function takes care of this. In particular it makes sure that the resulting costs
    are calculated correctly. 
    
    E.g.: r_ko_a (cost 1) and r_ko_b (cost 2) are lumped parallel: The resulting cost of r_ko_ab is 3
    If they are lumped as dependent reactions the resulting cost is 1. If one of the two reactions
    is an addition candidate, the resulting reaction will be an addition candidate when lumped as 
    dependent reactions and a knockout candidate when lumped in parallel. There are various possible
    cases that are treated by this function.
    
    Example:
        kocost, kicost, cmp_mapReac = compress_ki_ko_cost(kocost, kicost, cmp_mapReac)
    
    Args:
        kocost, kicost (dict):
            Knockout and addition cost vectors
        
        cmp_mapReac (list of dicts):
            Compression map obtained from cmp_mapReac = compress_model(model)
        
    Returns:
        (Tuple):
        Updated vectors of KO costs and KI costs and an updated compression map that contains information
        on how to expand strain designs and correctly distinguish between knockouts and additions.
    """
    # kocost of lumped reactions: when reacs sequential: lowest of ko costs, when parallel: sum of ko costs
    # kicost of lumped reactions: when reacs sequential: sum of ki costs, when parallel: lowest of ki costs
    for cmp in cmp_mapReac:
        reac_map_exp = cmp["reac_map_exp"]
        parallel = cmp["parallel"]
        cmp.update({KOCOST: kocost, KICOST: kicost})
        if kocost:
            ko_cost_new = {}
            for r in reac_map_exp:
                if np.any([s in kocost for s in reac_map_exp[r]]):
                    if not parallel and not np.any([s in kicost for s in reac_map_exp[r]]):
                        ko_cost_new[r] = np.min([kocost[s] for s in reac_map_exp[r] if s in kocost])
                    elif parallel:
                        ko_cost_new[r] = np.sum([kocost[s] for s in reac_map_exp[r] if s in kocost])
            kocost = ko_cost_new
        if kicost:
            ki_cost_new = {}
            for r in reac_map_exp:
                if np.any([s in kicost for s in reac_map_exp[r]]):
                    if not parallel:
                        ki_cost_new[r] = np.sum([kicost[s] for s in reac_map_exp[r] if s in kicost])
                    elif parallel and not np.any([s in kocost for s in reac_map_exp[r]]):
                        ki_cost_new[r] = np.min([kicost[s] for s in reac_map_exp[r] if s in kicost])
            kicost = ki_cost_new
    return kocost, kicost, cmp_mapReac


def estimate_expansion_size(compressed_sds, cmp_mapReac):
    """Estimate total expanded solutions without doing the expansion.

    Walks cmp_mapReac in reverse for each compressed solution. For each
    compression step, for each reaction that maps to multiple originals:
    - KO + parallel -> factor 1 (all ko'd together)
    - KO + coupled  -> factor = count of knockable originals
    - KI + parallel -> factor = count of KI-able originals
    - KI + coupled  -> factor 1

    Returns int (upper bound, exact for single-step compression).
    """
    if not cmp_mapReac:
        return len(compressed_sds)

    total = 0
    cmp_map = cmp_mapReac[::-1]
    for cmp_s in compressed_sds:
        # Track keys and their values through expansion steps
        key_vals = dict(cmp_s)
        factor = 1
        for exp in cmp_map:
            reac_map_exp = exp["reac_map_exp"]
            ko_cost = exp[KOCOST]
            ki_cost = exp[KICOST]
            par_reac_cmp = exp["parallel"]
            updates = {}
            removals = set()
            for r_cmp, r_orig in reac_map_exp.items():
                if r_cmp not in key_vals:
                    continue
                val = key_vals[r_cmp]
                removals.add(r_cmp)
                if len(r_orig) > 1:
                    if val < 0:  # KO
                        if not par_reac_cmp:  # coupled
                            knockable = sum(1 for d in r_orig if d in ko_cost)
                            factor *= max(knockable, 1)
                    elif val > 0:  # KI
                        if par_reac_cmp:  # parallel
                            ki_able = sum(1 for d in r_orig if d in ki_cost)
                            factor *= max(ki_able, 1)
                for d in r_orig:
                    updates[d] = val
            for r in removals:
                key_vals.pop(r, None)
            key_vals.update(updates)
        total += factor
    return total


def expand_sd(sd, cmp_mapReac):
    """Expand computed strain designs from a compressed to a full model
    
    Needed after computing strain designs in a compressed model
    
    Example:
        expanded_sd = expand_sd(compressed_sds, cmp_mapReac)
    
    Args:
        sd (SDSolutions):
            Solutions of a strain design computation that refer to a compressed model
        
        cmp_mapReac (list of dicts):
            Compression map obtained from cmp_mapReac = compress_model(model) and updated with
            kocost, kicost, cmp_mapReac = compress_ki_ko_cost(kocost, kicost, cmp_mapReac)
        
    Returns:
        (SDSolutions):
        Strain design solutions that refer to the uncompressed model
    """
    # expand mcs by applying the compression steps in the reverse order
    cmp_map = cmp_mapReac[::-1]
    for exp in cmp_map:
        reac_map_exp = exp["reac_map_exp"]  # expansion map
        ko_cost = exp[KOCOST]
        ki_cost = exp[KICOST]
        par_reac_cmp = exp["parallel"]  # if parallel or sequential reactions were lumped
        for r_cmp, r_orig in reac_map_exp.items():
            if len(r_orig) > 1:
                for m in sd.copy():
                    if r_cmp in m:
                        val = m[r_cmp]
                        del m[r_cmp]
                        if val < 0:  # case: KO
                            if par_reac_cmp:
                                new_m = m.copy()
                                for d in r_orig:
                                    if d in ko_cost:
                                        new_m[d] = val
                                sd += [new_m]
                            else:
                                for d in r_orig:
                                    if d in ko_cost:
                                        new_m = m.copy()
                                        new_m[d] = val
                                        sd += [new_m]
                        elif val > 0:  # case: KI
                            if par_reac_cmp:
                                for d in r_orig:
                                    if d in ki_cost:
                                        new_m = m.copy()
                                        new_m[d] = val
                                        # other reactions do not need to be knocked in
                                        for f in [e for e in r_orig if (e in ki_cost) and e != d]:
                                            new_m[f] = 0.0
                                        sd += [new_m]
                            else:
                                new_m = m.copy()
                                for d in r_orig:
                                    if d in ki_cost:
                                        new_m[d] = val
                                sd += [new_m]
                        elif val == 0:  # case: KI that was not introduced
                            new_m = m.copy()  # assume that none of the expanded
                            for d in r_orig:  # reactions are inserted, neither
                                if d in ki_cost:  # parallel, nor sequential
                                    new_m[d] = val
                            sd += [new_m]
                        sd.remove(m)
    return sd


def filter_sd_maxcost(sd, max_cost, kocost, kicost):
    """Filter out strain designs that exceed the maximum allowed intervention costs
    
    Returns:
        (SDSolutions):
        Strain design solutions complying with the intervention costs limit
    """
    # eliminate mcs that are too expensive
    if max_cost:
        costs = [np.sum([kocost[k] if v < 0 else (kicost[k] if v > 0 else 0) for k, v in m.items()]) for m in sd]
        sd = [sd[i] for i in range(len(sd)) if costs[i] <= max_cost + 1e-8]
        # sort strain designs by intervention costs
        [s.update({'**cost**': c}) for s, c in zip(sd, costs)]
        sd.sort(key=lambda x: x.pop('**cost**'))
    return sd


def modules_coeff2rational(sd_modules):
    """Convert coefficients to rational numbers using sympy.Rational"""
    from .compression import float_to_rational
    for i, module in enumerate(sd_modules):
        for param in [CONSTRAINTS, INNER_OBJECTIVE, OUTER_OBJECTIVE, PROD_ID]:
            if param in module and module[param] is not None:
                if param == CONSTRAINTS:
                    for constr in module[CONSTRAINTS]:
                        for reac in constr[0].keys():
                            constr[0][reac] = float_to_rational(constr[0][reac])
                if param in [INNER_OBJECTIVE, OUTER_OBJECTIVE, PROD_ID]:
                    for reac in module[param].keys():
                        module[param][reac] = float_to_rational(module[param][reac])
    return sd_modules


def modules_coeff2float(sd_modules):
    """Convert coefficients occurring in SDModule objects to floats"""
    for i, module in enumerate(sd_modules):
        for param in [CONSTRAINTS, INNER_OBJECTIVE, OUTER_OBJECTIVE, PROD_ID]:
            if param in module and module[param] is not None:
                if param == CONSTRAINTS:
                    for constr in module[CONSTRAINTS]:
                        for reac in constr[0].keys():
                            constr[0][reac] = float(constr[0][reac])
                if param in [INNER_OBJECTIVE, OUTER_OBJECTIVE, PROD_ID]:
                    for reac in module[param].keys():
                        module[param][reac] = float(module[param][reac])
    return sd_modules


def bound_blocked_or_irrevers_fva(model, **kwargs):
    """Use FVA to determine the flux ranges. Use this information to update the model bounds

    If flux ranges for a reaction are narrower than its bounds in the mode, these bounds can be omitted,
    since other reactions must constrain the reaction flux. If (upper or lower) flux bounds are found to
    be zero, the model bounds are updated to reduce the model complexity.
    """
    solver = kwargs.get('solver', None)
    # FVAs to identify blocked and irreversible reactions, as well as non-bounding bounds
    flux_limits = fva(model, **kwargs)
    if select_solver(solver) in [SCIP, GLPK]:
        tol = 1e-10  # use tolerance for tightening problem bounds
    else:
        tol = 0.0
    for (reac_id, limits) in flux_limits.iterrows():
        r = model.reactions.get_by_id(reac_id)
        # modify _lower_bound and _upper_bound to make changes permanent
        if r.lower_bound < 0.0 and limits.minimum - tol > r.lower_bound:
            r._lower_bound = -np.inf
        if limits.minimum >= tol:
            r._lower_bound = max([0.0, r._lower_bound])
        if r.upper_bound > 0.0 and limits.maximum + tol < r.upper_bound:
            r._upper_bound = np.inf
        if limits.maximum <= -tol:
            r._upper_bound = min([0.0, r._upper_bound])
