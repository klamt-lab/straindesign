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
import logging
import numpy as np
from re import search
from typing import List

from cobra import Model, Metabolite, Reaction
from straindesign import fva, select_solver, avail_solvers
from straindesign.names import *
from straindesign.parse_constr import parse_constraints

# Re-export compression functions for backwards compatibility
from straindesign.compression import (
    compress_model,
    compress_model_efmtool,
    compress_model_parallel,
    remove_blocked_reactions,
    remove_ext_mets,
    remove_conservation_relations,
    remove_dummy_bounds,
    stoichmat_coeff2rational,
    stoichmat_coeff2float,
)


def remove_irrelevant_genes(model, essential_reacs, gkis, gkos):
    """Remove genes whose that do not affect the flux space of the model using AST-based GPR parsing
    
    This function is used in preprocessing of computational strain design computations. Often,
    certain reactions, for instance, reactions essential for microbial growth can/must not be
    targeted by interventions. That can be exploited to reduce the set of genes in which 
    interventions need to be considered.
    
    Given a set of essential reactions that is to be maintained operational, some genes can be 
    removed from a metabolic model, either because they only affect only blocked reactions or 
    essential reactions, or because they are essential reactions and must not be removed. As a
    consequence, the GPR rules of a model can be simplified using AST parsing for both DNF and non-DNF rules.
    
        
    Example:
        remove_irrelevant_genes(model, essential_reacs, gkis, gkos):
    
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

    def evaluate_gpr_ast(node, gene_states):
        """
        Evaluate GPR AST with given gene states.
        gene_states: dict of {gene_id: True/False/None}
        Returns True, False, or None (undetermined)
        """
        if isinstance(node, ast.Name):
            return gene_states.get(node.id, None)
        elif isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                # AND: all children must be True, any False makes it False
                results = [evaluate_gpr_ast(child, gene_states) for child in node.values]
                if any(r is False for r in results):
                    return False
                elif all(r is True for r in results):
                    return True
                else:
                    return None  # undetermined
            elif isinstance(node.op, ast.Or):
                # OR: any child True makes it True, all False makes it False
                results = [evaluate_gpr_ast(child, gene_states) for child in node.values]
                if any(r is True for r in results):
                    return True
                elif all(r is False for r in results):
                    return False
                else:
                    return None  # undetermined
        else:
            raise ValueError(f"Unsupported AST node type: {type(node)}")

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
        return id[0:MAX_NAME_LEN - 16] + hex(abs(hash(id)))[2:]

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
    sd_modules = modules_coeff2float(sd_modules)
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
    from .flint_cmp_interface import float_to_rational
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


def bound_blocked_or_irrevers_fva(model, solver=None):
    """Use FVA to determine the flux ranges. Use this information to update the model bounds
    
    If flux ranges for a reaction are narrower than its bounds in the mode, these bounds can be omitted, 
    since other reactions must constrain the reaction flux. If (upper or lower) flux bounds are found to 
    be zero, the model bounds are updated to reduce the model complexity.
    """
    # FVAs to identify blocked and irreversible reactions, as well as non-bounding bounds
    flux_limits = fva(model)
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
