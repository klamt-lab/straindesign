#!/usr/bin/env python3
"""Test script to validate GPR rule processing improvements.

This script provides a testing framework for the extend_model_gpr function by:
1. Testing both DNF and non-DNF GPR rule formats
2. Comparing original vs improved GPR extension functions
3. Using manual GPR extension with reaction-based strain design calculation
4. Validating that results are equivalent across all approaches
"""

import cobra
import straindesign as sd
from straindesign.names import *
import numpy as np
from re import search
import logging
from cobra import Metabolite, Reaction

def extend_model_gpr_new(model, use_names=False):
    """Enhanced GPR integration using AST parsing instead of string parsing.
    
    This version processes GPR rules using the AST structure from reaction.gpr.body
    to properly handle non-DNF GPR rules without converting them to strings first.
    
    Args:
        model (cobra.Model): A metabolic model with GPR rules
        use_names (bool): Use gene names instead of IDs for pseudoreactions
            
    Returns:
        dict: Mapping of old to new reaction identifiers for split reversible reactions
    """
    import ast
    from straindesign import avail_solvers
    
    # Check if reaction names and gene names/IDs overlap
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

    # Cache for created metabolites and reactions to enable reuse
    created_metabolites = {}
    created_reactions = {}
    
    def get_gene_reaction_id(gene_id):
        """Get the reaction ID for a gene (name or id based on use_names)."""
        if use_names:
            return model.genes.get_by_id(gene_id).name
        return gene_id
    
    def create_gene_pseudoreaction(gene_id):
        """Create the basic gene pseudoreaction: gene_id -> g_gene_id"""
        gene_met_id = 'g_' + gene_id
        gene_reac_id = get_gene_reaction_id(gene_id)
        
        # Check name length
        if len(gene_met_id) > MAX_NAME_LEN and solver in {GUROBI, GLPK}:
            if truncate(gene_met_id) not in [m.id for m in model.metabolites]:
                warning_name_too_long(gene_met_id, gene_reac_id)
            gene_met_id = truncate(gene_met_id)
        
        # Create metabolite if not exists
        if gene_met_id not in model.metabolites.list_attr('id'):
            model.add_metabolites(Metabolite(gene_met_id))
        
        # Create reaction if not exists  
        if gene_reac_id not in created_reactions:
            # Check name length
            if len(gene_reac_id) > MAX_NAME_LEN and solver in {GUROBI, GLPK}:
                warning_name_too_long(gene_reac_id, gene_reac_id)
                gene_reac_id = truncate(gene_reac_id)
            
            w = Reaction(gene_reac_id)
            model.add_reactions([w])
            w.reaction = '--> ' + gene_met_id
            w._upper_bound = np.inf
            created_reactions[gene_reac_id] = w
        
        return gene_met_id
    
    def create_and_metabolite(metabolite_ids):
        """Create AND metabolite: met1 + met2 + ... -> met1_and_met2_and_..."""
        and_met_id = "_and_".join(sorted(metabolite_ids))
        
        if len(and_met_id) > MAX_NAME_LEN and solver in {GUROBI, GLPK}:
            if truncate(and_met_id) not in [m.id for m in model.metabolites]:
                warning_name_too_long(and_met_id, "AND combination")
            and_met_id = truncate(and_met_id)
        
        if and_met_id not in model.metabolites.list_attr('id'):
            model.add_metabolites(Metabolite(and_met_id))
            
            # Create the combining reaction
            and_reac_id = "R_" + and_met_id
            if len(and_reac_id) > MAX_NAME_LEN and solver in {GUROBI, GLPK}:
                warning_name_too_long(and_reac_id, "AND reaction")
                and_reac_id = truncate(and_reac_id)
            
            w = Reaction(and_reac_id)
            model.add_reactions([w])
            w.reaction = ' + '.join(metabolite_ids) + ' --> ' + and_met_id
            w._upper_bound = np.inf
        
        return and_met_id
    
    def create_or_metabolite(metabolite_ids, base_name=""):
        """Create OR metabolite with multiple pathways: met1 -> or_met, met2 -> or_met, ..."""
        or_met_id = "_or_".join(sorted(metabolite_ids))
        
        if len(or_met_id) > MAX_NAME_LEN and solver in {GUROBI, GLPK}:
            if truncate(or_met_id) not in [m.id for m in model.metabolites]:
                warning_name_too_long(or_met_id, "OR combination")
            or_met_id = truncate(or_met_id)
        
        if or_met_id not in model.metabolites.list_attr('id'):
            model.add_metabolites(Metabolite(or_met_id))
            
            # Create separate pathway reactions for each input
            for k, met_id in enumerate(metabolite_ids):
                or_reac_id = f"R{k}_{or_met_id}"
                if len(or_reac_id) > MAX_NAME_LEN and solver in {GUROBI, GLPK}:
                    warning_name_too_long(or_reac_id, "OR reaction")
                    or_reac_id = truncate(or_reac_id)
                
                w = Reaction(or_reac_id)
                model.add_reactions([w])
                w.reaction = met_id + ' --> ' + or_met_id
                w._upper_bound = np.inf
        
        return or_met_id
    
    def process_ast_node(node):
        """Recursively process AST nodes to build GPR network."""
        if isinstance(node, ast.Name):
            # Leaf node - create gene pseudoreaction
            return create_gene_pseudoreaction(node.id)
        
        elif isinstance(node, ast.BoolOp):
            # Process all child nodes first
            child_metabolites = [process_ast_node(child) for child in node.values]
            
            if isinstance(node.op, ast.And):
                # AND: all genes needed -> create combining reaction
                return create_and_metabolite(child_metabolites)
            
            elif isinstance(node.op, ast.Or):
                # OR: alternative pathways -> create multiple pathway reactions
                return create_or_metabolite(child_metabolites)
        
        else:
            raise Exception(f"Unsupported AST node type: {type(node)}")

    # Split reactions when necessary (same logic as original)
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

    # Process GPR rules using AST
    for r in model.reactions:
        if r.gene_reaction_rule and r.gpr and r.gpr.body:
            # Process the AST to get the final metabolite for this reaction
            final_metabolite_id = process_ast_node(r.gpr.body)
            
            # Connect the final metabolite to the reaction
            r.add_metabolites({model.metabolites.get_by_id(final_metabolite_id): -1.0})
            
    return reac_map

def create_test_models():
    """Create test models with different GPR rule formats.
    
    Returns:
        tuple: (dnf_model, non_dnf_model, converted_dnf_model)
    """
    # Load the original DNF model
    dnf_model = cobra.io.read_sbml_model('tests/model_gpr.xml')
    
    # Create non-DNF model
    non_dnf_model = cobra.io.read_sbml_model('tests/model_gpr.xml')
    for r in non_dnf_model.reactions:
        r.gene_reaction_rule = ''
    
    # Set GPR rules in non-DNF format to test GPR processing
    non_dnf_model.reactions.get_by_id('r1').gene_reaction_rule = 'g1 and (g2 or g3)'
    non_dnf_model.reactions.get_by_id('r2').gene_reaction_rule = 'g4 or g2'
    non_dnf_model.reactions.get_by_id('r3').gene_reaction_rule = 'g8 or (g3 and g6)'
    non_dnf_model.reactions.get_by_id('r4').gene_reaction_rule = 'g8 and (g1 or g4)'
    non_dnf_model.reactions.get_by_id('r5').gene_reaction_rule = '(g1 and g7) or (g1 and g5 and g9)'
    non_dnf_model.reactions.get_by_id('r6').gene_reaction_rule = 'g1 and g4'
    non_dnf_model.reactions.get_by_id('r7').gene_reaction_rule = 'g7 or g6'
    
    # Create manually converted DNF model for comparison
    converted_dnf_model = non_dnf_model.copy()
    converted_dnf_model.reactions.get_by_id('r1').gene_reaction_rule = '(g1 and g2) or (g1 and g3)'
    converted_dnf_model.reactions.get_by_id('r4').gene_reaction_rule = '(g8 and g1) or (g8 and g4)'
    
    return dnf_model, non_dnf_model, converted_dnf_model

def run_strain_design_with_manual_gpr(model, test_name, use_new_function=False):
    """Test strain design with manual GPR extension.
    
    This function replicates what compute_strain_designs() does internally:
    1. Extends the model with GPR pseudoreactions
    2. Maps gene costs to reaction costs
    3. Runs reaction-based strain design
    
    Args:
        model: COBRA model to test
        test_name: Description for logging
        use_new_function: Whether to use extend_model_gpr_new (True) or original (False)
        
    Returns:
        SDSolutions object or None if failed
    """
    print(f"\n--- {test_name} ---")
    model_copy = model.copy()
    
    # Step 1: GPR extension
    try:
        if use_new_function:
            reac_map = extend_model_gpr_new(model_copy)
        else:
            reac_map = sd.extend_model_gpr(model_copy)
        print(f"✓ GPR extension successful")
    except Exception as e:
        print(f"✗ GPR extension failed: {e}")
        return None
    
    # Step 2: Cost mapping
    kocost = {'rs_up': 1.0, 'rd_ex': 1.0, 'rp_ex': 1.1, 'r_bm': 0.75}
    gkocost = {'g1': 1.0, 'g2': 1.0, 'g4': 3.0, 'g5': 2.0, 'g6': 1.0, 'g7': 1.0, 'g8': 1.0, 'g9': 1.0}
    gkicost = {'g3': 1.0}
    regcost = {'g4 <= 0.4': 1.2}
    
    # Map gene costs to reaction costs
    reaction_ids = [r.id for r in model_copy.reactions]
    gene_reaction_ids = [g for g in ['g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8', 'g9'] if g in reaction_ids]
    
    for gene_id, cost in gkocost.items():
        if gene_id in reaction_ids:
            kocost[gene_id] = cost
    
    kicost = {}
    for gene_id, cost in gkicost.items():
        if gene_id in reaction_ids:
            kicost[gene_id] = cost
    
    mapped_ko = len([g for g in gkocost if g in reaction_ids])
    mapped_ki = len([g for g in gkicost if g in reaction_ids])
    print(f"✓ Mapped {mapped_ko}/9 gene knockout costs, {mapped_ki}/1 gene knockin costs")
    
    if mapped_ko == 0 and mapped_ki == 0:
        print(f"  Available gene reactions: {gene_reaction_ids[:5]}{'...' if len(gene_reaction_ids) > 5 else ''}")
        print(f"  Note: No gene costs mapped - this may cause the test to fail")
    
    # Step 3: Strain design computation
    modules = [sd.SDModule(model_copy, SUPPRESS, constraints=["1.0 rd_ex >= 1.0"])]
    modules += [sd.SDModule(model_copy, PROTECT, constraints=[[{'r_bm': 1.0}, '>=', 1.0]])]
    
    sd_setup = {
        MODULES: modules,
        MAX_COST: 2,
        MAX_SOLUTIONS: 100,
        SOLUTION_APPROACH: BEST,
        KOCOST: kocost,
        KICOST: kicost,
        REGCOST: regcost,
        SOLVER: 'cplex'
    }
    
    try:
        solution = sd.compute_strain_designs(model_copy, sd_setup=sd_setup)
        num_solutions = len(solution.reaction_sd)
        print(f"✓ Found {num_solutions} strain design solutions")
        
        # Show first few solutions for verification
        for i, sol in enumerate(solution.get_reaction_sd()[:4]):
            print(f"  Solution {i+1}: {dict(list(sol.items())[:4])}{'...' if len(sol) > 4 else ''}")
        if num_solutions > 4:
            print(f"  ... and {num_solutions - 4} more solutions")
            
        return solution
    except Exception as e:
        print(f"✗ Strain design failed: {e}")
        return None

def run_baseline_gene_test(model):
    """Run baseline gene-based strain design that exactly replicates test_mcs_gpr."""
    print("\n--- Baseline: Gene-based strain design (replicating test_mcs_gpr) ---")
    
    # Replicate exactly what test_mcs_gpr does
    modules = [sd.SDModule(model, SUPPRESS, constraints=["1.0 rd_ex >= 1.0 "])]
    modules += [sd.SDModule(model, PROTECT, constraints=[[{'r_bm': 1.0}, '>=', 1.0]])]
    
    kocost = {'rs_up': 1.0, 'rd_ex': 1.0, 'rp_ex': 1.1, 'r_bm': 0.75}
    gkocost = {
        'g1': 1.0,
        'g2': 1.0,
        'g4': 3.0,
        'g5': 2.0,
        'g6': 1.0,
        'g7': 1.0,
        'g8': 1.0,
        'g9': 1.0,
    }
    gkicost = {
        'g3': 1.0,
    }
    regcost = {'g4 <= 0.4': 1.2}
    
    sd_setup = {
        MODULES: modules,
        MAX_COST: 2,
        MAX_SOLUTIONS: 100,
        SOLUTION_APPROACH: BEST,
        KOCOST: kocost,
        GKOCOST: gkocost,
        GKICOST: gkicost,
        REGCOST: regcost,
        SOLVER: 'cplex'
    }
    
    try:
        solution = sd.compute_strain_designs(model, sd_setup=sd_setup)
        num_solutions = len(solution.gene_sd)
        print(f"✓ Found {num_solutions} gene-based strain design solutions")
        
        # Show first few solutions
        for i, sol in enumerate(solution.get_gene_sd()[:4]):
            print(f"  Solution {i+1}: {dict(list(sol.items())[:4])}{'...' if len(sol) > 4 else ''}")
        if num_solutions > 4:
            print(f"  ... and {num_solutions - 4} more solutions")
            
        return solution
    except Exception as e:
        print(f"✗ Baseline test failed: {e}")
        print(f"   This is unexpected since test_mcs_gpr works")
        return None

def compare_solutions(baseline, test_results):
    """Compare solution counts between baseline and test results."""
    baseline_count = len(baseline.gene_sd) if baseline else 0
    
    print(f"\n=== Results Summary ===")
    print(f"Baseline gene-based solutions: {baseline_count}")
    
    for name, result in test_results:
        if result:
            count = len(result.reaction_sd)
            status = "✓" if count == baseline_count else "⚠"
            print(f"{status} {name}: {count} solutions")
        else:
            print(f"✗ {name}: Failed")

def main():
    """Test framework for GPR rule processing improvements.
    
    Tests three specific cases:
    1. Original DNF model + original straindesign function (should yield 4 solutions)  
    2. Original DNF model + new GPR extension function (should yield 4 identical solutions)
    3. Non-DNF model + new GPR extension function (should fail or not produce 4 solutions)
    """
    print("=" * 60)
    print("GPR Rule Processing Test Framework")
    print("=" * 60)
    
    # Load test models
    dnf_model, non_dnf_model, converted_dnf_model = create_test_models()
    print(f"✓ Loaded test models")
    
    # Test Case 1: Original DNF model with standard gene-based strain design
    print(f"\n{'='*60}")
    print("TEST CASE 1: DNF model + original straindesign function")
    print("Expected: 4 solutions (this is our baseline/target)")
    print(f"{'='*60}")
    
    # Baseline test with full regulatory constraints (should work now with correct package)
    try:
        modules = [sd.SDModule(dnf_model, SUPPRESS, constraints=['1.0 rd_ex >= 1.0 '])]
        modules += [sd.SDModule(dnf_model, PROTECT, constraints=[[{'r_bm': 1.0}, '>=', 1.0]])]
        
        sd_setup = {
            MODULES: modules,
            MAX_COST: 2,
            MAX_SOLUTIONS: 100,
            SOLUTION_APPROACH: BEST,
            KOCOST: {'rs_up': 1.0, 'rd_ex': 1.0, 'rp_ex': 1.1, 'r_bm': 0.75},
            GKOCOST: {'g1': 1.0, 'g2': 1.0, 'g4': 3.0, 'g5': 2.0, 'g6': 1.0, 'g7': 1.0, 'g8': 1.0, 'g9': 1.0},
            GKICOST: {'g3': 1.0},
            REGCOST: {'g4 <= 0.4': 1.2},
            SOLVER: 'cplex'
        }
        
        baseline_result = sd.compute_strain_designs(dnf_model, sd_setup=sd_setup)
        baseline_count = len(baseline_result.gene_sd)
        print(f"✓ Found {baseline_count} gene-based strain design solutions")
        
        # Show first few solutions
        for i, sol in enumerate(baseline_result.get_gene_sd()[:4]):
            print(f"  Solution {i+1}: {dict(list(sol.items())[:4])}{'...' if len(sol) > 4 else ''}")
    except Exception as e:
        print(f"✗ Baseline test failed: {e}")
        baseline_result = None
        baseline_count = 0
    
    # Test Case 2: DNF model with new GPR extension (should match baseline)
    print(f"\n{'='*60}")  
    print("TEST CASE 2: DNF model + new GPR extension function")
    print("Expected: 4 solutions (identical to baseline)")
    print(f"{'='*60}")
    dnf_new_result = run_strain_design_with_manual_gpr(dnf_model, "DNF + new function", use_new_function=True)
    dnf_new_count = len(dnf_new_result.reaction_sd) if dnf_new_result else 0
    
    # Test Case 3: Non-DNF model with new GPR extension (should fail or differ)
    print(f"\n{'='*60}")
    print("TEST CASE 3: Non-DNF model + new GPR extension function") 
    print("Expected: Should fail or not produce 4 solutions (we're implementing this)")
    print(f"{'='*60}")
    non_dnf_result = run_strain_design_with_manual_gpr(non_dnf_model, "Non-DNF + new function", use_new_function=True)
    non_dnf_count = len(non_dnf_result.reaction_sd) if non_dnf_result else 0
    
    # Results summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Case 1 (DNF + original):     {baseline_count} solutions {'✓' if baseline_count == 4 else '✗'}")
    print(f"Case 2 (DNF + new):          {dnf_new_count} solutions {'✓' if dnf_new_count == baseline_count else '✗'}")  
    print(f"Case 3 (Non-DNF + new):      {non_dnf_count} solutions {'✓ (unexpected!)' if non_dnf_count == baseline_count else '✓ (expected difference)'}")
    
    print(f"\n{'='*60}")
    if baseline_count == 4 and dnf_new_count == 4:
        print("✓ FRAMEWORK READY: Both DNF cases work correctly")
        print("✓ Now implement non-DNF support in extend_model_gpr_new()")
    elif baseline_count == 4:
        print("⚠ ISSUE: DNF baseline works but new function doesn't match")
        print("→ Fix extend_model_gpr_new() to match original behavior first")  
    else:
        print("✗ ISSUE: Baseline DNF case doesn't work - check test setup")
    print(f"{'='*60}")
    
    return {
        'baseline': baseline_result,
        'dnf_new': dnf_new_result, 
        'non_dnf': non_dnf_result
    }

if __name__ == "__main__":
    main()
