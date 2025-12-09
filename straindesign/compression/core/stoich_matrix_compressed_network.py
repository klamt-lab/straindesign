#!/usr/bin/env python3
"""
EFMTool StoichMatrixCompressedMetabolicNetwork Implementation

Python port of ch.javasoft.metabolic.compress.StoichMatrixCompressedMetabolicNetwork
Compressed metabolic network with uncompression capabilities using stoichiometric matrices.

From Java source: efmtool_source/ch/javasoft/metabolic/compress/StoichMatrixCompressedMetabolicNetwork.java
Ported line-by-line for exact compatibility
"""

from typing import List, Optional, Dict, Union, TYPE_CHECKING
import threading

from .compressed_metabolic_network import CompressedMetabolicNetwork
from ..network.impl.fraction_stoich_network import FractionNumberStoichMetabolicNetwork
from ..network.impl.abstract_reaction import AbstractReaction
from ..network.impl.default_reaction_constraints import DEFAULT_CONSTRAINTS_NAN

if TYPE_CHECKING:
    from ..math.bigint_rational_matrix import BigIntegerRationalMatrix
    from ..math.big_fraction import BigFraction
    from ..network.metabolic_network import MetabolicNetwork
    from ..network.metabolite import Metabolite
    from ..network.reaction import Reaction
    from ..network.flux_distribution import FluxDistribution
    from ..network.metabolite_ratio import MetaboliteRatio
    from ..util.int_int_multi_value_map import IntIntMultiValueMap
    from ..network.reaction_constraints import ReactionConstraints
    from ..network.metabolic_network_visitor import MetabolicNetworkVisitor


class StoichMatrixCompressedMetabolicNetwork(FractionNumberStoichMetabolicNetwork, CompressedMetabolicNetwork):
    """
    StoichMatrixCompressedMetabolicNetwork extends the fraction number
    stoichiometric matrix network by uncompression methods.
    
    This class maintains:
    - Parent network for hierarchy tracking
    - Post matrix for flux uncompression (original_reactions x compressed_reactions)
    - Compressed stoichiometric matrix (compressed_metabolites x compressed_reactions)
    - Sparse post matrix for efficient uncompression
    """
    
    def __init__(self, stoich_orig_or_names, 
                 orig_reversible_or_reaction_names=None,
                 pre_or_orig_reversible=None, 
                 post_or_pre=None, 
                 compressed_stoich_or_post=None,
                 orig_stoich=None,
                 post=None,
                 compressed_stoich=None):
        """
        Create StoichMatrixCompressedMetabolicNetwork
        
        Multiple constructor patterns supported:
        1) (stoich_orig: BigIntegerRationalMatrix, orig_reversible: List[bool], 
           pre: BigIntegerRationalMatrix, post: BigIntegerRationalMatrix, 
           compressed_stoich: BigIntegerRationalMatrix)
        2) (orig_metabolite_names: List[str], orig_reaction_names: List[str], 
           orig_reversible: List[bool], pre: BigIntegerRationalMatrix, 
           orig_stoich: BigIntegerRationalMatrix, post: BigIntegerRationalMatrix, 
           compressed_stoich: BigIntegerRationalMatrix)
        3) (original: MetabolicNetwork, pre: BigIntegerRationalMatrix, 
           post: BigIntegerRationalMatrix, compressed_stoich: BigIntegerRationalMatrix)
        """
        if (len([x for x in [orig_stoich, post] if x is not None]) == 2 and compressed_stoich is None and
            isinstance(stoich_orig_or_names, list) and isinstance(orig_reversible_or_reaction_names, list)):
            # Pattern 2: Full names constructor
            # (orig_metabolite_names, orig_reaction_names, orig_reversible, pre, orig_stoich, post, compressed_stoich)
            orig_metabolite_names = stoich_orig_or_names
            orig_reaction_names = orig_reversible_or_reaction_names
            orig_reversible = pre_or_orig_reversible
            pre = post_or_pre
            # Map the positional parameters correctly:
            # pos 4 (compressed_stoich_or_post) -> orig_stoich  
            # pos 5 (orig_stoich) -> post
            # pos 6 (post) -> compressed_stoich
            actual_orig_stoich = compressed_stoich_or_post
            actual_post = orig_stoich  
            actual_compressed_stoich = post
            
            # Create parent network from original data
            parent_network = FractionNumberStoichMetabolicNetwork(
                orig_metabolite_names, orig_reaction_names, actual_orig_stoich, orig_reversible
            )
            # Set the matrices for later use
            pre = pre
            post = actual_post
            compressed_stoich = actual_compressed_stoich
        elif compressed_stoich_or_post is not None and isinstance(stoich_orig_or_names, list):
            # Error case - should not happen with correct parameters
            raise ValueError("Ambiguous constructor parameters")
        elif hasattr(stoich_orig_or_names, 'get_metabolites'):
            # Pattern 3: Network-based constructor
            # (original_network, pre, post, compressed_stoich)
            parent_network = stoich_orig_or_names
            pre = orig_reversible_or_reaction_names
            post = pre_or_orig_reversible
            compressed_stoich = post_or_pre
        else:
            # Pattern 1: Matrix-only constructor
            # (stoich_orig, orig_reversible, pre, post, compressed_stoich)
            stoich_orig = stoich_orig_or_names
            orig_reversible = orig_reversible_or_reaction_names
            pre = pre_or_orig_reversible
            post = post_or_pre
            compressed_stoich = compressed_stoich_or_post
            
            # Create parent network from original stoich matrix
            parent_network = FractionNumberStoichMetabolicNetwork(stoich_orig, orig_reversible)
        
        # Initialize parent class with compressed network data
        metabolite_names = self._get_metabolite_names(parent_network, pre, post, compressed_stoich)
        reaction_names = self._get_reaction_names(parent_network, pre, post, compressed_stoich)
        reversible = self._get_reversible(parent_network, pre, post, compressed_stoich)
        
        super().__init__(metabolite_names, reaction_names, compressed_stoich, reversible)
        
        # Store compression data
        self._parent_network = parent_network
        self._post = post
        self._stoich_compressed = compressed_stoich
        
        # Sparse post matrix for efficient uncompression (lazy initialization)
        self._sparse_post_indices: Optional[List[List[int]]] = None
        self._sparse_post_fractions: Optional[List[List['BigFraction']]] = None
        self._sparse_post_doubles: Optional[List[List[float]]] = None
        self._init_lock = threading.RLock()  # Use reentrant lock to prevent deadlock
    
    def get_parent_network(self) -> 'MetabolicNetwork':
        """Get the immediate parent network"""
        return self._parent_network
    
    def get_root_network(self) -> 'MetabolicNetwork':
        """Get the root (original, uncompressed) network"""
        if isinstance(self._parent_network, CompressedMetabolicNetwork):
            return self._parent_network.get_root_network()
        return self._parent_network
    
    def get_reaction_reversibilities(self) -> List[bool]:
        """
        Get reaction reversibility flags for compressed reactions.
        
        Returns:
            List of boolean flags indicating if each compressed reaction is reversible
        """
        return self._reversible
    
    def get_mapped_metabolites(self, original: List['Metabolite']) -> List[Optional['Metabolite']]:
        """
        Returns mapped metabolites in the same order as input.
        None if metabolite was removed (dead end).
        """
        result = []
        for metabolite in original:
            index = self.get_metabolite_index(metabolite.get_name())
            result.append(metabolite if index >= 0 else None)
        return result
    
    def get_mapped_reactions(self, original: List['Reaction']) -> List[Optional['Reaction']]:
        """
        Returns mapped reactions in the same order as input.
        None if reaction was removed (zero flux).
        """
        result = []
        for reaction in original:
            orig_index = self._parent_network.get_reaction_index(reaction.get_name())
            mappings = []
            
            # Find all compressed reactions that this original reaction maps to
            for new_index in range(self._stoich_compressed.get_column_count()):
                if self._post.get_signum_at(orig_index, new_index) != 0:
                    mappings.append(self.get_reactions()[new_index])
            
            if len(mappings) == 0:
                result.append(None)
            elif len(mappings) == 1:
                result.append(mappings[0])
            else:
                # Multiple mappings - create multiplexed reaction
                result.append(MultiplexedReaction())
        
        return result
    
    def uncompress_flux_distribution(self, flux_distribution: 'FluxDistribution') -> 'FluxDistribution':
        """
        Uncompress flux distribution to parent network space.
        See StoichMatrixCompressor.compress() for details on uncompression.
        """
        # Choose appropriate uncompression method based on preferred number type
        if flux_distribution.get_preferred_number_class() == float:
            uncompressed = self._uncompress_flux_distribution_double(flux_distribution)
        else:
            uncompressed = self._uncompress_flux_distribution_fractional(flux_distribution)
        
        # Recursively uncompress if parent is also compressed
        if isinstance(self._parent_network, CompressedMetabolicNetwork):
            return self._parent_network.uncompress_flux_distribution(uncompressed)
        
        return uncompressed
    
    def get_reaction_mapping(self) -> 'IntIntMultiValueMap':
        """
        Returns reaction mapping as one-to-many map.
        Key: original reaction index, Values: compressed reaction indices.
        """
        self._init_sparse_post_matrix()
        
        from ..util.int_int_multi_value_map import DefaultIntIntMultiValueMap
        mapping = DefaultIntIntMultiValueMap()
        
        # Avoid calling get_reactions() to prevent circular dependency
        if hasattr(self._parent_network, '_reaction_names'):
            orig_reaction_count = len(self._parent_network._reaction_names)
        else:
            orig_reaction_count = self._post.get_row_count()
            
        for orig_index in range(orig_reaction_count):
            mapping.add_all(orig_index, self._sparse_post_indices[orig_index])
        
        return mapping
    
    def _init_sparse_post_matrix(self) -> None:
        """Initialize sparse representation of post matrix for efficient access"""
        if self._sparse_post_indices is not None:
            return
        
        with self._init_lock:
            if self._sparse_post_indices is not None:
                return
            
            # CRITICAL FIX: Avoid calling self.get_reactions() which may cause infinite recursion
            # Instead, use the dimensions directly from matrices and reaction names
            comp_reaction_count = self._stoich_compressed.get_column_count()
            
            # For original reactions, also avoid get_reactions() to prevent circular dependency
            if hasattr(self._parent_network, '_reaction_names'):
                orig_reaction_count = len(self._parent_network._reaction_names)
            else:
                # Fallback to stoichiometry matrix dimensions
                orig_reaction_count = self._post.get_row_count()
            
            
            self._sparse_post_indices = []
            self._sparse_post_fractions = []
            
            for orig_index in range(orig_reaction_count):
                indices = []
                fractions = []
                
                for comp_index in range(comp_reaction_count):
                    if self._post.get_signum_at(orig_index, comp_index) != 0:
                        indices.append(comp_index)
                        fractions.append(self._post.get_big_fraction_value_at(orig_index, comp_index))
                
                self._sparse_post_indices.append(indices)
                self._sparse_post_fractions.append(fractions)
    
    def _init_sparse_post_matrix_double(self) -> None:
        """Initialize double version of sparse post matrix for faster computation"""
        if self._sparse_post_doubles is not None:
            return
        
        with self._init_lock:
            if self._sparse_post_doubles is not None:
                return
            
            self._init_sparse_post_matrix()
            
            self._sparse_post_doubles = []
            for fractions in self._sparse_post_fractions:
                # Ensure proper conversion of BigFraction to float
                doubles = []
                for frac in fractions:
                    if hasattr(frac, 'get_double'):
                        doubles.append(frac.get_double())
                    elif hasattr(frac, '__float__'):
                        doubles.append(float(frac))
                    else:
                        # Fallback for other numeric types
                        doubles.append(float(frac))
                self._sparse_post_doubles.append(doubles)
    
    def _uncompress_flux_distribution_fractional(self, comp_flux_dist: 'FluxDistribution') -> 'FluxDistribution':
        """
        Fraction number version of flux distribution uncompression.
        This is the default method for exact computation.
        """
        self._init_sparse_post_matrix()
        
        # Avoid calling get_reactions() to prevent circular dependency
        orig_flux_dist = comp_flux_dist.create(self._parent_network)
        orig_reaction_count = len(self._sparse_post_indices)
        
        from ..math.big_fraction import BigFraction
        
        for orig_index in range(orig_reaction_count):
            orig_flux = BigFraction.ZERO
            
            indices = self._sparse_post_indices[orig_index]
            fractions = self._sparse_post_fractions[orig_index]
            
            for i in range(len(indices)):
                comp_index = indices[i]
                comp_flux = BigFraction.value_of(comp_flux_dist.get_number_rate(comp_index))
                
                if not comp_flux.is_zero():
                    multiplier = fractions[i]
                    addition = comp_flux.multiply(multiplier)
                    orig_flux = orig_flux.add(addition).reduce()
            
            orig_flux_dist.set_rate(orig_index, orig_flux)
        
        return orig_flux_dist
    
    def _uncompress_flux_distribution_double(self, comp_flux_dist: 'FluxDistribution') -> 'FluxDistribution':
        """
        Double version of flux distribution uncompression for faster computation.
        Used when the resulting flux values are doubles anyway.
        """
        self._init_sparse_post_matrix_double()
        
        # Avoid calling get_reactions() to prevent circular dependency
        orig_flux_dist = comp_flux_dist.create(self._parent_network)
        max_orig_reactions = len(self._sparse_post_indices)
        comp_rates = comp_flux_dist.get_double_rates()
        max_comp_reactions = len(comp_rates)
        
        for orig_index in range(max_orig_reactions):
            orig_flux = 0.0
            
            if orig_index >= len(self._sparse_post_indices):
                break  # Safety check
            
            indices = self._sparse_post_indices[orig_index]
            doubles = self._sparse_post_doubles[orig_index]
            
            for i in range(len(indices)):
                comp_index = indices[i]
                
                # Bounds checking
                if comp_index >= max_comp_reactions:
                    continue
                
                comp_flux = comp_rates[comp_index]
                
                if comp_flux != 0.0:
                    orig_flux += comp_flux * doubles[i]
            
            # Use set_rate instead of direct array access for proper update
            orig_flux_dist.set_rate(orig_index, orig_flux)
        
        return orig_flux_dist
    
    @staticmethod
    def _get_reversible(original: 'MetabolicNetwork', pre: 'BigIntegerRationalMatrix', 
                       post: 'BigIntegerRationalMatrix', stoich_compressed: 'BigIntegerRationalMatrix') -> List[bool]:
        """
        Determine reversibility for compressed reactions.
        A compressed reaction is reversible only if ALL original reactions that map to it are reversible.
        """
        reversible = []
        
        # Get original reversibilities from the network
        orig_reversibilities = original.get_reaction_reversibilities()
        
        for new_index in range(stoich_compressed.get_column_count()):
            is_reversible = True
            
            for old_index in range(len(orig_reversibilities)):
                if post.get_signum_at(old_index, new_index) != 0:
                    if not orig_reversibilities[old_index]:
                        is_reversible = False
                        break
            
            reversible.append(is_reversible)
        
        return reversible
    
    @staticmethod
    def _get_metabolite_names(original: 'MetabolicNetwork', pre: 'BigIntegerRationalMatrix',
                             post: 'BigIntegerRationalMatrix', stoich_compressed: 'BigIntegerRationalMatrix') -> List[str]:
        """
        Generate metabolite names for compressed network.
        Composite names use '::' separator for metabolites that were combined.
        """
        names = []
        
        for new_index in range(stoich_compressed.get_row_count()):
            name_parts = []
            
            orig_metabolites = original.get_metabolites()
            for old_index in range(len(orig_metabolites)):
                if pre.get_signum_at(new_index, old_index) != 0:
                    name_parts.append(orig_metabolites[old_index].get_name())
            
            names.append("::".join(name_parts))
        
        return names
    
    @staticmethod
    def _get_reaction_names(original: 'MetabolicNetwork', pre: 'BigIntegerRationalMatrix',
                           post: 'BigIntegerRationalMatrix', stoich_compressed: 'BigIntegerRationalMatrix') -> List[str]:
        """
        Generate reaction names for compressed network.
        Composite names use '::' separator for reactions that were combined.
        """
        names = []
        
        for new_index in range(stoich_compressed.get_column_count()):
            name_parts = []
            
            orig_reactions = original.get_reactions()
            for old_index in range(len(orig_reactions)):
                if post.get_signum_at(old_index, new_index) != 0:
                    name_parts.append(orig_reactions[old_index].get_name())
            
            names.append("::".join(name_parts))
        
        return names


class MultiplexedReaction(AbstractReaction):
    """
    A reaction that has been mapped to multiple compressed reactions.
    Happens with uniquely consumed/produced metabolites.
    This is a dummy class used only in get_mapped_reactions().
    """
    
    def __init__(self):
        super().__init__()
    
    def get_constraints(self) -> 'ReactionConstraints':
        """Return NaN constraints for multiplexed reactions"""
        return DEFAULT_CONSTRAINTS_NAN
    
    def get_metabolite_ratios(self) -> List['MetaboliteRatio']:
        """Return empty ratios for multiplexed reactions"""
        return []
    
    def get_name(self) -> str:
        """Generate unique name using object hash"""
        return f"MultiplexedReaction[@{id(self)}]"
    
    def __hash__(self) -> int:
        """Use object identity for hashing"""
        return id(self)