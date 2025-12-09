"""
EFMTool Compressed Metabolic Network Interface

Python port of ch.javasoft.metabolic.compress.CompressedMetabolicNetwork
Interface for compressed metabolic networks with uncompression capabilities.

From Java source: efmtool_source/ch/javasoft/metabolic/compress/CompressedMetabolicNetwork.java
Ported line-by-line for exact compatibility
"""

from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..network.metabolite import Metabolite
    from ..network.reaction import Reaction
    from ..network.metabolic_network import MetabolicNetwork
    from ..network.flux_distribution import FluxDistribution
    from ..util.int_int_multi_value_map import IntIntMultiValueMap

# Forward declarations for classes not yet implemented
class MetabolicNetwork:
    """Placeholder for MetabolicNetwork interface - to be implemented"""
    pass

class FluxDistribution:
    """Placeholder for FluxDistribution class - to be implemented"""
    pass

class Metabolite:
    """Placeholder for Metabolite interface - already implemented but import issues"""
    pass

class Reaction:
    """Placeholder for Reaction interface - to be implemented"""
    pass

class IntIntMultiValueMap:
    """Placeholder for IntIntMultiValueMap utility - to be implemented"""
    pass


class CompressedMetabolicNetwork(MetabolicNetwork, ABC):
    """
    An extension of metabolic network supporting uncompression of flux modes
    associated with this compressed network.
    
    This interface extends MetabolicNetwork to provide capabilities for:
    - Tracking the original network hierarchy (root and parent networks)
    - Uncompressing flux distributions back to original space
    - Mapping metabolites and reactions between compressed and original networks
    - Providing reaction mapping information for compression algorithms
    """
    
    @abstractmethod
    def get_root_network(self) -> 'MetabolicNetwork':
        """
        Get the top level uncompressed network.
        
        Returns:
            The root (original, uncompressed) metabolic network
        """
        pass
    
    @abstractmethod
    def get_parent_network(self) -> 'MetabolicNetwork':
        """
        Get the next upper level uncompressed or compressed network.
        
        Returns:
            The immediate parent network (may be compressed or uncompressed)
        """
        pass
    
    @abstractmethod
    def uncompress_flux_distribution(self, flux_distribution: 'FluxDistribution') -> 'FluxDistribution':
        """
        Returns the uncompressed flux distributions, does not expand duplicate genes.
        
        Args:
            flux_distribution: The flux distribution in compressed space
        
        Returns:
            The flux distribution expanded to the parent network space
        """
        pass
    
    @abstractmethod
    def get_mapped_metabolites(self, original: List['Metabolite']) -> List[Optional['Metabolite']]:
        """
        Returns the mapped possibly compressed/composite metabolites in the
        same order as in the input list. The output list contains None if the
        respective metabolite has been removed (dead end).
        
        Args:
            original: The list of metabolites from the original uncompressed network
        
        Returns:
            The mapped metabolites in the same order as in original. They are 
            equal to the original if the metabolite has not been removed nor 
            compressed. They are None if they have been removed, and some new 
            instance of a composite metabolite otherwise.
        """
        pass
    
    @abstractmethod
    def get_mapped_reactions(self, original: List['Reaction']) -> List[Optional['Reaction']]:
        """
        Returns the mapped possibly compressed/composite reactions in the
        same order as in the input list. The output list contains None if the
        respective reaction has been removed (zero flux reaction).
        
        Args:
            original: The list of reactions from the original uncompressed network
        
        Returns:
            The mapped reactions in the same order as in original. They are 
            equal to the original if the reaction has not been removed nor 
            compressed. They are None if they have been removed, and some new 
            instance otherwise.
        """
        pass
    
    @abstractmethod
    def get_reaction_mapping(self) -> 'IntIntMultiValueMap':
        """
        Returns the reaction mapping as a one to many map. The key in the map is
        the original reaction index, the values are the new reaction indices in
        the compressed network. One reaction might be mapped to none, one or 
        multiple reactions, and a mapped reaction might consist of one to many
        original ones.
        
        Returns:
            The mapping of original to compressed reactions
        """
        pass