#!/usr/bin/env python3
"""
FluxDistribution interface - Python port of Java ch.javasoft.metabolic.FluxDistribution

Interface for flux distributions representing reaction rates in metabolic networks.

Java source: efmtool_source/ch/javasoft/metabolic/FluxDistribution.java
"""

from abc import ABC, abstractmethod
from typing import List, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .metabolic_network import MetabolicNetwork


class FluxDistribution(ABC):
    """
    Abstract interface for flux distributions.
    
    Represents the flux (rate) values for reactions in a metabolic network.
    """
    
    @abstractmethod
    def get_preferred_number_class(self) -> type:
        """Get the preferred number class for this flux distribution"""
        pass
    
    @abstractmethod
    def get_number_rate(self, reaction_index: int) -> Union[int, float, 'BigFraction']:
        """Get reaction rate as number (exact type)"""
        pass
    
    @abstractmethod
    def get_double_rates(self) -> List[float]:
        """Get all reaction rates as double array"""
        pass
    
    @abstractmethod
    def set_rate(self, reaction_index: int, rate: Union[int, float, 'BigFraction']) -> None:
        """Set reaction rate"""
        pass
    
    @abstractmethod
    def create(self, network: 'MetabolicNetwork') -> 'FluxDistribution':
        """Create new flux distribution for given network"""
        pass