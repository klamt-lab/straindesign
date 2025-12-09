"""
Network Infrastructure Module

This module provides the metabolic network representations and components:
- MetabolicNetwork interfaces and implementations
- Network components (Metabolite, Reaction, FluxDistribution)  
- Compressed network representations
- Network visitors and utilities

All components maintain exact precision and compatibility.
"""

from .metabolic_network import MetabolicNetwork
from .fraction_number_stoich_metabolic_network import FractionNumberStoichMetabolicNetwork
from .metabolite import Metabolite
from .reaction import Reaction
from .flux_distribution import FluxDistribution
from .annotateable import Annotateable

__all__ = [
    'MetabolicNetwork',
    'FractionNumberStoichMetabolicNetwork',
    'Metabolite',
    'Reaction', 
    'FluxDistribution',
    'Annotateable',
]