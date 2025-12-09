#!/usr/bin/env python3
"""
DefaultMetabolicNetwork utility functions - Python port of relevant parts of Java ch.javasoft.metabolic.impl.DefaultMetabolicNetwork

Provides utility functions for generating systematic names and creating metabolite arrays.

Java source: efmtool_source/ch/javasoft/metabolic/impl/DefaultMetabolicNetwork.java
"""

from typing import List
from .default_metabolite import DefaultMetabolite
from ..metabolite import Metabolite


class DefaultMetabolicNetwork:
    """
    Utility class providing static methods for metabolic network construction.
    
    Note: This is NOT a complete port of DefaultMetabolicNetwork, only the utility
    methods needed by FractionNumberStoichMetabolicNetwork.
    """
    
    @staticmethod
    def metabolite_names(length: int) -> List[str]:
        """
        Generate systematic metabolite names for a given count.
        
        Args:
            length: Number of metabolite names to generate
            
        Returns:
            List[str]: List of generated metabolite names like ["A", "B", "C", ...]
        """
        metabolite_prefix = ""  # Java: metabolitePrefix() returns ""
        return DefaultMetabolite.names(length, metabolite_prefix)
    
    @staticmethod 
    def reaction_names(length: int) -> List[str]:
        """
        Generate systematic reaction names for a given count.
        
        CRITICAL: Java uses NUMERIC naming (R0, R1, R2...) NOT alphabetic (RA, RB, RC...)
        
        Args:
            length: Number of reaction names to generate
            
        Returns:
            List[str]: List of generated reaction names like ["R0", "R1", "R2", ...]
        """
        return DefaultMetabolicNetwork._names(length, "R")
    
    @staticmethod
    def _names(count: int, prefix: str) -> List[str]:
        """
        CRITICAL: Generate names with NUMERIC suffixes to match Java exactly.
        
        Java implementation uses zero-padded numeric indices, NOT alphabetic names.
        
        Args:
            count: Number of names to generate
            prefix: Prefix for each name
            
        Returns:
            List[str]: Names like ["R0", "R1"] or ["R00", "R01", "R02"] etc.
        """
        if count == 0:
            return []
        
        # Calculate padding length based on maximum index
        max_index = count - 1
        padding_length = len(str(max_index))
        
        names = []
        for i in range(count):
            # Zero-pad the index
            padded_index = str(i).zfill(padding_length)
            names.append(prefix + padded_index)
        
        return names
    
    @staticmethod
    def metabolites(metabolite_names: List[str]) -> List[Metabolite]:
        """
        Create DefaultMetabolite objects from names.
        
        Args:
            metabolite_names: List of metabolite name strings
            
        Returns:
            List[Metabolite]: List of DefaultMetabolite objects
        """
        return [DefaultMetabolite(name) for name in metabolite_names]