#!/usr/bin/env python3
"""
Annotateable interface - Python port of Java ch.javasoft.metabolic.Annotateable

All parts of a metabolic network extend this interface, thus annotations can 
be added to the metabolic network for such elements.

An annotation could for instance be an EC number or gene name for a reaction, 
or the chemical formula for a metabolite, or the model version of the network 
itself.

Java source: efmtool_source/ch/javasoft/metabolic/Annotateable.java
"""

from abc import ABC, abstractmethod


class Annotateable(ABC):
    """
    All parts of a metabolic network extend this interface, thus annotations can 
    be added to the metabolic network for such elements.
    
    An annotation could for instance be an EC number or gene name for a reaction, 
    or the chemical formula for a metabolite, or the model version of the network 
    itself.
    """
    pass