#!/usr/bin/env python3
"""
Line-by-line translation of Java CoupledCombine compression method.
"""

from fractions import Fraction
from typing import List, Optional
import logging

from .rational_math import RationalMath, ZERO, ONE
from .rational_matrix import RationalMatrix
from .java_translation import (
    WorkRecord, NullspaceRecord, CompressionStatistics, 
    nullspace_coupled_reactions, identity, cancel, is_zero
)


class CoupledCombineJavaTranslation:
    """Line-by-line translation of Java CoupledCombine compression method."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def compress_coupled_combine(self, work_record: WorkRecord) -> bool:
        """
        Line-by-line translation of CoupledCombine compression workflow.
        
        This matches the nullspace workflow but only applies reaction combining.
        """
        # Line 211: NullspaceRecord nullspaceRecord = new NullspaceRecord(workRecord);
        nullspace_record = NullspaceRecord(work_record)
        
        # Line 212: boolean compressedAny = false;
        compressed_any = False
        
        # Line 214: if (doCpl) compressedAny |= nullspaceCoupledReactions(nullspaceRecord, inclCompression);
        # For CoupledCombine: do_contradicting=False, do_combine=True, incl_compression=True
        compressed_any = compressed_any or nullspace_coupled_reactions(
            nullspace_record, incl_compression=True, 
            do_contradicting=False, do_combine=True
        )
        
        # Line 215: if (compressedAny) workRecord.removeUnusedMetabolites();
        if compressed_any:
            work_record.remove_unused_metabolites()
        
        # Copy results back to work_record
        work_record.pre = nullspace_record.pre
        work_record.cmp = nullspace_record.cmp
        work_record.post = nullspace_record.post
        work_record.reversible = nullspace_record.reversible
        work_record.size = nullspace_record.size
        work_record.stats = nullspace_record.stats
        work_record.meta_names = nullspace_record.meta_names
        work_record.reac_names = nullspace_record.reac_names
        
        return compressed_any