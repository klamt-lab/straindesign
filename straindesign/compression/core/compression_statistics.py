"""
EFMTool Compression Statistics

Python port of ch.javasoft.metabolic.compress.CompressionStatistics
Tracks statistics for different compression operations and iterations.

From Java source: efmtool_source/ch/javasoft/metabolic/compress/CompressionStatistics.java
Ported line-by-line for exact compatibility
"""

import logging
from enum import Enum
from io import StringIO
from typing import List, TextIO, Union


class CompressionTypeR(Enum):
    """Compression types for reactions"""
    ZERO_FLUX = "ZeroFlux"
    CONTRADICTING = "Contradicting"
    COUPLED = "Coupled"
    UNIQUE_FLOW = "UniqueFlow"
    DEAD_END = "DeadEnd"
    DUPLICATE_GENE_SIMPLE = "DuplicateGeneSimple"
    DUPLICATE_GENE_COMPLEX = "DuplicateGeneComplex"


class CompressionTypeM(Enum):
    """Compression types for metabolites"""
    DEAD_END = "DeadEnd"
    UNIQUE_FLOW = "UniqueFlow"
    UNUSED = "Unused"
    INTERCHANGEABLE_METABOLITE_SIMPLE = "InterchangeableMetaboliteSimple"
    INTERCHANGEABLE_METABOLITE_COMPLEX = "InterchangeableMetaboliteComplex"


class CompressionStatistics:
    """
    Tracks statistics for different compression operations and iterations.
    
    This class maintains counts for various compression types across multiple
    iterations of the compression algorithm. It provides methods to increment
    counters for different compression operations and generate reports.
    """
    
    # Logger setup
    _logger = logging.getLogger(__name__ + ".stats")
    
    def __init__(self):
        """Initialize compression statistics tracking."""
        # Initialize count arrays for reactions and metabolites
        # Each array is indexed by compression type, containing arrays for iteration counts
        self._count_r = [[0] for _ in CompressionTypeR]  # Reaction counts by type and iteration
        self._count_m = [[0] for _ in CompressionTypeM]  # Metabolite counts by type and iteration
        self._compression_iteration = -1  # Current compression iteration (-1 = not started)
        self._max_non_zero_iteration = -1  # Highest iteration with non-zero counts
    
    # Iteration management
    
    def inc_compression_iteration(self) -> int:
        """
        Increments the compression iteration, which is -1 by default. Thus, an
        initial incrementation is needed.
        
        Returns:
            The new (increased) iteration count
        """
        self._compression_iteration += 1
        return self._compression_iteration
    
    def get_compression_iteration(self) -> int:
        """
        Get the current compression iteration.
        
        Returns:
            Current compression iteration (-1 if not started)
        """
        return self._compression_iteration
    
    # Reaction compression statistics
    
    def inc_zero_flux_reactions(self) -> None:
        """Increment count for zero flux reactions."""
        self._inc_r(CompressionTypeR.ZERO_FLUX)
    
    def inc_contradicting_reactions(self) -> None:
        """Increment count for contradicting reactions."""
        self._inc_r(CompressionTypeR.CONTRADICTING)
    
    def inc_coupled_reactions(self) -> None:
        """Increment count for coupled reactions (single reaction)."""
        self._inc_r(CompressionTypeR.COUPLED, 1)
    
    def inc_coupled_reactions_count(self, reaction_count: int) -> None:
        """
        Increment count for coupled reactions.
        
        Args:
            reaction_count: Number of coupled reactions to add
        """
        self._inc_r(CompressionTypeR.COUPLED, reaction_count)
    
    def inc_unique_flow_reactions(self) -> None:
        """Increment count for unique flow reactions (affects both reactions and metabolites)."""
        self._inc_r(CompressionTypeR.UNIQUE_FLOW)
        self._inc_m(CompressionTypeM.UNIQUE_FLOW)
    
    def inc_dead_end_metabolite_reactions(self, reaction_count: int) -> None:
        """
        Increment count for dead end metabolite reactions.
        
        Args:
            reaction_count: Number of dead end reactions to add
        """
        self._inc_r(CompressionTypeR.DEAD_END, reaction_count)
        self._inc_m(CompressionTypeM.DEAD_END)
    
    def inc_unused_metabolite(self) -> None:
        """Increment count for unused metabolites."""
        self._inc_m(CompressionTypeM.UNUSED)
    
    def inc_duplicate_gene_reactions(self, reaction_count: int) -> None:
        """
        Increment count for duplicate gene reactions (simple).
        
        Args:
            reaction_count: Number of duplicate gene reactions to add
        """
        self._inc_r(CompressionTypeR.DUPLICATE_GENE_SIMPLE, reaction_count)
    
    def inc_duplicate_gene_compound_reactions(self, reaction_count: int) -> None:
        """
        Increment count for duplicate gene compound reactions (complex).
        
        Args:
            reaction_count: Number of duplicate gene compound reactions to add
        """
        self._inc_r(CompressionTypeR.DUPLICATE_GENE_COMPLEX, reaction_count)
    
    def inc_interchangeable_metabolites(self) -> None:
        """Increment count for interchangeable metabolites (simple)."""
        self._inc_m(CompressionTypeM.INTERCHANGEABLE_METABOLITE_SIMPLE)
    
    def inc_interchangeable_metabolites_complex(self) -> None:
        """Increment count for interchangeable metabolites (complex)."""
        self._inc_m(CompressionTypeM.INTERCHANGEABLE_METABOLITE_COMPLEX)
    
    def get_duplicate_gene_reactions_count(self) -> int:
        """
        Get total count of duplicate gene reactions across all iterations.
        
        Returns:
            Total duplicate gene reaction count
        """
        total = 0
        type_index = list(CompressionTypeR).index(CompressionTypeR.DUPLICATE_GENE_SIMPLE)
        for iteration in range(self._max_non_zero_iteration + 1):
            total += self._get_r(type_index, iteration)
        return total
    
    def get_iteration_count(self) -> int:
        """
        Get the number of compression iterations performed.
        
        Returns:
            Number of iterations (same as get_compression_iteration())
        """
        return self.get_compression_iteration()
    
    def get_dead_end_metabolite_reactions_count(self) -> int:
        """
        Get total count of dead end metabolite reactions across all iterations.
        
        Returns:
            Total dead end metabolite reaction count
        """
        total = 0
        type_index = list(CompressionTypeR).index(CompressionTypeR.DEAD_END)
        for iteration in range(self._max_non_zero_iteration + 1):
            total += self._get_r(type_index, iteration)
        return total
    
    def get_unique_flow_reactions_count(self) -> int:
        """
        Get total count of unique flow reactions across all iterations.
        
        Returns:
            Total unique flow reaction count
        """
        total = 0
        type_index = list(CompressionTypeR).index(CompressionTypeR.UNIQUE_FLOW)
        for iteration in range(self._max_non_zero_iteration + 1):
            total += self._get_r(type_index, iteration)
        return total
    
    def get_zero_flux_reactions_count(self) -> int:
        """
        Get total count of zero flux reactions across all iterations.
        
        Returns:
            Total zero flux reaction count
        """
        total = 0
        type_index = list(CompressionTypeR).index(CompressionTypeR.ZERO_FLUX)
        for iteration in range(self._max_non_zero_iteration + 1):
            total += self._get_r(type_index, iteration)
        return total
    
    def get_coupled_reactions_count(self) -> int:
        """
        Get total count of coupled reactions across all iterations.
        
        Returns:
            Total coupled reaction count
        """
        total = 0
        type_index = list(CompressionTypeR).index(CompressionTypeR.COUPLED)
        for iteration in range(self._max_non_zero_iteration + 1):
            total += self._get_r(type_index, iteration)
        return total
    
    def get_total_reaction_compressions(self) -> int:
        """
        Get total count of all reaction compressions across all types and iterations.
        
        Returns:
            Total reaction compression count
        """
        total = 0
        for type_r in CompressionTypeR:
            type_index = list(CompressionTypeR).index(type_r)
            for iteration in range(self._max_non_zero_iteration + 1):
                total += self._get_r(type_index, iteration)
        return total
    
    def get_total_metabolite_compressions(self) -> int:
        """
        Get total count of all metabolite compressions across all types and iterations.
        
        Returns:
            Total metabolite compression count
        """
        total = 0
        for type_m in CompressionTypeM:
            type_index = list(CompressionTypeM).index(type_m)
            for iteration in range(self._max_non_zero_iteration + 1):
                total += self._get_m(type_index, iteration)
        return total
    
    def get_unique_flows_metabolite_reactions_count(self) -> int:
        """
        Get total count of unique flow metabolite reactions across all iterations.
        This is an alias for get_unique_flow_reactions_count for compatibility.
        
        Returns:
            Total unique flow reaction count
        """
        return self.get_unique_flow_reactions_count()
    
    # Internal increment methods
    
    def _inc_r(self, compression_type: CompressionTypeR, count: int = 1) -> None:
        """
        Increment reaction count for given compression type.
        
        Args:
            compression_type: Type of compression
            count: Number of increments (default: 1)
        """
        type_index = list(CompressionTypeR).index(compression_type)
        for _ in range(count):
            self._count_r[type_index] = self._inc_array(self._count_r[type_index])
    
    def _inc_m(self, compression_type: CompressionTypeM) -> None:
        """
        Increment metabolite count for given compression type.
        
        Args:
            compression_type: Type of compression
        """
        type_index = list(CompressionTypeM).index(compression_type)
        self._count_m[type_index] = self._inc_array(self._count_m[type_index])
    
    def _inc_array(self, array: List[int]) -> List[int]:
        """
        Increment count in array for current iteration, expanding array if needed.
        
        Args:
            array: Array to increment
            
        Returns:
            Updated array (may be new instance if expanded)
        """
        # Expand array if needed
        if len(array) <= self._compression_iteration:
            new_size = max(self._compression_iteration + 1, len(array) * 2)
            new_array = [0] * new_size
            new_array[:len(array)] = array
            array = new_array
        
        # Increment current iteration
        array[self._compression_iteration] += 1
        self._max_non_zero_iteration = self._compression_iteration
        return array
    
    # Data access methods
    
    def _get_r(self, which: int, iteration: int) -> int:
        """
        Get reaction count for specific type and iteration.
        
        Args:
            which: Index of compression type
            iteration: Compression iteration
            
        Returns:
            Count for the specified type and iteration
        """
        return self._get_array_value(self._count_r[which], iteration)
    
    def _get_m(self, which: int, iteration: int) -> int:
        """
        Get metabolite count for specific type and iteration.
        
        Args:
            which: Index of compression type
            iteration: Compression iteration
            
        Returns:
            Count for the specified type and iteration
        """
        return self._get_array_value(self._count_m[which], iteration)
    
    def _get_array_value(self, array: List[int], iteration: int) -> int:
        """
        Get value from array at given iteration, returning 0 if out of bounds.
        
        Args:
            array: Array to access
            iteration: Iteration index
            
        Returns:
            Value at iteration or 0 if out of bounds
        """
        return array[iteration] if 0 <= iteration < len(array) else 0
    
    # Output methods
    
    def __str__(self) -> str:
        """
        String representation of compression statistics.
        
        Returns:
            Formatted statistics string
        """
        output = StringIO()
        self.write(output)
        return output.getvalue()
    
    def write_to_log(self, level: int = logging.DEBUG) -> None:
        """
        Write statistics to logger.
        
        Args:
            level: Log level to use (default: logging.DEBUG)
        """
        if self._logger.isEnabledFor(level):
            # Split output into lines and log each one
            stats_str = str(self)
            for line in stats_str.splitlines():
                self._logger.log(level, line)
    
    def write(self, writer: Union[TextIO, StringIO]) -> None:
        """
        Write statistics to writer/stream.
        
        Args:
            writer: Text stream to write to
        """
        writer.write("compression statistics\n")
        
        # Write per-iteration statistics
        for iteration in range(self._max_non_zero_iteration + 1):
            # Metabolite statistics
            for which, compression_type in enumerate(CompressionTypeM):
                value = self._get_m(which, iteration)
                writer.write(f"  meta[{iteration}].{compression_type.value} = {value}\n")
            
            # Reaction statistics
            for which, compression_type in enumerate(CompressionTypeR):
                value = self._get_r(which, iteration)
                writer.write(f"  reac[{iteration}].{compression_type.value} = {value}\n")
        
        # Write totals for metabolites
        total_meta = 0
        for which, compression_type in enumerate(CompressionTypeM):
            total = 0
            for iteration in range(self._max_non_zero_iteration + 1):
                total += self._get_m(which, iteration)
            writer.write(f"  meta.{compression_type.value} = {total}\n")
            total_meta += total
        
        # Write totals for reactions
        total_reac = 0
        for which, compression_type in enumerate(CompressionTypeR):
            total = 0
            for iteration in range(self._max_non_zero_iteration + 1):
                total += self._get_r(which, iteration)
            writer.write(f"  reac.{compression_type.value} = {total}\n")
            total_reac += total
        
        # Write grand totals
        writer.write(f"  meta = {total_meta}\n")
        writer.write(f"  reac = {total_reac}\n")
        
        if hasattr(writer, 'flush'):
            writer.flush()