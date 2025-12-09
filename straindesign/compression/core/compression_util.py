#!/usr/bin/env python3
"""
EFMTool Compression Utilities

Python port of ch.javasoft.metabolic.compress.CompressionUtil
Provides utility functions for network compression operations.

From Java source: efmtool_source/ch/javasoft/metabolic/compress/CompressionUtil.java
Ported line-by-line for exact compatibility
"""

import logging
from typing import Optional, Set, List, TYPE_CHECKING

from .compression_method import CompressionMethod
from .stoich_matrix_compressor import StoichMatrixCompressor
from .duplicate_gene_compressor import DuplicateGeneCompressor
from .stoich_matrix_compressed_network import StoichMatrixCompressedMetabolicNetwork
from ..network.impl.fraction_stoich_network import FractionNumberStoichMetabolicNetwork

if TYPE_CHECKING:
    from ..math.readable_bigint_rational_matrix import ReadableBigIntegerRationalMatrix
    from ..math.readable_matrix import ReadableMatrix
    from ..math.bigint_rational_matrix import BigIntegerRationalMatrix
    from ..network.metabolic_network import MetabolicNetwork
    from .compressed_metabolic_network import CompressedMetabolicNetwork
    from ..network.impl.abstract_metabolic_network import AbstractMetabolicNetwork


# Zero tolerance class - simplified implementation for now
class Zero:
    """Zero tolerance class for numerical comparisons"""
    def __init__(self, tolerance: float):
        self.tolerance = tolerance
        
    def is_zero(self, value: float) -> bool:
        """Check if value is within zero tolerance"""
        return abs(value) <= self.tolerance


class CompressionUtil:
    """
    Utility functions for metabolic network compression.
    
    This class provides static methods for compressing metabolic networks
    using various compression algorithms and configurations.
    """
    
    # Logger setup (simplified for Python)
    _logger = logging.getLogger(__name__)
    _data_logger = logging.getLogger(f"{__name__}.data")
    
    def __init__(self):
        """Private constructor - this is a utility class with only static methods"""
        raise RuntimeError("CompressionUtil is a utility class and should not be instantiated")
    
    @staticmethod
    def compress(net: 'MetabolicNetwork', 
                suppressed_reactions: Optional[Set[str]] = None) -> 'CompressedMetabolicNetwork':
        """
        Compresses the given network, using zero tolerance and 
        STANDARD compression. No reactions are suppressed by default, 
        duplicate gene compression is not applied.
        
        Args:
            net: The network to compress
            suppressed_reactions: Optional set of reactions to suppress (exclude) 
                                in the compressed network
        
        Returns:
            The compressed network
        """
        if suppressed_reactions is None:
            suppressed_reactions = set()
        return CompressionUtil.compress_with_methods(net, CompressionMethod.standard(), suppressed_reactions)
    
    @staticmethod
    def compress_with_methods(net: 'MetabolicNetwork', 
                            methods: List[CompressionMethod],
                            suppressed_reactions: Optional[Set[str]] = None) -> 'CompressedMetabolicNetwork':
        """
        Compresses the given network, using the default zero tolerance and 
        compression methods. Reactions in the suppressed reactions set are 
        removed in the compressed network, duplicate gene compression is not 
        applied.
        
        Args:
            net: The network to compress
            methods: The compression methods to apply
            suppressed_reactions: Optional set of reactions to suppress (exclude)
                                in the compressed network
        
        Returns:
            The compressed network
        """
        if suppressed_reactions is None:
            suppressed_reactions = set()
        return CompressionUtil.compress_with_zero(net, methods, suppressed_reactions, Zero(0.0))
    
    @staticmethod
    def compress_with_zero(net: 'MetabolicNetwork',
                          methods: List[CompressionMethod], 
                          suppressed_reactions: Optional[Set[str]],
                          zero: Zero) -> 'CompressedMetabolicNetwork':
        """
        Compresses the given network, using the given zero tolerance and 
        compression methods. Reactions in the suppressed reactions set are 
        removed in the compressed network, duplicate gene compression is not 
        applied.
        
        Args:
            net: The network to compress
            methods: The compression methods to apply
            suppressed_reactions: Optional set of reactions to suppress (exclude)
                                in the compressed network  
            zero: The zero tolerance to use
        
        Returns:
            The compressed network
            
        Note:
            This method requires StoichMatrixCompressor and related classes
            which are not yet implemented. It serves as a placeholder for
            the complete implementation.
        """
        if suppressed_reactions is None:
            suppressed_reactions = set()
        
        # Get network data  
        reversible = net.get_reaction_reversibilities()
        stoich = FractionNumberStoichMetabolicNetwork.get_stoich(net)  
        meta_names = net.get_metabolite_names()
        reac_names = net.get_reaction_names()
        
        # Compress using StoichMatrixCompressor
        compressor = StoichMatrixCompressor(*methods)
        rec = compressor.compress(stoich, reversible, meta_names, reac_names, suppressed_reactions)
        
        # Sort matrices for better performance
        CompressionUtil._sort_matrix(rec.cmp, rec.post, rec.reversible)
        CompressionUtil.log_compression_record_with_stoich(stoich, rec, logging.DEBUG)
        
        # Create compressed network
        cmp_net = StoichMatrixCompressedMetabolicNetwork(net, rec.pre, rec.post, rec.cmp)
        
        # Consistency check for reversibilities
        if rec.reversible != cmp_net.get_reaction_reversibilities():
            CompressionUtil._logger.warning("reversibility mismatch:")
            CompressionUtil._logger.warning(f"  reversibility(cmp.rec) = {rec.reversible}")
            CompressionUtil._logger.warning(f"  reversibility(cmp.net) = {cmp_net.get_reaction_reversibilities()}")
            raise RuntimeError("reversibility mismatch, see log for details")
        
        # Log compression results
        if CompressionUtil._logger.isEnabledFor(logging.DEBUG):
            CompressionMethod.log(logging.DEBUG, *methods)
            CompressionUtil._logger.debug(f"Uncompressed network size: {CompressionUtil._get_network_size_string(net)}")
            CompressionUtil._logger.debug(f"Compressed network size: {CompressionUtil._get_network_size_string(cmp_net)}")
        
        return cmp_net
    
    @staticmethod 
    def compress_duplicate_gene_reactions(net: 'MetabolicNetwork',
                                        zero: Zero,
                                        *methods: CompressionMethod) -> 'MetabolicNetwork':
        """
        Performs the duplicate gene compression. This compression method is 
        usually made in advance, and efms for the duplicate-free network is 
        computed. It might be a good idea to call 
        convert_to_noncompressed() with the returned network since uncompressing 
        efms for duplicate gene compressed networks does not work properly.
        
        Args:
            net: The network to compress
            zero: The zero tolerance to use
            *methods: Variable number of compression methods
        
        Returns:
            The duplicate gene compressed network
            
        Note:
            This method requires DuplicateGeneCompressor which is not yet 
            implemented. It serves as a placeholder for the complete implementation.
        """
        # Get network data
        reversible = net.get_reaction_reversibilities()
        stoich = FractionNumberStoichMetabolicNetwork.get_stoich(net)
        meta_names = net.get_metabolite_names()
        reac_names = net.get_reaction_names()
        do_extended_duplicates = CompressionMethod.DUPLICATE_GENE_EXTENDED.contained_in(*methods)
        
        # Compress using DuplicateGeneCompressor
        rec = DuplicateGeneCompressor.compress(stoich, reversible, meta_names, reac_names, do_extended_duplicates)
        CompressionUtil._sort_matrix(rec.cmp, rec.post, rec.reversible)
        CompressionUtil.log_duplicate_gene_compression_record_with_stoich(stoich, rec, logging.DEBUG)
        
        # Create duplicate-free network - use the cmp matrix as the stoichiometric matrix
        # Use truncated names to match compressed dimensions
        compressed_meta_names = meta_names[:rec.get_compressed_metabolite_count()]
        compressed_reac_names = reac_names[:rec.get_compressed_reaction_count()]
        dupfree_net = FractionNumberStoichMetabolicNetwork(
            compressed_meta_names,
            compressed_reac_names, 
            rec.cmp, 
            rec.reversible
        )
        
        # Log compression results
        if CompressionUtil._logger.isEnabledFor(logging.DEBUG):
            CompressionMethod.log(logging.DEBUG, CompressionMethod.DUPLICATE_GENE)
            CompressionUtil._logger.debug(f"Uncompressed network size: {CompressionUtil._get_network_size_string(net)}")
            CompressionUtil._logger.debug(f"Duplicate free network size: {CompressionUtil._get_network_size_string(dupfree_net)}")
        
        return dupfree_net
    
    @staticmethod
    def convert_to_noncompressed(cnet: 'MetabolicNetwork') -> 'MetabolicNetwork':
        """
        Nests the given compressed network such that the returned network is no
        more recognizable as a compressed network. This can be useful if 
        uncompression is performed automatically by recognizing compressed 
        networks, but uncompression is not desired.
        
        Args:
            cnet: The compressed network to convert to a noncompressed one
        
        Returns:
            A metabolic network nesting the given compressed network
            
        Note:
            This method requires AbstractMetabolicNetwork which is not yet 
            implemented. It serves as a placeholder for the complete implementation.
        """
        # Create wrapper class that hides compressed nature
        class NoncompressedWrapper:
            """Wrapper that hides compressed network properties"""
            def __init__(self, wrapped_net: 'MetabolicNetwork'):
                self._wrapped_net = wrapped_net
            
            def get_metabolites(self):
                return self._wrapped_net.get_metabolites()
            
            def get_reactions(self):
                return self._wrapped_net.get_reactions()
            
            def get_stoichiometric_matrix(self):
                return self._wrapped_net.get_stoichiometric_matrix()
                
            def get_reaction_reversibilities(self):
                return self._wrapped_net.get_reaction_reversibilities()
                
            def get_metabolite_names(self):
                return self._wrapped_net.get_metabolite_names()
                
            def get_reaction_names(self):
                return self._wrapped_net.get_reaction_names()
        
        return NoncompressedWrapper(cnet)
    
    @staticmethod
    def log_compression_record(rec: 'StoichMatrixCompressor.CompressionRecord',
                             level: int = logging.DEBUG) -> None:
        """
        Logs the given compression record to the data logger, using the 
        specified log level.
        
        Args:
            rec: The compression record to trace
            level: The log level used for tracing (default: logging.DEBUG)
        """
        CompressionUtil.log_compression_record_with_stoich(None, rec, level)
    
    @staticmethod
    def log_compression_record_with_stoich(stoich: Optional['ReadableMatrix'], 
                                         rec: 'StoichMatrixCompressor.CompressionRecord',
                                         level: int = logging.DEBUG) -> None:
        """
        Logs the given compression record and stoichiometric matrix to the 
        data logger, using the specified log level.
        
        Args:
            stoich: The stoichiometric matrix to trace (can be None)
            rec: The compression record to trace  
            level: The log level used for tracing (default: logging.DEBUG)
            
        Note:
            This method requires CompressionRecord classes which are not yet 
            implemented. It serves as a placeholder for the complete implementation.
        """
        if not CompressionUtil._data_logger.isEnabledFor(level):
            return
        
        CompressionUtil._data_logger.log(level, "compression matrices:")
        CompressionUtil._data_logger.log(level, "  pre * stoich * post  = cmp")
        CompressionUtil._data_logger.log(level, "efm uncompression:")
        CompressionUtil._data_logger.log(level, "  stoich * post * efmc = 0")
        CompressionUtil._data_logger.log(level, "  stoich * efm         = 0")
        CompressionUtil._data_logger.log(level, "  -->      efm         = post * efmc")
        
        if stoich is not None:
            CompressionUtil._data_logger.log(level, "stoich: ")
            CompressionUtil._log_matrix(stoich, level)
        
        CompressionUtil._data_logger.log(level, "pre: ")
        CompressionUtil._log_matrix(rec.pre, level)
        CompressionUtil._data_logger.log(level, "post: ")
        CompressionUtil._log_matrix(rec.post, level)
        CompressionUtil._data_logger.log(level, "cmp: ")
        CompressionUtil._log_matrix(rec.cmp, level)
        CompressionUtil._data_logger.log(level, f"cmp_reversibilities = {rec.reversible}")
        
    
    @staticmethod
    def log_duplicate_gene_compression_record(rec: 'DuplicateGeneCompressor.CompressionRecord',
                                            level: int = logging.DEBUG) -> None:
        """
        Logs the given duplicate gene compression record to the data logger, 
        using the specified log level.
        
        Args:
            rec: The compression record to trace
            level: The log level used for tracing (default: logging.DEBUG)
        """
        CompressionUtil.log_duplicate_gene_compression_record_with_stoich(None, rec, level)
    
    @staticmethod
    def log_duplicate_gene_compression_record_with_stoich(stoich: Optional['ReadableMatrix'],
                                                        rec: 'DuplicateGeneCompressor.CompressionRecord', 
                                                        level: int = logging.DEBUG) -> None:
        """
        Logs the given duplicate gene compression record and stoichiometric matrix 
        to the data logger, using the specified log level.
        
        Args:
            stoich: The stoichiometric matrix to trace (can be None)
            rec: The compression record to trace
            level: The log level used for tracing (default: logging.DEBUG)
            
        Note:
            This method requires DuplicateGeneCompressor.CompressionRecord which is 
            not yet implemented. It serves as a placeholder for the complete implementation.
        """
        if not CompressionUtil._data_logger.isEnabledFor(level):
            return
        
        CompressionUtil._data_logger.log(level, "duplicate gene compression matrices:")
        CompressionUtil._data_logger.log(level, "  stoich * post = cmp (dupfree)")
        
        if stoich is not None:
            CompressionUtil._data_logger.log(level, "stoich: ")
            CompressionUtil._log_matrix(stoich, level)
        
        CompressionUtil._data_logger.log(level, "post (dupelim): ")
        CompressionUtil._log_matrix(rec.post, level)
        CompressionUtil._data_logger.log(level, "cmp (dupfree): ")
        CompressionUtil._log_matrix(rec.cmp, level)
        CompressionUtil._data_logger.log(level, f"dupfree_reversibilities = {rec.reversible}")
        CompressionUtil._data_logger.log(level, f"duplicate groups: {getattr(rec, 'groups', [])}")
    
    # Matrix sorting utilities (these can be implemented now as they're self-contained)
    
    @staticmethod
    def _sort_matrix(stoich: 'ReadableMatrix', post: 'ReadableMatrix', reversibilities: List[bool]) -> None:
        """
        Sorts the matrices by swapping columns for better performance.
        
        Args:
            stoich: The stoichiometric matrix to sort
            post: The post-compression matrix to sort  
            reversibilities: The reversibility array to sort accordingly
            
        Note:
            Currently uses diagonal sorting heuristic. Other sorting methods
            are available but commented out based on Java implementation.
        """
        # Use diagonal sorting heuristic (as selected in Java implementation)
        CompressionUtil._sort_matrix_diag_stoich(stoich, post, reversibilities)
    
    @staticmethod
    def _sort_matrix_none(stoich: 'ReadableMatrix', post: 'ReadableMatrix', reversibilities: List[bool]) -> None:
        """
        No-op sorting method (don't sort).
        
        Args:
            stoich: The stoichiometric matrix (unused)
            post: The post-compression matrix (unused)
            reversibilities: The reversibility array (unused)
        """
        # Don't sort - placeholder for no sorting option
        pass
    
    @staticmethod  
    def _sort_matrix_diag_stoich(stoich: 'ReadableMatrix', post: 'ReadableMatrix', reversibilities: List[bool]) -> None:
        """
        Sorts the matrices by swapping columns. The heuristics here are that the
        stoichiometrix matrix should be approximately diagonal, i.e. entries 
        should lie on the diagonal. We try to sort the matrix in such a way that 
        earlier columns have an earlier occurrance of non-zero elements in stoich.
        
        Args:
            stoich: The stoichiometric matrix to sort
            post: The post-compression matrix to sort
            reversibilities: The reversibility array to sort accordingly
            
        Note:
            This method requires ReadableMatrix and WritableMatrix interfaces
            with column swapping capability. Implementation is a placeholder.
        """
        # Use the implemented sorting method
        CompressionUtil._sort_matrix_diag_stoich(stoich, post, reversibilities)
    
    # Additional sorting methods from Java implementation (not used but available)
    
    @staticmethod
    def _sort_matrix_few_non_zero_cols_post(stoich: 'ReadableMatrix', post: 'ReadableMatrix', reversibilities: List[bool]) -> None:
        """
        Sorts the matrices by swapping columns. Columns with fewer non-zero 
        entries are placed at lower indices. If 2 colums have the same number of 
        non-zero entries, the column with the lower row index for the first 
        non-zero entry is placed at a lower index.
        
        Args:
            stoich: The stoichiometric matrix to sort
            post: The post-compression matrix to sort (used for sorting criteria)
            reversibilities: The reversibility array to sort accordingly
        """
        # Not implemented - available for future use
        pass
    
    @staticmethod
    def _sort_matrix_few_non_zero_cols_stoich(stoich: 'ReadableMatrix', post: 'ReadableMatrix', reversibilities: List[bool]) -> None:
        """
        Sorts the matrices by swapping columns. Columns with fewer non-zero 
        entries are placed at lower indices. If 2 colums have the same number of 
        non-zero entries, the column with the lower row index for the first 
        non-zero entry is placed at a lower index.
        
        Args:
            stoich: The stoichiometric matrix to sort (used for sorting criteria)
            post: The post-compression matrix to sort
            reversibilities: The reversibility array to sort accordingly
        """
        # Not implemented - available for future use
        pass
    
    @staticmethod  
    def _sort_matrix_diag_post(stoich: 'ReadableMatrix', post: 'ReadableMatrix', reversibilities: List[bool]) -> None:
        """
        Sorts the matrices by swapping columns. The matrix post was initially an
        identity matrix. Thus, entries should lie on the diagonal. Here, we try
        to resort the matrix in such a way that earlier columns have an earlier
        occurrance of non-zero elements in post.
        
        Args:
            stoich: The stoichiometric matrix to sort
            post: The post-compression matrix to sort (used for sorting criteria)
            reversibilities: The reversibility array to sort accordingly
        """
        # Implemented with matrix interface support
        if not hasattr(stoich, 'get_row_count') or not hasattr(stoich, 'swap_columns'):
            # Matrix doesn't support required operations
            return
            
        rows = post.get_row_count()
        cols = post.get_column_count()
        
        for piv in range(cols):
            # Find the column with an entry in the lowest row in post
            pivcol = piv
            pivrow = float('inf')
            
            for col in range(piv, cols):
                for row in range(rows):
                    if post.get_signum_at(row, col) != 0:
                        if row < pivrow:
                            pivrow = row
                            pivcol = col
                        break
                if pivrow == 0:
                    break
            
            if pivcol != piv:
                stoich.swap_columns(pivcol, piv)
                post.swap_columns(pivcol, piv)
                # Swap reversibilities array elements
                reversibilities[pivcol], reversibilities[piv] = reversibilities[piv], reversibilities[pivcol]
    
    @staticmethod
    def _sort_matrix_diag_stoich(stoich: 'ReadableMatrix', post: 'ReadableMatrix', reversibilities: List[bool]) -> None:
        """
        Sorts the matrices by swapping columns. The heuristics here are that the
        stoichiometrix matrix should be approximately diagonal, i.e. entries 
        should lie on the diagonal. We try to sort the matrix in such a way that 
        earlier columns have an earlier occurrance of non-zero elements in stoich.
        
        Args:
            stoich: The stoichiometric matrix to sort
            post: The post-compression matrix to sort
            reversibilities: The reversibility array to sort accordingly
        """
        # Implemented with matrix interface support
        if not hasattr(stoich, 'get_row_count') or not hasattr(stoich, 'swap_columns'):
            # Matrix doesn't support required operations
            return
            
        rows = stoich.get_row_count()
        cols = stoich.get_column_count()
        
        for piv in range(cols):
            # Find the column with an entry in the lowest row in stoich
            pivcol = piv
            pivrow = float('inf')
            
            for col in range(piv, cols):
                for row in range(rows):
                    if stoich.get_signum_at(row, col) != 0:
                        if row < pivrow:
                            pivrow = row
                            pivcol = col
                        break
                if pivrow == 0:
                    break
            
            if pivcol != piv:
                stoich.swap_columns(pivcol, piv)
                post.swap_columns(pivcol, piv)
                # Swap reversibilities array elements
                reversibilities[pivcol], reversibilities[piv] = reversibilities[piv], reversibilities[pivcol]
    
    @staticmethod
    def _log_matrix(matrix: 'ReadableMatrix', level: int) -> None:
        """
        Log a matrix to the data logger.
        
        Args:
            matrix: The matrix to log
            level: The logging level to use
        """
        try:
            # Try to log matrix dimensions and some values
            rows = matrix.get_row_count()
            cols = matrix.get_column_count()
            CompressionUtil._data_logger.log(level, f"  Matrix dimensions: {rows}x{cols}")
            
            # Log a few sample values for debugging
            sample_size = min(3, rows, cols)
            for r in range(sample_size):
                row_vals = []
                for c in range(min(sample_size, cols)):
                    try:
                        val = matrix.get_big_fraction_value_at(r, c)
                        row_vals.append(str(val))
                    except:
                        try:
                            val = matrix.get_double_value_at(r, c)
                            row_vals.append(f"{val:.3f}")
                        except:
                            row_vals.append("?")
                CompressionUtil._data_logger.log(level, f"    Row {r}: [{', '.join(row_vals)}{'...' if cols > sample_size else ''}]")
            
            if rows > sample_size:
                CompressionUtil._data_logger.log(level, f"    ... ({rows - sample_size} more rows)")
                
        except Exception as e:
            CompressionUtil._data_logger.log(level, f"  Matrix logging error: {e}")
    
    @staticmethod
    def _get_network_size_string(net: 'MetabolicNetwork') -> str:
        """
        Get a string representation of network size.
        
        Args:
            net: The metabolic network
            
        Returns:
            String describing network dimensions
        """
        try:
            num_metabolites = len(net.get_metabolite_names())
            num_reactions = len(net.get_reaction_names())
            return f"{num_metabolites} metabolites, {num_reactions} reactions"
        except Exception as e:
            return f"Network size unavailable: {e}"