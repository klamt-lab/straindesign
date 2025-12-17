"""
Metabolic network compression using nullspace-based coupling detection.

This module provides network compression for COBRA models. The main function
is compress_cobra_model() which compresses a model and returns transformation
matrices for converting between original and compressed flux spaces.

API:
    >>> from straindesign.compression import compress_cobra_model
    >>> result = compress_cobra_model(model)
    >>> compressed_model = result.compressed_model

Note: The model should be preprocessed first (rational coefficients,
conservation relations removed). Use networktools.compress_model() for
automatic preprocessing.
"""

import copy
import logging
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union, Any

from fractions import Fraction

# ALL math imports from flint_cmp_interface - NO flint imports in this module
from .flint_cmp_interface import (
    FLINT_AVAILABLE,
    RationalMatrix,
    nullspace,
    float_to_rational,
    basic_columns_from_numpy,
)

LOG = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class CompressionMethod(Enum):
    """Compression methods for metabolic network compression."""
    NULLSPACE = "Nullspace"  # Nullspace-based compression
    RECURSIVE = "Recursive"  # Iterate until no more compression possible

    @classmethod
    def all(cls) -> List['CompressionMethod']:
        return list(cls)

    @classmethod
    def none(cls) -> List['CompressionMethod']:
        return []

    @classmethod
    def standard(cls) -> List['CompressionMethod']:
        """Standard compression methods (recommended)."""
        return [cls.NULLSPACE, cls.RECURSIVE]


# =============================================================================
# Statistics
# =============================================================================

class CompressionStatistics:
    """Tracks compression statistics for logging."""

    def __init__(self):
        self.iteration_count = 0
        self.zero_flux_count = 0
        self.contradicting_count = 0
        self.coupled_count = 0
        self.unused_metabolite_count = 0

    def inc_compression_iteration(self) -> int:
        self.iteration_count += 1
        return self.iteration_count

    def get_compression_iteration(self) -> int:
        return self.iteration_count

    def inc_zero_flux_reactions(self) -> None:
        self.zero_flux_count += 1

    def inc_contradicting_reactions(self) -> None:
        self.contradicting_count += 1

    def inc_coupled_reactions_count(self, count: int) -> None:
        self.coupled_count += count

    def inc_unused_metabolite(self) -> None:
        self.unused_metabolite_count += 1

    def write_to_log(self) -> None:
        LOG.info(f"Compression complete: {self.iteration_count} iterations, "
                 f"{self.zero_flux_count} zero-flux, {self.contradicting_count} contradicting, "
                 f"{self.coupled_count} coupled, {self.unused_metabolite_count} unused metabolites")

    def __repr__(self):
        return (f"CompressionStatistics(iterations={self.iteration_count}, "
                f"zero_flux={self.zero_flux_count}, contradicting={self.contradicting_count}, "
                f"coupled={self.coupled_count})")


# =============================================================================
# Compression Record
# =============================================================================

class CompressionRecord:
    """
    Compression result with transformation matrices.

    Contains: pre @ stoich @ post == cmp
    For EFM expansion: efm_original = post @ efm_compressed
    """

    def __init__(self, pre: RationalMatrix, cmp: RationalMatrix,
                 post: RationalMatrix, reversible: List[bool],
                 meta_names: List[str],
                 stats: Optional[CompressionStatistics] = None):
        self.pre = pre    # metabolite transformation
        self.cmp = cmp    # compressed stoich
        self.post = post  # reaction transformation
        self.reversible = list(reversible)
        self.meta_names = list(meta_names)  # compressed metabolite names (row order in cmp)
        self.stats = stats


# =============================================================================
# Working State (Internal)
# =============================================================================

class _Size:
    """Mutable counter for active matrix dimensions during compression."""
    def __init__(self, metas: int, reacs: int):
        self.metas = metas
        self.reacs = reacs


class _WorkRecord:
    """Mutable state during compression algorithm."""

    def __init__(self, stoich: RationalMatrix, reversible: List[bool],
                 meta_names: List[str], reac_names: List[str]):
        rows, cols = stoich.get_row_count(), stoich.get_column_count()
        self.pre = RationalMatrix.identity(rows)
        self.cmp = stoich.clone()
        self.post = RationalMatrix.identity(cols)
        self.reversible = list(reversible)
        self.meta_names = list(meta_names)
        self.reac_names = list(reac_names)
        self.size = _Size(rows, cols)
        # Store original dimensions for get_truncated()
        self._orig_metas = rows
        self._orig_reacs = cols
        self.stats = CompressionStatistics()
        self.stats.inc_compression_iteration()

    def remove_reaction(self, idx: int) -> None:
        """Remove reaction by swapping to end and decrementing size."""
        self.size.reacs -= 1
        if idx != self.size.reacs:
            self.post.swap_columns(idx, self.size.reacs)
            self.cmp.swap_columns(idx, self.size.reacs)
            self.reversible[idx], self.reversible[self.size.reacs] = \
                self.reversible[self.size.reacs], self.reversible[idx]
            self.reac_names[idx], self.reac_names[self.size.reacs] = \
                self.reac_names[self.size.reacs], self.reac_names[idx]
        # Zero out removed reaction
        for meta in range(self.size.metas):
            self.cmp.set_rational(meta, self.size.reacs, 0, 1)

    def remove_reactions(self, suppressed: Optional[Set[str]]) -> bool:
        """Remove reactions by name - uses batch removal."""
        if not suppressed:
            return False
        indices = set()
        for name in suppressed:
            try:
                idx = self.reac_names.index(name)
                if idx < self.size.reacs:
                    indices.add(idx)
            except ValueError:
                continue
        if not indices:
            return False
        return self.remove_reactions_by_indices(indices)

    def remove_reactions_by_indices(self, indices: Set[int]) -> bool:
        """Remove reactions by index - uses batch removal for efficiency."""
        if not indices:
            return False

        # Filter to only valid indices
        valid_indices = {idx for idx in indices if idx < self.size.reacs}
        if not valid_indices:
            return False

        # Compute indices to keep
        keep_indices = [i for i in range(self.size.reacs) if i not in valid_indices]

        if not keep_indices:
            self.size.reacs = 0
            return True

        # Use batch column removal
        self._batch_remove_columns(keep_indices)
        return True

    def _batch_remove_columns(self, keep_indices: list) -> None:
        """Batch remove columns by keeping only specified indices."""
        LOG.debug(f"_batch_remove_columns: keeping {len(keep_indices)} of {self.size.reacs}")
        LOG.debug(f"  Before: cmp={self.cmp.get_column_count()}, post={self.post.get_column_count()}")

        # Remove columns from post and cmp matrices
        self.post.remove_columns(keep_indices)
        self.cmp.remove_columns(keep_indices)

        LOG.debug(f"  After: cmp={self.cmp.get_column_count()}, post={self.post.get_column_count()}")

        # Reindex names and reversibility
        new_names = [self.reac_names[i] for i in keep_indices]
        new_rev = [self.reversible[i] for i in keep_indices]

        # Update arrays (preserving total length for consistency)
        orig_len = len(self.reac_names)
        for i, (name, rev) in enumerate(zip(new_names, new_rev)):
            self.reac_names[i] = name
            self.reversible[i] = rev

        self.size.reacs = len(keep_indices)
        LOG.debug(f"  size.reacs={self.size.reacs}")

    def _batch_remove_rows(self, keep_indices: list) -> None:
        """Batch remove rows (metabolites) by keeping only specified indices."""
        LOG.debug(f"_batch_remove_rows: keeping {len(keep_indices)} of {self.size.metas}")

        # Remove rows from pre and cmp matrices
        self.pre.remove_rows(keep_indices)
        self.cmp.remove_rows(keep_indices)

        # Reindex metabolite names
        new_names = [self.meta_names[i] for i in keep_indices]
        for i, name in enumerate(new_names):
            self.meta_names[i] = name

        self.size.metas = len(keep_indices)
        LOG.debug(f"  size.metas={self.size.metas}")

    def remove_metabolites_by_indices(self, indices: Set[int]) -> bool:
        """Remove metabolites by index - uses batch removal for efficiency."""
        if not indices:
            return False

        # Filter to only valid indices
        valid_indices = {idx for idx in indices if idx < self.size.metas}
        if not valid_indices:
            return False

        # Compute indices to keep
        keep_indices = [i for i in range(self.size.metas) if i not in valid_indices]

        if not keep_indices:
            self.size.metas = 0
            return True

        # Use batch row removal
        self._batch_remove_rows(keep_indices)
        return True

    def remove_metabolite(self, idx: int) -> None:
        """Remove metabolite by swapping to end and decrementing size."""
        self.size.metas -= 1
        if idx != self.size.metas:
            self.pre.swap_rows(idx, self.size.metas)
            self.cmp.swap_rows(idx, self.size.metas)
            self.meta_names[idx], self.meta_names[self.size.metas] = \
                self.meta_names[self.size.metas], self.meta_names[idx]
        # Zero out removed metabolite
        for reac in range(self.size.reacs):
            self.cmp.set_rational(self.size.metas, reac, 0, 1)

    def remove_unused_metabolites(self) -> bool:
        """Remove metabolites with all-zero rows - uses batch removal.

        Uses sparse structure for O(1) zero-row detection per metabolite.
        """
        # Get active submatrix to check only active columns
        active_cmp = self.cmp.submatrix(self.size.metas, self.size.reacs)

        # Find zero rows using sparse structure (O(1) per row with CSR)
        unused_indices = set()
        if hasattr(active_cmp, '_numerators') and active_cmp._numerators is not None:
            # FLINT sparse path: zero row = no entries in CSR indptr range
            indptr = active_cmp._numerators.indptr
            for meta in range(self.size.metas):
                if indptr[meta] == indptr[meta + 1]:
                    unused_indices.add(meta)
                    self.stats.inc_unused_metabolite()
        else:
            # Fallback: iterate columns
            for meta in range(self.size.metas):
                has_nonzero = False
                for reac in range(self.size.reacs):
                    if self.cmp.get_signum(meta, reac) != 0:
                        has_nonzero = True
                        break
                if not has_nonzero:
                    unused_indices.add(meta)
                    self.stats.inc_unused_metabolite()

        # Batch remove all unused metabolites at once
        if unused_indices:
            self.remove_metabolites_by_indices(unused_indices)
            return True
        return False

    def get_truncated(self) -> CompressionRecord:
        """Create final CompressionRecord with truncated matrices."""
        mc, rc = self.size.metas, self.size.reacs
        # Use stored original dimensions (batch removal changes matrix sizes)
        m = self._orig_metas
        r = self._orig_reacs

        # Create truncated matrices using efficient submatrix extraction
        # pre: mc compressed metabolites × m original metabolites
        # cmp: mc compressed metabolites × rc compressed reactions
        # post: r original reactions × rc compressed reactions
        pre_trunc = self.pre.submatrix(mc, m)
        cmp_trunc = self.cmp.submatrix(mc, rc)
        post_trunc = self.post.submatrix(r, rc)

        rev_trunc = self.reversible[:rc]
        meta_names_trunc = self.meta_names[:mc]
        return CompressionRecord(pre_trunc, cmp_trunc, post_trunc, rev_trunc, meta_names_trunc, self.stats)


# =============================================================================
# Core Algorithm
# =============================================================================

class StoichMatrixCompressor:
    """Nullspace-based metabolic network compression."""

    def __init__(self, *methods: CompressionMethod):
        self._methods = list(methods) if methods else CompressionMethod.standard()

    def compress(self, stoich: RationalMatrix, reversible: List[bool],
                 meta_names: List[str], reac_names: List[str],
                 suppressed: Optional[Set[str]] = None) -> CompressionRecord:
        """Compress network, return transformation matrices.

        Uses Java efmtool's two-phase approach:
        1. Phase 1: Remove zero-flux and contradicting reactions (iteratively)
        2. Phase 2: Combine coupled reactions (iteratively)

        This separation prevents cascading effects where combining reactions
        could create new coupling patterns that weren't present in the original.
        """
        work = _WorkRecord(stoich, reversible, meta_names, reac_names)

        do_nullspace = CompressionMethod.NULLSPACE in self._methods
        do_recursive = CompressionMethod.RECURSIVE in self._methods

        work.remove_reactions(suppressed)

        if do_nullspace:
            # Phase 1: Remove zero-flux and contradicting reactions only
            # (matches Java's inclCompression=false phase)
            # Optimization: Continue only if nullspace compression found reactions.
            # Removing unused metabolites (all-zero rows) doesn't change the nullspace,
            # so if _nullspace_compress returns False, the next iteration would too.
            while True:
                work.stats.inc_compression_iteration()
                work.remove_unused_metabolites()
                found = self._nullspace_compress(work, incl_compression=False)
                if not (found and do_recursive):
                    break

            # Phase 2: Combine coupled reactions
            # (matches Java's inclCompression=true phase)
            while True:
                work.stats.inc_compression_iteration()
                work.remove_unused_metabolites()
                found = self._nullspace_compress(work, incl_compression=True)
                if not (found and do_recursive):
                    break

        work.remove_unused_metabolites()
        work.stats.write_to_log()
        return work.get_truncated()

    def _nullspace_compress(self, work: _WorkRecord, incl_compression: bool = True) -> bool:
        """One pass of nullspace compression. Returns True if reactions removed.

        Args:
            incl_compression: If False, only remove zero-flux and contradicting.
                              If True, only combine coupled reactions.
        """
        # Build active submatrix for nullspace computation
        active = work.cmp.submatrix(work.size.metas, work.size.reacs)
        kernel = nullspace(active)
        LOG.debug(f"Nullspace: {active.get_row_count()}x{active.get_column_count()} -> kernel {kernel.get_row_count()}x{kernel.get_column_count()}")

        # Convert kernel to sparse CSR once (179x faster pattern building)
        kernel_sparse, kernel_denom = kernel.to_sparse_csr()

        changed = False
        if not incl_compression:
            # Phase 1: Remove zero-flux and contradicting
            # IMPORTANT: Find both sets BEFORE any removal to avoid index mismatch!
            zero_flux = self._find_zero_flux(work, kernel_sparse)
            contradicting = self._find_contradicting(work, kernel_sparse)

            # Remove all at once
            all_to_remove = zero_flux | contradicting
            LOG.debug(f"Phase 1: {len(zero_flux)} zero-flux + {len(contradicting)} contradicting = {len(all_to_remove)} to remove")
            if all_to_remove:
                work.remove_reactions_by_indices(all_to_remove)
                changed = True
        else:
            # Phase 2: Combine coupled reactions
            changed |= self._handle_coupled_combine(work, kernel_sparse)

        if changed:
            work.remove_unused_metabolites()

        return changed

    def _find_zero_flux(self, work: _WorkRecord, kernel_sparse) -> set:
        """Find reactions with all-zero kernel rows (indices in current work)."""
        zero_flux = set()
        for reac in range(work.size.reacs):
            if kernel_sparse.indptr[reac] == kernel_sparse.indptr[reac + 1]:
                zero_flux.add(reac)
                work.stats.inc_zero_flux_reactions()
        return zero_flux

    def _find_coupled_groups(self, kernel_sparse, num_reacs):
        """Find groups of coupled reactions from kernel sparsity pattern.

        Returns (groups, ratios) where:
        - groups: list of lists, each containing indices of coupled reactions
        - ratios: list where ratios[i] is the coupling ratio for reaction i
        """
        # Group reactions by zero-pattern in kernel
        patterns = {}
        for reac in range(num_reacs):
            start = kernel_sparse.indptr[reac]
            end = kernel_sparse.indptr[reac + 1]
            pattern = tuple(kernel_sparse.indices[start:end])
            patterns.setdefault(pattern, []).append(reac)

        potential_groups = [idxs for idxs in patterns.values() if len(idxs) > 1]

        groups = []
        ratios = [None] * num_reacs

        for potential in potential_groups:
            ref_reac = potential[0]
            ref_start = kernel_sparse.indptr[ref_reac]
            ref_end = kernel_sparse.indptr[ref_reac + 1]
            nonzero_count = ref_end - ref_start
            if nonzero_count == 0:
                continue

            for i, reac_a in enumerate(potential):
                if ratios[reac_a] is not None:
                    continue
                group = None

                a_start = kernel_sparse.indptr[reac_a]
                a_vals = kernel_sparse.data[a_start:a_start + nonzero_count]

                for j in range(i + 1, len(potential)):
                    reac_b = potential[j]
                    if ratios[reac_b] is not None:
                        continue

                    b_start = kernel_sparse.indptr[reac_b]
                    b_vals = kernel_sparse.data[b_start:b_start + nonzero_count]

                    ratio_num = int(a_vals[0])
                    ratio_den = int(b_vals[0])
                    if ratio_den < 0:
                        ratio_num, ratio_den = -ratio_num, -ratio_den

                    is_consistent = True
                    for k in range(1, nonzero_count):
                        curr_num = int(a_vals[k])
                        curr_den = int(b_vals[k])
                        if curr_den < 0:
                            curr_num, curr_den = -curr_num, -curr_den
                        if ratio_num * curr_den != ratio_den * curr_num:
                            is_consistent = False
                            break

                    if is_consistent:
                        ratios[reac_b] = Fraction(ratio_num, ratio_den)
                        if group is None:
                            group = [reac_a]
                        group.append(reac_b)

                if group is not None:
                    groups.append(group)

        return groups, ratios

    def _find_contradicting(self, work: _WorkRecord, kernel_sparse) -> set:
        """Find reactions in contradicting coupled groups (indices in current work)."""
        groups, ratios = self._find_coupled_groups(kernel_sparse, work.size.reacs)

        # Find contradicting reactions
        contradicting = set()
        for group in groups:
            is_consistent = self._check_coupling_consistency(group, ratios, work.reversible)
            if not is_consistent:
                for idx in group:
                    contradicting.add(idx)
                    work.stats.inc_contradicting_reactions()

        return contradicting

    def _handle_coupled_combine(self, work: _WorkRecord, kernel_sparse) -> bool:
        """Phase 2: Combine consistent coupled reactions, remove contradicting.

        Returns True if any contradicting reactions were removed (flux space changed),
        which means new couplings might be revealed. Returns False if only merging
        happened, since merging doesn't change the flux space.
        """
        groups, ratios = self._find_coupled_groups(kernel_sparse, work.size.reacs)

        # Combine consistent groups, remove contradicting groups
        reactions_to_remove = set()
        contradicting_removed = False

        # Enter batch edit mode for cmp and post matrices to avoid repeated LIL conversions
        work.cmp.begin_batch_edit()
        work.post.begin_batch_edit()

        for group in groups:
            is_consistent = self._check_coupling_consistency(group, ratios, work.reversible)
            if is_consistent:
                self._combine_coupled(work, group, ratios)
                for idx in group[1:]:
                    reactions_to_remove.add(idx)
            else:
                # Contradicting coupled group: reactions cannot carry flux together
                # due to irreversibility constraints. Remove them to prune blocked paths,
                # which may reveal new couplings in subsequent iterations.
                LOG.debug(f"Removing contradicting coupled group: {[work.reac_names[r] for r in group]}")
                for idx in group:
                    reactions_to_remove.add(idx)
                    work.stats.inc_contradicting_reactions()
                contradicting_removed = True

        # End batch edit mode
        work.cmp.end_batch_edit()
        work.post.end_batch_edit()

        LOG.debug(f"Phase 2: {len(groups)} groups, removing {len(reactions_to_remove)} reactions, contradicting={contradicting_removed}")
        work.remove_reactions_by_indices(reactions_to_remove)

        # Only return True if contradicting reactions were removed (flux space changed).
        # Merging alone doesn't change the flux space, so no new couplings can emerge.
        return contradicting_removed

    def _check_coupling_consistency(self, group: List[int], ratios: List[Optional[Fraction]],
                                    reversible: List[bool]) -> bool:
        """Check if coupled group is consistent with reversibility."""
        for forward in [True, False]:
            all_consistent = forward or reversible[group[0]]
            for idx in group[1:]:
                ratio = ratios[idx]
                ratio_positive = ratio.numerator * ratio.denominator > 0
                if not ((forward == ratio_positive) or reversible[idx]):
                    all_consistent = False
                    break
            if all_consistent:
                return True
        return False

    def _combine_coupled(self, work: _WorkRecord, group: List[int],
                         ratios: List[Optional[Fraction]]) -> None:
        """Combine coupled reactions into master reaction.

        Uses batch column operations with native fmpq arithmetic for performance.
        """
        master = group[0]
        LOG.debug(f"Combining coupled: {[work.reac_names[r] for r in group]}")

        for slave in group[1:]:
            ratio = ratios[slave]
            # multiplier = 1/ratio = ratio.denominator / ratio.numerator
            mult_num, mult_den = ratio.denominator, ratio.numerator

            # Update stoichiometric matrix: cmp[:, master] += cmp[:, slave] * mult
            work.cmp.add_scaled_column(master, slave, mult_num, mult_den)

            # Update post matrix: post[:, master] += post[:, slave] * mult
            work.post.add_scaled_column(master, slave, mult_num, mult_den)

        work.stats.inc_coupled_reactions_count(len(group))


# =============================================================================
# COBRA Interface
# =============================================================================

class CompressionResult:
    """Result of COBRA model compression."""

    def __init__(self, compressed_model, compression_converter, pre_matrix, post_matrix,
                 reaction_map, metabolite_map, statistics, methods_used,
                 original_reaction_names, original_metabolite_names, flipped_reactions):
        self.compressed_model = compressed_model
        self.compression_converter = compression_converter
        self.pre_matrix = pre_matrix
        self.post_matrix = post_matrix
        self.reaction_map = reaction_map
        self.metabolite_map = metabolite_map
        self.statistics = statistics
        self.methods_used = methods_used
        self.original_reaction_names = original_reaction_names
        self.original_metabolite_names = original_metabolite_names
        self.flipped_reactions = flipped_reactions

    @property
    def compression_ratio(self) -> float:
        return len(self.compressed_model.reactions) / len(self.original_reaction_names)

    @property
    def reactions_removed(self) -> int:
        return len(self.original_reaction_names) - len(self.compressed_model.reactions)

    def summary(self) -> str:
        return f"Compressed {len(self.original_reaction_names)} -> {len(self.compressed_model.reactions)} reactions ({self.compression_ratio:.1%})"


class CompressionConverter:
    """Bidirectional transformer for expressions between original and compressed spaces."""

    def __init__(self, reaction_map: Dict[str, Dict[str, Union[float, Fraction]]],
                 metabolite_map: Dict[str, Dict[str, Union[float, Fraction]]],
                 flipped_reactions: List[str]):
        self.reaction_map = reaction_map
        self.metabolite_map = metabolite_map
        self.flipped_reactions = set(flipped_reactions)

    def expand_expression(self, expression: Dict[str, float],
                          remove_missing: bool = False) -> Dict[str, float]:
        """Transform expression from compressed back to original space."""
        expanded = {}
        for comp_rxn, comp_coeff in expression.items():
            if comp_rxn in self.reaction_map:
                for orig_rxn, scale in self.reaction_map[comp_rxn].items():
                    coeff = comp_coeff * float(scale)
                    if orig_rxn in self.flipped_reactions:
                        coeff = -coeff
                    expanded[orig_rxn] = expanded.get(orig_rxn, 0) + coeff
            elif not remove_missing:
                LOG.warning(f"Compressed reaction {comp_rxn} not found")
        return expanded


def remove_conservation_relations(model) -> None:
    """Remove conservation relations (dependent metabolites) from a model.

    This reduces the number of metabolites while maintaining the original flux space.
    Uses exact rational arithmetic to find linearly independent rows.

    Args:
        model: COBRA model to modify in-place
    """
    from cobra.util.array import create_stoichiometric_matrix
    stoich_mat = create_stoichiometric_matrix(model, array_type='lil')
    basic_mets = basic_columns_from_numpy(stoich_mat.transpose().toarray())
    dependent_mets = [model.metabolites[i].id
                      for i in set(range(len(model.metabolites))) - set(basic_mets)]
    for m_id in dependent_mets:
        model.metabolites.get_by_id(m_id).remove_from_model()


def compress_cobra_model(
    model,
    methods: Optional[List[Union[str, CompressionMethod]]] = None,
    in_place: bool = True,
    suppressed_reactions: Optional[Set[str]] = None
) -> CompressionResult:
    """
    Compress a COBRA model using nullspace-based coupling detection.

    Note: Model should be preprocessed first (rational coefficients,
    conservation relations removed). Use networktools.compress_model()
    for automatic preprocessing.

    Args:
        model: COBRA model to compress (should be preprocessed)
        methods: Compression methods. Default: CompressionMethod.standard()
        in_place: Modify original model (True) or work on copy (False)
        suppressed_reactions: Reaction names to exclude from compression

    Returns:
        CompressionResult with compressed model and transformation data
    """
    if not in_place:
        model = copy.deepcopy(model)

    original_reaction_names = [r.id for r in model.reactions]
    original_metabolite_names = [m.id for m in model.metabolites]

    # Parse methods
    if methods is None:
        compression_methods = CompressionMethod.standard()
    else:
        compression_methods = []
        for method in methods:
            if isinstance(method, CompressionMethod):
                compression_methods.append(method)
            elif isinstance(method, str):
                compression_methods.append(CompressionMethod[method.upper()])

    # Flip reactions that can only run backwards
    flipped_reactions = []
    for rxn in model.reactions:
        if rxn.upper_bound <= 0:
            rxn *= -1
            flipped_reactions.append(rxn.id)

    # Build stoichiometric matrix with exact arithmetic
    stoich_matrix = RationalMatrix.from_cobra_model(model)
    reversible = [rxn.reversibility for rxn in model.reactions]
    metabolite_names = [m.id for m in model.metabolites]
    reaction_names = [r.id for r in model.reactions]

    # Run compression
    compressor = StoichMatrixCompressor(*compression_methods)
    compression_record = compressor.compress(
        stoich_matrix, reversible, metabolite_names, reaction_names, suppressed_reactions
    )

    # Apply to model (uses direct manipulation, bypasses solver)
    reaction_map, objective_updates = _apply_compression_to_model(
        model, compression_record, reaction_names
    )

    # Rebuild solver after all direct modifications
    _rebuild_solver(model, objective_updates)

    pre_matrix = compression_record.pre.to_numpy()
    post_matrix = compression_record.post.to_numpy()

    converter = CompressionConverter(reaction_map, {}, flipped_reactions)

    return CompressionResult(
        compressed_model=model,
        compression_converter=converter,
        pre_matrix=pre_matrix,
        post_matrix=post_matrix,
        reaction_map=reaction_map,
        metabolite_map={},
        statistics=compression_record.stats,
        methods_used=compression_methods,
        original_reaction_names=original_reaction_names,
        original_metabolite_names=original_metabolite_names,
        flipped_reactions=flipped_reactions
    )


def _rebuild_solver(model, objective_updates: dict) -> None:
    """Rebuild solver after direct model modifications.

    This creates a fresh solver and populates it with the current model state.
    Must be called after using direct attribute manipulation that bypasses
    solver updates (e.g., _metabolites, _lower_bound, _upper_bound).

    Args:
        model: COBRA model with stale solver state
        objective_updates: dict mapping reactions to their objective coefficients
    """
    # Get solver interface type
    solver_interface = model.solver.interface

    # Create fresh solver
    new_solver = solver_interface.Model()
    model._solver = new_solver

    # Populate solver with current model state
    model._populate_solver(model.reactions, model.metabolites)

    # Set objective coefficients (must be done after solver is populated)
    for rxn, obj_coeff in objective_updates.items():
        if rxn in model.reactions:
            rxn.objective_coefficient = obj_coeff


def _apply_compression_to_model(model, compression_record, original_reaction_names):
    """Apply compression results to COBRA model using exact Fraction arithmetic.

    Sets stoichiometric coefficients and bounds as Fractions directly from
    the compressed matrices, avoiding floating-point arithmetic.

    Uses direct attribute manipulation to bypass solver updates during modification.
    Solver must be rebuilt after calling this function.
    """
    post = compression_record.post
    cmp = compression_record.cmp
    meta_names = compression_record.meta_names
    num_original = post.get_row_count()
    num_compressed = post.get_column_count()
    num_metas = cmp.get_row_count()

    # Track which original reactions to keep (main reactions)
    keep_rxns = [False] * num_original
    reaction_map = {}

    # Build metabolite lookup for compressed metabolites
    met_lookup = {name: model.metabolites.get_by_id(name) for name in meta_names}

    # Track objective coefficients to set at end (bypass solver updates)
    objective_updates = {}

    for j in range(num_compressed):
        # Find contributing original reactions from POST matrix (sparse iteration)
        contributing = list(post.iter_column_fractions(j))

        if not contributing:
            continue

        # Select "main" reaction: prefer one with non-zero objective coefficient
        main_idx = contributing[0][0]
        for idx, _ in contributing:
            if model.reactions[idx].objective_coefficient != 0:
                main_idx = idx
                break
        main_rxn = model.reactions[main_idx]
        keep_rxns[main_idx] = True

        # Scale objective coefficient by POST factors (store for later)
        combined_obj = Fraction(0)
        for idx, coeff in contributing:
            obj_coeff = model.reactions[idx].objective_coefficient
            if obj_coeff != 0:
                combined_obj += Fraction(obj_coeff).limit_denominator(10**12) * coeff
        objective_updates[main_rxn] = float(combined_obj)

        # Store subset info
        main_rxn.subset_rxns = [idx for idx, _ in contributing]
        main_rxn.subset_stoich = [coeff for _, coeff in contributing]

        # Build combined reaction name from contributing reactions
        for idx, _ in contributing[1:]:
            rxn = model.reactions[idx]
            if len(main_rxn.id) + len(rxn.id) < 220 and not main_rxn.id.endswith('...'):
                main_rxn.id += '*' + rxn.id
            elif not main_rxn.id.endswith('...'):
                main_rxn.id += '...'

        # Build new metabolites dict from cmp matrix (sparse iteration)
        new_metabolites = {}
        for m_idx, frac in cmp.iter_column_fractions(j):
            met = met_lookup[meta_names[m_idx]]
            new_metabolites[met] = frac

        # OPTIMIZATION: Direct _metabolites assignment (bypasses solver updates)
        # Clear old metabolite back-references
        for met in list(main_rxn._metabolites.keys()):
            if main_rxn in met._reaction:
                met._reaction.discard(main_rxn)
        # Set new metabolites directly
        main_rxn._metabolites = new_metabolites
        # Update new metabolite back-references
        for met in new_metabolites:
            met._reaction.add(main_rxn)

        # Compute bounds as Fractions from original reactions scaled by POST factors
        lb_candidates = []
        ub_candidates = []
        for idx, coeff in contributing:
            rxn = model.reactions[idx]
            lb, ub = rxn.lower_bound, rxn.upper_bound

            # v_original = coeff * v_compressed (from POST interpretation)
            # constraint: lb <= v_original <= ub
            # so: lb <= coeff * v_compressed <= ub
            # if coeff > 0: lb/coeff <= v_compressed <= ub/coeff
            # if coeff < 0: ub/coeff <= v_compressed <= lb/coeff (inequality flips)

            if coeff > 0:
                if lb != -float('inf'):
                    lb_candidates.append(Fraction(lb) / coeff if lb != 0 else Fraction(0))
                if ub != float('inf'):
                    ub_candidates.append(Fraction(ub) / coeff if ub != 0 else Fraction(0))
            else:  # coeff < 0
                if ub != float('inf'):
                    lb_candidates.append(Fraction(ub) / coeff if ub != 0 else Fraction(0))
                if lb != -float('inf'):
                    ub_candidates.append(Fraction(lb) / coeff if lb != 0 else Fraction(0))

        # OPTIMIZATION: Direct bounds assignment (bypasses solver updates)
        main_rxn._lower_bound = float(max(lb_candidates)) if lb_candidates else -float('inf')
        main_rxn._upper_bound = float(min(ub_candidates)) if ub_candidates else float('inf')

        # Build reaction map
        reaction_map[main_rxn.id] = {
            original_reaction_names[idx]: coeff for idx, coeff in contributing
        }

    # OPTIMIZATION: Batch reaction and metabolite removal
    # Use reaction objects directly (not IDs, since IDs are modified during compression)
    from cobra import DictList

    # Build set of reaction objects to keep
    keep_rxn_objs = {model.reactions[i] for i in range(num_original) if keep_rxns[i]}

    # Find metabolites that will be in kept reactions
    mets_in_kept_rxns = set()
    for rxn in keep_rxn_objs:
        mets_in_kept_rxns.update(rxn._metabolites.keys())

    # Build new reactions list (only kept reactions)
    new_reactions = DictList()
    for rxn in model.reactions:
        if rxn in keep_rxn_objs:
            new_reactions.append(rxn)
        else:
            # Clear metabolite back-references for removed reaction
            for met in list(rxn._metabolites.keys()):
                if rxn in met._reaction:
                    met._reaction.discard(rxn)
            rxn._model = None

    # Build new metabolites list (only those in kept reactions)
    new_metabolites = DictList()
    for met in model.metabolites:
        if met in mets_in_kept_rxns:
            new_metabolites.append(met)
        else:
            met._model = None

    # Replace model lists (using __dict__ to bypass any property setters)
    model.__dict__['reactions'] = new_reactions
    model.__dict__['metabolites'] = new_metabolites

    # Update model references
    for rxn in model.reactions:
        rxn._model = model
    for met in model.metabolites:
        met._model = model

    return reaction_map, objective_updates


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'compress_cobra_model',
    'CompressionResult',
    'CompressionRecord',
    'CompressionStatistics',
    'CompressionMethod',
    'CompressionConverter',
    'StoichMatrixCompressor',
]
