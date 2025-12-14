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

# ALL math imports from flint_interface - NO flint imports in this module
from .flint_interface import (
    FLINT_AVAILABLE,
    RationalMatrix,
    nullspace,
    float_to_rational,
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
                 stats: Optional[CompressionStatistics] = None):
        self.pre = pre    # metabolite transformation
        self.cmp = cmp    # compressed stoich
        self.post = post  # reaction transformation
        self.reversible = list(reversible)
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
        """Remove reactions by name."""
        if not suppressed:
            return False
        indices = []
        for name in suppressed:
            try:
                idx = self.reac_names.index(name)
                if idx < self.size.reacs:
                    indices.append(idx)
            except ValueError:
                continue
        if not indices:
            return False
        for idx in sorted(indices, reverse=True):
            self.remove_reaction(idx)
        return True

    def remove_reactions_by_indices(self, indices: Set[int]) -> bool:
        """Remove reactions by index."""
        if not indices:
            return False
        for idx in sorted(indices, reverse=True):
            if idx < self.size.reacs:
                self.remove_reaction(idx)
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
        """Remove metabolites with all-zero rows."""
        removed_any = False
        meta = 0
        while meta < self.size.metas:
            has_nonzero = False
            for reac in range(self.size.reacs):
                if self.cmp.get_signum(meta, reac) != 0:
                    has_nonzero = True
                    break
            if not has_nonzero:
                self.remove_metabolite(meta)
                self.stats.inc_unused_metabolite()
                removed_any = True
            else:
                meta += 1
        return removed_any

    def get_truncated(self) -> CompressionRecord:
        """Create final CompressionRecord with truncated matrices."""
        mc, rc = self.size.metas, self.size.reacs
        m = self.cmp.get_row_count()
        r = self.cmp.get_column_count()

        # Create truncated matrices
        pre_trunc = RationalMatrix(mc, m)
        cmp_trunc = RationalMatrix(mc, rc)
        post_trunc = RationalMatrix(r, rc)

        # Copy values
        for i in range(mc):
            for j in range(m):
                pre_trunc.set_rational(i, j, self.pre.get_numerator(i, j),
                                       self.pre.get_denominator(i, j))
        for i in range(mc):
            for j in range(rc):
                cmp_trunc.set_rational(i, j, self.cmp.get_numerator(i, j),
                                       self.cmp.get_denominator(i, j))
        for i in range(r):
            for j in range(rc):
                post_trunc.set_rational(i, j, self.post.get_numerator(i, j),
                                        self.post.get_denominator(i, j))

        rev_trunc = self.reversible[:rc]
        return CompressionRecord(pre_trunc, cmp_trunc, post_trunc, rev_trunc, self.stats)


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
            while True:
                work.stats.inc_compression_iteration()
                initial_metas = work.size.metas
                work.remove_unused_metabolites()
                compressed_any = (work.size.metas < initial_metas)
                compressed_any |= self._nullspace_compress(work, incl_compression=False)
                if not (compressed_any and do_recursive):
                    break

            # Phase 2: Combine coupled reactions
            # (matches Java's inclCompression=true phase)
            while True:
                work.stats.inc_compression_iteration()
                initial_metas = work.size.metas
                work.remove_unused_metabolites()
                compressed_any = (work.size.metas < initial_metas)
                compressed_any |= self._nullspace_compress(work, incl_compression=True)
                if not (compressed_any and do_recursive):
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
        active = RationalMatrix(work.size.metas, work.size.reacs)
        for i in range(work.size.metas):
            for j in range(work.size.reacs):
                active.set_rational(i, j, work.cmp.get_numerator(i, j),
                                    work.cmp.get_denominator(i, j))

        kernel = nullspace(active)
        LOG.debug(f"Nullspace kernel: {kernel.get_row_count()}x{kernel.get_column_count()}")

        changed = False
        if not incl_compression:
            # Phase 1: Remove zero-flux and contradicting
            changed |= self._remove_zero_flux(work, kernel)
            changed |= self._handle_coupled(work, kernel, incl_compression=False)
        else:
            # Phase 2: Combine coupled reactions
            changed |= self._handle_coupled(work, kernel, incl_compression=True)

        if changed:
            work.remove_unused_metabolites()

        return changed

    def _remove_zero_flux(self, work: _WorkRecord, kernel: RationalMatrix) -> bool:
        """Remove reactions with all-zero kernel rows."""
        kernel_cols = kernel.get_column_count()
        zero_flux = []

        for reac in range(work.size.reacs):
            all_zero = True
            for col in range(kernel_cols):
                if kernel.get_numerator(reac, col) != 0:
                    all_zero = False
                    break
            if all_zero:
                zero_flux.append(reac)
                LOG.debug(f"Zero flux reaction: {work.reac_names[reac]}")

        for idx in reversed(zero_flux):
            work.remove_reaction(idx)
            work.stats.inc_zero_flux_reactions()

        return len(zero_flux) > 0

    def _handle_coupled(self, work: _WorkRecord, kernel: RationalMatrix,
                        incl_compression: bool = True) -> bool:
        """Find and handle coupled reactions.

        Args:
            incl_compression: If False, only remove contradicting reactions.
                              If True, only combine consistent coupled reactions.
        """
        kernel_cols = kernel.get_column_count()
        num_reacs = work.size.reacs

        # Group by zero-pattern
        patterns = {}
        for reac in range(num_reacs):
            pattern = tuple(kernel.get_numerator(reac, col) == 0 for col in range(kernel_cols))
            if pattern not in patterns:
                patterns[pattern] = []
            patterns[pattern].append(reac)

        potential_groups = [idxs for idxs in patterns.values() if len(idxs) > 1]

        groups = []
        ratios = [None] * num_reacs

        for potential in potential_groups:
            ref_reac = potential[0]
            nonzero_cols = [c for c in range(kernel_cols) if kernel.get_numerator(ref_reac, c) != 0]
            if not nonzero_cols:
                continue

            for i, reac_a in enumerate(potential):
                if ratios[reac_a] is not None:
                    continue
                group = None

                for j in range(i + 1, len(potential)):
                    reac_b = potential[j]
                    if ratios[reac_b] is not None:
                        continue

                    # Check ratio consistency
                    ratio_num, ratio_den = None, None
                    for col in nonzero_cols:
                        num_a = kernel.get_numerator(reac_a, col)
                        den_a = kernel.get_denominator(reac_a, col)
                        num_b = kernel.get_numerator(reac_b, col)
                        den_b = kernel.get_denominator(reac_b, col)

                        # ratio = (num_a/den_a) / (num_b/den_b) = (num_a*den_b) / (den_a*num_b)
                        curr_num = num_a * den_b
                        curr_den = den_a * num_b
                        if curr_den < 0:
                            curr_num, curr_den = -curr_num, -curr_den

                        if ratio_num is None:
                            ratio_num, ratio_den = curr_num, curr_den
                        else:
                            # Check: ratio_num/ratio_den == curr_num/curr_den
                            if ratio_num * curr_den != ratio_den * curr_num:
                                ratio_num = 0
                                break

                    if ratio_num is not None and ratio_num != 0:
                        ratios[reac_b] = Fraction(ratio_num, ratio_den)
                        if group is None:
                            group = [reac_a]
                        group.append(reac_b)

                if group is not None:
                    groups.append(group)

        # Process groups based on phase
        reactions_to_remove = set()
        changed = False

        for group in groups:
            is_consistent = self._check_coupling_consistency(group, ratios, work.reversible)

            if not incl_compression:
                # Phase 1: Only remove contradicting reactions
                if not is_consistent:
                    LOG.debug(f"Contradicting coupled: {[work.reac_names[r] for r in group]}")
                    for idx in group:
                        reactions_to_remove.add(idx)
                        work.stats.inc_contradicting_reactions()
                    changed = True
            else:
                # Phase 2: Only combine consistent coupled reactions
                # (contradicting were already removed in phase 1)
                if is_consistent:
                    self._combine_coupled(work, group, ratios)
                    for idx in group[1:]:
                        reactions_to_remove.add(idx)
                    changed = True

        work.remove_reactions_by_indices(reactions_to_remove)
        return changed

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
        """Combine coupled reactions into master reaction."""
        master = group[0]
        LOG.debug(f"Combining coupled: {[work.reac_names[r] for r in group]}")

        for slave in group[1:]:
            ratio = ratios[slave]
            # multiplier = 1/ratio = ratio.denominator / ratio.numerator
            mult_num, mult_den = ratio.denominator, ratio.numerator

            # Update stoichiometric matrix
            for meta in range(work.size.metas):
                m_num = work.cmp.get_numerator(meta, master)
                m_den = work.cmp.get_denominator(meta, master)
                s_num = work.cmp.get_numerator(meta, slave)
                s_den = work.cmp.get_denominator(meta, slave)
                # new = m + s*mult = (m_num*m_den_s + s_num*mult_num*m_den) / (m_den*s_den*mult_den)
                # Use Fraction arithmetic (faster than sympy.Rational)
                m_val = Fraction(m_num, m_den)
                s_val = Fraction(s_num, s_den)
                mult = Fraction(mult_num, mult_den)
                new_val = m_val + s_val * mult
                work.cmp.set_rational(meta, master, new_val.numerator, new_val.denominator)

            # Update post matrix
            for orig in range(work.post.get_row_count()):
                m_num = work.post.get_numerator(orig, master)
                m_den = work.post.get_denominator(orig, master)
                s_num = work.post.get_numerator(orig, slave)
                s_den = work.post.get_denominator(orig, slave)
                m_val = Fraction(m_num, m_den)
                s_val = Fraction(s_num, s_den)
                mult = Fraction(mult_num, mult_den)
                new_val = m_val + s_val * mult
                work.post.set_rational(orig, master, new_val.numerator, new_val.denominator)

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
    stoich_matrix = _build_stoich_matrix(model)
    reversible = [rxn.reversibility for rxn in model.reactions]
    metabolite_names = [m.id for m in model.metabolites]
    reaction_names = [r.id for r in model.reactions]

    # Run compression
    compressor = StoichMatrixCompressor(*compression_methods)
    compression_record = compressor.compress(
        stoich_matrix, reversible, metabolite_names, reaction_names, suppressed_reactions
    )

    # Apply to model
    reaction_map = _apply_compression_to_model(model, compression_record, reaction_names)

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


def _build_stoich_matrix(model) -> RationalMatrix:
    """Build stoichiometric matrix with exact rational arithmetic."""
    num_mets = len(model.metabolites)
    num_rxns = len(model.reactions)
    matrix = RationalMatrix(num_mets, num_rxns)

    for j, rxn in enumerate(model.reactions):
        for met, coeff in rxn.metabolites.items():
            i = model.metabolites.index(met.id)
            if hasattr(coeff, 'p'):  # sympy.Rational
                matrix.set_rational(i, j, int(coeff.p), int(coeff.q))
            elif hasattr(coeff, 'numerator'):  # fractions.Fraction
                matrix.set_rational(i, j, int(coeff.numerator), int(coeff.denominator))
            else:
                rat = float_to_rational(coeff)
                matrix.set_rational(i, j, int(rat.numerator), int(rat.denominator))

    return matrix


def _apply_compression_to_model(model, compression_record, original_reaction_names):
    """Apply compression results to COBRA model, return reaction map."""
    post = compression_record.post
    num_original = post.get_row_count()
    num_compressed = post.get_column_count()

    post_matrix_np = post.to_numpy()

    del_rxns = np.logical_not(np.any(post_matrix_np, axis=1))
    reaction_map = {}

    for j in range(num_compressed):
        rxn_indices = post_matrix_np[:, j].nonzero()[0]
        if len(rxn_indices) == 0:
            continue

        main_idx = rxn_indices[0]
        main_rxn = model.reactions[main_idx]

        main_rxn.subset_rxns = list(rxn_indices)
        main_rxn.subset_stoich = [
            float_to_rational(post_matrix_np[r_idx, j])
            for r_idx in rxn_indices
        ]

        for r_idx in rxn_indices:
            factor_float = post_matrix_np[r_idx, j]
            if factor_float == 0:
                continue
            factor = float_to_rational(factor_float)
            rxn = model.reactions[r_idx]
            rxn *= factor
            if rxn.lower_bound not in (0, -float('inf')):
                rxn.lower_bound /= abs(factor_float)
            if rxn.upper_bound not in (0, float('inf')):
                rxn.upper_bound /= abs(factor_float)

        for r_idx in rxn_indices[1:]:
            rxn = model.reactions[r_idx]
            if len(main_rxn.id) + len(rxn.id) < 220 and not main_rxn.id.endswith('...'):
                main_rxn.id += '*' + rxn.id
            elif not main_rxn.id.endswith('...'):
                main_rxn.id += '...'
            main_rxn += rxn
            if rxn.lower_bound > main_rxn.lower_bound:
                main_rxn.lower_bound = rxn.lower_bound
            if rxn.upper_bound < main_rxn.upper_bound:
                main_rxn.upper_bound = rxn.upper_bound
            del_rxns[r_idx] = True

        reaction_map[main_rxn.id] = {
            original_reaction_names[r_idx]: main_rxn.subset_stoich[i]
            for i, r_idx in enumerate(rxn_indices)
        }

    for i in reversed(np.where(del_rxns)[0]):
        model.reactions[i].remove_from_model(remove_orphans=True)

    return reaction_map


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
