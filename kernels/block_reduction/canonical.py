"""Exact signed canonical forms for independent symbolic tensor networks.

Only dummy-index renaming, commuting factor order, and the finite signed
symmetries declared in ``TensorSpec`` are used.  With ``fixed_output=False``
the open indices can also be relabeled within their typed spaces; the result
records the transpose needed to recover the caller's output order.

The witness convention is always::

    value(input) == result.scalar * value(result.network).transpose(result.permutation)

Color refinement is a candidate partition, never an equivalence proof.  Any
remaining index cells are individualized exhaustively.  Factor-slot orbits
coarsen refinement safely without pretending that every permutation inside
an orbit is an actual tensor symmetry.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from operator import index as integer_index
from typing import Hashable, Mapping, Tuple

from .model import Factor, Network, TensorSpec, compose_transposes


_scoped_cache = ContextVar("block_reduction_canonicalization_cache", default=None)


@contextmanager
def canonicalization_cache(max_entries=16384):
    """Retain exact canonical results for one planning/verification scope.

    The ordinary global LRU remains small. A larger temporary LRU prevents
    normalization, search, and certificate replay from evicting one another's
    results, and is released on exit. Nested contexts restore their caller's
    cache even if an exception escapes.
    """
    if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1:
        raise ValueError("max_entries must be a positive integer")
    token = _scoped_cache.set((OrderedDict(), max_entries))
    try:
        yield
    finally:
        _scoped_cache.reset(token)


@dataclass(frozen=True)
class CanonicalNetwork:
    key: Hashable
    network: Network
    scalar: int
    permutation: Tuple[int, ...]
    # Complete signed output automorphisms of network in free-output mode.
    # The empty default keeps independently constructed legacy witnesses valid.
    output_symmetries: Tuple[Tuple[Tuple[int, ...], int], ...] = ()


@lru_cache(maxsize=256)
def _signed_group(rank, generators):
    """Close signed slot generators, including inconsistent sign witnesses.

    A negative identity is retained.  It proves that the declared tensor is
    zero, rather than being silently lost in a permutation-to-sign mapping.
    Composition uses the same order as successive NumPy transposes.
    """
    identity = tuple(range(rank))
    group = {(identity, 1)}
    pending = [(identity, 1)]
    for permutation, sign in generators:
        if len(permutation) != rank or sorted(permutation) != list(range(rank)):
            raise ValueError("tensor symmetry has an invalid slot permutation")
        if sign not in (-1, 1):
            raise ValueError("tensor symmetry sign must be +1 or -1")
    while pending:
        permutation, sign = pending.pop()
        for generator, generator_sign in generators:
            item = (tuple(permutation[i] for i in generator), sign * generator_sign)
            if item not in group:
                group.add(item)
                pending.append(item)
    return tuple(sorted(group))


@lru_cache(maxsize=256)
def _fingerprint(spec):
    generators = tuple(sorted(
        (tuple(symmetry.permutation), int(symmetry.sign))
        for symmetry in spec.symmetries
    ))
    group = _signed_group(len(spec.shape), generators)
    identity = spec.name if spec.identity is None else spec.identity
    return (
        identity, tuple(spec.shape), tuple(spec.axis_types), spec.kind,
        spec.mode, group,
    )


def _rank_signatures(signatures):
    """Assign deterministic colors to arbitrary hashable signatures."""
    ordered = sorted(set(signatures), key=repr)
    ranks = {signature: index for index, signature in enumerate(ordered)}
    return tuple(ranks[signature] for signature in signatures)


@lru_cache(maxsize=256)
def _slot_orbits(rank, group):
    cells = [set((i,)) for i in range(rank)]
    for permutation, _sign in group:
        for i, j in enumerate(permutation):
            combined = cells[i] | cells[j]
            for member in combined:
                cells[member] = combined
    unique = tuple(sorted({tuple(sorted(cell)) for cell in cells}))
    by_slot = {slot: index for index, cell in enumerate(unique) for slot in cell}
    return unique, tuple(by_slot[i] for i in range(rank))


def canonicalize(
    network: Network,
    specs: Mapping[str, TensorSpec],
    *,
    fixed_output: bool = True,
) -> CanonicalNetwork:
    """Return an exact canonical graph and its signed output-axis witness.

    ``fixed_output=True`` preserves every output position for algebraic sum
    matching.  ``False`` permits compute reuse across output views.  In the
    latter case opposite signs only prove zero when they have the *same*
    output permutation; an antisymmetric open tensor is not a zero tensor.

    Tensor aliases share an identity only when their full declared slot
    structure and symmetry group agree.  A deterministic available alias is
    used in the returned network.  The bounded cache contains no tensor data.
    """
    if not isinstance(network, Network):
        raise TypeError("canonicalize expects a symbolic Network")
    raw = (
        tuple((factor.tensor, tuple(factor.indices)) for factor in network.factors),
        tuple(network.output),
    )
    fingerprints = tuple(sorted(
        ((name, _fingerprint(spec)) for name, spec in specs.items()),
        key=lambda item: item[0],
    ))
    cache_key = (raw, fingerprints, bool(fixed_output))
    scoped = _scoped_cache.get()
    if scoped is None:
        return _canonicalize_cached(*cache_key)
    cache, max_entries = scoped
    if cache_key in cache:
        cache.move_to_end(cache_key)
        return cache[cache_key]
    result = _canonicalize_cached(*cache_key)
    cache[cache_key] = result
    if len(cache) > max_entries:
        cache.popitem(last=False)
    return result


@lru_cache(maxsize=2048)
def _canonicalize_cached(raw, fingerprints, fixed_output):
    raw_factors, original_output = raw
    definitions = dict(fingerprints)
    aliases = {}
    for name, token in fingerprints:
        aliases.setdefault(token, name)

    if len(set(original_output)) != len(original_output):
        raise ValueError("network output indices must be distinct")
    spaces = {}
    factors = []
    occurrences = defaultdict(list)
    tensor_zero = False
    for factor_index, (name, indices) in enumerate(raw_factors):
        if name not in definitions:
            raise ValueError(f"missing TensorSpec for operand {name!r}")
        token = definitions[name]
        _identity, shape, types, _kind, _mode, group = token
        if len(indices) != len(shape):
            raise ValueError(f"operand {name!r} has an incorrect index count")
        orbits, orbit_by_slot = _slot_orbits(len(shape), group)
        if (tuple(range(len(shape))), -1) in group:
            tensor_zero = True
        factors.append((token, indices, orbits, orbit_by_slot))
        for slot, label in enumerate(indices):
            space = (types[slot], shape[slot])
            previous = spaces.setdefault(label, space)
            if previous != space:
                raise ValueError(f"index {label!r} connects incompatible tensor spaces")
            occurrences[label].append((factor_index, orbit_by_slot[slot]))
    if any(label not in spaces for label in original_output):
        raise ValueError("network output includes an index absent from all factors")

    # The names only order traversal.  Exhaustive individualization eliminates
    # their influence on the final canonical key.
    labels = tuple(sorted(spaces))
    label_position = {label: index for index, label in enumerate(labels)}
    output_position = {label: index for index, label in enumerate(original_output)}
    initial = []
    for label in labels:
        if label not in output_position:
            role = ("dummy",)
        elif fixed_output:
            role = ("open", output_position[label])
        else:
            role = ("open",)
        initial.append((role, spaces[label]))
    initial_colors = _rank_signatures(initial)

    def refine(colors):
        while True:
            factor_signatures = tuple(
                (
                    token,
                    tuple(tuple(sorted(colors[label_position[indices[slot]]] for slot in orbit))
                          for orbit in orbits),
                )
                for token, indices, orbits, _orbit_by_slot in factors
            )
            factor_colors = _rank_signatures(factor_signatures)
            signatures = tuple(
                (
                    colors[index],
                    tuple(sorted((factor_colors[factor], orbit)
                                 for factor, orbit in occurrences[label])),
                )
                for index, label in enumerate(labels)
            )
            updated = _rank_signatures(signatures)
            # Old colors occur in every signature, so refinement cannot merge
            # cells.  Equal color counts therefore imply a stable partition.
            if len(set(updated)) == len(set(colors)):
                return updated
            colors = updated

    best_key = None
    best_order = None
    best_network = None
    witnesses = {}
    has_zero_witness = tensor_zero

    def leaf(colors):
        nonlocal best_key, best_order, best_network, witnesses, has_zero_witness
        dummy = sorted((label for label in labels if label not in output_position),
                       key=lambda label: colors[label_position[label]])
        if fixed_output:
            output = original_output
        else:
            output = tuple(sorted(original_output,
                                  key=lambda label: colors[label_position[label]]))
        renaming = {label: f"d{i}" for i, label in enumerate(dummy)}
        renaming.update({label: f"o{i}" for i, label in enumerate(output)})
        permutation = tuple(output.index(label) for label in original_output)
        scalar = 1
        records = []
        local_zero = tensor_zero
        for token, indices, _orbits, _orbit_by_slot in factors:
            mapped = tuple(renaming[label] for label in indices)
            variants = {}
            for slot_permutation, sign in token[-1]:
                variant = tuple(mapped[i] for i in slot_permutation)
                variants.setdefault(variant, set()).add(sign)
            selected = min(variants)
            signs = variants[selected]
            if len(signs) > 1:
                local_zero = True
            scalar *= min(signs)
            records.append((token, selected))
        records.sort(key=lambda item: (repr(item[0]), item[1]))
        canonical_output = tuple(f"o{i}" for i in range(len(output)))
        output_spaces = tuple(spaces[label] for label in output)
        key = (output_spaces, tuple(records))
        order = repr(key)
        candidate = Network(
            tuple(Factor(aliases[token], indices) for token, indices in records),
            canonical_output,
        )
        if best_order is None or order < best_order:
            best_key, best_order, best_network = key, order, candidate
            witnesses = {permutation: {scalar}}
        elif key == best_key:
            witnesses.setdefault(permutation, set()).add(scalar)
        if local_zero:
            has_zero_witness = True

    def visit(colors):
        colors = refine(colors)
        cells = defaultdict(list)
        for index, color in enumerate(colors):
            cells[color].append(index)
        unresolved = [(color, indices) for color, indices in cells.items() if len(indices) > 1]
        if not unresolved:
            leaf(colors)
            return
        # Small cells reduce branching, then deterministic color breaks ties.
        _color, cell = min(unresolved, key=lambda item: (len(item[1]), item[0]))
        for selected in cell:
            signatures = tuple((color, int(index == selected)) for index, color in enumerate(colors))
            visit(_rank_signatures(signatures))

    visit(initial_colors)
    assert best_network is not None
    if any(len(signs) > 1 for signs in witnesses.values()):
        has_zero_witness = True
    permutation = min(witnesses)
    scalar = min(witnesses[permutation])
    inverse = tuple(permutation.index(i) for i in range(len(permutation)))
    output_symmetries = tuple(sorted({
        (compose_transposes(p, inverse), scalar * sign)
        for p, signs in witnesses.items() for sign in signs
    }))
    if has_zero_witness:
        # Shape/type-compatible zero networks are algebraically identical;
        # the concrete canonical network is retained as a valid witness.
        best_key = ("zero", best_key[0])
        scalar = 0
    return CanonicalNetwork(best_key, best_network, scalar, permutation, output_symmetries)


def canonicalize_view(base: CanonicalNetwork, permutation, *, fixed_output=False):
    """Canonicalize a transpose of the input represented by a free witness.

    ``base`` must come from free-output canonicalization. Its complete signed
    output automorphisms replace graph individualization with a small finite
    permutation orbit. The returned witness represents
    ``value(base_input).transpose(permutation)``. Fixed-output keys belong to
    this API's orbit family, not the original ``canonicalize`` fixed keys.
    """
    rank = len(base.network.output)
    try:
        p = tuple(integer_index(i) for i in permutation)
    except TypeError as error:
        raise ValueError("invalid output transpose permutation") from error
    if sorted(p) != list(range(rank)):
        raise ValueError("invalid output transpose permutation")
    viewed = compose_transposes(base.permutation, p)
    identity = tuple(range(rank))
    automorphisms = base.output_symmetries or ((identity, 1),)
    variants = {}
    for q, sign in automorphisms:
        variants.setdefault(compose_transposes(q, viewed), set()).add(sign)
    normalized = min(variants)
    signs = variants[normalized]
    scalar = base.scalar * min(signs) if len(signs) == 1 else 0
    if not fixed_output:
        return CanonicalNetwork(base.key, base.network, scalar, normalized, automorphisms)
    network = Network(base.network.factors, tuple(base.network.output[i] for i in normalized))
    if scalar:
        key = ("fixed_output_orbit", base.key, normalized)
    else:
        spaces = base.key[1] if base.key[0] == "zero" else base.key[0]
        key = ("zero", tuple(spaces[i] for i in normalized))
    inverse = tuple(normalized.index(i) for i in range(rank))
    conjugated = tuple(sorted({
        (compose_transposes(compose_transposes(inverse, q), normalized), sign)
        for q, sign in automorphisms
    }))
    return CanonicalNetwork(key, network, scalar, identity, conjugated)


class OrbitCanonicalizer:
    """Reuse exact free graph forms across many views of a final expression.

    Use one instance consistently for fixed-output collection and matching.
    Its fixed keys differ from legacy fixed keys, while equivalence classes,
    signs, and numerical transpose witnesses agree. All caches are local to
    the instance and contain symbolic graphs only. Both instance caches are
    bounded independently; eviction affects work, never canonical results.
    """

    def __init__(self, specs: Mapping[str, TensorSpec], *, max_bases=2048, max_views=4096):
        for name, limit in (("max_bases", max_bases), ("max_views", max_views)):
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.specs = dict(specs)
        self.max_bases = max_bases
        self.max_views = max_views
        self._bases = OrderedDict()
        self._free_results = OrderedDict()

    @staticmethod
    def _remember(cache, key, value, limit):
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        cache[key] = value
        if len(cache) > limit:
            cache.popitem(last=False)
        return value

    def _free(self, network):
        if not isinstance(network, Network):
            raise TypeError("canonicalize expects a symbolic Network")
        if network in self._free_results:
            self._free_results.move_to_end(network)
            return self._free_results[network]
        seed_output = tuple(sorted(network.output))
        cache_key = (network.factors, seed_output)
        base = self._bases.get(cache_key)
        if base is None:
            base = canonicalize(Network(network.factors, seed_output), self.specs, fixed_output=False)
            self._remember(self._bases, cache_key, base, self.max_bases)
        else:
            self._bases.move_to_end(cache_key)
        p = tuple(seed_output.index(label) for label in network.output)
        result = canonicalize_view(base, p)
        self._remember(self._free_results, network, result, self.max_views)
        canonical = result.network
        self._remember(self._free_results, canonical, CanonicalNetwork(
            result.key, canonical, 0 if not result.scalar else 1,
            tuple(range(len(canonical.output))), result.output_symmetries,
        ), self.max_views)
        return result

    def canonicalize(self, network: Network, *, fixed_output=True):
        free = self._free(network)
        if not fixed_output:
            return free
        fixed = canonicalize_view(free, tuple(range(len(network.output))), fixed_output=True)
        # A fixed representative is a known view of the same free graph.
        # Register its witness now so scoring and proof replay need no graph
        # search when they encounter the freshly collected representative.
        p = tuple(free.network.output.index(label) for label in fixed.network.output)
        self._remember(self._free_results, fixed.network, CanonicalNetwork(
            free.key, free.network, 0 if not fixed.scalar else 1,
            p, free.output_symmetries,
        ), self.max_views)
        return fixed


def clear_cache():
    """Release symbolic canonical forms retained by the bounded caches."""
    _canonicalize_cached.cache_clear()
    _signed_group.cache_clear()
    _fingerprint.cache_clear()
    _slot_orbits.cache_clear()
    scoped = _scoped_cache.get()
    if scoped is not None:
        scoped[0].clear()


__all__ = ["CanonicalNetwork", "OrbitCanonicalizer", "canonicalize", "canonicalize_view",
           "canonicalization_cache", "clear_cache"]
