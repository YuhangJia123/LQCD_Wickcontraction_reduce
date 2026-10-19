"""Optional layout search over an explicitly supplied complete pairing family.

This module knows only symbolic tensor networks.  A caller may construct the
family from Wick data, but no Wick implementation is imported here.  Endpoint
classes and whole-network exchange actions are inferred from the supplied
networks, rather than from operator names or hard-coded slot permutations.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping as MappingABC, Sequence
from dataclasses import dataclass, field, replace
from itertools import combinations
from math import factorial, prod
from types import MappingProxyType
from time import monotonic
from typing import Mapping

from .canonical import OrbitCanonicalizer, canonicalize_view
from .model import (Factor, Network, TensorSpec, WeightedNetwork,
                    compose_transposes, exact_scalar)
from .pairing_count import PairingOrbitCount, count_pairing_layout
from .planner import _block_specs, _normal_only, _output_space


@dataclass(frozen=True)
class PairingFamily:
    """One signed term per bijection of identical endpoint classes.

    Factor positions identify the propagators in a stable sink-slot order.
    Each term may use different source port labels, but all other factors and
    each sink port must agree.  The full target is the sum of ``terms``.
    """

    terms: Sequence[WeightedNetwork]
    factor_positions: tuple[int, ...]


@dataclass(frozen=True)
class CompactPairingTerms(Sequence):
    """A complete alternating endpoint family stored in O(n^2) space.

    ``reference`` assigns source slot ``i`` to sink slot ``i``.  A class is a
    physical exchange domain; equal numerical propagators in different
    classes are not silently exchanged.  ``tensor_by_pair`` optionally names
    the propagator at each allowed sink/source combination.  The coefficient
    of a pairing is the reference coefficient times its parity in each class.
    Other coefficient laws should use an explicit ``PairingFamily``.
    """

    reference: WeightedNetwork
    factor_positions: tuple[int, ...]
    endpoint_classes: tuple[tuple[int, ...], ...]
    tensor_specs: Mapping[str, TensorSpec] = field(repr=False, compare=False)
    tensor_by_pair: tuple[tuple[str | None, ...], ...] = ()

    def __post_init__(self):
        specs = {name: spec if isinstance(spec, TensorSpec) else TensorSpec(name, **spec)
                 for name, spec in self.tensor_specs.items()}
        _normal_only(specs)
        object.__setattr__(self, "tensor_specs", MappingProxyType(specs))
        positions = tuple(self.factor_positions)
        classes = tuple(tuple(sorted(cls)) for cls in self.endpoint_classes)
        n = len(positions)
        if not n or len(set(positions)) != n or any(not 0 <= p < len(self.reference.network.factors)
                                                    for p in positions):
            raise ValueError("factor_positions must identify distinct propagators")
        if any(not cls or len(set(cls)) != len(cls) for cls in classes) or sorted(
            slot for cls in classes for slot in cls
        ) != list(range(n)):
            raise ValueError("endpoint_classes must partition all sink/source slots")
        if len(set(classes)) != len(classes):
            raise ValueError("endpoint_classes must be distinct")
        object.__setattr__(self, "factor_positions", positions)
        object.__setattr__(self, "endpoint_classes", classes)
        _output_space(self.reference.network, specs)
        sinks, sources, identities = [], [], []
        for position in positions:
            factor = self.reference.network.factors[position]
            spec = specs[factor.tensor]
            if spec.kind != "peram":
                raise ValueError("factor_positions must identify propagators")
            sink, source = _ports(factor, spec)
            sinks.append(sink)
            sources.append(source)
            identities.append(_identity(spec))
        if len(set(sinks)) != n or len(set(sources)) != n:
            raise ValueError("pairing endpoint ports must be unique")
        if not self.tensor_by_pair:
            names = tuple(self.reference.network.factors[p].tensor for p in positions)
            by_pair = tuple(tuple(names[i] if any(i in cls and j in cls for cls in classes)
                                  else None for j in range(n)) for i in range(n))
        else:
            by_pair = tuple(tuple(row) for row in self.tensor_by_pair)
            if len(by_pair) != n or any(len(row) != n for row in by_pair):
                raise ValueError("tensor_by_pair must be a square sink/source table")
        for i in range(n):
            for j in range(n):
                allowed = any(i in cls and j in cls for cls in classes)
                name = by_pair[i][j]
                if allowed != (name is not None):
                    raise ValueError("tensor_by_pair must cover exactly the exchange classes")
                if not allowed:
                    continue
                if name not in specs or specs[name].kind != "peram":
                    raise ValueError("tensor_by_pair must reference declared propagators")
                if _identity(specs[name]) != identities[i]:
                    raise ValueError("numerical propagator identity changes within a sink slot")
                if specs[name].peram_ports != specs[self.reference.network.factors[positions[i]].tensor].peram_ports:
                    raise ValueError("propagator port layout changes within a sink slot")
            if by_pair[i][i] != self.reference.network.factors[positions[i]].tensor:
                raise ValueError("reference network must be the identity source assignment")
        object.__setattr__(self, "tensor_by_pair", by_pair)

    def __len__(self):
        return prod(factorial(len(cls)) for cls in self.endpoint_classes)

    def pairings(self, pairs=()):
        """Yield each physical-class-preserving bijection in lexicographic order."""
        slot_class = {slot: cls for cls in self.endpoint_classes for slot in cls}
        available = {cls: set(cls) for cls in self.endpoint_classes}
        pairing = [0] * len(self.factor_positions)
        lower_partner = {right: left for left, right in pairs}

        def visit(slot):
            if slot == len(pairing):
                yield tuple(pairing)
                return
            remaining = available[slot_class[slot]]
            for source in sorted(remaining):
                if slot in lower_partner and source < pairing[lower_partner[slot]]:
                    continue
                remaining.remove(source)
                pairing[slot] = source
                yield from visit(slot + 1)
                remaining.add(source)

        yield from visit(0)

    def contains(self, pairing):
        if len(pairing) != len(self.factor_positions):
            return False
        return all(sorted(pairing[i] for i in cls) == list(cls)
                   for cls in self.endpoint_classes)

    def coefficient(self, pairing):
        if not self.contains(pairing):
            raise ValueError("pairing leaves its physical exchange class")
        parity = 1
        for cls in self.endpoint_classes:
            parity *= -1 if sum(pairing[i] > pairing[j]
                                for offset, i in enumerate(cls)
                                for j in cls[offset + 1:]) % 2 else 1
        return exact_scalar(self.reference.coefficient) * parity

    def term(self, pairing):
        coefficient = self.coefficient(pairing)
        factors = list(self.reference.network.factors)
        source_ports = tuple(_ports(factors[position],
                                    self.tensor_specs[factors[position].tensor])[1]
                             for position in self.factor_positions)
        for sink, source in enumerate(pairing):
            position = self.factor_positions[sink]
            old = factors[position]
            name = self.tensor_by_pair[sink][source]
            indices = list(old.indices)
            for axis, label in zip(self.tensor_specs[name].peram_ports[1], source_ports[source]):
                indices[axis] = label
            factors[position] = Factor(name, tuple(indices))
        return WeightedNetwork(Network(tuple(factors), self.reference.network.output), coefficient)

    def representatives(self, pairs):
        """Yield one lexicographic representative per local W exchange orbit."""
        yield from self.pairings(pairs)

    def __iter__(self):
        for pairing in self.pairings():
            yield self.term(pairing)

    def pairing_at(self, index):
        if isinstance(index, slice):
            return tuple(self.pairing_at(i) for i in range(*index.indices(len(self))))
        if not isinstance(index, int):
            raise TypeError("pairing term index must be an integer")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        slot_class = {slot: cls for cls in self.endpoint_classes for slot in cls}
        available = {cls: list(cls) for cls in self.endpoint_classes}
        pairing = []
        for slot in range(len(self.factor_positions)):
            cls = slot_class[slot]
            width = prod(factorial(len(values) - (key == cls))
                         for key, values in available.items())
            choice, index = divmod(index, width)
            pairing.append(available[cls].pop(choice))
        return tuple(pairing)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        return self.term(self.pairing_at(index))


class _PairingKeys(Sequence):
    def __init__(self, terms):
        self.terms = terms

    def __len__(self):
        return len(self.terms)

    def __iter__(self):
        return self.terms.pairings()

    def __getitem__(self, index):
        return self.terms.pairing_at(index)


class _CompactPairings(MappingABC):
    def __init__(self, terms: CompactPairingTerms):
        self.terms = terms

    def __len__(self):
        return len(self.terms)

    def __iter__(self):
        return self.terms.pairings()

    def __getitem__(self, pairing):
        if not self.terms.contains(pairing):
            raise KeyError(pairing)
        return self.terms.term(pairing)

    def __contains__(self, pairing):
        return self.terms.contains(pairing)


class _ExchangeOwners:
    def __init__(self, pairs):
        self.pairs = pairs

    def __getitem__(self, pairing):
        result = list(pairing)
        for left, right in self.pairs:
            if result[left] > result[right]:
                result[left], result[right] = result[right], result[left]
        return tuple(result)


@dataclass(frozen=True)
class LayoutResult:
    pairs: tuple[tuple[int, int], ...]
    network_count: int
    fused_terms: tuple[WeightedNetwork, ...]
    count: PairingOrbitCount | None


@dataclass(frozen=True)
class EndpointWitness:
    sink_permutation: tuple[int, ...]
    source_permutation: tuple[int, ...]
    output_permutation: tuple[int, ...]
    sign: int


@dataclass(frozen=True)
class GlobalLayoutResult:
    layouts: tuple[LayoutResult, ...]
    sink_types: tuple[int, ...]
    source_types: tuple[int, ...]
    endpoint_generators: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    endpoint_witnesses: tuple[EndpointWitness, ...]
    pairings: Sequence[tuple[int, ...]]
    tested_layouts: int
    complete: bool
    elapsed_seconds: float


def _identity(spec: TensorSpec) -> str:
    return spec.name if spec.identity is None else spec.identity


def _ports(factor: Factor, spec: TensorSpec):
    return tuple(tuple(factor.indices[i] for i in port)
                 for port in spec.peram_ports)


def _prepare(family: PairingFamily, specs: Mapping[str, TensorSpec]):
    if not family.terms:
        raise ValueError("a pairing family must contain at least one term")
    positions = family.factor_positions
    if not positions or len(set(positions)) != len(positions):
        raise ValueError("factor_positions must identify distinct propagators")
    if isinstance(family.terms, CompactPairingTerms):
        terms = family.terms
        used = {factor.tensor for factor in terms.reference.network.factors}
        used.update(name for row in terms.tensor_by_pair for name in row if name is not None)
        if positions != terms.factor_positions or any(
            specs.get(name) != terms.tensor_specs[name] for name in used
        ):
            raise ValueError("compact pairing metadata must match the family and tensor specs")
        first = terms.reference.network
        sinks, sources, identities = [], [], []
        for position in positions:
            factor = first.factors[position]
            sink, source = _ports(factor, specs[factor.tensor])
            sinks.append(sink)
            sources.append(source)
            identities.append(_identity(specs[factor.tensor]))
        sink_types = [0] * len(positions)
        for kind, cls in enumerate(terms.endpoint_classes):
            for slot in cls:
                sink_types[slot] = kind
        sink_types = tuple(sink_types)
        allowed = tuple((i, j) for i, j in combinations(range(len(positions)), 2)
                        if sink_types[i] == sink_types[j] and identities[i] == identities[j])
        return (_CompactPairings(terms), sink_types, sink_types, allowed,
                tuple(sinks), tuple(sources), tuple(identities))
    first = family.terms[0].network
    if any(not 0 <= position < len(first.factors) for position in positions):
        raise ValueError("factor_positions exceed the network")
    sources = []
    sinks = []
    identities = []
    for position in positions:
        factor = first.factors[position]
        spec = specs[factor.tensor]
        if spec.kind != "peram":
            raise ValueError("factor_positions must identify propagators")
        sink, source = _ports(factor, spec)
        sinks.append(sink)
        sources.append(source)
        identities.append(_identity(spec))
    if len(set(sources)) != len(sources) or len(set(sinks)) != len(sinks):
        raise ValueError("pairing endpoint ports must be unique")
    source_number = {source: i for i, source in enumerate(sources)}
    spectator = tuple(f for i, f in enumerate(first.factors) if i not in positions)
    pairings = {}
    for term in family.terms:
        network = term.network
        if len(network.factors) != len(first.factors) or network.output != first.output:
            raise ValueError("pairing family networks must share factor positions and output")
        if tuple(f for i, f in enumerate(network.factors) if i not in positions) != spectator:
            raise ValueError("pairing family spectators must be identical")
        pairing = []
        for slot, position in enumerate(positions):
            factor = network.factors[position]
            spec = specs[factor.tensor]
            if spec.kind != "peram" or _identity(spec) != identities[slot]:
                raise ValueError("the numerical propagator type varies within a sink slot")
            sink, source = _ports(factor, spec)
            if sink != sinks[slot] or source not in source_number:
                raise ValueError("a pairing changes a sink port or introduces a source port")
            pairing.append(source_number[source])
        pairing = tuple(pairing)
        if sorted(pairing) != list(range(len(positions))) or pairing in pairings:
            raise ValueError("a pairing must be a unique endpoint bijection")
        pairings[pairing] = term
    supports = tuple(frozenset(p[i] for p in pairings) for i in range(len(positions)))
    classes = {support: i for i, support in enumerate(sorted(set(supports), key=lambda x: (len(x), tuple(sorted(x)))))}
    sink_types = tuple(classes[support] for support in supports)
    source_types = tuple(next((classes[support] for support in supports if j in support), -1)
                         for j in range(len(positions)))
    if -1 in source_types or any(
        source_types[p[i]] != sink_types[i] for p in pairings for i in range(len(positions))
    ):
        raise ValueError("source supports are not disjoint endpoint classes")
    allowed = tuple((i, j) for i, j in combinations(range(len(positions)), 2)
                    if sink_types[i] == sink_types[j] and identities[i] == identities[j]
                    and all(tuple(p[j] if k == i else p[i] if k == j else p[k]
                                  for k in range(len(p))) in pairings
                            and pairings[tuple(p[j] if k == i else p[i] if k == j else p[k]
                                               for k in range(len(p)))].coefficient == -term.coefficient
                            for p, term in pairings.items()))
    return (pairings, sink_types, source_types, allowed, tuple(sinks),
            tuple(sources), tuple(identities))


def _endpoint_actions(family: PairingFamily, specs: Mapping[str, TensorSpec],
                      sinks, sources, sink_types, source_types, identities,
                      pairings):
    """Recover signed whole-network endpoint actions from a marked graph."""
    first = (family.terms.reference.network if isinstance(family.terms, CompactPairingTerms)
             else family.terms[0].network)
    n = len(sinks)
    positions = set(family.factor_positions)
    spectators = tuple(f for i, f in enumerate(first.factors) if i not in positions)
    marker_specs = dict(specs)
    taken_labels = {label for factor in first.factors for label in factor.indices}
    taken_labels.update(first.output)

    def unused_label(prefix):
        candidate = prefix
        suffix = 0
        while candidate in taken_labels:
            suffix += 1
            candidate = f"{prefix}_{suffix}"
        taken_labels.add(candidate)
        return candidate

    sink_tags = tuple(unused_label(f"__global_sink_tag_{i}") for i in range(n))
    source_tags = tuple(unused_label(f"__global_source_tag_{i}") for i in range(n))
    markers = []
    source_identity = [None] * n
    first_pairing = next(iter(pairings))
    for sink_slot, source_slot in enumerate(first_pairing):
        source_identity[source_slot] = identities[sink_slot]
    marker_colors = {}
    for side, ports, tags, classes, numerical in (
        ("sink", sinks, sink_tags, sink_types, identities),
        ("source", sources, source_tags, source_types, source_identity),
    ):
        for port, tag, kind, numerical_identity in zip(ports, tags, classes, numerical):
            color = (side, kind, numerical_identity)
            if color not in marker_colors:
                marker_colors[color] = len(marker_colors)
            base = f"__global_{side}_marker_{marker_colors[color]}"
            name = base
            suffix = 0
            while name in specs:
                suffix += 1
                name = f"{base}_{suffix}"
            marker_specs[name] = TensorSpec(name, (3, 4, n),
                                            ("color", "spin", side + "_endpoint"))
            markers.append(Factor(name, port + (tag,)))
    graph = Network(spectators + tuple(markers), first.output + sink_tags + source_tags)
    canonical = OrbitCanonicalizer(marker_specs).canonicalize(graph, fixed_output=False)
    rank = len(first.output)
    witnesses = set()
    identity = tuple(range(n))
    # output_symmetries are expressed in the canonical output order.  The
    # witness permutation maps the caller's order to that order; conjugate
    # before interpreting marker positions as endpoint numbers.
    witness = canonical.permutation
    for canonical_permutation, spectator_sign in canonical.output_symmetries:
        permutation = tuple(witness.index(canonical_permutation[witness[i]])
                            for i in range(len(witness)))
        inverse = tuple(permutation.index(i) for i in range(len(permutation)))
        if any(inverse[rank + i] < rank or inverse[rank + i] >= rank + n
               for i in range(n)):
            continue
        if any(inverse[rank + n + i] < rank + n or inverse[rank + n + i] >= rank + 2 * n
               for i in range(n)):
            continue
        sink_action = tuple(inverse[rank + i] - rank for i in range(n))
        source_action = tuple(inverse[rank + n + i] - rank - n for i in range(n))
        if not all(sink_types[i] == sink_types[sink_action[i]] and
                   identities[i] == identities[sink_action[i]] for i in range(n)):
            continue
        if not all(source_types[i] == source_types[source_action[i]] and
                   source_identity[i] == source_identity[source_action[i]] for i in range(n)):
            continue
        if (sink_action, source_action) == (identity, identity):
            continue
        output_permutation = permutation[:rank]
        if sorted(output_permutation) != list(range(rank)):
            continue
        if isinstance(pairings, _CompactPairings):
            # The alternating character makes this coefficient ratio exact
            # for every class-preserving bijection, without scanning them.
            ratio = 1
            for cls in family.terms.endpoint_classes:
                ratio *= (-1 if sum(sink_action[i] > sink_action[j]
                                    for offset, i in enumerate(cls)
                                    for j in cls[offset + 1:]) % 2 else 1)
                ratio *= (-1 if sum(source_action[i] > source_action[j]
                                    for offset, i in enumerate(cls)
                                    for j in cls[offset + 1:]) % 2 else 1)
        else:
            ratio = None
            for pairing, term in pairings.items():
                image = [None] * n
                for old_sink, old_source in enumerate(pairing):
                    image[sink_action[old_sink]] = source_action[old_source]
                partner = pairings.get(tuple(image))
                if partner is None or not term.coefficient or not partner.coefficient:
                    ratio = None
                    break
                current = (1 if exact_scalar(partner.coefficient) == exact_scalar(term.coefficient)
                           else -1 if exact_scalar(partner.coefficient) == -exact_scalar(term.coefficient)
                           else None)
                if current not in (-1, 1) or ratio is not None and current != ratio:
                    ratio = None
                    break
                ratio = current
        if ratio is not None:
            witnesses.add(EndpointWitness(sink_action, source_action,
                                          output_permutation, int(ratio * spectator_sign)))
    return tuple(sorted(witnesses, key=repr))


def _matchings(edges, limit=None):
    """Generate maximal disjoint edge sets; equivalent orders never recur."""
    adjacency = defaultdict(set)
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    vertices = frozenset(adjacency)
    best = 0
    found = set()

    def visit(remaining, pairs):
        nonlocal best
        if limit is not None and len(pairs) >= limit:
            found.add(tuple(sorted(pairs)))
            best = max(best, len(pairs))
            return
        if not remaining:
            if len(pairs) > best:
                found.clear()
                best = len(pairs)
            if len(pairs) == best:
                found.add(tuple(sorted(pairs)))
            return
        if len(pairs) + len(remaining) // 2 < best:
            return
        pivot = min(remaining)
        options = sorted(adjacency[pivot] & remaining)
        for other in options:
            visit(remaining - {pivot, other}, pairs + ((pivot, other),))
        visit(remaining - {pivot}, pairs)

    visit(vertices, ())
    return tuple(sorted(found))


def _all_matchings(edges, deadline):
    """Enumerate every nonempty matching lazily, larger choices first."""
    adjacency = defaultdict(set)
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)

    def visit(remaining, pairs):
        if monotonic() >= deadline:
            return
        if not remaining:
            if pairs:
                yield tuple(sorted(pairs))
            return
        pivot = min(remaining)
        for other in sorted(adjacency[pivot] & remaining):
            yield from visit(remaining - {pivot, other}, pairs + ((pivot, other),))
        yield from visit(remaining - {pivot}, pairs)

    yield from visit(frozenset(adjacency), ())


def _invariant_matchings(edges, actions, deadline):
    """Augment disjoint edge orbits instead of listing all layouts first."""
    edges = frozenset(tuple(sorted(edge)) for edge in edges)
    n = len(actions[0][0]) if actions else max((max(edge) for edge in edges), default=-1) + 1
    identity = tuple(range(n))
    group = {identity}
    pending = [identity]
    generators = {tuple(action[0][:n]) for action in actions}
    while pending:
        permutation = pending.pop()
        for generator in generators:
            image = tuple(generator[i] for i in permutation)
            if image not in group:
                group.add(image)
                pending.append(image)
    orbits = set()
    for edge in edges:
        orbit = frozenset(tuple(sorted((p[edge[0]], p[edge[1]]))) for p in group)
        flat = [slot for pair in orbit for slot in pair]
        if orbit <= edges and len(flat) == len(set(flat)):
            orbits.add(tuple(sorted(orbit)))
    orbits = tuple(sorted(orbits, key=lambda orbit: (-len(orbit), orbit)))
    best = 0
    found = set()

    def visit(index, used, pairs):
        nonlocal best
        if monotonic() >= deadline:
            return
        if index == len(orbits):
            if len(pairs) > best:
                found.clear()
                best = len(pairs)
            if len(pairs) == best:
                found.add(tuple(sorted(pairs)))
            return
        if len(pairs) + (n - len(used)) // 2 < best:
            return
        orbit = orbits[index]
        slots = frozenset(slot for pair in orbit for slot in pair)
        if not used.intersection(slots):
            visit(index + 1, used | slots, pairs + orbit)
        visit(index + 1, used, pairs)

    visit(0, frozenset(), ())
    return tuple(sorted(found))


def _orbit(pairing, pairs):
    result = []
    for mask in range(1 << len(pairs)):
        image = list(pairing)
        for bit, (left, right) in enumerate(pairs):
            if mask & (1 << bit):
                image[left], image[right] = image[right], image[left]
        result.append(tuple(image))
    return tuple(result)


def _fused(network, pairs, positions, specs, definitions):
    removed = set()
    blocks = []
    for left, right in pairs:
        i, j = positions[left], positions[right]
        first, second = network.factors[i], network.factors[j]
        a, source_i = _ports(first, specs[first.tensor])
        b, source_j = _ports(second, specs[second.tensor])
        definition = definitions[first.tensor]
        blocks.append(Factor(definition.name, a + b + source_i + source_j))
        removed.update((i, j))
    return Network(tuple(f for i, f in enumerate(network.factors) if i not in removed)
                   + tuple(blocks), network.output)


def _score_fused(fused_by_pairing, owner, pairs, witnesses, canonicalizer,
                 deadline):
    """Score signed fixed-axis terms using proven endpoint transpose actions.

    An endpoint action stabilizing the layout maps a complete W orbit to
    another complete W orbit. Its coefficient sign and open-axis transpose
    were checked against every supplied raw pairing in _endpoint_actions.
    Thus only one graph in each endpoint orbit needs free canonicalization;
    the other fixed-axis terms are exact signed views of that graph.
    """
    pair_set = frozenset(pairs)
    actions = tuple(w for w in witnesses if frozenset(
        tuple(sorted((w.sink_permutation[a], w.sink_permutation[b])))
        for a, b in pairs) == pair_set)
    rank = len(next(iter(fused_by_pairing.values())).network.output)
    identity = tuple(range(rank))
    unseen = set(fused_by_pairing)
    ledger = defaultdict(lambda: 0)
    free_keys = {}
    while unseen:
        if monotonic() >= deadline:
            return None
        base_pairing = min(unseen)
        base = fused_by_pairing[base_pairing]
        base_free = canonicalizer.canonicalize(base.network, fixed_output=False)
        # Store one path from base to each orbit member. Other paths encode
        # automorphisms of the same tensor and are already in base_free.
        views = {base_pairing: (1, identity)}
        queue = [base_pairing]
        for current in queue:
            sign, permutation = views[current]
            for action in actions:
                image = [None] * len(current)
                for sink, source in enumerate(current):
                    image[action.sink_permutation[sink]] = action.source_permutation[source]
                target = owner[tuple(image)]
                if target not in views:
                    views[target] = (sign * action.sign,
                                     compose_transposes(permutation,
                                                        action.output_permutation))
                    queue.append(target)
        unseen.difference_update(views)
        for sign, permutation in views.values():
            fixed = canonicalize_view(base_free, permutation, fixed_output=True)
            if fixed.scalar:
                ledger[fixed.key] += exact_scalar(base.coefficient) * sign * fixed.scalar
                free_keys[fixed.key] = base_free.key
    return len({free_keys[key] for key, coefficient in ledger.items() if coefficient})



def _lower_bound_fused(fused_by_pairing, owner, pairs, witnesses,
                       canonicalizer, deadline):
    """Return an admissible lower bound for one complete fused layout.

    Terms are first grouped by their exact free-output graph key, while all
    signed endpoint/output views in that free graph are accumulated together.
    A nonzero group coefficient guarantees that at least one fixed-output
    network survives.  Distinct free graph keys cannot become equivalent under
    the declared tensor symmetries, so the number of nonzero groups is a safe
    lower bound on ``_score_fused``.  This avoids fixed-output canonicalization
    for layouts that cannot beat the current best score.
    """
    pair_set = frozenset(pairs)
    actions = tuple(w for w in witnesses if frozenset(
        tuple(sorted((w.sink_permutation[a], w.sink_permutation[b])))
        for a, b in pairs) == pair_set)
    rank = len(next(iter(fused_by_pairing.values())).network.output)
    identity = tuple(range(rank))
    unseen = set(fused_by_pairing)
    weights = defaultdict(lambda: 0)
    while unseen:
        if monotonic() >= deadline:
            return None
        base_pairing = min(unseen)
        base = fused_by_pairing[base_pairing]
        base_free = canonicalizer.canonicalize(base.network, fixed_output=False)
        views = {base_pairing: (1, identity)}
        queue = [base_pairing]
        for current in queue:
            sign, permutation = views[current]
            for action in actions:
                image = [None] * len(current)
                for sink, source in enumerate(current):
                    image[action.sink_permutation[sink]] = action.source_permutation[source]
                target = owner[tuple(image)]
                if target not in views:
                    views[target] = (sign * action.sign,
                                     compose_transposes(permutation,
                                                        action.output_permutation))
                    queue.append(target)
        unseen.difference_update(views)
        for sign, permutation in views.values():
            fixed = canonicalize_view(base_free, permutation, fixed_output=True)
            if fixed.scalar:
                weights[base_free.key] += (
                    exact_scalar(base.coefficient) * sign * fixed.scalar
                )
    return sum(bool(value) for value in weights.values())


def _collect_compact_family(family: PairingFamily,
                            specs: Mapping[str, TensorSpec], *,
                            witnesses=None, canonicalizer=None):
    """Collect a compact raw family with one graph search per endpoint orbit.

    Every endpoint action is a signed whole-network identity established by
    ``_endpoint_actions``.  Its output transpose is applied *before* a term is
    inserted into the fixed-open-axis ledger.  Only orbit representatives are
    materialized as ``WeightedNetwork`` objects; the remaining pairings are
    represented by their signed transpose views.

    The returned keys have the same form as ``OrbitCanonicalizer`` fixed-output
    keys.  Pass the same canonicalizer to an independently collected target
    when comparing ledgers.
    """
    terms = family.terms
    if not isinstance(terms, CompactPairingTerms):
        raise TypeError("compact collection requires CompactPairingTerms")
    pairings, sink_types, source_types, _edges, sinks, sources, identities = _prepare(
        family, specs)
    if witnesses is None:
        witnesses = _endpoint_actions(family, specs, sinks, sources,
                                      sink_types, source_types, identities, pairings)
    if canonicalizer is None:
        canonicalizer = OrbitCanonicalizer(specs)
    rank = len(terms.reference.network.output)
    identity = tuple(range(rank))
    seen = set()
    ledger = {}
    for pairing in terms.pairings():
        if pairing in seen:
            continue
        base = terms.term(pairing)
        free = canonicalizer.canonicalize(base.network, fixed_output=False)
        views = {pairing: (1, identity)}
        queue = [pairing]
        seen.add(pairing)
        for current in queue:
            sign, permutation = views[current]
            for action in witnesses:
                image = [None] * len(current)
                for sink, source in enumerate(current):
                    image[action.sink_permutation[sink]] = action.source_permutation[source]
                target = tuple(image)
                if target not in views:
                    views[target] = (sign * action.sign,
                                     compose_transposes(permutation,
                                                        action.output_permutation))
                    queue.append(target)
                    seen.add(target)
        for sign, permutation in views.values():
            fixed = canonicalize_view(free, permutation, fixed_output=True)
            coefficient = exact_scalar(base.coefficient) * sign * fixed.scalar
            if not coefficient:
                continue
            old = ledger.get(fixed.key)
            coefficient += old.coefficient if old else 0
            if coefficient:
                ledger[fixed.key] = WeightedNetwork(fixed.network, coefficient)
            else:
                ledger.pop(fixed.key, None)
    return ledger


def find_pairing_layouts(
    family: PairingFamily, tensor_specs: Mapping[str, TensorSpec], *,
    max_w_types: int = 1, search_seconds: float = 5.0,
    top_k: int = 2,
) -> GlobalLayoutResult:
    """Find lowest-count full-W layouts, using only supplied symbolic networks.

    The caller must separately prove that this optional full pairing family
    expands to its three-table target.  Layout counts are actual canonical
    full-network counts, including every unfused residual term.

    """
    started = monotonic()
    deadline = started + search_seconds
    specs = {name: value if isinstance(value, TensorSpec) else TensorSpec(name, **value)
             for name, value in tensor_specs.items()}
    _normal_only(specs)
    if max_w_types < 1 or top_k < 1 or search_seconds < 0:
        raise ValueError("max_w_types and top_k must be positive, search_seconds nonnegative")
    pairings, sink_types, source_types, edges, sinks, sources, identities = _prepare(family, specs)
    compact = isinstance(family.terms, CompactPairingTerms)
    first = family.terms.reference.network if compact else family.terms[0].network
    witnesses = (() if monotonic() >= deadline else
                 _endpoint_actions(family, specs, sinks, sources, sink_types,
                                   source_types, identities, pairings))
    generators = tuple(sorted({(item.sink_permutation, item.source_permutation)
                               for item in witnesses}))
    # First enumerate edge-orbit augmentations preserving all discovered
    # sink actions.  The fallback is unrestricted, so this preference does
    # not impose preservation of every symmetry on the final answer.
    available_types = tuple(sorted({identities[a] for a, _b in edges}))
    def edge_families():
        if max_w_types == 1:
            for identity in available_types:
                if monotonic() >= deadline:
                    return
                yield tuple(edge for edge in edges if identities[edge[0]] == identity)
            return
        # The generic path may have many type subsets. Generate them on
        # demand so its budget also bounds candidate-family preparation.
        for count in range(1, min(max_w_types, len(available_types)) + 1):
            for selected in combinations(available_types, count):
                if monotonic() >= deadline:
                    return
                selected = frozenset(selected)
                yield tuple(edge for edge in edges if identities[edge[0]] in selected)

    layouts = tuple(sorted({layout for subedges in edge_families()
                            for layout in _invariant_matchings(subedges, generators, deadline)}))
    expanded_specs, definitions, by_prop = _block_specs(specs)
    raw_specs = OrbitCanonicalizer(expanded_specs)
    ranked = []
    tested = 0
    ordered = []
    for pairs in layouts:
        if len({by_prop[first.factors[family.factor_positions[a]].tensor].name
                for a, _b in pairs}) > max_w_types:
            continue
        count = None
        try:
            count = count_pairing_layout(sink_types, source_types, pairs, generators,
                                         pairings=None if compact else pairings.keys())
            if compact:
                count = replace(count, full_family_verified=True)
        except ValueError:
            pass
        ordered.append((pairs, count))
    # Burnside is a cheap ranking hint. Further graph symmetries and signed
    # cancellations can reduce actual calls, so it is not a pruning bound.
    ordered.sort(key=lambda item: (float("inf") if item[1] is None else item[1].orbit_count,
                                   item[0]))
    scheduled = set(layouts)
    generated = set(scheduled)
    pending_layouts = iter(ordered)
    fallback = False
    complete = True
    while True:
        try:
            pairs, count = next(pending_layouts)
        except StopIteration:
            if fallback or monotonic() >= deadline:
                if monotonic() >= deadline:
                    complete = False
                break
            fallback = True
            def remaining_layouts():
                for subedges in edge_families():
                    for layout in _all_matchings(subedges, deadline):
                        if layout in generated:
                            continue
                        generated.add(layout)
                        yield layout, None
            pending_layouts = remaining_layouts()
            try:
                pairs, count = next(pending_layouts)
            except StopIteration:
                if monotonic() >= deadline:
                    complete = False
                break
        if monotonic() >= deadline:
            complete = False
            break
        if len({by_prop[first.factors[family.factor_positions[a]].tensor].name
                for a, _b in pairs}) > max_w_types:
            continue
        fused = []
        fused_by_pairing = {}
        if compact:
            owner = _ExchangeOwners(pairs)
            representatives = family.terms.representatives(pairs)
        else:
            owner = {}
            representatives = sorted(pairings)
            seen = set()
        for pairing in representatives:
            if monotonic() >= deadline:
                complete = False
                break
            if not compact:
                if pairing in seen:
                    continue
                exchange_orbit = _orbit(pairing, pairs)
                if any(image not in pairings for image in exchange_orbit):
                    raise ValueError("layout exchange leaves the supplied pairing family")
                if any(pairings[image].coefficient != pairings[pairing].coefficient * (-1)**bin(mask).count("1")
                       for mask, image in enumerate(exchange_orbit)):
                    raise ValueError("layout exchange coefficients violate W antisymmetry")
                seen.update(exchange_orbit)
                owner.update((image, pairing) for image in exchange_orbit)
            term = pairings[pairing]
            fused_term = WeightedNetwork(_fused(term.network, pairs, family.factor_positions,
                                                expanded_specs, by_prop), term.coefficient)
            fused.append(fused_term)
            fused_by_pairing[pairing] = fused_term
        if not complete:
            break
        if not compact and len(seen) != len(pairings):
            raise AssertionError("layout orbit partition is incomplete")
        lower_bound = _lower_bound_fused(
            fused_by_pairing, owner, pairs, witnesses, raw_specs, deadline)
        if lower_bound is None:
            complete = False
            break
        if ranked:
            best_count = ranked[0].network_count
            # The bound is safe even when different fixed-output views cancel:
            # a nonzero free-graph aggregate implies at least one surviving
            # fixed-output key.  Equal-count layouts are retained when callers
            # request ties, while top_k=1 can stop evaluating them.
            if lower_bound > best_count or (lower_bound == best_count and top_k == 1):
                continue
        network_count = _score_fused(fused_by_pairing, owner, pairs, witnesses,
                                     raw_specs, deadline)
        if network_count is None:
            complete = False
            break
        actual = LayoutResult(pairs, network_count, tuple(fused), count)
        ranked.append(actual)
        ranked.sort(key=lambda item: (item.network_count, item.pairs))
        ranked = ranked[:top_k]
        tested += 1
        # Zero is an absolute lower bound for a weighted network sum. This
        # generic terminal case needs no workload-specific target or oracle.
        if network_count == 0:
            complete = True
            break
    return GlobalLayoutResult(tuple(ranked), sink_types, source_types, generators,
                              witnesses, _PairingKeys(family.terms) if compact
                              else tuple(sorted(pairings)), tested, complete,
                              monotonic() - started)


__all__ = ["PairingFamily", "CompactPairingTerms", "LayoutResult", "EndpointWitness", "GlobalLayoutResult",
           "find_pairing_layouts"]
