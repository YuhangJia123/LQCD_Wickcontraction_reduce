"""Connectivity labels and grouped coefficient results."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Mapping
from typing import TypeVar


Signature = tuple[int, int]
Key = TypeVar("Key")


def classify_output_pairing(pairs, field_owners) -> Signature:
    """Count sink-sink and source-source edges without graph traversal."""
    sink_pairs = 0
    source_pairs = 0
    for left, right in pairs:
        left_side = field_owners[left][0]
        right_side = field_owners[right][0]
        if left_side == right_side == "sink":
            sink_pairs += 1
        elif left_side == right_side == "source":
            source_pairs += 1
    return sink_pairs, source_pairs


def classify_pairing(pairs, field_owners):
    """Return same-side edge counts and the factor-graph topology signature.

    ``field_owners`` contains ``(side, factor_id)`` for each field slot.
    """
    nodes = set(field_owners)
    adjacency = {node: set() for node in nodes}
    for left, right in pairs:
        left_owner = field_owners[left]
        right_owner = field_owners[right]
        adjacency[left_owner].add(right_owner)
        adjacency[right_owner].add(left_owner)

    unseen = set(nodes)
    sink_only = source_only = mixed = 0
    while unseen:
        start = min(unseen)
        component = {start}
        stack = [start]
        unseen.remove(start)
        while stack:
            node = stack.pop()
            for neighbour in adjacency[node]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    component.add(neighbour)
                    stack.append(neighbour)
        sides = {side for side, _factor in component}
        if sides == {"sink"}:
            sink_only += 1
        elif sides == {"source"}:
            source_only += 1
        else:
            mixed += 1

    topology = (
        sink_only + max(mixed - 1, 0),
        source_only + max(mixed - 1, 0),
    )
    return classify_output_pairing(pairs, field_owners), topology


def signature_sort_key(signature: Signature):
    return (signature != (0, 0), signature)


class GroupedWickResult:
    """Coefficient maps grouped before Wick contributions are merged."""

    def __init__(
        self, groups=(), *, tolerance: float = 1.0e-12,
        total_groups=None,
        filter_groups: bool = False,
    ):
        raw_groups = dict(total_groups) if total_groups is not None else dict(groups)
        display_groups = dict(groups)
        if filter_groups:
            display_groups = {
                signature: _ToleranceView(values, tolerance)
                for signature, values in display_groups.items()
            }
        self.groups = OrderedDict(
            (signature, values)
            for signature, values in sorted(
                display_groups.items(), key=lambda item: signature_sort_key(item[0])
            )
        )
        self.tolerance = tolerance
        self._total_groups = raw_groups
        self._total = None

    @property
    def total(self):
        if self._total is None:
            combined = defaultdict(complex)
            for values in self._total_groups.values():
                for key, coefficient in values.items():
                    combined[key] += coefficient
            self._total = {
                key: value for key, value in combined.items()
                if abs(value) > self.tolerance
            }
        return self._total

    def __bool__(self):
        return any(self.groups.values())

    def __len__(self):
        return len(self.groups)

    def __iter__(self):
        return iter(self.groups)

    def __getitem__(self, signature):
        return self.groups[signature]


def flatten_connectivity(values):
    """Return a flat coefficient map for grouped or per-Wick results."""
    if isinstance(values, GroupedWickResult):
        return dict(values.total)
    if not isinstance(values, Mapping):
        raise TypeError("values must be a mapping or GroupedWickResult")
    return dict(values)


class _ToleranceView(Mapping):
    """Read-only nonzero view that avoids copying a group's raw terms."""

    def __init__(self, values, tolerance):
        self._values = values
        self._tolerance = tolerance

    def __getitem__(self, key):
        value = self._values[key]
        if abs(value) <= self._tolerance:
            raise KeyError(key)
        return value

    def __iter__(self):
        return (key for key, value in self._values.items()
                if abs(value) > self._tolerance)

    def __len__(self):
        return sum(abs(value) > self._tolerance for value in self._values.values())

    def __eq__(self, other):
        if not isinstance(other, Mapping):
            return NotImplemented
        return dict(self.items()) == dict(other.items())


__all__ = [
    "GroupedWickResult", "classify_output_pairing", "classify_pairing", "flatten_connectivity",
    "signature_sort_key",
]
