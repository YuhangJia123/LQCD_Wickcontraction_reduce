"""Strict local tensor permutation checks for factorized vertices."""

from __future__ import annotations

from collections import defaultdict

from .model import LocalVertex, parity


def _spin_permutation_phase(vertex: LocalVertex, slot_map, tolerance=1e-12):
    original = defaultdict(complex)
    for spins, coefficient in vertex.spin_terms:
        original[tuple(spins)] += complex(coefficient)
    original = {key: value for key, value in original.items() if value}
    transported = defaultdict(complex)
    for spins, coefficient in original.items():
        mapped = [None] * len(spins)
        for old_slot, new_slot in enumerate(slot_map):
            mapped[new_slot] = spins[old_slot]
        transported[tuple(mapped)] += coefficient
    keys = set(original) | set(transported)
    pivot = next((key for key in keys
                  if abs(original.get(key, 0j)) > tolerance
                  and abs(transported.get(key, 0j)) > tolerance), None)
    if pivot is None:
        return None
    phase = original[pivot] / transported[pivot]
    if abs(abs(phase) - 1) > tolerance:
        return None
    phase /= abs(phase)
    scale = max((abs(value) for value in original.values()), default=1.0)
    if any(abs(original.get(key, 0j) - phase * transported.get(key, 0j))
           > tolerance * scale for key in keys):
        return None
    return phase



def vertex_slot_permutation_phase(vertex: LocalVertex, slot_map):
    """Return a validated colour/Dirac phase for a local field permutation."""
    if (len(slot_map) != len(vertex.fields)
            or any(vertex.fields[old] != vertex.fields[new]
                   for old, new in enumerate(slot_map))):
        return None
    spin_phase = _spin_permutation_phase(vertex, slot_map)
    if spin_phase is None:
        return None
    colour_phase = parity(slot_map) if vertex.colour_kind == "epsilon" else 1
    return colour_phase * spin_phase


def identical_vertices(left: LocalVertex, right: LocalVertex) -> bool:
    """Check a complete local delta/epsilon and Dirac factor."""
    return (left.fields == right.fields
            and left.colour_kind == right.colour_kind
            and left.spin_terms == right.spin_terms
            and complex(left.phase) == complex(right.phase))
