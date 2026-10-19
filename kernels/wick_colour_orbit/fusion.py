"""Shared incidence, spin and global-colour S3 fusion kernel.

This module knows coordinate axes and delta/epsilon tensors only. Wick
scope, point routes, input adapters and output serialization live outside it.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from itertools import permutations, product
from typing import Dict, Tuple

from .model import parity

COLOUR_BASE = 3
SPIN_BASE = 4
ColourFactor = Tuple[str, Tuple[int, ...]]
Incidence = Tuple[ColourFactor, ...]
_EPSILON_TERMS = tuple((p, parity(p)) for p in permutations(range(COLOUR_BASE)))
_COLOUR_PERMUTATIONS = tuple(permutations(range(COLOUR_BASE)))


def encode_digits(values, base: int) -> int:
    result = 0
    multiplier = 1
    for value in values:
        result += int(value) * multiplier
        multiplier *= base
    return result


def decode_digits(code: int, length: int, base: int) -> Tuple[int, ...]:
    digits = []
    for _ in range(length):
        digits.append(code % base)
        code //= base
    return tuple(digits)


def _find(parent: list[int], item: int) -> int:
    while parent[item] != item:
        parent[item] = parent[parent[item]]
        item = parent[item]
    return item


def _union(parent: list[int], left: int, right: int) -> None:
    left_root = _find(parent, left)
    right_root = _find(parent, right)
    if left_root != right_root:
        parent[right_root] = left_root


@lru_cache(maxsize=256)
def _incidence_colour_polynomial(
    incidence: Incidence,
    line_count: int,
) -> Tuple[Tuple[int, int], ...]:
    """Expand typed colour incidence, contracting delta equal-colour edges."""

    parent = list(range(line_count))
    for kind, axes in incidence:
        if kind == "delta":
            _union(parent, axes[0], axes[1])

    epsilon_roots = []
    for kind, axes in incidence:
        if kind != "epsilon":
            continue
        roots = tuple(_find(parent, axis) for axis in axes)
        if len(set(roots)) != 3:
            return ()
        epsilon_roots.append(roots)

    all_roots = tuple(sorted({_find(parent, axis) for axis in range(line_count)}))
    epsilon_covered = {root for triple in epsilon_roots for root in triple}
    free_roots = tuple(root for root in all_roots if root not in epsilon_covered)
    coefficients: Dict[int, int] = defaultdict(int)

    for epsilon_assignment in product(_EPSILON_TERMS, repeat=len(epsilon_roots)):
        assigned: Dict[int, int] = {}
        coefficient = 1
        valid = True
        for roots, (colours, sign) in zip(epsilon_roots, epsilon_assignment):
            coefficient *= sign
            for root, colour in zip(roots, colours):
                previous = assigned.setdefault(root, colour)
                if previous != colour:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue

        free_assignments = product(range(COLOUR_BASE), repeat=len(free_roots))
        for free_colours in free_assignments:
            root_colours = dict(assigned)
            root_colours.update(zip(free_roots, free_colours))
            digits = tuple(root_colours[_find(parent, axis)] for axis in range(line_count))
            coefficients[encode_digits(digits, COLOUR_BASE)] += coefficient

    return tuple((code, value) for code, value in coefficients.items() if value)


def _relative_orbits(colour_codes, line_count: int, epsilon_count: int):
    representatives = set()
    for code in colour_codes:
        digits = decode_digits(code, line_count, COLOUR_BASE)
        representatives.add(min(
            encode_digits(tuple(permutation[colour] for colour in digits), COLOUR_BASE)
            for permutation in _COLOUR_PERMUTATIONS
        ))

    members_by_representative = {}
    zero_orbits = 0
    for representative in sorted(representatives):
        digits = decode_digits(representative, line_count, COLOUR_BASE)
        members = {}
        inconsistent = False
        for permutation in _COLOUR_PERMUTATIONS:
            code = encode_digits(
                tuple(permutation[colour] for colour in digits), COLOUR_BASE
            )
            character = parity(permutation) ** epsilon_count
            previous = members.setdefault(code, character)
            inconsistent |= previous != character
        if inconsistent:
            zero_orbits += 1
        else:
            members_by_representative[representative] = members
    return members_by_representative, zero_orbits


def _fuse_incidence_spin(incidence_spin, axis_count, epsilon_count, stats,
                         *, use_colour_orbits=True):
    """Fuse merged Wick spin maps on colour representatives.

    Both packed source coefficients and full graphs use this kernel.  Axes
    and spin keys belong to the caller's route frame; only delta/epsilon
    colour tensors act here.  No tolerance is applied before final output.
    """
    nonzero = {
        incidence: {spin: value for spin, value in spins.items() if value != 0}
        for incidence, spins in incidence_spin.items()
    }
    nonzero = {incidence: spins for incidence, spins in nonzero.items() if spins}
    polynomials = {
        incidence: dict(_incidence_colour_polynomial(incidence, axis_count))
        for incidence in nonzero
    }
    colours = {code for polynomial in polynomials.values() for code in polynomial}
    if use_colour_orbits:
        orbits, zero_orbits = _relative_orbits(colours, axis_count, epsilon_count)
    else:
        orbits = {code: {code: 1} for code in sorted(colours)}
        zero_orbits = 0
    stats["incidence_spin_pairs"] += sum(map(len, nonzero.values()))
    stats["exact_colour_codes"] += len(colours)
    stats["colour_orbits"] += len(orbits) + zero_orbits if use_colour_orbits else 0
    stats["zero_stabilizer_orbits"] += zero_orbits
    stats["reconstructed_colour_codes"] += sum(map(len, orbits.values()))
    for representative, members in orbits.items():
        spins = defaultdict(complex)
        for incidence, polynomial in polynomials.items():
            colour_coefficient = polynomial.get(representative, 0)
            if not colour_coefficient:
                continue
            stats["representative_incidence_checks"] += 1
            for spin, coefficient in nonzero[incidence].items():
                spins[spin] += colour_coefficient * coefficient
        spins = {spin: value for spin, value in spins.items() if value != 0}
        if spins:
            yield members, spins
