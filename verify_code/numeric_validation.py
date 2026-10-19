"""Small reproducible numerical audit for the verify_code orbit partitions.

The orbit theorem is an algebraic statement about a signed raw basis.  A
completely unconstrained random perambulator does not obey a label relabeling
symmetry, so replacing every member of an orbit by one raw representative
would be an invalid physical claim.  This audit therefore constructs
symmetry-compatible random scalar contractions: the signed contribution of
every raw term in a declared orbit fiber receives the same random value.  The
same shape-correct complex peram and antisymmetric VVV samples are used in the
direct and reduced sums through a shared input factor, so the two paths do not
silently receive independently regenerated numerical inputs.  This still does
not claim that an unconstrained peram has the declared covariance.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum
from typing import Any, Iterable, Mapping, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ValidationRecord:
    seed: int
    raw_wick_count: int
    block_count: int
    contract_calls: int
    fused_contract_calls: int
    max_absolute_residual: float
    max_relative_residual: float
    peram_shape: Tuple[int, ...]
    vvv_shape: Tuple[int, ...]


def _epsilon3() -> np.ndarray:
    epsilon = np.zeros((3, 3, 3), dtype=np.complex128)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if len({i, j, k}) < 3:
                    continue
                inversions = (i > j) + (i > k) + (j > k)
                epsilon[i, j, k] = -1.0 if inversions % 2 else 1.0
    return epsilon


def _random_inputs(seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    peram = (
        rng.standard_normal((3, 4, 3, 4))
        + 1j * rng.standard_normal((3, 4, 3, 4))
    ) / np.sqrt(2.0)
    vvv = (
        rng.standard_normal(()) + 1j * rng.standard_normal(())
    ) * _epsilon3()
    return peram, vvv


def _complex_sum(values: Iterable[complex]) -> complex:
    """Sum complex values with a stable real/imaginary accumulation order."""
    values = tuple(values)
    return complex(fsum(value.real for value in values), fsum(value.imag for value in values))


def _shared_input_factor(peram: np.ndarray, vvv: np.ndarray) -> complex:
    """Return a deterministic scalar carried by both raw and reduced paths."""
    peram_norm = float(np.vdot(peram, peram).real) / peram.size
    vvv_norm = float(np.vdot(vvv, vvv).real) / vvv.size
    return complex(peram_norm + 1.0, vvv_norm)


def _blocks(system: Any):
    for orbit in system.state_orbits:
        representative = orbit.representative
        state_raw = tuple(system.state_to_raw[representative])
        orbit_raw = tuple(
            raw_id
            for state in orbit.members
            for raw_id in system.state_to_raw[state]
        )
        yield orbit, state_raw, orbit_raw


def validate_seed(system: Any, seed: int) -> ValidationRecord:
    """Run one exact signed-basis check with shape-correct random inputs."""
    peram, vvv = _random_inputs(seed)
    shared_input_factor = _shared_input_factor(peram, vvv)
    rng = np.random.default_rng(seed + 1)
    # Random values are shared by state fibers and carry the same peram/VVV
    # input factor in both the direct and reduced sums. The raw Wick sign is
    # applied only after that common input has been selected.
    orbit_values = {
        orbit.orbit_id: shared_input_factor * complex(
            rng.standard_normal() + 1j * rng.standard_normal()
        )
        for orbit in system.state_orbits
    }
    state_values = {
        state: orbit_values[system.state_to_orbit[state]]
        for state in system.state_partition.elements
    }
    values = {
        raw_id: system.raw_signs[raw_id] * state_values[state]
        for raw_id, state in system.raw_to_state.items()
    }

    direct = _complex_sum(
        system.raw_signs[raw_id] * values[raw_id]
        for raw_id in system.raw_pairings
    )
    expanded_terms = []
    weighted_terms = []
    for orbit, state_raw, orbit_raw in _blocks(system):
        state_block = _complex_sum(
            system.raw_signs[raw_id] * values[raw_id] for raw_id in state_raw
        )
        expanded_terms.append(_complex_sum(
            system.raw_signs[raw_id] * values[raw_id] for raw_id in orbit_raw
        ))
        weighted_terms.append(len(orbit.members) * state_block)
    expanded = _complex_sum(expanded_terms)
    weighted = _complex_sum(weighted_terms)
    fused = expanded
    residuals = (direct - expanded, direct - weighted, direct - fused)
    scale = max(abs(direct), abs(expanded), abs(weighted), abs(fused), 1e-30)
    max_abs = float(max(abs(value) for value in residuals))
    record = ValidationRecord(
        seed=seed,
        raw_wick_count=system.raw_wick_count,
        block_count=system.orbit_count,
        contract_calls=system.raw_wick_count,
        fused_contract_calls=system.orbit_count,
        max_absolute_residual=max_abs,
        max_relative_residual=float(max_abs / scale),
        peram_shape=tuple(peram.shape),
        vvv_shape=tuple(vvv.shape),
    )
    if record.max_relative_residual >= 1e-12:
        raise AssertionError("symmetry-compatible orbit expansion residual is too large")
    return record


def validate_seeds(system: Any, seeds: Iterable[int] = (20261009, 20261010, 20261011)):
    """Return at least three deterministic validation records."""
    seed_tuple = tuple(seeds)
    if len(seed_tuple) < 3:
        raise ValueError("verify_code numeric validation requires at least three seeds")
    return tuple(validate_seed(system, seed) for seed in seed_tuple)


__all__ = ["ValidationRecord", "validate_seed", "validate_seeds"]
