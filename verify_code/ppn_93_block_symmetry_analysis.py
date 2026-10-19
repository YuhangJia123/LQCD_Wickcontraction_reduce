#!/usr/bin/env python3
"""Independent symmetry-only audit of the 93 ppn canonical blocks.

This module uses the ppn Wick pairing set and the declared label-action
generators.  It intentionally does not import ``wick_reducer.reducer`` and
does not inspect peram, Gamma, open-axis, or numerical Gram data.  Its purpose
is to identify exactly what the ppn group action can prove by itself.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Mapping, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from verify_code import ppnn_group_action as action
from verify_code.orbit_ppn import ppn_orbit_system


def _external_orbits(states: Iterable[action.He3State]) -> Tuple[Tuple[action.He3State, ...], ...]:
    """Re-enumerate the 93 state orbits from the two external swaps."""

    generators = (action.he3_sink_proton_swap, action.he3_source_proton_swap)
    unseen = set(states)
    records: List[Tuple[action.He3State, ...]] = []
    for seed in sorted(unseen):
        if seed not in unseen:
            continue
        orbit = {seed}
        pending = [seed]
        while pending:
            state = pending.pop()
            for generator in generators:
                image = generator(state)
                if image not in orbit:
                    orbit.add(image)
                    pending.append(image)
        unseen.difference_update(orbit)
        records.append(tuple(sorted(orbit)))
    return tuple(records)


def analyse() -> Mapping[str, object]:
    """Return symmetry-only counts and per-block invariants."""

    system = ppn_orbit_system()
    if (system.raw_wick_count, system.state_count, system.orbit_count) != (2880, 360, 93):
        raise AssertionError("unexpected ppn raw/state/orbit counts")

    state_fibers = system.state_to_raw
    if any(len(raw_ids) != 8 for raw_ids in state_fibers.values()):
        raise AssertionError("a ppn quotient state must contain eight raw Wick pairings")
    if any(len({system.raw_pairings[raw_id] for raw_id in raw_ids}) != 8
           for raw_ids in state_fibers.values()):
        raise AssertionError("the internal (C2)^3 action is not free on a state fiber")

    orbits = _external_orbits(system.state_partition.elements)
    if len(orbits) != 93:
        raise AssertionError("external ppn action did not produce 93 blocks")

    records: List[Dict[str, object]] = []
    for orbit_id, states in enumerate(orbits):
        representative = states[0]
        representative_raw = tuple(sorted(state_fibers[representative]))
        expanded_raw = tuple(sorted(
            raw_id for state in states for raw_id in state_fibers[state]
        ))
        if len(representative_raw) != 8 or len(expanded_raw) != 8 * len(states):
            raise AssertionError("canonical block expansion has the wrong size")
        signs = tuple(sorted(system.raw_signs[raw_id] for raw_id in representative_raw))
        if signs != (-1, -1, -1, -1, 1, 1, 1, 1):
            raise AssertionError("unexpected ppn sign character in a state fiber")
        records.append({
            "orbit_id": orbit_id,
            "state_orbit_size": len(states),
            "canonical_raw_count": len(representative_raw),
            "expanded_raw_count": len(expanded_raw),
            "internal_pairings_are_distinct": True,
            "sign_pattern": signs,
        })

    return {
        "raw_wick_count": system.raw_wick_count,
        "state_count": system.state_count,
        "canonical_block_count": len(records),
        "canonical_block_raw_count": sum(item["canonical_raw_count"] for item in records),
        "expanded_raw_count": sum(item["expanded_raw_count"] for item in records),
        "state_orbit_size_histogram": dict(Counter(item["state_orbit_size"] for item in records)),
        "canonical_raw_count_histogram": dict(Counter(item["canonical_raw_count"] for item in records)),
        "expanded_raw_count_histogram": dict(Counter(item["expanded_raw_count"] for item in records)),
        "sign_pattern_histogram": {str(key): value for key, value in Counter(
            item["sign_pattern"] for item in records
        ).items()},
        "all_internal_fibers_free": all(
            item["internal_pairings_are_distinct"] for item in records
        ),
        "blocks": records,
    }


def main() -> None:
    report = analyse()
    for key in (
        "raw_wick_count", "state_count", "canonical_block_count",
        "canonical_block_raw_count", "expanded_raw_count",
        "state_orbit_size_histogram", "canonical_raw_count_histogram",
        "expanded_raw_count_histogram", "sign_pattern_histogram",
        "all_internal_fibers_free",
    ):
        print(f"{key}: {report[key]}")


if __name__ == "__main__":
    main()
