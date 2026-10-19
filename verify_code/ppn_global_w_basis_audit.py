"""Audit one reusable two-peram W basis on the four-spin normal ppn example.

This is an output-stage research audit, not a production block planner.  One
12^4 complex W array is built from an isospin-degenerate propagator and may
occur four times in a contraction.  Candidate quark pairs may cross nucleon
boundaries.  No upper-two-component restriction is applied.

The printed 48 is an admissible count of contracted W-network expressions for
the fixed pp spin-singlet/n-up projection, not a minimum-operation proof and
not a reconstruction of PACS-CS's unpublished 93-contraction list.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from itertools import combinations
import json

import numpy as np
import opt_einsum

from kernels import correlator_wick_normalizer as core
from verify_code.normal_validation_inputs import _inputs
from verify_code.ppn_projected_93_audit import _projector_pp_singlet_n_up
from kernels.wick_reducer.expression_output import contraction_expression
from kernels.wick_reducer.operator_model import fields_for_operators, operator_library


def pair_matchings(indices):
    """All two-edge matchings of four of the given five quark positions."""
    for quartet in combinations(indices, 4):
        a, b, c, d = quartet
        yield ((a, b), (c, d))
        yield ((a, c), (b, d))
        yield ((a, d), (b, c))


def orbit(pairing, pairs):
    """Enumerate the complete, non-overlapping exchange orbit."""
    result = []
    for mask in range(1 << len(pairs)):
        image = list(pairing)
        for bit, (left, right) in enumerate(pairs):
            if mask & (1 << bit):
                image[left], image[right] = image[right], image[left]
        result.append(tuple(image))
    if len(set(result)) != len(result):
        raise AssertionError("exchange action has a nontrivial stabilizer")
    return tuple(result)


def source_proton_swap(pairing):
    """Exchange the two identical source proton operators."""
    slots = {0: 3, 3: 0, 1: 4, 4: 1, 2: 5, 5: 2}
    return tuple(slots.get(index, index) for index in pairing)


def sink_proton_swap(pairing):
    """Exchange the two identical sink proton operators."""
    slots = {0: 3, 3: 0, 1: 4, 4: 1, 2: 5, 5: 2}
    result = list(pairing)
    for old, new in slots.items():
        result[new] = pairing[old]
    return tuple(result)


def w_tensor(propagator):
    matrix = propagator.reshape(12, 12)
    product = opt_einsum.contract("ai,bj->abij", matrix, matrix)
    exchange = opt_einsum.contract("aj,bi->abij", matrix, matrix)
    return (product - exchange).reshape((3, 4, 3, 4, 3, 4, 3, 4))


class Case:
    def __init__(self):
        self.operators = tuple(
            replace(operator, mode="normal")
            for operator in operator_library()["ppn"]
        )
        _fields, self.quarks, self.antiquarks = fields_for_operators(self.operators)
        self.terms = tuple(core.enumerate_wicks(self.quarks, self.antiquarks))
        self.by_pairing = {term.pairing: term for term in self.terms}
        self.gamma_count = sum(bool(operator.gamma) for operator in self.operators)
        self.projector = _projector_pp_singlet_n_up()
        if len(self.terms) != 2880:
            raise AssertionError("unexpected ppn Wick count")

    def equation(self, pairing, *, isospin_degenerate=True):
        return contraction_expression(
            self.operators, pairing,
            isospin_degenerate=isospin_degenerate,
        )

    def direct(self, pairing, values, *, isospin_degenerate=True):
        equation, names, _ = self.equation(
            pairing, isospin_degenerate=isospin_degenerate,
        )
        return self.by_pairing[pairing].sign * opt_einsum.contract(
            equation, *(values[name] for name in names), optimize="greedy",
        )

    def fused(self, pairing, pairs, values, w, *, project=False):
        """Replace each selected peram pair with the *same* W array."""
        equation, names, output = self.equation(pairing)
        subscripts = equation.split("->", 1)[0].split(",")
        removed = set()
        w_indices = []
        for left, right in pairs:
            i, j = self.gamma_count + left, self.gamma_count + right
            removed.update((i, j))
            w_indices.append(
                subscripts[i][:2] + subscripts[j][:2]
                + subscripts[i][2:] + subscripts[j][2:]
            )
        factors = [
            subscript for index, subscript in enumerate(subscripts)
            if index not in removed
        ] + w_indices
        arrays = [
            values[name] for index, name in enumerate(names)
            if index not in removed
        ] + [w] * len(pairs)
        if project:
            factors.append("".join(output))
            arrays.append(self.projector)
            output = ()
        return self.by_pairing[pairing].sign * opt_einsum.contract(
            ",".join(factors) + "->" + "".join(output),
            *arrays, optimize="greedy",
        )


def relative_error(left, right):
    difference = np.linalg.norm(left - right)
    scale = max(np.linalg.norm(left), np.linalg.norm(right), 1e-30)
    return float(difference / scale)


def quotient_by_proton_swaps(representatives, pairs, *, exchange_sink):
    """Quotient a fixed W matching by admissible whole-proton swaps."""
    def canonical(pairing):
        return min(orbit(pairing, pairs))

    if exchange_sink:
        swaps = (
            lambda pairing: pairing, sink_proton_swap, source_proton_swap,
            lambda pairing: source_proton_swap(sink_proton_swap(pairing)),
        )
    else:
        swaps = (lambda pairing: pairing, source_proton_swap)
    remaining = set(representatives)
    output = []
    while remaining:
        pairing = min(remaining)
        members = {canonical(swap(pairing)) for swap in swaps}
        if not members <= remaining:
            raise AssertionError("whole-proton swap did not form W-network orbits")
        output.append((pairing, len(members), members))
        remaining.difference_update(members)
    return output


def audit(seed=20261013):
    case = Case()
    values = _inputs(seed)
    # One numerical peram family.  Physical u/d Wick pairing rules are unchanged.
    values["peram_d_sin_sou_x_x"] = values["peram_u_sin_sou_x_x"]
    w = w_tensor(values["peram_u_sin_sou_x_x"])
    first_pairing = case.terms[0].pairing
    flavor_indices = {flavor: tuple(
        index for index, field in enumerate(case.quarks)
        if field.flavor == flavor
    ) for flavor in ("u", "d")}

    pair_errors = []
    for flavor, indices in flavor_indices.items():
        for pair in combinations(indices, 2):
            left, right = orbit(first_pairing, (pair,))
            if case.by_pairing[left].sign != -case.by_pairing[right].sign:
                raise AssertionError("wrong Wick exchange sign")
            direct = case.direct(left, values) + case.direct(right, values)
            fused = case.fused(left, (pair,), values, w)
            pair_errors.append({
                "flavor": flavor, "quark_pair": pair,
                "same_operator": case.quarks[pair[0]].operator
                == case.quarks[pair[1]].operator,
                "relative_error": relative_error(direct, fused),
            })

    # Examine every way to use two u pairs and two d pairs with one W type.
    d_indices = flavor_indices["d"]
    d_matchings = (
        ((d_indices[0], d_indices[1]), (d_indices[2], d_indices[3])),
        ((d_indices[0], d_indices[2]), (d_indices[1], d_indices[3])),
        ((d_indices[0], d_indices[3]), (d_indices[1], d_indices[2])),
    )
    counts = Counter()
    matching_results = []
    for u_pairs in pair_matchings(flavor_indices["u"]):
        for d_pairs in d_matchings:
            pairs = u_pairs + d_pairs
            representatives = {
                min(orbit(term.pairing, pairs)) for term in case.terms
            }
            if len(representatives) != 180:
                raise AssertionError("four W factors did not partition 2880 Wicks")
            sink_permuted_pairs = {
                tuple(sorted((
                    {0: 3, 3: 0, 1: 4, 4: 1, 2: 5, 5: 2}.get(left, left),
                    {0: 3, 3: 0, 1: 4, 4: 1, 2: 5, 5: 2}.get(right, right),
                ))) for left, right in pairs
            }
            compatible_with_sink_exchange = sink_permuted_pairs == {
                tuple(sorted(pair)) for pair in pairs
            }
            classes = quotient_by_proton_swaps(
                representatives, pairs,
                exchange_sink=compatible_with_sink_exchange,
            )
            counts[len(classes)] += 1
            matching_results.append({
                "pairs": pairs,
                "projected_expression_count": len(classes),
                "sink_proton_swap_preserves_matching": compatible_with_sink_exchange,
            })

    # The best objective value found by exhaustive matching enumeration is
    # 48.  A second all-cross-nucleon u matching ties this value.
    pairs = ((0, 1), (3, 4), (2, 5), (6, 7))
    representatives = sorted({
        min(orbit(term.pairing, pairs)) for term in case.terms
    })
    fused_values = {
        pairing: case.fused(pairing, pairs, values, w)
        for pairing in representatives
    }
    open_fused = sum(fused_values.values())
    open_direct = sum(case.direct(term.pairing, values) for term in case.terms)
    open_error = relative_error(open_direct, open_fused)

    projected_orbits = quotient_by_proton_swaps(
        representatives, pairs, exchange_sink=True,
    )
    # Rebuild the *open* six-axis tensor from one contraction per orbit.
    open_from_48 = np.zeros_like(open_direct)
    for pairing, _weight, members in projected_orbits:
        representative_tensor = fused_values[pairing]
        views = (
            (pairing, 1, (0, 1, 2, 3, 4, 5)),
            (sink_proton_swap(pairing), -1, (1, 0, 2, 3, 4, 5)),
            (source_proton_swap(pairing), -1, (0, 1, 2, 4, 3, 5)),
            (source_proton_swap(sink_proton_swap(pairing)),
             1, (1, 0, 2, 4, 3, 5)),
        )
        used = set()
        for transformed, sign, axes in views:
            target = min(orbit(transformed, pairs))
            if target not in members:
                raise AssertionError("open-axis orbit view escaped its class")
            if target not in used:
                open_from_48 += sign * representative_tensor.transpose(axes)
                used.add(target)
        if used != members:
            raise AssertionError("open-axis orbit reconstruction was incomplete")
    open_from_48_error = relative_error(open_direct, open_from_48)
    projected_fused = sum(
        weight * opt_einsum.contract("ABCDEF,ABCDEF->",
                                     fused_values[pairing], case.projector)
        for pairing, weight, _members in projected_orbits
    )
    projected_direct = opt_einsum.contract(
        "ABCDEF,ABCDEF->", open_direct, case.projector,
    )
    projected_error = relative_error(projected_direct, projected_fused)
    sink_axes = (1, 0, 2, 3, 4, 5)
    source_axes = (0, 1, 2, 4, 3, 5)
    both_axes = (1, 0, 2, 4, 3, 5)
    open_transpose_error = max(
        relative_error(
            fused_values[min(orbit(swap(pairing), pairs))],
            sign * value.transpose(axes),
        )
        for pairing, value in fused_values.items()
        for swap, sign, axes in (
            (sink_proton_swap, -1, sink_axes),
            (source_proton_swap, -1, source_axes),
            (lambda p: source_proton_swap(sink_proton_swap(p)), 1, both_axes),
        )
    )
    largest_orbit_error = max(
        relative_error(
            opt_einsum.contract("ABCDEF,ABCDEF->", value, case.projector),
            opt_einsum.contract(
                "ABCDEF,ABCDEF->",
                fused_values[min(orbit(source_proton_swap(pairing), pairs))],
                case.projector,
            ),
        )
        for pairing, value in fused_values.items()
    )

    maximum_error = max(
        *(entry["relative_error"] for entry in pair_errors),
        open_error, open_from_48_error, projected_error, largest_orbit_error,
        open_transpose_error,
    )
    if maximum_error > 1e-10:
        raise AssertionError(f"W-basis validation failed: {maximum_error}")
    return {
        "model": "normal ppn, four quark Dirac components, common u/d numerical peram",
        "seed": seed,
        "raw_wicks": len(case.terms),
        "one_w_shape": list(w.shape),
        "one_w_elements": int(w.size),
        "one_w_bytes_complex128": int(w.nbytes),
        "quark_pair_choices": len(pair_errors),
        "cross_nucleon_pair_choices": sum(
            not entry["same_operator"] for entry in pair_errors
        ),
        "largest_one_pair_relative_error": max(
            entry["relative_error"] for entry in pair_errors
        ),
        "four_pair_matchings_checked": sum(counts.values()),
        "projected_expression_count_histogram": dict(counts),
        "matching_results": matching_results,
        "selected_matching": pairs,
        "open_w_network_expressions": len(representatives),
        "projected_w_network_expressions": len(projected_orbits),
        "projected_orbit_size_histogram": dict(Counter(
            weight for _pairing, weight, _members in projected_orbits
        )),
        "open_relative_error": open_error,
        "open_from_48_axis_views_relative_error": open_from_48_error,
        "projected_relative_error": projected_error,
        "largest_projected_orbit_error": largest_orbit_error,
        "largest_open_axis_transpose_error": open_transpose_error,
        "passed": True,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, ensure_ascii=False))
