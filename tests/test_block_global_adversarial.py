"""Adversarial invariants for optional independent global layout search."""

from collections import defaultdict
from itertools import permutations, product
import unittest

import numpy as np
from opt_einsum import contract

from kernels.block_reduction.global_layout import PairingFamily, find_pairing_layouts
from kernels.block_reduction.canonical import OrbitCanonicalizer
from kernels.block_reduction.model import Network, TensorSpec, WeightedNetwork, exact_scalar
from kernels.block_reduction.planner import _block_specs


def _two_family_fixture(*, zero=False):
    specs = {
        name: TensorSpec(name, (3, 4, 3, 4),
                         ("color", "spin", "color", "spin"),
                         kind="peram", peram_ports=((0, 1), (2, 3)))
        for name in ("S", "T")
    }
    for name in ("G1", "H1", "G2", "H2"):
        specs[name] = TensorSpec(name, (3, 4, 3, 4, 4),
                                 ("color", "spin", "color", "spin", "spin"))
    terms = []
    for first, second in product((False, True), repeat=2):
        source_one = ("kl", "ij") if first else ("ij", "kl")
        source_two = ("op", "mn") if second else ("mn", "op")
        equation = (f"ab{source_one[0]},cd{source_one[1]},"
                    f"ef{source_two[0]},gh{source_two[1]},"
                    "abcdE,ijklF,efghG,mnopH->EFGH")
        network = Network.from_einsum(equation,
                                      ("S", "S", "T", "T", "G1", "H1", "G2", "H2"))
        coefficient = 0 if zero else (-1) ** (first + second)
        terms.append(WeightedNetwork(network, coefficient))
    return PairingFamily(tuple(terms), (0, 1, 2, 3)), specs


class GlobalLayoutAdversarialTests(unittest.TestCase):
    def test_distinct_propagator_endpoint_types_never_cross(self):
        specs = {
            name: TensorSpec(name, (3, 4, 3, 4),
                             ("color", "spin", "color", "spin"),
                             kind="peram", peram_ports=((0, 1), (2, 3)))
            for name in ("S", "T")
        }
        specs["G"] = TensorSpec("G", (3, 4, 4), ("color", "spin", "spin"))
        specs.update({name: TensorSpec(name, (3, 4), ("color", "spin"))
                      for name in ("J0", "J1", "J2", "J3")})
        sinks = ("ab", "cd", "ef", "gh")
        sources = ("ij", "kl", "mn", "op")
        terms = []
        for pairing in permutations(range(4)):
            source_ports = ",".join(sinks[i] + sources[pairing[i]] for i in range(4))
            equation = (source_ports + ",abA,cdB,efC,ghD,ij,kl,mn,op->ABCD")
            sign = (-1) ** sum(pairing[i] > pairing[j] for i in range(4)
                               for j in range(i + 1, 4))
            terms.append(WeightedNetwork(Network.from_einsum(
                equation, ("S", "S", "T", "T", "G", "G", "G", "G",
                           "J0", "J1", "J2", "J3")), sign))
        family = PairingFamily(tuple(terms), (0, 1, 2, 3))
        result = find_pairing_layouts(family, specs, max_w_types=2,
                                      search_seconds=5, top_k=10)
        self.assertTrue(result.endpoint_witnesses)
        self.assertEqual(len(set(result.sink_types)), 1)
        for witness in result.endpoint_witnesses:
            self.assertEqual({witness.sink_permutation[0], witness.sink_permutation[1]},
                             {0, 1})
            self.assertEqual({witness.source_permutation[0], witness.source_permutation[1]},
                             {0, 1})
            self.assertEqual({witness.sink_permutation[2], witness.sink_permutation[3]},
                             {2, 3})
            self.assertEqual({witness.source_permutation[2], witness.source_permutation[3]},
                             {2, 3})
        self.assert_layout_scores_match_slow(result, specs)

    def assert_layout_scores_match_slow(self, result, specs):
        expanded, _, _ = _block_specs(specs)
        canonicalizer = OrbitCanonicalizer(expanded)
        for layout in result.layouts:
            fixed_weights = defaultdict(lambda: 0)
            fixed_networks = {}
            for term in layout.fused_terms:
                fixed = canonicalizer.canonicalize(term.network, fixed_output=True)
                if fixed.scalar:
                    fixed_weights[fixed.key] += exact_scalar(term.coefficient) * fixed.scalar
                    fixed_networks[fixed.key] = fixed.network
            slow_count = len({
                canonicalizer.canonicalize(fixed_networks[key], fixed_output=False).key
                for key, weight in fixed_weights.items() if weight
            })
            self.assertEqual(layout.network_count, slow_count)

    def test_maximum_w_type_budget_applies_to_actual_fused_layout(self):
        family, specs = _two_family_fixture()
        one = find_pairing_layouts(family, specs, max_w_types=1, search_seconds=5)
        self.assertTrue(one.layouts)
        for layout in one.layouts:
            used = {factor.tensor for term in layout.fused_terms
                    for factor in term.network.factors if factor.tensor not in specs}
            self.assertLessEqual(len(used), 1)
        two = find_pairing_layouts(family, specs, max_w_types=2, search_seconds=5)
        self.assertTrue(two.layouts)
        used = {factor.tensor for term in two.layouts[0].fused_terms
                for factor in term.network.factors if factor.tensor not in specs}
        self.assertEqual(len(used), 2)

    def test_zero_coefficients_do_not_score_as_live_networks(self):
        family, specs = _two_family_fixture(zero=True)
        result = find_pairing_layouts(family, specs, max_w_types=2, search_seconds=5)
        self.assertTrue(result.layouts)
        self.assertEqual(result.layouts[0].network_count, 0)

    def test_discovered_three_cycle_has_correct_signed_output_transpose(self):
        specs = {
            "S": TensorSpec("S", (3, 4, 3, 4),
                            ("color", "spin", "color", "spin"),
                            kind="peram", peram_ports=((0, 1), (2, 3))),
            "G": TensorSpec("G", (3, 4, 4), ("color", "spin", "spin")),
        }
        specs.update({name: TensorSpec(name, (3, 4), ("color", "spin"))
                      for name in ("J0", "J1", "J2")})
        sinks = ("ab", "cd", "ef")
        sources = ("ij", "kl", "mn")
        output = "ABC"
        by_pairing = {}
        terms = []
        for pairing in permutations(range(3)):
            inversions = sum(pairing[i] > pairing[j] for i in range(3)
                             for j in range(i + 1, 3))
            signs = (-1) ** inversions
            equation = (",".join(sinks[i] + sources[pairing[i]] for i in range(3))
                        + ",abA,cdB,efC,ij,kl,mn->" + output)
            term = WeightedNetwork(
                Network.from_einsum(equation,
                                    ("S", "S", "S", "G", "G", "G", "J0", "J1", "J2")),
                signs,
            )
            terms.append(term)
            by_pairing[pairing] = term
        family = PairingFamily(tuple(terms), (0, 1, 2))
        result = find_pairing_layouts(family, specs, search_seconds=5, top_k=10)
        self.assertTrue(result.endpoint_witnesses)
        cyclic = [item for item in result.endpoint_witnesses
                  if item.sink_permutation == (1, 2, 0)
                  and item.source_permutation == (0, 1, 2)]
        self.assertTrue(cyclic)
        self.assertTrue(any(item.sign == -1 for item in result.endpoint_witnesses))
        rng = np.random.default_rng(20261014)
        arrays = {name: rng.standard_normal(spec.shape)
                  for name, spec in specs.items()}

        def value(term):
            equation = (",".join("".join(f.indices) for f in term.network.factors)
                        + "->" + "".join(term.network.output))
            return term.coefficient * contract(
                equation, *(arrays[f.tensor] for f in term.network.factors),
                optimize="greedy",
            )

        for witness in result.endpoint_witnesses:
            for pairing in by_pairing:
                image = [None] * 3
                for old_sink, old_source in enumerate(pairing):
                    image[witness.sink_permutation[old_sink]] = (
                        witness.source_permutation[old_source])
                np.testing.assert_allclose(
                    value(by_pairing[tuple(image)]),
                    witness.sign * value(by_pairing[pairing]).transpose(
                        witness.output_permutation),
                    rtol=1e-12, atol=1e-12,
                )
        self.assert_layout_scores_match_slow(result, specs)


if __name__ == "__main__":
    unittest.main()
