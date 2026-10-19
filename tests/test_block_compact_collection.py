"""Endpoint-orbit collection must preserve every signed open-axis view."""

from dataclasses import replace
from fractions import Fraction
import unittest
from unittest.mock import patch

from kernels.block_reduction.canonical import OrbitCanonicalizer
from kernels.block_reduction.global_layout import (CompactPairingTerms, PairingFamily,
    _collect_compact_family, _endpoint_actions, _prepare)
from kernels.block_reduction.model import (Network, TensorSpec, TensorSymmetry,
                                   WeightedNetwork, compose_transposes)


def _family(*, coefficient=Fraction(-3, 2), output="ABCDEF", aliases=False):
    specs = {
        name: TensorSpec(name, (3, 4, 3, 4),
                         ("color", "spin", "color", "spin"),
                         kind="peram", identity="S",
                         peram_ports=((0, 1), (2, 3)))
        for name in ("S0", "S1")
    }
    for name in ("G", "H"):
        specs[name] = TensorSpec(name, (3, 4, 3), ("color", "spin", "color"))
    network = Network.from_einsum(
        "abgh,cdij,efkl,abA,cdB,efC,ghD,ijE,klF->" + output,
        ("S0", "S0", "S0", "G", "G", "G", "H", "H", "H"))
    by_pair = (tuple(tuple("S0" if i == j else "S1" for j in range(3))
                     for i in range(3)) if aliases else ())
    terms = CompactPairingTerms(WeightedNetwork(network, coefficient),
                                (0, 1, 2), ((0, 1, 2),), specs, by_pair)
    return PairingFamily(terms, terms.factor_positions), specs


def _direct(family, specs):
    canonicalizer = OrbitCanonicalizer(specs)
    ledger = {}
    for term in family.terms:
        canonical = canonicalizer.canonicalize(term.network, fixed_output=True)
        value = term.coefficient * canonical.scalar
        previous = ledger.get(canonical.key)
        value += previous.coefficient if previous else 0
        if value:
            ledger[canonical.key] = WeightedNetwork(canonical.network, value)
        else:
            ledger.pop(canonical.key, None)
    return ledger


class CompactCollectionTests(unittest.TestCase):
    def test_signed_alias_family_matches_direct_fixed_axis_collection(self):
        for aliases in (False, True):
            for output in ("ABCDEF", "BCADEF"):
                with self.subTest(aliases=aliases, output=output):
                    family, specs = _family(aliases=aliases, output=output)
                    expected = _direct(family, specs)
                    self.assertEqual(_collect_compact_family(family, specs), expected)
                    self.assertEqual(len(expected), 6)
                    self.assertEqual({term.coefficient for term in expected.values()},
                                     {Fraction(-3, 2), Fraction(3, 2)})

    def test_non_involutive_endpoint_views_and_composed_output(self):
        family, specs = _family(aliases=True)
        pairings, sink_types, source_types, _, sinks, sources, identities = _prepare(
            family, specs)
        witnesses = _endpoint_actions(family, specs, sinks, sources, sink_types,
                                      source_types, identities, pairings)
        self.assertTrue(any(compose_transposes(w.output_permutation,
                           w.output_permutation) != tuple(range(6)) for w in witnesses))
        p, q = (1, 2, 0, 4, 5, 3), (3, 0, 4, 1, 5, 2)
        output = family.terms.reference.network.output
        composed = compose_transposes(p, q)
        viewed = replace(family.terms, reference=replace(family.terms.reference,
            network=replace(family.terms.reference.network,
                            output=tuple(output[i] for i in composed))))
        family = PairingFamily(viewed, family.factor_positions)
        self.assertEqual(_collect_compact_family(family, specs), _direct(family, specs))

    def test_materializes_one_network_per_endpoint_orbit(self):
        family, specs = _family()
        expected = _direct(family, specs)
        with patch.object(CompactPairingTerms, "term", autospec=True,
                          side_effect=CompactPairingTerms.term) as materialize:
            self.assertEqual(_collect_compact_family(family, specs), expected)
            self.assertEqual(materialize.call_count, 1)

    def test_zero_coefficient_and_negative_identity_are_zero(self):
        family, specs = _family(coefficient=0)
        self.assertEqual(_collect_compact_family(family, specs), {})
        family, specs = _family()
        specs["G"] = replace(specs["G"],
                              symmetries=(TensorSymmetry((0, 1, 2), -1),))
        terms = replace(family.terms, tensor_specs=specs)
        family = PairingFamily(terms, family.factor_positions)
        self.assertEqual(_collect_compact_family(family, specs), _direct(family, specs))
        self.assertEqual(_collect_compact_family(family, specs), {})

    def test_signed_spectator_symmetry_keeps_nonzero_antisymmetric_views(self):
        family, specs = _family()
        specs["J"] = TensorSpec("J", (3, 4, 3, 4, 4, 4),
            ("color", "spin", "color", "spin", "spin", "spin"),
            symmetries=(TensorSymmetry((2, 3, 0, 1, 5, 4), -1),))
        network = Network.from_einsum(
            "abgh,cdij,efkl,abcdAB,efC,ghD,ijE,klF->BCADEF",
            ("S0", "S0", "S0", "J", "G", "H", "H", "H"))
        terms = replace(family.terms, tensor_specs=specs,
            reference=replace(family.terms.reference, network=network))
        family = PairingFamily(terms, family.factor_positions)
        expected = _direct(family, specs)
        self.assertTrue(expected)
        self.assertEqual(_collect_compact_family(family, specs), expected)

    def test_equal_numerical_propagators_keep_separate_physical_classes(self):
        family, specs = _family()
        terms = replace(family.terms, endpoint_classes=((0, 1), (2,)),
                        tensor_by_pair=())
        family = PairingFamily(terms, family.factor_positions)
        expected = _direct(family, specs)
        self.assertEqual(len(expected), 2)
        self.assertEqual(_collect_compact_family(family, specs), expected)

    def test_explicit_family_is_rejected(self):
        family, specs = _family()
        with self.assertRaisesRegex(TypeError, "CompactPairingTerms"):
            _collect_compact_family(replace(family, terms=tuple(family.terms)), specs)


if __name__ == "__main__":
    unittest.main()
