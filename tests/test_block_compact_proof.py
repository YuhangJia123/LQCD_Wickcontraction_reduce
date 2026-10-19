"""Compact certificates prove exchange partitions without expanding W graphs."""

from dataclasses import replace
import unittest
from unittest.mock import patch

from kernels.block_reduction import BasisView, block_reduce
from kernels.block_reduction import planner
from kernels.block_reduction.global_layout import CompactPairingTerms, PairingFamily
from tests.test_block_compact_family import _fixture


def _plan():
    explicit, compact, specs = _fixture()
    contracts = {i: term.network for i, term in enumerate(explicit.terms)}
    # An inverse-sensitive three-cycle remains part of the complete target.
    permutation = (1, 2, 0, 3)
    views = {i: BasisView(i, scalar=-1, permutation=permutation) for i in contracts}
    reference = compact.terms.reference
    raw = replace(compact.terms, reference=replace(reference, network=replace(
        reference.network, output=tuple(reference.network.output[i] for i in permutation))))
    family = PairingFamily(raw, compact.factor_positions)
    plan = block_reduce(contracts, views, [-term.coefficient for term in explicit.terms],
                        tensor_specs=specs, pairing_family=family, max_w_types=2)
    return plan, family


class CompactProofTests(unittest.TestCase):
    def test_internal_mapping_is_reused_but_public_verification_is_independent(self):
        calls = []
        family_calls = []
        original = planner._collect
        collect_family = planner._collect_family

        def observed(terms, specs):
            calls.append(terms)
            return original(terms, specs)

        def observed_family(family, specs):
            family_calls.append(family)
            return collect_family(family, specs)

        with patch.object(planner, '_collect', side_effect=observed), \
             patch.object(planner, '_collect_family', side_effect=observed_family):
            plan, family = _plan()
            self.assertEqual(len(family_calls), 1)
            self.assertIs(family_calls[0].terms, family.terms)
            self.assertEqual(sum(terms is plan.original_terms for terms in calls), 1)
            calls.clear()
            family_calls.clear()
            self.assertTrue(plan.verify())
            self.assertEqual(len(family_calls), 1)
            self.assertIs(family_calls[0].terms, family.terms)
            self.assertEqual(sum(terms is plan.original_terms for terms in calls), 1)

    def test_compact_mapping_matches_generic_ledger_with_same_orbit_engine(self):
        from kernels.block_reduction.canonical import OrbitCanonicalizer

        plan, family = _plan()
        specs = plan.tensor_specs
        engine = OrbitCanonicalizer(specs)
        with patch.object(planner, '_canonicalizers') as engines:
            engines.get.return_value = [engine]
            fast = planner._collect_family(family, specs)
            slow = planner._collect(family.terms, specs)
        self.assertEqual(planner._state_key(fast), planner._state_key(slow))

    def test_compact_subclass_is_verified_as_explicit_terms(self):
        plan, _family = _plan()
        certificate = plan.family_certificate

        class InheritedTerms(CompactPairingTerms):
            pass

        inherited = InheritedTerms(certificate.raw_terms.reference,
                                   certificate.raw_terms.factor_positions,
                                   certificate.raw_terms.endpoint_classes,
                                   certificate.raw_terms.tensor_specs,
                                   certificate.raw_terms.tensor_by_pair)
        with patch.object(planner, '_expanded_sum', wraps=planner._expanded_sum) as expanded:
            self.assertTrue(replace(plan, family_certificate=replace(
                certificate, raw_terms=inherited)).verify())
            self.assertEqual(expanded.call_count, 1)

        class AlteredTerms(CompactPairingTerms):
            def __iter__(self):
                for index, term in enumerate(super().__iter__()):
                    yield replace(term, coefficient=2 * term.coefficient) if index == 0 else term

        altered = AlteredTerms(certificate.raw_terms.reference,
                               certificate.raw_terms.factor_positions,
                               certificate.raw_terms.endpoint_classes,
                               certificate.raw_terms.tensor_specs,
                               certificate.raw_terms.tensor_by_pair)
        with patch.object(planner, '_expanded_sum', wraps=planner._expanded_sum) as expanded:
            self.assertFalse(replace(plan, family_certificate=replace(
                certificate, raw_terms=altered)).verify())
            self.assertEqual(expanded.call_count, 0)

    def test_compact_proof_does_not_expand_w_graphs(self):
        with patch.object(planner, '_expanded_sum', side_effect=AssertionError('expanded')):
            plan, _family = _plan()
            self.assertEqual(plan.rewrites, ())
            self.assertTrue(plan.verify())
        explicit = replace(plan.family_certificate,
                           raw_terms=tuple(plan.family_certificate.raw_terms))
        with patch.object(planner, '_expanded_sum', wraps=planner._expanded_sum) as expanded:
            self.assertTrue(replace(plan, family_certificate=explicit).verify())
            self.assertEqual(expanded.call_count, 1)

    def test_compact_placement_and_fused_term_tampering_is_rejected(self):
        plan, _family = _plan()
        certificate = plan.family_certificate
        first, second = certificate.pairs
        bad_pairs = (
            certificate.pairs[:1],
            (first, first),
            (tuple(reversed(first)), second),
            ((first[0], second[0]), (first[1], second[1])),
            ((first[0], second[0]), second),
            ((first[0], 99), second),
        )
        for pairs in bad_pairs:
            with self.subTest(pairs=pairs):
                self.assertFalse(replace(plan, family_certificate=replace(
                    certificate, pairs=pairs)).verify())
        for coefficient in (0, 2, -1):
            terms = (replace(certificate.fused_terms[0], coefficient=coefficient),)
            self.assertFalse(replace(plan, family_certificate=replace(
                certificate, fused_terms=terms)).verify())
        network = certificate.fused_terms[0].network
        factors = list(network.factors)
        block_index = next(i for i, factor in enumerate(factors)
                           if factor.tensor in {d.name for d in plan.w_definitions})
        factor = factors[block_index]
        factors[block_index] = replace(factor, indices=(factor.indices[2:4]
                                   + factor.indices[:2] + factor.indices[4:]))
        wrong = replace(certificate.fused_terms[0], network=replace(
            network, factors=tuple(factors)))
        self.assertFalse(replace(plan, family_certificate=replace(
            certificate, fused_terms=(wrong,))).verify())

    def test_public_verified_flag_cannot_hide_changed_raw_or_target(self):
        plan, _family = _plan()
        self.assertTrue(plan.stats['symbolic_verified'])
        certificate = plan.family_certificate
        raw = replace(certificate.raw_terms, reference=replace(
            certificate.raw_terms.reference, coefficient=2))
        self.assertFalse(replace(plan, family_certificate=replace(
            certificate, raw_terms=raw)).verify())
        original = list(plan.original_terms)
        original[0] = replace(original[0], coefficient=2 * original[0].coefficient)
        self.assertFalse(replace(plan, original_terms=tuple(original)).verify())
        reconstruction = list(plan.reconstruction)
        reconstruction[0] = replace(reconstruction[0],
                                    permutation=(2, 0, 1, 3))
        self.assertFalse(replace(plan, reconstruction=tuple(reconstruction)).verify())


if __name__ == '__main__':
    unittest.main()
