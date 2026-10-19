"""Optional family input remains subordinate to the complete three tables."""

from dataclasses import replace
import unittest

import numpy as np

from kernels.block_reduction import BasisView, block_reduce, evaluate_expression, evaluate_plan
from kernels.block_reduction.global_layout import PairingFamily
from kernels.block_reduction.model import WeightedNetwork
from tests.test_block_global_adversarial import _two_family_fixture


class FamilyCertificateTests(unittest.TestCase):
    def test_complete_target_and_placement_certificate(self):
        family, specs = _two_family_fixture()
        contracts = {i: term.network for i, term in enumerate(family.terms)}
        # Three-cycles stress the inverse-transpose convention.
        p = (1, 2, 0, 3)
        views = {i: BasisView(i, scalar=-1, permutation=p) for i in contracts}
        coefficients = [-term.coefficient for term in family.terms]
        viewed_family = PairingFamily(tuple(WeightedNetwork(
            replace(term.network, output=tuple(term.network.output[j] for j in p)),
            term.coefficient) for term in family.terms), family.factor_positions)
        plan = block_reduce(contracts, views, coefficients, tensor_specs=specs,
                            pairing_family=viewed_family, max_w_types=2)
        self.assertIsNotNone(plan.family_certificate)
        self.assertEqual(len(plan.networks), 1)
        self.assertTrue(plan.verify())
        self.assertFalse(replace(plan, w_definitions=()).verify())
        self.assertFalse(replace(plan, stats={**plan.stats, 'max_w_types': 1}).verify())
        self.assertEqual(plan.stats['search_seconds'], 5.0)
        rng = np.random.default_rng(191)
        arrays = {name: rng.standard_normal(spec.shape) +
                  1j * rng.standard_normal(spec.shape) for name, spec in specs.items()}
        np.testing.assert_allclose(evaluate_plan(plan, arrays),
            evaluate_expression(contracts, views, coefficients, arrays), rtol=1e-11, atol=1e-9)
        certificate = plan.family_certificate
        altered = replace(certificate, pairs=certificate.pairs[:1])
        self.assertFalse(replace(plan, family_certificate=altered).verify())
        altered = replace(certificate, fused_terms=(replace(certificate.fused_terms[0],
                         coefficient=3),) + certificate.fused_terms[1:])
        self.assertFalse(replace(plan, family_certificate=altered).verify())

    def test_wrong_raw_family_cannot_override_the_three_table_target(self):
        family, specs = _two_family_fixture()
        contracts = {i: term.network for i, term in enumerate(family.terms)}
        views = {i: BasisView(i) for i in contracts}
        coefficients = [term.coefficient for term in family.terms]
        broken = replace(family, terms=(replace(family.terms[0], coefficient=2),) + family.terms[1:])
        with self.assertRaisesRegex(ValueError, 'differs.*target'):
            block_reduce(contracts, views, coefficients, tensor_specs=specs, pairing_family=broken)

    def test_zero_search_budget_preserves_the_input_plan(self):
        family, specs = _two_family_fixture()
        contracts = {i: term.network for i, term in enumerate(family.terms)}
        views = {i: BasisView(i) for i in contracts}
        coefficients = [term.coefficient for term in family.terms]
        plan = block_reduce(contracts, views, coefficients, tensor_specs=specs,
                            pairing_family=family, search_budget=0)
        self.assertIsNone(plan.family_certificate)
        self.assertTrue(plan.verify())
        self.assertEqual(plan.stats['network_count'], plan.stats['initial_network_count'])


if __name__ == '__main__':
    unittest.main()
