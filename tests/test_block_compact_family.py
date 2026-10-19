"""A compact complete family must agree with explicit symbolic Wick terms."""

from dataclasses import replace
from collections import Counter
from itertools import permutations, product
from pathlib import Path
import tempfile
import unittest

from kernels.block_reduction.global_layout import (CompactPairingTerms, PairingFamily,
                                           _prepare, find_pairing_layouts)
from kernels.block_reduction.model import Network, TensorSpec, WeightedNetwork


def _fixture():
    specs = {
        name: TensorSpec(name, (3, 4, 3, 4), ("color", "spin", "color", "spin"),
                         kind="peram", mode="normal", peram_ports=((0, 1), (2, 3)))
        for name in ("S", "T")
    }
    for name in ("G1", "H1", "G2", "H2"):
        specs[name] = TensorSpec(name, (3, 4, 3, 4, 4),
                                 ("color", "spin", "color", "spin", "spin"))
    terms = []
    for one, two in product(permutations((0, 1)), permutations((2, 3))):
        source = ("ij", "kl", "mn", "op")
        pairing = one + two
        equation = (",".join(sink + source[pairing[i]] for i, sink in enumerate(
            ("ab", "cd", "ef", "gh"))) +
                    ",abcdE,ijklF,efghG,mnopH->EFGH")
        sign = (-1) ** ((one != (0, 1)) + (two != (2, 3)))
        terms.append(WeightedNetwork(Network.from_einsum(
            equation, ("S", "S", "T", "T", "G1", "H1", "G2", "H2")), sign))
    explicit = PairingFamily(tuple(terms), (0, 1, 2, 3))
    compact = PairingFamily(CompactPairingTerms(terms[0], explicit.factor_positions,
                                               ((0, 1), (2, 3)), specs),
                            explicit.factor_positions)
    return explicit, compact, specs


class CompactFamilyTests(unittest.TestCase):
    def test_terms_representatives_and_scoring_match_explicit_family(self):
        explicit, compact, specs = _fixture()
        self.assertEqual(tuple(compact.terms), explicit.terms)
        self.assertEqual(tuple(compact.terms.pairings()),
                         ((0, 1, 2, 3), (0, 1, 3, 2),
                          (1, 0, 2, 3), (1, 0, 3, 2)))
        self.assertEqual(tuple(compact.terms.representatives(((0, 1), (2, 3)))),
                         ((0, 1, 2, 3),))
        self.assertEqual(compact.terms[-1], explicit.terms[-1])
        self.assertEqual(compact.terms[1:3], explicit.terms[1:3])
        self.assertEqual(compact.terms.coefficient((1, 0, 3, 2)), 1)
        self.assertFalse(compact.terms.contains((2, 1, 0, 3)))
        with self.assertRaisesRegex(ValueError, "physical exchange"):
            compact.terms.coefficient((2, 1, 0, 3))
        prepared = _prepare(compact, specs)
        self.assertEqual(len(prepared[0]), 4)
        self.assertEqual(prepared[3], ((0, 1), (2, 3)))
        for k in (1, 2):
            lhs = find_pairing_layouts(explicit, specs, max_w_types=k,
                                       search_seconds=5, top_k=4)
            rhs = find_pairing_layouts(compact, specs, max_w_types=k,
                                       search_seconds=5, top_k=4)
            self.assertEqual([(x.pairs, x.network_count, x.fused_terms) for x in lhs.layouts],
                             [(x.pairs, x.network_count, x.fused_terms) for x in rhs.layouts])
            self.assertTrue(all(x.count.full_family_verified for x in rhs.layouts))

    def test_metadata_rejects_wrong_classes_and_numerical_identity(self):
        explicit, _compact, specs = _fixture()
        with self.assertRaisesRegex(ValueError, "partition"):
            CompactPairingTerms(explicit.terms[0], explicit.factor_positions,
                                ((0, 1), (1, 2)), specs)
        wrong = tuple(tuple("T" if i == 0 and j == 1 else
                            "S" if i < 2 and j < 2 else
                            "T" if i >= 2 and j >= 2 else None
                            for j in range(4)) for i in range(4))
        with self.assertRaisesRegex(ValueError, "identity"):
            CompactPairingTerms(explicit.terms[0], explicit.factor_positions,
                                ((0, 1), (2, 3)), specs, wrong)
        with self.assertRaisesRegex(ValueError, "metadata"):
            _prepare(PairingFamily(CompactPairingTerms(
                explicit.terms[0], explicit.factor_positions,
                ((0, 1), (2, 3)), specs), (1, 0, 2, 3)), specs)

    def test_normal_ppn_adapter_matches_explicit_raw_terms(self):
        from verify_code.normal_block_validation import generate_normal_fixture
        from kernels.block_reduction.output_adapter import load_normal_output, pairing_family_from_operators
        from kernels.wick_reducer.operator_model import operator_library

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "normal_ppn.py"
            generate_normal_fixture("ppn", path)
            specs = load_normal_output(path).tensor_specs
        operators = tuple(replace(operator, mode="normal")
                          for operator in operator_library()["ppn"])
        compact = pairing_family_from_operators(operators, tensor_specs=specs,
                                                compact=True)
        explicit = pairing_family_from_operators(operators, tensor_specs=specs)
        self.assertEqual(len(compact.terms), 2880)
        self.assertEqual(Counter(compact.terms), Counter(explicit.terms))

    def test_normal_ppnn_compact_adapter_keeps_physical_classes(self):
        from verify_code.normal_block_validation import tensor_environment
        from kernels.block_reduction.output_adapter import pairing_family_from_operators
        from verify_code.ppnn_state_quotient_audit import _operators
        from kernels.wick_reducer import expression_output, operator_model

        operators = tuple(replace(operator, mode="normal") for operator in _operators())
        fields, quarks, antiquarks = operator_model.fields_for_operators(operators)
        first = next(operator_model.core.enumerate_wicks(quarks, antiquarks))
        equation, operands, _output = expression_output._contraction_expression_from_fields(
            operators, fields, quarks, antiquarks, first.pairing,
            isospin_degenerate=True,
        )
        source = "value = contract(" + repr(equation) + ", " + ", ".join(operands) + ")"
        specs, _arrays = tensor_environment(source)
        compact = pairing_family_from_operators(operators, tensor_specs=specs,
                                                compact=True)
        self.assertEqual(len(compact.terms), 518400)
        self.assertEqual(len(compact.factor_positions), 12)
        self.assertEqual(sorted(map(len, compact.terms.endpoint_classes)), [6, 6])
        self.assertEqual(compact.terms[0].network,
                         Network.from_einsum(equation, operands))
        self.assertTrue(all(spec.mode == "normal" for spec in specs.values()))
        self.assertEqual(specs["peram_u_sin_sou_x_x"].shape, (3, 4, 3, 4))

    def test_compact_certificate_preserves_transpose_and_target_mapping(self):
        from kernels.block_reduction import BasisView, block_reduce, evaluate_expression, evaluate_plan
        import numpy as np

        explicit, compact, specs = _fixture()
        contracts = {i: term.network for i, term in enumerate(explicit.terms)}
        permutation = (1, 2, 0, 3)
        views = {i: BasisView(i, scalar=-1, permutation=permutation) for i in contracts}
        coefficients = [-term.coefficient for term in explicit.terms]
        reference = compact.terms.reference
        viewed = replace(compact.terms, reference=replace(reference, network=replace(
            reference.network, output=tuple(reference.network.output[i] for i in permutation))))
        family = PairingFamily(viewed, compact.factor_positions)
        plan = block_reduce(contracts, views, coefficients, tensor_specs=specs,
                            pairing_family=family, max_w_types=2)
        self.assertIsInstance(plan.family_certificate.raw_terms, CompactPairingTerms)
        self.assertEqual(len(plan.networks), 1)
        self.assertTrue(plan.verify())
        rng = np.random.default_rng(407)
        arrays = {name: rng.normal(size=spec.shape) + 1j * rng.normal(size=spec.shape)
                  for name, spec in specs.items()}
        np.testing.assert_allclose(evaluate_plan(plan, arrays),
            evaluate_expression(contracts, views, coefficients, arrays),
            rtol=1e-11, atol=1e-9)
        broken = replace(viewed, reference=replace(viewed.reference, coefficient=2))
        with self.assertRaisesRegex(ValueError, 'differs.*target'):
            block_reduce(contracts, views, coefficients, tensor_specs=specs,
                         pairing_family=PairingFamily(broken, family.factor_positions))


if __name__ == "__main__":
    unittest.main()
