"""Execution and static-reader regressions independent of Wick enumeration."""

from dataclasses import replace
from fractions import Fraction
import unittest
import weakref

import numpy as np

from kernels.block_reduction import (
    BasisView, BlockDefinition, BlockPlan, Factor, Network,
    ReconstructionTerm, TensorSpec, TensorSymmetry,
    evaluate_expression, evaluate_plan, load_symbolic_output,
)


def _w_plan(*, layout=(3, 4, 3, 4), ports=((0, 1), (2, 3)), identity=None):
    axis_types = tuple("spin" if i in (ports[0][1], ports[1][1]) else "color" for i in range(4))
    propagator = TensorSpec("S", layout, axis_types, kind="peram", identity=identity, peram_ports=ports)
    wedge = TensorSpec("W", (3, 4) * 4, ("color", "spin") * 4, kind="wedge")
    network = Network((Factor("W", tuple("abcdefgh")),), tuple("abcdefgh"))
    return BlockPlan(
        (BlockDefinition("W", "S", ports),), (network,),
        (ReconstructionTerm(0, 1, tuple(range(8))),),
        {"S": propagator, "W": wedge}, (), (), {"output_shape": (3, 4) * 4},
    )


class StaticExpressionReaderTests(unittest.TestCase):
    source = """
toy_coefficients = [1]
toy_basis_contract = {}
toy_basis_contract[7] = contract('abc->abc', X)
toy_basis = {}
toy_basis[0] = toy_basis_contract[7]
"""
    specs = {"X": TensorSpec("X", (4, 4, 4), ("spin",) * 3)}

    def test_hidden_writes_on_assignment_rhs_and_rebindings_are_rejected(self):
        mutations = (
            "ignored = toy_coefficients.append(2)",
            "ignored = toy_basis.update({0: -toy_basis_contract[7]})",
            "ignored = (toy_coefficients := [2])",
            "toy_basis[0][0] = 2",
            "unused, toy_coefficients = 0, [2]",
            "def toy_basis():\n    return {}",
            "from another_module import replacement as toy_coefficients",
            "del toy_basis[0]",
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):
                    load_symbolic_output(self.source + mutation + "\n", tensor_specs=self.specs)

    def test_mutable_tables_cannot_escape_through_aliases_or_unknown_consumers(self):
        escapes = (
            "alias = toy_coefficients\nalias[0] = 3",
            "holder = {'table': toy_coefficients}\nholder['table'][0] = 3",
            "writer = toy_coefficients.append\nwriter(2)",
            "mutate_target(toy_coefficients)",
            "np.copyto(toy_basis_contract[7], replacement)",
        )
        for escape in escapes:
            with self.subTest(escape=escape):
                with self.assertRaises(ValueError):
                    load_symbolic_output(self.source + escape + "\n", tensor_specs=self.specs)

    def test_nonexecuted_result_reader_and_nested_noninvolutive_transpose(self):
        source = self.source.replace(
            "toy_basis[0] = toy_basis_contract[7]",
            "toy_basis[0] = (-2 * toy_basis_contract[7]).transpose(1,2,0).transpose(2,1,0)",
        ) + """
from a_module_that_is_absent import contract
toy_result = sum(coefficient * toy_basis[i] for i, coefficient in enumerate(toy_coefficients))
raise RuntimeError('never execute this file')
"""
        loaded = load_symbolic_output(source, tensor_specs=self.specs)
        self.assertEqual(loaded.basis[0].permutation, (0, 2, 1))
        self.assertEqual(loaded.basis[0].scalar, -2)
        values = {"X": np.arange(64).reshape(4, 4, 4)}
        result = evaluate_expression(loaded.basis_contract, loaded.basis, loaded.coefficients, values)
        np.testing.assert_array_equal(result, -2 * values["X"].transpose(0, 2, 1))


class BlockNumericalExecutionTests(unittest.TestCase):
    def test_expression_releases_each_network_and_preserves_transpose_views(self):
        networks = {
            index: Network((Factor(f"X{index}", ("a", "b", "c")),), ("a", "b", "c"))
            for index in range(3)
        }
        basis = {
            "first": BasisView(0, scalar=-2, permutation=(1, 2, 0)),
            "second": BasisView(0, scalar=3, permutation=(2, 0, 1)),
            "third": BasisView(1, coefficient=Fraction(1, 2)),
            "fourth": BasisView(2, permutation=(1, 2, 0)),
        }
        source = {f"X{index}": np.arange(64).reshape(4, 4, 4) + index for index in networks}
        seen = []

        def counted(*args, **kwargs):
            from opt_einsum import contract
            if seen:
                self.assertIsNone(seen[-1]())
            value = np.array(contract(*args, **kwargs), copy=True)
            seen.append(weakref.ref(value))
            return value

        actual = evaluate_expression(networks, basis, None, source, counted)
        expected = (
            -2 * source["X0"].transpose(1, 2, 0)
            + 3 * source["X0"].transpose(2, 0, 1)
            + .5 * source["X1"]
            + source["X2"].transpose(1, 2, 0)
        )
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(len(seen), 3)
        self.assertTrue(all(reference() is None for reference in seen))

    def test_plan_releases_each_network_and_reuses_shape_only_path(self):
        from kernels.block_reduction.evaluate import _contraction_path

        network = Network((Factor("X", ("a", "b")), Factor("Y", ("b", "c"))), ("a", "c"))
        other = Network((Factor("Y", ("a", "b")), Factor("X", ("b", "c"))), ("a", "c"))
        specs = {name: TensorSpec(name, (4, 4), ("spin", "spin")) for name in ("X", "Y")}
        plan = BlockPlan(
            (), (network, other),
            (ReconstructionTerm(0, 2, (1, 0)), ReconstructionTerm(1, -1, (0, 1))),
            specs, (), (), {"output_shape": (4, 4)},
        )
        x = np.arange(16).reshape(4, 4)
        y = np.arange(16, 32).reshape(4, 4)
        _contraction_path.cache_clear()
        actual = evaluate_plan(plan, {"X": x, "Y": y})
        expected = 2 * (x @ y).T - y @ x
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(_contraction_path.cache_info().misses, 1)
        self.assertGreaterEqual(_contraction_path.cache_info().hits, 1)

        seen = []

        def counted(*args, **kwargs):
            from opt_einsum import contract
            if seen:
                self.assertIsNone(seen[-1]())
            value = np.array(contract(*args, **kwargs), copy=True)
            seen.append(weakref.ref(value))
            return value

        np.testing.assert_array_equal(evaluate_plan(plan, {"X": x, "Y": y}, counted), expected)
        self.assertTrue(all(reference() is None for reference in seen))
        self.assertEqual(len(seen), 2)

    def test_rank_eight_w_matches_port_exchange_and_both_antisymmetries(self):
        plan = _w_plan()
        rng = np.random.default_rng(20261014)
        s = rng.standard_normal((3, 4, 3, 4)) + 1j * rng.standard_normal((3, 4, 3, 4))
        expected = np.einsum("abij,cdkl->abcdijkl", s, s) - np.einsum("abkl,cdij->abcdijkl", s, s)
        result = evaluate_plan(plan, {"S": s})
        np.testing.assert_allclose(result, expected, rtol=1e-13, atol=1e-13)
        self.assertEqual(result.shape, (3, 4) * 4)
        self.assertGreater(np.linalg.norm(result), 1)
        np.testing.assert_allclose(result.transpose(2, 3, 0, 1, 4, 5, 6, 7), -result, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(result.transpose(0, 1, 2, 3, 6, 7, 4, 5), -result, rtol=1e-13, atol=1e-13)

    def test_alias_identity_only_array_and_noninvolutive_reconstruction(self):
        plan = _w_plan(identity="numeric_S")
        p = (2, 3, 4, 5, 0, 1, 6, 7)
        plan = replace(plan, reconstruction=(ReconstructionTerm(0, Fraction(-3, 2), p),))
        s = np.arange(144, dtype=float).reshape(3, 4, 3, 4)
        plain = _w_plan(identity="numeric_S")
        expected = -1.5 * evaluate_plan(plain, {"numeric_S": s}).transpose(p)
        result = evaluate_plan(plan, {"numeric_S": s})
        np.testing.assert_array_equal(result, expected)

    def test_incompatible_alias_port_layout_is_rejected(self):
        plan = _w_plan(identity="same")
        with self.assertRaisesRegex(ValueError, "peram_ports|color then spin"):
            replace(plan.tensor_specs["S"], name="Alias", peram_ports=((3, 2), (0, 1)))

    def test_incompatible_alias_symmetry_is_rejected(self):
        plan = _w_plan(identity="same")
        symmetry = TensorSymmetry((3, 2, 1, 0), 1)
        with self.assertRaisesRegex(ValueError, "symmetry"):
            replace(plan.tensor_specs["S"], name="Alias", symmetries=(symmetry,))

    def test_execution_rejects_non_normal_or_truncated_manual_plans(self):
        plan = _w_plan()
        invalid = (
            replace(plan.tensor_specs["S"], mode="distillation", shape=(3, 4, 3, 4),
                    axis_types=("color", "spin", "color", "spin"), peram_ports=((0, 1), (2, 3))),
            replace(plan.tensor_specs["S"], shape=(2, 3, 3, 2)),
            replace(plan.tensor_specs["S"], shape=(4, 2, 2, 4)),
        )
        for spec in invalid:
            with self.subTest(mode=spec.mode, shape=spec.shape):
                malformed = replace(plan, tensor_specs={**plan.tensor_specs, "S": spec})
                with self.assertRaises(ValueError):
                    evaluate_plan(malformed, {"S": np.ones(spec.shape)})

    def test_w_spec_axis_types_must_match_the_built_array(self):
        plan = _w_plan()
        spec = replace(plan.tensor_specs["W"], axis_types=("other",) * 8)
        with self.assertRaisesRegex(ValueError, "TensorSpec|definition"):
            evaluate_plan(replace(plan, tensor_specs={**plan.tensor_specs, "W": spec}), {"S": np.ones((3, 4, 3, 4))})

    def test_shared_w_and_final_network_are_evaluated_once(self):
        plan = _w_plan()
        p = (2, 3, 4, 5, 0, 1, 6, 7)
        plan = replace(plan, reconstruction=(ReconstructionTerm(0, 2, tuple(range(8))), ReconstructionTerm(0, -1, p)))
        calls = []

        def counted(*args, **kwargs):
            from opt_einsum import contract
            calls.append(args)
            return contract(*args, **kwargs)

        result = evaluate_plan(plan, {"S": np.arange(144).reshape(3, 4, 3, 4)}, counted)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.shape, (3, 4) * 4)

    def test_expression_none_or_mapping_coefficients_with_scalar_output(self):
        network = Network((Factor("scalar", ()),), ())
        contracts = {"constant": network}
        basis = {"view": BasisView("constant", scalar=-1, coefficient=Fraction(1, 3))}
        values = {"scalar": np.asarray(3 + 6j)}
        self.assertEqual(evaluate_expression(contracts, basis, None, values), -1 - 2j)
        self.assertEqual(evaluate_expression(contracts, basis, {"view": 2}, values), -2 - 4j)

    def test_empty_and_canceled_targets_preserve_zero_output_shape(self):
        empty = evaluate_expression({}, {}, None, {})
        self.assertEqual(empty.shape, ())
        self.assertEqual(empty, 0)
        network = Network((Factor("X", ("a", "b")),), ("a", "b"))
        values = {"X": np.arange(16).reshape(4, 4)}
        expression = evaluate_expression({1: network}, {"a": BasisView(1), "b": BasisView(1, scalar=-1)}, None, values)
        np.testing.assert_array_equal(expression, np.zeros((4, 4)))
        plan = BlockPlan((), (), (), {"X": TensorSpec("X", (4, 4), ("spin",) * 2)}, (), (), {"output_shape": (4, 4)})
        np.testing.assert_array_equal(evaluate_plan(plan, {}), np.zeros((4, 4)))


if __name__ == "__main__":
    unittest.main()
