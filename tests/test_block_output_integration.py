"""Project-facing normal block output integration."""

from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from kernels.block_reduction import evaluate_expression, load_symbolic_output
from verify_code.normal_block_validation import numerical_arrays, tensor_environment
from kernels.block_reduction.output_adapter import load_normal_output
from kernels.wick_reducer import expression_output, operator_model
from kernels.wick_reducer.reducer import reduce_operators


class BlockOutputIntegrationTests(unittest.TestCase):
    def _normal_delta(self):
        return tuple(
            replace(operator, mode="normal")
            for operator in operator_model.operator_library()["delta"]
        )

    def test_block_mode_rejects_distillation_operators(self):
        operators = operator_model.operator_library()["delta"]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "only for normal operators"):
                expression_output.write_output(
                    expression_output.split_operators(operators),
                    Path(directory) / "normal_wick_block.py",
                    name="delta", mode="block",
                )

    def test_block_output_executes_and_matches_independent_sum(self):
        operators = self._normal_delta()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            independent_path = root / "independent.py"
            block_path = root / "normal_wick_block.py"
            independent = expression_output.write_output(
                expression_output.split_operators(operators), independent_path,
                name="delta", mode="independent",
            )
            block = expression_output.write_output(
                expression_output.split_operators(operators), block_path,
                name="delta", mode="block",
                block_options={"search_seconds": 1, "search_budget": 10},
            )

            self.assertEqual(block["mode"], "block")
            self.assertGreaterEqual(block["block_network_count"], 1)
            source = independent_path.read_text(encoding="utf-8")
            specs, constants = tensor_environment(source)
            arrays = numerical_arrays(specs, constants, 20261014)
            adapted = load_normal_output(independent_path)
            expected = evaluate_expression(
                adapted.basis_contract, adapted.basis,
                adapted.coefficients, arrays,
            )
            namespace = dict(arrays)
            exec(compile(block_path.read_text(encoding="utf-8"),
                         str(block_path), "exec"), namespace)
            np.testing.assert_allclose(
                namespace["delta_result"], expected, rtol=1e-10, atol=1e-10,
            )

            text = block_path.read_text(encoding="utf-8")
            self.assertIn("Block definitions:", text)
            self.assertIn("delta_basis_contract", text)
            self.assertLess(text.index("Block definitions:"),
                            text.index("delta_basis_contract"))
            self.assertEqual(block["wick_count"], independent["wick_count"])
            self.assertEqual(block["independent_count"], independent["independent_count"])

    def test_block_output_round_trips_through_symbolic_loaders(self):
        """The block document keeps the independent three-table interface."""
        operators = self._normal_delta()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "normal_wick_block.py"
            expression_output.write_output(
                expression_output.split_operators(operators), path,
                name="delta", mode="block",
                block_options={"search_seconds": 1, "search_budget": 10},
            )
            adapted = load_normal_output(path)
            parsed = load_symbolic_output(
                path, tensor_specs=adapted.tensor_specs,
            )

        self.assertEqual(adapted.symbolic.prefix, "delta")
        self.assertEqual(parsed.prefix, "delta")
        self.assertEqual(parsed.basis_contract, adapted.basis_contract)
        self.assertEqual(parsed.basis, adapted.basis)
        self.assertEqual(parsed.coefficients, adapted.coefficients)

    def test_reduce_operators_block_flag_uses_same_project_entry(self):
        operators = self._normal_delta()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "normal_wick_block.py"
            result = reduce_operators(
                operators, label="delta", block=True, output_path=path,
                block_options={"search_seconds": 1, "search_budget": 10},
            )
        self.assertEqual(result["mode"], "block")
        self.assertTrue(path.name == "normal_wick_block.py")

    def test_reduce_operators_block_uses_default_path_when_omitted(self):
        normal = self._normal_delta()
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            os.chdir(directory)
            try:
                result = reduce_operators(
                    normal, label="delta", block=True,
                    block_options={"search_seconds": 1, "search_budget": 10},
                )
            finally:
                os.chdir(previous)
            self.assertEqual(result["mode"], "block")
            self.assertEqual(
                Path(result["output_path"]).as_posix(),
                "result/normal_wick_block/delta_normal_wick_block.py",
            )
            self.assertTrue(Path(directory, result["output_path"]).exists())

    def test_reduce_operators_block_requires_normal(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "only for normal"):
                reduce_operators(
                    operator_model.operator_library()["delta"], label="delta",
                    mode="block", output_path=Path(directory) / "out.py",
                )


if __name__ == "__main__":
    unittest.main()
