"""Production normal-wick-block acceptance tests."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from kernels.block_reduction import load_symbolic_output
from verify_code.normal_block_validation import audit, generate_normal_fixture, tensor_environment


class ActualNormalBlockIntegrationTests(unittest.TestCase):
    def test_public_normal_wick_block_entry_discovers_ppn_48_networks(self):
        """The production writer builds the raw family automatically."""
        from kernels.wick_reducer import expression_output, operator_model

        operators = tuple(
            replace(operator, mode="normal")
            for operator in operator_model.operator_library()["ppn"]
        )
        with tempfile.TemporaryDirectory() as directory:
            result = expression_output.write_output(
                expression_output.split_operators(operators),
                Path(directory) / "ppn_block.py", name="ppn", mode="block",
            )
        self.assertEqual(result["block_network_count"], 48)
        self.assertTrue(result["block_plan"].stats["pairing_family_supplied"])
        self.assertTrue(result["block_plan"].stats["symbolic_verified"])

    def test_real_normal_delta_full_spin_plan_matches_generated_target(self):
        with tempfile.TemporaryDirectory() as directory:
            result = audit("delta", directory=Path(directory), seeds=(20261013,),
                           search_seconds=1, search_budget=40)
            source = Path(result["source_path"]).read_text(encoding="utf-8")
        self.assertIn("epsilon_color", source)
        self.assertIn("peram_u_sin_sou_x_x", source)
        self.assertNotIn("VVV_sink", source)
        self.assertEqual(result["basis_contract_count"], 2)
        self.assertEqual(result["plan_stats"]["w_type_count"], 1)
        self.assertTrue(result["plan_stats"]["symbolic_verified"])
        self.assertEqual(result["w_elements"], 12**4)
        sample = result["samples"][0]
        self.assertEqual(sample["output_shape"], [4, 4])
        self.assertLess(sample["source_vs_parsed"]["maximum_relative"], 1e-10)
        self.assertLess(sample["residual"]["maximum_relative"], 1e-10)
        self.assertTrue(result["production_import_verified"])

    def test_actual_normal_ppn_export_preserves_complete_cyclic_views(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "normal_ppn.py"
            generated = generate_normal_fixture("ppn", path)
            source = path.read_text(encoding="utf-8")
            specs, _arrays = tensor_environment(source)
            loaded = load_symbolic_output(source, tensor_specs=specs)
        self.assertEqual(generated["operator_mode"], "normal")
        self.assertEqual(len(loaded.basis_contract), 219)
        self.assertEqual(len(loaded.basis), 2760)
        self.assertEqual(len(loaded.coefficients), 2760)
        noninvolutive = sum(
            tuple(p[p[i]] for i in range(len(p))) != tuple(range(len(p)))
            for p in (view.permutation for view in loaded.basis.values())
        )
        self.assertEqual(noninvolutive, 1262)
        self.assertTrue(all(spec.mode == "normal" for spec in specs.values()))
        self.assertEqual(specs["peram_u_sin_sou_x_x"].shape, (3, 4, 3, 4))
        self.assertNotIn("VVV_sink", specs)


if __name__ == "__main__":
    unittest.main()
