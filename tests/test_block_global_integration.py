"""End-to-end normal-wick-block acceptance for automatic global W layouts."""

from pathlib import Path
import tempfile
import unittest

from verify_code.audit_global_block import audit_global_ppn


class GlobalBlockIntegrationTests(unittest.TestCase):
    def test_two_48_layouts_and_full_six_axis_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            report = audit_global_ppn(directory=Path(directory),
                                      seeds=(20261013, 20261014),
                                      search_seconds=5.0)
        self.assertEqual(report["raw_wick_count"], 2880)
        self.assertEqual(report["basis_contract_count"], 219)
        self.assertEqual(report["basis_view_count"], 2760)
        self.assertTrue(report["exact_mapping"])
        self.assertEqual(report["fixed_output_ledger_size"], 2760)
        self.assertEqual(report["block_network_count"], 48)
        self.assertEqual(report["w_definition_count"], 1)
        stats = report["plan_stats"]
        self.assertTrue(stats["symbolic_verified"])
        self.assertTrue(stats["pairing_family_supplied"])
        self.assertGreaterEqual(stats["global_layouts_tested"], 2)
        layouts = stats["global_best_layouts"]
        self.assertEqual(len(layouts), 1)
        self.assertEqual([layout["network_count"] for layout in layouts], [48])
        self.assertEqual(report["alternative_layouts"], [])
        for sample in report["samples"]:
            self.assertEqual(sample["output_shape"], [4] * 6)
            self.assertEqual((sample["raw_canonicalized_calls"], sample["target_calls"],
                              sample["block_calls"]), (219, 219, 48))
            self.assertLess(sample["raw_vs_target"]["maximum_relative"], 1e-10)
            self.assertLess(sample["target_vs_block"]["maximum_relative"], 1e-10)
            self.assertLess(sample["raw_spotcheck_maximum_relative"], 1e-10)
            self.assertEqual(sample["alternative_layout_errors"], [])
        first, second = report["samples"]
        self.assertEqual(first["direct_raw_calls"], 2880)
        self.assertLess(first["direct_raw_vs_canonicalized"]["maximum_relative"], 1e-10)
        self.assertIsNone(second["direct_raw_calls"])


if __name__ == "__main__":
    unittest.main()
