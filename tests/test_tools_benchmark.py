import importlib.util
import unittest

from verify_code.benchmark_orbit_vs_reducer import (
    HISTORICAL_PPNN_REDUCER,
    _contract_sample,
    _production_measurement,
)


class OrbitBenchmarkTests(unittest.TestCase):
    def test_historical_ppnn_row_is_explicitly_unmeasured(self):
        row = _production_measurement("ppnn", "historical")
        self.assertEqual(row["production_basis_count"], 31092)
        self.assertFalse(row["measured_in_this_run"])
        self.assertEqual(row["source"], HISTORICAL_PPNN_REDUCER["source"])

    @unittest.skipUnless(
        importlib.util.find_spec("opt_einsum"),
        "install the optional numeric dependencies to run contraction samples",
    )
    def test_ppn_contract_sample_runs_real_calls(self):
        result = _contract_sample("ppn", seed=20261009, sample=2)
        self.assertEqual(result["sample_calls"], 2)
        self.assertGreater(result["contract_seconds"], 0.0)
        self.assertEqual(
            result["sample_output_shapes"],
            [(4, 4, 4, 4, 4, 4), (4, 4, 4, 4, 4, 4)],
        )


if __name__ == "__main__":
    unittest.main()
