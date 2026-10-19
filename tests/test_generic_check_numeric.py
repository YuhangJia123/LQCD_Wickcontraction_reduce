import unittest

from verify_code.generic_check_numeric import audit_one
from kernels.wick_reducer.operator_model import operator_library


class GenericCheckNumericTests(unittest.TestCase):
    def test_check_reconstructs_delta_at_three_and_six_colors(self):
        operators = operator_library()["delta"]
        for color_dimension in (3, 6):
            result = audit_one(
                "delta", operators, seed=20261011,
                color_dimension=color_dimension,
            )
            self.assertEqual(result["mode"], "check")
            self.assertEqual(result["raw_wick_count"], 6)
            self.assertLess(result["max_relative_residual"], 1.0e-10)


if __name__ == "__main__":
    unittest.main()

