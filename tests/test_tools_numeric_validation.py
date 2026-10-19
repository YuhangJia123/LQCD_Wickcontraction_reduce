import unittest

from verify_code.numeric_validation import validate_seeds
from verify_code.orbit_ppn import ppn_orbit_system


class NumericValidationTests(unittest.TestCase):
    def test_three_seed_symmetry_compatible_check(self):
        records = validate_seeds(ppn_orbit_system(), (1, 2, 3))
        self.assertEqual(len(records), 3)
        self.assertTrue(all(item.raw_wick_count == 2880 for item in records))
        self.assertTrue(all(item.peram_shape == (3, 4, 3, 4) for item in records))
        self.assertTrue(all(item.vvv_shape == (3, 3, 3) for item in records))
        self.assertTrue(all(item.max_relative_residual < 1e-12 for item in records))

    def test_fewer_than_three_seeds_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_seeds(ppn_orbit_system(), (1, 2))


if __name__ == "__main__":
    unittest.main()
