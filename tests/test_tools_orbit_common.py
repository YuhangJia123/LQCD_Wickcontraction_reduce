import unittest
from collections import Counter

from verify_code.orbit_common import (
    action_from_generators,
    burnside_count,
    enumerate_orbits,
    fixed_point_histogram,
)


class OrbitCommonTests(unittest.TestCase):
    def test_partition_and_burnside(self):
        swap = lambda value: 1 - value
        identity = action_from_generators((swap,), ())
        partition = enumerate_orbits((0, 1), (swap,))
        self.assertEqual(partition.count, 1)
        self.assertEqual(partition.size_histogram, Counter({2: 1}))
        self.assertEqual(fixed_point_histogram((0, 1), (identity, swap)), Counter({0: 1, 2: 1}))
        self.assertEqual(burnside_count((0, 1), (identity, swap)), 1)

    def test_generator_closure_rejects_out_of_domain_action(self):
        with self.assertRaises(ValueError):
            enumerate_orbits((0, 1), (lambda value: value + 2,))
if __name__ == "__main__":
    unittest.main()
