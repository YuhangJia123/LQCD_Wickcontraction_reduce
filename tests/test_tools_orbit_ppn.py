import unittest
from collections import Counter

from verify_code.orbit_ppn import ppn_orbit_system


class PpnOrbitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system = ppn_orbit_system()

    def test_counts_burnside_and_raw_histogram(self):
        system = self.system
        self.assertEqual((system.raw_wick_count, system.state_count, system.orbit_count), (2880, 360, 93))
        self.assertEqual(system.orbit_size_histogram, Counter({4: 87, 2: 6}))
        self.assertEqual(system.raw_orbit_size_histogram, Counter({32: 87, 16: 6}))
        self.assertEqual(system.state_fixed_point_histogram(), Counter({360: 1, 8: 1, 4: 1, 0: 1}))

    def test_fibers_and_partition_coverage(self):
        system = self.system
        self.assertEqual({len(raw_ids) for raw_ids in system.state_to_raw.values()}, {8})
        self.assertEqual(set(system.raw_to_state), set(range(1, 2881)))
        self.assertEqual(
            {raw_id for orbit in system.raw_orbits for raw_id in orbit.members},
            set(range(1, 2881)),
        )
        self.assertEqual(sum(record.size for record in system.state_orbits), 360)
        system.validate()
        system.check_quotient_equivariance()
if __name__ == "__main__":
    unittest.main()
