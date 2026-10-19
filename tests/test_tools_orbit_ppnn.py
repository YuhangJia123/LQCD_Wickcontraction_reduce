import unittest
from collections import Counter

from verify_code.orbit_ppnn import ppnn_orbit_system


class PpnnOrbitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system = ppnn_orbit_system()

    def test_counts_burnside_and_raw_histogram(self):
        system = self.system
        self.assertEqual(
            (system.raw_wick_count, system.state_count, system.orbit_count),
            (518400, 32400, 1068),
        )
        self.assertEqual(
            system.orbit_size_histogram,
            Counter({32: 967, 16: 83, 8: 14, 4: 4}),
        )
        self.assertEqual(
            system.raw_orbit_size_histogram,
            Counter({512: 967, 256: 83, 128: 14, 64: 4}),
        )
        self.assertEqual(
            system.state_fixed_point_histogram(),
            Counter({32400: 1, 576: 1, 180: 4, 96: 4, 24: 4, 0: 18}),
        )

    def test_fibers_and_full_raw_coverage(self):
        system = self.system
        self.assertEqual({len(raw_ids) for raw_ids in system.state_to_raw.values()}, {16})
        raw_ids = set(range(1, 518401))
        self.assertEqual(set(system.raw_to_state), raw_ids)
        self.assertEqual(
            {raw_id for orbit in system.raw_orbits for raw_id in orbit.members},
            raw_ids,
        )
        self.assertEqual(sum(record.size for record in system.state_orbits), 32400)
        system.validate()
if __name__ == "__main__":
    unittest.main()
